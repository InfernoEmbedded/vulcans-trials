// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use crate::board::{Board, Point, TrackSegment};
use crate::placer::{
    build_component_clusters, build_passive_ties, legalize_placement,
    optimize_pin_and_gate_swapping, run_auto_placer, PlacementConfig, PlacementState,
};
use crate::router::{
    add_teardrops, apply_any_angle_tangent_arcs, dist_point_to_pad, dist_point_to_segment,
    dist_segment_to_pad, dist_segment_to_segment, generate_bga_dogbone_escapes,
    gloss_tracks_with_mode, match_diff_pair_phase, prune_dangling_tracks_and_vias,
    reduce_vias, consolidate_single_vias, slide_and_align_vias,
    relax_homotopic_rubber_bands, route_bundle_along_guide, tune_net_length,
    validate_and_clean_copper, validate_crosstalk_and_coupling, AutoRouter, BusBundle,
    NudgeEngine, RoutingGrid,
};
use crate::vulkan::{SystemTelemetry, VulkanEngine};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

/// JSON-RPC 2.0 Request representation
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

/// JSON-RPC 2.0 Response representation
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 Error representation
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

/// Shared state for Vulcan's Trials MCP server
pub struct McpServerState {
    pub board: Option<Board>,
    pub vulkan: Option<Arc<VulkanEngine>>,
    pub telemetry: SystemTelemetry,
}

/// Thread-safe Model Context Protocol server instance
pub struct McpServer {
    pub state: Arc<Mutex<McpServerState>>,
}

impl McpServer {
    pub fn new(board: Option<Board>, vulkan: Option<Arc<VulkanEngine>>) -> Self {
        let compute_counter = vulkan.as_ref().map(|vk| Arc::clone(&vk.compute_busy_ns));
        let telemetry = SystemTelemetry::spawn(compute_counter);

        Self {
            state: Arc::new(Mutex::new(McpServerState {
                board,
                vulkan,
                telemetry,
            })),
        }
    }

    /// Process a raw JSON-RPC string message and produce a response string if one is required.
    pub fn handle_message(&self, message_str: &str) -> Option<String> {
        let trimmed = message_str.trim();
        if trimmed.is_empty() {
            return None;
        }

        let request: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(req) => req,
            Err(e) => {
                let err_resp = JsonRpcResponse::error(None, -32700, format!("Parse error: {}", e));
                return Some(serde_json::to_string(&err_resp).unwrap());
            }
        };

        let is_notification = request.id.is_none();
        let resp = self.dispatch_request(request);

        if is_notification {
            None
        } else {
            Some(serde_json::to_string(&resp).unwrap())
        }
    }

    fn dispatch_request(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let id = req.id.clone();
        match req.method.as_str() {
            "initialize" => {
                let result = json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {
                            "listChanged": false
                        },
                        "resources": {
                            "subscribe": false,
                            "listChanged": false
                        }
                    },
                    "serverInfo": {
                        "name": "vulcans-trials",
                        "version": "0.1.0"
                    }
                });
                JsonRpcResponse::success(id, result)
            }

            "notifications/initialized" => {
                // Client initialized notification: no response required
                JsonRpcResponse::success(id, json!({}))
            }

            "ping" => JsonRpcResponse::success(id, json!({})),

            "tools/list" => {
                let tools = get_all_tool_definitions();
                JsonRpcResponse::success(id, json!({ "tools": tools }))
            }

            "tools/call" => {
                let params = req.params.unwrap_or_else(|| json!({}));
                let name = match params.get("name").and_then(|v| v.as_str()) {
                    Some(n) => n,
                    None => {
                        return JsonRpcResponse::error(id, -32602, "Missing 'name' field in tools/call parameters");
                    }
                };

                let empty_args = json!({});
                let args = params.get("arguments").unwrap_or(&empty_args);

                match self.execute_tool(name, args) {
                    Ok(val) => {
                        let text_content = if val.is_string() {
                            val.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string_pretty(&val).unwrap_or_else(|_| val.to_string())
                        };

                        let result = json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": text_content
                                }
                            ],
                            "isError": false
                        });
                        JsonRpcResponse::success(id, result)
                    }
                    Err(err_msg) => {
                        let result = json!({
                            "content": [
                                {
                                    "type": "text",
                                    "text": format!("Error executing tool '{}': {}", name, err_msg)
                                }
                            ],
                            "isError": true
                        });
                        JsonRpcResponse::success(id, result)
                    }
                }
            }

            "resources/list" => {
                let resources = json!([
                    {
                        "uri": "kicad://board/current",
                        "name": "Current Board Layout",
                        "description": "Active KiCad board geometry, footprint count, and routing statistics",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "kicad://system/telemetry",
                        "name": "Engine Telemetry",
                        "description": "Real-time CPU and GPU utilization, distance field evaluations, and ratsnest status",
                        "mimeType": "application/json"
                    }
                ]);
                JsonRpcResponse::success(id, json!({ "resources": resources }))
            }

            "resources/read" => {
                let params = req.params.unwrap_or_else(|| json!({}));
                let uri = match params.get("uri").and_then(|v| v.as_str()) {
                    Some(u) => u,
                    None => {
                        return JsonRpcResponse::error(id, -32602, "Missing 'uri' parameter in resources/read");
                    }
                };

                match uri {
                    "kicad://board/current" => {
                        let guard = self.state.lock().unwrap();
                        let info = match guard.board {
                            Some(ref b) => collect_board_info(b),
                            None => json!({ "status": "no_board_loaded" }),
                        };
                        let contents = json!([
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": serde_json::to_string_pretty(&info).unwrap()
                            }
                        ]);
                        JsonRpcResponse::success(id, json!({ "contents": contents }))
                    }
                    "kicad://system/telemetry" => {
                        let guard = self.state.lock().unwrap();
                        let telem = collect_telemetry_info(&guard);
                        let contents = json!([
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": serde_json::to_string_pretty(&telem).unwrap()
                            }
                        ]);
                        JsonRpcResponse::success(id, json!({ "contents": contents }))
                    }
                    _ => JsonRpcResponse::error(id, -32602, format!("Resource not found: {}", uri)),
                }
            }

            unknown => JsonRpcResponse::error(id, -32601, format!("Method not found: {}", unknown)),
        }
    }

    /// Dispatch tool calls to engine functions
    fn execute_tool(&self, name: &str, args: &Value) -> Result<Value, String> {
        match name {
            // ==================== 1. Board Management ====================
            "load_board" => {
                let path_str = args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or("Parameter 'path' (string) is required")?;

                let mut extra_locked = HashSet::new();
                if let Some(arr) = args.get("extra_locked").and_then(|v| v.as_array()) {
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            extra_locked.insert(s.to_string());
                        }
                    }
                }

                let loaded_board = Board::load(path_str, &extra_locked)?;
                let mut guard = self.state.lock().unwrap();
                let info = json!({
                    "status": "loaded",
                    "path": loaded_board.file_path,
                    "dimensions_mm": {
                        "width": loaded_board.max_x - loaded_board.min_x,
                        "height": loaded_board.max_y - loaded_board.min_y,
                    },
                    "footprint_count": loaded_board.footprints.len(),
                    "locked_footprints": loaded_board.footprints.iter().filter(|f| f.is_locked).count(),
                    "net_count": loaded_board.nets.len(),
                    "cluster_count": loaded_board.clusters.len(),
                    "active_layers": loaded_board.copper_layers(),
                });
                guard.board = Some(loaded_board);
                Ok(info)
            }

            "save_board" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;
                let out_path = if let Some(p) = args.get("path").and_then(|v| v.as_str()) {
                    p.to_string()
                } else if !board.file_path.is_empty() {
                    board.file_path.clone()
                } else {
                    "vulcans_trials_out.kicad_pcb".to_string()
                };

                board.save_pcb(Path::new(&out_path))?;
                Ok(json!({
                    "status": "saved",
                    "path": out_path,
                    "track_count": board.tracks.len(),
                    "via_count": board.vias.len(),
                    "arc_count": board.arcs.len(),
                }))
            }

            "export_specctra_ses" => {
                let guard = self.state.lock().unwrap();
                let board = guard.board.as_ref().ok_or("No board currently loaded")?;
                let ses_path = if let Some(p) = args.get("path").and_then(|v| v.as_str()) {
                    Path::new(p).to_path_buf()
                } else if !board.file_path.is_empty() {
                    Path::new(&board.file_path).with_extension("ses")
                } else {
                    Path::new("vulcans_trials_out.ses").to_path_buf()
                };

                board.save_ses(&ses_path)?;
                Ok(json!({
                    "status": "exported",
                    "path": ses_path.display().to_string(),
                    "tracks_exported": board.tracks.len(),
                    "vias_exported": board.vias.len(),
                }))
            }

            "get_board_info" => {
                let guard = self.state.lock().unwrap();
                match guard.board {
                    Some(ref b) => Ok(collect_board_info(b)),
                    None => Ok(json!({ "loaded": false, "message": "No board loaded" })),
                }
            }

            // ==================== 2. Placement ====================
            "run_autoplacer" => {
                let mut guard = self.state.lock().unwrap();
                let vulkan = guard.vulkan.clone();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;

                let duration = args.get("duration_secs").and_then(|v| v.as_u64()).unwrap_or(10);
                let temp_scale = args.get("cooling_rate").or_else(|| args.get("temp_scale")).and_then(|v| v.as_f64()).unwrap_or(1.0);
                let crossing_weight = args.get("crossing_weight").and_then(|v| v.as_f64()).unwrap_or(25.0);
                let wirelength_weight = args.get("wirelength_weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
                let overlap_penalty = args.get("overlap_penalty").and_then(|v| v.as_f64()).unwrap_or(2500.0);
                let passive_affinity = args.get("passive_affinity_weight").and_then(|v| v.as_f64()).unwrap_or(35.0);
                let decoupling_affinity = args.get("decoupling_affinity_weight").and_then(|v| v.as_f64()).unwrap_or(60.0);
                let allow_layer_flip = args.get("allow_layer_flip").and_then(|v| v.as_bool()).unwrap_or(false);
                let escape_clearance = args.get("enable_escape_clearance").and_then(|v| v.as_bool()).unwrap_or(true);

                let cfg = PlacementConfig {
                    duration_secs: duration,
                    temp_scale,
                    crossing_weight,
                    wirelength_weight,
                    overlap_penalty,
                    passive_affinity_weight: passive_affinity,
                    decoupling_affinity_weight: decoupling_affinity,
                    allow_layer_flip,
                    enable_escape_clearance: escape_clearance,
                    vulkan,
                    ..PlacementConfig::default()
                };

                let stop_flag = Arc::new(AtomicBool::new(false));
                let start_t = Instant::now();
                let st = run_auto_placer(board, &cfg, None, stop_flag);
                let elapsed = start_t.elapsed().as_secs_f64();

                Ok(json!({
                    "status": "completed",
                    "crossings": st.crossings,
                    "wirelength_mm": st.wirelength,
                    "overlaps": st.overlaps,
                    "boundary_violations": st.boundary_violations,
                    "elapsed_secs": elapsed,
                }))
            }

            "legalize_placement" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;
                let min_spacing = args.get("min_spacing_mm").and_then(|v| v.as_f64()).unwrap_or(0.25);

                let cfg = PlacementConfig {
                    min_spacing_mm: min_spacing,
                    ..PlacementConfig::default()
                };

                let initial = PlacementState::from_board(board, &cfg);
                let overlaps_before = initial.overlaps;
                let legalized = legalize_placement(board, &cfg, initial, None);
                let overlaps_after = legalized.overlaps;
                let b_violations = legalized.boundary_violations;
                legalized.apply_to_board(board);

                Ok(json!({
                    "status": "legalized",
                    "overlaps_before": overlaps_before,
                    "overlaps_remaining": overlaps_after,
                    "boundary_violations_remaining": b_violations,
                    "is_legal": overlaps_after == 0 && b_violations == 0,
                }))
            }

            "pin_and_gate_swap" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;
                let cfg = PlacementConfig::default();
                let swaps_count = optimize_pin_and_gate_swapping(board, &cfg);

                let history_json: Vec<Value> = board
                    .pin_swap_history
                    .iter()
                    .map(|h| {
                        json!({
                            "reference": h.footprint_ref,
                            "pin_a": h.pad1_number,
                            "pin_b": h.pad2_number,
                            "net_a": h.pad1_net,
                            "net_b": h.pad2_net,
                            "reason": h.reason,
                        })
                    })
                    .collect();

                Ok(json!({
                    "swaps_performed": swaps_count,
                    "history": history_json,
                }))
            }

            "cluster_components" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;

                if let Some(sch) = args.get("schematic_path").and_then(|v| v.as_str()) {
                    let sch_path = Path::new(sch);
                    if sch_path.exists() {
                        board.load_schematic_clusters(sch_path);
                    }
                }

                let ties = build_passive_ties(board);
                let movable: Vec<usize> = board
                    .footprints
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| !f.is_locked)
                    .map(|(i, _)| i)
                    .collect();
                let passive_clusters = build_component_clusters(board, &ties, &movable);

                let mut cluster_list = Vec::new();
                for c in &board.clusters {
                    let anchor_ref = c.anchor_idx.map(|idx| board.footprints[idx].reference.clone());
                    cluster_list.push(json!({
                        "name": c.name,
                        "anchor": anchor_ref,
                        "members": c.member_refs,
                        "member_count": c.member_indices.len(),
                    }));
                }

                Ok(json!({
                    "schematic_clusters_count": board.clusters.len(),
                    "schematic_clusters": cluster_list,
                    "passive_ties_count": ties.len(),
                    "component_clusters_count": passive_clusters.len(),
                }))
            }

            // ==================== 3. Routing ====================
            "route_all" => {
                let mut guard = self.state.lock().unwrap();
                let vulkan = guard.vulkan.clone();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;

                let passes = args.get("passes").and_then(|v| v.as_u64()).unwrap_or(15) as usize;
                let res = args.get("res").and_then(|v| v.as_f64()).unwrap_or(0.10);
                let inner_pen = args.get("inner_plane_penalty").and_then(|v| v.as_f64()).unwrap_or(8.0);
                let any_angle = args.get("any_angle").and_then(|v| v.as_bool()).unwrap_or(false);
                let crosstalk_pen = args.get("crosstalk_penalty").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let allow_nudge = args.get("allow_nudge").and_then(|v| v.as_bool()).unwrap_or(true);

                let mut grid = RoutingGrid::new(board, res, vulkan).with_inner_layer_penalty(inner_pen);
                let router = AutoRouter::new(passes)
                    .with_inner_layer_penalty(inner_pen)
                    .with_any_angle_mode(any_angle)
                    .with_crosstalk_penalty(crosstalk_pen)
                    .with_allow_component_nudge(allow_nudge);

                let stop_flag = Arc::new(AtomicBool::new(false));
                let start_t = Instant::now();
                let (routed, total) = router.route_all(board, &mut grid, None, stop_flag);
                let elapsed = start_t.elapsed().as_secs_f64();

                validate_and_clean_copper(board);
                prune_dangling_tracks_and_vias(board);

                Ok(json!({
                    "status": "routed",
                    "routed_connections": routed,
                    "total_connections": total,
                    "completion_pct": if total > 0 { (routed as f64 / total as f64) * 100.0 } else { 100.0 },
                    "track_count": board.tracks.len(),
                    "via_count": board.vias.len(),
                    "elapsed_secs": elapsed,
                }))
            }

            "route_bus_bundle" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;

                let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("bus_bundle");
                let net_items = args
                    .get("nets")
                    .and_then(|v| v.as_array())
                    .ok_or("Parameter 'nets' (array of net names or IDs) is required")?;

                let mut net_ids = Vec::new();
                for item in net_items {
                    if let Some(s) = item.as_str() {
                        if let Some(&nid) = board.net_name_to_id.get(s) {
                            net_ids.push(nid);
                        } else {
                            return Err(format!("Net '{}' not found on board", s));
                        }
                    } else if let Some(n) = item.as_u64() {
                        net_ids.push(n as usize);
                    }
                }

                let pts_arr = args
                    .get("guide_points")
                    .and_then(|v| v.as_array())
                    .ok_or("Parameter 'guide_points' (array of [x, y] coordinates) is required")?;

                let mut guide_points = Vec::new();
                for p in pts_arr {
                    if let Some(coords) = p.as_array() {
                        if coords.len() >= 2 {
                            let x = coords[0].as_f64().ok_or("Invalid guide coordinate x")?;
                            let y = coords[1].as_f64().ok_or("Invalid guide coordinate y")?;
                            guide_points.push((x, y));
                        }
                    }
                }

                if guide_points.len() < 2 {
                    return Err("Guide points must contain at least 2 coordinate pairs".to_string());
                }

                let layer = args.get("layer").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let track_width = args.get("track_width").and_then(|v| v.as_f64()).unwrap_or(0.20);
                let track_spacing = args.get("track_spacing").and_then(|v| v.as_f64()).unwrap_or(0.20);

                let bundle = BusBundle::new(name, net_ids, guide_points, layer, track_width, track_spacing);
                let mut grid = RoutingGrid::new(board, 0.10, None);
                let tracks_before = board.tracks.len();
                let success = route_bundle_along_guide(board, &mut grid, &bundle);
                let tracks_added = board.tracks.len().saturating_sub(tracks_before);

                Ok(json!({
                    "success": success,
                    "bundle_name": name,
                    "tracks_added": tracks_added,
                    "nets_count": bundle.net_ids.len(),
                }))
            }

            "escape_bga" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;
                let grid_res = args.get("grid_res").and_then(|v| v.as_f64()).unwrap_or(0.10);

                let vias_before = board.vias.len();
                let tracks_before = board.tracks.len();
                let mut grid = RoutingGrid::new(board, grid_res, None);
                generate_bga_dogbone_escapes(board, &mut grid);

                Ok(json!({
                    "vias_added": board.vias.len().saturating_sub(vias_before),
                    "tracks_added": board.tracks.len().saturating_sub(tracks_before),
                    "total_vias": board.vias.len(),
                }))
            }

            "escape_dense_ics" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;
                let router = AutoRouter::new(1);
                let stubs = router.generate_escape_stubs(board);

                let mut tracks_added = 0;
                let mut vias_added = 0;
                for stub in &stubs {
                    board.tracks.push(TrackSegment {
                        start: stub.pad_pos,
                        end: stub.tip_pos,
                        width: stub.width,
                        layer: stub.layer,
                        net_id: stub.net_id,
                        is_locked: false,
                    });
                    tracks_added += 1;
                    if let Some(ref v) = stub.via {
                        board.vias.push(v.clone());
                        vias_added += 1;
                    }
                }

                Ok(json!({
                    "stubs_generated": stubs.len(),
                    "tracks_added": tracks_added,
                    "vias_added": vias_added,
                }))
            }

            "topor_any_angle" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;

                let arcs_before = board.arcs.len();
                relax_homotopic_rubber_bands(board);
                apply_any_angle_tangent_arcs(board);
                let arcs_after = board.arcs.len();

                let arc_details: Vec<Value> = board
                    .arcs
                    .iter()
                    .map(|a| {
                        json!({
                            "center": [a.center.0, a.center.1],
                            "radius": a.radius,
                            "sweep_angle_deg": a.sweep_angle(),
                            "layer": a.layer,
                            "net_id": a.net_id,
                        })
                    })
                    .collect();

                Ok(json!({
                    "arcs_created": arcs_after.saturating_sub(arcs_before),
                    "total_arcs": arcs_after,
                    "arc_details": arc_details,
                }))
            }

            "tune_length" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;
                let mut grid = RoutingGrid::new(board, 0.10, None);

                let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("diff_pair");
                if mode == "diff_pair" {
                    let net_p_str = args.get("net_p").and_then(|v| v.as_str());
                    let net_n_str = args.get("net_n").and_then(|v| v.as_str());
                    let (net_p, net_n) = match (net_p_str, net_n_str) {
                        (Some(p), Some(n)) => {
                            let pid = *board.net_name_to_id.get(p).ok_or(format!("Net '{}' not found", p))?;
                            let nid = *board.net_name_to_id.get(n).ok_or(format!("Net '{}' not found", n))?;
                            (pid, nid)
                        }
                        _ => {
                            // Find first diff pair on the board
                            let mut found = None;
                            for (name, &id) in &board.net_name_to_id {
                                if name.ends_with("_P") || name.ends_with('+') {
                                    let mate_suffix = if name.ends_with("_P") { "_N" } else { "-" };
                                    let base = &name[..name.len() - 2];
                                    let mate_name = format!("{}{}", base, mate_suffix);
                                    if let Some(&nid) = board.net_name_to_id.get(&mate_name) {
                                        found = Some((id, nid));
                                        break;
                                    }
                                }
                            }
                            found.ok_or("No differential pair specified and none detected automatically")?
                        }
                    };

                    let tolerance = args.get("tolerance_mm").and_then(|v| v.as_f64()).unwrap_or(0.10);
                    let success = match_diff_pair_phase(board, &mut grid, net_p, net_n, tolerance);

                    let calc_len = |nid: usize| -> f64 {
                        board
                            .tracks
                            .iter()
                            .filter(|t| t.net_id == nid)
                            .map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1))
                            .sum()
                    };
                    let len_p = calc_len(net_p);
                    let len_n = calc_len(net_n);

                    Ok(json!({
                        "mode": "diff_pair",
                        "success": success,
                        "net_p_length_mm": len_p,
                        "net_n_length_mm": len_n,
                        "skew_mm": (len_p - len_n).abs(),
                        "tolerance_mm": tolerance,
                    }))
                } else {
                    let net_id = if let Some(s) = args.get("net_name").and_then(|v| v.as_str()) {
                        *board.net_name_to_id.get(s).ok_or(format!("Net '{}' not found", s))?
                    } else if let Some(id) = args.get("net_id").and_then(|v| v.as_u64()) {
                        id as usize
                    } else {
                        return Err("Parameter 'net_name' or 'net_id' is required for mode 'net'".to_string());
                    };

                    let target_len = args
                        .get("target_length_mm")
                        .and_then(|v| v.as_f64())
                        .ok_or("Parameter 'target_length_mm' is required")?;
                    let max_amp = args.get("max_amplitude_mm").and_then(|v| v.as_f64()).unwrap_or(1.0);

                    let success = tune_net_length(board, &mut grid, net_id, target_len, max_amp);
                    let actual_len: f64 = board
                        .tracks
                        .iter()
                        .filter(|t| t.net_id == net_id)
                        .map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1))
                        .sum();

                    Ok(json!({
                        "mode": "net",
                        "success": success,
                        "net_id": net_id,
                        "actual_length_mm": actual_len,
                        "target_length_mm": target_len,
                    }))
                }
            }

            "stitch_ground_vias" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;
                let grid_res = args.get("grid_res").and_then(|v| v.as_f64()).unwrap_or(0.10);
                let grid = RoutingGrid::new(board, grid_res, None);
                let router = AutoRouter::new(1);

                let thermal_enabled = args.get("thermal_pads").and_then(|v| v.as_bool()).unwrap_or(true);
                let return_enabled = args.get("return_paths").and_then(|v| v.as_bool()).unwrap_or(true);

                let thermal_vias = if thermal_enabled {
                    router.add_ground_stitching_vias(board, &grid)
                } else {
                    0
                };

                let return_path_vias = if return_enabled {
                    router.add_return_path_ground_vias(board, &grid)
                } else {
                    0
                };

                Ok(json!({
                    "thermal_stitching_vias_added": thermal_vias,
                    "return_path_vias_added": return_path_vias,
                    "total_vias_added": thermal_vias + return_path_vias,
                    "total_board_vias": board.vias.len(),
                }))
            }

            "add_teardrops" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;
                let force_enable = args.get("force_enable").and_then(|v| v.as_bool()).unwrap_or(true);

                if force_enable {
                    for nc in board.netclasses.values_mut() {
                        nc.teardrop_enabled = true;
                    }
                }

                let tracks_before = board.tracks.len();
                add_teardrops(board);
                let added = board.tracks.len().saturating_sub(tracks_before);

                Ok(json!({
                    "teardrops_added": added,
                    "total_tracks": board.tracks.len(),
                }))
            }

            "gloss_tracks" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;
                let any_angle = args.get("any_angle").and_then(|v| v.as_bool()).unwrap_or(false);
                let grid_res = args.get("grid_res").and_then(|v| v.as_f64()).unwrap_or(0.10);

                let grid = RoutingGrid::new(board, grid_res, None);
                let tracks_before = board.tracks.len();
                gloss_tracks_with_mode(board, &grid, any_angle);
                let tracks_after = board.tracks.len();

                Ok(json!({
                    "tracks_before": tracks_before,
                    "tracks_after": tracks_after,
                    "difference": (tracks_before as isize) - (tracks_after as isize),
                }))
            }

            "optimize_vias" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;
                let grid_res = args.get("grid_res").and_then(|v| v.as_f64()).unwrap_or(0.10);

                let vias_before = board.vias.len();
                let mut grid = RoutingGrid::new(board, grid_res, None);

                reduce_vias(board, &mut grid);
                consolidate_single_vias(board, &grid);
                slide_and_align_vias(board, &grid);
                let vias_after = board.vias.len();

                Ok(json!({
                    "vias_before": vias_before,
                    "vias_after": vias_after,
                    "vias_eliminated": vias_before.saturating_sub(vias_after),
                }))
            }

            "shove_tracks" => {
                let mut guard = self.state.lock().unwrap();
                let board = guard.board.as_mut().ok_or("No board currently loaded")?;
                let nudge = NudgeEngine::new();

                if let Some(track_idx) = args.get("track_index").and_then(|v| v.as_u64()) {
                    let idx = track_idx as usize;
                    let offset = args.get("offset_mm").and_then(|v| v.as_f64()).unwrap_or(0.25);
                    let success = nudge.try_continuous_shove_deflect(board, idx, offset, None);
                    Ok(json!({
                        "mode": "deflect",
                        "track_index": idx,
                        "offset_mm": offset,
                        "success": success,
                    }))
                } else if let (Some(p1_arr), Some(p2_arr)) = (
                    args.get("intruder_p1").and_then(|v| v.as_array()),
                    args.get("intruder_p2").and_then(|v| v.as_array()),
                ) {
                    if p1_arr.len() < 2 || p2_arr.len() < 2 {
                        return Err("intruder_p1 and intruder_p2 must be [x, y] coordinates".to_string());
                    }
                    let p1 = (p1_arr[0].as_f64().unwrap(), p1_arr[1].as_f64().unwrap());
                    let p2 = (p2_arr[0].as_f64().unwrap(), p2_arr[1].as_f64().unwrap());
                    let width = args.get("intruder_width").and_then(|v| v.as_f64()).unwrap_or(0.20);
                    let net_id = if let Some(n) = args.get("intruder_net").and_then(|v| v.as_str()) {
                        board.net_name_to_id.get(n).copied().unwrap_or(1)
                    } else {
                        args.get("intruder_net_id").and_then(|v| v.as_u64()).unwrap_or(1) as usize
                    };

                    // Find first conflicting track
                    let mut shoved = false;
                    let mut shoved_idx = None;
                    for i in 0..board.tracks.len() {
                        if board.tracks[i].net_id != net_id && !board.tracks[i].is_locked {
                            let d = dist_segment_to_segment(p1, p2, board.tracks[i].start, board.tracks[i].end);
                            if d < (width + board.tracks[i].width) * 0.5 + 0.20 {
                                if nudge.try_continuous_shove_track(board, i, p1, p2, width, net_id, None) {
                                    shoved = true;
                                    shoved_idx = Some(i);
                                    break;
                                }
                            }
                        }
                    }

                    Ok(json!({
                        "mode": "intruder_shove",
                        "success": shoved,
                        "shoved_track_index": shoved_idx,
                    }))
                } else {
                    Err("Must provide either 'track_index' or 'intruder_p1' + 'intruder_p2'".to_string())
                }
            }

            // ==================== 4. DRC & Analysis ====================
            "run_drc" => {
                let guard = self.state.lock().unwrap();
                let board = guard.board.as_ref().ok_or("No board currently loaded")?;
                let custom_edge_clr = args.get("board_edge_clearance").and_then(|v| v.as_f64());
                let report = evaluate_drc(board, custom_edge_clr);
                Ok(serde_json::to_value(report).unwrap())
            }

            "check_signal_integrity" => {
                let guard = self.state.lock().unwrap();
                let board = guard.board.as_ref().ok_or("No board currently loaded")?;

                let crosstalk_violations = validate_crosstalk_and_coupling(board);
                let mut si_report = json!({
                    "crosstalk_violations_count": crosstalk_violations.len(),
                    "crosstalk_violations": crosstalk_violations.iter().map(|v| {
                        json!({
                            "net_a": v.net_a_name,
                            "net_b": v.net_b_name,
                            "coupled_length_mm": v.coupled_length_mm,
                            "span_start": [v.span_start.0, v.span_start.1],
                            "span_end": [v.span_end.0, v.span_end.1],
                        })
                    }).collect::<Vec<Value>>(),
                });

                // Return-path plane continuity check
                let mut return_path_warnings = Vec::new();
                let gnd_net_id = board.nets.iter().position(|n| n == "GND").unwrap_or(0);
                for v in &board.vias {
                    if v.net_id != gnd_net_id && v.net_id != 0 {
                        let nc = board.get_netclass_for_net(v.net_id);
                        if nc.require_return_path {
                            let has_gnd_via = board.vias.iter().any(|gv| {
                                gv.net_id == gnd_net_id
                                    && (gv.position.0 - v.position.0).hypot(gv.position.1 - v.position.1) <= 1.50
                            });
                            if !has_gnd_via {
                                return_path_warnings.push(json!({
                                    "type": "unshielded_layer_transition",
                                    "net_id": v.net_id,
                                    "net_name": board.nets.get(v.net_id).cloned().unwrap_or_default(),
                                    "via_position": [v.position.0, v.position.1],
                                    "message": "High-speed signal transition via lacks adjacent return-path ground via within 1.50mm",
                                }));
                            }
                        }
                    }
                }

                si_report["return_path_warnings_count"] = json!(return_path_warnings.len());
                si_report["return_path_warnings"] = json!(return_path_warnings);
                si_report["is_si_clean"] = json!(crosstalk_violations.is_empty() && return_path_warnings.is_empty());

                Ok(si_report)
            }

            "get_telemetry" => {
                let guard = self.state.lock().unwrap();
                Ok(collect_telemetry_info(&guard))
            }

            _ => Err(format!("Unknown tool: {}", name)),
        }
    }
}

/// Helper function to evaluate comprehensive physical DRC
fn evaluate_drc(board: &Board, custom_edge_clr: Option<f64>) -> Value {
    let mut shorts = Vec::new();
    let mut clearance_violations = Vec::new();
    let mut hole_to_hole_violations = Vec::new();
    let mut board_edge_violations = Vec::new();

    let track_count = board.tracks.len();
    let via_count = board.vias.len();

    // 1. Track-to-Track DRC
    for i in 0..track_count {
        let t1 = &board.tracks[i];
        if t1.net_id == 0 {
            continue;
        }
        let nc1 = board.get_netclass_for_net(t1.net_id);

        for j in (i + 1)..track_count {
            let t2 = &board.tracks[j];
            if t1.net_id == t2.net_id || t1.layer != t2.layer || t2.net_id == 0 {
                continue;
            }

            let nc2 = board.get_netclass_for_net(t2.net_id);
            let req_clr = nc1.clearance.max(nc2.clearance);
            let d = dist_segment_to_segment(t1.start, t1.end, t2.start, t2.end);
            let touch_dist = (t1.width + t2.width) * 0.5;

            if d < touch_dist - 1e-4 {
                shorts.push(json!({
                    "layer": t1.layer,
                    "net_a": board.nets.get(t1.net_id).cloned().unwrap_or_default(),
                    "net_b": board.nets.get(t2.net_id).cloned().unwrap_or_default(),
                    "entity_a": format!("Track #{} [({},{})->({},{})]", i, t1.start.0, t1.start.1, t1.end.0, t1.end.1),
                    "entity_b": format!("Track #{} [({},{})->({},{})]", j, t2.start.0, t2.start.1, t2.end.0, t2.end.1),
                    "distance_mm": d,
                }));
            } else if d < touch_dist + req_clr - 1e-4 {
                clearance_violations.push(json!({
                    "layer": t1.layer,
                    "net_a": board.nets.get(t1.net_id).cloned().unwrap_or_default(),
                    "net_b": board.nets.get(t2.net_id).cloned().unwrap_or_default(),
                    "actual_clearance_mm": d - touch_dist,
                    "required_clearance_mm": req_clr,
                }));
            }
        }
    }

    // 2. Track-to-Via DRC
    for (ti, t) in board.tracks.iter().enumerate() {
        if t.net_id == 0 {
            continue;
        }
        let nc_t = board.get_netclass_for_net(t.net_id);

        for (vi, v) in board.vias.iter().enumerate() {
            if t.net_id == v.net_id || v.net_id == 0 {
                continue;
            }
            let nc_v = board.get_netclass_for_net(v.net_id);
            let req_clr = nc_t.clearance.max(nc_v.clearance);
            let d = dist_point_to_segment(v.position, t.start, t.end);
            let touch_dist = (t.width + v.size) * 0.5;

            if d < touch_dist - 1e-4 {
                shorts.push(json!({
                    "layer": t.layer,
                    "net_a": board.nets.get(t.net_id).cloned().unwrap_or_default(),
                    "net_b": board.nets.get(v.net_id).cloned().unwrap_or_default(),
                    "entity_a": format!("Track #{}", ti),
                    "entity_b": format!("Via #{} at ({:.2},{:.2})", vi, v.position.0, v.position.1),
                    "distance_mm": d,
                }));
            } else if d < touch_dist + req_clr - 1e-4 {
                clearance_violations.push(json!({
                    "layer": t.layer,
                    "net_a": board.nets.get(t.net_id).cloned().unwrap_or_default(),
                    "net_b": board.nets.get(v.net_id).cloned().unwrap_or_default(),
                    "actual_clearance_mm": d - touch_dist,
                    "required_clearance_mm": req_clr,
                }));
            }
        }
    }

    // 3. Via-to-Via DRC and Hole-to-Hole
    for i in 0..via_count {
        let v1 = &board.vias[i];
        let nc1 = board.get_netclass_for_net(v1.net_id);

        for j in (i + 1)..via_count {
            let v2 = &board.vias[j];
            let d = (v1.position.0 - v2.position.0).hypot(v1.position.1 - v2.position.1);

            // Hole-to-Hole check
            let hole_edge_dist = d - (v1.drill + v2.drill) * 0.5;
            if hole_edge_dist < 0.25 - 1e-4 {
                hole_to_hole_violations.push(json!({
                    "entity_a": format!("Via #{}", i),
                    "entity_b": format!("Via #{}", j),
                    "position_a": [v1.position.0, v1.position.1],
                    "position_b": [v2.position.0, v2.position.1],
                    "actual_distance_mm": hole_edge_dist,
                    "required_distance_mm": 0.25,
                }));
            }

            if v1.net_id != v2.net_id && v1.net_id != 0 && v2.net_id != 0 {
                let nc2 = board.get_netclass_for_net(v2.net_id);
                let req_clr = nc1.clearance.max(nc2.clearance);
                let touch_dist = (v1.size + v2.size) * 0.5;

                if d < touch_dist - 1e-4 {
                    shorts.push(json!({
                        "layer": 0,
                        "net_a": board.nets.get(v1.net_id).cloned().unwrap_or_default(),
                        "net_b": board.nets.get(v2.net_id).cloned().unwrap_or_default(),
                        "entity_a": format!("Via #{}", i),
                        "entity_b": format!("Via #{}", j),
                        "distance_mm": d,
                    }));
                } else if d < touch_dist + req_clr - 1e-4 {
                    clearance_violations.push(json!({
                        "layer": 0,
                        "net_a": board.nets.get(v1.net_id).cloned().unwrap_or_default(),
                        "net_b": board.nets.get(v2.net_id).cloned().unwrap_or_default(),
                        "actual_clearance_mm": d - touch_dist,
                        "required_clearance_mm": req_clr,
                    }));
                }
            }
        }
    }

    // 4. Track-to-Pad DRC
    for (ti, t) in board.tracks.iter().enumerate() {
        if t.net_id == 0 {
            continue;
        }
        let nc_t = board.get_netclass_for_net(t.net_id);
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id != t.net_id && pad.net_id != 0 {
                    if pad.is_through_hole || fp.layer == t.layer {
                        let d = dist_segment_to_pad(t.start, t.end, fp, pad);
                        let pad_c = board.get_netclass_for_net(pad.net_id).clearance;
                        let req_clr = nc_t.clearance.max(pad_c);
                        let touch_dist = t.width * 0.5;
                        if d < touch_dist - 1e-4 {
                            shorts.push(json!({
                                "layer": t.layer,
                                "net_a": board.nets.get(t.net_id).cloned().unwrap_or_default(),
                                "net_b": board.nets.get(pad.net_id).cloned().unwrap_or_default(),
                                "entity_a": format!("Track #{}", ti),
                                "entity_b": format!("Pad {} of {}", pad.number, fp.reference),
                                "distance_mm": d,
                            }));
                        } else if d < touch_dist + req_clr - 1e-4 {
                            clearance_violations.push(json!({
                                "layer": t.layer,
                                "net_a": board.nets.get(t.net_id).cloned().unwrap_or_default(),
                                "net_b": board.nets.get(pad.net_id).cloned().unwrap_or_default(),
                                "actual_clearance_mm": d - touch_dist,
                                "required_clearance_mm": req_clr,
                            }));
                        }
                    }
                }
            }
        }
    }

    // 5. Via-to-Pad DRC
    for (vi, v) in board.vias.iter().enumerate() {
        if v.net_id == 0 {
            continue;
        }
        let nc_v = board.get_netclass_for_net(v.net_id);
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id != v.net_id && pad.net_id != 0 {
                    let d = dist_point_to_pad(v.position, fp, pad);
                    let pad_c = board.get_netclass_for_net(pad.net_id).clearance;
                    let req_clr = nc_v.clearance.max(pad_c);
                    let touch_dist = v.size * 0.5;
                    if d < touch_dist - 1e-4 {
                        shorts.push(json!({
                            "layer": 0,
                            "net_a": board.nets.get(v.net_id).cloned().unwrap_or_default(),
                            "net_b": board.nets.get(pad.net_id).cloned().unwrap_or_default(),
                            "entity_a": format!("Via #{}", vi),
                            "entity_b": format!("Pad {} of {}", pad.number, fp.reference),
                            "distance_mm": d,
                        }));
                    } else if d < touch_dist + req_clr - 1e-4 {
                        clearance_violations.push(json!({
                            "layer": 0,
                            "net_a": board.nets.get(v.net_id).cloned().unwrap_or_default(),
                            "net_b": board.nets.get(pad.net_id).cloned().unwrap_or_default(),
                            "actual_clearance_mm": d - touch_dist,
                            "required_clearance_mm": req_clr,
                        }));
                    }
                }
            }
        }
    }

    // 6. Board-Edge Clearance Checks
    if let Some(ref poly) = board.outline_polygon {
        for (ti, t) in board.tracks.iter().enumerate() {
            let edge_clr = custom_edge_clr.unwrap_or_else(|| board.get_netclass_for_net(t.net_id).board_edge_clearance);
            let req_dist = t.width * 0.5 + edge_clr;
            let d = poly.distance_to_segment(t.start, t.end);
            if d < req_dist - 1e-4 {
                board_edge_violations.push(json!({
                    "entity": format!("Track #{}", ti),
                    "net": board.nets.get(t.net_id).cloned().unwrap_or_default(),
                    "position": [t.start.0, t.start.1],
                    "actual_clearance_mm": d - t.width * 0.5,
                    "required_clearance_mm": edge_clr,
                }));
            }
        }

        for (vi, v) in board.vias.iter().enumerate() {
            let edge_clr = custom_edge_clr.unwrap_or_else(|| board.get_netclass_for_net(v.net_id).board_edge_clearance);
            let req_dist = v.size * 0.5 + edge_clr;
            let pt = Point::new(v.position.0, v.position.1);
            let d = poly.distance_to_point(&pt);
            if d < req_dist - 1e-4 {
                board_edge_violations.push(json!({
                    "entity": format!("Via #{}", vi),
                    "net": board.nets.get(v.net_id).cloned().unwrap_or_default(),
                    "position": [v.position.0, v.position.1],
                    "actual_clearance_mm": d - v.size * 0.5,
                    "required_clearance_mm": edge_clr,
                }));
            }
        }
    }

    let total = shorts.len() + clearance_violations.len() + hole_to_hole_violations.len() + board_edge_violations.len();
    json!({
        "is_clean": total == 0,
        "total_violations": total,
        "shorts_count": shorts.len(),
        "clearance_count": clearance_violations.len(),
        "hole_to_hole_count": hole_to_hole_violations.len(),
        "board_edge_count": board_edge_violations.len(),
        "shorts": shorts,
        "clearance_violations": clearance_violations,
        "hole_to_hole_violations": hole_to_hole_violations,
        "board_edge_violations": board_edge_violations,
    })
}

fn collect_board_info(board: &Board) -> Value {
    let pad_positions: Vec<Vec<Point>> = board
        .footprints
        .iter()
        .map(|fp| {
            fp.pads
                .iter()
                .map(|p| {
                    let (x, y) = fp.pad_absolute_position(p);
                    Point::new(x, y)
                })
                .collect()
        })
        .collect();

    let ratsnest = board.generate_ratsnest_segments(&pad_positions);
    let ratsnest_len: f64 = ratsnest.iter().map(|s| s.length()).sum();

    json!({
        "loaded": true,
        "path": board.file_path,
        "dimensions_mm": {
            "width": board.max_x - board.min_x,
            "height": board.max_y - board.min_y,
            "min_x": board.min_x,
            "max_x": board.max_x,
            "min_y": board.min_y,
            "max_y": board.max_y,
        },
        "layers": board.copper_layers(),
        "footprint_count": board.footprints.len(),
        "locked_footprints": board.footprints.iter().filter(|f| f.is_locked).count(),
        "net_count": board.nets.len(),
        "track_count": board.tracks.len(),
        "via_count": board.vias.len(),
        "arc_count": board.arcs.len(),
        "keepout_count": board.keepouts.len(),
        "cluster_count": board.clusters.len(),
        "ratsnest_segments_count": ratsnest.len(),
        "ratsnest_length_mm": ratsnest_len,
    })
}

fn collect_telemetry_info(state: &McpServerState) -> Value {
    let telem_guard = state.telemetry.state.lock().unwrap();
    let cpu_load = telem_guard.cpu_load_pct;
    let gpu_load = telem_guard.gpu_load_pct;

    let (gpu_name, backend, driver, compute_busy_ns) = if let Some(ref vk) = state.vulkan {
        (
            vk.device_name.clone(),
            vk.backend_name.clone(),
            vk.driver_info.clone(),
            vk.compute_busy_ns.load(Ordering::Relaxed),
        )
    } else {
        (
            "None (CPU Fallback)".to_string(),
            "CPU Multithreaded".to_string(),
            "Rayon SIMD".to_string(),
            0,
        )
    };

    let ratsnest_len = state.board.as_ref().map(|b| {
        let pad_positions: Vec<Vec<Point>> = b
            .footprints
            .iter()
            .map(|fp| {
                fp.pads
                    .iter()
                    .map(|p| {
                        let (x, y) = fp.pad_absolute_position(p);
                        Point::new(x, y)
                    })
                    .collect()
            })
            .collect();
        let segments = b.generate_ratsnest_segments(&pad_positions);
        segments.iter().map(|s| s.length()).sum::<f64>()
    }).unwrap_or(0.0);

    json!({
        "cpu_load_pct": cpu_load,
        "gpu_load_pct": gpu_load,
        "gpu_name": gpu_name,
        "backend": backend,
        "driver_info": driver,
        "compute_busy_ns": compute_busy_ns,
        "ratsnest_length_mm": ratsnest_len,
        "footprints": state.board.as_ref().map(|b| b.footprints.len()).unwrap_or(0),
        "tracks": state.board.as_ref().map(|b| b.tracks.len()).unwrap_or(0),
        "vias": state.board.as_ref().map(|b| b.vias.len()).unwrap_or(0),
    })
}

/// Reads the next JSON-RPC frame from a BufRead source.
/// Seamlessly supports:
/// 1. Line-delimited JSON (NDJSON, starting with `{`)
/// 2. LSP/MCP Content-Length framing (`Content-Length: <n>\r\n\r\n{...}`)
pub fn read_next_json_rpc_frame<R: BufRead>(reader: &mut R) -> std::io::Result<Option<String>> {
    let mut header_line = String::new();

    loop {
        header_line.clear();
        let bytes_read = reader.read_line(&mut header_line)?;
        if bytes_read == 0 {
            return Ok(None); // EOF
        }

        let trimmed = header_line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Case 1: Line-delimited JSON
        if trimmed.starts_with('{') {
            return Ok(Some(trimmed.to_string()));
        }

        // Case 2: Content-Length header framing
        let lower = trimmed.to_lowercase();
        if lower.starts_with("content-length:") {
            let len_str = trimmed["content-length:".len()..].trim();
            let content_len: usize = len_str
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            // Read remaining headers until empty line
            loop {
                let mut next_header = String::new();
                let h_read = reader.read_line(&mut next_header)?;
                if h_read == 0 || next_header.trim().is_empty() {
                    break;
                }
            }

            // Read exact payload bytes
            let mut payload_buf = vec![0u8; content_len];
            reader.read_exact(&mut payload_buf)?;
            let payload_str = String::from_utf8(payload_buf)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            return Ok(Some(payload_str));
        }
    }
}

/// Runs the MCP server on standard I/O (stdio).
/// Reads JSON-RPC from stdin, writes JSON-RPC to stdout.
/// Diagnostics and logging go strictly to stderr.
pub fn run_mcp_stdio(server: Arc<McpServer>) -> std::io::Result<()> {
    eprintln!("[Vulcan's Trials MCP] Stdio transport active (JSON-RPC 2.0)");
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut stdout = std::io::stdout();

    while let Some(msg) = read_next_json_rpc_frame(&mut reader)? {
        if let Some(resp) = server.handle_message(&msg) {
            writeln!(stdout, "{}", resp)?;
            stdout.flush()?;
        }
    }

    eprintln!("[Vulcan's Trials MCP] Stdio connection closed");
    Ok(())
}

/// Runs the MCP server over TCP, binding to the given address.
/// Spawns a dedicated thread for each connected socket client.
pub fn run_mcp_tcp(server: Arc<McpServer>, bind_addr: &str) -> std::io::Result<()> {
    let listener = TcpListener::bind(bind_addr)?;
    let local_addr = listener.local_addr()?;
    eprintln!("[Vulcan's Trials MCP] TCP transport listening on {}", local_addr);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let server_clone = Arc::clone(&server);
                thread::spawn(move || {
                    if let Err(e) = handle_tcp_client(server_clone, stream) {
                        eprintln!("[Vulcan's Trials MCP] TCP client handler error: {}", e);
                    }
                });
            }
            Err(e) => {
                eprintln!("[Vulcan's Trials MCP] TCP accept error: {}", e);
            }
        }
    }

    Ok(())
}

fn handle_tcp_client(server: Arc<McpServer>, mut stream: TcpStream) -> std::io::Result<()> {
    let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_else(|_| "unknown".to_string());
    eprintln!("[Vulcan's Trials MCP] Client connected: {}", peer);

    let mut reader = BufReader::new(stream.try_clone()?);
    while let Some(msg) = read_next_json_rpc_frame(&mut reader)? {
        if let Some(resp) = server.handle_message(&msg) {
            writeln!(stream, "{}", resp)?;
            stream.flush()?;
        }
    }

    eprintln!("[Vulcan's Trials MCP] Client disconnected: {}", peer);
    Ok(())
}

/// Returns definitions and JSON schemas for all available tools
pub fn get_all_tool_definitions() -> Vec<Value> {
    vec![
        // 1. Board Management
        json!({
            "name": "load_board",
            "description": "Load a KiCad PCB file (.kicad_pcb) into Vulcan's Trials with automatic detection of sibling .kicad_pro and .kicad_sch files.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute or relative path to the .kicad_pcb file" },
                    "extra_locked": { "type": "array", "items": { "type": "string" }, "description": "Optional list of component references to lock" }
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "save_board",
            "description": "Save the current PCB layout to a .kicad_pcb file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Target file path (defaults to loaded file path)" }
                }
            }
        }),
        json!({
            "name": "export_specctra_ses",
            "description": "Export the current routing layout as a Specctra .ses session file for back-importing into KiCad.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Target .ses file path" }
                }
            }
        }),
        json!({
            "name": "get_board_info",
            "description": "Query comprehensive physical and netlist statistics for the currently loaded PCB.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),

        // 2. Placement
        json!({
            "name": "run_autoplacer",
            "description": "Execute the Parallel Tempering Simulated Annealing autoplacer with multi-core CPU and Vulkan GPU acceleration.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "duration_secs": { "type": "integer", "description": "Placement duration in seconds (default: 10)" },
                    "cooling_rate": { "type": "number", "description": "Annealing temperature scaling factor (default: 1.0)" },
                    "crossing_weight": { "type": "number", "description": "Ratsnest crossing penalty weight (default: 25.0)" },
                    "wirelength_weight": { "type": "number", "description": "Total wirelength weight (default: 1.0)" },
                    "overlap_penalty": { "type": "number", "description": "Courtyard overlap penalty (default: 2500.0)" },
                    "passive_affinity_weight": { "type": "number", "description": "Decoupling and passive tie affinity weight (default: 35.0)" },
                    "decoupling_affinity_weight": { "type": "number", "description": "IC power pin decoupling affinity weight (default: 60.0)" },
                    "allow_layer_flip": { "type": "boolean", "description": "Allow flipping components between Top and Bottom copper (default: false)" },
                    "enable_escape_clearance": { "type": "boolean", "description": "Expand courtyards for routing escape corridors (default: true)" }
                }
            }
        }),
        json!({
            "name": "legalize_placement",
            "description": "Legalize all component courtyards, eliminating overlaps and boundary keepout violations using SAT convex polygon collision physics.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "min_spacing_mm": { "type": "number", "description": "Minimum clearance spacing between courtyards in mm (default: 0.25)" }
                }
            }
        }),
        json!({
            "name": "pin_and_gate_swap",
            "description": "Optimize pin and gate assignments using Kuhn-Munkres (Hungarian) bipartite matching to untangle crossed ratsnest connections.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "cluster_components",
            "description": "Extract and assign functional component clusters from schematic sheets or passive ties for rigid group movement.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "schematic_path": { "type": "string", "description": "Optional path to sibling .kicad_sch file" }
                }
            }
        }),

        // 3. Routing
        json!({
            "name": "route_all",
            "description": "Execute full multi-layer 3D negotiated-congestion A* autorouting with rip-up, reroute, and obstacle distance field guidance.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "passes": { "type": "integer", "description": "Maximum rip-up & reroute iterations (default: 15)" },
                    "res": { "type": "number", "description": "Routing grid resolution in mm (default: 0.10)" },
                    "inner_plane_penalty": { "type": "number", "description": "Penalty factor for routing non-plane signals across inner planes (default: 8.0)" },
                    "any_angle": { "type": "boolean", "description": "Enable TopoR-style any-angle topological routing (default: false)" },
                    "crosstalk_penalty": { "type": "number", "description": "Penalty weight for parallel coupled runs (default: 0.0)" },
                    "allow_nudge": { "type": "boolean", "description": "Allow nudging blocking components and tracks (default: true)" }
                }
            }
        }),
        json!({
            "name": "route_bus_bundle",
            "description": "River route an ordered parallel bus of nets along a user-specified polyline guide with uniform equidistant spacing and concentric mitered corners.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Bus bundle name identifier (e.g. 'SPI_BUS')" },
                    "nets": { "type": "array", "items": { "type": "string" }, "description": "Ordered list of net names in the bundle" },
                    "guide_points": {
                        "type": "array",
                        "items": { "type": "array", "items": { "type": "number" } },
                        "description": "List of [x, y] guide polyline waypoints in mm"
                    },
                    "layer": { "type": "integer", "description": "Copper layer index (0 = F.Cu, default: 0)" },
                    "track_width": { "type": "number", "description": "Trace width in mm (default: 0.20)" },
                    "track_spacing": { "type": "number", "description": "Clearance spacing between adjacent bundle traces in mm (default: 0.20)" }
                },
                "required": ["name", "nets", "guide_points"]
            }
        }),
        json!({
            "name": "escape_bga",
            "description": "Generate 45° quadrant dog-bones and VIPPO microvias for all BGA components on the board.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "grid_res": { "type": "number", "description": "Grid resolution in mm (default: 0.10)" }
                }
            }
        }),
        json!({
            "name": "escape_dense_ics",
            "description": "Generate staggered escape stubs for fine-pitch QFNs, MCUs, and dense connectors to open routing channels.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "grid_res": { "type": "number", "description": "Grid resolution in mm (default: 0.10)" }
                }
            }
        }),
        json!({
            "name": "topor_any_angle",
            "description": "Apply TopoR-style homotopic rubber-band trace relaxation and insert smooth inscribed tangent circular arcs into track corners.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "tune_length",
            "description": "Tune inter-net and intra-pair trace lengths using serpentine trombone meanders to satisfy timing skew constraints.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": { "type": "string", "enum": ["diff_pair", "net"], "description": "Tuning mode: 'diff_pair' (default) or 'net'" },
                    "net_name": { "type": "string", "description": "Net name (for 'net' mode)" },
                    "net_id": { "type": "integer", "description": "Net ID (for 'net' mode)" },
                    "target_length_mm": { "type": "number", "description": "Target track length in mm (for 'net' mode)" },
                    "max_amplitude_mm": { "type": "number", "description": "Maximum serpentine trombone amplitude in mm (default: 1.0)" },
                    "net_p": { "type": "string", "description": "Positive net name (for 'diff_pair' mode)" },
                    "net_n": { "type": "string", "description": "Negative net name (for 'diff_pair' mode)" },
                    "tolerance_mm": { "type": "number", "description": "Maximum allowed skew tolerance in mm (default: 0.10)" }
                }
            }
        }),
        json!({
            "name": "stitch_ground_vias",
            "description": "Place thermal and high-speed return-path ground plane stitching vias across copper ground pours and signal layer transitions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "grid_res": { "type": "number", "description": "Grid resolution in mm (default: 0.10)" },
                    "thermal_pads": { "type": "boolean", "description": "Stitch component thermal pads (default: true)" },
                    "return_paths": { "type": "boolean", "description": "Insert return-path ground vias near layer transitions (default: true)" }
                }
            }
        }),
        json!({
            "name": "add_teardrops",
            "description": "Generate DFM teardrop fillets at track-to-pad and track-to-via junctions to prevent breakout during board fabrication.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "force_enable": { "type": "boolean", "description": "Force teardrops on all netclasses (default: true)" }
                }
            }
        }),
        json!({
            "name": "gloss_tracks",
            "description": "Run post-route track glossing: eliminates acid traps, smooths sharp corners, removes stair-step jogs, and straightens routes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "any_angle": { "type": "boolean", "description": "Allow any-angle smooth chamfers (default: false)" },
                    "grid_res": { "type": "number", "description": "Grid resolution in mm (default: 0.10)" }
                }
            }
        }),
        json!({
            "name": "optimize_vias",
            "description": "Perform automated via optimization: via pair reduction, single via consolidation, sliding, and jog elimination.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "grid_res": { "type": "number", "description": "Grid resolution in mm (default: 0.10)" }
                }
            }
        }),
        json!({
            "name": "shove_tracks",
            "description": "Push and deflect existing copper tracks away from newly placed traces or vias using continuous 45-degree spring deflection.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "track_index": { "type": "integer", "description": "Index of track to deflect directly" },
                    "offset_mm": { "type": "number", "description": "Lateral deflection offset in mm (default: 0.25)" },
                    "intruder_p1": { "type": "array", "items": { "type": "number" }, "description": "Intruder start coordinate [x, y]" },
                    "intruder_p2": { "type": "array", "items": { "type": "number" }, "description": "Intruder end coordinate [x, y]" },
                    "intruder_width": { "type": "number", "description": "Intruder track width in mm (default: 0.20)" },
                    "intruder_net": { "type": "string", "description": "Intruder net name" }
                }
            }
        }),

        // 4. DRC & Analysis
        json!({
            "name": "run_drc",
            "description": "Execute full Design Rule Check (DRC): verifies copper shorts, netclass clearance, hole-to-hole spacing, and board edge clearance.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "board_edge_clearance": { "type": "number", "description": "Minimum copper-to-edge distance override in mm" }
                }
            }
        }),
        json!({
            "name": "check_signal_integrity",
            "description": "Analyze high-speed signal integrity: detect parallel crosstalk coupled run lengths and verify unbroken reference plane return paths.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "crosstalk_max_coupling_mm": { "type": "number", "description": "Maximum allowed coupled length override in mm" }
                }
            }
        }),
        json!({
            "name": "get_telemetry",
            "description": "Fetch live CPU utilization, GPU compute utilization, Vulkan engine status, distance field counter, and ratsnest length.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Footprint, NetClass, Pad, TrackSegment};
    use std::io::Cursor;

    fn create_test_mcp_server() -> McpServer {
        let mut board = Board::default();
        board.file_path = "/tmp/test_board.kicad_pcb".to_string();
        board.nets = vec!["".to_string(), "GND".to_string(), "VCC".to_string(), "SIG".to_string()];
        board.net_name_to_id.insert("GND".to_string(), 1);
        board.net_name_to_id.insert("VCC".to_string(), 2);
        board.net_name_to_id.insert("SIG".to_string(), 3);

        let mut nc = NetClass::default();
        nc.clearance = 0.20;
        nc.track_width = 0.25;
        board.netclasses.insert("Default".to_string(), nc);

        // Add 2 footprints
        let fp1 = Footprint {
            reference: "U1".to_string(),
            value: "SOIC-8".to_string(),
            x: 10.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                Pad::new("1", -1.0, -1.0, 0.5, 0.5, 1, "GND", false),
                Pad::new("2", 1.0, -1.0, 0.5, 0.5, 2, "VCC", false),
            ],
            width: 4.0,
            height: 4.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: vec![],
            gate_swap_groups: vec![],
        };
        let fp2 = Footprint {
            reference: "U2".to_string(),
            value: "SOIC-8".to_string(),
            x: 30.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                Pad::new("1", -1.0, -1.0, 0.5, 0.5, 1, "GND", false),
                Pad::new("2", 1.0, -1.0, 0.5, 0.5, 2, "VCC", false),
            ],
            width: 4.0,
            height: 4.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: vec![],
            gate_swap_groups: vec![],
        };

        board.footprints.push(fp1);
        board.footprints.push(fp2);

        McpServer::new(Some(board), None)
    }

    #[test]
    fn test_mcp_initialize_handshake() {
        let server = create_test_mcp_server();
        let init_req = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        });

        let resp_str = server.handle_message(&init_req.to_string()).expect("Expected response");
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();

        assert_eq!(resp.jsonrpc, "2.0");
        assert_eq!(resp.id, Some(json!(1)));
        assert!(resp.error.is_none());

        let res = resp.result.expect("Expected result");
        assert_eq!(res["protocolVersion"], "2024-11-05");
        assert_eq!(res["serverInfo"]["name"], "vulcans-trials");
        assert!(res["capabilities"]["tools"].is_object());
        assert!(res["capabilities"]["resources"].is_object());
    }

    #[test]
    fn test_mcp_tools_list_coverage() {
        let server = create_test_mcp_server();
        let list_req = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        });

        let resp_str = server.handle_message(&list_req.to_string()).expect("Expected response");
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().expect("tools must be array");

        // Verify all 22 required tools exist
        let expected_tools = [
            "load_board", "save_board", "export_specctra_ses", "get_board_info",
            "run_autoplacer", "legalize_placement", "pin_and_gate_swap", "cluster_components",
            "route_all", "route_bus_bundle", "escape_bga", "escape_dense_ics",
            "topor_any_angle", "tune_length", "stitch_ground_vias", "add_teardrops",
            "gloss_tracks", "optimize_vias", "shove_tracks",
            "run_drc", "check_signal_integrity", "get_telemetry",
        ];

        let tool_names: HashSet<&str> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(|v| v.as_str()))
            .collect();

        for expected in &expected_tools {
            assert!(
                tool_names.contains(expected),
                "Missing required tool: {}",
                expected
            );
        }

        // Validate that every tool has name, description, and an object inputSchema
        for t in tools {
            assert!(t["name"].is_string());
            assert!(t["description"].is_string());
            assert_eq!(t["inputSchema"]["type"], "object");
        }
    }

    #[test]
    fn test_mcp_tools_call_get_board_info_and_drc() {
        let server = create_test_mcp_server();

        // 1. get_board_info
        let info_req = json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "get_board_info",
                "arguments": {}
            }
        });

        let resp_str = server.handle_message(&info_req.to_string()).unwrap();
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert_eq!(resp.id, Some(json!(10)));
        let text = resp.result.unwrap()["content"][0]["text"].as_str().unwrap().to_string();
        let board_info: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(board_info["footprint_count"], 2);
        assert_eq!(board_info["net_count"], 4);

        // 2. run_drc
        let drc_req = json!({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "run_drc",
                "arguments": {}
            }
        });

        let resp_str = server.handle_message(&drc_req.to_string()).unwrap();
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        let drc_text = resp.result.unwrap()["content"][0]["text"].as_str().unwrap().to_string();
        let drc_val: Value = serde_json::from_str(&drc_text).unwrap();
        assert_eq!(drc_val["total_violations"], 0);
        assert_eq!(drc_val["is_clean"], true);
    }

    #[test]
    fn test_mcp_tools_call_legalize_and_teardrops() {
        let server = create_test_mcp_server();

        // Add overlapping tracks to test teardrops and drc
        {
            let mut guard = server.state.lock().unwrap();
            let b = guard.board.as_mut().unwrap();
            b.tracks.push(TrackSegment {
                start: (10.0, 10.0),
                end: (20.0, 10.0),
                width: 0.25,
                layer: 0,
                net_id: 1,
                is_locked: false,
            });
        }

        // Test add_teardrops
        let td_req = json!({
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {
                "name": "add_teardrops",
                "arguments": {
                    "force_enable": true
                }
            }
        });

        let resp_str = server.handle_message(&td_req.to_string()).unwrap();
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_none());

        // Test legalize_placement
        let leg_req = json!({
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {
                "name": "legalize_placement",
                "arguments": {
                    "min_spacing_mm": 0.25
                }
            }
        });

        let resp_str = server.handle_message(&leg_req.to_string()).unwrap();
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        let leg_val: Value = serde_json::from_str(resp.result.unwrap()["content"][0]["text"].as_str().unwrap()).unwrap();
        assert!(leg_val["is_legal"].as_bool().unwrap());
    }

    #[test]
    fn test_mcp_resources_list_and_read() {
        let server = create_test_mcp_server();

        // 1. resources/list
        let res_list_req = json!({
            "jsonrpc": "2.0",
            "id": 30,
            "method": "resources/list"
        });
        let resp_str = server.handle_message(&res_list_req.to_string()).unwrap();
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        let resources = resp.result.unwrap()["resources"].as_array().unwrap().clone();
        assert_eq!(resources.len(), 2);

        // 2. resources/read kicad://board/current
        let read_req = json!({
            "jsonrpc": "2.0",
            "id": 31,
            "method": "resources/read",
            "params": {
                "uri": "kicad://board/current"
            }
        });
        let resp_str = server.handle_message(&read_req.to_string()).unwrap();
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        let contents = resp.result.unwrap()["contents"].as_array().unwrap().clone();
        assert_eq!(contents[0]["uri"], "kicad://board/current");
    }

    #[test]
    fn test_mcp_content_length_framing() {
        let payload = r#"{"jsonrpc":"2.0","id":42,"method":"ping"}"#;
        let framed = format!("Content-Length: {}\r\n\r\n{}", payload.len(), payload);

        let mut cursor = Cursor::new(framed.into_bytes());
        let frame = read_next_json_rpc_frame(&mut cursor).unwrap().expect("Should parse framed message");
        assert_eq!(frame, payload);
    }

    #[test]
    fn test_mcp_tcp_transport_loopback() {
        let server = Arc::new(create_test_mcp_server());

        // Bind on ephemeral port 0
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let local_addr = listener.local_addr().unwrap();

        // Spawn TCP listener thread
        let srv_clone = Arc::clone(&server);
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_tcp_client(srv_clone, stream).unwrap();
        });

        // Connect client
        let mut client_stream = TcpStream::connect(local_addr).unwrap();
        let init_req = json!({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "initialize",
            "params": { "protocolVersion": "2024-11-05" }
        });

        writeln!(client_stream, "{}", init_req).unwrap();
        client_stream.flush().unwrap();

        let mut reader = BufReader::new(client_stream.try_clone().unwrap());
        let resp_line = read_next_json_rpc_frame(&mut reader).unwrap().expect("Expected TCP response");
        let resp: JsonRpcResponse = serde_json::from_str(&resp_line).unwrap();

        assert_eq!(resp.id, Some(json!(99)));
        assert_eq!(resp.result.unwrap()["serverInfo"]["name"], "vulcans-trials");

        // Send a tool call over the same TCP connection
        let call_req = json!({
            "jsonrpc": "2.0",
            "id": 100,
            "method": "tools/call",
            "params": {
                "name": "get_board_info",
                "arguments": {}
            }
        });

        writeln!(client_stream, "{}", call_req).unwrap();
        client_stream.flush().unwrap();

        let resp_line2 = read_next_json_rpc_frame(&mut reader).unwrap().expect("Expected TCP tools/call response");
        let resp2: JsonRpcResponse = serde_json::from_str(&resp_line2).unwrap();
        assert_eq!(resp2.id, Some(json!(100)));
        assert!(!resp2.result.unwrap()["content"][0]["text"].as_str().unwrap().is_empty());

        // Close connection cleanly
        let _ = client_stream.shutdown(std::net::Shutdown::Both);
        drop(reader);
        drop(client_stream);
        let _ = handle.join();
    }
}
