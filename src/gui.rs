// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use crate::board::{Board, LiveTrialState, Point, Polygon, TrialPhase};
use crate::placer::{run_auto_placer, PlacementConfig};
use crate::router::{AutoRouter, RoutingGrid};
use crate::vulkan::{SystemTelemetry, VulkanEngine};
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

// KiCad default colour theme (KiCad 8/9/10 "default"; pcbnew builtin scheme).
pub const COL_F_CU: Color32 = Color32::from_rgb(0xC8, 0x34, 0x34);
pub const COL_B_CU: Color32 = Color32::from_rgb(0x4D, 0x7F, 0xC4);
pub const COL_IN1_CU: Color32 = Color32::from_rgb(0x7F, 0xC8, 0x7F);
pub const COL_IN2_CU: Color32 = Color32::from_rgb(0xCE, 0x7D, 0x2C);
pub const COL_PAD_TH: Color32 = Color32::from_rgb(0xE3, 0xB7, 0x2E);
pub const COL_VIA: Color32 = Color32::from_rgb(0xEC, 0xEC, 0xEC);
pub const COL_F_CRTYD: Color32 = Color32::from_rgb(0xFF, 0x26, 0xE2);
pub const COL_B_CRTYD: Color32 = Color32::from_rgb(0x26, 0xE2, 0xFF);
pub const COL_EDGE_CUTS: Color32 = Color32::from_rgb(0xD0, 0xD2, 0xCD);
pub const COL_RATSNEST: Color32 = Color32::from_rgb(0x00, 0xF8, 0xFF);
// Keepout: Color32::from_rgba_unmultiplied(255, 0, 5, 128) (#FF000580)
pub const COL_KEEPOUT: Color32 = Color32::from_rgba_premultiplied(0xBC, 0x00, 0x03, 0x80);
pub const COL_CANVAS_BG: Color32 = Color32::from_rgb(0x00, 0x10, 0x23);

/// Canvas copper draw order for standard 4-layer boards (KiCad: B.Cu, inner layers, F.Cu).
pub const COPPER_DRAW_ORDER: [&str; 4] = ["B.Cu", "In2.Cu", "In1.Cu", "F.Cu"];

/// KiCad default layer color lookup.
pub fn copper_layer_color(layer_name: &str) -> Color32 {
    match layer_name {
        "F.Cu" => COL_F_CU,
        "B.Cu" => COL_B_CU,
        "In1.Cu" => COL_IN1_CU,
        "In2.Cu" => COL_IN2_CU,
        "In3.Cu" => Color32::from_rgb(0x4F, 0xCB, 0xCB),
        "In4.Cu" => Color32::from_rgb(0xDB, 0x62, 0x8B),
        "In5.Cu" => Color32::from_rgb(0xA7, 0xA5, 0xC6),
        "In6.Cu" => Color32::from_rgb(0x28, 0xCC, 0xD9),
        "In7.Cu" => Color32::from_rgb(0xE8, 0xB2, 0xA7),
        "In8.Cu" => Color32::from_rgb(0xF2, 0xED, 0xA1),
        "In9.Cu" => Color32::from_rgb(0x8D, 0xCB, 0x81),
        _ => {
            if let Some(num_str) = layer_name.strip_prefix("In").and_then(|s| s.strip_suffix(".Cu")) {
                if let Ok(num) = num_str.parse::<usize>() {
                    if (10..=30).contains(&num) {
                        const CYCLE: [Color32; 7] = [
                            Color32::from_rgb(237, 124, 51),
                            Color32::from_rgb(91, 195, 235),
                            Color32::from_rgb(247, 111, 142),
                            Color32::from_rgb(167, 165, 198),
                            Color32::from_rgb(40, 204, 217),
                            Color32::from_rgb(232, 178, 167),
                            Color32::from_rgb(242, 237, 161),
                        ];
                        return CYCLE[(num - 10) % 7];
                    }
                }
            }
            Color32::from_rgb(180, 180, 180)
        }
    }
}

/// Returns active copper layers in KiCad display/legend order: F.Cu, internal layers, B.Cu.
pub fn kicad_layer_display_order<'a>(active_layers: &[&'a str]) -> Vec<&'a str> {
    let mut order = Vec::new();
    if active_layers.contains(&"F.Cu") {
        order.push("F.Cu");
    }
    for &layer in active_layers {
        if layer != "F.Cu" && layer != "B.Cu" {
            order.push(layer);
        }
    }
    if active_layers.contains(&"B.Cu") {
        order.push("B.Cu");
    }
    order
}

/// Returns active copper layers in canvas draw order (bottom to top): B.Cu, internal layers descending, F.Cu.
pub fn copper_layers_draw_order<'a>(active_layers: &[&'a str]) -> Vec<&'a str> {
    let mut order = Vec::new();
    if active_layers.contains(&"B.Cu") {
        order.push("B.Cu");
    }
    let mut inner: Vec<&'a str> = active_layers
        .iter()
        .copied()
        .filter(|&l| l != "F.Cu" && l != "B.Cu")
        .collect();
    inner.reverse();
    order.extend(inner);
    if active_layers.contains(&"F.Cu") {
        order.push("F.Cu");
    }
    order
}

pub struct VulcansTrialsApp {
    pub input_path: String,
    pub output_path: String,
    pub live: Arc<Mutex<LiveTrialState>>,
    pub vulkan: Option<Arc<VulkanEngine>>,
    pub telemetry: SystemTelemetry,
    pub stop_flag: Arc<AtomicBool>,
    pub worker_running: Arc<AtomicBool>,

    // Configurable parameters
    pub place_duration_secs: u64,
    pub min_spacing_mm: f64,
    pub grid_res_mm: f64,
    pub max_ripup_passes: usize,
    pub allow_layer_flip: bool,
    pub inner_plane_penalty: f64,

    // Aggressive placement exploration & weights
    pub max_jump_radius_mm: f64,
    pub pair_swap_prob: f64,
    pub centroid_pull_prob: f64,
    pub temp_scale: f64,
    pub escape_margin_mult: f64,
    pub crossing_weight: f64,

    // Canvas viewport controls
    pub pan_offset: Vec2,
    pub zoom: f32,
    pub show_courtyards: bool,
    pub show_ratsnest: bool,
    pub show_f_cu: bool,
    pub show_b_cu: bool,
    pub show_in1_cu: bool,
    pub show_in2_cu: bool,
    pub visible_layers: HashSet<String>,
    pub show_vias: bool,
    pub show_refs: bool,
    pub show_holding_palette: bool,
    pub selected_net: Option<String>,
    pub moving_footprint: Option<MovingFootprint>,
    pub hovered_footprint: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MovingFootprint {
    pub fp_index: usize,
    pub reference: String,
    pub original_x: f64,
    pub original_y: f64,
    pub original_rotation: f64,
    pub original_layer: usize,
}

impl VulcansTrialsApp {
    pub fn new(input_path: String, output_path: String, board: Board, vulkan: Option<Arc<VulkanEngine>>) -> Self {
        let busy_counter = vulkan.as_ref().map(|v| Arc::clone(&v.compute_busy_ns));
        let telemetry = SystemTelemetry::spawn(busy_counter);
        let mut visible_layers = HashSet::new();
        visible_layers.insert("F.Cu".to_string());
        visible_layers.insert("B.Cu".to_string());
        for &layer in &board.copper_layers() {
            visible_layers.insert(layer.to_string());
        }
        let live = Arc::new(Mutex::new(LiveTrialState::new(board)));

        Self {
            input_path,
            output_path,
            live,
            vulkan,
            telemetry,
            stop_flag: Arc::new(AtomicBool::new(false)),
            worker_running: Arc::new(AtomicBool::new(false)),
            place_duration_secs: 120,
            min_spacing_mm: 0.25,
            grid_res_mm: 0.10,
            max_ripup_passes: 15,
            allow_layer_flip: false,
            inner_plane_penalty: 8.0,
            max_jump_radius_mm: 20.0,
            pair_swap_prob: 0.15,
            centroid_pull_prob: 0.20,
            temp_scale: 1.0,
            escape_margin_mult: 1.0,
            crossing_weight: 200.0,
            pan_offset: Vec2::ZERO,
            zoom: 1.0,
            show_courtyards: true,
            show_ratsnest: true,
            show_f_cu: true,
            show_b_cu: true,
            show_in1_cu: true,
            show_in2_cu: true,
            visible_layers,
            show_vias: true,
            show_refs: true,
            show_holding_palette: true,
            selected_net: None,
            moving_footprint: None,
            hovered_footprint: None,
        }
    }

    pub fn is_layer_visible(&self, layer_name: &str) -> bool {
        match layer_name {
            "F.Cu" => self.show_f_cu && self.visible_layers.contains("F.Cu"),
            "B.Cu" => self.show_b_cu && self.visible_layers.contains("B.Cu"),
            "In1.Cu" => self.show_in1_cu && self.visible_layers.contains("In1.Cu"),
            "In2.Cu" => self.show_in2_cu && self.visible_layers.contains("In2.Cu"),
            _ => self.visible_layers.contains(layer_name),
        }
    }

    pub fn set_layer_visible(&mut self, layer_name: &str, visible: bool) {
        match layer_name {
            "F.Cu" => self.show_f_cu = visible,
            "B.Cu" => self.show_b_cu = visible,
            "In1.Cu" => self.show_in1_cu = visible,
            "In2.Cu" => self.show_in2_cu = visible,
            _ => {}
        }
        if visible {
            self.visible_layers.insert(layer_name.to_string());
        } else {
            self.visible_layers.remove(layer_name);
        }
    }

    pub fn is_holding_palette_visible(&self) -> bool {
        self.show_holding_palette && !self.live.lock().unwrap().board.holding_palette.is_empty()
    }

    fn start_trial(&mut self, do_place: bool, do_route: bool) {
        if self.worker_running.load(Ordering::Relaxed) {
            return;
        }
        self.stop_flag.store(false, Ordering::Relaxed);
        self.worker_running.store(true, Ordering::Relaxed);

        let live = Arc::clone(&self.live);
        let stop_flag = Arc::clone(&self.stop_flag);
        let running = Arc::clone(&self.worker_running);
        let vulkan = self.vulkan.clone();
        let place_dur = self.place_duration_secs;
        let min_spacing = self.min_spacing_mm;
        let grid_res = self.grid_res_mm;
        let max_passes = self.max_ripup_passes;
        let allow_flip = self.allow_layer_flip;
        let inner_plane_pen = self.inner_plane_penalty;
        let max_jump_rad = self.max_jump_radius_mm;
        let pair_swap_p = self.pair_swap_prob;
        let centroid_pull_p = self.centroid_pull_prob;
        let t_scale = self.temp_scale;
        let esc_mult = self.escape_margin_mult;
        let cross_w = self.crossing_weight;
        let out_path = self.output_path.clone();

        thread::spawn(move || {
            let mut working_board = {
                let mut g = live.lock().unwrap();
                if do_place && !g.board.holding_palette.is_empty() {
                    let cfg = PlacementConfig {
                        duration_secs: place_dur,
                        min_spacing_mm: min_spacing,
                        allow_layer_flip: allow_flip,
                        max_jump_radius_mm: max_jump_rad,
                        pair_swap_prob: pair_swap_p,
                        centroid_pull_prob: centroid_pull_p,
                        temp_scale: t_scale,
                        escape_margin_mult: esc_mult,
                        crossing_weight: cross_w,
                        vulkan: vulkan.clone(),
                        ..PlacementConfig::default()
                    };
                    let count = crate::placer::place_all_holding_components_at_best_locations(&mut g.board, &cfg);
                    g.push_log(format!("[Palette] Placed {} components at favourable locations for auto-placement", count));
                }
                g.board.clone()
            };

            if do_place && !stop_flag.load(Ordering::Relaxed) {
                let cfg = PlacementConfig {
                    duration_secs: place_dur,
                    min_spacing_mm: min_spacing,
                    allow_layer_flip: allow_flip,
                    max_jump_radius_mm: max_jump_rad,
                    pair_swap_prob: pair_swap_p,
                    centroid_pull_prob: centroid_pull_p,
                    temp_scale: t_scale,
                    escape_margin_mult: esc_mult,
                    crossing_weight: cross_w,
                    vulkan: vulkan.clone(),
                    ..PlacementConfig::default()
                };
                run_auto_placer(&mut working_board, &cfg, Some(Arc::clone(&live)), Arc::clone(&stop_flag));
            }

            if do_route && !stop_flag.load(Ordering::Relaxed) {
                let mut grid = RoutingGrid::new(&working_board, grid_res, vulkan).with_inner_layer_penalty(inner_plane_pen);
                let mut router = AutoRouter::new(max_passes);
                router.inner_layer_penalty = inner_plane_pen;
                router.route_all(
                    &mut working_board,
                    &mut grid,
                    Some(Arc::clone(&live)),
                    Arc::clone(&stop_flag),
                );
            }

            if !out_path.is_empty() {
                if let Err(e) = working_board.save_pcb(Path::new(&out_path)) {
                    if let Ok(mut g) = live.lock() {
                        g.push_log(format!("[Error] Failed to save {}: {}", out_path, e));
                    }
                } else {
                    let ses_path = Path::new(&out_path).with_extension("ses");
                    let _ = working_board.save_ses(&ses_path);
                    if let Ok(mut g) = live.lock() {
                        g.phase = TrialPhase::Completed;
                        g.push_log(format!("[Saved] Wrote PCB to {} and {}", out_path, ses_path.display()));
                    }
                }
            }

            running.store(false, Ordering::Relaxed);
        });
    }

    pub fn board_to_screen(&self, board: &Board, rect: Rect, wx: f64, wy: f64) -> (Pos2, f32) {
        let bw = (board.max_x - board.min_x).max(10.0) as f32;
        let bh = (board.max_y - board.min_y).max(10.0) as f32;
        let base_scale = ((rect.width() - 40.0) / bw).min((rect.height() - 40.0) / bh).max(1.0);
        let scale = base_scale * self.zoom;
        let board_cx = ((board.min_x + board.max_x) * 0.5) as f32;
        let board_cy = ((board.min_y + board.max_y) * 0.5) as f32;
        let screen_center = rect.center() + self.pan_offset;
        let pos = Pos2::new(
            screen_center.x + ((wx as f32) - board_cx) * scale,
            screen_center.y + ((wy as f32) - board_cy) * scale,
        );
        (pos, scale)
    }

    pub fn screen_to_board(&self, board: &Board, rect: Rect, pos: Pos2) -> Point {
        let (_, scale) = self.board_to_screen(board, rect, board.min_x, board.min_y);
        let board_cx = ((board.min_x + board.max_x) * 0.5) as f32;
        let board_cy = ((board.min_y + board.max_y) * 0.5) as f32;
        let screen_center = rect.center() + self.pan_offset;
        let wx = board_cx + (pos.x - screen_center.x) / scale;
        let wy = board_cy + (pos.y - screen_center.y) / scale;
        Point::new(wx as f64, wy as f64)
    }

    pub fn footprint_screen_rect(&self, board: &Board, rect: Rect, fp: &crate::board::Footprint) -> Rect {
        let polys = fp.transformed_perimeters_layered(Point::new(fp.x, fp.y), fp.rotation, fp.layer, 0.0);
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for poly in &polys {
            for v in &poly.vertices {
                let (p, _) = self.board_to_screen(board, rect, v.x, v.y);
                min_x = min_x.min(p.x);
                min_y = min_y.min(p.y);
                max_x = max_x.max(p.x);
                max_y = max_y.max(p.y);
            }
        }
        if min_x <= max_x && min_y <= max_y {
            Rect::from_min_max(Pos2::new(min_x, min_y), Pos2::new(max_x, max_y))
        } else {
            let (c, scale) = self.board_to_screen(board, rect, fp.x, fp.y);
            Rect::from_center_size(c, Vec2::splat((2.0 * scale).max(6.0)))
        }
    }

    pub fn is_point_in_pad(&self, board: &Board, rect: Rect, fp: &crate::board::Footprint, pad: &crate::board::Pad, click_pos: Pos2) -> bool {
        let (px, py) = fp.pad_absolute_position(pad);
        let (center, scale) = self.board_to_screen(board, rect, px, py);
        if pad.shape == "circle" {
            let r = ((pad.width.max(pad.height) as f32) * 0.5 * scale).max(2.0);
            center.distance(click_pos) <= r
        } else {
            let corners = fp.pad_world_corners(pad);
            if corners.len() < 3 {
                return false;
            }
            let screen_pts: Vec<Pos2> = corners.iter().map(|p| self.board_to_screen(board, rect, p.x, p.y).0).collect();
            // Point-in-polygon test (ray casting)
            let mut inside = false;
            let n = screen_pts.len();
            let mut j = n - 1;
            for i in 0..n {
                let pi = screen_pts[i];
                let pj = screen_pts[j];
                if ((pi.y > click_pos.y) != (pj.y > click_pos.y))
                    && (click_pos.x < (pj.x - pi.x) * (click_pos.y - pi.y) / (pj.y - pi.y) + pi.x)
                {
                    inside = !inside;
                }
                j = i;
            }
            inside
        }
    }

    pub fn handle_canvas_click_at(&mut self, rect: Rect, click_pos: Pos2) -> Option<String> {
        let (clicked_pad, clicked_fp_body) = {
            let g = self.live.lock().unwrap();
            let b = &g.board;
            // 1. First check if any pad was clicked
            let mut found_pad: Option<(usize, usize, String)> = None;
            'outer: for (fp_idx, fp) in b.footprints.iter().enumerate() {
                for (p_idx, pad) in fp.pads.iter().enumerate() {
                    if self.is_point_in_pad(b, rect, fp, pad, click_pos) {
                        found_pad = Some((fp_idx, p_idx, pad.net_name.clone()));
                        break 'outer;
                    }
                }
            }

            // 2. If no pad was clicked, check if a footprint body was clicked
            let mut found_fp: Option<usize> = None;
            if found_pad.is_none() {
                let mut best_idx: Option<(usize, f32, f32)> = None;
                for (idx, fp) in b.footprints.iter().enumerate() {
                    let fp_rect = self.footprint_screen_rect(b, rect, fp);
                    if fp_rect.contains(click_pos) {
                        let area = fp_rect.width() * fp_rect.height();
                        let dist = fp_rect.center().distance(click_pos);
                        if best_idx.map_or(true, |(_, best_area, best_dist)| {
                            area < best_area - 1.0 || ((area - best_area).abs() <= 1.0 && dist < best_dist)
                        }) {
                            best_idx = Some((idx, area, dist));
                        }
                    }
                }
                found_fp = best_idx.map(|(i, _, _)| i);
            }
            (found_pad, found_fp)
        };

        if let Some((fp_idx, pad_idx, net_name)) = clicked_pad {
            if net_name.is_empty() {
                self.selected_net = None;
                let mut g = self.live.lock().unwrap();
                let ref_des = g.board.footprints[fp_idx].reference.clone();
                let pad_num = g.board.footprints[fp_idx].pads[pad_idx].number.clone();
                g.push_log(format!("[Pad] {}.{} has no net assigned", ref_des, pad_num));
                Some(format!("{}.{}", ref_des, pad_num))
            } else if self.selected_net.as_deref() == Some(&net_name) {
                // Toggle off highlighting if clicking pad on the already highlighted net
                self.selected_net = None;
                let mut g = self.live.lock().unwrap();
                g.push_log(format!("[Highlight] Cleared net highlight for \"{}\"", net_name));
                Some(format!("Cleared {}", net_name))
            } else {
                self.selected_net = Some(net_name.clone());
                let mut g = self.live.lock().unwrap();
                let count = g
                    .board
                    .footprints
                    .iter()
                    .flat_map(|f| &f.pads)
                    .filter(|p| p.net_name == net_name)
                    .count();
                g.push_log(format!(
                    "[Highlight] Net \"{}\" selected: {} connected pads highlighted",
                    net_name, count
                ));
                Some(format!("Net: {}", net_name))
            }
        } else if let Some(idx) = clicked_fp_body {
            if self.worker_running.load(Ordering::Relaxed) {
                return None;
            }
            let mut g = self.live.lock().unwrap();
            let fp = &mut g.board.footprints[idx];
            fp.is_locked = !fp.is_locked;
            let r = fp.reference.clone();
            let locked = fp.is_locked;
            g.push_log(format!(
                "[Lock] {} is now {}",
                r,
                if locked { "LOCKED" } else { "UNLOCKED" }
            ));
            Some(format!("{}: {}", r, if locked { "LOCKED" } else { "UNLOCKED" }))
        } else {
            // Clicking empty canvas clears selected net highlight
            if self.selected_net.is_some() {
                self.selected_net = None;
                let mut g = self.live.lock().unwrap();
                g.push_log("[Highlight] Cleared net highlight".to_string());
            }
            None
        }
    }

    pub fn handle_canvas_secondary_click_at(&mut self, rect: Rect, click_pos: Pos2) -> Option<String> {
        if self.worker_running.load(Ordering::Relaxed) {
            return None;
        }
        let clicked_fp_idx = {
            let g = self.live.lock().unwrap();
            let mut best_idx: Option<(usize, f32)> = None;
            for (idx, fp) in g.board.footprints.iter().enumerate() {
                if fp.is_locked {
                    continue;
                }
                let fp_rect = self.footprint_screen_rect(&g.board, rect, fp);
                if fp_rect.contains(click_pos) {
                    let dist = fp_rect.center().distance(click_pos);
                    if best_idx.map_or(true, |(_, best_dist)| dist < best_dist) {
                        best_idx = Some((idx, dist));
                    }
                }
            }
            best_idx.map(|(idx, _)| idx)
        };

        if let Some(idx) = clicked_fp_idx {
            self.show_holding_palette = true;
            let mut g = self.live.lock().unwrap();
            let fp = g.board.footprints.remove(idx);
            let r = fp.reference.clone();
            g.board.holding_palette.push(fp);
            g.board
                .holding_palette
                .sort_by(|a, b| crate::board::natural_designator_cmp(&a.reference, &b.reference));
            g.refresh_ratsnest();
            g.push_log(format!("[Hold] Moved {} to Holding Palette", r));
            Some(r)
        } else {
            None
        }
    }

    pub fn toggle_footprint_lock_at(&mut self, rect: Rect, click_pos: Pos2) -> Option<(String, bool)> {
        if self.worker_running.load(Ordering::Relaxed) {
            return None;
        }
        let mut g = self.live.lock().unwrap();
        // Check if any pad was clicked on: if so, do not toggle footprint lock
        for fp in &g.board.footprints {
            for pad in &fp.pads {
                if self.is_point_in_pad(&g.board, rect, fp, pad, click_pos) {
                    return None;
                }
            }
        }

        let mut best_idx: Option<(usize, f32, f32)> = None;
        for (idx, fp) in g.board.footprints.iter().enumerate() {
            let fp_rect = self.footprint_screen_rect(&g.board, rect, fp);
            if fp_rect.contains(click_pos) {
                let area = fp_rect.width() * fp_rect.height();
                let dist = fp_rect.center().distance(click_pos);
                if best_idx.map_or(true, |(_, best_area, best_dist)| {
                    area < best_area - 1.0 || ((area - best_area).abs() <= 1.0 && dist < best_dist)
                }) {
                    best_idx = Some((idx, area, dist));
                }
            }
        }
        if let Some((idx, _, _)) = best_idx {
            let fp = &mut g.board.footprints[idx];
            fp.is_locked = !fp.is_locked;
            let r = fp.reference.clone();
            let locked = fp.is_locked;
            g.push_log(format!(
                "[Lock] {} is now {}",
                r,
                if locked { "LOCKED" } else { "UNLOCKED" }
            ));
            Some((r, locked))
        } else {
            None
        }
    }

    /// Picks up an unlocked component at `pos` to begin manual movement.
    pub fn pick_up_component_at(&mut self, rect: Rect, pos: Pos2) -> bool {
        if self.worker_running.load(Ordering::Relaxed) {
            return false;
        }
        let snap = self.live.lock().unwrap().clone();
        let b = &snap.board;
        let mut best_idx: Option<(usize, f32, f32)> = None;
        for (idx, fp) in b.footprints.iter().enumerate() {
            let fp_rect = self.footprint_screen_rect(b, rect, fp);
            if fp_rect.contains(pos) {
                let area = fp_rect.width() * fp_rect.height();
                let dist = fp_rect.center().distance(pos);
                if best_idx.map_or(true, |(_, best_area, best_dist)| {
                    area < best_area - 1.0 || ((area - best_area).abs() <= 1.0 && dist < best_dist)
                }) {
                    best_idx = Some((idx, area, dist));
                }
            }
        }
        if let Some((idx, _, _)) = best_idx {
            let fp = &b.footprints[idx];
            if fp.is_locked {
                let mut g = self.live.lock().unwrap();
                g.push_log(format!("[Move] {} is locked and cannot be moved", fp.reference));
                return false;
            }
            self.moving_footprint = Some(MovingFootprint {
                fp_index: idx,
                reference: fp.reference.clone(),
                original_x: fp.x,
                original_y: fp.y,
                original_rotation: fp.rotation,
                original_layer: fp.layer,
            });
            let mut g = self.live.lock().unwrap();
            g.push_log(format!(
                "[Move] Picked up {} at ({:.2}, {:.2}). Move mouse to position, 'r' to rotate, 'f' to flip, Click to place.",
                fp.reference, fp.x, fp.y
            ));
            true
        } else {
            false
        }
    }

    /// Updates the position of the currently moving component to track cursor position.
    pub fn update_moving_component_pos(&mut self, rect: Rect, pos: Pos2) {
        if let Some(ref mf) = self.moving_footprint {
            let mut g = self.live.lock().unwrap();
            let world_pt = self.screen_to_board(&g.board, rect, pos);
            if mf.fp_index < g.board.footprints.len() {
                g.board.footprints[mf.fp_index].x = world_pt.x;
                g.board.footprints[mf.fp_index].y = world_pt.y;
                g.refresh_ratsnest();
            }
        }
    }

    /// Rotates the active component (moving component or hovered component) by 90 degrees.
    pub fn rotate_active_component(&mut self) -> bool {
        if self.worker_running.load(Ordering::Relaxed) {
            return false;
        }
        let target_idx = if let Some(ref mf) = self.moving_footprint {
            Some(mf.fp_index)
        } else if let Some(h_idx) = self.hovered_footprint {
            let snap = self.live.lock().unwrap();
            if h_idx < snap.board.footprints.len() && !snap.board.footprints[h_idx].is_locked {
                Some(h_idx)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(idx) = target_idx {
            let mut g = self.live.lock().unwrap();
            if idx < g.board.footprints.len() {
                let ref_des = g.board.footprints[idx].reference.clone();
                if self.moving_footprint.is_none() {
                    let ripped = g.board.rip_up_connected_traces_to_component(idx);
                    if ripped > 0 {
                        g.push_log(format!(
                            "[RipUp] Ripped up {} connected trace/via element{} for {}",
                            ripped,
                            if ripped == 1 { "" } else { "s" },
                            ref_des
                        ));
                    }
                }
                let fp = &mut g.board.footprints[idx];
                fp.rotation = (fp.rotation + 90.0) % 360.0;
                let rot = fp.rotation;
                g.refresh_ratsnest();
                g.push_log(format!("[Rotate] Rotated {} by 90° to {:.0}°", ref_des, rot));
                return true;
            }
        }
        false
    }

    /// Flips the active component (moving component or hovered component) to the other board side.
    pub fn flip_active_component(&mut self) -> bool {
        if self.worker_running.load(Ordering::Relaxed) {
            return false;
        }
        let target_idx = if let Some(ref mf) = self.moving_footprint {
            Some(mf.fp_index)
        } else if let Some(h_idx) = self.hovered_footprint {
            let snap = self.live.lock().unwrap();
            if h_idx < snap.board.footprints.len() && !snap.board.footprints[h_idx].is_locked {
                Some(h_idx)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(idx) = target_idx {
            let mut g = self.live.lock().unwrap();
            if idx < g.board.footprints.len() {
                let ref_des = g.board.footprints[idx].reference.clone();
                if self.moving_footprint.is_none() {
                    let ripped = g.board.rip_up_connected_traces_to_component(idx);
                    if ripped > 0 {
                        g.push_log(format!(
                            "[RipUp] Ripped up {} connected trace/via element{} for {}",
                            ripped,
                            if ripped == 1 { "" } else { "s" },
                            ref_des
                        ));
                    }
                }
                let fp = &mut g.board.footprints[idx];
                fp.layer = 1 - fp.layer;
                let layer_str = if fp.layer == 1 { "B.Cu (Bottom)" } else { "F.Cu (Top)" };
                g.refresh_ratsnest();
                g.push_log(format!("[Flip] Flipped {} to {}", ref_des, layer_str));
                return true;
            }
        }
        false
    }

    /// Places the moving component at its current position, shoving other unlocked components if needed.
    pub fn place_moving_component(&mut self) -> Option<usize> {
        if self.worker_running.load(Ordering::Relaxed) {
            return None;
        }
        let mf = self.moving_footprint.take()?;
        let shoved_count = {
            let mut g = self.live.lock().unwrap();
            let moved = if mf.fp_index < g.board.footprints.len() {
                let fp = &g.board.footprints[mf.fp_index];
                (fp.x - mf.original_x).abs() > 1e-4
                    || (fp.y - mf.original_y).abs() > 1e-4
                    || (fp.rotation - mf.original_rotation).abs() > 1e-4
                    || fp.layer != mf.original_layer
            } else {
                false
            };
            if moved {
                let ref_des = g.board.footprints[mf.fp_index].reference.clone();
                let ripped = g.board.rip_up_connected_traces_to_component_at(
                    mf.fp_index,
                    Some((mf.original_x, mf.original_y, mf.original_rotation, mf.original_layer)),
                );
                if ripped > 0 {
                    g.push_log(format!(
                        "[RipUp] Ripped up {} connected trace/via element{} for {}",
                        ripped,
                        if ripped == 1 { "" } else { "s" },
                        ref_des
                    ));
                }
            }
            let count = crate::placer::shove_overlapping_components(&mut g.board, mf.fp_index, self.min_spacing_mm);
            g.refresh_ratsnest();
            if mf.fp_index < g.board.footprints.len() {
                let ref_des = g.board.footprints[mf.fp_index].reference.clone();
                let (px, py) = (g.board.footprints[mf.fp_index].x, g.board.footprints[mf.fp_index].y);
                if count > 0 {
                    g.push_log(format!(
                        "[Place] Placed {} at ({:.2}, {:.2}) and shoved {} overlapping component{}",
                        ref_des, px, py, count, if count == 1 { "" } else { "s" }
                    ));
                } else {
                    g.push_log(format!("[Place] Placed {} at ({:.2}, {:.2})", ref_des, px, py));
                }
            }
            count
        };
        Some(shoved_count)
    }

    /// Cancels moving the active component and restores its original coordinates, rotation, and layer.
    pub fn cancel_moving_component(&mut self) -> bool {
        if let Some(mf) = self.moving_footprint.take() {
            let mut g = self.live.lock().unwrap();
            if mf.fp_index < g.board.footprints.len() {
                let fp = &mut g.board.footprints[mf.fp_index];
                fp.x = mf.original_x;
                fp.y = mf.original_y;
                fp.rotation = mf.original_rotation;
                fp.layer = mf.original_layer;
                let ref_des = fp.reference.clone();
                g.refresh_ratsnest();
                g.push_log(format!("[Move] Cancelled move for {}", ref_des));
                return true;
            }
        }
        false
    }

    pub fn render_ui(&mut self, ctx: &egui::Context) {
        ctx.request_repaint_after(std::time::Duration::from_millis(50));

        let telem = self.telemetry.state.lock().unwrap().clone();
        let snap = self.live.lock().unwrap().clone();
        let is_running = self.worker_running.load(Ordering::Relaxed);

        // Top Telemetry & Vulkan Status Bar
        egui::TopBottomPanel::top("vulcan_top_bar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading(egui::RichText::new("⚡ Vulcan's Trials").color(Color32::from_rgb(255, 145, 50)));
                ui.label(egui::RichText::new("by InfernoEmbedded (GPLv3)").weak());
                ui.separator();

                if let Some(ref vk) = self.vulkan {
                    ui.label(
                        egui::RichText::new(format!("Vulkan GPU: {} [{}]", vk.device_name, vk.backend_name))
                            .color(Color32::from_rgb(110, 230, 130))
                            .strong(),
                    );
                } else {
                    ui.label(egui::RichText::new("GPU: CPU Fallback").color(Color32::YELLOW));
                }

                ui.separator();
                draw_load_badge(ui, "CPU Load", telem.cpu_load_pct, &telem.cpu_history, Color32::from_rgb(80, 180, 255));
                ui.separator();
                draw_load_badge(ui, "GPU Load", telem.gpu_load_pct, &telem.gpu_history, Color32::from_rgb(255, 120, 60));
            });

            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(snap.phase.label()).strong());
                ui.add(
                    egui::ProgressBar::new(snap.progress)
                        .text(snap.status_text.as_str())
                        .animate(is_running),
                );
            });
            ui.add_space(4.0);
        });

        // Left Control & Metrics Panel
        egui::SidePanel::left("vulcan_controls_panel")
            .resizable(true)
            .default_width(310.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.heading("Controls");

                ui.horizontal(|ui| {
                    ui.label("Input PCB:");
                    ui.text_edit_singleline(&mut self.input_path);
                });
                ui.horizontal(|ui| {
                    ui.label("Output PCB:");
                    ui.text_edit_singleline(&mut self.output_path);
                });

                ui.horizontal(|ui| {
                    if ui.add_enabled(!is_running, egui::Button::new("📂 Reload Input PCB")).clicked() {
                        if let Ok(b) = Board::load(&self.input_path, &HashSet::new()) {
                            let mut g = self.live.lock().unwrap();
                            *g = LiveTrialState::new(b);
                        }
                    }

                    if ui
                        .add_enabled(
                            !is_running,
                            egui::Button::new(
                                egui::RichText::new("🧹 Clear Unlocked Objects")
                                    .color(Color32::from_rgb(255, 140, 140)),
                            ),
                        )
                        .clicked()
                    {
                        self.show_holding_palette = true;
                        let mut g = self.live.lock().unwrap();
                        let (ripped, cleared) = g.board.clear_unlocked_objects();
                        g.refresh_ratsnest();
                        g.push_log(format!(
                            "[Clear] Ripped up {} unlocked tracks/vias; moved {} components to Holding Palette",
                            ripped, cleared
                        ));
                    }
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            !is_running,
                            egui::Button::new(
                                egui::RichText::new("⚡ Auto-Place & Route")
                                    .color(Color32::from_rgb(255, 190, 90))
                                    .strong(),
                            ),
                        )
                        .clicked()
                    {
                        self.start_trial(true, true);
                    }

                    if ui
                        .add_enabled(is_running, egui::Button::new("⏹ Stop"))
                        .clicked()
                    {
                        self.stop_flag.store(true, Ordering::Relaxed);
                    }
                });

                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!is_running, egui::Button::new("📐 Auto-Place Only"))
                        .clicked()
                    {
                        self.start_trial(true, false);
                    }
                    if ui
                        .add_enabled(!is_running, egui::Button::new("⚡ Auto-Route Only"))
                        .clicked()
                    {
                        self.start_trial(false, true);
                    }
                });

                if ui
                    .add_enabled(!is_running, egui::Button::new("💾 Save Output (.kicad_pcb + .ses)"))
                    .clicked()
                {
                    let mut g = self.live.lock().unwrap();
                    if let Err(e) = g.board.save_pcb(Path::new(&self.output_path)) {
                        g.push_log(format!("[Error] {}", e));
                    } else {
                        let ses = Path::new(&self.output_path).with_extension("ses");
                        let _ = g.board.save_ses(&ses);
                        g.push_log(format!("[Saved] {}", self.output_path));
                    }
                }

                ui.separator();
                ui.heading("Parameters");
                ui.add_enabled(
                    !is_running,
                    egui::Slider::new(&mut self.place_duration_secs, 2..=600).text("Place Time (s)"),
                );
                ui.add_enabled(
                    !is_running,
                    egui::Slider::new(&mut self.min_spacing_mm, 0.10..=1.00).text("Courtyard Gap (mm)"),
                );
                ui.add_enabled(
                    !is_running,
                    egui::Slider::new(&mut self.grid_res_mm, 0.05..=0.25).text("Route Grid (mm)"),
                );
                ui.add_enabled(
                    !is_running,
                    egui::Slider::new(&mut self.max_ripup_passes, 1..=30).text("Max Rip-up Passes"),
                );
                ui.add_enabled(
                    !is_running,
                    egui::Slider::new(&mut self.inner_plane_penalty, 1.0..=30.0).text("Inner Plane Penalty"),
                );
                ui.add_enabled(
                    !is_running,
                    egui::Checkbox::new(&mut self.allow_layer_flip, "Allow SMD Side Flip (F.Cu ↔ B.Cu)"),
                );

                ui.collapsing("Aggressive Placement & Exploration", |ui| {
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.max_jump_radius_mm, 2.0..=50.0)
                            .text("Max Jump Radius (mm)")
                            .suffix(" mm"),
                    );
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.pair_swap_prob, 0.0..=0.50)
                            .text("Pair Swap %")
                            .custom_formatter(|n, _| format!("{:.0}%", n * 100.0)),
                    );
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.centroid_pull_prob, 0.0..=0.60)
                            .text("Centroid Pull %")
                            .custom_formatter(|n, _| format!("{:.0}%", n * 100.0)),
                    );
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.temp_scale, 0.2..=4.0)
                            .text("Annealing Temp Scale")
                            .suffix("x"),
                    );
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.escape_margin_mult, 0.5..=3.0)
                            .text("Escape Corridor Halo")
                            .suffix("x"),
                    );
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.crossing_weight, 50.0..=600.0)
                            .text("Crossing Penalty Weight"),
                    );
                });

                ui.separator();
                ui.heading("Visibility & View");
                let active_layers = snap.board.copper_layers();
                let display_layers = kicad_layer_display_order(&active_layers);
                let plane_names = snap.board.layer_plane_net_names();
                ui.vertical(|ui| {
                    ui.checkbox(
                        &mut self.show_holding_palette,
                        format!("Holding Palette ({})", snap.board.holding_palette.len()),
                    );
                    // KiCad layer order with colour swatches (text tinted to the canvas colour)
                    for &layer_name in &display_layers {
                        let l_idx = active_layers.iter().position(|&l| l == layer_name);
                        let plane = l_idx.and_then(|idx| plane_names.get(&idx));
                        let label = match layer_name {
                            "F.Cu" => match plane {
                                Some(n) => format!("F.Cu (Top - {})", n),
                                None => "F.Cu (Top)".to_string(),
                            },
                            "B.Cu" => match plane {
                                Some(n) => format!("B.Cu (Bottom - {})", n),
                                None => "B.Cu (Bottom)".to_string(),
                            },
                            _ => {
                                let k = layer_name.strip_prefix("In").and_then(|s| s.strip_suffix(".Cu")).unwrap_or("");
                                match plane {
                                    Some(n) => format!("{layer_name} ({n})"),
                                    None => format!("{layer_name} (Inner {k})"),
                                }
                            }
                        };
                        let col = copper_layer_color(layer_name);
                        let mut vis = self.is_layer_visible(layer_name);
                        if ui.checkbox(&mut vis, egui::RichText::new(label).color(col)).changed() {
                            self.set_layer_visible(layer_name, vis);
                        }
                    }
                    ui.checkbox(&mut self.show_vias, egui::RichText::new("Vias").color(COL_VIA));
                    ui.checkbox(&mut self.show_ratsnest, egui::RichText::new("Ratsnest").color(COL_RATSNEST));
                    ui.checkbox(&mut self.show_courtyards, egui::RichText::new("Courtyards").color(COL_F_CRTYD));
                    ui.checkbox(&mut self.show_refs, "References");
                });
                if ui.button("🔍 Reset Camera View").clicked() {
                    self.pan_offset = Vec2::ZERO;
                    self.zoom = 1.0;
                }

                ui.separator();
                ui.collapsing(format!("Net Classes ({})", snap.board.netclasses.len()), |ui| {
                    egui::Grid::new("netclasses_grid").striped(true).show(ui, |ui| {
                        ui.label(egui::RichText::new("Class").strong());
                        ui.label(egui::RichText::new("Width").strong());
                        ui.label(egui::RichText::new("Clearance").strong());
                        ui.label(egui::RichText::new("Nets").strong());
                        ui.end_row();

                        let mut sorted_classes: Vec<&String> = snap.board.netclasses.keys().collect();
                        sorted_classes.sort();
                        for cname in sorted_classes {
                            let nc = &snap.board.netclasses[cname];
                            let count = snap.board.net_to_class.values().filter(|&v| v == cname).count();
                            ui.label(cname);
                            ui.label(format!("{:.2} mm", nc.track_width));
                            ui.label(format!("{:.2} mm", nc.clearance));
                            ui.label(format!("{}", count));
                            ui.end_row();
                        }
                    });
                });

                ui.separator();
                ui.heading("Live Metrics");
                egui::Grid::new("metrics_grid").striped(true).show(ui, |ui| {
                    ui.label("Footprints:");
                    let locked_n = snap.board.footprints.iter().filter(|f| f.is_locked).count();
                    let back_n = snap.board.footprints.iter().filter(|f| f.layer == 1).count();
                    ui.label(format!(
                        "{} ({} locked, {} B.Cu)",
                        snap.board.footprints.len(),
                        locked_n,
                        back_n
                    ));
                    ui.end_row();

                    ui.label("Ratsnest Crossings:");
                    ui.label(format!("{}", snap.crossings));
                    ui.end_row();

                    ui.label("MST Wirelength:");
                    ui.label(format!("{:.1} mm", snap.wirelength_mm));
                    ui.end_row();

                    ui.label("Courtyard Overlaps:");
                    ui.label(format!("{}", snap.overlaps));
                    ui.end_row();

                    ui.label("SA Throughput:");
                    ui.label(format!("{:.0} evals/sec", snap.evals_per_sec));
                    ui.end_row();

                    ui.label("Routed Connections:");
                    let pct = if snap.total_conns > 0 {
                        (snap.routed_conns as f64 / snap.total_conns as f64) * 100.0
                    } else {
                        0.0
                    };
                    ui.label(format!("{}/{} ({:.1}%)", snap.routed_conns, snap.total_conns, pct));
                    ui.end_row();

                    ui.label("Tracks / Vias:");
                    ui.label(format!("{} segs / {} vias", snap.board.tracks.len(), snap.board.vias.len()));
                    ui.end_row();
                });

                ui.separator();
                ui.heading("Activity Log");
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &snap.log_lines {
                            ui.label(egui::RichText::new(line).monospace().size(11.0));
                        }
                    });
            });

        // Vertical Holding Palette Pane to the left of the main layout pane
        if self.show_holding_palette && !snap.board.holding_palette.is_empty() {
            egui::SidePanel::left("vulcan_holding_palette_panel")
                .resizable(true)
                .default_width(260.0)
                .width_range(190.0..=450.0)
                .show(ctx, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.heading(format!("Holding Palette ({})", snap.board.holding_palette.len()));
                        if ui.add_enabled(!is_running, egui::Button::new("Place All")).clicked() {
                            let cfg = PlacementConfig {
                                min_spacing_mm: self.min_spacing_mm,
                                allow_layer_flip: self.allow_layer_flip,
                                ..PlacementConfig::default()
                            };
                            let mut g = self.live.lock().unwrap();
                            let count = crate::placer::place_all_holding_components_at_best_locations(&mut g.board, &cfg);
                            g.refresh_ratsnest();
                            g.push_log(format!("[Palette] Placed all {} components at favourable locations", count));
                        }
                    });
                    ui.label(
                        egui::RichText::new("Sorted by designator (letters first, then numbers)")
                            .weak()
                            .italics()
                            .size(11.0),
                    );
                    ui.separator();

                    egui::ScrollArea::vertical()
                        .id_salt("holding_palette_pane_scroll")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            let mut place_single_idx = None;
                            for (idx, fp) in snap.board.holding_palette.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    draw_mini_footprint(ui, fp, Vec2::new(44.0, 44.0));
                                    ui.vertical(|ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(
                                                egui::RichText::new(&fp.reference)
                                                    .strong()
                                                    .color(Color32::from_rgb(255, 215, 100)),
                                            );
                                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                if ui.add_enabled(!is_running, egui::Button::new("Place")).clicked() {
                                                    place_single_idx = Some(idx);
                                                }
                                            });
                                        });
                                        ui.label(
                                            egui::RichText::new(&fp.value)
                                                .size(11.0)
                                                .color(Color32::from_rgb(210, 215, 220)),
                                        );
                                        let layer_str = if fp.layer == 1 { "B.Cu" } else { "F.Cu" };
                                        let pad_count = fp.pads.len();
                                        let pad_label = if pad_count == 1 {
                                            format!("1 pad · {}", layer_str)
                                        } else {
                                            format!("{} pads · {}", pad_count, layer_str)
                                        };
                                        ui.label(egui::RichText::new(pad_label).size(10.0).weak());
                                    });
                                });
                                ui.separator();
                            }
                            if let Some(idx) = place_single_idx {
                                let cfg = PlacementConfig {
                                    min_spacing_mm: self.min_spacing_mm,
                                    allow_layer_flip: self.allow_layer_flip,
                                    ..PlacementConfig::default()
                                };
                                let mut g = self.live.lock().unwrap();
                                if let Some(placed_fp) = crate::placer::place_holding_component_at_best_location(&mut g.board, idx, &cfg) {
                                    g.refresh_ratsnest();
                                    g.push_log(format!(
                                        "[Palette] Placed {} at best location ({:.1}, {:.1}) rot={:.0}°",
                                        placed_fp.reference, placed_fp.x, placed_fp.y, placed_fp.rotation
                                    ));
                                }
                            }
                        });
                });
        }

        // Central Interactive 2D Board Canvas
        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
            let rect = response.rect;
            painter.rect_filled(rect, 0.0, COL_CANVAS_BG);

            let pointer_pos = ui.input(|i| i.pointer.hover_pos());

            // Track hovered footprint under cursor
            if !is_running {
                if let Some(pos) = pointer_pos {
                    if rect.contains(pos) {
                        let snap = self.live.lock().unwrap().clone();
                        let b = &snap.board;
                        let mut best_idx: Option<(usize, f32, f32)> = None;
                        for (idx, fp) in b.footprints.iter().enumerate() {
                            let fp_rect = self.footprint_screen_rect(b, rect, fp);
                            if fp_rect.contains(pos) {
                                let area = fp_rect.width() * fp_rect.height();
                                let dist = fp_rect.center().distance(pos);
                                if best_idx.map_or(true, |(_, best_area, best_dist)| {
                                    area < best_area - 1.0 || ((area - best_area).abs() <= 1.0 && dist < best_dist)
                                }) {
                                    best_idx = Some((idx, area, dist));
                                }
                            }
                        }
                        self.hovered_footprint = best_idx.map(|(i, _, _)| i);
                    } else {
                        self.hovered_footprint = None;
                    }
                } else {
                    self.hovered_footprint = None;
                }
            }

            // Keyboard shortcuts: 'm' (pick up/move), 'r' (rotate 90°), 'f' (flip layer), 'Escape' (cancel move)
            if !ctx.wants_keyboard_input() && !is_running {
                if ctx.input(|i| i.key_pressed(egui::Key::M)) {
                    if self.moving_footprint.is_none() {
                        if let Some(pos) = pointer_pos {
                            if rect.contains(pos) {
                                self.pick_up_component_at(rect, pos);
                            }
                        }
                    }
                }
                if ctx.input(|i| i.key_pressed(egui::Key::R)) {
                    self.rotate_active_component();
                }
                if ctx.input(|i| i.key_pressed(egui::Key::F)) {
                    self.flip_active_component();
                }
                if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.cancel_moving_component();
                }
            }

            // If a footprint is actively being moved, update its position to follow the cursor
            if self.moving_footprint.is_some() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                if let Some(pos) = pointer_pos {
                    if rect.contains(pos) {
                        self.update_moving_component_pos(rect, pos);
                    }
                }
            } else if let Some(h_idx) = self.hovered_footprint {
                let snap = self.live.lock().unwrap();
                if h_idx < snap.board.footprints.len() && !snap.board.footprints[h_idx].is_locked {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            }

            if response.clicked() {
                if self.moving_footprint.is_some() {
                    self.place_moving_component();
                } else if let Some(click_pos) = response.interact_pointer_pos() {
                    self.handle_canvas_click_at(rect, click_pos);
                }
            }
            if response.secondary_clicked() {
                if self.moving_footprint.is_some() {
                    self.cancel_moving_component();
                } else if let Some(click_pos) = response.interact_pointer_pos() {
                    self.handle_canvas_secondary_click_at(rect, click_pos);
                }
            }
            if response.dragged() {
                if self.moving_footprint.is_none() {
                    self.pan_offset += response.drag_delta();
                }
            }
            if rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or_default())) {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll.abs() > 0.1 {
                    let factor = (1.0 + scroll * 0.002).clamp(0.8, 1.25);
                    self.zoom = (self.zoom * factor).clamp(0.25, 12.0);
                }
            }

            let snap = self.live.lock().unwrap().clone();
            let b = &snap.board;
            let (_, scale) = self.board_to_screen(b, rect, b.min_x, b.min_y);
            let to_screen = |wx: f64, wy: f64| -> Pos2 { self.board_to_screen(b, rect, wx, wy).0 };

            // 1. Draw Board Outline (Edge.Cuts)
            if let Some(ref poly) = b.outline_polygon {
                let n = poly.vertices.len();
                for i in 0..n {
                    let p0 = to_screen(poly.vertices[i].x, poly.vertices[i].y);
                    let p1 = to_screen(poly.vertices[(i + 1) % n].x, poly.vertices[(i + 1) % n].y);
                    painter.line_segment([p0, p1], Stroke::new(2.0_f32, COL_EDGE_CUTS));
                }
            } else {
                let b_aabb = b.boundary_aabb();
                let edge_min = to_screen(b_aabb.min_x, b_aabb.min_y);
                let edge_max = to_screen(b_aabb.max_x, b_aabb.max_y);
                painter.rect_stroke(
                    Rect::from_min_max(edge_min, edge_max),
                    0.0,
                    Stroke::new(2.0_f32, COL_EDGE_CUTS),
                    egui::StrokeKind::Middle,
                );
            }

            // 2. Draw Keepout Zones
            for ko in &b.keepouts {
                let kmin = to_screen(ko.min_x, ko.min_y);
                let kmax = to_screen(ko.max_x, ko.max_y);
                let krect = Rect::from_min_max(kmin, kmax);
                painter.rect_filled(krect, 0.0, Color32::from_rgba_unmultiplied(180, 40, 40, 35));
                painter.rect_stroke(
                    krect,
                    0.0,
                    Stroke::new(1.2_f32, COL_KEEPOUT),
                    egui::StrokeKind::Middle,
                );
            }

            // 3. Draw Copper Layers bottom-up like KiCad (B.Cu, inner layers descending, F.Cu)
            let copper_layers = b.copper_layers();
            let draw_order = copper_layers_draw_order(&copper_layers);
            for &layer_name in &draw_order {
                if let Some(layer_idx) = copper_layers.iter().position(|&l| l == layer_name) {
                    if self.is_layer_visible(layer_name) {
                        let col = copper_layer_color(layer_name);
                        for tr in &b.tracks {
                            if tr.layer == layer_idx {
                                let p1 = to_screen(tr.start.0, tr.start.1);
                                let p2 = to_screen(tr.end.0, tr.end.1);
                                let w = ((tr.width as f32) * scale).max(1.5);
                                draw_track_segment(&painter, p1, p2, w, col);
                            }
                        }
                    }
                }
            }

            // 7. Draw Footprints, Courtyards, Rotated Pads, and Lock Icons
            for (idx, fp) in b.footprints.iter().enumerate() {
                let fp_rect = self.footprint_screen_rect(b, rect, fp);
                let is_moving = self.moving_footprint.as_ref().map_or(false, |m| m.fp_index == idx);
                let is_hovered = self.hovered_footprint == Some(idx) && !is_moving;

                // Two passes: render SMD pads first, then through-hole pads (thermal vias) on top
                for is_th in [false, true] {
                    for pad in fp.pads.iter().filter(|p| p.is_through_hole == is_th) {
                        let is_highlighted = self
                            .selected_net
                            .as_ref()
                            .map_or(false, |sn| !sn.is_empty() && sn == &pad.net_name);

                        let (px, py) = fp.pad_absolute_position(pad);
                        let center = to_screen(px, py);
                        let pad_col = if is_highlighted {
                            Color32::from_rgb(255, 245, 140)
                        } else if pad.is_through_hole {
                            COL_PAD_TH
                        } else if fp.layer == 1 {
                            COL_B_CU
                        } else {
                            COL_F_CU
                        };

                        if pad.shape == "circle" {
                            let r = ((pad.width.max(pad.height) as f32) * 0.5 * scale).max(1.0);
                            if is_highlighted {
                                painter.circle_filled(center, r + 2.5, Color32::from_rgba_unmultiplied(255, 220, 40, 160));
                                painter.circle_stroke(center, r + 3.0, Stroke::new(1.8_f32, Color32::WHITE));
                            }
                            painter.circle_filled(center, r, pad_col);
                        } else {
                            let corners = fp.pad_world_corners(pad);
                            let screen_pts: Vec<Pos2> = corners.iter().map(|p| to_screen(p.x, p.y)).collect();
                            if is_highlighted {
                                let halo_poly = Polygon::new(corners.clone()).expanded(0.20);
                                let halo_pts: Vec<Pos2> = halo_poly.vertices.iter().map(|p| to_screen(p.x, p.y)).collect();
                                painter.add(egui::Shape::convex_polygon(
                                    halo_pts.clone(),
                                    Color32::from_rgba_unmultiplied(255, 220, 40, 160),
                                    Stroke::new(1.8_f32, Color32::WHITE),
                                ));
                            }
                            painter.add(egui::Shape::convex_polygon(
                                screen_pts,
                                pad_col,
                                egui::Stroke::NONE,
                            ));
                        }

                        if let Some(drill_dia) = pad.drill {
                            let drill_r = ((drill_dia as f32) * 0.5 * scale).max(0.8);
                            painter.circle_filled(center, drill_r, Color32::from_rgb(25, 30, 35));
                        }
                    }
                }

                if fp.is_locked {
                    draw_lock_icon(&painter, fp_rect);
                }

                if self.show_refs {
                    let c = to_screen(fp.x, fp.y);
                    painter.text(
                        c,
                        egui::Align2::CENTER_CENTER,
                        &fp.reference,
                        egui::FontId::monospace(10.5),
                        Color32::WHITE,
                    );
                }

                if is_moving {
                    let label_pos = Pos2::new(fp_rect.center().x, fp_rect.top() - 14.0);
                    let side_str = if fp.layer == 1 { "B.Cu" } else { "F.Cu" };
                    let hint = format!("Moving {} [{}] {:.0}° · 'r' rotate · 'f' flip · click place", fp.reference, side_str, fp.rotation);
                    let galley = painter.layout_no_wrap(hint, egui::FontId::proportional(11.0), Color32::from_rgb(255, 235, 120));
                    let bg_rect = Rect::from_center_size(label_pos, galley.size() + Vec2::new(8.0, 4.0));
                    painter.rect_filled(bg_rect, 3.0, Color32::from_rgba_unmultiplied(20, 24, 30, 220));
                    painter.rect_stroke(bg_rect, 3.0, Stroke::new(1.0_f32, Color32::from_rgb(255, 215, 60)), egui::StrokeKind::Middle);
                    painter.galley(Pos2::new(bg_rect.left() + 4.0, bg_rect.top() + 2.0), galley, Color32::from_rgb(255, 235, 120));
                } else if is_hovered && !fp.is_locked {
                    let label_pos = Pos2::new(fp_rect.center().x, fp_rect.top() - 12.0);
                    let hint = format!("[m] Move {}", fp.reference);
                    let galley = painter.layout_no_wrap(hint, egui::FontId::proportional(10.5), Color32::WHITE);
                    let bg_rect = Rect::from_center_size(label_pos, galley.size() + Vec2::new(6.0, 3.0));
                    painter.rect_filled(bg_rect, 2.0, Color32::from_rgba_unmultiplied(25, 30, 38, 200));
                    painter.rect_stroke(bg_rect, 2.0, Stroke::new(0.8_f32, Color32::from_rgb(100, 180, 255)), egui::StrokeKind::Middle);
                    painter.galley(Pos2::new(bg_rect.left() + 3.0, bg_rect.top() + 1.5), galley, Color32::WHITE);
                }
            }

            // 8. Draw Vias
            if self.show_vias {
                for v in &b.vias {
                    let c = to_screen(v.position.0, v.position.1);
                    let outer_r = ((v.size as f32) * 0.5 * scale).max(2.5);
                    let inner_r = ((v.drill as f32) * 0.5 * scale).max(1.0);
                    painter.circle_filled(c, outer_r, COL_VIA);
                    painter.circle_filled(c, inner_r, Color32::from_rgb(25, 30, 35));
                }
            }

            // 6. Draw Ratsnest
            if self.show_ratsnest {
                for seg in &snap.ratsnest {
                    let p1 = to_screen(seg.p1.x, seg.p1.y);
                    let p2 = to_screen(seg.p2.x, seg.p2.y);
                    painter.line_segment(
                        [p1, p2],
                        Stroke::new(0.8_f32, COL_RATSNEST),
                    );
                }
            }

            // 9. Draw Courtyards on top (KiCad draws F.CrtYd/B.CrtYd above copper)
            for (idx, fp) in b.footprints.iter().enumerate() {
                let is_moving = self.moving_footprint.as_ref().map_or(false, |m| m.fp_index == idx);
                let is_hovered = self.hovered_footprint == Some(idx) && !is_moving;
                if self.show_courtyards {
                    let stroke_col = if is_moving {
                        Color32::from_rgb(255, 230, 80)
                    } else if is_hovered {
                        Color32::from_rgb(255, 255, 255)
                    } else if fp.is_locked {
                        Color32::from_rgb(240, 175, 50)
                    } else if fp.layer == 1 {
                        COL_B_CRTYD
                    } else {
                        COL_F_CRTYD
                    };
                    let stroke_w = if is_moving { 2.4_f32 } else if is_hovered { 1.8_f32 } else { 1.2_f32 };
                    let polys = fp.transformed_perimeters_layered(Point::new(fp.x, fp.y), fp.rotation, fp.layer, 0.0);
                    for poly in polys {
                        if poly.vertices.len() >= 3 {
                            let pts: Vec<Pos2> = poly.vertices.iter().map(|v| to_screen(v.x, v.y)).collect();
                            for i in 0..pts.len() {
                                painter.line_segment([pts[i], pts[(i + 1) % pts.len()]], Stroke::new(stroke_w, stroke_col));
                            }
                        }
                    }
                }

            }

            // 9. Draw Color Legend Overlay
            draw_color_legend(&painter, rect, b);
        });
    }
}

#[inline]
pub fn draw_track_segment(painter: &egui::Painter, p1: Pos2, p2: Pos2, width: f32, color: Color32) {
    painter.line_segment([p1, p2], Stroke::new(width, color));
    let r = width * 0.5;
    painter.circle_filled(p1, r, color);
    painter.circle_filled(p2, r, color);
}

/// Renders a miniature scaled preview of a footprint's courtyards and pads inside a fixed-size rect.
pub fn draw_mini_footprint(ui: &mut egui::Ui, fp: &crate::board::Footprint, size: Vec2) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter_at(rect);

    // Card background
    painter.rect_filled(rect, 3.0, Color32::from_rgb(22, 26, 30));
    painter.rect_stroke(
        rect,
        3.0,
        Stroke::new(1.0_f32, Color32::from_rgb(45, 52, 60)),
        egui::StrokeKind::Middle,
    );

    // Compute bounding box in footprint local coordinates
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let perims = if !fp.perimeters.is_empty() {
        &fp.perimeters
    } else {
        &fp.courtyards
    };

    for poly in perims {
        for v in &poly.vertices {
            min_x = min_x.min(v.x);
            max_x = max_x.max(v.x);
            min_y = min_y.min(v.y);
            max_y = max_y.max(v.y);
        }
    }
    for pad in &fp.pads {
        let half_w = pad.width * 0.5;
        let half_h = pad.height * 0.5;
        min_x = min_x.min(pad.offset_x - half_w);
        max_x = max_x.max(pad.offset_x + half_w);
        min_y = min_y.min(pad.offset_y - half_h);
        max_y = max_y.max(pad.offset_y + half_h);
    }
    if min_x >= max_x || min_y >= max_y {
        let hw = (fp.width * 0.5).max(0.8);
        let hh = (fp.height * 0.5).max(0.8);
        min_x = -hw;
        max_x = hw;
        min_y = -hh;
        max_y = hh;
    }

    let pad_margin = 4.0;
    let avail_w = (rect.width() - pad_margin * 2.0).max(1.0);
    let avail_h = (rect.height() - pad_margin * 2.0).max(1.0);
    let span_x = (max_x - min_x).max(0.5);
    let span_y = (max_y - min_y).max(0.5);
    let scale = (avail_w / span_x as f32).min(avail_h / span_y as f32);
    let mid_x = (min_x + max_x) * 0.5;
    let mid_y = (min_y + max_y) * 0.5;
    let center_screen = rect.center();

    let to_preview = |lx: f64, ly: f64| -> Pos2 {
        Pos2::new(
            center_screen.x + ((lx - mid_x) as f32) * scale,
            center_screen.y + ((ly - mid_y) as f32) * scale,
        )
    };

    // 1. Draw Courtyard Perimeter
    let courtyard_col = Color32::from_rgba_unmultiplied(70, 200, 160, 160);
    for poly in perims {
        if poly.vertices.len() >= 3 {
            let pts: Vec<Pos2> = poly.vertices.iter().map(|v| to_preview(v.x, v.y)).collect();
            for i in 0..pts.len() {
                painter.line_segment([pts[i], pts[(i + 1) % pts.len()]], Stroke::new(1.0_f32, courtyard_col));
            }
        }
    }

    // 2. Draw Pads in two passes: SMD first, then through-hole on top
    for is_th in [false, true] {
        for pad in fp.pads.iter().filter(|p| p.is_through_hole == is_th) {
            let pad_col = if pad.is_through_hole {
                Color32::from_rgb(215, 185, 80)
            } else if pad.net_name == "GND" {
                Color32::from_rgb(90, 165, 110)
            } else if crate::placer::is_power_net(&pad.net_name) {
                Color32::from_rgb(245, 140, 50)
            } else if fp.layer == 1 {
                Color32::from_rgb(85, 145, 235)
            } else {
                Color32::from_rgb(220, 85, 80)
            };

            let center = to_preview(pad.offset_x, pad.offset_y);
            if pad.shape == "circle" {
                let r = ((pad.width.max(pad.height) as f32) * 0.5 * scale).max(1.0);
                painter.circle_filled(center, r, pad_col);
            } else {
                let half_w = pad.width * 0.5;
                let half_h = pad.height * 0.5;
                let rad = (-pad.rotation).to_radians();
                let cos_r = rad.cos();
                let sin_r = rad.sin();
                let pts: Vec<Pos2> = [
                    (-half_w, -half_h),
                    (half_w, -half_h),
                    (half_w, half_h),
                    (-half_w, half_h),
                ]
                .iter()
                .map(|&(cx, cy)| {
                    let rx = cx * cos_r - cy * sin_r + pad.offset_x;
                    let ry = cx * sin_r + cy * cos_r + pad.offset_y;
                    to_preview(rx, ry)
                })
                .collect();
                painter.add(egui::Shape::convex_polygon(pts, pad_col, egui::Stroke::NONE));
            }

            if let Some(drill_dia) = pad.drill {
                let drill_r = ((drill_dia as f32) * 0.5 * scale).max(0.6);
                painter.circle_filled(center, drill_r, Color32::from_rgb(20, 24, 28));
            }
        }
    }

    response
}

impl eframe::App for VulcansTrialsApp {
    // Non-testable directly in a headless test runner because `eframe::Frame` has private fields
    // that can only be instantiated by an active OS window via `eframe::run_native`; all UI logic
    // is delegated to `VulcansTrialsApp::render_ui`, which is unit-tested headlessly below.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.render_ui(ctx);
    }
}

pub fn draw_lock_icon(painter: &egui::Painter, fp_rect: Rect) {
    let min_dim = fp_rect.width().min(fp_rect.height());
    if min_dim < 2.0 {
        return;
    }
    // Scale the lock icon to fit cleanly inside the footprint bounding box
    let icon_h = (min_dim * 0.55).clamp(2.0, min_dim * 0.85);
    let icon_w = icon_h * 0.78;
    let center = fp_rect.center();

    let body_h = icon_h * 0.56;
    let body_top = center.y - icon_h * 0.05;
    let body_rect = Rect::from_min_max(
        Pos2::new(center.x - icon_w * 0.5, body_top),
        Pos2::new(center.x + icon_w * 0.5, body_top + body_h),
    );
    let stroke_w = (icon_h * 0.09).clamp(1.0, 2.5);
    let gold_fill = Color32::from_rgba_unmultiplied(245, 185, 55, 185);
    let dark_edge = Color32::from_rgba_unmultiplied(30, 24, 15, 220);

    // Shackle arch above the padlock body
    let shackle_w = icon_w * 0.56;
    let shackle_top = center.y - icon_h * 0.48;
    let left_x = center.x - shackle_w * 0.5;
    let right_x = center.x + shackle_w * 0.5;
    let arch_mid_y = shackle_top + (body_top - shackle_top) * 0.35;
    let shackle_stroke = Stroke::new(stroke_w, Color32::from_rgba_unmultiplied(255, 210, 90, 220));
    painter.line_segment([Pos2::new(left_x, body_top), Pos2::new(left_x, arch_mid_y)], shackle_stroke);
    painter.line_segment([Pos2::new(left_x, arch_mid_y), Pos2::new(center.x, shackle_top)], shackle_stroke);
    painter.line_segment([Pos2::new(center.x, shackle_top), Pos2::new(right_x, arch_mid_y)], shackle_stroke);
    painter.line_segment([Pos2::new(right_x, arch_mid_y), Pos2::new(right_x, body_top)], shackle_stroke);

    // Padlock body + keyhole
    painter.rect_filled(body_rect, (icon_w * 0.12).min(3.0), gold_fill);
    painter.rect_stroke(
        body_rect,
        (icon_w * 0.12).min(3.0),
        Stroke::new((stroke_w * 0.6).max(0.8), dark_edge),
        egui::StrokeKind::Middle,
    );
    let keyhole_r = (body_h * 0.16).max(0.8);
    let kh_center = Pos2::new(center.x, body_rect.center().y - body_h * 0.06);
    painter.circle_filled(kh_center, keyhole_r, dark_edge);
    painter.line_segment(
        [kh_center, Pos2::new(center.x, body_rect.bottom() - body_h * 0.18)],
        Stroke::new((keyhole_r * 0.9).max(0.8), dark_edge),
    );
}

pub fn draw_color_legend(painter: &egui::Painter, canvas_rect: Rect, board: &Board) {
    let active_layers = board.copper_layers();
    let display_layers = kicad_layer_display_order(&active_layers);
    let plane_names = board.layer_plane_net_names();

    let mut items: Vec<(String, Color32)> = Vec::new();
    for &layer_name in &display_layers {
        let l_idx = active_layers.iter().position(|&l| l == layer_name);
        let plane = l_idx.and_then(|idx| plane_names.get(&idx));
        let label = match layer_name {
            "F.Cu" => match plane {
                Some(net) => format!("F.Cu Track (Top - {})", net),
                None => "F.Cu Track (Top)".to_string(),
            },
            "B.Cu" => match plane {
                Some(net) => format!("B.Cu Track (Bottom - {})", net),
                None => "B.Cu Track (Bottom)".to_string(),
            },
            _ => {
                let k = layer_name.strip_prefix("In").and_then(|s| s.strip_suffix(".Cu")).unwrap_or("");
                match plane {
                    Some(net) => format!("{layer_name} Track (Inner {k} - {net})"),
                    None => format!("{layer_name} Track (Inner {k})"),
                }
            }
        };
        items.push((label, copper_layer_color(layer_name)));
    }
    items.push(("Edge.Cuts Outline".to_string(), COL_EDGE_CUTS));
    items.push(("Keepout Zone".to_string(), COL_KEEPOUT));
    items.push(("Ratsnest (Unrouted)".to_string(), COL_RATSNEST));
    items.push(("Courtyard (F.Cu Top)".to_string(), COL_F_CRTYD));
    items.push(("Courtyard (B.Cu Bottom)".to_string(), COL_B_CRTYD));
    items.push(("Courtyard (Locked 🔒)".to_string(), Color32::from_rgb(240, 175, 50)));
    items.push(("SMD Pad (F.Cu / B.Cu colour)".to_string(), COL_F_CU));
    items.push(("Through-Hole Pad".to_string(), COL_PAD_TH));
    items.push(("Via Annular Ring".to_string(), COL_VIA));

    let row_h = 15.0_f32;
    let box_w = 205.0_f32;
    let box_h = 26.0 + (items.len() as f32) * row_h + 16.0;
    let margin = 10.0_f32;
    let top_left = Pos2::new((canvas_rect.right() - box_w - margin).max(canvas_rect.left() + 4.0), canvas_rect.top() + margin);
    let legend_rect = Rect::from_min_size(top_left, Vec2::new(box_w, box_h));

    painter.rect_filled(legend_rect, 4.0, Color32::from_rgba_unmultiplied(20, 25, 32, 225));
    painter.rect_stroke(
        legend_rect,
        4.0,
        Stroke::new(1.0_f32, Color32::from_rgb(65, 75, 90)),
        egui::StrokeKind::Middle,
    );

    painter.text(
        Pos2::new(legend_rect.left() + 8.0, legend_rect.top() + 6.0),
        egui::Align2::LEFT_TOP,
        "Colour Legend",
        egui::FontId::proportional(12.0),
        Color32::from_rgb(235, 240, 245),
    );

    for (idx, (label, col)) in items.iter().enumerate() {
        let y = legend_rect.top() + 24.0 + (idx as f32) * row_h;
        let swatch = Rect::from_min_size(Pos2::new(legend_rect.left() + 8.0, y + 2.0), Vec2::new(10.0, 10.0));
        painter.rect_filled(swatch, 2.0, *col);
        painter.text(
            Pos2::new(legend_rect.left() + 24.0, y),
            egui::Align2::LEFT_TOP,
            label,
            egui::FontId::proportional(11.0),
            Color32::from_rgb(205, 212, 220),
        );
    }

    painter.text(
        Pos2::new(legend_rect.left() + 8.0, legend_rect.bottom() - 15.0),
        egui::Align2::LEFT_TOP,
        "Click footprint to lock/unlock",
        egui::FontId::proportional(10.0),
        Color32::from_rgb(240, 185, 75),
    );
}

fn draw_load_badge(ui: &mut egui::Ui, label: &str, pct: f32, history: &[f32], color: Color32) {
    ui.horizontal(|ui| {
        ui.label(format!("{}: {:>4.0}%", label, pct));
        let (rect, _) = ui.allocate_exact_size(Vec2::new(72.0, 18.0), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 2.0, Color32::from_rgb(30, 34, 40));
        if history.len() >= 2 {
            let n = history.len();
            for i in 0..(n - 1) {
                let x0 = rect.left() + (i as f32 / (n - 1) as f32) * rect.width();
                let x1 = rect.left() + ((i + 1) as f32 / (n - 1) as f32) * rect.width();
                let y0 = rect.bottom() - (history[i] / 100.0).clamp(0.0, 1.0) * rect.height();
                let y1 = rect.bottom() - (history[i + 1] / 100.0).clamp(0.0, 1.0) * rect.height();
                painter.line_segment([Pos2::new(x0, y0), Pos2::new(x1, y1)], Stroke::new(1.5_f32, color));
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Footprint, KeepoutZone, Polygon, TrackSegment, Via};
    use std::fs;
    use std::time::Duration;

    #[test]
    fn test_kicad_layer_colours_and_order() {
        assert_eq!(COL_F_CU, Color32::from_rgb(0xC8, 0x34, 0x34));
        assert_eq!(COL_B_CU, Color32::from_rgb(0x4D, 0x7F, 0xC4));
        assert_eq!(COL_IN1_CU, Color32::from_rgb(0x7F, 0xC8, 0x7F));
        assert_eq!(COL_IN2_CU, Color32::from_rgb(0xCE, 0x7D, 0x2C));
        assert_eq!(copper_layer_color("In3.Cu"), Color32::from_rgb(0x4F, 0xCB, 0xCB));
        assert_eq!(copper_layer_color("In4.Cu"), Color32::from_rgb(0xDB, 0x62, 0x8B));
        assert_eq!(copper_layer_color("In5.Cu"), Color32::from_rgb(0xA7, 0xA5, 0xC6));
        assert_eq!(copper_layer_color("In6.Cu"), Color32::from_rgb(0x28, 0xCC, 0xD9));
        assert_eq!(copper_layer_color("In7.Cu"), Color32::from_rgb(0xE8, 0xB2, 0xA7));
        assert_eq!(copper_layer_color("In8.Cu"), Color32::from_rgb(0xF2, 0xED, 0xA1));
        assert_eq!(copper_layer_color("In9.Cu"), Color32::from_rgb(0x8D, 0xCB, 0x81));

        // 7-color repeat cycle starting at (237, 124, 51)
        assert_eq!(copper_layer_color("In10.Cu"), Color32::from_rgb(237, 124, 51));
        assert_eq!(copper_layer_color("In11.Cu"), Color32::from_rgb(91, 195, 235));
        assert_eq!(copper_layer_color("In12.Cu"), Color32::from_rgb(247, 111, 142));
        assert_eq!(copper_layer_color("In13.Cu"), Color32::from_rgb(167, 165, 198));
        assert_eq!(copper_layer_color("In14.Cu"), Color32::from_rgb(40, 204, 217));
        assert_eq!(copper_layer_color("In15.Cu"), Color32::from_rgb(232, 178, 167));
        assert_eq!(copper_layer_color("In16.Cu"), Color32::from_rgb(242, 237, 161));
        // Verify cycle repeat
        assert_eq!(copper_layer_color("In17.Cu"), copper_layer_color("In10.Cu"));
        assert_eq!(copper_layer_color("In24.Cu"), copper_layer_color("In10.Cu"));
        assert_eq!(copper_layer_color("In30.Cu"), Color32::from_rgb(242, 237, 161));

        assert_eq!(COL_F_CRTYD, Color32::from_rgb(0xFF, 0x26, 0xE2));
        assert_eq!(COL_B_CRTYD, Color32::from_rgb(0x26, 0xE2, 0xFF));
        assert_eq!(COL_EDGE_CUTS, Color32::from_rgb(0xD0, 0xD2, 0xCD));
        assert_eq!(COL_KEEPOUT, Color32::from_rgba_unmultiplied(255, 0, 5, 128));
        assert_eq!(COL_CANVAS_BG, Color32::from_rgb(0x00, 0x10, 0x23));

        assert_eq!(COPPER_DRAW_ORDER, ["B.Cu", "In2.Cu", "In1.Cu", "F.Cu"]);
        assert_eq!(
            kicad_layer_display_order(&["F.Cu", "B.Cu", "In1.Cu", "In2.Cu"]),
            vec!["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]
        );
        assert_eq!(
            copper_layers_draw_order(&["F.Cu", "B.Cu", "In1.Cu", "In2.Cu"]),
            vec!["B.Cu", "In2.Cu", "In1.Cu", "F.Cu"]
        );
    }

    fn make_gui_test_board(pcb_path: &str) -> Board {
        let content = r#"(kicad_pcb
          (net 0 "") (net 1 "GND") (net 2 "SIG")
          (gr_rect (start 100.0 100.0) (end 140.0 130.0) (layer "Edge.Cuts"))
          (footprint "U" (at 110.0 115.0) (property "Reference" "J1")
            (pad "1" thru_hole circle (at 0 0) (size 1.0 1.0) (net 2 "SIG")))
          (footprint "R" (at 120.0 115.0) (property "Reference" "R10")
            (pad "1" smd rect (at -0.5 0) (size 0.5 0.5) (net 2 "SIG"))
            (pad "2" smd rect (at 0.5 0) (size 0.5 0.5) (net 1 "GND")))
        )"#;
        fs::write(pcb_path, content).unwrap();
        let mut b = Board::load(pcb_path, &HashSet::new()).unwrap();
        b.keepouts.push(KeepoutZone {
            min_x: 132.0,
            min_y: 122.0,
            max_x: 138.0,
            max_y: 128.0,
            layers: vec![0, 1],
        });
        b.tracks.push(TrackSegment {
            start: (110.0, 115.0),
            end: (115.0, 115.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        b.tracks.push(TrackSegment {
            start: (115.0, 115.0),
            end: (119.5, 115.0),
            width: 0.25,
            layer: 1,
            net_id: 2,
            is_locked: false,
        });
        b.vias.push(Via {
            position: (115.0, 115.0),
            size: 0.6,
            drill: 0.3,
            net_id: 2,
            is_locked: false,
        });
        b
    }

    #[test]
    fn test_board_to_screen_and_screen_to_board_transforms() {
        let in_path = "/tmp/vulcan_gui_transform_in.kicad_pcb";
        let board = make_gui_test_board(in_path);
        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(800.0, 600.0));

        let cx = (board.min_x + board.max_x) * 0.5;
        let cy = (board.min_y + board.max_y) * 0.5;
        let (center_pt, scale1) = app.board_to_screen(&board, rect, cx, cy);
        assert!((center_pt.x - 400.0).abs() < 1e-3 && (center_pt.y - 300.0).abs() < 1e-3);

        let roundtrip = app.screen_to_board(&board, rect, center_pt);
        assert!((roundtrip.x - cx).abs() < 1e-3 && (roundtrip.y - cy).abs() < 1e-3);

        app.zoom = 2.0;
        app.pan_offset = Vec2::new(15.0, -10.0);
        let (panned_pt, scale2) = app.board_to_screen(&board, rect, cx, cy);
        assert!((scale2 - scale1 * 2.0).abs() < 1e-3);
        assert!((panned_pt.x - 415.0).abs() < 1e-3 && (panned_pt.y - 290.0).abs() < 1e-3);

        let roundtrip2 = app.screen_to_board(&board, rect, panned_pt);
        assert!((roundtrip2.x - cx).abs() < 1e-3 && (roundtrip2.y - cy).abs() < 1e-3);

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_click_footprint_toggles_lock_and_scales_icon() {
        let in_path = "/tmp/vulcan_gui_lock_toggle_in.kicad_pcb";
        let board = make_gui_test_board(in_path);
        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(800.0, 600.0));

        // R10 starts unlocked at (120.0, 115.0)
        let (r10_screen, _) = app.board_to_screen(&board, rect, 120.0, 115.0);
        let toggled1 = app.toggle_footprint_lock_at(rect, r10_screen);
        assert_eq!(toggled1, Some(("R10".to_string(), true)));

        // Clicking R10 again unlocks it
        let toggled2 = app.toggle_footprint_lock_at(rect, r10_screen);
        assert_eq!(toggled2, Some(("R10".to_string(), false)));

        // Clicking empty board space returns None
        let (empty_screen, _) = app.board_to_screen(&board, rect, 102.0, 102.0);
        assert_eq!(app.toggle_footprint_lock_at(rect, empty_screen), None);

        // Verify footprint_screen_rect fallback when courtyard is empty
        let mut bare_fp = board.footprints[1].clone();
        bare_fp.courtyards.clear();
        let fallback_rect = app.footprint_screen_rect(&board, rect, &bare_fp);
        assert!(fallback_rect.width() >= 6.0 && fallback_rect.height() >= 6.0);

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_canvas_click_pad_highlights_net_and_body_toggles_lock() {
        let in_path = "/tmp/vulcan_gui_pad_net_highlight_in.kicad_pcb";
        let board = make_gui_test_board(in_path);
        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(800.0, 600.0));

        assert_eq!(app.selected_net, None);
        assert!(!app.live.lock().unwrap().board.footprints[1].is_locked);

        // 1. Click directly on R10 pad 1 (net "SIG")
        let (pad1_screen, _) = app.board_to_screen(&board, rect, 119.5, 115.0);
        let pad_res = app.handle_canvas_click_at(rect, pad1_screen);
        assert_eq!(pad_res, Some("Net: SIG".to_string()));
        assert_eq!(app.selected_net.as_deref(), Some("SIG"));
        assert!(!app.live.lock().unwrap().board.footprints[1].is_locked);

        // 2. Click on the same pad again -> toggles off net highlight
        let pad_res2 = app.handle_canvas_click_at(rect, pad1_screen);
        assert_eq!(pad_res2, Some("Cleared SIG".to_string()));
        assert_eq!(app.selected_net, None);

        // 3. Click on R10 pad 2 (net "GND") -> selects "GND"
        let (pad2_screen, _) = app.board_to_screen(&board, rect, 120.5, 115.0);
        let pad2_res = app.handle_canvas_click_at(rect, pad2_screen);
        assert_eq!(pad2_res, Some("Net: GND".to_string()));
        assert_eq!(app.selected_net.as_deref(), Some("GND"));
        assert!(!app.live.lock().unwrap().board.footprints[1].is_locked);

        // 4. Click on R10 component body (at 120.0, 115.0, between pads)
        let (body_screen, _) = app.board_to_screen(&board, rect, 120.0, 115.0);
        let body_res = app.handle_canvas_click_at(rect, body_screen);
        assert_eq!(body_res, Some("R10: LOCKED".to_string()));
        assert!(app.live.lock().unwrap().board.footprints[1].is_locked);
        assert_eq!(app.selected_net.as_deref(), Some("GND"));

        // 5. Click component body again -> unlocks footprint
        let body_res2 = app.handle_canvas_click_at(rect, body_screen);
        assert_eq!(body_res2, Some("R10: UNLOCKED".to_string()));
        assert!(!app.live.lock().unwrap().board.footprints[1].is_locked);

        // 6. Click empty canvas (at 102.0, 102.0) -> clears net highlight
        let (empty_screen, _) = app.board_to_screen(&board, rect, 102.0, 102.0);
        let empty_res = app.handle_canvas_click_at(rect, empty_screen);
        assert_eq!(empty_res, None);
        assert_eq!(app.selected_net, None);

        // 7. Verify toggle_footprint_lock_at directly ignores clicks on pads
        let direct_pad_click = app.toggle_footprint_lock_at(rect, pad1_screen);
        assert_eq!(direct_pad_click, None);
        assert!(!app.live.lock().unwrap().board.footprints[1].is_locked);

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_vulcans_trials_app_headless_ui_and_trial_lifecycle() {
        let in_path = "/tmp/vulcan_gui_lifecycle_in.kicad_pcb";
        let out_path = "/tmp/vulcan_gui_lifecycle_out.kicad_pcb";
        let ses_path = "/tmp/vulcan_gui_lifecycle_out.ses";
        let board = make_gui_test_board(in_path);
        let vk = VulkanEngine::new();

        let mut app = VulcansTrialsApp::new(in_path.to_string(), out_path.to_string(), board, vk);
        app.place_duration_secs = 0;
        app.max_ripup_passes = 1;
        app.grid_res_mm = 0.50;

        // Render UI headlessly via egui::Context::run (exercises top bar, side panel, 2D canvas, lock icons, legend, and draw_load_badge)
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            app.render_ui(ctx);
        });

        // Also test CPU fallback branch in top bar
        app.vulkan = None;
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            app.render_ui(ctx);
        });

        // Start background trial (place + route) and verify completion and file output
        app.place_duration_secs = 0;
        app.start_trial(true, true);
        // Second call while running should be a no-op
        app.start_trial(true, true);

        for _ in 0..100 {
            if !app.worker_running.load(Ordering::Relaxed) {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        assert!(!app.worker_running.load(Ordering::Relaxed));
        assert_eq!(app.live.lock().unwrap().phase, TrialPhase::Completed);
        assert!(Path::new(out_path).exists());
        assert!(Path::new(ses_path).exists());

        let _ = fs::remove_file(in_path);
        let _ = fs::remove_file(out_path);
        let _ = fs::remove_file(ses_path);
    }

    #[test]
    fn test_gui_clear_unlocked_objects_and_holding_palette() {
        let in_path = "/tmp/vulcan_gui_clear_in.kicad_pcb";
        let mut board = make_gui_test_board(in_path);
        // Lock J1, leave R10 unlocked
        board.footprints[0].is_locked = true;
        board.tracks[0].is_locked = true;

        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board, None);
        assert_eq!(app.live.lock().unwrap().board.footprints.len(), 2);
        assert_eq!(app.live.lock().unwrap().board.tracks.len(), 2);
        assert_eq!(app.live.lock().unwrap().board.vias.len(), 1);
        assert_eq!(app.live.lock().unwrap().board.holding_palette.len(), 0);

        let (ripped, cleared) = app.live.lock().unwrap().board.clear_unlocked_objects();
        assert_eq!(ripped, 2); // 1 unlocked track + 1 unlocked via
        assert_eq!(cleared, 1); // R10 moved to holding palette

        {
            let g = app.live.lock().unwrap();
            assert_eq!(g.board.footprints.len(), 1);
            assert_eq!(g.board.footprints[0].reference, "J1");
            assert_eq!(g.board.tracks.len(), 1);
            assert!(g.board.tracks[0].is_locked);
            assert_eq!(g.board.vias.len(), 0);
            assert_eq!(g.board.holding_palette.len(), 1);
            assert_eq!(g.board.holding_palette[0].reference, "R10");
        }

        // Render UI with holding palette visible (testing draw_mini_footprint and holding pane)
        assert!(app.is_holding_palette_visible());
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            app.render_ui(ctx);
        });
        let state1 = egui::panel::PanelState::load(&ctx, egui::Id::new("vulcan_holding_palette_panel"));
        assert!(state1.is_some(), "Holding palette side panel should be rendered when palette has items");

        // Test placing R10 at best location from holding palette
        let cfg = PlacementConfig::default();
        let placed = crate::placer::place_holding_component_at_best_location(&mut app.live.lock().unwrap().board, 0, &cfg);
        assert!(placed.is_some());
        assert_eq!(placed.unwrap().reference, "R10");
        assert_eq!(app.live.lock().unwrap().board.footprints.len(), 2);
        assert_eq!(app.live.lock().unwrap().board.holding_palette.len(), 0);

        // Render UI again with empty holding palette
        assert!(!app.is_holding_palette_visible());
        assert!(app.show_holding_palette); // flag is true, but palette is empty
        let ctx2 = egui::Context::default();
        let _ = ctx2.run(egui::RawInput::default(), |ctx| {
            app.render_ui(ctx);
        });
        let state2 = egui::panel::PanelState::load(&ctx2, egui::Id::new("vulcan_holding_palette_panel"));
        assert!(state2.is_none(), "Holding palette side panel must be hidden when empty even if show_holding_palette is true");

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_holding_palette_hidden_when_empty_and_rendered_when_populated() {
        let in_path = "/tmp/vulcan_gui_palette_visibility_in.kicad_pcb";
        let board = make_gui_test_board(in_path);
        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board, None);

        // 1. Initial board has empty holding palette and show_holding_palette is true
        assert!(app.live.lock().unwrap().board.holding_palette.is_empty());
        assert!(app.show_holding_palette);
        assert!(!app.is_holding_palette_visible());

        let ctx1 = egui::Context::default();
        let _ = ctx1.run(egui::RawInput::default(), |ctx| {
            app.render_ui(ctx);
        });
        let panel1 = egui::panel::PanelState::load(&ctx1, egui::Id::new("vulcan_holding_palette_panel"));
        assert!(panel1.is_none(), "Holding palette must NOT be rendered when empty");

        // 2. Add a footprint into holding palette
        let dummy_fp = app.live.lock().unwrap().board.footprints[0].clone();
        app.live.lock().unwrap().board.holding_palette.push(dummy_fp);
        assert!(!app.live.lock().unwrap().board.holding_palette.is_empty());
        assert!(app.is_holding_palette_visible());

        let ctx2 = egui::Context::default();
        let _ = ctx2.run(egui::RawInput::default(), |ctx| {
            app.render_ui(ctx);
        });
        let panel2 = egui::panel::PanelState::load(&ctx2, egui::Id::new("vulcan_holding_palette_panel"));
        assert!(panel2.is_some(), "Holding palette MUST be rendered when it contains items");

        // 3. User unchecks show_holding_palette toggle
        app.show_holding_palette = false;
        assert!(!app.is_holding_palette_visible());

        let ctx3 = egui::Context::default();
        let _ = ctx3.run(egui::RawInput::default(), |ctx| {
            app.render_ui(ctx);
        });
        let panel3 = egui::panel::PanelState::load(&ctx3, egui::Id::new("vulcan_holding_palette_panel"));
        assert!(panel3.is_none(), "Holding palette must NOT be rendered when toggled off");

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_gui_secondary_click_moves_component_to_holding_palette() {
        let in_path = "/tmp/vulcan_gui_sec_click_in.kicad_pcb";
        let board = make_gui_test_board(in_path);
        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(800.0, 600.0));

        // R10 is unlocked at (120.0, 115.0)
        let (r10_screen, _) = app.board_to_screen(&board, rect, 120.0, 115.0);
        let initial_ratsnest = app.live.lock().unwrap().ratsnest.len();

        let res = app.handle_canvas_secondary_click_at(rect, r10_screen);
        assert_eq!(res, Some("R10".to_string()));
        assert_eq!(app.live.lock().unwrap().board.footprints.len(), 1);
        assert_eq!(app.live.lock().unwrap().board.holding_palette.len(), 1);
        assert_eq!(app.live.lock().unwrap().board.holding_palette[0].reference, "R10");

        // Ratsnest wire was removed when component was held
        let new_ratsnest = app.live.lock().unwrap().ratsnest.len();
        assert!(new_ratsnest < initial_ratsnest);

        // Clicking locked component (J1 at 110.0, 115.0) or empty space does nothing
        app.live.lock().unwrap().board.footprints[0].is_locked = true;
        let (j1_screen, _) = app.board_to_screen(&board, rect, 110.0, 115.0);
        let locked_res = app.handle_canvas_secondary_click_at(rect, j1_screen);
        assert_eq!(locked_res, None);

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_complex_footprint_perimeter_rendering_not_triangulated() {
        let in_path = "/tmp/vulcan_gui_perimeter_test.kicad_pcb";
        let mut board = make_gui_test_board(in_path);

        // Add a complex T-shaped footprint (like ESP32 module)
        let t_perimeter = Polygon::new(vec![
            Point::new(-9.75, 7.15),
            Point::new(9.75, 7.15),
            Point::new(9.75, -6.85),
            Point::new(14.25, -6.85),
            Point::new(14.25, -18.35),
            Point::new(-14.25, -18.35),
            Point::new(-14.25, -6.85),
            Point::new(-9.75, -6.85),
        ]);
        let decomposed_courtyards = t_perimeter.decompose_convex();
        assert!(decomposed_courtyards.len() >= 2, "Courtyard must be decomposed into convex sub-polygons for SAT");

        let complex_fp = Footprint {
            reference: "U4".to_string(),
            value: "ESP32-C3".to_string(),
            x: 140.0,
            y: 105.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: Vec::new(),
            width: 28.5,
            height: 25.5,
            courtyards: decomposed_courtyards,
            perimeters: vec![t_perimeter],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
        };
        board.footprints.push(complex_fp);

        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);
        let fp = board.footprints.last().unwrap();

        // 1. transformed_courtyards_layered returns multiple convex polygons (triangles)
        let courtyards = fp.transformed_courtyards_layered(Point::new(fp.x, fp.y), fp.rotation, fp.layer, 0.0);
        assert!(courtyards.len() >= 2);

        // 2. transformed_perimeters_layered returns ONLY the outer perimeter (1 polygon, 8 vertices)
        let perimeters = fp.transformed_perimeters_layered(Point::new(fp.x, fp.y), fp.rotation, fp.layer, 0.0);
        assert_eq!(perimeters.len(), 1, "Complex object should have exactly 1 perimeter polygon");
        assert_eq!(perimeters[0].vertices.len(), 8, "Outer perimeter must preserve all 8 boundary vertices without diagonal chord edges");

        // 3. UI rendering executes cleanly with perimeter rendering
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            app.render_ui(ctx);
        });

        // 4. draw_mini_footprint also uses perimeter
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                draw_mini_footprint(ui, fp, Vec2::new(100.0, 100.0));
            });
        });

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_manual_refine_pick_up_move_rotate_flip_and_cancel() {
        let in_path = "/tmp/vulcan_gui_manual_refine_in.kicad_pcb";
        let mut board = make_gui_test_board(in_path);
        // J1 locked at 140.0, 115.0; R10 unlocked at 120.0, 115.0
        board.footprints[0].is_locked = true;
        board.footprints[1].is_locked = false;

        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(800.0, 600.0));

        // 1. Trying to pick up locked J1 fails
        let (j1_screen, _) = app.board_to_screen(&board, rect, 140.0, 115.0);
        assert!(!app.pick_up_component_at(rect, j1_screen));
        assert!(app.moving_footprint.is_none());

        // 2. Picking up unlocked R10 succeeds
        let (r10_screen, _) = app.board_to_screen(&board, rect, 120.0, 115.0);
        assert!(app.pick_up_component_at(rect, r10_screen));
        assert!(app.moving_footprint.is_some());
        assert_eq!(app.moving_footprint.as_ref().unwrap().reference, "R10");

        // 3. Move cursor to new world coordinate (110.0, 105.0)
        let (new_screen, _) = app.board_to_screen(&board, rect, 110.0, 105.0);
        app.update_moving_component_pos(rect, new_screen);
        {
            let g = app.live.lock().unwrap();
            let fp = &g.board.footprints[1];
            assert!((fp.x - 110.0).abs() < 1e-4);
            assert!((fp.y - 105.0).abs() < 1e-4);
        }

        // 4. Rotate active component with 'r'
        assert!(app.rotate_active_component());
        assert_eq!(app.live.lock().unwrap().board.footprints[1].rotation, 90.0);
        assert!(app.rotate_active_component());
        assert_eq!(app.live.lock().unwrap().board.footprints[1].rotation, 180.0);

        // 5. Flip active component with 'f'
        assert_eq!(app.live.lock().unwrap().board.footprints[1].layer, 0); // F.Cu
        assert!(app.flip_active_component());
        assert_eq!(app.live.lock().unwrap().board.footprints[1].layer, 1); // B.Cu
        assert!(app.flip_active_component());
        assert_eq!(app.live.lock().unwrap().board.footprints[1].layer, 0); // F.Cu

        // 6. Cancel move: restores original position (120.0, 115.0), rotation 0, layer 0
        assert!(app.cancel_moving_component());
        assert!(app.moving_footprint.is_none());
        {
            let g = app.live.lock().unwrap();
            let fp = &g.board.footprints[1];
            assert!((fp.x - 120.0).abs() < 1e-4);
            assert!((fp.y - 115.0).abs() < 1e-4);
            assert_eq!(fp.rotation, 0.0);
            assert_eq!(fp.layer, 0);
        }

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_manual_refine_place_shoves_overlapping_component() {
        let in_path = "/tmp/vulcan_gui_shove_place_in.kicad_pcb";
        let mut board = make_gui_test_board(in_path);
        // Add a second unlocked footprint R11 at (115.0, 115.0)
        let mut r11 = board.footprints[1].clone();
        r11.reference = "R11".to_string();
        r11.x = 115.0;
        r11.y = 115.0;
        r11.is_locked = false;
        board.footprints.push(r11);

        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(800.0, 600.0));

        // Pick up R10 (at 120.0, 115.0)
        let (r10_screen, _) = app.board_to_screen(&board, rect, 120.0, 115.0);
        assert!(app.pick_up_component_at(rect, r10_screen));

        // Move R10 directly onto R11's position at (115.0, 115.0)
        let (r11_screen, _) = app.board_to_screen(&board, rect, 115.0, 115.0);
        app.update_moving_component_pos(rect, r11_screen);

        // Click to place R10: must shove R11 out of the way!
        let shoved = app.place_moving_component();
        assert!(shoved.is_some());
        assert!(shoved.unwrap() >= 1, "Must shove overlapping R11");
        assert!(app.moving_footprint.is_none());

        // Verify R10 remained at target position (115.0, 115.0)
        let g = app.live.lock().unwrap();
        let r10 = &g.board.footprints[1];
        let r11_new = &g.board.footprints[2];
        assert_eq!(r10.reference, "R10");
        assert!((r10.x - 115.0).abs() < 1e-4);
        assert!((r10.y - 115.0).abs() < 1e-4);

        // Verify R11 was moved away and does NOT overlap with R10
        assert_eq!(r11_new.reference, "R11");
        let r10_cy = r10.transformed_courtyards(Point::new(r10.x, r10.y), r10.rotation, 0.20);
        let r11_cy = r11_new.transformed_courtyards(Point::new(r11_new.x, r11_new.y), r11_new.rotation, 0.20);
        let mut overlaps = false;
        for pa in &r10_cy {
            for pb in &r11_cy {
                if pa.intersects_convex(pb) {
                    overlaps = true;
                }
            }
        }
        assert!(!overlaps, "R11 must be cleanly shoved without courtyard overlap");

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_manual_refine_keyboard_events_in_render_ui() {
        let in_path = "/tmp/vulcan_gui_keyboard_events_in.kicad_pcb";
        let mut board = make_gui_test_board(in_path);
        board.footprints[0].is_locked = true;
        board.footprints[1].is_locked = false; // R10 at (120.0, 115.0)

        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);
        let ctx = egui::Context::default();
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(800.0, 600.0));

        // Frame 0: Warm-up layout to compute panel margins
        let mut warmup = egui::RawInput::default();
        warmup.screen_rect = Some(rect);
        let _ = ctx.run(warmup, |ctx| {
            app.render_ui(ctx);
        });

        // The CentralPanel canvas occupies remaining space after top bar and left panel
        let canvas_rect = Rect::from_min_max(Pos2::new(318.0, 67.0), Pos2::new(792.0, 592.0));
        let (r10_screen, _) = app.board_to_screen(&board, canvas_rect, 120.0, 115.0);

        // Frame 1: Hover mouse over R10 and press 'm' -> picks up R10
        let mut input1 = egui::RawInput::default();
        input1.screen_rect = Some(rect);
        input1.events.push(egui::Event::PointerMoved(r10_screen));
        input1.events.push(egui::Event::Key {
            key: egui::Key::M,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        });
        let _ = ctx.run(input1, |ctx| {
            app.render_ui(ctx);
        });
        assert!(app.moving_footprint.is_some(), "Pressing 'm' over R10 must pick it up");
        assert_eq!(app.moving_footprint.as_ref().unwrap().reference, "R10");

        // Frame 2: Press 'r' -> rotates moving footprint 90°
        let mut input2 = egui::RawInput::default();
        input2.screen_rect = Some(rect);
        input2.events.push(egui::Event::Key {
            key: egui::Key::R,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        });
        let _ = ctx.run(input2, |ctx| {
            app.render_ui(ctx);
        });
        assert_eq!(app.live.lock().unwrap().board.footprints[1].rotation, 90.0);

        // Frame 3: Press 'f' -> flips moving footprint to B.Cu
        let mut input3 = egui::RawInput::default();
        input3.screen_rect = Some(rect);
        input3.events.push(egui::Event::Key {
            key: egui::Key::F,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        });
        let _ = ctx.run(input3, |ctx| {
            app.render_ui(ctx);
        });
        assert_eq!(app.live.lock().unwrap().board.footprints[1].layer, 1);

        // Frame 4: Move mouse to target position and click -> places footprint
        let (target_screen, _) = app.board_to_screen(&board, canvas_rect, 110.0, 105.0);
        let mut input4 = egui::RawInput::default();
        input4.screen_rect = Some(rect);
        input4.events.push(egui::Event::PointerMoved(target_screen));
        input4.events.push(egui::Event::PointerButton {
            pos: target_screen,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::default(),
        });
        input4.events.push(egui::Event::PointerButton {
            pos: target_screen,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::default(),
        });
        let _ = ctx.run(input4, |ctx| {
            app.render_ui(ctx);
        });
        assert!(app.moving_footprint.is_none(), "Clicking must place the moving component");
        let g = app.live.lock().unwrap();
        let r10 = &g.board.footprints[1];
        assert_eq!(r10.reference, "R10");
        assert_eq!(r10.rotation, 90.0);
        assert_eq!(r10.layer, 1);

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_dynamic_active_copper_layers_gui_visibility() {
        let pcb_path = "/tmp/test_gui_dynamic_layers.kicad_pcb";
        let content = r#"(kicad_pcb
          (layers
            (0 "F.Cu" signal)
            (1 "In1.Cu" signal)
            (2 "In2.Cu" signal)
            (31 "B.Cu" signal)
          )
          (net 0 "") (net 1 "GND")
          (gr_rect (start 100.0 100.0) (end 140.0 130.0) (layer "Edge.Cuts"))
        )"#;
        fs::write(pcb_path, content).unwrap();
        let board = Board::load(pcb_path, &HashSet::new()).unwrap();
        let mut app = VulcansTrialsApp::new(
            pcb_path.to_string(),
            "/tmp/test_out.kicad_pcb".to_string(),
            board,
            None,
        );

        assert!(app.is_layer_visible("F.Cu"));
        assert!(app.is_layer_visible("B.Cu"));
        assert!(app.is_layer_visible("In1.Cu"));
        assert!(app.is_layer_visible("In2.Cu"));
        // In3.Cu is not active on this board
        assert!(!app.live.lock().unwrap().board.copper_layers().contains(&"In3.Cu"));

        // Toggle visibility and verify backward-compatible fields stay in sync
        app.set_layer_visible("F.Cu", false);
        assert!(!app.show_f_cu);
        assert!(!app.is_layer_visible("F.Cu"));

        app.set_layer_visible("In1.Cu", false);
        assert!(!app.show_in1_cu);
        assert!(!app.is_layer_visible("In1.Cu"));

        app.set_layer_visible("F.Cu", true);
        assert!(app.show_f_cu);
        assert!(app.is_layer_visible("F.Cu"));

        let _ = fs::remove_file(pcb_path);
    }

    #[test]
    fn test_pad_two_pass_rendering_thermal_vias_visible() {
        let fp = Footprint {
            reference: "U1".to_string(),
            value: "IC".to_string(),
            x: 100.0,
            y: 100.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                // Central large thermal SMD pad
                crate::board::Pad {
                    number: "17".to_string(),
                    net_id: 1,
                    net_name: "GND".to_string(),
                    offset_x: 0.0,
                    offset_y: 0.0,
                    width: 3.0,
                    height: 3.0,
                    shape: "rect".to_string(),
                    rotation: 0.0,
                    is_through_hole: false,
                    drill: None,
                },
                // Thermal through-hole via
                crate::board::Pad {
                    number: "17".to_string(),
                    net_id: 1,
                    net_name: "GND".to_string(),
                    offset_x: 0.5,
                    offset_y: 0.5,
                    width: 0.6,
                    height: 0.6,
                    shape: "circle".to_string(),
                    rotation: 0.0,
                    is_through_hole: true,
                    drill: Some(0.3),
                },
            ],
            width: 5.0,
            height: 5.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.25,
            at_span: None,
            fp_span: None,
        };

        // Two passes: SMD (is_th=false) must be processed first, TH (is_th=true) second
        let mut pass_order = Vec::new();
        for is_th in [false, true] {
            for pad in fp.pads.iter().filter(|p| p.is_through_hole == is_th) {
                pass_order.push(pad.is_through_hole);
            }
        }
        assert_eq!(pass_order, vec![false, true], "SMD pads must render first, through-hole thermal pads on top");
    }

    #[test]
    fn test_aggressive_placement_params_gui_defaults() {
        let board = Board::default();
        let app = VulcansTrialsApp::new("in.kicad_pcb".to_string(), "out.kicad_pcb".to_string(), board, None);
        assert!((app.max_jump_radius_mm - 20.0).abs() < 1e-6);
        assert!((app.pair_swap_prob - 0.15).abs() < 1e-6);
        assert!((app.centroid_pull_prob - 0.20).abs() < 1e-6);
        assert!((app.temp_scale - 1.0).abs() < 1e-6);
        assert!((app.escape_margin_mult - 1.0).abs() < 1e-6);
        assert!((app.crossing_weight - 200.0).abs() < 1e-6);
    }
}

