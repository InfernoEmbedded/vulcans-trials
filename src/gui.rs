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

    // Canvas viewport controls
    pub pan_offset: Vec2,
    pub zoom: f32,
    pub show_courtyards: bool,
    pub show_ratsnest: bool,
    pub show_f_cu: bool,
    pub show_b_cu: bool,
    pub show_in1_cu: bool,
    pub show_in2_cu: bool,
    pub show_vias: bool,
    pub show_refs: bool,
    pub show_holding_palette: bool,
    pub selected_net: Option<String>,
}

impl VulcansTrialsApp {
    pub fn new(input_path: String, output_path: String, board: Board, vulkan: Option<Arc<VulkanEngine>>) -> Self {
        let busy_counter = vulkan.as_ref().map(|v| Arc::clone(&v.compute_busy_ns));
        let telemetry = SystemTelemetry::spawn(busy_counter);
        let live = Arc::new(Mutex::new(LiveTrialState::new(board)));

        Self {
            input_path,
            output_path,
            live,
            vulkan,
            telemetry,
            stop_flag: Arc::new(AtomicBool::new(false)),
            worker_running: Arc::new(AtomicBool::new(false)),
            place_duration_secs: 15,
            min_spacing_mm: 0.25,
            grid_res_mm: 0.10,
            max_ripup_passes: 15,
            allow_layer_flip: false,
            inner_plane_penalty: 8.0,
            pan_offset: Vec2::ZERO,
            zoom: 1.0,
            show_courtyards: true,
            show_ratsnest: true,
            show_f_cu: true,
            show_b_cu: true,
            show_in1_cu: true,
            show_in2_cu: true,
            show_vias: true,
            show_refs: true,
            show_holding_palette: true,
            selected_net: None,
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
        let out_path = self.output_path.clone();

        thread::spawn(move || {
            let mut working_board = {
                let mut g = live.lock().unwrap();
                if do_place && !g.board.holding_palette.is_empty() {
                    let cfg = PlacementConfig {
                        duration_secs: place_dur,
                        min_spacing_mm: min_spacing,
                        allow_layer_flip: allow_flip,
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

                ui.separator();
                ui.heading("Visibility & View");
                let plane_names = snap.board.layer_plane_net_names();
                let in1_txt = match plane_names.get(&2) {
                    Some(n) => format!("In1.Cu ({})", n),
                    None => "In1.Cu (Inner 1)".to_string(),
                };
                let in2_txt = match plane_names.get(&3) {
                    Some(n) => format!("In2.Cu ({})", n),
                    None => "In2.Cu (Inner 2)".to_string(),
                };
                ui.vertical(|ui| {
                    ui.checkbox(
                        &mut self.show_holding_palette,
                        format!("Holding Palette ({})", snap.board.holding_palette.len()),
                    );
                    ui.checkbox(&mut self.show_courtyards, "Courtyards");
                    ui.checkbox(&mut self.show_ratsnest, "Ratsnest");
                    ui.checkbox(&mut self.show_f_cu, "F.Cu (Top)");
                    ui.checkbox(&mut self.show_b_cu, "B.Cu (Bottom)");
                    ui.checkbox(&mut self.show_in1_cu, in1_txt);
                    ui.checkbox(&mut self.show_in2_cu, in2_txt);
                    ui.checkbox(&mut self.show_vias, "Vias");
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
            painter.rect_filled(rect, 0.0, Color32::from_rgb(16, 20, 24));

            if response.clicked() {
                if let Some(click_pos) = response.interact_pointer_pos() {
                    self.handle_canvas_click_at(rect, click_pos);
                }
            }
            if response.secondary_clicked() {
                if let Some(click_pos) = response.interact_pointer_pos() {
                    self.handle_canvas_secondary_click_at(rect, click_pos);
                }
            }
            if response.dragged() {
                self.pan_offset += response.drag_delta();
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
            let edge_min = to_screen(b.min_x + 1.0, b.min_y + 1.0);
            let edge_max = to_screen(b.max_x - 1.0, b.max_y - 1.0);
            painter.rect_stroke(
                Rect::from_min_max(edge_min, edge_max),
                0.0,
                Stroke::new(2.0_f32, Color32::from_rgb(220, 200, 60)),
                egui::StrokeKind::Middle,
            );

            // 2. Draw Keepout Zones
            for ko in &b.keepouts {
                let kmin = to_screen(ko.min_x, ko.min_y);
                let kmax = to_screen(ko.max_x, ko.max_y);
                let krect = Rect::from_min_max(kmin, kmax);
                painter.rect_filled(krect, 0.0, Color32::from_rgba_unmultiplied(180, 40, 40, 35));
                painter.rect_stroke(
                    krect,
                    0.0,
                    Stroke::new(1.2_f32, Color32::from_rgb(200, 70, 70)),
                    egui::StrokeKind::Middle,
                );
            }

            // 3. Draw Inner Copper Layers (In2.Cu Purple, In1.Cu Green)
            if self.show_in2_cu {
                for tr in &b.tracks {
                    if tr.layer == 3 {
                        let p1 = to_screen(tr.start.0, tr.start.1);
                        let p2 = to_screen(tr.end.0, tr.end.1);
                        let w = ((tr.width as f32) * scale).max(1.5);
                        draw_track_segment(&painter, p1, p2, w, Color32::from_rgb(195, 95, 245));
                    }
                }
            }
            if self.show_in1_cu {
                for tr in &b.tracks {
                    if tr.layer == 2 {
                        let p1 = to_screen(tr.start.0, tr.start.1);
                        let p2 = to_screen(tr.end.0, tr.end.1);
                        let w = ((tr.width as f32) * scale).max(1.5);
                        draw_track_segment(&painter, p1, p2, w, Color32::from_rgb(60, 200, 95));
                    }
                }
            }

            // 4. Draw B.Cu Tracks (Blue)
            if self.show_b_cu {
                for tr in &b.tracks {
                    if tr.layer == 1 {
                        let p1 = to_screen(tr.start.0, tr.start.1);
                        let p2 = to_screen(tr.end.0, tr.end.1);
                        let w = ((tr.width as f32) * scale).max(1.5);
                        draw_track_segment(&painter, p1, p2, w, Color32::from_rgb(50, 130, 255));
                    }
                }
            }

            // 5. Draw F.Cu Tracks (Red)
            if self.show_f_cu {
                for tr in &b.tracks {
                    if tr.layer == 0 {
                        let p1 = to_screen(tr.start.0, tr.start.1);
                        let p2 = to_screen(tr.end.0, tr.end.1);
                        let w = ((tr.width as f32) * scale).max(1.5);
                        draw_track_segment(&painter, p1, p2, w, Color32::from_rgb(235, 65, 55));
                    }
                }
            }

            // 6. Draw Ratsnest
            if self.show_ratsnest {
                for seg in &snap.ratsnest {
                    let p1 = to_screen(seg.p1.x, seg.p1.y);
                    let p2 = to_screen(seg.p2.x, seg.p2.y);
                    painter.line_segment(
                        [p1, p2],
                        Stroke::new(0.8_f32, Color32::from_rgba_unmultiplied(180, 210, 255, 85)),
                    );
                }
            }

            // 7. Draw Footprints, Courtyards, Rotated Pads, and Lock Icons
            for fp in &b.footprints {
                let fp_rect = self.footprint_screen_rect(b, rect, fp);
                if self.show_courtyards {
                    let stroke_col = if fp.is_locked {
                        Color32::from_rgb(240, 175, 50)
                    } else if fp.layer == 1 {
                        Color32::from_rgb(80, 170, 255)
                    } else {
                        Color32::from_rgb(70, 210, 160)
                    };
                    let polys = fp.transformed_perimeters_layered(Point::new(fp.x, fp.y), fp.rotation, fp.layer, 0.0);
                    for poly in polys {
                        if poly.vertices.len() >= 3 {
                            let pts: Vec<Pos2> = poly.vertices.iter().map(|v| to_screen(v.x, v.y)).collect();
                            for i in 0..pts.len() {
                                painter.line_segment([pts[i], pts[(i + 1) % pts.len()]], Stroke::new(1.2_f32, stroke_col));
                            }
                        }
                    }
                }

                for pad in &fp.pads {
                    let is_highlighted = self
                        .selected_net
                        .as_ref()
                        .map_or(false, |sn| !sn.is_empty() && sn == &pad.net_name);

                    let (px, py) = fp.pad_absolute_position(pad);
                    let center = to_screen(px, py);
                    let pad_col = if is_highlighted {
                        Color32::from_rgb(255, 245, 140)
                    } else if pad.is_through_hole {
                        Color32::from_rgb(215, 185, 80)
                    } else if pad.net_name == "GND" {
                        Color32::from_rgb(90, 165, 110)
                    } else if fp.layer == 1 {
                        Color32::from_rgb(85, 145, 235)
                    } else {
                        Color32::from_rgb(200, 90, 90)
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
            }

            // 8. Draw Vias
            if self.show_vias {
                for v in &b.vias {
                    let c = to_screen(v.position.0, v.position.1);
                    let outer_r = ((v.size as f32) * 0.5 * scale).max(2.5);
                    let inner_r = ((v.drill as f32) * 0.5 * scale).max(1.0);
                    painter.circle_filled(c, outer_r, Color32::from_rgb(210, 195, 120));
                    painter.circle_filled(c, inner_r, Color32::from_rgb(25, 30, 35));
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

    // 2. Draw Pads
    for pad in &fp.pads {
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
    let plane_names = board.layer_plane_net_names();
    let f_label = match plane_names.get(&0) {
        Some(net) => format!("F.Cu Track (Top - {})", net),
        None => "F.Cu Track (Top)".to_string(),
    };
    let b_label = match plane_names.get(&1) {
        Some(net) => format!("B.Cu Track (Bottom - {})", net),
        None => "B.Cu Track (Bottom)".to_string(),
    };
    let in1_label = match plane_names.get(&2) {
        Some(net) => format!("In1.Cu Track (Inner 1 - {})", net),
        None => "In1.Cu Track (Inner 1)".to_string(),
    };
    let in2_label = match plane_names.get(&3) {
        Some(net) => format!("In2.Cu Track (Inner 2 - {})", net),
        None => "In2.Cu Track (Inner 2)".to_string(),
    };

    let items: &[(&str, Color32)] = &[
        ("Edge.Cuts Outline", Color32::from_rgb(220, 200, 60)),
        ("Keepout Zone", Color32::from_rgb(200, 70, 70)),
        (&f_label, Color32::from_rgb(235, 65, 55)),
        (&b_label, Color32::from_rgb(50, 130, 255)),
        (&in1_label, Color32::from_rgb(60, 200, 95)),
        (&in2_label, Color32::from_rgb(195, 95, 245)),
        ("Ratsnest (Unrouted)", Color32::from_rgba_unmultiplied(180, 210, 255, 180)),
        ("Courtyard (F.Cu Top)", Color32::from_rgb(70, 210, 160)),
        ("Courtyard (B.Cu Bottom)", Color32::from_rgb(80, 170, 255)),
        ("Courtyard (Locked 🔒)", Color32::from_rgb(240, 175, 50)),
        ("SMD Signal Pad", Color32::from_rgb(200, 90, 90)),
        ("SMD GND Pad", Color32::from_rgb(90, 165, 110)),
        ("Through-Hole Pad", Color32::from_rgb(215, 185, 80)),
        ("Via Annular Ring", Color32::from_rgb(210, 195, 120)),
    ];

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
            *label,
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
}

