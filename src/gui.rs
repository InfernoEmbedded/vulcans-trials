// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use crate::board::{Board, LiveTrialState, Point, TrialPhase};
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

    // Canvas viewport controls
    pub pan_offset: Vec2,
    pub zoom: f32,
    pub show_courtyards: bool,
    pub show_ratsnest: bool,
    pub show_f_cu: bool,
    pub show_b_cu: bool,
    pub show_vias: bool,
    pub show_refs: bool,
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
            pan_offset: Vec2::ZERO,
            zoom: 1.0,
            show_courtyards: true,
            show_ratsnest: true,
            show_f_cu: true,
            show_b_cu: true,
            show_vias: true,
            show_refs: true,
        }
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
        let out_path = self.output_path.clone();

        thread::spawn(move || {
            let mut working_board = {
                let g = live.lock().unwrap();
                g.board.clone()
            };

            if do_place && !stop_flag.load(Ordering::Relaxed) {
                let cfg = PlacementConfig {
                    duration_secs: place_dur,
                    min_spacing_mm: min_spacing,
                    ..PlacementConfig::default()
                };
                run_auto_placer(&mut working_board, &cfg, Some(Arc::clone(&live)), Arc::clone(&stop_flag));
            }

            if do_route && !stop_flag.load(Ordering::Relaxed) {
                let mut grid = RoutingGrid::new(&working_board, grid_res, vulkan);
                let router = AutoRouter::new(max_passes);
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
}

impl eframe::App for VulcansTrialsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(50));

        let telem = self.telemetry.state.lock().unwrap().clone();
        let snap = self.live.lock().unwrap().clone();
        let is_running = self.worker_running.load(Ordering::Relaxed);

        // Top Telemetry & Vulkan Status Bar
        egui::TopBottomPanel::top("vulcan_top_bar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading(egui::RichText::new("🔥 VULCAN'S TRIALS").color(Color32::from_rgb(255, 145, 50)));
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
                ui.heading("Trial Forging Controls");

                ui.horizontal(|ui| {
                    ui.label("Input PCB:");
                    ui.text_edit_singleline(&mut self.input_path);
                });
                ui.horizontal(|ui| {
                    ui.label("Output PCB:");
                    ui.text_edit_singleline(&mut self.output_path);
                });

                if ui.add_enabled(!is_running, egui::Button::new("📂 Reload Input PCB")).clicked() {
                    if let Ok(b) = Board::load(&self.input_path, &HashSet::new()) {
                        let mut g = self.live.lock().unwrap();
                        *g = LiveTrialState::new(b);
                    }
                }

                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            !is_running,
                            egui::Button::new(
                                egui::RichText::new("🔥 Full Trial (Place + Route)")
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

                ui.separator();
                ui.heading("Visibility & View");
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut self.show_courtyards, "Courtyards");
                    ui.checkbox(&mut self.show_ratsnest, "Ratsnest");
                    ui.checkbox(&mut self.show_f_cu, "F.Cu (Top)");
                    ui.checkbox(&mut self.show_b_cu, "B.Cu (Bottom)");
                    ui.checkbox(&mut self.show_vias, "Vias");
                    ui.checkbox(&mut self.show_refs, "References");
                });
                if ui.button("🔍 Reset Camera View").clicked() {
                    self.pan_offset = Vec2::ZERO;
                    self.zoom = 1.0;
                }

                ui.separator();
                ui.heading("Live Trial Telemetry");
                egui::Grid::new("metrics_grid").striped(true).show(ui, |ui| {
                    ui.label("Footprints:");
                    let locked_n = snap.board.footprints.iter().filter(|f| f.is_locked).count();
                    ui.label(format!("{} ({} locked)", snap.board.footprints.len(), locked_n));
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
                ui.heading("Forge Log");
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &snap.log_lines {
                            ui.label(egui::RichText::new(line).monospace().size(11.0));
                        }
                    });
            });

        // Central Interactive 2D Board Canvas
        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::drag());
            let rect = response.rect;
            painter.rect_filled(rect, 0.0, Color32::from_rgb(16, 20, 24));

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

            let b = &snap.board;
            let bw = (b.max_x - b.min_x).max(10.0) as f32;
            let bh = (b.max_y - b.min_y).max(10.0) as f32;
            let base_scale = ((rect.width() - 40.0) / bw).min((rect.height() - 40.0) / bh).max(1.0);
            let scale = base_scale * self.zoom;

            let board_cx = ((b.min_x + b.max_x) * 0.5) as f32;
            let board_cy = ((b.min_y + b.max_y) * 0.5) as f32;
            let screen_center = rect.center() + self.pan_offset;

            let to_screen = |wx: f64, wy: f64| -> Pos2 {
                Pos2::new(
                    screen_center.x + ((wx as f32) - board_cx) * scale,
                    screen_center.y + ((wy as f32) - board_cy) * scale,
                )
            };

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

            // 3. Draw B.Cu Tracks (Blue)
            if self.show_b_cu {
                for tr in &b.tracks {
                    if tr.layer == 1 {
                        let p1 = to_screen(tr.start.0, tr.start.1);
                        let p2 = to_screen(tr.end.0, tr.end.1);
                        let w = ((tr.width as f32) * scale).max(1.5);
                        painter.line_segment([p1, p2], Stroke::new(w, Color32::from_rgb(50, 130, 255)));
                    }
                }
            }

            // 4. Draw F.Cu Tracks (Red)
            if self.show_f_cu {
                for tr in &b.tracks {
                    if tr.layer == 0 {
                        let p1 = to_screen(tr.start.0, tr.start.1);
                        let p2 = to_screen(tr.end.0, tr.end.1);
                        let w = ((tr.width as f32) * scale).max(1.5);
                        painter.line_segment([p1, p2], Stroke::new(w, Color32::from_rgb(235, 65, 55)));
                    }
                }
            }

            // 5. Draw Ratsnest
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

            // 6. Draw Footprints, Courtyards, and Pads
            for fp in &b.footprints {
                if self.show_courtyards {
                    let stroke_col = if fp.is_locked {
                        Color32::from_rgb(240, 175, 50)
                    } else {
                        Color32::from_rgb(70, 210, 160)
                    };
                    let polys = fp.transformed_courtyards(Point::new(fp.x, fp.y), fp.rotation, 0.0);
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
                    let (px, py) = fp.pad_absolute_position(pad);
                    let center = to_screen(px, py);
                    let pw = ((pad.width as f32) * scale).max(2.0);
                    let ph = ((pad.height as f32) * scale).max(2.0);
                    let pad_col = if pad.is_through_hole {
                        Color32::from_rgb(215, 185, 80)
                    } else if pad.net_name == "GND" {
                        Color32::from_rgb(90, 165, 110)
                    } else {
                        Color32::from_rgb(200, 90, 90)
                    };
                    painter.rect_filled(Rect::from_center_size(center, Vec2::new(pw, ph)), 1.0, pad_col);
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

            // 7. Draw Vias
            if self.show_vias {
                for v in &b.vias {
                    let c = to_screen(v.position.0, v.position.1);
                    let outer_r = ((v.size as f32) * 0.5 * scale).max(2.5);
                    let inner_r = ((v.drill as f32) * 0.5 * scale).max(1.0);
                    painter.circle_filled(c, outer_r, Color32::from_rgb(210, 195, 120));
                    painter.circle_filled(c, inner_r, Color32::from_rgb(25, 30, 35));
                }
            }
        });
    }
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
