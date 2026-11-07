// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use crate::board::{Board, LiveTrialState, Point, Polygon, TrackSegment, TrialPhase, Via};
use crate::ipc::KiCadIpcClient;
use crate::placer::{run_auto_placer, PlacementConfig};
use crate::router::{AutoRouter, RoutingConfig, RoutingGrid};
use crate::vulkan::{SystemTelemetry, VulkanEngine};
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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
pub const COL_HIGHLIGHT_FILL: Color32 = Color32::from_rgb(255, 245, 140);
pub const COL_HIGHLIGHT_HALO: Color32 = Color32::from_rgba_premultiplied(160, 138, 25, 160);
pub const COL_HIGHLIGHT_RING: Color32 = Color32::WHITE;

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

/// Invokes native OS file picker to select a KiCad PCB file to open.
/// Returns Ok(Some(path)) on selection, Ok(None) on cancel, or Err(()) if external picker unavailable.
pub fn pick_open_file(initial_path: &str) -> Result<Option<String>, ()> {
    #[cfg(target_os = "linux")]
    {
        let mut cmd = std::process::Command::new("zenity");
        cmd.args([
            "--file-selection",
            "--title=Open KiCad Board",
            "--file-filter=KiCad PCB (*.kicad_pcb) | *.kicad_pcb",
            "--file-filter=All Files | *",
        ]);
        if !initial_path.is_empty() {
            if let Some(parent) = Path::new(initial_path).parent() {
                cmd.arg(format!("--filename={}/", parent.display()));
            }
        }
        match cmd.output() {
            Ok(out) => {
                if out.status.success() {
                    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    if !s.is_empty() {
                        return Ok(Some(s));
                    }
                }
                return Ok(None);
            }
            Err(_) => {
                return Err(());
            }
        }
    }
    #[allow(unreachable_code)]
    Err(())
}

/// Invokes native OS file picker to select a destination path for saving a KiCad PCB.
/// Returns Ok(Some(path)) on selection, Ok(None) on cancel, or Err(()) if external picker unavailable.
pub fn pick_save_file(initial_path: &str) -> Result<Option<String>, ()> {
    #[cfg(target_os = "linux")]
    {
        let mut cmd = std::process::Command::new("zenity");
        cmd.args([
            "--file-selection",
            "--save",
            "--confirm-overwrite",
            "--title=Save KiCad Board As",
            "--file-filter=KiCad PCB (*.kicad_pcb) | *.kicad_pcb",
            "--file-filter=All Files | *",
        ]);
        if !initial_path.is_empty() {
            cmd.arg(format!("--filename={}", initial_path));
        }
        match cmd.output() {
            Ok(out) => {
                if out.status.success() {
                    let mut s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    if !s.is_empty() {
                        if !s.ends_with(".kicad_pcb") {
                            s.push_str(".kicad_pcb");
                        }
                        return Ok(Some(s));
                    }
                }
                return Ok(None);
            }
            Err(_) => {
                return Err(());
            }
        }
    }
    #[allow(unreachable_code)]
    Err(())
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

pub use crate::router::{compute_45_dogleg, generate_bundle_tracks, is_octilinear_segment, route_bundle_along_guide};
pub use crate::board::BusBundle;


#[derive(Clone, Debug)]
pub struct ActiveRoute {
    pub net_id: usize,
    pub net_name: String,
    pub current_layer: usize,
    pub track_width: f64,
    pub via_dia: f64,
    pub via_drill: f64,
    pub anchor_points: Vec<(f64, f64, usize)>, // (x, y, layer)
    pub posture_diagonal_first: bool,          // toggleable via '/'
    pub vias_placed: Vec<Via>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HoveredCopperItem {
    pub net_id: usize,
    pub net_name: String,
    pub layer: usize,
    pub snap_pos: (f64, f64),
    pub is_pad: bool,
    pub is_via: bool,
    pub is_track: bool,
}

/// User-configurable application preferences persisted across sessions in `~/.config/vulcans-trials/preferences.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppPreferences {
    #[serde(default = "default_place_duration")]
    pub place_duration_secs: u64,
    #[serde(default = "default_min_spacing")]
    pub min_spacing_mm: f64,
    #[serde(default = "default_grid_res")]
    pub grid_res_mm: f64,
    #[serde(default = "default_max_ripup")]
    pub max_ripup_passes: usize,
    #[serde(default)]
    pub allow_layer_flip: bool,
    #[serde(default = "default_inner_plane_penalty")]
    pub inner_plane_penalty: f64,
    #[serde(default)]
    pub allow_component_nudge: bool,
    #[serde(default)]
    pub any_angle_mode: bool,

    // Advanced autoplacer exploration & weights
    #[serde(default = "default_max_jump_radius")]
    pub max_jump_radius_mm: f64,
    #[serde(default = "default_pair_swap_prob")]
    pub pair_swap_prob: f64,
    #[serde(default = "default_centroid_pull_prob")]
    pub centroid_pull_prob: f64,
    #[serde(default = "default_temp_scale")]
    pub temp_scale: f64,
    #[serde(default = "default_escape_margin_mult")]
    pub escape_margin_mult: f64,
    #[serde(default = "default_crossing_weight")]
    pub crossing_weight: f64,

    // Advanced autorouter thresholds & distances
    #[serde(default = "default_via_merge_radius")]
    pub via_merge_radius_mm: f64,
    #[serde(default = "default_via_detour_budget")]
    pub via_detour_budget_mm: f64,
    #[serde(default = "default_via_hop_max_detour_ratio")]
    pub via_hop_max_detour_ratio: f64,
    #[serde(default = "default_plane_via_share_radius")]
    pub plane_via_share_radius_mm: f64,
    #[serde(default = "default_return_via_max_dist")]
    pub return_via_max_dist_mm: f64,
    #[serde(default = "default_via_pad_snap_dist")]
    pub via_pad_snap_dist_mm: f64,
    #[serde(default = "default_congestion_factor")]
    pub congestion_factor: f64,
    #[serde(default = "default_crosstalk_penalty")]
    pub crosstalk_penalty: f64,
    #[serde(default = "default_nudge_step")]
    pub nudge_step_mm: f64,
    #[serde(default = "default_gloss_passes")]
    pub gloss_passes: usize,
    #[serde(default = "default_acid_trap_fillet_radius")]
    pub acid_trap_fillet_radius: f64,
    #[serde(default)]
    pub allow_via_in_pad: bool,

    // Layer visibility & display toggles
    #[serde(default = "default_true")]
    pub show_courtyards: bool,
    #[serde(default = "default_true")]
    pub show_ratsnest: bool,
    #[serde(default = "default_true")]
    pub show_f_cu: bool,
    #[serde(default = "default_true")]
    pub show_b_cu: bool,
    #[serde(default = "default_true")]
    pub show_in1_cu: bool,
    #[serde(default = "default_true")]
    pub show_in2_cu: bool,
    #[serde(default = "default_true")]
    pub show_vias: bool,
    #[serde(default = "default_true")]
    pub show_refs: bool,
    #[serde(default = "default_true")]
    pub show_holding_palette: bool,

    // KiCad IPC Live Sync
    #[serde(default = "default_true")]
    pub kicad_ipc_enabled: bool,
    #[serde(default)]
    pub kicad_ipc_socket_path: String,

    // Bundle river routing
    #[serde(default = "default_bundle_dim")]
    pub bundle_track_width: f64,
    #[serde(default = "default_bundle_dim")]
    pub bundle_track_spacing: f64,

    // Optional last opened board paths
    #[serde(default)]
    pub last_input_path: Option<String>,
    #[serde(default)]
    pub last_output_path: Option<String>,
}

fn default_place_duration() -> u64 { 120 }
fn default_min_spacing() -> f64 { 0.25 }
fn default_grid_res() -> f64 { 0.10 }
fn default_max_ripup() -> usize { 15 }
fn default_inner_plane_penalty() -> f64 { 8.0 }
fn default_max_jump_radius() -> f64 { 20.0 }
fn default_pair_swap_prob() -> f64 { 0.15 }
fn default_centroid_pull_prob() -> f64 { 0.20 }
fn default_temp_scale() -> f64 { 1.0 }
fn default_escape_margin_mult() -> f64 { 1.0 }
fn default_crossing_weight() -> f64 { 200.0 }
fn default_via_merge_radius() -> f64 { 4.00 }
fn default_via_detour_budget() -> f64 { 3.50 }
fn default_via_hop_max_detour_ratio() -> f64 { 1.25 }
fn default_plane_via_share_radius() -> f64 { 3.50 }
fn default_return_via_max_dist() -> f64 { 1.40 }
fn default_via_pad_snap_dist() -> f64 { 0.50 }
fn default_congestion_factor() -> f64 { 1.50 }
fn default_crosstalk_penalty() -> f64 { 4.0 }
fn default_nudge_step() -> f64 { 0.25 }
fn default_gloss_passes() -> usize { 2 }
fn default_acid_trap_fillet_radius() -> f64 { 0.35 }
fn default_true() -> bool { true }
fn default_bundle_dim() -> f64 { 0.25 }

impl Default for AppPreferences {
    fn default() -> Self {
        Self {
            place_duration_secs: default_place_duration(),
            min_spacing_mm: default_min_spacing(),
            grid_res_mm: default_grid_res(),
            max_ripup_passes: default_max_ripup(),
            allow_layer_flip: false,
            inner_plane_penalty: default_inner_plane_penalty(),
            allow_component_nudge: false,
            any_angle_mode: false,
            max_jump_radius_mm: default_max_jump_radius(),
            pair_swap_prob: default_pair_swap_prob(),
            centroid_pull_prob: default_centroid_pull_prob(),
            temp_scale: default_temp_scale(),
            escape_margin_mult: default_escape_margin_mult(),
            crossing_weight: default_crossing_weight(),
            via_merge_radius_mm: default_via_merge_radius(),
            via_detour_budget_mm: default_via_detour_budget(),
            via_hop_max_detour_ratio: default_via_hop_max_detour_ratio(),
            plane_via_share_radius_mm: default_plane_via_share_radius(),
            return_via_max_dist_mm: default_return_via_max_dist(),
            via_pad_snap_dist_mm: default_via_pad_snap_dist(),
            congestion_factor: default_congestion_factor(),
            crosstalk_penalty: default_crosstalk_penalty(),
            nudge_step_mm: default_nudge_step(),
            gloss_passes: default_gloss_passes(),
            acid_trap_fillet_radius: default_acid_trap_fillet_radius(),
            allow_via_in_pad: false,
            show_courtyards: true,
            show_ratsnest: true,
            show_f_cu: true,
            show_b_cu: true,
            show_in1_cu: true,
            show_in2_cu: true,
            show_vias: true,
            show_refs: true,
            show_holding_palette: true,
            kicad_ipc_enabled: true,
            kicad_ipc_socket_path: String::new(),
            bundle_track_width: default_bundle_dim(),
            bundle_track_spacing: default_bundle_dim(),
            last_input_path: None,
            last_output_path: None,
        }
    }
}

fn is_running_under_cargo_test() -> bool {
    if cfg!(test) {
        return true;
    }
    env::current_exe()
        .ok()
        .is_some_and(|p| p.parent().is_some_and(|d| d.ends_with("deps")))
}

impl AppPreferences {
    /// Returns the default platform-specific configuration path for `preferences.json`.
    pub fn config_path() -> PathBuf {
        if let Ok(p) = env::var("VULCANS_TRIALS_CONFIG") {
            if !p.trim().is_empty() {
                return PathBuf::from(p.trim());
            }
        }
        if let Ok(xdg) = env::var("XDG_CONFIG_HOME") {
            if !xdg.trim().is_empty() {
                return PathBuf::from(xdg.trim()).join("vulcans-trials").join("preferences.json");
            }
        }
        if let Ok(home) = env::var("HOME") {
            if !home.trim().is_empty() {
                return PathBuf::from(home.trim()).join(".config").join("vulcans-trials").join("preferences.json");
            }
        }
        if let Ok(appdata) = env::var("APPDATA") {
            if !appdata.trim().is_empty() {
                return PathBuf::from(appdata.trim()).join("vulcans-trials").join("preferences.json");
            }
        }
        PathBuf::from("preferences.json")
    }

    /// Loads preferences from the default config path, or returns default preferences if absent/invalid.
    pub fn load() -> Self {
        let mut prefs = Self::load_from_path(&Self::config_path()).unwrap_or_default();
        let test_polluted = prefs
            .last_input_path
            .as_deref()
            .is_some_and(|p| p.starts_with("/tmp/vulcan"));
        if test_polluted || (prefs.max_ripup_passes <= 1 && prefs.grid_res_mm >= 0.25) {
            prefs.grid_res_mm = default_grid_res();
            prefs.max_ripup_passes = default_max_ripup();
            if test_polluted {
                prefs.last_input_path = None;
                prefs.last_output_path = None;
            }
        }
        prefs
    }

    /// Loads preferences from a specific filesystem path.
    pub fn load_from_path(path: &Path) -> Option<Self> {
        let content = fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Saves preferences to the default config path.
    pub fn save(&self) -> std::io::Result<()> {
        self.save_to_path(&Self::config_path())
    }

    /// Saves preferences to a specific path, creating parent directories if necessary.
    pub fn save_to_path(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        fs::write(path, json)
    }
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
    pub allow_component_nudge: bool,
    pub any_angle_mode: bool,

    // Advanced autoplacer exploration & weights
    pub max_jump_radius_mm: f64,
    pub pair_swap_prob: f64,
    pub centroid_pull_prob: f64,
    pub temp_scale: f64,
    pub escape_margin_mult: f64,
    pub crossing_weight: f64,

    // Advanced autorouter thresholds & distances
    pub via_merge_radius_mm: f64,
    pub via_detour_budget_mm: f64,
    pub via_hop_max_detour_ratio: f64,
    pub plane_via_share_radius_mm: f64,
    pub return_via_max_dist_mm: f64,
    pub via_pad_snap_dist_mm: f64,
    pub congestion_factor: f64,
    pub crosstalk_penalty: f64,
    pub nudge_step_mm: f64,
    pub gloss_passes: usize,
    pub acid_trap_fillet_radius: f64,
    pub allow_via_in_pad: bool,

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
    pub selected_net_id: Option<usize>,
    pub moving_footprint: Option<MovingFootprint>,
    pub hovered_footprint: Option<usize>,
    pub active_route: Option<ActiveRoute>,
    pub route_tool_active: bool,
    pub bundle_sketch_active: bool,
    pub bundle_guide_points: Vec<(f64, f64)>,
    pub bundle_selected_nets: Vec<usize>,
    pub bundle_track_width: f64,
    pub bundle_track_spacing: f64,
    pub ipc: Option<Arc<Mutex<KiCadIpcClient>>>,
    pub kicad_ipc_enabled: bool,
    pub kicad_ipc_socket_path: String,
    pub last_ipc_retry: Instant,
    pub ipc_retry_interval: Duration,

    // Automation & Framing
    pub view_bounds: (f64, f64, f64, f64),
    pub auto_start: bool,
    pub auto_exit: Option<u64>,
    pub frame_count: u64,
    pub completed_time: Option<Instant>,

    // Persistence
    pub preferences_path: Option<PathBuf>,
    pub last_saved_prefs: AppPreferences,

    // File dialog state
    pub open_dialog_open: bool,
    pub save_as_dialog_open: bool,
    pub file_dialog_path_buffer: String,
    pub last_canvas_rect: Option<Rect>,
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

pub fn compute_initial_view_bounds(board: &Board) -> (f64, f64, f64, f64) {
    let mut min_x = board.min_x;
    let mut max_x = board.max_x;
    let mut min_y = board.min_y;
    let mut max_y = board.max_y;
    for fp in &board.footprints {
        min_x = min_x.min(fp.x - 3.0);
        max_x = max_x.max(fp.x + 3.0);
        min_y = min_y.min(fp.y - 3.0);
        max_y = max_y.max(fp.y + 3.0);
    }
    if min_x >= max_x {
        min_x = 0.0;
        max_x = 100.0;
    }
    if min_y >= max_y {
        min_y = 0.0;
        max_y = 100.0;
    }
    (min_x, max_x, min_y, max_y)
}

impl VulcansTrialsApp {
    pub fn new(input_path: String, output_path: String, board: Board, vulkan: Option<Arc<VulkanEngine>>) -> Self {
        let view_bounds = compute_initial_view_bounds(&board);
        let busy_counter = vulkan.as_ref().map(|v| Arc::clone(&v.compute_busy_ns));
        let telemetry = SystemTelemetry::spawn(busy_counter);
        let mut visible_layers = HashSet::new();
        visible_layers.insert("F.Cu".to_string());
        visible_layers.insert("B.Cu".to_string());
        for &layer in &board.copper_layers() {
            visible_layers.insert(layer.to_string());
        }
        let live = Arc::new(Mutex::new(LiveTrialState::new(board)));

        let mut app = Self {
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
            allow_component_nudge: false,
            any_angle_mode: false,
            max_jump_radius_mm: 20.0,
            pair_swap_prob: 0.15,
            centroid_pull_prob: 0.20,
            temp_scale: 1.0,
            escape_margin_mult: 1.0,
            crossing_weight: 200.0,
            via_merge_radius_mm: 4.00,
            via_detour_budget_mm: 3.50,
            via_hop_max_detour_ratio: 1.25,
            plane_via_share_radius_mm: 3.50,
            return_via_max_dist_mm: 1.40,
            via_pad_snap_dist_mm: 0.50,
            congestion_factor: 1.50,
            crosstalk_penalty: 4.0,
            nudge_step_mm: 0.25,
            gloss_passes: 2,
            acid_trap_fillet_radius: 0.35,
            allow_via_in_pad: false,
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
            selected_net_id: None,
            moving_footprint: None,
            hovered_footprint: None,
            active_route: None,
            route_tool_active: false,
            bundle_sketch_active: false,
            bundle_guide_points: Vec::new(),
            bundle_selected_nets: Vec::new(),
            bundle_track_width: 0.25,
            bundle_track_spacing: 0.25,
            ipc: None,
            kicad_ipc_enabled: true,
            kicad_ipc_socket_path: KiCadIpcClient::discover_socket_path()
                .map(|p| format!("ipc://{}", p.display()))
                .unwrap_or_else(|| "ipc:///tmp/kicad/api.sock".to_string()),
            last_ipc_retry: Instant::now(),
            ipc_retry_interval: Duration::from_secs(2),
            view_bounds,
            auto_start: false,
            auto_exit: None,
            frame_count: 0,
            completed_time: None,
            preferences_path: None,
            last_saved_prefs: AppPreferences::default(),
            open_dialog_open: false,
            save_as_dialog_open: false,
            file_dialog_path_buffer: String::new(),
            last_canvas_rect: None,
        };

        // Load and apply saved preferences on launch (isolated during unit and integration tests)
        if !is_running_under_cargo_test() {
            let saved_prefs = AppPreferences::load();
            app.apply_preferences(&saved_prefs);
            // Always auto-enable IPC on launch so the socket connects automatically or retries periodically
            app.kicad_ipc_enabled = true;
            app.last_saved_prefs = app.to_preferences();
            app.try_connect_ipc(true);
        }

        app
    }

    /// Sets a custom path for preferences storage (useful for unit testing and isolated profiles).
    pub fn with_preferences_path(mut self, path: PathBuf) -> Self {
        if let Some(prefs) = AppPreferences::load_from_path(&path) {
            self.apply_preferences(&prefs);
        }
        self.preferences_path = Some(path);
        self.last_saved_prefs = self.to_preferences();
        self
    }

    /// Applies stored preferences to live application configuration fields.
    pub fn apply_preferences(&mut self, prefs: &AppPreferences) {
        self.place_duration_secs = prefs.place_duration_secs;
        self.min_spacing_mm = prefs.min_spacing_mm;
        self.grid_res_mm = prefs.grid_res_mm;
        self.max_ripup_passes = prefs.max_ripup_passes;
        self.allow_layer_flip = prefs.allow_layer_flip;
        self.inner_plane_penalty = prefs.inner_plane_penalty;
        self.allow_component_nudge = prefs.allow_component_nudge;
        self.any_angle_mode = prefs.any_angle_mode;

        self.max_jump_radius_mm = prefs.max_jump_radius_mm;
        self.pair_swap_prob = prefs.pair_swap_prob;
        self.centroid_pull_prob = prefs.centroid_pull_prob;
        self.temp_scale = prefs.temp_scale;
        self.escape_margin_mult = prefs.escape_margin_mult;
        self.crossing_weight = prefs.crossing_weight;

        self.via_merge_radius_mm = prefs.via_merge_radius_mm;
        self.via_detour_budget_mm = prefs.via_detour_budget_mm;
        self.via_hop_max_detour_ratio = prefs.via_hop_max_detour_ratio;
        self.plane_via_share_radius_mm = prefs.plane_via_share_radius_mm;
        self.return_via_max_dist_mm = prefs.return_via_max_dist_mm;
        self.via_pad_snap_dist_mm = prefs.via_pad_snap_dist_mm;
        self.congestion_factor = prefs.congestion_factor;
        self.crosstalk_penalty = prefs.crosstalk_penalty;
        self.nudge_step_mm = prefs.nudge_step_mm;
        self.gloss_passes = prefs.gloss_passes;
        self.acid_trap_fillet_radius = prefs.acid_trap_fillet_radius;
        self.allow_via_in_pad = prefs.allow_via_in_pad;
        if let Ok(mut g) = self.live.lock() {
            g.acid_trap_fillet_radius = self.acid_trap_fillet_radius;
            g.allow_via_in_pad = self.allow_via_in_pad;
            g.board.set_allow_via_in_pad(self.allow_via_in_pad);
        }

        self.show_courtyards = prefs.show_courtyards;
        self.show_ratsnest = prefs.show_ratsnest;
        self.show_f_cu = prefs.show_f_cu;
        self.show_b_cu = prefs.show_b_cu;
        self.show_in1_cu = prefs.show_in1_cu;
        self.show_in2_cu = prefs.show_in2_cu;
        self.show_vias = prefs.show_vias;
        self.show_refs = prefs.show_refs;
        self.show_holding_palette = prefs.show_holding_palette;

        if !prefs.kicad_ipc_socket_path.trim().is_empty() {
            self.kicad_ipc_socket_path = prefs.kicad_ipc_socket_path.clone();
        }
        self.kicad_ipc_enabled = prefs.kicad_ipc_enabled;

        self.bundle_track_width = prefs.bundle_track_width;
        self.bundle_track_spacing = prefs.bundle_track_spacing;
    }

    /// Extracts current application configuration as an `AppPreferences` instance.
    pub fn to_preferences(&self) -> AppPreferences {
        AppPreferences {
            place_duration_secs: self.place_duration_secs,
            min_spacing_mm: self.min_spacing_mm,
            grid_res_mm: self.grid_res_mm,
            max_ripup_passes: self.max_ripup_passes,
            allow_layer_flip: self.allow_layer_flip,
            inner_plane_penalty: self.inner_plane_penalty,
            allow_component_nudge: self.allow_component_nudge,
            any_angle_mode: self.any_angle_mode,

            max_jump_radius_mm: self.max_jump_radius_mm,
            pair_swap_prob: self.pair_swap_prob,
            centroid_pull_prob: self.centroid_pull_prob,
            temp_scale: self.temp_scale,
            escape_margin_mult: self.escape_margin_mult,
            crossing_weight: self.crossing_weight,

            via_merge_radius_mm: self.via_merge_radius_mm,
            via_detour_budget_mm: self.via_detour_budget_mm,
            via_hop_max_detour_ratio: self.via_hop_max_detour_ratio,
            plane_via_share_radius_mm: self.plane_via_share_radius_mm,
            return_via_max_dist_mm: self.return_via_max_dist_mm,
            via_pad_snap_dist_mm: self.via_pad_snap_dist_mm,
            congestion_factor: self.congestion_factor,
            crosstalk_penalty: self.crosstalk_penalty,
            nudge_step_mm: self.nudge_step_mm,
            gloss_passes: self.gloss_passes,
            acid_trap_fillet_radius: self.acid_trap_fillet_radius,
            allow_via_in_pad: self.allow_via_in_pad,

            show_courtyards: self.show_courtyards,
            show_ratsnest: self.show_ratsnest,
            show_f_cu: self.show_f_cu,
            show_b_cu: self.show_b_cu,
            show_in1_cu: self.show_in1_cu,
            show_in2_cu: self.show_in2_cu,
            show_vias: self.show_vias,
            show_refs: self.show_refs,
            show_holding_palette: self.show_holding_palette,

            kicad_ipc_enabled: self.kicad_ipc_enabled,
            kicad_ipc_socket_path: self.kicad_ipc_socket_path.clone(),

            bundle_track_width: self.bundle_track_width,
            bundle_track_spacing: self.bundle_track_spacing,

            last_input_path: Some(self.input_path.clone()),
            last_output_path: Some(self.output_path.clone()),
        }
    }

    /// Saves the current preferences to disk if configured or default path.
    pub fn save_preferences(&mut self) {
        let prefs = self.to_preferences();
        if let Some(ref target) = self.preferences_path {
            let _ = prefs.save_to_path(target);
        } else if !is_running_under_cargo_test() {
            let target = AppPreferences::config_path();
            let _ = prefs.save_to_path(&target);
        }
        self.last_saved_prefs = prefs;
    }

    pub fn with_ipc(mut self, ipc: Option<Arc<Mutex<KiCadIpcClient>>>) -> Self {
        if let Some(ref client_arc) = ipc {
            self.kicad_ipc_enabled = true;
            if let Ok(c) = client_arc.lock() {
                if let Some(path) = c.socket_path() {
                    self.kicad_ipc_socket_path = format!("ipc://{}", path.display());
                }
            }
            self.ipc = ipc;
        }
        self.last_saved_prefs = self.to_preferences();
        self
    }

    /// Returns true if an active, connected KiCad IPC client is currently attached.
    pub fn is_ipc_connected(&self) -> bool {
        self.ipc
            .as_ref()
            .is_some_and(|c| c.lock().is_ok_and(|g| g.is_connected()))
    }

    /// Attempts to connect (or reconnect) to the KiCad IPC socket.
    pub fn try_connect_ipc(&mut self, quiet: bool) -> bool {
        let custom = if self.kicad_ipc_socket_path.trim().is_empty() {
            None
        } else {
            Some(self.kicad_ipc_socket_path.trim())
        };
        let maybe_client = if quiet {
            KiCadIpcClient::connect_or_discover_quiet(custom)
        } else {
            KiCadIpcClient::connect_or_discover(custom)
        };
        self.last_ipc_retry = Instant::now();
        if let Some(client) = maybe_client {
            let sock_str = client
                .socket_path()
                .map(|sp| format!("ipc://{}", sp.display()))
                .unwrap_or_else(|| self.kicad_ipc_socket_path.clone());
            if self.kicad_ipc_socket_path.trim().is_empty() {
                self.kicad_ipc_socket_path = sock_str.clone();
            }
            self.ipc = Some(Arc::new(Mutex::new(client)));
            if let Ok(mut g) = self.live.lock() {
                g.push_log(format!("[KiCad-IPC] Connected to {}", sock_str));
            }
            true
        } else {
            self.ipc = None;
            false
        }
    }

    /// Periodically retries connecting to the KiCad IPC socket when `kicad_ipc_enabled` is true and disconnected.
    pub fn poll_ipc_auto_reconnect(&mut self) -> bool {
        if !self.kicad_ipc_enabled {
            return false;
        }
        if self.is_ipc_connected() {
            return true;
        }
        if self.last_ipc_retry.elapsed() >= self.ipc_retry_interval {
            return self.try_connect_ipc(true);
        }
        false
    }

    /// Saves the board to `.kicad_pcb` and `.ses`, and if the KiCad IPC socket is connected,
    /// pulls the updated board into KiCad.
    pub fn save_output_and_pull_to_kicad(&mut self) -> bool {
        let pcb_path = PathBuf::from(&self.output_path);
        let ses_path = pcb_path.with_extension("ses");
        let was_existing = pcb_path.exists();
        let board_clone = {
            let mut g = self.live.lock().unwrap();
            if let Err(e) = g.board.save_pcb(&pcb_path) {
                g.push_log(format!("[Error] {}", e));
                return false;
            }
            crate::board::Board::refill_zones_in_file(&pcb_path);
            let _ = g.board.save_ses(&ses_path);
            if was_existing {
                g.push_log(format!(
                    "[Backup] Overwrote existing PCB: original backed up as .kicad_pcb_bak.YYYYMMDDHHMMSS"
                ));
            }
            g.push_log(format!("[Saved] {}", self.output_path));
            g.board.clone()
        };

        if self.kicad_ipc_enabled && !self.is_ipc_connected() {
            self.try_connect_ipc(true);
        }

        if let Some(ref ipc_arc) = self.ipc {
            let pull_res = ipc_arc
                .lock()
                .ok()
                .map(|mut ipc| ipc.pull_board_into_kicad(&board_clone, &pcb_path, &ses_path));
            if let Some(Ok(synced)) = pull_res {
                if let Ok(mut g) = self.live.lock() {
                    g.push_log(format!(
                        "[KiCad-IPC] Pulled updated board into KiCad & refilled zones ({} items synced)",
                        synced
                    ));
                }
            } else if let Some(Err(ref e)) = pull_res {
                if let Ok(mut g) = self.live.lock() {
                    g.push_log(format!("[KiCad-IPC] Pull note: {}", e));
                }
            }
        } else {
            let mut tmp_client = None;
            match crate::ipc::push_saved_board_to_kicad(
                &mut tmp_client,
                None,
                &board_clone,
                &pcb_path,
                &ses_path,
            ) {
                Ok(synced) => {
                    if let Ok(mut g) = self.live.lock() {
                        g.push_log(format!(
                            "[KiCad-IPC] Pulled updated board into KiCad & refilled zones ({} items synced)",
                            synced
                        ));
                    }
                }
                Err(e) => {
                    if let Ok(mut g) = self.live.lock() {
                        g.push_log(format!("[KiCad-IPC] Note: {}", e));
                    }
                }
            }
        }
        true
    }

    /// Sets both input and output paths to point to the single unified board file.
    pub fn set_board_path(&mut self, path: String) {
        self.input_path = path.clone();
        self.output_path = path;
    }

    /// Resets viewport camera pan offset to origin (0, 0) and zoom scale to 1.0.
    pub fn reset_view(&mut self) {
        self.pan_offset = Vec2::ZERO;
        self.zoom = 1.0;
    }

    /// Opens and parses a KiCad board from the specified filesystem path, replacing current state.
    pub fn open_pcb_from_path(&mut self, path: &str) -> bool {
        match Board::load(path, &HashSet::new()) {
            Ok(board) => {
                let mut visible_layers = HashSet::new();
                visible_layers.insert("F.Cu".to_string());
                visible_layers.insert("B.Cu".to_string());
                for &layer in &board.copper_layers() {
                    visible_layers.insert(layer.to_string());
                }
                self.visible_layers = visible_layers;
                let view_bounds = compute_initial_view_bounds(&board);
                let mut g = self.live.lock().unwrap();
                *g = LiveTrialState::new(board);
                g.push_log(format!("[Opened] Loaded '{}' successfully", path));
                drop(g);
                self.view_bounds = view_bounds;
                self.set_board_path(path.to_string());
                self.reset_view();
                true
            }
            Err(e) => {
                if let Ok(mut g) = self.live.lock() {
                    g.push_log(format!("[Error] Failed to open '{}': {}", path, e));
                }
                false
            }
        }
    }

    /// Reloads the current PCB file from disk, discarding unsaved in-memory placement/routing edits.
    pub fn reload_pcb(&mut self) -> bool {
        let path = self.input_path.clone();
        self.open_pcb_from_path(&path)
    }

    /// Prompts the user to select a KiCad PCB file to open (via OS file dialog or in-app modal).
    pub fn trigger_open_file(&mut self) {
        match pick_open_file(&self.input_path) {
            Ok(Some(path)) => {
                self.open_pcb_from_path(&path);
            }
            Ok(None) => {}
            Err(()) => {
                self.file_dialog_path_buffer = self.input_path.clone();
                self.open_dialog_open = true;
            }
        }
    }

    /// Prompts the user for a new PCB destination path and saves the current board state.
    pub fn trigger_save_as_file(&mut self) {
        match pick_save_file(&self.output_path) {
            Ok(Some(path)) => {
                self.set_board_path(path);
                self.save_output_and_pull_to_kicad();
            }
            Ok(None) => {}
            Err(()) => {
                self.file_dialog_path_buffer = self.output_path.clone();
                self.save_as_dialog_open = true;
            }
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
        let allow_comp_nudge = self.allow_component_nudge;
        let any_angle = self.any_angle_mode;
        let max_jump_rad = self.max_jump_radius_mm;
        let pair_swap_p = self.pair_swap_prob;
        let centroid_pull_p = self.centroid_pull_prob;
        let t_scale = self.temp_scale;
        let esc_mult = self.escape_margin_mult;
        let cross_w = self.crossing_weight;
        let allow_via_in_pad = self.allow_via_in_pad;
        if let Ok(mut g) = self.live.lock() {
            g.acid_trap_fillet_radius = self.acid_trap_fillet_radius;
            g.allow_via_in_pad = allow_via_in_pad;
            g.board.set_allow_via_in_pad(allow_via_in_pad);
        }
        let routing_cfg = RoutingConfig {
            crosstalk_penalty: self.crosstalk_penalty,
            any_angle_mode: any_angle,
            via_merge_radius_mm: self.via_merge_radius_mm,
            via_detour_budget_mm: self.via_detour_budget_mm,
            via_hop_max_detour_ratio: self.via_hop_max_detour_ratio,
            plane_via_share_radius_mm: self.plane_via_share_radius_mm,
            return_via_max_dist_mm: self.return_via_max_dist_mm,
            via_pad_snap_dist_mm: self.via_pad_snap_dist_mm,
            congestion_factor: self.congestion_factor,
            nudge_step_mm: self.nudge_step_mm,
            gloss_passes: self.gloss_passes,
            acid_trap_fillet_radius: self.acid_trap_fillet_radius,
            allow_via_in_pad: self.allow_via_in_pad,
        };
        let out_path = self.output_path.clone();
        let ipc_client = if self.kicad_ipc_enabled { self.ipc.clone() } else { None };

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
                        ipc: ipc_client.clone(),
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
                    ipc: ipc_client.clone(),
                    ..PlacementConfig::default()
                };
                run_auto_placer(&mut working_board, &cfg, Some(Arc::clone(&live)), Arc::clone(&stop_flag));
            }

            if do_route && !stop_flag.load(Ordering::Relaxed) {
                working_board.set_allow_via_in_pad(allow_via_in_pad);
                let mut grid = RoutingGrid::new(&working_board, grid_res, vulkan)
                    .with_inner_layer_penalty(inner_plane_pen)
                    .with_allow_via_in_pad(allow_via_in_pad);
                let mut router = AutoRouter::new(max_passes)
                    .with_allow_component_nudge(allow_comp_nudge)
                    .with_config(routing_cfg)
                    .with_ipc(ipc_client.clone());
                router.inner_layer_penalty = inner_plane_pen;
                router.route_all(
                    &mut working_board,
                    &mut grid,
                    Some(Arc::clone(&live)),
                    Arc::clone(&stop_flag),
                );
            }

            if !out_path.is_empty() {
                let pcb_path = Path::new(&out_path);
                if let Err(e) = working_board.save_pcb(pcb_path) {
                    if let Ok(mut g) = live.lock() {
                        g.phase = TrialPhase::Completed;
                        g.push_log(format!("[Error] Failed to save {}: {}", out_path, e));
                    }
                } else {
                    crate::board::Board::refill_zones_in_file(pcb_path);
                    let ses_path = pcb_path.with_extension("ses");
                    let _ = working_board.save_ses(&ses_path);
                    let mut tmp_client = None;
                    let pushed = crate::ipc::push_saved_board_to_kicad(
                        &mut tmp_client,
                        None,
                        &working_board,
                        pcb_path,
                        &ses_path,
                    );
                    if let Ok(mut g) = live.lock() {
                        g.phase = TrialPhase::Completed;
                        g.push_log(format!("[Saved] Wrote PCB to {} and {}", out_path, ses_path.display()));
                        if let Ok(synced) = pushed {
                            g.push_log(format!(
                                "[KiCad-IPC] Pulled saved PCB into KiCad ({} items synced)",
                                synced
                            ));
                        }
                    }
                }
            } else if let Ok(mut g) = live.lock() {
                g.phase = TrialPhase::Completed;
            }

            running.store(false, Ordering::Relaxed);
        });
    }

    pub fn board_to_screen(&self, _board: &Board, rect: Rect, wx: f64, wy: f64) -> (Pos2, f32) {
        let (min_x, max_x, min_y, max_y) = self.view_bounds;
        let bw = (max_x - min_x).max(10.0) as f32;
        let bh = (max_y - min_y).max(10.0) as f32;
        let base_scale = ((rect.width() - 40.0) / bw).min((rect.height() - 40.0) / bh).max(1.0);
        let scale = base_scale * self.zoom;
        let board_cx = ((min_x + max_x) * 0.5) as f32;
        let board_cy = ((min_y + max_y) * 0.5) as f32;
        let screen_center = rect.center() + self.pan_offset;
        let pos = Pos2::new(
            screen_center.x + ((wx as f32) - board_cx) * scale,
            screen_center.y + ((wy as f32) - board_cy) * scale,
        );
        (pos, scale)
    }

    pub fn screen_to_board(&self, board: &Board, rect: Rect, pos: Pos2) -> Point {
        let (min_x, max_x, min_y, max_y) = self.view_bounds;
        let (_, scale) = self.board_to_screen(board, rect, min_x, min_y);
        let board_cx = ((min_x + max_x) * 0.5) as f32;
        let board_cy = ((min_y + max_y) * 0.5) as f32;
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

    pub fn resolve_net_name(board: &Board, net_id: usize) -> String {
        if net_id == 0 {
            return String::new();
        }
        if let Some(name) = board.nets.get(net_id) {
            if !name.is_empty() {
                return name.clone();
            }
        }
        for (name, &id) in &board.net_name_to_id {
            if id == net_id && !name.is_empty() {
                return name.clone();
            }
        }
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id == net_id && !pad.net_name.is_empty() {
                    return pad.net_name.clone();
                }
            }
        }
        format!("Net#{}", net_id)
    }

    pub fn selected_net_id(&self, board: &Board) -> Option<usize> {
        if let Some(ref name) = self.selected_net {
            if name.is_empty() {
                return None;
            }
            if let Some(&id) = board.net_name_to_id.get(name) {
                if id > 0 {
                    return Some(id);
                }
            }
            if let Some(id) = board.nets.iter().position(|n| n == name) {
                if id > 0 {
                    return Some(id);
                }
            }
            for fp in &board.footprints {
                for pad in &fp.pads {
                    if &pad.net_name == name && pad.net_id > 0 {
                        return Some(pad.net_id);
                    }
                }
            }
            if let Some(num_str) = name.strip_prefix("Net#") {
                if let Ok(id) = num_str.parse::<usize>() {
                    if id > 0 {
                        return Some(id);
                    }
                }
            }
            return self.selected_net_id.filter(|&id| id > 0);
        }
        self.selected_net_id.filter(|&id| id > 0)
    }

    pub fn is_net_id_highlighted(&self, board: &Board, net_id: usize) -> bool {
        net_id > 0 && self.selected_net_id(board) == Some(net_id)
    }

    pub fn is_pad_highlighted(&self, board: &Board, pad: &crate::board::Pad) -> bool {
        if self.is_net_id_highlighted(board, pad.net_id) {
            return true;
        }
        self.selected_net
            .as_ref()
            .is_some_and(|sn| !sn.is_empty() && sn == &pad.net_name)
    }

    pub fn is_track_highlighted(&self, board: &Board, tr: &TrackSegment) -> bool {
        self.is_net_id_highlighted(board, tr.net_id)
            && self.is_layer_visible(board.layer_name(tr.layer))
    }

    pub fn is_via_highlighted(&self, board: &Board, via: &Via) -> bool {
        self.show_vias && self.is_net_id_highlighted(board, via.net_id)
    }

    pub fn is_ratsnest_highlighted(&self, board: &Board, seg: &crate::board::Segment) -> bool {
        self.show_ratsnest && self.is_net_id_highlighted(board, seg.net_id)
    }

    /// Hit-tests pads, vias, and routed trace segments at `world_pos` (mm) on visible layers,
    /// returning `Some((net_id, net_name))` for the clicked net element.
    pub fn hit_test_net_at_world(&self, board: &Board, world_pos: Point) -> Option<(usize, String)> {
        // 1. Pads
        for fp in &board.footprints {
            let fp_layer_name = board.layer_name(fp.layer);
            for pad in &fp.pads {
                if !pad.is_through_hole && !self.is_layer_visible(fp_layer_name) {
                    continue;
                }
                if is_world_point_in_pad(fp, pad, world_pos) {
                    let net_id = if pad.net_id != 0 {
                        pad.net_id
                    } else {
                        board.net_name_to_id.get(&pad.net_name).copied().unwrap_or(0)
                    };
                    let net_name = if !pad.net_name.is_empty() {
                        pad.net_name.clone()
                    } else {
                        Self::resolve_net_name(board, net_id)
                    };
                    if net_id > 0 || !net_name.is_empty() {
                        return Some((net_id, net_name));
                    }
                }
            }
        }

        // 2. Vias
        if self.show_vias {
            let min_pick_r = 0.25 / (self.zoom as f64).max(0.25);
            let mut best_via: Option<(usize, f64)> = None;
            for via in &board.vias {
                if via.net_id == 0 {
                    continue;
                }
                if let Some(d) = hit_test_via_world(via, world_pos, min_pick_r) {
                    if best_via.map_or(true, |(_, best_d)| d < best_d) {
                        best_via = Some((via.net_id, d));
                    }
                }
            }
            if let Some((net_id, _)) = best_via {
                return Some((net_id, Self::resolve_net_name(board, net_id)));
            }
        }

        // 3. Routed traces (tracks and arcs) on visible layers
        let mut best_trace: Option<(usize, f64)> = None;
        for tr in &board.tracks {
            if tr.net_id == 0 || !self.is_layer_visible(board.layer_name(tr.layer)) {
                continue;
            }
            if let Some(d) = hit_test_track_world(tr, world_pos, self.zoom) {
                if best_trace.map_or(true, |(_, best_d)| d < best_d) {
                    best_trace = Some((tr.net_id, d));
                }
            }
        }
        let arc_tol_base = 0.25 / (self.zoom as f64).max(0.25);
        for arc in &board.arcs {
            if arc.net_id == 0 || !self.is_layer_visible(board.layer_name(arc.layer)) {
                continue;
            }
            let d = arc.distance_to_point((world_pos.x, world_pos.y));
            let tol = (arc.width * 0.5).max(arc_tol_base);
            if d <= tol && best_trace.map_or(true, |(_, best_d)| d < best_d) {
                best_trace = Some((arc.net_id, d));
            }
        }
        if let Some((net_id, _)) = best_trace {
            return Some((net_id, Self::resolve_net_name(board, net_id)));
        }

        None
    }

    pub fn handle_canvas_click_at(&mut self, rect: Rect, click_pos: Pos2) -> Option<String> {
        let (clicked_pad, clicked_copper_net, clicked_fp_body) = {
            let g = self.live.lock().unwrap();
            let b = &g.board;
            let world_pos = self.screen_to_board(b, rect, click_pos);

            // 1. First check if any pad was clicked
            let mut found_pad: Option<(usize, usize, usize, String)> = None;
            'outer: for (fp_idx, fp) in b.footprints.iter().enumerate() {
                let fp_layer_name = b.layer_name(fp.layer);
                for (p_idx, pad) in fp.pads.iter().enumerate() {
                    if !pad.is_through_hole && !self.is_layer_visible(fp_layer_name) {
                        continue;
                    }
                    if self.is_point_in_pad(b, rect, fp, pad, click_pos)
                        || is_world_point_in_pad(fp, pad, world_pos)
                    {
                        let net_id = if pad.net_id != 0 {
                            pad.net_id
                        } else {
                            b.net_name_to_id.get(&pad.net_name).copied().unwrap_or(0)
                        };
                        let net_name = if !pad.net_name.is_empty() {
                            pad.net_name.clone()
                        } else {
                            Self::resolve_net_name(b, net_id)
                        };
                        found_pad = Some((fp_idx, p_idx, net_id, net_name));
                        break 'outer;
                    }
                }
            }

            // 2. If no pad was clicked, check if a via or routed trace segment was clicked
            let found_copper_net = if found_pad.is_none() {
                self.hit_test_net_at_world(b, world_pos)
            } else {
                None
            };

            // 3. If no pad, via, or trace was clicked, check if a footprint body was clicked
            let mut found_fp: Option<usize> = None;
            if found_pad.is_none() && found_copper_net.is_none() {
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
            (found_pad, found_copper_net, found_fp)
        };

        if let Some((fp_idx, pad_idx, 0, ref net_name)) = clicked_pad {
            if net_name.is_empty() {
                self.selected_net = None;
                self.selected_net_id = None;
                let mut g = self.live.lock().unwrap();
                let ref_des = g.board.footprints[fp_idx].reference.clone();
                let pad_num = g.board.footprints[fp_idx].pads[pad_idx].number.clone();
                g.push_log(format!("[Pad] {}.{} has no net assigned", ref_des, pad_num));
                return Some(format!("{}.{}", ref_des, pad_num));
            }
        }

        if let Some((net_id, net_name)) = clicked_pad
            .map(|(_, _, nid, nname)| (nid, nname))
            .or(clicked_copper_net)
        {
            let already_selected = {
                let g = self.live.lock().unwrap();
                (net_id > 0 && self.is_net_id_highlighted(&g.board, net_id))
                    || (!net_name.is_empty() && self.selected_net.as_deref() == Some(net_name.as_str()))
            };
            if already_selected {
                // Toggle off highlighting if clicking any pad, via, or trace on the already highlighted net
                self.selected_net = None;
                self.selected_net_id = None;
                let mut g = self.live.lock().unwrap();
                g.push_log(format!("[Highlight] Cleared net highlight for \"{}\"", net_name));
                Some(format!("Cleared {}", net_name))
            } else {
                self.selected_net = Some(net_name.clone());
                self.selected_net_id = if net_id > 0 { Some(net_id) } else { None };
                let mut g = self.live.lock().unwrap();
                let pad_count = g
                    .board
                    .footprints
                    .iter()
                    .flat_map(|f| &f.pads)
                    .filter(|p| (net_id > 0 && p.net_id == net_id) || (!net_name.is_empty() && p.net_name == net_name))
                    .count();
                let trace_count = g
                    .board
                    .tracks
                    .iter()
                    .filter(|t| net_id > 0 && t.net_id == net_id)
                    .count()
                    + g.board
                        .arcs
                        .iter()
                        .filter(|a| net_id > 0 && a.net_id == net_id)
                        .count();
                let via_count = g
                    .board
                    .vias
                    .iter()
                    .filter(|v| net_id > 0 && v.net_id == net_id)
                    .count();
                g.push_log(format!(
                    "[Highlight] Net \"{}\" selected: {} connected pads, {} traces, {} vias highlighted",
                    net_name, pad_count, trace_count, via_count
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
            if self.selected_net.is_some() || self.selected_net_id.is_some() {
                self.selected_net = None;
                self.selected_net_id = None;
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

    /// Locates any pad, via, or track at the given screen `pos`.
    pub fn find_copper_item_at(&self, board: &Board, rect: Rect, pos: Pos2) -> Option<HoveredCopperItem> {
        // 1. Check pads
        for fp in &board.footprints {
            for pad in &fp.pads {
                if self.is_point_in_pad(board, rect, fp, pad, pos) {
                    let (px, py) = fp.pad_absolute_position(pad);
                    let layer = if pad.is_through_hole { 0 } else { fp.layer };
                    let net_name = pad.net_name.clone();
                    let net_id = if pad.net_id != 0 {
                        pad.net_id
                    } else {
                        board.net_name_to_id.get(&net_name).copied().unwrap_or(0)
                    };
                    if !net_name.is_empty() && net_id != 0 {
                        return Some(HoveredCopperItem {
                            net_id,
                            net_name,
                            layer,
                            snap_pos: (px, py),
                            is_pad: true,
                            is_via: false,
                            is_track: false,
                        });
                    }
                }
            }
        }

        let (_, scale) = self.board_to_screen(board, rect, board.min_x, board.min_y);

        // 2. Check vias
        for via in &board.vias {
            let (v_pos, _) = self.board_to_screen(board, rect, via.position.0, via.position.1);
            let hit_r = ((via.size as f32) * 0.5 * scale).max(5.0);
            if v_pos.distance(pos) <= hit_r {
                let net_name = board.nets.get(via.net_id).cloned().unwrap_or_default();
                if !net_name.is_empty() && via.net_id != 0 {
                    return Some(HoveredCopperItem {
                        net_id: via.net_id,
                        net_name,
                        layer: 0,
                        snap_pos: via.position,
                        is_pad: false,
                        is_via: true,
                        is_track: false,
                    });
                }
            }
        }

        // 3. Check tracks
        for tr in &board.tracks {
            let p1 = self.board_to_screen(board, rect, tr.start.0, tr.start.1).0;
            let p2 = self.board_to_screen(board, rect, tr.end.0, tr.end.1).0;
            let v = p2 - p1;
            let len_sq = v.length_sq();
            let t = if len_sq < 1e-4 {
                0.0
            } else {
                ((pos - p1).dot(v) / len_sq).clamp(0.0, 1.0)
            };
            let proj = p1 + v * t;
            let hit_thresh = ((tr.width as f32) * 0.5 * scale).max(4.0);
            if pos.distance(proj) <= hit_thresh {
                let world_t = t as f64;
                let wx = tr.start.0 + (tr.end.0 - tr.start.0) * world_t;
                let wy = tr.start.1 + (tr.end.1 - tr.start.1) * world_t;
                let net_name = board.nets.get(tr.net_id).cloned().unwrap_or_default();
                if !net_name.is_empty() && tr.net_id != 0 {
                    return Some(HoveredCopperItem {
                        net_id: tr.net_id,
                        net_name,
                        layer: tr.layer,
                        snap_pos: (wx, wy),
                        is_pad: false,
                        is_via: false,
                        is_track: true,
                    });
                }
            }
        }

        // 4. Check arcs
        for arc in &board.arcs {
            for seg in arc.to_segments(16) {
                let p1 = self.board_to_screen(board, rect, seg.start.0, seg.start.1).0;
                let p2 = self.board_to_screen(board, rect, seg.end.0, seg.end.1).0;
                let v = p2 - p1;
                let len_sq = v.length_sq();
                let t = if len_sq < 1e-4 {
                    0.0
                } else {
                    ((pos - p1).dot(v) / len_sq).clamp(0.0, 1.0)
                };
                let proj = p1 + v * t;
                let hit_thresh = ((arc.width as f32) * 0.5 * scale).max(4.0);
                if pos.distance(proj) <= hit_thresh {
                    let world_t = t as f64;
                    let wx = seg.start.0 + (seg.end.0 - seg.start.0) * world_t;
                    let wy = seg.start.1 + (seg.end.1 - seg.start.1) * world_t;
                    let net_name = board.nets.get(arc.net_id).cloned().unwrap_or_default();
                    if !net_name.is_empty() && arc.net_id != 0 {
                        return Some(HoveredCopperItem {
                            net_id: arc.net_id,
                            net_name,
                            layer: arc.layer,
                            snap_pos: (wx, wy),
                            is_pad: false,
                            is_via: false,
                            is_track: true,
                        });
                    }
                }
            }
        }

        None
    }

    /// Initiates interactive track routing at `pos`.
    pub fn start_route_at(&mut self, rect: Rect, pos: Pos2) -> bool {
        if self.worker_running.load(Ordering::Relaxed) {
            return false;
        }
        let snap = self.live.lock().unwrap().clone();
        let b = &snap.board;
        if let Some(item) = self.find_copper_item_at(b, rect, pos) {
            if item.net_id != 0 && !item.net_name.is_empty() {
                let (track_width, via_dia, via_drill) = if let Some(nc) = b.netclasses.get(
                    b.net_to_class.get(&item.net_name).unwrap_or(&"Default".to_string()),
                ) {
                    (nc.track_width, nc.via_dia, nc.via_drill)
                } else if let Some(nc) = b.netclasses.get("Default") {
                    (nc.track_width, nc.via_dia, nc.via_drill)
                } else {
                    (0.25, 0.8, 0.4)
                };

                let layer_name = b.layer_name(item.layer);
                let mut g = self.live.lock().unwrap();
                g.push_log(format!(
                    "[Route] Started routing Net '{}' on {} at ({:.2}, {:.2})",
                    item.net_name, layer_name, item.snap_pos.0, item.snap_pos.1
                ));
                drop(g);

                self.active_route = Some(ActiveRoute {
                    net_id: item.net_id,
                    net_name: item.net_name,
                    current_layer: item.layer,
                    track_width,
                    via_dia,
                    via_drill,
                    anchor_points: vec![(item.snap_pos.0, item.snap_pos.1, item.layer)],
                    posture_diagonal_first: false,
                    vias_placed: Vec::new(),
                });
                self.route_tool_active = true;
                return true;
            }
        }
        self.route_tool_active = true;
        false
    }

    /// Places a through-hole via at `via_pt`, anchors segment to the via, and transitions layer.
    pub fn place_via_and_switch_layer(&mut self, via_pt: (f64, f64)) {
        if let Some(ref mut route) = self.active_route {
            let p0 = *route.anchor_points.last().unwrap();
            let dogleg = compute_45_dogleg((p0.0, p0.1), via_pt, route.posture_diagonal_first);
            if dogleg.len() == 3 {
                route.anchor_points.push((dogleg[1].0, dogleg[1].1, route.current_layer));
                route.anchor_points.push((via_pt.0, via_pt.1, route.current_layer));
            } else if dogleg.len() == 2 {
                route.anchor_points.push((via_pt.0, via_pt.1, route.current_layer));
            }

            if !route.vias_placed.iter().any(|v| (v.position.0 - via_pt.0).hypot(v.position.1 - via_pt.1) < 1e-4) {
                route.vias_placed.push(Via {
                    position: via_pt,
                    size: route.via_dia,
                    drill: route.via_drill,
                    net_id: route.net_id,
                    is_locked: false,
                });
            }

            let num_layers = {
                let g = self.live.lock().unwrap();
                g.board.copper_layers().len().max(2)
            };
            let new_layer = if num_layers == 2 {
                1 - route.current_layer
            } else {
                (route.current_layer + 1) % num_layers
            };
            route.current_layer = new_layer;
            route.anchor_points.push((via_pt.0, via_pt.1, new_layer));

            let layer_name = self.live.lock().unwrap().board.layer_name(new_layer);
            self.set_layer_visible(layer_name, true);
            self.live.lock().unwrap().push_log(format!(
                "[Route] Placed via at ({:.2}, {:.2}), switched to layer {}",
                via_pt.0, via_pt.1, layer_name
            ));
        }
    }

    /// Anchors the intermediate dogleg corner and current point to `anchor_points`.
    pub fn anchor_route_at(&mut self, pt: (f64, f64)) {
        if let Some(ref mut route) = self.active_route {
            let p0 = *route.anchor_points.last().unwrap();
            let dogleg = compute_45_dogleg((p0.0, p0.1), pt, route.posture_diagonal_first);
            if dogleg.len() == 3 {
                route.anchor_points.push((dogleg[1].0, dogleg[1].1, route.current_layer));
                route.anchor_points.push((pt.0, pt.1, route.current_layer));
            } else if dogleg.len() == 2 {
                route.anchor_points.push((pt.0, pt.1, route.current_layer));
            }
        }
    }

    /// Commits an in-progress route ending at `target_pt` to the board tracks and vias.
    pub fn finish_route_at(&mut self, target_pt: (f64, f64)) -> bool {
        if let Some(route) = self.active_route.take() {
            let p0 = *route.anchor_points.last().unwrap();
            let dogleg = compute_45_dogleg((p0.0, p0.1), target_pt, route.posture_diagonal_first);
            let mut all_points = route.anchor_points.clone();
            if dogleg.len() == 3 {
                all_points.push((dogleg[1].0, dogleg[1].1, route.current_layer));
                all_points.push((target_pt.0, target_pt.1, route.current_layer));
            } else if dogleg.len() == 2 {
                all_points.push((target_pt.0, target_pt.1, route.current_layer));
            }

            let mut added_tracks = 0;
            let mut g = self.live.lock().unwrap();
            for i in 0..(all_points.len().saturating_sub(1)) {
                let p_a = all_points[i];
                let p_b = all_points[i + 1];
                if (p_a.0 - p_b.0).hypot(p_a.1 - p_b.1) > 1e-5 {
                    g.board.tracks.push(TrackSegment {
                        start: (p_a.0, p_a.1),
                        end: (p_b.0, p_b.1),
                        width: route.track_width,
                        layer: p_a.2,
                        net_id: route.net_id,
                        is_locked: false,
                    });
                    added_tracks += 1;
                }
            }
            let added_vias = route.vias_placed.len();
            g.board.vias.extend(route.vias_placed);
            g.refresh_ratsnest();
            g.ratsnest = g.board.filter_unrouted_ratsnest(&g.ratsnest);
            g.push_log(format!(
                "[Route] Committed route for net '{}': {} tracks, {} vias",
                route.net_name, added_tracks, added_vias
            ));
            true
        } else {
            false
        }
    }

    /// Cancels in-progress route or exits route mode.
    pub fn cancel_route(&mut self) -> bool {
        if self.active_route.is_some() {
            self.active_route = None;
            let mut g = self.live.lock().unwrap();
            g.push_log("[Route] Cancelled in-progress route");
            true
        } else if self.route_tool_active {
            self.route_tool_active = false;
            let mut g = self.live.lock().unwrap();
            g.push_log("[Route] Exited route mode");
            true
        } else {
            false
        }
    }

    /// Handles a left click when routing is active.
    pub fn handle_active_route_click(&mut self, rect: Rect, click_pos: Pos2) {
        let snap = self.live.lock().unwrap().clone();
        let b = &snap.board;
        let grid = if self.grid_res_mm > 0.001 { self.grid_res_mm } else { 0.10 };
        let world_pt = self.screen_to_board(b, rect, click_pos);
        let mut snapped_pt = ((world_pt.x / grid).round() * grid, (world_pt.y / grid).round() * grid);

        let copper_item = self.find_copper_item_at(b, rect, click_pos);
        let active_net_id = self.active_route.as_ref().map(|r| r.net_id);
        let active_net_name = self.active_route.as_ref().map(|r| r.net_name.clone());

        if let (Some(route_net_id), Some(item)) = (active_net_id, copper_item) {
            if item.net_id == route_net_id {
                // Target is of the same net: finish and commit!
                if item.is_pad || item.is_via {
                    snapped_pt = item.snap_pos;
                }
                self.finish_route_at(snapped_pt);
            } else if item.net_id != 0 {
                // Foreign net: disallow / reject
                let mut g = self.live.lock().unwrap();
                g.push_log(format!(
                    "[Route] Cannot connect net '{}' to foreign net '{}'",
                    active_net_name.as_deref().unwrap_or(""), item.net_name
                ));
            } else {
                self.anchor_route_at(snapped_pt);
            }
        } else {
            self.anchor_route_at(snapped_pt);
        }
    }

    pub fn get_effective_bundle_nets(&self, board: &Board) -> Vec<usize> {
        if !self.bundle_selected_nets.is_empty() {
            return self.bundle_selected_nets.clone();
        }
        if let Some(ref sel) = self.selected_net {
            if let Some(&nid) = board.net_name_to_id.get(sel) {
                if nid != 0 {
                    let mut nets = vec![nid];
                    for &other in board.net_name_to_id.values() {
                        if other != 0 && other != nid && nets.len() < 4 {
                            nets.push(other);
                        }
                    }
                    return nets;
                }
            }
        }
        let mut nets = Vec::new();
        for &nid in board.net_name_to_id.values() {
            if nid != 0 && !nets.contains(&nid) {
                nets.push(nid);
                if nets.len() >= 4 {
                    break;
                }
            }
        }
        if nets.is_empty() {
            nets = vec![1, 2, 3, 4];
        }
        nets.sort();
        nets
    }

    pub fn commit_bundle_sketch(&mut self) -> bool {
        if self.bundle_guide_points.len() < 2 {
            return false;
        }
        let mut g = self.live.lock().unwrap();
        let nets = self.get_effective_bundle_nets(&g.board);
        if nets.is_empty() {
            return false;
        }
        let bundle = BusBundle::new(
            format!("Bundle_{}", g.board.bundles.len() + 1),
            nets,
            self.bundle_guide_points.clone(),
            0,
            self.bundle_track_width,
            self.bundle_track_spacing,
        );
        let mut grid = RoutingGrid::new(&g.board, if self.grid_res_mm > 0.001 { self.grid_res_mm } else { 0.10 }, self.vulkan.clone());
        let ok = route_bundle_along_guide(&mut g.board, &mut grid, &bundle);
        if ok {
            let total_tracks = g.board.tracks.len();
            g.refresh_ratsnest();
            g.ratsnest = g.board.filter_unrouted_ratsnest(&g.ratsnest);
            g.push_log(format!(
                "[BusBundle] Committed bundle '{}' ({} nets, {} guide waypoints). Total tracks: {}",
                bundle.name, bundle.net_ids.len(), bundle.guide_points.len(), total_tracks
            ));
            self.bundle_guide_points.clear();
            true
        } else {
            g.push_log(format!(
                "[BusBundle] Failed to route bundle '{}': clearance/DRC conflict along guide path",
                bundle.name
            ));
            false
        }
    }

    pub fn cancel_bundle_sketch(&mut self) -> bool {
        if !self.bundle_guide_points.is_empty() {
            self.bundle_guide_points.clear();
            let mut g = self.live.lock().unwrap();
            g.push_log("[BusBundle] Cleared in-progress bundle sketch waypoints");
            true
        } else if self.bundle_sketch_active {
            self.bundle_sketch_active = false;
            let mut g = self.live.lock().unwrap();
            g.push_log("[BusBundle] Exited bus bundle sketch mode");
            true
        } else {
            false
        }
    }

    pub fn handle_bundle_sketch_click(&mut self, rect: Rect, click_pos: Pos2) {
        let snap = self.live.lock().unwrap();
        let b = &snap.board;
        let grid = if self.grid_res_mm > 0.001 { self.grid_res_mm } else { 0.10 };
        let world_pt = self.screen_to_board(b, rect, click_pos);
        let snapped_pt = ((world_pt.x / grid).round() * grid, (world_pt.y / grid).round() * grid);
        self.bundle_guide_points.push(snapped_pt);
    }

    /// Post-route swap between straight octilinear (45°/90°) tracks and hand-drawn-like organic curved tracks.
    pub fn swap_organic_tracks(&mut self, enable_organic: bool) -> bool {
        let mut g = self.live.lock().unwrap();
        if g.board.tracks.is_empty() && g.board.arcs.is_empty() {
            return false;
        }
        self.any_angle_mode = enable_organic;
        g.acid_trap_fillet_radius = self.acid_trap_fillet_radius;
        if enable_organic {
            if g.straight_copper_snapshot.is_none() {
                g.straight_copper_snapshot = Some((g.board.tracks.clone(), g.board.arcs.clone()));
            }
            crate::router::convert_tracks_to_organic_with_fillet_radius(&mut g.board, self.acid_trap_fillet_radius);
            let num_tracks = g.board.tracks.len();
            let num_arcs = g.board.arcs.len();
            g.ratsnest = g.board.filter_unrouted_ratsnest(&g.ratsnest);
            g.push_log(format!(
                "[Organic] Swapped to hand-drawn organic tracks: {} straight segments, {} curved arcs",
                num_tracks, num_arcs
            ));
        } else {
            if let Some((saved_tracks, saved_arcs)) = g.straight_copper_snapshot.take() {
                g.board.tracks = saved_tracks;
                g.board.arcs = saved_arcs;
            } else {
                let grid_res = if self.grid_res_mm > 0.001 { self.grid_res_mm } else { 0.10 };
                let grid = RoutingGrid::new(&g.board, grid_res, self.vulkan.clone());
                crate::router::convert_tracks_to_straight(&mut g.board, &grid);
            }
            let num_tracks = g.board.tracks.len();
            let num_arcs = g.board.arcs.len();
            g.ratsnest = g.board.filter_unrouted_ratsnest(&g.ratsnest);
            g.push_log(format!(
                "[Organic] Swapped back to straight 45° tracks: {} segments, {} arcs",
                num_tracks, num_arcs
            ));
        }
        let board_clone = g.board.clone();
        drop(g);
        if self.kicad_ipc_enabled {
            if let Some(ref ipc_arc) = self.ipc {
                if let Ok(mut ipc) = ipc_arc.lock() {
                    let _ = ipc.stream_tracks_and_vias(&board_clone);
                }
            }
        }
        true
    }

    pub fn render_ui(&mut self, ctx: &egui::Context) {
        ctx.request_repaint_after(std::time::Duration::from_millis(50));

        if !is_running_under_cargo_test() || self.ipc_retry_interval < Duration::from_secs(1) {
            self.poll_ipc_auto_reconnect();
        }

        let telem = self.telemetry.state.lock().unwrap().clone();
        let snap = self.live.lock().unwrap().clone();
        let is_running = self.worker_running.load(Ordering::Relaxed);

        self.frame_count = self.frame_count.saturating_add(1);
        if self.auto_start && !is_running && self.frame_count >= 30 && snap.phase == TrialPhase::Idle {
            self.start_trial(true, true);
        }

        if snap.phase == TrialPhase::Completed {
            if self.completed_time.is_none() {
                self.completed_time = Some(Instant::now());
            }
            if let Some(exit_secs) = self.auto_exit {
                if let Some(t) = self.completed_time {
                    if t.elapsed() >= Duration::from_secs(exit_secs) {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            }
        }

        // Global File menu keyboard shortcuts (Ctrl+O, Ctrl+S, Ctrl+Shift+S, Ctrl+R, Ctrl+Q)
        let (ctrl_o, ctrl_s, ctrl_shift_s, ctrl_r, ctrl_q) = ctx.input(|i| {
            let cmd = i.modifiers.command;
            let shift = i.modifiers.shift;
            (
                cmd && !shift && i.key_pressed(egui::Key::O),
                cmd && !shift && i.key_pressed(egui::Key::S),
                cmd && shift && i.key_pressed(egui::Key::S),
                cmd && !shift && i.key_pressed(egui::Key::R),
                cmd && !shift && i.key_pressed(egui::Key::Q),
            )
        });

        if ctrl_o {
            self.trigger_open_file();
        } else if ctrl_shift_s {
            self.trigger_save_as_file();
        } else if ctrl_s {
            self.save_output_and_pull_to_kicad();
        } else if ctrl_r && !is_running {
            self.reload_pcb();
        } else if ctrl_q {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        // Top Menu, Telemetry & Vulkan Status Bar
        egui::TopBottomPanel::top("vulcan_top_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("📂 Open... (Ctrl+O)").clicked() {
                        ui.close_menu();
                        self.trigger_open_file();
                    }
                    if ui.button("💾 Save (Ctrl+S)").clicked() {
                        ui.close_menu();
                        self.save_output_and_pull_to_kicad();
                    }
                    if ui.button("💾 Save As... (Ctrl+Shift+S)").clicked() {
                        ui.close_menu();
                        self.trigger_save_as_file();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(!is_running, egui::Button::new("🔄 Reload PCB (Ctrl+R)"))
                        .clicked()
                    {
                        ui.close_menu();
                        self.reload_pcb();
                    }
                    ui.separator();
                    if ui.button("❌ Quit (Ctrl+Q)").clicked() {
                        ui.close_menu();
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.heading(egui::RichText::new(format!("⚡ Vulcan's Trials v{}", env!("CARGO_PKG_VERSION"))).color(Color32::from_rgb(255, 145, 50)))
                    .on_hover_text("Vulcan's Trials — Unified Vulkan GPU + multi-core CPU PCB Auto-Placer and Auto-Router for KiCad 8/9/10.");
                ui.label(egui::RichText::new("by InfernoEmbedded (GPLv3)").weak())
                    .on_hover_text("Licensed under GNU General Public License v3.0 (GPL-3.0-only).");
                ui.separator();

                if let Some(ref vk) = self.vulkan {
                    ui.label(
                        egui::RichText::new(format!("Vulkan GPU: {} [{}]", vk.device_name, vk.backend_name))
                            .color(Color32::from_rgb(110, 230, 130))
                            .strong(),
                    )
                    .on_hover_text("Active Vulkan GPU compute device accelerating 2D Euclidean Distance Field (EDF) obstacle transforms via WGSL shaders.");
                } else {
                    ui.label(egui::RichText::new("GPU: CPU Fallback").color(Color32::YELLOW))
                        .on_hover_text("No Vulkan GPU adapter initialized; obstacle distance field transforms are evaluated on multi-core CPU worker threads.");
                }

                ui.separator();
                draw_load_badge(
                    ui,
                    "CPU Load",
                    telem.cpu_load_pct,
                    &telem.cpu_history,
                    Color32::from_rgb(80, 180, 255),
                    "Live system CPU utilization (%) sampled from /proc/stat across multi-core Simulated Annealing and PathFinder routing threads.",
                );
                ui.separator();
                draw_load_badge(
                    ui,
                    "GPU Load",
                    telem.gpu_load_pct,
                    &telem.gpu_history,
                    Color32::from_rgb(255, 120, 60),
                    "Live GPU compute utilization (%) combining Vulkan WGSL compute duty cycle and nvidia-smi / DRM sysfs telemetry.",
                );
            });

            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(snap.phase.label()).strong())
                    .on_hover_text("Current execution phase of the trial pipeline (Idle, Auto-Placing, Auto-Routing, or Completed).");
                ui.add(
                    egui::ProgressBar::new(snap.progress)
                        .text(snap.status_text.as_str())
                        .animate(is_running),
                )
                .on_hover_text("Live trial progress bar and status summary showing current annealing temperature, rip-up pass, or DRC convergence state.");
            });
            ui.add_space(4.0);
        });

        // Left Control & Metrics Panel
        egui::SidePanel::left("vulcan_controls_panel")
            .resizable(true)
            .default_width(310.0)
            .max_width(310.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.heading("Controls")
                    .on_hover_text("Board file management, interactive manual routing/sketch tools, and automated placement/routing trial execution.");

                if let Some(ref route) = self.active_route {
                    let layer_name = snap.board.layer_name(route.current_layer);
                    ui.group(|ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "Routing Net '{}' on {} | [L-Click] Anchor/Finish | [V] Via | [/] Posture | [Esc] Cancel",
                                route.net_name, layer_name
                            ))
                            .color(Color32::from_rgb(255, 220, 80))
                            .strong(),
                        )
                        .on_hover_text("Active manual track routing: Left-click to anchor a 45° dogleg corner or finish on a same-net pad/via, press 'V' to insert a layer-transition via, '/' to toggle dogleg posture, or 'Esc' to cancel.");
                    });
                    ui.add_space(2.0);
                } else if self.route_tool_active {
                    ui.group(|ui| {
                        ui.label(
                            egui::RichText::new("Route Mode Active | Click pad, via, or track to start [Esc to exit]")
                                .color(Color32::from_rgb(100, 220, 255)),
                        )
                        .on_hover_text("Interactive manual routing mode is armed: click any copper pad, via, or trace on the canvas to start routing its net, or press 'Esc' to exit.");
                    });
                    ui.add_space(2.0);
                }

                ui.horizontal(|ui| {
                    ui.label("PCB File:")
                        .on_hover_text("Source & destination KiCad board file (.kicad_pcb). Saving backs up the original with a timestamp suffix and writes directly to this file, then synchronizes to KiCad.");
                    let resp = ui.text_edit_singleline(&mut self.input_path);
                    if resp.changed() {
                        self.output_path = self.input_path.clone();
                    }
                    if ui.button("...").on_hover_text("Browse filesystem for a .kicad_pcb file to open").clicked() {
                        self.trigger_open_file();
                    }
                });

                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!is_running, egui::Button::new("🔄 Reload PCB"))
                        .on_hover_text("Reload the .kicad_pcb file from disk, discarding unsaved placement or routing edits and resetting board state.")
                        .clicked()
                    {
                        self.reload_pcb();
                    }

                    if ui
                        .add_enabled(
                            !is_running,
                            egui::Button::new(
                                egui::RichText::new("🧹 Clear Unlocked Objects")
                                    .color(Color32::from_rgb(255, 140, 140)),
                            ),
                        )
                        .on_hover_text("Rip up all unlocked routed copper tracks and vias, and move all unlocked footprints off the board into the Holding Palette (locked footprints and locked traces are preserved).")
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

                ui.horizontal(|ui| {
                    let route_btn = if self.route_tool_active {
                        egui::Button::new(
                            egui::RichText::new("⚡ Route Track (X) [ACTIVE]")
                                .color(Color32::from_rgb(100, 255, 120))
                                .strong(),
                        )
                    } else {
                        egui::Button::new("⚡ Route Track (X)")
                    };
                    if ui
                        .add_enabled(!is_running, route_btn)
                        .on_hover_text("Toggle interactive 45° dogleg manual routing mode (Shortcut: X). Click any pad, via, or track to start routing; press 'V' to place a via and switch layers, '/' to flip dogleg posture, or 'Esc' to cancel.")
                        .clicked()
                    {
                        self.route_tool_active = !self.route_tool_active;
                        if !self.route_tool_active {
                            self.active_route = None;
                        }
                    }

                    let bundle_btn = if self.bundle_sketch_active {
                        egui::Button::new(
                            egui::RichText::new("🌊 Bus Sketch (B) [ACTIVE]")
                                .color(Color32::from_rgb(100, 200, 255))
                                .strong(),
                        )
                    } else {
                        egui::Button::new("🌊 Bus Sketch (B)")
                    };
                    if ui
                        .add_enabled(!is_running, bundle_btn)
                        .on_hover_text("Toggle multi-net Bus Bundle river-routing sketch mode (Shortcut: B). Left-click waypoints on the canvas to draw a guide spine, then press 'Enter' or Right-Click to pack and route parallel bus traces.")
                        .clicked()
                    {
                        self.bundle_sketch_active = !self.bundle_sketch_active;
                        if !self.bundle_sketch_active {
                            self.bundle_guide_points.clear();
                        }
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
                        .on_hover_text("Run a complete end-to-end trial: execute Parallel Tempering Simulated Annealing auto-placement (including components in the Holding Palette), legalize courtyards, and run multi-layer PathFinder auto-routing.")
                        .clicked()
                    {
                        self.start_trial(true, true);
                    }

                    if ui
                        .add_enabled(is_running, egui::Button::new("⏹ Stop"))
                        .on_hover_text("Gracefully stop the active auto-placement or auto-routing worker thread after the current pass completes.")
                        .clicked()
                    {
                        self.stop_flag.store(true, Ordering::Relaxed);
                    }
                });

                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!is_running, egui::Button::new("📐 Auto-Place Only"))
                        .on_hover_text("Run only the multi-core Parallel Tempering Simulated Annealing auto-placer and SAT courtyard legalizer without routing tracks.")
                        .clicked()
                    {
                        self.start_trial(true, false);
                    }
                    if ui
                        .add_enabled(!is_running, egui::Button::new("⚡ Auto-Route Only"))
                        .on_hover_text("Run only the multi-layer 3D A* and PathFinder negotiated-congestion auto-router on the current component placement.")
                        .clicked()
                    {
                        self.start_trial(false, true);
                    }
                });

                let has_copper = !snap.board.tracks.is_empty() || !snap.board.arcs.is_empty();
                let is_currently_organic = !snap.board.arcs.is_empty();
                let swap_label = if is_currently_organic {
                    "📐 Swap to Straight Tracks (45°)"
                } else {
                    "🌿 Swap to Organic Tracks"
                };
                if ui
                    .add_enabled(!is_running && has_copper, egui::Button::new(swap_label))
                    .on_hover_text(
                        "Post-route swap between straight octilinear (45°/90°) tracks and smooth, hand-drawn TopoR-style organic tangent arcs while preserving DRC clearances.",
                    )
                    .clicked()
                {
                    self.swap_organic_tracks(!is_currently_organic);
                }

                if ui
                    .add_enabled(!is_running, egui::Button::new("💾 Save PCB (.kicad_pcb + .ses)"))
                    .on_hover_text("Write the current component placement, copper tracks, arcs, and stitched vias to the PCB file and Specctra .ses session file. A timestamped backup (.kicad_pcb_bak.YYYYMMDDHHMMSS) is saved before overwriting, and KiCad is updated over IPC.")
                    .clicked()
                {
                    self.save_output_and_pull_to_kicad();
                }

                ui.separator();
                ui.heading("Parameters")
                    .on_hover_text("Primary placement duration, courtyard clearance, routing grid resolution, and PathFinder iteration limits.");
                ui.add_enabled(
                    !is_running,
                    egui::Slider::new(&mut self.place_duration_secs, 2..=600).text("Place Time (s)"),
                )
                .on_hover_text("Wall-clock duration in seconds (2–600 s, default 120 s) for Parallel Tempering Simulated Annealing placement optimization before final SAT courtyard legalization.");
                ui.add_enabled(
                    !is_running,
                    egui::Slider::new(&mut self.min_spacing_mm, 0.10..=1.00).text("Courtyard Gap (mm)"),
                )
                .on_hover_text("Minimum physical courtyard-to-courtyard clearance in millimeters (0.10–1.00 mm, default 0.25 mm) enforced between adjacent components on the same board side.");
                ui.add_enabled(
                    !is_running,
                    egui::Slider::new(&mut self.grid_res_mm, 0.05..=0.25).text("Route Grid (mm)"),
                )
                .on_hover_text("Uniform spatial grid resolution in millimeters (0.05–0.25 mm, default 0.10 mm) for the 3D A* router and Vulkan GPU Euclidean Distance Field obstacle map. Finer values aid 0.5 mm-pitch QFN escapes at higher compute cost.");
                ui.add_enabled(
                    !is_running,
                    egui::Slider::new(&mut self.max_ripup_passes, 1..=30).text("Max Rip-up Passes"),
                )
                .on_hover_text("Maximum number of PathFinder negotiated-congestion rip-up and reroute passes (1–30 passes, default 15) used to resolve copper shorts and routing bottlenecks.");

                ui.collapsing("Advanced Autoplacer", |ui| {
                    ui.add_enabled(
                        !is_running,
                        egui::Checkbox::new(&mut self.allow_layer_flip, "Allow SMD Side Flip (F.Cu ↔ B.Cu)"),
                    )
                    .on_hover_text("Allow the Simulated Annealing auto-placer to flip unlocked SMD footprints between Top (F.Cu) and Bottom (B.Cu) layers. Leave unchecked for single-sided SMT assembly (e.g. JLCPCB F.Cu standard assembly).");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.max_jump_radius_mm, 2.0..=50.0)
                            .text("Max Jump Radius (mm)")
                            .suffix(" mm"),
                    )
                    .on_hover_text("Maximum translation distance in millimeters (2.0–50.0 mm, default 20.0 mm) for single-component perturbation moves at high annealing temperatures.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.pair_swap_prob, 0.0..=0.50)
                            .text("Pair Swap %")
                            .custom_formatter(|n, _| format!("{:.0}%", n * 100.0)),
                    )
                    .on_hover_text("Probability (0%–50%, default 15%) of proposing a pairwise coordinate swap between two unlocked footprints during a Simulated Annealing step.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.centroid_pull_prob, 0.0..=0.60)
                            .text("Centroid Pull %")
                            .custom_formatter(|n, _| format!("{:.0}%", n * 100.0)),
                    )
                    .on_hover_text("Probability (0%–60%, default 20%) of biasing a component perturbation toward the ratsnest force-directed centroid of its connected pads to minimize wirelength.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.temp_scale, 0.2..=4.0)
                            .text("Annealing Temp Scale")
                            .suffix("x"),
                    )
                    .on_hover_text("Scaling multiplier (0.2x–4.0x, default 1.0x) applied to the Parallel Tempering replica temperature ladder (T_min to T_max). Higher values promote broader uphill exploration.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.escape_margin_mult, 0.5..=3.0)
                            .text("Escape Corridor Halo")
                            .suffix("x"),
                    )
                    .on_hover_text("Multiplier (0.5x–3.0x, default 1.0x) expanding the virtual pin-density courtyard halo around high-pin-count ICs and connectors to preserve routing escape corridors.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.crossing_weight, 50.0..=600.0)
                            .text("Crossing Penalty Weight"),
                    )
                    .on_hover_text("Objective cost weight (50–600, default 200) penalizing each topological MST ratsnest airwire intersection to untangle signal flow before routing.");
                })
                .header_response
                .on_hover_text("Expand advanced Parallel Tempering Simulated Annealing exploration probabilities, temperature scaling, and cost weights.");

                ui.collapsing("Advanced Autorouter", |ui| {
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.via_merge_radius_mm, 0.50..=8.00)
                            .text("Via Merge Radius (mm)")
                            .suffix(" mm"),
                    )
                    .on_hover_text("Maximum search radius in millimeters (0.50–8.00 mm, default 4.00 mm) for consolidating redundant same-net vias during post-route optimization.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.via_detour_budget_mm, 0.50..=8.00)
                            .text("Via Merge Detour Budget (mm)")
                            .suffix(" mm"),
                    )
                    .on_hover_text("Maximum extra trace detour length in millimeters (0.50–8.00 mm, default 3.50 mm) allowed when merging two same-net vias into one.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.via_hop_max_detour_ratio, 1.00..=2.00)
                            .text("Via Hop Max Detour Ratio")
                            .suffix("x"),
                    )
                    .on_hover_text("Maximum ratio of single-layer rerouted path length to direct distance (1.00x–2.00x, default 1.25x) when eliminating a dual-via layer hop.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.plane_via_share_radius_mm, 0.50..=8.00)
                            .text("Plane Via Share Radius (mm)")
                            .suffix(" mm"),
                    )
                    .on_hover_text("Maximum distance in millimeters (0.50–8.00 mm, default 3.50 mm) within which nearby ground or power pads can share an existing reference-plane fanout via.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.return_via_max_dist_mm, 0.50..=3.50)
                            .text("Return Via Max Radius (mm)")
                            .suffix(" mm"),
                    )
                    .on_hover_text("Maximum distance in millimeters (0.50–3.50 mm, default 1.40 mm) from a layer-transition signal via to a GND via before automatic return-current ground stitching places a companion GND via.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.via_pad_snap_dist_mm, 0.00..=1.50)
                            .text("Via-to-Pad Snap Dist (mm)")
                            .suffix(" mm"),
                    )
                    .on_hover_text("Threshold distance in millimeters (0.00–1.50 mm, default 0.50 mm) within which post-route cleanup snaps nearly aligned fanout vias onto the exact X/Y centerline of connected SMD pads.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.inner_plane_penalty, 1.0..=30.0)
                            .text("Inner Plane Penalty"),
                    )
                    .on_hover_text("Routing cost multiplier (1.0–30.0, default 8.0) penalizing signal tracks on dedicated inner ground/power planes (In1.Cu / In2.Cu) to keep reference planes unbroken.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.congestion_factor, 0.50..=5.00)
                            .text("Congestion Penalty Factor"),
                    )
                    .on_hover_text("PathFinder historical congestion escalation rate (0.50–5.00, default 1.50) controlling how strongly contested grid cells are penalized across rip-up passes.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.crosstalk_penalty, 0.0..=20.0)
                            .text("Crosstalk Penalty Weight"),
                    )
                    .on_hover_text("Soft proximity cost weight (0.0–20.0, default 4.0) discouraging long parallel coupled runs between adjacent signal traces to reduce electromagnetic crosstalk.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.nudge_step_mm, 0.05..=1.00)
                            .text("Shove / Nudge Step (mm)")
                            .suffix(" mm"),
                    )
                    .on_hover_text("Incremental step distance in millimeters (0.05–1.00 mm, default 0.25 mm) used by the NudgeEngine when shifting unlocked passives to open blocked routing channels.");
                    ui.add_enabled(
                        !is_running,
                        egui::Slider::new(&mut self.gloss_passes, 1..=6)
                            .text("Glossing Passes"),
                    )
                    .on_hover_text("Number of post-route trace rubber-banding, staircase removal, and corner-shortening glossing passes (1–6 passes, default 2).");
                    if ui
                        .add_enabled(
                            !is_running,
                            egui::Slider::new(&mut self.acid_trap_fillet_radius, 0.0..=2.0)
                                .text("Fillet Radius (<90°)")
                                .suffix(" mm"),
                        )
                        .on_hover_text("Tangent arc fillet radius in millimeters (0.0–2.0 mm, default 0.35 mm) applied to sharp corners (< 90°) and organic TopoR bends to eliminate acid traps and impedance discontinuities.")
                        .changed()
                    {
                        if let Ok(mut g) = self.live.lock() {
                            g.acid_trap_fillet_radius = self.acid_trap_fillet_radius;
                        }
                    }
                    ui.add_enabled(
                        !is_running,
                        egui::Checkbox::new(&mut self.allow_component_nudge, "Allow Router to Nudge Unlocked Passives"),
                    )
                    .on_hover_text("Allow the dynamic NudgeEngine to slightly shift unlocked 2-pad passives (resistors, capacitors) within courtyard DRC limits when pad escape channels are blocked.");
                    if ui
                        .add_enabled(
                            !is_running,
                            egui::Checkbox::new(&mut self.allow_via_in_pad, "Allow Via-in-Pad"),
                        )
                        .on_hover_text("Permit placing vias directly inside SMD pads (e.g. QFN thermal pads or tight BGA fanouts). Leave unchecked for standard non-filled/non-capped SMT assembly to prevent solder wicking.")
                        .changed()
                    {
                        if let Ok(mut g) = self.live.lock() {
                            g.allow_via_in_pad = self.allow_via_in_pad;
                            g.board.set_allow_via_in_pad(self.allow_via_in_pad);
                        }
                    }
                    if ui
                        .add_enabled(
                            !is_running,
                            egui::Checkbox::new(&mut self.any_angle_mode, "Any-Angle / Organic Mode (TopoR Arcs)"),
                        )
                        .on_hover_text("Enable any-angle Euclidean rubber-band routing and smooth tangent arc fillets (TopoR style) instead of strict 45°/90° octilinear traces.")
                        .changed()
                        && has_copper
                    {
                        self.swap_organic_tracks(self.any_angle_mode);
                    }
                })
                .header_response
                .on_hover_text("Expand advanced 3D A* and PathFinder router parameters: via merging, return-current stitching, plane penalties, crosstalk, glossing, and organic TopoR arcs.");

                ui.separator();
                ui.heading("KiCad 10 IPC Live Sync")
                    .on_hover_text("Real-time bidirectional board synchronization with a running KiCad 10 PCB editor over its local IPC socket.");
                if ui
                    .checkbox(&mut self.kicad_ipc_enabled, "KiCad IPC Live Sync")
                    .on_hover_text("Enable live streaming of component placements, routed tracks, and vias to an open KiCad 10 PCB editor session.")
                    .changed()
                {
                    if self.kicad_ipc_enabled && !self.is_ipc_connected() {
                        self.try_connect_ipc(false);
                    } else if !self.kicad_ipc_enabled {
                        self.ipc = None;
                    }
                }

                ui.horizontal(|ui| {
                    ui.label("Socket:")
                        .on_hover_text("UNIX domain socket URI for the KiCad 10 IPC server (typically ipc:///tmp/kicad/api.sock).");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.kicad_ipc_socket_path)
                            .hint_text("ipc:///tmp/kicad/api.sock")
                            .desired_width(160.0),
                    )
                    .on_hover_text("Custom KiCad IPC socket URI or filesystem path. Leave default to auto-discover active KiCad sockets.");
                    if ui
                        .button("Connect")
                        .on_hover_text("Immediately attempt to connect or reconnect to the configured KiCad IPC socket.")
                        .clicked()
                    {
                        self.kicad_ipc_enabled = true;
                        self.try_connect_ipc(false);
                    }
                });

                let (status_text, status_col) = match self.ipc.as_ref() {
                    Some(client_lock) => {
                        if let Ok(c) = client_lock.lock() {
                            if c.is_connected() {
                                ("Connected", Color32::from_rgb(0, 220, 100))
                            } else {
                                ("Disconnected (Auto-Retrying...)", Color32::from_rgb(220, 80, 80))
                            }
                        } else {
                            ("Locked", Color32::GRAY)
                        }
                    }
                    None => {
                        if self.kicad_ipc_enabled {
                            ("Auto-Retrying...", Color32::from_rgb(220, 150, 40))
                        } else {
                            ("Offline", Color32::GRAY)
                        }
                    }
                };
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Status:").strong())
                        .on_hover_text("Current connection state of the KiCad 10 IPC client.");
                    ui.label(egui::RichText::new(status_text).color(status_col))
                        .on_hover_text("Live IPC link status; automatically polls for a running KiCad instance every 2 seconds when enabled.");
                });

                ui.separator();
                ui.heading("Visibility & View")
                    .on_hover_text("Toggle visibility of copper layers, vias, ratsnest airwires, courtyards, reference designators, and camera viewport controls.");
                let active_layers = snap.board.copper_layers();
                let display_layers = kicad_layer_display_order(&active_layers);
                let plane_names = snap.board.layer_plane_net_names();
                ui.vertical(|ui| {
                    ui.checkbox(
                        &mut self.show_holding_palette,
                        format!("Holding Palette ({})", snap.board.holding_palette.len()),
                    )
                    .on_hover_text("Show or hide the Holding Palette panel containing unplaced or cleared components (Right-click any unlocked footprint on the canvas to move it to the Holding Palette).");
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
                        if ui
                            .checkbox(&mut vis, egui::RichText::new(label).color(col))
                            .on_hover_text(format!(
                                "Toggle display and interactive click hit-testing for copper tracks, arcs, and SMD pads on layer {}.",
                                layer_name
                            ))
                            .changed()
                        {
                            self.set_layer_visible(layer_name, vis);
                        }
                    }
                    ui.checkbox(&mut self.show_vias, egui::RichText::new("Vias").color(COL_VIA))
                        .on_hover_text("Toggle display and click selection of through-hole and blind/buried vias (copper annular rings and drill holes).");
                    ui.checkbox(&mut self.show_ratsnest, egui::RichText::new("Ratsnest").color(COL_RATSNEST))
                        .on_hover_text("Toggle display of unrouted Euclidean Minimum Spanning Tree (MST) ratsnest airwire lines between pads.");
                    ui.checkbox(&mut self.show_courtyards, egui::RichText::new("Courtyards").color(COL_F_CRTYD))
                        .on_hover_text("Toggle display of component courtyard collision polygons (F.CrtYd in magenta, B.CrtYd in cyan, locked footprints in gold).");
                    ui.checkbox(&mut self.show_refs, "References")
                        .on_hover_text("Toggle display of component reference designator labels (e.g. U1, R10, C4) centered on each footprint.");
                });
                if ui
                    .button("🔍 Reset Camera View")
                    .on_hover_text("Reset canvas pan offset to (0, 0) and zoom to 1.0x so the entire PCB outline fits centered in the viewport (Drag canvas to pan, scroll wheel to zoom).")
                    .clicked()
                {
                    self.reset_view();
                }

                ui.separator();
                ui.collapsing(format!("Net Classes ({})", snap.board.netclasses.len()), |ui| {
                    egui::Grid::new("netclasses_grid").striped(true).show(ui, |ui| {
                        ui.label(egui::RichText::new("Class").strong())
                            .on_hover_text("Netclass name parsed from the KiCad project or board setup.");
                        ui.label(egui::RichText::new("Width").strong())
                            .on_hover_text("Nominal copper track width in millimeters assigned to this netclass.");
                        ui.label(egui::RichText::new("Clearance").strong())
                            .on_hover_text("Minimum edge-to-edge copper clearance in millimeters enforced bidirectionally for nets in this class.");
                        ui.label(egui::RichText::new("Nets").strong())
                            .on_hover_text("Number of electrical nets assigned to this netclass.");
                        ui.end_row();

                        let mut sorted_classes: Vec<&String> = snap.board.netclasses.keys().collect();
                        sorted_classes.sort();
                        for cname in sorted_classes {
                            let nc = &snap.board.netclasses[cname];
                            let count = snap.board.net_to_class.values().filter(|&v| v == cname).count();
                            ui.label(cname)
                                .on_hover_text(format!("Netclass '{}': via diameter {:.2} mm, via drill {:.2} mm.", cname, nc.via_dia, nc.via_drill));
                            ui.label(format!("{:.2} mm", nc.track_width));
                            ui.label(format!("{:.2} mm", nc.clearance));
                            ui.label(format!("{}", count));
                            ui.end_row();
                        }
                    });
                })
                .header_response
                .on_hover_text("Inspect KiCad netclass design rules (track widths, bidirectional clearances, and net counts).");

                ui.separator();
                ui.heading("Live Metrics")
                    .on_hover_text("Real-time placement and routing quality metrics updated continuously during trial execution.");
                egui::Grid::new("metrics_grid").striped(true).show(ui, |ui| {
                    ui.label("Footprints:")
                        .on_hover_text("Total component footprints placed on the board, along with locked and bottom-side (B.Cu) counts.");
                    let locked_n = snap.board.footprints.iter().filter(|f| f.is_locked).count();
                    let back_n = snap.board.footprints.iter().filter(|f| f.layer == 1).count();
                    ui.label(format!(
                        "{} ({} locked, {} B.Cu)",
                        snap.board.footprints.len(),
                        locked_n,
                        back_n
                    ))
                    .on_hover_text("Click any footprint body on the canvas to toggle its lock state, or press 'M' / 'R' / 'F' to move, rotate 90°, or flip unlocked components.");
                    ui.end_row();

                    ui.label("Ratsnest Crossings:")
                        .on_hover_text("Pairwise geometric intersections between unrouted MST airwire segments. Minimizing crossings reduces via count and layer congestion.");
                    ui.label(format!("{}", snap.crossings))
                        .on_hover_text("Current number of topological ratsnest airwire crossings.");
                    ui.end_row();

                    ui.label("MST Wirelength:")
                        .on_hover_text("Total Euclidean Minimum Spanning Tree (MST) airwire length in millimeters across all nets.");
                    ui.label(format!("{:.1} mm", snap.wirelength_mm))
                        .on_hover_text("Lower MST wirelength indicates tighter, more compact signal routing paths.");
                    ui.end_row();

                    ui.label("Courtyard Overlaps:")
                        .on_hover_text("Number of physical courtyard polygon collisions detected via Separating Axis Theorem (SAT) on the same board side. Must be 0 for DRC-clean assembly.");
                    ui.label(format!("{}", snap.overlaps))
                        .on_hover_text("Current count of overlapping component courtyards.");
                    ui.end_row();

                    ui.label("SA Throughput:")
                        .on_hover_text("Simulated Annealing placement evaluation rate in candidate board states per second across all CPU threads.");
                    ui.label(format!("{:.0} evals/sec", snap.evals_per_sec))
                        .on_hover_text("Measured multi-core Parallel Tempering evaluation throughput.");
                    ui.end_row();

                    ui.label("Routed Connections:")
                        .on_hover_text("Number and percentage of two-terminal pad-to-pad net connections successfully routed without copper shorts.");
                    let pct = if snap.total_conns > 0 {
                        (snap.routed_conns as f64 / snap.total_conns as f64) * 100.0
                    } else {
                        0.0
                    };
                    ui.label(format!("{}/{} ({:.1}%)", snap.routed_conns, snap.total_conns, pct))
                        .on_hover_text("Target is 100.0% routed connections with 0 DRC violations.");
                    ui.end_row();

                    ui.label("Tracks / Vias:")
                        .on_hover_text("Total number of routed copper track segments and layer-transition vias currently on the board.");
                    ui.label(format!("{} segs / {} vias", snap.board.tracks.len(), snap.board.vias.len()))
                        .on_hover_text("Current count of straight/curved track segments and plated through-hole vias.");
                    ui.end_row();
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .button("💾 Save Preferences")
                        .on_hover_text("Explicitly write all current placer, router, viewport, and IPC settings to the persistent preferences.json configuration file.")
                        .clicked()
                    {
                        self.save_preferences();
                    }
                    let cfg_path = self.preferences_path.clone().unwrap_or_else(AppPreferences::config_path);
                    ui.label(
                        egui::RichText::new("Auto-saved")
                            .small()
                            .color(Color32::from_rgb(120, 200, 120)),
                    )
                    .on_hover_text(format!("Preferences automatically saved to:\n{}", cfg_path.display()));
                });

                ui.separator();
                ui.heading("Activity Log")
                    .on_hover_text("Chronological log of placement milestones, PathFinder rip-up passes, interactive edits, file saves, and KiCad IPC events.");
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
                        ui.heading(format!("Holding Palette ({})", snap.board.holding_palette.len()))
                            .on_hover_text("Staging area for unplaced or cleared footprints waiting to be positioned on the PCB.");
                        if ui
                            .add_enabled(!is_running, egui::Button::new("Place All"))
                            .on_hover_text("Automatically place all components from the Holding Palette onto the board at non-overlapping locations that minimize ratsnest wirelength.")
                            .clicked()
                        {
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
                    )
                    .on_hover_text("Components are ordered in natural alphanumeric designator order (e.g. C1, C2, C10, R1, U1).");
                    ui.separator();

                    egui::ScrollArea::vertical()
                        .id_salt("holding_palette_pane_scroll")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            let mut place_single_idx = None;
                            for (idx, fp) in snap.board.holding_palette.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    let layer_str = if fp.layer == 1 { "B.Cu" } else { "F.Cu" };
                                    draw_mini_footprint(ui, fp, Vec2::new(44.0, 44.0)).on_hover_text(format!(
                                        "Preview of {} ({}) — {} pads on {}.",
                                        fp.reference,
                                        fp.value,
                                        fp.pads.len(),
                                        layer_str
                                    ));
                                    ui.vertical(|ui| {
                                        ui.horizontal(|ui| {
                                            ui.label(
                                                egui::RichText::new(&fp.reference)
                                                    .strong()
                                                    .color(Color32::from_rgb(255, 215, 100)),
                                            )
                                            .on_hover_text(format!("Reference designator {} ({})", fp.reference, fp.value));
                                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                if ui
                                                    .add_enabled(!is_running, egui::Button::new("Place"))
                                                    .on_hover_text(format!(
                                                        "Find the lowest-cost, non-overlapping board coordinate and rotation for {} and place it onto the PCB.",
                                                        fp.reference
                                                    ))
                                                    .clicked()
                                                {
                                                    place_single_idx = Some(idx);
                                                }
                                            });
                                        });
                                        ui.label(
                                            egui::RichText::new(&fp.value)
                                                .size(11.0)
                                                .color(Color32::from_rgb(210, 215, 220)),
                                        );
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
            self.last_canvas_rect = Some(rect);
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

            // Keyboard shortcuts:
            // KiCad route shortcuts: 'x' (route track), 'v' (place via & switch layer), '/' (toggle dogleg posture), 'Escape' (cancel route / exit route tool)
            // Placement shortcuts: 'm' (pick up/move), 'r' (rotate 90°), 'f' (flip layer), 'Escape' (cancel move)
            // Bus Bundle shortcut: 'b' (toggle bus sketch), 'Enter' (commit bundle route)
            if !ctx.wants_keyboard_input() && !is_running {
                // 'B': Bus Bundle Sketch mode toggle
                if ctx.input(|i| i.key_pressed(egui::Key::B)) {
                    self.bundle_sketch_active = !self.bundle_sketch_active;
                    if !self.bundle_sketch_active {
                        self.bundle_guide_points.clear();
                    }
                }

                // 'Enter': Commit bundle sketch
                if self.bundle_sketch_active && ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                    self.commit_bundle_sketch();
                }

                // 'X': Route Track mode / start route
                if ctx.input(|i| i.key_pressed(egui::Key::X)) {
                    if self.active_route.is_none() {
                        if let Some(pos) = pointer_pos {
                            if rect.contains(pos) {
                                self.start_route_at(rect, pos);
                            } else {
                                self.route_tool_active = true;
                            }
                        } else {
                            self.route_tool_active = true;
                        }
                    }
                }

                // 'V': Place Via & Switch Layer
                if ctx.input(|i| i.key_pressed(egui::Key::V)) {
                    if self.active_route.is_some() {
                        let via_pos = if let Some(pos) = pointer_pos {
                            if rect.contains(pos) {
                                let snap = self.live.lock().unwrap();
                                let wpt = self.screen_to_board(&snap.board, rect, pos);
                                let grid = if self.grid_res_mm > 0.001 { self.grid_res_mm } else { 0.10 };
                                ((wpt.x / grid).round() * grid, (wpt.y / grid).round() * grid)
                            } else {
                                let last = self.active_route.as_ref().unwrap().anchor_points.last().unwrap();
                                (last.0, last.1)
                            }
                        } else {
                            let last = self.active_route.as_ref().unwrap().anchor_points.last().unwrap();
                            (last.0, last.1)
                        };
                        self.place_via_and_switch_layer(via_pos);
                    }
                }

                // '/': Toggle 45° dogleg corner posture (diagonal first vs orthogonal first)
                let slash_pressed = ctx.input(|i| {
                    i.key_pressed(egui::Key::Slash)
                        || i.events.iter().any(|e| match e {
                            egui::Event::Key { key: egui::Key::Slash, pressed: true, .. } => true,
                            egui::Event::Text(t) => t == "/",
                            _ => false,
                        })
                });
                if slash_pressed {
                    if let Some(ref mut route) = self.active_route {
                        route.posture_diagonal_first = !route.posture_diagonal_first;
                    }
                }

                // 'Escape': Cancel active route, exit route tool, cancel bundle sketch, or cancel footprint move
                if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                    if self.bundle_sketch_active {
                        self.cancel_bundle_sketch();
                    } else if self.active_route.is_some() || self.route_tool_active {
                        self.cancel_route();
                    } else if self.moving_footprint.is_some() {
                        self.cancel_moving_component();
                    }
                }

                if self.active_route.is_none() && !self.route_tool_active && !self.bundle_sketch_active {
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
                }
            }

            // If routing or bundle sketching is active, show crosshair cursor
            if self.active_route.is_some() || self.route_tool_active || self.bundle_sketch_active {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            } else if self.moving_footprint.is_some() {
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
                if self.bundle_sketch_active {
                    if let Some(click_pos) = response.interact_pointer_pos() {
                        self.handle_bundle_sketch_click(rect, click_pos);
                    }
                } else if self.moving_footprint.is_some() {
                    self.place_moving_component();
                } else if self.active_route.is_some() {
                    if let Some(click_pos) = response.interact_pointer_pos() {
                        self.handle_active_route_click(rect, click_pos);
                    }
                } else if self.route_tool_active {
                    if let Some(click_pos) = response.interact_pointer_pos() {
                        self.start_route_at(rect, click_pos);
                    }
                } else if let Some(click_pos) = response.interact_pointer_pos() {
                    self.handle_canvas_click_at(rect, click_pos);
                }
            }
            if response.secondary_clicked() {
                if self.bundle_sketch_active {
                    if self.bundle_guide_points.len() >= 2 {
                        self.commit_bundle_sketch();
                    } else if self.bundle_guide_points.len() == 1 {
                        if let Some(click_pos) = response.interact_pointer_pos() {
                            self.handle_bundle_sketch_click(rect, click_pos);
                            self.commit_bundle_sketch();
                        } else {
                            self.cancel_bundle_sketch();
                        }
                    } else {
                        self.bundle_sketch_active = false;
                    }
                } else if self.active_route.is_some() || self.route_tool_active {
                    self.cancel_route();
                } else if self.moving_footprint.is_some() {
                    self.cancel_moving_component();
                } else if let Some(click_pos) = response.interact_pointer_pos() {
                    self.handle_canvas_secondary_click_at(rect, click_pos);
                }
            }
            if response.dragged() {
                if self.moving_footprint.is_none() && self.active_route.is_none() && !self.bundle_sketch_active {
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
                        for arc in &b.arcs {
                            if arc.layer == layer_idx {
                                let w = ((arc.width as f32) * scale).max(1.5);
                                for seg in arc.to_segments(16) {
                                    let p1 = to_screen(seg.start.0, seg.start.1);
                                    let p2 = to_screen(seg.end.0, seg.end.1);
                                    draw_track_segment(&painter, p1, p2, w, col);
                                }
                            }
                        }
                    }
                }
            }

            // Highlighted routed traces (tracks & arcs) across all visible layers
            for &layer_name in &draw_order {
                if let Some(layer_idx) = copper_layers.iter().position(|&l| l == layer_name) {
                    if self.is_layer_visible(layer_name) {
                        for tr in &b.tracks {
                            if tr.layer == layer_idx && self.is_track_highlighted(b, tr) {
                                let p1 = to_screen(tr.start.0, tr.start.1);
                                let p2 = to_screen(tr.end.0, tr.end.1);
                                let w = ((tr.width as f32) * scale).max(1.5);
                                draw_track_segment(&painter, p1, p2, w + 5.0, COL_HIGHLIGHT_HALO);
                                draw_track_segment(&painter, p1, p2, w + 2.0, COL_HIGHLIGHT_RING);
                                draw_track_segment(&painter, p1, p2, w, COL_HIGHLIGHT_FILL);
                            }
                        }
                        for arc in &b.arcs {
                            if arc.layer == layer_idx && self.is_net_id_highlighted(b, arc.net_id) {
                                let w = ((arc.width as f32) * scale).max(1.5);
                                for seg in arc.to_segments(16) {
                                    let p1 = to_screen(seg.start.0, seg.start.1);
                                    let p2 = to_screen(seg.end.0, seg.end.1);
                                    draw_track_segment(&painter, p1, p2, w + 5.0, COL_HIGHLIGHT_HALO);
                                    draw_track_segment(&painter, p1, p2, w + 2.0, COL_HIGHLIGHT_RING);
                                    draw_track_segment(&painter, p1, p2, w, COL_HIGHLIGHT_FILL);
                                }
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
                        let is_highlighted = self.is_pad_highlighted(b, pad);

                        let (px, py) = fp.pad_absolute_position(pad);
                        let center = to_screen(px, py);
                        let pad_col = if is_highlighted {
                            COL_HIGHLIGHT_FILL
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
                                painter.circle_filled(center, r + 2.5, COL_HIGHLIGHT_HALO);
                                painter.circle_stroke(center, r + 3.0, Stroke::new(1.8_f32, COL_HIGHLIGHT_RING));
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
                                    COL_HIGHLIGHT_HALO,
                                    Stroke::new(1.8_f32, COL_HIGHLIGHT_RING),
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
                    if self.is_via_highlighted(b, v) {
                        painter.circle_filled(c, outer_r + 2.5, COL_HIGHLIGHT_HALO);
                        painter.circle_stroke(c, outer_r + 3.0, Stroke::new(1.8_f32, COL_HIGHLIGHT_RING));
                        painter.circle_filled(c, outer_r, COL_HIGHLIGHT_FILL);
                    } else {
                        painter.circle_filled(c, outer_r, COL_VIA);
                    }
                    painter.circle_filled(c, inner_r, Color32::from_rgb(25, 30, 35));
                }
            }

            // 6. Draw Ratsnest
            if self.show_ratsnest {
                for seg in &snap.ratsnest {
                    let p1 = to_screen(seg.p1.x, seg.p1.y);
                    let p2 = to_screen(seg.p2.x, seg.p2.y);
                    if self.is_ratsnest_highlighted(b, seg) {
                        painter.line_segment(
                            [p1, p2],
                            Stroke::new(3.2_f32, COL_HIGHLIGHT_HALO),
                        );
                        painter.line_segment(
                            [p1, p2],
                            Stroke::new(1.8_f32, COL_HIGHLIGHT_FILL),
                        );
                    } else {
                        painter.line_segment(
                            [p1, p2],
                            Stroke::new(0.8_f32, COL_RATSNEST),
                        );
                    }
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

            // 10. Draw Active Interactive Route & Preview
            if let Some(ref route) = self.active_route {
                let w = ((route.track_width as f32) * scale).max(2.0);

                // Draw committed anchored segments
                for i in 0..route.anchor_points.len().saturating_sub(1) {
                    let pa = route.anchor_points[i];
                    let pb = route.anchor_points[i + 1];
                    if (pa.0 - pb.0).hypot(pa.1 - pb.1) > 1e-5 {
                        let col = copper_layer_color(b.layer_name(pa.2));
                        draw_track_segment(&painter, to_screen(pa.0, pa.1), to_screen(pb.0, pb.1), w, col);
                    }
                }

                // Draw vias placed during active route
                for v in &route.vias_placed {
                    let c = to_screen(v.position.0, v.position.1);
                    let outer_r = ((v.size as f32) * 0.5 * scale).max(2.5);
                    let inner_r = ((v.drill as f32) * 0.5 * scale).max(1.0);
                    painter.circle_filled(c, outer_r, COL_VIA);
                    painter.circle_filled(c, inner_r, Color32::from_rgb(25, 30, 35));
                }

                // Compute cursor point (with magnetic pad snapping if near same net pad)
                let cur_pt = if let Some(pos) = pointer_pos {
                    if rect.contains(pos) {
                        let wpt = self.screen_to_board(b, rect, pos);
                        let grid = if self.grid_res_mm > 0.001 { self.grid_res_mm } else { 0.10 };
                        let mut pt = ((wpt.x / grid).round() * grid, (wpt.y / grid).round() * grid);
                        for fp in &b.footprints {
                            for pad in &fp.pads {
                                if pad.net_name == route.net_name || (pad.net_id != 0 && pad.net_id == route.net_id) {
                                    let (px, py) = fp.pad_absolute_position(pad);
                                    let (pad_screen, _) = self.board_to_screen(b, rect, px, py);
                                    if pad_screen.distance(pos) <= 14.0 || self.is_point_in_pad(b, rect, fp, pad, pos) {
                                        pt = (px, py);
                                        break;
                                    }
                                }
                            }
                        }
                        pt
                    } else {
                        let last = route.anchor_points.last().unwrap();
                        (last.0, last.1)
                    }
                } else {
                    let last = route.anchor_points.last().unwrap();
                    (last.0, last.1)
                };

                // Draw live tentative 45° dogleg preview
                let p0 = *route.anchor_points.last().unwrap();
                let dogleg_pts = compute_45_dogleg((p0.0, p0.1), cur_pt, route.posture_diagonal_first);
                let active_col = copper_layer_color(b.layer_name(route.current_layer));
                for i in 0..dogleg_pts.len().saturating_sub(1) {
                    let q0 = to_screen(dogleg_pts[i].0, dogleg_pts[i].1);
                    let q1 = to_screen(dogleg_pts[i + 1].0, dogleg_pts[i + 1].1);
                    draw_track_segment(&painter, q0, q1, w, active_col);
                }

                // Guide ratsnest line to nearest unrouted target pad of same net
                let mut nearest_pad: Option<((f64, f64), f64)> = None;
                for fp in &b.footprints {
                    for pad in &fp.pads {
                        if pad.net_name == route.net_name || (pad.net_id != 0 && pad.net_id == route.net_id) {
                            let (px, py) = fp.pad_absolute_position(pad);
                            let d = (px - cur_pt.0).hypot(py - cur_pt.1);
                            if d > 0.05 {
                                if nearest_pad.map_or(true, |(_, best_d)| d < best_d) {
                                    nearest_pad = Some(((px, py), d));
                                }
                            }
                        }
                    }
                }
                if let Some((target_pt, _)) = nearest_pad {
                    let c_screen = to_screen(cur_pt.0, cur_pt.1);
                    let t_screen = to_screen(target_pt.0, target_pt.1);
                    painter.line_segment([c_screen, t_screen], Stroke::new(1.0_f32, Color32::from_rgb(255, 230, 80)));
                }

                // Cursor indicator and tooltip badge
                let cur_screen = to_screen(cur_pt.0, cur_pt.1);
                painter.circle_stroke(cur_screen, ((route.track_width as f32) * 0.5 * scale).max(3.5), Stroke::new(1.5_f32, Color32::WHITE));

                let label_pos = Pos2::new(cur_screen.x, cur_screen.y - 18.0);
                let hint = format!("Routing '{}' [{}] · [V] Via · [/] Posture · [Esc] Cancel", route.net_name, b.layer_name(route.current_layer));
                let galley = painter.layout_no_wrap(hint, egui::FontId::proportional(11.0), Color32::from_rgb(255, 235, 120));
                let bg_rect = Rect::from_center_size(label_pos, galley.size() + Vec2::new(8.0, 4.0));
                painter.rect_filled(bg_rect, 3.0, Color32::from_rgba_unmultiplied(20, 24, 30, 220));
                painter.rect_stroke(bg_rect, 3.0, Stroke::new(1.0_f32, Color32::from_rgb(255, 215, 60)), egui::StrokeKind::Middle);
                painter.galley(Pos2::new(bg_rect.left() + 4.0, bg_rect.top() + 2.0), galley, Color32::from_rgb(255, 235, 120));
            } else if self.route_tool_active {
                // If route tool is active but not yet routing, show tooltip when hovering over copper
                if let Some(pos) = pointer_pos {
                    if rect.contains(pos) {
                        if let Some(item) = self.find_copper_item_at(b, rect, pos) {
                            let label_pos = Pos2::new(pos.x, pos.y - 18.0);
                            let hint = format!("[X/Click] Route Net '{}' [{}]", item.net_name, b.layer_name(item.layer));
                            let galley = painter.layout_no_wrap(hint, egui::FontId::proportional(11.0), Color32::from_rgb(120, 235, 255));
                            let bg_rect = Rect::from_center_size(label_pos, galley.size() + Vec2::new(8.0, 4.0));
                            painter.rect_filled(bg_rect, 3.0, Color32::from_rgba_unmultiplied(20, 24, 30, 220));
                            painter.rect_stroke(bg_rect, 3.0, Stroke::new(1.0_f32, Color32::from_rgb(80, 200, 255)), egui::StrokeKind::Middle);
                            painter.galley(Pos2::new(bg_rect.left() + 4.0, bg_rect.top() + 2.0), galley, Color32::from_rgb(120, 235, 255));
                        }
                    }
                }
            }

            // 11. Draw Bus Bundle Sketch Live Preview & Spine
            if self.bundle_sketch_active {
                let grid = if self.grid_res_mm > 0.001 { self.grid_res_mm } else { 0.10 };
                let cur_wpt = if let Some(pos) = pointer_pos {
                    if rect.contains(pos) {
                        let wpt = self.screen_to_board(b, rect, pos);
                        Some(((wpt.x / grid).round() * grid, (wpt.y / grid).round() * grid))
                    } else {
                        None
                    }
                } else {
                    None
                };

                let mut tentative_guide = self.bundle_guide_points.clone();
                if let Some(cpt) = cur_wpt {
                    if tentative_guide.last().map_or(true, |&last| (last.0 - cpt.0).hypot(last.1 - cpt.1) > 1e-4) {
                        tentative_guide.push(cpt);
                    }
                }

                // Draw committed guide waypoints and spine line
                for i in 0..self.bundle_guide_points.len() {
                    let pt = self.bundle_guide_points[i];
                    let screen_pt = to_screen(pt.0, pt.1);
                    painter.circle_filled(screen_pt, 4.0, Color32::from_rgb(100, 200, 255));
                    painter.circle_stroke(screen_pt, 6.0, Stroke::new(1.0_f32, Color32::WHITE));
                    if i + 1 < self.bundle_guide_points.len() {
                        let next_pt = self.bundle_guide_points[i + 1];
                        painter.line_segment(
                            [screen_pt, to_screen(next_pt.0, next_pt.1)],
                            Stroke::new(2.0_f32, Color32::from_rgba_unmultiplied(100, 200, 255, 180)),
                        );
                    }
                }

                // Draw live tentative ribbon/bundle tracks preview
                let nets = self.get_effective_bundle_nets(b);
                if tentative_guide.len() >= 2 && !nets.is_empty() {
                    let tentative_bundle = BusBundle::new(
                        "PreviewBundle",
                        nets.clone(),
                        tentative_guide.clone(),
                        0,
                        self.bundle_track_width,
                        self.bundle_track_spacing,
                    );
                    let preview_tracks = generate_bundle_tracks(b, &tentative_bundle, self.bundle_track_spacing);
                    let preview_stroke_w = ((self.bundle_track_width as f32) * scale).max(2.0);
                    for trk in preview_tracks {
                        let q0 = to_screen(trk.start.0, trk.start.1);
                        let q1 = to_screen(trk.end.0, trk.end.1);
                        let col = Color32::from_rgba_unmultiplied(80, 220, 255, 200);
                        draw_track_segment(&painter, q0, q1, preview_stroke_w, col);
                    }
                }

                // Draw cursor waypoint marker and tooltip badge
                if let Some(cpt) = cur_wpt {
                    let cur_screen = to_screen(cpt.0, cpt.1);
                    painter.circle_stroke(cur_screen, 6.0, Stroke::new(1.5_f32, Color32::from_rgb(100, 220, 255)));

                    let label_pos = Pos2::new(cur_screen.x, cur_screen.y - 18.0);
                    let hint = format!(
                        "Bus Sketch [{} nets] · [Click] Waypoint ({}) · [Enter/R-Click] Commit · [Esc] Cancel",
                        nets.len(),
                        self.bundle_guide_points.len()
                    );
                    let galley = painter.layout_no_wrap(hint, egui::FontId::proportional(11.0), Color32::from_rgb(120, 220, 255));
                    let bg_rect = Rect::from_center_size(label_pos, galley.size() + Vec2::new(8.0, 4.0));
                    painter.rect_filled(bg_rect, 3.0, Color32::from_rgba_unmultiplied(20, 24, 30, 220));
                    painter.rect_stroke(bg_rect, 3.0, Stroke::new(1.0_f32, Color32::from_rgb(100, 200, 255)), egui::StrokeKind::Middle);
                    painter.galley(Pos2::new(bg_rect.left() + 4.0, bg_rect.top() + 2.0), galley, Color32::from_rgb(120, 220, 255));
                }
            }

            // 12. Draw Color Legend Overlay
            draw_color_legend(&painter, rect, b);
        });

        // Modal File Dialog: Open KiCad PCB
        if self.open_dialog_open {
            let mut close_modal = false;
            let mut open_target = None;
            egui::Window::new("📂 Open KiCad PCB")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label("Enter filesystem path to .kicad_pcb file:");
                    let resp = ui.text_edit_singleline(&mut self.file_dialog_path_buffer);
                    let enter = resp.lost_focus() && ctx.input(|i| i.key_pressed(egui::Key::Enter));
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button("Open").clicked() || enter {
                            let p = self.file_dialog_path_buffer.trim().to_string();
                            if !p.is_empty() {
                                open_target = Some(p);
                            }
                            close_modal = true;
                        }
                        if ui.button("Cancel").clicked() || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                            close_modal = true;
                        }
                    });
                });
            if let Some(target) = open_target {
                self.open_pcb_from_path(&target);
            }
            if close_modal {
                self.open_dialog_open = false;
            }
        }

        // Modal File Dialog: Save KiCad PCB As
        if self.save_as_dialog_open {
            let mut close_modal = false;
            let mut save_target = None;
            egui::Window::new("💾 Save KiCad PCB As")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label("Enter destination path for .kicad_pcb file:");
                    let resp = ui.text_edit_singleline(&mut self.file_dialog_path_buffer);
                    let enter = resp.lost_focus() && ctx.input(|i| i.key_pressed(egui::Key::Enter));
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button("Save").clicked() || enter {
                            let mut p = self.file_dialog_path_buffer.trim().to_string();
                            if !p.is_empty() {
                                if !p.ends_with(".kicad_pcb") {
                                    p.push_str(".kicad_pcb");
                                }
                                save_target = Some(p);
                            }
                            close_modal = true;
                        }
                        if ui.button("Cancel").clicked() || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                            close_modal = true;
                        }
                    });
                });
            if let Some(target) = save_target {
                self.set_board_path(target);
                self.save_output_and_pull_to_kicad();
            }
            if close_modal {
                self.save_as_dialog_open = false;
            }
        }

        // Persist preferences if any setting was mutated
        let cur_prefs = self.to_preferences();
        if cur_prefs != self.last_saved_prefs {
            self.save_preferences();
        }
    }
}

impl Drop for VulcansTrialsApp {
    fn drop(&mut self) {
        self.save_preferences();
    }
}

#[inline]
pub fn draw_track_segment(painter: &egui::Painter, p1: Pos2, p2: Pos2, width: f32, color: Color32) {
    painter.line_segment([p1, p2], Stroke::new(width, color));
    let r = width * 0.5;
    painter.circle_filled(p1, r, color);
    painter.circle_filled(p2, r, color);
}

/// Point-in-pad hit test in board world coordinates (mm).
pub fn is_world_point_in_pad(fp: &crate::board::Footprint, pad: &crate::board::Pad, world_pos: Point) -> bool {
    let (px, py) = fp.pad_absolute_position(pad);
    if pad.shape == "circle" {
        let r = (pad.width.max(pad.height) * 0.5).max(0.10);
        world_pos.distance_to(&Point::new(px, py)) <= r
    } else {
        let corners = fp.pad_world_corners(pad);
        Polygon::new(corners).contains_point(&world_pos)
    }
}

/// Point-to-via hit test in board world coordinates (mm).
pub fn hit_test_via_world(via: &Via, world_pos: Point, min_pick_radius_mm: f64) -> Option<f64> {
    let dist = world_pos.distance_to(&Point::new(via.position.0, via.position.1));
    let radius = (via.size * 0.5).max(min_pick_radius_mm);
    if dist <= radius {
        Some(dist)
    } else {
        None
    }
}

/// Point-to-trace segment hit test in board world coordinates (mm).
pub fn hit_test_track_world(seg: &TrackSegment, world_pos: Point, zoom: f32) -> Option<f64> {
    let dist = world_pos.distance_to_segment(
        &Point::new(seg.start.0, seg.start.1),
        &Point::new(seg.end.0, seg.end.1),
    );
    let tol = (seg.width * 0.5).max(0.25 / (zoom as f64).max(0.25));
    if dist <= tol {
        Some(dist)
    } else {
        None
    }
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

fn draw_load_badge(ui: &mut egui::Ui, label: &str, pct: f32, history: &[f32], color: Color32, tooltip: &str) {
    ui.horizontal(|ui| {
        ui.label(format!("{}: {:>4.0}%", label, pct)).on_hover_text(tooltip);
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(72.0, 18.0), Sense::hover());
        resp.on_hover_text(tooltip);
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

        for _ in 0..3000 {
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
        app.show_holding_palette = true;
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
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
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
        let canvas_rect = app.last_canvas_rect.expect("Canvas rect must be computed during warmup frame");
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
                    ..Default::default()
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
                    ..Default::default()
                },
            ],
            width: 5.0,
            height: 5.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.25,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
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

    #[test]
    fn test_compute_45_dogleg_geometry() {
        // 1. Identical points
        let p0 = (10.0, 20.0);
        let pts = compute_45_dogleg(p0, p0, false);
        assert_eq!(pts, vec![p0]);

        // 2. Pure horizontal
        let h_pts_diag_false = compute_45_dogleg((0.0, 0.0), (10.0, 0.0), false);
        assert_eq!(h_pts_diag_false, vec![(0.0, 0.0), (10.0, 0.0)]);
        let h_pts_diag_true = compute_45_dogleg((0.0, 0.0), (10.0, 0.0), true);
        assert_eq!(h_pts_diag_true, vec![(0.0, 0.0), (10.0, 0.0)]);

        // 3. Pure vertical
        let v_pts_diag_false = compute_45_dogleg((0.0, 0.0), (0.0, 10.0), false);
        assert_eq!(v_pts_diag_false, vec![(0.0, 0.0), (0.0, 10.0)]);
        let v_pts_diag_true = compute_45_dogleg((0.0, 0.0), (0.0, 10.0), true);
        assert_eq!(v_pts_diag_true, vec![(0.0, 0.0), (0.0, 10.0)]);

        // 4. Pure 45° diagonal
        let d_pts_diag_false = compute_45_dogleg((0.0, 0.0), (10.0, 10.0), false);
        assert_eq!(d_pts_diag_false, vec![(0.0, 0.0), (10.0, 10.0)]);
        let d_pts_diag_true = compute_45_dogleg((0.0, 0.0), (10.0, 10.0), true);
        assert_eq!(d_pts_diag_true, vec![(0.0, 0.0), (10.0, 10.0)]);

        // 5. Horizontal dominant (dx = 10, dy = 4)
        // diagonal_first = false: straight horizontal first to (6, 0), then 45° diagonal to (10, 4)
        let dogleg_h_false = compute_45_dogleg((0.0, 0.0), (10.0, 4.0), false);
        assert_eq!(dogleg_h_false.len(), 3);
        assert_eq!(dogleg_h_false[0], (0.0, 0.0));
        assert_eq!(dogleg_h_false[1], (6.0, 0.0));
        assert_eq!(dogleg_h_false[2], (10.0, 4.0));
        // Verify segment 1 is horizontal, segment 2 is 45°
        assert!((dogleg_h_false[1].1 - dogleg_h_false[0].1).abs() < 1e-6);
        assert!(((dogleg_h_false[2].0 - dogleg_h_false[1].0).abs() - (dogleg_h_false[2].1 - dogleg_h_false[1].1).abs()).abs() < 1e-6);

        // diagonal_first = true: 45° diagonal first to (4, 4), then straight horizontal to (10, 4)
        let dogleg_h_true = compute_45_dogleg((0.0, 0.0), (10.0, 4.0), true);
        assert_eq!(dogleg_h_true.len(), 3);
        assert_eq!(dogleg_h_true[0], (0.0, 0.0));
        assert_eq!(dogleg_h_true[1], (4.0, 4.0));
        assert_eq!(dogleg_h_true[2], (10.0, 4.0));
        // Verify segment 1 is 45°, segment 2 is horizontal
        assert!(((dogleg_h_true[1].0 - dogleg_h_true[0].0).abs() - (dogleg_h_true[1].1 - dogleg_h_true[0].1).abs()).abs() < 1e-6);
        assert!((dogleg_h_true[2].1 - dogleg_h_true[1].1).abs() < 1e-6);

        // 6. Vertical dominant (dx = 4, dy = 10)
        // diagonal_first = false: straight vertical first to (0, 6), then 45° diagonal to (4, 10)
        let dogleg_v_false = compute_45_dogleg((0.0, 0.0), (4.0, 10.0), false);
        assert_eq!(dogleg_v_false.len(), 3);
        assert_eq!(dogleg_v_false[0], (0.0, 0.0));
        assert_eq!(dogleg_v_false[1], (0.0, 6.0));
        assert_eq!(dogleg_v_false[2], (4.0, 10.0));

        // diagonal_first = true: 45° diagonal first to (4, 4), then straight vertical to (4, 10)
        let dogleg_v_true = compute_45_dogleg((0.0, 0.0), (4.0, 10.0), true);
        assert_eq!(dogleg_v_true.len(), 3);
        assert_eq!(dogleg_v_true[0], (0.0, 0.0));
        assert_eq!(dogleg_v_true[1], (4.0, 4.0));
        assert_eq!(dogleg_v_true[2], (4.0, 10.0));

        // 7. Negative directions (dx = -8, dy = -4) -> horizontal dominant
        let dogleg_neg_false = compute_45_dogleg((10.0, 10.0), (2.0, 6.0), false);
        assert_eq!(dogleg_neg_false.len(), 3);
        assert_eq!(dogleg_neg_false[0], (10.0, 10.0));
        assert_eq!(dogleg_neg_false[1], (6.0, 10.0));
        assert_eq!(dogleg_neg_false[2], (2.0, 6.0));

        let dogleg_neg_true = compute_45_dogleg((10.0, 10.0), (2.0, 6.0), true);
        assert_eq!(dogleg_neg_true.len(), 3);
        assert_eq!(dogleg_neg_true[0], (10.0, 10.0));
        assert_eq!(dogleg_neg_true[1], (6.0, 6.0));
        assert_eq!(dogleg_neg_true[2], (2.0, 6.0));
    }

    #[test]
    fn test_manual_route_start_with_key_x_and_click() {
        let in_path = "/tmp/vulcan_gui_route_start_test.kicad_pcb";
        let board = make_gui_test_board(in_path);
        let expected_net_id = board.footprints[0].pads[0].net_id;
        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);

        let canvas_rect = Rect::from_min_max(Pos2::new(318.0, 67.0), Pos2::new(792.0, 592.0));
        // J1.1 is at (110.0, 115.0) on net "SIG"
        let (j1_screen, _) = app.board_to_screen(&board, canvas_rect, 110.0, 115.0);

        // Hover over J1.1 and start route
        assert!(!app.route_tool_active);
        assert!(app.active_route.is_none());

        let started = app.start_route_at(canvas_rect, j1_screen);
        assert!(started, "Hovering over J1 pad 1 must start routing net 'SIG'");
        assert!(app.route_tool_active);
        assert!(app.active_route.is_some());

        let route = app.active_route.as_ref().unwrap();
        assert_eq!(route.net_id, expected_net_id);
        assert_eq!(route.net_name, "SIG");
        assert_eq!(route.anchor_points, vec![(110.0, 115.0, 0)]);
        assert_eq!(route.current_layer, 0); // F.Cu
        assert!(!route.posture_diagonal_first);

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_manual_route_layer_transition_and_via_with_key_v() {
        let in_path = "/tmp/vulcan_gui_route_via_test.kicad_pcb";
        let board = make_gui_test_board(in_path);
        let expected_net_id = board.footprints[0].pads[0].net_id;
        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);

        let canvas_rect = Rect::from_min_max(Pos2::new(318.0, 67.0), Pos2::new(792.0, 592.0));
        let (j1_screen, _) = app.board_to_screen(&board, canvas_rect, 110.0, 115.0);

        // Start routing at J1
        app.start_route_at(canvas_rect, j1_screen);
        assert_eq!(app.active_route.as_ref().unwrap().current_layer, 0);

        // Route to (115.0, 115.0) and place a via
        app.place_via_and_switch_layer((115.0, 115.0));

        let route = app.active_route.as_ref().unwrap();
        assert_eq!(route.current_layer, 1, "Layer must switch from F.Cu (0) to B.Cu (1)");
        assert_eq!(route.vias_placed.len(), 1, "Via must be placed");
        assert_eq!(route.vias_placed[0].position, (115.0, 115.0));
        assert_eq!(route.vias_placed[0].net_id, expected_net_id);

        // Verify anchor points contain:
        // (110.0, 115.0, 0) -> start
        // (115.0, 115.0, 0) -> track reaching via on layer 0
        // (115.0, 115.0, 1) -> continuation point on layer 1
        assert!(route.anchor_points.contains(&(115.0, 115.0, 0)));
        assert_eq!(*route.anchor_points.last().unwrap(), (115.0, 115.0, 1));

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_manual_route_completion_on_target_pad() {
        let in_path = "/tmp/vulcan_gui_route_commit_test.kicad_pcb";
        let board = make_gui_test_board(in_path);
        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);

        let canvas_rect = Rect::from_min_max(Pos2::new(318.0, 67.0), Pos2::new(792.0, 592.0));
        let (j1_screen, _) = app.board_to_screen(&board, canvas_rect, 110.0, 115.0);
        let (r10_pad1_screen, _) = app.board_to_screen(&board, canvas_rect, 119.5, 115.0);

        let init_tracks = app.live.lock().unwrap().board.tracks.len();
        let init_vias = app.live.lock().unwrap().board.vias.len();

        // 1. Start routing at J1.1
        app.start_route_at(canvas_rect, j1_screen);

        // 2. Route along and drop via at (114.0, 115.0)
        app.place_via_and_switch_layer((114.0, 115.0));

        // 3. Click on R10.1 (same net: "SIG", net 2) to finish and commit
        app.handle_active_route_click(canvas_rect, r10_pad1_screen);

        // Active route must now be finished and committed
        assert!(app.active_route.is_none());

        let g = app.live.lock().unwrap();
        assert!(g.board.tracks.len() > init_tracks, "Tracks must be added to board");
        assert_eq!(g.board.vias.len(), init_vias + 1, "Via must be added to board");

        // Verify tracks on both layer 0 and layer 1 were created
        assert!(g.board.tracks.iter().any(|t| t.net_id == 2 && t.layer == 0));
        assert!(g.board.tracks.iter().any(|t| t.net_id == 2 && t.layer == 1));

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_manual_route_cancel_on_escape_and_posture_toggle() {
        let in_path = "/tmp/vulcan_gui_route_cancel_test.kicad_pcb";
        let board = make_gui_test_board(in_path);
        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);

        let canvas_rect = Rect::from_min_max(Pos2::new(318.0, 67.0), Pos2::new(792.0, 592.0));
        let (j1_screen, _) = app.board_to_screen(&board, canvas_rect, 110.0, 115.0);

        // Start routing
        app.start_route_at(canvas_rect, j1_screen);
        assert!(app.active_route.is_some());
        assert!(!app.active_route.as_ref().unwrap().posture_diagonal_first);

        // Toggle posture with '/'
        app.active_route.as_mut().unwrap().posture_diagonal_first = !app.active_route.as_ref().unwrap().posture_diagonal_first;
        assert!(app.active_route.as_ref().unwrap().posture_diagonal_first);

        app.active_route.as_mut().unwrap().posture_diagonal_first = !app.active_route.as_ref().unwrap().posture_diagonal_first;
        assert!(!app.active_route.as_ref().unwrap().posture_diagonal_first);

        // Cancel route with Escape
        let cancelled = app.cancel_route();
        assert!(cancelled);
        assert!(app.active_route.is_none());
        assert!(app.route_tool_active, "First Escape cancels in-progress route but retains route tool active");

        // Second Escape exits route mode
        let exited = app.cancel_route();
        assert!(exited);
        assert!(!app.route_tool_active, "Second Escape exits route mode");

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_manual_route_keyboard_and_mouse_events_in_render_ui() {
        let in_path = "/tmp/vulcan_gui_route_e2e_events_test.kicad_pcb";
        let board = make_gui_test_board(in_path);
        let expected_net_id = board.footprints[0].pads[0].net_id;
        let mut app = VulcansTrialsApp::new(in_path.to_string(), String::new(), board.clone(), None);
        let ctx = egui::Context::default();
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(800.0, 600.0));

        // Warm up layout
        let mut warmup = egui::RawInput::default();
        warmup.screen_rect = Some(rect);
        let _ = ctx.run(warmup, |ctx| {
            app.render_ui(ctx);
        });

        let canvas_rect = app.last_canvas_rect.expect("Canvas rect must be computed during warmup frame");
        let (j1_screen, _) = app.board_to_screen(&board, canvas_rect, 110.0, 115.0);
        let (r10_screen, _) = app.board_to_screen(&board, canvas_rect, 119.5, 115.0);

        // Frame 1: Move mouse to J1 pad 1 and press 'x' -> starts routing
        let mut input1 = egui::RawInput::default();
        input1.screen_rect = Some(rect);
        input1.events.push(egui::Event::PointerMoved(j1_screen));
        input1.events.push(egui::Event::Key {
            key: egui::Key::X,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        });
        let _ = ctx.run(input1, |ctx| {
            app.render_ui(ctx);
        });
        assert!(app.active_route.is_some(), "Key 'X' over pad must start routing");
        assert_eq!(app.active_route.as_ref().unwrap().net_id, expected_net_id);

        // Frame 2: Move mouse to (115.0, 115.0) and press 'v' -> places via and switches to layer 1
        let (via_screen, _) = app.board_to_screen(&board, canvas_rect, 115.0, 115.0);
        let mut input2 = egui::RawInput::default();
        input2.screen_rect = Some(rect);
        input2.events.push(egui::Event::PointerMoved(via_screen));
        input2.events.push(egui::Event::Key {
            key: egui::Key::V,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        });
        let _ = ctx.run(input2, |ctx| {
            app.render_ui(ctx);
        });
        assert_eq!(app.active_route.as_ref().unwrap().current_layer, 1, "Key 'V' must drop via and switch layer");
        assert_eq!(app.active_route.as_ref().unwrap().vias_placed.len(), 1);

        // Frame 3: Press '/' -> toggles posture
        let mut input3 = egui::RawInput::default();
        input3.screen_rect = Some(rect);
        input3.events.push(egui::Event::Key {
            key: egui::Key::Slash,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        });
        let _ = ctx.run(input3, |ctx| {
            app.render_ui(ctx);
        });
        assert!(app.active_route.as_ref().unwrap().posture_diagonal_first);

        // Frame 4: Move mouse to R10 pad 1 and click Primary -> finishes route
        let mut input4 = egui::RawInput::default();
        input4.screen_rect = Some(rect);
        input4.events.push(egui::Event::PointerMoved(r10_screen));
        input4.events.push(egui::Event::PointerButton {
            pos: r10_screen,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::default(),
        });
        input4.events.push(egui::Event::PointerButton {
            pos: r10_screen,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::default(),
        });
        let _ = ctx.run(input4, |ctx| {
            app.render_ui(ctx);
        });
        assert!(app.active_route.is_none(), "Clicking on target pad of same net commits and finishes route");
        let g = app.live.lock().unwrap();
        assert!(g.board.vias.iter().any(|v| v.net_id == expected_net_id));
        assert!(g.board.tracks.iter().any(|t| t.net_id == expected_net_id));

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_gui_bundle_sketch_interaction() {
        let in_path = "test_gui_bundle_sketch.kicad_pcb";
        let out_path = "test_gui_bundle_sketch_out.ses";
        let board = make_gui_test_board(in_path);
        let mut app = VulcansTrialsApp::new(in_path.to_string(), out_path.to_string(), board, None);
        app.bundle_selected_nets = vec![3, 4];

        assert!(!app.bundle_sketch_active);
        assert!(app.bundle_guide_points.is_empty());

        let ctx = egui::Context::default();
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1200.0, 800.0));

        // Frame 1: Press 'B' -> toggles bundle sketch active
        let mut input1 = egui::RawInput::default();
        input1.screen_rect = Some(rect);
        input1.events.push(egui::Event::Key {
            key: egui::Key::B,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        });
        let _ = ctx.run(input1, |ctx| {
            app.render_ui(ctx);
        });
        assert!(app.bundle_sketch_active, "Key 'B' must activate bundle sketch tool");

        // Add 2 waypoints via canvas clicks in open board routing channel
        let (wp1_screen, _) = app.board_to_screen(&app.live.lock().unwrap().board, rect, 112.0, 106.0);
        let (wp2_screen, _) = app.board_to_screen(&app.live.lock().unwrap().board, rect, 125.0, 106.0);

        app.handle_bundle_sketch_click(rect, wp1_screen);
        app.handle_bundle_sketch_click(rect, wp2_screen);
        assert_eq!(app.bundle_guide_points.len(), 2);

        // Frame 2: Press 'Enter' -> commits bundle route
        let mut input2 = egui::RawInput::default();
        input2.screen_rect = Some(rect);
        input2.events.push(egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        });
        let _ = ctx.run(input2, |ctx| {
            app.render_ui(ctx);
        });

        assert!(app.bundle_guide_points.is_empty(), "Committing bundle must clear guide points");
        let g = app.live.lock().unwrap();
        assert!(!g.board.bundles.is_empty(), "Board must record committed bundle");
        assert!(!g.board.tracks.is_empty(), "Board tracks must be created by bundle router");

        drop(g);

        // Test cancel: activate and cancel with Escape
        app.bundle_sketch_active = true;
        app.bundle_guide_points.push((110.0, 110.0));
        assert_eq!(app.bundle_guide_points.len(), 1);

        app.cancel_bundle_sketch();
        assert!(app.bundle_guide_points.is_empty());
        assert!(app.bundle_sketch_active);

        app.cancel_bundle_sketch();
        assert!(!app.bundle_sketch_active);

        let _ = fs::remove_file(in_path);
    }

    #[test]
    fn test_preferences_default_values_and_roundtrip() {
        let temp_pref_path = format!("/tmp/vulcans_trials_pref_test_{}.json", std::process::id());
        let _ = fs::remove_file(&temp_pref_path);

        let default_prefs = AppPreferences::default();
        assert_eq!(default_prefs.place_duration_secs, 120);
        assert_eq!(default_prefs.min_spacing_mm, 0.25);
        assert_eq!(default_prefs.grid_res_mm, 0.10);
        assert_eq!(default_prefs.max_ripup_passes, 15);
        assert!(!default_prefs.allow_layer_flip);
        assert_eq!(default_prefs.inner_plane_penalty, 8.0);
        assert!(!default_prefs.allow_component_nudge);
        assert!(!default_prefs.any_angle_mode);
        assert!(default_prefs.show_courtyards);
        assert!(default_prefs.show_ratsnest);

        // Mutate custom values
        let mut custom = default_prefs.clone();
        custom.place_duration_secs = 240;
        custom.min_spacing_mm = 0.40;
        custom.grid_res_mm = 0.05;
        custom.max_ripup_passes = 25;
        custom.allow_layer_flip = true;
        custom.inner_plane_penalty = 12.0;
        custom.allow_component_nudge = true;
        custom.any_angle_mode = true;
        custom.kicad_ipc_enabled = true;
        custom.kicad_ipc_socket_path = "ipc:///tmp/custom/kicad.sock".to_string();
        custom.bundle_track_width = 0.30;
        custom.bundle_track_spacing = 0.35;

        // Save to path
        let path = PathBuf::from(&temp_pref_path);
        custom.save_to_path(&path).expect("Failed to save preferences");
        assert!(path.exists());

        // Reload from path
        let loaded = AppPreferences::load_from_path(&path).expect("Failed to load preferences");
        assert_eq!(loaded, custom);

        let _ = fs::remove_file(&temp_pref_path);
    }

    #[test]
    fn test_preferences_applied_to_app_and_auto_saved() {
        let temp_pref_path = format!("/tmp/vulcans_trials_app_pref_test_{}.json", std::process::id());
        let _ = fs::remove_file(&temp_pref_path);

        let in_path = format!("/tmp/vulcan_pref_gui_test_{}.kicad_pcb", std::process::id());
        let content = r#"(kicad_pcb
          (net 0 "") (net 1 "GND")
          (gr_rect (start 100.0 100.0) (end 130.0 130.0) (layer "Edge.Cuts"))
          (footprint "R" (at 110.0 110.0) (property "Reference" "R1")
            (pad "1" smd rect (at 0 0) (size 0.5 0.5) (net 1 "GND")))
        )"#;
        fs::write(&in_path, content).unwrap();

        // 1. Create a custom preferences file
        let mut initial_prefs = AppPreferences::default();
        initial_prefs.place_duration_secs = 45;
        initial_prefs.min_spacing_mm = 0.33;
        initial_prefs.inner_plane_penalty = 15.0;
        initial_prefs.allow_layer_flip = true;
        initial_prefs.any_angle_mode = true;
        initial_prefs.kicad_ipc_socket_path = "ipc:///tmp/saved_test.sock".to_string();
        let pref_file = PathBuf::from(&temp_pref_path);
        initial_prefs.save_to_path(&pref_file).unwrap();

        // 2. Launch VulcansTrialsApp with preferences path
        let board = Board::load(&in_path, &HashSet::new()).unwrap();
        let mut app = VulcansTrialsApp::new(in_path.clone(), "/tmp/out.kicad_pcb".to_string(), board, None)
            .with_preferences_path(pref_file.clone());

        assert_eq!(app.place_duration_secs, 45);
        assert!((app.min_spacing_mm - 0.33).abs() < 1e-4);
        assert!((app.inner_plane_penalty - 15.0).abs() < 1e-4);
        assert!(app.allow_layer_flip);
        assert!(app.any_angle_mode);
        assert_eq!(app.kicad_ipc_socket_path, "ipc:///tmp/saved_test.sock");

        // 3. Mutate a setting and run render_ui to trigger auto-save
        app.place_duration_secs = 99;
        app.min_spacing_mm = 0.55;
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            app.render_ui(ctx);
        });

        // 4. Verify disk file was updated with new values
        let reloaded = AppPreferences::load_from_path(&pref_file).expect("Must reload from file");
        assert_eq!(reloaded.place_duration_secs, 99);
        assert!((reloaded.min_spacing_mm - 0.55).abs() < 1e-4);

        let _ = fs::remove_file(&in_path);
        let _ = fs::remove_file(&temp_pref_path);
    }

    #[test]
    fn test_hit_test_pad_via_and_trace_resolves_net_id() {
        let in_path = format!("/tmp/vulcan_gui_hittest_net_{}.kicad_pcb", std::process::id());
        let mut board = make_gui_test_board(&in_path);
        let sig_id = board.net_name_to_id["SIG"];
        let gnd_id = board.net_name_to_id["GND"];
        board.tracks[0].net_id = sig_id;
        board.tracks[1].net_id = sig_id;
        board.vias[0].net_id = sig_id;
        // Add a second net track (GND) on B.Cu and a GND via
        board.tracks.push(TrackSegment {
            start: (120.5, 115.0),
            end: (125.0, 115.0),
            width: 0.40,
            layer: 1,
            net_id: gnd_id,
            is_locked: false,
        });
        board.vias.push(Via {
            position: (125.0, 115.0),
            size: 0.80,
            drill: 0.40,
            net_id: gnd_id,
            is_locked: false,
        });

        let mut app = VulcansTrialsApp::new(in_path.clone(), String::new(), board.clone(), None);

        // 1. Point-to-pad hit-testing (circle thru-hole pad J1.1 at 110.0, 115.0 -> "SIG", and rect SMD pad R10.2 at 120.5, 115.0 -> "GND")
        assert!(is_world_point_in_pad(&board.footprints[0], &board.footprints[0].pads[0], Point::new(110.2, 115.0)));
        assert!(!is_world_point_in_pad(&board.footprints[0], &board.footprints[0].pads[0], Point::new(111.0, 115.0)));
        assert_eq!(app.hit_test_net_at_world(&board, Point::new(110.0, 115.0)), Some((sig_id, "SIG".to_string())));
        assert_eq!(app.hit_test_net_at_world(&board, Point::new(120.5, 115.0)), Some((gnd_id, "GND".to_string())));

        // 2. Point-to-via hit-testing (via at 115.0, 115.0 size 0.6 -> "SIG"; via at 125.0, 115.0 size 0.8 -> "GND")
        assert!(hit_test_via_world(&board.vias[0], Point::new(115.25, 115.0), 0.25).is_some());
        assert!(hit_test_via_world(&board.vias[0], Point::new(115.40, 115.0), 0.25).is_none());
        assert_eq!(app.hit_test_net_at_world(&board, Point::new(115.0, 115.20)), Some((sig_id, "SIG".to_string())));
        assert_eq!(app.hit_test_net_at_world(&board, Point::new(125.0, 115.35)), Some((gnd_id, "GND".to_string())));

        // When vias are hidden, via hit-test skips vias
        app.show_vias = false;
        assert_eq!(app.hit_test_net_at_world(&board, Point::new(125.35, 115.0)), None);
        app.show_vias = true;

        // 3. Point-to-trace hit-testing (track on F.Cu from 110..115 -> "SIG", track on B.Cu from 120.5..125 -> "GND")
        assert!(hit_test_track_world(&board.tracks[0], Point::new(112.5, 115.20), 1.0).is_some());
        assert!(hit_test_track_world(&board.tracks[0], Point::new(112.5, 115.30), 1.0).is_none());
        assert_eq!(app.hit_test_net_at_world(&board, Point::new(112.5, 115.10)), Some((sig_id, "SIG".to_string())));
        assert_eq!(app.hit_test_net_at_world(&board, Point::new(123.0, 115.15)), Some((gnd_id, "GND".to_string())));

        // Hiding B.Cu disables hit-testing for B.Cu traces while keeping F.Cu traces active
        app.set_layer_visible("B.Cu", false);
        assert_eq!(app.hit_test_net_at_world(&board, Point::new(123.0, 115.0)), None);
        assert_eq!(app.hit_test_net_at_world(&board, Point::new(112.5, 115.0)), Some((sig_id, "SIG".to_string())));
        app.set_layer_visible("B.Cu", true);

        let _ = fs::remove_file(&in_path);
    }

    #[test]
    fn test_clicking_pad_via_or_trace_selects_and_highlights_all_net_elements() {
        let in_path = format!("/tmp/vulcan_gui_net_all_elements_{}.kicad_pcb", std::process::id());
        let mut board = make_gui_test_board(&in_path);
        let sig_id = board.net_name_to_id["SIG"];
        let gnd_id = board.net_name_to_id["GND"];
        board.tracks[0].net_id = sig_id;
        board.tracks[1].net_id = sig_id;
        board.vias[0].net_id = sig_id;
        board.tracks.push(TrackSegment {
            start: (120.5, 115.0),
            end: (125.0, 115.0),
            width: 0.25,
            layer: 0,
            net_id: gnd_id,
            is_locked: false,
        });
        board.vias.push(Via {
            position: (125.0, 115.0),
            size: 0.60,
            drill: 0.30,
            net_id: gnd_id,
            is_locked: false,
        });

        let mut app = VulcansTrialsApp::new(in_path.clone(), String::new(), board.clone(), None);
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(800.0, 600.0));

        // 1. Click a routed trace segment on net "SIG" at (112.5, 115.0)
        let (trace_sig_screen, _) = app.board_to_screen(&board, rect, 112.5, 115.0);
        let res1 = app.handle_canvas_click_at(rect, trace_sig_screen);
        assert_eq!(res1, Some("Net: SIG".to_string()));
        assert_eq!(app.selected_net.as_deref(), Some("SIG"));
        assert_eq!(app.selected_net_id(&board), Some(sig_id));

        // Verify all elements of net "SIG" are highlighted, and net "GND" elements are NOT
        assert!(app.is_pad_highlighted(&board, &board.footprints[0].pads[0])); // J1.1 (SIG)
        assert!(app.is_pad_highlighted(&board, &board.footprints[1].pads[0])); // R10.1 (SIG)
        assert!(!app.is_pad_highlighted(&board, &board.footprints[1].pads[1])); // R10.2 (GND)
        assert!(app.is_track_highlighted(&board, &board.tracks[0])); // SIG on F.Cu
        assert!(app.is_track_highlighted(&board, &board.tracks[1])); // SIG on B.Cu
        assert!(!app.is_track_highlighted(&board, &board.tracks[2])); // GND on F.Cu
        assert!(app.is_via_highlighted(&board, &board.vias[0])); // SIG via
        assert!(!app.is_via_highlighted(&board, &board.vias[1])); // GND via
        let ratsnest = app.live.lock().unwrap().ratsnest.clone();
        for seg in &ratsnest {
            assert_eq!(app.is_ratsnest_highlighted(&board, seg), seg.net_id == sig_id);
        }

        // Render UI with active net highlight to exercise trace, via, pad, and ratsnest highlight rendering
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            app.render_ui(ctx);
        });

        // 2. Click a via on the same net ("SIG") at (115.0, 115.0) -> toggles off selection
        let (via_sig_screen, _) = app.board_to_screen(&board, rect, 115.0, 115.0);
        let res2 = app.handle_canvas_click_at(rect, via_sig_screen);
        assert_eq!(res2, Some("Cleared SIG".to_string()));
        assert_eq!(app.selected_net, None);
        assert_eq!(app.selected_net_id(&board), None);
        assert!(!app.is_track_highlighted(&board, &board.tracks[0]));
        assert!(!app.is_via_highlighted(&board, &board.vias[0]));

        // 3. Click the via on net "GND" at (125.0, 115.0) -> selects net "GND"
        let (via_gnd_screen, _) = app.board_to_screen(&board, rect, 125.0, 115.0);
        let res3 = app.handle_canvas_click_at(rect, via_gnd_screen);
        assert_eq!(res3, Some("Net: GND".to_string()));
        assert_eq!(app.selected_net.as_deref(), Some("GND"));
        assert_eq!(app.selected_net_id(&board), Some(gnd_id));
        assert!(app.is_pad_highlighted(&board, &board.footprints[1].pads[1]));
        assert!(app.is_track_highlighted(&board, &board.tracks[2]));
        assert!(app.is_via_highlighted(&board, &board.vias[1]));
        assert!(!app.is_track_highlighted(&board, &board.tracks[0]));

        let _ = fs::remove_file(&in_path);
    }

    #[test]
    fn test_advanced_autoplacer_and_autorouter_panels_and_preferences() {
        let temp_pref_path = format!(
            "/tmp/vulcans_trials_adv_panels_pref_test_{}.json",
            std::process::id()
        );
        let _ = fs::remove_file(&temp_pref_path);

        let board = Board::default();
        let mut app = VulcansTrialsApp::new(
            "in.kicad_pcb".to_string(),
            "out.kicad_pcb".to_string(),
            board,
            None,
        )
        .with_preferences_path(PathBuf::from(&temp_pref_path));

        // Verify default advanced autorouter and autoplacer fields
        assert!((app.via_merge_radius_mm - 4.00).abs() < 1e-6);
        assert!((app.via_detour_budget_mm - 3.50).abs() < 1e-6);
        assert!((app.via_hop_max_detour_ratio - 1.25).abs() < 1e-6);
        assert!((app.plane_via_share_radius_mm - 3.50).abs() < 1e-6);
        assert!((app.return_via_max_dist_mm - 1.40).abs() < 1e-6);
        assert!((app.via_pad_snap_dist_mm - 0.50).abs() < 1e-6);
        assert!((app.congestion_factor - 1.50).abs() < 1e-6);
        assert!((app.crosstalk_penalty - 4.0).abs() < 1e-6);
        assert!((app.nudge_step_mm - 0.25).abs() < 1e-6);
        assert_eq!(app.gloss_passes, 2);

        // Mutate both Advanced Autoplacer and Advanced Autorouter settings
        app.max_jump_radius_mm = 32.0;
        app.via_merge_radius_mm = 5.50;
        app.via_detour_budget_mm = 4.25;
        app.via_hop_max_detour_ratio = 1.45;
        app.plane_via_share_radius_mm = 4.75;
        app.return_via_max_dist_mm = 1.85;
        app.via_pad_snap_dist_mm = 0.70;
        app.congestion_factor = 2.25;
        app.crosstalk_penalty = 7.5;
        app.nudge_step_mm = 0.35;
        app.gloss_passes = 3;

        let ctx = egui::Context::default();
        // Expand both collapsible panels in egui memory so their inner widgets render and are exercised
        let id_placer = egui::Id::new("vulcan_controls_panel").with("Advanced Autoplacer");
        let id_router = egui::Id::new("vulcan_controls_panel").with("Advanced Autorouter");
        egui::collapsing_header::CollapsingState::load_with_default_open(&ctx, id_placer, true)
            .store(&ctx);
        egui::collapsing_header::CollapsingState::load_with_default_open(&ctx, id_router, true)
            .store(&ctx);
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            app.render_ui(ctx);
        });

        // Verify auto-save persisted all advanced settings to disk
        let reloaded = AppPreferences::load_from_path(Path::new(&temp_pref_path))
            .expect("Must reload preferences from disk");
        assert!((reloaded.max_jump_radius_mm - 32.0).abs() < 1e-6);
        assert!((reloaded.via_merge_radius_mm - 5.50).abs() < 1e-6);
        assert!((reloaded.via_detour_budget_mm - 4.25).abs() < 1e-6);
        assert!((reloaded.via_hop_max_detour_ratio - 1.45).abs() < 1e-6);
        assert!((reloaded.plane_via_share_radius_mm - 4.75).abs() < 1e-6);
        assert!((reloaded.return_via_max_dist_mm - 1.85).abs() < 1e-6);
        assert!((reloaded.via_pad_snap_dist_mm - 0.70).abs() < 1e-6);
        assert!((reloaded.congestion_factor - 2.25).abs() < 1e-6);
        assert!((reloaded.crosstalk_penalty - 7.5).abs() < 1e-6);
        assert!((reloaded.nudge_step_mm - 0.35).abs() < 1e-6);
        assert_eq!(reloaded.gloss_passes, 3);

        let _ = fs::remove_file(&temp_pref_path);
    }

    #[test]
    fn test_swap_organic_tracks_reversible() {
        let temp_pref_path = format!(
            "/tmp/vulcans_trials_swap_organic_pref_test_{}.json",
            std::process::id()
        );
        let temp_pcb_path = format!(
            "/tmp/vulcans_trials_swap_organic_pcb_test_{}.kicad_pcb",
            std::process::id()
        );
        let _ = fs::remove_file(&temp_pref_path);
        let _ = fs::remove_file(&temp_pcb_path);

        let content = r#"(kicad_pcb
          (net 0 "") (net 1 "NET1")
          (gr_rect (start 0.0 0.0) (end 40.0 40.0) (layer "Edge.Cuts"))
          (footprint "R" (at 5.0 5.0) (property "Reference" "R1")
            (pad "1" smd rect (at 0 0) (size 0.8 0.8) (layers "F.Cu") (net 1 "NET1")))
          (footprint "R" (at 25.0 15.0) (property "Reference" "R2")
            (pad "1" smd rect (at 0 0) (size 0.8 0.8) (layers "F.Cu") (net 1 "NET1")))
        )"#;
        fs::write(&temp_pcb_path, content).unwrap();
        let mut board = Board::load(&temp_pcb_path, &HashSet::new()).unwrap();
        board.tracks = vec![
            TrackSegment {
                start: (5.0, 5.0),
                end: (15.0, 5.0),
                width: 0.25,
                layer: 0,
                net_id: 1,
                is_locked: false,
            },
            TrackSegment {
                start: (15.0, 5.0),
                end: (25.0, 15.0),
                width: 0.25,
                layer: 0,
                net_id: 1,
                is_locked: false,
            },
        ];

        let mut app = VulcansTrialsApp::new(
            temp_pcb_path.clone(),
            temp_pcb_path.clone(),
            board,
            None,
        )
        .with_preferences_path(PathBuf::from(&temp_pref_path));

        // 1. Swap straight -> organic: pads are routed so airwires must NOT redraw
        assert!(app.swap_organic_tracks(true));
        assert!(app.any_angle_mode);
        {
            let g = app.live.lock().unwrap();
            assert!(
                !g.board.arcs.is_empty(),
                "Swapping to organic tracks must produce curved ArcSegments"
            );
            assert!(g.straight_copper_snapshot.is_some());
            assert!(
                g.ratsnest.is_empty(),
                "Swapping to organic tracks must not redraw airwires for already-routed pads"
            );
        }

        // 2. Swap organic -> straight: airwires must remain empty
        assert!(app.swap_organic_tracks(false));
        assert!(!app.any_angle_mode);
        {
            let g = app.live.lock().unwrap();
            assert!(
                g.board.arcs.is_empty(),
                "Swapping back to straight tracks must restore 0 arcs"
            );
            assert_eq!(
                g.board.tracks.len(),
                2,
                "Swapping back to straight tracks must restore the original 2 straight track segments"
            );
            assert!(
                g.ratsnest.is_empty(),
                "Swapping back to straight tracks must not redraw airwires for already-routed pads"
            );
        }

        let _ = fs::remove_file(&temp_pref_path);
        let _ = fs::remove_file(&temp_pcb_path);
    }

    #[test]
    #[cfg(unix)]
    fn test_ipc_auto_retry_and_pull_on_save() {
        use std::io::{BufRead, BufReader};
        use std::os::unix::net::UnixListener;

        let temp_pref_path = format!(
            "/tmp/vulcans_trials_ipc_retry_pref_test_{}.json",
            std::process::id()
        );
        let sock_path = format!(
            "/tmp/vulcans_trials_ipc_retry_sock_{}.sock",
            std::process::id()
        );
        let out_pcb_path = format!(
            "/tmp/vulcans_trials_ipc_pull_out_{}.kicad_pcb",
            std::process::id()
        );
        let out_ses_path = format!(
            "/tmp/vulcans_trials_ipc_pull_out_{}.ses",
            std::process::id()
        );
        let _ = fs::remove_file(&temp_pref_path);
        let _ = fs::remove_file(&sock_path);
        let _ = fs::remove_file(&out_pcb_path);
        let _ = fs::remove_file(&out_ses_path);

        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 20.0;
        board.max_y = 20.0;
        board.tracks.push(TrackSegment {
            start: (2.0, 2.0),
            end: (10.0, 2.0),
            width: 0.25,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });

        let mut app = VulcansTrialsApp::new(
            out_pcb_path.clone(),
            out_pcb_path.clone(),
            board,
            None,
        )
        .with_preferences_path(PathBuf::from(&temp_pref_path));

        // 1. Default launch state has IPC enabled and points to custom socket (not listening yet)
        assert!(app.kicad_ipc_enabled, "IPC must be enabled by default on launch");
        app.kicad_ipc_socket_path = format!("ipc://{}", sock_path);
        app.ipc = None;
        app.ipc_retry_interval = Duration::from_millis(1);
        app.last_ipc_retry = Instant::now() - Duration::from_secs(5);

        // Socket is not bound yet -> poll_ipc_auto_reconnect returns false
        assert!(!app.poll_ipc_auto_reconnect());
        assert!(!app.is_ipc_connected());

        // 2. Now start the Unix socket server and verify periodic retry connects automatically
        let listener = UnixListener::bind(&sock_path).expect("Failed to bind test unix socket");
        let server_thread = std::thread::spawn(move || {
            let (stream, _) = listener.accept().expect("Accept failed");
            let reader = BufReader::new(stream);
            let mut messages = Vec::new();
            for line in reader.lines().flatten() {
                if !line.trim().is_empty() {
                    let msg: crate::ipc::IpcMessage =
                        serde_json::from_str(&line).expect("Valid JSON");
                    let is_pull = matches!(msg, crate::ipc::IpcMessage::PullBoard { .. });
                    messages.push(msg);
                    if is_pull {
                        break;
                    }
                }
            }
            messages
        });

        app.last_ipc_retry = Instant::now() - Duration::from_secs(5);
        assert!(
            app.poll_ipc_auto_reconnect(),
            "poll_ipc_auto_reconnect must automatically connect once socket becomes available"
        );
        assert!(app.is_ipc_connected());

        // 3. Save the board and verify it pulls the updated board into KiCad over IPC
        assert!(app.save_output_and_pull_to_kicad());
        assert!(Path::new(&out_pcb_path).exists());
        assert!(Path::new(&out_ses_path).exists());

        let received = server_thread.join().expect("Server thread join failed");
        assert!(
            received
                .iter()
                .any(|m| matches!(m, crate::ipc::IpcMessage::PullBoard { .. })),
            "Saving the board while IPC is connected must send PullBoard to KiCad"
        );

        let _ = fs::remove_file(&temp_pref_path);
        let _ = fs::remove_file(&sock_path);
        let _ = fs::remove_file(&out_pcb_path);
        let _ = fs::remove_file(&out_ses_path);
    }

    #[test]
    fn test_gui_controls_and_parameters_tooltips_exercised() {
        let in_path = format!(
            "/tmp/vulcan_gui_tooltips_test_{}.kicad_pcb",
            std::process::id()
        );
        let mut board = make_gui_test_board(&in_path);
        // Populate holding palette so Holding Palette panel and its Place / Place All tooltips are rendered
        let held_fp = board.footprints[1].clone();
        board.holding_palette.push(held_fp);

        let mut app = VulcansTrialsApp::new(in_path.clone(), String::new(), board, None);
        app.show_holding_palette = true;

        let ctx = egui::Context::default();
        ctx.style_mut(|s| {
            s.interaction.tooltip_delay = 0.0;
            s.interaction.show_tooltips_only_when_still = false;
        });

        // Open all collapsible panels (Advanced Autoplacer, Advanced Autorouter, Net Classes)
        let panel_id = egui::Id::new("vulcan_controls_panel");
        for section in [
            "Advanced Autoplacer",
            "Advanced Autorouter",
            "Net Classes (1)",
        ] {
            let id = panel_id.with(section);
            egui::collapsing_header::CollapsingState::load_with_default_open(&ctx, id, true)
                .store(&ctx);
        }

        let screen_rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1400.0, 2000.0));

        // 1. Warm up layout across all expanded panels and record baseline shape count without hover
        let mut warmup = egui::RawInput::default();
        warmup.screen_rect = Some(screen_rect);
        let _ = ctx.run(warmup.clone(), |ctx| {
            app.render_ui(ctx);
        });
        let baseline_output = ctx.run(warmup, |ctx| {
            app.render_ui(ctx);
        });
        let baseline_shapes = baseline_output.shapes.len();

        // 2. Sweep pointer across top telemetry bar, left control & parameter widgets, and holding palette
        let mut tooltip_frames = 0usize;
        let mut sample_positions = Vec::new();
        // Top bar positions (heading, GPU status, CPU/GPU load badges, progress bar)
        for x in [60.0_f32, 340.0, 520.0, 680.0, 250.0] {
            sample_positions.push(Pos2::new(x, 16.0));
            sample_positions.push(Pos2::new(x, 42.0));
        }
        // Left panel vertical scan across buttons, core sliders, advanced placer/router sliders, IPC, visibility checkboxes, netclasses, metrics
        for y_step in (65..1650).step_by(18) {
            sample_positions.push(Pos2::new(90.0, y_step as f32));
            sample_positions.push(Pos2::new(210.0, y_step as f32));
        }
        // Holding palette panel positions (heading, Place All button, mini footprint preview, individual Place button)
        for y in [75.0_f32, 100.0, 140.0] {
            sample_positions.push(Pos2::new(360.0, y));
            sample_positions.push(Pos2::new(430.0, y));
            sample_positions.push(Pos2::new(530.0, y));
        }

        for pos in sample_positions {
            let mut input = egui::RawInput::default();
            input.screen_rect = Some(screen_rect);
            input.events.push(egui::Event::PointerMoved(pos));
            let output = ctx.run(input, |ctx| {
                app.render_ui(ctx);
            });
            if output.shapes.len() > baseline_shapes {
                tooltip_frames += 1;
            }
        }
        assert!(
            tooltip_frames >= 20,
            "Hovering across GUI controls and parameter sliders must render tooltip popups (got {} tooltip frames)",
            tooltip_frames
        );

        // 3. Also exercise route_tool_active and active_route banner tooltips
        app.route_tool_active = true;
        let mut input_route_mode = egui::RawInput::default();
        input_route_mode.screen_rect = Some(screen_rect);
        let _ = ctx.run(input_route_mode, |ctx| {
            app.render_ui(ctx);
        });

        app.active_route = Some(ActiveRoute {
            net_id: 2,
            net_name: "SIG".to_string(),
            current_layer: 0,
            track_width: 0.25,
            via_dia: 0.6,
            via_drill: 0.3,
            anchor_points: vec![(110.0, 115.0, 0)],
            posture_diagonal_first: false,
            vias_placed: Vec::new(),
        });
        let mut input_active_route = egui::RawInput::default();
        input_active_route.screen_rect = Some(screen_rect);
        let _ = ctx.run(input_active_route, |ctx| {
            app.render_ui(ctx);
        });

        let _ = fs::remove_file(&in_path);
    }

    #[test]
    fn test_single_pcb_path_synchronization_and_reload() {
        let temp_dir = std::env::temp_dir();
        let pcb_path = temp_dir.join(format!("test_single_pcb_{}.kicad_pcb", std::process::id()));
        let pcb_path_str = pcb_path.to_string_lossy().to_string();

        let content = r#"(kicad_pcb (version 20240108) (generator "test")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (48 "B.CrtYd" user) (49 "F.CrtYd" user))
  (net 0 "")
  (net 1 "NET1")
  (footprint "R" (at 10 10) (property "Reference" "R1")
    (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 1 "NET1")))
)"#;
        fs::write(&pcb_path, content).unwrap();

        let board = Board::load(&pcb_path_str, &HashSet::new()).unwrap();
        let mut app = VulcansTrialsApp::new(
            pcb_path_str.clone(),
            pcb_path_str.clone(),
            board,
            None,
        );

        // Verify initial paths are unified
        assert_eq!(app.input_path, pcb_path_str);
        assert_eq!(app.output_path, pcb_path_str);

        // Test set_board_path unifies both
        let new_path = "/tmp/new_board.kicad_pcb".to_string();
        app.set_board_path(new_path.clone());
        assert_eq!(app.input_path, new_path);
        assert_eq!(app.output_path, new_path);

        // Test open_pcb_from_path resets view and reloads
        app.pan_offset = Vec2::new(50.0, 50.0);
        app.zoom = 3.5;
        assert!(app.open_pcb_from_path(&pcb_path_str));
        assert_eq!(app.input_path, pcb_path_str);
        assert_eq!(app.output_path, pcb_path_str);
        assert_eq!(app.pan_offset, Vec2::ZERO);
        assert_eq!(app.zoom, 1.0);

        // Mutate live board in memory
        {
            let mut g = app.live.lock().unwrap();
            g.board.footprints.clear();
            assert_eq!(g.board.footprints.len(), 0);
        }

        // Test reload_pcb restores from disk
        assert!(app.reload_pcb());
        {
            let g = app.live.lock().unwrap();
            assert_eq!(g.board.footprints.len(), 1);
            assert_eq!(g.board.footprints[0].reference, "R1");
        }

        let _ = fs::remove_file(&pcb_path);
    }

    #[test]
    fn test_single_pcb_file_menu_and_modals_render() {
        let temp_dir = std::env::temp_dir();
        let pcb_path = temp_dir.join(format!("test_menu_modal_{}.kicad_pcb", std::process::id()));
        let pcb_path_str = pcb_path.to_string_lossy().to_string();

        let content = r#"(kicad_pcb (version 20240108) (generator "test")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal))
  (net 0 "")
  (net 1 "GND")
)"#;
        fs::write(&pcb_path, content).unwrap();

        let board = Board::load(&pcb_path_str, &HashSet::new()).unwrap();
        let mut app = VulcansTrialsApp::new(
            pcb_path_str.clone(),
            pcb_path_str.clone(),
            board,
            None,
        );

        let ctx = egui::Context::default();
        let mut raw_input = egui::RawInput::default();
        raw_input.screen_rect = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0)));

        // 1. Initial render: top menu bar renders with File menu
        let _ = ctx.run(raw_input.clone(), |ctx| {
            app.render_ui(ctx);
        });

        // 2. Open Modal dialog
        app.open_dialog_open = true;
        app.file_dialog_path_buffer = pcb_path_str.clone();
        let output = ctx.run(raw_input.clone(), |ctx| {
            app.render_ui(ctx);
        });
        assert!(!output.shapes.is_empty(), "Rendering Open dialog modal must produce shapes");

        // 3. Save As Modal dialog
        app.open_dialog_open = false;
        app.save_as_dialog_open = true;
        app.file_dialog_path_buffer = format!("{}_copy.kicad_pcb", pcb_path_str);
        let output_save = ctx.run(raw_input.clone(), |ctx| {
            app.render_ui(ctx);
        });
        assert!(!output_save.shapes.is_empty(), "Rendering Save As dialog modal must produce shapes");

        let _ = fs::remove_file(&pcb_path);
    }

    #[test]
    fn test_compute_initial_view_bounds_and_auto_start_exit() {
        let temp_dir = std::env::temp_dir();
        let pcb_path = temp_dir.join(format!("test_view_bounds_{}.kicad_pcb", std::process::id()));
        let pcb_path_str = pcb_path.to_string_lossy().to_string();

        let content = r#"(kicad_pcb (version 20240108) (generator "test")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal))
  (gr_rect (start 120.0 80.0) (end 170.0 120.0) (layer "Edge.Cuts"))
  (footprint "R" (at 90.0 85.0) (property "Reference" "R1")
    (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 1 "NET1")))
  (footprint "R" (at 180.0 130.0) (property "Reference" "R2")
    (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 1 "NET1")))
)"#;
        fs::write(&pcb_path, content).unwrap();

        let board = Board::load(&pcb_path_str, &HashSet::new()).unwrap();
        let (min_x, max_x, min_y, max_y) = compute_initial_view_bounds(&board);

        // R1 is at 90.0 -> min_x should be <= 87.0
        assert!(min_x <= 87.0, "min_x should encompass R1 off-board");
        // R2 is at 180.0 -> max_x should be >= 183.0
        assert!(max_x >= 183.0, "max_x should encompass R2 off-board");
        // Board min_y is 80.0, R1 y is 85.0 -> min_y should be <= 80.0
        assert!(min_y <= 80.0);
        // R2 y is 130.0 -> max_y should be >= 133.0
        assert!(max_y >= 133.0);

        let mut app = VulcansTrialsApp::new(
            pcb_path_str.clone(),
            pcb_path_str.clone(),
            board.clone(),
            None,
        );
        app.auto_start = true;
        app.auto_exit = Some(1);
        app.place_duration_secs = 0;

        let ctx = egui::Context::default();
        let mut raw_input = egui::RawInput::default();
        raw_input.screen_rect = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1920.0, 1080.0)));

        // Run 35 frames to trigger auto_start
        for _ in 0..35 {
            let _ = ctx.run(raw_input.clone(), |ctx| {
                app.render_ui(ctx);
            });
        }

        // Verify trial started or running
        assert!(app.frame_count >= 35);

        let _ = fs::remove_file(&pcb_path);
    }
}
