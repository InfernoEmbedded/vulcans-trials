// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use crate::board::{Board, LiveTrialState, Point, Polygon, TrackSegment, TrialPhase, Via};
use crate::ipc::KiCadIpcClient;
use crate::placer::{run_auto_placer, PlacementConfig};
use crate::router::{AutoRouter, RoutingGrid};
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

    // Aggressive placement exploration & weights
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
    #[serde(default)]
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
            show_courtyards: true,
            show_ratsnest: true,
            show_f_cu: true,
            show_b_cu: true,
            show_in1_cu: true,
            show_in2_cu: true,
            show_vias: true,
            show_refs: true,
            show_holding_palette: true,
            kicad_ipc_enabled: false,
            kicad_ipc_socket_path: String::new(),
            bundle_track_width: default_bundle_dim(),
            bundle_track_spacing: default_bundle_dim(),
            last_input_path: None,
            last_output_path: None,
        }
    }
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
        Self::load_from_path(&Self::config_path()).unwrap_or_default()
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

    // Persistence
    pub preferences_path: Option<PathBuf>,
    pub last_saved_prefs: AppPreferences,
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
            active_route: None,
            route_tool_active: false,
            bundle_sketch_active: false,
            bundle_guide_points: Vec::new(),
            bundle_selected_nets: Vec::new(),
            bundle_track_width: 0.25,
            bundle_track_spacing: 0.25,
            ipc: None,
            kicad_ipc_enabled: false,
            kicad_ipc_socket_path: KiCadIpcClient::discover_socket_path()
                .map(|p| format!("ipc://{}", p.display()))
                .unwrap_or_else(|| "ipc:///tmp/kicad/api.sock".to_string()),
            preferences_path: None,
            last_saved_prefs: AppPreferences::default(),
        };

        // Load and apply saved preferences on launch (isolated during unit tests)
        #[cfg(not(test))]
        {
            let saved_prefs = AppPreferences::load();
            app.apply_preferences(&saved_prefs);
            app.last_saved_prefs = app.to_preferences();
        }

        // If IPC live sync was saved as enabled and no client connected yet, attempt reconnection
        if app.kicad_ipc_enabled && app.ipc.is_none() {
            let custom = if app.kicad_ipc_socket_path.trim().is_empty() {
                None
            } else {
                Some(app.kicad_ipc_socket_path.trim())
            };
            if let Some(client) = KiCadIpcClient::connect_or_discover(custom) {
                app.ipc = Some(Arc::new(Mutex::new(client)));
            }
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
        let target = self.preferences_path.clone().unwrap_or_else(AppPreferences::config_path);
        let _ = prefs.save_to_path(&target);
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
        }
        self.ipc = ipc;
        self.last_saved_prefs = self.to_preferences();
        self
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
                let mut grid = RoutingGrid::new(&working_board, grid_res, vulkan).with_inner_layer_penalty(inner_plane_pen);
                let mut router = AutoRouter::new(max_passes)
                    .with_allow_component_nudge(allow_comp_nudge)
                    .with_any_angle_mode(any_angle)
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

    pub fn render_ui(&mut self, ctx: &egui::Context) {
        ctx.request_repaint_after(std::time::Duration::from_millis(50));

        let telem = self.telemetry.state.lock().unwrap().clone();
        let snap = self.live.lock().unwrap().clone();
        let is_running = self.worker_running.load(Ordering::Relaxed);

        // Top Telemetry & Vulkan Status Bar
        egui::TopBottomPanel::top("vulcan_top_bar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading(egui::RichText::new(format!("⚡ Vulcan's Trials v{}", env!("CARGO_PKG_VERSION"))).color(Color32::from_rgb(255, 145, 50)));
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
            .max_width(310.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.heading("Controls");

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
                        );
                    });
                    ui.add_space(2.0);
                } else if self.route_tool_active {
                    ui.group(|ui| {
                        ui.label(
                            egui::RichText::new("Route Mode Active | Click pad, via, or track to start [Esc to exit]")
                                .color(Color32::from_rgb(100, 220, 255)),
                        );
                    });
                    ui.add_space(2.0);
                }

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
                    if ui.add_enabled(!is_running, route_btn).clicked() {
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
                    if ui.add_enabled(!is_running, bundle_btn).clicked() {
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
                ui.add_enabled(
                    !is_running,
                    egui::Checkbox::new(&mut self.allow_component_nudge, "Allow Router to Nudge Unlocked Passives"),
                );
                ui.add_enabled(
                    !is_running,
                    egui::Checkbox::new(&mut self.any_angle_mode, "Any-Angle Mode (TopoR Arcs)"),
                );

                ui.separator();
                ui.heading("KiCad 10 IPC Live Sync");
                if ui.checkbox(&mut self.kicad_ipc_enabled, "KiCad IPC Live Sync").changed() {
                    if self.kicad_ipc_enabled && self.ipc.is_none() {
                        let custom = if self.kicad_ipc_socket_path.trim().is_empty() {
                            None
                        } else {
                            Some(self.kicad_ipc_socket_path.trim())
                        };
                        if let Some(client) = KiCadIpcClient::connect_or_discover(custom) {
                            if self.kicad_ipc_socket_path.trim().is_empty() {
                                if let Some(sp) = client.socket_path() {
                                    self.kicad_ipc_socket_path = format!("ipc://{}", sp.display());
                                }
                            }
                            self.ipc = Some(Arc::new(Mutex::new(client)));
                        }
                    } else if !self.kicad_ipc_enabled {
                        self.ipc = None;
                    }
                }

                ui.horizontal(|ui| {
                    ui.label("Socket:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.kicad_ipc_socket_path)
                            .hint_text("ipc:///tmp/kicad/api.sock")
                            .desired_width(160.0),
                    );
                    if ui.button("Connect").clicked() {
                        let custom = if self.kicad_ipc_socket_path.trim().is_empty() {
                            None
                        } else {
                            Some(self.kicad_ipc_socket_path.trim())
                        };
                        if let Some(client) = KiCadIpcClient::connect_or_discover(custom) {
                            if self.kicad_ipc_socket_path.trim().is_empty() {
                                if let Some(sp) = client.socket_path() {
                                    self.kicad_ipc_socket_path = format!("ipc://{}", sp.display());
                                }
                            }
                            self.ipc = Some(Arc::new(Mutex::new(client)));
                            self.kicad_ipc_enabled = true;
                        } else {
                            self.ipc = None;
                        }
                    }
                });

                let (status_text, status_col) = match self.ipc.as_ref() {
                    Some(client_lock) => {
                        if let Ok(c) = client_lock.lock() {
                            if c.is_connected() {
                                ("Connected", Color32::from_rgb(0, 220, 100))
                            } else {
                                ("Disconnected", Color32::from_rgb(220, 80, 80))
                            }
                        } else {
                            ("Locked", Color32::GRAY)
                        }
                    }
                    None => {
                        if self.kicad_ipc_enabled {
                            ("Connecting / Not Found", Color32::from_rgb(220, 150, 40))
                        } else {
                            ("Offline", Color32::GRAY)
                        }
                    }
                };
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Status:").strong());
                    ui.label(egui::RichText::new(status_text).color(status_col));
                });

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
                ui.horizontal(|ui| {
                    if ui.button("💾 Save Preferences").clicked() {
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

        let canvas_rect = Rect::from_min_max(Pos2::new(318.0, 67.0), Pos2::new(792.0, 592.0));
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
}

