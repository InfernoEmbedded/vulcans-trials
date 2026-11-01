// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use crate::board::{ArcSegment, Board, Footprint, LiveTrialState, NetClass, NetTopology, Pad, Point, TrackSegment, TrialPhase, Via};
pub use crate::board::BusBundle;
use crate::ipc::KiCadIpcClient;
use crate::vulkan::VulkanEngine;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub const KEEPOUT_NET: i32 = -1;
pub const FREE_NET: i32 = 0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingConfig {
    pub crosstalk_penalty: f64,
    #[serde(default)]
    pub any_angle_mode: bool,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            crosstalk_penalty: 4.0,
            any_angle_mode: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridNode {
    pub gx: usize,
    pub gy: usize,
    pub layer: usize,
}

pub struct RoutingGrid {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub res: f64,
    pub width: usize,
    pub height: usize,
    pub num_layers: usize,
    pub plane_nets: HashMap<usize, usize>,
    pub obstacle_net_ids: Vec<i32>,
    pub dist_field: Vec<f32>,
    pub congestion: Vec<f32>,
    pub nudgeable_fp: Vec<Option<usize>>,
    pub smd_pad_cells: Vec<bool>,
    pub net_clearances: Vec<f64>,
    pub max_net_clearance: f64,
    pub inner_layer_penalty: f64,
    pub vulkan: Option<Arc<VulkanEngine>>,
}

/// Checks whether a segment conforms to the standard octilinear 45-degree grid (0°, 45°, 90°, etc.).
#[inline]
pub fn is_octilinear_segment(p1: (f64, f64), p2: (f64, f64)) -> bool {
    let dx = (p2.0 - p1.0).abs();
    let dy = (p2.1 - p1.1).abs();
    dx < 1e-4 || dy < 1e-4 || (dx - dy).abs() < 1e-4
}

/// KiCad-standard 45° dogleg corner computation.
/// When `diagonal_first` is false: straight horizontal/vertical segment first, then 45° diagonal to p1.
/// When `diagonal_first` is true: 45° diagonal segment first, then straight horizontal/vertical to p1.
pub fn compute_45_dogleg(p0: (f64, f64), p1: (f64, f64), diagonal_first: bool) -> Vec<(f64, f64)> {
    let dx = p1.0 - p0.0;
    let dy = p1.1 - p0.1;
    let abs_dx = dx.abs();
    let abs_dy = dy.abs();

    if abs_dx < 1e-6 && abs_dy < 1e-6 {
        return vec![p0];
    }

    let sign_x = if dx >= 0.0 { 1.0 } else { -1.0 };
    let sign_y = if dy >= 0.0 { 1.0 } else { -1.0 };

    let corner = if abs_dx > abs_dy {
        // Dominant direction: horizontal
        if diagonal_first {
            (p0.0 + sign_x * abs_dy, p1.1)
        } else {
            (p1.0 - sign_x * abs_dy, p0.1)
        }
    } else {
        // Dominant direction: vertical (or equal)
        if diagonal_first {
            (p1.0, p0.1 + sign_y * abs_dx)
        } else {
            (p0.0, p1.1 - sign_y * abs_dx)
        }
    };

    let dist_to_p0 = (corner.0 - p0.0).hypot(corner.1 - p0.1);
    let dist_to_p1 = (corner.0 - p1.0).hypot(corner.1 - p1.1);

    if dist_to_p0 < 1e-6 || dist_to_p1 < 1e-6 {
        vec![p0, p1]
    } else {
        vec![p0, corner, p1]
    }
}

impl RoutingGrid {
    pub fn new(board: &Board, res: f64, vulkan: Option<Arc<VulkanEngine>>) -> Self {
        let min_x = board.min_x;
        let min_y = board.min_y;
        let max_x = board.max_x;
        let max_y = board.max_y;
        let width = ((max_x - min_x) / res).ceil() as usize;
        let height = ((max_y - min_y) / res).ceil() as usize;
        let num_layers = board.copper_layers().len().max(2);
        let plane_nets = board.inner_plane_nets();
        let total_cells = num_layers * width * height;

        let mut net_clearances = vec![0.20; board.nets.len()];
        let mut max_net_clearance = 0.25_f64;
        for i in 0..board.nets.len() {
            let mut c = board.get_netclass_for_net(i).clearance;
            let touches_fine_pitch = board.footprints.iter().any(|fp| {
                let smd_pads: Vec<&Pad> = fp.pads.iter().filter(|p| !p.is_through_hole).collect();
                if smd_pads.len() >= 4 {
                    let mut min_pitch = f64::INFINITY;
                    for p1_i in 0..smd_pads.len() {
                        let (x1, y1) = fp.pad_absolute_position(smd_pads[p1_i]);
                        for p2_i in (p1_i + 1)..smd_pads.len() {
                            let (x2, y2) = fp.pad_absolute_position(smd_pads[p2_i]);
                            let d = (x1 - x2).hypot(y1 - y2);
                            if d > 0.05 && d < min_pitch {
                                min_pitch = d;
                            }
                        }
                    }
                    if min_pitch <= 0.85 {
                        return fp.pads.iter().any(|p| p.net_id == i);
                    }
                }
                false
            });
            if touches_fine_pitch {
                c = c.min(0.15);
            }
            net_clearances[i] = c;
            max_net_clearance = max_net_clearance.max(c);
        }

        let mut grid = Self {
            min_x,
            min_y,
            max_x,
            max_y,
            res,
            width,
            height,
            num_layers,
            plane_nets,
            obstacle_net_ids: vec![FREE_NET; total_cells],
            dist_field: vec![999.0; total_cells],
            congestion: vec![0.0; total_cells],
            nudgeable_fp: vec![None; total_cells],
            smd_pad_cells: vec![false; total_cells],
            net_clearances,
            max_net_clearance,
            inner_layer_penalty: 8.0,
            vulkan,
        };

        grid.rasterize_obstacles(board);
        grid.update_distance_fields();
        grid
    }

    pub fn with_inner_layer_penalty(mut self, pen: f64) -> Self {
        self.inner_layer_penalty = pen;
        self
    }

    #[inline]
    pub fn idx(&self, gx: usize, gy: usize, layer: usize) -> usize {
        layer * (self.width * self.height) + gy * self.width + gx
    }

    pub fn world_to_grid(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        if x < self.min_x || x >= self.max_x || y < self.min_y || y >= self.max_y {
            return None;
        }
        let gx = ((x - self.min_x) / self.res) as usize;
        let gy = ((y - self.min_y) / self.res) as usize;
        if gx < self.width && gy < self.height {
            Some((gx, gy))
        } else {
            None
        }
    }

    pub fn grid_to_world(&self, gx: usize, gy: usize) -> (f64, f64) {
        (
            self.min_x + (gx as f64 + 0.5) * self.res,
            self.min_y + (gy as f64 + 0.5) * self.res,
        )
    }

    pub fn rasterize_obstacles(&mut self, board: &Board) {
        let max_world_x = self.min_x + (self.width as f64) * self.res;
        let max_world_y = self.min_y + (self.height as f64) * self.res;
        if let Some(ref poly) = board.outline_polygon {
            let boundary = poly.aabb();
            let edge_clearance = board.board_edge_clearance();
            for layer in 0..self.num_layers {
                for gy in 0..self.height {
                    for gx in 0..self.width {
                        let (wx, wy) = self.grid_to_world(gx, gy);
                        let is_edge_keepout = if poly.vertices.len() == 4 {
                            wx < boundary.min_x + edge_clearance
                                || wx > boundary.max_x - edge_clearance
                                || wy < boundary.min_y + edge_clearance
                                || wy > boundary.max_y - edge_clearance
                        } else {
                            let p = Point::new(wx, wy);
                            !poly.contains_point(&p) || poly.distance_to_point(&p) < edge_clearance
                        };
                        if is_edge_keepout {
                            let i = self.idx(gx, gy, layer);
                            self.obstacle_net_ids[i] = KEEPOUT_NET;
                        }
                    }
                }
            }
        }

        for ko in &board.keepouts {
            if ko.max_x < self.min_x || ko.min_x > max_world_x || ko.max_y < self.min_y || ko.min_y > max_world_y {
                continue;
            }
            let min_gx = ((ko.min_x - self.min_x) / self.res)
                .ceil()
                .clamp(0.0, self.width.saturating_sub(1) as f64) as usize;
            let min_gy = ((ko.min_y - self.min_y) / self.res)
                .ceil()
                .clamp(0.0, self.height.saturating_sub(1) as f64) as usize;
            let max_gx = ((ko.max_x - self.min_x) / self.res)
                .floor()
                .clamp(0.0, self.width.saturating_sub(1) as f64) as usize;
            let max_gy = ((ko.max_y - self.min_y) / self.res)
                .floor()
                .clamp(0.0, self.height.saturating_sub(1) as f64) as usize;
            for &layer in &ko.layers {
                if layer >= self.num_layers {
                    continue;
                }
                for gy in min_gy..=max_gy {
                    for gx in min_gx..=max_gx {
                        let i = self.idx(gx, gy, layer);
                        self.obstacle_net_ids[i] = KEEPOUT_NET;
                    }
                }
            }
        }

        let all_layers: Vec<usize> = (0..self.num_layers).collect();
        for (fp_idx, fp) in board.footprints.iter().enumerate() {
            let is_nudgeable = !fp.is_locked;
            let smd_layer = [fp.layer.min(self.num_layers - 1)];
            for pad in &fp.pads {
                let (px, py) = fp.pad_absolute_position(pad);
                let (rot_w, rot_h) = fp.pad_world_size(pad);
                let half_w = rot_w / 2.0;
                let half_h = rot_h / 2.0;
                let layers: &[usize] = if pad.is_through_hole {
                    &all_layers
                } else {
                    &smd_layer
                };

                let total_rot = ((fp.rotation + pad.rotation) % 360.0 + 360.0) % 360.0;
                let pad_hw = pad.width * 0.5;
                let pad_hh = pad.height * 0.5;

                for &layer in layers {
                    let min_gx = ((px - half_w - self.min_x) / self.res)
                        .floor()
                        .clamp(0.0, self.width.saturating_sub(1) as f64) as usize;
                    let max_gx = ((px + half_w - self.min_x) / self.res)
                        .floor()
                        .clamp(0.0, self.width.saturating_sub(1) as f64) as usize;
                    let min_gy = ((py - half_h - self.min_y) / self.res)
                        .floor()
                        .clamp(0.0, self.height.saturating_sub(1) as f64) as usize;
                    let max_gy = ((py + half_h - self.min_y) / self.res)
                        .floor()
                        .clamp(0.0, self.height.saturating_sub(1) as f64) as usize;

                    let net_val = if pad.net_id == 0
                        || pad.net_name.is_empty()
                        || pad.net_name.starts_with("unconnected")
                    {
                        KEEPOUT_NET
                    } else {
                        pad.net_id as i32
                    };

                    for gy in min_gy..=max_gy {
                        for gx in min_gx..=max_gx {
                            let (wx, wy) = self.grid_to_world(gx, gy);
                            let unrot = Point::new(wx - px, wy - py).rotate_ccw(-total_rot);
                            let dx = 0.0f64.max(unrot.x.abs() - pad_hw);
                            let dy = 0.0f64.max(unrot.y.abs() - pad_hh);
                            if dx.hypot(dy) > self.res * 0.5 + 1e-4 {
                                continue;
                            }

                            let i = self.idx(gx, gy, layer);
                            self.obstacle_net_ids[i] = net_val;
                            if is_nudgeable {
                                self.nudgeable_fp[i] = Some(fp_idx);
                            }
                            if !pad.is_through_hole && (pad.width < 1.5 || pad.height < 1.5) {
                                self.smd_pad_cells[i] = true;
                            }
                        }
                    }

                    // Always ensure the cell containing the pad origin is marked
                    if let Some((cgx, cgy)) = self.world_to_grid(px, py) {
                        let i = self.idx(cgx, cgy, layer);
                        self.obstacle_net_ids[i] = net_val;
                        if is_nudgeable {
                            self.nudgeable_fp[i] = Some(fp_idx);
                        }
                        if !pad.is_through_hole && (pad.width < 1.5 || pad.height < 1.5) {
                            self.smd_pad_cells[i] = true;
                        }
                    }
                }
            }
        }

        for trk in &board.tracks {
            if trk.is_locked {
                self.commit_track(trk);
            }
        }
        for v in &board.vias {
            if v.is_locked {
                self.commit_via(v);
            }
        }
    }

    pub fn commit_track(&mut self, trk: &TrackSegment) {
        if trk.layer >= self.num_layers {
            return;
        }
        let dist = (trk.end.0 - trk.start.0).hypot(trk.end.1 - trk.start.1);
        let steps = (dist / (self.res * 0.5)).ceil() as usize;
        let r_cells = (trk.width / (2.0 * self.res)).ceil() as isize;
        for s in 0..=steps {
            let t = s as f64 / steps.max(1) as f64;
            let wx = trk.start.0 + t * (trk.end.0 - trk.start.0);
            let wy = trk.start.1 + t * (trk.end.1 - trk.start.1);
            if let Some((gx, gy)) = self.world_to_grid(wx, wy) {
                for dy in -r_cells..=r_cells {
                    for dx in -r_cells..=r_cells {
                        if (dx * dx + dy * dy) as f64 * (self.res * self.res) <= (trk.width / 2.0).powi(2) + 1e-6 {
                            let nx = gx as isize + dx;
                            let ny = gy as isize + dy;
                            if nx >= 0 && nx < self.width as isize && ny >= 0 && ny < self.height as isize {
                                let idx = self.idx(nx as usize, ny as usize, trk.layer);
                                if self.obstacle_net_ids[idx] == FREE_NET {
                                    self.obstacle_net_ids[idx] = trk.net_id as i32;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn commit_via(&mut self, v: &Via) {
        let vr_radius = (v.size / 2.0) + (self.res * 0.40);
        let vr_cells = (vr_radius / self.res).ceil() as isize;
        if let Some((gx, gy)) = self.world_to_grid(v.position.0, v.position.1) {
            for cy in -vr_cells..=vr_cells {
                for cx in -vr_cells..=vr_cells {
                    if (cx * cx + cy * cy) as f64 * (self.res * self.res) <= vr_radius.powi(2) + 1e-6 {
                        let nx = gx as isize + cx;
                        let ny = gy as isize + cy;
                        if nx >= 0 && nx < self.width as isize && ny >= 0 && ny < self.height as isize {
                            for l in 0..self.num_layers {
                                let idx = self.idx(nx as usize, ny as usize, l);
                                if self.obstacle_net_ids[idx] == FREE_NET {
                                    self.obstacle_net_ids[idx] = v.net_id as i32;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn update_distance_fields(&mut self) {
        let layer_cells = self.width * self.height;
        if let Some(ref vk) = self.vulkan {
            let engines = vk.all_engines();
            if engines.len() > 1 && self.num_layers > 1 {
                let num_engines = engines.len();
                let w = self.width as u32;
                let h = self.height as u32;
                let res = self.res as f32;
                let num_layers = self.num_layers;

                let mut layer_obs: Vec<Vec<u32>> = Vec::with_capacity(num_layers);
                for layer in 0..num_layers {
                    let offset = layer * layer_cells;
                    let obs: Vec<u32> = self.obstacle_net_ids[offset..offset + layer_cells]
                        .iter()
                        .map(|&n| if n != FREE_NET { 1u32 } else { 0u32 })
                        .collect();
                    layer_obs.push(obs);
                }

                let mut layer_dfs: Vec<Option<Vec<f32>>> = (0..num_layers).map(|_| None).collect();
                std::thread::scope(|scope| {
                    let mut handles = Vec::new();
                    for (eng_idx, engine) in engines.iter().enumerate() {
                        let assigned_layers: Vec<usize> = (0..num_layers)
                            .filter(|&l| l % num_engines == eng_idx)
                            .collect();
                        let eng_ref = Arc::clone(engine);
                        let obs_refs: Vec<(usize, Vec<u32>)> = assigned_layers
                            .into_iter()
                            .map(|l| (l, layer_obs[l].clone()))
                            .collect();

                        handles.push(scope.spawn(move || {
                            let mut results = Vec::new();
                            for (l, obs) in obs_refs {
                                if let Some(df) = eng_ref.compute_distance_field(&obs, w, h, 12) {
                                    results.push((l, df));
                                }
                            }
                            results
                        }));
                    }
                    for handle in handles {
                        if let Ok(results) = handle.join() {
                            for (l, df) in results {
                                layer_dfs[l] = Some(df);
                            }
                        }
                    }
                });

                for (layer, df_opt) in layer_dfs.into_iter().enumerate() {
                    if let Some(df) = df_opt {
                        let offset = layer * layer_cells;
                        for (dst, &val) in self.dist_field[offset..offset + layer_cells].iter_mut().zip(df.iter()) {
                            *dst = val * res;
                        }
                    }
                }
            } else {
                for layer in 0..self.num_layers {
                    let offset = layer * layer_cells;
                    let obs: Vec<u32> = self.obstacle_net_ids[offset..offset + layer_cells]
                        .iter()
                        .map(|&n| if n != FREE_NET { 1u32 } else { 0u32 })
                        .collect();
                    if let Some(df) = vk.compute_distance_field(&obs, self.width as u32, self.height as u32, 12) {
                        for (dst, &val) in self.dist_field[offset..offset + layer_cells].iter_mut().zip(df.iter()) {
                            *dst = val * (self.res as f32);
                        }
                    }
                }
            }
        }
    }

    pub fn get_neighbors(
        &self,
        node: GridNode,
        target_net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
    ) -> Vec<(GridNode, f64)> {
        let mut neighbors = Vec::with_capacity(12);
        let gx = node.gx;
        let gy = node.gy;
        let layer = node.layer;

        let planar_dirs = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.4142),
            (1, -1, 1.4142),
            (-1, 1, 1.4142),
            (1, 1, 1.4142),
        ];

        let (h_mult, v_mult): (f64, f64) = if layer % 2 == 0 { (1.0, 1.35) } else { (1.35, 1.0) };

        let is_dedicated_plane = self.plane_nets.contains_key(&layer);
        let allow_planar = !is_dedicated_plane || self.plane_nets.get(&layer) == Some(&target_net_id);

        if allow_planar {
            for (dx, dy, base_len) in planar_dirs {
            let nx = gx as isize + dx;
            let ny = gy as isize + dy;
            if nx >= 0 && nx < self.width as isize && ny >= 0 && ny < self.height as isize {
                let ngx = nx as usize;
                let ngy = ny as usize;
                let n_idx = self.idx(ngx, ngy, layer);
                let net_id = self.obstacle_net_ids[n_idx];
                if net_id != FREE_NET && net_id != target_net_id as i32 {
                    continue;
                }

                if dx != 0 && dy != 0 {
                    let idx_h = self.idx(ngx, gy, layer);
                    let idx_v = self.idx(gx, ngy, layer);
                    let net_h = self.obstacle_net_ids[idx_h];
                    let net_v = self.obstacle_net_ids[idx_v];
                    if (net_h != FREE_NET && net_h != target_net_id as i32)
                        || (net_v != FREE_NET && net_v != target_net_id as i32)
                    {
                        continue;
                    }
                }

                let is_diag = dx != 0 && dy != 0;
                let diag_margin = if is_diag { (self.res * 0.7071).min(0.02) } else { 0.0 };
                let max_clearance = self.max_net_clearance.max(req_clearance);
                let search_clearance = (track_width / 2.0) + max_clearance + diag_margin;
                let r_cells = (search_clearance / self.res).ceil() as isize;
                let mut clearance_violation = false;
                for cy_off in -r_cells..=r_cells {
                    for cx_off in -r_cells..=r_cells {
                        let cx = ngx as isize + cx_off;
                        let cy = ngy as isize + cy_off;
                        if cx >= 0 && cx < self.width as isize && cy >= 0 && cy < self.height as isize {
                            let c_idx = self.idx(cx as usize, cy as usize, layer);
                            let c_net = self.obstacle_net_ids[c_idx];
                            if c_net != FREE_NET && c_net != target_net_id as i32 {
                                let raw_obst_c = if c_net > 0 && (c_net as usize) < self.net_clearances.len() {
                                    self.net_clearances[c_net as usize]
                                } else {
                                    req_clearance
                                };
                                let eff_c = if req_clearance <= 0.15 || self.smd_pad_cells[c_idx] {
                                    req_clearance.min(raw_obst_c).max(0.12)
                                } else {
                                    req_clearance.max(raw_obst_c)
                                };
                                let eff_clearance = (track_width / 2.0) + eff_c + diag_margin;
                                let c_dist_sq = (cx_off * cx_off + cy_off * cy_off) as f64 * (self.res * self.res);
                                if c_dist_sq < (eff_clearance - 1e-4).powi(2) {
                                    clearance_violation = true;
                                    break;
                                }
                            }
                        }
                    }
                    if clearance_violation {
                        break;
                    }
                }

                if clearance_violation {
                    continue;
                }

                let dir_factor = if dx != 0 && dy == 0 {
                    h_mult
                } else if dx == 0 && dy != 0 {
                    v_mult
                } else {
                    h_mult.max(v_mult)
                };

                let cong = self.congestion[n_idx] as f64;
                let mut step_cost = (base_len * self.res) * dir_factor * (1.0 + cong);
                if layer >= 2 {
                    if self.plane_nets.get(&layer) == Some(&target_net_id) {
                        step_cost *= 0.40;
                    } else {
                        step_cost += self.inner_layer_penalty * (base_len * self.res);
                    }
                }
                neighbors.push((GridNode { gx: ngx, gy: ngy, layer }, step_cost));
            }
        }
    }

        // Through-hole via transitions across all available copper layers (F.Cu, B.Cu, In1.Cu, In2.Cu)
        let mut inside_smd_pad = false;
        for l in 0..self.num_layers {
            if self.smd_pad_cells[self.idx(gx, gy, l)] {
                inside_smd_pad = true;
                break;
            }
        }

        if !inside_smd_pad {
            let max_clearance = self.max_net_clearance.max(req_clearance);
            let obst_half_w = (track_width / 2.0).max(0.10);
            let search_clearance = (via_dia / 2.0) + obst_half_w + max_clearance + 0.025;
            let vr_cells = (search_clearance / self.res).ceil() as isize;
            let mut via_clearance_violation = false;
            'via_check: for cy_off in -vr_cells..=vr_cells {
                for cx_off in -vr_cells..=vr_cells {
                    let cx = gx as isize + cx_off;
                    let cy = gy as isize + cy_off;
                    if cx >= 0 && cx < self.width as isize && cy >= 0 && cy < self.height as isize {
                        let d_sq = (cx_off * cx_off + cy_off * cy_off) as f64 * (self.res * self.res);
                        for l in 0..self.num_layers {
                            let c_idx = self.idx(cx as usize, cy as usize, l);
                            let c_net = self.obstacle_net_ids[c_idx];
                            if c_net != FREE_NET && c_net != target_net_id as i32 {
                                let obst_c = if c_net > 0 && (c_net as usize) < self.net_clearances.len() {
                                    self.net_clearances[c_net as usize]
                                } else {
                                    req_clearance
                                };
                                let eff_c_val = if req_clearance <= 0.15 || self.smd_pad_cells[c_idx] {
                                    req_clearance.min(obst_c).max(0.12)
                                } else {
                                    req_clearance.max(obst_c)
                                };
                                let eff_c = (via_dia / 2.0) + obst_half_w + eff_c_val + 0.025;
                                if d_sq < eff_c.powi(2) + 1e-6 {
                                    via_clearance_violation = true;
                                    break 'via_check;
                                }
                            }
                            if self.smd_pad_cells[c_idx] {
                                let obst_c = if c_net > 0 && (c_net as usize) < self.net_clearances.len() {
                                    self.net_clearances[c_net as usize]
                                } else {
                                    req_clearance
                                };
                                let eff_c = if c_net == target_net_id as i32 {
                                    (via_dia / 2.0) + self.res * 0.5
                                } else {
                                    let eff_c_val = if req_clearance <= 0.15 || self.smd_pad_cells[c_idx] {
                                        req_clearance.min(obst_c).max(0.12)
                                    } else {
                                        req_clearance.max(obst_c)
                                    };
                                    (via_dia / 2.0) + obst_half_w + eff_c_val + 0.025
                                };
                                if d_sq < eff_c.powi(2) - 1e-6 {
                                    via_clearance_violation = true;
                                    break 'via_check;
                                }
                            }
                        }
                    }
                }
            }

            for other_layer in 0..self.num_layers {
                if other_layer == layer {
                    continue;
                }
                if self.plane_nets.contains_key(&other_layer) && self.plane_nets.get(&other_layer) != Some(&target_net_id) {
                    continue;
                }
                let other_idx = self.idx(gx, gy, other_layer);
                let other_net = self.obstacle_net_ids[other_idx];
                if !via_clearance_violation && (other_net == FREE_NET || other_net == target_net_id as i32) {
                    let base_via_cost = if other_net == target_net_id as i32 {
                        0.1
                    } else if self.plane_nets.get(&other_layer) == Some(&target_net_id) {
                        1.2
                    } else {
                        2.0
                    };
                    let ic_penalty = if let Some(center) = u1_center {
                        let (wx, wy) = self.grid_to_world(gx, gy);
                        let d = (wx - center.0).hypot(wy - center.1);
                        if d < 2.8 && other_net != target_net_id as i32 {
                            2.5
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    let via_cost = base_via_cost + ic_penalty;
                    neighbors.push((GridNode { gx, gy, layer: other_layer }, via_cost));
                }
            }
        }

        neighbors
    }

    /// Evaluates proximity to existing routed tracks of foreign nets on the same layer.
    /// Returns true if the candidate step runs parallel to an adjacent foreign track
    /// at minimum clearance distance (d <= W + 2 * clearance).
    pub fn is_step_coupled(
        &self,
        from: GridNode,
        to: GridNode,
        net_id: usize,
        track_width: f64,
        req_clearance: f64,
    ) -> bool {
        if from.layer != to.layer {
            return false;
        }
        let layer = from.layer;
        let dx = to.gx as isize - from.gx as isize;
        let dy = to.gy as isize - from.gy as isize;
        if dx == 0 && dy == 0 {
            return false;
        }

        let perp_dirs = if dx != 0 && dy == 0 {
            [(0, 1), (0, -1)]
        } else if dx == 0 && dy != 0 {
            [(1, 0), (-1, 0)]
        } else {
            [(-dy, dx), (dy, -dx)]
        };

        let max_dist = track_width + 2.0 * req_clearance;
        let is_diag = dx != 0 && dy != 0;
        let step_scale = if is_diag { 1.41421356 } else { 1.0 };
        let max_k = ((max_dist / (self.res * step_scale)).ceil() as isize).max(1);

        for &(pdx, pdy) in &perp_dirs {
            for k in 1..=max_k {
                let dist = k as f64 * self.res * step_scale;
                if dist > max_dist + 1e-4 {
                    break;
                }
                let off_x = pdx * k;
                let off_y = pdy * k;

                let fx1 = from.gx as isize + off_x;
                let fy1 = from.gy as isize + off_y;
                let fx2 = to.gx as isize + off_x;
                let fy2 = to.gy as isize + off_y;

                if fx1 >= 0 && fx1 < self.width as isize && fy1 >= 0 && fy1 < self.height as isize
                    && fx2 >= 0 && fx2 < self.width as isize && fy2 >= 0 && fy2 < self.height as isize
                {
                    let idx1 = self.idx(fx1 as usize, fy1 as usize, layer);
                    let idx2 = self.idx(fx2 as usize, fy2 as usize, layer);
                    let net1 = self.obstacle_net_ids[idx1];
                    let net2 = self.obstacle_net_ids[idx2];

                    if net1 > 0 && net1 != net_id as i32 && net1 != KEEPOUT_NET {
                        if net1 == net2 {
                            return true;
                        }
                        for cdy in -1..=1 {
                            for cdx in -1..=1 {
                                let cfx = fx2 + cdx;
                                let cfy = fy2 + cdy;
                                if cfx >= 0 && cfx < self.width as isize && cfy >= 0 && cfy < self.height as isize {
                                    let cidx = self.idx(cfx as usize, cfy as usize, layer);
                                    if self.obstacle_net_ids[cidx] == net1 {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Determines the physically adjacent reference plane layer for a given signal layer.
    pub fn adjacent_reference_plane(&self, signal_layer: usize) -> Option<usize> {
        if self.plane_nets.is_empty() {
            return None;
        }
        match signal_layer {
            0 => {
                if self.plane_nets.contains_key(&2) {
                    Some(2)
                } else {
                    self.plane_nets.keys().copied().min()
                }
            }
            1 => {
                if self.plane_nets.contains_key(&3) {
                    Some(3)
                } else if self.plane_nets.contains_key(&2) {
                    Some(2)
                } else {
                    self.plane_nets.keys().copied().next()
                }
            }
            l => {
                self.plane_nets.keys().copied().min_by_key(|&p| (p as isize - l as isize).abs())
            }
        }
    }

    /// Checks if the reference plane at (gx, gy) on ref_layer has a split, cutout, or void.
    pub fn is_reference_plane_void_at(&self, gx: usize, gy: usize, ref_layer: usize) -> bool {
        if gx >= self.width || gy >= self.height || ref_layer >= self.num_layers {
            return true;
        }
        let idx = self.idx(gx, gy, ref_layer);
        let obst = self.obstacle_net_ids[idx];

        // 1. Explicit keepout/cutout zone on plane layer
        if obst == KEEPOUT_NET {
            return true;
        }

        // 2. Foreign net crossing on dedicated plane layer (e.g. split plane boundary into another net)
        if let Some(&plane_net) = self.plane_nets.get(&ref_layer) {
            if obst > 0 && obst != plane_net as i32 {
                return true;
            }
        }
        false
    }
}

pub fn dist_to_segment(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-6 {
        (px - x1).hypot(py - y1)
    } else {
        let t = (((px - x1) * dx + (py - y1) * dy) / len_sq).clamp(0.0, 1.0);
        (px - (x1 + t * dx)).hypot(py - (y1 + t * dy))
    }
}

pub fn segments_intersect(p1: (f64, f64), p2: (f64, f64), p3: (f64, f64), p4: (f64, f64)) -> bool {
    fn ccw(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
        (c.1 - a.1) * (b.0 - a.0) - (b.1 - a.1) * (c.0 - a.0)
    }
    let d1 = ccw(p3, p4, p1);
    let d2 = ccw(p3, p4, p2);
    let d3 = ccw(p1, p2, p3);
    let d4 = ccw(p1, p2, p4);

    ((d1 > 1e-9 && d2 < -1e-9) || (d1 < -1e-9 && d2 > 1e-9))
        && ((d3 > 1e-9 && d4 < -1e-9) || (d3 < -1e-9 && d4 > 1e-9))
}

pub fn dist_segment_to_segment(p1: (f64, f64), p2: (f64, f64), p3: (f64, f64), p4: (f64, f64)) -> f64 {
    if segments_intersect(p1, p2, p3, p4) {
        return 0.0;
    }
    dist_to_segment(p1.0, p1.1, p3.0, p3.1, p4.0, p4.1)
        .min(dist_to_segment(p2.0, p2.1, p3.0, p3.1, p4.0, p4.1))
        .min(dist_to_segment(p3.0, p3.1, p1.0, p1.1, p2.0, p2.1))
        .min(dist_to_segment(p4.0, p4.1, p1.0, p1.1, p2.0, p2.1))
}

pub struct NudgeEngine {
    pub step_mm: f64,
    pub max_displacement_mm: f64,
}

impl NudgeEngine {
    pub fn new() -> Self {
        Self {
            step_mm: 0.25,
            max_displacement_mm: 2.0,
        }
    }

    pub fn nudge_candidates(&self, push_dir: (f64, f64)) -> Vec<(f64, f64)> {
        let mag = (push_dir.0 * push_dir.0 + push_dir.1 * push_dir.1).sqrt();
        let (ux, uy) = if mag < 1e-6 {
            (1.0, 0.0)
        } else {
            (push_dir.0 / mag, push_dir.1 / mag)
        };
        let (px, py) = (-uy, ux);

        let mut candidates = Vec::new();
        // 1. Pure axial and lateral steps
        for &d in &[0.25, 0.50, 1.0, 1.5, 2.0] {
            if d <= self.max_displacement_mm + 1e-4 {
                candidates.push((ux * d, uy * d));
                candidates.push((px * d, py * d));
                candidates.push((-px * d, -py * d));
                candidates.push((-ux * d, -uy * d));
            }
        }
        // 2. Symmetric diagonal moves
        for &d in &[0.5, 1.0, 1.5, 2.0] {
            let diag = d * 0.70710678;
            candidates.push(((ux + px) * diag, (uy + py) * diag));
            candidates.push(((ux - px) * diag, (uy - py) * diag));
            candidates.push(((-ux + px) * diag, (-uy + py) * diag));
            candidates.push(((-ux - px) * diag, (-uy - py) * diag));
        }
        // 3. Corridor exit compound moves (push forward along corridor, shift laterally out of centerline)
        for &fwd in &[1.0f64, 1.5, 2.0] {
            for &lat in &[0.5f64, 1.0, 1.5] {
                if (fwd * fwd + lat * lat).sqrt() <= self.max_displacement_mm + 1e-4 {
                    candidates.push((ux * fwd + px * lat, uy * fwd + py * lat));
                    candidates.push((ux * fwd - px * lat, uy * fwd - py * lat));
                }
            }
        }

        let mut unique = Vec::with_capacity(candidates.len());
        for (cx, cy) in candidates {
            if !unique.iter().any(|(ux, uy): &(f64, f64)| (ux - cx).hypot(uy - cy) < 1e-4) {
                unique.push((cx, cy));
            }
        }
        unique
    }

    pub fn is_placement_valid(&self, board: &Board, fp_idx: usize, nx: f64, ny: f64) -> bool {
        if fp_idx >= board.footprints.len() {
            return false;
        }
        let rot = board.footprints[fp_idx].rotation;
        let fp_layer = board.footprints[fp_idx].layer;
        let fp_smd = board.footprints[fp_idx].is_smd_only();
        let boundary = board.boundary_aabb();

        let new_polys = board.footprints[fp_idx].transformed_courtyards(Point::new(nx, ny), rot, 0.20);

        for poly in &new_polys {
            for v in &poly.vertices {
                if !boundary.contains_point(v) {
                    return false;
                }
            }
        }

        for ko in &board.keepouts {
            if !ko.layers.contains(&fp_layer) {
                continue;
            }
            for poly in &new_polys {
                let aabb = poly.aabb();
                if aabb.max_x > ko.min_x && aabb.min_x < ko.max_x && aabb.max_y > ko.min_y && aabb.min_y < ko.max_y {
                    return false;
                }
            }
        }

        for (other_idx, other_fp) in board.footprints.iter().enumerate() {
            if other_idx == fp_idx {
                continue;
            }
            if fp_layer != other_fp.layer && fp_smd && other_fp.is_smd_only() {
                continue;
            }
            let other_polys = other_fp.transformed_courtyards(Point::new(other_fp.x, other_fp.y), other_fp.rotation, 0.20);
            for pa in &new_polys {
                for pb in &other_polys {
                    if pa.intersects_convex(pb) {
                        return false;
                    }
                }
            }
        }

        // Ensure moved footprint pads do not collide with existing tracks of other nets
        for pad in &board.footprints[fp_idx].pads {
            if pad.net_id == 0 {
                continue;
            }
            let oy = if board.footprints[fp_idx].layer != board.footprints[fp_idx].orig_layer {
                -pad.offset_y
            } else {
                pad.offset_y
            };
            let p = Point::new(pad.offset_x, oy).rotate_ccw(rot);
            let pad_x = nx + p.x;
            let pad_y = ny + p.y;
            let (pw, ph) = board.footprints[fp_idx].pad_world_size(pad);
            let pad_radius = (pw.max(ph) / 2.0).max(0.35);

            for track in &board.tracks {
                if track.layer != fp_layer || track.net_id == pad.net_id {
                    continue;
                }
                let track_c = 0.15;
                let d = dist_to_segment(pad_x, pad_y, track.start.0, track.start.1, track.end.0, track.end.1);
                if d < pad_radius + (track.width / 2.0) + track_c {
                    return false;
                }
            }
        }

        true
    }

    pub fn try_nudge_component(&self, board: &mut Board, fp_idx: usize, push_dir: (f64, f64)) -> bool {
        if fp_idx >= board.footprints.len() || board.footprints[fp_idx].is_locked {
            return false;
        }
        let mag = (push_dir.0 * push_dir.0 + push_dir.1 * push_dir.1).sqrt();
        if mag < 1e-6 {
            return false;
        }
        let orig_x = board.footprints[fp_idx].x;
        let orig_y = board.footprints[fp_idx].y;

        for (dx, dy) in self.nudge_candidates(push_dir) {
            let nx = orig_x + dx;
            let ny = orig_y + dy;
            if self.is_placement_valid(board, fp_idx, nx, ny) {
                board.rip_up_connected_traces_to_component(fp_idx);
                board.footprints[fp_idx].x = nx;
                board.footprints[fp_idx].y = ny;
                return true;
            }
        }
        false
    }

    /// Shoves an unlocked track segment perpendicular to its line to create clearance.
    pub fn try_shove_track(
        &self,
        board: &mut Board,
        track_idx: usize,
        dir_sign: f64,
        grid: Option<&RoutingGrid>,
    ) -> bool {
        if track_idx >= board.tracks.len() || board.tracks[track_idx].is_locked {
            return false;
        }
        let t = &board.tracks[track_idx];
        let vx = t.end.0 - t.start.0;
        let vy = t.end.1 - t.start.1;
        let len = vx.hypot(vy);
        if len < 1e-6 {
            return false;
        }
        let sign = if dir_sign >= 0.0 { 1.0 } else { -1.0 };
        let nx = (-vy / len) * sign;
        let ny = (vx / len) * sign;

        let shift_candidates = [
            (nx * self.step_mm, ny * self.step_mm),
            (-nx * self.step_mm, -ny * self.step_mm),
            (nx * self.step_mm * 2.0, ny * self.step_mm * 2.0),
            (-nx * self.step_mm * 2.0, -ny * self.step_mm * 2.0),
        ];

        let boundary = board.boundary_aabb();
        for (sx, sy) in shift_candidates {
            let new_start = (t.start.0 + sx, t.start.1 + sy);
            let new_end = (t.end.0 + sx, t.end.1 + sy);

            if !boundary.contains_point(&Point::new(new_start.0, new_start.1))
                || !boundary.contains_point(&Point::new(new_end.0, new_end.1))
            {
                continue;
            }

            // Clearance to pads of other nets
            let req_clearance = 0.20;
            let eff_dist = req_clearance + t.width / 2.0;
            let mut collides = false;
            for fp in &board.footprints {
                for pad in &fp.pads {
                    if pad.net_id != t.net_id && pad.net_id != 0 {
                        let (px, py) = fp.pad_absolute_position(pad);
                        let (pw, ph) = fp.pad_world_size(pad);
                        let pad_radius = pw.max(ph) / 2.0;
                        if (px - new_start.0).hypot(py - new_start.1) < eff_dist + pad_radius
                            || (px - new_end.0).hypot(py - new_end.1) < eff_dist + pad_radius
                        {
                            collides = true;
                            break;
                        }
                    }
                }
                if collides {
                    break;
                }
            }
            if collides {
                continue;
            }

            // Clearance to other tracks of other nets on the same layer
            for (other_i, other_t) in board.tracks.iter().enumerate() {
                if other_i == track_idx || other_t.layer != t.layer || other_t.net_id == t.net_id {
                    continue;
                }
                let seg_a = crate::board::Segment {
                    p1: Point::new(new_start.0, new_start.1),
                    p2: Point::new(new_end.0, new_end.1),
                    net_id: t.net_id,
                };
                let seg_b = crate::board::Segment {
                    p1: Point::new(other_t.start.0, other_t.start.1),
                    p2: Point::new(other_t.end.0, other_t.end.1),
                    net_id: other_t.net_id,
                };
                if seg_a.crosses(&seg_b) {
                    collides = true;
                    break;
                }
            }
            if collides {
                continue;
            }

            // Grid verification if provided
            if let Some(g) = grid {
                let steps = (len / (g.res * 0.5)).ceil() as usize;
                for s in 0..=steps {
                    let frac = s as f64 / steps.max(1) as f64;
                    let wx = new_start.0 + frac * (new_end.0 - new_start.0);
                    let wy = new_start.1 + frac * (new_end.1 - new_start.1);
                    if let Some((gx, gy)) = g.world_to_grid(wx, wy) {
                        let idx = g.idx(gx, gy, t.layer);
                        let obst = g.obstacle_net_ids[idx];
                        if obst != FREE_NET && obst != t.net_id as i32 {
                            collides = true;
                            break;
                        }
                    } else {
                        collides = true;
                        break;
                    }
                }
            }
            if collides {
                continue;
            }

            board.tracks[track_idx].start = new_start;
            board.tracks[track_idx].end = new_end;
            return true;
        }
        false
    }

    /// Shoves an unlocked via to a nearby legal location, extending or dogleg-connecting its incident tracks.
    pub fn try_shove_via(
        &self,
        board: &mut Board,
        via_idx: usize,
        push_dir: (f64, f64),
        grid: Option<&RoutingGrid>,
    ) -> bool {
        if via_idx >= board.vias.len() || board.vias[via_idx].is_locked {
            return false;
        }
        let orig_pos = board.vias[via_idx].position;
        let via_size = board.vias[via_idx].size;
        let via_net = board.vias[via_idx].net_id;
        let clearance = board.get_netclass_for_net(via_net).clearance;

        let mag = (push_dir.0 * push_dir.0 + push_dir.1 * push_dir.1).sqrt();
        let (pdx, pdy) = if mag > 1e-6 {
            (push_dir.0 / mag, push_dir.1 / mag)
        } else {
            (1.0, 0.0)
        };

        // Candidate radial and cardinal shifts
        let mut candidates = Vec::new();
        for mult in [1.0, 2.0, 3.0] {
            let s = self.step_mm * mult;
            candidates.push((pdx * s, pdy * s));
            candidates.push((-pdy * s, pdx * s));
            candidates.push((pdy * s, -pdx * s));
            candidates.push((-pdx * s, -pdy * s));
            candidates.push((s, 0.0));
            candidates.push((-s, 0.0));
            candidates.push((0.0, s));
            candidates.push((0.0, -s));
            candidates.push((s, s));
            candidates.push((-s, s));
            candidates.push((s, -s));
            candidates.push((-s, -s));
        }

        let q_orig = quant_point(orig_pos);
        let incident_track_indices: Vec<(usize, bool)> = board
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.net_id == via_net && !t.is_locked)
            .filter_map(|(idx, t)| {
                if quant_point(t.start) == q_orig {
                    Some((idx, true))
                } else if quant_point(t.end) == q_orig {
                    Some((idx, false))
                } else {
                    None
                }
            })
            .collect();

        let boundary = board.boundary_aabb();

        for (dx, dy) in candidates {
            let cand_pos = (orig_pos.0 + dx, orig_pos.1 + dy);
            if !boundary.contains_point(&Point::new(cand_pos.0, cand_pos.1)) {
                continue;
            }

            if !is_via_location_legal(board, cand_pos, via_size, via_net, clearance) {
                continue;
            }

            if let Some(g) = grid {
                if let Some((gx, gy)) = g.world_to_grid(cand_pos.0, cand_pos.1) {
                    let mut grid_ok = true;
                    for l in 0..g.num_layers {
                        let idx = g.idx(gx, gy, l);
                        let obst = g.obstacle_net_ids[idx];
                        if obst != FREE_NET && obst != via_net as i32 {
                            grid_ok = false;
                            break;
                        }
                    }
                    if !grid_ok {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            let mut tracks_valid = true;
            for &(trk_idx, at_start) in &incident_track_indices {
                let trk = &board.tracks[trk_idx];
                let anchor = if at_start { trk.end } else { trk.start };
                if !is_segment_clear(board, grid, anchor, cand_pos, trk.layer, via_net, trk.width)
                    || !is_track_segment_legal(board, anchor, cand_pos, trk.width, trk.layer, via_net, clearance)
                {
                    tracks_valid = false;
                    break;
                }
            }
            if !tracks_valid {
                continue;
            }

            board.vias[via_idx].position = cand_pos;
            for &(trk_idx, at_start) in &incident_track_indices {
                if at_start {
                    board.tracks[trk_idx].start = cand_pos;
                } else {
                    board.tracks[trk_idx].end = cand_pos;
                }
            }
            return true;
        }

        false
    }

    /// Shoves an unlocked track segment continuously away from an intruder segment,
    /// inserting 45-degree transition elbows to preserve original endpoint connectivity.
    /// Supports cascading shove into adjacent unlocked tracks up to 3 recursion hops.
    pub fn try_continuous_shove_track(
        &self,
        board: &mut Board,
        track_idx: usize,
        intruder_p1: (f64, f64),
        intruder_p2: (f64, f64),
        intruder_width: f64,
        intruder_net_id: usize,
        grid: Option<&RoutingGrid>,
    ) -> bool {
        let mut shoved_nets = std::collections::HashSet::new();
        shoved_nets.insert(intruder_net_id);
        let mut total_shoves = 0;
        self.try_continuous_shove_recursive(
            board,
            track_idx,
            intruder_p1,
            intruder_p2,
            intruder_width,
            intruder_net_id,
            grid,
            0,
            &mut shoved_nets,
            &mut total_shoves,
        )
    }

    /// Shoves an unlocked track segment with a specific lateral offset (in mm) along its perpendicular normal,
    /// inserting 45-degree transition elbows and cascading into neighbors if needed.
    pub fn try_continuous_shove_deflect(
        &self,
        board: &mut Board,
        track_idx: usize,
        offset_mm: f64,
        grid: Option<&RoutingGrid>,
    ) -> bool {
        if track_idx >= board.tracks.len() || board.tracks[track_idx].is_locked {
            return false;
        }
        let t = board.tracks[track_idx].clone();
        let vx = t.end.0 - t.start.0;
        let vy = t.end.1 - t.start.1;
        let len = vx.hypot(vy);
        if len < 1e-4 {
            return false;
        }
        let u = (vx / len, vy / len);
        let n = (-u.1, u.0);
        let normal = if offset_mm >= 0.0 { n } else { (-n.0, -n.1) };
        let mut shoved_nets = std::collections::HashSet::new();
        let mut total_shoves = 0;
        self.execute_continuous_deflection(
            board,
            track_idx,
            normal,
            offset_mm.abs(),
            grid,
            0,
            t.net_id,
            &mut shoved_nets,
            &mut total_shoves,
        )
    }

    pub fn try_continuous_shove_recursive(
        &self,
        board: &mut Board,
        track_idx: usize,
        intruder_p1: (f64, f64),
        intruder_p2: (f64, f64),
        intruder_width: f64,
        intruder_net_id: usize,
        grid: Option<&RoutingGrid>,
        depth: usize,
        shoved_nets: &mut std::collections::HashSet<usize>,
        total_shoves: &mut usize,
    ) -> bool {
        const MAX_DEPTH: usize = 3;
        const MAX_TOTAL_SHOVES: usize = 8;

        if depth >= MAX_DEPTH || *total_shoves >= MAX_TOTAL_SHOVES {
            return false;
        }
        if track_idx >= board.tracks.len() || board.tracks[track_idx].is_locked {
            return false;
        }

        let t = board.tracks[track_idx].clone();
        if shoved_nets.contains(&t.net_id) {
            return false;
        }

        let vx = t.end.0 - t.start.0;
        let vy = t.end.1 - t.start.1;
        let len = vx.hypot(vy);
        if len < 1e-4 {
            return false;
        }

        let dist = dist_segment_to_segment(t.start, t.end, intruder_p1, intruder_p2);
        let clr_t = board.get_netclass_for_net(t.net_id).clearance;
        let clr_i = board.get_netclass_for_net(intruder_net_id).clearance;
        let req_dist = (t.width + intruder_width) * 0.5 + clr_t.max(clr_i);

        if dist >= req_dist - 1e-4 {
            return true;
        }

        let penetration = req_dist - dist;
        let offset = (penetration + 0.05).max(0.10);

        let u = (vx / len, vy / len);
        let n = (-u.1, u.0);

        let mid_t = ((t.start.0 + t.end.0) * 0.5, (t.start.1 + t.end.1) * 0.5);
        let mid_i = ((intruder_p1.0 + intruder_p2.0) * 0.5, (intruder_p1.1 + intruder_p2.1) * 0.5);
        let w = (mid_t.0 - mid_i.0, mid_t.1 - mid_i.1);
        let dot = w.0 * n.0 + w.1 * n.1;

        let primary_dir = if dot >= 0.0 { n } else { (-n.0, -n.1) };
        let secondary_dir = (-primary_dir.0, -primary_dir.1);

        let snapshot_shoved = shoved_nets.clone();
        let snapshot_count = *total_shoves;
        if self.execute_continuous_deflection(
            board,
            track_idx,
            primary_dir,
            offset,
            grid,
            depth,
            intruder_net_id,
            shoved_nets,
            total_shoves,
        ) {
            return true;
        }

        if dot.abs() < 1e-4 {
            *shoved_nets = snapshot_shoved;
            *total_shoves = snapshot_count;
            if self.execute_continuous_deflection(
                board,
                track_idx,
                secondary_dir,
                offset,
                grid,
                depth,
                intruder_net_id,
                shoved_nets,
                total_shoves,
            ) {
                return true;
            }
        }

        false
    }

    fn execute_continuous_deflection(
        &self,
        board: &mut Board,
        track_idx: usize,
        dir_normal: (f64, f64),
        offset_mm: f64,
        grid: Option<&RoutingGrid>,
        depth: usize,
        intruder_net_id: usize,
        shoved_nets: &mut std::collections::HashSet<usize>,
        total_shoves: &mut usize,
    ) -> bool {
        if track_idx >= board.tracks.len() || board.tracks[track_idx].is_locked {
            return false;
        }

        let orig_track = board.tracks[track_idx].clone();
        let a = orig_track.start;
        let b = orig_track.end;
        let vx = b.0 - a.0;
        let vy = b.1 - a.1;
        let len = vx.hypot(vy);
        if len < 1e-4 {
            return false;
        }

        let u = (vx / len, vy / len);
        let d = (dir_normal.0 * offset_mm, dir_normal.1 * offset_mm);

        let candidate_segments: Vec<((f64, f64), (f64, f64))> = if len > 2.0 * offset_mm {
            let s = offset_mm;
            let j1 = (a.0 + s * u.0 + d.0, a.1 + s * u.1 + d.1);
            let j2 = (b.0 - s * u.0 + d.0, b.1 - s * u.1 + d.1);
            vec![(a, j1), (j1, j2), (j2, b)]
        } else {
            let m = ((a.0 + b.0) * 0.5 + d.0, (a.1 + b.1) * 0.5 + d.1);
            vec![(a, m), (m, b)]
        };

        // Guard: do not introduce non-octilinear track segments during continuous shove
        if candidate_segments.iter().any(|&(p1, p2)| !is_octilinear_segment(p1, p2)) {
            return false;
        }

        let backup_tracks = board.tracks.clone();

        // 1. Guard against board boundary
        let boundary = board.boundary_aabb();
        for &(p1, p2) in &candidate_segments {
            if !boundary.contains_point(&Point::new(p1.0, p1.1))
                || !boundary.contains_point(&Point::new(p2.0, p2.1))
            {
                return false;
            }
        }

        let clr_t = board.get_netclass_for_net(orig_track.net_id).clearance;
        let half_w = orig_track.width * 0.5;

        // 2. Guard against keepouts
        for &(p1, p2) in &candidate_segments {
            for ko in &board.keepouts {
                if ko.layers.contains(&orig_track.layer) {
                    let margin = half_w + clr_t;
                    let k_min_x = ko.min_x - margin;
                    let k_max_x = ko.max_x + margin;
                    let k_min_y = ko.min_y - margin;
                    let k_max_y = ko.max_y + margin;

                    if (p1.0 >= k_min_x && p1.0 <= k_max_x && p1.1 >= k_min_y && p1.1 <= k_max_y)
                        || (p2.0 >= k_min_x && p2.0 <= k_max_x && p2.1 >= k_min_y && p2.1 <= k_max_y)
                    {
                        return false;
                    }

                    let corners = [
                        (ko.min_x, ko.min_y),
                        (ko.max_x, ko.min_y),
                        (ko.max_x, ko.max_y),
                        (ko.min_x, ko.max_y),
                    ];
                    for k in 0..4 {
                        let e1 = corners[k];
                        let e2 = corners[(k + 1) % 4];
                        if dist_segment_to_segment(p1, p2, e1, e2) < margin - 1e-4 {
                            return false;
                        }
                    }
                }
            }
        }

        // 3. Guard against foreign pads
        for &(p1, p2) in &candidate_segments {
            for fp in &board.footprints {
                if fp.layer != orig_track.layer {
                    continue;
                }
                for pad in &fp.pads {
                    if pad.net_id != orig_track.net_id {
                        let (px, py) = fp.pad_absolute_position(pad);
                        let (pw, ph) = fp.pad_world_size(pad);
                        let pad_rad = pw.max(ph) * 0.5;
                        let pad_clr = board.get_netclass_for_net(pad.net_id).clearance;
                        let req_clr = clr_t.max(pad_clr);
                        if dist_to_segment(px, py, p1.0, p1.1, p2.0, p2.1) < half_w + pad_rad + req_clr - 1e-4 {
                            return false;
                        }
                    }
                }
            }
        }

        // 4. Guard against foreign vias
        for &(p1, p2) in &candidate_segments {
            for via in &board.vias {
                if via.net_id != orig_track.net_id {
                    let via_rad = via.size * 0.5;
                    let via_clr = board.get_netclass_for_net(via.net_id).clearance;
                    let req_clr = clr_t.max(via_clr);
                    if dist_to_segment(via.position.0, via.position.1, p1.0, p1.1, p2.0, p2.1) < half_w + via_rad + req_clr - 1e-4 {
                        return false;
                    }
                }
            }
        }

        // Mark orig_track's net as shoved
        shoved_nets.insert(orig_track.net_id);
        *total_shoves += 1;

        // 5. Guard against locked foreign tracks
        for &(p1, p2) in &candidate_segments {
            for (idx, trk) in board.tracks.iter().enumerate() {
                if idx == track_idx
                    || trk.net_id == orig_track.net_id
                    || trk.net_id == intruder_net_id
                    || trk.layer != orig_track.layer
                {
                    continue;
                }
                if trk.is_locked {
                    let trk_clr = board.get_netclass_for_net(trk.net_id).clearance;
                    let req_dist = half_w + trk.width * 0.5 + clr_t.max(trk_clr);
                    if dist_segment_to_segment(p1, p2, trk.start, trk.end) < req_dist - 1e-4 {
                        board.tracks = backup_tracks;
                        return false;
                    }
                }
            }
        }

        // 6. Cascading shove into unlocked foreign tracks
        // Snapshot existing track count to avoid unbounded iteration on newly appended tracks
        let initial_track_count = board.tracks.len();
        for &(p1, p2) in &candidate_segments {
            for o_idx in 0..initial_track_count {
                if o_idx >= board.tracks.len() {
                    break;
                }
                let o_trk = board.tracks[o_idx].clone();
                if o_idx == track_idx
                    || o_trk.is_locked
                    || o_trk.net_id == orig_track.net_id
                    || o_trk.net_id == intruder_net_id
                    || o_trk.layer != orig_track.layer
                    || shoved_nets.contains(&o_trk.net_id)
                {
                    continue;
                }

                let o_clr = board.get_netclass_for_net(o_trk.net_id).clearance;
                let req_dist = half_w + o_trk.width * 0.5 + clr_t.max(o_clr);
                let d_curr = dist_segment_to_segment(p1, p2, o_trk.start, o_trk.end);

                if d_curr < req_dist - 1e-4 {
                    if depth >= 3 || *total_shoves >= 8 {
                        board.tracks = backup_tracks;
                        return false;
                    }
                    if !self.try_continuous_shove_recursive(
                        board,
                        o_idx,
                        p1,
                        p2,
                        orig_track.width,
                        orig_track.net_id,
                        grid,
                        depth + 1,
                        shoved_nets,
                        total_shoves,
                    ) {
                        board.tracks = backup_tracks;
                        return false;
                    }
                }
            }
        }

        let pos = board.tracks.iter().position(|t| {
            t.net_id == orig_track.net_id
                && t.layer == orig_track.layer
                && (t.start.0 - a.0).hypot(t.start.1 - a.1) < 1e-4
                && (t.end.0 - b.0).hypot(t.end.1 - b.1) < 1e-4
        });

        if let Some(p_idx) = pos {
            board.tracks.remove(p_idx);
        } else if track_idx < board.tracks.len() {
            board.tracks.remove(track_idx);
        }

        let insert_start = board.tracks.len();
        for &(p1, p2) in &candidate_segments {
            if (p1.0 - p2.0).hypot(p1.1 - p2.1) > 1e-5 {
                board.tracks.push(TrackSegment {
                    start: p1,
                    end: p2,
                    width: orig_track.width,
                    layer: orig_track.layer,
                    net_id: orig_track.net_id,
                    is_locked: false,
                });
            }
        }
        let insert_end = board.tracks.len();

        // 7. Verify DRC clearance of newly placed segments against remaining tracks
        for ni in insert_start..insert_end {
            let seg = &board.tracks[ni];
            for oi in 0..board.tracks.len() {
                if oi >= insert_start && oi < insert_end {
                    continue;
                }
                let oth = &board.tracks[oi];
                if oth.layer != seg.layer || oth.net_id == seg.net_id || oth.net_id == intruder_net_id {
                    continue;
                }
                let clr_o = board.get_netclass_for_net(oth.net_id).clearance;
                let req = (seg.width + oth.width) * 0.5 + clr_t.max(clr_o);
                let d = dist_segment_to_segment(seg.start, seg.end, oth.start, oth.end);
                if d < req - 1e-4 {
                    board.tracks = backup_tracks;
                    return false;
                }
            }
        }

        true
    }
}

#[derive(Clone, Copy, PartialEq)]
struct AStarNode {
    node: GridNode,
    cost: f64,
    priority: f64,
}

impl Eq for AStarNode {}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.partial_cmp(&self.priority).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug)]
pub struct RouteConnection {
    pub net_id: usize,
    pub net_name: String,
    pub start: (f64, f64, usize),
    pub end: (f64, f64, usize),
    pub start_fp: usize,
    pub start_pad: usize,
    pub end_fp: usize,
    pub end_pad: usize,
    pub netclass: NetClass,
    pub diff_pair_leader: Option<usize>,
}

impl RouteConnection {
    pub fn swap_endpoints(&mut self) {
        std::mem::swap(&mut self.start, &mut self.end);
        std::mem::swap(&mut self.start_fp, &mut self.end_fp);
        std::mem::swap(&mut self.start_pad, &mut self.end_pad);
    }
}

#[derive(Clone, Debug)]
pub struct EscapeStub {
    pub fp_idx: usize,
    pub net_id: usize,
    pub pad_pos: (f64, f64),
    pub tip_pos: (f64, f64),
    pub width: f64,
    pub via: Option<Via>,
    pub tip_layer: usize,
    pub layer: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PadInfo {
    pub x: f64,
    pub y: f64,
    pub layer: usize,
    pub fp_idx: usize,
    pub pad_idx: usize,
    pub is_driver: bool,
}

impl PadInfo {
    pub fn new(x: f64, y: f64, layer: usize, fp_idx: usize, pad_idx: usize) -> Self {
        Self {
            x,
            y,
            layer,
            fp_idx,
            pad_idx,
            is_driver: false,
        }
    }

    pub fn with_driver(mut self, is_driver: bool) -> Self {
        self.is_driver = is_driver;
        self
    }
}

pub fn build_net_topology_connections(
    net_id: usize,
    net_name: &str,
    netclass: &NetClass,
    pads: &[PadInfo],
    topology: NetTopology,
) -> Vec<RouteConnection> {
    if pads.len() < 2 {
        return Vec::new();
    }
    if pads.len() == 2 {
        let (p_start, p_end) = if pads[1].is_driver && !pads[0].is_driver {
            (&pads[1], &pads[0])
        } else {
            (&pads[0], &pads[1])
        };
        return vec![RouteConnection {
            net_id,
            net_name: net_name.to_string(),
            start: (p_start.x, p_start.y, p_start.layer),
            end: (p_end.x, p_end.y, p_end.layer),
            start_fp: p_start.fp_idx,
            start_pad: p_start.pad_idx,
            end_fp: p_end.fp_idx,
            end_pad: p_end.pad_idx,
            netclass: netclass.clone(),
            diff_pair_leader: None,
        }];
    }

    match topology {
        NetTopology::EuclideanMst => {
            let mut connections = Vec::with_capacity(pads.len() - 1);
            let mut visited = vec![false; pads.len()];
            visited[0] = true;
            let mut in_tree = 1;

            while in_tree < pads.len() {
                let mut best_dist = f64::MAX;
                let mut best_pair = (0, 0);
                for i in 0..pads.len() {
                    if !visited[i] {
                        continue;
                    }
                    for j in 0..pads.len() {
                        if visited[j] {
                            continue;
                        }
                        let same_fp_penalty = if pads[i].fp_idx == pads[j].fp_idx && pads.len() > 2 {
                            let d = (pads[i].x - pads[j].x).hypot(pads[i].y - pads[j].y);
                            if d < 2.5 { 0.0 } else { 10.0 }
                        } else {
                            0.0
                        };
                        let dist = (pads[i].x - pads[j].x).hypot(pads[i].y - pads[j].y) + same_fp_penalty;
                        if dist < best_dist {
                            best_dist = dist;
                            best_pair = (i, j);
                        }
                    }
                }
                visited[best_pair.1] = true;
                in_tree += 1;

                let p_start = &pads[best_pair.0];
                let p_end = &pads[best_pair.1];
                connections.push(RouteConnection {
                    net_id,
                    net_name: net_name.to_string(),
                    start: (p_start.x, p_start.y, p_start.layer),
                    end: (p_end.x, p_end.y, p_end.layer),
                    start_fp: p_start.fp_idx,
                    start_pad: p_start.pad_idx,
                    end_fp: p_end.fp_idx,
                    end_pad: p_end.pad_idx,
                    netclass: netclass.clone(),
                    diff_pair_leader: None,
                });
            }
            connections
        }
        NetTopology::DaisyChain | NetTopology::FlyBy => {
            let driver_idx = pads.iter().position(|p| p.is_driver).unwrap_or(0);
            let mut chain = Vec::with_capacity(pads.len());
            chain.push(driver_idx);
            let mut visited = vec![false; pads.len()];
            visited[driver_idx] = true;

            while chain.len() < pads.len() {
                let curr = *chain.last().unwrap();
                let mut best_next = None;
                let mut best_dist = f64::MAX;
                for j in 0..pads.len() {
                    if !visited[j] {
                        let dist = (pads[curr].x - pads[j].x).hypot(pads[curr].y - pads[j].y);
                        if dist < best_dist {
                            best_dist = dist;
                            best_next = Some(j);
                        }
                    }
                }
                if let Some(next) = best_next {
                    visited[next] = true;
                    chain.push(next);
                } else {
                    break;
                }
            }

            let mut connections = Vec::with_capacity(chain.len() - 1);
            for w in chain.windows(2) {
                let p_start = &pads[w[0]];
                let p_end = &pads[w[1]];
                connections.push(RouteConnection {
                    net_id,
                    net_name: net_name.to_string(),
                    start: (p_start.x, p_start.y, p_start.layer),
                    end: (p_end.x, p_end.y, p_end.layer),
                    start_fp: p_start.fp_idx,
                    start_pad: p_start.pad_idx,
                    end_fp: p_end.fp_idx,
                    end_pad: p_end.pad_idx,
                    netclass: netclass.clone(),
                    diff_pair_leader: None,
                });
            }
            connections
        }
        NetTopology::StarPoint => {
            let center_idx = pads.iter().position(|p| p.is_driver).unwrap_or(0);
            let p_center = &pads[center_idx];
            let mut connections = Vec::with_capacity(pads.len() - 1);
            for (i, p_other) in pads.iter().enumerate() {
                if i == center_idx {
                    continue;
                }
                connections.push(RouteConnection {
                    net_id,
                    net_name: net_name.to_string(),
                    start: (p_center.x, p_center.y, p_center.layer),
                    end: (p_other.x, p_other.y, p_other.layer),
                    start_fp: p_center.fp_idx,
                    start_pad: p_center.pad_idx,
                    end_fp: p_other.fp_idx,
                    end_pad: p_other.pad_idx,
                    netclass: netclass.clone(),
                    diff_pair_leader: None,
                });
            }
            connections
        }
        NetTopology::TBranch => {
            let driver_idx = pads.iter().position(|p| p.is_driver).unwrap_or(0);
            let driver = &pads[driver_idx];
            let receivers: Vec<PadInfo> = pads
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != driver_idx)
                .map(|(_, p)| p.clone())
                .collect();

            let mut connections = Vec::new();
            build_t_branch_tree(
                (driver.x, driver.y, driver.layer),
                driver.fp_idx,
                driver.pad_idx,
                &receivers,
                driver,
                net_id,
                net_name,
                netclass,
                &mut connections,
            );
            connections
        }
    }
}

fn build_t_branch_tree(
    source_pt: (f64, f64, usize),
    source_fp: usize,
    source_pad: usize,
    receivers: &[PadInfo],
    driver: &PadInfo,
    net_id: usize,
    net_name: &str,
    netclass: &NetClass,
    connections: &mut Vec<RouteConnection>,
) {
    if receivers.is_empty() {
        return;
    }
    if receivers.len() == 1 {
        let r = &receivers[0];
        connections.push(RouteConnection {
            net_id,
            net_name: net_name.to_string(),
            start: source_pt,
            end: (r.x, r.y, r.layer),
            start_fp: source_fp,
            start_pad: source_pad,
            end_fp: r.fp_idx,
            end_pad: r.pad_idx,
            netclass: netclass.clone(),
            diff_pair_leader: None,
        });
        return;
    }
    if receivers.len() == 2 {
        let r1 = &receivers[0];
        let r2 = &receivers[1];
        let split_x = (r1.x + r2.x) * 0.5;
        let split_y = (r1.y + r2.y) * 0.5;
        let split_layer = r1.layer;
        let split_pt = (split_x, split_y, split_layer);

        // Trunk from source to split node
        connections.push(RouteConnection {
            net_id,
            net_name: net_name.to_string(),
            start: source_pt,
            end: split_pt,
            start_fp: source_fp,
            start_pad: source_pad,
            end_fp: driver.fp_idx,
            end_pad: driver.pad_idx,
            netclass: netclass.clone(),
            diff_pair_leader: None,
        });

        // Branch 1 to r1
        connections.push(RouteConnection {
            net_id,
            net_name: net_name.to_string(),
            start: split_pt,
            end: (r1.x, r1.y, r1.layer),
            start_fp: driver.fp_idx,
            start_pad: driver.pad_idx,
            end_fp: r1.fp_idx,
            end_pad: r1.pad_idx,
            netclass: netclass.clone(),
            diff_pair_leader: None,
        });

        // Branch 2 to r2
        connections.push(RouteConnection {
            net_id,
            net_name: net_name.to_string(),
            start: split_pt,
            end: (r2.x, r2.y, r2.layer),
            start_fp: driver.fp_idx,
            start_pad: driver.pad_idx,
            end_fp: r2.fp_idx,
            end_pad: r2.pad_idx,
            netclass: netclass.clone(),
            diff_pair_leader: None,
        });
        return;
    }

    // For > 2 receivers, split along axis of maximum variance to form balanced binary tree
    let mean_x = receivers.iter().map(|p| p.x).sum::<f64>() / receivers.len() as f64;
    let mean_y = receivers.iter().map(|p| p.y).sum::<f64>() / receivers.len() as f64;
    let var_x = receivers.iter().map(|p| (p.x - mean_x).powi(2)).sum::<f64>();
    let var_y = receivers.iter().map(|p| (p.y - mean_y).powi(2)).sum::<f64>();

    let mut sorted = receivers.to_vec();
    if var_x >= var_y {
        sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal));
    } else {
        sorted.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(Ordering::Equal));
    }

    let mid_idx = sorted.len() / 2;
    let left = &sorted[..mid_idx];
    let right = &sorted[mid_idx..];

    let left_cx = left.iter().map(|p| p.x).sum::<f64>() / left.len() as f64;
    let left_cy = left.iter().map(|p| p.y).sum::<f64>() / left.len() as f64;
    let right_cx = right.iter().map(|p| p.x).sum::<f64>() / right.len() as f64;
    let right_cy = right.iter().map(|p| p.y).sum::<f64>() / right.len() as f64;

    let split_x = (left_cx + right_cx) * 0.5;
    let split_y = (left_cy + right_cy) * 0.5;
    let split_pt = (split_x, split_y, driver.layer);

    connections.push(RouteConnection {
        net_id,
        net_name: net_name.to_string(),
        start: source_pt,
        end: split_pt,
        start_fp: source_fp,
        start_pad: source_pad,
        end_fp: driver.fp_idx,
        end_pad: driver.pad_idx,
        netclass: netclass.clone(),
        diff_pair_leader: None,
    });

    build_t_branch_tree(
        split_pt,
        driver.fp_idx,
        driver.pad_idx,
        left,
        driver,
        net_id,
        net_name,
        netclass,
        connections,
    );
    build_t_branch_tree(
        split_pt,
        driver.fp_idx,
        driver.pad_idx,
        right,
        driver,
        net_id,
        net_name,
        netclass,
        connections,
    );
}

pub struct AutoRouter {
    pub nudge_engine: NudgeEngine,
    pub max_ripup_iterations: usize,
    pub congestion_factor: f32,
    pub inner_layer_penalty: f64,
    pub allow_component_nudge: bool,
    pub crosstalk_penalty: f64,
    pub config: RoutingConfig,
    pub ipc: Option<Arc<Mutex<KiCadIpcClient>>>,
}

impl AutoRouter {
    pub fn new(max_ripup_iterations: usize) -> Self {
        let config = RoutingConfig::default();
        Self {
            nudge_engine: NudgeEngine::new(),
            max_ripup_iterations,
            congestion_factor: 1.5,
            inner_layer_penalty: 8.0,
            allow_component_nudge: false,
            crosstalk_penalty: config.crosstalk_penalty,
            config,
            ipc: None,
        }
    }

    pub fn with_ipc(mut self, ipc: Option<Arc<Mutex<KiCadIpcClient>>>) -> Self {
        self.ipc = ipc;
        self
    }

    pub fn with_crosstalk_penalty(mut self, pen: f64) -> Self {
        self.crosstalk_penalty = pen;
        self.config.crosstalk_penalty = pen;
        self
    }

    pub fn with_inner_layer_penalty(mut self, pen: f64) -> Self {
        self.inner_layer_penalty = pen;
        self
    }

    pub fn with_allow_component_nudge(mut self, allow: bool) -> Self {
        self.allow_component_nudge = allow;
        self
    }

    pub fn with_any_angle_mode(mut self, enabled: bool) -> Self {
        self.config.any_angle_mode = enabled;
        self
    }

    /// Returns true if `a` and `b` form a differential pair (e.g. `USB_DP`/`USB_DM`, `USB_D+`/`USB_D-`).
    pub fn are_diff_pair_mates(a: &str, b: &str) -> bool {
        if a == b {
            return false;
        }
        let ua = a.to_ascii_uppercase();
        let ub = b.to_ascii_uppercase();
        for (suf_p, suf_n) in [("DP", "DM"), ("D+", "D-"), ("_P", "_N"), ("+", "-"), ("P", "N")] {
            if let (Some(pa), Some(pb)) = (ua.strip_suffix(suf_p), ub.strip_suffix(suf_n)) {
                if !pa.is_empty() && pa == pb {
                    return true;
                }
            }
            if let (Some(pa), Some(pb)) = (ua.strip_suffix(suf_n), ub.strip_suffix(suf_p)) {
                if !pa.is_empty() && pa == pb {
                    return true;
                }
            }
        }
        false
    }

    pub fn build_diff_pair_corridor(
        &self,
        grid: &RoutingGrid,
        partner_path: &[GridNode],
        track_width: f64,
        req_clearance: f64,
    ) -> Vec<f32> {
        let mut corridor = vec![1.0f32; grid.num_layers * grid.width * grid.height];
        let ideal_pitch = track_width + req_clearance;
        let inner_band = ideal_pitch + grid.res * 0.5 + 0.35;
        let outer_band = ideal_pitch + grid.res * 0.5 + 0.90;
        let r_cells = (outer_band / grid.res).ceil() as isize;

        for &node in partner_path {
            for dy in -r_cells..=r_cells {
                for dx in -r_cells..=r_cells {
                    let nx = node.gx as isize + dx;
                    let ny = node.gy as isize + dy;
                    if nx < 0 || nx >= grid.width as isize || ny < 0 || ny >= grid.height as isize {
                        continue;
                    }
                    let dist = ((dx * dx + dy * dy) as f64).sqrt() * grid.res;
                    if dist <= outer_band {
                        let idx = grid.idx(nx as usize, ny as usize, node.layer);
                        let dev = (dist - ideal_pitch).abs();
                        let mult = if dev <= grid.res * 0.5 {
                            0.15f32
                        } else if dist <= inner_band {
                            0.35f32
                        } else {
                            0.65f32
                        };
                        if mult < corridor[idx] {
                            corridor[idx] = mult;
                        }
                    }
                }
            }
        }
        corridor
    }

    pub fn extract_connections(&self, board: &Board) -> Vec<RouteConnection> {
        let mut net_pads: HashMap<usize, Vec<PadInfo>> = HashMap::new();
        for (fp_idx, fp) in board.footprints.iter().enumerate() {
            for (pad_idx, pad) in fp.pads.iter().enumerate() {
                if pad.net_id == 0 {
                    continue;
                }
                let netname = &board.nets[pad.net_id];
                if netname == "GND" {
                    continue;
                }
                let (px, py) = fp.pad_absolute_position(pad);
                let pad_layer = if pad.is_through_hole { 0 } else { fp.layer };
                let entry = net_pads.entry(pad.net_id).or_default();
                if !entry
                    .iter()
                    .any(|p| p.layer == pad_layer && (p.x - px).hypot(p.y - py) < 0.05)
                {
                    let is_driver = fp.reference.starts_with('U')
                        || fp.reference.starts_with("IC")
                        || fp.reference.starts_with("MCU")
                        || fp.reference.starts_with('J')
                        || fp.reference.starts_with("VR")
                        || fp.reference.starts_with("REG");
                    entry.push(PadInfo {
                        x: px,
                        y: py,
                        layer: pad_layer,
                        fp_idx,
                        pad_idx,
                        is_driver,
                    });
                }
            }
        }

        let mut connections = Vec::new();
        let mut sorted_net_ids: Vec<usize> = net_pads.keys().copied().collect();
        sorted_net_ids.sort_unstable();
        for net_id in sorted_net_ids {
            let pads = &net_pads[&net_id];
            if pads.len() < 2 {
                continue;
            }
            let net_name = board.nets[net_id].clone();
            let netclass = board.get_netclass_for_net(net_id).clone();
            let eff_topo = netclass.resolve_topology(&net_name);

            let net_conns = build_net_topology_connections(net_id, &net_name, &netclass, pads, eff_topo);
            connections.extend(net_conns);
        }

        connections.sort_by(|a, b| {
            let a_diff = a.netclass.name == "USB_Diff"
                || a.net_name.ends_with("DP")
                || a.net_name.ends_with("DM")
                || a.net_name.ends_with("D+")
                || a.net_name.ends_with("D-");
            let b_diff = b.netclass.name == "USB_Diff"
                || b.net_name.ends_with("DP")
                || b.net_name.ends_with("DM")
                || b.net_name.ends_with("D+")
                || b.net_name.ends_with("D-");
            let prio_a = Self::netclass_priority(&a.netclass.name);
            let prio_b = Self::netclass_priority(&b.netclass.name);
            let a_is_ic = matches!(board.footprints[a.start_fp].reference.as_str(), "U1" | "J2")
                || matches!(board.footprints[a.end_fp].reference.as_str(), "U1" | "J2");
            let b_is_ic = matches!(board.footprints[b.start_fp].reference.as_str(), "U1" | "J2")
                || matches!(board.footprints[b.end_fp].reference.as_str(), "U1" | "J2");

            let d_a = (a.start.0 - a.end.0).hypot(a.start.1 - a.end.1);
            let d_b = (b.start.0 - b.end.0).hypot(b.start.1 - b.end.1);
            b_diff
                .cmp(&a_diff)
                .then_with(|| b_is_ic.cmp(&a_is_ic))
                .then_with(|| {
                    if a_is_ic && b_is_ic {
                        d_a.partial_cmp(&d_b).unwrap_or(Ordering::Equal)
                    } else {
                        prio_b.cmp(&prio_a)
                            .then_with(|| d_a.partial_cmp(&d_b).unwrap_or(Ordering::Equal))
                    }
                })
                .then_with(|| prio_b.cmp(&prio_a))
                .then_with(|| a.net_id.cmp(&b.net_id))
                .then_with(|| (a.start_fp, a.start_pad).cmp(&(b.start_fp, b.start_pad)))
                .then_with(|| (a.end_fp, a.end_pad).cmp(&(b.end_fp, b.end_pad)))
        });

        // Group differential pair and parallel multi-terminal connections (connecting two distinct footprints) adjacently
        // so the follower routes immediately alongside its leader trace.
        let mut i = 0;
        while i + 1 < connections.len() {
            if connections[i].diff_pair_leader.is_some() || connections[i].start_fp == connections[i].end_fp {
                i += 1;
                continue;
            }
            let fp_pair_i = (
                connections[i].start_fp.min(connections[i].end_fp),
                connections[i].start_fp.max(connections[i].end_fp),
            );
            let mut mate_idx = None;
            for j in (i + 1)..connections.len() {
                if connections[j].diff_pair_leader.is_some() || connections[j].start_fp == connections[j].end_fp {
                    continue;
                }
                let fp_pair_j = (
                    connections[j].start_fp.min(connections[j].end_fp),
                    connections[j].start_fp.max(connections[j].end_fp),
                );
                let is_diff = Self::are_diff_pair_mates(&connections[i].net_name, &connections[j].net_name)
                    || (connections[i].netclass.name == "USB_Diff" && connections[j].netclass.name == "USB_Diff");
                let is_twin = (board.footprints[fp_pair_i.0].reference.starts_with('C') || board.footprints[fp_pair_i.1].reference.starts_with('C'))
                    && (connections[i].net_name.contains("CAP") || connections[i].net_name.contains("HP"));

                if fp_pair_i == fp_pair_j && (is_diff || is_twin) {
                    mate_idx = Some((j, is_diff));
                    break;
                }
            }
            if let Some((j, is_diff)) = mate_idx {
                let mut follower = connections.remove(j);
                if connections[i].start_fp != follower.start_fp {
                    follower.swap_endpoints();
                }
                // Order: outer (longer Euclidean span) routes first, inner (shorter) routes second
                let d_i = (connections[i].start.0 - connections[i].end.0).hypot(connections[i].start.1 - connections[i].end.1);
                let d_j = (follower.start.0 - follower.end.0).hypot(follower.start.1 - follower.end.1);
                if d_i < d_j {
                    std::mem::swap(&mut connections[i], &mut follower);
                }
                if is_diff {
                    follower.diff_pair_leader = Some(i);
                }
                connections.insert(i + 1, follower);
                i += 2;
            } else {
                i += 1;
            }
        }

        connections
    }

    fn netclass_priority(name: &str) -> usize {
        match name {
            "Switching" | "Power_SW" => 10,
            "Power_15V" | "P1P2_Bus" | "Power" => 9,
            "USB_Diff" => 8,
            "HBS_Bus" | "Analog_Sense" => 7,
            "Power_3V3" | "Power_5V" => 6,
            _ => 1,
        }
    }

    pub fn route_all(
        &self,
        board: &mut Board,
        grid: &mut RoutingGrid,
        live: Option<Arc<Mutex<LiveTrialState>>>,
        stop_flag: Arc<AtomicBool>,
    ) -> (usize, usize) {
        // Clear out any old unlocked tracks/vias before routing so they don't block stubs or grid
        board.tracks.retain(|t| t.is_locked);
        board.vias.retain(|v| v.is_locked);

        if let Some(ref ipc_lock) = self.ipc {
            if let Ok(mut client) = ipc_lock.lock() {
                if client.is_connected() {
                    let _ = client.clear_tracks();
                    let _ = client.commit_batch();
                }
            }
        }

        grid.obstacle_net_ids.fill(FREE_NET);
        grid.nudgeable_fp.fill(None);
        grid.smd_pad_cells.fill(false);
        grid.rasterize_obstacles(board);
        grid.update_distance_fields();

        if self.allow_component_nudge && self.nudge_components_blocking_escapes(board) {
            grid.obstacle_net_ids.fill(FREE_NET);
            grid.nudgeable_fp.fill(None);
            grid.smd_pad_cells.fill(false);
            grid.rasterize_obstacles(board);
            grid.update_distance_fields();
        }

        let mut connections = self.extract_connections(board);
        let total_conns = connections.len();

        generate_bga_dogbone_escapes(board, grid);

        let escape_stubs = self.generate_escape_stubs(board);
        for stub in &escape_stubs {
            self.commit_stub_to_grid(grid, stub);
        }

        // Fanout First: Pre-route ground fanout/stitching for all SMD plane pads before signal routing
        let orig_locked_tracks: std::collections::HashSet<(usize, [u64; 4])> = board
            .tracks
            .iter()
            .filter(|t| t.is_locked)
            .map(|t| (t.net_id, [t.start.0.to_bits(), t.start.1.to_bits(), t.end.0.to_bits(), t.end.1.to_bits()]))
            .collect();
        let orig_locked_vias: std::collections::HashSet<(usize, [u64; 2])> = board
            .vias
            .iter()
            .filter(|v| v.is_locked)
            .map(|v| (v.net_id, [v.position.0.to_bits(), v.position.1.to_bits()]))
            .collect();

        let pre_tracks_count = board.tracks.len();
        let pre_vias_count = board.vias.len();

        self.connect_u1_ground_pins(board);

        for trk in &mut board.tracks[pre_tracks_count..] {
            trk.is_locked = true;
            grid.commit_track(trk);
        }
        for v in &mut board.vias[pre_vias_count..] {
            v.is_locked = true;
            grid.commit_via(v);
        }

        grid.update_distance_fields();

        for conn in &mut connections {
            for stub in &escape_stubs {
                if conn.net_id == stub.net_id
                    && conn.start.2 == stub.layer
                    && (conn.start.0 - stub.pad_pos.0).hypot(conn.start.1 - stub.pad_pos.1) < 0.15
                {
                    conn.start = (stub.tip_pos.0, stub.tip_pos.1, stub.tip_layer);
                    break;
                }
            }
            for stub in &escape_stubs {
                if conn.net_id == stub.net_id
                    && conn.end.2 == stub.layer
                    && (conn.end.0 - stub.pad_pos.0).hypot(conn.end.1 - stub.pad_pos.1) < 0.15
                {
                    conn.end = (stub.tip_pos.0, stub.tip_pos.1, stub.tip_layer);
                    break;
                }
            }
            let is_f1_f2_d2_d3 = board.footprints.get(conn.start_fp).map_or(false, |fp| matches!(fp.reference.as_str(), "F1" | "F2" | "D2" | "D3"))
                || board.footprints.get(conn.end_fp).map_or(false, |fp| matches!(fp.reference.as_str(), "F1" | "F2" | "D2" | "D3"));
            let is_passive_terminal = board.footprints.get(conn.start_fp).map_or(false, |fp| fp.reference.starts_with('R') || fp.reference.starts_with('C'))
                || board.footprints.get(conn.end_fp).map_or(false, |fp| fp.reference.starts_with('R') || fp.reference.starts_with('C'));
            if is_f1_f2_d2_d3 && !is_passive_terminal {
                if let Some(pwr) = board.netclasses.get("Power").or_else(|| board.netclasses.get("Power_15V")) {
                    conn.netclass = pwr.clone();
                }
            }
            let touches_fine_pitch = board.footprints.get(conn.start_fp).map_or(false, |fp| fp.reference == "U1" || fp.reference == "J2" || Self::is_dense_escape_footprint(fp))
                || board.footprints.get(conn.end_fp).map_or(false, |fp| fp.reference == "U1" || fp.reference == "J2" || Self::is_dense_escape_footprint(fp))
                || escape_stubs.iter().any(|s| {
                    (conn.start.0 - s.tip_pos.0).hypot(conn.start.1 - s.tip_pos.1) < 0.05
                        || (conn.end.0 - s.tip_pos.0).hypot(conn.end.1 - s.tip_pos.1) < 0.05
                });
            if touches_fine_pitch {
                if !conn.netclass.name.starts_with("Power_15V") {
                    conn.netclass.track_width = conn.netclass.track_width.min(0.15);
                }
                conn.netclass.diff_pair_width = conn.netclass.diff_pair_width.min(0.20);
                conn.netclass.diff_pair_gap = conn.netclass.diff_pair_gap.min(0.15);
                conn.netclass.clearance = conn.netclass.clearance.min(0.15);
                conn.netclass.via_dia = conn.netclass.via_dia.min(0.60);
            }
            if conn.netclass.name == "USB_Diff" || conn.net_name.contains("USB") {
                conn.netclass.diff_pair_width = conn.netclass.diff_pair_width.min(0.20);
                conn.netclass.diff_pair_gap = conn.netclass.diff_pair_gap.min(0.15);
            }
            if is_passive_terminal {
                if !conn.netclass.name.starts_with("Power_15V") {
                    conn.netclass.track_width = conn.netclass.track_width.min(0.20);
                }
                conn.netclass.clearance = conn.netclass.clearance.min(0.15);
                conn.netclass.via_dia = conn.netclass.via_dia.min(0.60);
            }
        }

        let u1_center = board
            .footprints
            .iter()
            .find(|fp| fp.reference == "U1")
            .map(|fp| (fp.x, fp.y));

        let num_threads = rayon::current_num_threads().max(2);
        grid.inner_layer_penalty = self.inner_layer_penalty;
        if let Some(ref st) = live {
            if let Ok(mut g) = st.lock() {
                g.phase = TrialPhase::AutoRouting;
                g.total_conns = total_conns;
                g.routed_conns = 0;
                let gpu_tag = if let Some(ref vk) = grid.vulkan {
                    format!(" + Vulkan GPU ({})", vk.device_name)
                } else {
                    String::new()
                };
                g.push_log(format!(
                    "[Auto-Router] Routing {} connections ({} layers) across {} CPU cores{} with {} escape stubs...",
                    total_conns,
                    grid.num_layers,
                    num_threads,
                    gpu_tag,
                    escape_stubs.len()
                ));
            }
        }

        let mut routed_paths: Vec<Option<Vec<GridNode>>> = vec![None; total_conns];
        let mut best_routed_paths: Vec<Option<Vec<GridNode>>> = vec![None; total_conns];
        let mut best_routed_count = 0;
        let mut unrouted_history: Vec<usize> = vec![0; total_conns];
        let gpu_done = AtomicBool::new(false);
        let shared_obs = Arc::new(Mutex::new(grid.obstacle_net_ids.clone()));
        let shared_df = Arc::new(Mutex::new(grid.dist_field.clone()));
        let vk_opt = grid.vulkan.clone();
        let grid_w = grid.width as u32;
        let grid_h = grid.height as u32;
        let grid_res = grid.res as f32;
        let layer_cells = grid.width * grid.height;
        let num_layers = grid.num_layers;

        std::thread::scope(|scope| {
            if let Some(vk) = vk_opt {
                let engines = vk.all_engines();
                let num_engines = engines.len();
                for (gpu_idx, vk_dev) in engines.into_iter().enumerate() {
                    let obs_ref = Arc::clone(&shared_obs);
                    let df_ref = Arc::clone(&shared_df);
                    let stop_ref = Arc::clone(&stop_flag);
                    let done_ref = &gpu_done;

                    scope.spawn(move || {
                        while !done_ref.load(AtomicOrdering::Relaxed)
                            && !stop_ref.load(AtomicOrdering::Relaxed)
                        {
                            let obs_snap = obs_ref.lock().unwrap().clone();
                            for layer in 0..num_layers {
                                if layer % num_engines != gpu_idx {
                                    continue;
                                }
                                if done_ref.load(AtomicOrdering::Relaxed)
                                    || stop_ref.load(AtomicOrdering::Relaxed)
                                {
                                    break;
                                }
                                let offset = layer * layer_cells;
                                let obs_u32: Vec<u32> = obs_snap[offset..offset + layer_cells]
                                    .iter()
                                    .map(|&n| if n != FREE_NET { 1u32 } else { 0u32 })
                                    .collect();
                                if let Some(df) = vk_dev.compute_distance_field(&obs_u32, grid_w, grid_h, 16) {
                                    if let Ok(mut g) = df_ref.lock() {
                                        for (dst, &val) in g[offset..offset + layer_cells]
                                            .iter_mut()
                                            .zip(df.iter())
                                        {
                                            *dst = val * grid_res;
                                        }
                                    }
                                }
                            }
                            std::thread::sleep(std::time::Duration::from_millis(5));
                        }
                    });
                }
            }

            let route_start_time = std::time::Instant::now();
            for iteration in 0..self.max_ripup_iterations {
                if stop_flag.load(AtomicOrdering::Relaxed) {
                    break;
                }
                if routed_paths.iter().all(|p| p.is_some()) {
                    best_routed_paths = routed_paths.clone();
                    best_routed_count = total_conns;
                    break;
                }

                for i in 0..total_conns {
                    if routed_paths[i].is_none() {
                        unrouted_history[i] += 1;
                    }
                }

                if let Ok(df_snap) = shared_df.lock() {
                    grid.dist_field.clone_from(&df_snap);
                }

                // PathFinder historical congestion decay to prevent late-pass grid saturation
                for c in &mut grid.congestion {
                    *c *= 0.85;
                }

                // ponytail: do not degrade user netclasses; maintain full physical widths and clearances

                // Phase 0: Coupled Differential Pair Bus Routing
                // Route any unrouted differential pair leader and immediately route its follower
                // within the coupled proximity corridor before single-ended nets can split the bus.
                for follower_idx in 0..total_conns {
                    let Some(leader_idx) = connections[follower_idx].diff_pair_leader else {
                        continue;
                    };
                    if routed_paths[leader_idx].is_some() && routed_paths[follower_idx].is_some() {
                        continue;
                    }
                    let lc = &connections[leader_idx];
                    let fc = &connections[follower_idx];
                    let diff_w = lc.netclass.diff_pair_width;
                    let diff_gap = lc.netclass.diff_pair_gap;

                    let mut was_leader_fresh = false;
                    if routed_paths[leader_idx].is_none() {
                        if let (Some((sgx, sgy)), Some((egx, egy))) = (
                            grid.world_to_grid(lc.start.0, lc.start.1),
                            grid.world_to_grid(lc.end.0, lc.end.1),
                        ) {
                            let s_node = GridNode { gx: sgx, gy: sgy, layer: lc.start.2 };
                            let e_node = GridNode { gx: egx, gy: egy, layer: lc.end.2 };
                            let lp_opt = self.find_path_with_constraints(
                                grid,
                                s_node,
                                e_node,
                                lc.net_id,
                                lc.netclass.clearance,
                                diff_w,
                                lc.netclass.via_dia,
                                u1_center,
                                1.0,
                                0.0,
                                lc.netclass.max_coupled_length_mm,
                                lc.netclass.require_return_path,
                                None,
                                &stop_flag,
                            ).or_else(|| self.find_path_parallel(
                                grid,
                                s_node,
                                e_node,
                                lc.net_id,
                                lc.netclass.clearance,
                                diff_w,
                                lc.netclass.via_dia,
                                u1_center,
                                lc.netclass.max_coupled_length_mm,
                                lc.netclass.require_return_path,
                                &stop_flag,
                            ));
                            if let Some(lp) = lp_opt {
                                self.commit_path_to_grid(
                                    grid,
                                    &lp,
                                    lc.net_id,
                                    diff_w,
                                    lc.netclass.via_dia,
                                );
                                routed_paths[leader_idx] = Some(lp);
                                was_leader_fresh = true;
                            }
                        }
                    }
                    if routed_paths[follower_idx].is_none() {
                        if let Some(ref leader_path) = routed_paths[leader_idx] {
                            let corridor = self.build_diff_pair_corridor(
                                grid,
                                leader_path,
                                diff_w,
                                diff_gap,
                            );
                            if let (Some((sgx, sgy)), Some((egx, egy))) = (
                                grid.world_to_grid(fc.start.0, fc.start.1),
                                grid.world_to_grid(fc.end.0, fc.end.1),
                            ) {
                                let s_node = GridNode { gx: sgx, gy: sgy, layer: fc.start.2 };
                                let e_node = GridNode { gx: egx, gy: egy, layer: fc.end.2 };
                                let fp_opt = self
                                    .find_path_with_constraints(
                                        grid,
                                        s_node,
                                        e_node,
                                        fc.net_id,
                                        fc.netclass.clearance,
                                        diff_w,
                                        fc.netclass.via_dia,
                                        u1_center,
                                        1.0,
                                        0.0,
                                        fc.netclass.max_coupled_length_mm,
                                        fc.netclass.require_return_path,
                                        Some(&corridor),
                                        &stop_flag,
                                    )
                                    .or_else(|| {
                                        self.find_path_parallel(
                                            grid,
                                            s_node,
                                            e_node,
                                            fc.net_id,
                                            fc.netclass.clearance,
                                            diff_w,
                                            fc.netclass.via_dia,
                                            u1_center,
                                            fc.netclass.max_coupled_length_mm,
                                            fc.netclass.require_return_path,
                                            &stop_flag,
                                        )
                                    });
                                if let Some(fp) = fp_opt {
                                    self.commit_path_to_grid(
                                        grid,
                                        &fp,
                                        fc.net_id,
                                        diff_w,
                                        fc.netclass.via_dia,
                                    );
                                    routed_paths[follower_idx] = Some(fp);
                                }
                            }
                        }
                        if routed_paths[follower_idx].is_none() && was_leader_fresh {
                            routed_paths[leader_idx] = None;
                            self.rebuild_grid_state(board, grid, &connections, &routed_paths, &escape_stubs);
                        }
                    }
                }

                let mut unrouted_indices: Vec<usize> = (0..total_conns)
                    .filter(|&i| routed_paths[i].is_none())
                    .collect();
                if unrouted_indices.is_empty() {
                    break;
                }
                unrouted_indices.sort_by_key(|&i| std::cmp::Reverse(unrouted_history[i]));

                // Phase 1: Speculative multi-core parallel A* wave across all remaining unrouted connections
                let wave_grid: &RoutingGrid = &*grid;
                let few_unrouted = unrouted_indices.len() <= 2;
                let speculative_paths: Vec<(usize, Option<Vec<GridNode>>)> = unrouted_indices
                    .par_iter()
                    .map(|&conn_idx| {
                        if stop_flag.load(AtomicOrdering::Relaxed) {
                            return (conn_idx, None);
                        }
                        let conn = &connections[conn_idx];
                        let Some((sgx, sgy)) = wave_grid.world_to_grid(conn.start.0, conn.start.1) else {
                            return (conn_idx, None);
                        };
                        let Some((egx, egy)) = wave_grid.world_to_grid(conn.end.0, conn.end.1) else {
                            return (conn_idx, None);
                        };
                        let start_grid = GridNode {
                            gx: sgx,
                            gy: sgy,
                            layer: conn.start.2,
                        };
                        let end_grid = GridNode {
                            gx: egx,
                            gy: egy,
                            layer: conn.end.2,
                        };
                        let p = if few_unrouted {
                            self.find_path_parallel(
                                wave_grid,
                                start_grid,
                                end_grid,
                                conn.net_id,
                                conn.netclass.clearance,
                                conn.netclass.track_width,
                                conn.netclass.via_dia,
                                u1_center,
                                conn.netclass.max_coupled_length_mm,
                                conn.netclass.require_return_path,
                                &stop_flag,
                            )
                        } else {
                            self.find_path_with_constraints(
                                wave_grid,
                                start_grid,
                                end_grid,
                                conn.net_id,
                                conn.netclass.clearance,
                                conn.netclass.track_width,
                                conn.netclass.via_dia,
                                u1_center,
                                1.0,
                                0.0,
                                conn.netclass.max_coupled_length_mm,
                                conn.netclass.require_return_path,
                                None,
                                &stop_flag,
                            )
                        };
                        (conn_idx, p)
                    })
                    .collect();

                // Phase 2: Commit conflict-free speculative paths in priority order; re-route conflicts in parallel across cores
                let mut _any_committed_in_wave = false;
                let mut sorted_speculative_paths = speculative_paths;
                sorted_speculative_paths.sort_by_key(|&(idx, _)| std::cmp::Reverse(unrouted_history[idx]));
                for (conn_idx, spec_path) in sorted_speculative_paths {
                    if stop_flag.load(AtomicOrdering::Relaxed) {
                        break;
                    }
                    let Some(p) = spec_path else {
                        continue;
                    };
                    let conn_start = connections[conn_idx].start;
                    let conn_end = connections[conn_idx].end;
                    let conn_net_id = connections[conn_idx].net_id;
                    let req_clearance = connections[conn_idx].netclass.clearance;
                    let track_width = connections[conn_idx].netclass.track_width;
                    let via_dia = connections[conn_idx].netclass.via_dia;

                    let start_grid = match grid.world_to_grid(conn_start.0, conn_start.1) {
                        Some((gx, gy)) => GridNode {
                            gx,
                            gy,
                            layer: conn_start.2,
                        },
                        None => continue,
                    };
                    let end_grid = match grid.world_to_grid(conn_end.0, conn_end.1) {
                        Some((gx, gy)) => GridNode {
                            gx,
                            gy,
                            layer: conn_end.2,
                        },
                        None => continue,
                    };

                    let valid_path = if self.is_path_conflict_free(
                            grid,
                            &p,
                            conn_net_id,
                            req_clearance,
                            track_width,
                            via_dia,
                            u1_center,
                            connections[conn_idx].netclass.require_return_path,
                        ) {
                        Some(p)
                    } else {
                        if let Ok(df_snap) = shared_df.lock() {
                            grid.dist_field.clone_from(&df_snap);
                        }
                        self.find_path_parallel(
                            grid,
                            start_grid,
                            end_grid,
                            conn_net_id,
                            req_clearance,
                            track_width,
                            via_dia,
                            u1_center,
                            connections[conn_idx].netclass.max_coupled_length_mm,
                            connections[conn_idx].netclass.require_return_path,
                            &stop_flag,
                        )
                    };

                    if let Some(p) = valid_path {
                        self.commit_path_to_grid(grid, &p, conn_net_id, track_width, via_dia);
                        routed_paths[conn_idx] = Some(p);
                        _any_committed_in_wave = true;
                        if let Ok(mut obs_g) = shared_obs.lock() {
                            obs_g.clone_from(&grid.obstacle_net_ids);
                        }
                    }

                    // Stream live progress to GUI every 4 connections
                    if conn_idx % 4 == 0 || conn_idx + 1 == total_conns {
                        if let Some(ref st) = live {
                            let done = routed_paths.iter().filter(|p| p.is_some()).count();
                            self.reconstruct_tracks_and_vias(board, grid, &connections, &routed_paths, &escape_stubs);
                            if let Some(ref ipc_lock) = self.ipc {
                                if let Ok(mut client) = ipc_lock.lock() {
                                    if client.is_connected() {
                                        let _ = client.clear_tracks();
                                        let _ = client.stream_tracks_and_vias(board);
                                    }
                                }
                            }
                            let remaining_ratsnest: Vec<crate::board::Segment> = connections
                                .iter()
                                .enumerate()
                                .filter(|&(i, _)| routed_paths[i].is_none())
                                .map(|(_, c)| {
                                    let p1 = board.footprints[c.start_fp].pad_absolute_position(&board.footprints[c.start_fp].pads[c.start_pad]);
                                    let p2 = board.footprints[c.end_fp].pad_absolute_position(&board.footprints[c.end_fp].pads[c.end_pad]);
                                    crate::board::Segment {
                                        p1: Point::new(p1.0, p1.1),
                                        p2: Point::new(p2.0, p2.1),
                                        net_id: c.net_id,
                                    }
                                })
                                .collect();
                            if let Ok(mut g) = st.lock() {
                                g.board.tracks = board.tracks.clone();
                                g.board.vias = board.vias.clone();
                                g.board.footprints = board.footprints.clone();
                                g.ratsnest = remaining_ratsnest;
                                g.routed_conns = done;
                                g.routing_pass = iteration + 1;
                                g.progress = 0.5 + 0.5 * (done as f32 / total_conns.max(1) as f32);
                                g.status_text = format!(
                                    "Pass {}/{}: Routed {}/{} ({:.1}%)",
                                    iteration + 1,
                                    self.max_ripup_iterations,
                                    done,
                                    total_conns,
                                    (done as f64 / total_conns.max(1) as f64) * 100.0
                                );
                            }
                        }
                    }
                }

                // Phase 2.5: Active Push & Nudge for unrouted connections
                // Attempt to nudge blocking unlocked components and shove/rip-up blocking tracks.
                for conn_idx in 0..total_conns {
                    if stop_flag.load(AtomicOrdering::Relaxed) {
                        break;
                    }
                    if routed_paths[conn_idx].is_some() {
                        continue;
                    }

                    // 1. Try nudging blocking unlocked components
                    if self.allow_component_nudge {
                        let nudged_comp = self.try_nudge_blocking_components(
                            board,
                            grid,
                            &mut connections,
                            &mut routed_paths,
                            &escape_stubs,
                            conn_idx,
                            u1_center,
                            &stop_flag,
                        );
                        if nudged_comp {
                            if let Ok(mut obs_g) = shared_obs.lock() {
                                obs_g.clone_from(&grid.obstacle_net_ids);
                            }
                            if let Ok(mut df_g) = shared_df.lock() {
                                df_g.clone_from(&grid.dist_field);
                            }
                            continue;
                        }
                    }

                    // 2. Try shoving or ripping up blocking unlocked tracks
                    let nudged_track = self.try_nudge_or_ripup_blocking_tracks(
                        board,
                        grid,
                        &connections,
                        &mut routed_paths,
                        &escape_stubs,
                        conn_idx,
                        u1_center,
                        &stop_flag,
                    );
                    if nudged_track {
                        if let Ok(mut obs_g) = shared_obs.lock() {
                            obs_g.clone_from(&grid.obstacle_net_ids);
                        }
                        if let Ok(mut df_g) = shared_df.lock() {
                            df_g.clone_from(&grid.dist_field);
                        }
                    }
                }

                let done = routed_paths.iter().filter(|p| p.is_some()).count();
                if done > best_routed_count {
                    best_routed_count = done;
                    best_routed_paths = routed_paths.clone();
                }
                println!(
                    "  [Router] Pass {}/{}: Routed {}/{} connections ({:.1}%) [{:.1}s elapsed]",
                    iteration + 1,
                    self.max_ripup_iterations,
                    done,
                    total_conns,
                    (done as f64 / total_conns.max(1) as f64) * 100.0,
                    route_start_time.elapsed().as_secs_f64()
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
                if done == total_conns {
                    break;
                }

                // Rip-up and Negotiated Congestion Update (budgeted to avoid wiping all progress)
                if iteration + 1 < self.max_ripup_iterations {
                    let mut rip_budget = 12;
                    for conn_idx in 0..total_conns {
                        if rip_budget == 0 {
                            break;
                        }
                        if routed_paths[conn_idx].is_none() {
                            let blocked_conn = &connections[conn_idx];
                            let b_start = blocked_conn.start;
                            let b_end = blocked_conn.end;

                            let mut ripped = false;
                            for rip_idx in 0..total_conns {
                                if rip_budget == 0 {
                                    break;
                                }
                                if rip_idx == conn_idx || routed_paths[rip_idx].is_none() {
                                    continue;
                                }
                                let p = routed_paths[rip_idx].as_ref().unwrap();
                                let is_at_pin = p.iter().any(|node| {
                                    let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                                    (wx - b_start.0).hypot(wy - b_start.1) < 1.0
                                        || (wx - b_end.0).hypot(wy - b_end.1) < 1.0
                                });
                                let airwire_len = (b_start.0 - b_end.0).hypot(b_start.1 - b_end.1);
                                let passes_near = is_at_pin
                                    || (airwire_len < 3.0 && p.iter().any(|node| {
                                        let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                                        dist_to_segment(wx, wy, b_start.0, b_start.1, b_end.0, b_end.1) < 0.6
                                    }));

                                let is_diff = connections[rip_idx].diff_pair_leader.is_some()
                                    || connections.iter().any(|c| c.diff_pair_leader == Some(rip_idx))
                                    || connections[rip_idx].net_name.contains("USB");
                                if is_diff {
                                    continue;
                                }

                                // Always allow ripping up tracks that directly choke a pin endpoint;
                                // for mid-span proximity, respect priority on pass 0
                                if !is_at_pin && iteration == 0 && Self::netclass_priority(&connections[rip_idx].netclass.name) > 2 {
                                    continue;
                                }

                                if passes_near {
                                    if let Some(p) = routed_paths[rip_idx].take() {
                                        for node in &p {
                                            let idx = grid.idx(node.gx, node.gy, node.layer);
                                            grid.congestion[idx] += self.congestion_factor * 4.0;
                                        }
                                        ripped = true;
                                        rip_budget -= 1;
                                    }
                                }
                            }
                            if ripped {
                                self.rebuild_grid_state(board, grid, &connections, &routed_paths, &escape_stubs);
                                if let Ok(mut obs_g) = shared_obs.lock() {
                                    obs_g.clone_from(&grid.obstacle_net_ids);
                                }
                            }
                        }
                    }
                }
            }

            gpu_done.store(true, AtomicOrdering::Relaxed);
        });

        if best_routed_count > routed_paths.iter().filter(|p| p.is_some()).count() {
            routed_paths = best_routed_paths;
            self.rebuild_grid_state(board, grid, &connections, &routed_paths, &escape_stubs);
        }

        for (conn_idx, path) in routed_paths.iter().enumerate() {
            if path.is_none() {
                let c = &connections[conn_idx];
                let mut start_blocking_nets = std::collections::BTreeSet::new();
                let mut end_blocking_nets = std::collections::BTreeSet::new();
                let mut start_free = false;
                let mut end_free = false;

                if let Some((sgx, sgy)) = grid.world_to_grid(c.start.0, c.start.1) {
                    for dy in -2isize..=2 {
                        for dx in -2isize..=2 {
                            if dx == 0 && dy == 0 { continue; }
                            let nx = sgx as isize + dx;
                            let ny = sgy as isize + dy;
                            if nx >= 0 && nx < grid.width as isize && ny >= 0 && ny < grid.height as isize {
                                let idx = grid.idx(nx as usize, ny as usize, c.start.2);
                                let obs = grid.obstacle_net_ids[idx];
                                if obs == FREE_NET || obs == c.net_id as i32 {
                                    start_free = true;
                                } else if obs > 0 {
                                    let b_net = board.nets.get(obs as usize).cloned().unwrap_or_else(|| format!("Net-{}", obs));
                                    start_blocking_nets.insert(b_net);
                                }
                            }
                        }
                    }
                }

                if let Some((egx, egy)) = grid.world_to_grid(c.end.0, c.end.1) {
                    for dy in -2isize..=2 {
                        for dx in -2isize..=2 {
                            if dx == 0 && dy == 0 { continue; }
                            let nx = egx as isize + dx;
                            let ny = egy as isize + dy;
                            if nx >= 0 && nx < grid.width as isize && ny >= 0 && ny < grid.height as isize {
                                let idx = grid.idx(nx as usize, ny as usize, c.end.2);
                                let obs = grid.obstacle_net_ids[idx];
                                if obs == FREE_NET || obs == c.net_id as i32 {
                                    end_free = true;
                                } else if obs > 0 {
                                    let b_net = board.nets.get(obs as usize).cloned().unwrap_or_else(|| format!("Net-{}", obs));
                                    end_blocking_nets.insert(b_net);
                                }
                            }
                        }
                    }
                }

                let reason = if !start_free && !end_free {
                    format!("Both pads choked (start blocked by {:?}, end blocked by {:?})", start_blocking_nets, end_blocking_nets)
                } else if !start_free {
                    format!("Start pad choked by {:?}", start_blocking_nets)
                } else if !end_free {
                    format!("End pad choked by {:?}", end_blocking_nets)
                } else {
                    "Routing channel congested / path search exhausted".to_string()
                };

                let diag = format!(
                    "[UNROUTED] Net '{}' (ID {}) from ({:.2}, {:.2}, L{}) to ({:.2}, {:.2}, L{}) — {}",
                    c.net_name, c.net_id, c.start.0, c.start.1, c.start.2, c.end.0, c.end.1, c.end.2, reason
                );
                eprintln!("  {}", diag);
                if let Some(ref st) = live {
                    if let Ok(mut g) = st.lock() {
                        g.push_log(format!("[Unrouted] {}", diag));
                    }
                }
            }
        }
        let completed_count = routed_paths.iter().filter(|p| p.is_some()).count();
        self.reconstruct_tracks_and_vias(board, grid, &connections, &routed_paths, &escape_stubs);

        // Phase 3: Via Reduction Pass (eliminates redundant layer-hopping via pairs)
        self.reduce_vias(board, grid);
        prune_dangling_tracks_and_vias(board);

        if let Some(ref ipc_lock) = self.ipc {
            if let Ok(mut client) = ipc_lock.lock() {
                if client.is_connected() {
                    let _ = client.clear_tracks();
                    let _ = client.stream_tracks_and_vias(board);
                }
            }
        }

        // Phase 3: Post-Route Glossing Pass (rubber-band straightening, stair-step elimination, 45-deg mitering)
        self.gloss_tracks(board, grid);
        prune_dangling_tracks_and_vias(board);

        // Phase 2: Coupled Diff Pair Precision Alignment and Intra-Pair Phase Matching
        for follower_idx in 0..total_conns {
            if let Some(leader_idx) = connections[follower_idx].diff_pair_leader {
                if routed_paths[leader_idx].is_some() && routed_paths[follower_idx].is_some() {
                    let lc = &connections[leader_idx];
                    let fc = &connections[follower_idx];
                    let diff_w = lc.netclass.diff_pair_width;
                    let diff_gap = lc.netclass.diff_pair_gap;
                    enforce_coupled_diff_pair_spacing(
                        board,
                        lc.net_id,
                        fc.net_id,
                        diff_w,
                        diff_gap,
                        lc.start,
                        lc.end,
                        fc.start,
                        fc.end,
                    );
                    match_diff_pair_phase(board, grid, lc.net_id, fc.net_id, 0.05);
                }
            }
        }

        self.connect_u1_ground_pins(board);
        self.add_ground_stitching_vias(board, grid);
        self.add_return_path_ground_vias(board, grid);

        // Phase 5: Automated DFM Teardrop Generation (pad & via junction strengthening)
        add_teardrops(board);

        for trk in &mut board.tracks {
            let key = (trk.net_id, [trk.start.0.to_bits(), trk.start.1.to_bits(), trk.end.0.to_bits(), trk.end.1.to_bits()]);
            trk.is_locked = orig_locked_tracks.contains(&key);
        }
        for v in &mut board.vias {
            let key = (v.net_id, [v.position.0.to_bits(), v.position.1.to_bits()]);
            v.is_locked = orig_locked_vias.contains(&key);
        }

        // Comprehensive Pre-Export DRC Verification & Cleaning
        validate_and_clean_copper(board);
        prune_dangling_tracks_and_vias(board);
        self.validate_crosstalk_and_coupling(board);

        if let Some(ref st) = live {
            let remaining_ratsnest: Vec<crate::board::Segment> = connections
                .iter()
                .enumerate()
                .filter(|&(i, _)| routed_paths[i].is_none())
                .map(|(_, c)| {
                    let p1 = board.footprints[c.start_fp].pad_absolute_position(&board.footprints[c.start_fp].pads[c.start_pad]);
                    let p2 = board.footprints[c.end_fp].pad_absolute_position(&board.footprints[c.end_fp].pads[c.end_pad]);
                    crate::board::Segment {
                        p1: Point::new(p1.0, p1.1),
                        p2: Point::new(p2.0, p2.1),
                        net_id: c.net_id,
                    }
                })
                .collect();
            if let Ok(mut g) = st.lock() {
                g.board.tracks = board.tracks.clone();
                g.board.vias = board.vias.clone();
                g.board.footprints = board.footprints.clone();
                g.ratsnest = remaining_ratsnest;
                g.routed_conns = completed_count;
                g.total_conns = total_conns;
                g.progress = 1.0;
                g.phase = TrialPhase::Completed;
                g.push_log(format!(
                    "[Auto-Router] Completed: {}/{} ({:.1}%) connections, {} tracks, {} vias.",
                    completed_count,
                    total_conns,
                    (completed_count as f64 / total_conns.max(1) as f64) * 100.0,
                    board.tracks.len(),
                    board.vias.len()
                ));
            }
        }

        if let Some(ref ipc_lock) = self.ipc {
            if let Ok(mut client) = ipc_lock.lock() {
                if client.is_connected() {
                    let _ = client.clear_tracks();
                    let _ = client.stream_tracks_and_vias(board);
                }
            }
        }

        (completed_count, total_conns)
    }

    fn is_path_conflict_free(
        &self,
        grid: &RoutingGrid,
        path: &[GridNode],
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
        require_return_path: bool,
    ) -> bool {
        if path.is_empty() {
            return false;
        }
        let s_idx = grid.idx(path[0].gx, path[0].gy, path[0].layer);
        let s_net = grid.obstacle_net_ids[s_idx];
        if s_net != FREE_NET && s_net != net_id as i32 {
            return false;
        }
        for i in 0..path.len().saturating_sub(1) {
            let u = path[i];
            let v = path[i + 1];
            let ok = grid
                .get_neighbors(u, net_id, req_clearance, track_width, via_dia, u1_center)
                .iter()
                .any(|&(nbr, _)| nbr == v);
            if !ok {
                return false;
            }
        }
        if require_return_path {
            for node in path {
                if let Some(ref_layer) = grid.adjacent_reference_plane(node.layer) {
                    if grid.is_reference_plane_void_at(node.gx, node.gy, ref_layer) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn find_path_parallel(
        &self,
        grid: &RoutingGrid,
        start: GridNode,
        goal: GridNode,
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
        max_coupled_length_mm: f64,
        require_return_path: bool,
        stop_flag: &AtomicBool,
    ) -> Option<Vec<GridNode>> {
        let strategies: &[(bool, f64, f64)] = &[
            (false, 1.0, 0.0),
            (true, 1.0, 0.0),
            (false, 1.05, 0.0),
            (true, 1.05, 0.0),
            (false, 1.15, 0.01),
            (true, 1.15, 0.01),
        ];
        let found = AtomicBool::new(false);

        let candidates: Vec<(usize, Vec<GridNode>)> = strategies
            .par_iter()
            .enumerate()
            .filter_map(|(s_idx, &(reverse, h_weight, dist_bias))| {
                let (s_node, g_node) = if reverse { (goal, start) } else { (start, goal) };
                let mut path = self.find_path_with_constraints(
                    grid,
                    s_node,
                    g_node,
                    net_id,
                    req_clearance,
                    track_width,
                    via_dia,
                    u1_center,
                    h_weight,
                    dist_bias,
                    max_coupled_length_mm,
                    require_return_path,
                    None,
                    &found,
                )?;
                if stop_flag.load(AtomicOrdering::Relaxed) {
                    return None;
                }
                if reverse {
                    path.reverse();
                    if !self.is_path_conflict_free(
                        grid,
                        &path,
                        net_id,
                        req_clearance,
                        track_width,
                        via_dia,
                        u1_center,
                        require_return_path,
                    ) {
                        return None;
                    }
                }
                found.store(true, AtomicOrdering::Relaxed);
                Some((s_idx, path))
            })
            .collect();

        candidates
            .into_iter()
            .min_by_key(|(s_idx, p)| (Self::count_path_bends(p) * 35 + p.len(), *s_idx))
            .map(|(_, p)| p)
    }

    pub fn count_path_bends(path: &[GridNode]) -> usize {
        if path.len() < 3 {
            return 0;
        }
        let mut bends = 0;
        for i in 1..(path.len() - 1) {
            if path[i - 1].layer != path[i].layer || path[i].layer != path[i + 1].layer {
                bends += 2;
                continue;
            }
            let dx1 = path[i].gx as isize - path[i - 1].gx as isize;
            let dy1 = path[i].gy as isize - path[i - 1].gy as isize;
            let dx2 = path[i + 1].gx as isize - path[i].gx as isize;
            let dy2 = path[i + 1].gy as isize - path[i].gy as isize;
            if (dx1, dy1) != (dx2, dy2) {
                bends += 1;
            }
        }
        bends
    }

    fn line_nodes(start: GridNode, end: GridNode) -> Vec<GridNode> {
        let dx = end.gx as isize - start.gx as isize;
        let dy = end.gy as isize - start.gy as isize;
        let steps = dx.abs().max(dy.abs());
        if steps == 0 {
            return vec![start];
        }
        let step_x = dx.signum();
        let step_y = dy.signum();
        let mut nodes = Vec::with_capacity(steps as usize + 1);
        for s in 0..=steps {
            nodes.push(GridNode {
                gx: (start.gx as isize + s * step_x) as usize,
                gy: (start.gy as isize + s * step_y) as usize,
                layer: start.layer,
            });
        }
        nodes
    }

    fn two_segment_candidates(start: GridNode, end: GridNode) -> Vec<Vec<GridNode>> {
        let dx = end.gx as isize - start.gx as isize;
        let dy = end.gy as isize - start.gy as isize;
        if dx == 0 || dy == 0 || dx.abs() == dy.abs() {
            return vec![Self::line_nodes(start, end)];
        }
        let s = dx.abs().min(dy.abs());
        let mut results = Vec::with_capacity(2);

        // Candidate 1: diagonal first, then straight
        let m1 = GridNode {
            gx: (start.gx as isize + s * dx.signum()) as usize,
            gy: (start.gy as isize + s * dy.signum()) as usize,
            layer: start.layer,
        };
        let mut c1 = Self::line_nodes(start, m1);
        let mut rest1 = Self::line_nodes(m1, end);
        if rest1.len() > 1 {
            c1.extend(rest1.drain(1..));
        }
        results.push(c1);

        // Candidate 2: straight first, then diagonal
        let m2 = if dx.abs() > dy.abs() {
            GridNode {
                gx: (start.gx as isize + (dx - s * dx.signum())) as usize,
                gy: start.gy,
                layer: start.layer,
            }
        } else {
            GridNode {
                gx: start.gx,
                gy: (start.gy as isize + (dy - s * dy.signum())) as usize,
                layer: start.layer,
            }
        };
        let mut c2 = Self::line_nodes(start, m2);
        let mut rest2 = Self::line_nodes(m2, end);
        if rest2.len() > 1 {
            c2.extend(rest2.drain(1..));
        }
        results.push(c2);

        results
    }

    fn smooth_grid_path(
        &self,
        grid: &RoutingGrid,
        mut path: Vec<GridNode>,
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
        require_return_path: bool,
        max_coupled_length_mm: f64,
    ) -> Vec<GridNode> {
        if path.len() < 4 {
            return path;
        }

        let is_valid_candidate = |cand: &[GridNode]| -> bool {
            if !self.is_path_conflict_free(
                grid,
                cand,
                net_id,
                req_clearance,
                track_width,
                via_dia,
                u1_center,
                require_return_path,
            ) {
                return false;
            }
            if max_coupled_length_mm > 0.0 {
                let mut run_len = 0.0;
                for k in 0..cand.len().saturating_sub(1) {
                    if grid.is_step_coupled(cand[k], cand[k + 1], net_id, track_width, req_clearance) {
                        let sl = if cand[k].gx != cand[k + 1].gx && cand[k].gy != cand[k + 1].gy {
                            1.41421356 * grid.res
                        } else {
                            1.0 * grid.res
                        };
                        run_len += sl;
                        if run_len > max_coupled_length_mm {
                            return false;
                        }
                    } else {
                        run_len = 0.0;
                    }
                }
            }
            true
        };

        // Pass 1: Line-of-sight octilinear shortcutting and staircase collapsing
        // Eliminates staircase zig-zags and rapid direction changes by connecting nodes with
        // 0 bends (straight line) or 1 bend (canonical 45-degree corner).
        let max_passes = 4;
        for _pass in 0..max_passes {
            let mut improved = false;
            let mut new_path: Vec<GridNode> = Vec::with_capacity(path.len());
            let mut i = 0;
            new_path.push(path[0]);

            while i < path.len() {
                if i + 1 >= path.len() {
                    break;
                }
                // Determine the end of the current same-layer run
                let current_layer = path[i].layer;
                let mut layer_run_end = i + 1;
                while layer_run_end < path.len() && path[layer_run_end].layer == current_layer {
                    layer_run_end += 1;
                }
                if layer_run_end <= i + 1 {
                    i += 1;
                    if i < path.len() {
                        new_path.push(path[i]);
                    }
                    continue;
                }

                let max_lookahead = (layer_run_end - 1).min(i + 64);
                let mut shortcut_found = false;

                // Priority 1: Direct 1-segment shortcut (0 bends)
                for j in (i + 2..=max_lookahead).rev() {
                    let p_i = path[i];
                    let p_j = path[j];
                    let dx = p_j.gx as isize - p_i.gx as isize;
                    let dy = p_j.gy as isize - p_i.gy as isize;
                    if dx == 0 || dy == 0 || dx.abs() == dy.abs() {
                        let candidate = Self::line_nodes(p_i, p_j);
                        if is_valid_candidate(&candidate) {
                            let i_prev = i.saturating_sub(1);
                            let j_next = (j + 1).min(path.len() - 1);
                            let bends_before = Self::count_path_bends(&path[i_prev..=j_next]);

                            let mut cand_slice = Vec::with_capacity(candidate.len() + 2);
                            if i > 0 && path[i - 1].layer == current_layer {
                                cand_slice.push(path[i - 1]);
                            }
                            cand_slice.extend_from_slice(&candidate);
                            if j + 1 < path.len() && path[j + 1].layer == current_layer {
                                cand_slice.push(path[j + 1]);
                            }
                            let bends_after = Self::count_path_bends(&cand_slice);

                            if bends_after < bends_before
                                || (bends_after == bends_before && candidate.len() <= (j - i + 1))
                            {
                                for node in candidate.into_iter().skip(1) {
                                    new_path.push(node);
                                }
                                i = j;
                                shortcut_found = true;
                                improved = true;
                                break;
                            }
                        }
                    }
                }

                if shortcut_found {
                    continue;
                }

                // Priority 2: 2-segment canonical shortcut (1 bend, replaces multi-bend staircases)
                for j in (i + 3..=max_lookahead).rev() {
                    let p_i = path[i];
                    let p_j = path[j];
                    let existing_bends = Self::count_path_bends(&path[i..=j]);
                    if existing_bends < 2 {
                        continue;
                    }
                    let candidates = Self::two_segment_candidates(p_i, p_j);
                    let mut best_candidate: Option<Vec<GridNode>> = None;
                    for cand in candidates {
                        if is_valid_candidate(&cand) {
                            let i_prev = i.saturating_sub(1);
                            let j_next = (j + 1).min(path.len() - 1);
                            let bends_before = Self::count_path_bends(&path[i_prev..=j_next]);

                            let mut cand_slice = Vec::with_capacity(cand.len() + 2);
                            if i > 0 && path[i - 1].layer == current_layer {
                                cand_slice.push(path[i - 1]);
                            }
                            cand_slice.extend_from_slice(&cand);
                            if j + 1 < path.len() && path[j + 1].layer == current_layer {
                                cand_slice.push(path[j + 1]);
                            }
                            let bends_after = Self::count_path_bends(&cand_slice);

                            if bends_after < bends_before {
                                best_candidate = Some(cand);
                                break;
                            }
                        }
                    }
                    if let Some(cand) = best_candidate {
                        for node in cand.into_iter().skip(1) {
                            new_path.push(node);
                        }
                        i = j;
                        shortcut_found = true;
                        improved = true;
                        break;
                    }
                }

                if !shortcut_found {
                    i += 1;
                    if i < path.len() {
                        new_path.push(path[i]);
                    }
                }
            }

            path = new_path;
            if !improved {
                break;
            }
        }

        path
    }

    pub fn find_path_with_params(
        &self,
        grid: &RoutingGrid,
        start: GridNode,
        goal: GridNode,
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
        h_weight: f64,
        dist_bias: f64,
        abort_flag: &AtomicBool,
    ) -> Option<Vec<GridNode>> {
        self.find_path_with_constraints(
            grid,
            start,
            goal,
            net_id,
            req_clearance,
            track_width,
            via_dia,
            u1_center,
            h_weight,
            dist_bias,
            15.0,
            false,
            None,
            abort_flag,
        )
    }

    #[allow(dead_code)]
    fn find_path_with_corridor(
        &self,
        grid: &RoutingGrid,
        start: GridNode,
        goal: GridNode,
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
        h_weight: f64,
        dist_bias: f64,
        corridor: Option<&[f32]>,
        abort_flag: &AtomicBool,
    ) -> Option<Vec<GridNode>> {
        self.find_path_with_constraints(
            grid,
            start,
            goal,
            net_id,
            req_clearance,
            track_width,
            via_dia,
            u1_center,
            h_weight,
            dist_bias,
            15.0,
            false,
            corridor,
            abort_flag,
        )
    }

    pub fn find_path_with_constraints(
        &self,
        grid: &RoutingGrid,
        start: GridNode,
        goal: GridNode,
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
        h_weight: f64,
        dist_bias: f64,
        max_coupled_length_mm: f64,
        require_return_path: bool,
        corridor: Option<&[f32]>,
        abort_flag: &AtomicBool,
    ) -> Option<Vec<GridNode>> {
        let mut open_set = BinaryHeap::new();
        let mut came_from: HashMap<GridNode, GridNode> = HashMap::new();
        let mut g_score: HashMap<GridNode, f64> = HashMap::new();
        let mut coupled_length: HashMap<GridNode, f64> = HashMap::new();

        let h_scale = 1.0;
        let h_start = self.heuristic(grid, start, goal) * h_weight * h_scale;
        g_score.insert(start, 0.0);
        coupled_length.insert(start, 0.0);
        open_set.push(AStarNode {
            node: start,
            cost: 0.0,
            priority: h_start,
        });

        // If start is at a through-hole pad of this net, seed open_set on all valid signal layers
        for l in 0..grid.num_layers {
            if l != start.layer {
                if grid.plane_nets.contains_key(&l) && grid.plane_nets.get(&l) != Some(&net_id) {
                    continue;
                }
                let idx = grid.idx(start.gx, start.gy, l);
                if grid.obstacle_net_ids[idx] == net_id as i32 {
                    let alt_start = GridNode { gx: start.gx, gy: start.gy, layer: l };
                    g_score.insert(alt_start, 0.0);
                    coupled_length.insert(alt_start, 0.0);
                    open_set.push(AStarNode {
                        node: alt_start,
                        cost: 0.0,
                        priority: self.heuristic(grid, alt_start, goal) * h_weight * h_scale,
                    });
                }
            }
        }

        let dist_span = (goal.gx as isize - start.gx as isize).abs() + (goal.gy as isize - start.gy as isize).abs();
        let max_expansions = if dist_span > 200 { 350_000 } else { 120_000 };
        let mut expansions = 0;

        while let Some(current) = open_set.pop() {
            expansions += 1;
            if expansions > max_expansions
                || (expansions % 256 == 0 && abort_flag.load(AtomicOrdering::Relaxed))
            {
                return None;
            }
            let u = current.node;
            if u == goal {
                let mut path = vec![goal];
                let mut curr = goal;
                while let Some(&prev) = came_from.get(&curr) {
                    path.push(prev);
                    curr = prev;
                }
                path.reverse();
                return Some(self.smooth_grid_path(
                    grid,
                    path,
                    net_id,
                    req_clearance,
                    track_width,
                    via_dia,
                    u1_center,
                    require_return_path,
                    max_coupled_length_mm,
                ));
            }

            let Some(&current_g) = g_score.get(&u) else {
                continue;
            };
            if current.cost > current_g + 1e-6 {
                continue;
            }

            let prev_dir: Option<(isize, isize)> = came_from.get(&u).and_then(|prev| {
                if prev.layer == u.layer {
                    Some((u.gx as isize - prev.gx as isize, u.gy as isize - prev.gy as isize))
                } else {
                    None
                }
            });

            for (neighbor, mut edge_cost) in
                grid.get_neighbors(u, net_id, req_clearance, track_width, via_dia, u1_center)
            {
                let mut next_coupled = 0.0;
                if neighbor.layer == u.layer {
                    if let Some((pdx, pdy)) = prev_dir {
                        let cur_dx = neighbor.gx as isize - u.gx as isize;
                        let cur_dy = neighbor.gy as isize - u.gy as isize;
                        if (cur_dx, cur_dy) != (pdx, pdy) {
                            let dot = pdx * cur_dx + pdy * cur_dy;
                            if dot > 0 {
                                // 45-degree bend penalty to prevent rapid zig-zag meandering
                                edge_cost += 5.0 * grid.res;
                            } else if dot == 0 {
                                // 90-degree bend penalty
                                edge_cost += 15.0 * grid.res;
                            } else {
                                // Sharp/obtuse (>90-degree) bend penalty
                                edge_cost += 45.0 * grid.res;
                            }
                        }
                    }

                    // Return-path continuity check for high-speed signals
                    if require_return_path {
                        if let Some(ref_layer) = grid.adjacent_reference_plane(neighbor.layer) {
                            if grid.is_reference_plane_void_at(neighbor.gx, neighbor.gy, ref_layer) {
                                edge_cost += 35.0 * grid.res + 5.0;
                            }
                        }
                    }

                    // Crosstalk & coupled length limiter
                    let is_coupled = grid.is_step_coupled(u, neighbor, net_id, track_width, req_clearance);
                    let step_len = if neighbor.gx != u.gx && neighbor.gy != u.gy {
                        1.41421356 * grid.res
                    } else {
                        1.0 * grid.res
                    };

                    if is_coupled {
                        let prev_coupled = coupled_length.get(&u).copied().unwrap_or(0.0);
                        next_coupled = prev_coupled + step_len;
                        if next_coupled > max_coupled_length_mm {
                            let excess = next_coupled - max_coupled_length_mm;
                            let penalty = self.crosstalk_penalty * step_len * (1.0 + excess * 0.5);
                            edge_cost += penalty;
                        }
                    }
                } else {
                    // Through-hole via penalty: equivalent to 2.0 mm of trace length
                    edge_cost += 2.0;
                }

                let n_idx = grid.idx(neighbor.gx, neighbor.gy, neighbor.layer);
                if dist_bias > 0.0 {
                    let df_val = (grid.dist_field[n_idx] as f64).clamp(0.05, 2.0);
                    edge_cost += dist_bias / df_val;
                }
                if let Some(corr) = corridor {
                    edge_cost *= corr[n_idx] as f64;
                }
                let tentative_g = current_g + edge_cost;
                let is_better = g_score.get(&neighbor).is_none_or(|&prev_g| tentative_g < prev_g - 1e-6);
                if is_better {
                    g_score.insert(neighbor, tentative_g);
                    coupled_length.insert(neighbor, next_coupled);
                    came_from.insert(neighbor, u);
                    let priority = tentative_g + self.heuristic(grid, neighbor, goal) * h_weight * h_scale;
                    open_set.push(AStarNode {
                        node: neighbor,
                        cost: tentative_g,
                        priority,
                    });
                }
            }
        }
        None
    }

    fn heuristic(&self, grid: &RoutingGrid, a: GridNode, b: GridNode) -> f64 {
        let dx = (a.gx as isize - b.gx as isize).abs() as f64 * grid.res;
        let dy = (a.gy as isize - b.gy as isize).abs() as f64 * grid.res;
        let diag = dx.min(dy);
        let straight = (dx - dy).abs();
        let layer_pen = if a.layer != b.layer { 2.0 } else { 0.0 };
        diag * std::f64::consts::SQRT_2 + straight + layer_pen
    }

    pub fn commit_path_to_grid(
        &self,
        grid: &mut RoutingGrid,
        path: &[GridNode],
        net_id: usize,
        track_width: f64,
        via_dia: f64,
    ) {
        let r_cells = (track_width / (2.0 * grid.res)).ceil() as isize;
        for i in 0..path.len() {
            let node = path[i];
            for dy in -r_cells..=r_cells {
                for dx in -r_cells..=r_cells {
                    if (dx * dx + dy * dy) as f64 * (grid.res * grid.res) <= (track_width / 2.0).powi(2) + 1e-6 {
                        let nx = node.gx as isize + dx;
                        let ny = node.gy as isize + dy;
                        if nx >= 0 && nx < grid.width as isize && ny >= 0 && ny < grid.height as isize {
                            let idx = grid.idx(nx as usize, ny as usize, node.layer);
                            if grid.obstacle_net_ids[idx] == FREE_NET {
                                grid.obstacle_net_ids[idx] = net_id as i32;
                            }
                        }
                    }
                }
            }

            if i + 1 < path.len() && path[i + 1].layer != node.layer {
                let vr_radius = (via_dia / 2.0) + (grid.res * 0.40);
                let vr_cells = (vr_radius / grid.res).ceil() as isize;
                for cy in -vr_cells..=vr_cells {
                    for cx in -vr_cells..=vr_cells {
                        if (cx * cx + cy * cy) as f64 * (grid.res * grid.res) <= vr_radius.powi(2) + 1e-6 {
                            let nx = node.gx as isize + cx;
                            let ny = node.gy as isize + cy;
                            if nx >= 0 && nx < grid.width as isize && ny >= 0 && ny < grid.height as isize {
                                for l in 0..grid.num_layers {
                                    let idx = grid.idx(nx as usize, ny as usize, l);
                                    if grid.obstacle_net_ids[idx] == FREE_NET {
                                        grid.obstacle_net_ids[idx] = net_id as i32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn rebuild_grid_state(
        &self,
        board: &Board,
        grid: &mut RoutingGrid,
        connections: &[RouteConnection],
        routed_paths: &[Option<Vec<GridNode>>],
        escape_stubs: &[EscapeStub],
    ) {
        grid.obstacle_net_ids.fill(FREE_NET);
        grid.nudgeable_fp.fill(None);
        grid.smd_pad_cells.fill(false);
        grid.rasterize_obstacles(board);
        for stub in escape_stubs {
            self.commit_stub_to_grid(grid, stub);
        }
        for (c_idx, path_opt) in routed_paths.iter().enumerate() {
            if let Some(path) = path_opt {
                self.commit_path_to_grid(
                    grid,
                    path,
                    connections[c_idx].net_id,
                    connections[c_idx].netclass.track_width,
                    connections[c_idx].netclass.via_dia,
                );
            }
        }
    }

    /// Pre-routing pass: nudges unlocked components away if they directly block escape stubs.
    pub fn nudge_components_blocking_escapes(&self, board: &mut Board) -> bool {
        if !self.allow_component_nudge {
            return false;
        }
        let provisional_stubs = self.generate_escape_stubs(board);
        let mut any_nudged = false;
        for stub in &provisional_stubs {
            let s_p1 = Point::new(stub.pad_pos.0, stub.pad_pos.1);
            let s_p2 = Point::new(stub.tip_pos.0, stub.tip_pos.1);
            for fp_idx in 0..board.footprints.len() {
                if fp_idx == stub.fp_idx {
                    continue;
                }
                let fp = &board.footprints[fp_idx];
                if fp.is_locked || fp.pads.len() > 2 {
                    continue;
                }
                let mut is_blocking = false;
                for pad in &fp.pads {
                    if pad.net_id == stub.net_id {
                        continue;
                    }
                    let (px, py) = fp.pad_absolute_position(pad);
                    let d = dist_to_segment(px, py, s_p1.x, s_p1.y, s_p2.x, s_p2.y);
                    if d < 0.20 + stub.width / 2.0 + 0.30 {
                        is_blocking = true;
                        break;
                    }
                }
                if is_blocking {
                    let push_dir = (board.footprints[fp_idx].x - stub.pad_pos.0, board.footprints[fp_idx].y - stub.pad_pos.1);
                    if self.nudge_engine.try_nudge_component(board, fp_idx, push_dir) {
                        any_nudged = true;
                    }
                }
            }
        }
        any_nudged
    }

    /// When routing a connection fails, identify blocking unlocked components,
    /// nudge them away into free space, and retry the path search.
    pub fn try_nudge_blocking_components(
        &self,
        board: &mut Board,
        grid: &mut RoutingGrid,
        connections: &mut [RouteConnection],
        routed_paths: &mut [Option<Vec<GridNode>>],
        escape_stubs: &[EscapeStub],
        conn_idx: usize,
        u1_center: Option<(f64, f64)>,
        stop_flag: &AtomicBool,
    ) -> bool {
        if !self.allow_component_nudge {
            return false;
        }
        let conn = &connections[conn_idx];
        let c_start = conn.start;
        let c_end = conn.end;
        let c_net_id = conn.net_id;
        let req_clearance = conn.netclass.clearance;
        let track_width = conn.netclass.track_width;
        let via_dia = conn.netclass.via_dia;

        let mut blocking_candidates = Vec::new();
        for (fp_idx, fp) in board.footprints.iter().enumerate() {
            if fp.is_locked || fp.pads.len() > 2 || fp_idx == conn.start_fp || fp_idx == conn.end_fp {
                continue;
            }
            let d_start = (fp.x - c_start.0).hypot(fp.y - c_start.1);
            let d_end = (fp.x - c_end.0).hypot(fp.y - c_end.1);
            let d_seg = dist_to_segment(fp.x, fp.y, c_start.0, c_start.1, c_end.0, c_end.1);

            let has_close_pad = fp.pads.iter().any(|pad| {
                if pad.net_id == c_net_id {
                    return false;
                }
                let (px, py) = fp.pad_absolute_position(pad);
                (px - c_start.0).hypot(py - c_start.1) < 2.5
                    || (px - c_end.0).hypot(py - c_end.1) < 2.5
                    || dist_to_segment(px, py, c_start.0, c_start.1, c_end.0, c_end.1) < 1.5
            });

            if d_start < 3.0 || d_end < 3.0 || d_seg < 2.5 || has_close_pad {
                let push_dir = if d_start < 2.5 {
                    (fp.x - c_start.0, fp.y - c_start.1)
                } else if d_end < 2.5 {
                    (fp.x - c_end.0, fp.y - c_end.1)
                } else {
                    let seg_dx = c_end.0 - c_start.0;
                    let seg_dy = c_end.1 - c_start.1;
                    let perp = (-seg_dy, seg_dx);
                    let dot = perp.0 * (fp.x - c_start.0) + perp.1 * (fp.y - c_start.1);
                    if dot >= 0.0 { perp } else { (-perp.0, -perp.1) }
                };
                blocking_candidates.push((fp_idx, push_dir, d_seg.min(d_start).min(d_end)));
            }
        }

        blocking_candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
        blocking_candidates.truncate(2);

        for (fp_idx, push_dir, _) in blocking_candidates {
            let orig_fp_x = board.footprints[fp_idx].x;
            let orig_fp_y = board.footprints[fp_idx].y;
            let mut resolved = false;

            for (dx, dy) in self.nudge_engine.nudge_candidates(push_dir) {
                let nx = orig_fp_x + dx;
                let ny = orig_fp_y + dy;
                if !self.nudge_engine.is_placement_valid(board, fp_idx, nx, ny) {
                    continue;
                }
                board.footprints[fp_idx].x = nx;
                board.footprints[fp_idx].y = ny;

                // Update connection endpoints for the nudged footprint
                for c in connections.iter_mut() {
                    if c.start_fp == fp_idx {
                        let (px, py) = board.footprints[fp_idx].pad_absolute_position(&board.footprints[fp_idx].pads[c.start_pad]);
                        c.start = (px, py, c.start.2);
                    }
                    if c.end_fp == fp_idx {
                        let (px, py) = board.footprints[fp_idx].pad_absolute_position(&board.footprints[fp_idx].pads[c.end_pad]);
                        c.end = (px, py, c.end.2);
                    }
                }
                // Clear any routed paths connected to the nudged footprint so they can re-route cleanly
                for (c_i, c) in connections.iter().enumerate() {
                    if c.start_fp == fp_idx || c.end_fp == fp_idx {
                        if routed_paths[c_i].is_some() {
                            routed_paths[c_i] = None;
                        }
                    }
                }

                self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);

                let updated_conn = &connections[conn_idx];
                let (Some((sgx, sgy)), Some((egx, egy))) = (
                    grid.world_to_grid(updated_conn.start.0, updated_conn.start.1),
                    grid.world_to_grid(updated_conn.end.0, updated_conn.end.1),
                ) else {
                    continue;
                };
                let s_node = GridNode { gx: sgx, gy: sgy, layer: updated_conn.start.2 };
                let e_node = GridNode { gx: egx, gy: egy, layer: updated_conn.end.2 };

                if let Some(p) = self.find_path_with_params(
                    grid,
                    s_node,
                    e_node,
                    c_net_id,
                    req_clearance,
                    track_width,
                    via_dia,
                    u1_center,
                    1.05,
                    0.0,
                    stop_flag,
                ) {
                    self.commit_path_to_grid(
                        grid,
                        &p,
                        c_net_id,
                        track_width,
                        via_dia,
                    );
                    routed_paths[conn_idx] = Some(p);
                    resolved = true;
                    break;
                }
            }

            if resolved {
                board.rip_up_connected_traces_to_component_at(
                    fp_idx,
                    Some((orig_fp_x, orig_fp_y, board.footprints[fp_idx].rotation, board.footprints[fp_idx].layer)),
                );
                grid.update_distance_fields();
                return true;
            } else {
                board.footprints[fp_idx].x = orig_fp_x;
                board.footprints[fp_idx].y = orig_fp_y;
                for c in connections.iter_mut() {
                    if c.start_fp == fp_idx {
                        let (px, py) = board.footprints[fp_idx].pad_absolute_position(&board.footprints[fp_idx].pads[c.start_pad]);
                        c.start = (px, py, c.start.2);
                    }
                    if c.end_fp == fp_idx {
                        let (px, py) = board.footprints[fp_idx].pad_absolute_position(&board.footprints[fp_idx].pads[c.end_pad]);
                        c.end = (px, py, c.end.2);
                    }
                }
                self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
            }
        }
        false
    }

/// Returns the number of reachable grid cells from (gx, gy, layer) using a local BFS,
/// along with any foreign obstacle nets bounding the pocket.
pub fn find_choked_pin_bottlenecks(
    grid: &RoutingGrid,
    gx: usize,
    gy: usize,
    layer: usize,
    net_id: usize,
    max_cells: usize,
) -> (usize, Vec<usize>) {
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    let mut blocking_nets = std::collections::BTreeSet::new();

    queue.push_back((gx, gy));
    visited.insert((gx, gy));

    while let Some((cx, cy)) = queue.pop_front() {
        if visited.len() >= max_cells {
            break;
        }
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] {
            let nx = cx as isize + dx;
            let ny = cy as isize + dy;
            if nx >= 0 && nx < grid.width as isize && ny >= 0 && ny < grid.height as isize {
                let ux = nx as usize;
                let uy = ny as usize;
                if !visited.contains(&(ux, uy)) {
                    let idx = grid.idx(ux, uy, layer);
                    let obs = grid.obstacle_net_ids[idx];
                    if obs == FREE_NET || obs == net_id as i32 {
                        visited.insert((ux, uy));
                        queue.push_back((ux, uy));
                    } else if obs > 0 {
                        blocking_nets.insert(obs as usize);
                    }
                }
            }
        }
    }

    (visited.len(), blocking_nets.into_iter().collect())
}

    /// When routing is blocked by an unlocked track of another net:
    /// Shoves the blocking track or rips it up so the higher-priority signal can claim the channel.
    pub fn try_nudge_or_ripup_blocking_tracks(
        &self,
        board: &mut Board,
        grid: &mut RoutingGrid,
        connections: &[RouteConnection],
        routed_paths: &mut [Option<Vec<GridNode>>],
        escape_stubs: &[EscapeStub],
        conn_idx: usize,
        u1_center: Option<(f64, f64)>,
        stop_flag: &AtomicBool,
    ) -> bool {
        let conn = &connections[conn_idx];
        let _my_prio = Self::netclass_priority(&conn.netclass.name);
        let c_start = conn.start;
        let c_end = conn.end;

        let total_conns = connections.len();

        // 1. Identify Candidate Blocking Connections:
        // Include:
        // (a) Choked pin bottlenecks (<= 25 reachable nodes)
        // (b) Tracks / vias in the corridor within 2.5 mm
        let (s_grid, e_grid) = (
            grid.world_to_grid(c_start.0, c_start.1),
            grid.world_to_grid(c_end.0, c_end.1),
        );
        let mut choked_blocking_nets = Vec::new();
        if let Some((sgx, sgy)) = s_grid {
            let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, sgx, sgy, c_start.2, conn.net_id, 30);
            if reach <= 25 && !nets.is_empty() {
                choked_blocking_nets.extend(nets);
            }
        }
        if let Some((egx, egy)) = e_grid {
            let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, egx, egy, c_end.2, conn.net_id, 30);
            if reach <= 25 && !nets.is_empty() {
                choked_blocking_nets.extend(nets);
            }
        }

        let mut candidate_conns: Vec<(usize, usize, f64, f64, bool)> = Vec::new(); // (idx, prio, min_pad_dist, min_dist, is_choked)

        for rip_idx in 0..total_conns {
            if rip_idx == conn_idx || routed_paths[rip_idx].is_none() {
                continue;
            }
            if connections[rip_idx].net_id == conn.net_id {
                continue;
            }
            let rip_prio = Self::netclass_priority(&connections[rip_idx].netclass.name);
            let is_diff = connections[rip_idx].diff_pair_leader.is_some()
                || connections[rip_idx].net_name.contains("USB");
            if is_diff {
                continue;
            }

            let is_choked_net = choked_blocking_nets.contains(&connections[rip_idx].net_id);

            let path = routed_paths[rip_idx].as_ref().unwrap();
            let mut min_pad_dist = f64::INFINITY;
            let mut min_dist = f64::INFINITY;
            for node in path {
                let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                let d_start = (wx - c_start.0).hypot(wy - c_start.1);
                let d_end = (wx - c_end.0).hypot(wy - c_end.1);
                let d_pad = d_start.min(d_end);
                if d_pad < min_pad_dist {
                    min_pad_dist = d_pad;
                }
                let d_seg = dist_to_segment(wx, wy, c_start.0, c_start.1, c_end.0, c_end.1);
                let d = d_pad.min(d_seg);
                if d < min_dist {
                    min_dist = d;
                }
            }

            if is_choked_net || min_dist < 2.5 {
                let is_near_pad = is_choked_net || min_pad_dist < 1.2;
                candidate_conns.push((rip_idx, rip_prio, min_pad_dist, min_dist, is_near_pad));
            }
        }

        // Prioritize: (1) choked/near-pad bottlenecks, (2) closest distance, (3) lower net priority
        candidate_conns.sort_by(|a, b| {
            b.4.cmp(&a.4)
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
                .then_with(|| a.1.cmp(&b.1))
        });
        candidate_conns.truncate(4);

        for (rip_idx, _, _, _, _) in candidate_conns {
            let orig_path = routed_paths[rip_idx].clone().unwrap();

            // ========================================================
            // STAGE 1: Via Shoving
            // If the blocking path has a via close to our corridor, attempt to slide the via into adjacent free cells.
            // ========================================================
            let other_net_id = connections[rip_idx].net_id;
            let other_clearance = connections[rip_idx].netclass.clearance;
            let other_via_dia = connections[rip_idx].netclass.via_dia;

            for v_step in 0..orig_path.len().saturating_sub(1) {
                if orig_path[v_step].layer != orig_path[v_step + 1].layer {
                    let v_node = orig_path[v_step];
                    let (vwx, vwy) = grid.grid_to_world(v_node.gx, v_node.gy);
                    let d_start = (vwx - c_start.0).hypot(vwy - c_start.1);
                    let d_end = (vwx - c_end.0).hypot(vwy - c_end.1);
                    let d_corridor = dist_to_segment(vwx, vwy, c_start.0, c_start.1, c_end.0, c_end.1);
                    if d_start.min(d_end).min(d_corridor) > 2.5 {
                        continue;
                    }

                    // Try candidate shifts: 1-3 grid steps in 8 directions
                    let shifts = [
                        (1, 0), (-1, 0), (0, 1), (0, -1),
                        (1, 1), (-1, 1), (1, -1), (-1, -1),
                        (2, 0), (-2, 0), (0, 2), (0, -2),
                        (2, 2), (-2, 2), (2, -2), (-2, -2),
                    ];

                    for (dx, dy) in shifts {
                        let ngx = v_node.gx as isize + dx;
                        let ngy = v_node.gy as isize + dy;
                        if ngx < 0 || ngx >= grid.width as isize || ngy < 0 || ngy >= grid.height as isize {
                            continue;
                        }
                        let (cand_gx, cand_gy) = (ngx as usize, ngy as usize);
                        let (nwx, nwy) = grid.grid_to_world(cand_gx, cand_gy);
                        if !is_via_location_legal(board, (nwx, nwy), other_via_dia, other_net_id, other_clearance) {
                            continue;
                        }

                        // Check grid cells across layers
                        let mut cells_ok = true;
                        for l in 0..grid.num_layers {
                            let idx = grid.idx(cand_gx, cand_gy, l);
                            let obst = grid.obstacle_net_ids[idx];
                            if obst != FREE_NET && obst != other_net_id as i32 {
                                cells_ok = false;
                                break;
                            }
                        }
                        if !cells_ok {
                            continue;
                        }

                        // Build candidate path with slid via
                        let l_in = v_node.layer;
                        let l_out = orig_path[v_step + 1].layer;
                        let cand_via_in = GridNode { gx: cand_gx, gy: cand_gy, layer: l_in };
                        let cand_via_out = GridNode { gx: cand_gx, gy: cand_gy, layer: l_out };

                        let lead_in_anchor = if v_step > 0 { orig_path[v_step - 1] } else { orig_path[0] };
                        let lead_out_anchor = if v_step + 2 < orig_path.len() { orig_path[v_step + 2] } else { orig_path[v_step + 1] };

                        let c1_list = Self::two_segment_candidates(lead_in_anchor, cand_via_in);
                        let c2_list = Self::two_segment_candidates(cand_via_out, lead_out_anchor);

                        let mut valid_slid_path = None;
                        'cand_loop: for c1 in &c1_list {
                            for c2 in &c2_list {
                                let mut cand_full = Vec::with_capacity(orig_path.len() + c1.len() + c2.len());
                                cand_full.extend_from_slice(&orig_path[0..v_step.saturating_sub(1).max(1)]);
                                cand_full.extend(c1.iter().cloned());
                                cand_full.push(cand_via_in);
                                cand_full.push(cand_via_out);
                                cand_full.extend(c2.iter().cloned());
                                if v_step + 2 < orig_path.len() {
                                    cand_full.extend_from_slice(&orig_path[(v_step + 2)..]);
                                }

                                if self.is_path_conflict_free(
                                    grid,
                                    &cand_full,
                                    other_net_id,
                                    other_clearance,
                                    connections[rip_idx].netclass.track_width,
                                    other_via_dia,
                                    u1_center,
                                    connections[rip_idx].netclass.require_return_path,
                                ) {
                                    valid_slid_path = Some(cand_full);
                                    break 'cand_loop;
                                }
                            }
                        }

                        if let Some(slid_path) = valid_slid_path {
                            routed_paths[rip_idx] = Some(slid_path);
                            self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);

                            let (Some((sgx, sgy)), Some((egx, egy))) = (
                                grid.world_to_grid(conn.start.0, conn.start.1),
                                grid.world_to_grid(conn.end.0, conn.end.1),
                            ) else {
                                continue;
                            };
                            let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                            let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };

                            if let Some(p) = self.find_path_with_params(
                                grid,
                                s_node,
                                e_node,
                                conn.net_id,
                                conn.netclass.clearance,
                                conn.netclass.track_width,
                                conn.netclass.via_dia,
                                u1_center,
                                1.05,
                                0.0,
                                stop_flag,
                            ) {
                                self.commit_path_to_grid(grid, &p, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
                                routed_paths[conn_idx] = Some(p);
                                return true; // Via shoved cleanly!
                            } else {
                                routed_paths[rip_idx] = Some(orig_path.clone());
                                self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
                            }
                        }
                    }
                }
            }

            // ========================================================
            // STAGE 2: Negotiated Shove-and-Reroute (Push & Shove / Spring Reroute)
            // Lift the blocking path, route the escaping/priority signal, then IMMEDIATELY
            // re-route the blocking net around the new signal. Both stay 100% routed!
            // ========================================================
            routed_paths[rip_idx] = None;
            self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);

            let (s_opt, e_opt) = (
                grid.world_to_grid(conn.start.0, conn.start.1),
                grid.world_to_grid(conn.end.0, conn.end.1),
            );
            if let (Some((sgx, sgy)), Some((egx, egy))) = (s_opt, e_opt) {
                let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };

                if let Some(p_conn) = self.find_path_parallel(
                    grid,
                    s_node,
                    e_node,
                    conn.net_id,
                    conn.netclass.clearance,
                    conn.netclass.track_width,
                    conn.netclass.via_dia,
                    u1_center,
                    conn.netclass.max_coupled_length_mm,
                    conn.netclass.require_return_path,
                    stop_flag,
                ) {
                    self.commit_path_to_grid(grid, &p_conn, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
                    routed_paths[conn_idx] = Some(p_conn);

                    // Now try to shove rip_idx by immediately re-routing it around conn_idx!
                    let rip_conn = &connections[rip_idx];
                    let (rs_opt, re_opt) = (
                        grid.world_to_grid(rip_conn.start.0, rip_conn.start.1),
                        grid.world_to_grid(rip_conn.end.0, rip_conn.end.1),
                    );
                    if let (Some((rsgx, rsgy)), Some((regx, regy))) = (rs_opt, re_opt) {
                        let r_s = GridNode { gx: rsgx, gy: rsgy, layer: rip_conn.start.2 };
                        let r_e = GridNode { gx: regx, gy: regy, layer: rip_conn.end.2 };

                        if let Some(p_shoved) = self.find_path_parallel(
                            grid,
                            r_s,
                            r_e,
                            rip_conn.net_id,
                            rip_conn.netclass.clearance,
                            rip_conn.netclass.track_width,
                            rip_conn.netclass.via_dia,
                            u1_center,
                            rip_conn.netclass.max_coupled_length_mm,
                            rip_conn.netclass.require_return_path,
                            stop_flag,
                        ) {
                            self.commit_path_to_grid(grid, &p_shoved, rip_conn.net_id, rip_conn.netclass.track_width, rip_conn.netclass.via_dia);
                            routed_paths[rip_idx] = Some(p_shoved);
                            return true; // BOTH nets are routed! Shove succeeded!
                        }
                    }

                    // rip_idx could not route around conn_idx immediately.
                    // Keep conn_idx routed and mark rip_idx for rip-up with congestion penalty.
                    for node in &orig_path {
                        let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                        if (wx - c_start.0).hypot(wy - c_start.1) < 2.0 || (wx - c_end.0).hypot(wy - c_end.1) < 2.0 {
                            let idx = grid.idx(node.gx, node.gy, node.layer);
                            grid.congestion[idx] += self.congestion_factor * 2.0;
                        }
                    }
                    return true;
                }
            }

            // If routing conn_idx failed even with rip_idx lifted, restore rip_idx and continue to next candidate
            routed_paths[rip_idx] = Some(orig_path);
            self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
        }

        false
    }

    fn reconstruct_tracks_and_vias(
        &self,
        board: &mut Board,
        grid: &RoutingGrid,
        connections: &[RouteConnection],
        paths: &[Option<Vec<GridNode>>],
        escape_stubs: &[EscapeStub],
    ) {
        let locked_tracks: Vec<TrackSegment> = board.tracks.iter().filter(|t| t.is_locked).cloned().collect();
        let locked_vias: Vec<Via> = board.vias.iter().filter(|v| v.is_locked).cloned().collect();
        board.tracks = locked_tracks;
        board.vias = locked_vias;

        for (conn_idx, path_opt) in paths.iter().enumerate() {
            let Some(path) = path_opt else {
                continue;
            };
            let conn = &connections[conn_idx];
            let is_diff_pair = conn.diff_pair_leader.is_some()
                || connections.iter().any(|c| c.diff_pair_leader == Some(conn_idx))
                || conn.netclass.name == "USB_Diff";
            let width = if is_diff_pair {
                conn.netclass.diff_pair_width
            } else {
                conn.netclass.track_width
            };
            let via_dia = conn.netclass.via_dia;
            let via_drill = conn.netclass.via_drill;
            if path.len() < 2 {
                if (conn.start.0 - conn.end.0).hypot(conn.start.1 - conn.end.1) > 1e-4 {
                    board.tracks.push(TrackSegment {
                        start: (conn.start.0, conn.start.1),
                        end: (conn.end.0, conn.end.1),
                        width,
                        layer: conn.start.2,
                        net_id: conn.net_id,
                        is_locked: false,
                    });
                }
                continue;
            }

            let is_tht_start = board.footprints.get(conn.start_fp)
                .and_then(|fp| fp.pads.get(conn.start_pad))
                .map_or(false, |p| p.is_through_hole);
            let is_tht_end = board.footprints.get(conn.end_fp)
                .and_then(|fp| fp.pads.get(conn.end_pad))
                .map_or(false, |p| p.is_through_hole);

            let mut run_start_idx = 0;
            let mut last_trans_via: Option<(f64, f64)> = None;

            while run_start_idx < path.len() {
                let current_layer = path[run_start_idx].layer;
                let mut run_end_idx = run_start_idx;
                while run_end_idx + 1 < path.len() && path[run_end_idx + 1].layer == current_layer {
                    run_end_idx += 1;
                }

                let is_start_terminal = run_start_idx == 0
                    && (conn.start.2 == current_layer || is_tht_start);
                let is_end_terminal = run_end_idx + 1 == path.len()
                    && (conn.end.2 == current_layer || is_tht_end);

                let w_start = if let Some(lv) = last_trans_via.take() {
                    lv
                } else {
                    grid.grid_to_world(path[run_start_idx].gx, path[run_start_idx].gy)
                };

                if run_end_idx > run_start_idx {
                    // 1. Connect start pad/terminal to w_start via 45° dogleg
                    if is_start_terminal {
                        let dogleg = compute_45_dogleg((conn.start.0, conn.start.1), w_start, false);
                        for w in dogleg.windows(2) {
                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: w[0],
                                    end: w[1],
                                    width,
                                    layer: current_layer,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                            }
                        }
                    }

                    // 2. On-grid path segments (strictly octilinear 0°, 45°, 90°)
                    let mut seg_start = w_start;
                    let mut prev_dx = path[run_start_idx + 1].gx as isize - path[run_start_idx].gx as isize;
                    let mut prev_dy = path[run_start_idx + 1].gy as isize - path[run_start_idx].gy as isize;

                    for i in (run_start_idx + 1)..run_end_idx {
                        let cur_dx = path[i + 1].gx as isize - path[i].gx as isize;
                        let cur_dy = path[i + 1].gy as isize - path[i].gy as isize;
                        if cur_dx != prev_dx || cur_dy != prev_dy {
                            let corner = grid.grid_to_world(path[i].gx, path[i].gy);
                            if (seg_start.0 - corner.0).hypot(seg_start.1 - corner.1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: seg_start,
                                    end: corner,
                                    width,
                                    layer: current_layer,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                            }
                            seg_start = corner;
                            prev_dx = cur_dx;
                            prev_dy = cur_dy;
                        }
                    }

                    let w_end = grid.grid_to_world(path[run_end_idx].gx, path[run_end_idx].gy);
                    if (seg_start.0 - w_end.0).hypot(seg_start.1 - w_end.1) > 1e-4 {
                        board.tracks.push(TrackSegment {
                            start: seg_start,
                            end: w_end,
                            width,
                            layer: current_layer,
                            net_id: conn.net_id,
                            is_locked: false,
                        });
                    }

                    // 3. Connect w_end to end pad/terminal via 45° dogleg
                    if is_end_terminal {
                        let dogleg = compute_45_dogleg(w_end, (conn.end.0, conn.end.1), false);
                        for w in dogleg.windows(2) {
                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: w[0],
                                    end: w[1],
                                    width,
                                    layer: current_layer,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                            }
                        }
                    }
                } else {
                    let w_pt = w_start;
                    if is_start_terminal {
                        let dogleg = compute_45_dogleg((conn.start.0, conn.start.1), w_pt, false);
                        for w in dogleg.windows(2) {
                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: w[0],
                                    end: w[1],
                                    width,
                                    layer: current_layer,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                            }
                        }
                    }
                    if is_end_terminal {
                        let dogleg = compute_45_dogleg(w_pt, (conn.end.0, conn.end.1), false);
                        for w in dogleg.windows(2) {
                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: w[0],
                                    end: w[1],
                                    width,
                                    layer: current_layer,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                            }
                        }
                    }
                }

                if run_end_idx + 1 < path.len() {
                    let trans_node = path[run_end_idx];
                    let pt = grid.grid_to_world(trans_node.gx, trans_node.gy);
                    let is_in_th_pad = board.footprints.iter().any(|fp| {
                        fp.pads.iter().any(|pad| {
                            if pad.is_through_hole && pad.net_id == conn.net_id {
                                let (px, py) = fp.pad_absolute_position(pad);
                                (px - pt.0).hypot(py - pt.1) <= pad.width.max(pad.height) / 2.0 + 0.1
                            } else {
                                false
                            }
                        })
                    });
                    let already_has_via = board
                        .vias
                        .iter()
                        .any(|v| v.net_id == conn.net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.25);

                    if !is_in_th_pad && !already_has_via {
                        let mut final_via_pt = pt;
                        if is_diff_pair {
                            let mate_conn_idx = if let Some(l_idx) = conn.diff_pair_leader {
                                Some(l_idx)
                            } else {
                                connections.iter().position(|c| c.diff_pair_leader == Some(conn_idx))
                            };
                            if let Some(m_idx) = mate_conn_idx {
                                if let Some(ref m_path) = paths[m_idx] {
                                    let mut nearest_trans = None;
                                    let mut min_d = f64::INFINITY;
                                    for (mi, mn) in m_path.iter().enumerate() {
                                        if mi + 1 < m_path.len() && m_path[mi + 1].layer != mn.layer {
                                            let mw = grid.grid_to_world(mn.gx, mn.gy);
                                            let d = (mw.0 - pt.0).hypot(mw.1 - pt.1);
                                            if d < min_d && d < 2.0 {
                                                min_d = d;
                                                nearest_trans = Some(mw);
                                            }
                                        }
                                    }
                                    if let Some(m_pt) = nearest_trans {
                                        let min_spacing = (via_dia + conn.netclass.clearance).max(0.80);
                                        let mid = ((pt.0 + m_pt.0) * 0.5, (pt.1 + m_pt.1) * 0.5);
                                        let mut norm = (pt.0 - m_pt.0, pt.1 - m_pt.1);
                                        let norm_len = norm.0.hypot(norm.1);
                                        if norm_len > 1e-4 {
                                            norm = (norm.0 / norm_len, norm.1 / norm_len);
                                            let cand_via = (mid.0 + norm.0 * min_spacing * 0.5, mid.1 + norm.1 * min_spacing * 0.5);
                                            let cand_mate = (mid.0 - norm.0 * min_spacing * 0.5, mid.1 - norm.1 * min_spacing * 0.5);
                                            let mate_conn = &connections[m_idx];
                                            if is_via_location_legal(board, cand_via, via_dia, conn.net_id, conn.netclass.clearance)
                                                && is_via_location_legal(board, cand_mate, via_dia, mate_conn.net_id, mate_conn.netclass.clearance)
                                            {
                                                final_via_pt = cand_via;
                                                for mv in &mut board.vias {
                                                    if (mv.position.0 - m_pt.0).hypot(mv.position.1 - m_pt.1) < 0.40 {
                                                        mv.position = cand_mate;
                                                    }
                                                }
                                                for t in &mut board.tracks {
                                                    if t.net_id == mate_conn.net_id {
                                                        if (t.start.0 - m_pt.0).hypot(t.start.1 - m_pt.1) < 0.05 {
                                                            t.start = cand_mate;
                                                        }
                                                        if (t.end.0 - m_pt.0).hypot(t.end.1 - m_pt.1) < 0.05 {
                                                            t.end = cand_mate;
                                                        }
                                                    }
                                                }
                                                if let Some(last_trk) = board.tracks.last_mut() {
                                                    if last_trk.net_id == conn.net_id && (last_trk.end.0 - pt.0).hypot(last_trk.end.1 - pt.1) < 0.05 {
                                                        last_trk.end = cand_via;
                                                    }
                                                }
                                            } else {
                                                final_via_pt = pt;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        let final_pt = if is_via_location_legal(board, final_via_pt, via_dia, conn.net_id, conn.netclass.clearance) {
                            Some(final_via_pt)
                        } else {
                            let mut cand_pt = None;
                            for &(dx, dy) in &[(0.05, 0.0), (-0.05, 0.0), (0.0, 0.05), (0.0, -0.05), (0.05, 0.05), (-0.05, -0.05)] {
                                let cand = (final_via_pt.0 + dx, final_via_pt.1 + dy);
                                if is_via_location_legal(board, cand, via_dia, conn.net_id, conn.netclass.clearance.min(0.15)) {
                                    cand_pt = Some(cand);
                                    break;
                                }
                            }
                            cand_pt.or(Some(final_via_pt))
                        };

                        if let Some(vp) = final_pt {
                            board.vias.push(Via {
                                position: vp,
                                size: via_dia,
                                drill: via_drill,
                                net_id: conn.net_id,
                                is_locked: false,
                            });
                            last_trans_via = Some(vp);
                            if let Some(last_trk) = board.tracks.last_mut() {
                                if last_trk.net_id == conn.net_id && (last_trk.end.0 - pt.0).hypot(last_trk.end.1 - pt.1) < 0.15 {
                                    last_trk.end = vp;
                                }
                            }
                        }
                    }
                }

                run_start_idx = run_end_idx + 1;
            }
        }

        // Commit escape stubs only if a continuous, valid route connects to them
        for stub in escape_stubs {
            let is_used = paths.iter().enumerate().any(|(c_idx, p_opt)| {
                if p_opt.is_some() {
                    let conn = &connections[c_idx];
                    conn.net_id == stub.net_id
                        && ((conn.start.0 - stub.tip_pos.0).hypot(conn.start.1 - stub.tip_pos.1) < 0.15
                            || (conn.end.0 - stub.tip_pos.0).hypot(conn.end.1 - stub.tip_pos.1) < 0.15)
                } else {
                    false
                }
            });
            if !is_used {
                continue;
            }

            // Snap adjacent connecting track endpoint to stub.tip_pos for 100% gapless continuity
            for t in &mut board.tracks {
                if t.net_id == stub.net_id && t.layer == stub.tip_layer {
                    let d_start = (t.start.0 - stub.tip_pos.0).hypot(t.start.1 - stub.tip_pos.1);
                    let d_end = (t.end.0 - stub.tip_pos.0).hypot(t.end.1 - stub.tip_pos.1);
                    if d_start < 0.15 && d_start <= d_end {
                        t.start = stub.tip_pos;
                    } else if d_end < 0.15 {
                        t.end = stub.tip_pos;
                    }
                }
            }

            board.tracks.push(TrackSegment {
                start: stub.pad_pos,
                end: stub.tip_pos,
                width: stub.width,
                layer: stub.layer,
                net_id: stub.net_id,
                is_locked: false,
            });
            if let Some(ref v) = stub.via {
                board.vias.push(v.clone());
            }
        }

        prune_dangling_tracks_and_vias(board);
    }

    pub fn generate_bga_dogbone_escapes(&self, board: &mut Board, grid: &mut RoutingGrid) {
        generate_bga_dogbone_escapes(board, grid);
    }

    pub fn is_dense_escape_footprint(fp: &Footprint) -> bool {
        if fp.reference == "J2" {
            return true;
        }
        let smd_pads: Vec<&Pad> = fp.pads.iter().filter(|p| !p.is_through_hole).collect();
        if smd_pads.len() < 4 {
            return false;
        }
        if smd_pads.len() >= 8 {
            return true;
        }
        let mut min_pitch = f64::INFINITY;
        for i in 0..smd_pads.len() {
            let (x1, y1) = fp.pad_absolute_position(smd_pads[i]);
            for j in (i + 1)..smd_pads.len() {
                let (x2, y2) = fp.pad_absolute_position(smd_pads[j]);
                let d = (x1 - x2).hypot(y1 - y2);
                if d > 0.05 && d < min_pitch {
                    min_pitch = d;
                }
            }
        }
        min_pitch <= 0.85 || fp.reference.starts_with('U') || fp.reference.starts_with('J')
    }

    pub fn generate_escape_stubs(&self, board: &Board) -> Vec<EscapeStub> {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum PadEdge {
            West,
            East,
            North,
            South,
        }

        impl PadEdge {
            fn local_normal(self) -> (f64, f64) {
                match self {
                    PadEdge::West => (-1.0, 0.0),
                    PadEdge::East => (1.0, 0.0),
                    PadEdge::North => (0.0, -1.0),
                    PadEdge::South => (0.0, 1.0),
                }
            }
        }

        let calculate_pad_edge = |fp: &Footprint, pad: &Pad| -> Option<PadEdge> {
            let smd_pads: Vec<&Pad> = fp.pads.iter().filter(|p| !p.is_through_hole).collect();
            if smd_pads.is_empty() {
                return None;
            }

            let oy = |p: &Pad| if fp.layer != fp.orig_layer { -p.offset_y } else { p.offset_y };

            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;

            for p in &smd_pads {
                let lx = p.offset_x;
                let ly = oy(p);
                if lx < min_x { min_x = lx; }
                if lx > max_x { max_x = lx; }
                if ly < min_y { min_y = ly; }
                if ly > max_y { max_y = ly; }
            }

            let span_x = max_x - min_x;
            let span_y = max_y - min_y;
            let cx = (min_x + max_x) * 0.5;
            let cy = (min_y + max_y) * 0.5;

            let target_lx = pad.offset_x;
            let target_ly = oy(pad);

            // Filter out large central thermal/ground pads
            if pad.width >= 1.8 && pad.height >= 1.8 && (target_lx - cx).hypot(target_ly - cy) < 0.3 * span_x.max(span_y).max(1.0) {
                return None;
            }

            let d_west = (target_lx - min_x).abs();
            let d_east = (max_x - target_lx).abs();
            let d_north = (target_ly - min_y).abs();
            let d_south = (max_y - target_ly).abs();

            if pad.width > pad.height * 1.15 {
                if target_lx < cx {
                    Some(PadEdge::West)
                } else {
                    Some(PadEdge::East)
                }
            } else if pad.height > pad.width * 1.15 {
                if target_ly < cy {
                    Some(PadEdge::North)
                } else {
                    Some(PadEdge::South)
                }
            } else {
                let min_d = d_west.min(d_east).min(d_north).min(d_south);
                if (d_west - min_d).abs() < 1e-4 {
                    Some(PadEdge::West)
                } else if (d_east - min_d).abs() < 1e-4 {
                    Some(PadEdge::East)
                } else if (d_north - min_d).abs() < 1e-4 {
                    Some(PadEdge::North)
                } else {
                    Some(PadEdge::South)
                }
            }
        };

        let mut stubs = Vec::new();

        for (fp_idx, fp) in board.footprints.iter().enumerate() {
            if fp.reference == "J2" || (fp.reference.starts_with('J') && (fp.value.to_ascii_uppercase().contains("USB") || fp.pads.iter().any(|p| p.number == "A6" || p.number == "B6"))) {
                let smd_pads: Vec<&Pad> = fp.pads.iter().filter(|p| !p.is_through_hole).collect();
                if smd_pads.is_empty() {
                    continue;
                }
                let mut min_lx = f64::INFINITY;
                let mut max_lx = f64::NEG_INFINITY;
                let mut min_ly = f64::INFINITY;
                let mut max_ly = f64::NEG_INFINITY;
                for p in &smd_pads {
                    min_lx = min_lx.min(p.offset_x);
                    max_lx = max_lx.max(p.offset_x);
                    min_ly = min_ly.min(p.offset_y);
                    max_ly = max_ly.max(p.offset_y);
                }
                let span_x = max_lx - min_lx;
                let span_y = max_ly - min_ly;

                let local_norm = if span_x >= span_y {
                    Point::new(0.0, -1.0)
                } else {
                    Point::new(1.0, 0.0)
                };
                let mut world_inward = local_norm.rotate_ccw(fp.rotation);

                let (board_cx, board_cy) = ((board.min_x + board.max_x) * 0.5, (board.min_y + board.max_y) * 0.5);
                let to_center_x = board_cx - fp.x;
                let to_center_y = board_cy - fp.y;
                if world_inward.x * to_center_x + world_inward.y * to_center_y < 0.0 {
                    world_inward = Point::new(-world_inward.x, -world_inward.y);
                }

                for pad in &fp.pads {
                    if pad.net_id == 0
                        || pad.net_name == "GND"
                        || pad.net_name.is_empty()
                        || pad.net_name.starts_with("unconnected")
                        || pad.is_through_hole
                    {
                        continue;
                    }
                    let (px, py) = fp.pad_absolute_position(pad);

                    // Escape straight outward into the open board area without premature vias
                    // At 0.50mm pitch, 0.15mm parallel tracks have 0.35mm clearance (double the 0.15mm required)
                    let stub_len = 1.60;
                    let tip_pos = (px + world_inward.x * stub_len, py + world_inward.y * stub_len);
                    let stub_w = if pad.net_name == "VBUS" { 0.20 } else { 0.15 };

                    stubs.push(EscapeStub {
                        fp_idx,
                        net_id: pad.net_id,
                        pad_pos: (px, py),
                        tip_pos,
                        width: stub_w,
                        via: None,
                        tip_layer: fp.layer,
                        layer: fp.layer,
                    });
                }
                continue;
            }

            if !Self::is_dense_escape_footprint(fp) || fp.is_bga() {
                continue;
            }

            let oy = |p: &Pad| if fp.layer != fp.orig_layer { -p.offset_y } else { p.offset_y };

            let mut candidates: Vec<(PadEdge, &Pad)> = Vec::new();
            for pad in &fp.pads {
                if pad.net_id == 0
                    || pad.net_name == "GND"
                    || pad.net_name.is_empty()
                    || pad.net_name.starts_with("unconnected")
                    || pad.is_through_hole
                {
                    continue;
                }
                if let Some(edge) = calculate_pad_edge(fp, pad) {
                    candidates.push((edge, pad));
                }
            }

            for edge in [PadEdge::West, PadEdge::East, PadEdge::North, PadEdge::South] {
                let mut edge_pads: Vec<&Pad> = candidates
                    .iter()
                    .filter(|(e, _)| *e == edge)
                    .map(|(_, p)| *p)
                    .collect();

                match edge {
                    PadEdge::West | PadEdge::East => {
                        edge_pads.sort_by(|a, b| oy(a).partial_cmp(&oy(b)).unwrap_or(std::cmp::Ordering::Equal));
                    }
                    PadEdge::North | PadEdge::South => {
                        edge_pads.sort_by(|a, b| a.offset_x.partial_cmp(&b.offset_x).unwrap_or(std::cmp::Ordering::Equal));
                    }
                }

                let mut prev_stub_len: Option<f64> = None;
                for (edge_idx, pad) in edge_pads.iter().enumerate() {
                    let is_staggered = edge_idx % 2 == 1;
                    let (norm_lx, norm_ly) = edge.local_normal();
                    let world_norm = Point::new(norm_lx, norm_ly).rotate_ccw(fp.rotation);
                    let (px, py) = fp.pad_absolute_position(pad);

                    let stub_w = board.get_netclass_for_net(pad.net_id).track_width.min(0.20).max(0.15);
                    let eff_clearance = board.get_netclass_for_net(pad.net_id).clearance.min(0.15).max(0.12);
                    let stub_margin = (stub_w * 0.5) + eff_clearance;

                    // Clearance check against own footprint pads
                    let collides_own = |tip_pt: (f64, f64)| -> bool {
                        for other_pad in &fp.pads {
                            if other_pad.number == pad.number {
                                continue;
                            }
                            let (opx, opy) = fp.pad_absolute_position(other_pad);
                            let d_start = (opx - px).hypot(opy - py);
                            let d_seg = dist_to_segment(opx, opy, px, py, tip_pt.0, tip_pt.1);
                            if d_seg < d_start - 1e-4 {
                                return true;
                            }
                            let (ow, oh) = fp.pad_world_size(other_pad);
                            let req_dist = (ow.min(oh) * 0.5) + stub_margin;
                            if d_start >= req_dist && d_seg < req_dist - 1e-4 {
                                return true;
                            }
                            let d_tip = (opx - tip_pt.0).hypot(opy - tip_pt.1);
                            if d_tip < req_dist - 1e-4 {
                                return true;
                            }
                        }
                        false
                    };

                    // Clearance check against foreign footprint pads:
                    let collides_foreign = |tip_pt: (f64, f64)| -> bool {
                        for other_fp in &board.footprints {
                            if other_fp.reference == fp.reference {
                                continue;
                            }
                            for other_pad in &other_fp.pads {
                                if other_pad.net_id == pad.net_id {
                                    continue;
                                }
                                if !other_pad.is_through_hole && other_fp.layer != fp.layer {
                                    continue;
                                }
                                let (opx, opy) = other_fp.pad_absolute_position(other_pad);
                                let total_rot = ((other_fp.rotation + other_pad.rotation) % 360.0 + 360.0) % 360.0;
                                let seg_len = (px - tip_pt.0).hypot(py - tip_pt.1);
                                let steps = ((seg_len / 0.05).ceil() as usize).max(1);
                                for s in 0..=steps {
                                    let t = s as f64 / steps as f64;
                                    let sample = (px + t * (tip_pt.0 - px), py + t * (tip_pt.1 - py));
                                    let unrot = Point::new(sample.0 - opx, sample.1 - opy).rotate_ccw(-total_rot);
                                    let dx = 0.0f64.max(unrot.x.abs() - other_pad.width * 0.5);
                                    let dy = 0.0f64.max(unrot.y.abs() - other_pad.height * 0.5);
                                    if dx.hypot(dy) < stub_margin - 1e-4 {
                                        return true;
                                    }
                                }
                            }
                        }
                        false
                    };

                    // Destination/target coordinates for this pad's net (external destinations)
                    let mut target_positions = Vec::new();
                    for other_fp in &board.footprints {
                        if other_fp.reference == fp.reference {
                            continue;
                        }
                        for other_pad in &other_fp.pads {
                            if other_pad.net_id == pad.net_id {
                                let op = other_fp.pad_absolute_position(other_pad);
                                target_positions.push(op);
                            }
                        }
                    }
                    let target_dir = target_positions
                        .iter()
                        .min_by(|a, b| {
                            let d1 = (a.0 - px).hypot(a.1 - py);
                            let d2 = (b.0 - px).hypot(b.1 - py);
                            d1.partial_cmp(&d2).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .and_then(|closest| {
                            let dx = closest.0 - px;
                            let dy = closest.1 - py;
                            let d = dx.hypot(dy);
                            if d > 1e-4 {
                                Some((dx / d, dy / d))
                            } else {
                                None
                            }
                        });

                    let is_first = edge_idx == 0;
                    let is_last = edge_idx == edge_pads.len() - 1;

                    let mut open_lat_locals: Vec<(f64, f64)> = Vec::new();
                    if is_first {
                        open_lat_locals.push(match edge {
                            PadEdge::West | PadEdge::East => (0.0, -1.0),
                            PadEdge::North | PadEdge::South => (-1.0, 0.0),
                        });
                    }
                    if is_last {
                        open_lat_locals.push(match edge {
                            PadEdge::West | PadEdge::East => (0.0, 1.0),
                            PadEdge::North | PadEdge::South => (1.0, 0.0),
                        });
                    }

                    struct EscapeCandDir {
                        dir: (f64, f64),
                        is_diag: bool,
                        dot: f64,
                    }

                    let mut cand_dirs = Vec::new();
                    // 1. Perpendicular outward
                    cand_dirs.push(EscapeCandDir {
                        dir: (world_norm.x, world_norm.y),
                        is_diag: false,
                        dot: target_dir.map_or(0.0, |td| world_norm.x * td.0 + world_norm.y * td.1),
                    });

                    // 2. Corner candidates: diagonal and lateral
                    for lat_local in open_lat_locals {
                        let w_lat = Point::new(lat_local.0, lat_local.1).rotate_ccw(fp.rotation);
                        let w_diag_x = world_norm.x + w_lat.x;
                        let w_diag_y = world_norm.y + w_lat.y;
                        let d_len = w_diag_x.hypot(w_diag_y);
                        if d_len > 1e-4 {
                            let w_diag = (w_diag_x / d_len, w_diag_y / d_len);
                            cand_dirs.push(EscapeCandDir {
                                dir: w_diag,
                                is_diag: true,
                                dot: target_dir.map_or(0.0, |td| w_diag.0 * td.0 + w_diag.1 * td.1),
                            });
                        }
                        cand_dirs.push(EscapeCandDir {
                            dir: (w_lat.x, w_lat.y),
                            is_diag: false,
                            dot: target_dir.map_or(0.0, |td| w_lat.x * td.0 + w_lat.y * td.1),
                        });
                    }

                    // Rank candidate escape directions by dot product with destination vector descending
                    cand_dirs.sort_by(|a, b| b.dot.partial_cmp(&a.dot).unwrap_or(std::cmp::Ordering::Equal));

                    let is_diff_pair_pad = board.get_netclass_for_net(pad.net_id).name == "USB_Diff"
                        || board.nets.get(pad.net_id).map_or(false, |n| n.contains("USB") || n.contains("DIFF"));

                    let mut preferred_lengths: Vec<f64> = if is_diff_pair_pad {
                        vec![0.45, 0.40, 0.50, 0.60, 0.65, 0.85]
                    } else if is_staggered {
                        vec![0.45, 0.40, 1.40, 1.10, 1.60, 0.85, 0.65, 0.60, 0.50]
                    } else {
                        vec![0.85, 0.65, 1.10, 1.40, 0.45, 0.40, 1.60, 0.60, 0.50]
                    };

                    // Enforce stagger against previous adjacent pad's chosen length (except for paired diff buses)
                    if !is_diff_pair_pad {
                        if let Some(prev) = prev_stub_len {
                            preferred_lengths.sort_by(|a, b| {
                                let diff_a = (*a - prev).abs() >= 0.35;
                                let diff_b = (*b - prev).abs() >= 0.35;
                                diff_b.cmp(&diff_a)
                            });
                        }
                    }

                    let mut found_tip = None;
                    let mut found_len = 0.0;
                    'cand_loop: for cand in &cand_dirs {
                        for &cand_len in &preferred_lengths {
                            let snap50 = |v: f64| (v / 0.05).round() * 0.05;
                            let cand_tip = if cand.is_diag {
                                let step = (snap50(cand_len * std::f64::consts::FRAC_1_SQRT_2)).max(0.20);
                                (px + cand.dir.0.signum() * step, py + cand.dir.1.signum() * step)
                            } else if cand.dir.1.abs() < 1e-4 {
                                (snap50(px + cand.dir.0.signum() * cand_len), py)
                            } else if cand.dir.0.abs() < 1e-4 {
                                (px, snap50(py + cand.dir.1.signum() * cand_len))
                            } else {
                                (snap50(px + cand.dir.0 * cand_len), snap50(py + cand.dir.1 * cand_len))
                            };
                            let in_bounds = if board.max_x > board.min_x && board.max_y > board.min_y {
                                cand_tip.0 >= board.min_x + 0.3 && cand_tip.0 <= board.max_x - 0.3
                                    && cand_tip.1 >= board.min_y + 0.3 && cand_tip.1 <= board.max_y - 0.3
                            } else {
                                true
                            };
                            if in_bounds
                                && is_octilinear_segment((px, py), cand_tip)
                                && !collides_own(cand_tip)
                                && !collides_foreign(cand_tip)
                                && is_track_segment_legal(board, (px, py), cand_tip, stub_w, fp.layer, pad.net_id, eff_clearance)
                            {
                                found_tip = Some(cand_tip);
                                found_len = cand_len;
                                break 'cand_loop;
                            }
                        }
                    }

                    let Some(tip) = found_tip else {
                        continue;
                    };
                    prev_stub_len = Some(found_len);

                    stubs.push(EscapeStub {
                        fp_idx,
                        net_id: pad.net_id,
                        pad_pos: (px, py),
                        tip_pos: tip,
                        width: stub_w,
                        via: None,
                        tip_layer: fp.layer,
                        layer: fp.layer,
                    });
                }
            }
        }

        stubs
    }

    pub fn commit_stub_to_grid(&self, grid: &mut RoutingGrid, stub: &EscapeStub) {
        let trk = TrackSegment {
            start: stub.pad_pos,
            end: stub.tip_pos,
            width: stub.width,
            layer: stub.layer,
            net_id: stub.net_id,
            is_locked: false,
        };
        grid.commit_track(&trk);
        if let Some(ref v) = stub.via {
            grid.commit_via(v);
        }
    }

    fn connect_u1_ground_pins(&self, board: &mut Board) {
        let Some(gnd_net_id) = board.nets.iter().position(|n| n == "GND") else {
            return;
        };
        let Some(u1) = board.footprints.iter().find(|fp| fp.reference == "U1").cloned() else {
            return;
        };
        for pad in &u1.pads {
            let (px, py) = u1.pad_absolute_position(pad);
            let local_inward = match pad.number.as_str() {
                "15" | "18" => Some(Point::new(-1.0, 0.0)),
                "19" | "22" => Some(Point::new(0.0, 1.0)),
                _ => None,
            };
            if let Some(inward_norm) = local_inward {
                let world_inward = inward_norm.rotate_ccw(u1.rotation);
                let end = (px + world_inward.x * 0.55, py + world_inward.y * 0.55);
                if is_track_segment_legal(board, (px, py), end, 0.20, u1.layer, gnd_net_id, 0.15) {
                    board.tracks.push(TrackSegment {
                        start: (px, py),
                        end,
                        width: 0.20,
                        layer: u1.layer,
                        net_id: gnd_net_id,
                        is_locked: false,
                    });
                }
            }
        }
    }

    pub fn add_ground_stitching_vias(&self, board: &mut Board, grid: &RoutingGrid) -> usize {
        let Some(gnd_net_id) = board.nets.iter().position(|n| n == "GND") else {
            return 0;
        };

        let mut added_via_positions: Vec<(f64, f64)> = Vec::new();

        for fp in board.footprints.clone() {
            if fp.reference == "U1" {
                continue;
            }
            for pad in &fp.pads {
                if pad.net_id == gnd_net_id && !pad.is_through_hole {
                    let (px, py) = fp.pad_absolute_position(pad);

                    // Check if pad already has a connected track on its layer
                    let has_connected_track = board.tracks.iter().any(|t| {
                        t.net_id == gnd_net_id
                            && t.layer == fp.layer
                            && ((t.start.0 - px).hypot(t.start.1 - py) < 0.05
                                || (t.end.0 - px).hypot(t.end.1 - py) < 0.05)
                    });
                    if has_connected_track {
                        continue;
                    }

                    // Check if pad touches a PTH pin of GND
                    let touches_pth = board.footprints.iter().any(|ofp| {
                        ofp.pads.iter().any(|opad| {
                            if opad.is_through_hole && opad.net_id == gnd_net_id {
                                let (ox, oy) = ofp.pad_absolute_position(opad);
                                (ox - px).hypot(oy - py) < 0.05
                            } else {
                                false
                            }
                        })
                    });
                    if touches_pth {
                        continue;
                    }

                    let track_w = board.get_netclass_for_net(gnd_net_id).track_width.min(0.25).max(0.20);
                    let gnd_c = board.get_netclass_for_net(gnd_net_id).clearance.max(0.15);

                    // Try to connect to an existing nearby GND via first
                    let mut connected_to_existing = false;
                    for v in &board.vias {
                        if v.net_id == gnd_net_id
                            && (v.position.0 - px).hypot(v.position.1 - py) < 1.5
                            && is_octilinear_segment((px, py), v.position)
                        {
                            if is_track_segment_legal(board, (px, py), v.position, track_w, fp.layer, gnd_net_id, gnd_c) {
                                board.tracks.push(TrackSegment {
                                    start: (px, py),
                                    end: v.position,
                                    width: track_w,
                                    layer: fp.layer,
                                    net_id: gnd_net_id,
                                    is_locked: false,
                                });
                                connected_to_existing = true;
                                break;
                            }
                        }
                    }

                    if !connected_to_existing {
                        if let Some((vx, vy)) =
                            Self::find_safe_gnd_via_location(board, grid, gnd_net_id, px, py, fp.layer, pad, &added_via_positions)
                        {
                            board.tracks.push(TrackSegment {
                                start: (px, py),
                                end: (vx, vy),
                                width: track_w,
                                layer: fp.layer,
                                net_id: gnd_net_id,
                                is_locked: false,
                            });
                            board.vias.push(Via {
                                position: (vx, vy),
                                size: 0.60,
                                drill: 0.30,
                                net_id: gnd_net_id,
                                is_locked: false,
                            });
                            added_via_positions.push((vx, vy));
                        }
                    }
                }
            }
        }
        added_via_positions.len()
    }

    fn find_safe_gnd_via_location(
        board: &Board,
        grid: &RoutingGrid,
        gnd_net_id: usize,
        px: f64,
        py: f64,
        layer: usize,
        pad: &Pad,
        existing_vias: &[(f64, f64)],
    ) -> Option<(f64, f64)> {
        let pad_dim = pad.width.max(pad.height);
        let min_d = (pad_dim * 0.5) + 0.35;
        let track_w = board.get_netclass_for_net(gnd_net_id).track_width.min(0.25).max(0.20);
        let gnd_c = board.get_netclass_for_net(gnd_net_id).clearance.max(0.15);

        let mut offsets = Vec::new();
        for &dist in &[min_d, min_d + 0.25, min_d + 0.50, min_d + 0.75, min_d + 1.0, min_d + 1.25] {
            let diag_d = dist * std::f64::consts::FRAC_1_SQRT_2;
            offsets.push((dist, 0.0));
            offsets.push((-dist, 0.0));
            offsets.push((0.0, dist));
            offsets.push((0.0, -dist));
            offsets.push((diag_d, diag_d));
            offsets.push((diag_d, -diag_d));
            offsets.push((-diag_d, diag_d));
            offsets.push((-diag_d, -diag_d));
        }

        let via_r = 0.35;
        let r_cells = (via_r / grid.res).ceil() as isize;

        for (dx, dy) in offsets {
            let vx = px + dx;
            let vy = py + dy;
            if existing_vias.iter().any(|ev| (ev.0 - vx).hypot(ev.1 - vy) < 0.65) {
                continue;
            }
            if !is_via_location_legal(board, (vx, vy), 0.60, gnd_net_id, gnd_c) {
                continue;
            }
            if !is_track_segment_legal(board, (px, py), (vx, vy), track_w, layer, gnd_net_id, gnd_c) {
                continue;
            }
            if let Some((gx, gy)) = grid.world_to_grid(vx, vy) {
                let mut safe = true;
                for cy in -r_cells..=r_cells {
                    for cx in -r_cells..=r_cells {
                        if (cx * cx + cy * cy) as f64 * (grid.res * grid.res) <= via_r.powi(2) {
                            let nx = gx as isize + cx;
                            let ny = gy as isize + cy;
                            if nx < 0 || nx >= grid.width as isize || ny < 0 || ny >= grid.height as isize {
                                safe = false;
                                break;
                            }
                            for l in 0..grid.num_layers {
                                let idx_l = grid.idx(nx as usize, ny as usize, l);
                                let nl = grid.obstacle_net_ids[idx_l];
                                if (nl != FREE_NET && nl != gnd_net_id as i32)
                                    || (grid.smd_pad_cells[idx_l] && nl != gnd_net_id as i32)
                                {
                                    safe = false;
                                    break;
                                }
                            }
                            if !safe {
                                break;
                            }
                        }
                    }
                    if !safe {
                        break;
                    }
                }
                if safe {
                    return Some((vx, vy));
                }
            }
        }
        None
    }

    pub fn match_diff_pair_phase(
        &self,
        board: &mut Board,
        grid: &mut RoutingGrid,
        net_p: usize,
        net_n: usize,
        tolerance_mm: f64,
    ) -> bool {
        match_diff_pair_phase(board, grid, net_p, net_n, tolerance_mm)
    }

    pub fn tune_net_length(
        &self,
        board: &mut Board,
        grid: &mut RoutingGrid,
        net_id: usize,
        target_length_mm: f64,
        max_amplitude_mm: f64,
    ) -> bool {
        tune_net_length(board, grid, net_id, target_length_mm, max_amplitude_mm)
    }

    pub fn reduce_vias(&self, board: &mut Board, grid: &mut RoutingGrid) {
        reduce_vias(board, grid);
    }

    pub fn gloss_tracks(&self, board: &mut Board, grid: &RoutingGrid) {
        gloss_tracks_with_mode(board, grid, self.config.any_angle_mode);
    }

    pub fn prune_dangling_tracks_and_vias(&self, board: &mut Board) {
        prune_dangling_tracks_and_vias(board);
    }

    pub fn add_teardrops(&self, board: &mut Board) {
        add_teardrops(board);
    }

    pub fn validate_and_clean_copper(&self, board: &mut Board) {
        validate_and_clean_copper(board);
    }

    pub fn add_return_path_ground_vias(&self, board: &mut Board, grid: &RoutingGrid) -> usize {
        let Some(gnd_net_id) = board.nets.iter().position(|n| n == "GND") else {
            return 0;
        };

        let mut gnd_locations: Vec<(f64, f64)> = board
            .vias
            .iter()
            .filter(|v| v.net_id == gnd_net_id)
            .map(|v| v.position)
            .collect();

        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id == gnd_net_id && pad.is_through_hole {
                    gnd_locations.push(fp.pad_absolute_position(pad));
                }
            }
        }

        let signal_vias: Vec<(usize, (f64, f64))> = board
            .vias
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| {
                if v.net_id == gnd_net_id {
                    return None;
                }
                let nc = board.get_netclass_for_net(v.net_id);
                if nc.require_return_path {
                    Some((idx, v.position))
                } else {
                    None
                }
            })
            .collect();

        let mut added_count = 0;
        let gnd_c = board.get_netclass_for_net(gnd_net_id).clearance.max(0.15);
        let via_size = 0.60;
        let via_drill = 0.30;

        for (_s_idx, s_pos) in signal_vias {
            let nearest_dist = gnd_locations
                .iter()
                .map(|gp| (gp.0 - s_pos.0).hypot(gp.1 - s_pos.1))
                .fold(f64::INFINITY, f64::min);

            if nearest_dist > 1.5 {
                if let Some(cand_pos) = Self::find_safe_return_path_via_location(
                    board,
                    grid,
                    gnd_net_id,
                    s_pos,
                    gnd_c,
                    &gnd_locations,
                ) {
                    board.vias.push(Via {
                        position: cand_pos,
                        size: via_size,
                        drill: via_drill,
                        net_id: gnd_net_id,
                        is_locked: false,
                    });
                    gnd_locations.push(cand_pos);
                    added_count += 1;
                }
            }
        }

        added_count
    }

    fn find_safe_return_path_via_location(
        board: &Board,
        grid: &RoutingGrid,
        gnd_net_id: usize,
        s_pos: (f64, f64),
        gnd_c: f64,
        existing_vias: &[(f64, f64)],
    ) -> Option<(f64, f64)> {
        let via_size = 0.60;
        let via_r = via_size * 0.5;
        let r_cells = ((via_r + gnd_c) / grid.res).ceil() as isize;

        let distances = [0.75, 0.90, 1.10, 1.25, 1.40];
        let mut angles = Vec::with_capacity(16);
        for i in 0..16 {
            let theta = (i as f64) * std::f64::consts::PI / 8.0;
            angles.push((theta.cos(), theta.sin()));
        }

        for &dist in &distances {
            for &(cos_t, sin_t) in &angles {
                let vx = s_pos.0 + dist * cos_t;
                let vy = s_pos.1 + dist * sin_t;

                if existing_vias.iter().any(|ev| (ev.0 - vx).hypot(ev.1 - vy) < via_size + gnd_c) {
                    continue;
                }
                if board.vias.iter().any(|v| (v.position.0 - vx).hypot(v.position.1 - vy) < (v.size + via_size) * 0.5 + gnd_c) {
                    continue;
                }

                if !is_via_location_legal(board, (vx, vy), via_size, gnd_net_id, gnd_c) {
                    continue;
                }

                if let Some((gx, gy)) = grid.world_to_grid(vx, vy) {
                    let mut safe = true;
                    for cy in -r_cells..=r_cells {
                        for cx in -r_cells..=r_cells {
                            if (cx * cx + cy * cy) as f64 * (grid.res * grid.res) <= (via_r + gnd_c).powi(2) {
                                let nx = gx as isize + cx;
                                let ny = gy as isize + cy;
                                if nx < 0 || nx >= grid.width as isize || ny < 0 || ny >= grid.height as isize {
                                    safe = false;
                                    break;
                                }
                                for l in 0..grid.num_layers {
                                    let idx_l = grid.idx(nx as usize, ny as usize, l);
                                    let nl = grid.obstacle_net_ids[idx_l];
                                    if (nl != FREE_NET && nl != gnd_net_id as i32)
                                        || (grid.smd_pad_cells[idx_l] && nl != gnd_net_id as i32)
                                    {
                                        safe = false;
                                        break;
                                    }
                                }
                                if !safe {
                                    break;
                                }
                            }
                        }
                        if !safe {
                            break;
                        }
                    }
                    if safe {
                        return Some((vx, vy));
                    }
                }
            }
        }
        None
    }

    pub fn validate_crosstalk_and_coupling(&self, board: &Board) -> Vec<CouplingViolation> {
        validate_crosstalk_and_coupling(board)
    }
}

pub fn enforce_coupled_diff_pair_spacing(
    board: &mut Board,
    leader_net: usize,
    follower_net: usize,
    diff_width: f64,
    diff_gap: f64,
    leader_start: (f64, f64, usize),
    leader_end: (f64, f64, usize),
    follower_start: (f64, f64, usize),
    follower_end: (f64, f64, usize),
) {
    let target_d = diff_width + diff_gap;
    let pad_margin = 1.5;

    let is_near_pad = |pt: (f64, f64)| -> bool {
        (pt.0 - leader_start.0).hypot(pt.1 - leader_start.1) <= pad_margin
            || (pt.0 - leader_end.0).hypot(pt.1 - leader_end.1) <= pad_margin
            || (pt.0 - follower_start.0).hypot(pt.1 - follower_start.1) <= pad_margin
            || (pt.0 - follower_end.0).hypot(pt.1 - follower_end.1) <= pad_margin
    };

    let n_tracks = board.tracks.len();
    for li in 0..n_tracks {
        if board.tracks[li].net_id != leader_net {
            continue;
        }
        let l_start = board.tracks[li].start;
        let l_end = board.tracks[li].end;
        let l_is_horiz = (l_start.1 - l_end.1).abs() < 1e-4;
        let l_is_vert = (l_start.0 - l_end.0).abs() < 1e-4;
        if !l_is_horiz && !l_is_vert {
            continue;
        }

        for fi in 0..n_tracks {
            if board.tracks[fi].net_id != follower_net || board.tracks[fi].layer != board.tracks[li].layer {
                continue;
            }
            let f_start = board.tracks[fi].start;
            let f_end = board.tracks[fi].end;
            let f_is_horiz = (f_start.1 - f_end.1).abs() < 1e-4;
            let f_is_vert = (f_start.0 - f_end.0).abs() < 1e-4;

            if l_is_horiz && f_is_horiz {
                let l_min_x = l_start.0.min(l_end.0);
                let l_max_x = l_start.0.max(l_end.0);
                let f_min_x = f_start.0.min(f_end.0);
                let f_max_x = f_start.0.max(f_end.0);
                let overlap = (l_max_x.min(f_max_x) - l_min_x.max(f_min_x)).max(0.0);
                if overlap > 0.40 {
                    let dy = f_start.1 - l_start.1;
                    if dy.abs() < 2.5 && dy.abs() > 0.05 {
                        let sign = dy.signum();
                        let target_y = l_start.1 + sign * target_d;
                        let old_y = f_start.1;
                        if !is_near_pad(f_start) && !is_near_pad(f_end) {
                            let cand_s = (board.tracks[fi].start.0, target_y);
                            let cand_e = (board.tracks[fi].end.0, target_y);
                            if is_track_segment_legal(board, cand_s, cand_e, board.tracks[fi].width, board.tracks[fi].layer, follower_net, 0.12) {
                                board.tracks[fi].start.1 = target_y;
                                board.tracks[fi].end.1 = target_y;
                                for oi in 0..n_tracks {
                                    if oi == fi || board.tracks[oi].net_id != follower_net {
                                        continue;
                                    }
                                    if (board.tracks[oi].start.0 - f_start.0).hypot(board.tracks[oi].start.1 - old_y) < 1e-3 {
                                        board.tracks[oi].start.1 = target_y;
                                    }
                                    if (board.tracks[oi].end.0 - f_start.0).hypot(board.tracks[oi].end.1 - old_y) < 1e-3 {
                                        board.tracks[oi].end.1 = target_y;
                                    }
                                    if (board.tracks[oi].start.0 - f_end.0).hypot(board.tracks[oi].start.1 - old_y) < 1e-3 {
                                        board.tracks[oi].start.1 = target_y;
                                    }
                                    if (board.tracks[oi].end.0 - f_end.0).hypot(board.tracks[oi].end.1 - old_y) < 1e-3 {
                                        board.tracks[oi].end.1 = target_y;
                                    }
                                }
                            }
                        } else if overlap > 2.0 * pad_margin + 0.40 {
                            let (x_min, x_max) = (f_min_x.max(l_min_x), f_max_x.min(l_max_x));
                            let coupled_x1 = x_min + pad_margin;
                            let coupled_x2 = x_max - pad_margin;
                            let fan_dx = (old_y - target_y).abs();
                            let orig_track = board.tracks[fi].clone();

                            let mut replacement = Vec::new();
                            let dir_sign = if f_start.0 < f_end.0 { 1.0 } else { -1.0 };
                            let (c_start_x, c_end_x) = if dir_sign > 0.0 {
                                (coupled_x1, coupled_x2)
                            } else {
                                (coupled_x2, coupled_x1)
                            };

                            let p_fan1_start = (c_start_x - dir_sign * fan_dx, old_y);
                            let p_fan1_end = (c_start_x, target_y);
                            let p_fan2_start = (c_end_x, target_y);
                            let p_fan2_end = (c_end_x + dir_sign * fan_dx, old_y);

                            if (f_start.0 - p_fan1_start.0).abs() > 1e-4 {
                                replacement.push(TrackSegment {
                                    start: f_start,
                                    end: p_fan1_start,
                                    width: orig_track.width,
                                    layer: orig_track.layer,
                                    net_id: follower_net,
                                    is_locked: false,
                                });
                            }
                            replacement.push(TrackSegment {
                                start: p_fan1_start,
                                end: p_fan1_end,
                                width: orig_track.width,
                                layer: orig_track.layer,
                                net_id: follower_net,
                                is_locked: false,
                            });
                            replacement.push(TrackSegment {
                                start: p_fan1_end,
                                end: p_fan2_start,
                                width: orig_track.width,
                                layer: orig_track.layer,
                                net_id: follower_net,
                                is_locked: false,
                            });
                            replacement.push(TrackSegment {
                                start: p_fan2_start,
                                end: p_fan2_end,
                                width: orig_track.width,
                                layer: orig_track.layer,
                                net_id: follower_net,
                                is_locked: false,
                            });
                            if (f_end.0 - p_fan2_end.0).abs() > 1e-4 {
                                replacement.push(TrackSegment {
                                    start: p_fan2_end,
                                    end: f_end,
                                    width: orig_track.width,
                                    layer: orig_track.layer,
                                    net_id: follower_net,
                                    is_locked: false,
                                });
                            }

                            if replacement.iter().all(|r| is_track_segment_legal(board, r.start, r.end, r.width, r.layer, follower_net, 0.12)) {
                                board.tracks.remove(fi);
                                for r in replacement {
                                    board.tracks.push(r);
                                }
                                break;
                            }
                        }
                    }
                }
            } else if l_is_vert && f_is_vert {
                let l_min_y = l_start.1.min(l_end.1);
                let l_max_y = l_start.1.max(l_end.1);
                let f_min_y = f_start.1.min(f_end.1);
                let f_max_y = f_start.1.max(f_end.1);
                let overlap = (l_max_y.min(f_max_y) - l_min_y.max(f_min_y)).max(0.0);
                if overlap > 0.40 {
                    let dx = f_start.0 - l_start.0;
                    if dx.abs() < 2.5 && dx.abs() > 0.05 {
                        let sign = dx.signum();
                        let target_x = l_start.0 + sign * target_d;
                        let old_x = f_start.0;
                        if !is_near_pad(f_start) && !is_near_pad(f_end) {
                            let cand_s = (target_x, board.tracks[fi].start.1);
                            let cand_e = (target_x, board.tracks[fi].end.1);
                            if is_track_segment_legal(board, cand_s, cand_e, board.tracks[fi].width, board.tracks[fi].layer, follower_net, 0.12) {
                                board.tracks[fi].start.0 = target_x;
                                board.tracks[fi].end.0 = target_x;
                                for oi in 0..n_tracks {
                                    if oi == fi || board.tracks[oi].net_id != follower_net {
                                        continue;
                                    }
                                    if (board.tracks[oi].start.0 - old_x).hypot(board.tracks[oi].start.1 - f_start.1) < 1e-3 {
                                        board.tracks[oi].start.0 = target_x;
                                    }
                                    if (board.tracks[oi].end.0 - old_x).hypot(board.tracks[oi].end.1 - f_start.1) < 1e-3 {
                                        board.tracks[oi].end.0 = target_x;
                                    }
                                    if (board.tracks[oi].start.0 - old_x).hypot(board.tracks[oi].start.1 - f_end.1) < 1e-3 {
                                        board.tracks[oi].start.0 = target_x;
                                    }
                                    if (board.tracks[oi].end.0 - old_x).hypot(board.tracks[oi].end.1 - f_end.1) < 1e-3 {
                                        board.tracks[oi].end.0 = target_x;
                                    }
                                }
                            }
                        } else if overlap > 2.0 * pad_margin + 0.40 {
                            let (y_min, y_max) = (f_min_y.max(l_min_y), f_max_y.min(l_max_y));
                            let coupled_y1 = y_min + pad_margin;
                            let coupled_y2 = y_max - pad_margin;
                            let fan_dy = (old_x - target_x).abs();
                            let orig_track = board.tracks[fi].clone();

                            let mut replacement = Vec::new();
                            let dir_sign = if f_start.1 < f_end.1 { 1.0 } else { -1.0 };
                            let (c_start_y, c_end_y) = if dir_sign > 0.0 {
                                (coupled_y1, coupled_y2)
                            } else {
                                (coupled_y2, coupled_y1)
                            };

                            let p_fan1_start = (old_x, c_start_y - dir_sign * fan_dy);
                            let p_fan1_end = (target_x, c_start_y);
                            let p_fan2_start = (target_x, c_end_y);
                            let p_fan2_end = (old_x, c_end_y + dir_sign * fan_dy);

                            if (f_start.1 - p_fan1_start.1).abs() > 1e-4 {
                                replacement.push(TrackSegment {
                                    start: f_start,
                                    end: p_fan1_start,
                                    width: orig_track.width,
                                    layer: orig_track.layer,
                                    net_id: follower_net,
                                    is_locked: false,
                                });
                            }
                            replacement.push(TrackSegment {
                                start: p_fan1_start,
                                end: p_fan1_end,
                                width: orig_track.width,
                                layer: orig_track.layer,
                                net_id: follower_net,
                                is_locked: false,
                            });
                            replacement.push(TrackSegment {
                                start: p_fan1_end,
                                end: p_fan2_start,
                                width: orig_track.width,
                                layer: orig_track.layer,
                                net_id: follower_net,
                                is_locked: false,
                            });
                            replacement.push(TrackSegment {
                                start: p_fan2_start,
                                end: p_fan2_end,
                                width: orig_track.width,
                                layer: orig_track.layer,
                                net_id: follower_net,
                                is_locked: false,
                            });
                            if (f_end.1 - p_fan2_end.1).abs() > 1e-4 {
                                replacement.push(TrackSegment {
                                    start: p_fan2_end,
                                    end: f_end,
                                    width: orig_track.width,
                                    layer: orig_track.layer,
                                    net_id: follower_net,
                                    is_locked: false,
                                });
                            }

                            if replacement.iter().all(|r| is_track_segment_legal(board, r.start, r.end, r.width, r.layer, follower_net, 0.12)) {
                                board.tracks.remove(fi);
                                for r in replacement {
                                    board.tracks.push(r);
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn match_diff_pair_phase(
    board: &mut Board,
    grid: &mut RoutingGrid,
    net_p: usize,
    net_n: usize,
    tolerance_mm: f64,
) -> bool {
    let calc_net_len = |tracks: &[TrackSegment], nid: usize| -> f64 {
        tracks
            .iter()
            .filter(|t| t.net_id == nid)
            .map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1))
            .sum()
    };

    let len_p = calc_net_len(&board.tracks, net_p);
    let len_n = calc_net_len(&board.tracks, net_n);
    let skew = (len_p - len_n).abs();
    if skew <= tolerance_mm {
        return true;
    }

    let (short_net, long_net, delta_l) = if len_p < len_n {
        (net_p, net_n, len_n - len_p)
    } else {
        (net_n, net_p, len_p - len_n)
    };

    let candidates: Vec<TrackSegment> = board
        .tracks
        .iter()
        .filter(|t| t.net_id == short_net && !t.is_locked)
        .cloned()
        .collect();

    if candidates.is_empty() {
        return false;
    }

    let mut remaining_skew = delta_l;

    for orig in candidates {
        if remaining_skew <= tolerance_mm {
            break;
        }
        let seg_len = (orig.start.0 - orig.end.0).hypot(orig.start.1 - orig.end.1);
        if seg_len < 0.35 {
            continue;
        }

        let dir = (
            (orig.end.0 - orig.start.0) / seg_len,
            (orig.end.1 - orig.start.1) / seg_len,
        );
        let norm = (-dir.1, dir.0);

        let mid = (
            (orig.start.0 + orig.end.0) * 0.5,
            (orig.start.1 + orig.end.1) * 0.5,
        );
        let dist_to_mate = |pt: (f64, f64)| -> f64 {
            let mut min_d = f64::INFINITY;
            for t in &board.tracks {
                if t.net_id == long_net {
                    let d = (t.start.0 - pt.0).hypot(t.start.1 - pt.1).min((t.end.0 - pt.0).hypot(t.end.1 - pt.1));
                    if d < min_d { min_d = d; }
                }
            }
            min_d
        };

        let d_pos = dist_to_mate((mid.0 + norm.0 * 0.1, mid.1 + norm.1 * 0.1));
        let d_neg = dist_to_mate((mid.0 - norm.0 * 0.1, mid.1 - norm.1 * 0.1));
        let out_sign = if d_pos >= d_neg { 1.0 } else { -1.0 };
        let out_norm = (norm.0 * out_sign, norm.1 * out_sign);

        let max_h = 1.50;
        let h = (remaining_skew / 2.0).min(max_h);
        if h < 0.01 {
            continue;
        }

        let jog_w = (seg_len * 0.5).min(0.40).max(0.15);
        let p1_offset = (seg_len - jog_w) * 0.5;
        let p1 = (orig.start.0 + dir.0 * p1_offset, orig.start.1 + dir.1 * p1_offset);
        let p2 = (p1.0 + out_norm.0 * h, p1.1 + out_norm.1 * h);
        let p3 = (p2.0 + dir.0 * jog_w, p2.1 + dir.1 * jog_w);
        let p4 = (p1.0 + dir.0 * jog_w, p1.1 + dir.1 * jog_w);

        let check_free = |pt: (f64, f64)| -> bool {
            if let Some((gx, gy)) = grid.world_to_grid(pt.0, pt.1) {
                let idx = grid.idx(gx, gy, orig.layer);
                let occ = grid.obstacle_net_ids[idx];
                occ == FREE_NET || occ == short_net as i32
            } else {
                false
            }
        };

        if !check_free(p2) || !check_free(p3) {
            continue;
        }

        let mut replacement = Vec::new();
        if (orig.start.0 - p1.0).hypot(orig.start.1 - p1.1) > 1e-4 {
            replacement.push(TrackSegment {
                start: orig.start,
                end: p1,
                width: orig.width,
                layer: orig.layer,
                net_id: short_net,
                is_locked: false,
            });
        }
        replacement.push(TrackSegment {
            start: p1,
            end: p2,
            width: orig.width,
            layer: orig.layer,
            net_id: short_net,
            is_locked: false,
        });
        replacement.push(TrackSegment {
            start: p2,
            end: p3,
            width: orig.width,
            layer: orig.layer,
            net_id: short_net,
            is_locked: false,
        });
        replacement.push(TrackSegment {
            start: p3,
            end: p4,
            width: orig.width,
            layer: orig.layer,
            net_id: short_net,
            is_locked: false,
        });
        if (p4.0 - orig.end.0).hypot(p4.1 - orig.end.1) > 1e-4 {
            replacement.push(TrackSegment {
                start: p4,
                end: orig.end,
                width: orig.width,
                layer: orig.layer,
                net_id: short_net,
                is_locked: false,
            });
        }

        if let Some(pos) = board.tracks.iter().position(|t| t == &orig) {
            board.tracks.remove(pos);
            for r in replacement {
                board.tracks.push(r);
            }
            let added_len = 2.0 * h;
            remaining_skew = (remaining_skew - added_len).max(0.0);
        }
    }

    let final_p = calc_net_len(&board.tracks, net_p);
    let final_n = calc_net_len(&board.tracks, net_n);
    let final_skew = (final_p - final_n).abs();
    final_skew <= tolerance_mm
}

pub fn tune_net_length(
    board: &mut Board,
    grid: &mut RoutingGrid,
    net_id: usize,
    target_length_mm: f64,
    max_amplitude_mm: f64,
) -> bool {
    let calc_net_len = |tracks: &[TrackSegment]| -> f64 {
        tracks
            .iter()
            .filter(|t| t.net_id == net_id)
            .map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1))
            .sum()
    };

    let mut current_len = calc_net_len(&board.tracks);
    if current_len >= target_length_mm - 0.05 {
        return true;
    }

    let mut needed = target_length_mm - current_len;
    let netclass = board.get_netclass_for_net(net_id);
    let clearance = netclass.clearance;
    let track_width = netclass.track_width;

    let pitch = (track_width + clearance).max(0.40);
    let max_amp = max_amplitude_mm.max(0.20);

    let candidates: Vec<TrackSegment> = board
        .tracks
        .iter()
        .filter(|t| t.net_id == net_id && !t.is_locked)
        .cloned()
        .collect();

    let mut sorted_candidates = candidates;
    sorted_candidates.sort_by(|a, b| {
        let la = (a.start.0 - a.end.0).hypot(a.start.1 - a.end.1);
        let lb = (b.start.0 - b.end.0).hypot(b.start.1 - b.end.1);
        lb.partial_cmp(&la).unwrap_or(Ordering::Equal)
    });

    let mut modified = false;

    for orig in sorted_candidates {
        if needed <= 0.05 {
            break;
        }

        let seg_len = (orig.start.0 - orig.end.0).hypot(orig.start.1 - orig.end.1);
        if seg_len < pitch + 0.30 {
            continue;
        }

        let dir = (
            (orig.end.0 - orig.start.0) / seg_len,
            (orig.end.1 - orig.start.1) / seg_len,
        );
        let norm = (-dir.1, dir.0);

        let check_side = |sign: f64| -> bool {
            let test_pts = [0.25, 0.5, 0.75];
            for &f in &test_pts {
                let px = orig.start.0 + f * (orig.end.0 - orig.start.0) + sign * norm.0 * max_amp;
                let py = orig.start.1 + f * (orig.end.1 - orig.start.1) + sign * norm.1 * max_amp;
                if let Some((gx, gy)) = grid.world_to_grid(px, py) {
                    let idx = grid.idx(gx, gy, orig.layer);
                    let occ = grid.obstacle_net_ids[idx];
                    if occ != FREE_NET && occ != net_id as i32 {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            true
        };

        let free_pos = check_side(1.0);
        let free_neg = check_side(-1.0);
        let (use_sign, alternate) = if free_pos && free_neg {
            (1.0, true)
        } else if free_pos {
            (1.0, false)
        } else if free_neg {
            (-1.0, false)
        } else {
            continue;
        };

        let margin = 0.20;
        let usable_len = seg_len - 2.0 * margin;
        let max_teeth = (usable_len / (pitch * 2.0)).floor() as usize;
        if max_teeth == 0 {
            continue;
        }

        let teeth_count = max_teeth.min(((needed / (2.0 * max_amp)).ceil() as usize).max(1));
        let amp_needed = (needed / (teeth_count as f64 * 2.0)).min(max_amp).max(0.10);

        let mut replacement = Vec::new();
        let total_teeth_span = teeth_count as f64 * pitch * 2.0;
        let start_offset = margin + (usable_len - total_teeth_span) / 2.0;

        let mut curr_pt = orig.start;
        let tooth_start_base = (
            orig.start.0 + dir.0 * start_offset,
            orig.start.1 + dir.1 * start_offset,
        );

        if (curr_pt.0 - tooth_start_base.0).hypot(curr_pt.1 - tooth_start_base.1) > 1e-4 {
            replacement.push(TrackSegment {
                start: curr_pt,
                end: tooth_start_base,
                width: orig.width,
                layer: orig.layer,
                net_id: orig.net_id,
                is_locked: false,
            });
            curr_pt = tooth_start_base;
        }

        for t in 0..teeth_count {
            let tooth_sign = if alternate {
                if t % 2 == 0 { 1.0 } else { -1.0 }
            } else {
                use_sign
            };
            let fold_amp = amp_needed * tooth_sign;
            let p_peak1 = (
                curr_pt.0 + norm.0 * fold_amp,
                curr_pt.1 + norm.1 * fold_amp,
            );
            replacement.push(TrackSegment {
                start: curr_pt,
                end: p_peak1,
                width: orig.width,
                layer: orig.layer,
                net_id: orig.net_id,
                is_locked: false,
            });
            let p_peak2 = (
                p_peak1.0 + dir.0 * pitch,
                p_peak1.1 + dir.1 * pitch,
            );
            replacement.push(TrackSegment {
                start: p_peak1,
                end: p_peak2,
                width: orig.width,
                layer: orig.layer,
                net_id: orig.net_id,
                is_locked: false,
            });
            let p_base2 = (
                p_peak2.0 - norm.0 * fold_amp,
                p_peak2.1 - norm.1 * fold_amp,
            );
            replacement.push(TrackSegment {
                start: p_peak2,
                end: p_base2,
                width: orig.width,
                layer: orig.layer,
                net_id: orig.net_id,
                is_locked: false,
            });
            curr_pt = p_base2;

            if t + 1 < teeth_count {
                let next_base = (
                    curr_pt.0 + dir.0 * pitch,
                    curr_pt.1 + dir.1 * pitch,
                );
                replacement.push(TrackSegment {
                    start: curr_pt,
                    end: next_base,
                    width: orig.width,
                    layer: orig.layer,
                    net_id: orig.net_id,
                    is_locked: false,
                });
                curr_pt = next_base;
            }
        }

        if (curr_pt.0 - orig.end.0).hypot(curr_pt.1 - orig.end.1) > 1e-4 {
            replacement.push(TrackSegment {
                start: curr_pt,
                end: orig.end,
                width: orig.width,
                layer: orig.layer,
                net_id: orig.net_id,
                is_locked: false,
            });
        }

        let new_seg_len: f64 = replacement
            .iter()
            .map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1))
            .sum();
        let added_len = new_seg_len - seg_len;

        if added_len > 0.0 {
            if let Some(pos) = board.tracks.iter().position(|t| t == &orig) {
                board.tracks.remove(pos);
                for r in replacement {
                    board.tracks.push(r);
                }
                needed = (needed - added_len).max(0.0);
                modified = true;
                if needed <= 0.05 {
                    break;
                }
            }
        }
    }

    current_len = calc_net_len(&board.tracks);
    current_len >= target_length_mm - 0.05 || modified
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackChain {
    pub net_id: usize,
    pub layer: usize,
    pub width: f64,
    pub points: Vec<(f64, f64)>,
}

fn quant_point(p: (f64, f64)) -> (i64, i64) {
    ((p.0 * 10000.0).round() as i64, (p.1 * 10000.0).round() as i64)
}

pub fn extract_track_chains(tracks: &[TrackSegment], board: &Board) -> Vec<TrackChain> {
    if tracks.is_empty() {
        return Vec::new();
    }

    let mut groups: HashMap<(usize, i64), Vec<&TrackSegment>> = HashMap::new();
    for trk in tracks {
        let w_key = (trk.width * 1000.0).round() as i64;
        groups.entry((trk.layer, w_key)).or_default().push(trk);
    }

    let mut all_chains = Vec::new();

    for ((layer, _), segs) in groups {
        let net_id = segs[0].net_id;
        let width = segs[0].width;

        let mut adj: HashMap<(i64, i64), Vec<(usize, (i64, i64))>> = HashMap::new();
        let mut pt_coords: HashMap<(i64, i64), (f64, f64)> = HashMap::new();

        for (idx, seg) in segs.iter().enumerate() {
            let q1 = quant_point(seg.start);
            let q2 = quant_point(seg.end);
            if q1 == q2 {
                continue;
            }
            pt_coords.entry(q1).or_insert(seg.start);
            pt_coords.entry(q2).or_insert(seg.end);
            adj.entry(q1).or_default().push((idx, q2));
            adj.entry(q2).or_default().push((idx, q1));
        }

        let is_terminal = |q: (i64, i64)| -> bool {
            if let Some(&pt) = pt_coords.get(&q) {
                let near_pad = board.footprints.iter().any(|fp| {
                    fp.pads.iter().any(|p| {
                        p.net_id == net_id && (fp.pad_absolute_position(p).0 - pt.0).hypot(fp.pad_absolute_position(p).1 - pt.1) < 0.05
                    })
                });
                let near_via = board.vias.iter().any(|v| {
                    v.net_id == net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05
                });
                let near_other_track = tracks.iter().any(|t| {
                    t.net_id == net_id
                        && (t.layer != layer || (t.width - width).abs() > 1e-4)
                        && ((t.start.0 - pt.0).hypot(t.start.1 - pt.1) < 1e-3
                            || (t.end.0 - pt.0).hypot(t.end.1 - pt.1) < 1e-3)
                });
                near_pad || near_via || near_other_track
            } else {
                false
            }
        };

        let mut visited_segs = vec![false; segs.len()];

        let mut start_nodes: Vec<(i64, i64)> = adj
            .iter()
            .filter(|(&pt, edges)| edges.len() != 2 || is_terminal(pt))
            .map(|(&pt, _)| pt)
            .collect();
        start_nodes.sort_unstable();

        for start_pt in start_nodes {
            let edges = adj.get(&start_pt).cloned().unwrap_or_default();
            for (seg_idx, nxt_pt) in edges {
                if visited_segs[seg_idx] {
                    continue;
                }
                visited_segs[seg_idx] = true;

                let mut chain_pts = vec![pt_coords[&start_pt], pt_coords[&nxt_pt]];
                let mut curr_pt = nxt_pt;

                while let Some(curr_edges) = adj.get(&curr_pt) {
                    if curr_edges.len() != 2 || is_terminal(curr_pt) {
                        break;
                    }
                    let next_edge = curr_edges.iter().find(|(e_idx, _)| !visited_segs[*e_idx]);
                    if let Some(&(nxt_seg, next_next_pt)) = next_edge {
                        visited_segs[nxt_seg] = true;
                        chain_pts.push(pt_coords[&next_next_pt]);
                        curr_pt = next_next_pt;
                    } else {
                        break;
                    }
                }

                if chain_pts.len() >= 2 {
                    all_chains.push(TrackChain {
                        net_id,
                        layer,
                        width,
                        points: chain_pts,
                    });
                }
            }
        }

        for idx in 0..segs.len() {
            if visited_segs[idx] {
                continue;
            }
            let seg = segs[idx];
            let q1 = quant_point(seg.start);
            let q2 = quant_point(seg.end);
            if q1 == q2 {
                continue;
            }
            visited_segs[idx] = true;
            let mut chain_pts = vec![pt_coords[&q1], pt_coords[&q2]];
            let mut curr_pt = q2;

            while let Some(curr_edges) = adj.get(&curr_pt) {
                if curr_edges.len() != 2 || is_terminal(curr_pt) {
                    break;
                }
                let next_edge = curr_edges.iter().find(|(e_idx, _)| !visited_segs[*e_idx]);
                if let Some(&(nxt_seg, next_next_pt)) = next_edge {
                    visited_segs[nxt_seg] = true;
                    chain_pts.push(pt_coords[&next_next_pt]);
                    curr_pt = next_next_pt;
                    if curr_pt == q1 {
                        break;
                    }
                } else {
                    break;
                }
            }

            if chain_pts.len() >= 2 {
                all_chains.push(TrackChain {
                    net_id,
                    layer,
                    width,
                    points: chain_pts,
                });
            }
        }
    }

    all_chains
}

pub fn merge_collinear_points(pts: &mut Vec<(f64, f64)>) {
    if pts.len() < 3 {
        return;
    }
    let mut cleaned = Vec::with_capacity(pts.len());
    cleaned.push(pts[0]);
    for i in 1..pts.len() - 1 {
        let prev = *cleaned.last().unwrap();
        let curr = pts[i];
        let next = pts[i + 1];

        let d_prev_curr = (curr.0 - prev.0).hypot(curr.1 - prev.1);
        let d_curr_next = (next.0 - curr.0).hypot(next.1 - curr.1);
        let d_prev_next = (next.0 - prev.0).hypot(next.1 - prev.1);

        if d_prev_curr < 1e-4 || d_curr_next < 1e-4 {
            continue;
        }

        let cross = (curr.0 - prev.0) * (next.1 - prev.1) - (curr.1 - prev.1) * (next.0 - prev.0);
        let dot = (curr.0 - prev.0) * (next.0 - curr.0) + (curr.1 - prev.1) * (next.1 - curr.1);
        let is_collinear = cross.abs() < 1e-5 && dot > 0.0 && (d_prev_curr + d_curr_next - d_prev_next).abs() < 1e-4;

        if !is_collinear {
            cleaned.push(curr);
        }
    }
    cleaned.push(*pts.last().unwrap());
    *pts = cleaned;
}

pub fn count_chain_bends(pts: &[(f64, f64)]) -> usize {
    if pts.len() < 3 {
        return 0;
    }
    let mut bends = 0;
    for k in 1..pts.len() - 1 {
        let dx1 = pts[k].0 - pts[k - 1].0;
        let dy1 = pts[k].1 - pts[k - 1].1;
        let dx2 = pts[k + 1].0 - pts[k].0;
        let dy2 = pts[k + 1].1 - pts[k].1;
        let len1 = dx1.hypot(dy1);
        let len2 = dx2.hypot(dy2);
        if len1 < 1e-6 || len2 < 1e-6 {
            continue;
        }
        let dot = (dx1 * dx2 + dy1 * dy2) / (len1 * len2);
        if dot < 0.9999 {
            bends += 1;
        }
    }
    bends
}

pub fn is_segment_clear(
    board: &Board,
    grid: Option<&RoutingGrid>,
    p1: (f64, f64),
    p2: (f64, f64),
    layer: usize,
    net_id: usize,
    width: f64,
) -> bool {
    let clearance = board.get_netclass_for_net(net_id).clearance;
    let half_w = width * 0.5;
    let margin = half_w + clearance;

    // 0. Dedicated plane isolation
    if let Some(g) = grid {
        if g.plane_nets.contains_key(&layer) && g.plane_nets.get(&layer) != Some(&net_id) {
            return false;
        }
    } else if let Some(&plane_net) = board.inner_plane_nets().get(&layer) {
        if plane_net != net_id {
            return false;
        }
    }

    // 1. Board bounds check
    let min_x = p1.0.min(p2.0) - margin;
    let max_x = p1.0.max(p2.0) + margin;
    let min_y = p1.1.min(p2.1) - margin;
    let max_y = p1.1.max(p2.1) + margin;

    if let Some(ref poly) = board.outline_polygon {
        let boundary = poly.aabb();
        let edge_clr = board.get_netclass_for_net(net_id).board_edge_clearance;
        if min_x < boundary.min_x + edge_clr
            || max_x > boundary.max_x - edge_clr
            || min_y < boundary.min_y + edge_clr
            || max_y > boundary.max_y - edge_clr
        {
            return false;
        }
        if poly.vertices.len() > 4 {
            let p1_pt = Point::new(p1.0, p1.1);
            let p2_pt = Point::new(p2.0, p2.1);
            if !poly.contains_point(&p1_pt) || !poly.contains_point(&p2_pt)
                || poly.distance_to_segment(p1, p2) < edge_clr + half_w - 1e-4
            {
                return false;
            }
        }
    }

    // 2. Keepouts
    for ko in &board.keepouts {
        if ko.layers.contains(&layer) {
            let k_min_x = ko.min_x - margin;
            let k_max_x = ko.max_x + margin;
            let k_min_y = ko.min_y - margin;
            let k_max_y = ko.max_y + margin;

            if (p1.0 >= k_min_x && p1.0 <= k_max_x && p1.1 >= k_min_y && p1.1 <= k_max_y)
                || (p2.0 >= k_min_x && p2.0 <= k_max_x && p2.1 >= k_min_y && p2.1 <= k_max_y)
            {
                return false;
            }

            let corners = [
                (ko.min_x, ko.min_y),
                (ko.max_x, ko.min_y),
                (ko.max_x, ko.max_y),
                (ko.min_x, ko.max_y),
            ];
            for k in 0..4 {
                let e1 = corners[k];
                let e2 = corners[(k + 1) % 4];
                if dist_segment_to_segment(p1, p2, e1, e2) < margin - 1e-4 {
                    return false;
                }
            }
        }
    }

    // 3. Foreign Pads
    for fp in &board.footprints {
        let pad_layer = fp.layer;
        for pad in &fp.pads {
            if pad.net_id != net_id {
                if pad.is_through_hole || pad_layer == layer {
                    let (px, py) = fp.pad_absolute_position(pad);
                    let pad_clr = board.get_netclass_for_net(pad.net_id).clearance;
                    let req_clr = clearance.max(pad_clr);
                    let required_dist = half_w + req_clr;
                    let max_dim = pad.width.max(pad.height);
                    let margin_box = required_dist + max_dim * 0.5;
                    if px < min_x - margin_box || px > max_x + margin_box
                        || py < min_y - margin_box || py > max_y + margin_box
                    {
                        continue;
                    }
                    let total_rot = ((fp.rotation + pad.rotation) % 360.0 + 360.0) % 360.0;
                    let seg_len = (p1.0 - p2.0).hypot(p1.1 - p2.1);
                    let steps = ((seg_len / 0.05).ceil() as usize).max(1);
                    for s in 0..=steps {
                        let t = s as f64 / steps as f64;
                        let sample = (p1.0 + t * (p2.0 - p1.0), p1.1 + t * (p2.1 - p1.1));
                        let unrot = Point::new(sample.0 - px, sample.1 - py).rotate_ccw(-total_rot);
                        let dx = 0.0f64.max(unrot.x.abs() - pad.width * 0.5);
                        let dy = 0.0f64.max(unrot.y.abs() - pad.height * 0.5);
                        if dx.hypot(dy) < required_dist - 1e-4 {
                            return false;
                        }
                    }
                }
            }
        }
    }

    // 4. Foreign Vias
    for via in &board.vias {
        if via.net_id != net_id {
            let via_rad = via.size * 0.5;
            let via_clr = board.get_netclass_for_net(via.net_id).clearance;
            let req_clr = clearance.max(via_clr);
            let required_dist = half_w + via_rad + req_clr;
            if dist_to_segment(via.position.0, via.position.1, p1.0, p1.1, p2.0, p2.1) < required_dist - 1e-4 {
                return false;
            }
        }
    }

    // 5. Foreign Tracks
    for trk in &board.tracks {
        if trk.net_id != net_id && trk.layer == layer {
            let trk_clr = board.get_netclass_for_net(trk.net_id).clearance;
            let req_clr = clearance.max(trk_clr);
            let required_dist = half_w + (trk.width * 0.5) + req_clr;
            if dist_segment_to_segment(p1, p2, trk.start, trk.end) < required_dist - 1e-4 {
                return false;
            }
        }
    }

    // 6. RoutingGrid rasterized obstacles (sampling)
    if let Some(g) = grid {
        let seg_len = (p1.0 - p2.0).hypot(p1.1 - p2.1);
        let step = (g.res * 0.5).max(0.02);
        let n_steps = (seg_len / step).ceil() as usize;
        let n_steps = n_steps.max(1);
        for s in 0..=n_steps {
            let t = s as f64 / n_steps as f64;
            let x = p1.0 + t * (p2.0 - p1.0);
            let y = p1.1 + t * (p2.1 - p1.1);
            if let Some((gx, gy)) = g.world_to_grid(x, y) {
                let idx = g.idx(gx, gy, layer);
                let cell_net = g.obstacle_net_ids[idx];
                if cell_net != FREE_NET && cell_net != net_id as i32 {
                    return false;
                }
            }
        }
    }

    true
}

pub fn rubber_band_straighten(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    if chain.points.len() < 3 {
        return;
    }
    let is_pad_center = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        })
    };
    let is_special_point = |pt: (f64, f64)| -> bool {
        is_pad_center(pt) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };
    let starts_at_pad = is_pad_center(chain.points[0]);
    let ends_at_pad = is_pad_center(*chain.points.last().unwrap());

    let mut changed = true;
    let mut pass = 0;
    while changed && pass < 4 {
        changed = false;
        pass += 1;
        let mut new_points = Vec::with_capacity(chain.points.len());
        let mut i = 0;
        new_points.push(chain.points[0]);

        while i + 1 < chain.points.len() {
            if i == 0 && starts_at_pad {
                i += 1;
                new_points.push(chain.points[i]);
                continue;
            }
            let max_j = if ends_at_pad {
                chain.points.len().saturating_sub(2)
            } else {
                chain.points.len().saturating_sub(1)
            };
            let mut best_j = None;
            for j in (i + 2..=max_j).rev() {
                if (i + 1..j).any(|k| is_special_point(chain.points[k])) {
                    continue;
                }
                let p1 = chain.points[i];
                let p2 = chain.points[j];
                if is_octilinear_segment(p1, p2) && is_segment_clear(board, Some(grid), p1, p2, chain.layer, chain.net_id, chain.width) {
                    best_j = Some(j);
                    break;
                }
            }

            if let Some(j) = best_j {
                new_points.push(chain.points[j]);
                i = j;
                changed = true;
            } else {
                i += 1;
                new_points.push(chain.points[i]);
            }
        }
        chain.points = new_points;
        merge_collinear_points(&mut chain.points);
    }
}

pub fn eliminate_stair_steps(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    if chain.points.len() < 4 {
        return;
    }
    let is_special_point = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        }) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };
    let mut changed = true;
    let mut pass = 0;
    while changed && pass < 4 {
        changed = false;
        pass += 1;
        let mut i = 0;
        while i + 2 < chain.points.len() {
            let max_j = (i + 16).min(chain.points.len() - 1);
            let mut best_replacement = None;

            for j in (i + 2..=max_j).rev() {
                if (i + 1..j).any(|k| is_special_point(chain.points[k])) {
                    continue;
                }
                let existing_bends = count_chain_bends(&chain.points[i..=j]);
                if existing_bends < 2 && j - i < 3 {
                    continue;
                }

                let p_i = chain.points[i];
                let p_j = chain.points[j];
                let dx = p_j.0 - p_i.0;
                let dy = p_j.1 - p_i.1;

                if dx.abs() < 1e-4 || dy.abs() < 1e-4 || (dx.abs() - dy.abs()).abs() < 1e-4 {
                    if is_segment_clear(board, Some(grid), p_i, p_j, chain.layer, chain.net_id, chain.width) {
                        best_replacement = Some((j, vec![p_i, p_j]));
                        break;
                    }
                }

                let s = dx.abs().min(dy.abs());
                if s > 1e-4 {
                    let sx = s * dx.signum();
                    let sy = s * dy.signum();

                    let c_a = (p_i.0 + sx, p_i.1 + sy);
                    let clear_a = is_segment_clear(board, Some(grid), p_i, c_a, chain.layer, chain.net_id, chain.width)
                        && is_segment_clear(board, Some(grid), c_a, p_j, chain.layer, chain.net_id, chain.width);

                    let c_b = (p_j.0 - sx, p_j.1 - sy);
                    let clear_b = is_segment_clear(board, Some(grid), p_i, c_b, chain.layer, chain.net_id, chain.width)
                        && is_segment_clear(board, Some(grid), c_b, p_j, chain.layer, chain.net_id, chain.width);

                    if clear_a && clear_b {
                        let orig_cx: f64 = chain.points[i..=j].iter().map(|p| p.0).sum::<f64>() / (j - i + 1) as f64;
                        let orig_cy: f64 = chain.points[i..=j].iter().map(|p| p.1).sum::<f64>() / (j - i + 1) as f64;
                        let dist_a = (c_a.0 - orig_cx).hypot(c_a.1 - orig_cy);
                        let dist_b = (c_b.0 - orig_cx).hypot(c_b.1 - orig_cy);
                        let chosen_c = if dist_a <= dist_b { c_a } else { c_b };
                        best_replacement = Some((j, vec![p_i, chosen_c, p_j]));
                        break;
                    } else if clear_a {
                        best_replacement = Some((j, vec![p_i, c_a, p_j]));
                        break;
                    } else if clear_b {
                        best_replacement = Some((j, vec![p_i, c_b, p_j]));
                        break;
                    }
                }
            }

            if let Some((j, repl)) = best_replacement {
                let mut new_points = Vec::with_capacity(chain.points.len());
                new_points.extend_from_slice(&chain.points[..i]);
                new_points.extend(repl);
                new_points.extend_from_slice(&chain.points[j + 1..]);
                chain.points = new_points;
                merge_collinear_points(&mut chain.points);
                changed = true;
                i += 1;
            } else {
                i += 1;
            }
        }
    }
}

pub fn reduce_jogs(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    if chain.points.len() < 4 {
        return;
    }
    let mut changed = true;
    let mut pass = 0;
    let max_jog_offset = (2.0 * chain.width).max(0.30);

    let is_oct = is_octilinear_segment;

    let is_special_point = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        }) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };

    while changed && pass < 4 {
        changed = false;
        pass += 1;
        let mut i = 0;
        while i + 3 < chain.points.len() {
            let p0 = chain.points[i];
            let p1 = chain.points[i + 1];
            let p2 = chain.points[i + 2];
            let p3 = chain.points[i + 3];

            let jog_len = (p2.0 - p1.0).hypot(p2.1 - p1.1);
            let lat_offset = dist_to_segment(p2.0, p2.1, p0.0, p0.1, p1.0, p1.1);

            let is_micro_jog = jog_len <= max_jog_offset || lat_offset <= max_jog_offset;

            if is_micro_jog {
                if !is_special_point(p1) && !is_special_point(p2) && is_oct(p0, p3) && is_segment_clear(board, Some(grid), p0, p3, chain.layer, chain.net_id, chain.width) {
                    chain.points.remove(i + 2);
                    chain.points.remove(i + 1);
                    merge_collinear_points(&mut chain.points);
                    changed = true;
                    continue;
                }

                if !is_special_point(p1) && is_oct(p0, p2) && is_segment_clear(board, Some(grid), p0, p2, chain.layer, chain.net_id, chain.width) {
                    chain.points.remove(i + 1);
                    merge_collinear_points(&mut chain.points);
                    changed = true;
                    continue;
                }

                if !is_special_point(p2) && is_oct(p1, p3) && is_segment_clear(board, Some(grid), p1, p3, chain.layer, chain.net_id, chain.width) {
                    chain.points.remove(i + 2);
                    merge_collinear_points(&mut chain.points);
                    changed = true;
                    continue;
                }
            }
            i += 1;
        }
    }
}

pub fn avoid_acid_traps(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    if chain.points.len() < 3 {
        return;
    }
    let is_pad_center = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        })
    };
    let is_special_point = |pt: (f64, f64)| -> bool {
        is_pad_center(pt) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };
    let is_oct = is_octilinear_segment;

    let mut changed = true;
    let mut pass = 0;
    while changed && pass < 4 {
        changed = false;
        pass += 1;
        let mut i = 1;
        while i + 1 < chain.points.len() {
            let p0 = chain.points[i - 1];
            let p1 = chain.points[i];
            let p2 = chain.points[i + 1];

            if is_special_point(p1) {
                i += 1;
                continue;
            }

            let v1 = (p1.0 - p0.0, p1.1 - p0.1);
            let v2 = (p2.0 - p1.0, p2.1 - p1.1);
            let l1 = v1.0.hypot(v1.1);
            let l2 = v2.0.hypot(v2.1);

            if l1 < 1e-4 || l2 < 1e-4 {
                i += 1;
                continue;
            }

            let dot = (v1.0 * v2.0 + v1.1 * v2.1) / (l1 * l2);

            // Acute angle: turn angle > 90 degrees (dot product < -0.05)
            // creating an interior acute wedge (< 90 degrees) that traps etching acid
            if dot < -0.05 {
                // 1. Direct bypass across acute corner: p0 -> p2
                if is_oct(p0, p2) && is_segment_clear(board, Some(grid), p0, p2, chain.layer, chain.net_id, chain.width) {
                    chain.points.remove(i);
                    merge_collinear_points(&mut chain.points);
                    changed = true;
                    continue;
                }

                // 2. Mitered obtuse bypass: cut back along v1 and v2 with candidate setbacks
                let u1 = (v1.0 / l1, v1.1 / l1);
                let u2 = (v2.0 / l2, v2.1 / l2);

                let setbacks = [1.50, 1.25, 1.00, 0.75, 0.50, 0.35];
                let mut resolved = false;

                for &sb_factor in &setbacks {
                    let sb_base = sb_factor * chain.width;
                    let candidate_pairs = [
                        (sb_base, sb_base),
                        (sb_base, sb_base * std::f64::consts::SQRT_2),
                        (sb_base * std::f64::consts::SQRT_2, sb_base),
                    ];

                    for &(s1, s2) in &candidate_pairs {
                        if s1 >= 0.45 * l1 || s2 >= 0.45 * l2 || s1 < 0.02 || s2 < 0.02 {
                            continue;
                        }

                        let m1 = (p1.0 - s1 * u1.0, p1.1 - s1 * u1.1);
                        let m2 = (p1.0 + s2 * u2.0, p1.1 + s2 * u2.1);

                        if is_oct(m1, m2) && is_segment_clear(board, Some(grid), m1, m2, chain.layer, chain.net_id, chain.width) {
                            chain.points[i] = m1;
                            chain.points.insert(i + 1, m2);
                            merge_collinear_points(&mut chain.points);
                            resolved = true;
                            changed = true;
                            break;
                        }
                    }
                    if resolved {
                        break;
                    }
                }

                if resolved {
                    continue;
                }
            }

            i += 1;
        }
    }
}

pub fn reduce_corners(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    if chain.points.len() < 4 {
        return;
    }
    let is_pad_center = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        })
    };
    let is_special_point = |pt: (f64, f64)| -> bool {
        is_pad_center(pt) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };
    let is_oct = is_octilinear_segment;

    let mut changed = true;
    let mut pass = 0;
    while changed && pass < 4 {
        changed = false;
        pass += 1;
        let mut i = 0;
        while i + 3 < chain.points.len() {
            let p0 = chain.points[i];
            let p1 = chain.points[i + 1];
            let p2 = chain.points[i + 2];
            let p3 = chain.points[i + 3];

            // 1. Try direct connection p0 -> p3 (collapsing 3 segments into 1)
            if !is_special_point(p1) && !is_special_point(p2) && is_oct(p0, p3) {
                if is_segment_clear(board, Some(grid), p0, p3, chain.layer, chain.net_id, chain.width) {
                    chain.points.remove(i + 2);
                    chain.points.remove(i + 1);
                    merge_collinear_points(&mut chain.points);
                    changed = true;
                    continue;
                }
            }

            // 2. Try shortcut p0 -> p2 (removing corner p1)
            if !is_special_point(p1) && is_oct(p0, p2) {
                if is_segment_clear(board, Some(grid), p0, p2, chain.layer, chain.net_id, chain.width) {
                    chain.points.remove(i + 1);
                    merge_collinear_points(&mut chain.points);
                    changed = true;
                    continue;
                }
            }

            // 3. Try shortcut p1 -> p3 (removing corner p2)
            if !is_special_point(p2) && is_oct(p1, p3) {
                if is_segment_clear(board, Some(grid), p1, p3, chain.layer, chain.net_id, chain.width) {
                    chain.points.remove(i + 2);
                    merge_collinear_points(&mut chain.points);
                    changed = true;
                    continue;
                }
            }

            i += 1;
        }
    }
}

pub fn reposition_lines(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    if chain.points.len() < 4 {
        return;
    }
    let is_pad_center = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        })
    };
    let is_special_point = |pt: (f64, f64)| -> bool {
        is_pad_center(pt) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };

    let sample_min_obstacle_dist = |p1: (f64, f64), p2: (f64, f64)| -> f64 {
        let mut min_d = f64::INFINITY;
        let samples = [0.15, 0.35, 0.50, 0.65, 0.85];
        for &t in &samples {
            let sx = p1.0 + t * (p2.0 - p1.0);
            let sy = p1.1 + t * (p2.1 - p1.1);
            if let Some((gx, gy)) = grid.world_to_grid(sx, sy) {
                if gx < grid.width && gy < grid.height {
                    let d = grid.dist_field[grid.idx(gx, gy, chain.layer)] as f64;
                    if d < min_d {
                        min_d = d;
                    }
                }
            }
        }
        min_d
    };

    let mut i = 0;
    while i + 3 < chain.points.len() {
        let p0 = chain.points[i];
        let p1 = chain.points[i + 1];
        let p2 = chain.points[i + 2];
        let p3 = chain.points[i + 3];

        if is_special_point(p1) || is_special_point(p2) {
            i += 1;
            continue;
        }

        let dx = p2.0 - p1.0;
        let dy = p2.1 - p1.1;
        let seg_len = dx.hypot(dy);

        // Only evaluate non-trivial axis-aligned segments
        if seg_len < 0.35 {
            i += 1;
            continue;
        }

        let is_horiz = dy.abs() < 1e-4 && dx.abs() > 1e-4;
        let is_vert = dx.abs() < 1e-4 && dy.abs() > 1e-4;

        if !is_horiz && !is_vert {
            i += 1;
            continue;
        }

        // Check if connecting legs p0->p1 and p2->p3 are perpendicular to p1->p2
        let p0_p1_perp = if is_horiz {
            (p1.0 - p0.0).abs() < 1e-4 && (p1.1 - p0.1).abs() > 1e-4
        } else {
            (p1.1 - p0.1).abs() < 1e-4 && (p1.0 - p0.0).abs() > 1e-4
        };
        let p2_p3_perp = if is_horiz {
            (p3.0 - p2.0).abs() < 1e-4 && (p3.1 - p2.1).abs() > 1e-4
        } else {
            (p3.1 - p2.1).abs() < 1e-4 && (p3.0 - p2.0).abs() > 1e-4
        };

        if !p0_p1_perp || !p2_p3_perp {
            i += 1;
            continue;
        }

        let normal = if is_horiz { (0.0, 1.0) } else { (1.0, 0.0) };
        let d_curr = sample_min_obstacle_dist(p1, p2);

        let candidate_offsets = [0.10, 0.05, -0.05, -0.10];
        let mut best_shift = None;
        let mut best_dist = d_curr;

        for &shift in &candidate_offsets {
            let p1_new = (p1.0 + normal.0 * shift, p1.1 + normal.1 * shift);
            let p2_new = (p2.0 + normal.0 * shift, p2.1 + normal.1 * shift);

            let l_in = (p1_new.0 - p0.0).hypot(p1_new.1 - p0.1);
            let l_out = (p3.0 - p2_new.0).hypot(p3.1 - p2_new.1);
            if l_in < 0.05 || l_out < 0.05 {
                continue;
            }

            if !is_segment_clear(board, Some(grid), p0, p1_new, chain.layer, chain.net_id, chain.width) {
                continue;
            }
            if !is_segment_clear(board, Some(grid), p1_new, p2_new, chain.layer, chain.net_id, chain.width) {
                continue;
            }
            if !is_segment_clear(board, Some(grid), p2_new, p3, chain.layer, chain.net_id, chain.width) {
                continue;
            }

            let d_cand = sample_min_obstacle_dist(p1_new, p2_new);
            // Accept shift if it strictly increases clearance to the closest obstacle by >= 0.03mm
            if d_cand > best_dist + 0.03 {
                best_dist = d_cand;
                best_shift = Some((p1_new, p2_new));
            }
        }

        if let Some((p1_new, p2_new)) = best_shift {
            chain.points[i + 1] = p1_new;
            chain.points[i + 2] = p2_new;
        }

        i += 1;
    }
}

pub fn smoothen_corners(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    if chain.points.len() < 3 {
        return;
    }
    let is_pad_center = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        })
    };
    let is_special_point = |pt: (f64, f64)| -> bool {
        is_pad_center(pt) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };

    let mut new_points = Vec::with_capacity(chain.points.len() * 2);
    new_points.push(chain.points[0]);

    for i in 1..chain.points.len() - 1 {
        let prev = *new_points.last().unwrap();
        let curr = chain.points[i];
        let next = chain.points[i + 1];

        if is_special_point(curr) {
            new_points.push(curr);
            continue;
        }

        let v1 = (curr.0 - prev.0, curr.1 - prev.1);
        let v2 = (next.0 - curr.0, next.1 - curr.1);
        let l1 = v1.0.hypot(v1.1);
        let l2 = v2.0.hypot(v2.1);

        if l1 < 1e-4 || l2 < 1e-4 {
            new_points.push(curr);
            continue;
        }

        let dot = (v1.0 * v2.0 + v1.1 * v2.1) / (l1 * l2);
        let cross = (v1.0 * v2.1 - v1.1 * v2.0) / (l1 * l2);

        let is_90_deg = dot.abs() < 0.05 && cross.abs() > 0.95;

        if is_90_deg {
            // Adaptive 45° chamfering: try largest setback first, falling back to tighter clearances
            let candidate_multipliers = [1.50, 1.25, 1.00, 0.75, 0.50, 0.35];
            let mut chamfered = false;

            for &mult in &candidate_multipliers {
                let setback = (mult * chain.width).min(0.45 * l1).min(0.45 * l2);
                if setback < 0.02 {
                    continue;
                }
                let m1 = (curr.0 - setback * (v1.0 / l1), curr.1 - setback * (v1.1 / l1));
                let m2 = (curr.0 + setback * (v2.0 / l2), curr.1 + setback * (v2.1 / l2));

                if is_octilinear_segment(m1, m2)
                    && is_segment_clear(board, Some(grid), m1, m2, chain.layer, chain.net_id, chain.width)
                {
                    new_points.push(m1);
                    new_points.push(m2);
                    chamfered = true;
                    break;
                }
            }

            if chamfered {
                continue;
            }
        }

        new_points.push(curr);
    }

    new_points.push(*chain.points.last().unwrap());
    chain.points = new_points;
}

pub fn smooth_45_miters(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    smoothen_corners(chain, board, grid);
}

pub fn reduce_vias(board: &mut Board, grid: &mut RoutingGrid) {
    let max_passes = 8;
    for _pass in 0..max_passes {
        let changed_pairs = eliminate_via_pairs(board, grid);
        let changed_single = consolidate_single_vias(board, grid);
        let changed_slide = slide_and_align_vias(board, grid);
        if !changed_pairs && !changed_single && !changed_slide {
            break;
        }
    }
    prune_dangling_tracks_and_vias(board);
}

pub fn reduce_vias_single_pass(board: &mut Board, grid: &RoutingGrid) -> bool {
    let p = eliminate_via_pairs(board, grid);
    let s = consolidate_single_vias(board, grid);
    let l = slide_and_align_vias(board, grid);
    p || s || l
}

/// Eliminates redundant layer-hopping via pairs (L_orig -> L_alt -> L_orig).
pub fn eliminate_via_pairs(board: &mut Board, grid: &RoutingGrid) -> bool {
    let mut any_changed = false;
    for _ in 0..20 {
        if !eliminate_via_pair_once(board, grid) {
            break;
        }
        any_changed = true;
    }
    any_changed
}

fn eliminate_via_pair_once(board: &mut Board, grid: &RoutingGrid) -> bool {
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        net_name == "GND"
            || net_name.starts_with("GND")
            || grid.plane_nets.values().any(|&pn| pn == nid)
    };

    // 1. Via pair layer-hopping elimination (L_orig -> L_alt -> L_orig)
    let mut net_ids: Vec<usize> = board
        .vias
        .iter()
        .filter(|v| !v.is_locked && !is_plane_net(v.net_id))
        .map(|v| v.net_id)
        .collect();
    net_ids.sort_unstable();
    net_ids.dedup();

    for net_id in net_ids {
        let vias_info: Vec<(usize, (f64, f64))> = board
            .vias
            .iter()
            .enumerate()
            .filter(|(_, v)| v.net_id == net_id && !v.is_locked)
            .map(|(i, v)| (i, v.position))
            .collect();

        if vias_info.len() < 2 {
            continue;
        }

        for &(v1_orig_idx, v1_pos) in &vias_info {
            let q1 = quant_point(v1_pos);

            let incident_v1: Vec<(usize, usize, (f64, f64))> = board
                .tracks
                .iter()
                .enumerate()
                .filter(|(_, t)| t.net_id == net_id && !t.is_locked)
                .filter_map(|(idx, t)| {
                    if quant_point(t.start) == q1 {
                        Some((idx, t.layer, t.end))
                    } else if quant_point(t.end) == q1 {
                        Some((idx, t.layer, t.start))
                    } else {
                        None
                    }
                })
                .collect();

            for &(t1_idx, l_alt, next_pt) in &incident_v1 {
                let l_orig_candidates: Vec<usize> = incident_v1
                    .iter()
                    .filter(|&&(idx, l, _)| idx != t1_idx && l != l_alt)
                    .map(|&(_, l, _)| l)
                    .collect();

                if l_orig_candidates.is_empty() {
                    continue;
                }
                let l_orig = l_orig_candidates[0];

                let mut chain_segs = vec![t1_idx];
                let mut chain_pts = vec![v1_pos, next_pt];
                let mut curr_pt = next_pt;
                let mut found_v2 = None;

                while chain_segs.len() < 32 {
                    let curr_q = quant_point(curr_pt);

                    if let Some(&(v2_idx, v2_pos)) = vias_info
                        .iter()
                        .find(|&&(idx, pos)| idx != v1_orig_idx && quant_point(pos) == curr_q)
                    {
                        found_v2 = Some((v2_idx, v2_pos));
                        break;
                    }

                    let next_track = board
                        .tracks
                        .iter()
                        .enumerate()
                        .filter(|&(idx, t)| {
                            t.net_id == net_id
                                && !t.is_locked
                                && t.layer == l_alt
                                && !chain_segs.contains(&idx)
                        })
                        .find_map(|(idx, t)| {
                            if quant_point(t.start) == curr_q {
                                Some((idx, t.end))
                            } else if quant_point(t.end) == curr_q {
                                Some((idx, t.start))
                            } else {
                                None
                            }
                        });

                    if let Some((next_seg_idx, nxt_p)) = next_track {
                        chain_segs.push(next_seg_idx);
                        chain_pts.push(nxt_p);
                        curr_pt = nxt_p;
                    } else {
                        break;
                    }
                }

                let Some((_v2_orig_idx, v2_pos)) = found_v2 else {
                    continue;
                };

                let q2 = quant_point(v2_pos);

                let v2_has_l_orig = board.tracks.iter().enumerate().any(|(idx, t)| {
                    t.net_id == net_id
                        && !t.is_locked
                        && t.layer == l_orig
                        && !chain_segs.contains(&idx)
                        && (quant_point(t.start) == q2 || quant_point(t.end) == q2)
                });

                if !v2_has_l_orig {
                    continue;
                }

                // Check if any SMD pad of this net along chain_pts is on l_hop.
                // If so, moving the chain to l_orig and deleting the vias would leave that SMD pad disconnected!
                let smd_hop_conflict = board.footprints.iter().any(|fp| {
                    fp.pads.iter().any(|p| {
                        if p.net_id != net_id || p.is_through_hole {
                            return false;
                        }
                        let (px, py) = fp.pad_absolute_position(p);
                        chain_pts.iter().any(|pt| (px - pt.0).hypot(py - pt.1) < 0.15) && fp.layer != l_orig
                    })
                });
                if smd_hop_conflict {
                    continue;
                }

                let track_width = board.tracks[t1_idx].width;
                let orig_len: f64 = chain_pts
                    .windows(2)
                    .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
                    .sum();

                let mut winner_path = None;

                // Priority 1: Direct segment on l_orig (strictly octilinear 0°, 45°, 90°)
                let direct_len = (v2_pos.0 - v1_pos.0).hypot(v2_pos.1 - v1_pos.1);
                if is_octilinear_segment(v1_pos, v2_pos)
                    && direct_len <= orig_len * 1.25 + 1e-4
                    && (direct_len - orig_len) <= 5.0 + 1e-4
                    && is_segment_clear(board, Some(grid), v1_pos, v2_pos, l_orig, net_id, track_width)
                {
                    winner_path = Some(vec![v1_pos, v2_pos]);
                }

                // Priority 2: Same path coordinates on l_orig
                if winner_path.is_none() {
                    let all_clear = chain_pts
                        .windows(2)
                        .all(|w| is_segment_clear(board, Some(grid), w[0], w[1], l_orig, net_id, track_width));
                    if all_clear {
                        winner_path = Some(chain_pts.clone());
                    }
                }

                // Priority 3: Canonical 45-degree detour on l_orig within length budget
                if winner_path.is_none() {
                    let dx = v2_pos.0 - v1_pos.0;
                    let dy = v2_pos.1 - v1_pos.1;
                    let s = dx.abs().min(dy.abs());
                    if s > 1e-4 {
                        let sx = s * dx.signum();
                        let sy = s * dy.signum();
                        let c_a = (v1_pos.0 + sx, v1_pos.1 + sy);
                        let c_b = (v2_pos.0 - sx, v2_pos.1 - sy);

                        for c in [c_a, c_b] {
                            let d1 = (c.0 - v1_pos.0).hypot(c.1 - v1_pos.1);
                            let d2 = (v2_pos.0 - c.0).hypot(v2_pos.1 - c.1);
                            let cand_len = d1 + d2;
                            let budget_ok = cand_len <= orig_len * 1.25 + 1e-4
                                && (cand_len - orig_len) <= 5.0 + 1e-4;
                            if budget_ok
                                && is_segment_clear(board, Some(grid), v1_pos, c, l_orig, net_id, track_width)
                                && is_segment_clear(board, Some(grid), c, v2_pos, l_orig, net_id, track_width)
                            {
                                winner_path = Some(vec![v1_pos, c, v2_pos]);
                                break;
                            }
                        }
                    }
                }

                if let Some(new_pts) = winner_path {
                    chain_segs.sort_unstable();
                    for &idx in chain_segs.iter().rev() {
                        board.tracks.remove(idx);
                    }

                    for w in new_pts.windows(2) {
                        if (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1) > 1e-5 {
                            board.tracks.push(TrackSegment {
                                start: w[0],
                                end: w[1],
                                width: track_width,
                                layer: l_orig,
                                net_id,
                                is_locked: false,
                            });
                        }
                    }

                    let mut to_remove = Vec::new();
                    for (i, v) in board.vias.iter().enumerate() {
                        if v.net_id == net_id && !v.is_locked {
                            let q = quant_point(v.position);
                            if q == q1 || q == q2 {
                                to_remove.push(i);
                            }
                        }
                    }
                    to_remove.sort_unstable();
                    for &idx in to_remove.iter().rev() {
                        board.vias.remove(idx);
                    }

                    return true;
                }
            }
        }
    }

    false
}

/// Automated BGA Escape Solver:
/// Analyzes regular 2D grid array BGA footprints, assigns pads to 4 quadrants (NE, NW, SE, SW),
/// and generates deterministic fanouts:
/// - Ring 0 (outer perimeter): direct surface escape tracks shooting outward without vias
/// - Ring 1 & inner rings: diagonal 45° dog-bone escape stubs placed at interstitial cell centers
///   connecting pad to dog-bone via and routing out through interstitial avenue corridors
/// - Via-in-Pad (VIPPO) fallback: centered microvia (0.30mm dia, 0.15mm drill) for pitch < 0.50mm
///   or where standard annular ring cannot clear adjacent pads.
pub fn generate_bga_dogbone_escapes(board: &mut Board, grid: &mut RoutingGrid) {
    let num_layers = board.copper_layers().len().max(2);

    for fp_idx in 0..board.footprints.len() {
        let fp = board.footprints[fp_idx].clone();
        let Some(bga_info) = fp.bga_grid_info() else {
            continue;
        };

        let pitch = bga_info.pitch_x.min(bga_info.pitch_y);
        let (cx, cy) = (bga_info.center_x, bga_info.center_y);

        // Distance from pad center to adjacent 4-pad interstitial cell center:
        let inter_dist = (bga_info.pitch_x * 0.5).hypot(bga_info.pitch_y * 0.5);

        // Standard dog-bone via parameters
        let std_via_dia = 0.45;
        let std_via_drill = 0.20;
        let std_via_r = std_via_dia * 0.5;

        // Microvia / VIPPO parameters
        let vippo_via_dia = 0.30;
        let vippo_via_drill = 0.15;

        // Check if VIPPO fallback is required:
        // Fallback condition: pitch < 0.50mm, OR standard dog-bone annular ring (0.45mm)
        // cannot clear adjacent pads with netclass clearance.
        let default_c = 0.12;
        let max_pad_r = fp
            .pads
            .iter()
            .filter(|p| !p.is_through_hole)
            .map(|p| p.width.max(p.height) * 0.5)
            .fold(0.20f64, f64::max);
        let vippo_required = pitch < 0.50 || (inter_dist < std_via_r + max_pad_r + default_c);

        let oy = |p: &Pad| if fp.layer != fp.orig_layer { -p.offset_y } else { p.offset_y };

        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if pad.is_through_hole {
                continue;
            }
            if pad.net_name.starts_with("unconnected") {
                continue;
            }

            let pad_net_id = if pad.net_id > 0 {
                pad.net_id
            } else if !pad.net_name.is_empty() {
                board.net_name_to_id.get(&pad.net_name).copied().unwrap_or(0)
            } else {
                0
            };

            let (px_world, py_world) = fp.pad_absolute_position(pad);

            // Avoid duplicate escapes if already present
            let already_routed = board.tracks.iter().any(|t| {
                (t.start.0 - px_world).hypot(t.start.1 - py_world) < 1e-4
                    || (t.end.0 - px_world).hypot(t.end.1 - py_world) < 1e-4
            }) || board.vias.iter().any(|v| {
                (v.position.0 - px_world).hypot(v.position.1 - py_world) < 1e-4
            });
            if already_routed {
                continue;
            }

            let lx = pad.offset_x;
            let ly = oy(pad);
            let dx = lx - cx;
            let dy = ly - cy;

            // Quadrant signs (+/- p/2 away from center)
            let sign_x = if dx >= 0.0 { 1.0 } else { -1.0 };
            let sign_y = if dy >= 0.0 { 1.0 } else { -1.0 };

            let ring = bga_info.pad_rings[pad_idx];
            let netclass = board.get_netclass_for_net(pad_net_id);
            let trk_w = (netclass.track_width).min(0.20);
            let stub_w = (netclass.track_width).min(0.15);

            if ring == 0 {
                // Ring 0 (outer perimeter): direct surface escape tracks shooting outward away from chip without vias
                let (col, row) = bga_info.pad_grid_indices[pad_idx].unwrap_or((0, 0));
                let outward_local = if col == 0 && row == 0 {
                    (-1.0 / 2.0f64.sqrt(), -1.0 / 2.0f64.sqrt())
                } else if col == bga_info.cols - 1 && row == 0 {
                    (1.0 / 2.0f64.sqrt(), -1.0 / 2.0f64.sqrt())
                } else if col == 0 && row == bga_info.rows - 1 {
                    (-1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt())
                } else if col == bga_info.cols - 1 && row == bga_info.rows - 1 {
                    (1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt())
                } else if col == 0 {
                    (-1.0, 0.0)
                } else if col == bga_info.cols - 1 {
                    (1.0, 0.0)
                } else if row == 0 {
                    (0.0, -1.0)
                } else if row == bga_info.rows - 1 {
                    (0.0, 1.0)
                } else {
                    let len = dx.hypot(dy).max(1e-6);
                    (dx / len, dy / len)
                };

                let world_dir = Point::new(outward_local.0, outward_local.1).rotate_ccw(fp.rotation);
                let escape_len = (pitch * 1.5).max(1.0);
                let tip = (px_world + world_dir.x * escape_len, py_world + world_dir.y * escape_len);

                let trk = TrackSegment {
                    start: (px_world, py_world),
                    end: tip,
                    width: trk_w,
                    layer: fp.layer,
                    net_id: pad_net_id,
                    is_locked: false,
                };
                board.tracks.push(trk.clone());
                grid.commit_track(&trk);
            } else if vippo_required {
                // Ring 1 & inner rings with VIPPO fallback:
                // Centered via-in-pad microvia (0.30mm diameter, 0.15mm drill)
                let via_pos = (px_world, py_world);
                let target_layer = if num_layers <= 2 {
                    1
                } else if ring == 1 {
                    1
                } else {
                    2 + (ring - 2) % (num_layers - 2)
                };

                let vippo_via = Via {
                    position: via_pos,
                    size: vippo_via_dia,
                    drill: vippo_via_drill,
                    net_id: pad_net_id,
                    is_locked: false,
                };
                board.vias.push(vippo_via.clone());
                grid.commit_via(&vippo_via);

                // Escape track radiating out from microvia on target layer
                let outward_local = (sign_x / 2.0f64.sqrt(), sign_y / 2.0f64.sqrt());
                let world_dir = Point::new(outward_local.0, outward_local.1).rotate_ccw(fp.rotation);
                let escape_len = (pitch * 2.0).max(1.0);
                let tip = (px_world + world_dir.x * escape_len, py_world + world_dir.y * escape_len);

                let chan_trk = TrackSegment {
                    start: (px_world, py_world),
                    end: tip,
                    width: stub_w,
                    layer: target_layer,
                    net_id: pad_net_id,
                    is_locked: false,
                };
                board.tracks.push(chan_trk.clone());
                grid.commit_track(&chan_trk);
            } else {
                // Ring 1 & inner rings: standard 45° dog-bone escape stubs
                // Dog-bone via placed at diagonal center of adjacent 4-pad interstitial cell (dx = +/- p/2, dy = +/- p/2)
                let delta_x = sign_x * (bga_info.pitch_x * 0.5);
                let delta_y = sign_y * (bga_info.pitch_y * 0.5);
                let via_loc_x = lx + delta_x;
                let via_loc_y = ly + delta_y;

                let world_via_pt = Point::new(delta_x, delta_y).rotate_ccw(fp.rotation);
                let world_via = (px_world + world_via_pt.x, py_world + world_via_pt.y);

                // Stub track connecting pad center to dog-bone via
                let stub_trk = TrackSegment {
                    start: (px_world, py_world),
                    end: world_via,
                    width: stub_w,
                    layer: fp.layer,
                    net_id: pad_net_id,
                    is_locked: false,
                };

                let target_layer = if num_layers <= 2 {
                    1
                } else if ring == 1 {
                    1
                } else {
                    2 + (ring - 2) % (num_layers - 2)
                };

                let dogbone_via = Via {
                    position: world_via,
                    size: std_via_dia,
                    drill: std_via_drill,
                    net_id: pad_net_id,
                    is_locked: false,
                };

                // Channel escape corridor: radiating out through interstitial avenue
                let (target_local_x, target_local_y) = if dx.abs() >= dy.abs() {
                    let end_x = if sign_x > 0.0 {
                        bga_info.max_x + pitch * 1.5
                    } else {
                        bga_info.min_x - pitch * 1.5
                    };
                    (end_x, via_loc_y)
                } else {
                    let end_y = if sign_y > 0.0 {
                        bga_info.max_y + pitch * 1.5
                    } else {
                        bga_info.min_y - pitch * 1.5
                    };
                    (via_loc_x, end_y)
                };

                let corridor_disp = Point::new(target_local_x - lx, target_local_y - ly).rotate_ccw(fp.rotation);
                let world_corridor_end = (px_world + corridor_disp.x, py_world + corridor_disp.y);

                let chan_trk = TrackSegment {
                    start: world_via,
                    end: world_corridor_end,
                    width: stub_w,
                    layer: target_layer,
                    net_id: pad_net_id,
                    is_locked: false,
                };

                board.tracks.push(stub_trk.clone());
                grid.commit_track(&stub_trk);

                board.vias.push(dogbone_via.clone());
                grid.commit_via(&dogbone_via);

                board.tracks.push(chan_trk.clone());
                grid.commit_track(&chan_trk);
            }
        }
    }

    grid.update_distance_fields();
}

/// Identifies connections with a single via transitioning between layers,
/// testing whether the alternate layer segment chain can be moved/projected onto
/// the primary layer without DRC violations, eliminating the solitary via.
pub fn consolidate_single_vias(board: &mut Board, grid: &RoutingGrid) -> bool {
    let mut any_changed = false;
    for _ in 0..20 {
        if !consolidate_single_via_once(board, grid) {
            break;
        }
        any_changed = true;
    }
    any_changed
}

fn consolidate_single_via_once(board: &mut Board, grid: &RoutingGrid) -> bool {
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        net_name == "GND"
            || net_name.starts_with("GND")
            || grid.plane_nets.values().any(|&pn| pn == nid)
    };

    let vias_info: Vec<(usize, (f64, f64), usize)> = board
        .vias
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_locked && !is_plane_net(v.net_id))
        .map(|(i, v)| (i, v.position, v.net_id))
        .collect();

    for (v_idx, v_pos, net_id) in vias_info {
        let q_v = quant_point(v_pos);
        let clearance = board.get_netclass_for_net(net_id).clearance;

        let incident: Vec<(usize, usize, (f64, f64), f64)> = board
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.net_id == net_id && !t.is_locked)
            .filter_map(|(idx, t)| {
                if quant_point(t.start) == q_v {
                    Some((idx, t.layer, t.end, t.width))
                } else if quant_point(t.end) == q_v {
                    Some((idx, t.layer, t.start, t.width))
                } else {
                    None
                }
            })
            .collect();

        let mut layers: Vec<usize> = incident.iter().map(|&(_, l, _, _)| l).collect();
        layers.sort_unstable();
        layers.dedup();
        if layers.len() < 2 {
            continue;
        }

        for &l_prim in &layers {
            let alt_layers: Vec<usize> = layers.iter().copied().filter(|&l| l != l_prim).collect();
            for &l_alt in &alt_layers {
                let alt_incidents: Vec<(usize, (f64, f64), f64)> = incident
                    .iter()
                    .filter(|&&(_, l, _, _)| l == l_alt)
                    .map(|&(idx, _, pt, w)| (idx, pt, w))
                    .collect();

                if alt_incidents.len() != 1 {
                    continue;
                }

                let (t1_idx, next_pt, track_width) = alt_incidents[0];
                let mut chain_segs = vec![t1_idx];
                let mut chain_pts = vec![v_pos, next_pt];
                let mut curr_pt = next_pt;
                let mut hit_another_via = false;

                while chain_segs.len() < 32 {
                    let curr_q = quant_point(curr_pt);
                    if board.vias.iter().enumerate().any(|(i, other_v)| {
                        i != v_idx && other_v.net_id == net_id && quant_point(other_v.position) == curr_q
                    }) {
                        hit_another_via = true;
                        break;
                    }

                    let next_track = board
                        .tracks
                        .iter()
                        .enumerate()
                        .filter(|&(idx, t)| {
                            t.net_id == net_id
                                && !t.is_locked
                                && t.layer == l_alt
                                && !chain_segs.contains(&idx)
                        })
                        .find_map(|(idx, t)| {
                            if quant_point(t.start) == curr_q {
                                Some((idx, t.end))
                            } else if quant_point(t.end) == curr_q {
                                Some((idx, t.start))
                            } else {
                                None
                            }
                        });

                    if let Some((next_seg_idx, nxt_p)) = next_track {
                        chain_segs.push(next_seg_idx);
                        chain_pts.push(nxt_p);
                        curr_pt = nxt_p;
                    } else {
                        break;
                    }
                }

                if hit_another_via {
                    continue;
                }

                // Check if any SMD pad of this net along chain_pts (or at v_pos / curr_pt) belongs to a layer other than l_prim.
                // If so, moving the chain to l_prim and deleting the via would leave that SMD pad disconnected!
                let smd_layer_conflict = board.footprints.iter().any(|fp| {
                    fp.pads.iter().any(|p| {
                        if p.net_id != net_id || p.is_through_hole {
                            return false;
                        }
                        let (px, py) = fp.pad_absolute_position(p);
                        chain_pts.iter().any(|pt| (px - pt.0).hypot(py - pt.1) < 0.15) && fp.layer != l_prim
                    })
                });
                if smd_layer_conflict {
                    continue;
                }

                // Check if curr_pt connects to a valid anchor on l_prim (either an SMD/TH pad on l_prim, or an existing track on l_prim)
                let pad_on_l_prim = board.footprints.iter().any(|fp| {
                    fp.pads.iter().any(|p| {
                        if p.net_id != net_id {
                            return false;
                        }
                        let (px, py) = fp.pad_absolute_position(p);
                        if (px - curr_pt.0).hypot(py - curr_pt.1) < 0.15 {
                            p.is_through_hole || fp.layer == l_prim
                        } else {
                            false
                        }
                    })
                });

                let curr_q = quant_point(curr_pt);
                let connects_to_track_on_l_prim = board.tracks.iter().enumerate().any(|(idx, t)| {
                    t.net_id == net_id
                        && t.layer == l_prim
                        && !t.is_locked
                        && !chain_segs.contains(&idx)
                        && (quant_point(t.start) == curr_q || quant_point(t.end) == curr_q)
                });

                if !pad_on_l_prim && !connects_to_track_on_l_prim {
                    continue;
                }

                let orig_len: f64 = chain_pts
                    .windows(2)
                    .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
                    .sum();

                let mut winner_path = None;

                // Priority 1: Direct segment on l_prim (strictly octilinear 0°, 45°, 90°)
                let direct_len = (curr_pt.0 - v_pos.0).hypot(curr_pt.1 - v_pos.1);
                if is_octilinear_segment(v_pos, curr_pt)
                    && direct_len <= orig_len * 1.25 + 1e-4
                    && (direct_len - orig_len) <= 5.0 + 1e-4
                    && is_segment_clear(board, Some(grid), v_pos, curr_pt, l_prim, net_id, track_width)
                    && is_track_segment_legal(board, v_pos, curr_pt, track_width, l_prim, net_id, clearance)
                {
                    winner_path = Some(vec![v_pos, curr_pt]);
                }

                // Priority 2: Same path coordinates on l_prim
                if winner_path.is_none() {
                    let all_clear = chain_pts.windows(2).all(|w| {
                        is_segment_clear(board, Some(grid), w[0], w[1], l_prim, net_id, track_width)
                            && is_track_segment_legal(board, w[0], w[1], track_width, l_prim, net_id, clearance)
                    });
                    if all_clear {
                        winner_path = Some(chain_pts.clone());
                    }
                }

                // Priority 3: Canonical 45-degree detour on l_prim
                if winner_path.is_none() {
                    let dx = curr_pt.0 - v_pos.0;
                    let dy = curr_pt.1 - v_pos.1;
                    let s = dx.abs().min(dy.abs());
                    if s > 1e-4 {
                        let sx = s * dx.signum();
                        let sy = s * dy.signum();
                        let c_a = (v_pos.0 + sx, v_pos.1 + sy);
                        let c_b = (curr_pt.0 - sx, curr_pt.1 - sy);
                        for c in [c_a, c_b] {
                            let d1 = (c.0 - v_pos.0).hypot(c.1 - v_pos.1);
                            let d2 = (curr_pt.0 - c.0).hypot(curr_pt.1 - c.1);
                            let cand_len = d1 + d2;
                            let budget_ok = cand_len <= orig_len * 1.25 + 1e-4
                                && (cand_len - orig_len) <= 5.0 + 1e-4;
                            if budget_ok
                                && is_segment_clear(board, Some(grid), v_pos, c, l_prim, net_id, track_width)
                                && is_segment_clear(board, Some(grid), c, curr_pt, l_prim, net_id, track_width)
                                && is_track_segment_legal(board, v_pos, c, track_width, l_prim, net_id, clearance)
                                && is_track_segment_legal(board, c, curr_pt, track_width, l_prim, net_id, clearance)
                            {
                                winner_path = Some(vec![v_pos, c, curr_pt]);
                                break;
                            }
                        }
                    }
                }

                if let Some(new_pts) = winner_path {
                    chain_segs.sort_unstable();
                    for &idx in chain_segs.iter().rev() {
                        board.tracks.remove(idx);
                    }

                    for w in new_pts.windows(2) {
                        if (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1) > 1e-5 {
                            board.tracks.push(TrackSegment {
                                start: w[0],
                                end: w[1],
                                width: track_width,
                                layer: l_prim,
                                net_id,
                                is_locked: false,
                            });
                        }
                    }

                    if let Some(pos) = board.vias.iter().position(|v| {
                        v.net_id == net_id && !v.is_locked && quant_point(v.position) == q_v
                    }) {
                        board.vias.remove(pos);
                    }

                    return true;
                }
            }
        }
    }

    false
}

/// Slides and aligns signal vias along incident track vectors to eliminate micro-jogs,
/// doglegs, reduce wirelength, and snap close vias to through-hole pads.
pub fn slide_and_align_vias(board: &mut Board, grid: &RoutingGrid) -> bool {
    let mut any_changed = false;
    for _ in 0..20 {
        if !slide_and_align_via_once(board, grid) {
            break;
        }
        any_changed = true;
    }
    any_changed
}

fn slide_and_align_via_once(board: &mut Board, grid: &RoutingGrid) -> bool {
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        net_name == "GND"
            || net_name.starts_with("GND")
            || grid.plane_nets.values().any(|&pn| pn == nid)
    };

    let num_vias = board.vias.len();
    for v_idx in 0..num_vias {
        let (v_pos, net_id, via_size, is_locked) = {
            let v = &board.vias[v_idx];
            (v.position, v.net_id, v.size, v.is_locked)
        };
        if is_locked || is_plane_net(net_id) {
            continue;
        }

        let clearance = board.get_netclass_for_net(net_id).clearance;

        // Incident tracks
        let incident: Vec<(usize, usize, (f64, f64), f64)> = board
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.net_id == net_id && !t.is_locked)
            .filter_map(|(idx, t)| {
                if (t.start.0 - v_pos.0).hypot(t.start.1 - v_pos.1) < 1e-4 {
                    Some((idx, t.layer, t.end, t.width))
                } else if (t.end.0 - v_pos.0).hypot(t.end.1 - v_pos.1) < 1e-4 {
                    Some((idx, t.layer, t.start, t.width))
                } else {
                    None
                }
            })
            .collect();

        if incident.is_empty() {
            continue;
        }

        // 1. Pad Snapping: check if close to a same-net through-hole pad (< 0.50mm)
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id == net_id && pad.is_through_hole {
                    let (px, py) = fp.pad_absolute_position(pad);
                    let d = (v_pos.0 - px).hypot(v_pos.1 - py);
                    if d > 1e-4 && d < 0.50 {
                        let all_clear = incident.iter().all(|&(_, l, other, w)| {
                            is_octilinear_segment((px, py), other)
                                && is_track_segment_legal(board, (px, py), other, w, l, net_id, clearance)
                                && is_segment_clear(board, Some(grid), (px, py), other, l, net_id, w)
                        });
                        if all_clear {
                            for &(t_idx, _, _, _) in &incident {
                                let t = &mut board.tracks[t_idx];
                                if (t.start.0 - v_pos.0).hypot(t.start.1 - v_pos.1) < 1e-4 {
                                    t.start = (px, py);
                                } else {
                                    t.end = (px, py);
                                }
                            }
                            board.tracks.retain(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1) > 1e-4);
                            board.vias.remove(v_idx);
                            return true;
                        }
                    }
                }
            }
        }

        // 2. Collinear & Jog Elimination:
        // If via is connected to a track that has a dogleg or micro-jog (< 1.0mm) before turning
        for &(t_idx, layer, other_pt, _) in &incident {
            let seg_len = (other_pt.0 - v_pos.0).hypot(other_pt.1 - v_pos.1);
            if seg_len < 1.0 && seg_len > 1e-4 {
                let is_pad = board.footprints.iter().any(|fp| {
                    fp.pads.iter().any(|p| {
                        let (px, py) = fp.pad_absolute_position(p);
                        (px - other_pt.0).hypot(py - other_pt.1) < 1e-3
                    })
                });
                if !is_pad {
                    let has_connecting_track = board.tracks.iter().enumerate().any(|(i, t)| {
                        i != t_idx
                            && t.net_id == net_id
                            && t.layer == layer
                            && !t.is_locked
                            && ((t.start.0 - other_pt.0).hypot(t.start.1 - other_pt.1) < 1e-4
                                || (t.end.0 - other_pt.0).hypot(t.end.1 - other_pt.1) < 1e-4)
                    });

                    if has_connecting_track {
                        if is_via_location_legal_ignoring(board, other_pt, via_size, net_id, clearance, Some(v_pos)) {
                            let other_tracks_ok = incident.iter().all(|&(ot_idx, ot_layer, ot_other, ot_w)| {
                                if ot_idx == t_idx {
                                    true
                                } else if (ot_other.0 - other_pt.0).hypot(ot_other.1 - other_pt.1) < 1e-4 {
                                    true
                                } else {
                                    is_octilinear_segment(other_pt, ot_other)
                                        && is_track_segment_legal(board, other_pt, ot_other, ot_w, ot_layer, net_id, clearance)
                                        && is_segment_clear(board, Some(grid), other_pt, ot_other, ot_layer, net_id, ot_w)
                                }
                            });

                            if other_tracks_ok {
                                board.vias[v_idx].position = other_pt;
                                for &(ot_idx, _, _, _) in &incident {
                                    let t = &mut board.tracks[ot_idx];
                                    if (t.start.0 - v_pos.0).hypot(t.start.1 - v_pos.1) < 1e-4 {
                                        t.start = other_pt;
                                    } else {
                                        t.end = other_pt;
                                    }
                                }
                                board.tracks.retain(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1) > 1e-4);
                                return true;
                            }
                        }
                    }
                }
            }
        }

        // 3. Vector Sliding:
        // Test candidate positions along vectors of connected tracks
        let orig_tot_len: f64 = incident.iter().map(|&(_, _, other, _)| (other.0 - v_pos.0).hypot(other.1 - v_pos.1)).sum();
        let mut best_cand: Option<((f64, f64), f64)> = None;

        for &(_, _, other_pt, _) in &incident {
            let dx = other_pt.0 - v_pos.0;
            let dy = other_pt.1 - v_pos.1;
            let dist = dx.hypot(dy);
            if dist < 0.15 {
                continue;
            }
            let ux = dx / dist;
            let uy = dy / dist;

            let steps = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.50, 2.00];
            for &s in &steps {
                if s >= dist - 0.05 {
                    break;
                }
                let cand = (v_pos.0 + s * ux, v_pos.1 + s * uy);
                if !is_via_location_legal_ignoring(board, cand, via_size, net_id, clearance, Some(v_pos)) {
                    continue;
                }

                let all_ok = incident.iter().all(|&(_, l, other, w)| {
                    if (other.0 - cand.0).hypot(other.1 - cand.1) < 1e-4 {
                        true
                    } else {
                        is_octilinear_segment(cand, other)
                            && is_track_segment_legal(board, cand, other, w, l, net_id, clearance)
                            && is_segment_clear(board, Some(grid), cand, other, l, net_id, w)
                    }
                });

                if all_ok {
                    let cand_len: f64 = incident.iter().map(|&(_, _, other, _)| (other.0 - cand.0).hypot(other.1 - cand.1)).sum();
                    if cand_len < orig_tot_len - 0.05 {
                        let min_len = best_cand.map(|(_, l)| l).unwrap_or(orig_tot_len - 0.05);
                        if cand_len < min_len - 1e-4 {
                            best_cand = Some((cand, cand_len));
                        }
                    }
                }
            }
        }

        if let Some((best_pos, _)) = best_cand {
            board.vias[v_idx].position = best_pos;
            for &(t_idx, _, _, _) in &incident {
                let t = &mut board.tracks[t_idx];
                if (t.start.0 - v_pos.0).hypot(t.start.1 - v_pos.1) < 1e-4 {
                    t.start = best_pos;
                } else {
                    t.end = best_pos;
                }
            }
            board.tracks.retain(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1) > 1e-4);
            return true;
        }
    }

    false
}

/// Decomposes any non-octilinear segment in a track chain into standard 45° doglegs.
pub fn enforce_45_grid(chain: &mut TrackChain) {
    if chain.points.len() < 2 {
        return;
    }
    let mut new_points = Vec::with_capacity(chain.points.len() * 2);
    new_points.push(chain.points[0]);

    for k in 0..chain.points.len() - 1 {
        let p0 = *new_points.last().unwrap();
        let p1 = chain.points[k + 1];
        if (p0.0 - p1.0).hypot(p0.1 - p1.1) < 1e-5 {
            continue;
        }
        if is_octilinear_segment(p0, p1) {
            new_points.push(p1);
        } else {
            let dogleg = compute_45_dogleg(p0, p1, false);
            for &dp in &dogleg[1..] {
                if (dp.0 - new_points.last().unwrap().0).hypot(dp.1 - new_points.last().unwrap().1) > 1e-5 {
                    new_points.push(dp);
                }
            }
        }
    }

    chain.points = new_points;
    merge_collinear_points(&mut chain.points);
}

pub fn gloss_tracks(board: &mut Board, grid: &RoutingGrid) {
    gloss_tracks_with_mode(board, grid, false);
}

pub fn gloss_tracks_with_mode(board: &mut Board, grid: &RoutingGrid, any_angle_mode: bool) {
    if any_angle_mode {
        gloss_tracks_standard(board, grid);
        relax_homotopic_rubber_bands(board);
        apply_any_angle_tangent_arcs(board);
    } else {
        gloss_tracks_standard(board, grid);
    }
}

pub fn gloss_tracks_standard(board: &mut Board, grid: &RoutingGrid) {
    let mut net_ids: Vec<usize> = board
        .tracks
        .iter()
        .filter(|t| !t.is_locked)
        .map(|t| t.net_id)
        .collect();
    net_ids.sort_unstable();
    net_ids.dedup();

    for net_id in net_ids {
        let net_tracks: Vec<TrackSegment> = board
            .tracks
            .iter()
            .filter(|t| t.net_id == net_id && !t.is_locked)
            .cloned()
            .collect();
        if net_tracks.is_empty() {
            continue;
        }

        board.tracks.retain(|t| t.net_id != net_id || t.is_locked);

        let mut chains = extract_track_chains(&net_tracks, board);
        for chain in &mut chains {
            enforce_45_grid(chain);
            merge_collinear_points(&mut chain.points);
            avoid_acid_traps(chain, board, grid);
            merge_collinear_points(&mut chain.points);
            rubber_band_straighten(chain, board, grid);
            eliminate_stair_steps(chain, board, grid);
            reduce_jogs(chain, board, grid);
            reduce_corners(chain, board, grid);
            merge_collinear_points(&mut chain.points);
            reposition_lines(chain, board, grid);
            merge_collinear_points(&mut chain.points);
            smoothen_corners(chain, board, grid);
            avoid_acid_traps(chain, board, grid);
            enforce_45_grid(chain);
            merge_collinear_points(&mut chain.points);
        }

        for chain in chains {
            for k in 0..chain.points.len().saturating_sub(1) {
                let p1 = chain.points[k];
                let p2 = chain.points[k + 1];
                if (p1.0 - p2.0).hypot(p1.1 - p2.1) > 1e-5 {
                    board.tracks.push(TrackSegment {
                        start: p1,
                        end: p2,
                        width: chain.width,
                        layer: chain.layer,
                        net_id: chain.net_id,
                        is_locked: false,
                    });
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ObstacleInfo {
    pub center: (f64, f64),
    pub safe_radius: f64,
}

pub fn point_in_polygon(pt: (f64, f64), poly: &[(f64, f64)]) -> bool {
    let mut inside = false;
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        let intersect = ((yi > pt.1) != (yj > pt.1))
            && (pt.0 < (xj - xi) * (pt.1 - yi) / (yj - yi + 1e-12) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

pub fn collect_obstacles_for_net(board: &Board, layer: usize, net_id: usize, track_width: f64) -> Vec<ObstacleInfo> {
    let net_clr = board.get_netclass_for_net(net_id).clearance;
    let half_w = track_width * 0.5;
    let mut obstacles = Vec::new();

    // 1. Foreign Pads
    for fp in &board.footprints {
        let pad_layer = fp.layer;
        for pad in &fp.pads {
            if pad.net_id != net_id && (pad.is_through_hole || pad_layer == layer) {
                let (px, py) = fp.pad_absolute_position(pad);
                let pad_clr = board.get_netclass_for_net(pad.net_id).clearance;
                let req_clr = net_clr.max(pad_clr);
                let pad_r = (pad.width * pad.width + pad.height * pad.height).sqrt() * 0.5;
                obstacles.push(ObstacleInfo {
                    center: (px, py),
                    safe_radius: pad_r + req_clr + half_w,
                });
            }
        }
    }

    // 2. Foreign Vias
    for via in &board.vias {
        if via.net_id != net_id {
            let via_clr = board.get_netclass_for_net(via.net_id).clearance;
            let req_clr = net_clr.max(via_clr);
            let via_r = via.size * 0.5;
            obstacles.push(ObstacleInfo {
                center: via.position,
                safe_radius: via_r + req_clr + half_w,
            });
        }
    }

    obstacles
}

pub fn relax_homotopic_rubber_bands(board: &mut Board) {
    let mut net_ids: Vec<usize> = board
        .tracks
        .iter()
        .filter(|t| !t.is_locked)
        .map(|t| t.net_id)
        .collect();
    net_ids.sort_unstable();
    net_ids.dedup();

    for net_id in net_ids {
        let net_tracks: Vec<TrackSegment> = board
            .tracks
            .iter()
            .filter(|t| t.net_id == net_id && !t.is_locked)
            .cloned()
            .collect();
        if net_tracks.is_empty() {
            continue;
        }

        board.tracks.retain(|t| t.net_id != net_id || t.is_locked);

        let mut chains = extract_track_chains(&net_tracks, board);
        for chain in &mut chains {
            relax_chain_homotopic(chain, board);
        }

        for chain in chains {
            for k in 0..chain.points.len().saturating_sub(1) {
                let p1 = chain.points[k];
                let p2 = chain.points[k + 1];
                if (p1.0 - p2.0).hypot(p1.1 - p2.1) > 1e-5 {
                    board.tracks.push(TrackSegment {
                        start: p1,
                        end: p2,
                        width: chain.width,
                        layer: chain.layer,
                        net_id: chain.net_id,
                        is_locked: false,
                    });
                }
            }
        }
    }
}

pub fn relax_chain_homotopic(chain: &mut TrackChain, board: &Board) {
    if chain.points.len() < 3 {
        return;
    }

    let obstacles = collect_obstacles_for_net(board, chain.layer, chain.net_id, chain.width);

    // Iterative rubber-band string pulling passes
    for _pass in 0..6 {
        merge_collinear_points(&mut chain.points);
        if chain.points.len() < 3 {
            break;
        }

        // 1. Homotopic Triangle Shortcutting
        let mut i = 1;
        while i + 1 < chain.points.len() {
            let p_prev = chain.points[i - 1];
            let p_curr = chain.points[i];
            let p_next = chain.points[i + 1];

            let tri = [p_prev, p_curr, p_next];
            let homotopy_ok = obstacles.iter().all(|obs| !point_in_polygon(obs.center, &tri));
            if homotopy_ok && is_segment_clear(board, None, p_prev, p_next, chain.layer, chain.net_id, chain.width) {
                chain.points.remove(i);
            } else {
                i += 1;
            }
        }

        // 2. Multi-point Chord Shortcutting
        let mut start_idx = 0;
        while start_idx < chain.points.len() {
            let mut best_j = None;
            for j in ((start_idx + 2)..chain.points.len()).rev() {
                let p_start = chain.points[start_idx];
                let p_end = chain.points[j];
                let poly = &chain.points[start_idx..=j];
                let homotopy_ok = obstacles.iter().all(|obs| !point_in_polygon(obs.center, poly));
                if homotopy_ok && is_segment_clear(board, None, p_start, p_end, chain.layer, chain.net_id, chain.width) {
                    best_j = Some(j);
                    break;
                }
            }
            if let Some(j) = best_j {
                chain.points.drain((start_idx + 1)..j);
            }
            start_idx += 1;
        }

        // 3. Obstacle Tangent Apex Contraction
        for k in 1..chain.points.len().saturating_sub(1) {
            let p_prev = chain.points[k - 1];
            let p_curr = chain.points[k];
            let p_next = chain.points[k + 1];

            let blocking_obs = obstacles
                .iter()
                .filter(|obs| dist_point_to_segment(obs.center, p_prev, p_next) < obs.safe_radius)
                .min_by(|a, b| {
                    let da = (a.center.0 - p_curr.0).hypot(a.center.1 - p_curr.1);
                    let db = (b.center.0 - p_curr.0).hypot(b.center.1 - p_curr.1);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some(obs) = blocking_obs {
                let vx = p_curr.0 - obs.center.0;
                let vy = p_curr.1 - obs.center.1;
                let v_len = vx.hypot(vy);
                if v_len > 1e-4 {
                    let target_pt = (
                        obs.center.0 + (vx / v_len) * obs.safe_radius,
                        obs.center.1 + (vy / v_len) * obs.safe_radius,
                    );
                    let tri1 = [p_prev, p_curr, target_pt];
                    let tri2 = [p_curr, p_next, target_pt];
                    let h1 = obstacles.iter().all(|o| !point_in_polygon(o.center, &tri1));
                    let h2 = obstacles.iter().all(|o| !point_in_polygon(o.center, &tri2));
                    if h1 && h2
                        && is_segment_clear(board, None, p_prev, target_pt, chain.layer, chain.net_id, chain.width)
                        && is_segment_clear(board, None, target_pt, p_next, chain.layer, chain.net_id, chain.width)
                    {
                        chain.points[k] = target_pt;
                    }
                }
            }
        }

        // 4. Binary-search Taut Relaxation towards Midpoint
        for k in 1..chain.points.len().saturating_sub(1) {
            let p_prev = chain.points[k - 1];
            let p_curr = chain.points[k];
            let p_next = chain.points[k + 1];
            let target = ((p_prev.0 + p_next.0) * 0.5, (p_prev.1 + p_next.1) * 0.5);

            let mut low = 0.0;
            let mut high = 1.0;
            let mut best_t = 0.0;

            for _ in 0..8 {
                let mid = (low + high) * 0.5;
                let cand = (
                    p_curr.0 + mid * (target.0 - p_curr.0),
                    p_curr.1 + mid * (target.1 - p_curr.1),
                );
                let tri1 = [p_prev, p_curr, cand];
                let tri2 = [p_curr, p_next, cand];
                let h1 = obstacles.iter().all(|o| !point_in_polygon(o.center, &tri1));
                let h2 = obstacles.iter().all(|o| !point_in_polygon(o.center, &tri2));
                if h1 && h2
                    && is_segment_clear(board, None, p_prev, cand, chain.layer, chain.net_id, chain.width)
                    && is_segment_clear(board, None, cand, p_next, chain.layer, chain.net_id, chain.width)
                {
                    best_t = mid;
                    low = mid;
                } else {
                    high = mid;
                }
            }

            if best_t > 0.05 {
                chain.points[k] = (
                    p_curr.0 + best_t * (target.0 - p_curr.0),
                    p_curr.1 + best_t * (target.1 - p_curr.1),
                );
            }
        }
    }

    merge_collinear_points(&mut chain.points);
}

pub fn is_arc_clear(board: &Board, arc: &ArcSegment) -> bool {
    let clearance = board.get_netclass_for_net(arc.net_id).clearance;
    let half_w = arc.width * 0.5;

    // 1. Foreign Pads
    for fp in &board.footprints {
        let pad_layer = fp.layer;
        for pad in &fp.pads {
            if pad.net_id != arc.net_id && (pad.is_through_hole || pad_layer == arc.layer) {
                let (px, py) = fp.pad_absolute_position(pad);
                let pad_clr = board.get_netclass_for_net(pad.net_id).clearance;
                let req_clr = clearance.max(pad_clr);
                let required_dist = half_w + req_clr;

                let max_dim = pad.width.max(pad.height);
                let d_center = arc.distance_to_point((px, py));
                if d_center < required_dist + max_dim * 0.5 - 1e-4 {
                    let segs = arc.to_segments(8);
                    for s in &segs {
                        if dist_segment_to_pad(s.start, s.end, fp, pad) < required_dist - 1e-4 {
                            return false;
                        }
                    }
                }
            }
        }
    }

    // 2. Foreign Vias
    for via in &board.vias {
        if via.net_id != arc.net_id {
            let via_clr = board.get_netclass_for_net(via.net_id).clearance;
            let req_clr = clearance.max(via_clr);
            let required_dist = half_w + via.size * 0.5 + req_clr;
            if arc.distance_to_point(via.position) < required_dist - 1e-4 {
                return false;
            }
        }
    }

    // 3. Keepouts
    for ko in &board.keepouts {
        if ko.layers.contains(&arc.layer) {
            let segs = arc.to_segments(8);
            for s in &segs {
                if !is_segment_clear(board, None, s.start, s.end, arc.layer, arc.net_id, arc.width) {
                    return false;
                }
            }
        }
    }

    // 4. Foreign Tracks
    for trk in &board.tracks {
        if trk.net_id != arc.net_id && trk.layer == arc.layer {
            let trk_clr = board.get_netclass_for_net(trk.net_id).clearance;
            let req_clr = clearance.max(trk_clr);
            let required_dist = half_w + (trk.width * 0.5) + req_clr;
            let segs = arc.to_segments(8);
            for s in &segs {
                if dist_segment_to_segment(s.start, s.end, trk.start, trk.end) < required_dist - 1e-4 {
                    return false;
                }
            }
        }
    }

    true
}

pub fn apply_any_angle_tangent_arcs(board: &mut Board) {
    let mut net_ids: Vec<usize> = board
        .tracks
        .iter()
        .filter(|t| !t.is_locked)
        .map(|t| t.net_id)
        .collect();
    net_ids.sort_unstable();
    net_ids.dedup();

    for net_id in net_ids {
        let net_tracks: Vec<TrackSegment> = board
            .tracks
            .iter()
            .filter(|t| t.net_id == net_id && !t.is_locked)
            .cloned()
            .collect();
        if net_tracks.is_empty() {
            continue;
        }

        board.tracks.retain(|t| t.net_id != net_id || t.is_locked);

        let chains = extract_track_chains(&net_tracks, board);
        for chain in chains {
            let n = chain.points.len();
            if n < 3 {
                for k in 0..n.saturating_sub(1) {
                    board.tracks.push(TrackSegment {
                        start: chain.points[k],
                        end: chain.points[k + 1],
                        width: chain.width,
                        layer: chain.layer,
                        net_id: chain.net_id,
                        is_locked: false,
                    });
                }
                continue;
            }

            let mut corner_arcs: Vec<Option<((f64, f64), (f64, f64), ArcSegment)>> = vec![None; n];

            for i in 1..n - 1 {
                let p_prev = chain.points[i - 1];
                let p_curr = chain.points[i];
                let p_next = chain.points[i + 1];

                let v1 = (p_prev.0 - p_curr.0, p_prev.1 - p_curr.1);
                let v2 = (p_next.0 - p_curr.0, p_next.1 - p_curr.1);
                let l_a = v1.0.hypot(v1.1);
                let l_b = v2.0.hypot(v2.1);
                if l_a < 1e-4 || l_b < 1e-4 {
                    continue;
                }

                let u1 = (v1.0 / l_a, v1.1 / l_a);
                let u2 = (v2.0 / l_b, v2.1 / l_b);
                let dot = (u1.0 * u2.0 + u1.1 * u2.1).clamp(-1.0, 1.0);
                let theta = dot.acos();
                if theta < 0.05 || theta > (std::f64::consts::PI - 0.05) {
                    continue;
                }

                let alpha = theta * 0.5;
                let b_sum = (u1.0 + u2.0, u1.1 + u2.1);
                let b_len = b_sum.0.hypot(b_sum.1);
                if b_len < 1e-6 {
                    continue;
                }
                let b = (b_sum.0 / b_len, b_sum.1 / b_len);

                let d_max = (l_a * 0.49).min(l_b * 0.49);
                let r_max_geom = d_max * alpha.tan();
                let target_radius: f64 = 1.0;
                let r_init = target_radius.min(r_max_geom).min(l_a * 0.5).min(l_b * 0.5);

                for scale in [1.0, 0.8, 0.6, 0.4, 0.25] {
                    let r = r_init * scale;
                    if r < 0.02 {
                        continue;
                    }
                    let d = r / alpha.tan();
                    let t1 = (p_curr.0 + d * u1.0, p_curr.1 + d * u1.1);
                    let t2 = (p_curr.0 + d * u2.0, p_curr.1 + d * u2.1);
                    let h = r / alpha.sin();
                    let c = (p_curr.0 + h * b.0, p_curr.1 + h * b.1);
                    let m_arc = (c.0 - r * b.0, c.1 - r * b.1);

                    if let Some(arc) = ArcSegment::from_3pts(t1, m_arc, t2, chain.width, chain.layer, chain.net_id) {
                        if is_arc_clear(board, &arc) {
                            corner_arcs[i] = Some((t1, t2, arc));
                            break;
                        }
                    }
                }
            }

            for k in 0..n - 1 {
                let start_pt = if k > 0 && corner_arcs[k].is_some() {
                    corner_arcs[k].as_ref().unwrap().1
                } else {
                    chain.points[k]
                };

                let end_pt = if corner_arcs[k + 1].is_some() {
                    corner_arcs[k + 1].as_ref().unwrap().0
                } else {
                    chain.points[k + 1]
                };

                if (end_pt.0 - start_pt.0).hypot(end_pt.1 - start_pt.1) > 1e-4 {
                    board.tracks.push(TrackSegment {
                        start: start_pt,
                        end: end_pt,
                        width: chain.width,
                        layer: chain.layer,
                        net_id: chain.net_id,
                        is_locked: false,
                    });
                }
            }

            for opt_arc in corner_arcs.into_iter().flatten() {
                board.arcs.push(opt_arc.2);
            }
        }
    }
}

// ============================================================================
// Robust Copper Geometry, DRC Validation & Automated Pruning
// ============================================================================

/// Computes the Euclidean distance from a point `p` to a line segment `(a, b)`.
pub fn dist_point_to_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    dist_to_segment(p.0, p.1, a.0, a.1, b.0, b.1)
}

/// Computes the minimum distance from point `pt` to the copper area of `pad`.
pub fn dist_point_to_pad(pt: (f64, f64), fp: &Footprint, pad: &Pad) -> f64 {
    let (px, py) = fp.pad_absolute_position(pad);
    if pad.shape == "circle" {
        let r = pad.width.max(pad.height) * 0.5;
        let d = (pt.0 - px).hypot(pt.1 - py);
        return (d - r).max(0.0);
    }
    let flipped = fp.layer != fp.orig_layer;
    let pad_rot = if flipped { -pad.rotation } else { pad.rotation };
    let total_rot = ((fp.rotation + pad_rot) % 360.0 + 360.0) % 360.0;
    let rad = (-total_rot).to_radians();
    let rx = pt.0 - px;
    let ry = pt.1 - py;
    let lx = rx * rad.cos() - ry * rad.sin();
    let ly = rx * rad.sin() + ry * rad.cos();
    let hw = pad.width * 0.5;
    let hh = pad.height * 0.5;

    if pad.shape == "oval" {
        let r = hw.min(hh);
        if hw >= hh {
            let cap_dx = hw - r;
            let clamped_x = lx.clamp(-cap_dx, cap_dx);
            let d = (lx - clamped_x).hypot(ly);
            (d - r).max(0.0)
        } else {
            let cap_dy = hh - r;
            let clamped_y = ly.clamp(-cap_dy, cap_dy);
            let d = lx.hypot(ly - clamped_y);
            (d - r).max(0.0)
        }
    } else {
        let dx = (lx.abs() - hw).max(0.0);
        let dy = (ly.abs() - hh).max(0.0);
        dx.hypot(dy)
    }
}

/// Computes the minimum distance from line segment `(s, e)` to the copper area of `pad`.
pub fn dist_segment_to_pad(s: (f64, f64), e: (f64, f64), fp: &Footprint, pad: &Pad) -> f64 {
    let (px, py) = fp.pad_absolute_position(pad);
    if pad.shape == "circle" {
        let r = pad.width.max(pad.height) * 0.5;
        let d = dist_point_to_segment((px, py), s, e);
        return (d - r).max(0.0);
    }
    let flipped = fp.layer != fp.orig_layer;
    let pad_rot = if flipped { -pad.rotation } else { pad.rotation };
    let total_rot = ((fp.rotation + pad_rot) % 360.0 + 360.0) % 360.0;
    let rad = (-total_rot).to_radians();
    let cos_r = rad.cos();
    let sin_r = rad.sin();
    let to_local = |p: (f64, f64)| -> (f64, f64) {
        let rx = p.0 - px;
        let ry = p.1 - py;
        (rx * cos_r - ry * sin_r, rx * sin_r + ry * cos_r)
    };
    let ls = to_local(s);
    let le = to_local(e);
    let hw = pad.width * 0.5;
    let hh = pad.height * 0.5;

    if ls.0.abs() <= hw && ls.1.abs() <= hh {
        return 0.0;
    }
    if le.0.abs() <= hw && le.1.abs() <= hh {
        return 0.0;
    }

    let c1 = (-hw, -hh);
    let c2 = (hw, -hh);
    let c3 = (hw, hh);
    let c4 = (-hw, hh);

    if segments_intersect(ls, le, c1, c2)
        || segments_intersect(ls, le, c2, c3)
        || segments_intersect(ls, le, c3, c4)
        || segments_intersect(ls, le, c4, c1)
    {
        return 0.0;
    }

    let d1 = (ls.0.abs() - hw).max(0.0).hypot((ls.1.abs() - hh).max(0.0));
    let d2 = (le.0.abs() - hw).max(0.0).hypot((le.1.abs() - hh).max(0.0));
    let d3 = dist_point_to_segment(c1, ls, le);
    let d4 = dist_point_to_segment(c2, ls, le);
    let d5 = dist_point_to_segment(c3, ls, le);
    let d6 = dist_point_to_segment(c4, ls, le);
    d1.min(d2).min(d3).min(d4).min(d5).min(d6)
}

/// Verifies whether candidate via location `pos` is completely legal and clearance-free
/// against all foreign pads, tracks, vias, board boundary, and keepouts, as well as
/// preventing overlapping or violating clearance against same-net SMD pads.
/// An optional `ignore_pos` can be supplied to skip drill and proximity collision with
/// an existing via undergoing repositioning or sliding.
pub fn is_via_location_legal_ignoring(
    board: &Board,
    pos: (f64, f64),
    via_dia: f64,
    net_id: usize,
    clearance: f64,
    ignore_pos: Option<(f64, f64)>,
) -> bool {
    let via_r = via_dia * 0.5;

    // 1. Board boundary & edge cuts clearance
    let edge_clr = board.get_netclass_for_net(net_id).board_edge_clearance;
    let req_edge_dist = via_r + edge_clr;
    if let Some(ref poly) = board.outline_polygon {
        let b = poly.aabb();
        if pos.0 < b.min_x + req_edge_dist
            || pos.0 > b.max_x - req_edge_dist
            || pos.1 < b.min_y + req_edge_dist
            || pos.1 > b.max_y - req_edge_dist
        {
            return false;
        }
        if poly.vertices.len() > 4 {
            let pt = Point::new(pos.0, pos.1);
            if !poly.contains_point(&pt) || poly.distance_to_point(&pt) < req_edge_dist - 1e-4 {
                return false;
            }
        }
    } else if !board.contains_point_in_boundary(&Point::new(pos.0, pos.1)) {
        return false;
    }

    // 2. Keepouts
    for ko in &board.keepouts {
        let r = via_r + clearance;
        if pos.0 >= ko.min_x - r && pos.0 <= ko.max_x + r && pos.1 >= ko.min_y - r && pos.1 <= ko.max_y + r {
            return false;
        }
    }

    // 3. Footprint pads (foreign pads on all layers, including net 0; and same-net SMD pads)
    for fp in &board.footprints {
        for pad in &fp.pads {
            let d = dist_point_to_pad(pos, fp, pad);
            let (px, py) = fp.pad_absolute_position(pad);
            if pad.net_id != net_id {
                let min_c = if fp.is_bga() && fp.bga_pitch().map_or(false, |p| p.0 < 0.5 || p.1 < 0.5) {
                    0.05
                } else {
                    0.12
                };
                let pad_c = clearance.max(board.get_netclass_for_net(pad.net_id).clearance).max(min_c);
                if d < via_r + pad_c - 1e-4 {
                    return false;
                }
            } else if !pad.is_through_hole {
                // Same-net SMD pad: via must NEVER overlap pad copper, UNLESS it is a centered VIPPO microvia on a BGA pad
                let is_vippo = fp.is_bga() && via_dia <= 0.35 && (pos.0 - px).hypot(pos.1 - py) < 1e-3;
                if !is_vippo && d < via_r - 1e-4 {
                    return false;
                }
            }
        }
    }

    // 4. Foreign tracks on all copper layers
    let min_c = if clearance < 0.12 { clearance } else { 0.12 };
    for trk in &board.tracks {
        if trk.net_id != net_id {
            let trk_c = clearance.max(board.get_netclass_for_net(trk.net_id).clearance).max(min_c);
            let d = dist_point_to_segment(pos, trk.start, trk.end);
            if d < via_r + trk.width * 0.5 + trk_c - 1e-4 {
                return false;
            }
        }
    }

    // 5. Foreign vias
    for v in &board.vias {
        if let Some(ig) = ignore_pos {
            if (v.position.0 - ig.0).hypot(v.position.1 - ig.1) < 1e-4 {
                continue;
            }
        }
        if (pos.0 - v.position.0).hypot(pos.1 - v.position.1) < 1e-4 {
            continue; // self
        }
        if v.net_id != net_id {
            let v_c = clearance.max(board.get_netclass_for_net(v.net_id).clearance).max(min_c);
            let d = (pos.0 - v.position.0).hypot(pos.1 - v.position.1);
            if d < via_r + v.size * 0.5 + v_c - 1e-4 {
                return false;
            }
        }
    }

    // 6. Hole-to-hole drill clearance against all through-hole pads (including same net)
    let via_drill = if via_dia <= 0.35 { 0.15 } else { 0.30 };
    for fp in &board.footprints {
        for pad in &fp.pads {
            if pad.is_through_hole {
                if let Some(drill) = pad.drill {
                    if drill > 0.0 {
                        let (px, py) = fp.pad_absolute_position(pad);
                        let d = (pos.0 - px).hypot(pos.1 - py);
                        let min_hole_dist = (via_drill * 0.5) + (drill * 0.5) + 0.25;
                        if d < min_hole_dist - 1e-4 {
                            return false;
                        }
                    }
                }
            }
        }
    }

    // 7. Hole-to-hole drill clearance against all other vias (including same net)
    for v in &board.vias {
        if let Some(ig) = ignore_pos {
            if (v.position.0 - ig.0).hypot(v.position.1 - ig.1) < 1e-4 {
                continue;
            }
        }
        if (pos.0 - v.position.0).hypot(pos.1 - v.position.1) < 1e-4 {
            continue; // self
        }
        let d = (pos.0 - v.position.0).hypot(pos.1 - v.position.1);
        let min_hole_dist = if via_dia <= 0.35 && v.drill <= 0.18 {
            (via_drill * 0.5) + (v.drill * 0.5) + 0.10
        } else {
            (via_drill * 0.5) + (v.drill * 0.5) + 0.25
        };
        if d < min_hole_dist - 1e-4 {
            return false;
        }
    }

    true
}

/// Verifies whether candidate via location `pos` is completely legal and clearance-free
/// against all foreign pads, tracks, vias, board boundary, and keepouts, as well as
/// preventing overlapping or violating clearance against same-net SMD pads.
pub fn is_via_location_legal(
    board: &Board,
    pos: (f64, f64),
    via_dia: f64,
    net_id: usize,
    clearance: f64,
) -> bool {
    is_via_location_legal_ignoring(board, pos, via_dia, net_id, clearance, None)
}

/// Verifies whether a candidate track segment is legal and clearance-free against foreign copper and keepouts.
pub fn is_track_segment_legal(
    board: &Board,
    start: (f64, f64),
    end: (f64, f64),
    width: f64,
    layer: usize,
    net_id: usize,
    clearance: f64,
) -> bool {
    let half_w = width * 0.5;

    // 1. Board boundary & edge cuts clearance
    let edge_clr = board.get_netclass_for_net(net_id).board_edge_clearance;
    let req_edge_dist = half_w + edge_clr;
    if let Some(ref poly) = board.outline_polygon {
        let b = poly.aabb();
        let s_min_x = start.0.min(end.0);
        let s_max_x = start.0.max(end.0);
        let s_min_y = start.1.min(end.1);
        let s_max_y = start.1.max(end.1);
        if s_min_x < b.min_x + req_edge_dist
            || s_max_x > b.max_x - req_edge_dist
            || s_min_y < b.min_y + req_edge_dist
            || s_max_y > b.max_y - req_edge_dist
        {
            return false;
        }
        if poly.vertices.len() > 4 {
            let p1 = Point::new(start.0, start.1);
            let p2 = Point::new(end.0, end.1);
            if !poly.contains_point(&p1) || !poly.contains_point(&p2)
                || poly.distance_to_segment(start, end) < req_edge_dist - 1e-4
            {
                return false;
            }
        }
    } else if !board.contains_point_in_boundary(&Point::new(start.0, start.1))
        || !board.contains_point_in_boundary(&Point::new(end.0, end.1))
    {
        return false;
    }

    // 2. Keepouts
    for ko in &board.keepouts {
        if !ko.layers.contains(&layer) {
            continue;
        }
        let r = half_w + clearance;
        let s_min_x = start.0.min(end.0);
        let s_max_x = start.0.max(end.0);
        let s_min_y = start.1.min(end.1);
        let s_max_y = start.1.max(end.1);
        if s_max_x >= ko.min_x - r && s_min_x <= ko.max_x + r && s_max_y >= ko.min_y - r && s_min_y <= ko.max_y + r {
            return false;
        }
    }

    // 3. Foreign pads (SMD on same layer, through-hole on all layers, including net 0)
    for fp in &board.footprints {
        let is_origin_fp = fp.pads.iter().any(|p| {
            let (px, py) = fp.pad_absolute_position(p);
            (px - start.0).hypot(py - start.1) < 1e-3 || (px - end.0).hypot(py - end.1) < 1e-3
        });
        if is_origin_fp {
            continue;
        }
        for pad in &fp.pads {
            if pad.net_id != net_id {
                if pad.is_through_hole || fp.layer == layer {
                    let pad_c = clearance.max(board.get_netclass_for_net(pad.net_id).clearance).max(0.12);
                    let d = dist_segment_to_pad(start, end, fp, pad);
                    if d < half_w + pad_c - 1e-4 {
                        return false;
                    }
                }
            }
        }
    }

    // 4. Foreign tracks on same layer
    for trk in &board.tracks {
        if trk.net_id != net_id && trk.layer == layer {
            let trk_c = clearance.max(board.get_netclass_for_net(trk.net_id).clearance).max(0.12);
            let d = dist_segment_to_segment(start, end, trk.start, trk.end);
            if d < half_w + trk.width * 0.5 + trk_c - 1e-4 {
                return false;
            }
        }
    }

    // 5. Foreign vias on all layers
    for v in &board.vias {
        if v.net_id != net_id {
            let v_c = clearance.max(board.get_netclass_for_net(v.net_id).clearance).max(0.12);
            let d = dist_point_to_segment(v.position, start, end);
            if d < half_w + v.size * 0.5 + v_c - 1e-4 {
                return false;
            }
        }
    }

    true
}

/// Automated dangling track & via pruning.
/// Iteratively removes dead-end degree-1 track segments (antennas) and unconnected/single-layer vias.
pub fn prune_dangling_tracks_and_vias(board: &mut Board) {
    if board.footprints.is_empty() {
        return;
    }
    let plane_nets = board.inner_plane_nets();
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        net_name == "GND"
            || net_name.starts_with("GND")
            || plane_nets.values().any(|&pn| pn == nid)
    };

    const EPS: f64 = 0.05;

    let mut changed = true;
    let mut pass = 0;
    while changed && pass < 50 {
        changed = false;
        pass += 1;

        let num_tracks = board.tracks.len();
        let num_vias = board.vias.len();
        let mut pruned_tracks = vec![false; num_tracks];
        let mut pruned_vias = vec![false; num_vias];

        // 1. Identify track dead-ends
        for i in 0..num_tracks {
            let t = &board.tracks[i];
            if t.is_locked {
                continue;
            }

            let check_endpoint = |pt: (f64, f64)| -> bool {
                // A. Check footprint pads of matching net
                for fp in &board.footprints {
                    for pad in &fp.pads {
                        if pad.net_id == t.net_id {
                            if pad.is_through_hole || fp.layer == t.layer {
                                let (px, py) = fp.pad_absolute_position(pad);
                                if (pt.0 - px).hypot(pt.1 - py) < 0.12
                                    || dist_point_to_pad(pt, fp, pad) < EPS
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }

                // B. Check vias of matching net
                for (v_idx, v) in board.vias.iter().enumerate() {
                    if v.net_id == t.net_id && !pruned_vias[v_idx] {
                        if (pt.0 - v.position.0).hypot(pt.1 - v.position.1) <= v.size * 0.5 + EPS {
                            return true;
                        }
                    }
                }

                // C. Check other tracks of matching net on same layer
                for j in 0..num_tracks {
                    if i != j && !pruned_tracks[j] {
                        let o = &board.tracks[j];
                        if o.net_id == t.net_id && o.layer == t.layer {
                            if (pt.0 - o.start.0).hypot(pt.1 - o.start.1) < EPS
                                || (pt.0 - o.end.0).hypot(pt.1 - o.end.1) < EPS
                                || dist_point_to_segment(pt, o.start, o.end) < EPS
                            {
                                return true;
                            }
                        }
                    }
                }

                false
            };

            let start_connected = check_endpoint(t.start);
            let end_connected = check_endpoint(t.end);

            if !start_connected || !end_connected {
                pruned_tracks[i] = true;
                changed = true;
            }
        }

        // 2. Identify dangling / single-layer vias
        for (v_idx, v) in board.vias.iter().enumerate() {
            if v.is_locked || pruned_vias[v_idx] {
                continue;
            }

            let touches_pad = board.footprints.iter().any(|fp| {
                fp.pads.iter().any(|p| {
                    if p.net_id == v.net_id {
                        let (px, py) = fp.pad_absolute_position(p);
                        (v.position.0 - px).hypot(v.position.1 - py) <= p.width.max(p.height) * 0.5 + EPS
                            || dist_point_to_pad(v.position, fp, p) < EPS
                    } else {
                        false
                    }
                })
            });

            let has_plane = is_plane_net(v.net_id);

            let mut connected_tracks = 0;
            let mut connected_layers = HashSet::new();

            for (t_idx, t) in board.tracks.iter().enumerate() {
                if t.net_id == v.net_id && !pruned_tracks[t_idx] {
                    let d_start = (t.start.0 - v.position.0).hypot(t.start.1 - v.position.1);
                    let d_end = (t.end.0 - v.position.0).hypot(t.end.1 - v.position.1);
                    if d_start <= v.size * 0.5 + EPS || d_end <= v.size * 0.5 + EPS {
                        connected_tracks += 1;
                        connected_layers.insert(t.layer);
                    }
                }
            }

            if !touches_pad && !has_plane {
                if connected_tracks <= 1 || connected_layers.len() < 2 {
                    pruned_vias[v_idx] = true;
                    changed = true;
                }
            }
        }

        if changed {
            let mut new_tracks = Vec::with_capacity(num_tracks);
            for (idx, trk) in board.tracks.drain(..).enumerate() {
                if !pruned_tracks[idx] {
                    new_tracks.push(trk);
                }
            }
            board.tracks = new_tracks;

            let mut new_vias = Vec::with_capacity(num_vias);
            for (idx, v) in board.vias.drain(..).enumerate() {
                if !pruned_vias[idx] {
                    new_vias.push(v);
                }
            }
            board.vias = new_vias;
        }
    }

    // 3. Remove orphan components disconnected from any pad or plane
    prune_orphan_components(board);
}

/// Prunes disconnected islands of copper that do not connect to any pad or plane.
fn prune_orphan_components(board: &mut Board) {
    if board.tracks.is_empty() {
        return;
    }
    let plane_nets = board.inner_plane_nets();
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        net_name == "GND"
            || net_name.starts_with("GND")
            || plane_nets.values().any(|&pn| pn == nid)
    };

    const EPS: f64 = 0.005;

    let mut net_ids: Vec<usize> = board.tracks.iter().map(|t| t.net_id).collect();
    net_ids.sort_unstable();
    net_ids.dedup();

    let mut keep_track = vec![true; board.tracks.len()];
    let mut keep_via = vec![true; board.vias.len()];

    for net_id in net_ids {
        let track_indices: Vec<usize> = board
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.net_id == net_id)
            .map(|(i, _)| i)
            .collect();
        let via_indices: Vec<usize> = board
            .vias
            .iter()
            .enumerate()
            .filter(|(_, v)| v.net_id == net_id)
            .map(|(i, _)| i)
            .collect();

        if track_indices.is_empty() {
            continue;
        }

        let pad_positions: Vec<(f64, f64, usize, bool)> = board
            .footprints
            .iter()
            .flat_map(|fp| {
                fp.pads
                    .iter()
                    .filter(|p| p.net_id == net_id)
                    .map(move |p| {
                        let (px, py) = fp.pad_absolute_position(p);
                        (px, py, fp.layer, p.is_through_hole)
                    })
            })
            .collect();

        let has_plane = is_plane_net(net_id);

        let n_t = track_indices.len();
        let n_v = via_indices.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_t + n_v];

        for i in 0..n_t {
            let ti = &board.tracks[track_indices[i]];
            for j in (i + 1)..n_t {
                let tj = &board.tracks[track_indices[j]];
                if ti.layer == tj.layer {
                    let connected = (ti.start.0 - tj.start.0).hypot(ti.start.1 - tj.start.1) < EPS
                        || (ti.start.0 - tj.end.0).hypot(ti.start.1 - tj.end.1) < EPS
                        || (ti.end.0 - tj.start.0).hypot(ti.end.1 - tj.start.1) < EPS
                        || (ti.end.0 - tj.end.0).hypot(ti.end.1 - tj.end.1) < EPS
                        || dist_point_to_segment(ti.start, tj.start, tj.end) < EPS
                        || dist_point_to_segment(ti.end, tj.start, tj.end) < EPS
                        || dist_point_to_segment(tj.start, ti.start, ti.end) < EPS
                        || dist_point_to_segment(tj.end, ti.start, ti.end) < EPS;
                    if connected {
                        adj[i].push(j);
                        adj[j].push(i);
                    }
                }
            }
        }

        for i in 0..n_t {
            let ti = &board.tracks[track_indices[i]];
            for vi in 0..n_v {
                let v = &board.vias[via_indices[vi]];
                let node_v = n_t + vi;
                let d_s = (ti.start.0 - v.position.0).hypot(ti.start.1 - v.position.1);
                let d_e = (ti.end.0 - v.position.0).hypot(ti.end.1 - v.position.1);
                if d_s <= v.size * 0.5 + EPS || d_e <= v.size * 0.5 + EPS {
                    adj[i].push(node_v);
                    adj[node_v].push(i);
                }
            }
        }

        let mut is_anchor = vec![false; n_t + n_v];
        for i in 0..n_t {
            let ti = &board.tracks[track_indices[i]];
            if ti.is_locked {
                is_anchor[i] = true;
                continue;
            }
            for &(px, py, pad_layer, is_th) in &pad_positions {
                if is_th || pad_layer == ti.layer {
                    if (ti.start.0 - px).hypot(ti.start.1 - py) < 0.10
                        || (ti.end.0 - px).hypot(ti.end.1 - py) < 0.10
                        || dist_point_to_segment((px, py), ti.start, ti.end) < 0.05
                    {
                        is_anchor[i] = true;
                        break;
                    }
                }
            }
        }

        for vi in 0..n_v {
            let v = &board.vias[via_indices[vi]];
            let node_v = n_t + vi;
            if v.is_locked || has_plane {
                is_anchor[node_v] = true;
                continue;
            }
            for &(px, py, _, _) in &pad_positions {
                if (v.position.0 - px).hypot(v.position.1 - py) <= v.size * 0.5 + 0.10 {
                    is_anchor[node_v] = true;
                    break;
                }
            }
        }

        let mut visited = vec![false; n_t + n_v];
        let mut queue = VecDeque::new();
        for node in 0..(n_t + n_v) {
            if is_anchor[node] {
                visited[node] = true;
                queue.push_back(node);
            }
        }

        while let Some(curr) = queue.pop_front() {
            for &nxt in &adj[curr] {
                if !visited[nxt] {
                    visited[nxt] = true;
                    queue.push_back(nxt);
                }
            }
        }

        for i in 0..n_t {
            if !visited[i] && !board.tracks[track_indices[i]].is_locked {
                keep_track[track_indices[i]] = false;
            }
        }
        for vi in 0..n_v {
            if !visited[n_t + vi] && !board.vias[via_indices[vi]].is_locked {
                keep_via[via_indices[vi]] = false;
            }
        }
    }

    let mut new_tracks = Vec::with_capacity(board.tracks.len());
    for (i, t) in board.tracks.drain(..).enumerate() {
        if keep_track[i] {
            new_tracks.push(t);
        }
    }
    board.tracks = new_tracks;

    let mut new_vias = Vec::with_capacity(board.vias.len());
    for (i, v) in board.vias.drain(..).enumerate() {
        if keep_via[i] {
            new_vias.push(v);
        }
    }
    board.vias = new_vias;
}

/// Helper function to check if a candidate teardrop segment violates clearance against
/// foreign pads, tracks, vias, keepouts, or board edge cuts.
pub fn is_teardrop_segment_clear(board: &Board, seg: &TrackSegment, extra_tracks: &[TrackSegment]) -> bool {
    let half_w = seg.width * 0.5;
    let net_clr = board.get_netclass_for_net(seg.net_id).clearance.max(0.15);

    // 1. Board edge cuts clearance
    if let Some(ref poly) = board.outline_polygon {
        let edge_clr = board.get_netclass_for_net(seg.net_id).board_edge_clearance;
        let req_edge_dist = half_w + edge_clr;
        let b = poly.aabb();
        let s_min_x = seg.start.0.min(seg.end.0);
        let s_max_x = seg.start.0.max(seg.end.0);
        let s_min_y = seg.start.1.min(seg.end.1);
        let s_max_y = seg.start.1.max(seg.end.1);
        if s_min_x < b.min_x + req_edge_dist
            || s_max_x > b.max_x - req_edge_dist
            || s_min_y < b.min_y + req_edge_dist
            || s_max_y > b.max_y - req_edge_dist
        {
            return false;
        }
        if poly.vertices.len() > 4 {
            let p1 = Point::new(seg.start.0, seg.start.1);
            let p2 = Point::new(seg.end.0, seg.end.1);
            if !poly.contains_point(&p1) || !poly.contains_point(&p2)
                || poly.distance_to_segment(seg.start, seg.end) < req_edge_dist - 1e-4
            {
                return false;
            }
        }
    } else if !board.contains_point_in_boundary(&Point::new(seg.start.0, seg.start.1))
        || !board.contains_point_in_boundary(&Point::new(seg.end.0, seg.end.1))
    {
        return false;
    }

    // 2. Keepouts
    for ko in &board.keepouts {
        if ko.layers.contains(&seg.layer) {
            let r = half_w + net_clr;
            let s_min_x = seg.start.0.min(seg.end.0);
            let s_max_x = seg.start.0.max(seg.end.0);
            let s_min_y = seg.start.1.min(seg.end.1);
            let s_max_y = seg.start.1.max(seg.end.1);
            if s_max_x >= ko.min_x - r && s_min_x <= ko.max_x + r && s_max_y >= ko.min_y - r && s_min_y <= ko.max_y + r {
                let corners = [
                    (ko.min_x, ko.min_y),
                    (ko.max_x, ko.min_y),
                    (ko.max_x, ko.max_y),
                    (ko.min_x, ko.max_y),
                ];
                for k in 0..4 {
                    if dist_segment_to_segment(seg.start, seg.end, corners[k], corners[(k + 1) % 4]) < r - 1e-4 {
                        return false;
                    }
                }
            }
        }
    }

    // 3. Foreign pads
    for fp in &board.footprints {
        for pad in &fp.pads {
            if pad.net_id != seg.net_id && (pad.is_through_hole || fp.layer == seg.layer) {
                let (_px, _py) = fp.pad_absolute_position(pad);
                let pad_c = board.get_netclass_for_net(pad.net_id).clearance.max(0.15);
                let req_c = net_clr.max(pad_c);
                let min_dist = half_w + req_c;
                let d = dist_segment_to_pad(seg.start, seg.end, fp, pad);
                if d < min_dist - 1e-4 {
                    return false;
                }
            }
        }
    }

    // 4. Foreign tracks on same layer (committed tracks + extra uncommitted teardrops)
    for trk in board.tracks.iter().chain(extra_tracks.iter()) {
        if trk.net_id != seg.net_id && trk.layer == seg.layer {
            let trk_c = board.get_netclass_for_net(trk.net_id).clearance.max(0.15);
            let req_c = net_clr.max(trk_c);
            let min_dist = half_w + trk.width * 0.5 + req_c;
            let d = dist_segment_to_segment(seg.start, seg.end, trk.start, trk.end);
            if d < min_dist - 1e-4 {
                return false;
            }
        }
    }

    // 5. Foreign vias
    for v in &board.vias {
        if v.net_id != seg.net_id {
            let via_c = board.get_netclass_for_net(v.net_id).clearance.max(0.15);
            let req_c = net_clr.max(via_c);
            let min_dist = half_w + v.size * 0.5 + req_c;
            let d = dist_point_to_segment(v.position, seg.start, seg.end);
            if d < min_dist - 1e-4 {
                return false;
            }
        }
    }

    true
}

/// Automated DFM Teardrop Generator:
/// For every track segment connected to a footprint pad or through-hole via of the same net,
/// computes outward tangent flares bridging from the track edge to the perimeter of the pad/via.
/// Validates that teardrop copper does not violate clearance rules against foreign copper,
/// and adds clean teardrop flare segments to `board.tracks`.
pub fn add_teardrops(board: &mut Board) {
    if board.tracks.is_empty() {
        return;
    }

    let initial_tracks = board.tracks.clone();
    let mut new_teardrops: Vec<TrackSegment> = Vec::new();

    for track in &initial_tracks {
        let netclass = board.get_netclass_for_net(track.net_id);
        if !netclass.teardrop_enabled {
            continue;
        }

        let length_ratio = netclass.teardrop_length_ratio;
        let width_ratio = netclass.teardrop_width_ratio;
        let track_w = track.width;

        // Examine both endpoints of the track segment for pad/via junctions
        for (t_end, other_end) in [(track.start, track.end), (track.end, track.start)] {
            let d_vec = (other_end.0 - t_end.0, other_end.1 - t_end.1);
            let seg_len = d_vec.0.hypot(d_vec.1);
            if seg_len < 0.05 {
                continue;
            }
            let u = (d_vec.0 / seg_len, d_vec.1 / seg_len);
            let n = (-u.1, u.0);

            // 1. Check for connected footprint pad of matching net
            let mut junction: Option<((f64, f64), f64, f64, bool)> = None; // (center, R_u, R_n, is_circle)

            for fp in &board.footprints {
                if !fp.pads.iter().any(|p| p.net_id == track.net_id && (p.is_through_hole || fp.layer == track.layer)) {
                    continue;
                }
                for pad in &fp.pads {
                    if pad.net_id != track.net_id || (!pad.is_through_hole && fp.layer != track.layer) {
                        continue;
                    }
                    let (px, py) = fp.pad_absolute_position(pad);
                    let dist_to_center = (t_end.0 - px).hypot(t_end.1 - py);
                    let is_connected = dist_to_center <= 0.05
                        || dist_point_to_pad(t_end, fp, pad) <= 0.05
                        || (pad.shape == "circle" && dist_to_center <= pad.width.max(pad.height) * 0.5 + 0.05);

                    if is_connected {
                        if pad.shape == "circle" {
                            let r = pad.width.max(pad.height) * 0.5;
                            junction = Some(((px, py), r, r, true));
                            break;
                        } else {
                            let flipped = fp.layer != fp.orig_layer;
                            let pad_rot = if flipped { -pad.rotation } else { pad.rotation };
                            let total_rot = ((fp.rotation + pad_rot) % 360.0 + 360.0) % 360.0;
                            let rad = (-total_rot).to_radians();
                            let cos_r = rad.cos();
                            let sin_r = rad.sin();
                            let u_loc = (u.0 * cos_r - u.1 * sin_r, u.0 * sin_r + u.1 * cos_r);
                            let n_loc = (n.0 * cos_r - n.1 * sin_r, n.0 * sin_r + n.1 * cos_r);
                            let hw = pad.width * 0.5;
                            let hh = pad.height * 0.5;
                            let tu_x = if u_loc.0.abs() > 1e-6 { hw / u_loc.0.abs() } else { f64::INFINITY };
                            let tu_y = if u_loc.1.abs() > 1e-6 { hh / u_loc.1.abs() } else { f64::INFINITY };
                            let r_u = tu_x.min(tu_y);
                            let tn_x = if n_loc.0.abs() > 1e-6 { hw / n_loc.0.abs() } else { f64::INFINITY };
                            let tn_y = if n_loc.1.abs() > 1e-6 { hh / n_loc.1.abs() } else { f64::INFINITY };
                            let r_n = tn_x.min(tn_y);
                            junction = Some(((px, py), r_u, r_n, false));
                            break;
                        }
                    }
                }
                if junction.is_some() {
                    break;
                }
            }

            // 2. Check for connected through-hole via of matching net if no pad matched
            if junction.is_none() {
                for via in &board.vias {
                    if via.net_id == track.net_id {
                        let dist_to_center = (t_end.0 - via.position.0).hypot(t_end.1 - via.position.1);
                        let r = via.size * 0.5;
                        if dist_to_center <= 0.05 || dist_to_center <= r + 0.05 {
                            junction = Some((via.position, r, r, true));
                            break;
                        }
                    }
                }
            }

            let Some((center, r_u, r_n, is_circle)) = junction else {
                continue;
            };

            let teardrop_len = r_u * length_ratio;
            let flare_half_w = r_n * width_ratio;

            if flare_half_w <= track_w * 0.5 + 0.01 {
                continue;
            }

            // Distance along u from pad/via center to the other end of this segment
            let dist_to_other = (other_end.0 - center.0) * u.0 + (other_end.1 - center.1) * u.1;
            if dist_to_other <= r_u + 0.05 {
                continue;
            }

            let max_d = (r_u + 0.05).max(dist_to_other * 0.85).min(r_u + teardrop_len);
            let a = (center.0 + max_d * u.0, center.1 + max_d * u.1);

            let (b_left, b_right) = if is_circle {
                let h = flare_half_w.min(r_n * 0.98);
                let x_pad = (r_u * r_u - h * h).max(0.0).sqrt();
                (
                    (center.0 + x_pad * u.0 + h * n.0, center.1 + x_pad * u.1 + h * n.1),
                    (center.0 + x_pad * u.0 - h * n.0, center.1 + x_pad * u.1 - h * n.1),
                )
            } else {
                let h = flare_half_w.min(r_n * 0.95);
                (
                    (center.0 + r_u * u.0 + h * n.0, center.1 + r_u * u.1 + h * n.1),
                    (center.0 + r_u * u.0 - h * n.0, center.1 + r_u * u.1 - h * n.1),
                )
            };

            let cand_left = TrackSegment {
                start: a,
                end: b_left,
                width: track_w,
                layer: track.layer,
                net_id: track.net_id,
                is_locked: false,
            };

            let cand_right = TrackSegment {
                start: a,
                end: b_right,
                width: track_w,
                layer: track.layer,
                net_id: track.net_id,
                is_locked: false,
            };

            let exists = |cand: &TrackSegment, list: &[TrackSegment]| -> bool {
                list.iter().any(|t| {
                    t.net_id == cand.net_id
                        && t.layer == cand.layer
                        && (((t.start.0 - cand.start.0).hypot(t.start.1 - cand.start.1) < 1e-3
                            && (t.end.0 - cand.end.0).hypot(t.end.1 - cand.end.1) < 1e-3)
                            || ((t.start.0 - cand.end.0).hypot(t.start.1 - cand.end.1) < 1e-3
                                && (t.end.0 - cand.start.0).hypot(t.end.1 - cand.start.1) < 1e-3))
                })
            };

            if !exists(&cand_left, &board.tracks) && !exists(&cand_left, &new_teardrops) {
                if is_teardrop_segment_clear(board, &cand_left, &new_teardrops) {
                    new_teardrops.push(cand_left);
                }
            }

            if !exists(&cand_right, &board.tracks) && !exists(&cand_right, &new_teardrops) {
                if is_teardrop_segment_clear(board, &cand_right, &new_teardrops) {
                    new_teardrops.push(cand_right);
                }
            }
        }
    }

    board.tracks.extend(new_teardrops);
}

/// Comprehensive pre-export DRC validation: detects and eliminates any residual shorts
/// or clearance violations between tracks, vias, and pads of different nets.
pub fn validate_and_clean_copper(board: &mut Board) {
    let mut changed = true;
    let mut iteration = 0;
    while changed && iteration < 10 {
        changed = false;
        iteration += 1;

        let mut bad_track_indices = HashSet::new();
        let track_count = board.tracks.len();
        for i in 0..track_count {
            if board.tracks[i].is_locked {
                continue;
            }
            let t1 = &board.tracks[i];

            // Copper-to-board-edge clearance check
            if let Some(ref poly) = board.outline_polygon {
                let edge_clr = board.get_netclass_for_net(t1.net_id).board_edge_clearance;
                let req_dist = t1.width * 0.5 + edge_clr;
                let b = poly.aabb();
                let s_min_x = t1.start.0.min(t1.end.0);
                let s_max_x = t1.start.0.max(t1.end.0);
                let s_min_y = t1.start.1.min(t1.end.1);
                let s_max_y = t1.start.1.max(t1.end.1);
                if s_min_x < b.min_x + req_dist
                    || s_max_x > b.max_x - req_dist
                    || s_min_y < b.min_y + req_dist
                    || s_max_y > b.max_y - req_dist
                {
                    bad_track_indices.insert(i);
                    continue;
                }
                if poly.vertices.len() > 4 {
                    let p1 = Point::new(t1.start.0, t1.start.1);
                    let p2 = Point::new(t1.end.0, t1.end.1);
                    if !poly.contains_point(&p1) || !poly.contains_point(&p2)
                        || poly.distance_to_segment(t1.start, t1.end) < req_dist - 1e-4
                    {
                        bad_track_indices.insert(i);
                        continue;
                    }
                }
            }

            for j in 0..track_count {
                if i == j {
                    continue;
                }
                let t2 = &board.tracks[j];
                if t1.net_id != t2.net_id && t1.layer == t2.layer {
                    let short_dist = (t1.width + t2.width) * 0.5;
                    let d = dist_segment_to_segment(t1.start, t1.end, t2.start, t2.end);
                    if d < short_dist - 1e-4 {
                        bad_track_indices.insert(i);
                        break;
                    }
                }
            }
            if bad_track_indices.contains(&i) {
                continue;
            }

            for fp in &board.footprints {
                for pad in &fp.pads {
                    if pad.net_id != t1.net_id {
                        if pad.is_through_hole || fp.layer == t1.layer {
                            let short_dist = t1.width * 0.5;
                            let d = dist_segment_to_pad(t1.start, t1.end, fp, pad);
                            if d < short_dist - 1e-4 {
                                bad_track_indices.insert(i);
                                break;
                            }
                        }
                    }
                }
                if bad_track_indices.contains(&i) {
                    break;
                }
            }
            if bad_track_indices.contains(&i) {
                continue;
            }

            for v in &board.vias {
                if v.net_id != t1.net_id {
                    let short_dist = (t1.width + v.size) * 0.5;
                    let d = dist_point_to_segment(v.position, t1.start, t1.end);
                    if d < short_dist - 1e-4 {
                        bad_track_indices.insert(i);
                        break;
                    }
                }
            }
        }

        let mut bad_via_indices = HashSet::new();
        let via_count = board.vias.len();
        for i in 0..via_count {
            if board.vias[i].is_locked {
                continue;
            }
            let v1 = &board.vias[i];

            // Copper-to-board-edge clearance check
            if let Some(ref poly) = board.outline_polygon {
                let edge_clr = board.get_netclass_for_net(v1.net_id).board_edge_clearance;
                let req_dist = v1.size * 0.5 + edge_clr;
                let b = poly.aabb();
                if v1.position.0 < b.min_x + req_dist
                    || v1.position.0 > b.max_x - req_dist
                    || v1.position.1 < b.min_y + req_dist
                    || v1.position.1 > b.max_y - req_dist
                {
                    bad_via_indices.insert(i);
                    continue;
                }
                if poly.vertices.len() > 4 {
                    let pt = Point::new(v1.position.0, v1.position.1);
                    if !poly.contains_point(&pt) || poly.distance_to_point(&pt) < req_dist - 1e-4 {
                        bad_via_indices.insert(i);
                        continue;
                    }
                }
            }

            for fp in &board.footprints {
                for pad in &fp.pads {
                    let d = dist_point_to_pad(v1.position, fp, pad);
                    if pad.net_id != v1.net_id {
                        let short_dist = v1.size * 0.5;
                        if d < short_dist - 1e-4 {
                            bad_via_indices.insert(i);
                            break;
                        }
                    } else if !pad.is_through_hole {
                        let min_dist = v1.size * 0.5;
                        if d < min_dist - 1e-4 {
                            bad_via_indices.insert(i);
                            break;
                        }
                    }
                }
                if bad_via_indices.contains(&i) {
                    break;
                }
            }
            if bad_via_indices.contains(&i) {
                continue;
            }

            for fp in &board.footprints {
                for pad in &fp.pads {
                    if pad.is_through_hole {
                        if let Some(drill) = pad.drill {
                            if drill > 0.0 {
                                let (px, py) = fp.pad_absolute_position(pad);
                                let d = (v1.position.0 - px).hypot(v1.position.1 - py);
                                let min_hole_dist = (v1.drill * 0.5) + (drill * 0.5) + 0.20;
                                if d < min_hole_dist - 1e-4 {
                                    bad_via_indices.insert(i);
                                    break;
                                }
                            }
                        }
                    }
                }
                if bad_via_indices.contains(&i) {
                    break;
                }
            }
            if bad_via_indices.contains(&i) {
                continue;
            }

            for j in 0..via_count {
                if i == j {
                    continue;
                }
                let v2 = &board.vias[j];
                let min_hole_dist = (v1.drill * 0.5) + (v2.drill * 0.5) + 0.20;
                let d = (v1.position.0 - v2.position.0).hypot(v1.position.1 - v2.position.1);
                if d < min_hole_dist - 1e-4 {
                    bad_via_indices.insert(i);
                    break;
                }
                if v1.net_id != v2.net_id {
                    let short_dist = (v1.size + v2.size) * 0.5;
                    if d < short_dist - 1e-4 {
                        bad_via_indices.insert(i);
                        break;
                    }
                }
            }
        }

        if !bad_track_indices.is_empty() {
            let mut new_tracks = Vec::with_capacity(board.tracks.len() - bad_track_indices.len());
            for (idx, trk) in board.tracks.drain(..).enumerate() {
                if !bad_track_indices.contains(&idx) {
                    new_tracks.push(trk);
                }
            }
            board.tracks = new_tracks;
            changed = true;
        }

        if !bad_via_indices.is_empty() {
            let mut new_vias = Vec::with_capacity(board.vias.len() - bad_via_indices.len());
            for (idx, v) in board.vias.drain(..).enumerate() {
                if !bad_via_indices.contains(&idx) {
                    new_vias.push(v);
                }
            }
            board.vias = new_vias;
            changed = true;
        }

        prune_dangling_tracks_and_vias(board);
    }
}

/// Represents an SI violation where parallel trace coupling between two nets exceeds the permitted threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CouplingViolation {
    pub net_a: usize,
    pub net_b: usize,
    pub net_a_name: String,
    pub net_b_name: String,
    pub coupled_length_mm: f64,
    pub max_allowed_mm: f64,
    pub span_start: (f64, f64),
    pub span_end: (f64, f64),
}

/// Evaluates post-routing parallel trace coupling between every net pair on each copper layer.
/// Identifies parallel runs where centerline separation d <= (W_a + W_b)/2 + 2 * max(C_a, C_b) + 0.05 mm,
/// accumulates coupled length, and emits warnings and violations when exceeding max_coupled_length_mm.
pub fn validate_crosstalk_and_coupling(board: &Board) -> Vec<CouplingViolation> {
    let mut violations = Vec::new();
    let track_count = board.tracks.len();
    if track_count == 0 {
        return violations;
    }

    let mut coupling_map: HashMap<(usize, usize), (f64, f64, f64, f64, f64)> = HashMap::new();

    for i in 0..track_count {
        let t1 = &board.tracks[i];
        if t1.net_id == 0 {
            continue;
        }
        let v1_x = t1.end.0 - t1.start.0;
        let v1_y = t1.end.1 - t1.start.1;
        let len1 = v1_x.hypot(v1_y);
        if len1 < 1e-4 {
            continue;
        }
        let u1 = (v1_x / len1, v1_y / len1);
        let n1 = (-u1.1, u1.0);

        let nc1 = board.get_netclass_for_net(t1.net_id);

        for j in (i + 1)..track_count {
            let t2 = &board.tracks[j];
            if t1.net_id == t2.net_id || t1.layer != t2.layer || t2.net_id == 0 {
                continue;
            }

            let v2_x = t2.end.0 - t2.start.0;
            let v2_y = t2.end.1 - t2.start.1;
            let len2 = v2_x.hypot(v2_y);
            if len2 < 1e-4 {
                continue;
            }
            let u2 = (v2_x / len2, v2_y / len2);
            let dot = (u1.0 * u2.0 + u1.1 * u2.1).abs();
            if dot < 0.99 {
                continue;
            }

            let nc2 = board.get_netclass_for_net(t2.net_id);
            let coupling_thresh = (t1.width + t2.width) * 0.5 + 2.0 * nc1.clearance.max(nc2.clearance) + 0.05;

            let d_start = ((t2.start.0 - t1.start.0) * n1.0 + (t2.start.1 - t1.start.1) * n1.1).abs();
            let d_end = ((t2.end.0 - t1.start.0) * n1.0 + (t2.end.1 - t1.start.1) * n1.1).abs();
            let d_perp = (d_start + d_end) * 0.5;

            if d_perp > coupling_thresh {
                continue;
            }

            let p_start = (t2.start.0 - t1.start.0) * u1.0 + (t2.start.1 - t1.start.1) * u1.1;
            let p_end = (t2.end.0 - t1.start.0) * u1.0 + (t2.end.1 - t1.start.1) * u1.1;
            let t2_min = p_start.min(p_end);
            let t2_max = p_start.max(p_end);

            let overlap_start = 0.0f64.max(t2_min);
            let overlap_end = len1.min(t2_max);

            if overlap_end > overlap_start + 1e-4 {
                let coupled_len = overlap_end - overlap_start;
                let pair_key = if t1.net_id < t2.net_id {
                    (t1.net_id, t2.net_id)
                } else {
                    (t2.net_id, t1.net_id)
                };

                let pt_a = (t1.start.0 + overlap_start * u1.0, t1.start.1 + overlap_start * u1.1);
                let pt_b = (t1.start.0 + overlap_end * u1.0, t1.start.1 + overlap_end * u1.1);

                let entry = coupling_map.entry(pair_key).or_insert((
                    0.0,
                    pt_a.0.min(pt_b.0),
                    pt_a.1.min(pt_b.1),
                    pt_a.0.max(pt_b.0),
                    pt_a.1.max(pt_b.1),
                ));
                entry.0 += coupled_len;
                entry.1 = entry.1.min(pt_a.0).min(pt_b.0);
                entry.2 = entry.2.min(pt_a.1).min(pt_b.1);
                entry.3 = entry.3.max(pt_a.0).max(pt_b.0);
                entry.4 = entry.4.max(pt_a.1).max(pt_b.1);
            }
        }
    }

    for (&(net_a, net_b), &(tot_len, min_x, min_y, max_x, max_y)) in &coupling_map {
        let nc_a = board.get_netclass_for_net(net_a);
        let nc_b = board.get_netclass_for_net(net_b);
        let max_allowed = nc_a.max_coupled_length_mm.min(nc_b.max_coupled_length_mm);

        if tot_len > max_allowed + 1e-3 {
            let net_a_name = board.nets.get(net_a).cloned().unwrap_or_else(|| format!("Net{}", net_a));
            let net_b_name = board.nets.get(net_b).cloned().unwrap_or_else(|| format!("Net{}", net_b));
            let span_start = (min_x, min_y);
            let span_end = (max_x, max_y);

            eprintln!(
                "[SI WARNING] Crosstalk coupled length between '{}' (net {}) and '{}' (net {}) is {:.2} mm (max allowed: {:.2} mm) spanning ({:.2}, {:.2}) to ({:.2}, {:.2})",
                net_a_name, net_a, net_b_name, net_b, tot_len, max_allowed, span_start.0, span_start.1, span_end.0, span_end.1
            );

            violations.push(CouplingViolation {
                net_a,
                net_b,
                net_a_name,
                net_b_name,
                coupled_length_mm: tot_len,
                max_allowed_mm: max_allowed,
                span_start,
                span_end,
            });
        }
    }

    violations
}

/// Generates the parallel equidistant track segments for all nets in a BusBundle along its guide polyline.
/// Handles corner mitering (concentric 45° chamfers) and terminal pad fan-in/fan-out.
pub fn generate_bundle_tracks(
    board: &Board,
    bundle: &BusBundle,
    spacing: f64,
) -> Vec<TrackSegment> {
    if bundle.guide_points.len() < 2 || bundle.net_ids.is_empty() {
        return Vec::new();
    }

    // 1. Sanitize raw guide points by removing adjacent duplicate/micro points
    let mut raw_points: Vec<(f64, f64)> = Vec::new();
    for &pt in &bundle.guide_points {
        if let Some(&last) = raw_points.last() {
            if (pt.0 - last.0).hypot(pt.1 - last.1) < 1e-4 {
                continue;
            }
        }
        raw_points.push(pt);
    }
    if raw_points.len() < 2 {
        return Vec::new();
    }

    let m = bundle.net_ids.len();
    let half_span = (m as f64 - 1.0) * 0.5;
    let pitch = bundle.track_width + spacing;
    let o_max = half_span * pitch;

    // 2. Corner beveling on guide polyline:
    // Any turn sharper than ~45° (cos_theta < 0.707) is beveled with 45° miter chamfers
    let mut beveled_guide: Vec<(f64, f64)> = Vec::new();
    beveled_guide.push(raw_points[0]);

    for j in 1..raw_points.len() - 1 {
        let prev = raw_points[j - 1];
        let curr = raw_points[j];
        let next = raw_points[j + 1];

        let v1 = (curr.0 - prev.0, curr.1 - prev.1);
        let v2 = (next.0 - curr.0, next.1 - curr.1);
        let l1 = v1.0.hypot(v1.1);
        let l2 = v2.0.hypot(v2.1);

        if l1 < 1e-4 || l2 < 1e-4 {
            continue;
        }

        let u1 = (v1.0 / l1, v1.1 / l1);
        let u2 = (v2.0 / l2, v2.1 / l2);

        let cos_theta = u1.0 * u2.0 + u1.1 * u2.1;

        if cos_theta > 0.9999 {
            // Collinear, omit redundant joint
            continue;
        }

        // If turn angle is sharp (> ~45°, including 90° corners):
        if cos_theta < 0.707 {
            // Setback needed so even the innermost track has a positive-length 45° chamfer
            let min_sb = o_max * (2.0 - std::f64::consts::SQRT_2) + 1.2 * pitch;
            let sb = min_sb.min(0.45 * l1).min(0.45 * l2);

            if sb > 0.05 {
                let p_a = (curr.0 - sb * u1.0, curr.1 - sb * u1.1);
                let p_b = (curr.0 + sb * u2.0, curr.1 + sb * u2.1);
                beveled_guide.push(p_a);
                beveled_guide.push(p_b);
                continue;
            }
        }

        beveled_guide.push(curr);
    }
    beveled_guide.push(*raw_points.last().unwrap());

    if beveled_guide.len() < 2 {
        return Vec::new();
    }

    // 3. Compute parallel lane tracks for each net in the bundle
    let num_segs = beveled_guide.len() - 1;
    let mut seg_unit = Vec::with_capacity(num_segs);
    let mut seg_norm = Vec::with_capacity(num_segs);

    for k in 0..num_segs {
        let p0 = beveled_guide[k];
        let p1 = beveled_guide[k + 1];
        let dx = p1.0 - p0.0;
        let dy = p1.1 - p0.1;
        let len = dx.hypot(dy).max(1e-6);
        let u = (dx / len, dy / len);
        let n = (-u.1, u.0);
        seg_unit.push(u);
        seg_norm.push(n);
    }

    let mut all_tracks = Vec::new();

    for (lane_idx, &net_id) in bundle.net_ids.iter().enumerate() {
        let offset = (lane_idx as f64 - half_span) * pitch;

        // Generate lane vertices along the beveled guide
        let mut lane_pts: Vec<(f64, f64)> = Vec::with_capacity(beveled_guide.len());

        // First point:
        let start_v = (
            beveled_guide[0].0 + offset * seg_norm[0].0,
            beveled_guide[0].1 + offset * seg_norm[0].1,
        );
        lane_pts.push(start_v);

        // Intermediate corners:
        for k in 1..num_segs {
            let u_prev = seg_unit[k - 1];
            let n_prev = seg_norm[k - 1];
            let u_curr = seg_unit[k];
            let n_curr = seg_norm[k];

            let a0 = (
                beveled_guide[k - 1].0 + offset * n_prev.0,
                beveled_guide[k - 1].1 + offset * n_prev.1,
            );
            let b0 = (
                beveled_guide[k].0 + offset * n_curr.0,
                beveled_guide[k].1 + offset * n_curr.1,
            );

            let d = u_prev.0 * u_curr.1 - u_prev.1 * u_curr.0;
            if d.abs() < 1e-6 {
                lane_pts.push((
                    beveled_guide[k].0 + offset * n_curr.0,
                    beveled_guide[k].1 + offset * n_curr.1,
                ));
            } else {
                let t = ((b0.0 - a0.0) * u_curr.1 - (b0.1 - a0.1) * u_curr.0) / d;
                let inter = (a0.0 + t * u_prev.0, a0.1 + t * u_prev.1);
                lane_pts.push(inter);
            }
        }

        // Last point:
        let end_v = (
            beveled_guide[num_segs].0 + offset * seg_norm[num_segs - 1].0,
            beveled_guide[num_segs].1 + offset * seg_norm[num_segs - 1].1,
        );
        lane_pts.push(end_v);

        // Sanitize lane points: deduplicate
        let mut clean_lane_pts: Vec<(f64, f64)> = Vec::new();
        for &pt in &lane_pts {
            if let Some(&last) = clean_lane_pts.last() {
                if (pt.0 - last.0).hypot(pt.1 - last.1) < 1e-4 {
                    continue;
                }
            }
            clean_lane_pts.push(pt);
        }

        // 4. Terminal fan-in and fan-out connection to pads
        let mut matching_pads: Vec<(f64, f64)> = Vec::new();
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id == net_id && (pad.is_through_hole || fp.layer == bundle.layer) {
                    let p_pos = fp.pad_absolute_position(pad);
                    if !matching_pads.iter().any(|m: &(f64, f64)| (m.0 - p_pos.0).hypot(m.1 - p_pos.1) < 1e-4) {
                        matching_pads.push(p_pos);
                    }
                }
            }
        }

        let p0_guide = beveled_guide[0];
        let pk_guide = *beveled_guide.last().unwrap();

        let mut start_pad: Option<(f64, f64)> = None;
        let mut end_pad: Option<(f64, f64)> = None;

        if matching_pads.len() >= 2 {
            let mut sorted = matching_pads.clone();
            sorted.sort_by(|a, b| {
                let d_a = (a.0 - p0_guide.0).hypot(a.1 - p0_guide.1);
                let d_b = (b.0 - p0_guide.0).hypot(b.1 - p0_guide.1);
                d_a.partial_cmp(&d_b).unwrap_or(Ordering::Equal)
            });
            start_pad = Some(sorted[0]);

            let mut sorted_end = matching_pads.clone();
            sorted_end.retain(|p| (p.0 - sorted[0].0).hypot(p.1 - sorted[0].1) > 1e-4);
            sorted_end.sort_by(|a, b| {
                let d_a = (a.0 - pk_guide.0).hypot(a.1 - pk_guide.1);
                let d_b = (b.0 - pk_guide.0).hypot(b.1 - pk_guide.1);
                d_a.partial_cmp(&d_b).unwrap_or(Ordering::Equal)
            });
            if !sorted_end.is_empty() {
                end_pad = Some(sorted_end[0]);
            }
        } else if matching_pads.len() == 1 {
            let p = matching_pads[0];
            let d0 = (p.0 - p0_guide.0).hypot(p.1 - p0_guide.1);
            let dk = (p.0 - pk_guide.0).hypot(p.1 - pk_guide.1);
            if d0 < dk {
                start_pad = Some(p);
            } else {
                end_pad = Some(p);
            }
        }

        let mut full_chain: Vec<(f64, f64)> = Vec::new();

        // Prepend fan-in
        if let Some(sp) = start_pad {
            if let Some(&lane_start) = clean_lane_pts.first() {
                let fanin = compute_45_dogleg(sp, lane_start, false);
                for pt in fanin {
                    if full_chain.last().map_or(true, |&last| (last.0 - pt.0).hypot(last.1 - pt.1) > 1e-4) {
                        full_chain.push(pt);
                    }
                }
            }
        }

        // Add spine lane points
        for pt in clean_lane_pts {
            if full_chain.last().map_or(true, |&last| (last.0 - pt.0).hypot(last.1 - pt.1) > 1e-4) {
                full_chain.push(pt);
            }
        }

        // Append fan-out
        if let Some(ep) = end_pad {
            if let Some(&lane_end) = full_chain.last() {
                let fanout = compute_45_dogleg(lane_end, ep, true);
                for pt in fanout {
                    if full_chain.last().map_or(true, |&last| (last.0 - pt.0).hypot(last.1 - pt.1) > 1e-4) {
                        full_chain.push(pt);
                    }
                }
            }
        }

        // Convert full chain to TrackSegments
        for w in 0..full_chain.len().saturating_sub(1) {
            let p_start = full_chain[w];
            let p_end = full_chain[w + 1];
            if (p_start.0 - p_end.0).hypot(p_start.1 - p_end.1) > 1e-4 {
                all_tracks.push(TrackSegment {
                    start: p_start,
                    end: p_end,
                    width: bundle.track_width,
                    layer: bundle.layer,
                    net_id,
                    is_locked: false,
                });
            }
        }
    }

    all_tracks
}

/// Routes an entire bus bundle along a user-sketched guide polyline using equidistant river routing.
/// Offsets each net symmetrically from the guide spine, applies 45° concentric corner miters,
/// connects terminal endpoints to component pads, and runs DRC checks with dynamic bottleneck compression.
pub fn route_bundle_along_guide(
    board: &mut Board,
    grid: &mut RoutingGrid,
    bundle: &BusBundle,
) -> bool {
    if bundle.guide_points.len() < 2 {
        return false;
    }
    if bundle.net_ids.is_empty() {
        return true;
    }

    // Determine min design clearance for nets in this bundle
    let min_class_clearance = bundle
        .net_ids
        .iter()
        .map(|&nid| board.get_netclass_for_net(nid).clearance)
        .fold(0.12_f64, f64::max);

    // Try nominal spacing first, then dynamically compress if localized bottlenecks exist
    let mut candidate_spacings = vec![bundle.track_spacing];
    if bundle.track_spacing > min_class_clearance + 0.005 {
        let steps = 6;
        for s in 1..=steps {
            let frac = s as f64 / steps as f64;
            let comp = bundle.track_spacing - frac * (bundle.track_spacing - min_class_clearance);
            candidate_spacings.push(comp);
        }
    }

    for spacing in candidate_spacings {
        let tracks = generate_bundle_tracks(board, bundle, spacing);
        if tracks.is_empty() {
            continue;
        }

        // Check DRC legality of all generated segments against foreign obstacles
        let mut legal = true;
        for trk in &tracks {
            let clearance = board.get_netclass_for_net(trk.net_id).clearance.max(0.12);
            if !is_track_segment_legal(
                board,
                trk.start,
                trk.end,
                trk.width,
                trk.layer,
                trk.net_id,
                clearance,
            ) {
                legal = false;
                break;
            }
        }

        if legal {
            // Success! Commit tracks to board
            board.tracks.extend(tracks);
            if !board.bundles.iter().any(|b| b.name == bundle.name && b.guide_points == bundle.guide_points) {
                board.bundles.push(bundle.clone());
            }
            grid.rasterize_obstacles(board);
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Footprint, KeepoutZone, Pad, Polygon};

    fn make_fp(reference: &str, x: f64, y: f64, is_locked: bool, pads: Vec<Pad>) -> Footprint {
        Footprint {
            reference: reference.to_string(),
            value: "V".to_string(),
            x,
            y,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked,
            pads,
            width: 2.0,
            height: 2.0,
            courtyards: vec![Polygon::new(vec![
                Point::new(-1.0, -1.0),
                Point::new(1.0, -1.0),
                Point::new(1.0, 1.0),
                Point::new(-1.0, 1.0),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        }
    }

    fn make_router_test_board() -> Board {
        let nets = vec![
            "".to_string(),
            "GND".to_string(),
            "SIG1".to_string(),
            "SIG2".to_string(),
            "U2-SW".to_string(),
        ];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        netclasses.insert(
            "Switching".to_string(),
            NetClass {
                name: "Switching".to_string(),
                track_width: 0.40,
                clearance: 0.25,
                via_dia: 0.60,
                via_drill: 0.30,
                diff_pair_gap: 0.20,
                diff_pair_width: 0.20,
                topology: NetTopology::default(),
                ..Default::default()
            },
        );
        let mut net_to_class = HashMap::new();
        net_to_class.insert("".to_string(), "Default".to_string());
        net_to_class.insert("GND".to_string(), "Default".to_string());
        net_to_class.insert("SIG1".to_string(), "Default".to_string());
        net_to_class.insert("SIG2".to_string(), "Default".to_string());
        net_to_class.insert("U2-SW".to_string(), "Switching".to_string());

        let u1_pads: Vec<Pad> = [
            ("1", -1.5, -1.0, 2, "SIG1"),
            ("2", -1.5, -0.5, 3, "SIG2"),
            ("3", -1.5, 0.0, 2, "SIG1"),
            ("4", -1.5, 0.5, 3, "SIG2"),
            ("6", -1.5, 1.0, 2, "SIG1"),
            ("7", -0.5, 1.5, 2, "SIG1"),
            ("8", 0.0, 1.5, 3, "SIG2"),
            ("9", 0.5, 1.5, 2, "SIG1"),
            ("10", 1.0, 1.5, 3, "SIG2"),
            ("12", 1.5, 1.5, 2, "SIG1"),
            ("13", 1.5, 0.5, 2, "SIG1"),
            ("15", 1.5, 0.0, 1, "GND"),
            ("16", 1.5, -0.5, 3, "SIG2"),
            ("17", 1.5, -1.0, 2, "SIG1"),
            ("18", 1.5, -1.5, 1, "GND"),
            ("19", 0.5, -1.5, 1, "GND"),
            ("20", 0.0, -1.5, 2, "SIG1"),
            ("21", -0.5, -1.5, 3, "SIG2"),
            ("22", -1.0, -1.5, 1, "GND"),
            ("23", -1.2, -1.5, 2, "SIG1"),
            ("24", -1.4, -1.5, 3, "SIG2"),
        ]
        .iter()
        .map(|&(num, ox, oy, nid, nname)| Pad::new(num, ox, oy, 0.25, 0.25, nid, nname, false))
        .collect();

        let j2_pads = vec![
            Pad::new("A5", 0.0, -0.5, 0.3, 0.3, 2, "SIG1", false),
            Pad::new("A6", 0.0, 0.5, 0.3, 0.3, 3, "SIG2", false),
            Pad::new("S1", 0.0, 1.5, 1.0, 1.0, 0, "unconnected", true),
        ];

        let c14 = make_fp(
            "C14",
            142.0,
            95.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "GND", false)],
        );

        let r14 = make_fp(
            "R14",
            146.0,
            95.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "GND", false)],
        );

        let r_gnd = make_fp(
            "R20",
            150.0,
            95.0,
            false,
            vec![
                Pad::new("1", -1.0, 0.0, 0.6, 0.6, 1, "GND", false),
                Pad::new("2", 1.0, 0.0, 0.6, 0.6, 4, "U2-SW", false),
            ],
        );

        Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 130.0,
            min_y: 90.0,
            max_x: 160.0,
            max_y: 115.0,
            footprints: vec![
                make_fp("U1", 145.0, 102.0, false, u1_pads),
                make_fp("J2", 135.0, 102.0, true, j2_pads),
                c14,
                r14,
                r_gnd,
            ],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![KeepoutZone {
                min_x: 156.0,
                min_y: 110.0,
                max_x: 159.0,
                max_y: 114.0,
                layers: vec![0, 1],
            }],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        }
    }

    #[test]
    fn test_routing_grid_coords_rasterize_and_neighbors() {
        let board = make_router_test_board();
        let vk = VulkanEngine::new();
        let grid = RoutingGrid::new(&board, 0.25, vk);

        assert!(grid.world_to_grid(129.0, 95.0).is_none());
        assert!(grid.world_to_grid(161.0, 95.0).is_none());
        let (gx, gy) = grid.world_to_grid(140.0, 100.0).unwrap();
        let (wx, wy) = grid.grid_to_world(gx, gy);
        assert!((wx - 140.125).abs() < 1e-4 && (wy - 100.125).abs() < 1e-4);

        // Verify keepout zone was rasterized as KEEPOUT_NET
        let (kgx, kgy) = grid.world_to_grid(157.0, 112.0).unwrap();
        assert_eq!(grid.obstacle_net_ids[grid.idx(kgx, kgy, 0)], KEEPOUT_NET);

        // Verify get_neighbors on free space (planar + via transition) vs near U1 center
        let free_node = GridNode { gx, gy, layer: 0 };
        let nbrs = grid.get_neighbors(free_node, 2, 0.20, 0.20, 0.60, Some((140.0, 100.0)));
        assert!(!nbrs.is_empty());
        assert!(nbrs.iter().any(|(n, _)| n.layer == 1));
    }

    #[test]
    fn test_nudge_engine_and_astar_node_ordering() {
        let mut board = make_router_test_board();
        let router = AutoRouter::new(4);
        assert!((router.nudge_engine.step_mm - 0.25).abs() < 1e-6);
        assert!((router.nudge_engine.max_displacement_mm - 2.0).abs() < 1e-6);

        // Out of range fp_idx, locked fp (J2 at idx 1), and zero push vector must return false
        assert!(!router.nudge_engine.try_nudge_component(&mut board, 999, (1.0, 0.0)));
        assert!(!router.nudge_engine.try_nudge_component(&mut board, 1, (1.0, 0.0)));
        assert!(!router.nudge_engine.try_nudge_component(&mut board, 2, (0.0, 0.0)));

        // Add a track connecting C14 (idx 2) to R14 (idx 3)
        board.tracks.push(TrackSegment {
            start: (142.0, 95.0),
            end: (146.0, 95.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        assert_eq!(board.tracks.len(), 1);

        // Valid nudge of unlocked C14 (idx 2)
        let old_x = board.footprints[2].x;
        assert!(router.nudge_engine.try_nudge_component(&mut board, 2, (1.0, 0.0)));
        assert!((board.footprints[2].x - (old_x + 0.25)).abs() < 1e-6);
        // Trace connected to C14 must have been ripped up!
        assert_eq!(board.tracks.len(), 0);

        // AStarNode min-heap ordering check
        let mut heap = BinaryHeap::new();
        heap.push(AStarNode {
            node: GridNode { gx: 0, gy: 0, layer: 0 },
            cost: 5.0,
            priority: 10.0,
        });
        heap.push(AStarNode {
            node: GridNode { gx: 1, gy: 1, layer: 0 },
            cost: 1.0,
            priority: 2.0,
        });
        assert_eq!(heap.pop().unwrap().node.gx, 1);
    }

    #[test]
    fn test_escape_stubs_connection_extraction_and_priority() {
        assert_eq!(AutoRouter::netclass_priority("Switching"), 10);
        assert_eq!(AutoRouter::netclass_priority("Power_15V"), 9);
        assert_eq!(AutoRouter::netclass_priority("USB_Diff"), 8);
        assert_eq!(AutoRouter::netclass_priority("HBS_Bus"), 7);
        assert_eq!(AutoRouter::netclass_priority("Power_3V3"), 6);
        assert_eq!(AutoRouter::netclass_priority("Default"), 1);

        let board = make_router_test_board();
        let router = AutoRouter::new(3);
        let stubs = router.generate_escape_stubs(&board);
        assert_eq!(stubs.len(), 19); // 17 U1 non-GND pins + 2 J2 SMD signal pins

        let conns = router.extract_connections(&board);
        assert!(!conns.is_empty());
        for c in &conns {
            assert!(c.start_fp < board.footprints.len());
            assert!(c.end_fp < board.footprints.len());
            assert!(c.start_pad < board.footprints[c.start_fp].pads.len());
            assert!(c.end_pad < board.footprints[c.end_fp].pads.len());
        }
    }

    #[test]
    fn test_escape_stubs_direction_and_pad_clearance() {
        let mut board = make_router_test_board();
        let router = AutoRouter::new(3);

        // Test with 0, 90, 180, 270 degree rotation of U1
        for rot in [0.0, 90.0, 180.0, 270.0] {
            board.footprints[0].rotation = rot;
            let stubs = router.generate_escape_stubs(&board);
            let u1_stubs: Vec<_> = stubs.iter().filter(|s| {
                s.layer == board.footprints[0].layer && (s.pad_pos.0 - board.footprints[0].x).hypot(s.pad_pos.1 - board.footprints[0].y) < 5.0
            }).collect();
            assert_eq!(u1_stubs.len(), 17, "All 17 non-GND pins of U1 must generate escape stubs at rot={}", rot);

            for stub in &u1_stubs {
                // Verify stub projects outward away from footprint center
                let pad_dist = (stub.pad_pos.0 - board.footprints[0].x).hypot(stub.pad_pos.1 - board.footprints[0].y);
                let tip_dist = (stub.tip_pos.0 - board.footprints[0].x).hypot(stub.tip_pos.1 - board.footprints[0].y);
                assert!(tip_dist > pad_dist, "Escape stub tip must project outward away from footprint center (rot={})", rot);

                // Verify stub never shoots towards any neighboring pad
                for pad in &board.footprints[0].pads {
                    let (px, py) = board.footprints[0].pad_absolute_position(pad);
                    let pad_to_pad_dist = (px - stub.pad_pos.0).hypot(py - stub.pad_pos.1);
                    if pad_to_pad_dist < 1e-3 {
                        continue; // same pad
                    }
                    let d = dist_to_segment(px, py, stub.pad_pos.0, stub.pad_pos.1, stub.tip_pos.0, stub.tip_pos.1);
                    assert!(d >= pad_to_pad_dist - 1e-4, "Escape stub must not shoot towards pad {} on U1 (d={}, pad_to_pad={})", pad.number, d, pad_to_pad_dist);
                }
            }
        }
    }

    #[test]
    fn test_dedicated_plane_isolation() {
        let mut board = make_router_test_board();
        board.raw_content = "(\"In1.Cu\" \"In2.Cu\")".to_string();
        let mut grid = RoutingGrid::new(&board, 0.25, None);
        // Add dedicated inner plane layer 2 (GND = net 1) and layer 3 (PWR = net 4)
        grid.plane_nets.insert(2, 1);
        grid.plane_nets.insert(3, 4);

        let signal_net_id = 2; // SIG1
        let center_gx = grid.width / 2;
        let center_gy = grid.height / 2;

        // Test 1: Planar steps on dedicated plane layer 2 are prohibited for signal net
        let node_plane = GridNode { gx: center_gx, gy: center_gy, layer: 2 };
        let nbrs_plane = grid.get_neighbors(node_plane, signal_net_id, 0.2, 0.2, 0.6, None);
        let planar_steps: Vec<_> = nbrs_plane.iter().filter(|(n, _)| n.layer == 2).collect();
        assert!(planar_steps.is_empty(), "Signal net must not take planar steps on dedicated plane layer 2");

        // Test 2: Via transitions from layer 0 cannot transition to dedicated plane layer 2 or 3 for signal net
        let node_top = GridNode { gx: center_gx, gy: center_gy, layer: 0 };
        let nbrs_top = grid.get_neighbors(node_top, signal_net_id, 0.2, 0.2, 0.6, None);
        for (nbr, _) in &nbrs_top {
            assert_ne!(nbr.layer, 2, "Signal net must not transition via to plane layer 2");
            assert_ne!(nbr.layer, 3, "Signal net must not transition via to plane layer 3");
        }

        // Test 3: Plane net itself (GND = 1) CAN take planar steps on its own plane layer
        let nbrs_gnd_plane = grid.get_neighbors(node_plane, 1, 0.2, 0.2, 0.6, None);
        let gnd_planar_steps: Vec<_> = nbrs_gnd_plane.iter().filter(|(n, _)| n.layer == 2).collect();
        assert!(!gnd_planar_steps.is_empty(), "Plane net must be allowed to route planar steps on its plane layer");
    }

    #[test]
    fn test_auto_router_route_all_ripup_and_ground_stitching() {
        // Simple 2-net board where net 2 and net 3 cross, exercising A*, layer transitions,
        // rip-up & reroute, U1 ground pin stubs, and ground via stitching.
        let mut board = make_router_test_board();
        // Keep only a compact set of pads on U1 and J2 so routing finishes in <30ms
        board.footprints[0].pads.retain(|p| matches!(p.number.as_str(), "1" | "2" | "15" | "19"));
        let mut grid = RoutingGrid::new(&board, 0.25, None);
        let router = AutoRouter::new(2);
        let live = Arc::new(Mutex::new(LiveTrialState::new(board.clone())));
        let stop = Arc::new(AtomicBool::new(false));

        let (routed, total) = router.route_all(&mut board, &mut grid, Some(Arc::clone(&live)), stop);
        assert!(total >= 2);
        assert!(routed <= total);
        assert!(!board.tracks.is_empty());
        assert!(!board.vias.is_empty());

        let g = live.lock().unwrap();
        assert_eq!(g.phase, TrialPhase::Completed);
        assert_eq!(g.total_conns, total);
    }

    #[test]
    fn test_direction_change_penalty_and_staircase_smoothing() {
        let board = make_router_test_board();
        let grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(2);

        // Construct an alternating staircase path: (+1,0), (+1,+1), (+1,0), (+1,+1), (+1,0)
        // around an open region of the board (e.g. gx=10..15, gy=10..12)
        let zigzag = vec![
            GridNode { gx: 10, gy: 10, layer: 1 },
            GridNode { gx: 11, gy: 10, layer: 1 },
            GridNode { gx: 12, gy: 11, layer: 1 },
            GridNode { gx: 13, gy: 11, layer: 1 },
            GridNode { gx: 14, gy: 12, layer: 1 },
            GridNode { gx: 15, gy: 12, layer: 1 },
        ];
        assert_eq!(AutoRouter::count_path_bends(&zigzag), 4);

        let smoothed = router.smooth_grid_path(&grid, zigzag, 2, 0.20, 0.20, 0.60, None, false, 15.0);
        assert_eq!(smoothed.first().unwrap().gx, 10);
        assert_eq!(smoothed.last().unwrap().gx, 15);
        assert_eq!(
            AutoRouter::count_path_bends(&smoothed),
            1,
            "Staircase path should collapse into a single 45-degree bend"
        );
    }

    #[test]
    fn test_diff_pair_mates_and_coupled_bus_routing() {
        assert!(AutoRouter::are_diff_pair_mates("USB_DP", "USB_DM"));
        assert!(AutoRouter::are_diff_pair_mates("USB_D+", "USB_D-"));
        assert!(AutoRouter::are_diff_pair_mates("ETH_TX_P", "ETH_TX_N"));
        assert!(!AutoRouter::are_diff_pair_mates("USB_DP", "USB_DP"));
        assert!(!AutoRouter::are_diff_pair_mates("SIG1", "SIG2"));

        let mut board = make_router_test_board();
        board.nets = vec![
            "".to_string(),
            "GND".to_string(),
            "USB_DP".to_string(),
            "USB_DM".to_string(),
        ];
        board.footprints[0].pads = vec![
            Pad::new("13", -1.0, -0.4, 0.3, 0.3, 2, "USB_DP", false),
            Pad::new("14", -1.0, 0.4, 0.3, 0.3, 3, "USB_DM", false),
        ];
        board.footprints[1].pads = vec![
            Pad::new("A6", 1.0, -0.4, 0.3, 0.3, 2, "USB_DP", false),
            Pad::new("A7", 1.0, 0.4, 0.3, 0.3, 3, "USB_DM", false),
        ];
        board.footprints.truncate(2);

        let router = AutoRouter::new(2);
        let conns = router.extract_connections(&board);
        assert_eq!(conns.len(), 2);
        assert_eq!(conns[0].diff_pair_leader, None);
        assert_eq!(conns[1].diff_pair_leader, Some(0));

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let stop = Arc::new(AtomicBool::new(false));
        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);
        assert_eq!(routed, 2);
        assert_eq!(total, 2);

        // Verify the two coupled traces stay close to each other at their midpoints
        let dp_tracks: Vec<&TrackSegment> = board.tracks.iter().filter(|t| t.net_id == 2).collect();
        let dm_tracks: Vec<&TrackSegment> = board.tracks.iter().filter(|t| t.net_id == 3).collect();
        assert!(!dp_tracks.is_empty() && !dm_tracks.is_empty());
        let dp_mid_y: f64 = dp_tracks.iter().map(|t| (t.start.1 + t.end.1) * 0.5).sum::<f64>()
            / dp_tracks.len() as f64;
        let dm_mid_y: f64 = dm_tracks.iter().map(|t| (t.start.1 + t.end.1) * 0.5).sum::<f64>()
            / dm_tracks.len() as f64;
        assert!(
            (dp_mid_y - dm_mid_y).abs() < 1.8,
            "Coupled USB_DP/USB_DM traces should route together in a tight corridor (delta_y = {:.2})",
            (dp_mid_y - dm_mid_y).abs()
        );
    }

    #[test]
    fn test_coupled_diff_pair_gap_and_via_symmetry() {
        let mut board = make_router_test_board();
        board.nets = vec![
            "".to_string(),
            "GND".to_string(),
            "USB_DP".to_string(),
            "USB_DM".to_string(),
        ];
        board.netclasses.insert(
            "USB_Diff".to_string(),
            NetClass {
                name: "USB_Diff".to_string(),
                track_width: 0.20,
                clearance: 0.20,
                via_dia: 0.60,
                via_drill: 0.30,
                diff_pair_gap: 0.20,
                diff_pair_width: 0.20,
                topology: NetTopology::default(),
                ..Default::default()
            },
        );
        board.net_to_class.insert("USB_DP".to_string(), "USB_Diff".to_string());
        board.net_to_class.insert("USB_DM".to_string(), "USB_Diff".to_string());
        board.footprints[0].pads = vec![
            Pad::new("13", -1.0, -0.4, 0.3, 0.3, 2, "USB_DP", false),
            Pad::new("14", -1.0, 0.4, 0.3, 0.3, 3, "USB_DM", false),
        ];
        board.footprints[1].pads = vec![
            Pad::new("A6", 1.0, -0.4, 0.3, 0.3, 2, "USB_DP", false),
            Pad::new("A7", 1.0, 0.4, 0.3, 0.3, 3, "USB_DM", false),
        ];
        board.footprints.truncate(2);

        let router = AutoRouter::new(2);
        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let stop = Arc::new(AtomicBool::new(false));
        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);
        assert_eq!(routed, 2);
        assert_eq!(total, 2);

        // Verify differential pair track widths (0.20mm for main tracks, 0.15mm for fine-pitch escape stubs)
        let dp_tracks: Vec<&TrackSegment> = board.tracks.iter().filter(|t| t.net_id == 2).collect();
        let dm_tracks: Vec<&TrackSegment> = board.tracks.iter().filter(|t| t.net_id == 3).collect();
        assert!(!dp_tracks.is_empty() && !dm_tracks.is_empty());
        assert!(dp_tracks.iter().any(|t| (t.width - 0.20).abs() < 1e-4));
        for t in &dp_tracks {
            assert!((t.width - 0.20).abs() < 1e-4 || (t.width - 0.15).abs() < 1e-4);
        }
        for t in &dm_tracks {
            assert!((t.width - 0.20).abs() < 1e-4 || (t.width - 0.15).abs() < 1e-4);
        }

        // Verify that parallel segments maintain centerline distance W + S = 0.40mm
        // (edge-to-edge gap diff_pair_gap = 0.20mm)
        let mut found_coupled_parallel = false;
        for t_p in &dp_tracks {
            for t_n in &dm_tracks {
                if t_p.layer == t_n.layer {
                    let p_horiz = (t_p.start.1 - t_p.end.1).abs() < 1e-4;
                    let n_horiz = (t_n.start.1 - t_n.end.1).abs() < 1e-4;
                    if p_horiz && n_horiz {
                        let center_dist = (t_p.start.1 - t_n.start.1).abs();
                        if (center_dist - 0.40).abs() < 0.05 || (center_dist - 0.80).abs() < 0.05 {
                            found_coupled_parallel = true;
                            let gap = center_dist - t_p.width;
                            assert!(gap >= 0.15, "Differential pair gap must be >= 0.15mm, got {:.3}", gap);
                        }
                    }
                }
            }
        }
        assert!(found_coupled_parallel, "Must have coupled parallel segments with W+S centerline distance");

        // Verify symmetric paired vias side-by-side with matched spacing
        let mut board_vias = make_router_test_board();
        board_vias.nets = board.nets.clone();
        board_vias.netclasses = board.netclasses.clone();
        board_vias.net_to_class = board.net_to_class.clone();
        board_vias.footprints[0].pads = vec![
            Pad::new("13", -1.0, -0.4, 0.3, 0.3, 2, "USB_DP", false),
            Pad::new("14", -1.0, 0.4, 0.3, 0.3, 3, "USB_DM", false),
        ];
        board_vias.footprints[1].layer = 1;
        board_vias.footprints[1].pads = vec![
            Pad::new("A6", 1.0, -0.4, 0.3, 0.3, 2, "USB_DP", false),
            Pad::new("A7", 1.0, 0.4, 0.3, 0.3, 3, "USB_DM", false),
        ];
        board_vias.footprints.truncate(2);

        let mut grid_vias = RoutingGrid::new(&board_vias, 0.20, None);
        let stop_v = Arc::new(AtomicBool::new(false));
        let (r_v, _) = router.route_all(&mut board_vias, &mut grid_vias, None, stop_v);
        if r_v == 2 {
            let dp_vias: Vec<&Via> = board_vias.vias.iter().filter(|v| v.net_id == 2).collect();
            let dm_vias: Vec<&Via> = board_vias.vias.iter().filter(|v| v.net_id == 3).collect();
            if !dp_vias.is_empty() && !dm_vias.is_empty() {
                let via_dist = (dp_vias[0].position.0 - dm_vias[0].position.0).hypot(dp_vias[0].position.1 - dm_vias[0].position.1);
                assert!(via_dist >= 0.79, "Paired vias must maintain matched DRC clearance >= 0.80mm, got {:.3}", via_dist);
                assert_eq!(dp_vias[0].size, dm_vias[0].size);
                assert_eq!(dp_vias[0].drill, dm_vias[0].drill);
            }
        }
    }

    #[test]
    fn test_intra_pair_phase_matching_corrects_skew() {
        let mut board = make_router_test_board();
        board.nets = vec!["".to_string(), "GND".to_string(), "USB_DP".to_string(), "USB_DM".to_string()];
        board.netclasses.insert(
            "USB_Diff".to_string(),
            NetClass {
                name: "USB_Diff".to_string(),
                track_width: 0.20,
                clearance: 0.20,
                via_dia: 0.60,
                via_drill: 0.30,
                diff_pair_gap: 0.20,
                diff_pair_width: 0.20,
                topology: NetTopology::default(),
                ..Default::default()
            },
        );
        board.net_to_class.insert("USB_DP".to_string(), "USB_Diff".to_string());
        board.net_to_class.insert("USB_DM".to_string(), "USB_Diff".to_string());
        let net_p = 2;
        let net_n = 3;

        // Long trace for P (length = 10.0mm in clear board routing area)
        board.tracks.push(TrackSegment {
            start: (135.0, 107.0),
            end: (145.0, 107.0),
            width: 0.20,
            layer: 0,
            net_id: net_p,
            is_locked: false,
        });

        // Shorter trace for N (length = 9.4mm, so skew = 0.60mm > 0.05mm)
        board.tracks.push(TrackSegment {
            start: (135.0, 107.4),
            end: (144.4, 107.4),
            width: 0.20,
            layer: 0,
            net_id: net_n,
            is_locked: false,
        });

        let calc_len = |tracks: &[TrackSegment], nid: usize| -> f64 {
            tracks.iter().filter(|t| t.net_id == nid).map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1)).sum()
        };

        let initial_skew = (calc_len(&board.tracks, net_p) - calc_len(&board.tracks, net_n)).abs();
        assert!((initial_skew - 0.60).abs() < 1e-4, "Initial skew must be 0.60mm");

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let ok = match_diff_pair_phase(&mut board, &mut grid, net_p, net_n, 0.05);
        assert!(ok, "Phase matching must succeed");

        let final_skew = (calc_len(&board.tracks, net_p) - calc_len(&board.tracks, net_n)).abs();
        assert!(
            final_skew <= 0.05,
            "Intra-pair phase matching must equalize trace lengths within +/- 0.05mm, got final skew {:.4}mm",
            final_skew
        );
        let n_segs = board.tracks.iter().filter(|t| t.net_id == net_n).count();
        assert!(n_segs > 1, "Shorter trace must have tuning bump segments inserted");
    }

    #[test]
    fn test_trombone_serpentine_length_tuning() {
        let mut board = make_router_test_board();
        board.nets = vec!["".to_string(), "GND".to_string(), "SDIO_CLK".to_string()];
        board.net_to_class.insert("SDIO_CLK".to_string(), "Default".to_string());
        let net_id = 2;

        // Straight track of length 8.0mm in clear board routing area
        board.tracks.push(TrackSegment {
            start: (135.0, 107.0),
            end: (143.0, 107.0),
            width: 0.20,
            layer: 0,
            net_id,
            is_locked: false,
        });

        let calc_len = |tracks: &[TrackSegment]| -> f64 {
            tracks.iter().filter(|t| t.net_id == net_id).map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1)).sum()
        };

        let initial_len = calc_len(&board.tracks);
        assert!((initial_len - 8.0).abs() < 1e-4);

        let target_len = 11.50; // Needs +3.5mm of trombone/accordion tuning
        let mut grid = RoutingGrid::new(&board, 0.20, None);

        let success = tune_net_length(&mut board, &mut grid, net_id, target_len, 1.20);
        assert!(success, "Trombone/accordion tuning must succeed");

        let final_len = calc_len(&board.tracks);
        assert!(
            final_len >= target_len - 0.05,
            "Final tuned length must reach target ({}mm +/- 0.05mm), got {:.3}mm",
            target_len, final_len
        );

        let n_segs = board.tracks.iter().filter(|t| t.net_id == net_id).count();
        assert!(n_segs >= 5, "Accordion pattern must replace straight segment with multiple meander segments, got {}", n_segs);
    }

    #[test]
    fn test_nudge_unlocked_component_frees_blocked_routing_corridor() {
        let nets = vec!["".to_string(), "GND".to_string(), "SIG1".to_string(), "BLOCK".to_string()];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        for n in &nets {
            net_to_class.insert(n.clone(), "Default".to_string());
        }

        // Two endpoints for SIG1 (net 2): J1 at (105.0, 110.0) and J2 at (125.0, 110.0)
        let j1 = make_fp("J1", 105.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "SIG1", false)]);
        let j2 = make_fp("J_END", 125.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "SIG1", false)]);

        // Unlocked component C1 (net 3) placed in the channel in front of J1
        let mut c1 = make_fp("C1", 107.0, 110.0, false, vec![Pad::new("1", 0.0, 0.0, 0.6, 0.6, 3, "BLOCK", false)]);
        c1.courtyards = vec![Polygon::new(vec![
            Point::new(-0.35, -0.35),
            Point::new(0.35, -0.35),
            Point::new(0.35, 0.35),
            Point::new(-0.35, 0.35),
        ])];

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 104.0,
            min_y: 100.0,
            max_x: 130.0,
            max_y: 120.0,
            footprints: vec![j1, j2, c1],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            // Keepouts constrict J1 so it can only escape in +x through the narrow bottleneck blocked by C1
            keepouts: vec![
                KeepoutZone {
                    min_x: 104.0,
                    min_y: 110.5,
                    max_x: 108.0,
                    max_y: 120.0,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 104.0,
                    min_y: 100.0,
                    max_x: 108.0,
                    max_y: 109.5,
                    layers: vec![0, 1],
                },
            ],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(3).with_allow_component_nudge(true);
        let stop = Arc::new(AtomicBool::new(false));

        let orig_c1_x = board.footprints[2].x;
        let orig_c1_y = board.footprints[2].y;

        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);

        // Verify C1 was nudged away from its blocking position
        let new_c1_x = board.footprints[2].x;
        let new_c1_y = board.footprints[2].y;
        assert!(
            (new_c1_x - orig_c1_x).hypot(new_c1_y - orig_c1_y) > 0.1,
            "Unlocked C1 must be nudged away to free the escape/corridor"
        );

        // Verify connection between J1 and J2 routed successfully
        assert_eq!(total, 1);
        assert_eq!(routed, 1, "Connection must route cleanly after nudging blocking component");
        assert!(!board.tracks.is_empty());
    }

    #[test]
    fn test_shove_and_ripup_blocking_unlocked_track() {
        // Part 1: Direct test of NudgeEngine::try_shove_track
        let mut board = make_router_test_board();
        let router = AutoRouter::new(3);

        board.tracks.push(TrackSegment {
            start: (140.0, 100.0),
            end: (150.0, 100.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });

        // Unlocked track should be shoved perpendicularly (+y)
        assert!(router.nudge_engine.try_shove_track(&mut board, 0, 1.0, None));
        assert!((board.tracks[0].start.1 - 100.25).abs() < 1e-4);
        assert!((board.tracks[0].end.1 - 100.25).abs() < 1e-4);

        // Locked track must not be shoved
        board.tracks[0].is_locked = true;
        assert!(!router.nudge_engine.try_shove_track(&mut board, 0, 1.0, None));

        // Part 2: Router active track shove / rip-up during route_all
        let nets = vec!["".to_string(), "GND".to_string(), "HIGH_PRIO".to_string(), "LOW_PRIO".to_string()];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        netclasses.insert("USB_Diff".to_string(), NetClass {
            name: "USB_Diff".to_string(),
            track_width: 0.20,
            clearance: 0.20,
            via_dia: 0.60,
            via_drill: 0.30,
            diff_pair_gap: 0.20,
            diff_pair_width: 0.20,
            topology: NetTopology::default(),
            ..Default::default()
        });
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("".to_string(), "Default".to_string());
        net_to_class.insert("GND".to_string(), "Default".to_string());
        net_to_class.insert("HIGH_PRIO".to_string(), "USB_Diff".to_string()); // Priority 8
        net_to_class.insert("LOW_PRIO".to_string(), "Default".to_string());   // Priority 1

        let hp_j1 = make_fp("HP1", 105.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "HIGH_PRIO", false)]);
        let hp_j2 = make_fp("HP2", 125.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "HIGH_PRIO", false)]);
        let lp_j1 = make_fp("LP1", 115.0, 105.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "LOW_PRIO", false)]);
        let lp_j2 = make_fp("LP2", 115.0, 115.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "LOW_PRIO", false)]);

        let mut route_board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 100.0,
            min_y: 100.0,
            max_x: 130.0,
            max_y: 120.0,
            footprints: vec![hp_j1, hp_j2, lp_j1, lp_j2],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&route_board, 0.20, None);
        let stop = Arc::new(AtomicBool::new(false));
        let (routed, total) = router.route_all(&mut route_board, &mut grid, None, stop);

        assert_eq!(total, 2);
        assert_eq!(routed, 2, "Both high and low priority signals must be successfully routed");
        let hp_tracks: Vec<&TrackSegment> = route_board.tracks.iter().filter(|t| t.net_id == 2).collect();
        assert!(!hp_tracks.is_empty(), "High-priority signal must have routed tracks");
    }

    #[test]
    fn test_nudge_engine_try_shove_via() {
        let mut board = make_router_test_board();
        let router = AutoRouter::new(3);

        // Add a via at (138.0, 108.0) in free board area (boundary is [130..160, 90..115])
        board.vias.push(Via {
            position: (138.0, 108.0),
            size: 0.60,
            drill: 0.30,
            net_id: 2,
            is_locked: false,
        });

        // Add two incident tracks connecting to the via (one on F.Cu, one on B.Cu)
        board.tracks.push(TrackSegment {
            start: (134.0, 108.0),
            end: (138.0, 108.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: (138.0, 108.0),
            end: (142.0, 108.0),
            width: 0.25,
            layer: 1,
            net_id: 2,
            is_locked: false,
        });

        // Shove the via in +y direction
        let ok = router.nudge_engine.try_shove_via(&mut board, 0, (0.0, 1.0), None);
        assert!(ok, "Unlocked via with incident tracks must be successfully shoved");
        assert!((board.vias[0].position.1 - 108.25).abs() < 1e-4 || (board.vias[0].position.1 - 108.50).abs() < 1e-4);

        // Verify incident tracks on both layers were updated to match the new via position
        let new_vpos = board.vias[0].position;
        assert_eq!(board.tracks[0].end, new_vpos, "Track on F.Cu must connect to new via position");
        assert_eq!(board.tracks[1].start, new_vpos, "Track on B.Cu must connect to new via position");

        // Locked via must not be shoved
        board.vias[0].is_locked = true;
        assert!(!router.nudge_engine.try_shove_via(&mut board, 0, (0.0, 1.0), None));
    }

    #[test]
    fn test_autorouter_shove_blocking_via_and_track() {
        let nets = vec!["".to_string(), "GND".to_string(), "NET_A".to_string(), "NET_B".to_string()];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass {
            name: "Default".to_string(),
            track_width: 0.25,
            clearance: 0.20,
            via_dia: 0.60,
            via_drill: 0.30,
            diff_pair_gap: 0.20,
            diff_pair_width: 0.20,
            topology: NetTopology::default(),
            ..Default::default()
        });
        let mut net_to_class = HashMap::new();
        net_to_class.insert("".to_string(), "Default".to_string());
        net_to_class.insert("GND".to_string(), "Default".to_string());
        net_to_class.insert("NET_A".to_string(), "Default".to_string());
        net_to_class.insert("NET_B".to_string(), "Default".to_string());

        // NET_A: Horizontal route from (105, 110) to (125, 110)
        let a1 = make_fp("A1", 105.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "NET_A", false)]);
        let a2 = make_fp("A2", 125.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "NET_A", false)]);

        // NET_B: Vertical route crossing at (115, 105) to (115, 115)
        let b1 = make_fp("B1", 115.0, 105.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "NET_B", false)]);
        let b2 = make_fp("B2", 115.0, 115.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "NET_B", false)]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 100.0,
            min_y: 100.0,
            max_x: 130.0,
            max_y: 120.0,
            footprints: vec![a1, a2, b1, b2],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(3);
        let stop = Arc::new(AtomicBool::new(false));
        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);

        assert_eq!(total, 2);
        assert_eq!(routed, 2, "Both crossing signals must be routed with shove-and-reroute");
        assert!(!board.tracks.is_empty());
    }

    #[test]
    fn test_net_topology_daisy_chain_and_fly_by() {
        let nc = NetClass::default();
        let pads = vec![
            PadInfo::new(0.0, 0.0, 0, 0, 0).with_driver(true),
            PadInfo::new(10.0, 0.0, 0, 1, 0),
            PadInfo::new(20.0, 0.0, 0, 2, 0),
            PadInfo::new(30.0, 0.0, 0, 3, 0),
        ];

        // 1. DaisyChain: verifies 4 pads connected in strict sequential order P0 -> P1 -> P2 -> P3 with 0 branching
        let daisy_conns = build_net_topology_connections(1, "SPI_BUS", &nc, &pads, NetTopology::DaisyChain);
        assert_eq!(daisy_conns.len(), 3);
        assert_eq!(daisy_conns[0].start, (0.0, 0.0, 0));
        assert_eq!(daisy_conns[0].end, (10.0, 0.0, 0));
        assert_eq!(daisy_conns[1].start, (10.0, 0.0, 0));
        assert_eq!(daisy_conns[1].end, (20.0, 0.0, 0));
        assert_eq!(daisy_conns[2].start, (20.0, 0.0, 0));
        assert_eq!(daisy_conns[2].end, (30.0, 0.0, 0));

        // Verify 0 branching (each internal pad has degree 2, endpoints have degree 1)
        let mut degree = vec![0; 4];
        for c in &daisy_conns {
            let s_idx = pads.iter().position(|p| (p.x, p.y, p.layer) == c.start).unwrap();
            let e_idx = pads.iter().position(|p| (p.x, p.y, p.layer) == c.end).unwrap();
            degree[s_idx] += 1;
            degree[e_idx] += 1;
        }
        assert_eq!(degree[0], 1, "Driver P0 must have degree 1");
        assert_eq!(degree[1], 2, "Intermediate P1 must have degree 2");
        assert_eq!(degree[2], 2, "Intermediate P2 must have degree 2");
        assert_eq!(degree[3], 1, "End receiver P3 must have degree 1");
        assert!(degree.iter().all(|&d| d <= 2), "Daisy chain must have zero branching");

        // 2. FlyBy: verifies sequential flight-time chain without stubs
        let flyby_conns = build_net_topology_connections(2, "DDR_ADDR", &nc, &pads, NetTopology::FlyBy);
        assert_eq!(flyby_conns.len(), 3);
        assert_eq!(flyby_conns[0].start, (0.0, 0.0, 0));
        assert_eq!(flyby_conns[0].end, (10.0, 0.0, 0));
        assert_eq!(flyby_conns[1].start, (10.0, 0.0, 0));
        assert_eq!(flyby_conns[1].end, (20.0, 0.0, 0));
        assert_eq!(flyby_conns[2].start, (20.0, 0.0, 0));
        assert_eq!(flyby_conns[2].end, (30.0, 0.0, 0));
    }

    #[test]
    fn test_net_topology_star_point() {
        let nc = NetClass::default();
        let star_center = (15.0, 15.0, 0);
        let pads = vec![
            PadInfo::new(15.0, 15.0, 0, 0, 0).with_driver(true), // Center / power source
            PadInfo::new(0.0, 0.0, 0, 1, 0),
            PadInfo::new(30.0, 0.0, 0, 2, 0),
            PadInfo::new(15.0, 30.0, 0, 3, 0),
        ];

        let conns = build_net_topology_connections(1, "+3.3V", &nc, &pads, NetTopology::StarPoint);
        assert_eq!(conns.len(), 3);

        // Verify all connections share the star center pad as start endpoint
        for c in &conns {
            assert_eq!(c.start, star_center);
            assert_eq!(c.start_fp, 0);
            assert_eq!(c.start_pad, 0);
        }

        let endpoints: Vec<(f64, f64, usize)> = conns.iter().map(|c| c.end).collect();
        assert!(endpoints.contains(&(0.0, 0.0, 0)));
        assert!(endpoints.contains(&(30.0, 0.0, 0)));
        assert!(endpoints.contains(&(15.0, 30.0, 0)));
    }

    #[test]
    fn test_net_topology_t_branch() {
        let nc = NetClass::default();
        // Driver at (0, 0), Receivers symmetrically placed at (20, 10) and (20, -10)
        let pads = vec![
            PadInfo::new(0.0, 0.0, 0, 0, 0).with_driver(true),
            PadInfo::new(20.0, 10.0, 0, 1, 0),
            PadInfo::new(20.0, -10.0, 0, 2, 0),
        ];

        let conns = build_net_topology_connections(1, "CLK_NET", &nc, &pads, NetTopology::TBranch);
        assert_eq!(conns.len(), 3); // 1 trunk + 2 symmetric branches

        // Trunk: Driver (0, 0) -> Split node (20, 0)
        let trunk = &conns[0];
        assert_eq!(trunk.start, (0.0, 0.0, 0));
        assert_eq!(trunk.end, (20.0, 0.0, 0));

        // Branches from split node to receivers
        let branch1 = &conns[1];
        let branch2 = &conns[2];
        assert_eq!(branch1.start, (20.0, 0.0, 0));
        assert_eq!(branch1.end, (20.0, 10.0, 0));
        assert_eq!(branch2.start, (20.0, 0.0, 0));
        assert_eq!(branch2.end, (20.0, -10.0, 0));

        // Verify symmetric balanced split with matched branch lengths
        let len1 = (branch1.start.0 - branch1.end.0).hypot(branch1.start.1 - branch1.end.1);
        let len2 = (branch2.start.0 - branch2.end.0).hypot(branch2.start.1 - branch2.end.1);
        assert!((len1 - len2).abs() < 1e-6);
        assert!((len1 - 10.0).abs() < 1e-6);
        assert!((len2 - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_net_topology_parsing_and_defaults() {
        assert_eq!(NetTopology::default(), NetTopology::EuclideanMst);
        let mut nc = NetClass::default();
        assert_eq!(nc.topology, NetTopology::EuclideanMst);

        // Verify heuristic detection
        assert_eq!(nc.resolve_topology("SYS_CLK"), NetTopology::TBranch);
        assert_eq!(nc.resolve_topology("DDR4_ADDR1"), NetTopology::FlyBy);
        assert_eq!(nc.resolve_topology("I2C_SCL"), NetTopology::DaisyChain);
        assert_eq!(nc.resolve_topology("+5V"), NetTopology::StarPoint);
        assert_eq!(nc.resolve_topology("SIG_GENERAL"), NetTopology::EuclideanMst);

        // Verify explicit setting overrides heuristic
        nc.topology = NetTopology::StarPoint;
        assert_eq!(nc.resolve_topology("SYS_CLK"), NetTopology::StarPoint);
    }

    #[test]
    fn test_post_route_glossing_straightens_rubber_band() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        // A track on SIG1 with an unnecessary V-shaped detour in open space:
        // (10.0, 10.0) -> (15.0, 15.0) -> (20.0, 10.0)
        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (15.0, 15.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 15.0),
                    end: (20.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(3);

        let initial_len: f64 = board.tracks.iter().map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1)).sum();
        assert!((initial_len - 14.142).abs() < 0.05);

        router.gloss_tracks(&mut board, &grid);

        let glossed_len: f64 = board.tracks.iter().map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1)).sum();
        assert_eq!(board.tracks.len(), 1, "Rubber-band straightening should collapse 2 detour segments into 1 direct segment");
        assert!((glossed_len - 10.0).abs() < 1e-4, "Straightened length must equal direct horizontal distance (10.0 mm)");
        assert_eq!(board.tracks[0].start, (10.0, 10.0));
        assert_eq!(board.tracks[0].end, (20.0, 10.0));
    }

    #[test]
    fn test_post_route_glossing_eliminates_stair_steps() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        // Staircase zig-zag: (10, 10) -> (12, 10) -> (12, 12) -> (14, 12) -> (14, 14) -> (16, 14) -> (16, 16) -> (20, 16)
        let pts = [
            (10.0, 10.0),
            (12.0, 10.0),
            (12.0, 12.0),
            (14.0, 12.0),
            (14.0, 14.0),
            (16.0, 14.0),
            (16.0, 16.0),
            (20.0, 16.0),
        ];
        let mut tracks = Vec::new();
        for i in 0..pts.len() - 1 {
            tracks.push(TrackSegment {
                start: pts[i],
                end: pts[i + 1],
                width: 0.25,
                layer: 0,
                net_id: 1,
                is_locked: false,
            });
        }

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks,
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(3);

        let initial_len: f64 = board.tracks.iter().map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1)).sum();
        let initial_bends = board.tracks.len().saturating_sub(1);
        assert_eq!(initial_bends, 6);

        router.gloss_tracks(&mut board, &grid);

        let glossed_len: f64 = board.tracks.iter().map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1)).sum();
        assert!(
            board.tracks.len() <= 3,
            "Staircase elimination must collapse 7 segments into <= 3 canonical segments, got {}",
            board.tracks.len()
        );
        assert!(
            glossed_len < initial_len - 1.5,
            "Glossed path length ({:.3}) must be shorter than staircase ({:.3})",
            glossed_len,
            initial_len
        );
    }

    #[test]
    fn test_post_route_glossing_preserves_clearance() {
        let nets = vec!["".to_string(), "SIG1".to_string(), "BLOCKER".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        net_name_to_id.insert("BLOCKER".to_string(), 2);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());
        net_to_class.insert("BLOCKER".to_string(), "Default".to_string());

        let blocker_fp = make_fp("U_BLOCK", 12.0, 12.0, true, vec![
            Pad::new("1", 0.0, 0.0, 2.0, 2.0, 2, "BLOCKER", false),
        ]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: vec![blocker_fp],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (10.0, 20.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 20.0),
                    end: (20.0, 20.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        grid.rasterize_obstacles(&board);
        let router = AutoRouter::new(3);

        router.gloss_tracks(&mut board, &grid);

        for trk in &board.tracks {
            let dist = dist_to_segment(12.0, 12.0, trk.start.0, trk.start.1, trk.end.0, trk.end.1);
            let req_clearance = 0.20 + 0.25 / 2.0 + 1.0; // pad_rad + track_half_w + clearance
            assert!(
                dist >= req_clearance - 0.05,
                "Track segment from ({:.2}, {:.2}) to ({:.2}, {:.2}) encroaches on blocker at (12, 12) with dist {:.3} < {:.3}",
                trk.start.0, trk.start.1, trk.end.0, trk.end.1, dist, req_clearance
            );
        }
    }

    #[test]
    fn test_via_reduction_eliminates_redundant_via_pair() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        // Net 1 has 3 track segments:
        // L0: (10, 10) -> (15, 10)
        // Via 1 at (15, 10)
        // L1: (15, 10) -> (25, 10)
        // Via 2 at (25, 10)
        // L0: (25, 10) -> (30, 10)
        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (15.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 10.0),
                    end: (25.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (25.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (15.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (25.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(3);

        assert_eq!(board.vias.len(), 2);
        router.reduce_vias(&mut board, &mut grid);

        assert_eq!(board.vias.len(), 0, "Via reduction pass must remove redundant via pair in free space");
        let l1_tracks = board.tracks.iter().filter(|t| t.layer == 1).count();
        assert_eq!(l1_tracks, 0, "All tracks must be unified on layer 0 with 0 tracks on layer 1");
    }

    #[test]
    fn test_via_consolidation_single_via_elimination() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        // Net 1 with Pad A at (10, 10) on F.Cu and Pad B at (30, 10) on F.Cu
        // Connected via 1 via at (15, 10) and tracks on B.Cu (layer 1) to (30, 10).
        let pad_a = Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false);
        let fp_a = make_fp("U1", 10.0, 10.0, false, vec![pad_a]);
        let pad_b = Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false);
        let fp_b = make_fp("U2", 30.0, 10.0, false, vec![pad_b]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: vec![fp_a, fp_b],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (15.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (15.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        assert_eq!(board.vias.len(), 1);
        assert_eq!(board.tracks.iter().filter(|t| t.layer == 1).count(), 1);

        reduce_vias(&mut board, &mut grid);

        assert_eq!(board.vias.len(), 0, "Via consolidation must eliminate the solitary via");
        let l1_tracks = board.tracks.iter().filter(|t| t.layer == 1).count();
        assert_eq!(l1_tracks, 0, "All tracks must be moved to F.Cu (layer 0)");
    }

    #[test]
    fn test_via_sliding_eliminates_jog() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        let pad_a = Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false);
        let fp_a = make_fp("U1", 0.0, 0.0, false, vec![pad_a]);
        let pad_b = Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false);
        let mut fp_b = make_fp("U2", 10.0, 10.0, false, vec![pad_b]);
        fp_b.layer = 1;

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: -10.0,
            min_y: -10.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: vec![fp_a, fp_b],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (0.0, 0.0),
                    end: (10.0, 0.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 0.0),
                    end: (10.0, 0.5),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 0.5),
                    end: (10.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (10.0, 0.5),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let grid = RoutingGrid::new(&board, 0.20, None);
        assert_eq!(board.vias.len(), 1);
        assert_eq!(board.tracks.len(), 3);

        let changed = slide_and_align_vias(&mut board, &grid);
        assert!(changed, "slide_and_align_vias should return true");

        assert_eq!(board.vias.len(), 1);
        assert_eq!(board.vias[0].position, (10.0, 0.0), "Via should slide to the corner at (10.0, 0.0)");
        assert_eq!(board.tracks.len(), 2, "Dogleg jog track should be eliminated");

        let l0_track = board.tracks.iter().find(|t| t.layer == 0).unwrap();
        assert_eq!(l0_track.start, (0.0, 0.0));
        assert_eq!(l0_track.end, (10.0, 0.0));

        let l1_track = board.tracks.iter().find(|t| t.layer == 1).unwrap();
        assert_eq!(l1_track.start, (10.0, 0.0));
        assert_eq!(l1_track.end, (10.0, 10.0));
    }

    #[test]
    fn test_via_reduction_preserves_necessary_vias_over_obstacles() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        // Keepout on layer 0 blocking (15, 10) to (25, 10)
        let ko = KeepoutZone {
            min_x: 18.0,
            min_y: 5.0,
            max_x: 22.0,
            max_y: 15.0,
            layers: vec![0], // only blocks layer 0
        };

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![ko],
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (15.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 10.0),
                    end: (25.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (25.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (15.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (25.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        grid.rasterize_obstacles(&board);
        let router = AutoRouter::new(3);

        router.reduce_vias(&mut board, &mut grid);

        assert_eq!(
            board.vias.len(),
            2,
            "Necessary vias crossing an obstacle on layer 0 must be preserved"
        );
        let l1_tracks = board.tracks.iter().filter(|t| t.layer == 1).count();
        assert_eq!(l1_tracks, 1, "Layer 1 track segment must be preserved");
    }

    #[test]
    fn test_via_reduction_respects_length_budget() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        // Short hop of length 2.0 mm on layer 1:
        // L0: (10, 10) -> (15, 10)
        // V1 at (15, 10), V2 at (17, 10)
        // L1: (15, 10) -> (17, 10)
        // L0: (17, 10) -> (22, 10)
        let ko = KeepoutZone {
            min_x: 15.5,
            min_y: 0.0,
            max_x: 16.5,
            max_y: 20.0,
            layers: vec![0], // spans y from 0 to 20, completely blocking layer 0
        };

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![ko],
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (15.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 10.0),
                    end: (17.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (17.0, 10.0),
                    end: (22.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (15.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (17.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        grid.rasterize_obstacles(&board);
        let router = AutoRouter::new(3);

        router.reduce_vias(&mut board, &mut grid);

        assert_eq!(
            board.vias.len(),
            2,
            "Vias must be preserved when detour exceeds length budget or is blocked"
        );
    }

    #[test]
    fn test_continuous_track_shove_deflects_with_45_degree_transitions() {
        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets: vec!["".to_string(), "SIG1".to_string(), "INTRUDER".to_string()],
            net_name_to_id: [("SIG1".to_string(), 1), ("INTRUDER".to_string(), 2)].into_iter().collect(),
            netclasses: [("Default".to_string(), NetClass::default())].into_iter().collect(),
            net_to_class: [("SIG1".to_string(), "Default".to_string()), ("INTRUDER".to_string(), "Default".to_string())].into_iter().collect(),
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let nudge = NudgeEngine::new();
        // Intruder on net 2 from (18.0, 10.20) to (22.0, 10.20), width 0.25
        let shoved = nudge.try_continuous_shove_track(
            &mut board,
            0,
            (18.0, 10.20),
            (22.0, 10.20),
            0.25,
            2,
            None,
        );

        assert!(shoved, "Continuous track shove must succeed in open space");

        let has_start = board.tracks.iter().any(|t| (t.start.0 - 10.0).hypot(t.start.1 - 10.0) < 1e-4 || (t.end.0 - 10.0).hypot(t.end.1 - 10.0) < 1e-4);
        let has_end = board.tracks.iter().any(|t| (t.start.0 - 30.0).hypot(t.start.1 - 10.0) < 1e-4 || (t.end.0 - 30.0).hypot(t.end.1 - 10.0) < 1e-4);
        assert!(has_start, "Start endpoint (10, 10) must be preserved");
        assert!(has_end, "End endpoint (30, 10) must be preserved");

        assert_eq!(board.tracks.len(), 3, "Shoved track must produce 3 segments (lead-in, deflected middle, lead-out)");

        let lead_in = board.tracks.iter().find(|t| (t.start.0 - 10.0).hypot(t.start.1 - 10.0) < 1e-4 || (t.end.0 - 10.0).hypot(t.end.1 - 10.0) < 1e-4).unwrap();
        let dx = (lead_in.end.0 - lead_in.start.0).abs();
        let dy = (lead_in.end.1 - lead_in.start.1).abs();
        assert!((dx - dy).abs() < 1e-4, "Lead-in transition must be at 45 degrees (dx == dy), got dx={:.3}, dy={:.3}", dx, dy);

        for t in &board.tracks {
            let dist = dist_segment_to_segment(t.start, t.end, (18.0, 10.20), (22.0, 10.20));
            assert!(dist >= 0.45 - 1e-4, "All deflected segments must maintain required clearance >= 0.45mm, got {:.3}", dist);
        }
    }

    #[test]
    fn test_continuous_track_shove_cascades_multiple_tracks() {
        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets: vec!["".to_string(), "SIG1".to_string(), "SIG2".to_string(), "INTRUDER".to_string()],
            net_name_to_id: [("SIG1".to_string(), 1), ("SIG2".to_string(), 2), ("INTRUDER".to_string(), 3)].into_iter().collect(),
            netclasses: [("Default".to_string(), NetClass::default())].into_iter().collect(),
            net_to_class: [
                ("SIG1".to_string(), "Default".to_string()),
                ("SIG2".to_string(), "Default".to_string()),
                ("INTRUDER".to_string(), "Default".to_string()),
            ].into_iter().collect(),
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 9.35),
                    end: (30.0, 9.35),
                    width: 0.25,
                    layer: 0,
                    net_id: 2,
                    is_locked: false,
                },
            ],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let nudge = NudgeEngine::new();
        let shoved = nudge.try_continuous_shove_track(
            &mut board,
            0,
            (18.0, 10.20),
            (22.0, 10.20),
            0.25,
            3,
            None,
        );

        assert!(shoved, "Cascading continuous shove across multiple tracks must succeed");

        for t1 in board.tracks.iter().filter(|t| t.net_id == 1) {
            let dist_to_intruder = dist_segment_to_segment(t1.start, t1.end, (18.0, 10.20), (22.0, 10.20));
            assert!(dist_to_intruder >= 0.45 - 1e-4, "Net 1 must be clear of intruder");

            for t2 in board.tracks.iter().filter(|t| t.net_id == 2) {
                let dist_between = dist_segment_to_segment(t1.start, t1.end, t2.start, t2.end);
                assert!(dist_between >= 0.45 - 1e-4, "Cascaded Net 2 must be clear of shoved Net 1, got {:.3}", dist_between);
            }
        }
    }

    #[test]
    fn test_continuous_track_shove_rejects_on_locked_or_keepout() {
        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets: vec!["".to_string(), "SIG1".to_string(), "LOCKED".to_string(), "INTRUDER".to_string()],
            net_name_to_id: [("SIG1".to_string(), 1), ("LOCKED".to_string(), 2), ("INTRUDER".to_string(), 3)].into_iter().collect(),
            netclasses: [("Default".to_string(), NetClass::default())].into_iter().collect(),
            net_to_class: [
                ("SIG1".to_string(), "Default".to_string()),
                ("LOCKED".to_string(), "Default".to_string()),
                ("INTRUDER".to_string(), "Default".to_string()),
            ].into_iter().collect(),
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 9.35),
                    end: (30.0, 9.35),
                    width: 0.25,
                    layer: 0,
                    net_id: 2,
                    is_locked: true,
                },
            ],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let orig_tracks = board.tracks.clone();
        let nudge = NudgeEngine::new();

        let shoved = nudge.try_continuous_shove_track(
            &mut board,
            0,
            (18.0, 10.20),
            (22.0, 10.20),
            0.25,
            3,
            None,
        );

        assert!(!shoved, "Continuous shove must safely abort when blocked by a locked track");
        assert_eq!(board.tracks.len(), 2, "Tracks must not be altered when shove fails");
        assert_eq!(board.tracks[0].start, orig_tracks[0].start);
        assert_eq!(board.tracks[0].end, orig_tracks[0].end);
    }

    #[test]
    fn test_prune_dangling_tracks_and_stubs() {
        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 50.0,
            max_y: 50.0,
            footprints: vec![
                make_fp("U1", 10.0, 10.0, false, vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false)]),
                make_fp("U2", 30.0, 10.0, false, vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false)]),
            ],
            holding_palette: Vec::new(),
            nets: vec!["".to_string(), "SIG1".to_string()],
            net_name_to_id: [("SIG1".to_string(), 1)].into_iter().collect(),
            netclasses: [("Default".to_string(), NetClass::default())].into_iter().collect(),
            net_to_class: [("SIG1".to_string(), "Default".to_string())].into_iter().collect(),
            keepouts: Vec::new(),
            tracks: vec![
                // 1. Valid connected route between U1 and U2:
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (20.0, 10.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (20.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // 2. Dangling stub branching off the route into empty space:
                TrackSegment {
                    start: (20.0, 10.0),
                    end: (20.0, 15.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // 3. Multi-segment antenna extension on the stub:
                TrackSegment {
                    start: (20.0, 15.0),
                    end: (20.0, 20.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // 4. Isolated floating track segment in empty space:
                TrackSegment {
                    start: (40.0, 40.0),
                    end: (45.0, 40.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                // Dangling via at the end of the multi-segment antenna:
                Via {
                    position: (20.0, 20.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        prune_dangling_tracks_and_vias(&mut board);

        // Connected routes between U1 and U2 must be 100% preserved
        assert_eq!(board.tracks.len(), 2, "Only the 2 connected route segments must survive");
        assert_eq!(board.tracks[0].start, (10.0, 10.0));
        assert_eq!(board.tracks[0].end, (20.0, 10.0));
        assert_eq!(board.tracks[1].start, (20.0, 10.0));
        assert_eq!(board.tracks[1].end, (30.0, 10.0));

        // Dangling vias and stubs must be completely pruned
        assert!(board.vias.is_empty(), "Dangling via must be completely pruned");
    }

    #[test]
    fn test_via_clearance_against_foreign_pads_and_tracks() {
        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 50.0,
            max_y: 50.0,
            footprints: vec![
                make_fp("U1", 10.0, 10.0, false, vec![Pad::new("1", 0.0, 0.0, 0.50, 0.50, 1, "SIG1", false)]),
            ],
            holding_palette: Vec::new(),
            nets: vec!["".to_string(), "SIG1".to_string(), "SIG2".to_string()],
            net_name_to_id: [("SIG1".to_string(), 1), ("SIG2".to_string(), 2)].into_iter().collect(),
            netclasses: [
                ("Default".to_string(), NetClass { clearance: 0.20, track_width: 0.20, via_dia: 0.60, via_drill: 0.30, ..NetClass::default() }),
            ].into_iter().collect(),
            net_to_class: [("SIG1".to_string(), "Default".to_string()), ("SIG2".to_string(), "Default".to_string())].into_iter().collect(),
            keepouts: Vec::new(),
            tracks: vec![
                // Foreign track of net 1 on layer 0 from (10.0, 10.0) to (10.0, 30.0)
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (10.0, 30.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let via_dia = 0.60;
        let clearance = 0.20;

        // 1. Candidate via directly overlapping foreign SMD pad at (10.0, 10.0) -> illegal
        assert!(!is_via_location_legal(&board, (10.0, 10.0), via_dia, 2, clearance));

        // 2. Candidate via within clearance of foreign SMD pad at (10.0, 10.45)
        // Pad boundary extends to y = 10.25. Via radius = 0.30, clearance = 0.20. Required edge dist = 0.50.
        // At y = 10.45, dist = 0.20 < 0.50 -> illegal
        assert!(!is_via_location_legal(&board, (10.0, 10.45), via_dia, 2, clearance));

        // 3. Candidate via directly on top of foreign track at (10.0, 20.0) -> illegal
        assert!(!is_via_location_legal(&board, (10.0, 20.0), via_dia, 2, clearance));

        // 4. Candidate via violating clearance to foreign track at (10.35, 20.0)
        // Required center-to-segment distance = via_r(0.30) + trk_w/2(0.10) + clearance(0.20) = 0.60 mm.
        // Distance is 0.35 < 0.60 -> illegal
        assert!(!is_via_location_legal(&board, (10.35, 20.0), via_dia, 2, clearance));

        // 5. Candidate via safely placed outside clearance corridor at (10.80, 20.0)
        // Distance is 0.80 >= 0.60 -> legal
        assert!(is_via_location_legal(&board, (10.80, 20.0), via_dia, 2, clearance));

        // 6. Test validate_and_clean_copper removes any accidentally placed colliding vias
        board.vias.push(Via {
            position: (10.0, 20.0), // Collision with foreign track!
            size: 0.60,
            drill: 0.30,
            net_id: 2,
            is_locked: false,
        });
        validate_and_clean_copper(&mut board);
        assert!(board.vias.is_empty(), "Colliding via must be eliminated by validate_and_clean_copper");

        // 7. Same-net SMD pad clearance test:
        // Candidate via of net 1 directly on top of net 1 SMD pad at (10.0, 10.0) -> MUST be illegal!
        assert!(
            !is_via_location_legal(&board, (10.0, 10.0), via_dia, 1, clearance),
            "Via must never be legal overlapping its own net SMD pad"
        );
        // Candidate via of net 1 within clearance of net 1 SMD pad at (10.0, 10.45) -> MUST be illegal!
        assert!(
            !is_via_location_legal(&board, (10.0, 10.45), via_dia, 1, clearance),
            "Via must never be legal violating clearance against its own net SMD pad"
        );
        // Candidate via of net 1 safely outside clearance at (10.0, 10.80) -> legal
        assert!(
            is_via_location_legal(&board, (10.0, 10.80), via_dia, 1, clearance),
            "Via safely outside clearance of its own net SMD pad should be legal"
        );

        // 8. validate_and_clean_copper eliminates via overlapping same-net SMD pad
        board.vias.push(Via {
            position: (10.0, 10.0), // Collision with same-net SMD pad!
            size: 0.60,
            drill: 0.30,
            net_id: 1,
            is_locked: false,
        });
        validate_and_clean_copper(&mut board);
        assert!(board.vias.is_empty(), "Same-net SMD pad colliding via must be eliminated by validate_and_clean_copper");

        // 9. Net 0 unconnected pad clearance test
        let mut board_nc = board.clone();
        board_nc.footprints.push(make_fp("U2", 30.0, 30.0, false, vec![Pad::new("1", 0.0, 0.0, 0.60, 0.60, 0, "", false)]));
        assert!(
            !is_via_location_legal(&board_nc, (30.0, 30.0), via_dia, 1, clearance),
            "Via must never be legal overlapping an unconnected net 0 pad"
        );

        // 10. Oval pad and flipped footprint clearance test
        let mut board_oval = board.clone();
        let mut oval_pad = Pad::new("1", 0.0, 0.0, 1.45, 0.30, 1, "SIG1", false);
        oval_pad.shape = "oval".to_string();
        oval_pad.rotation = 90.0;
        let mut fp_flipped = make_fp("J2", 25.0, 25.0, false, vec![oval_pad]);
        fp_flipped.layer = 1; // Flipped to B.Cu
        fp_flipped.orig_layer = 0;
        board_oval.footprints.push(fp_flipped);

        // Candidate via directly on oval pad
        assert!(!is_via_location_legal(&board_oval, (25.0, 25.0), via_dia, 1, clearance));
        // Candidate via right at edge of oval pad
        assert!(!is_via_location_legal(&board_oval, (25.0, 25.50), via_dia, 1, clearance));
        // Candidate via safely away from flipped oval pad
        assert!(is_via_location_legal(&board_oval, (25.0, 27.0), via_dia, 1, clearance));
    }

    #[test]
    fn test_route_between_pads_of_component() {
        let nets = vec!["".to_string(), "GND".to_string(), "PAD1_NET".to_string(), "PAD2_NET".to_string(), "SIG_NET".to_string()];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        let mut nc = NetClass::default();
        nc.track_width = 0.20;
        nc.clearance = 0.20;
        netclasses.insert("Default".to_string(), nc);
        let mut net_to_class = HashMap::new();
        for n in &nets {
            net_to_class.insert(n.clone(), "Default".to_string());
        }

        // Component R1 at (10.0, 10.05) with two pads:
        // Pad 1 at (10.0, 9.45), size (0.80, 0.60) -> top edge is at 9.75
        // Pad 2 at (10.0, 10.65), size (0.80, 0.60) -> bottom edge is at 10.35
        // Physical gap between Pad 1 and Pad 2: 10.35 - 9.75 = 0.60 mm.
        // Required space for 0.20 mm track with 0.20 mm clearance on both sides:
        // clearance (0.20) + track_width (0.20) + clearance (0.20) = 0.60 mm.
        // Since 0.60 mm >= 0.60 mm, clearance & width allow routing between the pads!
        let r1_pads = vec![
            Pad::new("1", 0.0, -0.60, 0.80, 0.60, 2, "PAD1_NET", false),
            Pad::new("2", 0.0, 0.60, 0.80, 0.60, 3, "PAD2_NET", false),
        ];
        let r1 = make_fp("R1", 10.0, 10.05, true, r1_pads);

        // SIG_NET endpoints: J1 at (5.0, 10.05), J2 at (15.0, 10.05)
        let j1 = make_fp("J1", 5.0, 10.05, true, vec![Pad::new("1", 0.0, 0.0, 0.40, 0.40, 4, "SIG_NET", false)]);
        let j2 = make_fp("J2", 15.0, 10.05, true, vec![Pad::new("1", 0.0, 0.0, 0.40, 0.40, 4, "SIG_NET", false)]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 20.0,
            max_y: 20.0,
            footprints: vec![r1, j1, j2],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            // Keepouts force the signal to pass directly through the channel between R1 pads at y = 10.05 on layer 0 (F.Cu)
            keepouts: vec![
                KeepoutZone {
                    min_x: 8.0,
                    min_y: 11.5,
                    max_x: 12.0,
                    max_y: 20.0,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 8.0,
                    min_y: 0.0,
                    max_x: 12.0,
                    max_y: 8.5,
                    layers: vec![0, 1],
                },
                // Block layer 1 under R1 completely so route cannot bypass layer 0
                KeepoutZone {
                    min_x: 8.0,
                    min_y: 8.5,
                    max_x: 12.0,
                    max_y: 11.5,
                    layers: vec![1],
                },
            ],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.10, None);
        let router = AutoRouter::new(3);
        let stop = Arc::new(AtomicBool::new(false));

        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);
        assert_eq!(total, 1);
        assert_eq!(
            routed, 1,
            "Router route_all must be able to route between pads of R1 when clearance (0.20) and width (0.20) fit in 0.60 mm gap"
        );

        // Verify a track segment passes through the gap between R1 pads on layer 0 (F.Cu)
        let passes_between = board.tracks.iter().any(|t| {
            t.net_id == 4
                && t.layer == 0
                && ((t.start.0 <= 10.0 && t.end.0 >= 10.0) || (t.start.0 >= 10.0 && t.end.0 <= 10.0))
                && t.start.1 > 9.68 && t.start.1 < 10.32
                && t.end.1 > 9.68 && t.end.1 < 10.32
        });
        assert!(passes_between, "Track must physically pass between Pad 1 and Pad 2 of R1 on F.Cu");
    }

    #[test]
    fn test_route_between_pads_of_rotated_component() {
        let nets = vec!["".to_string(), "GND".to_string(), "PAD1_NET".to_string(), "PAD2_NET".to_string(), "SIG_NET".to_string()];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        let mut nc = NetClass::default();
        nc.track_width = 0.20;
        nc.clearance = 0.20;
        netclasses.insert("Default".to_string(), nc);
        let mut net_to_class = HashMap::new();
        for n in &nets {
            net_to_class.insert(n.clone(), "Default".to_string());
        }

        // Component R1 at (10.05, 10.0) rotated 90 degrees
        // In local coords: pads at (0.0, -0.60) and (0.0, 0.60), size (0.80, 0.60).
        // Rotated 90 degrees: pads are at (9.45, 10.0) and (10.65, 10.0).
        // Physical gap between Pad 1 and Pad 2 in X: 0.60 mm.
        // Vertical channel between pads runs at x = 10.05 from y = 5.0 to y = 15.0.
        let r1_pads = vec![
            Pad::new("1", 0.0, -0.60, 0.80, 0.60, 2, "PAD1_NET", false),
            Pad::new("2", 0.0, 0.60, 0.80, 0.60, 3, "PAD2_NET", false),
        ];
        let mut r1 = make_fp("R1", 10.05, 10.0, true, r1_pads);
        r1.rotation = 90.0;

        // SIG_NET endpoints: J1 at (10.05, 5.0), J2 at (10.05, 15.0)
        let j1 = make_fp("J1", 10.05, 5.0, true, vec![Pad::new("1", 0.0, 0.0, 0.40, 0.40, 4, "SIG_NET", false)]);
        let j2 = make_fp("J2", 10.05, 15.0, true, vec![Pad::new("1", 0.0, 0.0, 0.40, 0.40, 4, "SIG_NET", false)]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 20.0,
            max_y: 20.0,
            footprints: vec![r1, j1, j2],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![
                KeepoutZone {
                    min_x: 11.5,
                    min_y: 8.0,
                    max_x: 20.0,
                    max_y: 12.0,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 0.0,
                    min_y: 8.0,
                    max_x: 8.5,
                    max_y: 12.0,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 8.5,
                    min_y: 8.0,
                    max_x: 11.5,
                    max_y: 12.0,
                    layers: vec![1],
                },
            ],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.10, None);
        let router = AutoRouter::new(3);
        let stop = Arc::new(AtomicBool::new(false));

        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);

        assert_eq!(total, 1);
        assert_eq!(
            routed, 1,
            "Router must route vertically between pads of 90-degree rotated R1 on F.Cu"
        );

        let passes_between = board.tracks.iter().any(|t| {
            t.net_id == 4
                && t.layer == 0
                && ((t.start.1 <= 10.0 && t.end.1 >= 10.0) || (t.start.1 >= 10.0 && t.end.1 <= 10.0))
                && t.start.0 > 9.68 && t.start.0 < 10.32
                && t.end.0 > 9.68 && t.end.0 < 10.32
        });
        assert!(passes_between, "Track must physically pass vertically between Pad 1 and Pad 2 of 90-deg rotated R1");
    }

    #[test]
    fn test_route_between_through_hole_pins() {
        let nets = vec!["".to_string(), "GND".to_string(), "PIN1_NET".to_string(), "PIN2_NET".to_string(), "SIG_NET".to_string()];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        let mut nc = NetClass::default();
        nc.track_width = 0.20;
        nc.clearance = 0.20;
        netclasses.insert("Default".to_string(), nc);
        let mut net_to_class = HashMap::new();
        for n in &nets {
            net_to_class.insert(n.clone(), "Default".to_string());
        }

        // 2.54mm header with 2 through-hole pins:
        // Pin 1 at (10.0, 8.73), Pin 2 at (10.0, 11.27) -> pitch 2.54 mm.
        // Pad size: 1.60 x 1.60 mm.
        // Gap = 2.54 - 1.60 = 0.94 mm.
        // Channel at y = 10.0. Required for 0.20 track + 2x0.20 clearance = 0.60 mm. Fits with 0.34 mm margin.
        let j_hdr_pads = vec![
            Pad::new("1", 0.0, -1.27, 1.60, 1.60, 2, "PIN1_NET", true),
            Pad::new("2", 0.0, 1.27, 1.60, 1.60, 3, "PIN2_NET", true),
        ];
        let j_hdr = make_fp("J_HDR", 10.0, 10.05, true, j_hdr_pads);

        let j1 = make_fp("J1", 5.0, 10.05, true, vec![Pad::new("1", 0.0, 0.0, 0.40, 0.40, 4, "SIG_NET", false)]);
        let j2 = make_fp("J2", 15.0, 10.05, true, vec![Pad::new("1", 0.0, 0.0, 0.40, 0.40, 4, "SIG_NET", false)]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 20.0,
            max_y: 20.0,
            footprints: vec![j_hdr, j1, j2],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![
                KeepoutZone {
                    min_x: 8.0,
                    min_y: 12.5,
                    max_x: 12.0,
                    max_y: 20.0,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 8.0,
                    min_y: 0.0,
                    max_x: 12.0,
                    max_y: 7.5,
                    layers: vec![0, 1],
                },
            ],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.10, None);
        let router = AutoRouter::new(3);
        let stop = Arc::new(AtomicBool::new(false));

        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);

        assert_eq!(total, 1);
        assert_eq!(routed, 1, "Router must route between through-hole pins of 2.54mm header");
    }

    #[test]
    fn test_side_and_diagonal_escape_stubs_for_dense_ic_corners() {
        let mut board = make_router_test_board();
        // In make_router_test_board, U1 is at (145.0, 102.0).
        // Pad 6 on U1 is the corner pad at (-1.5, 1.0), so absolute position is (143.5, 103.0).
        // Place a target destination footprint for net 2 ("SIG1") South-West of Pad 6 at (140.0, 106.5).
        let target_fp = make_fp(
            "TARGET_SW",
            140.0,
            106.5,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "SIG1", false)],
        );
        board.footprints.push(target_fp);

        let router = AutoRouter::new(3);
        let stubs = router.generate_escape_stubs(&board);

        // Find the escape stub for Pad 6 on U1
        let pad6_stub = stubs.iter().find(|s| {
            s.net_id == 2 && (s.pad_pos.0 - 143.5).hypot(s.pad_pos.1 - 103.0) < 0.1
        });
        assert!(pad6_stub.is_some(), "Pad 6 on U1 must generate an escape stub");
        let stub = pad6_stub.unwrap();

        // Vector from pad to tip
        let dx = stub.tip_pos.0 - stub.pad_pos.0;
        let dy = stub.tip_pos.1 - stub.pad_pos.1;
        
        // Pad 6 at bottom-left corner with target to the South-West should escape diagonally towards the destination
        assert!(dx < -1e-4, "Escape stub must project westward towards target: dx={}", dx);
        assert!(dy > 1e-4, "Escape stub must project southward towards target (diagonal escape): dy={}", dy);
        assert!((dx.abs() - dy.abs()).abs() < 1e-3, "Diagonal escape stub must be 45-degree octilinear (|dx| == |dy|): dx={}, dy={}", dx, dy);
    }

    #[test]
    fn test_avoid_acid_traps_eliminates_acute_angles() {
        let board = make_router_test_board();
        let grid = RoutingGrid::new(&board, 0.10, None);

        // Create a chain with an acute angle (135° turn / 45° acute interior wedge)
        // (0.0, 0.0) -> (2.0, 0.0) -> (1.0, 1.0)
        // v1 = (2.0, 0.0), v2 = (-1.0, 1.0). dot = -2.0 / (2.0 * sqrt(2)) = -0.707 < -0.05
        let mut chain = TrackChain {
            net_id: 2,
            layer: 0,
            width: 0.20,
            points: vec![(0.0, 0.0), (2.0, 0.0), (1.0, 1.0)],
        };

        avoid_acid_traps(&mut chain, &board, &grid);

        // Verify no consecutive segments form an acute angle (dot product < -0.05)
        for i in 1..chain.points.len() - 1 {
            let p0 = chain.points[i - 1];
            let p1 = chain.points[i];
            let p2 = chain.points[i + 1];
            let v1 = (p1.0 - p0.0, p1.1 - p0.1);
            let v2 = (p2.0 - p1.0, p2.1 - p1.1);
            let l1 = v1.0.hypot(v1.1);
            let l2 = v2.0.hypot(v2.1);
            if l1 > 1e-4 && l2 > 1e-4 {
                let dot = (v1.0 * v2.0 + v1.1 * v2.1) / (l1 * l2);
                assert!(dot >= -0.05, "Angle at {:?} must not be acute (dot = {})", p1, dot);
            }
        }
    }

    #[test]
    fn test_reduce_corners_simplifies_dogleg() {
        let board = make_router_test_board();
        let grid = RoutingGrid::new(&board, 0.10, None);

        // 4 points where p0 -> p3 can connect directly via an octilinear segment:
        // (0.0, 0.0) -> (0.0, 1.0) -> (1.0, 1.0) -> (2.0, 2.0)
        // Notice p0 -> p3 is (0, 0) -> (2, 2) which is a direct 45° octilinear connection!
        let mut chain = TrackChain {
            net_id: 2,
            layer: 0,
            width: 0.20,
            points: vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 2.0)],
        };

        reduce_corners(&mut chain, &board, &grid);

        // Intermediate corners should be collapsed directly into p0 -> p3
        assert_eq!(chain.points.len(), 2, "Chain must be simplified to [p0, p3]");
        assert_eq!(chain.points[0], (0.0, 0.0));
        assert_eq!(chain.points[1], (2.0, 2.0));
    }

    #[test]
    fn test_smoothen_corners_adaptive_chamfering() {
        let board = make_router_test_board();
        let grid = RoutingGrid::new(&board, 0.10, None);

        // A simple 90-degree corner: (0.0, 0.0) -> (1.0, 0.0) -> (1.0, 1.0)
        let mut chain = TrackChain {
            net_id: 2,
            layer: 0,
            width: 0.20,
            points: vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        };

        smoothen_corners(&mut chain, &board, &grid);

        // The 90-degree corner at (1.0, 0.0) should be chamfered with 2 miter points at 45 degrees
        assert_eq!(chain.points.len(), 4, "90-degree corner must be chamfered into 4 points");
        assert_eq!(chain.points[0], (0.0, 0.0));
        assert_eq!(chain.points[3], (1.0, 1.0));

        let m1 = chain.points[1];
        let m2 = chain.points[2];
        assert!(m1.0 < 1.0 && m1.1 == 0.0, "m1 must be set back on horizontal segment: {:?}", m1);
        assert!(m2.0 == 1.0 && m2.1 > 0.0, "m2 must be set back on vertical segment: {:?}", m2);

        let dx = (m2.0 - m1.0).abs();
        let dy = (m2.1 - m1.1).abs();
        assert!((dx - dy).abs() < 1e-4, "Miter segment must be 45 degrees: dx={}, dy={}", dx, dy);
    }

    #[test]
    fn test_octilinear_segment_math_and_doglegs() {
        // 1. Horizontal & Vertical segments
        assert!(is_octilinear_segment((0.0, 0.0), (10.0, 0.0)));
        assert!(is_octilinear_segment((0.0, 0.0), (0.0, -5.5)));
        assert!(is_octilinear_segment((141.0, 102.65), (142.95, 102.65)));

        // 2. 45-degree diagonal segments
        assert!(is_octilinear_segment((0.0, 0.0), (3.0, 3.0)));
        assert!(is_octilinear_segment((10.0, 10.0), (5.0, 15.0)));
        assert!(is_octilinear_segment((142.92, 102.62), (142.95, 102.65)));

        // 3. Non-octilinear segments (should be rejected)
        assert!(!is_octilinear_segment((141.0, 102.62), (142.95, 102.65))); // 0.88° tilt
        assert!(!is_octilinear_segment((0.0, 0.0), (1.0, 2.0))); // ~63.4°
        assert!(!is_octilinear_segment((10.0, 10.0), (12.0, 13.0))); // non-45°

        // 4. compute_45_dogleg decomposes non-octilinear points into strictly octilinear segments
        let dogleg = compute_45_dogleg((141.0, 102.62), (142.95, 102.65), false);
        assert_eq!(dogleg.len(), 3);
        assert!(is_octilinear_segment(dogleg[0], dogleg[1]));
        assert!(is_octilinear_segment(dogleg[1], dogleg[2]));

        let dogleg_diag = compute_45_dogleg((141.0, 102.62), (142.95, 102.65), true);
        assert_eq!(dogleg_diag.len(), 3);
        assert!(is_octilinear_segment(dogleg_diag[0], dogleg_diag[1]));
        assert!(is_octilinear_segment(dogleg_diag[1], dogleg_diag[2]));
    }

    #[test]
    fn test_enforce_45_grid_decomposes_arbitrary_angles() {
        // A track chain with intentional non-octilinear segments (e.g. from off-grid pad to grid corner)
        let mut chain = TrackChain {
            net_id: 1,
            layer: 0,
            width: 0.20,
            points: vec![
                (141.000, 102.620), // off-grid pad
                (142.950, 102.650), // grid corner (0.88 deg tilt)
                (150.000, 102.650), // horizontal
                (152.000, 104.650), // 45 degree
                (153.250, 105.388), // weird angle ~48.4 deg
            ],
        };

        enforce_45_grid(&mut chain);

        // Every segment in the chain must strictly conform to 0°, 45°, or 90°
        for w in chain.points.windows(2) {
            let p1 = w[0];
            let p2 = w[1];
            assert!(
                is_octilinear_segment(p1, p2),
                "Segment from {:?} to {:?} must be octilinear (0°, 45°, 90°)",
                p1,
                p2
            );
        }
    }

    #[test]
    fn test_teardrop_generation_tangent_to_pad_and_via() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "NET1".to_string(), "NET2".to_string(), "NET3".to_string()];
        board.net_name_to_id.insert("NET1".to_string(), 1);
        board.net_name_to_id.insert("NET2".to_string(), 2);
        board.net_name_to_id.insert("NET3".to_string(), 3);

        // Circular pad at (10.0, 10.0), net 1, radius 0.40 (dia 0.80)
        let circ_pad = Pad {
            number: "1".to_string(),
            shape: "circle".to_string(),
            offset_x: 0.0,
            offset_y: 0.0,
            width: 0.80,
            height: 0.80,
            rotation: 0.0,
            drill: None,
            net_id: 1,
            net_name: "NET1".to_string(),
            is_through_hole: false,
            ..Default::default()
        };
        let fp_circ = make_fp("U1", 10.0, 10.0, false, vec![circ_pad]);

        // Rectangular pad at (30.0, 30.0), net 2, width 0.80, height 1.60
        let rect_pad = Pad {
            number: "1".to_string(),
            shape: "rect".to_string(),
            offset_x: 0.0,
            offset_y: 0.0,
            width: 0.80,
            height: 1.60,
            rotation: 0.0,
            drill: None,
            net_id: 2,
            net_name: "NET2".to_string(),
            is_through_hole: false,
            ..Default::default()
        };
        let fp_rect = make_fp("U2", 30.0, 30.0, false, vec![rect_pad]);

        board.footprints = vec![fp_circ, fp_rect];

        // Through-hole via at (50.0, 50.0), net 3, size 0.60
        board.vias = vec![Via {
            position: (50.0, 50.0),
            size: 0.60,
            drill: 0.30,
            net_id: 3,
            is_locked: false,
        }];

        // Tracks meeting pads and via
        // Track 1 (net 1): from circular pad center (10.0, 10.0) heading East to (20.0, 10.0)
        let t1 = TrackSegment {
            start: (10.0, 10.0),
            end: (20.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        };
        // Track 2 (net 2): from rectangular pad center (30.0, 30.0) heading North to (30.0, 40.0)
        let t2 = TrackSegment {
            start: (30.0, 30.0),
            end: (30.0, 40.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        };
        // Track 3 (net 3): from via position (50.0, 50.0) heading East to (60.0, 50.0)
        let t3 = TrackSegment {
            start: (50.0, 50.0),
            end: (60.0, 50.0),
            width: 0.20,
            layer: 0,
            net_id: 3,
            is_locked: false,
        };

        board.tracks = vec![t1, t2, t3];

        add_teardrops(&mut board);

        // Expect 2 teardrop flares for each junction: 3 original + 6 flares = 9 tracks
        assert_eq!(board.tracks.len(), 9, "Each junction must generate 2 smooth flared teardrop tracks");

        // Verify circular pad teardrops
        let t1_teardrops: Vec<&TrackSegment> = board.tracks[3..]
            .iter()
            .filter(|t| t.net_id == 1)
            .collect();
        assert_eq!(t1_teardrops.len(), 2);
        let flare_w1 = (t1_teardrops[0].end.1 - t1_teardrops[1].end.1).abs();
        assert!(flare_w1 > 0.20, "Circular pad flare width must be wider than track width (0.20)");
        for td in &t1_teardrops {
            let dist_to_center = (td.end.0 - 10.0).hypot(td.end.1 - 10.0);
            assert!((dist_to_center - 0.40).abs() < 1e-2, "Flare end must be on perimeter of circular pad (r=0.40)");
        }

        // Verify rectangular pad teardrops
        let t2_teardrops: Vec<&TrackSegment> = board.tracks[3..]
            .iter()
            .filter(|t| t.net_id == 2)
            .collect();
        assert_eq!(t2_teardrops.len(), 2);
        let flare_w2 = (t2_teardrops[0].end.0 - t2_teardrops[1].end.0).abs();
        assert!(flare_w2 > 0.25, "Rectangular pad flare width must be wider than track width (0.25)");
        for td in &t2_teardrops {
            assert!((td.end.1 - 30.80).abs() < 1e-2, "Flare end must be on North perimeter of rectangular pad (y=30.80)");
        }

        // Verify through-hole via teardrops
        let t3_teardrops: Vec<&TrackSegment> = board.tracks[3..]
            .iter()
            .filter(|t| t.net_id == 3)
            .collect();
        assert_eq!(t3_teardrops.len(), 2);
        let flare_w3 = (t3_teardrops[0].end.1 - t3_teardrops[1].end.1).abs();
        assert!(flare_w3 > 0.20, "Via flare width must be wider than track width (0.20)");
        for td in &t3_teardrops {
            let dist_to_via = (td.end.0 - 50.0).hypot(td.end.1 - 50.0);
            assert!((dist_to_via - 0.30).abs() < 1e-2, "Flare end must be on perimeter of via (r=0.30)");
        }
    }

    #[test]
    fn test_teardrop_suppressed_on_clearance_conflict() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "NET1".to_string(), "NET2".to_string()];
        board.net_name_to_id.insert("NET1".to_string(), 1);
        board.net_name_to_id.insert("NET2".to_string(), 2);

        // Circular pad at (10.0, 10.0), net 1, radius 0.40
        let target_pad = Pad {
            number: "1".to_string(),
            shape: "circle".to_string(),
            offset_x: 0.0,
            offset_y: 0.0,
            width: 0.80,
            height: 0.80,
            rotation: 0.0,
            drill: None,
            net_id: 1,
            net_name: "NET1".to_string(),
            is_through_hole: false,
            ..Default::default()
        };
        let fp_target = make_fp("U1", 10.0, 10.0, false, vec![target_pad]);

        // Foreign pads of net 2 immediately flanking the pad-track junction within clearance
        let foreign_pad_left = Pad {
            number: "1".to_string(),
            shape: "circle".to_string(),
            offset_x: 0.0,
            offset_y: 0.0,
            width: 0.40,
            height: 0.40,
            rotation: 0.0,
            drill: None,
            net_id: 2,
            net_name: "NET2".to_string(),
            is_through_hole: false,
            ..Default::default()
        };
        let fp_foreign_left = make_fp("R1", 10.24, 10.45, false, vec![foreign_pad_left]);

        let foreign_pad_right = Pad {
            number: "1".to_string(),
            shape: "circle".to_string(),
            offset_x: 0.0,
            offset_y: 0.0,
            width: 0.40,
            height: 0.40,
            rotation: 0.0,
            drill: None,
            net_id: 2,
            net_name: "NET2".to_string(),
            is_through_hole: false,
            ..Default::default()
        };
        let fp_foreign_right = make_fp("R2", 10.24, 9.55, false, vec![foreign_pad_right]);

        board.footprints = vec![fp_target, fp_foreign_left, fp_foreign_right];

        // Track of net 1
        board.tracks = vec![TrackSegment {
            start: (10.0, 10.0),
            end: (20.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        }];

        add_teardrops(&mut board);

        // Teardrops must be suppressed because both flanking flares conflict with foreign pads
        assert_eq!(board.tracks.len(), 1, "Teardrops must be suppressed when conflicting with adjacent foreign copper");
    }

    #[test]
    fn test_copper_to_edge_cuts_clearance_enforced() {
        let mut board = Board::default();
        board.outline_polygon = Some(Polygon::from_rect(0.0, 0.0, 50.0, 50.0));
        board.netclasses.insert("Default".to_string(), NetClass {
            board_edge_clearance: 0.35,
            ..NetClass::default()
        });
        board.nets = vec!["".to_string(), "NET1".to_string()];
        board.net_name_to_id.insert("NET1".to_string(), 1);

        // Track 1: distance to x=0 edge is 0.20 - 0.10 = 0.10 mm < 0.35 mm (ILLEGAL)
        let trk_illegal = TrackSegment {
            start: (0.20, 10.0),
            end: (10.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        };

        // Track 2: distance to x=0 edge is 1.00 - 0.10 = 0.90 mm >= 0.35 mm (LEGAL)
        let trk_legal = TrackSegment {
            start: (1.00, 20.0),
            end: (10.0, 20.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        };

        // Via 1: distance to x=0 edge is 0.40 - 0.30 = 0.10 mm < 0.35 mm (ILLEGAL)
        let via_illegal = Via {
            position: (0.40, 30.0),
            size: 0.60,
            drill: 0.30,
            net_id: 1,
            is_locked: false,
        };

        // Via 2: distance to x=0 edge is 1.00 - 0.30 = 0.70 mm >= 0.35 mm (LEGAL)
        let via_legal = Via {
            position: (1.00, 35.0),
            size: 0.60,
            drill: 0.30,
            net_id: 1,
            is_locked: false,
        };

        // 1. Verify routing geometry checkers reject near-edge copper
        assert!(!is_track_segment_legal(&board, (0.20, 10.0), (10.0, 10.0), 0.20, 0, 1, 0.20));
        assert!(is_track_segment_legal(&board, (1.00, 20.0), (10.0, 20.0), 0.20, 0, 1, 0.20));
        assert!(!is_via_location_legal(&board, (0.40, 30.0), 0.60, 1, 0.20));
        assert!(is_via_location_legal(&board, (1.00, 35.0), 0.60, 1, 0.20));

        board.tracks = vec![trk_illegal, trk_legal];
        board.vias = vec![via_illegal, via_legal];

        // 2. Verify validate_and_clean_copper eliminates illegal copper
        validate_and_clean_copper(&mut board);

        assert_eq!(board.tracks.len(), 1, "Illegal track violating 0.35mm edge clearance must be eliminated");
        assert_eq!(board.tracks[0].start.0, 1.00);

        assert_eq!(board.vias.len(), 1, "Illegal via violating 0.35mm edge clearance must be eliminated");
        assert_eq!(board.vias[0].position.0, 1.00);

        // 3. Verify routing grid rasterization expands edge keepout by 0.35 mm
        let mut grid = RoutingGrid::new(&board, 0.5, None);
        grid.rasterize_obstacles(&board);
        // (0.25, 25.0) is 0.25 mm from left boundary (x=0) < 0.35 mm -> must be keepout
        let Some((gx, gy)) = grid.world_to_grid(0.25, 25.0) else {
            panic!("(0.25, 25.0) must map into grid");
        };
        let obs = grid.obstacle_net_ids[grid.idx(gx, gy, 0)];
        assert_eq!(obs, KEEPOUT_NET, "Grid cell at 0.25mm from edge must be marked as keepout");
    }

    #[test]
    fn test_crosstalk_parallelism_limiter_diverts_long_coupled_trace() {
        let mut board = make_router_test_board();
        let mut sig2_nc = NetClass::default();
        sig2_nc.max_coupled_length_mm = 3.0;
        sig2_nc.track_width = 0.20;
        sig2_nc.clearance = 0.20;
        board.netclasses.insert("Sig2Class".to_string(), sig2_nc);
        board.net_to_class.insert("SIG2".to_string(), "Sig2Class".to_string());

        // Place an existing track for SIG1 (net 2)
        board.tracks.push(TrackSegment {
            start: (135.0, 100.0),
            end: (155.0, 100.0),
            width: 0.20,
            layer: 0,
            net_id: 2,
            is_locked: true,
        });

        let router = AutoRouter::new(2).with_crosstalk_penalty(5.0);
        let grid = RoutingGrid::new(&board, 0.25, None);

        // Route SIG2 from (135.0, 100.5) to (155.0, 100.5).
        let start_node = grid.world_to_grid(135.0, 100.5).map(|(gx, gy)| GridNode { gx, gy, layer: 0 }).unwrap();
        let goal_node = grid.world_to_grid(155.0, 100.5).map(|(gx, gy)| GridNode { gx, gy, layer: 0 }).unwrap();

        // 1. With crosstalk limiter active (max_coupled = 3.0mm):
        let stop_flag = AtomicBool::new(false);
        let path_constrained = router.find_path_with_constraints(
            &grid,
            start_node,
            goal_node,
            3, // SIG2
            0.20,
            0.20,
            0.60,
            None,
            1.0,
            0.0,
            3.0,
            false,
            None,
            &stop_flag,
        );
        assert!(path_constrained.is_some(), "Path should find a valid route");
        let path = path_constrained.unwrap();

        let max_y = path.iter().map(|n| {
            let (_, wy) = grid.grid_to_world(n.gx, n.gy);
            wy
        }).fold(0.0f64, f64::max);
        assert!(max_y > 101.0, "Path should divert away from parallel track to avoid crosstalk penalty, max_y={:.2}", max_y);

        // 2. Also test post-route crosstalk validation
        let mut drc_board = make_router_test_board();
        drc_board.netclasses.get_mut("Default").unwrap().max_coupled_length_mm = 5.0;
        drc_board.tracks.push(TrackSegment {
            start: (135.0, 100.0),
            end: (145.0, 100.0),
            width: 0.20,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        drc_board.tracks.push(TrackSegment {
            start: (135.0, 100.4),
            end: (145.0, 100.4),
            width: 0.20,
            layer: 0,
            net_id: 3,
            is_locked: false,
        });

        let violations = validate_crosstalk_and_coupling(&drc_board);
        assert_eq!(violations.len(), 1, "Should report 1 coupling violation");
        assert!(violations[0].coupled_length_mm >= 9.9);
        assert_eq!(violations[0].max_allowed_mm, 5.0);
    }

    #[test]
    fn test_return_path_continuity_penalizes_split_plane_crossing() {
        let mut board = make_router_test_board();
        board.raw_content = r#"
            (kicad_pcb
                (layers
                    (0 "F.Cu" signal)
                    (1 "In1.Cu" power)
                    (2 "In2.Cu" power)
                    (31 "B.Cu" signal)
                )
                (zone (net 1) (net_name "GND") (layers "In1.Cu"))
            )
        "#.to_string();

        let mut hs_nc = NetClass::default();
        hs_nc.require_return_path = true;
        board.netclasses.insert("HighSpeed".to_string(), hs_nc);
        board.net_to_class.insert("SIG2".to_string(), "HighSpeed".to_string());

        board.keepouts.push(KeepoutZone {
            min_x: 144.0,
            min_y: 95.0,
            max_x: 146.0,
            max_y: 105.0,
            layers: vec![2],
        });

        let router = AutoRouter::new(4);
        let grid = RoutingGrid::new(&board, 0.25, None);

        let ref_layer = grid.adjacent_reference_plane(0).expect("Layer 0 must have reference plane");
        assert_eq!(ref_layer, 2, "In 4-layer stackup, In1.Cu is layer 2");

        let (split_gx, split_gy) = grid.world_to_grid(145.0, 100.0).unwrap();
        assert!(
            grid.is_reference_plane_void_at(split_gx, split_gy, ref_layer),
            "Cell (145.0, 100.0) on reference plane In1.Cu must detect void"
        );

        let (free_gx, free_gy) = grid.world_to_grid(140.0, 100.0).unwrap();
        assert!(
            !grid.is_reference_plane_void_at(free_gx, free_gy, ref_layer),
            "Cell (140.0, 100.0) on reference plane In1.Cu must have continuous copper"
        );

        let start_node = GridNode { gx: free_gx, gy: free_gy, layer: 0 };
        let goal_node = grid.world_to_grid(150.0, 100.0).map(|(gx, gy)| GridNode { gx, gy, layer: 0 }).unwrap();

        let stop_flag = AtomicBool::new(false);
        let path_with_si = router.find_path_with_constraints(
            &grid,
            start_node,
            goal_node,
            3, // SIG2
            0.20,
            0.20,
            0.60,
            None,
            1.0,
            0.0,
            15.0,
            true, // require_return_path = true
            None,
            &stop_flag,
        );
        assert!(path_with_si.is_some(), "Should find a route respecting return path");
        let path = path_with_si.unwrap();

        let crossed_void = path.iter().any(|n| {
            let (wx, wy) = grid.grid_to_world(n.gx, n.gy);
            wx >= 144.0 && wx <= 146.0 && wy >= 95.0 && wy <= 105.0
        });
        assert!(!crossed_void, "High-speed track with require_return_path=true must detour around split plane void");
    }

    #[test]
    fn test_return_path_ground_via_insertion_near_signal_transition() {
        let mut board = make_router_test_board();
        let mut nc = NetClass::default();
        nc.name = "HighSpeed".to_string();
        nc.require_return_path = true;
        board.netclasses.insert("HighSpeed".to_string(), nc);
        board.net_to_class.insert("SIG2".to_string(), "HighSpeed".to_string());

        board.vias.push(Via {
            position: (138.0, 95.0),
            size: 0.60,
            drill: 0.30,
            net_id: 3,
            is_locked: false,
        });

        let router = AutoRouter::new(2);
        let grid = RoutingGrid::new(&board, 0.25, None);

        let initial_via_count = board.vias.len();
        let added = router.add_return_path_ground_vias(&mut board, &grid);

        assert_eq!(added, 1, "Should insert exactly 1 ground stitching return-path via");
        assert_eq!(board.vias.len(), initial_via_count + 1);

        let gnd_via = board.vias.iter().find(|v| v.net_id == 1 && (v.position.0 - 138.0).hypot(v.position.1 - 95.0) <= 1.5);
        assert!(gnd_via.is_some(), "A GND via must be inserted within 1.5mm of the signal via");

        let gnd_pos = gnd_via.unwrap().position;
        let dist = (gnd_pos.0 - 138.0).hypot(gnd_pos.1 - 95.0);
        assert!(dist >= 0.60, "GND via must maintain DRC clearance to signal via: dist={:.2}", dist);
        assert!(dist <= 1.50, "GND via must be within 1.5mm of signal via: dist={:.2}", dist);

        let added_second = router.add_return_path_ground_vias(&mut board, &grid);
        assert_eq!(added_second, 0, "Second pass must not add redundant GND vias");
    }

    #[test]
    fn test_quadrant_dogbone_fanout_geometry() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 50.0;
        board.max_y = 50.0;
        board.netclasses.get_mut("Default").unwrap().clearance = 0.12;
        for i in 0..=20 {
            board.nets.push(format!("NET{}", i));
        }

        let mut fp = Footprint {
            reference: "U_BGA".to_string(),
            value: "BGA-16".to_string(),
            x: 25.0,
            y: 25.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: Vec::new(),
            width: 5.0,
            height: 5.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        // 4x4 BGA grid with 0.8mm pitch:
        // Columns: x = -1.2, -0.4, 0.4, 1.2
        // Rows:    y = -1.2, -0.4, 0.4, 1.2
        let xs = [-1.2, -0.4, 0.4, 1.2];
        let ys = [-1.2, -0.4, 0.4, 1.2];
        let mut pad_num = 1;
        for &y in &ys {
            for &x in &xs {
                let mut p = Pad::new(&format!("P{}", pad_num), x, y, 0.40, 0.40, pad_num, &format!("NET{}", pad_num), false);
                p.shape = "circle".to_string();
                fp.pads.push(p);
                pad_num += 1;
            }
        }
        board.footprints.push(fp);

        let mut grid = RoutingGrid::new(&board, 0.10, None);
        generate_bga_dogbone_escapes(&mut board, &mut grid);

        // Ring 1 pads: inner 4 pads:
        // (1,1) -> row 1, col 1: pad 6 (index 5) at local (-0.4, -0.4), NW quadrant
        // (2,1) -> row 1, col 2: pad 7 (index 6) at local (0.4, -0.4), NE quadrant
        // (1,2) -> row 2, col 1: pad 10 (index 9) at local (-0.4, 0.4), SW quadrant
        // (2,2) -> row 2, col 2: pad 11 (index 10) at local (0.4, 0.4), SE quadrant
        assert_eq!(board.vias.len(), 4, "Expected exactly 4 dog-bone vias for Ring 1 pads of 4x4 BGA");

        let bga_fp = &board.footprints[0];
        let inner_indices = [5, 6, 9, 10];

        for &pad_idx in &inner_indices {
            let pad = &bga_fp.pads[pad_idx];
            let (px, py) = bga_fp.pad_absolute_position(pad);

            // Find matching via for this net
            let via = board.vias.iter().find(|v| v.net_id == pad.net_id)
                .expect("Each Ring 1 pad must have a dog-bone via");

            let dx = via.position.0 - px;
            let dy = via.position.1 - py;

            // Verifies 45° dog-bone via placement in interstitial centers (|dx| == |dy| == p/2)
            assert!((dx.abs() - 0.4).abs() < 1e-3, "Via dx must be +/- 0.4mm (p/2): dx={}", dx);
            assert!((dy.abs() - 0.4).abs() < 1e-3, "Via dy must be +/- 0.4mm (p/2): dy={}", dy);
            assert!((dx.abs() - dy.abs()).abs() < 1e-4, "Dog-bone via must be at 45 degrees (|dx| == |dy|)");

            // Verify quadrant direction:
            let lx = pad.offset_x;
            let ly = pad.offset_y;
            if lx > 0.0 {
                assert!(dx > 0.0, "Positive local X pad must dog-bone in +X direction");
            } else {
                assert!(dx < 0.0, "Negative local X pad must dog-bone in -X direction");
            }
            if ly > 0.0 {
                assert!(dy > 0.0, "Positive local Y pad must dog-bone in +Y direction");
            } else {
                assert!(dy < 0.0, "Negative local Y pad must dog-bone in -Y direction");
            }

            // Verify distance to pad center is sqrt(0.4^2 + 0.4^2) ~ 0.5657 mm
            let dist_to_pad = dx.hypot(dy);
            assert!((dist_to_pad - (0.4f64.hypot(0.4))).abs() < 1e-3, "Distance must be interstitial diagonal center");

            // Verify clearance against all foreign pads in the BGA (no shorts!)
            for foreign_pad in &bga_fp.pads {
                if foreign_pad.net_id != pad.net_id {
                    let (fx, fy) = bga_fp.pad_absolute_position(foreign_pad);
                    let d = (via.position.0 - fx).hypot(via.position.1 - fy);
                    let min_allowed = (via.size * 0.5) + (foreign_pad.width * 0.5) + 0.12;
                    assert!(
                        d >= min_allowed - 1e-4,
                        "Dog-bone via for net {} at ({:.2}, {:.2}) must not short against foreign pad {} at ({:.2}, {:.2}): d={:.3}, min_allowed={:.3}",
                        pad.net_id, via.position.0, via.position.1, foreign_pad.net_id, fx, fy, d, min_allowed
                    );
                }
            }

            // Verify is_via_location_legal returns true
            assert!(
                is_via_location_legal(&board, via.position, via.size, via.net_id, 0.12),
                "Dog-bone via location must be strictly legal and clearance-free"
            );
        }

        // Verify that Ring 0 pads generated escape tracks
        for (i, pad) in bga_fp.pads.iter().enumerate() {
            if bga_fp.bga_pad_ring_index(i) == 0 {
                let (px, py) = bga_fp.pad_absolute_position(pad);
                let has_trk = board.tracks.iter().any(|t| {
                    (t.start.0 - px).hypot(t.start.1 - py) < 1e-4 && t.net_id == pad.net_id
                });
                assert!(has_trk, "Ring 0 pad {} must have an escape track", pad.number);
            }
        }
    }

    #[test]
    fn test_bus_bundle_equidistant_offsets() {
        let mut board = Board::default();
        board.nets = vec![
            "".to_string(),
            "NET1".to_string(),
            "NET2".to_string(),
            "NET3".to_string(),
            "NET4".to_string(),
        ];
        for (i, n) in board.nets.iter().enumerate() {
            if !n.is_empty() {
                board.net_name_to_id.insert(n.clone(), i);
            }
        }
        let mut grid = RoutingGrid::new(&board, 0.25, None);

        let bundle = BusBundle::new(
            "DataBus",
            vec![1, 2, 3, 4],
            vec![(10.0, 10.0), (30.0, 10.0), (50.0, 30.0)],
            0,
            0.25,
            0.25,
        );

        let ok = route_bundle_along_guide(&mut board, &mut grid, &bundle);
        assert!(ok, "route_bundle_along_guide must succeed for valid 2-segment guide line");

        // Verify tracks created for all 4 nets
        for &nid in &bundle.net_ids {
            let count = board.tracks.iter().filter(|t| t.net_id == nid).count();
            assert!(count >= 2, "Net {} should have at least 2 segments along the polyline", nid);
        }

        // Along the first straight segment (x from 10 to 30), examine tracks at x = 20.0
        let mut lane_ys = Vec::new();
        for &nid in &bundle.net_ids {
            let trk = board
                .tracks
                .iter()
                .find(|t| t.net_id == nid && t.start.0 <= 20.0 && t.end.0 >= 20.0 && (t.start.1 - t.end.1).abs() < 1e-4)
                .expect("Each net must have a horizontal segment passing x=20.0");
            lane_ys.push((nid, trk.start.1));
        }

        // Verify sorted y-coordinates have exact (W + S) = 0.50mm spacing everywhere
        lane_ys.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for i in 0..lane_ys.len() - 1 {
            let pitch = lane_ys[i + 1].1 - lane_ys[i].1;
            assert!(
                (pitch - 0.50).abs() < 1e-4,
                "Spacing between lane {} and {} must be exactly W + S = 0.50mm, got {:.4}mm",
                lane_ys[i].0,
                lane_ys[i + 1].0,
                pitch
            );
        }

        // Verify that along the second diagonal segment (direction (1, 1)), perpendicular distance is exactly 0.50mm
        let mut diag_lines = Vec::new();
        for &nid in &bundle.net_ids {
            let trk = board
                .tracks
                .iter()
                .find(|t| {
                    t.net_id == nid
                        && t.start.0 >= 29.0
                        && ((t.end.0 - t.start.0) - (t.end.1 - t.start.1)).abs() < 1e-4
                        && (t.end.0 - t.start.0) > 1.0
                })
                .expect("Each net must have a 45-degree segment along the second guide segment");
            let c = trk.start.0 - trk.start.1;
            diag_lines.push((nid, c));
        }
        diag_lines.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for i in 0..diag_lines.len() - 1 {
            let perp_dist = (diag_lines[i + 1].1 - diag_lines[i].1) / std::f64::consts::SQRT_2;
            assert!(
                (perp_dist - 0.50).abs() < 1e-4,
                "Diagonal perpendicular spacing must be exactly 0.50mm, got {:.4}mm",
                perp_dist
            );
        }
    }

    #[test]
    fn test_bundle_corner_concentric_miters() {
        let mut board = Board::default();
        board.nets = vec![
            "".to_string(),
            "NET1".to_string(),
            "NET2".to_string(),
            "NET3".to_string(),
            "NET4".to_string(),
        ];
        for (i, n) in board.nets.iter().enumerate() {
            if !n.is_empty() {
                board.net_name_to_id.insert(n.clone(), i);
            }
        }
        let mut grid = RoutingGrid::new(&board, 0.25, None);

        let w = 0.25;
        let s = 0.25;
        let bundle = BusBundle::new(
            "CornerBus",
            vec![1, 2, 3, 4],
            vec![(10.0, 10.0), (40.0, 10.0), (40.0, 40.0)],
            0,
            w,
            s,
        );

        let ok = route_bundle_along_guide(&mut board, &mut grid, &bundle);
        assert!(ok, "route_bundle_along_guide must successfully route around 90-degree bend");

        // 1. Verify that 45-degree miter segments were generated around the 90-degree turn
        for &nid in &bundle.net_ids {
            let miter_trk = board.tracks.iter().find(|t| {
                t.net_id == nid
                    && ((t.end.0 - t.start.0).abs() - (t.end.1 - t.start.1)).abs() < 1e-4
                    && (t.end.0 - t.start.0).abs() > 0.05
                    && (t.end.1 - t.start.1).abs() > 0.05
            });
            assert!(
                miter_trk.is_some(),
                "Net {} must have a 45-degree concentric miter segment at corner",
                nid
            );
        }

        // 2. Verify concentric clearance: no tracks pinch or violate minimum spacing anywhere!
        let bundle_tracks: Vec<&TrackSegment> = board
            .tracks
            .iter()
            .filter(|t| bundle.net_ids.contains(&t.net_id))
            .collect();

        for i in 0..bundle_tracks.len() {
            for j in (i + 1)..bundle_tracks.len() {
                let t1 = bundle_tracks[i];
                let t2 = bundle_tracks[j];
                if t1.net_id != t2.net_id {
                    let d = dist_segment_to_segment(t1.start, t1.end, t2.start, t2.end);
                    let copper_clearance = d - (t1.width * 0.5 + t2.width * 0.5);
                    assert!(
                        copper_clearance >= s - 1e-4,
                        "Clearance between Net {} and Net {} pinched to {:.4}mm (minimum required: {:.4}mm)",
                        t1.net_id,
                        t2.net_id,
                        copper_clearance,
                        s
                    );
                }
            }
        }
    }

    #[test]
    fn test_fine_pitch_vippo_fallback() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 50.0;
        board.max_y = 50.0;
        board.netclasses.get_mut("Default").unwrap().clearance = 0.08;
        for i in 0..=20 {
            board.nets.push(format!("NET{}", i));
        }

        let mut fp = Footprint {
            reference: "U_BGA_FINE".to_string(),
            value: "BGA-16-0.4mm".to_string(),
            x: 25.0,
            y: 25.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: Vec::new(),
            width: 3.0,
            height: 3.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        // 4x4 BGA grid with fine 0.4mm pitch (< 0.50mm):
        // Columns: x = -0.6, -0.2, 0.2, 0.6
        // Rows:    y = -0.6, -0.2, 0.2, 0.6
        let xs = [-0.6, -0.2, 0.2, 0.6];
        let ys = [-0.6, -0.2, 0.2, 0.6];
        let mut pad_num = 1;
        for &y in &ys {
            for &x in &xs {
                let mut p = Pad::new(&format!("P{}", pad_num), x, y, 0.22, 0.22, pad_num, &format!("NET{}", pad_num), false);
                p.shape = "circle".to_string();
                fp.pads.push(p);
                pad_num += 1;
            }
        }
        board.footprints.push(fp);

        let mut grid = RoutingGrid::new(&board, 0.05, None);
        generate_bga_dogbone_escapes(&mut board, &mut grid);

        // For pitch < 0.5mm, VIPPO fallback places centered via-in-pad microvias (0.30mm dia, 0.15mm drill)
        assert_eq!(board.vias.len(), 4, "Expected exactly 4 VIPPO microvias for Ring 1 pads of fine-pitch BGA");

        let bga_fp = &board.footprints[0];
        let inner_indices = [5, 6, 9, 10];

        for &pad_idx in &inner_indices {
            let pad = &bga_fp.pads[pad_idx];
            let (px, py) = bga_fp.pad_absolute_position(pad);

            let via = board.vias.iter().find(|v| v.net_id == pad.net_id)
                .expect("Each Ring 1 fine-pitch pad must have a VIPPO microvia");

            // Verify via is centered directly on pad (via-in-pad)
            let dist_to_pad = (via.position.0 - px).hypot(via.position.1 - py);
            assert!(dist_to_pad < 1e-4, "VIPPO microvia must be centered directly on pad: dist={}", dist_to_pad);

            // Verify microvia dimensions
            assert!((via.size - 0.30).abs() < 1e-4, "VIPPO microvia diameter must be 0.30mm: size={}", via.size);
            assert!((via.drill - 0.15).abs() < 1e-4, "VIPPO microvia drill must be 0.15mm: drill={}", via.drill);

            // Verify legality of via-in-pad microvia transition
            assert!(
                is_via_location_legal(&board, via.position, via.size, via.net_id, 0.08),
                "VIPPO microvia transition must be legal"
            );
        }
    }

    #[test]
    fn test_bundle_fanout_connects_to_pads() {
        let mut board = Board::default();
        board.nets = vec![
            "".to_string(),
            "D0".to_string(),
            "D1".to_string(),
            "D2".to_string(),
            "D3".to_string(),
        ];
        for (i, name) in board.nets.iter().enumerate() {
            if !name.is_empty() {
                board.net_name_to_id.insert(name.clone(), i);
            }
        }

        let j1_pads = vec![
            Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "D0", false),
            Pad::new("2", 0.0, 1.0, 0.6, 0.6, 2, "D1", false),
            Pad::new("3", 0.0, 2.0, 0.6, 0.6, 3, "D2", false),
            Pad::new("4", 0.0, 3.0, 0.6, 0.6, 4, "D3", false),
        ];
        let j1 = make_fp("J1", 5.0, 10.0, true, j1_pads);

        let j2_pads = vec![
            Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "D0", false),
            Pad::new("2", 0.0, 1.0, 0.6, 0.6, 2, "D1", false),
            Pad::new("3", 0.0, 2.0, 0.6, 0.6, 3, "D2", false),
            Pad::new("4", 0.0, 3.0, 0.6, 0.6, 4, "D3", false),
        ];
        let j2 = make_fp("J2", 45.0, 10.0, true, j2_pads);

        board.footprints.push(j1);
        board.footprints.push(j2);

        let mut grid = RoutingGrid::new(&board, 0.25, None);

        let bundle = BusBundle::new(
            "DataBus",
            vec![1, 2, 3, 4],
            vec![(10.0, 11.5), (40.0, 11.5)],
            0,
            0.25,
            0.25,
        );

        let ok = route_bundle_along_guide(&mut board, &mut grid, &bundle);
        assert!(ok, "route_bundle_along_guide must successfully connect pads to bundle");

        // Verify each net connects to its respective pad on J1 and J2
        for (idx, &nid) in bundle.net_ids.iter().enumerate() {
            let j1_pad_pos = (5.0, 10.0 + idx as f64 * 1.0);
            let j2_pad_pos = (45.0, 10.0 + idx as f64 * 1.0);

            let connects_j1 = board.tracks.iter().any(|t| {
                t.net_id == nid
                    && ((t.start.0 - j1_pad_pos.0).hypot(t.start.1 - j1_pad_pos.1) < 1e-3
                        || (t.end.0 - j1_pad_pos.0).hypot(t.end.1 - j1_pad_pos.1) < 1e-3)
            });
            assert!(
                connects_j1,
                "Net {} must connect to J1 pad at ({:.1}, {:.1})",
                nid,
                j1_pad_pos.0,
                j1_pad_pos.1
            );

            let connects_j2 = board.tracks.iter().any(|t| {
                t.net_id == nid
                    && ((t.start.0 - j2_pad_pos.0).hypot(t.start.1 - j2_pad_pos.1) < 1e-3
                        || (t.end.0 - j2_pad_pos.0).hypot(t.end.1 - j2_pad_pos.1) < 1e-3)
            });
            assert!(
                connects_j2,
                "Net {} must connect to J2 pad at ({:.1}, {:.1})",
                nid,
                j2_pad_pos.0,
                j2_pad_pos.1
            );
        }
    }

    #[test]
    fn test_homotopic_rubber_band_shortening() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "SIG".to_string(), "GND".to_string()];
        board.net_name_to_id.insert("SIG".to_string(), 1);
        board.net_name_to_id.insert("GND".to_string(), 2);

        // Obstacle via for GND (net 2) at (5.0, 5.0), size 1.0 (radius 0.5)
        board.vias.push(Via {
            position: (5.0, 5.0),
            size: 1.0,
            drill: 0.5,
            net_id: 2,
            is_locked: true,
        });

        // Net 1 (SIG) has a zigzag trace passing around the obstacle on the +y side:
        // (0.0, 0.0) -> (0.0, 8.0) -> (3.0, 8.0) -> (3.0, 9.0) -> (7.0, 9.0) -> (7.0, 8.0) -> (10.0, 8.0) -> (10.0, 0.0)
        let pts = [
            (0.0, 0.0),
            (0.0, 8.0),
            (3.0, 8.0),
            (3.0, 9.0),
            (7.0, 9.0),
            (7.0, 8.0),
            (10.0, 8.0),
            (10.0, 0.0),
        ];
        for k in 0..pts.len() - 1 {
            board.tracks.push(TrackSegment {
                start: pts[k],
                end: pts[k + 1],
                width: 0.20,
                layer: 0,
                net_id: 1,
                is_locked: false,
            });
        }

        let initial_length: f64 = board.tracks.iter().map(|t| t.length()).sum();
        assert!((initial_length - 28.0).abs() < 1e-4);

        // Relax homotopic rubber bands
        relax_homotopic_rubber_bands(&mut board);

        let relaxed_length: f64 = board.tracks.iter().map(|t| t.length()).sum();

        // 1. Length must be significantly shortened by taut rubber-band pulling (> 25% reduction)
        assert!(
            relaxed_length < initial_length * 0.75,
            "Relaxed length {:.2} must be < 75% of initial {:.2}",
            relaxed_length,
            initial_length
        );

        // 2. All tracks must maintain required clearance from net 2 via at (5.0, 5.0)
        let via_pos = (5.0, 5.0);
        let req_clr = board.get_netclass_for_net(1).clearance.max(board.get_netclass_for_net(2).clearance);
        let min_centerline_dist = 0.5 + req_clr + 0.10; // via radius (0.5) + clearance (0.2) + half width (0.1)
        for trk in &board.tracks {
            let d = dist_point_to_segment(via_pos, trk.start, trk.end);
            assert!(
                d >= min_centerline_dist - 1e-4,
                "Track segment {:?}->{:?} violates clearance: dist={:.4}, req={:.4}",
                trk.start, trk.end, d, min_centerline_dist
            );
        }

        // 3. Homotopy check: trace must remain on the +y side of the obstacle at (5.0, 5.0)
        for trk in &board.tracks {
            if (trk.start.0 <= 5.0 && trk.end.0 >= 5.0) || (trk.start.0 >= 5.0 && trk.end.0 <= 5.0) {
                let t = if (trk.end.0 - trk.start.0).abs() > 1e-6 {
                    (5.0 - trk.start.0) / (trk.end.0 - trk.start.0)
                } else {
                    0.5
                };
                let y_at_5 = trk.start.1 + t * (trk.end.1 - trk.start.1);
                assert!(y_at_5 > 5.0, "Homotopy violated: trace crossed to -y side of obstacle");
            }
        }
    }

    #[test]
    fn test_tangent_arc_geometry_and_clearance() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "SIG1".to_string(), "SIG2".to_string(), "GND".to_string()];
        board.net_name_to_id.insert("SIG1".to_string(), 1);
        board.net_name_to_id.insert("SIG2".to_string(), 2);
        board.net_name_to_id.insert("GND".to_string(), 3);

        // 1. 90-degree corner on SIG1
        board.tracks.push(TrackSegment {
            start: (0.0, 0.0),
            end: (10.0, 0.0),
            width: 0.25,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: (10.0, 0.0),
            end: (10.0, 10.0),
            width: 0.25,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });

        // 2. 45-degree corner on SIG2
        board.tracks.push(TrackSegment {
            start: (0.0, 20.0),
            end: (10.0, 20.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: (10.0, 20.0),
            end: (15.0, 25.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });

        apply_any_angle_tangent_arcs(&mut board);

        // Verify that 2 tangent arcs were created (one for 90 deg, one for 45 deg)
        assert_eq!(board.arcs.len(), 2, "Expected 2 tangent arcs inscribed at corners");

        // Verify 90-degree arc properties (net 1)
        let arc90 = board.arcs.iter().find(|a| a.net_id == 1).expect("Arc for net 1 not found");
        let start90 = arc90.start();
        let end90 = arc90.end();
        assert!((start90.1 - 0.0).abs() < 1e-3);
        assert!(start90.0 < 10.0 && start90.0 > 8.0);
        assert!((end90.0 - 10.0).abs() < 1e-3);
        assert!(end90.1 > 0.0 && end90.1 < 2.0);
        assert!((arc90.radius - 1.0).abs() < 1e-3);

        // Verify 45-degree arc properties (net 2)
        let arc45 = board.arcs.iter().find(|a| a.net_id == 2).expect("Arc for net 2 not found");
        let start45 = arc45.start();
        assert!((start45.1 - 20.0).abs() < 1e-3);
        assert!(start45.0 < 10.0);

        // 3. Test NetClass clearance constraint with a foreign obstacle inside corner
        let mut drc_board = Board::default();
        drc_board.nets = vec!["".to_string(), "SIG".to_string(), "GND".to_string()];
        drc_board.net_name_to_id.insert("SIG".to_string(), 1);
        drc_board.net_name_to_id.insert("GND".to_string(), 2);

        drc_board.tracks.push(TrackSegment {
            start: (0.0, 0.0),
            end: (10.0, 0.0),
            width: 0.25,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        drc_board.tracks.push(TrackSegment {
            start: (10.0, 0.0),
            end: (10.0, 10.0),
            width: 0.25,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });

        // Place foreign via inside the 90-degree corner at (9.4, 0.6)
        drc_board.vias.push(Via {
            position: (9.4, 0.6),
            size: 0.60,
            drill: 0.30,
            net_id: 2,
            is_locked: true,
        });

        apply_any_angle_tangent_arcs(&mut drc_board);

        // Verify that if an arc is inscribed, its clearance to the foreign via strictly satisfies DRC
        let via_pos = (9.4, 0.6);
        let req_clr = drc_board.get_netclass_for_net(1).clearance.max(drc_board.get_netclass_for_net(2).clearance);
        let required_dist = 0.30 + req_clr + 0.125;
        for arc in &drc_board.arcs {
            let d = arc.distance_to_point(via_pos);
            assert!(
                d >= required_dist - 1e-4,
                "Arc {:?} violates netclass clearance to via at {:?}: dist={:.4}, req={:.4}",
                arc, via_pos, d, required_dist
            );
        }
    }

    #[test]
    fn test_p1p2esp_fine_pitch_connections_route_cleanly() {
        let pcb_path = std::path::Path::new("/home/deece/src/p1p2esp/p1p2esp.kicad_pcb");
        if !pcb_path.exists() {
            return;
        }
        let locked: std::collections::HashSet<String> = crate::board::DEFAULT_LOCKED_REFS.iter().map(|s| s.to_string()).collect();
        let board = Board::load(pcb_path.to_str().unwrap(), &locked).unwrap();
        let router = AutoRouter::new(2);
        let test_board = board.clone();
        let connections = router.extract_connections(&test_board);
        let mut grid = RoutingGrid::new(&test_board, 0.10, None);
        let escape_stubs = router.generate_escape_stubs(&test_board);
        for stub in &escape_stubs {
            router.commit_stub_to_grid(&mut grid, stub);
        }

        let stop = std::sync::atomic::AtomicBool::new(false);
        for c in &connections {
            if c.net_name == "TERM" || c.net_name == "USB_DP" || c.net_name == "USB_DM" || c.net_name == "VRAW" || c.net_name.contains("HNA") || c.net_name.contains("HPA") || c.net_name == "VBUS" || c.net_name.contains("FB") {
                let mut start_pt = c.start;
                let mut end_pt = c.end;
                for s in &escape_stubs {
                    if s.net_id == c.net_id && (start_pt.0 - s.pad_pos.0).hypot(start_pt.1 - s.pad_pos.1) < 0.15 {
                        start_pt = (s.tip_pos.0, s.tip_pos.1, s.tip_layer);
                    }
                    if s.net_id == c.net_id && (end_pt.0 - s.pad_pos.0).hypot(end_pt.1 - s.pad_pos.1) < 0.15 {
                        end_pt = (s.tip_pos.0, s.tip_pos.1, s.tip_layer);
                    }
                }
                let touches_fine_pitch = board.footprints.get(c.start_fp).map_or(false, |fp| fp.reference == "U1" || fp.reference == "J2" || AutoRouter::is_dense_escape_footprint(fp))
                    || board.footprints.get(c.end_fp).map_or(false, |fp| fp.reference == "U1" || fp.reference == "J2" || AutoRouter::is_dense_escape_footprint(fp))
                    || escape_stubs.iter().any(|s| {
                        (start_pt.0 - s.tip_pos.0).hypot(start_pt.1 - s.tip_pos.1) < 0.05
                            || (end_pt.0 - s.tip_pos.0).hypot(end_pt.1 - s.tip_pos.1) < 0.05
                    });
                let mut conn_nc = c.netclass.clone();
                if touches_fine_pitch {
                    conn_nc.track_width = conn_nc.track_width.min(0.15);
                    conn_nc.clearance = conn_nc.clearance.min(0.127);
                }

                let sg = grid.world_to_grid(start_pt.0, start_pt.1);
                let eg = grid.world_to_grid(end_pt.0, end_pt.1);
                if let (Some(s), Some(e)) = (sg, eg) {
                    let s_node = GridNode { gx: s.0, gy: s.1, layer: start_pt.2 };
                    let e_node = GridNode { gx: e.0, gy: e.1, layer: end_pt.2 };
                    let path = router.find_path_with_constraints(
                        &grid, s_node, e_node, c.net_id, conn_nc.clearance, conn_nc.track_width, conn_nc.via_dia, None, 1.0, 0.0, conn_nc.max_coupled_length_mm, conn_nc.require_return_path, None, &stop
                    );
                    assert!(path.is_some(), "Connection '{}' (net {}) must route successfully", c.net_name, c.net_id);
                }
            }
        }
    }
}


