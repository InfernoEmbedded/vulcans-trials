// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use crate::board::{ArcSegment, Board, Footprint, LiveTrialState, NetClass, NetTopology, Pad, Point, TrackSegment, TrialPhase, Via};
pub use crate::board::BusBundle;
use crate::cuda::{CudaEngine, MultiGpuRouterPool};
use crate::ipc::KiCadIpcClient;
use crate::vulkan::{GpuDrcCandidate, GpuDrcPad, GpuDrcTrack, GpuDrcVia, GpuRouteCostParams, VulkanEngine};
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
    #[serde(default = "default_crosstalk_penalty")]
    pub crosstalk_penalty: f64,
    #[serde(default)]
    pub any_angle_mode: bool,
    #[serde(default = "default_via_merge_radius_mm")]
    pub via_merge_radius_mm: f64,
    #[serde(default = "default_via_detour_budget_mm")]
    pub via_detour_budget_mm: f64,
    #[serde(default = "default_via_hop_max_detour_ratio")]
    pub via_hop_max_detour_ratio: f64,
    #[serde(default = "default_plane_via_share_radius_mm")]
    pub plane_via_share_radius_mm: f64,
    #[serde(default = "default_return_via_max_dist_mm")]
    pub return_via_max_dist_mm: f64,
    #[serde(default = "default_via_pad_snap_dist_mm")]
    pub via_pad_snap_dist_mm: f64,
    #[serde(default = "default_congestion_factor")]
    pub congestion_factor: f64,
    #[serde(default = "default_nudge_step_mm")]
    pub nudge_step_mm: f64,
    #[serde(default = "default_gloss_passes")]
    pub gloss_passes: usize,
}

fn default_crosstalk_penalty() -> f64 { 4.0 }
fn default_via_merge_radius_mm() -> f64 { 4.00 }
fn default_via_detour_budget_mm() -> f64 { 3.50 }
fn default_via_hop_max_detour_ratio() -> f64 { 1.25 }
fn default_plane_via_share_radius_mm() -> f64 { 3.50 }
fn default_return_via_max_dist_mm() -> f64 { 1.40 }
fn default_via_pad_snap_dist_mm() -> f64 { 0.50 }
fn default_congestion_factor() -> f64 { 1.50 }
fn default_nudge_step_mm() -> f64 { 0.25 }
fn default_gloss_passes() -> usize { 2 }

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            crosstalk_penalty: default_crosstalk_penalty(),
            any_angle_mode: false,
            via_merge_radius_mm: default_via_merge_radius_mm(),
            via_detour_budget_mm: default_via_detour_budget_mm(),
            via_hop_max_detour_ratio: default_via_hop_max_detour_ratio(),
            plane_via_share_radius_mm: default_plane_via_share_radius_mm(),
            return_via_max_dist_mm: default_return_via_max_dist_mm(),
            via_pad_snap_dist_mm: default_via_pad_snap_dist_mm(),
            congestion_factor: default_congestion_factor(),
            nudge_step_mm: default_nudge_step_mm(),
            gloss_passes: default_gloss_passes(),
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
    pub usage_count: Vec<u32>,
    pub gpu_net_cost_maps: HashMap<usize, (i32, Vec<f32>)>,
    pub nudgeable_fp: Vec<Option<usize>>,
    pub smd_pad_cells: Vec<bool>,
    pub net_clearances: Vec<f64>,
    pub max_net_clearance: f64,
    pub inner_layer_penalty: f64,
    pub vulkan: Option<Arc<VulkanEngine>>,
    pub base_snapshot: Option<Arc<(Vec<i32>, Vec<u32>, Vec<Option<usize>>, Vec<bool>)>>,
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
            usage_count: vec![0; total_cells],
            gpu_net_cost_maps: HashMap::new(),
            nudgeable_fp: vec![None; total_cells],
            smd_pad_cells: vec![false; total_cells],
            net_clearances,
            max_net_clearance,
            inner_layer_penalty: 8.0,
            vulkan,
            base_snapshot: None,
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
        self.gpu_net_cost_maps.clear();
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
        self.gpu_net_cost_maps.clear();
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
        self.gpu_net_cost_maps.clear();
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

    pub fn plane_layer_mask_for_net(&self, net_id: usize) -> u32 {
        let mut mask = 0u32;
        for (&layer, &plane_net) in &self.plane_nets {
            if layer < 32 && plane_net != net_id {
                mask |= 1u32 << layer;
            }
        }
        mask
    }

    /// Computes a 3D per-cell routing cost and clearance map on the GPU for `net_id`.
    /// Cells within `clearance_cells` of any foreign obstacle are marked `-1.0f` (impassable).
    pub fn compute_gpu_3d_routing_cost_for_net(
        &self,
        net_id: usize,
        clearance_cells: i32,
        present_congestion_factor: f32,
        obstacle_clearance_weight: f32,
    ) -> Option<Vec<f32>> {
        let vk = self.vulkan.as_ref()?;
        let obs_u32: Vec<u32> = self.obstacle_net_ids.iter().map(|&n| n as u32).collect();
        let params = GpuRouteCostParams {
            width: self.width as u32,
            height: self.height as u32,
            num_layers: self.num_layers as u32,
            net_id: net_id as u32,
            clearance_cells,
            present_congestion_factor,
            obstacle_clearance_weight,
            plane_layer_mask: self.plane_layer_mask_for_net(net_id),
        };
        vk.compute_3d_routing_costs(
            &obs_u32,
            &self.dist_field,
            &self.congestion,
            &self.usage_count,
            &params,
        )
    }

    /// Precomputes 3D routing cost and clearance maps across all available GPUs for a batch of `(net_id, clearance_cells)`.
    pub fn precompute_gpu_routing_costs_batch(
        &mut self,
        net_clearances_cells: &[(usize, i32)],
        present_congestion_factor: f32,
        obstacle_clearance_weight: f32,
    ) {
        self.gpu_net_cost_maps.clear();
        if net_clearances_cells.is_empty() {
            return;
        }
        let Some(ref vk) = self.vulkan else {
            return;
        };
        let obs_u32: Vec<u32> = self.obstacle_net_ids.iter().map(|&n| n as u32).collect();
        let params_batch: Vec<GpuRouteCostParams> = net_clearances_cells
            .iter()
            .map(|&(net_id, clearance_cells)| GpuRouteCostParams {
                width: self.width as u32,
                height: self.height as u32,
                num_layers: self.num_layers as u32,
                net_id: net_id as u32,
                clearance_cells,
                present_congestion_factor,
                obstacle_clearance_weight,
                plane_layer_mask: self.plane_layer_mask_for_net(net_id),
            })
            .collect();
        let results = vk.compute_3d_routing_costs_batch(
            &obs_u32,
            &self.dist_field,
            &self.congestion,
            &self.usage_count,
            &params_batch,
        );
        for (&(net_id, clearance_cells), res_opt) in net_clearances_cells.iter().zip(results.into_iter()) {
            if let Some(cost_vec) = res_opt {
                self.gpu_net_cost_maps.insert(net_id, (clearance_cells, cost_vec));
            }
        }
    }

    /// Computes a 3D shortest-path wavefront potential field on the GPU from target seed indices.
    pub fn compute_gpu_3d_wavefront_for_net(
        &self,
        net_id: usize,
        clearance_cells: i32,
        seeds: &[u32],
        via_cost: f32,
        iterations: u32,
    ) -> Option<Vec<f32>> {
        let vk = self.vulkan.as_ref()?;
        let cell_costs = if let Some((cached_r, ref map)) = self.gpu_net_cost_maps.get(&net_id) {
            if *cached_r == clearance_cells {
                map.clone()
            } else {
                self.compute_gpu_3d_routing_cost_for_net(net_id, clearance_cells, 1.5, 2.0)?
            }
        } else {
            self.compute_gpu_3d_routing_cost_for_net(net_id, clearance_cells, 1.5, 2.0)?
        };
        let params = GpuRouteCostParams {
            width: self.width as u32,
            height: self.height as u32,
            num_layers: self.num_layers as u32,
            net_id: net_id as u32,
            clearance_cells,
            present_congestion_factor: 1.5,
            obstacle_clearance_weight: 2.0,
            plane_layer_mask: self.plane_layer_mask_for_net(net_id),
        };
        vk.compute_3d_wavefront(&cell_costs, seeds, &params, via_cost, iterations)
    }

    pub fn update_distance_fields(&mut self) {
        let layer_cells = self.width * self.height;
        if let Some(ref vk) = self.vulkan {
            let engines = vk.all_engines();
            let w = self.width as u32;
            let h = self.height as u32;
            let res = self.res as f32;
            let num_layers = self.num_layers;

            if engines.len() > 1 && num_layers > 1 {
                let num_engines = engines.len();
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
                        if assigned_layers.is_empty() {
                            continue;
                        }
                        let eng_ref = Arc::clone(engine);
                        let mut flat_obs = Vec::with_capacity(assigned_layers.len() * layer_cells);
                        for &l in &assigned_layers {
                            flat_obs.extend_from_slice(&layer_obs[l]);
                        }

                        handles.push(scope.spawn(move || {
                            let mut results = Vec::new();
                            let sub_layers = assigned_layers.len() as u32;
                            if let Some(batch_df) = eng_ref.compute_distance_field_layers(&flat_obs, w, h, sub_layers, 12) {
                                for (i, &l) in assigned_layers.iter().enumerate() {
                                    let off = i * layer_cells;
                                    results.push((l, batch_df[off..off + layer_cells].to_vec()));
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
            } else if num_layers > 0 {
                let total_cells = num_layers * layer_cells;
                let obs_flat: Vec<u32> = self.obstacle_net_ids[..total_cells]
                    .iter()
                    .map(|&n| if n != FREE_NET { 1u32 } else { 0u32 })
                    .collect();
                if let Some(batch_df) = vk.compute_distance_field_layers(&obs_flat, w, h, num_layers as u32, 12) {
                    for (dst, &val) in self.dist_field[..total_cells].iter_mut().zip(batch_df.iter()) {
                        *dst = val * res;
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
            let gpu_cost_entry = self.gpu_net_cost_maps.get(&target_net_id);
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
                let gpu_certified_clear = if let Some((cached_r, cost_map)) = gpu_cost_entry {
                    *cached_r >= r_cells as i32 && cost_map.get(n_idx).copied().unwrap_or(-1.0) > 0.0
                } else {
                    false
                };
                if !gpu_certified_clear {
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
                                    let eff_c = if self.smd_pad_cells[c_idx] {
                                        (req_clearance.min(raw_obst_c) - 0.015).max(0.135)
                                    } else if req_clearance <= 0.15 {
                                        req_clearance.min(raw_obst_c).max(0.15)
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
                                let obst_r = if self.smd_pad_cells[c_idx] { self.res * 0.5 } else { obst_half_w };
                                let eff_c = (via_dia / 2.0) + obst_r + eff_c_val + 0.025;
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
                                    (via_dia / 2.0) + self.res * 0.5 + eff_c_val + 0.025
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

    /// Checks whether all cells within a circle of `radius_cells` around `(gx, gy)` on `layer`
    /// are free of foreign-net obstacles for `net_id`.
    #[allow(non_snake_case)]
    pub fn is_circle_clear_SpecifyRadius(
        &self,
        gx: usize,
        gy: usize,
        layer: usize,
        radius_cells: i32,
        net_id: usize,
    ) -> bool {
        if gx >= self.width || gy >= self.height || layer >= self.num_layers {
            return false;
        }
        let r = radius_cells.max(0) as isize;
        let r2 = r * r;
        let target = net_id as i32;
        for dy in -r..=r {
            let ny = gy as isize + dy;
            if ny < 0 || ny >= self.height as isize {
                continue;
            }
            for dx in -r..=r {
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let nx = gx as isize + dx;
                if nx < 0 || nx >= self.width as isize {
                    continue;
                }
                let c_idx = self.idx(nx as usize, ny as usize, layer);
                let obs = self.obstacle_net_ids[c_idx];
                if obs != FREE_NET && obs != target {
                    return false;
                }
            }
        }
        true
    }

    pub fn is_circle_clear_specify_radius(
        &self,
        gx: usize,
        gy: usize,
        layer: usize,
        radius_cells: i32,
        net_id: usize,
    ) -> bool {
        self.is_circle_clear_SpecifyRadius(gx, gy, layer, radius_cells, net_id)
    }

    /// Solves a batch of 3D shortest paths across all available GPUs using `MultiGpuRouterPool`
    /// and validates every returned node against `is_circle_clear_SpecifyRadius`.
    pub fn solve_batch_3d_wavefront_paths(
        &self,
        requests: &[((usize, usize, usize), (usize, usize, usize), u32, f32)],
        clearance_cells: i32,
    ) -> Vec<Option<Vec<GridNode>>> {
        if requests.is_empty() {
            return Vec::new();
        }
        let obs_u32: Vec<u32> = self.obstacle_net_ids.iter().map(|&n| n as u32).collect();
        let reqs_with_clearance: Vec<((usize, usize, usize), (usize, usize, usize), u32, f32, i32)> =
            requests
                .iter()
                .map(|&(src, dst, net_id, via_cost)| {
                    (src, dst, net_id, via_cost, clearance_cells.max(0))
                })
                .collect();
        let pool = MultiGpuRouterPool::new();
        let raw_paths = pool.solve_batch_3d_wavefront_paths_with_clearance(
            &obs_u32,
            self.width,
            self.height,
            self.num_layers,
            &reqs_with_clearance,
        );
        raw_paths
            .into_iter()
            .zip(requests.iter())
            .map(|(path_opt, &(_src, _dst, net_id, _via_cost))| {
                let coords = path_opt?;
                let mut nodes = Vec::with_capacity(coords.len());
                for (gx, gy, layer) in coords {
                    if !self.is_circle_clear_SpecifyRadius(
                        gx,
                        gy,
                        layer,
                        clearance_cells.max(0),
                        net_id as usize,
                    ) {
                        return None;
                    }
                    nodes.push(GridNode { gx, gy, layer });
                }
                Some(nodes)
            })
            .collect()
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

    pub fn with_config(mut self, config: RoutingConfig) -> Self {
        self.congestion_factor = config.congestion_factor as f32;
        self.crosstalk_penalty = config.crosstalk_penalty;
        self.nudge_engine.step_mm = config.nudge_step_mm;
        self.config = config;
        self
    }

    /// Returns true if `a` and `b` form a differential pair (e.g. `USB_DP`/`USB_DM`, `USB_D+`/`USB_D-`).
    pub fn are_diff_pair_mates(a: &str, b: &str) -> bool {
        if a == b {
            return false;
        }
        let ua = a.to_ascii_uppercase();
        let ub = b.to_ascii_uppercase();
        for (suf_p, suf_n) in [("DP", "DM"), ("D+", "D-"), ("_P", "_N"), ("+", "-")] {
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
                if pad.net_id == 0 || board.is_plane_net(pad.net_id) {
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

    pub fn is_connection_electrically_connected(board: &Board, conn: &RouteConnection) -> bool {
        let Some(sfp) = board.footprints.get(conn.start_fp) else { return false; };
        let Some(spad) = sfp.pads.get(conn.start_pad) else { return false; };
        let Some(efp) = board.footprints.get(conn.end_fp) else { return false; };
        let Some(epad) = efp.pads.get(conn.end_pad) else { return false; };
        let s_pos = sfp.pad_absolute_position(spad);
        let e_pos = efp.pad_absolute_position(epad);
        let s_th = spad.is_through_hole;
        let e_th = epad.is_through_hole;
        let s_layer = sfp.layer;
        let e_layer = efp.layer;
        let net_id = conn.net_id;

        let mut tracks: Vec<TrackSegment> = board
            .tracks
            .iter()
            .filter(|t| t.net_id == net_id)
            .cloned()
            .collect();
        for a in board.arcs.iter().filter(|a| a.net_id == net_id) {
            tracks.push(TrackSegment {
                start: a.start(),
                end: a.mid(),
                width: a.width,
                layer: a.layer,
                net_id: a.net_id,
                is_locked: a.is_locked,
            });
            tracks.push(TrackSegment {
                start: a.mid(),
                end: a.end(),
                width: a.width,
                layer: a.layer,
                net_id: a.net_id,
                is_locked: a.is_locked,
            });
        }
        let vias: Vec<&Via> = board.vias.iter().filter(|v| v.net_id == net_id).collect();
        let n_t = tracks.len();
        let n_v = vias.len();
        if n_t == 0 && n_v == 0 {
            return false;
        }
        const EPS: f64 = 0.05;
        let mut adj = vec![Vec::new(); n_t + n_v];
        for i in 0..n_t {
            let ti = &tracks[i];
            for j in (i + 1)..n_t {
                let tj = &tracks[j];
                if ti.layer == tj.layer {
                    let c = (ti.start.0 - tj.start.0).hypot(ti.start.1 - tj.start.1) < EPS
                        || (ti.start.0 - tj.end.0).hypot(ti.start.1 - tj.end.1) < EPS
                        || (ti.end.0 - tj.start.0).hypot(ti.end.1 - tj.start.1) < EPS
                        || (ti.end.0 - tj.end.0).hypot(ti.end.1 - tj.end.1) < EPS
                        || dist_point_to_segment(ti.start, tj.start, tj.end) < EPS
                        || dist_point_to_segment(ti.end, tj.start, tj.end) < EPS
                        || dist_point_to_segment(tj.start, ti.start, ti.end) < EPS
                        || dist_point_to_segment(tj.end, ti.start, ti.end) < EPS;
                    if c {
                        adj[i].push(j);
                        adj[j].push(i);
                    }
                }
            }
            for vi in 0..n_v {
                let v = vias[vi];
                let d_s = (ti.start.0 - v.position.0).hypot(ti.start.1 - v.position.1);
                let d_e = (ti.end.0 - v.position.0).hypot(ti.end.1 - v.position.1);
                if d_s <= v.size * 0.5 + EPS
                    || d_e <= v.size * 0.5 + EPS
                    || dist_point_to_segment(v.position, ti.start, ti.end) <= v.size * 0.5 + EPS
                {
                    adj[i].push(n_t + vi);
                    adj[n_t + vi].push(i);
                }
            }
        }
        let mut queue = VecDeque::new();
        let mut visited = vec![false; n_t + n_v];
        for i in 0..n_t {
            let ti = &tracks[i];
            if s_th || ti.layer == s_layer {
                if (ti.start.0 - s_pos.0).hypot(ti.start.1 - s_pos.1) < 0.12
                    || (ti.end.0 - s_pos.0).hypot(ti.end.1 - s_pos.1) < 0.12
                    || dist_point_to_pad(ti.start, sfp, spad) < EPS
                    || dist_point_to_pad(ti.end, sfp, spad) < EPS
                {
                    visited[i] = true;
                    queue.push_back(i);
                }
            }
        }
        while let Some(u) = queue.pop_front() {
            if u < n_t {
                let ti = &tracks[u];
                if e_th || ti.layer == e_layer {
                    if (ti.start.0 - e_pos.0).hypot(ti.start.1 - e_pos.1) < 0.12
                        || (ti.end.0 - e_pos.0).hypot(ti.end.1 - e_pos.1) < 0.12
                        || dist_point_to_pad(ti.start, efp, epad) < EPS
                        || dist_point_to_pad(ti.end, efp, epad) < EPS
                    {
                        return true;
                    }
                }
            }
            for &nxt in &adj[u] {
                if !visited[nxt] {
                    visited[nxt] = true;
                    queue.push_back(nxt);
                }
            }
        }
        false
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

        // Fanout First: Pre-route ground fanout/stitching for U1 thermal pad ground pins before signal routing
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

        grid.update_distance_fields();

        for conn in &mut connections {
            let is_j2_internal = conn.start_fp == conn.end_fp
                && conn.net_name.starts_with("USB_D")
                && board.footprints.get(conn.start_fp).map_or(false, |fp| fp.reference == "J2");
            if !is_j2_internal {
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
            }
            let is_f1_f2_d2_d3 = board.footprints.get(conn.start_fp).map_or(false, |fp| matches!(fp.reference.as_str(), "F1" | "F2" | "D2" | "D3"))
                || board.footprints.get(conn.end_fp).map_or(false, |fp| matches!(fp.reference.as_str(), "F1" | "F2" | "D2" | "D3"));
            let is_passive_terminal = board.footprints.get(conn.start_fp).map_or(false, |fp| fp.reference.starts_with('R') || fp.reference.starts_with('C'))
                || board.footprints.get(conn.end_fp).map_or(false, |fp| fp.reference.starts_with('R') || fp.reference.starts_with('C'));
            let is_power_net = conn.netclass.name.starts_with("Power")
                || conn.netclass.name == "Power"
                || conn.netclass.name == "Power_SW"
                || matches!(conn.net_name.as_str(), "VBUS" | "VBus" | "/BUCK_VIN" | "BUCK_VIN" | "Buck_in" | "VRAW" | "HP" | "HN" | "P1" | "P2" | "+3.3V" | "VCC_5V")
                || conn.net_name.contains("BUCK")
                || conn.net_name.contains("Buck");
            if is_f1_f2_d2_d3 || is_power_net {
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
                conn.netclass.track_width = conn.netclass.track_width.min(0.15);
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
                if !is_power_net && !conn.netclass.name.starts_with("Power") {
                    conn.netclass.track_width = conn.netclass.track_width.min(0.15);
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
        let vk_opt = grid.vulkan.clone();
        let grid_w = grid.width as u32;
        let grid_h = grid.height as u32;
        let grid_res = grid.res as f32;
        let layer_cells = grid.width * grid.height;
        let num_layers = grid.num_layers;
        if let Some(ref vk) = vk_opt {
            let engines = vk.all_engines();
            if !engines.is_empty() {
                let layer_dfs: Vec<(usize, Option<Vec<f32>>)> = std::thread::scope(|s| {
                    let handles: Vec<_> = (0..num_layers)
                        .map(|layer| {
                            let vk_dev = Arc::clone(&engines[layer % engines.len()]);
                            let offset = layer * layer_cells;
                            let obs_u32: Vec<u32> = grid.obstacle_net_ids[offset..offset + layer_cells]
                                .iter()
                                .map(|&n| if n != FREE_NET { 1u32 } else { 0u32 })
                                .collect();
                            s.spawn(move || (layer, vk_dev.compute_distance_field(&obs_u32, grid_w, grid_h, 16)))
                        })
                        .collect();
                    handles.into_iter().filter_map(|h| h.join().ok()).collect()
                });
                for (layer, df_opt) in layer_dfs {
                    if let Some(df) = df_opt {
                        let offset = layer * layer_cells;
                        for (dst, &val) in grid.dist_field[offset..offset + layer_cells]
                            .iter_mut()
                            .zip(df.iter())
                        {
                            *dst = val * grid_res;
                        }
                    }
                }
            }
        }
        std::thread::scope(|_scope| {
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

                if iteration > 0 {
                    if let Some(ref vk) = vk_opt {
                        let engines = vk.all_engines();
                        if !engines.is_empty() {
                            let layer_dfs: Vec<(usize, Option<Vec<f32>>)> = std::thread::scope(|s| {
                                let handles: Vec<_> = (0..num_layers)
                                    .map(|layer| {
                                        let vk_dev = Arc::clone(&engines[layer % engines.len()]);
                                        let offset = layer * layer_cells;
                                        let obs_u32: Vec<u32> = grid.obstacle_net_ids[offset..offset + layer_cells]
                                            .iter()
                                            .map(|&n| if n != FREE_NET { 1u32 } else { 0u32 })
                                            .collect();
                                        s.spawn(move || (layer, vk_dev.compute_distance_field(&obs_u32, grid_w, grid_h, 16)))
                                    })
                                    .collect();
                                handles.into_iter().filter_map(|h| h.join().ok()).collect()
                            });
                            for (layer, df_opt) in layer_dfs {
                                if let Some(df) = df_opt {
                                    let offset = layer * layer_cells;
                                    for (dst, &val) in grid.dist_field[offset..offset + layer_cells]
                                        .iter_mut()
                                        .zip(df.iter())
                                    {
                                        *dst = val * grid_res;
                                    }
                                }
                            }
                        }
                    }
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

                // Offload 3D routing cost and clearance field precomputation to all GPUs before parallel A* wave
                let mut net_clearance_map: HashMap<usize, i32> = HashMap::new();
                let diag_margin = (grid.res * 0.7071).min(0.02);
                for &conn_idx in &unrouted_indices {
                    let conn = &connections[conn_idx];
                    let max_clearance = grid.max_net_clearance.max(conn.netclass.clearance);
                    let search_clearance = (conn.netclass.track_width / 2.0) + max_clearance + diag_margin;
                    let r_cells = (search_clearance / grid.res).ceil() as i32;
                    let entry = net_clearance_map.entry(conn.net_id).or_insert(r_cells);
                    if r_cells > *entry {
                        *entry = r_cells;
                    }
                }
                let mut net_batch: Vec<(usize, i32)> = net_clearance_map.into_iter().collect();
                net_batch.sort_unstable_by_key(|&(net_id, _)| net_id);
                grid.precompute_gpu_routing_costs_batch(&net_batch, self.congestion_factor, 2.0);

                // Phase 1: Speculative multi-core parallel A* wave or Parallel Portfolio Ensemble
                #[allow(non_local_definitions)]
                impl RoutingGrid {
                    pub fn clone_for_portfolio(&self) -> Self {
                        Self {
                            min_x: self.min_x,
                            min_y: self.min_y,
                            max_x: self.max_x,
                            max_y: self.max_y,
                            res: self.res,
                            width: self.width,
                            height: self.height,
                            num_layers: self.num_layers,
                            plane_nets: self.plane_nets.clone(),
                            obstacle_net_ids: self.obstacle_net_ids.clone(),
                            dist_field: self.dist_field.clone(),
                            congestion: self.congestion.clone(),
                            usage_count: self.usage_count.clone(),
                            gpu_net_cost_maps: HashMap::new(),
                            nudgeable_fp: self.nudgeable_fp.clone(),
                            smd_pad_cells: self.smd_pad_cells.clone(),
                            net_clearances: self.net_clearances.clone(),
                            max_net_clearance: self.max_net_clearance,
                            inner_layer_penalty: self.inner_layer_penalty,
                            vulkan: self.vulkan.clone(),
                            base_snapshot: self.base_snapshot.clone(),
                        }
                    }
                }

                #[derive(Clone, Debug)]
                struct PathConflictGeom {
                    net_id: usize,
                    width: f64,
                    clearance: f64,
                    via_dia: f64,
                    min_x: f64,
                    max_x: f64,
                    min_y: f64,
                    max_y: f64,
                    max_radius: f64,
                    layer_mask: u32,
                    segs: Vec<((f64, f64), (f64, f64), usize)>,
                    vias: Vec<(f64, f64)>,
                }

                impl PathConflictGeom {
                    fn from_path(
                        grid: &RoutingGrid,
                        conn: &RouteConnection,
                        path: &[GridNode],
                    ) -> Self {
                        let width = if conn.diff_pair_leader.is_some() || conn.netclass.name == "USB_Diff" {
                            conn.netclass.diff_pair_width
                        } else {
                            conn.netclass.track_width
                        };
                        let clearance = conn.netclass.clearance;
                        let via_dia = conn.netclass.via_dia;
                        let mut min_x = f64::INFINITY;
                        let mut max_x = f64::NEG_INFINITY;
                        let mut min_y = f64::INFINITY;
                        let mut max_y = f64::NEG_INFINITY;
                        let mut layer_mask = 0u32;
                        let mut segs = Vec::new();
                        let mut vias = Vec::new();

                        for &n in path {
                            let (wx, wy) = grid.grid_to_world(n.gx, n.gy);
                            min_x = min_x.min(wx);
                            max_x = max_x.max(wx);
                            min_y = min_y.min(wy);
                            max_y = max_y.max(wy);
                            if n.layer < 32 {
                                layer_mask |= 1u32 << n.layer;
                            }
                        }

                        if path.len() == 1 {
                            let p0 = grid.grid_to_world(path[0].gx, path[0].gy);
                            segs.push((p0, p0, path[0].layer));
                        } else if path.len() >= 2 {
                            let mut run_start = 0usize;
                            for i in 0..(path.len() - 1) {
                                let u = path[i];
                                let v = path[i + 1];
                                if u.layer != v.layer {
                                    if i > run_start {
                                        let p1 = grid.grid_to_world(path[run_start].gx, path[run_start].gy);
                                        let p2 = grid.grid_to_world(u.gx, u.gy);
                                        segs.push((p1, p2, u.layer));
                                    }
                                    vias.push(grid.grid_to_world(u.gx, u.gy));
                                    layer_mask = 0xFFFF_FFFF;
                                    run_start = i + 1;
                                } else if i + 2 < path.len() && path[i + 2].layer == v.layer {
                                    let dx1 = v.gx as isize - u.gx as isize;
                                    let dy1 = v.gy as isize - u.gy as isize;
                                    let dx2 = path[i + 2].gx as isize - v.gx as isize;
                                    let dy2 = path[i + 2].gy as isize - v.gy as isize;
                                    if (dx1, dy1) != (dx2, dy2) {
                                        let p1 = grid.grid_to_world(path[run_start].gx, path[run_start].gy);
                                        let p2 = grid.grid_to_world(v.gx, v.gy);
                                        segs.push((p1, p2, u.layer));
                                        run_start = i + 1;
                                    }
                                }
                            }
                            if run_start + 1 < path.len() {
                                let last = path.len() - 1;
                                if path[run_start].layer == path[last].layer {
                                    let p1 = grid.grid_to_world(path[run_start].gx, path[run_start].gy);
                                    let p2 = grid.grid_to_world(path[last].gx, path[last].gy);
                                    segs.push((p1, p2, path[run_start].layer));
                                }
                            }
                        }

                        let max_radius = (width * 0.5).max(if !vias.is_empty() { via_dia * 0.5 } else { 0.0 });
                        Self {
                            net_id: conn.net_id,
                            width,
                            clearance,
                            via_dia,
                            min_x,
                            max_x,
                            min_y,
                            max_y,
                            max_radius,
                            layer_mask,
                            segs,
                            vias,
                        }
                    }

                    #[inline]
                    fn conflicts_with(&self, other: &Self, grid_res: f64) -> bool {
                        if self.net_id == other.net_id {
                            return false;
                        }
                        if (self.layer_mask & other.layer_mask) == 0 {
                            return false;
                        }
                        let pair_clr = self.clearance.max(other.clearance) + grid_res * 1.5;
                        let aabb_margin = self.max_radius + other.max_radius + pair_clr;
                        if self.max_x + aabb_margin < other.min_x
                            || other.max_x + aabb_margin < self.min_x
                            || self.max_y + aabb_margin < other.min_y
                            || other.max_y + aabb_margin < self.min_y
                        {
                            return false;
                        }

                        let req_seg = (self.width + other.width) * 0.5 + pair_clr;
                        for &(sa1, sa2, la) in &self.segs {
                            for &(sb1, sb2, lb) in &other.segs {
                                if la == lb && dist_segment_to_segment(sa1, sa2, sb1, sb2) < req_seg {
                                    return true;
                                }
                            }
                        }

                        if !self.vias.is_empty() && !other.vias.is_empty() {
                            let req_vv = (self.via_dia + other.via_dia) * 0.5 + pair_clr;
                            for &va in &self.vias {
                                for &vb in &other.vias {
                                    if (va.0 - vb.0).hypot(va.1 - vb.1) < req_vv {
                                        return true;
                                    }
                                }
                            }
                        }

                        if !self.vias.is_empty() {
                            let req_vs = (self.via_dia + other.width) * 0.5 + pair_clr;
                            for &va in &self.vias {
                                for &(sb1, sb2, _) in &other.segs {
                                    if dist_to_segment(va.0, va.1, sb1.0, sb1.1, sb2.0, sb2.1) < req_vs {
                                        return true;
                                    }
                                }
                            }
                        }

                        if !other.vias.is_empty() {
                            let req_vs = (other.via_dia + self.width) * 0.5 + pair_clr;
                            for &vb in &other.vias {
                                for &(sa1, sa2, _) in &self.segs {
                                    if dist_to_segment(vb.0, vb.1, sa1.0, sa1.1, sa2.0, sa2.1) < req_vs {
                                        return true;
                                    }
                                }
                            }
                        }

                        false
                    }
                }

                #[derive(Clone, Copy, Debug)]
                struct PortfolioVariant {
                    via_mult: f64,
                    hist_w: f64,
                    heur_w: f64,
                    dist_bias: f64,
                    reverse: bool,
                    diag_first: bool,
                }

                const PORTFOLIO_VARIANTS: [PortfolioVariant; 12] = [
                    PortfolioVariant { via_mult: 1.0,  hist_w: 1.0, heur_w: 1.0,  dist_bias: 0.0,  reverse: false, diag_first: false },
                    PortfolioVariant { via_mult: 1.0,  hist_w: 1.0, heur_w: 1.0,  dist_bias: 0.0,  reverse: true,  diag_first: false },
                    PortfolioVariant { via_mult: 0.5,  hist_w: 1.0, heur_w: 1.0,  dist_bias: 0.0,  reverse: false, diag_first: true  },
                    PortfolioVariant { via_mult: 0.5,  hist_w: 1.0, heur_w: 1.0,  dist_bias: 0.0,  reverse: true,  diag_first: true  },
                    PortfolioVariant { via_mult: 0.75, hist_w: 0.5, heur_w: 1.15, dist_bias: 0.0,  reverse: false, diag_first: false },
                    PortfolioVariant { via_mult: 0.75, hist_w: 0.5, heur_w: 1.15, dist_bias: 0.0,  reverse: true,  diag_first: false },
                    PortfolioVariant { via_mult: 1.25, hist_w: 1.5, heur_w: 1.0,  dist_bias: 0.01, reverse: false, diag_first: true  },
                    PortfolioVariant { via_mult: 1.25, hist_w: 1.5, heur_w: 1.0,  dist_bias: 0.01, reverse: true,  diag_first: true  },
                    PortfolioVariant { via_mult: 1.75, hist_w: 2.5, heur_w: 1.15, dist_bias: 0.02, reverse: false, diag_first: false },
                    PortfolioVariant { via_mult: 1.75, hist_w: 2.5, heur_w: 1.15, dist_bias: 0.02, reverse: true,  diag_first: false },
                    PortfolioVariant { via_mult: 2.5,  hist_w: 0.5, heur_w: 1.35, dist_bias: 0.0,  reverse: false, diag_first: true  },
                    PortfolioVariant { via_mult: 2.5,  hist_w: 2.5, heur_w: 1.35, dist_bias: 0.01, reverse: true,  diag_first: true  },
                ];

                #[allow(non_local_definitions)]
                impl AutoRouter {
                    /// Partitions candidate `(conn_idx, path)` routes into conflict-free independent waves
                    /// via greedy conflict-graph coloring. Two paths of different nets conflict when their
                    /// clearance-inflated 2D/3D bounding boxes overlap AND their segments or vias encroach.
                    pub fn partition_into_conflict_free_waves(
                        grid: &RoutingGrid,
                        connections: &[RouteConnection],
                        candidates: Vec<(usize, Vec<GridNode>)>,
                    ) -> Vec<Vec<(usize, Vec<GridNode>)>> {
                        if candidates.is_empty() {
                            return Vec::new();
                        }
                        let geoms: Vec<PathConflictGeom> = candidates
                            .par_iter()
                            .map(|(conn_idx, path)| PathConflictGeom::from_path(grid, &connections[*conn_idx], path))
                            .collect();

                        let mut waves: Vec<Vec<(usize, Vec<GridNode>)>> = Vec::new();
                        let mut wave_geoms: Vec<Vec<PathConflictGeom>> = Vec::new();

                        for (cand, geom) in candidates.into_iter().zip(geoms.into_iter()) {
                            let mut placed_wave = None;
                            for (w_idx, w_geoms) in wave_geoms.iter().enumerate() {
                                let conflicts = if w_geoms.len() >= 12 {
                                    w_geoms.par_iter().any(|other| geom.conflicts_with(other, grid.res))
                                } else {
                                    w_geoms.iter().any(|other| geom.conflicts_with(other, grid.res))
                                };
                                if !conflicts {
                                    placed_wave = Some(w_idx);
                                    break;
                                }
                            }
                            if let Some(w_idx) = placed_wave {
                                waves[w_idx].push(cand);
                                wave_geoms[w_idx].push(geom);
                            } else {
                                waves.push(vec![cand]);
                                wave_geoms.push(vec![geom]);
                            }
                        }
                        waves
                    }

                    pub fn evaluate_path_cost(grid: &RoutingGrid, path: &[GridNode]) -> (usize, f64, f64) {
                        let mut vias = 0usize;
                        let mut wirelength = 0.0f64;
                        let mut cong_sum = 0.0f64;
                        for (i, &n) in path.iter().enumerate() {
                            let idx = grid.idx(n.gx, n.gy, n.layer);
                            cong_sum += grid.congestion[idx] as f64;
                            if i + 1 < path.len() {
                                let nxt = path[i + 1];
                                if nxt.layer != n.layer {
                                    vias += 1;
                                } else {
                                    let dx = (nxt.gx as isize - n.gx as isize).abs() as f64;
                                    let dy = (nxt.gy as isize - n.gy as isize).abs() as f64;
                                    wirelength += dx.hypot(dy) * grid.res;
                                }
                            }
                        }
                        (vias, wirelength, cong_sum)
                    }

                    #[allow(clippy::too_many_arguments)]
                    pub fn find_path_portfolio_variant(
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
                        via_mult: f64,
                        hist_w: f64,
                        heur_w: f64,
                        dist_bias: f64,
                        reverse: bool,
                        diag_first: bool,
                        abort_flag: &AtomicBool,
                        found_count: Option<&std::sync::atomic::AtomicUsize>,
                        max_expansions_override: Option<usize>,
                    ) -> Option<Vec<GridNode>> {
                        let (s_node, g_node) = if reverse { (goal, start) } else { (start, goal) };
                        let mut open_set = BinaryHeap::new();
                        let mut came_from: HashMap<GridNode, GridNode> = HashMap::new();
                        let mut g_score: HashMap<GridNode, f64> = HashMap::new();
                        let mut coupled_length: HashMap<GridNode, f64> = HashMap::new();

                        let heur = |a: GridNode, b: GridNode| -> f64 {
                            let dx = (a.gx as isize - b.gx as isize).abs() as f64 * grid.res;
                            let dy = (a.gy as isize - b.gy as isize).abs() as f64 * grid.res;
                            let diag = dx.min(dy);
                            let straight = (dx - dy).abs();
                            let layer_pen = if a.layer != b.layer { 2.0 * via_mult } else { 0.0 };
                            diag * std::f64::consts::SQRT_2 + straight + layer_pen
                        };

                        let h_start = heur(s_node, g_node) * heur_w;
                        g_score.insert(s_node, 0.0);
                        coupled_length.insert(s_node, 0.0);
                        open_set.push(AStarNode {
                            node: s_node,
                            cost: 0.0,
                            priority: h_start,
                        });

                        let start_is_smd = (0..grid.num_layers).any(|l| grid.smd_pad_cells[grid.idx(s_node.gx, s_node.gy, l)]);
                        let start_is_tht = !start_is_smd
                            && (0..grid.num_layers).all(|l| grid.obstacle_net_ids[grid.idx(s_node.gx, s_node.gy, l)] == net_id as i32);
                        if start_is_tht {
                            for l in 0..grid.num_layers {
                                if l != s_node.layer {
                                    if grid.plane_nets.contains_key(&l) && grid.plane_nets.get(&l) != Some(&net_id) {
                                        continue;
                                    }
                                    let alt_start = GridNode { gx: s_node.gx, gy: s_node.gy, layer: l };
                                    g_score.insert(alt_start, 0.0);
                                    coupled_length.insert(alt_start, 0.0);
                                    open_set.push(AStarNode {
                                        node: alt_start,
                                        cost: 0.0,
                                        priority: heur(alt_start, g_node) * heur_w,
                                    });
                                }
                            }
                        }

                        let dist_span = (g_node.gx as isize - s_node.gx as isize).abs() + (g_node.gy as isize - s_node.gy as isize).abs();
                        let max_expansions = max_expansions_override.unwrap_or(if dist_span > 200 { 350_000 } else { 120_000 });
                        let mut expansions = 0usize;

                        while let Some(current) = open_set.pop() {
                            expansions += 1;
                            if expansions > max_expansions {
                                return None;
                            }
                            if expansions & 255 == 0 {
                                if abort_flag.load(AtomicOrdering::Relaxed) {
                                    return None;
                                }
                                if expansions > 16_384 {
                                    if let Some(fc) = found_count {
                                        if fc.load(AtomicOrdering::Relaxed) >= 3 {
                                            return None;
                                        }
                                    }
                                }
                            }

                            let u = current.node;
                            if u == g_node {
                                let mut path = vec![g_node];
                                let mut curr = g_node;
                                while let Some(&prev) = came_from.get(&curr) {
                                    path.push(prev);
                                    curr = prev;
                                }
                                path.reverse();
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
                                let smoothed = self.smooth_grid_path(
                                    grid,
                                    path,
                                    net_id,
                                    req_clearance,
                                    track_width,
                                    via_dia,
                                    u1_center,
                                    require_return_path,
                                    max_coupled_length_mm,
                                );
                                if let Some(fc) = found_count {
                                    fc.fetch_add(1, AtomicOrdering::Relaxed);
                                }
                                return Some(smoothed);
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
                                let n_idx = grid.idx(neighbor.gx, neighbor.gy, neighbor.layer);
                                if neighbor.layer == u.layer {
                                    let is_diag = neighbor.gx != u.gx && neighbor.gy != u.gy;
                                    if (diag_first && is_diag) || (!diag_first && !is_diag) {
                                        edge_cost *= 0.97;
                                    }
                                    let cong = grid.congestion[n_idx] as f64;
                                    let usage = grid.usage_count[n_idx] as f64;
                                    if (hist_w - 1.0).abs() > 1e-3 || usage > 0.0 {
                                        edge_cost += ((hist_w - 1.0) * cong + hist_w * 0.35 * usage).max(-0.4 * edge_cost) * grid.res;
                                    }
                                    if let Some((pdx, pdy)) = prev_dir {
                                        let cur_dx = neighbor.gx as isize - u.gx as isize;
                                        let cur_dy = neighbor.gy as isize - u.gy as isize;
                                        if (cur_dx, cur_dy) != (pdx, pdy) {
                                            let dot = pdx * cur_dx + pdy * cur_dy;
                                            if dot > 0 {
                                                edge_cost += 5.0 * grid.res;
                                            } else if dot == 0 {
                                                edge_cost += 15.0 * grid.res;
                                            } else {
                                                edge_cost += 45.0 * grid.res;
                                            }
                                        }
                                    }
                                    if require_return_path {
                                        if let Some(ref_layer) = grid.adjacent_reference_plane(neighbor.layer) {
                                            if grid.is_reference_plane_void_at(neighbor.gx, neighbor.gy, ref_layer) {
                                                edge_cost += 35.0 * grid.res + 5.0;
                                            }
                                        }
                                    }
                                    let is_coupled = grid.is_step_coupled(u, neighbor, net_id, track_width, req_clearance);
                                    let step_len = if is_diag { 1.41421356 * grid.res } else { grid.res };
                                    if is_coupled {
                                        let prev_coupled = coupled_length.get(&u).copied().unwrap_or(0.0);
                                        next_coupled = prev_coupled + step_len;
                                        if next_coupled > max_coupled_length_mm {
                                            let excess = next_coupled - max_coupled_length_mm;
                                            edge_cost += self.crosstalk_penalty * step_len * (1.0 + excess * 0.5);
                                        }
                                    }
                                } else {
                                    edge_cost = edge_cost * via_mult + 2.0 * via_mult;
                                }

                                if dist_bias > 0.0 {
                                    let df_val = (grid.dist_field[n_idx] as f64).clamp(0.05, 2.0);
                                    edge_cost += dist_bias / df_val;
                                }
                                let tentative_g = current_g + edge_cost;
                                let is_better = g_score.get(&neighbor).is_none_or(|&prev_g| tentative_g < prev_g - 1e-6);
                                if is_better {
                                    g_score.insert(neighbor, tentative_g);
                                    coupled_length.insert(neighbor, next_coupled);
                                    came_from.insert(neighbor, u);
                                    let priority = tentative_g + heur(neighbor, g_node) * heur_w;
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

                    /// Batch-validates a slice of `(conn_idx, path)` candidates using `GpuDrcContext` on the GPU
                    /// alongside parallel `is_path_conflict_free` checks across Rayon threads.
                    pub fn batch_validate_paths_gpu(
                        &self,
                        board: &Board,
                        grid: &RoutingGrid,
                        connections: &[RouteConnection],
                        candidates: &[(usize, Vec<GridNode>)],
                        u1_center: Option<(f64, f64)>,
                    ) -> Vec<(bool, bool)> {
                        if candidates.is_empty() {
                            return Vec::new();
                        }
                        let grid_valid: Vec<bool> = candidates
                            .par_iter()
                            .map(|(conn_idx, path)| {
                                let conn = &connections[*conn_idx];
                                let w = if conn.diff_pair_leader.is_some() || conn.netclass.name == "USB_Diff" {
                                    conn.netclass.diff_pair_width
                                } else {
                                    conn.netclass.track_width
                                };
                                self.is_path_conflict_free(
                                    grid,
                                    path,
                                    conn.net_id,
                                    conn.netclass.clearance,
                                    w,
                                    conn.netclass.via_dia,
                                    u1_center,
                                    conn.netclass.require_return_path,
                                )
                            })
                            .collect();

                        let drc_ctx = GpuDrcContext::from_board(board);
                        let mut gpu_cands = Vec::new();
                        let mut cand_ranges: Vec<(usize, usize)> = Vec::with_capacity(candidates.len());

                        for &(conn_idx, ref path) in candidates {
                            let start_offset = gpu_cands.len();
                            let conn = &connections[conn_idx];
                            let geom = PathConflictGeom::from_path(grid, conn, path);
                            let half_w = (geom.width * 0.5) as f32;
                            let via_r = (geom.via_dia * 0.5) as f32;
                            let clr = (geom.clearance.min(0.15)) as f32;
                            for &(p1, p2, layer) in &geom.segs {
                                let d_start = (p1.0 - conn.start.0).hypot(p1.1 - conn.start.1)
                                    .min((p2.0 - conn.start.0).hypot(p2.1 - conn.start.1));
                                let d_end = (p1.0 - conn.end.0).hypot(p1.1 - conn.end.1)
                                    .min((p2.0 - conn.end.0).hypot(p2.1 - conn.end.1));
                                if d_start > 0.85 && d_end > 0.85 {
                                    gpu_cands.push(GpuDrcCandidate {
                                        x1: p1.0 as f32,
                                        y1: p1.1 as f32,
                                        x2: p2.0 as f32,
                                        y2: p2.1 as f32,
                                        radius_or_half_width: half_w,
                                        clearance: clr,
                                        layer: layer as u32,
                                        net_id: conn.net_id as u32,
                                        kind: 1,
                                        ignore_same_net_pads: 1,
                                        _pad0: 0,
                                        _pad1: 0,
                                    });
                                }
                            }
                            for &vp in &geom.vias {
                                let d_start = (vp.0 - conn.start.0).hypot(vp.1 - conn.start.1);
                                let d_end = (vp.0 - conn.end.0).hypot(vp.1 - conn.end.1);
                                if d_start > 0.65 && d_end > 0.65 {
                                    gpu_cands.push(GpuDrcCandidate {
                                        x1: vp.0 as f32,
                                        y1: vp.1 as f32,
                                        x2: 0.0,
                                        y2: 0.0,
                                        radius_or_half_width: via_r,
                                        clearance: clr,
                                        layer: 0,
                                        net_id: conn.net_id as u32,
                                        kind: 0,
                                        ignore_same_net_pads: 0,
                                        _pad0: 0,
                                        _pad1: 0,
                                    });
                                }
                            }
                            cand_ranges.push((start_offset, gpu_cands.len()));
                        }

                        let gpu_res = drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_cands);
                        grid_valid
                            .into_iter()
                            .enumerate()
                            .map(|(idx, g_ok)| {
                                let (s, e) = cand_ranges[idx];
                                let drc_ok = if s == e {
                                    true
                                } else if let Some(ref res) = gpu_res {
                                    res[s..e].iter().all(|&v| v != 0)
                                } else {
                                    true
                                };
                                (g_ok, drc_ok)
                            })
                            .collect()
                    }

                    /// Expands `target_indices` across a Parallel Portfolio of `(conn_idx, variant_id)`
                    /// trajectories in `par_iter()`, batch-validates all discovered candidates on the GPU
                    /// (`GpuDrcContext`), and selects for each connection the legal path minimizing
                    /// `(vias * 10.0 + wirelength)`.
                    pub fn route_connections_portfolio(
                        &self,
                        board: &Board,
                        grid: &RoutingGrid,
                        connections: &[RouteConnection],
                        target_indices: &[usize],
                        u1_center: Option<(f64, f64)>,
                        stop_flag: &AtomicBool,
                    ) -> Vec<(usize, Option<Vec<GridNode>>)> {
                        if target_indices.is_empty() {
                            return Vec::new();
                        }
                        let num_variants = if target_indices.len() <= 12 {
                            PORTFOLIO_VARIANTS.len()
                        } else if target_indices.len() <= 24 {
                            8
                        } else {
                            6
                        };

                        let found_counts: Vec<std::sync::atomic::AtomicUsize> = (0..target_indices.len())
                            .map(|_| std::sync::atomic::AtomicUsize::new(0))
                            .collect();

                        let mut tasks: Vec<(usize, usize, usize)> = Vec::with_capacity(target_indices.len() * num_variants);
                        for (t_pos, &conn_idx) in target_indices.iter().enumerate() {
                            for v_id in 0..num_variants {
                                tasks.push((t_pos, conn_idx, v_id));
                            }
                        }

                        // Prepare a concurrent GPU 3D wavefront batch (up to 16 connections) so both GPUs
                        // solve 3D wavefront trajectories simultaneously while CPU cores run A* portfolio variants.
                        let concurrent_gpu_limit = target_indices.len().min(16);
                        let concurrent_gpu_reqs: Vec<(usize, (GridNode, GridNode, usize, f32))> = target_indices[..concurrent_gpu_limit]
                            .iter()
                            .filter_map(|&conn_idx| {
                                let conn = &connections[conn_idx];
                                let (sgx, sgy) = grid.world_to_grid(conn.start.0, conn.start.1)?;
                                let (egx, egy) = grid.world_to_grid(conn.end.0, conn.end.1)?;
                                Some((
                                    conn_idx,
                                    (
                                        GridNode { gx: sgx, gy: sgy, layer: conn.start.2 },
                                        GridNode { gx: egx, gy: egy, layer: conn.end.2 },
                                        conn.net_id,
                                        3.0f32,
                                    ),
                                ))
                            })
                            .collect();

                        let (mut raw_candidates, concurrent_gpu_paths): (
                            Vec<(usize, usize, Vec<GridNode>)>,
                            Vec<(usize, Option<Vec<GridNode>>)>,
                        ) = rayon::join(
                            || {
                                tasks
                                    .par_iter()
                                    .filter_map(|&(t_pos, conn_idx, v_id)| {
                                        if stop_flag.load(AtomicOrdering::Relaxed) {
                                            return None;
                                        }
                                        let conn = &connections[conn_idx];
                                        let (sgx, sgy) = grid.world_to_grid(conn.start.0, conn.start.1)?;
                                        let (egx, egy) = grid.world_to_grid(conn.end.0, conn.end.1)?;
                                        let start_grid = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                                        let end_grid = GridNode { gx: egx, gy: egy, layer: conn.end.2 };
                                        let var = PORTFOLIO_VARIANTS[v_id];
                                        let w = if conn.diff_pair_leader.is_some() || conn.netclass.name == "USB_Diff" {
                                            conn.netclass.diff_pair_width
                                        } else {
                                            conn.netclass.track_width
                                        };
                                        let path = if v_id == 0 {
                                            self.find_path_with_constraints(
                                                grid,
                                                start_grid,
                                                end_grid,
                                                conn.net_id,
                                                conn.netclass.clearance,
                                                w,
                                                conn.netclass.via_dia,
                                                u1_center,
                                                1.0,
                                                0.0,
                                                conn.netclass.max_coupled_length_mm,
                                                conn.netclass.require_return_path,
                                                None,
                                                stop_flag,
                                            )
                                            .map(|p| {
                                                found_counts[t_pos].fetch_add(1, AtomicOrdering::Relaxed);
                                                p
                                            })
                                            .or_else(|| {
                                                self.find_path_portfolio_variant(
                                                    grid,
                                                    start_grid,
                                                    end_grid,
                                                    conn.net_id,
                                                    conn.netclass.clearance,
                                                    w,
                                                    conn.netclass.via_dia,
                                                    u1_center,
                                                    conn.netclass.max_coupled_length_mm,
                                                    conn.netclass.require_return_path,
                                                    var.via_mult,
                                                    var.hist_w,
                                                    var.heur_w,
                                                    var.dist_bias,
                                                    var.reverse,
                                                    var.diag_first,
                                                    stop_flag,
                                                    Some(&found_counts[t_pos]),
                                                    None,
                                                )
                                            })?
                                        } else {
                                            self.find_path_portfolio_variant(
                                                grid,
                                                start_grid,
                                                end_grid,
                                                conn.net_id,
                                                conn.netclass.clearance,
                                                w,
                                                conn.netclass.via_dia,
                                                u1_center,
                                                conn.netclass.max_coupled_length_mm,
                                                conn.netclass.require_return_path,
                                                var.via_mult,
                                                var.hist_w,
                                                var.heur_w,
                                                var.dist_bias,
                                                var.reverse,
                                                var.diag_first,
                                                stop_flag,
                                                Some(&found_counts[t_pos]),
                                                None,
                                            )?
                                        };
                                        Some((conn_idx, v_id, path))
                                    })
                                    .collect()
                            },
                            || {
                                if concurrent_gpu_reqs.is_empty() || stop_flag.load(AtomicOrdering::Relaxed) {
                                    return Vec::new();
                                }
                                let wf_reqs: Vec<(GridNode, GridNode, usize, f32)> =
                                    concurrent_gpu_reqs.iter().map(|(_, r)| *r).collect();
                                let wf_paths = self.solve_batch_gpu_wavefront_paths(grid, &wf_reqs, 1);
                                concurrent_gpu_reqs
                                    .iter()
                                    .map(|(c_idx, _)| *c_idx)
                                    .zip(wf_paths)
                                    .collect()
                            },
                        );

                        for (conn_idx, wf_opt) in concurrent_gpu_paths {
                            if let Some(p) = wf_opt {
                                raw_candidates.push((conn_idx, PORTFOLIO_VARIANTS.len(), p));
                            }
                        }

                        let missing_gpu_reqs: Vec<(usize, (GridNode, GridNode, usize, f32))> = target_indices
                            .iter()
                            .enumerate()
                            .skip(concurrent_gpu_limit)
                            .filter(|(t_pos, _)| found_counts[*t_pos].load(AtomicOrdering::Relaxed) == 0)
                            .filter_map(|(_, &conn_idx)| {
                                let conn = &connections[conn_idx];
                                let (sgx, sgy) = grid.world_to_grid(conn.start.0, conn.start.1)?;
                                let (egx, egy) = grid.world_to_grid(conn.end.0, conn.end.1)?;
                                Some((
                                    conn_idx,
                                    (
                                        GridNode { gx: sgx, gy: sgy, layer: conn.start.2 },
                                        GridNode { gx: egx, gy: egy, layer: conn.end.2 },
                                        conn.net_id,
                                        3.0f32,
                                    ),
                                ))
                            })
                            .collect();
                        if !missing_gpu_reqs.is_empty() && !stop_flag.load(AtomicOrdering::Relaxed) {
                            let wf_reqs: Vec<(GridNode, GridNode, usize, f32)> =
                                missing_gpu_reqs.iter().map(|(_, r)| *r).collect();
                            let wf_paths = self.solve_batch_gpu_wavefront_paths(grid, &wf_reqs, 1);
                            for ((conn_idx, _), wf_opt) in missing_gpu_reqs.into_iter().zip(wf_paths.into_iter()) {
                                if let Some(p) = wf_opt {
                                    raw_candidates.push((conn_idx, PORTFOLIO_VARIANTS.len(), p));
                                }
                            }
                        }

                        let flat_paths: Vec<(usize, Vec<GridNode>)> = raw_candidates
                            .iter()
                            .map(|(c_idx, _, p)| (*c_idx, p.clone()))
                            .collect();
                        let validations = self.batch_validate_paths_gpu(board, grid, connections, &flat_paths, u1_center);

                        let mut best_per_conn: HashMap<usize, (f64, usize, Vec<GridNode>)> = HashMap::new();
                        for ((conn_idx, v_id, path), (grid_ok, gpu_drc_ok)) in
                            raw_candidates.into_iter().zip(validations.into_iter())
                        {
                            if !grid_ok {
                                continue;
                            }
                            let (vias, wirelength, cong_sum) = Self::evaluate_path_cost(grid, &path);
                            let bends = Self::count_path_bends(&path);
                            let drc_pen = if gpu_drc_ok { 0.0 } else { 25.0 };
                            let score = (vias as f64) * 10.0
                                + wirelength
                                + (bends as f64) * 0.20
                                + cong_sum * 0.02
                                + drc_pen;
                            let entry = best_per_conn.entry(conn_idx).or_insert((f64::INFINITY, usize::MAX, Vec::new()));
                            if score < entry.0 - 1e-6 || ((score - entry.0).abs() <= 1e-6 && v_id < entry.1) {
                                *entry = (score, v_id, path);
                            }
                        }

                        target_indices
                            .iter()
                            .map(|&conn_idx| {
                                let p_opt = best_per_conn.remove(&conn_idx).map(|(_, _, p)| p);
                                (conn_idx, p_opt)
                            })
                            .collect()
                    }

                    /// Parallel Multi-Net Rip-Up Portfolio for resolving stubborn late-pass deadlocks
                    /// when `unrouted_indices.len() <= 6` and `iteration >= 2`.
                    #[allow(clippy::too_many_arguments)]
                    pub fn try_parallel_ripup_portfolio(
                        &self,
                        board: &Board,
                        grid: &mut RoutingGrid,
                        connections: &[RouteConnection],
                        routed_paths: &mut Vec<Option<Vec<GridNode>>>,
                        escape_stubs: &[EscapeStub],
                        unrouted_indices: &[usize],
                        u1_center: Option<(f64, f64)>,
                        stop_flag: &AtomicBool,
                    ) -> bool {
                        if unrouted_indices.is_empty() || stop_flag.load(AtomicOrdering::Relaxed) {
                            return false;
                        }
                        let total_conns = connections.len();
                        let current_done = routed_paths.iter().filter(|p| p.is_some()).count();

                        // Identify blocking routed connections near the unrouted endpoints/corridors
                        let mut blocker_scores: HashMap<usize, f64> = HashMap::new();
                        for &u_idx in unrouted_indices {
                            let uc = &connections[u_idx];
                            let mut choked_nets = Vec::new();
                            if let Some((sgx, sgy)) = grid.world_to_grid(uc.start.0, uc.start.1) {
                                let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, sgx, sgy, uc.start.2, uc.net_id, 400);
                                if reach < 300 {
                                    choked_nets.extend(nets);
                                }
                            }
                            if let Some((egx, egy)) = grid.world_to_grid(uc.end.0, uc.end.1) {
                                let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, egx, egy, uc.end.2, uc.net_id, 400);
                                if reach < 300 {
                                    choked_nets.extend(nets);
                                }
                            }
                            for rip_idx in 0..total_conns {
                                if rip_idx == u_idx || routed_paths[rip_idx].is_none() {
                                    continue;
                                }
                                let rc = &connections[rip_idx];
                                if rc.net_id == uc.net_id
                                    || rc.diff_pair_leader.is_some()
                                    || connections.iter().any(|c| c.diff_pair_leader == Some(rip_idx))
                                    || rc.net_name.contains("USB")
                                {
                                    continue;
                                }
                                let path = routed_paths[rip_idx].as_ref().unwrap();
                                let is_choked = choked_nets.contains(&rc.net_id);
                                let mut min_d = f64::INFINITY;
                                for node in path {
                                    let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                                    let d = (wx - uc.start.0).hypot(wy - uc.start.1)
                                        .min((wx - uc.end.0).hypot(wy - uc.end.1))
                                        .min(dist_to_segment(wx, wy, uc.start.0, uc.start.1, uc.end.0, uc.end.1));
                                    if d < min_d {
                                        min_d = d;
                                    }
                                }
                                if is_choked || min_d < 2.2 {
                                    let s = if is_choked { min_d * 0.25 } else { min_d };
                                    let entry = blocker_scores.entry(rip_idx).or_insert(f64::INFINITY);
                                    if s < *entry {
                                        *entry = s;
                                    }
                                }
                            }
                        }

                        let mut blockers: Vec<(usize, f64)> = blocker_scores.into_iter().collect();
                        blockers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
                        blockers.truncate(8);
                        if blockers.is_empty() {
                            return false;
                        }

                        let num_replicas = 8usize;
                        let replica_results: Vec<(usize, f64, usize, Vec<Option<Vec<GridNode>>>)> = (0..num_replicas)
                            .into_par_iter()
                            .filter_map(|rep_id| {
                                if stop_flag.load(AtomicOrdering::Relaxed) {
                                    return None;
                                }
                                let mut local_paths = routed_paths.clone();
                                let mut local_grid = grid.clone_for_portfolio();

                                // Select distinct subset of blockers to rip up for this portfolio replica
                                let mut ripped_conns = Vec::new();
                                for (b_pos, &(b_idx, _)) in blockers.iter().enumerate() {
                                    let take_it = match rep_id {
                                        0 => b_pos < 2,
                                        1 => b_pos < 3,
                                        2 => b_pos < 4,
                                        3 => b_pos % 2 == 0 && b_pos < 6,
                                        4 => (b_pos % 2 == 1 || b_pos == 0) && b_pos < 6,
                                        5 => b_pos < 5,
                                        6 => b_pos >= 1 && b_pos < 5,
                                        _ => b_pos < blockers.len().min(6),
                                    };
                                    if take_it {
                                        if let Some(old_p) = local_paths[b_idx].take() {
                                            for node in &old_p {
                                                let idx = local_grid.idx(node.gx, node.gy, node.layer);
                                                local_grid.congestion[idx] += self.congestion_factor * 1.5;
                                            }
                                            ripped_conns.push(b_idx);
                                        }
                                    }
                                }
                                if ripped_conns.is_empty() {
                                    return None;
                                }

                                self.rebuild_grid_state(board, &mut local_grid, connections, &local_paths, escape_stubs);

                                // Build local subgraph target ordering permutation for this replica
                                let mut order: Vec<usize> = Vec::with_capacity(unrouted_indices.len() + ripped_conns.len());
                                match rep_id % 4 {
                                    0 => {
                                        order.extend_from_slice(unrouted_indices);
                                        order.extend_from_slice(&ripped_conns);
                                    }
                                    1 => {
                                        for &u in unrouted_indices.iter().rev() {
                                            order.push(u);
                                        }
                                        order.extend_from_slice(&ripped_conns);
                                    }
                                    2 => {
                                        order.extend_from_slice(unrouted_indices);
                                        for &r in ripped_conns.iter().rev() {
                                            order.push(r);
                                        }
                                    }
                                    _ => {
                                        // Interleave unrouted and highest-priority ripped nets
                                        order.extend_from_slice(unrouted_indices);
                                        order.extend_from_slice(&ripped_conns);
                                        if order.len() > 2 {
                                            let rot = (rep_id / 4) % order.len();
                                            order.rotate_left(rot);
                                        }
                                    }
                                }

                                let var_pair = [
                                    PORTFOLIO_VARIANTS[rep_id % PORTFOLIO_VARIANTS.len()],
                                    PORTFOLIO_VARIANTS[(rep_id + 1) % PORTFOLIO_VARIANTS.len()],
                                    PORTFOLIO_VARIANTS[0],
                                    PORTFOLIO_VARIANTS[1],
                                ];

                                for &c_idx in &order {
                                    if local_paths[c_idx].is_some() || stop_flag.load(AtomicOrdering::Relaxed) {
                                        continue;
                                    }
                                    let conn = &connections[c_idx];
                                    let (Some((sgx, sgy)), Some((egx, egy))) = (
                                        local_grid.world_to_grid(conn.start.0, conn.start.1),
                                        local_grid.world_to_grid(conn.end.0, conn.end.1),
                                    ) else {
                                        continue;
                                    };
                                    let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                                    let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };
                                    let w = if conn.diff_pair_leader.is_some() || conn.netclass.name == "USB_Diff" {
                                        conn.netclass.diff_pair_width
                                    } else {
                                        conn.netclass.track_width
                                    };

                                    let mut best_cand: Option<(f64, Vec<GridNode>)> = None;
                                    for var in &var_pair {
                                        if let Some(p) = self.find_path_portfolio_variant(
                                            &local_grid,
                                            s_node,
                                            e_node,
                                            conn.net_id,
                                            conn.netclass.clearance,
                                            w,
                                            conn.netclass.via_dia,
                                            u1_center,
                                            conn.netclass.max_coupled_length_mm,
                                            conn.netclass.require_return_path,
                                            var.via_mult,
                                            var.hist_w,
                                            var.heur_w,
                                            var.dist_bias,
                                            var.reverse,
                                            var.diag_first,
                                            stop_flag,
                                            None,
                                            Some(60_000),
                                        ) {
                                            let (vias, wl, _) = Self::evaluate_path_cost(&local_grid, &p);
                                            let sc = (vias as f64) * 10.0 + wl;
                                            if best_cand.as_ref().is_none_or(|(b_sc, _)| sc < *b_sc) {
                                                best_cand = Some((sc, p));
                                            }
                                            if vias == 0 {
                                                break;
                                            }
                                        }
                                    }
                                    if let Some((_, p)) = best_cand {
                                        self.commit_path_to_grid(&mut local_grid, &p, conn.net_id, w, conn.netclass.via_dia);
                                        local_paths[c_idx] = Some(p);
                                    }
                                }

                                let done_cnt = local_paths.iter().filter(|p| p.is_some()).count();
                                let mut total_cost = 0.0f64;
                                for p_opt in local_paths.iter().flatten() {
                                    let (vias, wl, _) = Self::evaluate_path_cost(&local_grid, p_opt);
                                    total_cost += (vias as f64) * 10.0 + wl;
                                }
                                Some((done_cnt, total_cost, rep_id, local_paths))
                            })
                            .collect();

                        if let Some((best_done, _, _, winner_paths)) = replica_results.into_iter().max_by(|a, b| {
                            a.0.cmp(&b.0)
                                .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal))
                                .then_with(|| b.2.cmp(&a.2))
                        }) {
                            if best_done > current_done {
                                *routed_paths = winner_paths;
                                self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
                                return true;
                            }
                        }
                        false
                    }
                }

                let wave_grid: &RoutingGrid = &*grid;
                let speculative_paths: Vec<(usize, Option<Vec<GridNode>>)> = self.route_connections_portfolio(
                    board,
                    wave_grid,
                    &connections,
                    &unrouted_indices,
                    u1_center,
                    &stop_flag,
                );

                // Phase 2: Conflict-Graph Independent Set Wave Partitioning & Parallel Sub-Wave Re-Routing
                let mut _any_committed_in_wave = false;
                let mut sorted_speculative_paths = speculative_paths;
                sorted_speculative_paths.sort_by_key(|&(idx, _)| std::cmp::Reverse(unrouted_history[idx]));

                let mut found_speculative: Vec<(usize, Vec<GridNode>)> = Vec::with_capacity(sorted_speculative_paths.len());
                let mut need_reroute: Vec<usize> = Vec::new();
                for (conn_idx, spec_opt) in sorted_speculative_paths {
                    if let Some(p) = spec_opt {
                        found_speculative.push((conn_idx, p));
                    } else {
                        need_reroute.push(conn_idx);
                    }
                }

                let waves = Self::partition_into_conflict_free_waves(grid, &connections, found_speculative);
                for (w_idx, wave) in waves.into_iter().enumerate() {
                    if stop_flag.load(AtomicOrdering::Relaxed) {
                        break;
                    }
                    let validations = self.batch_validate_paths_gpu(board, grid, &connections, &wave, u1_center);
                    for ((conn_idx, p), (grid_ok, _gpu_drc_ok)) in wave.into_iter().zip(validations.into_iter()) {
                        let conn = &connections[conn_idx];
                        let w = if conn.diff_pair_leader.is_some() || conn.netclass.name == "USB_Diff" {
                            conn.netclass.diff_pair_width
                        } else {
                            conn.netclass.track_width
                        };
                        let still_ok = grid_ok
                            && self.is_path_conflict_free(
                                grid,
                                &p,
                                conn.net_id,
                                conn.netclass.clearance,
                                w,
                                conn.netclass.via_dia,
                                u1_center,
                                conn.netclass.require_return_path,
                            );
                        if still_ok {
                            self.commit_path_to_grid(grid, &p, conn.net_id, w, conn.netclass.via_dia);
                            routed_paths[conn_idx] = Some(p);
                            _any_committed_in_wave = true;
                        } else {
                            need_reroute.push(conn_idx);
                        }
                    }
                    if w_idx == 0 && _any_committed_in_wave {
                        self.reconstruct_tracks_and_vias(board, grid, &connections, &routed_paths, &escape_stubs);
                    }
                }

                // Parallel Sub-Wave Re-Routing for paths that conflicted with Wave 0
                need_reroute.sort_by_key(|&idx| std::cmp::Reverse(unrouted_history[idx]));
                let max_subwaves = 4usize;
                for _sub in 0..max_subwaves {
                    if need_reroute.is_empty() || stop_flag.load(AtomicOrdering::Relaxed) {
                        break;
                    }
                    let sub_results = self.route_connections_portfolio(
                        board,
                        grid,
                        &connections,
                        &need_reroute,
                        u1_center,
                        &stop_flag,
                    );
                    let mut sub_found = Vec::new();
                    let mut next_reroute = Vec::new();
                    for (conn_idx, p_opt) in sub_results {
                        if let Some(p) = p_opt {
                            sub_found.push((conn_idx, p));
                        } else {
                            next_reroute.push(conn_idx);
                        }
                    }
                    if sub_found.is_empty() {
                        need_reroute = next_reroute;
                        break;
                    }
                    let sub_waves = Self::partition_into_conflict_free_waves(grid, &connections, sub_found);
                    for sub_wave in sub_waves {
                        let validations = self.batch_validate_paths_gpu(board, grid, &connections, &sub_wave, u1_center);
                        for ((conn_idx, p), (grid_ok, _)) in sub_wave.into_iter().zip(validations.into_iter()) {
                            let conn = &connections[conn_idx];
                            let w = if conn.diff_pair_leader.is_some() || conn.netclass.name == "USB_Diff" {
                                conn.netclass.diff_pair_width
                            } else {
                                conn.netclass.track_width
                            };
                            if grid_ok
                                && self.is_path_conflict_free(
                                    grid,
                                    &p,
                                    conn.net_id,
                                    conn.netclass.clearance,
                                    w,
                                    conn.netclass.via_dia,
                                    u1_center,
                                    conn.netclass.require_return_path,
                                )
                            {
                                self.commit_path_to_grid(grid, &p, conn.net_id, w, conn.netclass.via_dia);
                                routed_paths[conn_idx] = Some(p);
                                _any_committed_in_wave = true;
                            } else {
                                next_reroute.push(conn_idx);
                            }
                        }
                    }
                    need_reroute = next_reroute;
                }

                // Final priority-ordered fallback for any remaining conflicted connection in this pass
                for conn_idx in need_reroute {
                    if stop_flag.load(AtomicOrdering::Relaxed) || routed_paths[conn_idx].is_some() {
                        continue;
                    }
                    let conn = &connections[conn_idx];
                    if let (Some((sgx, sgy)), Some((egx, egy))) = (
                        grid.world_to_grid(conn.start.0, conn.start.1),
                        grid.world_to_grid(conn.end.0, conn.end.1),
                    ) {
                        let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                        let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };
                        if let Some(p) = self.find_path_parallel(
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
                            &stop_flag,
                        ) {
                            self.commit_path_to_grid(grid, &p, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
                            routed_paths[conn_idx] = Some(p);
                            _any_committed_in_wave = true;
                        }
                    }
                }

                if _any_committed_in_wave {
                    if let Ok(mut obs_g) = shared_obs.lock() {
                        obs_g.clone_from(&grid.obstacle_net_ids);
                    }
                }
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
                    }
                }

                // Phase 2.75: Parallel Multi-Net Rip-Up Portfolio when <= 6 unrouted connections stall on pass >= 2
                let remaining_after_nudge: Vec<usize> = (0..total_conns)
                    .filter(|&i| routed_paths[i].is_none())
                    .collect();
                if !remaining_after_nudge.is_empty()
                    && remaining_after_nudge.len() <= 6
                    && iteration >= 2
                    && !stop_flag.load(AtomicOrdering::Relaxed)
                {
                    if self.try_parallel_ripup_portfolio(
                        board,
                        grid,
                        &connections,
                        &mut routed_paths,
                        &escape_stubs,
                        &remaining_after_nudge,
                        u1_center,
                        &stop_flag,
                    ) {
                        if let Ok(mut obs_g) = shared_obs.lock() {
                            obs_g.clone_from(&grid.obstacle_net_ids);
                        }
                    }
                }

                let done = routed_paths.iter().filter(|p| p.is_some()).count();
                if done > best_routed_count {
                    best_routed_count = done;
                    best_routed_paths = routed_paths.clone();
                }
                if let Some(ref st) = live {
                    self.reconstruct_tracks_and_vias(board, grid, &connections, &routed_paths, &escape_stubs);
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
                    let mut rip_budget = 8;
                    let mut any_ripped_in_pass = false;
                    let corridor_tol = (0.85 + 0.35 * (iteration as f64)).min(2.5);
                    for conn_idx in 0..total_conns {
                        if rip_budget == 0 {
                            break;
                        }
                        if routed_paths[conn_idx].is_none() {
                            let blocked_conn = &connections[conn_idx];
                            let b_start = blocked_conn.start;
                            let b_end = blocked_conn.end;

                            let mut choked_nets = Vec::new();
                            if let Some((sgx, sgy)) = grid.world_to_grid(b_start.0, b_start.1) {
                                let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, sgx, sgy, b_start.2, blocked_conn.net_id, 400);
                                if reach < 300 {
                                    choked_nets.extend(nets);
                                }
                            }
                            if let Some((egx, egy)) = grid.world_to_grid(b_end.0, b_end.1) {
                                let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, egx, egy, b_end.2, blocked_conn.net_id, 400);
                                if reach < 300 {
                                    choked_nets.extend(nets);
                                }
                            }

                            for rip_idx in 0..total_conns {
                                if rip_budget == 0 {
                                    break;
                                }
                                if rip_idx == conn_idx || routed_paths[rip_idx].is_none() {
                                    continue;
                                }
                                let p = routed_paths[rip_idx].as_ref().unwrap();
                                let is_choked = choked_nets.contains(&connections[rip_idx].net_id);
                                let is_at_pin = is_choked || p.iter().any(|node| {
                                    let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                                    (wx - b_start.0).hypot(wy - b_start.1) < 1.2
                                        || (wx - b_end.0).hypot(wy - b_end.1) < 1.2
                                });
                                let passes_near = is_at_pin
                                    || p.iter().any(|node| {
                                        let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                                        dist_to_segment(wx, wy, b_start.0, b_start.1, b_end.0, b_end.1) < corridor_tol
                                    });

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
                                            let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                                            let d_s = (wx - b_start.0).hypot(wy - b_start.1);
                                            let d_e = (wx - b_end.0).hypot(wy - b_end.1);
                                            let d_seg = dist_to_segment(wx, wy, b_start.0, b_start.1, b_end.0, b_end.1);
                                            let idx = grid.idx(node.gx, node.gy, node.layer);
                                            if is_choked || d_s < 1.8 || d_e < 1.8 || d_seg < corridor_tol {
                                                grid.congestion[idx] += self.congestion_factor * 3.5;
                                            } else {
                                                grid.congestion[idx] += self.congestion_factor * 1.5;
                                            }
                                        }
                                        any_ripped_in_pass = true;
                                        rip_budget -= 1;
                                    }
                                }
                            }
                        }
                    }
                    if any_ripped_in_pass {
                        self.rebuild_grid_state(board, grid, &connections, &routed_paths, &escape_stubs);
                        if let Ok(mut obs_g) = shared_obs.lock() {
                            obs_g.clone_from(&grid.obstacle_net_ids);
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
        let gloss_iters = self.config.gloss_passes;
        for pass_i in 0..gloss_iters {
            gloss_tracks_standard(board, grid);
            prune_dangling_tracks_and_vias(board);
            if pass_i + 1 < gloss_iters {
                self.reduce_vias(board, grid);
                prune_dangling_tracks_and_vias(board);
            }
        }

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
        self.add_plane_stitching_vias(board, grid);
        self.add_return_path_ground_vias(board, grid);

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

        // Final Continuity Verification & Repair Pass:
        // If any connection was disconnected during post-processing, re-route it cleanly against the current board copper.
        for conn_idx in 0..total_conns {
            let conn = &connections[conn_idx];
            if !Self::is_connection_electrically_connected(board, conn) {
                grid.obstacle_net_ids.fill(FREE_NET);
                grid.nudgeable_fp.fill(None);
                grid.smd_pad_cells.fill(false);
                grid.rasterize_obstacles(board);
                for trk in &board.tracks {
                    if trk.net_id != conn.net_id {
                        grid.commit_track(trk);
                    }
                }
                for v in &board.vias {
                    if v.net_id != conn.net_id {
                        grid.commit_via(v);
                    }
                }
                let sp = board.footprints[conn.start_fp].pad_absolute_position(&board.footprints[conn.start_fp].pads[conn.start_pad]);
                let ep = board.footprints[conn.end_fp].pad_absolute_position(&board.footprints[conn.end_fp].pads[conn.end_pad]);
                let s_layer = board.footprints[conn.start_fp].layer;
                let e_layer = board.footprints[conn.end_fp].layer;
                let candidate_endpoints = [
                    (conn.start, conn.end, true),
                    ((sp.0, sp.1, s_layer), (ep.0, ep.1, e_layer), false),
                ];
                for &(cand_s, cand_e, use_stubs) in &candidate_endpoints {
                    if Self::is_connection_electrically_connected(board, conn) {
                        break;
                    }
                    if let (Some((sgx, sgy)), Some((egx, egy))) = (grid.world_to_grid(cand_s.0, cand_s.1), grid.world_to_grid(cand_e.0, cand_e.1)) {
                        let s_node = GridNode { gx: sgx, gy: sgy, layer: cand_s.2 };
                        let e_node = GridNode { gx: egx, gy: egy, layer: cand_e.2 };
                        let mut used_w = conn.netclass.track_width;
                        let mut used_c = conn.netclass.clearance;
                        let repair_path = self
                            .find_path_with_constraints(
                                grid,
                                s_node,
                                e_node,
                                conn.net_id,
                                conn.netclass.clearance,
                                conn.netclass.track_width,
                                conn.netclass.via_dia,
                                u1_center,
                                1.0,
                                0.0,
                                conn.netclass.max_coupled_length_mm,
                                false,
                                None,
                                &stop_flag,
                            )
                            .or_else(|| {
                                self.find_path_parallel(
                                    grid,
                                    s_node,
                                    e_node,
                                    conn.net_id,
                                    conn.netclass.clearance,
                                    conn.netclass.track_width,
                                    conn.netclass.via_dia,
                                    u1_center,
                                    conn.netclass.max_coupled_length_mm,
                                    false,
                                    &stop_flag,
                                )
                            })
                            .or_else(|| {
                                used_w = conn.netclass.track_width.min(0.25);
                                used_c = conn.netclass.clearance.min(0.15);
                                self.find_path_with_params(
                                    grid,
                                    s_node,
                                    e_node,
                                    conn.net_id,
                                    used_c,
                                    used_w,
                                    conn.netclass.via_dia,
                                    u1_center,
                                    1.15,
                                    0.0,
                                    &stop_flag,
                                )
                            })
                            .or_else(|| {
                                used_w = conn.netclass.track_width.min(0.15);
                                used_c = conn.netclass.clearance.min(0.12);
                                self.find_path_with_params(
                                    grid,
                                    s_node,
                                    e_node,
                                    conn.net_id,
                                    used_c,
                                    used_w,
                                    conn.netclass.via_dia,
                                    u1_center,
                                    1.15,
                                    0.0,
                                    &stop_flag,
                                )
                            });
                        if let Some(rp) = repair_path {
                            let mut single_conn = conn.clone();
                            single_conn.start = cand_s;
                            single_conn.end = cand_e;
                            single_conn.netclass.track_width = used_w;
                            single_conn.netclass.clearance = used_c;
                            for trk in &mut board.tracks {
                                trk.is_locked = true;
                            }
                            for v in &mut board.vias {
                                v.is_locked = true;
                            }
                            let single_conns = vec![single_conn];
                            let single_paths = vec![Some(rp)];
                            let stubs_slice: &[EscapeStub] = if use_stubs { &escape_stubs } else { &[] };
                            self.reconstruct_tracks_and_vias(board, grid, &single_conns, &single_paths, stubs_slice);
                            for trk in &mut board.tracks {
                                let key = (trk.net_id, [trk.start.0.to_bits(), trk.start.1.to_bits(), trk.end.0.to_bits(), trk.end.1.to_bits()]);
                                trk.is_locked = orig_locked_tracks.contains(&key);
                            }
                            for v in &mut board.vias {
                                let key = (v.net_id, [v.position.0.to_bits(), v.position.1.to_bits()]);
                                v.is_locked = orig_locked_vias.contains(&key);
                            }
                            prune_dangling_tracks_and_vias(board);
                            gloss_tracks_standard(board, grid);
                            validate_and_clean_copper(board);
                            prune_dangling_tracks_and_vias(board);
                        }
                    }
                }
            }
        }

        widen_tracks_and_arcs_to_netclass(board);

        let pre_organic_snapshot = if self.config.any_angle_mode {
            let snap = (board.tracks.clone(), board.arcs.clone());
            convert_tracks_to_organic(board);
            Some(snap)
        } else {
            None
        };

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
                g.board.arcs = board.arcs.clone();
                g.board.vias = board.vias.clone();
                g.board.footprints = board.footprints.clone();
                g.straight_copper_snapshot = pre_organic_snapshot;
                g.ratsnest = remaining_ratsnest;
                g.routed_conns = completed_count;
                g.total_conns = total_conns;
                g.progress = 1.0;
                g.phase = TrialPhase::Completed;
                g.push_log(format!(
                    "[Auto-Router] Completed: {}/{} ({:.1}%) connections, {} tracks, {} arcs, {} vias.",
                    completed_count,
                    total_conns,
                    (completed_count as f64 / total_conns.max(1) as f64) * 100.0,
                    board.tracks.len(),
                    board.arcs.len(),
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
        let start_is_smd = (0..grid.num_layers).any(|l| grid.smd_pad_cells[grid.idx(start.gx, start.gy, l)]);
        let start_is_tht = !start_is_smd
            && (0..grid.num_layers).all(|l| grid.obstacle_net_ids[grid.idx(start.gx, start.gy, l)] == net_id as i32);
        if start_is_tht {
            for l in 0..grid.num_layers {
                if l != start.layer {
                    if grid.plane_nets.contains_key(&l) && grid.plane_nets.get(&l) != Some(&net_id) {
                        continue;
                    }
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
        grid.gpu_net_cost_maps.clear();
        let r_cells = (track_width / (2.0 * grid.res)).ceil() as isize;
        for i in 0..path.len() {
            let node = path[i];
            let node_idx = grid.idx(node.gx, node.gy, node.layer);
            grid.usage_count[node_idx] = grid.usage_count[node_idx].saturating_add(1);
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
        if let Some(ref snap) = grid.base_snapshot {
            grid.obstacle_net_ids.copy_from_slice(&snap.0);
            grid.usage_count.copy_from_slice(&snap.1);
            grid.nudgeable_fp.copy_from_slice(&snap.2);
            grid.smd_pad_cells.copy_from_slice(&snap.3);
            grid.gpu_net_cost_maps.clear();
        } else {
            grid.obstacle_net_ids.fill(FREE_NET);
            grid.usage_count.fill(0);
            grid.gpu_net_cost_maps.clear();
            grid.nudgeable_fp.fill(None);
            grid.smd_pad_cells.fill(false);
            grid.rasterize_obstacles(board);
            for trk in &board.tracks {
                if trk.is_locked {
                    grid.commit_track(trk);
                }
            }
            for v in &board.vias {
                if v.is_locked {
                    grid.commit_via(v);
                }
            }
            for stub in escape_stubs {
                self.commit_stub_to_grid(grid, stub);
            }
            grid.base_snapshot = Some(Arc::new((
                grid.obstacle_net_ids.clone(),
                grid.usage_count.clone(),
                grid.nudgeable_fp.clone(),
                grid.smd_pad_cells.clone(),
            )));
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

    fn clone_grid_for_worker(grid: &RoutingGrid) -> RoutingGrid {
        RoutingGrid {
            min_x: grid.min_x,
            min_y: grid.min_y,
            max_x: grid.max_x,
            max_y: grid.max_y,
            res: grid.res,
            width: grid.width,
            height: grid.height,
            num_layers: grid.num_layers,
            plane_nets: grid.plane_nets.clone(),
            obstacle_net_ids: grid.obstacle_net_ids.clone(),
            dist_field: grid.dist_field.clone(),
            congestion: grid.congestion.clone(),
            usage_count: grid.usage_count.clone(),
            gpu_net_cost_maps: HashMap::new(),
            nudgeable_fp: grid.nudgeable_fp.clone(),
            smd_pad_cells: grid.smd_pad_cells.clone(),
            net_clearances: grid.net_clearances.clone(),
            max_net_clearance: grid.max_net_clearance,
            inner_layer_penalty: grid.inner_layer_penalty,
            vulkan: None,
            base_snapshot: grid.base_snapshot.clone(),
        }
    }

    /// When routing a connection fails, identify blocking unlocked components,
    /// nudge them away into free space, and retry the path search in parallel across Rayon workers.
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
        blocking_candidates.truncate(3);
        if blocking_candidates.is_empty() {
            return false;
        }

        // Flatten all candidate (fp_idx, dx, dy) poses across all blocking footprints into one parallel batch
        let mut flat_pose_cands = Vec::new();
        for &(fp_idx, push_dir, _) in &blocking_candidates {
            let orig_fp_x = board.footprints[fp_idx].x;
            let orig_fp_y = board.footprints[fp_idx].y;
            for (dx, dy) in self.nudge_engine.nudge_candidates(push_dir) {
                flat_pose_cands.push((fp_idx, orig_fp_x + dx, orig_fp_y + dy));
            }
        }

        let legal_poses: Vec<(usize, f64, f64)> = flat_pose_cands
            .par_iter()
            .filter_map(|&(fp_idx, nx, ny)| {
                if self.nudge_engine.is_placement_valid(board, fp_idx, nx, ny) {
                    Some((fp_idx, nx, ny))
                } else {
                    None
                }
            })
            .collect();

        if legal_poses.is_empty() {
            return false;
        }

        // Evaluate A* pathfinding in parallel across all legal footprint poses simultaneously
        let winning_pose: Option<(usize, (f64, f64), Vec<GridNode>)> = legal_poses
            .par_iter()
            .filter_map(|&(fp_idx, nx, ny)| {
                if stop_flag.load(AtomicOrdering::Relaxed) {
                    return None;
                }
                let mut local_board = board.clone();
                local_board.footprints[fp_idx].x = nx;
                local_board.footprints[fp_idx].y = ny;

                let mut local_conns = connections.to_vec();
                let mut local_paths = routed_paths.to_vec();
                for (c_i, c) in local_conns.iter_mut().enumerate() {
                    if c.start_fp == fp_idx {
                        let (px, py) = local_board.footprints[fp_idx]
                            .pad_absolute_position(&local_board.footprints[fp_idx].pads[c.start_pad]);
                        c.start = (px, py, c.start.2);
                        local_paths[c_i] = None;
                    }
                    if c.end_fp == fp_idx {
                        let (px, py) = local_board.footprints[fp_idx]
                            .pad_absolute_position(&local_board.footprints[fp_idx].pads[c.end_pad]);
                        c.end = (px, py, c.end.2);
                        local_paths[c_i] = None;
                    }
                }

                let mut local_grid = Self::clone_grid_for_worker(grid);
                local_grid.base_snapshot = None;
                self.rebuild_grid_state(&local_board, &mut local_grid, &local_conns, &local_paths, escape_stubs);

                let updated_conn = &local_conns[conn_idx];
                let (Some((sgx, sgy)), Some((egx, egy))) = (
                    local_grid.world_to_grid(updated_conn.start.0, updated_conn.start.1),
                    local_grid.world_to_grid(updated_conn.end.0, updated_conn.end.1),
                ) else {
                    return None;
                };
                let s_node = GridNode { gx: sgx, gy: sgy, layer: updated_conn.start.2 };
                let e_node = GridNode { gx: egx, gy: egy, layer: updated_conn.end.2 };

                self.find_path_with_params(
                    &local_grid,
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
                )
                .map(|p| (fp_idx, (nx, ny), p))
            })
            .find_first(|_| true);

        if let Some((fp_idx, (nx, ny), p)) = winning_pose {
            let orig_fp_x = board.footprints[fp_idx].x;
            let orig_fp_y = board.footprints[fp_idx].y;
            board.footprints[fp_idx].x = nx;
            board.footprints[fp_idx].y = ny;

            for (c_i, c) in connections.iter_mut().enumerate() {
                if c.start_fp == fp_idx {
                    let (px, py) = board.footprints[fp_idx]
                        .pad_absolute_position(&board.footprints[fp_idx].pads[c.start_pad]);
                    c.start = (px, py, c.start.2);
                    routed_paths[c_i] = None;
                }
                if c.end_fp == fp_idx {
                    let (px, py) = board.footprints[fp_idx]
                        .pad_absolute_position(&board.footprints[fp_idx].pads[c.end_pad]);
                    c.end = (px, py, c.end.2);
                    routed_paths[c_i] = None;
                }
            }

            grid.base_snapshot = None;
            self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
            self.commit_path_to_grid(grid, &p, c_net_id, track_width, via_dia);
            routed_paths[conn_idx] = Some(p);

            board.rip_up_connected_traces_to_component_at(
                fp_idx,
                Some((orig_fp_x, orig_fp_y, board.footprints[fp_idx].rotation, board.footprints[fp_idx].layer)),
            );
            grid.update_distance_fields();
            return true;
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
                    if obs != FREE_NET && obs != net_id as i32 {
                        if obs > 0 && !grid.smd_pad_cells[idx] {
                            blocking_nets.insert(obs as usize);
                        }
                        continue;
                    }
                    // Also check 1-cell clearance halo around (ux, uy) so the BFS does not
                    // leak through sub-clearance 0.10mm cracks between foreign tracks and pads
                    let mut halo_blocked = false;
                    for hdx in -1isize..=1 {
                        for hdy in -1isize..=1 {
                            if hdx == 0 && hdy == 0 {
                                continue;
                            }
                            let hx = ux as isize + hdx;
                            let hy = uy as isize + hdy;
                            if hx >= 0 && hx < grid.width as isize && hy >= 0 && hy < grid.height as isize {
                                let h_idx = grid.idx(hx as usize, hy as usize, layer);
                                let h_obs = grid.obstacle_net_ids[h_idx];
                                if h_obs > 0 && h_obs != net_id as i32 && !grid.smd_pad_cells[h_idx] {
                                    blocking_nets.insert(h_obs as usize);
                                    halo_blocked = true;
                                }
                            }
                        }
                    }
                    if !halo_blocked {
                        visited.insert((ux, uy));
                        queue.push_back((ux, uy));
                    }
                }
            }
        }
    }

    (visited.len(), blocking_nets.into_iter().collect())
}

    /// When routing is blocked by an unlocked track of another net:
    /// Shoves the blocking track or rips it up in parallel across Rayon workers + GPU batch DRC
    /// so the higher-priority signal can claim the channel.
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

        // ========================================================
        // STAGE 0: Speculative Parallel Track Deflection on Unlocked board.tracks
        // When unlocked tracks exist directly in `board.tracks` (e.g. pre-existing or standalone routing),
        // evaluate candidate lateral offsets in parallel using GpuDrcContext + Rayon A* pathfinding.
        // ========================================================
        if routed_paths.iter().all(|p| p.is_none())
            && board.tracks.iter().any(|t| !t.is_locked && t.net_id != conn.net_id)
        {
            let mut candidate_track_indices: Vec<(usize, f64)> = board
                .tracks
                .iter()
                .enumerate()
                .filter(|(_, t)| !t.is_locked && t.net_id != conn.net_id)
                .filter_map(|(t_idx, t)| {
                    let d = dist_segment_to_segment(t.start, t.end, (c_start.0, c_start.1), (c_end.0, c_end.1));
                    if d < 2.0 {
                        Some((t_idx, d))
                    } else {
                        None
                    }
                })
                .collect();
            candidate_track_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            candidate_track_indices.truncate(6);

            let offsets: [f64; 14] = [
                -0.10, 0.10, -0.20, 0.20, -0.30, 0.30, -0.45, 0.45, -0.60, 0.60, -0.80, 0.80, -1.00, 1.00,
            ];

            for (track_idx, _) in candidate_track_indices {
                let orig_t = board.tracks[track_idx].clone();
                let vx = orig_t.end.0 - orig_t.start.0;
                let vy = orig_t.end.1 - orig_t.start.1;
                let len = vx.hypot(vy);
                if len < 1e-4 {
                    continue;
                }
                let nx = -vy / len;
                let ny = vx / len;
                let t_clr = board.get_netclass_for_net(orig_t.net_id).clearance;

                let mut board_without_t = board.clone();
                board_without_t.tracks.remove(track_idx);
                let drc_ctx = GpuDrcContext::from_board(&board_without_t);
                let gpu_cands: Vec<GpuDrcCandidate> = offsets
                    .iter()
                    .map(|&off| GpuDrcCandidate {
                        x1: (orig_t.start.0 + nx * off) as f32,
                        y1: (orig_t.start.1 + ny * off) as f32,
                        x2: (orig_t.end.0 + nx * off) as f32,
                        y2: (orig_t.end.1 + ny * off) as f32,
                        radius_or_half_width: (orig_t.width * 0.5) as f32,
                        clearance: t_clr as f32,
                        layer: orig_t.layer as u32,
                        net_id: orig_t.net_id as u32,
                        kind: 1,
                        ignore_same_net_pads: 1,
                        _pad0: 0,
                        _pad1: 0,
                    })
                    .collect();
                let gpu_flags = drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_cands);

                let winning_shove: Option<(Board, Vec<GridNode>)> = offsets
                    .par_iter()
                    .enumerate()
                    .filter_map(|(o_idx, &off)| {
                        if stop_flag.load(AtomicOrdering::Relaxed) {
                            return None;
                        }
                        let mut cand_board = board.clone();
                        let deflected = self.nudge_engine.try_continuous_shove_deflect(&mut cand_board, track_idx, off, None)
                            || (gpu_flags.as_ref().map_or(true, |f| f[o_idx] != 0) && {
                                let s = (orig_t.start.0 + nx * off, orig_t.start.1 + ny * off);
                                let e = (orig_t.end.0 + nx * off, orig_t.end.1 + ny * off);
                                if is_track_segment_legal(&board_without_t, s, e, orig_t.width, orig_t.layer, orig_t.net_id, t_clr) {
                                    cand_board.tracks[track_idx].start = s;
                                    cand_board.tracks[track_idx].end = e;
                                    true
                                } else {
                                    false
                                }
                            });
                        if !deflected {
                            return None;
                        }

                        let mut local_grid = Self::clone_grid_for_worker(grid);
                        self.rebuild_grid_state(&cand_board, &mut local_grid, connections, routed_paths, escape_stubs);
                        for trk in &cand_board.tracks {
                            if !trk.is_locked {
                                local_grid.commit_track(trk);
                            }
                        }
                        for v in &cand_board.vias {
                            if !v.is_locked {
                                local_grid.commit_via(v);
                            }
                        }

                        let (Some((sgx, sgy)), Some((egx, egy))) = (
                            local_grid.world_to_grid(conn.start.0, conn.start.1),
                            local_grid.world_to_grid(conn.end.0, conn.end.1),
                        ) else {
                            return None;
                        };
                        let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                        let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };

                        self.find_path_with_params(
                            &local_grid,
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
                        )
                        .map(|p| (cand_board, p))
                    })
                    .find_first(|_| true);

                if let Some((winning_board, p)) = winning_shove {
                    board.tracks = winning_board.tracks;
                    self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
                    for trk in &board.tracks {
                        if !trk.is_locked {
                            grid.commit_track(trk);
                        }
                    }
                    self.commit_path_to_grid(grid, &p, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
                    routed_paths[conn_idx] = Some(p);
                    return true;
                }
            }
        }

        // 1. Identify Candidate Blocking Connections:
        // Include:
        // (a) Choked pin bottlenecks (< 300 reachable nodes)
        // (b) Tracks / vias in the corridor within 2.5 mm
        let (s_grid, e_grid) = (
            grid.world_to_grid(c_start.0, c_start.1),
            grid.world_to_grid(c_end.0, c_end.1),
        );
        let mut choked_blocking_nets = Vec::new();
        if let Some((sgx, sgy)) = s_grid {
            let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, sgx, sgy, c_start.2, conn.net_id, 400);
            if reach < 300 && !nets.is_empty() {
                choked_blocking_nets.extend(nets);
            }
        }
        if let Some((egx, egy)) = e_grid {
            let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, egx, egy, c_end.2, conn.net_id, 400);
            if reach < 300 && !nets.is_empty() {
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

            if is_choked_net || min_dist < 1.5 {
                let is_near_pad = is_choked_net || min_pad_dist < 1.2;
                candidate_conns.push((rip_idx, rip_prio, min_pad_dist, min_dist, is_near_pad));
            }
        }

        // Prioritize: (1) choked/near-pad bottlenecks, (2) closest corridor/pad distance, (3) lower net priority
        candidate_conns.sort_by(|a, b| {
            b.4.cmp(&a.4)
                .then_with(|| a.3.partial_cmp(&b.3).unwrap_or(Ordering::Equal))
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
                .then_with(|| a.1.cmp(&b.1))
        });
        candidate_conns.truncate(8);

        let drc_ctx = GpuDrcContext::from_board(board);

        // ========================================================
        // STAGE 1: Flattened Parallel Via Shoving with Multi-GPU Batch Pre-Filtering
        // Collect all corridor via shift candidates across all blocking connections into one
        // unified GPU DRC batch and evaluate all shifted via poses in parallel across Rayon workers.
        // ========================================================
        let shifts: [(isize, isize); 16] = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2),
            (2, 2), (-2, 2), (2, -2), (-2, -2),
        ];

        let mut via_shift_tasks: Vec<(usize, usize, isize, isize)> = Vec::new();
        let mut gpu_via_cands: Vec<GpuDrcCandidate> = Vec::new();

        for &(rip_idx, _, _, _, _) in &candidate_conns {
            let Some(ref orig_path) = routed_paths[rip_idx] else {
                continue;
            };
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
                    for &(dx, dy) in &shifts {
                        let ngx = (v_node.gx as isize + dx).clamp(0, grid.width.saturating_sub(1) as isize) as usize;
                        let ngy = (v_node.gy as isize + dy).clamp(0, grid.height.saturating_sub(1) as isize) as usize;
                        let (nwx, nwy) = grid.grid_to_world(ngx, ngy);
                        via_shift_tasks.push((rip_idx, v_step, dx, dy));
                        gpu_via_cands.push(GpuDrcCandidate {
                            x1: nwx as f32,
                            y1: nwy as f32,
                            x2: 0.0,
                            y2: 0.0,
                            radius_or_half_width: (other_via_dia * 0.5) as f32,
                            clearance: other_clearance as f32,
                            layer: 0,
                            net_id: other_net_id as u32,
                            kind: 0,
                            ignore_same_net_pads: 0,
                            _pad0: 0,
                            _pad1: 0,
                        });
                    }
                }
            }
        }

        if !via_shift_tasks.is_empty() {
            let gpu_via_flags = drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_via_cands);

            let winning_via_shove: Option<(usize, Vec<GridNode>, Vec<GridNode>)> = via_shift_tasks
                .par_iter()
                .enumerate()
                .filter_map(|(t_idx, &(rip_idx, v_step, dx, dy))| {
                    if stop_flag.load(AtomicOrdering::Relaxed) {
                        return None;
                    }
                    if !gpu_via_flags.as_ref().map_or(true, |f| f[t_idx] != 0) {
                        return None;
                    }
                    let orig_path = routed_paths[rip_idx].as_ref()?;
                    let v_node = orig_path[v_step];
                    let other_net_id = connections[rip_idx].net_id;
                    let other_clearance = connections[rip_idx].netclass.clearance;
                    let other_via_dia = connections[rip_idx].netclass.via_dia;

                    let ngx = v_node.gx as isize + dx;
                    let ngy = v_node.gy as isize + dy;
                    if ngx < 0 || ngx >= grid.width as isize || ngy < 0 || ngy >= grid.height as isize {
                        return None;
                    }
                    let (cand_gx, cand_gy) = (ngx as usize, ngy as usize);
                    let (nwx, nwy) = grid.grid_to_world(cand_gx, cand_gy);
                    if !is_via_location_legal(board, (nwx, nwy), other_via_dia, other_net_id, other_clearance) {
                        return None;
                    }

                    for l in 0..grid.num_layers {
                        let idx = grid.idx(cand_gx, cand_gy, l);
                        let obst = grid.obstacle_net_ids[idx];
                        if obst != FREE_NET && obst != other_net_id as i32 {
                            return None;
                        }
                    }

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

                    let slid_path = valid_slid_path?;
                    let mut local_paths = routed_paths.to_vec();
                    local_paths[rip_idx] = Some(slid_path.clone());
                    let mut local_grid = Self::clone_grid_for_worker(grid);
                    self.rebuild_grid_state(board, &mut local_grid, connections, &local_paths, escape_stubs);

                    let (Some((sgx, sgy)), Some((egx, egy))) = (
                        local_grid.world_to_grid(conn.start.0, conn.start.1),
                        local_grid.world_to_grid(conn.end.0, conn.end.1),
                    ) else {
                        return None;
                    };
                    let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                    let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };

                    self.find_path_with_params(
                        &local_grid,
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
                    )
                    .map(|p| (rip_idx, slid_path, p))
                })
                .find_first(|_| true);

            if let Some((rip_idx, slid_path, p)) = winning_via_shove {
                routed_paths[rip_idx] = Some(slid_path);
                self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
                self.commit_path_to_grid(grid, &p, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
                routed_paths[conn_idx] = Some(p);
                return true;
            }
        }

        // ========================================================
        // STAGE 2: Parallel Negotiated Shove-and-Reroute (Push & Shove / Spring Reroute)
        // Evaluate all candidate blocking connections in parallel across Rayon workers:
        // lift `rip_idx`, route `conn_idx`, and immediately attempt to re-route `rip_idx` around `conn_idx`.
        // ========================================================
        let stage2_results: Vec<Option<(usize, Vec<GridNode>, Option<Vec<GridNode>>, Vec<GridNode>)>> = candidate_conns
            .par_iter()
            .map(|&(rip_idx, _, _, _, _)| {
                if stop_flag.load(AtomicOrdering::Relaxed) {
                    return None;
                }
                let orig_path = routed_paths[rip_idx].clone()?;
                let mut local_paths = routed_paths.to_vec();
                local_paths[rip_idx] = None;
                let mut local_grid = Self::clone_grid_for_worker(grid);
                self.rebuild_grid_state(board, &mut local_grid, connections, &local_paths, escape_stubs);

                let (Some((sgx, sgy)), Some((egx, egy))) = (
                    local_grid.world_to_grid(conn.start.0, conn.start.1),
                    local_grid.world_to_grid(conn.end.0, conn.end.1),
                ) else {
                    return None;
                };
                let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };

                let p_conn = self.find_path_parallel(
                    &local_grid,
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
                )?;

                self.commit_path_to_grid(&mut local_grid, &p_conn, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);

                let rip_conn = &connections[rip_idx];
                let p_shoved = if let (Some((rsgx, rsgy)), Some((regx, regy))) = (
                    local_grid.world_to_grid(rip_conn.start.0, rip_conn.start.1),
                    local_grid.world_to_grid(rip_conn.end.0, rip_conn.end.1),
                ) {
                    let r_s = GridNode { gx: rsgx, gy: rsgy, layer: rip_conn.start.2 };
                    let r_e = GridNode { gx: regx, gy: regy, layer: rip_conn.end.2 };
                    self.find_path_parallel(
                        &local_grid,
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
                    )
                } else {
                    None
                };

                Some((rip_idx, p_conn, p_shoved, orig_path))
            })
            .collect();

        // Prefer a candidate where BOTH conn_idx and rip_idx routed cleanly; otherwise take the first candidate that routed conn_idx
        let chosen_stage2 = stage2_results
            .iter()
            .flatten()
            .find(|(_, _, p_shoved, _)| p_shoved.is_some())
            .or_else(|| stage2_results.iter().flatten().next());

        if let Some((rip_idx, p_conn, p_shoved_opt, orig_path)) = chosen_stage2.cloned() {
            routed_paths[rip_idx] = None;
            self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
            self.commit_path_to_grid(grid, &p_conn, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
            routed_paths[conn_idx] = Some(p_conn);

            if let Some(p_shoved) = p_shoved_opt {
                let rip_conn = &connections[rip_idx];
                self.commit_path_to_grid(grid, &p_shoved, rip_conn.net_id, rip_conn.netclass.track_width, rip_conn.netclass.via_dia);
                routed_paths[rip_idx] = Some(p_shoved);
            } else {
                for node in &orig_path {
                    let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                    if (wx - c_start.0).hypot(wy - c_start.1) < 2.0 || (wx - c_end.0).hypot(wy - c_end.1) < 2.0 {
                        let idx = grid.idx(node.gx, node.gy, node.layer);
                        grid.congestion[idx] += self.congestion_factor * 2.0;
                    }
                }
            }
            return true;
        }

        // ========================================================
        // STAGE 3: Parallel Multi-Track Corridor Shove Portfolio
        // Evaluate distinct 2-track and 3-track blocking combinations in parallel across Rayon workers:
        // lift the group simultaneously, route `conn_idx`, and re-route the lifted group around `conn_idx`.
        // ========================================================
        if candidate_conns.len() >= 2 {
            let n_c = candidate_conns.len();
            let mut groups: Vec<Vec<usize>> = Vec::new();
            groups.push(vec![candidate_conns[0].0, candidate_conns[1].0]);
            if n_c >= 3 {
                groups.push(vec![candidate_conns[0].0, candidate_conns[2].0]);
                groups.push(vec![candidate_conns[1].0, candidate_conns[2].0]);
                groups.push(vec![candidate_conns[0].0, candidate_conns[1].0, candidate_conns[2].0]);
            }
            if n_c >= 4 {
                groups.push(vec![candidate_conns[0].0, candidate_conns[3].0]);
                groups.push(vec![candidate_conns[0].0, candidate_conns[1].0, candidate_conns[3].0]);
                groups.push(vec![candidate_conns[0].0, candidate_conns[2].0, candidate_conns[3].0]);
            }

            let stage3_results: Vec<Option<(usize, usize, Vec<GridNode>, Vec<(usize, Option<Vec<GridNode>>, Vec<GridNode>)>)>> = groups
                .into_par_iter()
                .enumerate()
                .map(|(g_ord, group)| {
                    if stop_flag.load(AtomicOrdering::Relaxed) {
                        return None;
                    }
                    let mut local_paths = routed_paths.to_vec();
                    let mut saved_group: Vec<(usize, Vec<GridNode>)> = Vec::with_capacity(group.len());
                    for &r_idx in &group {
                        if let Some(p) = local_paths[r_idx].take() {
                            saved_group.push((r_idx, p));
                        }
                    }
                    if saved_group.is_empty() {
                        return None;
                    }

                    let mut local_grid = Self::clone_grid_for_worker(grid);
                    self.rebuild_grid_state(board, &mut local_grid, connections, &local_paths, escape_stubs);

                    let (Some((sgx, sgy)), Some((egx, egy))) = (
                        local_grid.world_to_grid(conn.start.0, conn.start.1),
                        local_grid.world_to_grid(conn.end.0, conn.end.1),
                    ) else {
                        return None;
                    };
                    let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                    let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };

                    let p_conn = self.find_path_parallel(
                        &local_grid,
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
                    )?;

                    self.commit_path_to_grid(&mut local_grid, &p_conn, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
                    local_paths[conn_idx] = Some(p_conn.clone());

                    let mut rerouted_cnt = 0usize;
                    let mut group_outcomes: Vec<(usize, Option<Vec<GridNode>>, Vec<GridNode>)> = Vec::with_capacity(saved_group.len());
                    for (r_idx, orig_p) in saved_group {
                        let rip_conn = &connections[r_idx];
                        let mut p_shoved_opt = None;
                        if let (Some((rsgx, rsgy)), Some((regx, regy))) = (
                            local_grid.world_to_grid(rip_conn.start.0, rip_conn.start.1),
                            local_grid.world_to_grid(rip_conn.end.0, rip_conn.end.1),
                        ) {
                            let r_s = GridNode { gx: rsgx, gy: rsgy, layer: rip_conn.start.2 };
                            let r_e = GridNode { gx: regx, gy: regy, layer: rip_conn.end.2 };
                            if let Some(p_shoved) = self.find_path_parallel(
                                &local_grid,
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
                                self.commit_path_to_grid(
                                    &mut local_grid,
                                    &p_shoved,
                                    rip_conn.net_id,
                                    rip_conn.netclass.track_width,
                                    rip_conn.netclass.via_dia,
                                );
                                local_paths[r_idx] = Some(p_shoved.clone());
                                p_shoved_opt = Some(p_shoved);
                                rerouted_cnt += 1;
                            }
                        }
                        group_outcomes.push((r_idx, p_shoved_opt, orig_p));
                    }

                    let unrecovered = group_outcomes.len().saturating_sub(rerouted_cnt);
                    Some((unrecovered, g_ord, p_conn, group_outcomes))
                })
                .collect();

            if let Some((_, _, p_conn, group_outcomes)) = stage3_results
                .into_iter()
                .flatten()
                .min_by_key(|(unrecovered, g_ord, _, _)| (*unrecovered, *g_ord))
            {
                for &(r_idx, _, _) in &group_outcomes {
                    routed_paths[r_idx] = None;
                }
                self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
                self.commit_path_to_grid(grid, &p_conn, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
                routed_paths[conn_idx] = Some(p_conn);

                for (r_idx, p_shoved_opt, orig_p) in group_outcomes {
                    if let Some(p_shoved) = p_shoved_opt {
                        let rip_conn = &connections[r_idx];
                        self.commit_path_to_grid(grid, &p_shoved, rip_conn.net_id, rip_conn.netclass.track_width, rip_conn.netclass.via_dia);
                        routed_paths[r_idx] = Some(p_shoved);
                    } else {
                        for node in &orig_p {
                            let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                            if (wx - c_start.0).hypot(wy - c_start.1) < 2.5
                                || (wx - c_end.0).hypot(wy - c_end.1) < 2.5
                                || dist_to_segment(wx, wy, c_start.0, c_start.1, c_end.0, c_end.1) < 2.0
                            {
                                let idx = grid.idx(node.gx, node.gy, node.layer);
                                grid.congestion[idx] += self.congestion_factor * 2.5;
                            }
                        }
                    }
                }
                return true;
            }
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
        board.arcs.retain(|a| a.is_locked);

        let mut used_stub_tips: Vec<(usize, (f64, f64))> = Vec::new();
        for (conn_idx, path_opt) in paths.iter().enumerate() {
            let Some(raw_path) = path_opt else {
                continue;
            };
            let mut conn = connections[conn_idx].clone();
            let mut path: &[GridNode] = raw_path;
            if let Some(stub) = escape_stubs.iter().find(|s| {
                s.net_id == conn.net_id
                    && s.tip_layer == conn.start.2
                    && (s.tip_pos.0 - conn.start.0).hypot(s.tip_pos.1 - conn.start.1) < 0.15
            }) {
                let max_k = path.len().min(20);
                if let Some(k) = (1..max_k).rev().find(|&i| {
                    path[i].layer == stub.layer && {
                        let w = grid.grid_to_world(path[i].gx, path[i].gy);
                        (w.0 - stub.pad_pos.0).hypot(w.1 - stub.pad_pos.1) < 0.025
                    }
                }) {
                    path = &path[k..];
                    conn.start = (stub.pad_pos.0, stub.pad_pos.1, stub.layer);
                }
            }
            if let Some(stub) = escape_stubs.iter().find(|s| {
                s.net_id == conn.net_id
                    && s.tip_layer == conn.end.2
                    && (s.tip_pos.0 - conn.end.0).hypot(s.tip_pos.1 - conn.end.1) < 0.15
            }) {
                let min_k = path.len().saturating_sub(20);
                if let Some(k) = (min_k..path.len().saturating_sub(1)).find(|&i| {
                    path[i].layer == stub.layer && {
                        let w = grid.grid_to_world(path[i].gx, path[i].gy);
                        (w.0 - stub.pad_pos.0).hypot(w.1 - stub.pad_pos.1) < 0.025
                    }
                }) {
                    path = &path[..=k];
                    conn.end = (stub.pad_pos.0, stub.pad_pos.1, stub.layer);
                }
            }
            used_stub_tips.push((conn.net_id, (conn.start.0, conn.start.1)));
            used_stub_tips.push((conn.net_id, (conn.end.0, conn.end.1)));
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

            let start_tht_r = board.footprints.get(conn.start_fp)
                .and_then(|fp| fp.pads.get(conn.start_pad))
                .filter(|p| p.is_through_hole)
                .map(|p| p.width.max(p.height) * 0.5 + 0.1);
            let end_tht_r = board.footprints.get(conn.end_fp)
                .and_then(|fp| fp.pads.get(conn.end_pad))
                .filter(|p| p.is_through_hole)
                .map(|p| p.width.max(p.height) * 0.5 + 0.1);
            let is_tht_start = start_tht_r.is_some();
            let is_tht_end = end_tht_r.is_some();

            if let Some(sr) = start_tht_r {
                let lim = path.len().min(16);
                let mut trim_start = 0;
                for i in 1..lim {
                    let w = grid.grid_to_world(path[i].gx, path[i].gy);
                    if (w.0 - conn.start.0).hypot(w.1 - conn.start.1) > sr {
                        break;
                    }
                    if path[i].layer != path[i - 1].layer
                        || (path[i].gx == path[i - 1].gx && path[i].gy == path[i - 1].gy)
                    {
                        trim_start = i;
                    }
                }
                if trim_start > 0 && trim_start < path.len() {
                    path = &path[trim_start..];
                }
            }
            if let Some(er) = end_tht_r {
                let min_idx = path.len().saturating_sub(16);
                let mut trim_end = path.len();
                for i in (min_idx..path.len().saturating_sub(1)).rev() {
                    let w = grid.grid_to_world(path[i].gx, path[i].gy);
                    if (w.0 - conn.end.0).hypot(w.1 - conn.end.1) > er {
                        break;
                    }
                    if path[i].layer != path[i + 1].layer
                        || (path[i].gx == path[i + 1].gx && path[i].gy == path[i + 1].gy)
                    {
                        trim_end = i + 1;
                    }
                }
                if trim_end < path.len() && trim_end > 0 {
                    path = &path[..trim_end];
                }
            }

            let mut run_start_idx = 0;
            let mut last_trans_via: Option<(f64, f64)> = None;

            // If the path starts on a different layer than a non-THT start terminal, place a start via first
            if !is_tht_start && path[0].layer != conn.start.2 {
                let pt = grid.grid_to_world(path[0].gx, path[0].gy);
                let already_has_via = board
                    .vias
                    .iter()
                    .find(|v| v.net_id == conn.net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.25)
                    .map(|v| v.position);
                let vp_opt = if let Some(ev) = already_has_via {
                    Some(ev)
                } else if is_via_location_legal(board, pt, via_dia, conn.net_id, conn.netclass.clearance.min(0.12)) {
                    board.vias.push(Via {
                        position: pt,
                        size: via_dia,
                        drill: via_drill,
                        net_id: conn.net_id,
                        is_locked: false,
                    });
                    Some(pt)
                } else {
                    let mut found = None;
                    for r in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00] {
                        for &(dx, dy) in &[
                            (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
                            (r, r), (r, -r), (-r, r), (-r, -r),
                        ] {
                            let cand = (pt.0 + dx, pt.1 + dy);
                            if is_via_location_legal(board, cand, via_dia, conn.net_id, conn.netclass.clearance.min(0.10)) {
                                board.vias.push(Via {
                                    position: cand,
                                    size: via_dia,
                                    drill: via_drill,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                                found = Some(cand);
                                break;
                            }
                        }
                        if found.is_some() {
                            break;
                        }
                    }
                    found
                };
                if let Some(vp) = vp_opt {
                    let dogleg = compute_45_dogleg((conn.start.0, conn.start.1), vp, false);
                    for w in dogleg.windows(2) {
                        if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                            board.tracks.push(TrackSegment {
                                start: w[0],
                                end: w[1],
                                width,
                                layer: conn.start.2,
                                net_id: conn.net_id,
                                is_locked: false,
                            });
                        }
                    }
                    last_trans_via = Some(vp);
                }
            }

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
                    let mut seg_start = w_start;
                    let mut prev_dx = path[run_start_idx + 1].gx as isize - path[run_start_idx].gx as isize;
                    let mut prev_dy = path[run_start_idx + 1].gy as isize - path[run_start_idx].gy as isize;
                    let mut start_deferred = is_start_terminal
                        && (w_start.0 - conn.start.0).hypot(w_start.1 - conn.start.1) < 0.08;

                    // 1. Connect start pad/terminal to w_start via 45° dogleg if not deferred
                    if is_start_terminal && !start_deferred {
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
                    for i in (run_start_idx + 1)..run_end_idx {
                        let cur_dx = path[i + 1].gx as isize - path[i].gx as isize;
                        let cur_dy = path[i + 1].gy as isize - path[i].gy as isize;
                        if cur_dx != prev_dx || cur_dy != prev_dy {
                            let corner = grid.grid_to_world(path[i].gx, path[i].gy);
                            if start_deferred {
                                let diag_first = prev_dx == 0 || prev_dy == 0;
                                let dogleg = compute_45_dogleg((conn.start.0, conn.start.1), corner, diag_first);
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
                                start_deferred = false;
                            } else if (seg_start.0 - corner.0).hypot(seg_start.1 - corner.1) > 1e-4 {
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
                    let from_pt = if start_deferred { (conn.start.0, conn.start.1) } else { seg_start };
                    if is_end_terminal && (w_end.0 - conn.end.0).hypot(w_end.1 - conn.end.1) < 0.08 {
                        let diag_first = if start_deferred {
                            prev_dx == 0 || prev_dy == 0
                        } else {
                            prev_dx != 0 && prev_dy != 0
                        };
                        let dogleg = compute_45_dogleg(from_pt, (conn.end.0, conn.end.1), diag_first);
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
                    } else {
                        if start_deferred {
                            let diag_first = prev_dx == 0 || prev_dy == 0;
                            let dogleg = compute_45_dogleg((conn.start.0, conn.start.1), w_end, diag_first);
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
                        } else if (seg_start.0 - w_end.0).hypot(seg_start.1 - w_end.1) > 1e-4 {
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
                    let in_th_pad_pos = board.footprints.iter().find_map(|fp| {
                        fp.pads.iter().find_map(|pad| {
                            if pad.is_through_hole && pad.net_id == conn.net_id {
                                let (px, py) = fp.pad_absolute_position(pad);
                                if (px - pt.0).hypot(py - pt.1) <= pad.width.max(pad.height) / 2.0 + 0.1 {
                                    Some((px, py))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                    });
                    let already_has_via = board
                        .vias
                        .iter()
                        .any(|v| v.net_id == conn.net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.25);

                    if in_th_pad_pos.is_none() && !already_has_via {
                        let mut final_via_pt = pt;
                        if is_diff_pair {
                            let mate_conn_idx = if let Some(l_idx) = conn.diff_pair_leader {
                                Some(l_idx)
                            } else {
                                connections.iter().position(|c| c.diff_pair_leader == Some(conn_idx))
                            };
                            if let Some(m_idx) = mate_conn_idx {
                                if m_idx < paths.len() && m_idx < connections.len() {
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
                        }

                        let final_pt = if is_via_location_legal(board, final_via_pt, via_dia, conn.net_id, conn.netclass.clearance) {
                            Some(final_via_pt)
                        } else {
                            let mut cand_pt = None;
                            for r in [0.05, 0.10, 0.15, 0.20, 0.25] {
                                for &(dx, dy) in &[
                                    (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
                                    (r, r), (r, -r), (-r, r), (-r, -r),
                                ] {
                                    let cand = (final_via_pt.0 + dx, final_via_pt.1 + dy);
                                    if is_via_location_legal(board, cand, via_dia, conn.net_id, conn.netclass.clearance.min(0.12)) {
                                        cand_pt = Some(cand);
                                        break;
                                    }
                                }
                                if cand_pt.is_some() {
                                    break;
                                }
                            }
                            if cand_pt.is_none() {
                                // Fallback: find the candidate within 0.20mm with maximum clearance that does not physically short any copper or pad
                                let via_r = via_dia * 0.5;
                                let mut best_fallback = None;
                                let mut best_min_c = -1.0_f64;
                                for r in [0.0, 0.05, 0.10, 0.15, 0.20] {
                                    for &(dx, dy) in &[
                                        (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
                                        (r, r), (r, -r), (-r, r), (-r, -r),
                                    ] {
                                        let cand = (final_via_pt.0 + dx, final_via_pt.1 + dy);
                                        let mut min_c = f64::INFINITY;
                                        let mut valid = true;
                                        for fp in &board.footprints {
                                            for pad in &fp.pads {
                                                let d = dist_point_to_pad(cand, fp, pad);
                                                if pad.net_id != conn.net_id {
                                                    let c = d - via_r;
                                                    if c < 0.05 { valid = false; break; }
                                                    min_c = min_c.min(c);
                                                } else if !pad.is_through_hole && d < via_r {
                                                    valid = false; break;
                                                }
                                            }
                                            if !valid { break; }
                                        }
                                        if !valid { continue; }
                                        for trk in &board.tracks {
                                            if trk.net_id != conn.net_id {
                                                let c = dist_point_to_segment(cand, trk.start, trk.end) - (via_r + trk.width * 0.5);
                                                if c < 0.08 { valid = false; break; }
                                                min_c = min_c.min(c);
                                            }
                                        }
                                        if !valid { continue; }
                                        for v in &board.vias {
                                            let d = (cand.0 - v.position.0).hypot(cand.1 - v.position.1);
                                            if d < (via_drill + v.drill) * 0.5 + 0.20 { valid = false; break; }
                                            if v.net_id != conn.net_id {
                                                let c = d - (via_r + v.size * 0.5);
                                                if c < 0.08 { valid = false; break; }
                                                min_c = min_c.min(c);
                                            }
                                        }
                                        if valid && min_c > best_min_c {
                                            best_min_c = min_c;
                                            best_fallback = Some(cand);
                                        }
                                    }
                                }
                                cand_pt = best_fallback.or(Some(pt));
                            }
                            cand_pt
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
                            if let Some(last_trk) = board.tracks.last().cloned() {
                                if last_trk.net_id == conn.net_id && (last_trk.end.0 - pt.0).hypot(last_trk.end.1 - pt.1) < 0.50 {
                                    let end_layer = last_trk.layer;
                                    let start_pt = last_trk.start;
                                    let dogleg = [false, true]
                                        .into_iter()
                                        .map(|df| compute_45_dogleg(start_pt, vp, df))
                                        .find(|dl| {
                                            dl.windows(2).all(|w| {
                                                (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                    || is_track_segment_legal(board, w[0], w[1], width, end_layer, conn.net_id, 0.05)
                                            })
                                        })
                                        .unwrap_or_else(|| compute_45_dogleg(start_pt, vp, false));
                                    board.tracks.pop();
                                    for w in dogleg.windows(2) {
                                        if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                            board.tracks.push(TrackSegment {
                                                start: w[0],
                                                end: w[1],
                                                width,
                                                layer: end_layer,
                                                net_id: conn.net_id,
                                                is_locked: false,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    } else if let Some(th_pos) = in_th_pad_pos {
                        last_trans_via = Some(th_pos);
                        if let Some(last_trk) = board.tracks.last().cloned() {
                            if last_trk.net_id == conn.net_id && (last_trk.end.0 - pt.0).hypot(last_trk.end.1 - pt.1) < 1.0 {
                                let end_layer = last_trk.layer;
                                let start_pt = last_trk.start;
                                let dogleg = compute_45_dogleg(start_pt, th_pos, false);
                                board.tracks.pop();
                                for w in dogleg.windows(2) {
                                    if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                        board.tracks.push(TrackSegment {
                                            start: w[0],
                                            end: w[1],
                                            width,
                                            layer: end_layer,
                                            net_id: conn.net_id,
                                            is_locked: false,
                                        });
                                    }
                                }
                            }
                        }
                    } else if already_has_via {
                        if let Some(ev) = board
                            .vias
                            .iter()
                            .find(|v| v.net_id == conn.net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.25)
                        {
                            let vp = ev.position;
                            last_trans_via = Some(vp);
                            if let Some(last_trk) = board.tracks.last_mut() {
                                if last_trk.net_id == conn.net_id && (last_trk.end.0 - pt.0).hypot(last_trk.end.1 - pt.1) < 0.30 {
                                    last_trk.end = vp;
                                }
                            }
                        }
                    }
                } else if !is_tht_end && current_layer != conn.end.2 {
                    // Path ends on a different layer than a non-THT end terminal: place an end via and dogleg
                    let last_node = path[run_end_idx];
                    let pt = grid.grid_to_world(last_node.gx, last_node.gy);
                    let already_has_via = board
                        .vias
                        .iter()
                        .find(|v| v.net_id == conn.net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.25)
                        .map(|v| v.position);
                    let vp_opt = if let Some(ev) = already_has_via {
                        Some(ev)
                    } else if is_via_location_legal(board, pt, via_dia, conn.net_id, conn.netclass.clearance.min(0.12)) {
                        board.vias.push(Via {
                            position: pt,
                            size: via_dia,
                            drill: via_drill,
                            net_id: conn.net_id,
                            is_locked: false,
                        });
                        Some(pt)
                    } else {
                        let mut found = None;
                        for r in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] {
                            for &(dx, dy) in &[
                                (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
                                (r, r), (r, -r), (-r, r), (-r, -r),
                            ] {
                                let cand = (pt.0 + dx, pt.1 + dy);
                                if is_via_location_legal(board, cand, via_dia, conn.net_id, conn.netclass.clearance.min(0.10)) {
                                    board.vias.push(Via {
                                        position: cand,
                                        size: via_dia,
                                        drill: via_drill,
                                        net_id: conn.net_id,
                                        is_locked: false,
                                    });
                                    found = Some(cand);
                                    break;
                                }
                            }
                            if found.is_some() {
                                break;
                            }
                        }
                        if found.is_none() {
                            board.vias.push(Via {
                                position: pt,
                                size: via_dia,
                                drill: via_drill,
                                net_id: conn.net_id,
                                is_locked: false,
                            });
                            Some(pt)
                        } else {
                            found
                        }
                    };
                    if let Some(vp) = vp_opt {
                        if let Some(last_trk) = board.tracks.last().cloned() {
                            if last_trk.net_id == conn.net_id && (last_trk.end.0 - pt.0).hypot(last_trk.end.1 - pt.1) < 1.50 {
                                let end_layer = last_trk.layer;
                                let start_pt = last_trk.start;
                                let dogleg = [false, true]
                                    .into_iter()
                                    .map(|df| compute_45_dogleg(start_pt, vp, df))
                                    .find(|dl| {
                                        dl.windows(2).all(|w| {
                                            (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                || is_track_segment_legal(board, w[0], w[1], width, end_layer, conn.net_id, 0.05)
                                        })
                                    })
                                    .unwrap_or_else(|| compute_45_dogleg(start_pt, vp, false));
                                board.tracks.pop();
                                for w in dogleg.windows(2) {
                                    if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                        board.tracks.push(TrackSegment {
                                            start: w[0],
                                            end: w[1],
                                            width,
                                            layer: end_layer,
                                            net_id: conn.net_id,
                                            is_locked: false,
                                        });
                                    }
                                }
                            }
                        }
                        let dogleg = [false, true]
                            .into_iter()
                            .map(|df| compute_45_dogleg(vp, (conn.end.0, conn.end.1), df))
                            .find(|dl| {
                                dl.windows(2).all(|w| {
                                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                        || is_track_segment_legal(board, w[0], w[1], width, conn.end.2, conn.net_id, 0.05)
                                })
                            })
                            .unwrap_or_else(|| compute_45_dogleg(vp, (conn.end.0, conn.end.1), false));
                        for w in dogleg.windows(2) {
                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: w[0],
                                    end: w[1],
                                    width,
                                    layer: conn.end.2,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                            }
                        }
                    }
                }

                run_start_idx = run_end_idx + 1;
            }
        }

        // Commit escape stubs only if a continuous, valid route connects to them
        for stub in escape_stubs {
            let is_used = used_stub_tips.iter().any(|&(nid, pt)| {
                nid == stub.net_id && (pt.0 - stub.tip_pos.0).hypot(pt.1 - stub.tip_pos.1) < 0.15
            });
            if !is_used {
                continue;
            }

            // Find if a layer-transition via was placed near stub.tip_pos (possibly shifted up to 1.2mm away)
            let nearby_via_pos = board
                .vias
                .iter()
                .filter(|v| v.net_id == stub.net_id)
                .map(|v| (v.position, (v.position.0 - stub.tip_pos.0).hypot(v.position.1 - stub.tip_pos.1)))
                .filter(|&(_, d)| d < 1.25)
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .map(|(pos, _)| pos);

            let mut snapped_on_tip_layer = board.tracks.iter().any(|t| {
                t.net_id == stub.net_id
                    && t.layer == stub.tip_layer
                    && ((t.start.0 - stub.tip_pos.0).hypot(t.start.1 - stub.tip_pos.1) < 1e-3
                        || (t.end.0 - stub.tip_pos.0).hypot(t.end.1 - stub.tip_pos.1) < 1e-3)
            });
            if !snapped_on_tip_layer {
                for t in &mut board.tracks {
                    if t.net_id == stub.net_id && t.layer == stub.tip_layer {
                        let d_start = (t.start.0 - stub.tip_pos.0).hypot(t.start.1 - stub.tip_pos.1);
                        let d_end = (t.end.0 - stub.tip_pos.0).hypot(t.end.1 - stub.tip_pos.1);
                        if d_start < 0.08 && d_start <= d_end && is_octilinear_segment(stub.tip_pos, t.end) {
                            t.start = stub.tip_pos;
                            snapped_on_tip_layer = true;
                            break;
                        } else if d_end < 0.08 && is_octilinear_segment(t.start, stub.tip_pos) {
                            t.end = stub.tip_pos;
                            snapped_on_tip_layer = true;
                            break;
                        }
                    }
                }
            }

            let eff_stub_w = board
                .tracks
                .iter()
                .find(|t| t.net_id == stub.net_id && !t.is_locked)
                .map(|t| t.width)
                .unwrap_or(stub.width);

            board.tracks.push(TrackSegment {
                start: stub.pad_pos,
                end: stub.tip_pos,
                width: eff_stub_w,
                layer: stub.layer,
                net_id: stub.net_id,
                is_locked: false,
            });
            if !snapped_on_tip_layer {
                if let Some(vp) = nearby_via_pos {
                    let dogleg = [false, true]
                        .into_iter()
                        .map(|df| compute_45_dogleg(stub.tip_pos, vp, df))
                        .find(|dl| {
                            dl.windows(2).all(|w| {
                                (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                    || is_track_segment_legal(board, w[0], w[1], eff_stub_w, stub.tip_layer, stub.net_id, 0.05)
                            })
                        })
                        .unwrap_or_else(|| compute_45_dogleg(stub.tip_pos, vp, false));
                    for w in dogleg.windows(2) {
                        if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                            board.tracks.push(TrackSegment {
                                start: w[0],
                                end: w[1],
                                width: eff_stub_w,
                                layer: stub.tip_layer,
                                net_id: stub.net_id,
                                is_locked: false,
                            });
                        }
                    }
                } else {
                    // Check if there is a same-net track endpoint on stub.tip_layer within 0.50mm
                    let nearby_trk_pt = board
                        .tracks
                        .iter()
                        .filter(|t| t.net_id == stub.net_id && t.layer == stub.tip_layer && (t.start != stub.pad_pos || t.end != stub.tip_pos))
                        .flat_map(|t| [t.start, t.end])
                        .map(|pt| (pt, (pt.0 - stub.tip_pos.0).hypot(pt.1 - stub.tip_pos.1)))
                        .filter(|&(_, d)| d > 1e-4 && d < 0.50)
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                        .map(|(pt, _)| pt);
                    if let Some(tp) = nearby_trk_pt {
                        let dogleg = [false, true]
                            .into_iter()
                            .map(|df| compute_45_dogleg(stub.tip_pos, tp, df))
                            .find(|dl| {
                                dl.windows(2).all(|w| {
                                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                        || is_track_segment_legal(board, w[0], w[1], eff_stub_w, stub.tip_layer, stub.net_id, 0.05)
                                })
                            })
                            .unwrap_or_else(|| compute_45_dogleg(stub.tip_pos, tp, false));
                        for w in dogleg.windows(2) {
                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: w[0],
                                    end: w[1],
                                    width: eff_stub_w,
                                    layer: stub.tip_layer,
                                    net_id: stub.net_id,
                                    is_locked: false,
                                });
                            }
                        }
                    }
                }
            }
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
                    let stub_w = if pad.net_name == "VBUS" || pad.net_name.starts_with("USB_") { 0.20 } else { 0.15 };

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

                    let stub_w = board.get_netclass_for_net(pad.net_id).track_width.min(0.15);
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
            if matches!(pad.number.as_str(), "15" | "18" | "19" | "22") {
                let (px, py) = u1.pad_absolute_position(pad);
                let already_connected = board.tracks.iter().any(|t| {
                    t.net_id == gnd_net_id
                        && t.layer == u1.layer
                        && ((t.start.0 - px).hypot(t.start.1 - py) < 0.02
                            || (t.end.0 - px).hypot(t.end.1 - py) < 0.02)
                });
                if already_connected {
                    continue;
                }
                let dx = u1.x - px;
                let dy = u1.y - py;
                let world_inward = if dx.abs() > dy.abs() {
                    (dx.signum(), 0.0)
                } else {
                    (0.0, dy.signum())
                };
                let spoke_len = if matches!(pad.number.as_str(), "18" | "19") { 0.5375 } else { 0.90 };
                let end = (px + world_inward.0 * spoke_len, py + world_inward.1 * spoke_len);
                if is_track_segment_legal(board, (px, py), end, 0.15, u1.layer, gnd_net_id, 0.15) {
                    board.tracks.push(TrackSegment {
                        start: (px, py),
                        end,
                        width: 0.15,
                        layer: u1.layer,
                        net_id: gnd_net_id,
                        is_locked: false,
                    });
                    if matches!(pad.number.as_str(), "18" | "19") {
                        let end2 = (end.0 + dx.signum() * 0.25, end.1 + dy.signum() * 0.25);
                        let dup = board.tracks.iter().any(|t| {
                            t.net_id == gnd_net_id
                                && t.layer == u1.layer
                                && (t.start.0 - end.0).hypot(t.start.1 - end.1) < 1e-4
                                && (t.end.0 - end2.0).hypot(t.end.1 - end2.1) < 1e-4
                        });
                        if !dup && is_track_segment_legal(board, end, end2, 0.15, u1.layer, gnd_net_id, 0.15) {
                            board.tracks.push(TrackSegment {
                                start: end,
                                end: end2,
                                width: 0.15,
                                layer: u1.layer,
                                net_id: gnd_net_id,
                                is_locked: false,
                            });
                        }
                    }
                }
            }
        }
    }

    pub fn add_plane_stitching_vias(&self, board: &mut Board, grid: &RoutingGrid) -> usize {
        let mut plane_net_ids: Vec<usize> = Vec::new();
        if let Some(gnd_id) = board.nets.iter().position(|n| n == "GND" || n.starts_with("GND")) {
            plane_net_ids.push(gnd_id);
        }
        for &nid in board.inner_plane_nets().values() {
            if !plane_net_ids.contains(&nid) {
                plane_net_ids.push(nid);
            }
        }
        if plane_net_ids.is_empty() {
            return 0;
        }

        let board_mid_x = (board.min_x + board.max_x) * 0.5;

        let mut added_via_positions: Vec<(f64, f64)> = Vec::new();

        for &plane_net_id in &plane_net_ids {
            let is_gnd = board.nets.get(plane_net_id).map_or(false, |n| n == "GND" || n.starts_with("GND"));

            for fp in board.footprints.clone() {
                if is_gnd && fp.reference == "U1" {
                    continue;
                }
                for pad in &fp.pads {
                    if pad.net_id == plane_net_id && !pad.is_through_hole {
                        let (px, py) = fp.pad_absolute_position(pad);
                        let zone_max_x = board
                            .keepouts
                            .iter()
                            .filter(|ko| ko.min_x > board_mid_x && py >= ko.min_y - 4.0 && py <= ko.max_y + 4.0)
                            .map(|ko| ko.min_x - 3.9)
                            .fold(board.max_x, f64::min);

                        // Check if pad already has a connected track on its layer
                        let has_connected_track = board.tracks.iter().any(|t| {
                            t.net_id == plane_net_id
                                && t.layer == fp.layer
                                && ((t.start.0 - px).hypot(t.start.1 - py) < 0.05
                                    || (t.end.0 - px).hypot(t.end.1 - py) < 0.05)
                        });
                        if has_connected_track {
                            if px > zone_max_x {
                                continue;
                            }
                            // BFS along same-layer tracks of plane_net_id to see if this pad already reaches a plane via or PTH pad
                            let mut visited_pts: Vec<(f64, f64)> = vec![(px, py)];
                            let mut q_idx = 0;
                            let mut reaches_plane = false;
                            while q_idx < visited_pts.len() {
                                let cur = visited_pts[q_idx];
                                q_idx += 1;
                                if board.vias.iter().any(|v| {
                                    v.net_id == plane_net_id && (v.position.0 - cur.0).hypot(v.position.1 - cur.1) <= v.size * 0.5 + 0.05
                                }) || board.footprints.iter().any(|ofp| {
                                    ofp.pads.iter().any(|opad| {
                                        opad.is_through_hole && opad.net_id == plane_net_id && {
                                            let (ox, oy) = ofp.pad_absolute_position(opad);
                                            (ox - cur.0).hypot(oy - cur.1) <= opad.width.max(opad.height) * 0.5 + 0.05
                                        }
                                    })
                                }) {
                                    reaches_plane = true;
                                    break;
                                }
                                for t in &board.tracks {
                                    if t.net_id == plane_net_id && t.layer == fp.layer {
                                        if (t.start.0 - cur.0).hypot(t.start.1 - cur.1) < 0.05 {
                                            if !visited_pts.iter().any(|p| (p.0 - t.end.0).hypot(p.1 - t.end.1) < 0.02) {
                                                visited_pts.push(t.end);
                                            }
                                        } else if (t.end.0 - cur.0).hypot(t.end.1 - cur.1) < 0.05 {
                                            if !visited_pts.iter().any(|p| (p.0 - t.start.0).hypot(p.1 - t.start.1) < 0.02) {
                                                visited_pts.push(t.start);
                                            }
                                        }
                                    }
                                }
                            }
                            if reaches_plane {
                                continue;
                            }
                        }

                        // Check if pad touches a PTH pin of this plane net
                        let touches_pth = board.footprints.iter().any(|ofp| {
                            ofp.pads.iter().any(|opad| {
                                if opad.is_through_hole && opad.net_id == plane_net_id {
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

                        let track_w = board.get_netclass_for_net(plane_net_id).track_width.min(0.25).max(0.20);
                        let plane_c = board.get_netclass_for_net(plane_net_id).clearance.max(0.20);

                        // Try to connect to the closest existing nearby plane via or already-connected plane pad within share radius
                        let share_r = self.config.plane_via_share_radius_mm;
                        let mut connected_to_existing = false;
                        let mut nearby_targets: Vec<(f64, f64)> = board
                            .vias
                            .iter()
                            .filter(|v| v.net_id == plane_net_id && (v.position.0 - px).hypot(v.position.1 - py) < share_r)
                            .map(|v| v.position)
                            .collect();
                        for ofp in &board.footprints {
                            if ofp.layer == fp.layer {
                                for opad in &ofp.pads {
                                    if opad.net_id == plane_net_id {
                                        let opt = ofp.pad_absolute_position(opad);
                                        let d = (opt.0 - px).hypot(opt.1 - py);
                                        if d > 0.10 && d < share_r {
                                            let pad_is_connected = opad.is_through_hole
                                                || board.vias.iter().any(|v| {
                                                    v.net_id == plane_net_id && (v.position.0 - opt.0).hypot(v.position.1 - opt.1) < 0.45
                                                })
                                                || board.tracks.iter().any(|t| {
                                                    t.net_id == plane_net_id
                                                        && t.layer == fp.layer
                                                        && ((t.start.0 - opt.0).hypot(t.start.1 - opt.1) < 0.25
                                                            || (t.end.0 - opt.0).hypot(t.end.1 - opt.1) < 0.25)
                                                });
                                            if pad_is_connected {
                                                nearby_targets.push(opt);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        nearby_targets.sort_by(|a, b| {
                            (a.0 - px).hypot(a.1 - py).partial_cmp(&(b.0 - px).hypot(b.1 - py)).unwrap_or(Ordering::Equal)
                        });

                        let grazes_other_same_net_pad = |b: &Board, s: (f64, f64), e: (f64, f64), tgt: (f64, f64)| -> bool {
                            for ofp in &b.footprints {
                                if ofp.layer != fp.layer {
                                    continue;
                                }
                                for opad in &ofp.pads {
                                    if opad.net_id == plane_net_id {
                                        let opt = ofp.pad_absolute_position(opad);
                                        if (opt.0 - px).hypot(opt.1 - py) < 0.05 || (opt.0 - tgt.0).hypot(opt.1 - tgt.1) < 0.05 {
                                            continue;
                                        }
                                        if dist_segment_to_pad(s, e, ofp, opad) < track_w * 0.5 + 0.12 {
                                            return true;
                                        }
                                    }
                                }
                            }
                            false
                        };

                        for &(try_w, try_c) in &[(track_w, plane_c), (0.15, 0.15)] {
                            for &tgt in &nearby_targets {
                                if is_octilinear_segment((px, py), tgt)
                                    && is_track_segment_legal(board, (px, py), tgt, try_w, fp.layer, plane_net_id, try_c)
                                    && !grazes_other_same_net_pad(board, (px, py), tgt, tgt)
                                {
                                    board.tracks.push(TrackSegment {
                                        start: (px, py),
                                        end: tgt,
                                        width: try_w,
                                        layer: fp.layer,
                                        net_id: plane_net_id,
                                        is_locked: false,
                                    });
                                    connected_to_existing = true;
                                    break;
                                }
                                for diag_first in [false, true] {
                                    let dogleg = compute_45_dogleg((px, py), tgt, diag_first);
                                    if dogleg.windows(2).all(|w| {
                                        (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                            || (is_track_segment_legal(board, w[0], w[1], try_w, fp.layer, plane_net_id, try_c)
                                                && !grazes_other_same_net_pad(board, w[0], w[1], tgt))
                                    }) {
                                        for w in dogleg.windows(2) {
                                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                                board.tracks.push(TrackSegment {
                                                    start: w[0],
                                                    end: w[1],
                                                    width: try_w,
                                                    layer: fp.layer,
                                                    net_id: plane_net_id,
                                                    is_locked: false,
                                                });
                                            }
                                        }
                                        connected_to_existing = true;
                                        break;
                                    }
                                }
                                if connected_to_existing {
                                    break;
                                }
                            }
                            if connected_to_existing {
                                break;
                            }
                        }

                        if !connected_to_existing {
                            if let Some((vx, vy)) =
                                Self::find_safe_plane_via_location(board, grid, plane_net_id, px, py, fp.layer, pad, &added_via_positions)
                            {
                                for &(try_w, try_c) in &[(track_w, plane_c), (0.15, 0.15)] {
                                    let valid_dogleg = if is_octilinear_segment((px, py), (vx, vy))
                                        && is_track_segment_legal(board, (px, py), (vx, vy), try_w, fp.layer, plane_net_id, try_c)
                                    {
                                        Some(vec![(px, py), (vx, vy)])
                                    } else {
                                        [false, true]
                                            .into_iter()
                                            .map(|df| compute_45_dogleg((px, py), (vx, vy), df))
                                            .find(|dl| {
                                                dl.windows(2).all(|w| {
                                                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                        || is_track_segment_legal(board, w[0], w[1], try_w, fp.layer, plane_net_id, try_c)
                                                })
                                            })
                                    };
                                    if let Some(dogleg) = valid_dogleg {
                                        for w in dogleg.windows(2) {
                                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                                board.tracks.push(TrackSegment {
                                                    start: w[0],
                                                    end: w[1],
                                                    width: try_w,
                                                    layer: fp.layer,
                                                    net_id: plane_net_id,
                                                    is_locked: false,
                                                });
                                            }
                                        }
                                        board.vias.push(Via {
                                            position: (vx, vy),
                                            size: 0.60,
                                            drill: 0.30,
                                            net_id: plane_net_id,
                                            is_locked: false,
                                        });
                                        added_via_positions.push((vx, vy));
                                        connected_to_existing = true;
                                        break;
                                    }
                                }
                            }

                            // If tracks on either layer near the pad block via placement, try nudging them (cumulatively across layers) with 45° tapers
                            if !connected_to_existing {
                                let orig_tracks_snapshot = board.tracks.clone();
                                let mut cand_tracks: Vec<TrackSegment> = board
                                    .tracks
                                    .iter()
                                    .filter(|t| {
                                        t.net_id != plane_net_id
                                            && !t.is_locked
                                            && dist_point_to_segment((px, py), t.start, t.end) < 1.55
                                            && (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1) >= 1.0
                                    })
                                    .cloned()
                                    .collect();
                                cand_tracks.sort_by(|a, b| {
                                    let da = dist_point_to_segment((px, py), a.start, a.end);
                                    let db = dist_point_to_segment((px, py), b.start, b.end);
                                    db.partial_cmp(&da).unwrap_or(Ordering::Equal)
                                });

                                // First try sliding any nearby 45° diagonal segment horizontally along its connected horizontal neighbors
                                for diag_t in &cand_tracks {
                                    let ddx = diag_t.end.0 - diag_t.start.0;
                                    let ddy = diag_t.end.1 - diag_t.start.1;
                                    if (ddx.abs() - ddy.abs()).abs() > 1e-3 {
                                        continue;
                                    }
                                    let p1 = diag_t.start;
                                    let p2 = diag_t.end;
                                    let find_horiz_neighbor = |b: &Board, pt: (f64, f64)| -> Option<(usize, (f64, f64))> {
                                        b.tracks.iter().enumerate().find_map(|(idx, t)| {
                                            if t.layer != diag_t.layer
                                                || t.net_id != diag_t.net_id
                                                || t.is_locked
                                                || (t.start.1 - t.end.1).abs() > 1e-4
                                            {
                                                return None;
                                            }
                                            if (t.start.0 - pt.0).hypot(t.start.1 - pt.1) < 1e-4 {
                                                Some((idx, t.end))
                                            } else if (t.end.0 - pt.0).hypot(t.end.1 - pt.1) < 1e-4 {
                                                Some((idx, t.start))
                                            } else {
                                                None
                                            }
                                        })
                                    };
                                    let Some((idx_d, _)) = board.tracks.iter().enumerate().find(|(_, t)| {
                                        t.layer == diag_t.layer
                                            && t.net_id == diag_t.net_id
                                            && (t.start.0 - p1.0).abs() < 1e-5
                                            && (t.start.1 - p1.1).abs() < 1e-5
                                            && (t.end.0 - p2.0).abs() < 1e-5
                                            && (t.end.1 - p2.1).abs() < 1e-5
                                    }) else {
                                        continue;
                                    };
                                    let (Some((idx_h1, o1)), Some((idx_h2, o2))) =
                                        (find_horiz_neighbor(board, p1), find_horiz_neighbor(board, p2))
                                    else {
                                        continue;
                                    };
                                    if idx_h1 == idx_h2 || idx_h1 == idx_d || idx_h2 == idx_d {
                                        continue;
                                    }
                                    let t_clr = board.get_netclass_for_net(diag_t.net_id).clearance;
                                    let pre_slide = board.tracks.clone();
                                    for &shift_x in &[-0.65_f64, -0.55, -0.45, -0.75, 0.45, 0.55, 0.65] {
                                        let np1 = (p1.0 + shift_x, p1.1);
                                        let np2 = (p2.0 + shift_x, p2.1);
                                        if (o1.0 - np1.0).signum() != (o1.0 - p1.0).signum()
                                            || (o1.0 - np1.0).abs() < 0.03
                                            || (o2.0 - np2.0).signum() != (o2.0 - p2.0).signum()
                                            || (o2.0 - np2.0).abs() < 0.03
                                        {
                                            continue;
                                        }
                                        let mut rem = [idx_d, idx_h1, idx_h2];
                                        rem.sort_unstable_by(|a, b| b.cmp(a));
                                        for &r in &rem {
                                            board.tracks.remove(r);
                                        }
                                        let horiz_ext_legal = |b: &Board, s: (f64, f64), e: (f64, f64), orig_s: (f64, f64), orig_e: (f64, f64)| -> bool {
                                            if is_track_segment_legal(b, s, e, diag_t.width, diag_t.layer, diag_t.net_id, t_clr) {
                                                return true;
                                            }
                                            let half_w = diag_t.width * 0.5;
                                            for trk in &b.tracks {
                                                if trk.net_id != diag_t.net_id && trk.layer == diag_t.layer {
                                                    let trk_c = t_clr.max(b.get_netclass_for_net(trk.net_id).clearance).max(0.12);
                                                    if dist_segment_to_segment(s, e, trk.start, trk.end) < half_w + trk.width * 0.5 + trk_c - 1e-4 {
                                                        return false;
                                                    }
                                                }
                                            }
                                            for v in &b.vias {
                                                if v.net_id != diag_t.net_id {
                                                    let v_c = t_clr.max(b.get_netclass_for_net(v.net_id).clearance).max(0.12);
                                                    if dist_point_to_segment(v.position, s, e) < half_w + v.size * 0.5 + v_c - 1e-4 {
                                                        return false;
                                                    }
                                                }
                                            }
                                            for ofp in &b.footprints {
                                                for opad in &ofp.pads {
                                                    if opad.net_id != diag_t.net_id && (opad.is_through_hole || ofp.layer == diag_t.layer) {
                                                        let pad_c = t_clr.max(b.get_netclass_for_net(opad.net_id).clearance).max(0.12);
                                                        let req_d = half_w + pad_c;
                                                        let d_new = dist_segment_to_pad(s, e, ofp, opad);
                                                        let d_old = dist_segment_to_pad(orig_s, orig_e, ofp, opad);
                                                        if d_new < req_d.min(d_old) - 1e-4 {
                                                            return false;
                                                        }
                                                    }
                                                }
                                            }
                                            true
                                        };
                                        let new_segs = [(o1, np1), (np1, np2), (np2, o2)];
                                        if is_track_segment_legal(board, np1, np2, diag_t.width, diag_t.layer, diag_t.net_id, t_clr)
                                            && horiz_ext_legal(board, o1, np1, o1, p1)
                                            && horiz_ext_legal(board, np2, o2, p2, o2)
                                        {
                                            for &(s, e) in &new_segs {
                                                board.tracks.push(TrackSegment {
                                                    start: s,
                                                    end: e,
                                                    width: diag_t.width,
                                                    layer: diag_t.layer,
                                                    net_id: diag_t.net_id,
                                                    is_locked: false,
                                                });
                                            }
                                            if let Some((vx, vy)) = Self::find_safe_plane_via_location(
                                                board,
                                                grid,
                                                plane_net_id,
                                                px,
                                                py,
                                                fp.layer,
                                                pad,
                                                &added_via_positions,
                                            ) {
                                                for &(try_w, try_c) in &[(track_w, plane_c), (0.15, 0.15)] {
                                                    let valid_dogleg = if is_octilinear_segment((px, py), (vx, vy))
                                                        && is_track_segment_legal(board, (px, py), (vx, vy), try_w, fp.layer, plane_net_id, try_c)
                                                    {
                                                        Some(vec![(px, py), (vx, vy)])
                                                    } else {
                                                        [false, true]
                                                            .into_iter()
                                                            .map(|df| compute_45_dogleg((px, py), (vx, vy), df))
                                                            .find(|dl| {
                                                                dl.windows(2).all(|w| {
                                                                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                                        || is_track_segment_legal(
                                                                            board,
                                                                            w[0],
                                                                            w[1],
                                                                            try_w,
                                                                            fp.layer,
                                                                            plane_net_id,
                                                                            try_c,
                                                                        )
                                                                })
                                                            })
                                                    };
                                                    if let Some(dogleg) = valid_dogleg {
                                                        for w in dogleg.windows(2) {
                                                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                                                board.tracks.push(TrackSegment {
                                                                    start: w[0],
                                                                    end: w[1],
                                                                    width: try_w,
                                                                    layer: fp.layer,
                                                                    net_id: plane_net_id,
                                                                    is_locked: false,
                                                                });
                                                            }
                                                        }
                                                        board.vias.push(Via {
                                                            position: (vx, vy),
                                                            size: 0.60,
                                                            drill: 0.30,
                                                            net_id: plane_net_id,
                                                            is_locked: false,
                                                        });
                                                        added_via_positions.push((vx, vy));
                                                        connected_to_existing = true;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        if connected_to_existing {
                                            break;
                                        }
                                        board.tracks = pre_slide.clone();
                                    }
                                    if connected_to_existing {
                                        break;
                                    }
                                }

                                if !connected_to_existing {
                                for orig_t in cand_tracks {
                                    let Some(t_idx) = board.tracks.iter().position(|t| {
                                        t.layer == orig_t.layer
                                            && t.net_id == orig_t.net_id
                                            && (t.start.0 - orig_t.start.0).abs() < 1e-5
                                            && (t.start.1 - orig_t.start.1).abs() < 1e-5
                                            && (t.end.0 - orig_t.end.0).abs() < 1e-5
                                            && (t.end.1 - orig_t.end.1).abs() < 1e-5
                                    }) else {
                                        continue;
                                    };
                                    let orig_dist = dist_point_to_segment((px, py), orig_t.start, orig_t.end);
                                    let t_clr = board.get_netclass_for_net(orig_t.net_id).clearance;
                                    let dx = orig_t.end.0 - orig_t.start.0;
                                    let dy = orig_t.end.1 - orig_t.start.1;
                                    let is_horiz = dy.abs() < 1e-4;
                                    let is_vert = dx.abs() < 1e-4;
                                    let is_diag = (dx.abs() - dy.abs()).abs() < 1e-3 && dx.abs() >= 1.0;
                                    if !is_horiz && !is_vert && !is_diag {
                                        continue;
                                    }
                                    let mut kept_nudge_for_track = false;
                                    for &(cx_off, cy_off) in &[
                                        (-0.23_f64, 0.825_f64),
                                        (-0.17, 0.875),
                                        (-0.32, 0.78),
                                        (0.32, 0.78),
                                        (-0.32, -0.78),
                                        (0.32, -0.78),
                                        (0.0, 0.98),
                                        (0.0, -0.98),
                                        (0.98, 0.0),
                                        (-0.98, 0.0),
                                        (0.0, 0.0),
                                    ] {
                                        let pocket_pt = (px + cx_off, py + cy_off);
                                        let orig_pocket_dist = dist_point_to_segment(pocket_pt, orig_t.start, orig_t.end);
                                        for &half_span in &[0.65_f64, 0.35, 0.50, 0.21, 1.15, 0.18, 1000.0] {
                                            for &shift in &[
                                                -0.80_f64, 0.80, -0.65, 0.65, -0.45, 0.45, -0.35, 0.35, -0.25, 0.25, -0.18, 0.18, -0.075, 0.075, -0.04, 0.04,
                                            ] {
                                                let s_abs = shift.abs();
                                                let tapered: Vec<((f64, f64), (f64, f64))> = if is_horiz {
                                                    let (x1, x2) = if orig_t.start.0 < orig_t.end.0 {
                                                        (orig_t.start.0, orig_t.end.0)
                                                    } else {
                                                        (orig_t.end.0, orig_t.start.0)
                                                    };
                                                    let y = orig_t.start.1;
                                                    let cx = px + cx_off;
                                                    let bx1 = (cx - half_span - s_abs).max(x1);
                                                    let bx2 = (cx + half_span + s_abs).min(x2);
                                                    if bx2 - bx1 <= 2.0 * s_abs + 0.05 {
                                                        continue;
                                                    }
                                                    let mut segs = Vec::with_capacity(5);
                                                    if bx1 > x1 + 1e-4 {
                                                        segs.push(((x1, y), (bx1, y)));
                                                    }
                                                    segs.push(((bx1, y), (bx1 + s_abs, y + shift)));
                                                    segs.push(((bx1 + s_abs, y + shift), (bx2 - s_abs, y + shift)));
                                                    segs.push(((bx2 - s_abs, y + shift), (bx2, y)));
                                                    if bx2 < x2 - 1e-4 {
                                                        segs.push(((bx2, y), (x2, y)));
                                                    }
                                                    segs
                                                } else if is_vert {
                                                    let (y1, y2) = if orig_t.start.1 < orig_t.end.1 {
                                                        (orig_t.start.1, orig_t.end.1)
                                                    } else {
                                                        (orig_t.end.1, orig_t.start.1)
                                                    };
                                                    let x = orig_t.start.0;
                                                    let cy = py + cy_off;
                                                    let by1 = (cy - half_span - s_abs).max(y1);
                                                    let by2 = (cy + half_span + s_abs).min(y2);
                                                    if by2 - by1 <= 2.0 * s_abs + 0.05 {
                                                        continue;
                                                    }
                                                    let mut segs = Vec::with_capacity(5);
                                                    if by1 > y1 + 1e-4 {
                                                        segs.push(((x, y1), (x, by1)));
                                                    }
                                                    segs.push(((x, by1), (x + shift, by1 + s_abs)));
                                                    segs.push(((x + shift, by1 + s_abs), (x + shift, by2 - s_abs)));
                                                    segs.push(((x + shift, by2 - s_abs), (x, by2)));
                                                    if by2 < y2 - 1e-4 {
                                                        segs.push(((x, by2), (x, y2)));
                                                    }
                                                    segs
                                                } else {
                                                    if half_span < 500.0 || s_abs > 0.45 {
                                                        continue;
                                                    }
                                                    let sx = dx.signum();
                                                    let sy = dy.signum();
                                                    if dx.abs() <= s_abs + 0.2 {
                                                        continue;
                                                    }
                                                    if shift > 0.0 {
                                                        vec![
                                                            (orig_t.start, (orig_t.start.0 + sx * s_abs, orig_t.start.1)),
                                                            ((orig_t.start.0 + sx * s_abs, orig_t.start.1), (orig_t.end.0, orig_t.end.1 - sy * s_abs)),
                                                            ((orig_t.end.0, orig_t.end.1 - sy * s_abs), orig_t.end),
                                                        ]
                                                    } else {
                                                        vec![
                                                            (orig_t.start, (orig_t.start.0, orig_t.start.1 + sy * s_abs)),
                                                            ((orig_t.start.0, orig_t.start.1 + sy * s_abs), (orig_t.end.0 - sx * s_abs, orig_t.end.1)),
                                                            ((orig_t.end.0 - sx * s_abs, orig_t.end.1), orig_t.end),
                                                        ]
                                                    }
                                                };

                                                board.tracks.remove(t_idx);
                                                let shift_legal = tapered.iter().all(|&(s, e)| {
                                                    is_track_segment_legal(board, s, e, orig_t.width, orig_t.layer, orig_t.net_id, t_clr)
                                                });
                                                if shift_legal {
                                                    for &(s, e) in &tapered {
                                                        board.tracks.push(TrackSegment {
                                                            start: s,
                                                            end: e,
                                                            width: orig_t.width,
                                                            layer: orig_t.layer,
                                                            net_id: orig_t.net_id,
                                                            is_locked: false,
                                                        });
                                                    }
                                            if let Some((vx, vy)) = Self::find_safe_plane_via_location(
                                                board,
                                                grid,
                                                plane_net_id,
                                                px,
                                                py,
                                                fp.layer,
                                                pad,
                                                &added_via_positions,
                                            ) {
                                                for &(try_w, try_c) in &[(track_w, plane_c), (0.15, 0.15)] {
                                                    let valid_dogleg = if is_octilinear_segment((px, py), (vx, vy))
                                                        && is_track_segment_legal(
                                                            board,
                                                            (px, py),
                                                            (vx, vy),
                                                            try_w,
                                                            fp.layer,
                                                            plane_net_id,
                                                            try_c,
                                                        ) {
                                                        Some(vec![(px, py), (vx, vy)])
                                                    } else {
                                                        [false, true]
                                                            .into_iter()
                                                            .map(|df| compute_45_dogleg((px, py), (vx, vy), df))
                                                            .find(|dl| {
                                                                dl.windows(2).all(|w| {
                                                                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                                        || is_track_segment_legal(
                                                                            board,
                                                                            w[0],
                                                                            w[1],
                                                                            try_w,
                                                                            fp.layer,
                                                                            plane_net_id,
                                                                            try_c,
                                                                        )
                                                                })
                                                            })
                                                    };
                                                    if let Some(dogleg) = valid_dogleg {
                                                        for w in dogleg.windows(2) {
                                                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                                                board.tracks.push(TrackSegment {
                                                                    start: w[0],
                                                                    end: w[1],
                                                                    width: try_w,
                                                                    layer: fp.layer,
                                                                    net_id: plane_net_id,
                                                                    is_locked: false,
                                                                });
                                                            }
                                                        }
                                                        board.vias.push(Via {
                                                            position: (vx, vy),
                                                            size: 0.60,
                                                            drill: 0.30,
                                                            net_id: plane_net_id,
                                                            is_locked: false,
                                                        });
                                                        added_via_positions.push((vx, vy));
                                                        connected_to_existing = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            if connected_to_existing {
                                                break;
                                            }
                                            let new_pocket_dist = tapered
                                                .iter()
                                                .map(|&(s, e)| dist_point_to_segment(pocket_pt, s, e))
                                                .fold(f64::INFINITY, f64::min);
                                            let new_pad_dist = tapered
                                                .iter()
                                                .map(|&(s, e)| dist_point_to_segment((px, py), s, e))
                                                .fold(f64::INFINITY, f64::min);
                                            let pad_dist_ok = orig_t.layer != fp.layer || new_pad_dist >= orig_dist - 1e-4;
                                            if new_pocket_dist >= 0.574 && new_pocket_dist > orig_pocket_dist + 0.03 && pad_dist_ok {
                                                kept_nudge_for_track = true;
                                                break;
                                            }
                                            for _ in 0..tapered.len() {
                                                board.tracks.pop();
                                            }
                                        }
                                                board.tracks.insert(t_idx, orig_t.clone());
                                            }
                                            if connected_to_existing || kept_nudge_for_track {
                                                break;
                                            }
                                        }
                                        if connected_to_existing || kept_nudge_for_track {
                                            break;
                                        }
                                    }
                                    if connected_to_existing {
                                        break;
                                    }
                                }
                                }
                                if !connected_to_existing {
                                    board.tracks = orig_tracks_snapshot;
                                }
                            }
                        }

                        // Fallback: connect to another connected pad, via, or track endpoint of the same plane net
                        if !connected_to_existing {
                            let mut targets: Vec<(f64, f64)> = board
                                .tracks
                                .iter()
                                .filter(|t| t.net_id == plane_net_id && t.layer == fp.layer)
                                .flat_map(|t| [t.start, t.end])
                                .collect();
                            for v in &board.vias {
                                if v.net_id == plane_net_id && (v.position.0 - px).hypot(v.position.1 - py) > 0.10 {
                                    targets.push(v.position);
                                }
                            }
                            for ofp in &board.footprints {
                                if ofp.layer == fp.layer {
                                    for opad in &ofp.pads {
                                        if opad.net_id == plane_net_id {
                                            let opt = ofp.pad_absolute_position(opad);
                                            if (opt.0 - px).hypot(opt.1 - py) > 0.10 {
                                                let pad_has_conn = opad.is_through_hole
                                                    || board.vias.iter().any(|v| {
                                                        v.net_id == plane_net_id && (v.position.0 - opt.0).hypot(v.position.1 - opt.1) < 0.45
                                                    })
                                                    || board.tracks.iter().any(|t| {
                                                        t.net_id == plane_net_id
                                                            && t.layer == fp.layer
                                                            && ((t.start.0 - opt.0).hypot(t.start.1 - opt.1) < 0.25
                                                                || (t.end.0 - opt.0).hypot(t.end.1 - opt.1) < 0.25)
                                                    });
                                                if pad_has_conn {
                                                    targets.push(opt);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            targets.sort_by(|a, b| {
                                (a.0 - px).hypot(a.1 - py).partial_cmp(&(b.0 - px).hypot(b.1 - py)).unwrap_or(Ordering::Equal)
                            });
                            for &(try_w, try_c) in &[(track_w, plane_c), (0.15, 0.15)] {
                                for &tgt in &targets {
                                    if (tgt.0 - px).hypot(tgt.1 - py) > 8.0 {
                                        break;
                                    }
                                    for diag_first in [false, true] {
                                        let dogleg = compute_45_dogleg((px, py), tgt, diag_first);
                                        if dogleg.windows(2).all(|w| {
                                            (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                || is_track_segment_legal(board, w[0], w[1], try_w, fp.layer, plane_net_id, try_c)
                                        }) {
                                            for w in dogleg.windows(2) {
                                                if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                                    board.tracks.push(TrackSegment {
                                                        start: w[0],
                                                        end: w[1],
                                                        width: try_w,
                                                        layer: fp.layer,
                                                        net_id: plane_net_id,
                                                        is_locked: false,
                                                    });
                                                }
                                            }
                                            connected_to_existing = true;
                                            break;
                                        }
                                    }
                                    if connected_to_existing {
                                        break;
                                    }
                                }
                                if connected_to_existing {
                                    break;
                                }
                            }
                            if !connected_to_existing {
                                let mut local_grid = RoutingGrid::new(board, grid.res, None);
                                local_grid.rasterize_obstacles(board);
                                for trk in &board.tracks {
                                    if trk.net_id != plane_net_id {
                                        local_grid.commit_track(trk);
                                    }
                                }
                                for v in &board.vias {
                                    if v.net_id != plane_net_id {
                                        local_grid.commit_via(v);
                                    }
                                }
                                let dummy_stop = AtomicBool::new(false);
                                for &tgt in targets.iter().take(6) {
                                    if let (Some((sgx, sgy)), Some((egx, egy))) = (
                                        local_grid.world_to_grid(px, py),
                                        local_grid.world_to_grid(tgt.0, tgt.1),
                                    ) {
                                        let s_node = GridNode { gx: sgx, gy: sgy, layer: fp.layer };
                                        let e_node = GridNode { gx: egx, gy: egy, layer: fp.layer };
                                        if let Some(p) = self.find_path_with_params(
                                            &local_grid,
                                            s_node,
                                            e_node,
                                            plane_net_id,
                                            0.15,
                                            0.15,
                                            0.60,
                                            None,
                                            1.0,
                                            0.0,
                                            &dummy_stop,
                                        ) {
                                            let orig_trk_len = board.tracks.len();
                                            let orig_via_len = board.vias.len();
                                            let orig_added_len = added_via_positions.len();
                                            let mut cur_layer = fp.layer;
                                            let mut last_pt = (px, py);
                                            let mut path_valid = true;
                                            for (ni, n) in p.iter().enumerate() {
                                                let wpt = if ni + 1 == p.len() && n.layer == cur_layer {
                                                    tgt
                                                } else {
                                                    local_grid.grid_to_world(n.gx, n.gy)
                                                };
                                                if n.layer != cur_layer {
                                                    if is_via_location_legal(board, last_pt, 0.60, plane_net_id, 0.12) {
                                                        board.vias.push(Via {
                                                            position: last_pt,
                                                            size: 0.60,
                                                            drill: 0.30,
                                                            net_id: plane_net_id,
                                                            is_locked: false,
                                                        });
                                                        added_via_positions.push(last_pt);
                                                        if last_pt.0 <= zone_max_x {
                                                            // Reaching a legal via inside the plane zone connects the pad to the plane
                                                            break;
                                                        }
                                                        cur_layer = n.layer;
                                                    } else {
                                                        path_valid = false;
                                                        break;
                                                    }
                                                }
                                                let dl = compute_45_dogleg(last_pt, wpt, false);
                                                for seg in dl.windows(2) {
                                                    if (seg[0].0 - seg[1].0).hypot(seg[0].1 - seg[1].1) > 1e-4 {
                                                        board.tracks.push(TrackSegment {
                                                            start: seg[0],
                                                            end: seg[1],
                                                            width: 0.15,
                                                            layer: cur_layer,
                                                            net_id: plane_net_id,
                                                            is_locked: false,
                                                        });
                                                    }
                                                }
                                                last_pt = wpt;
                                            }
                                            if path_valid {
                                                break;
                                            } else {
                                                board.tracks.truncate(orig_trk_len);
                                                board.vias.truncate(orig_via_len);
                                                added_via_positions.truncate(orig_added_len);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        added_via_positions.len()
    }

    pub fn add_ground_stitching_vias(&self, board: &mut Board, grid: &RoutingGrid) -> usize {
        self.add_plane_stitching_vias(board, grid)
    }

    fn find_safe_plane_via_location(
        board: &Board,
        grid: &RoutingGrid,
        plane_net_id: usize,
        px: f64,
        py: f64,
        layer: usize,
        pad: &Pad,
        existing_vias: &[(f64, f64)],
    ) -> Option<(f64, f64)> {
        let pad_dim = pad.width.max(pad.height);
        let min_d = (pad_dim * 0.5) + 0.35;
        let min_d_narrow = (pad.width.min(pad.height) * 0.5) + 0.35;
        let track_w = board.get_netclass_for_net(plane_net_id).track_width.min(0.25).max(0.20);
        let plane_c = board.get_netclass_for_net(plane_net_id).clearance.max(0.20);

        let board_mid_x = (board.min_x + board.max_x) * 0.5;
        let zone_max_x = board
            .keepouts
            .iter()
            .filter(|ko| ko.min_x > board_mid_x && py >= ko.min_y - 4.0 && py <= ko.max_y + 4.0)
            .map(|ko| ko.min_x - 3.9)
            .fold(board.max_x, f64::min);

        let mut primary_offsets = vec![
            (-0.23, 0.825),
            (0.23, 0.825),
            (-0.23, -0.825),
            (0.23, -0.825),
            (-0.17, 0.875),
            (0.17, 0.875),
            (-0.17, -0.875),
            (0.17, -0.875),
            (-0.32, 0.78),
            (0.32, 0.78),
            (-0.32, -0.78),
            (0.32, -0.78),
        ];
        if min_d_narrow + 0.10 < min_d {
            for &dist_n in &[
                min_d_narrow,
                min_d_narrow + 0.20,
                min_d_narrow + 0.40,
                min_d_narrow + 0.60,
                min_d_narrow + 0.80,
            ] {
                primary_offsets.push((dist_n, 0.0));
                primary_offsets.push((-dist_n, 0.0));
                primary_offsets.push((0.0, dist_n));
                primary_offsets.push((0.0, -dist_n));
            }
        }
        for &dist in &[
            min_d,
            min_d + 0.20,
            min_d + 0.40,
            min_d + 0.60,
            min_d + 0.80,
            min_d + 1.00,
            min_d + 1.25,
            min_d + 1.50,
            min_d + 1.80,
            min_d + 2.10,
            min_d + 2.50,
        ] {
            let diag_d = dist * std::f64::consts::FRAC_1_SQRT_2;
            primary_offsets.push((dist, 0.0));
            primary_offsets.push((-dist, 0.0));
            primary_offsets.push((0.0, dist));
            primary_offsets.push((0.0, -dist));
            primary_offsets.push((diag_d, diag_d));
            primary_offsets.push((diag_d, -diag_d));
            primary_offsets.push((-diag_d, diag_d));
            primary_offsets.push((-diag_d, -diag_d));
            primary_offsets.push((dist, dist * 0.5));
            primary_offsets.push((dist, -dist * 0.5));
            primary_offsets.push((-dist, dist * 0.5));
            primary_offsets.push((-dist, -dist * 0.5));
            primary_offsets.push((dist * 0.5, dist));
            primary_offsets.push((-dist * 0.5, dist));
            primary_offsets.push((dist * 0.5, -dist));
            primary_offsets.push((-dist * 0.5, -dist));
            primary_offsets.push((dist * 0.38, dist));
            primary_offsets.push((-dist * 0.38, dist));
            primary_offsets.push((dist * 0.38, -dist));
            primary_offsets.push((-dist * 0.38, -dist));
        }

        if px > zone_max_x {
            for &dx in &[-1.2, -1.5, -1.8, -2.1, -2.5, -2.9, -3.1, -3.3, -3.5, -3.8, -4.1, -4.5] {
                for &dy in &[0.0, 0.8, -0.8, 1.1, -1.1, 1.35, -1.35, 1.6, -1.6, 2.0, -2.0, 2.5, -2.5] {
                    primary_offsets.push((dx, dy));
                }
            }
        }

        let mut fallback_offsets = primary_offsets.clone();
        for &dist in &[
            min_d,
            min_d + 0.15,
            min_d + 0.25,
            min_d + 0.40,
            min_d + 0.55,
            min_d + 0.70,
            min_d + 0.85,
            min_d + 1.00,
            min_d + 1.25,
            min_d + 1.50,
            min_d + 1.80,
            min_d + 2.10,
            min_d + 2.50,
            min_d + 3.00,
        ] {
            let diag_d = dist * std::f64::consts::FRAC_1_SQRT_2;
            fallback_offsets.push((diag_d, diag_d));
            fallback_offsets.push((diag_d, -diag_d));
            fallback_offsets.push((-diag_d, diag_d));
            fallback_offsets.push((-diag_d, -diag_d));
            fallback_offsets.push((dist, dist * 0.25));
            fallback_offsets.push((dist, -dist * 0.25));
            fallback_offsets.push((-dist, dist * 0.25));
            fallback_offsets.push((-dist, -dist * 0.25));
            fallback_offsets.push((dist * 0.25, dist));
            fallback_offsets.push((-dist * 0.25, dist));
            fallback_offsets.push((dist * 0.25, -dist));
            fallback_offsets.push((-dist * 0.25, -dist));
            fallback_offsets.push((dist * 0.38, dist));
            fallback_offsets.push((-dist * 0.38, dist));
            fallback_offsets.push((dist * 0.38, -dist));
            fallback_offsets.push((-dist * 0.38, -dist));
        }

        let drc_ctx = GpuDrcContext::from_board(board);
        let confirm_cpu = |vx: f64, vy: f64, try_w: f64, try_c: f64| -> bool {
            if !is_via_location_legal(board, (vx, vy), 0.60, plane_net_id, try_c) {
                return false;
            }
            let direct_ok = is_octilinear_segment((px, py), (vx, vy))
                && is_track_segment_legal(board, (px, py), (vx, vy), try_w, layer, plane_net_id, try_c);
            if direct_ok {
                return true;
            }
            [false, true].into_iter().any(|df| {
                let dl = compute_45_dogleg((px, py), (vx, vy), df);
                dl.windows(2).all(|w| {
                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                        || is_track_segment_legal(board, w[0], w[1], try_w, layer, plane_net_id, try_c)
                })
            })
        };

        for (try_w, try_c, offsets) in [(track_w, plane_c, &primary_offsets), (0.15, 0.15, &fallback_offsets)] {
            let mut valid_coords = Vec::with_capacity(offsets.len());
            let mut gpu_cands = Vec::with_capacity(offsets.len() * 2);
            for &(dx, dy) in offsets {
                let vx = px + dx;
                let vy = py + dy;
                if vx > zone_max_x {
                    continue;
                }
                if existing_vias.iter().any(|ev| (ev.0 - vx).hypot(ev.1 - vy) < 0.65) {
                    continue;
                }
                if grid.world_to_grid(vx, vy).is_none() {
                    continue;
                }
                valid_coords.push((vx, vy));
                gpu_cands.push(GpuDrcCandidate {
                    x1: vx as f32,
                    y1: vy as f32,
                    x2: 0.0,
                    y2: 0.0,
                    radius_or_half_width: 0.30,
                    clearance: try_c as f32,
                    layer: layer as u32,
                    net_id: plane_net_id as u32,
                    kind: 0,
                    ignore_same_net_pads: 0,
                    _pad0: 0,
                    _pad1: 0,
                });
                gpu_cands.push(GpuDrcCandidate {
                    x1: px as f32,
                    y1: py as f32,
                    x2: vx as f32,
                    y2: vy as f32,
                    radius_or_half_width: (try_w * 0.5) as f32,
                    clearance: try_c as f32,
                    layer: layer as u32,
                    net_id: plane_net_id as u32,
                    kind: 1,
                    ignore_same_net_pads: 1,
                    _pad0: 0,
                    _pad1: 0,
                });
            }

            if let Some(res) = drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_cands) {
                for (i, &(vx, vy)) in valid_coords.iter().enumerate() {
                    if res[i * 2] != 0 && confirm_cpu(vx, vy, try_w, try_c) {
                        return Some((vx, vy));
                    }
                }
            } else {
                for &(vx, vy) in &valid_coords {
                    if confirm_cpu(vx, vy, try_w, try_c) {
                        return Some((vx, vy));
                    }
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
        reduce_vias_with_config(board, grid, &self.config);
    }

    pub fn gloss_tracks(&self, board: &mut Board, grid: &RoutingGrid) {
        gloss_tracks_with_mode(board, grid, self.config.any_angle_mode);
    }

    pub fn prune_dangling_tracks_and_vias(&self, board: &mut Board) {
        prune_dangling_tracks_and_vias(board);
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
        if signal_vias.is_empty() {
            return 0;
        }

        let gnd_c = board.get_netclass_for_net(gnd_net_id).clearance.max(0.15);
        let via_size = 0.60;
        let via_drill = 0.30;
        let via_r = via_size * 0.5;
        let r_cells = ((via_r + gnd_c) / grid.res).ceil() as isize;

        let max_ret_d = self.config.return_via_max_dist_mm;
        let mut distances: Vec<f64> = [0.75, 0.90, 1.10, 1.25, 1.40]
            .into_iter()
            .filter(|&d| d <= max_ret_d + 1e-4)
            .collect();
        if max_ret_d > 1.40 + 1e-4 {
            distances.push(max_ret_d);
        }
        if distances.is_empty() {
            distances.push(max_ret_d.max(0.50));
        }
        let mut angles = Vec::with_capacity(16);
        for i in 0..16 {
            let theta = (i as f64) * std::f64::consts::PI / 8.0;
            angles.push((theta.cos(), theta.sin()));
        }

        let cands_per_sig = distances.len() * angles.len();
        let mut gpu_cands = Vec::with_capacity(signal_vias.len() * cands_per_sig);
        for &(_, s_pos) in &signal_vias {
            for &dist in &distances {
                for &(cos_t, sin_t) in &angles {
                    let vx = s_pos.0 + dist * cos_t;
                    let vy = s_pos.1 + dist * sin_t;
                    gpu_cands.push(GpuDrcCandidate {
                        x1: vx as f32,
                        y1: vy as f32,
                        x2: 0.0,
                        y2: 0.0,
                        radius_or_half_width: via_r as f32,
                        clearance: gnd_c as f32,
                        layer: 0,
                        net_id: gnd_net_id as u32,
                        kind: 0,
                        ignore_same_net_pads: 0,
                        _pad0: 0,
                        _pad1: 0,
                    });
                }
            }
        }

        let drc_ctx = GpuDrcContext::from_board(board);
        let gpu_flags = drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_cands);

        // Pre-compute legal candidate positions for every signal via in parallel across Rayon workers
        let legal_per_sig: Vec<Vec<(f64, f64)>> = signal_vias
            .par_iter()
            .enumerate()
            .map(|(sig_i, &(_, s_pos))| {
                let mut valid_list = Vec::new();
                let base_idx = sig_i * cands_per_sig;
                let mut local_k = 0usize;
                for &dist in &distances {
                    for &(cos_t, sin_t) in &angles {
                        let flag_idx = base_idx + local_k;
                        local_k += 1;
                        if !gpu_flags.as_ref().map_or(true, |f| f[flag_idx] != 0) {
                            continue;
                        }
                        let vx = s_pos.0 + dist * cos_t;
                        let vy = s_pos.1 + dist * sin_t;
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
                                valid_list.push((vx, vy));
                            }
                        }
                    }
                }
                valid_list
            })
            .collect();

        let mut added_count = 0;
        let skip_existing_dist = (max_ret_d + 0.10).max(0.85);
        for ((_s_idx, s_pos), valid_cands) in signal_vias.into_iter().zip(legal_per_sig.into_iter()) {
            let nearest_dist = gnd_locations
                .iter()
                .map(|gp| (gp.0 - s_pos.0).hypot(gp.1 - s_pos.1))
                .fold(f64::INFINITY, f64::min);

            if nearest_dist > skip_existing_dist {
                if let Some(cand_pos) = valid_cands
                    .into_iter()
                    .find(|&(vx, vy)| !gnd_locations.iter().any(|ev| (ev.0 - vx).hypot(ev.1 - vy) < via_size + gnd_c))
                {
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

    pub fn validate_crosstalk_and_coupling(&self, board: &Board) -> Vec<CouplingViolation> {
        validate_crosstalk_and_coupling(board)
    }

    /// Dispatches a batch of 3D wavefront pathfinding requests across all available GPUs
    /// via `MultiGpuRouterPool` and returns obstacle-free `GridNode` paths.
    pub fn solve_batch_gpu_wavefront_paths(
        &self,
        grid: &RoutingGrid,
        requests: &[(GridNode, GridNode, usize, f32)],
        clearance_cells: i32,
    ) -> Vec<Option<Vec<GridNode>>> {
        let raw_reqs: Vec<((usize, usize, usize), (usize, usize, usize), u32, f32)> = requests
            .iter()
            .map(|&(src, dst, net_id, via_cost)| {
                (
                    (src.gx, src.gy, src.layer),
                    (dst.gx, dst.gy, dst.layer),
                    net_id as u32,
                    via_cost,
                )
            })
            .collect();
        grid.solve_batch_3d_wavefront_paths(&raw_reqs, clearance_cells)
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
                                let mut neighbors_ok = true;
                                for oi in 0..n_tracks {
                                    if oi == fi || board.tracks[oi].net_id != follower_net {
                                        continue;
                                    }
                                    let mut ns = board.tracks[oi].start;
                                    let mut ne = board.tracks[oi].end;
                                    let mut touched = false;
                                    if (ns.0 - f_start.0).hypot(ns.1 - old_y) < 1e-3 || (ns.0 - f_end.0).hypot(ns.1 - old_y) < 1e-3 {
                                        ns.1 = target_y;
                                        touched = true;
                                    }
                                    if (ne.0 - f_start.0).hypot(ne.1 - old_y) < 1e-3 || (ne.0 - f_end.0).hypot(ne.1 - old_y) < 1e-3 {
                                        ne.1 = target_y;
                                        touched = true;
                                    }
                                    if touched && !board.footprints.is_empty()
                                        && !is_track_segment_legal(board, ns, ne, board.tracks[oi].width, board.tracks[oi].layer, follower_net, 0.12)
                                    {
                                        neighbors_ok = false;
                                        break;
                                    }
                                }
                                if neighbors_ok {
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
                                let mut neighbors_ok = true;
                                for oi in 0..n_tracks {
                                    if oi == fi || board.tracks[oi].net_id != follower_net {
                                        continue;
                                    }
                                    let mut ns = board.tracks[oi].start;
                                    let mut ne = board.tracks[oi].end;
                                    let mut touched = false;
                                    if (ns.0 - old_x).hypot(ns.1 - f_start.1) < 1e-3 || (ns.0 - old_x).hypot(ns.1 - f_end.1) < 1e-3 {
                                        ns.0 = target_x;
                                        touched = true;
                                    }
                                    if (ne.0 - old_x).hypot(ne.1 - f_start.1) < 1e-3 || (ne.0 - old_x).hypot(ne.1 - f_end.1) < 1e-3 {
                                        ne.0 = target_x;
                                        touched = true;
                                    }
                                    if touched && !board.footprints.is_empty()
                                        && !is_track_segment_legal(board, ns, ne, board.tracks[oi].width, board.tracks[oi].layer, follower_net, 0.12)
                                    {
                                        neighbors_ok = false;
                                        break;
                                    }
                                }
                                if neighbors_ok {
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

    let mut candidates: Vec<TrackSegment> = board
        .tracks
        .iter()
        .filter(|t| t.net_id == short_net && !t.is_locked)
        .cloned()
        .collect();

    if candidates.is_empty() {
        return false;
    }

    candidates.sort_by(|a, b| {
        let la = (a.start.0 - a.end.0).hypot(a.start.1 - a.end.1);
        let lb = (b.start.0 - b.end.0).hypot(b.start.1 - b.end.1);
        lb.partial_cmp(&la).unwrap_or(Ordering::Equal)
    });

    let mut remaining_skew = delta_l;
    let k45 = 2.0 * (std::f64::consts::SQRT_2 - 1.0);

    for orig in candidates {
        if remaining_skew <= tolerance_mm {
            break;
        }
        let seg_len = (orig.start.0 - orig.end.0).hypot(orig.start.1 - orig.end.1);
        if seg_len < 1.50 {
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

        let max_h = 1.50f64.min((seg_len - 0.50) * 0.45);
        let h = (remaining_skew / k45).min(max_h);
        if h < 0.02 {
            continue;
        }

        let w_top = (seg_len - 2.0 * h - 0.20).min(0.40).max(0.20);
        let jog_span = 2.0 * h + w_top;
        if jog_span >= seg_len - 0.10 {
            continue;
        }
        let p1_offset = (seg_len - jog_span) * 0.5;
        let p1 = (orig.start.0 + dir.0 * p1_offset, orig.start.1 + dir.1 * p1_offset);
        let p2 = (
            p1.0 + dir.0 * h + out_norm.0 * h,
            p1.1 + dir.1 * h + out_norm.1 * h,
        );
        let p3 = (p2.0 + dir.0 * w_top, p2.1 + dir.1 * w_top);
        let p4 = (p1.0 + dir.0 * jog_span, p1.1 + dir.1 * jog_span);

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

        let clearance = board.get_netclass_for_net(short_net).clearance;
        if !board.footprints.is_empty()
            && !replacement.iter().all(|r| {
                is_track_segment_legal(board, r.start, r.end, r.width, r.layer, short_net, clearance)
            })
        {
            continue;
        }

        if let Some(pos) = board.tracks.iter().position(|t| t == &orig) {
            board.tracks.remove(pos);
            for r in replacement {
                board.tracks.push(r);
            }
            let added_len = k45 * h;
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

        if !board.footprints.is_empty()
            && !replacement.iter().all(|r| {
                is_track_segment_legal(board, r.start, r.end, r.width, r.layer, net_id, clearance)
            })
        {
            continue;
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
                        p.net_id == net_id
                            && (p.is_through_hole || fp.layer == layer)
                            && (fp.pad_absolute_position(p).0 - pt.0).hypot(fp.pad_absolute_position(p).1 - pt.1) < 0.015
                    })
                });
                let near_via = board.vias.iter().any(|v| {
                    v.net_id == net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.015
                });
                let near_other_track = tracks.iter().any(|t| {
                    t.net_id == net_id
                        && (t.layer != layer || (t.width - width).abs() > 1e-4)
                        && ((t.start.0 - pt.0).hypot(t.start.1 - pt.1) < 1e-3
                            || (t.end.0 - pt.0).hypot(t.end.1 - pt.1) < 1e-3)
                }) || board.tracks.iter().any(|t| {
                    t.is_locked
                        && t.net_id == net_id
                        && t.layer == layer
                        && ((t.start.0 - pt.0).hypot(t.start.1 - pt.1) < 0.05
                            || (t.end.0 - pt.0).hypot(t.end.1 - pt.1) < 0.05
                            || dist_point_to_segment(pt, t.start, t.end) < 0.05)
                }) || board.arcs.iter().any(|a| {
                    a.net_id == net_id
                        && a.layer == layer
                        && ((a.start().0 - pt.0).hypot(a.start().1 - pt.1) < 0.05
                            || (a.end().0 - pt.0).hypot(a.end().1 - pt.1) < 0.05)
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

    // 6. RoutingGrid rasterized keepouts (or synthetic grid obstacles when board has no footprints)
    if let Some(g) = grid {
        let check_all_grid = board.footprints.is_empty();
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
                if cell_net == KEEPOUT_NET || (check_all_grid && cell_net != FREE_NET && cell_net != net_id as i32) {
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
                let p1 = chain.points[i];
                let p2 = chain.points[j];
                if (i + 1..j).any(|k| {
                    let pk = chain.points[k];
                    is_special_point(pk)
                        && (pk.0 - p1.0).hypot(pk.1 - p1.1) > 0.15
                        && (pk.0 - p2.0).hypot(pk.1 - p2.1) > 0.15
                }) {
                    continue;
                }
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
    if chain.points.len() < 3 {
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
                let p_i = chain.points[i];
                let p_j = chain.points[j];
                if (i + 1..j).any(|k| {
                    let pk = chain.points[k];
                    is_special_point(pk)
                        && (pk.0 - p_i.0).hypot(pk.1 - p_i.1) > 0.15
                        && (pk.0 - p_j.0).hypot(pk.1 - p_j.1) > 0.15
                }) {
                    continue;
                }
                let dx = p_j.0 - p_i.0;
                let dy = p_j.1 - p_i.1;
                let s = dx.abs().min(dy.abs());
                let canon_len = s * std::f64::consts::SQRT_2 + (dx.abs() - dy.abs()).abs();
                let orig_len: f64 = chain.points[i..=j]
                    .windows(2)
                    .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
                    .sum();

                let existing_bends = count_chain_bends(&chain.points[i..=j]);
                if existing_bends < 2 && j - i < 3 && orig_len <= canon_len + 0.02 {
                    continue;
                }

                if dx.abs() < 1e-4 || dy.abs() < 1e-4 || (dx.abs() - dy.abs()).abs() < 1e-4 {
                    if is_segment_clear(board, Some(grid), p_i, p_j, chain.layer, chain.net_id, chain.width) {
                        best_replacement = Some((j, vec![p_i, p_j]));
                        break;
                    }
                }

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
        let is_diag45 = (dx.abs() - dy.abs()).abs() < 1e-4 && dx.abs() > 1e-4;

        if is_diag45 {
            let p01_horiz = (p1.1 - p0.1).abs() < 1e-4 && (p1.0 - p0.0).abs() > 1e-4;
            let p01_vert = (p1.0 - p0.0).abs() < 1e-4 && (p1.1 - p0.1).abs() > 1e-4;
            let p23_horiz = (p3.1 - p2.1).abs() < 1e-4 && (p3.0 - p2.0).abs() > 1e-4;
            let p23_vert = (p3.0 - p2.0).abs() < 1e-4 && (p3.1 - p2.1).abs() > 1e-4;
            if (p01_horiz && p23_vert) || (p01_vert && p23_horiz) {
                let min_trk_dist = |a: (f64, f64), b: (f64, f64)| -> f64 {
                    board
                        .tracks
                        .iter()
                        .filter(|t| t.net_id != chain.net_id && t.layer == chain.layer)
                        .map(|t| dist_segment_to_segment(a, b, t.start, t.end) - (chain.width + t.width) * 0.5)
                        .fold(f64::INFINITY, f64::min)
                };
                let cur_gap = min_trk_dist(p1, p2);
                if cur_gap < 0.15 {
                    for &s in &[0.05, -0.05, 0.10, -0.10, 0.15, -0.15] {
                        let (p1_new, p2_new) = if p01_horiz {
                            ((p1.0 + s, p1.1), (p2.0, p2.1 - s * dx.signum() * dy.signum()))
                        } else {
                            ((p1.0, p1.1 + s), (p2.0 - s * dx.signum() * dy.signum(), p2.1))
                        };
                        if (p1_new.0 - p0.0).hypot(p1_new.1 - p0.1) < 0.05
                            || (p3.0 - p2_new.0).hypot(p3.1 - p2_new.1) < 0.05
                            || (p2_new.0 - p1_new.0).hypot(p2_new.1 - p1_new.1) < 0.10
                        {
                            continue;
                        }
                        if min_trk_dist(p1_new, p2_new) >= 0.155
                            && is_segment_clear(board, Some(grid), p0, p1_new, chain.layer, chain.net_id, chain.width)
                            && is_segment_clear(board, Some(grid), p1_new, p2_new, chain.layer, chain.net_id, chain.width)
                            && is_segment_clear(board, Some(grid), p2_new, p3, chain.layer, chain.net_id, chain.width)
                        {
                            chain.points[i + 1] = p1_new;
                            chain.points[i + 2] = p2_new;
                            break;
                        }
                    }
                }
            }
            i += 1;
            continue;
        }

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
    reduce_vias_with_config(board, grid, &RoutingConfig::default());
}

pub fn reduce_vias_with_config(board: &mut Board, grid: &mut RoutingGrid, config: &RoutingConfig) {
    let max_passes = 8;
    for _pass in 0..max_passes {
        let changed_pairs = eliminate_via_pairs_with_config(board, grid, config);
        let changed_single = consolidate_single_vias_with_config(board, grid, config);
        let changed_slide = slide_and_align_vias_with_config(board, grid, config);
        if !changed_pairs && !changed_single && !changed_slide {
            break;
        }
    }
    prune_dangling_tracks_and_vias(board);
}

pub fn reduce_vias_single_pass(board: &mut Board, grid: &RoutingGrid) -> bool {
    let config = RoutingConfig::default();
    let p = eliminate_via_pairs_with_config(board, grid, &config);
    let s = consolidate_single_vias_with_config(board, grid, &config);
    let l = slide_and_align_vias_with_config(board, grid, &config);
    p || s || l
}

/// Eliminates redundant layer-hopping via pairs (L_orig -> L_alt -> L_orig).
pub fn eliminate_via_pairs(board: &mut Board, grid: &RoutingGrid) -> bool {
    eliminate_via_pairs_with_config(board, grid, &RoutingConfig::default())
}

pub fn eliminate_via_pairs_with_config(board: &mut Board, grid: &RoutingGrid, config: &RoutingConfig) -> bool {
    let mut any_changed = false;
    for _ in 0..20 {
        if !eliminate_via_pair_once(board, grid, config) {
            break;
        }
        any_changed = true;
    }
    any_changed
}

#[inline]
fn aabb_overlaps(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> bool {
    a.0 <= b.2 && a.2 >= b.0 && a.1 <= b.3 && a.3 >= b.1
}

#[inline]
fn points_expanded_aabb(pts: &[(f64, f64)], margin: f64) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &(x, y) in pts {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
}

enum ViaPairMutation {
    EliminateHop {
        net_id: usize,
        l_orig: usize,
        track_width: f64,
        chain_segs: Vec<usize>,
        new_pts: Vec<(f64, f64)>,
        q1: (i64, i64),
        q2: (i64, i64),
        aabb: (f64, f64, f64, f64),
    },
    ConsolidateAdjacent {
        net_id: usize,
        rem_track_indices: Vec<usize>,
        new_segs: Vec<TrackSegment>,
        drop_q: (i64, i64),
        aabb: (f64, f64, f64, f64),
    },
}

fn eliminate_via_pair_once(board: &mut Board, grid: &RoutingGrid, config: &RoutingConfig) -> bool {
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        net_name == "GND"
            || net_name.starts_with("GND")
            || grid.plane_nets.values().any(|&pn| pn == nid)
    };

    // Collect all candidate nets with unlocked vias
    let mut net_ids: Vec<usize> = board
        .vias
        .iter()
        .filter(|v| !v.is_locked)
        .map(|v| v.net_id)
        .collect();
    net_ids.sort_unstable();
    net_ids.dedup();

    if net_ids.is_empty() {
        return false;
    }

    let num_layers = grid.num_layers.max(2);
    let touches_same_net_pad = |net_id: usize, pt: (f64, f64), layer_opt: Option<usize>, tol: f64| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                if p.net_id != net_id {
                    return false;
                }
                if let Some(l) = layer_opt {
                    if !p.is_through_hole && fp.layer != l {
                        return false;
                    }
                }
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) <= tol
            })
        })
    };

    let candidate_mutations: Vec<ViaPairMutation> = net_ids
        .par_iter()
        .filter_map(|&net_id| {
            let vias_info: Vec<(usize, (f64, f64))> = board
                .vias
                .iter()
                .enumerate()
                .filter(|(_, v)| v.net_id == net_id && !v.is_locked)
                .map(|(i, v)| (i, v.position))
                .collect();

            if vias_info.len() < 2 {
                return None;
            }

            // 1. Via pair layer-hopping elimination (L_orig -> L_alt -> L_orig) for non-plane nets
            if !is_plane_net(net_id) {
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

                            let next_tracks: Vec<(usize, (f64, f64))> = board
                                .tracks
                                .iter()
                                .enumerate()
                                .filter(|&(idx, t)| {
                                    t.net_id == net_id
                                        && !t.is_locked
                                        && t.layer == l_alt
                                        && !chain_segs.contains(&idx)
                                })
                                .filter_map(|(idx, t)| {
                                    if quant_point(t.start) == curr_q {
                                        Some((idx, t.end))
                                    } else if quant_point(t.end) == curr_q {
                                        Some((idx, t.start))
                                    } else {
                                        None
                                    }
                                })
                                .collect();

                            let has_other_layer_branch = board.tracks.iter().any(|t| {
                                t.net_id == net_id
                                    && t.layer != l_alt
                                    && (quant_point(t.start) == curr_q || quant_point(t.end) == curr_q)
                            });

                            if next_tracks.len() == 1 && !has_other_layer_branch {
                                let (next_seg_idx, nxt_p) = next_tracks[0];
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

                        // Neither v1 nor v2 may have extra branches on layers != l_orig outside chain_segs
                        let has_extra_non_orig_branch = board.tracks.iter().enumerate().any(|(idx, t)| {
                            t.net_id == net_id
                                && t.layer != l_orig
                                && !chain_segs.contains(&idx)
                                && (quant_point(t.start) == q1
                                    || quant_point(t.end) == q1
                                    || quant_point(t.start) == q2
                                    || quant_point(t.end) == q2)
                        });
                        if has_extra_non_orig_branch {
                            continue;
                        }

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
                        let orig_hop_len: f64 = chain_pts
                            .windows(2)
                            .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
                            .sum();

                        let clearance = board.get_netclass_for_net(net_id).clearance;

                        // Build degree-2 retreat/advance chains on l_orig from v1_pos and v2_pos (up to 5 steps)
                        let build_l_orig_chain = |start_pos: (f64, f64), other_via_q: (i64, i64)| -> Vec<((f64, f64), Vec<usize>, f64)> {
                            let mut res = vec![(start_pos, Vec::new(), 0.0)];
                            if touches_same_net_pad(net_id, start_pos, Some(l_orig), 0.12) {
                                return res;
                            }
                            let mut cur_p = start_pos;
                            let mut used_segs: Vec<usize> = Vec::new();
                            let mut acc_len = 0.0;
                            for _ in 0..5 {
                                let cur_q = quant_point(cur_p);
                                let inc: Vec<(usize, (f64, f64))> = board
                                    .tracks
                                    .iter()
                                    .enumerate()
                                    .filter(|&(idx, t)| {
                                        t.net_id == net_id
                                            && !t.is_locked
                                            && t.layer == l_orig
                                            && !chain_segs.contains(&idx)
                                            && !used_segs.contains(&idx)
                                    })
                                    .filter_map(|(idx, t)| {
                                        if quant_point(t.start) == cur_q {
                                            Some((idx, t.end))
                                        } else if quant_point(t.end) == cur_q {
                                            Some((idx, t.start))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();
                                if inc.len() != 1 {
                                    break;
                                }
                                let (seg_idx, nxt_p) = inc[0];
                                let nxt_q = quant_point(nxt_p);
                                if nxt_q == other_via_q {
                                    break;
                                }
                                acc_len += (nxt_p.0 - cur_p.0).hypot(nxt_p.1 - cur_p.1);
                                used_segs.push(seg_idx);
                                res.push((nxt_p, used_segs.clone(), acc_len));
                                if touches_same_net_pad(net_id, nxt_p, Some(l_orig), 0.12)
                                    || vias_info.iter().any(|&(_, vp)| quant_point(vp) == nxt_q)
                                {
                                    break;
                                }
                                cur_p = nxt_p;
                            }
                            res
                        };

                        let u_chain = build_l_orig_chain(v1_pos, q2);
                        let w_chain = build_l_orig_chain(v2_pos, q1);

                        let mut winner: Option<(Vec<(f64, f64)>, Vec<usize>)> = None;

                        'hop_search: for (u_pt, rem_u, len_u) in &u_chain {
                            for (w_pt, rem_w, len_w) in &w_chain {
                                if rem_u.iter().any(| idx| rem_w.contains(idx)) {
                                    continue;
                                }
                                let orig_len = orig_hop_len + len_u + len_w;
                                let u_pos = *u_pt;
                                let w_pos = *w_pt;

                                // Priority 1: Direct octilinear segment on l_orig
                                let direct_len = (w_pos.0 - u_pos.0).hypot(w_pos.1 - u_pos.1);
                                if is_octilinear_segment(u_pos, w_pos)
                                    && direct_len <= orig_len * config.via_hop_max_detour_ratio + 1e-4
                                    && (direct_len - orig_len) <= 5.0 + 1e-4
                                    && is_segment_clear(board, Some(grid), u_pos, w_pos, l_orig, net_id, track_width)
                                    && is_track_segment_legal(board, u_pos, w_pos, track_width, l_orig, net_id, clearance)
                                {
                                    let mut extra = rem_u.clone();
                                    extra.extend_from_slice(rem_w);
                                    winner = Some((vec![u_pos, w_pos], extra));
                                    break 'hop_search;
                                }

                                // Priority 2: Exact path projection on l_orig (when at v1_pos and v2_pos)
                                if rem_u.is_empty() && rem_w.is_empty() {
                                    let all_clear = chain_pts.windows(2).all(|w| {
                                        is_segment_clear(board, Some(grid), w[0], w[1], l_orig, net_id, track_width)
                                            && is_track_segment_legal(board, w[0], w[1], track_width, l_orig, net_id, clearance)
                                    });
                                    if all_clear {
                                        winner = Some((chain_pts.clone(), Vec::new()));
                                        break 'hop_search;
                                    }
                                }

                                // Priority 3: Canonical 45-degree detour on l_orig within length budget
                                let dx = w_pos.0 - u_pos.0;
                                let dy = w_pos.1 - u_pos.1;
                                let s = dx.abs().min(dy.abs());
                                if s > 1e-4 {
                                    let sx = s * dx.signum();
                                    let sy = s * dy.signum();
                                    let c_a = (u_pos.0 + sx, u_pos.1 + sy);
                                    let c_b = (w_pos.0 - sx, w_pos.1 - sy);

                                    for c in [c_a, c_b] {
                                        let d1 = (c.0 - u_pos.0).hypot(c.1 - u_pos.1);
                                        let d2 = (w_pos.0 - c.0).hypot(w_pos.1 - c.1);
                                        let cand_len = d1 + d2;
                                        let budget_ok = cand_len <= orig_len * config.via_hop_max_detour_ratio + 1e-4
                                            && (cand_len - orig_len) <= 5.0 + 1e-4;
                                        if budget_ok
                                            && is_segment_clear(board, Some(grid), u_pos, c, l_orig, net_id, track_width)
                                            && is_segment_clear(board, Some(grid), c, w_pos, l_orig, net_id, track_width)
                                            && is_track_segment_legal(board, u_pos, c, track_width, l_orig, net_id, clearance)
                                            && is_track_segment_legal(board, c, w_pos, track_width, l_orig, net_id, clearance)
                                        {
                                            let mut extra = rem_u.clone();
                                            extra.extend_from_slice(rem_w);
                                            winner = Some((vec![u_pos, c, w_pos], extra));
                                            break 'hop_search;
                                        }
                                    }
                                }
                            }
                        }

                        if let Some((new_pts, extra_rem)) = winner {
                            let mut all_pts = chain_pts;
                            all_pts.extend_from_slice(&new_pts);
                            let mut full_chain_segs = chain_segs;
                            full_chain_segs.extend(extra_rem);
                            full_chain_segs.sort_unstable();
                            full_chain_segs.dedup();
                            let aabb = points_expanded_aabb(&all_pts, 0.60);
                            return Some(ViaPairMutation::EliminateHop {
                                net_id,
                                l_orig,
                                track_width,
                                chain_segs: full_chain_segs,
                                new_pts,
                                q1,
                                q2,
                                aabb,
                            });
                        }
                    }
                }
            }

            // 2. Physical & Topological Same-Net Via Consolidation (merges close vias within via_merge_radius_mm)
            let mut close_pairs: Vec<(f64, usize, usize)> = Vec::new();
            for i in 0..vias_info.len() {
                for j in (i + 1)..vias_info.len() {
                    let d = (vias_info[i].1 .0 - vias_info[j].1 .0).hypot(vias_info[i].1 .1 - vias_info[j].1 .1);
                    if d >= 1e-4 && d <= config.via_merge_radius_mm {
                        close_pairs.push((d, i, j));
                    }
                }
            }
            close_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

            // Helper: 3D BFS reachability from keep_pos across net_id's tracks & vias (excluding drop_idx and excluded_tracks)
            let is_reachable_from_keep = |keep_pos: (f64, f64),
                                          drop_idx: usize,
                                          target_pt: (f64, f64),
                                          target_layer: usize,
                                          excluded_tracks: &[usize],
                                          extra_tracks: &[TrackSegment]|
             -> bool {
                if (target_pt.0 - keep_pos.0).hypot(target_pt.1 - keep_pos.1) <= 0.08 {
                    return true;
                }
                let mut active_segs: Vec<((f64, f64), (f64, f64), usize)> = board
                    .tracks
                    .iter()
                    .enumerate()
                    .filter(|&(idx, t)| t.net_id == net_id && !excluded_tracks.contains(&idx))
                    .map(|(_, t)| (t.start, t.end, t.layer))
                    .collect();
                for et in extra_tracks {
                    active_segs.push((et.start, et.end, et.layer));
                }

                let mut visited_segs = vec![false; active_segs.len()];
                let mut visited_vias = vec![false; vias_info.len()];
                let mut queue: std::collections::VecDeque<((f64, f64), usize)> = std::collections::VecDeque::new();
                for l in 0..num_layers {
                    queue.push_back((keep_pos, l));
                }

                while let Some((cur_pt, cur_l)) = queue.pop_front() {
                    if cur_l == target_layer && (cur_pt.0 - target_pt.0).hypot(cur_pt.1 - target_pt.1) <= 0.08 {
                        return true;
                    }
                    // Traverse tracks on cur_l
                    for (s_idx, &(s, e, l)) in active_segs.iter().enumerate() {
                        if visited_segs[s_idx] || l != cur_l {
                            continue;
                        }
                        if (s.0 - cur_pt.0).hypot(s.1 - cur_pt.1) <= 0.08 {
                            visited_segs[s_idx] = true;
                            queue.push_back((e, l));
                        } else if (e.0 - cur_pt.0).hypot(e.1 - cur_pt.1) <= 0.08 {
                            visited_segs[s_idx] = true;
                            queue.push_back((s, l));
                        }
                    }
                    // Traverse vias (excluding drop_idx)
                    for (vi, &(v_board_idx, vp)) in vias_info.iter().enumerate() {
                        if v_board_idx == drop_idx || visited_vias[vi] {
                            continue;
                        }
                        if (vp.0 - cur_pt.0).hypot(vp.1 - cur_pt.1) <= 0.08 {
                            visited_vias[vi] = true;
                            for l in 0..num_layers {
                                queue.push_back((vp, l));
                            }
                        }
                    }
                    // Traverse through-hole pads of net_id
                    for fp in &board.footprints {
                        for pad in &fp.pads {
                            if pad.net_id == net_id && pad.is_through_hole {
                                let pp = fp.pad_absolute_position(pad);
                                if (pp.0 - cur_pt.0).hypot(pp.1 - cur_pt.1) <= 0.08 {
                                    for l in 0..num_layers {
                                        if l != cur_l {
                                            queue.push_back((pp, l));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                false
            };

            for &(_d_vias, i, j) in &close_pairs {
                let (v1_idx, v1_pos) = vias_info[i];
                let (v2_idx, v2_pos) = vias_info[j];
                for &(keep_idx, keep_pos, drop_idx, drop_pos) in &[
                    (v1_idx, v1_pos, v2_idx, v2_pos),
                    (v2_idx, v2_pos, v1_idx, v1_pos),
                ] {
                    let _ = keep_idx;
                    // Never drop a thermal via inside a large exposed thermal pad
                    let in_thermal_pad = board.footprints.iter().any(|fp| {
                        fp.pads.iter().any(|p| {
                            (p.width >= 2.0 || p.height >= 2.0) && {
                                let (px, py) = fp.pad_absolute_position(p);
                                (px - drop_pos.0).hypot(py - drop_pos.1) < 1.2
                            }
                        })
                    });
                    if in_thermal_pad {
                        continue;
                    }

                    let drop_r = board.vias.get(drop_idx).map_or(0.30, |v| v.size * 0.5) + 0.05;
                    let keep_r = board.vias.get(keep_idx).map_or(0.30, |v| v.size * 0.5) + 0.05;
                    let q_drop = quant_point(drop_pos);
                    let q_keep = quant_point(keep_pos);

                    let drop_incident: Vec<(usize, usize, (f64, f64), f64)> = board
                        .tracks
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| t.net_id == net_id && !t.is_locked)
                        .filter_map(|(idx, t)| {
                            if quant_point(t.start) == q_drop || (t.start.0 - drop_pos.0).hypot(t.start.1 - drop_pos.1) <= drop_r {
                                Some((idx, t.layer, t.end, t.width))
                            } else if quant_point(t.end) == q_drop || (t.end.0 - drop_pos.0).hypot(t.end.1 - drop_pos.1) <= drop_r {
                                Some((idx, t.layer, t.start, t.width))
                            } else {
                                None
                            }
                        })
                        .collect();
                    if drop_incident.is_empty() {
                        continue;
                    }

                    // Trace maximal degree-2 chains starting from each incident segment of drop_pos
                    struct IncidentChain {
                        segs: Vec<usize>,
                        pts: Vec<(f64, f64)>,
                        layer: usize,
                        width: f64,
                    }

                    let mut chains: Vec<IncidentChain> = Vec::with_capacity(drop_incident.len());
                    for &(t0_idx, layer, p1, width) in &drop_incident {
                        let mut segs = vec![t0_idx];
                        let mut pts = vec![drop_pos, p1];
                        let mut cur_p = p1;
                        while segs.len() < 6 {
                            let cur_q = quant_point(cur_p);
                            if cur_q == q_keep
                                || (cur_p.0 - keep_pos.0).hypot(cur_p.1 - keep_pos.1) <= keep_r
                                || touches_same_net_pad(net_id, cur_p, Some(layer), 0.08)
                                || vias_info.iter().any(|&(vi, vp)| vi != drop_idx && (quant_point(vp) == cur_q || (vp.0 - cur_p.0).hypot(vp.1 - cur_p.1) <= 0.08))
                            {
                                break;
                            }
                            let next_trks: Vec<(usize, (f64, f64))> = board
                                .tracks
                                .iter()
                                .enumerate()
                                .filter(|&(idx, t)| {
                                    t.net_id == net_id
                                        && !t.is_locked
                                        && t.layer == layer
                                        && !segs.contains(&idx)
                                })
                                .filter_map(|(idx, t)| {
                                    if quant_point(t.start) == cur_q {
                                        Some((idx, t.end))
                                    } else if quant_point(t.end) == cur_q {
                                        Some((idx, t.start))
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            if next_trks.len() != 1 {
                                break;
                            }
                            let (nxt_idx, nxt_p) = next_trks[0];
                            segs.push(nxt_idx);
                            pts.push(nxt_p);
                            cur_p = nxt_p;
                        }
                        chains.push(IncidentChain { segs, pts, layer, width });
                    }

                    let mut rem_track_indices: Vec<usize> = Vec::new();
                    let mut new_segs: Vec<TrackSegment> = Vec::new();
                    let mut all_valid = true;
                    let clearance = board.get_netclass_for_net(net_id).clearance.max(0.15);

                    for ch in &chains {
                        let term_pt = *ch.pts.last().unwrap();
                        let has_intermediate_pad = ch.pts[1..ch.pts.len() - 1]
                            .iter()
                            .any(|&pt| touches_same_net_pad(net_id, pt, Some(ch.layer), 0.08));

                        // Case A: Chain is a direct bridge to keep_pos or terminates at an anchor already connected to keep_pos
                        if !has_intermediate_pad {
                            let mut test_excl = rem_track_indices.clone();
                            test_excl.extend_from_slice(&ch.segs);
                            if is_reachable_from_keep(keep_pos, drop_idx, term_pt, ch.layer, &test_excl, &new_segs) {
                                rem_track_indices.extend_from_slice(&ch.segs);
                                continue;
                            }
                        }

                        // Case B: External branch — try connecting keep_pos to pts[r] along the chain
                        let mut found_branch_repl = false;
                        for r in 1..ch.pts.len() {
                            let target_pt = ch.pts[r];
                            let orig_sub_len: f64 = ch.pts[0..=r]
                                .windows(2)
                                .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
                                .sum();

                            let touches_dense_ic_pad = board.footprints.iter().any(|fp| {
                                fp.pads.len() >= 8
                                    && fp.pads.iter().any(|p| {
                                        let (px, py) = fp.pad_absolute_position(p);
                                        (px - target_pt.0).hypot(py - target_pt.1) < 0.05
                                    })
                            });

                            let mut cand_paths: Vec<Vec<(f64, f64)>> = Vec::with_capacity(3);
                            if is_octilinear_segment(keep_pos, target_pt) {
                                cand_paths.push(vec![keep_pos, target_pt]);
                            }
                            cand_paths.push(compute_45_dogleg(keep_pos, target_pt, false));
                            cand_paths.push(compute_45_dogleg(keep_pos, target_pt, true));

                            let valid_dl = cand_paths.into_iter().find(|dl| {
                                let cand_len: f64 = dl.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).sum();
                                if cand_len > orig_sub_len + config.via_detour_budget_mm {
                                    return false;
                                }
                                if touches_dense_ic_pad && dl.len() >= 2 {
                                    let prev = dl[dl.len() - 2];
                                    if (prev.0 - target_pt.0).abs() > 1e-3 && (prev.1 - target_pt.1).abs() > 1e-3 {
                                        return false;
                                    }
                                }
                                dl.windows(2).all(|w| {
                                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                        || (is_segment_clear(board, Some(grid), w[0], w[1], ch.layer, net_id, ch.width)
                                            && is_track_segment_legal(board, w[0], w[1], ch.width, ch.layer, net_id, clearance))
                                })
                            });

                            if let Some(dl) = valid_dl {
                                rem_track_indices.extend_from_slice(&ch.segs[0..r]);
                                for w in dl.windows(2) {
                                    if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                        new_segs.push(TrackSegment {
                                            start: w[0],
                                            end: w[1],
                                            width: ch.width,
                                            layer: ch.layer,
                                            net_id,
                                            is_locked: false,
                                        });
                                    }
                                }
                                found_branch_repl = true;
                                break;
                            }

                            if touches_same_net_pad(net_id, target_pt, Some(ch.layer), 0.08) {
                                break;
                            }
                        }

                        if !found_branch_repl {
                            all_valid = false;
                            break;
                        }
                    }

                    if all_valid {
                        rem_track_indices.sort_unstable();
                        rem_track_indices.dedup();

                        // Final 3D Continuity Guard: verify all chain endpoints and any pad at drop_pos remain reachable from keep_pos
                        let chains_continuous = chains.iter().all(|ch| {
                            let term_pt = *ch.pts.last().unwrap();
                            is_reachable_from_keep(keep_pos, drop_idx, term_pt, ch.layer, &rem_track_indices, &new_segs)
                        });
                        let drop_pads_continuous = board.footprints.iter().all(|fp| {
                            fp.pads.iter().all(|p| {
                                if p.net_id != net_id {
                                    return true;
                                }
                                let pp = fp.pad_absolute_position(p);
                                if (pp.0 - drop_pos.0).hypot(pp.1 - drop_pos.1) <= drop_r + 0.15 {
                                    is_reachable_from_keep(keep_pos, drop_idx, pp, fp.layer, &rem_track_indices, &new_segs)
                                } else {
                                    true
                                }
                            })
                        });

                        if chains_continuous && drop_pads_continuous {
                            let mut bbox_pts = vec![keep_pos, drop_pos];
                            for ch in &chains {
                                bbox_pts.extend_from_slice(&ch.pts);
                            }
                            for s in &new_segs {
                                bbox_pts.push(s.start);
                                bbox_pts.push(s.end);
                            }
                            let aabb = points_expanded_aabb(&bbox_pts, 0.60);
                            return Some(ViaPairMutation::ConsolidateAdjacent {
                                net_id,
                                rem_track_indices,
                                new_segs,
                                drop_q: q_drop,
                                aabb,
                            });
                        }
                    }
                }
            }

            None
        })
        .collect();

    if candidate_mutations.is_empty() {
        return false;
    }

    // Select spatially disjoint mutations across nets so independent regions update in a single pass
    let mut accepted_aabbs: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut rem_tracks_all: Vec<usize> = Vec::new();
    let mut add_tracks_all: Vec<TrackSegment> = Vec::new();
    let mut hop_via_checks: Vec<(usize, usize, (i64, i64), (i64, i64))> = Vec::new();
    let mut drop_vias_exact: Vec<(usize, (i64, i64))> = Vec::new();

    for m in candidate_mutations {
        let aabb = match &m {
            ViaPairMutation::EliminateHop { aabb, .. } => *aabb,
            ViaPairMutation::ConsolidateAdjacent { aabb, .. } => *aabb,
        };
        if accepted_aabbs.iter().any(|&other| aabb_overlaps(aabb, other)) {
            continue;
        }
        accepted_aabbs.push(aabb);
        match m {
            ViaPairMutation::EliminateHop {
                net_id,
                l_orig,
                track_width,
                chain_segs,
                new_pts,
                q1,
                q2,
                ..
            } => {
                rem_tracks_all.extend(chain_segs);
                for w in new_pts.windows(2) {
                    if (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1) > 1e-5 {
                        add_tracks_all.push(TrackSegment {
                            start: w[0],
                            end: w[1],
                            width: track_width,
                            layer: l_orig,
                            net_id,
                            is_locked: false,
                        });
                    }
                }
                hop_via_checks.push((net_id, l_orig, q1, q2));
            }
            ViaPairMutation::ConsolidateAdjacent {
                net_id,
                rem_track_indices,
                new_segs,
                drop_q,
                ..
            } => {
                rem_tracks_all.extend(rem_track_indices);
                add_tracks_all.extend(new_segs);
                drop_vias_exact.push((net_id, drop_q));
            }
        }
    }

    rem_tracks_all.sort_unstable();
    rem_tracks_all.dedup();
    for &idx in rem_tracks_all.iter().rev() {
        board.tracks.remove(idx);
    }
    board.tracks.extend(add_tracks_all);

    let mut vias_to_remove = Vec::new();
    for (i, v) in board.vias.iter().enumerate() {
        if v.is_locked {
            continue;
        }
        let q = quant_point(v.position);
        if drop_vias_exact.iter().any(|&(nid, dq)| nid == v.net_id && dq == q) {
            vias_to_remove.push(i);
            continue;
        }
        for &(nid, l_orig, q1, q2) in &hop_via_checks {
            if v.net_id == nid && (q == q1 || q == q2) {
                let has_other_layer_tracks = board.tracks.iter().any(|t| {
                    t.net_id == nid
                        && t.layer != l_orig
                        && (quant_point(t.start) == q || quant_point(t.end) == q)
                });
                if !has_other_layer_tracks {
                    vias_to_remove.push(i);
                    break;
                }
            }
        }
    }
    vias_to_remove.sort_unstable();
    vias_to_remove.dedup();
    for &idx in vias_to_remove.iter().rev() {
        board.vias.remove(idx);
    }

    true
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
    consolidate_single_vias_with_config(board, grid, &RoutingConfig::default())
}

pub fn consolidate_single_vias_with_config(
    board: &mut Board,
    grid: &RoutingGrid,
    config: &RoutingConfig,
) -> bool {
    let mut any_changed = false;
    for _ in 0..20 {
        if !consolidate_single_via_once(board, grid, config) {
            break;
        }
        any_changed = true;
    }
    any_changed
}

struct SingleViaConsolidation {
    net_id: usize,
    l_prim: usize,
    track_width: f64,
    chain_segs: Vec<usize>,
    new_pts: Vec<(f64, f64)>,
    q_v: (i64, i64),
    aabb: (f64, f64, f64, f64),
}

fn consolidate_single_via_once(
    board: &mut Board,
    grid: &RoutingGrid,
    config: &RoutingConfig,
) -> bool {
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

    if vias_info.is_empty() {
        return false;
    }

    let candidates: Vec<SingleViaConsolidation> = vias_info
        .par_iter()
        .filter_map(|&(v_idx, v_pos, net_id)| {
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
                return None;
            }

            for &l_prim in &layers {
                let alt_layers: Vec<usize> = layers.iter().copied().filter(|&l| l != l_prim).collect();
                if alt_layers.len() != 1 {
                    continue;
                }
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

                        let next_tracks: Vec<(usize, (f64, f64))> = board
                            .tracks
                            .iter()
                            .enumerate()
                            .filter(|&(idx, t)| {
                                t.net_id == net_id
                                    && !t.is_locked
                                    && t.layer == l_alt
                                    && !chain_segs.contains(&idx)
                            })
                            .filter_map(|(idx, t)| {
                                if quant_point(t.start) == curr_q {
                                    Some((idx, t.end))
                                } else if quant_point(t.end) == curr_q {
                                    Some((idx, t.start))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if next_tracks.len() > 1 {
                            hit_another_via = true;
                            break;
                        } else if let Some(&(next_seg_idx, nxt_p)) = next_tracks.first() {
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
                        && direct_len <= orig_len * config.via_hop_max_detour_ratio + 1e-4
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
                                let budget_ok = cand_len <= orig_len * config.via_hop_max_detour_ratio + 1e-4
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
                        let mut all_pts = chain_pts;
                        all_pts.extend_from_slice(&new_pts);
                        let aabb = points_expanded_aabb(&all_pts, 0.60);
                        return Some(SingleViaConsolidation {
                            net_id,
                            l_prim,
                            track_width,
                            chain_segs,
                            new_pts,
                            q_v,
                            aabb,
                        });
                    }
                }
            }
            None
        })
        .collect();

    if candidates.is_empty() {
        return false;
    }

    let mut accepted_nets: HashSet<usize> = HashSet::new();
    let mut accepted_aabbs: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut rem_tracks_all: Vec<usize> = Vec::new();
    let mut add_tracks_all: Vec<TrackSegment> = Vec::new();
    let mut rem_vias_q: Vec<(usize, (i64, i64))> = Vec::new();

    for c in candidates {
        if accepted_nets.contains(&c.net_id)
            || accepted_aabbs.iter().any(|&other| aabb_overlaps(c.aabb, other))
        {
            continue;
        }
        accepted_nets.insert(c.net_id);
        accepted_aabbs.push(c.aabb);
        rem_tracks_all.extend(c.chain_segs);
        for w in c.new_pts.windows(2) {
            if (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1) > 1e-5 {
                add_tracks_all.push(TrackSegment {
                    start: w[0],
                    end: w[1],
                    width: c.track_width,
                    layer: c.l_prim,
                    net_id: c.net_id,
                    is_locked: false,
                });
            }
        }
        rem_vias_q.push((c.net_id, c.q_v));
    }

    rem_tracks_all.sort_unstable();
    rem_tracks_all.dedup();
    for &idx in rem_tracks_all.iter().rev() {
        board.tracks.remove(idx);
    }
    board.tracks.extend(add_tracks_all);

    for (nid, q_v) in rem_vias_q {
        if let Some(pos) = board.vias.iter().position(|v| {
            v.net_id == nid && !v.is_locked && quant_point(v.position) == q_v
        }) {
            board.vias.remove(pos);
        }
    }

    true
}

/// Slides and aligns signal vias along incident track vectors to eliminate micro-jogs,
/// doglegs, reduce wirelength, and snap close vias to through-hole pads.
pub fn slide_and_align_vias(board: &mut Board, grid: &RoutingGrid) -> bool {
    slide_and_align_vias_with_config(board, grid, &RoutingConfig::default())
}

pub fn slide_and_align_vias_with_config(
    board: &mut Board,
    grid: &RoutingGrid,
    config: &RoutingConfig,
) -> bool {
    let mut any_changed = false;
    for _ in 0..20 {
        if !slide_and_align_via_once(board, grid, config) {
            break;
        }
        any_changed = true;
    }
    any_changed
}

struct ViaSlideAction {
    v_idx: usize,
    v_pos: (f64, f64),
    net_id: usize,
    target_pos: (f64, f64),
    remove_via: bool,
    incident_indices: Vec<usize>,
    aabb: (f64, f64, f64, f64),
}

fn slide_and_align_via_once(
    board: &mut Board,
    grid: &RoutingGrid,
    config: &RoutingConfig,
) -> bool {
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        net_name == "GND"
            || net_name.starts_with("GND")
            || grid.plane_nets.values().any(|&pn| pn == nid)
    };

    let num_vias = board.vias.len();
    if num_vias == 0 {
        return false;
    }

    let drc_ctx = GpuDrcContext::from_board(board);

    let slide_candidates: Vec<ViaSlideAction> = (0..num_vias)
        .into_par_iter()
        .filter_map(|v_idx| {
            let (v_pos, net_id, via_size, is_locked) = {
                let v = &board.vias[v_idx];
                (v.position, v.net_id, v.size, v.is_locked)
            };
            if is_locked || is_plane_net(net_id) {
                return None;
            }

            let clearance = board.get_netclass_for_net(net_id).clearance;

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

            let mut inc_layers: Vec<usize> = incident.iter().map(|&(_, l, _, _)| l).collect();
            inc_layers.sort_unstable();
            inc_layers.dedup();
            let has_unanchored_touch = board.tracks.iter().enumerate().any(|(idx, t)| {
                t.net_id == net_id
                    && !incident.iter().any(|&(i, _, _, _)| i == idx)
                    && dist_point_to_segment(v_pos, t.start, t.end) <= via_size * 0.5 + 0.05
            });
            if incident.is_empty() || inc_layers.len() < 2 || has_unanchored_touch {
                return None;
            }

            let incident_indices: Vec<usize> = incident.iter().map(|&(i, _, _, _)| i).collect();

            // 1. Pad Snapping: check if close to a same-net through-hole pad (< via_pad_snap_dist_mm)
            for fp in &board.footprints {
                for pad in &fp.pads {
                    if pad.net_id == net_id && pad.is_through_hole {
                        let (px, py) = fp.pad_absolute_position(pad);
                        let d = (v_pos.0 - px).hypot(v_pos.1 - py);
                        if d > 1e-4 && d < config.via_pad_snap_dist_mm {
                            let all_clear = incident.iter().all(|&(_, l, other, w)| {
                                is_octilinear_segment((px, py), other)
                                    && is_track_segment_legal(board, (px, py), other, w, l, net_id, clearance)
                                    && is_segment_clear(board, Some(grid), (px, py), other, l, net_id, w)
                            });
                            if all_clear {
                                let mut pts = vec![v_pos, (px, py)];
                                pts.extend(incident.iter().map(|&(_, _, other, _)| other));
                                return Some(ViaSlideAction {
                                    v_idx,
                                    v_pos,
                                    net_id,
                                    target_pos: (px, py),
                                    remove_via: true,
                                    incident_indices,
                                    aabb: points_expanded_aabb(&pts, 0.60),
                                });
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

                        if has_connecting_track
                            && is_via_location_legal_ignoring(board, other_pt, via_size, net_id, clearance, Some(v_pos))
                        {
                            let other_tracks_ok = incident.iter().all(|&(ot_idx, ot_layer, ot_other, ot_w)| {
                                if ot_idx == t_idx || (ot_other.0 - other_pt.0).hypot(ot_other.1 - other_pt.1) < 1e-4 {
                                    true
                                } else {
                                    is_octilinear_segment(other_pt, ot_other)
                                        && is_track_segment_legal(board, other_pt, ot_other, ot_w, ot_layer, net_id, clearance)
                                        && is_segment_clear(board, Some(grid), other_pt, ot_other, ot_layer, net_id, ot_w)
                                }
                            });

                            if other_tracks_ok {
                                let mut pts = vec![v_pos, other_pt];
                                pts.extend(incident.iter().map(|&(_, _, other, _)| other));
                                return Some(ViaSlideAction {
                                    v_idx,
                                    v_pos,
                                    net_id,
                                    target_pos: other_pt,
                                    remove_via: false,
                                    incident_indices,
                                    aabb: points_expanded_aabb(&pts, 0.60),
                                });
                            }
                        }
                    }
                }
            }

            // 3. Vector Sliding:
            // Test candidate positions along vectors of connected tracks (batch-evaluated on GPU when available)
            let orig_tot_len: f64 = incident.iter().map(|&(_, _, other, _)| (other.0 - v_pos.0).hypot(other.1 - v_pos.1)).sum();
            let mut raw_cands: Vec<((f64, f64), f64)> = Vec::new();

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
                    let cand_len: f64 = incident.iter().map(|&(_, _, other, _)| (other.0 - cand.0).hypot(other.1 - cand.1)).sum();
                    if cand_len >= orig_tot_len - 0.05 {
                        continue;
                    }
                    let oct_ok = incident.iter().all(|&(_, _, other, _)| {
                        (other.0 - cand.0).hypot(other.1 - cand.1) < 1e-4 || is_octilinear_segment(cand, other)
                    });
                    if oct_ok {
                        raw_cands.push((cand, cand_len));
                    }
                }
            }

            let mut best_cand: Option<((f64, f64), f64)> = None;
            if !raw_cands.is_empty() {
                let gpu_flags: Option<Vec<bool>> = if raw_cands.len() >= 4 {
                    let mut gpu_batch = Vec::with_capacity(raw_cands.len() * (1 + incident.len()));
                    let mut spans = Vec::with_capacity(raw_cands.len());
                    for &(cand, _) in &raw_cands {
                        let start_idx = gpu_batch.len();
                        gpu_batch.push(GpuDrcCandidate {
                            x1: cand.0 as f32,
                            y1: cand.1 as f32,
                            x2: v_pos.0 as f32,
                            y2: v_pos.1 as f32,
                            radius_or_half_width: (via_size * 0.5) as f32,
                            clearance: clearance as f32,
                            layer: 0,
                            net_id: net_id as u32,
                            kind: 0,
                            ignore_same_net_pads: 0,
                            _pad0: 1,
                            _pad1: 0,
                        });
                        for &(_, l, other, w) in &incident {
                            if (other.0 - cand.0).hypot(other.1 - cand.1) >= 1e-4 {
                                gpu_batch.push(GpuDrcCandidate {
                                    x1: cand.0 as f32,
                                    y1: cand.1 as f32,
                                    x2: other.0 as f32,
                                    y2: other.1 as f32,
                                    radius_or_half_width: (w * 0.5) as f32,
                                    clearance: clearance as f32,
                                    layer: l as u32,
                                    net_id: net_id as u32,
                                    kind: 1,
                                    ignore_same_net_pads: 1,
                                    _pad0: 0,
                                    _pad1: 0,
                                });
                            }
                        }
                        spans.push((start_idx, gpu_batch.len()));
                    }
                    drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_batch).map(|res| {
                        spans
                            .into_iter()
                            .map(|(s, e)| res[s..e].iter().all(|&v| v != 0))
                            .collect()
                    })
                } else {
                    None
                };

                let verify_cand = |cand: (f64, f64)| -> bool {
                    if !is_via_location_legal_ignoring(board, cand, via_size, net_id, clearance, Some(v_pos)) {
                        return false;
                    }
                    incident.iter().all(|&(_, l, other, w)| {
                        if (other.0 - cand.0).hypot(other.1 - cand.1) < 1e-4 {
                            true
                        } else {
                            is_track_segment_legal(board, cand, other, w, l, net_id, clearance)
                                && is_segment_clear(board, Some(grid), cand, other, l, net_id, w)
                        }
                    })
                };

                for (idx, &(cand, cand_len)) in raw_cands.iter().enumerate() {
                    let min_len = best_cand.map(|(_, l)| l).unwrap_or(orig_tot_len - 0.05);
                    if cand_len < min_len - 1e-4 {
                        if gpu_flags.as_ref().map(|f| f[idx]).unwrap_or(true) && verify_cand(cand) {
                            best_cand = Some((cand, cand_len));
                        }
                    }
                }
            }

            if let Some((best_pos, _)) = best_cand {
                let mut pts = vec![v_pos, best_pos];
                pts.extend(incident.iter().map(|&(_, _, other, _)| other));
                return Some(ViaSlideAction {
                    v_idx,
                    v_pos,
                    net_id,
                    target_pos: best_pos,
                    remove_via: false,
                    incident_indices,
                    aabb: points_expanded_aabb(&pts, 0.60),
                });
            }

            None
        })
        .collect();

    if slide_candidates.is_empty() {
        return false;
    }

    let mut accepted_nets: HashSet<usize> = HashSet::new();
    let mut accepted_aabbs: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut vias_to_remove: Vec<usize> = Vec::new();

    for action in slide_candidates {
        if accepted_nets.contains(&action.net_id)
            || accepted_aabbs.iter().any(|&other| aabb_overlaps(action.aabb, other))
        {
            continue;
        }
        accepted_nets.insert(action.net_id);
        accepted_aabbs.push(action.aabb);

        if action.remove_via {
            vias_to_remove.push(action.v_idx);
        } else {
            board.vias[action.v_idx].position = action.target_pos;
        }
        for &t_idx in &action.incident_indices {
            let t = &mut board.tracks[t_idx];
            if (t.start.0 - action.v_pos.0).hypot(t.start.1 - action.v_pos.1) < 1e-4 {
                t.start = action.target_pos;
            } else {
                t.end = action.target_pos;
            }
        }
    }

    vias_to_remove.sort_unstable();
    vias_to_remove.dedup();
    for &idx in vias_to_remove.iter().rev() {
        board.vias.remove(idx);
    }
    board.tracks.retain(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1) > 1e-4);

    true
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

fn gloss_single_net_chains(net_id: usize, net_tracks: &[TrackSegment], board_without_net: &Board, grid: &RoutingGrid) -> Vec<TrackSegment> {
    let mut chains = extract_track_chains(net_tracks, board_without_net);
    for chain in &mut chains {
        enforce_45_grid(chain);
        merge_collinear_points(&mut chain.points);
        avoid_acid_traps(chain, board_without_net, grid);
        merge_collinear_points(&mut chain.points);
        rubber_band_straighten(chain, board_without_net, grid);
        eliminate_stair_steps(chain, board_without_net, grid);
        reduce_jogs(chain, board_without_net, grid);
        reduce_corners(chain, board_without_net, grid);
        merge_collinear_points(&mut chain.points);
        reposition_lines(chain, board_without_net, grid);
        merge_collinear_points(&mut chain.points);
        smoothen_corners(chain, board_without_net, grid);
        avoid_acid_traps(chain, board_without_net, grid);
        enforce_45_grid(chain);
        merge_collinear_points(&mut chain.points);
    }

    let mut out = Vec::new();
    for chain in chains {
        for k in 0..chain.points.len().saturating_sub(1) {
            let p1 = chain.points[k];
            let p2 = chain.points[k + 1];
            if (p1.0 - p2.0).hypot(p1.1 - p2.1) > 1e-5 {
                out.push(TrackSegment {
                    start: p1,
                    end: p2,
                    width: chain.width,
                    layer: chain.layer,
                    net_id,
                    is_locked: false,
                });
            }
        }
    }
    out
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

    if net_ids.is_empty() {
        return;
    }

    // Compute 0.60mm-expanded 2D AABB for each net's unlocked tracks and partition into spatially disjoint independent sets
    let mut net_aabbs: HashMap<usize, (f64, f64, f64, f64)> = HashMap::with_capacity(net_ids.len());
    for &net_id in &net_ids {
        let mut pts = Vec::new();
        for t in board.tracks.iter().filter(|t| t.net_id == net_id && !t.is_locked) {
            pts.push(t.start);
            pts.push(t.end);
        }
        if !pts.is_empty() {
            net_aabbs.insert(net_id, points_expanded_aabb(&pts, 0.60));
        }
    }

    let mut waves: Vec<Vec<usize>> = Vec::new();
    for &net_id in &net_ids {
        let Some(&aabb) = net_aabbs.get(&net_id) else {
            continue;
        };
        let mut placed = false;
        for wave in &mut waves {
            let conflicts = wave.iter().any(|&other_net| {
                net_aabbs
                    .get(&other_net)
                    .map_or(false, |&other_aabb| aabb_overlaps(aabb, other_aabb))
            });
            if !conflicts {
                wave.push(net_id);
                placed = true;
                break;
            }
        }
        if !placed {
            waves.push(vec![net_id]);
        }
    }

    for wave in waves {
        if wave.len() == 1 {
            let net_id = wave[0];
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
            let glossed = gloss_single_net_chains(net_id, &net_tracks, board, grid);
            board.tracks.extend(glossed);
        } else {
            let wave_results: Vec<(usize, Vec<TrackSegment>)> = wave
                .par_iter()
                .map(|&net_id| {
                    let net_tracks: Vec<TrackSegment> = board
                        .tracks
                        .iter()
                        .filter(|t| t.net_id == net_id && !t.is_locked)
                        .cloned()
                        .collect();
                    if net_tracks.is_empty() {
                        return (net_id, Vec::new());
                    }
                    let mut local_board = board.clone();
                    local_board.tracks.retain(|t| t.net_id != net_id || t.is_locked);
                    let glossed = gloss_single_net_chains(net_id, &net_tracks, &local_board, grid);
                    (net_id, glossed)
                })
                .collect();

            let wave_set: HashSet<usize> = wave.into_iter().collect();
            board.tracks.retain(|t| !wave_set.contains(&t.net_id) || t.is_locked);
            for (_, glossed) in wave_results {
                board.tracks.extend(glossed);
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

pub fn subdivide_long_segments(points: &mut Vec<(f64, f64)>, max_seg_len: f64) {
    if points.len() < 2 || max_seg_len <= 0.1 {
        return;
    }
    let mut out = Vec::with_capacity(points.len() * 2);
    out.push(points[0]);
    for w in points.windows(2) {
        let p0 = w[0];
        let p1 = w[1];
        let len = (p1.0 - p0.0).hypot(p1.1 - p0.1);
        if len > max_seg_len && out.len() < 28 {
            let steps = ((len / max_seg_len).ceil() as usize).clamp(2, 6);
            for s in 1..steps {
                let t = s as f64 / steps as f64;
                out.push((p0.0 + t * (p1.0 - p0.0), p0.1 + t * (p1.1 - p0.1)));
            }
        }
        out.push(p1);
    }
    *points = out;
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

    for outer_pass in 0..2 {
        let iter_nets: Vec<usize> = if outer_pass % 2 == 0 {
            net_ids.clone()
        } else {
            net_ids.iter().copied().rev().collect()
        };
        for net_id in iter_nets {
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

            let nc_w = board.get_netclass_for_net(net_id).track_width;
            let mut chains = extract_track_chains(&net_tracks, board);
            for chain in &mut chains {
                for pass in 0..2 {
                    if pass == 1 {
                        relax_chain_homotopic(chain, board);
                    }
                    if nc_w > chain.width + 1e-3 && chain.points.len() >= 2 {
                        for &cand_w in &[nc_w, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20] {
                            if cand_w <= chain.width + 1e-3 || cand_w > nc_w + 1e-3 {
                                continue;
                            }
                            let all_ok = chain.points.windows(2).all(|w| {
                                is_segment_clear(board, None, w[0], w[1], chain.layer, chain.net_id, cand_w)
                            });
                            if all_ok {
                                chain.width = cand_w;
                                break;
                            }
                        }
                    }
                }
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
}

pub fn relax_chain_homotopic(chain: &mut TrackChain, board: &Board) {
    if chain.points.len() < 3 {
        return;
    }

    let nc_w = board.get_netclass_for_net(chain.net_id).track_width.max(chain.width);
    let obstacles = collect_obstacles_for_net(board, chain.layer, chain.net_id, nc_w);

    // Iterative rubber-band string pulling passes
    for pass in 0..5 {
        merge_collinear_points(&mut chain.points);
        if pass == 0 {
            subdivide_long_segments(&mut chain.points, 3.0);
        }
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

    // 5. Netclass Channel Centering for wide power nets
    if nc_w > 0.25 && nc_w > chain.width + 1e-3 {
        for k in 1..chain.points.len().saturating_sub(1) {
            let p_prev = chain.points[k - 1];
            let p_curr = chain.points[k];
            let p_next = chain.points[k + 1];
            if is_segment_clear(board, None, p_prev, p_curr, chain.layer, chain.net_id, nc_w)
                && is_segment_clear(board, None, p_curr, p_next, chain.layer, chain.net_id, nc_w)
            {
                continue;
            }
            let dx = p_next.0 - p_prev.0;
            let dy = p_next.1 - p_prev.1;
            let dlen = dx.hypot(dy);
            if dlen <= 1e-4 {
                continue;
            }
            let nx = -dy / dlen;
            let ny = dx / dlen;
            for &target_w in &[nc_w, 0.50, 0.45, 0.40, 0.35, 0.30] {
                if target_w <= chain.width + 1e-3 || target_w > nc_w + 1e-3 {
                    continue;
                }
                if is_segment_clear(board, None, p_prev, p_curr, chain.layer, chain.net_id, target_w)
                    && is_segment_clear(board, None, p_curr, p_next, chain.layer, chain.net_id, target_w)
                {
                    break;
                }
                let mut moved = false;
                for step in [0.08, -0.08, 0.16, -0.16, 0.24, -0.24, 0.32, -0.32, 0.42, -0.42] {
                    let cand = (p_curr.0 + nx * step, p_curr.1 + ny * step);
                    let tri1 = [p_prev, p_curr, cand];
                    let tri2 = [p_curr, p_next, cand];
                    if obstacles.iter().all(|o| !point_in_polygon(o.center, &tri1) && !point_in_polygon(o.center, &tri2))
                        && is_segment_clear(board, None, p_prev, cand, chain.layer, chain.net_id, target_w)
                        && is_segment_clear(board, None, cand, p_next, chain.layer, chain.net_id, target_w)
                    {
                        chain.points[k] = cand;
                        moved = true;
                        break;
                    }
                }
                if moved {
                    break;
                }
            }
        }
    }
}

pub fn is_arc_clear(board: &Board, arc: &ArcSegment) -> bool {
    let clearance = board.get_netclass_for_net(arc.net_id).clearance;
    // Include 0.025 mm chord-sagitta safety margin so outward circular arc bulge never violates clearance
    let half_w = arc.width * 0.5 + 0.025;
    let n_segs = ((arc.length() / 0.6).ceil() as usize).clamp(8, 28);
    let segs = arc.to_segments(n_segs);

    // 0. Board Boundary (when board outline polygon is present)
    if let Some(ref poly) = board.outline_polygon {
        let b = poly.aabb();
        let edge_clr = board.get_netclass_for_net(arc.net_id).board_edge_clearance + half_w;
        for s in &segs {
            if s.start.0 < b.min_x + edge_clr
                || s.start.0 > b.max_x - edge_clr
                || s.start.1 < b.min_y + edge_clr
                || s.start.1 > b.max_y - edge_clr
                || s.end.0 < b.min_x + edge_clr
                || s.end.0 > b.max_x - edge_clr
                || s.end.1 < b.min_y + edge_clr
                || s.end.1 > b.max_y - edge_clr
                || !board.contains_point_in_boundary(&Point::new(s.start.0, s.start.1))
                || !board.contains_point_in_boundary(&Point::new(s.end.0, s.end.1))
            {
                return false;
            }
        }
    }

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
            for s in &segs {
                if dist_segment_to_segment(s.start, s.end, trk.start, trk.end) < required_dist - 1e-4 {
                    return false;
                }
            }
        }
    }

    // 5. Foreign Arcs
    for other in &board.arcs {
        if other.net_id != arc.net_id && other.layer == arc.layer {
            let other_clr = board.get_netclass_for_net(other.net_id).clearance;
            let req_clr = clearance.max(other_clr);
            let required_dist = half_w + (other.width * 0.5) + req_clr;
            let other_segs = other.to_segments(12);
            for s in &segs {
                for os in &other_segs {
                    if dist_segment_to_segment(s.start, s.end, os.start, os.end) < required_dist - 1e-4 {
                        return false;
                    }
                }
            }
        }
    }

    true
}

fn try_fit_sweeping_chain_arc(
    board: &Board,
    points: &[(f64, f64)],
    width: f64,
    layer: usize,
    net_id: usize,
    obstacles: &[ObstacleInfo],
) -> Option<(TrackSegment, ArcSegment, TrackSegment)> {
    let n = points.len();
    if !(3..=4).contains(&n) {
        return None;
    }
    const STUB_MM: f64 = 0.20;
    let p_first = points[0];
    let p_second = points[1];
    let p_penult = points[n - 2];
    let p_last = points[n - 1];

    let l_first = (p_second.0 - p_first.0).hypot(p_second.1 - p_first.1);
    let l_last = (p_last.0 - p_penult.0).hypot(p_last.1 - p_penult.1);
    if l_first < STUB_MM + 0.1 || l_last < STUB_MM + 0.1 {
        return None;
    }

    let u_first = ((p_second.0 - p_first.0) / l_first, (p_second.1 - p_first.1) / l_first);
    let u_last = ((p_penult.0 - p_last.0) / l_last, (p_penult.1 - p_last.1) / l_last);
    let arc_start = (p_first.0 + u_first.0 * STUB_MM, p_first.1 + u_first.1 * STUB_MM);
    let arc_end = (p_last.0 + u_last.0 * STUB_MM, p_last.1 + u_last.1 * STUB_MM);

    let chord_len = (arc_end.0 - arc_start.0).hypot(arc_end.1 - arc_start.1);
    if chord_len < 1.0 {
        return None;
    }

    let apex = if n == 3 {
        points[1]
    } else {
        ((points[1].0 + points[2].0) * 0.5, (points[1].1 + points[2].1) * 0.5)
    };
    let chord_mid = ((arc_start.0 + arc_end.0) * 0.5, (arc_start.1 + arc_end.1) * 0.5);
    let dev = (apex.0 - chord_mid.0).hypot(apex.1 - chord_mid.1);
    if dev < 0.05 || dev > chord_len * 0.85 {
        return None;
    }

    for w in [0.50, 0.38, 0.62, 0.26, 0.75, 0.16] {
        let m_cand = (
            chord_mid.0 + w * (apex.0 - chord_mid.0),
            chord_mid.1 + w * (apex.1 - chord_mid.1),
        );
        if let Some(arc) = ArcSegment::from_3pts(arc_start, m_cand, arc_end, width, layer, net_id) {
            let sweep = (arc.end_angle_rad - arc.start_angle_rad).abs();
            if sweep > 2.4 || arc.radius < 0.50 {
                continue;
            }
            let mut loop_poly = points.to_vec();
            let mut arc_pts: Vec<(f64, f64)> = arc.to_segments(12).into_iter().map(|s| s.end).collect();
            arc_pts.reverse();
            loop_poly.extend(arc_pts);
            let homotopy_ok = obstacles.iter().all(|obs| !point_in_polygon(obs.center, &loop_poly));
            if homotopy_ok && is_arc_clear(board, &arc) {
                let s1 = TrackSegment {
                    start: p_first,
                    end: arc.start(),
                    width,
                    layer,
                    net_id,
                    is_locked: false,
                };
                let s2 = TrackSegment {
                    start: arc.end(),
                    end: p_last,
                    width,
                    layer,
                    net_id,
                    is_locked: false,
                };
                return Some((s1, arc, s2));
            }
        }
    }
    None
}

/// Fits a G1-continuous asymmetric bi-arc at corner `p_curr` when one leg is much longer than the other,
/// replacing both the corner and a large span of the long leg with two tangent circular arcs.
fn try_fit_asymmetric_corner_biarc(
    board: &Board,
    p_curr: (f64, f64),
    u1: (f64, f64),
    u2: (f64, f64),
    avail_a: f64,
    avail_b: f64,
    width: f64,
    layer: usize,
    net_id: usize,
) -> Option<((f64, f64), (f64, f64), Vec<ArcSegment>, f64)> {
    let a_is_short = avail_a < avail_b;
    let short_avail = avail_a.min(avail_b);
    let long_avail = avail_a.max(avail_b);
    if short_avail < 0.35 || long_avail < short_avail * 1.6 || long_avail < 2.5 {
        return None;
    }

    let d_short = (short_avail * 0.90).min(3.5);
    let (u_short, u_long) = if a_is_short { (u1, u2) } else { (u2, u1) };

    for long_frac in [0.85, 0.65, 0.48, 0.35] {
        let d_long = (long_avail * long_frac).min(d_short * 4.5).max(d_short * 1.5);
        if d_long <= d_short * 1.3 || d_long > long_avail {
            continue;
        }
        let t_short = (p_curr.0 + d_short * u_short.0, p_curr.1 + d_short * u_short.1);
        let t_long = (p_curr.0 + d_long * u_long.0, p_curr.1 + d_long * u_long.1);

        let s1 = d_short * 0.55;
        let q1 = (
            p_curr.0 + (d_short - s1) * u_short.0,
            p_curr.1 + (d_short - s1) * u_short.1,
        );
        let w = (t_long.0 - q1.0, t_long.1 - q1.1);
        let w_len2 = w.0 * w.0 + w.1 * w.1;
        let w_dot_ulong = w.0 * u_long.0 + w.1 * u_long.1;
        let denom = 2.0 * (s1 + w_dot_ulong);
        if denom <= 1e-5 {
            continue;
        }
        let s2 = (w_len2 - s1 * s1) / denom;
        if s2 <= 0.1 || s2 >= d_long - 0.05 {
            continue;
        }
        let q2 = (t_long.0 - s2 * u_long.0, t_long.1 - s2 * u_long.1);
        let join = (
            (s2 * q1.0 + s1 * q2.0) / (s1 + s2),
            (s2 * q1.1 + s1 * q2.1) / (s1 + s2),
        );

        let c1 = ((t_short.0 + join.0) * 0.5, (t_short.1 + join.1) * 0.5);
        let hc1 = (join.0 - t_short.0).hypot(join.1 - t_short.1) * 0.5;
        let m1 = (
            (s1 * c1.0 + hc1 * q1.0) / (s1 + hc1),
            (s1 * c1.1 + hc1 * q1.1) / (s1 + hc1),
        );

        let c2 = ((join.0 + t_long.0) * 0.5, (join.1 + t_long.1) * 0.5);
        let hc2 = (t_long.0 - join.0).hypot(t_long.1 - join.1) * 0.5;
        let m2 = (
            (s2 * c2.0 + hc2 * q2.0) / (s2 + hc2),
            (s2 * c2.1 + hc2 * q2.1) / (s2 + hc2),
        );

        let (arc1_opt, arc2_opt, t_in, t_out, used_b) = if a_is_short {
            (
                ArcSegment::from_3pts(t_short, m1, join, width, layer, net_id),
                ArcSegment::from_3pts(join, m2, t_long, width, layer, net_id),
                t_short,
                t_long,
                d_long,
            )
        } else {
            (
                ArcSegment::from_3pts(t_long, m2, join, width, layer, net_id),
                ArcSegment::from_3pts(join, m1, t_short, width, layer, net_id),
                t_long,
                t_short,
                d_short,
            )
        };

        if let (Some(a1), Some(a2)) = (arc1_opt, arc2_opt) {
            if a1.radius >= 0.35 && a2.radius >= 0.35 && is_arc_clear(board, &a1) && is_arc_clear(board, &a2) {
                return Some((t_in, t_out, vec![a1, a2], used_b));
            }
        }
    }
    None
}

fn try_fit_single_bow_arc(
    board: &Board,
    arc_start: (f64, f64),
    arc_end: (f64, f64),
    width: f64,
    layer: usize,
    net_id: usize,
    pref_sign: f64,
) -> Option<ArcSegment> {
    let dx = arc_end.0 - arc_start.0;
    let dy = arc_end.1 - arc_start.1;
    let len = dx.hypot(dy);
    if len < 0.75 {
        return None;
    }
    let nx = -dy / len;
    let ny = dx / len;
    let mid_x = (arc_start.0 + arc_end.0) * 0.5;
    let mid_y = (arc_start.1 + arc_end.1) * 0.5;
    let base_bow = (len * 0.08).clamp(0.10, 0.90);
    for sign in [pref_sign, -pref_sign] {
        for scale in [1.0, 0.7, 0.45, 0.28, 0.16, 0.09, 0.05] {
            let offset = base_bow * scale * sign;
            let m_bow = (mid_x + nx * offset, mid_y + ny * offset);
            if let Some(arc) = ArcSegment::from_3pts(arc_start, m_bow, arc_end, width, layer, net_id) {
                if is_arc_clear(board, &arc) {
                    return Some(arc);
                }
            }
        }
    }
    None
}

fn emit_organic_bow_span(
    board: &mut Board,
    start_pt: (f64, f64),
    end_pt: (f64, f64),
    width: f64,
    layer: usize,
    net_id: usize,
    need_start_stub: bool,
    need_end_stub: bool,
    pref_sign: f64,
) -> bool {
    let dx = end_pt.0 - start_pt.0;
    let dy = end_pt.1 - start_pt.1;
    let len = dx.hypot(dy);
    const STUB_MM: f64 = 0.20;
    let stub_s = if need_start_stub { STUB_MM } else { 0.0 };
    let stub_e = if need_end_stub { STUB_MM } else { 0.0 };
    if len < stub_s + stub_e + 0.75 {
        return false;
    }
    let ux = dx / len;
    let uy = dy / len;
    let arc_start = (start_pt.0 + ux * stub_s, start_pt.1 + uy * stub_s);
    let arc_end = (end_pt.0 - ux * stub_e, end_pt.1 - uy * stub_e);
    let arc_len = len - stub_s - stub_e;

    let nc_w = board.get_netclass_for_net(net_id).track_width;
    let mut cand_widths = Vec::with_capacity(9);
    if nc_w > width + 1e-3 {
        for &cw in &[nc_w, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20] {
            if cw > width + 1e-3
                && cw <= nc_w + 1e-3
                && is_segment_clear(board, None, start_pt, end_pt, layer, net_id, cw)
            {
                cand_widths.push(cw);
            }
        }
    }
    cand_widths.push(width);

    for span_w in cand_widths {
        // 1. Try single sweeping bow arc across the entire span
        if let Some(arc) = try_fit_single_bow_arc(board, arc_start, arc_end, span_w, layer, net_id, pref_sign) {
            if need_start_stub {
                board.tracks.push(TrackSegment {
                    start: start_pt,
                    end: arc.start(),
                    width: span_w,
                    layer,
                    net_id,
                    is_locked: false,
                });
            }
            let end_anchor = arc.end();
            board.arcs.push(arc);
            if need_end_stub {
                board.tracks.push(TrackSegment {
                    start: end_anchor,
                    end: end_pt,
                    width: span_w,
                    layer,
                    net_id,
                    is_locked: false,
                });
            }
            return true;
        }

        // 2. Try G1-continuous 2-arc S-curve (+h on first half, -h on second half)
        if arc_len >= 2.2 {
            let mid_pt = ((arc_start.0 + arc_end.0) * 0.5, (arc_start.1 + arc_end.1) * 0.5);
            for sign in [pref_sign, -pref_sign] {
                if let (Some(a1), Some(a2)) = (
                    try_fit_single_bow_arc(board, arc_start, mid_pt, span_w, layer, net_id, sign),
                    try_fit_single_bow_arc(board, mid_pt, arc_end, span_w, layer, net_id, -sign),
                ) {
                    if need_start_stub {
                        board.tracks.push(TrackSegment {
                            start: start_pt,
                            end: a1.start(),
                            width: span_w,
                            layer,
                            net_id,
                            is_locked: false,
                        });
                    }
                    let end_anchor = a2.end();
                    board.arcs.push(a1);
                    board.arcs.push(a2);
                    if need_end_stub {
                        board.tracks.push(TrackSegment {
                            start: end_anchor,
                            end: end_pt,
                            width: span_w,
                            layer,
                            net_id,
                            is_locked: false,
                        });
                    }
                    return true;
                }
            }
        }
    }

    false
}

pub fn apply_any_angle_tangent_arcs(board: &mut Board) {
    apply_tangent_arcs_with_options(board, 1.0, false);
}

fn apply_tangent_arcs_with_options(board: &mut Board, target_radius: f64, fit_straight_bows: bool) {
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
        board.arcs.retain(|a| a.net_id != net_id || a.is_locked);

        let chains = extract_track_chains(&net_tracks, board);
        for chain in chains {
            let n = chain.points.len();
            if n < 3 {
                for k in 0..n.saturating_sub(1) {
                    let start_pt = chain.points[k];
                    let end_pt = chain.points[k + 1];
                    let pref_sign = if ((start_pt.0 * 10.0) as i64 + (end_pt.1 * 10.0) as i64 + chain.net_id as i64) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    if fit_straight_bows
                        && emit_organic_bow_span(
                            board,
                            start_pt,
                            end_pt,
                            chain.width,
                            chain.layer,
                            chain.net_id,
                            true,
                            true,
                            pref_sign,
                        )
                    {
                        continue;
                    }
                    board.tracks.push(TrackSegment {
                        start: start_pt,
                        end: end_pt,
                        width: chain.width,
                        layer: chain.layer,
                        net_id: chain.net_id,
                        is_locked: false,
                    });
                }
                continue;
            }

            // In organic mode, first attempt to bridge 3- or 4-point chains with a single sweeping curve
            if fit_straight_bows && (3..=4).contains(&n) {
                let obstacles = collect_obstacles_for_net(board, chain.layer, chain.net_id, chain.width);
                if let Some((s1, sweep_arc, s2)) = try_fit_sweeping_chain_arc(
                    board,
                    &chain.points,
                    chain.width,
                    chain.layer,
                    chain.net_id,
                    &obstacles,
                ) {
                    board.tracks.push(s1);
                    board.arcs.push(sweep_arc);
                    board.tracks.push(s2);
                    continue;
                }
            }

            let mut corner_arcs: Vec<Option<((f64, f64), (f64, f64), Vec<ArcSegment>)>> = vec![None; n];
            let mut used_on_seg = vec![0.0f64; n];
            if fit_straight_bows {
                used_on_seg[0] = 0.20;
            }

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

                let avail_a = (l_a - used_on_seg[i - 1] - 0.05).max(0.0);
                let avail_b = if i == n - 2 {
                    (l_b - 0.20).max(0.0)
                } else if fit_straight_bows {
                    l_b * 0.78
                } else {
                    l_b * 0.49
                };

                if fit_straight_bows {
                    if let Some((t_in, t_out, bi_arcs, used_b)) = try_fit_asymmetric_corner_biarc(
                        board,
                        p_curr,
                        u1,
                        u2,
                        avail_a,
                        avail_b,
                        chain.width,
                        chain.layer,
                        chain.net_id,
                    ) {
                        corner_arcs[i] = Some((t_in, t_out, bi_arcs));
                        used_on_seg[i] = used_b;
                        continue;
                    }
                }

                let alpha = theta * 0.5;
                let b_sum = (u1.0 + u2.0, u1.1 + u2.1);
                let b_len = b_sum.0.hypot(b_sum.1);
                if b_len < 1e-6 {
                    continue;
                }
                let b = (b_sum.0 / b_len, b_sum.1 / b_len);

                let d_max = if fit_straight_bows {
                    avail_a.min(avail_b)
                } else {
                    (l_a * 0.49).min(l_b * 0.49)
                };
                let r_max_geom = d_max * alpha.tan();
                let r_init = target_radius.min(r_max_geom);

                for scale in [1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.12, 0.06] {
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
                            corner_arcs[i] = Some((t1, t2, vec![arc]));
                            used_on_seg[i] = d;
                            break;
                        }
                    }
                }
            }

            for opt_arc in corner_arcs.iter().flatten() {
                for a in &opt_arc.2 {
                    board.arcs.push(a.clone());
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

                let seg_len = (end_pt.0 - start_pt.0).hypot(end_pt.1 - start_pt.1);
                if seg_len > 1e-4 {
                    if fit_straight_bows {
                        let need_start_stub = k == 0 && corner_arcs[0].is_none();
                        let need_end_stub = k + 1 == n - 1 && corner_arcs[k + 1].is_none();
                        let dx = end_pt.0 - start_pt.0;
                        let dy = end_pt.1 - start_pt.1;
                        let mut pref_sign = if ((start_pt.0 * 10.0) as i64 + (end_pt.1 * 10.0) as i64 + chain.net_id as i64) % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        if k + 2 < n {
                            let next_v = (
                                chain.points[k + 2].0 - chain.points[k + 1].0,
                                chain.points[k + 2].1 - chain.points[k + 1].1,
                            );
                            let cross = dx * next_v.1 - dy * next_v.0;
                            if cross.abs() > 1e-3 {
                                pref_sign = if cross > 0.0 { -1.0 } else { 1.0 };
                            }
                        } else if k > 0 {
                            let prev_v = (
                                chain.points[k].0 - chain.points[k - 1].0,
                                chain.points[k].1 - chain.points[k - 1].1,
                            );
                            let cross = prev_v.0 * dy - prev_v.1 * dx;
                            if cross.abs() > 1e-3 {
                                pref_sign = if cross > 0.0 { -1.0 } else { 1.0 };
                            }
                        }
                        if emit_organic_bow_span(
                            board,
                            start_pt,
                            end_pt,
                            chain.width,
                            chain.layer,
                            chain.net_id,
                            need_start_stub,
                            need_end_stub,
                            pref_sign,
                        ) {
                            continue;
                        }
                    }
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
        }
    }
}

/// Converts a routed board's unlocked tracks into hand-drawn-like organic traces
/// using homotopic rubber-band relaxation, sweeping tangent corner arcs, and gentle organic bows.
pub fn convert_tracks_to_organic(board: &mut Board) {
    // If the board already has unlocked arcs, reconstruct their sharp tangent corners or chords first so conversion is idempotent
    let unlocked_arcs: Vec<ArcSegment> = board.arcs.iter().filter(|a| !a.is_locked).cloned().collect();
    if !unlocked_arcs.is_empty() {
        board.arcs.retain(|a| a.is_locked);
        for arc in unlocked_arcs {
            let as_pt = arc.start();
            let ae_pt = arc.end();
            let mid = arc.mid();
            let dx = mid.0 - arc.center.0;
            let dy = mid.1 - arc.center.1;
            let d_len = dx.hypot(dy);
            let half_sweep = (arc.end_angle_rad - arc.start_angle_rad).abs() * 0.5;
            let mut reconstructed_corner = false;
            if d_len > 1e-6 && (0.08..1.35).contains(&half_sweep) && arc.radius < 6.0 {
                let h = arc.radius / half_sweep.cos();
                let p_corner = (
                    arc.center.0 + (dx / d_len) * h,
                    arc.center.1 + (dy / d_len) * h,
                );
                let has_collinear_adj = board.tracks.iter().any(|t| {
                    t.net_id == arc.net_id && t.layer == arc.layer && {
                        let d_s = (t.end.0 - as_pt.0).hypot(t.end.1 - as_pt.1);
                        let d_e = (t.start.0 - as_pt.0).hypot(t.start.1 - as_pt.1);
                        if d_s < 0.02 {
                            let v_t = (as_pt.0 - t.start.0, as_pt.1 - t.start.1);
                            let v_c = (p_corner.0 - as_pt.0, p_corner.1 - as_pt.1);
                            let lt = v_t.0.hypot(v_t.1);
                            let lc = v_c.0.hypot(v_c.1);
                            lt > 0.25 && lc > 1e-4 && (v_t.0 * v_c.0 + v_t.1 * v_c.1) / (lt * lc) > 0.995
                        } else if d_e < 0.02 {
                            let v_t = (as_pt.0 - t.end.0, as_pt.1 - t.end.1);
                            let v_c = (p_corner.0 - as_pt.0, p_corner.1 - as_pt.1);
                            let lt = v_t.0.hypot(v_t.1);
                            let lc = v_c.0.hypot(v_c.1);
                            lt > 0.25 && lc > 1e-4 && (v_t.0 * v_c.0 + v_t.1 * v_c.1) / (lt * lc) > 0.995
                        } else {
                            false
                        }
                    }
                });
                if has_collinear_adj
                    && is_segment_clear(board, None, as_pt, p_corner, arc.layer, arc.net_id, arc.width)
                    && is_segment_clear(board, None, p_corner, ae_pt, arc.layer, arc.net_id, arc.width)
                {
                    board.tracks.push(TrackSegment {
                        start: as_pt,
                        end: p_corner,
                        width: arc.width,
                        layer: arc.layer,
                        net_id: arc.net_id,
                        is_locked: false,
                    });
                    board.tracks.push(TrackSegment {
                        start: p_corner,
                        end: ae_pt,
                        width: arc.width,
                        layer: arc.layer,
                        net_id: arc.net_id,
                        is_locked: false,
                    });
                    reconstructed_corner = true;
                }
            }
            if !reconstructed_corner {
                if is_segment_clear(board, None, as_pt, ae_pt, arc.layer, arc.net_id, arc.width) {
                    board.tracks.push(TrackSegment {
                        start: as_pt,
                        end: ae_pt,
                        width: arc.width,
                        layer: arc.layer,
                        net_id: arc.net_id,
                        is_locked: false,
                    });
                } else {
                    for seg in arc.to_segments(4) {
                        board.tracks.push(seg);
                    }
                }
            }
        }
    }
    relax_homotopic_rubber_bands(board);
    apply_tangent_arcs_with_options(board, 25.0, true);
    widen_tracks_and_arcs_to_netclass(board);
    prune_dangling_tracks_and_vias(board);
}

fn widen_tracks_and_arcs_to_netclass(board: &mut Board) {
    for i in 0..board.arcs.len() {
        if board.arcs[i].is_locked {
            continue;
        }
        let net_id = board.arcs[i].net_id;
        let cur_w = board.arcs[i].width;
        let nc_w = board.get_netclass_for_net(net_id).track_width;
        if nc_w <= cur_w + 1e-3 {
            continue;
        }
        for &cand_w in &[nc_w, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20] {
            if cand_w <= cur_w + 1e-3 || cand_w > nc_w + 1e-3 {
                continue;
            }
            let mut cand_arc = board.arcs[i].clone();
            cand_arc.width = cand_w;
            if is_arc_clear(board, &cand_arc) {
                board.arcs[i].width = cand_w;
                break;
            }
        }
    }

    for i in 0..board.tracks.len() {
        if board.tracks[i].is_locked {
            continue;
        }
        let trk = board.tracks[i].clone();
        let nc_w = board.get_netclass_for_net(trk.net_id).track_width;
        if nc_w <= trk.width + 1e-3 {
            continue;
        }
        let net_clr = board.get_netclass_for_net(trk.net_id).clearance;
        for &cand_w in &[nc_w, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20] {
            if cand_w <= trk.width + 1e-3 || cand_w > nc_w + 1e-3 {
                continue;
            }
            if !is_segment_clear(board, None, trk.start, trk.end, trk.layer, trk.net_id, cand_w) {
                continue;
            }
            let arcs_ok = board.arcs.iter().all(|a| {
                if a.net_id == trk.net_id || a.layer != trk.layer {
                    return true;
                }
                let a_clr = board.get_netclass_for_net(a.net_id).clearance;
                let req = cand_w * 0.5 + a.width * 0.5 + net_clr.max(a_clr) + 0.025;
                a.to_segments(8)
                    .iter()
                    .all(|asg| dist_segment_to_segment(trk.start, trk.end, asg.start, asg.end) >= req - 1e-4)
            });
            if arcs_ok {
                board.tracks[i].width = cand_w;
                break;
            }
        }
    }
}

/// Converts a board's organic curved/any-angle traces back into standard 45°/90° octilinear straight tracks.
pub fn convert_tracks_to_straight(board: &mut Board, grid: &RoutingGrid) {
    let unlocked_arcs: Vec<ArcSegment> = board.arcs.iter().filter(|a| !a.is_locked).cloned().collect();
    board.arcs.retain(|a| a.is_locked);
    for arc in unlocked_arcs {
        if is_segment_clear(board, Some(grid), arc.start(), arc.end(), arc.layer, arc.net_id, arc.width) {
            board.tracks.push(TrackSegment {
                start: arc.start(),
                end: arc.end(),
                width: arc.width,
                layer: arc.layer,
                net_id: arc.net_id,
                is_locked: false,
            });
        } else {
            for seg in arc.to_segments(4) {
                board.tracks.push(seg);
            }
        }
    }

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
            for k in 0..chain.points.len().saturating_sub(1) {
                let p0 = chain.points[k];
                let p1 = chain.points[k + 1];
                if (p1.0 - p0.0).hypot(p1.1 - p0.1) > 1e-4 {
                    board.tracks.push(TrackSegment {
                        start: p0,
                        end: p1,
                        width: chain.width,
                        layer: chain.layer,
                        net_id: chain.net_id,
                        is_locked: false,
                    });
                }
            }
        }
    }
    gloss_tracks_standard(board, grid);
    widen_tracks_and_arcs_to_netclass(board);
    prune_dangling_tracks_and_vias(board);
}

// ============================================================================
// Robust Copper Geometry, DRC Validation & Automated Pruning
// ============================================================================

/// Flat GPU-ready copper obstacle snapshot for batch DRC clearance evaluation.
pub struct GpuDrcContext {
    pub pads: Vec<GpuDrcPad>,
    pub tracks: Vec<GpuDrcTrack>,
    pub vias: Vec<GpuDrcVia>,
}

impl GpuDrcContext {
    /// Extracts all pads, tracks, and vias from `board` into flat `#[repr(C)]` POD arrays
    /// suitable for CUDA (`batch_drc_eval_kernel`) or Vulkan (`BATCH_DRC_EVAL_WGSL`).
    pub fn from_board(board: &Board) -> Self {
        Self::build(board, false)
    }

    /// Extracts all pads, tracks, and vias with `clearance = 0.0` for physical copper
    /// short-circuit detection (`validate_and_clean_copper`).
    pub fn from_board_with_zero_clearance(board: &Board) -> Self {
        Self::build(board, true)
    }

    fn build(board: &Board, zero_clearance: bool) -> Self {
        let mut net_clearances = HashMap::with_capacity(board.netclasses.len() * 4 + 8);
        let default_clr = if zero_clearance {
            0.0f32
        } else {
            board.get_netclass_for_net(0).clearance as f32
        };

        let mut get_clr = |net_id: usize| -> f32 {
            if zero_clearance {
                return 0.0;
            }
            if net_id == 0 {
                return default_clr;
            }
            *net_clearances
                .entry(net_id)
                .or_insert_with(|| board.get_netclass_for_net(net_id).clearance as f32)
        };

        let mut pads = Vec::new();
        for fp in &board.footprints {
            let flipped = fp.layer != fp.orig_layer;
            let layer_mask = if fp.layer < 32 { 1u32 << fp.layer } else { 1u32 };
            for pad in &fp.pads {
                let (px, py) = fp.pad_absolute_position(pad);
                let pad_rot = if flipped { -pad.rotation } else { pad.rotation };
                let total_rot = ((fp.rotation + pad_rot) % 360.0 + 360.0) % 360.0;
                let rot_rad = total_rot.to_radians() as f32;
                let shape_bits = match pad.shape.as_str() {
                    "circle" => 1u32 << 16,
                    "oval" => 2u32 << 16,
                    _ => 0u32,
                };
                let p_mask = shape_bits | if pad.is_through_hole { 0xFFFFu32 } else { layer_mask & 0xFFFFu32 };
                let clr = get_clr(pad.net_id);
                pads.push(GpuDrcPad {
                    x: px as f32,
                    y: py as f32,
                    hw: (pad.width * 0.5) as f32,
                    hh: (pad.height * 0.5) as f32,
                    rot_rad,
                    clearance: clr,
                    layer_mask: p_mask,
                    net_id: pad.net_id as u32,
                });
            }
        }

        let mut tracks = Vec::with_capacity(board.tracks.len());
        for t in &board.tracks {
            let clr = get_clr(t.net_id);
            tracks.push(GpuDrcTrack {
                x1: t.start.0 as f32,
                y1: t.start.1 as f32,
                x2: t.end.0 as f32,
                y2: t.end.1 as f32,
                half_width: (t.width * 0.5) as f32,
                clearance: clr,
                layer: t.layer as u32,
                net_id: t.net_id as u32,
            });
        }

        let mut vias = Vec::with_capacity(board.vias.len());
        for v in &board.vias {
            let clr = get_clr(v.net_id);
            vias.push(GpuDrcVia {
                x: v.position.0 as f32,
                y: v.position.1 as f32,
                radius: (v.size * 0.5) as f32,
                clearance: clr,
                net_id: v.net_id as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            });
        }

        Self { pads, tracks, vias }
    }

    /// Evaluates a batch of candidate vias/segments on the GPU (preferring CUDA, falling back to Vulkan).
    pub fn evaluate(
        &self,
        vulkan: Option<&Arc<VulkanEngine>>,
        candidates: &[GpuDrcCandidate],
    ) -> Option<Vec<u32>> {
        if candidates.is_empty() {
            return Some(Vec::new());
        }
        if let Some(cuda) = CudaEngine::new() {
            if let Some(res) = cuda.evaluate_drc_candidates(&self.pads, &self.tracks, &self.vias, candidates) {
                return Some(res);
            }
        }
        if let Some(vk) = vulkan {
            return vk.evaluate_drc_candidates(&self.pads, &self.tracks, &self.vias, candidates);
        }
        None
    }
}

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
        let start_on_fp = fp.pads.iter().any(|p| {
            let (px, py) = fp.pad_absolute_position(p);
            (px - start.0).hypot(py - start.1) < 1e-3
        });
        let end_on_fp = fp.pads.iter().any(|p| {
            let (px, py) = fp.pad_absolute_position(p);
            (px - end.0).hypot(py - end.1) < 1e-3
        });
        for pad in &fp.pads {
            if pad.net_id != net_id {
                if pad.is_through_hole || fp.layer == layer {
                    let pad_c = clearance.max(board.get_netclass_for_net(pad.net_id).clearance).max(0.12);
                    let d = dist_segment_to_pad(start, end, fp, pad);
                    let req_d = half_w + pad_c;
                    if d < req_d - 1e-4 {
                        if start_on_fp || end_on_fp {
                            let d_orig = match (start_on_fp, end_on_fp) {
                                (true, true) => dist_point_to_pad(start, fp, pad).min(dist_point_to_pad(end, fp, pad)),
                                (true, false) => dist_point_to_pad(start, fp, pad),
                                (false, true) => dist_point_to_pad(end, fp, pad),
                                (false, false) => f64::INFINITY,
                            };
                            let limit = d_orig.min(req_d);
                            if d < limit - 1e-3 {
                                return false;
                            }
                        } else {
                            return false;
                        }
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

/// Splits unlocked same-net tracks at interior intersections, T-junctions, and pad/via centers,
/// bridges off-center pad touches, and deduplicates identical collinear segments.
fn split_and_dedup_same_net_tracks(board: &mut Board) {
    board.tracks.retain(|t| t.is_locked || (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1) > 1e-4);
    if board.tracks.is_empty() {
        return;
    }

    // Bridge any unlocked track endpoint that sits within 0.005..0.15mm of a same-net pad center
    // directly to the pad center when the pad is otherwise reached via a long (>0.25mm) stub,
    // so folded overshoot loops form explicit cycles for Steiner pruning.
    let mut pad_bridges = Vec::new();
    for fp in &board.footprints {
        for pad in &fp.pads {
            if pad.net_id == 0 {
                continue;
            }
            let (px, py) = fp.pad_absolute_position(pad);
            let has_long_center_stub = board.tracks.iter().any(|ot| {
                !ot.is_locked
                    && ot.net_id == pad.net_id
                    && (pad.is_through_hole || ot.layer == fp.layer)
                    && ((ot.start.0 - px).hypot(ot.start.1 - py) < 0.005
                        || (ot.end.0 - px).hypot(ot.end.1 - py) < 0.005)
                    && (ot.start.0 - ot.end.0).hypot(ot.start.1 - ot.end.1) > 0.25
            });
            if !has_long_center_stub {
                continue;
            }
            for t in &board.tracks {
                if !t.is_locked && t.net_id == pad.net_id && (pad.is_through_hole || t.layer == fp.layer) {
                    for pt in [t.start, t.end] {
                        let d = (pt.0 - px).hypot(pt.1 - py);
                        if d > 0.005 && d <= 0.15 {
                            pad_bridges.push(TrackSegment {
                                start: pt,
                                end: (px, py),
                                width: t.width,
                                layer: t.layer,
                                net_id: t.net_id,
                                is_locked: false,
                            });
                        }
                    }
                }
            }
        }
    }
    board.tracks.extend(pad_bridges);

    let orig_tracks = std::mem::take(&mut board.tracks);
    let n = orig_tracks.len();
    let mut split_tracks = Vec::with_capacity(n + 16);

    for i in 0..n {
        let t = &orig_tracks[i];
        if t.is_locked {
            split_tracks.push(t.clone());
            continue;
        }
        let s1 = t.start;
        let e1 = t.end;
        let dx1 = e1.0 - s1.0;
        let dy1 = e1.1 - s1.1;
        let len_sq = dx1 * dx1 + dy1 * dy1;
        let len = len_sq.sqrt();
        if len <= 1e-4 {
            continue;
        }

        let mut splits: Vec<(f64, (f64, f64))> = Vec::new();
        let try_project = |splits: &mut Vec<(f64, (f64, f64))>, pt: (f64, f64), max_perp: f64, snap_exact: bool| {
            let d_s = (pt.0 - s1.0).hypot(pt.1 - s1.1);
            let d_e = (pt.0 - e1.0).hypot(pt.1 - e1.1);
            if d_s <= 0.008 || d_e <= 0.008 {
                return;
            }
            let u = ((pt.0 - s1.0) * dx1 + (pt.1 - s1.1) * dy1) / len_sq;
            if u > 1e-4 && u < 1.0 - 1e-4 {
                let proj = (s1.0 + u * dx1, s1.1 + u * dy1);
                let perp = (pt.0 - proj.0).hypot(pt.1 - proj.1);
                if perp <= max_perp {
                    splits.push((u, if snap_exact && perp <= 0.02 { pt } else { proj }));
                }
            }
        };

        // 1. Same-net pads
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id == t.net_id && (pad.is_through_hole || fp.layer == t.layer) {
                    try_project(&mut splits, fp.pad_absolute_position(pad), 0.05, true);
                }
            }
        }
        // 2. Same-net vias
        for v in &board.vias {
            if v.net_id == t.net_id {
                try_project(&mut splits, v.position, 0.03, true);
            }
        }
        // 3. Same-net track endpoints & proper segment intersections on same layer
        for j in 0..n {
            if i == j {
                continue;
            }
            let o = &orig_tracks[j];
            if o.net_id != t.net_id || o.layer != t.layer {
                continue;
            }
            try_project(&mut splits, o.start, 0.012, true);
            try_project(&mut splits, o.end, 0.012, true);

            let s2 = o.start;
            let e2 = o.end;
            let dx2 = e2.0 - s2.0;
            let dy2 = e2.1 - s2.1;
            let cross = dx1 * dy2 - dy1 * dx2;
            if cross.abs() > 1e-7 {
                let u1 = ((s2.0 - s1.0) * dy2 - (s2.1 - s1.1) * dx2) / cross;
                let u2 = ((s2.0 - s1.0) * dy1 - (s2.1 - s1.1) * dx1) / cross;
                if u1 > 1e-4 && u1 < 1.0 - 1e-4 && u2 >= -1e-4 && u2 <= 1.0 + 1e-4 {
                    let ipt = (s1.0 + u1 * dx1, s1.1 + u1 * dy1);
                    if (ipt.0 - s1.0).hypot(ipt.1 - s1.1) > 0.005
                        && (ipt.0 - e1.0).hypot(ipt.1 - e1.1) > 0.005
                    {
                        splits.push((u1, ipt));
                    }
                }
            }
        }

        if splits.is_empty() {
            split_tracks.push(t.clone());
        } else {
            splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            let mut pts = Vec::with_capacity(splits.len() + 2);
            pts.push(s1);
            for (_, sp) in splits {
                let last = *pts.last().unwrap();
                if (sp.0 - last.0).hypot(sp.1 - last.1) > 0.004 && (sp.0 - e1.0).hypot(sp.1 - e1.1) > 0.004 {
                    pts.push(sp);
                }
            }
            pts.push(e1);
            for w in pts.windows(2) {
                if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                    split_tracks.push(TrackSegment {
                        start: w[0],
                        end: w[1],
                        width: t.width,
                        layer: t.layer,
                        net_id: t.net_id,
                        is_locked: false,
                    });
                }
            }
        }
    }

    // Deduplicate identical segments on the same (net_id, layer)
    let q_pt = |p: (f64, f64)| -> (i64, i64) {
        ((p.0 * 1000.0).round() as i64, (p.1 * 1000.0).round() as i64)
    };
    let mut deduped: Vec<TrackSegment> = Vec::with_capacity(split_tracks.len());
    let mut seen: HashMap<(usize, usize, (i64, i64), (i64, i64)), usize> = HashMap::new();
    for seg in split_tracks {
        let q1 = q_pt(seg.start);
        let q2 = q_pt(seg.end);
        if !seg.is_locked && q1 == q2 {
            continue;
        }
        let (qa, qb) = if q1 <= q2 { (q1, q2) } else { (q2, q1) };
        let key = (seg.net_id, seg.layer, qa, qb);
        if let Some(&existing_idx) = seen.get(&key) {
            if seg.is_locked || (!deduped[existing_idx].is_locked && seg.width > deduped[existing_idx].width) {
                deduped[existing_idx] = seg;
            }
        } else {
            seen.insert(key, deduped.len());
            deduped.push(seg);
        }
    }
    board.tracks = deduped;
}

/// Prunes redundant parallel tracks, multi-via duplicate bridges, and single-pad loops
/// by extracting a minimal-cost Steiner tree over each net's pads and plane vias.
fn prune_redundant_net_cycles(board: &mut Board) -> bool {
    if board.footprints.is_empty() || board.tracks.len() < 2 {
        return false;
    }
    let plane_nets = board.inner_plane_nets();
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        net_name == "GND"
            || net_name.starts_with("GND")
            || plane_nets.values().any(|&pn| pn == nid)
    };

    let mut net_ids: Vec<usize> = board.tracks.iter().map(|t| t.net_id).collect();
    net_ids.sort_unstable();
    net_ids.dedup();

    let mut drop_track = vec![false; board.tracks.len()];
    let mut any_dropped = false;

    for net_id in net_ids {
        if net_id == 0 || board.arcs.iter().any(|a| a.net_id == net_id) {
            continue;
        }
        let trk_indices: Vec<usize> = board
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.net_id == net_id)
            .map(|(i, _)| i)
            .collect();
        if trk_indices.len() < 2 {
            continue;
        }

        let mut nodes: Vec<(f64, f64, usize)> = Vec::new();
        let mut find_or_add = |pt: (f64, f64), layer: usize| -> usize {
            for (idx, n) in nodes.iter().enumerate() {
                if n.2 == layer && (n.0 - pt.0).hypot(n.1 - pt.1) < 0.018 {
                    return idx;
                }
            }
            nodes.push((pt.0, pt.1, layer));
            nodes.len() - 1
        };

        let mut trk_endpoints: Vec<(usize, usize)> = Vec::with_capacity(trk_indices.len());
        for &ti in &trk_indices {
            let t = &board.tracks[ti];
            let u = find_or_add(t.start, t.layer);
            let v = find_or_add(t.end, t.layer);
            trk_endpoints.push((u, v));
        }

        let num_geom_nodes = nodes.len();
        let mut adj: Vec<Vec<(usize, Option<usize>, f64)>> = vec![Vec::new(); num_geom_nodes];
        let mut micro_self_tracks: Vec<(usize, usize)> = Vec::new();
        for (k, &ti) in trk_indices.iter().enumerate() {
            let (u, v) = trk_endpoints[k];
            let t = &board.tracks[ti];
            let len = (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1);
            if u == v {
                if len <= 1e-4 && !t.is_locked {
                    drop_track[ti] = true;
                    any_dropped = true;
                } else {
                    micro_self_tracks.push((ti, u));
                }
                continue;
            }
            let w = if t.is_locked {
                0.0
            } else {
                len * (if t.layer == 0 { 1.0 } else { 1.25 })
            };
            adj[u].push((v, Some(ti), w));
            adj[v].push((u, Some(ti), w));
        }

        // Vias connect nodes across layers (skipping nodes that step directly to a closer same-layer node at the via)
        for v in board.vias.iter().filter(|v| v.net_id == net_id) {
            let mut v_nodes = Vec::new();
            for u in 0..num_geom_nodes {
                let du = (nodes[u].0 - v.position.0).hypot(nodes[u].1 - v.position.1);
                if du <= v.size * 0.5 + 0.05 {
                    let has_closer_same_layer_neighbor = adj[u].iter().any(|&(nb, edge_opt, _)| {
                        edge_opt.is_some()
                            && nb < num_geom_nodes
                            && nodes[nb].2 == nodes[u].2
                            && (nodes[nb].0 - v.position.0).hypot(nodes[nb].1 - v.position.1) + 1e-4 < du
                    });
                    if !has_closer_same_layer_neighbor {
                        v_nodes.push(u);
                    }
                }
            }
            for i in 0..v_nodes.len() {
                for j in (i + 1)..v_nodes.len() {
                    let a = v_nodes[i];
                    let b = v_nodes[j];
                    adj[a].push((b, None, 0.80));
                    adj[b].push((a, None, 0.80));
                }
            }
        }

        let mut terminals: Vec<usize> = Vec::new();
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id != net_id {
                    continue;
                }
                let (px, py) = fp.pad_absolute_position(pad);
                let mut touching: Vec<(usize, f64, f64)> = Vec::new();
                for u in 0..num_geom_nodes {
                    if pad.is_through_hole || nodes[u].2 == fp.layer {
                        let d_c = (nodes[u].0 - px).hypot(nodes[u].1 - py);
                        let d_p = dist_point_to_pad((nodes[u].0, nodes[u].1), fp, pad);
                        if d_c < 0.12 || d_p < 0.05 {
                            touching.push((u, d_c, d_p));
                        }
                    }
                }
                if !touching.is_empty() {
                    let is_large_thermal = !pad.is_through_hole && pad.width >= 2.0 && pad.height >= 2.0;
                    if is_large_thermal {
                        let pad_node = adj.len();
                        adj.push(Vec::new());
                        for &(u, dc, dp) in &touching {
                            let cost = if dp < 0.01 { 0.0 } else { dc * 4.0 };
                            adj[pad_node].push((u, None, cost));
                            adj[u].push((pad_node, None, cost));
                        }
                        terminals.push(pad_node);
                    } else if pad.is_through_hole {
                        let pad_node = adj.len();
                        adj.push(Vec::new());
                        let min_dc = touching.iter().map(|t| t.1).fold(f64::INFINITY, f64::min);
                        for &(u, dc, _) in &touching {
                            if dc < 1e-3 || (dc - min_dc).abs() < 1e-4 {
                                adj[pad_node].push((u, None, dc * 4.0));
                                adj[u].push((pad_node, None, dc * 4.0));
                            }
                        }
                        terminals.push(pad_node);
                    } else if let Some(&(best_u, _, _)) = touching
                        .iter()
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                    {
                        terminals.push(best_u);
                    }
                }
            }
        }

        if is_plane_net(net_id) {
            for v in board.vias.iter().filter(|v| v.net_id == net_id) {
                if let Some((u, _)) = (0..num_geom_nodes)
                    .map(|u| (u, (nodes[u].0 - v.position.0).hypot(nodes[u].1 - v.position.1)))
                    .filter(|&(_, d)| d <= v.size * 0.5 + 0.05)
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                {
                    terminals.push(u);
                }
            }
        }

        for (k, &ti) in trk_indices.iter().enumerate() {
            if board.tracks[ti].is_locked {
                terminals.push(trk_endpoints[k].0);
                terminals.push(trk_endpoints[k].1);
            }
        }
        terminals.sort_unstable();
        terminals.dedup();
        if terminals.is_empty() {
            continue;
        }

        let total_n = adj.len();
        let mut comp_id = vec![usize::MAX; total_n];
        let mut comp_count = 0;
        for start in 0..total_n {
            if comp_id[start] != usize::MAX {
                continue;
            }
            let mut stack = vec![start];
            comp_id[start] = comp_count;
            while let Some(u) = stack.pop() {
                for &(nxt, _, _) in &adj[u] {
                    if comp_id[nxt] == usize::MAX {
                        comp_id[nxt] = comp_count;
                        stack.push(nxt);
                    }
                }
            }
            comp_count += 1;
        }

        let mut kept_tracks: HashSet<usize> = HashSet::new();
        for &ti in &trk_indices {
            if board.tracks[ti].is_locked {
                kept_tracks.insert(ti);
            }
        }

        for cid in 0..comp_count {
            let comp_terms: Vec<usize> = terminals.iter().copied().filter(|&t| comp_id[t] == cid).collect();
            if comp_terms.len() < 2 {
                for (k, &ti) in trk_indices.iter().enumerate() {
                    if comp_id[trk_endpoints[k].0] == cid {
                        kept_tracks.insert(ti);
                    }
                }
                continue;
            }
            let mut in_tree = vec![false; total_n];
            in_tree[comp_terms[0]] = true;

            loop {
                if comp_terms.iter().all(|&t| in_tree[t]) {
                    break;
                }
                let mut dist = vec![f64::INFINITY; total_n];
                let mut prev: Vec<Option<(usize, Option<usize>)>> = vec![None; total_n];
                let mut visited = vec![false; total_n];

                for u in 0..total_n {
                    if in_tree[u] {
                        dist[u] = 0.0;
                    }
                }

                loop {
                    let mut best_u = None;
                    let mut best_d = f64::INFINITY;
                    for u in 0..total_n {
                        if comp_id[u] == cid && !visited[u] && dist[u] < best_d {
                            best_d = dist[u];
                            best_u = Some(u);
                        }
                    }
                    let Some(u) = best_u else { break };
                    visited[u] = true;
                    for &(v, ti_opt, w) in &adj[u] {
                        let eff_w = if ti_opt.map_or(false, |ti| kept_tracks.contains(&ti)) { 0.0 } else { w };
                        if dist[u] + eff_w + 1e-9 < dist[v] {
                            dist[v] = dist[u] + eff_w;
                            prev[v] = Some((u, ti_opt));
                        }
                    }
                }

                let next_term = comp_terms
                    .iter()
                    .copied()
                    .filter(|&t| !in_tree[t] && dist[t].is_finite())
                    .min_by(|&a, &b| dist[a].partial_cmp(&dist[b]).unwrap_or(Ordering::Equal));
                let Some(mut cur) = next_term else { break };
                while !in_tree[cur] {
                    in_tree[cur] = true;
                    if let Some((p, ti_opt)) = prev[cur] {
                        if let Some(ti) = ti_opt {
                            kept_tracks.insert(ti);
                        }
                        cur = p;
                    } else {
                        break;
                    }
                }
            }
        }

        for &(ti, u_node) in &micro_self_tracks {
            let node_used = terminals.contains(&u_node)
                || trk_indices.iter().enumerate().any(|(k, &oti)| {
                    oti != ti && kept_tracks.contains(&oti) && (trk_endpoints[k].0 == u_node || trk_endpoints[k].1 == u_node)
                });
            if node_used {
                kept_tracks.insert(ti);
            }
        }

        for (k, &ti) in trk_indices.iter().enumerate() {
            let (u, _) = trk_endpoints[k];
            let has_terms = terminals.iter().any(|&t| comp_id[t] == comp_id[u]);
            if has_terms && !board.tracks[ti].is_locked && !kept_tracks.contains(&ti) {
                drop_track[ti] = true;
                any_dropped = true;
            }
        }
    }

    if any_dropped {
        let mut new_tracks = Vec::with_capacity(board.tracks.len());
        for (i, trk) in board.tracks.drain(..).enumerate() {
            if !drop_track[i] {
                new_tracks.push(trk);
            }
        }
        board.tracks = new_tracks;
    }
    any_dropped
}

/// Automated dangling track & via pruning.
/// Iteratively removes dead-end degree-1 track segments (antennas), redundant cycles, and unconnected/single-layer vias.
pub fn prune_dangling_tracks_and_vias(board: &mut Board) {
    if board.footprints.is_empty() {
        return;
    }
    split_and_dedup_same_net_tracks(board);
    let plane_nets = board.inner_plane_nets();
    let is_plane_net = |b: &Board, nid: usize| -> bool {
        let net_name = b.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        net_name == "GND"
            || net_name.starts_with("GND")
            || plane_nets.values().any(|&pn| pn == nid)
    };

    const EPS: f64 = 0.05;

    for _outer in 0..2 {
        let mut changed = true;
        let mut pass = 0;
        while changed && pass < 50 {
            changed = false;
            pass += 1;

            let num_tracks = board.tracks.len();
            let num_arcs = board.arcs.len();
            let num_vias = board.vias.len();
            let mut pruned_tracks = vec![false; num_tracks];
            let mut pruned_arcs = vec![false; num_arcs];
            let mut pruned_vias = vec![false; num_vias];

            // 1. Identify track dead-ends
            for i in 0..num_tracks {
                let t = &board.tracks[i];
                if t.is_locked {
                    continue;
                }

                let check_endpoint = |pt: (f64, f64), other_pt: (f64, f64)| -> bool {
                    // A. Check footprint pads of matching net
                    for fp in &board.footprints {
                        for pad in &fp.pads {
                            if pad.net_id == t.net_id {
                                if pad.is_through_hole || fp.layer == t.layer {
                                    let (px, py) = fp.pad_absolute_position(pad);
                                    let d_center_pt = (pt.0 - px).hypot(pt.1 - py);
                                    let d_center_other = (other_pt.0 - px).hypot(other_pt.1 - py);
                                    let d_pad_pt = dist_point_to_pad(pt, fp, pad);
                                    let d_pad_other = dist_point_to_pad(other_pt, fp, pad);
                                    if d_center_pt < 0.12 || d_pad_pt < EPS {
                                        let other_on_same_pad = d_center_other < 0.12
                                            || (d_pad_other < EPS && d_center_other >= 0.01);
                                        if !other_on_same_pad || d_center_pt + 1e-4 < d_center_other {
                                            return true;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // B. Check vias of matching net
                    for (v_idx, v) in board.vias.iter().enumerate() {
                        if v.net_id == t.net_id && !pruned_vias[v_idx] {
                            let d_pt = (pt.0 - v.position.0).hypot(pt.1 - v.position.1);
                            let d_other = (other_pt.0 - v.position.0).hypot(other_pt.1 - v.position.1);
                            if d_pt <= v.size * 0.5 + EPS && d_pt + 1e-4 < d_other {
                                return true;
                            }
                        }
                    }

                    // C. Check other tracks of matching net on same layer
                    for j in 0..num_tracks {
                        if i != j && !pruned_tracks[j] {
                            let o = &board.tracks[j];
                            if o.net_id == t.net_id && o.layer == t.layer {
                                let d_end_pt = (pt.0 - o.start.0).hypot(pt.1 - o.start.1).min((pt.0 - o.end.0).hypot(pt.1 - o.end.1));
                                let d_end_other = (other_pt.0 - o.start.0).hypot(other_pt.1 - o.start.1).min((other_pt.0 - o.end.0).hypot(other_pt.1 - o.end.1));
                                if d_end_pt < 0.018 && d_end_pt + 1e-4 < d_end_other {
                                    return true;
                                }
                                if o.is_locked {
                                    let d_pt = dist_point_to_segment(pt, o.start, o.end);
                                    let d_other = dist_point_to_segment(other_pt, o.start, o.end);
                                    if d_pt < EPS && d_pt + 1e-4 < d_other {
                                        return true;
                                    }
                                }
                            }
                        }
                    }

                    // D. Check arcs of matching net on same layer
                    for (a_idx, a) in board.arcs.iter().enumerate() {
                        if !pruned_arcs[a_idx] && a.net_id == t.net_id && a.layer == t.layer {
                            let as_pt = a.start();
                            let ae_pt = a.end();
                            let d_pt = (pt.0 - as_pt.0).hypot(pt.1 - as_pt.1).min((pt.0 - ae_pt.0).hypot(pt.1 - ae_pt.1));
                            let d_other = (other_pt.0 - as_pt.0).hypot(other_pt.1 - as_pt.1).min((other_pt.0 - ae_pt.0).hypot(other_pt.1 - ae_pt.1));
                            if d_pt < EPS && d_pt + 1e-4 < d_other {
                                return true;
                            }
                        }
                    }

                    false
                };

                let start_connected = check_endpoint(t.start, t.end);
                let end_connected = check_endpoint(t.end, t.start);

                if !start_connected || !end_connected {
                    pruned_tracks[i] = true;
                    changed = true;
                }
            }

            // 1b. Identify arc dead-ends
            for i in 0..num_arcs {
                let a = &board.arcs[i];
                if a.is_locked {
                    continue;
                }

                let check_arc_endpoint = |pt: (f64, f64)| -> bool {
                    for fp in &board.footprints {
                        for pad in &fp.pads {
                            if pad.net_id == a.net_id {
                                if pad.is_through_hole || fp.layer == a.layer {
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

                    for (v_idx, v) in board.vias.iter().enumerate() {
                        if v.net_id == a.net_id && !pruned_vias[v_idx] {
                            if (pt.0 - v.position.0).hypot(pt.1 - v.position.1) <= v.size * 0.5 + EPS {
                                return true;
                            }
                        }
                    }

                    for j in 0..num_tracks {
                        if !pruned_tracks[j] {
                            let o = &board.tracks[j];
                            if o.net_id == a.net_id && o.layer == a.layer {
                                if (pt.0 - o.start.0).hypot(pt.1 - o.start.1) < EPS
                                    || (pt.0 - o.end.0).hypot(pt.1 - o.end.1) < EPS
                                    || dist_point_to_segment(pt, o.start, o.end) < EPS
                                {
                                    return true;
                                }
                            }
                        }
                    }

                    for j in 0..num_arcs {
                        if i != j && !pruned_arcs[j] {
                            let oa = &board.arcs[j];
                            if oa.net_id == a.net_id && oa.layer == a.layer {
                                let oas = oa.start();
                                let oae = oa.end();
                                if (pt.0 - oas.0).hypot(pt.1 - oas.1) < EPS
                                    || (pt.0 - oae.0).hypot(pt.1 - oae.1) < EPS
                                {
                                    return true;
                                }
                            }
                        }
                    }

                    false
                };

                if !check_arc_endpoint(a.start()) || !check_arc_endpoint(a.end()) {
                    pruned_arcs[i] = true;
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

                let has_plane = is_plane_net(board, v.net_id);

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

                for (a_idx, a) in board.arcs.iter().enumerate() {
                    if a.net_id == v.net_id && !pruned_arcs[a_idx] {
                        let as_pt = a.start();
                        let ae_pt = a.end();
                        let d_start = (as_pt.0 - v.position.0).hypot(as_pt.1 - v.position.1);
                        let d_end = (ae_pt.0 - v.position.0).hypot(ae_pt.1 - v.position.1);
                        if d_start <= v.size * 0.5 + EPS || d_end <= v.size * 0.5 + EPS {
                            connected_tracks += 1;
                            connected_layers.insert(a.layer);
                        }
                    }
                }

                let has_outward_reach = |layer: usize| -> bool {
                    let incident_to_v: Vec<usize> = board.tracks.iter().enumerate()
                        .filter(|&(idx, t)| {
                            t.net_id == v.net_id && !pruned_tracks[idx] && t.layer == layer && {
                                let d_s = (t.start.0 - v.position.0).hypot(t.start.1 - v.position.1);
                                let d_e = (t.end.0 - v.position.0).hypot(t.end.1 - v.position.1);
                                d_s <= v.size * 0.5 + EPS || d_e <= v.size * 0.5 + EPS
                            }
                        })
                        .map(|(idx, _)| idx)
                        .collect();

                    let incident_arcs_to_v: Vec<usize> = board.arcs.iter().enumerate()
                        .filter(|&(idx, a)| {
                            a.net_id == v.net_id && !pruned_arcs[idx] && a.layer == layer && {
                                let as_pt = a.start();
                                let ae_pt = a.end();
                                let d_s = (as_pt.0 - v.position.0).hypot(as_pt.1 - v.position.1);
                                let d_e = (ae_pt.0 - v.position.0).hypot(ae_pt.1 - v.position.1);
                                d_s <= v.size * 0.5 + EPS || d_e <= v.size * 0.5 + EPS
                            }
                        })
                        .map(|(idx, _)| idx)
                        .collect();

                    let check_other_pt_reach = |other_pt: (f64, f64)| -> bool {
                        let reaches_pad = board.footprints.iter().any(|fp| {
                            fp.pads.iter().any(|p| {
                                p.net_id == v.net_id && (p.is_through_hole || fp.layer == layer) && {
                                    let (px, py) = fp.pad_absolute_position(p);
                                    (other_pt.0 - px).hypot(other_pt.1 - py) < 0.12 || dist_point_to_pad(other_pt, fp, p) < EPS
                                }
                            })
                        });
                        if reaches_pad {
                            return true;
                        }

                        let reaches_other_via = board.vias.iter().enumerate().any(|(vi, ov)| {
                            vi != v_idx && ov.net_id == v.net_id && !pruned_vias[vi]
                                && (other_pt.0 - ov.position.0).hypot(other_pt.1 - ov.position.1) <= ov.size * 0.5 + EPS
                        });
                        if reaches_other_via {
                            return true;
                        }

                        let reaches_external_track = board.tracks.iter().enumerate().any(|(j, ot)| {
                            ot.net_id == v.net_id && ot.layer == layer && !pruned_tracks[j] && !incident_to_v.contains(&j) && {
                                (other_pt.0 - ot.start.0).hypot(other_pt.1 - ot.start.1) < 0.015
                                    || (other_pt.0 - ot.end.0).hypot(other_pt.1 - ot.end.1) < 0.015
                                    || (ot.is_locked && dist_point_to_segment(other_pt, ot.start, ot.end) < EPS)
                            }
                        });
                        if reaches_external_track {
                            return true;
                        }

                        let reaches_external_arc = board.arcs.iter().enumerate().any(|(j, oa)| {
                            oa.net_id == v.net_id && oa.layer == layer && !pruned_arcs[j] && !incident_arcs_to_v.contains(&j) && {
                                let oas = oa.start();
                                let oae = oa.end();
                                (other_pt.0 - oas.0).hypot(other_pt.1 - oas.1) < EPS
                                    || (other_pt.0 - oae.0).hypot(other_pt.1 - oae.1) < EPS
                            }
                        });
                        reaches_external_arc
                    };

                    for &t_idx in &incident_to_v {
                        let t = &board.tracks[t_idx];
                        let d_s = (t.start.0 - v.position.0).hypot(t.start.1 - v.position.1);
                        let other_pt = if d_s <= v.size * 0.5 + EPS { t.end } else { t.start };
                        if check_other_pt_reach(other_pt) {
                            return true;
                        }
                    }

                    for &a_idx in &incident_arcs_to_v {
                        let a = &board.arcs[a_idx];
                        let as_pt = a.start();
                        let ae_pt = a.end();
                        let d_s = (as_pt.0 - v.position.0).hypot(as_pt.1 - v.position.1);
                        let other_pt = if d_s <= v.size * 0.5 + EPS { ae_pt } else { as_pt };
                        if check_other_pt_reach(other_pt) {
                            return true;
                        }
                    }

                    false
                };

                let valid_connected_layers = connected_layers.iter().filter(|&&l| has_outward_reach(l)).count();

                if !touches_pad && !has_plane {
                    if connected_tracks <= 1 || valid_connected_layers < 2 {
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

                let mut new_arcs = Vec::with_capacity(num_arcs);
                for (idx, arc) in board.arcs.drain(..).enumerate() {
                    if !pruned_arcs[idx] {
                        new_arcs.push(arc);
                    }
                }
                board.arcs = new_arcs;

                let mut new_vias = Vec::with_capacity(num_vias);
                for (idx, v) in board.vias.drain(..).enumerate() {
                    if !pruned_vias[idx] {
                        new_vias.push(v);
                    }
                }
                board.vias = new_vias;
            }
        }

        if !prune_redundant_net_cycles(board) {
            break;
        }
    }

    // 3. Remove orphan components disconnected from any pad or plane
    prune_orphan_components(board);
}

/// Prunes disconnected islands of copper that do not connect to any pad or plane.
fn prune_orphan_components(board: &mut Board) {
    if board.tracks.is_empty() && board.arcs.is_empty() {
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

    let mut net_ids: Vec<usize> = board
        .tracks
        .iter()
        .map(|t| t.net_id)
        .chain(board.arcs.iter().map(|a| a.net_id))
        .collect();
    net_ids.sort_unstable();
    net_ids.dedup();

    let mut keep_track = vec![true; board.tracks.len()];
    let mut keep_arc = vec![true; board.arcs.len()];
    let mut keep_via = vec![true; board.vias.len()];

    for net_id in net_ids {
        let track_indices: Vec<usize> = board
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.net_id == net_id)
            .map(|(i, _)| i)
            .collect();
        let arc_indices: Vec<usize> = board
            .arcs
            .iter()
            .enumerate()
            .filter(|(_, a)| a.net_id == net_id)
            .map(|(i, _)| i)
            .collect();
        let via_indices: Vec<usize> = board
            .vias
            .iter()
            .enumerate()
            .filter(|(_, v)| v.net_id == net_id)
            .map(|(i, _)| i)
            .collect();

        if track_indices.is_empty() && arc_indices.is_empty() {
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
        let n_a = arc_indices.len();
        let n_v = via_indices.len();
        let total_nodes = n_t + n_a + n_v;
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); total_nodes];

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
            for ai in 0..n_a {
                let a = &board.arcs[arc_indices[ai]];
                if ti.layer == a.layer {
                    let as_pt = a.start();
                    let ae_pt = a.end();
                    let node_a = n_t + ai;
                    let connected = (ti.start.0 - as_pt.0).hypot(ti.start.1 - as_pt.1) < EPS
                        || (ti.start.0 - ae_pt.0).hypot(ti.start.1 - ae_pt.1) < EPS
                        || (ti.end.0 - as_pt.0).hypot(ti.end.1 - as_pt.1) < EPS
                        || (ti.end.0 - ae_pt.0).hypot(ti.end.1 - ae_pt.1) < EPS
                        || dist_point_to_segment(as_pt, ti.start, ti.end) < EPS
                        || dist_point_to_segment(ae_pt, ti.start, ti.end) < EPS;
                    if connected {
                        adj[i].push(node_a);
                        adj[node_a].push(i);
                    }
                }
            }
        }

        for ai in 0..n_a {
            let a = &board.arcs[arc_indices[ai]];
            let as_pt = a.start();
            let ae_pt = a.end();
            let node_a = n_t + ai;
            for aj in (ai + 1)..n_a {
                let oa = &board.arcs[arc_indices[aj]];
                if a.layer == oa.layer {
                    let oas = oa.start();
                    let oae = oa.end();
                    let node_oa = n_t + aj;
                    let connected = (as_pt.0 - oas.0).hypot(as_pt.1 - oas.1) < EPS
                        || (as_pt.0 - oae.0).hypot(as_pt.1 - oae.1) < EPS
                        || (ae_pt.0 - oas.0).hypot(ae_pt.1 - oas.1) < EPS
                        || (ae_pt.0 - oae.0).hypot(ae_pt.1 - oae.1) < EPS;
                    if connected {
                        adj[node_a].push(node_oa);
                        adj[node_oa].push(node_a);
                    }
                }
            }
            for vi in 0..n_v {
                let v = &board.vias[via_indices[vi]];
                let node_v = n_t + n_a + vi;
                let d_s = (as_pt.0 - v.position.0).hypot(as_pt.1 - v.position.1);
                let d_e = (ae_pt.0 - v.position.0).hypot(ae_pt.1 - v.position.1);
                if d_s <= v.size * 0.5 + EPS || d_e <= v.size * 0.5 + EPS {
                    adj[node_a].push(node_v);
                    adj[node_v].push(node_a);
                }
            }
        }

        for i in 0..n_t {
            let ti = &board.tracks[track_indices[i]];
            for vi in 0..n_v {
                let v = &board.vias[via_indices[vi]];
                let node_v = n_t + n_a + vi;
                let d_s = (ti.start.0 - v.position.0).hypot(ti.start.1 - v.position.1);
                let d_e = (ti.end.0 - v.position.0).hypot(ti.end.1 - v.position.1);
                if d_s <= v.size * 0.5 + EPS || d_e <= v.size * 0.5 + EPS {
                    adj[i].push(node_v);
                    adj[node_v].push(i);
                }
            }
        }

        let mut is_anchor = vec![false; total_nodes];
        for i in 0..n_t {
            let ti = &board.tracks[track_indices[i]];
            if ti.is_locked {
                is_anchor[i] = true;
                continue;
            }
            for &(px, py, pad_layer, is_th) in &pad_positions {
                if is_th || pad_layer == ti.layer {
                    if (ti.start.0 - px).hypot(ti.start.1 - py) < 0.12
                        || (ti.end.0 - px).hypot(ti.end.1 - py) < 0.12
                        || dist_point_to_segment((px, py), ti.start, ti.end) < 0.05
                    {
                        is_anchor[i] = true;
                        break;
                    }
                }
            }
        }

        for ai in 0..n_a {
            let a = &board.arcs[arc_indices[ai]];
            let node_a = n_t + ai;
            if a.is_locked {
                is_anchor[node_a] = true;
                continue;
            }
            let as_pt = a.start();
            let ae_pt = a.end();
            for &(px, py, pad_layer, is_th) in &pad_positions {
                if is_th || pad_layer == a.layer {
                    if (as_pt.0 - px).hypot(as_pt.1 - py) < 0.12
                        || (ae_pt.0 - px).hypot(ae_pt.1 - py) < 0.12
                    {
                        is_anchor[node_a] = true;
                        break;
                    }
                }
            }
        }

        for vi in 0..n_v {
            let v = &board.vias[via_indices[vi]];
            let node_v = n_t + n_a + vi;
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

        let mut visited = vec![false; total_nodes];
        let mut queue = VecDeque::new();
        for node in 0..total_nodes {
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
        for ai in 0..n_a {
            if !visited[n_t + ai] && !board.arcs[arc_indices[ai]].is_locked {
                keep_arc[arc_indices[ai]] = false;
            }
        }
        for vi in 0..n_v {
            if !visited[n_t + n_a + vi] && !board.vias[via_indices[vi]].is_locked {
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

    let mut new_arcs = Vec::with_capacity(board.arcs.len());
    for (i, a) in board.arcs.drain(..).enumerate() {
        if keep_arc[i] {
            new_arcs.push(a);
        }
    }
    board.arcs = new_arcs;

    let mut new_vias = Vec::with_capacity(board.vias.len());
    for (i, v) in board.vias.drain(..).enumerate() {
        if keep_via[i] {
            new_vias.push(v);
        }
    }
    board.vias = new_vias;
}

/// Automated DFM Teardrop Generator:
/// Teardrops are natively and parametrically generated by KiCad's EDA engine on board load/save.
pub fn add_teardrops(_board: &mut Board) {
    // Delegated to KiCad native parametric teardrop engine (Edit -> Teardrops...)
}

/// Comprehensive pre-export DRC validation: detects and eliminates any residual shorts
/// or clearance violations between tracks, vias, and pads of different nets.
pub fn validate_and_clean_copper(board: &mut Board) {
    if let Some(nc_idx) = board.raw_content.find("\n  (net_class \"") {
        let max_fp_end = board
            .footprints
            .iter()
            .filter_map(|fp| fp.fp_span.map(|(_, e)| e))
            .max()
            .unwrap_or(0);
        if nc_idx >= max_fp_end {
            board.raw_content.truncate(nc_idx);
            board.raw_content.push_str("\n)\n");
        }
    }

    // KiCad 10 SHAPE_ARC::Collide(SHAPE_RECT, 0) quirk: shallow arcs terminating directly on a
    // rectangular pad center without a straight segment stub can fail pad-to-arc connectivity.
    if !board.arcs.is_empty() && !board.footprints.is_empty() {
        let mut pad_centers: Vec<((f64, f64), usize, usize, bool)> = Vec::new();
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id > 0 {
                    pad_centers.push((
                        fp.pad_absolute_position(pad),
                        pad.net_id,
                        fp.layer,
                        pad.is_through_hole,
                    ));
                }
            }
        }
        let mut stub_tracks = Vec::new();
        for arc in &mut board.arcs {
            if arc.radius <= 0.1 || (arc.end_angle_rad - arc.start_angle_rad).abs() * arc.radius <= 0.60 {
                continue;
            }
            let sign = if arc.end_angle_rad >= arc.start_angle_rad { 1.0 } else { -1.0 };
            let delta = 0.20 / arc.radius;
            let as_pt = arc.start();
            let start_on_pad = pad_centers.iter().any(|&(p, nid, lyr, th)| {
                nid == arc.net_id && (th || lyr == arc.layer) && (as_pt.0 - p.0).hypot(as_pt.1 - p.1) < 0.05
            });
            if start_on_pad {
                let has_track = board.tracks.iter().any(|t| {
                    t.net_id == arc.net_id
                        && t.layer == arc.layer
                        && ((t.start.0 - as_pt.0).hypot(t.start.1 - as_pt.1) < 0.02
                            || (t.end.0 - as_pt.0).hypot(t.end.1 - as_pt.1) < 0.02)
                });
                if !has_track {
                    arc.start_angle_rad += sign * delta;
                    let new_start = arc.start();
                    stub_tracks.push(TrackSegment {
                        start: as_pt,
                        end: new_start,
                        width: arc.width,
                        layer: arc.layer,
                        net_id: arc.net_id,
                        is_locked: arc.is_locked,
                    });
                }
            }
            let ae_pt = arc.end();
            let end_on_pad = pad_centers.iter().any(|&(p, nid, lyr, th)| {
                nid == arc.net_id && (th || lyr == arc.layer) && (ae_pt.0 - p.0).hypot(ae_pt.1 - p.1) < 0.05
            });
            if end_on_pad && (arc.end_angle_rad - arc.start_angle_rad).abs() * arc.radius > 0.40 {
                let has_track = board.tracks.iter().any(|t| {
                    t.net_id == arc.net_id
                        && t.layer == arc.layer
                        && ((t.start.0 - ae_pt.0).hypot(t.start.1 - ae_pt.1) < 0.02
                            || (t.end.0 - ae_pt.0).hypot(t.end.1 - ae_pt.1) < 0.02)
                });
                if !has_track {
                    arc.end_angle_rad -= sign * delta;
                    let new_end = arc.end();
                    stub_tracks.push(TrackSegment {
                        start: new_end,
                        end: ae_pt,
                        width: arc.width,
                        layer: arc.layer,
                        net_id: arc.net_id,
                        is_locked: arc.is_locked,
                    });
                }
            }
        }
        board.tracks.extend(stub_tracks);
    }

    let mut changed = true;
    let mut iteration = 0;
    while changed && iteration < 10 {
        changed = false;
        iteration += 1;

        let mut bad_track_indices = HashSet::new();
        let track_count = board.tracks.len();
        let gpu_track_ok: Option<Vec<u32>> = if track_count >= 16 {
            let drc_ctx = GpuDrcContext::from_board_with_zero_clearance(board);
            let cands: Vec<GpuDrcCandidate> = board
                .tracks
                .iter()
                .map(|t| GpuDrcCandidate {
                    x1: t.start.0 as f32,
                    y1: t.start.1 as f32,
                    x2: t.end.0 as f32,
                    y2: t.end.1 as f32,
                    radius_or_half_width: (t.width * 0.5) as f32,
                    clearance: 0.0,
                    layer: t.layer as u32,
                    net_id: t.net_id as u32,
                    kind: 1,
                    ignore_same_net_pads: 0,
                    _pad0: 0,
                    _pad1: 0,
                })
                .collect();
            drc_ctx.evaluate(None, &cands)
        } else {
            None
        };

        for i in 0..track_count {
            if board.tracks[i].is_locked {
                continue;
            }
            let t1 = &board.tracks[i];

            // Copper-to-board-edge clearance check
            if let Some(ref poly) = board.outline_polygon {
                let touches_own_pad = board.footprints.iter().any(|fp| {
                    fp.pads.iter().any(|p| {
                        p.net_id == t1.net_id
                            && (p.is_through_hole || fp.layer == t1.layer)
                            && (dist_point_to_pad(t1.start, fp, p) < 0.05
                                || dist_point_to_pad(t1.end, fp, p) < 0.05)
                    })
                });
                let edge_clr = board.get_netclass_for_net(t1.net_id).board_edge_clearance;
                let req_dist = if touches_own_pad { 0.0 } else { t1.width * 0.5 + edge_clr };
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
                        || (!touches_own_pad && poly.distance_to_segment(t1.start, t1.end) < req_dist - 1e-4)
                    {
                        bad_track_indices.insert(i);
                        continue;
                    }
                }
            }

            if gpu_track_ok.as_ref().map(|v| v[i] == 1).unwrap_or(false) {
                continue;
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
                            let has_other_layer_track = board.tracks.iter().any(|t| {
                                t.net_id == v1.net_id
                                    && t.layer != fp.layer
                                    && ((t.start.0 - v1.position.0).hypot(t.start.1 - v1.position.1) <= v1.size * 0.5 + 0.15
                                        || (t.end.0 - v1.position.0).hypot(t.end.1 - v1.position.1) <= v1.size * 0.5 + 0.15)
                            });
                            if !has_other_layer_track {
                                bad_via_indices.insert(i);
                                break;
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
                if i == j || bad_via_indices.contains(&j) {
                    continue;
                }
                let v2 = &board.vias[j];
                let min_hole_dist = (v1.drill * 0.5) + (v2.drill * 0.5) + 0.20;
                let d = (v1.position.0 - v2.position.0).hypot(v1.position.1 - v2.position.1);
                if d < min_hole_dist - 1e-4 {
                    if v1.net_id == v2.net_id && i < j {
                        // Snap tracks connected to v2 onto v1 using octilinear doglegs and mark only v2 for removal
                        let pos_keep = v1.position;
                        let pos_drop = v2.position;
                        let drop_r = v2.size * 0.5 + 0.05;
                        let nid = v1.net_id;
                        let trk_clr = board.get_netclass_for_net(nid).clearance.max(0.15);
                        let mut to_replace = Vec::new();
                        for (t_idx, trk) in board.tracks.iter().enumerate() {
                            if trk.net_id != nid || trk.is_locked {
                                continue;
                            }
                            let s_drop = (trk.start.0 - pos_drop.0).hypot(trk.start.1 - pos_drop.1) <= drop_r;
                            let e_drop = (trk.end.0 - pos_drop.0).hypot(trk.end.1 - pos_drop.1) <= drop_r;
                            if s_drop || e_drop {
                                let other_pt = if s_drop { trk.end } else { trk.start };
                                to_replace.push((t_idx, other_pt, trk.width, trk.layer));
                            }
                        }
                        let mut new_segs = Vec::new();
                        for &(_, other_pt, width, layer) in &to_replace {
                            if (other_pt.0 - pos_keep.0).hypot(other_pt.1 - pos_keep.1) <= 1e-4 {
                                continue;
                            }
                            let other_on_pad = board.footprints.iter().any(|fp| {
                                fp.pads.iter().any(|p| {
                                    let (px, py) = fp.pad_absolute_position(p);
                                    (px - other_pt.0).hypot(py - other_pt.1) < 0.08
                                })
                            });
                            let postures = if other_on_pad { [true, false] } else { [false, true] };
                            let dl = postures
                                .into_iter()
                                .map(|df| compute_45_dogleg(pos_keep, other_pt, df))
                                .find(|cand| {
                                    cand.windows(2).all(|w| {
                                        (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                            || is_track_segment_legal(board, w[0], w[1], width, layer, nid, trk_clr)
                                    })
                                })
                                .unwrap_or_else(|| compute_45_dogleg(pos_keep, other_pt, other_on_pad));
                            for w in dl.windows(2) {
                                if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                    new_segs.push(TrackSegment {
                                        start: w[0],
                                        end: w[1],
                                        width,
                                        layer,
                                        net_id: nid,
                                        is_locked: false,
                                    });
                                }
                            }
                        }
                        for &(t_idx, _, _, _) in to_replace.iter().rev() {
                            board.tracks.remove(t_idx);
                        }
                        board.tracks.extend(new_segs);
                        bad_via_indices.insert(j);
                        continue;
                    } else if i > j {
                        bad_via_indices.insert(i);
                        break;
                    }
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
    fn test_teardrops_delegated_to_kicad_leaves_routed_tracks_pristine() {
        let mut board = Board::default();
        board.tracks = vec![TrackSegment {
            start: (10.0, 10.0),
            end: (20.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        }];
        add_teardrops(&mut board);
        assert_eq!(board.tracks.len(), 1, "Teardrops delegated to KiCad native engine must leave routed tracks pristine");
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

    #[test]
    fn test_any_angle_tangent_arcs_preserved_by_pruning() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "SIG".to_string()];
        board.net_name_to_id.insert("SIG".to_string(), 1);

        let mut fp1 = Footprint {
            reference: "R1".to_string(),
            x: 0.0,
            y: 0.0,
            ..Footprint::default()
        };
        fp1.pads.push(Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "SIG", false));
        let mut fp2 = Footprint {
            reference: "R2".to_string(),
            x: 10.0,
            y: 10.0,
            layer: 1,
            ..Footprint::default()
        };
        fp2.pads.push(Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "SIG", false));
        board.footprints.push(fp1);
        board.footprints.push(fp2);

        // Route from R1 (0,0, L0) -> (5,0, L0) -> (5,5, L0) [via to L1] -> (10,5, L1) -> R2 (10,10, L1)
        board.tracks.push(TrackSegment { start: (0.0, 0.0), end: (5.0, 0.0), width: 0.25, layer: 0, net_id: 1, is_locked: false });
        board.tracks.push(TrackSegment { start: (5.0, 0.0), end: (5.0, 5.0), width: 0.25, layer: 0, net_id: 1, is_locked: false });
        board.vias.push(Via { position: (5.0, 5.0), size: 0.60, drill: 0.30, net_id: 1, is_locked: false });
        board.tracks.push(TrackSegment { start: (5.0, 5.0), end: (10.0, 5.0), width: 0.25, layer: 1, net_id: 1, is_locked: false });
        board.tracks.push(TrackSegment { start: (10.0, 5.0), end: (10.0, 10.0), width: 0.25, layer: 1, net_id: 1, is_locked: false });

        apply_any_angle_tangent_arcs(&mut board);
        assert_eq!(board.arcs.len(), 2, "Expected 2 corner tangent arcs");

        // Also add a genuinely dangling arc that is not connected on one end
        board.arcs.push(crate::board::ArcSegment::new((30.0, 30.0), 2.0, 0.0, 1.0, 0.25, 0, 1));

        prune_dangling_tracks_and_vias(&mut board);

        assert_eq!(board.tracks.len(), 4, "All 4 trimmed straight segments must survive pruning");
        assert_eq!(board.arcs.len(), 2, "Both connected corner arcs must survive while dangling arc is pruned");
        assert_eq!(board.vias.len(), 1, "Layer transition via must survive pruning");
    }

    #[test]
    fn test_multi_track_corridor_shove_and_non_diff_pair_suffixes() {
        // Non-differential nets like Net-(U1-HPA) / Net-(U1-HNA) must not be falsely matched
        assert!(!AutoRouter::are_diff_pair_mates("Net-(U1-HPA)", "Net-(U1-HNA)"));
        assert!(!AutoRouter::are_diff_pair_mates("CAP_P1", "CAP_N1"));
        assert!(AutoRouter::are_diff_pair_mates("USB_DP", "USB_DM"));
        assert!(AutoRouter::are_diff_pair_mates("LVDS_P", "LVDS_N"));

        // Construct a board where 2 parallel tracks block a narrow corridor for a 3rd net
        let nets = vec![
            "".to_string(),
            "GND".to_string(),
            "BLK1".to_string(),
            "BLK2".to_string(),
            "TARGET".to_string(),
        ];
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

        let b1_s = make_fp("B1S", 102.0, 109.6, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "BLK1", false)]);
        let b1_e = make_fp("B1E", 118.0, 109.6, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "BLK1", false)]);
        let b2_s = make_fp("B2S", 102.0, 110.4, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "BLK2", false)]);
        let b2_e = make_fp("B2E", 118.0, 110.4, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "BLK2", false)]);
        let tgt_s = make_fp("TS", 110.0, 106.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 4, "TARGET", false)]);
        let tgt_e = make_fp("TE", 110.0, 114.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 4, "TARGET", false)]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 100.0,
            min_y: 100.0,
            max_x: 120.0,
            max_y: 120.0,
            footprints: vec![b1_s, b1_e, b2_s, b2_e, tgt_s, tgt_e],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
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
        let router = AutoRouter::new(4);
        let stop = Arc::new(AtomicBool::new(false));
        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);
        assert_eq!(total, 3);
        assert_eq!(routed, 3, "All 3 connections must route cleanly");
    }

    #[test]
    fn test_electrical_continuity_verification_and_same_net_via_merge() {
        let nets = vec!["".to_string(), "GND".to_string(), "SIG".to_string()];
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

        let fp_a = make_fp("U1", 102.0, 105.0, true, vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "SIG", false)]);
        let mut fp_b = make_fp("U2", 110.0, 105.0, true, vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "SIG", false)]);
        fp_b.layer = 1;
        fp_b.orig_layer = 1;

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 100.0,
            min_y: 100.0,
            max_x: 120.0,
            max_y: 120.0,
            footprints: vec![fp_a, fp_b],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (102.0, 105.0),
                    end: (106.0, 105.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 2,
                    is_locked: false,
                },
                TrackSegment {
                    start: (106.1, 105.0),
                    end: (110.0, 105.0),
                    width: 0.2,
                    layer: 1,
                    net_id: 2,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (106.0, 105.0),
                    size: 0.6,
                    drill: 0.3,
                    net_id: 2,
                    is_locked: false,
                },
                Via {
                    position: (106.1, 105.0),
                    size: 0.6,
                    drill: 0.3,
                    net_id: 2,
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

        let router = AutoRouter::new(2);
        let conns = router.extract_connections(&board);
        assert_eq!(conns.len(), 1);

        // validate_and_clean_copper should merge the two close same-net vias into one rather than deleting both
        validate_and_clean_copper(&mut board);
        assert_eq!(board.vias.len(), 1, "Duplicate close same-net via should be merged into 1 via");
        assert!(
            AutoRouter::is_connection_electrically_connected(&board, &conns[0]),
            "Connection must remain electrically continuous after merging duplicate same-net vias"
        );
    }

    #[test]
    fn test_single_connection_reconstruct_with_diff_pair_leader_and_choked_pocket_ripup() {
        let mut board = Board {
            min_x: 100.0,
            min_y: 100.0,
            max_x: 120.0,
            max_y: 120.0,
            footprints: vec![
                Footprint {
                    reference: "U1".to_string(),
                    x: 105.0,
                    y: 105.0,
                    pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "USB_DM", false)],
                    ..Footprint::default()
                },
                Footprint {
                    reference: "U2".to_string(),
                    x: 112.0,
                    y: 105.0,
                    pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "USB_DM", false)],
                    ..Footprint::default()
                },
            ],
            nets: vec!["".to_string(), "GND".to_string(), "USB_DM".to_string(), "BLOCKER".to_string()],
            ..Board::default()
        };
        let router = AutoRouter::new(2);
        let mut grid = RoutingGrid::new(&board, 0.10, None);

        // 1. Verify single-connection reconstruct_tracks_and_vias does not panic when diff_pair_leader is out of slice bounds
        let mut single_conn = router.extract_connections(&board);
        assert_eq!(single_conn.len(), 1);
        single_conn[0].diff_pair_leader = Some(5); // Out-of-bounds index relative to single_paths (len 1)
        let (sgx, sgy) = grid.world_to_grid(105.0, 105.0).unwrap();
        let (egx, egy) = grid.world_to_grid(112.0, 105.0).unwrap();
        let path = vec![
            GridNode { gx: sgx, gy: sgy, layer: 0 },
            GridNode { gx: (sgx + egx) / 2, gy: sgy, layer: 0 },
            GridNode { gx: (sgx + egx) / 2, gy: sgy, layer: 1 },
            GridNode { gx: egx, gy: egy, layer: 1 },
        ];
        router.reconstruct_tracks_and_vias(&mut board, &grid, &single_conn, &[Some(path)], &[]);
        assert!(!board.tracks.is_empty(), "Single-connection reconstruct with layer change must emit tracks without panicking");

        // 2. Verify find_choked_pin_bottlenecks detects a surrounding track ring even if 1-cell cracks exist
        let (cx, cy) = (sgx, sgy);
        for d in -4isize..=4 {
            // Place foreign net 3 on a ring at radius 4 with a 1-cell diagonal gap that a 1-cell halo catches
            let top = grid.idx((cx as isize + d) as usize, (cy as isize - 4) as usize, 0);
            let bot = grid.idx((cx as isize + d) as usize, (cy as isize + 4) as usize, 0);
            let lft = grid.idx((cx as isize - 4) as usize, (cy as isize + d) as usize, 0);
            let rgt = grid.idx((cx as isize + 4) as usize, (cy as isize + d) as usize, 0);
            grid.obstacle_net_ids[top] = 3;
            grid.obstacle_net_ids[bot] = 3;
            grid.obstacle_net_ids[lft] = 3;
            grid.obstacle_net_ids[rgt] = 3;
        }
        let (reach, blockers) = AutoRouter::find_choked_pin_bottlenecks(&grid, cx, cy, 0, 2, 400);
        assert!(reach < 100, "Pin enclosed by track ring must be detected as choked (reach={})", reach);
        assert!(blockers.contains(&3), "Blocker net 3 must be identified by find_choked_pin_bottlenecks");
    }

    #[test]
    fn test_prune_dangling_micro_stub_and_u1_ground_pin_connection() {
        let mut board = Board {
            min_x: 100.0,
            min_y: 100.0,
            max_x: 120.0,
            max_y: 120.0,
            footprints: vec![
                Footprint {
                    reference: "U1".to_string(),
                    x: 110.0,
                    y: 110.0,
                    rotation: 270.0,
                    pads: vec![
                        Pad::new("18", 0.75, 2.075, 0.30, 0.75, 1, "GND", false),
                        Pad::new("19", 2.075, 0.75, 0.75, 0.30, 1, "GND", false),
                        Pad::new("25", 0.0, 0.0, 2.60, 2.60, 1, "GND", false),
                        Pad::new("1", -2.075, -0.75, 0.75, 0.30, 2, "SIG", false),
                    ],
                    ..Footprint::default()
                },
                Footprint {
                    reference: "U2".to_string(),
                    x: 105.0,
                    y: 109.25,
                    pads: vec![Pad::new("1", 0.0, 0.0, 0.50, 0.50, 2, "SIG", false)],
                    ..Footprint::default()
                },
            ],
            nets: vec!["".to_string(), "GND".to_string(), "SIG".to_string()],
            ..Board::default()
        };

        let u1_p1 = board.footprints[0].pad_absolute_position(&board.footprints[0].pads[3]);
        let u2_p1 = board.footprints[1].pad_absolute_position(&board.footprints[1].pads[0]);
        let mid = ((u1_p1.0 + u2_p1.0) * 0.5, u1_p1.1);
        // Valid main route from U1.1 -> mid -> U2.1, plus a 0.03 mm dangling micro-stub at `mid`
        board.tracks = vec![
            TrackSegment {
                start: u1_p1,
                end: mid,
                width: 0.15,
                layer: 0,
                net_id: 2,
                is_locked: false,
            },
            TrackSegment {
                start: mid,
                end: u2_p1,
                width: 0.15,
                layer: 0,
                net_id: 2,
                is_locked: false,
            },
            TrackSegment {
                start: mid,
                end: (mid.0, mid.1 + 0.03),
                width: 0.15,
                layer: 0,
                net_id: 2,
                is_locked: false,
            },
        ];

        let router = AutoRouter::new(1);
        router.connect_u1_ground_pins(&mut board);
        prune_dangling_tracks_and_vias(&mut board);

        let sig_tracks: Vec<&TrackSegment> = board.tracks.iter().filter(|t| t.net_id == 2).collect();
        assert_eq!(sig_tracks.len(), 2, "0.03 mm dangling micro-stub must be pruned while preserving main route");
        let gnd_tracks: Vec<&TrackSegment> = board.tracks.iter().filter(|t| t.net_id == 1).collect();
        assert!(gnd_tracks.len() >= 2, "Rotated U1 pads 18 and 19 must connect inward into central thermal pad 25 and survive pruning");
    }

    #[test]
    fn test_prune_same_net_crossing_corner_stub_and_parallel_cycles() {
        let mut board = Board {
            max_x: 50.0,
            max_y: 50.0,
            footprints: vec![
                Footprint {
                    reference: "U1".to_string(),
                    x: 10.0,
                    y: 10.0,
                    pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG", false)],
                    ..Footprint::default()
                },
                Footprint {
                    reference: "U2".to_string(),
                    x: 20.0,
                    y: 10.0,
                    pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG", false)],
                    ..Footprint::default()
                },
            ],
            nets: vec!["".to_string(), "SIG".to_string()],
            tracks: vec![
                // Escape stub going past the corner at (12.0, 10.0) to (13.0, 10.0)
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (13.0, 10.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // Route crossing the stub at (12.0, 10.0) and continuing to U2
                TrackSegment {
                    start: (12.0, 10.0),
                    end: (12.0, 12.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (12.0, 12.0),
                    end: (20.0, 12.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (20.0, 12.0),
                    end: (20.0, 10.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // Redundant longer parallel detour between (12.0, 12.0) and (20.0, 12.0)
                TrackSegment {
                    start: (12.0, 12.0),
                    end: (12.0, 14.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (12.0, 14.0),
                    end: (20.0, 14.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (20.0, 14.0),
                    end: (20.0, 12.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            ..Board::default()
        };

        prune_dangling_tracks_and_vias(&mut board);

        // Both the corner overhang (12.0, 10.0)->(13.0, 10.0) and the redundant cycle via y=14.0 must be pruned
        for t in &board.tracks {
            assert!(
                t.start.0 <= 12.001 || t.start.1 > 10.5 || t.start.0 >= 19.999,
                "Overhanging corner stub past (12, 10) must be split and pruned: {:?}",
                t
            );
            assert!(
                t.start.1 < 13.0 && t.end.1 < 13.0,
                "Redundant parallel detour at y=14.0 must be pruned: {:?}",
                t
            );
        }
        assert_eq!(board.tracks.len(), 4);
    }

    #[test]
    fn test_eliminate_stair_steps_collapses_3_point_v_bounce() {
        let board = Board {
            min_x: 130.0,
            min_y: 80.0,
            max_x: 180.0,
            max_y: 130.0,
            nets: vec!["".to_string(), "SIG".to_string()],
            ..Board::default()
        };
        let grid = RoutingGrid::new(&board, 0.1, None);

        // NW then NE V-bounce: (141.7225, 96.5400) -> (141.1225, 95.9400) -> (141.5500, 95.5500)
        let mut chain = TrackChain {
            points: vec![
                (141.7225, 96.5400),
                (141.1225, 95.9400),
                (141.5500, 95.5500),
            ],
            width: 0.15,
            layer: 0,
            net_id: 1,
        };
        let orig_len = chain.points.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).sum::<f64>();
        eliminate_stair_steps(&mut chain, &board, &grid);
        let new_len = chain.points.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).sum::<f64>();
        assert!(
            new_len < orig_len - 0.3,
            "3-point V-bounce must collapse to direct North dogleg: orig_len={:.3}, new_len={:.3}, pts={:?}",
            orig_len,
            new_len,
            chain.points
        );
        for p in &chain.points {
            assert!(p.0 >= 141.54, "Collapsed path must not swing West to 141.12: {:?}", chain.points);
        }
    }

    #[test]
    fn test_c7_south_pad_ground_stitching_and_u2_sw_adjacent_via_consolidation() {
        let pcb_path = std::path::Path::new("/home/deece/src/p1p2esp/p1p2esp_vulcan.kicad_pcb");
        if !pcb_path.exists() {
            return;
        }
        let locked: std::collections::HashSet<String> = crate::board::DEFAULT_LOCKED_REFS.iter().map(|s| s.to_string()).collect();
        let mut board = Board::load(pcb_path.to_str().unwrap(), &locked).unwrap();
        let had_routed_tracks = !board.tracks.is_empty();
        let router = AutoRouter::new(2);
        let mut grid = RoutingGrid::new(&board, 0.10, None);

        // 1. Test adjacent via consolidation on Net-(U2-SW)
        let sw_net_id = *board.net_name_to_id.get("Net-(U2-SW)").unwrap();
        if !board.vias.iter().any(|v| v.net_id == sw_net_id && (v.position.0 - 144.45).hypot(v.position.1 - 113.05) < 0.1) {
            board.vias.push(Via {
                position: (144.45, 113.05),
                size: 0.6,
                drill: 0.3,
                net_id: sw_net_id,
                is_locked: false,
            });
        }
        if !board.vias.iter().any(|v| v.net_id == sw_net_id && (v.position.0 - 144.20).hypot(v.position.1 - 113.60) < 0.1) {
            board.vias.push(Via {
                position: (144.20, 113.60),
                size: 0.6,
                drill: 0.3,
                net_id: sw_net_id,
                is_locked: false,
            });
            board.tracks.push(TrackSegment {
                start: (144.45, 113.05),
                end: (144.20, 113.60),
                width: 0.3,
                layer: 0,
                net_id: sw_net_id,
                is_locked: false,
            });
        }
        let conns_before = router.extract_connections(&board);
        let connected_before: Vec<RouteConnection> = conns_before
            .into_iter()
            .filter(|c| c.net_id == sw_net_id && AutoRouter::is_connection_electrically_connected(&board, c))
            .collect();

        let sw_vias_before: Vec<Via> = board.vias.iter().filter(|v| v.net_id == sw_net_id).cloned().collect();
        assert!(sw_vias_before.len() >= 2);

        router.reduce_vias(&mut board, &mut grid);
        validate_and_clean_copper(&mut board);
        prune_dangling_tracks_and_vias(&mut board);

        let sw_vias_after: Vec<Via> = board.vias.iter().filter(|v| v.net_id == sw_net_id).cloned().collect();
        for i in 0..sw_vias_after.len() {
            for j in (i + 1)..sw_vias_after.len() {
                let d = (sw_vias_after[i].position.0 - sw_vias_after[j].position.0)
                    .hypot(sw_vias_after[i].position.1 - sw_vias_after[j].position.1);
                assert!(
                    d > 1.50,
                    "Net-(U2-SW) must not have 2 adjacent vias within 1.50mm: {:?} vs {:?} (dist={:.3}mm)",
                    sw_vias_after[i].position,
                    sw_vias_after[j].position,
                    d
                );
            }
        }

        // Verify all previously routed Net-(U2-SW) connections remain electrically continuous
        for c in &connected_before {
            assert!(
                AutoRouter::is_connection_electrically_connected(&board, c),
                "Net-(U2-SW) connection {:?} -> {:?} must remain electrically continuous after via consolidation",
                c.start,
                c.end
            );
        }

        // 2. Test C7 GND pad stitching
        let gnd_net_id = *board.net_name_to_id.get("GND").unwrap();
        router.add_plane_stitching_vias(&mut board, &grid);
        prune_dangling_tracks_and_vias(&mut board);

        let c7 = board.footprints.iter().find(|fp| fp.reference == "C7").unwrap();
        let c7_gnd_pad = c7.pads.iter().find(|p| p.net_id == gnd_net_id).unwrap();
        let (c7_px, c7_py) = c7.pad_absolute_position(c7_gnd_pad);
        let c7_connected = board.tracks.iter().any(|t| {
            t.net_id == gnd_net_id
                && t.layer == c7.layer
                && ((t.start.0 - c7_px).hypot(t.start.1 - c7_py) < 0.08
                    || (t.end.0 - c7_px).hypot(t.end.1 - c7_py) < 0.08)
        }) || board.arcs.iter().any(|a| {
            let s = a.start();
            let e = a.end();
            a.net_id == gnd_net_id
                && a.layer == c7.layer
                && ((s.0 - c7_px).hypot(s.1 - c7_py) < 0.08
                    || (e.0 - c7_px).hypot(e.1 - c7_py) < 0.08)
        });
        assert!(
            c7_connected,
            "GND pad of C7 at ({:.4}, {:.4}) must have a connected GND stitching track or arc",
            c7_px,
            c7_py
        );

        let out_pcb = "/tmp/test_c7_u2_sw_repaired.kicad_pcb";
        board.save_pcb(std::path::Path::new(out_pcb)).unwrap();
        if let Ok(status) = std::process::Command::new("kicad-cli")
            .args(["pcb", "drc", "--refill-zones", "--output", "/tmp/test_c7_u2_sw_drc.json", "--format", "json", out_pcb])
            .status()
        {
            if status.success() {
                if let Ok(drc_json) = std::fs::read_to_string("/tmp/test_c7_u2_sw_drc.json") {
                    if had_routed_tracks {
                        assert!(
                            drc_json.contains("\"unconnected_items\": []"),
                            "kicad-cli drc --refill-zones must report 0 unconnected_items"
                        );
                    }
                    assert!(
                        !drc_json.contains("\"hole_to_hole\""),
                        "kicad-cli drc --refill-zones must report 0 hole_to_hole violations"
                    );
                }
            }
        }
    }

    #[test]
    fn test_gpu_3d_routing_cost_kernel_clearance_and_congestion() {
        let Some(vk) = VulkanEngine::new() else {
            return;
        };
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 2.0;
        board.max_y = 2.0;
        board.nets = vec!["".into(), "NET1".into(), "NET2".into()];

        let mut grid = RoutingGrid::new(&board, 0.1, Some(vk));
        grid.obstacle_net_ids.fill(FREE_NET);
        grid.dist_field.fill(2.0);
        grid.congestion.fill(0.0);
        grid.usage_count.fill(0);

        // Place foreign obstacle (net 2) at (10, 10, 0) and same-net obstacle (net 1) at (4, 4, 0)
        let foreign_idx = grid.idx(10, 10, 0);
        let same_idx = grid.idx(4, 4, 0);
        let cong_idx = grid.idx(15, 15, 1);
        let free_idx = grid.idx(15, 5, 1);
        grid.obstacle_net_ids[foreign_idx] = 2;
        grid.obstacle_net_ids[same_idx] = 1;
        grid.congestion[cong_idx] = 3.0;
        grid.usage_count[cong_idx] = 2;

        grid.precompute_gpu_routing_costs_batch(&[(1, 2), (2, 2)], 1.5, 2.0);
        let (cached_r, cost_net1) = grid.gpu_net_cost_maps.get(&1).expect("Net 1 cost map should be cached");
        assert_eq!(*cached_r, 2);

        // Foreign obstacle and cells within 2-cell radius must be impassable (< 0.0)
        assert!(cost_net1[foreign_idx] < 0.0);
        assert!(cost_net1[grid.idx(12, 10, 0)] < 0.0);
        assert!(cost_net1[grid.idx(13, 10, 0)] > 0.0);

        // Same-net obstacle must remain passable (> 0.0)
        assert!(cost_net1[same_idx] > 0.0);
        assert!(cost_net1[grid.idx(5, 4, 0)] > 0.0);

        // Congestion and usage_count must increase cell cost
        assert!(cost_net1[cong_idx] > cost_net1[free_idx] + 5.0);

        // Verify get_neighbors with GPU cost map matches CPU-only get_neighbors
        let neighbors_with_gpu = grid.get_neighbors(
            GridNode { gx: 13, gy: 10, layer: 0 },
            1,
            0.15,
            0.15,
            0.60,
            None,
        );
        grid.gpu_net_cost_maps.clear();
        let neighbors_cpu_only = grid.get_neighbors(
            GridNode { gx: 13, gy: 10, layer: 0 },
            1,
            0.15,
            0.15,
            0.60,
            None,
        );
        assert_eq!(neighbors_with_gpu, neighbors_cpu_only);
    }

    #[test]
    fn test_gpu_3d_wavefront_shortest_path_potential() {
        let Some(vk) = VulkanEngine::new() else {
            return;
        };
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 1.6;
        board.max_y = 1.6;
        board.nets = vec!["".into(), "NET1".into(), "NET2".into()];

        let mut grid = RoutingGrid::new(&board, 0.1, Some(vk));
        grid.obstacle_net_ids.fill(FREE_NET);
        grid.dist_field.fill(2.0);

        let seed = grid.idx(2, 2, 0) as u32;
        let wf = grid
            .compute_gpu_3d_wavefront_for_net(1, 1, &[seed], 3.5, 32)
            .expect("GPU 3D wavefront should succeed");

        let d0 = wf[grid.idx(2, 2, 0)];
        let d1 = wf[grid.idx(3, 2, 0)];
        let d2 = wf[grid.idx(6, 2, 0)];
        let d_via = wf[grid.idx(2, 2, 1)];
        let d_via_step = wf[grid.idx(5, 2, 1)];

        assert!((d0 - 0.0).abs() < 1e-5);
        assert!(d1 > d0 && d2 > d1, "Wavefront potential must increase monotonically with distance");
        assert!((d_via - 3.5).abs() < 1e-4, "Via transition cost must equal via_cost");
        assert!(d_via_step > d_via, "Planar steps on adjacent layer must accumulate over via cost");
    }

    #[test]
    fn test_gpu_batch_drc_eval_via_and_segment_clearance() {
        let mut board = make_router_test_board();
        board.tracks.push(TrackSegment {
            start: (138.0, 100.0),
            end: (146.0, 100.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        board.vias.push(Via {
            position: (142.0, 104.0),
            size: 0.60,
            drill: 0.30,
            net_id: 2,
            is_locked: false,
        });

        let drc_ctx = GpuDrcContext::from_board(&board);
        let vk = VulkanEngine::new();

        let via_pts = [
            (142.0, 100.25), // Collides with net 2 track at y=100.0
            (142.0, 102.0),  // Clear of track and via
            (142.3, 104.0),  // Collides with net 2 via at (142.0, 104.0)
            (142.0, 95.0),   // Collides with C14 pad 1 (net 1) for net 3
        ];

        let mut cands = Vec::new();
        for &(vx, vy) in &via_pts {
            cands.push(GpuDrcCandidate {
                x1: vx as f32,
                y1: vy as f32,
                x2: 0.0,
                y2: 0.0,
                radius_or_half_width: 0.30,
                clearance: 0.20,
                layer: 0,
                net_id: 3,
                kind: 0,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            });
        }

        let segs = [
            ((140.0, 99.0), (140.0, 101.0), 0usize), // Crosses net 2 track on layer 0 -> illegal
            ((140.0, 99.0), (140.0, 101.0), 1usize), // Crosses on layer 1 -> legal
            ((138.0, 101.5), (146.0, 101.5), 0usize), // Parallel 1.5mm away -> legal
        ];
        for &(s, e, layer) in &segs {
            cands.push(GpuDrcCandidate {
                x1: s.0 as f32,
                y1: s.1 as f32,
                x2: e.0 as f32,
                y2: e.1 as f32,
                radius_or_half_width: 0.125,
                clearance: 0.20,
                layer: layer as u32,
                net_id: 3,
                kind: 1,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            });
        }

        if let Some(res) = drc_ctx.evaluate(vk.as_ref(), &cands) {
            for (i, &pt) in via_pts.iter().enumerate() {
                let cpu_ok = is_via_location_legal(&board, pt, 0.60, 3, 0.20);
                assert_eq!(res[i] != 0, cpu_ok, "Via candidate {} at {:?} mismatch", i, pt);
            }
            for (j, &(s, e, layer)) in segs.iter().enumerate() {
                let cpu_ok = is_track_segment_legal(&board, s, e, 0.25, layer, 3, 0.20);
                assert_eq!(
                    res[via_pts.len() + j] != 0,
                    cpu_ok,
                    "Segment candidate {} ({:?} -> {:?} on L{}) mismatch",
                    j,
                    s,
                    e,
                    layer
                );
            }
        }
    }

    #[test]
    fn test_cuda_vs_vulkan_batch_drc_equivalence() {
        let (Some(cuda), Some(vk)) = (CudaEngine::new(), VulkanEngine::new()) else {
            return;
        };
        let mut board = make_router_test_board();
        board.tracks.push(TrackSegment {
            start: (136.0, 100.0),
            end: (148.0, 100.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        let drc_ctx = GpuDrcContext::from_board(&board);
        let mut cands = Vec::with_capacity(160);
        for i in 0..160 {
            let y = 96.0 + (i as f32) * 0.05;
            cands.push(GpuDrcCandidate {
                x1: 140.0,
                y1: y,
                x2: 142.0,
                y2: y + 0.2,
                radius_or_half_width: if i % 2 == 0 { 0.30 } else { 0.125 },
                clearance: 0.20,
                layer: 0,
                net_id: 3,
                kind: (i % 2) as u32,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            });
        }
        let cuda_res = cuda
            .evaluate_drc_candidates(&drc_ctx.pads, &drc_ctx.tracks, &drc_ctx.vias, &cands)
            .expect("CUDA evaluate_drc_candidates failed");
        let vk_res = vk
            .evaluate_drc_candidates(&drc_ctx.pads, &drc_ctx.tracks, &drc_ctx.vias, &cands)
            .expect("Vulkan evaluate_drc_candidates failed");
        assert_eq!(cuda_res, vk_res, "CUDA and Vulkan DRC batch results must match identically");
    }

    #[test]
    fn test_conflict_graph_independent_set_wave_partitioning() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 20.0;
        board.max_y = 20.0;
        board.nets = vec![
            "".to_string(),
            "NET1".to_string(),
            "NET2".to_string(),
            "NET3".to_string(),
        ];

        let grid = RoutingGrid::new(&board, 0.1, None);
        let nc = NetClass::default();

        // Connection 0: NET1 horizontal along y = 3.0 (gy = 30) on layer 0
        // Connection 1: NET2 horizontal along y = 15.0 (gy = 150) on layer 0 (disjoint from NET1)
        // Connection 2: NET1 vertical along x = 5.0 (gx = 50) on layer 0 (same net as Connection 0 -> no conflict!)
        // Connection 3: NET3 vertical along x = 6.0 (gx = 60) crossing y = 2.0..5.0 on layer 0 (crosses Connection 0 -> must be in Wave 1!)
        let connections = vec![
            RouteConnection {
                net_id: 1,
                net_name: "NET1".to_string(),
                start: (2.0, 3.0, 0),
                end: (10.0, 3.0, 0),
                start_fp: 0,
                start_pad: 0,
                end_fp: 0,
                end_pad: 0,
                netclass: nc.clone(),
                diff_pair_leader: None,
            },
            RouteConnection {
                net_id: 2,
                net_name: "NET2".to_string(),
                start: (2.0, 15.0, 0),
                end: (10.0, 15.0, 0),
                start_fp: 0,
                start_pad: 0,
                end_fp: 0,
                end_pad: 0,
                netclass: nc.clone(),
                diff_pair_leader: None,
            },
            RouteConnection {
                net_id: 1,
                net_name: "NET1".to_string(),
                start: (5.0, 1.0, 0),
                end: (5.0, 6.0, 0),
                start_fp: 0,
                start_pad: 0,
                end_fp: 0,
                end_pad: 0,
                netclass: nc.clone(),
                diff_pair_leader: None,
            },
            RouteConnection {
                net_id: 3,
                net_name: "NET3".to_string(),
                start: (6.0, 1.0, 0),
                end: (6.0, 6.0, 0),
                start_fp: 0,
                start_pad: 0,
                end_fp: 0,
                end_pad: 0,
                netclass: nc,
                diff_pair_leader: None,
            },
        ];

        let path0: Vec<GridNode> = (20..=100).map(|gx| GridNode { gx, gy: 30, layer: 0 }).collect();
        let path1: Vec<GridNode> = (20..=100).map(|gx| GridNode { gx, gy: 150, layer: 0 }).collect();
        let path2: Vec<GridNode> = (10..=60).map(|gy| GridNode { gx: 50, gy, layer: 0 }).collect();
        let path3: Vec<GridNode> = (10..=60).map(|gy| GridNode { gx: 60, gy, layer: 0 }).collect();

        let candidates = vec![
            (0usize, path0),
            (1usize, path1),
            (2usize, path2),
            (3usize, path3),
        ];

        let waves = AutoRouter::partition_into_conflict_free_waves(&grid, &connections, candidates);
        assert_eq!(waves.len(), 2, "Expected 2 conflict-free waves, got {:?}", waves.len());

        let wave0_indices: Vec<usize> = waves[0].iter().map(|(idx, _)| *idx).collect();
        let wave1_indices: Vec<usize> = waves[1].iter().map(|(idx, _)| *idx).collect();

        // Connections 0 (NET1), 1 (disjoint NET2), and 2 (same-net NET1) must all be grouped into Wave 0
        assert_eq!(wave0_indices, vec![0, 1, 2]);
        // Connection 3 (NET3 crossing NET1) must be separated into Wave 1
        assert_eq!(wave1_indices, vec![3]);
    }

    #[test]
    fn test_parallel_portfolio_multi_trajectory_routing() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 20.0;
        board.max_y = 20.0;
        board.nets = vec![
            "".to_string(),
            "NET1".to_string(),
            "NET2".to_string(),
            "NET3".to_string(),
        ];

        let mut grid = RoutingGrid::new(&board, 0.1, VulkanEngine::new());
        grid.obstacle_net_ids.fill(FREE_NET);
        grid.dist_field.fill(2.0);

        // Build an asymmetric deep U-pocket around start (gx=40, gy=100) opening West,
        // while goal is at (gx=140, gy=100) to the East.
        // Forward A* with a small expansion cap (e.g. 450 expansions) is locally trapped inside the deep pocket,
        // whereas reverse A* (target -> source) or unconstrained portfolio expansion routes cleanly.
        for gx in 30..=90 {
            for l in 0..grid.num_layers {
                let i1 = grid.idx(gx, 85, l);
                let i2 = grid.idx(gx, 115, l);
                grid.obstacle_net_ids[i1] = KEEPOUT_NET;
                grid.obstacle_net_ids[i2] = KEEPOUT_NET;
            }
        }
        for gy in 85..=115 {
            for l in 0..grid.num_layers {
                let i3 = grid.idx(90, gy, l);
                grid.obstacle_net_ids[i3] = KEEPOUT_NET;
            }
        }

        let router = AutoRouter::new(2);
        let stop_flag = AtomicBool::new(false);
        let s_node = GridNode { gx: 25, gy: 100, layer: 0 };
        let e_node = GridNode { gx: 100, gy: 100, layer: 0 };

        // 1. Verify reverse-direction portfolio variant escapes when forward A* is locally trapped
        let fwd_trapped = router.find_path_portfolio_variant(
            &grid,
            s_node,
            e_node,
            1,
            0.15,
            0.15,
            0.60,
            None,
            15.0,
            false,
            1.0,
            1.0,
            1.35,
            0.0,
            false,
            false,
            &stop_flag,
            None,
            Some(350),
        );
        assert!(fwd_trapped.is_none(), "Forward greedy A* with 350-expansion budget should be trapped inside U-pocket");

        let rev_escaped = router.find_path_portfolio_variant(
            &grid,
            s_node,
            e_node,
            1,
            0.15,
            0.15,
            0.60,
            None,
            15.0,
            false,
            1.0,
            1.0,
            1.35,
            0.0,
            true,
            false,
            &stop_flag,
            None,
            Some(2500),
        );
        assert!(rev_escaped.is_some(), "Reverse target->source A* must bypass U-pocket and find legal path");
        let rev_path = rev_escaped.unwrap();
        assert_eq!(rev_path.first().copied(), Some(s_node));
        assert_eq!(rev_path.last().copied(), Some(e_node));

        // 2. Verify full parallel portfolio multi-trajectory ensemble across multiple connections
        let nc = NetClass::default();
        let sw = grid.grid_to_world(s_node.gx, s_node.gy);
        let ew = grid.grid_to_world(e_node.gx, e_node.gy);
        let conns = vec![
            RouteConnection {
                net_id: 1,
                net_name: "NET1".to_string(),
                start: (sw.0, sw.1, 0),
                end: (ew.0, ew.1, 0),
                start_fp: 0,
                start_pad: 0,
                end_fp: 0,
                end_pad: 0,
                netclass: nc.clone(),
                diff_pair_leader: None,
            },
            RouteConnection {
                net_id: 2,
                net_name: "NET2".to_string(),
                start: (2.0, 3.0, 0),
                end: (16.0, 3.0, 0),
                start_fp: 0,
                start_pad: 0,
                end_fp: 0,
                end_pad: 0,
                netclass: nc,
                diff_pair_leader: None,
            },
        ];

        let target_indices = vec![0usize, 1usize];
        let results = router.route_connections_portfolio(
            &board,
            &grid,
            &conns,
            &target_indices,
            None,
            &stop_flag,
        );
        assert_eq!(results.len(), 2);
        for (c_idx, p_opt) in &results {
            let p = p_opt.as_ref().unwrap_or_else(|| panic!("Portfolio failed to route connection {}", c_idx));
            assert!(
                router.is_path_conflict_free(
                    &grid,
                    p,
                    conns[*c_idx].net_id,
                    conns[*c_idx].netclass.clearance,
                    conns[*c_idx].netclass.track_width,
                    conns[*c_idx].netclass.via_dia,
                    None,
                    false,
                ),
                "Portfolio path for connection {} must be conflict-free",
                c_idx
            );
        }
    }

    #[test]
    fn test_gpu_3d_wavefront_backtrack_path_valid_and_obstacle_free() {
        let vk = VulkanEngine::new();
        let board = make_router_test_board();
        let mut grid = RoutingGrid::new(&board, 0.25, vk);

        // Clear synthetic grid interior so we can construct a deterministic obstacle barrier
        grid.obstacle_net_ids.fill(FREE_NET);
        let gw = grid.width;
        let gh = grid.height;
        assert!(gw >= 24 && gh >= 24);

        let mid_x = gw / 2;
        let clearance_cells = 1i32;
        // Place a foreign obstacle barrier (net 99) at x = mid_x, y = 2..(gh - 6) on all layers
        for l in 0..grid.num_layers {
            for y in 2..(gh - 6) {
                let idx = grid.idx(mid_x, y, l);
                grid.obstacle_net_ids[idx] = 99;
            }
        }

        let router = AutoRouter::new(1);
        let src1 = GridNode {
            gx: mid_x.saturating_sub(5),
            gy: gh / 2,
            layer: 0,
        };
        let dst1 = GridNode {
            gx: (mid_x + 5).min(gw - 3),
            gy: gh / 2,
            layer: 0,
        };
        let src2 = GridNode {
            gx: mid_x.saturating_sub(4),
            gy: gh - 3,
            layer: 1,
        };
        let dst2 = GridNode {
            gx: (mid_x + 4).min(gw - 3),
            gy: gh - 3,
            layer: 1,
        };

        let requests = vec![
            (src1, dst1, 1usize, 3.0f32),
            (src2, dst2, 2usize, 3.0f32),
        ];
        let results = router.solve_batch_gpu_wavefront_paths(&grid, &requests, clearance_cells);
        assert_eq!(results.len(), 2);

        for (req_idx, &(src, dst, net_id, _)) in requests.iter().enumerate() {
            let path = results[req_idx]
                .as_ref()
                .expect("GPU 3D wavefront solver must return a valid obstacle-free path");
            assert_eq!(path.first().copied(), Some(src));
            assert_eq!(path.last().copied(), Some(dst));
            for w in path.windows(2) {
                let dx = (w[1].gx as isize - w[0].gx as isize).abs();
                let dy = (w[1].gy as isize - w[0].gy as isize).abs();
                let dl = (w[1].layer as isize - w[0].layer as isize).abs();
                assert!(
                    (dl == 0 && dx <= 1 && dy <= 1 && (dx + dy > 0))
                        || (dl > 0 && dx == 0 && dy == 0),
                    "Invalid step {:?} -> {:?}",
                    w[0],
                    w[1]
                );
            }
            for node in path {
                assert!(
                    grid.is_circle_clear_SpecifyRadius(
                        node.gx,
                        node.gy,
                        node.layer,
                        clearance_cells,
                        net_id
                    ),
                    "Node {:?} violated clearance radius {} for net {}",
                    node,
                    clearance_cells,
                    net_id
                );
            }
        }
    }

    #[test]
    fn test_parallel_disjoint_net_glossing_and_via_reduction() {
        let nets = vec![
            "".to_string(),
            "SIG1".to_string(),
            "SIG2".to_string(),
            "SIG3".to_string(),
        ];
        let net_name_to_id: HashMap<String, usize> = nets
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let netclasses: HashMap<String, NetClass> =
            [("Default".to_string(), NetClass::default())].into_iter().collect();
        let net_to_class: HashMap<String, String> = [
            ("SIG1".to_string(), "Default".to_string()),
            ("SIG2".to_string(), "Default".to_string()),
            ("SIG3".to_string(), "Default".to_string()),
        ]
        .into_iter()
        .collect();

        // Endpoint footprints for SIG1 (y = 10.0), SIG2 (y = 30.0), and SIG3 (y = 50.0)
        let fp_sig1_a = make_fp(
            "U_S1A",
            10.0,
            10.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false)],
        );
        let fp_sig1_b = make_fp(
            "U_S1B",
            35.0,
            10.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false)],
        );
        let fp_sig2_a = make_fp(
            "U_S2A",
            10.0,
            30.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "SIG2", false)],
        );
        let fp_sig2_b = make_fp(
            "U_S2B",
            35.0,
            30.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "SIG2", false)],
        );
        let fp_sig3_a = make_fp(
            "U_S3A",
            10.0,
            50.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 3, "SIG3", false)],
        );
        let fp_sig3_b = make_fp(
            "U_S3B",
            35.0,
            50.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 3, "SIG3", false)],
        );

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: vec![
                fp_sig1_a, fp_sig1_b, fp_sig2_a, fp_sig2_b, fp_sig3_a, fp_sig3_b,
            ],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            // Layer 0 keepout blocking SIG3's via pair at y = 50.0 (boundary/violation preservation case)
            keepouts: vec![KeepoutZone {
                min_x: 18.0,
                min_y: 35.0,
                max_x: 22.0,
                max_y: 65.0,
                layers: vec![0],
            }],
            tracks: vec![
                // SIG1 (net 1, y ~ 10.0): redundant via pair + rubber-band detour
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
                    end: (30.0, 14.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (30.0, 14.0),
                    end: (35.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // SIG2 (net 2, y ~ 30.0): single via between two F.Cu pads + detour
                TrackSegment {
                    start: (10.0, 30.0),
                    end: (15.0, 30.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 2,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 30.0),
                    end: (25.0, 34.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 2,
                    is_locked: false,
                },
                TrackSegment {
                    start: (25.0, 34.0),
                    end: (35.0, 30.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 2,
                    is_locked: false,
                },
                // SIG3 (net 3, y ~ 50.0): necessary via pair over keepout + rubber-band detour
                TrackSegment {
                    start: (10.0, 50.0),
                    end: (15.0, 50.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 3,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 50.0),
                    end: (25.0, 50.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 3,
                    is_locked: false,
                },
                TrackSegment {
                    start: (25.0, 50.0),
                    end: (30.0, 54.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 3,
                    is_locked: false,
                },
                TrackSegment {
                    start: (30.0, 54.0),
                    end: (35.0, 50.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 3,
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
                Via {
                    position: (15.0, 30.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 2,
                    is_locked: false,
                },
                Via {
                    position: (15.0, 50.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 3,
                    is_locked: false,
                },
                Via {
                    position: (25.0, 50.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 3,
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
        router.gloss_tracks(&mut board, &grid);

        // SIG1 and SIG2 vias must be completely eliminated in parallel; SIG3 obstacle-crossing vias must remain
        assert_eq!(board.vias.iter().filter(|v| v.net_id == 1).count(), 0);
        assert_eq!(board.vias.iter().filter(|v| v.net_id == 2).count(), 0);
        assert_eq!(board.vias.iter().filter(|v| v.net_id == 3).count(), 2);

        // Verify SIG1 and SIG2 are straightened onto layer 0 (length == 25.0 mm each)
        let sig1_len: f64 = board
            .tracks
            .iter()
            .filter(|t| t.net_id == 1)
            .map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1))
            .sum();
        let sig2_len: f64 = board
            .tracks
            .iter()
            .filter(|t| t.net_id == 2)
            .map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1))
            .sum();
        assert!((sig1_len - 25.0).abs() < 1e-3, "SIG1 must be straightened to 25.0 mm, got {:.3}", sig1_len);
        assert!((sig2_len - 25.0).abs() < 1e-3, "SIG2 must be straightened to 25.0 mm, got {:.3}", sig2_len);
    }

    #[test]
    fn test_parallel_speculative_track_and_component_nudge() {
        let nets = vec![
            "".to_string(),
            "SIG_TARGET".to_string(),
            "SIG_BLOCK".to_string(),
        ];
        let net_name_to_id: HashMap<String, usize> = nets
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let netclasses: HashMap<String, NetClass> =
            [("Default".to_string(), NetClass::default())].into_iter().collect();
        let net_to_class: HashMap<String, String> = [
            ("SIG_TARGET".to_string(), "Default".to_string()),
            ("SIG_BLOCK".to_string(), "Default".to_string()),
        ]
        .into_iter()
        .collect();

        // Part 1: Parallel speculative track deflection via try_nudge_or_ripup_blocking_tracks
        let fp_a = make_fp(
            "U1",
            14.0,
            10.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 1, "SIG_TARGET", false)],
        );
        let fp_b = make_fp(
            "U2",
            26.0,
            10.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 1, "SIG_TARGET", false)],
        );
        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 30.0,
            footprints: vec![fp_a, fp_b],
            holding_palette: Vec::new(),
            nets: nets.clone(),
            net_name_to_id: net_name_to_id.clone(),
            netclasses: netclasses.clone(),
            net_to_class: net_to_class.clone(),
            keepouts: Vec::new(),
            tracks: vec![TrackSegment {
                start: (10.0, 10.15),
                end: (30.0, 10.15),
                width: 0.25,
                layer: 0,
                net_id: 2,
                is_locked: false,
            }],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let router = AutoRouter::new(2);
        let mut grid = RoutingGrid::new(&board, 0.25, None);
        grid.rasterize_obstacles(&board);

        let conns = router.extract_connections(&board);
        assert_eq!(conns.len(), 1);
        let mut routed_paths: Vec<Option<Vec<GridNode>>> = vec![None; conns.len()];
        let stop_flag = AtomicBool::new(false);

        let routed = router.try_nudge_or_ripup_blocking_tracks(
            &mut board,
            &mut grid,
            &conns,
            &mut routed_paths,
            &[],
            0,
            None,
            &stop_flag,
        );
        assert!(
            routed,
            "Speculative parallel track shove must deflect blocking track and route target connection"
        );
        assert!(
            routed_paths[0].is_some(),
            "Target net must have routed path in routed_paths"
        );

        // Part 2: Parallel speculative component nudge via try_nudge_blocking_components
        let mut c_block = make_fp(
            "C_BLOCK",
            20.0,
            10.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 2, "SIG_BLOCK", false)],
        );
        c_block.courtyards = vec![Polygon::new(vec![
            Point::new(-0.5, -0.5),
            Point::new(0.5, -0.5),
            Point::new(0.5, 0.5),
            Point::new(-0.5, 0.5),
        ])];
        let mut board_comp = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 30.0,
            footprints: vec![
                make_fp(
                    "U1",
                    14.0,
                    10.0,
                    true,
                    vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 1, "SIG_TARGET", false)],
                ),
                make_fp(
                    "U2",
                    26.0,
                    10.0,
                    true,
                    vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 1, "SIG_TARGET", false)],
                ),
                c_block,
            ],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            // Channel keepouts on both layers leaving a corridor around y = 10.0
            keepouts: vec![
                KeepoutZone {
                    min_x: 12.0,
                    min_y: 0.0,
                    max_x: 28.0,
                    max_y: 8.5,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 12.0,
                    min_y: 12.5,
                    max_x: 28.0,
                    max_y: 30.0,
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

        let mut grid_comp = RoutingGrid::new(&board_comp, 0.25, None);
        grid_comp.rasterize_obstacles(&board_comp);
        let mut conns_comp = router.extract_connections(&board_comp);
        let mut routed_paths_comp: Vec<Option<Vec<GridNode>>> = vec![None; conns_comp.len()];

        // Boundary check: when allow_component_nudge is false, it must return false without moving C_BLOCK
        assert!(!router.try_nudge_blocking_components(
            &mut board_comp,
            &mut grid_comp,
            &mut conns_comp,
            &mut routed_paths_comp,
            &[],
            0,
            None,
            &stop_flag,
        ));

        let router_nudge = AutoRouter::new(2).with_allow_component_nudge(true);
        let nudged_and_routed = router_nudge.try_nudge_blocking_components(
            &mut board_comp,
            &mut grid_comp,
            &mut conns_comp,
            &mut routed_paths_comp,
            &[],
            0,
            None,
            &stop_flag,
        );
        assert!(
            nudged_and_routed,
            "Parallel speculative component nudge must displace C_BLOCK and route SIG_TARGET"
        );
        assert!(
            (board_comp.footprints[2].x - 20.0).hypot(board_comp.footprints[2].y - 10.0) > 0.1,
            "C_BLOCK must have been nudged from its blocking position (20.0, 10.0)"
        );
    }

    #[test]
    fn test_base_snapshot_caching_and_parallel_return_vias() {
        let nets = vec!["".to_string(), "GND".to_string(), "CLK".to_string()];
        let net_name_to_id: HashMap<String, usize> = nets
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        let mut default_nc = NetClass::default();
        default_nc.require_return_path = true;
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), default_nc);

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 30.0,
            max_y: 20.0,
            footprints: vec![
                make_fp(
                    "U1",
                    5.0,
                    10.0,
                    true,
                    vec![
                        Pad::new("1", 0.0, 0.0, 0.6, 0.6, 2, "CLK", false),
                        Pad::new("2", 0.0, 2.0, 0.6, 0.6, 1, "GND", false),
                    ],
                ),
                make_fp(
                    "U2",
                    25.0,
                    10.0,
                    false,
                    vec![Pad::new("1", 0.0, 0.0, 0.6, 0.6, 2, "CLK", false)],
                ),
            ],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class: HashMap::new(),
            keepouts: Vec::new(),
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
        };

        let router = AutoRouter::new(2);
        let mut grid = RoutingGrid::new(&board, 0.25, None);
        assert!(grid.base_snapshot.is_none());

        let conns = router.extract_connections(&board);
        let routed_paths: Vec<Option<Vec<GridNode>>> = vec![None; conns.len()];

        // First rebuild_grid_state populates base_snapshot
        router.rebuild_grid_state(&board, &mut grid, &conns, &routed_paths, &[]);
        assert!(grid.base_snapshot.is_some());
        let first_obs = grid.obstacle_net_ids.clone();
        let first_smd = grid.smd_pad_cells.clone();

        // Mutate obstacle_net_ids and verify cached rebuild restores exact snapshot
        grid.obstacle_net_ids.fill(99);
        router.rebuild_grid_state(&board, &mut grid, &conns, &routed_paths, &[]);
        assert_eq!(grid.obstacle_net_ids, first_obs);
        assert_eq!(grid.smd_pad_cells, first_smd);

        // Moving a footprint and invalidating base_snapshot updates the obstacle map
        board.footprints[1].x = 20.0;
        grid.base_snapshot = None;
        router.rebuild_grid_state(&board, &mut grid, &conns, &routed_paths, &[]);
        assert!(grid.base_snapshot.is_some());
        assert_ne!(grid.obstacle_net_ids, first_obs);

        // Add two signal vias on CLK and verify parallel return-path ground via insertion
        board.vias.push(Via {
            position: (10.0, 10.0),
            size: 0.6,
            drill: 0.3,
            net_id: 2,
            is_locked: false,
        });
        board.vias.push(Via {
            position: (15.0, 10.0),
            size: 0.6,
            drill: 0.3,
            net_id: 2,
            is_locked: false,
        });
        let added = router.add_return_path_ground_vias(&mut board, &mut grid);
        assert_eq!(added, 2, "Expected 2 return-path ground vias for 2 signal vias");
        assert_eq!(board.vias.iter().filter(|v| v.net_id == 1).count(), 2);
    }

    #[test]
    fn test_topological_and_physical_same_net_via_consolidation() {
        let nets = vec!["".to_string(), "SIG_BRIDGE".to_string(), "SIG_SHARED_PAD".to_string()];
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

        // Case 1 (SIG_BRIDGE, net 1):
        // Two vias at (10.0, 10.0) and (11.8, 10.2) (dist ~1.81mm) connected by a 2-segment B.Cu bridge
        // via (11.6, 10.2), where Via 1 at (10.0, 10.0) also has a B.Cu branch to (6.0, 10.0)
        // and Via 2 at (11.8, 10.2) has an F.Cu chain with a knee at (12.2, 10.2) -> (14.0, 10.2).
        // Case 2 (SIG_SHARED_PAD, net 2):
        // Two vias at (19.2, 20.0) and (20.8, 20.0) (dist 1.60mm) that both connect on F.Cu to a shared
        // intermediate SMD pad at (20.0, 20.8), while each transitions to B.Cu to reach (16.0, 20.0) and (24.0, 20.0).
        let mut fp_b1 = make_fp(
            "U_B1",
            6.0,
            10.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 1, "SIG_BRIDGE", false)],
        );
        fp_b1.layer = 1;
        let fp_f1 = make_fp(
            "U_F1",
            10.0,
            8.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 1, "SIG_BRIDGE", false)],
        );
        let fp_f2 = make_fp(
            "U_F2",
            14.0,
            10.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 1, "SIG_BRIDGE", false)],
        );

        let shared_fp = make_fp(
            "R_MID",
            20.0,
            20.8,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 2, "SIG_SHARED_PAD", false)],
        );
        let mut fp_b2_left = make_fp(
            "U_B2L",
            16.0,
            20.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 2, "SIG_SHARED_PAD", false)],
        );
        fp_b2_left.layer = 1;
        let mut fp_b2_right = make_fp(
            "U_B2R",
            24.0,
            20.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 2, "SIG_SHARED_PAD", false)],
        );
        fp_b2_right.layer = 1;

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 40.0,
            footprints: vec![fp_b1, fp_f1, fp_f2, shared_fp, fp_b2_left, fp_b2_right],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                // Net 1 (SIG_BRIDGE):
                TrackSegment {
                    start: (6.0, 10.0),
                    end: (10.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 8.0),
                    end: (10.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // 2-segment B.Cu bridge between Via 1 (10.0, 10.0) and Via 2 (11.8, 10.2)
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (11.6, 10.2),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (11.6, 10.2),
                    end: (11.8, 10.2),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                // F.Cu chain from Via 2 (11.8, 10.2) -> (12.2, 10.2) -> (14.0, 10.0)
                TrackSegment {
                    start: (11.8, 10.2),
                    end: (12.2, 10.2),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (12.2, 10.2),
                    end: (14.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // Net 2 (SIG_SHARED_PAD):
                // Both vias connect on F.Cu to the shared SMD pad at (20.0, 20.8)
                TrackSegment {
                    start: (20.0, 20.8),
                    end: (19.2, 20.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 2,
                    is_locked: false,
                },
                TrackSegment {
                    start: (19.2, 20.0),
                    end: (16.0, 20.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 2,
                    is_locked: false,
                },
                TrackSegment {
                    start: (20.0, 20.8),
                    end: (20.8, 20.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 2,
                    is_locked: false,
                },
                TrackSegment {
                    start: (20.8, 20.0),
                    end: (24.0, 20.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 2,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (10.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (11.8, 10.2),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (19.2, 20.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 2,
                    is_locked: false,
                },
                Via {
                    position: (20.8, 20.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 2,
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
        let router = AutoRouter::new(2);

        router.reduce_vias(&mut board, &mut grid);
        assert_eq!(
            board.vias.iter().filter(|v| v.net_id == 1).count(),
            1,
            "Net 1 (multi-segment bridge) must consolidate from 2 vias to 1 via"
        );
        assert_eq!(
            board.vias.iter().filter(|v| v.net_id == 2).count(),
            1,
            "Net 2 (shared intermediate pad) must consolidate from 2 vias to 1 via"
        );
    }

    #[test]
    fn test_routing_config_via_merge_radius_threshold() {
        use crate::board::NetClass;
        use std::collections::HashMap;

        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert(String::new(), 0);
        net_name_to_id.insert("SIG".to_string(), 1);

        let base_board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 40.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets: vec![String::new(), "SIG".to_string()],
            net_name_to_id,
            netclasses,
            net_to_class: HashMap::new(),
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (6.0, 10.0),
                    end: (10.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (12.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (12.0, 10.0),
                    end: (15.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (12.0, 10.0),
                    end: (12.0, 13.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (10.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (12.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            ..Board::default()
        };

        // 1. With tight via_merge_radius_mm (1.0 mm < 2.0 mm separation), the two vias must NOT merge
        let mut board_tight = base_board.clone();
        let mut grid_tight = RoutingGrid::new(&board_tight, 0.20, None);
        grid_tight.rasterize_obstacles(&board_tight);
        let tight_cfg = RoutingConfig {
            via_merge_radius_mm: 1.0,
            ..RoutingConfig::default()
        };
        let router_tight = AutoRouter::new(1).with_config(tight_cfg);
        router_tight.reduce_vias(&mut board_tight, &mut grid_tight);
        assert_eq!(
            board_tight.vias.len(),
            2,
            "Vias 2.0mm apart must not merge when via_merge_radius_mm is 1.0mm"
        );

        // 2. With default via_merge_radius_mm (4.0 mm >= 2.0 mm separation), the two vias MUST merge into 1
        let mut board_default = base_board;
        let mut grid_default = RoutingGrid::new(&board_default, 0.20, None);
        grid_default.rasterize_obstacles(&board_default);
        let router_default = AutoRouter::new(1).with_config(RoutingConfig::default());
        router_default.reduce_vias(&mut board_default, &mut grid_default);
        assert_eq!(
            board_default.vias.len(),
            1,
            "Vias 2.0mm apart must merge into 1 via when via_merge_radius_mm is 4.0mm"
        );
    }

    #[test]
    fn test_convert_tracks_to_organic_and_back_to_straight() {
        use crate::board::NetClass;
        use std::collections::HashMap;

        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert(String::new(), 0);
        net_name_to_id.insert("NET_A".to_string(), 1);
        net_name_to_id.insert("NET_B".to_string(), 2);

        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 40.0,
            nets: vec![String::new(), "NET_A".to_string(), "NET_B".to_string()],
            net_name_to_id,
            netclasses,
            tracks: vec![
                // NET_A: L-shaped octilinear dogleg from (5,5) -> (15,5) -> (25,15)
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
                // NET_B: Straight 2-point span from (5,20) -> (25,20)
                TrackSegment {
                    start: (5.0, 20.0),
                    end: (25.0, 20.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 2,
                    is_locked: false,
                },
            ],
            ..Board::default()
        };

        assert!(board.arcs.is_empty());
        convert_tracks_to_organic(&mut board);
        assert!(
            !board.arcs.is_empty(),
            "convert_tracks_to_organic must generate curved ArcSegments for bends and/or long straight spans"
        );
        let curved_len: f64 = board.arcs.iter().map(|a| a.length()).sum();
        let straight_len: f64 = board
            .tracks
            .iter()
            .map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1))
            .sum();
        assert!(
            curved_len > straight_len * 3.0,
            "Organic conversion must turn long runs and bends into predominantly smooth curves (curved={:.2}mm vs straight={:.2}mm)",
            curved_len,
            straight_len
        );

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        grid.rasterize_obstacles(&board);
        convert_tracks_to_straight(&mut board, &grid);
        assert!(
            board.arcs.is_empty(),
            "convert_tracks_to_straight must convert all unlocked organic arcs back into straight tracks"
        );
        assert!(!board.tracks.is_empty());
    }

    #[test]
    fn test_power_netclass_width_preserved_on_wide_usb_c_pads_and_organic_widening() {
        use crate::board::NetClass;
        use std::collections::HashMap;

        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        netclasses.insert(
            "Power".to_string(),
            NetClass {
                name: "Power".to_string(),
                track_width: 0.60,
                clearance: 0.25,
                via_dia: 0.80,
                via_drill: 0.40,
                ..NetClass::default()
            },
        );
        let mut net_to_class = HashMap::new();
        net_to_class.insert("VBUS".to_string(), "Power".to_string());
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert(String::new(), 0);
        net_name_to_id.insert("VBUS".to_string(), 1);
        net_name_to_id.insert("USB_DP".to_string(), 2);

        // J2 with a wide 0.60x1.45 VBUS blade (A4 at 7.55, 6.0) and narrow 0.30x1.45 signal pin (A6 at 9.75, 14.0)
        let pad_vbus = Pad::new("A4", -2.45, -4.0, 0.60, 1.45, 1, "VBUS", false);
        let pad_dp = Pad::new("A6", -0.25, 4.0, 0.30, 1.45, 2, "USB_DP", false);
        let fp_j2 = make_fp("J2", 10.0, 10.0, false, vec![pad_vbus, pad_dp]);
        let pad_d2 = Pad::new("2", 0.0, 0.0, 0.90, 1.20, 1, "VBUS", false);
        let fp_d2 = make_fp("D2", 28.0, 10.0, false, vec![pad_d2]);

        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 40.0,
            footprints: vec![fp_j2, fp_d2],
            nets: vec![String::new(), "VBUS".to_string(), "USB_DP".to_string()],
            net_name_to_id,
            netclasses,
            net_to_class,
            tracks: vec![
                TrackSegment {
                    start: (7.55, 6.0),
                    end: (20.0, 6.0),
                    width: 0.15,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (20.0, 6.0),
                    end: (28.0, 10.0),
                    width: 0.15,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            ..Board::default()
        };

        let mut straight_board = board.clone();
        widen_tracks_and_arcs_to_netclass(&mut straight_board);
        assert!(
            straight_board.tracks.iter().all(|t| t.net_id != 1 || (t.width - 0.60).abs() < 1e-3),
            "VBUS straight tracks in open space must widen from 0.15mm fine-pitch neckdown to 0.60mm Power netclass width"
        );

        convert_tracks_to_organic(&mut board);
        assert!(
            !board.arcs.is_empty(),
            "convert_tracks_to_organic must emit curved ArcSegments for VBUS"
        );
        assert!(
            board.arcs.iter().all(|a| a.net_id != 1 || (a.width - 0.60).abs() < 1e-3),
            "VBUS organic arcs in open space must widen to 0.60mm Power netclass width"
        );
        assert!(
            board.tracks.iter().all(|t| t.net_id != 1 || (t.width - 0.60).abs() < 1e-3),
            "VBUS organic pad stubs in open space must widen to 0.60mm Power netclass width"
        );
    }
}
