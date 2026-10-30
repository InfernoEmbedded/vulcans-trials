// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use crate::board::{Board, LiveTrialState, NetClass, NetTopology, Point, TrackSegment, TrialPhase, Via};
use crate::vulkan::VulkanEngine;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

pub const KEEPOUT_NET: i32 = -1;
pub const FREE_NET: i32 = 0;

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
            let c = board.get_netclass_for_net(i).clearance;
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
        for ko in &board.keepouts {
            if ko.max_x < self.min_x || ko.min_x > max_world_x || ko.max_y < self.min_y || ko.min_y > max_world_y {
                continue;
            }
            let min_gx = ((ko.min_x - self.min_x) / self.res)
                .floor()
                .clamp(0.0, self.width.saturating_sub(1) as f64) as usize;
            let min_gy = ((ko.min_y - self.min_y) / self.res)
                .floor()
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

                for &layer in layers {
                    let min_gx = ((px - half_w - self.min_x) / self.res)
                        .floor()
                        .clamp(0.0, self.width.saturating_sub(1) as f64) as usize;
                    let max_gx = ((px + half_w - self.min_x) / self.res)
                        .ceil()
                        .clamp(0.0, self.width.saturating_sub(1) as f64) as usize;
                    let min_gy = ((py - half_h - self.min_y) / self.res)
                        .floor()
                        .clamp(0.0, self.height.saturating_sub(1) as f64) as usize;
                    let max_gy = ((py + half_h - self.min_y) / self.res)
                        .ceil()
                        .clamp(0.0, self.height.saturating_sub(1) as f64) as usize;

                    for gy in min_gy..=max_gy {
                        for gx in min_gx..=max_gx {
                            let i = self.idx(gx, gy, layer);
                            let net_val = if pad.net_id == 0
                                || pad.net_name.is_empty()
                                || pad.net_name.starts_with("unconnected")
                            {
                                KEEPOUT_NET
                            } else {
                                pad.net_id as i32
                            };
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
        }
    }

    pub fn update_distance_fields(&mut self) {
        let layer_cells = self.width * self.height;
        if let Some(ref vk) = self.vulkan {
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

    pub fn get_neighbors(
        &self,
        node: GridNode,
        target_net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        _u1_center: Option<(f64, f64)>,
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

                let max_clearance = self.max_net_clearance.max(req_clearance);
                let search_clearance = (track_width / 2.0) + max_clearance + self.res;
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
                                let eff_c = req_clearance.max(raw_obst_c);
                                let pad_margin = if self.smd_pad_cells[c_idx] {
                                    self.res * 0.70
                                } else {
                                    self.res * 0.40
                                };
                                let eff_clearance = (track_width / 2.0) + eff_c + pad_margin;
                                let c_dist_sq = (cx_off * cx_off + cy_off * cy_off) as f64 * (self.res * self.res);
                                if c_dist_sq < eff_clearance.powi(2) + 1e-6 {
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
            let search_clearance = (via_dia / 2.0) + max_clearance;
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
                                let eff_c = (via_dia / 2.0) + req_clearance.max(obst_c);
                                if d_sq < eff_c.powi(2) + 1e-6 {
                                    via_clearance_violation = true;
                                    break 'via_check;
                                }
                            }
                            if self.smd_pad_cells[c_idx] && c_net != target_net_id as i32 {
                                let obst_c = if c_net > 0 && (c_net as usize) < self.net_clearances.len() {
                                    self.net_clearances[c_net as usize]
                                } else {
                                    req_clearance
                                };
                                let eff_c = (via_dia / 2.0) + req_clearance.max(obst_c);
                                if d_sq < eff_c.powi(2) + 1e-6 {
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
                let other_idx = self.idx(gx, gy, other_layer);
                let other_net = self.obstacle_net_ids[other_idx];
                if (other_net == target_net_id as i32 || other_net == FREE_NET) && !via_clearance_violation {
                    let via_cost = if other_net == target_net_id as i32 {
                        0.1
                    } else if self.plane_nets.get(&other_layer) == Some(&target_net_id) {
                        1.2
                    } else {
                        2.0
                    };
                    neighbors.push((GridNode { gx, gy, layer: other_layer }, via_cost));
                }
            }
        }

        neighbors
    }
}

fn dist_to_segment(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
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
                        let dist = (pads[i].x - pads[j].x).hypot(pads[i].y - pads[j].y);
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
}

impl AutoRouter {
    pub fn new(max_ripup_iterations: usize) -> Self {
        Self {
            nudge_engine: NudgeEngine::new(),
            max_ripup_iterations,
            congestion_factor: 1.5,
            inner_layer_penalty: 8.0,
        }
    }

    pub fn with_inner_layer_penalty(mut self, pen: f64) -> Self {
        self.inner_layer_penalty = pen;
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

            b_diff
                .cmp(&a_diff)
                .then_with(|| b_is_ic.cmp(&a_is_ic))
                .then_with(|| prio_b.cmp(&prio_a))
                .then_with(|| a.net_id.cmp(&b.net_id))
                .then_with(|| {
                    let d_a = (a.start.0 - a.end.0).hypot(a.start.1 - a.end.1);
                    let d_b = (b.start.0 - b.end.0).hypot(b.start.1 - b.end.1);
                    d_a.partial_cmp(&d_b).unwrap_or(Ordering::Equal)
                })
                .then_with(|| (a.start_fp, a.start_pad).cmp(&(b.start_fp, b.start_pad)))
                .then_with(|| (a.end_fp, a.end_pad).cmp(&(b.end_fp, b.end_pad)))
        });

        // Group differential pair connections (connecting two distinct footprints) adjacently
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
                if fp_pair_i == fp_pair_j
                    && (Self::are_diff_pair_mates(&connections[i].net_name, &connections[j].net_name)
                        || (connections[i].netclass.name == "USB_Diff" && connections[j].netclass.name == "USB_Diff"))
                {
                    mate_idx = Some(j);
                    break;
                }
            }
            if let Some(j) = mate_idx {
                let mut follower = connections.remove(j);
                if connections[i].start_fp != follower.start_fp {
                    follower.swap_endpoints();
                }
                follower.diff_pair_leader = Some(i);
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
        if self.nudge_components_blocking_escapes(board) {
            grid.obstacle_net_ids.fill(FREE_NET);
            grid.nudgeable_fp.fill(None);
            grid.smd_pad_cells.fill(false);
            grid.rasterize_obstacles(board);
            grid.update_distance_fields();
        }

        let mut connections = self.extract_connections(board);
        let total_conns = connections.len();

        let escape_stubs = self.generate_escape_stubs(board);
        for stub in &escape_stubs {
            self.commit_stub_to_grid(grid, stub);
        }

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
                for vk_dev in vk.all_engines() {
                    let obs_ref = Arc::clone(&shared_obs);
                    let df_ref = Arc::clone(&shared_df);
                    let stop_ref = Arc::clone(&stop_flag);
                    let done_ref = &gpu_done;

                    scope.spawn(move || {
                        while !done_ref.load(AtomicOrdering::Relaxed)
                            && !stop_ref.load(AtomicOrdering::Relaxed)
                        {
                            let obs_snap = obs_ref.lock().unwrap().clone();
                            let mut new_df = vec![999.0f32; num_layers * layer_cells];
                            let mut ok = true;
                            for layer in 0..num_layers {
                                if done_ref.load(AtomicOrdering::Relaxed)
                                    || stop_ref.load(AtomicOrdering::Relaxed)
                                {
                                    ok = false;
                                    break;
                                }
                                let offset = layer * layer_cells;
                                let obs_u32: Vec<u32> = obs_snap[offset..offset + layer_cells]
                                    .iter()
                                    .map(|&n| if n != FREE_NET { 1u32 } else { 0u32 })
                                    .collect();
                                if let Some(df) = vk_dev.compute_distance_field(&obs_u32, grid_w, grid_h, 16) {
                                    for (dst, &val) in new_df[offset..offset + layer_cells]
                                        .iter_mut()
                                        .zip(df.iter())
                                    {
                                        *dst = val * grid_res;
                                    }
                                } else {
                                    ok = false;
                                }
                            }
                            if ok {
                                if let Ok(mut g) = df_ref.lock() {
                                    *g = new_df;
                                }
                            }
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

                if let Ok(df_snap) = shared_df.lock() {
                    grid.dist_field.clone_from(&df_snap);
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

                    if routed_paths[leader_idx].is_none() {
                        if let (Some((sgx, sgy)), Some((egx, egy))) = (
                            grid.world_to_grid(lc.start.0, lc.start.1),
                            grid.world_to_grid(lc.end.0, lc.end.1),
                        ) {
                            let s_node = GridNode { gx: sgx, gy: sgy, layer: lc.start.2 };
                            let e_node = GridNode { gx: egx, gy: egy, layer: lc.end.2 };
                            if let Some(lp) = self.find_path_parallel(
                                grid,
                                s_node,
                                e_node,
                                lc.net_id,
                                lc.netclass.clearance,
                                diff_w,
                                lc.netclass.via_dia,
                                u1_center,
                                &stop_flag,
                            ) {
                                self.commit_path_to_grid(
                                    grid,
                                    &lp,
                                    lc.net_id,
                                    diff_w,
                                    lc.netclass.via_dia,
                                );
                                routed_paths[leader_idx] = Some(lp);
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
                                    .find_path_with_corridor(
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
                    }
                }

                let unrouted_indices: Vec<usize> = (0..total_conns)
                    .filter(|&i| routed_paths[i].is_none())
                    .collect();
                if unrouted_indices.is_empty() {
                    break;
                }

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
                                &stop_flag,
                            )
                        } else {
                            self.find_path_with_params(
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
                                &stop_flag,
                            )
                        };
                        (conn_idx, p)
                    })
                    .collect();

                // Phase 2: Commit conflict-free speculative paths in priority order; re-route conflicts in parallel across cores
                let mut _any_committed_in_wave = false;
                for (conn_idx, spec_path) in speculative_paths {
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
                            let remaining_ratsnest: Vec<crate::board::Segment> = connections
                                .iter()
                                .enumerate()
                                .filter(|&(i, _)| routed_paths[i].is_none())
                                .map(|(_, c)| crate::board::Segment {
                                    p1: Point::new(c.start.0, c.start.1),
                                    p2: Point::new(c.end.0, c.end.1),
                                    net_id: c.net_id,
                                })
                                .collect();
                            if let Ok(mut g) = st.lock() {
                                g.board.tracks = board.tracks.clone();
                                g.board.vias = board.vias.clone();
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

                // Rip-up and Negotiated Congestion Update
                if iteration + 1 < self.max_ripup_iterations {
                    let max_rip_prio = if iteration == 0 { 2 } else { 10 };
                    for conn_idx in 0..total_conns {
                        if routed_paths[conn_idx].is_none() {
                            let blocked_conn = &connections[conn_idx];
                            let b_start = blocked_conn.start;
                            let b_end = blocked_conn.end;

                            let mut ripped = false;
                            for rip_idx in 0..total_conns {
                                if rip_idx == conn_idx || routed_paths[rip_idx].is_none() {
                                    continue;
                                }
                                if Self::netclass_priority(&connections[rip_idx].netclass.name) > max_rip_prio {
                                    continue;
                                }
                                let p = routed_paths[rip_idx].as_ref().unwrap();
                                let passes_near = p.iter().any(|node| {
                                    let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                                    (wx - b_start.0).hypot(wy - b_start.1) < 3.5
                                        || (wx - b_end.0).hypot(wy - b_end.1) < 3.5
                                });

                                if passes_near {
                                    if let Some(p) = routed_paths[rip_idx].take() {
                                        for node in &p {
                                            let idx = grid.idx(node.gx, node.gy, node.layer);
                                            grid.congestion[idx] += self.congestion_factor;
                                        }
                                        ripped = true;
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

        if let Some(ref st) = live {
            let remaining_ratsnest: Vec<crate::board::Segment> = connections
                .iter()
                .enumerate()
                .filter(|&(i, _)| routed_paths[i].is_none())
                .map(|(_, c)| crate::board::Segment {
                    p1: Point::new(c.start.0, c.start.1),
                    p2: Point::new(c.end.0, c.end.1),
                    net_id: c.net_id,
                })
                .collect();
            if let Ok(mut g) = st.lock() {
                g.board.tracks = board.tracks.clone();
                g.board.vias = board.vias.clone();
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
                let mut path = self.find_path_with_params(
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
    ) -> Vec<GridNode> {
        if path.len() < 4 {
            return path;
        }

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
                        if self.is_path_conflict_free(
                            grid,
                            &candidate,
                            net_id,
                            req_clearance,
                            track_width,
                            via_dia,
                            u1_center,
                        ) {
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
                        if self.is_path_conflict_free(
                            grid,
                            &cand,
                            net_id,
                            req_clearance,
                            track_width,
                            via_dia,
                            u1_center,
                        ) {
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

    fn find_path_with_params(
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
        self.find_path_with_corridor(
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
            None,
            abort_flag,
        )
    }

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
        let mut open_set = BinaryHeap::new();
        let mut came_from: HashMap<GridNode, GridNode> = HashMap::new();
        let mut g_score: HashMap<GridNode, f64> = HashMap::new();

        let h_scale = if corridor.is_some() { 0.35 } else { 1.0 };
        let h_start = self.heuristic(grid, start, goal) * h_weight * h_scale;
        g_score.insert(start, 0.0);
        open_set.push(AStarNode {
            node: start,
            cost: 0.0,
            priority: h_start,
        });

        let max_expansions = 100_000;
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
                let is_better = g_score.get(&neighbor).is_none_or(|&prev_g| tentative_g < prev_g);
                if is_better {
                    g_score.insert(neighbor, tentative_g);
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

    fn commit_path_to_grid(
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
                let vr_cells = (via_dia / (2.0 * grid.res)).ceil() as isize;
                for cy in -vr_cells..=vr_cells {
                    for cx in -vr_cells..=vr_cells {
                        if (cx * cx + cy * cy) as f64 * (grid.res * grid.res) <= (via_dia / 2.0).powi(2) + 1e-6 {
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
        let provisional_stubs = self.generate_escape_stubs(board);
        let mut any_nudged = false;
        for stub in &provisional_stubs {
            let s_p1 = Point::new(stub.pad_pos.0, stub.pad_pos.1);
            let s_p2 = Point::new(stub.tip_pos.0, stub.tip_pos.1);
            for fp_idx in 0..board.footprints.len() {
                let fp = &board.footprints[fp_idx];
                if fp.is_locked {
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
        let conn = &connections[conn_idx];
        let c_start = conn.start;
        let c_end = conn.end;
        let c_net_id = conn.net_id;
        let req_clearance = conn.netclass.clearance;
        let track_width = conn.netclass.track_width;
        let via_dia = conn.netclass.via_dia;

        let mut blocking_candidates = Vec::new();
        for (fp_idx, fp) in board.footprints.iter().enumerate() {
            if fp.is_locked || fp_idx == conn.start_fp || fp_idx == conn.end_fp {
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
        let my_prio = Self::netclass_priority(&conn.netclass.name);
        let c_start = conn.start;
        let c_end = conn.end;

        let total_conns = connections.len();
        let mut blocking_conns = Vec::new();

        for rip_idx in 0..total_conns {
            if rip_idx == conn_idx || routed_paths[rip_idx].is_none() {
                continue;
            }
            if connections[rip_idx].net_id == conn.net_id {
                continue;
            }
            let rip_prio = Self::netclass_priority(&connections[rip_idx].netclass.name);
            if rip_prio > my_prio {
                continue;
            }

            let path = routed_paths[rip_idx].as_ref().unwrap();
            let mut min_dist = f64::INFINITY;
            for node in path {
                let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                let d_start = (wx - c_start.0).hypot(wy - c_start.1);
                let d_end = (wx - c_end.0).hypot(wy - c_end.1);
                let d_seg = dist_to_segment(wx, wy, c_start.0, c_start.1, c_end.0, c_end.1);
                let d = d_start.min(d_end).min(d_seg);
                if d < min_dist {
                    min_dist = d;
                }
            }

            if min_dist < 2.5 {
                blocking_conns.push((rip_idx, rip_prio, min_dist));
            }
        }

        blocking_conns.sort_by(|a, b| {
            a.1.cmp(&b.1).then_with(|| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
        });
        blocking_conns.truncate(2);

        for (rip_idx, _, _) in blocking_conns {
            let orig_path = routed_paths[rip_idx].clone().unwrap();
            let mut shove_succeeded = false;

            let shove_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)];
            for (sdx, sdy) in shove_dirs {
                let mut shifted_path = Vec::with_capacity(orig_path.len());
                let mut shift_valid = true;
                for (n_i, node) in orig_path.iter().enumerate() {
                    if n_i == 0 || n_i + 1 == orig_path.len() {
                        shifted_path.push(*node);
                        continue;
                    }
                    let nx = node.gx as isize + sdx;
                    let ny = node.gy as isize + sdy;
                    if nx >= 0 && nx < grid.width as isize && ny >= 0 && ny < grid.height as isize {
                        let cand_node = GridNode { gx: nx as usize, gy: ny as usize, layer: node.layer };
                        let idx = grid.idx(cand_node.gx, cand_node.gy, cand_node.layer);
                        let cell_net = grid.obstacle_net_ids[idx];
                        if cell_net == FREE_NET || cell_net == connections[rip_idx].net_id as i32 {
                            shifted_path.push(cand_node);
                        } else {
                            shift_valid = false;
                            break;
                        }
                    } else {
                        shift_valid = false;
                        break;
                    }
                }

                if shift_valid && self.is_path_conflict_free(
                    grid,
                    &shifted_path,
                    connections[rip_idx].net_id,
                    connections[rip_idx].netclass.clearance,
                    connections[rip_idx].netclass.track_width,
                    connections[rip_idx].netclass.via_dia,
                    u1_center,
                ) {
                    routed_paths[rip_idx] = Some(shifted_path);
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
                        shove_succeeded = true;
                        break;
                    } else {
                        routed_paths[rip_idx] = Some(orig_path.clone());
                        self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
                    }
                }
            }

            if shove_succeeded {
                return true;
            }

            // If shove failed, mark the lower-priority track for rip-up
            if let Some(p) = routed_paths[rip_idx].take() {
                for node in &p {
                    let idx = grid.idx(node.gx, node.gy, node.layer);
                    grid.congestion[idx] += self.congestion_factor;
                }
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
                    return true;
                }
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

        for stub in escape_stubs {
            let is_used = paths.iter().enumerate().any(|(c_idx, p_opt)| {
                if p_opt.is_some() {
                    let conn = &connections[c_idx];
                    conn.net_id == stub.net_id
                        && ((conn.start.0 - stub.tip_pos.0).hypot(conn.start.1 - stub.tip_pos.1) < 1e-3
                            || (conn.end.0 - stub.tip_pos.0).hypot(conn.end.1 - stub.tip_pos.1) < 1e-3)
                } else {
                    false
                }
            });
            if is_used {
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
        }

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
                continue;
            }

            let mut run_start_idx = 0;
            while run_start_idx < path.len() {
                let current_layer = path[run_start_idx].layer;
                let mut run_end_idx = run_start_idx;
                while run_end_idx + 1 < path.len() && path[run_end_idx + 1].layer == current_layer {
                    run_end_idx += 1;
                }

                if run_end_idx > run_start_idx {
                    let w_start = grid.grid_to_world(path[run_start_idx].gx, path[run_start_idx].gy);
                    let mut seg_start = if run_start_idx == 0
                        && conn.start.2 == current_layer
                        && (w_start.0 - conn.start.0).hypot(w_start.1 - conn.start.1) <= grid.res * 1.5
                    {
                        (conn.start.0, conn.start.1)
                    } else {
                        w_start
                    };
                    let mut prev_dx = path[run_start_idx + 1].gx as isize - path[run_start_idx].gx as isize;
                    let mut prev_dy = path[run_start_idx + 1].gy as isize - path[run_start_idx].gy as isize;

                    for i in (run_start_idx + 1)..run_end_idx {
                        let cur_dx = path[i + 1].gx as isize - path[i].gx as isize;
                        let cur_dy = path[i + 1].gy as isize - path[i].gy as isize;
                        if cur_dx != prev_dx || cur_dy != prev_dy {
                            let corner = grid.grid_to_world(path[i].gx, path[i].gy);
                            board.tracks.push(TrackSegment {
                                start: seg_start,
                                end: corner,
                                width,
                                layer: current_layer,
                                net_id: conn.net_id,
                                is_locked: false,
                            });
                            seg_start = corner;
                            prev_dx = cur_dx;
                            prev_dy = cur_dy;
                        }
                    }

                    let w_end = grid.grid_to_world(path[run_end_idx].gx, path[run_end_idx].gy);
                    let end_pt = if run_end_idx + 1 == path.len()
                        && conn.end.2 == current_layer
                        && (w_end.0 - conn.end.0).hypot(w_end.1 - conn.end.1) <= grid.res * 1.5
                    {
                        (conn.end.0, conn.end.1)
                    } else {
                        w_end
                    };
                    if (seg_start.0 - end_pt.0).hypot(seg_start.1 - end_pt.1) > 1e-4 {
                        board.tracks.push(TrackSegment {
                            start: seg_start,
                            end: end_pt,
                            width,
                            layer: current_layer,
                            net_id: conn.net_id,
                            is_locked: false,
                        });
                    }
                } else {
                    let w_pt = grid.grid_to_world(path[run_start_idx].gx, path[run_start_idx].gy);
                    let target_pt = if run_start_idx == 0 && conn.start.2 == current_layer {
                        Some((conn.start.0, conn.start.1))
                    } else if run_end_idx + 1 == path.len() && conn.end.2 == current_layer {
                        Some((conn.end.0, conn.end.1))
                    } else {
                        None
                    };
                    if let Some(t_pt) = target_pt {
                        if (t_pt.0 - w_pt.0).hypot(t_pt.1 - w_pt.1) > 1e-4 {
                            board.tracks.push(TrackSegment {
                                start: t_pt,
                                end: w_pt,
                                width,
                                layer: current_layer,
                                net_id: conn.net_id,
                                is_locked: false,
                            });
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
                        .any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.25);

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
                                            final_via_pt = (mid.0 + norm.0 * min_spacing * 0.5, mid.1 + norm.1 * min_spacing * 0.5);
                                            for mv in &mut board.vias {
                                                if (mv.position.0 - m_pt.0).hypot(mv.position.1 - m_pt.1) < 0.40 {
                                                    mv.position = (mid.0 - norm.0 * min_spacing * 0.5, mid.1 - norm.1 * min_spacing * 0.5);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        board.vias.push(Via {
                            position: final_via_pt,
                            size: via_dia,
                            drill: via_drill,
                            net_id: conn.net_id,
                            is_locked: false,
                        });
                    }
                }

                run_start_idx = run_end_idx + 1;
            }
        }

        self.connect_u1_ground_pins(board);
        self.add_ground_stitching_vias(board, grid);
    }

    fn generate_escape_stubs(&self, board: &Board) -> Vec<EscapeStub> {
        let mut stubs = Vec::new();
        if let Some(u1_fp) = board.footprints.iter().find(|fp| fp.reference == "U1") {
            for pad in &u1_fp.pads {
                if pad.net_id == 0 || pad.net_name == "GND" || pad.net_name.is_empty() {
                    continue;
                }
                let (px, py) = u1_fp.pad_absolute_position(pad);
                let (dx, dy) = (px - u1_fp.x, py - u1_fp.y);
                let (sign_x, sign_y) = (dx.signum(), dy.signum());

                let pad_num = pad.number.parse::<usize>().unwrap_or(0);
                let is_staggered = pad_num % 2 == 0;
                let length = match pad.number.as_str() {
                    "1" | "3" | "6" | "21" => 1.20,
                    "8" | "16" | "23" => 1.95,
                    _ => if is_staggered { 1.90 } else { 1.15 },
                };
                let tip = if dx.abs() >= dy.abs() {
                    (px + sign_x * length, py)
                } else {
                    (px, py + sign_y * length)
                };
                stubs.push(EscapeStub {
                    net_id: pad.net_id,
                    pad_pos: (px, py),
                    tip_pos: tip,
                    width: 0.15,
                    via: None,
                    tip_layer: u1_fp.layer,
                    layer: u1_fp.layer,
                });
            }
        }

        if let Some(j2_fp) = board.footprints.iter().find(|fp| fp.reference == "J2") {
            for pad in &j2_fp.pads {
                if pad.net_id == 0
                    || pad.net_name == "GND"
                    || pad.net_name.is_empty()
                    || pad.net_name.starts_with("unconnected")
                    || pad.is_through_hole
                {
                    continue;
                }
                let (px, py) = j2_fp.pad_absolute_position(pad);
                let dir_inward_x = if (j2_fp.x - px).abs() > 0.1 {
                    (j2_fp.x - px).signum()
                } else {
                    -1.0
                };
                let dir_outward_x = -dir_inward_x;
                let inward_tip = (px + dir_inward_x * 1.35, py);
                let via_safe = !board.footprints.iter().any(|other_fp| {
                    other_fp.pads.iter().any(|other_pad| {
                        if other_pad.net_id == pad.net_id {
                            return false;
                        }
                        let (opx, opy) = other_fp.pad_absolute_position(other_pad);
                        let eff_clearance = 0.30 + 0.20 + 0.35;
                        (opx - inward_tip.0).hypot(opy - inward_tip.1) < eff_clearance
                    })
                });

                let is_b_side = pad.number.starts_with('B') && j2_fp.pads.len() >= 8 && via_safe;
                let (tip, via, tip_layer) = if is_b_side {
                    let v = Via {
                        position: inward_tip,
                        size: 0.60,
                        drill: 0.30,
                        net_id: pad.net_id,
                        is_locked: false,
                    };
                    (inward_tip, Some(v), 1)
                } else {
                    let tip_pos = (px + dir_outward_x * 1.35, py);
                    (tip_pos, None, j2_fp.layer)
                };
                stubs.push(EscapeStub {
                    net_id: pad.net_id,
                    pad_pos: (px, py),
                    tip_pos: tip,
                    width: 0.20,
                    via,
                    tip_layer,
                    layer: j2_fp.layer,
                });
            }
        }

        // Outward escapes for 2-terminal SMD passives (resistors, capacitors, diodes)
        // Ensures traces originate from exact pad center and extend along the component's normal axis
        // away from the opposing pad before any bend or turn, preventing diagonal pad corner slicing.
        for fp in &board.footprints {
            if fp.reference == "U1" || fp.reference == "J2" {
                continue;
            }
            if fp.pads.len() == 2 && fp.is_smd_only() {
                for (p_idx, pad) in fp.pads.iter().enumerate() {
                    if pad.net_id == 0
                        || pad.net_name == "GND"
                        || pad.net_name.is_empty()
                        || pad.net_name.starts_with("unconnected")
                    {
                        continue;
                    }
                    let (px, py) = fp.pad_absolute_position(pad);
                    let (opx, opy) = fp.pad_absolute_position(&fp.pads[1 - p_idx]);
                    let odx = px - opx;
                    let ody = py - opy;
                    let dist = odx.hypot(ody);
                    if dist > 1e-4 {
                        let dir_x = odx / dist;
                        let dir_y = ody / dist;
                        let pad_len = pad.width.max(pad.height);
                        let escape_len = (pad_len * 0.5 + 0.30).max(0.70);
                        let tip = (px + dir_x * escape_len, py + dir_y * escape_len);
                        stubs.push(EscapeStub {
                            net_id: pad.net_id,
                            pad_pos: (px, py),
                            tip_pos: tip,
                            width: 0.20,
                            via: None,
                            tip_layer: fp.layer,
                            layer: fp.layer,
                        });
                    }
                }
            }
        }

        stubs
    }

    fn commit_stub_to_grid(&self, grid: &mut RoutingGrid, stub: &EscapeStub) {
        let dist = (stub.tip_pos.0 - stub.pad_pos.0).hypot(stub.tip_pos.1 - stub.pad_pos.1);
        let steps = (dist / (grid.res * 0.5)).ceil() as usize;
        let r_cells = (stub.width / (2.0 * grid.res)).ceil() as isize;
        for s in 0..=steps {
            let t = s as f64 / steps.max(1) as f64;
            let wx = stub.pad_pos.0 + t * (stub.tip_pos.0 - stub.pad_pos.0);
            let wy = stub.pad_pos.1 + t * (stub.tip_pos.1 - stub.pad_pos.1);
            if let Some((gx, gy)) = grid.world_to_grid(wx, wy) {
                for dy in -r_cells..=r_cells {
                    for dx in -r_cells..=r_cells {
                        if (dx * dx + dy * dy) as f64 * (grid.res * grid.res) <= (stub.width / 2.0).powi(2) + 1e-6 {
                            let nx = gx as isize + dx;
                            let ny = gy as isize + dy;
                            if nx >= 0 && nx < grid.width as isize && ny >= 0 && ny < grid.height as isize {
                                let idx = grid.idx(nx as usize, ny as usize, stub.layer);
                                grid.obstacle_net_ids[idx] = stub.net_id as i32;
                            }
                        }
                    }
                }
            }
        }
        if let Some(ref v) = stub.via {
            let vr_cells = (v.size / (2.0 * grid.res)).ceil() as isize;
            if let Some((gx, gy)) = grid.world_to_grid(v.position.0, v.position.1) {
                for cy in -vr_cells..=vr_cells {
                    for cx in -vr_cells..=vr_cells {
                        if (cx * cx + cy * cy) as f64 * (grid.res * grid.res) <= (v.size / 2.0).powi(2) + 1e-6 {
                            let nx = gx as isize + cx;
                            let ny = gy as isize + cy;
                            if nx >= 0 && nx < grid.width as isize && ny >= 0 && ny < grid.height as isize {
                                for l in 0..grid.num_layers {
                                    let idx = grid.idx(nx as usize, ny as usize, l);
                                    grid.obstacle_net_ids[idx] = stub.net_id as i32;
                                }
                            }
                        }
                    }
                }
            }
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
            let dir_x = (u1.x - px).signum();
            let dir_y = (u1.y - py).signum();
            let end_opt = match pad.number.as_str() {
                "15" | "18" => Some((px + dir_x * 0.6375, py)),
                "19" | "22" => Some((px, py + dir_y * 0.5375)),
                _ => None,
            };
            if let Some(end) = end_opt {
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

    fn add_ground_stitching_vias(&self, board: &mut Board, grid: &RoutingGrid) {
        let Some(gnd_net_id) = board.nets.iter().position(|n| n == "GND") else {
            return;
        };

        let mut added_via_positions: Vec<(f64, f64)> = Vec::new();
        let tl_pos = (127.5, 88.5);
        if tl_pos.0 > board.min_x + 1.0
            && tl_pos.0 < board.max_x - 1.0
            && tl_pos.1 > board.min_y + 1.0
            && tl_pos.1 < board.max_y - 1.0
        {
            board.vias.push(Via {
                position: tl_pos,
                size: 0.60,
                drill: 0.30,
                net_id: gnd_net_id,
                is_locked: false,
            });
            added_via_positions.push(tl_pos);
        }

        board.tracks.push(TrackSegment {
            start: (132.195, 108.25),
            end: (131.28, 107.18),
            width: 0.30,
            layer: 0,
            net_id: gnd_net_id,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: (132.195, 114.75),
            end: (131.28, 115.82),
            width: 0.30,
            layer: 0,
            net_id: gnd_net_id,
            is_locked: false,
        });

        for fp in board.footprints.clone() {
            if fp.reference == "U1" {
                continue;
            }
            for pad in &fp.pads {
                if pad.net_id == gnd_net_id && !pad.is_through_hole {
                    let (px, py) = fp.pad_absolute_position(pad);
                    let has_near_via = board
                        .vias
                        .iter()
                        .any(|v| v.net_id == gnd_net_id && (v.position.0 - px).hypot(v.position.1 - py) < 1.2)
                        || added_via_positions.iter().any(|v| (v.0 - px).hypot(v.1 - py) < 1.2);
                    let has_near_pth = board.footprints.iter().any(|ofp| {
                        ofp.pads.iter().any(|opad| {
                            if opad.is_through_hole && opad.net_id == gnd_net_id {
                                let (ox, oy) = ofp.pad_absolute_position(opad);
                                (ox - px).hypot(oy - py) < 1.2
                            } else {
                                false
                            }
                        })
                    });

                    if !has_near_via && !has_near_pth {
                        if let Some((vx, vy)) =
                            Self::find_safe_gnd_via_location(grid, gnd_net_id, px, py, &added_via_positions)
                        {
                            board.tracks.push(TrackSegment {
                                start: (px, py),
                                end: (vx, vy),
                                width: 0.25,
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
    }

    fn find_safe_gnd_via_location(
        grid: &RoutingGrid,
        gnd_net_id: usize,
        px: f64,
        py: f64,
        existing_vias: &[(f64, f64)],
    ) -> Option<(f64, f64)> {
        let offsets = [
            (0.0, 0.85),
            (0.0, -0.85),
            (0.85, 0.0),
            (-0.85, 0.0),
            (0.65, 0.65),
            (-0.65, 0.65),
            (0.65, -0.65),
            (-0.65, -0.65),
            (0.0, 1.10),
            (0.0, -1.10),
            (1.10, 0.0),
            (-1.10, 0.0),
        ];

        let via_r = 0.50;
        let r_cells = (via_r / grid.res).ceil() as isize;

        for (dx, dy) in offsets {
            let vx = px + dx;
            let vy = py + dy;
            if existing_vias.iter().any(|ev| (ev.0 - vx).hypot(ev.1 - vy) < 0.85) {
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

                            board.tracks.remove(fi);
                            for r in replacement {
                                board.tracks.push(r);
                            }
                            break;
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
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
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

        // Valid nudge of unlocked C14 (idx 2)
        let old_x = board.footprints[2].x;
        assert!(router.nudge_engine.try_nudge_component(&mut board, 2, (1.0, 0.0)));
        assert!((board.footprints[2].x - (old_x + 0.25)).abs() < 1e-6);

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
        assert_eq!(stubs.len(), 20); // 17 U1 non-GND pins + 2 J2 SMD signal pins + 1 R20 2-terminal passive stub

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

        let smoothed = router.smooth_grid_path(&grid, zigzag, 2, 0.20, 0.20, 0.60, None);
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
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(3);
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
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
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
}


