// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use crate::board::{Board, LiveTrialState, NetClass, Point, TrackSegment, TrialPhase, Via};
use crate::vulkan::VulkanEngine;
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
    pub obstacle_net_ids: Vec<i32>,
    pub dist_field: Vec<f32>,
    pub congestion: Vec<f32>,
    pub nudgeable_fp: Vec<Option<usize>>,
    pub smd_pad_cells: Vec<bool>,
    pub net_clearances: Vec<f64>,
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
        let num_layers = 2;
        let total_cells = num_layers * width * height;

        let mut net_clearances = vec![0.20; board.nets.len()];
        for i in 0..board.nets.len() {
            net_clearances[i] = board.get_netclass_for_net(i).clearance;
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
            obstacle_net_ids: vec![FREE_NET; total_cells],
            dist_field: vec![999.0; total_cells],
            congestion: vec![0.0; total_cells],
            nudgeable_fp: vec![None; total_cells],
            smd_pad_cells: vec![false; total_cells],
            net_clearances,
            vulkan,
        };

        grid.rasterize_obstacles(board);
        grid.update_distance_fields();
        grid
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
        for ko in &board.keepouts {
            let (min_gx, min_gy) = self.world_to_grid(ko.min_x, ko.min_y).unwrap_or((0, 0));
            let (max_gx, max_gy) = self
                .world_to_grid(ko.max_x, ko.max_y)
                .unwrap_or((self.width.saturating_sub(1), self.height.saturating_sub(1)));
            for &layer in &ko.layers {
                if layer >= self.num_layers {
                    continue;
                }
                for gy in min_gy..=max_gy.min(self.height - 1) {
                    for gx in min_gx..=max_gx.min(self.width - 1) {
                        let i = self.idx(gx, gy, layer);
                        self.obstacle_net_ids[i] = KEEPOUT_NET;
                    }
                }
            }
        }

        for (fp_idx, fp) in board.footprints.iter().enumerate() {
            let is_nudgeable = !fp.is_locked;
            for pad in &fp.pads {
                let (px, py) = fp.pad_absolute_position(pad);
                let rad = (-fp.rotation).to_radians();
                let cos_r = rad.cos().abs();
                let sin_r = rad.sin().abs();
                let half_w = (pad.width * cos_r + pad.height * sin_r) / 2.0;
                let half_h = (pad.width * sin_r + pad.height * cos_r) / 2.0;
                let layers: &[usize] = if pad.is_through_hole { &[0, 1] } else { &[0] };

                for &layer in layers {
                    let min_gx_calc = ((px - half_w - self.min_x) / self.res).ceil().max(0.0) as usize;
                    let max_gx_calc = ((px + half_w - self.min_x) / self.res)
                        .floor()
                        .min(self.width as f64 - 1.0) as usize;
                    let min_gy_calc = ((py - half_h - self.min_y) / self.res).ceil().max(0.0) as usize;
                    let max_gy_calc = ((py + half_h - self.min_y) / self.res)
                        .floor()
                        .min(self.height as f64 - 1.0) as usize;

                    let (min_gx, max_gx) = if min_gx_calc > max_gx_calc {
                        let mid = ((px - self.min_x) / self.res)
                            .round()
                            .clamp(0.0, self.width as f64 - 1.0) as usize;
                        (mid, mid)
                    } else {
                        (min_gx_calc, max_gx_calc)
                    };
                    let (min_gy, max_gy) = if min_gy_calc > max_gy_calc {
                        let mid = ((py - self.min_y) / self.res)
                            .round()
                            .clamp(0.0, self.height as f64 - 1.0) as usize;
                        (mid, mid)
                    } else {
                        (min_gy_calc, max_gy_calc)
                    };

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
        u1_center: Option<(f64, f64)>,
    ) -> Vec<(GridNode, f64)> {
        let mut neighbors = Vec::with_capacity(10);
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

        let (h_mult, v_mult): (f64, f64) = if layer == 0 { (1.0, 1.35) } else { (1.35, 1.0) };

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

                let max_clearance = 0.25;
                let search_clearance = (track_width / 2.0) + max_clearance;
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
                                let obst_c = if c_net > 0 && (c_net as usize) < self.net_clearances.len() {
                                    self.net_clearances[c_net as usize]
                                } else {
                                    req_clearance
                                };
                                let eff_clearance = (track_width / 2.0) + req_clearance.max(obst_c);
                                let c_dist_sq = (cx_off * cx_off + cy_off * cy_off) as f64 * (self.res * self.res);
                                if c_dist_sq < eff_clearance.powi(2) {
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

                let mut qfn_penalty = 0.0;
                if layer == 0 {
                    if let Some((u1x, u1y)) = u1_center {
                        let wx = ngx as f64 * self.res + self.min_x;
                        let wy = ngy as f64 * self.res + self.min_y;
                        let u1_dx = wx - u1x;
                        let u1_dy = wy - u1y;
                        if u1_dx.abs() <= 3.8 && u1_dy.abs() <= 3.8 {
                            if u1_dx.abs() >= u1_dy.abs() {
                                if dy != 0 {
                                    qfn_penalty = 30.0;
                                }
                            } else if dx != 0 {
                                qfn_penalty = 30.0;
                            }
                        }
                    }
                }

                let cong = self.congestion[n_idx] as f64;
                let step_cost = (base_len * self.res) * dir_factor * (1.0 + cong) + qfn_penalty;
                neighbors.push((GridNode { gx: ngx, gy: ngy, layer }, step_cost));
            }
        }

        let other_layer = 1 - layer;
        let curr_idx = self.idx(gx, gy, layer);
        let other_idx = self.idx(gx, gy, other_layer);
        let other_net = self.obstacle_net_ids[other_idx];
        let inside_smd_pad = self.smd_pad_cells[curr_idx] || self.smd_pad_cells[other_idx];

        if other_net == target_net_id as i32 && !inside_smd_pad {
            neighbors.push((GridNode { gx, gy, layer: other_layer }, 0.1));
        } else if other_net == FREE_NET && !inside_smd_pad {
            let max_clearance = 0.25;
            let search_clearance = (via_dia / 2.0) + max_clearance;
            let vr_cells = (search_clearance / self.res).ceil() as isize;
            let mut via_clearance_violation = false;
            for cy_off in -vr_cells..=vr_cells {
                for cx_off in -vr_cells..=vr_cells {
                    let cx = gx as isize + cx_off;
                    let cy = gy as isize + cy_off;
                    if cx >= 0 && cx < self.width as isize && cy >= 0 && cy < self.height as isize {
                        let c_idx_other = self.idx(cx as usize, cy as usize, other_layer);
                        let c_idx_curr = self.idx(cx as usize, cy as usize, layer);
                        let c_net_other = self.obstacle_net_ids[c_idx_other];
                        let c_net_curr = self.obstacle_net_ids[c_idx_curr];
                        let d_sq = (cx_off * cx_off + cy_off * cy_off) as f64 * (self.res * self.res);

                        if c_net_curr != FREE_NET && c_net_curr != target_net_id as i32 {
                            let obst_c = if c_net_curr > 0 && (c_net_curr as usize) < self.net_clearances.len() {
                                self.net_clearances[c_net_curr as usize]
                            } else {
                                req_clearance
                            };
                            let eff_c = (via_dia / 2.0) + req_clearance.max(obst_c);
                            if d_sq < eff_c.powi(2) {
                                via_clearance_violation = true;
                                break;
                            }
                        }

                        if c_net_other != FREE_NET && c_net_other != target_net_id as i32 {
                            let obst_c = if c_net_other > 0 && (c_net_other as usize) < self.net_clearances.len() {
                                self.net_clearances[c_net_other as usize]
                            } else {
                                req_clearance
                            };
                            let eff_c = (via_dia / 2.0) + req_clearance.max(obst_c);
                            if d_sq < eff_c.powi(2) {
                                via_clearance_violation = true;
                                break;
                            }
                        }

                        if (self.smd_pad_cells[c_idx_curr] && c_net_curr != target_net_id as i32)
                            || (self.smd_pad_cells[c_idx_other] && c_net_other != target_net_id as i32)
                        {
                            let eff_c = via_dia / 2.0;
                            if d_sq < eff_c.powi(2) {
                                via_clearance_violation = true;
                                break;
                            }
                        }
                    }
                }
                if via_clearance_violation {
                    break;
                }
            }

            if !via_clearance_violation {
                neighbors.push((GridNode { gx, gy, layer: other_layer }, 2.0));
            }
        }

        neighbors
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

    pub fn try_nudge_component(&self, board: &mut Board, fp_idx: usize, push_dir: (f64, f64)) -> bool {
        if fp_idx >= board.footprints.len() || board.footprints[fp_idx].is_locked {
            return false;
        }
        let mag = (push_dir.0 * push_dir.0 + push_dir.1 * push_dir.1).sqrt();
        if mag < 1e-6 {
            return false;
        }
        let ux = push_dir.0 / mag;
        let uy = push_dir.1 / mag;

        let candidates = [
            (ux * self.step_mm, uy * self.step_mm),
            (-uy * self.step_mm, ux * self.step_mm),
            (uy * self.step_mm, -ux * self.step_mm),
            (ux * self.step_mm * 2.0, uy * self.step_mm * 2.0),
        ];

        let orig_x = board.footprints[fp_idx].x;
        let orig_y = board.footprints[fp_idx].y;
        let rot = board.footprints[fp_idx].rotation;
        let boundary = board.boundary_aabb();

        for (dx, dy) in candidates {
            let nx = orig_x + dx;
            let ny = orig_y + dy;
            let new_polys = board.footprints[fp_idx].transformed_courtyards(Point::new(nx, ny), rot, 0.25);

            let mut valid = true;
            for poly in &new_polys {
                for v in &poly.vertices {
                    if !boundary.contains_point(v) {
                        valid = false;
                        break;
                    }
                }
            }
            if !valid {
                continue;
            }

            for (other_idx, other_fp) in board.footprints.iter().enumerate() {
                if other_idx == fp_idx {
                    continue;
                }
                let other_polys = other_fp.transformed_courtyards(Point::new(other_fp.x, other_fp.y), other_fp.rotation, 0.25);
                for pa in &new_polys {
                    for pb in &other_polys {
                        if pa.intersects_convex(pb) {
                            valid = false;
                            break;
                        }
                    }
                }
                if !valid {
                    break;
                }
            }

            if valid {
                board.footprints[fp_idx].x = nx;
                board.footprints[fp_idx].y = ny;
                return true;
            }
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
}

#[derive(Clone, Debug)]
pub struct EscapeStub {
    pub net_id: usize,
    pub pad_pos: (f64, f64),
    pub tip_pos: (f64, f64),
    pub width: f64,
}

pub struct AutoRouter {
    pub nudge_engine: NudgeEngine,
    pub max_ripup_iterations: usize,
    pub congestion_factor: f32,
}

impl AutoRouter {
    pub fn new(max_ripup_iterations: usize) -> Self {
        Self {
            nudge_engine: NudgeEngine::new(),
            max_ripup_iterations,
            congestion_factor: 1.5,
        }
    }

    pub fn extract_connections(&self, board: &Board) -> Vec<RouteConnection> {
        #[derive(Clone)]
        struct PadInfo {
            x: f64,
            y: f64,
            layer: usize,
            fp_idx: usize,
            pad_idx: usize,
        }

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
                let entry = net_pads.entry(pad.net_id).or_default();
                if !entry.iter().any(|p| (p.x - px).hypot(p.y - py) < 0.05) {
                    entry.push(PadInfo {
                        x: px,
                        y: py,
                        layer: 0,
                        fp_idx,
                        pad_idx,
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
                let start_pt = (p_start.x, p_start.y, p_start.layer);
                let end_pt = (p_end.x, p_end.y, p_end.layer);

                let is_fine_pitch = board.footprints.iter().any(|fp| {
                    matches!(fp.reference.as_str(), "U1" | "J2" | "U2")
                        && fp.pads.iter().any(|p| {
                            let (px, py) = fp.pad_absolute_position(p);
                            (px - start_pt.0).hypot(py - start_pt.1) < 0.1
                                || (px - end_pt.0).hypot(py - end_pt.1) < 0.1
                        })
                });

                let mut eff_netclass = netclass.clone();
                if is_fine_pitch {
                    eff_netclass.track_width = eff_netclass.track_width.min(0.20);
                    eff_netclass.clearance = eff_netclass.clearance.min(0.20);
                }

                connections.push(RouteConnection {
                    net_id,
                    net_name: net_name.clone(),
                    start: start_pt,
                    end: end_pt,
                    start_fp: p_start.fp_idx,
                    start_pad: p_start.pad_idx,
                    end_fp: p_end.fp_idx,
                    end_pad: p_end.pad_idx,
                    netclass: eff_netclass,
                });
            }
        }

        connections.sort_by(|a, b| {
            let prio_a = Self::netclass_priority(&a.netclass.name);
            let prio_b = Self::netclass_priority(&b.netclass.name);
            let a_is_ic = matches!(board.footprints[a.start_fp].reference.as_str(), "U1" | "J2")
                || matches!(board.footprints[a.end_fp].reference.as_str(), "U1" | "J2");
            let b_is_ic = matches!(board.footprints[b.start_fp].reference.as_str(), "U1" | "J2")
                || matches!(board.footprints[b.end_fp].reference.as_str(), "U1" | "J2");

            b_is_ic
                .cmp(&a_is_ic)
                .then_with(|| prio_b.cmp(&prio_a))
                .then_with(|| {
                    let d_a = (a.start.0 - a.end.0).hypot(a.start.1 - a.end.1);
                    let d_b = (b.start.0 - b.end.0).hypot(b.start.1 - b.end.1);
                    d_a.partial_cmp(&d_b).unwrap_or(Ordering::Equal)
                })
        });

        connections
    }

    fn netclass_priority(name: &str) -> usize {
        match name {
            "Switching" => 10,
            "Power_15V" => 9,
            "USB_Diff" => 8,
            "HBS_Bus" => 7,
            "Power_3V3" => 6,
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
        let mut connections = self.extract_connections(board);
        let total_conns = connections.len();

        let escape_stubs = self.generate_escape_stubs(board);
        for stub in &escape_stubs {
            self.commit_stub_to_grid(grid, stub);
        }

        for conn in &mut connections {
            for stub in &escape_stubs {
                if (conn.start.0 - stub.pad_pos.0).hypot(conn.start.1 - stub.pad_pos.1) < 0.1 {
                    conn.start = (stub.tip_pos.0, stub.tip_pos.1, 0);
                    break;
                }
            }
            for stub in &escape_stubs {
                if (conn.end.0 - stub.pad_pos.0).hypot(conn.end.1 - stub.pad_pos.1) < 0.1 {
                    conn.end = (stub.tip_pos.0, stub.tip_pos.1, 0);
                    break;
                }
            }
        }

        let u1_center = board
            .footprints
            .iter()
            .find(|fp| fp.reference == "U1")
            .map(|fp| (fp.x, fp.y));

        if let Some(ref st) = live {
            if let Ok(mut g) = st.lock() {
                g.phase = TrialPhase::AutoRouting;
                g.total_conns = total_conns;
                g.routed_conns = 0;
                g.push_log(format!(
                    "[Auto-Router] Routing {} connections with {} escape stubs...",
                    total_conns,
                    escape_stubs.len()
                ));
            }
        }

        let mut routed_paths: Vec<Option<Vec<GridNode>>> = vec![None; total_conns];

        for iteration in 0..self.max_ripup_iterations {
            if stop_flag.load(AtomicOrdering::Relaxed) {
                break;
            }
            let unrouted_count = routed_paths.iter().filter(|p| p.is_none()).count();
            if unrouted_count == 0 {
                break;
            }

            for conn_idx in 0..total_conns {
                if stop_flag.load(AtomicOrdering::Relaxed) {
                    break;
                }
                if routed_paths[conn_idx].is_some() {
                    continue;
                }
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

                let path = self.find_path(
                    grid,
                    start_grid,
                    end_grid,
                    conn_net_id,
                    req_clearance,
                    track_width,
                    via_dia,
                    u1_center,
                );

                if let Some(p) = path {
                    self.commit_path_to_grid(grid, &p, conn_net_id, track_width, via_dia);
                    routed_paths[conn_idx] = Some(p);
                }

                // Stream live progress to GUI every 4 connections
                if conn_idx % 4 == 0 || conn_idx + 1 == total_conns {
                    if let Some(ref st) = live {
                        let done = routed_paths.iter().filter(|p| p.is_some()).count();
                        self.reconstruct_tracks_and_vias(board, grid, &connections, &routed_paths, &escape_stubs);
                        if let Ok(mut g) = st.lock() {
                            g.board.tracks = board.tracks.clone();
                            g.board.vias = board.vias.clone();
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

            // Rip-up and Negotiated Congestion Update
            if iteration + 1 < self.max_ripup_iterations {
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
                            if Self::netclass_priority(&connections[rip_idx].netclass.name) > 2 {
                                continue;
                            }
                            let p = routed_paths[rip_idx].as_ref().unwrap();
                            let passes_near = p.iter().any(|node| {
                                let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                                (wx - b_start.0).hypot(wy - b_start.1) < 2.5
                                    || (wx - b_end.0).hypot(wy - b_end.1) < 2.5
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
                            grid.update_distance_fields();
                        }

                        if !ripped {
                            let cycle_idx = (iteration * 13 + conn_idx) % total_conns;
                            for offset in 0..total_conns {
                                let rip_idx = (cycle_idx + offset) % total_conns;
                                if routed_paths[rip_idx].is_some()
                                    && Self::netclass_priority(&connections[rip_idx].netclass.name) <= 2
                                {
                                    if let Some(p) = routed_paths[rip_idx].take() {
                                        for node in &p {
                                            let idx = grid.idx(node.gx, node.gy, node.layer);
                                            grid.congestion[idx] += self.congestion_factor;
                                        }
                                        self.rebuild_grid_state(board, grid, &connections, &routed_paths, &escape_stubs);
                                        grid.update_distance_fields();
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (conn_idx, path) in routed_paths.iter().enumerate() {
            if path.is_none() {
                let c = &connections[conn_idx];
                println!(
                    "  [UNROUTED] Net '{}' (ID {}) from ({:.2}, {:.2}, L{}) to ({:.2}, {:.2}, L{})",
                    c.net_name, c.net_id, c.start.0, c.start.1, c.start.2, c.end.0, c.end.1, c.end.2
                );
            }
        }
        let completed_count = routed_paths.iter().filter(|p| p.is_some()).count();
        self.reconstruct_tracks_and_vias(board, grid, &connections, &routed_paths, &escape_stubs);

        if let Some(ref st) = live {
            if let Ok(mut g) = st.lock() {
                g.board.tracks = board.tracks.clone();
                g.board.vias = board.vias.clone();
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

    fn find_path(
        &self,
        grid: &RoutingGrid,
        start: GridNode,
        goal: GridNode,
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
    ) -> Option<Vec<GridNode>> {
        let mut open_set = BinaryHeap::new();
        let mut came_from: HashMap<GridNode, GridNode> = HashMap::new();
        let mut g_score: HashMap<GridNode, f64> = HashMap::new();

        let h_start = self.heuristic(grid, start, goal);
        g_score.insert(start, 0.0);
        open_set.push(AStarNode {
            node: start,
            cost: 0.0,
            priority: h_start,
        });

        let max_expansions = 1_000_000;
        let mut expansions = 0;

        while let Some(current) = open_set.pop() {
            expansions += 1;
            if expansions > max_expansions {
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
                return Some(path);
            }

            let Some(&current_g) = g_score.get(&u) else {
                continue;
            };

            for (neighbor, edge_cost) in
                grid.get_neighbors(u, net_id, req_clearance, track_width, via_dia, u1_center)
            {
                let tentative_g = current_g + edge_cost;
                let is_better = g_score.get(&neighbor).is_none_or(|&prev_g| tentative_g < prev_g);
                if is_better {
                    g_score.insert(neighbor, tentative_g);
                    came_from.insert(neighbor, u);
                    let priority = tentative_g + self.heuristic(grid, neighbor, goal);
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
        let layer_pen = if a.layer != b.layer { 1.5 } else { 0.0 };
        dx + dy + layer_pen
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
                    if (dx * dx + dy * dy) as f64 * (grid.res * grid.res) <= (track_width / 2.0).powi(2) {
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
                        if (cx * cx + cy * cy) as f64 * (grid.res * grid.res) <= (via_dia / 2.0).powi(2) {
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

    fn reconstruct_tracks_and_vias(
        &self,
        board: &mut Board,
        grid: &RoutingGrid,
        connections: &[RouteConnection],
        paths: &[Option<Vec<GridNode>>],
        escape_stubs: &[EscapeStub],
    ) {
        board.tracks.clear();
        board.vias.clear();

        for stub in escape_stubs {
            board.tracks.push(TrackSegment {
                start: stub.pad_pos,
                end: stub.tip_pos,
                width: stub.width,
                layer: 0,
                net_id: stub.net_id,
            });
        }

        for (conn_idx, path_opt) in paths.iter().enumerate() {
            let Some(path) = path_opt else {
                continue;
            };
            let conn = &connections[conn_idx];
            let width = conn.netclass.track_width;
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
                        board.vias.push(Via {
                            position: pt,
                            size: via_dia,
                            drill: via_drill,
                            net_id: conn.net_id,
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
                let tip = match pad.number.as_str() {
                    "1" => (px + 1.50, py),
                    "2" => (px + 0.70, py),
                    "3" => (px + 1.50, py),
                    "4" => (px + 0.70, py),
                    "6" => (px + 1.50, py),
                    "7" => (px, py - 0.45),
                    "8" => (px, py - 1.20),
                    "9" => (px, py - 0.45),
                    "10" => (px, py - 1.20),
                    "12" => (px, py - 0.45),
                    "13" => (px - 0.60, py),
                    "16" => (px - 1.40, py),
                    "17" => (px - 0.60, py),
                    "20" => (px, py + 0.60),
                    "21" => (px, py + 1.40),
                    "23" => (px, py + 0.60),
                    "24" => (px, py + 1.40),
                    _ => continue,
                };
                stubs.push(EscapeStub {
                    net_id: pad.net_id,
                    pad_pos: (px, py),
                    tip_pos: tip,
                    width: 0.20,
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
                let length = match pad.number.as_str() {
                    "A5" | "B5" | "A7" | "B7" => 1.20,
                    _ => 0.80,
                };
                stubs.push(EscapeStub {
                    net_id: pad.net_id,
                    pad_pos: (px, py),
                    tip_pos: (px + length, py),
                    width: 0.20,
                });
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
                        if (dx * dx + dy * dy) as f64 * (grid.res * grid.res) <= (stub.width / 2.0).powi(2) {
                            let nx = gx as isize + dx;
                            let ny = gy as isize + dy;
                            if nx >= 0 && nx < grid.width as isize && ny >= 0 && ny < grid.height as isize {
                                let idx = grid.idx(nx as usize, ny as usize, 0);
                                grid.obstacle_net_ids[idx] = stub.net_id as i32;
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
            let end_opt = match pad.number.as_str() {
                "15" | "18" => Some((px + 0.6375, py)),
                "19" | "22" => Some((px, py - 0.5375)),
                _ => None,
            };
            if let Some(end) = end_opt {
                board.tracks.push(TrackSegment {
                    start: (px, py),
                    end,
                    width: 0.20,
                    layer: 0,
                    net_id: gnd_net_id,
                });
            }
        }
    }

    fn add_ground_stitching_vias(&self, board: &mut Board, grid: &RoutingGrid) {
        let Some(gnd_net_id) = board.nets.iter().position(|n| n == "GND") else {
            return;
        };

        board.tracks.push(TrackSegment {
            start: (132.195, 108.25),
            end: (131.28, 107.18),
            width: 0.30,
            layer: 0,
            net_id: gnd_net_id,
        });
        board.tracks.push(TrackSegment {
            start: (132.195, 114.75),
            end: (131.28, 115.82),
            width: 0.30,
            layer: 0,
            net_id: gnd_net_id,
        });

        let mut added_via_positions: Vec<(f64, f64)> = Vec::new();

        for (ref_name, dy_off) in [("C14", 0.90), ("R14", -0.85)] {
            if let Some(fp) = board.footprints.iter().find(|f| f.reference == ref_name).cloned() {
                for pad in &fp.pads {
                    if pad.net_id == gnd_net_id {
                        let (px, py) = fp.pad_absolute_position(pad);
                        let (vx, vy) = (px, py + dy_off);
                        board.tracks.push(TrackSegment {
                            start: (px, py),
                            end: (vx, vy),
                            width: 0.25,
                            layer: 0,
                            net_id: gnd_net_id,
                        });
                        board.vias.push(Via {
                            position: (vx, vy),
                            size: 0.60,
                            drill: 0.30,
                            net_id: gnd_net_id,
                        });
                        added_via_positions.push((vx, vy));
                    }
                }
            }
        }

        for fp in board.footprints.clone() {
            if matches!(fp.reference.as_str(), "U1" | "C14" | "R14") {
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
                                layer: 0,
                                net_id: gnd_net_id,
                            });
                            board.vias.push(Via {
                                position: (vx, vy),
                                size: 0.60,
                                drill: 0.30,
                                net_id: gnd_net_id,
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
                            let idx0 = grid.idx(nx as usize, ny as usize, 0);
                            let idx1 = grid.idx(nx as usize, ny as usize, 1);
                            let n0 = grid.obstacle_net_ids[idx0];
                            let n1 = grid.obstacle_net_ids[idx1];
                            if (n0 != FREE_NET && n0 != gnd_net_id as i32)
                                || (n1 != FREE_NET && n1 != gnd_net_id as i32)
                                || (grid.smd_pad_cells[idx0] && n0 != gnd_net_id as i32)
                                || (grid.smd_pad_cells[idx1] && n1 != gnd_net_id as i32)
                            {
                                safe = false;
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
}
