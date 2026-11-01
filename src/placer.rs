// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use crate::board::{Board, LiveTrialState, Point, Polygon, TrialPhase, AABB};
use crate::vulkan::{GpuSaParams, VulkanEngine};
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub use crate::swap::optimize_pin_and_gate_swapping;

#[derive(Debug, Clone)]
pub struct PlacementConfig {
    pub duration_secs: u64,
    pub grid_step_mm: f64,
    pub min_spacing_mm: f64,
    pub crossing_weight: f64,
    pub wirelength_weight: f64,
    pub overlap_penalty: f64,
    pub boundary_penalty: f64,
    pub passive_affinity_weight: f64,
    pub passive_max_dist: f64,
    pub decoupling_affinity_weight: f64,
    pub decoupling_max_dist: f64,
    pub allow_layer_flip: bool,
    pub enable_escape_clearance: bool,
    pub escape_margin_mult: f64,
    pub max_jump_radius_mm: f64,
    pub pair_swap_prob: f64,
    pub centroid_pull_prob: f64,
    pub temp_scale: f64,
    pub vulkan: Option<Arc<VulkanEngine>>,
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            duration_secs: 20,
            grid_step_mm: 0.635,
            min_spacing_mm: 0.25,
            crossing_weight: 25.0,
            wirelength_weight: 1.0,
            overlap_penalty: 2500.0,
            boundary_penalty: 50000.0,
            passive_affinity_weight: 35.0,
            passive_max_dist: 3.5,
            decoupling_affinity_weight: 60.0,
            decoupling_max_dist: 2.0,
            allow_layer_flip: false,
            enable_escape_clearance: true,
            escape_margin_mult: 1.0,
            max_jump_radius_mm: 20.0,
            pair_swap_prob: 0.15,
            centroid_pull_prob: 0.20,
            temp_scale: 1.0,
            vulkan: None,
        }
    }
}

impl crate::board::Footprint {
    /// Computes the effective courtyard placement spacing for this footprint given config.
    pub fn effective_spacing(&self, config: &PlacementConfig) -> f64 {
        if config.enable_escape_clearance {
            config.min_spacing_mm + 2.0 * self.escape_margin * config.escape_margin_mult
        } else {
            config.min_spacing_mm
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlacementState {
    pub positions: Vec<Point>,
    pub rotations: Vec<f64>,
    pub layers: Vec<usize>,
    pub cached_courtyards: Vec<Vec<Polygon>>,
    pub cached_bounds: Vec<AABB>,
    pub cached_pads: Vec<Vec<Point>>,
    pub cost: f64,
    pub crossings: usize,
    pub wirelength: f64,
    pub overlaps: usize,
    pub boundary_violations: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassiveTie {
    pub passive_idx: usize,
    pub passive_pad_idx: usize,
    pub anchor_idx: usize,
    pub anchor_pad_idx: usize,
    pub max_dist: f64,
    pub weight_mult: f64,
    pub is_decoupling: bool,
}

pub fn is_ground_net(name: &str) -> bool {
    let s = name.trim();
    if s.is_empty() {
        return false;
    }
    let u = s.to_ascii_uppercase();
    u == "GND"
        || u == "/GND"
        || u == "GROUND"
        || u.starts_with("GND_")
        || u.ends_with("_GND")
        || u.contains("AGND")
        || u.contains("DGND")
        || u.contains("PGND")
}

pub fn is_power_net(name: &str) -> bool {
    let s = name.trim();
    if s.is_empty() || is_ground_net(s) {
        return false;
    }
    let u = s.to_ascii_uppercase();
    u.starts_with('+')
        || u.starts_with("VCC")
        || u.starts_with("VDD")
        || u.starts_with("VBUS")
        || u.starts_with("VRAW")
        || u.starts_with("VIN")
        || u.starts_with("VBAT")
        || u.starts_with("VOUT")
        || u.contains("3V3")
        || u.contains("5V")
        || u.contains("12V")
        || u.contains("BUCK_VIN")
        || u.contains("POWER")
}

pub fn build_passive_ties(board: &Board) -> Vec<PassiveTie> {
    // 1. Map of non-power, non-ground signal nets -> Vec<(fp_idx, pad_idx)>
    let mut net_pads: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (fp_idx, fp) in board.footprints.iter().enumerate() {
        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if pad.net_id == 0
                || pad.net_name.is_empty()
                || is_ground_net(&pad.net_name)
                || is_power_net(&pad.net_name)
            {
                continue;
            }
            net_pads.entry(pad.net_id).or_default().push((fp_idx, pad_idx));
        }
    }

    // 2. Map of power net_id -> Vec<(active_device_idx, pad_idx)>
    let mut power_net_device_pads: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (fp_idx, fp) in board.footprints.iter().enumerate() {
        let is_active = fp.reference.starts_with('U')
            || fp.reference.starts_with('Q')
            || fp.reference.starts_with('J')
            || fp.reference.starts_with('M');
        if !is_active {
            continue;
        }
        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if pad.net_id != 0 && is_power_net(&pad.net_name) {
                power_net_device_pads
                    .entry(pad.net_id)
                    .or_default()
                    .push((fp_idx, pad_idx));
            }
        }
    }

    let mut pin_usage: HashMap<(usize, usize), usize> = HashMap::new();
    let mut ties = Vec::new();

    for (fp_idx, fp) in board.footprints.iter().enumerate() {
        let is_passive = fp.reference.starts_with('R')
            || fp.reference.starts_with('C')
            || fp.reference.starts_with('L')
            || fp.reference.starts_with('F')
            || fp.reference.starts_with('D');
        if !is_passive || fp.is_locked {
            continue;
        }

        // Check if fp is a 2-pin decoupling/bulk capacitor: one pin on GND, one pin on a power net
        let is_2pin_cap = fp.reference.starts_with('C') && fp.pads.len() == 2;
        let is_bulk_cap = fp.width > 3.8
            || fp.height > 3.8
            || fp.value.to_lowercase().contains("low-esr")
            || fp.value.to_lowercase().contains("330uf")
            || fp.value.to_lowercase().contains("100uf")
            || fp.value.to_lowercase().contains("47uf");
        let mut gnd_pin = None;
        let mut pwr_pin = None;
        if is_2pin_cap {
            for (p_idx, pad) in fp.pads.iter().enumerate() {
                if is_ground_net(&pad.net_name) && gnd_pin.is_none() {
                    gnd_pin = Some(p_idx);
                } else if is_power_net(&pad.net_name) && pwr_pin.is_none() {
                    pwr_pin = Some(p_idx);
                }
            }
        }

        if let (Some(_gnd_idx), Some(pwr_pad_idx)) = (gnd_pin, pwr_pin) {
            let pwr_net_id = fp.pads[pwr_pad_idx].net_id;
            if let Some(candidate_pads) = power_net_device_pads.get(&pwr_net_id) {
                let mut best_candidate: Option<(usize, usize, f64)> = None;

                for &(dev_idx, dev_pad_idx) in candidate_pads {
                    if dev_idx == fp_idx {
                        continue;
                    }
                    let dev = &board.footprints[dev_idx];
                    let dev_ref = &dev.reference;
                    let type_tier = if dev_ref.starts_with('U') {
                        3.0
                    } else if dev_ref.starts_with('Q') {
                        2.0
                    } else {
                        1.0
                    };

                    let mut score = 0.0;
                    if let (Some(fp_sch), Some(dev_sch)) = (
                        board.schematic_positions.get(&fp.reference),
                        board.schematic_positions.get(dev_ref),
                    ) {
                        let d_sch = (fp_sch.0 - dev_sch.0).hypot(fp_sch.1 - dev_sch.1);
                        score += d_sch;
                    } else {
                        let d_phys = (fp.x - dev.x).hypot(fp.y - dev.y);
                        score += d_phys * 2.0;

                        let fp_num: f64 = fp
                            .reference
                            .trim_start_matches(|c: char| !c.is_ascii_digit())
                            .parse()
                            .unwrap_or(0.0);
                        let dev_num: f64 = dev_ref
                            .trim_start_matches(|c: char| !c.is_ascii_digit())
                            .parse()
                            .unwrap_or(0.0);
                        score += (fp_num - dev_num).abs() * 0.5;
                    }

                    for cl in &board.clusters {
                        if cl.member_indices.contains(&fp_idx) && cl.member_indices.contains(&dev_idx) {
                            score -= 200.0;
                            break;
                        }
                    }

                    score -= type_tier * 50.0;

                    let usage = *pin_usage.get(&(dev_idx, dev_pad_idx)).unwrap_or(&0);
                    score += (usage as f64) * 15.0;

                    if best_candidate.is_none_or(|b| score < b.2) {
                        best_candidate = Some((dev_idx, dev_pad_idx, score));
                    }
                }

                if let Some((best_dev, best_pad, _)) = best_candidate {
                    *pin_usage.entry((best_dev, best_pad)).or_insert(0) += 1;
                    let (max_dist, weight_mult, is_decoupling) = if is_bulk_cap {
                        (8.0, 1.0, false)
                    } else {
                        (2.0, 1.8, true)
                    };
                    ties.push(PassiveTie {
                        passive_idx: fp_idx,
                        passive_pad_idx: pwr_pad_idx,
                        anchor_idx: best_dev,
                        anchor_pad_idx: best_pad,
                        max_dist,
                        weight_mult,
                        is_decoupling,
                    });
                    continue;
                }
            }
        }

        // Non-decoupling passive (e.g. pullups, termination resistors, signal caps)
        let mut best: Option<(usize, usize, usize, usize)> = None;
        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if pad.net_name.is_empty()
                || is_ground_net(&pad.net_name)
                || is_power_net(&pad.net_name)
            {
                continue;
            }
            if let Some(peers) = net_pads.get(&pad.net_id) {
                for &(peer_fp_idx, peer_pad_idx) in peers {
                    if peer_fp_idx == fp_idx {
                        continue;
                    }
                    let peer_ref = &board.footprints[peer_fp_idx].reference;
                    let priority = if peer_ref.starts_with('U') {
                        3
                    } else if peer_ref.starts_with('J') || peer_ref.starts_with('Q') {
                        2
                    } else {
                        1
                    };
                    if best.is_none_or(|b| priority > b.3) {
                        best = Some((pad_idx, peer_fp_idx, peer_pad_idx, priority));
                    }
                }
            }
        }
        if let Some((p_pad, a_fp, a_pad, _)) = best {
            ties.push(PassiveTie {
                passive_idx: fp_idx,
                passive_pad_idx: p_pad,
                anchor_idx: a_fp,
                anchor_pad_idx: a_pad,
                max_dist: 3.5,
                weight_mult: 1.0,
                is_decoupling: false,
            });
        }
    }
    ties
}

impl PlacementState {
    pub fn from_board(board: &Board, config: &PlacementConfig) -> Self {
        let n = board.footprints.len();
        let mut positions = Vec::with_capacity(n);
        let mut rotations = Vec::with_capacity(n);
        let mut layers = Vec::with_capacity(n);
        let mut cached_courtyards = Vec::with_capacity(n);
        let mut cached_bounds = Vec::with_capacity(n);
        let mut cached_pads = Vec::with_capacity(n);

        for fp in &board.footprints {
            let pos = Point::new(fp.x, fp.y);
            let rot = fp.rotation;
            let layer = fp.layer;
            positions.push(pos);
            rotations.push(rot);
            layers.push(layer);
            let spacing = fp.effective_spacing(config);
            let polys = fp.transformed_courtyards_layered(pos, rot, layer, spacing);
            cached_bounds.push(union_aabb(&polys, pos));
            cached_courtyards.push(polys);
            let pads = (0..fp.pads.len())
                .map(|pi| fp.pad_world_point_layered(pi, pos, rot, layer))
                .collect();
            cached_pads.push(pads);
        }

        let mut st = Self {
            positions,
            rotations,
            layers,
            cached_courtyards,
            cached_bounds,
            cached_pads,
            cost: 0.0,
            crossings: 0,
            wirelength: 0.0,
            overlaps: 0,
            boundary_violations: 0,
        };
        let ties = build_passive_ties(board);
        st.evaluate(board, config, &ties);
        st
    }

    fn update_component(&mut self, idx: usize, board: &Board, config: &PlacementConfig) {
        let fp = &board.footprints[idx];
        let pos = self.positions[idx];
        let rot = self.rotations[idx];
        let layer = self.layers[idx];
        let spacing = fp.effective_spacing(config);
        let polys = fp.transformed_courtyards_layered(pos, rot, layer, spacing);
        self.cached_bounds[idx] = union_aabb(&polys, pos);
        self.cached_courtyards[idx] = polys;
        for pi in 0..fp.pads.len() {
            self.cached_pads[idx][pi] = fp.pad_world_point_layered(pi, pos, rot, layer);
        }
    }

    fn evaluate(&mut self, board: &Board, config: &PlacementConfig, ties: &[PassiveTie]) {
        let segments = board.generate_ratsnest_segments(&self.cached_pads);
        let wirelength: f64 = segments.iter().map(|s| s.length()).sum();
        let mut crossings = 0usize;
        for i in 0..segments.len() {
            for j in (i + 1)..segments.len() {
                if segments[i].crosses(&segments[j]) {
                    crossings += 1;
                }
            }
        }

        let n = board.footprints.len();
        let mut overlaps = 0usize;
        let mut overlap_area_cost = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if board.footprints[i].is_locked && board.footprints[j].is_locked {
                    continue;
                }
                // Two SMD-only footprints on opposite board sides (F.Cu vs B.Cu) do not collide
                if self.layers[i] != self.layers[j]
                    && board.footprints[i].is_smd_only()
                    && board.footprints[j].is_smd_only()
                {
                    continue;
                }
                if !self.cached_bounds[i].overlaps(&self.cached_bounds[j]) {
                    continue;
                }
                let mut hit = false;
                'outer: for pa in &self.cached_courtyards[i] {
                    for pb in &self.cached_courtyards[j] {
                        if pa.intersects_convex(pb) {
                            hit = true;
                            break 'outer;
                        }
                    }
                }
                if hit {
                    overlaps += 1;
                    let ox = (self.cached_bounds[i].max_x.min(self.cached_bounds[j].max_x)
                        - self.cached_bounds[i].min_x.max(self.cached_bounds[j].min_x))
                    .max(0.1);
                    let oy = (self.cached_bounds[i].max_y.min(self.cached_bounds[j].max_y)
                        - self.cached_bounds[i].min_y.max(self.cached_bounds[j].min_y))
                    .max(0.1);
                    overlap_area_cost += config.overlap_penalty * (1.0 + ox * oy);
                }
            }
        }

        let boundary = board.boundary_aabb();
        let mut boundary_cost = 0.0;
        let mut boundary_violations = 0usize;
        let mut flip_cost = 0.0;
        for i in 0..n {
            if board.footprints[i].is_locked {
                continue;
            }
            if self.layers[i] != board.footprints[i].orig_layer {
                flip_cost += 3.5;
            }
            let mut fp_violated = false;
            for poly in &self.cached_courtyards[i] {
                for v in &poly.vertices {
                    if !board.contains_point_in_boundary(v) {
                        let dx = (boundary.min_x - v.x).max(0.0) + (v.x - boundary.max_x).max(0.0);
                        let dy = (boundary.min_y - v.y).max(0.0) + (v.y - boundary.max_y).max(0.0);
                        boundary_cost += config.boundary_penalty * (10.0 + dx * 10.0 + dy * 10.0);
                        fp_violated = true;
                    }
                }
            }
            for p in &self.cached_pads[i] {
                if !board.contains_point_in_boundary(p) {
                    let dx = (boundary.min_x - p.x).max(0.0) + (p.x - boundary.max_x).max(0.0);
                    let dy = (boundary.min_y - p.y).max(0.0) + (p.y - boundary.max_y).max(0.0);
                    boundary_cost += config.boundary_penalty * (10.0 + dx * 10.0 + dy * 10.0);
                    fp_violated = true;
                }
            }
            if !fp_violated {
                let fp_layer = self.layers[i];
                for ko in &board.keepouts {
                    if !ko.layers.contains(&fp_layer) {
                        continue;
                    }
                    let ko_poly = Polygon::from_rect(ko.min_x, ko.min_y, ko.max_x - ko.min_x, ko.max_y - ko.min_y);
                    for poly in &self.cached_courtyards[i] {
                        if poly.intersects_convex(&ko_poly) {
                            boundary_cost += config.boundary_penalty * 20.0;
                            fp_violated = true;
                            break;
                        }
                    }
                    if !fp_violated {
                        for p in &self.cached_pads[i] {
                            if ko_poly.contains_point(p) {
                                boundary_cost += config.boundary_penalty * 20.0;
                                fp_violated = true;
                                break;
                            }
                        }
                    }
                    if fp_violated {
                        break;
                    }
                }
            }
            if fp_violated {
                boundary_violations += 1;
            }
        }

        let mut cluster_cost = 0.0;
        for cl in &board.clusters {
            let center = if let Some(a_idx) = cl.anchor_idx {
                self.positions[a_idx]
            } else {
                let k = cl.member_indices.len() as f64;
                Point::new(
                    cl.member_indices.iter().map(|&idx| self.positions[idx].x).sum::<f64>() / k,
                    cl.member_indices.iter().map(|&idx| self.positions[idx].y).sum::<f64>() / k,
                )
            };
            let w = (cl.weight * 0.25).min(25.0);
            for &m_idx in &cl.member_indices {
                if Some(m_idx) == cl.anchor_idx || board.footprints[m_idx].is_locked {
                    continue;
                }
                let d = self.positions[m_idx].distance_to(&center);
                if d > cl.max_radius {
                    let ex = d - cl.max_radius;
                    let penalty = if ex <= 2.0 {
                        0.5 * ex * ex
                    } else {
                        2.0 + 2.0 * (ex - 2.0)
                    };
                    cluster_cost += w * (0.1 * cl.max_radius + penalty);
                } else {
                    cluster_cost += (w * 0.1) * d;
                }
            }
        }

        let mut affinity_cost = 0.0;
        for tie in ties {
            let p_pt = self.cached_pads[tie.passive_idx][tie.passive_pad_idx];
            let a_pt = self.cached_pads[tie.anchor_idx][tie.anchor_pad_idx];
            let anchor_escape = if config.enable_escape_clearance {
                board.footprints[tie.anchor_idx].escape_margin * config.escape_margin_mult
            } else {
                0.0
            };
            let d = p_pt.distance_to(&a_pt);
            let (max_d, weight) = if tie.is_decoupling {
                (
                    config.decoupling_max_dist + anchor_escape,
                    config.decoupling_affinity_weight * tie.weight_mult.min(2.0),
                )
            } else {
                (
                    config.passive_max_dist + anchor_escape * 0.5,
                    config.passive_affinity_weight * tie.weight_mult.min(1.5),
                )
            };
            if d > max_d {
                let ex = d - max_d;
                let penalty = if ex <= 2.0 {
                    0.5 * ex * ex
                } else {
                    2.0 + 2.0 * (ex - 2.0)
                };
                affinity_cost += weight * (0.25 * max_d + penalty);
            } else {
                affinity_cost += weight * 0.25 * d;
            }
        }

        self.wirelength = wirelength;
        self.crossings = crossings;
        self.overlaps = overlaps;
        self.boundary_violations = boundary_violations;
        self.cost = wirelength * config.wirelength_weight
            + (crossings as f64) * config.crossing_weight
            + overlap_area_cost
            + boundary_cost
            + cluster_cost
            + affinity_cost
            + flip_cost;
    }

    pub fn apply_to_board(&self, board: &mut Board) {
        for (i, fp) in board.footprints.iter_mut().enumerate() {
            if !fp.is_locked {
                fp.x = self.positions[i].x;
                fp.y = self.positions[i].y;
                fp.rotation = self.rotations[i];
                fp.layer = self.layers[i];
            }
        }
    }
}

fn union_aabb(polys: &[Polygon], fallback: Point) -> AABB {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in polys {
        let b = p.aabb();
        min_x = min_x.min(b.min_x);
        min_y = min_y.min(b.min_y);
        max_x = max_x.max(b.max_x);
        max_y = max_y.max(b.max_y);
    }
    if min_x > max_x {
        AABB::new(fallback.x - 0.5, fallback.y - 0.5, fallback.x + 0.5, fallback.y + 0.5)
    } else {
        AABB::new(min_x, min_y, max_x, max_y)
    }
}

fn snap_to_grid(v: f64, step: f64) -> f64 {
    if step <= 0.0 {
        v
    } else {
        (v / step).round() * step
    }
}

fn build_gpu_sa_descriptors(
    board: &Board,
    config: &PlacementConfig,
    ties: &[PassiveTie],
    base: &PlacementState,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let mut fp_descs = Vec::with_capacity(board.footprints.len());
    for fp in &board.footprints {
        let spacing = fp.effective_spacing(config);
        let polys = fp.transformed_courtyards(Point::new(0.0, 0.0), 0.0, spacing);
        let b = union_aabb(&polys, Point::new(0.0, 0.0));
        let hw = ((b.max_x - b.min_x) * 0.5).max(0.4) as f32;
        let hh = ((b.max_y - b.min_y) * 0.5).max(0.4) as f32;
        let locked = if fp.is_locked { 1.0 } else { 0.0 };
        fp_descs.push([hw, hh, locked, 0.0]);
    }

    let mut net_pads: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (fp_idx, fp) in board.footprints.iter().enumerate() {
        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if pad.net_id == 0 || pad.net_name.is_empty() || pad.net_name == "GND" {
                continue;
            }
            net_pads.entry(pad.net_id).or_default().push((fp_idx, pad_idx));
        }
    }

    let mut edges: Vec<[f32; 4]> = Vec::new();
    let mut net_ids: Vec<usize> = net_pads.keys().copied().collect();
    net_ids.sort_unstable();
    for nid in net_ids {
        let pads = &net_pads[&nid];
        if pads.len() < 2 {
            continue;
        }
        let mut visited = vec![false; pads.len()];
        visited[0] = true;
        let mut count = 1usize;
        while count < pads.len() {
            let mut best_d = f64::INFINITY;
            let mut best_pair = (0usize, 0usize);
            for i in 0..pads.len() {
                if !visited[i] {
                    continue;
                }
                let pi = base.cached_pads[pads[i].0][pads[i].1];
                for j in 0..pads.len() {
                    if visited[j] {
                        continue;
                    }
                    let pj = base.cached_pads[pads[j].0][pads[j].1];
                    let d = pi.distance_to(&pj);
                    if d < best_d {
                        best_d = d;
                        best_pair = (i, j);
                    }
                }
            }
            visited[best_pair.1] = true;
            count += 1;
            let (fpa, pa) = pads[best_pair.0];
            let (fpb, pb) = pads[best_pair.1];
            if fpa != fpb {
                let pad_a = &board.footprints[fpa].pads[pa];
                let pad_b = &board.footprints[fpb].pads[pb];
                edges.push([fpa as f32, pad_a.offset_x as f32, pad_a.offset_y as f32, 0.0]);
                edges.push([fpb as f32, pad_b.offset_x as f32, pad_b.offset_y as f32, 0.0]);
            }
        }
    }

    for tie in ties {
        let pad_p = &board.footprints[tie.passive_idx].pads[tie.passive_pad_idx];
        let pad_a = &board.footprints[tie.anchor_idx].pads[tie.anchor_pad_idx];
        let tag = if tie.is_decoupling { 2.0 } else { 1.0 };
        edges.push([
            tie.passive_idx as f32,
            pad_p.offset_x as f32,
            pad_p.offset_y as f32,
            tag,
        ]);
        edges.push([
            tie.anchor_idx as f32,
            pad_a.offset_x as f32,
            pad_a.offset_y as f32,
            tag,
        ]);
    }

    (fp_descs, edges)
}

fn run_gpu_placement_batch(
    vk: &VulkanEngine,
    board: &Board,
    config: &PlacementConfig,
    ties: &[PassiveTie],
    movable: &[usize],
    base: &PlacementState,
    fp_descs: &[[f32; 4]],
    edges: &[[f32; 4]],
    seed_offset: u64,
    cooling: f64,
) -> Option<(PlacementState, usize)> {
    if movable.is_empty() || board.footprints.is_empty() {
        return None;
    }
    let nf = board.footprints.len();
    let ne = edges.len() / 2;
    let boundary = board.boundary_aabb();

    let (num_chains, steps_per_pass, num_passes): (u32, u32, u32) = if nf > 8 {
        (4096, 18, 4)
    } else {
        (128, 6, 2)
    };

    let mut init_poses = Vec::with_capacity((num_chains as usize) * nf);
    for _ in 0..num_chains {
        for i in 0..nf {
            init_poses.push([
                base.positions[i].x as f32,
                base.positions[i].y as f32,
                base.rotations[i] as f32,
                0.0,
            ]);
        }
    }

    let seed = (0xA5A5_5A5Au32).wrapping_add((seed_offset as u32).wrapping_mul(0x9E37_79B9));
    let sa_params = GpuSaParams {
        num_chains,
        num_footprints: nf as u32,
        num_edges: ne as u32,
        steps_per_pass,
        seed,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        board_min_x: boundary.min_x as f32,
        board_min_y: boundary.min_y as f32,
        board_max_x: boundary.max_x as f32,
        board_max_y: boundary.max_y as f32,
        grid_step: config.grid_step_mm as f32,
        wirelength_weight: config.wirelength_weight as f32,
        crossing_weight: config.crossing_weight as f32,
        overlap_penalty: config.overlap_penalty as f32,
        boundary_penalty: config.boundary_penalty as f32,
        passive_affinity_weight: config.passive_affinity_weight as f32,
        passive_max_dist: config.passive_max_dist as f32,
        base_temp: (8.0 * cooling).max(0.5) as f32,
    };

    let (best_poses, best_scores) =
        vk.run_gpu_sa_placer_chains(fp_descs, edges, &init_poses, &sa_params, num_passes)?;

    let mut ranked: Vec<(usize, f32)> = best_scores
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s[1]))
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let total_gpu_evals = (num_chains as usize) * (steps_per_pass as usize) * (num_passes as usize);
    let mut best_verified: Option<PlacementState> = None;

    for &(chain_idx, _) in ranked.iter().take(4) {
        let mut cand = base.clone();
        let c_base = chain_idx * nf;
        for i in 0..nf {
            if !board.footprints[i].is_locked {
                let p = best_poses[c_base + i];
                cand.positions[i] = Point::new(p[0] as f64, p[1] as f64);
                cand.rotations[i] = p[2] as f64;
                cand.update_component(i, board, config);
            }
        }
        cand.evaluate(board, config, ties);
        if cand.overlaps > 0 && cand.overlaps <= 2 && cand.boundary_violations == 0 && cand.wirelength < base.wirelength {
            let mut resolved = cand.clone();
            for _ in 0..15 {
                if resolved.overlaps == 0 { break; }
                let mut ov_pair = None;
                'find_ov: for i in 0..nf {
                    for j in (i + 1)..nf {
                        if !resolved.cached_bounds[i].overlaps(&resolved.cached_bounds[j]) { continue; }
                        for pa in &resolved.cached_courtyards[i] {
                            for pb in &resolved.cached_courtyards[j] {
                                if pa.intersects_convex(pb) {
                                    ov_pair = Some((i, j));
                                    break 'find_ov;
                                }
                            }
                        }
                    }
                }
                if let Some((fa, fb)) = ov_pair {
                    let target = if !board.footprints[fa].is_locked { fa } else { fb };
                    if !board.footprints[target].is_locked {
                        let other = if target == fa { fb } else { fa };
                        let dx = resolved.positions[target].x - resolved.positions[other].x;
                        let dy = resolved.positions[target].y - resolved.positions[other].y;
                        let dist = dx.hypot(dy).max(1e-4);
                        let (cx_min, cx_max, cy_min, cy_max) = board.footprints[target]
                            .origin_clamp_range(resolved.rotations[target], resolved.layers[target], config.min_spacing_mm, &boundary);
                        let push_step = config.grid_step_mm;
                        let nx = snap_to_grid((resolved.positions[target].x + (dx / dist) * push_step).clamp(cx_min, cx_max), config.grid_step_mm);
                        let ny = snap_to_grid((resolved.positions[target].y + (dy / dist) * push_step).clamp(cy_min, cy_max), config.grid_step_mm);
                        resolved.positions[target] = Point::new(nx, ny);
                        resolved.update_component(target, board, config);
                        resolved.evaluate(board, config, ties);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            if resolved.overlaps < cand.overlaps {
                cand = resolved;
            }
        }
        if best_verified.as_ref().is_none_or(|b| {
            cand.boundary_violations < b.boundary_violations
                || (cand.boundary_violations == b.boundary_violations && cand.overlaps < b.overlaps)
                || (cand.boundary_violations == b.boundary_violations
                    && cand.overlaps == b.overlaps
                    && cand.cost < b.cost)
        }) {
            best_verified = Some(cand);
        }
    }

    best_verified.map(|c| (c, total_gpu_evals))
}

pub fn legalize_placement(
    board: &Board,
    config: &PlacementConfig,
    initial: PlacementState,
    live: Option<&Arc<Mutex<LiveTrialState>>>,
) -> PlacementState {
    if initial.overlaps == 0 && initial.boundary_violations == 0 {
        return initial;
    }
    let num_threads = rayon::current_num_threads().max(2);
    if let Some(st) = live {
        if let Ok(mut g) = st.lock() {
            g.phase = TrialPhase::Legalizing;
            g.push_log(format!(
                "[Legalizer] Resolving {} initial courtyard overlaps across {} CPU cores...",
                initial.overlaps, num_threads
            ));
        }
    }

    let movable: Vec<usize> = board
        .footprints
        .iter()
        .enumerate()
        .filter(|(_, fp)| !fp.is_locked)
        .map(|(i, _)| i)
        .collect();
    if movable.is_empty() {
        return initial;
    }

    let boundary = board.boundary_aabb();
    let ties = build_passive_ties(board);
    let solved = AtomicBool::new(false);
    let steps_per_thread = (10000 / num_threads).max(2500);

    let results: Vec<PlacementState> = (0..num_threads)
        .into_par_iter()
        .map(|tid| {
            let seed = if tid == 0 {
                0x517CC1B727220A95u64
            } else {
                0x517CC1B727220A95u64 ^ ((tid as u64).wrapping_mul(0x9E3779B97F4A7C15))
            };
            let mut rng = StdRng::seed_from_u64(seed);
            let mut cur = initial.clone();
            let mut local_best = cur.clone();

            for step in 0..steps_per_thread {
                if cur.overlaps == 0 && cur.boundary_violations == 0 {
                    solved.store(true, Ordering::Relaxed);
                    break;
                }
                if step % 16 == 0 && solved.load(Ordering::Relaxed) {
                    break;
                }

                let overlapping_pairs: Vec<(usize, usize)> = if cur.overlaps > 0 && rng.gen_bool(0.90) {
                    let mut ov_pairs = Vec::new();
                    for i in 0..cur.cached_bounds.len() {
                        for j in (i + 1)..cur.cached_bounds.len() {
                            if board.footprints[i].is_locked && board.footprints[j].is_locked {
                                continue;
                            }
                            if board.footprints[i].is_smd_only() && board.footprints[j].is_smd_only() && cur.layers[i] != cur.layers[j] {
                                continue;
                            }
                            if cur.cached_bounds[i].overlaps(&cur.cached_bounds[j]) {
                                let mut hit = false;
                                for pa in &cur.cached_courtyards[i] {
                                    for pb in &cur.cached_courtyards[j] {
                                        if pa.intersects_convex(pb) {
                                            hit = true;
                                            break;
                                        }
                                    }
                                    if hit { break; }
                                }
                                if hit {
                                    if !board.footprints[i].is_locked { ov_pairs.push((i, j)); }
                                    if !board.footprints[j].is_locked { ov_pairs.push((j, i)); }
                                }
                            }
                        }
                    }
                    ov_pairs
                } else {
                    Vec::new()
                };

                let (idx, other_opt) = if !overlapping_pairs.is_empty() {
                    let p = overlapping_pairs[rng.gen_range(0..overlapping_pairs.len())];
                    (p.0, Some(p.1))
                } else {
                    (movable[rng.gen_range(0..movable.len())], None)
                };

                let old_pos = cur.positions[idx];
                let old_rot = cur.rotations[idx];
                let old_layer = cur.layers[idx];
                let old_cost = cur.cost;
                let old_ov = cur.overlaps;
                let old_bv = cur.boundary_violations;

                let roll = rng.gen_range(0..100);
                if roll < 70 {
                    let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx].origin_clamp_range(
                        cur.rotations[idx],
                        cur.layers[idx],
                        config.min_spacing_mm,
                        &boundary,
                    );
                    if let Some(other) = other_opt {
                        if rng.gen_bool(0.65) {
                            let dx = cur.positions[idx].x - cur.positions[other].x;
                            let dy = cur.positions[idx].y - cur.positions[other].y;
                            let dist = dx.hypot(dy).max(1e-4);
                            let (ux, uy) = (dx / dist, dy / dist);
                            let push_dist = rng.gen_range(1.0..=12.0);
                            let jitter = rng.gen_range(-1.5..=1.5);
                            let nx = snap_to_grid(
                                (old_pos.x + ux * push_dist + jitter).clamp(cx_min, cx_max),
                                config.grid_step_mm,
                            );
                            let ny = snap_to_grid(
                                (old_pos.y + uy * push_dist + jitter).clamp(cy_min, cy_max),
                                config.grid_step_mm,
                            );
                            cur.positions[idx] = Point::new(nx, ny);
                        } else {
                            let r = if step < steps_per_thread / 3 { 8.0 } else { 3.5 };
                            let nx = snap_to_grid(
                                (old_pos.x + rng.gen_range(-r..=r)).clamp(cx_min, cx_max),
                                config.grid_step_mm,
                            );
                            let ny = snap_to_grid(
                                (old_pos.y + rng.gen_range(-r..=r)).clamp(cy_min, cy_max),
                                config.grid_step_mm,
                            );
                            cur.positions[idx] = Point::new(nx, ny);
                        }
                    } else {
                        let r = if step < steps_per_thread / 3 { 6.0 } else { 2.5 };
                        let nx = snap_to_grid(
                            (old_pos.x + rng.gen_range(-r..=r)).clamp(cx_min, cx_max),
                            config.grid_step_mm,
                        );
                        let ny = snap_to_grid(
                            (old_pos.y + rng.gen_range(-r..=r)).clamp(cy_min, cy_max),
                            config.grid_step_mm,
                        );
                        cur.positions[idx] = Point::new(nx, ny);
                    }
                } else if roll < 90 || !config.allow_layer_flip || !board.footprints[idx].is_smd_only() {
                    let angles = [0.0, 90.0, 180.0, 270.0];
                    let new_rot = angles[rng.gen_range(0..4)];
                    cur.rotations[idx] = new_rot;
                    let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx].origin_clamp_range(
                        new_rot,
                        cur.layers[idx],
                        config.min_spacing_mm,
                        &boundary,
                    );
                    cur.positions[idx].x = snap_to_grid(cur.positions[idx].x.clamp(cx_min, cx_max), config.grid_step_mm);
                    cur.positions[idx].y = snap_to_grid(cur.positions[idx].y.clamp(cy_min, cy_max), config.grid_step_mm);
                } else {
                    cur.layers[idx] = 1 - cur.layers[idx];
                    let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx].origin_clamp_range(
                        cur.rotations[idx],
                        cur.layers[idx],
                        config.min_spacing_mm,
                        &boundary,
                    );
                    cur.positions[idx].x = snap_to_grid(cur.positions[idx].x.clamp(cx_min, cx_max), config.grid_step_mm);
                    cur.positions[idx].y = snap_to_grid(cur.positions[idx].y.clamp(cy_min, cy_max), config.grid_step_mm);
                }
                cur.update_component(idx, board, config);
                cur.evaluate(board, config, &ties);

                let accept = (cur.boundary_violations < old_bv)
                    || (cur.boundary_violations == old_bv && (
                        cur.overlaps < old_ov
                            || (cur.overlaps == old_ov && cur.boundary_violations == 0 && (cur.cost <= old_cost || rng.gen_bool(0.15)))
                    ));

                if accept {
                    let is_better = if cur.boundary_violations != local_best.boundary_violations {
                        cur.boundary_violations < local_best.boundary_violations
                    } else if cur.overlaps != local_best.overlaps {
                        cur.overlaps < local_best.overlaps
                    } else {
                        cur.cost < local_best.cost
                    };
                    if is_better {
                        local_best = cur.clone();
                    }
                    if local_best.overlaps == 0 && local_best.boundary_violations == 0 {
                        solved.store(true, Ordering::Relaxed);
                        break;
                    }
                } else {
                    cur.positions[idx] = old_pos;
                    cur.rotations[idx] = old_rot;
                    cur.layers[idx] = old_layer;
                    cur.update_component(idx, board, config);
                    cur.cost = old_cost;
                    cur.overlaps = old_ov;
                    cur.boundary_violations = old_bv;
                }
            }
            local_best
        })
        .collect();

    results
        .into_iter()
        .min_by(|a, b| {
            a.boundary_violations
                .cmp(&b.boundary_violations)
                .then_with(|| a.overlaps.cmp(&b.overlaps))
                .then_with(|| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal))
        })
        .unwrap_or(initial)
}

/// Positions components in priority order before Simulated Annealing starts:
/// 1. High priority differential pair components (aligned and placed with direct corridors).
/// 2. Decoupling capacitors situated as close as possible to their connected non-ground device power pin.
/// 3. Remaining components placed around their cluster anchors or schematic positions.
pub fn prioritize_and_position_initial_components(
    board: &mut Board,
    config: &PlacementConfig,
    live: Option<&Arc<Mutex<LiveTrialState>>>,
) {
    if board.footprints.is_empty() {
        return;
    }

    let boundary = board.boundary_aabb();
    let ties = build_passive_ties(board);

    // Track which movable footprints have been situated
    let mut placed = vec![false; board.footprints.len()];
    for (i, fp) in board.footprints.iter().enumerate() {
        if fp.is_locked {
            placed[i] = true;
        }
    }

    // 1. Priority 1: High priority differential pair components
    let diff_nets: HashSet<usize> = board
        .nets
        .iter()
        .enumerate()
        .filter(|(_, name)| {
            board.net_to_class.get(name.as_str()).map_or(false, |c| c == "USB_Diff")
                || name.contains("USB_D")
                || name.contains("DP")
                || name.contains("DM")
                || name.contains("D+")
                || name.contains("D-")
        })
        .map(|(id, _)| id)
        .collect();

    let mut diff_pair_fps: Vec<usize> = Vec::new();
    for (idx, fp) in board.footprints.iter().enumerate() {
        if !placed[idx] && fp.pads.iter().any(|p| diff_nets.contains(&p.net_id)) {
            diff_pair_fps.push(idx);
        }
    }

    let locked_diff_anchor = board
        .footprints
        .iter()
        .enumerate()
        .find(|(i, fp)| placed[*i] && fp.pads.iter().any(|p| diff_nets.contains(&p.net_id)))
        .map(|(i, _)| i);

    for &diff_idx in &diff_pair_fps {
        if placed[diff_idx] {
            continue;
        }
        if let Some(anchor_idx) = locked_diff_anchor {
            let anchor_pos = Point::new(board.footprints[anchor_idx].x, board.footprints[anchor_idx].y);
            let (target_x, target_y) = if anchor_pos.x > (boundary.min_x + boundary.max_x) * 0.5 {
                (anchor_pos.x - 12.0, anchor_pos.y)
            } else {
                (anchor_pos.x + 12.0, anchor_pos.y)
            };
            let (cx_min, cx_max, cy_min, cy_max) = board.footprints[diff_idx].origin_clamp_range(
                board.footprints[diff_idx].rotation,
                board.footprints[diff_idx].layer,
                config.min_spacing_mm,
                &boundary,
            );
            board.footprints[diff_idx].x = snap_to_grid(
                target_x.clamp(cx_min, cx_max),
                config.grid_step_mm,
            );
            board.footprints[diff_idx].y = snap_to_grid(
                target_y.clamp(cy_min, cy_max),
                config.grid_step_mm,
            );
        }
        placed[diff_idx] = true;
    }

    // 2. Priority 2: Decoupling capacitors — situating them as close to the connected non-ground pin as possible!
    for tie in &ties {
        if !tie.is_decoupling {
            continue;
        }
        let cap_idx = tie.passive_idx;
        if placed[cap_idx] {
            continue;
        }
        let anchor_idx = tie.anchor_idx;
        let anchor_pad_idx = tie.anchor_pad_idx;
        let cap_pad_idx = tie.passive_pad_idx;

        let dev = &board.footprints[anchor_idx];
        let dev_pos = Point::new(dev.x, dev.y);
        let dev_pin = &dev.pads[anchor_pad_idx];
        let (pin_x, pin_y) = dev.pad_absolute_position(dev_pin);
        let pin_pt = Point::new(pin_x, pin_y);

        let mut dx = pin_x - dev_pos.x;
        let mut dy = pin_y - dev_pos.y;
        let len = dx.hypot(dy);
        if len > 1e-4 {
            dx /= len;
            dy /= len;
        } else {
            dx = 1.0;
            dy = 0.0;
        }
        let base_angle = dy.atan2(dx);

        let mut placed_courtyards: Vec<Vec<Polygon>> = Vec::new();
        for (i, fp) in board.footprints.iter().enumerate() {
            if placed[i] && i != cap_idx {
                placed_courtyards.push(fp.transformed_courtyards_layered(
                    Point::new(fp.x, fp.y),
                    fp.rotation,
                    fp.layer,
                    fp.effective_spacing(config),
                ));
            }
        }

        let mut best_cand: Option<(f64, f64, f64, f64)> = None;

        let dist_steps = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0];
        let angle_deltas = [
            0.0, 0.20, -0.20, 0.40, -0.40, 0.60, -0.60, 0.80, -0.80, 1.0, -1.0, 1.2, -1.2, 1.4, -1.4,
        ];
        let rotations = [0.0, 90.0, 180.0, 270.0];

        let cap = &board.footprints[cap_idx];
        let cap_spacing = cap.effective_spacing(config);
        let anchor_escape = if config.enable_escape_clearance {
            dev.escape_margin * config.escape_margin_mult
        } else {
            0.0
        };

        'search: for &d in &dist_steps {
            for &da in &angle_deltas {
                let angle = base_angle + da;
                for &rot in &rotations {
                    let (cx_min, cx_max, cy_min, cy_max) = cap.origin_clamp_range(rot, cap.layer, config.min_spacing_mm, &boundary);
                    let cx = snap_to_grid(
                        (pin_x + d * angle.cos()).clamp(cx_min, cx_max),
                        config.grid_step_mm,
                    );
                    let cy = snap_to_grid(
                        (pin_y + d * angle.sin()).clamp(cy_min, cy_max),
                        config.grid_step_mm,
                    );
                    let c_pos = Point::new(cx, cy);

                    let polys = cap.transformed_courtyards_layered(c_pos, rot, cap.layer, cap_spacing);
                    let mut collides = false;
                    'check: for group in &placed_courtyards {
                        for p_placed in group {
                            for p_cand in &polys {
                                if p_cand.intersects_convex(p_placed) {
                                    collides = true;
                                    break 'check;
                                }
                            }
                        }
                    }
                    if !collides && is_placement_legal(board, cap, c_pos, rot, cap.layer, config.min_spacing_mm) {
                        let cap_pad_world = cap.pad_world_point_layered(cap_pad_idx, c_pos, rot, cap.layer);
                        let pad_dist = cap_pad_world.distance_to(&pin_pt);
                        if best_cand.is_none_or(|b| pad_dist < b.0) {
                            best_cand = Some((pad_dist, cx, cy, rot));
                            if pad_dist <= 1.0 + anchor_escape {
                                break 'search;
                            }
                        }
                    }
                }
            }
        }

        if let Some((_, best_x, best_y, best_rot)) = best_cand {
            board.footprints[cap_idx].x = best_x;
            board.footprints[cap_idx].y = best_y;
            board.footprints[cap_idx].rotation = best_rot;
            placed[cap_idx] = true;
        }
    }

    // Decoupling caps have been placed directly at their power pins.
    // Remaining components retain their valid positions.
    if let Some(st) = live {
        if let Ok(mut g) = st.lock() {
            g.push_log("[Auto-Placer] Prioritized differential pairs and situated decoupling caps near device power pins".to_string());
        }
    }
}

/// Checks if placing `fp` at `pos` with `rot` and `layer` is legal (within board boundary, no keepout intersection, no courtyard collision).
pub fn is_placement_legal(
    board: &Board,
    fp: &crate::board::Footprint,
    pos: Point,
    rot: f64,
    layer: usize,
    min_spacing_mm: f64,
) -> bool {
    let cand_spacing = fp.placement_spacing(min_spacing_mm);
    let cand_polys = fp.transformed_courtyards_layered(pos, rot, layer, cand_spacing);

    // 1. Must be entirely inside the board outline (origin, courtyards, AND copper pads/legs)
    if !board.contains_point_in_boundary(&pos) {
        return false;
    }
    let raw_polys = fp.transformed_courtyards_layered(pos, rot, layer, min_spacing_mm);
    for poly in &raw_polys {
        for v in &poly.vertices {
            if !board.contains_point_in_boundary(v) {
                return false;
            }
        }
    }

    for pad in &fp.pads {
        for corner_pt in fp.pad_corners_at(pad, pos, rot, layer) {
            if !board.contains_point_in_boundary(&corner_pt) {
                return false;
            }
        }
    }

    // 2. Must not overlap any keepout zones
    for ko in &board.keepouts {
        let ko_aabb = AABB::new(ko.min_x, ko.min_y, ko.max_x, ko.max_y);
        for poly in &raw_polys {
            if poly.aabb().overlaps(&ko_aabb) {
                return false;
            }
        }
    }

    // 3. Must not collide with any already placed footprints (honoring escape clearances)
    for placed_fp in &board.footprints {
        if placed_fp.reference == fp.reference {
            continue;
        }
        // Two SMD-only footprints on opposite layers do not collide
        if layer != placed_fp.layer && fp.is_smd_only() && placed_fp.is_smd_only() {
            continue;
        }
        let placed_spacing = placed_fp.placement_spacing(min_spacing_mm);
        let placed_polys = placed_fp.transformed_courtyards_layered(
            Point::new(placed_fp.x, placed_fp.y),
            placed_fp.rotation,
            placed_fp.layer,
            placed_spacing,
        );
        for pa in &cand_polys {
            for pb in &placed_polys {
                if pa.intersects_convex(pb) {
                    return false;
                }
            }
        }
    }

    true
}

/// Shoves unlocked components that overlap with `placed_idx` away from it,
/// preserving legal courtyard spacing and boundary/keepout clearance.
/// If direct directional shove leaves residual overlaps, runs local simulated annealing legalization.
/// Returns the number of shoved components.
pub fn shove_overlapping_components(
    board: &mut Board,
    placed_idx: usize,
    min_spacing_mm: f64,
) -> usize {
    if placed_idx >= board.footprints.len() {
        return 0;
    }

    let mut shoved_count = 0;
    let placed_fp = board.footprints[placed_idx].clone();
    let placed_pos = Point::new(placed_fp.x, placed_fp.y);
    let placed_polys = placed_fp.transformed_courtyards_layered(
        placed_pos,
        placed_fp.rotation,
        placed_fp.layer,
        min_spacing_mm,
    );

    // 1. Identify all colliding unlocked footprints
    let mut colliding_unlocked_indices = Vec::new();
    for (i, fp) in board.footprints.iter().enumerate() {
        if i == placed_idx || fp.is_locked {
            continue;
        }
        if placed_fp.layer != fp.layer && placed_fp.is_smd_only() && fp.is_smd_only() {
            continue;
        }
        let fp_polys = fp.transformed_courtyards_layered(
            Point::new(fp.x, fp.y),
            fp.rotation,
            fp.layer,
            min_spacing_mm,
        );
        let mut hit = false;
        'check: for pa in &placed_polys {
            for pb in &fp_polys {
                if pa.intersects_convex(pb) {
                    hit = true;
                    break 'check;
                }
            }
        }
        if hit {
            colliding_unlocked_indices.push(i);
        }
    }

    if colliding_unlocked_indices.is_empty() {
        return 0;
    }

    let orig_positions: Vec<(f64, f64, f64, usize)> = board
        .footprints
        .iter()
        .map(|fp| (fp.x, fp.y, fp.rotation, fp.layer))
        .collect();

    // 2. Directional shove for each colliding unlocked component
    for &idx in &colliding_unlocked_indices {
        let other_fp = board.footprints[idx].clone();
        let dx = other_fp.x - placed_fp.x;
        let dy = other_fp.y - placed_fp.y;
        let dist = dx.hypot(dy);
        let (ux, uy) = if dist < 1e-4 {
            (1.0, 0.0)
        } else {
            (dx / dist, dy / dist)
        };

        let angles: [f64; 7] = [0.0, 0.35, -0.35, 0.785, -0.785, 1.5708, -1.5708];
        let distances: [f64; 12] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0];
        let mut best_pos = None;

        'search: for &d in &distances {
            for &a in &angles {
                let dir_x = ux * a.cos() - uy * a.sin();
                let dir_y = ux * a.sin() + uy * a.cos();
                let nx = other_fp.x + dir_x * d;
                let ny = other_fp.y + dir_y * d;

                let cand_pt = Point::new(nx, ny);
                if is_placement_legal(board, &other_fp, cand_pt, other_fp.rotation, other_fp.layer, min_spacing_mm) {
                    best_pos = Some((nx, ny));
                    break 'search;
                }
            }
        }

        if let Some((nx, ny)) = best_pos {
            board.footprints[idx].x = nx;
            board.footprints[idx].y = ny;
            shoved_count += 1;
        }
    }

    // 3. If any overlaps remain on the board, run quick simulated annealing legalization
    // while keeping the placed component locked so its position is preserved.
    let cfg = PlacementConfig {
        min_spacing_mm,
        duration_secs: 0,
        ..PlacementConfig::default()
    };
    let initial_state = PlacementState::from_board(board, &cfg);
    if initial_state.overlaps > 0 {
        let was_locked = board.footprints[placed_idx].is_locked;
        board.footprints[placed_idx].is_locked = true;
        let legalized = legalize_placement(board, &cfg, initial_state, None);
        legalized.apply_to_board(board);
        board.footprints[placed_idx].is_locked = was_locked;
        shoved_count = shoved_count.max(colliding_unlocked_indices.len());
    }

    // Rip up connected traces for any footprints that were shoved or legalized
    for (i, &orig_pos) in orig_positions.iter().enumerate() {
        if i == placed_idx || i >= board.footprints.len() {
            continue;
        }
        let fp = &board.footprints[i];
        let moved = (fp.x - orig_pos.0).abs() > 1e-4
            || (fp.y - orig_pos.1).abs() > 1e-4
            || (fp.rotation - orig_pos.2).abs() > 1e-4
            || fp.layer != orig_pos.3;
        if moved {
            board.rip_up_connected_traces_to_component_at(i, Some(orig_pos));
        }
    }

    shoved_count
}


/// Finds the most favourable location (x, y), orientation (rotation in degrees), and layer for `fp` on `board`.
/// - Decoupling capacitors are placed directly adjacent to their connected IC's non-ground power pin, oriented with their power pad facing the IC pin.
/// - Differential pairs and connected passives/ICs are placed adjacent to their connected pins, minimizing Euclidean wirelength and crossings.
/// - Unconnected components are placed in the nearest legal, non-colliding location.
pub fn find_most_favourable_placement(
    board: &Board,
    fp: &crate::board::Footprint,
    config: &PlacementConfig,
) -> (f64, f64, f64, usize) {
    let boundary = board.boundary_aabb();
    let min_spacing = config.min_spacing_mm;
    let grid_step = config.grid_step_mm.max(0.1);
    let rotations = [0.0, 90.0, 180.0, 270.0];

    // Check if fp is a decoupling capacitor
    let is_cap = fp.reference.starts_with('C');
    let mut gnd_pad_idx = None;
    let mut pwr_pad_idx = None;
    for (pidx, p) in fp.pads.iter().enumerate() {
        if is_ground_net(&p.net_name) {
            gnd_pad_idx = Some(pidx);
        } else if is_power_net(&p.net_name) {
            pwr_pad_idx = Some(pidx);
        }
    }
    let is_decoupling = is_cap && gnd_pad_idx.is_some() && pwr_pad_idx.is_some();

    if is_decoupling {
        let pwr_idx = pwr_pad_idx.unwrap();
        let pwr_net = &fp.pads[pwr_idx].net_name;

        // Find candidate active devices on the board sharing this power net, ranked by schematic and cluster affinity
        let mut candidate_pins: Vec<(&crate::board::Footprint, usize, f64)> = Vec::new();
        let fp_sch_pos = board.schematic_positions.get(&fp.reference);
        let fp_cluster_idx = board.clusters.iter().position(|c| c.member_refs.contains(&fp.reference));

        for placed_dev in &board.footprints {
            let is_active = placed_dev.reference.starts_with('U')
                || placed_dev.reference.starts_with('Q')
                || placed_dev.reference.starts_with('J')
                || placed_dev.reference.starts_with('M');
            if !is_active {
                continue;
            }
            for (pad_idx, pad) in placed_dev.pads.iter().enumerate() {
                if pad.net_name == *pwr_net {
                    let pin_pos = placed_dev.pad_absolute_position(pad);
                    // Count how many caps are already within 3mm of this pin
                    let existing_caps = board
                        .footprints
                        .iter()
                        .filter(|f| f.reference.starts_with('C'))
                        .filter(|f| {
                            let (cx, cy) = (f.x, f.y);
                            (cx - pin_pos.0).hypot(cy - pin_pos.1) < 3.0
                        })
                        .count();

                    let mut score = 1000.0;
                    if let Some(pos) = fp_sch_pos {
                        if let Some(dev_pos) = board.schematic_positions.get(&placed_dev.reference) {
                            let sch_dist = (pos.0 - dev_pos.0).hypot(pos.1 - dev_pos.1);
                            score += sch_dist * 10.0;
                        }
                    }
                    if let Some(c_idx) = fp_cluster_idx {
                        if board.clusters[c_idx].member_refs.contains(&placed_dev.reference) {
                            score -= 800.0;
                        }
                    }
                    score += (existing_caps as f64) * 40.0;

                    candidate_pins.push((placed_dev, pad_idx, score));
                }
            }
        }

        candidate_pins.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        for (dev, pad_idx, _) in candidate_pins {
            let pin = &dev.pads[pad_idx];
            let (pin_x, pin_y) = dev.pad_absolute_position(pin);
            let pin_pt = Point::new(pin_x, pin_y);
            let dev_pos = Point::new(dev.x, dev.y);

            let dx = pin_x - dev_pos.x;
            let dy = pin_y - dev_pos.y;
            let len = dx.hypot(dy);
            let base_angle = if len > 1e-4 { dy.atan2(dx) } else { 0.0 };

            let dist_steps = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0];
            let angle_deltas = [
                0.0, 0.20, -0.20, 0.40, -0.40, 0.60, -0.60, 0.80, -0.80, 1.0, -1.0, 1.2, -1.2, 1.5, -1.5,
                2.0, -2.0, 2.5, -2.5, 3.0, -3.0, std::f64::consts::PI,
            ];

            let mut best_cand: Option<(f64, f64, f64, f64, usize)> = None;

            for &d in &dist_steps {
                for &da in &angle_deltas {
                    let angle = base_angle + da;
                    for &rot in &rotations {
                    let (cx_min, cx_max, cy_min, cy_max) = fp.origin_clamp_range(rot, fp.layer, min_spacing, &boundary);
                    let cx = snap_to_grid(
                        (pin_x + d * angle.cos()).clamp(cx_min, cx_max),
                        grid_step,
                    );
                    let cy = snap_to_grid(
                        (pin_y + d * angle.sin()).clamp(cy_min, cy_max),
                        grid_step,
                    );
                    let c_pos = Point::new(cx, cy);

                    if is_placement_legal(board, fp, c_pos, rot, fp.layer, min_spacing) {
                        let pwr_pad_world = fp.pad_world_point_layered(pwr_idx, c_pos, rot, fp.layer);
                        let pad_dist = pwr_pad_world.distance_to(&pin_pt);
                        if best_cand.as_ref().map_or(true, |b| pad_dist < b.0) {
                            best_cand = Some((pad_dist, cx, cy, rot, fp.layer));
                            if pad_dist <= 1.0 {
                                return (cx, cy, rot, fp.layer);
                            }
                        }
                    }
                }
                }
            }

            if let Some((_, bx, by, brot, blayer)) = best_cand {
                return (bx, by, brot, blayer);
            }
        }
    }

    // General case: connected component or other passives/ICs
    // Find all pins on placed components that share nets with this footprint
    let mut targets: Vec<(usize, Point, f64)> = Vec::new();
    for (pidx, pad) in fp.pads.iter().enumerate() {
        if pad.net_name.is_empty() {
            continue;
        }
        let is_gnd = is_ground_net(&pad.net_name);
        let is_pwr = is_power_net(&pad.net_name);
        let weight = if !is_gnd && !is_pwr { 5.0 } else if is_pwr { 2.0 } else { 0.5 };

        for placed_fp in &board.footprints {
            for other_pad in &placed_fp.pads {
                if other_pad.net_name == pad.net_name {
                    let (ox, oy) = placed_fp.pad_absolute_position(other_pad);
                    targets.push((pidx, Point::new(ox, oy), weight));
                }
            }
        }
    }

    let target_center = if !targets.is_empty() {
        let total_weight: f64 = targets.iter().map(|t| t.2).sum();
        let sum_x: f64 = targets.iter().map(|t| t.1.x * t.2).sum();
        let sum_y: f64 = targets.iter().map(|t| t.1.y * t.2).sum();
        Point::new(sum_x / total_weight, sum_y / total_weight)
    } else {
        Point::new((boundary.min_x + boundary.max_x) * 0.5, (boundary.min_y + boundary.max_y) * 0.5)
    };

    // Concentric search around target_center
    let radii = [
        0.0, 0.8, 1.5, 2.5, 3.5, 5.0, 6.5, 8.0, 10.0, 12.5, 15.0, 18.0, 22.0, 26.0, 30.0, 35.0, 42.0, 50.0,
    ];
    let mut best_cand: Option<(f64, f64, f64, f64, usize)> = None;

    for &r in &radii {
        let steps = if r == 0.0 {
            1
        } else {
            ((2.0 * std::f64::consts::PI * r / grid_step) as usize).clamp(8, 32)
        };

        for s in 0..steps {
            let theta = (s as f64) * (2.0 * std::f64::consts::PI / steps as f64);
            for &rot in &rotations {
                let (cx_min, cx_max, cy_min, cy_max) = fp.origin_clamp_range(rot, fp.layer, min_spacing, &boundary);
                let cx = snap_to_grid(
                    (target_center.x + r * theta.cos()).clamp(cx_min, cx_max),
                    grid_step,
                );
                let cy = snap_to_grid(
                    (target_center.y + r * theta.sin()).clamp(cy_min, cy_max),
                    grid_step,
                );
                let c_pos = Point::new(cx, cy);

                if is_placement_legal(board, fp, c_pos, rot, fp.layer, min_spacing) {
                    let mut cost = 0.1 * r;
                    for &(pidx, target_pt, w) in &targets {
                        let pad_pt = fp.pad_world_point_layered(pidx, c_pos, rot, fp.layer);
                        cost += w * pad_pt.distance_to(&target_pt);
                    }
                    if best_cand.as_ref().map_or(true, |b| cost < b.0) {
                        best_cand = Some((cost, cx, cy, rot, fp.layer));
                    }
                }
            }
        }
        if r >= 6.0 && best_cand.is_some() {
            break;
        }
    }

    if let Some((_, bx, by, brot, blayer)) = best_cand {
        return (bx, by, brot, blayer);
    }

    // Fallback: clamped within board outline with component's exact bounds
    let (cx_min, cx_max, cy_min, cy_max) = fp.origin_clamp_range(fp.rotation, fp.layer, min_spacing, &boundary);
    (
        fp.x.clamp(cx_min, cx_max),
        fp.y.clamp(cy_min, cy_max),
        fp.rotation,
        fp.layer,
    )
}

/// Places a single component from the holding palette at its most favourable location and orientation on the board.
pub fn place_holding_component_at_best_location(
    board: &mut Board,
    idx: usize,
    config: &PlacementConfig,
) -> Option<crate::board::Footprint> {
    if idx >= board.holding_palette.len() {
        return None;
    }
    let mut fp = board.holding_palette.remove(idx);
    let (bx, by, brot, blayer) = find_most_favourable_placement(board, &fp, config);
    fp.x = bx;
    fp.y = by;
    fp.rotation = brot;
    fp.layer = blayer;
    board.footprints.push(fp.clone());
    Some(fp)
}

/// Places all components in the holding palette at their most favourable locations and orientations on the board.
pub fn place_all_holding_components_at_best_locations(
    board: &mut Board,
    config: &PlacementConfig,
) -> usize {
    if board.holding_palette.is_empty() {
        return 0;
    }
    let mut palette = std::mem::take(&mut board.holding_palette);
    // Sort by priority: differential pairs first, then decoupling caps, then active ICs, then passives
    palette.sort_by(|a, b| {
        let prio = |fp: &crate::board::Footprint| -> usize {
            let is_diff = fp.pads.iter().any(|p| {
                let u = p.net_name.to_ascii_uppercase();
                u.contains("USB") || u.contains("DP") || u.contains("DM") || u.contains("D+") || u.contains("D-")
            });
            if is_diff {
                return 0;
            }
            let has_gnd = fp.pads.iter().any(|p| is_ground_net(&p.net_name));
            let has_pwr = fp.pads.iter().any(|p| is_power_net(&p.net_name));
            if fp.reference.starts_with('C') && has_gnd && has_pwr {
                return 1;
            }
            if fp.reference.starts_with('U') || fp.reference.starts_with('Q') {
                return 2;
            }
            3
        };
        prio(a).cmp(&prio(b))
    });

    let count = palette.len();
    for mut fp in palette {
        let (bx, by, brot, blayer) = find_most_favourable_placement(board, &fp, config);
        fp.x = bx;
        fp.y = by;
        fp.rotation = brot;
        fp.layer = blayer;
        board.footprints.push(fp);
    }
    count
}

/// Precompute functional component clusters for rigid group moves (e.g. IC + tied decoupling caps/resistors).
pub fn build_component_clusters(
    board: &Board,
    ties: &[PassiveTie],
    movable: &[usize],
) -> Vec<Vec<usize>> {
    let mut clusters: Vec<Vec<usize>> = (0..board.footprints.len()).map(|i| vec![i]).collect();
    for tie in ties {
        if movable.contains(&tie.anchor_idx) && movable.contains(&tie.passive_idx) {
            if !clusters[tie.anchor_idx].contains(&tie.passive_idx) {
                clusters[tie.anchor_idx].push(tie.passive_idx);
            }
        }
    }
    for tie in ties {
        if movable.contains(&tie.anchor_idx) && movable.contains(&tie.passive_idx) {
            let anchor_cluster = clusters[tie.anchor_idx].clone();
            clusters[tie.passive_idx] = anchor_cluster;
        }
    }
    clusters
}

/// Rigidly translate an entire cluster of components by (dx, dy) if all members stay within board bounds.
pub fn apply_cluster_group_move(
    state: &mut PlacementState,
    board: &Board,
    config: &PlacementConfig,
    cluster: &[usize],
    dx: f64,
    dy: f64,
    boundary: &AABB,
) -> bool {
    if cluster.is_empty() || (dx.abs() <= 1e-4 && dy.abs() <= 1e-4) {
        return false;
    }
    let all_fit = cluster.iter().all(|&m| {
        let (cx_min, cx_max, cy_min, cy_max) = board.footprints[m]
            .origin_clamp_range(state.rotations[m], state.layers[m], config.min_spacing_mm, boundary);
        let nx = state.positions[m].x + dx;
        let ny = state.positions[m].y + dy;
        nx >= cx_min && nx <= cx_max && ny >= cy_min && ny <= cy_max
    });
    if all_fit {
        for &m in cluster {
            state.positions[m].x += dx;
            state.positions[m].y += dy;
            state.update_component(m, board, config);
        }
        true
    } else {
        false
    }
}

/// Metropolis-Hastings replica exchange between adjacent temperature rungs in Parallel Tempering.
/// Swaps configurations between adjacent rungs (i, i+1) with acceptance probability:
/// P_swap = min(1, exp((beta_i - beta_{i+1}) * (E_i - E_{i+1})))
pub fn metropolis_replica_exchange(
    replicas: &mut [PlacementState],
    num_threads: usize,
    epoch: usize,
    progress: f64,
    temp_scale: f64,
    thermal_boost: f64,
    rng: &mut impl rand::Rng,
) -> usize {
    if replicas.len() < 2 {
        return 0;
    }
    let cooling = (1.0 - progress * 0.85).max(0.05);
    let start_idx = epoch % 2;
    let mut accepted_swaps = 0;
    for i in (start_idx..replicas.len().saturating_sub(1)).step_by(2) {
        let j = i + 1;
        let t_i = 8.0 * (350.0f64 / 8.0).powf((i as f64) / ((num_threads - 1).max(1) as f64)) * temp_scale * cooling * thermal_boost;
        let t_j = 8.0 * (350.0f64 / 8.0).powf((j as f64) / ((num_threads - 1).max(1) as f64)) * temp_scale * cooling * thermal_boost;
        let beta_i = 1.0 / t_i.max(1e-4);
        let beta_j = 1.0 / t_j.max(1e-4);
        let delta_beta = beta_i - beta_j; // beta_i > beta_j since rung i has lower temp than rung j
        let delta_energy = replicas[i].cost - replicas[j].cost;
        let log_p = delta_beta * delta_energy;
        let swap = if log_p >= 0.0 {
            true
        } else {
            rng.gen::<f64>() < log_p.exp()
        };
        if swap {
            replicas.swap(i, j);
            accepted_swaps += 1;
        }
    }
    accepted_swaps
}

pub fn run_auto_placer(
    board: &mut Board,
    config: &PlacementConfig,
    live: Option<Arc<Mutex<LiveTrialState>>>,
    stop_flag: Arc<AtomicBool>,
) -> PlacementState {
    // 1. Place any components from holding palette at favourable locations
    if !board.holding_palette.is_empty() {
        let count = place_all_holding_components_at_best_locations(board, config);
        if let Some(ref st) = live {
            if let Ok(mut g) = st.lock() {
                g.push_log(format!("[Auto-Placer] Placed {} components from holding palette", count));
            }
        }
    }

    // 2. Prioritize initial placement if it improves upon the existing board placement
    let current_st = PlacementState::from_board(board, config);
    let current_legal = legalize_placement(board, config, current_st, live.as_ref());

    let mut prioritized_board = board.clone();
    prioritize_and_position_initial_components(&mut prioritized_board, config, live.as_ref());
    let prioritized_st = PlacementState::from_board(&prioritized_board, config);
    let prioritized_legal = legalize_placement(&prioritized_board, config, prioritized_st, live.as_ref());

    let legalized = if prioritized_legal.cost < current_legal.cost && prioritized_legal.overlaps <= current_legal.overlaps {
        *board = prioritized_board;
        prioritized_legal
    } else {
        current_legal
    };

    // 3. Optimize pin and gate swapping on initial placement
    let initial_swaps = optimize_pin_and_gate_swapping(board, config);
    if initial_swaps > 0 {
        if let Some(ref st) = live {
            if let Ok(mut g) = st.lock() {
                g.push_log(format!("[Auto-Placer] Initial pin/gate swapping untangled {} connections", initial_swaps));
                g.refresh_ratsnest();
            }
        }
    }

    let movable: Vec<usize> = board
        .footprints
        .iter()
        .enumerate()
        .filter(|(_, fp)| !fp.is_locked)
        .map(|(i, _)| i)
        .collect();

    if movable.is_empty() || config.duration_secs == 0 {
        legalized.apply_to_board(board);
        return legalized;
    }

    let active_vulkan = config.vulkan.clone();
    let num_threads = rayon::current_num_threads().max(2);
    let global_best = Arc::new(Mutex::new(legalized.clone()));
    let total_evals = Arc::new(AtomicUsize::new(0));
    let start_time = Instant::now();
    let total_dur = Duration::from_secs(config.duration_secs);
    let boundary = board.boundary_aabb();
    let ties = Arc::new(build_passive_ties(board));

    // Precompute connected foreign pad references for barycentric centroid pull moves.
    // Exclude plane ground and wide distribution nets so components are not pulled into central IC ground pads.
    let mut connected_pins: Vec<Vec<(usize, usize)>> = vec![Vec::new(); board.footprints.len()];
    for (net_id, net_name) in board.nets.iter().enumerate() {
        if net_id == 0 || is_ground_net(net_name) {
            continue;
        }
        let mut net_pads: Vec<(usize, usize)> = Vec::new();
        for (f_idx, fp) in board.footprints.iter().enumerate() {
            for (p_idx, p) in fp.pads.iter().enumerate() {
                if p.net_id == net_id {
                    net_pads.push((f_idx, p_idx));
                }
            }
        }
        if net_pads.len() > 8 {
            continue;
        }
        for &(fa, _) in &net_pads {
            for &(fb, pb) in &net_pads {
                if fa != fb && !connected_pins[fa].contains(&(fb, pb)) {
                    connected_pins[fa].push((fb, pb));
                }
            }
        }
    }
    let connected_pins = Arc::new(connected_pins);
    let component_clusters = Arc::new(build_component_clusters(board, &ties, &movable));

    if let Some(ref st) = live {
        if let Ok(mut g) = st.lock() {
            g.phase = TrialPhase::AutoPlacing;
            let gpu_tag = if let Some(ref vk) = active_vulkan {
                format!(" + Vulkan GPU ({})", vk.device_name)
            } else {
                String::new()
            };
            g.push_log(format!(
                "[Auto-Placer] Launching {} Parallel Tempering CPU SA replicas{} for {}s (jump_r={:.1}mm, swap={:.0}%, centroid={:.0}%)...",
                num_threads, gpu_tag, config.duration_secs,
                config.max_jump_radius_mm, config.pair_swap_prob * 100.0, config.centroid_pull_prob * 100.0
            ));
        }
    }

    let num_epochs = (config.duration_secs * 4).max(1) as usize;
    let epoch_dur = Duration::from_millis(250);
    let mut replicas: Vec<PlacementState> = (0..num_threads)
        .map(|tid| {
            if tid == 0 {
                legalized.clone()
            } else {
                let mut rep = legalized.clone();
                // Higher rungs start from the legal state and optionally apply a small number of
                // compatible passive pair swaps to diversify topological exploration without overlaps.
                if tid > 0 && movable.len() >= 2 {
                    let mut init_rng = StdRng::seed_from_u64(0xD15EA5Eu64.wrapping_mul((tid as u64) + 1));
                    let swaps = (tid.min(4)).min(movable.len() / 2);
                    for _ in 0..swaps {
                        let a = movable[init_rng.gen_range(0..movable.len())];
                        let fp_a = &board.footprints[a];
                        let compat: Vec<usize> = movable.iter().copied()
                            .filter(|&m| m != a && {
                                let fp_m = &board.footprints[m];
                                (fp_m.width - fp_a.width).abs() <= 1.0 && (fp_m.height - fp_a.height).abs() <= 1.0
                            })
                            .collect();
                        if !compat.is_empty() {
                            let b = compat[init_rng.gen_range(0..compat.len())];
                            let pos_a = rep.positions[a];
                            let pos_b = rep.positions[b];
                            rep.positions[a] = pos_b;
                            rep.positions[b] = pos_a;
                            rep.update_component(a, board, config);
                            rep.update_component(b, board, config);
                        }
                    }
                    rep.evaluate(board, config, &ties);
                    // If initial swaps created any overlap or boundary violation, revert cleanly to legalized
                    if rep.overlaps > legalized.overlaps || rep.boundary_violations > 0 {
                        rep = legalized.clone();
                    }
                }
                rep
            }
        })
        .collect();
    let gpu_done = AtomicBool::new(false);

    std::thread::scope(|scope| {
        if let Some(ref vk) = active_vulkan {
            let (fp_descs, edges) = build_gpu_sa_descriptors(board, config, &ties, &legalized);
            let fp_descs = Arc::new(fp_descs);
            let edges = Arc::new(edges);
            for (gpu_idx, vk_dev) in vk.all_engines().into_iter().enumerate() {
                let board_ref = &*board;
                let ties_ref = Arc::clone(&ties);
                let movable_ref = &movable;
                let global_best_ref = Arc::clone(&global_best);
                let total_evals_ref = Arc::clone(&total_evals);
                let stop_ref = Arc::clone(&stop_flag);
                let gpu_done_ref = &gpu_done;
                let fp_descs_ref = Arc::clone(&fp_descs);
                let edges_ref = Arc::clone(&edges);

                scope.spawn(move || {
                    let mut batch_idx = (gpu_idx as u64).wrapping_mul(10_000);
                    while !gpu_done_ref.load(Ordering::Relaxed)
                        && !stop_ref.load(Ordering::Relaxed)
                        && start_time.elapsed() + Duration::from_millis(25) < total_dur
                    {
                        let progress = (start_time.elapsed().as_secs_f64() / total_dur.as_secs_f64()).clamp(0.0, 1.0);
                        let cooling = (1.0 - progress * 0.85).max(0.05);
                        let base = global_best_ref.lock().unwrap().clone();
                        if let Some((cand, gpu_evals)) = run_gpu_placement_batch(
                            &vk_dev,
                            board_ref,
                            config,
                            &ties_ref,
                            movable_ref,
                            &base,
                            &fp_descs_ref,
                            &edges_ref,
                            batch_idx,
                            cooling,
                        ) {
                            total_evals_ref.fetch_add(gpu_evals, Ordering::Relaxed);
                            if let Ok(mut g) = global_best_ref.lock() {
                                let is_better = if cand.boundary_violations != g.boundary_violations {
                                    cand.boundary_violations < g.boundary_violations
                                } else if cand.overlaps != g.overlaps {
                                    cand.overlaps < g.overlaps
                                } else {
                                    cand.cost < g.cost
                                };
                                if is_better {
                                    *g = cand;
                                }
                            }
                        }
                        batch_idx = batch_idx.wrapping_add(1);
                    }
                });
            }
        }

        let mut last_best_cost = f64::MAX;
        let mut stagnant_epochs = 0usize;
        let mut thermal_pulse_epochs = 0usize;

        for epoch in 0..num_epochs {
            if stop_flag.load(Ordering::Relaxed) || start_time.elapsed() >= total_dur {
                break;
            }

            let cur_best_cost = global_best.lock().unwrap().cost;
            if cur_best_cost < last_best_cost - (last_best_cost.abs() * 0.005) {
                last_best_cost = cur_best_cost;
                stagnant_epochs = 0;
            } else {
                stagnant_epochs += 1;
            }

            if stagnant_epochs >= 4 {
                thermal_pulse_epochs = 3;
                stagnant_epochs = 0;
                if let Some(ref st) = live {
                    if let Ok(mut g) = st.lock() {
                        g.push_log(format!(
                            "[Auto-Placer] Stagnation detected at cost {:.1}: triggering thermal pulse & legal basin hop...",
                            cur_best_cost
                        ));
                    }
                }
                // Basin Hop: reset hottest replica to global_best and apply a few compatible pair swaps
                if let Some(hottest) = replicas.last_mut() {
                    let best = global_best.lock().unwrap().clone();
                    *hottest = best;
                    let mut hop_rng = StdRng::seed_from_u64(0xBA510_709u64.wrapping_add(epoch as u64));
                    let swaps = 3.min(movable.len() / 2);
                    for _ in 0..swaps {
                        let a = movable[hop_rng.gen_range(0..movable.len())];
                        let fp_a = &board.footprints[a];
                        let compat: Vec<usize> = movable.iter().copied()
                            .filter(|&m| m != a && {
                                let fp_m = &board.footprints[m];
                                (fp_m.width - fp_a.width).abs() <= 1.0 && (fp_m.height - fp_a.height).abs() <= 1.0
                            })
                            .collect();
                        if !compat.is_empty() {
                            let b = compat[hop_rng.gen_range(0..compat.len())];
                            let pos_a = hottest.positions[a];
                            let pos_b = hottest.positions[b];
                            hottest.positions[a] = pos_b;
                            hottest.positions[b] = pos_a;
                            hottest.update_component(a, board, config);
                            hottest.update_component(b, board, config);
                        }
                    }
                    hottest.evaluate(board, config, &ties);
                    if hottest.overlaps > 0 || hottest.boundary_violations > 0 {
                        *hottest = global_best.lock().unwrap().clone();
                    }
                }
            }

            let thermal_boost = if thermal_pulse_epochs > 0 {
                thermal_pulse_epochs -= 1;
                3.5
            } else {
                1.0
            };

            let progress = (start_time.elapsed().as_secs_f64() / total_dur.as_secs_f64()).clamp(0.0, 1.0);
            let ep_end = (Instant::now() + epoch_dur).min(start_time + total_dur);

            replicas
                .par_iter_mut()
                .enumerate()
                .for_each(|(tid, current)| {
                    let mut rng = StdRng::seed_from_u64(
                        0x9E3779B97F4A7C15u64
                            .wrapping_mul((tid as u64) + 1)
                            .wrapping_add(epoch as u64),
                    );
                    let temp_ratio = (tid as f64) / ((num_threads - 1).max(1) as f64);
                    let t_base = 8.0 * (350.0f64 / 8.0).powf(temp_ratio) * config.temp_scale;
                    let cooling = (1.0 - progress * 0.85).max(0.05);
                    let temp = t_base * cooling * thermal_boost;
                    let mut local_evals = 0usize;

                    let p_swap = (config.pair_swap_prob * 100.0).clamp(0.0, 30.0) as usize;
                    let p_centroid = (config.centroid_pull_prob * 100.0).clamp(0.0, 40.0) as usize;
                    let p_cluster = 15usize;
                    let p_rot = 16usize;
                    let p_flip = if config.allow_layer_flip { 8usize } else { 0usize };
                    let p_affinity = if !ties.is_empty() { 12usize } else { 0usize };

                    while Instant::now() < ep_end && !stop_flag.load(Ordering::Relaxed) {
                        for _ in 0..40 {
                            let old_positions = current.positions.clone();
                            let old_rotations = current.rotations.clone();
                            let old_layers = current.layers.clone();
                            let old_cost = current.cost;
                            let old_crossings = current.crossings;
                            let old_wl = current.wirelength;
                            let old_ov = current.overlaps;
                            let old_bv = current.boundary_violations;
                            let mut modified: Vec<usize> = Vec::with_capacity(8);

                            let roll = rng.gen_range(0..100);
                            let t1 = p_swap;
                            let t2 = t1 + p_centroid;
                            let t3 = t2 + p_cluster;
                            let t4 = t3 + p_rot;
                            let t5 = t4 + p_flip;
                            let t6 = t5 + p_affinity;

                            if roll < t1 && movable.len() >= 2 {
                                // 1. Component Pair Swap
                                let a = movable[rng.gen_range(0..movable.len())];
                                let fp_a = &board.footprints[a];
                                let pos_a = current.positions[a];
                                let b = if rng.gen_bool(0.75) {
                                    let compatible: Vec<usize> = movable
                                        .iter()
                                        .copied()
                                        .filter(|&m| {
                                            if m == a {
                                                return false;
                                            }
                                            let fp_m = &board.footprints[m];
                                            let dim_match = (fp_m.width - fp_a.width).abs() <= 1.2
                                                && (fp_m.height - fp_a.height).abs() <= 1.2;
                                            let nearby = current.positions[m].distance_to(&pos_a) <= 15.0;
                                            dim_match || nearby
                                        })
                                        .collect();
                                    if !compatible.is_empty() {
                                        compatible[rng.gen_range(0..compatible.len())]
                                    } else {
                                        movable[rng.gen_range(0..movable.len())]
                                    }
                                } else {
                                    movable[rng.gen_range(0..movable.len())]
                                };

                                if a != b {
                                    let rot_a = current.rotations[a];
                                    let rot_b = current.rotations[b];
                                    let layer_a = current.layers[a];
                                    let layer_b = current.layers[b];
                                    let (a_min_x, a_max_x, a_min_y, a_max_y) = board.footprints[a]
                                        .origin_clamp_range(rot_b, layer_b, config.min_spacing_mm, &boundary);
                                    let (b_min_x, b_max_x, b_min_y, b_max_y) = board.footprints[b]
                                        .origin_clamp_range(rot_a, layer_a, config.min_spacing_mm, &boundary);
                                    let pos_b = current.positions[b];
                                    current.positions[a] = Point::new(pos_b.x.clamp(a_min_x, a_max_x), pos_b.y.clamp(a_min_y, a_max_y));
                                    current.positions[b] = Point::new(pos_a.x.clamp(b_min_x, b_max_x), pos_a.y.clamp(b_min_y, b_max_y));
                                    current.rotations[a] = rot_b;
                                    current.rotations[b] = rot_a;
                                    current.layers[a] = layer_b;
                                    current.layers[b] = layer_a;
                                    modified.push(a);
                                    modified.push(b);
                                }
                            } else if roll < t2 {
                                // 2. Barycentric Centroid Pull (magnetic pull towards connected nets)
                                let idx = movable[rng.gen_range(0..movable.len())];
                                let pins = &connected_pins[idx];
                                if !pins.is_empty() {
                                    let mut sum_x = 0.0;
                                    let mut sum_y = 0.0;
                                    for &(other_fp, other_pad) in pins {
                                        let pt = current.cached_pads[other_fp][other_pad];
                                        sum_x += pt.x;
                                        sum_y += pt.y;
                                    }
                                    let cx = sum_x / pins.len() as f64;
                                    let cy = sum_y / pins.len() as f64;
                                    let p_curr = current.positions[idx];
                                    let vec_x = cx - p_curr.x;
                                    let vec_y = cy - p_curr.y;
                                    let dist = vec_x.hypot(vec_y);
                                    if dist > 0.1 {
                                        let max_step = (3.5 * cooling).max(config.grid_step_mm * 1.5);
                                        let step = rng.gen_range(0.5 * config.grid_step_mm..=max_step).min(dist * 0.4);
                                        let jitter = (1.5 * cooling).max(config.grid_step_mm * 0.5);
                                        let target_x = p_curr.x + (vec_x / dist) * step + rng.gen_range(-jitter..=jitter);
                                        let target_y = p_curr.y + (vec_y / dist) * step + rng.gen_range(-jitter..=jitter);

                                        let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx]
                                            .origin_clamp_range(current.rotations[idx], current.layers[idx], config.min_spacing_mm, &boundary);
                                        let nx = snap_to_grid(target_x.clamp(cx_min, cx_max), config.grid_step_mm);
                                        let ny = snap_to_grid(target_y.clamp(cy_min, cy_max), config.grid_step_mm);
                                        current.positions[idx] = Point::new(nx, ny);
                                        modified.push(idx);
                                    }
                                }
                            } else if roll < t3 {
                                // 3. Cluster Rigid Group Move (move IC and tied decoupling passives together)
                                let idx = movable[rng.gen_range(0..movable.len())];
                                let cluster = &component_clusters[idx];
                                if cluster.len() > 1 {
                                    let max_shift = (config.max_jump_radius_mm * cooling).max(config.grid_step_mm * 2.0);
                                    let (shift_x, shift_y) = if rng.gen_bool(0.08) {
                                        (
                                            rng.gen_range(-config.max_jump_radius_mm..=config.max_jump_radius_mm),
                                            rng.gen_range(-config.max_jump_radius_mm..=config.max_jump_radius_mm),
                                        )
                                    } else {
                                        (
                                            rng.gen_range(-max_shift..=max_shift),
                                            rng.gen_range(-max_shift..=max_shift),
                                        )
                                    };
                                    let dx = snap_to_grid(shift_x, config.grid_step_mm);
                                    let dy = snap_to_grid(shift_y, config.grid_step_mm);
                                    if dx.abs() > 1e-4 || dy.abs() > 1e-4 {
                                        let all_fit = cluster.iter().all(|&m| {
                                            let (cx_min, cx_max, cy_min, cy_max) = board.footprints[m]
                                                .origin_clamp_range(current.rotations[m], current.layers[m], config.min_spacing_mm, &boundary);
                                            let nx = current.positions[m].x + dx;
                                            let ny = current.positions[m].y + dy;
                                            nx >= cx_min && nx <= cx_max && ny >= cy_min && ny <= cy_max
                                        });
                                        if all_fit {
                                            for &m in cluster {
                                                current.positions[m].x += dx;
                                                current.positions[m].y += dy;
                                                modified.push(m);
                                            }
                                        }
                                    }
                                } else {
                                    // Single component shift fallback
                                    let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx]
                                        .origin_clamp_range(current.rotations[idx], current.layers[idx], config.min_spacing_mm, &boundary);
                                    let max_shift = (config.max_jump_radius_mm * cooling).max(config.grid_step_mm * 2.0);
                                    let (shift_x, shift_y) = if rng.gen_bool(0.06) {
                                        (
                                            rng.gen_range(-config.max_jump_radius_mm..=config.max_jump_radius_mm),
                                            rng.gen_range(-config.max_jump_radius_mm..=config.max_jump_radius_mm),
                                        )
                                    } else {
                                        (
                                            rng.gen_range(-max_shift..=max_shift),
                                            rng.gen_range(-max_shift..=max_shift),
                                        )
                                    };
                                    let nx = snap_to_grid((current.positions[idx].x + shift_x).clamp(cx_min, cx_max), config.grid_step_mm);
                                    let ny = snap_to_grid((current.positions[idx].y + shift_y).clamp(cy_min, cy_max), config.grid_step_mm);
                                    current.positions[idx] = Point::new(nx, ny);
                                    modified.push(idx);
                                }
                            } else if roll < t4 {
                                // 4. Rotation (90°, 180°, 270°)
                                let idx = movable[rng.gen_range(0..movable.len())];
                                // 180° rotation preserves bounding box extents and never introduces overlaps for rectangular passives!
                                let delta = if rng.gen_bool(0.60) {
                                    180.0
                                } else {
                                    [90.0, 270.0][rng.gen_range(0..2)]
                                };
                                let new_rot = (current.rotations[idx] + delta) % 360.0;
                                current.rotations[idx] = new_rot;
                                let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx]
                                    .origin_clamp_range(new_rot, current.layers[idx], config.min_spacing_mm, &boundary);
                                current.positions[idx].x = current.positions[idx].x.clamp(cx_min, cx_max);
                                current.positions[idx].y = current.positions[idx].y.clamp(cy_min, cy_max);
                                modified.push(idx);
                            } else if roll < t5 && config.allow_layer_flip {
                                // 5. Layer Flip (F.Cu <-> B.Cu)
                                let idx = movable[rng.gen_range(0..movable.len())];
                                if board.footprints[idx].is_smd_only() {
                                    let new_layer = 1 - current.layers[idx];
                                    current.layers[idx] = new_layer;
                                    let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx]
                                        .origin_clamp_range(current.rotations[idx], new_layer, config.min_spacing_mm, &boundary);
                                    current.positions[idx].x = current.positions[idx].x.clamp(cx_min, cx_max);
                                    current.positions[idx].y = current.positions[idx].y.clamp(cy_min, cy_max);
                                    modified.push(idx);
                                }
                            } else if roll < t6 && !ties.is_empty() {
                                // 6. Decoupling & Passive Tie Affinity Move
                                let tie = &ties[rng.gen_range(0..ties.len())];
                                let idx = tie.passive_idx;
                                let anchor_pt = current.cached_pads[tie.anchor_idx][tie.anchor_pad_idx];
                                let angles = [0.0, 90.0, 180.0, 270.0];
                                let new_rot = angles[rng.gen_range(0..4)];
                                current.rotations[idx] = new_rot;
                                let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx]
                                    .origin_clamp_range(new_rot, current.layers[idx], config.min_spacing_mm, &boundary);

                                if tie.is_decoupling {
                                    let anchor_center = current.positions[tie.anchor_idx];
                                    let outward_angle = (anchor_pt.y - anchor_center.y)
                                        .atan2(anchor_pt.x - anchor_center.x);
                                    let theta = outward_angle + rng.gen_range(-1.2..=1.2);
                                    let anchor_escape = if config.enable_escape_clearance {
                                        board.footprints[tie.anchor_idx].escape_margin * config.escape_margin_mult
                                    } else {
                                        0.0
                                    };
                                    let min_r = 1.1 + anchor_escape;
                                    let max_r = 2.2 + anchor_escape * 1.5;
                                    let r = rng.gen_range(min_r..=max_r);
                                    let nx = snap_to_grid(
                                        (anchor_pt.x + r * theta.cos())
                                            .clamp(cx_min, cx_max),
                                        config.grid_step_mm,
                                    );
                                    let ny = snap_to_grid(
                                        (anchor_pt.y + r * theta.sin())
                                            .clamp(cy_min, cy_max),
                                        config.grid_step_mm,
                                    );
                                    current.positions[idx] = Point::new(nx, ny);
                                    modified.push(idx);
                                } else {
                                    let nx = snap_to_grid(
                                        (anchor_pt.x + rng.gen_range(-2.5..=2.5))
                                            .clamp(cx_min, cx_max),
                                        config.grid_step_mm,
                                    );
                                    let ny = snap_to_grid(
                                        (anchor_pt.y + rng.gen_range(-2.5..=2.5))
                                            .clamp(cy_min, cy_max),
                                        config.grid_step_mm,
                                    );
                                    current.positions[idx] = Point::new(nx, ny);
                                    modified.push(idx);
                                }
                            } else {
                                // 7. Positional Shift & Long-Range Jumps
                                let idx = movable[rng.gen_range(0..movable.len())];
                                let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx]
                                    .origin_clamp_range(current.rotations[idx], current.layers[idx], config.min_spacing_mm, &boundary);
                                let (shift_x, shift_y) = if rng.gen_bool(0.04) {
                                    // 4% chance of macro teleport jump across full max_jump_radius
                                    (
                                        rng.gen_range(-config.max_jump_radius_mm..=config.max_jump_radius_mm),
                                        rng.gen_range(-config.max_jump_radius_mm..=config.max_jump_radius_mm),
                                    )
                                } else if rng.gen_bool(0.40) {
                                    // 40% chance of small fine-tuning grid nudge (1-2 grid steps)
                                    let step = config.grid_step_mm;
                                    let dx = [-step, 0.0, step][rng.gen_range(0..3)];
                                    let dy = [-step, 0.0, step][rng.gen_range(0..3)];
                                    (dx, dy)
                                } else {
                                    let max_shift = (config.max_jump_radius_mm * cooling * 0.5).max(config.grid_step_mm * 2.0);
                                    (
                                        rng.gen_range(-max_shift..=max_shift),
                                        rng.gen_range(-max_shift..=max_shift),
                                    )
                                };
                                let nx = snap_to_grid(
                                    (current.positions[idx].x + shift_x).clamp(cx_min, cx_max),
                                    config.grid_step_mm,
                                );
                                let ny = snap_to_grid(
                                    (current.positions[idx].y + shift_y).clamp(cy_min, cy_max),
                                    config.grid_step_mm,
                                );
                                current.positions[idx] = Point::new(nx, ny);
                                modified.push(idx);
                            }

                            if modified.is_empty() {
                                continue;
                            }
                            for &m in &modified {
                                current.update_component(m, board, config);
                            }
                            current.evaluate(board, config, &ties);
                            local_evals += 1;

                            let delta = current.cost - old_cost;
                            let accept = if current.boundary_violations > old_bv {
                                false
                            } else if delta <= 0.0 {
                                true
                            } else {
                                current.boundary_violations == 0 && rng.gen::<f64>() < (-delta / temp).exp()
                            };

                            if accept {
                                if let Ok(mut g) = global_best.lock() {
                                    let is_better = if current.boundary_violations != g.boundary_violations {
                                        current.boundary_violations < g.boundary_violations
                                    } else if current.overlaps != g.overlaps {
                                        current.overlaps < g.overlaps
                                    } else {
                                        current.cost < g.cost
                                    };
                                    if is_better {
                                        *g = current.clone();
                                    }
                                }
                            } else {
                                current.positions = old_positions;
                                current.rotations = old_rotations;
                                current.layers = old_layers;
                                for &m in &modified {
                                    current.update_component(m, board, config);
                                }
                                current.cost = old_cost;
                                current.crossings = old_crossings;
                                current.wirelength = old_wl;
                                current.overlaps = old_ov;
                                current.boundary_violations = old_bv;
                            }
                        }
                    }
                    total_evals.fetch_add(local_evals, Ordering::Relaxed);
                });

            // Parallel Tempering Metropolis-Hastings Replica Exchange:
            let mut exchange_rng = StdRng::seed_from_u64(0x5EA7_E7C4u64.wrapping_add(epoch as u64));
            metropolis_replica_exchange(
                &mut replicas,
                num_threads,
                epoch,
                progress,
                config.temp_scale,
                thermal_boost,
                &mut exchange_rng,
            );

            // Sync global_best if any replica discovered a lower cost or fewer violations
            {
                let mut g = global_best.lock().unwrap();
                for rep in &replicas {
                    let is_better = if rep.boundary_violations != g.boundary_violations {
                        rep.boundary_violations < g.boundary_violations
                    } else if rep.overlaps != g.overlaps {
                        rep.overlaps < g.overlaps
                    } else {
                        rep.cost < g.cost
                    };
                    if is_better {
                        *g = rep.clone();
                    }
                }
            }
            let best_snap = global_best.lock().unwrap().clone();

            let elapsed = start_time.elapsed().as_secs_f64().max(1e-3);
            let rate = total_evals.load(Ordering::Relaxed) as f64 / elapsed;

            if let Some(ref st) = live {
                if let Ok(mut g) = st.lock() {
                    best_snap.apply_to_board(&mut g.board);
                    g.ratsnest = g.board.generate_ratsnest_segments(&best_snap.cached_pads);
                    g.crossings = best_snap.crossings;
                    g.wirelength_mm = best_snap.wirelength;
                    g.overlaps = best_snap.overlaps;
                    g.evals_per_sec = rate;
                    g.progress = (progress as f32) * 0.5;
                    g.status_text = format!(
                        "Auto-Placing: Crossings={} | Wirelength={:.1}mm | {:.0} evals/s",
                        best_snap.crossings, best_snap.wirelength, rate
                    );
                }
            }
        }

        gpu_done.store(true, Ordering::Relaxed);
    });

    let mut final_best = global_best.lock().unwrap().clone();

    // Final containment verification pass:
    // Guarantee that every component's origin, courtyard vertices, and pad corners stay strictly inside board outline.
    let boundary = board.boundary_aabb();
    let min_spacing = config.min_spacing_mm;
    let board_center = Point::new(
        (boundary.min_x + boundary.max_x) * 0.5,
        (boundary.min_y + boundary.max_y) * 0.5,
    );

    for (i, fp) in board.footprints.iter().enumerate() {
        if fp.is_locked {
            continue;
        }
        let rot = final_best.rotations[i];
        let layer = final_best.layers[i];
        let (cx_min, cx_max, cy_min, cy_max) = fp.origin_clamp_range(rot, layer, min_spacing, &boundary);
        let mut pos = final_best.positions[i];
        let clamped_x = pos.x.clamp(cx_min, cx_max);
        let clamped_y = pos.y.clamp(cy_min, cy_max);
        if (clamped_x - pos.x).abs() > 1e-4 || (clamped_y - pos.y).abs() > 1e-4 {
            pos = Point::new(clamped_x, clamped_y);
            final_best.positions[i] = pos;
            final_best.update_component(i, board, config);
        }

        // Nudge towards center if any pad or courtyard vertex still violates board contour (e.g. rounded corners)
        let is_contained = {
            let mut ok = board.contains_point_in_boundary(&pos);
            if ok {
                let polys = fp.transformed_courtyards_layered(pos, rot, layer, min_spacing);
                for poly in &polys {
                    for v in &poly.vertices {
                        if !board.contains_point_in_boundary(v) {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        break;
                    }
                }
            }
            if ok {
                for pad in &fp.pads {
                    for corner_pt in fp.pad_corners_at(pad, pos, rot, layer) {
                        if !board.contains_point_in_boundary(&corner_pt) {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        break;
                    }
                }
            }
            ok
        };

        if !is_contained {
            let dx = board_center.x - pos.x;
            let dy = board_center.y - pos.y;
            let dist = dx.hypot(dy).max(1e-4);
            let ux = dx / dist;
            let uy = dy / dist;

            let mut step = 0.25;
            let mut resolved = false;
            while step <= 15.0 {
                let cand_pos = Point::new(
                    snap_to_grid(pos.x + ux * step, config.grid_step_mm),
                    snap_to_grid(pos.y + uy * step, config.grid_step_mm),
                );
                let cand_contained = {
                    let mut ok = board.contains_point_in_boundary(&cand_pos);
                    if ok {
                        let polys = fp.transformed_courtyards_layered(cand_pos, rot, layer, min_spacing);
                        for poly in &polys {
                            for v in &poly.vertices {
                                if !board.contains_point_in_boundary(v) {
                                    ok = false;
                                    break;
                                }
                            }
                            if !ok {
                                break;
                            }
                        }
                    }
                    if ok {
                        for pad in &fp.pads {
                            for corner_pt in fp.pad_corners_at(pad, cand_pos, rot, layer) {
                                if !board.contains_point_in_boundary(&corner_pt) {
                                    ok = false;
                                    break;
                                }
                            }
                            if !ok {
                                break;
                            }
                        }
                    }
                    ok
                };
                if cand_contained {
                    final_best.positions[i] = cand_pos;
                    final_best.update_component(i, board, config);
                    resolved = true;
                    break;
                }
                step += 0.25;
            }
            if !resolved {
                // ponytail: In extreme geometries fallback to clamped boundary
                final_best.positions[i] = Point::new(clamped_x, clamped_y);
                final_best.update_component(i, board, config);
            }
        }
    }

    final_best.apply_to_board(board);
    let final_swaps = optimize_pin_and_gate_swapping(board, config);
    if final_swaps > 0 {
        final_best = PlacementState::from_board(board, config);
        if let Some(ref st) = live {
            if let Ok(mut g) = st.lock() {
                g.push_log(format!("[Auto-Placer] Final pin/gate swapping untangled {} connections", final_swaps));
            }
        }
    }

    final_best.evaluate(board, config, &ties);
    if final_best.overlaps > 0 || final_best.boundary_violations > 0 {
        final_best = legalize_placement(board, config, final_best, live.as_ref());
    }
    final_best.apply_to_board(board);
    if let Some(ref st) = live {
        if let Ok(mut g) = st.lock() {
            final_best.apply_to_board(&mut g.board);
            g.ratsnest = g.board.generate_ratsnest_segments(&final_best.cached_pads);
            g.crossings = final_best.crossings;
            g.wirelength_mm = final_best.wirelength;
            g.overlaps = final_best.overlaps;
            g.push_log(format!(
                "[Auto-Placer] Finished: Crossings={}, Wirelength={:.1}mm, Overlaps={}",
                final_best.crossings, final_best.wirelength, final_best.overlaps
            ));
        }
    }
    final_best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{CompiledCluster, Footprint, NetClass, Pad};
    use std::collections::HashSet;
    use std::fs;

    fn make_test_fp(reference: &str, x: f64, y: f64, is_locked: bool, pads: Vec<Pad>) -> Footprint {
        Footprint {
            reference: reference.to_string(),
            value: "VAL".to_string(),
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

    fn make_synthetic_board() -> Board {
        let nets = vec![
            "".to_string(),
            "GND".to_string(),
            "+3.3V".to_string(),
            "SIG_A".to_string(),
            "SIG_B".to_string(),
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

        let u1 = make_test_fp(
            "U1",
            140.0,
            100.0,
            true,
            vec![
                Pad::new("1", -0.5, 0.0, 0.5, 0.5, 3, "SIG_A", false),
                Pad::new("2", 0.5, 0.0, 0.5, 0.5, 4, "SIG_B", false),
            ],
        );

        let j3 = make_test_fp(
            "J3",
            155.0,
            100.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 4, "SIG_B", true)],
        );

        // R10 overlaps C10 initially at (145.0, 100.0)
        let r10 = make_test_fp(
            "R10",
            145.0,
            100.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.5, 0.5, 3, "SIG_A", false),
                Pad::new("2", 0.5, 0.0, 0.5, 0.5, 4, "SIG_B", false),
            ],
        );

        let c10 = make_test_fp(
            "C10",
            145.2,
            100.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.5, 0.5, 4, "SIG_B", false),
                Pad::new("2", 0.5, 0.0, 0.5, 0.5, 1, "GND", false),
            ],
        );

        Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 125.0,
            min_y: 85.0,
            max_x: 170.0,
            max_y: 120.0,
            footprints: vec![u1, j3, r10, c10],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: vec![
                CompiledCluster {
                    name: "ANCHORED".to_string(),
                    anchor_idx: Some(0),
                    member_indices: vec![0, 2],
                    member_refs: vec!["U1".to_string(), "C1".to_string()],
                    max_radius: 2.0,
                    weight: 100.0,
                },
                CompiledCluster {
                    name: "CENTROID".to_string(),
                    anchor_idx: None,
                    member_indices: vec![2, 3],
                    member_refs: vec!["C1".to_string(), "R12".to_string()],
                    max_radius: 10.0,
                    weight: 60.0,
                },
            ],
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
        }
    }

    #[test]
    fn test_union_aabb_and_snap_to_grid() {
        let fb = union_aabb(&[], Point::new(5.0, 10.0));
        assert_eq!(fb, AABB::new(4.5, 9.5, 5.5, 10.5));

        let p = Polygon::new(vec![Point::new(1.0, 2.0), Point::new(4.0, 6.0)]);
        assert_eq!(union_aabb(&[p], Point::new(0.0, 0.0)), AABB::new(1.0, 2.0, 4.0, 6.0));

        assert!((snap_to_grid(1.23, 0.0) - 1.23).abs() < 1e-6);
        assert!((snap_to_grid(1.23, -1.0) - 1.23).abs() < 1e-6);
        assert!((snap_to_grid(1.30, 0.5) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_build_passive_ties_and_evaluation() {
        let mut board = make_synthetic_board();
        let cfg = PlacementConfig::default();
        let ties = build_passive_ties(&board);
        // R10 and C10 are unlocked passives; U1 has priority 3
        assert_eq!(ties.len(), 2);
        assert!(ties.iter().all(|t| t.anchor_idx == 0));

        let mut st = PlacementState::from_board(&board, &cfg);
        assert_eq!(st.overlaps, 1);

        // Flip C10 to B.Cu (layer = 1) at the exact same XY position: SMD parts on opposite sides must not overlap
        st.layers[3] = 1;
        st.update_component(3, &board, &cfg);
        st.evaluate(&board, &cfg, &ties);
        assert_eq!(st.overlaps, 0, "SMD footprints on opposite layers (F.Cu vs B.Cu) must not collide");

        // Move C10 out of bounds to exercise boundary_cost and far passive affinity
        st.positions[3] = Point::new(100.0, 50.0);
        st.update_component(3, &board, &cfg);
        st.evaluate(&board, &cfg, &ties);
        assert_eq!(st.overlaps, 0);
        assert!(st.cost > cfg.boundary_penalty);

        st.apply_to_board(&mut board);
        assert!((board.footprints[3].x - 100.0).abs() < 1e-6);
        assert_eq!(board.footprints[3].layer, 1);
        // Locked U1 must not move even if state position changed
        assert!((board.footprints[0].x - 140.0).abs() < 1e-6);
    }

    #[test]
    fn test_legalize_placement_and_run_auto_placer() {
        let mut board = make_synthetic_board();
        let vk = VulkanEngine::new();
        let cfg = PlacementConfig {
            duration_secs: 1,
            allow_layer_flip: true,
            vulkan: vk.clone(),
            ..PlacementConfig::default()
        };
        let live = Arc::new(Mutex::new(LiveTrialState::new(board.clone())));
        let initial = PlacementState::from_board(&board, &cfg);
        assert_eq!(initial.overlaps, 1);

        let legalized = legalize_placement(&board, &cfg, initial, Some(&live));
        assert_eq!(legalized.overlaps, 0);
        // Calling legalize_placement when overlaps == 0 returns immediately
        let already_legal = legalize_placement(&board, &cfg, legalized.clone(), None);
        assert_eq!(already_legal.overlaps, 0);

        // Calling legalize_placement when all footprints are locked returns immediately
        let mut all_locked = board.clone();
        for fp in &mut all_locked.footprints {
            fp.is_locked = true;
        }
        let mut locked_state = PlacementState::from_board(&all_locked, &cfg);
        locked_state.overlaps = 1;
        let locked_res = legalize_placement(&all_locked, &cfg, locked_state, None);
        assert_eq!(locked_res.overlaps, 1);

        // Test duration_secs == 0 fast path
        let cfg_zero = PlacementConfig {
            duration_secs: 0,
            ..PlacementConfig::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let st_zero = run_auto_placer(&mut board.clone(), &cfg_zero, None, Arc::clone(&stop));
        assert_eq!(st_zero.overlaps, 0);

        // Test full parallel tempering SA + Vulkan GPU batch auto-placer epoch with live telemetry
        let final_st = run_auto_placer(&mut board, &cfg, Some(Arc::clone(&live)), stop);
        assert_eq!(final_st.overlaps, 0);
        if let Some(ref v) = vk {
            assert!(v.compute_busy_ns.load(Ordering::Relaxed) > 0);
        }
        let g = live.lock().unwrap();
        assert_eq!(g.overlaps, 0);
        assert!(!g.log_lines.is_empty());
        let _ = HashSet::<String>::new();
    }

    #[test]
    fn test_is_ground_and_power_net() {
        assert!(is_ground_net("GND"));
        assert!(is_ground_net("/GND"));
        assert!(is_ground_net("AGND"));
        assert!(is_ground_net("DGND"));
        assert!(is_ground_net("PGND"));
        assert!(is_ground_net("GND_EARTH"));
        assert!(!is_ground_net("+3.3V"));
        assert!(!is_ground_net(""));

        assert!(is_power_net("+3.3V"));
        assert!(is_power_net("+5V"));
        assert!(is_power_net("+12V"));
        assert!(is_power_net("VCC_5V"));
        assert!(is_power_net("VRAW"));
        assert!(is_power_net("VBUS"));
        assert!(is_power_net("/BUCK_VIN"));
        assert!(is_power_net("VDD"));
        assert!(is_power_net("VBAT"));
        assert!(is_power_net("VIN"));
        assert!(is_power_net("3V3_OUT"));
        assert!(!is_power_net("GND"));
        assert!(!is_power_net("SIG_A"));
        assert!(!is_power_net("USB_DP"));
        assert!(!is_power_net(""));
    }

    #[test]
    fn test_decoupling_cap_identification_and_power_pin_targeting() {
        let nets = vec![
            "".to_string(),
            "GND".to_string(),
            "+3.3V".to_string(),
            "SIG_A".to_string(),
            "SIG_B".to_string(),
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

        // Active IC U1 at (100, 100) with Pin 1 (+3.3V power), Pin 2 (GND), Pin 3 (SIG_A)
        let u1 = make_test_fp(
            "U1",
            100.0,
            100.0,
            true,
            vec![
                Pad::new("1", -1.0, 0.0, 0.5, 0.5, 2, "+3.3V", false),
                Pad::new("2", 0.0, -1.0, 0.5, 0.5, 1, "GND", false),
                Pad::new("3", 1.0, 0.0, 0.5, 0.5, 3, "SIG_A", false),
            ],
        );

        // Active IC U2 at (180, 100) with Pin 1 (+3.3V power), Pin 2 (GND), Pin 3 (SIG_B)
        let u2 = make_test_fp(
            "U2",
            180.0,
            100.0,
            true,
            vec![
                Pad::new("1", -1.0, 0.0, 0.5, 0.5, 2, "+3.3V", false),
                Pad::new("2", 0.0, -1.0, 0.5, 0.5, 1, "GND", false),
                Pad::new("3", 1.0, 0.0, 0.5, 0.5, 4, "SIG_B", false),
            ],
        );

        // Decoupling cap C1 for U1: Pin 1 (+3.3V), Pin 2 (GND)
        let c1 = make_test_fp(
            "C1",
            102.0,
            100.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.4, 0.4, 2, "+3.3V", false),
                Pad::new("2", 0.5, 0.0, 0.4, 0.4, 1, "GND", false),
            ],
        );

        // Decoupling cap C2 for U2: Pin 1 (+3.3V), Pin 2 (GND)
        let c2 = make_test_fp(
            "C2",
            182.0,
            100.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.4, 0.4, 2, "+3.3V", false),
                Pad::new("2", 0.5, 0.0, 0.4, 0.4, 1, "GND", false),
            ],
        );

        let mut schematic_positions = HashMap::new();
        schematic_positions.insert("U1".to_string(), (10.0, 10.0));
        schematic_positions.insert("C1".to_string(), (12.0, 10.0)); // Close to U1
        schematic_positions.insert("U2".to_string(), (100.0, 10.0));
        schematic_positions.insert("C2".to_string(), (102.0, 10.0)); // Close to U2

        let board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 50.0,
            min_y: 50.0,
            max_x: 220.0,
            max_y: 150.0,
            footprints: vec![u1, u2, c1, c2],
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
            schematic_positions,
            outline_polygon: None,
            pin_swap_history: Vec::new(),
        };

        let ties = build_passive_ties(&board);
        assert_eq!(ties.len(), 2, "Both C1 and C2 must be tied as decoupling capacitors");

        let tie_c1 = ties.iter().find(|t| t.passive_idx == 2).expect("C1 tie must exist");
        assert!(tie_c1.is_decoupling);
        assert_eq!(tie_c1.anchor_idx, 0, "C1 must bind to U1 based on schematic proximity");
        assert_eq!(tie_c1.anchor_pad_idx, 0, "C1 must bind directly to U1's +3.3V power pad (pad 1)");
        assert_eq!(tie_c1.passive_pad_idx, 0, "C1's power pad (+3.3V) must be the passive pad");
        assert_eq!(tie_c1.max_dist, 2.0);

        let tie_c2 = ties.iter().find(|t| t.passive_idx == 3).expect("C2 tie must exist");
        assert!(tie_c2.is_decoupling);
        assert_eq!(tie_c2.anchor_idx, 1, "C2 must bind to U2 based on schematic proximity");
        assert_eq!(tie_c2.anchor_pad_idx, 0, "C2 must bind directly to U2's +3.3V power pad (pad 1)");
        assert_eq!(tie_c2.passive_pad_idx, 0, "C2's power pad (+3.3V) must be the passive pad");
    }

    #[test]
    fn test_decoupling_cap_multi_power_pin_distribution() {
        let nets = vec![
            "".to_string(),
            "GND".to_string(),
            "+3.3V".to_string(),
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

        // Active IC U1 with TWO separate +3.3V power pins: pad 1 and pad 4
        let u1 = make_test_fp(
            "U1",
            100.0,
            100.0,
            true,
            vec![
                Pad::new("1", -2.0, 0.0, 0.5, 0.5, 2, "+3.3V", false),
                Pad::new("2", 0.0, -2.0, 0.5, 0.5, 1, "GND", false),
                Pad::new("3", 0.0, 2.0, 0.5, 0.5, 1, "GND", false),
                Pad::new("4", 2.0, 0.0, 0.5, 0.5, 2, "+3.3V", false),
            ],
        );

        let c1 = make_test_fp(
            "C1",
            97.0,
            100.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.4, 0.4, 2, "+3.3V", false),
                Pad::new("2", 0.5, 0.0, 0.4, 0.4, 1, "GND", false),
            ],
        );

        let c2 = make_test_fp(
            "C2",
            103.0,
            100.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.4, 0.4, 2, "+3.3V", false),
                Pad::new("2", 0.5, 0.0, 0.4, 0.4, 1, "GND", false),
            ],
        );

        let board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 50.0,
            min_y: 50.0,
            max_x: 150.0,
            max_y: 150.0,
            footprints: vec![u1, c1, c2],
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
        };

        let ties = build_passive_ties(&board);
        assert_eq!(ties.len(), 2);
        // Both caps must bind to U1 (index 0)
        assert_eq!(ties[0].anchor_idx, 0);
        assert_eq!(ties[1].anchor_idx, 0);
        // The two caps should be distributed across both power pads (pad 0 and pad 3)
        let anchor_pads: HashSet<usize> = ties.iter().map(|t| t.anchor_pad_idx).collect();
        assert_eq!(anchor_pads.len(), 2, "Decoupling caps must distribute across both power pins of U1");
        assert!(anchor_pads.contains(&0) && anchor_pads.contains(&3));
    }

    #[test]
    fn test_decoupling_cap_affinity_cost_steepness() {
        let nets = vec![
            "".to_string(),
            "GND".to_string(),
            "+3.3V".to_string(),
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

        let u1 = make_test_fp(
            "U1",
            100.0,
            100.0,
            true,
            vec![
                Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "+3.3V", false),
                Pad::new("2", 1.0, 0.0, 0.5, 0.5, 1, "GND", false),
            ],
        );

        let c1 = make_test_fp(
            "C1",
            102.5,
            100.0,
            false,
            vec![
                Pad::new("1", -0.8, 0.0, 0.4, 0.4, 2, "+3.3V", false),
                Pad::new("2", 0.8, 0.0, 0.4, 0.4, 1, "GND", false),
            ],
        );

        let board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 50.0,
            min_y: 50.0,
            max_x: 150.0,
            max_y: 150.0,
            footprints: vec![u1, c1],
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
        };

        let cfg = PlacementConfig::default();
        let ties = build_passive_ties(&board);
        assert_eq!(ties.len(), 1);
        assert!(ties[0].is_decoupling);

        let mut st_close = PlacementState::from_board(&board, &cfg);
        // Position C1 so pad 1 is at 102.5 - 0.8 = 101.7 (dist = 1.7mm <= decoupling_max_dist 2.0mm, no courtyard overlap)
        st_close.positions[1] = Point::new(102.5, 100.0);
        st_close.update_component(1, &board, &cfg);
        st_close.evaluate(&board, &cfg, &ties);
        assert_eq!(st_close.overlaps, 0);
        let close_cost = st_close.cost;

        let mut st_far = PlacementState::from_board(&board, &cfg);
        // Position C1 so pad 1 is at 107.0 - 0.8 = 106.2 (dist = 6.2mm > decoupling_max_dist 2.0mm)
        st_far.positions[1] = Point::new(107.0, 100.0);
        st_far.update_component(1, &board, &cfg);
        st_far.evaluate(&board, &cfg, &ties);
        assert_eq!(st_far.overlaps, 0);
        let far_cost = st_far.cost;

        assert!(
            far_cost > close_cost + 150.0,
            "Drifting away from power pin must incur strong penalty: far_cost {} vs close_cost {}",
            far_cost, close_cost
        );
    }

    #[test]
    fn test_p1p2esp_decoupling_caps_binding_if_present() {
        let pcb_path = "/home/deece/src/p1p2esp/p1p2esp.kicad_pcb";
        if !std::path::Path::new(pcb_path).exists() {
            return;
        }

        let mut board = Board::load(pcb_path, &HashSet::new()).expect("Failed to load p1p2esp PCB");
        let cfg = PlacementConfig::default();
        prioritize_and_position_initial_components(&mut board, &cfg, None);
        let initial = PlacementState::from_board(&board, &cfg);
        let legalized = legalize_placement(&board, &cfg, initial, None);
        assert_eq!(legalized.overlaps, 0, "p1p2esp placement must be completely legalized with 0 courtyard overlaps");
        assert_eq!(legalized.boundary_violations, 0, "p1p2esp placement must have 0 boundary violations");
        let ties = build_passive_ties(&board);

        let fp_index = |ref_des: &str| -> usize {
            board.footprints.iter().position(|f| f.reference == ref_des).unwrap()
        };

        let u1_idx = fp_index("U1");
        let u2_idx = fp_index("U2");
        let u4_idx = fp_index("U4");

        // Verify C14 and C15 (ESP32-C3 +3.3V decoupling) bind directly to U4's +3.3V pin
        let c14_idx = fp_index("C14");
        let c15_idx = fp_index("C15");
        let tie_c14 = ties.iter().find(|t| t.passive_idx == c14_idx).expect("C14 must have tie");
        let tie_c15 = ties.iter().find(|t| t.passive_idx == c15_idx).expect("C15 must have tie");
        assert!(tie_c14.is_decoupling && tie_c14.anchor_idx == u4_idx, "C14 must bind to U4");
        assert!(tie_c15.is_decoupling && tie_c15.anchor_idx == u4_idx, "C15 must bind to U4");
        assert_eq!(board.footprints[u4_idx].pads[tie_c14.anchor_pad_idx].net_name, "+3.3V");
        assert_eq!(board.footprints[u4_idx].pads[tie_c15.anchor_pad_idx].net_name, "+3.3V");

        // Verify C5 and C6 (MAX22088 power caps) bind to U1 power pins
        let c5_idx = fp_index("C5");
        let c6_idx = fp_index("C6");
        let tie_c5 = ties.iter().find(|t| t.passive_idx == c5_idx).expect("C5 must have tie");
        let tie_c6 = ties.iter().find(|t| t.passive_idx == c6_idx).expect("C6 must have tie");
        assert!(tie_c5.is_decoupling && tie_c5.anchor_idx == u1_idx, "C5 must bind to U1");
        assert!(tie_c6.is_decoupling && tie_c6.anchor_idx == u1_idx, "C6 must bind to U1");
        assert_eq!(board.footprints[u1_idx].pads[tie_c5.anchor_pad_idx].net_name, "VCC_5V");
        assert_eq!(board.footprints[u1_idx].pads[tie_c6.anchor_pad_idx].net_name, "VRAW");

        // Verify C7 (Buck regulator input decoupling cap on /BUCK_VIN) binds to U2
        let c7_idx = fp_index("C7");
        let tie_c7 = ties.iter().find(|t| t.passive_idx == c7_idx).expect("C7 must have tie");
        assert!(tie_c7.is_decoupling && tie_c7.anchor_idx == u2_idx, "C7 must bind to U2");
        assert_eq!(board.footprints[u2_idx].pads[tie_c7.anchor_pad_idx].net_name, "/BUCK_VIN");
    }

    #[test]
    fn test_prioritize_and_position_diff_pairs_and_decoupling_caps() {
        let pcb_path = "/tmp/vulcan_test_priority_placement.kicad_pcb";
        let content = r#"(kicad_pcb
          (net 0 "") (net 1 "GND") (net 2 "+3.3V") (net 3 "USB_D+") (net 4 "USB_D-")
          (gr_rect (start 100.0 100.0) (end 160.0 140.0) (layer "Edge.Cuts"))
          (footprint "U" (at 120.0 120.0) (property "Reference" "U1") (locked yes)
            (fp_line (start -3.0 -3.0) (end 3.0 -3.0) (layer "F.CrtYd"))
            (fp_line (start 3.0 -3.0) (end 3.0 3.0) (layer "F.CrtYd"))
            (fp_line (start 3.0 3.0) (end -3.0 3.0) (layer "F.CrtYd"))
            (fp_line (start -3.0 3.0) (end -3.0 -3.0) (layer "F.CrtYd"))
            (pad "1" smd rect (at 0.0 -2.5) (size 0.5 0.5) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.0 2.5) (size 0.5 0.5) (net 1 "GND")))
          (footprint "C" (at 150.0 135.0) (property "Reference" "C1")
            (fp_line (start -0.8 -0.4) (end 0.8 -0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 -0.4) (end 0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 0.4) (end -0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start -0.8 0.4) (end -0.8 -0.4) (layer "F.CrtYd"))
            (pad "1" smd rect (at -0.5 0.0) (size 0.4 0.4) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.5 0.0) (size 0.4 0.4) (net 1 "GND")))
          (footprint "J" (at 105.0 120.0) (property "Reference" "J1") (locked yes)
            (pad "1" smd rect (at 0.0 -0.5) (size 0.4 0.4) (net 3 "USB_D+"))
            (pad "2" smd rect (at 0.0 0.5) (size 0.4 0.4) (net 4 "USB_D-")))
          (footprint "U" (at 155.0 135.0) (property "Reference" "U2")
            (pad "1" smd rect (at 0.0 -0.5) (size 0.4 0.4) (net 3 "USB_D+"))
            (pad "2" smd rect (at 0.0 0.5) (size 0.4 0.4) (net 4 "USB_D-")))
        )"#;
        fs::write(pcb_path, content).unwrap();

        let mut board = Board::load(pcb_path, &HashSet::new()).unwrap();
        let cfg = PlacementConfig {
            grid_step_mm: 0.1,
            ..PlacementConfig::default()
        };

        assert_eq!(board.footprints[1].x, 150.0);

        prioritize_and_position_initial_components(&mut board, &cfg, None);

        let u1_pin1_x = 120.0;
        let u1_pin1_y = 117.5;
        let c1_pin1_pos = board.footprints[1].pad_absolute_position(&board.footprints[1].pads[0]);
        let dist = ((c1_pin1_pos.0 - u1_pin1_x).powi(2) + (c1_pin1_pos.1 - u1_pin1_y).powi(2)).sqrt();
        assert!(dist < 2.0, "Decoupling cap power pin must be situated within 2.0 mm of IC power pin, got {:.2} mm", dist);
        assert!(board.footprints[3].x < 130.0, "Diff pair IC U2 should be moved near locked connector J1");

        let _ = fs::remove_file(pcb_path);
    }

    #[test]
    fn test_find_most_favourable_placement_and_holding_palette() {
        let pcb_path = "/tmp/vulcan_fav_placement_test.kicad_pcb";
        let content = r#"(kicad_pcb (version 20240108)
          (net 0 "") (net 1 "GND") (net 2 "+3.3V") (net 3 "SIG_NET")
          (gr_rect (start 100.0 100.0) (end 160.0 140.0) (layer "Edge.Cuts"))
          (footprint "U" (at 120.0 120.0) (property "Reference" "U1") (locked yes)
            (fp_line (start -3.0 -3.0) (end 3.0 -3.0) (layer "F.CrtYd"))
            (fp_line (start 3.0 -3.0) (end 3.0 3.0) (layer "F.CrtYd"))
            (fp_line (start 3.0 3.0) (end -3.0 3.0) (layer "F.CrtYd"))
            (fp_line (start -3.0 3.0) (end -3.0 -3.0) (layer "F.CrtYd"))
            (pad "1" smd rect (at 0.0 -2.5) (size 0.5 0.5) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.0 2.5) (size 0.5 0.5) (net 1 "GND"))
            (pad "3" smd rect (at 2.5 0.0) (size 0.5 0.5) (net 3 "SIG_NET")))
          (footprint "C" (at 150.0 135.0) (property "Reference" "C1")
            (fp_line (start -0.8 -0.4) (end 0.8 -0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 -0.4) (end 0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 0.4) (end -0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start -0.8 0.4) (end -0.8 -0.4) (layer "F.CrtYd"))
            (pad "1" smd rect (at -0.5 0.0) (size 0.4 0.4) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.5 0.0) (size 0.4 0.4) (net 1 "GND")))
          (footprint "R" (at 155.0 135.0) (property "Reference" "R12")
            (fp_line (start -0.8 -0.4) (end 0.8 -0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 -0.4) (end 0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 0.4) (end -0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start -0.8 0.4) (end -0.8 -0.4) (layer "F.CrtYd"))
            (pad "1" smd rect (at -0.5 0.0) (size 0.4 0.4) (net 3 "SIG_NET"))
            (pad "2" smd rect (at 0.5 0.0) (size 0.4 0.4) (net 1 "GND")))
        )"#;
        fs::write(pcb_path, content).unwrap();

        let mut board = Board::load(pcb_path, &HashSet::new()).unwrap();
        let cfg = PlacementConfig {
            grid_step_mm: 0.1,
            ..PlacementConfig::default()
        };

        // Clear unlocked objects (C1 and R12 move to holding palette)
        let (_ripped, cleared) = board.clear_unlocked_objects();
        assert_eq!(cleared, 2);
        assert_eq!(board.holding_palette.len(), 2);
        assert_eq!(board.footprints.len(), 1); // Only U1 on board

        // Place C1 individually at best location
        let placed_c1 = place_holding_component_at_best_location(&mut board, 0, &cfg).unwrap();
        assert_eq!(placed_c1.reference, "C1");
        assert_eq!(board.footprints.len(), 2);
        assert_eq!(board.holding_palette.len(), 1);

        // Verify C1 was placed at most favourable location near U1 pin 1 (120.0, 117.5), NOT at (150.0, 135.0)
        let orig_dist = (placed_c1.x - 150.0).hypot(placed_c1.y - 135.0);
        assert!(orig_dist > 5.0, "C1 must not be placed at its original cleared position");

        let u1_pin1_x = 120.0;
        let u1_pin1_y = 117.5;
        let c1_pin1 = placed_c1.pad_absolute_position(&placed_c1.pads[0]);
        let pin_dist = (c1_pin1.0 - u1_pin1_x).hypot(c1_pin1.1 - u1_pin1_y);
        assert!(pin_dist < 2.0, "C1 power pin must be within 2.0 mm of U1 power pin, got {:.2} mm", pin_dist);

        // Place all remaining components (R12)
        let placed_all = place_all_holding_components_at_best_locations(&mut board, &cfg);
        assert_eq!(placed_all, 1);
        assert_eq!(board.holding_palette.len(), 0);
        assert_eq!(board.footprints.len(), 3);

        // Verify 0 courtyard collisions among all 3 placed components
        let st = PlacementState::from_board(&board, &cfg);
        assert_eq!(st.overlaps, 0, "All placed components must have 0 courtyard collisions");

        let _ = fs::remove_file(pcb_path);
    }

    #[test]
    fn test_decoupling_caps_c14_c15_esp32_t_shape_notch_affinity() {
        let pcb_path = "/tmp/vulcan_esp32_t_shape_caps_test.kicad_pcb";
        let content = r#"(kicad_pcb (version 20240108)
          (net 0 "") (net 1 "GND") (net 2 "+3.3V")
          (gr_rect (start 90.0 80.0) (end 190.0 140.0) (layer "Edge.Cuts"))
          (footprint "Buck_IC" (at 110.0 110.0) (property "Reference" "U1") (locked yes)
            (fp_line (start -2.0 -2.0) (end 2.0 -2.0) (layer "F.CrtYd"))
            (fp_line (start 2.0 -2.0) (end 2.0 2.0) (layer "F.CrtYd"))
            (fp_line (start 2.0 2.0) (end -2.0 2.0) (layer "F.CrtYd"))
            (fp_line (start -2.0 2.0) (end -2.0 -2.0) (layer "F.CrtYd"))
            (pad "1" smd rect (at 0.0 -1.5) (size 0.5 0.5) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.0 1.5) (size 0.5 0.5) (net 1 "GND")))
          (footprint "ESP32-C3-WROOM-02" (at 165.0 105.0 -90) (property "Reference" "U4") (locked yes)
            (fp_line (start -9.75 7.15) (end 9.75 7.15) (layer "F.CrtYd"))
            (fp_line (start -9.75 7.15) (end -9.75 -6.85) (layer "F.CrtYd"))
            (fp_line (start -14.25 -6.85) (end -9.75 -6.85) (layer "F.CrtYd"))
            (fp_line (start 9.75 -6.85) (end 9.75 7.15) (layer "F.CrtYd"))
            (fp_line (start 9.75 -6.85) (end 14.25 -6.85) (layer "F.CrtYd"))
            (fp_line (start -14.25 -18.35) (end -14.25 -6.85) (layer "F.CrtYd"))
            (fp_line (start -14.25 -18.35) (end 14.25 -18.35) (layer "F.CrtYd"))
            (fp_line (start 14.25 -18.35) (end 14.25 -6.85) (layer "F.CrtYd"))
            (pad "1" smd rect (at -8.75 6.0) (size 0.9 1.5) (net 2 "+3.3V"))
            (pad "9" smd rect (at 8.75 6.0) (size 0.9 1.5) (net 1 "GND")))
          (footprint "C_0603" (at 100.0 90.0) (property "Reference" "C14")
            (fp_line (start -0.8 -0.4) (end 0.8 -0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 -0.4) (end 0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 0.4) (end -0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start -0.8 0.4) (end -0.8 -0.4) (layer "F.CrtYd"))
            (pad "1" smd rect (at -0.5 0.0) (size 0.4 0.4) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.5 0.0) (size 0.4 0.4) (net 1 "GND")))
          (footprint "C_0603" (at 105.0 90.0) (property "Reference" "C15")
            (fp_line (start -0.8 -0.4) (end 0.8 -0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 -0.4) (end 0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 0.4) (end -0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start -0.8 0.4) (end -0.8 -0.4) (layer "F.CrtYd"))
            (pad "1" smd rect (at -0.5 0.0) (size 0.4 0.4) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.5 0.0) (size 0.4 0.4) (net 1 "GND")))
        )"#;
        fs::write(pcb_path, content).unwrap();

        let mut board = Board::load(pcb_path, &HashSet::new()).unwrap();
        // Set schematic cluster & positions
        board.clusters = vec![
            CompiledCluster {
                name: "ESP32 & CONTROLS".to_string(),
                anchor_idx: Some(1),
                member_indices: vec![1, 2, 3],
                member_refs: vec!["U4".to_string(), "C14".to_string(), "C15".to_string()],
                max_radius: 15.0,
                weight: 100.0,
            },
        ];
        board.schematic_positions.insert("U1".to_string(), (50.0, 50.0));
        board.schematic_positions.insert("U4".to_string(), (320.0, 160.0));
        board.schematic_positions.insert("C14".to_string(), (305.0, 155.0));
        board.schematic_positions.insert("C15".to_string(), (305.0, 165.0));

        let cfg = PlacementConfig {
            grid_step_mm: 0.1,
            ..PlacementConfig::default()
        };

        // Clear unlocked objects (C14 and C15 moved to palette)
        let (_ripped, cleared) = board.clear_unlocked_objects();
        assert_eq!(cleared, 2);
        assert_eq!(board.holding_palette.len(), 2);
        assert_eq!(board.footprints.len(), 2); // U1 and U4

        // Place C14
        let placed_c14 = place_holding_component_at_best_location(&mut board, 0, &cfg).unwrap();
        assert_eq!(placed_c14.reference, "C14");

        // Place C15
        let placed_c15 = place_holding_component_at_best_location(&mut board, 0, &cfg).unwrap();
        assert_eq!(placed_c15.reference, "C15");

        // Compute U4 pin 1 absolute position
        let u4 = board.footprints.iter().find(|f| f.reference == "U4").unwrap();
        let (u4_p1_x, u4_p1_y) = u4.pad_absolute_position(&u4.pads[0]);

        // Verify C14 is placed right next to U4 pin 1, NOT near U1 (110.0, 110.0)
        let c14_to_u4_pin = (placed_c14.x - u4_p1_x).hypot(placed_c14.y - u4_p1_y);
        let c14_to_u1 = (placed_c14.x - 110.0).hypot(placed_c14.y - 110.0);
        assert!(c14_to_u4_pin < 3.5, "C14 should be placed adjacent to U4 pin 1, got {:.2} mm", c14_to_u4_pin);
        assert!(c14_to_u1 > 20.0, "C14 must not be placed near U1");

        // Verify C15 is also placed right next to U4 pin 1 / C14
        let c15_to_u4_pin = (placed_c15.x - u4_p1_x).hypot(placed_c15.y - u4_p1_y);
        assert!(c15_to_u4_pin < 4.0, "C15 should be placed adjacent to U4 pin 1, got {:.2} mm", c15_to_u4_pin);

        // Verify zero courtyard collisions across all components!
        let st = PlacementState::from_board(&board, &cfg);
        assert_eq!(st.overlaps, 0, "All components including U4, C14, C15 must have 0 courtyard collisions");

        let _ = fs::remove_file(pcb_path);
    }

    #[test]
    fn test_pin_density_courtyard_expansion_heuristic() {
        // 1. Dense 24-QFN (e.g. MAX22088: 24 pins around 4x4mm, 0.50mm pitch)
        let mut qfn_pads = Vec::new();
        // 6 pads per side, 0.50mm pitch
        for i in 0..6 {
            let offset = -1.25 + (i as f64) * 0.50;
            qfn_pads.push(Pad::new(&format!("{}", i + 1), offset, -1.9, 0.25, 0.6, 1, "SIG", false));
            qfn_pads.push(Pad::new(&format!("{}", i + 7), 1.9, offset, 0.6, 0.25, 1, "SIG", false));
            qfn_pads.push(Pad::new(&format!("{}", i + 13), -offset, 1.9, 0.25, 0.6, 1, "SIG", false));
            qfn_pads.push(Pad::new(&format!("{}", i + 19), -1.9, -offset, 0.6, 0.25, 1, "SIG", false));
        }
        let qfn = Footprint {
            reference: "U1".to_string(),
            value: "MAX22088".to_string(),
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: qfn_pads,
            width: 4.0,
            height: 4.0,
            courtyards: vec![Polygon::new(vec![
                Point::new(-2.2, -2.2),
                Point::new(2.2, -2.2),
                Point::new(2.2, 2.2),
                Point::new(-2.2, 2.2),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };
        assert!((qfn.min_pad_pitch() - 0.50).abs() < 1e-4, "QFN pitch must be 0.50mm");
        assert!(qfn.pad_density() > 1.0, "QFN pad density must be > 1.0 pads/mm²");
        let qfn_escape = qfn.compute_escape_margin();
        assert_eq!(qfn_escape, 1.20, "Dense 0.50mm QFN must receive 1.20mm escape margin");
        assert_eq!(qfn.placement_spacing(0.25), 0.25 + 2.0 * 1.20);

        // 2. ESP32 RF module (31 pads, 0.55mm pitch, 18x25.5mm package)
        let mut esp_pads = Vec::new();
        for i in 0..31 {
            esp_pads.push(Pad::new(&format!("{}", i + 1), -8.75 + (i as f64) * 0.55, 6.0, 0.35, 1.0, 1, "SIG", false));
        }
        let esp32 = Footprint {
            reference: "U4".to_string(),
            value: "ESP32-C3-WROOM-02".to_string(),
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: esp_pads,
            width: 18.0,
            height: 25.5,
            courtyards: vec![Polygon::new(vec![
                Point::new(-9.0, -12.75),
                Point::new(9.0, -12.75),
                Point::new(9.0, 12.75),
                Point::new(-9.0, 12.75),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };
        assert!((esp32.min_pad_pitch() - 0.55).abs() < 1e-4);
        assert_eq!(esp32.compute_escape_margin(), 1.00, "ESP32 module must receive 1.00mm escape margin");

        // 3. SOT-23-6 BUCK IC (6 pads, 0.95mm pitch)
        let mut sot_pads = Vec::new();
        for i in 0..3 {
            sot_pads.push(Pad::new(&format!("{}", i + 1), -0.95 + (i as f64) * 0.95, -1.3, 0.45, 0.6, 1, "SIG", false));
            sot_pads.push(Pad::new(&format!("{}", i + 4), 0.95 - (i as f64) * 0.95, 1.3, 0.45, 0.6, 1, "SIG", false));
        }
        let sot23_6 = Footprint {
            reference: "U2".to_string(),
            value: "BUCK_IC".to_string(),
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: sot_pads,
            width: 3.0,
            height: 3.0,
            courtyards: vec![Polygon::new(vec![
                Point::new(-1.6, -1.6),
                Point::new(1.6, -1.6),
                Point::new(1.6, 1.6),
                Point::new(-1.6, 1.6),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };
        assert!((sot23_6.min_pad_pitch() - 0.95).abs() < 1e-4);
        assert_eq!(sot23_6.compute_escape_margin(), 0.75, "SOT-23-6 BUCK IC must receive 0.75mm escape margin");

        // 4. Standard 0603 passive (2 pads, 1.60mm pitch)
        let r0603 = Footprint {
            reference: "R1".to_string(),
            value: "10k".to_string(),
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                Pad::new("1", -0.8, 0.0, 0.6, 0.7, 1, "SIG", false),
                Pad::new("2", 0.8, 0.0, 0.6, 0.7, 1, "SIG", false),
            ],
            width: 2.2,
            height: 1.2,
            courtyards: vec![Polygon::new(vec![
                Point::new(-1.1, -0.6),
                Point::new(1.1, -0.6),
                Point::new(1.1, 0.6),
                Point::new(-1.1, 0.6),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };
        assert_eq!(r0603.compute_escape_margin(), 0.0, "0603 passive must receive 0.0mm escape margin (standard clearance)");
        assert_eq!(r0603.placement_spacing(0.25), 0.25);
    }

    #[test]
    fn test_escape_clearance_enforcement_in_placement_and_legalization() {
        let pcb_path = "/tmp/vulcan_escape_clearance_test.kicad_pcb";
        // Board with QFN-24 (locked at 100, 100) and an unlocked 0603 capacitor initially placed too close
        let content = r#"(kicad_pcb (version 20240108)
          (net 0 "") (net 1 "GND") (net 2 "+3.3V")
          (gr_rect (start 80.0 80.0) (end 130.0 130.0) (layer "Edge.Cuts"))
          (footprint "Package_DFN_QFN:QFN-24" (at 100.0 100.0) (property "Reference" "U1") (locked yes)
            (fp_rect (start -2.0 -2.0) (end 2.0 2.0) (layer "F.CrtYd"))
            (pad "1" smd rect (at -1.8 -1.25) (size 0.5 0.25) (net 2 "+3.3V"))
            (pad "2" smd rect (at -1.8 -0.75) (size 0.5 0.25) (net 1 "GND"))
            (pad "3" smd rect (at -1.8 -0.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "4" smd rect (at -1.8 0.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "5" smd rect (at -1.8 0.75) (size 0.5 0.25) (net 1 "GND"))
            (pad "6" smd rect (at -1.8 1.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "7" smd rect (at -1.25 1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "8" smd rect (at -0.75 1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "9" smd rect (at -0.25 1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "10" smd rect (at 0.25 1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "11" smd rect (at 0.75 1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "12" smd rect (at 1.25 1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "13" smd rect (at 1.8 1.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "14" smd rect (at 1.8 0.75) (size 0.5 0.25) (net 1 "GND"))
            (pad "15" smd rect (at 1.8 0.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "16" smd rect (at 1.8 -0.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "17" smd rect (at 1.8 -0.75) (size 0.5 0.25) (net 1 "GND"))
            (pad "18" smd rect (at 1.8 -1.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "19" smd rect (at 1.25 -1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "20" smd rect (at 0.75 -1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "21" smd rect (at 0.25 -1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "22" smd rect (at -0.25 -1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "23" smd rect (at -0.75 -1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "24" smd rect (at -1.25 -1.8) (size 0.25 0.5) (net 1 "GND")))
          (footprint "Capacitor_SMD:C_0603" (at 103.5 100.0) (property "Reference" "C1")
            (fp_rect (start -1.0 -0.6) (end 1.0 0.6) (layer "F.CrtYd"))
            (pad "1" smd rect (at -0.7 0.0) (size 0.6 0.6) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.7 0.0) (size 0.6 0.6) (net 1 "GND")))
        )"#;
        fs::write(pcb_path, content).unwrap();

        let mut board = Board::load(pcb_path, &HashSet::new()).unwrap();
        let cfg = PlacementConfig {
            grid_step_mm: 0.1,
            ..PlacementConfig::default()
        };

        // U1 has 24 pads with 0.50mm pitch -> 1.20mm escape margin
        let u1 = &board.footprints[0];
        assert_eq!(u1.escape_margin, 1.20);

        // At x = 103.5, C1 is 3.5mm from U1 center (2.0mm QFN half + 1.0mm C1 half = 3.0mm,
        // distance between raw courtyards is only 0.5mm, which violates the 1.20mm escape corridor!)
        let c1 = &board.footprints[1];
        assert!(!is_placement_legal(&board, c1, Point::new(103.5, 100.0), 0.0, 0, cfg.min_spacing_mm),
            "C1 at 103.5mm sits inside the 1.20mm escape halo of U1 and must be deemed illegal");

        // At x = 104.5, C1 is outside the 1.20mm escape halo and is legal
        assert!(is_placement_legal(&board, c1, Point::new(104.5, 100.0), 0.0, 0, cfg.min_spacing_mm),
            "C1 at 104.5mm sits outside the escape halo and must be legal");

        // Verify that legalization resolves initial overlap by pushing C1 outside the escape halo
        let initial_st = PlacementState::from_board(&board, &cfg);
        assert!(initial_st.overlaps > 0, "Initial placement must detect escape corridor overlap");
        let legalized_st = legalize_placement(&mut board, &cfg, initial_st, None);
        assert_eq!(legalized_st.overlaps, 0, "Legalizer must resolve escape corridor overlap");
        let c1_legal_x = legalized_st.positions[1].x;
        let c1_legal_y = legalized_st.positions[1].y;
        let dist_to_u1 = (c1_legal_x - 100.0).hypot(c1_legal_y - 100.0);
        assert!(dist_to_u1 >= 4.2, "Legalized C1 must maintain at least 4.2mm distance (2.0 + 1.0 + 1.2), got {:.2}mm", dist_to_u1);

        let _ = fs::remove_file(pcb_path);
    }

    #[test]
    fn test_board_outline_boundary_containment_and_legalization() {
        // Octagonal outline: [100, 150] x [100, 150] with 5mm corner chamfers
        let outline = Polygon::new(vec![
            Point::new(105.0, 100.0),
            Point::new(145.0, 100.0),
            Point::new(150.0, 105.0),
            Point::new(150.0, 145.0),
            Point::new(145.0, 150.0),
            Point::new(105.0, 150.0),
            Point::new(100.0, 145.0),
            Point::new(100.0, 105.0),
        ]);

        let j3 = Footprint {
            reference: "J3".to_string(),
            value: "HEADER".to_string(),
            x: 120.0,
            y: 145.0, // Pin 4 extends to 145 + 7.62 = 152.62 > max_y 150.0!
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                Pad::new("1", 0.0, 0.0, 1.4, 1.4, 1, "GND", false),
                Pad::new("2", 0.0, 2.54, 1.4, 1.4, 1, "GND", false),
                Pad::new("3", 0.0, 5.08, 1.4, 1.4, 1, "GND", false),
                Pad::new("4", 0.0, 7.62, 1.4, 1.4, 1, "GND", false),
            ],
            width: 2.5,
            height: 9.62,
            courtyards: vec![Polygon::new(vec![
                Point::new(-1.25, -1.0),
                Point::new(1.25, -1.0),
                Point::new(1.25, 8.62),
                Point::new(-1.25, 8.62),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        // Small 0603 passive in the chamfered corner: (101, 101) is inside AABB [100, 150], but outside chamfered contour
        let c1 = Footprint {
            reference: "C1".to_string(),
            value: "100nF".to_string(),
            x: 101.5,
            y: 101.5,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                Pad::new("1", -0.7, 0.0, 0.6, 0.6, 1, "GND", false),
                Pad::new("2", 0.7, 0.0, 0.6, 0.6, 2, "VCC", false),
            ],
            width: 2.0,
            height: 1.2,
            courtyards: vec![Polygon::new(vec![
                Point::new(-1.0, -0.6),
                Point::new(1.0, -0.6),
                Point::new(1.0, 0.6),
                Point::new(-1.0, 0.6),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let nets = vec!["".to_string(), "GND".to_string(), "VCC".to_string()];
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

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 100.0,
            min_y: 100.0,
            max_x: 150.0,
            max_y: 150.0,
            footprints: vec![j3.clone(), c1.clone()],
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
            outline_polygon: Some(outline),
            pin_swap_history: Vec::new(),
        };

        let cfg = PlacementConfig {
            grid_step_mm: 0.25,
            min_spacing_mm: 0.25,
            duration_secs: 1,
            ..PlacementConfig::default()
        };

        // 1. Verify that J3 at (120, 145) is rejected because pin 4 exceeds the boundary
        assert!(
            !is_placement_legal(&board, &j3, Point::new(120.0, 145.0), 0.0, 0, cfg.min_spacing_mm),
            "J3 with pin 4 protruding past boundary must be illegal"
        );

        // 2. Verify that C1 at (101.5, 101.5) is rejected because it is inside the chamfer corner
        assert!(
            !is_placement_legal(&board, &c1, Point::new(101.5, 101.5), 0.0, 0, cfg.min_spacing_mm),
            "C1 inside corner chamfer cutout must be illegal"
        );

        // 3. Verify that initial state has boundary violations
        let init_st = PlacementState::from_board(&board, &cfg);
        assert!(
            init_st.boundary_violations > 0,
            "Initial placement must register boundary violations"
        );

        // 4. Legalize placement and verify that all boundary violations are eliminated
        let legalized = legalize_placement(&mut board, &cfg, init_st, None);
        assert_eq!(
            legalized.boundary_violations, 0,
            "Legalizer must eliminate all boundary violations"
        );
        assert_eq!(legalized.overlaps, 0, "Legalizer must eliminate overlaps");

        // Verify every pad corner of both components is inside the board outline polygon
        legalized.apply_to_board(&mut board);
        for fp in &board.footprints {
            let pos = Point::new(fp.x, fp.y);
            assert!(board.contains_point_in_boundary(&pos), "Footprint origin {} must be inside", fp.reference);
            for pad in &fp.pads {
                for corner in fp.pad_corners_at(pad, pos, fp.rotation, fp.layer) {
                    assert!(
                        board.contains_point_in_boundary(&corner),
                        "Pad {} corner ({:.2}, {:.2}) of {} must be inside board outline",
                        pad.number, corner.x, corner.y, fp.reference
                    );
                }
            }
        }
    }

    #[test]
    fn test_aggressive_placement_moves_and_centroid_pull() {
        use std::sync::atomic::AtomicBool;

        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 60.0;
        board.max_y = 60.0;
        board.outline_polygon = Some(Polygon::from_rect(0.0, 0.0, 60.0, 60.0));

        let net_id = 1;
        board.nets.insert(net_id, "VCC_NET".to_string());

        // J1: Locked anchor at (45, 45)
        let j1 = Footprint {
            reference: "J1".to_string(),
            value: "CONN".to_string(),
            x: 45.0,
            y: 45.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: true,
            pads: vec![crate::board::Pad {
                number: "1".to_string(),
                net_id,
                net_name: "VCC_NET".to_string(),
                offset_x: 0.0,
                offset_y: 0.0,
                width: 1.0,
                height: 1.0,
                shape: "rect".to_string(),
                rotation: 0.0,
                is_through_hole: false,
                drill: None,
                ..Default::default()
            }],
            width: 4.0,
            height: 4.0,
            courtyards: vec![Polygon::from_rect(43.0, 43.0, 4.0, 4.0)],
            perimeters: vec![],
            escape_margin: 0.25,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        // U1: Unlocked IC initially at (10, 10)
        let u1 = Footprint {
            reference: "U1".to_string(),
            value: "MCU".to_string(),
            x: 10.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![crate::board::Pad {
                number: "1".to_string(),
                net_id,
                net_name: "VCC_NET".to_string(),
                offset_x: 0.0,
                offset_y: 0.0,
                width: 1.0,
                height: 1.0,
                shape: "rect".to_string(),
                rotation: 0.0,
                is_through_hole: false,
                drill: None,
                ..Default::default()
            }],
            width: 5.0,
            height: 5.0,
            courtyards: vec![Polygon::from_rect(7.5, 7.5, 5.0, 5.0)],
            perimeters: vec![],
            escape_margin: 0.25,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        // C1: Unlocked capacitor initially at (15, 10)
        let c1 = Footprint {
            reference: "C1".to_string(),
            value: "100nF".to_string(),
            x: 15.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![crate::board::Pad {
                number: "1".to_string(),
                net_id,
                net_name: "VCC_NET".to_string(),
                offset_x: 0.0,
                offset_y: 0.0,
                width: 0.6,
                height: 0.6,
                shape: "rect".to_string(),
                rotation: 0.0,
                is_through_hole: false,
                drill: None,
                ..Default::default()
            }],
            width: 2.0,
            height: 1.5,
            courtyards: vec![Polygon::from_rect(14.0, 9.25, 2.0, 1.5)],
            perimeters: vec![],
            escape_margin: 0.25,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        board.footprints = vec![j1, u1, c1];

        let cfg = PlacementConfig {
            duration_secs: 1,
            grid_step_mm: 0.5,
            min_spacing_mm: 0.25,
            max_jump_radius_mm: 25.0,
            pair_swap_prob: 0.25,
            centroid_pull_prob: 0.35,
            temp_scale: 1.5,
            ..PlacementConfig::default()
        };

        let stop = Arc::new(AtomicBool::new(false));
        let result = run_auto_placer(&mut board, &cfg, None, stop);

        assert_eq!(result.boundary_violations, 0, "All components must be within board outline");
        assert_eq!(result.overlaps, 0, "No courtyard overlaps allowed");

        // Verify locked component stayed fixed
        let j1_after = board.footprints.iter().find(|f| f.reference == "J1").unwrap();
        assert!((j1_after.x - 45.0).abs() < 1e-6);
        assert!((j1_after.y - 45.0).abs() < 1e-6);

        // Verify all components are legally contained
        for fp in &board.footprints {
            let p = Point::new(fp.x, fp.y);
            assert!(board.contains_point_in_boundary(&p), "Footprint {} must be within boundary", fp.reference);
        }
    }

    #[test]
    fn test_esp32_footprint_and_keepout_exclusion() {
        let mut board = make_synthetic_board();
        board.clusters.clear();
        board.min_x = 100.0;
        board.min_y = 80.0;
        board.max_x = 190.0;
        board.max_y = 130.0;

        let esp32_t_poly = Polygon::new(vec![
            Point::new(-9.75, 7.15),
            Point::new(9.75, 7.15),
            Point::new(9.75, -6.85),
            Point::new(14.25, -6.85),
            Point::new(14.25, -18.35),
            Point::new(-14.25, -18.35),
            Point::new(-14.25, -6.85),
            Point::new(-9.75, -6.85),
        ]);
        let esp32_courtyards = esp32_t_poly.decompose_convex();

        let u4 = Footprint {
            reference: "U4".to_string(),
            value: "ESP32-C3-WROOM-02".to_string(),
            x: 165.0,
            y: 105.0,
            rotation: -90.0,
            layer: 0,
            orig_layer: 0,
            is_locked: true,
            pads: vec![
                Pad::new("1", -9.0, 6.7, 0.4, 0.8, 1, "+3.3V", false),
                Pad::new("19", 0.0, 0.0, 3.0, 3.0, 0, "GND", false),
            ],
            width: 18.0,
            height: 20.0,
            courtyards: esp32_courtyards,
            perimeters: Vec::new(),
            escape_margin: 0.70,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let mut c1 = make_test_fp(
            "C1",
            165.0,
            105.0, // directly on ESP32 center
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "+3.3V", false)],
        );

        board.footprints = vec![u4, c1.clone()];
        board.keepouts = vec![crate::board::KeepoutZone {
            min_x: 172.1,
            min_y: 91.0,
            max_x: 183.1,
            max_y: 119.0,
            layers: vec![0, 1],
        }];

        let cfg = PlacementConfig {
            duration_secs: 1,
            grid_step_mm: 0.1,
            min_spacing_mm: 0.25,
            ..PlacementConfig::default()
        };

        // 1. Placing C1 directly on ESP32 module body causes overlap
        let st_body = PlacementState::from_board(&board, &cfg);
        assert!(st_body.overlaps > 0, "Placing on ESP32 body must cause courtyard overlap");

        // 2. Placing C1 inside ESP32 antenna keepout zone causes boundary/keepout violation
        c1.x = 175.0;
        c1.y = 105.0;
        board.footprints[1] = c1.clone();
        let st_ko = PlacementState::from_board(&board, &cfg);
        assert!(st_ko.boundary_violations > 0, "Placing in antenna keepout must cause keepout violation");

        // 3. Placing C1 in the legitimate concave notch outside the module body has 0 overlap
        c1.x = 160.0;
        c1.y = 93.0;
        board.footprints[1] = c1;
        let st_notch = PlacementState::from_board(&board, &cfg);
        assert_eq!(st_notch.boundary_violations, 0, "Notch outside module body must not violate keepout");

        // 4. Auto-placer resolves any initial overlap and keeps C1 outside module body and keepout
        let stop = Arc::new(AtomicBool::new(false));
        let solved = run_auto_placer(&mut board, &cfg, None, stop);
        assert_eq!(solved.overlaps, 0, "Auto-placer must achieve 0 overlaps");
        assert_eq!(solved.boundary_violations, 0, "Auto-placer must achieve 0 keepout violations");

        let c1_final = board.footprints.iter().find(|f| f.reference == "C1").unwrap();
        // Verify C1 is not in the antenna keepout
        let in_ko = c1_final.x >= 172.1 && c1_final.x <= 183.1 && c1_final.y >= 91.0 && c1_final.y <= 119.0;
        assert!(!in_ko, "Final placement must not place component in antenna keepout");
    }

    #[test]
    fn test_metropolis_replica_exchange_accepts_uphill_and_downhill() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let board = make_synthetic_board();
        let cfg = PlacementConfig::default();
        let base_st = PlacementState::from_board(&board, &cfg);

        // Case 1: Downhill move (Hotter replica has LOWER cost than colder replica)
        // Must swap 100% of the time (P = 1.0)
        let mut st0 = base_st.clone();
        st0.cost = 500.0;
        let mut st1 = base_st.clone();
        st1.cost = 300.0;

        let mut replicas = vec![st0, st1];
        let mut rng = StdRng::seed_from_u64(42);
        let swaps = metropolis_replica_exchange(
            &mut replicas,
            2,
            0, // epoch 0 -> starts with pair (0, 1)
            0.5,
            1.0,
            1.0,
            &mut rng,
        );
        assert_eq!(swaps, 1, "Downhill swap where hotter replica is better must always be accepted");
        assert_eq!(replicas[0].cost, 300.0, "Colder replica must receive the better configuration");
        assert_eq!(replicas[1].cost, 500.0, "Hotter replica receives the prior state to continue exploring");

        // Case 2: Uphill move (Hotter replica has HIGHER cost than colder replica)
        // At low temperature (cooling = 0.05, thermal_boost = 1.0), large uphill moves (delta = 500.0) should rarely or never swap.
        // At high temperature (thermal_boost = 10.0), uphill moves have significantly higher probability of acceptance.
        let mut cold_swaps = 0;
        let mut hot_swaps = 0;
        for seed in 0..100 {
            let mut cold_reps = vec![
                { let mut s = base_st.clone(); s.cost = 100.0; s },
                { let mut s = base_st.clone(); s.cost = 300.0; s },
            ];
            let mut rng_cold = StdRng::seed_from_u64(seed);
            cold_swaps += metropolis_replica_exchange(&mut cold_reps, 2, 0, 0.95, 1.0, 1.0, &mut rng_cold);

            let mut hot_reps = vec![
                { let mut s = base_st.clone(); s.cost = 100.0; s },
                { let mut s = base_st.clone(); s.cost = 300.0; s },
            ];
            let mut rng_hot = StdRng::seed_from_u64(seed);
            hot_swaps += metropolis_replica_exchange(&mut hot_reps, 2, 0, 0.0, 1.0, 5.0, &mut rng_hot);
        }
        assert!(hot_swaps > cold_swaps, "Elevated temperature must accept significantly more uphill replica swaps (hot={}, cold={})", hot_swaps, cold_swaps);
    }

    #[test]
    fn test_build_component_clusters_and_rigid_group_move() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 100.0;
        board.max_y = 100.0;
        let boundary = board.boundary_aabb();

        let u1 = make_test_fp("U1", 50.0, 50.0, false, vec![
            Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "VCC", false),
        ]);
        let c1 = make_test_fp("C1", 48.0, 50.0, false, vec![
            Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "VCC", false),
        ]);
        let c2 = make_test_fp("C2", 52.0, 50.0, false, vec![
            Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "VCC", false),
        ]);
        board.footprints = vec![u1, c1, c2];

        let ties = vec![
            PassiveTie {
                anchor_idx: 0,
                anchor_pad_idx: 0,
                passive_idx: 1,
                passive_pad_idx: 0,
                max_dist: 5.0,
                weight_mult: 2.0,
                is_decoupling: true,
            },
            PassiveTie {
                anchor_idx: 0,
                anchor_pad_idx: 0,
                passive_idx: 2,
                passive_pad_idx: 0,
                max_dist: 5.0,
                weight_mult: 2.0,
                is_decoupling: true,
            },
        ];
        let movable = vec![0, 1, 2];
        let clusters = build_component_clusters(&board, &ties, &movable);

        // U1 cluster should contain U1, C1, and C2
        assert_eq!(clusters[0].len(), 3, "U1 cluster must contain anchor and both passives");
        assert!(clusters[0].contains(&0));
        assert!(clusters[0].contains(&1));
        assert!(clusters[0].contains(&2));
        // C1 and C2 should point to the same group
        assert_eq!(clusters[1], clusters[0]);
        assert_eq!(clusters[2], clusters[0]);

        let cfg = PlacementConfig { min_spacing_mm: 0.25, grid_step_mm: 0.1, ..Default::default() };
        let mut state = PlacementState::from_board(&board, &cfg);

        // Apply a legal rigid group move: dx = 10.0, dy = -5.0
        let success = apply_cluster_group_move(&mut state, &board, &cfg, &clusters[0], 10.0, -5.0, &boundary);
        assert!(success, "Cluster move within board bounds must succeed");
        assert!((state.positions[0].x - 60.0).abs() < 1e-6);
        assert!((state.positions[0].y - 45.0).abs() < 1e-6);
        assert!((state.positions[1].x - 58.0).abs() < 1e-6);
        assert!((state.positions[1].y - 45.0).abs() < 1e-6);
        assert!((state.positions[2].x - 62.0).abs() < 1e-6);
        assert!((state.positions[2].y - 45.0).abs() < 1e-6);

        // Relative distances between U1 and C1/C2 are preserved exactly
        let dist_u1_c1 = ((state.positions[0].x - state.positions[1].x).powi(2) + (state.positions[0].y - state.positions[1].y).powi(2)).sqrt();
        assert!((dist_u1_c1 - 2.0).abs() < 1e-6, "Relative decoupling distance must be strictly preserved");

        // Attempt out-of-bounds group move: dx = 60.0 (would push U1 to x=120 > max_x=100)
        let oob_success = apply_cluster_group_move(&mut state, &board, &cfg, &clusters[0], 60.0, 0.0, &boundary);
        assert!(!oob_success, "Out of bounds cluster move must be rejected");
        assert!((state.positions[0].x - 60.0).abs() < 1e-6, "Positions must remain unchanged when rejected");
    }
}

