// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use crate::board::{Board, LiveTrialState, Point, Polygon, TrialPhase, AABB};
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            duration_secs: 20,
            grid_step_mm: 0.635,
            min_spacing_mm: 0.25,
            crossing_weight: 200.0,
            wirelength_weight: 1.0,
            overlap_penalty: 25000.0,
            boundary_penalty: 50000.0,
            passive_affinity_weight: 80.0,
            passive_max_dist: 3.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlacementState {
    pub positions: Vec<Point>,
    pub rotations: Vec<f64>,
    pub cached_courtyards: Vec<Vec<Polygon>>,
    pub cached_bounds: Vec<AABB>,
    pub cached_pads: Vec<Vec<Point>>,
    pub cost: f64,
    pub crossings: usize,
    pub wirelength: f64,
    pub overlaps: usize,
}

#[derive(Debug, Clone, Copy)]
struct PassiveTie {
    passive_idx: usize,
    passive_pad_idx: usize,
    anchor_idx: usize,
    anchor_pad_idx: usize,
}

fn build_passive_ties(board: &Board) -> Vec<PassiveTie> {
    let mut net_pads: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (fp_idx, fp) in board.footprints.iter().enumerate() {
        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if pad.net_id == 0 || pad.net_name.is_empty() || pad.net_name == "GND" {
                continue;
            }
            net_pads.entry(pad.net_id).or_default().push((fp_idx, pad_idx));
        }
    }

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
        let mut best: Option<(usize, usize, usize, usize)> = None;
        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if matches!(
                pad.net_name.as_str(),
                "" | "GND" | "+3.3V" | "VRAW" | "VCC_5V" | "VBUS" | "/BUCK_VIN"
            ) {
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
        let mut cached_courtyards = Vec::with_capacity(n);
        let mut cached_bounds = Vec::with_capacity(n);
        let mut cached_pads = Vec::with_capacity(n);

        for fp in &board.footprints {
            let pos = Point::new(fp.x, fp.y);
            let rot = fp.rotation;
            positions.push(pos);
            rotations.push(rot);
            let polys = fp.transformed_courtyards(pos, rot, config.min_spacing_mm);
            cached_bounds.push(union_aabb(&polys, pos));
            cached_courtyards.push(polys);
            let pads = (0..fp.pads.len())
                .map(|pi| fp.pad_world_point(pi, pos, rot))
                .collect();
            cached_pads.push(pads);
        }

        let mut st = Self {
            positions,
            rotations,
            cached_courtyards,
            cached_bounds,
            cached_pads,
            cost: 0.0,
            crossings: 0,
            wirelength: 0.0,
            overlaps: 0,
        };
        let ties = build_passive_ties(board);
        st.evaluate(board, config, &ties);
        st
    }

    fn update_component(&mut self, idx: usize, board: &Board, config: &PlacementConfig) {
        let fp = &board.footprints[idx];
        let pos = self.positions[idx];
        let rot = self.rotations[idx];
        let polys = fp.transformed_courtyards(pos, rot, config.min_spacing_mm);
        self.cached_bounds[idx] = union_aabb(&polys, pos);
        self.cached_courtyards[idx] = polys;
        for pi in 0..fp.pads.len() {
            self.cached_pads[idx][pi] = fp.pad_world_point(pi, pos, rot);
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
        for i in 0..n {
            if board.footprints[i].is_locked {
                continue;
            }
            for poly in &self.cached_courtyards[i] {
                for v in &poly.vertices {
                    if !boundary.contains_point(v) {
                        let dx = (boundary.min_x - v.x).max(0.0) + (v.x - boundary.max_x).max(0.0);
                        let dy = (boundary.min_y - v.y).max(0.0) + (v.y - boundary.max_y).max(0.0);
                        boundary_cost += config.boundary_penalty * (1.0 + dx + dy);
                    }
                }
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
            for &m_idx in &cl.member_indices {
                if Some(m_idx) == cl.anchor_idx || board.footprints[m_idx].is_locked {
                    continue;
                }
                let d = self.positions[m_idx].distance_to(&center);
                if d > cl.max_radius {
                    let ex = d - cl.max_radius;
                    cluster_cost += cl.weight * ex * ex;
                } else {
                    cluster_cost += (cl.weight * 0.1) * d;
                }
            }
        }

        let mut affinity_cost = 0.0;
        for tie in ties {
            let p_pt = self.cached_pads[tie.passive_idx][tie.passive_pad_idx];
            let a_pt = self.cached_pads[tie.anchor_idx][tie.anchor_pad_idx];
            let d = p_pt.distance_to(&a_pt);
            if d > config.passive_max_dist {
                let ex = d - config.passive_max_dist;
                affinity_cost += config.passive_affinity_weight * (d + ex * ex * 2.5);
            } else {
                affinity_cost += config.passive_affinity_weight * 0.25 * d;
            }
        }

        self.wirelength = wirelength;
        self.crossings = crossings;
        self.overlaps = overlaps;
        self.cost = wirelength * config.wirelength_weight
            + (crossings as f64) * config.crossing_weight
            + overlap_area_cost
            + boundary_cost
            + cluster_cost
            + affinity_cost;
    }

    pub fn apply_to_board(&self, board: &mut Board) {
        for (i, fp) in board.footprints.iter_mut().enumerate() {
            if !fp.is_locked {
                fp.x = self.positions[i].x;
                fp.y = self.positions[i].y;
                fp.rotation = self.rotations[i];
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

pub fn legalize_placement(
    board: &Board,
    config: &PlacementConfig,
    mut initial: PlacementState,
    live: Option<&Arc<Mutex<LiveTrialState>>>,
) -> PlacementState {
    if initial.overlaps == 0 {
        return initial;
    }
    if let Some(st) = live {
        if let Ok(mut g) = st.lock() {
            g.phase = TrialPhase::Legalizing;
            g.push_log(format!(
                "[Legalizer] Resolving {} initial courtyard overlaps...",
                initial.overlaps
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
    let mut rng = StdRng::seed_from_u64(0x517CC1B727220A95);
    let mut best = initial.clone();

    for step in 0..4500 {
        if initial.overlaps == 0 {
            break;
        }
        let idx = movable[rng.gen_range(0..movable.len())];
        let old_pos = initial.positions[idx];
        let old_rot = initial.rotations[idx];
        let old_cost = initial.cost;
        let old_ov = initial.overlaps;

        if rng.gen_bool(0.75) {
            let r = if step < 1500 { 6.0 } else { 2.5 };
            let nx = snap_to_grid(
                (old_pos.x + rng.gen_range(-r..=r)).clamp(boundary.min_x + 1.5, boundary.max_x - 1.5),
                config.grid_step_mm,
            );
            let ny = snap_to_grid(
                (old_pos.y + rng.gen_range(-r..=r)).clamp(boundary.min_y + 1.5, boundary.max_y - 1.5),
                config.grid_step_mm,
            );
            initial.positions[idx] = Point::new(nx, ny);
        } else {
            let angles = [0.0, 90.0, 180.0, 270.0];
            initial.rotations[idx] = angles[rng.gen_range(0..4)];
        }
        initial.update_component(idx, board, config);
        initial.evaluate(board, config, &ties);

        if initial.overlaps < old_ov || (initial.overlaps == old_ov && initial.cost < old_cost) {
            if initial.overlaps < best.overlaps || (initial.overlaps == best.overlaps && initial.cost < best.cost) {
                best = initial.clone();
            }
        } else {
            initial.positions[idx] = old_pos;
            initial.rotations[idx] = old_rot;
            initial.update_component(idx, board, config);
            initial.cost = old_cost;
            initial.overlaps = old_ov;
        }
    }
    best
}

pub fn run_auto_placer(
    board: &mut Board,
    config: &PlacementConfig,
    live: Option<Arc<Mutex<LiveTrialState>>>,
    stop_flag: Arc<AtomicBool>,
) -> PlacementState {
    let initial = PlacementState::from_board(board, config);
    let legalized = legalize_placement(board, config, initial, live.as_ref());

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

    let num_threads = rayon::current_num_threads().max(2);
    let global_best = Arc::new(Mutex::new(legalized.clone()));
    let total_evals = Arc::new(AtomicUsize::new(0));
    let start_time = Instant::now();
    let total_dur = Duration::from_secs(config.duration_secs);
    let boundary = board.boundary_aabb();
    let ties = Arc::new(build_passive_ties(board));

    if let Some(ref st) = live {
        if let Ok(mut g) = st.lock() {
            g.phase = TrialPhase::AutoPlacing;
            g.push_log(format!(
                "[Auto-Placer] Launching {} Parallel Tempering SA replicas for {}s...",
                num_threads, config.duration_secs
            ));
        }
    }

    let num_epochs = (config.duration_secs * 2).max(1) as usize;
    let epoch_dur = Duration::from_millis(500);
    let mut replicas: Vec<PlacementState> = (0..num_threads).map(|_| legalized.clone()).collect();

    for epoch in 0..num_epochs {
        if stop_flag.load(Ordering::Relaxed) || start_time.elapsed() >= total_dur {
            break;
        }
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
                let t_base = 8.0 * (350.0f64 / 8.0).powf(temp_ratio);
                let cooling = (1.0 - progress * 0.85).max(0.05);
                let temp = t_base * cooling;
                let mut local_evals = 0usize;

                while Instant::now() < ep_end && !stop_flag.load(Ordering::Relaxed) {
                    for _ in 0..40 {
                        let old_positions = current.positions.clone();
                        let old_rotations = current.rotations.clone();
                        let old_cost = current.cost;
                        let old_crossings = current.crossings;
                        let old_wl = current.wirelength;
                        let old_ov = current.overlaps;
                        let mut modified: Vec<usize> = Vec::with_capacity(2);

                        let move_type = rng.gen_range(0..100);
                        if move_type < 50 {
                            let idx = movable[rng.gen_range(0..movable.len())];
                            let max_shift = (10.0 * cooling).max(config.grid_step_mm * 2.0);
                            let nx = snap_to_grid(
                                (current.positions[idx].x + rng.gen_range(-max_shift..=max_shift))
                                    .clamp(boundary.min_x + 1.2, boundary.max_x - 1.2),
                                config.grid_step_mm,
                            );
                            let ny = snap_to_grid(
                                (current.positions[idx].y + rng.gen_range(-max_shift..=max_shift))
                                    .clamp(boundary.min_y + 1.2, boundary.max_y - 1.2),
                                config.grid_step_mm,
                            );
                            current.positions[idx] = Point::new(nx, ny);
                            modified.push(idx);
                        } else if move_type < 75 {
                            let idx = movable[rng.gen_range(0..movable.len())];
                            let deltas = [90.0, 180.0, 270.0];
                            current.rotations[idx] =
                                (current.rotations[idx] + deltas[rng.gen_range(0..3)]) % 360.0;
                            modified.push(idx);
                        } else if move_type < 90 && movable.len() >= 2 {
                            let a = movable[rng.gen_range(0..movable.len())];
                            let b = movable[rng.gen_range(0..movable.len())];
                            if a != b {
                                current.positions.swap(a, b);
                                modified.push(a);
                                modified.push(b);
                            }
                        } else if !ties.is_empty() {
                            let tie = &ties[rng.gen_range(0..ties.len())];
                            let idx = tie.passive_idx;
                            let anchor_pt = current.cached_pads[tie.anchor_idx][tie.anchor_pad_idx];
                            let nx = snap_to_grid(
                                (anchor_pt.x + rng.gen_range(-2.5..=2.5))
                                    .clamp(boundary.min_x + 1.2, boundary.max_x - 1.2),
                                config.grid_step_mm,
                            );
                            let ny = snap_to_grid(
                                (anchor_pt.y + rng.gen_range(-2.5..=2.5))
                                    .clamp(boundary.min_y + 1.2, boundary.max_y - 1.2),
                                config.grid_step_mm,
                            );
                            current.positions[idx] = Point::new(nx, ny);
                            let angles = [0.0, 90.0, 180.0, 270.0];
                            current.rotations[idx] = angles[rng.gen_range(0..4)];
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
                        let accept = if delta <= 0.0 {
                            true
                        } else {
                            rng.gen::<f64>() < (-delta / temp).exp()
                        };

                        if accept {
                            if current.overlaps == 0 {
                                if let Ok(mut g) = global_best.lock() {
                                    if current.cost < g.cost || g.overlaps > 0 {
                                        *g = current.clone();
                                    }
                                }
                            }
                        } else {
                            current.positions = old_positions;
                            current.rotations = old_rotations;
                            for &m in &modified {
                                current.update_component(m, board, config);
                            }
                            current.cost = old_cost;
                            current.crossings = old_crossings;
                            current.wirelength = old_wl;
                            current.overlaps = old_ov;
                        }
                    }
                }
                total_evals.fetch_add(local_evals, Ordering::Relaxed);
            });

        // Re-seed highest-cost replica from global best and update GUI state
        let best_snap = global_best.lock().unwrap().clone();
        if let Some(worst) = replicas.iter_mut().max_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap()) {
            *worst = best_snap.clone();
        }

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

    let final_best = global_best.lock().unwrap().clone();
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
