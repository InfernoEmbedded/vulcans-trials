// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! Interactive and automated shove/nudge engine for footprints, tracks, and vias.

use super::*;

/// Interactive and automated shove/nudge engine that displaces unlocked footprints, tracks, and vias
/// to open routing channels when congestion cannot be resolved by pure pathfinding alone.
pub struct NudgeEngine {
    /// Base displacement increment in millimeters (`mm`) per shove step.
    pub step_mm: f64,
    /// Maximum allowed total displacement in millimeters (`mm`) when nudging a component.
    pub max_displacement_mm: f64,
}

impl NudgeEngine {
    /// Creates a new [`NudgeEngine`] with default `0.25 mm` step size and `2.0 mm` maximum displacement.
    pub fn new() -> Self {
        Self {
            step_mm: 0.25,
            max_displacement_mm: 2.0,
        }
    }

    /// Generates an ordered list of unique 2D displacement vectors `(dx, dy)` (in `mm`) oriented along
    /// and perpendicular to `push_dir`, up to [`NudgeEngine::max_displacement_mm`].
    ///
    /// Includes axial/lateral steps, 45° diagonal steps, and compound corridor-exit shifts.
    ///
    /// # Arguments
    /// * `push_dir` - Preferred push direction vector `(vx, vy)`.
    ///
    /// # Returns
    /// Deduplicated vector of candidate offset vectors `(dx, dy)` in millimeters (`mm`).
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

    /// Validates whether moving footprint `fp_idx` to world position `(nx, ny)` (in `mm`) preserves
    /// board outline containment, keepout zones, SAT convex courtyard separation (`0.20 mm` margin),
    /// and pad-to-track copper clearances.
    ///
    /// # Arguments
    /// * `board` - Reference to the [`Board`] containing footprints, keepouts, and routed tracks.
    /// * `fp_idx` - Index of the candidate footprint in `board.footprints`.
    /// * `nx` - Candidate world X coordinate (`mm`).
    /// * `ny` - Candidate world Y coordinate (`mm`).
    ///
    /// # Returns
    /// `true` if the proposed footprint placement is completely legal, or `false` if any collision occurs.
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

    /// Attempts to nudge an unlocked footprint `fp_idx` along `push_dir` to open a blocked routing channel.
    ///
    /// If a legal candidate position is found, rips up any traces currently connected to `fp_idx`
    /// and updates the footprint's `(x, y)` coordinates in `board`.
    ///
    /// # Arguments
    /// * `board` - Mutable reference to the [`Board`].
    /// * `fp_idx` - Footprint index in `board.footprints`.
    /// * `push_dir` - Preferred displacement direction vector `(vx, vy)`.
    ///
    /// # Returns
    /// `true` if the footprint was legally displaced, or `false` if locked or no valid nudge candidate exists.
    ///
    /// # Side Effects
    /// Mutates `board.footprints[fp_idx]` coordinates and removes connected traces in `board.tracks` / `board.vias`.
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
    ///
    /// # Arguments
    /// * `board` - Mutable reference to the [`Board`].
    /// * `track_idx` - Index of the target track in `board.tracks`.
    /// * `dir_sign` - Sign (`>= 0.0` or `< 0.0`) selecting which perpendicular normal direction to try first.
    /// * `grid` - Optional reference to [`RoutingGrid`] for obstacle cell verification.
    ///
    /// # Returns
    /// `true` if the track segment was shifted to a collision-free parallel position.
    ///
    /// # Side Effects
    /// Updates `board.tracks[track_idx].start` and `board.tracks[track_idx].end` in place.
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

    /// Shoves an unlocked via to a nearby legal location, extending or updating its incident track endpoints.
    ///
    /// # Arguments
    /// * `board` - Mutable reference to the [`Board`].
    /// * `via_idx` - Index of the target via in `board.vias`.
    /// * `push_dir` - Preferred push direction vector `(vx, vy)`.
    /// * `grid` - Optional reference to [`RoutingGrid`] for obstacle verification across all layers.
    ///
    /// # Returns
    /// `true` if the via and all incident tracks were legally relocated.
    ///
    /// # Side Effects
    /// Updates `board.vias[via_idx].position` and connected endpoints in `board.tracks`.
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
    ///
    /// Supports cascading shove into adjacent unlocked tracks up to 3 recursion hops.
    ///
    /// # Arguments
    /// * `board` - Mutable reference to the [`Board`].
    /// * `track_idx` - Index of the victim track in `board.tracks`.
    /// * `intruder_p1` - Start point `(x, y)` of the intruding segment (`mm`).
    /// * `intruder_p2` - End point `(x, y)` of the intruding segment (`mm`).
    /// * `intruder_width` - Copper width (`mm`) of the intruding segment.
    /// * `intruder_net_id` - Net ID of the intruding segment.
    /// * `grid` - Optional reference to [`RoutingGrid`].
    ///
    /// # Returns
    /// `true` if the victim track (and any cascaded neighbors) was deflected with full DRC legality.
    ///
    /// # Side Effects
    /// Replaces `board.tracks[track_idx]` (and any cascaded tracks) with 2 or 3 octilinear elbow segments.
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

    /// Shoves an unlocked track segment with a specific lateral offset (in `mm`) along its perpendicular normal,
    /// inserting 45-degree transition elbows and cascading into neighbors if needed.
    ///
    /// # Arguments
    /// * `board` - Mutable reference to the [`Board`].
    /// * `track_idx` - Index of the target track in `board.tracks`.
    /// * `offset_mm` - Signed lateral deflection distance in millimeters (`mm`).
    /// * `grid` - Optional reference to [`RoutingGrid`].
    ///
    /// # Returns
    /// `true` if the deflection succeeded without violating DRC rules.
    ///
    /// # Side Effects
    /// Replaces `board.tracks[track_idx]` with deflected octilinear segments.
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

    /// Recursive worker for [`NudgeEngine::try_continuous_shove_track`], bounding recursion depth to
    /// `MAX_DEPTH = 3` and total cascaded track deflections to `MAX_TOTAL_SHOVES = 8`.
    ///
    /// # Arguments
    /// * `board` - Mutable reference to the [`Board`].
    /// * `track_idx` - Index of the track to deflect in `board.tracks`.
    /// * `intruder_p1` - Start point `(x, y)` of the intruding segment (`mm`).
    /// * `intruder_p2` - End point `(x, y)` of the intruding segment (`mm`).
    /// * `intruder_width` - Width (`mm`) of the intruding segment.
    /// * `intruder_net_id` - Net ID of the intruding segment.
    /// * `grid` - Optional reference to [`RoutingGrid`].
    /// * `depth` - Current recursion depth (`0..3`).
    /// * `shoved_nets` - Set of net IDs already deflected in the current cascade chain to prevent cycles.
    /// * `total_shoves` - Counter of total tracks deflected in the cascade chain.
    ///
    /// # Returns
    /// `true` if `track_idx` already meets clearance or was successfully deflected.
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

    /// Synthesizes the 45° elbow-deflected replacement segments for `track_idx`, validates them
    /// against board bounds, keepouts, pads, vias, and locked tracks, and recursively shoves any
    /// conflicting unlocked tracks before committing the replacement to `board.tracks`.
    ///
    /// # Arguments
    /// * `board` - Mutable reference to the [`Board`].
    /// * `track_idx` - Index of the track in `board.tracks` being deflected.
    /// * `dir_normal` - Unit normal vector `(nx, ny)` pointing in the deflection direction.
    /// * `offset_mm` - Deflection distance in millimeters (`mm`).
    /// * `grid` - Optional reference to [`RoutingGrid`].
    /// * `depth` - Current cascade recursion depth.
    /// * `intruder_net_id` - Net ID of the intruding feature causing this deflection.
    /// * `shoved_nets` - Set of net IDs already deflected in this cascade.
    /// * `total_shoves` - Mutable counter of total shoves performed.
    ///
    /// # Returns
    /// `true` if the deflection and all child cascades succeeded, or `false` (restoring `board.tracks` from backup) on failure.
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
