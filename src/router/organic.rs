// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! Homotopic rubber-band relaxation, cubic Bézier splines, tangent circular arc fitting,
//! acute-junction acid-trap filleting, and netclass copper widening.

use super::*;

/// Circular obstacle representation used during homotopic rubber-band relaxation.
#[derive(Clone, Debug)]
pub struct ObstacleInfo {
    /// World-space center `(x, y)` in millimeters (`mm`) of the foreign pad or via.
    pub center: (f64, f64),
    /// Minimum safe clearance radius in millimeters (`mm`) from `center` to a trace's centerline.
    pub safe_radius: f64,
}

/// Tests whether a 2D point lies strictly inside a simple polygon using the ray-casting (even-odd) rule.
///
/// Used by homotopic rubber-band relaxation to ensure a shortcut chord or triangle contraction
/// does not jump across a foreign pad or via obstacle (preserving path homotopy).
///
/// # Arguments
/// * `pt` - Query point `(x, y)` in millimeters (`mm`).
/// * `poly` - Ordered polygon vertices `(x, y)` in millimeters (`mm`).
///
/// # Returns
/// `true` if `pt` lies inside `poly`.
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

/// Collects all foreign pads and vias on `layer` as circular [`ObstacleInfo`] primitives for homotopic relaxation.
///
/// # Arguments
/// * `board` - Board state containing footprints and vias.
/// * `layer` - Copper layer index.
/// * `net_id` - Net ID of the trace being relaxed.
/// * `track_width` - Trace width in millimeters (`mm`).
///
/// # Returns
/// Vector of [`ObstacleInfo`] items with precomputed safe clearance radii.
pub fn collect_obstacles_for_net(board: &Board, layer: usize, net_id: usize, track_width: f64) -> Vec<ObstacleInfo> {
    let net_clr = board.get_netclass_for_net(net_id).clearance;
    let half_w = track_width * 0.5;
    let mut obstacles = Vec::new();

    // 1. Foreign Pads
    for fp in &board.footprints {
        for pad in &fp.pads {
            let pad_layer = fp.pad_layer(pad);
            if pad.net_id != net_id && (pad.is_through_hole || pad_layer == layer) {
                let (px, py) = fp.pad_absolute_position(pad);
                let pad_clr = board.get_netclass_for_net(pad.net_id).clearance;
                let req_clr = net_clr.max(pad_clr);
                let pad_r = (pad.width * pad.width + pad.height * pad.height).sqrt() * 0.5;
                obstacles.push(ObstacleInfo {
                    center: (px, py),
                    safe_radius: pad_r + req_clr + half_w + 0.06,
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
                safe_radius: via_r + req_clr + half_w + 0.06,
            });
        }
    }

    obstacles
}

/// Subdivides polyline segments longer than `max_seg_len` by inserting intermediate vertices.
///
/// Gives homotopic rubber-band relaxation intermediate degrees of freedom to bend smoothly
/// around nearby obstacles when pulling a long segment taut.
///
/// # Arguments
/// * `points` - Mutable vector of polyline vertices `(x, y)` in millimeters (`mm`).
/// * `max_seg_len` - Maximum segment length threshold in millimeters (`mm`) before subdivision.
///
/// # Side Effects
/// Mutates `points` in place (capped at 28 vertices).
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

/// Relaxes unlocked tracks across all nets on `board` into taut any-angle homotopic rubber-band polylines.
///
/// # Arguments
/// * `board` - Mutable board state whose unlocked tracks are relaxed.
///
/// # Side Effects
/// Replaces unlocked non-differential tracks in `board.tracks`.
pub fn relax_homotopic_rubber_bands(board: &mut Board) {
    relax_homotopic_rubber_bands_with_grid(board, None);
}

/// Relaxes unlocked tracks across all nets on `board` into taut any-angle homotopic rubber-band polylines,
/// using optional GPU DRC acceleration from `grid`.
///
/// Also widens power/ground chains toward their netclass target width where clearance permits.
///
/// # Arguments
/// * `board` - Mutable board state whose unlocked tracks are relaxed.
/// * `grid` - Optional routing grid providing a Vulkan compute engine for GPU shortcut screening.
///
/// # Side Effects
/// Mutates `board.tracks`.
pub fn relax_homotopic_rubber_bands_with_grid(board: &mut Board, grid: Option<&RoutingGrid>) {
    let mut net_ids: Vec<usize> = board
        .tracks
        .iter()
        .filter(|t| !t.is_locked)
        .map(|t| t.net_id)
        .collect();
    net_ids.sort_unstable();
    net_ids.dedup();

    let has_vulkan = grid.and_then(|g| g.vulkan.as_ref()).is_some();

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

        let nc = board.get_netclass_for_net(net_id);
        let nc_w = net_target_width(board, net_id);
        let net_name = board.nets.get(net_id).map(|s| s.as_str()).unwrap_or("");
        let is_coupled_diff_pair = nc.name == "USB_Diff"
            || board.nets.iter().enumerate().any(|(other_id, other_name)| {
                other_id != net_id && AutoRouter::are_diff_pair_mates(net_name, other_name)
            });
        let drc_ctx = if has_vulkan && !is_coupled_diff_pair {
            Some(GpuDrcContext::from_board(board))
        } else {
            None
        };
        let mut chains = extract_track_chains_for_organic(&net_tracks, board);
        for chain in &mut chains {
            if !is_coupled_diff_pair {
                relax_chain_homotopic_with_gpu(chain, board, grid, drc_ctx.as_ref());
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

/// Relaxes a single [`TrackChain`] into a taut homotopic rubber-band polyline while preserving obstacle topology.
///
/// # Arguments
/// * `chain` - Mutable track chain to relax.
/// * `board` - Board state with obstacles.
///
/// # Side Effects
/// Mutates `chain.points` and `chain.width` in place.
pub fn relax_chain_homotopic(chain: &mut TrackChain, board: &Board) {
    relax_chain_homotopic_with_gpu(chain, board, None, None);
}

/// Internal implementation of homotopic rubber-band chain relaxation with optional GPU DRC batch screening.
///
/// Performs 5 stages per pass:
/// 1. Homotopic triangle shortcutting (collapsing 3-vertex bends if no obstacle lies inside the triangle).
/// 2. Multi-point chord shortcutting (collapsing sub-chains if no obstacle lies inside the enclosed polygon).
/// 3. Obstacle tangent apex contraction (pulling vertices radially onto the safe clearance boundary of blocking obstacles).
/// 4. Binary-search taut relaxation toward neighbor midpoints.
/// 5. Netclass channel centering for wide power traces.
///
/// # Arguments
/// * `chain` - Mutable track chain to relax.
/// * `board` - Board state with obstacles.
/// * `grid` - Optional routing grid.
/// * `drc_ctx` - Optional GPU DRC context for batch shortcut screening.
///
/// # Side Effects
/// Mutates `chain.points` and `chain.width` in place.
pub(crate) fn relax_chain_homotopic_with_gpu(
    chain: &mut TrackChain,
    board: &Board,
    grid: Option<&RoutingGrid>,
    drc_ctx: Option<&GpuDrcContext>,
) {
    if chain.points.len() < 3 {
        return;
    }

    let nc_w = net_target_width(board, chain.net_id).max(chain.width);
    let mut relax_w = chain.width;
    if nc_w > chain.width + 1e-3 {
        for &cand_w in &[nc_w, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25] {
            if cand_w <= chain.width + 1e-3 || cand_w > nc_w + 1e-3 {
                continue;
            }
            if chain.points.windows(2).all(|w| {
                is_segment_clear(board, None, w[0], w[1], chain.layer, chain.net_id, cand_w)
            }) {
                relax_w = cand_w;
                break;
            }
        }
    }

    // Collapse tiny pad-to-grid micro-segments (< 0.12mm) at the start or end of the chain
    let mut k_micro = 1;
    while k_micro + 1 < chain.points.len() {
        let p_prev = chain.points[k_micro - 1];
        let p_curr = chain.points[k_micro];
        let p_next = chain.points[k_micro + 1];
        let l1 = (p_curr.0 - p_prev.0).hypot(p_curr.1 - p_prev.1);
        let l2 = (p_next.0 - p_curr.0).hypot(p_next.1 - p_curr.1);
        if (l1 < 0.12 || l2 < 0.12)
            && is_segment_clear(board, None, p_prev, p_next, chain.layer, chain.net_id, relax_w)
        {
            chain.points.remove(k_micro);
        } else {
            k_micro += 1;
        }
    }
    if chain.points.len() < 3 {
        chain.width = relax_w;
        return;
    }

    let obstacles = collect_obstacles_for_net(board, chain.layer, chain.net_id, nc_w);
    let clearance = board.get_netclass_for_net(chain.net_id).clearance as f32;

    // Iterative rubber-band string pulling passes with early convergence exit
    for pass in 0..4 {
        merge_collinear_points(&mut chain.points);
        if pass == 0 {
            subdivide_long_segments(&mut chain.points, 3.0);
        }
        if chain.points.len() < 3 {
            break;
        }
        let mut changed = false;

        let gpu_shortcut_map: Option<GpuShortcutMap> = if let (Some(g), Some(ctx)) = (grid, drc_ctx) {
            if g.vulkan.is_some() {
                let n = chain.points.len();
                let mut keys = Vec::with_capacity(n * (n.saturating_sub(2)) / 2);
                let mut shortcut_cands = Vec::with_capacity(keys.capacity());
                let mut seen = HashSet::new();
                let half_w = (relax_w * 0.5) as f32;
                for a in 0..n {
                    for b in (a + 2)..n {
                        let p1 = chain.points[a];
                        let p2 = chain.points[b];
                        if (p1.0 - p2.0).hypot(p1.1 - p2.1) <= 1e-4 {
                            continue;
                        }
                        let key = canonical_shortcut_key(p1, p2, chain.layer, chain.net_id);
                        if seen.insert(key) {
                            keys.push(key);
                            shortcut_cands.push(GpuDrcCandidate {
                                x1: p1.0 as f32,
                                y1: p1.1 as f32,
                                x2: p2.0 as f32,
                                y2: p2.1 as f32,
                                radius_or_half_width: half_w,
                                clearance,
                                layer: chain.layer as u32,
                                net_id: chain.net_id as u32,
                                kind: 1,
                                ignore_same_net_pads: 1,
                                _pad0: 0,
                                _pad1: 0,
                            });
                        }
                    }
                }
                if !shortcut_cands.is_empty() {
                    ctx.evaluate(g.vulkan.as_ref(), &shortcut_cands).map(|res| {
                        keys.into_iter().zip(res.into_iter()).map(|(k, v)| (k, v != 0)).collect()
                    })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let gpu_allows = |p1: (f64, f64), p2: (f64, f64)| -> bool {
            if let Some(ref map) = gpu_shortcut_map {
                let key = canonical_shortcut_key(p1, p2, chain.layer, chain.net_id);
                if let Some(&false) = map.get(&key) {
                    return false;
                }
            }
            true
        };

        // 1. Homotopic Triangle Shortcutting
        let mut i = 1;
        while i + 1 < chain.points.len() {
            let p_prev = chain.points[i - 1];
            let p_curr = chain.points[i];
            let p_next = chain.points[i + 1];

            if gpu_allows(p_prev, p_next) {
                let tri = [p_prev, p_curr, p_next];
                let homotopy_ok = obstacles.iter().all(|obs| !point_in_polygon(obs.center, &tri));
                if homotopy_ok && is_segment_clear(board, None, p_prev, p_next, chain.layer, chain.net_id, relax_w) {
                    chain.points.remove(i);
                    changed = true;
                    continue;
                }
            }
            i += 1;
        }

        // 2. Multi-point Chord Shortcutting
        let mut start_idx = 0;
        while start_idx < chain.points.len() {
            let mut best_j = None;
            for j in ((start_idx + 2)..chain.points.len()).rev() {
                let p_start = chain.points[start_idx];
                let p_end = chain.points[j];
                if !gpu_allows(p_start, p_end) {
                    continue;
                }
                let poly = &chain.points[start_idx..=j];
                let homotopy_ok = obstacles.iter().all(|obs| !point_in_polygon(obs.center, poly));
                if homotopy_ok && is_segment_clear(board, None, p_start, p_end, chain.layer, chain.net_id, relax_w) {
                    best_j = Some(j);
                    break;
                }
            }
            if let Some(j) = best_j {
                chain.points.drain((start_idx + 1)..j);
                changed = true;
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
                    if (target_pt.0 - p_curr.0).hypot(target_pt.1 - p_curr.1) > 0.01 {
                        let tri1 = [p_prev, p_curr, target_pt];
                        let tri2 = [p_curr, p_next, target_pt];
                        let h1 = obstacles.iter().all(|o| !point_in_polygon(o.center, &tri1));
                        let h2 = obstacles.iter().all(|o| !point_in_polygon(o.center, &tri2));
                        if h1
                            && h2
                            && is_segment_clear(board, None, p_prev, target_pt, chain.layer, chain.net_id, relax_w)
                            && is_segment_clear(board, None, target_pt, p_next, chain.layer, chain.net_id, relax_w)
                        {
                            chain.points[k] = target_pt;
                            changed = true;
                        }
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
            let move_dist = (target.0 - p_curr.0).hypot(target.1 - p_curr.1);
            if move_dist < 0.02 {
                continue;
            }

            let mut low = 0.0;
            let mut high = 1.0;
            let mut best_t = 0.0;

            for _ in 0..5 {
                let mid = (low + high) * 0.5;
                let cand = (
                    p_curr.0 + mid * (target.0 - p_curr.0),
                    p_curr.1 + mid * (target.1 - p_curr.1),
                );
                let tri1 = [p_prev, p_curr, cand];
                let tri2 = [p_curr, p_next, cand];
                let h1 = obstacles.iter().all(|o| !point_in_polygon(o.center, &tri1));
                let h2 = obstacles.iter().all(|o| !point_in_polygon(o.center, &tri2));
                if h1
                    && h2
                    && is_segment_clear(board, None, p_prev, cand, chain.layer, chain.net_id, relax_w)
                    && is_segment_clear(board, None, cand, p_next, chain.layer, chain.net_id, relax_w)
                {
                    best_t = mid;
                    low = mid;
                } else {
                    high = mid;
                }
            }

            if best_t * move_dist > 0.01 {
                chain.points[k] = (
                    p_curr.0 + best_t * (target.0 - p_curr.0),
                    p_curr.1 + best_t * (target.1 - p_curr.1),
                );
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    chain.width = relax_w;
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

/// Checks whether an [`ArcSegment`] satisfies all board boundary, foreign pad, via, keepout, track, and arc clearances.
///
/// Discretizes the arc into chord segments and enforces symmetric bidirectional netclass copper
/// and drill-hole clearances against all board obstacles.
///
/// # Arguments
/// * `board` - Board state containing obstacles and netclasses.
/// * `arc` - Candidate [`ArcSegment`] to validate.
///
/// # Returns
/// `true` if the arc is completely free of DRC violations.
pub fn is_arc_clear(board: &Board, arc: &ArcSegment) -> bool {
    let clearance = board.get_netclass_for_net(arc.net_id).clearance;
    let half_w = arc.width * 0.5;
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

    let mut arc_min_x = f64::INFINITY;
    let mut arc_max_x = f64::NEG_INFINITY;
    let mut arc_min_y = f64::INFINITY;
    let mut arc_max_y = f64::NEG_INFINITY;
    for s in &segs {
        arc_min_x = arc_min_x.min(s.start.0).min(s.end.0);
        arc_max_x = arc_max_x.max(s.start.0).max(s.end.0);
        arc_min_y = arc_min_y.min(s.start.1).min(s.end.1);
        arc_max_y = arc_max_y.max(s.start.1).max(s.end.1);
    }
    let max_board_clr = board
        .netclasses
        .values()
        .map(|c| c.clearance)
        .fold(clearance, f64::max);

    // 1. Foreign Pads
    let min_hole_clr = board.get_netclass_for_net(arc.net_id).min_hole_clearance;
    for fp in &board.footprints {
        for pad in &fp.pads {
            let pad_layer = fp.pad_layer(pad);
            if pad.net_id != arc.net_id && (pad.is_through_hole || pad_layer == arc.layer) {
                let max_dim = pad.width.max(pad.height);
                let coarse_box = half_w + max_board_clr.max(min_hole_clr) + max_dim;
                let (px, py) = fp.pad_absolute_position(pad);
                if px < arc_min_x - coarse_box
                    || px > arc_max_x + coarse_box
                    || py < arc_min_y - coarse_box
                    || py > arc_max_y + coarse_box
                {
                    continue;
                }
                let d_center = arc.distance_to_point((px, py));
                if pad.is_through_hole {
                    if let Some(drill) = pad.drill {
                        if drill > 0.0 && d_center < half_w + drill * 0.5 + min_hole_clr - 1e-4 {
                            return false;
                        }
                    }
                }
                let pad_clr = board.get_netclass_for_net(pad.net_id).clearance;
                let req_clr = clearance.max(pad_clr);
                let required_dist = half_w + req_clr;

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
            let coarse_v = half_w + via.size * 0.5 + max_board_clr.max(min_hole_clr);
            if via.position.0 < arc_min_x - coarse_v
                || via.position.0 > arc_max_x + coarse_v
                || via.position.1 < arc_min_y - coarse_v
                || via.position.1 > arc_max_y + coarse_v
            {
                continue;
            }
            let via_clr = board.get_netclass_for_net(via.net_id).clearance;
            let req_clr = clearance.max(via_clr);
            let required_dist = half_w + (via.size * 0.5 + req_clr).max(via.drill * 0.5 + min_hole_clr);
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
            let coarse_t = half_w + trk.width * 0.5 + max_board_clr;
            let t_min_x = trk.start.0.min(trk.end.0);
            let t_max_x = trk.start.0.max(trk.end.0);
            let t_min_y = trk.start.1.min(trk.end.1);
            let t_max_y = trk.start.1.max(trk.end.1);
            if t_max_x < arc_min_x - coarse_t
                || t_min_x > arc_max_x + coarse_t
                || t_max_y < arc_min_y - coarse_t
                || t_min_y > arc_max_y + coarse_t
            {
                continue;
            }
            let trk_clr = board.get_netclass_for_net(trk.net_id).clearance;
            let req_clr = clearance.max(trk_clr);
            let required_dist = half_w + (trk.width * 0.5) + req_clr;
            for s in &segs {
                let s_min_x = s.start.0.min(s.end.0);
                let s_max_x = s.start.0.max(s.end.0);
                let s_min_y = s.start.1.min(s.end.1);
                let s_max_y = s.start.1.max(s.end.1);
                if t_max_x < s_min_x - required_dist
                    || t_min_x > s_max_x + required_dist
                    || t_max_y < s_min_y - required_dist
                    || t_min_y > s_max_y + required_dist
                {
                    continue;
                }
                if dist_segment_to_segment(s.start, s.end, trk.start, trk.end) < required_dist - 1e-4 {
                    return false;
                }
            }
        }
    }

    // 5. Foreign Arcs
    for other in &board.arcs {
        if other.net_id != arc.net_id && other.layer == arc.layer {
            let os_pt = other.start();
            let om_pt = other.mid();
            let oe_pt = other.end();
            let o_min_x = os_pt.0.min(om_pt.0).min(oe_pt.0);
            let o_max_x = os_pt.0.max(om_pt.0).max(oe_pt.0);
            let o_min_y = os_pt.1.min(om_pt.1).min(oe_pt.1);
            let o_max_y = os_pt.1.max(om_pt.1).max(oe_pt.1);
            let sag = 0.5 * (om_pt.0 - os_pt.0).hypot(om_pt.1 - os_pt.1)
                + other.width * 0.5
                + half_w
                + max_board_clr;
            if o_max_x + sag < arc_min_x
                || o_min_x - sag > arc_max_x
                || o_max_y + sag < arc_min_y
                || o_min_y - sag > arc_max_y
            {
                continue;
            }
            let other_clr = board.get_netclass_for_net(other.net_id).clearance;
            let req_clr = clearance.max(other_clr);
            let required_dist = half_w + (other.width * 0.5) + req_clr;
            let other_segs = other.to_segments(12);
            for s in &segs {
                let s_min_x = s.start.0.min(s.end.0) - required_dist;
                let s_max_x = s.start.0.max(s.end.0) + required_dist;
                let s_min_y = s.start.1.min(s.end.1) - required_dist;
                let s_max_y = s.start.1.max(s.end.1) + required_dist;
                for os in &other_segs {
                    if os.start.0.max(os.end.0) < s_min_x
                        || os.start.0.min(os.end.0) > s_max_x
                        || os.start.1.max(os.end.1) < s_min_y
                        || os.start.1.min(os.end.1) > s_max_y
                    {
                        continue;
                    }
                    if dist_segment_to_segment(s.start, s.end, os.start, os.end) < required_dist - 1e-4 {
                        return false;
                    }
                }
            }
        }
    }

    true
}

/// Batch-screens candidate [`ArcSegment`] primitives on the GPU ([`GpuDrcContext`], 8 chords per arc)
/// and verifies all GPU-passing candidates in parallel via [`is_arc_clear`].
///
/// # Arguments
/// * `board` - Board state with obstacles and netclasses.
/// * `drc_ctx` - GPU DRC snapshot used for hardware-accelerated screening.
/// * `arcs` - Slice of candidate [`ArcSegment`] items to validate.
///
/// # Returns
/// Vector of booleans (one per element of `arcs`) indicating whether each arc is DRC-clean.
pub fn batch_validate_arcs_gpu(
    board: &Board,
    drc_ctx: &GpuDrcContext,
    arcs: &[ArcSegment],
) -> Vec<bool> {
    if arcs.is_empty() {
        return Vec::new();
    }

    const STEPS: usize = 8;
    let mut cands = Vec::with_capacity(arcs.len() * STEPS);
    for arc in arcs {
        let segs = arc.to_segments(STEPS);
        let half_w = (arc.width * 0.5) as f32;
        let clr = board.get_netclass_for_net(arc.net_id).clearance.max(0.15) as f32;
        for s in segs {
            cands.push(GpuDrcCandidate {
                x1: s.start.0 as f32,
                y1: s.start.1 as f32,
                x2: s.end.0 as f32,
                y2: s.end.1 as f32,
                radius_or_half_width: half_w,
                clearance: clr,
                layer: arc.layer as u32,
                net_id: arc.net_id as u32,
                kind: 1,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            });
        }
    }

    let gpu_res = drc_ctx.evaluate(None, &cands);
    arcs.par_iter()
        .enumerate()
        .map(|(idx, arc)| {
            let gpu_ok = gpu_res
                .as_ref()
                .map(|res| {
                    let base = idx * STEPS;
                    res.get(base..base + STEPS)
                        .map(|slice| slice.iter().all(|&v| v == 1))
                        .unwrap_or(true)
                })
                .unwrap_or(true);
            gpu_ok && is_arc_clear(board, arc)
        })
        .collect()
}

/// Cubic Bézier curve $B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3$
/// used to represent $G^1$-continuous organic track corners and arches before decomposing
/// into strictly tangent KiCad circular [`ArcSegment`] primitives.
#[derive(Debug, Clone, Copy)]
pub struct CubicBezier {
    /// Start endpoint $P_0 = (x, y)$ in millimeters (`mm`) at $t = 0$.
    pub p0: (f64, f64),
    /// First interior control point $P_1 = (x, y)$ in millimeters (`mm`) governing the entry tangent $P_1 - P_0$.
    pub p1: (f64, f64),
    /// Second interior control point $P_2 = (x, y)$ in millimeters (`mm`) governing the exit tangent $P_3 - P_2$.
    pub p2: (f64, f64),
    /// End endpoint $P_3 = (x, y)$ in millimeters (`mm`) at $t = 1$.
    pub p3: (f64, f64),
}

impl CubicBezier {
    /// Evaluates the point $B(t)$ on the cubic Bézier curve at parameter `t` in `[0.0, 1.0]`.
    ///
    /// # Arguments
    /// * `t` - Curve parameter in `[0.0, 1.0]`.
    ///
    /// # Returns
    /// World coordinates `(x, y)` in millimeters (`mm`).
    pub fn eval(&self, t: f64) -> (f64, f64) {
        let u = 1.0 - t;
        let b0 = u * u * u;
        let b1 = 3.0 * u * u * t;
        let b2 = 3.0 * u * t * t;
        let b3 = t * t * t;
        (
            b0 * self.p0.0 + b1 * self.p1.0 + b2 * self.p2.0 + b3 * self.p3.0,
            b0 * self.p0.1 + b1 * self.p1.1 + b2 * self.p2.1 + b3 * self.p3.1,
        )
    }

    /// Evaluates the first derivative vector $B'(t)$ of the cubic Bézier curve at parameter `t` in `[0.0, 1.0]`.
    ///
    /// # Arguments
    /// * `t` - Curve parameter in `[0.0, 1.0]`.
    ///
    /// # Returns
    /// Tangent derivative vector `(dx/dt, dy/dt)` in millimeters (`mm`).
    pub fn derivative(&self, t: f64) -> (f64, f64) {
        let u = 1.0 - t;
        let d0 = 3.0 * u * u;
        let d1 = 6.0 * u * t;
        let d2 = 3.0 * t * t;
        (
            d0 * (self.p1.0 - self.p0.0) + d1 * (self.p2.0 - self.p1.0) + d2 * (self.p3.0 - self.p2.0),
            d0 * (self.p1.1 - self.p0.1) + d1 * (self.p2.1 - self.p1.1) + d2 * (self.p3.1 - self.p2.1),
        )
    }

    /// Constructs a $G^1$-tangent corner Bézier curve from entry point `t_in`, corner apex `p_corner`,
    /// and exit point `t_out`.
    ///
    /// The tangents at $t=0$ and $t=1$ point strictly along `p_corner - t_in` and `t_out - p_corner`.
    ///
    /// # Arguments
    /// * `t_in` - Entry tangent point `(x, y)` in `mm` on the incoming arm.
    /// * `p_corner` - Sharp corner apex `(x, y)` in `mm`.
    /// * `t_out` - Exit tangent point `(x, y)` in `mm` on the outgoing arm.
    ///
    /// # Returns
    /// A [`CubicBezier`] curve smoothly rounding the corner at `p_corner`.
    pub fn from_corner(t_in: (f64, f64), p_corner: (f64, f64), t_out: (f64, f64)) -> Self {
        const KAPPA: f64 = 0.5522847498;
        let p1 = (
            t_in.0 + KAPPA * (p_corner.0 - t_in.0),
            t_in.1 + KAPPA * (p_corner.1 - t_in.1),
        );
        let p2 = (
            t_out.0 + KAPPA * (p_corner.0 - t_out.0),
            t_out.1 + KAPPA * (p_corner.1 - t_out.1),
        );
        Self {
            p0: t_in,
            p1,
            p2,
            p3: t_out,
        }
    }

    /// Constructs a smooth single-span arch Bézier between `start` and `end` with midpoint normal
    /// deflection `normal_offset`.
    ///
    /// # Arguments
    /// * `start` - Start point `(x, y)` in `mm`.
    /// * `end` - End point `(x, y)` in `mm`.
    /// * `normal_offset` - Signed perpendicular bow deflection in `mm` at the curve midpoint.
    ///
    /// # Returns
    /// A [`CubicBezier`] arch connecting `start` and `end`.
    pub fn from_arch(start: (f64, f64), end: (f64, f64), normal_offset: f64) -> Self {
        let dx = end.0 - start.0;
        let dy = end.1 - start.1;
        let len = dx.hypot(dy).max(1e-6);
        let nx = -dy / len;
        let ny = dx / len;
        let ctrl_off = normal_offset * (4.0 / 3.0);
        let p1 = (start.0 + dx / 3.0 + nx * ctrl_off, start.1 + dy / 3.0 + ny * ctrl_off);
        let p2 = (start.0 + 2.0 * dx / 3.0 + nx * ctrl_off, start.1 + 2.0 * dy / 3.0 + ny * ctrl_off);
        Self {
            p0: start,
            p1,
            p2,
            p3: end,
        }
    }

    /// Decomposes the cubic Bézier curve into 1 (symmetric) or 2 (asymmetric $G^1$ biarc) KiCad
    /// [`ArcSegment`] primitives whose tangents at `p0`, the interior biarc join, and `p3` are
    /// strictly $G^1$-continuous ($0.0^\circ$ kink).
    ///
    /// # Arguments
    /// * `width` - Trace width in millimeters (`mm`).
    /// * `layer` - Copper layer index.
    /// * `net_id` - Net ID of the arc(s).
    ///
    /// # Returns
    /// Vector of 1 or 2 tangent [`ArcSegment`] primitives (or empty if degenerate).
    pub fn to_kicad_arcs(&self, width: f64, layer: usize, net_id: usize) -> Vec<ArcSegment> {
        let v_in = (self.p1.0 - self.p0.0, self.p1.1 - self.p0.1);
        let v_out = (self.p2.0 - self.p3.0, self.p2.1 - self.p3.1);
        let l_in = v_in.0.hypot(v_in.1);
        let l_out = v_out.0.hypot(v_out.1);
        if l_in < 1e-6 || l_out < 1e-6 {
            return Vec::new();
        }
        // Unit vectors pointing from p0 and p3 toward the corner apex
        let ray_in = (v_in.0 / l_in, v_in.1 / l_in);
        let ray_out = (v_out.0 / l_out, v_out.1 / l_out);

        // Find ray intersection (corner apex P_corner) of P0 + d_in * ray_in and P3 + d_out * ray_out
        let rhs = (self.p3.0 - self.p0.0, self.p3.1 - self.p0.1);
        let det = -ray_in.0 * ray_out.1 + ray_in.1 * ray_out.0;
        if det.abs() > 1e-5 {
            let d_in = (-rhs.0 * ray_out.1 + rhs.1 * ray_out.0) / det;
            let d_out = (ray_in.0 * rhs.1 - ray_in.1 * rhs.0) / det;
            if d_in > 1e-4 && d_out > 1e-4 {
                let p_corner = (self.p0.0 + d_in * ray_in.0, self.p0.1 + d_in * ray_in.1);
                // Unit vectors pointing from p_corner back toward p0 and p3
                let u_in = (-ray_in.0, -ray_in.1);
                let u_out = (-ray_out.0, -ray_out.1);
                // Case 1: Symmetric leg lengths -> exact single circular arc tangent to both rays
                if (d_in - d_out).abs() <= 1e-3 * d_in.max(d_out).max(1.0) {
                    let dot = (u_in.0 * u_out.0 + u_in.1 * u_out.1).clamp(-1.0, 1.0);
                    let alpha = dot.acos() * 0.5;
                    let b_sum = (u_in.0 + u_out.0, u_in.1 + u_out.1);
                    let b_len = b_sum.0.hypot(b_sum.1);
                    if b_len > 1e-6 && alpha > 0.01 {
                        let b = (b_sum.0 / b_len, b_sum.1 / b_len);
                        let r = d_in * alpha.tan();
                        let h = r / alpha.sin();
                        let c = (p_corner.0 + h * b.0, p_corner.1 + h * b.1);
                        let m_arc = (c.0 - r * b.0, c.1 - r * b.1);
                        if let Some(arc) = ArcSegment::from_3pts(self.p0, m_arc, self.p3, width, layer, net_id) {
                            return vec![arc];
                        }
                    }
                }

                // Case 2: Asymmetric leg lengths -> exact G1 biarc decomposition
                let in_is_short = d_in < d_out;
                let (d_short, d_long, u_short, u_long, t_short, t_long) = if in_is_short {
                    (d_in, d_out, u_in, u_out, self.p0, self.p3)
                } else {
                    (d_out, d_in, u_out, u_in, self.p3, self.p0)
                };
                let s1 = d_short * 0.55;
                let q1 = (
                    p_corner.0 + (d_short - s1) * u_short.0,
                    p_corner.1 + (d_short - s1) * u_short.1,
                );
                let w = (t_long.0 - q1.0, t_long.1 - q1.1);
                let w_len2 = w.0 * w.0 + w.1 * w.1;
                let w_dot_ulong = w.0 * u_long.0 + w.1 * u_long.1;
                let denom = 2.0 * (s1 + w_dot_ulong);
                if denom > 1e-5 {
                    let s2 = (w_len2 - s1 * s1) / denom;
                    if s2 > 0.05 && s2 < d_long - 0.02 {
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
                        let (a1_opt, a2_opt) = if in_is_short {
                            (
                                ArcSegment::from_3pts(t_short, m1, join, width, layer, net_id),
                                ArcSegment::from_3pts(join, m2, t_long, width, layer, net_id),
                            )
                        } else {
                            (
                                ArcSegment::from_3pts(t_long, m2, join, width, layer, net_id),
                                ArcSegment::from_3pts(join, m1, t_short, width, layer, net_id),
                            )
                        };
                        if let (Some(a1), Some(a2)) = (a1_opt, a2_opt) {
                            return vec![a1, a2];
                        }
                    }
                }
            }
            return Vec::new();
        }

        // Parallel tangents (CubicBezier::from_arch): single circular arc through B(0), B(0.5), B(1)
        let mid = self.eval(0.5);
        ArcSegment::from_3pts(self.p0, mid, self.p3, width, layer, net_id)
            .into_iter()
            .collect()
    }
}

/// Fits a $G^1$-continuous asymmetric [`CubicBezier`] corner curve at `p_curr` when one leg is longer
/// than the other, decomposing it into two tangent circular arcs.
///
/// # Arguments
/// * `board` - Board state with obstacles.
/// * `drc_ctx` - GPU DRC snapshot for batch arc validation.
/// * `p_curr` - Corner apex `(x, y)` in `mm`.
/// * `u1` - Unit vector pointing from `p_curr` along the incoming leg.
/// * `u2` - Unit vector pointing from `p_curr` along the outgoing leg.
/// * `avail_a` - Available length in `mm` along `u1`.
/// * `avail_b` - Available length in `mm` along `u2`.
/// * `width` - Trace width in `mm`.
/// * `layer` - Copper layer index.
/// * `net_id` - Net ID of the trace.
///
/// # Returns
/// `Some((t_in, t_out, bi_arcs, used_b))` if a DRC-clean biarc fits, or `None` if blocked or too short.
pub(crate) fn try_fit_asymmetric_corner_biarc(
    board: &Board,
    drc_ctx: &GpuDrcContext,
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
    if short_avail < 0.30 || long_avail < short_avail * 1.4 || long_avail < 1.8 {
        return None;
    }

    let d_short = (short_avail * 0.92).min(6.0);
    let mut proposals: Vec<((f64, f64), (f64, f64), Vec<ArcSegment>, f64)> = Vec::new();
    let mut flat_arcs: Vec<ArcSegment> = Vec::new();
    for long_frac in [0.88, 0.70, 0.52, 0.36] {
        let d_long = (long_avail * long_frac).min(d_short * 4.8).max(d_short * 1.35);
        if d_long <= d_short * 1.25 || d_long > long_avail {
            continue;
        }
        let (d_a, d_b) = if a_is_short {
            (d_short, d_long)
        } else {
            (d_long, d_short)
        };
        let t_in = (p_curr.0 + d_a * u1.0, p_curr.1 + d_a * u1.1);
        let t_out = (p_curr.0 + d_b * u2.0, p_curr.1 + d_b * u2.1);

        let bezier = CubicBezier::from_corner(t_in, p_curr, t_out);
        let bi_arcs = bezier.to_kicad_arcs(width, layer, net_id);
        if bi_arcs.len() == 2 && bi_arcs.iter().all(|a| a.radius >= 0.35) {
            flat_arcs.extend_from_slice(&bi_arcs);
            proposals.push((t_in, t_out, bi_arcs, d_b));
        }
    }
    let valid_flags = batch_validate_arcs_gpu(board, drc_ctx, &flat_arcs);
    for (idx, prop) in proposals.into_iter().enumerate() {
        let base = idx * 2;
        if valid_flags.get(base..base + 2).is_some_and(|s| s[0] && s[1]) {
            return Some(prop);
        }
    }
    None
}

/// Fits a single smooth [`CubicBezier::from_arch`] arc directly from `start_pt` to `end_pt` for
/// standalone 2-point chains with no straight end stubs (so there are zero stub-to-arc kinks).
///
/// # Arguments
/// * `board` - Board state with obstacles.
/// * `drc_ctx` - GPU DRC snapshot for batch arc validation.
/// * `start_pt` - Start endpoint `(x, y)` in `mm`.
/// * `end_pt` - End endpoint `(x, y)` in `mm`.
/// * `width` - Trace width in `mm`.
/// * `layer` - Copper layer index.
/// * `net_id` - Net ID of the trace.
/// * `pref_sign` - Preferred perpendicular bow sign (`1.0` or `-1.0`).
///
/// # Returns
/// `Some(arc)` if a DRC-clean gentle arch fits between `start_pt` and `end_pt`, or `None`.
pub(crate) fn try_fit_standalone_arch(
    board: &Board,
    drc_ctx: &GpuDrcContext,
    start_pt: (f64, f64),
    end_pt: (f64, f64),
    width: f64,
    layer: usize,
    net_id: usize,
    pref_sign: f64,
) -> Option<ArcSegment> {
    let len = (end_pt.0 - start_pt.0).hypot(end_pt.1 - start_pt.1);
    if len < 1.2 {
        return None;
    }
    let base_bow = (len * 0.06).clamp(0.08, 0.65);
    let mut cand_arcs = Vec::with_capacity(10);
    for sign in [pref_sign, -pref_sign] {
        for scale in [1.0, 0.65, 0.40, 0.22, 0.12] {
            let offset = base_bow * scale * sign;
            let bezier = CubicBezier::from_arch(start_pt, end_pt, offset);
            let mut arcs = bezier.to_kicad_arcs(width, layer, net_id);
            if arcs.len() == 1 {
                cand_arcs.push(arcs.pop().unwrap());
            }
        }
    }
    let valid_flags = batch_validate_arcs_gpu(board, drc_ctx, &cand_arcs);
    for (arc, ok) in cand_arcs.into_iter().zip(valid_flags) {
        if ok {
            return Some(arc);
        }
    }
    None
}

/// Replaces interior polyline bends across all unlocked tracks with $G^1$-tangent circular [`ArcSegment`] fillets (`1.0 mm` target radius).
///
/// # Arguments
/// * `board` - Mutable board state whose unlocked tracks are rounded with tangent arcs.
///
/// # Side Effects
/// Mutates `board.tracks` and `board.arcs`.
pub fn apply_any_angle_tangent_arcs(board: &mut Board) {
    apply_tangent_arcs_with_options(board, 1.0, false);
}

/// Internal tangent-arc fitting engine supporting both standard radiused corners and wide organic biarcs/arches.
///
/// # Arguments
/// * `board` - Mutable board state whose unlocked tracks and arcs are updated.
/// * `target_radius` - Target fillet radius in millimeters (`mm`).
/// * `organic_mode` - When `true`, also fits asymmetric corner biarcs and standalone 2-point arches.
///
/// # Side Effects
/// Mutates `board.tracks` and `board.arcs`.
pub(crate) fn apply_tangent_arcs_with_options(board: &mut Board, target_radius: f64, organic_mode: bool) {
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
        let drc_ctx = GpuDrcContext::from_board(board);

        let chains = extract_track_chains_for_organic(&net_tracks, board);
        let chain_endpoints: Vec<((f64, f64), (f64, f64), usize)> = chains
            .iter()
            .filter(|c| c.points.len() >= 2)
            .map(|c| (c.points[0], *c.points.last().unwrap(), c.layer))
            .collect();

        for (c_idx, chain) in chains.into_iter().enumerate() {
            let n = chain.points.len();
            if n < 3 {
                for k in 0..n.saturating_sub(1) {
                    let start_pt = chain.points[k];
                    let end_pt = chain.points[k + 1];
                    if organic_mode {
                        let touches_other_same_layer = chain_endpoints.iter().enumerate().any(|(j, &(s, e, l))| {
                            j != c_idx
                                && l == chain.layer
                                && ((s.0 - start_pt.0).hypot(s.1 - start_pt.1) < 0.02
                                    || (e.0 - start_pt.0).hypot(e.1 - start_pt.1) < 0.02
                                    || (s.0 - end_pt.0).hypot(s.1 - end_pt.1) < 0.02
                                    || (e.0 - end_pt.0).hypot(e.1 - end_pt.1) < 0.02)
                        }) || board.tracks.iter().any(|t| {
                            t.net_id == chain.net_id
                                && t.layer == chain.layer
                                && ((t.start.0 - start_pt.0).hypot(t.start.1 - start_pt.1) < 0.02
                                    || (t.end.0 - start_pt.0).hypot(t.end.1 - start_pt.1) < 0.02
                                    || (t.start.0 - end_pt.0).hypot(t.start.1 - end_pt.1) < 0.02
                                    || (t.end.0 - end_pt.0).hypot(t.end.1 - end_pt.1) < 0.02)
                        });
                        if !touches_other_same_layer {
                            let pref_sign = if ((start_pt.0 * 10.0) as i64
                                + (end_pt.1 * 10.0) as i64
                                + chain.net_id as i64)
                                % 2
                                == 0
                            {
                                1.0
                            } else {
                                -1.0
                            };
                            if let Some(arc) = try_fit_standalone_arch(
                                board,
                                &drc_ctx,
                                start_pt,
                                end_pt,
                                chain.width,
                                chain.layer,
                                chain.net_id,
                                pref_sign,
                            ) {
                                board.arcs.push(arc);
                                continue;
                            }
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
                continue;
            }

            let mut corner_arcs: Vec<Option<((f64, f64), (f64, f64), Vec<ArcSegment>)>> = vec![None; n];
            let mut used_on_seg = vec![0.0f64; n];

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

                let min_stub_a = (l_a * 0.15).clamp(0.05, 0.20);
                let min_stub_b = (l_b * 0.15).clamp(0.05, 0.20);
                let avail_a = (l_a - used_on_seg[i - 1] - min_stub_a).max(0.0);
                let avail_b = if i == n - 2 {
                    (l_b - min_stub_b).max(0.0)
                } else if organic_mode {
                    (l_b * 0.55).min((l_b - min_stub_b).max(0.0))
                } else {
                    l_b * 0.49
                };

                if organic_mode {
                    if let Some((t_in, t_out, bi_arcs, used_b)) = try_fit_asymmetric_corner_biarc(
                        board,
                        &drc_ctx,
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
                let d_max = if organic_mode {
                    avail_a.min(avail_b)
                } else {
                    (l_a * 0.49).min(l_b * 0.49)
                };
                let r_max_geom = d_max * alpha.tan();
                let r_init = target_radius.min(r_max_geom);

                let mut corner_proposals: Vec<((f64, f64), (f64, f64), f64, Vec<ArcSegment>, usize, usize)> = Vec::new();
                let mut flat_corner_arcs: Vec<ArcSegment> = Vec::new();
                for scale in [1.0, 0.88, 0.76, 0.64, 0.52, 0.40, 0.30, 0.20, 0.12, 0.06, 0.03] {
                    let r = r_init * scale;
                    if r < 0.015 {
                        continue;
                    }
                    let d = r / alpha.tan();
                    let t1 = (p_curr.0 + d * u1.0, p_curr.1 + d * u1.1);
                    let t2 = (p_curr.0 + d * u2.0, p_curr.1 + d * u2.1);

                    let bezier = CubicBezier::from_corner(t1, p_curr, t2);
                    let arcs = bezier.to_kicad_arcs(chain.width, chain.layer, chain.net_id);
                    if !arcs.is_empty() {
                        let start_idx = flat_corner_arcs.len();
                        flat_corner_arcs.extend_from_slice(&arcs);
                        let end_idx = flat_corner_arcs.len();
                        corner_proposals.push((t1, t2, d, arcs, start_idx, end_idx));
                    }
                }
                let valid_flags = batch_validate_arcs_gpu(board, &drc_ctx, &flat_corner_arcs);
                for (t1, t2, d, arcs, s_idx, e_idx) in corner_proposals {
                    if valid_flags.get(s_idx..e_idx).is_some_and(|slice| slice.iter().all(|&ok| ok)) {
                        corner_arcs[i] = Some((t1, t2, arcs));
                        used_on_seg[i] = d;
                        break;
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
/// using homotopic rubber-band relaxation and $G^1$-continuous [`CubicBezier`] corner curves.
///
/// Uses `default_acid_trap_fillet_radius` for acute-junction filleting.
///
/// # Arguments
/// * `board` - Mutable board state whose unlocked tracks and arcs are converted.
///
/// # Side Effects
/// Mutates `board.tracks`, `board.arcs`, and `board.vias` (with per-stage electrical connectivity rollback).
pub fn convert_tracks_to_organic(board: &mut Board) {
    convert_tracks_to_organic_with_fillet_radius(board, default_acid_trap_fillet_radius());
}

/// Converts a routed board's unlocked tracks into organic curved traces with an explicit acute-junction `fillet_radius`.
///
/// Snapshot-verifies electrical connectivity after each transformation stage (arc reconstruction,
/// T-junction splitting, homotopic relaxation, tangent-arc fitting, acute-junction filleting,
/// netclass widening, and dangling stub pruning) and automatically restores any net whose
/// pad-to-pad connectivity would be broken.
///
/// # Arguments
/// * `board` - Mutable board state whose unlocked tracks, arcs, and vias are updated.
/// * `fillet_radius` - Target fillet radius in millimeters (`mm`) for acute-angle junctions (`<= 0.0` skips filleting).
///
/// # Side Effects
/// Mutates `board.tracks`, `board.arcs`, and `board.vias`.
pub fn convert_tracks_to_organic_with_fillet_radius(board: &mut Board, fillet_radius: f64) {
    let conns = AutoRouter::new(1).extract_connections(board);
    let pre_connected: Vec<bool> = conns
        .iter()
        .map(|c| AutoRouter::is_connection_electrically_connected(board, c))
        .collect();
    let restore_broken_nets = |board: &mut Board, snap: &(Vec<TrackSegment>, Vec<ArcSegment>, Vec<Via>)| {
        if conns.is_empty() {
            return;
        }
        let mut broken: HashSet<usize> = HashSet::new();
        for (i, c) in conns.iter().enumerate() {
            if pre_connected[i] && !AutoRouter::is_connection_electrically_connected(board, c) {
                broken.insert(c.net_id);
            }
        }
        if !broken.is_empty() {
            board.tracks.retain(|t| !broken.contains(&t.net_id));
            board.arcs.retain(|a| !broken.contains(&a.net_id));
            board.vias.retain(|v| !broken.contains(&v.net_id));
            for t in &snap.0 {
                if broken.contains(&t.net_id) {
                    board.tracks.push(t.clone());
                }
            }
            for a in &snap.1 {
                if broken.contains(&a.net_id) {
                    board.arcs.push(a.clone());
                }
            }
            for v in &snap.2 {
                if broken.contains(&v.net_id) {
                    board.vias.push(v.clone());
                }
            }
        }
    };
    let mut stage_snap = (board.tracks.clone(), board.arcs.clone(), board.vias.clone());

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
    // Split any unlocked segment at mid-segment T-junctions (track endpoints, vias, or pads of the same net)
    // so rubber-band relaxation and tangent-arc fitting treat the junction as a pinned degree>=3 vertex.
    {
        let mut split_pts_by_net: HashMap<usize, Vec<((f64, f64), Option<usize>)>> = HashMap::new();
        for t in &board.tracks {
            split_pts_by_net.entry(t.net_id).or_default().push((t.start, Some(t.layer)));
            split_pts_by_net.entry(t.net_id).or_default().push((t.end, Some(t.layer)));
        }
        for v in &board.vias {
            split_pts_by_net.entry(v.net_id).or_default().push((v.position, None));
        }
        for fp in &board.footprints {
            for p in &fp.pads {
                if p.net_id > 0 {
                    let lyr = if p.is_through_hole { None } else { Some(fp.layer) };
                    split_pts_by_net.entry(p.net_id).or_default().push((fp.pad_absolute_position(p), lyr));
                }
            }
        }
        let mut new_tracks = Vec::with_capacity(board.tracks.len() + 16);
        for seg in board.tracks.drain(..) {
            if seg.is_locked {
                new_tracks.push(seg);
                continue;
            }
            let seg_len = (seg.end.0 - seg.start.0).hypot(seg.end.1 - seg.start.1);
            if seg_len < 0.06 {
                new_tracks.push(seg);
                continue;
            }
            let mut cuts: Vec<(f64, (f64, f64))> = Vec::new();
            if let Some(cands) = split_pts_by_net.get(&seg.net_id) {
                for &(pt, lyr_opt) in cands {
                    if lyr_opt.is_some_and(|l| l != seg.layer) {
                        continue;
                    }
                    let d_s = (pt.0 - seg.start.0).hypot(pt.1 - seg.start.1);
                    let d_e = (pt.0 - seg.end.0).hypot(pt.1 - seg.end.1);
                    if d_s > 0.025 && d_e > 0.025 && d_s < seg_len && d_e < seg_len {
                        if dist_point_to_segment(pt, seg.start, seg.end) < 0.025 {
                            cuts.push((d_s, pt));
                        }
                    }
                }
            }
            if cuts.is_empty() {
                new_tracks.push(seg);
            } else {
                cuts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut cur = seg.start;
                for (_, pt) in cuts {
                    if (pt.0 - cur.0).hypot(pt.1 - cur.1) > 0.02 {
                        new_tracks.push(TrackSegment {
                            start: cur,
                            end: pt,
                            width: seg.width,
                            layer: seg.layer,
                            net_id: seg.net_id,
                            is_locked: false,
                        });
                        cur = pt;
                    }
                }
                if (seg.end.0 - cur.0).hypot(seg.end.1 - cur.1) > 0.02 {
                    new_tracks.push(TrackSegment {
                        start: cur,
                        end: seg.end,
                        width: seg.width,
                        layer: seg.layer,
                        net_id: seg.net_id,
                        is_locked: false,
                    });
                }
            }
        }
        board.tracks = new_tracks;
    }
    restore_broken_nets(board, &stage_snap);
    stage_snap = (board.tracks.clone(), board.arcs.clone(), board.vias.clone());

    crate::memlog::checkpoint_board("convert_tracks_to_organic::before_relax_homotopic", board);
    relax_homotopic_rubber_bands(board);
    restore_broken_nets(board, &stage_snap);
    stage_snap = (board.tracks.clone(), board.arcs.clone(), board.vias.clone());

    crate::memlog::checkpoint_board("convert_tracks_to_organic::before_apply_tangent_arcs", board);
    apply_tangent_arcs_with_options(board, 25.0, true);
    restore_broken_nets(board, &stage_snap);
    stage_snap = (board.tracks.clone(), board.arcs.clone(), board.vias.clone());

    if fillet_radius > 0.0 {
        crate::memlog::checkpoint_board("convert_tracks_to_organic::before_fillet_acute_angle_junctions", board);
        fillet_acute_angle_junctions(board, fillet_radius);
        restore_broken_nets(board, &stage_snap);
        stage_snap = (board.tracks.clone(), board.arcs.clone(), board.vias.clone());
    }
    crate::memlog::checkpoint_board("convert_tracks_to_organic::before_widen_tracks_and_arcs", board);
    widen_tracks_and_arcs_to_netclass(board);
    crate::memlog::checkpoint_board("convert_tracks_to_organic::before_prune_dangling", board);
    prune_dangling_tracks_and_vias(board);
    restore_broken_nets(board, &stage_snap);
    widen_tracks_and_arcs_to_netclass(board);
    restore_broken_nets(board, &stage_snap);
    crate::memlog::checkpoint_board("convert_tracks_to_organic::done", board);
}

/// Computes the unit tangent direction vector leaving `arc` at its start (`at_start == true`) or end point, pointing into the arc interior.
///
/// # Arguments
/// * `arc` - Circular arc segment.
/// * `at_start` - `true` to query the tangent at `arc.start()`, `false` for `arc.end()`.
///
/// # Returns
/// Unit direction vector `(ux, uy)` pointing along the arc away from the queried endpoint.
pub fn arc_arm_dir(arc: &ArcSegment, at_start: bool) -> (f64, f64) {
    let sweep = (arc.end_angle_rad - arc.start_angle_rad).signum();
    let (th, s) = if at_start { (arc.start_angle_rad, sweep) } else { (arc.end_angle_rad, -sweep) };
    (-th.sin() * s, th.cos() * s)
}

/// Replaces unlocked arcs meeting a same-net junction at `< 90°` with straight chords so
/// track-based acute-junction filleting can see and round off forks whose acute side was formed by an arc.
///
/// # Arguments
/// * `board` - Mutable board state whose acute junction arcs are exploded into chord [`TrackSegment`] items.
///
/// # Side Effects
/// Removes qualifying [`ArcSegment`] items from `board.arcs` and appends replacement [`TrackSegment`] chords to `board.tracks`.
pub(crate) fn explode_arcs_at_acute_junctions(board: &mut Board) {
    let near = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).hypot(a.1 - b.1) < 2e-3;
    let mut explode = Vec::new();
    for (ai, arc) in board.arcs.iter().enumerate() {
        if arc.is_locked {
            continue;
        }
        for at_start in [true, false] {
            let p = if at_start { arc.start() } else { arc.end() };
            let u = arc_arm_dir(arc, at_start);
            let mut others: Vec<(f64, f64)> = board
                .tracks
                .iter()
                .filter(|t| t.net_id == arc.net_id && t.layer == arc.layer)
                .filter_map(|t| {
                    let far = if near(t.start, p) { t.end } else if near(t.end, p) { t.start } else { return None };
                    let l = (far.0 - p.0).hypot(far.1 - p.1);
                    (l > 1e-4).then(|| ((far.0 - p.0) / l, (far.1 - p.1) / l))
                })
                .collect();
            for (oi, o) in board.arcs.iter().enumerate() {
                if oi != ai && o.net_id == arc.net_id && o.layer == arc.layer {
                    if near(o.start(), p) {
                        others.push(arc_arm_dir(o, true));
                    } else if near(o.end(), p) {
                        others.push(arc_arm_dir(o, false));
                    }
                }
            }
            if others.iter().any(|o| u.0 * o.0 + u.1 * o.1 > 0.0087) {
                explode.push(ai);
                break;
            }
        }
    }
    for &ai in explode.iter().rev() {
        let arc = board.arcs.remove(ai);
        // One chord gives the fillet a long arm to work with; fall back to finer chords if the
        // shortcut across the bend is not clear.
        let chord_clear = is_segment_clear(board, None, arc.start(), arc.end(), arc.layer, arc.net_id, arc.width);
        let sweep = (arc.end_angle_rad - arc.start_angle_rad).abs();
        let n = if chord_clear { 1 } else { (sweep / 30f64.to_radians()).ceil().max(1.0) as usize };
        let mut chords = arc.to_segments(n);
        // Arc endpoints are recomputed from centre/radius; snap them onto the exact junction
        // coordinates so the vertex-keyed fillet pass sees one junction, not two.
        let snap = |p: (f64, f64), b: &Board| {
            b.tracks
                .iter()
                .filter(|t| t.net_id == arc.net_id && t.layer == arc.layer)
                .flat_map(|t| [t.start, t.end])
                .find(|&q| near(p, q))
                .unwrap_or(p)
        };
        let first = snap(chords[0].start, board);
        let last = snap(chords[n - 1].end, board);
        chords[0].start = first;
        chords[n - 1].end = last;
        board.tracks.extend(chords);
    }
}

/// A rectangular pad corner used as a diagonal track exit, storing the pad's local frame `(cos, sin)` and corner quadrant signs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CornerPort {
    /// World coordinates `(x, y)` in `mm` of the pad corner.
    corner: (f64, f64),
    /// Cosine of the pad's world rotation angle.
    cs: f64,
    /// Sine of the pad's world rotation angle.
    sn: f64,
    /// Local X quadrant sign (`-1` or `+1`) of the corner.
    cx: i8,
    /// Local Y quadrant sign (`-1` or `+1`) of the corner.
    cy: i8,
}

/// Exit "ports" of a rectangular pad in its local frame: sides (outward normal) or corners.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum PadPort {
    /// Orthogonal exit through a pad edge with local outward normal `(nx, ny)`.
    Side(i8, i8),
    /// Diagonal exit through a pad corner in local quadrant `(cx, cy)`.
    Corner(i8, i8),
}

/// Re-anchors tracks at rectangular pads so each leaves cleanly: perpendicular through a side, or
/// with its centerline through a corner (heading inside that corner's 90° cone), one track per
/// port.
///
/// Ensures neither copper edge meets the pad outline at `< 90°` and two tracks never leave an
/// acute wedge between them outside the pad. Pad-center vertices are skipped by the fillet pass,
/// so this provides the acid-trap treatment for pad entries.
///
/// # Arguments
/// * `board` - Mutable board state whose pad-incident tracks are re-anchored.
///
/// # Side Effects
/// Mutates `board.tracks` in place.
// ponytail: tracks only; arcs ending on pads and oval/circle pads (radial exits are already clean) are left alone.
pub(crate) fn align_pad_exits(board: &mut Board) {
    const AXIS_SIN: f64 = 0.035; // ~2 deg
    // (track, pad end is start, new pad-end point, optional leg point, corner + its local frame)
    let mut edits: Vec<(usize, bool, (f64, f64), Option<(f64, f64)>, Option<CornerPort>)> = Vec::new();
    let mut inside: Vec<usize> = Vec::new();
    for fp in &board.footprints {
        for pad in &fp.pads {
            if pad.net_id == 0 || pad.shape == "circle" || pad.shape == "oval" {
                continue;
            }
            let c = fp.pad_absolute_position(pad);
            let flipped = fp.layer != fp.orig_layer;
            let rot = (fp.rotation + if flipped { -pad.rotation } else { pad.rotation }).to_radians();
            let (sn, cs) = rot.sin_cos();
            let to_world = |l: (f64, f64)| (c.0 + l.0 * cs - l.1 * sn, c.1 + l.0 * sn + l.1 * cs);
            let to_local = |w: (f64, f64)| {
                let (x, y) = (w.0 - c.0, w.1 - c.1);
                (x * cs + y * sn, -x * sn + y * cs)
            };
            let (hw, hh) = (pad.width * 0.5, pad.height * 0.5);
            let pad_layer = fp.pad_layer(pad);
            // (track idx, pad end is start, far point, local unit dir)
            let mut arms: Vec<(usize, bool, (f64, f64), (f64, f64))> = Vec::new();
            for (k, t) in board.tracks.iter().enumerate() {
                if t.net_id != pad.net_id || t.is_locked || !(pad.is_through_hole || t.layer == pad_layer) {
                    continue;
                }
                let (s_in, e_in) = (dist_point_to_pad(t.start, fp, pad) < 1e-9, dist_point_to_pad(t.end, fp, pad) < 1e-9);
                let (at_start, far) = match (s_in, e_in) {
                    // Wholly inside the pad: redundant with the pad copper (snap/dedup debris).
                    (true, true) => {
                        inside.push(k);
                        continue;
                    }
                    (true, false) => (true, t.end),
                    (false, true) => (false, t.start),
                    (false, false) => continue,
                };
                let l = to_local(far);
                let n = l.0.hypot(l.1);
                arms.push((k, at_start, far, (l.0 / n, l.1 / n)));
            }
            // Most diagonal tracks claim corners first; axis-aligned ones claim sides.
            arms.sort_by(|a, b| {
                let da = a.3 .0.abs().min(a.3 .1.abs());
                let db = b.3 .0.abs().min(b.3 .1.abs());
                db.partial_cmp(&da).unwrap_or(Ordering::Equal)
            });
            let sg = |v: f64| if v >= 0.0 { 1i8 } else { -1i8 };
            let mut used: Vec<PadPort> = Vec::new();
            // Perpendicular exits per side as (side normal, offset along the side); parallel exits
            // through one side are fine if they don't overlap.
            let mut slides: Vec<((i8, i8), f64)> = Vec::new();
            let slide_free = |slides: &Vec<((i8, i8), f64)>, side: (i8, i8), off: f64, w: f64| {
                !slides.iter().any(|&(s2, o2)| s2 == side && (o2 - off).abs() < w + 0.2)
            };
            for &(k, at_start, far, d) in &arms {
                let w = board.tracks[k].width;
                let (sx, sy) = (sg(d.0), sg(d.1));
                // Best case: the far point lies "in front of" a side, so slide the anchor along the
                // pad and leave perpendicularly without bending the track (short pad-to-via fanouts).
                // A wide track may overhang the pad's end; its edge is then still perpendicular to
                // the side, so no acute notch forms.
                let fl = to_local(far);
                let slide = if fl.1.abs() > hh && fl.0.abs() < hw {
                    Some(((0, sg(fl.1)), fl.0, (fl.0, 0.0)))
                } else if fl.0.abs() > hw && fl.1.abs() < hh {
                    Some(((sg(fl.0), 0), fl.1, (0.0, fl.1)))
                } else {
                    None
                };
                if let Some((side, off, anchor_l)) = slide {
                    if slide_free(&slides, side, off, w) {
                        slides.push((side, off));
                        used.push(PadPort::Side(side.0, side.1));
                        let anchor = to_world(anchor_l);
                        let pad_end = if at_start { board.tracks[k].start } else { board.tracks[k].end };
                        if (pad_end.0 - anchor.0).hypot(pad_end.1 - anchor.1) > 1e-6 {
                            edits.push((k, at_start, anchor, None, None));
                        }
                        continue;
                    }
                }
                let prefs = if d.1.abs() < AXIS_SIN {
                    vec![PadPort::Side(sx, 0), PadPort::Corner(sx, 1), PadPort::Corner(sx, -1)]
                } else if d.0.abs() < AXIS_SIN {
                    vec![PadPort::Side(0, sy), PadPort::Corner(1, sy), PadPort::Corner(-1, sy)]
                } else {
                    vec![PadPort::Corner(sx, sy), PadPort::Side(sx, 0), PadPort::Side(0, sy)]
                };
                for &port in &prefs {
                    if let PadPort::Side(nx, ny) = port {
                        if !slide_free(&slides, (nx, ny), 0.0, w) {
                            continue;
                        }
                    } else if used.contains(&port) {
                        continue;
                    }
                    // Leg exit along side (nx, ny): anchor slid along the side towards the far point
                    // (clamped inside the pad), leave perpendicular, then turn <= 90 deg towards it.
                    let side_leg = |nx: i8, ny: i8| -> ((f64, f64), (f64, f64), f64) {
                        let lf = to_local(far);
                        let (ext, along, span) = if nx != 0 { (hw, lf.1, hh) } else { (hh, lf.0, hw) };
                        let off = along.clamp(-span * 0.9, span * 0.9);
                        let o = ext + w * 0.5 + 0.05;
                        let (a_l, out_l) = if nx != 0 {
                            ((0.0, off), (nx as f64 * o, off))
                        } else {
                            ((off, 0.0), (off, ny as f64 * o))
                        };
                        (to_world(a_l), to_world(out_l), off)
                    };
                    // None = port unusable, Some(None) = already clean, Some(Some(p)) = re-anchor at p.
                    let edit: Option<Option<(f64, f64)>> = match port {
                        // Already leaving perpendicularly from the centre.
                        PadPort::Side(..) if port == prefs[0] => Some(None),
                        PadPort::Side(nx, ny) => {
                            let (_, out, _) = side_leg(nx, ny);
                            let lf = to_local(far);
                            let lo = to_local(out);
                            ((lf.0 - lo.0) * nx as f64 + (lf.1 - lo.1) * ny as f64 > 0.0).then_some(Some(out))
                        }
                        PadPort::Corner(cx, cy) => {
                            let corner = to_world((cx as f64 * hw, cy as f64 * hh));
                            let l = (far.0 - corner.0).hypot(far.1 - corner.1);
                            let dw = ((far.0 - corner.0) / l, (far.1 - corner.1) / l);
                            let dl = (dw.0 * cs + dw.1 * sn, -dw.0 * sn + dw.1 * cs);
                            let anchor = (corner.0 - dw.0 * w * 0.5, corner.1 - dw.1 * w * 0.5);
                            let in_cone = dl.0 * cx as f64 >= -1e-9 && dl.1 * cy as f64 >= -1e-9;
                            (in_cone && dist_point_to_pad(anchor, fp, pad) < 1e-9).then_some(Some(anchor))
                        }
                    };
                    let Some(edit) = edit else { continue };
                    used.push(port);
                    if let PadPort::Side(nx, ny) = port {
                        slides.push(((nx, ny), if port == prefs[0] { 0.0 } else { side_leg(nx, ny).2 }));
                    }
                    let pad_end = if at_start { board.tracks[k].start } else { board.tracks[k].end };
                    match (port, edit) {
                        // Perpendicular side exit needs the track to start from the pad centre.
                        (_, None) if (pad_end.0 - c.0).hypot(pad_end.1 - c.1) > 1e-6 => {
                            edits.push((k, at_start, c, None, None))
                        }
                        (_, None) => {}
                        (PadPort::Corner(cx, cy), Some(anchor)) => {
                            let corner = to_world((cx as f64 * hw, cy as f64 * hh));
                            edits.push((k, at_start, anchor, None, Some(CornerPort { corner, cs, sn, cx, cy })))
                        }
                        (PadPort::Side(nx, ny), Some(out)) => edits.push((k, at_start, side_leg(nx, ny).0, Some(out), None)),
                    }
                    break;
                }
            }
        }
    }
    // A pad-to-pad track can be re-anchored at both ends: merge edits per track into one polyline.
    // A corner exit must aim at whatever the track now reaches on its other end (the other pad's
    // corner or leg point), so re-aim it and drop it if that leaves the corner's cone.
    /// Re-anchored endpoint descriptor `(pad_end_pt, optional_leg_pt, optional_corner_port)`.
    type End = ((f64, f64), Option<(f64, f64)>, Option<CornerPort>);
    let mut per_track: HashMap<usize, [Option<End>; 2]> = HashMap::new();
    for (k, at_start, pad_end, leg, corner) in edits {
        per_track.entry(k).or_default()[if at_start { 0 } else { 1 }] = Some((pad_end, leg, corner));
    }
    let mut merged: Vec<(usize, [Option<((f64, f64), Option<(f64, f64)>)>; 2])> = Vec::new();
    for (k, mut ends) in per_track {
        let t = &board.tracks[k];
        let w = t.width;
        // Point each end's first piece heads towards, as seen from the pad end.
        let target = |ends: &[Option<End>; 2], i: usize| -> (f64, f64) {
            let other = 1 - i;
            match ends[other] {
                Some((_, Some(leg), _)) => leg,
                Some((_, None, Some(cp))) => cp.corner,
                Some((p, None, None)) => p,
                None => if other == 0 { t.start } else { t.end },
            }
        };
        for i in 0..2 {
            if let Some((_, None, Some(cp))) = ends[i] {
                let f = target(&ends, i);
                let l = (f.0 - cp.corner.0).hypot(f.1 - cp.corner.1);
                let d = ((f.0 - cp.corner.0) / l, (f.1 - cp.corner.1) / l);
                let dl = (d.0 * cp.cs + d.1 * cp.sn, -d.0 * cp.sn + d.1 * cp.cs);
                ends[i] = (dl.0 * cp.cx as f64 >= -1e-9 && dl.1 * cp.cy as f64 >= -1e-9)
                    .then(|| ((cp.corner.0 - d.0 * w * 0.5, cp.corner.1 - d.1 * w * 0.5), None, Some(cp)));
            }
        }
        merged.push((k, [ends[0].map(|e| (e.0, e.1)), ends[1].map(|e| (e.0, e.1))]));
    }
    let mut remove: Vec<usize> = Vec::new();
    let mut add = Vec::new();
    for (k, ends) in merged {
        if ends[0].is_none() && ends[1].is_none() {
            continue;
        }
        let t = board.tracks[k].clone();
        let mut pts = Vec::new();
        match ends[0] {
            Some((p, leg)) => {
                pts.push(p);
                pts.extend(leg);
            }
            None => pts.push(t.start),
        }
        match ends[1] {
            Some((p, leg)) => {
                pts.extend(leg);
                pts.push(p);
            }
            None => pts.push(t.end),
        }
        let pieces: Vec<((f64, f64), (f64, f64))> = pts.windows(2).map(|w| (w[0], w[1])).collect();
        if pieces.iter().all(|&(a, b)| is_segment_clear(board, None, a, b, t.layer, t.net_id, t.width)) {
            remove.push(k);
            add.extend(pieces.into_iter().map(|(a, b)| TrackSegment { start: a, end: b, ..t.clone() }));
        }
    }
    remove.extend(inside);
    remove.sort_unstable();
    remove.dedup();
    for k in remove.into_iter().rev() {
        board.tracks.remove(k);
    }
    board.tracks.extend(add);
}

/// Automatically detects and fillets any acute-angle (`< 90°`) copper wedges (acid traps)
/// at degree-2 corners and degree-3+ forks / T-junctions across all unlocked tracks.
///
/// - At a degree-3+ fork where a branch $B \to P$ joins arm $A \to P$ at $\theta < 90^\circ$, the
///   branch is bent with radius up to `fillet_radius` so it lands on the trunk at $90^\circ$ (a
///   tangential merge would still leave an acute notch at the copper edges).
/// - At a degree-2 acute corner $A \to P \to B$ ($\theta < 90^\circ$), the sharp apex $P$ is replaced
///   by a $G^1$-tangent circular fillet [`ArcSegment`] (or a mitered obtuse bridge if an arc is blocked).
///
/// # Arguments
/// * `board` - Mutable board state whose tracks and arcs are filleted.
/// * `fillet_radius` - Target fillet radius in millimeters (`mm`).
///
/// # Side Effects
/// Mutates `board.tracks` and `board.arcs`.
pub fn fillet_acute_angle_junctions(board: &mut Board, fillet_radius: f64) {
    if fillet_radius <= 1e-4 {
        return;
    }
    align_pad_exits(board);
    explode_arcs_at_acute_junctions(board);
    if board.tracks.len() < 2 {
        return;
    }

    split_and_dedup_same_net_tracks(board);

    let q_vtx = |p: (f64, f64)| -> (i64, i64) {
        ((p.0 * 500.0).round() as i64, (p.1 * 500.0).round() as i64)
    };

    for _pass in 0..8 {
        let mut changed = false;
        let num_tracks = board.tracks.len();
        let mut adj: HashMap<(usize, usize, (i64, i64)), Vec<(usize, bool)>> = HashMap::new();
        for (idx, t) in board.tracks.iter().enumerate() {
            if t.net_id == 0 {
                continue;
            }
            let len = (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1);
            if len <= 1e-4 {
                continue;
            }
            adj.entry((t.net_id, t.layer, q_vtx(t.start)))
                .or_default()
                .push((idx, true));
            adj.entry((t.net_id, t.layer, q_vtx(t.end)))
                .or_default()
                .push((idx, false));
        }

        let mut keys: Vec<(usize, usize, (i64, i64))> = adj
            .iter()
            .filter(|(_, inc)| inc.len() >= 2)
            .map(|(&k, _)| k)
            .collect();
        keys.sort_unstable();

        let mut modified_tracks: HashSet<usize> = HashSet::new();
        let mut tracks_to_remove: HashSet<usize> = HashSet::new();
        let mut tracks_to_add: Vec<TrackSegment> = Vec::new();
        let mut arcs_to_add: Vec<ArcSegment> = Vec::new();
        let drc_ctx = GpuDrcContext::from_board(board);

        for key in keys {
            let (net_id, layer, _) = key;
            let inc = &adj[&key];
            if inc.iter().any(|&(ti, _)| modified_tracks.contains(&ti)) {
                continue;
            }

            // Compute shared vertex coordinate P
            let (t0_idx, t0_is_start) = inc[0];
            let p_vtx = if t0_is_start {
                board.tracks[t0_idx].start
            } else {
                board.tracks[t0_idx].end
            };

            // Outgoing vectors from p_vtx for each incident segment
            let mut arms: Vec<(usize, bool, (f64, f64), (f64, f64), f64, f64, bool)> = Vec::with_capacity(inc.len());
            for &(ti, is_start) in inc {
                let trk = &board.tracks[ti];
                let far_pt = if is_start { trk.end } else { trk.start };
                let vx = far_pt.0 - p_vtx.0;
                let vy = far_pt.1 - p_vtx.1;
                let len = vx.hypot(vy);
                if len > 1e-4 {
                    arms.push((
                        ti,
                        is_start,
                        far_pt,
                        (vx / len, vy / len),
                        len,
                        trk.width,
                        trk.is_locked,
                    ));
                }
            }
            if arms.len() < 2 {
                continue;
            }

            // Find the sharpest acute pair (theta < 89.5 deg -> cos(theta) > 0.0087)
            let mut best_pair: Option<(usize, usize, f64)> = None;
            for i in 0..arms.len() {
                for j in (i + 1)..arms.len() {
                    if arms[i].6 && arms[j].6 {
                        continue;
                    }
                    let u1 = arms[i].3;
                    let u2 = arms[j].3;
                    let dot = (u1.0 * u2.0 + u1.1 * u2.1).clamp(-1.0, 1.0);
                    if dot > 0.0087 {
                        let theta = dot.acos();
                        if best_pair.map_or(true, |(_, _, best_th)| theta < best_th) {
                            best_pair = Some((i, j, theta));
                        }
                    }
                }
            }

            let Some((ai, aj, theta)) = best_pair else {
                continue;
            };

            // Degenerate overlapping fold-back (theta < 2.5 deg)
            if theta < 0.045 {
                let (shorter_arm, longer_arm) = if arms[ai].4 <= arms[aj].4 {
                    (arms[ai], arms[aj])
                } else {
                    (arms[aj], arms[ai])
                };
                if !shorter_arm.6 {
                    let far_short = shorter_arm.2;
                    let far_long = longer_arm.2;
                    let w_bridge = shorter_arm.5.min(longer_arm.5);
                    if arms.len() >= 3 {
                        // At a fork, shorter_arm is completely redundant or can snap onto longer_arm
                        let proj_d = shorter_arm.4.min(longer_arm.4);
                        let snap_pt = (
                            p_vtx.0 + proj_d * longer_arm.3 .0,
                            p_vtx.1 + proj_d * longer_arm.3 .1,
                        );
                        tracks_to_remove.insert(shorter_arm.0);
                        modified_tracks.insert(shorter_arm.0);
                        if (far_short.0 - snap_pt.0).hypot(far_short.1 - snap_pt.1) > 1e-3 {
                            tracks_to_add.push(TrackSegment {
                                start: far_short,
                                end: snap_pt,
                                width: w_bridge,
                                layer,
                                net_id,
                                is_locked: false,
                            });
                        }
                        if !longer_arm.6 && proj_d > 0.01 && proj_d + 0.01 < longer_arm.4 {
                            tracks_to_remove.insert(longer_arm.0);
                            modified_tracks.insert(longer_arm.0);
                            tracks_to_add.push(TrackSegment {
                                start: p_vtx,
                                end: snap_pt,
                                width: longer_arm.5,
                                layer,
                                net_id,
                                is_locked: false,
                            });
                            tracks_to_add.push(TrackSegment {
                                start: snap_pt,
                                end: far_long,
                                width: longer_arm.5,
                                layer,
                                net_id,
                                is_locked: false,
                            });
                        }
                        changed = true;
                        continue;
                    } else if (far_short.0 - far_long.0).hypot(far_short.1 - far_long.1) > 1e-3
                        && is_segment_clear(board, None, far_short, far_long, layer, net_id, w_bridge)
                    {
                        tracks_to_remove.insert(shorter_arm.0);
                        tracks_to_remove.insert(longer_arm.0);
                        modified_tracks.insert(shorter_arm.0);
                        modified_tracks.insert(longer_arm.0);
                        tracks_to_add.push(TrackSegment {
                            start: far_short,
                            end: far_long,
                            width: w_bridge,
                            layer,
                            net_id,
                            is_locked: false,
                        });
                        changed = true;
                        continue;
                    }
                }
            }

            if arms.len() == 2 {
                // Degree-2 acute corner at p_vtx
                let vtx_anchored = board.vias.iter().any(|v| {
                    v.net_id == net_id && (v.position.0 - p_vtx.0).hypot(v.position.1 - p_vtx.1) < 0.05
                }) || board.footprints.iter().any(|fp| {
                    fp.pads.iter().any(|p| {
                        p.net_id == net_id
                            && (p.is_through_hole || fp.layer == layer)
                            && {
                                let (px, py) = fp.pad_absolute_position(p);
                                (px - p_vtx.0).hypot(py - p_vtx.1) < 0.05
                            }
                    })
                });
                if vtx_anchored || arms[ai].6 || arms[aj].6 {
                    continue;
                }

                let arm1 = arms[ai];
                let arm2 = arms[aj];
                let u1 = arm1.3;
                let u2 = arm2.3;
                let l1 = arm1.4;
                let l2 = arm2.4;
                let w = arm1.5.min(arm2.5);
                let alpha = (theta * 0.5).max(0.03);
                let d_max = (l1 * 0.85).min(l2 * 0.85);
                let r_target = fillet_radius.min(d_max * alpha.tan());

                let mut fitted = false;
                let mut deg2_cands: Vec<((f64, f64), (f64, f64), ArcSegment)> = Vec::new();
                let mut deg2_arcs: Vec<ArcSegment> = Vec::new();
                for scale in [1.0, 0.85, 0.70, 0.55, 0.40, 0.28, 0.18, 0.10, 0.05] {
                    let r = r_target * scale;
                    let d = r / alpha.tan();
                    if r < 0.015 || d < 0.015 || d > l1 - 0.01 || d > l2 - 0.01 {
                        continue;
                    }
                    let t1 = (p_vtx.0 + d * u1.0, p_vtx.1 + d * u1.1);
                    let t2 = (p_vtx.0 + d * u2.0, p_vtx.1 + d * u2.1);
                    let b_sum = (u1.0 + u2.0, u1.1 + u2.1);
                    let b_len = b_sum.0.hypot(b_sum.1);
                    if b_len <= 1e-6 {
                        continue;
                    }
                    let b = (b_sum.0 / b_len, b_sum.1 / b_len);
                    let h = r / alpha.sin();
                    let c = (p_vtx.0 + h * b.0, p_vtx.1 + h * b.1);
                    let m_arc = (c.0 - r * b.0, c.1 - r * b.1);
                    if let Some(arc) = ArcSegment::from_3pts(t1, m_arc, t2, w, layer, net_id) {
                        deg2_arcs.push(arc.clone());
                        deg2_cands.push((t1, t2, arc));
                    }
                }
                let deg2_valid = batch_validate_arcs_gpu(board, &drc_ctx, &deg2_arcs);
                for ((t1, t2, arc), ok) in deg2_cands.into_iter().zip(deg2_valid) {
                    if ok {
                        tracks_to_remove.insert(arm1.0);
                        tracks_to_remove.insert(arm2.0);
                        modified_tracks.insert(arm1.0);
                        modified_tracks.insert(arm2.0);
                        if (arm1.2 .0 - t1.0).hypot(arm1.2 .1 - t1.1) > 1e-4 {
                            tracks_to_add.push(TrackSegment {
                                start: arm1.2,
                                end: t1,
                                width: arm1.5,
                                layer,
                                net_id,
                                is_locked: false,
                            });
                        }
                        if (t2.0 - arm2.2 .0).hypot(t2.1 - arm2.2 .1) > 1e-4 {
                            tracks_to_add.push(TrackSegment {
                                start: t2,
                                end: arm2.2,
                                width: arm2.5,
                                layer,
                                net_id,
                                is_locked: false,
                            });
                        }
                        arcs_to_add.push(arc);
                        fitted = true;
                        changed = true;
                        break;
                    }
                }

                if !fitted {
                    // Fallback: direct chord or obtuse bevel across acute corner
                    for frac in [0.65, 0.45, 0.30, 0.18] {
                        let d = (l1.min(l2) * frac).min(fillet_radius * 1.5);
                        if d < 0.02 || d > l1 - 0.01 || d > l2 - 0.01 {
                            continue;
                        }
                        let t1 = (p_vtx.0 + d * u1.0, p_vtx.1 + d * u1.1);
                        let t2 = (p_vtx.0 + d * u2.0, p_vtx.1 + d * u2.1);
                        if is_segment_clear(board, None, t1, t2, layer, net_id, w) {
                            tracks_to_remove.insert(arm1.0);
                            tracks_to_remove.insert(arm2.0);
                            modified_tracks.insert(arm1.0);
                            modified_tracks.insert(arm2.0);
                            if (arm1.2 .0 - t1.0).hypot(arm1.2 .1 - t1.1) > 1e-4 {
                                tracks_to_add.push(TrackSegment {
                                    start: arm1.2,
                                    end: t1,
                                    width: arm1.5,
                                    layer,
                                    net_id,
                                    is_locked: false,
                                });
                            }
                            tracks_to_add.push(TrackSegment {
                                start: t1,
                                end: t2,
                                width: w,
                                layer,
                                net_id,
                                is_locked: false,
                            });
                            if (t2.0 - arm2.2 .0).hypot(t2.1 - arm2.2 .1) > 1e-4 {
                                tracks_to_add.push(TrackSegment {
                                    start: t2,
                                    end: arm2.2,
                                    width: arm2.5,
                                    layer,
                                    net_id,
                                    is_locked: false,
                                });
                            }
                            changed = true;
                            break;
                        }
                    }
                }
            } else {
                // Degree-3+ fork / T-junction at p_vtx:
                // Choose which of (ai, aj) is the branch B (whose join at p_vtx will be detached and
                // redirected into a smooth G1 tangent fillet arc merging onto trunk arm A at T_A)
                // and which is the trunk arm A (which stays connected to p_vtx via T_A -> p_vtx).
                let score_as_trunk = |idx: usize| -> f64 {
                    let u = arms[idx].3;
                    let mut min_other_dot = 1.0f64;
                    for (k, other) in arms.iter().enumerate() {
                        if k != ai && k != aj {
                            let d = u.0 * other.3 .0 + u.1 * other.3 .1;
                            min_other_dot = min_other_dot.min(d);
                        }
                    }
                    // More negative dot with a 3rd arm means closer to a 180-deg straight trunk
                    (-min_other_dot) * 2.0 + arms[idx].4 * 0.25 + if arms[idx].6 { 100.0 } else { 0.0 }
                };

                let (branch_arm, trunk_arm) = if arms[ai].6 {
                    (arms[aj], arms[ai])
                } else if arms[aj].6 {
                    (arms[ai], arms[aj])
                } else if score_as_trunk(ai) >= score_as_trunk(aj) {
                    (arms[aj], arms[ai])
                } else {
                    (arms[ai], arms[aj])
                };

                if branch_arm.6 {
                    continue;
                }

                let u_b = branch_arm.3;
                let u_a = trunk_arm.3;
                let l_b = branch_arm.4;
                let l_a = trunk_arm.4;
                let w_arc = branch_arm.5.min(trunk_arm.5);
                // A fillet that merges tangentially into the trunk still leaves an acute notch at the
                // copper edges (~56 deg for r = 0.35, w = 0.2). Instead, bend the branch with radius
                // r so it lands on the trunk at 90 deg: the arc leaves the branch tangentially at Q and
                // its centre C lies on the trunk line, so it meets the trunk perpendicularly at F.
                //   d = r / tan(theta), |pC| = d / cos(theta), |pF| = d (1 - sin(theta)) / cos(theta)
                let (sin_t, cos_t) = theta.sin_cos();
                let mut fitted = false;
                let mut fork_cands: Vec<((f64, f64), (f64, f64), ArcSegment)> = Vec::new();
                let mut fork_arcs: Vec<ArcSegment> = Vec::new();
                for scale in [1.0, 0.85, 0.70, 0.55, 0.40, 0.28, 0.18, 0.10, 0.05] {
                    let r = fillet_radius * scale;
                    let d = r * cos_t / sin_t.max(1e-6);
                    let f_dist = d * (1.0 - sin_t) / cos_t.max(1e-6);
                    if r < 0.015 || d < 0.015 || d > l_b - 0.005 || f_dist > l_a - 0.025 {
                        continue;
                    }
                    let q = (p_vtx.0 + d * u_b.0, p_vtx.1 + d * u_b.1);
                    let c_dist = d / cos_t;
                    let c = (p_vtx.0 + c_dist * u_a.0, p_vtx.1 + c_dist * u_a.1);
                    let f = if f_dist < 1e-3 { p_vtx } else { (p_vtx.0 + f_dist * u_a.0, p_vtx.1 + f_dist * u_a.1) };
                    let (mq, mf) = (((q.0 - c.0) / r, (q.1 - c.1) / r), ((f.0 - c.0) / r, (f.1 - c.1) / r));
                    let bis = (mq.0 + mf.0, mq.1 + mf.1);
                    let bl = bis.0.hypot(bis.1);
                    if bl <= 1e-6 {
                        continue;
                    }
                    let m_arc = (c.0 + r * bis.0 / bl, c.1 + r * bis.1 / bl);
                    let Some(arc) = ArcSegment::from_3pts(q, m_arc, f, w_arc, layer, net_id) else {
                        continue;
                    };
                    fork_arcs.push(arc.clone());
                    fork_cands.push((q, f, arc));
                }
                let fork_valid = batch_validate_arcs_gpu(board, &drc_ctx, &fork_arcs);
                for ((q, f, arc), ok) in fork_cands.into_iter().zip(fork_valid) {
                    if !ok {
                        continue;
                    }
                    tracks_to_remove.insert(branch_arm.0);
                    modified_tracks.insert(branch_arm.0);
                    if (branch_arm.2 .0 - q.0).hypot(branch_arm.2 .1 - q.1) > 1e-4 {
                        tracks_to_add.push(TrackSegment {
                            start: branch_arm.2,
                            end: q,
                            width: branch_arm.5,
                            layer,
                            net_id,
                            is_locked: false,
                        });
                    }
                    arcs_to_add.push(arc);
                    if f != p_vtx && !trunk_arm.6 {
                        tracks_to_remove.insert(trunk_arm.0);
                        modified_tracks.insert(trunk_arm.0);
                        tracks_to_add.push(TrackSegment { start: p_vtx, end: f, width: trunk_arm.5, layer, net_id, is_locked: false });
                        tracks_to_add.push(TrackSegment { start: f, end: trunk_arm.2, width: trunk_arm.5, layer, net_id, is_locked: false });
                    }
                    fitted = true;
                    changed = true;
                    break;
                }

                if !fitted && l_a > 0.055 {
                    // Fallback for short fork arms: re-aim branch_arm to join trunk_arm perpendicularly (90 deg)
                    // at projection point P_perp = p_vtx + d_proj * u_a
                    let d_proj = (l_b * theta.cos()).clamp(0.025, l_a - 0.025);
                    if d_proj > 0.02 && d_proj < l_a - 0.02 {
                        let t_a = (p_vtx.0 + d_proj * u_a.0, p_vtx.1 + d_proj * u_a.1);
                        if (branch_arm.2 .0 - t_a.0).hypot(branch_arm.2 .1 - t_a.1) > 1e-4
                            && is_segment_clear(board, None, branch_arm.2, t_a, layer, net_id, branch_arm.5)
                        {
                            tracks_to_remove.insert(branch_arm.0);
                            modified_tracks.insert(branch_arm.0);
                            tracks_to_add.push(TrackSegment {
                                start: branch_arm.2,
                                end: t_a,
                                width: branch_arm.5,
                                layer,
                                net_id,
                                is_locked: false,
                            });
                            if !trunk_arm.6 {
                                tracks_to_remove.insert(trunk_arm.0);
                                modified_tracks.insert(trunk_arm.0);
                                tracks_to_add.push(TrackSegment {
                                    start: p_vtx,
                                    end: t_a,
                                    width: trunk_arm.5,
                                    layer,
                                    net_id,
                                    is_locked: false,
                                });
                                tracks_to_add.push(TrackSegment {
                                    start: t_a,
                                    end: trunk_arm.2,
                                    width: trunk_arm.5,
                                    layer,
                                    net_id,
                                    is_locked: false,
                                });
                            }
                            changed = true;
                        }
                    }
                }
            }
        }

        if !changed {
            break;
        }

        let mut next_tracks = Vec::with_capacity(num_tracks + tracks_to_add.len());
        for (idx, trk) in board.tracks.drain(..).enumerate() {
            if !tracks_to_remove.contains(&idx) {
                next_tracks.push(trk);
            }
        }
        next_tracks.extend(tracks_to_add);
        board.tracks = next_tracks;
        board.arcs.extend(arcs_to_add);
    }
}

/// Adjusts unlocked track and arc widths across `board` to match netclass target widths while respecting DRC clearances.
///
/// Performs four stages:
/// 1. Necks down any segment or arc whose current width violates copper or drill-hole clearance.
/// 2. Micro-nudges vias and interior vertices that slightly encroach on clearances.
/// 3. Widens wide power/ground traces (`target_width > 0.25 mm`) up toward their netclass width where clearance allows.
/// 4. Harmonizes widths across connected components on each `(net_id, layer)` so traces maintain a uniform body width (allowing only short terminal pad-entry stubs to remain necked).
///
/// # Arguments
/// * `board` - Mutable board state whose tracks, arcs, and vias are adjusted.
///
/// # Side Effects
/// Mutates `board.tracks`, `board.arcs`, and `board.vias`.
pub(crate) fn widen_tracks_and_arcs_to_netclass(board: &mut Board) {
    // 1. Neck down any track or arc whose current width violates netclass or hole clearance
    let drc_ctx = GpuDrcContext::from_board(board);
    let init_arc_ok = batch_validate_arcs_gpu(board, &drc_ctx, &board.arcs);
    for i in 0..board.arcs.len() {
        if board.arcs[i].is_locked {
            continue;
        }
        let arc = board.arcs[i].clone();
        let min_w = board.get_netclass_for_net(arc.net_id).min_track_width.max(0.15);
        if arc.width > min_w + 1e-3 && !init_arc_ok.get(i).copied().unwrap_or(false) {
            for &cand_w in &[0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.18, min_w] {
                if cand_w >= arc.width - 1e-3 || cand_w < min_w - 1e-3 {
                    continue;
                }
                let mut cand_arc = arc.clone();
                cand_arc.width = cand_w;
                if is_arc_clear(board, &cand_arc) {
                    board.arcs[i].width = cand_w;
                    break;
                }
            }
            if !is_arc_clear(board, &board.arcs[i].clone()) {
                board.arcs[i].width = min_w;
            }
        }
    }

    for i in 0..board.tracks.len() {
        if board.tracks[i].is_locked {
            continue;
        }
        let trk = board.tracks[i].clone();
        let min_w = board.get_netclass_for_net(trk.net_id).min_track_width.max(0.15);
        if trk.width > min_w + 1e-3
            && !is_segment_clear(board, None, trk.start, trk.end, trk.layer, trk.net_id, trk.width)
        {
            let mut healed = false;
            for &cand_w in &[0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.18, min_w] {
                if cand_w >= trk.width - 1e-3 || cand_w < min_w - 1e-3 {
                    continue;
                }
                if is_segment_clear(board, None, trk.start, trk.end, trk.layer, trk.net_id, cand_w) {
                    board.tracks[i].width = cand_w;
                    healed = true;
                    break;
                }
            }
            if !healed {
                board.tracks[i].width = min_w;
            }
        }
    }

    // 2. Micro-nudge any via that slightly violates hole_to_hole, hole_clearance, or copper clearance
    let dirs_16: [(f64, f64); 16] = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071),
        (0.9239, 0.3827), (-0.9239, 0.3827), (0.9239, -0.3827), (-0.9239, -0.3827),
        (0.3827, 0.9239), (-0.3827, 0.9239), (0.3827, -0.9239), (-0.3827, -0.9239),
    ];
    for vi in 0..board.vias.len() {
        if board.vias[vi].is_locked {
            continue;
        }
        let v = board.vias[vi].clone();
        let v_clr = board.get_netclass_for_net(v.net_id).clearance;
        if is_via_location_legal_ignoring(board, v.position, v.size, v.net_id, v_clr, Some(v.position)) {
            continue;
        }
        'via_nudge: for &r in &[0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.22, 0.28, 0.35, 0.45, 0.60] {
            for &(dx, dy) in &dirs_16 {
                let cand = (v.position.0 + dx * r, v.position.1 + dy * r);
                if !is_via_location_legal_ignoring(board, cand, v.size, v.net_id, v_clr, Some(v.position)) {
                    continue;
                }
                let tracks_ok = board.tracks.iter().all(|t| {
                    if t.net_id != v.net_id {
                        return true;
                    }
                    let s_on = (t.start.0 - v.position.0).hypot(t.start.1 - v.position.1) < 1e-3;
                    let e_on = (t.end.0 - v.position.0).hypot(t.end.1 - v.position.1) < 1e-3;
                    if !s_on && !e_on {
                        return true;
                    }
                    let ns = if s_on { cand } else { t.start };
                    let ne = if e_on { cand } else { t.end };
                    is_track_segment_legal(board, ns, ne, t.width, t.layer, t.net_id, v_clr)
                });
                let arcs_ok = !board.arcs.iter().any(|a| {
                    a.net_id == v.net_id && {
                        let as_pt = a.start();
                        let ae_pt = a.end();
                        (as_pt.0 - v.position.0).hypot(as_pt.1 - v.position.1) < 1e-2
                            || (ae_pt.0 - v.position.0).hypot(ae_pt.1 - v.position.1) < 1e-2
                            || board.tracks.iter().any(|t| {
                                t.net_id == v.net_id && t.layer == a.layer && {
                                    let s_on = (t.start.0 - v.position.0).hypot(t.start.1 - v.position.1) < 1e-3;
                                    let e_on = (t.end.0 - v.position.0).hypot(t.end.1 - v.position.1) < 1e-3;
                                    if s_on {
                                        (as_pt.0 - t.end.0).hypot(as_pt.1 - t.end.1) < 0.02
                                            || (ae_pt.0 - t.end.0).hypot(ae_pt.1 - t.end.1) < 0.02
                                    } else if e_on {
                                        (as_pt.0 - t.start.0).hypot(as_pt.1 - t.start.1) < 0.02
                                            || (ae_pt.0 - t.start.0).hypot(ae_pt.1 - t.start.1) < 0.02
                                    } else {
                                        false
                                    }
                                }
                            })
                    }
                });
                if tracks_ok && arcs_ok {
                    for t in &mut board.tracks {
                        if t.net_id == v.net_id {
                            if (t.start.0 - v.position.0).hypot(t.start.1 - v.position.1) < 1e-3 {
                                t.start = cand;
                            }
                            if (t.end.0 - v.position.0).hypot(t.end.1 - v.position.1) < 1e-3 {
                                t.end = cand;
                            }
                        }
                    }
                    board.vias[vi].position = cand;
                    break 'via_nudge;
                }
            }
        }
    }

    // 3. Micro-nudge any shared interior track vertex whose incident segment slightly violates clearance
    let is_anchored = |pt: (f64, f64), layer: usize, net_id: usize, b: &Board| -> bool {
        b.vias.iter().any(|v| v.net_id == net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
            || b.footprints.iter().any(|fp| {
                fp.pads.iter().any(|p| {
                    p.net_id == net_id
                        && (p.is_through_hole || fp.layer == layer)
                        && {
                            let (px, py) = fp.pad_absolute_position(p);
                            (px - pt.0).hypot(py - pt.1) < 0.05
                        }
                })
            })
            || b.arcs.iter().any(|a| {
                a.net_id == net_id && a.layer == layer && {
                    let s = a.start();
                    let e = a.end();
                    (s.0 - pt.0).hypot(s.1 - pt.1) < 0.02 || (e.0 - pt.0).hypot(e.1 - pt.1) < 0.02
                }
            })
    };
    let n_trks = board.tracks.len();
    for i in 0..n_trks {
        if board.tracks[i].is_locked {
            continue;
        }
        let ti = board.tracks[i].clone();
        if is_segment_clear(board, None, ti.start, ti.end, ti.layer, ti.net_id, ti.width) {
            continue;
        }
        for &end_is_end in &[false, true] {
            let v_pt = if end_is_end { ti.end } else { ti.start };
            if is_anchored(v_pt, ti.layer, ti.net_id, board) {
                continue;
            }
            let incident: Vec<(usize, bool)> = (0..n_trks)
                .filter_map(|k| {
                    let tk = &board.tracks[k];
                    if tk.net_id != ti.net_id || tk.layer != ti.layer || tk.is_locked {
                        return None;
                    }
                    if (tk.start.0 - v_pt.0).hypot(tk.start.1 - v_pt.1) < 1e-3 {
                        Some((k, false))
                    } else if (tk.end.0 - v_pt.0).hypot(tk.end.1 - v_pt.1) < 1e-3 {
                        Some((k, true))
                    } else {
                        None
                    }
                })
                .collect();
            if incident.is_empty()
                || incident.len() > 3
                || incident.iter().any(|&(k, is_e)| {
                    let tk = &board.tracks[k];
                    let other_pt = if is_e { tk.start } else { tk.end };
                    board.arcs.iter().any(|a| {
                        a.net_id == tk.net_id && a.layer == tk.layer && {
                            let s = a.start();
                            let e = a.end();
                            (s.0 - other_pt.0).hypot(s.1 - other_pt.1) < 0.02
                                || (e.0 - other_pt.0).hypot(e.1 - other_pt.1) < 0.02
                        }
                    })
                })
            {
                continue;
            }
            'vtx_nudge: for &r in &[0.015, 0.03, 0.05, 0.08, 0.12, 0.16, 0.22] {
                for &(dx, dy) in &dirs_16 {
                    let cand = (v_pt.0 + dx * r, v_pt.1 + dy * r);
                    let all_clear = incident.iter().all(|&(k, is_e)| {
                        let tk = &board.tracks[k];
                        let ns = if is_e { tk.start } else { cand };
                        let ne = if is_e { cand } else { tk.end };
                        is_segment_clear(board, None, ns, ne, tk.layer, tk.net_id, tk.width)
                    });
                    if all_clear {
                        for &(k, is_e) in &incident {
                            if is_e {
                                board.tracks[k].end = cand;
                            } else {
                                board.tracks[k].start = cand;
                            }
                        }
                        break 'vtx_nudge;
                    }
                }
            }
        }
    }

    // 4a. Widen wide power nets (nc_w > 0.25) per-segment up to netclass target width where clearance permits
    for i in 0..board.arcs.len() {
        if board.arcs[i].is_locked {
            continue;
        }
        let net_id = board.arcs[i].net_id;
        let cur_w = board.arcs[i].width;
        let nc_w = net_target_width(board, net_id);
        if nc_w <= 0.25 + 1e-3 || nc_w <= cur_w + 1e-3 {
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

    let max_board_clr = board
        .netclasses
        .values()
        .map(|c| c.clearance)
        .fold(0.25, f64::max);
    for i in 0..board.tracks.len() {
        if board.tracks[i].is_locked {
            continue;
        }
        let trk = board.tracks[i].clone();
        let nc_w = net_target_width(board, trk.net_id);
        if nc_w <= 0.25 + 1e-3 || nc_w <= trk.width + 1e-3 {
            continue;
        }
        let net_clr = board.get_netclass_for_net(trk.net_id).clearance;
        let t_min_x = trk.start.0.min(trk.end.0);
        let t_max_x = trk.start.0.max(trk.end.0);
        let t_min_y = trk.start.1.min(trk.end.1);
        let t_max_y = trk.start.1.max(trk.end.1);
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
                let os_pt = a.start();
                let om_pt = a.mid();
                let oe_pt = a.end();
                let o_min_x = os_pt.0.min(om_pt.0).min(oe_pt.0);
                let o_max_x = os_pt.0.max(om_pt.0).max(oe_pt.0);
                let o_min_y = os_pt.1.min(om_pt.1).min(oe_pt.1);
                let o_max_y = os_pt.1.max(om_pt.1).max(oe_pt.1);
                let sag = 0.5 * (om_pt.0 - os_pt.0).hypot(om_pt.1 - os_pt.1)
                    + a.width * 0.5
                    + cand_w * 0.5
                    + max_board_clr
                    + 0.05;
                if o_max_x + sag < t_min_x
                    || o_min_x - sag > t_max_x
                    || o_max_y + sag < t_min_y
                    || o_min_y - sag > t_max_y
                {
                    return true;
                }
                let a_clr = board.get_netclass_for_net(a.net_id).clearance;
                let req = cand_w * 0.5 + a.width * 0.5 + net_clr.max(a_clr);
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

    // 4a'. A wide net necked only because its via was dropped beside a foreign pad (e.g. a +3.3V
    // via wedged against a decoupling cap's GND pad): slide the via so every incident track fits
    // at the target width, instead of letting 4b narrow the whole chain.
    for vi in 0..board.vias.len() {
        let v = board.vias[vi].clone();
        if v.is_locked {
            continue;
        }
        let tw = net_target_width(board, v.net_id);
        if tw <= 0.25 + 1e-3 {
            continue;
        }
        let at_via = |p: (f64, f64)| (p.0 - v.position.0).hypot(p.1 - v.position.1) < 1e-3;
        let incident: Vec<usize> = (0..board.tracks.len())
            .filter(|&k| {
                let t = &board.tracks[k];
                t.net_id == v.net_id && !t.is_locked && (at_via(t.start) || at_via(t.end))
            })
            .collect();
        if incident.is_empty() || incident.iter().all(|&k| board.tracks[k].width >= tw - 1e-3) {
            continue;
        }
        if board.arcs.iter().any(|a| a.net_id == v.net_id && (at_via(a.start()) || at_via(a.end()))) {
            continue;
        }
        let v_clr = board.get_netclass_for_net(v.net_id).clearance;
        let moved = |cand: (f64, f64), t: &TrackSegment| {
            (if at_via(t.start) { cand } else { t.start }, if at_via(t.end) { cand } else { t.end })
        };
        'slide: for &r in &[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40] {
            for &(dx, dy) in &dirs_16 {
                let cand = (v.position.0 + dx * r, v.position.1 + dy * r);
                if r > 0.0 && !is_via_location_legal_ignoring(board, cand, v.size, v.net_id, v_clr, Some(v.position)) {
                    continue;
                }
                let fits = incident.iter().all(|&k| {
                    let t = &board.tracks[k];
                    let (s, e) = moved(cand, t);
                    is_segment_clear(board, None, s, e, t.layer, t.net_id, tw.max(t.width))
                });
                if fits {
                    for &k in &incident {
                        let (s, e) = moved(cand, &board.tracks[k]);
                        let t = &mut board.tracks[k];
                        t.start = s;
                        t.end = e;
                        t.width = tw.max(t.width);
                    }
                    board.vias[vi].position = cand;
                    break 'slide;
                }
                if r == 0.0 {
                    break;
                }
            }
        }
    }

    // 4b. Enforce uniform width per connected component on each (net_id, layer) across ALL nets
    // (both signal and power nets) so traces never step through varying widths mid-span.
    // Only short terminal pad-entry stubs (< 0.85mm directly touching a pad) may remain necked down
    // when the rest of the trace can legally use a wider uniform width.
    let mut net_layers: Vec<(usize, usize)> = Vec::new();
    for t in &board.tracks {
        if !t.is_locked {
            net_layers.push((t.net_id, t.layer));
        }
    }
    for a in &board.arcs {
        if !a.is_locked {
            net_layers.push((a.net_id, a.layer));
        }
    }
    net_layers.sort_unstable();
    net_layers.dedup();

    for (net_id, layer) in net_layers {
        let t_indices: Vec<usize> = board
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.is_locked && t.net_id == net_id && t.layer == layer)
            .map(|(i, _)| i)
            .collect();
        let a_indices: Vec<usize> = board
            .arcs
            .iter()
            .enumerate()
            .filter(|(_, a)| !a.is_locked && a.net_id == net_id && a.layer == layer)
            .map(|(i, _)| i)
            .collect();
        let n_t = t_indices.len();
        let n_a = a_indices.len();
        let total = n_t + n_a;
        if total == 0 {
            continue;
        }

        let mut endpoints: Vec<((f64, f64), (f64, f64))> = Vec::with_capacity(total);
        for &ti in &t_indices {
            endpoints.push((board.tracks[ti].start, board.tracks[ti].end));
        }
        for &ai in &a_indices {
            endpoints.push((board.arcs[ai].start(), board.arcs[ai].end()));
        }

        let mut terminal_anchors: Vec<(f64, f64)> = Vec::new();
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id == net_id && (pad.is_through_hole || fp.pad_layer(pad) == layer) {
                    terminal_anchors.push(fp.pad_absolute_position(pad));
                }
            }
        }
        for v in &board.vias {
            if v.net_id == net_id {
                terminal_anchors.push(v.position);
            }
        }
        let is_terminal_anchor = |pt: (f64, f64)| -> bool {
            terminal_anchors.iter().any(|&(ax, ay)| (pt.0 - ax).hypot(pt.1 - ay) < 0.03)
        };

        let mut visited = vec![false; total];
        for start_k in 0..total {
            if visited[start_k] {
                continue;
            }
            visited[start_k] = true;
            let mut comp = vec![start_k];
            let mut q = 0;
            while q < comp.len() {
                let u = comp[q];
                q += 1;
                let (us, ue) = endpoints[u];
                for v in 0..total {
                    if !visited[v] {
                        let (vs, ve) = endpoints[v];
                        let shared_non_anchor = ((us.0 - vs.0).hypot(us.1 - vs.1) < 0.025 && !is_terminal_anchor(us))
                            || ((us.0 - ve.0).hypot(us.1 - ve.1) < 0.025 && !is_terminal_anchor(us))
                            || ((ue.0 - vs.0).hypot(ue.1 - vs.1) < 0.025 && !is_terminal_anchor(ue))
                            || ((ue.0 - ve.0).hypot(ue.1 - ve.1) < 0.025 && !is_terminal_anchor(ue));
                        if shared_non_anchor {
                            visited[v] = true;
                            comp.push(v);
                        }
                    }
                }
            }

            if comp.len() <= 1 {
                continue;
            }

            // Identify if an element in `comp` is a short terminal pad-entry stub (< 0.85mm touching a pad).
            // Diff pairs fan out of fine-pitch connectors with the partner's escape alongside, so their
            // first segment (< 3mm, starting within 2mm of a pad) may neck without narrowing the whole pair.
            let is_pair_net = board.nets.iter().any(|o| AutoRouter::are_diff_pair_mates(&board.nets[net_id], o));
            let is_terminal_pad_stub = |k: usize| -> bool {
                if k >= n_t {
                    return false;
                }
                let trk = &board.tracks[t_indices[k]];
                let len = (trk.end.0 - trk.start.0).hypot(trk.end.1 - trk.start.1);
                let (max_len, reach) = if is_pair_net { (3.0, 2.0) } else { (0.85, 0.03) };
                if len > max_len {
                    return false;
                }
                board.footprints.iter().any(|fp| {
                    (fp.layer == layer || fp.pads.iter().any(|p| p.is_through_hole))
                        && fp.pads.iter().any(|pad| {
                            if pad.net_id != net_id || (!pad.is_through_hole && fp.layer != layer) {
                                return false;
                            }
                            let (px, py) = fp.pad_absolute_position(pad);
                            (trk.start.0 - px).hypot(trk.start.1 - py) < reach
                                || (trk.end.0 - px).hypot(trk.end.1 - py) < reach
                        })
                })
            };

            let elem_w = |k: usize| -> f64 {
                if k < n_t {
                    board.tracks[t_indices[k]].width
                } else {
                    board.arcs[a_indices[k - n_t]].width
                }
            };

            let comp_info: Vec<( usize, bool, f64)> = comp
                .iter()
                .map(|&k| (k, is_terminal_pad_stub(k), elem_w(k)))
                .collect();

            let min_all_w = comp_info.iter().map(|&(_, _, w)| w).fold(f64::INFINITY, f64::min);
            let min_body_w = comp_info
                .iter()
                .filter(|&&(_, is_stub, _)| !is_stub)
                .map(|&(_, _, w)| w)
                .fold(f64::INFINITY, f64::min);

            // Diff pairs: one neck at a fanout must not narrow the whole pair; re-widen each
            // element to the pair width wherever it is clear and leave genuine necks local.
            if is_pair_net {
                let tw = net_target_width(board, net_id);
                for &(k, _, w) in &comp_info {
                    if w >= tw - 1e-3 {
                        continue;
                    }
                    if k < n_t {
                        let t = board.tracks[t_indices[k]].clone();
                        if is_segment_clear(board, None, t.start, t.end, t.layer, t.net_id, tw) {
                            board.tracks[t_indices[k]].width = tw;
                        }
                    } else {
                        let mut a = board.arcs[a_indices[k - n_t]].clone();
                        a.width = tw;
                        if is_arc_clear(board, &a) {
                            board.arcs[a_indices[k - n_t]].width = tw;
                        }
                    }
                }
                continue;
            }

            if min_body_w.is_finite() {
                for &(k, is_stub, w) in &comp_info {
                    let target_w = if is_stub { w.min(min_body_w) } else { min_body_w };
                    if k < n_t {
                        board.tracks[t_indices[k]].width = target_w;
                    } else {
                        board.arcs[a_indices[k - n_t]].width = target_w;
                    }
                }
            } else {
                for &(k, _, _) in &comp_info {
                    if k < n_t {
                        board.tracks[t_indices[k]].width = min_all_w;
                    } else {
                        board.arcs[a_indices[k - n_t]].width = min_all_w;
                    }
                }
            }
        }
    }
}

/// Converts a board's organic curved/any-angle traces back into standard 45°/90° octilinear straight tracks.
///
/// Replaces all unlocked [`ArcSegment`] items with straight chords or discretized segments,
/// enforces the 45° octilinear dogleg grid across all chains, runs standard octilinear glossing,
/// harmonizes netclass trace widths, and prunes dangling stubs.
///
/// # Arguments
/// * `board` - Mutable board state whose unlocked arcs and tracks are converted to straight octilinear segments.
/// * `grid` - Routing grid used for clearance validation during conversion and glossing.
///
/// # Side Effects
/// Mutates `board.arcs`, `board.tracks`, and `board.vias`.
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
