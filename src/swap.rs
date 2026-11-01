// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva

use std::collections::{HashMap, HashSet};
use crate::board::{Board, PinSwapRecord, Point, Segment};
use crate::placer::PlacementConfig;

/// Solves the linear sum assignment problem (minimum cost bipartite matching)
/// for an N x M cost matrix (where N <= M) using the Kuhn-Munkres (Hungarian) algorithm.
/// Returns a vector `matching` of length N where `matching[i]` is the column assigned to row `i`.
pub fn hungarian_min_assignment(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    let m = cost[0].len();
    if n > m {
        // Transpose the matrix so rows <= cols, then invert assignment
        let mut transposed = vec![vec![0.0; n]; m];
        for i in 0..n {
            for j in 0..m {
                transposed[j][i] = cost[i][j];
            }
        }
        let col_match = hungarian_min_assignment(&transposed);
        let mut row_match = vec![0usize; n];
        for (col, &row) in col_match.iter().enumerate() {
            if row < n {
                row_match[row] = col;
            }
        }
        return row_match;
    }

    // 1-indexed potentials and min-slack augmenting path algorithm
    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;

            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut result = vec![0usize; n];
    for j in 1..=m {
        if p[j] > 0 {
            result[p[j] - 1] = j - 1;
        }
    }
    result
}

/// Optimizes pin and gate swapping across the board using bipartite matching (Hungarian algorithm)
/// and 2-opt airwire crossing untangling.
///
/// Returns the total number of swaps performed.
pub fn optimize_pin_and_gate_swapping(board: &mut Board, config: &PlacementConfig) -> usize {
    let mut total_swaps = 0;

    // 1. Ensure all footprints have their swappable pin groups inferred if not explicitly specified
    for fp in &mut board.footprints {
        fp.infer_swap_groups();
    }

    // 2. Perform Gate Swapping (macro-level functional gate unit matching)
    total_swaps += optimize_gate_swapping_internal(board, config);

    // 3. Iteratively perform Pin Swapping until convergence (up to 4 passes)
    for _ in 0..4 {
        let pass_swaps = optimize_pin_swapping_internal(board, config);
        if pass_swaps == 0 {
            break;
        }
        total_swaps += pass_swaps;
    }

    total_swaps
}

/// Internal helper to perform functional gate unit swapping.
fn optimize_gate_swapping_internal(board: &mut Board, _config: &PlacementConfig) -> usize {
    let mut total_swaps = 0;
    let pad_positions: Vec<Vec<Point>> = board
        .footprints
        .iter()
        .map(|fp| {
            (0..fp.pads.len())
                .map(|pi| fp.pad_world_point(pi, Point::new(fp.x, fp.y), fp.rotation))
                .collect()
        })
        .collect();

    for fp_idx in 0..board.footprints.len() {
        let gate_groups = board.footprints[fp_idx].gate_swap_groups.clone();
        for group in gate_groups {
            let g_count = group.gate_units.len();
            if g_count < 2 {
                continue;
            }
            let unit_len = group.gate_units[0].len();
            if unit_len == 0 {
                continue;
            }
            let fp = &board.footprints[fp_idx];
            if !group.gate_units.iter().all(|u| u.len() == unit_len && u.iter().all(|&pi| pi < fp.pads.len())) {
                continue;
            }

            // Find remote targets for each unit's pins
            let mut unit_targets: Vec<Vec<Option<Point>>> = vec![vec![None; unit_len]; g_count];
            for (u, unit) in group.gate_units.iter().enumerate() {
                for (l, &pi) in unit.iter().enumerate() {
                    let net_id = fp.pads[pi].net_id;
                    if net_id == 0 {
                        continue;
                    }
                    let mut remote_pts = Vec::new();
                    for (other_fp_idx, other_fp) in board.footprints.iter().enumerate() {
                        for (other_pi, other_pad) in other_fp.pads.iter().enumerate() {
                            if other_fp_idx == fp_idx && unit.contains(&other_pi) {
                                continue;
                            }
                            if other_pad.net_id == net_id {
                                remote_pts.push(pad_positions[other_fp_idx][other_pi]);
                            }
                        }
                    }
                    if !remote_pts.is_empty() {
                        let cx = remote_pts.iter().map(|p| p.x).sum::<f64>() / remote_pts.len() as f64;
                        let cy = remote_pts.iter().map(|p| p.y).sum::<f64>() / remote_pts.len() as f64;
                        unit_targets[u][l] = Some(Point::new(cx, cy));
                    }
                }
            }

            // Build cost matrix: cost[v][u] = cost of placing netset v on physical gate unit u
            let mut cost_matrix = vec![vec![0.0; g_count]; g_count];
            for v in 0..g_count {
                for u in 0..g_count {
                    let mut sum_d = 0.0;
                    for l in 0..unit_len {
                        let pad_pt = pad_positions[fp_idx][group.gate_units[u][l]];
                        if let Some(target) = unit_targets[v][l] {
                            sum_d += pad_pt.distance_to(&target);
                        }
                    }
                    cost_matrix[v][u] = sum_d;
                }
            }

            let assignment = hungarian_min_assignment(&cost_matrix);
            let is_identity = assignment.iter().enumerate().all(|(v, &u)| v == u);
            if is_identity {
                continue;
            }

            let old_cost: f64 = (0..g_count).map(|v| cost_matrix[v][v]).sum();
            let new_cost: f64 = (0..g_count).map(|v| cost_matrix[v][assignment[v]]).sum();
            if new_cost < old_cost - 1e-4 {
                let old_nets: Vec<Vec<(usize, String)>> = group.gate_units
                    .iter()
                    .map(|unit| {
                        unit.iter().map(|&pi| (board.footprints[fp_idx].pads[pi].net_id, board.footprints[fp_idx].pads[pi].net_name.clone())).collect()
                    })
                    .collect();

                for v in 0..g_count {
                    let u = assignment[v];
                    for l in 0..unit_len {
                        let pad_idx = group.gate_units[u][l];
                        board.footprints[fp_idx].pads[pad_idx].net_id = old_nets[v][l].0;
                        board.footprints[fp_idx].pads[pad_idx].net_name = old_nets[v][l].1.clone();
                    }
                }

                board.pin_swap_history.push(PinSwapRecord {
                    footprint_ref: board.footprints[fp_idx].reference.clone(),
                    pad1_number: format!("GateUnit_{}", assignment[0]),
                    pad1_net: "swapped".to_string(),
                    pad2_number: format!("GateUnit_{}", 0),
                    pad2_net: "swapped".to_string(),
                    reason: format!("Gate unit swapping minimized netset wirelength by {:.2}mm", old_cost - new_cost),
                });
                total_swaps += 1;
            }
        }
    }
    total_swaps
}

/// Internal helper to perform pin swapping within swappable pin groups.
fn optimize_pin_swapping_internal(board: &mut Board, _config: &PlacementConfig) -> usize {
    let mut total_swaps = 0;
    let pad_positions: Vec<Vec<Point>> = board
        .footprints
        .iter()
        .map(|fp| {
            (0..fp.pads.len())
                .map(|pi| fp.pad_world_point(pi, Point::new(fp.x, fp.y), fp.rotation))
                .collect()
        })
        .collect();

    for fp_idx in 0..board.footprints.len() {
        let pin_groups = board.footprints[fp_idx].pin_swap_groups.clone();
        for group in pin_groups {
            if group.len() < 2 {
                continue;
            }
            let fp = &board.footprints[fp_idx];
            if !group.iter().all(|&pi| pi < fp.pads.len()) {
                continue;
            }

            struct PinNetInfo {
                group_idx: usize,
                _pad_idx: usize,
                net_id: usize,
                net_name: String,
                target: Option<Point>,
            }

            let mut pin_infos: Vec<PinNetInfo> = Vec::new();
            for (g_idx, &pi) in group.iter().enumerate() {
                let pad = &fp.pads[pi];
                let net_id = pad.net_id;
                let mut target = None;
                if net_id != 0 && !pad.net_name.is_empty() {
                    let mut remote_pts = Vec::new();
                    for (other_fp_idx, other_fp) in board.footprints.iter().enumerate() {
                        for (other_pi, other_pad) in other_fp.pads.iter().enumerate() {
                            if other_fp_idx == fp_idx && group.contains(&other_pi) {
                                continue;
                            }
                            if other_pad.net_id == net_id {
                                remote_pts.push(pad_positions[other_fp_idx][other_pi]);
                            }
                        }
                    }
                    if !remote_pts.is_empty() {
                        let cx = remote_pts.iter().map(|p| p.x).sum::<f64>() / remote_pts.len() as f64;
                        let cy = remote_pts.iter().map(|p| p.y).sum::<f64>() / remote_pts.len() as f64;
                        target = Some(Point::new(cx, cy));
                    }
                }
                pin_infos.push(PinNetInfo {
                    group_idx: g_idx,
                    _pad_idx: pi,
                    net_id,
                    net_name: pad.net_name.clone(),
                    target,
                });
            }

            let active_indices: Vec<usize> = (0..pin_infos.len())
                .filter(|&i| pin_infos[i].target.is_some())
                .collect();

            if active_indices.len() < 2 {
                continue;
            }

            let m = active_indices.len();
            let k = group.len();
            let mut cost_matrix = vec![vec![0.0; k]; m];
            for (r, &act_idx) in active_indices.iter().enumerate() {
                let target = pin_infos[act_idx].target.unwrap();
                for c in 0..k {
                    let pad_pt = pad_positions[fp_idx][group[c]];
                    cost_matrix[r][c] = target.distance_to(&pad_pt);
                }
            }

            let mut assignment = hungarian_min_assignment(&cost_matrix);

            // Untangle any remaining airwire crossings with 2-opt
            let mut untangle_improved = true;
            while untangle_improved {
                untangle_improved = false;
                for i in 0..m {
                    for j in (i + 1)..m {
                        let c_i = assignment[i];
                        let c_j = assignment[j];
                        let pt_i = pad_positions[fp_idx][group[c_i]];
                        let pt_j = pad_positions[fp_idx][group[c_j]];
                        let tgt_i = pin_infos[active_indices[i]].target.unwrap();
                        let tgt_j = pin_infos[active_indices[j]].target.unwrap();

                        let s1 = Segment { p1: pt_i, p2: tgt_i, net_id: pin_infos[active_indices[i]].net_id };
                        let s2 = Segment { p1: pt_j, p2: tgt_j, net_id: pin_infos[active_indices[j]].net_id };
                        if s1.crosses(&s2) {
                            assignment.swap(i, j);
                            untangle_improved = true;
                        }
                    }
                }
            }

            let mut is_different = false;
            for (r, &act_idx) in active_indices.iter().enumerate() {
                if assignment[r] != pin_infos[act_idx].group_idx {
                    is_different = true;
                    break;
                }
            }
            if !is_different {
                continue;
            }

            let mut old_dist = 0.0;
            let mut old_crossings = 0usize;
            for (r1, &act_idx1) in active_indices.iter().enumerate() {
                let p1 = pad_positions[fp_idx][group[pin_infos[act_idx1].group_idx]];
                let t1 = pin_infos[act_idx1].target.unwrap();
                old_dist += p1.distance_to(&t1);
                let s1 = Segment { p1, p2: t1, net_id: pin_infos[act_idx1].net_id };
                for (_r2, &act_idx2) in active_indices.iter().enumerate().skip(r1 + 1) {
                    let p2 = pad_positions[fp_idx][group[pin_infos[act_idx2].group_idx]];
                    let t2 = pin_infos[act_idx2].target.unwrap();
                    let s2 = Segment { p1: p2, p2: t2, net_id: pin_infos[act_idx2].net_id };
                    if s1.crosses(&s2) {
                        old_crossings += 1;
                    }
                }
            }

            let mut new_dist = 0.0;
            let mut new_crossings = 0usize;
            for r1 in 0..m {
                let p1 = pad_positions[fp_idx][group[assignment[r1]]];
                let t1 = pin_infos[active_indices[r1]].target.unwrap();
                new_dist += p1.distance_to(&t1);
                let s1 = Segment { p1, p2: t1, net_id: pin_infos[active_indices[r1]].net_id };
                for r2 in (r1 + 1)..m {
                    let p2 = pad_positions[fp_idx][group[assignment[r2]]];
                    let t2 = pin_infos[active_indices[r2]].target.unwrap();
                    let s2 = Segment { p1: p2, p2: t2, net_id: pin_infos[active_indices[r2]].net_id };
                    if s1.crosses(&s2) {
                        new_crossings += 1;
                    }
                }
            }

            if new_crossings < old_crossings || (new_crossings == old_crossings && new_dist < old_dist - 1e-4) {
                let assigned_pads_set: HashSet<usize> = assignment.iter().copied().collect();
                let unassigned_pads: Vec<usize> = (0..k).filter(|c| !assigned_pads_set.contains(c)).collect();
                let unassigned_nets: Vec<usize> = (0..pin_infos.len()).filter(|i| !active_indices.contains(i)).collect();

                let mut pad_to_net: HashMap<usize, (usize, String)> = HashMap::new();
                for (r, &act_idx) in active_indices.iter().enumerate() {
                    let pad_group_idx = assignment[r];
                    pad_to_net.insert(pad_group_idx, (pin_infos[act_idx].net_id, pin_infos[act_idx].net_name.clone()));
                }
                for (pad_g_idx, net_idx) in unassigned_pads.into_iter().zip(unassigned_nets.into_iter()) {
                    pad_to_net.insert(pad_g_idx, (pin_infos[net_idx].net_id, pin_infos[net_idx].net_name.clone()));
                }

                let mut group_swaps = 0;
                let mut swapped_pad_numbers = Vec::new();
                for g_idx in 0..k {
                    let pi = group[g_idx];
                    let old_net_id = board.footprints[fp_idx].pads[pi].net_id;
                    if let Some((new_id, new_name)) = pad_to_net.get(&g_idx) {
                        if *new_id != old_net_id {
                            board.footprints[fp_idx].pads[pi].net_id = *new_id;
                            board.footprints[fp_idx].pads[pi].net_name = new_name.clone();
                            swapped_pad_numbers.push(board.footprints[fp_idx].pads[pi].number.clone());
                            group_swaps += 1;
                        }
                    }
                }

                if group_swaps > 0 {
                    let pairwise_swaps = (group_swaps + 1) / 2;
                    total_swaps += pairwise_swaps;
                    let p1_num = swapped_pad_numbers.first().cloned().unwrap_or_else(|| "1".to_string());
                    let p2_num = swapped_pad_numbers.get(1).cloned().unwrap_or_else(|| "2".to_string());
                    board.pin_swap_history.push(PinSwapRecord {
                        footprint_ref: board.footprints[fp_idx].reference.clone(),
                        pad1_number: p1_num,
                        pad1_net: "swapped".to_string(),
                        pad2_number: p2_num,
                        pad2_net: "swapped".to_string(),
                        reason: format!(
                            "Bipartite pin swap: crossings {}->{}, wirelength {:.2}mm->{:.2}mm",
                            old_crossings, new_crossings, old_dist, new_dist
                        ),
                    });
                }
            }
        }
    }
    total_swaps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Footprint, Pad, Point, Segment, SwappableGateGroup};
    use crate::placer::PlacementConfig;

    fn make_test_board() -> Board {
        let mut board = Board::default();
        board.nets = vec![
            "".to_string(),
            "NET_A".to_string(),
            "NET_B".to_string(),
            "NET_C".to_string(),
            "NET_D".to_string(),
        ];
        board.net_name_to_id.insert("NET_A".to_string(), 1);
        board.net_name_to_id.insert("NET_B".to_string(), 2);
        board.net_name_to_id.insert("NET_C".to_string(), 3);
        board.net_name_to_id.insert("NET_D".to_string(), 4);
        board
    }

    #[test]
    fn test_hungarian_min_assignment_basic() {
        // 2x2 matrix
        let cost = vec![
            vec![20.0, 10.0],
            vec![5.0, 30.0],
        ];
        let matching = hungarian_min_assignment(&cost);
        assert_eq!(matching, vec![1, 0]);

        // Rectangular 2x3 matrix
        let cost_rect = vec![
            vec![100.0, 10.0, 100.0],
            vec![100.0, 100.0, 5.0],
        ];
        let matching_rect = hungarian_min_assignment(&cost_rect);
        assert_eq!(matching_rect, vec![1, 2]);
    }

    #[test]
    fn test_pin_swapping_untangles_crossing_pairs() {
        let mut board = make_test_board();

        // U1 has 2 swappable pins: Pad 0 at (10, 8), Pad 1 at (10, 12)
        // Connected in crossed configuration to remote targets:
        // Pad 0 (at 10, 8) has NET_B, pointing to B_REMOTE at (30, 12)
        // Pad 1 (at 10, 12) has NET_A, pointing to A_REMOTE at (30, 8)
        let u1 = Footprint {
            reference: "U1".to_string(),
            value: "IC".to_string(),
            x: 10.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                Pad::new("1", 0.0, -2.0, 0.5, 0.5, 2, "NET_B", false),
                Pad::new("2", 0.0, 2.0, 0.5, 0.5, 1, "NET_A", false),
            ],
            width: 2.0,
            height: 6.0,
            courtyards: Vec::new(),
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: vec![vec![0, 1]],
            gate_swap_groups: Vec::new(),
        };

        let a_remote = Footprint {
            reference: "A_REMOTE".to_string(),
            value: "HDR".to_string(),
            x: 30.0,
            y: 8.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: true,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: Vec::new(),
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let b_remote = Footprint {
            reference: "B_REMOTE".to_string(),
            value: "HDR".to_string(),
            x: 30.0,
            y: 12.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: true,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "NET_B", false)],
            width: 1.0,
            height: 1.0,
            courtyards: Vec::new(),
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        board.footprints = vec![u1, a_remote, b_remote];

        // Check initial crossings
        let seg1 = Segment { p1: Point::new(10.0, 12.0), p2: Point::new(30.0, 8.0), net_id: 1 };
        let seg2 = Segment { p1: Point::new(10.0, 8.0), p2: Point::new(30.0, 12.0), net_id: 2 };
        assert!(seg1.crosses(&seg2), "Initial configuration must cross");

        let swaps = optimize_pin_and_gate_swapping(&mut board, &PlacementConfig::default());
        assert_eq!(swaps, 1, "Must perform exactly 1 swap");

        // Verify pins untangled
        assert_eq!(board.footprints[0].pads[0].net_name, "NET_A");
        assert_eq!(board.footprints[0].pads[0].net_id, 1);
        assert_eq!(board.footprints[0].pads[1].net_name, "NET_B");
        assert_eq!(board.footprints[0].pads[1].net_id, 2);

        // Verify new segments do NOT cross
        let new_seg1 = Segment { p1: Point::new(10.0, 8.0), p2: Point::new(30.0, 8.0), net_id: 1 };
        let new_seg2 = Segment { p1: Point::new(10.0, 12.0), p2: Point::new(30.0, 12.0), net_id: 2 };
        assert!(!new_seg1.crosses(&new_seg2), "Untangled configuration must have 0 crossings");
        assert!(new_seg1.length() + new_seg2.length() < seg1.length() + seg2.length());
    }

    #[test]
    fn test_passive_pin_swap_direction() {
        let mut board = make_test_board();

        // 2-pin resistor R1 placed at (20, 20):
        // Pad 0 at (19, 20) on NET_A, but NET_A target is to the right at (40, 20)
        // Pad 1 at (21, 20) on NET_B, but NET_B target is to the left at (0, 20)
        let r1 = Footprint {
            reference: "R1".to_string(),
            value: "10k".to_string(),
            x: 20.0,
            y: 20.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                Pad::new("1", -1.0, 0.0, 0.6, 0.6, 1, "NET_A", false),
                Pad::new("2", 1.0, 0.0, 0.6, 0.6, 2, "NET_B", false),
            ],
            width: 2.6,
            height: 1.0,
            courtyards: Vec::new(),
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(), // Will be auto-inferred!
            gate_swap_groups: Vec::new(),
        };

        let src = Footprint {
            reference: "U_SRC".to_string(),
            value: "SRC".to_string(),
            x: 40.0,
            y: 20.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: true,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: Vec::new(),
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let sink = Footprint {
            reference: "U_SINK".to_string(),
            value: "SINK".to_string(),
            x: 0.0,
            y: 20.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: true,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "NET_B", false)],
            width: 1.0,
            height: 1.0,
            courtyards: Vec::new(),
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        board.footprints = vec![r1, src, sink];

        let initial_len = (19.0f64 - 40.0).abs() + (21.0f64 - 0.0).abs(); // 21 + 21 = 42
        assert_eq!(initial_len, 42.0);

        let swaps = optimize_pin_and_gate_swapping(&mut board, &PlacementConfig::default());
        assert_eq!(swaps, 1, "Passive pins must swap orientation to eliminate airwire twist");

        // Pad 0 (left at 19, 20) now connected to NET_B (pointing to SINK at 0, 20, dist=19)
        // Pad 1 (right at 21, 20) now connected to NET_A (pointing to SRC at 40, 20, dist=19)
        assert_eq!(board.footprints[0].pads[0].net_name, "NET_B");
        assert_eq!(board.footprints[0].pads[1].net_name, "NET_A");

        let new_len = (19.0f64 - 0.0).abs() + (21.0f64 - 40.0).abs(); // 19 + 19 = 38
        assert_eq!(new_len, 38.0);
        assert!(new_len < initial_len);
    }

    #[test]
    fn test_gpio_bank_bipartite_matching() {
        let mut board = make_test_board();

        // MCU U1 has 4 GPIO pins at y = [10.0, 12.0, 14.0, 16.0]
        // Currently connected in reverse order:
        // Pad 0 (y=10) has NET_D (target at y=16)
        // Pad 1 (y=12) has NET_C (target at y=14)
        // Pad 2 (y=14) has NET_B (target at y=12)
        // Pad 3 (y=16) has NET_A (target at y=10)
        let mut pads = Vec::new();
        pads.push(Pad {
            number: "1".to_string(),
            shape: "rect".to_string(),
            offset_x: 0.0,
            offset_y: 0.0,
            width: 0.5,
            height: 0.5,
            rotation: 0.0,
            drill: None,
            net_id: 4,
            net_name: "NET_D".to_string(),
            is_through_hole: false,
            pin_name: "GPIO0".to_string(),
            swap_group: None,
            net_span: None,
        });
        pads.push(Pad {
            number: "2".to_string(),
            shape: "rect".to_string(),
            offset_x: 0.0,
            offset_y: 2.0,
            width: 0.5,
            height: 0.5,
            rotation: 0.0,
            drill: None,
            net_id: 3,
            net_name: "NET_C".to_string(),
            is_through_hole: false,
            pin_name: "GPIO1".to_string(),
            swap_group: None,
            net_span: None,
        });
        pads.push(Pad {
            number: "3".to_string(),
            shape: "rect".to_string(),
            offset_x: 0.0,
            offset_y: 4.0,
            width: 0.5,
            height: 0.5,
            rotation: 0.0,
            drill: None,
            net_id: 2,
            net_name: "NET_B".to_string(),
            is_through_hole: false,
            pin_name: "GPIO2".to_string(),
            swap_group: None,
            net_span: None,
        });
        pads.push(Pad {
            number: "4".to_string(),
            shape: "rect".to_string(),
            offset_x: 0.0,
            offset_y: 6.0,
            width: 0.5,
            height: 0.5,
            rotation: 0.0,
            drill: None,
            net_id: 1,
            net_name: "NET_A".to_string(),
            is_through_hole: false,
            pin_name: "GPIO3".to_string(),
            swap_group: None,
            net_span: None,
        });

        let u1 = Footprint {
            reference: "U1".to_string(),
            value: "MCU".to_string(),
            x: 10.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads,
            width: 4.0,
            height: 8.0,
            courtyards: Vec::new(),
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(), // Auto-inferred via GPIO prefix!
            gate_swap_groups: Vec::new(),
        };

        // Header J1 with 4 targets at (30, 10), (30, 12), (30, 14), (30, 16)
        let j1 = Footprint {
            reference: "J1".to_string(),
            value: "HDR".to_string(),
            x: 30.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: true,
            pads: vec![
                Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false),
                Pad::new("2", 0.0, 2.0, 0.5, 0.5, 2, "NET_B", false),
                Pad::new("3", 0.0, 4.0, 0.5, 0.5, 3, "NET_C", false),
                Pad::new("4", 0.0, 6.0, 0.5, 0.5, 4, "NET_D", false),
            ],
            width: 2.0,
            height: 8.0,
            courtyards: Vec::new(),
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        board.footprints = vec![u1, j1];

        let swaps = optimize_pin_and_gate_swapping(&mut board, &PlacementConfig::default());
        assert!(swaps > 0, "Must perform bipartite swaps on GPIO bank");

        // After bipartite matching, Pad 0..3 align directly with NET_A..NET_D
        assert_eq!(board.footprints[0].pads[0].net_name, "NET_A");
        assert_eq!(board.footprints[0].pads[1].net_name, "NET_B");
        assert_eq!(board.footprints[0].pads[2].net_name, "NET_C");
        assert_eq!(board.footprints[0].pads[3].net_name, "NET_D");
    }

    #[test]
    fn test_gate_swapping_quad_opamp() {
        let mut board = make_test_board();

        // U1 has 2 gate units:
        // Unit A (pads 0, 1) at y = -5
        // Unit B (pads 2, 3) at y = 5
        let u1 = Footprint {
            reference: "U1".to_string(),
            value: "LM358".to_string(),
            x: 10.0,
            y: 20.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                Pad::new("1", 0.0, -5.0, 0.5, 0.5, 1, "NET_A", false),
                Pad::new("2", 1.0, -5.0, 0.5, 0.5, 2, "NET_B", false),
                Pad::new("3", 0.0, 5.0, 0.5, 0.5, 3, "NET_C", false),
                Pad::new("4", 1.0, 5.0, 0.5, 0.5, 4, "NET_D", false),
            ],
            width: 4.0,
            height: 12.0,
            courtyards: Vec::new(),
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: vec![SwappableGateGroup {
                name: "OPAMP_UNITS".to_string(),
                gate_units: vec![vec![0, 1], vec![2, 3]],
            }],
        };

        // Targets for NET_A and NET_B are located at y = 50.0 (near Unit B)
        // Targets for NET_C and NET_D are located at y = 0.0 (near Unit A)
        let remote_high = Footprint {
            reference: "REMOTE_HIGH".to_string(),
            value: "CONN".to_string(),
            x: 40.0,
            y: 50.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: true,
            pads: vec![
                Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false),
                Pad::new("2", 0.0, 1.0, 0.5, 0.5, 2, "NET_B", false),
            ],
            width: 2.0,
            height: 2.0,
            courtyards: Vec::new(),
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let remote_low = Footprint {
            reference: "REMOTE_LOW".to_string(),
            value: "CONN".to_string(),
            x: 40.0,
            y: 0.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: true,
            pads: vec![
                Pad::new("1", 0.0, 0.0, 0.5, 0.5, 3, "NET_C", false),
                Pad::new("2", 0.0, 1.0, 0.5, 0.5, 4, "NET_D", false),
            ],
            width: 2.0,
            height: 2.0,
            courtyards: Vec::new(),
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        board.footprints = vec![u1, remote_high, remote_low];

        let swaps = optimize_pin_and_gate_swapping(&mut board, &PlacementConfig::default());
        assert_eq!(swaps, 1, "Gate units must be swapped");

        // Unit A (pads 0, 1 at y=-5) now has NET_C, NET_D (closer to y=0)
        // Unit B (pads 2, 3 at y=+5) now has NET_A, NET_B (closer to y=50)
        assert_eq!(board.footprints[0].pads[0].net_name, "NET_C");
        assert_eq!(board.footprints[0].pads[1].net_name, "NET_D");
        assert_eq!(board.footprints[0].pads[2].net_name, "NET_A");
        assert_eq!(board.footprints[0].pads[3].net_name, "NET_B");
    }
}
