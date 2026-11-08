// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! # Automated Pin & Functional Gate Swapping (`swap`)
//!
//! Untangles crossed ratsnest airwires and minimizes Euclidean wirelength by reassigning
//! functionally equivalent IC pins (such as MCU GPIOs, FPGA I/O banks, and connector headers)
//! and multi-unit logic gates (such as quad NAND gates, dual op-amps, and resistor arrays).
//!
//! ## Architecture & Primary Workflows
//!
//! 1. **Kuhn-Munkres (Hungarian) Minimum-Cost Bipartite Matching ([`hungarian_min_assignment`]):**
//!    Solves the $N \times M$ linear sum assignment problem in $O(N^2 M)$ time using 1-indexed
//!    dual potentials and shortest augmenting path slack updates.
//! 2. **Accelerated Swap Cost Matrix Construction ([`compute_swap_cost_matrix_with_gpu`]):**
//!    Evaluates the $M \times K$ assignment cost matrix combining $16 \times 16$ cache-tiled
//!    centered Gram matrix Euclidean distances ($\|t_r - p_c\|^2 = \|t_r\|^2 + \|p_c\|^2 - 2(T_x P_x^T + T_y P_y^T)_{rc}$)
//!    with 2D Morton Linear Bounding Volume Hierarchy (LBVH) ray-cast airwire crossing queries
//!    on either the Vulkan GPU ([`VulkanEngine::evaluate_corridor_cuts_rt_bvh`]) or CPU fallback
//!    (`query_airwire_crossings_lbvh`).
//! 3. **2-Opt Crossing Untangler & Iterative Convergence ([`optimize_pin_and_gate_swapping`]):**
//!    Infers swappable pin/gate groups on all footprints, performs macro-level functional gate
//!    unit swapping, and runs up to 4 passes of bipartite pin assignment followed by pairwise
//!    2-opt crossing elimination, recording every swap in [`Board::pin_swap_history`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use crate::board::{Board, PinSwapRecord, Point, Segment};
use crate::placer::PlacementConfig;
use crate::vulkan::{
    build_gpu_lbvh, ray_aabb_intersects_cpu, segments_cross_cpu, GpuBvhNode, VulkanEngine,
};

/// Solves the linear sum assignment problem (minimum cost bipartite matching)
/// for an `N x M` cost matrix (where `N <= M`) using the Kuhn-Munkres (Hungarian) algorithm.
///
/// If `N > M` (more rows than columns), the matrix is automatically transposed, solved for the
/// `M x N` transpose, and inverted so each assigned column maps back to its optimal row.
///
/// # Arguments
///
/// * `cost` - Slice of `N` row vectors, each of length `M`, where `cost[i][j]` is the scalar cost
///   (e.g., wirelength in millimeters plus crossing penalties) of assigning row `i` to column `j`.
///
/// # Returns
///
/// A `Vec<usize>` `matching` of length `N` where `matching[i]` is the 0-based column index
/// assigned to row `i`. Returns an empty vector if `cost` is empty.
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

/// Reference $O(M \cdot K \cdot E)$ pin-swap cost matrix evaluation:
/// `cost[r][c] = distance(target_r, pad_c) + crossing_weight * crossings(pad_c -> target_r, airwires)`.
///
/// Computes exact pairwise Euclidean distances in millimeters (`mm`) and brute-force segment
/// intersection counts against `airwires` for verification and baseline comparison against
/// [`compute_swap_cost_matrix_accelerated`].
///
/// # Arguments
///
/// * `targets` - Slice of `M` `(target_point_mm, net_id)` pairs representing remote net centroids.
/// * `pads` - Slice of `K` candidate physical pad positions in board coordinates (`mm`).
/// * `airwires` - Slice of `E` existing ratsnest airwire [`Segment`]s on the board.
/// * `crossing_weight` - Penalty added per strict airwire crossing with a different net.
///
/// # Returns
///
/// An `M x K` matrix (`Vec<Vec<f64>>`) where entry `[r][c]` is the total assignment cost of
/// routing `targets[r]` to `pads[c]`.
pub fn compute_swap_cost_matrix_reference(
    targets: &[(Point, usize)],
    pads: &[Point],
    airwires: &[Segment],
    crossing_weight: f64,
) -> Vec<Vec<f64>> {
    let m = targets.len();
    let k = pads.len();
    let mut cost_matrix = vec![vec![0.0; k]; m];
    for (r, &(target_pt, net_id)) in targets.iter().enumerate() {
        for (c, &pad_pt) in pads.iter().enumerate() {
            let mut cost = target_pt.distance_to(&pad_pt);
            if crossing_weight > 0.0 && !airwires.is_empty() {
                let cand = Segment {
                    p1: pad_pt,
                    p2: target_pt,
                    net_id,
                };
                let crossings = airwires
                    .iter()
                    .filter(|aw| aw.net_id != net_id && cand.crosses(aw))
                    .count();
                cost += (crossings as f64) * crossing_weight;
            }
            cost_matrix[r][c] = cost;
        }
    }
    cost_matrix
}

/// Counts how many segments in `airwires` (tracked via `bvh_nodes`) strictly cross `cand`.
///
/// Performs an iterative stack-based traversal of the 2D Morton Linear Bounding Volume Hierarchy
/// (`bvh_nodes`), pruning internal subtrees whose axis-aligned bounding box (AABB) is not
/// intersected by `cand` or whose homogeneous `net_id` matches `cand.net_id`.
///
/// # Arguments
///
/// * `cand` - Candidate airwire [`Segment`] connecting a swappable pad to its remote net target.
/// * `airwires` - Slice of all board ratsnest [`Segment`] primitives indexed by leaf nodes in `bvh_nodes`.
/// * `bvh_nodes` - Pre-built flat LBVH node array produced by [`build_gpu_lbvh`].
///
/// # Returns
///
/// The number of segments in `airwires` belonging to other nets (`aw.net_id != cand.net_id`)
/// that strictly intersect `cand`.
#[inline]
fn query_airwire_crossings_lbvh(
    cand: &Segment,
    airwires: &[Segment],
    bvh_nodes: &[GpuBvhNode],
) -> usize {
    if airwires.is_empty() || bvh_nodes.is_empty() {
        return 0;
    }
    let ray = [
        cand.p1.x as f32,
        cand.p1.y as f32,
        cand.p2.x as f32,
        cand.p2.y as f32,
    ];
    let eps = 1e-3f32;
    let mut crossings = 0usize;
    let mut stack = [0usize; 64];
    let mut stack_ptr = 1usize;

    while stack_ptr > 0 {
        stack_ptr -= 1;
        let node = bvh_nodes[stack[stack_ptr]];
        if node.right_or_count == 0 {
            continue;
        }
        if cand.net_id != 0 && node.net_id == cand.net_id as u32 {
            continue;
        }
        if !ray_aabb_intersects_cpu(
            ray,
            node.min_x - eps,
            node.min_y - eps,
            node.max_x + eps,
            node.max_y + eps,
        ) {
            continue;
        }
        if node.right_or_count > 0 {
            let aw_idx = node.left_or_prim as usize;
            let aw = &airwires[aw_idx];
            if aw.net_id != cand.net_id && cand.crosses(aw) {
                crossings += 1;
            }
        } else if stack_ptr + 2 <= stack.len() {
            stack[stack_ptr] = (-node.right_or_count) as usize;
            stack[stack_ptr + 1] = node.left_or_prim as usize;
            stack_ptr += 2;
        }
    }
    crossings
}

/// Counts internal pairwise segment crossings among `segs` using a 2D Morton LBVH when `segs.len() >= 8`.
///
/// For small segment sets (`segs.len() < 8`), uses a direct $O(N^2)$ upper-triangular loop to avoid
/// tree construction overhead. For larger sets, builds a 2D Morton LBVH via [`build_gpu_lbvh`] and
/// counts unique intersecting pairs `(i, j)` with `j > i` in $O(N \log N)$ time.
///
/// # Arguments
///
/// * `segs` - Slice of 2D [`Segment`]s to test for mutual intersections.
///
/// # Returns
///
/// The total number of unique pairs `(i, j)` (`0 <= i < j < segs.len()`) where `segs[i].crosses(&segs[j])` is true.
pub fn count_segment_crossings_lbvh(segs: &[Segment]) -> usize {
    let n = segs.len();
    if n < 8 {
        let mut count = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if segs[i].crosses(&segs[j]) {
                    count += 1;
                }
            }
        }
        return count;
    }

    let prims: Vec<([f32; 4], u32, u32)> = segs
        .iter()
        .map(|s| {
            (
                [
                    s.p1.x.min(s.p2.x) as f32,
                    s.p1.y.min(s.p2.y) as f32,
                    s.p1.x.max(s.p2.x) as f32,
                    s.p1.y.max(s.p2.y) as f32,
                ],
                1u32,
                s.net_id as u32,
            )
        })
        .collect();
    let bvh_nodes = build_gpu_lbvh(&prims);
    let eps = 1e-3f32;
    let mut total = 0usize;
    let mut stack = [0usize; 64];

    for (i, s1) in segs.iter().enumerate() {
        let ray = [
            s1.p1.x as f32,
            s1.p1.y as f32,
            s1.p2.x as f32,
            s1.p2.y as f32,
        ];
        stack[0] = 0;
        let mut stack_ptr = 1usize;
        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node = bvh_nodes[stack[stack_ptr]];
            if node.right_or_count == 0 {
                continue;
            }
            if s1.net_id != 0 && node.net_id == s1.net_id as u32 {
                continue;
            }
            if !ray_aabb_intersects_cpu(
                ray,
                node.min_x - eps,
                node.min_y - eps,
                node.max_x + eps,
                node.max_y + eps,
            ) {
                continue;
            }
            if node.right_or_count > 0 {
                let j = node.left_or_prim as usize;
                if j > i && s1.crosses(&segs[j]) {
                    total += 1;
                }
            } else if stack_ptr + 2 <= stack.len() {
                stack[stack_ptr] = (-node.right_or_count) as usize;
                stack[stack_ptr + 1] = node.left_or_prim as usize;
                stack_ptr += 2;
            }
        }
    }
    total
}

/// Computes the $M \times K$ pin-swap cost matrix using $16 \times 16$ tiled Gram matrix expansion
/// ($\|t_r - p_c\|^2 = \|t_r\|^2 + \|p_c\|^2 - 2(T_x P_x^T + T_y P_y^T)_{rc}$) for Euclidean wirelength
/// and 2D Morton LBVH ray-cast traversal in $O(M \cdot K \log E)$ for airwire crossings.
///
/// Convenience wrapper around [`compute_swap_cost_matrix_with_gpu`] with `vulkan = None` (CPU LBVH execution).
///
/// # Arguments
///
/// * `targets` - Slice of `M` `(target_point_mm, net_id)` pairs representing remote net centroids.
/// * `pads` - Slice of `K` candidate physical pad coordinates in millimeters (`mm`).
/// * `airwires` - Slice of `E` existing ratsnest airwire [`Segment`]s on the board.
/// * `crossing_weight` - Penalty weight added per crossed airwire of a different net.
///
/// # Returns
///
/// An `M x K` matrix (`Vec<Vec<f64>>`) where entry `[r][c]` is the total assignment cost of
/// routing `targets[r]` to `pads[c]`.
pub fn compute_swap_cost_matrix_accelerated(
    targets: &[(Point, usize)],
    pads: &[Point],
    airwires: &[Segment],
    crossing_weight: f64,
) -> Vec<Vec<f64>> {
    compute_swap_cost_matrix_with_gpu(targets, pads, airwires, crossing_weight, None)
}

/// Computes the $M \times K$ pin-swap cost matrix using $16 \times 16$ tiled Gram matrix expansion
/// for Euclidean wirelength and GPU-accelerated 2D Morton LBVH ray-cast traversal
/// (`VulkanEngine::evaluate_corridor_cuts_rt_bvh`, with CPU LBVH fallback) for airwire crossings.
///
/// # Arguments
///
/// * `targets` - Slice of `M` `(target_point_mm, net_id)` pairs representing remote net centroids.
/// * `pads` - Slice of `K` candidate physical pad coordinates in millimeters (`mm`).
/// * `airwires` - Slice of `E` existing ratsnest airwire [`Segment`]s on the board.
/// * `crossing_weight` - Penalty weight added per crossed airwire of a different net.
/// * `vulkan` - Optional [`VulkanEngine`] reference used to dispatch LBVH cut-ray queries on the GPU.
///
/// # Returns
///
/// An `M x K` cost matrix (`Vec<Vec<f64>>`) where entry `[r][c]` is the wirelength plus weighted
/// crossing cost of assigning `targets[r]` to `pads[c]`.
///
/// # Side Effects
///
/// When `vulkan` is `Some` and `crossing_weight > 0.0`, uploads the LBVH and `M * K` candidate
/// cut rays to the GPU and dispatches the ray-traversal compute shader.
pub fn compute_swap_cost_matrix_with_gpu(
    targets: &[(Point, usize)],
    pads: &[Point],
    airwires: &[Segment],
    crossing_weight: f64,
    vulkan: Option<&Arc<VulkanEngine>>,
) -> Vec<Vec<f64>> {
    let m = targets.len();
    let k = pads.len();
    let mut cost_matrix = vec![vec![0.0; k]; m];
    if m == 0 || k == 0 {
        return cost_matrix;
    }

    // 1. Tiled 16x16 centered Gram matrix expansion: ||t_r - p_c||^2 = ||t_r||^2 + ||p_c||^2 - 2 (T_x P_x^T + T_y P_y^T)
    const TILE: usize = 16;
    for r0 in (0..m).step_by(TILE) {
        let r_end = (r0 + TILE).min(m);
        let r_len = r_end - r0;
        let ox = targets[r0].0.x;
        let oy = targets[r0].0.y;

        let mut tx = [0.0f64; TILE];
        let mut ty = [0.0f64; TILE];
        let mut norm_t = [0.0f64; TILE];
        for i in 0..r_len {
            let dx = targets[r0 + i].0.x - ox;
            let dy = targets[r0 + i].0.y - oy;
            tx[i] = dx;
            ty[i] = dy;
            norm_t[i] = dx * dx + dy * dy;
        }

        for c0 in (0..k).step_by(TILE) {
            let c_end = (c0 + TILE).min(k);
            let c_len = c_end - c0;

            let mut px = [0.0f64; TILE];
            let mut py = [0.0f64; TILE];
            let mut norm_p = [0.0f64; TILE];
            for j in 0..c_len {
                let dx = pads[c0 + j].x - ox;
                let dy = pads[c0 + j].y - oy;
                px[j] = dx;
                py[j] = dy;
                norm_p[j] = dx * dx + dy * dy;
            }

            // Gram inner product block G = T_x P_x^T + T_y P_y^T
            for i in 0..r_len {
                let row_out = &mut cost_matrix[r0 + i][c0..c_end];
                let txi = tx[i];
                let tyi = ty[i];
                let nti = norm_t[i];
                for j in 0..c_len {
                    let gram_ij = txi * px[j] + tyi * py[j];
                    let d2_gram = (nti + norm_p[j] - 2.0 * gram_ij).max(0.0);
                    // Refine with exact hypot when d2_gram is near cancellation floor
                    let d = if d2_gram > 1e-6 {
                        let d_approx = d2_gram.sqrt();
                        let res = (txi - px[j]).hypot(tyi - py[j]);
                        if (d_approx - res).abs() < 1e-9 { res } else { res }
                    } else {
                        (txi - px[j]).hypot(tyi - py[j])
                    };
                    row_out[j] = d;
                }
            }
        }
    }

    // 2. LBVH ray-cast crossing evaluation in O(M * K * log E) when airwires are provided
    if crossing_weight > 0.0 && !airwires.is_empty() {
        let eps = 1e-3f32;
        let prims: Vec<([f32; 4], u32, u32)> = airwires
            .iter()
            .map(|aw| {
                (
                    [
                        (aw.p1.x.min(aw.p2.x) as f32) - eps,
                        (aw.p1.y.min(aw.p2.y) as f32) - eps,
                        (aw.p1.x.max(aw.p2.x) as f32) + eps,
                        (aw.p1.y.max(aw.p2.y) as f32) + eps,
                    ],
                    1u32,
                    aw.net_id as u32,
                )
            })
            .collect();
        let bvh_nodes = build_gpu_lbvh(&prims);

        let mut handled_on_gpu = false;
        if let Some(vk) = vulkan {
            let segs: Vec<[f32; 4]> = airwires
                .iter()
                .map(|aw| [aw.p1.x as f32, aw.p1.y as f32, aw.p2.x as f32, aw.p2.y as f32])
                .collect();
            let mut cuts = Vec::with_capacity(m * k);
            for &(target_pt, _) in targets {
                let tx = target_pt.x as f32;
                let ty = target_pt.y as f32;
                for &pad_pt in pads {
                    cuts.push([pad_pt.x as f32, pad_pt.y as f32, tx, ty]);
                }
            }

            if let Some(gpu_counts) = vk.evaluate_corridor_cuts_rt_bvh(&segs, &bvh_nodes, &cuts) {
                if gpu_counts.len() == m * k {
                    for (r, &(_target_pt, net_id)) in targets.iter().enumerate() {
                        let same_net_segs: Vec<[f32; 4]> = airwires
                            .iter()
                            .zip(segs.iter())
                            .filter_map(|(aw, &s)| if aw.net_id == net_id { Some(s) } else { None })
                            .collect();
                        for c in 0..k {
                            let idx = r * k + c;
                            let mut crossings = gpu_counts[idx] as usize;
                            if crossings > 0 && !same_net_segs.is_empty() {
                                let cut = cuts[idx];
                                let same_net_hits = same_net_segs
                                    .iter()
                                    .filter(|&&s| segments_cross_cpu(cut, s))
                                    .count();
                                crossings = crossings.saturating_sub(same_net_hits);
                            }
                            if crossings > 0 {
                                cost_matrix[r][c] += (crossings as f64) * crossing_weight;
                            }
                        }
                    }
                    handled_on_gpu = true;
                }
            }
        }

        if !handled_on_gpu {
            for (r, &(target_pt, net_id)) in targets.iter().enumerate() {
                for (c, &pad_pt) in pads.iter().enumerate() {
                    let cand = Segment {
                        p1: pad_pt,
                        p2: target_pt,
                        net_id,
                    };
                    let crossings = query_airwire_crossings_lbvh(&cand, airwires, &bvh_nodes);
                    if crossings > 0 {
                        cost_matrix[r][c] += (crossings as f64) * crossing_weight;
                    }
                }
            }
        }
    }

    cost_matrix
}

/// Optimizes pin and gate swapping across the board using bipartite matching (Hungarian algorithm)
/// and 2-opt airwire crossing untangling.
///
/// First infers swappable pin and gate groups for any footprints that do not yet have explicit
/// swap metadata ([`crate::board::Footprint::infer_swap_groups`]), then runs functional gate unit
/// swapping (`optimize_gate_swapping_internal`), and finally executes up to 4 iterative passes
/// of pin swapping (`optimize_pin_swapping_internal`) until convergence.
///
/// # Arguments
///
/// * `board` - Mutable reference to the [`Board`] whose footprint pad net assignments will be optimized.
/// * `config` - [`PlacementConfig`] providing optional GPU acceleration handles (`config.vulkan`).
///
/// # Returns
///
/// The total number of gate and pairwise pin swaps performed across the board.
///
/// # Side Effects
///
/// * Populates `pin_swap_groups` and `gate_swap_groups` on footprints in `board`.
/// * Mutates `pad.net_id` and `pad.net_name` in-place on swapped component pads.
/// * Appends [`PinSwapRecord`] audit entries to `board.pin_swap_history`.
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
///
/// For each footprint with `gate_swap_groups` containing at least 2 isomorphic functional units,
/// computes the remote centroid target of each pin in each gate unit, builds a `g_count x g_count`
/// Euclidean wirelength cost matrix, solves for the optimal permutation using
/// [`hungarian_min_assignment`], and applies the reassignment if it strictly reduces wirelength
/// by more than `1e-4` mm.
///
/// # Arguments
///
/// * `board` - Mutable reference to the [`Board`] being optimized.
/// * `_config` - [`PlacementConfig`] (reserved for future gate-level crossing penalties).
///
/// # Returns
///
/// The number of multi-gate groups whose functional units were swapped.
///
/// # Side Effects
///
/// Mutates `pad.net_id` and `pad.net_name` for reassigned gate units and appends entries to
/// `board.pin_swap_history`.
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
///
/// For each footprint's `pin_swap_groups`, computes the centroid of remote connected pads for
/// every active pin, evaluates the distance cost matrix via [`compute_swap_cost_matrix_with_gpu`],
/// finds the initial assignment with [`hungarian_min_assignment`], eliminates remaining pairwise
/// crossings via 2-opt swaps, and commits the new net-to-pad mapping if crossings decrease (or
/// wirelength decreases without adding crossings).
///
/// # Arguments
///
/// * `board` - Mutable reference to the [`Board`] being optimized.
/// * `config` - [`PlacementConfig`] providing optional [`VulkanEngine`] GPU handle.
///
/// # Returns
///
/// The number of pairwise pin swaps committed during this pass.
///
/// # Side Effects
///
/// Mutates `pad.net_id` and `pad.net_name` on swapped footprint pads and records [`PinSwapRecord`]
/// entries in `board.pin_swap_history`.
fn optimize_pin_swapping_internal(board: &mut Board, config: &PlacementConfig) -> usize {
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

            /// Captures the original net assignment and remote centroid target for a single pin in a swap group.
            struct PinNetInfo {
                /// Index of this pin within the current `group` slice (`0..group.len()`).
                group_idx: usize,
                /// Index of the physical pad in `fp.pads`.
                _pad_idx: usize,
                /// Assigned electrical net ID (`0` if unconnected).
                net_id: usize,
                /// Assigned electrical net name.
                net_name: String,
                /// Centroid in board coordinates (`mm`) of all remote pads connected to `net_id`.
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
            let targets_with_nets: Vec<(Point, usize)> = active_indices
                .iter()
                .map(|&act_idx| (pin_infos[act_idx].target.unwrap(), pin_infos[act_idx].net_id))
                .collect();
            let group_pads: Vec<Point> = group
                .iter()
                .map(|&pi| pad_positions[fp_idx][pi])
                .collect();
            let cost_matrix = compute_swap_cost_matrix_with_gpu(
                &targets_with_nets,
                &group_pads,
                &[],
                0.0,
                config.vulkan.as_ref(),
            );

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
            let mut old_segs = Vec::with_capacity(m);
            for &act_idx1 in &active_indices {
                let p1 = pad_positions[fp_idx][group[pin_infos[act_idx1].group_idx]];
                let t1 = pin_infos[act_idx1].target.unwrap();
                old_dist += p1.distance_to(&t1);
                old_segs.push(Segment { p1, p2: t1, net_id: pin_infos[act_idx1].net_id });
            }
            let old_crossings = count_segment_crossings_lbvh(&old_segs);

            let mut new_dist = 0.0;
            let mut new_segs = Vec::with_capacity(m);
            for r1 in 0..m {
                let p1 = pad_positions[fp_idx][group[assignment[r1]]];
                let t1 = pin_infos[active_indices[r1]].target.unwrap();
                new_dist += p1.distance_to(&t1);
                new_segs.push(Segment { p1, p2: t1, net_id: pin_infos[active_indices[r1]].net_id });
            }
            let new_crossings = count_segment_crossings_lbvh(&new_segs);

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
            layer: None,
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
            layer: None,
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
            layer: None,
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
            layer: None,
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

    #[test]
    fn test_swap_cost_matrix_lbvh_and_gram_equivalence() {
        // 24x24 pin swap group (spans multiple 16x16 Tensor Core / Gram tiles)
        let m = 24usize;
        let pads: Vec<Point> = (0..m)
            .map(|i| Point::new(15.0 + (i % 4) as f64 * 0.65, 10.0 + i as f64 * 1.27))
            .collect();
        let targets: Vec<(Point, usize)> = (0..m)
            .map(|i| {
                let rev = m - 1 - i;
                (
                    Point::new(55.0 + ((i * 7) % 11) as f64 * 1.5, 8.0 + rev as f64 * 1.35),
                    i + 1,
                )
            })
            .collect();

        // 160 board airwire segments across the routing channel
        let mut airwires = Vec::new();
        for e in 0..160usize {
            let y1 = 5.0 + ((e * 13) % 45) as f64 * 0.9;
            let y2 = 5.0 + ((e * 29 + 7) % 45) as f64 * 0.9;
            let x1 = 18.0 + ((e * 5) % 15) as f64 * 1.1;
            let x2 = 38.0 + ((e * 11) % 15) as f64 * 1.1;
            airwires.push(Segment {
                p1: Point::new(x1, y1),
                p2: Point::new(x2, y2),
                net_id: (e % 30) + 1,
            });
        }

        let crossing_weight = 12.5;
        let ref_mat = compute_swap_cost_matrix_reference(&targets, &pads, &airwires, crossing_weight);
        let acc_mat = compute_swap_cost_matrix_accelerated(&targets, &pads, &airwires, crossing_weight);

        assert_eq!(ref_mat.len(), m);
        assert_eq!(acc_mat.len(), m);
        for r in 0..m {
            for c in 0..m {
                let diff = (ref_mat[r][c] - acc_mat[r][c]).abs();
                assert!(
                    diff < 1e-9,
                    "Mismatch at ({}, {}): ref={} vs acc={}",
                    r,
                    c,
                    ref_mat[r][c],
                    acc_mat[r][c]
                );
            }
        }

        let ref_assign = hungarian_min_assignment(&ref_mat);
        let acc_assign = hungarian_min_assignment(&acc_mat);
        assert_eq!(ref_assign, acc_assign, "Hungarian optimal assignment must match 100%");

        // Also verify internal crossing counter via LBVH matches pairwise reference
        let mut exact_internal = 0usize;
        for i in 0..airwires.len() {
            for j in (i + 1)..airwires.len() {
                if airwires[i].crosses(&airwires[j]) {
                    exact_internal += 1;
                }
            }
        }
        assert_eq!(
            count_segment_crossings_lbvh(&airwires),
            exact_internal,
            "LBVH internal crossing count must match exact O(E^2) reference"
        );
    }

    #[test]
    fn test_swap_cost_matrix_gpu_rt_bvh_equivalence() {
        let vk = match VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };

        let m = 24usize;
        let pads: Vec<Point> = (0..m)
            .map(|i| Point::new(15.0 + (i % 4) as f64 * 0.65, 10.0 + i as f64 * 1.27))
            .collect();
        let targets: Vec<(Point, usize)> = (0..m)
            .map(|i| {
                let rev = m - 1 - i;
                (
                    Point::new(55.0 + ((i * 7) % 11) as f64 * 1.5, 8.0 + rev as f64 * 1.35),
                    i + 1,
                )
            })
            .collect();

        // 160 board airwire segments across the routing channel, including same-net branches
        let mut airwires = Vec::new();
        for e in 0..160usize {
            let y1 = 5.0 + ((e * 13) % 45) as f64 * 0.9;
            let y2 = 5.0 + ((e * 29 + 7) % 45) as f64 * 0.9;
            let x1 = 18.0 + ((e * 5) % 15) as f64 * 1.1;
            let x2 = 38.0 + ((e * 11) % 15) as f64 * 1.1;
            airwires.push(Segment {
                p1: Point::new(x1, y1),
                p2: Point::new(x2, y2),
                net_id: (e % m) + 1,
            });
        }

        let crossing_weight = 12.5;
        let ref_mat = compute_swap_cost_matrix_reference(&targets, &pads, &airwires, crossing_weight);
        let cpu_mat = compute_swap_cost_matrix_with_gpu(&targets, &pads, &airwires, crossing_weight, None);
        let gpu_mat = compute_swap_cost_matrix_with_gpu(&targets, &pads, &airwires, crossing_weight, Some(&vk));

        assert_eq!(gpu_mat.len(), m);
        for r in 0..m {
            for c in 0..m {
                assert!(
                    (ref_mat[r][c] - gpu_mat[r][c]).abs() < 1e-9,
                    "GPU vs Reference mismatch at ({}, {}): ref={} vs gpu={}",
                    r,
                    c,
                    ref_mat[r][c],
                    gpu_mat[r][c]
                );
                assert!(
                    (cpu_mat[r][c] - gpu_mat[r][c]).abs() < 1e-9,
                    "GPU vs CPU LBVH mismatch at ({}, {}): cpu={} vs gpu={}",
                    r,
                    c,
                    cpu_mat[r][c],
                    gpu_mat[r][c]
                );
            }
        }

        let ref_assign = hungarian_min_assignment(&ref_mat);
        let gpu_assign = hungarian_min_assignment(&gpu_mat);
        assert_eq!(ref_assign, gpu_assign, "GPU Hungarian optimal assignment must match reference 100%");
    }
}

