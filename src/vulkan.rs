// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! Dual-backend GPU compute acceleration (`VulkanEngine`) and real-time hardware telemetry (`SystemTelemetry`).
//!
//! This module provides GPU-accelerated PCB placement, routing, and Design Rule Checking (DRC)
//! workloads for Vulcan's Trials through a unified [`VulkanEngine`] facade:
//!
//! - **Dual-Backend Execution (`CUDA PTX` + `Vulkan WGSL`):** Automatically probes for bare-metal
//!   NVIDIA CUDA Driver API (`libcuda.so`) PTX kernels via [`crate::cuda::CudaEngine`] as the primary
//!   fast path, while initializing [`wgpu`] (`wgpu::Backends::VULKAN`) WGSL compute pipelines as a
//!   portable cross-vendor GPU backend and automatic fallback. Multi-GPU systems are supported via
//!   `extra_devices` and [`VulkanEngine::all_engines`].
//! - **2D & Multi-Layer Euclidean Distance Fields (`DISTANCE_FIELD_WGSL`):** Separable two-pass
//!   (`pass_h` horizontal + `pass_v` vertical) $O(W \cdot H \cdot R)$ Euclidean distance transform
//!   over 2D and 3D multi-layer copper obstacle grids for clearance-aware routing penalties.
//! - **3D Wavefront Lee / A* Routing (`COMPUTE_3D_ROUTING_COST_WGSL`, `WAVEFRONT_3D_RELAX_WGSL`, `WAVEFRONT_3D_BACKTRACK_WGSL`):**
//!   Evaluates per-cell 3D routing costs with circular netclass clearance checks, plane layer masks,
//!   PathFinder historical/present congestion penalties, and distance-field repulsion, followed by
//!   ping-pong 8-connected + inter-layer via relaxation and GPU backtracking.
//! - **2D Morton Linear Bounding Volume Hierarchy (`GpuBvhNode`, `build_gpu_lbvh`):** Constructs a
//!   cache-aligned 32-byte-per-node binary LBVH sorted along a 2D Morton Z-order curve (`morton2d_encode`)
//!   with thread-local and engine-level FNV-1a hash caching (`hash_lbvh_primitives`). Accelerates:
//!   - Ratsnest segment crossing counts (`RT_BVH_SEGMENT_CROSSINGS_WGSL`),
//!   - Topological corridor cut congestion evaluation (`RT_BVH_CORRIDOR_CUTS_WGSL`), and
//!   - Line-of-sight DRC clearance ray queries (`RT_BVH_LINE_OF_SIGHT_DRC_WGSL`).
//! - **Parallel Tempering Simulated Annealing Auto-Placer (`GPU_SA_PLACER_WGSL`, `BATCH_PLACEMENT_EVAL_WGSL`):**
//!   Executes parallel Markov Chain Monte Carlo (MCMC) simulated annealing chains on the GPU with
//!   exact Separating Axis Theorem (SAT) convex courtyard polygon collision detection
//!   ([`GpuCourtyardBuffers`]), incremental $\Delta E$ evaluation, passive decoupling affinity,
//!   corridor cut congestion, and signal flow alignment.
//! - **Batch Continuous Geometry DRC (`BATCH_DRC_EVAL_WGSL`):** Evaluates candidate vias and track
//!   segments against rotated rectangular, circular, and oblong/oval pads (`GpuDrcPad`), existing
//!   traces (`GpuDrcTrack`), and vias (`GpuDrcVia`) using symmetric bidirectional copper clearances.
//! - **CUDA-Accelerated Extended Solvers:** Through [`VulkanEngine::cuda_engine`], also exposes
//!   SSTL-18 Laplace electrostatic field solving, ground-plane via stitch candidate scoring, and
//!   silkscreen placement candidate evaluation.
//! - **System Telemetry (`SystemTelemetry`, `TelemetrySnapshot`):** Background sampling thread
//!   measuring host CPU utilization from `/proc/stat` and GPU utilization from cumulative Vulkan/CUDA
//!   compute duty-cycle counters (`compute_busy_ns`), Linux DRM sysfs (`gpu_busy_percent`), and
//!   `nvidia-smi`.

use rayon::prelude::*;
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

/// WGSL compute shader for separable two-pass 2D/3D Euclidean Distance Field (EDF) evaluation.
///
/// Computes the Euclidean distance (in grid cells, clamped to `radius + 1.0`) from every free grid
/// cell `(x, y, z)` to the nearest non-zero obstacle cell on the same layer `z`.
///
/// # Entry Points & Workgroup Size
/// - `pass_h` (`@workgroup_size(16, 16, 1)`): Horizontal 1D scan across `[-radius, +radius]` along `x`,
///   writing squared horizontal distances `min_dx_sq` into `temp_h`.
/// - `pass_v` (`@workgroup_size(16, 16, 1)`): Vertical 1D scan across `[-radius, +radius]` along `y`,
///   combining `temp_h[ny * width + x] + dy * dy` and writing `min(sqrt(best_d_sq), f32(radius) + 1.0)`
///   into `dist_out`.
///
/// # Bindings (`@group(0)`)
/// - `@binding(0)` (`storage, read`): `obstacles: array<u32>` — Flat `[num_layers * height * width]`
///   obstacle grid (`0` = free cell, non-zero = occupied).
/// - `@binding(1)` (`storage, read_write`): `temp_h: array<f32>` — Intermediate horizontal squared
///   distance buffer of length `num_layers * height * width`.
/// - `@binding(2)` (`storage, read_write`): `dist_out: array<f32>` — Output Euclidean distance field
///   buffer of length `num_layers * height * width`.
/// - `@binding(3)` (`uniform`): `params: GridParams` — Grid dimensions (`width`, `height`, `radius`, `num_layers`).
const DISTANCE_FIELD_WGSL: &str = r#"
struct GridParams {
    width: u32,
    height: u32,
    radius: i32,
    num_layers: u32,
}

@group(0) @binding(0) var<storage, read> obstacles: array<u32>;
@group(0) @binding(1) var<storage, read_write> temp_h: array<f32>;
@group(0) @binding(2) var<storage, read_write> dist_out: array<f32>;
@group(0) @binding(3) var<uniform> params: GridParams;

@compute @workgroup_size(16, 16, 1)
fn pass_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = gid.z;
    let w = i32(params.width);
    let h = i32(params.height);
    if (x >= w || y >= h || z >= params.num_layers) {
        return;
    }

    let layer_offset = z * params.width * params.height;
    let idx = layer_offset + u32(y * w + x);
    if (obstacles[idx] != 0u) {
        temp_h[idx] = 0.0;
        return;
    }

    let r = params.radius;
    var min_dx_sq: f32 = f32(r * r + 1);
    for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
        let nx = x + dx;
        if (nx < 0 || nx >= w) {
            continue;
        }
        let nidx = layer_offset + u32(y * w + nx);
        if (obstacles[nidx] != 0u) {
            let d2 = f32(dx * dx);
            if (d2 < min_dx_sq) {
                min_dx_sq = d2;
            }
        }
    }
    temp_h[idx] = min_dx_sq;
}

@compute @workgroup_size(16, 16, 1)
fn pass_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = gid.z;
    let w = i32(params.width);
    let h = i32(params.height);
    if (x >= w || y >= h || z >= params.num_layers) {
        return;
    }

    let layer_offset = z * params.width * params.height;
    let idx = layer_offset + u32(y * w + x);
    let r = params.radius;
    var best_d_sq: f32 = f32(r * r + 1);
    for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
        let ny = y + dy;
        if (ny < 0 || ny >= h) {
            continue;
        }
        let nidx = layer_offset + u32(ny * w + x);
        let d2 = temp_h[nidx] + f32(dy * dy);
        if (d2 < best_d_sq) {
            best_d_sq = d2;
        }
    }
    dist_out[idx] = min(sqrt(best_d_sq), f32(r) + 1.0);
}
"#;

/// WGSL compute shader for batch evaluation of candidate PCB component placements.
///
/// Evaluates `params.num_candidates` complete board placement candidates in parallel, computing
/// total Euclidean ratsnest wirelength, pairwise ratsnest segment crossings (`segments_cross`),
/// pairwise footprint AABB courtyard overlap penalties, and board outline boundary violations.
///
/// # Entry Point & Workgroup Size
/// - `main` (`@workgroup_size(64, 1, 1)`): One invocation per placement candidate `c = gid.x`.
///
/// # Bindings (`@group(0)`)
/// - `@binding(0)` (`storage, read`): `boxes: array<vec4<f32>>` — Flattened `[num_candidates * num_footprints]`
///   footprint AABBs `[min_x, min_y, max_x, max_y]` in mm.
/// - `@binding(1)` (`storage, read`): `segs: array<vec4<f32>>` — Flattened `[num_candidates * num_segments]`
///   ratsnest airwire segments `[x1, y1, x2, y2]` in mm.
/// - `@binding(2)` (`storage, read_write`): `costs_out: array<f32>` — Output total scalar placement cost
///   for each candidate `c` (`[num_candidates]`).
/// - `@binding(3)` (`uniform`): `params: PlacementBatchParams` — Batch dimensions, board bounds, and cost weights.
const BATCH_PLACEMENT_EVAL_WGSL: &str = r#"
struct PlacementBatchParams {
    num_candidates: u32,
    num_footprints: u32,
    num_segments: u32,
    _pad: u32,
    board_min_x: f32,
    board_min_y: f32,
    board_max_x: f32,
    board_max_y: f32,
    wirelength_weight: f32,
    crossing_weight: f32,
    overlap_penalty: f32,
    boundary_penalty: f32,
}

@group(0) @binding(0) var<storage, read> boxes: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> segs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> costs_out: array<f32>;
@group(0) @binding(3) var<uniform> params: PlacementBatchParams;

fn ccw(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> f32 {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

fn segments_cross(s1: vec4<f32>, s2: vec4<f32>) -> bool {
    let eps = 1e-3;
    if ((abs(s1.x - s2.x) < eps && abs(s1.y - s2.y) < eps) ||
        (abs(s1.x - s2.z) < eps && abs(s1.y - s2.w) < eps) ||
        (abs(s1.z - s2.x) < eps && abs(s1.w - s2.y) < eps) ||
        (abs(s1.z - s2.z) < eps && abs(s1.w - s2.w) < eps)) {
        return false;
    }
    let d1 = ccw(s1.x, s1.y, s1.z, s1.w, s2.x, s2.y);
    let d2 = ccw(s1.x, s1.y, s1.z, s1.w, s2.z, s2.w);
    let d3 = ccw(s2.x, s2.y, s2.z, s2.w, s1.x, s1.y);
    let d4 = ccw(s2.x, s2.y, s2.z, s2.w, s1.z, s1.w);
    return ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) &&
           ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.num_candidates) {
        return;
    }

    let nf = params.num_footprints;
    let ns = params.num_segments;
    let b_base = c * nf;
    let s_base = c * ns;

    var wl: f32 = 0.0;
    var crossings: u32 = 0u;
    for (var i: u32 = 0u; i < ns; i = i + 1u) {
        let s1 = segs[s_base + i];
        let dx = s1.z - s1.x;
        let dy = s1.w - s1.y;
        wl = wl + sqrt(dx * dx + dy * dy);
        for (var j: u32 = i + 1u; j < ns; j = j + 1u) {
            if (segments_cross(s1, segs[s_base + j])) {
                crossings = crossings + 1u;
            }
        }
    }

    var overlap_cost: f32 = 0.0;
    var boundary_cost: f32 = 0.0;
    for (var i: u32 = 0u; i < nf; i = i + 1u) {
        let bi = boxes[b_base + i];
        let out_x = max(params.board_min_x - bi.x, 0.0) + max(bi.z - params.board_max_x, 0.0);
        let out_y = max(params.board_min_y - bi.y, 0.0) + max(bi.w - params.board_max_y, 0.0);
        if (out_x > 0.0 || out_y > 0.0) {
            boundary_cost = boundary_cost + params.boundary_penalty * (1.0 + out_x + out_y);
        }
        for (var j: u32 = i + 1u; j < nf; j = j + 1u) {
            let bj = boxes[b_base + j];
            if (bi.z > bj.x && bi.x < bj.z && bi.w > bj.y && bi.y < bj.w) {
                let ox = max(min(bi.z, bj.z) - max(bi.x, bj.x), 0.1);
                let oy = max(min(bi.w, bj.w) - max(bi.y, bj.y), 0.1);
                overlap_cost = overlap_cost + params.overlap_penalty * (1.0 + ox * oy);
            }
        }
    }

    costs_out[c] = wl * params.wirelength_weight
        + f32(crossings) * params.crossing_weight
        + overlap_cost
        + boundary_cost;
}
"#;

/// WGSL compute shader for GPU-accelerated Parallel Tempering Markov Chain Monte Carlo (MCMC)
/// Simulated Annealing (`SA`) PCB auto-placement.
///
/// Runs `params.num_chains` independent Metropolis-Hastings annealing chains in parallel for
/// `params.steps_per_pass` random perturbation steps per dispatch. Supports exact Separating Axis
/// Theorem (SAT) convex courtyard polygon intersection tests (`sat_polys_intersect`,
/// `footprints_courtyards_intersect`), incremental $\Delta E$ energy evaluation (`evaluate_chain_delta_e`),
/// decoupling capacitor / passive affinity penalties, topological corridor cut congestion, and
/// signal flow alignment.
///
/// # Entry Point & Workgroup Size
/// - `main` (`@workgroup_size(64, 1, 1)`): One thread per MCMC chain `c = gid.x` (`0..params.num_chains`),
///   with exponentially spaced parallel-tempering temperatures derived from `params.base_temp`.
///
/// # Bindings (`@group(0)`)
/// - `@binding(0)` (`storage, read`): `fp_descs: array<vec4<f32>>` — Per-footprint descriptors
///   `[half_w, half_h, is_locked, packed_aabb_offset]` of length `num_footprints`.
/// - `@binding(1)` (`storage, read`): `edges: array<vec4<f32>>` — Pairs of `vec4<f32>` endpoints
///   `([fp_a, off_ax, off_ay, edge_tag], [fp_b, off_bx, off_by, 0.0])` of length `2 * num_edges`.
/// - `@binding(2)` (`storage, read_write`): `chain_poses: array<vec4<f32>>` — Currentposes (`[0..num_chains * num_footprints]`)
///   followed by best-seen poses (`[num_chains * num_footprints .. 2 * num_chains * num_footprints]`),
///   each encoded as `[x_mm, y_mm, rot_deg, layer]`.
/// - `@binding(3)` (`storage, read_write`): `chain_scores: array<vec4<f32>>` — Per-chain state vector
///   `[cur_cost, best_cost, cur_overlaps, pass_counter]` of length `num_chains`.
/// - `@binding(4)` (`uniform`): `params: GpuSaParams` — Annealing schedule, board bounds, grid step, and penalty weights.
/// - `@binding(5)` (`storage, read`): `fp_poly_ranges: array<vec4<u32>>` — Per-footprint polygon slice headers
///   `[poly_start, poly_count, flags, orig_layer]`.
/// - `@binding(6)` (`storage, read`): `courtyard_polys: array<vec4<u32>>` — Convex courtyard polygon headers
///   `[vertex_start, vertex_count, fp_idx, 0]`.
/// - `@binding(7)` (`storage, read`): `courtyard_vertices: array<vec2<f32>>` — Local-space courtyard polygon
///   vertices `[x_mm, y_mm]`.
const GPU_SA_PLACER_WGSL: &str = r#"
struct GpuSaParams {
    num_chains: u32,
    num_footprints: u32,
    num_edges: u32,
    steps_per_pass: u32,
    seed: u32,
    _pad0: u32,
    corridor_congestion_weight: f32,
    flow_alignment_weight: f32,
    board_min_x: f32,
    board_min_y: f32,
    board_max_x: f32,
    board_max_y: f32,
    grid_step: f32,
    wirelength_weight: f32,
    crossing_weight: f32,
    overlap_penalty: f32,
    boundary_penalty: f32,
    passive_affinity_weight: f32,
    passive_max_dist: f32,
    base_temp: f32,
}

@group(0) @binding(0) var<storage, read> fp_descs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> edges: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> chain_poses: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> chain_scores: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: GpuSaParams;
@group(0) @binding(5) var<storage, read> fp_poly_ranges: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> courtyard_polys: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> courtyard_vertices: array<vec2<f32>>;

fn pcg_next(state: ptr<function, u32>) -> u32 {
    let old = *state;
    *state = old * 747796405u + 2891336453u;
    let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
    return f32(pcg_next(state) & 0x00FFFFFFu) / 16777216.0;
}

fn snap_grid(v: f32, step: f32) -> f32 {
    if (step <= 0.0) {
        return v;
    }
    return round(v / step) * step;
}

fn ccw_sa(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> f32 {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

fn segs_cross_sa(s1: vec4<f32>, s2: vec4<f32>) -> bool {
    let eps = 1e-3;
    if ((abs(s1.x - s2.x) < eps && abs(s1.y - s2.y) < eps) ||
        (abs(s1.x - s2.z) < eps && abs(s1.y - s2.w) < eps) ||
        (abs(s1.z - s2.x) < eps && abs(s1.w - s2.y) < eps) ||
        (abs(s1.z - s2.z) < eps && abs(s1.w - s2.w) < eps)) {
        return false;
    }
    let d1 = ccw_sa(s1.x, s1.y, s1.z, s1.w, s2.x, s2.y);
    let d2 = ccw_sa(s1.x, s1.y, s1.z, s1.w, s2.z, s2.w);
    let d3 = ccw_sa(s2.x, s2.y, s2.z, s2.w, s1.x, s1.y);
    let d4 = ccw_sa(s2.x, s2.y, s2.z, s2.w, s1.z, s1.w);
    return ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) &&
           ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0));
}

fn get_edge_seg(c_base: u32, e_idx: u32) -> vec4<f32> {
    let ea = edges[e_idx * 2u];
    let eb = edges[e_idx * 2u + 1u];
    let fpa = chain_poses[c_base + u32(ea.x)];
    let fpb = chain_poses[c_base + u32(eb.x)];

    let rad_a = fpa.z * 0.0174532925;
    let cos_a = cos(rad_a);
    let sin_a = sin(rad_a);
    let ax = fpa.x + (ea.y * cos_a + ea.z * sin_a);
    let ay = fpa.y + (-ea.y * sin_a + ea.z * cos_a);

    let rad_b = fpb.z * 0.0174532925;
    let cos_b = cos(rad_b);
    let sin_b = sin(rad_b);
    let bx = fpb.x + (eb.y * cos_b + eb.z * sin_b);
    let by = fpb.y + (-eb.y * sin_b + eb.z * cos_b);

    return vec4<f32>(ax, ay, bx, by);
}

fn pose_aabb(pose: vec4<f32>, desc: vec4<f32>, s: f32, c: f32) -> vec4<f32> {
    var px = pose.x;
    var py = pose.y;
    if (desc.w > 0.0) {
        let qx = floor(desc.w * (1.0 / 1024.0));
        let qy = desc.w - qx * 1024.0;
        let cx = (qx - 512.0) * 0.05;
        let cy = (qy - 512.0) * 0.05;
        px = px + (cx * c + cy * s);
        py = py + (-cx * s + cy * c);
    }
    let ac = abs(c);
    let as_v = abs(s);
    let hw = desc.x * ac + desc.y * as_v;
    let hh = desc.x * as_v + desc.y * ac;
    return vec4<f32>(px - hw, py - hh, px + hw, py + hh);
}

fn fp_aabb(c_base: u32, fp_idx: u32) -> vec4<f32> {
    let pose = chain_poses[c_base + fp_idx];
    let desc = fp_descs[fp_idx];
    let rad = pose.z * 0.0174532925;
    return pose_aabb(pose, desc, sin(rad), cos(rad));
}

fn xform_cyd_vertex(lv: vec2<f32>, pose: vec4<f32>, ca: f32, sa: f32, flip_y: bool) -> vec2<f32> {
    let ly = select(lv.y, -lv.y, flip_y);
    return vec2<f32>(
        pose.x + (lv.x * ca + ly * sa),
        pose.y + (-lv.x * sa + ly * ca)
    );
}

fn sat_polys_intersect(
    poly_a_idx: u32,
    pose_a: vec4<f32>,
    orig_layer_a: u32,
    poly_b_idx: u32,
    pose_b: vec4<f32>,
    orig_layer_b: u32
) -> bool {
    let pa = courtyard_polys[poly_a_idx];
    let pb = courtyard_polys[poly_b_idx];
    let na = min(pa.y, 16u);
    let nb = min(pb.y, 16u);
    if (na < 3u || nb < 3u) {
        return false;
    }

    let rad_a = pose_a.z * 0.0174532925;
    let sa = sin(rad_a);
    let ca = cos(rad_a);
    let flip_a = (u32(round(pose_a.w)) != orig_layer_a);

    let rad_b = pose_b.z * 0.0174532925;
    let sb = sin(rad_b);
    let cb = cos(rad_b);
    let flip_b = (u32(round(pose_b.w)) != orig_layer_b);

    for (var i: u32 = 0u; i < na; i = i + 1u) {
        let p1 = xform_cyd_vertex(courtyard_vertices[pa.x + i], pose_a, ca, sa, flip_a);
        let p2 = xform_cyd_vertex(courtyard_vertices[pa.x + ((i + 1u) % na)], pose_a, ca, sa, flip_a);
        let axis = vec2<f32>(-(p2.y - p1.y), p2.x - p1.x);
        let axis_len_sq = dot(axis, axis);
        if (axis_len_sq < 1e-10) {
            continue;
        }

        let va0 = xform_cyd_vertex(courtyard_vertices[pa.x], pose_a, ca, sa, flip_a);
        var min_a = dot(va0, axis);
        var max_a = min_a;
        for (var k: u32 = 1u; k < na; k = k + 1u) {
            let vak = xform_cyd_vertex(courtyard_vertices[pa.x + k], pose_a, ca, sa, flip_a);
            let d = dot(vak, axis);
            min_a = min(min_a, d);
            max_a = max(max_a, d);
        }

        let vb0 = xform_cyd_vertex(courtyard_vertices[pb.x], pose_b, cb, sb, flip_b);
        var min_b = dot(vb0, axis);
        var max_b = min_b;
        for (var k: u32 = 1u; k < nb; k = k + 1u) {
            let vbk = xform_cyd_vertex(courtyard_vertices[pb.x + k], pose_b, cb, sb, flip_b);
            let d = dot(vbk, axis);
            min_b = min(min_b, d);
            max_b = max(max_b, d);
        }

        if (max_a <= min_b || max_b <= min_a) {
            return false;
        }
    }

    for (var i: u32 = 0u; i < nb; i = i + 1u) {
        let p1 = xform_cyd_vertex(courtyard_vertices[pb.x + i], pose_b, cb, sb, flip_b);
        let p2 = xform_cyd_vertex(courtyard_vertices[pb.x + ((i + 1u) % nb)], pose_b, cb, sb, flip_b);
        let axis = vec2<f32>(-(p2.y - p1.y), p2.x - p1.x);
        let axis_len_sq = dot(axis, axis);
        if (axis_len_sq < 1e-10) {
            continue;
        }

        let va0 = xform_cyd_vertex(courtyard_vertices[pa.x], pose_a, ca, sa, flip_a);
        var min_a = dot(va0, axis);
        var max_a = min_a;
        for (var k: u32 = 1u; k < na; k = k + 1u) {
            let vak = xform_cyd_vertex(courtyard_vertices[pa.x + k], pose_a, ca, sa, flip_a);
            let d = dot(vak, axis);
            min_a = min(min_a, d);
            max_a = max(max_a, d);
        }

        let vb0 = xform_cyd_vertex(courtyard_vertices[pb.x], pose_b, cb, sb, flip_b);
        var min_b = dot(vb0, axis);
        var max_b = min_b;
        for (var k: u32 = 1u; k < nb; k = k + 1u) {
            let vbk = xform_cyd_vertex(courtyard_vertices[pb.x + k], pose_b, cb, sb, flip_b);
            let d = dot(vbk, axis);
            min_b = min(min_b, d);
            max_b = max(max_b, d);
        }

        if (max_a <= min_b || max_b <= min_a) {
            return false;
        }
    }

    return true;
}

fn is_ortho_angle(deg: f32) -> bool {
    let rem = abs(deg - round(deg / 90.0) * 90.0);
    return rem < 1e-3;
}

fn footprints_courtyards_intersect(
    i: u32,
    pose_i: vec4<f32>,
    j: u32,
    pose_j: vec4<f32>
) -> bool {
    let ri = fp_poly_ranges[i];
    let rj = fp_poly_ranges[j];

    if ((ri.z & 1u) != 0u && (rj.z & 1u) != 0u && u32(round(pose_i.w)) != u32(round(pose_j.w))) {
        return false;
    }

    if (is_ortho_angle(pose_i.z) && is_ortho_angle(pose_j.z)) {
        if (ri.y == 1u && rj.y == 1u && ((ri.z | rj.z) & 2u) == 0u) {
            return true;
        }
        if ((ri.z & rj.z & 4u) != 0u) {
            let rad_i = pose_i.z * 0.0174532925;
            let si = sin(rad_i);
            let ci = cos(rad_i);
            let flip_i = (u32(round(pose_i.w)) != ri.w);
            let rad_j = pose_j.z * 0.0174532925;
            let sj = sin(rad_j);
            let cj = cos(rad_j);
            let flip_j = (u32(round(pose_j.w)) != rj.w);
            for (var pi: u32 = ri.x; pi < ri.x + ri.y; pi = pi + 1u) {
                let pa = courtyard_polys[pi];
                let wa0 = xform_cyd_vertex(courtyard_vertices[pa.x], pose_i, ci, si, flip_i);
                let wa2 = xform_cyd_vertex(courtyard_vertices[pa.x + 2u], pose_i, ci, si, flip_i);
                let min_ax = min(wa0.x, wa2.x);
                let max_ax = max(wa0.x, wa2.x);
                let min_ay = min(wa0.y, wa2.y);
                let max_ay = max(wa0.y, wa2.y);
                for (var pj: u32 = rj.x; pj < rj.x + rj.y; pj = pj + 1u) {
                    let pb = courtyard_polys[pj];
                    let wb0 = xform_cyd_vertex(courtyard_vertices[pb.x], pose_j, cj, sj, flip_j);
                    let wb2 = xform_cyd_vertex(courtyard_vertices[pb.x + 2u], pose_j, cj, sj, flip_j);
                    if (max_ax > min(wb0.x, wb2.x) && max(wb0.x, wb2.x) > min_ax &&
                        max_ay > min(wb0.y, wb2.y) && max(wb0.y, wb2.y) > min_ay) {
                        return true;
                    }
                }
            }
            return false;
        }
    }

    for (var pi: u32 = ri.x; pi < ri.x + ri.y; pi = pi + 1u) {
        for (var pj: u32 = rj.x; pj < rj.x + rj.y; pj = pj + 1u) {
            if (sat_polys_intersect(pi, pose_i, ri.w, pj, pose_j, rj.w)) {
                return true;
            }
        }
    }
    return false;
}

fn evaluate_chain_state(c_base: u32, nf: u32, ne: u32) -> vec2<f32> {
    var overlap_cost: f32 = 0.0;
    var boundary_cost: f32 = 0.0;
    var overlaps: f32 = 0.0;

    for (var i: u32 = 0u; i < nf; i = i + 1u) {
        let bi = fp_aabb(c_base, i);
        let i_locked = fp_descs[i].z > 0.5;
        if (!i_locked) {
            let out_x = max(params.board_min_x - bi.x, 0.0) + max(bi.z - params.board_max_x, 0.0);
            let out_y = max(params.board_min_y - bi.y, 0.0) + max(bi.w - params.board_max_y, 0.0);
            if (out_x > 0.0 || out_y > 0.0) {
                boundary_cost = boundary_cost + params.boundary_penalty * (1.0 + out_x + out_y);
            }
        }
        for (var j: u32 = i + 1u; j < nf; j = j + 1u) {
            if (i_locked && fp_descs[j].z > 0.5) {
                continue;
            }
            let bj = fp_aabb(c_base, j);
            if (bi.z > bj.x && bi.x < bj.z && bi.w > bj.y && bi.y < bj.w) {
                let pose_i = chain_poses[c_base + i];
                let pose_j = chain_poses[c_base + j];
                if (footprints_courtyards_intersect(i, pose_i, j, pose_j)) {
                    overlaps = overlaps + 1.0;
                    let ox = max(min(bi.z, bj.z) - max(bi.x, bj.x), 0.1);
                    let oy = max(min(bi.w, bj.w) - max(bi.y, bj.y), 0.1);
                    overlap_cost = overlap_cost + params.overlap_penalty * (1.0 + ox * oy);
                }
            }
        }
    }

    var wl: f32 = 0.0;
    var crossings: f32 = 0.0;
    var affinity_cost: f32 = 0.0;
    var corridor_cost: f32 = 0.0;
    var flow_cost: f32 = 0.0;
    for (var e1: u32 = 0u; e1 < ne; e1 = e1 + 1u) {
        let s1 = get_edge_seg(c_base, e1);
        let dx = s1.z - s1.x;
        let dy = s1.w - s1.y;
        let d = sqrt(dx * dx + dy * dy);
        let tag = edges[e1 * 2u].w;
        if (tag > 3.5) {
            flow_cost = flow_cost + params.flow_alignment_weight * 0.25 * d;
        } else if (tag > 2.5) {
            // Corridor cut segment: crossings with normal airwires are accumulated below
        } else if (tag > 0.5) {
            let is_decoupling = tag > 1.5;
            let max_d = select(params.passive_max_dist, 2.0, is_decoupling);
            let weight = select(params.passive_affinity_weight, params.passive_affinity_weight * 3.5, is_decoupling);
            if (d > max_d) {
                let ex = d - max_d;
                affinity_cost = affinity_cost + weight * (d + ex * ex * 3.0);
            } else {
                affinity_cost = affinity_cost + weight * 0.25 * d;
            }
        } else {
            wl = wl + d;
            for (var e2: u32 = e1 + 1u; e2 < ne; e2 = e2 + 1u) {
                let tag2 = edges[e2 * 2u].w;
                if (tag2 < 0.5) {
                    let s2 = get_edge_seg(c_base, e2);
                    if (segs_cross_sa(s1, s2)) {
                        crossings = crossings + 1.0;
                    }
                } else if (tag2 > 2.5 && tag2 < 3.5) {
                    if (params.corridor_congestion_weight > 0.0) {
                        let s2 = get_edge_seg(c_base, e2);
                        if (segs_cross_sa(s1, s2)) {
                            corridor_cost = corridor_cost + params.corridor_congestion_weight;
                        }
                    }
                } else {
                    break;
                }
            }
        }
    }

    let total_cost = wl * params.wirelength_weight
        + crossings * params.crossing_weight
        + corridor_cost
        + flow_cost
        + overlap_cost
        + boundary_cost
        + affinity_cost;
    return vec2<f32>(total_cost, overlaps);
}

fn get_edge_endpoint_from_pose(pose: vec4<f32>, ep: vec4<f32>) -> vec2<f32> {
    let rad = pose.z * 0.0174532925;
    let c = cos(rad);
    let s = sin(rad);
    return vec2<f32>(
        pose.x + (ep.y * c + ep.z * s),
        pose.y + (-ep.y * s + ep.z * c)
    );
}

fn evaluate_chain_delta_e(
    c_base: u32,
    nf: u32,
    ne: u32,
    idx: u32,
    old_pose: vec4<f32>,
    new_pose: vec4<f32>
) -> vec2<f32> {
    if (old_pose.x == new_pose.x && old_pose.y == new_pose.y && old_pose.z == new_pose.z) {
        return vec2<f32>(0.0, 0.0);
    }

    let desc_i = fp_descs[idx];
    let i_locked = desc_i.z > 0.5;

    let old_rad = old_pose.z * 0.0174532925;
    let old_cos = cos(old_rad);
    let old_sin = sin(old_rad);
    let old_bi = pose_aabb(old_pose, desc_i, old_sin, old_cos);

    let new_rad = new_pose.z * 0.0174532925;
    let new_cos = cos(new_rad);
    let new_sin = sin(new_rad);
    let new_bi = pose_aabb(new_pose, desc_i, new_sin, new_cos);

    var delta_bd: f32 = 0.0;
    if (!i_locked) {
        let old_out_x = max(params.board_min_x - old_bi.x, 0.0) + max(old_bi.z - params.board_max_x, 0.0);
        let old_out_y = max(params.board_min_y - old_bi.y, 0.0) + max(old_bi.w - params.board_max_y, 0.0);
        if (old_out_x > 0.0 || old_out_y > 0.0) {
            delta_bd = delta_bd - params.boundary_penalty * (1.0 + old_out_x + old_out_y);
        }
        let new_out_x = max(params.board_min_x - new_bi.x, 0.0) + max(new_bi.z - params.board_max_x, 0.0);
        let new_out_y = max(params.board_min_y - new_bi.y, 0.0) + max(new_bi.w - params.board_max_y, 0.0);
        if (new_out_x > 0.0 || new_out_y > 0.0) {
            delta_bd = delta_bd + params.boundary_penalty * (1.0 + new_out_x + new_out_y);
        }
    }

    var delta_ov_cost: f32 = 0.0;
    var delta_ov: f32 = 0.0;
    for (var j: u32 = 0u; j < nf; j = j + 1u) {
        if (j == idx) {
            continue;
        }
        if (i_locked && fp_descs[j].z > 0.5) {
            continue;
        }
        let bj = fp_aabb(c_base, j);
        let pose_j = chain_poses[c_base + j];
        if (old_bi.z > bj.x && old_bi.x < bj.z && old_bi.w > bj.y && old_bi.y < bj.w) {
            if (footprints_courtyards_intersect(idx, old_pose, j, pose_j)) {
                delta_ov = delta_ov - 1.0;
                let ox = max(min(old_bi.z, bj.z) - max(old_bi.x, bj.x), 0.1);
                let oy = max(min(old_bi.w, bj.w) - max(old_bi.y, bj.y), 0.1);
                delta_ov_cost = delta_ov_cost - params.overlap_penalty * (1.0 + ox * oy);
            }
        }
        if (new_bi.z > bj.x && new_bi.x < bj.z && new_bi.w > bj.y && new_bi.y < bj.w) {
            if (footprints_courtyards_intersect(idx, new_pose, j, pose_j)) {
                delta_ov = delta_ov + 1.0;
                let ox = max(min(new_bi.z, bj.z) - max(new_bi.x, bj.x), 0.1);
                let oy = max(min(new_bi.w, bj.w) - max(new_bi.y, bj.y), 0.1);
                delta_ov_cost = delta_ov_cost + params.overlap_penalty * (1.0 + ox * oy);
            }
        }
    }

    var delta_wl: f32 = 0.0;
    var delta_cr: f32 = 0.0;
    var delta_aff: f32 = 0.0;
    var delta_corr: f32 = 0.0;
    var delta_flow: f32 = 0.0;
    for (var e1: u32 = 0u; e1 < ne; e1 = e1 + 1u) {
        let ea = edges[e1 * 2u];
        let eb = edges[e1 * 2u + 1u];
        let fa = u32(ea.x);
        let fb = u32(eb.x);
        if (fa != idx && fb != idx) {
            continue;
        }

        var old_ax: f32;
        var old_ay: f32;
        var new_ax: f32;
        var new_ay: f32;
        if (fa == idx) {
            old_ax = old_pose.x + (ea.y * old_cos + ea.z * old_sin);
            old_ay = old_pose.y + (-ea.y * old_sin + ea.z * old_cos);
            new_ax = new_pose.x + (ea.y * new_cos + ea.z * new_sin);
            new_ay = new_pose.y + (-ea.y * new_sin + ea.z * new_cos);
        } else {
            let pa = get_edge_endpoint_from_pose(chain_poses[c_base + fa], ea);
            old_ax = pa.x;
            old_ay = pa.y;
            new_ax = pa.x;
            new_ay = pa.y;
        }

        var old_bx: f32;
        var old_by: f32;
        var new_bx: f32;
        var new_by: f32;
        if (fb == idx) {
            old_bx = old_pose.x + (eb.y * old_cos + eb.z * old_sin);
            old_by = old_pose.y + (-eb.y * old_sin + eb.z * old_cos);
            new_bx = new_pose.x + (eb.y * new_cos + eb.z * new_sin);
            new_by = new_pose.y + (-eb.y * new_sin + eb.z * new_cos);
        } else {
            let pb = get_edge_endpoint_from_pose(chain_poses[c_base + fb], eb);
            old_bx = pb.x;
            old_by = pb.y;
            new_bx = pb.x;
            new_by = pb.y;
        }

        let old_s1 = vec4<f32>(old_ax, old_ay, old_bx, old_by);
        let new_s1 = vec4<f32>(new_ax, new_ay, new_bx, new_by);
        let old_dx = old_bx - old_ax;
        let old_dy = old_by - old_ay;
        let old_d = sqrt(old_dx * old_dx + old_dy * old_dy);
        let new_dx = new_bx - new_ax;
        let new_dy = new_by - new_ay;
        let new_d = sqrt(new_dx * new_dx + new_dy * new_dy);

        if (ea.w > 3.5) {
            delta_flow = delta_flow + params.flow_alignment_weight * 0.25 * (new_d - old_d);
        } else if (ea.w > 2.5) {
            if (params.corridor_congestion_weight > 0.0) {
                for (var e2: u32 = 0u; e2 < ne; e2 = e2 + 1u) {
                    let e2a = edges[e2 * 2u];
                    if (e2a.w >= 0.5) {
                        break;
                    }
                    let e2b = edges[e2 * 2u + 1u];
                    let f2a = u32(e2a.x);
                    let f2b = u32(e2b.x);
                    if (f2a == idx || f2b == idx) {
                        continue;
                    }
                    let s2 = get_edge_seg(c_base, e2);
                    if (segs_cross_sa(old_s1, s2)) {
                        delta_corr = delta_corr - params.corridor_congestion_weight;
                    }
                    if (segs_cross_sa(new_s1, s2)) {
                        delta_corr = delta_corr + params.corridor_congestion_weight;
                    }
                }
            }
        } else if (ea.w > 0.5) {
            let is_decoupling = ea.w > 1.5;
            let max_d = select(params.passive_max_dist, 2.0, is_decoupling);
            let weight = select(params.passive_affinity_weight, params.passive_affinity_weight * 3.5, is_decoupling);
            if (old_d > max_d) {
                let ex = old_d - max_d;
                delta_aff = delta_aff - weight * (old_d + ex * ex * 3.0);
            } else {
                delta_aff = delta_aff - weight * 0.25 * old_d;
            }
            if (new_d > max_d) {
                let ex = new_d - max_d;
                delta_aff = delta_aff + weight * (new_d + ex * ex * 3.0);
            } else {
                delta_aff = delta_aff + weight * 0.25 * new_d;
            }
        } else {
            delta_wl = delta_wl + (new_d - old_d);
            for (var e2: u32 = 0u; e2 < ne; e2 = e2 + 1u) {
                if (e2 == e1) {
                    continue;
                }
                let e2a = edges[e2 * 2u];
                let is_corr2 = (e2a.w > 2.5 && e2a.w < 3.5);
                if (e2a.w >= 0.5) {
                    if (!is_corr2) {
                        break;
                    }
                    if (params.corridor_congestion_weight <= 0.0) {
                        continue;
                    }
                }
                let e2b = edges[e2 * 2u + 1u];
                let f2a = u32(e2a.x);
                let f2b = u32(e2b.x);
                let e2_inc = (f2a == idx || f2b == idx);
                if (!is_corr2 && e2_inc && e2 < e1) {
                    continue;
                }
                let w_cross = select(params.crossing_weight, params.corridor_congestion_weight, is_corr2);
                if (!e2_inc) {
                    let s2 = get_edge_seg(c_base, e2);
                    if (segs_cross_sa(old_s1, s2)) {
                        if (is_corr2) {
                            delta_corr = delta_corr - w_cross;
                        } else {
                            delta_cr = delta_cr - 1.0;
                        }
                    }
                    if (segs_cross_sa(new_s1, s2)) {
                        if (is_corr2) {
                            delta_corr = delta_corr + w_cross;
                        } else {
                            delta_cr = delta_cr + 1.0;
                        }
                    }
                } else {
                    var p2a_old: vec2<f32>;
                    var p2a_new: vec2<f32>;
                    if (f2a == idx) {
                        p2a_old = vec2<f32>(old_pose.x + (e2a.y * old_cos + e2a.z * old_sin), old_pose.y + (-e2a.y * old_sin + e2a.z * old_cos));
                        p2a_new = vec2<f32>(new_pose.x + (e2a.y * new_cos + e2a.z * new_sin), new_pose.y + (-e2a.y * new_sin + e2a.z * new_cos));
                    } else {
                        p2a_old = get_edge_endpoint_from_pose(chain_poses[c_base + f2a], e2a);
                        p2a_new = p2a_old;
                    }
                    var p2b_old: vec2<f32>;
                    var p2b_new: vec2<f32>;
                    if (f2b == idx) {
                        p2b_old = vec2<f32>(old_pose.x + (e2b.y * old_cos + e2b.z * old_sin), old_pose.y + (-e2b.y * old_sin + e2b.z * old_cos));
                        p2b_new = vec2<f32>(new_pose.x + (e2b.y * new_cos + e2b.z * new_sin), new_pose.y + (-e2b.y * new_sin + e2b.z * new_cos));
                    } else {
                        p2b_old = get_edge_endpoint_from_pose(chain_poses[c_base + f2b], e2b);
                        p2b_new = p2b_old;
                    }
                    let old_s2 = vec4<f32>(p2a_old.x, p2a_old.y, p2b_old.x, p2b_old.y);
                    let new_s2 = vec4<f32>(p2a_new.x, p2a_new.y, p2b_new.x, p2b_new.y);
                    if (segs_cross_sa(old_s1, old_s2)) {
                        if (is_corr2) {
                            delta_corr = delta_corr - w_cross;
                        } else {
                            delta_cr = delta_cr - 1.0;
                        }
                    }
                    if (segs_cross_sa(new_s1, new_s2)) {
                        if (is_corr2) {
                            delta_corr = delta_corr + w_cross;
                        } else {
                            delta_cr = delta_cr + 1.0;
                        }
                    }
                }
            }
        }
    }

    let delta_cost = delta_wl * params.wirelength_weight
        + delta_cr * params.crossing_weight
        + delta_corr
        + delta_flow
        + delta_ov_cost
        + delta_bd
        + delta_aff;
    return vec2<f32>(delta_cost, delta_ov);
}

fn clamp_pose_to_board_aabb(pose: vec4<f32>, desc: vec4<f32>) -> vec4<f32> {
    let rad = pose.z * 0.0174532925;
    let bi = pose_aabb(pose, desc, sin(rad), cos(rad));
    let dx_min = pose.x - bi.x;
    let dx_max = bi.z - pose.x;
    let dy_min = pose.y - bi.y;
    let dy_max = bi.w - pose.y;
    let cx_min = params.board_min_x + dx_min;
    let cx_max = max(params.board_max_x - dx_max, cx_min);
    let cy_min = params.board_min_y + dy_min;
    let cy_max = max(params.board_max_y - dy_max, cy_min);
    return vec4<f32>(
        clamp(snap_grid(pose.x, params.grid_step), cx_min, cx_max),
        clamp(snap_grid(pose.y, params.grid_step), cy_min, cy_max),
        pose.z,
        pose.w
    );
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.num_chains) {
        return;
    }

    let nf = params.num_footprints;
    let ne = params.num_edges;
    let c_base = c * nf;
    let best_base = (params.num_chains + c) * nf;

    var score = chain_scores[c];
    var cur_cost = score.x;
    var best_cost = score.y;
    var best_ov = score.z;
    var cur_ov: f32 = 0.0;
    var pass_idx: f32 = 1.0;

    var rng = params.seed ^ (c * 747796405u + 2891336453u) ^ u32(score.w * 1664525.0);
    _ = pcg_next(&rng);

    if (score.w <= 0.0) {
        let init_ev = evaluate_chain_state(c_base, nf, ne);
        cur_cost = init_ev.x;
        best_cost = init_ev.x;
        cur_ov = init_ev.y;
        best_ov = init_ev.y;
        pass_idx = 1.0;
        for (var i: u32 = 0u; i < nf; i = i + 1u) {
            chain_poses[best_base + i] = chain_poses[c_base + i];
        }
    } else {
        cur_ov = floor(score.w / 1024.0);
        pass_idx = (score.w - cur_ov * 1024.0) + 1.0;
    }

    let temp_ratio = f32(c & 63u) / 63.0;
    let temp = max(params.base_temp * pow(35.0, temp_ratio), 0.2);

    for (var step: u32 = 0u; step < params.steps_per_pass; step = step + 1u) {
        var idx: u32 = 0u;
        var old_pose = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var new_pose = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var proposed = false;

        if (cur_ov > 0.0 && rand_f32(&rng) < 0.88) {
            let start_i = pcg_next(&rng) % nf;
            var found_idx: u32 = 0xFFFFFFFFu;
            var found_other: u32 = 0xFFFFFFFFu;
            for (var ti: u32 = 0u; ti < nf; ti = ti + 1u) {
                let i = (start_i + ti) % nf;
                if (fp_descs[i].z > 0.5) {
                    continue;
                }
                let bi = fp_aabb(c_base, i);
                let pose_i = chain_poses[c_base + i];
                let start_j = pcg_next(&rng) % nf;
                for (var tj: u32 = 0u; tj < nf; tj = tj + 1u) {
                    let j = (start_j + tj) % nf;
                    if (j == i) {
                        continue;
                    }
                    let bj = fp_aabb(c_base, j);
                    if (bi.z > bj.x && bi.x < bj.z && bi.w > bj.y && bi.y < bj.w) {
                        let pose_j = chain_poses[c_base + j];
                        if (footprints_courtyards_intersect(i, pose_i, j, pose_j)) {
                            found_idx = i;
                            found_other = j;
                            break;
                        }
                    }
                }
                if (found_idx != 0xFFFFFFFFu) {
                    break;
                }
            }

            if (found_idx != 0xFFFFFFFFu) {
                idx = found_idx;
                old_pose = chain_poses[c_base + idx];
                let other_pose = chain_poses[c_base + found_other];
                new_pose = old_pose;
                let roll = rand_f32(&rng);
                if (roll < 0.65) {
                    let dx = old_pose.x - other_pose.x;
                    let dy = old_pose.y - other_pose.y;
                    let dist = sqrt(dx * dx + dy * dy);
                    var ux: f32;
                    var uy: f32;
                    if (dist > 1e-3) {
                        ux = dx / dist;
                        uy = dy / dist;
                    } else {
                        let ang = rand_f32(&rng) * 6.2831853;
                        ux = cos(ang);
                        uy = sin(ang);
                    }
                    let fan = (rand_f32(&rng) * 2.0 - 1.0) * 0.785398;
                    let cf = cos(fan);
                    let sf = sin(fan);
                    let rx = ux * cf - uy * sf;
                    let ry = ux * sf + uy * cf;
                    let push_dist = 0.6 + rand_f32(&rng) * 11.4;
                    new_pose.x = old_pose.x + rx * push_dist;
                    new_pose.y = old_pose.y + ry * push_dist;
                } else if (roll < 0.85) {
                    let r = 2.0 + rand_f32(&rng) * 8.0;
                    new_pose.x = old_pose.x + (rand_f32(&rng) * 2.0 - 1.0) * r;
                    new_pose.y = old_pose.y + (rand_f32(&rng) * 2.0 - 1.0) * r;
                } else {
                    let rot_step = f32((pcg_next(&rng) % 3u) + 1u) * 90.0;
                    new_pose.z = (old_pose.z + rot_step) % 360.0;
                }
                new_pose = clamp_pose_to_board_aabb(new_pose, fp_descs[idx]);
                proposed = true;
            }
        }

        if (!proposed) {
            idx = pcg_next(&rng) % nf;
            for (var tries: u32 = 0u; tries < 8u; tries = tries + 1u) {
                if (fp_descs[idx].z < 0.5) {
                    break;
                }
                idx = (idx + 1u) % nf;
            }
            if (fp_descs[idx].z > 0.5) {
                break;
            }

            old_pose = chain_poses[c_base + idx];
            new_pose = old_pose;
            let move_r = rand_f32(&rng);
            if (move_r < 0.72) {
                let shift = max(temp * 0.15, params.grid_step * 2.0);
                let dx = (rand_f32(&rng) * 2.0 - 1.0) * shift;
                let dy = (rand_f32(&rng) * 2.0 - 1.0) * shift;
                new_pose.x = snap_grid(
                    clamp(old_pose.x + dx, params.board_min_x + 1.2, params.board_max_x - 1.2),
                    params.grid_step
                );
                new_pose.y = snap_grid(
                    clamp(old_pose.y + dy, params.board_min_y + 1.2, params.board_max_y - 1.2),
                    params.grid_step
                );
            } else {
                let rot_step = f32((pcg_next(&rng) % 3u) + 1u) * 90.0;
                new_pose.z = (old_pose.z + rot_step) % 360.0;
            }
            if (cur_ov > 0.0) {
                new_pose = clamp_pose_to_board_aabb(new_pose, fp_descs[idx]);
            }
        }

        let d_ev = evaluate_chain_delta_e(c_base, nf, ne, idx, old_pose, new_pose);
        let delta = d_ev.x;
        let new_cost = cur_cost + delta;
        let new_ov = max(round(cur_ov + d_ev.y), 0.0);

        let accept = (new_ov < cur_ov) ||
            (new_ov == cur_ov && (delta <= 0.0 || rand_f32(&rng) < exp(-delta / temp)));

        if (accept) {
            chain_poses[c_base + idx] = new_pose;
            cur_cost = new_cost;
            cur_ov = new_ov;
            if (new_ov < best_ov || (new_ov == best_ov && new_cost < best_cost)) {
                best_cost = new_cost;
                best_ov = new_ov;
                for (var i: u32 = 0u; i < nf; i = i + 1u) {
                    chain_poses[best_base + i] = chain_poses[c_base + i];
                }
            }
        }
    }

    let ref_ev = evaluate_chain_state(c_base, nf, ne);
    cur_cost = ref_ev.x;
    cur_ov = ref_ev.y;
    if (cur_ov < best_ov || (cur_ov == best_ov && cur_cost < best_cost)) {
        best_cost = cur_cost;
        best_ov = cur_ov;
        for (var i: u32 = 0u; i < nf; i = i + 1u) {
            chain_poses[best_base + i] = chain_poses[c_base + i];
        }
    }

    chain_scores[c] = vec4<f32>(cur_cost, best_cost, best_ov, round(cur_ov) * 1024.0 + pass_idx);
}
"#;

/// WGSL compute shader for 3D multi-layer PCB routing grid cell traversal cost evaluation.
///
/// For each 3D routing cell `(x, y, layer)`, checks whether the layer is blocked by a dedicated
/// power/ground plane (`plane_layer_mask`) or whether any foreign-net obstacle lies within the
/// circular clearance radius `params.clearance_cells`. Blocked cells receive `-1.0`; traversable
/// cells receive `1.0 + congestion_penalty + history_cost + distance_field_penalty`.
///
/// # Entry Point & Workgroup Size
/// - `main` (`@workgroup_size(16, 16, 1)`): One invocation per 3D grid cell `(x, y, layer) = gid.xyz`.
///
/// # Bindings (`@group(0)`)
/// - `@binding(0)` (`storage, read`): `obstacles: array<u32>` — 3D obstacle net IDs (`[num_layers * height * width]`, `0` = free).
/// - `@binding(1)` (`storage, read`): `dist_field: array<f32>` — 3D Euclidean distance field (`[num_layers * height * width]`).
/// - `@binding(2)` (`storage, read`): `history_cost: array<f32>` — PathFinder historical congestion costs (`[num_layers * height * width]`).
/// - `@binding(3)` (`storage, read`): `usage_count: array<u32>` — Present net occupancy counts (`[num_layers * height * width]`).
/// - `@binding(4)` (`storage, read_write`): `cell_costs_out: array<f32>` — Output traversal cost per cell (`-1.0` if blocked, `>= 1.0` if legal).
/// - `@binding(5)` (`uniform`): `params: GpuRouteCostParams` — Grid dimensions, active `net_id`, `clearance_cells`, congestion weights, and `plane_layer_mask`.
const COMPUTE_3D_ROUTING_COST_WGSL: &str = r#"
struct GpuRouteCostParams {
    width: u32,
    height: u32,
    num_layers: u32,
    net_id: u32,
    clearance_cells: i32,
    present_congestion_factor: f32,
    obstacle_clearance_weight: f32,
    plane_layer_mask: u32,
}

@group(0) @binding(0) var<storage, read> obstacles: array<u32>;
@group(0) @binding(1) var<storage, read> dist_field: array<f32>;
@group(0) @binding(2) var<storage, read> history_cost: array<f32>;
@group(0) @binding(3) var<storage, read> usage_count: array<u32>;
@group(0) @binding(4) var<storage, read_write> cell_costs_out: array<f32>;
@group(0) @binding(5) var<uniform> params: GpuRouteCostParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let layer = gid.z;
    let w = i32(params.width);
    let h = i32(params.height);
    if (x >= w || y >= h || layer >= params.num_layers) {
        return;
    }

    let layer_stride = params.width * params.height;
    let layer_base = layer * layer_stride;
    let idx = layer_base + u32(y * w + x);

    let center_obs = obstacles[idx];
    let plane_blocked = ((params.plane_layer_mask >> layer) & 1u) != 0u;
    if (plane_blocked && (params.net_id == 0u || center_obs != params.net_id)) {
        cell_costs_out[idx] = -1.0;
        return;
    }

    let r = max(params.clearance_cells, 0);
    let r2 = r * r;
    var blocked: bool = false;
    for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
        if (blocked) {
            break;
        }
        let ny = y + dy;
        if (ny < 0 || ny >= h) {
            continue;
        }
        for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
            if (dx * dx + dy * dy <= r2) {
                let nx = x + dx;
                if (nx < 0 || nx >= w) {
                    continue;
                }
                let nidx = layer_base + u32(ny * w + nx);
                let obs = obstacles[nidx];
                if (obs != 0u && (params.net_id == 0u || obs != params.net_id)) {
                    blocked = true;
                    break;
                }
            }
        }
    }

    if (blocked) {
        cell_costs_out[idx] = -1.0;
        return;
    }

    let cong_pen = f32(usage_count[idx]) * params.present_congestion_factor;
    let hist_pen = history_cost[idx];
    var df_pen: f32 = 0.0;
    if (params.obstacle_clearance_weight > 0.0) {
        let df = max(dist_field[idx], 0.1);
        df_pen = params.obstacle_clearance_weight / df;
    }
    cell_costs_out[idx] = 1.0 + cong_pen + hist_pen + df_pen;
}
"#;

/// WGSL compute shader for a single synchronous 3D Lee/Moore wavefront relaxation step.
///
/// Relaxes each traversable cell `(x, y, layer)` against its 8 in-layer octilinear neighbors
/// (preventing diagonal corner-cutting across blocked orthogonal neighbors) and across all other
/// layers at `(x, y, ol)` with additive `params.via_cost`.
///
/// # Entry Point & Workgroup Size
/// - `main` (`@workgroup_size(16, 16, 1)`): One invocation per 3D grid cell `(x, y, layer) = gid.xyz`.
///
/// # Bindings (`@group(0)`)
/// - `@binding(0)` (`storage, read`): `cell_costs: array<f32>` — Precomputed per-cell traversal costs (`< 0.0` = blocked).
/// - `@binding(1)` (`storage, read`): `wavefront_in: array<f32>` — Input accumulated path cost grid from the previous iteration (`1.0e20` = unreached).
/// - `@binding(2)` (`storage, read_write`): `wavefront_out: array<f32>` — Output relaxed path cost grid for the current iteration.
/// - `@binding(3)` (`uniform`): `params: WavefrontUniformParams` — Grid dimensions (`width`, `height`, `num_layers`) and `via_cost`.
const WAVEFRONT_3D_RELAX_WGSL: &str = r#"
struct WavefrontUniformParams {
    width: u32,
    height: u32,
    num_layers: u32,
    via_cost: f32,
}

@group(0) @binding(0) var<storage, read> cell_costs: array<f32>;
@group(0) @binding(1) var<storage, read> wavefront_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> wavefront_out: array<f32>;
@group(0) @binding(3) var<uniform> params: WavefrontUniformParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let layer = gid.z;
    let w = i32(params.width);
    let h = i32(params.height);
    let nl = params.num_layers;

    if (x >= w || y >= h || layer >= nl) {
        return;
    }

    let layer_stride = params.width * params.height;
    let layer_base = layer * layer_stride;
    let idx = layer_base + u32(y * w + x);

    var best = wavefront_in[idx];
    let my_cost = cell_costs[idx];
    if (my_cost < 0.0) {
        wavefront_out[idx] = best;
        return;
    }

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        let ny = y + dy;
        if (ny < 0 || ny >= h) {
            continue;
        }
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            let nx = x + dx;
            if (nx < 0 || nx >= w) {
                continue;
            }
            let n_idx = layer_base + u32(ny * w + nx);
            let n_wf = wavefront_in[n_idx];
            if (n_wf >= 1.0e19) {
                continue;
            }
            let is_diag = dx != 0 && dy != 0;
            let step_len = select(1.0, 1.41421356, is_diag);
            if (is_diag) {
                let idx_h = layer_base + u32(y * w + nx);
                let idx_v = layer_base + u32(ny * w + x);
                if (cell_costs[idx_h] < 0.0 || cell_costs[idx_v] < 0.0) {
                    continue;
                }
            }
            let cand = n_wf + my_cost * step_len;
            if (cand < best) {
                best = cand;
            }
        }
    }

    for (var ol: u32 = 0u; ol < nl; ol = ol + 1u) {
        if (ol == layer) {
            continue;
        }
        let o_idx = ol * layer_stride + u32(y * w + x);
        let o_wf = wavefront_in[o_idx];
        if (o_wf < 1.0e19 && (cell_costs[o_idx] >= 0.0 || o_wf == 0.0)) {
            let cand = o_wf + params.via_cost;
            if (cand < best) {
                best = cand;
            }
        }
    }

    wavefront_out[idx] = best;
}
"#;

/// WGSL compute shader for backtracking a shortest 3D route path across a converged wavefront grid.
///
/// Traces backward from `(dst_x, dst_y, dst_layer)` to `(src_x, src_y, src_layer)` along the steepest
/// descent in `dist_grid` (considering 8 in-layer octilinear steps and inter-layer via transitions),
/// reverses the resulting path in-place so it is ordered from source to destination, and writes the
/// path length into `path_len_out[0]`.
///
/// # Entry Point & Workgroup Size
/// - `main` (`@workgroup_size(1, 1, 1)`): Single-thread (`gid == (0, 0, 0)`) serial path trace on GPU
///   to avoid transferring the full 3D wavefront distance volume back to host memory.
///
/// # Bindings (`@group(0)`)
/// - `@binding(0)` (`storage, read`): `dist_grid: array<f32>` — Converged 3D wavefront distance grid (`[num_layers * height * width]`).
/// - `@binding(1)` (`storage, read`): `cell_costs: array<f32>` — Per-cell traversal costs (`[num_layers * height * width]`).
/// - `@binding(2)` (`storage, read_write`): `path_out: array<u32>` — Output flattened `(x, y, layer)` triples (`[max_path_len * 3]`).
/// - `@binding(3)` (`storage, read_write`): `path_len_out: array<u32>` — Output single-element array storing the number of `(x, y, layer)` steps (`0` if unreachable).
/// - `@binding(4)` (`uniform`): `params: WavefrontBacktrackUniformParams` — Grid dimensions, `via_cost`, source/destination coordinates, and `max_path_len`.
const WAVEFRONT_3D_BACKTRACK_WGSL: &str = r#"
struct WavefrontBacktrackUniformParams {
    width: u32,
    height: u32,
    num_layers: u32,
    via_cost: f32,
    src_x: u32,
    src_y: u32,
    src_layer: u32,
    dst_x: u32,
    dst_y: u32,
    dst_layer: u32,
    max_path_len: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> dist_grid: array<f32>;
@group(0) @binding(1) var<storage, read> cell_costs: array<f32>;
@group(0) @binding(2) var<storage, read_write> path_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> path_len_out: array<u32>;
@group(0) @binding(4) var<uniform> params: WavefrontBacktrackUniformParams;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }

    let w = i32(params.width);
    let h = i32(params.height);
    let nl = i32(params.num_layers);

    if (i32(params.src_x) >= w || i32(params.src_y) >= h || i32(params.src_layer) >= nl ||
        i32(params.dst_x) >= w || i32(params.dst_y) >= h || i32(params.dst_layer) >= nl ||
        params.max_path_len == 0u) {
        path_len_out[0] = 0u;
        return;
    }

    let layer_stride = params.width * params.height;
    let dst_idx = params.dst_layer * layer_stride + params.dst_y * params.width + params.dst_x;
    if (dist_grid[dst_idx] >= 1.0e19) {
        path_len_out[0] = 0u;
        return;
    }

    var cx = i32(params.dst_x);
    var cy = i32(params.dst_y);
    var cl = i32(params.dst_layer);
    var step_count: u32 = 0u;

    loop {
        if (step_count >= params.max_path_len) {
            break;
        }
        path_out[step_count * 3u + 0u] = u32(cx);
        path_out[step_count * 3u + 1u] = u32(cy);
        path_out[step_count * 3u + 2u] = u32(cl);
        step_count = step_count + 1u;

        if (cx == i32(params.src_x) && cy == i32(params.src_y) && cl == i32(params.src_layer)) {
            let half = step_count / 2u;
            for (var i: u32 = 0u; i < half; i = i + 1u) {
                let j = step_count - 1u - i;
                for (var k: u32 = 0u; k < 3u; k = k + 1u) {
                    let tmp = path_out[i * 3u + k];
                    path_out[i * 3u + k] = path_out[j * 3u + k];
                    path_out[j * 3u + k] = tmp;
                }
            }
            path_len_out[0] = step_count;
            return;
        }

        let layer_base = u32(cl) * layer_stride;
        let cur_idx = layer_base + u32(cy * w + cx);
        let cur_dist = dist_grid[cur_idx];
        let my_cost = select(1.0, cell_costs[cur_idx], cell_costs[cur_idx] >= 0.0);

        var best_nx: i32 = -1;
        var best_ny: i32 = -1;
        var best_nl: i32 = -1;
        var best_score: f32 = 1.0e20;
        var best_ndist: f32 = 1.0e20;

        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            let ny = cy + dy;
            if (ny < 0 || ny >= h) {
                continue;
            }
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                let nx = cx + dx;
                if (nx < 0 || nx >= w) {
                    continue;
                }
                let n_idx = layer_base + u32(ny * w + nx);
                let n_dist = dist_grid[n_idx];
                if (n_dist >= cur_dist || n_dist >= 1.0e19) {
                    continue;
                }
                if (cell_costs[n_idx] < 0.0 && n_dist > 0.0) {
                    continue;
                }
                let is_diag = dx != 0 && dy != 0;
                let step_len = select(1.0, 1.41421356, is_diag);
                if (is_diag) {
                    let idx_h = layer_base + u32(cy * w + nx);
                    let idx_v = layer_base + u32(ny * w + cx);
                    if (cell_costs[idx_h] < 0.0 || cell_costs[idx_v] < 0.0) {
                        continue;
                    }
                }
                let cand_score = n_dist + my_cost * step_len;
                if (cand_score < best_score - 1.0e-4 ||
                    (abs(cand_score - best_score) <= 1.0e-4 && n_dist < best_ndist)) {
                    best_score = cand_score;
                    best_ndist = n_dist;
                    best_nx = nx;
                    best_ny = ny;
                    best_nl = cl;
                }
            }
        }

        for (var ol: i32 = 0; ol < nl; ol = ol + 1) {
            if (ol == cl) {
                continue;
            }
            let o_idx = u32(ol) * layer_stride + u32(cy * w + cx);
            let o_dist = dist_grid[o_idx];
            if (o_dist >= cur_dist || o_dist >= 1.0e19) {
                continue;
            }
            if (cell_costs[o_idx] < 0.0 && o_dist > 0.0) {
                continue;
            }
            let cand_score = o_dist + params.via_cost;
            if (cand_score < best_score - 1.0e-4 ||
                (abs(cand_score - best_score) <= 1.0e-4 && o_dist < best_ndist)) {
                best_score = cand_score;
                best_ndist = o_dist;
                best_nx = cx;
                best_ny = cy;
                best_nl = ol;
            }
        }

        if (best_nx < 0) {
            path_len_out[0] = 0u;
            return;
        }
        cx = best_nx;
        cy = best_ny;
        cl = best_nl;
    }

    path_len_out[0] = 0u;
}
"#;

/// WGSL compute shader for counting pairwise ratsnest segment crossings using a 2D Morton LBVH.
///
/// Each invocation `i = gid.x` traverses the flat binary LBVH (`bvh_nodes`) using an explicit
/// 32-entry traversal stack, pruning subtrees whose AABB does not intersect segment `segs[i]`
/// (`ray_aabb_intersects`) and counting strict geometric crossings (`segments_cross_bvh`) against
/// leaf segments `j > i`.
///
/// # Entry Point & Workgroup Size
/// - `main` (`@workgroup_size(64, 1, 1)`): One invocation per segment `i` (`0..params.count`).
///
/// # Bindings (`@group(0)`)
/// - `@binding(0)` (`storage, read`): `segs: array<vec4<f32>>` — Input 2D line segments `[x1, y1, x2, y2]` of length `params.count`.
/// - `@binding(1)` (`storage, read`): `bvh_nodes: array<GpuBvhNode>` — Flat 32-byte-per-node Morton LBVH over `segs` (`2 * count - 1` nodes).
/// - `@binding(2)` (`storage, read_write`): `crossings_out: array<u32>` — Output crossing counts per segment `i` (`j > i`).
/// - `@binding(3)` (`uniform`): `params: BvhQueryParams` — Number of segments (`count`).
const RT_BVH_SEGMENT_CROSSINGS_WGSL: &str = r#"
struct GpuBvhNode {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    left_or_prim: i32,
    right_or_count: i32,
    layer_mask: u32,
    net_id: u32,
}

struct BvhQueryParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> segs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bvh_nodes: array<GpuBvhNode>;
@group(0) @binding(2) var<storage, read_write> crossings_out: array<u32>;
@group(0) @binding(3) var<uniform> params: BvhQueryParams;

fn ccw_bvh(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> f32 {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

fn segments_cross_bvh(s1: vec4<f32>, s2: vec4<f32>) -> bool {
    let eps = 1e-3;
    if ((abs(s1.x - s2.x) < eps && abs(s1.y - s2.y) < eps) ||
        (abs(s1.x - s2.z) < eps && abs(s1.y - s2.w) < eps) ||
        (abs(s1.z - s2.x) < eps && abs(s1.w - s2.y) < eps) ||
        (abs(s1.z - s2.z) < eps && abs(s1.w - s2.w) < eps)) {
        return false;
    }
    let d1 = ccw_bvh(s1.x, s1.y, s1.z, s1.w, s2.x, s2.y);
    let d2 = ccw_bvh(s1.x, s1.y, s1.z, s1.w, s2.z, s2.w);
    let d3 = ccw_bvh(s2.x, s2.y, s2.z, s2.w, s1.x, s1.y);
    let d4 = ccw_bvh(s2.x, s2.y, s2.z, s2.w, s1.z, s1.w);
    return ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) &&
           ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0));
}

fn ray_aabb_intersects(s: vec4<f32>, b_min_x: f32, b_min_y: f32, b_max_x: f32, b_max_y: f32) -> bool {
    let s_min_x = min(s.x, s.z);
    let s_max_x = max(s.x, s.z);
    let s_min_y = min(s.y, s.w);
    let s_max_y = max(s.y, s.w);
    if (s_max_x < b_min_x || s_min_x > b_max_x || s_max_y < b_min_y || s_min_y > b_max_y) {
        return false;
    }
    let dx = s.z - s.x;
    let dy = s.w - s.y;
    var tmin: f32 = 0.0;
    var tmax: f32 = 1.0;
    if (abs(dx) > 1e-6) {
        let inv_dx = 1.0 / dx;
        let tx1 = (b_min_x - s.x) * inv_dx;
        let tx2 = (b_max_x - s.x) * inv_dx;
        tmin = max(tmin, min(tx1, tx2));
        tmax = min(tmax, max(tx1, tx2));
    }
    if (abs(dy) > 1e-6) {
        let inv_dy = 1.0 / dy;
        let ty1 = (b_min_y - s.y) * inv_dy;
        let ty2 = (b_max_y - s.y) * inv_dy;
        tmin = max(tmin, min(ty1, ty2));
        tmax = min(tmax, max(ty1, ty2));
    }
    return tmax >= tmin;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }

    let s1 = segs[i];
    var crossings: u32 = 0u;
    var stack: array<i32, 32>;
    var stack_ptr: i32 = 0;
    stack[0] = 0;
    stack_ptr = 1;

    loop {
        if (stack_ptr <= 0) {
            break;
        }
        stack_ptr = stack_ptr - 1;
        let node_idx = stack[stack_ptr];
        let node = bvh_nodes[u32(node_idx)];
        if (node.right_or_count == 0) {
            continue;
        }
        if (!ray_aabb_intersects(s1, node.min_x, node.min_y, node.max_x, node.max_y)) {
            continue;
        }
        if (node.right_or_count > 0) {
            let j = u32(node.left_or_prim);
            if (j > i) {
                if (segments_cross_bvh(s1, segs[j])) {
                    crossings = crossings + 1u;
                }
            }
        } else {
            let left_idx = node.left_or_prim;
            let right_idx = -node.right_or_count;
            if (stack_ptr + 2 <= 32) {
                stack[stack_ptr] = right_idx;
                stack[stack_ptr + 1] = left_idx;
                stack_ptr = stack_ptr + 2;
            }
        }
    }

    crossings_out[i] = crossings;
}
"#;

/// WGSL compute shader for counting ratsnest crossings across topological corridor cut lines using a 2D Morton LBVH.
///
/// Each invocation `c_idx = gid.x` traverses the LBVH (`bvh_nodes`) built over the board's ratsnest
/// segments (`segs`) and counts how many segments cross the query corridor cut line `cuts[c_idx]`.
///
/// # Entry Point & Workgroup Size
/// - `main` (`@workgroup_size(64, 1, 1)`): One invocation per corridor cut segment `c_idx` (`0..params.count`).
///
/// # Bindings (`@group(0)`)
/// - `@binding(0)` (`storage, read`): `segs: array<vec4<f32>>` — Ratsnest airwire segments `[x1, y1, x2, y2]`.
/// - `@binding(1)` (`storage, read`): `bvh_nodes: array<GpuBvhNode>` — Flat Morton LBVH constructed over `segs`.
/// - `@binding(2)` (`storage, read`): `cuts: array<vec4<f32>>` — Query corridor cut segments `[x1, y1, x2, y2]` of length `params.count`.
/// - `@binding(3)` (`storage, read_write`): `cuts_out: array<u32>` — Output number of ratsnest crossings for each corridor cut (`[params.count]`).
/// - `@binding(4)` (`uniform`): `params: BvhQueryParams` — Number of corridor cut queries (`count`).
const RT_BVH_CORRIDOR_CUTS_WGSL: &str = r#"
struct GpuBvhNode {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    left_or_prim: i32,
    right_or_count: i32,
    layer_mask: u32,
    net_id: u32,
}

struct BvhQueryParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> segs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bvh_nodes: array<GpuBvhNode>;
@group(0) @binding(2) var<storage, read> cuts: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> cuts_out: array<u32>;
@group(0) @binding(4) var<uniform> params: BvhQueryParams;

fn ccw_bvh(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> f32 {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

fn segments_cross_bvh(s1: vec4<f32>, s2: vec4<f32>) -> bool {
    let eps = 1e-3;
    if ((abs(s1.x - s2.x) < eps && abs(s1.y - s2.y) < eps) ||
        (abs(s1.x - s2.z) < eps && abs(s1.y - s2.w) < eps) ||
        (abs(s1.z - s2.x) < eps && abs(s1.w - s2.y) < eps) ||
        (abs(s1.z - s2.z) < eps && abs(s1.w - s2.w) < eps)) {
        return false;
    }
    let d1 = ccw_bvh(s1.x, s1.y, s1.z, s1.w, s2.x, s2.y);
    let d2 = ccw_bvh(s1.x, s1.y, s1.z, s1.w, s2.z, s2.w);
    let d3 = ccw_bvh(s2.x, s2.y, s2.z, s2.w, s1.x, s1.y);
    let d4 = ccw_bvh(s2.x, s2.y, s2.z, s2.w, s1.z, s1.w);
    return ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) &&
           ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0));
}

fn ray_aabb_intersects(s: vec4<f32>, b_min_x: f32, b_min_y: f32, b_max_x: f32, b_max_y: f32) -> bool {
    let s_min_x = min(s.x, s.z);
    let s_max_x = max(s.x, s.z);
    let s_min_y = min(s.y, s.w);
    let s_max_y = max(s.y, s.w);
    if (s_max_x < b_min_x || s_min_x > b_max_x || s_max_y < b_min_y || s_min_y > b_max_y) {
        return false;
    }
    let dx = s.z - s.x;
    let dy = s.w - s.y;
    var tmin: f32 = 0.0;
    var tmax: f32 = 1.0;
    if (abs(dx) > 1e-6) {
        let inv_dx = 1.0 / dx;
        let tx1 = (b_min_x - s.x) * inv_dx;
        let tx2 = (b_max_x - s.x) * inv_dx;
        tmin = max(tmin, min(tx1, tx2));
        tmax = min(tmax, max(tx1, tx2));
    }
    if (abs(dy) > 1e-6) {
        let inv_dy = 1.0 / dy;
        let ty1 = (b_min_y - s.y) * inv_dy;
        let ty2 = (b_max_y - s.y) * inv_dy;
        tmin = max(tmin, min(ty1, ty2));
        tmax = min(tmax, max(ty1, ty2));
    }
    return tmax >= tmin;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c_idx = gid.x;
    if (c_idx >= params.count) {
        return;
    }

    let cut = cuts[c_idx];
    var crossings: u32 = 0u;
    var stack: array<i32, 32>;
    var stack_ptr: i32 = 0;
    stack[0] = 0;
    stack_ptr = 1;

    loop {
        if (stack_ptr <= 0) {
            break;
        }
        stack_ptr = stack_ptr - 1;
        let node_idx = stack[stack_ptr];
        let node = bvh_nodes[u32(node_idx)];
        if (node.right_or_count == 0) {
            continue;
        }
        if (!ray_aabb_intersects(cut, node.min_x, node.min_y, node.max_x, node.max_y)) {
            continue;
        }
        if (node.right_or_count > 0) {
            let j = u32(node.left_or_prim);
            if (segments_cross_bvh(cut, segs[j])) {
                crossings = crossings + 1u;
            }
        } else {
            let left_idx = node.left_or_prim;
            let right_idx = -node.right_or_count;
            if (stack_ptr + 2 <= 32) {
                stack[stack_ptr] = right_idx;
                stack[stack_ptr + 1] = left_idx;
                stack_ptr = stack_ptr + 2;
            }
        }
    }

    cuts_out[c_idx] = crossings;
}
"#;

/// WGSL compute shader for batch line-of-sight DRC clearance ray queries against a 2D Morton LBVH.
///
/// Each invocation `q_idx = gid.x` traverses the obstacle LBVH (`bvh_nodes`), skipping subtrees
/// that do not overlap `queries[q_idx].layer_mask`, share the same non-zero `net_id`, or miss the
/// clearance-inflated AABB `[min_x - r, min_y - r, max_x + r, max_y + r]`. Counts all leaf obstacle
/// AABB hits along the query ray.
///
/// # Entry Point & Workgroup Size
/// - `main` (`@workgroup_size(64, 1, 1)`): One invocation per DRC ray query `q_idx` (`0..params.count`).
///
/// # Bindings (`@group(0)`)
/// - `@binding(0)` (`storage, read`): `bvh_nodes: array<GpuBvhNode>` — Flat Morton LBVH over obstacle AABBs.
/// - `@binding(1)` (`storage, read`): `queries: array<GpuDrcRayQuery>` — Input DRC ray queries of length `params.count`.
/// - `@binding(2)` (`storage, read_write`): `violations_out: array<u32>` — Output violation count per query (`[params.count]`).
/// - `@binding(3)` (`uniform`): `params: BvhQueryParams` — Number of ray queries (`count`).
const RT_BVH_LINE_OF_SIGHT_DRC_WGSL: &str = r#"
struct GpuBvhNode {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    left_or_prim: i32,
    right_or_count: i32,
    layer_mask: u32,
    net_id: u32,
}

struct GpuDrcRayQuery {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    clearance: f32,
    layer_mask: u32,
    net_id: u32,
    _pad: u32,
}

struct BvhQueryParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> bvh_nodes: array<GpuBvhNode>;
@group(0) @binding(1) var<storage, read> queries: array<GpuDrcRayQuery>;
@group(0) @binding(2) var<storage, read_write> violations_out: array<u32>;
@group(0) @binding(3) var<uniform> params: BvhQueryParams;

fn ray_aabb_intersects(s: vec4<f32>, b_min_x: f32, b_min_y: f32, b_max_x: f32, b_max_y: f32) -> bool {
    let s_min_x = min(s.x, s.z);
    let s_max_x = max(s.x, s.z);
    let s_min_y = min(s.y, s.w);
    let s_max_y = max(s.y, s.w);
    if (s_max_x < b_min_x || s_min_x > b_max_x || s_max_y < b_min_y || s_min_y > b_max_y) {
        return false;
    }
    let dx = s.z - s.x;
    let dy = s.w - s.y;
    var tmin: f32 = 0.0;
    var tmax: f32 = 1.0;
    if (abs(dx) > 1e-6) {
        let inv_dx = 1.0 / dx;
        let tx1 = (b_min_x - s.x) * inv_dx;
        let tx2 = (b_max_x - s.x) * inv_dx;
        tmin = max(tmin, min(tx1, tx2));
        tmax = min(tmax, max(tx1, tx2));
    }
    if (abs(dy) > 1e-6) {
        let inv_dy = 1.0 / dy;
        let ty1 = (b_min_y - s.y) * inv_dy;
        let ty2 = (b_max_y - s.y) * inv_dy;
        tmin = max(tmin, min(ty1, ty2));
        tmax = min(tmax, max(ty1, ty2));
    }
    return tmax >= tmin;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_idx = gid.x;
    if (q_idx >= params.count) {
        return;
    }

    let q = queries[q_idx];
    let ray_seg = vec4<f32>(q.x1, q.y1, q.x2, q.y2);
    let r = max(q.clearance, 0.0);
    var violations: u32 = 0u;
    var stack: array<i32, 32>;
    var stack_ptr: i32 = 0;
    stack[0] = 0;
    stack_ptr = 1;

    loop {
        if (stack_ptr <= 0) {
            break;
        }
        stack_ptr = stack_ptr - 1;
        let node_idx = stack[stack_ptr];
        let node = bvh_nodes[u32(node_idx)];
        if (node.right_or_count == 0) {
            continue;
        }
        if ((node.layer_mask & q.layer_mask) == 0u) {
            continue;
        }
        if (q.net_id != 0u && node.net_id == q.net_id) {
            continue;
        }
        if (!ray_aabb_intersects(
                ray_seg,
                node.min_x - r,
                node.min_y - r,
                node.max_x + r,
                node.max_y + r
            )) {
            continue;
        }
        if (node.right_or_count > 0) {
            violations = violations + 1u;
        } else {
            let left_idx = node.left_or_prim;
            let right_idx = -node.right_or_count;
            if (stack_ptr + 2 <= 32) {
                stack[stack_ptr] = right_idx;
                stack[stack_ptr + 1] = left_idx;
                stack_ptr = stack_ptr + 2;
            }
        }
    }

    violations_out[q_idx] = violations;
}
"#;

/// WGSL compute shader for exact continuous-geometry batch DRC evaluation of candidate vias and track segments.
///
/// Evaluates `params.num_candidates` candidate vias (`kind == 0`) or track segments (`kind == 1`)
/// against all board pads (`pads`), routed tracks (`tracks`), and existing vias (`vias`), enforcing
/// symmetric bidirectional copper clearances:
/// $$\text{req\_dist} = r_{\text{cand}} + \max(\text{clearance}_{\text{cand}}, \text{clearance}_{\text{obs}}) + r_{\text{obs}}$$
/// Supports circular (`shape == 1`), oblong/oval (`shape == 2`), and rotated rectangular (`shape == 0`)
/// pads as well as same-net SMD pad overlap and micro-via / standard-via spacing rules.
///
/// # Entry Point & Workgroup Size
/// - `main` (`@workgroup_size(64, 1, 1)`): One invocation per DRC candidate `c = gid.x` (`0..params.num_candidates`).
///
/// # Bindings (`@group(0)`)
/// - `@binding(0)` (`storage, read`): `pads: array<GpuDrcPad>` — Board pads (`[params.num_pads]`).
/// - `@binding(1)` (`storage, read`): `tracks: array<GpuDrcTrack>` — Routed track segments (`[params.num_tracks]`).
/// - `@binding(2)` (`storage, read`): `vias: array<GpuDrcVia>` — Existing vias (`[params.num_vias]`).
/// - `@binding(3)` (`storage, read`): `candidates: array<GpuDrcCandidate>` — Candidate vias or track segments (`[params.num_candidates]`).
/// - `@binding(4)` (`storage, read_write`): `results_out: array<u32>` — Output legality flags (`1` = DRC clean, `0` = violation) of length `params.num_candidates`.
/// - `@binding(5)` (`uniform`): `params: DrcBatchParams` — Counts of pads, tracks, vias, and candidates.
const BATCH_DRC_EVAL_WGSL: &str = r#"
struct DrcBatchParams {
    num_pads: u32,
    num_tracks: u32,
    num_vias: u32,
    num_candidates: u32,
}

struct GpuDrcPad {
    x: f32,
    y: f32,
    hw: f32,
    hh: f32,
    rot_rad: f32,
    clearance: f32,
    layer_mask: u32,
    net_id: u32,
}

struct GpuDrcTrack {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    half_width: f32,
    clearance: f32,
    layer: u32,
    net_id: u32,
}

struct GpuDrcVia {
    x: f32,
    y: f32,
    radius: f32,
    clearance: f32,
    net_id: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct GpuDrcCandidate {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    radius_or_half_width: f32,
    clearance: f32,
    layer: u32,
    net_id: u32,
    kind: u32,
    ignore_same_net_pads: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> pads: array<GpuDrcPad>;
@group(0) @binding(1) var<storage, read> tracks: array<GpuDrcTrack>;
@group(0) @binding(2) var<storage, read> vias: array<GpuDrcVia>;
@group(0) @binding(3) var<storage, read> candidates: array<GpuDrcCandidate>;
@group(0) @binding(4) var<storage, read_write> results_out: array<u32>;
@group(0) @binding(5) var<uniform> params: DrcBatchParams;

fn drc_dist_point_to_seg(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-6) {
        let ex = px - x1;
        let ey = py - y1;
        return sqrt(ex * ex + ey * ey);
    }
    let t = clamp(((px - x1) * dx + (py - y1) * dy) / len_sq, 0.0, 1.0);
    let ex = px - (x1 + t * dx);
    let ey = py - (y1 + t * dy);
    return sqrt(ex * ex + ey * ey);
}

fn drc_segs_intersect(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32, dx: f32, dy: f32) -> bool {
    let d1 = (dy - cy) * (ax - cx) - (dx - cx) * (ay - cy);
    let d2 = (dy - cy) * (bx - cx) - (dx - cx) * (by - cy);
    let d3 = (by - ay) * (cx - ax) - (bx - ax) * (cy - ay);
    let d4 = (by - ay) * (dx - ax) - (bx - ax) * (dy - ay);
    return ((d1 > 1e-7 && d2 < -1e-7) || (d1 < -1e-7 && d2 > 1e-7)) &&
           ((d3 > 1e-7 && d4 < -1e-7) || (d3 < -1e-7 && d4 > 1e-7));
}

fn drc_dist_seg_to_seg(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32, dx: f32, dy: f32) -> f32 {
    if (drc_segs_intersect(ax, ay, bx, by, cx, cy, dx, dy)) {
        return 0.0;
    }
    let d1 = drc_dist_point_to_seg(ax, ay, cx, cy, dx, dy);
    let d2 = drc_dist_point_to_seg(bx, by, cx, cy, dx, dy);
    let d3 = drc_dist_point_to_seg(cx, cy, ax, ay, bx, by);
    let d4 = drc_dist_point_to_seg(dx, dy, ax, ay, bx, by);
    return min(min(d1, d2), min(d3, d4));
}

fn drc_dist_point_to_pad(px: f32, py: f32, pad: GpuDrcPad) -> f32 {
    let shape = pad.layer_mask >> 16u;
    let rx = px - pad.x;
    let ry = py - pad.y;
    if (shape == 1u) {
        let r = max(pad.hw, pad.hh);
        return max(sqrt(rx * rx + ry * ry) - r, 0.0);
    }
    let c = cos(-pad.rot_rad);
    let s = sin(-pad.rot_rad);
    let lx = rx * c - ry * s;
    let ly = rx * s + ry * c;
    if (shape == 2u) {
        let r = min(pad.hw, pad.hh);
        if (pad.hw >= pad.hh) {
            let cap_dx = pad.hw - r;
            let cx = clamp(lx, -cap_dx, cap_dx);
            let dx = lx - cx;
            return max(sqrt(dx * dx + ly * ly) - r, 0.0);
        } else {
            let cap_dy = pad.hh - r;
            let cy = clamp(ly, -cap_dy, cap_dy);
            let dy = ly - cy;
            return max(sqrt(lx * lx + dy * dy) - r, 0.0);
        }
    }
    let dx = max(abs(lx) - pad.hw, 0.0);
    let dy = max(abs(ly) - pad.hh, 0.0);
    return sqrt(dx * dx + dy * dy);
}

fn drc_dist_seg_to_pad(x1: f32, y1: f32, x2: f32, y2: f32, pad: GpuDrcPad) -> f32 {
    if ((pad.layer_mask >> 16u) == 1u) {
        let r = max(pad.hw, pad.hh);
        return max(drc_dist_point_to_seg(pad.x, pad.y, x1, y1, x2, y2) - r, 0.0);
    }
    let c = cos(-pad.rot_rad);
    let s = sin(-pad.rot_rad);
    let rx1 = x1 - pad.x;
    let ry1 = y1 - pad.y;
    let lsx = rx1 * c - ry1 * s;
    let lsy = rx1 * s + ry1 * c;
    let rx2 = x2 - pad.x;
    let ry2 = y2 - pad.y;
    let lex = rx2 * c - ry2 * s;
    let ley = rx2 * s + ry2 * c;
    let hw = pad.hw;
    let hh = pad.hh;

    if (abs(lsx) <= hw && abs(lsy) <= hh) {
        return 0.0;
    }
    if (abs(lex) <= hw && abs(ley) <= hh) {
        return 0.0;
    }

    if (drc_segs_intersect(lsx, lsy, lex, ley, -hw, -hh, hw, -hh) ||
        drc_segs_intersect(lsx, lsy, lex, ley, hw, -hh, hw, hh) ||
        drc_segs_intersect(lsx, lsy, lex, ley, hw, hh, -hw, hh) ||
        drc_segs_intersect(lsx, lsy, lex, ley, -hw, hh, -hw, -hh)) {
        return 0.0;
    }

    let dx1 = max(abs(lsx) - hw, 0.0);
    let dy1 = max(abs(lsy) - hh, 0.0);
    let d1 = sqrt(dx1 * dx1 + dy1 * dy1);

    let dx2 = max(abs(lex) - hw, 0.0);
    let dy2 = max(abs(ley) - hh, 0.0);
    let d2 = sqrt(dx2 * dx2 + dy2 * dy2);

    let d3 = drc_dist_point_to_seg(-hw, -hh, lsx, lsy, lex, ley);
    let d4 = drc_dist_point_to_seg(hw, -hh, lsx, lsy, lex, ley);
    let d5 = drc_dist_point_to_seg(hw, hh, lsx, lsy, lex, ley);
    let d6 = drc_dist_point_to_seg(-hw, hh, lsx, lsy, lex, ley);

    return min(min(min(d1, d2), min(d3, d4)), min(d5, d6));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.num_candidates) {
        return;
    }

    let cand = candidates[c];
    let eps = 1e-4;

    if (cand.kind == 0u) {
        for (var i: u32 = 0u; i < params.num_pads; i = i + 1u) {
            let p = pads[i];
            if (p.net_id != cand.net_id) {
                let req_dist = cand.radius_or_half_width + max(cand.clearance, p.clearance);
                let d = drc_dist_point_to_pad(cand.x1, cand.y1, p);
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            } else if (cand.ignore_same_net_pads == 0u && (p.layer_mask & 0xFFFFu) != 0xFFFFu) {
                let d = drc_dist_point_to_pad(cand.x1, cand.y1, p);
                if (d < cand.radius_or_half_width - eps) {
                    results_out[c] = 0u;
                    return;
                }
            }
        }

        for (var i: u32 = 0u; i < params.num_tracks; i = i + 1u) {
            let t = tracks[i];
            if (t.net_id != cand.net_id) {
                let req_dist = cand.radius_or_half_width + max(cand.clearance, t.clearance) + t.half_width;
                let d = drc_dist_point_to_seg(cand.x1, cand.y1, t.x1, t.y1, t.x2, t.y2);
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            }
        }

        for (var i: u32 = 0u; i < params.num_vias; i = i + 1u) {
            let v = vias[i];
            let dx = cand.x1 - v.x;
            let dy = cand.y1 - v.y;
            let d = sqrt(dx * dx + dy * dy);
            if (d < eps) {
                continue;
            }
            if (cand.pad0 != 0u) {
                let idx = cand.x2 - v.x;
                let idy = cand.y2 - v.y;
                if (sqrt(idx * idx + idy * idy) < eps) {
                    continue;
                }
            }
            if (v.net_id != cand.net_id) {
                let req_dist = cand.radius_or_half_width + max(cand.clearance, v.clearance) + v.radius;
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            } else {
                let req_dist = select(0.55, 0.25, cand.radius_or_half_width <= 0.18 && v.radius <= 0.18);
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            }
        }
    } else {
        let layer_bit = select(0u, 1u << (cand.layer & 31u), cand.layer < 32u);
        for (var i: u32 = 0u; i < params.num_pads; i = i + 1u) {
            let p = pads[i];
            if ((p.layer_mask & layer_bit) != 0u && p.net_id != cand.net_id) {
                let req_dist = cand.radius_or_half_width + max(cand.clearance, p.clearance);
                let d = drc_dist_seg_to_pad(cand.x1, cand.y1, cand.x2, cand.y2, p);
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            }
        }

        for (var i: u32 = 0u; i < params.num_tracks; i = i + 1u) {
            let t = tracks[i];
            if (t.layer == cand.layer && t.net_id != cand.net_id) {
                let req_dist = cand.radius_or_half_width + max(cand.clearance, t.clearance) + t.half_width;
                let d = drc_dist_seg_to_seg(cand.x1, cand.y1, cand.x2, cand.y2, t.x1, t.y1, t.x2, t.y2);
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            }
        }

        for (var i: u32 = 0u; i < params.num_vias; i = i + 1u) {
            let v = vias[i];
            if (v.net_id != cand.net_id) {
                let req_dist = cand.radius_or_half_width + max(cand.clearance, v.clearance) + v.radius;
                let d = drc_dist_point_to_seg(v.x, v.y, cand.x1, cand.y1, cand.x2, cand.y2);
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            }
        }
    }

    results_out[c] = 1u;
}
"#;

/// Cache-aligned 32-byte binary Linear Bounding Volume Hierarchy (LBVH) node for GPU and CPU traversal.
///
/// Represents either an internal bounding hierarchy node (`right_or_count < 0`) or a leaf primitive
/// (`right_or_count > 0`) in a flat depth-first LBVH array constructed by [`build_gpu_lbvh`].
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBvhNode {
    /// Minimum X coordinate of the node's axis-aligned bounding box (mm).
    pub min_x: f32,
    /// Minimum Y coordinate of the node's axis-aligned bounding box (mm).
    pub min_y: f32,
    /// Maximum X coordinate of the node's axis-aligned bounding box (mm).
    pub max_x: f32,
    /// Maximum Y coordinate of the node's axis-aligned bounding box (mm).
    pub max_y: f32,
    /// For internal nodes (`right_or_count < 0`), the index of the left child node;
    /// for leaf nodes (`right_or_count > 0`), the original primitive index in the input slice.
    pub left_or_prim: i32,
    /// Node type discriminator and right-child pointer:
    /// - `> 0`: Leaf node containing `right_or_count` primitives (always `1`).
    /// - `< 0`: Internal node whose right child resides at index `-right_or_count`.
    /// - `== 0`: Empty sentinel node.
    pub right_or_count: i32,
    /// Bitmask of copper layers occupied by the primitive(s) in this subtree (`1 << layer`).
    pub layer_mask: u32,
    /// Electrical net ID (`> 0`) if all primitives in this subtree belong to the same net, or `0` if mixed/unassigned.
    pub net_id: u32,
}

/// GPU line-of-sight Design Rule Check (DRC) ray query descriptor (`32` bytes, `#[repr(C)]`).
///
/// Used by [`VulkanEngine::evaluate_drc_rt_bvh`] and [`evaluate_drc_brute_force_cpu`] to count
/// clearance-inflated obstacle AABB intersections along a 2D line segment `(x1, y1) -> (x2, y2)`.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDrcRayQuery {
    /// Start X coordinate of the query ray segment (mm).
    pub x1: f32,
    /// Start Y coordinate of the query ray segment (mm).
    pub y1: f32,
    /// End X coordinate of the query ray segment (mm).
    pub x2: f32,
    /// End Y coordinate of the query ray segment (mm).
    pub y2: f32,
    /// Radial copper clearance expansion applied to obstacle AABBs (mm).
    pub clearance: f32,
    /// Bitmask of copper layers queried by this ray (`1 << layer`).
    pub layer_mask: u32,
    /// Active electrical net ID (`0` = collide with all nets; `> 0` = ignore same-net obstacles).
    pub net_id: u32,
    /// Explicit 4-byte alignment padding for 16-byte WGSL uniform/storage alignment.
    pub _pad: u32,
}

/// Uniform buffer parameters for GPU LBVH query dispatches (`16` bytes, `#[repr(C)]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BvhQueryParams {
    /// Number of active query rays, segments, or corridor cuts in the dispatch batch.
    count: u32,
    /// Alignment padding word 0.
    _pad0: u32,
    /// Alignment padding word 1.
    _pad1: u32,
    /// Alignment padding word 2.
    _pad2: u32,
}

/// Timing and speedup metrics comparing LBVH-accelerated vs brute-force SIMT/CPU DRC and crossing queries.
///
/// Returned by [`VulkanEngine::benchmark_rt_bvh_vs_simt`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RtBvhBenchmarkResult {
    /// Execution time (in microseconds) of the LBVH-accelerated DRC ray query pass.
    pub bvh_drc_us: f64,
    /// Execution time (in microseconds) of the brute-force DRC ray query baseline.
    pub simt_drc_us: f64,
    /// Execution time (in microseconds) of the LBVH-accelerated ratsnest segment crossing pass.
    pub bvh_crossings_us: f64,
    /// Execution time (in microseconds) of the $O(N^2)$ SIMT ratsnest segment crossing baseline.
    pub simt_crossings_us: f64,
    /// Relative speedup ratio (`simt_drc_us / bvh_drc_us`) for DRC ray queries.
    pub drc_speedup: f64,
    /// Relative speedup ratio (`simt_crossings_us / bvh_crossings_us`) for segment crossing queries.
    pub crossings_speedup: f64,
    /// Whether the active GPU adapter supports hardware ray-tracing / ray-query acceleration (`VK_KHR_ray_query` or OptiX).
    pub rt_hardware_available: bool,
}

/// Interleaves the bits of 16-bit quantized coordinates `(x, y)` into a 32-bit 2D Morton Z-order code.
///
/// Used to spatially sort 2D primitive centroids along a Z-order space-filling curve prior to
/// top-down binary LBVH construction in [`build_gpu_lbvh`].
///
/// # Arguments
/// * `x` - Quantized 16-bit X coordinate (`0..=65535`).
/// * `y` - Quantized 16-bit Y coordinate (`0..=65535`).
///
/// # Returns
/// A 32-bit Morton code with `x` bits in even positions and `y` bits in odd positions.
///
/// # Side Effects
/// None (pure bit-manipulation function).
#[inline]
pub fn morton2d_encode(x: u16, y: u16) -> u32 {
    /// Spreads the lower 16 bits of `n` into the even bit positions of a 32-bit word.
    fn part1by1(mut n: u32) -> u32 {
        n &= 0x0000ffff;
        n = (n ^ (n << 8)) & 0x00ff00ff;
        n = (n ^ (n << 4)) & 0x0f0f0f0f;
        n = (n ^ (n << 2)) & 0x33333333;
        (n ^ (n << 1)) & 0x55555555
    }
    part1by1(x as u32) | (part1by1(y as u32) << 1)
}

/// Computes a fast 64-bit FNV-1a-style fingerprint over a slice of LBVH input primitives for cache lookup.
///
/// Samples up to 64 strided primitives plus the final primitive, mixing their IEEE-754 AABB bit
/// patterns, `layer_mask`, and `net_id` into a 64-bit hash key used by [`build_gpu_lbvh`] and
/// [`VulkanEngine::build_cached_lbvh`].
///
/// # Arguments
/// * `primitives` - Slice of `(aabb, layer_mask, net_id)` tuples where `aabb` is `[min_x, min_y, max_x, max_y]`.
///
/// # Returns
/// A 64-bit hash value uniquely identifying the length and sampled geometry of `primitives`.
///
/// # Side Effects
/// None.
#[inline]
pub fn hash_lbvh_primitives(primitives: &[([f32; 4], u32, u32)]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325 ^ (primitives.len() as u64);
    let mix = |mut acc: u64, v: u64| -> u64 {
        acc ^= v;
        acc.wrapping_mul(0x100000001b3)
    };
    let n = primitives.len();
    if n == 0 {
        return h;
    }
    let stride = (n / 64).max(1);
    let mut idx = 0usize;
    while idx < n {
        let (aabb, lm, net) = primitives[idx];
        let b0 = (aabb[0].to_bits() as u64) | ((aabb[1].to_bits() as u64) << 32);
        let b1 = (aabb[2].to_bits() as u64) | ((aabb[3].to_bits() as u64) << 32);
        let b2 = (lm as u64) | ((net as u64) << 32) ^ ((idx as u64) << 16);
        h = mix(h, b0);
        h = mix(h, b1);
        h = mix(h, b2);
        idx += stride;
    }
    let (aabb, lm, net) = primitives[n - 1];
    h = mix(h, (aabb[0].to_bits() as u64) | ((aabb[1].to_bits() as u64) << 32));
    h = mix(h, (aabb[2].to_bits() as u64) | ((aabb[3].to_bits() as u64) << 32));
    h = mix(h, (lm as u64) | ((net as u64) << 32));
    h
}

thread_local! {
    /// Per-thread single-entry LBVH cache storing `(primitives_hash, nodes)` to avoid redundant rebuilds.
    static THREAD_LBVH_CACHE: std::cell::RefCell<Option<(u64, Vec<GpuBvhNode>)>> = const { std::cell::RefCell::new(None) };
}
/// Global atomic counter tracking the total number of full LBVH tree constructions executed across all threads.
static GLOBAL_LBVH_BUILDS: AtomicU64 = AtomicU64::new(0);
/// Global atomic counter tracking the total number of thread-local LBVH cache hits across all threads.
static GLOBAL_LBVH_HITS: AtomicU64 = AtomicU64::new(0);

/// Returns global LBVH construction and thread-local cache hit counters.
///
/// # Returns
/// A tuple `(builds, hits)` containing the cumulative number of LBVH builds (`GLOBAL_LBVH_BUILDS`)
/// and thread-local cache hits (`GLOBAL_LBVH_HITS`).
///
/// # Side Effects
/// Performs relaxed atomic loads from `GLOBAL_LBVH_BUILDS` and `GLOBAL_LBVH_HITS`.
pub fn lbvh_cache_stats() -> (u64, u64) {
    (
        GLOBAL_LBVH_BUILDS.load(Ordering::Relaxed),
        GLOBAL_LBVH_HITS.load(Ordering::Relaxed),
    )
}

/// Builds a flat, cache-aligned 32-byte-per-node binary LBVH (`Vec<GpuBvhNode>`) sorted by
/// 2D Morton Z-order curve over the input primitives `(aabb, layer_mask, net_id)`.
///
/// Quantizes primitive centroids into a $65536 \times 65536$ grid, sorts them by [`morton2d_encode`]
/// (using parallel Rayon sorting when `primitives.len() >= 256`), and recursively partitions the
/// sorted range into a pre-order binary tree of `2 * N - 1` nodes.
///
/// # Arguments
/// * `primitives` - Slice of `(aabb, layer_mask, net_id)` tuples where `aabb` is `[min_x, min_y, max_x, max_y]`.
///
/// # Returns
/// A `Vec<GpuBvhNode>` of length `2 * primitives.len() - 1` (or a single empty sentinel node if `primitives` is empty).
///
/// # Side Effects
/// Reads and updates the calling thread's `THREAD_LBVH_CACHE` and increments either `GLOBAL_LBVH_HITS`
/// or `GLOBAL_LBVH_BUILDS`.
pub fn build_gpu_lbvh(primitives: &[([f32; 4], u32, u32)]) -> Vec<GpuBvhNode> {
    if primitives.is_empty() {
        return vec![GpuBvhNode {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 0.0,
            max_y: 0.0,
            left_or_prim: 0,
            right_or_count: 0,
            layer_mask: 0,
            net_id: 0,
        }];
    }

    let key = hash_lbvh_primitives(primitives);
    if let Some(cached) = THREAD_LBVH_CACHE.with(|cell| {
        if let Some((cached_key, ref cached_nodes)) = *cell.borrow() {
            if cached_key == key && cached_nodes.len() == primitives.len() * 2 - 1 {
                GLOBAL_LBVH_HITS.fetch_add(1, Ordering::Relaxed);
                return Some(cached_nodes.clone());
            }
        }
        None
    }) {
        return cached;
    }

    GLOBAL_LBVH_BUILDS.fetch_add(1, Ordering::Relaxed);

    let (c_min_x, c_min_y, c_max_x, c_max_y) = if primitives.len() >= 256 {
        primitives
            .par_iter()
            .fold(
                || (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
                |(min_x, min_y, max_x, max_y), &(aabb, _, _)| {
                    let cx = 0.5 * (aabb[0] + aabb[2]);
                    let cy = 0.5 * (aabb[1] + aabb[3]);
                    (min_x.min(cx), min_y.min(cy), max_x.max(cx), max_y.max(cy))
                },
            )
            .reduce(
                || (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
                |a, b| (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)),
            )
    } else {
        let mut c_min_x = f32::INFINITY;
        let mut c_min_y = f32::INFINITY;
        let mut c_max_x = f32::NEG_INFINITY;
        let mut c_max_y = f32::NEG_INFINITY;
        for &(aabb, _, _) in primitives {
            let cx = 0.5 * (aabb[0] + aabb[2]);
            let cy = 0.5 * (aabb[1] + aabb[3]);
            c_min_x = c_min_x.min(cx);
            c_min_y = c_min_y.min(cy);
            c_max_x = c_max_x.max(cx);
            c_max_y = c_max_y.max(cy);
        }
        (c_min_x, c_min_y, c_max_x, c_max_y)
    };

    let span_x = (c_max_x - c_min_x).max(1e-5);
    let span_y = (c_max_y - c_min_y).max(1e-5);

    let mut morton_pairs: Vec<(u32, usize)> = if primitives.len() >= 256 {
        primitives
            .par_iter()
            .enumerate()
            .map(|(idx, &(aabb, _, _))| {
                let cx = 0.5 * (aabb[0] + aabb[2]);
                let cy = 0.5 * (aabb[1] + aabb[3]);
                let qx = (((cx - c_min_x) / span_x).clamp(0.0, 1.0) * 65535.0) as u16;
                let qy = (((cy - c_min_y) / span_y).clamp(0.0, 1.0) * 65535.0) as u16;
                (morton2d_encode(qx, qy), idx)
            })
            .collect()
    } else {
        primitives
            .iter()
            .enumerate()
            .map(|(idx, &(aabb, _, _))| {
                let cx = 0.5 * (aabb[0] + aabb[2]);
                let cy = 0.5 * (aabb[1] + aabb[3]);
                let qx = (((cx - c_min_x) / span_x).clamp(0.0, 1.0) * 65535.0) as u16;
                let qy = (((cy - c_min_y) / span_y).clamp(0.0, 1.0) * 65535.0) as u16;
                (morton2d_encode(qx, qy), idx)
            })
            .collect()
    };

    if primitives.len() >= 256 {
        morton_pairs.par_sort_unstable();
    } else {
        morton_pairs.sort_unstable();
    }

    let total_nodes = primitives.len() * 2 - 1;
    let mut nodes = vec![
        GpuBvhNode {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 0.0,
            max_y: 0.0,
            left_or_prim: 0,
            right_or_count: 0,
            layer_mask: 0,
            net_id: 0,
        };
        total_nodes
    ];

    /// Recursively populates `out_nodes` in pre-order layout from a Morton-sorted slice of `(morton_code, prim_idx)`.
    fn build_slice(
        slice: &[(u32, usize)],
        primitives: &[([f32; 4], u32, u32)],
        base_idx: usize,
        out_nodes: &mut [GpuBvhNode],
    ) {
        if slice.len() == 1 {
            let orig_idx = slice[0].1;
            let (aabb, layer_mask, net_id) = primitives[orig_idx];
            out_nodes[0] = GpuBvhNode {
                min_x: aabb[0].min(aabb[2]),
                min_y: aabb[1].min(aabb[3]),
                max_x: aabb[0].max(aabb[2]),
                max_y: aabb[1].max(aabb[3]),
                left_or_prim: orig_idx as i32,
                right_or_count: 1,
                layer_mask,
                net_id,
            };
            return;
        }

        let mid = slice.len() / 2;
        let left_nodes_len = 2 * mid - 1;
        let left_idx = base_idx + 1;
        let right_idx = base_idx + 1 + left_nodes_len;

        let (_, rest) = out_nodes.split_at_mut(1);
        let (left_out, right_out) = rest.split_at_mut(left_nodes_len);

        if slice.len() >= 256 {
            rayon::join(
                || build_slice(&slice[..mid], primitives, left_idx, left_out),
                || build_slice(&slice[mid..], primitives, right_idx, right_out),
            );
        } else {
            build_slice(&slice[..mid], primitives, left_idx, left_out);
            build_slice(&slice[mid..], primitives, right_idx, right_out);
        }

        let left = left_out[0];
        let right = right_out[0];
        let same_net = if left.net_id != 0 && left.net_id == right.net_id {
            left.net_id
        } else {
            0
        };

        out_nodes[0] = GpuBvhNode {
            min_x: left.min_x.min(right.min_x),
            min_y: left.min_y.min(right.min_y),
            max_x: left.max_x.max(right.max_x),
            max_y: left.max_y.max(right.max_y),
            left_or_prim: left_idx as i32,
            right_or_count: -(right_idx as i32),
            layer_mask: left.layer_mask | right.layer_mask,
            net_id: same_net,
        };
    }

    build_slice(&morton_pairs, primitives, 0, &mut nodes);
    THREAD_LBVH_CACHE.with(|cell| {
        *cell.borrow_mut() = Some((key, nodes.clone()));
    });
    nodes
}

/// Convenience helper to build a 2D Morton LBVH directly over 2D line segments `[x1, y1, x2, y2]`.
///
/// Converts each segment into its bounding box `[min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]`
/// with `layer_mask = 1` and `net_id = 0`, then delegates to [`build_gpu_lbvh`].
///
/// # Arguments
/// * `segs` - Slice of 2D line segments `[x1, y1, x2, y2]` in mm.
///
/// # Returns
/// A flat binary LBVH (`Vec<GpuBvhNode>`) indexing `segs`.
///
/// # Side Effects
/// Updates `THREAD_LBVH_CACHE` and global LBVH statistics via [`build_gpu_lbvh`].
pub fn build_gpu_lbvh_from_segments(segs: &[[f32; 4]]) -> Vec<GpuBvhNode> {
    let prims: Vec<([f32; 4], u32, u32)> = segs
        .iter()
        .map(|&s| ([s[0].min(s[2]), s[1].min(s[3]), s[0].max(s[2]), s[1].max(s[3])], 1, 0))
        .collect();
    build_gpu_lbvh(&prims)
}

/// CPU slab-clipping intersection test between a 2D line segment `s` and an axis-aligned bounding box.
///
/// Mirrors the WGSL `ray_aabb_intersects` helper used in the LBVH traversal shaders.
///
/// # Arguments
/// * `s` - Line segment endpoints `[x1, y1, x2, y2]` in mm.
/// * `b_min_x` - Minimum X coordinate of the AABB (mm).
/// * `b_min_y` - Minimum Y coordinate of the AABB (mm).
/// * `b_max_x` - Maximum X coordinate of the AABB (mm).
/// * `b_max_y` - Maximum Y coordinate of the AABB (mm).
///
/// # Returns
/// `true` if the closed segment `[x1, y1] -> [x2, y2]` intersects the closed box `[b_min_x, b_min_y, b_max_x, b_max_y]`.
///
/// # Side Effects
/// None.
#[inline]
pub fn ray_aabb_intersects_cpu(s: [f32; 4], b_min_x: f32, b_min_y: f32, b_max_x: f32, b_max_y: f32) -> bool {
    let s_min_x = s[0].min(s[2]);
    let s_max_x = s[0].max(s[2]);
    let s_min_y = s[1].min(s[3]);
    let s_max_y = s[1].max(s[3]);
    if s_max_x < b_min_x || s_min_x > b_max_x || s_max_y < b_min_y || s_min_y > b_max_y {
        return false;
    }
    let dx = s[2] - s[0];
    let dy = s[3] - s[1];
    let mut tmin = 0.0f32;
    let mut tmax = 1.0f32;
    if dx.abs() > 1e-6 {
        let inv_dx = 1.0 / dx;
        let tx1 = (b_min_x - s[0]) * inv_dx;
        let tx2 = (b_max_x - s[0]) * inv_dx;
        tmin = tmin.max(tx1.min(tx2));
        tmax = tmax.min(tx1.max(tx2));
    }
    if dy.abs() > 1e-6 {
        let inv_dy = 1.0 / dy;
        let ty1 = (b_min_y - s[1]) * inv_dy;
        let ty2 = (b_max_y - s[1]) * inv_dy;
        tmin = tmin.max(ty1.min(ty2));
        tmax = tmax.min(ty1.max(ty2));
    }
    tmax >= tmin
}

/// CPU proper intersection test between two 2D line segments `s1` and `s2`, excluding shared endpoints.
///
/// Returns `false` if any pair of endpoints lies within `1e-3` mm so adjacent ratsnest edges
/// sharing a pin do not register as crossings.
///
/// # Arguments
/// * `s1` - First line segment `[x1, y1, x2, y2]` in mm.
/// * `s2` - Second line segment `[x1, y1, x2, y2]` in mm.
///
/// # Returns
/// `true` if `s1` and `s2` strictly cross in their interiors; `false` otherwise.
///
/// # Side Effects
/// None.
#[inline]
pub fn segments_cross_cpu(s1: [f32; 4], s2: [f32; 4]) -> bool {
    let eps = 1e-3f32;
    if ((s1[0] - s2[0]).abs() < eps && (s1[1] - s2[1]).abs() < eps)
        || ((s1[0] - s2[2]).abs() < eps && (s1[1] - s2[3]).abs() < eps)
        || ((s1[2] - s2[0]).abs() < eps && (s1[3] - s2[1]).abs() < eps)
        || ((s1[2] - s2[2]).abs() < eps && (s1[3] - s2[3]).abs() < eps)
    {
        return false;
    }
    let ccw = |ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32| -> f32 {
        (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    };
    let d1 = ccw(s1[0], s1[1], s1[2], s1[3], s2[0], s2[1]);
    let d2 = ccw(s1[0], s1[1], s1[2], s1[3], s2[2], s2[3]);
    let d3 = ccw(s2[0], s2[1], s2[2], s2[3], s1[0], s1[1]);
    let d4 = ccw(s2[0], s2[1], s2[2], s2[3], s1[2], s1[3]);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// CPU brute-force reference implementation for batch line-of-sight DRC ray queries.
///
/// Tests every query in `queries` against every obstacle in `obstacles`, counting intersections
/// where the layer masks overlap, the nets differ (or `q.net_id == 0`), and the query segment
/// intersects the obstacle AABB expanded by `q.clearance`.
///
/// # Arguments
/// * `obstacles` - Slice of `(aabb, layer_mask, net_id)` obstacle primitives.
/// * `queries` - Slice of [`GpuDrcRayQuery`] ray descriptors.
///
/// # Returns
/// A `Vec<u32>` of length `queries.len()` containing the violation count for each query.
///
/// # Side Effects
/// None.
pub fn evaluate_drc_brute_force_cpu(
    obstacles: &[([f32; 4], u32, u32)],
    queries: &[GpuDrcRayQuery],
) -> Vec<u32> {
    queries
        .iter()
        .map(|q| {
            let ray = [q.x1, q.y1, q.x2, q.y2];
            let r = q.clearance.max(0.0);
            let mut count = 0u32;
            for &(aabb, layer_mask, net_id) in obstacles {
                if (layer_mask & q.layer_mask) == 0 {
                    continue;
                }
                if q.net_id != 0 && net_id == q.net_id {
                    continue;
                }
                let min_x = aabb[0].min(aabb[2]) - r;
                let min_y = aabb[1].min(aabb[3]) - r;
                let max_x = aabb[0].max(aabb[2]) + r;
                let max_y = aabb[1].max(aabb[3]) + r;
                if ray_aabb_intersects_cpu(ray, min_x, min_y, max_x, max_y) {
                    count += 1;
                }
            }
            count
        })
        .collect()
}

/// CPU $O(N^2)$ brute-force reference implementation for counting pairwise segment crossings.
///
/// For each segment `i`, counts how many subsequent segments `j > i` strictly cross `segs[i]`.
///
/// # Arguments
/// * `segs` - Slice of 2D line segments `[x1, y1, x2, y2]` in mm.
///
/// # Returns
/// A `Vec<u32>` of length `segs.len()` where entry `i` is the number of crossings with `j > i`.
///
/// # Side Effects
/// None.
pub fn evaluate_segment_crossings_brute_force_cpu(segs: &[[f32; 4]]) -> Vec<u32> {
    let n = segs.len();
    (0..n)
        .map(|i| {
            let s1 = segs[i];
            let mut c = 0u32;
            for j in (i + 1)..n {
                if segments_cross_cpu(s1, segs[j]) {
                    c += 1;
                }
            }
            c
        })
        .collect()
}

/// Traverses a CPU/GPU Morton LBVH (`bvh_nodes`) along a 2D query segment `ray` and invokes `on_leaf`
/// for every candidate leaf primitive index whose AABB intersects `ray`.
///
/// Uses a fixed 32-entry stack on the CPU stack frame without heap allocation. Subtrees whose
/// `net_id` matches a non-zero `query_net_id` are pruned automatically.
///
/// # Arguments
/// * `ray` - Query line segment `[x1, y1, x2, y2]` in mm.
/// * `query_net_id` - Net ID to ignore (`0` = test all nets).
/// * `bvh_nodes` - Flat LBVH node slice produced by [`build_gpu_lbvh`].
/// * `on_leaf` - Callback invoked with the primitive index `usize` for each intersecting leaf node.
///
/// # Side Effects
/// Invokes `on_leaf` for each leaf AABB hit.
#[inline]
pub fn query_lbvh_crossing_segments<F: FnMut(usize)>(
    ray: [f32; 4],
    query_net_id: u32,
    bvh_nodes: &[GpuBvhNode],
    mut on_leaf: F,
) {
    if bvh_nodes.is_empty() {
        return;
    }
    const PAD: f32 = 1e-3;
    let mut stack = [0i32; 32];
    let mut stack_ptr = 1usize;

    while stack_ptr > 0 {
        stack_ptr -= 1;
        let node_idx = stack[stack_ptr] as usize;
        let node = &bvh_nodes[node_idx];
        if node.right_or_count == 0 {
            continue;
        }
        if query_net_id != 0 && node.net_id == query_net_id {
            continue;
        }
        if !ray_aabb_intersects_cpu(
            ray,
            node.min_x - PAD,
            node.min_y - PAD,
            node.max_x + PAD,
            node.max_y + PAD,
        ) {
            continue;
        }
        if node.right_or_count > 0 {
            on_leaf(node.left_or_prim as usize);
        } else {
            let left_idx = node.left_or_prim;
            let right_idx = -node.right_or_count;
            if stack_ptr + 2 <= 32 {
                stack[stack_ptr] = right_idx;
                stack[stack_ptr + 1] = left_idx;
                stack_ptr += 2;
            }
        }
    }
}

/// CPU brute-force reference implementation for counting ratsnest crossings across corridor cut lines.
///
/// # Arguments
/// * `segs` - Slice of ratsnest segments `[x1, y1, x2, y2]` in mm.
/// * `cuts` - Slice of corridor cut segments `[x1, y1, x2, y2]` in mm.
///
/// # Returns
/// A `Vec<u32>` of length `cuts.len()` containing the crossing count for each corridor cut.
///
/// # Side Effects
/// None.
pub fn evaluate_corridor_cuts_brute_force_cpu(segs: &[[f32; 4]], cuts: &[[f32; 4]]) -> Vec<u32> {
    cuts.iter()
        .map(|&cut| {
            let mut c = 0u32;
            for &s in segs {
                if segments_cross_cpu(cut, s) {
                    c += 1;
                }
            }
            c
        })
        .collect()
}

/// CPU LBVH-accelerated evaluation of ratsnest crossings across topological corridor cut lines.
///
/// Uses [`query_lbvh_crossing_segments`] to prune non-intersecting subtrees and [`segments_cross_cpu`]
/// to verify exact segment intersections at the leaves.
///
/// # Arguments
/// * `segs` - Slice of ratsnest segments `[x1, y1, x2, y2]` in mm.
/// * `bvh_nodes` - Pre-built Morton LBVH over `segs` (from [`build_gpu_lbvh_from_segments`]).
/// * `cuts` - Slice of corridor cut segments `[x1, y1, x2, y2]` in mm.
///
/// # Returns
/// A `Vec<u32>` of length `cuts.len()` containing the crossing count for each corridor cut.
///
/// # Side Effects
/// None.
pub fn evaluate_corridor_cuts_lbvh_cpu(
    segs: &[[f32; 4]],
    bvh_nodes: &[GpuBvhNode],
    cuts: &[[f32; 4]],
) -> Vec<u32> {
    cuts.iter()
        .map(|&cut| {
            let mut c = 0u32;
            query_lbvh_crossing_segments(cut, 0, bvh_nodes, |j| {
                if segments_cross_cpu(cut, segs[j]) {
                    c += 1;
                }
            });
            c
        })
        .collect()
}

/// Uniform buffer parameters for the 2D/3D Euclidean Distance Field shader (`DISTANCE_FIELD_WGSL`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GridParams {
    /// Grid width in cells.
    width: u32,
    /// Grid height in cells.
    height: u32,
    /// Maximum search radius in grid cells (`[-radius, +radius]`).
    radius: i32,
    /// Number of routing layers in the 3D obstacle grid.
    num_layers: u32,
}

/// Uniform buffer parameters for 3D routing cell traversal cost evaluation (`COMPUTE_3D_ROUTING_COST_WGSL`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRouteCostParams {
    /// Routing grid width in cells.
    pub width: u32,
    /// Routing grid height in cells.
    pub height: u32,
    /// Number of copper routing layers.
    pub num_layers: u32,
    /// Active electrical net ID being routed (`0` = treat all non-zero cells as foreign obstacles).
    pub net_id: u32,
    /// Required circular clearance radius in grid cells around foreign-net obstacles.
    pub clearance_cells: i32,
    /// PathFinder present congestion multiplier applied to `usage_count`.
    pub present_congestion_factor: f32,
    /// Weight applied to inverse Euclidean distance field penalty (`obstacle_clearance_weight / dist_field`).
    pub obstacle_clearance_weight: f32,
    /// Bitmask of dedicated power/ground plane layers (`1 << layer`) blocked to foreign nets.
    pub plane_layer_mask: u32,
}

/// Uniform buffer parameters for 3D Lee/Moore wavefront relaxation (`WAVEFRONT_3D_RELAX_WGSL`).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WavefrontUniformParams {
    /// Routing grid width in cells.
    width: u32,
    /// Routing grid height in cells.
    height: u32,
    /// Number of copper routing layers.
    num_layers: u32,
    /// Additive cost penalty incurred when transitioning between layers through a via.
    via_cost: f32,
}

/// Uniform buffer parameters for GPU 3D wavefront path backtracking (`WAVEFRONT_3D_BACKTRACK_WGSL`).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WavefrontBacktrackUniformParams {
    /// Routing grid width in cells.
    width: u32,
    /// Routing grid height in cells.
    height: u32,
    /// Number of copper routing layers.
    num_layers: u32,
    /// Additive cost penalty for inter-layer via transitions.
    via_cost: f32,
    /// Source cell X coordinate.
    src_x: u32,
    /// Source cell Y coordinate.
    src_y: u32,
    /// Source cell layer index.
    src_layer: u32,
    /// Destination cell X coordinate.
    dst_x: u32,
    /// Destination cell Y coordinate.
    dst_y: u32,
    /// Destination cell layer index.
    dst_layer: u32,
    /// Maximum number of `(x, y, layer)` steps that `path_out` can hold.
    max_path_len: u32,
    /// Explicit 4-byte alignment padding for 16-byte WGSL uniform alignment.
    pad: u32,
}

/// Uniform buffer counts for continuous-geometry batch DRC evaluation (`BATCH_DRC_EVAL_WGSL`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrcBatchParams {
    /// Number of board pads in `pads`.
    pub num_pads: u32,
    /// Number of routed track segments in `tracks`.
    pub num_tracks: u32,
    /// Number of existing vias in `vias`.
    pub num_vias: u32,
    /// Number of candidate vias or track segments in `candidates`.
    pub num_candidates: u32,
}

/// GPU representation of a PCB pad for continuous-geometry batch DRC evaluation (`32` bytes, `#[repr(C)]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDrcPad {
    /// World X coordinate of the pad center (mm).
    pub x: f32,
    /// World Y coordinate of the pad center (mm).
    pub y: f32,
    /// Half-width of the pad in local X (mm).
    pub hw: f32,
    /// Half-height of the pad in local Y (mm).
    pub hh: f32,
    /// Pad rotation angle in radians.
    pub rot_rad: f32,
    /// Minimum copper clearance required around this pad (mm).
    pub clearance: f32,
    /// Lower 16 bits: copper layer bitmask (`1 << layer`, or `0xFFFF` for through-hole);
    /// upper 16 bits (`layer_mask >> 16`): pad shape (`0` = rect/roundrect, `1` = circle, `2` = oval/oblong).
    pub layer_mask: u32,
    /// Electrical net ID (`0` = unconnected).
    pub net_id: u32,
}

/// GPU representation of a routed PCB track segment for batch DRC evaluation (`32` bytes, `#[repr(C)]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDrcTrack {
    /// Track segment start X coordinate (mm).
    pub x1: f32,
    /// Track segment start Y coordinate (mm).
    pub y1: f32,
    /// Track segment end X coordinate (mm).
    pub x2: f32,
    /// Track segment end Y coordinate (mm).
    pub y2: f32,
    /// Half of the track copper width (`width * 0.5`, in mm).
    pub half_width: f32,
    /// Netclass copper clearance of the track's net (mm).
    pub clearance: f32,
    /// Copper layer index on which the track is routed.
    pub layer: u32,
    /// Electrical net ID of the track.
    pub net_id: u32,
}

/// GPU representation of an existing PCB via for batch DRC evaluation (`32` bytes, `#[repr(C)]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDrcVia {
    /// Via center X coordinate (mm).
    pub x: f32,
    /// Via center Y coordinate (mm).
    pub y: f32,
    /// Outer copper annular ring radius (`size * 0.5`, in mm).
    pub radius: f32,
    /// Netclass copper clearance of the via's net (mm).
    pub clearance: f32,
    /// Electrical net ID of the via.
    pub net_id: u32,
    /// Alignment padding word 0.
    pub _pad0: u32,
    /// Alignment padding word 1.
    pub _pad1: u32,
    /// Alignment padding word 2.
    pub _pad2: u32,
}

/// Candidate via or track segment submitted to [`VulkanEngine::evaluate_drc_candidates`] (`48` bytes, `#[repr(C)]`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDrcCandidate {
    /// Candidate via X coordinate (when `kind == 0`) or track start X coordinate (when `kind == 1`), in mm.
    pub x1: f32,
    /// Candidate via Y coordinate (when `kind == 0`) or track start Y coordinate (when `kind == 1`), in mm.
    pub y1: f32,
    /// Optional ignored via X coordinate (when `kind == 0` and `_pad0 != 0`) or track end X coordinate (when `kind == 1`), in mm.
    pub x2: f32,
    /// Optional ignored via Y coordinate (when `kind == 0` and `_pad0 != 0`) or track end Y coordinate (when `kind == 1`), in mm.
    pub y2: f32,
    /// Via outer radius (when `kind == 0`) or track half-width (when `kind == 1`), in mm.
    pub radius_or_half_width: f32,
    /// Required copper clearance for the candidate's netclass (mm).
    pub clearance: f32,
    /// Copper layer index (used when `kind == 1`).
    pub layer: u32,
    /// Electrical net ID of the candidate.
    pub net_id: u32,
    /// Candidate geometry kind: `0` for a via at `(x1, y1)`, `1` for a track segment `(x1, y1) -> (x2, y2)`.
    pub kind: u32,
    /// When non-zero (`1`), allows candidate vias (`kind == 0`) to overlap same-net SMD pads.
    pub ignore_same_net_pads: u32,
    /// Flag indicating whether `(x2, y2)` holds an existing via position to ignore during via nudge checks.
    pub _pad0: u32,
    /// Explicit alignment padding word.
    pub _pad1: u32,
}

/// Uniform buffer parameters for batch PCB placement evaluation (`BATCH_PLACEMENT_EVAL_WGSL`).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlacementBatchParams {
    /// Number of candidate board placements in the batch.
    pub num_candidates: u32,
    /// Number of footprints per candidate placement.
    pub num_footprints: u32,
    /// Number of ratsnest airwire segments per candidate placement.
    pub num_segments: u32,
    /// Explicit 4-byte alignment padding.
    pub _pad: u32,
    /// Board outline minimum X coordinate (mm).
    pub board_min_x: f32,
    /// Board outline minimum Y coordinate (mm).
    pub board_min_y: f32,
    /// Board outline maximum X coordinate (mm).
    pub board_max_x: f32,
    /// Board outline maximum Y coordinate (mm).
    pub board_max_y: f32,
    /// Multiplier for total Euclidean ratsnest wirelength.
    pub wirelength_weight: f32,
    /// Penalty added per pairwise ratsnest airwire crossing.
    pub crossing_weight: f32,
    /// Penalty multiplier for pairwise footprint courtyard AABB overlaps.
    pub overlap_penalty: f32,
    /// Penalty multiplier for footprints protruding outside the board outline.
    pub boundary_penalty: f32,
}

/// Uniform buffer parameters for the GPU Parallel Tempering Simulated Annealing auto-placer (`GPU_SA_PLACER_WGSL`).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSaParams {
    /// Number of parallel MCMC simulated annealing chains.
    pub num_chains: u32,
    /// Number of footprints per chain.
    pub num_footprints: u32,
    /// Number of ratsnest / affinity / corridor / flow edges in `edges`.
    pub num_edges: u32,
    /// Number of Metropolis-Hastings perturbation steps executed per chain per compute pass.
    pub steps_per_pass: u32,
    /// Base PCG pseudo-random number generator seed.
    pub seed: u32,
    /// Explicit 4-byte alignment padding.
    pub _pad0: u32,
    /// Penalty added when a ratsnest airwire crosses a topological corridor cut segment.
    pub corridor_congestion_weight: f32,
    /// Weight applied to schematic signal-flow alignment edges (`tag > 3.5`).
    pub flow_alignment_weight: f32,
    /// Board outline minimum X coordinate (mm).
    pub board_min_x: f32,
    /// Board outline minimum Y coordinate (mm).
    pub board_min_y: f32,
    /// Board outline maximum X coordinate (mm).
    pub board_max_x: f32,
    /// Board outline maximum Y coordinate (mm).
    pub board_max_y: f32,
    /// Placement grid snapping increment in mm (`<= 0.0` disables snapping).
    pub grid_step: f32,
    /// Weight applied to total Euclidean ratsnest wirelength.
    pub wirelength_weight: f32,
    /// Penalty added per pairwise ratsnest segment crossing.
    pub crossing_weight: f32,
    /// Penalty multiplier for courtyard polygon overlaps.
    pub overlap_penalty: f32,
    /// Penalty multiplier for board boundary excursions.
    pub boundary_penalty: f32,
    /// Weight applied to passive / decoupling capacitor affinity edges (`0.5 < tag < 2.5`).
    pub passive_affinity_weight: f32,
    /// Threshold distance (mm) beyond which passive affinity edges incur quadratic distance penalties.
    pub passive_max_dist: f32,
    /// Base annealing temperature for the coldest chain in the geometric temperature ladder.
    pub base_temp: f32,
}

/// Flattened convex courtyard polygon buffers uploaded to the GPU for exact SAT collision checks
/// in [`VulkanEngine::run_gpu_sa_placer_chains`].
#[derive(Clone, Debug, Default)]
pub struct GpuCourtyardBuffers {
    /// Per-footprint polygon ranges: `[poly_start, poly_count, is_smd_only, orig_layer]`.
    pub fp_poly_ranges: Vec<[u32; 4]>,
    /// Courtyard polygon headers: `[vertex_start, vertex_count, fp_idx, 0]`.
    pub courtyard_polys: Vec<[u32; 4]>,
    /// Local vertex coordinates `[x, y]` in mm (pre-expanded by clearance/spacing).
    pub courtyard_vertices: Vec<[f32; 2]>,
}

impl GpuCourtyardBuffers {
    /// Synthesizes default rectangular 4-vertex courtyard polygons from footprint half-extents `fp_descs`.
    ///
    /// Used as a fallback by [`VulkanEngine::run_gpu_sa_placer_chains`] when explicit convex
    /// courtyard polygons are not provided by the caller.
    ///
    /// # Arguments
    /// * `fp_descs` - Slice of footprint descriptors `[half_w, half_h, is_locked, packed_offset]`.
    ///
    /// # Returns
    /// A populated [`GpuCourtyardBuffers`] containing one axis-aligned 4-vertex box polygon per footprint.
    ///
    /// # Side Effects
    /// None.
    pub fn from_fp_descs(fp_descs: &[[f32; 4]]) -> Self {
        let mut fp_poly_ranges = Vec::with_capacity(fp_descs.len());
        let mut courtyard_polys = Vec::with_capacity(fp_descs.len());
        let mut courtyard_vertices = Vec::with_capacity(fp_descs.len() * 4);

        for (i, d) in fp_descs.iter().enumerate() {
            let poly_start = courtyard_polys.len() as u32;
            let v_start = courtyard_vertices.len() as u32;
            let hw = d[0].max(0.1);
            let hh = d[1].max(0.1);
            courtyard_vertices.push([-hw, -hh]);
            courtyard_vertices.push([hw, -hh]);
            courtyard_vertices.push([hw, hh]);
            courtyard_vertices.push([-hw, hh]);
            courtyard_polys.push([v_start, 4, i as u32, 0]);
            fp_poly_ranges.push([poly_start, 1, 0, 0]);
        }

        Self {
            fp_poly_ranges,
            courtyard_polys,
            courtyard_vertices,
        }
    }
}

/// Active GPU compute backend selected by [`VulkanEngine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendKind {
    /// Cross-vendor `wgpu` Vulkan WGSL compute backend (`wgpu::Backends::VULKAN`).
    Vulkan,
    /// Bare-metal NVIDIA CUDA Driver API (`libcuda.so`) PTX kernel backend with Vulkan fallback ready.
    CudaPtx,
}

/// Reusable pool of `wgpu::Buffer` allocations for 3D wavefront routing and backtracking.
///
/// Cached inside [`VulkanEngine::wavefront_buffers`] via `VulkanEngine::with_wavefront_buffers`
/// so repeated 3D wavefront solves on the same grid dimensions reuse device memory without
/// per-route buffer creation/destruction overhead.
pub struct CachedWavefrontBuffers {
    /// Total number of 3D grid cells (`width * height * num_layers`) for which the buffers are sized.
    pub total_cells: usize,
    /// Maximum number of `(x, y, layer)` steps allocated in `path_out_buf` and `staging_path_buf`.
    pub max_path_len: u32,
    /// Device storage buffer holding 3D obstacle net IDs (`[total_cells]`).
    pub obst_buf: wgpu::Buffer,
    /// Device storage buffer holding 3D Euclidean distance field values (`[total_cells]`).
    pub df_buf: wgpu::Buffer,
    /// Device storage buffer holding 3D PathFinder historical congestion costs (`[total_cells]`).
    pub hist_buf: wgpu::Buffer,
    /// Device storage buffer holding 3D present usage counts (`[total_cells]`).
    pub usage_buf: wgpu::Buffer,
    /// Device storage buffer holding per-cell traversal costs (`[total_cells]`).
    pub cost_buf: wgpu::Buffer,
    /// Ping-pong wavefront distance buffer A (`[total_cells]`).
    pub dist_a_buf: wgpu::Buffer,
    /// Ping-pong wavefront distance buffer B (`[total_cells]`).
    pub dist_b_buf: wgpu::Buffer,
    /// Device storage buffer receiving backtracked `(x, y, layer)` path triples (`[max_path_len * 3]`).
    pub path_out_buf: wgpu::Buffer,
    /// Single-`u32` device storage buffer receiving the backtracked path length.
    pub path_len_buf: wgpu::Buffer,
    /// Host-mappable staging buffer for reading back full 3D wavefront distance fields (`[total_cells]`).
    pub staging_buf: wgpu::Buffer,
    /// Host-mappable staging buffer for reading back `path_len_buf` (`4` bytes).
    pub staging_len_buf: wgpu::Buffer,
    /// Host-mappable staging buffer for reading back `path_out_buf` (`[max_path_len * 3]`).
    pub staging_path_buf: wgpu::Buffer,
    /// Number of times this buffer pool was (re)allocated due to grid dimension changes.
    pub alloc_count: usize,
    /// Number of times existing buffers in this pool were reused without reallocation.
    pub reuse_count: usize,
}

/// Unified GPU compute engine managing both NVIDIA CUDA PTX fast-path kernels and `wgpu` Vulkan WGSL pipelines.
///
/// `VulkanEngine` compiles and caches all WGSL compute pipelines at initialization time, probes for
/// an optional bare-metal [`crate::cuda::CudaEngine`], and exposes thread-safe methods for:
/// - 2D and multi-layer Euclidean Distance Fields (`compute_distance_field`, `compute_distance_field_layers`),
/// - Batch placement evaluation and Parallel Tempering SA auto-placement (`evaluate_placement_batch`, `run_gpu_sa_placer_chains`),
/// - 3D routing cell cost evaluation and Lee/Moore wavefront routing (`compute_3d_routing_costs`, `compute_3d_wavefront`, `solve_3d_wavefront_path_with_clearance`),
/// - 2D Morton LBVH construction and traversal for ratsnest crossings, corridor cuts, and line-of-sight DRC (`build_cached_lbvh`, `evaluate_segment_crossings_rt_bvh`, `evaluate_corridor_cuts_rt_bvh`, `evaluate_drc_rt_bvh`),
/// - Continuous-geometry batch DRC candidate checking (`evaluate_drc_candidates`).
pub struct VulkanEngine {
    /// Human-readable GPU adapter name (or `" + "`-joined names when multiple GPUs are active).
    pub device_name: String,
    /// Active backend description string (e.g., `"Vulkan"` or `"CUDA PTX (NVIDIA bare-metal) [Vulkan fallback ready]"`).
    pub backend_name: String,
    /// GPU driver version and vendor description string.
    pub driver_info: String,
    /// Selected primary compute backend ([`GpuBackendKind::CudaPtx`] or [`GpuBackendKind::Vulkan`]).
    pub backend_kind: GpuBackendKind,
    /// Whether hardware ray query / acceleration structure support was detected on this adapter.
    pub vulkan_rt_ray_query_supported: bool,
    /// Optional bare-metal NVIDIA CUDA Driver API engine used when `backend_kind == GpuBackendKind::CudaPtx`.
    pub cuda_engine: Option<Arc<crate::cuda::CudaEngine>>,
    /// Underlying `wgpu` logical device handle.
    device: wgpu::Device,
    /// Underlying `wgpu` command submission queue.
    queue: wgpu::Queue,
    /// Compute pipeline for horizontal distance-field pass (`DISTANCE_FIELD_WGSL::pass_h`).
    pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for vertical distance-field pass (`DISTANCE_FIELD_WGSL::pass_v`).
    df_v_pipeline: wgpu::ComputePipeline,
    /// Bind group layout shared by `pipeline` (`pass_h`) and `df_v_pipeline` (`pass_v`).
    bind_group_layout: wgpu::BindGroupLayout,
    /// Compute pipeline for batch placement cost evaluation (`BATCH_PLACEMENT_EVAL_WGSL`).
    placement_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for `placement_pipeline`.
    placement_bgl: wgpu::BindGroupLayout,
    /// Compute pipeline for Parallel Tempering Simulated Annealing (`GPU_SA_PLACER_WGSL`).
    sa_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for `sa_pipeline`.
    sa_bgl: wgpu::BindGroupLayout,
    /// Compute pipeline for 3D routing cell cost evaluation (`COMPUTE_3D_ROUTING_COST_WGSL`).
    route_cost_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for `route_cost_pipeline`.
    route_cost_bgl: wgpu::BindGroupLayout,
    /// Compute pipeline for 3D wavefront relaxation (`WAVEFRONT_3D_RELAX_WGSL`).
    wavefront_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for `wavefront_pipeline`.
    wavefront_bgl: wgpu::BindGroupLayout,
    /// Compute pipeline for 3D wavefront GPU backtracking (`WAVEFRONT_3D_BACKTRACK_WGSL`).
    wavefront_backtrack_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for `wavefront_backtrack_pipeline`.
    wavefront_backtrack_bgl: wgpu::BindGroupLayout,
    /// Compute pipeline for LBVH ratsnest segment crossings (`RT_BVH_SEGMENT_CROSSINGS_WGSL`).
    bvh_crossings_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for `bvh_crossings_pipeline`.
    bvh_crossings_bgl: wgpu::BindGroupLayout,
    /// Compute pipeline for LBVH corridor cut crossings (`RT_BVH_CORRIDOR_CUTS_WGSL`).
    bvh_corridor_cuts_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for `bvh_corridor_cuts_pipeline`.
    bvh_corridor_cuts_bgl: wgpu::BindGroupLayout,
    /// Compute pipeline for LBVH line-of-sight DRC ray queries (`RT_BVH_LINE_OF_SIGHT_DRC_WGSL`).
    bvh_drc_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for `bvh_drc_pipeline`.
    bvh_drc_bgl: wgpu::BindGroupLayout,
    /// Compute pipeline for continuous-geometry batch DRC evaluation (`BATCH_DRC_EVAL_WGSL`).
    drc_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for `drc_pipeline`.
    drc_bgl: wgpu::BindGroupLayout,
    /// Mutex serializing `wgpu` queue submissions and synchronous staging buffer map/poll cycles.
    dispatch_lock: Mutex<()>,
    /// Cached reusable GPU buffers for 3D wavefront relaxation and backtracking.
    pub wavefront_buffers: Mutex<Option<CachedWavefrontBuffers>>,
    /// Engine-level single-entry cache storing `(primitives_hash, bvh_nodes)`.
    pub lbvh_cache: Mutex<Option<(u64, Vec<GpuBvhNode>)>>,
    /// Atomic counter of cache hits in [`Self::lbvh_cache`].
    pub lbvh_cache_hits: AtomicU64,
    /// Cumulative nanoseconds spent executing GPU compute dispatches, sampled by [`SystemTelemetry`].
    pub compute_busy_ns: Arc<AtomicU64>,
    /// Additional GPU engines discovered on multi-GPU systems for parallel batch distribution.
    pub extra_devices: Vec<Arc<VulkanEngine>>,
}

impl std::fmt::Debug for VulkanEngine {
    /// Formats a concise summary of [`VulkanEngine`] showing device, backend, and ray-query support.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanEngine")
            .field("device_name", &self.device_name)
            .field("backend_name", &self.backend_name)
            .field("backend_kind", &self.backend_kind)
            .field("vulkan_rt_ray_query_supported", &self.vulkan_rt_ray_query_supported)
            .finish()
    }
}

impl VulkanEngine {
    /// Initializes the primary [`VulkanEngine`] instance, preferring the bare-metal NVIDIA CUDA PTX
    /// backend when available and falling back to `wgpu` Vulkan compute pipelines.
    ///
    /// Blocks the current thread on `Self::init_async` via `pollster::block_on`.
    ///
    /// # Returns
    /// `Some(Arc<VulkanEngine>)` if at least one hardware GPU adapter (or software adapter when
    /// `VULCANS_ALLOW_SOFTWARE_GPU=1`) was initialized; `None` if no suitable GPU adapter is available.
    ///
    /// # Side Effects
    /// Enumerates Vulkan/`wgpu` adapters, probes `libcuda.so` via [`crate::cuda::CudaEngine::all_devices`]
    /// (unless `VULCANS_FORCE_VULKAN=1`), and compiles all WGSL compute pipelines on the device.
    pub fn new() -> Option<Arc<Self>> {
        pollster::block_on(Self::init_async(false)).map(Arc::new)
    }

    /// Initializes a [`VulkanEngine`] instance forced to use the `wgpu` Vulkan WGSL backend only,
    /// bypassing any NVIDIA CUDA Driver API fast path.
    ///
    /// # Returns
    /// `Some(Arc<VulkanEngine>)` configured with [`GpuBackendKind::Vulkan`], or `None` if no
    /// suitable Vulkan adapter is found.
    ///
    /// # Side Effects
    /// Enumerates Vulkan/`wgpu` adapters and compiles all WGSL compute shaders on the logical device.
    pub fn new_vulkan_only() -> Option<Arc<Self>> {
        pollster::block_on(Self::init_async(true)).map(Arc::new)
    }

    /// Returns `true` if this engine is actively dispatching compute workloads through the
    /// bare-metal NVIDIA CUDA PTX backend ([`GpuBackendKind::CudaPtx`]).
    ///
    /// # Returns
    /// `true` when `self.backend_kind == GpuBackendKind::CudaPtx`, `false` when using Vulkan WGSL.
    pub fn is_cuda_active(&self) -> bool {
        self.backend_kind == GpuBackendKind::CudaPtx
    }

    /// Returns a list containing this primary [`VulkanEngine`] followed by any secondary GPU
    /// engines in [`Self::extra_devices`].
    ///
    /// # Returns
    /// A `Vec<Arc<VulkanEngine>>` of length `1 + self.extra_devices.len()`.
    ///
    /// # Side Effects
    /// Clones `Arc` pointers to the primary and secondary engines.
    pub fn all_engines(self: &Arc<Self>) -> Vec<Arc<Self>> {
        let mut list = Vec::with_capacity(1 + self.extra_devices.len());
        list.push(Arc::clone(self));
        for dev in &self.extra_devices {
            list.push(Arc::clone(dev));
        }
        list
    }

    /// Asynchronously discovers hardware GPU adapters, pairs them with optional CUDA PTX devices,
    /// and constructs the primary [`VulkanEngine`] and any multi-GPU [`Self::extra_devices`].
    ///
    /// Rejects CPU/software rasterizers (`llvmpipe`, `lavapipe`, `SwiftShader`, `WARP`, `paravirtual`)
    /// by default to prevent multi-threaded software driver deadlocks in headless environments unless
    /// `VULCANS_ALLOW_SOFTWARE_GPU=1` is set.
    ///
    /// # Arguments
    /// * `force_vulkan` - When `true`, skips probing [`crate::cuda::CudaEngine`] and forces [`GpuBackendKind::Vulkan`].
    ///
    /// # Returns
    /// `Some(VulkanEngine)` if device and pipeline creation succeed; `None` if no compatible adapter exists.
    ///
    /// # Side Effects
    /// Reads environment variables `VULCANS_ALLOW_SOFTWARE_GPU`, `VULCANS_FORCE_VULKAN`, and
    /// `VULCANS_GPU_BACKEND`, creates `wgpu::Instance` and `wgpu::Device` handles, and initializes
    /// CUDA contexts when available.
    async fn init_async(force_vulkan: bool) -> Option<Self> {
        let allow_software = std::env::var("VULCANS_ALLOW_SOFTWARE_GPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        // Hardware adapter filter: in headless CI environments without physical GPUs,
        // software / CPU rasterizers (llvmpipe, lavapipe, WARP, SwiftShader, Apple Paravirtual)
        // are rejected to avoid multi-thread software emulation deadlocks and driver crashes.
        let is_hw = |a: &wgpu::Adapter| -> bool {
            if allow_software {
                return true;
            }
            let info = a.get_info();
            if matches!(info.device_type, wgpu::DeviceType::Cpu | wgpu::DeviceType::VirtualGpu) {
                return false;
            }
            let name = info.name.to_lowercase();
            !name.contains("llvmpipe")
                && !name.contains("lavapipe")
                && !name.contains("basic render")
                && !name.contains("warp")
                && !name.contains("paravirtual")
                && !name.contains("swiftshader")
        };

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let mut adapters: Vec<wgpu::Adapter> = instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .into_iter()
            .filter(&is_hw)
            .collect();

        if adapters.is_empty() {
            let fallback_inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            adapters = fallback_inst
                .enumerate_adapters(wgpu::Backends::all())
                .into_iter()
                .filter(&is_hw)
                .collect();
        }

        if adapters.is_empty() && allow_software {
            // Only allow CPU software rasterizers if explicitly enabled via VULCANS_ALLOW_SOFTWARE_GPU=1
            let sw_inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            adapters = sw_inst.enumerate_adapters(wgpu::Backends::all());
        }

        if adapters.is_empty() {
            return None;
        }

        let is_vulkan_forced = force_vulkan
            || std::env::var("VULCANS_FORCE_VULKAN")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
            || std::env::var("VULCANS_GPU_BACKEND")
                .map(|v| v.eq_ignore_ascii_case("vulkan"))
                .unwrap_or(false);

        let cuda_devices = if !is_vulkan_forced {
            crate::cuda::CudaEngine::all_devices()
        } else {
            Vec::new()
        };

        let busy_ns = Arc::new(AtomicU64::new(0));

        if !cuda_devices.is_empty() {
            // Build secondary engines first so they can be bundled into primary.extra_devices
            let mut extra_devices = Vec::new();
            for (idx, cuda) in cuda_devices.iter().enumerate().skip(1) {
                let adapter = adapters.iter()
                    .find(|a| a.get_info().name == cuda.device_name
                        || cuda.device_name.contains(&a.get_info().name)
                        || a.get_info().name.contains(&cuda.device_name))
                    .cloned()
                    .unwrap_or_else(|| adapters[idx.min(adapters.len() - 1)].clone());

                if let Some(sec) = Self::from_adapter(adapter, Arc::clone(&busy_ns), Vec::new(), force_vulkan, Some(Arc::clone(cuda))).await {
                    extra_devices.push(Arc::new(sec));
                }
            }

            let primary_cuda = &cuda_devices[0];
            let primary_adapter = adapters.iter()
                .find(|a| a.get_info().name == primary_cuda.device_name
                    || primary_cuda.device_name.contains(&a.get_info().name)
                    || a.get_info().name.contains(&primary_cuda.device_name))
                .cloned()
                .unwrap_or_else(|| adapters[0].clone());

            return Self::from_adapter(primary_adapter, busy_ns, extra_devices, force_vulkan, Some(Arc::clone(primary_cuda))).await;
        }

        // Pure Vulkan / non-CUDA fallback: bundle secondary adapters into extra_devices
        let mut extra_devices = Vec::new();
        for cand in adapters.iter().skip(1) {
            if let Some(eng) = Self::from_adapter(cand.clone(), Arc::clone(&busy_ns), Vec::new(), force_vulkan, None).await {
                extra_devices.push(Arc::new(eng));
            }
        }

        Self::from_adapter(adapters[0].clone(), busy_ns, extra_devices, force_vulkan, None).await
    }

    /// Constructs a [`VulkanEngine`] from a selected `wgpu::Adapter`, compiling all WGSL compute
    /// shaders and optionally binding a bare-metal [`crate::cuda::CudaEngine`].
    ///
    /// # Arguments
    /// * `adapter` - The `wgpu::Adapter` to request a logical device and queue from.
    /// * `compute_busy_ns` - Shared atomic nanosecond accumulator updated by all compute dispatches.
    /// * `extra_devices` - Secondary GPU engines bundled into the primary engine for multi-GPU batching.
    /// * `force_vulkan_arg` - When `true`, disables CUDA PTX probing even if a CUDA device is present.
    /// * `cuda_engine_arg` - Pre-initialized [`crate::cuda::CudaEngine`] to pair with this adapter, if any.
    ///
    /// # Returns
    /// `Some(VulkanEngine)` if `adapter.request_device(...)` succeeds; `None` on device request failure.
    ///
    /// # Side Effects
    /// Requests a `wgpu::Device`, compiles 10 WGSL compute pipelines (`pass_h`, `pass_v`,
    /// `placement_pipeline`, `sa_pipeline`, `route_cost_pipeline`, `wavefront_pipeline`,
    /// `wavefront_backtrack_pipeline`, `bvh_crossings_pipeline`, `bvh_corridor_cuts_pipeline`,
    /// `bvh_drc_pipeline`, `drc_pipeline`), and runs a $16 \times 16$ CUDA distance-field smoke test
    /// if CUDA is enabled.
    async fn from_adapter(
        adapter: wgpu::Adapter,
        compute_busy_ns: Arc<AtomicU64>,
        extra_devices: Vec<Arc<VulkanEngine>>,
        force_vulkan_arg: bool,
        cuda_engine_arg: Option<Arc<crate::cuda::CudaEngine>>,
    ) -> Option<Self> {
        let info = adapter.get_info();
        let adapter_features = adapter.features();
        let hw_name_upper = info.name.to_uppercase();
        let mut vulkan_rt_ray_query_supported = adapter_features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
            || adapter_features.contains(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            || hw_name_upper.contains("RTX")
            || hw_name_upper.contains("RADEON RX 6")
            || hw_name_upper.contains("RADEON RX 7")
            || hw_name_upper.contains("RADEON RX 8")
            || hw_name_upper.contains("RADEON RX 9")
            || hw_name_upper.contains("ARC ");

        let mut device_name = if extra_devices.is_empty() {
            info.name.clone()
        } else {
            let mut names = vec![info.name.clone()];
            for d in &extra_devices {
                names.push(d.device_name.clone());
            }
            names.join(" + ")
        };
        let mut backend_name = format!("{:?}", info.backend);
        let mut driver_info = if info.driver_info.is_empty() {
            info.driver.clone()
        } else {
            format!("{} ({})", info.driver, info.driver_info)
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("VulcansTrials Vulkan Compute Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vulkan Distance Field Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(DISTANCE_FIELD_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DistanceField BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DistanceField PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DistanceField PassH ComputePipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("pass_h"),
            compilation_options: Default::default(),
            cache: None,
        });

        let df_v_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DistanceField PassV ComputePipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("pass_v"),
            compilation_options: Default::default(),
            cache: None,
        });

        let placement_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vulkan Batch Placement Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(BATCH_PLACEMENT_EVAL_WGSL.into()),
        });

        let placement_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PlacementBatch BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let placement_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PlacementBatch PipelineLayout"),
            bind_group_layouts: &[&placement_bgl],
            push_constant_ranges: &[],
        });

        let placement_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PlacementBatch ComputePipeline"),
            layout: Some(&placement_pl_layout),
            module: &placement_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let sa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vulkan SA MCMC Placer Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_SA_PLACER_WGSL.into()),
        });

        let sa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GpuSaPlacer BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sa_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GpuSaPlacer PipelineLayout"),
            bind_group_layouts: &[&sa_bgl],
            push_constant_ranges: &[],
        });

        let sa_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GpuSaPlacer ComputePipeline"),
            layout: Some(&sa_pl_layout),
            module: &sa_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let route_cost_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vulkan 3D Routing Cost Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_3D_ROUTING_COST_WGSL.into()),
        });

        let route_cost_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RouteCost3D BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let route_cost_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RouteCost3D PipelineLayout"),
            bind_group_layouts: &[&route_cost_bgl],
            push_constant_ranges: &[],
        });

        let route_cost_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RouteCost3D ComputePipeline"),
            layout: Some(&route_cost_pl_layout),
            module: &route_cost_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let wavefront_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vulkan 3D Wavefront Relaxation Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(WAVEFRONT_3D_RELAX_WGSL.into()),
        });

        let wavefront_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Wavefront3D BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let wavefront_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Wavefront3D PipelineLayout"),
            bind_group_layouts: &[&wavefront_bgl],
            push_constant_ranges: &[],
        });

        let wavefront_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Wavefront3D ComputePipeline"),
            layout: Some(&wavefront_pl_layout),
            module: &wavefront_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let wavefront_backtrack_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vulkan 3D Wavefront Backtrack Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(WAVEFRONT_3D_BACKTRACK_WGSL.into()),
        });

        let wavefront_backtrack_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WavefrontBacktrack3D BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let wavefront_backtrack_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WavefrontBacktrack3D PipelineLayout"),
            bind_group_layouts: &[&wavefront_backtrack_bgl],
            push_constant_ranges: &[],
        });

        let wavefront_backtrack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("WavefrontBacktrack3D ComputePipeline"),
            layout: Some(&wavefront_backtrack_pl_layout),
            module: &wavefront_backtrack_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bvh_crossings_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vulkan RT BVH Segment Crossings Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(RT_BVH_SEGMENT_CROSSINGS_WGSL.into()),
        });

        let bvh_crossings_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RtBvhCrossings BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bvh_crossings_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RtBvhCrossings PipelineLayout"),
            bind_group_layouts: &[&bvh_crossings_bgl],
            push_constant_ranges: &[],
        });

        let bvh_crossings_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RtBvhCrossings ComputePipeline"),
            layout: Some(&bvh_crossings_pl_layout),
            module: &bvh_crossings_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bvh_corridor_cuts_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vulkan RT BVH Corridor Cuts Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(RT_BVH_CORRIDOR_CUTS_WGSL.into()),
        });

        let bvh_corridor_cuts_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RtBvhCorridorCuts BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bvh_corridor_cuts_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RtBvhCorridorCuts PipelineLayout"),
            bind_group_layouts: &[&bvh_corridor_cuts_bgl],
            push_constant_ranges: &[],
        });

        let bvh_corridor_cuts_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RtBvhCorridorCuts ComputePipeline"),
            layout: Some(&bvh_corridor_cuts_pl_layout),
            module: &bvh_corridor_cuts_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bvh_drc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vulkan RT BVH Line-of-Sight DRC Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(RT_BVH_LINE_OF_SIGHT_DRC_WGSL.into()),
        });

        let bvh_drc_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RtBvhDrc BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bvh_drc_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RtBvhDrc PipelineLayout"),
            bind_group_layouts: &[&bvh_drc_bgl],
            push_constant_ranges: &[],
        });

        let bvh_drc_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RtBvhDrc ComputePipeline"),
            layout: Some(&bvh_drc_pl_layout),
            module: &bvh_drc_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let drc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vulkan Batch DRC Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(BATCH_DRC_EVAL_WGSL.into()),
        });

        let drc_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BatchDrc BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let drc_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BatchDrc PipelineLayout"),
            bind_group_layouts: &[&drc_bgl],
            push_constant_ranges: &[],
        });

        let drc_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BatchDrc ComputePipeline"),
            layout: Some(&drc_pl_layout),
            module: &drc_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut cuda_engine = None;
        let mut backend_kind = GpuBackendKind::Vulkan;

        let force_vulkan = force_vulkan_arg
            || std::env::var("VULCANS_FORCE_VULKAN")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
            || std::env::var("VULCANS_GPU_BACKEND")
                .map(|v| v.eq_ignore_ascii_case("vulkan"))
                .unwrap_or(false);

        if !force_vulkan {
            let cand_cuda = cuda_engine_arg.or_else(crate::cuda::CudaEngine::new);
            if let Some(cuda) = cand_cuda {
                let test_obs = vec![0u32; 16 * 16];
                if cuda.compute_distance_field(&test_obs, 16, 16, 4).is_some() {
                    backend_kind = GpuBackendKind::CudaPtx;
                    backend_name = "CUDA PTX (NVIDIA bare-metal) [Vulkan fallback ready]".to_string();
                    device_name = cuda.device_name.clone();
                    driver_info = format!("{} / Vulkan {}", cuda.driver_info, driver_info);
                    if cuda.optix_available {
                        vulkan_rt_ray_query_supported = true;
                    }
                    cuda_engine = Some(cuda);
                }
            }
        }

        if !extra_devices.is_empty() {
            let mut names = vec![device_name.clone()];
            for d in &extra_devices {
                names.push(d.device_name.clone());
            }
            device_name = format!("{} [{} GPUs Active]", names.join(" + "), 1 + extra_devices.len());
        }

        Some(Self {
            device_name,
            backend_name,
            driver_info,
            backend_kind,
            vulkan_rt_ray_query_supported,
            cuda_engine,
            device,
            queue,
            pipeline,
            df_v_pipeline,
            bind_group_layout,
            placement_pipeline,
            placement_bgl,
            sa_pipeline,
            sa_bgl,
            route_cost_pipeline,
            route_cost_bgl,
            wavefront_pipeline,
            wavefront_bgl,
            wavefront_backtrack_pipeline,
            wavefront_backtrack_bgl,
            bvh_crossings_pipeline,
            bvh_crossings_bgl,
            bvh_corridor_cuts_pipeline,
            bvh_corridor_cuts_bgl,
            bvh_drc_pipeline,
            bvh_drc_bgl,
            drc_pipeline,
            drc_bgl,
            dispatch_lock: Mutex::new(()),
            wavefront_buffers: Mutex::new(None),
            lbvh_cache: Mutex::new(None),
            lbvh_cache_hits: AtomicU64::new(0),
            compute_busy_ns,
            extra_devices,
        })
    }

    /// Builds or retrieves a cached 2D Morton LBVH (`Vec<GpuBvhNode>`) for `primitives` using
    /// the engine-level [`Self::lbvh_cache`].
    ///
    /// Computes an FNV-1a fingerprint via [`hash_lbvh_primitives`] and returns a cloned node vector
    /// immediately on a cache hit, or delegates to [`build_gpu_lbvh`] and stores the result on a miss.
    ///
    /// # Arguments
    /// * `primitives` - Slice of `(aabb, layer_mask, net_id)` tuples where `aabb` is `[min_x, min_y, max_x, max_y]`.
    ///
    /// # Returns
    /// A `Vec<GpuBvhNode>` of length `2 * primitives.len() - 1` (or `1` sentinel node if empty).
    ///
    /// # Side Effects
    /// Locks `self.lbvh_cache`, increments `self.lbvh_cache_hits` on a cache hit, or updates
    /// `self.lbvh_cache` and thread-local LBVH cache on a miss.
    pub fn build_cached_lbvh(&self, primitives: &[([f32; 4], u32, u32)]) -> Vec<GpuBvhNode> {
        let key = hash_lbvh_primitives(primitives);
        if let Ok(guard) = self.lbvh_cache.lock() {
            if let Some((cached_key, ref cached_nodes)) = *guard {
                if cached_key == key && (primitives.is_empty() || cached_nodes.len() == primitives.len() * 2 - 1) {
                    self.lbvh_cache_hits.fetch_add(1, Ordering::Relaxed);
                    return cached_nodes.clone();
                }
            }
        }
        let nodes = build_gpu_lbvh(primitives);
        if let Ok(mut guard) = self.lbvh_cache.lock() {
            *guard = Some((key, nodes.clone()));
        }
        nodes
    }

    /// Computes a single-layer 2D Euclidean Distance Field (EDF) on the GPU for an obstacle grid.
    ///
    /// Dispatches to the CUDA PTX kernel (`CudaEngine::compute_distance_field`) when
    /// `self.backend_kind == GpuBackendKind::CudaPtx`, falling back automatically to the two-pass
    /// Vulkan WGSL pipeline ([`Self::compute_distance_field_vulkan_wgsl`]).
    ///
    /// # Arguments
    /// * `obstacles` - Flat `[width * height]` slice of obstacle flags (`0` = free, non-zero = obstacle).
    /// * `width` - Grid width in cells.
    /// * `height` - Grid height in cells.
    /// * `radius` - Maximum search radius in cells.
    ///
    /// # Returns
    /// `Some(Vec<f32>)` of length `width * height` containing Euclidean distances in cells (clamped to
    /// `radius + 1.0`), or `None` if `obstacles.len() != width * height` or `width * height == 0`.
    ///
    /// # Side Effects
    /// Uploads `obstacles` to the GPU, executes horizontal and vertical distance transform compute
    /// passes, reads back the distance buffer, and increments `self.compute_busy_ns`.
    pub fn compute_distance_field(&self, obstacles: &[u32], width: u32, height: u32, radius: i32) -> Option<Vec<f32>> {
        let total = (width * height) as usize;
        if obstacles.len() != total || total == 0 {
            return None;
        }

        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                if let Some(res) = cuda.compute_distance_field(obstacles, width, height, radius) {
                    self.compute_busy_ns.fetch_add(cuda.compute_busy_ns.load(Ordering::Relaxed), Ordering::Relaxed);
                    return Some(res);
                }
            }
        }

        self.compute_distance_field_vulkan_wgsl(obstacles, width, height, 1, radius)
    }

    /// Executes the separable two-pass 2D/3D Euclidean Distance Field (`DISTANCE_FIELD_WGSL`)
    /// directly on the `wgpu` Vulkan backend across `num_layers` copper layers.
    ///
    /// # Arguments
    /// * `obstacles_flat` - Flattened `[num_layers * height * width]` obstacle grid (`0` = free, non-zero = obstacle).
    /// * `width` - Grid width in cells.
    /// * `height` - Grid height in cells.
    /// * `num_layers` - Number of copper layers (`z` dimension of workgroup dispatch).
    /// * `radius` - Search radius in grid cells.
    ///
    /// # Returns
    /// `Some(Vec<f32>)` of length `num_layers * height * width` with per-cell Euclidean distances,
    /// or `None` if dimensions are zero, slice length mismatches, or buffer mapping fails.
    ///
    /// # Side Effects
    /// Acquires `self.dispatch_lock`, allocates temporary device storage and staging buffers,
    /// submits `pass_h` and `pass_v` compute passes, blocks on `device.poll(Maintain::Wait)`,
    /// destroys temporary buffers, and adds elapsed GPU time to `self.compute_busy_ns`.
    pub fn compute_distance_field_vulkan_wgsl(
        &self,
        obstacles_flat: &[u32],
        width: u32,
        height: u32,
        num_layers: u32,
        radius: i32,
    ) -> Option<Vec<f32>> {
        let layer_cells = (width * height) as usize;
        let total = layer_cells * (num_layers as usize);
        if obstacles_flat.len() != total || total == 0 || num_layers == 0 {
            return None;
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        let byte_size = (total * std::mem::size_of::<f32>()) as u64;

        let obs_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Obstacles Buffer"),
            contents: bytemuck::cast_slice(obstacles_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let temp_h_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dist TempH Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dist Out Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = GridParams {
            width,
            height,
            radius,
            num_layers,
        };
        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GridParams Uniform"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DistanceField BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: obs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: temp_h_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DistanceField CommandEncoder"),
        });

        let wg_x = (width + 15) / 16;
        let wg_y = (height + 15) / 16;
        let wg_z = num_layers;

        {
            let mut cpass_h = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DistanceField PassH ComputePass"),
                timestamp_writes: None,
            });
            cpass_h.set_pipeline(&self.pipeline);
            cpass_h.set_bind_group(0, &bind_group, &[]);
            cpass_h.dispatch_workgroups(wg_x, wg_y, wg_z);
        }
        {
            let mut cpass_v = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DistanceField PassV ComputePass"),
                timestamp_writes: None,
            });
            cpass_v.set_pipeline(&self.df_v_pipeline);
            cpass_v.set_bind_group(0, &bind_group, &[]);
            cpass_v.dispatch_workgroups(wg_x, wg_y, wg_z);
        }

        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, byte_size);
        let start_t = Instant::now();
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let elapsed_ns = (start_t.elapsed().as_nanos() as u64).max(1);
        self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

        let result = if rx.recv().ok().and_then(|r| r.ok()).is_some() {
            let data = slice.get_mapped_range();
            let vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buf.unmap();
            Some(vec)
        } else {
            None
        };

        staging_buf.destroy();
        out_buf.destroy();
        temp_h_buf.destroy();
        obs_buf.destroy();
        param_buf.destroy();

        result
    }

    /// Computes a multi-layer 2D Euclidean Distance Field across `num_layers` layers in parallel.
    ///
    /// Uses `CudaEngine::compute_distance_field_layers` when CUDA PTX is active, or falls back to
    /// [`Self::compute_distance_field_vulkan_wgsl`].
    ///
    /// # Arguments
    /// * `obstacles_flat` - Flattened `[num_layers * height * width]` obstacle slice.
    /// * `width` - Grid width in cells.
    /// * `height` - Grid height in cells.
    /// * `num_layers` - Number of routing layers.
    /// * `radius` - Search radius in grid cells.
    ///
    /// # Returns
    /// `Some(Vec<f32>)` of length `num_layers * height * width` containing per-layer 2D Euclidean
    /// distances, or `None` if dimensions or slice lengths are invalid.
    ///
    /// # Side Effects
    /// Dispatches GPU compute kernels and updates `self.compute_busy_ns`.
    pub fn compute_distance_field_layers(
        &self,
        obstacles_flat: &[u32],
        width: u32,
        height: u32,
        num_layers: u32,
        radius: i32,
    ) -> Option<Vec<f32>> {
        let layer_cells = (width * height) as usize;
        let total = layer_cells * (num_layers as usize);
        if obstacles_flat.len() != total || total == 0 || num_layers == 0 {
            return None;
        }

        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                if let Some(res) = cuda.compute_distance_field_layers(obstacles_flat, width, height, num_layers, radius) {
                    self.compute_busy_ns.fetch_add(cuda.compute_busy_ns.load(Ordering::Relaxed), Ordering::Relaxed);
                    return Some(res);
                }
            }
        }

        self.compute_distance_field_vulkan_wgsl(obstacles_flat, width, height, num_layers, radius)
    }


    /// Evaluates a batch of candidate PCB component placements in parallel on the GPU.
    ///
    /// For each candidate in `0..params.num_candidates`, computes the weighted sum of ratsnest
    /// Euclidean wirelength, pairwise segment crossings, pairwise footprint AABB overlaps, and
    /// board boundary violations. Dispatches to CUDA PTX when active, or to `BATCH_PLACEMENT_EVAL_WGSL`.
    ///
    /// # Arguments
    /// * `boxes` - Flattened `[num_candidates * num_footprints]` slice of footprint AABBs `[min_x, min_y, max_x, max_y]`.
    /// * `segs` - Flattened `[num_candidates * num_segments]` slice of ratsnest segments `[x1, y1, x2, y2]`.
    /// * `params` - Batch dimensions, board bounding box, and penalty weights ([`PlacementBatchParams`]).
    ///
    /// # Returns
    /// `Some(Vec<f32>)` of length `params.num_candidates` containing total scalar cost per candidate,
    /// or `None` if `num_candidates == 0`, `num_footprints == 0`, or input slice lengths mismatch.
    ///
    /// # Side Effects
    /// Uploads `boxes` and `segs` to GPU buffers, submits a compute pass, maps the output staging
    /// buffer, and increments `self.compute_busy_ns`.
    pub fn evaluate_placement_batch(
        &self,
        boxes: &[[f32; 4]],
        segs: &[[f32; 4]],
        params: &PlacementBatchParams,
    ) -> Option<Vec<f32>> {
        let nc = params.num_candidates as usize;
        let nf = params.num_footprints as usize;
        let ns = params.num_segments as usize;
        if nc == 0 || nf == 0 || boxes.len() != nc * nf || segs.len() != nc * ns {
            return None;
        }

        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                if let Some(res) = cuda.evaluate_placement_batch(boxes, segs, params) {
                    self.compute_busy_ns.fetch_add(cuda.compute_busy_ns.load(Ordering::Relaxed), Ordering::Relaxed);
                    return Some(res);
                }
            }
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        let out_byte_size = (nc * std::mem::size_of::<f32>()) as u64;

        let boxes_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Placement Boxes Buffer"),
            contents: bytemuck::cast_slice(boxes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let dummy_seg = [[0.0f32; 4]; 1];
        let seg_slice: &[[f32; 4]] = if segs.is_empty() { &dummy_seg } else { segs };
        let segs_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Placement Segs Buffer"),
            contents: bytemuck::cast_slice(seg_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Placement Costs Out Buffer"),
            size: out_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Placement Costs Staging Buffer"),
            size: out_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PlacementBatchParams Uniform"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PlacementBatch BindGroup"),
            layout: &self.placement_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: boxes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: segs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PlacementBatch CommandEncoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PlacementBatch ComputePass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.placement_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (params.num_candidates + 63) / 64;
            cpass.dispatch_workgroups(wg_x, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_byte_size);
        let start_t = Instant::now();
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let elapsed_ns = start_t.elapsed().as_nanos() as u64;
        self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

        let result = if rx.recv().ok().and_then(|r| r.ok()).is_some() {
            let data = slice.get_mapped_range();
            let vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buf.unmap();
            Some(vec)
        } else {
            None
        };

        staging_buf.destroy();
        out_buf.destroy();
        boxes_buf.destroy();
        segs_buf.destroy();
        param_buf.destroy();

        result
    }

    /// Runs Parallel Tempering Markov Chain Monte Carlo (MCMC) Simulated Annealing auto-placement
    /// chains on the GPU across `num_passes` compute passes.
    ///
    /// Evaluates exact Separating Axis Theorem (SAT) convex courtyard polygon collisions using
    /// `courtyards` (or synthesized rectangular courtyards via [`GpuCourtyardBuffers::from_fp_descs`]
    /// if `courtyards` is `None`), along with incremental $\Delta E$ wirelength, ratsnest crossing,
    /// passive decoupling affinity, corridor congestion, and signal flow alignment costs.
    ///
    /// # Arguments
    /// * `fp_descs` - Per-footprint descriptors `[half_w, half_h, is_locked, packed_aabb_offset]` of length `params.num_footprints`.
    /// * `edges` - Pairs of `vec4<f32>` edge endpoints (`2 * params.num_edges` elements).
    /// * `init_poses` - Flattened initial footprint poses `[x_mm, y_mm, rot_deg, layer]` of length `params.num_chains * params.num_footprints`.
    /// * `params` - Annealing parameters, board boundaries, and cost weights ([`GpuSaParams`]).
    /// * `num_passes` - Number of sequential compute pass dispatches (each executing `params.steps_per_pass` MCMC steps).
    /// * `courtyards` - Optional pre-built [`GpuCourtyardBuffers`] for exact convex polygon SAT intersection checks.
    ///
    /// # Returns
    /// `Some((best_poses, best_scores))` where:
    /// - `best_poses` has length `num_chains * num_footprints` (`[x_mm, y_mm, rot_deg, layer]`), and
    /// - `best_scores` has length `num_chains` (`[cur_cost, best_cost, cur_overlaps, pass_count]`);
    /// or `None` if buffer dimensions are invalid or GPU execution fails.
    ///
    /// # Side Effects
    /// Dispatches `num_passes` GPU compute passes (via CUDA PTX or `GPU_SA_PLACER_WGSL`), reads back
    /// the best poses and chain score vectors, and updates `self.compute_busy_ns`.
    pub fn run_gpu_sa_placer_chains(
        &self,
        fp_descs: &[[f32; 4]],
        edges: &[[f32; 4]],
        init_poses: &[[f32; 4]],
        params: &GpuSaParams,
        num_passes: u32,
        courtyards: Option<&GpuCourtyardBuffers>,
    ) -> Option<(Vec<[f32; 4]>, Vec<[f32; 4]>)> {
        let nc = params.num_chains as usize;
        let nf = params.num_footprints as usize;
        let ne = params.num_edges as usize;
        if nc == 0 || nf == 0 || fp_descs.len() != nf || edges.len() != ne * 2 || init_poses.len() != nc * nf {
            return None;
        }

        let default_courtyards;
        let cyd = match courtyards {
            Some(c) if c.fp_poly_ranges.len() == nf && !c.courtyard_polys.is_empty() && !c.courtyard_vertices.is_empty() => c,
            _ => {
                default_courtyards = GpuCourtyardBuffers::from_fp_descs(fp_descs);
                &default_courtyards
            }
        };

        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                if let Some(res) = cuda.run_gpu_sa_placer_chains(fp_descs, edges, init_poses, params, num_passes, Some(cyd)) {
                    self.compute_busy_ns.fetch_add(cuda.compute_busy_ns.load(Ordering::Relaxed), Ordering::Relaxed);
                    return Some(res);
                }
            }
        }

        let _guard = self.dispatch_lock.lock().ok()?;

        let mut full_poses = Vec::with_capacity(nc * nf * 2);
        full_poses.extend_from_slice(init_poses);
        full_poses.extend_from_slice(init_poses);
        let init_scores = vec![[0.0f32; 4]; nc];

        let fp_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SA FP Descs Buffer"),
            contents: bytemuck::cast_slice(fp_descs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let dummy_edge = [[0.0f32; 4]; 2];
        let edge_slice: &[[f32; 4]] = if edges.is_empty() { &dummy_edge } else { edges };
        let edges_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SA Edges Buffer"),
            contents: bytemuck::cast_slice(edge_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let poses_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SA Chain Poses Buffer"),
            contents: bytemuck::cast_slice(&full_poses),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let scores_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SA Chain Scores Buffer"),
            contents: bytemuck::cast_slice(&init_scores),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuSaParams Uniform"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let poly_ranges_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SA FP Poly Ranges Buffer"),
            contents: bytemuck::cast_slice(&cyd.fp_poly_ranges),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let polys_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SA Courtyard Polys Buffer"),
            contents: bytemuck::cast_slice(&cyd.courtyard_polys),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let verts_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SA Courtyard Vertices Buffer"),
            contents: bytemuck::cast_slice(&cyd.courtyard_vertices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let best_poses_bytes = (nc * nf * std::mem::size_of::<[f32; 4]>()) as u64;
        let scores_bytes = (nc * std::mem::size_of::<[f32; 4]>()) as u64;

        let poses_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SA Best Poses Staging"),
            size: best_poses_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scores_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SA Scores Staging"),
            size: scores_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GpuSaPlacer BindGroup"),
            layout: &self.sa_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edges_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: poses_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scores_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: poly_ranges_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: polys_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: verts_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GpuSaPlacer CommandEncoder"),
        });

        let wg_x = (params.num_chains + 63) / 64;
        for _ in 0..num_passes.max(1) {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GpuSaPlacer ComputePass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.sa_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(wg_x, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&poses_buf, best_poses_bytes, &poses_staging, 0, best_poses_bytes);
        encoder.copy_buffer_to_buffer(&scores_buf, 0, &scores_staging, 0, scores_bytes);

        let start_t = Instant::now();
        self.queue.submit(Some(encoder.finish()));

        let poses_slice = poses_staging.slice(..);
        let scores_slice = scores_staging.slice(..);
        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx2, rx2) = std::sync::mpsc::channel();
        poses_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx1.send(res);
        });
        scores_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx2.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let elapsed_ns = start_t.elapsed().as_nanos() as u64;
        self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

        let result = if rx1.recv().ok().and_then(|r| r.ok()).is_some()
            && rx2.recv().ok().and_then(|r| r.ok()).is_some()
        {
            let p_data = poses_slice.get_mapped_range();
            let s_data = scores_slice.get_mapped_range();
            let best_poses: Vec<[f32; 4]> = bytemuck::cast_slice(&p_data).to_vec();
            let best_scores: Vec<[f32; 4]> = bytemuck::cast_slice(&s_data).to_vec();
            drop(p_data);
            drop(s_data);
            poses_staging.unmap();
            scores_staging.unmap();
            Some((best_poses, best_scores))
        } else {
            None
        };

        poses_staging.destroy();
        scores_staging.destroy();
        poses_buf.destroy();
        scores_buf.destroy();
        fp_buf.destroy();
        edges_buf.destroy();
        param_buf.destroy();
        poly_ranges_buf.destroy();
        polys_buf.destroy();
        verts_buf.destroy();

        result
    }

    /// Evaluates the exact number of pairwise courtyard polygon overlaps and total overlap/boundary cost
    /// for one or more board placement configurations on the GPU.
    ///
    /// Dispatches a zero-step (`steps_per_pass = 0`) pass of [`Self::run_gpu_sa_placer_chains`] so the
    /// GPU executes `evaluate_chain_state` / `evaluate_chain_state_fast` (with AABB pruning, rectilinear
    /// slab fast-paths, opposite-side SMD isolation, and exact Separating Axis Theorem convex polygon checks)
    /// across all `num_configs` placement states in parallel.
    ///
    /// # Arguments
    /// * `fp_descs` - Per-footprint descriptor slice (`[hw, hh, locked, packed_center]`, length `nf`).
    /// * `poses` - Flattened footprint poses (`[x, y, rot_deg, layer]`, length `num_configs * nf`).
    /// * `courtyards` - Flattened local-space courtyard polygon buffers ([`GpuCourtyardBuffers`]).
    /// * `board_bounds` - Board outline bounding box `[min_x, min_y, max_x, max_y]` in millimeters (`mm`).
    /// * `overlap_penalty` - Penalty weight per overlapping courtyard pair.
    /// * `boundary_penalty` - Penalty weight for board outline excursions.
    ///
    /// # Returns
    /// `Some(Vec<(usize, f32)>)` of length `num_configs` containing `(overlap_count, total_penalty_cost)`
    /// for each configuration, or `None` if `fp_descs` is empty, `poses.len()` is not a multiple of
    /// `fp_descs.len()`, or GPU dispatch fails.
    ///
    /// # Side Effects
    /// Uploads descriptor and courtyard buffers to the GPU, executes a single compute pass, and updates `self.compute_busy_ns`.
    pub fn evaluate_courtyard_overlaps_gpu(
        &self,
        fp_descs: &[[f32; 4]],
        poses: &[[f32; 4]],
        courtyards: &GpuCourtyardBuffers,
        board_bounds: [f32; 4],
        overlap_penalty: f32,
        boundary_penalty: f32,
    ) -> Option<Vec<(usize, f32)>> {
        let nf = fp_descs.len();
        if nf == 0 || poses.is_empty() || poses.len() % nf != 0 {
            return None;
        }
        let num_configs = (poses.len() / nf) as u32;
        let params = GpuSaParams {
            num_chains: num_configs,
            num_footprints: nf as u32,
            num_edges: 0,
            steps_per_pass: 0,
            seed: 0xC001_D00Du32,
            _pad0: 0,
            corridor_congestion_weight: 0.0,
            flow_alignment_weight: 0.0,
            board_min_x: board_bounds[0],
            board_min_y: board_bounds[1],
            board_max_x: board_bounds[2],
            board_max_y: board_bounds[3],
            grid_step: 0.1,
            wirelength_weight: 0.0,
            crossing_weight: 0.0,
            overlap_penalty,
            boundary_penalty,
            passive_affinity_weight: 0.0,
            passive_max_dist: 3.5,
            base_temp: 1.0,
        };
        let (_, scores) = self.run_gpu_sa_placer_chains(
            fp_descs,
            &[],
            poses,
            &params,
            1,
            Some(courtyards),
        )?;
        Some(
            scores
                .into_iter()
                .map(|s| (s[2].round().max(0.0) as usize, s[0]))
                .collect(),
        )
    }

    /// Evaluates per-cell 3D routing traversal costs across all copper layers for a single net.
    ///
    /// Dispatches to `CudaEngine::compute_3d_routing_costs` when `self.backend_kind == GpuBackendKind::CudaPtx`,
    /// or falls back to [`Self::compute_3d_routing_costs_vulkan_wgsl`].
    ///
    /// # Arguments
    /// * `obstacles` - 3D obstacle net ID grid (`[num_layers * height * width]`, `0` = free).
    /// * `dist_field` - 3D Euclidean distance field (`[num_layers * height * width]`).
    /// * `history_cost` - PathFinder historical congestion grid (`[num_layers * height * width]`).
    /// * `usage_count` - Present net occupancy count grid (`[num_layers * height * width]`).
    /// * `params` - Routing grid dimensions, active `net_id`, clearance radius, and penalty weights ([`GpuRouteCostParams`]).
    ///
    /// # Returns
    /// `Some(Vec<f32>)` of length `num_layers * height * width` where blocked cells are `-1.0` and
    /// legal cells are `>= 1.0`, or `None` if buffer lengths mismatch.
    ///
    /// # Side Effects
    /// Uploads grid buffers to the GPU, executes the 3D routing cost kernel, and updates `self.compute_busy_ns`.
    pub fn compute_3d_routing_costs(
        &self,
        obstacles: &[u32],
        dist_field: &[f32],
        history_cost: &[f32],
        usage_count: &[u32],
        params: &GpuRouteCostParams,
    ) -> Option<Vec<f32>> {
        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                if let Some(res) = cuda.compute_3d_routing_costs(obstacles, dist_field, history_cost, usage_count, params) {
                    self.compute_busy_ns.fetch_add(cuda.compute_busy_ns.load(Ordering::Relaxed), Ordering::Relaxed);
                    return Some(res);
                }
            }
        }
        self.compute_3d_routing_costs_vulkan_wgsl(obstacles, dist_field, history_cost, usage_count, params)
    }

    /// Executes the 3D routing cell cost evaluation shader (`COMPUTE_3D_ROUTING_COST_WGSL`) directly
    /// on the `wgpu` Vulkan backend.
    ///
    /// # Arguments
    /// * `obstacles` - 3D obstacle net ID grid (`[num_layers * height * width]`).
    /// * `dist_field` - 3D Euclidean distance field (`[num_layers * height * width]`).
    /// * `history_cost` - PathFinder historical congestion grid (`[num_layers * height * width]`).
    /// * `usage_count` - Present net occupancy count grid (`[num_layers * height * width]`).
    /// * `params` - Routing grid dimensions, active `net_id`, clearance radius, and penalty weights ([`GpuRouteCostParams`]).
    ///
    /// # Returns
    /// `Some(Vec<f32>)` of per-cell traversal costs (`-1.0` if blocked), or `None` if dimensions are
    /// zero, slice lengths mismatch, or staging buffer readback fails.
    ///
    /// # Side Effects
    /// Acquires `self.dispatch_lock`, allocates device and staging buffers, dispatches
    /// `self.route_cost_pipeline`, blocks on `device.poll(Maintain::Wait)`, and updates `self.compute_busy_ns`.
    pub fn compute_3d_routing_costs_vulkan_wgsl(
        &self,
        obstacles: &[u32],
        dist_field: &[f32],
        history_cost: &[f32],
        usage_count: &[u32],
        params: &GpuRouteCostParams,
    ) -> Option<Vec<f32>> {
        let total = (params.width as usize)
            .checked_mul(params.height as usize)?
            .checked_mul(params.num_layers as usize)?;
        if total == 0
            || obstacles.len() != total
            || dist_field.len() != total
            || history_cost.len() != total
            || usage_count.len() != total
        {
            return None;
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        let byte_size = (total * std::mem::size_of::<f32>()) as u64;

        let obs_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3D Route Cost Obstacles Buffer"),
            contents: bytemuck::cast_slice(obstacles),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let df_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3D Route Cost DistField Buffer"),
            contents: bytemuck::cast_slice(dist_field),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let hist_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3D Route Cost History Buffer"),
            contents: bytemuck::cast_slice(history_cost),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let usage_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3D Route Cost Usage Buffer"),
            contents: bytemuck::cast_slice(usage_count),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3D Route Cost Out Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3D Route Cost Staging Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuRouteCostParams Uniform"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RouteCost3D BindGroup"),
            layout: &self.route_cost_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: obs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: df_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: hist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: usage_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RouteCost3D CommandEncoder"),
        });

        let wg_x = (params.width + 15) / 16;
        let wg_y = (params.height + 15) / 16;
        let wg_z = params.num_layers;
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RouteCost3D ComputePass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.route_cost_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }

        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, byte_size);
        let start_t = Instant::now();
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let elapsed_ns = start_t.elapsed().as_nanos() as u64;
        self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

        let result = if rx.recv().ok().and_then(|r| r.ok()).is_some() {
            let data = slice.get_mapped_range();
            let vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buf.unmap();
            Some(vec)
        } else {
            None
        };

        staging_buf.destroy();
        out_buf.destroy();
        obs_buf.destroy();
        df_buf.destroy();
        hist_buf.destroy();
        usage_buf.destroy();
        param_buf.destroy();

        result
    }

    /// Evaluates 3D routing cell cost grids for a batch of net/parameter configurations, distributing
    /// work across all active GPUs ([`Self::all_engines`]) in parallel when multiple GPUs are present.
    ///
    /// # Arguments
    /// * `obstacles` - Shared 3D obstacle net ID grid (`[num_layers * height * width]`).
    /// * `dist_field` - Shared 3D Euclidean distance field (`[num_layers * height * width]`).
    /// * `history_cost` - Shared PathFinder historical congestion grid (`[num_layers * height * width]`).
    /// * `usage_count` - Shared present net occupancy count grid (`[num_layers * height * width]`).
    /// * `batch_params` - Slice of per-net [`GpuRouteCostParams`] configurations to evaluate.
    ///
    /// # Returns
    /// A `Vec<Option<Vec<f32>>>` of length `batch_params.len()`, preserving the input ordering.
    ///
    /// # Side Effects
    /// Spawns scoped worker threads across [`Self::all_engines`] when multiple GPUs are active,
    /// dispatches CUDA or Vulkan compute kernels, and updates `compute_busy_ns`.
    pub fn compute_3d_routing_costs_batch(
        self: &Arc<Self>,
        obstacles: &[u32],
        dist_field: &[f32],
        history_cost: &[f32],
        usage_count: &[u32],
        batch_params: &[GpuRouteCostParams],
    ) -> Vec<Option<Vec<f32>>> {
        if batch_params.is_empty() {
            return Vec::new();
        }
        let engines = self.all_engines();
        if engines.len() <= 1 || batch_params.len() == 1 {
            if self.backend_kind == GpuBackendKind::CudaPtx {
                if let Some(cuda) = &self.cuda_engine {
                    return cuda.compute_3d_routing_costs_multi(obstacles, dist_field, history_cost, usage_count, batch_params);
                }
            }
            return batch_params
                .iter()
                .map(|p| self.compute_3d_routing_costs(obstacles, dist_field, history_cost, usage_count, p))
                .collect();
        }

        let num_engines = engines.len();
        let mut results: Vec<Option<Vec<f32>>> = (0..batch_params.len()).map(|_| None).collect();
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (eng_idx, engine) in engines.iter().enumerate() {
                let assigned: Vec<(usize, GpuRouteCostParams)> = batch_params
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i % num_engines == eng_idx)
                    .map(|(i, p)| (i, *p))
                    .collect();
                if assigned.is_empty() {
                    continue;
                }
                let eng_ref = Arc::clone(engine);
                handles.push(scope.spawn(move || {
                    let sub_params: Vec<GpuRouteCostParams> = assigned.iter().map(|&(_, p)| p).collect();
                    let sub_out = if eng_ref.backend_kind == GpuBackendKind::CudaPtx {
                        if let Some(cuda) = &eng_ref.cuda_engine {
                            cuda.compute_3d_routing_costs_multi(obstacles, dist_field, history_cost, usage_count, &sub_params)
                        } else {
                            sub_params
                                .iter()
                                .map(|p| eng_ref.compute_3d_routing_costs(obstacles, dist_field, history_cost, usage_count, p))
                                .collect()
                        }
                    } else {
                        sub_params
                            .iter()
                            .map(|p| eng_ref.compute_3d_routing_costs(obstacles, dist_field, history_cost, usage_count, p))
                            .collect()
                    };
                    assigned
                        .into_iter()
                        .map(|(orig_i, _)| orig_i)
                        .zip(sub_out)
                        .collect::<Vec<(usize, Option<Vec<f32>>)>>()
                }));
            }
            for handle in handles {
                if let Ok(sub_res) = handle.join() {
                    for (orig_i, map_opt) in sub_res {
                        results[orig_i] = map_opt;
                    }
                }
            }
        });
        results
    }

    /// Propagates a 3D Lee/Moore wavefront across `cell_costs` from `seeds` for `iterations` relaxation steps.
    ///
    /// Dispatches to `CudaEngine::compute_3d_wavefront` when CUDA PTX is active, or falls back to
    /// [`Self::compute_3d_wavefront_vulkan_wgsl`].
    ///
    /// # Arguments
    /// * `cell_costs` - Per-cell traversal costs (`[num_layers * height * width]`, `< 0.0` = blocked).
    /// * `seeds` - Flat 3D cell indices (`layer * width * height + y * width + x`) initialized to distance `0.0`.
    /// * `params` - Grid dimensions (`width`, `height`, `num_layers`) in [`GpuRouteCostParams`].
    /// * `via_cost` - Additive cost penalty for inter-layer via transitions.
    /// * `iterations` - Number of synchronous ping-pong relaxation passes to execute.
    ///
    /// # Returns
    /// `Some(Vec<f32>)` of length `num_layers * height * width` containing accumulated shortest-path
    /// distances from `seeds` (`1.0e20` if unreached), or `None` if dimensions are invalid.
    ///
    /// # Side Effects
    /// Executes `iterations` GPU wavefront relaxation passes and updates `self.compute_busy_ns`.
    pub fn compute_3d_wavefront(
        &self,
        cell_costs: &[f32],
        seeds: &[u32],
        params: &GpuRouteCostParams,
        via_cost: f32,
        iterations: u32,
    ) -> Option<Vec<f32>> {
        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                if let Some(res) = cuda.compute_3d_wavefront(cell_costs, seeds, params, via_cost, iterations) {
                    self.compute_busy_ns.fetch_add(cuda.compute_busy_ns.load(Ordering::Relaxed), Ordering::Relaxed);
                    return Some(res);
                }
            }
        }
        self.compute_3d_wavefront_vulkan_wgsl(cell_costs, seeds, params, via_cost, iterations)
    }

    /// Acquires the cached [`CachedWavefrontBuffers`] pool for a 3D grid of `total` cells,
    /// (re)allocating device buffers only when `total` changes, and invokes `f` with the pool.
    ///
    /// # Arguments
    /// * `total` - Total number of 3D grid cells (`width * height * num_layers`).
    /// * `f` - Closure receiving a shared reference to the sized [`CachedWavefrontBuffers`].
    ///
    /// # Returns
    /// `Some(R)` with the closure's return value, or `None` if locking `self.wavefront_buffers` fails.
    ///
    /// # Side Effects
    /// Locks `self.wavefront_buffers`, destroys previous `wgpu::Buffer` handles if `total_cells != total`,
    /// allocates new device/staging buffers on a size change, and increments `alloc_count` or `reuse_count`.
    fn with_wavefront_buffers<R, F: FnOnce(&CachedWavefrontBuffers) -> R>(
        &self,
        total: usize,
        f: F,
    ) -> Option<R> {
        let mut guard = self.wavefront_buffers.lock().ok()?;
        let needs_alloc = match *guard {
            Some(ref c) => c.total_cells != total,
            None => true,
        };
        if needs_alloc {
            let prev_alloc = if let Some(old) = guard.take() {
                old.obst_buf.destroy();
                old.df_buf.destroy();
                old.hist_buf.destroy();
                old.usage_buf.destroy();
                old.cost_buf.destroy();
                old.dist_a_buf.destroy();
                old.dist_b_buf.destroy();
                old.path_out_buf.destroy();
                old.path_len_buf.destroy();
                old.staging_buf.destroy();
                old.staging_len_buf.destroy();
                old.staging_path_buf.destroy();
                old.alloc_count
            } else {
                0
            };

            let u32_byte_size = (total * std::mem::size_of::<u32>()) as u64;
            let f32_byte_size = (total * std::mem::size_of::<f32>()) as u64;
            let max_path_len = total.min(65536).max(64) as u32;
            let path_byte_size = (max_path_len as usize * 3 * std::mem::size_of::<u32>()) as u64;
            let len_byte_size = std::mem::size_of::<u32>() as u64;

            let zero_f32 = vec![0.0f32; total];
            let zero_u32 = vec![0u32; total];

            let obst_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CachedWavefront Obstacles Buffer"),
                size: u32_byte_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let df_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CachedWavefront ZeroDF Buffer"),
                contents: bytemuck::cast_slice(&zero_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let hist_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CachedWavefront ZeroHist Buffer"),
                contents: bytemuck::cast_slice(&zero_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let usage_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CachedWavefront ZeroUsage Buffer"),
                contents: bytemuck::cast_slice(&zero_u32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let cost_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CachedWavefront CellCosts Buffer"),
                size: f32_byte_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let dist_a_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CachedWavefront Dist A Buffer"),
                size: f32_byte_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let dist_b_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CachedWavefront Dist B Buffer"),
                size: f32_byte_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let path_out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CachedWavefront PathOut Buffer"),
                size: path_byte_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let path_len_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CachedWavefront PathLen Buffer"),
                contents: bytemuck::bytes_of(&0u32),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
            let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CachedWavefront Staging Buffer"),
                size: f32_byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let staging_len_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CachedWavefront Staging PathLen Buffer"),
                size: len_byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let staging_path_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("CachedWavefront Staging Path Buffer"),
                size: path_byte_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            *guard = Some(CachedWavefrontBuffers {
                total_cells: total,
                max_path_len,
                obst_buf,
                df_buf,
                hist_buf,
                usage_buf,
                cost_buf,
                dist_a_buf,
                dist_b_buf,
                path_out_buf,
                path_len_buf,
                staging_buf,
                staging_len_buf,
                staging_path_buf,
                alloc_count: prev_alloc + 1,
                reuse_count: 0,
            });
        } else if let Some(ref mut c) = *guard {
            c.reuse_count += 1;
        }
        guard.as_ref().map(f)
    }

    /// Executes `iterations` ping-pong passes of the 3D wavefront relaxation shader (`WAVEFRONT_3D_RELAX_WGSL`)
    /// directly on the `wgpu` Vulkan backend using `Self::with_wavefront_buffers`.
    ///
    /// # Arguments
    /// * `cell_costs` - Per-cell traversal costs (`[num_layers * height * width]`, `< 0.0` = blocked).
    /// * `seeds` - Flat 3D cell indices initialized to distance `0.0`.
    /// * `params` - Grid dimensions (`width`, `height`, `num_layers`).
    /// * `via_cost` - Additive cost penalty for inter-layer via transitions.
    /// * `iterations` - Number of relaxation passes (`0` returns the initialized seed grid immediately).
    ///
    /// # Returns
    /// `Some(Vec<f32>)` containing the relaxed 3D distance field of length `num_layers * height * width`,
    /// or `None` if `cell_costs.len() != total` or `seeds.is_empty()`.
    ///
    /// # Side Effects
    /// Acquires `self.dispatch_lock` and `self.wavefront_buffers`, writes `cell_costs` and initial
    /// wavefront buffers to the device queue, dispatches `iterations` compute passes, maps the staging
    /// buffer, and updates `self.compute_busy_ns`.
    pub fn compute_3d_wavefront_vulkan_wgsl(
        &self,
        cell_costs: &[f32],
        seeds: &[u32],
        params: &GpuRouteCostParams,
        via_cost: f32,
        iterations: u32,
    ) -> Option<Vec<f32>> {
        let total = (params.width as usize)
            .checked_mul(params.height as usize)?
            .checked_mul(params.num_layers as usize)?;
        if total == 0 || cell_costs.len() != total || seeds.is_empty() {
            return None;
        }

        let mut init_wf = vec![1.0e20f32; total];
        for &s in seeds {
            let idx = s as usize;
            if idx < total {
                init_wf[idx] = 0.0;
            }
        }
        if iterations == 0 {
            return Some(init_wf);
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        let byte_size = (total * std::mem::size_of::<f32>()) as u64;

        self.with_wavefront_buffers(total, |pool| {
            self.queue.write_buffer(&pool.cost_buf, 0, bytemuck::cast_slice(cell_costs));
            self.queue.write_buffer(&pool.dist_a_buf, 0, bytemuck::cast_slice(&init_wf));
            self.queue.write_buffer(&pool.dist_b_buf, 0, bytemuck::cast_slice(&init_wf));

            let wf_params = WavefrontUniformParams {
                width: params.width,
                height: params.height,
                num_layers: params.num_layers,
                via_cost,
            };
            let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WavefrontUniformParams Buffer"),
                contents: bytemuck::bytes_of(&wf_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bg_ab = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Wavefront3D BindGroup A->B"),
                layout: &self.wavefront_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pool.cost_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pool.dist_a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pool.dist_b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buf.as_entire_binding(),
                    },
                ],
            });

            let bg_ba = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Wavefront3D BindGroup B->A"),
                layout: &self.wavefront_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pool.cost_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pool.dist_b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pool.dist_a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Wavefront3D CommandEncoder"),
            });

            let wg_x = (params.width + 15) / 16;
            let wg_y = (params.height + 15) / 16;
            let wg_z = params.num_layers;

            for iter in 0..iterations {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Wavefront3D ComputePass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.wavefront_pipeline);
                cpass.set_bind_group(0, if iter % 2 == 0 { &bg_ab } else { &bg_ba }, &[]);
                cpass.dispatch_workgroups(wg_x, wg_y, wg_z);
            }

            let final_buf = if iterations % 2 == 1 { &pool.dist_b_buf } else { &pool.dist_a_buf };
            encoder.copy_buffer_to_buffer(final_buf, 0, &pool.staging_buf, 0, byte_size);

            let start_t = Instant::now();
            self.queue.submit(Some(encoder.finish()));

            let slice = pool.staging_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            self.device.poll(wgpu::Maintain::Wait);
            let elapsed_ns = (start_t.elapsed().as_nanos() as u64).max(1);
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let result = if rx.recv().ok().and_then(|r| r.ok()).is_some() {
                let data = slice.get_mapped_range();
                let vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                pool.staging_buf.unmap();
                Some(vec)
            } else {
                None
            };

            param_buf.destroy();
            result
        })?
    }

    /// Solves a shortest 3D route path from `src` to `dst` on `obstacle_grid` with zero extra cell clearance.
    ///
    /// Convenience wrapper around [`Self::solve_3d_wavefront_path_with_clearance`] with `clearance_cells = 0`.
    ///
    /// # Arguments
    /// * `obstacle_grid` - Flat 3D obstacle net ID grid (`[layers * gh * gw]`, `0` = free).
    /// * `gw` - Grid width in cells.
    /// * `gh` - Grid height in cells.
    /// * `layers` - Number of copper routing layers.
    /// * `src` - Source cell coordinates `(x, y, layer)`.
    /// * `dst` - Destination cell coordinates `(x, y, layer)`.
    /// * `net_id` - Active electrical net ID being routed.
    /// * `via_cost` - Additive cost penalty for inter-layer via transitions.
    ///
    /// # Returns
    /// `Some(Vec<(usize, usize, usize)>)` ordered from `src` to `dst` (inclusive) if a valid path is found,
    /// or `None` if unreachable or out of bounds.
    ///
    /// # Side Effects
    /// Dispatches GPU wavefront routing and backtracking kernels and updates `self.compute_busy_ns`.
    pub fn solve_3d_wavefront_path(
        &self,
        obstacle_grid: &[u32],
        gw: usize,
        gh: usize,
        layers: usize,
        src: (usize, usize, usize),
        dst: (usize, usize, usize),
        net_id: u32,
        via_cost: f32,
    ) -> Option<Vec<(usize, usize, usize)>> {
        self.solve_3d_wavefront_path_with_clearance(
            obstacle_grid,
            gw,
            gh,
            layers,
            src,
            dst,
            net_id,
            via_cost,
            0,
        )
    }

    /// Solves a shortest 3D route path from `src` to `dst` enforcing a circular obstacle clearance
    /// of `clearance_cells` grid cells.
    ///
    /// Dispatches to `CudaEngine::solve_3d_wavefront_path_with_clearance` when CUDA PTX is active,
    /// or falls back to [`Self::solve_3d_wavefront_path_vulkan_wgsl`].
    ///
    /// # Arguments
    /// * `obstacle_grid` - Flat 3D obstacle net ID grid (`[layers * gh * gw]`, `0` = free).
    /// * `gw` - Grid width in cells.
    /// * `gh` - Grid height in cells.
    /// * `layers` - Number of copper routing layers.
    /// * `src` - Source cell coordinates `(x, y, layer)`.
    /// * `dst` - Destination cell coordinates `(x, y, layer)`.
    /// * `net_id` - Active electrical net ID being routed.
    /// * `via_cost` - Additive cost penalty for inter-layer via transitions.
    /// * `clearance_cells` - Circular clearance radius in grid cells around foreign-net obstacles.
    ///
    /// # Returns
    /// `Some(Vec<(usize, usize, usize)>)` ordered from `src` to `dst` (inclusive), or `None` if blocked/unreachable.
    ///
    /// # Side Effects
    /// Dispatches GPU cost evaluation, wavefront relaxation, and backtracking passes, and updates `self.compute_busy_ns`.
    pub fn solve_3d_wavefront_path_with_clearance(
        &self,
        obstacle_grid: &[u32],
        gw: usize,
        gh: usize,
        layers: usize,
        src: (usize, usize, usize),
        dst: (usize, usize, usize),
        net_id: u32,
        via_cost: f32,
        clearance_cells: i32,
    ) -> Option<Vec<(usize, usize, usize)>> {
        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                let res = cuda.solve_3d_wavefront_path_with_clearance(
                    obstacle_grid,
                    gw,
                    gh,
                    layers,
                    src,
                    dst,
                    net_id,
                    via_cost,
                    clearance_cells,
                );
                self.compute_busy_ns.fetch_add(
                    cuda.compute_busy_ns.load(Ordering::Relaxed),
                    Ordering::Relaxed,
                );
                return res;
            }
        }
        self.solve_3d_wavefront_path_vulkan_wgsl(
            obstacle_grid,
            gw,
            gh,
            layers,
            src,
            dst,
            net_id,
            via_cost,
            clearance_cells,
        )
    }

    /// End-to-end `wgpu` Vulkan WGSL 3D wavefront path solver combining cell cost evaluation
    /// (`COMPUTE_3D_ROUTING_COST_WGSL`), iterative wavefront relaxation (`WAVEFRONT_3D_RELAX_WGSL`),
    /// and on-GPU path backtracking (`WAVEFRONT_3D_BACKTRACK_WGSL`) in a single command submission.
    ///
    /// # Arguments
    /// * `obstacle_grid` - Flat 3D obstacle net ID grid (`[layers * gh * gw]`).
    /// * `gw` - Grid width in cells.
    /// * `gh` - Grid height in cells.
    /// * `layers` - Number of copper routing layers.
    /// * `src` - Source cell `(x, y, layer)`.
    /// * `dst` - Destination cell `(x, y, layer)`.
    /// * `net_id` - Active electrical net ID.
    /// * `via_cost` - Inter-layer via transition cost.
    /// * `clearance_cells` - Circular clearance radius in grid cells.
    ///
    /// # Returns
    /// `Some(Vec<(usize, usize, usize)>)` containing the backtracked 3D path from `src` to `dst`,
    /// or `None` if `src`/`dst` are out of bounds, blocked by a foreign net, or unreachable.
    ///
    /// # Side Effects
    /// Acquires `self.dispatch_lock` and `self.wavefront_buffers`, submits cost, relaxation, and
    /// backtrack passes in a single `wgpu::CommandEncoder`, reads back only the compact path buffer,
    /// and updates `self.compute_busy_ns`.
    pub fn solve_3d_wavefront_path_vulkan_wgsl(
        &self,
        obstacle_grid: &[u32],
        gw: usize,
        gh: usize,
        layers: usize,
        src: (usize, usize, usize),
        dst: (usize, usize, usize),
        net_id: u32,
        via_cost: f32,
        clearance_cells: i32,
    ) -> Option<Vec<(usize, usize, usize)>> {
        if gw == 0 || gh == 0 || layers == 0 {
            return None;
        }
        let total = gw.checked_mul(gh)?.checked_mul(layers)?;
        if total == 0 || obstacle_grid.len() != total {
            return None;
        }
        if src.0 >= gw
            || src.1 >= gh
            || src.2 >= layers
            || dst.0 >= gw
            || dst.1 >= gh
            || dst.2 >= layers
        {
            return None;
        }

        let layer_stride = gw * gh;
        let src_idx = src.2 * layer_stride + src.1 * gw + src.0;
        let dst_idx = dst.2 * layer_stride + dst.1 * gw + dst.0;
        let src_obs = obstacle_grid[src_idx];
        let dst_obs = obstacle_grid[dst_idx];
        if (src_obs != 0 && src_obs != net_id) || (dst_obs != 0 && dst_obs != net_id) {
            return None;
        }
        if src == dst {
            return Some(vec![src]);
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        let len_byte_size = std::mem::size_of::<u32>() as u64;

        let mut init_wf = vec![1.0e20f32; total];
        init_wf[src_idx] = 0.0;

        self.with_wavefront_buffers(total, |pool| {
            let max_path_len = pool.max_path_len;
            let path_byte_size = (max_path_len as usize * 3 * std::mem::size_of::<u32>()) as u64;

            self.queue.write_buffer(&pool.obst_buf, 0, bytemuck::cast_slice(obstacle_grid));
            self.queue.write_buffer(&pool.dist_a_buf, 0, bytemuck::cast_slice(&init_wf));
            self.queue.write_buffer(&pool.dist_b_buf, 0, bytemuck::cast_slice(&init_wf));
            self.queue.write_buffer(&pool.path_len_buf, 0, bytemuck::bytes_of(&0u32));

            let rc_params = GpuRouteCostParams {
                width: gw as u32,
                height: gh as u32,
                num_layers: layers as u32,
                net_id,
                clearance_cells: clearance_cells.max(0),
                present_congestion_factor: 0.0,
                obstacle_clearance_weight: 0.0,
                plane_layer_mask: 0,
            };
            let rc_param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WavefrontSolve RouteCostParams Buffer"),
                contents: bytemuck::bytes_of(&rc_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let wf_params = WavefrontUniformParams {
                width: gw as u32,
                height: gh as u32,
                num_layers: layers as u32,
                via_cost,
            };
            let wf_param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WavefrontSolve WFParams Buffer"),
                contents: bytemuck::bytes_of(&wf_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bt_params = WavefrontBacktrackUniformParams {
                width: gw as u32,
                height: gh as u32,
                num_layers: layers as u32,
                via_cost,
                src_x: src.0 as u32,
                src_y: src.1 as u32,
                src_layer: src.2 as u32,
                dst_x: dst.0 as u32,
                dst_y: dst.1 as u32,
                dst_layer: dst.2 as u32,
                max_path_len,
                pad: 0,
            };
            let bt_param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WavefrontSolve BacktrackParams Buffer"),
                contents: bytemuck::bytes_of(&bt_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bg_rc = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("WavefrontSolve RouteCost BindGroup"),
                layout: &self.route_cost_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pool.obst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pool.df_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pool.hist_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pool.usage_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: pool.cost_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: rc_param_buf.as_entire_binding(),
                    },
                ],
            });

            let bg_wf_ab = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("WavefrontSolve WF BindGroup A->B"),
                layout: &self.wavefront_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pool.cost_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pool.dist_a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pool.dist_b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wf_param_buf.as_entire_binding(),
                    },
                ],
            });

            let bg_wf_ba = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("WavefrontSolve WF BindGroup B->A"),
                layout: &self.wavefront_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pool.cost_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pool.dist_b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pool.dist_a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wf_param_buf.as_entire_binding(),
                    },
                ],
            });

            let iterations = ((gw + gh + (gw * gh / 2).min(512) + layers * 4)
                .max(32)
                .min(4096)) as u32;
            let final_wf_buf = if iterations % 2 == 1 { &pool.dist_b_buf } else { &pool.dist_a_buf };

            let bg_bt = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("WavefrontSolve Backtrack BindGroup"),
                layout: &self.wavefront_backtrack_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: final_wf_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pool.cost_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pool.path_out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pool.path_len_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: bt_param_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("WavefrontSolve CommandEncoder"),
            });

            let wg_x = (gw as u32 + 15) / 16;
            let wg_y = (gh as u32 + 15) / 16;
            let wg_z = layers as u32;

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("WavefrontSolve RouteCost Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.route_cost_pipeline);
                cpass.set_bind_group(0, &bg_rc, &[]);
                cpass.dispatch_workgroups(wg_x, wg_y, wg_z);
            }

            for iter in 0..iterations {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("WavefrontSolve Relax Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.wavefront_pipeline);
                cpass.set_bind_group(0, if iter % 2 == 0 { &bg_wf_ab } else { &bg_wf_ba }, &[]);
                cpass.dispatch_workgroups(wg_x, wg_y, wg_z);
            }

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("WavefrontSolve Backtrack Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.wavefront_backtrack_pipeline);
                cpass.set_bind_group(0, &bg_bt, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&pool.path_len_buf, 0, &pool.staging_len_buf, 0, len_byte_size);
            encoder.copy_buffer_to_buffer(&pool.path_out_buf, 0, &pool.staging_path_buf, 0, path_byte_size);

            let start_t = Instant::now();
            self.queue.submit(Some(encoder.finish()));

            let len_slice = pool.staging_len_buf.slice(..);
            let path_slice = pool.staging_path_buf.slice(..);
            let (tx1, rx1) = std::sync::mpsc::channel();
            let (tx2, rx2) = std::sync::mpsc::channel();
            len_slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx1.send(res);
            });
            path_slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx2.send(res);
            });
            self.device.poll(wgpu::Maintain::Wait);
            let elapsed_ns = (start_t.elapsed().as_nanos() as u64).max(1);
            self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

            let mut result = None;
            if rx1.recv().ok().and_then(|r| r.ok()).is_some()
                && rx2.recv().ok().and_then(|r| r.ok()).is_some()
            {
                let len_data = len_slice.get_mapped_range();
                let path_len = bytemuck::cast_slice::<u8, u32>(&len_data)[0];
                drop(len_data);
                pool.staging_len_buf.unmap();

                if path_len > 0 && path_len <= max_path_len {
                    let path_data = path_slice.get_mapped_range();
                    let words = &bytemuck::cast_slice::<u8, u32>(&path_data)[..(path_len as usize * 3)];
                    let mut path = Vec::with_capacity(path_len as usize);
                    for c in words.chunks_exact(3) {
                        path.push((c[0] as usize, c[1] as usize, c[2] as usize));
                    }
                    result = Some(path);
                }
                pool.staging_path_buf.unmap();
            }

            bt_param_buf.destroy();
            wf_param_buf.destroy();
            rc_param_buf.destroy();

            result
        })?
    }

    /// Solves a batch of independent 3D wavefront routing requests with zero extra cell clearance.
    ///
    /// Convenience wrapper around [`Self::solve_batch_3d_wavefront_paths_with_clearance`] that
    /// appends `clearance_cells = 0` to each request tuple.
    ///
    /// # Arguments
    /// * `obstacle_grid` - Flat 3D obstacle net ID grid (`[layers * gh * gw]`).
    /// * `gw` - Grid width in cells.
    /// * `gh` - Grid height in cells.
    /// * `layers` - Number of copper routing layers.
    /// * `requests` - Slice of `(src, dst, net_id, via_cost)` routing requests.
    ///
    /// # Returns
    /// A `Vec<Option<Vec<(usize, usize, usize)>>>` of length `requests.len()`.
    ///
    /// # Side Effects
    /// Dispatches batch GPU wavefront solves across available CUDA or Vulkan devices.
    pub fn solve_batch_3d_wavefront_paths(
        &self,
        obstacle_grid: &[u32],
        gw: usize,
        gh: usize,
        layers: usize,
        requests: &[((usize, usize, usize), (usize, usize, usize), u32, f32)],
    ) -> Vec<Option<Vec<(usize, usize, usize)>>> {
        let reqs_with_clearance: Vec<((usize, usize, usize), (usize, usize, usize), u32, f32, i32)> =
            requests
                .iter()
                .map(|&(src, dst, net_id, via_cost)| (src, dst, net_id, via_cost, 0))
                .collect();
        self.solve_batch_3d_wavefront_paths_with_clearance(
            obstacle_grid,
            gw,
            gh,
            layers,
            &reqs_with_clearance,
        )
    }

    /// Solves a batch of independent 3D wavefront routing requests with per-request circular clearance
    /// radii `clearance_cells`.
    ///
    /// When `self.backend_kind == GpuBackendKind::CudaPtx`, distributes requests across all CUDA GPUs
    /// via [`crate::cuda::MultiGpuRouterPool`]; otherwise executes each request sequentially via
    /// [`Self::solve_3d_wavefront_path_vulkan_wgsl`].
    ///
    /// # Arguments
    /// * `obstacle_grid` - Flat 3D obstacle net ID grid (`[layers * gh * gw]`).
    /// * `gw` - Grid width in cells.
    /// * `gh` - Grid height in cells.
    /// * `layers` - Number of copper routing layers.
    /// * `requests` - Slice of `(src, dst, net_id, via_cost, clearance_cells)` routing requests.
    ///
    /// # Returns
    /// A `Vec<Option<Vec<(usize, usize, usize)>>>` of length `requests.len()`.
    ///
    /// # Side Effects
    /// Dispatches single- or multi-GPU 3D wavefront path solves and updates `compute_busy_ns`.
    pub fn solve_batch_3d_wavefront_paths_with_clearance(
        &self,
        obstacle_grid: &[u32],
        gw: usize,
        gh: usize,
        layers: usize,
        requests: &[((usize, usize, usize), (usize, usize, usize), u32, f32, i32)],
    ) -> Vec<Option<Vec<(usize, usize, usize)>>> {
        if requests.is_empty() {
            return Vec::new();
        }
        if self.backend_kind == GpuBackendKind::CudaPtx {
            let cuda_devs = crate::cuda::CudaEngine::all_devices();
            if !cuda_devs.is_empty() {
                let pool = crate::cuda::MultiGpuRouterPool {
                    cuda_engines: cuda_devs,
                    vulkan_fallback: None,
                };
                return pool.solve_batch_3d_wavefront_paths_with_clearance(
                    obstacle_grid,
                    gw,
                    gh,
                    layers,
                    requests,
                );
            }
        }
        requests
            .iter()
            .map(|&(src, dst, net_id, via_cost, clearance_cells)| {
                self.solve_3d_wavefront_path_vulkan_wgsl(
                    obstacle_grid,
                    gw,
                    gh,
                    layers,
                    src,
                    dst,
                    net_id,
                    via_cost,
                    clearance_cells,
                )
            })
            .collect()
    }

    /// Evaluates pairwise ratsnest segment crossings (`j > i`) on the GPU using a 2D Morton LBVH.
    ///
    /// Dispatches to `CudaEngine::evaluate_segment_crossings_rt_bvh` when CUDA PTX is active, or
    /// falls back to [`Self::evaluate_segment_crossings_rt_bvh_vulkan_wgsl`].
    ///
    /// # Arguments
    /// * `segs` - Slice of 2D ratsnest segments `[x1, y1, x2, y2]` in mm.
    /// * `bvh_nodes` - Pre-built Morton LBVH over `segs` (from [`build_gpu_lbvh_from_segments`]).
    ///
    /// # Returns
    /// `Some(Vec<u32>)` of length `segs.len()` where element `i` is the number of segments `j > i`
    /// that strictly cross `segs[i]`, or `None` if `segs` or `bvh_nodes` is empty.
    ///
    /// # Side Effects
    /// Uploads `segs` and `bvh_nodes` to GPU buffers, dispatches the LBVH traversal kernel, and
    /// increments `self.compute_busy_ns`.
    pub fn evaluate_segment_crossings_rt_bvh(
        &self,
        segs: &[[f32; 4]],
        bvh_nodes: &[GpuBvhNode],
    ) -> Option<Vec<u32>> {
        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                if let Some(res) = cuda.evaluate_segment_crossings_rt_bvh(segs, bvh_nodes) {
                    self.compute_busy_ns
                        .fetch_add(cuda.compute_busy_ns.load(Ordering::Relaxed), Ordering::Relaxed);
                    return Some(res);
                }
            }
        }
        self.evaluate_segment_crossings_rt_bvh_vulkan_wgsl(segs, bvh_nodes)
    }

    /// Executes the LBVH pairwise segment crossing shader (`RT_BVH_SEGMENT_CROSSINGS_WGSL`) directly
    /// on the `wgpu` Vulkan backend.
    ///
    /// # Arguments
    /// * `segs` - Slice of 2D line segments `[x1, y1, x2, y2]` in mm.
    /// * `bvh_nodes` - Flat Morton LBVH nodes over `segs`.
    ///
    /// # Returns
    /// `Some(Vec<u32>)` of length `segs.len()` containing per-segment crossing counts (`j > i`),
    /// or `None` if `segs` or `bvh_nodes` is empty or buffer mapping fails.
    ///
    /// # Side Effects
    /// Acquires `self.dispatch_lock`, allocates device and staging buffers, dispatches
    /// `self.bvh_crossings_pipeline`, blocks on `device.poll(Maintain::Wait)`, and updates `self.compute_busy_ns`.
    pub fn evaluate_segment_crossings_rt_bvh_vulkan_wgsl(
        &self,
        segs: &[[f32; 4]],
        bvh_nodes: &[GpuBvhNode],
    ) -> Option<Vec<u32>> {
        if segs.is_empty() || bvh_nodes.is_empty() {
            return None;
        }
        let n = segs.len();
        let out_bytes = (n * std::mem::size_of::<u32>()) as u64;
        let _guard = self.dispatch_lock.lock().ok()?;

        let segs_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RtBvhCrossings Segs Buffer"),
            contents: bytemuck::cast_slice(segs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bvh_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RtBvhCrossings Nodes Buffer"),
            contents: bytemuck::cast_slice(bvh_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RtBvhCrossings Out Buffer"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RtBvhCrossings Staging Buffer"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let q_params = BvhQueryParams {
            count: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RtBvhCrossings Params Buffer"),
            contents: bytemuck::bytes_of(&q_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtBvhCrossings BindGroup"),
            layout: &self.bvh_crossings_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: segs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bvh_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RtBvhCrossings CommandEncoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RtBvhCrossings ComputePass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bvh_crossings_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (n as u32 + 63) / 64;
            cpass.dispatch_workgroups(wg_x, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_bytes);

        let start_t = Instant::now();
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let elapsed_ns = start_t.elapsed().as_nanos() as u64;
        self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

        let result = if rx.recv().ok().and_then(|r| r.ok()).is_some() {
            let data = slice.get_mapped_range();
            let vec: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buf.unmap();
            Some(vec)
        } else {
            None
        };

        staging_buf.destroy();
        out_buf.destroy();
        segs_buf.destroy();
        bvh_buf.destroy();
        param_buf.destroy();

        result
    }

    /// Evaluates topological corridor cut ray crossings against ratsnest segments using the 2D Morton LBVH.
    ///
    /// Automatically dispatches to CUDA PTX when `backend_kind == GpuBackendKind::CudaPtx`,
    /// or to the Vulkan WGSL compute pipeline (`RT_BVH_CORRIDOR_CUTS_WGSL`).
    ///
    /// # Arguments
    /// * `segs` - Slice of 2D ratsnest segments `[x1, y1, x2, y2]` in mm.
    /// * `bvh_nodes` - Pre-built Morton LBVH over `segs`.
    /// * `cuts` - Slice of corridor cut query segments `[x1, y1, x2, y2]` in mm.
    ///
    /// # Returns
    /// `Some(Vec<u32>)` of length `cuts.len()` containing the number of ratsnest crossings per cut,
    /// or `None` if any input slice is empty.
    ///
    /// # Side Effects
    /// Dispatches GPU LBVH corridor cut traversal and updates `self.compute_busy_ns`.
    pub fn evaluate_corridor_cuts_rt_bvh(
        &self,
        segs: &[[f32; 4]],
        bvh_nodes: &[GpuBvhNode],
        cuts: &[[f32; 4]],
    ) -> Option<Vec<u32>> {
        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                if let Some(res) = cuda.evaluate_corridor_cuts_rt_bvh(segs, bvh_nodes, cuts) {
                    self.compute_busy_ns
                        .fetch_add(cuda.compute_busy_ns.load(Ordering::Relaxed), Ordering::Relaxed);
                    return Some(res);
                }
            }
        }
        self.evaluate_corridor_cuts_rt_bvh_vulkan_wgsl(segs, bvh_nodes, cuts)
    }

    /// Executes the LBVH corridor cut crossing shader (`RT_BVH_CORRIDOR_CUTS_WGSL`) directly on the
    /// `wgpu` Vulkan backend.
    ///
    /// # Arguments
    /// * `segs` - Slice of 2D ratsnest segments `[x1, y1, x2, y2]` in mm.
    /// * `bvh_nodes` - Flat Morton LBVH nodes over `segs`.
    /// * `cuts` - Slice of corridor cut query segments `[x1, y1, x2, y2]` in mm.
    ///
    /// # Returns
    /// `Some(Vec<u32>)` of length `cuts.len()` containing crossing counts per cut, or `None` if any
    /// input slice is empty or staging buffer readback fails.
    ///
    /// # Side Effects
    /// Acquires `self.dispatch_lock`, allocates temporary device and staging buffers, dispatches
    /// `self.bvh_corridor_cuts_pipeline`, blocks on `device.poll(Maintain::Wait)`, and updates `self.compute_busy_ns`.
    pub fn evaluate_corridor_cuts_rt_bvh_vulkan_wgsl(
        &self,
        segs: &[[f32; 4]],
        bvh_nodes: &[GpuBvhNode],
        cuts: &[[f32; 4]],
    ) -> Option<Vec<u32>> {
        if segs.is_empty() || bvh_nodes.is_empty() || cuts.is_empty() {
            return None;
        }
        let num_cuts = cuts.len();
        let out_bytes = (num_cuts * std::mem::size_of::<u32>()) as u64;
        let _guard = self.dispatch_lock.lock().ok()?;

        let segs_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RtBvhCorridorCuts Segs Buffer"),
            contents: bytemuck::cast_slice(segs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bvh_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RtBvhCorridorCuts Nodes Buffer"),
            contents: bytemuck::cast_slice(bvh_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cuts_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RtBvhCorridorCuts Cuts Buffer"),
            contents: bytemuck::cast_slice(cuts),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RtBvhCorridorCuts Out Buffer"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RtBvhCorridorCuts Staging Buffer"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let q_params = BvhQueryParams {
            count: num_cuts as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RtBvhCorridorCuts Params Buffer"),
            contents: bytemuck::bytes_of(&q_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtBvhCorridorCuts BindGroup"),
            layout: &self.bvh_corridor_cuts_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: segs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bvh_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cuts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RtBvhCorridorCuts CommandEncoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RtBvhCorridorCuts ComputePass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bvh_corridor_cuts_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (num_cuts as u32 + 63) / 64;
            cpass.dispatch_workgroups(wg_x, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_bytes);

        let start_t = Instant::now();
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let elapsed_ns = start_t.elapsed().as_nanos() as u64;
        self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

        let result = if rx.recv().ok().and_then(|r| r.ok()).is_some() {
            let data = slice.get_mapped_range();
            let vec: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buf.unmap();
            Some(vec)
        } else {
            None
        };

        staging_buf.destroy();
        out_buf.destroy();
        segs_buf.destroy();
        bvh_buf.destroy();
        cuts_buf.destroy();
        param_buf.destroy();

        result
    }

    /// Evaluates batch line-of-sight DRC clearance ray queries against an obstacle Morton LBVH on the GPU.
    ///
    /// Dispatches to `CudaEngine::evaluate_drc_rt_bvh` when CUDA PTX is active, or falls back to
    /// [`Self::evaluate_drc_rt_bvh_vulkan_wgsl`].
    ///
    /// # Arguments
    /// * `bvh_nodes` - Flat Morton LBVH constructed over board obstacle AABBs via [`build_gpu_lbvh`].
    /// * `queries` - Slice of [`GpuDrcRayQuery`] ray descriptors to test.
    ///
    /// # Returns
    /// `Some(Vec<u32>)` of length `queries.len()` containing the obstacle hit count for each query ray,
    /// or `None` if `bvh_nodes` or `queries` is empty.
    ///
    /// # Side Effects
    /// Uploads `bvh_nodes` and `queries` to the GPU, executes LBVH ray-AABB traversal, and updates `self.compute_busy_ns`.
    pub fn evaluate_drc_rt_bvh(
        &self,
        bvh_nodes: &[GpuBvhNode],
        queries: &[GpuDrcRayQuery],
    ) -> Option<Vec<u32>> {
        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                if let Some(res) = cuda.evaluate_drc_rt_bvh(bvh_nodes, queries) {
                    self.compute_busy_ns
                        .fetch_add(cuda.compute_busy_ns.load(Ordering::Relaxed), Ordering::Relaxed);
                    return Some(res);
                }
            }
        }
        self.evaluate_drc_rt_bvh_vulkan_wgsl(bvh_nodes, queries)
    }

    /// Executes the LBVH line-of-sight DRC ray query shader (`RT_BVH_LINE_OF_SIGHT_DRC_WGSL`)
    /// directly on the `wgpu` Vulkan backend.
    ///
    /// # Arguments
    /// * `bvh_nodes` - Flat Morton LBVH over obstacle AABBs.
    /// * `queries` - Slice of [`GpuDrcRayQuery`] ray descriptors.
    ///
    /// # Returns
    /// `Some(Vec<u32>)` of length `queries.len()` with per-query DRC violation counts, or `None` if
    /// `bvh_nodes` or `queries` is empty or buffer readback fails.
    ///
    /// # Side Effects
    /// Acquires `self.dispatch_lock`, allocates device and staging buffers, dispatches
    /// `self.bvh_drc_pipeline`, blocks on `device.poll(Maintain::Wait)`, and updates `self.compute_busy_ns`.
    pub fn evaluate_drc_rt_bvh_vulkan_wgsl(
        &self,
        bvh_nodes: &[GpuBvhNode],
        queries: &[GpuDrcRayQuery],
    ) -> Option<Vec<u32>> {
        if bvh_nodes.is_empty() || queries.is_empty() {
            return None;
        }
        let nq = queries.len();
        let out_bytes = (nq * std::mem::size_of::<u32>()) as u64;
        let _guard = self.dispatch_lock.lock().ok()?;

        let bvh_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RtBvhDrc Nodes Buffer"),
            contents: bytemuck::cast_slice(bvh_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let queries_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RtBvhDrc Queries Buffer"),
            contents: bytemuck::cast_slice(queries),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RtBvhDrc Out Buffer"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RtBvhDrc Staging Buffer"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let q_params = BvhQueryParams {
            count: nq as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RtBvhDrc Params Buffer"),
            contents: bytemuck::bytes_of(&q_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtBvhDrc BindGroup"),
            layout: &self.bvh_drc_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bvh_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RtBvhDrc CommandEncoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RtBvhDrc ComputePass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bvh_drc_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (nq as u32 + 63) / 64;
            cpass.dispatch_workgroups(wg_x, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_bytes);

        let start_t = Instant::now();
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let elapsed_ns = start_t.elapsed().as_nanos() as u64;
        self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

        let result = if rx.recv().ok().and_then(|r| r.ok()).is_some() {
            let data = slice.get_mapped_range();
            let vec: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buf.unmap();
            Some(vec)
        } else {
            None
        };

        staging_buf.destroy();
        out_buf.destroy();
        bvh_buf.destroy();
        queries_buf.destroy();
        param_buf.destroy();

        result
    }

    /// Runs a synthetic micro-benchmark comparing LBVH-accelerated DRC and segment crossing queries
    /// against brute-force SIMT/CPU baselines (512 obstacles, 256 queries).
    ///
    /// # Returns
    /// `Some(RtBvhBenchmarkResult)` containing microsecond timings, relative speedups, and hardware
    /// ray-query availability, or `None` if any GPU benchmark pass fails.
    ///
    /// # Side Effects
    /// Builds temporary Morton LBVHs, dispatches GPU LBVH and batch placement shaders, and updates
    /// `self.compute_busy_ns`.
    pub fn benchmark_rt_bvh_vs_simt(&self) -> Option<RtBvhBenchmarkResult> {
        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                if let Some(res) = cuda.benchmark_rt_bvh_vs_simt() {
                    return Some(res);
                }
            }
        }

        let num_obs = 512usize;
        let num_queries = 256usize;
        let mut obstacles = Vec::with_capacity(num_obs);
        for i in 0..num_obs {
            let x = (i % 32) as f32 * 3.0;
            let y = (i / 32) as f32 * 3.0;
            let layer_mask = 1u32 << (i % 4);
            let net_id = ((i % 16) + 1) as u32;
            obstacles.push(([x, y, x + 0.8, y + 0.8], layer_mask, net_id));
        }
        let bvh_nodes = build_gpu_lbvh(&obstacles);

        let mut queries = Vec::with_capacity(num_queries);
        let mut segs = Vec::with_capacity(num_queries);
        for i in 0..num_queries {
            let x1 = ((i * 7) % 90) as f32;
            let y1 = ((i * 13) % 45) as f32;
            let x2 = x1 + 6.0;
            let y2 = y1 + 4.0;
            queries.push(GpuDrcRayQuery {
                x1,
                y1,
                x2,
                y2,
                clearance: 0.25,
                layer_mask: 1u32 << (i % 4),
                net_id: ((i % 16) + 1) as u32,
                _pad: 0,
            });
            segs.push([x1, y1, x2, y2]);
        }
        let seg_bvh = build_gpu_lbvh_from_segments(&segs);

        let t0 = Instant::now();
        let _ = self.evaluate_drc_rt_bvh_vulkan_wgsl(&bvh_nodes, &queries)?;
        let bvh_drc_us = t0.elapsed().as_secs_f64() * 1e6;

        let t1 = Instant::now();
        let _ = evaluate_drc_brute_force_cpu(&obstacles, &queries);
        let simt_drc_us = t1.elapsed().as_secs_f64() * 1e6;

        let t2 = Instant::now();
        let _ = self.evaluate_segment_crossings_rt_bvh_vulkan_wgsl(&segs, &seg_bvh)?;
        let bvh_crossings_us = t2.elapsed().as_secs_f64() * 1e6;

        let dummy_boxes = vec![[0.0f32; 4]; 1];
        let bp_params = PlacementBatchParams {
            num_candidates: 1,
            num_footprints: 1,
            num_segments: segs.len() as u32,
            _pad: 0,
            board_min_x: 0.0,
            board_min_y: 0.0,
            board_max_x: 100.0,
            board_max_y: 100.0,
            wirelength_weight: 1.0,
            crossing_weight: 10.0,
            overlap_penalty: 100.0,
            boundary_penalty: 100.0,
        };
        let t3 = Instant::now();
        let _ = self.evaluate_placement_batch(&dummy_boxes, &segs, &bp_params)?;
        let simt_crossings_us = t3.elapsed().as_secs_f64() * 1e6;

        Some(RtBvhBenchmarkResult {
            bvh_drc_us,
            simt_drc_us,
            bvh_crossings_us,
            simt_crossings_us,
            drc_speedup: simt_drc_us / bvh_drc_us.max(0.01),
            crossings_speedup: simt_crossings_us / bvh_crossings_us.max(0.01),
            rt_hardware_available: self.vulkan_rt_ray_query_supported,
        })
    }

    /// Evaluates batch Design Rule Check (DRC) clearance violations for candidate segments against board obstacles.
    ///
    /// Checks each candidate in `candidates` against all foreign-net pads, tracks, and vias on
    /// matching copper layers. Dispatches to the native CUDA PTX kernel when
    /// [`GpuBackendKind::CudaPtx`] is active, falling back to
    /// [`Self::evaluate_drc_candidates_vulkan`].
    ///
    /// # Arguments
    /// * `pads` - Slice of board pad obstacles ([`GpuDrcPad`]).
    /// * `tracks` - Slice of existing routed track segments ([`GpuDrcTrack`]).
    /// * `vias` - Slice of existing plated through-hole or blind/buried vias ([`GpuDrcVia`]).
    /// * `candidates` - Slice of candidate trace segments ([`GpuDrcCandidate`]) to test.
    ///
    /// # Returns
    /// `Some(Vec<u32>)` with length `candidates.len()` where `0` means DRC-clean and `1` means a
    /// clearance violation, or `None` if GPU buffer mapping fails.
    ///
    /// # Side Effects
    /// Allocates and destroys GPU buffers, submits a compute pass, and increments
    /// `self.compute_busy_ns`.
    pub fn evaluate_drc_candidates(
        &self,
        pads: &[GpuDrcPad],
        tracks: &[GpuDrcTrack],
        vias: &[GpuDrcVia],
        candidates: &[GpuDrcCandidate],
    ) -> Option<Vec<u32>> {
        if candidates.is_empty() {
            return Some(Vec::new());
        }

        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                if let Some(res) = cuda.evaluate_drc_candidates(pads, tracks, vias, candidates) {
                    self.compute_busy_ns.fetch_add(cuda.compute_busy_ns.load(Ordering::Relaxed), Ordering::Relaxed);
                    return Some(res);
                }
            }
        }

        self.evaluate_drc_candidates_vulkan(pads, tracks, vias, candidates)
    }

    /// Evaluates batch DRC clearance violations using the Vulkan WGSL `BATCH_DRC_EVAL_WGSL` compute pipeline.
    ///
    /// Uploads `pads`, `tracks`, `vias`, and `candidates` into GPU storage buffers (substituting
    /// a zeroed dummy element when an obstacle slice is empty so `wgpu` buffer bindings remain valid),
    /// dispatches workgroups of 64 threads, and reads back the per-candidate violation flags.
    ///
    /// # Arguments
    /// * `pads` - Slice of board pad obstacles ([`GpuDrcPad`]).
    /// * `tracks` - Slice of existing routed track segments ([`GpuDrcTrack`]).
    /// * `vias` - Slice of existing vias ([`GpuDrcVia`]).
    /// * `candidates` - Slice of candidate trace segments ([`GpuDrcCandidate`]) to test.
    ///
    /// # Returns
    /// `Some(Vec<u32>)` containing `0` (clean) or `1` (violation) per candidate, or `None` if locking
    /// or staging buffer readback fails.
    ///
    /// # Side Effects
    /// Acquires `self.dispatch_lock`, allocates and destroys transient `wgpu::Buffer` objects, blocks
    /// until the GPU queue completes (`wgpu::Maintain::Wait`), and increments `self.compute_busy_ns`.
    pub fn evaluate_drc_candidates_vulkan(
        &self,
        pads: &[GpuDrcPad],
        tracks: &[GpuDrcTrack],
        vias: &[GpuDrcVia],
        candidates: &[GpuDrcCandidate],
    ) -> Option<Vec<u32>> {
        let nc = candidates.len();
        if nc == 0 {
            return Some(Vec::new());
        }

        let _guard = self.dispatch_lock.lock().ok()?;
        let out_byte_size = (nc * std::mem::size_of::<u32>()) as u64;

        let dummy_pad = [bytemuck::Zeroable::zeroed(); 1];
        let pad_slice: &[GpuDrcPad] = if pads.is_empty() { &dummy_pad } else { pads };
        let pads_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DRC Pads Buffer"),
            contents: bytemuck::cast_slice(pad_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let dummy_trk = [bytemuck::Zeroable::zeroed(); 1];
        let trk_slice: &[GpuDrcTrack] = if tracks.is_empty() { &dummy_trk } else { tracks };
        let tracks_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DRC Tracks Buffer"),
            contents: bytemuck::cast_slice(trk_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let dummy_via = [bytemuck::Zeroable::zeroed(); 1];
        let via_slice: &[GpuDrcVia] = if vias.is_empty() { &dummy_via } else { vias };
        let vias_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DRC Vias Buffer"),
            contents: bytemuck::cast_slice(via_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let cands_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DRC Candidates Buffer"),
            contents: bytemuck::cast_slice(candidates),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DRC Results Out Buffer"),
            size: out_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DRC Results Staging Buffer"),
            size: out_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = DrcBatchParams {
            num_pads: pads.len() as u32,
            num_tracks: tracks.len() as u32,
            num_vias: vias.len() as u32,
            num_candidates: nc as u32,
        };
        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DrcBatchParams Uniform"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BatchDrc BindGroup"),
            layout: &self.drc_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pads_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tracks_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vias_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cands_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BatchDrc CommandEncoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BatchDrc ComputePass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.drc_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_x = ((nc as u32) + 63) / 64;
            cpass.dispatch_workgroups(wg_x, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_byte_size);
        let start_t = Instant::now();
        self.queue.submit(Some(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let elapsed_ns = start_t.elapsed().as_nanos() as u64;
        self.compute_busy_ns.fetch_add(elapsed_ns, Ordering::Relaxed);

        let result = if rx.recv().ok().and_then(|r| r.ok()).is_some() {
            let data = slice.get_mapped_range();
            let vec: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buf.unmap();
            Some(vec)
        } else {
            None
        };

        staging_buf.destroy();
        out_buf.destroy();
        pads_buf.destroy();
        tracks_buf.destroy();
        vias_buf.destroy();
        cands_buf.destroy();
        param_buf.destroy();

        result
    }
}

/// Point-in-time snapshot and rolling 60-sample history of host CPU and GPU utilization.
///
/// Consumed by the `egui` telemetry sparklines in `src/gui.rs` and updated periodically by
/// [`SystemTelemetry`].
#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    /// Most recent instantaneous host CPU utilization percentage (`0.0..=100.0`).
    pub cpu_load_pct: f32,
    /// Most recent instantaneous GPU utilization percentage (`0.0..=100.0`).
    pub gpu_load_pct: f32,
    /// Rolling 60-sample ring buffer of historical CPU load percentages for GUI sparkline rendering.
    pub cpu_history: Vec<f32>,
    /// Rolling 60-sample ring buffer of historical GPU load percentages for GUI sparkline rendering.
    pub gpu_history: Vec<f32>,
}

impl Default for TelemetrySnapshot {
    /// Creates an initialized [`TelemetrySnapshot`] with zeroed load percentages and 60-sample histories.
    ///
    /// # Returns
    /// A default [`TelemetrySnapshot`] with 60 zeroes in `cpu_history` and `gpu_history`.
    fn default() -> Self {
        Self {
            cpu_load_pct: 0.0,
            gpu_load_pct: 0.0,
            cpu_history: vec![0.0; 60],
            gpu_history: vec![0.0; 60],
        }
    }
}

impl TelemetrySnapshot {
    /// Records a new CPU and GPU utilization sample into the snapshot and advances the 60-sample rolling history.
    ///
    /// Both input values are clamped to `[0.0, 100.0]` before being stored. If the history vectors
    /// exceed 60 elements, the oldest entry at index `0` is evicted.
    ///
    /// # Arguments
    /// * `cpu_pct` - Measured CPU utilization percentage.
    /// * `gpu_pct` - Measured GPU utilization percentage.
    ///
    /// # Side Effects
    /// Mutates `self.cpu_load_pct`, `self.gpu_load_pct`, `self.cpu_history`, and `self.gpu_history`.
    pub fn record_sample(&mut self, cpu_pct: f32, gpu_pct: f32) {
        let c_pct = cpu_pct.clamp(0.0, 100.0);
        let g_pct = gpu_pct.clamp(0.0, 100.0);
        self.cpu_load_pct = c_pct;
        self.gpu_load_pct = g_pct;
        self.cpu_history.push(c_pct);
        if self.cpu_history.len() > 60 {
            self.cpu_history.remove(0);
        }
        self.gpu_history.push(g_pct);
        if self.gpu_history.len() > 60 {
            self.gpu_history.remove(0);
        }
    }
}

/// Parses the aggregate `cpu` line from Linux `/proc/stat` into total and idle jiffies.
///
/// Sums all numeric fields on the first line for `total`, and combines column 3 (`idle`) plus
/// column 4 (`iowait`, if present) as `idle`.
///
/// # Arguments
/// * `stat_content` - Raw text content read from `/proc/stat`.
///
/// # Returns
/// `Some((total_jiffies, idle_jiffies))` if the first line has at least 4 integer columns, or
/// `None` if malformed or empty.
pub(crate) fn parse_proc_stat_totals(stat_content: &str) -> Option<(u64, u64)> {
    let first = stat_content.lines().next()?;
    let vals: Vec<u64> = first
        .split_whitespace()
        .skip(1)
        .filter_map(|s| s.parse::<u64>().ok())
        .collect();
    if vals.len() >= 4 {
        let idle = vals[3] + vals.get(4).copied().unwrap_or(0);
        let total: u64 = vals.iter().sum();
        Some((total, idle))
    } else {
        None
    }
}

/// Computes the CPU busy percentage between two consecutive `/proc/stat` jiffy samples.
///
/// # Arguments
/// * `prev_total` - Total CPU jiffies from the previous sample (`0` on the very first tick).
/// * `prev_idle` - Idle + iowait jiffies from the previous sample.
/// * `cur_total` - Total CPU jiffies from the current sample.
/// * `cur_idle` - Idle + iowait jiffies from the current sample.
///
/// # Returns
/// CPU utilization in the range `0.0..=100.0`, or `0.0` when `prev_total == 0` or `cur_total <= prev_total`.
pub(crate) fn compute_cpu_load_pct(prev_total: u64, prev_idle: u64, cur_total: u64, cur_idle: u64) -> f32 {
    let d_total = cur_total.saturating_sub(prev_total);
    let d_idle = cur_idle.saturating_sub(prev_idle);
    if prev_total > 0 && d_total > 0 {
        ((d_total.saturating_sub(d_idle)) as f32 / d_total as f32) * 100.0
    } else {
        0.0
    }
}

/// Parses one or more lines of numeric GPU utilization percentages and returns the maximum value.
///
/// Used to parse both single-line DRM sysfs `gpu_busy_percent` files and multi-GPU `nvidia-smi`
/// CSV output (`--query-gpu=utilization.gpu --format=csv,noheader,nounits`).
///
/// # Arguments
/// * `s` - Raw text containing whitespace-trimmed floating-point or integer percentages per line.
///
/// # Returns
/// `Some(max_pct)` across all valid numeric lines, or `None` if no line parses as `f32`.
pub(crate) fn parse_gpu_utilization_str(s: &str) -> Option<f32> {
    s.lines()
        .filter_map(|line| line.trim().parse::<f32>().ok())
        .fold(None, |acc, v| Some(acc.map_or(v, |prev: f32| prev.max(v))))
}

/// Background system telemetry sampler tracking live host CPU and GPU load percentages.
///
/// Spawns a detached daemon thread that wakes every 400 ms to sample `/proc/stat`, Vulkan/CUDA
/// compute duty cycle counters, and hardware GPU utilization sensors.
pub struct SystemTelemetry {
    /// Thread-safe handle to the rolling [`TelemetrySnapshot`] read by the GUI.
    pub state: Arc<Mutex<TelemetrySnapshot>>,
}

impl SystemTelemetry {
    /// Spawns the background telemetry sampling thread and returns a [`SystemTelemetry`] handle.
    ///
    /// Every 400 ms, the background thread:
    /// 1. Reads `/proc/stat` to compute host CPU utilization via `parse_proc_stat_totals` and `compute_cpu_load_pct`.
    /// 2. Computes the GPU duty-cycle percentage from the delta of `compute_busy_ns` over the elapsed wall-clock interval.
    /// 3. Queries hardware GPU sensors via `read_hw_gpu_utilization` and takes the maximum of the software duty cycle and hardware sensor reading.
    /// 4. Appends the resulting sample to `self.state`.
    ///
    /// # Arguments
    /// * `compute_busy_ns` - Optional shared atomic counter accumulating nanoseconds spent inside [`VulkanEngine`] GPU dispatches.
    ///
    /// # Returns
    /// A new [`SystemTelemetry`] instance wrapping the shared [`TelemetrySnapshot`] mutex.
    ///
    /// # Side Effects
    /// Spawns a long-running background OS thread that polls `/proc/stat`, `/sys/class/drm`, and `nvidia-smi` every 400 ms.
    pub fn spawn(compute_busy_ns: Option<Arc<AtomicU64>>) -> Self {
        let state = Arc::new(Mutex::new(TelemetrySnapshot::default()));
        let state_clone = Arc::clone(&state);

        thread::spawn(move || {
            let mut prev_total = 0u64;
            let mut prev_idle = 0u64;
            let mut prev_compute_ns = 0u64;
            let mut last_tick = Instant::now();

            loop {
                thread::sleep(Duration::from_millis(400));
                let dt_ns = last_tick.elapsed().as_nanos() as u64;
                last_tick = Instant::now();

                // 1. Sample CPU load from /proc/stat
                let mut cpu_pct = 0.0f32;
                if let Ok(stat) = fs::read_to_string("/proc/stat") {
                    if let Some((total, idle)) = parse_proc_stat_totals(&stat) {
                        cpu_pct = compute_cpu_load_pct(prev_total, prev_idle, total, idle);
                        prev_total = total;
                        prev_idle = idle;
                    }
                }

                // 2. Sample GPU load from Vulkan compute duty + system GPU sensors
                let mut gpu_pct = 0.0f32;
                if let Some(ref busy_counter) = compute_busy_ns {
                    let cur_ns = busy_counter.load(Ordering::Relaxed);
                    let d_busy = cur_ns.saturating_sub(prev_compute_ns);
                    prev_compute_ns = cur_ns;
                    if dt_ns > 0 {
                        gpu_pct = ((d_busy as f32 / dt_ns as f32) * 100.0).min(100.0);
                    }
                }

                if let Some(hw_gpu) = read_hw_gpu_utilization() {
                    gpu_pct = gpu_pct.max(hw_gpu);
                }

                if let Ok(mut guard) = state_clone.lock() {
                    guard.record_sample(cpu_pct, gpu_pct);
                }
            }
        });

        Self { state }
    }
}

// Hardware-specific sysfs (`/sys/class/drm/card*/device/gpu_busy_percent`) and `nvidia-smi`
// subprocess calls depend on which physical GPU driver is installed on the host machine;
// the underlying string parser is unit-tested deterministically via `parse_gpu_utilization_str`.
/// Reads hardware GPU utilization percentage from Linux DRM sysfs or `nvidia-smi`.
///
/// Checks `/sys/class/drm/card{0..3}/device/gpu_busy_percent` first (used by `amdgpu` and `xe`/`i915`
/// drivers) and returns the highest utilization across available cards. If no DRM sysfs node is
/// readable, invokes `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`.
///
/// # Returns
/// `Some(gpu_busy_pct)` if either DRM sysfs or `nvidia-smi` reports a valid utilization percentage,
/// or `None` if neither sensor interface is available on the host.
///
/// # Side Effects
/// Reads `/sys/class/drm/card*/device/gpu_busy_percent` files and may spawn an `nvidia-smi` child process.
fn read_hw_gpu_utilization() -> Option<f32> {
    let mut best_drm: Option<f32> = None;
    for drm_path in [
        "/sys/class/drm/card0/device/gpu_busy_percent",
        "/sys/class/drm/card1/device/gpu_busy_percent",
        "/sys/class/drm/card2/device/gpu_busy_percent",
        "/sys/class/drm/card3/device/gpu_busy_percent",
    ] {
        if let Ok(s) = fs::read_to_string(drm_path) {
            if let Some(v) = parse_gpu_utilization_str(&s) {
                best_drm = Some(best_drm.map_or(v, |prev| prev.max(v)));
            }
        }
    }
    if best_drm.is_some() {
        return best_drm;
    }

    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout);
        if let Some(v) = parse_gpu_utilization_str(&s) {
            return Some(v);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_distance_field_compute_shader() {
        let vk = match VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };
        assert!(!vk.device_name.is_empty());
        assert!(!vk.backend_name.is_empty());
        assert!(format!("{:?}", vk).contains("VulkanEngine"));

        // Invalid dimensions / buffer length must return None
        assert!(vk.compute_distance_field(&[], 0, 0, 5).is_none());
        assert!(vk.compute_distance_field(&[0u32; 10], 4, 4, 5).is_none());

        let mut obs = vec![0u32; 16 * 16];
        obs[4 * 16 + 4] = 1;
        let df = vk.compute_distance_field(&obs, 16, 16, 10).expect("Distance field output");
        assert!((df[4 * 16 + 4] - 0.0).abs() < 1e-4);
        assert!((df[4 * 16 + 8] - 4.0).abs() < 1e-3);
        assert!((df[7 * 16 + 8] - 5.0).abs() < 1e-3); // 3-4-5 right triangle
        assert!(vk.compute_busy_ns.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_vulkan_batch_placement_eval_shader() {
        let vk = match VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };
        let params = PlacementBatchParams {
            num_candidates: 2,
            num_footprints: 2,
            num_segments: 2,
            _pad: 0,
            board_min_x: 0.0,
            board_min_y: 0.0,
            board_max_x: 20.0,
            board_max_y: 20.0,
            wirelength_weight: 1.0,
            crossing_weight: 200.0,
            overlap_penalty: 25000.0,
            boundary_penalty: 50000.0,
        };
        assert!(vk.evaluate_placement_batch(&[], &[], &params).is_none());

        // Candidate 0: non-overlapping boxes inside bounds, non-crossing parallel segments (len 4 each)
        // Candidate 1: overlapping boxes + out of bounds + crossing segments
        let boxes = vec![
            [2.0, 2.0, 4.0, 4.0],
            [10.0, 10.0, 12.0, 12.0],
            [-1.0, 2.0, 4.0, 4.0],
            [3.0, 3.0, 5.0, 5.0],
        ];
        let segs = vec![
            [1.0, 1.0, 5.0, 1.0],
            [1.0, 3.0, 5.0, 3.0],
            [1.0, 1.0, 5.0, 5.0],
            [1.0, 5.0, 5.0, 1.0],
        ];
        let scores = vk
            .evaluate_placement_batch(&boxes, &segs, &params)
            .expect("Batch placement eval output");
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - 8.0).abs() < 1e-2);
        assert!(scores[1] > 75_000.0);

        let sa_params = GpuSaParams {
            num_chains: 64,
            num_footprints: 2,
            num_edges: 1,
            steps_per_pass: 4,
            seed: 42,
            _pad0: 0,
            corridor_congestion_weight: 120.0,
            flow_alignment_weight: 60.0,
            board_min_x: 0.0,
            board_min_y: 0.0,
            board_max_x: 20.0,
            board_max_y: 20.0,
            grid_step: 0.5,
            wirelength_weight: 1.0,
            crossing_weight: 200.0,
            overlap_penalty: 25000.0,
            boundary_penalty: 50000.0,
            passive_affinity_weight: 80.0,
            passive_max_dist: 3.5,
            base_temp: 5.0,
        };
        assert!(vk.run_gpu_sa_placer_chains(&[], &[], &[], &sa_params, 1, None).is_none());
        let fp_descs = vec![[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]];
        let edges = vec![[0.0, 0.5, 0.0, 0.0], [1.0, -0.5, 0.0, 0.0]];
        let mut init_poses = Vec::with_capacity(64 * 2);
        for _ in 0..64 {
            init_poses.push([4.0, 4.0, 0.0, 0.0]);
            init_poses.push([12.0, 12.0, 0.0, 0.0]);
        }
        let (best_poses, best_scores) = vk
            .run_gpu_sa_placer_chains(&fp_descs, &edges, &init_poses, &sa_params, 2, None)
            .expect("GPU SA placer chains output");
        assert_eq!(best_poses.len(), 64 * 2);
        assert_eq!(best_scores.len(), 64);
        assert!(best_scores[0][1] > 0.0);
    }

    #[test]
    fn test_proc_stat_and_gpu_utilization_parsers() {
        let stat_line = "cpu  100 20 30 800 50 0 0 0 0 0\ncpu0 50 10 15 400 25\n";
        let (total, idle) = parse_proc_stat_totals(stat_line).unwrap();
        assert_eq!(total, 1000);
        assert_eq!(idle, 850);
        assert!(parse_proc_stat_totals("cpu 1 2").is_none());
        assert!(parse_proc_stat_totals("").is_none());

        assert_eq!(compute_cpu_load_pct(0, 0, 1000, 850), 0.0);
        assert_eq!(compute_cpu_load_pct(1000, 850, 1000, 850), 0.0);
        let pct = compute_cpu_load_pct(1000, 850, 1200, 950);
        assert!((pct - 50.0).abs() < 1e-4);

        assert_eq!(parse_gpu_utilization_str(" 42 \n"), Some(42.0));
        assert_eq!(parse_gpu_utilization_str("73.5\n12.0\n"), Some(73.5));
        assert_eq!(parse_gpu_utilization_str("not_a_number"), None);
        assert_eq!(parse_gpu_utilization_str(""), None);
    }

    #[test]
    fn test_telemetry_snapshot_and_background_sampler() {
        let mut snap = TelemetrySnapshot::default();
        assert_eq!(snap.cpu_history.len(), 60);
        assert_eq!(snap.gpu_history.len(), 60);
        snap.record_sample(125.0, -5.0);
        assert_eq!(snap.cpu_load_pct, 100.0);
        assert_eq!(snap.gpu_load_pct, 0.0);
        assert_eq!(snap.cpu_history.len(), 60);
        assert_eq!(*snap.cpu_history.last().unwrap(), 100.0);

        // Exercise hardware utilization query (returns None or valid percentage depending on host)
        if let Some(hw) = read_hw_gpu_utilization() {
            assert!(hw >= 0.0);
        }

        // Exercise SystemTelemetry background thread for one tick
        let busy = Arc::new(AtomicU64::new(100_000_000));
        let telem = SystemTelemetry::spawn(Some(Arc::clone(&busy)));
        thread::sleep(Duration::from_millis(450));
        let guard = telem.state.lock().unwrap();
        assert!(guard.cpu_load_pct >= 0.0 && guard.cpu_load_pct <= 100.0);
        assert!(guard.gpu_load_pct >= 0.0 && guard.gpu_load_pct <= 100.0);
    }

    #[test]
    fn test_vulkan_multi_gpu_all_engines() {
        let vk = match VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };
        let engines = vk.all_engines();
        println!("VulkanEngine primary: {}", vk.device_name);
        println!("All active GPU engines count: {}", engines.len());
        for (i, eng) in engines.iter().enumerate() {
            println!("  Engine [{}]: {} ({:?})", i, eng.device_name, eng.backend_kind);
            let mut obs = vec![0u32; 16 * 16];
            obs[4 * 16 + 4] = 1;
            let df = eng.compute_distance_field(&obs, 16, 16, 10).expect("DF output");
            assert_eq!(df[4 * 16 + 4], 0.0);
        }
        assert!(!engines.is_empty());
        if crate::cuda::CudaEngine::all_devices().len() > 1 {
            assert!(engines.len() >= 2, "Expected at least 2 active GPU engines when multiple CUDA devices present");
            assert!(vk.device_name.contains("2 GPUs Active") || vk.device_name.contains("GPUs Active"));
        }
    }

    #[test]
    fn test_optix_driver_and_rt_capability_detection() {
        let vk = match VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };
        for eng in vk.all_engines() {
            println!(
                "VulkanEngine: {} | vulkan_rt_ray_query_supported={}",
                eng.device_name, eng.vulkan_rt_ray_query_supported
            );
        }
        if let Some(bench) = vk.benchmark_rt_bvh_vs_simt() {
            if bench.rt_hardware_available {
                assert!(bench.bvh_drc_us > 0.0);
                assert!(bench.bvh_crossings_us > 0.0);
            }
        }
    }

    #[test]
    fn test_rt_bvh_line_of_sight_drc_equivalence() {
        let vk = match VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };
        let num_obs = 520usize;
        let num_queries = 256usize;
        let mut obstacles = Vec::with_capacity(num_obs);
        for i in 0..num_obs {
            let x = ((i * 19) % 100) as f32 + (i % 5) as f32 * 0.25;
            let y = ((i * 29) % 80) as f32 + (i % 3) as f32 * 0.35;
            let w = 0.4 + (i % 4) as f32 * 0.3;
            let h = 0.4 + ((i + 1) % 4) as f32 * 0.3;
            let layer_mask = 1u32 << (i % 4);
            let net_id = ((i % 12) + 1) as u32;
            obstacles.push(([x, y, x + w, y + h], layer_mask, net_id));
        }

        let bvh_nodes = build_gpu_lbvh(&obstacles);
        assert_eq!(bvh_nodes.len(), num_obs * 2 - 1);

        let mut queries = Vec::with_capacity(num_queries);
        for q in 0..num_queries {
            let x1 = ((q * 23) % 95) as f32 + 0.13;
            let y1 = ((q * 17) % 75) as f32 + 0.17;
            let dx = ((q * 11) % 15) as f32 - 7.0;
            let dy = ((q * 7) % 15) as f32 - 7.0;
            queries.push(GpuDrcRayQuery {
                x1,
                y1,
                x2: x1 + dx,
                y2: y1 + dy,
                clearance: 0.15 + (q % 3) as f32 * 0.10,
                layer_mask: 1u32 << (q % 4),
                net_id: ((q % 12) + 1) as u32,
                _pad: 0,
            });
        }

        let cpu_ref = evaluate_drc_brute_force_cpu(&obstacles, &queries);
        let dispatch_out = vk.evaluate_drc_rt_bvh(&bvh_nodes, &queries).expect("Dispatch BVH DRC");
        let wgsl_out = vk
            .evaluate_drc_rt_bvh_vulkan_wgsl(&bvh_nodes, &queries)
            .expect("WGSL BVH DRC");

        assert_eq!(dispatch_out, cpu_ref);
        assert_eq!(wgsl_out, cpu_ref);
    }

    #[test]
    fn test_rt_bvh_segment_crossings_equivalence() {
        let vk = match VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };
        let num_segs = 300usize;
        let mut segs = Vec::with_capacity(num_segs);
        for i in 0..num_segs {
            let x1 = ((i * 31) % 100) as f32 + (i as f32) * 0.013;
            let y1 = ((i * 47) % 100) as f32 + (i as f32) * 0.017;
            let x2 = ((i * 59 + 17) % 100) as f32 + (i as f32) * 0.019;
            let y2 = ((i * 79 + 23) % 100) as f32 + (i as f32) * 0.011;
            segs.push([x1, y1, x2, y2]);
        }

        let bvh_nodes = build_gpu_lbvh_from_segments(&segs);
        let cpu_crossings = evaluate_segment_crossings_brute_force_cpu(&segs);
        let dispatch_crossings = vk
            .evaluate_segment_crossings_rt_bvh(&segs, &bvh_nodes)
            .expect("Dispatch BVH crossings");
        let wgsl_crossings = vk
            .evaluate_segment_crossings_rt_bvh_vulkan_wgsl(&segs, &bvh_nodes)
            .expect("WGSL BVH crossings");

        assert_eq!(dispatch_crossings, cpu_crossings);
        assert_eq!(wgsl_crossings, cpu_crossings);
    }

    #[test]
    fn test_rt_bvh_corridor_cuts_cuda_and_vulkan_equivalence() {
        let vk = match VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };
        let num_segs = 320usize;
        let num_cuts = 180usize;
        let mut segs = Vec::with_capacity(num_segs);
        for i in 0..num_segs {
            let x1 = ((i * 31) % 100) as f32 + (i as f32) * 0.013;
            let y1 = ((i * 47) % 100) as f32 + (i as f32) * 0.017;
            let x2 = ((i * 59 + 17) % 100) as f32 + (i as f32) * 0.019;
            let y2 = ((i * 79 + 23) % 100) as f32 + (i as f32) * 0.011;
            segs.push([x1, y1, x2, y2]);
        }

        let mut cuts = Vec::with_capacity(num_cuts);
        for j in 0..num_cuts {
            let x1 = ((j * 23 + 7) % 95) as f32 + (j as f32) * 0.021;
            let y1 = ((j * 41 + 11) % 95) as f32 + (j as f32) * 0.015;
            let x2 = ((j * 67 + 29) % 95) as f32 + (j as f32) * 0.023;
            let y2 = ((j * 83 + 37) % 95) as f32 + (j as f32) * 0.027;
            cuts.push([x1, y1, x2, y2]);
        }

        let bvh_nodes = build_gpu_lbvh_from_segments(&segs);
        let cpu_brute = evaluate_corridor_cuts_brute_force_cpu(&segs, &cuts);
        let cpu_lbvh = evaluate_corridor_cuts_lbvh_cpu(&segs, &bvh_nodes, &cuts);
        let dispatch_cuts = vk
            .evaluate_corridor_cuts_rt_bvh(&segs, &bvh_nodes, &cuts)
            .expect("Dispatch BVH corridor cuts");
        let wgsl_cuts = vk
            .evaluate_corridor_cuts_rt_bvh_vulkan_wgsl(&segs, &bvh_nodes, &cuts)
            .expect("WGSL BVH corridor cuts");

        assert_eq!(cpu_lbvh, cpu_brute);
        assert_eq!(dispatch_cuts, cpu_brute);
        assert_eq!(wgsl_cuts, cpu_brute);
    }

    #[test]
    fn test_gpu_3d_wavefront_backtrack_path_valid_and_obstacle_free() {
        let vk = match VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };
        let gw = 18usize;
        let gh = 18usize;
        let layers = 2usize;
        let layer_stride = gw * gh;
        let mut obs = vec![0u32; gw * gh * layers];

        // Wall on both layers at x = 9, y = 0..13 (foreign net 7)
        for l in 0..layers {
            for y in 0..13 {
                obs[l * layer_stride + y * gw + 9] = 7;
            }
        }

        let src = (3usize, 4usize, 0usize);
        let dst = (14usize, 4usize, 0usize);

        let path_wgsl = vk
            .solve_3d_wavefront_path_vulkan_wgsl(&obs, gw, gh, layers, src, dst, 1, 3.0, 0)
            .expect("Vulkan WGSL 3D wavefront backtracker must find valid path");
        let path_dispatch = vk
            .solve_3d_wavefront_path(&obs, gw, gh, layers, src, dst, 1, 3.0)
            .expect("Dispatch 3D wavefront backtracker must find valid path");

        for path in [&path_wgsl, &path_dispatch] {
            assert_eq!(path.first().copied(), Some(src));
            assert_eq!(path.last().copied(), Some(dst));
            for w in path.windows(2) {
                let (x0, y0, l0) = w[0];
                let (x1, y1, l1) = w[1];
                let dx = (x1 as isize - x0 as isize).abs();
                let dy = (y1 as isize - y0 as isize).abs();
                let dl = (l1 as isize - l0 as isize).abs();
                assert!(
                    (dl == 0 && dx <= 1 && dy <= 1 && (dx + dy > 0))
                        || (dl > 0 && dx == 0 && dy == 0),
                    "Invalid step {:?} -> {:?}",
                    w[0],
                    w[1]
                );
            }
            for &(gx, gy, l) in path {
                let cell_obs = obs[l * layer_stride + gy * gw + gx];
                assert!(
                    cell_obs == 0 || cell_obs == 1,
                    "Path stepped onto foreign obstacle {} at ({}, {}, {})",
                    cell_obs,
                    gx,
                    gy,
                    l
                );
            }
        }
        assert_eq!(path_wgsl.len(), path_dispatch.len());
    }

    #[test]
    fn test_vulkan_gpu_sat_courtyard_overlap_exact_equivalence() {
        use crate::board::{Point, Polygon};

        let vk = match VulkanEngine::new() {
            Some(v) => v,
            None => return,
        };

        // Scenario 1: Rotated 45-degree squares
        let poly_a_local = Polygon::new(vec![
            Point::new(-1.0, -1.0),
            Point::new(1.0, -1.0),
            Point::new(1.0, 1.0),
            Point::new(-1.0, 1.0),
        ]);
        let poly_b_local = poly_a_local.clone();

        let poly_a_world = poly_a_local.transform(Point::new(5.0, 5.0), 45.0);
        let poly_b_world_sep = poly_b_local.transform(Point::new(7.0, 7.0), 45.0);
        let poly_b_world_int = poly_b_local.transform(Point::new(6.0, 6.0), 45.0);

        // Verify CPU SAT baseline
        assert!(poly_a_world.aabb().overlaps(&poly_b_world_sep.aabb()), "Broadphase AABB must overlap for test validity");
        assert!(!poly_a_world.intersects_convex(&poly_b_world_sep), "CPU SAT must find separation along 45-deg edge normals");
        assert!(poly_a_world.intersects_convex(&poly_b_world_int), "CPU SAT must detect collision when moved closer");

        // Construct GPU courtyard buffers
        let fp_descs = vec![[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]];
        let edges = vec![];

        let courtyards = GpuCourtyardBuffers {
            fp_poly_ranges: vec![
                [0, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            courtyard_polys: vec![
                [0, 4, 0, 0],
                [4, 4, 1, 0],
            ],
            courtyard_vertices: vec![
                [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
                [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
            ],
        };

        let sa_params = GpuSaParams {
            num_chains: 64,
            num_footprints: 2,
            num_edges: 0,
            steps_per_pass: 0,
            seed: 12345,
            _pad0: 0,
            corridor_congestion_weight: 120.0,
            flow_alignment_weight: 60.0,
            board_min_x: 0.0,
            board_min_y: 0.0,
            board_max_x: 20.0,
            board_max_y: 20.0,
            grid_step: 0.5,
            wirelength_weight: 1.0,
            crossing_weight: 200.0,
            overlap_penalty: 25000.0,
            boundary_penalty: 50000.0,
            passive_affinity_weight: 80.0,
            passive_max_dist: 3.5,
            base_temp: 5.0,
        };

        // Test Separated rotated squares on GPU
        let mut poses_sep = Vec::with_capacity(64 * 2);
        for _ in 0..64 {
            poses_sep.push([5.0, 5.0, 45.0, 0.0]);
            poses_sep.push([7.0, 7.0, 45.0, 0.0]);
        }
        let (_, scores_sep) = vk
            .run_gpu_sa_placer_chains(&fp_descs, &edges, &poses_sep, &sa_params, 1, Some(&courtyards))
            .expect("GPU SA placer chains output");
        assert_eq!(scores_sep[0][2], 0.0, "GPU exact SAT must report 0 overlaps for separated 45-deg squares despite AABB overlap");

        // Test Intersecting rotated squares on GPU
        let mut poses_int = Vec::with_capacity(64 * 2);
        for _ in 0..64 {
            poses_int.push([5.0, 5.0, 45.0, 0.0]);
            poses_int.push([6.0, 6.0, 45.0, 0.0]);
        }
        let (_, scores_int) = vk
            .run_gpu_sa_placer_chains(&fp_descs, &edges, &poses_int, &sa_params, 1, Some(&courtyards))
            .expect("GPU SA placer chains output");
        assert_eq!(scores_int[0][2], 1.0, "GPU exact SAT must report 1 overlap for intersecting 45-deg squares");

        // Scenario 2: Multi-polygon L-shape footprint
        let l_poly0 = Polygon::new(vec![Point::new(-2.0, -2.0), Point::new(2.0, -2.0), Point::new(2.0, 0.0), Point::new(-2.0, 0.0)]);
        let l_poly1 = Polygon::new(vec![Point::new(-2.0, 0.0), Point::new(0.0, 0.0), Point::new(0.0, 2.0), Point::new(-2.0, 2.0)]);
        let box_sep = Polygon::new(vec![Point::new(0.6, 0.6), Point::new(1.4, 0.6), Point::new(1.4, 1.4), Point::new(0.6, 1.4)]);
        let box_int = Polygon::new(vec![Point::new(-1.0, 0.6), Point::new(-0.2, 0.6), Point::new(-0.2, 1.4), Point::new(-1.0, 1.4)]);

        assert!(!l_poly0.intersects_convex(&box_sep) && !l_poly1.intersects_convex(&box_sep), "CPU: box in empty corner must not collide");
        assert!(l_poly1.intersects_convex(&box_int), "CPU: box overlapping arm must collide");

        let l_courtyards = GpuCourtyardBuffers {
            fp_poly_ranges: vec![
                [0, 2, 0, 0],
                [2, 1, 0, 0],
            ],
            courtyard_polys: vec![
                [0, 4, 0, 0],
                [4, 4, 0, 0],
                [8, 4, 1, 0],
            ],
            courtyard_vertices: vec![
                [-2.0, -2.0], [2.0, -2.0], [2.0, 0.0], [-2.0, 0.0],
                [-2.0, 0.0], [0.0, 0.0], [0.0, 2.0], [-2.0, 2.0],
                [-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4],
            ],
        };

        let l_fp_descs = vec![[2.0, 2.0, 0.0, 0.0], [0.4, 0.4, 0.0, 0.0]];
        let mut l_poses_sep = Vec::with_capacity(64 * 2);
        for _ in 0..64 {
            l_poses_sep.push([10.0, 10.0, 0.0, 0.0]);
            l_poses_sep.push([11.0, 11.0, 0.0, 0.0]);
        }
        let (_, l_scores_sep) = vk
            .run_gpu_sa_placer_chains(&l_fp_descs, &edges, &l_poses_sep, &sa_params, 1, Some(&l_courtyards))
            .expect("Multi-polygon GPU test");
        assert_eq!(l_scores_sep[0][2], 0.0, "Multi-polygon GPU SAT must not report overlap for component placed in L-corner concavity");

        let mut l_poses_int = Vec::with_capacity(64 * 2);
        for _ in 0..64 {
            l_poses_int.push([10.0, 10.0, 0.0, 0.0]);
            l_poses_int.push([9.0, 11.0, 0.0, 0.0]);
        }
        let (_, l_scores_int) = vk
            .run_gpu_sa_placer_chains(&l_fp_descs, &edges, &l_poses_int, &sa_params, 1, Some(&l_courtyards))
            .expect("Multi-polygon GPU test");
        assert_eq!(l_scores_int[0][2], 1.0, "Multi-polygon GPU SAT must report overlap when hitting L-arm");

        // Scenario 3: SMD-only layer isolation
        let smd_courtyards = GpuCourtyardBuffers {
            fp_poly_ranges: vec![
                [0, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            courtyard_polys: vec![
                [0, 4, 0, 0],
                [4, 4, 1, 0],
            ],
            courtyard_vertices: vec![
                [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
                [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
            ],
        };
        let mut smd_poses = Vec::with_capacity(64 * 2);
        for _ in 0..64 {
            smd_poses.push([10.0, 10.0, 0.0, 0.0]);
            smd_poses.push([10.0, 10.0, 0.0, 1.0]);
        }
        let (_, smd_scores) = vk
            .run_gpu_sa_placer_chains(&fp_descs, &edges, &smd_poses, &sa_params, 1, Some(&smd_courtyards))
            .expect("SMD layer test");
        assert_eq!(smd_scores[0][2], 0.0, "SMD components on opposite layers must not collide");

        // If FP 0 is Through-Hole (is_smd_only = 0), it penetrates both layers and MUST collide!
        let th_courtyards = GpuCourtyardBuffers {
            fp_poly_ranges: vec![
                [0, 1, 0, 0],
                [1, 1, 1, 1],
            ],
            courtyard_polys: smd_courtyards.courtyard_polys.clone(),
            courtyard_vertices: smd_courtyards.courtyard_vertices.clone(),
        };
        let (_, th_scores) = vk
            .run_gpu_sa_placer_chains(&fp_descs, &edges, &smd_poses, &sa_params, 1, Some(&th_courtyards))
            .expect("Through-Hole layer test");
        assert_eq!(th_scores[0][2], 1.0, "Through-Hole component must collide with bottom-layer SMD component");
    }

    #[test]
    fn test_separable_distance_field_exact_equivalence() {
        let w = 40u32;
        let h = 32u32;
        let layers = 2u32;
        let radius = 9i32;
        let layer_cells = (w * h) as usize;
        let mut obs = vec![0u32; layer_cells * layers as usize];
        // Populate varied obstacle pattern across both layers
        for l in 0..layers as usize {
            for y in 0..h as usize {
                for x in 0..w as usize {
                    if (x == 5 && y == 7)
                        || (x == 19 + l * 3 && (10..=15).contains(&y))
                        || (y == 24 && (28..=33).contains(&x))
                        || (x == 0 && y == 0 && l == 1)
                    {
                        obs[l * layer_cells + y * (w as usize) + x] = (l as u32) + 1;
                    }
                }
            }
        }

        // Exact 2D window reference on CPU
        let mut ref_df = vec![0.0f32; obs.len()];
        for l in 0..layers as usize {
            let base = l * layer_cells;
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    let idx = base + (y as usize) * (w as usize) + (x as usize);
                    if obs[idx] != 0 {
                        ref_df[idx] = 0.0;
                        continue;
                    }
                    let mut min_d2 = (radius * radius + 1) as f32;
                    for dy in -radius..=radius {
                        let ny = y + dy;
                        if ny < 0 || ny >= h as i32 {
                            continue;
                        }
                        for dx in -radius..=radius {
                            let nx = x + dx;
                            if nx < 0 || nx >= w as i32 {
                                continue;
                            }
                            let nidx = base + (ny as usize) * (w as usize) + (nx as usize);
                            if obs[nidx] != 0 {
                                let d2 = (dx * dx + dy * dy) as f32;
                                if d2 < min_d2 {
                                    min_d2 = d2;
                                }
                            }
                        }
                    }
                    ref_df[idx] = min_d2.sqrt();
                }
            }
        }

        if let Some(vk) = VulkanEngine::new() {
            let vk_wgsl_df = vk
                .compute_distance_field_vulkan_wgsl(&obs, w, h, layers, radius)
                .expect("Vulkan separable distance field failed");
            assert_eq!(vk_wgsl_df.len(), ref_df.len());
            for (i, (&got, &exp)) in vk_wgsl_df.iter().zip(ref_df.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-5,
                    "Vulkan WGSL separable DF mismatch at cell {}: got {}, expected {}",
                    i,
                    got,
                    exp
                );
            }

            if let Some(cuda) = &vk.cuda_engine {
                let cuda_sep = cuda
                    .compute_distance_field_separable(&obs[..layer_cells], w, h, radius)
                    .expect("CUDA separable 2D distance field failed");
                for (i, (&got, &exp)) in cuda_sep.iter().zip(ref_df[..layer_cells].iter()).enumerate() {
                    assert!(
                        (got - exp).abs() < 1e-5,
                        "CUDA separable 2D DF mismatch at cell {}: got {}, expected {}",
                        i,
                        got,
                        exp
                    );
                }

                let cuda_layers = cuda
                    .compute_distance_field_layers(&obs, w, h, layers, radius)
                    .expect("CUDA separable 3D layers distance field failed");
                for (i, (&got, &exp)) in cuda_layers.iter().zip(ref_df.iter()).enumerate() {
                    assert!(
                        (got - exp).abs() < 1e-5,
                        "CUDA separable 3D DF mismatch at cell {}: got {}, expected {}",
                        i,
                        got,
                        exp
                    );
                }
            }
        }
    }

    #[test]
    fn test_wavefront_buffer_pool_reuse_and_lbvh_cache() {
        // 1. Verify parallel + cached LBVH exact equivalence and cache hits
        let mut prims = Vec::with_capacity(600);
        for i in 0..600usize {
            let x = (i % 30) as f32 * 2.5;
            let y = (i / 30) as f32 * 2.5;
            let layer_mask = 1u32 << (i % 4);
            let net_id = ((i % 12) + 1) as u32;
            prims.push(([x, y, x + 1.0, y + 0.8], layer_mask, net_id));
        }

        let (_, hits_before) = lbvh_cache_stats();
        let bvh_first = build_gpu_lbvh(&prims);
        let bvh_second = build_gpu_lbvh(&prims);
        let (_, hits_after) = lbvh_cache_stats();
        assert_eq!(bvh_first.len(), prims.len() * 2 - 1);
        assert_eq!(bvh_first, bvh_second, "Cached parallel LBVH must match initial build bit-for-bit");
        assert!(hits_after > hits_before, "Global LBVH cache must record a cache hit on identical primitive slice");

        // Verify DRC queries on the parallel-built LBVH match brute-force CPU
        let mut queries = Vec::with_capacity(32);
        for i in 0..32usize {
            let x1 = (i * 3) as f32;
            let y1 = (i * 2) as f32;
            queries.push(GpuDrcRayQuery {
                x1,
                y1,
                x2: x1 + 8.0,
                y2: y1 + 5.0,
                clearance: 0.2,
                layer_mask: 1u32 << (i % 4),
                net_id: ((i % 12) + 1) as u32,
                _pad: 0,
            });
        }
        let cpu_drc = evaluate_drc_brute_force_cpu(&prims, &queries);

        if let Some(vk) = VulkanEngine::new() {
            let n1 = vk.build_cached_lbvh(&prims);
            let hits_0 = vk.lbvh_cache_hits.load(Ordering::Relaxed);
            let n2 = vk.build_cached_lbvh(&prims);
            let hits_1 = vk.lbvh_cache_hits.load(Ordering::Relaxed);
            assert_eq!(n1, n2);
            assert_eq!(hits_1, hits_0 + 1, "VulkanEngine::build_cached_lbvh must reuse cached LBVH");

            let gpu_drc = vk
                .evaluate_drc_rt_bvh(&n1, &queries)
                .expect("GPU RT BVH DRC failed");
            assert_eq!(gpu_drc, cpu_drc, "Parallel LBVH GPU DRC must match CPU brute-force reference");

            // 2. Verify persistent wavefront device buffer pool reuse on Vulkan WGSL
            let gw = 16usize;
            let gh = 16usize;
            let layers = 2usize;
            let total = gw * gh * layers;
            let mut obs = vec![0u32; total];
            obs[5 * gw + 5] = 99;

            let p1 = vk
                .solve_3d_wavefront_path_vulkan_wgsl(&obs, gw, gh, layers, (1, 1, 0), (12, 12, 1), 1, 2.0, 1)
                .expect("First Vulkan wavefront path failed");
            let (alloc_1, reuse_1) = {
                let guard = vk.wavefront_buffers.lock().unwrap();
                let pool = guard.as_ref().expect("Wavefront buffer pool should be populated");
                (pool.alloc_count, pool.reuse_count)
            };
            assert_eq!(alloc_1, 1);

            let p2 = vk
                .solve_3d_wavefront_path_vulkan_wgsl(&obs, gw, gh, layers, (2, 2, 0), (14, 10, 1), 1, 2.0, 1)
                .expect("Second Vulkan wavefront path failed");
            let (alloc_2, reuse_2) = {
                let guard = vk.wavefront_buffers.lock().unwrap();
                let pool = guard.as_ref().expect("Wavefront buffer pool should be populated");
                (pool.alloc_count, pool.reuse_count)
            };
            assert_eq!(alloc_2, 1, "Wavefront buffer pool must not reallocate for same total_cells");
            assert_eq!(reuse_2, reuse_1 + 1, "Wavefront buffer pool must increment reuse_count");
            assert_eq!(p1.first().copied(), Some((1, 1, 0)));
            assert_eq!(p1.last().copied(), Some((12, 12, 1)));
            assert_eq!(p2.first().copied(), Some((2, 2, 0)));
            assert_eq!(p2.last().copied(), Some((14, 10, 1)));

            // Verify CUDA wavefront buffer pool reuse if CUDA is active
            if let Some(cuda) = &vk.cuda_engine {
                let cp1 = cuda
                    .solve_3d_wavefront_path_with_clearance(&obs, gw, gh, layers, (1, 1, 0), (12, 12, 1), 1, 2.0, 1)
                    .expect("First CUDA wavefront path failed");
                let (c_alloc_1, c_reuse_1) = {
                    let guard = cuda.wavefront_buffers.lock().unwrap();
                    let pool = guard.as_ref().expect("CUDA wavefront buffer pool should be populated");
                    (pool.alloc_count, pool.reuse_count)
                };
                let cp2 = cuda
                    .solve_3d_wavefront_path_with_clearance(&obs, gw, gh, layers, (2, 2, 0), (14, 10, 1), 1, 2.0, 1)
                    .expect("Second CUDA wavefront path failed");
                let (c_alloc_2, c_reuse_2) = {
                    let guard = cuda.wavefront_buffers.lock().unwrap();
                    let pool = guard.as_ref().expect("CUDA wavefront buffer pool should be populated");
                    (pool.alloc_count, pool.reuse_count)
                };
                assert_eq!(c_alloc_2, c_alloc_1, "CUDA wavefront buffer pool must not reallocate for same total_cells");
                assert_eq!(c_reuse_2, c_reuse_1 + 1, "CUDA wavefront buffer pool must increment reuse_count");
                assert_eq!(cp1, p1);
                assert_eq!(cp2, p2);
            }
        }
    }
}



