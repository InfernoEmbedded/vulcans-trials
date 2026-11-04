// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const DISTANCE_FIELD_WGSL: &str = r#"
struct GridParams {
    width: u32,
    height: u32,
    radius: i32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> obstacles: array<u32>;
@group(0) @binding(1) var<storage, read_write> dist_out: array<f32>;
@group(0) @binding(2) var<uniform> params: GridParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let w = i32(params.width);
    let h = i32(params.height);
    if (x >= w || y >= h) {
        return;
    }

    let idx = u32(y * w + x);
    if (obstacles[idx] != 0u) {
        dist_out[idx] = 0.0;
        return;
    }

    let r = params.radius;
    var min_d2: f32 = f32(r * r + 1);
    for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
        let ny = y + dy;
        if (ny < 0 || ny >= h) {
            continue;
        }
        for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
            let nx = x + dx;
            if (nx < 0 || nx >= w) {
                continue;
            }
            let nidx = u32(ny * w + nx);
            if (obstacles[nidx] != 0u) {
                let d2 = f32(dx * dx + dy * dy);
                if (d2 < min_d2) {
                    min_d2 = d2;
                }
            }
        }
    }
    dist_out[idx] = sqrt(min_d2);
}
"#;

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

const GPU_SA_PLACER_WGSL: &str = r#"
struct GpuSaParams {
    num_chains: u32,
    num_footprints: u32,
    num_edges: u32,
    steps_per_pass: u32,
    seed: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

fn fp_aabb(c_base: u32, fp_idx: u32) -> vec4<f32> {
    let pose = chain_poses[c_base + fp_idx];
    let desc = fp_descs[fp_idx];
    let rad = pose.z * 0.0174532925;
    let c = abs(cos(rad));
    let s = abs(sin(rad));
    let hw = desc.x * c + desc.y * s;
    let hh = desc.x * s + desc.y * c;
    return vec4<f32>(pose.x - hw, pose.y - hh, pose.x + hw, pose.y + hh);
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
                overlaps = overlaps + 1.0;
                let ox = max(min(bi.z, bj.z) - max(bi.x, bj.x), 0.1);
                let oy = max(min(bi.w, bj.w) - max(bi.y, bj.y), 0.1);
                overlap_cost = overlap_cost + params.overlap_penalty * (1.0 + ox * oy);
            }
        }
    }

    var wl: f32 = 0.0;
    var crossings: f32 = 0.0;
    var affinity_cost: f32 = 0.0;
    for (var e1: u32 = 0u; e1 < ne; e1 = e1 + 1u) {
        let s1 = get_edge_seg(c_base, e1);
        let dx = s1.z - s1.x;
        let dy = s1.w - s1.y;
        let d = sqrt(dx * dx + dy * dy);
        let is_tie = edges[e1 * 2u].w > 0.5;
        if (is_tie) {
            let is_decoupling = edges[e1 * 2u].w > 1.5;
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
                if (edges[e2 * 2u].w < 0.5) {
                    let s2 = get_edge_seg(c_base, e2);
                    if (segs_cross_sa(s1, s2)) {
                        crossings = crossings + 1.0;
                    }
                }
            }
        }
    }

    let total_cost = wl * params.wirelength_weight
        + crossings * params.crossing_weight
        + overlap_cost
        + boundary_cost
        + affinity_cost;
    return vec2<f32>(total_cost, overlaps);
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
    var rng = params.seed ^ (c * 747796405u + 2891336453u) ^ u32(score.w * 1664525.0);
    _ = pcg_next(&rng);

    if (score.y <= 0.0) {
        let init_ev = evaluate_chain_state(c_base, nf, ne);
        score = vec4<f32>(init_ev.x, init_ev.x, init_ev.y, 1.0);
        for (var i: u32 = 0u; i < nf; i = i + 1u) {
            chain_poses[best_base + i] = chain_poses[c_base + i];
        }
    } else {
        score.w = score.w + 1.0;
    }

    let temp_ratio = f32(c & 63u) / 63.0;
    let temp = max(params.base_temp * pow(35.0, temp_ratio), 0.2);

    for (var step: u32 = 0u; step < params.steps_per_pass; step = step + 1u) {
        var idx: u32 = pcg_next(&rng) % nf;
        for (var tries: u32 = 0u; tries < 8u; tries = tries + 1u) {
            if (fp_descs[idx].z < 0.5) {
                break;
            }
            idx = (idx + 1u) % nf;
        }
        if (fp_descs[idx].z > 0.5) {
            break;
        }

        let old_pose = chain_poses[c_base + idx];
        var new_pose = old_pose;
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

        chain_poses[c_base + idx] = new_pose;
        let ev = evaluate_chain_state(c_base, nf, ne);
        let new_cost = ev.x;
        let new_ov = ev.y;
        let delta = new_cost - score.x;

        let accept = (new_ov < score.z) ||
            (new_ov == score.z && (delta <= 0.0 || rand_f32(&rng) < exp(-delta / temp)));

        if (accept) {
            score.x = new_cost;
            score.z = new_ov;
            if (new_ov == 0.0 && new_cost < score.y) {
                score.y = new_cost;
                for (var i: u32 = 0u; i < nf; i = i + 1u) {
                    chain_poses[best_base + i] = chain_poses[c_base + i];
                }
            }
        } else {
            chain_poses[c_base + idx] = old_pose;
        }
    }

    chain_scores[c] = score;
}
"#;

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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GridParams {
    width: u32,
    height: u32,
    radius: i32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRouteCostParams {
    pub width: u32,
    pub height: u32,
    pub num_layers: u32,
    pub net_id: u32,
    pub clearance_cells: i32,
    pub present_congestion_factor: f32,
    pub obstacle_clearance_weight: f32,
    pub plane_layer_mask: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WavefrontUniformParams {
    width: u32,
    height: u32,
    num_layers: u32,
    via_cost: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlacementBatchParams {
    pub num_candidates: u32,
    pub num_footprints: u32,
    pub num_segments: u32,
    pub _pad: u32,
    pub board_min_x: f32,
    pub board_min_y: f32,
    pub board_max_x: f32,
    pub board_max_y: f32,
    pub wirelength_weight: f32,
    pub crossing_weight: f32,
    pub overlap_penalty: f32,
    pub boundary_penalty: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSaParams {
    pub num_chains: u32,
    pub num_footprints: u32,
    pub num_edges: u32,
    pub steps_per_pass: u32,
    pub seed: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub board_min_x: f32,
    pub board_min_y: f32,
    pub board_max_x: f32,
    pub board_max_y: f32,
    pub grid_step: f32,
    pub wirelength_weight: f32,
    pub crossing_weight: f32,
    pub overlap_penalty: f32,
    pub boundary_penalty: f32,
    pub passive_affinity_weight: f32,
    pub passive_max_dist: f32,
    pub base_temp: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendKind {
    Vulkan,
    CudaPtx,
}

pub struct VulkanEngine {
    pub device_name: String,
    pub backend_name: String,
    pub driver_info: String,
    pub backend_kind: GpuBackendKind,
    pub cuda_engine: Option<Arc<crate::cuda::CudaEngine>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    placement_pipeline: wgpu::ComputePipeline,
    placement_bgl: wgpu::BindGroupLayout,
    sa_pipeline: wgpu::ComputePipeline,
    sa_bgl: wgpu::BindGroupLayout,
    route_cost_pipeline: wgpu::ComputePipeline,
    route_cost_bgl: wgpu::BindGroupLayout,
    wavefront_pipeline: wgpu::ComputePipeline,
    wavefront_bgl: wgpu::BindGroupLayout,
    dispatch_lock: Mutex<()>,
    pub compute_busy_ns: Arc<AtomicU64>,
    pub extra_devices: Vec<Arc<VulkanEngine>>,
}

impl std::fmt::Debug for VulkanEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanEngine")
            .field("device_name", &self.device_name)
            .field("backend_name", &self.backend_name)
            .field("backend_kind", &self.backend_kind)
            .finish()
    }
}

impl VulkanEngine {
    pub fn new() -> Option<Arc<Self>> {
        pollster::block_on(Self::init_async(false)).map(Arc::new)
    }

    pub fn new_vulkan_only() -> Option<Arc<Self>> {
        pollster::block_on(Self::init_async(true)).map(Arc::new)
    }

    pub fn is_cuda_active(&self) -> bool {
        self.backend_kind == GpuBackendKind::CudaPtx
    }

    pub fn all_engines(self: &Arc<Self>) -> Vec<Arc<Self>> {
        let mut list = Vec::with_capacity(1 + self.extra_devices.len());
        list.push(Arc::clone(self));
        for dev in &self.extra_devices {
            list.push(Arc::clone(dev));
        }
        list
    }

    async fn init_async(force_vulkan: bool) -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let mut adapters: Vec<wgpu::Adapter> = instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .into_iter()
            .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
            .collect();

        if adapters.is_empty() {
            let fallback_inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            adapters = fallback_inst
                .enumerate_adapters(wgpu::Backends::all())
                .into_iter()
                .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
                .collect();
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

    async fn from_adapter(
        adapter: wgpu::Adapter,
        compute_busy_ns: Arc<AtomicU64>,
        extra_devices: Vec<Arc<VulkanEngine>>,
        force_vulkan_arg: bool,
        cuda_engine_arg: Option<Arc<crate::cuda::CudaEngine>>,
    ) -> Option<Self> {
        let info = adapter.get_info();
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
            label: Some("DistanceField ComputePipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
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
            cuda_engine,
            device,
            queue,
            pipeline,
            bind_group_layout,
            placement_pipeline,
            placement_bgl,
            sa_pipeline,
            sa_bgl,
            route_cost_pipeline,
            route_cost_bgl,
            wavefront_pipeline,
            wavefront_bgl,
            dispatch_lock: Mutex::new(()),
            compute_busy_ns,
            extra_devices,
        })
    }

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

        let _guard = self.dispatch_lock.lock().ok()?;
        let byte_size = (total * std::mem::size_of::<f32>()) as u64;

        let obs_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Obstacles Buffer"),
            contents: bytemuck::cast_slice(obstacles),
            usage: wgpu::BufferUsages::STORAGE,
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
            _pad: 0,
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
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DistanceField CommandEncoder"),
        });

        let wg_x = (width + 15) / 16;
        let wg_y = (height + 15) / 16;
        let passes = if total >= 4096 && radius > 12 {
            64
        } else if total >= 4096 {
            8
        } else {
            1
        };
        for _ in 0..passes {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DistanceField ComputePass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
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
        param_buf.destroy();

        result
    }

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

        let mut out = Vec::with_capacity(total);
        for l in 0..(num_layers as usize) {
            let offset = l * layer_cells;
            let df = self.compute_distance_field(&obstacles_flat[offset..offset + layer_cells], width, height, radius)?;
            out.extend_from_slice(&df);
        }
        Some(out)
    }


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

    pub fn run_gpu_sa_placer_chains(
        &self,
        fp_descs: &[[f32; 4]],
        edges: &[[f32; 4]],
        init_poses: &[[f32; 4]],
        params: &GpuSaParams,
        num_passes: u32,
    ) -> Option<(Vec<[f32; 4]>, Vec<[f32; 4]>)> {
        let nc = params.num_chains as usize;
        let nf = params.num_footprints as usize;
        let ne = params.num_edges as usize;
        if nc == 0 || nf == 0 || fp_descs.len() != nf || edges.len() != ne * 2 || init_poses.len() != nc * nf {
            return None;
        }

        if self.backend_kind == GpuBackendKind::CudaPtx {
            if let Some(cuda) = &self.cuda_engine {
                if let Some(res) = cuda.run_gpu_sa_placer_chains(fp_descs, edges, init_poses, params, num_passes) {
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

        result
    }

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

        let costs_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wavefront3D CellCosts Buffer"),
            contents: bytemuck::cast_slice(cell_costs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let wf_buf_a = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wavefront3D Buffer A"),
            contents: bytemuck::cast_slice(&init_wf),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let wf_buf_b = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wavefront3D Buffer B"),
            contents: bytemuck::cast_slice(&init_wf),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Wavefront3D Staging Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
                    resource: costs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wf_buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wf_buf_b.as_entire_binding(),
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
                    resource: costs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wf_buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wf_buf_a.as_entire_binding(),
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

        let final_buf = if iterations % 2 == 1 { &wf_buf_b } else { &wf_buf_a };
        encoder.copy_buffer_to_buffer(final_buf, 0, &staging_buf, 0, byte_size);

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
        wf_buf_a.destroy();
        wf_buf_b.destroy();
        costs_buf.destroy();
        param_buf.destroy();

        result
    }
}

#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    pub cpu_load_pct: f32,
    pub gpu_load_pct: f32,
    pub cpu_history: Vec<f32>,
    pub gpu_history: Vec<f32>,
}

impl Default for TelemetrySnapshot {
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

pub(crate) fn compute_cpu_load_pct(prev_total: u64, prev_idle: u64, cur_total: u64, cur_idle: u64) -> f32 {
    let d_total = cur_total.saturating_sub(prev_total);
    let d_idle = cur_idle.saturating_sub(prev_idle);
    if prev_total > 0 && d_total > 0 {
        ((d_total.saturating_sub(d_idle)) as f32 / d_total as f32) * 100.0
    } else {
        0.0
    }
}

pub(crate) fn parse_gpu_utilization_str(s: &str) -> Option<f32> {
    s.lines()
        .filter_map(|line| line.trim().parse::<f32>().ok())
        .fold(None, |acc, v| Some(acc.map_or(v, |prev: f32| prev.max(v))))
}

pub struct SystemTelemetry {
    pub state: Arc<Mutex<TelemetrySnapshot>>,
}

impl SystemTelemetry {
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
        let vk = VulkanEngine::new().expect("VulkanEngine must initialize");
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
        let vk = VulkanEngine::new().expect("VulkanEngine must initialize");
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
            _pad1: 0,
            _pad2: 0,
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
        assert!(vk.run_gpu_sa_placer_chains(&[], &[], &[], &sa_params, 1).is_none());
        let fp_descs = vec![[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]];
        let edges = vec![[0.0, 0.5, 0.0, 0.0], [1.0, -0.5, 0.0, 0.0]];
        let mut init_poses = Vec::with_capacity(64 * 2);
        for _ in 0..64 {
            init_poses.push([4.0, 4.0, 0.0, 0.0]);
            init_poses.push([12.0, 12.0, 0.0, 0.0]);
        }
        let (best_poses, best_scores) = vk
            .run_gpu_sa_placer_chains(&fp_descs, &edges, &init_poses, &sa_params, 2)
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
        let vk = VulkanEngine::new().expect("VulkanEngine must initialize");
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
}


