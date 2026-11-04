// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

#include <cuda_runtime.h>
#include <math.h>

extern "C" {

struct GridParams {
    unsigned int width;
    unsigned int height;
    int radius;
    unsigned int _pad;
};

struct PlacementBatchParams {
    unsigned int num_candidates;
    unsigned int num_footprints;
    unsigned int num_segments;
    unsigned int _pad;
    float board_min_x;
    float board_min_y;
    float board_max_x;
    float board_max_y;
    float wirelength_weight;
    float crossing_weight;
    float overlap_penalty;
    float boundary_penalty;
};

struct GpuSaParams {
    unsigned int num_chains;
    unsigned int num_footprints;
    unsigned int num_edges;
    unsigned int steps_per_pass;
    unsigned int seed;
    unsigned int _pad0;
    unsigned int _pad1;
    unsigned int _pad2;
    float board_min_x;
    float board_min_y;
    float board_max_x;
    float board_max_y;
    float grid_step;
    float wirelength_weight;
    float crossing_weight;
    float overlap_penalty;
    float boundary_penalty;
    float passive_affinity_weight;
    float passive_max_dist;
    float base_temp;
};

// -----------------------------------------------------------------------------
// 1. Distance Field Kernel
// -----------------------------------------------------------------------------

__global__ void distance_field_kernel(
    const unsigned int* __restrict__ obstacles,
    float* __restrict__ dist_out,
    unsigned int width,
    unsigned int height,
    int radius
) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int w = (int)width;
    int h = (int)height;

    if (x >= w || y >= h) {
        return;
    }

    unsigned int idx = (unsigned int)(y * w + x);
    if (obstacles[idx] != 0u) {
        dist_out[idx] = 0.0f;
        return;
    }

    int r = radius;
    float min_d2 = (float)(r * r + 1);

    for (int dy = -r; dy <= r; ++dy) {
        int ny = y + dy;
        if (ny < 0 || ny >= h) {
            continue;
        }
        for (int dx = -r; dx <= r; ++dx) {
            int nx = x + dx;
            if (nx < 0 || nx >= w) {
                continue;
            }
            unsigned int nidx = (unsigned int)(ny * w + nx);
            if (obstacles[nidx] != 0u) {
                float d2 = (float)(dx * dx + dy * dy);
                if (d2 < min_d2) {
                    min_d2 = d2;
                }
            }
        }
    }
    dist_out[idx] = sqrtf(min_d2);
}

// Optimized Distance Field Kernel with 2D Shared Memory Tiling
__global__ void distance_field_tiled_kernel(
    const unsigned int* __restrict__ obstacles,
    float* __restrict__ dist_out,
    unsigned int width,
    unsigned int height,
    int radius
) {
    extern __shared__ unsigned char smem_raw[];
    unsigned int* s_tile = (unsigned int*)smem_raw;

    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;

    int w = (int)width;
    int h = (int)height;
    int r = radius;

    int tile_w = blockDim.x + 2 * r;
    int tile_h = blockDim.y + 2 * r;
    int total_smem_elems = tile_w * tile_h;

    for (int i = tid; i < total_smem_elems; i += num_threads) {
        int smem_y = i / tile_w;
        int smem_x = i % tile_w;
        int gx = bx - r + smem_x;
        int gy = by - r + smem_y;
        if (gx >= 0 && gx < w && gy >= 0 && gy < h) {
            s_tile[i] = obstacles[gy * w + gx];
        } else {
            s_tile[i] = 0u;
        }
    }

    __syncthreads();

    int gx = bx + tx;
    int gy = by + ty;
    if (gx >= w || gy >= h) {
        return;
    }

    unsigned int idx = (unsigned int)(gy * w + gx);
    int center_smem_idx = (ty + r) * tile_w + (tx + r);

    if (s_tile[center_smem_idx] != 0u) {
        dist_out[idx] = 0.0f;
        return;
    }

    float min_d2 = (float)(r * r + 1);

    for (int dy = -r; dy <= r; ++dy) {
        int smem_row = (ty + r + dy) * tile_w;
        for (int dx = -r; dx <= r; ++dx) {
            if (s_tile[smem_row + (tx + r + dx)] != 0u) {
                float d2 = (float)(dx * dx + dy * dy);
                if (d2 < min_d2) {
                    min_d2 = d2;
                }
            }
        }
    }

    dist_out[idx] = sqrtf(min_d2);
}

// -----------------------------------------------------------------------------
// 2. Batch Placement Evaluation Kernel
// -----------------------------------------------------------------------------

__device__ __forceinline__ float ccw(float ax, float ay, float bx, float by, float cx, float cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

__device__ __forceinline__ bool segments_cross(float4 s1, float4 s2) {
    const float eps = 1e-3f;
    if ((fabsf(s1.x - s2.x) < eps && fabsf(s1.y - s2.y) < eps) ||
        (fabsf(s1.x - s2.z) < eps && fabsf(s1.y - s2.w) < eps) ||
        (fabsf(s1.z - s2.x) < eps && fabsf(s1.w - s2.y) < eps) ||
        (fabsf(s1.z - s2.z) < eps && fabsf(s1.w - s2.w) < eps)) {
        return false;
    }
    float d1 = ccw(s1.x, s1.y, s1.z, s1.w, s2.x, s2.y);
    float d2 = ccw(s1.x, s1.y, s1.z, s1.w, s2.z, s2.w);
    float d3 = ccw(s2.x, s2.y, s2.z, s2.w, s1.x, s1.y);
    float d4 = ccw(s2.x, s2.y, s2.z, s2.w, s1.z, s1.w);
    return (((d1 > 0.0f && d2 < 0.0f) || (d1 < 0.0f && d2 > 0.0f)) &&
            ((d3 > 0.0f && d4 < 0.0f) || (d3 < 0.0f && d4 > 0.0f)));
}

__global__ void batch_placement_eval_kernel(
    const float4* __restrict__ boxes,
    const float4* __restrict__ segs,
    float* __restrict__ costs_out,
    PlacementBatchParams params
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= params.num_candidates) {
        return;
    }

    const unsigned int nf = params.num_footprints;
    const unsigned int ns = params.num_segments;
    const unsigned int b_base = c * nf;
    const unsigned int s_base = c * ns;

    const float b_min_x = params.board_min_x;
    const float b_min_y = params.board_min_y;
    const float b_max_x = params.board_max_x;
    const float b_max_y = params.board_max_y;
    const float wl_weight = params.wirelength_weight;
    const float cr_weight = params.crossing_weight;
    const float ov_penalty = params.overlap_penalty;
    const float bd_penalty = params.boundary_penalty;

    float wl = 0.0f;
    unsigned int crossings = 0u;
    for (unsigned int i = 0u; i < ns; ++i) {
        float4 s1 = segs[s_base + i];
        float dx = s1.z - s1.x;
        float dy = s1.w - s1.y;
        wl += sqrtf(dx * dx + dy * dy);
        for (unsigned int j = i + 1u; j < ns; ++j) {
            if (segments_cross(s1, segs[s_base + j])) {
                crossings += 1u;
            }
        }
    }

    float overlap_cost = 0.0f;
    float boundary_cost = 0.0f;
    for (unsigned int i = 0u; i < nf; ++i) {
        float4 bi = boxes[b_base + i];
        float out_x = fmaxf(b_min_x - bi.x, 0.0f) + fmaxf(bi.z - b_max_x, 0.0f);
        float out_y = fmaxf(b_min_y - bi.y, 0.0f) + fmaxf(bi.w - b_max_y, 0.0f);
        if (out_x > 0.0f || out_y > 0.0f) {
            boundary_cost += bd_penalty * (1.0f + out_x + out_y);
        }
        for (unsigned int j = i + 1u; j < nf; ++j) {
            float4 bj = boxes[b_base + j];
            if (bi.z > bj.x && bi.x < bj.z && bi.w > bj.y && bi.y < bj.w) {
                float ox = fmaxf(fminf(bi.z, bj.z) - fmaxf(bi.x, bj.x), 0.1f);
                float oy = fmaxf(fminf(bi.w, bj.w) - fmaxf(bi.y, bj.y), 0.1f);
                overlap_cost += ov_penalty * (1.0f + ox * oy);
            }
        }
    }

    costs_out[c] = wl * wl_weight
        + (float)crossings * cr_weight
        + overlap_cost
        + boundary_cost;
}

// -----------------------------------------------------------------------------
// 3. GPU Simulated Annealing MCMC Placer Kernel
// -----------------------------------------------------------------------------

__device__ __forceinline__ unsigned int pcg_next(unsigned int* state) {
    unsigned int old = *state;
    *state = old * 747796405u + 2891336453u;
    unsigned int word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ float rand_f32(unsigned int* state) {
    return (float)(pcg_next(state) & 0x00FFFFFFu) / 16777216.0f;
}

__device__ __forceinline__ float snap_grid(float v, float step) {
    if (step <= 0.0f) {
        return v;
    }
    return roundf(v / step) * step;
}

__device__ __forceinline__ float ccw_sa(float ax, float ay, float bx, float by, float cx, float cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

__device__ __forceinline__ bool segs_cross_sa(float4 s1, float4 s2) {
    const float eps = 1e-3f;
    if ((fabsf(s1.x - s2.x) < eps && fabsf(s1.y - s2.y) < eps) ||
        (fabsf(s1.x - s2.z) < eps && fabsf(s1.y - s2.w) < eps) ||
        (fabsf(s1.z - s2.x) < eps && fabsf(s1.w - s2.y) < eps) ||
        (fabsf(s1.z - s2.z) < eps && fabsf(s1.w - s2.w) < eps)) {
        return false;
    }
    float d1 = ccw_sa(s1.x, s1.y, s1.z, s1.w, s2.x, s2.y);
    float d2 = ccw_sa(s1.x, s1.y, s1.z, s1.w, s2.z, s2.w);
    float d3 = ccw_sa(s2.x, s2.y, s2.z, s2.w, s1.x, s1.y);
    float d4 = ccw_sa(s2.x, s2.y, s2.z, s2.w, s1.z, s1.w);
    return (((d1 > 0.0f && d2 < 0.0f) || (d1 < 0.0f && d2 > 0.0f)) &&
            ((d3 > 0.0f && d4 < 0.0f) || (d3 < 0.0f && d4 > 0.0f)));
}

__device__ __forceinline__ float4 get_edge_seg(
    const float4* __restrict__ chain_poses,
    const float4* __restrict__ edges,
    unsigned int c_base,
    unsigned int e_idx
) {
    float4 ea = edges[e_idx * 2u];
    float4 eb = edges[e_idx * 2u + 1u];
    float4 fpa = chain_poses[c_base + (unsigned int)ea.x];
    float4 fpb = chain_poses[c_base + (unsigned int)eb.x];

    float rad_a = fpa.z * 0.0174532925f;
    float cos_a = cosf(rad_a);
    float sin_a = sinf(rad_a);
    float ax = fpa.x + (ea.y * cos_a + ea.z * sin_a);
    float ay = fpa.y + (-ea.y * sin_a + ea.z * cos_a);

    float rad_b = fpb.z * 0.0174532925f;
    float cos_b = cosf(rad_b);
    float sin_b = sinf(rad_b);
    float bx = fpb.x + (eb.y * cos_b + eb.z * sin_b);
    float by = fpb.y + (-eb.y * sin_b + eb.z * cos_b);

    return make_float4(ax, ay, bx, by);
}

__device__ __forceinline__ float4 fp_aabb(
    const float4* __restrict__ chain_poses,
    const float4* __restrict__ fp_descs,
    unsigned int c_base,
    unsigned int fp_idx
) {
    float4 pose = chain_poses[c_base + fp_idx];
    float4 desc = fp_descs[fp_idx];
    float rad = pose.z * 0.0174532925f;
    float c = fabsf(cosf(rad));
    float s = fabsf(sinf(rad));
    float hw = desc.x * c + desc.y * s;
    float hh = desc.x * s + desc.y * c;
    return make_float4(pose.x - hw, pose.y - hh, pose.x + hw, pose.y + hh);
}

__device__ float2 evaluate_chain_state_fast(
    const float4* __restrict__ chain_poses,
    const float4* __restrict__ fp_descs,
    const float4* __restrict__ edges,
    unsigned int c_base,
    unsigned int nf,
    unsigned int ne,
    float b_min_x,
    float b_min_y,
    float b_max_x,
    float b_max_y,
    float wl_weight,
    float cr_weight,
    float ov_penalty,
    float bd_penalty,
    float pass_aff_weight,
    float pass_max_d
) {
    float overlap_cost = 0.0f;
    float boundary_cost = 0.0f;
    float overlaps = 0.0f;

    for (unsigned int i = 0u; i < nf; ++i) {
        float4 bi = fp_aabb(chain_poses, fp_descs, c_base, i);
        bool i_locked = fp_descs[i].z > 0.5f;
        if (!i_locked) {
            float out_x = fmaxf(b_min_x - bi.x, 0.0f) + fmaxf(bi.z - b_max_x, 0.0f);
            float out_y = fmaxf(b_min_y - bi.y, 0.0f) + fmaxf(bi.w - b_max_y, 0.0f);
            if (out_x > 0.0f || out_y > 0.0f) {
                boundary_cost += bd_penalty * (1.0f + out_x + out_y);
            }
        }
        for (unsigned int j = i + 1u; j < nf; ++j) {
            if (i_locked && fp_descs[j].z > 0.5f) {
                continue;
            }
            float4 bj = fp_aabb(chain_poses, fp_descs, c_base, j);
            if (bi.z > bj.x && bi.x < bj.z && bi.w > bj.y && bi.y < bj.w) {
                overlaps += 1.0f;
                float ox = fmaxf(fminf(bi.z, bj.z) - fmaxf(bi.x, bj.x), 0.1f);
                float oy = fmaxf(fminf(bi.w, bj.w) - fmaxf(bi.y, bj.y), 0.1f);
                overlap_cost += ov_penalty * (1.0f + ox * oy);
            }
        }
    }

    float wl = 0.0f;
    float crossings = 0.0f;
    float affinity_cost = 0.0f;
    for (unsigned int e1 = 0u; e1 < ne; ++e1) {
        float4 s1 = get_edge_seg(chain_poses, edges, c_base, e1);
        float dx = s1.z - s1.x;
        float dy = s1.w - s1.y;
        float d = sqrtf(dx * dx + dy * dy);
        bool is_tie = edges[e1 * 2u].w > 0.5f;
        if (is_tie) {
            bool is_decoupling = edges[e1 * 2u].w > 1.5f;
            float max_d = is_decoupling ? 2.0f : pass_max_d;
            float weight = is_decoupling ? (pass_aff_weight * 3.5f) : pass_aff_weight;
            if (d > max_d) {
                float ex = d - max_d;
                affinity_cost += weight * (d + ex * ex * 3.0f);
            } else {
                affinity_cost += weight * 0.25f * d;
            }
        } else {
            wl += d;
            for (unsigned int e2 = e1 + 1u; e2 < ne; ++e2) {
                if (edges[e2 * 2u].w < 0.5f) {
                    float4 s2 = get_edge_seg(chain_poses, edges, c_base, e2);
                    if (segs_cross_sa(s1, s2)) {
                        crossings += 1.0f;
                    }
                }
            }
        }
    }

    float total_cost = wl * wl_weight
        + crossings * cr_weight
        + overlap_cost
        + boundary_cost
        + affinity_cost;
    return make_float2(total_cost, overlaps);
}

__global__ void gpu_sa_placer_kernel(
    const float4* __restrict__ fp_descs,
    const float4* __restrict__ edges,
    float4* __restrict__ chain_poses,
    float4* __restrict__ chain_scores,
    GpuSaParams params,
    unsigned int num_passes
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= params.num_chains) {
        return;
    }

    const unsigned int nf = params.num_footprints;
    const unsigned int ne = params.num_edges;
    const unsigned int c_base = c * nf;
    const unsigned int best_base = (params.num_chains + c) * nf;

    const float b_min_x = params.board_min_x;
    const float b_min_y = params.board_min_y;
    const float b_max_x = params.board_max_x;
    const float b_max_y = params.board_max_y;
    const float wl_weight = params.wirelength_weight;
    const float cr_weight = params.crossing_weight;
    const float ov_penalty = params.overlap_penalty;
    const float bd_penalty = params.boundary_penalty;
    const float pass_aff_weight = params.passive_affinity_weight;
    const float pass_max_d = params.passive_max_dist;

    float4 score = chain_scores[c];
    unsigned int rng = params.seed ^ (c * 747796405u + 2891336453u) ^ (unsigned int)(score.w * 1664525.0f);
    pcg_next(&rng);

    if (score.y <= 0.0f) {
        float2 init_ev = evaluate_chain_state_fast(
            chain_poses, fp_descs, edges, c_base, nf, ne,
            b_min_x, b_min_y, b_max_x, b_max_y,
            wl_weight, cr_weight, ov_penalty, bd_penalty,
            pass_aff_weight, pass_max_d
        );
        score = make_float4(init_ev.x, init_ev.x, init_ev.y, 1.0f);
        for (unsigned int i = 0u; i < nf; ++i) {
            chain_poses[best_base + i] = chain_poses[c_base + i];
        }
    } else {
        score.w += 1.0f;
    }

    float temp_ratio = (float)(c & 63u) / 63.0f;
    float temp = fmaxf(params.base_temp * powf(35.0f, temp_ratio), 0.2f);

    for (unsigned int p = 0; p < num_passes; ++p) {
        for (unsigned int step = 0u; step < params.steps_per_pass; ++step) {
            unsigned int idx = pcg_next(&rng) % nf;
            for (unsigned int tries = 0u; tries < 8u; ++tries) {
                if (fp_descs[idx].z < 0.5f) {
                    break;
                }
                idx = (idx + 1u) % nf;
            }
            if (fp_descs[idx].z > 0.5f) {
                break;
            }

            float4 old_pose = chain_poses[c_base + idx];
            float4 new_pose = old_pose;
            float move_r = rand_f32(&rng);
            if (move_r < 0.72f) {
                float shift = fmaxf(temp * 0.15f, params.grid_step * 2.0f);
                float dx = (rand_f32(&rng) * 2.0f - 1.0f) * shift;
                float dy = (rand_f32(&rng) * 2.0f - 1.0f) * shift;
                new_pose.x = snap_grid(
                    fminf(fmaxf(old_pose.x + dx, b_min_x + 1.2f), b_max_x - 1.2f),
                    params.grid_step
                );
                new_pose.y = snap_grid(
                    fminf(fmaxf(old_pose.y + dy, b_min_y + 1.2f), b_max_y - 1.2f),
                    params.grid_step
                );
            } else {
                float rot_step = (float)((pcg_next(&rng) % 3u) + 1u) * 90.0f;
                new_pose.z = fmodf(old_pose.z + rot_step, 360.0f);
            }

            chain_poses[c_base + idx] = new_pose;
            float2 ev = evaluate_chain_state_fast(
                chain_poses, fp_descs, edges, c_base, nf, ne,
                b_min_x, b_min_y, b_max_x, b_max_y,
                wl_weight, cr_weight, ov_penalty, bd_penalty,
                pass_aff_weight, pass_max_d
            );
            float new_cost = ev.x;
            float new_ov = ev.y;
            float delta = new_cost - score.x;

            bool accept = (new_ov < score.z) ||
                (new_ov == score.z && (delta <= 0.0f || rand_f32(&rng) < expf(-delta / temp)));

            if (accept) {
                score.x = new_cost;
                score.z = new_ov;
                if (new_ov == 0.0f && new_cost < score.y) {
                    score.y = new_cost;
                    for (unsigned int i = 0u; i < nf; ++i) {
                        chain_poses[best_base + i] = chain_poses[c_base + i];
                    }
                }
            } else {
                chain_poses[c_base + idx] = old_pose;
            }
        }
    }

    chain_scores[c] = score;
}

// -----------------------------------------------------------------------------
// 4. Batch DRC Clearance & Candidate Via/Segment Evaluation Kernel
// -----------------------------------------------------------------------------

struct DrcBatchParams {
    unsigned int num_pads;
    unsigned int num_tracks;
    unsigned int num_vias;
    unsigned int num_candidates;
};

struct GpuDrcPad {
    float x;
    float y;
    float hw;
    float hh;
    float rot_rad;
    float clearance;
    unsigned int layer_mask;
    unsigned int net_id;
};

struct GpuDrcTrack {
    float x1;
    float y1;
    float x2;
    float y2;
    float half_width;
    float clearance;
    unsigned int layer;
    unsigned int net_id;
};

struct GpuDrcVia {
    float x;
    float y;
    float radius;
    float clearance;
    unsigned int net_id;
    unsigned int _pad0;
    unsigned int _pad1;
    unsigned int _pad2;
};

struct GpuDrcCandidate {
    float x1;
    float y1;
    float x2;
    float y2;
    float radius_or_half_width;
    float clearance;
    unsigned int layer;
    unsigned int net_id;
    unsigned int kind;
    unsigned int ignore_same_net_pads;
    unsigned int _pad0;
    unsigned int _pad1;
};

__device__ __forceinline__ float drc_dist_point_to_seg(
    float px, float py,
    float x1, float y1,
    float x2, float y2
) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-6f) {
        float ex = px - x1;
        float ey = py - y1;
        return sqrtf(ex * ex + ey * ey);
    }
    float t = fminf(fmaxf(((px - x1) * dx + (py - y1) * dy) / len_sq, 0.0f), 1.0f);
    float ex = px - (x1 + t * dx);
    float ey = py - (y1 + t * dy);
    return sqrtf(ex * ex + ey * ey);
}

__device__ __forceinline__ bool drc_segs_intersect(
    float ax, float ay, float bx, float by,
    float cx, float cy, float dx, float dy
) {
    float d1 = (dy - cy) * (ax - cx) - (dx - cx) * (ay - cy);
    float d2 = (dy - cy) * (bx - cx) - (dx - cx) * (by - cy);
    float d3 = (by - ay) * (cx - ax) - (bx - ax) * (cy - ay);
    float d4 = (by - ay) * (dx - ax) - (bx - ax) * (dy - ay);
    return (((d1 > 1e-7f && d2 < -1e-7f) || (d1 < -1e-7f && d2 > 1e-7f)) &&
            ((d3 > 1e-7f && d4 < -1e-7f) || (d3 < -1e-7f && d4 > 1e-7f)));
}

__device__ __forceinline__ float drc_dist_seg_to_seg(
    float ax, float ay, float bx, float by,
    float cx, float cy, float dx, float dy
) {
    if (drc_segs_intersect(ax, ay, bx, by, cx, cy, dx, dy)) {
        return 0.0f;
    }
    float d1 = drc_dist_point_to_seg(ax, ay, cx, cy, dx, dy);
    float d2 = drc_dist_point_to_seg(bx, by, cx, cy, dx, dy);
    float d3 = drc_dist_point_to_seg(cx, cy, ax, ay, bx, by);
    float d4 = drc_dist_point_to_seg(dx, dy, ax, ay, bx, by);
    return fminf(fminf(d1, d2), fminf(d3, d4));
}

__device__ __forceinline__ float drc_dist_point_to_pad(
    float px, float py,
    GpuDrcPad pad
) {
    unsigned int shape = pad.layer_mask >> 16u;
    float rx = px - pad.x;
    float ry = py - pad.y;
    if (shape == 1u) {
        float r = fmaxf(pad.hw, pad.hh);
        return fmaxf(sqrtf(rx * rx + ry * ry) - r, 0.0f);
    }
    float c = cosf(-pad.rot_rad);
    float s = sinf(-pad.rot_rad);
    float lx = rx * c - ry * s;
    float ly = rx * s + ry * c;
    if (shape == 2u) {
        float r = fminf(pad.hw, pad.hh);
        if (pad.hw >= pad.hh) {
            float cap_dx = pad.hw - r;
            float cx = fminf(fmaxf(lx, -cap_dx), cap_dx);
            float dx = lx - cx;
            return fmaxf(sqrtf(dx * dx + ly * ly) - r, 0.0f);
        } else {
            float cap_dy = pad.hh - r;
            float cy = fminf(fmaxf(ly, -cap_dy), cap_dy);
            float dy = ly - cy;
            return fmaxf(sqrtf(lx * lx + dy * dy) - r, 0.0f);
        }
    }
    float dx = fmaxf(fabsf(lx) - pad.hw, 0.0f);
    float dy = fmaxf(fabsf(ly) - pad.hh, 0.0f);
    return sqrtf(dx * dx + dy * dy);
}

__device__ __forceinline__ float drc_dist_seg_to_pad(
    float x1, float y1, float x2, float y2,
    GpuDrcPad pad
) {
    if ((pad.layer_mask >> 16u) == 1u) {
        float r = fmaxf(pad.hw, pad.hh);
        return fmaxf(drc_dist_point_to_seg(pad.x, pad.y, x1, y1, x2, y2) - r, 0.0f);
    }
    float c = cosf(-pad.rot_rad);
    float s = sinf(-pad.rot_rad);
    float rx1 = x1 - pad.x;
    float ry1 = y1 - pad.y;
    float lsx = rx1 * c - ry1 * s;
    float lsy = rx1 * s + ry1 * c;
    float rx2 = x2 - pad.x;
    float ry2 = y2 - pad.y;
    float lex = rx2 * c - ry2 * s;
    float ley = rx2 * s + ry2 * c;
    float hw = pad.hw;
    float hh = pad.hh;

    if (fabsf(lsx) <= hw && fabsf(lsy) <= hh) {
        return 0.0f;
    }
    if (fabsf(lex) <= hw && fabsf(ley) <= hh) {
        return 0.0f;
    }

    if (drc_segs_intersect(lsx, lsy, lex, ley, -hw, -hh, hw, -hh) ||
        drc_segs_intersect(lsx, lsy, lex, ley, hw, -hh, hw, hh) ||
        drc_segs_intersect(lsx, lsy, lex, ley, hw, hh, -hw, hh) ||
        drc_segs_intersect(lsx, lsy, lex, ley, -hw, hh, -hw, -hh)) {
        return 0.0f;
    }

    float dx1 = fmaxf(fabsf(lsx) - hw, 0.0f);
    float dy1 = fmaxf(fabsf(lsy) - hh, 0.0f);
    float d1 = sqrtf(dx1 * dx1 + dy1 * dy1);

    float dx2 = fmaxf(fabsf(lex) - hw, 0.0f);
    float dy2 = fmaxf(fabsf(ley) - hh, 0.0f);
    float d2 = sqrtf(dx2 * dx2 + dy2 * dy2);

    float d3 = drc_dist_point_to_seg(-hw, -hh, lsx, lsy, lex, ley);
    float d4 = drc_dist_point_to_seg(hw, -hh, lsx, lsy, lex, ley);
    float d5 = drc_dist_point_to_seg(hw, hh, lsx, lsy, lex, ley);
    float d6 = drc_dist_point_to_seg(-hw, hh, lsx, lsy, lex, ley);

    return fminf(fminf(fminf(d1, d2), fminf(d3, d4)), fminf(d5, d6));
}

__global__ void batch_drc_eval_kernel(
    const GpuDrcPad* __restrict__ pads,
    const GpuDrcTrack* __restrict__ tracks,
    const GpuDrcVia* __restrict__ vias,
    const GpuDrcCandidate* __restrict__ candidates,
    unsigned int* __restrict__ results_out,
    DrcBatchParams params
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= params.num_candidates) {
        return;
    }

    GpuDrcCandidate cand = candidates[c];
    const float eps = 1e-4f;

    if (cand.kind == 0u) {
        // Candidate Via at (cand.x1, cand.y1) checking all layers
        for (unsigned int i = 0u; i < params.num_pads; ++i) {
            GpuDrcPad p = pads[i];
            if (p.net_id != cand.net_id) {
                float req_dist = cand.radius_or_half_width + fmaxf(cand.clearance, p.clearance);
                float d = drc_dist_point_to_pad(cand.x1, cand.y1, p);
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            } else if (cand.ignore_same_net_pads == 0u && (p.layer_mask & 0xFFFFu) != 0xFFFFu) {
                float d = drc_dist_point_to_pad(cand.x1, cand.y1, p);
                if (d < cand.radius_or_half_width - eps) {
                    results_out[c] = 0u;
                    return;
                }
            }
        }

        for (unsigned int i = 0u; i < params.num_tracks; ++i) {
            GpuDrcTrack t = tracks[i];
            if (t.net_id != cand.net_id) {
                float req_dist = cand.radius_or_half_width + fmaxf(cand.clearance, t.clearance) + t.half_width;
                float d = drc_dist_point_to_seg(cand.x1, cand.y1, t.x1, t.y1, t.x2, t.y2);
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            }
        }

        for (unsigned int i = 0u; i < params.num_vias; ++i) {
            GpuDrcVia v = vias[i];
            float dx = cand.x1 - v.x;
            float dy = cand.y1 - v.y;
            float d = sqrtf(dx * dx + dy * dy);
            if (d < eps) {
                continue;
            }
            if (cand._pad0 != 0u) {
                float idx = cand.x2 - v.x;
                float idy = cand.y2 - v.y;
                if (sqrtf(idx * idx + idy * idy) < eps) {
                    continue;
                }
            }
            if (v.net_id != cand.net_id) {
                float req_dist = cand.radius_or_half_width + fmaxf(cand.clearance, v.clearance) + v.radius;
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            } else {
                float req_dist = (cand.radius_or_half_width <= 0.18f && v.radius <= 0.18f) ? 0.25f : 0.55f;
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            }
        }
    } else {
        // Candidate Track Segment (cand.x1, cand.y1) -> (cand.x2, cand.y2) on cand.layer
        unsigned int layer_bit = (cand.layer < 32u) ? (1u << cand.layer) : 0u;
        for (unsigned int i = 0u; i < params.num_pads; ++i) {
            GpuDrcPad p = pads[i];
            if ((p.layer_mask & layer_bit) != 0u && p.net_id != cand.net_id) {
                float req_dist = cand.radius_or_half_width + fmaxf(cand.clearance, p.clearance);
                float d = drc_dist_seg_to_pad(cand.x1, cand.y1, cand.x2, cand.y2, p);
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            }
        }

        for (unsigned int i = 0u; i < params.num_tracks; ++i) {
            GpuDrcTrack t = tracks[i];
            if (t.layer == cand.layer && t.net_id != cand.net_id) {
                float req_dist = cand.radius_or_half_width + fmaxf(cand.clearance, t.clearance) + t.half_width;
                float d = drc_dist_seg_to_seg(cand.x1, cand.y1, cand.x2, cand.y2, t.x1, t.y1, t.x2, t.y2);
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            }
        }

        for (unsigned int i = 0u; i < params.num_vias; ++i) {
            GpuDrcVia v = vias[i];
            if (v.net_id != cand.net_id) {
                float req_dist = cand.radius_or_half_width + fmaxf(cand.clearance, v.clearance) + v.radius;
                float d = drc_dist_point_to_seg(v.x, v.y, cand.x1, cand.y1, cand.x2, cand.y2);
                if (d < req_dist - eps) {
                    results_out[c] = 0u;
                    return;
                }
            }
        }
    }

    results_out[c] = 1u;
}

} // extern "C"
