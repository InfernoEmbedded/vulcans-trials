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

// Multi-Layer 3D Batched Distance Field Kernel with 2D Shared Memory Tiling per Z-Layer
__global__ void distance_field_3d_tiled_kernel(
    const unsigned int* __restrict__ obstacles,
    float* __restrict__ dist_out,
    unsigned int width,
    unsigned int height,
    unsigned int num_layers,
    int radius
) {
    unsigned int layer = blockIdx.z;
    if (layer >= num_layers) {
        return;
    }

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
    unsigned int layer_offset = layer * width * height;

    int tile_w = blockDim.x + 2 * r;
    int tile_h = blockDim.y + 2 * r;
    int total_smem_elems = tile_w * tile_h;

    for (int i = tid; i < total_smem_elems; i += num_threads) {
        int smem_y = i / tile_w;
        int smem_x = i % tile_w;
        int gx = bx - r + smem_x;
        int gy = by - r + smem_y;
        if (gx >= 0 && gx < w && gy >= 0 && gy < h) {
            s_tile[i] = obstacles[layer_offset + (unsigned int)(gy * w + gx)];
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

    unsigned int idx = layer_offset + (unsigned int)(gy * w + gx);
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

} // extern "C"
