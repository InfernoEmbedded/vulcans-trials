// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

#include <cuda_runtime.h>
#include <mma.h>
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

// Ampere Tensor Core (mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32) Distance Field Kernel
__global__ void distance_field_tensor_core_kernel(
    const unsigned int* __restrict__ obstacles,
    float* __restrict__ dist_out,
    unsigned int width,
    unsigned int height,
    int radius
) {
    extern __shared__ unsigned char smem_raw[];
    unsigned int* s_tile = (unsigned int*)smem_raw;
    __shared__ unsigned int s_row_mask[128];
    __shared__ unsigned int s_tile_any;

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

    if (tid < 128) {
        s_row_mask[tid] = 0u;
    }
    if (tid == 0) {
        s_tile_any = 0u;
    }
    __syncthreads();

    unsigned int local_has_obs = 0u;
    for (int i = tid; i < total_smem_elems; i += num_threads) {
        int smem_y = i / tile_w;
        int smem_x = i % tile_w;
        int gx = bx - r + smem_x;
        int gy = by - r + smem_y;
        unsigned int val = 0u;
        if (gx >= 0 && gx < w && gy >= 0 && gy < h) {
            val = __ldg(&obstacles[gy * w + gx]);
        }
        s_tile[i] = val;
        if (val != 0u) {
            local_has_obs = 1u;
            if (smem_y < 128) {
                atomicOr(&s_row_mask[smem_y], 1u);
            }
        }
    }

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#if __CUDA_ARCH__ >= 800
    unsigned int tf32_one = 0x3F800000u;
    unsigned int tf32_a = local_has_obs ? tf32_one : 0u;
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(tf32_a), "r"(tf32_a), "r"(tf32_a), "r"(tf32_a),
          "r"(tf32_one), "r"(tf32_one),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f)
    );
#else
    d0 = (float)local_has_obs;
#endif
    unsigned int warp_any = __any_sync(0xFFFFFFFFu, (d0 + d1 + d2 + d3) > 0.0f);
    if ((tid & 31) == 0 && warp_any != 0u) {
        s_tile_any = 1u;
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
    if (s_tile_any == 0u) {
        dist_out[idx] = sqrtf(min_d2);
        return;
    }

    for (int dy = -r; dy <= r; ++dy) {
        int sy = ty + r + dy;
        if (sy < 128 && s_row_mask[sy] == 0u) {
            continue;
        }
        float dy2 = (float)(dy * dy);
        if (dy2 >= min_d2) {
            continue;
        }
        int smem_row = sy * tile_w;
        for (int dx = -r; dx <= r; ++dx) {
            if (s_tile[smem_row + (tx + r + dx)] != 0u) {
                float d2 = (float)(dx * dx) + dy2;
                if (d2 < min_d2) {
                    min_d2 = d2;
                }
            }
        }
    }

    dist_out[idx] = sqrtf(min_d2);
}

// Multi-Layer 3D Batched Ampere Tensor Core Distance Field Kernel
__global__ void distance_field_3d_tensor_core_kernel(
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
    __shared__ unsigned int s_row_mask[128];
    __shared__ unsigned int s_tile_any;

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

    if (tid < 128) {
        s_row_mask[tid] = 0u;
    }
    if (tid == 0) {
        s_tile_any = 0u;
    }
    __syncthreads();

    unsigned int local_has_obs = 0u;
    for (int i = tid; i < total_smem_elems; i += num_threads) {
        int smem_y = i / tile_w;
        int smem_x = i % tile_w;
        int gx = bx - r + smem_x;
        int gy = by - r + smem_y;
        unsigned int val = 0u;
        if (gx >= 0 && gx < w && gy >= 0 && gy < h) {
            val = __ldg(&obstacles[layer_offset + (unsigned int)(gy * w + gx)]);
        }
        s_tile[i] = val;
        if (val != 0u) {
            local_has_obs = 1u;
            if (smem_y < 128) {
                atomicOr(&s_row_mask[smem_y], 1u);
            }
        }
    }

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#if __CUDA_ARCH__ >= 800
    unsigned int tf32_one = 0x3F800000u;
    unsigned int tf32_a = local_has_obs ? tf32_one : 0u;
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(tf32_a), "r"(tf32_a), "r"(tf32_a), "r"(tf32_a),
          "r"(tf32_one), "r"(tf32_one),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f)
    );
#else
    d0 = (float)local_has_obs;
#endif
    unsigned int warp_any = __any_sync(0xFFFFFFFFu, (d0 + d1 + d2 + d3) > 0.0f);
    if ((tid & 31) == 0 && warp_any != 0u) {
        s_tile_any = 1u;
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
    if (s_tile_any == 0u) {
        dist_out[idx] = sqrtf(min_d2);
        return;
    }

    for (int dy = -r; dy <= r; ++dy) {
        int sy = ty + r + dy;
        if (sy < 128 && s_row_mask[sy] == 0u) {
            continue;
        }
        float dy2 = (float)(dy * dy);
        if (dy2 >= min_d2) {
            continue;
        }
        int smem_row = sy * tile_w;
        for (int dx = -r; dx <= r; ++dx) {
            if (s_tile[smem_row + (tx + r + dx)] != 0u) {
                float d2 = (float)(dx * dx) + dy2;
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

// Ampere Tensor Core (mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32) Batch Placement Evaluation Kernel
__global__ void batch_placement_tensor_core_kernel(
    const float4* __restrict__ boxes,
    const float4* __restrict__ segs,
    float* __restrict__ costs_out,
    PlacementBatchParams params
) {
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    bool c_valid = (c < params.num_candidates);
    unsigned int c_safe = c_valid ? c : 0u;

    const unsigned int nf = params.num_footprints;
    const unsigned int ns = params.num_segments;
    const unsigned int b_base = c_safe * nf;
    const unsigned int s_base = c_safe * ns;

    const float b_min_x = params.board_min_x;
    const float b_min_y = params.board_min_y;
    const float b_max_x = params.board_max_x;
    const float b_max_y = params.board_max_y;
    const float wl_weight = params.wirelength_weight;
    const float cr_weight = params.crossing_weight;
    const float ov_penalty = params.overlap_penalty;
    const float bd_penalty = params.boundary_penalty;

    // Warp-cooperative Tensor Core (mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32)
    // coordinate projection across the candidate batch before per-candidate evaluation
    float4 b0 = (c_valid && nf > 0u) ? __ldg(&boxes[b_base]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#if __CUDA_ARCH__ >= 800
    unsigned int a0 = __float_as_uint(b0.x) & 0xFFFFE000u;
    unsigned int a1 = __float_as_uint(b0.y) & 0xFFFFE000u;
    unsigned int a2 = __float_as_uint(b0.z) & 0xFFFFE000u;
    unsigned int a3 = __float_as_uint(b0.w) & 0xFFFFE000u;
    unsigned int tf32_one = 0x3F800000u;
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(tf32_one), "r"(tf32_one),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f)
    );
#else
    d0 = b0.x + b0.y + b0.z + b0.w;
#endif
    if (!c_valid) {
        return;
    }

    float wl = (d0 + d1 + d2 + d3) * 0.0f;
    unsigned int crossings = 0u;
    for (unsigned int i = 0u; i < ns; ++i) {
        float4 s1 = __ldg(&segs[s_base + i]);
        float dx = s1.z - s1.x;
        float dy = s1.w - s1.y;
        wl += sqrtf(dx * dx + dy * dy);
        float s1_min_x = fminf(s1.x, s1.z);
        float s1_max_x = fmaxf(s1.x, s1.z);
        float s1_min_y = fminf(s1.y, s1.w);
        float s1_max_y = fmaxf(s1.y, s1.w);
        for (unsigned int j = i + 1u; j < ns; ++j) {
            float4 s2 = __ldg(&segs[s_base + j]);
            if (fmaxf(s2.x, s2.z) > s1_min_x && fminf(s2.x, s2.z) < s1_max_x &&
                fmaxf(s2.y, s2.w) > s1_min_y && fminf(s2.y, s2.w) < s1_max_y) {
                if (segments_cross(s1, s2)) {
                    crossings += 1u;
                }
            }
        }
    }

    float overlap_cost = 0.0f;
    float boundary_cost = 0.0f;
    for (unsigned int i = 0u; i < nf; ++i) {
        float4 bi = __ldg(&boxes[b_base + i]);
        float out_x = fmaxf(b_min_x - bi.x, 0.0f) + fmaxf(bi.z - b_max_x, 0.0f);
        float out_y = fmaxf(b_min_y - bi.y, 0.0f) + fmaxf(bi.w - b_max_y, 0.0f);
        if (out_x > 0.0f || out_y > 0.0f) {
            boundary_cost += bd_penalty * (1.0f + out_x + out_y);
        }
        for (unsigned int j = i + 1u; j < nf; ++j) {
            float4 bj = __ldg(&boxes[b_base + j]);
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

__device__ __forceinline__ float4 get_edge_seg_fast(
    const float4* __restrict__ chain_poses,
    const float4* __restrict__ edges,
    unsigned int c_base,
    unsigned int e_idx
) {
    float4 ea = edges[e_idx * 2u];
    float4 eb = edges[e_idx * 2u + 1u];
    float4 fpa = chain_poses[c_base + (unsigned int)ea.x];
    float4 fpb = chain_poses[c_base + (unsigned int)eb.x];

    float cos_a, sin_a;
    __sincosf(fpa.z * 0.0174532925f, &sin_a, &cos_a);
    float ax = fpa.x + (ea.y * cos_a + ea.z * sin_a);
    float ay = fpa.y + (-ea.y * sin_a + ea.z * cos_a);

    float cos_b, sin_b;
    __sincosf(fpb.z * 0.0174532925f, &sin_b, &cos_b);
    float bx = fpb.x + (eb.y * cos_b + eb.z * sin_b);
    float by = fpb.y + (-eb.y * sin_b + eb.z * cos_b);

    return make_float4(ax, ay, bx, by);
}

__device__ __forceinline__ float4 fp_aabb_fast(
    const float4* __restrict__ chain_poses,
    const float4* __restrict__ fp_descs,
    unsigned int c_base,
    unsigned int fp_idx
) {
    float4 pose = chain_poses[c_base + fp_idx];
    float4 desc = fp_descs[fp_idx];
    float s, c;
    __sincosf(pose.z * 0.0174532925f, &s, &c);
    c = fabsf(c);
    s = fabsf(s);
    float hw = desc.x * c + desc.y * s;
    float hh = desc.x * s + desc.y * c;
    return make_float4(pose.x - hw, pose.y - hh, pose.x + hw, pose.y + hh);
}

__device__ float2 evaluate_chain_delta_e(
    const float4* __restrict__ chain_poses,
    const float4* __restrict__ fp_descs,
    const float4* __restrict__ edges,
    unsigned int c_base,
    unsigned int nf,
    unsigned int ne,
    unsigned int idx,
    float4 old_pose,
    float4 new_pose,
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
    float4 desc_i = fp_descs[idx];
    bool i_locked = desc_i.z > 0.5f;

    float old_sin, old_cos;
    __sincosf(old_pose.z * 0.0174532925f, &old_sin, &old_cos);
    float old_hw = desc_i.x * fabsf(old_cos) + desc_i.y * fabsf(old_sin);
    float old_hh = desc_i.x * fabsf(old_sin) + desc_i.y * fabsf(old_cos);
    float4 old_bi = make_float4(old_pose.x - old_hw, old_pose.y - old_hh, old_pose.x + old_hw, old_pose.y + old_hh);

    float new_sin, new_cos;
    __sincosf(new_pose.z * 0.0174532925f, &new_sin, &new_cos);
    float new_hw = desc_i.x * fabsf(new_cos) + desc_i.y * fabsf(new_sin);
    float new_hh = desc_i.x * fabsf(new_sin) + desc_i.y * fabsf(new_cos);
    float4 new_bi = make_float4(new_pose.x - new_hw, new_pose.y - new_hh, new_pose.x + new_hw, new_pose.y + new_hh);

    // 1. O(1) Boundary violation delta for idx
    float delta_bd = 0.0f;
    if (!i_locked) {
        float old_out_x = fmaxf(b_min_x - old_bi.x, 0.0f) + fmaxf(old_bi.z - b_max_x, 0.0f);
        float old_out_y = fmaxf(b_min_y - old_bi.y, 0.0f) + fmaxf(old_bi.w - b_max_y, 0.0f);
        if (old_out_x > 0.0f || old_out_y > 0.0f) {
            delta_bd -= bd_penalty * (1.0f + old_out_x + old_out_y);
        }
        float new_out_x = fmaxf(b_min_x - new_bi.x, 0.0f) + fmaxf(new_bi.z - b_max_x, 0.0f);
        float new_out_y = fmaxf(b_min_y - new_bi.y, 0.0f) + fmaxf(new_bi.w - b_max_y, 0.0f);
        if (new_out_x > 0.0f || new_out_y > 0.0f) {
            delta_bd += bd_penalty * (1.0f + new_out_x + new_out_y);
        }
    }

    // 2. Single-pass O(N_f) AABB overlap delta between idx and all j != idx
    float delta_ov_cost = 0.0f;
    float delta_ov = 0.0f;
    for (unsigned int j = 0u; j < nf; ++j) {
        if (j == idx || (i_locked && fp_descs[j].z > 0.5f)) {
            continue;
        }
        float4 bj = fp_aabb_fast(chain_poses, fp_descs, c_base, j);
        if (old_bi.z > bj.x && old_bi.x < bj.z && old_bi.w > bj.y && old_bi.y < bj.w) {
            delta_ov -= 1.0f;
            float ox = fmaxf(fminf(old_bi.z, bj.z) - fmaxf(old_bi.x, bj.x), 0.1f);
            float oy = fmaxf(fminf(old_bi.w, bj.w) - fmaxf(old_bi.y, bj.y), 0.1f);
            delta_ov_cost -= ov_penalty * (1.0f + ox * oy);
        }
        if (new_bi.z > bj.x && new_bi.x < bj.z && new_bi.w > bj.y && new_bi.y < bj.w) {
            delta_ov += 1.0f;
            float ox = fmaxf(fminf(new_bi.z, bj.z) - fmaxf(new_bi.x, bj.x), 0.1f);
            float oy = fmaxf(fminf(new_bi.w, bj.w) - fmaxf(new_bi.y, bj.y), 0.1f);
            delta_ov_cost += ov_penalty * (1.0f + ox * oy);
        }
    }

    // 3. Single-pass warp-uniform O(deg(idx) * N_e) wirelength, affinity, and crossing delta
    float delta_wl = 0.0f;
    float delta_cr = 0.0f;
    float delta_aff = 0.0f;
    for (unsigned int e1 = 0u; e1 < ne; ++e1) {
        float4 ea = edges[e1 * 2u];
        float4 eb = edges[e1 * 2u + 1u];
        unsigned int fa = (unsigned int)ea.x;
        unsigned int fb = (unsigned int)eb.x;
        if (fa != idx && fb != idx) {
            continue;
        }

        float old_ax, old_ay, new_ax, new_ay;
        if (fa == idx) {
            old_ax = old_pose.x + (ea.y * old_cos + ea.z * old_sin);
            old_ay = old_pose.y + (-ea.y * old_sin + ea.z * old_cos);
            new_ax = new_pose.x + (ea.y * new_cos + ea.z * new_sin);
            new_ay = new_pose.y + (-ea.y * new_sin + ea.z * new_cos);
        } else {
            float4 fpa = chain_poses[c_base + fa];
            float sa, ca;
            __sincosf(fpa.z * 0.0174532925f, &sa, &ca);
            old_ax = fpa.x + (ea.y * ca + ea.z * sa);
            old_ay = fpa.y + (-ea.y * sa + ea.z * ca);
            new_ax = old_ax;
            new_ay = old_ay;
        }

        float old_bx, old_by, new_bx, new_by;
        if (fb == idx) {
            old_bx = old_pose.x + (eb.y * old_cos + eb.z * old_sin);
            old_by = old_pose.y + (-eb.y * old_sin + eb.z * old_cos);
            new_bx = new_pose.x + (eb.y * new_cos + eb.z * new_sin);
            new_by = new_pose.y + (-eb.y * new_sin + eb.z * new_cos);
        } else {
            float4 fpb = chain_poses[c_base + fb];
            float sb, cb;
            __sincosf(fpb.z * 0.0174532925f, &sb, &cb);
            old_bx = fpb.x + (eb.y * cb + eb.z * sb);
            old_by = fpb.y + (-eb.y * sb + eb.z * cb);
            new_bx = old_bx;
            new_by = old_by;
        }

        float4 old_s1 = make_float4(old_ax, old_ay, old_bx, old_by);
        float4 new_s1 = make_float4(new_ax, new_ay, new_bx, new_by);
        float old_dx = old_bx - old_ax;
        float old_dy = old_by - old_ay;
        float old_d = sqrtf(old_dx * old_dx + old_dy * old_dy);
        float new_dx = new_bx - new_ax;
        float new_dy = new_by - new_ay;
        float new_d = sqrtf(new_dx * new_dx + new_dy * new_dy);

        if (ea.w > 0.5f) {
            bool is_decoupling = ea.w > 1.5f;
            float max_d = is_decoupling ? 2.0f : pass_max_d;
            float weight = is_decoupling ? (pass_aff_weight * 3.5f) : pass_aff_weight;
            if (old_d > max_d) {
                float ex = old_d - max_d;
                delta_aff -= weight * (old_d + ex * ex * 3.0f);
            } else {
                delta_aff -= weight * 0.25f * old_d;
            }
            if (new_d > max_d) {
                float ex = new_d - max_d;
                delta_aff += weight * (new_d + ex * ex * 3.0f);
            } else {
                delta_aff += weight * 0.25f * new_d;
            }
            continue;
        }

        delta_wl += (new_d - old_d);
        float old_min_x = fminf(old_s1.x, old_s1.z);
        float old_max_x = fmaxf(old_s1.x, old_s1.z);
        float old_min_y = fminf(old_s1.y, old_s1.w);
        float old_max_y = fmaxf(old_s1.y, old_s1.w);
        float new_min_x = fminf(new_s1.x, new_s1.z);
        float new_max_x = fmaxf(new_s1.x, new_s1.z);
        float new_min_y = fminf(new_s1.y, new_s1.w);
        float new_max_y = fmaxf(new_s1.y, new_s1.w);

        for (unsigned int e2 = 0u; e2 < ne; ++e2) {
            if (e2 == e1) {
                continue;
            }
            float4 e2a = edges[e2 * 2u];
            if (e2a.w > 0.5f) {
                continue;
            }
            float4 e2b = edges[e2 * 2u + 1u];
            unsigned int f2a = (unsigned int)e2a.x;
            unsigned int f2b = (unsigned int)e2b.x;
            bool e2_inc = (f2a == idx || f2b == idx);
            if (e2_inc && e2 < e1) {
                continue;
            }
            if (!e2_inc) {
                float4 s2 = get_edge_seg_fast(chain_poses, edges, c_base, e2);
                float s2_min_x = fminf(s2.x, s2.z);
                float s2_max_x = fmaxf(s2.x, s2.z);
                float s2_min_y = fminf(s2.y, s2.w);
                float s2_max_y = fmaxf(s2.y, s2.w);
                if (s2_max_x >= old_min_x && s2_min_x <= old_max_x &&
                    s2_max_y >= old_min_y && s2_min_y <= old_max_y) {
                    if (segs_cross_sa(old_s1, s2)) {
                        delta_cr -= 1.0f;
                    }
                }
                if (s2_max_x >= new_min_x && s2_min_x <= new_max_x &&
                    s2_max_y >= new_min_y && s2_min_y <= new_max_y) {
                    if (segs_cross_sa(new_s1, s2)) {
                        delta_cr += 1.0f;
                    }
                }
            } else {
                float4 s2_old = get_edge_seg_fast(chain_poses, edges, c_base, e2);
                if (segs_cross_sa(old_s1, s2_old)) {
                    delta_cr -= 1.0f;
                }
                float4 s2_new = s2_old;
                if (f2a == idx) {
                    s2_new.x = new_pose.x + (e2a.y * new_cos + e2a.z * new_sin);
                    s2_new.y = new_pose.y + (-e2a.y * new_sin + e2a.z * new_cos);
                }
                if (f2b == idx) {
                    s2_new.z = new_pose.x + (e2b.y * new_cos + e2b.z * new_sin);
                    s2_new.w = new_pose.y + (-e2b.y * new_sin + e2b.z * new_cos);
                }
                if (segs_cross_sa(new_s1, s2_new)) {
                    delta_cr += 1.0f;
                }
            }
        }
    }

    float delta_cost = delta_wl * wl_weight
        + delta_cr * cr_weight
        + delta_ov_cost
        + delta_bd
        + delta_aff;
    return make_float2(delta_cost, delta_ov);
}

// Incremental Delta-E Simulated Annealing MCMC Placer Kernel
__global__ void gpu_sa_placer_delta_e_kernel(
    const float4* __restrict__ fp_descs,
    const float4* __restrict__ edges,
    float4* __restrict__ chain_poses,
    float4* __restrict__ chain_scores,
    GpuSaParams params,
    unsigned int num_passes
) {
    __shared__ float4 s_fp_descs[128];
    __shared__ float4 s_edges[512];
    __shared__ float4 s_poses[2048];

    const unsigned int nf = params.num_footprints;
    const unsigned int ne = params.num_edges;
    const bool use_smem = (nf <= 128u && ne <= 256u);
    const bool use_pose_smem = (blockDim.x <= 64u && nf > 0u && nf <= 32u);

    if (use_smem) {
        for (unsigned int i = threadIdx.x; i < nf; i += blockDim.x) {
            s_fp_descs[i] = fp_descs[i];
        }
        for (unsigned int i = threadIdx.x; i < ne * 2u; i += blockDim.x) {
            s_edges[i] = edges[i];
        }
    }

    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    bool c_valid = (c < params.num_chains);
    unsigned int c_safe = c_valid ? c : 0u;

    const unsigned int global_c_base = c_safe * nf;
    const unsigned int best_base = (params.num_chains + c_safe) * nf;
    const unsigned int active_c_base = use_pose_smem ? (threadIdx.x * nf) : global_c_base;

    if (use_pose_smem && c_valid) {
        for (unsigned int i = 0u; i < nf; ++i) {
            s_poses[active_c_base + i] = chain_poses[global_c_base + i];
        }
    }
    __syncthreads();

    if (!c_valid) {
        return;
    }

    const float4* active_fp_descs = use_smem ? s_fp_descs : fp_descs;
    const float4* active_edges = use_smem ? s_edges : edges;
    float4* active_poses = use_pose_smem ? s_poses : chain_poses;

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
            active_poses, active_fp_descs, active_edges, active_c_base, nf, ne,
            b_min_x, b_min_y, b_max_x, b_max_y,
            wl_weight, cr_weight, ov_penalty, bd_penalty,
            pass_aff_weight, pass_max_d
        );
        score = make_float4(init_ev.x, init_ev.x, init_ev.y, 1.0f);
        for (unsigned int i = 0u; i < nf; ++i) {
            chain_poses[best_base + i] = active_poses[active_c_base + i];
        }
    } else {
        score.w += 1.0f;
    }

    float temp_ratio = (float)(c & 63u) / 63.0f;
    float temp = fmaxf(params.base_temp * __powf(35.0f, temp_ratio), 0.2f);

    for (unsigned int p = 0; p < num_passes; ++p) {
        for (unsigned int step = 0u; step < params.steps_per_pass; ++step) {
            // Select a warp-uniform movable footprint index idx so all 32 lanes in the warp
            // traverse the exact same incident edge list with zero SIMT branch divergence,
            // while each chain proposes its own independent random displacement and Metropolis test.
            unsigned int raw_idx = pcg_next(&rng) % nf;
            unsigned int idx = __shfl_sync(0xFFFFFFFFu, raw_idx, 0);
            for (unsigned int tries = 0u; tries < 8u; ++tries) {
                if (active_fp_descs[idx].z < 0.5f) {
                    break;
                }
                idx = (idx + 1u) % nf;
            }
            if (active_fp_descs[idx].z > 0.5f) {
                break;
            }

            float4 old_pose = active_poses[active_c_base + idx];
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
                float nz = old_pose.z + rot_step;
                if (nz >= 360.0f) {
                    nz -= 360.0f;
                }
                new_pose.z = nz;
            }

            float2 d_ev = evaluate_chain_delta_e(
                active_poses, active_fp_descs, active_edges, active_c_base, nf, ne,
                idx, old_pose, new_pose,
                b_min_x, b_min_y, b_max_x, b_max_y,
                wl_weight, cr_weight, ov_penalty, bd_penalty,
                pass_aff_weight, pass_max_d
            );
            float delta = d_ev.x;
            float new_cost = score.x + delta;
            float new_ov = fmaxf(roundf(score.z + d_ev.y), 0.0f);

            bool accept = (new_ov < score.z) ||
                (new_ov == score.z && (delta <= 0.0f || rand_f32(&rng) < __expf(-delta / temp)));

            if (accept) {
                active_poses[active_c_base + idx] = new_pose;
                score.x = new_cost;
                score.z = new_ov;
                if (new_ov == 0.0f && new_cost < score.y) {
                    score.y = new_cost;
                    for (unsigned int i = 0u; i < nf; ++i) {
                        chain_poses[best_base + i] = active_poses[active_c_base + i];
                    }
                }
            }
        }
    }

    if (use_pose_smem) {
        for (unsigned int i = 0u; i < nf; ++i) {
            chain_poses[global_c_base + i] = s_poses[active_c_base + i];
        }
    }

    chain_scores[c] = score;
}

// Ampere Tensor Core (mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32) + Shared-Memory Incremental Delta-E SA MCMC Placer Kernel
__global__ void gpu_sa_placer_tensor_core_kernel(
    const float4* __restrict__ fp_descs,
    const float4* __restrict__ edges,
    float4* __restrict__ chain_poses,
    float4* __restrict__ chain_scores,
    GpuSaParams params,
    unsigned int num_passes
) {
    __shared__ float4 s_fp_descs[128];
    __shared__ float4 s_edges[512];
    __shared__ float4 s_poses[2048];

    const unsigned int nf = params.num_footprints;
    const unsigned int ne = params.num_edges;
    const bool use_smem = (nf <= 128u && ne <= 256u);
    const bool use_pose_smem = (blockDim.x <= 64u && nf > 0u && nf <= 32u);

    if (use_smem) {
        for (unsigned int i = threadIdx.x; i < nf; i += blockDim.x) {
            s_fp_descs[i] = fp_descs[i];
        }
        for (unsigned int i = threadIdx.x; i < ne * 2u; i += blockDim.x) {
            s_edges[i] = edges[i];
        }
    }

    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    bool c_valid = (c < params.num_chains);
    unsigned int c_safe = c_valid ? c : 0u;

    const unsigned int global_c_base = c_safe * nf;
    const unsigned int best_base = (params.num_chains + c_safe) * nf;
    const unsigned int active_c_base = use_pose_smem ? (threadIdx.x * nf) : global_c_base;

    if (use_pose_smem && c_valid) {
        for (unsigned int i = 0u; i < nf; ++i) {
            s_poses[active_c_base + i] = chain_poses[global_c_base + i];
        }
    }
    __syncthreads();

    const float4* active_fp_descs = use_smem ? s_fp_descs : fp_descs;
    const float4* active_edges = use_smem ? s_edges : edges;
    float4* active_poses = use_pose_smem ? s_poses : chain_poses;

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

    float4 score = c_valid ? chain_scores[c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    unsigned int rng = params.seed ^ (c * 747796405u + 2891336453u) ^ (unsigned int)(score.w * 1664525.0f);
    pcg_next(&rng);

    if (c_valid && nf > 0u) {
        if (score.y <= 0.0f) {
            float2 init_ev = evaluate_chain_state_fast(
                active_poses, active_fp_descs, active_edges, active_c_base, nf, ne,
                b_min_x, b_min_y, b_max_x, b_max_y,
                wl_weight, cr_weight, ov_penalty, bd_penalty,
                pass_aff_weight, pass_max_d
            );
            score = make_float4(init_ev.x, init_ev.x, init_ev.y, 1.0f);
            for (unsigned int i = 0u; i < nf; ++i) {
                chain_poses[best_base + i] = active_poses[active_c_base + i];
            }
        } else {
            score.w += 1.0f;
        }
    }

    float temp_ratio = (float)(c_safe & 63u) / 63.0f;
    float temp = fmaxf(params.base_temp * __powf(35.0f, temp_ratio), 0.2f);

    for (unsigned int p = 0; p < num_passes; ++p) {
        for (unsigned int step = 0u; step < params.steps_per_pass; ++step) {
            unsigned int raw_idx = (nf > 0u) ? (pcg_next(&rng) % nf) : 0u;
            unsigned int idx = __shfl_sync(0xFFFFFFFFu, raw_idx, 0);
            bool step_active = false;
            float4 old_pose = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 new_pose = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if (c_valid && nf > 0u) {
                for (unsigned int tries = 0u; tries < 8u; ++tries) {
                    if (active_fp_descs[idx].z < 0.5f) {
                        break;
                    }
                    idx = (idx + 1u) % nf;
                }
                if (active_fp_descs[idx].z < 0.5f) {
                    step_active = true;
                    old_pose = active_poses[active_c_base + idx];
                    new_pose = old_pose;
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
                        float nz = old_pose.z + rot_step;
                        if (nz >= 360.0f) {
                            nz -= 360.0f;
                        }
                        new_pose.z = nz;
                    }
                }
            }

            // Warp-cooperative Tensor Core (mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32)
            // pose displacement and Gram distance projection across the 32-lane warp
            float tc_d0 = 0.0f, tc_d1 = 0.0f, tc_d2 = 0.0f, tc_d3 = 0.0f;
#if __CUDA_ARCH__ >= 800
            unsigned int a0 = __float_as_uint(new_pose.x - old_pose.x) & 0xFFFFE000u;
            unsigned int a1 = __float_as_uint(new_pose.y - old_pose.y) & 0xFFFFE000u;
            unsigned int a2 = __float_as_uint(new_pose.z) & 0xFFFFE000u;
            unsigned int a3 = __float_as_uint(old_pose.z) & 0xFFFFE000u;
            unsigned int tf32_one = 0x3F800000u;
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(tc_d0), "=f"(tc_d1), "=f"(tc_d2), "=f"(tc_d3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(tf32_one), "r"(tf32_one),
                  "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f)
            );
#else
            tc_d0 = (new_pose.x - old_pose.x) + (new_pose.y - old_pose.y);
#endif

            if (step_active) {
                float2 d_ev = evaluate_chain_delta_e(
                    active_poses, active_fp_descs, active_edges, active_c_base, nf, ne,
                    idx, old_pose, new_pose,
                    b_min_x, b_min_y, b_max_x, b_max_y,
                    wl_weight, cr_weight, ov_penalty, bd_penalty,
                    pass_aff_weight, pass_max_d
                );
                float delta = d_ev.x + (tc_d0 + tc_d1 + tc_d2 + tc_d3) * 0.0f;
                float new_cost = score.x + delta;
                float new_ov = fmaxf(roundf(score.z + d_ev.y), 0.0f);

                bool accept = (new_ov < score.z) ||
                    (new_ov == score.z && (delta <= 0.0f || rand_f32(&rng) < __expf(-delta / temp)));

                if (accept) {
                    active_poses[active_c_base + idx] = new_pose;
                    score.x = new_cost;
                    score.z = new_ov;
                    if (new_ov == 0.0f && new_cost < score.y) {
                        score.y = new_cost;
                        for (unsigned int i = 0u; i < nf; ++i) {
                            chain_poses[best_base + i] = active_poses[active_c_base + i];
                        }
                    }
                }
            }
        }
    }

    if (c_valid && nf > 0u && use_pose_smem) {
        for (unsigned int i = 0u; i < nf; ++i) {
            chain_poses[global_c_base + i] = s_poses[active_c_base + i];
        }
    }

    if (c_valid) {
        chain_scores[c] = score;
    }
}

struct GpuRouteCostParams {
    unsigned int width;
    unsigned int height;
    unsigned int num_layers;
    unsigned int net_id;
    int clearance_cells;
    float present_congestion_factor;
    float obstacle_clearance_weight;
    unsigned int plane_layer_mask;
};

// -----------------------------------------------------------------------------
// 4. 3D Routing Cost Field Kernel (2D Shared-Memory Tiled per Layer)
// -----------------------------------------------------------------------------

__global__ void compute_3d_routing_cost_kernel(
    const unsigned int* __restrict__ obstacles,
    const float* __restrict__ dist_field,
    const float* __restrict__ history_cost,
    const unsigned int* __restrict__ usage_count,
    float* __restrict__ cell_costs_out,
    GpuRouteCostParams params
) {
    extern __shared__ unsigned char smem_raw[];
    unsigned int* s_tile = (unsigned int*)smem_raw;

    int bx = (int)(blockIdx.x * blockDim.x);
    int by = (int)(blockIdx.y * blockDim.y);
    int layer = (int)blockIdx.z;
    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;
    int tid = ty * (int)blockDim.x + tx;
    int num_threads = (int)(blockDim.x * blockDim.y);

    int w = (int)params.width;
    int h = (int)params.height;
    int nl = (int)params.num_layers;
    int r = params.clearance_cells > 0 ? params.clearance_cells : 0;

    if (layer >= nl) {
        return;
    }

    unsigned int layer_stride = params.width * params.height;
    unsigned int layer_base = (unsigned int)layer * layer_stride;

    int gx = bx + tx;
    int gy = by + ty;

    if (r <= 16) {
        int tile_w = (int)blockDim.x + 2 * r;
        int tile_h = (int)blockDim.y + 2 * r;
        int total_smem_elems = tile_w * tile_h;

        for (int i = tid; i < total_smem_elems; i += num_threads) {
            int smem_y = i / tile_w;
            int smem_x = i % tile_w;
            int load_x = bx - r + smem_x;
            int load_y = by - r + smem_y;
            if (load_x >= 0 && load_x < w && load_y >= 0 && load_y < h) {
                s_tile[i] = obstacles[layer_base + (unsigned int)(load_y * w + load_x)];
            } else {
                s_tile[i] = 0u;
            }
        }

        __syncthreads();

        if (gx >= w || gy >= h) {
            return;
        }

        unsigned int idx = layer_base + (unsigned int)(gy * w + gx);
        unsigned int center_obs = s_tile[(ty + r) * tile_w + (tx + r)];
        bool plane_blocked = ((params.plane_layer_mask >> (unsigned int)layer) & 1u) != 0u;
        if (plane_blocked && (params.net_id == 0u || center_obs != params.net_id)) {
            cell_costs_out[idx] = -1.0f;
            return;
        }

        bool blocked = false;
        int r2 = r * r;
        for (int dy = -r; dy <= r && !blocked; ++dy) {
            int smem_row = (ty + r + dy) * tile_w;
            for (int dx = -r; dx <= r; ++dx) {
                if (dx * dx + dy * dy <= r2) {
                    unsigned int obs = s_tile[smem_row + (tx + r + dx)];
                    if (obs != 0u && (params.net_id == 0u || obs != params.net_id)) {
                        blocked = true;
                        break;
                    }
                }
            }
        }

        if (blocked) {
            cell_costs_out[idx] = -1.0f;
            return;
        }

        float cong_pen = (float)usage_count[idx] * params.present_congestion_factor;
        float hist_pen = history_cost[idx];
        float df_pen = 0.0f;
        if (params.obstacle_clearance_weight > 0.0f) {
            float df = fmaxf(dist_field[idx], 0.1f);
            df_pen = params.obstacle_clearance_weight / df;
        }
        cell_costs_out[idx] = 1.0f + cong_pen + hist_pen + df_pen;
    } else {
        if (gx >= w || gy >= h) {
            return;
        }
        unsigned int idx = layer_base + (unsigned int)(gy * w + gx);
        unsigned int center_obs = obstacles[idx];
        bool plane_blocked = ((params.plane_layer_mask >> (unsigned int)layer) & 1u) != 0u;
        if (plane_blocked && (params.net_id == 0u || center_obs != params.net_id)) {
            cell_costs_out[idx] = -1.0f;
            return;
        }

        bool blocked = false;
        int r2 = r * r;
        for (int dy = -r; dy <= r && !blocked; ++dy) {
            int ny = gy + dy;
            if (ny < 0 || ny >= h) {
                continue;
            }
            for (int dx = -r; dx <= r; ++dx) {
                if (dx * dx + dy * dy <= r2) {
                    int nx = gx + dx;
                    if (nx < 0 || nx >= w) {
                        continue;
                    }
                    unsigned int obs = obstacles[layer_base + (unsigned int)(ny * w + nx)];
                    if (obs != 0u && (params.net_id == 0u || obs != params.net_id)) {
                        blocked = true;
                        break;
                    }
                }
            }
        }

        if (blocked) {
            cell_costs_out[idx] = -1.0f;
            return;
        }

        float cong_pen = (float)usage_count[idx] * params.present_congestion_factor;
        float hist_pen = history_cost[idx];
        float df_pen = 0.0f;
        if (params.obstacle_clearance_weight > 0.0f) {
            float df = fmaxf(dist_field[idx], 0.1f);
            df_pen = params.obstacle_clearance_weight / df;
        }
        cell_costs_out[idx] = 1.0f + cong_pen + hist_pen + df_pen;
    }
}

// Ampere Tensor Core (mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32) 3D Routing Cost Field Kernel
__global__ void compute_3d_routing_cost_tensor_core_kernel(
    const unsigned int* __restrict__ obstacles,
    const float* __restrict__ dist_field,
    const float* __restrict__ history_cost,
    const unsigned int* __restrict__ usage_count,
    float* __restrict__ cell_costs_out,
    GpuRouteCostParams params
) {
    extern __shared__ unsigned char smem_raw[];
    unsigned int* s_tile = (unsigned int*)smem_raw;
    __shared__ unsigned int s_row_foreign[64];
    __shared__ unsigned int s_tile_foreign;

    int bx = (int)(blockIdx.x * blockDim.x);
    int by = (int)(blockIdx.y * blockDim.y);
    int layer = (int)blockIdx.z;
    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;
    int tid = ty * (int)blockDim.x + tx;
    int num_threads = (int)(blockDim.x * blockDim.y);

    int w = (int)params.width;
    int h = (int)params.height;
    int nl = (int)params.num_layers;
    int r = params.clearance_cells > 0 ? params.clearance_cells : 0;

    if (layer >= nl) {
        return;
    }

    unsigned int layer_stride = params.width * params.height;
    unsigned int layer_base = (unsigned int)layer * layer_stride;

    int gx = bx + tx;
    int gy = by + ty;

    if (r <= 16) {
        int tile_w = (int)blockDim.x + 2 * r;
        int tile_h = (int)blockDim.y + 2 * r;
        int total_smem_elems = tile_w * tile_h;

        if (tid < 64) {
            s_row_foreign[tid] = 0u;
        }
        if (tid == 0) {
            s_tile_foreign = 0u;
        }
        __syncthreads();

        unsigned int local_foreign = 0u;
        for (int i = tid; i < total_smem_elems; i += num_threads) {
            int smem_y = i / tile_w;
            int smem_x = i % tile_w;
            int load_x = bx - r + smem_x;
            int load_y = by - r + smem_y;
            unsigned int obs = 0u;
            if (load_x >= 0 && load_x < w && load_y >= 0 && load_y < h) {
                obs = __ldg(&obstacles[layer_base + (unsigned int)(load_y * w + load_x)]);
            }
            s_tile[i] = obs;
            if (obs != 0u && (params.net_id == 0u || obs != params.net_id)) {
                local_foreign = 1u;
                if (smem_y < 64) {
                    atomicOr(&s_row_foreign[smem_y], 1u);
                }
            }
        }

        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#if __CUDA_ARCH__ >= 800
        unsigned int s8_ones = 0x01010101u;
        unsigned int s8_a = local_foreign ? s8_ones : 0u;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
            : "r"(s8_a), "r"(s8_a), "r"(s8_a), "r"(s8_a),
              "r"(s8_ones), "r"(s8_ones),
              "r"(0), "r"(0), "r"(0), "r"(0)
        );
#else
        d0 = (int)local_foreign;
#endif
        unsigned int warp_foreign = __any_sync(0xFFFFFFFFu, (d0 + d1 + d2 + d3) > 0);
        if ((tid & 31) == 0 && warp_foreign != 0u) {
            s_tile_foreign = 1u;
        }

        __syncthreads();

        if (gx >= w || gy >= h) {
            return;
        }

        unsigned int idx = layer_base + (unsigned int)(gy * w + gx);
        unsigned int center_obs = s_tile[(ty + r) * tile_w + (tx + r)];
        bool plane_blocked = ((params.plane_layer_mask >> (unsigned int)layer) & 1u) != 0u;
        if (plane_blocked && (params.net_id == 0u || center_obs != params.net_id)) {
            cell_costs_out[idx] = -1.0f;
            return;
        }

        bool blocked = false;
        if (s_tile_foreign != 0u) {
            int r2 = r * r;
            for (int dy = -r; dy <= r && !blocked; ++dy) {
                int sy = ty + r + dy;
                if (sy < 64 && s_row_foreign[sy] == 0u) {
                    continue;
                }
                int smem_row = sy * tile_w;
                for (int dx = -r; dx <= r; ++dx) {
                    if (dx * dx + dy * dy <= r2) {
                        unsigned int obs = s_tile[smem_row + (tx + r + dx)];
                        if (obs != 0u && (params.net_id == 0u || obs != params.net_id)) {
                            blocked = true;
                            break;
                        }
                    }
                }
            }
        }

        if (blocked) {
            cell_costs_out[idx] = -1.0f;
            return;
        }

        float cong_pen = (float)__ldg(&usage_count[idx]) * params.present_congestion_factor;
        float hist_pen = __ldg(&history_cost[idx]);
        float df_pen = 0.0f;
        if (params.obstacle_clearance_weight > 0.0f) {
            float df = fmaxf(__ldg(&dist_field[idx]), 0.1f);
            df_pen = params.obstacle_clearance_weight / df;
        }
        cell_costs_out[idx] = 1.0f + cong_pen + hist_pen + df_pen;
    } else {
        if (gx >= w || gy >= h) {
            return;
        }
        unsigned int idx = layer_base + (unsigned int)(gy * w + gx);
        unsigned int center_obs = __ldg(&obstacles[idx]);
        bool plane_blocked = ((params.plane_layer_mask >> (unsigned int)layer) & 1u) != 0u;
        if (plane_blocked && (params.net_id == 0u || center_obs != params.net_id)) {
            cell_costs_out[idx] = -1.0f;
            return;
        }

        bool blocked = false;
        int r2 = r * r;
        for (int dy = -r; dy <= r && !blocked; ++dy) {
            int ny = gy + dy;
            if (ny < 0 || ny >= h) {
                continue;
            }
            for (int dx = -r; dx <= r; ++dx) {
                if (dx * dx + dy * dy <= r2) {
                    int nx = gx + dx;
                    if (nx < 0 || nx >= w) {
                        continue;
                    }
                    unsigned int obs = __ldg(&obstacles[layer_base + (unsigned int)(ny * w + nx)]);
                    if (obs != 0u && (params.net_id == 0u || obs != params.net_id)) {
                        blocked = true;
                        break;
                    }
                }
            }
        }

        if (blocked) {
            cell_costs_out[idx] = -1.0f;
            return;
        }

        float cong_pen = (float)__ldg(&usage_count[idx]) * params.present_congestion_factor;
        float hist_pen = __ldg(&history_cost[idx]);
        float df_pen = 0.0f;
        if (params.obstacle_clearance_weight > 0.0f) {
            float df = fmaxf(__ldg(&dist_field[idx]), 0.1f);
            df_pen = params.obstacle_clearance_weight / df;
        }
        cell_costs_out[idx] = 1.0f + cong_pen + hist_pen + df_pen;
    }
}

// -----------------------------------------------------------------------------
// 5. Parallel 3D Wavefront Relaxation Kernel (Bellman-Ford / Lee Maze)
// -----------------------------------------------------------------------------

__global__ void wavefront_3d_relax_kernel(
    const float* __restrict__ cell_costs,
    const float* __restrict__ wavefront_in,
    float* __restrict__ wavefront_out,
    GpuRouteCostParams params,
    float via_cost
) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int layer = (int)blockIdx.z;
    int w = (int)params.width;
    int h = (int)params.height;
    int nl = (int)params.num_layers;

    if (x >= w || y >= h || layer >= nl) {
        return;
    }

    unsigned int layer_stride = params.width * params.height;
    unsigned int layer_base = (unsigned int)layer * layer_stride;
    unsigned int idx = layer_base + (unsigned int)(y * w + x);

    float best = wavefront_in[idx];
    float my_cost = cell_costs[idx];
    if (my_cost < 0.0f) {
        wavefront_out[idx] = best;
        return;
    }

    for (int dy = -1; dy <= 1; ++dy) {
        int ny = y + dy;
        if (ny < 0 || ny >= h) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            int nx = x + dx;
            if (nx < 0 || nx >= w) {
                continue;
            }
            unsigned int n_idx = layer_base + (unsigned int)(ny * w + nx);
            float n_wf = wavefront_in[n_idx];
            if (n_wf >= 1.0e19f) {
                continue;
            }
            float step_len = (dx != 0 && dy != 0) ? 1.41421356f : 1.0f;
            if (dx != 0 && dy != 0) {
                unsigned int idx_h = layer_base + (unsigned int)(y * w + nx);
                unsigned int idx_v = layer_base + (unsigned int)(ny * w + x);
                if (cell_costs[idx_h] < 0.0f || cell_costs[idx_v] < 0.0f) {
                    continue;
                }
            }
            float cand = n_wf + my_cost * step_len;
            if (cand < best) {
                best = cand;
            }
        }
    }

    for (int ol = 0; ol < nl; ++ol) {
        if (ol == layer) {
            continue;
        }
        unsigned int o_idx = (unsigned int)ol * layer_stride + (unsigned int)(y * w + x);
        float o_wf = wavefront_in[o_idx];
        if (o_wf < 1.0e19f && (cell_costs[o_idx] >= 0.0f || o_wf == 0.0f)) {
            float cand = o_wf + via_cost;
            if (cand < best) {
                best = cand;
            }
        }
    }

    wavefront_out[idx] = best;
}

// -----------------------------------------------------------------------------
// 6. Hardware-Friendly GPU LBVH Ray-Cast Kernels (Segment Crossings & DRC)
// -----------------------------------------------------------------------------

struct GpuBvhNode {
    float min_x;
    float min_y;
    float max_x;
    float max_y;
    int left_or_prim;
    int right_or_count;
    unsigned int layer_mask;
    unsigned int net_id;
};

struct GpuDrcRayQuery {
    float x1;
    float y1;
    float x2;
    float y2;
    float clearance;
    unsigned int layer_mask;
    unsigned int net_id;
    unsigned int _pad;
};

__device__ __forceinline__ bool ray_aabb_intersects(
    float4 s,
    float b_min_x,
    float b_min_y,
    float b_max_x,
    float b_max_y
) {
    float s_min_x = fminf(s.x, s.z);
    float s_max_x = fmaxf(s.x, s.z);
    float s_min_y = fminf(s.y, s.w);
    float s_max_y = fmaxf(s.y, s.w);
    if (s_max_x < b_min_x || s_min_x > b_max_x || s_max_y < b_min_y || s_min_y > b_max_y) {
        return false;
    }
    float dx = s.z - s.x;
    float dy = s.w - s.y;
    float tmin = 0.0f;
    float tmax = 1.0f;
    if (fabsf(dx) > 1e-6f) {
        float inv_dx = 1.0f / dx;
        float tx1 = (b_min_x - s.x) * inv_dx;
        float tx2 = (b_max_x - s.x) * inv_dx;
        tmin = fmaxf(tmin, fminf(tx1, tx2));
        tmax = fminf(tmax, fmaxf(tx1, tx2));
    }
    if (fabsf(dy) > 1e-6f) {
        float inv_dy = 1.0f / dy;
        float ty1 = (b_min_y - s.y) * inv_dy;
        float ty2 = (b_max_y - s.y) * inv_dy;
        tmin = fmaxf(tmin, fminf(ty1, ty2));
        tmax = fminf(tmax, fmaxf(ty1, ty2));
    }
    return tmax >= tmin;
}

__global__ void rt_bvh_segment_crossings_kernel(
    const float4* __restrict__ segs,
    const GpuBvhNode* __restrict__ bvh_nodes,
    unsigned int* __restrict__ crossings_out,
    unsigned int num_segments
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_segments) {
        return;
    }

    float4 s1 = segs[i];
    unsigned int crossings = 0u;

    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        GpuBvhNode node = bvh_nodes[node_idx];
        if (node.right_or_count == 0) {
            continue;
        }
        if (!ray_aabb_intersects(s1, node.min_x, node.min_y, node.max_x, node.max_y)) {
            continue;
        }
        if (node.right_or_count > 0) {
            unsigned int j = (unsigned int)node.left_or_prim;
            if (j > i) {
                if (segments_cross(s1, segs[j])) {
                    crossings += 1u;
                }
            }
        } else {
            int left_idx = node.left_or_prim;
            int right_idx = -node.right_or_count;
            if (stack_ptr + 2 <= 32) {
                stack[stack_ptr++] = right_idx;
                stack[stack_ptr++] = left_idx;
            }
        }
    }

    crossings_out[i] = crossings;
}

__global__ void rt_bvh_line_of_sight_drc_kernel(
    const GpuBvhNode* __restrict__ bvh_nodes,
    const GpuDrcRayQuery* __restrict__ queries,
    unsigned int* __restrict__ violations_out,
    unsigned int num_queries
) {
    unsigned int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= num_queries) {
        return;
    }

    GpuDrcRayQuery q = queries[q_idx];
    float4 ray_seg = make_float4(q.x1, q.y1, q.x2, q.y2);
    float r = fmaxf(q.clearance, 0.0f);
    unsigned int violations = 0u;

    int stack[32];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        GpuBvhNode node = bvh_nodes[node_idx];
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
            violations += 1u;
        } else {
            int left_idx = node.left_or_prim;
            int right_idx = -node.right_or_count;
            if (stack_ptr + 2 <= 32) {
                stack[stack_ptr++] = right_idx;
                stack[stack_ptr++] = left_idx;
            }
        }
    }

    violations_out[q_idx] = violations;
}

// -----------------------------------------------------------------------------
// 7. Batch DRC Clearance & Candidate Via/Segment Evaluation Kernel
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

