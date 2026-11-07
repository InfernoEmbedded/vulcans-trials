// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use std::time::Instant;
use vulcans_trials::cuda::CudaEngine;
use vulcans_trials::vulkan::{GpuSaParams, PlacementBatchParams, VulkanEngine};

struct BenchStats {
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

fn measure_fn<F: FnMut()>(mut f: F, warmup: usize, iterations: usize) -> BenchStats {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t0 = Instant::now();
        f();
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed);
    }
    let sum: f64 = times.iter().sum();
    let mean_ms = sum / (times.len() as f64);
    let min_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = times.iter().cloned().fold(0.0, f64::max);
    BenchStats { mean_ms, min_ms, max_ms }
}

fn generate_obstacles(w: u32, h: u32, seed: u64) -> Vec<u32> {
    let mut obs = vec![0u32; (w * h) as usize];
    let mut rng = seed;
    // Generate scattered obstacle clusters representing components and traces
    for y in 0..h {
        for x in 0..w {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng >> 32) as u32;
            if (r % 100) < 15 {
                obs[(y * w + x) as usize] = 1;
            }
        }
    }
    obs
}

fn generate_placement_data(nc: usize, nf: usize, ns: usize) -> (Vec<[f32; 4]>, Vec<[f32; 4]>, PlacementBatchParams) {
    let mut boxes = Vec::with_capacity(nc * nf);
    let mut segs = Vec::with_capacity(nc * ns);
    let mut rng = 1337u64;

    for _ in 0..nc {
        for f in 0..nf {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bx = ((rng >> 32) as u32 % 180) as f32 + 5.0;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let by = ((rng >> 32) as u32 % 180) as f32 + 5.0;
            let w = 4.0 + ((f % 5) as f32);
            let h = 3.0 + ((f % 3) as f32);
            boxes.push([bx, by, bx + w, by + h]);
        }
        for s in 0..ns {
            let f1 = s % nf;
            let f2 = (s * 3 + 1) % nf;
            let b1 = boxes[boxes.len() - nf + f1];
            let b2 = boxes[boxes.len() - nf + f2];
            segs.push([b1[0] + 1.0, b1[1] + 1.0, b2[0] + 1.0, b2[1] + 1.0]);
        }
    }

    let params = PlacementBatchParams {
        num_candidates: nc as u32,
        num_footprints: nf as u32,
        num_segments: ns as u32,
        _pad: 0,
        board_min_x: 0.0,
        board_min_y: 0.0,
        board_max_x: 200.0,
        board_max_y: 200.0,
        wirelength_weight: 1.0,
        crossing_weight: 150.0,
        overlap_penalty: 20000.0,
        boundary_penalty: 30000.0,
    };

    (boxes, segs, params)
}

fn generate_sa_chains_data(nc: usize, nf: usize, ne: usize) -> (Vec<[f32; 4]>, Vec<[f32; 4]>, Vec<[f32; 4]>, GpuSaParams) {
    let mut fp_descs = Vec::with_capacity(nf);
    for i in 0..nf {
        let hw = 2.0 + ((i % 4) as f32) * 0.5;
        let hh = 1.5 + ((i % 3) as f32) * 0.5;
        let locked = if i == 0 || i == 1 { 1.0 } else { 0.0 };
        fp_descs.push([hw, hh, locked, 0.0]);
    }

    let mut edges = Vec::with_capacity(ne * 2);
    for e in 0..ne {
        let fpa = (e % nf) as f32;
        let fpb = ((e * 2 + 1) % nf) as f32;
        let is_tie = if e % 4 == 0 { 1.0 } else { 0.0 };
        edges.push([fpa, 0.5, 0.5, is_tie]);
        edges.push([fpb, -0.5, -0.5, 0.0]);
    }

    let mut init_poses = Vec::with_capacity(nc * nf);
    for _ in 0..nc {
        for f in 0..nf {
            let x = 10.0 + ((f % 10) as f32) * 8.0;
            let y = 10.0 + ((f / 10) as f32) * 8.0;
            init_poses.push([x, y, 0.0, 0.0]);
        }
    }

    let params = GpuSaParams {
        num_chains: nc as u32,
        num_footprints: nf as u32,
        num_edges: ne as u32,
        steps_per_pass: 8,
        seed: 0xDEADBEEF,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        board_min_x: 0.0,
        board_min_y: 0.0,
        board_max_x: 100.0,
        board_max_y: 100.0,
        grid_step: 0.5,
        wirelength_weight: 1.0,
        crossing_weight: 100.0,
        overlap_penalty: 10000.0,
        boundary_penalty: 20000.0,
        passive_affinity_weight: 50.0,
        passive_max_dist: 4.0,
        base_temp: 10.0,
    };

    (fp_descs, edges, init_poses, params)
}

fn main() {
    println!("================================================================================");
    println!("     Vulcan's Trials — Compute Benchmark: Vulkan WGSL vs Bare-Metal CUDA PTX     ");
    println!("                   Copyright (c) 2026 Alastair D'Silva                           ");
    println!("================================================================================\n");

    // 1. Initialization / Context Creation Benchmark
    println!("--- 1. Backend Initialization Latency ---");
    let vk_init = measure_fn(|| {
        let _ = VulkanEngine::new_vulkan_only();
    }, 1, 5);
    println!("  Vulkan (wgpu instance/adapter/device): {:.2} ms (min: {:.2} ms, max: {:.2} ms)",
        vk_init.mean_ms, vk_init.min_ms, vk_init.max_ms);

    let cu_init = measure_fn(|| {
        let _ = CudaEngine::new();
    }, 1, 5);
    println!("  CUDA PTX (libcuda.so / cuCtxCreate):   {:.2} ms (min: {:.2} ms, max: {:.2} ms)",
        cu_init.mean_ms, cu_init.min_ms, cu_init.max_ms);
    println!("  Speedup: {:.2}x faster initialization\n", vk_init.mean_ms / cu_init.mean_ms);

    let vk = VulkanEngine::new_vulkan_only().expect("Vulkan init");
    let cuda = CudaEngine::new().expect("CUDA init");
    println!("Active Devices:");
    println!("  Vulkan:   {} [{}] ({})", vk.device_name, vk.backend_name, vk.driver_info);
    println!("  CUDA PTX: {} [{}] ({})\n", cuda.device_name, cuda.backend_name, cuda.driver_info);

    // 2. Per-Dispatch Launch Overhead
    println!("--- 2. Per-Dispatch Launch Overhead (16x16 grid, single pass) ---");
    let obs_tiny = vec![0u32; 16 * 16];
    let vk_launch = measure_fn(|| {
        let _ = vk.compute_distance_field(&obs_tiny, 16, 16, 4);
    }, 5, 50);
    let cu_launch = measure_fn(|| {
        let _ = cuda.compute_distance_field(&obs_tiny, 16, 16, 4);
    }, 5, 50);
    println!("  Vulkan dispatch + sync:   {:.3} ms (min: {:.3} ms)", vk_launch.mean_ms, vk_launch.min_ms);
    println!("  CUDA PTX launch + sync:   {:.3} ms (min: {:.3} ms)", cu_launch.mean_ms, cu_launch.min_ms);
    println!("  Speedup: {:.2}x lower dispatch overhead\n", vk_launch.mean_ms / cu_launch.mean_ms);

    // 3. Distance Field Compute Throughput
    println!("--- 3. Distance Field Compute Throughput ---");
    let test_grids = [
        ("p1p2esp Board (250x205)", 250, 205, 12, 20),
        ("Medium Grid (512x512)", 512, 512, 12, 15),
        ("Large Grid (1024x1024, r=12)", 1024, 1024, 12, 10),
        ("Dense Grid (1024x1024, r=16)", 1024, 1024, 16, 8),
    ];

    println!("{:<28} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<8}",
        "Benchmark Grid", "Vulkan(ms)", "NVCC Dir", "NVCC Tile", "Amp Dir", "Amp Tile", "Speedup");
    println!("{:-<28}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<10}-+-{:-<8}",
        "", "", "", "", "", "", "");

    for (label, w, h, r, iters) in test_grids {
        let obs = generate_obstacles(w, h, 42);

        let vk_stat = measure_fn(|| {
            let _ = vk.compute_distance_field(&obs, w, h, r);
        }, 2, iters);

        let cu_dir_stat = measure_fn(|| {
            let _ = cuda.compute_distance_field_mode(&obs, w, h, r, false);
        }, 2, iters);

        let cu_tile_stat = measure_fn(|| {
            let _ = cuda.compute_distance_field_mode(&obs, w, h, r, true);
        }, 2, iters);

        let amp_dir_stat = measure_fn(|| {
            let _ = cuda.compute_distance_field_ampere(&obs, w, h, r, false);
        }, 2, iters);

        let amp_tile_stat = measure_fn(|| {
            let _ = cuda.compute_distance_field_ampere(&obs, w, h, r, true);
        }, 2, iters);

        let best_amp = amp_dir_stat.mean_ms.min(amp_tile_stat.mean_ms);
        let speedup = vk_stat.mean_ms / best_amp;

        println!("{:<28} | {:<10.3} | {:<10.3} | {:<10.3} | {:<10.3} | {:<10.3} | {:<7.2}x",
            label, vk_stat.mean_ms, cu_dir_stat.mean_ms, cu_tile_stat.mean_ms,
            amp_dir_stat.mean_ms, amp_tile_stat.mean_ms, speedup);
    }
    println!();

    // 4. Batch Placement Evaluation Throughput
    println!("--- 4. Batch Placement Energy Evaluation Throughput ---");
    let placement_configs = [
        ("Small Batch (64 cand, 50 fp, 120 seg)", 64, 50, 120, 20),
        ("Medium Batch (256 cand, 100 fp, 250 seg)", 256, 100, 250, 15),
        ("Large Batch (1024 cand, 100 fp, 250 seg)", 1024, 100, 250, 10),
    ];

    println!("{:<42} | {:<12} | {:<12} | {:<12} | {:<8}",
        "Placement Batch", "Vulkan (ms)", "NVCC PTX(ms)", "Ampere PTX(ms)", "Speedup");
    println!("{:-<42}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<8}",
        "", "", "", "", "");

    for (label, nc, nf, ns, iters) in placement_configs {
        let (boxes, segs, params) = generate_placement_data(nc, nf, ns);
        let vk_stat = measure_fn(|| {
            let _ = vk.evaluate_placement_batch(&boxes, &segs, &params);
        }, 2, iters);

        let cu_stat = measure_fn(|| {
            let _ = cuda.evaluate_placement_batch(&boxes, &segs, &params);
        }, 2, iters);

        let amp_stat = measure_fn(|| {
            let _ = cuda.evaluate_placement_ampere(&boxes, &segs, &params);
        }, 2, iters);

        let speedup = vk_stat.mean_ms / amp_stat.mean_ms;

        println!("{:<42} | {:<12.3} | {:<12.3} | {:<15.3} | {:<7.2}x",
            label, vk_stat.mean_ms, cu_stat.mean_ms, amp_stat.mean_ms, speedup);
    }
    println!();

    // 5. GPU SA Placer Chains Throughput
    println!("--- 5. GPU SA Placer Chains Throughput (64 chains, 60 footprints, 150 edges) ---");
    let (fp_descs, edges, init_poses, sa_params) = generate_sa_chains_data(64, 60, 150);
    let vk_sa = measure_fn(|| {
        let _ = vk.run_gpu_sa_placer_chains(&fp_descs, &edges, &init_poses, &sa_params, 4, None);
    }, 2, 10);
    let cu_sa = measure_fn(|| {
        let _ = cuda.run_gpu_sa_placer_chains(&fp_descs, &edges, &init_poses, &sa_params, 4, None);
    }, 2, 10);
    let sa_speedup = vk_sa.mean_ms / cu_sa.mean_ms;
    println!("  Vulkan SA Chains:   {:.3} ms", vk_sa.mean_ms);
    println!("  CUDA PTX SA Chains: {:.3} ms", cu_sa.mean_ms);
    println!("  Speedup: {:.2}x faster Simulated Annealing sampling\n", sa_speedup);

    println!("================================================================================");
    println!("                             Benchmark Complete                                 ");
    println!("================================================================================");
}
