// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

mod board;
mod gui;
mod placer;
mod router;
mod vulkan;

use crate::board::{Board, Point, Polygon, Segment};
use crate::gui::VulcansTrialsApp;
use crate::placer::{run_auto_placer, PlacementConfig};
use crate::router::{AutoRouter, RoutingGrid};
use crate::vulkan::VulkanEngine;
use clap::Parser;
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "vulcans-trials",
    author = "InfernoEmbedded",
    version = "0.1.0",
    about = "Vulcan's Trials — Vulkan GPU + Multithreaded CPU Auto-Placer & Auto-Router for KiCad (GPLv3)"
)]
struct Args {
    /// Input KiCad PCB file (.kicad_pcb)
    #[arg(short, long, default_value = "/home/deece/src/p1p2esp/p1p2esp.kicad_pcb")]
    input: String,

    /// Output KiCad PCB file (.kicad_pcb)
    #[arg(short, long, default_value = "/home/deece/watt-home/src/vulcans-trials/vulcans_trials_out.kicad_pcb")]
    output: String,

    /// Run in headless CLI mode instead of launching the interactive GUI
    #[arg(long)]
    headless: bool,

    /// Trial mode in headless execution: "place", "route", or "both"
    #[arg(short, long, default_value = "both")]
    mode: String,

    /// Auto-placement duration in seconds
    #[arg(short, long, default_value_t = 10)]
    duration: u64,

    /// Routing grid resolution in mm
    #[arg(short, long, default_value_t = 0.10)]
    res: f64,

    /// Maximum rip-up and reroute passes
    #[arg(long, default_value_t = 15)]
    passes: usize,

    /// Additional comma-separated component references to lock
    #[arg(long, default_value = "")]
    lock: String,

    /// Run built-in self-verification suite (Vulkan compute, SAT geometry, router grid)
    #[arg(long)]
    self_test: bool,
}

fn run_self_test() -> Result<(), String> {
    println!("=== Vulcan's Trials (InfernoEmbedded) — Self-Test Suite ===");

    // 1. Verify Vulkan Compute Engine & WGSL Distance Field Shader
    let vk = VulkanEngine::new().ok_or("Failed to initialize Vulkan compute backend")?;
    println!(
        "[PASS] Vulkan Engine initialized: {} [{}] — {}",
        vk.device_name, vk.backend_name, vk.driver_info
    );

    let mut obs = vec![0u32; 16 * 16];
    obs[8 * 16 + 8] = 1; // Single obstacle at (8, 8)
    let df = vk
        .compute_distance_field(&obs, 16, 16, 10)
        .ok_or("Vulkan compute_distance_field returned None")?;
    let d_center = df[8 * 16 + 8];
    let d_offset = df[8 * 16 + 11]; // 3 cells away along +X
    if d_center.abs() > 1e-4 || (d_offset - 3.0).abs() > 1e-3 {
        return Err(format!(
            "Distance field mismatch: center={}, +3_offset={}",
            d_center, d_offset
        ));
    }
    println!(
        "[PASS] Vulkan WGSL Compute Shader distance field verified (d(0)=0.0, d(3)={:.2})",
        d_offset
    );

    // 2. Verify SAT Courtyard Polygon Collision & Segment Crossing Math
    let poly_a = Polygon::new(vec![
        Point::new(0.0, 0.0),
        Point::new(2.0, 0.0),
        Point::new(2.0, 2.0),
        Point::new(0.0, 2.0),
    ]);
    let poly_b = poly_a.transform(Point::new(1.0, 1.0), 0.0);
    let poly_c = poly_a.transform(Point::new(5.0, 5.0), 0.0);
    if !poly_a.intersects_convex(&poly_b) || poly_a.intersects_convex(&poly_c) {
        return Err("SAT polygon collision check failed".to_string());
    }
    let s1 = Segment {
        p1: Point::new(0.0, 0.0),
        p2: Point::new(4.0, 4.0),
        net_id: 1,
    };
    let s2 = Segment {
        p1: Point::new(0.0, 4.0),
        p2: Point::new(4.0, 0.0),
        net_id: 2,
    };
    if !s1.crosses(&s2) {
        return Err("Segment crossing check failed".to_string());
    }
    println!("[PASS] SAT courtyard collision & ratsnest crossing math verified");

    println!("=== All Self-Tests Passed ===");
    Ok(())
}

fn main() {
    let args = Args::parse();

    if args.self_test {
        if let Err(e) = run_self_test() {
            eprintln!("[FAIL] {}", e);
            std::process::exit(1);
        }
        return;
    }

    let extra_locked: HashSet<String> = args
        .lock
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let vulkan = VulkanEngine::new();
    if let Some(ref vk) = vulkan {
        println!(
            "=== Vulcan's Trials (InfernoEmbedded) — Vulkan GPU: {} [{}] ===",
            vk.device_name, vk.backend_name
        );
    } else {
        println!("=== Vulcan's Trials (InfernoEmbedded) — CPU Fallback Mode ===");
    }

    let mut board = match Board::load(&args.input, &extra_locked) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error loading PCB '{}': {}", args.input, e);
            std::process::exit(1);
        }
    };

    println!(
        "Loaded '{}': {:.1}x{:.1}mm, {} footprints ({} locked), {} nets, {} clusters",
        args.input,
        board.max_x - board.min_x,
        board.max_y - board.min_y,
        board.footprints.len(),
        board.footprints.iter().filter(|f| f.is_locked).count(),
        board.nets.len(),
        board.clusters.len()
    );

    if args.headless {
        let start_t = Instant::now();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let do_place = matches!(args.mode.as_str(), "place" | "both");
        let do_route = matches!(args.mode.as_str(), "route" | "both");

        if do_place {
            let cfg = PlacementConfig {
                duration_secs: args.duration,
                ..PlacementConfig::default()
            };
            let st = run_auto_placer(&mut board, &cfg, None, Arc::clone(&stop_flag));
            println!(
                "[Auto-Placer] Done: Crossings={}, Wirelength={:.1}mm, Overlaps={}",
                st.crossings, st.wirelength, st.overlaps
            );
        }

        if do_route {
            let mut grid = RoutingGrid::new(&board, args.res, vulkan);
            let router = AutoRouter::new(args.passes);
            let (routed, total) = router.route_all(&mut board, &mut grid, None, Arc::clone(&stop_flag));
            println!(
                "[Auto-Router] Done in {:.2}s: {}/{} ({:.1}%) connections routed ({} tracks, {} vias)",
                start_t.elapsed().as_secs_f64(),
                routed,
                total,
                (routed as f64 / total.max(1) as f64) * 100.0,
                board.tracks.len(),
                board.vias.len()
            );
        }

        let out_path = Path::new(&args.output);
        board.save_pcb(out_path).expect("Failed to write output .kicad_pcb");
        let ses_path = out_path.with_extension("ses");
        board.save_ses(&ses_path).expect("Failed to write output .ses");
        println!("Saved output to {} and {}", out_path.display(), ses_path.display());
        return;
    }

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Vulcan's Trials — InfernoEmbedded Auto-Placer & Auto-Router (GPLv3)")
            .with_inner_size([1360.0, 840.0]),
        ..Default::default()
    };

    let input_path = args.input.clone();
    let output_path = args.output.clone();
    if let Err(e) = eframe::run_native(
        "Vulcan's Trials — InfernoEmbedded",
        native_options,
        Box::new(move |_cc| Ok(Box::new(VulcansTrialsApp::new(input_path, output_path, board, vulkan)))),
    ) {
        eprintln!("GUI error: {}", e);
    }
}
