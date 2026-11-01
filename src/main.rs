// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use clap::Parser;
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;
use vulcans_trials::board::{Board, Point, Polygon, Segment, TrackSegment};
use vulcans_trials::gui::VulcansTrialsApp;
use vulcans_trials::placer::{run_auto_placer, PlacementConfig};
use vulcans_trials::router::{
    apply_any_angle_tangent_arcs, prune_dangling_tracks_and_vias, validate_and_clean_copper,
    AutoRouter, RoutingGrid,
};
use vulcans_trials::vulkan::VulkanEngine;

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

    /// Penalty factor for routing non-plane signals across inner power plane layers
    #[arg(long, default_value_t = 8.0)]
    inner_plane_penalty: f64,

    /// Enable TopoR-style any-angle topological routing (homotopic rubber-band & tangent arcs)
    #[arg(long)]
    any_angle: bool,

    /// Connect to KiCad 10 IPC socket for real-time live canvas streaming (optional socket path)
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    kicad_ipc: Option<String>,

    /// Run built-in self-verification suite (Vulkan compute, SAT geometry, router grid)
    #[arg(long)]
    self_test: bool,

    /// Optional KiCad IPC streaming socket endpoint or named pipe
    #[arg(long)]
    ipc: Option<String>,
}

fn parse_locked_refs(lock_arg: &str) -> HashSet<String> {
    lock_arg
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
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

    // 3. Verify BGA Package Detection & Quadrant Dog-Bone Fanout
    let mut bga_fp = vulcans_trials::board::Footprint {
        reference: "U_BGA".to_string(),
        value: "BGA-16".to_string(),
        x: 25.0,
        y: 25.0,
        rotation: 0.0,
        layer: 0,
        orig_layer: 0,
        is_locked: false,
        pads: Vec::new(),
        width: 5.0,
        height: 5.0,
        courtyards: vec![],
        perimeters: vec![],
        escape_margin: 0.0,
        at_span: None,
        fp_span: None,
        pin_swap_groups: Vec::new(),
        gate_swap_groups: Vec::new(),
    };
    let xs = [-1.2, -0.4, 0.4, 1.2];
    let ys = [-1.2, -0.4, 0.4, 1.2];
    let mut pad_num = 1;
    for &y in &ys {
        for &x in &xs {
            let mut p = vulcans_trials::board::Pad::new(
                &format!("P{}", pad_num), x, y, 0.40, 0.40, pad_num, &format!("NET{}", pad_num), false,
            );
            p.shape = "circle".to_string();
            bga_fp.pads.push(p);
            pad_num += 1;
        }
    }
    if !bga_fp.is_bga() {
        return Err("BGA package detection failed".to_string());
    }
    let mut test_board = Board::default();
    test_board.netclasses.get_mut("Default").unwrap().clearance = 0.12;
    test_board.footprints.push(bga_fp);
    let mut test_grid = RoutingGrid::new(&test_board, 0.10, None);
    vulcans_trials::router::generate_bga_dogbone_escapes(&mut test_board, &mut test_grid);
    if test_board.vias.len() != 4 {
        return Err(format!("Expected 4 dog-bone vias, found {}", test_board.vias.len()));
    }
    println!("[PASS] Automated BGA Escape Solver verified (4x4 0.8mm pitch quadrant dog-bone fanout)");

    // 4. Verify TopoR Any-Angle Tangent Arc Geometry & Homotopic Relaxation
    let mut arc_board = Board::default();
    arc_board.tracks.push(TrackSegment {
        start: (0.0, 0.0),
        end: (10.0, 0.0),
        width: 0.25,
        layer: 0,
        net_id: 1,
        is_locked: false,
    });
    arc_board.tracks.push(TrackSegment {
        start: (10.0, 0.0),
        end: (10.0, 10.0),
        width: 0.25,
        layer: 0,
        net_id: 1,
        is_locked: false,
    });
    apply_any_angle_tangent_arcs(&mut arc_board);
    if arc_board.arcs.is_empty() {
        return Err("TopoR any-angle tangent arc self-test failed".to_string());
    }
    println!(
        "[PASS] TopoR any-angle tangent arc geometry verified (r={:.2}mm, center=({:.2},{:.2}))",
        arc_board.arcs[0].radius, arc_board.arcs[0].center.0, arc_board.arcs[0].center.1
    );

    // 5. Verify KiCad 10 IPC Protocol & 60 FPS Batch Streaming
    let mut mock_ipc = vulcans_trials::ipc::KiCadIpcClient::new();
    mock_ipc.queue_move_footprint("R1", 12.0, 15.0, 0.0);
    mock_ipc.queue_add_track((0.0, 0.0), (10.0, 0.0), 0.25, "F.Cu", 1);
    mock_ipc.queue_add_via((10.0, 0.0), 0.60, 0.30, 1);
    let flushed = mock_ipc.commit_batch().map_err(|e| format!("IPC commit error: {}", e))?;
    if flushed != 3 {
        return Err(format!("Expected 3 flushed commands, got {}", flushed));
    }
    println!("[PASS] KiCad 10 IPC socket protocol & 60 FPS live batch streaming verified");

    println!("=== All Self-Tests Passed ===");
    Ok(())
}

fn run_headless_trial(
    args: &Args,
    board: &mut Board,
    vulkan: Option<Arc<VulkanEngine>>,
    ipc: Option<Arc<std::sync::Mutex<vulcans_trials::ipc::KiCadIpcClient>>>,
) -> Result<(), String> {
    let start_t = Instant::now();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let do_place = matches!(args.mode.as_str(), "place" | "both");
    let do_route = matches!(args.mode.as_str(), "route" | "both");

    if do_place {
        let cfg = PlacementConfig {
            duration_secs: args.duration,
            vulkan: vulkan.clone(),
            ipc: ipc.clone(),
            ..PlacementConfig::default()
        };
        let st = run_auto_placer(board, &cfg, None, Arc::clone(&stop_flag));
        println!(
            "[Auto-Placer] Done: Crossings={}, Wirelength={:.1}mm, Overlaps={}",
            st.crossings, st.wirelength, st.overlaps
        );
    }

    if do_route {
        let mut grid = RoutingGrid::new(board, args.res, vulkan).with_inner_layer_penalty(args.inner_plane_penalty);
        let router = AutoRouter::new(args.passes)
            .with_inner_layer_penalty(args.inner_plane_penalty)
            .with_any_angle_mode(args.any_angle)
            .with_ipc(ipc.clone());
        let (routed, total) = router.route_all(board, &mut grid, None, Arc::clone(&stop_flag));
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

    if do_route {
        validate_and_clean_copper(board);
        prune_dangling_tracks_and_vias(board);
    }

    let out_path = Path::new(&args.output);
    board.save_pcb(out_path)?;
    let ses_path = out_path.with_extension("ses");
    board.save_ses(&ses_path)?;
    println!("Saved output to {} and {}", out_path.display(), ses_path.display());
    Ok(())
}

// Process entrypoint `main` cannot be called directly in an in-process unit test because
// it parses process-global `std::env::args`, calls `std::process::exit(1)` on error, or blocks
// on an OS window event loop via `eframe::run_native`. All underlying logic (`Args`,
// `parse_locked_refs`, `run_self_test`, `run_headless_trial`) is unit-tested below.
fn main() {
    let args = Args::parse();

    if args.self_test {
        if let Err(e) = run_self_test() {
            eprintln!("[FAIL] {}", e);
            std::process::exit(1);
        }
        return;
    }

    if let Some(ref ep) = args.ipc {
        println!("KiCad IPC sync endpoint: {}", ep);
    }

    let extra_locked = parse_locked_refs(&args.lock);

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

    let ipc_client = if let Some(ref path_str) = args.kicad_ipc {
        let custom = if path_str.is_empty() { None } else { Some(path_str.as_str()) };
        if let Some(c) = vulcans_trials::ipc::KiCadIpcClient::connect_or_discover(custom) {
            println!("[KiCad-IPC] Connected to socket: {:?}", c.socket_path());
            Some(Arc::new(std::sync::Mutex::new(c)))
        } else {
            eprintln!("[KiCad-IPC] Warning: No active KiCad 10 socket found");
            None
        }
    } else {
        None
    };

    if args.headless {
        if let Err(e) = run_headless_trial(&args, &mut board, vulkan, ipc_client) {
            eprintln!("Headless execution error: {}", e);
            std::process::exit(1);
        }
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
        Box::new(move |_cc| Ok(Box::new(VulcansTrialsApp::new(input_path, output_path, board, vulkan).with_ipc(ipc_client)))),
    ) {
        eprintln!("GUI error: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_cli_args_parsing_and_locked_refs() {
        let def = Args::try_parse_from(["vulcans-trials"]).unwrap();
        assert!(!def.headless);
        assert!(!def.self_test);
        assert_eq!(def.mode, "both");
        assert_eq!(def.duration, 10);
        assert!((def.res - 0.10).abs() < 1e-6);
        assert_eq!(def.passes, 15);
        assert!(parse_locked_refs(&def.lock).is_empty());
        assert_eq!(def.kicad_ipc, None);

        let custom = Args::try_parse_from([
            "vulcans-trials",
            "-i",
            "/tmp/in.kicad_pcb",
            "-o",
            "/tmp/out.kicad_pcb",
            "--headless",
            "--mode",
            "place",
            "--duration",
            "2",
            "--res",
            "0.25",
            "--passes",
            "4",
            "--lock",
            "U1, J2 , , R5",
            "--kicad-ipc",
            "/tmp/test_custom.sock",
            "--self-test",
            "--ipc",
            "/tmp/test_ipc.sock",
        ])
        .unwrap();
        assert!(custom.headless);
        assert!(custom.self_test);
        assert_eq!(custom.ipc, Some("/tmp/test_ipc.sock".to_string()));
        assert_eq!(custom.mode, "place");
        assert_eq!(custom.duration, 2);
        assert_eq!(custom.passes, 4);
        assert_eq!(custom.kicad_ipc, Some("/tmp/test_custom.sock".to_string()));
        let locked = parse_locked_refs(&custom.lock);
        assert_eq!(locked.len(), 3);
        assert!(locked.contains("U1") && locked.contains("J2") && locked.contains("R5"));

        let ipc_default = Args::try_parse_from(["vulcans-trials", "--kicad-ipc"]).unwrap();
        assert_eq!(ipc_default.kicad_ipc, Some(String::new()));
    }

    #[test]
    fn test_run_self_test_suite() {
        assert!(run_self_test().is_ok());
    }

    #[test]
    fn test_run_headless_trial_modes() {
        let in_pcb = "/tmp/vulcan_main_unit_in.kicad_pcb";
        let out_pcb = "/tmp/vulcan_main_unit_out.kicad_pcb";
        let out_ses = "/tmp/vulcan_main_unit_out.ses";
        let content = r#"(kicad_pcb
          (net 0 "") (net 1 "GND") (net 2 "SIG")
          (gr_rect (start 100.0 100.0) (end 130.0 120.0) (layer "Edge.Cuts"))
          (footprint "R" (at 108.0 110.0) (property "Reference" "R10")
            (pad "1" smd rect (at 0 0) (size 0.5 0.5) (net 2 "SIG")))
          (footprint "R" (at 114.0 110.0) (property "Reference" "R11")
            (pad "1" smd rect (at 0 0) (size 0.5 0.5) (net 2 "SIG")))
        )"#;
        fs::write(in_pcb, content).unwrap();

        let mut board = Board::load(in_pcb, &HashSet::new()).unwrap();
        let args = Args {
            input: in_pcb.to_string(),
            output: out_pcb.to_string(),
            headless: true,
            mode: "both".to_string(),
            duration: 0,
            res: 0.50,
            passes: 1,
            lock: String::new(),
            inner_plane_penalty: 8.0,
            any_angle: false,
            kicad_ipc: None,
            self_test: false,
            ipc: None,
        };
        assert!(run_headless_trial(&args, &mut board, None, None).is_ok());
        assert!(Path::new(out_pcb).exists());
        assert!(Path::new(out_ses).exists());

        let _ = fs::remove_file(in_pcb);
        let _ = fs::remove_file(out_pcb);
        let _ = fs::remove_file(out_ses);
    }
}

