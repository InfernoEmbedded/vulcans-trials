// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use vulcans_trials::board::{Board, LiveTrialState, Point, Segment, TrialPhase, DEFAULT_LOCKED_REFS};
use vulcans_trials::placer::{legalize_placement, run_auto_placer, PlacementConfig, PlacementState};
use vulcans_trials::router::{AutoRouter, GridNode, NudgeEngine, RoutingGrid, FREE_NET, KEEPOUT_NET};
use vulcans_trials::vulkan::{SystemTelemetry, VulkanEngine};

fn temp_test_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vulcans_trials_it_{}_{}", std::process::id(), tag));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create temp test directory");
    dir
}

/// Writes a self-contained multi-component `.kicad_pcb` and sibling `.kicad_sch` fixture
/// exercising netclasses, keepout zones, locked/unlocked footprints, overlapping courtyards,
/// fine-pitch U1/J2 escape stubs, and GND stitching targets.
fn write_synthetic_project(dir: &Path, with_overlap: bool) -> (PathBuf, PathBuf) {
    let pcb_path = dir.join("synthetic_board.kicad_pcb");
    let sch_path = dir.join("synthetic_board.kicad_sch");

    let (r3_x, r3_y) = if with_overlap {
        (114.2, 110.0) // Overlaps R4 at (114.5, 110.0)
    } else {
        (114.0, 105.0)
    };

    let pcb_content = format!(
        r#"(kicad_pcb (version 20240108) (generator "vulcans_trials_test")
  (net 0 "")
  (net 1 "GND")
  (net 2 "VRAW")
  (net 3 "+3.3V")
  (net 4 "VBUS")
  (net 5 "Net-(U2-SW)")
  (net 6 "HPA")
  (net 7 "USB_D+")
  (net 8 "SIG_A")
  (gr_rect (start 100.0 98.0) (end 136.0 126.0) (layer "Edge.Cuts"))
  (zone (keepout (tracks not_allowed) (vias not_allowed))
    (polygon (pts (xy 131.0 99.0) (xy 135.0 99.0) (xy 135.0 103.0) (xy 131.0 103.0)))
  )
  (footprint "Connector:Conn_01x02" (layer "F.Cu")
    (at 104.0 104.0)
    (property "Reference" "J1")
    (property "Value" "Conn_Power")
    (fp_rect (start -1.5 -1.5) (end 1.5 1.5) (layer "F.CrtYd"))
    (pad "1" thru_hole circle (at 0.0 -0.8) (size 1.2 1.2) (layers "*.Cu") (net 2 "VRAW"))
    (pad "2" thru_hole circle (at 0.0 0.8) (size 1.2 1.2) (layers "*.Cu") (net 1 "GND"))
  )
  (footprint "Connector_USB:USB_C_Receptacle" (layer "F.Cu")
    (at 104.0 118.0)
    (property "Reference" "J2")
    (property "Value" "USB_C")
    (fp_rect (start -2.0 -2.5) (end 2.0 2.5) (layer "F.CrtYd"))
    (pad "A5" smd rect (at 1.0 -0.5) (size 0.6 0.3) (layers "F.Cu") (net 7 "USB_D+"))
    (pad "B5" smd rect (at 1.0 0.5) (size 0.6 0.3) (layers "F.Cu") (net 4 "VBUS"))
    (pad "S1" thru_hole oval (at -1.0 0.0) (size 1.0 1.5) (layers "*.Cu") (net 1 "GND"))
  )
  (footprint "Package_DFN_QFN:QFN-24" (layer "F.Cu")
    (at 122.0 116.0)
    (property "Reference" "U1")
    (property "Value" "MAX22088")
    (locked yes)
    (fp_rect (start -2.2 -2.2) (end 2.2 2.2) (layer "F.CrtYd"))
    (pad "1" smd rect (at 1.8 -0.5) (size 0.7 0.25) (layers "F.Cu") (net 6 "HPA"))
    (pad "2" smd rect (at 1.8 0.0) (size 0.7 0.25) (layers "F.Cu") (net 8 "SIG_A"))
    (pad "15" smd rect (at -1.8 0.0) (size 0.7 0.25) (layers "F.Cu") (net 1 "GND"))
  )
  (footprint "Package_SO:SOIC-8" (layer "F.Cu")
    (at 120.0 104.0)
    (property "Reference" "U2")
    (property "Value" "BUCK_IC")
    (fp_rect (start -1.8 -1.8) (end 1.8 1.8) (layer "F.CrtYd"))
    (pad "1" smd rect (at -1.0 -0.6) (size 0.8 0.4) (layers "F.Cu") (net 2 "VRAW"))
    (pad "2" smd rect (at 1.0 -0.6) (size 0.8 0.4) (layers "F.Cu") (net 5 "Net-(U2-SW)"))
    (pad "3" smd rect (at 1.0 0.6) (size 0.8 0.4) (layers "F.Cu") (net 3 "+3.3V"))
  )
  (footprint "Resistor_SMD:R_0603" (layer "F.Cu")
    (at {r3_x:.4} {r3_y:.4} 90.0)
    (property "Reference" "R3")
    (property "Value" "10k")
    (fp_rect (start -1.0 -0.8) (end 1.0 0.8) (layer "F.CrtYd"))
    (pad "1" smd rect (at -0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 6 "HPA"))
    (pad "2" smd rect (at 0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 3 "+3.3V"))
  )
  (footprint "Resistor_SMD:R_0603" (layer "F.Cu")
    (at 114.5 110.0)
    (property "Reference" "R4")
    (property "Value" "4.7k")
    (fp_rect (start -1.0 -0.8) (end 1.0 0.8) (layer "F.CrtYd"))
    (pad "1" smd rect (at -0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 8 "SIG_A"))
    (pad "2" smd rect (at 0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 7 "USB_D+"))
  )
  (footprint "Capacitor_SMD:C_0603" (layer "F.Cu")
    (at 127.0 104.0)
    (property "Reference" "C5")
    (property "Value" "100nF")
    (fp_rect (start -1.0 -0.8) (end 1.0 0.8) (layer "F.CrtYd"))
    (pad "1" smd rect (at -0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 5 "Net-(U2-SW)"))
    (pad "2" smd rect (at 0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 4 "VBUS"))
  )
  (footprint "Capacitor_SMD:C_0603" (layer "F.Cu")
    (at 128.0 110.0)
    (property "Reference" "C14")
    (property "Value" "1uF")
    (fp_rect (start -1.0 -0.8) (end 1.0 0.8) (layer "F.CrtYd"))
    (pad "1" smd rect (at -0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 3 "+3.3V"))
    (pad "2" smd rect (at 0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 1 "GND"))
  )
  (footprint "Resistor_SMD:R_0603" (layer "F.Cu")
    (at 109.0 104.0)
    (property "Reference" "R14")
    (property "Value" "100k")
    (fp_rect (start -1.0 -0.8) (end 1.0 0.8) (layer "F.CrtYd"))
    (pad "1" smd rect (at -0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 1 "GND"))
    (pad "2" smd rect (at 0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 2 "VRAW"))
  )
  (footprint "Capacitor_SMD:C_0603" (layer "F.Cu")
    (at 112.0 120.0)
    (property "Reference" "C9")
    (property "Value" "10uF")
    (fp_rect (start -1.0 -0.8) (end 1.0 0.8) (layer "F.CrtYd"))
    (pad "1" smd rect (at -0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 1 "GND"))
    (pad "2" smd rect (at 0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 4 "VBUS"))
  )
)
"#
    );

    let sch_content = r#"(kicad_sch (version 20231120) (generator "vulcans_trials_test")
  (rectangle (start 35.0 35.0) (end 65.0 60.0) (stroke (width 0.254) (type default)) (fill (type none)))
  (text "1. BUCK POWER SUPPLY" (at 40.0 40.0 0))
  (rectangle (start 175.0 35.0) (end 205.0 65.0) (stroke (width 0.254) (type default)) (fill (type none)))
  (text "2. MAX22088 TRANSCEIVER" (at 180.0 40.0 0))
  (rectangle (start 35.0 155.0) (end 65.0 185.0) (stroke (width 0.254) (type default)) (fill (type none)))
  (text "3. USB INPUT & SURGE" (at 40.0 160.0 0))
  (symbol (lib_id "Regulator_Switching:BUCK") (at 48.0 48.0 0)
    (property "Reference" "U2" (at 48.0 45.0 0))
  )
  (symbol (lib_id "Device:C") (at 55.0 50.0 0)
    (property "Reference" "C5" (at 55.0 47.0 0))
  )
  (symbol (lib_id "Device:R") (at 58.0 52.0 0)
    (property "Reference" "R14" (at 58.0 49.0 0))
  )
  (symbol (lib_id "Interface:MAX22088") (at 190.0 50.0 0)
    (property "Reference" "U1" (at 190.0 47.0 0))
  )
  (symbol (lib_id "Device:R") (at 195.0 55.0 0)
    (property "Reference" "R3" (at 195.0 52.0 0))
  )
  (symbol (lib_id "Device:R") (at 198.0 58.0 0)
    (property "Reference" "R4" (at 198.0 55.0 0))
  )
  (symbol (lib_id "Connector:USB_C") (at 50.0 170.0 0)
    (property "Reference" "J2" (at 50.0 167.0 0))
  )
  (symbol (lib_id "Device:C") (at 55.0 175.0 0)
    (property "Reference" "C9" (at 55.0 172.0 0))
  )
)
"#;

    fs::write(&pcb_path, pcb_content).expect("Failed to write synthetic .kicad_pcb");
    fs::write(&sch_path, sch_content).expect("Failed to write synthetic .kicad_sch");
    (pcb_path, sch_path)
}

#[test]
fn test_board_and_schematic_loading_netclasses_and_clusters() {
    let dir = temp_test_dir("load_sch_clusters");
    let (pcb_path, _) = write_synthetic_project(&dir, false);

    let mut extra_locked = HashSet::new();
    extra_locked.insert("U2".to_string());

    let board = Board::load(pcb_path.to_str().unwrap(), &extra_locked).expect("Failed to load synthetic board");

    // Verify footprints, locking rules, and board outline margin (+/- 1.5mm around Edge.Cuts 100..136 x 98..126)
    assert_eq!(board.footprints.len(), 10);
    assert!((board.min_x - 98.5).abs() < 1e-4);
    assert!((board.max_x - 137.5).abs() < 1e-4);
    assert!((board.min_y - 96.5).abs() < 1e-4);
    assert!((board.max_y - 127.5).abs() < 1e-4);
    assert_eq!(board.keepouts.len(), 1);

    // J1 and J2 are in DEFAULT_LOCKED_REFS; U1 has (locked yes); U2 is in extra_locked; R3 is movable
    assert!(DEFAULT_LOCKED_REFS.contains(&"J1"));
    let find_fp = |r: &str| board.footprints.iter().find(|f| f.reference == r).unwrap();
    assert!(find_fp("J1").is_locked);
    assert!(find_fp("J2").is_locked);
    assert!(find_fp("U1").is_locked);
    assert!(find_fp("U2").is_locked);
    assert!(!find_fp("R3").is_locked);

    // Verify NetClass classification and parameters
    let check_nc = |net_name: &str, exp_class: &str, exp_w: f64, exp_c: f64| {
        let net_id = *board.net_name_to_id.get(net_name).expect("Net should exist");
        let nc = board.get_netclass_for_net(net_id);
        assert_eq!(nc.name, exp_class, "Net {} class mismatch", net_name);
        assert!((nc.track_width - exp_w).abs() < 1e-4);
        assert!((nc.clearance - exp_c).abs() < 1e-4);
    };
    check_nc("VRAW", "Power_15V", 0.60, 0.25);
    check_nc("+3.3V", "Power_3V3", 0.50, 0.25);
    check_nc("VBUS", "Power_5V", 0.35, 0.20);
    check_nc("Net-(U2-SW)", "Switching", 0.60, 0.25);
    check_nc("HPA", "HBS_Bus", 0.20, 0.20);
    check_nc("USB_D+", "USB_Diff", 0.20, 0.20);
    check_nc("SIG_A", "Default", 0.20, 0.20);

    // Verify sibling .kicad_sch cluster extraction and anchor assignment
    assert_eq!(board.clusters.len(), 3, "Should detect 3 functional schematic clusters");
    let buck_cluster = board
        .clusters
        .iter()
        .find(|c| c.name.contains("BUCK"))
        .expect("BUCK cluster missing");
    assert!((buck_cluster.max_radius - 4.5).abs() < 1e-4);
    assert!((buck_cluster.weight - 150.0).abs() < 1e-4);
    let u2_idx = board.footprints.iter().position(|f| f.reference == "U2").unwrap();
    assert_eq!(buck_cluster.anchor_idx, Some(u2_idx));

    let xcvr_cluster = board
        .clusters
        .iter()
        .find(|c| c.name.contains("MAX22088"))
        .expect("MAX22088 cluster missing");
    let u1_idx = board.footprints.iter().position(|f| f.reference == "U1").unwrap();
    assert_eq!(xcvr_cluster.anchor_idx, Some(u1_idx));
    assert_eq!(xcvr_cluster.member_indices.len(), 3);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_vulkan_gpu_distance_field_on_routing_grid_and_telemetry() {
    let dir = temp_test_dir("vulkan_df");
    let (pcb_path, _) = write_synthetic_project(&dir, false);
    let board = Board::load(pcb_path.to_str().unwrap(), &HashSet::new()).unwrap();

    let vk = VulkanEngine::new();
    if let Some(ref vk_eng) = vk {
        assert!(!vk_eng.device_name.is_empty());
        assert!(!vk_eng.backend_name.is_empty());
    }

    let grid = RoutingGrid::new(&board, 0.20, vk.clone());
    assert!(grid.width > 100 && grid.height > 100);
    let layer_cells = grid.width * grid.height;
    assert_eq!(grid.dist_field.len(), 2 * layer_cells);

    // Every rasterized obstacle cell on F.Cu must have distance field == 0.0,
    // and free cells far from obstacles must have positive distance field values.
    let mut obstacle_count = 0usize;
    let mut positive_df_count = 0usize;
    for idx in 0..layer_cells {
        if grid.obstacle_net_ids[idx] != FREE_NET {
            obstacle_count += 1;
            assert!(
                grid.dist_field[idx].abs() < 1e-4,
                "Obstacle cell {} must have distance field 0.0, got {}",
                idx,
                grid.dist_field[idx]
            );
        } else if grid.dist_field[idx] > 0.15 {
            positive_df_count += 1;
        }
    }
    assert!(obstacle_count > 10, "Board obstacle grid must contain rasterized pads/keepouts");
    assert!(positive_df_count > 100, "Free cells must have positive Euclidean distances");
    if let Some(ref vk_eng) = vk {
        assert!(
            vk_eng.compute_busy_ns.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "Vulkan compute telemetry counter must record GPU execution time"
        );
    }

    // Verify keepout zone cells were marked as KEEPOUT_NET (-1) on both layers
    let (ko_gx, ko_gy) = grid.world_to_grid(133.0, 101.0).expect("Keepout point in grid");
    assert_eq!(grid.obstacle_net_ids[grid.idx(ko_gx, ko_gy, 0)], KEEPOUT_NET);
    assert_eq!(grid.obstacle_net_ids[grid.idx(ko_gx, ko_gy, 1)], KEEPOUT_NET);

    // Verify SystemTelemetry initializes cleanly with Vulkan compute counter
    let telemetry = SystemTelemetry::spawn(vk.as_ref().map(|v| Arc::clone(&v.compute_busy_ns)));
    let snap = telemetry.state.lock().unwrap().clone();
    assert_eq!(snap.cpu_history.len(), 60);
    assert_eq!(snap.gpu_history.len(), 60);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_courtyard_legalization_and_simulated_annealing_auto_placer() {
    let dir = temp_test_dir("auto_placer");
    // Create synthetic board with deliberate courtyard overlap between R3 and R4
    let (pcb_path, _) = write_synthetic_project(&dir, true);
    let mut board = Board::load(pcb_path.to_str().unwrap(), &HashSet::new()).unwrap();

    let vk = VulkanEngine::new();
    let cfg = PlacementConfig {
        duration_secs: 1,
        vulkan: vk.clone(),
        ..PlacementConfig::default()
    };

    let initial_state = PlacementState::from_board(&board, &cfg);
    assert!(
        initial_state.overlaps >= 1,
        "Fixture with overlapping R3/R4 must start with >= 1 courtyard overlap"
    );

    // Record initial positions of all locked components
    let locked_before: Vec<(String, f64, f64, f64)> = board
        .footprints
        .iter()
        .filter(|fp| fp.is_locked)
        .map(|fp| (fp.reference.clone(), fp.x, fp.y, fp.rotation))
        .collect();
    assert!(!locked_before.is_empty());

    // 1. Test direct courtyard legalization
    let live = Arc::new(Mutex::new(LiveTrialState::new(board.clone())));
    let legalized = legalize_placement(&board, &cfg, initial_state, Some(&live));
    assert_eq!(legalized.overlaps, 0, "legalize_placement must resolve all courtyard overlaps");
    assert_eq!(live.lock().unwrap().phase, TrialPhase::Legalizing);

    // 2. Test full Parallel Tempering Simulated Annealing auto-placer with Vulkan GPU batch evaluation
    let final_state = run_auto_placer(
        &mut board,
        &cfg,
        Some(Arc::clone(&live)),
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(final_state.overlaps, 0, "Auto-placed board must have 0 courtyard overlaps");
    if let Some(ref vk_eng) = vk {
        assert!(
            vk_eng.compute_busy_ns.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "Auto-placer must utilize Vulkan GPU batch placement shader"
        );
    }

    // Verify locked components remained strictly fixed
    for (ref_name, bx, by, brot) in &locked_before {
        let fp = board.footprints.iter().find(|f| &f.reference == ref_name).unwrap();
        assert!((fp.x - bx).abs() < 1e-6, "Locked footprint {} moved in X", ref_name);
        assert!((fp.y - by).abs() < 1e-6, "Locked footprint {} moved in Y", ref_name);
        assert!((fp.rotation - brot).abs() < 1e-6, "Locked footprint {} rotated", ref_name);
    }

    // Verify all unlocked component courtyards stay inside the board outline boundary
    let boundary = board.boundary_aabb();
    for (i, fp) in board.footprints.iter().enumerate() {
        if fp.is_locked {
            continue;
        }
        let polys = fp.transformed_courtyards(Point::new(fp.x, fp.y), fp.rotation, cfg.min_spacing_mm);
        for poly in &polys {
            for v in &poly.vertices {
                assert!(
                    boundary.contains_point(v),
                    "Unlocked component {} vertex ({:.2}, {:.2}) outside boundary {:?}",
                    fp.reference,
                    v.x,
                    v.y,
                    boundary
                );
            }
        }
        // Pairwise check that no unlocked component overlaps any other component
        for (j, other) in board.footprints.iter().enumerate() {
            if i >= j || (fp.is_locked && other.is_locked) {
                continue;
            }
            let other_polys = other.transformed_courtyards(
                Point::new(other.x, other.y),
                other.rotation,
                cfg.min_spacing_mm,
            );
            for pa in &polys {
                for pb in &other_polys {
                    assert!(
                        !pa.intersects_convex(pb),
                        "Courtyard overlap detected between {} and {}",
                        fp.reference,
                        other.reference
                    );
                }
            }
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_multilayer_autorouter_escape_stubs_nudge_and_via_stitching() {
    let dir = temp_test_dir("autorouter");
    let (pcb_path, _) = write_synthetic_project(&dir, false);
    let mut board = Board::load(pcb_path.to_str().unwrap(), &HashSet::new()).unwrap();

    // Test NudgeEngine on movable vs locked components
    let nudge = NudgeEngine::new();
    let j1_idx = board.footprints.iter().position(|f| f.reference == "J1").unwrap();
    let r3_idx = board.footprints.iter().position(|f| f.reference == "R3").unwrap();
    assert!(!nudge.try_nudge_component(&mut board, j1_idx, (1.0, 0.0)), "Locked J1 must not nudge");
    let r3_orig = (board.footprints[r3_idx].x, board.footprints[r3_idx].y);
    assert!(nudge.try_nudge_component(&mut board, r3_idx, (1.0, 0.0)), "Unlocked R3 should nudge");
    assert!(
        (board.footprints[r3_idx].x - r3_orig.0).hypot(board.footprints[r3_idx].y - r3_orig.1) > 0.1,
        "Nudged R3 position should update"
    );

    let vk = VulkanEngine::new();
    let mut grid = RoutingGrid::new(&board, 0.10, vk);
    let router = AutoRouter::new(6);
    let conns = router.extract_connections(&board);
    assert_eq!(conns.len(), 10, "Expected 10 routable non-GND MST connections");

    let live = Arc::new(Mutex::new(LiveTrialState::new(board.clone())));
    let (routed, total) = router.route_all(
        &mut board,
        &mut grid,
        Some(Arc::clone(&live)),
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(total, 10);
    assert_eq!(routed, total, "All 10 connections on synthetic board must route cleanly");
    assert_eq!(live.lock().unwrap().phase, TrialPhase::Completed);

    // Verify fine-pitch escape stubs (U1 pads 1 & 2, J2 pads A5 & B5) and U1 GND stub (pad 15) were generated
    assert!(!board.tracks.is_empty(), "Router must emit tracks");
    assert!(!board.vias.is_empty(), "Router must emit ground-stitching and/or layer-transition vias");

    // Verify netclass track widths are honored on routed segments
    let vraw_id = board.net_name_to_id["VRAW"];
    let gnd_id = board.net_name_to_id["GND"];
    assert!(
        board.tracks.iter().any(|t| t.net_id == vraw_id && (t.width - 0.60).abs() < 1e-4),
        "Power_15V net VRAW must use 0.60mm track width"
    );
    assert!(
        board.tracks.iter().any(|t| t.net_id == gnd_id),
        "GND escape and stitching tracks must be emitted"
    );

    // Verify all vias have valid drill & annular ring dimensions (0.3mm drill / 0.6mm size)
    assert!(
        board.vias.iter().any(|v| v.net_id == gnd_id),
        "Ground-plane stitching vias must be generated for SMD GND pads"
    );
    for v in &board.vias {
        assert!((v.size - 0.60).abs() < 1e-4, "Via diameter must be 0.60mm");
        assert!((v.drill - 0.30).abs() < 1e-4, "Via drill must be 0.30mm");
        assert!(v.position.0 > board.min_x && v.position.0 < board.max_x);
        assert!(v.position.1 > board.min_y && v.position.1 < board.max_y);
    }

    // Verify 0 same-layer trace crossings between different nets
    for (i, ta) in board.tracks.iter().enumerate() {
        for tb in board.tracks.iter().skip(i + 1) {
            if ta.layer != tb.layer || ta.net_id == tb.net_id {
                continue;
            }
            let sa = Segment {
                p1: Point::new(ta.start.0, ta.start.1),
                p2: Point::new(ta.end.0, ta.end.1),
                net_id: ta.net_id,
            };
            let sb = Segment {
                p1: Point::new(tb.start.0, tb.start.1),
                p2: Point::new(tb.end.0, tb.end.1),
                net_id: tb.net_id,
            };
            assert!(
                !sa.crosses(&sb),
                "Same-layer trace crossing detected on layer {} between net {} and net {}",
                ta.layer,
                ta.net_id,
                tb.net_id
            );
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_roundtrip_kicad_pcb_and_specctra_ses_export() {
    let dir = temp_test_dir("roundtrip_io");
    let (pcb_path, _) = write_synthetic_project(&dir, false);
    let mut board = Board::load(pcb_path.to_str().unwrap(), &HashSet::new()).unwrap();

    // Move an unlocked component (R3) to a known position and rotation, then route
    let r3_idx = board.footprints.iter().position(|f| f.reference == "R3").unwrap();
    board.footprints[r3_idx].x = 115.25;
    board.footprints[r3_idx].y = 108.75;
    board.footprints[r3_idx].rotation = 180.0;

    let mut grid = RoutingGrid::new(&board, 0.10, None);
    let router = AutoRouter::new(4);
    let (routed, total) = router.route_all(&mut board, &mut grid, None, Arc::new(AtomicBool::new(false)));
    assert_eq!(routed, total);

    let out_pcb = dir.join("roundtrip_out.kicad_pcb");
    let out_ses = dir.join("roundtrip_out.ses");
    board.save_pcb(&out_pcb).expect("save_pcb must succeed");
    board.save_ses(&out_ses).expect("save_ses must succeed");

    // 1. Re-parse the written .kicad_pcb and verify component coordinates persist
    let reloaded = Board::load(out_pcb.to_str().unwrap(), &HashSet::new()).expect("Reloading saved PCB must succeed");
    assert_eq!(reloaded.footprints.len(), board.footprints.len());
    let r3_reloaded = reloaded.footprints.iter().find(|f| f.reference == "R3").unwrap();
    assert!((r3_reloaded.x - 115.25).abs() < 1e-3);
    assert!((r3_reloaded.y - 108.75).abs() < 1e-3);
    assert!((r3_reloaded.rotation - 180.0).abs() < 1e-1);

    // Verify all emitted (segment ...) and (via ...) S-expressions are present in the written file
    let raw_out = fs::read_to_string(&out_pcb).unwrap();
    let seg_count = raw_out.matches("(segment (start ").count();
    let via_count = raw_out.matches("(via (at ").count();
    assert_eq!(seg_count, board.tracks.len(), "Written segment count must match board.tracks");
    assert_eq!(via_count, board.vias.len(), "Written via count must match board.vias");

    // 2. Verify Specctra .ses session file structure, placement entries, and routed network paths
    let ses_text = fs::read_to_string(&out_ses).unwrap();
    assert!(ses_text.starts_with("(session vulcans_trials"));
    assert!(ses_text.contains("(placement\n    (resolution um 10)"));
    assert!(ses_text.contains("(component \"R3\" (place \"R3\" 1152500 -1087500 front 180.0))"));
    assert!(ses_text.contains("(was_is"));
    assert!(ses_text.contains("(routes\n    (resolution um 10)"));
    assert!(ses_text.contains("(library_out"));
    assert!(ses_text.contains("(network_out"));
    assert!(ses_text.contains("(net \"VRAW\""));
    assert!(ses_text.contains("(wire (path F.Cu "));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_cli_binary_self_test_and_headless_modes() {
    let bin = env!("CARGO_BIN_EXE_vulcans-trials");
    let dir = temp_test_dir("cli_e2e");
    let (pcb_path, _) = write_synthetic_project(&dir, true);

    // 1. CLI --self-test
    let self_test_out = Command::new(bin)
        .arg("--self-test")
        .output()
        .expect("Failed to execute vulcans-trials --self-test");
    assert!(
        self_test_out.status.success(),
        "--self-test failed: {}",
        String::from_utf8_lossy(&self_test_out.stderr)
    );
    let stdout_st = String::from_utf8_lossy(&self_test_out.stdout);
    assert!(stdout_st.contains("All Self-Tests Passed"));

    // 2. CLI --headless --mode place
    let out_place = dir.join("cli_place.kicad_pcb");
    let place_cmd = Command::new(bin)
        .args([
            "--headless",
            "--mode",
            "place",
            "--duration",
            "1",
            "--input",
            pcb_path.to_str().unwrap(),
            "--output",
            out_place.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run --mode place");
    assert!(place_cmd.status.success());
    assert!(out_place.exists());
    assert!(out_place.with_extension("ses").exists());
    let place_pcb_str = fs::read_to_string(&out_place).unwrap();
    assert_eq!(
        place_pcb_str.matches("(segment (start ").count(),
        0,
        "--mode place should not emit routed track segments"
    );

    // 3. CLI --headless --mode route
    let out_route = dir.join("cli_route.kicad_pcb");
    let route_cmd = Command::new(bin)
        .args([
            "--headless",
            "--mode",
            "route",
            "--res",
            "0.15",
            "--passes",
            "4",
            "--input",
            out_place.to_str().unwrap(),
            "--output",
            out_route.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run --mode route");
    assert!(route_cmd.status.success());
    let route_pcb_str = fs::read_to_string(&out_route).unwrap();
    assert!(
        route_pcb_str.matches("(segment (start ").count() > 0,
        "--mode route must emit routed track segments"
    );

    // 4. CLI --headless --mode both with --lock flag
    let out_both = dir.join("cli_both.kicad_pcb");
    let both_cmd = Command::new(bin)
        .args([
            "--headless",
            "--mode",
            "both",
            "--duration",
            "1",
            "--res",
            "0.15",
            "--passes",
            "4",
            "--lock",
            "C5,R14",
            "--input",
            pcb_path.to_str().unwrap(),
            "--output",
            out_both.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run --mode both");
    assert!(both_cmd.status.success());
    let both_stdout = String::from_utf8_lossy(&both_cmd.stdout);
    assert!(both_stdout.contains("[Auto-Placer] Done:"));
    assert!(both_stdout.contains("[Auto-Router] Done in"));
    assert!(out_both.exists());
    assert!(out_both.with_extension("ses").exists());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_p1p2esp_project_board_integration_if_present() {
    let pcb_path = "/home/deece/src/p1p2esp/p1p2esp.kicad_pcb";
    if !Path::new(pcb_path).exists() {
        return;
    }

    let vk = VulkanEngine::new();
    assert!(vk.is_some(), "Vulkan backend should initialize");

    let mut board = Board::load(pcb_path, &HashSet::new()).expect("Failed to load p1p2esp PCB");
    assert!(
        board.footprints.len() >= 53,
        "Expected at least 53 footprints in p1p2esp, got {}",
        board.footprints.len()
    );
    assert!(!board.clusters.is_empty());

    // Verify p1p2esp.kicad_pro netclasses are loaded and honored
    for power_net in ["VBUS", "VRAW", "/BUCK_VIN", "+3.3V", "VCC_5V"] {
        if let Some(&nid) = board.net_name_to_id.get(power_net) {
            let nc = board.get_netclass_for_net(nid);
            assert_eq!(nc.name, "Power", "Net {power_net} must be in Power class");
            assert!(
                ((nc.track_width - 0.50).abs() < 1e-6 || (nc.track_width - 0.60).abs() < 1e-6)
                    && (nc.clearance - 0.15).abs() < 1e-6
                    && ((nc.via_dia - 0.60).abs() < 1e-6 || (nc.via_dia - 0.80).abs() < 1e-6)
            );
        }
    }
    for bus_net in ["P1", "P2", "HP", "HN"] {
        if let Some(&nid) = board.net_name_to_id.get(bus_net) {
            let nc = board.get_netclass_for_net(nid);
            assert_eq!(nc.name, "P1P2_Bus", "Net {bus_net} must be in P1P2_Bus class");
        }
    }
    let aio_nid = *board.net_name_to_id.get("AIO").or_else(|| board.net_name_to_id.get("/AIO")).unwrap();
    let aio_nc = board.get_netclass_for_net(aio_nid);
    assert_eq!(aio_nc.name, "Analog_Sense");
    assert!((aio_nc.track_width - 0.25).abs() < 1e-6 && (aio_nc.clearance - 0.15).abs() < 1e-6);
    let usb_nid = *board.net_name_to_id.get("USB_DP").or_else(|| board.net_name_to_id.get("/USB_DP")).unwrap();
    let usb_nc = board.get_netclass_for_net(usb_nid);
    assert_eq!(usb_nc.name, "USB_Diff");
    assert!((usb_nc.track_width - 0.30).abs() < 1e-6 && (usb_nc.clearance - 0.20).abs() < 1e-6);

    let j1_before = board
        .footprints
        .iter()
        .find(|f| f.reference == "J1")
        .map(|f| (f.x, f.y))
        .unwrap();

    let cfg = PlacementConfig {
        duration_secs: 1,
        ..PlacementConfig::default()
    };
    let st = run_auto_placer(&mut board, &cfg, None, Arc::new(AtomicBool::new(false)));
    assert_eq!(st.overlaps, 0, "Legalized placement must have 0 courtyard overlaps");

    let j1_after = board
        .footprints
        .iter()
        .find(|f| f.reference == "J1")
        .map(|f| (f.x, f.y))
        .unwrap();
    assert_eq!(j1_before, j1_after, "Locked footprint J1 must not move");

    let mut grid = RoutingGrid::new(&board, 0.10, vk);
    assert!(grid.width > 100 && grid.height > 100);
    assert_eq!(grid.num_layers, 4, "p1p2esp is a 4-layer board (F.Cu, B.Cu, In1.Cu, In2.Cu)");
    assert_eq!(board.copper_layers(), vec!["F.Cu", "B.Cu", "In1.Cu", "In2.Cu"]);
    let planes = board.inner_plane_nets();
    assert_eq!(planes.get(&2).copied(), Some(board.net_name_to_id["GND"]));
    assert_eq!(planes.get(&3).copied(), Some(board.net_name_to_id["+3.3V"]));

    // Verify J2 (-90 deg rotated USB-C receptacle) pad world size swaps width/height cleanly
    let j2 = board.footprints.iter().find(|f| f.reference == "J2").unwrap();
    let j2_a5 = j2.pads.iter().find(|p| p.number == "A5").unwrap();
    let (rw, rh) = j2.pad_world_size(j2_a5);
    assert!(
        (rw - 1.45).abs() < 1e-3 && (rh - 0.30).abs() < 1e-3,
        "Rotated J2 pad A5 world size should be 1.45x0.30mm, got {:.2}x{:.2}",
        rw,
        rh
    );

    let router = AutoRouter::new(2);
    let conns = router.extract_connections(&board);
    assert!(conns.len() >= 66);
    assert!(
        conns.iter().any(|c| c.diff_pair_leader.is_some() && (c.net_name.ends_with("USB_DP") || c.net_name.ends_with("USB_DM"))),
        "USB_DP/USB_DM between J2 and U4 must be paired as a coupled differential bus"
    );

    let mid = GridNode {
        gx: grid.width / 2,
        gy: grid.height / 2,
        layer: 1,
    };
    let neighbors = grid.get_neighbors(mid, 1, 0.20, 0.20, 0.60, None);
    assert!(!neighbors.is_empty());

    // Verify USB_DP and USB_DM route successfully as a coupled differential pair
    let mut dp_conn = conns.iter().find(|c| (c.net_name.ends_with("USB_DP") || c.net_name.ends_with("USB_DM")) && c.diff_pair_leader.is_none() && c.start_fp != c.end_fp).unwrap().clone();
    let mut dm_conn = conns.iter().find(|c| (c.net_name.ends_with("USB_DP") || c.net_name.ends_with("USB_DM")) && c.diff_pair_leader.is_some() && c.start_fp != c.end_fp).unwrap().clone();
    let escape_stubs = router.generate_escape_stubs(&board);

    for stub in &escape_stubs {
        if dp_conn.net_id == stub.net_id && dp_conn.start.2 == stub.layer && (dp_conn.start.0 - stub.pad_pos.0).hypot(dp_conn.start.1 - stub.pad_pos.1) < 0.15 {
            dp_conn.start = (stub.tip_pos.0, stub.tip_pos.1, stub.tip_layer);
            break;
        }
    }
    for stub in &escape_stubs {
        if dp_conn.net_id == stub.net_id && dp_conn.end.2 == stub.layer && (dp_conn.end.0 - stub.pad_pos.0).hypot(dp_conn.end.1 - stub.pad_pos.1) < 0.15 {
            dp_conn.end = (stub.tip_pos.0, stub.tip_pos.1, stub.tip_layer);
            break;
        }
    }
    for stub in &escape_stubs {
        if dm_conn.net_id == stub.net_id && dm_conn.start.2 == stub.layer && (dm_conn.start.0 - stub.pad_pos.0).hypot(dm_conn.start.1 - stub.pad_pos.1) < 0.15 {
            dm_conn.start = (stub.tip_pos.0, stub.tip_pos.1, stub.tip_layer);
            break;
        }
    }
    for stub in &escape_stubs {
        if dm_conn.net_id == stub.net_id && dm_conn.end.2 == stub.layer && (dm_conn.end.0 - stub.pad_pos.0).hypot(dm_conn.end.1 - stub.pad_pos.1) < 0.15 {
            dm_conn.end = (stub.tip_pos.0, stub.tip_pos.1, stub.tip_layer);
            break;
        }
    }

    let (Some((sgx, sgy)), Some((egx, egy))) = (
        grid.world_to_grid(dp_conn.start.0, dp_conn.start.1),
        grid.world_to_grid(dp_conn.end.0, dp_conn.end.1),
    ) else { panic!("Start or end out of grid bounds"); };

    let s_node = GridNode { gx: sgx, gy: sgy, layer: dp_conn.start.2 };
    let e_node = GridNode { gx: egx, gy: egy, layer: dp_conn.end.2 };
    let stop_flag = AtomicBool::new(false);
    let dp_path = router.find_path_with_constraints(
        &grid,
        s_node,
        e_node,
        dp_conn.net_id,
        dp_conn.netclass.clearance,
        dp_conn.netclass.diff_pair_width,
        dp_conn.netclass.via_dia,
        None,
        1.0,
        0.0,
        dp_conn.netclass.max_coupled_length_mm,
        dp_conn.netclass.require_return_path,
        None,
        &stop_flag,
    );
    assert!(dp_path.is_some(), "USB_DP path must be found");
    let p = dp_path.unwrap();
    router.commit_path_to_grid(&mut grid, &p, dp_conn.net_id, dp_conn.netclass.diff_pair_width, dp_conn.netclass.via_dia);

    let corridor = router.build_diff_pair_corridor(&grid, &p, dp_conn.netclass.diff_pair_width, dp_conn.netclass.diff_pair_gap, 0.0);
    let (Some((dmg_sx, dmg_sy)), Some((dmg_ex, dmg_ey))) = (
        grid.world_to_grid(dm_conn.start.0, dm_conn.start.1),
        grid.world_to_grid(dm_conn.end.0, dm_conn.end.1),
    ) else { panic!("DM start or end out of grid"); };
    let dm_s_node = GridNode { gx: dmg_sx, gy: dmg_sy, layer: dm_conn.start.2 };
    let dm_e_node = GridNode { gx: dmg_ex, gy: dmg_ey, layer: dm_conn.end.2 };

    let dm_path = router.find_path_with_constraints(
        &grid,
        dm_s_node,
        dm_e_node,
        dm_conn.net_id,
        dm_conn.netclass.clearance,
        dm_conn.netclass.diff_pair_width,
        dm_conn.netclass.via_dia,
        None,
        1.0,
        0.0,
        dm_conn.netclass.max_coupled_length_mm,
        dm_conn.netclass.require_return_path,
        Some(&corridor),
        &stop_flag,
    );
    assert!(dm_path.is_some(), "USB_DM follower path in corridor must be found");
}

#[test]
fn test_gui_click_footprint_lock_toggle_and_legend_e2e() {
    use eframe::egui::{self, Pos2, Rect};
    use vulcans_trials::gui::VulcansTrialsApp;

    let dir = temp_test_dir("gui_lock_e2e");
    let (pcb_path, _) = write_synthetic_project(&dir, false);
    let board = Board::load(pcb_path.to_str().unwrap(), &HashSet::new()).unwrap();

    let mut app = VulcansTrialsApp::new(pcb_path.to_str().unwrap().to_string(), String::new(), board.clone(), None);
    let canvas_rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 750.0));

    // Find an unlocked component (C5) and click its screen position to lock it
    let c5 = board.footprints.iter().find(|f| f.reference == "C5").unwrap();
    assert!(!c5.is_locked);
    let (c5_pos, _) = app.board_to_screen(&board, canvas_rect, c5.x, c5.y);
    assert_eq!(
        app.toggle_footprint_lock_at(canvas_rect, c5_pos),
        Some(("C5".to_string(), true))
    );

    // Render full GUI frame including newly locked C5's scaled lock icon and the color legend
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.render_ui(ctx);
    });

    // Click C5 again to unlock it
    assert_eq!(
        app.toggle_footprint_lock_at(canvas_rect, c5_pos),
        Some(("C5".to_string(), false))
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_smd_side_flip_and_4_layer_stackup_e2e() {
    let dir = temp_test_dir("flip_4layer_e2e");
    let pcb_path = dir.join("board_4layer.kicad_pcb");
    let pcb_content = r#"(kicad_pcb (version 20240108)
  (layers
    (0 "F.Cu" signal)
    (4 "In1.Cu" signal)
    (6 "In2.Cu" signal)
    (2 "B.Cu" signal)
  )
  (net 0 "")
  (net 1 "GND")
  (net 2 "+3.3V")
  (net 3 "USB_DP")
  (net 4 "USB_DM")
  (gr_rect (start 100.0 100.0) (end 124.0 118.0) (layer "Edge.Cuts"))
  (zone (net 1) (net_name "GND") (layer "In1.Cu"))
  (zone (net 2) (net_name "+3.3V") (layer "In2.Cu"))
  (footprint "Resistor_SMD:R_0603" (layer "F.Cu")
    (at 108.0 108.0)
    (property "Reference" "R1")
    (fp_rect (start -1.0 -0.8) (end 1.0 0.8) (layer "F.CrtYd"))
    (pad "1" smd rect (at -0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 3 "USB_DP"))
    (pad "2" smd rect (at 0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 2 "+3.3V"))
  )
  (footprint "Resistor_SMD:R_0603" (layer "F.Cu")
    (at 108.1 108.0)
    (property "Reference" "R2")
    (fp_rect (start -1.0 -0.8) (end 1.0 0.8) (layer "F.CrtYd"))
    (pad "1" smd rect (at -0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 4 "USB_DM"))
    (pad "2" smd rect (at 0.7 0.0) (size 0.7 0.7) (layers "F.Cu") (net 2 "+3.3V"))
  )
  (footprint "Package_SO:SOIC-8" (layer "B.Cu")
    (at 116.0 108.0)
    (property "Reference" "U4")
    (locked yes)
    (fp_rect (start -1.8 -1.8) (end 1.8 1.8) (layer "B.CrtYd"))
    (pad "1" smd rect (at -1.0 -0.5) (size 0.8 0.4) (layers "B.Cu") (net 3 "USB_DP"))
    (pad "2" smd rect (at -1.0 0.5) (size 0.8 0.4) (layers "B.Cu") (net 4 "USB_DM"))
  )
)"#;
    fs::write(&pcb_path, pcb_content).unwrap();

    let mut board = Board::load(pcb_path.to_str().unwrap(), &HashSet::new()).unwrap();
    assert_eq!(board.copper_layers(), vec!["F.Cu", "B.Cu", "In1.Cu", "In2.Cu"]);
    assert_eq!(board.footprints[2].layer, 1, "U4 on B.Cu must load with layer == 1");

    // Flip R2 onto B.Cu and verify courtyard overlap between R1 (F.Cu) and R2 (B.Cu) is 0
    board.footprints[1].layer = 1;
    let cfg = PlacementConfig {
        allow_layer_flip: true,
        duration_secs: 0,
        ..PlacementConfig::default()
    };
    let st = PlacementState::from_board(&board, &cfg);
    assert_eq!(st.overlaps, 0, "SMD parts on opposite sides (F.Cu vs B.Cu) must not collide");

    let mut grid = RoutingGrid::new(&board, 0.20, None);
    assert_eq!(grid.num_layers, 4);
    let router = AutoRouter::new(4);
    let (routed, total) = router.route_all(&mut board, &mut grid, None, Arc::new(AtomicBool::new(false)));
    assert_eq!(routed, total);

    let out_pcb = dir.join("board_4layer_out.kicad_pcb");
    let out_ses = dir.join("board_4layer_out.ses");
    board.save_pcb(&out_pcb).unwrap();
    board.save_ses(&out_ses).unwrap();

    let saved_pcb = fs::read_to_string(&out_pcb).unwrap();
    let saved_ses = fs::read_to_string(&out_ses).unwrap();
    assert!(saved_ses.contains("back"), "Flipped B.Cu footprints must export as 'back' in .ses");
    let reloaded = Board::load(out_pcb.to_str().unwrap(), &HashSet::new()).unwrap();
    let r2_reloaded = reloaded.footprints.iter().find(|f| f.reference == "R2").unwrap();
    assert_eq!(r2_reloaded.layer, 1, "Flipped R2 must reload on B.Cu (layer 1)");
    assert!(saved_pcb.contains("(layer \"B.Cu\")"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_clear_unlocked_objects_holding_palette_and_priority_placement_e2e() {
    let dir = temp_test_dir("vulcan_test_clear_palette_e2e");
    let pcb_path = dir.join("board.kicad_pcb");

    let pcb_content = r#"(kicad_pcb (version 20240108)
  (net 0 "")
  (net 1 "GND")
  (net 2 "+3.3V")
  (net 3 "USB_DP")
  (net 4 "USB_DM")
  (net 5 "SIG_NET")
  (gr_rect (start 100.0 100.0) (end 160.0 140.0) (layer "Edge.Cuts"))
  (footprint "U" (at 120.0 120.0) (property "Reference" "U1") (locked yes)
    (fp_line (start -3.0 -3.0) (end 3.0 -3.0) (layer "F.CrtYd"))
    (fp_line (start 3.0 -3.0) (end 3.0 3.0) (layer "F.CrtYd"))
    (fp_line (start 3.0 3.0) (end -3.0 3.0) (layer "F.CrtYd"))
    (fp_line (start -3.0 3.0) (end -3.0 -3.0) (layer "F.CrtYd"))
    (pad "1" smd rect (at 0.0 -2.5) (size 0.5 0.5) (net 2 "+3.3V"))
    (pad "2" smd rect (at 0.0 2.5) (size 0.5 0.5) (net 1 "GND")))
  (footprint "C" (at 150.0 135.0) (property "Reference" "C20")
    (fp_line (start -0.8 -0.4) (end 0.8 -0.4) (layer "F.CrtYd"))
    (fp_line (start 0.8 -0.4) (end 0.8 0.4) (layer "F.CrtYd"))
    (fp_line (start 0.8 0.4) (end -0.8 0.4) (layer "F.CrtYd"))
    (fp_line (start -0.8 0.4) (end -0.8 -0.4) (layer "F.CrtYd"))
    (pad "1" smd rect (at -0.5 0.0) (size 0.4 0.4) (net 2 "+3.3V"))
    (pad "2" smd rect (at 0.5 0.0) (size 0.4 0.4) (net 1 "GND")))
  (footprint "C" (at 152.0 135.0) (property "Reference" "C5")
    (fp_line (start -0.8 -0.4) (end 0.8 -0.4) (layer "F.CrtYd"))
    (fp_line (start 0.8 -0.4) (end 0.8 0.4) (layer "F.CrtYd"))
    (fp_line (start 0.8 0.4) (end -0.8 0.4) (layer "F.CrtYd"))
    (fp_line (start -0.8 0.4) (end -0.8 -0.4) (layer "F.CrtYd"))
    (pad "1" smd rect (at -0.5 0.0) (size 0.4 0.4) (net 2 "+3.3V"))
    (pad "2" smd rect (at 0.5 0.0) (size 0.4 0.4) (net 1 "GND")))
  (footprint "J" (at 105.0 120.0) (property "Reference" "J1") (locked yes)
    (pad "1" smd rect (at 0.0 -0.5) (size 0.4 0.4) (net 3 "USB_DP"))
    (pad "2" smd rect (at 0.0 0.5) (size 0.4 0.4) (net 4 "USB_DM")))
  (footprint "U" (at 155.0 135.0) (property "Reference" "U2")
    (pad "1" smd rect (at 0.0 -0.5) (size 0.4 0.4) (net 3 "USB_DP"))
    (pad "2" smd rect (at 0.0 0.5) (size 0.4 0.4) (net 4 "USB_DM")))
  (footprint "R" (at 157.0 135.0) (property "Reference" "R30")
    (pad "1" smd rect (at 0.0 0.0) (size 0.4 0.4) (net 5 "SIG_NET")))
  (segment (start 105.0 120.0) (end 110.0 120.0) (width 0.25) (layer "F.Cu") (net 3) (locked yes))
  (segment (start 110.0 120.0) (end 115.0 120.0) (width 0.25) (layer "F.Cu") (net 3))
  (via (at 110.0 120.0) (size 0.6) (drill 0.3) (layers "F.Cu" "B.Cu") (net 3))
)"#;
    fs::write(&pcb_path, pcb_content).unwrap();

    let mut board = Board::load(pcb_path.to_str().unwrap(), &HashSet::new()).unwrap();
    assert_eq!(board.footprints.len(), 6);
    assert_eq!(board.tracks.len(), 2);
    assert!(board.tracks[0].is_locked);
    assert!(!board.tracks[1].is_locked);
    assert_eq!(board.vias.len(), 1);

    // 1. Clear unlocked objects
    let (ripped, cleared) = board.clear_unlocked_objects();
    assert_eq!(ripped, 2);
    assert_eq!(cleared, 4);
    assert_eq!(board.footprints.len(), 2);
    assert_eq!(board.tracks.len(), 1);
    assert!(board.tracks[0].is_locked);
    assert_eq!(board.vias.len(), 0);

    // 2. Verify natural designator sorting: letters first, then numbers, so C5 before C20!
    assert_eq!(board.holding_palette.len(), 4);
    assert_eq!(board.holding_palette[0].reference, "C5");
    assert_eq!(board.holding_palette[1].reference, "C20");
    assert_eq!(board.holding_palette[2].reference, "R30");
    assert_eq!(board.holding_palette[3].reference, "U2");

    let cfg = PlacementConfig {
        duration_secs: 1,
        grid_step_mm: 0.1,
        ..PlacementConfig::default()
    };

    // 3. Test placing C5 individually via place_holding_component_at_best_location
    let placed_c5 = vulcans_trials::placer::place_holding_component_at_best_location(&mut board, 0, &cfg).unwrap();
    assert_eq!(placed_c5.reference, "C5");
    assert_eq!(board.holding_palette.len(), 3);
    assert_eq!(board.footprints.len(), 3);

    // Verify C5 was placed at favourable location near U1 power pin (120.0, 117.5), not at original (152.0, 135.0)
    let orig_dist = (placed_c5.x - 152.0).hypot(placed_c5.y - 135.0);
    assert!(orig_dist > 5.0, "C5 must not be placed at its original cleared position");

    let u1_pin1_x = 120.0;
    let u1_pin1_y = 117.5;
    let c5_pin1 = placed_c5.pad_absolute_position(&placed_c5.pads[0]);
    let dist_c5 = (c5_pin1.0 - u1_pin1_x).hypot(c5_pin1.1 - u1_pin1_y);
    assert!(dist_c5 < 2.0, "Decoupling cap C5 must be situated close to U1 power pin (< 2.0mm), got {:.2}mm", dist_c5);

    // 4. Place remaining components using place_all_holding_components_at_best_locations
    let placed_all = vulcans_trials::placer::place_all_holding_components_at_best_locations(&mut board, &cfg);
    assert_eq!(placed_all, 3);
    assert_eq!(board.holding_palette.len(), 0);
    assert_eq!(board.footprints.len(), 6);

    // 5. Verify simulated annealing auto-placer works cleanly with 0 courtyard overlaps
    let st = run_auto_placer(&mut board, &cfg, None, Arc::new(AtomicBool::new(false)));
    assert_eq!(board.holding_palette.len(), 0, "Holding palette must be drained for placement");
    assert_eq!(board.footprints.len(), 6, "All 6 footprints must be placed on board");
    assert_eq!(st.overlaps, 0, "Placement must have 0 courtyard overlaps");

    let _ = fs::remove_dir_all(&dir);
}



#[test]
fn test_u1_test_pcb_full_route_pad_centering_and_drc_clean() {
    let board_path = "tests/boards/u1_test.kicad_pcb";
    let mut board = Board::load(board_path, &HashSet::new()).expect("Failed to load u1_test.kicad_pcb");
    assert_eq!(board.footprints.len(), 16);
    let vulkan = VulkanEngine::new();
    let mut grid = RoutingGrid::new(&board, 0.10, vulkan).with_inner_layer_penalty(8.0);
    let router = AutoRouter::new(15).with_inner_layer_penalty(8.0);
    let (routed, total) = router.route_all(&mut board, &mut grid, None, Arc::new(AtomicBool::new(false)));

    assert_eq!(
        routed, total,
        "All connections on u1_test must route to 100% completion (routed {}/{})",
        routed, total
    );

    // Verify pad entry: every signal pad has a track ending on its copper (at the centre, or slid/
    // corner-anchored by the acid-trap pad-exit pass)
    for fp in &board.footprints {
        for pad in &fp.pads {
            if pad.net_id == 0 || pad.net_name == "GND" || pad.net_name.is_empty() || pad.net_name.starts_with("unconnected") {
                continue;
            }
            let (px, py) = fp.pad_absolute_position(pad);
            let has_centered_track = board.tracks.iter().any(|t| {
                t.net_id == pad.net_id
                    && (vulcans_trials::router::dist_point_to_pad(t.start, fp, pad) < 1e-4
                        || vulcans_trials::router::dist_point_to_pad(t.end, fp, pad) < 1e-4)
            });
            assert!(
                has_centered_track,
                "Pad {} on {} (net {} '{}') at ({:.4}, {:.4}) must have a track ending on its copper",
                pad.number, fp.reference, pad.net_id, pad.net_name, px, py
            );
        }
    }

    // Save and verify with KiCad DRC CLI if available
    let temp_dir = temp_test_dir("u1_full_route");
    let out_pcb = temp_dir.join("u1_test_routed.kicad_pcb");
    let out_drc = temp_dir.join("u1_drc.json");

    board.save_pcb(&out_pcb).expect("Failed to save routed board");

    if let Ok(kicad_check) = Command::new("kicad-cli").arg("--version").output() {
        if kicad_check.status.success() {
            let drc_status = Command::new("kicad-cli")
                .args([
                    "pcb",
                    "drc",
                    "--refill-zones",
                    "--format",
                    "json",
                    "--output",
                    out_drc.to_str().unwrap(),
                    out_pcb.to_str().unwrap(),
                ])
                .status()
                .expect("Failed to execute kicad-cli pcb drc");

            assert!(drc_status.success(), "kicad-cli pcb drc command failed");
            let drc_json = fs::read_to_string(&out_drc).expect("Failed to read DRC JSON report");
            let v: serde_json::Value = serde_json::from_str(&drc_json).expect("Invalid DRC JSON format");

            let unconnected_count = v["unconnected_items"]
                .as_array()
                .map(|a| a.len())
                .unwrap_or(0);
            assert_eq!(
                unconnected_count, 0,
                "KiCad DRC reported {} unconnected items: {:?}",
                unconnected_count, v["unconnected_items"]
            );

            let error_violations: Vec<_> = v["violations"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter(|item| item["severity"].as_str() == Some("error"))
                        .collect()
                })
                .unwrap_or_default();

            assert!(
                error_violations.is_empty(),
                "KiCad DRC reported {} error violations: {:?}",
                error_violations.len(),
                error_violations
            );
        }
    }
}

#[test]
fn test_u1_test_pcb_can_be_fully_routed() {
    let board_path = "tests/boards/u1_test.kicad_pcb";
    let mut board = Board::load(board_path, &HashSet::new()).expect("Failed to load u1_test.kicad_pcb");
    assert_eq!(board.footprints.len(), 16);
    let orig_positions: Vec<(f64, f64)> = board.footprints.iter().map(|f| (f.x, f.y)).collect();

    let vulkan = VulkanEngine::new();
    let mut grid = RoutingGrid::new(&board, 0.10, vulkan).with_inner_layer_penalty(8.0);
    let router = AutoRouter::new(15).with_inner_layer_penalty(8.0);
    let (routed, total) = router.route_all(&mut board, &mut grid, None, Arc::new(AtomicBool::new(false)));

    assert!(total > 0, "U1 test board must have connections to route");
    assert_eq!(
        routed, total,
        "U1 test PCB must be 100% fully routed (routed {}/{} connections)",
        routed, total
    );

    // Verify footprints did not move during autorouting
    let new_positions: Vec<(f64, f64)> = board.footprints.iter().map(|f| (f.x, f.y)).collect();
    for (i, (orig, new)) in orig_positions.iter().zip(new_positions.iter()).enumerate() {
        assert!(
            (orig.0 - new.0).hypot(orig.1 - new.1) < 1e-4,
            "Footprint {} ({}) should not move during autorouting",
            i, board.footprints[i].reference
        );
    }
}

#[test]
fn test_rip_up_connected_traces_on_move_and_shove_e2e() {
    use eframe::egui::{Pos2, Rect};
    use vulcans_trials::board::{TrackSegment, Via};
    use vulcans_trials::gui::VulcansTrialsApp;

    let dir = temp_test_dir("rip_up_e2e");
    let (pcb_path, _) = write_synthetic_project(&dir, false);
    let mut board = Board::load(pcb_path.to_str().unwrap(), &HashSet::new()).unwrap();

    // Find C5 (unlocked), R3 (unlocked), and R4 (unlocked)
    let c5_idx = board.footprints.iter().position(|f| f.reference == "C5").unwrap();
    let r3_idx = board.footprints.iter().position(|f| f.reference == "R3").unwrap();
    let r4_idx = board.footprints.iter().position(|f| f.reference == "R4").unwrap();

    let c5_pad_pos = board.footprints[c5_idx].pad_absolute_position(&board.footprints[c5_idx].pads[0]);
    let r3_pad_pos = board.footprints[r3_idx].pad_absolute_position(&board.footprints[r3_idx].pads[0]);
    let r4_pad_pos = board.footprints[r4_idx].pad_absolute_position(&board.footprints[r4_idx].pads[0]);

    let net_id = board.footprints[c5_idx].pads[0].net_id;
    board.footprints[r3_idx].pads[0].net_id = net_id;
    board.footprints[r4_idx].pads[0].net_id = net_id;

    // Connect C5 -> R3 via 2 tracks and 1 via
    let mid_x = (c5_pad_pos.0 + r3_pad_pos.0) * 0.5;
    let mid_y = (c5_pad_pos.1 + r3_pad_pos.1) * 0.5;
    board.tracks.push(TrackSegment {
        start: c5_pad_pos,
        end: (mid_x, mid_y),
        width: 0.20,
        layer: 0,
        net_id,
        is_locked: false,
    });
    board.vias.push(Via {
        position: (mid_x, mid_y),
        size: 0.60,
        drill: 0.30,
        net_id,
        is_locked: false,
    });
    board.tracks.push(TrackSegment {
        start: (mid_x, mid_y),
        end: r3_pad_pos,
        width: 0.20,
        layer: 1,
        net_id,
        is_locked: false,
    });

    // Connect R3 -> R4 via 1 track on layer 0
    board.tracks.push(TrackSegment {
        start: r3_pad_pos,
        end: r4_pad_pos,
        width: 0.20,
        layer: 0,
        net_id,
        is_locked: false,
    });

    assert_eq!(board.tracks.len(), 3);
    assert_eq!(board.vias.len(), 1);

    let mut app = VulcansTrialsApp::new(
        pcb_path.to_str().unwrap().to_string(),
        String::new(),
        board.clone(),
        None,
    );
    let canvas_rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1000.0, 750.0));

    // 1. Pick up C5
    let (c5_screen, _) = app.board_to_screen(&board, canvas_rect, board.footprints[c5_idx].x, board.footprints[c5_idx].y);
    assert!(app.pick_up_component_at(canvas_rect, c5_screen));

    // 2. Drag C5 to a new location (e.g. +5mm in X and Y)
    let (target_screen, _) = app.board_to_screen(&board, canvas_rect, board.footprints[c5_idx].x + 5.0, board.footprints[c5_idx].y + 5.0);
    app.update_moving_component_pos(canvas_rect, target_screen);

    // 3. Place C5
    let shoved = app.place_moving_component();
    assert!(shoved.is_some());

    // 4. Verify C5 -> R3 trace (2 tracks and 1 via) was ripped up, while R3 -> R4 track remains intact!
    let snap = app.live.lock().unwrap();
    assert_eq!(snap.board.tracks.len(), 1);
    assert_eq!(snap.board.vias.len(), 0);
    assert_eq!(snap.board.tracks[0].start, r3_pad_pos);
    assert_eq!(snap.board.tracks[0].end, r4_pad_pos);
}

#[test]
fn test_p1p2esp_full_routing_completion() {
    let pcb_path = "/home/deece/src/p1p2esp/p1p2esp.kicad_pcb";
    if !std::path::Path::new(pcb_path).exists() {
        return;
    }
    let locked: std::collections::HashSet<String> = vulcans_trials::board::DEFAULT_LOCKED_REFS.iter().map(|s| s.to_string()).collect();
    let mut board = vulcans_trials::board::Board::load(pcb_path, &locked).unwrap();
    let vulkan = vulcans_trials::vulkan::VulkanEngine::new();
    let mut grid = vulcans_trials::router::RoutingGrid::new(&board, 0.10, vulkan);
    let router = vulcans_trials::router::AutoRouter::new(12);
    let (routed, total) = router.route_all(&mut board, &mut grid, None, std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));
    assert_eq!(routed, total, "Should route all {}/{} connections, got {}/{}", total, total, routed, total);

    // Verify every single extracted connection is 100% electrically connected end-to-end after post-processing
    let conns = router.extract_connections(&board);
    for (idx, conn) in conns.iter().enumerate() {
        assert!(
            vulcans_trials::router::AutoRouter::is_connection_electrically_connected(&board, conn),
            "Connection #{} '{}' ({}.pad{} -> {}.pad{}) is not electrically continuous after post-processing",
            idx,
            conn.net_name,
            board.footprints[conn.start_fp].reference,
            conn.start_pad,
            board.footprints[conn.end_fp].reference,
            conn.end_pad
        );
    }

    assert_usb_pair_coupled(&board);

    // Verify TERM (id 2) has connected tracks to U1
    let term_nid = *board.net_name_to_id.get("TERM").or_else(|| board.net_name_to_id.get("/TERM")).unwrap();
    let term_tracks: Vec<_> = board.tracks.iter().filter(|t| t.net_id == term_nid).collect();
    assert!(!term_tracks.is_empty(), "TERM net must have route segments connecting U1 to R4, got {}", term_tracks.len());

    // Verify U1 Pad 25 (B.Cu exposed thermal pad on F.Cu footprint) is parsed with layer = Some(1) and is_through_hole = false
    let u1_fp = board.footprints.iter().find(|f| f.reference == "U1").expect("U1 footprint must exist");
    let pad25_bcu = u1_fp.pads.iter().find(|p| p.number == "25" && p.width > 2.0 && p.layer == Some(1)).expect("U1 B.Cu thermal pad 25 must exist");
    assert!(!pad25_bcu.is_through_hole, "U1 Pad 25 on B.Cu must not be marked through-hole");
    assert_eq!(u1_fp.pad_layer(pad25_bcu), 1, "U1 Pad 25 must be assigned to B.Cu (layer 1)");

    // Verify 0 vias overlap any SMD pad (including same-net SMD pads like D1) when allow_via_in_pad is false (default)
    assert!(!board.allow_via_in_pad());
    for v in &board.vias {
        for fp in &board.footprints {
            for pad in &fp.pads {
                if !pad.is_through_hole {
                    let d = vulcans_trials::router::dist_point_to_pad(v.position, fp, pad);
                    assert!(
                        d >= v.size * 0.5 - 1e-4,
                        "Via at ({:.3}, {:.3}) (net {}) overlaps SMD pad {}.{} (net {}, dist={:.3} < {:.3})",
                        v.position.0,
                        v.position.1,
                        v.net_id,
                        fp.reference,
                        pad.number,
                        pad.net_id,
                        d,
                        v.size * 0.5
                    );
                }
            }
        }
    }

    // Verify organic conversion preserves 100% electrical connectivity and uniform 0.60mm width on /BUCK_VIN and +3.3V
    vulcans_trials::router::convert_tracks_to_organic_with_fillet_radius(&mut board, 0.35);
    for (idx, conn) in conns.iter().enumerate() {
        assert!(
            vulcans_trials::router::AutoRouter::is_connection_electrically_connected(&board, conn),
            "Connection #{} '{}' ({}.pad{} -> {}.pad{}) lost electrical continuity after organic conversion",
            idx,
            conn.net_name,
            board.footprints[conn.start_fp].reference,
            conn.start_pad,
            board.footprints[conn.end_fp].reference,
            conn.end_pad
        );
    }
    if let Some(&p3v3_nid) = board.net_name_to_id.get("+3.3V") {
        let expected_w = board.get_netclass_for_net(p3v3_nid).track_width;
        let widths: Vec<f64> = board
            .tracks
            .iter()
            .filter(|t| t.net_id == p3v3_nid)
            .map(|t| t.width)
            .chain(board.arcs.iter().filter(|a| a.net_id == p3v3_nid).map(|a| a.width))
            .collect();
        assert!(!widths.is_empty());
        for w in &widths {
            assert!(
                (*w - expected_w).abs() < 1e-3,
                "Power net +3.3V must maintain uniform {:.2} mm width, found {:.3} mm (all: {:?})",
                expected_w,
                w,
                widths
            );
        }
    }
    if let Some(&buck_nid) = board.net_name_to_id.get("/BUCK_VIN") {
        let expected_w = board.get_netclass_for_net(buck_nid).track_width;
        let widths: Vec<f64> = board
            .tracks
            .iter()
            .filter(|t| t.net_id == buck_nid)
            .map(|t| t.width)
            .chain(board.arcs.iter().filter(|a| a.net_id == buck_nid).map(|a| a.width))
            .collect();
        assert!(!widths.is_empty());
        assert!(
            widths.iter().any(|&w| (w - expected_w).abs() < 1e-3),
            "/BUCK_VIN main trunk must widen to its {:.2} mm Power netclass width (all: {:?})",
            expected_w,
            widths
        );
        for w in &widths {
            assert!(
                *w >= 0.30 - 1e-3,
                "Power net /BUCK_VIN must not have narrow < 0.30 mm segments/arcs, found {:.3} mm (all: {:?})",
                w,
                widths
            );
        }
    }
}

/// USB_DP / USB_DM must run side by side (not split around components) at the pair width.
fn assert_usb_pair_coupled(board: &vulcans_trials::board::Board) {
    let dp = board.net_name_to_id["/USB_DP"];
    let dm = board.net_name_to_id["/USB_DM"];
    for (a, b) in [(dp, dm), (dm, dp)] {
        let c = vulcans_trials::router::board_net_coupling_mm(board, a, b);
        assert!(c < 0.8, "{} strays {:.2} mm (mean) from its pair partner", board.nets[a], c);
    }
    let pair_w = vulcans_trials::router::net_target_width(board, dp);
    for t in board.tracks.iter().filter(|t| t.net_id == dp || t.net_id == dm) {
        assert!(t.width <= pair_w + 1e-6, "{} track wider than pair width: {:.3}", board.nets[t.net_id], t.width);
    }
    for net in [dp, dm] {
        let (mut wide, mut total) = (0.0, 0.0);
        for t in board.tracks.iter().filter(|t| t.net_id == net) {
            let l = (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1);
            total += l;
            if (t.width - pair_w).abs() < 1e-6 {
                wide += l;
            }
        }
        assert!(wide > 0.8 * total, "{} only {:.0}% at pair width", board.nets[net], 100.0 * wide / total);
    }
}

#[test]
fn test_p1p2esp_placed_board_usb_pair_coupled() {
    // The user's placed board (R14/L1 moved) previously split the pair around L1.
    let pcb_path = "/home/deece/src/p1p2esp/p1p2esp_vulcan.kicad_pcb";
    if !std::path::Path::new(pcb_path).exists() {
        return;
    }
    let locked: std::collections::HashSet<String> = vulcans_trials::board::DEFAULT_LOCKED_REFS.iter().map(|s| s.to_string()).collect();
    let mut board = vulcans_trials::board::Board::load(pcb_path, &locked).unwrap();
    board.tracks.clear();
    board.arcs.clear();
    board.vias.clear();
    let vulkan = vulcans_trials::vulkan::VulkanEngine::new();
    let mut grid = vulcans_trials::router::RoutingGrid::new(&board, 0.10, vulkan);
    let router = vulcans_trials::router::AutoRouter::new(12);
    let (routed, total) = router.route_all(&mut board, &mut grid, None, std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));
    assert_eq!(routed, total);
    assert_usb_pair_coupled(&board);
}

/// Reproduces the p1p2esp "+3.3V airwire from the via south of R9 to R17's south pad": the two
/// pads only meet through the In2.Cu +3.3V plane, but the input's stored zone fill predates the
/// R9 via (143.915, 91.82), so KiCad's file-open ratsnest (no refill) reported them disconnected.
#[test]
fn test_saved_board_drops_stale_zone_fill_and_refills_plane_connectivity() {
    let kicad_ok = Command::new("kicad-cli").arg("version").output().is_ok_and(|o| o.status.success());
    if !kicad_ok {
        return; // ponytail: plane connectivity can only be checked with KiCad's own filler
    }
    let dir = temp_test_dir("stale_zone_fill");
    let r = |reference: &str, x: f64, y: f64| {
        format!(
            r#"  (footprint "Resistor_SMD:R_0603_1608Metric" (layer "F.Cu") (at {x} {y} 270)
    (property "Reference" "{reference}" (at 0 0 0) (layer "F.SilkS"))
    (pad "1" smd rect (at -0.825 0 270) (size 0.8 0.95) (layers "F.Cu" "F.Mask" "F.Paste") (net "SIG_{reference}"))
    (pad "2" smd rect (at 0.825 0 270) (size 0.8 0.95) (layers "F.Cu" "F.Mask" "F.Paste") (net "+3.3V"))
  )
"#
        )
    };
    let stale_pcb = format!(
        r#"(kicad_pcb (version 20260206) (generator "pcbnew") (generator_version "10.0")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" signal) (2 "B.Cu" signal)
    (1 "F.Mask" user) (3 "B.Mask" user) (13 "F.Paste" user) (5 "F.SilkS" user "F.Silkscreen") (25 "Edge.Cuts" user))
  (setup (pad_to_mask_clearance 0))
  (net 0 "") (net 1 "+3.3V") (net 2 "SIG_R9") (net 3 "SIG_R17")
{r9}{r17}  (gr_rect (start 125 86) (end 169 124) (stroke (width 0.1) (type default)) (fill no) (layer "Edge.Cuts"))
  (zone (net 1) (net_name "+3.3V") (layer "In2.Cu") (hatch edge 0.5) (priority 1)
    (connect_pads (clearance 0.25)) (min_thickness 0.2)
    (fill yes (thermal_gap 0.5) (thermal_bridge_width 0.5) (island_removal_mode 0))
    (polygon (pts (xy 125.2 86.2) (xy 168.5 86.2) (xy 168.5 123.8) (xy 125.2 123.8)))
    (filled_polygon (layer "In2.Cu") (pts (xy 146 86.3) (xy 168.4 86.3) (xy 168.4 123.7) (xy 146 123.7)))
  )
  (segment (start 143.9150 91.8200) (end 144.1450 90.9950) (width 0.6000) (layer "F.Cu") (net 1))
  (segment (start 154.9400 96.0750) (end 155.7650 96.0750) (width 0.6000) (layer "F.Cu") (net 1))
  (via (at 143.9150 91.8200) (size 0.6000) (drill 0.3000) (layers "F.Cu" "B.Cu") (net 1))
  (via (at 155.7650 96.0750) (size 0.6000) (drill 0.3000) (layers "F.Cu" "B.Cu") (net 1))
)
"#,
        r9 = r("R9", 144.145, 90.17),
        r17 = r("R17", 154.94, 95.25)
    );
    let unconnected_without_refill = |pcb: &Path| -> usize {
        let report = pcb.with_extension("drc.json");
        let st = Command::new("kicad-cli")
            .args(["pcb", "drc", "--format", "json", "--output"])
            .arg(&report)
            .arg(pcb)
            .status()
            .expect("kicad-cli drc");
        assert!(st.success());
        let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(&report).unwrap()).unwrap();
        v["unconnected_items"].as_array().map_or(0, |a| a.len())
    };

    let in_pcb = dir.join("stale.kicad_pcb");
    fs::write(&in_pcb, &stale_pcb).unwrap();
    assert_eq!(unconnected_without_refill(&in_pcb), 1, "fixture must reproduce the stale-fill airwire");

    let mut board = Board::load(in_pcb.to_str().unwrap(), &HashSet::new()).unwrap();
    let out_pcb = dir.join("out.kicad_pcb");
    board.save_pcb(&out_pcb).unwrap();
    let saved = fs::read_to_string(&out_pcb).unwrap();
    assert!(!saved.contains("filled_polygon"), "stale zone fills must not be written back");
    assert!(saved.contains("(zone"), "zone outline must be preserved");

    assert!(Board::refill_zones_in_file(&out_pcb), "kicad-cli zone refill must succeed");
    assert_eq!(unconnected_without_refill(&out_pcb), 0, "refilled In2.Cu plane must join R9 and R17 +3.3V pads");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_esphome_1wire_sensor_example_autoplace_and_autoroute() {
    let pcb_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("esphome_1wire_sensor")
        .join("esphome_1wire_sensor.kicad_pcb");
    assert!(pcb_path.exists(), "Sample project PCB must exist at {:?}", pcb_path);

    let mut board = Board::load(pcb_path.to_str().unwrap(), &HashSet::new()).expect("Load sample PCB");
    assert_eq!(board.footprints.len(), 15, "Sample board must have 15 footprints");
    assert_eq!(board.clusters.len(), 4, "Sample schematic must define 4 functional clusters");
    let gnd_id = *board.net_name_to_id.get("/GND").expect("Sample board must contain /GND net");
    assert!(board.is_plane_net(gnd_id), "Root-sheet /GND must be recognized as ground plane net");

    let locked_refs: HashSet<String> = board
        .footprints
        .iter()
        .filter(|fp| fp.is_locked)
        .map(|fp| fp.reference.clone())
        .collect();
    assert!(locked_refs.contains("J1"), "J1 USB-C must be locked");
    assert!(locked_refs.contains("J2"), "J2 1-Wire terminal must be locked");
    assert!(locked_refs.contains("U3"), "U3 ESP32-C3-WROOM-02 must be locked");

    let vk = VulkanEngine::new();
    let live = Arc::new(Mutex::new(LiveTrialState::new(board.clone())));

    // Verify auto-routing on the sample board layout achieves 100% (23/23) completion
    let mut route_board = board.clone();
    let mut grid = RoutingGrid::new(&route_board, 0.10, vk.clone());
    let router = AutoRouter::new(10);
    let (routed, total) = router.route_all(
        &mut route_board,
        &mut grid,
        Some(Arc::clone(&live)),
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(total, 23, "Sample board must have 23 non-plane ratsnest connections");
    assert_eq!(routed, total, "Auto-router must route 100% (23/23) of connections on sample board");

    // Verify auto-placer legalizes and achieves 0 courtyard overlaps
    let cfg = PlacementConfig {
        duration_secs: 1,
        vulkan: vk,
        ..PlacementConfig::default()
    };
    let placed = run_auto_placer(
        &mut board,
        &cfg,
        Some(Arc::clone(&live)),
        Arc::new(AtomicBool::new(false)),
    );
    assert_eq!(placed.overlaps, 0, "Auto-placed sample board must have 0 courtyard overlaps");
}

#[test]
fn test_save_pcb_overwrites_with_timestamped_backup() {
    let dir = temp_test_dir("overwrite_backup");
    let target_pcb = dir.join("my_board.kicad_pcb");
    fs::write(&target_pcb, "(kicad_pcb (version 20240108) (generator \"original_board\"))\n").unwrap();

    let mut board = Board {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 10.0,
        max_y: 10.0,
        raw_content: "(kicad_pcb (version 20240108) (generator \"updated_board\"))\n".to_string(),
        ..Board::default()
    };

    board.save_pcb(&target_pcb).expect("save_pcb should succeed");

    // The target file now has updated content
    let content = fs::read_to_string(&target_pcb).unwrap();
    assert!(content.contains("updated_board"));

    // A backup file with suffix .kicad_pcb_bak.YYYYMMDDHHMMSS must exist
    let files: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    let bak_file = files
        .iter()
        .find(|f| f.starts_with("my_board.kicad_pcb_bak."));
    assert!(bak_file.is_some(), "Expected backup file, found: {:?}", files);

    let bak_path = dir.join(bak_file.unwrap());
    let bak_content = fs::read_to_string(&bak_path).unwrap();
    assert!(bak_content.contains("original_board"), "Backup must contain original content");

    let _ = fs::remove_dir_all(&dir);
}
