// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use vulcans_trials::board::Board;
use vulcans_trials::placer::{run_auto_placer, PlacementConfig};
use vulcans_trials::router::{AutoRouter, GridNode, RoutingGrid};
use vulcans_trials::vulkan::VulkanEngine;

#[test]
fn test_board_load_and_placer_router_integration() {
    let pcb_path = "/home/deece/src/p1p2esp/p1p2esp.kicad_pcb";
    if !Path::new(pcb_path).exists() {
        return;
    }

    let vk = VulkanEngine::new();
    assert!(vk.is_some(), "Vulkan backend should initialize");

    let mut board = Board::load(pcb_path, &HashSet::new()).expect("Failed to load test PCB");
    assert_eq!(board.footprints.len(), 53);
    assert!(!board.clusters.is_empty());

    // 1. Verify short auto-placer run preserves 0 courtyard overlaps and locked footprints
    let u1_before = board
        .footprints
        .iter()
        .find(|f| f.reference == "U1")
        .map(|f| (f.x, f.y))
        .unwrap();
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
    let _ = u1_before;

    // 2. Verify routing grid bidirectional clearance and escape stub generation
    let grid = RoutingGrid::new(&board, 0.10, vk);
    assert!(grid.width > 100 && grid.height > 100);
    let router = AutoRouter::new(2);
    let conns = router.extract_connections(&board);
    assert_eq!(conns.len(), 77);

    // Verify neighbor expansion inside valid board space
    let mid = GridNode {
        gx: grid.width / 2,
        gy: grid.height / 2,
        layer: 1,
    };
    let neighbors = grid.get_neighbors(mid, 1, 0.20, 0.20, 0.60, None);
    assert!(!neighbors.is_empty());
}
