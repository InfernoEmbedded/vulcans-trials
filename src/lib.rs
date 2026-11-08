// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! # Vulcan's Trials (`vulcans_trials`)
//!
//! Unified Vulkan GPU and multithreaded CPU PCB **Auto-Placer** and **Auto-Router**
//! library for KiCad (`8.0` / `9.0` / `10.0`), developed by InfernoEmbedded.
//!
//! ## Architecture & Modules
//!
//! - [`board`]: Unified S-expression `.kicad_pcb` and `.kicad_sch` parser, Separating Axis Theorem (SAT)
//!   convex polygon courtyard collision math, Prim's Euclidean Minimum Spanning Tree (MST) ratsnest
//!   generator, in-place `.kicad_pcb` writer, Specctra `.ses` exporter, and shared live trial state.
//! - [`cuda`]: Optional bare-metal NVIDIA Driver API (`libcuda.so`) PTX compute backend for distance
//!   fields, placement batch evaluation, and parallel Simulated Annealing Markov chains.
//! - [`gui`]: Real-time interactive `eframe`/`egui` 2D PCB canvas (`VulcansTrialsApp`) with layer
//!   visibility toggles, pan/zoom controls, and live CPU/GPU load sparklines.
//! - [`ipc`]: KiCad 10 IPC Unix domain socket / named pipe client for 60 FPS live board mutation
//!   streaming and board synchronization.
//! - [`mcp`]: Model Context Protocol (MCP) JSON-RPC 2.0 server over standard I/O (`stdio`) and TCP,
//!   exposing placement, routing, DRC, signal integrity, and telemetry tools to AI agents.
//! - [`memlog`]: Low-overhead Linux `/proc/self/statm` and `/proc/self/status` Resident Set Size (RSS)
//!   memory logger and 20 ms background out-of-memory (OOM) watchdog thread.
//! - [`placer`]: Courtyard overlap legalizer and multi-core + GPU Parallel Tempering Simulated Annealing
//!   (PT-SA) auto-placer with schematic cluster and decoupling capacitor affinity.
//! - [`router`]: Multi-layer 3D A* and PathFinder negotiated-congestion autorouter, continuous 45-degree
//!   track shove/nudge engine, BGA dog-bone/VIPPO escape solver, fine-pitch staggered escape stub
//!   generator, TopoR-style any-angle rubber-band relaxation, differential pair phase matching,
//!   meander length tuning, and automated ground-plane via stitching.
//! - [`swap`]: Kuhn-Munkres (Hungarian) minimum-cost bipartite matching and 2D Morton Linear Bounding
//!   Volume Hierarchy (LBVH) crossing untangler for automated pin and functional gate swapping.
//! - [`vulkan`]: Cross-vendor `wgpu` (Vulkan backend) compute engine for 2D Euclidean Distance Field (EDF)
//!   transforms, LBVH construction and ray-cast crossing queries, batch placement scoring, and system
//!   CPU/GPU telemetry sampling.

pub mod board;
pub mod cuda;
pub mod gui;
pub mod ipc;
pub mod mcp;
pub mod memlog;
pub mod placer;
pub mod router;
pub mod swap;
pub mod vulkan;
