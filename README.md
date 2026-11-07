# 🔥 Vulcan's Trials (`vulcans-trials`)

**Vulkan GPU + Multithreaded CPU Auto-Placer & Auto-Router for KiCad**  
Published by **InfernoEmbedded** — Licensed under **GNU GPLv3 (`GPL-3.0-only`)**

---

## Overview

**Vulcan's Trials** is a unified, hardware-accelerated PCB Auto-Placer and Auto-Router written in Rust for KiCad (`8.0` / `9.0` / `10.0`). It combines:

1. **Schematic-Aware Parallel Tempering Auto-Placer (`src/placer.rs`):**
   - Automatically parses sibling `.kicad_sch` files to extract functional circuit clusters (e.g., Buck Converter, Surge/ESD Front-End, Transceiver, MCU) and passive decoupling/bias affinities.
   - Enforces exact polygon courtyard clearances using Separating Axis Theorem (SAT) collision detection (`0` courtyard overlaps) while respecting locked components and off-board RF antenna overhang zones.
   - Minimizes ratsnest topological crossings and Euclidean Minimum Spanning Tree (MST) wirelength across all CPU cores using Parallel Tempering Simulated Annealing.

2. **Vulkan Compute + Negotiated-Congestion 3D Auto-Router (`src/vulkan.rs`, `src/router.rs`):**
   - Explicitly targets **Vulkan** (`wgpu::Backends::VULKAN`) with WGSL compute shaders (`DISTANCE_FIELD_WGSL`) to accelerate 2D obstacle Euclidean Distance Field transforms on the GPU.
   - Enforces bidirectional per-netclass clearance ($\max(\text{clearance}_A, \text{clearance}_B)$), project netclass track widths (Power, Differential Pairs, High-Speed Buses, Default), keepout zones, and fine-pitch SMD pad-to-via exclusions.
   - Generates deterministic staggered pin-escape stubs ($\Delta \ge 0.75\text{ mm}$) for fine-pitch QFN, BGA, and USB-C packages, plus automated ground-plane via stitching.
   - Supports dynamic component nudging (`NudgeEngine`) and multi-pass PathFinder negotiated-congestion rip-up and reroute.

3. **Real-Time Interactive GUI (`src/gui.rs`):**
   - Live 2D PCB viewport with pan/zoom, layer toggles (`F.Cu`, `B.Cu`, Courtyards, Ratsnest, Vias, Keepouts, Reference Designators), and live trial progress.
   - Real-time **CPU Load %** (`/proc/stat`) and **Vulkan GPU Load %** (`VulkanEngine` compute duty cycle + `nvidia-smi` / DRM sysfs) gauges and sparkline histories.

---

## Quick Start

### 1. Build Release Binary & Run Self-Test
```bash
git clone https://github.com/InfernoEmbedded/vulcans-trials.git
cd vulcans-trials
cargo build --release
./target/release/vulcans-trials --self-test
```

### 2. Launch the Interactive GUI
```bash
./target/release/vulcans-trials --input /path/to/board.kicad_pcb --output /path/to/board_routed.kicad_pcb
```

### 3. Run in Headless CLI Mode
```bash
# Auto-route only
./target/release/vulcans-trials --headless --mode route \
  --input /path/to/board.kicad_pcb \
  --output /path/to/board_routed.kicad_pcb

# Full Trial (Auto-Place + Auto-Route)
./target/release/vulcans-trials --headless --mode both --duration 30 \
  --input /path/to/board.kicad_pcb \
  --output /path/to/board_vulcan.kicad_pcb
```

### 4. KiCad Action Plugin Installation

#### Option A: KiCad Plugin and Content Manager (PCM)
In KiCad (8.0, 9.0, or 10.0), open **Plugin and Content Manager → Settings (gear icon) → Add Repository** and add:
```
https://infernoembedded.github.io/vulcans-trials/repository.json
```
Then search for **Vulcan's Trials** and click **Install**.

#### Option B: Manual Plugin Symlink
Symlink `kicad_plugin` into your KiCad scripting plugins directory:
```bash
ln -s "$(pwd)/kicad_plugin" ~/.local/share/kicad/10.0/scripting/plugins/vulcans_trials
```

---

## Commercial Feature Comparison & Roadmap
See [`ROADMAP.md`](./ROADMAP.md) for the full feature comparison against commercial EDA autorouters (Cadence Allegro / Specctra, Altium ActiveRoute, Siemens Xpedition, Zuken CR-8000, Eremex TopoR) and the 6-phase development roadmap.

## License
This project is licensed under the **GNU General Public License v3.0 only** (`GPL-3.0-only`). See [`LICENSE`](./LICENSE) for full terms.
