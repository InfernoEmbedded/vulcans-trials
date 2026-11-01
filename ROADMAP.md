# Vulcan's Trials — Commercial EDA Feature Matrix & Roadmap

**Organization:** InfernoEmbedded  
**License:** GNU General Public License v3.0 (GPL-3.0-only)  
**Architecture:** Rust + Vulkan Compute (`wgpu` / SPIR-V) + Rayon Multi-Core CPU + `egui` Live Telemetry GUI

---

## 1. Commercial Autorouter & Autoplacer Feature Landscape

Below is a comprehensive breakdown of features offered by flagship commercial PCB Place & Route suites (**Cadence Allegro / Specctra**, **Altium Designer ActiveRoute**, **Siemens Xpedition**, **Zuken CR-8000**, and **Eremex TopoR**) and their status in **Vulcan's Trials**.

| Category | Commercial Feature | Reference Tools | Vulcan's Trials Status |
| :--- | :--- | :--- | :--- |
| **Placement** | Courtyard SAT collision & minimum spacing legalization | Allegro, Altium | **Implemented (v0.1)** |
| **Placement** | Parallel Tempering / Simulated Annealing wirelength & crossing minimization | Allegro, Xpedition | **Implemented (v0.1)** |
| **Placement** | Schematic-driven functional clustering (power stages, transceivers, MCU) | Altium, Zuken | **Implemented (v0.1)** |
| **Placement** | Locked component & mounting hole preservation | All | **Implemented (v0.1)** |
| **Routing Core** | Negotiated Congestion Rip-Up & Reroute (PathFinder / Maze A*) | Specctra, Xpedition | **Implemented (v0.1)** |
| **Routing Core** | Vulkan GPU Euclidean Distance Field & obstacle cost transform | Custom (Vulcan's Trials) | **Implemented (v0.1)** |
| **Routing Core** | Dynamic Push-and-Shove component nudging to open blocked channels | Specctra, TopoR | **Implemented (v0.1)** |
| **Design Rules** | Symmetric bidirectional netclass clearance ($\max(C_A, C_B)$) | All | **Implemented (v0.1)** |
| **Design Rules** | Keepout zones (per-layer copper, via, and antenna overhang keepouts) | All | **Implemented (v0.1)** |
| **IC Escape** | Deterministic fine-pitch QFN / USB-C escape generator with staggered vias | Allegro, Xpedition | **Implemented (v0.1)** |
| **Power & GND** | Automated SMD-to-plane ground stitching via placement | Altium, Allegro | **Implemented (v0.1)** |
| **High-Speed** | Coupled Differential Pair routing (constant gap, coupled turns, via symmetry) | Allegro, Altium, Xpedition | **Phase 2 (v0.2)** |
| **High-Speed** | Serpentine / Trombone delay & phase length tuning (ps / mm matching) | Allegro, Xpedition, Zuken | **Phase 2 (v0.2)** |
| **High-Speed** | Impedance-controlled layer-specific track width adjustment | Allegro, Altium | **Phase 2 (v0.2)** |
| **Topological** | Any-angle / Arc / Teardrop topological rubber-band routing | Eremex TopoR, Xpedition | **Phase 3 (v0.3)** |
| **Topological** | Post-route trace glossing, corner mitering, and via minimization pass | Specctra, Altium | **Phase 3 (v0.3)** |
| **HDI & BGA** | Dog-bone, via-in-pad, blind/buried & microvia BGA fanout solver | Allegro, Zuken, Xpedition | **Implemented (v0.4)** |
| **HDI & BGA** | Pin & gate swapping optimization (FPGA / MCU / multi-unit op-amps) | Allegro, Xpedition | **Implemented (v0.4)** |
| **SI / PI / DFM** | Crosstalk-aware parallel run-length limit & guard-trace insertion | Allegro, HyperLynx | **Implemented (v0.5)** |
| **SI / PI / DFM** | Return-path continuity & split-plane crossing avoidance | Allegro, Zuken | **Implemented (v0.5)** |
| **SI / PI / DFM** | Acid-trap removal, solder-mask sliver prevention, and teardrop generation | Altium, Valor NPI | **Implemented (v0.5)** |
| **Interactive** | Guided River Routing / Bus Bundle sketching ("ActiveRoute" style) | Altium, Xpedition | **Implemented (v1.0)** |

---

## 2. Phased Development Roadmap

### Phase 1: v0.1 — Foundation & Unified Vulkan Place-and-Route (Current)
- [x] Unified `.kicad_pcb` and `.kicad_sch` parser + Specctra `.ses` exporter.
- [x] Vulkan GPU compute backend (`wgpu` + WGSL compute shaders) for 2D obstacle distance fields and batch cost evaluation.
- [x] Multi-threaded CPU Parallel Tempering Simulated Annealing autoplacer with schematic cluster awareness.
- [x] 3D multi-layer negotiated-congestion A* autorouter with symmetric netclass clearances.
- [x] Deterministic staggered pin escapes ($\Delta \ge 0.75\text{ mm}$) for $0.5\text{ mm}$ pitch QFN and USB-C packages.
- [x] Dynamic component push-and-shove nudging when routing corridors are blocked.
- [x] Automated ground plane stitching vias for isolated SMD ground pads.
- [x] Native real-time GUI (`egui`) displaying live board state, ratsnest, routed F.Cu/B.Cu tracks, vias, and CPU/GPU load telemetry.
- [x] KiCad Action Plugin integration (`kicad_plugin/`).

### Phase 2: v0.2 — High-Speed Digital & Differential Pairs
- [x] **Coupled Differential Pair Router:** Simultaneous dual-wavefront A* search maintaining exact differential gap (`diff_pair_gap`) and neck-down rules near fine-pitch pads.
- [x] **Intra-Pair Phase Matching:** Automatic asymmetrical bump insertion right at the launch/termination mismatch to correct skew within $\pm 0.05\text{ mm}$.
- [x] **Inter-Net Length Tuning (Trombone / Accordion):** Automated serpentine meander generator for DDR, SDIO, QSPI, and RMII buses with configurable amplitude and miter radius.
- [x] **Net Topology Constraints:** Daisy-chain, fly-by, T-branch, and star-point routing rules per net.

### Phase 3: v0.3 — Glossing, Via Optimization & Any-Angle Topological Routing
- [x] **Post-Route Glossing Pass:** Rubber-band trace straightening, stair-step elimination, and $45^\circ$ / arc corner smoothing.
- [x] **Via Reduction Pass:** Iterative layer-reassignment pass that removes redundant layer transitions when a single-layer detour is within length budget.
- [x] **Track Push-and-Shove (Continuous Coordinates):** Off-grid continuous shove engine allowing existing tracks to flex smoothly around newly routed nets.
- [ ] **TopoR-Style Any-Angle Mode:** Delaunay triangulation + rubber-band topological routing for ultra-compact boards and curved enclosures.

### Phase 4: v0.4 — HDI, BGA Fanout & Pin/Gate Swapping
- [x] **Automated BGA Escape Solver:** Quadrant dog-bone fanout, channel-routing, and blind/buried microvia stack support for $0.4\text{ mm}$–$0.8\text{ mm}$ pitch BGAs.
- [x] **Pin & Gate Swapping:** Bipartite matching (Hungarian algorithm) on swappable MCU GPIO banks and multi-unit logic/op-amp gates during placement to untangle ratsnest crossings before routing begins.
- [ ] **Multi-Board & Rigid-Flex Zones:** Layer stackup regions with bend-area trace perpendicularity constraints.

### Phase 5: v0.5 — Signal Integrity (SI), Power Integrity (PI) & DFM
- [x] **Crosstalk & Parallelism Limiter:** Penalize parallel adjacent trace segments exceeding a maximum coupled length threshold ($L_{\text{coupled}}$).
- [x] **Reference Plane Return-Path Check:** Penalize high-speed signals crossing splits or cutouts on the adjacent reference plane layer; auto-insert return-path stitching capacitors/vias near signal vias.
- [x] **DFM Hardening:** Automatic teardrops on pad/via junctions, solder mask bridge validation against fab rules (JLCPCB / PCBWay presets), and copper-to-edge clearance enforcement.

### Phase 6: v1.0 — Interactive Guided Routing & KiCad IPC Live Sync
- [x] **User-Guided Bundle Sketching:** Draw a rough poly-line path in the GUI or KiCad to guide a bus of nets along a desired channel.
- [ ] **Full KiCad 10 IPC Socket Live Streaming:** Stream placement and trace updates directly into the open KiCad canvas at 60 FPS without file reloads.
- [ ] **One-Click KiCad PCM Package:** Automated CI/CD builds for Linux, Windows, and macOS published to the official KiCad Plugin and Content Manager repository.
