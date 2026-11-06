# AGENTS.md — Vulcan's Trials (`vulcans-trials`)

**Organization:** InfernoEmbedded  
**License:** GNU GPLv3 (`GPL-3.0-only`) — Every source file must retain its `SPDX-License-Identifier: GPL-3.0-only` header.

---

## 1. Project Mission & Handoff Context

**Vulcan's Trials** is a unified, Vulkan GPU + multithreaded CPU PCB **Auto-Placer** and **Auto-Router** for KiCad (`8.0` / `9.0` / `10.0`), written in Rust with an interactive `eframe`/`egui` 2D GUI and live CPU/GPU telemetry.

### Current Architecture
- [`src/board.rs`](./src/board.rs): Unified S-expression `.kicad_pcb` and `.kicad_sch` parser, SAT convex polygon courtyard collision math (`Polygon::intersects_convex`), Prim's Euclidean MST ratsnest generator, in-place `.kicad_pcb` writer, Specctra `.ses` exporter, and shared [`LiveTrialState`](./src/board.rs) state.
- [`src/vulkan.rs`](./src/vulkan.rs): [`VulkanEngine`](./src/vulkan.rs) targeting `wgpu::Backends::VULKAN` with a WGSL compute shader (`DISTANCE_FIELD_WGSL`) for 2D Euclidean Distance Field transforms on the routing obstacle grid, plus [`SystemTelemetry`](./src/vulkan.rs) sampling `/proc/stat` (CPU %) and Vulkan compute duty cycle + `nvidia-smi` / DRM sysfs (GPU %).
- [`src/placer.rs`](./src/placer.rs): Courtyard legalizer (`legalize_placement`) and multi-core Parallel Tempering Simulated Annealing auto-placer (`run_auto_placer`) with schematic cluster and passive decoupling affinity.
- [`src/router.rs`](./src/router.rs): Multi-layer 3D A* and PathFinder negotiated-congestion autorouter (`AutoRouter`, `RoutingGrid`, `NudgeEngine`) with bidirectional netclass clearances, staggered fine-pitch IC escape stubs, and automated ground-plane via stitching.
- [`src/gui.rs`](./src/gui.rs): Real-time `eframe::App` GUI (`VulcansTrialsApp`) with interactive 2D PCB canvas (pan/zoom, layer toggles) and live CPU/GPU load sparklines.
- [`src/main.rs`](./src/main.rs): CLI + GUI entrypoint (`--headless`, `--mode place|route|both`, `--self-test`).
- [`kicad_plugin/`](./kicad_plugin): KiCad `ActionPlugin` toolbar wrapper (`__init__.py`, `metadata.json`).
- [`ROADMAP.md`](./ROADMAP.md): Commercial EDA feature matrix and 6-phase development roadmap.

---

## 2. Ponytail Engineering Rules (Mandatory)

All agents working on this repository MUST follow the **Ponytail** discipline: write the simplest, shortest, most direct code that solves the actual problem without speculative bloat.

1. **Question Existence First (YAGNI):**
   - Do not build config layers, plugin registries, abstract factory traits, or wrapper types for hypothetical future use cases. Solve the concrete requirement in front of you.
2. **Stdlib & Existing Crates Before New Dependencies:**
   - Reach for the Rust standard library (`std::fs`, `std::collections`, `std::sync`) and already-included crates (`wgpu`, `eframe`, `egui`, `rayon`, `clap`, `rand`, `serde`) before adding any crate to `Cargo.toml`.
   - Never add heavy third-party crates for trivial tasks (e.g., `/proc/stat` parsing takes 20 lines of `std::fs::read_to_string`).
3. **One Function Before a Trait Hierarchy:**
   - Prefer flat structs, enums, and free functions. Do not introduce a `Trait` if there is only one implementation.
   - Inline helpers until there are at least 3 genuine call sites.
4. **Legible, Self-Documenting Constants:**
   - Name physical and algorithmic constants clearly (e.g., `KEEPOUT_NET`, `FREE_NET`, `DEFAULT_LOCKED_REFS`) so the math reads cleanly without verbose comment essays.
5. **Debt Honesty (`// ponytail:`):**
   - Whenever you deliberately take a shortcut or defer an edge case to keep an implementation lean, leave a grep-friendly comment:
     ```rust
     // ponytail: <what was deferred and why>
     ```
6. **Never Compromise PCB Physical & Electrical Correctness:**
   - Simplicity applies to software structure, **never** to DRC/ERC rules.
   - Always enforce symmetric bidirectional copper clearances:
     $$\text{Clearance}(A, B) = \frac{\text{Width}_A}{2} + \max(\text{Clearance}_A, \text{Clearance}_B) + \frac{\text{Width}_B}{2}$$
   - Never allow vias to overlap fine-pitch SMD pads of another net (`smd_pad_cells`).
   - Always maintain $\Delta \ge 0.75\text{ mm}$ stagger between adjacent 0.5 mm-pitch QFN/USB-C escape stubs so via annular rings ($0.60\text{ mm}$) + clearance ($0.20\text{ mm}$) do not choke adjacent pin channels.

---

## 3. Mandatory Testing Policy (Required on ALL Commits)

**Every commit MUST include both unit tests and integration tests, and the full test suite must pass before committing.**

### 1. Unit Tests (`#[cfg(test)]` in `src/*.rs`)
- Any new geometry primitive, cost function, parser rule, clearance check, or routing heuristic MUST have accompanying unit tests in the module's `#[cfg(test)] mod tests` block.
- Unit tests must be deterministic, fast (< 100 ms each), and test both valid and boundary/violation cases (e.g., polygon overlap vs non-overlap, segment crossing vs shared-endpoint non-crossing, netclass clearance enforcement).

### 2. Integration Tests (`tests/*.rs` and `--self-test`)
- End-to-end integration tests reside in [`tests/integration_test.rs`](./tests/integration_test.rs) and test the full pipeline across board loading, Vulkan compute distance field evaluation, courtyard legalization, auto-placement, and multi-layer auto-routing, plus `.kicad_pcb` and `.ses` output generation.
- The built-in CLI self-test (`./target/release/vulcans-trials --self-test`) must also remain green.

### 3. Pre-Commit Verification Gate
Before creating any git commit, run:
```bash
cargo test --release
./target/release/vulcans-trials --self-test
```
Both commands MUST exit with code `0` and zero test failures.

---

## 4. Mandatory Commit Message Requirements

Every commit in this repository MUST follow strict structured commit formatting. Never create single-line commits or generic commit messages.

### Commit Format Template
```text
<type>(<scope>): <concise subject summary under 72 chars>

<Detailed explanation of WHY this change was made: the electrical,
physical (DRC clearances, impedance coupling, acid traps, annular rings),
performance, or architectural problem being addressed.>

<Detailed explanation of WHAT changed: specific algorithms, data structures,
functions added/modified, files touched, CLI flags/GUI controls added,
and verification tests.>

Signed-off-by: <Author Name> <author-email@example.com>
Assisted-by: <Agent Name> <agent-email@example.com>
```

### Commit Guidelines
1. **Conventional Prefix & Scope:** Use standard types (`feat`, `fix`, `docs`, `refactor`, `test`, `perf`) with meaningful scope (e.g. `router`, `placer`, `board`, `gui`, `ipc`, `gpu`, `cuda`, `memlog`).
2. **Explain "Why" and "What":** The commit body must provide technical depth. State the root cause or engineering motivation, followed by concrete implementation details.
3. **Mandatory Developer Trailers:** Every commit MUST conclude with both the author's Signed-off-by (matching author git identity) and the assistant's Assisted-by trailers:
   - `Signed-off-by: <Author Name> <author-email@example.com>`
   - `Assisted-by: <Agent Name> <agent-email@example.com>`
4. **Atomic Commits:** Separate unrelated features, refactors, and bug fixes into distinct commits that build and pass tests independently.

---

## 5. KiCad File & IPC Safety Rules

- **Never corrupt an open `.kicad_pcb`:** When testing against an external project board (such as `/home/deece/src/p1p2esp/p1p2esp.kicad_pcb`), treat the input file as read-only and always write `--output` to a separate file or temporary path unless explicitly instructed otherwise.
- **DRC Verification:** When modifying routing or via-stitching logic, verify the generated `.kicad_pcb` using KiCad's CLI DRC engine:
  ```bash
  kicad-cli pcb drc --output /tmp/vulcan_drc.json --format json <output.kicad_pcb>
  ```
  Ensure `0` unconnected component pads and `0` copper shorts.
