// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! Multi-objective PCB component auto-placement, courtyard legalization, and spatial indexing.
//!
//! This module implements the complete component placement pipeline for Vulcan's Trials,
//! combining analytical and topological seeding with hybrid Vulkan GPU + multi-core CPU
//! Parallel Tempering Simulated Annealing (SA).
//!
//! # Architecture & Pipeline
//!
//! 1. **Holding Palette & Initial Seeding ([`place_all_holding_components_at_best_locations`], [`prioritize_and_position_initial_components`]):**
//!    Unplaced components from the staging palette and unlocked board footprints are situated in
//!    strict priority order:
//!    - High-speed differential pair components (e.g., USB `D+`/`D-` ESD/series passives) are aligned
//!      along direct routing corridors adjacent to their connector or IC anchors.
//!    - Bypass and decoupling capacitors are placed radially outside their target active device's
//!      power pin with the power pad oriented inward to minimize high-frequency loop inductance.
//!    - Remaining passives and ICs are positioned near the weighted barycenter of their connected
//!      signal and power pins or schematic cluster anchors.
//!
//! 2. **Spatial Indexing & Courtyard Legalization ([`PlacedBoardSpatialIndex`], [`legalize_placement`], [`shove_overlapping_components`]):**
//!    A 2D Morton Linear Bounding Volume Hierarchy (LBVH) over placed footprint courtyards and
//!    keepouts accelerates candidate pose legality queries to $O(\log N_f)$ AABB pruning followed
//!    by exact Separating Axis Theorem (SAT) convex polygon intersection checks (`Polygon::intersects_convex`),
//!    honoring per-component fine-pitch IC escape margins and opposite-side (`F.Cu` vs. `B.Cu`) SMD
//!    non-collision rules. Overlapping initial states are resolved via a GPU SA pre-pass and
//!    multi-threaded CPU repulsive legalization.
//!
//! 3. **Topological & Routability Cost Evaluation ([`PlacementState::evaluate_cost`]):**
//!    Each candidate placement state is scored against a composite energy objective:
//!    - **Euclidean MST Wirelength:** Total length of Prim's Minimum Spanning Tree ratsnest segments.
//!    - **Ratsnest Crossings:** Exact segment-segment intersection count accelerated via a 2D Morton
//!      LBVH over ratsnest edges ($O(E \log E)$).
//!    - **Corridor Cut Congestion ([`compute_corridor_capacity`]):** Evaluates physical copper track
//!      capacity across narrow apertures between adjacent component courtyards against the number of
//!      shared and cut ratsnest nets traversing the gap.
//!    - **Signal Flow & Bus Ribbon Alignment ([`compute_flow_alignment_penalty_for_link`]):** Penalizes
//!      lateral pin order inversions (ribbon twisting), direct wire crossings, and angular skew across
//!      multi-pin component buses.
//!    - **Schematic Cluster & Passive Affinity ([`build_passive_ties`]):** Quadratic/linear distance
//!      penalties keeping decoupling capacitors, pull-up/termination passives, and schematic hierarchy
//!      clusters within their target radius.
//!    - **Hard Physical Constraints:** Heavy penalties for courtyard polygon overlaps, board outline
//!      excursions, keepout zone intrusions, and unnecessary layer flips.
//!
//! 4. **Hybrid GPU + Multi-Core CPU Parallel Tempering SA ([`run_auto_placer`]):**
//!    - **Vulkan GPU Chains (`run_gpu_placement_batch`):** Thousands of independent Markov chains
//!      evaluate candidate moves in parallel on the GPU using flattened courtyard polygon buffers
//!      ([`build_gpu_sa_descriptors`]), periodically injecting improved legal states into the global best.
//!    - **CPU Parallel Tempering Replicas:** Multiple Rayon worker threads run SA across a geometric
//!      temperature ladder using 7 move operators (pair swap, barycentric centroid pull, rigid cluster
//!      group shift, $90^\circ/180^\circ/270^\circ$ rotation, SMD layer flip, passive tie affinity jump,
//!      and multi-scale grid shifts).
//!    - **Replica Exchange & Basin Hopping ([`metropolis_replica_exchange`]):** Adjacent temperature
//!      rungs exchange states via the Metropolis-Hastings criterion at epoch boundaries, while stagnation
//!      triggers thermal pulses and legal basin-hopping swaps.
//!    - **Pin/Gate Swapping & Silkscreen Polish:** Pre- and post-placement passes invoke
//!      [`optimize_pin_and_gate_swapping`] to untangle swappable MCU/FPGA pins and multi-unit gates,
//!      while board-level silkscreen reference designator placement (`Board::auto_place_silkscreen_Step`)
//!      ensures legible, non-overlapping labels after component placement settles.

use crate::board::{Board, LiveTrialState, Point, Polygon, Segment, TrialPhase, AABB};
use crate::ipc::KiCadIpcClient;
use crate::vulkan::{
    build_gpu_lbvh, query_lbvh_crossing_segments, GpuBvhNode, GpuCourtyardBuffers, GpuSaParams,
    PlacementBatchParams, VulkanEngine,
};
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Re-exported pin and gate swap optimizer invoked before and after Simulated Annealing placement.
pub use crate::swap::optimize_pin_and_gate_swapping;

/// Hyperparameters, physical clearances, energy weights, and hardware handles for the PCB auto-placer.
#[derive(Debug, Clone)]
pub struct PlacementConfig {
    /// Total wall-clock duration in seconds for the Simulated Annealing auto-placement run (`s`).
    pub duration_secs: u64,
    /// Placement coordinate quantization grid step in millimeters (`mm`, default `0.635` mm / 25 mil).
    pub grid_step_mm: f64,
    /// Minimum physical clearance margin added around component courtyards in millimeters (`mm`).
    pub min_spacing_mm: f64,
    /// Energy penalty weight per intersecting pair of ratsnest airwire segments.
    pub crossing_weight: f64,
    /// Energy penalty weight applied to squared routing corridor demand overflow between nearby footprints.
    pub corridor_congestion_weight: f64,
    /// Energy penalty weight for multi-pin bus lateral order inversions, crossings, and angular skew.
    pub flow_alignment_weight: f64,
    /// Energy penalty weight per millimeter (`mm`) of total Euclidean MST ratsnest wirelength.
    pub wirelength_weight: f64,
    /// Base energy penalty weight per pair of overlapping component courtyards (scaled by overlap area in $\text{mm}^2$).
    pub overlap_penalty: f64,
    /// Base energy penalty weight for courtyard vertices or pads violating the board outline or keepout zones.
    pub boundary_penalty: f64,
    /// Energy penalty weight attracting non-decoupling passives (pull-ups, series resistors) to their anchor pin.
    pub passive_affinity_weight: f64,
    /// Target maximum pad-to-pad distance in millimeters (`mm`) before quadratic/linear passive affinity penalties ramp up.
    pub passive_max_dist: f64,
    /// Energy penalty weight attracting 2-pin decoupling capacitors to their active IC power pin.
    pub decoupling_affinity_weight: f64,
    /// Target maximum pad-to-pad distance in millimeters (`mm`) between a decoupling capacitor and its IC power pin.
    pub decoupling_max_dist: f64,
    /// Whether the placer is permitted to flip SMD-only components between `F.Cu` (`0`) and `B.Cu` (`1`).
    pub allow_layer_flip: bool,
    /// Whether to inflate component placement spacing by each footprint's fine-pitch escape routing margin.
    pub enable_escape_clearance: bool,
    /// Dimensionless multiplier applied to each footprint's `escape_margin` when `enable_escape_clearance` is enabled.
    pub escape_margin_mult: f64,
    /// Maximum single-move coordinate jump radius in millimeters (`mm`) at full annealing temperature.
    pub max_jump_radius_mm: f64,
    /// Probability in `[0.0, 1.0]` of selecting the pairwise component swap move operator during SA.
    pub pair_swap_prob: f64,
    /// Probability in `[0.0, 1.0]` of selecting the barycentric net-centroid pull move operator during SA.
    pub centroid_pull_prob: f64,
    /// Dimensionless scaling factor applied to the Parallel Tempering base temperature ladder.
    pub temp_scale: f64,
    /// Optional shared handle to the Vulkan compute engine for GPU-accelerated SA chains and batch pose scoring.
    pub vulkan: Option<Arc<VulkanEngine>>,
    /// Optional shared handle to a live KiCad IPC client for real-time footprint delta streaming.
    pub ipc: Option<Arc<Mutex<KiCadIpcClient>>>,
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            duration_secs: 20,
            grid_step_mm: 0.635,
            min_spacing_mm: 0.25,
            crossing_weight: 25.0,
            corridor_congestion_weight: 20.0,
            flow_alignment_weight: 15.0,
            wirelength_weight: 1.0,
            overlap_penalty: 2500.0,
            boundary_penalty: 50000.0,
            passive_affinity_weight: 35.0,
            passive_max_dist: 3.5,
            decoupling_affinity_weight: 60.0,
            decoupling_max_dist: 2.0,
            allow_layer_flip: false,
            enable_escape_clearance: true,
            escape_margin_mult: 1.0,
            max_jump_radius_mm: 20.0,
            pair_swap_prob: 0.15,
            centroid_pull_prob: 0.20,
            temp_scale: 1.0,
            vulkan: None,
            ipc: None,
        }
    }
}

impl crate::board::Footprint {
    /// Computes the effective courtyard placement expansion spacing in millimeters (`mm`) for this
    /// footprint under the given [`PlacementConfig`].
    ///
    /// When `config.enable_escape_clearance` is `true`, the base `config.min_spacing_mm` is
    /// augmented by `2.0 * self.escape_margin * config.escape_margin_mult` so dense QFN/BGA/connector
    /// footprints reserve surrounding routing channels for staggered escape vias.
    ///
    /// # Arguments
    /// * `config` - Placement configuration specifying base spacing and escape clearance settings.
    ///
    /// # Returns
    /// Effective outward courtyard inflation distance in millimeters (`mm`).
    pub fn effective_spacing(&self, config: &PlacementConfig) -> f64 {
        if config.enable_escape_clearance {
            config.min_spacing_mm + 2.0 * self.escape_margin * config.escape_margin_mult
        } else {
            config.min_spacing_mm
        }
    }
}

/// Complete snapshot of a candidate PCB placement configuration and its evaluated objective metrics.
///
/// Stores world-space poses (`positions`, `rotations`, `layers`), pre-transformed geometry caches
/// (`cached_courtyards`, `cached_bounds`, `cached_pads`) for fast incremental updates during
/// Simulated Annealing, and the decomposed cost components computed by [`PlacementState::evaluate_cost`].
#[derive(Debug, Clone)]
pub struct PlacementState {
    /// World-space origin coordinates `(x, y)` in millimeters (`mm`) for each footprint, indexed by footprint index.
    pub positions: Vec<Point>,
    /// Orientation angle in degrees (`[0.0, 360.0)`) for each footprint, indexed by footprint index.
    pub rotations: Vec<f64>,
    /// Copper layer index (`0` for `F.Cu`, `1` for `B.Cu`) for each footprint, indexed by footprint index.
    pub layers: Vec<usize>,
    /// Cached world-space inflated courtyard convex polygons for each footprint.
    pub cached_courtyards: Vec<Vec<Polygon>>,
    /// Cached world-space union axis-aligned bounding box ([`AABB`]) for each footprint's courtyards.
    pub cached_bounds: Vec<AABB>,
    /// Cached world-space pad center coordinates in millimeters (`mm`) for each footprint (`cached_pads[fp_idx][pad_idx]`).
    pub cached_pads: Vec<Vec<Point>>,
    /// Hash of the board's pad-to-net assignment used to invalidate `cached_connectivity` after pin/gate swaps.
    pub cached_connectivity_hash: u64,
    /// Shared topological connectivity cache mapping component pairs `(comp_a, comp_b)` to shared net IDs and multi-pin [`ComponentBusLink`]s.
    pub cached_connectivity: Arc<(HashMap<(usize, usize), Vec<usize>>, Vec<ComponentBusLink>)>,
    /// Composite weighted objective energy score (lower is better).
    pub cost: f64,
    /// Number of pairwise intersections among Euclidean MST ratsnest segments.
    pub crossings: usize,
    /// Total squared routing track demand overflow across narrow inter-component corridors.
    pub corridor_congestion: f64,
    /// Total topological bus ribbon twisting, crossing, and angular skew penalty across [`ComponentBusLink`]s.
    pub flow_alignment_penalty: f64,
    /// Total Euclidean MST ratsnest wirelength in millimeters (`mm`).
    pub wirelength: f64,
    /// Number of colliding footprint courtyard pairs.
    pub overlaps: usize,
    /// Number of unlocked footprints violating the board outline or keepout zones.
    pub boundary_violations: usize,
}

/// Directed affinity tether binding a passive component pad to an anchor component pad.
///
/// Used by both the initial prioritizer and the Simulated Annealing cost function to keep
/// decoupling capacitors, bulk capacitors, pull-up/pull-down resistors, and series termination
/// passives tightly coupled to the specific IC or connector pin they serve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassiveTie {
    /// Footprint index of the passive component (`R`, `C`, `L`, `F`, or `D`) in `board.footprints`.
    pub passive_idx: usize,
    /// Pad index on the passive footprint (`board.footprints[passive_idx].pads[passive_pad_idx]`).
    pub passive_pad_idx: usize,
    /// Footprint index of the anchor device (typically an IC `U*`, transistor `Q*`, or connector `J*`).
    pub anchor_idx: usize,
    /// Pad index on the anchor footprint (`board.footprints[anchor_idx].pads[anchor_pad_idx]`).
    pub anchor_pad_idx: usize,
    /// Maximum target Euclidean distance in millimeters (`mm`) between the tied pads before excess penalties apply.
    pub max_dist: f64,
    /// Dimensionless multiplier scaling the affinity energy weight for this tie.
    pub weight_mult: f64,
    /// `true` if this tie represents a high-priority high-frequency decoupling capacitor on a power rail.
    pub is_decoupling: bool,
}

/// Determines whether a KiCad net name represents a ground reference rail (`GND`, `AGND`, `DGND`, `PGND`, etc.).
///
/// # Arguments
/// * `name` - Net name string slice from a pad or netlist.
///
/// # Returns
/// `true` if the normalized uppercase net name matches standard ground naming conventions; `false` otherwise.
pub fn is_ground_net(name: &str) -> bool {
    let s = name.trim();
    if s.is_empty() {
        return false;
    }
    let u = s.to_ascii_uppercase();
    u == "GND"
        || u == "/GND"
        || u == "GROUND"
        || u.starts_with("GND_")
        || u.ends_with("_GND")
        || u.contains("AGND")
        || u.contains("DGND")
        || u.contains("PGND")
}

/// Determines whether a KiCad net name represents a DC power supply rail (`+3V3`, `+5V`, `VCC`, `VDD`, `VBUS`, `VIN`, etc.).
///
/// Explicitly excludes ground nets (checked via [`is_ground_net`]) and empty net names.
///
/// # Arguments
/// * `name` - Net name string slice from a pad or netlist.
///
/// # Returns
/// `true` if the net name matches a non-ground power supply rail convention; `false` otherwise.
pub fn is_power_net(name: &str) -> bool {
    let s = name.trim().trim_start_matches('/');
    if s.is_empty() || is_ground_net(s) {
        return false;
    }
    let u = s.to_ascii_uppercase();
    u.starts_with('+')
        || u.starts_with("VCC")
        || u.starts_with("VDD")
        || u.starts_with("VBUS")
        || u.starts_with("VRAW")
        || u.starts_with("VIN")
        || u.starts_with("VBAT")
        || u.starts_with("VOUT")
        || u.contains("3V3")
        || u.contains("5V")
        || u.contains("12V")
        || u.contains("BUCK_VIN")
        || u.contains("POWER")
}

/// Constructs the list of [`PassiveTie`] affinity constraints for all unlocked passive components on `board`.
///
/// # Algorithm
/// 1. **Decoupling & Bulk Capacitors:** Identifies 2-pin capacitors (`C*`) connected between a ground
///    net ([`is_ground_net`]) and a power net ([`is_power_net`]). Candidate active device power pins
///    (`U*`, `Q*`, `J*`, `M*`) on the same rail are scored by schematic sheet proximity (or physical/reference
///    number distance if schematic coordinates are unavailable), shared schematic cluster membership,
///    device tier (`U` > `Q` > others), and existing capacitor load per pin to distribute decoupling caps
///    evenly across multi-pin IC supplies. Small ceramic bypass caps receive `is_decoupling = true`
///    (`max_dist = 2.0` mm, `weight_mult = 1.8`), whereas bulk capacitors (`> 3.8` mm or `>= 47uF`)
///    receive `max_dist = 8.0` mm (`is_decoupling = false`).
/// 2. **Signal Passives:** Remaining unlocked passives (`R*`, `C*`, `L*`, `F*`, `D*`) on non-power,
///    non-ground signal nets are tied to the highest-priority peer component (`U` > `J`/`Q` > passive)
///    sharing that signal net (`max_dist = 3.5` mm, `weight_mult = 1.0`).
///
/// # Arguments
/// * `board` - Board containing footprints, pad net assignments, schematic positions, and clusters.
///
/// # Returns
/// Vector of [`PassiveTie`] records linking passive pads to their selected anchor pads.
pub fn build_passive_ties(board: &Board) -> Vec<PassiveTie> {
    // 1. Map of non-power, non-ground signal nets -> Vec<(fp_idx, pad_idx)>
    let mut net_pads: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (fp_idx, fp) in board.footprints.iter().enumerate() {
        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if pad.net_id == 0
                || pad.net_name.is_empty()
                || is_ground_net(&pad.net_name)
                || is_power_net(&pad.net_name)
            {
                continue;
            }
            net_pads.entry(pad.net_id).or_default().push((fp_idx, pad_idx));
        }
    }

    // 2. Map of power net_id -> Vec<(active_device_idx, pad_idx)>
    let mut power_net_device_pads: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (fp_idx, fp) in board.footprints.iter().enumerate() {
        let is_active = fp.reference.starts_with('U')
            || fp.reference.starts_with('Q')
            || fp.reference.starts_with('J')
            || fp.reference.starts_with('M');
        if !is_active {
            continue;
        }
        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if pad.net_id != 0 && is_power_net(&pad.net_name) {
                power_net_device_pads
                    .entry(pad.net_id)
                    .or_default()
                    .push((fp_idx, pad_idx));
            }
        }
    }

    let mut pin_usage: HashMap<(usize, usize), usize> = HashMap::new();
    let mut ties = Vec::new();

    for (fp_idx, fp) in board.footprints.iter().enumerate() {
        let is_passive = fp.reference.starts_with('R')
            || fp.reference.starts_with('C')
            || fp.reference.starts_with('L')
            || fp.reference.starts_with('F')
            || fp.reference.starts_with('D');
        if !is_passive || fp.is_locked {
            continue;
        }

        // Check if fp is a 2-pin decoupling/bulk capacitor: one pin on GND, one pin on a power net
        let is_2pin_cap = fp.reference.starts_with('C') && fp.pads.len() == 2;
        let is_bulk_cap = fp.width > 3.8
            || fp.height > 3.8
            || fp.value.to_lowercase().contains("low-esr")
            || fp.value.to_lowercase().contains("330uf")
            || fp.value.to_lowercase().contains("100uf")
            || fp.value.to_lowercase().contains("47uf");
        let mut gnd_pin = None;
        let mut pwr_pin = None;
        if is_2pin_cap {
            for (p_idx, pad) in fp.pads.iter().enumerate() {
                if is_ground_net(&pad.net_name) && gnd_pin.is_none() {
                    gnd_pin = Some(p_idx);
                } else if is_power_net(&pad.net_name) && pwr_pin.is_none() {
                    pwr_pin = Some(p_idx);
                }
            }
        }

        if let (Some(_gnd_idx), Some(pwr_pad_idx)) = (gnd_pin, pwr_pin) {
            let pwr_net_id = fp.pads[pwr_pad_idx].net_id;
            if let Some(candidate_pads) = power_net_device_pads.get(&pwr_net_id) {
                let mut best_candidate: Option<(usize, usize, f64)> = None;

                for &(dev_idx, dev_pad_idx) in candidate_pads {
                    if dev_idx == fp_idx {
                        continue;
                    }
                    let dev = &board.footprints[dev_idx];
                    let dev_ref = &dev.reference;
                    let type_tier = if dev_ref.starts_with('U') {
                        3.0
                    } else if dev_ref.starts_with('Q') {
                        2.0
                    } else {
                        1.0
                    };

                    let mut score = 0.0;
                    if let (Some(fp_sch), Some(dev_sch)) = (
                        board.schematic_positions.get(&fp.reference),
                        board.schematic_positions.get(dev_ref),
                    ) {
                        let d_sch = (fp_sch.0 - dev_sch.0).hypot(fp_sch.1 - dev_sch.1);
                        score += d_sch;
                    } else {
                        let d_phys = (fp.x - dev.x).hypot(fp.y - dev.y);
                        score += d_phys * 2.0;

                        let fp_num: f64 = fp
                            .reference
                            .trim_start_matches(|c: char| !c.is_ascii_digit())
                            .parse()
                            .unwrap_or(0.0);
                        let dev_num: f64 = dev_ref
                            .trim_start_matches(|c: char| !c.is_ascii_digit())
                            .parse()
                            .unwrap_or(0.0);
                        score += (fp_num - dev_num).abs() * 0.5;
                    }

                    for cl in &board.clusters {
                        if cl.member_indices.contains(&fp_idx) && cl.member_indices.contains(&dev_idx) {
                            score -= 200.0;
                            break;
                        }
                    }

                    score -= type_tier * 50.0;

                    let usage = *pin_usage.get(&(dev_idx, dev_pad_idx)).unwrap_or(&0);
                    score += (usage as f64) * 15.0;

                    if best_candidate.is_none_or(|b| score < b.2) {
                        best_candidate = Some((dev_idx, dev_pad_idx, score));
                    }
                }

                if let Some((best_dev, best_pad, _)) = best_candidate {
                    *pin_usage.entry((best_dev, best_pad)).or_insert(0) += 1;
                    let (max_dist, weight_mult, is_decoupling) = if is_bulk_cap {
                        (8.0, 1.0, false)
                    } else {
                        (2.0, 1.8, true)
                    };
                    ties.push(PassiveTie {
                        passive_idx: fp_idx,
                        passive_pad_idx: pwr_pad_idx,
                        anchor_idx: best_dev,
                        anchor_pad_idx: best_pad,
                        max_dist,
                        weight_mult,
                        is_decoupling,
                    });
                    continue;
                }
            }
        }

        // Non-decoupling passive (e.g. pullups, termination resistors, signal caps)
        let mut best: Option<(usize, usize, usize, usize)> = None;
        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if pad.net_name.is_empty()
                || is_ground_net(&pad.net_name)
                || is_power_net(&pad.net_name)
            {
                continue;
            }
            if let Some(peers) = net_pads.get(&pad.net_id) {
                for &(peer_fp_idx, peer_pad_idx) in peers {
                    if peer_fp_idx == fp_idx {
                        continue;
                    }
                    let peer_ref = &board.footprints[peer_fp_idx].reference;
                    let priority = if peer_ref.starts_with('U') {
                        3
                    } else if peer_ref.starts_with('J') || peer_ref.starts_with('Q') {
                        2
                    } else {
                        1
                    };
                    if best.is_none_or(|b| priority > b.3) {
                        best = Some((pad_idx, peer_fp_idx, peer_pad_idx, priority));
                    }
                }
            }
        }
        if let Some((p_pad, a_fp, a_pad, _)) = best {
            ties.push(PassiveTie {
                passive_idx: fp_idx,
                passive_pad_idx: p_pad,
                anchor_idx: a_fp,
                anchor_pad_idx: a_pad,
                max_dist: 3.5,
                weight_mult: 1.0,
                is_decoupling: false,
            });
        }
    }
    ties
}

/// Computes the minimum Euclidean boundary-to-boundary distance in millimeters (`mm`) between two convex polygons.
///
/// First tests for overlap via [`Polygon::intersects_convex`], returning `0.0` immediately if the
/// polygons intersect or either polygon is empty. Otherwise evaluates point-to-segment distances
/// between every vertex of `p1` and edge of `p2` (and vice versa).
///
/// # Arguments
/// * `p1` - First convex polygon in world coordinates (`mm`).
/// * `p2` - Second convex polygon in world coordinates (`mm`).
///
/// # Returns
/// Minimum separation distance in millimeters (`mm`), or `0.0` if intersecting or empty.
pub fn polygon_clearance_gap(p1: &Polygon, p2: &Polygon) -> f64 {
    if p1.vertices.is_empty() || p2.vertices.is_empty() {
        return 0.0;
    }
    if p1.intersects_convex(p2) {
        return 0.0;
    }
    let mut min_d = f64::INFINITY;
    let n1 = p1.vertices.len();
    let n2 = p2.vertices.len();
    for v in &p1.vertices {
        for j in 0..n2 {
            let d = v.distance_to_segment(&p2.vertices[j], &p2.vertices[(j + 1) % n2]);
            if d < min_d {
                min_d = d;
            }
        }
    }
    for v in &p2.vertices {
        for i in 0..n1 {
            let d = v.distance_to_segment(&p1.vertices[i], &p1.vertices[(i + 1) % n1]);
            if d < min_d {
                min_d = d;
            }
        }
    }
    if min_d.is_infinite() {
        0.0
    } else {
        min_d
    }
}

/// Computes the minimum physical clearance aperture $W$ in millimeters (`mm`) between two sets of courtyard polygons.
///
/// Evaluates [`polygon_clearance_gap`] across all pairs `(pa, pb)` in `polys_a` $\times$ `polys_b`
/// and short-circuits to `0.0` as soon as any pair intersects (`gap <= 1e-6`).
///
/// # Arguments
/// * `polys_a` - Courtyard polygon slice for the first footprint.
/// * `polys_b` - Courtyard polygon slice for the second footprint.
///
/// # Returns
/// Minimum unobstructed gap $W$ in millimeters (`mm`), or `0.0` if intersecting or either slice is empty.
pub fn courtyards_clearance_gap(polys_a: &[Polygon], polys_b: &[Polygon]) -> f64 {
    if polys_a.is_empty() || polys_b.is_empty() {
        return 0.0;
    }
    let mut min_gap = f64::INFINITY;
    for pa in polys_a {
        for pb in polys_b {
            let d = polygon_clearance_gap(pa, pb);
            if d < min_gap {
                min_gap = d;
                if min_gap <= 1e-6 {
                    return 0.0;
                }
            }
        }
    }
    if min_gap.is_infinite() {
        0.0
    } else {
        min_gap
    }
}

/// Computes the multi-layer routing corridor track capacity across a physical gap of width $W$:
///
/// $$\text{Capacity} = N_{\text{layers}} \times \max\left(0, \left\lfloor \frac{W - C_{\min}}{W_{\text{track}} + C_{\min}} \right\rfloor\right)$$
///
/// # Arguments
/// * `gap_w` - Physical clearance aperture $W$ between two component courtyards in millimeters (`mm`).
/// * `track_width` - Nominal copper trace width $W_{\text{track}}$ in millimeters (`mm`).
/// * `min_clearance` - Minimum copper-to-copper clearance $C_{\min}$ in millimeters (`mm`).
/// * `num_layers` - Number of routable copper layers $N_{\text{layers}}$.
///
/// # Returns
/// Total number of parallel copper traces that can legally pass through the corridor across all layers.
pub fn compute_corridor_capacity(
    gap_w: f64,
    track_width: f64,
    min_clearance: f64,
    num_layers: usize,
) -> usize {
    if gap_w < min_clearance || (track_width + min_clearance) <= 1e-6 {
        return 0;
    }
    let pitch = track_width + min_clearance;
    let tracks_per_layer = ((gap_w - min_clearance) / pitch).floor().max(0.0) as usize;
    num_layers.saturating_mul(tracks_per_layer)
}

/// Represents a multi-pin bus connection ($M \ge 2$ shared signal/power nets) between two footprints.
///
/// Used by [`compute_flow_alignment_penalty_for_link`] to evaluate pin ribbon untangling and
/// relative orientation alignment between coupled ICs, connectors, and resistor arrays.
#[derive(Debug, Clone)]
pub struct ComponentBusLink {
    /// Footprint index of the first component (`comp_a < comp_b`).
    pub comp_a: usize,
    /// Footprint index of the second component (`comp_a < comp_b`).
    pub comp_b: usize,
    /// Pairs of `(pad_idx_on_a, pad_idx_on_b)` sharing the same non-ground net between `comp_a` and `comp_b`.
    pub shared_pads: Vec<(usize, usize)>,
}

/// Discovers shared non-ground nets and multi-pin [`ComponentBusLink`]s across all footprints on `board`.
///
/// Ignores unconnected pads (`net_id == 0`) and ground rails ([`is_ground_net`]) so ground plane
/// connections do not create artificial point-to-point bus links.
///
/// # Arguments
/// * `board` - Board whose footprints and pad net assignments are analyzed.
///
/// # Returns
/// A tuple `(pair_shared_nets, bus_links)` where:
/// * `pair_shared_nets` maps ordered component index pairs `(comp_a, comp_b)` to their deduplicated list of shared net IDs.
/// * `bus_links` contains a [`ComponentBusLink`] for every component pair sharing at least 2 pad connections.
pub fn build_board_topological_connectivity(
    board: &Board,
) -> (HashMap<(usize, usize), Vec<usize>>, Vec<ComponentBusLink>) {
    let mut net_pads: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (fp_idx, fp) in board.footprints.iter().enumerate() {
        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if pad.net_id == 0 || pad.net_name.is_empty() || is_ground_net(&pad.net_name) {
                continue;
            }
            net_pads.entry(pad.net_id).or_default().push((fp_idx, pad_idx));
        }
    }

    let mut pair_shared_nets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut pair_shared_pads: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();

    for (&net_id, pads) in &net_pads {
        for i in 0..pads.len() {
            for j in (i + 1)..pads.len() {
                let (fp_a, pad_a) = pads[i];
                let (fp_b, pad_b) = pads[j];
                if fp_a != fp_b {
                    let key = if fp_a < fp_b { (fp_a, fp_b) } else { (fp_b, fp_a) };
                    let pad_pair = if fp_a < fp_b { (pad_a, pad_b) } else { (pad_b, pad_a) };
                    let nets = pair_shared_nets.entry(key).or_default();
                    if !nets.contains(&net_id) {
                        nets.push(net_id);
                    }
                    pair_shared_pads.entry(key).or_default().push(pad_pair);
                }
            }
        }
    }

    let mut bus_links = Vec::new();
    for ((comp_a, comp_b), shared_pads) in pair_shared_pads {
        if shared_pads.len() >= 2 {
            bus_links.push(ComponentBusLink {
                comp_a,
                comp_b,
                shared_pads,
            });
        }
    }

    (pair_shared_nets, bus_links)
}

/// Evaluates topological signal flow alignment and pin ribbon untangling between two connected components.
///
/// Projects pad coordinates onto the normal axis $\mathbf{v} = (-u_y, u_x)$ perpendicular to the
/// component-to-component axis $\mathbf{u} = \frac{\text{pos}_b - \text{pos}_a}{\|\text{pos}_b - \text{pos}_a\|}$
/// and accumulates penalties across all pad pairs $(j, k)$ for:
/// 1. **Lateral Order Inversion (`+1.0`):** When $(s_{a,j} - s_{a,k}) \cdot (s_{b,j} - s_{b,k}) < 0$, indicating a twisted bus ribbon.
/// 2. **Direct Segment Crossing (`+1.5`):** When straight airwires $\overline{a_j b_j}$ and $\overline{a_k b_k}$ intersect.
/// 3. **Angular Skew (`+0.5 * (1 - \cos\theta)`):** When airwire vectors diverge from parallel alignment.
///
/// # Arguments
/// * `pads_a` - World-space coordinates (`mm`) of the bus pads on component A.
/// * `pads_b` - Corresponding world-space coordinates (`mm`) of the connected bus pads on component B.
/// * `pos_a` - World-space origin (`mm`) of component A.
/// * `pos_b` - World-space origin (`mm`) of component B.
///
/// # Returns
/// Dimensionless non-negative flow alignment penalty (`0.0` if fewer than 2 pad pairs exist).
pub fn compute_flow_alignment_penalty_for_link(
    pads_a: &[Point],
    pads_b: &[Point],
    pos_a: Point,
    pos_b: Point,
) -> f64 {
    let m = pads_a.len().min(pads_b.len());
    if m < 2 {
        return 0.0;
    }
    let dx = pos_b.x - pos_a.x;
    let dy = pos_b.y - pos_a.y;
    let dist = dx.hypot(dy);
    let (ux, uy) = if dist > 1e-4 {
        (dx / dist, dy / dist)
    } else {
        (1.0, 0.0)
    };
    let (vx, vy) = (-uy, ux);

    let mut penalty = 0.0;
    for j in 0..m {
        for k in (j + 1)..m {
            let s_aj = (pads_a[j].x - pos_a.x) * vx + (pads_a[j].y - pos_a.y) * vy;
            let s_ak = (pads_a[k].x - pos_a.x) * vx + (pads_a[k].y - pos_a.y) * vy;
            let s_bj = (pads_b[j].x - pos_b.x) * vx + (pads_b[j].y - pos_b.y) * vy;
            let s_bk = (pads_b[k].x - pos_b.x) * vx + (pads_b[k].y - pos_b.y) * vy;

            let d_sa = s_aj - s_ak;
            let d_sb = s_bj - s_bk;

            // 1. Lateral order inversion (topological ribbon twisting)
            if d_sa.abs() > 1e-4 && d_sb.abs() > 1e-4 && d_sa * d_sb < 0.0 {
                penalty += 1.0;
            }

            // 2. Direct segment crossing
            let seg_j = Segment { p1: pads_a[j], p2: pads_b[j], net_id: 1 };
            let seg_k = Segment { p1: pads_a[k], p2: pads_b[k], net_id: 2 };
            if seg_j.crosses(&seg_k) {
                penalty += 1.5;
            }

            // 3. Angular divergence / parallel skew
            let wjx = pads_b[j].x - pads_a[j].x;
            let wjy = pads_b[j].y - pads_a[j].y;
            let wkx = pads_b[k].x - pads_a[k].x;
            let wky = pads_b[k].y - pads_a[k].y;
            let len_j = wjx.hypot(wjy);
            let len_k = wkx.hypot(wky);
            if len_j > 1e-4 && len_k > 1e-4 {
                let dot = (wjx * wkx + wjy * wky) / (len_j * len_k);
                let angle_skew = (1.0 - dot).clamp(0.0, 2.0);
                penalty += 0.5 * angle_skew;
            }
        }
    }
    penalty
}

/// Computes a fast 64-bit hash over all footprint pad `net_id` assignments on `board`.
///
/// Used by [`PlacementState::evaluate_cost`] to detect when pin or gate swapping has mutated
/// pad net assignments and lazily rebuild `PlacementState::cached_connectivity`.
///
/// # Arguments
/// * `board` - Board whose footprint pad net IDs are hashed.
///
/// # Returns
/// 64-bit hash signature of the board's pad connectivity.
fn board_connectivity_hash(board: &Board) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    board.footprints.len().hash(&mut h);
    for fp in &board.footprints {
        fp.pads.len().hash(&mut h);
        for pad in &fp.pads {
            pad.net_id.hash(&mut h);
        }
    }
    h.finish()
}

impl PlacementState {
    /// Constructs a [`PlacementState`] from the current footprint poses on `board` and evaluates its initial cost.
    ///
    /// Pre-transforms all footprint courtyards (inflated by [`crate::board::Footprint::effective_spacing`]),
    /// computes union AABBs and world-space pad coordinates, builds the topological connectivity cache
    /// via [`build_board_topological_connectivity`], and runs a full [`PlacementState::evaluate`] pass.
    ///
    /// # Arguments
    /// * `board` - Board whose footprint coordinates, rotations, layers, and nets are snapshotted.
    /// * `config` - Placement hyperparameters and weights used to evaluate initial spacing and cost.
    ///
    /// # Returns
    /// Fully initialized and evaluated [`PlacementState`].
    pub fn from_board(board: &Board, config: &PlacementConfig) -> Self {
        let n = board.footprints.len();
        let mut positions = Vec::with_capacity(n);
        let mut rotations = Vec::with_capacity(n);
        let mut layers = Vec::with_capacity(n);
        let mut cached_courtyards = Vec::with_capacity(n);
        let mut cached_bounds = Vec::with_capacity(n);
        let mut cached_pads = Vec::with_capacity(n);

        for fp in &board.footprints {
            let pos = Point::new(fp.x, fp.y);
            let rot = fp.rotation;
            let layer = fp.layer;
            positions.push(pos);
            rotations.push(rot);
            layers.push(layer);
            let spacing = fp.effective_spacing(config);
            let polys = fp.transformed_courtyards_layered(pos, rot, layer, spacing);
            cached_bounds.push(union_aabb(&polys, pos));
            cached_courtyards.push(polys);
            let pads = (0..fp.pads.len())
                .map(|pi| fp.pad_world_point_layered(pi, pos, rot, layer))
                .collect();
            cached_pads.push(pads);
        }

        let cached_connectivity_hash = board_connectivity_hash(board);
        let cached_connectivity = Arc::new(build_board_topological_connectivity(board));

        let mut st = Self {
            positions,
            rotations,
            layers,
            cached_courtyards,
            cached_bounds,
            cached_pads,
            cached_connectivity_hash,
            cached_connectivity,
            cost: 0.0,
            crossings: 0,
            corridor_congestion: 0.0,
            flow_alignment_penalty: 0.0,
            wirelength: 0.0,
            overlaps: 0,
            boundary_violations: 0,
        };
        let ties = build_passive_ties(board);
        st.evaluate(board, config, &ties);
        st
    }

    /// Incrementally recomputes the cached world-space courtyards, bounding box, and pad coordinates
    /// for a single modified footprint `idx`.
    ///
    /// # Arguments
    /// * `idx` - Footprint index in `board.footprints` whose pose (`positions[idx]`, `rotations[idx]`, `layers[idx]`) changed.
    /// * `board` - Reference to the [`Board`] containing the footprint geometry definitions.
    /// * `config` - Placement configuration specifying courtyard spacing margins.
    ///
    /// # Side Effects
    /// Mutates `self.cached_bounds[idx]`, `self.cached_courtyards[idx]`, and `self.cached_pads[idx]`.
    fn update_component(&mut self, idx: usize, board: &Board, config: &PlacementConfig) {
        let fp = &board.footprints[idx];
        let pos = self.positions[idx];
        let rot = self.rotations[idx];
        let layer = self.layers[idx];
        let spacing = fp.effective_spacing(config);
        let polys = fp.transformed_courtyards_layered(pos, rot, layer, spacing);
        self.cached_bounds[idx] = union_aabb(&polys, pos);
        self.cached_courtyards[idx] = polys;
        for pi in 0..fp.pads.len() {
            self.cached_pads[idx][pi] = fp.pad_world_point_layered(pi, pos, rot, layer);
        }
    }

    /// Evaluates all objective metrics and updates `self.cost` for the current placement state.
    ///
    /// Delegates directly to [`PlacementState::evaluate_cost`].
    ///
    /// # Arguments
    /// * `board` - Board geometry, netlist, clusters, and keepout definitions.
    /// * `config` - Placement weights and clearance parameters.
    /// * `ties` - Precomputed [`PassiveTie`] affinity constraints.
    ///
    /// # Side Effects
    /// Updates `self.wirelength`, `self.crossings`, `self.corridor_congestion`,
    /// `self.flow_alignment_penalty`, `self.overlaps`, `self.boundary_violations`, and `self.cost`.
    pub fn evaluate(&mut self, board: &Board, config: &PlacementConfig, ties: &[PassiveTie]) {
        self.evaluate_cost(board, config, ties);
    }

    /// Computes the composite multi-objective placement energy function and updates all diagnostic counters on `self`.
    ///
    /// # Cost Formulation
    /// 1. **Ratsnest Wirelength & Crossings:** Generates Prim's Euclidean MST ratsnest segments over
    ///    `self.cached_pads` and constructs a 2D Morton LBVH ([`build_gpu_lbvh`]) when $E \ge 16$ so
    ///    pairwise airwire crossings are counted in $O(E \log E)$ time.
    /// 2. **Courtyard Collisions:** Tests all footprint pairs with overlapping `cached_bounds` using
    ///    exact convex SAT (`Polygon::intersects_convex`), skipping locked-locked pairs and opposite-side
    ///    SMD-only pairs. Adds `config.overlap_penalty * (1.0 + ox * oy)` per colliding pair.
    /// 3. **Boundary & Keepout Violations:** Penalizes any unlocked footprint whose courtyard vertices
    ///    or pad centers fall outside the board outline or inside a layer-matching keepout zone, plus a
    ///    small penalty (`3.5`) for flipping a component away from its original layer.
    /// 4. **Schematic Cluster Cohesion & Passive Affinity:** Applies linear inside-radius and
    ///    quadratic/linear outside-radius penalties for schematic clusters (`board.clusters`) and
    ///    [`PassiveTie`] tethers.
    /// 5. **Corridor Cut Congestion & Bus Flow Alignment:** For nearby component pairs within `15.0` mm,
    ///    compares physical routing capacity ([`compute_corridor_capacity`]) against shared nets plus
    ///    ratsnest segments cut by the centroid-to-centroid ray, adding squared overflow penalties, and
    ///    evaluates [`compute_flow_alignment_penalty_for_link`] over all [`ComponentBusLink`]s.
    ///
    /// # Arguments
    /// * `board` - Board containing footprints, nets, netclasses, clusters, boundary, and keepouts.
    /// * `config` - Placement energy weights and physical clearance settings.
    /// * `ties` - Slice of [`PassiveTie`] constraints binding passives to anchor pins.
    ///
    /// # Side Effects
    /// Mutates `self.wirelength`, `self.crossings`, `self.corridor_congestion`,
    /// `self.flow_alignment_penalty`, `self.overlaps`, `self.boundary_violations`, `self.cost`,
    /// and refreshes `self.cached_connectivity` if `board_connectivity_hash(board)` changed.
    pub fn evaluate_cost(&mut self, board: &Board, config: &PlacementConfig, ties: &[PassiveTie]) {
        let segments = board.generate_ratsnest_segments(&self.cached_pads);
        let wirelength: f64 = segments.iter().map(|s| s.length()).sum();

        // Build a 2D Morton LBVH over the ratsnest segments when E >= 16 so both
        // ratsnest crossing count and topological corridor cut ray-casting run in O(log E).
        let seg_bvh: Vec<GpuBvhNode> = if segments.len() >= 16 {
            let mut obs = Vec::with_capacity(segments.len());
            for seg in &segments {
                let min_x = seg.p1.x.min(seg.p2.x) as f32;
                let min_y = seg.p1.y.min(seg.p2.y) as f32;
                let max_x = seg.p1.x.max(seg.p2.x) as f32;
                let max_y = seg.p1.y.max(seg.p2.y) as f32;
                obs.push(([min_x, min_y, max_x, max_y], 0xFFFF_FFFF, seg.net_id as u32));
            }
            build_gpu_lbvh(&obs)
        } else {
            Vec::new()
        };

        let mut crossings = 0usize;
        if !seg_bvh.is_empty() {
            for (i, seg_i) in segments.iter().enumerate() {
                let ray = [
                    seg_i.p1.x as f32,
                    seg_i.p1.y as f32,
                    seg_i.p2.x as f32,
                    seg_i.p2.y as f32,
                ];
                query_lbvh_crossing_segments(ray, seg_i.net_id as u32, &seg_bvh, |j| {
                    if j > i && seg_i.crosses(&segments[j]) {
                        crossings += 1;
                    }
                });
            }
        } else {
            for i in 0..segments.len() {
                for j in (i + 1)..segments.len() {
                    if segments[i].crosses(&segments[j]) {
                        crossings += 1;
                    }
                }
            }
        }

        let n = board.footprints.len();
        let mut overlaps = 0usize;
        let mut overlap_area_cost = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if board.footprints[i].is_locked && board.footprints[j].is_locked {
                    continue;
                }
                // Two SMD-only footprints on opposite board sides (F.Cu vs B.Cu) do not collide
                if self.layers[i] != self.layers[j]
                    && board.footprints[i].is_smd_only()
                    && board.footprints[j].is_smd_only()
                {
                    continue;
                }
                if !self.cached_bounds[i].overlaps(&self.cached_bounds[j]) {
                    continue;
                }
                let mut hit = false;
                'outer: for pa in &self.cached_courtyards[i] {
                    for pb in &self.cached_courtyards[j] {
                        if pa.intersects_convex(pb) {
                            hit = true;
                            break 'outer;
                        }
                    }
                }
                if hit {
                    overlaps += 1;
                    let ox = (self.cached_bounds[i].max_x.min(self.cached_bounds[j].max_x)
                        - self.cached_bounds[i].min_x.max(self.cached_bounds[j].min_x))
                    .max(0.1);
                    let oy = (self.cached_bounds[i].max_y.min(self.cached_bounds[j].max_y)
                        - self.cached_bounds[i].min_y.max(self.cached_bounds[j].min_y))
                    .max(0.1);
                    overlap_area_cost += config.overlap_penalty * (1.0 + ox * oy);
                }
            }
        }

        let boundary = board.boundary_aabb();
        let mut boundary_cost = 0.0;
        let mut boundary_violations = 0usize;
        let mut flip_cost = 0.0;
        for i in 0..n {
            if board.footprints[i].is_locked {
                continue;
            }
            if self.layers[i] != board.footprints[i].orig_layer {
                flip_cost += 3.5;
            }
            let mut fp_violated = false;
            for poly in &self.cached_courtyards[i] {
                for v in &poly.vertices {
                    if !board.contains_point_in_boundary(v) {
                        let dx = (boundary.min_x - v.x).max(0.0) + (v.x - boundary.max_x).max(0.0);
                        let dy = (boundary.min_y - v.y).max(0.0) + (v.y - boundary.max_y).max(0.0);
                        boundary_cost += config.boundary_penalty * (10.0 + dx * 10.0 + dy * 10.0);
                        fp_violated = true;
                    }
                }
            }
            for p in &self.cached_pads[i] {
                if !board.contains_point_in_boundary(p) {
                    let dx = (boundary.min_x - p.x).max(0.0) + (p.x - boundary.max_x).max(0.0);
                    let dy = (boundary.min_y - p.y).max(0.0) + (p.y - boundary.max_y).max(0.0);
                    boundary_cost += config.boundary_penalty * (10.0 + dx * 10.0 + dy * 10.0);
                    fp_violated = true;
                }
            }
            if !fp_violated {
                let fp_layer = self.layers[i];
                for ko in &board.keepouts {
                    if !ko.layers.contains(&fp_layer) {
                        continue;
                    }
                    let ko_poly = Polygon::from_rect(ko.min_x, ko.min_y, ko.max_x - ko.min_x, ko.max_y - ko.min_y);
                    for poly in &self.cached_courtyards[i] {
                        if poly.intersects_convex(&ko_poly) {
                            boundary_cost += config.boundary_penalty * 20.0;
                            fp_violated = true;
                            break;
                        }
                    }
                    if !fp_violated {
                        for p in &self.cached_pads[i] {
                            if ko_poly.contains_point(p) {
                                boundary_cost += config.boundary_penalty * 20.0;
                                fp_violated = true;
                                break;
                            }
                        }
                    }
                    if fp_violated {
                        break;
                    }
                }
            }
            if fp_violated {
                boundary_violations += 1;
            }
        }

        let mut cluster_cost = 0.0;
        for cl in &board.clusters {
            let center = if let Some(a_idx) = cl.anchor_idx {
                self.positions[a_idx]
            } else {
                let k = cl.member_indices.len() as f64;
                Point::new(
                    cl.member_indices.iter().map(|&idx| self.positions[idx].x).sum::<f64>() / k,
                    cl.member_indices.iter().map(|&idx| self.positions[idx].y).sum::<f64>() / k,
                )
            };
            let w = (cl.weight * 0.25).min(25.0);
            for &m_idx in &cl.member_indices {
                if Some(m_idx) == cl.anchor_idx || board.footprints[m_idx].is_locked {
                    continue;
                }
                let d = self.positions[m_idx].distance_to(&center);
                if d > cl.max_radius {
                    let ex = d - cl.max_radius;
                    let penalty = if ex <= 2.0 {
                        0.5 * ex * ex
                    } else {
                        2.0 + 2.0 * (ex - 2.0)
                    };
                    cluster_cost += w * (0.1 * cl.max_radius + penalty);
                } else {
                    cluster_cost += (w * 0.1) * d;
                }
            }
        }

        let mut affinity_cost = 0.0;
        for tie in ties {
            let p_pt = self.cached_pads[tie.passive_idx][tie.passive_pad_idx];
            let a_pt = self.cached_pads[tie.anchor_idx][tie.anchor_pad_idx];
            let anchor_escape = if config.enable_escape_clearance {
                board.footprints[tie.anchor_idx].escape_margin * config.escape_margin_mult
            } else {
                0.0
            };
            let d = p_pt.distance_to(&a_pt);
            let (max_d, weight) = if tie.is_decoupling {
                (
                    config.decoupling_max_dist + anchor_escape,
                    config.decoupling_affinity_weight * tie.weight_mult.min(2.0),
                )
            } else {
                (
                    config.passive_max_dist + anchor_escape * 0.5,
                    config.passive_affinity_weight * tie.weight_mult.min(1.5),
                )
            };
            if d > max_d {
                let ex = d - max_d;
                let penalty = if ex <= 2.0 {
                    0.5 * ex * ex
                } else {
                    2.0 + 2.0 * (ex - 2.0)
                };
                affinity_cost += weight * (0.25 * max_d + penalty);
            } else {
                affinity_cost += weight * 0.25 * d;
            }
        }

        // Topological Corridor Capacity & Cut Congestion Model
        let mut corridor_overflow_total = 0.0;
        let mut flow_alignment_penalty_total = 0.0;

        if config.corridor_congestion_weight > 0.0 || config.flow_alignment_weight > 0.0 {
            let conn_hash = board_connectivity_hash(board);
            if self.cached_connectivity_hash != conn_hash {
                self.cached_connectivity_hash = conn_hash;
                self.cached_connectivity = Arc::new(build_board_topological_connectivity(board));
            }
            let (shared_nets, bus_links) = (&self.cached_connectivity.0, &self.cached_connectivity.1);

            if config.corridor_congestion_weight > 0.0 && n >= 2 {
                let num_layers = board.copper_layers().len().max(2);
                let default_nc = board.netclasses.get("Default");
                let track_width = default_nc.map(|nc| nc.track_width).unwrap_or(0.20).max(0.05);
                let min_clearance = default_nc.map(|nc| nc.clearance).unwrap_or(0.20).max(0.05);
                let mut demand_nets: Vec<usize> = Vec::with_capacity(16);

                for i in 0..n {
                    for j in (i + 1)..n {
                        let bi = &self.cached_bounds[i];
                        let bj = &self.cached_bounds[j];
                        let dx = (bi.min_x - bj.max_x).max(bj.min_x - bi.max_x).max(0.0);
                        let dy = (bi.min_y - bj.max_y).max(bj.min_y - bi.max_y).max(0.0);
                        if dx > 15.0 || dy > 15.0 || (dx * dx + dy * dy) > 225.0 {
                            continue;
                        }
                        if self.layers[i] != self.layers[j]
                            && board.footprints[i].is_smd_only()
                            && board.footprints[j].is_smd_only()
                        {
                            continue;
                        }

                        let gap_w = if !self.cached_courtyards[i].is_empty() && !self.cached_courtyards[j].is_empty() {
                            courtyards_clearance_gap(&self.cached_courtyards[i], &self.cached_courtyards[j])
                        } else {
                            dx.hypot(dy)
                        };

                        let capacity = compute_corridor_capacity(gap_w, track_width, min_clearance, num_layers);

                        demand_nets.clear();
                        if let Some(shared) = shared_nets.get(&(i, j)) {
                            demand_nets.extend_from_slice(shared);
                        }

                        let c_i = self.positions[i];
                        let c_j = self.positions[j];
                        if c_i.distance_to(&c_j) > 1e-4 {
                            let cut_seg = Segment { p1: c_i, p2: c_j, net_id: 0 };
                            if !seg_bvh.is_empty() {
                                let cut_ray = [
                                    c_i.x as f32,
                                    c_i.y as f32,
                                    c_j.x as f32,
                                    c_j.y as f32,
                                ];
                                query_lbvh_crossing_segments(cut_ray, 0, &seg_bvh, |seg_idx| {
                                    if cut_seg.crosses(&segments[seg_idx]) {
                                        demand_nets.push(segments[seg_idx].net_id);
                                    }
                                });
                            } else {
                                for seg in &segments {
                                    if cut_seg.crosses(seg) {
                                        demand_nets.push(seg.net_id);
                                    }
                                }
                            }
                        }

                        demand_nets.sort_unstable();
                        demand_nets.dedup();
                        let demand = demand_nets.len();

                        if demand > capacity {
                            let overflow = (demand - capacity) as f64;
                            corridor_overflow_total += overflow * overflow;
                        }
                    }
                }
            }

            if config.flow_alignment_weight > 0.0 {
                let mut pads_a = Vec::with_capacity(8);
                let mut pads_b = Vec::with_capacity(8);
                for link in bus_links {
                    let comp_a = link.comp_a;
                    let comp_b = link.comp_b;
                    let pos_a = self.positions[comp_a];
                    let pos_b = self.positions[comp_b];

                    pads_a.clear();
                    pads_b.clear();
                    for &(pa_idx, pb_idx) in &link.shared_pads {
                        pads_a.push(self.cached_pads[comp_a][pa_idx]);
                        pads_b.push(self.cached_pads[comp_b][pb_idx]);
                    }

                    let link_penalty = compute_flow_alignment_penalty_for_link(
                        &pads_a,
                        &pads_b,
                        pos_a,
                        pos_b,
                    );
                    flow_alignment_penalty_total += link_penalty;
                }
            }
        }

        self.wirelength = wirelength;
        self.crossings = crossings;
        self.corridor_congestion = corridor_overflow_total;
        self.flow_alignment_penalty = flow_alignment_penalty_total;
        self.overlaps = overlaps;
        self.boundary_violations = boundary_violations;
        self.cost = wirelength * config.wirelength_weight
            + (crossings as f64) * config.crossing_weight
            + corridor_overflow_total * config.corridor_congestion_weight
            + flow_alignment_penalty_total * config.flow_alignment_weight
            + overlap_area_cost
            + boundary_cost
            + cluster_cost
            + affinity_cost
            + flip_cost;
    }

    /// Writes the unlocked footprint poses (`x`, `y`, `rotation`, `layer`) from `self` back onto `board`.
    ///
    /// Locked footprints (`fp.is_locked == true`) are never modified.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] whose unlocked footprints are updated in-place.
    ///
    /// # Side Effects
    /// Mutates `fp.x`, `fp.y`, `fp.rotation`, and `fp.layer` for every unlocked footprint in `board.footprints`.
    pub fn apply_to_board(&self, board: &mut Board) {
        for (i, fp) in board.footprints.iter_mut().enumerate() {
            if !fp.is_locked {
                fp.x = self.positions[i].x;
                fp.y = self.positions[i].y;
                fp.rotation = self.rotations[i];
                fp.layer = self.layers[i];
            }
        }
    }
}

/// Computes the union axis-aligned bounding box ([`AABB`]) enclosing all polygons in `polys`,
/// or a $\pm 0.5\text{ mm}$ fallback box around `fallback` if `polys` is empty.
///
/// # Arguments
/// * `polys` - Slice of convex courtyard [`Polygon`]s in world coordinates (`mm`).
/// * `fallback` - Fallback center [`Point`] (`mm`) used when `polys` has no vertices.
///
/// # Returns
/// Enclosing [`AABB`] in millimeters (`mm`).
fn union_aabb(polys: &[Polygon], fallback: Point) -> AABB {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in polys {
        let b = p.aabb();
        min_x = min_x.min(b.min_x);
        min_y = min_y.min(b.min_y);
        max_x = max_x.max(b.max_x);
        max_y = max_y.max(b.max_y);
    }
    if min_x > max_x {
        AABB::new(fallback.x - 0.5, fallback.y - 0.5, fallback.x + 0.5, fallback.y + 0.5)
    } else {
        AABB::new(min_x, min_y, max_x, max_y)
    }
}

/// Quantizes a scalar coordinate `v` in millimeters (`mm`) to the nearest multiple of `step`.
///
/// # Arguments
/// * `v` - Coordinate value in millimeters (`mm`).
/// * `step` - Grid step size in millimeters (`mm`). If `step <= 0.0`, `v` is returned unchanged.
///
/// # Returns
/// Grid-snapped coordinate in millimeters (`mm`).
fn snap_to_grid(v: f64, step: f64) -> f64 {
    if step <= 0.0 {
        v
    } else {
        (v / step).round() * step
    }
}

/// Packs footprint bounding descriptors, tagged connectivity edges, and flattened courtyard polygons
/// into GPU-ready storage buffers for the Vulkan WGSL Simulated Annealing compute shader.
///
/// # Buffer Encoding
/// 1. **Footprint Descriptors (`fp_descs: Vec<[f32; 4]>`):** Each `[hw, hh, locked, packed_center]`
///    entry stores the half-width `hw` (`mm`), half-height `hh` (`mm`), lock flag (`1.0` if locked,
///    `0.0` if movable), and quantized local courtyard center offset (`packed_center`).
/// 2. **Courtyard Buffers ([`GpuCourtyardBuffers`]):** Flattens each footprint's local-space inflated
///    courtyard polygons into `courtyard_vertices`, `courtyard_polys`, and `fp_poly_ranges` (`[poly_start, count, flags, orig_layer]`),
///    setting bit flags for SMD-only (`1`), non-centered/multi-polygon SAT required (`2`), and axis-aligned rectangle fast-path (`4`).
/// 3. **Tagged Connectivity Edges (`edges: Vec<[f32; 4]>`):** Pairs of `[fp_idx, pad_offset_x, pad_offset_y, tag]`
///    ordered strictly so inner GPU crossing loops can break early:
///    - `tag = 0.0`: Prim's MST ratsnest airwire segments.
///    - `tag = 3.0`: Inter-component corridor cut segments.
///    - `tag = 1.0` / `2.0`: Non-decoupling (`1.0`) and decoupling (`2.0`) [`PassiveTie`] tethers.
///    - `tag = 4.0`: Bus flow alignment and cluster anchor tethers.
///
/// # Arguments
/// * `board` - Board containing footprints, pads, nets, and clusters.
/// * `config` - Placement configuration specifying spacing and active penalty weights.
/// * `ties` - Slice of [`PassiveTie`] constraints.
/// * `base` - Reference [`PlacementState`] providing initial world-space pad coordinates for MST construction.
///
/// # Returns
/// Tuple `(fp_descs, edges, courtyards)` ready for upload to [`VulkanEngine::run_gpu_sa_placer_chains`].
pub fn build_gpu_sa_descriptors(
    board: &Board,
    config: &PlacementConfig,
    ties: &[PassiveTie],
    base: &PlacementState,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>, GpuCourtyardBuffers) {
    let mut fp_descs = Vec::with_capacity(board.footprints.len());
    let mut fp_poly_ranges = Vec::with_capacity(board.footprints.len());
    let mut courtyard_polys = Vec::new();
    let mut courtyard_vertices = Vec::new();

    for (i, fp) in board.footprints.iter().enumerate() {
        let spacing = fp.effective_spacing(config);
        let polys = fp.transformed_courtyards_layered(Point::new(0.0, 0.0), 0.0, fp.orig_layer, spacing);
        let b = union_aabb(&polys, Point::new(0.0, 0.0));
        let raw_hw = ((b.max_x - b.min_x) * 0.5) as f32;
        let raw_hh = ((b.max_y - b.min_y) * 0.5) as f32;
        let cx = ((b.min_x + b.max_x) * 0.5) as f32;
        let cy_orig = ((b.min_y + b.max_y) * 0.5) as f32;
        let cy_eff = if fp.layer != fp.orig_layer { -cy_orig } else { cy_orig };

        let (packed_center, pad_x, pad_y, center_exact) = if cx.abs() > 1e-2 || cy_eff.abs() > 1e-2 {
            let qx = (cx * 20.0).round().clamp(-500.0, 500.0);
            let qy = (cy_eff * 20.0).round().clamp(-500.0, 500.0);
            let rx = (cx - qx * 0.05).abs();
            let ry = (cy_eff - qy * 0.05).abs();
            (
                (qx + 512.0) * 1024.0 + (qy + 512.0),
                rx,
                ry,
                rx <= 1e-2 && ry <= 1e-2,
            )
        } else {
            (0.0f32, cx.abs(), cy_eff.abs(), true)
        };

        let hw = (raw_hw + pad_x).max(0.4);
        let hh = (raw_hh + pad_y).max(0.4);
        let locked = if fp.is_locked { 1.0 } else { 0.0 };
        fp_descs.push([hw, hh, locked, packed_center]);

        let is_smd = if fp.is_smd_only() { 1u32 } else { 0u32 };
        let orig_layer = fp.orig_layer as u32;
        let poly_start = courtyard_polys.len() as u32;

        let mut valid_poly_count = 0u32;
        let mut all_polys_rect = true;
        for poly in polys {
            if poly.vertices.len() >= 3 {
                let v_start = courtyard_vertices.len() as u32;
                let pb = poly.aabb();
                let is_poly_rect = poly.vertices.len() == 4
                    && poly.vertices.iter().all(|v| {
                        ((v.x - pb.min_x).abs() <= 1e-3 || (v.x - pb.max_x).abs() <= 1e-3)
                            && ((v.y - pb.min_y).abs() <= 1e-3 || (v.y - pb.max_y).abs() <= 1e-3)
                    });
                if is_poly_rect {
                    courtyard_vertices.push([pb.min_x as f32, pb.min_y as f32]);
                    courtyard_vertices.push([pb.max_x as f32, pb.min_y as f32]);
                    courtyard_vertices.push([pb.max_x as f32, pb.max_y as f32]);
                    courtyard_vertices.push([pb.min_x as f32, pb.max_y as f32]);
                    courtyard_polys.push([v_start, 4, i as u32, 1]);
                } else {
                    all_polys_rect = false;
                    let v_count = (poly.vertices.len() as u32).min(16);
                    for v in &poly.vertices[..v_count as usize] {
                        courtyard_vertices.push([v.x as f32, v.y as f32]);
                    }
                    courtyard_polys.push([v_start, v_count, i as u32, 0]);
                }
                valid_poly_count += 1;
            }
        }

        let mut flags = is_smd;
        if valid_poly_count == 0 {
            let v_start = courtyard_vertices.len() as u32;
            courtyard_vertices.push([-hw, -hh]);
            courtyard_vertices.push([hw, -hh]);
            courtyard_vertices.push([hw, hh]);
            courtyard_vertices.push([-hw, hh]);
            courtyard_polys.push([v_start, 4, i as u32, 1]);
            valid_poly_count = 1;
            flags |= 4u32;
        } else {
            if all_polys_rect {
                flags |= 4u32;
            }
            let is_single_exact_rect = valid_poly_count == 1
                && all_polys_rect
                && center_exact
                && (raw_hw - hw).abs() <= 1e-2
                && (raw_hh - hh).abs() <= 1e-2;
            if !is_single_exact_rect {
                flags |= 2u32;
            }
        }

        fp_poly_ranges.push([poly_start, valid_poly_count, flags, orig_layer]);
    }

    let mut net_pads: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (fp_idx, fp) in board.footprints.iter().enumerate() {
        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if pad.net_id == 0 || pad.net_name.is_empty() || is_ground_net(&pad.net_name) {
                continue;
            }
            net_pads.entry(pad.net_id).or_default().push((fp_idx, pad_idx));
        }
    }

    let mut edges: Vec<[f32; 4]> = Vec::new();
    let mut net_ids: Vec<usize> = net_pads.keys().copied().collect();
    net_ids.sort_unstable();
    for nid in net_ids {
        let pads = &net_pads[&nid];
        if pads.len() < 2 {
            continue;
        }
        let mut visited = vec![false; pads.len()];
        visited[0] = true;
        let mut count = 1usize;
        while count < pads.len() {
            let mut best_d = f64::INFINITY;
            let mut best_pair = (0usize, 0usize);
            for i in 0..pads.len() {
                if !visited[i] {
                    continue;
                }
                let pi = base.cached_pads[pads[i].0][pads[i].1];
                for j in 0..pads.len() {
                    if visited[j] {
                        continue;
                    }
                    let pj = base.cached_pads[pads[j].0][pads[j].1];
                    let d = pi.distance_to(&pj);
                    if d < best_d {
                        best_d = d;
                        best_pair = (i, j);
                    }
                }
            }
            visited[best_pair.1] = true;
            count += 1;
            let (fpa, pa) = pads[best_pair.0];
            let (fpb, pb) = pads[best_pair.1];
            if fpa != fpb {
                let pad_a = &board.footprints[fpa].pads[pa];
                let pad_b = &board.footprints[fpb].pads[pb];
                edges.push([fpa as f32, pad_a.offset_x as f32, pad_a.offset_y as f32, 0.0]);
                edges.push([fpb as f32, pad_b.offset_x as f32, pad_b.offset_y as f32, 0.0]);
            }
        }
    }

    // Place corridor cut segments (tag = 3.0) immediately after ratsnest airwires (tag = 0.0)
    // so inner crossing loops can break as soon as they reach passive ties (1.0/2.0) or flow edges (4.0).
    if config.corridor_congestion_weight > 0.0 {
        let locked_indices: Vec<usize> = board
            .footprints
            .iter()
            .enumerate()
            .filter(|(_, fp)| fp.is_locked)
            .map(|(i, _)| i)
            .collect();
        let mut corr_count = 0usize;
        for i in 0..locked_indices.len() {
            for j in (i + 1)..locked_indices.len() {
                if corr_count >= 2 {
                    break;
                }
                edges.push([locked_indices[i] as f32, 0.0, 0.0, 3.0]);
                edges.push([locked_indices[j] as f32, 0.0, 0.0, 3.0]);
                corr_count += 1;
            }
        }
        if corr_count == 0 {
            for link in base.cached_connectivity.1.iter().take(2) {
                edges.push([link.comp_a as f32, 0.0, 0.0, 3.0]);
                edges.push([link.comp_b as f32, 0.0, 0.0, 3.0]);
            }
        }
    }

    for tie in ties {
        let pad_p = &board.footprints[tie.passive_idx].pads[tie.passive_pad_idx];
        let pad_a = &board.footprints[tie.anchor_idx].pads[tie.anchor_pad_idx];
        let tag = if tie.is_decoupling { 2.0 } else { 1.0 };
        edges.push([
            tie.passive_idx as f32,
            pad_p.offset_x as f32,
            pad_p.offset_y as f32,
            tag,
        ]);
        edges.push([
            tie.anchor_idx as f32,
            pad_a.offset_x as f32,
            pad_a.offset_y as f32,
            tag,
        ]);
    }

    if config.flow_alignment_weight > 0.0 {
        let mut flow_count = 0usize;
        for link in base.cached_connectivity.1.iter().take(2) {
            if let Some(&(pa_idx, pb_idx)) = link.shared_pads.first() {
                let pad_a = &board.footprints[link.comp_a].pads[pa_idx];
                let pad_b = &board.footprints[link.comp_b].pads[pb_idx];
                edges.push([
                    link.comp_a as f32,
                    pad_a.offset_x as f32,
                    pad_a.offset_y as f32,
                    4.0,
                ]);
                edges.push([
                    link.comp_b as f32,
                    pad_b.offset_x as f32,
                    pad_b.offset_y as f32,
                    4.0,
                ]);
                flow_count += 1;
            }
        }
        'cl_loop: for cl in &board.clusters {
            if let Some(a_idx) = cl.anchor_idx {
                for &m_idx in cl.member_indices.iter().take(2) {
                    if m_idx != a_idx && !board.footprints[m_idx].is_locked {
                        edges.push([m_idx as f32, 0.0, 0.0, 4.0]);
                        edges.push([a_idx as f32, 0.0, 0.0, 4.0]);
                        flow_count += 1;
                        if flow_count >= 3 {
                            break 'cl_loop;
                        }
                    }
                }
            }
        }
    }

    let courtyards = GpuCourtyardBuffers {
        fp_poly_ranges,
        courtyard_polys,
        courtyard_vertices,
    };

    (fp_descs, edges, courtyards)
}

/// Dispatches a parallel batch of Simulated Annealing Markov chains on a Vulkan GPU device and
/// verifies the top-scoring candidate states on the CPU.
///
/// Launches `4096` chains (`128` for small boards with $N_f \le 8$) seeded from `base` via
/// [`VulkanEngine::run_gpu_sa_placer_chains`]. The top 4 GPU-ranked chains are reconstructed into
/// full CPU [`PlacementState`]s, evaluated with exact SAT courtyard and LBVH crossing checks, and
/// subjected to a fast 15-step local overlap-push repair if `1..=2` minor overlaps remain alongside
/// improved wirelength.
///
/// # Arguments
/// * `vk` - Active [`VulkanEngine`] device to dispatch the compute shader on.
/// * `board` - Board geometry and netlist reference.
/// * `config` - Placement hyperparameters and weights.
/// * `ties` - Precomputed [`PassiveTie`] constraints.
/// * `movable` - Slice of unlocked footprint indices.
/// * `base` - Starting [`PlacementState`] used to seed all GPU Markov chains.
/// * `fp_descs` - Prebuilt GPU footprint descriptor buffer (`[f32; 4]` per footprint).
/// * `edges` - Prebuilt GPU connectivity and affinity edge buffer (`[f32; 4]` per endpoint).
/// * `courtyards` - Prebuilt [`GpuCourtyardBuffers`] for GPU polygon collision checks.
/// * `seed_offset` - Monotonically increasing batch index mixed into the PRNG seed.
/// * `cooling` - Current annealing cooling factor in `[0.05, 1.0]`.
///
/// # Returns
/// `Some((best_verified_state, total_gpu_evaluations))` if the GPU dispatch succeeded, or `None` if
/// `movable` is empty or the GPU compute pipeline returned no candidates.
fn run_gpu_placement_batch(
    vk: &VulkanEngine,
    board: &Board,
    config: &PlacementConfig,
    ties: &[PassiveTie],
    movable: &[usize],
    base: &PlacementState,
    fp_descs: &[[f32; 4]],
    edges: &[[f32; 4]],
    courtyards: &GpuCourtyardBuffers,
    seed_offset: u64,
    cooling: f64,
) -> Option<(PlacementState, usize)> {
    if movable.is_empty() || board.footprints.is_empty() {
        return None;
    }
    let nf = board.footprints.len();
    let ne = edges.len() / 2;
    let boundary = board.boundary_aabb();

    let (num_chains, steps_per_pass, num_passes): (u32, u32, u32) = if nf > 8 {
        (4096, 18, 4)
    } else {
        (128, 6, 2)
    };

    let mut init_poses = Vec::with_capacity((num_chains as usize) * nf);
    for _ in 0..num_chains {
        for i in 0..nf {
            init_poses.push([
                base.positions[i].x as f32,
                base.positions[i].y as f32,
                base.rotations[i] as f32,
                base.layers[i] as f32,
            ]);
        }
    }

    let seed = (0xA5A5_5A5Au32).wrapping_add((seed_offset as u32).wrapping_mul(0x9E37_79B9));
    let sa_params = GpuSaParams {
        num_chains,
        num_footprints: nf as u32,
        num_edges: ne as u32,
        steps_per_pass,
        seed,
        _pad0: 0,
        corridor_congestion_weight: config.corridor_congestion_weight as f32,
        flow_alignment_weight: config.flow_alignment_weight as f32,
        board_min_x: boundary.min_x as f32,
        board_min_y: boundary.min_y as f32,
        board_max_x: boundary.max_x as f32,
        board_max_y: boundary.max_y as f32,
        grid_step: config.grid_step_mm as f32,
        wirelength_weight: config.wirelength_weight as f32,
        crossing_weight: config.crossing_weight as f32,
        overlap_penalty: config.overlap_penalty as f32,
        boundary_penalty: config.boundary_penalty as f32,
        passive_affinity_weight: config.passive_affinity_weight as f32,
        passive_max_dist: config.passive_max_dist as f32,
        base_temp: (8.0 * cooling).max(0.5) as f32,
    };

    let (best_poses, best_scores) =
        vk.run_gpu_sa_placer_chains(fp_descs, edges, &init_poses, &sa_params, num_passes, Some(courtyards))?;

    let mut ranked: Vec<(usize, f32)> = best_scores
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s[1]))
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let total_gpu_evals = (num_chains as usize) * (steps_per_pass as usize) * (num_passes as usize);
    let mut best_verified: Option<PlacementState> = None;

    for &(chain_idx, _) in ranked.iter().take(4) {
        let mut cand = base.clone();
        let c_base = chain_idx * nf;
        for i in 0..nf {
            if !board.footprints[i].is_locked {
                let p = best_poses[c_base + i];
                cand.positions[i] = Point::new(p[0] as f64, p[1] as f64);
                cand.rotations[i] = p[2] as f64;
                cand.update_component(i, board, config);
            }
        }
        cand.evaluate(board, config, ties);
        if cand.overlaps > 0 && cand.overlaps <= 2 && cand.boundary_violations == 0 && cand.wirelength < base.wirelength {
            let mut resolved = cand.clone();
            for _ in 0..15 {
                if resolved.overlaps == 0 { break; }
                let mut ov_pair = None;
                'find_ov: for i in 0..nf {
                    for j in (i + 1)..nf {
                        if !resolved.cached_bounds[i].overlaps(&resolved.cached_bounds[j]) { continue; }
                        for pa in &resolved.cached_courtyards[i] {
                            for pb in &resolved.cached_courtyards[j] {
                                if pa.intersects_convex(pb) {
                                    ov_pair = Some((i, j));
                                    break 'find_ov;
                                }
                            }
                        }
                    }
                }
                if let Some((fa, fb)) = ov_pair {
                    let target = if !board.footprints[fa].is_locked { fa } else { fb };
                    if !board.footprints[target].is_locked {
                        let other = if target == fa { fb } else { fa };
                        let dx = resolved.positions[target].x - resolved.positions[other].x;
                        let dy = resolved.positions[target].y - resolved.positions[other].y;
                        let dist = dx.hypot(dy).max(1e-4);
                        let (cx_min, cx_max, cy_min, cy_max) = board.footprints[target]
                            .origin_clamp_range(resolved.rotations[target], resolved.layers[target], config.min_spacing_mm, &boundary);
                        let push_step = config.grid_step_mm;
                        let nx = snap_to_grid((resolved.positions[target].x + (dx / dist) * push_step).clamp(cx_min, cx_max), config.grid_step_mm);
                        let ny = snap_to_grid((resolved.positions[target].y + (dy / dist) * push_step).clamp(cy_min, cy_max), config.grid_step_mm);
                        resolved.positions[target] = Point::new(nx, ny);
                        resolved.update_component(target, board, config);
                        resolved.evaluate(board, config, ties);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            if resolved.overlaps < cand.overlaps {
                cand = resolved;
            }
        }
        if best_verified.as_ref().is_none_or(|b| {
            cand.boundary_violations < b.boundary_violations
                || (cand.boundary_violations == b.boundary_violations && cand.overlaps < b.overlaps)
                || (cand.boundary_violations == b.boundary_violations
                    && cand.overlaps == b.overlaps
                    && cand.cost < b.cost)
        }) {
            best_verified = Some(cand);
        }
    }

    best_verified.map(|c| (c, total_gpu_evals))
}

/// Resolves courtyard overlaps and board outline / keepout violations in `initial` using a
/// hybrid GPU SA pre-pass followed by multi-core CPU repulsive legalization.
///
/// # Algorithm
/// 1. Returns `initial` immediately if `overlaps == 0 && boundary_violations == 0`.
/// 2. Clamps all unlocked footprints strictly within their valid board outline origin range
///    (`Footprint::origin_clamp_range`) and re-evaluates.
/// 3. If `config.vulkan` is present, runs a high-penalty GPU SA pre-pass (`1024` chains, `4x` overlap
///    and boundary penalties) and returns immediately if any chain achieves zero violations.
/// 4. Otherwise spawns `num_threads` parallel Rayon workers that iteratively identify colliding
///    footprint pairs and apply repulsive separation vectors, random neighborhood shifts, $90^\circ$
///    rotations, or optional SMD layer flips until all overlaps and boundary violations are eliminated
///    (short-circuiting across all threads via an `AtomicBool` once solved).
///
/// # Arguments
/// * `board` - Board geometry, footprints, boundary, and keepouts.
/// * `config` - Placement configuration specifying clearances, grid step, and optional Vulkan engine.
/// * `initial` - Starting [`PlacementState`] potentially containing courtyard overlaps or boundary excursions.
/// * `live` - Optional shared [`LiveTrialState`] for GUI phase and log updates.
///
/// # Returns
/// Legalized [`PlacementState`] minimizing boundary violations first, courtyard overlaps second, and total cost third.
///
/// # Side Effects
/// Updates `live` phase to `TrialPhase::Legalizing` and appends diagnostic log entries when `live` is `Some`.
pub fn legalize_placement(
    board: &Board,
    config: &PlacementConfig,
    initial: PlacementState,
    live: Option<&Arc<Mutex<LiveTrialState>>>,
) -> PlacementState {
    if initial.overlaps == 0 && initial.boundary_violations == 0 {
        return initial;
    }
    let num_threads = rayon::current_num_threads().max(2);
    if let Some(st) = live {
        if let Ok(mut g) = st.lock() {
            g.phase = TrialPhase::Legalizing;
            g.push_log(format!(
                "[Legalizer] Resolving {} initial courtyard overlaps across {} CPU cores...",
                initial.overlaps, num_threads
            ));
        }
    }

    let movable: Vec<usize> = board
        .footprints
        .iter()
        .enumerate()
        .filter(|(_, fp)| !fp.is_locked)
        .map(|(i, _)| i)
        .collect();
    if movable.is_empty() {
        return initial;
    }

    let boundary = board.boundary_aabb();
    let ties = build_passive_ties(board);
    let mut initial = initial;
    for i in &movable {
        let (cx_min, cx_max, cy_min, cy_max) = board.footprints[*i].origin_clamp_range(
            initial.rotations[*i],
            initial.layers[*i],
            config.min_spacing_mm,
            &boundary,
        );
        initial.positions[*i].x = initial.positions[*i].x.clamp(cx_min, cx_max);
        initial.positions[*i].y = initial.positions[*i].y.clamp(cy_min, cy_max);
        initial.update_component(*i, board, config);
    }
    initial.evaluate(board, config, &ties);
    if initial.overlaps == 0 && initial.boundary_violations == 0 {
        return initial;
    }

    if let Some(ref vk) = config.vulkan {
        let (fp_descs, edges, courtyards) =
            build_gpu_sa_descriptors(board, config, &ties, &initial);
        let nf = board.footprints.len();
        let ne = edges.len() / 2;
        let (num_chains, steps_per_pass, num_passes): (u32, u32, u32) = if nf > 8 {
            (1024, 24, 4)
        } else {
            (128, 12, 3)
        };
        let mut init_poses = Vec::with_capacity((num_chains as usize) * nf);
        for _ in 0..num_chains {
            for i in 0..nf {
                init_poses.push([
                    initial.positions[i].x as f32,
                    initial.positions[i].y as f32,
                    initial.rotations[i] as f32,
                    initial.layers[i] as f32,
                ]);
            }
        }
        let sa_params = GpuSaParams {
            num_chains,
            num_footprints: nf as u32,
            num_edges: ne as u32,
            steps_per_pass,
            seed: 0x517C_C1B7u32,
            _pad0: 0,
            corridor_congestion_weight: config.corridor_congestion_weight as f32,
            flow_alignment_weight: config.flow_alignment_weight as f32,
            board_min_x: boundary.min_x as f32,
            board_min_y: boundary.min_y as f32,
            board_max_x: boundary.max_x as f32,
            board_max_y: boundary.max_y as f32,
            grid_step: config.grid_step_mm as f32,
            wirelength_weight: config.wirelength_weight as f32,
            crossing_weight: config.crossing_weight as f32,
            overlap_penalty: (config.overlap_penalty * 4.0).max(100_000.0) as f32,
            boundary_penalty: (config.boundary_penalty * 4.0).max(200_000.0) as f32,
            passive_affinity_weight: config.passive_affinity_weight as f32,
            passive_max_dist: config.passive_max_dist as f32,
            base_temp: 12.0,
        };
        if let Some((best_poses, best_scores)) = vk.run_gpu_sa_placer_chains(
            &fp_descs,
            &edges,
            &init_poses,
            &sa_params,
            num_passes,
            Some(&courtyards),
        ) {
            let mut ranked: Vec<(usize, f32, f32)> = best_scores
                .iter()
                .enumerate()
                .map(|(i, s)| (i, s[2], s[1]))
                .collect();
            ranked.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            });
            let mut best_gpu: Option<PlacementState> = None;
            for &(chain_idx, _, _) in ranked.iter().take(8) {
                let mut cand = initial.clone();
                let c_base = chain_idx * nf;
                for i in 0..nf {
                    if !board.footprints[i].is_locked {
                        let p = best_poses[c_base + i];
                        let rot = p[2] as f64;
                        let (cx_min, cx_max, cy_min, cy_max) =
                            board.footprints[i].origin_clamp_range(
                                rot,
                                cand.layers[i],
                                config.min_spacing_mm,
                                &boundary,
                            );
                        cand.positions[i] = Point::new(
                            snap_to_grid((p[0] as f64).clamp(cx_min, cx_max), config.grid_step_mm),
                            snap_to_grid((p[1] as f64).clamp(cy_min, cy_max), config.grid_step_mm),
                        );
                        cand.rotations[i] = rot;
                        cand.update_component(i, board, config);
                    }
                }
                cand.evaluate(board, config, &ties);
                if cand.overlaps == 0 && cand.boundary_violations == 0 {
                    if let Some(st) = live {
                        if let Ok(mut g) = st.lock() {
                            g.push_log(format!(
                                "[Legalizer] GPU SA pre-pass resolved all overlaps across {} chains",
                                num_chains
                            ));
                        }
                    }
                    return cand;
                }
                if best_gpu.as_ref().is_none_or(|b| {
                    cand.boundary_violations < b.boundary_violations
                        || (cand.boundary_violations == b.boundary_violations
                            && cand.overlaps < b.overlaps)
                        || (cand.boundary_violations == b.boundary_violations
                            && cand.overlaps == b.overlaps
                            && cand.cost < b.cost)
                }) {
                    best_gpu = Some(cand);
                }
            }
            if let Some(gpu_cand) = best_gpu {
                if gpu_cand.boundary_violations < initial.boundary_violations
                    || (gpu_cand.boundary_violations == initial.boundary_violations
                        && gpu_cand.overlaps < initial.overlaps)
                    || (gpu_cand.boundary_violations == initial.boundary_violations
                        && gpu_cand.overlaps == initial.overlaps
                        && gpu_cand.cost < initial.cost)
                {
                    initial = gpu_cand;
                }
            }
        }
    }

    let solved = AtomicBool::new(false);
    let steps_per_thread = (10000 / num_threads).max(2500);

    let results: Vec<PlacementState> = (0..num_threads)
        .into_par_iter()
        .map(|tid| {
            let seed = if tid == 0 {
                0x517CC1B727220A95u64
            } else {
                0x517CC1B727220A95u64 ^ ((tid as u64).wrapping_mul(0x9E3779B97F4A7C15))
            };
            let mut rng = StdRng::seed_from_u64(seed);
            let mut cur = initial.clone();
            let mut local_best = cur.clone();

            for step in 0..steps_per_thread {
                if cur.overlaps == 0 && cur.boundary_violations == 0 {
                    solved.store(true, Ordering::Relaxed);
                    break;
                }
                if step % 16 == 0 && solved.load(Ordering::Relaxed) {
                    break;
                }

                let overlapping_pairs: Vec<(usize, usize)> = if cur.overlaps > 0 && rng.gen_bool(0.90) {
                    let mut ov_pairs = Vec::new();
                    for i in 0..cur.cached_bounds.len() {
                        for j in (i + 1)..cur.cached_bounds.len() {
                            if board.footprints[i].is_locked && board.footprints[j].is_locked {
                                continue;
                            }
                            if board.footprints[i].is_smd_only() && board.footprints[j].is_smd_only() && cur.layers[i] != cur.layers[j] {
                                continue;
                            }
                            if cur.cached_bounds[i].overlaps(&cur.cached_bounds[j]) {
                                let mut hit = false;
                                for pa in &cur.cached_courtyards[i] {
                                    for pb in &cur.cached_courtyards[j] {
                                        if pa.intersects_convex(pb) {
                                            hit = true;
                                            break;
                                        }
                                    }
                                    if hit { break; }
                                }
                                if hit {
                                    if !board.footprints[i].is_locked { ov_pairs.push((i, j)); }
                                    if !board.footprints[j].is_locked { ov_pairs.push((j, i)); }
                                }
                            }
                        }
                    }
                    ov_pairs
                } else {
                    Vec::new()
                };

                let (idx, other_opt) = if !overlapping_pairs.is_empty() {
                    let p = overlapping_pairs[rng.gen_range(0..overlapping_pairs.len())];
                    (p.0, Some(p.1))
                } else {
                    (movable[rng.gen_range(0..movable.len())], None)
                };

                let old_pos = cur.positions[idx];
                let old_rot = cur.rotations[idx];
                let old_layer = cur.layers[idx];
                let old_cost = cur.cost;
                let old_ov = cur.overlaps;
                let old_bv = cur.boundary_violations;

                let roll = rng.gen_range(0..100);
                if roll < 70 {
                    let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx].origin_clamp_range(
                        cur.rotations[idx],
                        cur.layers[idx],
                        config.min_spacing_mm,
                        &boundary,
                    );
                    if let Some(other) = other_opt {
                        if rng.gen_bool(0.65) {
                            let dx = cur.positions[idx].x - cur.positions[other].x;
                            let dy = cur.positions[idx].y - cur.positions[other].y;
                            let dist = dx.hypot(dy).max(1e-4);
                            let (ux, uy) = (dx / dist, dy / dist);
                            let push_dist = rng.gen_range(1.0..=12.0);
                            let jitter = rng.gen_range(-1.5..=1.5);
                            let nx = snap_to_grid(
                                (old_pos.x + ux * push_dist + jitter).clamp(cx_min, cx_max),
                                config.grid_step_mm,
                            );
                            let ny = snap_to_grid(
                                (old_pos.y + uy * push_dist + jitter).clamp(cy_min, cy_max),
                                config.grid_step_mm,
                            );
                            cur.positions[idx] = Point::new(nx, ny);
                        } else {
                            let r = if step < steps_per_thread / 3 { 8.0 } else { 3.5 };
                            let nx = snap_to_grid(
                                (old_pos.x + rng.gen_range(-r..=r)).clamp(cx_min, cx_max),
                                config.grid_step_mm,
                            );
                            let ny = snap_to_grid(
                                (old_pos.y + rng.gen_range(-r..=r)).clamp(cy_min, cy_max),
                                config.grid_step_mm,
                            );
                            cur.positions[idx] = Point::new(nx, ny);
                        }
                    } else {
                        let r = if step < steps_per_thread / 3 { 6.0 } else { 2.5 };
                        let nx = snap_to_grid(
                            (old_pos.x + rng.gen_range(-r..=r)).clamp(cx_min, cx_max),
                            config.grid_step_mm,
                        );
                        let ny = snap_to_grid(
                            (old_pos.y + rng.gen_range(-r..=r)).clamp(cy_min, cy_max),
                            config.grid_step_mm,
                        );
                        cur.positions[idx] = Point::new(nx, ny);
                    }
                } else if roll < 90 || !config.allow_layer_flip || !board.footprints[idx].is_smd_only() {
                    let angles = [0.0, 90.0, 180.0, 270.0];
                    let new_rot = angles[rng.gen_range(0..4)];
                    cur.rotations[idx] = new_rot;
                    let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx].origin_clamp_range(
                        new_rot,
                        cur.layers[idx],
                        config.min_spacing_mm,
                        &boundary,
                    );
                    cur.positions[idx].x = snap_to_grid(cur.positions[idx].x.clamp(cx_min, cx_max), config.grid_step_mm);
                    cur.positions[idx].y = snap_to_grid(cur.positions[idx].y.clamp(cy_min, cy_max), config.grid_step_mm);
                } else {
                    cur.layers[idx] = 1 - cur.layers[idx];
                    let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx].origin_clamp_range(
                        cur.rotations[idx],
                        cur.layers[idx],
                        config.min_spacing_mm,
                        &boundary,
                    );
                    cur.positions[idx].x = snap_to_grid(cur.positions[idx].x.clamp(cx_min, cx_max), config.grid_step_mm);
                    cur.positions[idx].y = snap_to_grid(cur.positions[idx].y.clamp(cy_min, cy_max), config.grid_step_mm);
                }
                cur.update_component(idx, board, config);
                cur.evaluate(board, config, &ties);

                let accept = (cur.boundary_violations < old_bv)
                    || (cur.boundary_violations == old_bv && (
                        cur.overlaps < old_ov
                            || (cur.overlaps == old_ov && cur.boundary_violations == 0 && (cur.cost <= old_cost || rng.gen_bool(0.15)))
                    ));

                if accept {
                    let is_better = if cur.boundary_violations != local_best.boundary_violations {
                        cur.boundary_violations < local_best.boundary_violations
                    } else if cur.overlaps != local_best.overlaps {
                        cur.overlaps < local_best.overlaps
                    } else {
                        cur.cost < local_best.cost
                    };
                    if is_better {
                        local_best = cur.clone();
                    }
                    if local_best.overlaps == 0 && local_best.boundary_violations == 0 {
                        solved.store(true, Ordering::Relaxed);
                        break;
                    }
                } else {
                    cur.positions[idx] = old_pos;
                    cur.rotations[idx] = old_rot;
                    cur.layers[idx] = old_layer;
                    cur.update_component(idx, board, config);
                    cur.cost = old_cost;
                    cur.overlaps = old_ov;
                    cur.boundary_violations = old_bv;
                }
            }
            local_best
        })
        .collect();

    results
        .into_iter()
        .min_by(|a, b| {
            a.boundary_violations
                .cmp(&b.boundary_violations)
                .then_with(|| a.overlaps.cmp(&b.overlaps))
                .then_with(|| a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal))
        })
        .unwrap_or(initial)
}

/// Positions unlocked board components in strict electrical priority order before Simulated Annealing begins.
///
/// # Priority Order
/// 1. **High-Priority Differential Pair Components:** Unlocked footprints connected to differential pair
///    nets (`USB_Diff` netclass or `USB_D*`/`DP`/`DM`/`D+`/`D-`) are positioned along direct horizontal
///    routing corridors (`12.0` mm inward from any locked differential anchor such as a USB connector).
/// 2. **Decoupling Capacitors:** Each 2-pin decoupling capacitor identified via [`build_passive_ties`]
///    is placed on a radial polar grid (`0.8..=4.0` mm) outward from its target active IC power pin,
///    filtered for legality via [`PlacedBoardSpatialIndex`] and exact courtyard SAT checks, and optionally
///    batch-scored on the GPU via [`score_candidate_poses_gpu`] to minimize pad-to-pin loop length.
/// 3. **Boundary Containment:** Any remaining unlocked footprints staged outside the board outline are
///    clamped inside `Footprint::origin_clamp_range`.
///
/// # Arguments
/// * `board` - Mutable [`Board`] whose unlocked footprint coordinates and rotations are updated.
/// * `config` - Placement configuration specifying grid step, spacing, escape margins, and optional Vulkan engine.
/// * `live` - Optional shared [`LiveTrialState`] to record progress log messages.
///
/// # Side Effects
/// Mutates `x`, `y`, and `rotation` of unlocked footprints in `board.footprints` and appends a log
/// message to `live` when `Some`.
pub fn prioritize_and_position_initial_components(
    board: &mut Board,
    config: &PlacementConfig,
    live: Option<&Arc<Mutex<LiveTrialState>>>,
) {
    if board.footprints.is_empty() {
        return;
    }

    let boundary = board.boundary_aabb();
    let ties = build_passive_ties(board);

    // Track which movable footprints have been situated
    let mut placed = vec![false; board.footprints.len()];
    for (i, fp) in board.footprints.iter().enumerate() {
        if fp.is_locked {
            placed[i] = true;
        }
    }

    // 1. Priority 1: High priority differential pair components
    let diff_nets: HashSet<usize> = board
        .nets
        .iter()
        .enumerate()
        .filter(|(_, name)| {
            board.net_to_class.get(name.as_str()).map_or(false, |c| c == "USB_Diff")
                || name.contains("USB_D")
                || name.contains("DP")
                || name.contains("DM")
                || name.contains("D+")
                || name.contains("D-")
        })
        .map(|(id, _)| id)
        .collect();

    let mut diff_pair_fps: Vec<usize> = Vec::new();
    for (idx, fp) in board.footprints.iter().enumerate() {
        if !placed[idx] && fp.pads.iter().any(|p| diff_nets.contains(&p.net_id)) {
            diff_pair_fps.push(idx);
        }
    }

    let locked_diff_anchor = board
        .footprints
        .iter()
        .enumerate()
        .find(|(i, fp)| placed[*i] && fp.pads.iter().any(|p| diff_nets.contains(&p.net_id)))
        .map(|(i, _)| i);

    for &diff_idx in &diff_pair_fps {
        if placed[diff_idx] {
            continue;
        }
        if let Some(anchor_idx) = locked_diff_anchor {
            let anchor_pos = Point::new(board.footprints[anchor_idx].x, board.footprints[anchor_idx].y);
            let (target_x, target_y) = if anchor_pos.x > (boundary.min_x + boundary.max_x) * 0.5 {
                (anchor_pos.x - 12.0, anchor_pos.y)
            } else {
                (anchor_pos.x + 12.0, anchor_pos.y)
            };
            let (cx_min, cx_max, cy_min, cy_max) = board.footprints[diff_idx].origin_clamp_range(
                board.footprints[diff_idx].rotation,
                board.footprints[diff_idx].layer,
                config.min_spacing_mm,
                &boundary,
            );
            board.footprints[diff_idx].x = snap_to_grid(
                target_x.clamp(cx_min, cx_max),
                config.grid_step_mm,
            );
            board.footprints[diff_idx].y = snap_to_grid(
                target_y.clamp(cy_min, cy_max),
                config.grid_step_mm,
            );
        }
        placed[diff_idx] = true;
    }

    let mut spatial_index = PlacedBoardSpatialIndex::from_board(board, config.min_spacing_mm);

    // 2. Priority 2: Decoupling capacitors — situating them as close to the connected non-ground pin as possible!
    for tie in &ties {
        if !tie.is_decoupling {
            continue;
        }
        let cap_idx = tie.passive_idx;
        if placed[cap_idx] {
            continue;
        }
        let anchor_idx = tie.anchor_idx;
        let anchor_pad_idx = tie.anchor_pad_idx;
        let cap_pad_idx = tie.passive_pad_idx;

        let dev = &board.footprints[anchor_idx];
        let dev_pos = Point::new(dev.x, dev.y);
        let dev_pin = &dev.pads[anchor_pad_idx];
        let (pin_x, pin_y) = dev.pad_absolute_position(dev_pin);
        let pin_pt = Point::new(pin_x, pin_y);

        let mut dx = pin_x - dev_pos.x;
        let mut dy = pin_y - dev_pos.y;
        let len = dx.hypot(dy);
        if len > 1e-4 {
            dx /= len;
            dy /= len;
        } else {
            dx = 1.0;
            dy = 0.0;
        }
        let base_angle = dy.atan2(dx);

        let mut placed_courtyards: Vec<(AABB, Vec<Polygon>)> = Vec::new();
        for (i, fp) in board.footprints.iter().enumerate() {
            if placed[i] && i != cap_idx {
                let polys = fp.transformed_courtyards_layered(
                    Point::new(fp.x, fp.y),
                    fp.rotation,
                    fp.layer,
                    fp.effective_spacing(config),
                );
                let mut aabb = AABB::new(f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for p in &polys {
                    let pa = p.aabb();
                    aabb.min_x = aabb.min_x.min(pa.min_x);
                    aabb.min_y = aabb.min_y.min(pa.min_y);
                    aabb.max_x = aabb.max_x.max(pa.max_x);
                    aabb.max_y = aabb.max_y.max(pa.max_y);
                }
                placed_courtyards.push((aabb, polys));
            }
        }

        let mut best_cand: Option<(f64, f64, f64, f64)> = None;

        let dist_steps = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0];
        let angle_deltas = [
            0.0, 0.20, -0.20, 0.40, -0.40, 0.60, -0.60, 0.80, -0.80, 1.0, -1.0, 1.2, -1.2, 1.4, -1.4,
        ];
        let rotations = [0.0, 90.0, 180.0, 270.0];

        let cap = &board.footprints[cap_idx];
        let cap_spacing = cap.effective_spacing(config);
        let anchor_escape = if config.enable_escape_clearance {
            dev.escape_margin * config.escape_margin_mult
        } else {
            0.0
        };

        let mut legal_cands: Vec<(Point, f64, usize)> = Vec::new();
        for &d in &dist_steps {
            for &da in &angle_deltas {
                let angle = base_angle + da;
                for &rot in &rotations {
                    let (cx_min, cx_max, cy_min, cy_max) =
                        cap.origin_clamp_range(rot, cap.layer, config.min_spacing_mm, &boundary);
                    let cx = snap_to_grid(
                        (pin_x + d * angle.cos()).clamp(cx_min, cx_max),
                        config.grid_step_mm,
                    );
                    let cy = snap_to_grid(
                        (pin_y + d * angle.sin()).clamp(cy_min, cy_max),
                        config.grid_step_mm,
                    );
                    let c_pos = Point::new(cx, cy);

                    if !spatial_index.is_placement_legal(board, cap, c_pos, rot, cap.layer) {
                        continue;
                    }

                    let polys =
                        cap.transformed_courtyards_layered(c_pos, rot, cap.layer, cap_spacing);
                    let mut collides = false;
                    'check: for (group_aabb, group) in &placed_courtyards {
                        for p_cand in &polys {
                            if !p_cand.aabb().overlaps(group_aabb) {
                                continue;
                            }
                            for p_placed in group {
                                if p_cand.intersects_convex(p_placed) {
                                    collides = true;
                                    break 'check;
                                }
                            }
                        }
                    }
                    if !collides {
                        legal_cands.push((c_pos, rot, cap.layer));
                    }
                }
            }
        }

        let gpu_scores = config.vulkan.as_ref().and_then(|vk| {
            score_candidate_poses_gpu(
                vk,
                board,
                cap,
                &legal_cands,
                &[(cap_pad_idx, pin_pt, 1.0)],
                config,
            )
        });

        for (i, &(c_pos, rot, _)) in legal_cands.iter().enumerate() {
            let cap_pad_world = cap.pad_world_point_layered(cap_pad_idx, c_pos, rot, cap.layer);
            let pad_dist = cap_pad_world.distance_to(&pin_pt);
            let score = gpu_scores
                .as_ref()
                .and_then(|s| s.get(i).copied())
                .map(|gs| (gs as f64) * 0.01 + pad_dist)
                .unwrap_or(pad_dist);
            if best_cand.is_none_or(|b| score < b.0) {
                best_cand = Some((score, c_pos.x, c_pos.y, rot));
                if pad_dist <= 1.0 + anchor_escape {
                    break;
                }
            }
        }

        if let Some((_, best_x, best_y, best_rot)) = best_cand {
            board.footprints[cap_idx].x = best_x;
            board.footprints[cap_idx].y = best_y;
            board.footprints[cap_idx].rotation = best_rot;
            placed[cap_idx] = true;
            spatial_index.update_footprint(board, cap_idx);
        }
    }

    // Decoupling caps have been placed directly at their power pins.
    // Ensure any remaining unlocked components staged outside board boundary are pulled inside.
    for fp in board.footprints.iter_mut() {
        if !fp.is_locked {
            let (cx_min, cx_max, cy_min, cy_max) = fp.origin_clamp_range(fp.rotation, fp.layer, config.min_spacing_mm, &boundary);
            fp.x = fp.x.clamp(cx_min, cx_max);
            fp.y = fp.y.clamp(cy_min, cy_max);
        }
    }

    if let Some(st) = live {
        if let Ok(mut g) = st.lock() {
            g.push_log("[Auto-Placer] Prioritized differential pairs and situated decoupling caps near device power pins".to_string());
        }
    }
}

/// Cached world-space courtyard polygons and bounding boxes for a single placed footprint in [`PlacedBoardSpatialIndex`].
#[derive(Clone, Debug)]
struct PlacedFootprintSpatialEntry {
    /// Component reference designator (e.g., `"U1"`, `"C12"`), used to skip self-collision checks.
    reference: String,
    /// Copper layer index (`0` for `F.Cu`, `1` for `B.Cu`).
    layer: usize,
    /// `true` if the footprint has only SMD pads (`false` if any THT/NPTH pad spans all layers).
    is_smd_only: bool,
    /// Pre-transformed convex courtyard polygons paired with their individual [`AABB`]s.
    courtyards: Vec<(Polygon, AABB)>,
    /// Union [`AABB`] enclosing all polygons in `courtyards`.
    overall_aabb: AABB,
}

/// Precomputed spatial index over placed board footprints and keepouts using a 2D Morton LBVH
/// (`Vec<GpuBvhNode>`) to accelerate candidate pose legality queries from $O(N_f)$ polygon
/// transformations per candidate to $O(\log N_f)$ AABB-BVH pruning + exact SAT verification.
#[derive(Clone, Debug)]
pub struct PlacedBoardSpatialIndex {
    /// Base courtyard clearance margin in millimeters (`mm`) baked into the indexed polygons.
    pub min_spacing_mm: f64,
    /// Per-footprint cached world-space courtyard polygons and bounding boxes.
    footprints: Vec<PlacedFootprintSpatialEntry>,
    /// Raw LBVH leaf primitive tuples `([min_x, min_y, max_x, max_y], layer_mask, fp_idx)`.
    primitives: Vec<([f32; 4], u32, u32)>,
    /// Precomputed [`AABB`]s for all board keepout zones.
    keepout_aabbs: Vec<AABB>,
    /// Flattened 2D Morton Linear Bounding Volume Hierarchy nodes built via [`build_gpu_lbvh`].
    bvh_nodes: Vec<crate::vulkan::GpuBvhNode>,
}

impl PlacedBoardSpatialIndex {
    /// Builds a [`PlacedFootprintSpatialEntry`] and its corresponding LBVH primitive tuple for `fp`.
    ///
    /// # Arguments
    /// * `fp` - Footprint whose world-space courtyards and layer mask are computed.
    /// * `min_spacing_mm` - Base courtyard spacing in millimeters (`mm`).
    /// * `idx` - Footprint index stored as the LBVH leaf primitive payload.
    ///
    /// # Returns
    /// Tuple `(entry, ([min_x, min_y, max_x, max_y], layer_mask, idx))`.
    fn build_entry(
        fp: &crate::board::Footprint,
        min_spacing_mm: f64,
        idx: usize,
    ) -> (PlacedFootprintSpatialEntry, ([f32; 4], u32, u32)) {
        let spacing = fp.placement_spacing(min_spacing_mm);
        let polys = fp.transformed_courtyards_layered(
            Point::new(fp.x, fp.y),
            fp.rotation,
            fp.layer,
            spacing,
        );
        let mut overall = AABB::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        let mut courtyards = Vec::with_capacity(polys.len());
        for p in polys {
            let aabb = p.aabb();
            overall.min_x = overall.min_x.min(aabb.min_x);
            overall.min_y = overall.min_y.min(aabb.min_y);
            overall.max_x = overall.max_x.max(aabb.max_x);
            overall.max_y = overall.max_y.max(aabb.max_y);
            courtyards.push((p, aabb));
        }
        if courtyards.is_empty() {
            overall = AABB::new(fp.x, fp.y, fp.x, fp.y);
        }
        let is_smd = fp.is_smd_only();
        let layer_mask = if is_smd {
            1u32 << (fp.layer & 31)
        } else {
            0xFFFF_FFFFu32
        };
        let eps = 1e-3f32;
        let prim = (
            [
                (overall.min_x as f32) - eps,
                (overall.min_y as f32) - eps,
                (overall.max_x as f32) + eps,
                (overall.max_y as f32) + eps,
            ],
            layer_mask,
            idx as u32,
        );
        (
            PlacedFootprintSpatialEntry {
                reference: fp.reference.clone(),
                layer: fp.layer,
                is_smd_only: is_smd,
                courtyards,
                overall_aabb: overall,
            },
            prim,
        )
    }

    /// Constructs a [`PlacedBoardSpatialIndex`] over all footprints and keepouts currently on `board`.
    ///
    /// # Arguments
    /// * `board` - Board whose footprints and keepout zones are indexed.
    /// * `min_spacing_mm` - Base courtyard clearance margin in millimeters (`mm`).
    ///
    /// # Returns
    /// Newly constructed [`PlacedBoardSpatialIndex`] with a built 2D Morton LBVH.
    pub fn from_board(board: &Board, min_spacing_mm: f64) -> Self {
        let mut footprints = Vec::with_capacity(board.footprints.len());
        let mut primitives = Vec::with_capacity(board.footprints.len());
        for (idx, fp) in board.footprints.iter().enumerate() {
            let (entry, prim) = Self::build_entry(fp, min_spacing_mm, idx);
            footprints.push(entry);
            primitives.push(prim);
        }
        let keepout_aabbs = board
            .keepouts
            .iter()
            .map(|ko| AABB::new(ko.min_x, ko.min_y, ko.max_x, ko.max_y))
            .collect();
        let bvh_nodes = crate::vulkan::build_gpu_lbvh(&primitives);
        Self {
            min_spacing_mm,
            footprints,
            primitives,
            keepout_aabbs,
            bvh_nodes,
        }
    }

    /// Recomputes the spatial entry for a single moved footprint `fp_idx` and rebuilds the LBVH tree.
    ///
    /// Does nothing if `fp_idx` is out of bounds.
    ///
    /// # Arguments
    /// * `board` - Board containing the updated footprint pose at `board.footprints[fp_idx]`.
    /// * `fp_idx` - Index of the modified footprint.
    ///
    /// # Side Effects
    /// Mutates `self.footprints[fp_idx]`, `self.primitives[fp_idx]`, and rebuilds `self.bvh_nodes`.
    pub fn update_footprint(&mut self, board: &Board, fp_idx: usize) {
        if fp_idx >= board.footprints.len() || fp_idx >= self.footprints.len() {
            return;
        }
        let (entry, prim) = Self::build_entry(&board.footprints[fp_idx], self.min_spacing_mm, fp_idx);
        self.footprints[fp_idx] = entry;
        self.primitives[fp_idx] = prim;
        self.bvh_nodes = crate::vulkan::build_gpu_lbvh(&self.primitives);
    }

    /// Appends a newly placed footprint `fp` to the spatial index and rebuilds the LBVH tree.
    ///
    /// # Arguments
    /// * `fp` - Newly placed footprint to insert.
    ///
    /// # Side Effects
    /// Pushes onto `self.footprints` and `self.primitives`, and rebuilds `self.bvh_nodes`.
    pub fn push_footprint(&mut self, fp: &crate::board::Footprint) {
        let idx = self.footprints.len();
        let (entry, prim) = Self::build_entry(fp, self.min_spacing_mm, idx);
        self.footprints.push(entry);
        self.primitives.push(prim);
        self.bvh_nodes = crate::vulkan::build_gpu_lbvh(&self.primitives);
    }

    /// Queries the 2D Morton LBVH to determine whether candidate courtyard polygons `cand_polys`
    /// for footprint `fp` on `layer` collide with any already-indexed footprint courtyard.
    ///
    /// Traverses `self.bvh_nodes` using an explicit 64-entry stack, pruning internal subtrees via
    /// layer bitmasks and float AABBs before performing exact convex SAT checks (`Polygon::intersects_convex`)
    /// at leaf nodes. Automatically ignores self-collisions (`entry.reference == fp.reference`) and
    /// opposite-side SMD-only pairs.
    ///
    /// # Arguments
    /// * `fp` - Candidate footprint being tested.
    /// * `layer` - Candidate copper layer index (`0` for `F.Cu`, `1` for `B.Cu`).
    /// * `cand_polys` - Pre-transformed world-space courtyard polygons for the candidate pose.
    ///
    /// # Returns
    /// `true` if any courtyard collision is detected; `false` otherwise.
    #[inline]
    pub fn collides_with_placed_courtyards(
        &self,
        fp: &crate::board::Footprint,
        layer: usize,
        cand_polys: &[Polygon],
    ) -> bool {
        if self.footprints.is_empty() || cand_polys.is_empty() {
            return false;
        }

        let mut cand_overall = AABB::new(
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        let mut cand_aabbs: Vec<AABB> = Vec::with_capacity(cand_polys.len());
        for p in cand_polys {
            let a = p.aabb();
            cand_overall.min_x = cand_overall.min_x.min(a.min_x);
            cand_overall.min_y = cand_overall.min_y.min(a.min_y);
            cand_overall.max_x = cand_overall.max_x.max(a.max_x);
            cand_overall.max_y = cand_overall.max_y.max(a.max_y);
            cand_aabbs.push(a);
        }

        let cand_is_smd = fp.is_smd_only();
        let query_layer_mask = if cand_is_smd {
            1u32 << (layer & 31)
        } else {
            0xFFFF_FFFFu32
        };
        let eps = 1e-3f32;
        let q_min_x = (cand_overall.min_x as f32) - eps;
        let q_min_y = (cand_overall.min_y as f32) - eps;
        let q_max_x = (cand_overall.max_x as f32) + eps;
        let q_max_y = (cand_overall.max_y as f32) + eps;

        let mut stack = [0usize; 64];
        let mut stack_ptr = 1usize;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node = self.bvh_nodes[stack[stack_ptr]];
            if node.right_or_count == 0 {
                continue;
            }
            if (node.layer_mask & query_layer_mask) == 0 {
                continue;
            }
            if q_max_x < node.min_x
                || q_min_x > node.max_x
                || q_max_y < node.min_y
                || q_min_y > node.max_y
            {
                continue;
            }
            if node.right_or_count > 0 {
                let placed_idx = node.left_or_prim as usize;
                let entry = &self.footprints[placed_idx];
                if entry.reference == fp.reference {
                    continue;
                }
                if layer != entry.layer && cand_is_smd && entry.is_smd_only {
                    continue;
                }
                if !cand_overall.overlaps(&entry.overall_aabb) {
                    continue;
                }
                for (pa, pa_aabb) in cand_polys.iter().zip(cand_aabbs.iter()) {
                    for (pb, pb_aabb) in &entry.courtyards {
                        if pa_aabb.overlaps(pb_aabb) && pa.intersects_convex(pb) {
                            return true;
                        }
                    }
                }
            } else if stack_ptr + 2 <= stack.len() {
                stack[stack_ptr] = (-node.right_or_count) as usize;
                stack[stack_ptr + 1] = node.left_or_prim as usize;
                stack_ptr += 2;
            }
        }

        false
    }

    /// Evaluates whether placing `fp` at `(pos, rot, layer)` is legal using the precomputed
    /// 2D Morton LBVH over placed footprints and keepouts, with 100% exact SAT courtyard and boundary checks.
    ///
    /// Verifies that:
    /// 1. The component origin `pos`, all courtyard polygon vertices, and all pad corners lie inside `board`'s outline.
    /// 2. No courtyard polygon AABB overlaps any keepout zone in `self.keepout_aabbs`.
    /// 3. The placement-spaced courtyards do not collide with any indexed footprint via [`PlacedBoardSpatialIndex::collides_with_placed_courtyards`].
    ///
    /// # Arguments
    /// * `board` - Board providing the outer contour boundary test (`Board::contains_point_in_boundary`).
    /// * `fp` - Candidate footprint to test.
    /// * `pos` - Candidate world-space origin `(x, y)` in millimeters (`mm`).
    /// * `rot` - Candidate orientation angle in degrees.
    /// * `layer` - Candidate copper layer index (`0` for `F.Cu`, `1` for `B.Cu`).
    ///
    /// # Returns
    /// `true` if the pose is completely legal; `false` otherwise.
    pub fn is_placement_legal(
        &self,
        board: &Board,
        fp: &crate::board::Footprint,
        pos: Point,
        rot: f64,
        layer: usize,
    ) -> bool {
        if !board.contains_point_in_boundary(&pos) {
            return false;
        }

        let raw_polys = fp.transformed_courtyards_layered(pos, rot, layer, self.min_spacing_mm);
        for poly in &raw_polys {
            for v in &poly.vertices {
                if !board.contains_point_in_boundary(v) {
                    return false;
                }
            }
            if !self.keepout_aabbs.is_empty() {
                let pa = poly.aabb();
                for ko_aabb in &self.keepout_aabbs {
                    if pa.overlaps(ko_aabb) {
                        return false;
                    }
                }
            }
        }

        for pad in &fp.pads {
            for corner_pt in fp.pad_corners_at(pad, pos, rot, layer) {
                if !board.contains_point_in_boundary(&corner_pt) {
                    return false;
                }
            }
        }

        let cand_spacing = fp.placement_spacing(self.min_spacing_mm);
        if (cand_spacing - self.min_spacing_mm).abs() < 1e-9 {
            !self.collides_with_placed_courtyards(fp, layer, &raw_polys)
        } else {
            let cand_polys = fp.transformed_courtyards_layered(pos, rot, layer, cand_spacing);
            !self.collides_with_placed_courtyards(fp, layer, &cand_polys)
        }
    }

    /// Batch-evaluates candidate `(pos, rot, layer)` poses against the spatial index.
    ///
    /// # Arguments
    /// * `board` - Board providing outline containment checks.
    /// * `fp` - Candidate footprint being evaluated.
    /// * `candidates` - Slice of `(pos, rot, layer)` candidate poses.
    ///
    /// # Returns
    /// Vector of booleans corresponding 1-to-1 with `candidates`, where `true` indicates a legal pose.
    pub fn filter_legal_poses_batch(
        &self,
        board: &Board,
        fp: &crate::board::Footprint,
        candidates: &[(Point, f64, usize)],
    ) -> Vec<bool> {
        candidates
            .iter()
            .map(|&(pos, rot, layer)| self.is_placement_legal(board, fp, pos, rot, layer))
            .collect()
    }
}

/// Checks if placing `fp` at `pos` with `rot` and `layer` is legal (within board boundary, no keepout intersection, no courtyard collision).
///
/// Performs a standalone $O(N_f)$ legality check without requiring a prebuilt [`PlacedBoardSpatialIndex`].
/// For repeated queries across many candidates, prefer [`PlacedBoardSpatialIndex::is_placement_legal`].
///
/// # Arguments
/// * `board` - Board containing the outline, keepouts, and placed footprints.
/// * `fp` - Footprint to test.
/// * `pos` - Candidate world-space origin `(x, y)` in millimeters (`mm`).
/// * `rot` - Candidate rotation angle in degrees.
/// * `layer` - Candidate copper layer index (`0` for `F.Cu`, `1` for `B.Cu`).
/// * `min_spacing_mm` - Base courtyard clearance spacing in millimeters (`mm`).
///
/// # Returns
/// `true` if `fp` lies strictly inside the board outline, avoids keepouts, and does not overlap any other footprint.
pub fn is_placement_legal(
    board: &Board,
    fp: &crate::board::Footprint,
    pos: Point,
    rot: f64,
    layer: usize,
    min_spacing_mm: f64,
) -> bool {
    let cand_spacing = fp.placement_spacing(min_spacing_mm);
    let cand_polys = fp.transformed_courtyards_layered(pos, rot, layer, cand_spacing);

    // 1. Must be entirely inside the board outline (origin, courtyards, AND copper pads/legs)
    if !board.contains_point_in_boundary(&pos) {
        return false;
    }
    let raw_polys = fp.transformed_courtyards_layered(pos, rot, layer, min_spacing_mm);
    for poly in &raw_polys {
        for v in &poly.vertices {
            if !board.contains_point_in_boundary(v) {
                return false;
            }
        }
    }

    for pad in &fp.pads {
        for corner_pt in fp.pad_corners_at(pad, pos, rot, layer) {
            if !board.contains_point_in_boundary(&corner_pt) {
                return false;
            }
        }
    }

    // 2. Must not overlap any keepout zones
    for ko in &board.keepouts {
        let ko_aabb = AABB::new(ko.min_x, ko.min_y, ko.max_x, ko.max_y);
        for poly in &raw_polys {
            if poly.aabb().overlaps(&ko_aabb) {
                return false;
            }
        }
    }

    // 3. Must not collide with any already placed footprints (honoring escape clearances)
    for placed_fp in &board.footprints {
        if placed_fp.reference == fp.reference {
            continue;
        }
        // Two SMD-only footprints on opposite layers do not collide
        if layer != placed_fp.layer && fp.is_smd_only() && placed_fp.is_smd_only() {
            continue;
        }
        let placed_spacing = placed_fp.placement_spacing(min_spacing_mm);
        let placed_polys = placed_fp.transformed_courtyards_layered(
            Point::new(placed_fp.x, placed_fp.y),
            placed_fp.rotation,
            placed_fp.layer,
            placed_spacing,
        );
        for pa in &cand_polys {
            for pb in &placed_polys {
                if pa.intersects_convex(pb) {
                    return false;
                }
            }
        }
    }

    true
}

/// Shoves unlocked components that overlap with `placed_idx` away from it,
/// preserving legal courtyard spacing and boundary/keepout clearance.
///
/// # Algorithm
/// 1. Identifies all unlocked footprints on `board` whose courtyards intersect `board.footprints[placed_idx]`.
/// 2. Searches outward along 7 fan angles (`0..=±90°`) and 12 radial step distances (`0.5..=12.0` mm)
///    along the center-to-center repulsion vector using [`PlacedBoardSpatialIndex::is_placement_legal`].
/// 3. If any residual overlaps remain on the board after directional shoving, temporarily locks
///    `placed_idx` and runs [`legalize_placement`] to resolve cascading collisions.
/// 4. Calls `Board::rip_up_connected_traces_to_component_at` for every footprint whose pose changed
///    so stale copper traces connected to displaced pads are cleanly removed.
///
/// # Arguments
/// * `board` - Mutable [`Board`] containing the stationary footprint `placed_idx` and surrounding footprints.
/// * `placed_idx` - Index in `board.footprints` of the newly placed or dragged component.
/// * `min_spacing_mm` - Minimum courtyard clearance spacing in millimeters (`mm`).
///
/// # Returns
/// Number of unlocked footprints that were shoved or legalized.
///
/// # Side Effects
/// Mutates positions/rotations of colliding unlocked footprints in `board.footprints` and rips up
/// any routed copper traces connected to moved components.
pub fn shove_overlapping_components(
    board: &mut Board,
    placed_idx: usize,
    min_spacing_mm: f64,
) -> usize {
    if placed_idx >= board.footprints.len() {
        return 0;
    }

    let mut shoved_count = 0;
    let placed_fp = board.footprints[placed_idx].clone();
    let placed_pos = Point::new(placed_fp.x, placed_fp.y);
    let placed_polys = placed_fp.transformed_courtyards_layered(
        placed_pos,
        placed_fp.rotation,
        placed_fp.layer,
        min_spacing_mm,
    );

    // 1. Identify all colliding unlocked footprints
    let mut colliding_unlocked_indices = Vec::new();
    for (i, fp) in board.footprints.iter().enumerate() {
        if i == placed_idx || fp.is_locked {
            continue;
        }
        if placed_fp.layer != fp.layer && placed_fp.is_smd_only() && fp.is_smd_only() {
            continue;
        }
        let fp_polys = fp.transformed_courtyards_layered(
            Point::new(fp.x, fp.y),
            fp.rotation,
            fp.layer,
            min_spacing_mm,
        );
        let mut hit = false;
        'check: for pa in &placed_polys {
            for pb in &fp_polys {
                if pa.intersects_convex(pb) {
                    hit = true;
                    break 'check;
                }
            }
        }
        if hit {
            colliding_unlocked_indices.push(i);
        }
    }

    if colliding_unlocked_indices.is_empty() {
        return 0;
    }

    let orig_positions: Vec<(f64, f64, f64, usize)> = board
        .footprints
        .iter()
        .map(|fp| (fp.x, fp.y, fp.rotation, fp.layer))
        .collect();

    let mut spatial_index = PlacedBoardSpatialIndex::from_board(board, min_spacing_mm);

    // 2. Directional shove for each colliding unlocked component
    for &idx in &colliding_unlocked_indices {
        let other_fp = board.footprints[idx].clone();
        let dx = other_fp.x - placed_fp.x;
        let dy = other_fp.y - placed_fp.y;
        let dist = dx.hypot(dy);
        let (ux, uy) = if dist < 1e-4 {
            (1.0, 0.0)
        } else {
            (dx / dist, dy / dist)
        };

        let angles: [f64; 7] = [0.0, 0.35, -0.35, 0.785, -0.785, 1.5708, -1.5708];
        let distances: [f64; 12] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0];
        let mut best_pos = None;

        'search: for &d in &distances {
            for &a in &angles {
                let dir_x = ux * a.cos() - uy * a.sin();
                let dir_y = ux * a.sin() + uy * a.cos();
                let nx = other_fp.x + dir_x * d;
                let ny = other_fp.y + dir_y * d;

                let cand_pt = Point::new(nx, ny);
                if spatial_index.is_placement_legal(board, &other_fp, cand_pt, other_fp.rotation, other_fp.layer) {
                    best_pos = Some((nx, ny));
                    break 'search;
                }
            }
        }

        if let Some((nx, ny)) = best_pos {
            board.footprints[idx].x = nx;
            board.footprints[idx].y = ny;
            spatial_index.update_footprint(board, idx);
            shoved_count += 1;
        }
    }

    // 3. If any overlaps remain on the board, run quick simulated annealing legalization
    // while keeping the placed component locked so its position is preserved.
    let cfg = PlacementConfig {
        min_spacing_mm,
        duration_secs: 0,
        ..PlacementConfig::default()
    };
    let initial_state = PlacementState::from_board(board, &cfg);
    if initial_state.overlaps > 0 {
        let was_locked = board.footprints[placed_idx].is_locked;
        board.footprints[placed_idx].is_locked = true;
        let legalized = legalize_placement(board, &cfg, initial_state, None);
        legalized.apply_to_board(board);
        board.footprints[placed_idx].is_locked = was_locked;
        shoved_count = shoved_count.max(colliding_unlocked_indices.len());
    }

    // Rip up connected traces for any footprints that were shoved or legalized
    for (i, &orig_pos) in orig_positions.iter().enumerate() {
        if i == placed_idx || i >= board.footprints.len() {
            continue;
        }
        let fp = &board.footprints[i];
        let moved = (fp.x - orig_pos.0).abs() > 1e-4
            || (fp.y - orig_pos.1).abs() > 1e-4
            || (fp.rotation - orig_pos.2).abs() > 1e-4
            || fp.layer != orig_pos.3;
        if moved {
            board.rip_up_connected_traces_to_component_at(i, Some(orig_pos));
        }
    }

    shoved_count
}


/// Batch-evaluates candidate `(pos, rot, layer)` poses for `fp` on the GPU using [`VulkanEngine::evaluate_placement_batch`].
///
/// Packs rotated bounding boxes (`[min_x, min_y, max_x, max_y]`) and up to 16 target airwire
/// segments per candidate pose into GPU buffers and dispatches a single compute pass.
///
/// # Arguments
/// * `vk` - Active [`VulkanEngine`] instance.
/// * `board` - Board providing the outer boundary AABB.
/// * `fp` - Footprint whose candidate poses are being scored.
/// * `candidates` - Slice of `(pos, rot, layer)` candidate poses to evaluate.
/// * `targets` - Slice of `(pad_idx_on_fp, target_world_point, weight)` connections (capped at 16).
/// * `config` - Placement weights and spacing configuration.
///
/// # Returns
/// `Some(scores)` with one `f32` cost per candidate if `candidates` is non-empty and GPU dispatch succeeds, or `None`.
pub fn score_candidate_poses_gpu(
    vk: &VulkanEngine,
    board: &Board,
    fp: &crate::board::Footprint,
    candidates: &[(Point, f64, usize)],
    targets: &[(usize, Point, f64)],
    config: &PlacementConfig,
) -> Option<Vec<f32>> {
    if candidates.is_empty() {
        return None;
    }
    let nc = candidates.len();
    let ns = targets.len().min(16);
    let spacing = fp.effective_spacing(config);
    let base_polys = fp.transformed_courtyards(Point::new(0.0, 0.0), 0.0, spacing);
    let b0 = union_aabb(&base_polys, Point::new(0.0, 0.0));
    let hw = ((b0.max_x - b0.min_x) * 0.5).max(0.4) as f32;
    let hh = ((b0.max_y - b0.min_y) * 0.5).max(0.4) as f32;

    let mut boxes = Vec::with_capacity(nc);
    let mut segs = Vec::with_capacity(nc * ns);
    for &(pos, rot, layer) in candidates {
        let rad = (rot as f32).to_radians();
        let c = rad.cos().abs();
        let s = rad.sin().abs();
        let r_hw = hw * c + hh * s;
        let r_hh = hw * s + hh * c;
        let px = pos.x as f32;
        let py = pos.y as f32;
        boxes.push([px - r_hw, py - r_hh, px + r_hw, py + r_hh]);
        for &(pidx, target_pt, _) in targets.iter().take(ns) {
            let pad_pt = fp.pad_world_point_layered(pidx, pos, rot, layer);
            segs.push([
                pad_pt.x as f32,
                pad_pt.y as f32,
                target_pt.x as f32,
                target_pt.y as f32,
            ]);
        }
    }

    let boundary = board.boundary_aabb();
    let params = PlacementBatchParams {
        num_candidates: nc as u32,
        num_footprints: 1,
        num_segments: ns as u32,
        _pad: 0,
        board_min_x: boundary.min_x as f32,
        board_min_y: boundary.min_y as f32,
        board_max_x: boundary.max_x as f32,
        board_max_y: boundary.max_y as f32,
        wirelength_weight: config.wirelength_weight as f32,
        crossing_weight: config.crossing_weight as f32,
        overlap_penalty: config.overlap_penalty as f32,
        boundary_penalty: config.boundary_penalty as f32,
    };
    vk.evaluate_placement_batch(&boxes, &segs, &params)
}

/// Finds the most favourable location `(x, y)` in millimeters (`mm`), orientation (`rotation` in degrees),
/// and copper `layer` for `fp` on `board`.
///
/// Builds a one-off [`PlacedBoardSpatialIndex`] and delegates to [`find_most_favourable_placement_with_index`]:
/// - Decoupling capacitors are placed directly adjacent to their connected IC's non-ground power pin,
///   oriented with their power pad facing the IC pin.
/// - Differential pairs and connected passives/ICs are placed adjacent to their connected pins,
///   minimizing weighted Euclidean wirelength and crossings.
/// - Unconnected components are placed in the nearest legal, non-colliding location around the board center.
///
/// # Arguments
/// * `board` - Board containing existing placed footprints, outline, keepouts, and schematic metadata.
/// * `fp` - Footprint to find a placement pose for.
/// * `config` - Placement clearances, grid step, and optional Vulkan engine.
///
/// # Returns
/// Tuple `(x_mm, y_mm, rotation_deg, layer_idx)` representing the best legal pose.
pub fn find_most_favourable_placement(
    board: &Board,
    fp: &crate::board::Footprint,
    config: &PlacementConfig,
) -> (f64, f64, f64, usize) {
    let spatial_index = PlacedBoardSpatialIndex::from_board(board, config.min_spacing_mm);
    find_most_favourable_placement_with_index(board, fp, config, &spatial_index)
}

/// Finds the most favourable legal placement pose `(x_mm, y_mm, rotation_deg, layer_idx)` for `fp`
/// using a caller-supplied [`PlacedBoardSpatialIndex`].
///
/// Avoids rebuilding the 2D Morton LBVH when placing multiple components sequentially (such as in
/// [`place_all_holding_components_at_best_locations`]).
///
/// # Arguments
/// * `board` - Board containing existing placed footprints, nets, and schematic clusters.
/// * `fp` - Footprint to place.
/// * `config` - Placement configuration specifying grid step, spacing, and optional GPU scoring.
/// * `spatial_index` - Prebuilt [`PlacedBoardSpatialIndex`] over currently placed footprints.
///
/// # Returns
/// Tuple `(x_mm, y_mm, rotation_deg, layer_idx)` for the lowest-cost legal candidate pose, or a
/// boundary-clamped fallback pose if no candidate satisfies all constraints.
pub fn find_most_favourable_placement_with_index(
    board: &Board,
    fp: &crate::board::Footprint,
    config: &PlacementConfig,
    spatial_index: &PlacedBoardSpatialIndex,
) -> (f64, f64, f64, usize) {
    let boundary = board.boundary_aabb();
    let min_spacing = config.min_spacing_mm;
    let grid_step = config.grid_step_mm.max(0.1);
    let rotations = [0.0, 90.0, 180.0, 270.0];

    // Check if fp is a decoupling capacitor
    let is_cap = fp.reference.starts_with('C');
    let mut gnd_pad_idx = None;
    let mut pwr_pad_idx = None;
    for (pidx, p) in fp.pads.iter().enumerate() {
        if is_ground_net(&p.net_name) {
            gnd_pad_idx = Some(pidx);
        } else if is_power_net(&p.net_name) {
            pwr_pad_idx = Some(pidx);
        }
    }
    let is_decoupling = is_cap && gnd_pad_idx.is_some() && pwr_pad_idx.is_some();

    if is_decoupling {
        let pwr_idx = pwr_pad_idx.unwrap();
        let pwr_net = &fp.pads[pwr_idx].net_name;

        // Find candidate active devices on the board sharing this power net, ranked by schematic and cluster affinity
        let mut candidate_pins: Vec<(&crate::board::Footprint, usize, f64)> = Vec::new();
        let fp_sch_pos = board.schematic_positions.get(&fp.reference);
        let fp_cluster_idx = board.clusters.iter().position(|c| c.member_refs.contains(&fp.reference));

        for placed_dev in &board.footprints {
            let is_active = placed_dev.reference.starts_with('U')
                || placed_dev.reference.starts_with('Q')
                || placed_dev.reference.starts_with('J')
                || placed_dev.reference.starts_with('M');
            if !is_active {
                continue;
            }
            for (pad_idx, pad) in placed_dev.pads.iter().enumerate() {
                if pad.net_name == *pwr_net {
                    let pin_pos = placed_dev.pad_absolute_position(pad);
                    // Count how many caps are already within 3mm of this pin
                    let existing_caps = board
                        .footprints
                        .iter()
                        .filter(|f| f.reference.starts_with('C'))
                        .filter(|f| {
                            let (cx, cy) = (f.x, f.y);
                            (cx - pin_pos.0).hypot(cy - pin_pos.1) < 3.0
                        })
                        .count();

                    let mut score = 1000.0;
                    if let Some(pos) = fp_sch_pos {
                        if let Some(dev_pos) = board.schematic_positions.get(&placed_dev.reference) {
                            let sch_dist = (pos.0 - dev_pos.0).hypot(pos.1 - dev_pos.1);
                            score += sch_dist * 10.0;
                        }
                    }
                    if let Some(c_idx) = fp_cluster_idx {
                        if board.clusters[c_idx].member_refs.contains(&placed_dev.reference) {
                            score -= 800.0;
                        }
                    }
                    score += (existing_caps as f64) * 40.0;

                    candidate_pins.push((placed_dev, pad_idx, score));
                }
            }
        }

        candidate_pins.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        for (dev, pad_idx, _) in candidate_pins {
            let pin = &dev.pads[pad_idx];
            let (pin_x, pin_y) = dev.pad_absolute_position(pin);
            let pin_pt = Point::new(pin_x, pin_y);
            let dev_pos = Point::new(dev.x, dev.y);

            let dx = pin_x - dev_pos.x;
            let dy = pin_y - dev_pos.y;
            let len = dx.hypot(dy);
            let base_angle = if len > 1e-4 { dy.atan2(dx) } else { 0.0 };

            let dist_steps = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0];
            let angle_deltas = [
                0.0, 0.20, -0.20, 0.40, -0.40, 0.60, -0.60, 0.80, -0.80, 1.0, -1.0, 1.2, -1.2, 1.5, -1.5,
                2.0, -2.0, 2.5, -2.5, 3.0, -3.0, std::f64::consts::PI,
            ];

            let mut best_cand: Option<(f64, f64, f64, f64, usize)> = None;

            let mut legal_cands: Vec<(Point, f64, usize)> = Vec::new();
            for &d in &dist_steps {
                for &da in &angle_deltas {
                    let angle = base_angle + da;
                    for &rot in &rotations {
                        let (cx_min, cx_max, cy_min, cy_max) =
                            fp.origin_clamp_range(rot, fp.layer, min_spacing, &boundary);
                        let cx = snap_to_grid(
                            (pin_x + d * angle.cos()).clamp(cx_min, cx_max),
                            grid_step,
                        );
                        let cy = snap_to_grid(
                            (pin_y + d * angle.sin()).clamp(cy_min, cy_max),
                            grid_step,
                        );
                        let c_pos = Point::new(cx, cy);

                        if spatial_index.is_placement_legal(board, fp, c_pos, rot, fp.layer) {
                            legal_cands.push((c_pos, rot, fp.layer));
                        }
                    }
                }
            }

            let gpu_scores = config.vulkan.as_ref().and_then(|vk| {
                score_candidate_poses_gpu(
                    vk,
                    board,
                    fp,
                    &legal_cands,
                    &[(pwr_idx, pin_pt, 1.0)],
                    config,
                )
            });

            for (i, &(c_pos, rot, layer)) in legal_cands.iter().enumerate() {
                let pwr_pad_world = fp.pad_world_point_layered(pwr_idx, c_pos, rot, layer);
                let pad_dist = pwr_pad_world.distance_to(&pin_pt);
                let score = gpu_scores
                    .as_ref()
                    .and_then(|s| s.get(i).copied())
                    .map(|gs| (gs as f64) * 0.01 + pad_dist)
                    .unwrap_or(pad_dist);
                if best_cand.as_ref().map_or(true, |b| score < b.0) {
                    best_cand = Some((score, c_pos.x, c_pos.y, rot, layer));
                    if pad_dist <= 1.0 {
                        return (c_pos.x, c_pos.y, rot, layer);
                    }
                }
            }

            if let Some((_, bx, by, brot, blayer)) = best_cand {
                return (bx, by, brot, blayer);
            }
        }
    }

    // General case: connected component or other passives/ICs
    // Find all pins on placed components that share nets with this footprint
    let mut targets: Vec<(usize, Point, f64)> = Vec::new();
    for (pidx, pad) in fp.pads.iter().enumerate() {
        if pad.net_name.is_empty() {
            continue;
        }
        let is_gnd = is_ground_net(&pad.net_name);
        let is_pwr = is_power_net(&pad.net_name);
        let weight = if !is_gnd && !is_pwr { 5.0 } else if is_pwr { 2.0 } else { 0.5 };

        for placed_fp in &board.footprints {
            for other_pad in &placed_fp.pads {
                if other_pad.net_name == pad.net_name {
                    let (ox, oy) = placed_fp.pad_absolute_position(other_pad);
                    targets.push((pidx, Point::new(ox, oy), weight));
                }
            }
        }
    }

    let target_center = if !targets.is_empty() {
        let total_weight: f64 = targets.iter().map(|t| t.2).sum();
        let sum_x: f64 = targets.iter().map(|t| t.1.x * t.2).sum();
        let sum_y: f64 = targets.iter().map(|t| t.1.y * t.2).sum();
        Point::new(sum_x / total_weight, sum_y / total_weight)
    } else {
        Point::new((boundary.min_x + boundary.max_x) * 0.5, (boundary.min_y + boundary.max_y) * 0.5)
    };

    // Concentric search around target_center
    let radii = [
        0.0, 0.8, 1.5, 2.5, 3.5, 5.0, 6.5, 8.0, 10.0, 12.5, 15.0, 18.0, 22.0, 26.0, 30.0, 35.0, 42.0, 50.0,
    ];
    let mut best_cand: Option<(f64, f64, f64, f64, usize)> = None;

    let mut legal_cands: Vec<(Point, f64, usize, f64)> = Vec::new();
    for &r in &radii {
        let steps = if r == 0.0 {
            1
        } else {
            ((2.0 * std::f64::consts::PI * r / grid_step) as usize).clamp(8, 32)
        };

        for s in 0..steps {
            let theta = (s as f64) * (2.0 * std::f64::consts::PI / steps as f64);
            for &rot in &rotations {
                let (cx_min, cx_max, cy_min, cy_max) =
                    fp.origin_clamp_range(rot, fp.layer, min_spacing, &boundary);
                let cx = snap_to_grid(
                    (target_center.x + r * theta.cos()).clamp(cx_min, cx_max),
                    grid_step,
                );
                let cy = snap_to_grid(
                    (target_center.y + r * theta.sin()).clamp(cy_min, cy_max),
                    grid_step,
                );
                let c_pos = Point::new(cx, cy);

                if spatial_index.is_placement_legal(board, fp, c_pos, rot, fp.layer) {
                    legal_cands.push((c_pos, rot, fp.layer, r));
                }
            }
        }
        if r >= 6.0 && !legal_cands.is_empty() {
            break;
        }
    }

    let gpu_poses: Vec<(Point, f64, usize)> = legal_cands
        .iter()
        .map(|&(p, rot, layer, _)| (p, rot, layer))
        .collect();
    let gpu_scores = config.vulkan.as_ref().and_then(|vk| {
        score_candidate_poses_gpu(vk, board, fp, &gpu_poses, &targets, config)
    });

    for (i, &(c_pos, rot, layer, r)) in legal_cands.iter().enumerate() {
        let mut cost = 0.1 * r;
        for &(pidx, target_pt, w) in &targets {
            let pad_pt = fp.pad_world_point_layered(pidx, c_pos, rot, layer);
            cost += w * pad_pt.distance_to(&target_pt);
        }
        if let Some(gs) = gpu_scores.as_ref().and_then(|s| s.get(i).copied()) {
            cost += (gs as f64) * 0.01;
        }
        if best_cand.as_ref().map_or(true, |b| cost < b.0) {
            best_cand = Some((cost, c_pos.x, c_pos.y, rot, layer));
        }
    }

    if let Some((_, bx, by, brot, blayer)) = best_cand {
        return (bx, by, brot, blayer);
    }

    // Fallback: clamped within board outline with component's exact bounds
    let (cx_min, cx_max, cy_min, cy_max) = fp.origin_clamp_range(fp.rotation, fp.layer, min_spacing, &boundary);
    (
        fp.x.clamp(cx_min, cx_max),
        fp.y.clamp(cy_min, cy_max),
        fp.rotation,
        fp.layer,
    )
}

/// Places a single component from the holding palette at its most favourable location and orientation on the board.
///
/// Removes `board.holding_palette[idx]`, computes its optimal non-colliding pose via
/// [`find_most_favourable_placement`], appends the placed footprint to `board.footprints`, and returns it.
///
/// # Arguments
/// * `board` - Mutable [`Board`] containing the `holding_palette` and existing `footprints`.
/// * `idx` - Index into `board.holding_palette` of the component to place.
/// * `config` - Placement configuration specifying clearances, grid step, and optional GPU scoring.
///
/// # Returns
/// `Some(placed_footprint)` if `idx < board.holding_palette.len()`, or `None` if `idx` is out of bounds.
///
/// # Side Effects
/// Removes the item at `idx` from `board.holding_palette` and pushes it onto `board.footprints`.
pub fn place_holding_component_at_best_location(
    board: &mut Board,
    idx: usize,
    config: &PlacementConfig,
) -> Option<crate::board::Footprint> {
    if idx >= board.holding_palette.len() {
        return None;
    }
    let mut fp = board.holding_palette.remove(idx);
    let (bx, by, brot, blayer) = find_most_favourable_placement(board, &fp, config);
    fp.x = bx;
    fp.y = by;
    fp.rotation = brot;
    fp.layer = blayer;
    board.footprints.push(fp.clone());
    Some(fp)
}

/// Places all components in the holding palette at their most favourable locations and orientations on the board.
///
/// Drains `board.holding_palette`, sorts the unplaced footprints by electrical priority
/// (`0` = differential pairs, `1` = 2-pin decoupling capacitors, `2` = active ICs/transistors `U*`/`Q*`,
/// `3` = general passives), and places each component sequentially using an incrementally updated
/// [`PlacedBoardSpatialIndex`] via [`find_most_favourable_placement_with_index`].
///
/// # Arguments
/// * `board` - Mutable [`Board`] whose `holding_palette` is drained into `board.footprints`.
/// * `config` - Placement configuration specifying clearances, grid step, and optional GPU scoring.
///
/// # Returns
/// Total number of footprints moved from `board.holding_palette` onto `board.footprints`.
///
/// # Side Effects
/// Empties `board.holding_palette` and appends all placed footprints to `board.footprints`.
pub fn place_all_holding_components_at_best_locations(
    board: &mut Board,
    config: &PlacementConfig,
) -> usize {
    if board.holding_palette.is_empty() {
        return 0;
    }
    let mut palette = std::mem::take(&mut board.holding_palette);
    // Sort by priority: differential pairs first, then decoupling caps, then active ICs, then passives
    palette.sort_by(|a, b| {
        let prio = |fp: &crate::board::Footprint| -> usize {
            let is_diff = fp.pads.iter().any(|p| {
                let u = p.net_name.to_ascii_uppercase();
                u.contains("USB") || u.contains("DP") || u.contains("DM") || u.contains("D+") || u.contains("D-")
            });
            if is_diff {
                return 0;
            }
            let has_gnd = fp.pads.iter().any(|p| is_ground_net(&p.net_name));
            let has_pwr = fp.pads.iter().any(|p| is_power_net(&p.net_name));
            if fp.reference.starts_with('C') && has_gnd && has_pwr {
                return 1;
            }
            if fp.reference.starts_with('U') || fp.reference.starts_with('Q') {
                return 2;
            }
            3
        };
        prio(a).cmp(&prio(b))
    });

    let mut spatial_index = PlacedBoardSpatialIndex::from_board(board, config.min_spacing_mm);
    let count = palette.len();
    for mut fp in palette {
        let (bx, by, brot, blayer) = find_most_favourable_placement_with_index(board, &fp, config, &spatial_index);
        fp.x = bx;
        fp.y = by;
        fp.rotation = brot;
        fp.layer = blayer;
        spatial_index.push_footprint(&fp);
        board.footprints.push(fp);
    }
    count
}

/// Precomputes functional component clusters for rigid group moves (e.g. an active IC together with
/// its tied decoupling capacitors and termination resistors).
///
/// For each footprint $i \in [0, N_f)$, builds a list of movable footprint indices that belong to
/// its local functional group based on [`PassiveTie`] links where both `anchor_idx` and `passive_idx`
/// are unlocked (`movable`).
///
/// # Arguments
/// * `board` - Board containing all footprints.
/// * `ties` - Slice of [`PassiveTie`] affinity relationships.
/// * `movable` - Slice of unlocked footprint indices.
///
/// # Returns
/// Vector of length `board.footprints.len()`, where `clusters[i]` contains the footprint indices in
/// component `i`'s rigid move group.
pub fn build_component_clusters(
    board: &Board,
    ties: &[PassiveTie],
    movable: &[usize],
) -> Vec<Vec<usize>> {
    let mut clusters: Vec<Vec<usize>> = (0..board.footprints.len()).map(|i| vec![i]).collect();
    for tie in ties {
        if movable.contains(&tie.anchor_idx) && movable.contains(&tie.passive_idx) {
            if !clusters[tie.anchor_idx].contains(&tie.passive_idx) {
                clusters[tie.anchor_idx].push(tie.passive_idx);
            }
        }
    }
    for tie in ties {
        if movable.contains(&tie.anchor_idx) && movable.contains(&tie.passive_idx) {
            let anchor_cluster = clusters[tie.anchor_idx].clone();
            clusters[tie.passive_idx] = anchor_cluster;
        }
    }
    clusters
}

/// Rigidly translates an entire cluster of components by `(dx, dy)` in millimeters (`mm`) if and only if
/// all cluster members remain strictly within their legal board origin clamp ranges.
///
/// # Arguments
/// * `state` - Mutable [`PlacementState`] to update if the group translation is valid.
/// * `board` - Board providing footprint geometry and clamp bounds.
/// * `config` - Placement configuration specifying `min_spacing_mm`.
/// * `cluster` - Slice of footprint indices belonging to the cluster (from [`build_component_clusters`]).
/// * `dx` - Horizontal translation delta in millimeters (`mm`).
/// * `dy` - Vertical translation delta in millimeters (`mm`).
/// * `boundary` - Outer board [`AABB`] used by `Footprint::origin_clamp_range`.
///
/// # Returns
/// `true` if all members fit within the board boundary and `state` was updated; `false` if `cluster`
/// is empty, the translation is zero, or any member would violate its clamp range (leaving `state` unmodified).
///
/// # Side Effects
/// When returning `true`, translates `state.positions[m]` and refreshes cached geometry via
/// `state.update_component(m, board, config)` for every `m` in `cluster`.
pub fn apply_cluster_group_move(
    state: &mut PlacementState,
    board: &Board,
    config: &PlacementConfig,
    cluster: &[usize],
    dx: f64,
    dy: f64,
    boundary: &AABB,
) -> bool {
    if cluster.is_empty() || (dx.abs() <= 1e-4 && dy.abs() <= 1e-4) {
        return false;
    }
    let all_fit = cluster.iter().all(|&m| {
        let (cx_min, cx_max, cy_min, cy_max) = board.footprints[m]
            .origin_clamp_range(state.rotations[m], state.layers[m], config.min_spacing_mm, boundary);
        let nx = state.positions[m].x + dx;
        let ny = state.positions[m].y + dy;
        nx >= cx_min && nx <= cx_max && ny >= cy_min && ny <= cy_max
    });
    if all_fit {
        for &m in cluster {
            state.positions[m].x += dx;
            state.positions[m].y += dy;
            state.update_component(m, board, config);
        }
        true
    } else {
        false
    }
}

/// Performs Metropolis-Hastings replica exchange between adjacent temperature rungs in Parallel Tempering.
///
/// Alternates between even `(0, 1), (2, 3), ...` and odd `(1, 2), (3, 4), ...` adjacent rung pairs
/// based on `epoch % 2`, swapping their [`PlacementState`] configurations with probability:
///
/// $$P_{\text{swap}}(i, i+1) = \min\left(1, \exp\big((\beta_i - \beta_{i+1})(E_i - E_{i+1})\big)\right)$$
///
/// where $\beta_k = 1 / T_k$ is the inverse temperature of rung $k$ and $E_k$ is `replicas[k].cost`.
///
/// # Arguments
/// * `replicas` - Mutable slice of [`PlacementState`]s ordered from coldest rung (`0`) to hottest rung (`num_threads - 1`).
/// * `num_threads` - Total number of temperature rungs in the geometric ladder.
/// * `epoch` - Current epoch index (determines even vs. odd pairing parity).
/// * `progress` - Normalized elapsed time in `[0.0, 1.0]` used to compute the cooling schedule.
/// * `temp_scale` - Global temperature scaling factor from [`PlacementConfig::temp_scale`].
/// * `thermal_boost` - Temporary reheating multiplier (`3.5` during stagnation thermal pulses, `1.0` otherwise).
/// * `rng` - Random number generator used for Metropolis acceptance rolls.
///
/// # Returns
/// Number of adjacent replica swaps accepted during this exchange pass.
///
/// # Side Effects
/// Swaps elements in `replicas` in-place and advances `rng`.
pub fn metropolis_replica_exchange(
    replicas: &mut [PlacementState],
    num_threads: usize,
    epoch: usize,
    progress: f64,
    temp_scale: f64,
    thermal_boost: f64,
    rng: &mut impl rand::Rng,
) -> usize {
    if replicas.len() < 2 {
        return 0;
    }
    let cooling = (1.0 - progress * 0.85).max(0.05);
    let start_idx = epoch % 2;
    let mut accepted_swaps = 0;
    for i in (start_idx..replicas.len().saturating_sub(1)).step_by(2) {
        let j = i + 1;
        let t_i = 8.0 * (350.0f64 / 8.0).powf((i as f64) / ((num_threads - 1).max(1) as f64)) * temp_scale * cooling * thermal_boost;
        let t_j = 8.0 * (350.0f64 / 8.0).powf((j as f64) / ((num_threads - 1).max(1) as f64)) * temp_scale * cooling * thermal_boost;
        let beta_i = 1.0 / t_i.max(1e-4);
        let beta_j = 1.0 / t_j.max(1e-4);
        let delta_beta = beta_i - beta_j; // beta_i > beta_j since rung i has lower temp than rung j
        let delta_energy = replicas[i].cost - replicas[j].cost;
        let log_p = delta_beta * delta_energy;
        let swap = if log_p >= 0.0 {
            true
        } else {
            rng.gen::<f64>() < log_p.exp()
        };
        if swap {
            replicas.swap(i, j);
            accepted_swaps += 1;
        }
    }
    accepted_swaps
}

/// Executes the complete hybrid Vulkan GPU + multi-core CPU Parallel Tempering Simulated Annealing
/// PCB auto-placement pipeline on `board`.
///
/// # Pipeline Stages
/// 1. **Holding Palette Placement:** Places any unplaced components from `board.holding_palette`
///    via [`place_all_holding_components_at_best_locations`].
/// 2. **Initial Prioritization & Legalization:** Compares the legalized existing board state against
///    [`prioritize_and_position_initial_components`] + [`legalize_placement`], retaining whichever
///    achieves lower cost and fewer overlaps.
/// 3. **Pre-Annealing Pin/Gate Swapping:** Calls [`optimize_pin_and_gate_swapping`] to untangle
///    initial ratsnest crossings before annealing.
/// 4. **Parallel Tempering & Vulkan GPU Co-Execution:**
///    - Spawns background threads for each available [`VulkanEngine`] device to continuously run
///      `run_gpu_placement_batch` and inject verified improvements into `global_best`.
///    - Runs `num_threads` CPU replicas across a geometric temperature ladder ($T \in [8.0, 350.0] \times \text{temp\_scale}$),
///      applying 7 move operators per step, performing [`metropolis_replica_exchange`] at epoch boundaries,
///      and triggering thermal pulses + legal basin hops upon stagnation.
///    - Streams live footprint deltas over KiCad IPC (`config.ipc`) and updates GUI telemetry (`live`).
/// 5. **Final Containment, Pin/Gate Swap & Legalization Pass:** Verifies strict board outline containment
///    for all unlocked footprints, runs a final [`optimize_pin_and_gate_swapping`] pass, re-legalizes if
///    needed, applies the winning [`PlacementState`] to `board`, and commits the final IPC batch.
///
/// # Arguments
/// * `board` - Mutable [`Board`] to auto-place in-place.
/// * `config` - Placement hyperparameters, clearance rules, duration, and optional Vulkan/IPC handles.
/// * `live` - Optional shared [`LiveTrialState`] for real-time GUI canvas, ratsnest, and sparkline telemetry.
/// * `stop_flag` - Shared atomic flag allowing early user cancellation from the GUI or CLI signal handler.
///
/// # Returns
/// The best evaluated [`PlacementState`] discovered and applied to `board`.
///
/// # Side Effects
/// Mutates `board.footprints`, `board.holding_palette`, and pad net assignments (via pin/gate swapping),
/// streams live pose updates to `live` and `config.ipc`, and dispatches compute work to `config.vulkan`.
pub fn run_auto_placer(
    board: &mut Board,
    config: &PlacementConfig,
    live: Option<Arc<Mutex<LiveTrialState>>>,
    stop_flag: Arc<AtomicBool>,
) -> PlacementState {
    // 1. Place any components from holding palette at favourable locations
    if !board.holding_palette.is_empty() {
        let count = place_all_holding_components_at_best_locations(board, config);
        if let Some(ref st) = live {
            if let Ok(mut g) = st.lock() {
                g.push_log(format!("[Auto-Placer] Placed {} components from holding palette", count));
            }
        }
    }

    // 2. Prioritize initial placement if it improves upon the existing board placement
    let current_st = PlacementState::from_board(board, config);
    let current_legal = legalize_placement(board, config, current_st, live.as_ref());

    let mut prioritized_board = board.clone();
    prioritize_and_position_initial_components(&mut prioritized_board, config, live.as_ref());
    let prioritized_st = PlacementState::from_board(&prioritized_board, config);
    let prioritized_legal = legalize_placement(&prioritized_board, config, prioritized_st, live.as_ref());

    let legalized = if prioritized_legal.cost < current_legal.cost && prioritized_legal.overlaps <= current_legal.overlaps {
        *board = prioritized_board;
        prioritized_legal
    } else {
        current_legal
    };

    // 3. Optimize pin and gate swapping on initial placement
    let initial_swaps = optimize_pin_and_gate_swapping(board, config);
    if initial_swaps > 0 {
        if let Some(ref st) = live {
            if let Ok(mut g) = st.lock() {
                g.push_log(format!("[Auto-Placer] Initial pin/gate swapping untangled {} connections", initial_swaps));
                g.refresh_ratsnest();
            }
        }
    }

    let movable: Vec<usize> = board
        .footprints
        .iter()
        .enumerate()
        .filter(|(_, fp)| !fp.is_locked)
        .map(|(i, _)| i)
        .collect();

    if movable.is_empty() || config.duration_secs == 0 {
        legalized.apply_to_board(board);
        return legalized;
    }

    let active_vulkan = config.vulkan.clone();
    let num_threads = rayon::current_num_threads().max(2);
    let global_best = Arc::new(Mutex::new(legalized.clone()));
    let total_evals = Arc::new(AtomicUsize::new(0));
    let start_time = Instant::now();
    let total_dur = Duration::from_secs(config.duration_secs);
    let boundary = board.boundary_aabb();
    let ties = Arc::new(build_passive_ties(board));

    // Precompute connected foreign pad references for barycentric centroid pull moves.
    // Exclude plane ground and wide distribution nets so components are not pulled into central IC ground pads.
    let mut connected_pins: Vec<Vec<(usize, usize)>> = vec![Vec::new(); board.footprints.len()];
    for (net_id, net_name) in board.nets.iter().enumerate() {
        if net_id == 0 || is_ground_net(net_name) {
            continue;
        }
        let mut net_pads: Vec<(usize, usize)> = Vec::new();
        for (f_idx, fp) in board.footprints.iter().enumerate() {
            for (p_idx, p) in fp.pads.iter().enumerate() {
                if p.net_id == net_id {
                    net_pads.push((f_idx, p_idx));
                }
            }
        }
        if net_pads.len() > 8 {
            continue;
        }
        for &(fa, _) in &net_pads {
            for &(fb, pb) in &net_pads {
                if fa != fb && !connected_pins[fa].contains(&(fb, pb)) {
                    connected_pins[fa].push((fb, pb));
                }
            }
        }
    }
    let connected_pins = Arc::new(connected_pins);
    let component_clusters = Arc::new(build_component_clusters(board, &ties, &movable));

    if let Some(ref st) = live {
        if let Ok(mut g) = st.lock() {
            g.phase = TrialPhase::AutoPlacing;
            let gpu_tag = if let Some(ref vk) = active_vulkan {
                format!(" + Vulkan GPU ({})", vk.device_name)
            } else {
                String::new()
            };
            g.push_log(format!(
                "[Auto-Placer] Launching {} Parallel Tempering CPU SA replicas{} for {}s (jump_r={:.1}mm, swap={:.0}%, centroid={:.0}%)...",
                num_threads, gpu_tag, config.duration_secs,
                config.max_jump_radius_mm, config.pair_swap_prob * 100.0, config.centroid_pull_prob * 100.0
            ));
        }
    }

    let is_ipc = config.ipc.as_ref().map_or(false, |l| l.lock().map_or(false, |c| c.is_connected()));
    let (num_epochs, epoch_dur, stagnant_threshold, thermal_pulse_reset) = if is_ipc {
        ((config.duration_secs * 60).max(1) as usize, Duration::from_millis(16), 60usize, 45usize)
    } else {
        ((config.duration_secs * 4).max(1) as usize, Duration::from_millis(250), 4usize, 3usize)
    };
    let mut replicas: Vec<PlacementState> = (0..num_threads)
        .map(|tid| {
            if tid == 0 {
                legalized.clone()
            } else {
                let mut rep = legalized.clone();
                // Higher rungs start from the legal state and optionally apply a small number of
                // compatible passive pair swaps to diversify topological exploration without overlaps.
                if tid > 0 && movable.len() >= 2 {
                    let mut init_rng = StdRng::seed_from_u64(0xD15EA5Eu64.wrapping_mul((tid as u64) + 1));
                    let swaps = (tid.min(4)).min(movable.len() / 2);
                    for _ in 0..swaps {
                        let a = movable[init_rng.gen_range(0..movable.len())];
                        let fp_a = &board.footprints[a];
                        let compat: Vec<usize> = movable.iter().copied()
                            .filter(|&m| m != a && {
                                let fp_m = &board.footprints[m];
                                (fp_m.width - fp_a.width).abs() <= 1.0 && (fp_m.height - fp_a.height).abs() <= 1.0
                            })
                            .collect();
                        if !compat.is_empty() {
                            let b = compat[init_rng.gen_range(0..compat.len())];
                            let pos_a = rep.positions[a];
                            let pos_b = rep.positions[b];
                            rep.positions[a] = pos_b;
                            rep.positions[b] = pos_a;
                            rep.update_component(a, board, config);
                            rep.update_component(b, board, config);
                        }
                    }
                    rep.evaluate(board, config, &ties);
                    // If initial swaps created any overlap or boundary violation, revert cleanly to legalized
                    if rep.overlaps > legalized.overlaps || rep.boundary_violations > 0 {
                        rep = legalized.clone();
                    }
                }
                rep
            }
        })
        .collect();
    let gpu_done = AtomicBool::new(false);

    std::thread::scope(|scope| {
        if let Some(ref vk) = active_vulkan {
            let (fp_descs, edges, courtyards) = build_gpu_sa_descriptors(board, config, &ties, &legalized);
            let fp_descs = Arc::new(fp_descs);
            let edges = Arc::new(edges);
            let courtyards = Arc::new(courtyards);
            for (gpu_idx, vk_dev) in vk.all_engines().into_iter().enumerate() {
                let board_ref = &*board;
                let ties_ref = Arc::clone(&ties);
                let movable_ref = &movable;
                let global_best_ref = Arc::clone(&global_best);
                let total_evals_ref = Arc::clone(&total_evals);
                let stop_ref = Arc::clone(&stop_flag);
                let gpu_done_ref = &gpu_done;
                let fp_descs_ref = Arc::clone(&fp_descs);
                let edges_ref = Arc::clone(&edges);
                let courtyards_ref = Arc::clone(&courtyards);

                scope.spawn(move || {
                    let mut batch_idx = (gpu_idx as u64).wrapping_mul(10_000);
                    while !gpu_done_ref.load(Ordering::Relaxed)
                        && !stop_ref.load(Ordering::Relaxed)
                        && start_time.elapsed() + Duration::from_millis(25) < total_dur
                    {
                        let progress = (start_time.elapsed().as_secs_f64() / total_dur.as_secs_f64()).clamp(0.0, 1.0);
                        let cooling = (1.0 - progress * 0.85).max(0.05);
                        let base = global_best_ref.lock().unwrap().clone();
                        if let Some((cand, gpu_evals)) = run_gpu_placement_batch(
                            &vk_dev,
                            board_ref,
                            config,
                            &ties_ref,
                            movable_ref,
                            &base,
                            &fp_descs_ref,
                            &edges_ref,
                            &courtyards_ref,
                            batch_idx,
                            cooling,
                        ) {
                            total_evals_ref.fetch_add(gpu_evals, Ordering::Relaxed);
                            if let Ok(mut g) = global_best_ref.lock() {
                                let is_better = if cand.boundary_violations != g.boundary_violations {
                                    cand.boundary_violations < g.boundary_violations
                                } else if cand.overlaps != g.overlaps {
                                    cand.overlaps < g.overlaps
                                } else {
                                    cand.cost < g.cost
                                };
                                if is_better {
                                    *g = cand;
                                }
                            }
                        }
                        batch_idx = batch_idx.wrapping_add(1);
                    }
                });
            }
        }

        let mut last_best_cost = f64::MAX;
        let mut stagnant_epochs = 0usize;
        let mut thermal_pulse_epochs = 0usize;

        for epoch in 0..num_epochs {
            if stop_flag.load(Ordering::Relaxed) || start_time.elapsed() >= total_dur {
                break;
            }

            let cur_best_cost = global_best.lock().unwrap().cost;
            if cur_best_cost < last_best_cost - (last_best_cost.abs() * 0.005) {
                last_best_cost = cur_best_cost;
                stagnant_epochs = 0;
            } else {
                stagnant_epochs += 1;
            }

            if stagnant_epochs >= stagnant_threshold {
                thermal_pulse_epochs = thermal_pulse_reset;
                stagnant_epochs = 0;
                if let Some(ref st) = live {
                    if let Ok(mut g) = st.lock() {
                        g.push_log(format!(
                            "[Auto-Placer] Stagnation detected at cost {:.1}: triggering thermal pulse & legal basin hop...",
                            cur_best_cost
                        ));
                    }
                }
                // Basin Hop: reset hottest replica to global_best and apply a few compatible pair swaps
                if let Some(hottest) = replicas.last_mut() {
                    let best = global_best.lock().unwrap().clone();
                    *hottest = best;
                    let mut hop_rng = StdRng::seed_from_u64(0xBA510_709u64.wrapping_add(epoch as u64));
                    let swaps = 3.min(movable.len() / 2);
                    for _ in 0..swaps {
                        let a = movable[hop_rng.gen_range(0..movable.len())];
                        let fp_a = &board.footprints[a];
                        let compat: Vec<usize> = movable.iter().copied()
                            .filter(|&m| m != a && {
                                let fp_m = &board.footprints[m];
                                (fp_m.width - fp_a.width).abs() <= 1.0 && (fp_m.height - fp_a.height).abs() <= 1.0
                            })
                            .collect();
                        if !compat.is_empty() {
                            let b = compat[hop_rng.gen_range(0..compat.len())];
                            let pos_a = hottest.positions[a];
                            let pos_b = hottest.positions[b];
                            hottest.positions[a] = pos_b;
                            hottest.positions[b] = pos_a;
                            hottest.update_component(a, board, config);
                            hottest.update_component(b, board, config);
                        }
                    }
                    hottest.evaluate(board, config, &ties);
                    if hottest.overlaps > 0 || hottest.boundary_violations > 0 {
                        *hottest = global_best.lock().unwrap().clone();
                    }
                }
            }

            let thermal_boost = if thermal_pulse_epochs > 0 {
                thermal_pulse_epochs -= 1;
                3.5
            } else {
                1.0
            };

            let progress = (start_time.elapsed().as_secs_f64() / total_dur.as_secs_f64()).clamp(0.0, 1.0);
            let ep_end = (Instant::now() + epoch_dur).min(start_time + total_dur);

            replicas
                .par_iter_mut()
                .enumerate()
                .for_each(|(tid, current)| {
                    let mut rng = StdRng::seed_from_u64(
                        0x9E3779B97F4A7C15u64
                            .wrapping_mul((tid as u64) + 1)
                            .wrapping_add(epoch as u64),
                    );
                    let temp_ratio = (tid as f64) / ((num_threads - 1).max(1) as f64);
                    let t_base = 8.0 * (350.0f64 / 8.0).powf(temp_ratio) * config.temp_scale;
                    let cooling = (1.0 - progress * 0.85).max(0.05);
                    let temp = t_base * cooling * thermal_boost;
                    let mut local_evals = 0usize;

                    let p_swap = (config.pair_swap_prob * 100.0).clamp(0.0, 30.0) as usize;
                    let p_centroid = (config.centroid_pull_prob * 100.0).clamp(0.0, 40.0) as usize;
                    let p_cluster = 15usize;
                    let p_rot = 16usize;
                    let p_flip = if config.allow_layer_flip { 8usize } else { 0usize };
                    let p_affinity = if !ties.is_empty() { 12usize } else { 0usize };

                    while Instant::now() < ep_end && !stop_flag.load(Ordering::Relaxed) {
                        for _ in 0..40 {
                            let old_positions = current.positions.clone();
                            let old_rotations = current.rotations.clone();
                            let old_layers = current.layers.clone();
                            let old_cost = current.cost;
                            let old_crossings = current.crossings;
                            let old_cc = current.corridor_congestion;
                            let old_fa = current.flow_alignment_penalty;
                            let old_wl = current.wirelength;
                            let old_ov = current.overlaps;
                            let old_bv = current.boundary_violations;
                            let mut modified: Vec<usize> = Vec::with_capacity(8);

                            let roll = rng.gen_range(0..100);
                            let t1 = p_swap;
                            let t2 = t1 + p_centroid;
                            let t3 = t2 + p_cluster;
                            let t4 = t3 + p_rot;
                            let t5 = t4 + p_flip;
                            let t6 = t5 + p_affinity;

                            if roll < t1 && movable.len() >= 2 {
                                // 1. Component Pair Swap
                                let a = movable[rng.gen_range(0..movable.len())];
                                let fp_a = &board.footprints[a];
                                let pos_a = current.positions[a];
                                let b = if rng.gen_bool(0.75) {
                                    let compatible: Vec<usize> = movable
                                        .iter()
                                        .copied()
                                        .filter(|&m| {
                                            if m == a {
                                                return false;
                                            }
                                            let fp_m = &board.footprints[m];
                                            let dim_match = (fp_m.width - fp_a.width).abs() <= 1.2
                                                && (fp_m.height - fp_a.height).abs() <= 1.2;
                                            let nearby = current.positions[m].distance_to(&pos_a) <= 15.0;
                                            dim_match || nearby
                                        })
                                        .collect();
                                    if !compatible.is_empty() {
                                        compatible[rng.gen_range(0..compatible.len())]
                                    } else {
                                        movable[rng.gen_range(0..movable.len())]
                                    }
                                } else {
                                    movable[rng.gen_range(0..movable.len())]
                                };

                                if a != b {
                                    let rot_a = current.rotations[a];
                                    let rot_b = current.rotations[b];
                                    let layer_a = current.layers[a];
                                    let layer_b = current.layers[b];
                                    let (a_min_x, a_max_x, a_min_y, a_max_y) = board.footprints[a]
                                        .origin_clamp_range(rot_b, layer_b, config.min_spacing_mm, &boundary);
                                    let (b_min_x, b_max_x, b_min_y, b_max_y) = board.footprints[b]
                                        .origin_clamp_range(rot_a, layer_a, config.min_spacing_mm, &boundary);
                                    let pos_b = current.positions[b];
                                    current.positions[a] = Point::new(pos_b.x.clamp(a_min_x, a_max_x), pos_b.y.clamp(a_min_y, a_max_y));
                                    current.positions[b] = Point::new(pos_a.x.clamp(b_min_x, b_max_x), pos_a.y.clamp(b_min_y, b_max_y));
                                    current.rotations[a] = rot_b;
                                    current.rotations[b] = rot_a;
                                    current.layers[a] = layer_b;
                                    current.layers[b] = layer_a;
                                    modified.push(a);
                                    modified.push(b);
                                }
                            } else if roll < t2 {
                                // 2. Barycentric Centroid Pull (magnetic pull towards connected nets)
                                let idx = movable[rng.gen_range(0..movable.len())];
                                let pins = &connected_pins[idx];
                                if !pins.is_empty() {
                                    let mut sum_x = 0.0;
                                    let mut sum_y = 0.0;
                                    for &(other_fp, other_pad) in pins {
                                        let pt = current.cached_pads[other_fp][other_pad];
                                        sum_x += pt.x;
                                        sum_y += pt.y;
                                    }
                                    let cx = sum_x / pins.len() as f64;
                                    let cy = sum_y / pins.len() as f64;
                                    let p_curr = current.positions[idx];
                                    let vec_x = cx - p_curr.x;
                                    let vec_y = cy - p_curr.y;
                                    let dist = vec_x.hypot(vec_y);
                                    if dist > 0.1 {
                                        let max_step = (3.5 * cooling).max(config.grid_step_mm * 1.5);
                                        let step = rng.gen_range(0.5 * config.grid_step_mm..=max_step).min(dist * 0.4);
                                        let jitter = (1.5 * cooling).max(config.grid_step_mm * 0.5);
                                        let target_x = p_curr.x + (vec_x / dist) * step + rng.gen_range(-jitter..=jitter);
                                        let target_y = p_curr.y + (vec_y / dist) * step + rng.gen_range(-jitter..=jitter);

                                        let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx]
                                            .origin_clamp_range(current.rotations[idx], current.layers[idx], config.min_spacing_mm, &boundary);
                                        let nx = snap_to_grid(target_x.clamp(cx_min, cx_max), config.grid_step_mm);
                                        let ny = snap_to_grid(target_y.clamp(cy_min, cy_max), config.grid_step_mm);
                                        current.positions[idx] = Point::new(nx, ny);
                                        modified.push(idx);
                                    }
                                }
                            } else if roll < t3 {
                                // 3. Cluster Rigid Group Move (move IC and tied decoupling passives together)
                                let idx = movable[rng.gen_range(0..movable.len())];
                                let cluster = &component_clusters[idx];
                                if cluster.len() > 1 {
                                    let max_shift = (config.max_jump_radius_mm * cooling).max(config.grid_step_mm * 2.0);
                                    let (shift_x, shift_y) = if rng.gen_bool(0.08) {
                                        (
                                            rng.gen_range(-config.max_jump_radius_mm..=config.max_jump_radius_mm),
                                            rng.gen_range(-config.max_jump_radius_mm..=config.max_jump_radius_mm),
                                        )
                                    } else {
                                        (
                                            rng.gen_range(-max_shift..=max_shift),
                                            rng.gen_range(-max_shift..=max_shift),
                                        )
                                    };
                                    let dx = snap_to_grid(shift_x, config.grid_step_mm);
                                    let dy = snap_to_grid(shift_y, config.grid_step_mm);
                                    if dx.abs() > 1e-4 || dy.abs() > 1e-4 {
                                        let all_fit = cluster.iter().all(|&m| {
                                            let (cx_min, cx_max, cy_min, cy_max) = board.footprints[m]
                                                .origin_clamp_range(current.rotations[m], current.layers[m], config.min_spacing_mm, &boundary);
                                            let nx = current.positions[m].x + dx;
                                            let ny = current.positions[m].y + dy;
                                            nx >= cx_min && nx <= cx_max && ny >= cy_min && ny <= cy_max
                                        });
                                        if all_fit {
                                            for &m in cluster {
                                                current.positions[m].x += dx;
                                                current.positions[m].y += dy;
                                                modified.push(m);
                                            }
                                        }
                                    }
                                } else {
                                    // Single component shift fallback
                                    let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx]
                                        .origin_clamp_range(current.rotations[idx], current.layers[idx], config.min_spacing_mm, &boundary);
                                    let max_shift = (config.max_jump_radius_mm * cooling).max(config.grid_step_mm * 2.0);
                                    let (shift_x, shift_y) = if rng.gen_bool(0.06) {
                                        (
                                            rng.gen_range(-config.max_jump_radius_mm..=config.max_jump_radius_mm),
                                            rng.gen_range(-config.max_jump_radius_mm..=config.max_jump_radius_mm),
                                        )
                                    } else {
                                        (
                                            rng.gen_range(-max_shift..=max_shift),
                                            rng.gen_range(-max_shift..=max_shift),
                                        )
                                    };
                                    let nx = snap_to_grid((current.positions[idx].x + shift_x).clamp(cx_min, cx_max), config.grid_step_mm);
                                    let ny = snap_to_grid((current.positions[idx].y + shift_y).clamp(cy_min, cy_max), config.grid_step_mm);
                                    current.positions[idx] = Point::new(nx, ny);
                                    modified.push(idx);
                                }
                            } else if roll < t4 {
                                // 4. Rotation (90°, 180°, 270°)
                                let idx = movable[rng.gen_range(0..movable.len())];
                                // 180° rotation preserves bounding box extents and never introduces overlaps for rectangular passives!
                                let delta = if rng.gen_bool(0.60) {
                                    180.0
                                } else {
                                    [90.0, 270.0][rng.gen_range(0..2)]
                                };
                                let new_rot = (current.rotations[idx] + delta) % 360.0;
                                current.rotations[idx] = new_rot;
                                let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx]
                                    .origin_clamp_range(new_rot, current.layers[idx], config.min_spacing_mm, &boundary);
                                current.positions[idx].x = current.positions[idx].x.clamp(cx_min, cx_max);
                                current.positions[idx].y = current.positions[idx].y.clamp(cy_min, cy_max);
                                modified.push(idx);
                            } else if roll < t5 && config.allow_layer_flip {
                                // 5. Layer Flip (F.Cu <-> B.Cu)
                                let idx = movable[rng.gen_range(0..movable.len())];
                                if board.footprints[idx].is_smd_only() {
                                    let new_layer = 1 - current.layers[idx];
                                    current.layers[idx] = new_layer;
                                    let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx]
                                        .origin_clamp_range(current.rotations[idx], new_layer, config.min_spacing_mm, &boundary);
                                    current.positions[idx].x = current.positions[idx].x.clamp(cx_min, cx_max);
                                    current.positions[idx].y = current.positions[idx].y.clamp(cy_min, cy_max);
                                    modified.push(idx);
                                }
                            } else if roll < t6 && !ties.is_empty() {
                                // 6. Decoupling & Passive Tie Affinity Move
                                let tie = &ties[rng.gen_range(0..ties.len())];
                                let idx = tie.passive_idx;
                                let anchor_pt = current.cached_pads[tie.anchor_idx][tie.anchor_pad_idx];
                                let angles = [0.0, 90.0, 180.0, 270.0];
                                let new_rot = angles[rng.gen_range(0..4)];
                                current.rotations[idx] = new_rot;
                                let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx]
                                    .origin_clamp_range(new_rot, current.layers[idx], config.min_spacing_mm, &boundary);

                                if tie.is_decoupling {
                                    let anchor_center = current.positions[tie.anchor_idx];
                                    let outward_angle = (anchor_pt.y - anchor_center.y)
                                        .atan2(anchor_pt.x - anchor_center.x);
                                    let theta = outward_angle + rng.gen_range(-1.2..=1.2);
                                    let anchor_escape = if config.enable_escape_clearance {
                                        board.footprints[tie.anchor_idx].escape_margin * config.escape_margin_mult
                                    } else {
                                        0.0
                                    };
                                    let min_r = 1.1 + anchor_escape;
                                    let max_r = 2.2 + anchor_escape * 1.5;
                                    let r = rng.gen_range(min_r..=max_r);
                                    let nx = snap_to_grid(
                                        (anchor_pt.x + r * theta.cos())
                                            .clamp(cx_min, cx_max),
                                        config.grid_step_mm,
                                    );
                                    let ny = snap_to_grid(
                                        (anchor_pt.y + r * theta.sin())
                                            .clamp(cy_min, cy_max),
                                        config.grid_step_mm,
                                    );
                                    current.positions[idx] = Point::new(nx, ny);
                                    modified.push(idx);
                                } else {
                                    let nx = snap_to_grid(
                                        (anchor_pt.x + rng.gen_range(-2.5..=2.5))
                                            .clamp(cx_min, cx_max),
                                        config.grid_step_mm,
                                    );
                                    let ny = snap_to_grid(
                                        (anchor_pt.y + rng.gen_range(-2.5..=2.5))
                                            .clamp(cy_min, cy_max),
                                        config.grid_step_mm,
                                    );
                                    current.positions[idx] = Point::new(nx, ny);
                                    modified.push(idx);
                                }
                            } else {
                                // 7. Positional Shift & Long-Range Jumps
                                let idx = movable[rng.gen_range(0..movable.len())];
                                let (cx_min, cx_max, cy_min, cy_max) = board.footprints[idx]
                                    .origin_clamp_range(current.rotations[idx], current.layers[idx], config.min_spacing_mm, &boundary);
                                let (shift_x, shift_y) = if rng.gen_bool(0.04) {
                                    // 4% chance of macro teleport jump across full max_jump_radius
                                    (
                                        rng.gen_range(-config.max_jump_radius_mm..=config.max_jump_radius_mm),
                                        rng.gen_range(-config.max_jump_radius_mm..=config.max_jump_radius_mm),
                                    )
                                } else if rng.gen_bool(0.40) {
                                    // 40% chance of small fine-tuning grid nudge (1-2 grid steps)
                                    let step = config.grid_step_mm;
                                    let dx = [-step, 0.0, step][rng.gen_range(0..3)];
                                    let dy = [-step, 0.0, step][rng.gen_range(0..3)];
                                    (dx, dy)
                                } else {
                                    let max_shift = (config.max_jump_radius_mm * cooling * 0.5).max(config.grid_step_mm * 2.0);
                                    (
                                        rng.gen_range(-max_shift..=max_shift),
                                        rng.gen_range(-max_shift..=max_shift),
                                    )
                                };
                                let nx = snap_to_grid(
                                    (current.positions[idx].x + shift_x).clamp(cx_min, cx_max),
                                    config.grid_step_mm,
                                );
                                let ny = snap_to_grid(
                                    (current.positions[idx].y + shift_y).clamp(cy_min, cy_max),
                                    config.grid_step_mm,
                                );
                                current.positions[idx] = Point::new(nx, ny);
                                modified.push(idx);
                            }

                            if modified.is_empty() {
                                continue;
                            }
                            for &m in &modified {
                                current.update_component(m, board, config);
                            }
                            current.evaluate(board, config, &ties);
                            local_evals += 1;

                            let delta = current.cost - old_cost;
                            let accept = if current.boundary_violations > old_bv {
                                false
                            } else if delta <= 0.0 {
                                true
                            } else {
                                current.boundary_violations == 0 && rng.gen::<f64>() < (-delta / temp).exp()
                            };

                            if accept {
                                if let Ok(mut g) = global_best.lock() {
                                    let is_better = if current.boundary_violations != g.boundary_violations {
                                        current.boundary_violations < g.boundary_violations
                                    } else if current.overlaps != g.overlaps {
                                        current.overlaps < g.overlaps
                                    } else {
                                        current.cost < g.cost
                                    };
                                    if is_better {
                                        *g = current.clone();
                                    }
                                }
                            } else {
                                current.positions = old_positions;
                                current.rotations = old_rotations;
                                current.layers = old_layers;
                                for &m in &modified {
                                    current.update_component(m, board, config);
                                }
                                current.cost = old_cost;
                                current.crossings = old_crossings;
                                current.corridor_congestion = old_cc;
                                current.flow_alignment_penalty = old_fa;
                                current.wirelength = old_wl;
                                current.overlaps = old_ov;
                                current.boundary_violations = old_bv;
                            }
                        }
                    }
                    total_evals.fetch_add(local_evals, Ordering::Relaxed);
                });

            // Parallel Tempering Metropolis-Hastings Replica Exchange:
            let mut exchange_rng = StdRng::seed_from_u64(0x5EA7_E7C4u64.wrapping_add(epoch as u64));
            metropolis_replica_exchange(
                &mut replicas,
                num_threads,
                epoch,
                progress,
                config.temp_scale,
                thermal_boost,
                &mut exchange_rng,
            );

            // Sync global_best if any replica discovered a lower cost or fewer violations
            {
                let mut g = global_best.lock().unwrap();
                for rep in &replicas {
                    let is_better = if rep.boundary_violations != g.boundary_violations {
                        rep.boundary_violations < g.boundary_violations
                    } else if rep.overlaps != g.overlaps {
                        rep.overlaps < g.overlaps
                    } else {
                        rep.cost < g.cost
                    };
                    if is_better {
                        *g = rep.clone();
                    }
                }
            }
            let best_snap = global_best.lock().unwrap().clone();

            if let Some(ref ipc_lock) = config.ipc {
                if let Ok(mut client) = ipc_lock.lock() {
                    if client.is_connected() {
                        client.stream_footprint_deltas(
                            &board.footprints,
                            &best_snap.positions,
                            &best_snap.rotations,
                        );
                    }
                }
            }

            let elapsed = start_time.elapsed().as_secs_f64().max(1e-3);
            let rate = total_evals.load(Ordering::Relaxed) as f64 / elapsed;

            if let Some(ref st) = live {
                if let Ok(mut g) = st.lock() {
                    best_snap.apply_to_board(&mut g.board);
                    g.ratsnest = g.board.generate_ratsnest_segments(&best_snap.cached_pads);
                    g.crossings = best_snap.crossings;
                    g.wirelength_mm = best_snap.wirelength;
                    g.overlaps = best_snap.overlaps;
                    g.evals_per_sec = rate;
                    g.progress = (progress as f32) * 0.5;
                    g.status_text = format!(
                        "Auto-Placing: Crossings={} | Wirelength={:.1}mm | {:.0} evals/s",
                        best_snap.crossings, best_snap.wirelength, rate
                    );
                }
            }
        }

        gpu_done.store(true, Ordering::Relaxed);
    });

    let final_rate = total_evals.load(Ordering::Relaxed) as f64 / start_time.elapsed().as_secs_f64().max(1e-3);
    let mut final_best = global_best.lock().unwrap().clone();

    // Final containment verification pass:
    // Guarantee that every component's origin, courtyard vertices, and pad corners stay strictly inside board outline.
    let boundary = board.boundary_aabb();
    let min_spacing = config.min_spacing_mm;
    let board_center = Point::new(
        (boundary.min_x + boundary.max_x) * 0.5,
        (boundary.min_y + boundary.max_y) * 0.5,
    );

    for (i, fp) in board.footprints.iter().enumerate() {
        if fp.is_locked {
            continue;
        }
        let rot = final_best.rotations[i];
        let layer = final_best.layers[i];
        let (cx_min, cx_max, cy_min, cy_max) = fp.origin_clamp_range(rot, layer, min_spacing, &boundary);
        let mut pos = final_best.positions[i];
        let clamped_x = pos.x.clamp(cx_min, cx_max);
        let clamped_y = pos.y.clamp(cy_min, cy_max);
        if (clamped_x - pos.x).abs() > 1e-4 || (clamped_y - pos.y).abs() > 1e-4 {
            pos = Point::new(clamped_x, clamped_y);
            final_best.positions[i] = pos;
            final_best.update_component(i, board, config);
        }

        // Nudge towards center if any pad or courtyard vertex still violates board contour (e.g. rounded corners)
        let is_contained = {
            let mut ok = board.contains_point_in_boundary(&pos);
            if ok {
                let polys = fp.transformed_courtyards_layered(pos, rot, layer, min_spacing);
                for poly in &polys {
                    for v in &poly.vertices {
                        if !board.contains_point_in_boundary(v) {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        break;
                    }
                }
            }
            if ok {
                for pad in &fp.pads {
                    for corner_pt in fp.pad_corners_at(pad, pos, rot, layer) {
                        if !board.contains_point_in_boundary(&corner_pt) {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        break;
                    }
                }
            }
            ok
        };

        if !is_contained {
            let dx = board_center.x - pos.x;
            let dy = board_center.y - pos.y;
            let dist = dx.hypot(dy).max(1e-4);
            let ux = dx / dist;
            let uy = dy / dist;

            let mut step = 0.25;
            let mut resolved = false;
            while step <= 15.0 {
                let cand_pos = Point::new(
                    snap_to_grid(pos.x + ux * step, config.grid_step_mm),
                    snap_to_grid(pos.y + uy * step, config.grid_step_mm),
                );
                let cand_contained = {
                    let mut ok = board.contains_point_in_boundary(&cand_pos);
                    if ok {
                        let polys = fp.transformed_courtyards_layered(cand_pos, rot, layer, min_spacing);
                        for poly in &polys {
                            for v in &poly.vertices {
                                if !board.contains_point_in_boundary(v) {
                                    ok = false;
                                    break;
                                }
                            }
                            if !ok {
                                break;
                            }
                        }
                    }
                    if ok {
                        for pad in &fp.pads {
                            for corner_pt in fp.pad_corners_at(pad, cand_pos, rot, layer) {
                                if !board.contains_point_in_boundary(&corner_pt) {
                                    ok = false;
                                    break;
                                }
                            }
                            if !ok {
                                break;
                            }
                        }
                    }
                    ok
                };
                if cand_contained {
                    final_best.positions[i] = cand_pos;
                    final_best.update_component(i, board, config);
                    resolved = true;
                    break;
                }
                step += 0.25;
            }
            if !resolved {
                // ponytail: In extreme geometries fallback to clamped boundary
                final_best.positions[i] = Point::new(clamped_x, clamped_y);
                final_best.update_component(i, board, config);
            }
        }
    }

    final_best.apply_to_board(board);
    let final_swaps = optimize_pin_and_gate_swapping(board, config);
    if final_swaps > 0 {
        final_best = PlacementState::from_board(board, config);
        if let Some(ref st) = live {
            if let Ok(mut g) = st.lock() {
                g.push_log(format!("[Auto-Placer] Final pin/gate swapping untangled {} connections", final_swaps));
            }
        }
    }

    final_best.evaluate(board, config, &ties);
    if final_best.overlaps > 0 || final_best.boundary_violations > 0 {
        final_best = legalize_placement(board, config, final_best, live.as_ref());
    }
    final_best.apply_to_board(board);
    if let Some(ref st) = live {
        if let Ok(mut g) = st.lock() {
            final_best.apply_to_board(&mut g.board);
            g.ratsnest = g.board.generate_ratsnest_segments(&final_best.cached_pads);
            g.crossings = final_best.crossings;
            g.wirelength_mm = final_best.wirelength;
            g.overlaps = final_best.overlaps;
            g.evals_per_sec = final_rate;
            g.push_log(format!(
                "[Auto-Placer] Finished: Crossings={}, Wirelength={:.1}mm, Overlaps={} ({:.0} evals/s)",
                final_best.crossings, final_best.wirelength, final_best.overlaps, final_rate
            ));
        }
    }

    if let Some(ref ipc_lock) = config.ipc {
        if let Ok(mut client) = ipc_lock.lock() {
            if client.is_connected() {
                for fp in &board.footprints {
                    client.queue_move_footprint(&fp.reference, fp.x, fp.y, fp.rotation);
                }
                let _ = client.commit_batch();
            }
        }
    }

    final_best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{CompiledCluster, Footprint, NetClass, Pad};
    use std::collections::HashSet;
    use std::fs;

    fn make_test_fp(reference: &str, x: f64, y: f64, is_locked: bool, pads: Vec<Pad>) -> Footprint {
        Footprint {
            reference: reference.to_string(),
            value: "VAL".to_string(),
            x,
            y,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked,
            pads,
            width: 2.0,
            height: 2.0,
            courtyards: vec![Polygon::new(vec![
                Point::new(-1.0, -1.0),
                Point::new(1.0, -1.0),
                Point::new(1.0, 1.0),
                Point::new(-1.0, 1.0),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        }
    }

    fn make_synthetic_board() -> Board {
        let nets = vec![
            "".to_string(),
            "GND".to_string(),
            "+3.3V".to_string(),
            "SIG_A".to_string(),
            "SIG_B".to_string(),
        ];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        for n in &nets {
            net_to_class.insert(n.clone(), "Default".to_string());
        }

        let u1 = make_test_fp(
            "U1",
            140.0,
            100.0,
            true,
            vec![
                Pad::new("1", -0.5, 0.0, 0.5, 0.5, 3, "SIG_A", false),
                Pad::new("2", 0.5, 0.0, 0.5, 0.5, 4, "SIG_B", false),
            ],
        );

        let j3 = make_test_fp(
            "J3",
            155.0,
            100.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 4, "SIG_B", true)],
        );

        // R10 overlaps C10 initially at (145.0, 100.0)
        let r10 = make_test_fp(
            "R10",
            145.0,
            100.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.5, 0.5, 3, "SIG_A", false),
                Pad::new("2", 0.5, 0.0, 0.5, 0.5, 4, "SIG_B", false),
            ],
        );

        let c10 = make_test_fp(
            "C10",
            145.2,
            100.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.5, 0.5, 4, "SIG_B", false),
                Pad::new("2", 0.5, 0.0, 0.5, 0.5, 1, "GND", false),
            ],
        );

        Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 125.0,
            min_y: 85.0,
            max_x: 170.0,
            max_y: 120.0,
            footprints: vec![u1, j3, r10, c10],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: vec![
                CompiledCluster {
                    name: "ANCHORED".to_string(),
                    anchor_idx: Some(0),
                    member_indices: vec![0, 2],
                    member_refs: vec!["U1".to_string(), "C1".to_string()],
                    max_radius: 2.0,
                    weight: 100.0,
                },
                CompiledCluster {
                    name: "CENTROID".to_string(),
                    anchor_idx: None,
                    member_indices: vec![2, 3],
                    member_refs: vec!["C1".to_string(), "R12".to_string()],
                    max_radius: 10.0,
                    weight: 60.0,
                },
            ],
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        }
    }

    #[test]
    fn test_union_aabb_and_snap_to_grid() {
        let fb = union_aabb(&[], Point::new(5.0, 10.0));
        assert_eq!(fb, AABB::new(4.5, 9.5, 5.5, 10.5));

        let p = Polygon::new(vec![Point::new(1.0, 2.0), Point::new(4.0, 6.0)]);
        assert_eq!(union_aabb(&[p], Point::new(0.0, 0.0)), AABB::new(1.0, 2.0, 4.0, 6.0));

        assert!((snap_to_grid(1.23, 0.0) - 1.23).abs() < 1e-6);
        assert!((snap_to_grid(1.23, -1.0) - 1.23).abs() < 1e-6);
        assert!((snap_to_grid(1.30, 0.5) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_build_passive_ties_and_evaluation() {
        let mut board = make_synthetic_board();
        let cfg = PlacementConfig::default();
        let ties = build_passive_ties(&board);
        // R10 and C10 are unlocked passives; U1 has priority 3
        assert_eq!(ties.len(), 2);
        assert!(ties.iter().all(|t| t.anchor_idx == 0));

        let mut st = PlacementState::from_board(&board, &cfg);
        assert_eq!(st.overlaps, 1);

        // Flip C10 to B.Cu (layer = 1) at the exact same XY position: SMD parts on opposite sides must not overlap
        st.layers[3] = 1;
        st.update_component(3, &board, &cfg);
        st.evaluate(&board, &cfg, &ties);
        assert_eq!(st.overlaps, 0, "SMD footprints on opposite layers (F.Cu vs B.Cu) must not collide");

        // Move C10 out of bounds to exercise boundary_cost and far passive affinity
        st.positions[3] = Point::new(100.0, 50.0);
        st.update_component(3, &board, &cfg);
        st.evaluate(&board, &cfg, &ties);
        assert_eq!(st.overlaps, 0);
        assert!(st.cost > cfg.boundary_penalty);

        st.apply_to_board(&mut board);
        assert!((board.footprints[3].x - 100.0).abs() < 1e-6);
        assert_eq!(board.footprints[3].layer, 1);
        // Locked U1 must not move even if state position changed
        assert!((board.footprints[0].x - 140.0).abs() < 1e-6);
    }

    #[test]
    fn test_legalize_placement_and_run_auto_placer() {
        let mut board = make_synthetic_board();
        let vk = VulkanEngine::new();
        let cfg = PlacementConfig {
            duration_secs: 1,
            allow_layer_flip: true,
            vulkan: vk.clone(),
            ..PlacementConfig::default()
        };
        let live = Arc::new(Mutex::new(LiveTrialState::new(board.clone())));
        let initial = PlacementState::from_board(&board, &cfg);
        assert_eq!(initial.overlaps, 1);

        let legalized = legalize_placement(&board, &cfg, initial, Some(&live));
        assert_eq!(legalized.overlaps, 0);
        // Calling legalize_placement when overlaps == 0 returns immediately
        let already_legal = legalize_placement(&board, &cfg, legalized.clone(), None);
        assert_eq!(already_legal.overlaps, 0);

        // Calling legalize_placement when all footprints are locked returns immediately
        let mut all_locked = board.clone();
        for fp in &mut all_locked.footprints {
            fp.is_locked = true;
        }
        let mut locked_state = PlacementState::from_board(&all_locked, &cfg);
        locked_state.overlaps = 1;
        let locked_res = legalize_placement(&all_locked, &cfg, locked_state, None);
        assert_eq!(locked_res.overlaps, 1);

        // Test duration_secs == 0 fast path
        let cfg_zero = PlacementConfig {
            duration_secs: 0,
            ..PlacementConfig::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let st_zero = run_auto_placer(&mut board.clone(), &cfg_zero, None, Arc::clone(&stop));
        assert_eq!(st_zero.overlaps, 0);

        // Test full parallel tempering SA + Vulkan GPU batch auto-placer epoch with live telemetry
        let final_st = run_auto_placer(&mut board, &cfg, Some(Arc::clone(&live)), stop);
        assert_eq!(final_st.overlaps, 0);
        if let Some(ref v) = vk {
            assert!(v.compute_busy_ns.load(Ordering::Relaxed) > 0);
        }
        let g = live.lock().unwrap();
        assert_eq!(g.overlaps, 0);
        assert!(!g.log_lines.is_empty());
        let _ = HashSet::<String>::new();
    }

    #[test]
    fn test_is_ground_and_power_net() {
        assert!(is_ground_net("GND"));
        assert!(is_ground_net("/GND"));
        assert!(is_ground_net("AGND"));
        assert!(is_ground_net("DGND"));
        assert!(is_ground_net("PGND"));
        assert!(is_ground_net("GND_EARTH"));
        assert!(!is_ground_net("+3.3V"));
        assert!(!is_ground_net(""));

        assert!(is_power_net("+3.3V"));
        assert!(is_power_net("+5V"));
        assert!(is_power_net("+12V"));
        assert!(is_power_net("VCC_5V"));
        assert!(is_power_net("VRAW"));
        assert!(is_power_net("VBUS"));
        assert!(is_power_net("/BUCK_VIN"));
        assert!(is_power_net("VDD"));
        assert!(is_power_net("VBAT"));
        assert!(is_power_net("VIN"));
        assert!(is_power_net("3V3_OUT"));
        assert!(!is_power_net("GND"));
        assert!(!is_power_net("SIG_A"));
        assert!(!is_power_net("USB_DP"));
        assert!(!is_power_net(""));
    }

    #[test]
    fn test_decoupling_cap_identification_and_power_pin_targeting() {
        let nets = vec![
            "".to_string(),
            "GND".to_string(),
            "+3.3V".to_string(),
            "SIG_A".to_string(),
            "SIG_B".to_string(),
        ];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        for n in &nets {
            net_to_class.insert(n.clone(), "Default".to_string());
        }

        // Active IC U1 at (100, 100) with Pin 1 (+3.3V power), Pin 2 (GND), Pin 3 (SIG_A)
        let u1 = make_test_fp(
            "U1",
            100.0,
            100.0,
            true,
            vec![
                Pad::new("1", -1.0, 0.0, 0.5, 0.5, 2, "+3.3V", false),
                Pad::new("2", 0.0, -1.0, 0.5, 0.5, 1, "GND", false),
                Pad::new("3", 1.0, 0.0, 0.5, 0.5, 3, "SIG_A", false),
            ],
        );

        // Active IC U2 at (180, 100) with Pin 1 (+3.3V power), Pin 2 (GND), Pin 3 (SIG_B)
        let u2 = make_test_fp(
            "U2",
            180.0,
            100.0,
            true,
            vec![
                Pad::new("1", -1.0, 0.0, 0.5, 0.5, 2, "+3.3V", false),
                Pad::new("2", 0.0, -1.0, 0.5, 0.5, 1, "GND", false),
                Pad::new("3", 1.0, 0.0, 0.5, 0.5, 4, "SIG_B", false),
            ],
        );

        // Decoupling cap C1 for U1: Pin 1 (+3.3V), Pin 2 (GND)
        let c1 = make_test_fp(
            "C1",
            102.0,
            100.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.4, 0.4, 2, "+3.3V", false),
                Pad::new("2", 0.5, 0.0, 0.4, 0.4, 1, "GND", false),
            ],
        );

        // Decoupling cap C2 for U2: Pin 1 (+3.3V), Pin 2 (GND)
        let c2 = make_test_fp(
            "C2",
            182.0,
            100.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.4, 0.4, 2, "+3.3V", false),
                Pad::new("2", 0.5, 0.0, 0.4, 0.4, 1, "GND", false),
            ],
        );

        let mut schematic_positions = HashMap::new();
        schematic_positions.insert("U1".to_string(), (10.0, 10.0));
        schematic_positions.insert("C1".to_string(), (12.0, 10.0)); // Close to U1
        schematic_positions.insert("U2".to_string(), (100.0, 10.0));
        schematic_positions.insert("C2".to_string(), (102.0, 10.0)); // Close to U2

        let board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 50.0,
            min_y: 50.0,
            max_x: 220.0,
            max_y: 150.0,
            footprints: vec![u1, u2, c1, c2],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions,
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let ties = build_passive_ties(&board);
        assert_eq!(ties.len(), 2, "Both C1 and C2 must be tied as decoupling capacitors");

        let tie_c1 = ties.iter().find(|t| t.passive_idx == 2).expect("C1 tie must exist");
        assert!(tie_c1.is_decoupling);
        assert_eq!(tie_c1.anchor_idx, 0, "C1 must bind to U1 based on schematic proximity");
        assert_eq!(tie_c1.anchor_pad_idx, 0, "C1 must bind directly to U1's +3.3V power pad (pad 1)");
        assert_eq!(tie_c1.passive_pad_idx, 0, "C1's power pad (+3.3V) must be the passive pad");
        assert_eq!(tie_c1.max_dist, 2.0);

        let tie_c2 = ties.iter().find(|t| t.passive_idx == 3).expect("C2 tie must exist");
        assert!(tie_c2.is_decoupling);
        assert_eq!(tie_c2.anchor_idx, 1, "C2 must bind to U2 based on schematic proximity");
        assert_eq!(tie_c2.anchor_pad_idx, 0, "C2 must bind directly to U2's +3.3V power pad (pad 1)");
        assert_eq!(tie_c2.passive_pad_idx, 0, "C2's power pad (+3.3V) must be the passive pad");
    }

    #[test]
    fn test_decoupling_cap_multi_power_pin_distribution() {
        let nets = vec![
            "".to_string(),
            "GND".to_string(),
            "+3.3V".to_string(),
        ];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        for n in &nets {
            net_to_class.insert(n.clone(), "Default".to_string());
        }

        // Active IC U1 with TWO separate +3.3V power pins: pad 1 and pad 4
        let u1 = make_test_fp(
            "U1",
            100.0,
            100.0,
            true,
            vec![
                Pad::new("1", -2.0, 0.0, 0.5, 0.5, 2, "+3.3V", false),
                Pad::new("2", 0.0, -2.0, 0.5, 0.5, 1, "GND", false),
                Pad::new("3", 0.0, 2.0, 0.5, 0.5, 1, "GND", false),
                Pad::new("4", 2.0, 0.0, 0.5, 0.5, 2, "+3.3V", false),
            ],
        );

        let c1 = make_test_fp(
            "C1",
            97.0,
            100.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.4, 0.4, 2, "+3.3V", false),
                Pad::new("2", 0.5, 0.0, 0.4, 0.4, 1, "GND", false),
            ],
        );

        let c2 = make_test_fp(
            "C2",
            103.0,
            100.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.4, 0.4, 2, "+3.3V", false),
                Pad::new("2", 0.5, 0.0, 0.4, 0.4, 1, "GND", false),
            ],
        );

        let board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 50.0,
            min_y: 50.0,
            max_x: 150.0,
            max_y: 150.0,
            footprints: vec![u1, c1, c2],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let ties = build_passive_ties(&board);
        assert_eq!(ties.len(), 2);
        // Both caps must bind to U1 (index 0)
        assert_eq!(ties[0].anchor_idx, 0);
        assert_eq!(ties[1].anchor_idx, 0);
        // The two caps should be distributed across both power pads (pad 0 and pad 3)
        let anchor_pads: HashSet<usize> = ties.iter().map(|t| t.anchor_pad_idx).collect();
        assert_eq!(anchor_pads.len(), 2, "Decoupling caps must distribute across both power pins of U1");
        assert!(anchor_pads.contains(&0) && anchor_pads.contains(&3));
    }

    #[test]
    fn test_decoupling_cap_affinity_cost_steepness() {
        let nets = vec![
            "".to_string(),
            "GND".to_string(),
            "+3.3V".to_string(),
        ];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        for n in &nets {
            net_to_class.insert(n.clone(), "Default".to_string());
        }

        let u1 = make_test_fp(
            "U1",
            100.0,
            100.0,
            true,
            vec![
                Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "+3.3V", false),
                Pad::new("2", 1.0, 0.0, 0.5, 0.5, 1, "GND", false),
            ],
        );

        let c1 = make_test_fp(
            "C1",
            102.5,
            100.0,
            false,
            vec![
                Pad::new("1", -0.8, 0.0, 0.4, 0.4, 2, "+3.3V", false),
                Pad::new("2", 0.8, 0.0, 0.4, 0.4, 1, "GND", false),
            ],
        );

        let board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 50.0,
            min_y: 50.0,
            max_x: 150.0,
            max_y: 150.0,
            footprints: vec![u1, c1],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let cfg = PlacementConfig::default();
        let ties = build_passive_ties(&board);
        assert_eq!(ties.len(), 1);
        assert!(ties[0].is_decoupling);

        let mut st_close = PlacementState::from_board(&board, &cfg);
        // Position C1 so pad 1 is at 102.5 - 0.8 = 101.7 (dist = 1.7mm <= decoupling_max_dist 2.0mm, no courtyard overlap)
        st_close.positions[1] = Point::new(102.5, 100.0);
        st_close.update_component(1, &board, &cfg);
        st_close.evaluate(&board, &cfg, &ties);
        assert_eq!(st_close.overlaps, 0);
        let close_cost = st_close.cost;

        let mut st_far = PlacementState::from_board(&board, &cfg);
        // Position C1 so pad 1 is at 107.0 - 0.8 = 106.2 (dist = 6.2mm > decoupling_max_dist 2.0mm)
        st_far.positions[1] = Point::new(107.0, 100.0);
        st_far.update_component(1, &board, &cfg);
        st_far.evaluate(&board, &cfg, &ties);
        assert_eq!(st_far.overlaps, 0);
        let far_cost = st_far.cost;

        assert!(
            far_cost > close_cost + 150.0,
            "Drifting away from power pin must incur strong penalty: far_cost {} vs close_cost {}",
            far_cost, close_cost
        );
    }

    #[test]
    fn test_p1p2esp_decoupling_caps_binding_if_present() {
        let pcb_path = "/home/deece/src/p1p2esp/p1p2esp.kicad_pcb";
        if !std::path::Path::new(pcb_path).exists() {
            return;
        }

        let mut board = Board::load(pcb_path, &HashSet::new()).expect("Failed to load p1p2esp PCB");
        let cfg = PlacementConfig::default();
        prioritize_and_position_initial_components(&mut board, &cfg, None);
        let initial = PlacementState::from_board(&board, &cfg);
        let legalized = legalize_placement(&board, &cfg, initial, None);
        assert_eq!(legalized.overlaps, 0, "p1p2esp placement must be completely legalized with 0 courtyard overlaps");
        assert_eq!(legalized.boundary_violations, 0, "p1p2esp placement must have 0 boundary violations");
        let ties = build_passive_ties(&board);

        let fp_index = |ref_des: &str| -> usize {
            board.footprints.iter().position(|f| f.reference == ref_des).unwrap()
        };

        let u1_idx = fp_index("U1");
        let u2_idx = fp_index("U2");
        let u4_idx = fp_index("U4");

        // Verify C14 and C15 (ESP32-C3 +3.3V decoupling) bind directly to U4's +3.3V pin
        let c14_idx = fp_index("C14");
        let c15_idx = fp_index("C15");
        let tie_c14 = ties.iter().find(|t| t.passive_idx == c14_idx).expect("C14 must have tie");
        let tie_c15 = ties.iter().find(|t| t.passive_idx == c15_idx).expect("C15 must have tie");
        assert!(tie_c14.is_decoupling && tie_c14.anchor_idx == u4_idx, "C14 must bind to U4");
        assert!(tie_c15.is_decoupling && tie_c15.anchor_idx == u4_idx, "C15 must bind to U4");
        assert_eq!(board.footprints[u4_idx].pads[tie_c14.anchor_pad_idx].net_name, "+3.3V");
        assert_eq!(board.footprints[u4_idx].pads[tie_c15.anchor_pad_idx].net_name, "+3.3V");

        // Verify C5 and C6 (MAX22088 power caps) bind to U1 power pins
        let c5_idx = fp_index("C5");
        let c6_idx = fp_index("C6");
        let tie_c5 = ties.iter().find(|t| t.passive_idx == c5_idx).expect("C5 must have tie");
        let tie_c6 = ties.iter().find(|t| t.passive_idx == c6_idx).expect("C6 must have tie");
        assert!(tie_c5.is_decoupling && tie_c5.anchor_idx == u1_idx, "C5 must bind to U1");
        assert!(tie_c6.is_decoupling && tie_c6.anchor_idx == u1_idx, "C6 must bind to U1");
        assert_eq!(board.footprints[u1_idx].pads[tie_c5.anchor_pad_idx].net_name.trim_start_matches('/'), "VCC_5V");
        assert_eq!(board.footprints[u1_idx].pads[tie_c6.anchor_pad_idx].net_name.trim_start_matches('/'), "VRAW");

        // Verify C7 (Buck regulator input decoupling cap on /BUCK_VIN) binds to U2
        let c7_idx = fp_index("C7");
        let tie_c7 = ties.iter().find(|t| t.passive_idx == c7_idx).expect("C7 must have tie");
        assert!(tie_c7.is_decoupling && tie_c7.anchor_idx == u2_idx, "C7 must bind to U2");
        assert_eq!(board.footprints[u2_idx].pads[tie_c7.anchor_pad_idx].net_name, "/BUCK_VIN");
    }

    #[test]
    fn test_prioritize_and_position_diff_pairs_and_decoupling_caps() {
        let pcb_path = "/tmp/vulcan_test_priority_placement.kicad_pcb";
        let content = r#"(kicad_pcb
          (net 0 "") (net 1 "GND") (net 2 "+3.3V") (net 3 "USB_D+") (net 4 "USB_D-")
          (gr_rect (start 100.0 100.0) (end 160.0 140.0) (layer "Edge.Cuts"))
          (footprint "U" (at 120.0 120.0) (property "Reference" "U1") (locked yes)
            (fp_line (start -3.0 -3.0) (end 3.0 -3.0) (layer "F.CrtYd"))
            (fp_line (start 3.0 -3.0) (end 3.0 3.0) (layer "F.CrtYd"))
            (fp_line (start 3.0 3.0) (end -3.0 3.0) (layer "F.CrtYd"))
            (fp_line (start -3.0 3.0) (end -3.0 -3.0) (layer "F.CrtYd"))
            (pad "1" smd rect (at 0.0 -2.5) (size 0.5 0.5) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.0 2.5) (size 0.5 0.5) (net 1 "GND")))
          (footprint "C" (at 150.0 135.0) (property "Reference" "C1")
            (fp_line (start -0.8 -0.4) (end 0.8 -0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 -0.4) (end 0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 0.4) (end -0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start -0.8 0.4) (end -0.8 -0.4) (layer "F.CrtYd"))
            (pad "1" smd rect (at -0.5 0.0) (size 0.4 0.4) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.5 0.0) (size 0.4 0.4) (net 1 "GND")))
          (footprint "J" (at 105.0 120.0) (property "Reference" "J1") (locked yes)
            (pad "1" smd rect (at 0.0 -0.5) (size 0.4 0.4) (net 3 "USB_D+"))
            (pad "2" smd rect (at 0.0 0.5) (size 0.4 0.4) (net 4 "USB_D-")))
          (footprint "U" (at 155.0 135.0) (property "Reference" "U2")
            (pad "1" smd rect (at 0.0 -0.5) (size 0.4 0.4) (net 3 "USB_D+"))
            (pad "2" smd rect (at 0.0 0.5) (size 0.4 0.4) (net 4 "USB_D-")))
        )"#;
        fs::write(pcb_path, content).unwrap();

        let mut board = Board::load(pcb_path, &HashSet::new()).unwrap();
        let cfg = PlacementConfig {
            grid_step_mm: 0.1,
            ..PlacementConfig::default()
        };

        assert_eq!(board.footprints[1].x, 150.0);

        prioritize_and_position_initial_components(&mut board, &cfg, None);

        let u1_pin1_x = 120.0;
        let u1_pin1_y = 117.5;
        let c1_pin1_pos = board.footprints[1].pad_absolute_position(&board.footprints[1].pads[0]);
        let dist = ((c1_pin1_pos.0 - u1_pin1_x).powi(2) + (c1_pin1_pos.1 - u1_pin1_y).powi(2)).sqrt();
        assert!(dist < 2.0, "Decoupling cap power pin must be situated within 2.0 mm of IC power pin, got {:.2} mm", dist);
        assert!(board.footprints[3].x < 130.0, "Diff pair IC U2 should be moved near locked connector J1");

        let _ = fs::remove_file(pcb_path);
    }

    #[test]
    fn test_find_most_favourable_placement_and_holding_palette() {
        let pcb_path = "/tmp/vulcan_fav_placement_test.kicad_pcb";
        let content = r#"(kicad_pcb (version 20240108)
          (net 0 "") (net 1 "GND") (net 2 "+3.3V") (net 3 "SIG_NET")
          (gr_rect (start 100.0 100.0) (end 160.0 140.0) (layer "Edge.Cuts"))
          (footprint "U" (at 120.0 120.0) (property "Reference" "U1") (locked yes)
            (fp_line (start -3.0 -3.0) (end 3.0 -3.0) (layer "F.CrtYd"))
            (fp_line (start 3.0 -3.0) (end 3.0 3.0) (layer "F.CrtYd"))
            (fp_line (start 3.0 3.0) (end -3.0 3.0) (layer "F.CrtYd"))
            (fp_line (start -3.0 3.0) (end -3.0 -3.0) (layer "F.CrtYd"))
            (pad "1" smd rect (at 0.0 -2.5) (size 0.5 0.5) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.0 2.5) (size 0.5 0.5) (net 1 "GND"))
            (pad "3" smd rect (at 2.5 0.0) (size 0.5 0.5) (net 3 "SIG_NET")))
          (footprint "C" (at 150.0 135.0) (property "Reference" "C1")
            (fp_line (start -0.8 -0.4) (end 0.8 -0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 -0.4) (end 0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 0.4) (end -0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start -0.8 0.4) (end -0.8 -0.4) (layer "F.CrtYd"))
            (pad "1" smd rect (at -0.5 0.0) (size 0.4 0.4) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.5 0.0) (size 0.4 0.4) (net 1 "GND")))
          (footprint "R" (at 155.0 135.0) (property "Reference" "R12")
            (fp_line (start -0.8 -0.4) (end 0.8 -0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 -0.4) (end 0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 0.4) (end -0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start -0.8 0.4) (end -0.8 -0.4) (layer "F.CrtYd"))
            (pad "1" smd rect (at -0.5 0.0) (size 0.4 0.4) (net 3 "SIG_NET"))
            (pad "2" smd rect (at 0.5 0.0) (size 0.4 0.4) (net 1 "GND")))
        )"#;
        fs::write(pcb_path, content).unwrap();

        let mut board = Board::load(pcb_path, &HashSet::new()).unwrap();
        let cfg = PlacementConfig {
            grid_step_mm: 0.1,
            ..PlacementConfig::default()
        };

        // Clear unlocked objects (C1 and R12 move to holding palette)
        let (_ripped, cleared) = board.clear_unlocked_objects();
        assert_eq!(cleared, 2);
        assert_eq!(board.holding_palette.len(), 2);
        assert_eq!(board.footprints.len(), 1); // Only U1 on board

        // Place C1 individually at best location
        let placed_c1 = place_holding_component_at_best_location(&mut board, 0, &cfg).unwrap();
        assert_eq!(placed_c1.reference, "C1");
        assert_eq!(board.footprints.len(), 2);
        assert_eq!(board.holding_palette.len(), 1);

        // Verify C1 was placed at most favourable location near U1 pin 1 (120.0, 117.5), NOT at (150.0, 135.0)
        let orig_dist = (placed_c1.x - 150.0).hypot(placed_c1.y - 135.0);
        assert!(orig_dist > 5.0, "C1 must not be placed at its original cleared position");

        let u1_pin1_x = 120.0;
        let u1_pin1_y = 117.5;
        let c1_pin1 = placed_c1.pad_absolute_position(&placed_c1.pads[0]);
        let pin_dist = (c1_pin1.0 - u1_pin1_x).hypot(c1_pin1.1 - u1_pin1_y);
        assert!(pin_dist < 2.0, "C1 power pin must be within 2.0 mm of U1 power pin, got {:.2} mm", pin_dist);

        // Place all remaining components (R12)
        let placed_all = place_all_holding_components_at_best_locations(&mut board, &cfg);
        assert_eq!(placed_all, 1);
        assert_eq!(board.holding_palette.len(), 0);
        assert_eq!(board.footprints.len(), 3);

        // Verify 0 courtyard collisions among all 3 placed components
        let st = PlacementState::from_board(&board, &cfg);
        assert_eq!(st.overlaps, 0, "All placed components must have 0 courtyard collisions");

        let _ = fs::remove_file(pcb_path);
    }

    #[test]
    fn test_decoupling_caps_c14_c15_esp32_t_shape_notch_affinity() {
        let pcb_path = "/tmp/vulcan_esp32_t_shape_caps_test.kicad_pcb";
        let content = r#"(kicad_pcb (version 20240108)
          (net 0 "") (net 1 "GND") (net 2 "+3.3V")
          (gr_rect (start 90.0 80.0) (end 190.0 140.0) (layer "Edge.Cuts"))
          (footprint "Buck_IC" (at 110.0 110.0) (property "Reference" "U1") (locked yes)
            (fp_line (start -2.0 -2.0) (end 2.0 -2.0) (layer "F.CrtYd"))
            (fp_line (start 2.0 -2.0) (end 2.0 2.0) (layer "F.CrtYd"))
            (fp_line (start 2.0 2.0) (end -2.0 2.0) (layer "F.CrtYd"))
            (fp_line (start -2.0 2.0) (end -2.0 -2.0) (layer "F.CrtYd"))
            (pad "1" smd rect (at 0.0 -1.5) (size 0.5 0.5) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.0 1.5) (size 0.5 0.5) (net 1 "GND")))
          (footprint "ESP32-C3-WROOM-02" (at 165.0 105.0 -90) (property "Reference" "U4") (locked yes)
            (fp_line (start -9.75 7.15) (end 9.75 7.15) (layer "F.CrtYd"))
            (fp_line (start -9.75 7.15) (end -9.75 -6.85) (layer "F.CrtYd"))
            (fp_line (start -14.25 -6.85) (end -9.75 -6.85) (layer "F.CrtYd"))
            (fp_line (start 9.75 -6.85) (end 9.75 7.15) (layer "F.CrtYd"))
            (fp_line (start 9.75 -6.85) (end 14.25 -6.85) (layer "F.CrtYd"))
            (fp_line (start -14.25 -18.35) (end -14.25 -6.85) (layer "F.CrtYd"))
            (fp_line (start -14.25 -18.35) (end 14.25 -18.35) (layer "F.CrtYd"))
            (fp_line (start 14.25 -18.35) (end 14.25 -6.85) (layer "F.CrtYd"))
            (pad "1" smd rect (at -8.75 6.0) (size 0.9 1.5) (net 2 "+3.3V"))
            (pad "9" smd rect (at 8.75 6.0) (size 0.9 1.5) (net 1 "GND")))
          (footprint "C_0603" (at 100.0 90.0) (property "Reference" "C14")
            (fp_line (start -0.8 -0.4) (end 0.8 -0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 -0.4) (end 0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 0.4) (end -0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start -0.8 0.4) (end -0.8 -0.4) (layer "F.CrtYd"))
            (pad "1" smd rect (at -0.5 0.0) (size 0.4 0.4) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.5 0.0) (size 0.4 0.4) (net 1 "GND")))
          (footprint "C_0603" (at 105.0 90.0) (property "Reference" "C15")
            (fp_line (start -0.8 -0.4) (end 0.8 -0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 -0.4) (end 0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start 0.8 0.4) (end -0.8 0.4) (layer "F.CrtYd"))
            (fp_line (start -0.8 0.4) (end -0.8 -0.4) (layer "F.CrtYd"))
            (pad "1" smd rect (at -0.5 0.0) (size 0.4 0.4) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.5 0.0) (size 0.4 0.4) (net 1 "GND")))
        )"#;
        fs::write(pcb_path, content).unwrap();

        let mut board = Board::load(pcb_path, &HashSet::new()).unwrap();
        // Set schematic cluster & positions
        board.clusters = vec![
            CompiledCluster {
                name: "ESP32 & CONTROLS".to_string(),
                anchor_idx: Some(1),
                member_indices: vec![1, 2, 3],
                member_refs: vec!["U4".to_string(), "C14".to_string(), "C15".to_string()],
                max_radius: 15.0,
                weight: 100.0,
            },
        ];
        board.schematic_positions.insert("U1".to_string(), (50.0, 50.0));
        board.schematic_positions.insert("U4".to_string(), (320.0, 160.0));
        board.schematic_positions.insert("C14".to_string(), (305.0, 155.0));
        board.schematic_positions.insert("C15".to_string(), (305.0, 165.0));

        let cfg = PlacementConfig {
            grid_step_mm: 0.1,
            ..PlacementConfig::default()
        };

        // Clear unlocked objects (C14 and C15 moved to palette)
        let (_ripped, cleared) = board.clear_unlocked_objects();
        assert_eq!(cleared, 2);
        assert_eq!(board.holding_palette.len(), 2);
        assert_eq!(board.footprints.len(), 2); // U1 and U4

        // Place C14
        let placed_c14 = place_holding_component_at_best_location(&mut board, 0, &cfg).unwrap();
        assert_eq!(placed_c14.reference, "C14");

        // Place C15
        let placed_c15 = place_holding_component_at_best_location(&mut board, 0, &cfg).unwrap();
        assert_eq!(placed_c15.reference, "C15");

        // Compute U4 pin 1 absolute position
        let u4 = board.footprints.iter().find(|f| f.reference == "U4").unwrap();
        let (u4_p1_x, u4_p1_y) = u4.pad_absolute_position(&u4.pads[0]);

        // Verify C14 is placed right next to U4 pin 1, NOT near U1 (110.0, 110.0)
        let c14_to_u4_pin = (placed_c14.x - u4_p1_x).hypot(placed_c14.y - u4_p1_y);
        let c14_to_u1 = (placed_c14.x - 110.0).hypot(placed_c14.y - 110.0);
        assert!(c14_to_u4_pin < 3.5, "C14 should be placed adjacent to U4 pin 1, got {:.2} mm", c14_to_u4_pin);
        assert!(c14_to_u1 > 20.0, "C14 must not be placed near U1");

        // Verify C15 is also placed right next to U4 pin 1 / C14
        let c15_to_u4_pin = (placed_c15.x - u4_p1_x).hypot(placed_c15.y - u4_p1_y);
        assert!(c15_to_u4_pin < 4.0, "C15 should be placed adjacent to U4 pin 1, got {:.2} mm", c15_to_u4_pin);

        // Verify zero courtyard collisions across all components!
        let st = PlacementState::from_board(&board, &cfg);
        assert_eq!(st.overlaps, 0, "All components including U4, C14, C15 must have 0 courtyard collisions");

        let _ = fs::remove_file(pcb_path);
    }

    #[test]
    fn test_pin_density_courtyard_expansion_heuristic() {
        // 1. Dense 24-QFN (e.g. MAX22088: 24 pins around 4x4mm, 0.50mm pitch)
        let mut qfn_pads = Vec::new();
        // 6 pads per side, 0.50mm pitch
        for i in 0..6 {
            let offset = -1.25 + (i as f64) * 0.50;
            qfn_pads.push(Pad::new(&format!("{}", i + 1), offset, -1.9, 0.25, 0.6, 1, "SIG", false));
            qfn_pads.push(Pad::new(&format!("{}", i + 7), 1.9, offset, 0.6, 0.25, 1, "SIG", false));
            qfn_pads.push(Pad::new(&format!("{}", i + 13), -offset, 1.9, 0.25, 0.6, 1, "SIG", false));
            qfn_pads.push(Pad::new(&format!("{}", i + 19), -1.9, -offset, 0.6, 0.25, 1, "SIG", false));
        }
        let qfn = Footprint {
            reference: "U1".to_string(),
            value: "MAX22088".to_string(),
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: qfn_pads,
            width: 4.0,
            height: 4.0,
            courtyards: vec![Polygon::new(vec![
                Point::new(-2.2, -2.2),
                Point::new(2.2, -2.2),
                Point::new(2.2, 2.2),
                Point::new(-2.2, 2.2),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };
        assert!((qfn.min_pad_pitch() - 0.50).abs() < 1e-4, "QFN pitch must be 0.50mm");
        assert!(qfn.pad_density() > 1.0, "QFN pad density must be > 1.0 pads/mm²");
        let qfn_escape = qfn.compute_escape_margin();
        assert_eq!(qfn_escape, 1.20, "Dense 0.50mm QFN must receive 1.20mm escape margin");
        assert_eq!(qfn.placement_spacing(0.25), 0.25 + 2.0 * 1.20);

        // 2. ESP32 RF module (31 pads, 0.55mm pitch, 18x25.5mm package)
        let mut esp_pads = Vec::new();
        for i in 0..31 {
            esp_pads.push(Pad::new(&format!("{}", i + 1), -8.75 + (i as f64) * 0.55, 6.0, 0.35, 1.0, 1, "SIG", false));
        }
        let esp32 = Footprint {
            reference: "U4".to_string(),
            value: "ESP32-C3-WROOM-02".to_string(),
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: esp_pads,
            width: 18.0,
            height: 25.5,
            courtyards: vec![Polygon::new(vec![
                Point::new(-9.0, -12.75),
                Point::new(9.0, -12.75),
                Point::new(9.0, 12.75),
                Point::new(-9.0, 12.75),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };
        assert!((esp32.min_pad_pitch() - 0.55).abs() < 1e-4);
        assert_eq!(esp32.compute_escape_margin(), 1.00, "ESP32 module must receive 1.00mm escape margin");

        // 3. SOT-23-6 BUCK IC (6 pads, 0.95mm pitch)
        let mut sot_pads = Vec::new();
        for i in 0..3 {
            sot_pads.push(Pad::new(&format!("{}", i + 1), -0.95 + (i as f64) * 0.95, -1.3, 0.45, 0.6, 1, "SIG", false));
            sot_pads.push(Pad::new(&format!("{}", i + 4), 0.95 - (i as f64) * 0.95, 1.3, 0.45, 0.6, 1, "SIG", false));
        }
        let sot23_6 = Footprint {
            reference: "U2".to_string(),
            value: "BUCK_IC".to_string(),
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: sot_pads,
            width: 3.0,
            height: 3.0,
            courtyards: vec![Polygon::new(vec![
                Point::new(-1.6, -1.6),
                Point::new(1.6, -1.6),
                Point::new(1.6, 1.6),
                Point::new(-1.6, 1.6),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };
        assert!((sot23_6.min_pad_pitch() - 0.95).abs() < 1e-4);
        assert_eq!(sot23_6.compute_escape_margin(), 0.75, "SOT-23-6 BUCK IC must receive 0.75mm escape margin");

        // 4. Standard 0603 passive (2 pads, 1.60mm pitch)
        let r0603 = Footprint {
            reference: "R1".to_string(),
            value: "10k".to_string(),
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                Pad::new("1", -0.8, 0.0, 0.6, 0.7, 1, "SIG", false),
                Pad::new("2", 0.8, 0.0, 0.6, 0.7, 1, "SIG", false),
            ],
            width: 2.2,
            height: 1.2,
            courtyards: vec![Polygon::new(vec![
                Point::new(-1.1, -0.6),
                Point::new(1.1, -0.6),
                Point::new(1.1, 0.6),
                Point::new(-1.1, 0.6),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };
        assert_eq!(r0603.compute_escape_margin(), 0.0, "0603 passive must receive 0.0mm escape margin (standard clearance)");
        assert_eq!(r0603.placement_spacing(0.25), 0.25);
    }

    #[test]
    fn test_escape_clearance_enforcement_in_placement_and_legalization() {
        let pcb_path = "/tmp/vulcan_escape_clearance_test.kicad_pcb";
        // Board with QFN-24 (locked at 100, 100) and an unlocked 0603 capacitor initially placed too close
        let content = r#"(kicad_pcb (version 20240108)
          (net 0 "") (net 1 "GND") (net 2 "+3.3V")
          (gr_rect (start 80.0 80.0) (end 130.0 130.0) (layer "Edge.Cuts"))
          (footprint "Package_DFN_QFN:QFN-24" (at 100.0 100.0) (property "Reference" "U1") (locked yes)
            (fp_rect (start -2.0 -2.0) (end 2.0 2.0) (layer "F.CrtYd"))
            (pad "1" smd rect (at -1.8 -1.25) (size 0.5 0.25) (net 2 "+3.3V"))
            (pad "2" smd rect (at -1.8 -0.75) (size 0.5 0.25) (net 1 "GND"))
            (pad "3" smd rect (at -1.8 -0.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "4" smd rect (at -1.8 0.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "5" smd rect (at -1.8 0.75) (size 0.5 0.25) (net 1 "GND"))
            (pad "6" smd rect (at -1.8 1.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "7" smd rect (at -1.25 1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "8" smd rect (at -0.75 1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "9" smd rect (at -0.25 1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "10" smd rect (at 0.25 1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "11" smd rect (at 0.75 1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "12" smd rect (at 1.25 1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "13" smd rect (at 1.8 1.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "14" smd rect (at 1.8 0.75) (size 0.5 0.25) (net 1 "GND"))
            (pad "15" smd rect (at 1.8 0.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "16" smd rect (at 1.8 -0.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "17" smd rect (at 1.8 -0.75) (size 0.5 0.25) (net 1 "GND"))
            (pad "18" smd rect (at 1.8 -1.25) (size 0.5 0.25) (net 1 "GND"))
            (pad "19" smd rect (at 1.25 -1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "20" smd rect (at 0.75 -1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "21" smd rect (at 0.25 -1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "22" smd rect (at -0.25 -1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "23" smd rect (at -0.75 -1.8) (size 0.25 0.5) (net 1 "GND"))
            (pad "24" smd rect (at -1.25 -1.8) (size 0.25 0.5) (net 1 "GND")))
          (footprint "Capacitor_SMD:C_0603" (at 103.5 100.0) (property "Reference" "C1")
            (fp_rect (start -1.0 -0.6) (end 1.0 0.6) (layer "F.CrtYd"))
            (pad "1" smd rect (at -0.7 0.0) (size 0.6 0.6) (net 2 "+3.3V"))
            (pad "2" smd rect (at 0.7 0.0) (size 0.6 0.6) (net 1 "GND")))
        )"#;
        fs::write(pcb_path, content).unwrap();

        let mut board = Board::load(pcb_path, &HashSet::new()).unwrap();
        let cfg = PlacementConfig {
            grid_step_mm: 0.1,
            ..PlacementConfig::default()
        };

        // U1 has 24 pads with 0.50mm pitch -> 1.20mm escape margin
        let u1 = &board.footprints[0];
        assert_eq!(u1.escape_margin, 1.20);

        // At x = 103.5, C1 is 3.5mm from U1 center (2.0mm QFN half + 1.0mm C1 half = 3.0mm,
        // distance between raw courtyards is only 0.5mm, which violates the 1.20mm escape corridor!)
        let c1 = &board.footprints[1];
        assert!(!is_placement_legal(&board, c1, Point::new(103.5, 100.0), 0.0, 0, cfg.min_spacing_mm),
            "C1 at 103.5mm sits inside the 1.20mm escape halo of U1 and must be deemed illegal");

        // At x = 104.5, C1 is outside the 1.20mm escape halo and is legal
        assert!(is_placement_legal(&board, c1, Point::new(104.5, 100.0), 0.0, 0, cfg.min_spacing_mm),
            "C1 at 104.5mm sits outside the escape halo and must be legal");

        // Verify that legalization resolves initial overlap by pushing C1 outside the escape halo
        let initial_st = PlacementState::from_board(&board, &cfg);
        assert!(initial_st.overlaps > 0, "Initial placement must detect escape corridor overlap");
        let legalized_st = legalize_placement(&mut board, &cfg, initial_st, None);
        assert_eq!(legalized_st.overlaps, 0, "Legalizer must resolve escape corridor overlap");
        let c1_legal_x = legalized_st.positions[1].x;
        let c1_legal_y = legalized_st.positions[1].y;
        let dist_to_u1 = (c1_legal_x - 100.0).hypot(c1_legal_y - 100.0);
        assert!(dist_to_u1 >= 4.2, "Legalized C1 must maintain at least 4.2mm distance (2.0 + 1.0 + 1.2), got {:.2}mm", dist_to_u1);

        let _ = fs::remove_file(pcb_path);
    }

    #[test]
    fn test_board_outline_boundary_containment_and_legalization() {
        // Octagonal outline: [100, 150] x [100, 150] with 5mm corner chamfers
        let outline = Polygon::new(vec![
            Point::new(105.0, 100.0),
            Point::new(145.0, 100.0),
            Point::new(150.0, 105.0),
            Point::new(150.0, 145.0),
            Point::new(145.0, 150.0),
            Point::new(105.0, 150.0),
            Point::new(100.0, 145.0),
            Point::new(100.0, 105.0),
        ]);

        let j3 = Footprint {
            reference: "J3".to_string(),
            value: "HEADER".to_string(),
            x: 120.0,
            y: 145.0, // Pin 4 extends to 145 + 7.62 = 152.62 > max_y 150.0!
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                Pad::new("1", 0.0, 0.0, 1.4, 1.4, 1, "GND", false),
                Pad::new("2", 0.0, 2.54, 1.4, 1.4, 1, "GND", false),
                Pad::new("3", 0.0, 5.08, 1.4, 1.4, 1, "GND", false),
                Pad::new("4", 0.0, 7.62, 1.4, 1.4, 1, "GND", false),
            ],
            width: 2.5,
            height: 9.62,
            courtyards: vec![Polygon::new(vec![
                Point::new(-1.25, -1.0),
                Point::new(1.25, -1.0),
                Point::new(1.25, 8.62),
                Point::new(-1.25, 8.62),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        // Small 0603 passive in the chamfered corner: (101, 101) is inside AABB [100, 150], but outside chamfered contour
        let c1 = Footprint {
            reference: "C1".to_string(),
            value: "100nF".to_string(),
            x: 101.5,
            y: 101.5,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                Pad::new("1", -0.7, 0.0, 0.6, 0.6, 1, "GND", false),
                Pad::new("2", 0.7, 0.0, 0.6, 0.6, 2, "VCC", false),
            ],
            width: 2.0,
            height: 1.2,
            courtyards: vec![Polygon::new(vec![
                Point::new(-1.0, -0.6),
                Point::new(1.0, -0.6),
                Point::new(1.0, 0.6),
                Point::new(-1.0, 0.6),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let nets = vec!["".to_string(), "GND".to_string(), "VCC".to_string()];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        for n in &nets {
            net_to_class.insert(n.clone(), "Default".to_string());
        }

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 100.0,
            min_y: 100.0,
            max_x: 150.0,
            max_y: 150.0,
            footprints: vec![j3.clone(), c1.clone()],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: Some(outline),
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let cfg = PlacementConfig {
            grid_step_mm: 0.25,
            min_spacing_mm: 0.25,
            duration_secs: 1,
            ..PlacementConfig::default()
        };

        // 1. Verify that J3 at (120, 145) is rejected because pin 4 exceeds the boundary
        assert!(
            !is_placement_legal(&board, &j3, Point::new(120.0, 145.0), 0.0, 0, cfg.min_spacing_mm),
            "J3 with pin 4 protruding past boundary must be illegal"
        );

        // 2. Verify that C1 at (101.5, 101.5) is rejected because it is inside the chamfer corner
        assert!(
            !is_placement_legal(&board, &c1, Point::new(101.5, 101.5), 0.0, 0, cfg.min_spacing_mm),
            "C1 inside corner chamfer cutout must be illegal"
        );

        // 3. Verify that initial state has boundary violations
        let init_st = PlacementState::from_board(&board, &cfg);
        assert!(
            init_st.boundary_violations > 0,
            "Initial placement must register boundary violations"
        );

        // 4. Legalize placement and verify that all boundary violations are eliminated
        let legalized = legalize_placement(&mut board, &cfg, init_st, None);
        assert_eq!(
            legalized.boundary_violations, 0,
            "Legalizer must eliminate all boundary violations"
        );
        assert_eq!(legalized.overlaps, 0, "Legalizer must eliminate overlaps");

        // Verify every pad corner of both components is inside the board outline polygon
        legalized.apply_to_board(&mut board);
        for fp in &board.footprints {
            let pos = Point::new(fp.x, fp.y);
            assert!(board.contains_point_in_boundary(&pos), "Footprint origin {} must be inside", fp.reference);
            for pad in &fp.pads {
                for corner in fp.pad_corners_at(pad, pos, fp.rotation, fp.layer) {
                    assert!(
                        board.contains_point_in_boundary(&corner),
                        "Pad {} corner ({:.2}, {:.2}) of {} must be inside board outline",
                        pad.number, corner.x, corner.y, fp.reference
                    );
                }
            }
        }
    }

    #[test]
    fn test_aggressive_placement_moves_and_centroid_pull() {
        use std::sync::atomic::AtomicBool;

        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 60.0;
        board.max_y = 60.0;
        board.outline_polygon = Some(Polygon::from_rect(0.0, 0.0, 60.0, 60.0));

        let net_id = 1;
        board.nets.insert(net_id, "VCC_NET".to_string());

        // J1: Locked anchor at (45, 45)
        let j1 = Footprint {
            reference: "J1".to_string(),
            value: "CONN".to_string(),
            x: 45.0,
            y: 45.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: true,
            pads: vec![crate::board::Pad {
                number: "1".to_string(),
                net_id,
                net_name: "VCC_NET".to_string(),
                offset_x: 0.0,
                offset_y: 0.0,
                width: 1.0,
                height: 1.0,
                shape: "rect".to_string(),
                rotation: 0.0,
                is_through_hole: false,
                drill: None,
                ..Default::default()
            }],
            width: 4.0,
            height: 4.0,
            courtyards: vec![Polygon::from_rect(43.0, 43.0, 4.0, 4.0)],
            perimeters: vec![],
            escape_margin: 0.25,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        // U1: Unlocked IC initially at (10, 10)
        let u1 = Footprint {
            reference: "U1".to_string(),
            value: "MCU".to_string(),
            x: 10.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![crate::board::Pad {
                number: "1".to_string(),
                net_id,
                net_name: "VCC_NET".to_string(),
                offset_x: 0.0,
                offset_y: 0.0,
                width: 1.0,
                height: 1.0,
                shape: "rect".to_string(),
                rotation: 0.0,
                is_through_hole: false,
                drill: None,
                ..Default::default()
            }],
            width: 5.0,
            height: 5.0,
            courtyards: vec![Polygon::from_rect(7.5, 7.5, 5.0, 5.0)],
            perimeters: vec![],
            escape_margin: 0.25,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        // C1: Unlocked capacitor initially at (15, 10)
        let c1 = Footprint {
            reference: "C1".to_string(),
            value: "100nF".to_string(),
            x: 15.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![crate::board::Pad {
                number: "1".to_string(),
                net_id,
                net_name: "VCC_NET".to_string(),
                offset_x: 0.0,
                offset_y: 0.0,
                width: 0.6,
                height: 0.6,
                shape: "rect".to_string(),
                rotation: 0.0,
                is_through_hole: false,
                drill: None,
                ..Default::default()
            }],
            width: 2.0,
            height: 1.5,
            courtyards: vec![Polygon::from_rect(14.0, 9.25, 2.0, 1.5)],
            perimeters: vec![],
            escape_margin: 0.25,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        board.footprints = vec![j1, u1, c1];

        let cfg = PlacementConfig {
            duration_secs: 1,
            grid_step_mm: 0.5,
            min_spacing_mm: 0.25,
            max_jump_radius_mm: 25.0,
            pair_swap_prob: 0.25,
            centroid_pull_prob: 0.35,
            temp_scale: 1.5,
            ..PlacementConfig::default()
        };

        let stop = Arc::new(AtomicBool::new(false));
        let result = run_auto_placer(&mut board, &cfg, None, stop);

        assert_eq!(result.boundary_violations, 0, "All components must be within board outline");
        assert_eq!(result.overlaps, 0, "No courtyard overlaps allowed");

        // Verify locked component stayed fixed
        let j1_after = board.footprints.iter().find(|f| f.reference == "J1").unwrap();
        assert!((j1_after.x - 45.0).abs() < 1e-6);
        assert!((j1_after.y - 45.0).abs() < 1e-6);

        // Verify all components are legally contained
        for fp in &board.footprints {
            let p = Point::new(fp.x, fp.y);
            assert!(board.contains_point_in_boundary(&p), "Footprint {} must be within boundary", fp.reference);
        }
    }

    #[test]
    fn test_esp32_footprint_and_keepout_exclusion() {
        let mut board = make_synthetic_board();
        board.clusters.clear();
        board.min_x = 100.0;
        board.min_y = 80.0;
        board.max_x = 190.0;
        board.max_y = 130.0;

        let esp32_t_poly = Polygon::new(vec![
            Point::new(-9.75, 7.15),
            Point::new(9.75, 7.15),
            Point::new(9.75, -6.85),
            Point::new(14.25, -6.85),
            Point::new(14.25, -18.35),
            Point::new(-14.25, -18.35),
            Point::new(-14.25, -6.85),
            Point::new(-9.75, -6.85),
        ]);
        let esp32_courtyards = esp32_t_poly.decompose_convex();

        let u4 = Footprint {
            reference: "U4".to_string(),
            value: "ESP32-C3-WROOM-02".to_string(),
            x: 165.0,
            y: 105.0,
            rotation: -90.0,
            layer: 0,
            orig_layer: 0,
            is_locked: true,
            pads: vec![
                Pad::new("1", -9.0, 6.7, 0.4, 0.8, 1, "+3.3V", false),
                Pad::new("19", 0.0, 0.0, 3.0, 3.0, 0, "GND", false),
            ],
            width: 18.0,
            height: 20.0,
            courtyards: esp32_courtyards,
            perimeters: Vec::new(),
            escape_margin: 0.70,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let mut c1 = make_test_fp(
            "C1",
            165.0,
            105.0, // directly on ESP32 center
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "+3.3V", false)],
        );

        board.footprints = vec![u4, c1.clone()];
        board.keepouts = vec![crate::board::KeepoutZone {
            min_x: 172.1,
            min_y: 91.0,
            max_x: 183.1,
            max_y: 119.0,
            layers: vec![0, 1],
        }];

        let cfg = PlacementConfig {
            duration_secs: 1,
            grid_step_mm: 0.1,
            min_spacing_mm: 0.25,
            ..PlacementConfig::default()
        };

        // 1. Placing C1 directly on ESP32 module body causes overlap
        let st_body = PlacementState::from_board(&board, &cfg);
        assert!(st_body.overlaps > 0, "Placing on ESP32 body must cause courtyard overlap");

        // 2. Placing C1 inside ESP32 antenna keepout zone causes boundary/keepout violation
        c1.x = 175.0;
        c1.y = 105.0;
        board.footprints[1] = c1.clone();
        let st_ko = PlacementState::from_board(&board, &cfg);
        assert!(st_ko.boundary_violations > 0, "Placing in antenna keepout must cause keepout violation");

        // 3. Placing C1 in the legitimate concave notch outside the module body has 0 overlap
        c1.x = 160.0;
        c1.y = 93.0;
        board.footprints[1] = c1;
        let st_notch = PlacementState::from_board(&board, &cfg);
        assert_eq!(st_notch.boundary_violations, 0, "Notch outside module body must not violate keepout");

        // 4. Auto-placer resolves any initial overlap and keeps C1 outside module body and keepout
        let stop = Arc::new(AtomicBool::new(false));
        let solved = run_auto_placer(&mut board, &cfg, None, stop);
        assert_eq!(solved.overlaps, 0, "Auto-placer must achieve 0 overlaps");
        assert_eq!(solved.boundary_violations, 0, "Auto-placer must achieve 0 keepout violations");

        let c1_final = board.footprints.iter().find(|f| f.reference == "C1").unwrap();
        // Verify C1 is not in the antenna keepout
        let in_ko = c1_final.x >= 172.1 && c1_final.x <= 183.1 && c1_final.y >= 91.0 && c1_final.y <= 119.0;
        assert!(!in_ko, "Final placement must not place component in antenna keepout");
    }

    #[test]
    fn test_metropolis_replica_exchange_accepts_uphill_and_downhill() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let board = make_synthetic_board();
        let cfg = PlacementConfig::default();
        let base_st = PlacementState::from_board(&board, &cfg);

        // Case 1: Downhill move (Hotter replica has LOWER cost than colder replica)
        // Must swap 100% of the time (P = 1.0)
        let mut st0 = base_st.clone();
        st0.cost = 500.0;
        let mut st1 = base_st.clone();
        st1.cost = 300.0;

        let mut replicas = vec![st0, st1];
        let mut rng = StdRng::seed_from_u64(42);
        let swaps = metropolis_replica_exchange(
            &mut replicas,
            2,
            0, // epoch 0 -> starts with pair (0, 1)
            0.5,
            1.0,
            1.0,
            &mut rng,
        );
        assert_eq!(swaps, 1, "Downhill swap where hotter replica is better must always be accepted");
        assert_eq!(replicas[0].cost, 300.0, "Colder replica must receive the better configuration");
        assert_eq!(replicas[1].cost, 500.0, "Hotter replica receives the prior state to continue exploring");

        // Case 2: Uphill move (Hotter replica has HIGHER cost than colder replica)
        // At low temperature (cooling = 0.05, thermal_boost = 1.0), large uphill moves (delta = 500.0) should rarely or never swap.
        // At high temperature (thermal_boost = 10.0), uphill moves have significantly higher probability of acceptance.
        let mut cold_swaps = 0;
        let mut hot_swaps = 0;
        for seed in 0..100 {
            let mut cold_reps = vec![
                { let mut s = base_st.clone(); s.cost = 100.0; s },
                { let mut s = base_st.clone(); s.cost = 300.0; s },
            ];
            let mut rng_cold = StdRng::seed_from_u64(seed);
            cold_swaps += metropolis_replica_exchange(&mut cold_reps, 2, 0, 0.95, 1.0, 1.0, &mut rng_cold);

            let mut hot_reps = vec![
                { let mut s = base_st.clone(); s.cost = 100.0; s },
                { let mut s = base_st.clone(); s.cost = 300.0; s },
            ];
            let mut rng_hot = StdRng::seed_from_u64(seed);
            hot_swaps += metropolis_replica_exchange(&mut hot_reps, 2, 0, 0.0, 1.0, 5.0, &mut rng_hot);
        }
        assert!(hot_swaps > cold_swaps, "Elevated temperature must accept significantly more uphill replica swaps (hot={}, cold={})", hot_swaps, cold_swaps);
    }

    #[test]
    fn test_build_component_clusters_and_rigid_group_move() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 100.0;
        board.max_y = 100.0;
        let boundary = board.boundary_aabb();

        let u1 = make_test_fp("U1", 50.0, 50.0, false, vec![
            Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "VCC", false),
        ]);
        let c1 = make_test_fp("C1", 48.0, 50.0, false, vec![
            Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "VCC", false),
        ]);
        let c2 = make_test_fp("C2", 52.0, 50.0, false, vec![
            Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "VCC", false),
        ]);
        board.footprints = vec![u1, c1, c2];

        let ties = vec![
            PassiveTie {
                anchor_idx: 0,
                anchor_pad_idx: 0,
                passive_idx: 1,
                passive_pad_idx: 0,
                max_dist: 5.0,
                weight_mult: 2.0,
                is_decoupling: true,
            },
            PassiveTie {
                anchor_idx: 0,
                anchor_pad_idx: 0,
                passive_idx: 2,
                passive_pad_idx: 0,
                max_dist: 5.0,
                weight_mult: 2.0,
                is_decoupling: true,
            },
        ];
        let movable = vec![0, 1, 2];
        let clusters = build_component_clusters(&board, &ties, &movable);

        // U1 cluster should contain U1, C1, and C2
        assert_eq!(clusters[0].len(), 3, "U1 cluster must contain anchor and both passives");
        assert!(clusters[0].contains(&0));
        assert!(clusters[0].contains(&1));
        assert!(clusters[0].contains(&2));
        // C1 and C2 should point to the same group
        assert_eq!(clusters[1], clusters[0]);
        assert_eq!(clusters[2], clusters[0]);

        let cfg = PlacementConfig { min_spacing_mm: 0.25, grid_step_mm: 0.1, ..Default::default() };
        let mut state = PlacementState::from_board(&board, &cfg);

        // Apply a legal rigid group move: dx = 10.0, dy = -5.0
        let success = apply_cluster_group_move(&mut state, &board, &cfg, &clusters[0], 10.0, -5.0, &boundary);
        assert!(success, "Cluster move within board bounds must succeed");
        assert!((state.positions[0].x - 60.0).abs() < 1e-6);
        assert!((state.positions[0].y - 45.0).abs() < 1e-6);
        assert!((state.positions[1].x - 58.0).abs() < 1e-6);
        assert!((state.positions[1].y - 45.0).abs() < 1e-6);
        assert!((state.positions[2].x - 62.0).abs() < 1e-6);
        assert!((state.positions[2].y - 45.0).abs() < 1e-6);

        // Relative distances between U1 and C1/C2 are preserved exactly
        let dist_u1_c1 = ((state.positions[0].x - state.positions[1].x).powi(2) + (state.positions[0].y - state.positions[1].y).powi(2)).sqrt();
        assert!((dist_u1_c1 - 2.0).abs() < 1e-6, "Relative decoupling distance must be strictly preserved");

        // Attempt out-of-bounds group move: dx = 60.0 (would push U1 to x=120 > max_x=100)
        let oob_success = apply_cluster_group_move(&mut state, &board, &cfg, &clusters[0], 60.0, 0.0, &boundary);
        assert!(!oob_success, "Out of bounds cluster move must be rejected");
        assert!((state.positions[0].x - 60.0).abs() < 1e-6, "Positions must remain unchanged when rejected");
    }

    #[test]
    fn test_topological_corridor_capacity_calculation() {
        // Test polygon clearance gap calculation
        let p1 = Polygon::from_rect(0.0, 0.0, 2.0, 2.0);
        let p2 = Polygon::from_rect(4.0, 0.0, 2.0, 2.0);
        let gap = polygon_clearance_gap(&p1, &p2);
        assert!((gap - 2.0).abs() < 1e-6);

        let overlapping = Polygon::from_rect(1.0, 0.0, 2.0, 2.0);
        assert_eq!(polygon_clearance_gap(&p1, &overlapping), 0.0);

        // Test multi-layer corridor capacity calculation
        // W = 2.0, track_width = 0.2, min_clearance = 0.2 -> pitch = 0.4
        // (2.0 - 0.2) / 0.4 = 4.5 -> floor = 4 tracks/layer
        assert_eq!(compute_corridor_capacity(2.0, 0.2, 0.2, 2), 8);
        assert_eq!(compute_corridor_capacity(2.0, 0.2, 0.2, 4), 16);
        assert_eq!(compute_corridor_capacity(2.0, 0.2, 0.2, 6), 24);

        // Narrow bottleneck: W = 0.3, clearance = 0.2, track = 0.2 -> (0.3 - 0.2)/0.4 = 0.25 -> 0 tracks
        assert_eq!(compute_corridor_capacity(0.3, 0.2, 0.2, 2), 0);
        assert_eq!(compute_corridor_capacity(0.3, 0.2, 0.2, 4), 0);

        // Gap below clearance
        assert_eq!(compute_corridor_capacity(0.1, 0.2, 0.2, 2), 0);
        assert_eq!(compute_corridor_capacity(0.0, 0.2, 0.2, 2), 0);
    }

    #[test]
    fn test_corridor_congestion_penalizes_bottleneck_overflow() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 50.0;
        board.max_y = 50.0;
        board.nets = vec![
            "".to_string(),
            "SIG1".to_string(),
            "SIG2".to_string(),
            "SIG3".to_string(),
            "SIG4".to_string(),
        ];

        // Connector J1 at (5, 25) with 4 signal pins
        let j1 = make_test_fp("J1", 5.0, 25.0, true, vec![
            Pad::new("1", 0.0, -1.5, 0.5, 0.5, 1, "SIG1", false),
            Pad::new("2", 0.0, -0.5, 0.5, 0.5, 2, "SIG2", false),
            Pad::new("3", 0.0, 0.5, 0.5, 0.5, 3, "SIG3", false),
            Pad::new("4", 0.0, 1.5, 0.5, 0.5, 4, "SIG4", false),
        ]);

        // Connector J2 at (45, 25) with 4 signal pins
        let j2 = make_test_fp("J2", 45.0, 25.0, true, vec![
            Pad::new("1", 0.0, -1.5, 0.5, 0.5, 1, "SIG1", false),
            Pad::new("2", 0.0, -0.5, 0.5, 0.5, 2, "SIG2", false),
            Pad::new("3", 0.0, 0.5, 0.5, 0.5, 3, "SIG3", false),
            Pad::new("4", 0.0, 1.5, 0.5, 0.5, 4, "SIG4", false),
        ]);

        // Obstacles U1 and U2 in the middle at x = 25 forming a corridor
        // Wide configuration: U1 at (25, 15), U2 at (25, 35) -> gap between courtyards = 18 mm (large capacity)
        let mut board_wide = board.clone();
        let u1_wide = make_test_fp("U1", 25.0, 15.0, false, vec![]);
        let u2_wide = make_test_fp("U2", 25.0, 35.0, false, vec![]);
        board_wide.footprints = vec![j1.clone(), j2.clone(), u1_wide, u2_wide];

        let cfg = PlacementConfig {
            corridor_congestion_weight: 50.0,
            crossing_weight: 0.0,
            wirelength_weight: 0.0,
            ..PlacementConfig::default()
        };

        let ties = build_passive_ties(&board_wide);
        let mut st_wide = PlacementState::from_board(&board_wide, &cfg);
        st_wide.evaluate(&board_wide, &cfg, &ties);
        assert_eq!(st_wide.corridor_congestion, 0.0, "Wide corridor should have zero congestion overflow");

        // Bottleneck configuration: U1 at (25, 23.8), U2 at (25, 26.2)
        // Courtyard height = 2.0 (extending -1.0 to +1.0)
        // U1 courtyard max_y = 24.8, U2 courtyard min_y = 25.2 -> gap W = 0.4 mm
        // With default pitch = 0.4 mm and clearance = 0.2 mm, capacity is 0 tracks!
        let mut board_narrow = board;
        let u1_narrow = make_test_fp("U1", 25.0, 23.8, false, vec![]);
        let u2_narrow = make_test_fp("U2", 25.0, 26.2, false, vec![]);
        board_narrow.footprints = vec![j1, j2, u1_narrow, u2_narrow];

        let mut st_narrow = PlacementState::from_board(&board_narrow, &cfg);
        st_narrow.evaluate(&board_narrow, &cfg, &ties);
        assert!(st_narrow.corridor_congestion > 0.0, "Narrow bottleneck must incur corridor congestion overflow");
        assert!(st_narrow.cost > st_wide.cost, "Narrow congested bottleneck cost must exceed wide corridor cost");
    }

    #[test]
    fn test_flow_alignment_penalizes_twisted_bus() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 40.0;
        board.max_y = 40.0;
        board.nets = vec![
            "".to_string(),
            "BUS0".to_string(),
            "BUS1".to_string(),
            "BUS2".to_string(),
            "BUS3".to_string(),
        ];

        // Component U1 with 4 bus pads along Y
        let u1 = make_test_fp("U1", 10.0, 20.0, false, vec![
            Pad::new("1", 0.0, -1.5, 0.5, 0.5, 1, "BUS0", false),
            Pad::new("2", 0.0, -0.5, 0.5, 0.5, 2, "BUS1", false),
            Pad::new("3", 0.0, 0.5, 0.5, 0.5, 3, "BUS2", false),
            Pad::new("4", 0.0, 1.5, 0.5, 0.5, 4, "BUS3", false),
        ]);

        // Aligned U2: Pin 1 faces Pin 1, Pin 2 faces Pin 2, etc.
        let u2_aligned = make_test_fp("U2", 30.0, 20.0, false, vec![
            Pad::new("1", 0.0, -1.5, 0.5, 0.5, 1, "BUS0", false),
            Pad::new("2", 0.0, -0.5, 0.5, 0.5, 2, "BUS1", false),
            Pad::new("3", 0.0, 0.5, 0.5, 0.5, 3, "BUS2", false),
            Pad::new("4", 0.0, 1.5, 0.5, 0.5, 4, "BUS3", false),
        ]);

        // Twisted U2: Pin 1 at top, Pin 4 at bottom (inverted pin order)
        let mut u2_twisted = u2_aligned.clone();
        u2_twisted.pads = vec![
            Pad::new("1", 0.0, 1.5, 0.5, 0.5, 1, "BUS0", false),
            Pad::new("2", 0.0, 0.5, 0.5, 0.5, 2, "BUS1", false),
            Pad::new("3", 0.0, -0.5, 0.5, 0.5, 3, "BUS2", false),
            Pad::new("4", 0.0, -1.5, 0.5, 0.5, 4, "BUS3", false),
        ];

        let cfg = PlacementConfig {
            flow_alignment_weight: 25.0,
            corridor_congestion_weight: 0.0,
            crossing_weight: 0.0,
            wirelength_weight: 0.0,
            ..PlacementConfig::default()
        };

        let mut board_aligned = board.clone();
        board_aligned.footprints = vec![u1.clone(), u2_aligned];
        let ties_aligned = build_passive_ties(&board_aligned);
        let mut st_aligned = PlacementState::from_board(&board_aligned, &cfg);
        st_aligned.evaluate(&board_aligned, &cfg, &ties_aligned);

        let mut board_twisted = board;
        board_twisted.footprints = vec![u1, u2_twisted];
        let ties_twisted = build_passive_ties(&board_twisted);
        let mut st_twisted = PlacementState::from_board(&board_twisted, &cfg);
        st_twisted.evaluate(&board_twisted, &cfg, &ties_twisted);

        assert_eq!(st_aligned.flow_alignment_penalty, 0.0, "Aligned bus ribbon must have zero twist penalty");
        assert!(st_twisted.flow_alignment_penalty > 0.0, "Twisted bus ribbon must incur flow alignment penalty");
        assert!(st_aligned.cost < st_twisted.cost, "Aligned configuration cost must be strictly less than twisted configuration");
    }

    #[test]
    fn test_topological_placement_annealing_convergence() {
        let mut board = make_synthetic_board();
        let cfg = PlacementConfig {
            duration_secs: 1,
            corridor_congestion_weight: 20.0,
            flow_alignment_weight: 15.0,
            crossing_weight: 10.0,
            wirelength_weight: 1.0,
            ..PlacementConfig::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let final_st = run_auto_placer(&mut board, &cfg, None, stop);

        assert_eq!(final_st.overlaps, 0, "Annealing must resolve all component courtyard overlaps");
        assert_eq!(final_st.boundary_violations, 0, "Annealing must keep all components within board boundary");
        assert!(final_st.cost.is_finite(), "Placement cost must be finite");
    }

    #[test]
    fn test_placed_board_spatial_index_lbvh_legality_equivalence() {
        let mut board = make_synthetic_board();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 60.0;
        board.max_y = 60.0;
        board.keepouts.push(crate::board::KeepoutZone {
            min_x: 20.0,
            min_y: 20.0,
            max_x: 28.0,
            max_y: 28.0,
            layers: vec![0, 1],
        });

        // Populate 20 footprints across both F.Cu (layer 0) and B.Cu (layer 1), plus through-hole
        for k in 0..16usize {
            let fx = 6.0 + (k % 4) as f64 * 12.0;
            let fy = 6.0 + (k / 4) as f64 * 12.0;
            let is_th = k % 5 == 0;
            let mut fp = make_test_fp(
                &format!("U_GRID_{}", k),
                fx,
                fy,
                true,
                vec![Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "SIG1", is_th)],
            );
            fp.layer = if k % 3 == 0 { 1 } else { 0 };
            fp.rotation = ((k % 4) as f64) * 90.0;
            if k % 4 == 0 {
                fp.escape_margin = 0.8;
            }
            board.footprints.push(fp);
        }

        let min_spacing = 0.25;
        let spatial_index = PlacedBoardSpatialIndex::from_board(&board, min_spacing);

        let mut cand_fp = make_test_fp(
            "C_PROBE",
            10.0,
            10.0,
            false,
            vec![
                Pad::new("1", -0.8, 0.0, 0.6, 0.6, 1, "+3V3", false),
                Pad::new("2", 0.8, 0.0, 0.6, 0.6, 2, "GND", false),
            ],
        );
        cand_fp.escape_margin = 0.35;

        let mut rng = StdRng::seed_from_u64(0xCAFE_BABE_2026);
        let rotations = [0.0, 45.0, 90.0, 180.0, 270.0];
        let mut candidates = Vec::with_capacity(260);
        let mut legal_count = 0usize;
        let mut illegal_count = 0usize;

        for i in 0..260usize {
            let x = rng.gen_range(-2.0..62.0);
            let y = rng.gen_range(-2.0..62.0);
            let rot = rotations[i % rotations.len()];
            let layer = i % 2;
            let pos = Point::new(x, y);
            candidates.push((pos, rot, layer));

            let ref_legal = is_placement_legal(&board, &cand_fp, pos, rot, layer, min_spacing);
            let bvh_legal = spatial_index.is_placement_legal(&board, &cand_fp, pos, rot, layer);
            assert_eq!(
                bvh_legal, ref_legal,
                "Legality mismatch at candidate {} pos=({:.2}, {:.2}), rot={}, layer={}",
                i, x, y, rot, layer
            );
            if ref_legal {
                legal_count += 1;
            } else {
                illegal_count += 1;
            }
        }

        assert!(legal_count > 20, "Must test sufficient legal poses");
        assert!(illegal_count > 20, "Must test sufficient colliding/out-of-bounds poses");

        let batch_res = spatial_index.filter_legal_poses_batch(&board, &cand_fp, &candidates);
        assert_eq!(batch_res.len(), candidates.len());
        for (i, &(pos, rot, layer)) in candidates.iter().enumerate() {
            assert_eq!(
                batch_res[i],
                is_placement_legal(&board, &cand_fp, pos, rot, layer, min_spacing)
            );
        }
    }

    #[test]
    fn test_placer_lbvh_corridor_and_crossings_exact_match() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 80.0;
        board.max_y = 80.0;

        let num_nets = 28usize;
        board.nets = (0..=num_nets).map(|i| if i == 0 { "".to_string() } else { format!("SIG_{}", i) }).collect();

        // Build left and right banks with inverted/crossed pad assignments (>= 16 segments)
        let mut pads_left = Vec::new();
        let mut pads_right = Vec::new();
        for n in 1..=num_nets {
            let y_left = -14.0 + (n as f64) * 1.0;
            let y_right = 14.0 - (n as f64) * 1.0 + ((n % 5) as f64) * 0.35;
            pads_left.push(Pad::new(&format!("{}", n), 0.0, y_left, 0.4, 0.4, n, &format!("SIG_{}", n), false));
            pads_right.push(Pad::new(&format!("{}", n), 0.0, y_right, 0.4, 0.4, n, &format!("SIG_{}", n), false));
        }

        let j_left = make_test_fp("J1", 10.0, 40.0, true, pads_left);
        let j_right = make_test_fp("J2", 70.0, 40.0, true, pads_right);

        // Place several components in the middle creating narrow corridor cuts across the ratsnest bundle
        let u1 = make_test_fp("U1", 38.0, 38.6, false, vec![
            Pad::new("1", -0.5, 0.0, 0.4, 0.4, 1, "SIG_1", false),
            Pad::new("2", 0.5, 0.0, 0.4, 0.4, 2, "SIG_2", false),
        ]);
        let u2 = make_test_fp("U2", 38.0, 41.4, false, vec![
            Pad::new("1", -0.5, 0.0, 0.4, 0.4, 2, "SIG_2", false),
            Pad::new("2", 0.5, 0.0, 0.4, 0.4, 3, "SIG_3", false),
        ]);
        let u3 = make_test_fp("U3", 44.0, 39.5, false, vec![
            Pad::new("1", 0.0, -0.5, 0.4, 0.4, 4, "SIG_4", false),
        ]);
        let u4 = make_test_fp("U4", 44.0, 42.3, false, vec![
            Pad::new("1", 0.0, 0.5, 0.4, 0.4, 5, "SIG_5", false),
        ]);

        board.footprints = vec![j_left, j_right, u1, u2, u3, u4];

        let cfg = PlacementConfig::default();
        let ties = build_passive_ties(&board);
        let mut st = PlacementState::from_board(&board, &cfg);
        st.evaluate_cost(&board, &cfg, &ties);

        let segments = board.generate_ratsnest_segments(&st.cached_pads);
        assert!(segments.len() >= 16, "Expected >= 16 ratsnest segments to exercise LBVH path");

        // Compute brute-force reference crossings
        let mut ref_crossings = 0usize;
        for i in 0..segments.len() {
            for j in (i + 1)..segments.len() {
                if segments[i].crosses(&segments[j]) {
                    ref_crossings += 1;
                }
            }
        }
        assert!(ref_crossings > 0, "Expected non-zero crossings in test setup");
        assert_eq!(st.crossings, ref_crossings, "LBVH ratsnest crossings must exactly match brute-force crossings");

        // Compute brute-force reference corridor congestion
        let (shared_nets, _) = build_board_topological_connectivity(&board);
        let num_layers = board.copper_layers().len().max(2);
        let default_nc = board.netclasses.get("Default");
        let track_width = default_nc.map(|nc| nc.track_width).unwrap_or(0.20).max(0.05);
        let min_clearance = default_nc.map(|nc| nc.clearance).unwrap_or(0.20).max(0.05);
        let n = board.footprints.len();
        let mut ref_corridor_overflow = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                let bi = &st.cached_bounds[i];
                let bj = &st.cached_bounds[j];
                let dx = (bi.min_x - bj.max_x).max(bj.min_x - bi.max_x).max(0.0);
                let dy = (bi.min_y - bj.max_y).max(bj.min_y - bi.max_y).max(0.0);
                if dx > 15.0 || dy > 15.0 || (dx * dx + dy * dy) > 225.0 {
                    continue;
                }
                if st.layers[i] != st.layers[j]
                    && board.footprints[i].is_smd_only()
                    && board.footprints[j].is_smd_only()
                {
                    continue;
                }
                let gap_w = if !st.cached_courtyards[i].is_empty() && !st.cached_courtyards[j].is_empty() {
                    courtyards_clearance_gap(&st.cached_courtyards[i], &st.cached_courtyards[j])
                } else {
                    dx.hypot(dy)
                };
                let capacity = compute_corridor_capacity(gap_w, track_width, min_clearance, num_layers);
                let mut demand_nets: Vec<usize> = Vec::new();
                if let Some(shared) = shared_nets.get(&(i, j)) {
                    demand_nets.extend_from_slice(shared);
                }
                let c_i = st.positions[i];
                let c_j = st.positions[j];
                if c_i.distance_to(&c_j) > 1e-4 {
                    let cut_seg = Segment { p1: c_i, p2: c_j, net_id: 0 };
                    for seg in &segments {
                        if cut_seg.crosses(seg) {
                            demand_nets.push(seg.net_id);
                        }
                    }
                }
                demand_nets.sort_unstable();
                demand_nets.dedup();
                let demand = demand_nets.len();
                if demand > capacity {
                    let overflow = (demand - capacity) as f64;
                    ref_corridor_overflow += overflow * overflow;
                }
            }
        }
        assert!(ref_corridor_overflow > 0.0, "Expected non-zero corridor congestion overflow in test setup");
        assert_eq!(
            st.corridor_congestion, ref_corridor_overflow,
            "LBVH corridor congestion overflow must exactly match brute-force loop"
        );
    }

    #[test]
    fn test_gpu_legalize_placement_prepass() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 30.0;
        board.max_y = 30.0;
        board.nets = vec!["".to_string(), "NET_1".to_string(), "NET_2".to_string()];

        let j1 = make_test_fp(
            "J1",
            4.0,
            15.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_1", false)],
        );
        let u1 = make_test_fp(
            "U1",
            15.0,
            15.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.4, 0.4, 1, "NET_1", false),
                Pad::new("2", 0.5, 0.0, 0.4, 0.4, 2, "NET_2", false),
            ],
        );
        let u2 = make_test_fp(
            "U2",
            15.2,
            15.1,
            false,
            vec![Pad::new("1", -0.5, 0.0, 0.4, 0.4, 2, "NET_2", false)],
        );
        let u3 = make_test_fp(
            "U3",
            14.9,
            15.0,
            false,
            vec![Pad::new("1", 0.5, 0.0, 0.4, 0.4, 1, "NET_1", false)],
        );
        board.footprints = vec![j1, u1, u2, u3];

        let mut cfg = PlacementConfig::default();
        cfg.vulkan = VulkanEngine::new();
        let initial = PlacementState::from_board(&board, &cfg);
        assert!(initial.overlaps > 0, "Expected initial overlaps before legalization");

        let legalized = legalize_placement(&board, &cfg, initial, None);
        assert_eq!(legalized.overlaps, 0, "Expected 0 overlaps after GPU+CPU legalization");
        assert_eq!(
            legalized.boundary_violations, 0,
            "Expected 0 boundary violations after legalization"
        );
    }

    #[test]
    fn test_gpu_sa_corridor_and_flow_alignment_terms() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 40.0;
        board.max_y = 30.0;
        board.nets = vec![
            "".to_string(),
            "BUS_1".to_string(),
            "BUS_2".to_string(),
            "CROSS_NET".to_string(),
        ];

        let j1 = make_test_fp(
            "J1",
            10.0,
            15.0,
            true,
            vec![
                Pad::new("1", 0.0, -0.8, 0.4, 0.4, 1, "BUS_1", false),
                Pad::new("2", 0.0, 0.8, 0.4, 0.4, 2, "BUS_2", false),
            ],
        );
        let j2 = make_test_fp(
            "J2",
            30.0,
            15.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 1, "BUS_1", false)],
        );
        let u1 = make_test_fp(
            "U1",
            20.0,
            8.0,
            false,
            vec![
                Pad::new("1", 0.0, -0.8, 0.4, 0.4, 1, "BUS_1", false),
                Pad::new("2", 0.0, 0.8, 0.4, 0.4, 2, "BUS_2", false),
                Pad::new("3", 0.5, 0.0, 0.4, 0.4, 3, "CROSS_NET", false),
            ],
        );
        let u2 = make_test_fp(
            "U2",
            20.0,
            22.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "CROSS_NET", false)],
        );
        board.footprints = vec![j1, j2, u1, u2];

        let cfg = PlacementConfig {
            corridor_congestion_weight: 500.0,
            flow_alignment_weight: 400.0,
            ..PlacementConfig::default()
        };
        let ties = build_passive_ties(&board);
        let base = PlacementState::from_board(&board, &cfg);
        let (fp_descs, edges, courtyards) = build_gpu_sa_descriptors(&board, &cfg, &ties, &base);

        // Verify virtual corridor edges (tag == 3.0) and flow alignment edges (tag == 4.0) are emitted
        let has_corridor_edge = edges.iter().any(|e| (e[3] - 3.0).abs() < 1e-4);
        let has_flow_edge = edges.iter().any(|e| (e[3] - 4.0).abs() < 1e-4);
        assert!(has_corridor_edge, "Expected virtual corridor edge (tag 3.0) in GPU SA descriptors");
        assert!(has_flow_edge, "Expected virtual flow alignment edge (tag 4.0) in GPU SA descriptors");

        if let Some(vk) = VulkanEngine::new() {
            let nf = board.footprints.len();
            let ne = edges.len() / 2;
            let num_chains = 64u32;
            let mut poses = Vec::with_capacity((num_chains as usize) * nf);
            for _ in 0..num_chains {
                for i in 0..nf {
                    poses.push([
                        base.positions[i].x as f32,
                        base.positions[i].y as f32,
                        base.rotations[i] as f32,
                        base.layers[i] as f32,
                    ]);
                }
            }
            let mut params_zero = GpuSaParams {
                num_chains,
                num_footprints: nf as u32,
                num_edges: ne as u32,
                steps_per_pass: 0,
                seed: 42,
                _pad0: 0,
                corridor_congestion_weight: 0.0,
                flow_alignment_weight: 0.0,
                board_min_x: 0.0,
                board_min_y: 0.0,
                board_max_x: 40.0,
                board_max_y: 30.0,
                grid_step: 0.5,
                wirelength_weight: 1.0,
                crossing_weight: 200.0,
                overlap_penalty: 25000.0,
                boundary_penalty: 50000.0,
                passive_affinity_weight: 80.0,
                passive_max_dist: 3.5,
                base_temp: 5.0,
            };
            let (_, scores_zero) = vk
                .run_gpu_sa_placer_chains(&fp_descs, &edges, &poses, &params_zero, 1, Some(&courtyards))
                .expect("GPU SA zero weights eval");

            params_zero.corridor_congestion_weight = 500.0;
            params_zero.flow_alignment_weight = 400.0;
            let (_, scores_active) = vk
                .run_gpu_sa_placer_chains(&fp_descs, &edges, &poses, &params_zero, 1, Some(&courtyards))
                .expect("GPU SA active weights eval");

            assert!(
                scores_active[0][0] > scores_zero[0][0] + 500.0,
                "Expected GPU SA cost with corridor & flow weights ({}) to exceed zero-weight cost ({})",
                scores_active[0][0],
                scores_zero[0][0]
            );
        }
    }

    #[test]
    fn test_gpu_holding_palette_batch_evaluation() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 40.0;
        board.max_y = 40.0;
        board.nets = vec![
            "".to_string(),
            "GND".to_string(),
            "+3V3".to_string(),
            "SIG_A".to_string(),
        ];

        let u1 = make_test_fp(
            "U1",
            20.0,
            20.0,
            true,
            vec![
                Pad::new("1", -1.0, 0.0, 0.4, 0.4, 2, "+3V3", false),
                Pad::new("2", 1.0, 0.0, 0.4, 0.4, 3, "SIG_A", false),
                Pad::new("3", 0.0, 1.0, 0.4, 0.4, 1, "GND", false),
            ],
        );
        board.footprints = vec![u1];

        let c1 = make_test_fp(
            "C1",
            -10.0,
            -10.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.4, 0.4, 2, "+3V3", false),
                Pad::new("2", 0.5, 0.0, 0.4, 0.4, 1, "GND", false),
            ],
        );
        let r1 = make_test_fp(
            "R1",
            -10.0,
            -15.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.4, 0.4, 3, "SIG_A", false),
                Pad::new("2", 0.5, 0.0, 0.4, 0.4, 1, "GND", false),
            ],
        );
        board.holding_palette = vec![c1, r1];

        let mut cfg = PlacementConfig::default();
        cfg.vulkan = VulkanEngine::new();

        if let Some(ref vk) = cfg.vulkan {
            let candidates = vec![
                (Point::new(17.5, 20.0), 0.0, 0),
                (Point::new(35.0, 35.0), 0.0, 0),
            ];
            let targets = vec![(0usize, Point::new(19.0, 20.0), 1.0)];
            let scores = score_candidate_poses_gpu(
                vk,
                &board,
                &board.holding_palette[0],
                &candidates,
                &targets,
                &cfg,
            )
            .expect("GPU batch candidate scores");
            assert_eq!(scores.len(), 2);
            assert!(
                scores[0] < scores[1],
                "Closer pose must have lower GPU batch cost ({} vs {})",
                scores[0],
                scores[1]
            );
        }

        let placed_count = place_all_holding_components_at_best_locations(&mut board, &cfg);
        assert_eq!(placed_count, 2);
        assert!(board.holding_palette.is_empty());
        assert_eq!(board.footprints.len(), 3);

        let st = PlacementState::from_board(&board, &cfg);
        assert_eq!(st.overlaps, 0);
        assert_eq!(st.boundary_violations, 0);
    }

    #[test]
    fn test_gpu_sa_descriptors_edge_ordering_and_centered_rect_flags() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 50.0;
        board.max_y = 40.0;
        board.nets = vec![
            "".to_string(),
            "GND".to_string(),
            "VCC".to_string(),
            "SIG1".to_string(),
            "SIG2".to_string(),
        ];

        // U1: centered rectangular courtyard -> bit 1 (non-centered-rect) must be 0
        let u1 = make_test_fp(
            "U1",
            15.0,
            20.0,
            false,
            vec![
                Pad::new("1", -1.5, -1.0, 0.5, 0.5, 2, "VCC", false),
                Pad::new("2", -1.5, 1.0, 0.5, 0.5, 1, "GND", false),
                Pad::new("3", 1.5, -1.0, 0.5, 0.5, 3, "SIG1", false),
                Pad::new("4", 1.5, 1.0, 0.5, 0.5, 4, "SIG2", false),
            ],
        );

        // C1: decoupling cap with centered rectangular courtyard -> bit 1 must be 0, bit 0 (is_smd) must be 1
        let mut c1 = make_test_fp(
            "C1",
            22.0,
            20.0,
            false,
            vec![
                Pad::new("1", -0.5, 0.0, 0.5, 0.5, 2, "VCC", false),
                Pad::new("2", 0.5, 0.0, 0.5, 0.5, 1, "GND", false),
            ],
        );
        c1.value = "100nF".to_string();

        // J1: non-rectangular / offset courtyard -> bit 1 (2u32) must be set
        let mut j1 = make_test_fp(
            "J1",
            35.0,
            20.0,
            false,
            vec![
                Pad::new("1", 0.0, -1.0, 0.8, 0.8, 3, "SIG1", true),
                Pad::new("2", 0.0, 1.0, 0.8, 0.8, 4, "SIG2", true),
            ],
        );
        j1.courtyards = vec![Polygon::new(vec![
            Point::new(-1.0, -3.0),
            Point::new(3.0, -3.0),
            Point::new(3.0, 3.0),
            Point::new(0.0, 4.0),
            Point::new(-1.0, 3.0),
        ])];

        board.footprints = vec![u1, c1, j1];
        let cfg = PlacementConfig::default();
        let ties = build_passive_ties(&board);
        let state = PlacementState::from_board(&board, &cfg);
        let (_fp_descs, edges, courtyards) = build_gpu_sa_descriptors(&board, &cfg, &ties, &state);

        assert_eq!(courtyards.fp_poly_ranges.len(), 3);
        // U1 and C1 are SMD (bit 0 = 1) and centered rectangles (bit 1 = 0)
        assert_eq!(courtyards.fp_poly_ranges[0][2] & 1, 1);
        assert_eq!(courtyards.fp_poly_ranges[0][2] & 2, 0);
        assert_eq!(courtyards.fp_poly_ranges[1][2] & 1, 1);
        assert_eq!(courtyards.fp_poly_ranges[1][2] & 2, 0);
        // J1 has THT pads (bit 0 = 0) and a 5-vertex non-centered courtyard (bit 1 = 2)
        assert_eq!(courtyards.fp_poly_ranges[2][2] & 1, 0);
        assert_eq!(courtyards.fp_poly_ranges[2][2] & 2, 2);

        // Verify edge tag ordering: all tag 0.0 (airwires) and tag 3.0 (corridor cuts)
        // must precede tag 1.0/2.0 (passive ties) and tag 4.0 (flow edges) so GPU inner loops can break early.
        let mut seen_non_crossing_tag = false;
        for pair in edges.chunks_exact(2) {
            let tag = pair[0][3];
            let is_airwire_or_corridor = tag < 0.5 || (tag > 2.5 && tag < 3.5);
            if !is_airwire_or_corridor {
                seen_non_crossing_tag = true;
            } else {
                assert!(
                    !seen_non_crossing_tag,
                    "Airwire/corridor edge (tag={}) appeared after passive/flow edge",
                    tag
                );
            }
        }
    }
}


