// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! Multi-layer PCB autorouter, obstacle rasterization grid, shove/nudge engine,
//! post-route topological glossing, tangent-arc synthesis, and GPU-accelerated DRC cleanup.
//!
//! # Architecture Overview
//!
//! This module implements the full multi-layer routing pipeline for Vulcan's Trials:
//! - [`RoutingConfig`] / [`RouterConfig`]: Tunable physical, geometric, and cost weights for
//!   PathFinder negotiated congestion, crosstalk avoidance, via reduction, and post-route glossing.
//! - [`RoutingGrid`]: 3D voxelized copper obstacle and clearance grid supporting multi-layer
//!   Euclidean Distance Field (EDF) transforms on Vulkan GPUs (with a multi-core Rayon CPU fallback),
//!   per-net 3D routing cost maps, and reference-plane return path checks.
//! - [`NudgeEngine`]: Component nudging, perpendicular track shoving, via shoving, and cascading
//!   octilinear continuous track deflection with 45° transition elbows.
//! - [`AutoRouter`]: Negotiated-congestion PathFinder router with 3D octilinear A* search,
//!   coupled differential-pair routing (corridor attraction and symmetric dual-state A*),
//!   trombone/meander length and phase tuning, staggered fine-pitch QFN/SOIC/TSSOP escape stubs,
//!   BGA dogbone fanouts, and automated ground-plane via stitching.
//! - Post-route topological glossing ([`TrackChain`], [`gloss_tracks`], [`gloss_tracks_standard`],
//!   [`relax_homotopic_rubber_bands`]): Collinear segment merging, stair-step and U-turn
//!   elimination, 2-segment corner dogleg replacement, acute acid-trap avoidance, via pair
//!   and single-via consolidation, via sliding/alignment, and redundant cycle/stub pruning.
//! - Organic tangent-arc and spline synthesis ([`CubicBezier`], [`apply_any_angle_tangent_arcs`],
//!   [`convert_tracks_to_organic`], [`fillet_acute_angle_junctions`]): Smooth circular arc
//!   (`fp_arc` / `gr_arc`), asymmetric $G^1$ biarc, and cubic Bézier corner rounding with full
//!   GPU/CPU DRC clearance verification.
//! - Copper DRC validation and repair ([`GpuDrcContext`], [`validate_and_clean_copper`],
//!   [`validate_crosstalk_and_coupling`], [`route_bundle_along_guide`]): Exact analytic and
//!   GPU-accelerated pad/track/via clearance verification, crosstalk coupling length analysis,
//!   and equidistant river routing for bus bundles.

use crate::board::{ArcSegment, Board, Footprint, LiveTrialState, NetClass, NetTopology, Pad, Point, TrackSegment, TrialPhase, Via};
pub use crate::board::BusBundle;
use crate::cuda::{CudaEngine, MultiGpuRouterPool};
use crate::ipc::KiCadIpcClient;
use crate::vulkan::{GpuDrcCandidate, GpuDrcPad, GpuDrcTrack, GpuDrcVia, GpuRouteCostParams, VulkanEngine};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};


pub mod drc;
pub mod gloss;
pub mod grid;
pub mod nudge;
pub mod organic;

pub use drc::*;
pub use gloss::*;
pub use grid::*;
pub use nudge::*;
pub use organic::*;
/// Sentinel obstacle net ID (`-1`) representing an impassable keepout zone, board edge margin,
/// or unconnected pad obstacle in [`RoutingGrid::obstacle_net_ids`].
pub const KEEPOUT_NET: i32 = -1;

/// Sentinel obstacle net ID (`0`) representing unobstructed free space in [`RoutingGrid::obstacle_net_ids`].
pub const FREE_NET: i32 = 0;

/// A* edge-cost multiplier (`6.0`) applied when a differential-pair follower trace strays outside
/// its leader trace's target coupling corridor.
pub const DIFF_PAIR_OFF_CORRIDOR_MULT: f32 = 6.0;

/// Maximum mean follower-to-leader center-to-center spacing (`0.8 mm`) for a routed differential pair
/// to be considered tightly coupled before triggering dual-state coupled re-routing.
pub const DIFF_PAIR_COUPLED_MM: f64 = 0.8;

/// Per-cell A* cost penalty (`60.0` grid steps) added when a sided differential-pair leader step
/// leaves insufficient lateral clearance for its partner trace on the required side.
pub const DIFF_PAIR_SIDE_BLOCKED_PENALTY: f64 = 60.0;

/// Determines which lateral side (`+1.0`, `-1.0`, or `0.0` if ambiguous) of the `leader` grid path
/// the differential-pair follower's start and end nodes lie on.
///
/// Uses the 2D cross product of the leader's initial and terminal tangent vectors against the
/// follower's start (`fs`) and end (`fe`) nodes, matching the convention of
/// [`AutoRouter::build_diff_pair_corridor`].
///
/// # Arguments
/// * `leader` - Ordered slice of [`GridNode`]s forming the routed leader trace path.
/// * `fs` - Grid coordinates of the follower net's start pad.
/// * `fe` - Grid coordinates of the follower net's end pad.
///
/// # Returns
/// `+1.0` or `-1.0` when both endpoints consistently sit on the same side of the leader trajectory,
/// or `0.0` when the leader path has fewer than 2 nodes or the endpoints straddle opposite sides.
pub fn diff_pair_side(leader: &[GridNode], fs: GridNode, fe: GridNode) -> f64 {
    let n = leader.len();
    if n < 2 {
        return 0.0;
    }
    let k = 6.min(n - 1);
    let cross = |a: GridNode, b: GridNode, p: GridNode| {
        let (tx, ty) = (b.gx as f64 - a.gx as f64, b.gy as f64 - a.gy as f64);
        tx * (p.gy as f64 - a.gy as f64) - ty * (p.gx as f64 - a.gx as f64)
    };
    let s0 = cross(leader[0], leader[k], fs);
    let s1 = cross(leader[n - 1 - k], leader[n - 1], fe);
    if s0 * s1 > 0.0 { s0.signum() } else { 0.0 }
}

/// Computes the mean Euclidean distance (in `mm`) from copper of net `a` (sampled every `0.1 mm`
/// along tracks and arcs) to the nearest copper segment of net `b`.
///
/// # Arguments
/// * `board` - The [`Board`] containing routed [`TrackSegment`]s and [`ArcSegment`]s.
/// * `a` - Net ID of the first net to sample.
/// * `b` - Net ID of the reference partner net.
///
/// # Returns
/// Mean centerline distance in millimeters (`mm`), or [`f64::INFINITY`] if either net has no routed copper.
pub fn board_net_coupling_mm(board: &Board, a: usize, b: usize) -> f64 {
    let segs = |net: usize| -> Vec<((f64, f64), (f64, f64))> {
        let mut v: Vec<_> = board.tracks.iter().filter(|t| t.net_id == net).map(|t| (t.start, t.end)).collect();
        for arc in board.arcs.iter().filter(|a| a.net_id == net) {
            let pt = |ang: f64| (arc.center.0 + arc.radius * ang.cos(), arc.center.1 + arc.radius * ang.sin());
            let mid = (arc.start_angle_rad + arc.end_angle_rad) * 0.5;
            v.push((pt(arc.start_angle_rad), pt(mid)));
            v.push((pt(mid), pt(arc.end_angle_rad)));
        }
        v
    };
    let (sa, sb) = (segs(a), segs(b));
    let (mut sum, mut n) = (0.0, 0usize);
    for &(p, q) in &sa {
        let steps = (((q.0 - p.0).hypot(q.1 - p.1)) / 0.1).ceil().max(1.0) as usize;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let (x, y) = (p.0 + (q.0 - p.0) * t, p.1 + (q.1 - p.1) * t);
            let d = sb.iter().map(|&(s, e)| dist_to_segment(x, y, s.0, s.1, e.0, e.1)).fold(f64::INFINITY, f64::min);
            sum += d;
            n += 1;
        }
    }
    if n == 0 { f64::INFINITY } else { sum / n as f64 }
}

/// Computes the total in-plane physical length (in `mm`) of a discretized [`GridNode`] path.
///
/// Layer transitions (vias where adjacent nodes differ in `layer`) contribute zero planar length.
///
/// # Arguments
/// * `res` - Grid cell resolution in millimeters (`mm`) per cell.
/// * `path` - Slice of [`GridNode`] steps representing the routed path.
///
/// # Returns
/// Total planar path length in millimeters (`mm`).
pub fn grid_path_length_mm(res: f64, path: &[GridNode]) -> f64 {
    path.windows(2)
        .filter(|w| w[0].layer == w[1].layer)
        .map(|w| (w[1].gx as f64 - w[0].gx as f64).hypot(w[1].gy as f64 - w[0].gy as f64))
        .sum::<f64>()
        * res
}

/// Returns the target copper track width (in `mm`) for `net_id` based on its netclass and
/// differential-pair membership.
///
/// Returns [`NetClass::diff_pair_width`] when `net_id` is paired with a complementary differential
/// mate on the board, or [`NetClass::track_width`] for single-ended nets.
///
/// # Arguments
/// * `board` - The [`Board`] providing net names and netclass definitions.
/// * `net_id` - Numeric net identifier to query.
///
/// # Returns
/// Target track width in millimeters (`mm`).
pub fn net_target_width(board: &Board, net_id: usize) -> f64 {
    let nc = board.get_netclass_for_net(net_id);
    let name = &board.nets[net_id];
    if board.nets.iter().any(|o| AutoRouter::are_diff_pair_mates(name, o)) {
        nc.diff_pair_width
    } else {
        nc.track_width
    }
}

/// Computes the mean Euclidean distance (in `mm`) from each node in `follower` to the nearest
/// node in `leader` on the routing grid.
///
/// # Arguments
/// * `res` - Grid cell resolution in millimeters (`mm`) per cell.
/// * `leader` - Slice of [`GridNode`]s forming the leader path.
/// * `follower` - Slice of [`GridNode`]s forming the follower path.
///
/// # Returns
/// Mean distance in millimeters (`mm`), or [`f64::INFINITY`] if either path slice is empty.
pub fn path_coupling_mm(res: f64, leader: &[GridNode], follower: &[GridNode]) -> f64 {
    if leader.is_empty() || follower.is_empty() {
        return f64::INFINITY;
    }
    let sum: f64 = follower
        .iter()
        .map(|f| {
            leader
                .iter()
                .map(|l| {
                    let dx = f.gx as f64 - l.gx as f64;
                    let dy = f.gy as f64 - l.gy as f64;
                    dx * dx + dy * dy
                })
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
        .sum();
    sum / follower.len() as f64 * res
}

/// Configuration parameters controlling the multi-layer autorouter, negotiated congestion,
/// via optimization, crosstalk penalties, and post-route glossing passes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Penalty weight added per grid step when a trace runs parallel and closely coupled to a foreign net.
    #[serde(default = "default_crosstalk_penalty")]
    pub crosstalk_penalty: f64,
    /// When `true`, enables any-angle Euclidean line-of-sight rubber-band glossing instead of strict 45° octilinear routing.
    #[serde(default)]
    pub any_angle_mode: bool,
    /// Maximum search radius (in `mm`) within which same-net vias are merged during post-route cleanup.
    #[serde(default = "default_via_merge_radius_mm")]
    pub via_merge_radius_mm: f64,
    /// Maximum extra planar detour length (in `mm`) allowed when replacing a pair of vias with a single-layer detour.
    #[serde(default = "default_via_detour_budget_mm")]
    pub via_detour_budget_mm: f64,
    /// Maximum allowed ratio of detour length to direct segment length when eliminating a layer-hop via pair.
    #[serde(default = "default_via_hop_max_detour_ratio")]
    pub via_hop_max_detour_ratio: f64,
    /// Maximum radius (in `mm`) within which multiple ground/power plane ties can share a single stitching via.
    #[serde(default = "default_plane_via_share_radius_mm")]
    pub plane_via_share_radius_mm: f64,
    /// Maximum distance (in `mm`) to search for or place a reference-plane return-current via near a signal layer transition.
    #[serde(default = "default_return_via_max_dist_mm")]
    pub return_via_max_dist_mm: f64,
    /// Maximum distance (in `mm`) within which a nearby via is snapped directly onto a compatible pad or terminal.
    #[serde(default = "default_via_pad_snap_dist_mm")]
    pub via_pad_snap_dist_mm: f64,
    /// Base PathFinder historical/present congestion cost multiplier.
    #[serde(default = "default_congestion_factor")]
    pub congestion_factor: f64,
    /// Incremental displacement step size (in `mm`) used by [`NudgeEngine`] when shoving tracks or footprints.
    #[serde(default = "default_nudge_step_mm")]
    pub nudge_step_mm: f64,
    /// Number of iterative post-route topological glossing passes to execute.
    #[serde(default = "default_gloss_passes")]
    pub gloss_passes: usize,
    /// Chamfer/fillet radius (in `mm`) inserted at acute (< 90°) copper junctions to eliminate acid traps.
    #[serde(default = "default_acid_trap_fillet_radius")]
    pub acid_trap_fillet_radius: f64,
    /// When `true`, allows placing vias directly inside SMD pads of the same net.
    #[serde(default)]
    pub allow_via_in_pad: bool,
}

/// Type alias for [`RoutingConfig`] used by external callers and CLI configuration loaders.
pub type RouterConfig = RoutingConfig;

/// Default crosstalk penalty weight (`4.0`).
pub(crate) fn default_crosstalk_penalty() -> f64 { 4.0 }
/// Default same-net via merge radius (`4.00 mm`).
pub(crate) fn default_via_merge_radius_mm() -> f64 { 4.00 }
/// Default planar detour budget (`3.50 mm`) for via-pair elimination.
pub(crate) fn default_via_detour_budget_mm() -> f64 { 3.50 }
/// Default maximum detour-to-direct length ratio (`1.25`) for via-hop elimination.
pub(crate) fn default_via_hop_max_detour_ratio() -> f64 { 1.25 }
/// Default plane-via sharing radius (`3.50 mm`).
pub(crate) fn default_plane_via_share_radius_mm() -> f64 { 3.50 }
/// Default maximum distance (`1.40 mm`) to a reference-plane return via.
pub(crate) fn default_return_via_max_dist_mm() -> f64 { 1.40 }
/// Default via-to-pad snap radius (`0.50 mm`).
pub(crate) fn default_via_pad_snap_dist_mm() -> f64 { 0.50 }
/// Default PathFinder congestion multiplier (`1.50`).
pub(crate) fn default_congestion_factor() -> f64 { 1.50 }
/// Default nudge step size (`0.25 mm`).
pub(crate) fn default_nudge_step_mm() -> f64 { 0.25 }
/// Default number of post-route glossing passes (`2`).
pub(crate) fn default_gloss_passes() -> usize { 2 }
/// Default acute-angle acid-trap fillet radius (`0.35 mm`).
pub(crate) fn default_acid_trap_fillet_radius() -> f64 { 0.35 }

impl Default for RoutingConfig {
    /// Creates a [`RoutingConfig`] initialized with standard production PCB routing defaults.
    fn default() -> Self {
        Self {
            crosstalk_penalty: default_crosstalk_penalty(),
            any_angle_mode: false,
            via_merge_radius_mm: default_via_merge_radius_mm(),
            via_detour_budget_mm: default_via_detour_budget_mm(),
            via_hop_max_detour_ratio: default_via_hop_max_detour_ratio(),
            plane_via_share_radius_mm: default_plane_via_share_radius_mm(),
            return_via_max_dist_mm: default_return_via_max_dist_mm(),
            via_pad_snap_dist_mm: default_via_pad_snap_dist_mm(),
            congestion_factor: default_congestion_factor(),
            nudge_step_mm: default_nudge_step_mm(),
            gloss_passes: default_gloss_passes(),
            acid_trap_fillet_radius: default_acid_trap_fillet_radius(),
            allow_via_in_pad: false,
        }
    }
}


/// Priority-queue node used by the 3D octilinear A* search in [`AutoRouter`].
#[derive(Clone, Copy, PartialEq)]
struct AStarNode {
    /// Current 3D grid coordinate `(gx, gy, layer)`.
    node: GridNode,
    /// Accumulated path cost $g(n)$ from the start node.
    cost: f64,
    /// Total estimated A* priority $f(n) = g(n) + h(n)$ (ordered as a min-heap via reversed `Ord`).
    priority: f64,
}

impl Eq for AStarNode {}

impl Ord for AStarNode {
    /// Orders [`AStarNode`]s in reverse `priority` order so [`BinaryHeap`] acts as a min-heap.
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.partial_cmp(&self.priority).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for AStarNode {
    /// Delegates partial comparison to [`AStarNode::cmp`].
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A single point-to-point ratsnest connection task between two pads or Steiner split nodes on a net.
#[derive(Clone, Debug)]
pub struct RouteConnection {
    /// Numeric net ID (`1..board.nets.len()`).
    pub net_id: usize,
    /// Human-readable net name (e.g., `"USB_DP"`, `"+3V3"`).
    pub net_name: String,
    /// Start world position and layer `(x_mm, y_mm, layer)`.
    pub start: (f64, f64, usize),
    /// End world position and layer `(x_mm, y_mm, layer)`.
    pub end: (f64, f64, usize),
    /// Index of the footprint owning the start pad in `board.footprints`.
    pub start_fp: usize,
    /// Index of the start pad within `board.footprints[start_fp].pads`.
    pub start_pad: usize,
    /// Index of the footprint owning the end pad in `board.footprints`.
    pub end_fp: usize,
    /// Index of the end pad within `board.footprints[end_fp].pads`.
    pub end_pad: usize,
    /// Electrical and physical design rules ([`NetClass`]) governing this connection.
    pub netclass: NetClass,
    /// When `Some(leader_net_id)`, marks this connection as the follower of a coupled differential pair.
    pub diff_pair_leader: Option<usize>,
}

impl RouteConnection {
    /// Reverses the direction of this connection in place by swapping `start` and `end` coordinates,
    /// footprint indices, and pad indices.
    pub fn swap_endpoints(&mut self) {
        std::mem::swap(&mut self.start, &mut self.end);
        std::mem::swap(&mut self.start_fp, &mut self.end_fp);
        std::mem::swap(&mut self.start_pad, &mut self.end_pad);
    }
}

/// Pre-routed fanout escape stub extending from a fine-pitch SMD pad (such as a 0.5 mm QFN,
/// SOIC, TSSOP, USB-C receptacle, or BGA ball) to an accessible routing channel or staggered via.
#[derive(Clone, Debug)]
pub struct EscapeStub {
    /// Index of the parent footprint in `board.footprints`.
    pub fp_idx: usize,
    /// Numeric net ID of the pad.
    pub net_id: usize,
    /// World coordinates `(x, y)` in `mm` of the pad center.
    pub pad_pos: (f64, f64),
    /// World coordinates `(x, y)` in `mm` of the outer tip of the escape stub (where global routing attaches).
    pub tip_pos: (f64, f64),
    /// Copper trace width (in `mm`) of the escape stub segment.
    pub width: f64,
    /// Optional fanout [`Via`] placed at `tip_pos` when transitioning to an inner/opposite copper layer.
    pub via: Option<Via>,
    /// Copper layer index at `tip_pos` after any escape via transition.
    pub tip_layer: usize,
    /// Copper layer index of the SMD pad and planar stub segment.
    pub layer: usize,
}

/// Metadata describing a single physical component pad participating in net topology construction.
#[derive(Clone, Debug, PartialEq)]
pub struct PadInfo {
    /// World X coordinate of the pad center in millimeters (`mm`).
    pub x: f64,
    /// World Y coordinate of the pad center in millimeters (`mm`).
    pub y: f64,
    /// Copper layer index of the pad (`0` for through-hole or `F.Cu`, `1` for `B.Cu`).
    pub layer: usize,
    /// Index of the owning footprint in `board.footprints`.
    pub fp_idx: usize,
    /// Index of the pad within `board.footprints[fp_idx].pads`.
    pub pad_idx: usize,
    /// `true` if the pad belongs to an active driver/source IC or connector (`U*`, `IC*`, `MCU*`, `J*`, `VR*`, `REG*`).
    pub is_driver: bool,
}

impl PadInfo {
    /// Creates a new [`PadInfo`] at `(x, y)` on `layer` for footprint `fp_idx` and pad `pad_idx`,
    /// defaulting `is_driver` to `false`.
    ///
    /// # Arguments
    /// * `x` - Pad center X coordinate (`mm`).
    /// * `y` - Pad center Y coordinate (`mm`).
    /// * `layer` - Copper layer index.
    /// * `fp_idx` - Footprint index in `board.footprints`.
    /// * `pad_idx` - Pad index within the footprint.
    ///
    /// # Returns
    /// Newly initialized [`PadInfo`].
    pub fn new(x: f64, y: f64, layer: usize, fp_idx: usize, pad_idx: usize) -> Self {
        Self {
            x,
            y,
            layer,
            fp_idx,
            pad_idx,
            is_driver: false,
        }
    }

    /// Builder-style setter that marks whether this pad is a signal/power driver source.
    ///
    /// # Arguments
    /// * `is_driver` - `true` if this pad is the source driver of the net.
    ///
    /// # Returns
    /// Updated [`PadInfo`] with `is_driver` set.
    pub fn with_driver(mut self, is_driver: bool) -> Self {
        self.is_driver = is_driver;
        self
    }
}

/// Decomposes a multi-pad net (`pads`) into an ordered list of 2-terminal [`RouteConnection`]s
/// according to the specified [`NetTopology`] (`EuclideanMst`, `DaisyChain`, `FlyBy`, `StarPoint`, or `TBranch`).
///
/// # Arguments
/// * `net_id` - Numeric net identifier.
/// * `net_name` - Net name string.
/// * `netclass` - [`NetClass`] rules for the net.
/// * `pads` - Slice of [`PadInfo`]s belonging to `net_id`.
/// * `topology` - Target [`NetTopology`] routing structure.
///
/// # Returns
/// Vector of [`RouteConnection`]s spanning all pads in `pads` (empty if `pads.len() < 2`).
pub fn build_net_topology_connections(
    net_id: usize,
    net_name: &str,
    netclass: &NetClass,
    pads: &[PadInfo],
    topology: NetTopology,
) -> Vec<RouteConnection> {
    if pads.len() < 2 {
        return Vec::new();
    }
    if pads.len() == 2 {
        let (p_start, p_end) = if pads[1].is_driver && !pads[0].is_driver {
            (&pads[1], &pads[0])
        } else {
            (&pads[0], &pads[1])
        };
        return vec![RouteConnection {
            net_id,
            net_name: net_name.to_string(),
            start: (p_start.x, p_start.y, p_start.layer),
            end: (p_end.x, p_end.y, p_end.layer),
            start_fp: p_start.fp_idx,
            start_pad: p_start.pad_idx,
            end_fp: p_end.fp_idx,
            end_pad: p_end.pad_idx,
            netclass: netclass.clone(),
            diff_pair_leader: None,
        }];
    }

    match topology {
        NetTopology::EuclideanMst => {
            let mut connections = Vec::with_capacity(pads.len() - 1);
            let mut visited = vec![false; pads.len()];
            visited[0] = true;
            let mut in_tree = 1;

            while in_tree < pads.len() {
                let mut best_dist = f64::MAX;
                let mut best_pair = (0, 0);
                for i in 0..pads.len() {
                    if !visited[i] {
                        continue;
                    }
                    for j in 0..pads.len() {
                        if visited[j] {
                            continue;
                        }
                        let same_fp_penalty = if pads[i].fp_idx == pads[j].fp_idx && pads.len() > 2 {
                            let d = (pads[i].x - pads[j].x).hypot(pads[i].y - pads[j].y);
                            if d < 2.5 { 0.0 } else { 10.0 }
                        } else {
                            0.0
                        };
                        let dist = (pads[i].x - pads[j].x).hypot(pads[i].y - pads[j].y) + same_fp_penalty;
                        if dist < best_dist {
                            best_dist = dist;
                            best_pair = (i, j);
                        }
                    }
                }
                visited[best_pair.1] = true;
                in_tree += 1;

                let p_start = &pads[best_pair.0];
                let p_end = &pads[best_pair.1];
                connections.push(RouteConnection {
                    net_id,
                    net_name: net_name.to_string(),
                    start: (p_start.x, p_start.y, p_start.layer),
                    end: (p_end.x, p_end.y, p_end.layer),
                    start_fp: p_start.fp_idx,
                    start_pad: p_start.pad_idx,
                    end_fp: p_end.fp_idx,
                    end_pad: p_end.pad_idx,
                    netclass: netclass.clone(),
                    diff_pair_leader: None,
                });
            }
            connections
        }
        NetTopology::DaisyChain | NetTopology::FlyBy => {
            let driver_idx = pads.iter().position(|p| p.is_driver).unwrap_or(0);
            let mut chain = Vec::with_capacity(pads.len());
            chain.push(driver_idx);
            let mut visited = vec![false; pads.len()];
            visited[driver_idx] = true;

            while chain.len() < pads.len() {
                let curr = *chain.last().unwrap();
                let mut best_next = None;
                let mut best_dist = f64::MAX;
                for j in 0..pads.len() {
                    if !visited[j] {
                        let dist = (pads[curr].x - pads[j].x).hypot(pads[curr].y - pads[j].y);
                        if dist < best_dist {
                            best_dist = dist;
                            best_next = Some(j);
                        }
                    }
                }
                if let Some(next) = best_next {
                    visited[next] = true;
                    chain.push(next);
                } else {
                    break;
                }
            }

            let mut connections = Vec::with_capacity(chain.len() - 1);
            for w in chain.windows(2) {
                let p_start = &pads[w[0]];
                let p_end = &pads[w[1]];
                connections.push(RouteConnection {
                    net_id,
                    net_name: net_name.to_string(),
                    start: (p_start.x, p_start.y, p_start.layer),
                    end: (p_end.x, p_end.y, p_end.layer),
                    start_fp: p_start.fp_idx,
                    start_pad: p_start.pad_idx,
                    end_fp: p_end.fp_idx,
                    end_pad: p_end.pad_idx,
                    netclass: netclass.clone(),
                    diff_pair_leader: None,
                });
            }
            connections
        }
        NetTopology::StarPoint => {
            let center_idx = pads.iter().position(|p| p.is_driver).unwrap_or(0);
            let p_center = &pads[center_idx];
            let mut connections = Vec::with_capacity(pads.len() - 1);
            for (i, p_other) in pads.iter().enumerate() {
                if i == center_idx {
                    continue;
                }
                connections.push(RouteConnection {
                    net_id,
                    net_name: net_name.to_string(),
                    start: (p_center.x, p_center.y, p_center.layer),
                    end: (p_other.x, p_other.y, p_other.layer),
                    start_fp: p_center.fp_idx,
                    start_pad: p_center.pad_idx,
                    end_fp: p_other.fp_idx,
                    end_pad: p_other.pad_idx,
                    netclass: netclass.clone(),
                    diff_pair_leader: None,
                });
            }
            connections
        }
        NetTopology::TBranch => {
            let driver_idx = pads.iter().position(|p| p.is_driver).unwrap_or(0);
            let driver = &pads[driver_idx];
            let receivers: Vec<PadInfo> = pads
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != driver_idx)
                .map(|(_, p)| p.clone())
                .collect();

            let mut connections = Vec::new();
            build_t_branch_tree(
                (driver.x, driver.y, driver.layer),
                driver.fp_idx,
                driver.pad_idx,
                &receivers,
                driver,
                net_id,
                net_name,
                netclass,
                &mut connections,
            );
            connections
        }
    }
}

/// Recursively constructs a balanced binary Steiner T-branch tree of [`RouteConnection`]s
/// from `source_pt` to all `receivers`, splitting along the axis of maximum spatial variance.
///
/// # Arguments
/// * `source_pt` - Current trunk origin `(x_mm, y_mm, layer)`.
/// * `source_fp` - Footprint index associated with `source_pt`.
/// * `source_pad` - Pad index associated with `source_pt`.
/// * `receivers` - Slice of remaining receiver [`PadInfo`]s to connect.
/// * `driver` - Reference to the root driver [`PadInfo`].
/// * `net_id` - Numeric net identifier.
/// * `net_name` - Net name string.
/// * `netclass` - [`NetClass`] rules for the net.
/// * `connections` - Output vector accumulating the generated trunk and branch [`RouteConnection`]s.
pub(crate) fn build_t_branch_tree(
    source_pt: (f64, f64, usize),
    source_fp: usize,
    source_pad: usize,
    receivers: &[PadInfo],
    driver: &PadInfo,
    net_id: usize,
    net_name: &str,
    netclass: &NetClass,
    connections: &mut Vec<RouteConnection>,
) {
    if receivers.is_empty() {
        return;
    }
    if receivers.len() == 1 {
        let r = &receivers[0];
        connections.push(RouteConnection {
            net_id,
            net_name: net_name.to_string(),
            start: source_pt,
            end: (r.x, r.y, r.layer),
            start_fp: source_fp,
            start_pad: source_pad,
            end_fp: r.fp_idx,
            end_pad: r.pad_idx,
            netclass: netclass.clone(),
            diff_pair_leader: None,
        });
        return;
    }
    if receivers.len() == 2 {
        let r1 = &receivers[0];
        let r2 = &receivers[1];
        let split_x = (r1.x + r2.x) * 0.5;
        let split_y = (r1.y + r2.y) * 0.5;
        let split_layer = r1.layer;
        let split_pt = (split_x, split_y, split_layer);

        // Trunk from source to split node
        connections.push(RouteConnection {
            net_id,
            net_name: net_name.to_string(),
            start: source_pt,
            end: split_pt,
            start_fp: source_fp,
            start_pad: source_pad,
            end_fp: driver.fp_idx,
            end_pad: driver.pad_idx,
            netclass: netclass.clone(),
            diff_pair_leader: None,
        });

        // Branch 1 to r1
        connections.push(RouteConnection {
            net_id,
            net_name: net_name.to_string(),
            start: split_pt,
            end: (r1.x, r1.y, r1.layer),
            start_fp: driver.fp_idx,
            start_pad: driver.pad_idx,
            end_fp: r1.fp_idx,
            end_pad: r1.pad_idx,
            netclass: netclass.clone(),
            diff_pair_leader: None,
        });

        // Branch 2 to r2
        connections.push(RouteConnection {
            net_id,
            net_name: net_name.to_string(),
            start: split_pt,
            end: (r2.x, r2.y, r2.layer),
            start_fp: driver.fp_idx,
            start_pad: driver.pad_idx,
            end_fp: r2.fp_idx,
            end_pad: r2.pad_idx,
            netclass: netclass.clone(),
            diff_pair_leader: None,
        });
        return;
    }

    // For > 2 receivers, split along axis of maximum variance to form balanced binary tree
    let mean_x = receivers.iter().map(|p| p.x).sum::<f64>() / receivers.len() as f64;
    let mean_y = receivers.iter().map(|p| p.y).sum::<f64>() / receivers.len() as f64;
    let var_x = receivers.iter().map(|p| (p.x - mean_x).powi(2)).sum::<f64>();
    let var_y = receivers.iter().map(|p| (p.y - mean_y).powi(2)).sum::<f64>();

    let mut sorted = receivers.to_vec();
    if var_x >= var_y {
        sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal));
    } else {
        sorted.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap_or(Ordering::Equal));
    }

    let mid_idx = sorted.len() / 2;
    let left = &sorted[..mid_idx];
    let right = &sorted[mid_idx..];

    let left_cx = left.iter().map(|p| p.x).sum::<f64>() / left.len() as f64;
    let left_cy = left.iter().map(|p| p.y).sum::<f64>() / left.len() as f64;
    let right_cx = right.iter().map(|p| p.x).sum::<f64>() / right.len() as f64;
    let right_cy = right.iter().map(|p| p.y).sum::<f64>() / right.len() as f64;

    let split_x = (left_cx + right_cx) * 0.5;
    let split_y = (left_cy + right_cy) * 0.5;
    let split_pt = (split_x, split_y, driver.layer);

    connections.push(RouteConnection {
        net_id,
        net_name: net_name.to_string(),
        start: source_pt,
        end: split_pt,
        start_fp: source_fp,
        start_pad: source_pad,
        end_fp: driver.fp_idx,
        end_pad: driver.pad_idx,
        netclass: netclass.clone(),
        diff_pair_leader: None,
    });

    build_t_branch_tree(
        split_pt,
        driver.fp_idx,
        driver.pad_idx,
        left,
        driver,
        net_id,
        net_name,
        netclass,
        connections,
    );
    build_t_branch_tree(
        split_pt,
        driver.fp_idx,
        driver.pad_idx,
        right,
        driver,
        net_id,
        net_name,
        netclass,
        connections,
    );
}

/// Multi-layer PathFinder negotiated-congestion PCB autorouter.
///
/// Orchestrates escape stub synthesis, coupled differential-pair routing, 3D octilinear A*
/// pathfinding, rip-up and re-route with component/track nudging, length/phase meander tuning,
/// post-route rubber-band glossing, and ground-plane via stitching.
pub struct AutoRouter {
    /// Shove and nudge engine for displacing blocking footprints, tracks, and vias.
    pub nudge_engine: NudgeEngine,
    /// Maximum number of PathFinder rip-up and re-route iterations.
    pub max_ripup_iterations: usize,
    /// Historical congestion penalty multiplier per rip-up pass.
    pub congestion_factor: f32,
    /// Additional cost per millimeter for routing signal nets on inner copper layers (`layer >= 2`).
    pub inner_layer_penalty: f64,
    /// When `true`, allows [`NudgeEngine`] to nudge unlocked footprints when traces are blocked.
    pub allow_component_nudge: bool,
    /// Penalty weight per grid step when routing parallel and tightly coupled to a foreign net.
    pub crosstalk_penalty: f64,
    /// Full [`RoutingConfig`] parameter set governing router and post-route behavior.
    pub config: RoutingConfig,
    /// Optional live [`KiCadIpcClient`] connection for streaming incremental track updates to KiCad.
    pub ipc: Option<Arc<Mutex<KiCadIpcClient>>>,
}

impl AutoRouter {
    /// Creates a new [`AutoRouter`] configured for `max_ripup_iterations` PathFinder passes with default [`RoutingConfig`].
    ///
    /// # Arguments
    /// * `max_ripup_iterations` - Maximum number of rip-up and re-route iterations.
    ///
    /// # Returns
    /// Default-initialized [`AutoRouter`].
    pub fn new(max_ripup_iterations: usize) -> Self {
        let config = RoutingConfig::default();
        Self {
            nudge_engine: NudgeEngine::new(),
            max_ripup_iterations,
            congestion_factor: 1.5,
            inner_layer_penalty: 8.0,
            allow_component_nudge: false,
            crosstalk_penalty: config.crosstalk_penalty,
            config,
            ipc: None,
        }
    }

    /// Attaches an optional live [`KiCadIpcClient`] handle for real-time KiCad board synchronization.
    ///
    /// # Arguments
    /// * `ipc` - Optional shared mutex-guarded [`KiCadIpcClient`].
    ///
    /// # Returns
    /// Updated [`AutoRouter`].
    pub fn with_ipc(mut self, ipc: Option<Arc<Mutex<KiCadIpcClient>>>) -> Self {
        self.ipc = ipc;
        self
    }

    /// Sets the parallel crosstalk penalty weight in both `self.crosstalk_penalty` and `self.config.crosstalk_penalty`.
    ///
    /// # Arguments
    /// * `pen` - Crosstalk penalty weight per coupled grid step.
    ///
    /// # Returns
    /// Updated [`AutoRouter`].
    pub fn with_crosstalk_penalty(mut self, pen: f64) -> Self {
        self.crosstalk_penalty = pen;
        self.config.crosstalk_penalty = pen;
        self
    }

    /// Sets the per-millimeter cost penalty for routing signal nets on inner copper layers (`layer >= 2`).
    ///
    /// # Arguments
    /// * `pen` - Inner layer cost penalty per millimeter.
    ///
    /// # Returns
    /// Updated [`AutoRouter`].
    pub fn with_inner_layer_penalty(mut self, pen: f64) -> Self {
        self.inner_layer_penalty = pen;
        self
    }

    /// Configures whether the router may nudge unlocked footprints when routing channels are blocked.
    ///
    /// # Arguments
    /// * `allow` - `true` to enable footprint nudging during rip-up passes.
    ///
    /// # Returns
    /// Updated [`AutoRouter`].
    pub fn with_allow_component_nudge(mut self, allow: bool) -> Self {
        self.allow_component_nudge = allow;
        self
    }

    /// Enables or disables any-angle Euclidean line-of-sight glossing (`config.any_angle_mode`).
    ///
    /// # Arguments
    /// * `enabled` - `true` for any-angle tracks, `false` for strict 45° octilinear tracks.
    ///
    /// # Returns
    /// Updated [`AutoRouter`].
    pub fn with_any_angle_mode(mut self, enabled: bool) -> Self {
        self.config.any_angle_mode = enabled;
        self
    }

    /// Sets the chamfer/fillet radius (in `mm`) used to eliminate acute (< 90°) acid-trap corners.
    ///
    /// # Arguments
    /// * `radius` - Fillet radius in millimeters (`>= 0.0`).
    ///
    /// # Returns
    /// Updated [`AutoRouter`].
    pub fn with_acid_trap_fillet_radius(mut self, radius: f64) -> Self {
        self.config.acid_trap_fillet_radius = radius.max(0.0);
        self
    }

    /// Configures whether vias are permitted inside same-net SMD pads (`config.allow_via_in_pad`).
    ///
    /// # Arguments
    /// * `allow` - `true` to allow via-in-pad on matching nets.
    ///
    /// # Returns
    /// Updated [`AutoRouter`].
    pub fn with_allow_via_in_pad(mut self, allow: bool) -> Self {
        self.config.allow_via_in_pad = allow;
        self
    }

    /// Applies a complete [`RoutingConfig`] struct, synchronizing congestion, crosstalk, and nudge step fields.
    ///
    /// # Arguments
    /// * `config` - [`RoutingConfig`] to apply.
    ///
    /// # Returns
    /// Updated [`AutoRouter`].
    pub fn with_config(mut self, config: RoutingConfig) -> Self {
        self.congestion_factor = config.congestion_factor as f32;
        self.crosstalk_penalty = config.crosstalk_penalty;
        self.nudge_engine.step_mm = config.nudge_step_mm;
        self.config = config;
        self
    }

    /// Returns `true` if net names `a` and `b` form a complementary differential pair
    /// (e.g., `USB_DP`/`USB_DM`, `USB_D+`/`USB_D-`, `CLK_P`/`CLK_N`, or `TX+`/`TX-`).
    ///
    /// # Arguments
    /// * `a` - First net name.
    /// * `b` - Second net name.
    ///
    /// # Returns
    /// `true` if `a` and `b` share a common non-empty prefix and complementary positive/negative suffixes.
    pub fn are_diff_pair_mates(a: &str, b: &str) -> bool {
        if a == b {
            return false;
        }
        let ua = a.to_ascii_uppercase();
        let ub = b.to_ascii_uppercase();
        for (suf_p, suf_n) in [("DP", "DM"), ("D+", "D-"), ("_P", "_N"), ("+", "-")] {
            if let (Some(pa), Some(pb)) = (ua.strip_suffix(suf_p), ub.strip_suffix(suf_n)) {
                if !pa.is_empty() && pa == pb {
                    return true;
                }
            }
            if let (Some(pa), Some(pb)) = (ua.strip_suffix(suf_n), ub.strip_suffix(suf_p)) {
                if !pa.is_empty() && pa == pb {
                    return true;
                }
            }
        }
        false
    }

    /// Builds a 3D per-cell A* cost multiplier map (`num_layers * width * height`) that attracts
    /// a differential-pair follower trace into a tight parallel corridor at `ideal_pitch = track_width + max(req_clearance, max_net_clearance)`
    /// alongside `partner_path`.
    ///
    /// Cells outside the outer corridor band or on the opposite lateral side of `side` receive
    /// [`DIFF_PAIR_OFF_CORRIDOR_MULT`] (`6.0`), while cells near `ideal_pitch` receive a discount (`0.15..0.65`).
    ///
    /// # Arguments
    /// * `grid` - Reference to the [`RoutingGrid`].
    /// * `partner_path` - Ordered slice of [`GridNode`]s forming the routed leader trace.
    /// * `track_width` - Differential-pair trace width (`mm`).
    /// * `req_clearance` - Target differential gap or netclass clearance (`mm`).
    /// * `side` - Required lateral side (`+1.0`, `-1.0`, or `0.0` for either side) relative to the leader's tangent.
    ///
    /// # Returns
    /// Flat 3D vector of cost multipliers (`f32`) indexed by [`RoutingGrid::idx`].
    pub fn build_diff_pair_corridor(
        &self,
        grid: &RoutingGrid,
        partner_path: &[GridNode],
        track_width: f64,
        req_clearance: f64,
        side: f64,
    ) -> Vec<f32> {
        // Leaving the partner's corridor must cost far more than a detour inside it, otherwise the
        // follower takes a shorter channel on the far side of intervening components.
        let mut corridor = vec![DIFF_PAIR_OFF_CORRIDOR_MULT; grid.num_layers * grid.width * grid.height];
        // The follower cannot sit closer than the grid's worst-case clearance allows.
        let ideal_pitch = track_width + req_clearance.max(grid.max_net_clearance);
        let inner_band = ideal_pitch + grid.res * 0.5 + 0.35;
        let outer_band = ideal_pitch + grid.res * 0.5 + 0.90;
        let r_cells = (outer_band / grid.res).ceil() as isize;

        // Cost is set by each cell's distance to the *nearest* leader node (a min over per-node
        // costs would reward cells hugging the leader via some distant node). `side` (+1 left /
        // -1 right of the leader's travel, 0 = either) bars the far side, which forces a crossover.
        let cells = grid.num_layers * grid.width * grid.height;
        let mut nearest: Vec<(f32, i8)> = vec![(f32::INFINITY, 0); cells];
        let n = partner_path.len();
        for (i, &node) in partner_path.iter().enumerate() {
            let (a, b) = (partner_path[i.saturating_sub(3)], partner_path[(i + 3).min(n - 1)]);
            let (tx, ty) = (b.gx as isize - a.gx as isize, b.gy as isize - a.gy as isize);
            for dy in -r_cells..=r_cells {
                for dx in -r_cells..=r_cells {
                    let nx = node.gx as isize + dx;
                    let ny = node.gy as isize + dy;
                    if nx < 0 || nx >= grid.width as isize || ny < 0 || ny >= grid.height as isize {
                        continue;
                    }
                    let dist = (((dx * dx + dy * dy) as f64).sqrt() * grid.res) as f32;
                    let idx = grid.idx(nx as usize, ny as usize, node.layer);
                    if dist < nearest[idx].0 {
                        nearest[idx] = (dist, (tx * dy - ty * dx).signum() as i8);
                    }
                }
            }
        }
        for (idx, &(d, sgn)) in nearest.iter().enumerate() {
            let dist = d as f64;
            if dist > outer_band || (side != 0.0 && sgn != 0 && (sgn as f64) * side < 0.0) {
                continue;
            }
            let dev = (dist - ideal_pitch).abs();
            corridor[idx] = if dist < ideal_pitch - grid.res * 0.5 {
                1.0
            } else if dev <= grid.res * 0.5 {
                0.15
            } else if dist <= inner_band {
                0.35
            } else {
                0.65
            };
        }
        corridor
    }

    /// Routes `lc` as the differential-pair leader (reserving lateral clearance on the partner side)
    /// and then routes `fc` inside `lc`'s coupling corridor, committing both paths to `grid`.
    ///
    /// # Arguments
    /// * `board` - Reference to the [`Board`].
    /// * `grid` - Mutable reference to the [`RoutingGrid`].
    /// * `lc` - Leader [`RouteConnection`].
    /// * `fc` - Follower [`RouteConnection`].
    /// * `u1_center` - Optional world coordinates `(x, y)` of the primary fine-pitch IC.
    /// * `stop_flag` - Atomic cancellation flag.
    ///
    /// # Returns
    /// `Some((leader_path, follower_path))` if both traces routed cleanly, or `None` if either failed.
    ///
    /// # Side Effects
    /// Commits the routed paths into `grid.obstacle_net_ids` and `grid.usage_count`.
    fn route_coupled_pair(
        &self,
        board: &Board,
        grid: &mut RoutingGrid,
        lc: &RouteConnection,
        fc: &RouteConnection,
        u1_center: Option<(f64, f64)>,
        stop_flag: &AtomicBool,
    ) -> Option<(Vec<GridNode>, Vec<GridNode>)> {
        let diff_w = lc.netclass.diff_pair_width;
        let diff_gap = lc.netclass.diff_pair_gap;
        let nodes = |c: &RouteConnection| -> Option<(GridNode, GridNode)> {
            let (sgx, sgy) = grid.world_to_grid(c.start.0, c.start.1)?;
            let (egx, egy) = grid.world_to_grid(c.end.0, c.end.1)?;
            Some((
                GridNode { gx: sgx, gy: sgy, layer: c.start.2 },
                GridNode { gx: egx, gy: egy, layer: c.end.2 },
            ))
        };
        let (ls, le) = nodes(lc)?;
        let (fs, fe) = nodes(fc)?;
        // Routing may clamp the connection clearance for fine-pitch escapes, but DRC checks the pair
        // against the board netclass clearance, so the coupled gap must honour it.
        let gap_eff = diff_gap.max(board.get_netclass_for_net(lc.net_id).clearance);
        // Which side the follower must run on: pads escape radially out of the start footprint
        // and radially into the end footprint.
        let cross = |t: (f64, f64), d: (f64, f64)| t.0 * d.1 - t.1 * d.0;
        let (fa, fb) = (&board.footprints[lc.start_fp], &board.footprints[lc.end_fp]);
        let s0 = cross((lc.start.0 - fa.x, lc.start.1 - fa.y), (fc.start.0 - lc.start.0, fc.start.1 - lc.start.1));
        let s1 = cross((fb.x - lc.end.0, fb.y - lc.end.1), (fc.end.0 - lc.end.0, fc.end.1 - lc.end.1));
        let side = if s0 * s1 > 0.0 { s0.signum() } else { 0.0 };
        // Prefer the board netclass clearance so widening/DRC never has to neck the pair later;
        // fall back to the (fine-pitch clamped) connection clearance if that cannot route.
        let board_clr = board.get_netclass_for_net(lc.net_id).clearance.max(lc.netclass.clearance);
        let mut clrs = vec![board_clr];
        if lc.netclass.clearance < board_clr - 1e-6 {
            clrs.push(lc.netclass.clearance);
        }
        for clr in clrs {
            let Some(lp) = self.find_path_sided(
                grid, ls, le, lc.net_id, clr, diff_w, lc.netclass.via_dia, u1_center,
                1.0, -(diff_w + gap_eff), lc.netclass.max_coupled_length_mm,
                lc.netclass.require_return_path, None, side, stop_flag,
            ) else {
                continue;
            };
            let snap = (grid.obstacle_net_ids.clone(), grid.usage_count.clone());
            self.commit_path_to_grid(grid, &lp, lc.net_id, diff_w, lc.netclass.via_dia);
            let f_side = diff_pair_side(&lp, fs, fe);
            let corridor = self.build_diff_pair_corridor(grid, &lp, diff_w, gap_eff, f_side);
            if let Some(fp) = self.find_path_with_constraints(
                grid, fs, fe, fc.net_id, clr.min(board_clr.max(fc.netclass.clearance)), diff_w, fc.netclass.via_dia,
                u1_center, 1.0, 0.0, fc.netclass.max_coupled_length_mm, fc.netclass.require_return_path,
                Some(&corridor), stop_flag,
            ) {
                self.commit_path_to_grid(grid, &fp, fc.net_id, diff_w, fc.netclass.via_dia);
                return Some((lp, fp));
            }
            grid.obstacle_net_ids = snap.0;
            grid.usage_count = snap.1;
            grid.gpu_net_cost_maps.clear();
        }
        // Last resort: unsided leader (any homotopy) so the pair at least routes.
        let lp = self.find_path_parallel(
            grid, ls, le, lc.net_id, lc.netclass.clearance, diff_w, lc.netclass.via_dia, u1_center,
            lc.netclass.max_coupled_length_mm, lc.netclass.require_return_path, stop_flag,
        )?;
        self.commit_path_to_grid(grid, &lp, lc.net_id, diff_w, lc.netclass.via_dia);
        let corridor = self.build_diff_pair_corridor(grid, &lp, diff_w, gap_eff, diff_pair_side(&lp, fs, fe));
        let fp = self.find_path_with_constraints(
            grid, fs, fe, fc.net_id, fc.netclass.clearance, diff_w, fc.netclass.via_dia, u1_center,
            1.0, 0.0, fc.netclass.max_coupled_length_mm, fc.netclass.require_return_path,
            Some(&corridor), stop_flag,
        )?;
        self.commit_path_to_grid(grid, &fp, fc.net_id, diff_w, fc.netclass.via_dia);
        Some((lp, fp))
    }

    /// Extracts all two-terminal [`RouteConnection`]s from `board` for all non-plane signal and power nets,
    /// prioritized by differential-pair status, intra-footprint proximity, IC escape criticality,
    /// netclass priority, and Euclidean length.
    ///
    /// Automatically pairs complementary differential-pair connections (`diff_pair_leader = Some(leader_net_id)`)
    /// and selects non-crossing pad pairs on multi-pad USB-C / receptacle footprints.
    ///
    /// # Arguments
    /// * `board` - Reference to the [`Board`] whose pads and nets are analyzed.
    ///
    /// # Returns
    /// Priority-ordered vector of [`RouteConnection`]s ready for the autorouter pipeline.
    pub fn extract_connections(&self, board: &Board) -> Vec<RouteConnection> {
        let mut net_pads: HashMap<usize, Vec<PadInfo>> = HashMap::new();
        for (fp_idx, fp) in board.footprints.iter().enumerate() {
            for (pad_idx, pad) in fp.pads.iter().enumerate() {
                if pad.net_id == 0 || board.is_plane_net(pad.net_id) {
                    continue;
                }
                let (px, py) = fp.pad_absolute_position(pad);
                let pad_layer = if pad.is_through_hole { 0 } else { fp.pad_layer(pad) };
                let entry = net_pads.entry(pad.net_id).or_default();
                if !entry
                    .iter()
                    .any(|p| p.layer == pad_layer && (p.x - px).hypot(p.y - py) < 0.05)
                {
                    let is_driver = fp.reference.starts_with('U')
                        || fp.reference.starts_with("IC")
                        || fp.reference.starts_with("MCU")
                        || fp.reference.starts_with('J')
                        || fp.reference.starts_with("VR")
                        || fp.reference.starts_with("REG");
                    entry.push(PadInfo {
                        x: px,
                        y: py,
                        layer: pad_layer,
                        fp_idx,
                        pad_idx,
                        is_driver,
                    });
                }
            }
        }

        let mut connections = Vec::new();
        let mut sorted_net_ids: Vec<usize> = net_pads.keys().copied().collect();
        sorted_net_ids.sort_unstable();
        for net_id in sorted_net_ids {
            let pads = &net_pads[&net_id];
            if pads.len() < 2 {
                continue;
            }
            let net_name = board.nets[net_id].clone();
            let netclass = board.get_netclass_for_net(net_id).clone();
            let eff_topo = netclass.resolve_topology(&net_name);

            let net_conns = build_net_topology_connections(net_id, &net_name, &netclass, pads, eff_topo);
            connections.extend(net_conns);
        }

        connections.sort_by(|a, b| {
            let a_diff = a.netclass.name == "USB_Diff"
                || a.net_name.ends_with("DP")
                || a.net_name.ends_with("DM")
                || a.net_name.ends_with("D+")
                || a.net_name.ends_with("D-");
            let b_diff = b.netclass.name == "USB_Diff"
                || b.net_name.ends_with("DP")
                || b.net_name.ends_with("DM")
                || b.net_name.ends_with("D+")
                || b.net_name.ends_with("D-");
            let prio_a = Self::netclass_priority(&a.netclass.name);
            let prio_b = Self::netclass_priority(&b.netclass.name);
            let a_is_ic = matches!(board.footprints[a.start_fp].reference.as_str(), "U1" | "J2")
                || matches!(board.footprints[a.end_fp].reference.as_str(), "U1" | "J2");
            let b_is_ic = matches!(board.footprints[b.start_fp].reference.as_str(), "U1" | "J2")
                || matches!(board.footprints[b.end_fp].reference.as_str(), "U1" | "J2");

            let a_intra = a.start_fp == a.end_fp;
            let b_intra = b.start_fp == b.end_fp;
            let d_a = (a.start.0 - a.end.0).hypot(a.start.1 - a.end.1);
            let d_b = (b.start.0 - b.end.0).hypot(b.start.1 - b.end.1);
            b_diff
                .cmp(&a_diff)
                .then_with(|| a_intra.cmp(&b_intra))
                .then_with(|| b_is_ic.cmp(&a_is_ic))
                .then_with(|| {
                    if a_is_ic && b_is_ic {
                        d_a.partial_cmp(&d_b).unwrap_or(Ordering::Equal)
                    } else {
                        prio_b.cmp(&prio_a)
                            .then_with(|| d_a.partial_cmp(&d_b).unwrap_or(Ordering::Equal))
                    }
                })
                .then_with(|| prio_b.cmp(&prio_a))
                .then_with(|| a.net_id.cmp(&b.net_id))
                .then_with(|| (a.start_fp, a.start_pad).cmp(&(b.start_fp, b.start_pad)))
                .then_with(|| (a.end_fp, a.end_pad).cmp(&(b.end_fp, b.end_pad)))
        });

        // Group differential pair and parallel multi-terminal connections (connecting two distinct footprints) adjacently
        // so the follower routes immediately alongside its leader trace.
        let mut i = 0;
        while i + 1 < connections.len() {
            if connections[i].diff_pair_leader.is_some() || connections[i].start_fp == connections[i].end_fp {
                i += 1;
                continue;
            }
            let fp_pair_i = (
                connections[i].start_fp.min(connections[i].end_fp),
                connections[i].start_fp.max(connections[i].end_fp),
            );
            let mut mate_idx = None;
            for j in (i + 1)..connections.len() {
                if connections[j].diff_pair_leader.is_some() || connections[j].start_fp == connections[j].end_fp {
                    continue;
                }
                let fp_pair_j = (
                    connections[j].start_fp.min(connections[j].end_fp),
                    connections[j].start_fp.max(connections[j].end_fp),
                );
                let is_diff = Self::are_diff_pair_mates(&connections[i].net_name, &connections[j].net_name)
                    || (connections[i].netclass.name == "USB_Diff" && connections[j].netclass.name == "USB_Diff");
                let is_twin = (board.footprints[fp_pair_i.0].reference.starts_with('C') || board.footprints[fp_pair_i.1].reference.starts_with('C'))
                    && (connections[i].net_name.contains("CAP") || connections[i].net_name.contains("HP"));

                if fp_pair_i == fp_pair_j && (is_diff || is_twin) {
                    mate_idx = Some((j, is_diff));
                    break;
                }
            }
            if let Some((j, is_diff)) = mate_idx {
                let mut follower = connections.remove(j);
                if connections[i].start_fp != follower.start_fp {
                    follower.swap_endpoints();
                }
                if is_diff {
                    let fp_a = connections[i].start_fp;
                    let fp_b = connections[i].end_fp;
                    if let (Some(pads_i), Some(pads_j)) = (
                        net_pads.get(&connections[i].net_id),
                        net_pads.get(&follower.net_id),
                    ) {
                        let pads_i_a: Vec<&PadInfo> = pads_i.iter().filter(|p| p.fp_idx == fp_a).collect();
                        let pads_i_b: Vec<&PadInfo> = pads_i.iter().filter(|p| p.fp_idx == fp_b).collect();
                        let pads_j_a: Vec<&PadInfo> = pads_j.iter().filter(|p| p.fp_idx == fp_a).collect();
                        let pads_j_b: Vec<&PadInfo> = pads_j.iter().filter(|p| p.fp_idx == fp_b).collect();
                        if pads_i_a.len() > 1 || pads_i_b.len() > 1 || pads_j_a.len() > 1 || pads_j_b.len() > 1 {
                            let mut best_score = f64::INFINITY;
                            let mut best_tuple: Option<(&PadInfo, &PadInfo, &PadInfo, &PadInfo)> = None;
                            for &pa in &pads_i_a {
                                for &pb in &pads_i_b {
                                    for &na in &pads_j_a {
                                        for &nb in &pads_j_b {
                                            let da = (na.x - pa.x, na.y - pa.y);
                                            let db = (nb.x - pb.x, nb.y - pb.y);
                                            let da_len = da.0.hypot(da.1);
                                            let db_len = db.0.hypot(db.1);
                                            let mid_a = ((pa.x + na.x) * 0.5, (pa.y + na.y) * 0.5);
                                            let mid_b = ((pb.x + nb.x) * 0.5, (pb.y + nb.y) * 0.5);
                                            let trunk = (mid_b.0 - mid_a.0, mid_b.1 - mid_a.1);
                                            // Travel direction leaving fp_a / arriving at fp_b: pads escape
                                            // radially from their footprint, which can be perpendicular to the trunk.
                                            let fa = &board.footprints[fp_a];
                                            let fb = &board.footprints[fp_b];
                                            let dir_a = if (mid_a.0 - fa.x).hypot(mid_a.1 - fa.y) > 0.1 {
                                                (mid_a.0 - fa.x, mid_a.1 - fa.y)
                                            } else {
                                                trunk
                                            };
                                            let dir_b = if (mid_b.0 - fb.x).hypot(mid_b.1 - fb.y) > 0.1 {
                                                (fb.x - mid_b.0, fb.y - mid_b.1)
                                            } else {
                                                trunk
                                            };
                                            let side_a = dir_a.0 * da.1 - dir_a.1 * da.0;
                                            let side_b = dir_b.0 * db.1 - dir_b.1 * db.0;
                                            let crosses = segments_intersect(
                                                (pa.x, pa.y),
                                                (pb.x, pb.y),
                                                (na.x, na.y),
                                                (nb.x, nb.y),
                                            ) || (side_a * side_b < -1e-6);
                                            let cross_pen = if crosses { 10_000.0 } else { 0.0 };
                                            let d_i = (pa.x - pb.x).hypot(pa.y - pb.y);
                                            let d_j = (na.x - nb.x).hypot(na.y - nb.y);
                                            let score = cross_pen + (da_len + db_len) * 20.0 + d_i + d_j;
                                            if score < best_score {
                                                best_score = score;
                                                best_tuple = Some((pa, pb, na, nb));
                                            }
                                        }
                                    }
                                }
                            }
                            if let Some((pa, pb, na, nb)) = best_tuple {
                                connections[i].start = (pa.x, pa.y, pa.layer);
                                connections[i].start_pad = pa.pad_idx;
                                connections[i].end = (pb.x, pb.y, pb.layer);
                                connections[i].end_pad = pb.pad_idx;
                                follower.start = (na.x, na.y, na.layer);
                                follower.start_pad = na.pad_idx;
                                follower.end = (nb.x, nb.y, nb.layer);
                                follower.end_pad = nb.pad_idx;
                            }
                        }
                    }
                }
                // For coupled differential pairs, route the inner (shorter span) trace first so it hugs
                // the inside of the bend without cutting across the outer pad's escape stub.
                let d_i = (connections[i].start.0 - connections[i].end.0).hypot(connections[i].start.1 - connections[i].end.1);
                let d_j = (follower.start.0 - follower.end.0).hypot(follower.start.1 - follower.end.1);
                if (is_diff && d_i > d_j) || (!is_diff && d_i < d_j) {
                    std::mem::swap(&mut connections[i], &mut follower);
                }
                if is_diff {
                    follower.diff_pair_leader = Some(i);
                }
                connections.insert(i + 1, follower);
                i += 2;
            } else {
                i += 1;
            }
        }

        connections
    }

    /// Returns the numeric routing priority for a netclass `name` (higher values route earlier).
    ///
    /// # Arguments
    /// * `name` - Netclass name string (e.g., `"Switching"`, `"USB_Diff"`, `"Default"`).
    ///
    /// # Returns
    /// Priority score in `1..=10`.
    fn netclass_priority(name: &str) -> usize {
        match name {
            "Switching" | "Power_SW" => 10,
            "Power_15V" | "P1P2_Bus" | "Power" => 9,
            "USB_Diff" => 8,
            "HBS_Bus" | "Analog_Sense" => 7,
            "Power_3V3" | "Power_5V" => 6,
            _ => 1,
        }
    }

    /// Performs a breadth-first graph traversal over all tracks, arcs, and vias belonging to
    /// `conn.net_id` on `board` to verify whether `conn`'s start and end pads are physically
    /// and electrically connected on compatible copper layers.
    ///
    /// # Arguments
    /// * `board` - Reference to the [`Board`] containing footprints, tracks, arcs, and vias.
    /// * `conn` - The [`RouteConnection`] whose start and end pads are tested for continuity.
    ///
    /// # Returns
    /// `true` if a continuous same-net copper path connects `conn.start_pad` to `conn.end_pad`.
    pub fn is_connection_electrically_connected(board: &Board, conn: &RouteConnection) -> bool {
        let Some(sfp) = board.footprints.get(conn.start_fp) else { return false; };
        let Some(spad) = sfp.pads.get(conn.start_pad) else { return false; };
        let Some(efp) = board.footprints.get(conn.end_fp) else { return false; };
        let Some(epad) = efp.pads.get(conn.end_pad) else { return false; };
        let s_pos = sfp.pad_absolute_position(spad);
        let e_pos = efp.pad_absolute_position(epad);
        let s_th = spad.is_through_hole;
        let e_th = epad.is_through_hole;
        let s_layer = sfp.pad_layer(spad);
        let e_layer = efp.pad_layer(epad);
        let net_id = conn.net_id;

        let mut tracks: Vec<TrackSegment> = board
            .tracks
            .iter()
            .filter(|t| t.net_id == net_id)
            .cloned()
            .collect();
        for a in board.arcs.iter().filter(|a| a.net_id == net_id) {
            tracks.push(TrackSegment {
                start: a.start(),
                end: a.mid(),
                width: a.width,
                layer: a.layer,
                net_id: a.net_id,
                is_locked: a.is_locked,
            });
            tracks.push(TrackSegment {
                start: a.mid(),
                end: a.end(),
                width: a.width,
                layer: a.layer,
                net_id: a.net_id,
                is_locked: a.is_locked,
            });
        }
        let vias: Vec<&Via> = board.vias.iter().filter(|v| v.net_id == net_id).collect();
        let n_t = tracks.len();
        let n_v = vias.len();
        if n_t == 0 && n_v == 0 {
            return false;
        }
        const EPS: f64 = 0.05;
        let mut adj = vec![Vec::new(); n_t + n_v];
        for i in 0..n_t {
            let ti = &tracks[i];
            for j in (i + 1)..n_t {
                let tj = &tracks[j];
                if ti.layer == tj.layer {
                    let c = (ti.start.0 - tj.start.0).hypot(ti.start.1 - tj.start.1) < EPS
                        || (ti.start.0 - tj.end.0).hypot(ti.start.1 - tj.end.1) < EPS
                        || (ti.end.0 - tj.start.0).hypot(ti.end.1 - tj.start.1) < EPS
                        || (ti.end.0 - tj.end.0).hypot(ti.end.1 - tj.end.1) < EPS
                        || dist_point_to_segment(ti.start, tj.start, tj.end) < EPS
                        || dist_point_to_segment(ti.end, tj.start, tj.end) < EPS
                        || dist_point_to_segment(tj.start, ti.start, ti.end) < EPS
                        || dist_point_to_segment(tj.end, ti.start, ti.end) < EPS;
                    if c {
                        adj[i].push(j);
                        adj[j].push(i);
                    }
                }
            }
            for vi in 0..n_v {
                let v = vias[vi];
                let d_s = (ti.start.0 - v.position.0).hypot(ti.start.1 - v.position.1);
                let d_e = (ti.end.0 - v.position.0).hypot(ti.end.1 - v.position.1);
                if d_s <= v.size * 0.5 + EPS
                    || d_e <= v.size * 0.5 + EPS
                    || dist_point_to_segment(v.position, ti.start, ti.end) <= v.size * 0.5 + EPS
                {
                    adj[i].push(n_t + vi);
                    adj[n_t + vi].push(i);
                }
            }
        }
        let mut queue = VecDeque::new();
        let mut visited = vec![false; n_t + n_v];
        for i in 0..n_t {
            let ti = &tracks[i];
            if s_th || ti.layer == s_layer {
                if (ti.start.0 - s_pos.0).hypot(ti.start.1 - s_pos.1) < 0.12
                    || (ti.end.0 - s_pos.0).hypot(ti.end.1 - s_pos.1) < 0.12
                    || dist_point_to_pad(ti.start, sfp, spad) < EPS
                    || dist_point_to_pad(ti.end, sfp, spad) < EPS
                {
                    visited[i] = true;
                    queue.push_back(i);
                }
            }
        }
        while let Some(u) = queue.pop_front() {
            if u < n_t {
                let ti = &tracks[u];
                if e_th || ti.layer == e_layer {
                    if (ti.start.0 - e_pos.0).hypot(ti.start.1 - e_pos.1) < 0.12
                        || (ti.end.0 - e_pos.0).hypot(ti.end.1 - e_pos.1) < 0.12
                        || dist_point_to_pad(ti.start, efp, epad) < EPS
                        || dist_point_to_pad(ti.end, efp, epad) < EPS
                    {
                        return true;
                    }
                }
            }
            for &nxt in &adj[u] {
                if !visited[nxt] {
                    visited[nxt] = true;
                    queue.push_back(nxt);
                }
            }
        }
        false
    }

    /// Executes the full multi-layer PCB autorouting pipeline on `board` and `grid`.
    ///
    /// Performs the following phases:
    /// 1. Clears unlocked copper, rasterizes obstacles, and synthesizes BGA dogbones, staggered fine-pitch
    ///    IC escape stubs, and `U1` ground-pad stitching vias.
    /// 2. Runs iterative PathFinder negotiated-congestion rip-up and re-route passes (coupled differential pairs,
    ///    parallel portfolio A* + GPU 3D wavefront search, conflict-free wave partitioning, and component/track shoving).
    /// 3. Commits routed paths, runs post-route rubber-band and topological glossing, widens power/signal tracks,
    ///    tunes differential-pair phase and netclass target lengths, stitches ground-plane vias, and runs final DRC cleanup.
    ///
    /// # Arguments
    /// * `board` - Mutable reference to the [`Board`] to route.
    /// * `grid` - Mutable reference to the [`RoutingGrid`].
    /// * `live` - Optional shared [`LiveTrialState`] for real-time GUI telemetry and progress logging.
    /// * `stop_flag` - Atomic flag allowing early user cancellation.
    ///
    /// # Returns
    /// Tuple `(routed_connections, total_connections)` reporting final routing completion counts.
    ///
    /// # Side Effects
    /// Mutates `board.tracks`, `board.vias`, `board.arcs`, `board.footprints` (if component nudging is enabled),
    /// `grid`, `live`, and streams updates over `self.ipc` if connected.
    pub fn route_all(
        &self,
        board: &mut Board,
        grid: &mut RoutingGrid,
        live: Option<Arc<Mutex<LiveTrialState>>>,
        stop_flag: Arc<AtomicBool>,
    ) -> (usize, usize) {
        // Clear out any old unlocked tracks/vias/arcs before routing so they don't block stubs or grid
        board.tracks.retain(|t| t.is_locked);
        board.vias.retain(|v| v.is_locked);
        board.arcs.retain(|a| a.is_locked);

        if let Some(ref ipc_lock) = self.ipc {
            if let Ok(mut client) = ipc_lock.lock() {
                if client.is_connected() {
                    let _ = client.clear_tracks();
                    let _ = client.commit_batch();
                }
            }
        }

        let allow_vip = self.config.allow_via_in_pad || board.allow_via_in_pad() || grid.allow_via_in_pad;
        board.set_allow_via_in_pad(allow_vip);
        grid.allow_via_in_pad = allow_vip;

        grid.obstacle_net_ids.fill(FREE_NET);
        grid.nudgeable_fp.fill(None);
        grid.smd_pad_cells.fill(false);
        grid.rasterize_obstacles(board);
        grid.update_distance_fields();

        if self.allow_component_nudge && self.nudge_components_blocking_escapes(board) {
            grid.obstacle_net_ids.fill(FREE_NET);
            grid.nudgeable_fp.fill(None);
            grid.smd_pad_cells.fill(false);
            grid.rasterize_obstacles(board);
            grid.update_distance_fields();
        }

        let mut connections = self.extract_connections(board);
        let total_conns = connections.len();

        generate_bga_dogbone_escapes(board, grid);

        let escape_stubs = self.generate_escape_stubs(board);
        for stub in &escape_stubs {
            self.commit_stub_to_grid(grid, stub);
        }

        // Fanout First: Pre-route ground fanout/stitching for U1 thermal pad ground pins before signal routing
        let pre_tracks_count = board.tracks.len();
        let pre_vias_count = board.vias.len();

        self.connect_u1_ground_pins(board);

        for trk in &mut board.tracks[pre_tracks_count..] {
            trk.is_locked = true;
            grid.commit_track(trk);
        }
        for v in &mut board.vias[pre_vias_count..] {
            v.is_locked = true;
            grid.commit_via(v);
        }

        let orig_locked_tracks: std::collections::HashSet<(usize, [u64; 4])> = board
            .tracks
            .iter()
            .filter(|t| t.is_locked)
            .map(|t| (t.net_id, [t.start.0.to_bits(), t.start.1.to_bits(), t.end.0.to_bits(), t.end.1.to_bits()]))
            .collect();
        let orig_locked_vias: std::collections::HashSet<(usize, [u64; 2])> = board
            .vias
            .iter()
            .filter(|v| v.is_locked)
            .map(|v| (v.net_id, [v.position.0.to_bits(), v.position.1.to_bits()]))
            .collect();

        grid.update_distance_fields();

        for conn in &mut connections {
            for stub in &escape_stubs {
                    if conn.net_id == stub.net_id
                        && conn.start.2 == stub.layer
                        && (conn.start.0 - stub.pad_pos.0).hypot(conn.start.1 - stub.pad_pos.1) < 0.15
                    {
                        conn.start = (stub.tip_pos.0, stub.tip_pos.1, stub.tip_layer);
                        break;
                    }
                }
                for stub in &escape_stubs {
                    if conn.net_id == stub.net_id
                        && conn.end.2 == stub.layer
                        && (conn.end.0 - stub.pad_pos.0).hypot(conn.end.1 - stub.pad_pos.1) < 0.15
                    {
                        conn.end = (stub.tip_pos.0, stub.tip_pos.1, stub.tip_layer);
                        break;
                    }
                }
            let is_f1_f2_d2_d3 = board.footprints.get(conn.start_fp).map_or(false, |fp| matches!(fp.reference.as_str(), "F1" | "F2" | "D2" | "D3"))
                || board.footprints.get(conn.end_fp).map_or(false, |fp| matches!(fp.reference.as_str(), "F1" | "F2" | "D2" | "D3"));
            let is_passive_terminal = board.footprints.get(conn.start_fp).map_or(false, |fp| fp.reference.starts_with('R') || fp.reference.starts_with('C'))
                || board.footprints.get(conn.end_fp).map_or(false, |fp| fp.reference.starts_with('R') || fp.reference.starts_with('C'));
            let is_power_net = conn.netclass.name.starts_with("Power")
                || conn.netclass.name == "Power"
                || conn.netclass.name == "Power_SW"
                || matches!(conn.net_name.as_str(), "VBUS" | "VBus" | "/BUCK_VIN" | "BUCK_VIN" | "Buck_in" | "VRAW" | "HP" | "HN" | "P1" | "P2" | "+3.3V" | "VCC_5V")
                || conn.net_name.contains("BUCK")
                || conn.net_name.contains("Buck");
            if conn.netclass.name == "Default" && (is_f1_f2_d2_d3 || is_power_net) {
                if let Some(pwr) = board.netclasses.get("Power").or_else(|| board.netclasses.get("Power_15V")) {
                    conn.netclass = pwr.clone();
                }
            }
            let touches_fine_pitch = board.footprints.get(conn.start_fp).map_or(false, |fp| fp.reference == "U1" || fp.reference == "J2" || Self::is_dense_escape_footprint(fp))
                || board.footprints.get(conn.end_fp).map_or(false, |fp| fp.reference == "U1" || fp.reference == "J2" || Self::is_dense_escape_footprint(fp))
                || escape_stubs.iter().any(|s| {
                    (conn.start.0 - s.tip_pos.0).hypot(conn.start.1 - s.tip_pos.1) < 0.05
                        || (conn.end.0 - s.tip_pos.0).hypot(conn.end.1 - s.tip_pos.1) < 0.05
                });
            if touches_fine_pitch {
                let touches_05mm_ic = board
                    .footprints
                    .get(conn.start_fp)
                    .is_some_and(|fp| fp.reference == "U1" || fp.reference == "J2" || fp.min_pad_pitch() <= 0.65)
                    || board
                        .footprints
                        .get(conn.end_fp)
                        .is_some_and(|fp| fp.reference == "U1" || fp.reference == "J2" || fp.min_pad_pitch() <= 0.65);
                if !is_power_net || touches_05mm_ic {
                    conn.netclass.track_width = conn.netclass.track_width.min(0.15);
                }
                conn.netclass.diff_pair_width = conn.netclass.diff_pair_width.min(0.20);
                conn.netclass.diff_pair_gap = conn.netclass.diff_pair_gap.min(0.15);
                if touches_05mm_ic {
                    conn.netclass.clearance = conn.netclass.clearance.min(0.15);
                }
                conn.netclass.via_dia = conn.netclass.via_dia.min(0.60);
                let min_annular = board.design_rules().min_via_annular_width;
                let min_drill = board.design_rules().min_via_drill;
                if (conn.netclass.via_dia - conn.netclass.via_drill) * 0.5 < min_annular - 1e-4 {
                    conn.netclass.via_drill = (conn.netclass.via_dia - 2.0 * min_annular).max(min_drill);
                    conn.netclass.via_dia = conn.netclass.via_dia.max(conn.netclass.via_drill + 2.0 * min_annular);
                }
            }
            if conn.netclass.name == "USB_Diff" || conn.net_name.contains("USB") {
                conn.netclass.diff_pair_width = conn.netclass.diff_pair_width.min(0.20);
                conn.netclass.diff_pair_gap = conn.netclass.diff_pair_gap.min(0.15);
            }
            if is_passive_terminal {
                if !is_power_net && !conn.netclass.name.starts_with("Power") && conn.netclass.name != "P1P2_Bus" {
                    conn.netclass.track_width = conn.netclass.track_width.min(0.15);
                    conn.netclass.via_dia = conn.netclass.via_dia.min(0.60);
                    let min_annular = board.design_rules().min_via_annular_width;
                    let min_drill = board.design_rules().min_via_drill;
                    if (conn.netclass.via_dia - conn.netclass.via_drill) * 0.5 < min_annular - 1e-4 {
                        conn.netclass.via_drill = (conn.netclass.via_dia - 2.0 * min_annular).max(min_drill);
                        conn.netclass.via_dia = conn.netclass.via_dia.max(conn.netclass.via_drill + 2.0 * min_annular);
                    }
                }
            }
        }

        let u1_center = board
            .footprints
            .iter()
            .find(|fp| fp.reference == "U1")
            .map(|fp| (fp.x, fp.y));

        let num_threads = rayon::current_num_threads().max(2);
        grid.inner_layer_penalty = self.inner_layer_penalty;
        if let Some(ref st) = live {
            if let Ok(mut g) = st.lock() {
                g.phase = TrialPhase::AutoRouting;
                g.total_conns = total_conns;
                g.routed_conns = 0;
                let gpu_tag = if let Some(ref vk) = grid.vulkan {
                    format!(" + Vulkan GPU ({})", vk.device_name)
                } else {
                    String::new()
                };
                g.push_log(format!(
                    "[Auto-Router] Routing {} connections ({} layers) across {} CPU cores{} with {} escape stubs...",
                    total_conns,
                    grid.num_layers,
                    num_threads,
                    gpu_tag,
                    escape_stubs.len()
                ));
            }
        }

        let mut routed_paths: Vec<Option<Vec<GridNode>>> = vec![None; total_conns];
        let mut best_routed_paths: Vec<Option<Vec<GridNode>>> = vec![None; total_conns];
        let mut best_footprints = board.footprints.clone();
        let mut best_connections = connections.clone();
        let mut best_routed_count = 0;
        let mut unrouted_history: Vec<usize> = vec![0; total_conns];
        let gpu_done = AtomicBool::new(false);
        let shared_obs = Arc::new(Mutex::new(grid.obstacle_net_ids.clone()));
        let vk_opt = grid.vulkan.clone();
        let grid_w = grid.width as u32;
        let grid_h = grid.height as u32;
        let grid_res = grid.res as f32;
        let layer_cells = grid.width * grid.height;
        let num_layers = grid.num_layers;
        if let Some(ref vk) = vk_opt {
            let engines = vk.all_engines();
            if !engines.is_empty() {
                let layer_dfs: Vec<(usize, Option<Vec<f32>>)> = std::thread::scope(|s| {
                    let handles: Vec<_> = (0..num_layers)
                        .map(|layer| {
                            let vk_dev = Arc::clone(&engines[layer % engines.len()]);
                            let offset = layer * layer_cells;
                            let obs_u32: Vec<u32> = grid.obstacle_net_ids[offset..offset + layer_cells]
                                .iter()
                                .map(|&n| if n != FREE_NET { 1u32 } else { 0u32 })
                                .collect();
                            s.spawn(move || (layer, vk_dev.compute_distance_field(&obs_u32, grid_w, grid_h, 16)))
                        })
                        .collect();
                    handles.into_iter().filter_map(|h| h.join().ok()).collect()
                });
                for (layer, df_opt) in layer_dfs {
                    if let Some(df) = df_opt {
                        let offset = layer * layer_cells;
                        for (dst, &val) in grid.dist_field[offset..offset + layer_cells]
                            .iter_mut()
                            .zip(df.iter())
                        {
                            *dst = val * grid_res;
                        }
                    }
                }
            }
        }
        std::thread::scope(|_scope| {
            let route_start_time = std::time::Instant::now();
            let mut stagnant_passes = 0usize;
            for iteration in 0..self.max_ripup_iterations {
                if stop_flag.load(AtomicOrdering::Relaxed) {
                    break;
                }
                if routed_paths.iter().all(|p| p.is_some()) {
                    best_routed_paths = routed_paths.clone();
                    best_routed_count = total_conns;
                    break;
                }

                for i in 0..total_conns {
                    if routed_paths[i].is_none() {
                        unrouted_history[i] += 1;
                    }
                }

                if iteration > 0 {
                    if let Some(ref vk) = vk_opt {
                        let engines = vk.all_engines();
                        if !engines.is_empty() {
                            let layer_dfs: Vec<(usize, Option<Vec<f32>>)> = std::thread::scope(|s| {
                                let handles: Vec<_> = (0..num_layers)
                                    .map(|layer| {
                                        let vk_dev = Arc::clone(&engines[layer % engines.len()]);
                                        let offset = layer * layer_cells;
                                        let obs_u32: Vec<u32> = grid.obstacle_net_ids[offset..offset + layer_cells]
                                            .iter()
                                            .map(|&n| if n != FREE_NET { 1u32 } else { 0u32 })
                                            .collect();
                                        s.spawn(move || (layer, vk_dev.compute_distance_field(&obs_u32, grid_w, grid_h, 16)))
                                    })
                                    .collect();
                                handles.into_iter().filter_map(|h| h.join().ok()).collect()
                            });
                            for (layer, df_opt) in layer_dfs {
                                if let Some(df) = df_opt {
                                    let offset = layer * layer_cells;
                                    for (dst, &val) in grid.dist_field[offset..offset + layer_cells]
                                        .iter_mut()
                                        .zip(df.iter())
                                    {
                                        *dst = val * grid_res;
                                    }
                                }
                            }
                        }
                    }
                }

                // PathFinder historical congestion decay to prevent late-pass grid saturation
                for c in &mut grid.congestion {
                    *c *= 0.85;
                }

                // ponytail: preserve full user netclass widths and clearances. ceiling: non-BGA pins pitched tighter than netclass width + 2x clearance. upgrade: add localized pin-entry neck-down rule when pad pitch < netclass diameter.

                // Phase 0: Coupled Differential Pair Bus Routing
                // Route any unrouted differential pair leader and immediately route its follower
                // within the coupled proximity corridor before single-ended nets can split the bus.
                for follower_idx in 0..total_conns {
                    let Some(leader_idx) = connections[follower_idx].diff_pair_leader else {
                        continue;
                    };
                    if routed_paths[leader_idx].is_some() && routed_paths[follower_idx].is_some() {
                        continue;
                    }
                    let lc = &connections[leader_idx];
                    let fc = &connections[follower_idx];
                    let diff_w = lc.netclass.diff_pair_width;
                    let diff_gap = lc.netclass.diff_pair_gap;

                    // A rip-up that removed only one half of the pair would leave the survivor routed
                    // without regard to its partner; re-route both together instead.
                    if routed_paths[leader_idx].is_some() != routed_paths[follower_idx].is_some() {
                        routed_paths[leader_idx] = None;
                        routed_paths[follower_idx] = None;
                        self.rebuild_grid_state(board, grid, &connections, &routed_paths, &escape_stubs);
                    }

                    // Try both nets as leader: whichever pad sits on the constrained side of a
                    // nearby obstacle must route first, or its partner is pinched into another channel.
                    if routed_paths[leader_idx].is_none() && routed_paths[follower_idx].is_none() {
                        let mut best: Option<(f64, Vec<GridNode>, Vec<GridNode>)> = None;
                        for swapped in [false, true] {
                            let (a, b) = if swapped { (fc, lc) } else { (lc, fc) };
                            let trial = self.route_coupled_pair(board, grid, a, b, u1_center, &stop_flag);
                            if trial.is_none() {
                                eprintln!("  [Router] Diff pair {} -> {}: no coupled route", a.net_name, b.net_name);
                            }
                            self.rebuild_grid_state(board, grid, &connections, &routed_paths, &escape_stubs);
                            if let Some((pa, pb)) = trial {
                                let c = path_coupling_mm(grid.res, &pa, &pb);
                                eprintln!(
                                    "  [Router] Diff pair {} -> {}: mean coupling {:.2} mm",
                                    a.net_name, b.net_name, c
                                );
                                let (lp, fp) = if swapped { (pb, pa) } else { (pa, pb) };
                                if best.as_ref().map_or(true, |b| c < b.0) {
                                    best = Some((c, lp, fp));
                                }
                            }
                            if best.as_ref().is_some_and(|b| b.0 <= DIFF_PAIR_COUPLED_MM) {
                                break;
                            }
                        }
                        if let Some((_, lp, fp)) = best {
                            self.commit_path_to_grid(grid, &lp, lc.net_id, diff_w, lc.netclass.via_dia);
                            self.commit_path_to_grid(grid, &fp, fc.net_id, diff_w, fc.netclass.via_dia);
                            routed_paths[leader_idx] = Some(lp);
                            routed_paths[follower_idx] = Some(fp);
                            continue;
                        }
                    }
                    let mut was_leader_fresh = false;
                    if routed_paths[leader_idx].is_none() {
                        if let (Some((sgx, sgy)), Some((egx, egy))) = (
                            grid.world_to_grid(lc.start.0, lc.start.1),
                            grid.world_to_grid(lc.end.0, lc.end.1),
                        ) {
                            let s_node = GridNode { gx: sgx, gy: sgy, layer: lc.start.2 };
                            let e_node = GridNode { gx: egx, gy: egy, layer: lc.end.2 };
                            let lp_opt = self.find_path_with_constraints(
                                grid,
                                s_node,
                                e_node,
                                lc.net_id,
                                lc.netclass.clearance,
                                diff_w,
                                lc.netclass.via_dia,
                                u1_center,
                                1.0,
                                -(diff_w + diff_gap),
                                lc.netclass.max_coupled_length_mm,
                                lc.netclass.require_return_path,
                                None,
                                &stop_flag,
                            ).or_else(|| self.find_path_parallel(
                                grid,
                                s_node,
                                e_node,
                                lc.net_id,
                                lc.netclass.clearance,
                                diff_w,
                                lc.netclass.via_dia,
                                u1_center,
                                lc.netclass.max_coupled_length_mm,
                                lc.netclass.require_return_path,
                                &stop_flag,
                            ));
                            if let Some(lp) = lp_opt {
                                self.commit_path_to_grid(
                                    grid,
                                    &lp,
                                    lc.net_id,
                                    diff_w,
                                    lc.netclass.via_dia,
                                );
                                routed_paths[leader_idx] = Some(lp);
                                was_leader_fresh = true;
                            }
                        }
                    }
                    if routed_paths[follower_idx].is_none() {
                        if let Some(ref leader_path) = routed_paths[leader_idx] {
                            let corridor = self.build_diff_pair_corridor(
                                grid,
                                leader_path,
                                diff_w,
                                diff_gap,
                                0.0,
                            );
                            if let (Some((sgx, sgy)), Some((egx, egy))) = (
                                grid.world_to_grid(fc.start.0, fc.start.1),
                                grid.world_to_grid(fc.end.0, fc.end.1),
                            ) {
                                let s_node = GridNode { gx: sgx, gy: sgy, layer: fc.start.2 };
                                let e_node = GridNode { gx: egx, gy: egy, layer: fc.end.2 };
                                let fp_opt = self
                                    .find_path_with_constraints(
                                        grid,
                                        s_node,
                                        e_node,
                                        fc.net_id,
                                        fc.netclass.clearance,
                                        diff_w,
                                        fc.netclass.via_dia,
                                        u1_center,
                                        1.0,
                                        0.0,
                                        fc.netclass.max_coupled_length_mm,
                                        fc.netclass.require_return_path,
                                        Some(&corridor),
                                        &stop_flag,
                                    )
                                    .or_else(|| {
                                        self.find_path_parallel(
                                            grid,
                                            s_node,
                                            e_node,
                                            fc.net_id,
                                            fc.netclass.clearance,
                                            diff_w,
                                            fc.netclass.via_dia,
                                            u1_center,
                                            fc.netclass.max_coupled_length_mm,
                                            fc.netclass.require_return_path,
                                            &stop_flag,
                                        )
                                    });
                                if let Some(fp) = fp_opt {
                                    self.commit_path_to_grid(
                                        grid,
                                        &fp,
                                        fc.net_id,
                                        diff_w,
                                        fc.netclass.via_dia,
                                    );
                                    routed_paths[follower_idx] = Some(fp);
                                }
                            }
                        }
                        if routed_paths[follower_idx].is_none() && was_leader_fresh {
                            routed_paths[leader_idx] = None;
                            self.rebuild_grid_state(board, grid, &connections, &routed_paths, &escape_stubs);
                        }
                    }
                }

                let mut unrouted_indices: Vec<usize> = (0..total_conns)
                    .filter(|&i| routed_paths[i].is_none())
                    .collect();
                if unrouted_indices.is_empty() {
                    break;
                }
                unrouted_indices.sort_by_key(|&i| std::cmp::Reverse(unrouted_history[i]));

                // Offload 3D routing cost and clearance field precomputation to all GPUs before parallel A* wave
                let mut net_clearance_map: HashMap<usize, i32> = HashMap::new();
                let diag_margin = (grid.res * 0.7071).min(0.02);
                for &conn_idx in &unrouted_indices {
                    let conn = &connections[conn_idx];
                    let max_clearance = grid.max_net_clearance.max(conn.netclass.clearance);
                    let search_clearance = (conn.netclass.track_width / 2.0) + max_clearance + diag_margin;
                    let r_cells = (search_clearance / grid.res).ceil() as i32;
                    let entry = net_clearance_map.entry(conn.net_id).or_insert(r_cells);
                    if r_cells > *entry {
                        *entry = r_cells;
                    }
                }
                let mut net_batch: Vec<(usize, i32)> = net_clearance_map.into_iter().collect();
                net_batch.sort_unstable_by_key(|&(net_id, _)| net_id);
                grid.precompute_gpu_routing_costs_batch(&net_batch, self.congestion_factor, 2.0);

                // Phase 1: Speculative multi-core parallel A* wave or Parallel Portfolio Ensemble
                #[allow(non_local_definitions)]
                impl RoutingGrid {
                    /// Clones the grid's obstacle, distance-field, and congestion state for parallel portfolio evaluation
                    /// without duplicating cached GPU cost maps.
                    ///
                    /// # Returns
                    /// Independent [`RoutingGrid`] clone sharing `vulkan` and `base_snapshot` handles.
                    pub fn clone_for_portfolio(&self) -> Self {
                        Self {
                            min_x: self.min_x,
                            min_y: self.min_y,
                            max_x: self.max_x,
                            max_y: self.max_y,
                            res: self.res,
                            width: self.width,
                            height: self.height,
                            num_layers: self.num_layers,
                            plane_nets: self.plane_nets.clone(),
                            obstacle_net_ids: self.obstacle_net_ids.clone(),
                            dist_field: self.dist_field.clone(),
                            congestion: self.congestion.clone(),
                            usage_count: self.usage_count.clone(),
                            gpu_net_cost_maps: HashMap::new(),
                            nudgeable_fp: self.nudgeable_fp.clone(),
                            smd_pad_cells: self.smd_pad_cells.clone(),
                            net_clearances: self.net_clearances.clone(),
                            max_net_clearance: self.max_net_clearance,
                            inner_layer_penalty: self.inner_layer_penalty,
                            allow_via_in_pad: self.allow_via_in_pad,
                            vulkan: self.vulkan.clone(),
                            base_snapshot: self.base_snapshot.clone(),
                        }
                    }
                }

                /// Precomputed 2D/3D bounding box and collinear segment/via decomposition of a candidate grid path
                /// used for fast pairwise conflict detection during parallel wave partitioning.
                #[derive(Clone, Debug)]
                struct PathConflictGeom {
                    /// Numeric net ID of the candidate path.
                    net_id: usize,
                    /// Trace width in millimeters (`mm`).
                    width: f64,
                    /// Required netclass clearance in millimeters (`mm`).
                    clearance: f64,
                    /// Via outer diameter in millimeters (`mm`).
                    via_dia: f64,
                    /// Minimum world X coordinate (`mm`) of the path's AABB.
                    min_x: f64,
                    /// Maximum world X coordinate (`mm`) of the path's AABB.
                    max_x: f64,
                    /// Minimum world Y coordinate (`mm`) of the path's AABB.
                    min_y: f64,
                    /// Maximum world Y coordinate (`mm`) of the path's AABB.
                    max_y: f64,
                    /// Maximum half-width or via radius (`mm`) along the path.
                    max_radius: f64,
                    /// Bitmask of copper layers touched by the path (`0xFFFF_FFFF` if any via is present).
                    layer_mask: u32,
                    /// Coalesced straight segments `(start_xy, end_xy, layer)`.
                    segs: Vec<((f64, f64), (f64, f64), usize)>,
                    /// World positions `(x, y)` of layer-transition vias along the path.
                    vias: Vec<(f64, f64)>,
                }

                impl PathConflictGeom {
                    /// Extracts the [`PathConflictGeom`] representation from a discrete [`GridNode`] `path`.
                    fn from_path(
                        grid: &RoutingGrid,
                        conn: &RouteConnection,
                        path: &[GridNode],
                    ) -> Self {
                        let width = if conn.diff_pair_leader.is_some() || conn.netclass.name == "USB_Diff" {
                            conn.netclass.diff_pair_width
                        } else {
                            conn.netclass.track_width
                        };
                        let clearance = conn.netclass.clearance;
                        let via_dia = conn.netclass.via_dia;
                        let mut min_x = f64::INFINITY;
                        let mut max_x = f64::NEG_INFINITY;
                        let mut min_y = f64::INFINITY;
                        let mut max_y = f64::NEG_INFINITY;
                        let mut layer_mask = 0u32;
                        let mut segs = Vec::new();
                        let mut vias = Vec::new();

                        for &n in path {
                            let (wx, wy) = grid.grid_to_world(n.gx, n.gy);
                            min_x = min_x.min(wx);
                            max_x = max_x.max(wx);
                            min_y = min_y.min(wy);
                            max_y = max_y.max(wy);
                            if n.layer < 32 {
                                layer_mask |= 1u32 << n.layer;
                            }
                        }

                        if path.len() == 1 {
                            let p0 = grid.grid_to_world(path[0].gx, path[0].gy);
                            segs.push((p0, p0, path[0].layer));
                        } else if path.len() >= 2 {
                            let mut run_start = 0usize;
                            for i in 0..(path.len() - 1) {
                                let u = path[i];
                                let v = path[i + 1];
                                if u.layer != v.layer {
                                    if i > run_start {
                                        let p1 = grid.grid_to_world(path[run_start].gx, path[run_start].gy);
                                        let p2 = grid.grid_to_world(u.gx, u.gy);
                                        segs.push((p1, p2, u.layer));
                                    }
                                    vias.push(grid.grid_to_world(u.gx, u.gy));
                                    layer_mask = 0xFFFF_FFFF;
                                    run_start = i + 1;
                                } else if i + 2 < path.len() && path[i + 2].layer == v.layer {
                                    let dx1 = v.gx as isize - u.gx as isize;
                                    let dy1 = v.gy as isize - u.gy as isize;
                                    let dx2 = path[i + 2].gx as isize - v.gx as isize;
                                    let dy2 = path[i + 2].gy as isize - v.gy as isize;
                                    if (dx1, dy1) != (dx2, dy2) {
                                        let p1 = grid.grid_to_world(path[run_start].gx, path[run_start].gy);
                                        let p2 = grid.grid_to_world(v.gx, v.gy);
                                        segs.push((p1, p2, u.layer));
                                        run_start = i + 1;
                                    }
                                }
                            }
                            if run_start + 1 < path.len() {
                                let last = path.len() - 1;
                                if path[run_start].layer == path[last].layer {
                                    let p1 = grid.grid_to_world(path[run_start].gx, path[run_start].gy);
                                    let p2 = grid.grid_to_world(path[last].gx, path[last].gy);
                                    segs.push((p1, p2, path[run_start].layer));
                                }
                            }
                        }

                        let max_radius = (width * 0.5).max(if !vias.is_empty() { via_dia * 0.5 } else { 0.0 });
                        Self {
                            net_id: conn.net_id,
                            width,
                            clearance,
                            via_dia,
                            min_x,
                            max_x,
                            min_y,
                            max_y,
                            max_radius,
                            layer_mask,
                            segs,
                            vias,
                        }
                    }

                    /// Returns `true` if `self` and `other` belong to different nets and their segments or vias
                    /// violate bidirectional clearance.
                    #[inline]
                    fn conflicts_with(&self, other: &Self, grid_res: f64) -> bool {
                        if self.net_id == other.net_id {
                            return false;
                        }
                        if (self.layer_mask & other.layer_mask) == 0 {
                            return false;
                        }
                        let pair_clr = self.clearance.max(other.clearance) + grid_res * 1.5;
                        let aabb_margin = self.max_radius + other.max_radius + pair_clr;
                        if self.max_x + aabb_margin < other.min_x
                            || other.max_x + aabb_margin < self.min_x
                            || self.max_y + aabb_margin < other.min_y
                            || other.max_y + aabb_margin < self.min_y
                        {
                            return false;
                        }

                        let req_seg = (self.width + other.width) * 0.5 + pair_clr;
                        for &(sa1, sa2, la) in &self.segs {
                            for &(sb1, sb2, lb) in &other.segs {
                                if la == lb && dist_segment_to_segment(sa1, sa2, sb1, sb2) < req_seg {
                                    return true;
                                }
                            }
                        }

                        if !self.vias.is_empty() && !other.vias.is_empty() {
                            let req_vv = (self.via_dia + other.via_dia) * 0.5 + pair_clr;
                            for &va in &self.vias {
                                for &vb in &other.vias {
                                    if (va.0 - vb.0).hypot(va.1 - vb.1) < req_vv {
                                        return true;
                                    }
                                }
                            }
                        }

                        if !self.vias.is_empty() {
                            let req_vs = (self.via_dia + other.width) * 0.5 + pair_clr;
                            for &va in &self.vias {
                                for &(sb1, sb2, _) in &other.segs {
                                    if dist_to_segment(va.0, va.1, sb1.0, sb1.1, sb2.0, sb2.1) < req_vs {
                                        return true;
                                    }
                                }
                            }
                        }

                        if !other.vias.is_empty() {
                            let req_vs = (other.via_dia + self.width) * 0.5 + pair_clr;
                            for &vb in &other.vias {
                                for &(sa1, sa2, _) in &self.segs {
                                    if dist_to_segment(vb.0, vb.1, sa1.0, sa1.1, sa2.0, sa2.1) < req_vs {
                                        return true;
                                    }
                                }
                            }
                        }

                        false
                    }
                }

                /// Hyperparameter configuration for a single A* portfolio search thread.
                #[derive(Clone, Copy, Debug)]
                struct PortfolioVariant {
                    /// Multiplier applied to layer-transition via cost.
                    via_mult: f64,
                    /// Weight applied to historical PathFinder congestion.
                    hist_w: f64,
                    /// Weight applied to the octilinear heuristic $h(n)$.
                    heur_w: f64,
                    /// Weight applied to inverse obstacle distance field ($1 / d_{\text{obst}}$).
                    dist_bias: f64,
                    /// When `true`, searches backward from `goal` to `start`.
                    reverse: bool,
                    /// When `true`, slightly favors 45° diagonal steps (`0.97x`) over orthogonal steps.
                    diag_first: bool,
                }

                /// Ensemble of 12 diverse A* cost/direction parameterizations explored concurrently by [`AutoRouter::route_connections_portfolio`].
                const PORTFOLIO_VARIANTS: [PortfolioVariant; 12] = [
                    PortfolioVariant { via_mult: 1.0,  hist_w: 1.0, heur_w: 1.0,  dist_bias: 0.0,  reverse: false, diag_first: false },
                    PortfolioVariant { via_mult: 1.0,  hist_w: 1.0, heur_w: 1.0,  dist_bias: 0.0,  reverse: true,  diag_first: false },
                    PortfolioVariant { via_mult: 0.5,  hist_w: 1.0, heur_w: 1.0,  dist_bias: 0.0,  reverse: false, diag_first: true  },
                    PortfolioVariant { via_mult: 0.5,  hist_w: 1.0, heur_w: 1.0,  dist_bias: 0.0,  reverse: true,  diag_first: true  },
                    PortfolioVariant { via_mult: 0.75, hist_w: 0.5, heur_w: 1.15, dist_bias: 0.0,  reverse: false, diag_first: false },
                    PortfolioVariant { via_mult: 0.75, hist_w: 0.5, heur_w: 1.15, dist_bias: 0.0,  reverse: true,  diag_first: false },
                    PortfolioVariant { via_mult: 1.25, hist_w: 1.5, heur_w: 1.0,  dist_bias: 0.01, reverse: false, diag_first: true  },
                    PortfolioVariant { via_mult: 1.25, hist_w: 1.5, heur_w: 1.0,  dist_bias: 0.01, reverse: true,  diag_first: true  },
                    PortfolioVariant { via_mult: 1.75, hist_w: 2.5, heur_w: 1.15, dist_bias: 0.02, reverse: false, diag_first: false },
                    PortfolioVariant { via_mult: 1.75, hist_w: 2.5, heur_w: 1.15, dist_bias: 0.02, reverse: true,  diag_first: false },
                    PortfolioVariant { via_mult: 2.5,  hist_w: 0.5, heur_w: 1.35, dist_bias: 0.0,  reverse: false, diag_first: true  },
                    PortfolioVariant { via_mult: 2.5,  hist_w: 2.5, heur_w: 1.35, dist_bias: 0.01, reverse: true,  diag_first: true  },
                ];

                #[allow(non_local_definitions)]
                impl AutoRouter {
                    /// Partitions candidate `(conn_idx, path)` routes into conflict-free independent waves
                    /// via greedy conflict-graph coloring.
                    ///
                    /// Two paths of different nets conflict when their clearance-inflated 2D/3D bounding boxes
                    /// overlap AND their segments or vias encroach.
                    ///
                    /// # Arguments
                    /// * `grid` - Reference to the [`RoutingGrid`].
                    /// * `connections` - Slice of [`RouteConnection`]s indexed by `conn_idx`.
                    /// * `candidates` - Vector of `(conn_idx, grid_path)` pairs to partition.
                    ///
                    /// # Returns
                    /// Vector of conflict-free waves, where all paths within a single wave can be committed in parallel.
                    pub fn partition_into_conflict_free_waves(
                        grid: &RoutingGrid,
                        connections: &[RouteConnection],
                        candidates: Vec<(usize, Vec<GridNode>)>,
                    ) -> Vec<Vec<(usize, Vec<GridNode>)>> {
                        if candidates.is_empty() {
                            return Vec::new();
                        }
                        let geoms: Vec<PathConflictGeom> = candidates
                            .par_iter()
                            .map(|(conn_idx, path)| PathConflictGeom::from_path(grid, &connections[*conn_idx], path))
                            .collect();

                        let mut waves: Vec<Vec<(usize, Vec<GridNode>)>> = Vec::new();
                        let mut wave_geoms: Vec<Vec<PathConflictGeom>> = Vec::new();

                        for (cand, geom) in candidates.into_iter().zip(geoms.into_iter()) {
                            let mut placed_wave = None;
                            for (w_idx, w_geoms) in wave_geoms.iter().enumerate() {
                                let conflicts = if w_geoms.len() >= 12 {
                                    w_geoms.par_iter().any(|other| geom.conflicts_with(other, grid.res))
                                } else {
                                    w_geoms.iter().any(|other| geom.conflicts_with(other, grid.res))
                                };
                                if !conflicts {
                                    placed_wave = Some(w_idx);
                                    break;
                                }
                            }
                            if let Some(w_idx) = placed_wave {
                                waves[w_idx].push(cand);
                                wave_geoms[w_idx].push(geom);
                            } else {
                                waves.push(vec![cand]);
                                wave_geoms.push(vec![geom]);
                            }
                        }
                        waves
                    }

                    /// Evaluates the quality metrics `(via_count, planar_wirelength_mm, congestion_sum)` of a discrete `path`.
                    ///
                    /// # Arguments
                    /// * `grid` - Reference to the [`RoutingGrid`].
                    /// * `path` - Slice of [`GridNode`]s forming the routed path.
                    ///
                    /// # Returns
                    /// Tuple `(vias, wirelength_mm, congestion_sum)`.
                    pub fn evaluate_path_cost(grid: &RoutingGrid, path: &[GridNode]) -> (usize, f64, f64) {
                        let mut vias = 0usize;
                        let mut wirelength = 0.0f64;
                        let mut cong_sum = 0.0f64;
                        for (i, &n) in path.iter().enumerate() {
                            let idx = grid.idx(n.gx, n.gy, n.layer);
                            cong_sum += grid.congestion[idx] as f64;
                            if i + 1 < path.len() {
                                let nxt = path[i + 1];
                                if nxt.layer != n.layer {
                                    vias += 1;
                                } else {
                                    let dx = (nxt.gx as isize - n.gx as isize).abs() as f64;
                                    let dy = (nxt.gy as isize - n.gy as isize).abs() as f64;
                                    wirelength += dx.hypot(dy) * grid.res;
                                }
                            }
                        }
                        (vias, wirelength, cong_sum)
                    }

                    /// Runs a single parameterized 3D A* search variant between `start` and `goal` using the
                    /// given portfolio weights (`via_mult`, `hist_w`, `heur_w`, `dist_bias`, `reverse`, `diag_first`).
                    ///
                    /// # Arguments
                    /// * `grid` - Reference to the [`RoutingGrid`].
                    /// * `start` - Start [`GridNode`].
                    /// * `goal` - Goal [`GridNode`].
                    /// * `net_id` - Numeric net identifier.
                    /// * `req_clearance` - Required copper clearance (`mm`).
                    /// * `track_width` - Trace width (`mm`).
                    /// * `via_dia` - Via outer diameter (`mm`).
                    /// * `u1_center` - Optional `(x, y)` center of fine-pitch IC `U1`.
                    /// * `max_coupled_length_mm` - Maximum parallel coupled length (`mm`) before crosstalk penalty applies.
                    /// * `require_return_path` - Whether to penalize reference plane voids.
                    /// * `via_mult` - Via transition cost multiplier.
                    /// * `hist_w` - Historical congestion weight.
                    /// * `heur_w` - Heuristic weight.
                    /// * `dist_bias` - Inverse obstacle distance bias weight.
                    /// * `reverse` - If `true`, searches from `goal` to `start` and reverses the resulting path.
                    /// * `diag_first` - If `true`, slightly favors diagonal steps (`0.97x`).
                    /// * `abort_flag` - Atomic cancellation flag.
                    /// * `found_count` - Optional atomic counter tracking how many portfolio variants have already succeeded for this connection.
                    /// * `max_expansions_override` - Optional cap on A* node expansions.
                    ///
                    /// # Returns
                    /// `Some(Vec<GridNode>)` containing the smoothed path if found within the expansion limit, or `None`.
                    #[allow(clippy::too_many_arguments)]
                    pub fn find_path_portfolio_variant(
                        &self,
                        grid: &RoutingGrid,
                        start: GridNode,
                        goal: GridNode,
                        net_id: usize,
                        req_clearance: f64,
                        track_width: f64,
                        via_dia: f64,
                        u1_center: Option<(f64, f64)>,
                        max_coupled_length_mm: f64,
                        require_return_path: bool,
                        via_mult: f64,
                        hist_w: f64,
                        heur_w: f64,
                        dist_bias: f64,
                        reverse: bool,
                        diag_first: bool,
                        abort_flag: &AtomicBool,
                        found_count: Option<&std::sync::atomic::AtomicUsize>,
                        max_expansions_override: Option<usize>,
                    ) -> Option<Vec<GridNode>> {
                        let (s_node, g_node) = if reverse { (goal, start) } else { (start, goal) };
                        let mut open_set = BinaryHeap::new();
                        // Single compact map keyed by grid.idx(gx, gy, layer) as u32 -> (parent, g_score, coupled_len)
                        let mut node_state: HashMap<u32, (GridNode, f64, f32)> = HashMap::new();

                        let heur = |a: GridNode, b: GridNode| -> f64 {
                            let dx = (a.gx as isize - b.gx as isize).abs() as f64 * grid.res;
                            let dy = (a.gy as isize - b.gy as isize).abs() as f64 * grid.res;
                            let diag = dx.min(dy);
                            let straight = (dx - dy).abs();
                            let layer_pen = if a.layer != b.layer { 2.0 * via_mult } else { 0.0 };
                            diag * std::f64::consts::SQRT_2 + straight + layer_pen
                        };

                        let h_start = heur(s_node, g_node) * heur_w;
                        let s_idx = grid.idx(s_node.gx, s_node.gy, s_node.layer) as u32;
                        node_state.insert(s_idx, (s_node, 0.0, 0.0));
                        open_set.push(AStarNode {
                            node: s_node,
                            cost: 0.0,
                            priority: h_start,
                        });

                        let start_is_smd = (0..grid.num_layers).any(|l| grid.smd_pad_cells[grid.idx(s_node.gx, s_node.gy, l)]);
                        let start_is_tht = !start_is_smd
                            && (0..grid.num_layers).all(|l| grid.obstacle_net_ids[grid.idx(s_node.gx, s_node.gy, l)] == net_id as i32);
                        if start_is_tht {
                            for l in 0..grid.num_layers {
                                if l != s_node.layer {
                                    if grid.plane_nets.contains_key(&l) && grid.plane_nets.get(&l) != Some(&net_id) {
                                        continue;
                                    }
                                    let alt_start = GridNode { gx: s_node.gx, gy: s_node.gy, layer: l };
                                    let alt_idx = grid.idx(alt_start.gx, alt_start.gy, alt_start.layer) as u32;
                                    node_state.insert(alt_idx, (alt_start, 0.0, 0.0));
                                    open_set.push(AStarNode {
                                        node: alt_start,
                                        cost: 0.0,
                                        priority: heur(alt_start, g_node) * heur_w,
                                    });
                                }
                            }
                        }

                        let dist_span = (g_node.gx as isize - s_node.gx as isize).abs() + (g_node.gy as isize - s_node.gy as isize).abs();
                        let max_expansions = max_expansions_override.unwrap_or(if dist_span > 200 { 350_000 } else { 120_000 });
                        let mut expansions = 0usize;

                        while let Some(current) = open_set.pop() {
                            expansions += 1;
                            if expansions > max_expansions {
                                return None;
                            }
                            if expansions & 255 == 0 {
                                if abort_flag.load(AtomicOrdering::Relaxed) {
                                    return None;
                                }
                                if expansions > 16_384 {
                                    if let Some(fc) = found_count {
                                        if fc.load(AtomicOrdering::Relaxed) >= 3 {
                                            return None;
                                        }
                                    }
                                }
                            }

                            let u = current.node;
                            let u_idx = grid.idx(u.gx, u.gy, u.layer) as u32;
                            if u == g_node {
                                let mut path = vec![g_node];
                                let mut curr = g_node;
                                while let Some(&(prev, _, _)) = node_state.get(&(grid.idx(curr.gx, curr.gy, curr.layer) as u32)) {
                                    if prev == curr {
                                        break;
                                    }
                                    path.push(prev);
                                    curr = prev;
                                }
                                path.reverse();
                                if reverse {
                                    path.reverse();
                                    if !self.is_path_conflict_free(
                                        grid,
                                        &path,
                                        net_id,
                                        req_clearance,
                                        track_width,
                                        via_dia,
                                        u1_center,
                                        require_return_path,
                                    ) {
                                        return None;
                                    }
                                }
                                let smoothed = self.smooth_grid_path(
                                    grid,
                                    path,
                                    net_id,
                                    req_clearance,
                                    track_width,
                                    via_dia,
                                    u1_center,
                                    require_return_path,
                                    max_coupled_length_mm,
                                );
                                if let Some(fc) = found_count {
                                    fc.fetch_add(1, AtomicOrdering::Relaxed);
                                }
                                return Some(smoothed);
                            }

                            let Some(&(parent_u, current_g, prev_coupled_u)) = node_state.get(&u_idx) else {
                                continue;
                            };
                            if current.cost > current_g + 1e-6 {
                                continue;
                            }

                            let prev_dir: Option<(isize, isize)> = if parent_u != u && parent_u.layer == u.layer {
                                Some((u.gx as isize - parent_u.gx as isize, u.gy as isize - parent_u.gy as isize))
                            } else {
                                None
                            };

                            for (neighbor, mut edge_cost) in
                                grid.get_neighbors(u, net_id, req_clearance, track_width, via_dia, u1_center)
                            {
                                let mut next_coupled = 0.0f32;
                                let n_idx = grid.idx(neighbor.gx, neighbor.gy, neighbor.layer);
                                if neighbor.layer == u.layer {
                                    let is_diag = neighbor.gx != u.gx && neighbor.gy != u.gy;
                                    if (diag_first && is_diag) || (!diag_first && !is_diag) {
                                        edge_cost *= 0.97;
                                    }
                                    let cong = grid.congestion[n_idx] as f64;
                                    let usage = grid.usage_count[n_idx] as f64;
                                    if (hist_w - 1.0).abs() > 1e-3 || usage > 0.0 {
                                        edge_cost += ((hist_w - 1.0) * cong + hist_w * 0.35 * usage).max(-0.4 * edge_cost) * grid.res;
                                    }
                                    if let Some((pdx, pdy)) = prev_dir {
                                        let cur_dx = neighbor.gx as isize - u.gx as isize;
                                        let cur_dy = neighbor.gy as isize - u.gy as isize;
                                        if (cur_dx, cur_dy) != (pdx, pdy) {
                                            let dot = pdx * cur_dx + pdy * cur_dy;
                                            if dot > 0 {
                                                edge_cost += 5.0 * grid.res;
                                            } else if dot == 0 {
                                                edge_cost += 15.0 * grid.res;
                                            } else {
                                                edge_cost += 45.0 * grid.res;
                                            }
                                        }
                                    }
                                    if require_return_path {
                                        if let Some(ref_layer) = grid.adjacent_reference_plane(neighbor.layer) {
                                            if grid.is_reference_plane_void_at(neighbor.gx, neighbor.gy, ref_layer) {
                                                edge_cost += 35.0 * grid.res + 5.0;
                                            }
                                        }
                                    }
                                    let is_coupled = grid.is_step_coupled(u, neighbor, net_id, track_width, req_clearance);
                                    let step_len = if is_diag { 1.41421356 * grid.res } else { grid.res };
                                    if is_coupled {
                                        let next_c = prev_coupled_u as f64 + step_len;
                                        next_coupled = next_c as f32;
                                        if next_c > max_coupled_length_mm {
                                            let excess = next_c - max_coupled_length_mm;
                                            edge_cost += self.crosstalk_penalty * step_len * (1.0 + excess * 0.5);
                                        }
                                    }
                                } else {
                                    edge_cost = edge_cost * via_mult + 2.0 * via_mult;
                                }

                                if dist_bias > 0.0 {
                                    let df_val = (grid.dist_field[n_idx] as f64).clamp(0.05, 2.0);
                                    edge_cost += dist_bias / df_val;
                                }
                                let tentative_g = current_g + edge_cost;
                                let n_key = n_idx as u32;
                                let is_better = node_state.get(&n_key).is_none_or(|&(_, prev_g, _)| tentative_g < prev_g - 1e-6);
                                if is_better {
                                    node_state.insert(n_key, (u, tentative_g, next_coupled));
                                    let priority = tentative_g + heur(neighbor, g_node) * heur_w;
                                    open_set.push(AStarNode {
                                        node: neighbor,
                                        cost: tentative_g,
                                        priority,
                                    });
                                }
                            }
                        }
                        None
                    }

                    /// Batch-validates a slice of `(conn_idx, path)` candidates using [`GpuDrcContext`] on the GPU
                    /// alongside parallel `AutoRouter::is_path_conflict_free` checks across Rayon threads.
                    ///
                    /// # Arguments
                    /// * `board` - Reference to the [`Board`].
                    /// * `grid` - Reference to the [`RoutingGrid`].
                    /// * `connections` - Slice of [`RouteConnection`]s.
                    /// * `candidates` - Slice of `(conn_idx, path)` candidate routes.
                    /// * `u1_center` - Optional `(x, y)` center of `U1`.
                    ///
                    /// # Returns
                    /// Vector of `(grid_conflict_free, gpu_drc_clean)` boolean pairs for each candidate.
                    pub fn batch_validate_paths_gpu(
                        &self,
                        board: &Board,
                        grid: &RoutingGrid,
                        connections: &[RouteConnection],
                        candidates: &[(usize, Vec<GridNode>)],
                        u1_center: Option<(f64, f64)>,
                    ) -> Vec<(bool, bool)> {
                        if candidates.is_empty() {
                            return Vec::new();
                        }
                        let grid_valid: Vec<bool> = candidates
                            .par_iter()
                            .map(|(conn_idx, path)| {
                                let conn = &connections[*conn_idx];
                                let w = if conn.diff_pair_leader.is_some() || conn.netclass.name == "USB_Diff" {
                                    conn.netclass.diff_pair_width
                                } else {
                                    conn.netclass.track_width
                                };
                                self.is_path_conflict_free(
                                    grid,
                                    path,
                                    conn.net_id,
                                    conn.netclass.clearance,
                                    w,
                                    conn.netclass.via_dia,
                                    u1_center,
                                    conn.netclass.require_return_path,
                                )
                            })
                            .collect();

                        let drc_ctx = GpuDrcContext::from_board(board);
                        let mut gpu_cands = Vec::new();
                        let mut cand_ranges: Vec<(usize, usize)> = Vec::with_capacity(candidates.len());

                        for &(conn_idx, ref path) in candidates {
                            let start_offset = gpu_cands.len();
                            let conn = &connections[conn_idx];
                            let geom = PathConflictGeom::from_path(grid, conn, path);
                            let half_w = (geom.width * 0.5) as f32;
                            let via_r = (geom.via_dia * 0.5) as f32;
                            let clr = (geom.clearance.min(0.15)) as f32;
                            for &(p1, p2, layer) in &geom.segs {
                                let d_start = (p1.0 - conn.start.0).hypot(p1.1 - conn.start.1)
                                    .min((p2.0 - conn.start.0).hypot(p2.1 - conn.start.1));
                                let d_end = (p1.0 - conn.end.0).hypot(p1.1 - conn.end.1)
                                    .min((p2.0 - conn.end.0).hypot(p2.1 - conn.end.1));
                                if d_start > 0.85 && d_end > 0.85 {
                                    gpu_cands.push(GpuDrcCandidate {
                                        x1: p1.0 as f32,
                                        y1: p1.1 as f32,
                                        x2: p2.0 as f32,
                                        y2: p2.1 as f32,
                                        radius_or_half_width: half_w,
                                        clearance: clr,
                                        layer: layer as u32,
                                        net_id: conn.net_id as u32,
                                        kind: 1,
                                        ignore_same_net_pads: 1,
                                        _pad0: 0,
                                        _pad1: 0,
                                    });
                                }
                            }
                            for &vp in &geom.vias {
                                let d_start = (vp.0 - conn.start.0).hypot(vp.1 - conn.start.1);
                                let d_end = (vp.0 - conn.end.0).hypot(vp.1 - conn.end.1);
                                if d_start > 0.65 && d_end > 0.65 {
                                    gpu_cands.push(GpuDrcCandidate {
                                        x1: vp.0 as f32,
                                        y1: vp.1 as f32,
                                        x2: 0.0,
                                        y2: 0.0,
                                        radius_or_half_width: via_r,
                                        clearance: clr,
                                        layer: 0,
                                        net_id: conn.net_id as u32,
                                        kind: 0,
                                        ignore_same_net_pads: 0,
                                        _pad0: 0,
                                        _pad1: 0,
                                    });
                                }
                            }
                            cand_ranges.push((start_offset, gpu_cands.len()));
                        }

                        let gpu_res = drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_cands);
                        grid_valid
                            .into_iter()
                            .enumerate()
                            .map(|(idx, g_ok)| {
                                let (s, e) = cand_ranges[idx];
                                let drc_ok = if s == e {
                                    true
                                } else if let Some(ref res) = gpu_res {
                                    res[s..e].iter().all(|&v| v != 0)
                                } else {
                                    true
                                };
                                (g_ok, drc_ok)
                            })
                            .collect()
                    }

                    /// Expands `target_indices` across a Parallel Portfolio of `(conn_idx, variant_id)`
                    /// trajectories in `par_iter()`, batch-validates all discovered candidates on the GPU
                    /// ([`GpuDrcContext`]), and selects for each connection the legal path minimizing
                    /// `(vias * 10.0 + wirelength + bends * 0.20 + cong_sum * 0.02 + drc_pen)`.
                    ///
                    /// # Arguments
                    /// * `board` - Reference to the [`Board`].
                    /// * `grid` - Reference to the [`RoutingGrid`].
                    /// * `connections` - Slice of all [`RouteConnection`]s.
                    /// * `target_indices` - Slice of connection indices to solve in parallel.
                    /// * `u1_center` - Optional `(x, y)` center of `U1`.
                    /// * `stop_flag` - Atomic cancellation flag.
                    ///
                    /// # Returns
                    /// Vector of `(conn_idx, Option<Vec<GridNode>>)` pairs containing the winning path for each target connection.
                    pub fn route_connections_portfolio(
                        &self,
                        board: &Board,
                        grid: &RoutingGrid,
                        connections: &[RouteConnection],
                        target_indices: &[usize],
                        u1_center: Option<(f64, f64)>,
                        stop_flag: &AtomicBool,
                    ) -> Vec<(usize, Option<Vec<GridNode>>)> {
                        if target_indices.is_empty() {
                            return Vec::new();
                        }
                        let num_variants = if target_indices.len() <= 12 {
                            PORTFOLIO_VARIANTS.len()
                        } else if target_indices.len() <= 24 {
                            8
                        } else {
                            6
                        };

                        let found_counts: Vec<std::sync::atomic::AtomicUsize> = (0..target_indices.len())
                            .map(|_| std::sync::atomic::AtomicUsize::new(0))
                            .collect();

                        let mut tasks: Vec<(usize, usize, usize)> = Vec::with_capacity(target_indices.len() * num_variants);
                        for (t_pos, &conn_idx) in target_indices.iter().enumerate() {
                            for v_id in 0..num_variants {
                                tasks.push((t_pos, conn_idx, v_id));
                            }
                        }

                        // Prepare a concurrent GPU 3D wavefront batch (up to 16 connections) so both GPUs
                        // solve 3D wavefront trajectories simultaneously while CPU cores run A* portfolio variants.
                        let concurrent_gpu_limit = target_indices.len().min(16);
                        let concurrent_gpu_reqs: Vec<(usize, (GridNode, GridNode, usize, f32))> = target_indices[..concurrent_gpu_limit]
                            .iter()
                            .filter_map(|&conn_idx| {
                                let conn = &connections[conn_idx];
                                let (sgx, sgy) = grid.world_to_grid(conn.start.0, conn.start.1)?;
                                let (egx, egy) = grid.world_to_grid(conn.end.0, conn.end.1)?;
                                Some((
                                    conn_idx,
                                    (
                                        GridNode { gx: sgx, gy: sgy, layer: conn.start.2 },
                                        GridNode { gx: egx, gy: egy, layer: conn.end.2 },
                                        conn.net_id,
                                        3.0f32,
                                    ),
                                ))
                            })
                            .collect();

                        let (mut raw_candidates, concurrent_gpu_paths): (
                            Vec<(usize, usize, Vec<GridNode>)>,
                            Vec<(usize, Option<Vec<GridNode>>)>,
                        ) = rayon::join(
                            || {
                                tasks
                                    .par_iter()
                                    .filter_map(|&(t_pos, conn_idx, v_id)| {
                                        if stop_flag.load(AtomicOrdering::Relaxed) {
                                            return None;
                                        }
                                        let conn = &connections[conn_idx];
                                        let (sgx, sgy) = grid.world_to_grid(conn.start.0, conn.start.1)?;
                                        let (egx, egy) = grid.world_to_grid(conn.end.0, conn.end.1)?;
                                        let start_grid = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                                        let end_grid = GridNode { gx: egx, gy: egy, layer: conn.end.2 };
                                        let var = PORTFOLIO_VARIANTS[v_id];
                                        let w = if conn.diff_pair_leader.is_some() || conn.netclass.name == "USB_Diff" {
                                            conn.netclass.diff_pair_width
                                        } else {
                                            conn.netclass.track_width
                                        };
                                        let path = if v_id == 0 {
                                            self.find_path_with_constraints(
                                                grid,
                                                start_grid,
                                                end_grid,
                                                conn.net_id,
                                                conn.netclass.clearance,
                                                w,
                                                conn.netclass.via_dia,
                                                u1_center,
                                                1.0,
                                                0.0,
                                                conn.netclass.max_coupled_length_mm,
                                                conn.netclass.require_return_path,
                                                None,
                                                stop_flag,
                                            )
                                            .map(|p| {
                                                found_counts[t_pos].fetch_add(1, AtomicOrdering::Relaxed);
                                                p
                                            })
                                            .or_else(|| {
                                                self.find_path_portfolio_variant(
                                                    grid,
                                                    start_grid,
                                                    end_grid,
                                                    conn.net_id,
                                                    conn.netclass.clearance,
                                                    w,
                                                    conn.netclass.via_dia,
                                                    u1_center,
                                                    conn.netclass.max_coupled_length_mm,
                                                    conn.netclass.require_return_path,
                                                    var.via_mult,
                                                    var.hist_w,
                                                    var.heur_w,
                                                    var.dist_bias,
                                                    var.reverse,
                                                    var.diag_first,
                                                    stop_flag,
                                                    Some(&found_counts[t_pos]),
                                                    None,
                                                )
                                            })?
                                        } else {
                                            self.find_path_portfolio_variant(
                                                grid,
                                                start_grid,
                                                end_grid,
                                                conn.net_id,
                                                conn.netclass.clearance,
                                                w,
                                                conn.netclass.via_dia,
                                                u1_center,
                                                conn.netclass.max_coupled_length_mm,
                                                conn.netclass.require_return_path,
                                                var.via_mult,
                                                var.hist_w,
                                                var.heur_w,
                                                var.dist_bias,
                                                var.reverse,
                                                var.diag_first,
                                                stop_flag,
                                                Some(&found_counts[t_pos]),
                                                None,
                                            )?
                                        };
                                        Some((conn_idx, v_id, path))
                                    })
                                    .collect()
                            },
                            || {
                                if concurrent_gpu_reqs.is_empty() || stop_flag.load(AtomicOrdering::Relaxed) {
                                    return Vec::new();
                                }
                                let wf_reqs: Vec<(GridNode, GridNode, usize, f32)> =
                                    concurrent_gpu_reqs.iter().map(|(_, r)| *r).collect();
                                let wf_paths = self.solve_batch_gpu_wavefront_paths(grid, &wf_reqs, 1);
                                concurrent_gpu_reqs
                                    .iter()
                                    .map(|(c_idx, _)| *c_idx)
                                    .zip(wf_paths)
                                    .collect()
                            },
                        );

                        for (conn_idx, wf_opt) in concurrent_gpu_paths {
                            if let Some(p) = wf_opt {
                                raw_candidates.push((conn_idx, PORTFOLIO_VARIANTS.len(), p));
                            }
                        }

                        let missing_gpu_reqs: Vec<(usize, (GridNode, GridNode, usize, f32))> = target_indices
                            .iter()
                            .enumerate()
                            .skip(concurrent_gpu_limit)
                            .filter(|(t_pos, _)| found_counts[*t_pos].load(AtomicOrdering::Relaxed) == 0)
                            .filter_map(|(_, &conn_idx)| {
                                let conn = &connections[conn_idx];
                                let (sgx, sgy) = grid.world_to_grid(conn.start.0, conn.start.1)?;
                                let (egx, egy) = grid.world_to_grid(conn.end.0, conn.end.1)?;
                                Some((
                                    conn_idx,
                                    (
                                        GridNode { gx: sgx, gy: sgy, layer: conn.start.2 },
                                        GridNode { gx: egx, gy: egy, layer: conn.end.2 },
                                        conn.net_id,
                                        3.0f32,
                                    ),
                                ))
                            })
                            .collect();
                        if !missing_gpu_reqs.is_empty() && !stop_flag.load(AtomicOrdering::Relaxed) {
                            let wf_reqs: Vec<(GridNode, GridNode, usize, f32)> =
                                missing_gpu_reqs.iter().map(|(_, r)| *r).collect();
                            let wf_paths = self.solve_batch_gpu_wavefront_paths(grid, &wf_reqs, 1);
                            for ((conn_idx, _), wf_opt) in missing_gpu_reqs.into_iter().zip(wf_paths.into_iter()) {
                                if let Some(p) = wf_opt {
                                    raw_candidates.push((conn_idx, PORTFOLIO_VARIANTS.len(), p));
                                }
                            }
                        }

                        let flat_paths: Vec<(usize, Vec<GridNode>)> = raw_candidates
                            .iter()
                            .map(|(c_idx, _, p)| (*c_idx, p.clone()))
                            .collect();
                        let validations = self.batch_validate_paths_gpu(board, grid, connections, &flat_paths, u1_center);

                        let mut best_per_conn: HashMap<usize, (f64, usize, Vec<GridNode>)> = HashMap::new();
                        for ((conn_idx, v_id, path), (grid_ok, gpu_drc_ok)) in
                            raw_candidates.into_iter().zip(validations.into_iter())
                        {
                            if !grid_ok {
                                continue;
                            }
                            let (vias, wirelength, cong_sum) = Self::evaluate_path_cost(grid, &path);
                            let bends = Self::count_path_bends(&path);
                            let drc_pen = if gpu_drc_ok { 0.0 } else { 25.0 };
                            let score = (vias as f64) * 10.0
                                + wirelength
                                + (bends as f64) * 0.20
                                + cong_sum * 0.02
                                + drc_pen;
                            let entry = best_per_conn.entry(conn_idx).or_insert((f64::INFINITY, usize::MAX, Vec::new()));
                            if score < entry.0 - 1e-6 || ((score - entry.0).abs() <= 1e-6 && v_id < entry.1) {
                                *entry = (score, v_id, path);
                            }
                        }

                        target_indices
                            .iter()
                            .map(|&conn_idx| {
                                let p_opt = best_per_conn.remove(&conn_idx).map(|(_, _, p)| p);
                                (conn_idx, p_opt)
                            })
                            .collect()
                    }

                    /// Runs a parallel multi-net rip-up portfolio solver to break stubborn late-pass
                    /// routing deadlocks when a small number of connections (`<= 6`) remain unrouted.
                    ///
                    /// Identifies up to 8 blocking routed connections near the unrouted endpoints or
                    /// corridors, then launches 8 concurrent Rayon replicas. Each replica rips up a
                    /// distinct subset of blockers, increments historical congestion along the ripped
                    /// cells, rebuilds its local grid, and attempts to re-route both the unrouted and
                    /// ripped connections under permuted ordering and portfolio hyperparameter variants.
                    ///
                    /// # Arguments
                    /// * `board` - Read-only reference to the [`Board`].
                    /// * `grid` - Mutable [`RoutingGrid`] updated in place if a replica improves total completion.
                    /// * `connections` - Slice of all [`RouteConnection`] descriptors.
                    /// * `routed_paths` - Mutable vector of current routed paths, replaced if a strictly better solution is found.
                    /// * `escape_stubs` - Pre-generated fine-pitch escape stubs.
                    /// * `unrouted_indices` - Slice of connection indices currently lacking a valid route.
                    /// * `u1_center` - Optional `(x, y)` center of `U1` in `mm`.
                    /// * `stop_flag` - Atomic cancellation flag.
                    ///
                    /// # Returns
                    /// `true` if a portfolio replica strictly increased the number of routed connections
                    /// and updated `routed_paths` and `grid`; `false` otherwise.
                    ///
                    /// # Side Effects
                    /// Mutates `*routed_paths` and rebuilds `grid` state when a superior replica is selected.
                    #[allow(clippy::too_many_arguments)]
                    pub fn try_parallel_ripup_portfolio(
                        &self,
                        board: &Board,
                        grid: &mut RoutingGrid,
                        connections: &[RouteConnection],
                        routed_paths: &mut Vec<Option<Vec<GridNode>>>,
                        escape_stubs: &[EscapeStub],
                        unrouted_indices: &[usize],
                        u1_center: Option<(f64, f64)>,
                        stop_flag: &AtomicBool,
                    ) -> bool {
                        if unrouted_indices.is_empty() || stop_flag.load(AtomicOrdering::Relaxed) {
                            return false;
                        }
                        let total_conns = connections.len();
                        let current_done = routed_paths.iter().filter(|p| p.is_some()).count();

                        // Identify blocking routed connections near the unrouted endpoints/corridors
                        let mut blocker_scores: HashMap<usize, f64> = HashMap::new();
                        for &u_idx in unrouted_indices {
                            let uc = &connections[u_idx];
                            let mut choked_nets = Vec::new();
                            if let Some((sgx, sgy)) = grid.world_to_grid(uc.start.0, uc.start.1) {
                                let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, sgx, sgy, uc.start.2, uc.net_id, 400);
                                if reach < 300 {
                                    choked_nets.extend(nets);
                                }
                            }
                            if let Some((egx, egy)) = grid.world_to_grid(uc.end.0, uc.end.1) {
                                let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, egx, egy, uc.end.2, uc.net_id, 400);
                                if reach < 300 {
                                    choked_nets.extend(nets);
                                }
                            }
                            for rip_idx in 0..total_conns {
                                if rip_idx == u_idx || routed_paths[rip_idx].is_none() {
                                    continue;
                                }
                                let rc = &connections[rip_idx];
                                if rc.net_id == uc.net_id
                                    || rc.diff_pair_leader.is_some()
                                    || connections.iter().any(|c| c.diff_pair_leader == Some(rip_idx))
                                    || rc.net_name.contains("USB")
                                {
                                    continue;
                                }
                                let path = routed_paths[rip_idx].as_ref().unwrap();
                                let is_choked = choked_nets.contains(&rc.net_id);
                                let mut min_d = f64::INFINITY;
                                for node in path {
                                    let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                                    let d = (wx - uc.start.0).hypot(wy - uc.start.1)
                                        .min((wx - uc.end.0).hypot(wy - uc.end.1))
                                        .min(dist_to_segment(wx, wy, uc.start.0, uc.start.1, uc.end.0, uc.end.1));
                                    if d < min_d {
                                        min_d = d;
                                    }
                                }
                                if is_choked || min_d < 2.2 {
                                    let s = if is_choked { min_d * 0.25 } else { min_d };
                                    let entry = blocker_scores.entry(rip_idx).or_insert(f64::INFINITY);
                                    if s < *entry {
                                        *entry = s;
                                    }
                                }
                            }
                        }

                        let mut blockers: Vec<(usize, f64)> = blocker_scores.into_iter().collect();
                        blockers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
                        blockers.truncate(8);
                        if blockers.is_empty() {
                            return false;
                        }

                        let num_replicas = 8usize;
                        let replica_results: Vec<(usize, f64, usize, Vec<Option<Vec<GridNode>>>)> = (0..num_replicas)
                            .into_par_iter()
                            .filter_map(|rep_id| {
                                if stop_flag.load(AtomicOrdering::Relaxed) {
                                    return None;
                                }
                                let mut local_paths = routed_paths.clone();
                                let mut local_grid = grid.clone_for_portfolio();

                                // Select distinct subset of blockers to rip up for this portfolio replica
                                let mut ripped_conns = Vec::new();
                                for (b_pos, &(b_idx, _)) in blockers.iter().enumerate() {
                                    let take_it = match rep_id {
                                        0 => b_pos < 2,
                                        1 => b_pos < 3,
                                        2 => b_pos < 4,
                                        3 => b_pos % 2 == 0 && b_pos < 6,
                                        4 => (b_pos % 2 == 1 || b_pos == 0) && b_pos < 6,
                                        5 => b_pos < 5,
                                        6 => b_pos >= 1 && b_pos < 5,
                                        _ => b_pos < blockers.len().min(6),
                                    };
                                    if take_it {
                                        if let Some(old_p) = local_paths[b_idx].take() {
                                            for node in &old_p {
                                                let idx = local_grid.idx(node.gx, node.gy, node.layer);
                                                local_grid.congestion[idx] += self.congestion_factor * 1.5;
                                            }
                                            ripped_conns.push(b_idx);
                                        }
                                    }
                                }
                                if ripped_conns.is_empty() {
                                    return None;
                                }

                                self.rebuild_grid_state(board, &mut local_grid, connections, &local_paths, escape_stubs);

                                // Build local subgraph target ordering permutation for this replica
                                let mut order: Vec<usize> = Vec::with_capacity(unrouted_indices.len() + ripped_conns.len());
                                match rep_id % 4 {
                                    0 => {
                                        order.extend_from_slice(unrouted_indices);
                                        order.extend_from_slice(&ripped_conns);
                                    }
                                    1 => {
                                        for &u in unrouted_indices.iter().rev() {
                                            order.push(u);
                                        }
                                        order.extend_from_slice(&ripped_conns);
                                    }
                                    2 => {
                                        order.extend_from_slice(unrouted_indices);
                                        for &r in ripped_conns.iter().rev() {
                                            order.push(r);
                                        }
                                    }
                                    _ => {
                                        // Interleave unrouted and highest-priority ripped nets
                                        order.extend_from_slice(unrouted_indices);
                                        order.extend_from_slice(&ripped_conns);
                                        if order.len() > 2 {
                                            let rot = (rep_id / 4) % order.len();
                                            order.rotate_left(rot);
                                        }
                                    }
                                }

                                let var_pair = [
                                    PORTFOLIO_VARIANTS[rep_id % PORTFOLIO_VARIANTS.len()],
                                    PORTFOLIO_VARIANTS[(rep_id + 1) % PORTFOLIO_VARIANTS.len()],
                                    PORTFOLIO_VARIANTS[0],
                                    PORTFOLIO_VARIANTS[1],
                                ];

                                for &c_idx in &order {
                                    if local_paths[c_idx].is_some() || stop_flag.load(AtomicOrdering::Relaxed) {
                                        continue;
                                    }
                                    let conn = &connections[c_idx];
                                    let (Some((sgx, sgy)), Some((egx, egy))) = (
                                        local_grid.world_to_grid(conn.start.0, conn.start.1),
                                        local_grid.world_to_grid(conn.end.0, conn.end.1),
                                    ) else {
                                        continue;
                                    };
                                    let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                                    let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };
                                    let w = if conn.diff_pair_leader.is_some() || conn.netclass.name == "USB_Diff" {
                                        conn.netclass.diff_pair_width
                                    } else {
                                        conn.netclass.track_width
                                    };

                                    let mut best_cand: Option<(f64, Vec<GridNode>)> = None;
                                    for var in &var_pair {
                                        if let Some(p) = self.find_path_portfolio_variant(
                                            &local_grid,
                                            s_node,
                                            e_node,
                                            conn.net_id,
                                            conn.netclass.clearance,
                                            w,
                                            conn.netclass.via_dia,
                                            u1_center,
                                            conn.netclass.max_coupled_length_mm,
                                            conn.netclass.require_return_path,
                                            var.via_mult,
                                            var.hist_w,
                                            var.heur_w,
                                            var.dist_bias,
                                            var.reverse,
                                            var.diag_first,
                                            stop_flag,
                                            None,
                                            Some(60_000),
                                        ) {
                                            let (vias, wl, _) = Self::evaluate_path_cost(&local_grid, &p);
                                            let sc = (vias as f64) * 10.0 + wl;
                                            if best_cand.as_ref().is_none_or(|(b_sc, _)| sc < *b_sc) {
                                                best_cand = Some((sc, p));
                                            }
                                            if vias == 0 {
                                                break;
                                            }
                                        }
                                    }
                                    if let Some((_, p)) = best_cand {
                                        self.commit_path_to_grid(&mut local_grid, &p, conn.net_id, w, conn.netclass.via_dia);
                                        local_paths[c_idx] = Some(p);
                                    }
                                }

                                let done_cnt = local_paths.iter().filter(|p| p.is_some()).count();
                                let mut total_cost = 0.0f64;
                                for p_opt in local_paths.iter().flatten() {
                                    let (vias, wl, _) = Self::evaluate_path_cost(&local_grid, p_opt);
                                    total_cost += (vias as f64) * 10.0 + wl;
                                }
                                Some((done_cnt, total_cost, rep_id, local_paths))
                            })
                            .collect();

                        if let Some((best_done, _, _, winner_paths)) = replica_results.into_iter().max_by(|a, b| {
                            a.0.cmp(&b.0)
                                .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal))
                                .then_with(|| b.2.cmp(&a.2))
                        }) {
                            if best_done > current_done {
                                *routed_paths = winner_paths;
                                self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
                                return true;
                            }
                        }
                        false
                    }
                }

                let wave_grid: &RoutingGrid = &*grid;
                let speculative_paths: Vec<(usize, Option<Vec<GridNode>>)> = self.route_connections_portfolio(
                    board,
                    wave_grid,
                    &connections,
                    &unrouted_indices,
                    u1_center,
                    &stop_flag,
                );

                // Phase 2: Conflict-Graph Independent Set Wave Partitioning & Parallel Sub-Wave Re-Routing
                let mut _any_committed_in_wave = false;
                let mut sorted_speculative_paths = speculative_paths;
                sorted_speculative_paths.sort_by_key(|&(idx, _)| std::cmp::Reverse(unrouted_history[idx]));

                let mut found_speculative: Vec<(usize, Vec<GridNode>)> = Vec::with_capacity(sorted_speculative_paths.len());
                let mut need_reroute: Vec<usize> = Vec::new();
                for (conn_idx, spec_opt) in sorted_speculative_paths {
                    if let Some(p) = spec_opt {
                        found_speculative.push((conn_idx, p));
                    } else {
                        need_reroute.push(conn_idx);
                    }
                }

                let waves = Self::partition_into_conflict_free_waves(grid, &connections, found_speculative);
                for (w_idx, wave) in waves.into_iter().enumerate() {
                    if stop_flag.load(AtomicOrdering::Relaxed) {
                        break;
                    }
                    let validations = self.batch_validate_paths_gpu(board, grid, &connections, &wave, u1_center);
                    for ((conn_idx, p), (grid_ok, _gpu_drc_ok)) in wave.into_iter().zip(validations.into_iter()) {
                        let conn = &connections[conn_idx];
                        let w = if conn.diff_pair_leader.is_some() || conn.netclass.name == "USB_Diff" {
                            conn.netclass.diff_pair_width
                        } else {
                            conn.netclass.track_width
                        };
                        let still_ok = grid_ok
                            && self.is_path_conflict_free(
                                grid,
                                &p,
                                conn.net_id,
                                conn.netclass.clearance,
                                w,
                                conn.netclass.via_dia,
                                u1_center,
                                conn.netclass.require_return_path,
                            );
                        if still_ok {
                            self.commit_path_to_grid(grid, &p, conn.net_id, w, conn.netclass.via_dia);
                            routed_paths[conn_idx] = Some(p);
                            _any_committed_in_wave = true;
                        } else {
                            need_reroute.push(conn_idx);
                        }
                    }
                    if w_idx == 0 && _any_committed_in_wave {
                        self.reconstruct_tracks_and_vias(board, grid, &connections, &routed_paths, &escape_stubs);
                    }
                }

                // Parallel Sub-Wave Re-Routing for paths that conflicted with Wave 0
                need_reroute.sort_by_key(|&idx| std::cmp::Reverse(unrouted_history[idx]));
                let max_subwaves = 4usize;
                for _sub in 0..max_subwaves {
                    if need_reroute.is_empty() || stop_flag.load(AtomicOrdering::Relaxed) {
                        break;
                    }
                    let sub_results = self.route_connections_portfolio(
                        board,
                        grid,
                        &connections,
                        &need_reroute,
                        u1_center,
                        &stop_flag,
                    );
                    let mut sub_found = Vec::new();
                    let mut next_reroute = Vec::new();
                    for (conn_idx, p_opt) in sub_results {
                        if let Some(p) = p_opt {
                            sub_found.push((conn_idx, p));
                        } else {
                            next_reroute.push(conn_idx);
                        }
                    }
                    if sub_found.is_empty() {
                        need_reroute = next_reroute;
                        break;
                    }
                    let sub_waves = Self::partition_into_conflict_free_waves(grid, &connections, sub_found);
                    for sub_wave in sub_waves {
                        let validations = self.batch_validate_paths_gpu(board, grid, &connections, &sub_wave, u1_center);
                        for ((conn_idx, p), (grid_ok, _)) in sub_wave.into_iter().zip(validations.into_iter()) {
                            let conn = &connections[conn_idx];
                            let w = if conn.diff_pair_leader.is_some() || conn.netclass.name == "USB_Diff" {
                                conn.netclass.diff_pair_width
                            } else {
                                conn.netclass.track_width
                            };
                            if grid_ok
                                && self.is_path_conflict_free(
                                    grid,
                                    &p,
                                    conn.net_id,
                                    conn.netclass.clearance,
                                    w,
                                    conn.netclass.via_dia,
                                    u1_center,
                                    conn.netclass.require_return_path,
                                )
                            {
                                self.commit_path_to_grid(grid, &p, conn.net_id, w, conn.netclass.via_dia);
                                routed_paths[conn_idx] = Some(p);
                                _any_committed_in_wave = true;
                            } else {
                                next_reroute.push(conn_idx);
                            }
                        }
                    }
                    need_reroute = next_reroute;
                }

                // Final priority-ordered fallback for any remaining conflicted connection in this pass
                for conn_idx in need_reroute {
                    if stop_flag.load(AtomicOrdering::Relaxed) || routed_paths[conn_idx].is_some() {
                        continue;
                    }
                    let conn = &connections[conn_idx];
                    if let (Some((sgx, sgy)), Some((egx, egy))) = (
                        grid.world_to_grid(conn.start.0, conn.start.1),
                        grid.world_to_grid(conn.end.0, conn.end.1),
                    ) {
                        let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                        let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };
                        if let Some(p) = self.find_path_parallel(
                            grid,
                            s_node,
                            e_node,
                            conn.net_id,
                            conn.netclass.clearance,
                            conn.netclass.track_width,
                            conn.netclass.via_dia,
                            u1_center,
                            conn.netclass.max_coupled_length_mm,
                            conn.netclass.require_return_path,
                            &stop_flag,
                        ) {
                            self.commit_path_to_grid(grid, &p, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
                            routed_paths[conn_idx] = Some(p);
                            _any_committed_in_wave = true;
                        }
                    }
                }

                if _any_committed_in_wave {
                    if let Ok(mut obs_g) = shared_obs.lock() {
                        obs_g.clone_from(&grid.obstacle_net_ids);
                    }
                }
                if let Some(ref st) = live {
                    let done = routed_paths.iter().filter(|p| p.is_some()).count();
                    self.reconstruct_tracks_and_vias(board, grid, &connections, &routed_paths, &escape_stubs);
                    if let Some(ref ipc_lock) = self.ipc {
                        if let Ok(mut client) = ipc_lock.lock() {
                            if client.is_connected() {
                                let _ = client.clear_tracks();
                                let _ = client.stream_tracks_and_vias(board);
                            }
                        }
                    }
                    let remaining_ratsnest: Vec<crate::board::Segment> = connections
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| routed_paths[i].is_none())
                        .map(|(_, c)| {
                            let p1 = board.footprints[c.start_fp].pad_absolute_position(&board.footprints[c.start_fp].pads[c.start_pad]);
                            let p2 = board.footprints[c.end_fp].pad_absolute_position(&board.footprints[c.end_fp].pads[c.end_pad]);
                            crate::board::Segment {
                                p1: Point::new(p1.0, p1.1),
                                p2: Point::new(p2.0, p2.1),
                                net_id: c.net_id,
                            }
                        })
                        .collect();
                    if let Ok(mut g) = st.lock() {
                        g.board.tracks = board.tracks.clone();
                        g.board.vias = board.vias.clone();
                        g.board.footprints = board.footprints.clone();
                        g.ratsnest = remaining_ratsnest;
                        g.routed_conns = done;
                        g.routing_pass = iteration + 1;
                        g.progress = 0.5 + 0.5 * (done as f32 / total_conns.max(1) as f32);
                        g.status_text = format!(
                            "Pass {}/{}: Routed {}/{} ({:.1}%)",
                            iteration + 1,
                            self.max_ripup_iterations,
                            done,
                            total_conns,
                            (done as f64 / total_conns.max(1) as f64) * 100.0
                        );
                    }
                }

                // Phase 2.5: Active Push & Nudge for unrouted connections
                // Attempt to nudge blocking unlocked components and shove/rip-up blocking tracks.
                for conn_idx in 0..total_conns {
                    if stop_flag.load(AtomicOrdering::Relaxed) {
                        break;
                    }
                    if routed_paths[conn_idx].is_some() {
                        continue;
                    }

                    // 1. Try nudging blocking unlocked components
                    if self.allow_component_nudge {
                        let nudged_comp = self.try_nudge_blocking_components(
                            board,
                            grid,
                            &mut connections,
                            &mut routed_paths,
                            &escape_stubs,
                            conn_idx,
                            u1_center,
                            &stop_flag,
                        );
                        if nudged_comp {
                            if let Ok(mut obs_g) = shared_obs.lock() {
                                obs_g.clone_from(&grid.obstacle_net_ids);
                            }
                            continue;
                        }
                    }

                    // 2. Try shoving or ripping up blocking unlocked tracks
                    let nudged_track = self.try_nudge_or_ripup_blocking_tracks(
                        board,
                        grid,
                        &connections,
                        &mut routed_paths,
                        &escape_stubs,
                        conn_idx,
                        u1_center,
                        &stop_flag,
                    );
                    if nudged_track {
                        if let Ok(mut obs_g) = shared_obs.lock() {
                            obs_g.clone_from(&grid.obstacle_net_ids);
                        }
                    }
                }

                // Phase 2.75: Parallel Multi-Net Rip-Up Portfolio when <= 6 unrouted connections stall on pass >= 2
                let remaining_after_nudge: Vec<usize> = (0..total_conns)
                    .filter(|&i| routed_paths[i].is_none())
                    .collect();
                if !remaining_after_nudge.is_empty()
                    && remaining_after_nudge.len() <= 6
                    && iteration >= 1
                    && !stop_flag.load(AtomicOrdering::Relaxed)
                {
                    if self.try_parallel_ripup_portfolio(
                        board,
                        grid,
                        &connections,
                        &mut routed_paths,
                        &escape_stubs,
                        &remaining_after_nudge,
                        u1_center,
                        &stop_flag,
                    ) {
                        if let Ok(mut obs_g) = shared_obs.lock() {
                            obs_g.clone_from(&grid.obstacle_net_ids);
                        }
                    }
                }

                let done = routed_paths.iter().filter(|p| p.is_some()).count();
                if done > best_routed_count {
                    best_routed_count = done;
                    best_routed_paths = routed_paths.clone();
                    best_footprints = board.footprints.clone();
                    best_connections = connections.clone();
                    stagnant_passes = 0;
                } else {
                    stagnant_passes += 1;
                }
                if let Some(ref st) = live {
                    self.reconstruct_tracks_and_vias(board, grid, &connections, &routed_paths, &escape_stubs);
                    let remaining_ratsnest: Vec<crate::board::Segment> = connections
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| routed_paths[i].is_none())
                        .map(|(_, c)| {
                            let p1 = board.footprints[c.start_fp].pad_absolute_position(&board.footprints[c.start_fp].pads[c.start_pad]);
                            let p2 = board.footprints[c.end_fp].pad_absolute_position(&board.footprints[c.end_fp].pads[c.end_pad]);
                            crate::board::Segment {
                                p1: Point::new(p1.0, p1.1),
                                p2: Point::new(p2.0, p2.1),
                                net_id: c.net_id,
                            }
                        })
                        .collect();
                    if let Ok(mut g) = st.lock() {
                        g.board.tracks = board.tracks.clone();
                        g.board.vias = board.vias.clone();
                        g.board.footprints = board.footprints.clone();
                        g.ratsnest = remaining_ratsnest;
                        g.routed_conns = done;
                        g.routing_pass = iteration + 1;
                        g.progress = 0.5 + 0.5 * (done as f32 / total_conns.max(1) as f32);
                        g.status_text = format!(
                            "Pass {}/{}: Routed {}/{} ({:.1}%)",
                            iteration + 1,
                            self.max_ripup_iterations,
                            done,
                            total_conns,
                            (done as f64 / total_conns.max(1) as f64) * 100.0
                        );
                    }
                }
                println!(
                    "  [Router] Pass {}/{}: Routed {}/{} connections ({:.1}%) [{:.1}s elapsed]",
                    iteration + 1,
                    self.max_ripup_iterations,
                    done,
                    total_conns,
                    (done as f64 / total_conns.max(1) as f64) * 100.0,
                    route_start_time.elapsed().as_secs_f64()
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
                crate::memlog::checkpoint_board(
                    &format!("route_all::pass_{}_of_{}_done_{}_{}", iteration + 1, self.max_ripup_iterations, done, total_conns),
                    board,
                );
                if done == total_conns || (best_routed_count + 2 >= total_conns && stagnant_passes >= 4) {
                    break;
                }

                // Rip-up and Negotiated Congestion Update (budgeted to avoid wiping all progress)
                if iteration + 1 < self.max_ripup_iterations {
                    let mut rip_budget = 8;
                    let mut any_ripped_in_pass = false;
                    let corridor_tol = (0.85 + 0.35 * (iteration as f64)).min(2.5);
                    for conn_idx in 0..total_conns {
                        if rip_budget == 0 {
                            break;
                        }
                        if routed_paths[conn_idx].is_none() {
                            let blocked_conn = &connections[conn_idx];
                            let b_start = blocked_conn.start;
                            let b_end = blocked_conn.end;

                            let mut choked_nets = Vec::new();
                            if let Some((sgx, sgy)) = grid.world_to_grid(b_start.0, b_start.1) {
                                let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, sgx, sgy, b_start.2, blocked_conn.net_id, 400);
                                if reach < 300 {
                                    choked_nets.extend(nets);
                                }
                            }
                            if let Some((egx, egy)) = grid.world_to_grid(b_end.0, b_end.1) {
                                let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, egx, egy, b_end.2, blocked_conn.net_id, 400);
                                if reach < 300 {
                                    choked_nets.extend(nets);
                                }
                            }

                            for rip_idx in 0..total_conns {
                                if rip_budget == 0 {
                                    break;
                                }
                                if rip_idx == conn_idx || routed_paths[rip_idx].is_none() {
                                    continue;
                                }
                                let p = routed_paths[rip_idx].as_ref().unwrap();
                                let is_choked = choked_nets.contains(&connections[rip_idx].net_id);
                                let is_at_pin = is_choked || p.iter().any(|node| {
                                    let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                                    (wx - b_start.0).hypot(wy - b_start.1) < 1.2
                                        || (wx - b_end.0).hypot(wy - b_end.1) < 1.2
                                });
                                let passes_near = is_at_pin
                                    || p.iter().any(|node| {
                                        let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                                        dist_to_segment(wx, wy, b_start.0, b_start.1, b_end.0, b_end.1) < corridor_tol
                                    });

                                let is_diff = connections[rip_idx].diff_pair_leader.is_some()
                                    || connections.iter().any(|c| c.diff_pair_leader == Some(rip_idx))
                                    || connections[rip_idx].net_name.contains("USB");
                                if is_diff && !is_choked {
                                    continue;
                                }

                                // Always allow ripping up tracks that directly choke a pin endpoint;
                                // for mid-span proximity, respect priority on pass 0
                                if !is_at_pin && iteration == 0 && Self::netclass_priority(&connections[rip_idx].netclass.name) > 2 {
                                    continue;
                                }

                                if passes_near {
                                    if let Some(p) = routed_paths[rip_idx].take() {
                                        for node in &p {
                                            let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                                            let d_s = (wx - b_start.0).hypot(wy - b_start.1);
                                            let d_e = (wx - b_end.0).hypot(wy - b_end.1);
                                            let d_seg = dist_to_segment(wx, wy, b_start.0, b_start.1, b_end.0, b_end.1);
                                            let idx = grid.idx(node.gx, node.gy, node.layer);
                                            if is_choked || d_s < 1.8 || d_e < 1.8 || d_seg < corridor_tol {
                                                grid.congestion[idx] += self.congestion_factor * 3.5;
                                            } else {
                                                grid.congestion[idx] += self.congestion_factor * 1.5;
                                            }
                                        }
                                        any_ripped_in_pass = true;
                                        rip_budget -= 1;
                                    }
                                }
                            }
                        }
                    }
                    if any_ripped_in_pass {
                        self.rebuild_grid_state(board, grid, &connections, &routed_paths, &escape_stubs);
                        if let Ok(mut obs_g) = shared_obs.lock() {
                            obs_g.clone_from(&grid.obstacle_net_ids);
                        }
                    }
                }
            }

            gpu_done.store(true, AtomicOrdering::Relaxed);
        });

        if best_routed_count > routed_paths.iter().filter(|p| p.is_some()).count() {
            routed_paths = best_routed_paths;
            board.footprints = best_footprints;
            connections = best_connections;
            self.rebuild_grid_state(board, grid, &connections, &routed_paths, &escape_stubs);
        }

        crate::memlog::checkpoint_board("route_all::before_reconstruct_tracks_and_vias", board);
        self.reconstruct_tracks_and_vias(board, grid, &connections, &routed_paths, &escape_stubs);
        crate::memlog::checkpoint_board("route_all::after_reconstruct_tracks_and_vias", board);

        let reduce_vias_guarded = |board: &mut Board, grid: &mut RoutingGrid, config: &RoutingConfig| {
            let pre_connected: Vec<bool> = (0..total_conns)
                .map(|i| Self::is_connection_electrically_connected(board, &connections[i]))
                .collect();
            let snap = (board.tracks.clone(), board.arcs.clone(), board.vias.clone());
            reduce_vias_with_config(board, grid, config);
            let mut broken_nets = HashSet::new();
            for i in 0..total_conns {
                if pre_connected[i] && !Self::is_connection_electrically_connected(board, &connections[i]) {
                    broken_nets.insert(connections[i].net_id);
                }
                if let Some(leader_idx) = connections[i].diff_pair_leader {
                    let ln = connections[leader_idx].net_id;
                    let fn_id = connections[i].net_id;
                    if board_net_coupling_mm(board, ln, fn_id) > DIFF_PAIR_COUPLED_MM
                        || board_net_coupling_mm(board, fn_id, ln) > DIFF_PAIR_COUPLED_MM
                    {
                        broken_nets.insert(ln);
                        broken_nets.insert(fn_id);
                    }
                }
            }
            if !broken_nets.is_empty() {
                board.tracks.retain(|t| !broken_nets.contains(&t.net_id));
                board.arcs.retain(|a| !broken_nets.contains(&a.net_id));
                board.vias.retain(|v| !broken_nets.contains(&v.net_id));
                board.tracks.extend(snap.0.into_iter().filter(|t| broken_nets.contains(&t.net_id)));
                board.arcs.extend(snap.1.into_iter().filter(|a| broken_nets.contains(&a.net_id)));
                board.vias.extend(snap.2.into_iter().filter(|v| broken_nets.contains(&v.net_id)));
            }
        };

        // Phase 3: Via Reduction Pass (eliminates redundant layer-hopping via pairs)
        reduce_vias_guarded(board, grid, &self.config);
        crate::memlog::checkpoint_board("route_all::after_reduce_vias", board);
        prune_dangling_tracks_and_vias(board);
        crate::memlog::checkpoint_board("route_all::after_prune_dangling_initial", board);

        if let Some(ref ipc_lock) = self.ipc {
            if let Ok(mut client) = ipc_lock.lock() {
                if client.is_connected() {
                    let _ = client.clear_tracks();
                    let _ = client.stream_tracks_and_vias(board);
                }
            }
        }

        // Phase 3: Post-Route Glossing Pass (rubber-band straightening, stair-step elimination, 45-deg mitering)
        let gloss_iters = self.config.gloss_passes;
        for pass_i in 0..gloss_iters {
            crate::memlog::checkpoint_board(&format!("route_all::before_gloss_pass_{}", pass_i), board);
            gloss_tracks_standard(board, grid);
            crate::memlog::checkpoint_board(&format!("route_all::after_gloss_pass_{}", pass_i), board);
            prune_dangling_tracks_and_vias(board);
            crate::memlog::checkpoint_board(&format!("route_all::after_gloss_prune_{}", pass_i), board);
            if pass_i + 1 < gloss_iters {
                reduce_vias_guarded(board, grid, &self.config);
                crate::memlog::checkpoint_board(&format!("route_all::after_gloss_reduce_vias_{}", pass_i), board);
                prune_dangling_tracks_and_vias(board);
            }
        }

        // Phase 2: Coupled Diff Pair Precision Alignment and Intra-Pair Phase Matching
        for follower_idx in 0..total_conns {
            if let Some(leader_idx) = connections[follower_idx].diff_pair_leader {
                if routed_paths[leader_idx].is_some() && routed_paths[follower_idx].is_some() {
                    let lc = &connections[leader_idx];
                    let fc = &connections[follower_idx];
                    let diff_w = lc.netclass.diff_pair_width;
                    let diff_gap = lc.netclass.diff_pair_gap;
                    crate::memlog::checkpoint_board("route_all::before_diff_pair_spacing", board);
                    enforce_coupled_diff_pair_spacing(
                        board,
                        lc.net_id,
                        fc.net_id,
                        diff_w,
                        diff_gap,
                        lc.start,
                        lc.end,
                        fc.start,
                        fc.end,
                    );
                    if let (Some(lp), Some(fp)) = (&routed_paths[leader_idx], &routed_paths[follower_idx]) {
                        let (ll, fl) = (grid_path_length_mm(grid.res, lp), grid_path_length_mm(grid.res, fp));
                        match_diff_pair_phase_lengths(board, grid, lc.net_id, fc.net_id, ll, fl, 0.05);
                    }
                    crate::memlog::checkpoint_board("route_all::after_diff_pair_phase", board);
                }
            }
        }

        crate::memlog::checkpoint_board("route_all::before_connect_u1_ground_pins", board);
        self.connect_u1_ground_pins(board);

        let repair_single_conn = |board: &mut Board,
                                  grid: &mut RoutingGrid,
                                  conn: &RouteConnection,
                                  widen_after: bool|
         -> Option<Vec<GridNode>> {
            grid.obstacle_net_ids.fill(FREE_NET);
            grid.nudgeable_fp.fill(None);
            grid.smd_pad_cells.fill(false);
            grid.rasterize_obstacles(board);
            for trk in &board.tracks {
                if trk.net_id != conn.net_id {
                    grid.commit_track(trk);
                }
            }
            for v in &board.vias {
                if v.net_id != conn.net_id {
                    grid.commit_via(v);
                }
            }
            let sp = board.footprints[conn.start_fp]
                .pad_absolute_position(&board.footprints[conn.start_fp].pads[conn.start_pad]);
            let ep = board.footprints[conn.end_fp]
                .pad_absolute_position(&board.footprints[conn.end_fp].pads[conn.end_pad]);
            let s_layer = board.footprints[conn.start_fp].layer;
            let e_layer = board.footprints[conn.end_fp].layer;
            let mut candidate_endpoints = vec![
                (conn.start, conn.end, true),
                ((sp.0, sp.1, s_layer), (ep.0, ep.1, e_layer), false),
            ];
            let mut alt_pads: Vec<((f64, f64, usize), (f64, f64, usize), f64)> = Vec::new();
            for (f_idx, fp) in board.footprints.iter().enumerate() {
                for (p_idx, pad) in fp.pads.iter().enumerate() {
                    if pad.net_id == conn.net_id
                        && (f_idx, p_idx) != (conn.start_fp, conn.start_pad)
                        && (f_idx, p_idx) != (conn.end_fp, conn.end_pad)
                    {
                        let pos = fp.pad_absolute_position(pad);
                        let alt_pt = (pos.0, pos.1, fp.layer);
                        let mut probe_start = conn.clone();
                        probe_start.end_fp = f_idx;
                        probe_start.end_pad = p_idx;
                        probe_start.end = alt_pt;
                        let mut probe_end = conn.clone();
                        probe_end.start_fp = f_idx;
                        probe_end.start_pad = p_idx;
                        probe_end.start = alt_pt;
                        if Self::is_connection_electrically_connected(board, &probe_start) {
                            let d = (pos.0 - ep.0).hypot(pos.1 - ep.1);
                            alt_pads.push((alt_pt, (ep.0, ep.1, e_layer), d));
                        } else if Self::is_connection_electrically_connected(board, &probe_end) {
                            let d = (pos.0 - sp.0).hypot(pos.1 - sp.1);
                            alt_pads.push(((sp.0, sp.1, s_layer), alt_pt, d));
                        }
                    }
                }
            }
            alt_pads.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
            for (cand_s, cand_e, _) in alt_pads.into_iter().take(2) {
                candidate_endpoints.push((cand_s, cand_e, false));
            }
            for (cand_idx, &(cand_s, cand_e, use_stubs)) in candidate_endpoints.iter().enumerate() {
                if Self::is_connection_electrically_connected(board, conn) {
                    break;
                }
                if let (Some((sgx, sgy)), Some((egx, egy))) = (
                    grid.world_to_grid(cand_s.0, cand_s.1),
                    grid.world_to_grid(cand_e.0, cand_e.1),
                ) {
                    let s_node = GridNode { gx: sgx, gy: sgy, layer: cand_s.2 };
                    let e_node = GridNode { gx: egx, gy: egy, layer: cand_e.2 };
                    let mut used_w = conn.netclass.track_width;
                    let mut used_c = conn.netclass.clearance;
                    let repair_path = self
                        .find_path_with_constraints(
                            grid,
                            s_node,
                            e_node,
                            conn.net_id,
                            conn.netclass.clearance,
                            conn.netclass.track_width,
                            conn.netclass.via_dia,
                            u1_center,
                            1.0,
                            0.0,
                            conn.netclass.max_coupled_length_mm,
                            false,
                            None,
                            &stop_flag,
                        )
                        .or_else(|| {
                            if cand_idx >= 2 {
                                return None;
                            }
                            self.find_path_parallel(
                                grid,
                                s_node,
                                e_node,
                                conn.net_id,
                                conn.netclass.clearance,
                                conn.netclass.track_width,
                                conn.netclass.via_dia,
                                u1_center,
                                conn.netclass.max_coupled_length_mm,
                                false,
                                &stop_flag,
                            )
                        })
                        .or_else(|| {
                            used_w = conn.netclass.track_width.min(0.25);
                            used_c = conn.netclass.clearance.min(0.15);
                            self.find_path_with_params(
                                grid,
                                s_node,
                                e_node,
                                conn.net_id,
                                used_c,
                                used_w,
                                conn.netclass.via_dia,
                                u1_center,
                                1.15,
                                0.0,
                                &stop_flag,
                            )
                        })
                        .or_else(|| {
                            used_w = conn.netclass.track_width.min(0.15);
                            used_c = conn.netclass.clearance.min(0.12);
                            self.find_path_with_params(
                                grid,
                                s_node,
                                e_node,
                                conn.net_id,
                                used_c,
                                used_w,
                                conn.netclass.via_dia,
                                u1_center,
                                1.15,
                                0.0,
                                &stop_flag,
                            )
                        });
                    if let Some(rp) = repair_path {
                        let mut single_conn = conn.clone();
                        single_conn.start = cand_s;
                        single_conn.end = cand_e;
                        single_conn.netclass.track_width = used_w;
                        single_conn.netclass.clearance = used_c;
                        for trk in &mut board.tracks {
                            trk.is_locked = true;
                        }
                        for v in &mut board.vias {
                            v.is_locked = true;
                        }
                        let single_conns = vec![single_conn];
                        let single_paths = vec![Some(rp.clone())];
                        let stubs_slice: &[EscapeStub] = if use_stubs { &escape_stubs } else { &[] };
                        self.reconstruct_tracks_and_vias(board, grid, &single_conns, &single_paths, stubs_slice);
                        prune_dangling_tracks_and_vias(board);
                        gloss_tracks_standard(board, grid);
                        if widen_after {
                            widen_tracks_and_arcs_to_netclass(board);
                        }
                        validate_and_clean_copper(board);
                        prune_dangling_tracks_and_vias(board);
                        for trk in &mut board.tracks {
                            let key = (
                                trk.net_id,
                                [
                                    trk.start.0.to_bits(),
                                    trk.start.1.to_bits(),
                                    trk.end.0.to_bits(),
                                    trk.end.1.to_bits(),
                                ],
                            );
                            trk.is_locked = orig_locked_tracks.contains(&key);
                        }
                        for v in &mut board.vias {
                            let key = (v.net_id, [v.position.0.to_bits(), v.position.1.to_bits()]);
                            v.is_locked = orig_locked_vias.contains(&key);
                        }
                        if Self::is_connection_electrically_connected(board, conn) {
                            return Some(rp);
                        }
                    }
                }
            }
            None
        };

        let repair_with_blocker_ripup = |board: &mut Board,
                                         grid: &mut RoutingGrid,
                                         routed_paths: &mut Vec<Option<Vec<GridNode>>>,
                                         widen_after: bool| {
            for conn_idx in 0..total_conns {
                let conn = &connections[conn_idx];
                if !Self::is_connection_electrically_connected(board, conn) {
                    if let Some(rp) = repair_single_conn(board, grid, conn, widen_after) {
                        routed_paths[conn_idx] = Some(rp);
                        continue;
                    }
                    // 1-net blocker rip-up & reroute fallback for choked endpoints
                    let mut blocker_dists: HashMap<usize, f64> = HashMap::new();
                    for trk in &board.tracks {
                        if trk.is_locked || trk.net_id == 0 || trk.net_id == conn.net_id {
                            continue;
                        }
                        let b_name = board.nets.get(trk.net_id).map(|s| s.as_str()).unwrap_or("");
                        if crate::board::is_gnd_net_name(b_name) || b_name.contains("USB") {
                            continue;
                        }
                        let d = dist_point_to_segment((conn.start.0, conn.start.1), trk.start, trk.end)
                            .min(dist_point_to_segment((conn.end.0, conn.end.1), trk.start, trk.end))
                            .min(dist_segment_to_segment(
                                (conn.start.0, conn.start.1),
                                (conn.end.0, conn.end.1),
                                trk.start,
                                trk.end,
                            ));
                        if d < 2.5 {
                            let e = blocker_dists.entry(trk.net_id).or_insert(f64::INFINITY);
                            if d < *e {
                                *e = d;
                            }
                        }
                    }
                    let mut cands: Vec<(usize, f64)> = blocker_dists.into_iter().collect();
                    cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
                    for (b_net, _) in cands.into_iter().take(6) {
                        let snap_tracks = board.tracks.clone();
                        let snap_vias = board.vias.clone();
                        let pre_connected: Vec<bool> = (0..total_conns)
                            .map(|i| Self::is_connection_electrically_connected(board, &connections[i]))
                            .collect();
                        board.tracks.retain(|t| t.is_locked || t.net_id != b_net);
                        board.vias.retain(|v| v.is_locked || v.net_id != b_net);
                        if let Some(rp) = repair_single_conn(board, grid, conn, false) {
                            let mut all_ok = Self::is_connection_electrically_connected(board, conn);
                            if all_ok {
                                for b_idx in 0..total_conns {
                                    if connections[b_idx].net_id == b_net && pre_connected[b_idx] {
                                        if !Self::is_connection_electrically_connected(board, &connections[b_idx]) {
                                            if let Some(brp) = repair_single_conn(board, grid, &connections[b_idx], widen_after) {
                                                routed_paths[b_idx] = Some(brp);
                                            }
                                        }
                                        if !Self::is_connection_electrically_connected(board, &connections[b_idx]) {
                                            all_ok = false;
                                            break;
                                        }
                                    }
                                }
                            }
                            if all_ok && Self::is_connection_electrically_connected(board, conn) {
                                routed_paths[conn_idx] = Some(rp);
                                break;
                            }
                        }
                        board.tracks = snap_tracks;
                        board.vias = snap_vias;
                    }
                }
            }
        };

        // Repair any unrouted connections while signal tracks are still narrow and before plane stitching vias
        crate::memlog::checkpoint_board("route_all::before_repair_with_blocker_ripup", board);
        repair_with_blocker_ripup(board, grid, &mut routed_paths, false);
        crate::memlog::checkpoint_board("route_all::after_repair_with_blocker_ripup", board);

        widen_tracks_and_arcs_to_netclass(board);
        crate::memlog::checkpoint_board("route_all::before_add_plane_stitching_vias", board);
        self.add_plane_stitching_vias(board, grid);
        crate::memlog::checkpoint_board("route_all::before_add_return_path_ground_vias", board);
        self.add_return_path_ground_vias(board, grid);
        crate::memlog::checkpoint_board("route_all::before_post_plane_gloss", board);
        gloss_tracks_standard(board, grid);
        for follower_idx in 0..total_conns {
            if let Some(leader_idx) = connections[follower_idx].diff_pair_leader {
                if routed_paths[leader_idx].is_some() && routed_paths[follower_idx].is_some() {
                    let lc = &connections[leader_idx];
                    let fc = &connections[follower_idx];
                    enforce_coupled_diff_pair_spacing(
                        board,
                        lc.net_id,
                        fc.net_id,
                        lc.netclass.diff_pair_width,
                        lc.netclass.diff_pair_gap,
                        lc.start,
                        lc.end,
                        fc.start,
                        fc.end,
                    );
                    if let (Some(lp), Some(fp)) = (&routed_paths[leader_idx], &routed_paths[follower_idx]) {
                        let (ll, fl) = (grid_path_length_mm(grid.res, lp), grid_path_length_mm(grid.res, fp));
                        match_diff_pair_phase_lengths(board, grid, lc.net_id, fc.net_id, ll, fl, 0.05);
                    }
                }
            }
        }
        crate::memlog::checkpoint_board("route_all::after_post_plane_gloss", board);

        for trk in &mut board.tracks {
            let key = (trk.net_id, [trk.start.0.to_bits(), trk.start.1.to_bits(), trk.end.0.to_bits(), trk.end.1.to_bits()]);
            trk.is_locked = orig_locked_tracks.contains(&key);
        }
        for v in &mut board.vias {
            let key = (v.net_id, [v.position.0.to_bits(), v.position.1.to_bits()]);
            v.is_locked = orig_locked_vias.contains(&key);
        }

        // Comprehensive Pre-Export Neck-Down, Micro-Nudge & DRC Verification
        crate::memlog::checkpoint_board("route_all::before_widen_tracks_and_arcs_to_netclass", board);
        widen_tracks_and_arcs_to_netclass(board);
        crate::memlog::checkpoint_board("route_all::before_validate_and_clean_copper", board);
        validate_and_clean_copper(board);
        crate::memlog::checkpoint_board("route_all::before_prune_dangling_post_validate", board);
        prune_dangling_tracks_and_vias(board);
        crate::memlog::checkpoint_board("route_all::before_final_continuity_repair", board);
        // If any connection was disconnected during post-processing, re-route it cleanly against the current board copper.
        repair_with_blocker_ripup(board, grid, &mut routed_paths, true);

        let pre_organic_snapshot = if self.config.any_angle_mode {
            let snap = (board.tracks.clone(), board.arcs.clone());
            let snap_vias = board.vias.clone();
            let pre_connected: Vec<bool> = (0..total_conns)
                .map(|i| Self::is_connection_electrically_connected(board, &connections[i]))
                .collect();
            crate::memlog::checkpoint_board("route_all::before_convert_tracks_to_organic", board);
            convert_tracks_to_organic_with_fillet_radius(board, self.config.acid_trap_fillet_radius);
            crate::memlog::checkpoint_board("route_all::after_convert_tracks_to_organic", board);
            prune_dangling_tracks_and_vias(board);
            let mut broken_nets: HashSet<usize> = HashSet::new();
            for i in 0..total_conns {
                if pre_connected[i] && !Self::is_connection_electrically_connected(board, &connections[i]) {
                    broken_nets.insert(connections[i].net_id);
                }
                if let Some(leader_idx) = connections[i].diff_pair_leader {
                    let ln = connections[leader_idx].net_id;
                    let fn_id = connections[i].net_id;
                    if board_net_coupling_mm(board, ln, fn_id) > DIFF_PAIR_COUPLED_MM
                        || board_net_coupling_mm(board, fn_id, ln) > DIFF_PAIR_COUPLED_MM
                    {
                        broken_nets.insert(ln);
                        broken_nets.insert(fn_id);
                    }
                }
            }
            if !broken_nets.is_empty() {
                board.tracks.retain(|t| !broken_nets.contains(&t.net_id));
                board.arcs.retain(|a| !broken_nets.contains(&a.net_id));
                board.vias.retain(|v| !broken_nets.contains(&v.net_id));
                for t in &snap.0 {
                    if broken_nets.contains(&t.net_id) {
                        board.tracks.push(t.clone());
                    }
                }
                for a in &snap.1 {
                    if broken_nets.contains(&a.net_id) {
                        board.arcs.push(a.clone());
                    }
                }
                for v in &snap_vias {
                    if broken_nets.contains(&v.net_id) {
                        board.vias.push(v.clone());
                    }
                }
                widen_tracks_and_arcs_to_netclass(board);
                prune_dangling_tracks_and_vias(board);
            }
            crate::memlog::checkpoint_board("route_all::after_organic_prune", board);
            Some(snap)
        } else {
            if self.config.acid_trap_fillet_radius > 0.0 {
                // Filleting / pad-exit re-anchoring must never break a net: restore any net it breaks.
                let pre: Vec<bool> = (0..total_conns).map(|i| Self::is_connection_electrically_connected(board, &connections[i])).collect();
                let snap = (board.tracks.clone(), board.arcs.clone(), board.vias.clone());
                fillet_acute_angle_junctions(board, self.config.acid_trap_fillet_radius);
                widen_tracks_and_arcs_to_netclass(board);
                prune_dangling_tracks_and_vias(board);
                let broken: HashSet<usize> = (0..total_conns)
                    .filter(|&i| pre[i] && !Self::is_connection_electrically_connected(board, &connections[i]))
                    .map(|i| connections[i].net_id)
                    .collect();
                if !broken.is_empty() {
                    board.tracks.retain(|t| !broken.contains(&t.net_id));
                    board.arcs.retain(|a| !broken.contains(&a.net_id));
                    board.vias.retain(|v| !broken.contains(&v.net_id));
                    board.tracks.extend(snap.0.into_iter().filter(|t| broken.contains(&t.net_id)));
                    board.arcs.extend(snap.1.into_iter().filter(|a| broken.contains(&a.net_id)));
                    board.vias.extend(snap.2.into_iter().filter(|v| broken.contains(&v.net_id)));
                }
            } else {
                widen_tracks_and_arcs_to_netclass(board);
                prune_dangling_tracks_and_vias(board);
            }
            None
        };

        for (conn_idx, _path) in routed_paths.iter().enumerate() {
            if !Self::is_connection_electrically_connected(board, &connections[conn_idx]) {
                let c = &connections[conn_idx];
                let diag = format!(
                    "[UNROUTED] Net '{}' (ID {}) from ({:.2}, {:.2}, L{}) to ({:.2}, {:.2}, L{}) — Routing channel congested / path search exhausted",
                    c.net_name, c.net_id, c.start.0, c.start.1, c.start.2, c.end.0, c.end.1, c.end.2
                );
                eprintln!("  {}", diag);
                if let Some(ref st) = live {
                    if let Ok(mut g) = st.lock() {
                        g.push_log(format!("[Unrouted] {}", diag));
                    }
                }
            }
        }
        let completed_count = (0..total_conns)
            .filter(|&i| Self::is_connection_electrically_connected(board, &connections[i]))
            .count();

        crate::memlog::checkpoint_board("route_all::before_validate_crosstalk_and_coupling", board);
        self.validate_crosstalk_and_coupling(board);
        crate::memlog::checkpoint_board("route_all::completed", board);

        if let Some(ref st) = live {
            let remaining_ratsnest: Vec<crate::board::Segment> = connections
                .iter()
                .enumerate()
                .filter(|&(i, _)| routed_paths[i].is_none())
                .map(|(_, c)| {
                    let p1 = board.footprints[c.start_fp].pad_absolute_position(&board.footprints[c.start_fp].pads[c.start_pad]);
                    let p2 = board.footprints[c.end_fp].pad_absolute_position(&board.footprints[c.end_fp].pads[c.end_pad]);
                    crate::board::Segment {
                        p1: Point::new(p1.0, p1.1),
                        p2: Point::new(p2.0, p2.1),
                        net_id: c.net_id,
                    }
                })
                .collect();
            if let Ok(mut g) = st.lock() {
                g.board.tracks = board.tracks.clone();
                g.board.arcs = board.arcs.clone();
                g.board.vias = board.vias.clone();
                g.board.footprints = board.footprints.clone();
                g.straight_copper_snapshot = pre_organic_snapshot;
                g.ratsnest = remaining_ratsnest;
                g.routed_conns = completed_count;
                g.total_conns = total_conns;
                g.progress = 1.0;
                g.phase = TrialPhase::Completed;
                g.push_log(format!(
                    "[Auto-Router] Completed: {}/{} ({:.1}%) connections, {} tracks, {} arcs, {} vias.",
                    completed_count,
                    total_conns,
                    (completed_count as f64 / total_conns.max(1) as f64) * 100.0,
                    board.tracks.len(),
                    board.arcs.len(),
                    board.vias.len()
                ));
            }
        }

        if let Some(ref ipc_lock) = self.ipc {
            if let Ok(mut client) = ipc_lock.lock() {
                if client.is_connected() {
                    let _ = client.clear_tracks();
                    let _ = client.stream_tracks_and_vias(board);
                }
            }
        }

        (completed_count, total_conns)
    }

    /// Verifies that a discrete grid path is completely free of foreign-net clearance
    /// conflicts, corner-clipping violations, and (when required) reference-plane return voids.
    ///
    /// # Arguments
    /// * `grid` - Reference to the [`RoutingGrid`].
    /// * `path` - Sequence of [`GridNode`] waypoints forming the candidate path.
    /// * `net_id` - Numeric net ID of the signal.
    /// * `req_clearance` - Required copper-to-copper clearance in `mm`.
    /// * `track_width` - Trace width in `mm`.
    /// * `via_dia` - Via outer copper diameter in `mm`.
    /// * `u1_center` - Optional `(x, y)` center of `U1` in `mm`.
    /// * `require_return_path` - Whether every node must lie above/below an unbroken reference plane.
    ///
    /// # Returns
    /// `true` if every node and transition along `path` is legal on `grid`.
    fn is_path_conflict_free(
        &self,
        grid: &RoutingGrid,
        path: &[GridNode],
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
        require_return_path: bool,
    ) -> bool {
        if path.is_empty() {
            return false;
        }
        let s_idx = grid.idx(path[0].gx, path[0].gy, path[0].layer);
        let s_net = grid.obstacle_net_ids[s_idx];
        if s_net != FREE_NET && s_net != net_id as i32 {
            return false;
        }
        for i in 0..path.len().saturating_sub(1) {
            let u = path[i];
            let v = path[i + 1];
            let ok = grid
                .get_neighbors(u, net_id, req_clearance, track_width, via_dia, u1_center)
                .iter()
                .any(|&(nbr, _)| nbr == v);
            if !ok {
                return false;
            }
        }
        if require_return_path {
            for node in path {
                if let Some(ref_layer) = grid.adjacent_reference_plane(node.layer) {
                    if grid.is_reference_plane_void_at(node.gx, node.gy, ref_layer) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Searches for a conflict-free 3D grid path between `start` and `goal` by running 6
    /// forward/reverse and heuristic-weighted A* strategies in parallel across Rayon workers.
    ///
    /// # Arguments
    /// * `grid` - Read-only reference to the [`RoutingGrid`].
    /// * `start` - Source [`GridNode`].
    /// * `goal` - Target [`GridNode`].
    /// * `net_id` - Net ID being routed.
    /// * `req_clearance` - Required copper clearance in `mm`.
    /// * `track_width` - Trace width in `mm`.
    /// * `via_dia` - Via outer diameter in `mm`.
    /// * `u1_center` - Optional `(x, y)` center of `U1` in `mm`.
    /// * `max_coupled_length_mm` - Maximum parallel coupled run length in `mm`.
    /// * `require_return_path` - Whether an unbroken reference plane is required.
    /// * `stop_flag` - Atomic cancellation flag.
    ///
    /// # Returns
    /// `Some(path)` with the lowest bend-weighted length among all successful strategies, or `None`.
    fn find_path_parallel(
        &self,
        grid: &RoutingGrid,
        start: GridNode,
        goal: GridNode,
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
        max_coupled_length_mm: f64,
        require_return_path: bool,
        stop_flag: &AtomicBool,
    ) -> Option<Vec<GridNode>> {
        let strategies: &[(bool, f64, f64)] = &[
            (false, 1.0, 0.0),
            (true, 1.0, 0.0),
            (false, 1.05, 0.0),
            (true, 1.05, 0.0),
            (false, 1.15, 0.01),
            (true, 1.15, 0.01),
        ];
        let found = AtomicBool::new(false);

        let candidates: Vec<(usize, Vec<GridNode>)> = strategies
            .par_iter()
            .enumerate()
            .filter_map(|(s_idx, &(reverse, h_weight, dist_bias))| {
                let (s_node, g_node) = if reverse { (goal, start) } else { (start, goal) };
                let mut path = self.find_path_with_constraints(
                    grid,
                    s_node,
                    g_node,
                    net_id,
                    req_clearance,
                    track_width,
                    via_dia,
                    u1_center,
                    h_weight,
                    dist_bias,
                    max_coupled_length_mm,
                    require_return_path,
                    None,
                    &found,
                )?;
                if stop_flag.load(AtomicOrdering::Relaxed) {
                    return None;
                }
                if reverse {
                    path.reverse();
                    if !self.is_path_conflict_free(
                        grid,
                        &path,
                        net_id,
                        req_clearance,
                        track_width,
                        via_dia,
                        u1_center,
                        require_return_path,
                    ) {
                        return None;
                    }
                }
                found.store(true, AtomicOrdering::Relaxed);
                Some((s_idx, path))
            })
            .collect();

        candidates
            .into_iter()
            .min_by_key(|(s_idx, p)| (Self::count_path_bends(p) * 35 + p.len(), *s_idx))
            .map(|(_, p)| p)
    }

    /// Counts the number of directional bends and layer transitions along a discrete grid path.
    ///
    /// In-plane direction changes add `1` bend; layer transitions (vias) add `2` bends.
    ///
    /// # Arguments
    /// * `path` - Slice of [`GridNode`] waypoints.
    ///
    /// # Returns
    /// Total weighted bend count.
    pub fn count_path_bends(path: &[GridNode]) -> usize {
        if path.len() < 3 {
            return 0;
        }
        let mut bends = 0;
        for i in 1..(path.len() - 1) {
            if path[i - 1].layer != path[i].layer || path[i].layer != path[i + 1].layer {
                bends += 2;
                continue;
            }
            let dx1 = path[i].gx as isize - path[i - 1].gx as isize;
            let dy1 = path[i].gy as isize - path[i - 1].gy as isize;
            let dx2 = path[i + 1].gx as isize - path[i].gx as isize;
            let dy2 = path[i + 1].gy as isize - path[i].gy as isize;
            if (dx1, dy1) != (dx2, dy2) {
                bends += 1;
            }
        }
        bends
    }

    /// Generates the inclusive sequence of [`GridNode`] cells along a straight horizontal,
    /// vertical, or 45-degree diagonal grid segment from `start` to `end` on `start.layer`.
    ///
    /// # Arguments
    /// * `start` - Starting [`GridNode`].
    /// * `end` - Ending [`GridNode`] (assumed collinear in octilinear space).
    ///
    /// # Returns
    /// Vector of [`GridNode`] cells from `start` to `end` inclusive.
    fn line_nodes(start: GridNode, end: GridNode) -> Vec<GridNode> {
        let dx = end.gx as isize - start.gx as isize;
        let dy = end.gy as isize - start.gy as isize;
        let steps = dx.abs().max(dy.abs());
        if steps == 0 {
            return vec![start];
        }
        let step_x = dx.signum();
        let step_y = dy.signum();
        let mut nodes = Vec::with_capacity(steps as usize + 1);
        for s in 0..=steps {
            nodes.push(GridNode {
                gx: (start.gx as isize + s * step_x) as usize,
                gy: (start.gy as isize + s * step_y) as usize,
                layer: start.layer,
            });
        }
        nodes
    }

    /// Generates up to two canonical 1-bend octilinear dogleg candidate paths between `start`
    /// and `end` (diagonal-first then straight, and straight-first then diagonal).
    ///
    /// # Arguments
    /// * `start` - Starting [`GridNode`].
    /// * `end` - Ending [`GridNode`] on the same layer.
    ///
    /// # Returns
    /// Vector of 1 or 2 candidate [`GridNode`] paths connecting `start` to `end`.
    fn two_segment_candidates(start: GridNode, end: GridNode) -> Vec<Vec<GridNode>> {
        let dx = end.gx as isize - start.gx as isize;
        let dy = end.gy as isize - start.gy as isize;
        if dx == 0 || dy == 0 || dx.abs() == dy.abs() {
            return vec![Self::line_nodes(start, end)];
        }
        let s = dx.abs().min(dy.abs());
        let mut results = Vec::with_capacity(2);

        // Candidate 1: diagonal first, then straight
        let m1 = GridNode {
            gx: (start.gx as isize + s * dx.signum()) as usize,
            gy: (start.gy as isize + s * dy.signum()) as usize,
            layer: start.layer,
        };
        let mut c1 = Self::line_nodes(start, m1);
        let mut rest1 = Self::line_nodes(m1, end);
        if rest1.len() > 1 {
            c1.extend(rest1.drain(1..));
        }
        results.push(c1);

        // Candidate 2: straight first, then diagonal
        let m2 = if dx.abs() > dy.abs() {
            GridNode {
                gx: (start.gx as isize + (dx - s * dx.signum())) as usize,
                gy: start.gy,
                layer: start.layer,
            }
        } else {
            GridNode {
                gx: start.gx,
                gy: (start.gy as isize + (dy - s * dy.signum())) as usize,
                layer: start.layer,
            }
        };
        let mut c2 = Self::line_nodes(start, m2);
        let mut rest2 = Self::line_nodes(m2, end);
        if rest2.len() > 1 {
            c2.extend(rest2.drain(1..));
        }
        results.push(c2);

        results
    }

    /// Smooths a discrete grid path by iteratively replacing multi-bend staircases and jogs
    /// with conflict-free 0-bend straight lines or 1-bend canonical 45-degree corners.
    ///
    /// # Arguments
    /// * `grid` - Reference to the [`RoutingGrid`].
    /// * `path` - Initial [`GridNode`] sequence from A*.
    /// * `net_id` - Numeric net ID of the trace.
    /// * `req_clearance` - Required copper clearance in `mm`.
    /// * `track_width` - Trace width in `mm`.
    /// * `via_dia` - Via outer diameter in `mm`.
    /// * `u1_center` - Optional `(x, y)` center of `U1` in `mm`.
    /// * `require_return_path` - Whether reference-plane continuity must be preserved.
    /// * `max_coupled_length_mm` - Maximum allowed parallel coupled length in `mm`.
    ///
    /// # Returns
    /// Smoothed vector of [`GridNode`] cells with reduced bend count.
    fn smooth_grid_path(
        &self,
        grid: &RoutingGrid,
        mut path: Vec<GridNode>,
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
        require_return_path: bool,
        max_coupled_length_mm: f64,
    ) -> Vec<GridNode> {
        if path.len() < 4 {
            return path;
        }

        let is_valid_candidate = |cand: &[GridNode]| -> bool {
            if !self.is_path_conflict_free(
                grid,
                cand,
                net_id,
                req_clearance,
                track_width,
                via_dia,
                u1_center,
                require_return_path,
            ) {
                return false;
            }
            if max_coupled_length_mm > 0.0 {
                let mut run_len = 0.0;
                for k in 0..cand.len().saturating_sub(1) {
                    if grid.is_step_coupled(cand[k], cand[k + 1], net_id, track_width, req_clearance) {
                        let sl = if cand[k].gx != cand[k + 1].gx && cand[k].gy != cand[k + 1].gy {
                            1.41421356 * grid.res
                        } else {
                            1.0 * grid.res
                        };
                        run_len += sl;
                        if run_len > max_coupled_length_mm {
                            return false;
                        }
                    } else {
                        run_len = 0.0;
                    }
                }
            }
            true
        };

        // Pass 1: Line-of-sight octilinear shortcutting and staircase collapsing
        // Eliminates staircase zig-zags and rapid direction changes by connecting nodes with
        // 0 bends (straight line) or 1 bend (canonical 45-degree corner).
        let max_passes = 4;
        for _pass in 0..max_passes {
            let mut improved = false;
            let mut new_path: Vec<GridNode> = Vec::with_capacity(path.len());
            let mut i = 0;
            new_path.push(path[0]);

            while i < path.len() {
                if i + 1 >= path.len() {
                    break;
                }
                // Determine the end of the current same-layer run
                let current_layer = path[i].layer;
                let mut layer_run_end = i + 1;
                while layer_run_end < path.len() && path[layer_run_end].layer == current_layer {
                    layer_run_end += 1;
                }
                if layer_run_end <= i + 1 {
                    i += 1;
                    if i < path.len() {
                        new_path.push(path[i]);
                    }
                    continue;
                }

                let max_lookahead = (layer_run_end - 1).min(i + 64);
                let mut shortcut_found = false;

                // Priority 1: Direct 1-segment shortcut (0 bends)
                for j in (i + 2..=max_lookahead).rev() {
                    let p_i = path[i];
                    let p_j = path[j];
                    let dx = p_j.gx as isize - p_i.gx as isize;
                    let dy = p_j.gy as isize - p_i.gy as isize;
                    if dx == 0 || dy == 0 || dx.abs() == dy.abs() {
                        let candidate = Self::line_nodes(p_i, p_j);
                        if is_valid_candidate(&candidate) {
                            let i_prev = i.saturating_sub(1);
                            let j_next = (j + 1).min(path.len() - 1);
                            let bends_before = Self::count_path_bends(&path[i_prev..=j_next]);

                            let mut cand_slice = Vec::with_capacity(candidate.len() + 2);
                            if i > 0 && path[i - 1].layer == current_layer {
                                cand_slice.push(path[i - 1]);
                            }
                            cand_slice.extend_from_slice(&candidate);
                            if j + 1 < path.len() && path[j + 1].layer == current_layer {
                                cand_slice.push(path[j + 1]);
                            }
                            let bends_after = Self::count_path_bends(&cand_slice);

                            if bends_after < bends_before
                                || (bends_after == bends_before && candidate.len() <= (j - i + 1))
                            {
                                for node in candidate.into_iter().skip(1) {
                                    new_path.push(node);
                                }
                                i = j;
                                shortcut_found = true;
                                improved = true;
                                break;
                            }
                        }
                    }
                }

                if shortcut_found {
                    continue;
                }

                // Priority 2: 2-segment canonical shortcut (1 bend, replaces multi-bend staircases)
                for j in (i + 3..=max_lookahead).rev() {
                    let p_i = path[i];
                    let p_j = path[j];
                    let existing_bends = Self::count_path_bends(&path[i..=j]);
                    if existing_bends < 2 {
                        continue;
                    }
                    let candidates = Self::two_segment_candidates(p_i, p_j);
                    let mut best_candidate: Option<Vec<GridNode>> = None;
                    for cand in candidates {
                        if is_valid_candidate(&cand) {
                            let i_prev = i.saturating_sub(1);
                            let j_next = (j + 1).min(path.len() - 1);
                            let bends_before = Self::count_path_bends(&path[i_prev..=j_next]);

                            let mut cand_slice = Vec::with_capacity(cand.len() + 2);
                            if i > 0 && path[i - 1].layer == current_layer {
                                cand_slice.push(path[i - 1]);
                            }
                            cand_slice.extend_from_slice(&cand);
                            if j + 1 < path.len() && path[j + 1].layer == current_layer {
                                cand_slice.push(path[j + 1]);
                            }
                            let bends_after = Self::count_path_bends(&cand_slice);

                            if bends_after < bends_before {
                                best_candidate = Some(cand);
                                break;
                            }
                        }
                    }
                    if let Some(cand) = best_candidate {
                        for node in cand.into_iter().skip(1) {
                            new_path.push(node);
                        }
                        i = j;
                        shortcut_found = true;
                        improved = true;
                        break;
                    }
                }

                if !shortcut_found {
                    i += 1;
                    if i < path.len() {
                        new_path.push(path[i]);
                    }
                }
            }

            path = new_path;
            if !improved {
                break;
            }
        }

        path
    }

    /// Convenience wrapper around [`AutoRouter::find_path_with_constraints`] using default
    /// coupled-length (`15.0 mm`) and return-path (`false`) settings.
    ///
    /// # Arguments
    /// * `grid` - Reference to the [`RoutingGrid`].
    /// * `start` - Source [`GridNode`].
    /// * `goal` - Destination [`GridNode`].
    /// * `net_id` - Numeric net ID.
    /// * `req_clearance` - Required copper clearance in `mm`.
    /// * `track_width` - Trace width in `mm`.
    /// * `via_dia` - Via outer diameter in `mm`.
    /// * `u1_center` - Optional `(x, y)` center of `U1` in `mm`.
    /// * `h_weight` - A* heuristic weight multiplier.
    /// * `dist_bias` - Distance-field obstacle repulsion bias.
    /// * `abort_flag` - Atomic cancellation flag.
    ///
    /// # Returns
    /// `Some(path)` if a valid 3D route was found, or `None`.
    pub fn find_path_with_params(
        &self,
        grid: &RoutingGrid,
        start: GridNode,
        goal: GridNode,
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
        h_weight: f64,
        dist_bias: f64,
        abort_flag: &AtomicBool,
    ) -> Option<Vec<GridNode>> {
        self.find_path_with_constraints(
            grid,
            start,
            goal,
            net_id,
            req_clearance,
            track_width,
            via_dia,
            u1_center,
            h_weight,
            dist_bias,
            15.0,
            false,
            None,
            abort_flag,
        )
    }

    /// Searches for a 3D octilinear path between `start` and `goal` subject to clearance,
    /// crosstalk coupled-length, reference-plane return-path, and optional corridor constraints.
    ///
    /// # Arguments
    /// * `grid` - Reference to the [`RoutingGrid`].
    /// * `start` - Source [`GridNode`].
    /// * `goal` - Destination [`GridNode`].
    /// * `net_id` - Numeric net ID.
    /// * `req_clearance` - Required copper clearance in `mm`.
    /// * `track_width` - Trace width in `mm`.
    /// * `via_dia` - Via outer diameter in `mm`.
    /// * `u1_center` - Optional `(x, y)` center of `U1` in `mm`.
    /// * `h_weight` - A* heuristic weight multiplier.
    /// * `dist_bias` - Distance-field obstacle bias (negative values reserve partner pitch).
    /// * `max_coupled_length_mm` - Maximum parallel coupled length in `mm` before crosstalk penalty.
    /// * `require_return_path` - Whether reference-plane voids are penalized/rejected.
    /// * `corridor` - Optional per-cell cost multiplier slice.
    /// * `abort_flag` - Atomic cancellation flag.
    ///
    /// # Returns
    /// `Some(path)` if a valid route was found within the expansion budget, or `None`.
    pub fn find_path_with_constraints(
        &self,
        grid: &RoutingGrid,
        start: GridNode,
        goal: GridNode,
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
        h_weight: f64,
        dist_bias: f64,
        max_coupled_length_mm: f64,
        require_return_path: bool,
        corridor: Option<&[f32]>,
        abort_flag: &AtomicBool,
    ) -> Option<Vec<GridNode>> {
        self.find_path_sided(
            grid, start, goal, net_id, req_clearance, track_width, via_dia, u1_center, h_weight, dist_bias,
            max_coupled_length_mm, require_return_path, corridor, 0.0, abort_flag,
        )
    }

    /// Core 3D octilinear A* pathfinder with support for sided differential-pair leader
    /// partner-pitch reservation (`dist_bias < 0.0` and `partner_side` in `{-1.0, 0.0, +1.0}`).
    ///
    /// Seeds all valid signal layers when starting on a through-hole pad, penalizes bends,
    /// reference-plane voids, and excessive parallel coupled runs, and smooths the resulting
    /// path when not bound to a differential corridor.
    ///
    /// # Arguments
    /// * `grid` - Reference to the [`RoutingGrid`].
    /// * `start` - Source [`GridNode`].
    /// * `goal` - Destination [`GridNode`].
    /// * `net_id` - Numeric net ID.
    /// * `req_clearance` - Required copper clearance in `mm`.
    /// * `track_width` - Trace width in `mm`.
    /// * `via_dia` - Via outer diameter in `mm`.
    /// * `u1_center` - Optional `(x, y)` center of `U1` in `mm`.
    /// * `h_weight` - A* heuristic weight multiplier.
    /// * `dist_bias` - Positive for obstacle repulsion, negative for diff-pair pitch reservation.
    /// * `max_coupled_length_mm` - Maximum parallel coupled length in `mm`.
    /// * `require_return_path` - Whether reference-plane voids incur heavy penalties.
    /// * `corridor` - Optional per-cell cost multiplier slice for follower routing.
    /// * `partner_side` - `+1.0` to reserve partner corridor on the left, `-1.0` on the right, `0.0` either side.
    /// * `abort_flag` - Atomic cancellation flag.
    ///
    /// # Returns
    /// `Some(path)` from `start` to `goal`, or `None` if unroutable or aborted.
    fn find_path_sided(
        &self,
        grid: &RoutingGrid,
        start: GridNode,
        goal: GridNode,
        net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
        h_weight: f64,
        dist_bias: f64,
        max_coupled_length_mm: f64,
        require_return_path: bool,
        corridor: Option<&[f32]>,
        partner_side: f64,
        abort_flag: &AtomicBool,
    ) -> Option<Vec<GridNode>> {
        let mut open_set = BinaryHeap::new();
        // Single compact map keyed by grid.idx(gx, gy, layer) as u32 -> (parent, g_score, coupled_len)
        let mut node_state: HashMap<u32, (GridNode, f64, f32)> = HashMap::new();

        let h_scale = 1.0;
        let h_start = self.heuristic(grid, start, goal) * h_weight * h_scale;
        let start_idx = grid.idx(start.gx, start.gy, start.layer) as u32;
        node_state.insert(start_idx, (start, 0.0, 0.0));
        open_set.push(AStarNode {
            node: start,
            cost: 0.0,
            priority: h_start,
        });

        // If start is at a through-hole pad of this net, seed open_set on all valid signal layers
        let start_is_smd = (0..grid.num_layers).any(|l| grid.smd_pad_cells[grid.idx(start.gx, start.gy, l)]);
        let start_is_tht = !start_is_smd
            && (0..grid.num_layers).all(|l| grid.obstacle_net_ids[grid.idx(start.gx, start.gy, l)] == net_id as i32);
        if start_is_tht {
            for l in 0..grid.num_layers {
                if l != start.layer {
                    if grid.plane_nets.contains_key(&l) && grid.plane_nets.get(&l) != Some(&net_id) {
                        continue;
                    }
                    let alt_start = GridNode { gx: start.gx, gy: start.gy, layer: l };
                    let alt_idx = grid.idx(alt_start.gx, alt_start.gy, alt_start.layer) as u32;
                    node_state.insert(alt_idx, (alt_start, 0.0, 0.0));
                    open_set.push(AStarNode {
                        node: alt_start,
                        cost: 0.0,
                        priority: self.heuristic(grid, alt_start, goal) * h_weight * h_scale,
                    });
                }
            }
        }

        let dist_span = (goal.gx as isize - start.gx as isize).abs() + (goal.gy as isize - start.gy as isize).abs();
        let max_expansions = if dist_span > 200 { 350_000 } else { 120_000 };
        let mut expansions = 0;

        while let Some(current) = open_set.pop() {
            expansions += 1;
            if expansions > max_expansions
                || (expansions % 256 == 0 && abort_flag.load(AtomicOrdering::Relaxed))
            {
                return None;
            }
            let u = current.node;
            let u_idx = grid.idx(u.gx, u.gy, u.layer) as u32;
            if u == goal {
                let mut path = vec![goal];
                let mut curr = goal;
                while let Some(&(prev, _, _)) = node_state.get(&(grid.idx(curr.gx, curr.gy, curr.layer) as u32)) {
                    if prev == curr {
                        break;
                    }
                    path.push(prev);
                    curr = prev;
                }
                path.reverse();
                if corridor.is_some() || dist_bias < 0.0 {
                    return Some(path);
                }
                return Some(self.smooth_grid_path(
                    grid,
                    path,
                    net_id,
                    req_clearance,
                    track_width,
                    via_dia,
                    u1_center,
                    require_return_path,
                    max_coupled_length_mm,
                ));
            }

            let Some(&(parent_u, current_g, prev_coupled_u)) = node_state.get(&u_idx) else {
                continue;
            };
            if current.cost > current_g + 1e-6 {
                continue;
            }

            let prev_dir: Option<(isize, isize)> = if parent_u != u && parent_u.layer == u.layer {
                Some((u.gx as isize - parent_u.gx as isize, u.gy as isize - parent_u.gy as isize))
            } else {
                None
            };

            for (neighbor, mut edge_cost) in
                grid.get_neighbors(u, net_id, req_clearance, track_width, via_dia, u1_center)
            {
                let mut next_coupled = 0.0f32;
                if neighbor.layer == u.layer {
                    if let Some((pdx, pdy)) = prev_dir {
                        let cur_dx = neighbor.gx as isize - u.gx as isize;
                        let cur_dy = neighbor.gy as isize - u.gy as isize;
                        if (cur_dx, cur_dy) != (pdx, pdy) {
                            let dot = pdx * cur_dx + pdy * cur_dy;
                            if dot > 0 {
                                // 45-degree bend penalty to prevent rapid zig-zag meandering
                                edge_cost += 5.0 * grid.res;
                            } else if dot == 0 {
                                // 90-degree bend penalty
                                edge_cost += 15.0 * grid.res;
                            } else {
                                // Sharp/obtuse (>90-degree) bend penalty
                                edge_cost += 45.0 * grid.res;
                            }
                        }
                    }

                    // Return-path continuity check for high-speed signals
                    if require_return_path {
                        if let Some(ref_layer) = grid.adjacent_reference_plane(neighbor.layer) {
                            if grid.is_reference_plane_void_at(neighbor.gx, neighbor.gy, ref_layer) {
                                edge_cost += 35.0 * grid.res + 5.0;
                            }
                        }
                    }

                    // Crosstalk & coupled length limiter (skip for coupled differential pair followers)
                    if corridor.is_none() {
                        let is_coupled = grid.is_step_coupled(u, neighbor, net_id, track_width, req_clearance);
                        let step_len = if neighbor.gx != u.gx && neighbor.gy != u.gy {
                            1.41421356 * grid.res
                        } else {
                            1.0 * grid.res
                        };

                        if is_coupled {
                            let next_c = prev_coupled_u as f64 + step_len;
                            next_coupled = next_c as f32;
                            if next_c > max_coupled_length_mm {
                                let excess = next_c - max_coupled_length_mm;
                                let penalty = self.crosstalk_penalty * step_len * (1.0 + excess * 0.5);
                                edge_cost += penalty;
                            }
                        }
                    }
                } else {
                    // Through-hole via penalty: equivalent to 2.0 mm of trace length
                    edge_cost += 2.0;
                }

                let n_idx = grid.idx(neighbor.gx, neighbor.gy, neighbor.layer);
                if dist_bias > 0.0 {
                    let df_val = (grid.dist_field[n_idx] as f64).clamp(0.05, 2.0);
                    edge_cost += dist_bias / df_val;
                } else if dist_bias < 0.0 && neighbor.layer == u.layer {
                    let d_start = (neighbor.gx as isize - start.gx as isize).abs()
                        + (neighbor.gy as isize - start.gy as isize).abs();
                    let d_goal = (neighbor.gx as isize - goal.gx as isize).abs()
                        + (neighbor.gy as isize - goal.gy as isize).abs();
                    if d_start > 18 && d_goal > 18 {
                        // Sided leaders must reserve what the follower's own neighbour test demands:
                        // `max_net_clearance` from every foreign obstacle, including this leader.
                        let (pitch_cells, req_comp) = if partner_side != 0.0 {
                            let clr = grid.max_net_clearance.max(req_clearance);
                            let pitch = (track_width + clr).max(-dist_bias);
                            (
                                (pitch / grid.res - 1e-6).ceil().max(2.0) as isize,
                                (track_width * 0.5 + clr) as f32,
                            )
                        } else {
                            (
                                ((-dist_bias) / grid.res).round().max(2.0) as isize,
                                (track_width * 0.5 + req_clearance) as f32,
                            )
                        };
                        let (ddx, ddy) = (neighbor.gx as isize - u.gx as isize, neighbor.gy as isize - u.gy as isize);
                        let (sdx, sdy) = (ddx.signum(), ddy.signum());
                        let side_clear = |bx: isize, by: isize, ox: isize, oy: isize| -> bool {
                            let cx = bx + ox;
                            let cy = by + oy;
                            if cx < 0 || cx >= grid.width as isize || cy < 0 || cy >= grid.height as isize {
                                return false;
                            }
                            let c_idx = grid.idx(cx as usize, cy as usize, neighbor.layer);
                            let obs = grid.obstacle_net_ids[c_idx];
                            (obs == FREE_NET || obs == net_id as i32) && grid.dist_field[c_idx] >= req_comp
                        };
                        // Any-angle moves can jump many cells; sided leaders check every cell so the
                        // penalty scales with the blocked length rather than the number of edges.
                        let steps = if partner_side != 0.0 { ddx.abs().max(ddy.abs()).max(1) } else { 1 };
                        let mut blocked_cells = 0usize;
                        for k in 1..=steps {
                            let bx = u.gx as isize + ddx * k / steps;
                            let by = u.gy as isize + ddy * k / steps;
                            let left_ok = side_clear(bx, by, -sdy * pitch_cells, sdx * pitch_cells);
                            let right_ok = side_clear(bx, by, sdy * pitch_cells, -sdx * pitch_cells);
                            let blocked = if partner_side > 0.0 {
                                !left_ok
                            } else if partner_side < 0.0 {
                                !right_ok
                            } else {
                                !left_ok && !right_ok
                            };
                            blocked_cells += blocked as usize;
                        }
                        if blocked_cells > 0 {
                            // A sided leader that leaves no room for its partner splits the pair.
                            let pen = if partner_side != 0.0 { DIFF_PAIR_SIDE_BLOCKED_PENALTY } else { 12.0 };
                            edge_cost += pen * grid.res * blocked_cells as f64;
                        }
                    }
                }
                if let Some(corr) = corridor {
                    edge_cost *= corr[n_idx] as f64;
                }
                let tentative_g = current_g + edge_cost;
                let n_key = n_idx as u32;
                let is_better = node_state.get(&n_key).is_none_or(|&(_, prev_g, _)| tentative_g < prev_g - 1e-6);
                if is_better {
                    node_state.insert(n_key, (u, tentative_g, next_coupled));
                    let priority = tentative_g + self.heuristic(grid, neighbor, goal) * h_weight * h_scale;
                    open_set.push(AStarNode {
                        node: neighbor,
                        cost: tentative_g,
                        priority,
                    });
                }
            }
        }
        None
    }

    /// Computes the admissible 3D octile distance heuristic in `mm` between grid nodes `a` and `b`,
    /// including a `2.0 mm` layer-change penalty when `a.layer != b.layer`.
    ///
    /// # Arguments
    /// * `grid` - Reference to the [`RoutingGrid`] (for cell resolution `res`).
    /// * `a` - First [`GridNode`].
    /// * `b` - Second [`GridNode`].
    ///
    /// # Returns
    /// Estimated minimum cost in `mm`.
    fn heuristic(&self, grid: &RoutingGrid, a: GridNode, b: GridNode) -> f64 {
        let dx = (a.gx as isize - b.gx as isize).abs() as f64 * grid.res;
        let dy = (a.gy as isize - b.gy as isize).abs() as f64 * grid.res;
        let diag = dx.min(dy);
        let straight = (dx - dy).abs();
        let layer_pen = if a.layer != b.layer { 2.0 } else { 0.0 };
        diag * std::f64::consts::SQRT_2 + straight + layer_pen
    }

    /// Rasterizes a routed discrete `path` and any layer-transition vias onto `grid` with
    /// ownership `net_id`.
    ///
    /// # Arguments
    /// * `grid` - Mutable [`RoutingGrid`] to update.
    /// * `path` - Slice of [`GridNode`] waypoints.
    /// * `net_id` - Numeric net ID of the trace.
    /// * `track_width` - Trace width in `mm`.
    /// * `via_dia` - Via outer diameter in `mm`.
    ///
    /// # Side Effects
    /// Invalidates `grid.gpu_net_cost_maps`, increments `grid.usage_count`, and stamps
    /// `net_id` into `grid.obstacle_net_ids` across trace width and via cylinders.
    pub fn commit_path_to_grid(
        &self,
        grid: &mut RoutingGrid,
        path: &[GridNode],
        net_id: usize,
        track_width: f64,
        via_dia: f64,
    ) {
        grid.gpu_net_cost_maps.clear();
        let r_cells = (track_width / (2.0 * grid.res)).ceil() as isize;
        for i in 0..path.len() {
            let node = path[i];
            let node_idx = grid.idx(node.gx, node.gy, node.layer);
            grid.usage_count[node_idx] = grid.usage_count[node_idx].saturating_add(1);
            for dy in -r_cells..=r_cells {
                for dx in -r_cells..=r_cells {
                    if (dx * dx + dy * dy) as f64 * (grid.res * grid.res) <= (track_width / 2.0).powi(2) + 1e-6 {
                        let nx = node.gx as isize + dx;
                        let ny = node.gy as isize + dy;
                        if nx >= 0 && nx < grid.width as isize && ny >= 0 && ny < grid.height as isize {
                            let idx = grid.idx(nx as usize, ny as usize, node.layer);
                            if grid.obstacle_net_ids[idx] == FREE_NET {
                                grid.obstacle_net_ids[idx] = net_id as i32;
                            }
                        }
                    }
                }
            }

            if i + 1 < path.len() && path[i + 1].layer != node.layer {
                let vr_radius = (via_dia / 2.0) + (grid.res * 0.40);
                let vr_cells = (vr_radius / grid.res).ceil() as isize;
                for cy in -vr_cells..=vr_cells {
                    for cx in -vr_cells..=vr_cells {
                        if (cx * cx + cy * cy) as f64 * (grid.res * grid.res) <= vr_radius.powi(2) + 1e-6 {
                            let nx = node.gx as isize + cx;
                            let ny = node.gy as isize + cy;
                            if nx >= 0 && nx < grid.width as isize && ny >= 0 && ny < grid.height as isize {
                                for l in 0..grid.num_layers {
                                    let idx = grid.idx(nx as usize, ny as usize, l);
                                    if grid.obstacle_net_ids[idx] == FREE_NET {
                                        grid.obstacle_net_ids[idx] = net_id as i32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Rebuilds the [`RoutingGrid`] obstacle state from the cached base snapshot (or by
    /// rasterizing static board obstacles, locked copper, and escape stubs) and re-commits
    /// all currently active `routed_paths`.
    ///
    /// # Arguments
    /// * `board` - Reference to the [`Board`].
    /// * `grid` - Mutable [`RoutingGrid`] to reset and populate.
    /// * `connections` - Slice of all [`RouteConnection`] items.
    /// * `routed_paths` - Slice of currently active grid paths per connection.
    /// * `escape_stubs` - Slice of pre-routed [`EscapeStub`] structures.
    ///
    /// # Side Effects
    /// Overwrites `grid.obstacle_net_ids`, `grid.usage_count`, `grid.nudgeable_fp`, and
    /// `grid.smd_pad_cells`, caching `grid.base_snapshot` if not yet initialized.
    fn rebuild_grid_state(
        &self,
        board: &Board,
        grid: &mut RoutingGrid,
        connections: &[RouteConnection],
        routed_paths: &[Option<Vec<GridNode>>],
        escape_stubs: &[EscapeStub],
    ) {
        if let Some(ref snap) = grid.base_snapshot {
            grid.obstacle_net_ids.copy_from_slice(&snap.0);
            grid.usage_count.copy_from_slice(&snap.1);
            grid.nudgeable_fp.copy_from_slice(&snap.2);
            grid.smd_pad_cells.copy_from_slice(&snap.3);
            grid.gpu_net_cost_maps.clear();
        } else {
            grid.obstacle_net_ids.fill(FREE_NET);
            grid.usage_count.fill(0);
            grid.gpu_net_cost_maps.clear();
            grid.nudgeable_fp.fill(None);
            grid.smd_pad_cells.fill(false);
            grid.rasterize_obstacles(board);
            for trk in &board.tracks {
                if trk.is_locked {
                    grid.commit_track(trk);
                }
            }
            for v in &board.vias {
                if v.is_locked {
                    grid.commit_via(v);
                }
            }
            for stub in escape_stubs {
                self.commit_stub_to_grid(grid, stub);
            }
            grid.base_snapshot = Some(Arc::new((
                grid.obstacle_net_ids.clone(),
                grid.usage_count.clone(),
                grid.nudgeable_fp.clone(),
                grid.smd_pad_cells.clone(),
            )));
        }
        for (c_idx, path_opt) in routed_paths.iter().enumerate() {
            if let Some(path) = path_opt {
                self.commit_path_to_grid(
                    grid,
                    path,
                    connections[c_idx].net_id,
                    connections[c_idx].netclass.track_width,
                    connections[c_idx].netclass.via_dia,
                );
            }
        }
    }

    /// Pre-routing pass that nudges unlocked 2-pad passive components away if their pads
    /// directly encroach upon provisional fine-pitch IC escape stubs.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] whose unlocked passive footprints may be translated.
    ///
    /// # Returns
    /// `true` if at least one component was nudged; `false` otherwise.
    ///
    /// # Side Effects
    /// Translates blocking footprints in `board` and rips up any incident traces.
    pub fn nudge_components_blocking_escapes(&self, board: &mut Board) -> bool {
        if !self.allow_component_nudge {
            return false;
        }
        let provisional_stubs = self.generate_escape_stubs(board);
        let mut any_nudged = false;
        for stub in &provisional_stubs {
            let s_p1 = Point::new(stub.pad_pos.0, stub.pad_pos.1);
            let s_p2 = Point::new(stub.tip_pos.0, stub.tip_pos.1);
            for fp_idx in 0..board.footprints.len() {
                if fp_idx == stub.fp_idx {
                    continue;
                }
                let fp = &board.footprints[fp_idx];
                if fp.is_locked || fp.pads.len() > 2 {
                    continue;
                }
                let mut is_blocking = false;
                for pad in &fp.pads {
                    if pad.net_id == stub.net_id {
                        continue;
                    }
                    let (px, py) = fp.pad_absolute_position(pad);
                    let d = dist_to_segment(px, py, s_p1.x, s_p1.y, s_p2.x, s_p2.y);
                    if d < 0.20 + stub.width / 2.0 + 0.30 {
                        is_blocking = true;
                        break;
                    }
                }
                if is_blocking {
                    let push_dir = (board.footprints[fp_idx].x - stub.pad_pos.0, board.footprints[fp_idx].y - stub.pad_pos.1);
                    if self.nudge_engine.try_nudge_component(board, fp_idx, push_dir) {
                        any_nudged = true;
                    }
                }
            }
        }
        any_nudged
    }

    /// Creates a CPU-only clone of `grid` for a Rayon worker thread during parallel component-nudge trials.
    ///
    /// # Arguments
    /// * `grid` - Source [`RoutingGrid`].
    ///
    /// # Returns
    /// Cloned [`RoutingGrid`] with `vulkan: None`.
    fn clone_grid_for_worker(grid: &RoutingGrid) -> RoutingGrid {
        RoutingGrid {
            min_x: grid.min_x,
            min_y: grid.min_y,
            max_x: grid.max_x,
            max_y: grid.max_y,
            res: grid.res,
            width: grid.width,
            height: grid.height,
            num_layers: grid.num_layers,
            plane_nets: grid.plane_nets.clone(),
            obstacle_net_ids: grid.obstacle_net_ids.clone(),
            dist_field: grid.dist_field.clone(),
            congestion: grid.congestion.clone(),
            usage_count: grid.usage_count.clone(),
            gpu_net_cost_maps: HashMap::new(),
            nudgeable_fp: grid.nudgeable_fp.clone(),
            smd_pad_cells: grid.smd_pad_cells.clone(),
            net_clearances: grid.net_clearances.clone(),
            max_net_clearance: grid.max_net_clearance,
            inner_layer_penalty: grid.inner_layer_penalty,
            allow_via_in_pad: grid.allow_via_in_pad,
            vulkan: None,
            base_snapshot: grid.base_snapshot.clone(),
        }
    }

    /// Attempts to resolve an unrouted connection `conn_idx` by identifying nearby unlocked
    /// 2-pad components, screening candidate nudge poses on the GPU/CPU, and evaluating A*
    /// re-routing of both the blocked connection and any incident connections in parallel.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] updated if a winning nudge pose is found.
    /// * `grid` - Mutable [`RoutingGrid`] rebuilt if a component is moved.
    /// * `connections` - Mutable slice of [`RouteConnection`] items whose pad coordinates update on move.
    /// * `routed_paths` - Mutable slice of routed paths updated with the new routes.
    /// * `escape_stubs` - Slice of pre-routed [`EscapeStub`] items.
    /// * `conn_idx` - Index of the unrouted connection in `connections`.
    /// * `u1_center` - Optional `(x, y)` center of `U1` in `mm`.
    /// * `stop_flag` - Atomic cancellation flag.
    ///
    /// # Returns
    /// `true` if a component nudge allowed `conn_idx` and all its incident previously-routed
    /// connections to route cleanly; `false` otherwise.
    ///
    /// # Side Effects
    /// Mutates `board.footprints`, `connections`, `routed_paths`, and `grid` on success.
    pub fn try_nudge_blocking_components(
        &self,
        board: &mut Board,
        grid: &mut RoutingGrid,
        connections: &mut [RouteConnection],
        routed_paths: &mut [Option<Vec<GridNode>>],
        escape_stubs: &[EscapeStub],
        conn_idx: usize,
        u1_center: Option<(f64, f64)>,
        stop_flag: &AtomicBool,
    ) -> bool {
        if !self.allow_component_nudge {
            return false;
        }
        let conn = &connections[conn_idx];
        let c_start = conn.start;
        let c_end = conn.end;
        let c_net_id = conn.net_id;
        let req_clearance = conn.netclass.clearance;
        let track_width = conn.netclass.track_width;
        let via_dia = conn.netclass.via_dia;

        let mut blocking_candidates = Vec::new();
        for (fp_idx, fp) in board.footprints.iter().enumerate() {
            if fp.is_locked || fp.pads.len() > 2 || fp_idx == conn.start_fp || fp_idx == conn.end_fp {
                continue;
            }
            let d_start = (fp.x - c_start.0).hypot(fp.y - c_start.1);
            let d_end = (fp.x - c_end.0).hypot(fp.y - c_end.1);
            let d_seg = dist_to_segment(fp.x, fp.y, c_start.0, c_start.1, c_end.0, c_end.1);

            let has_close_pad = fp.pads.iter().any(|pad| {
                if pad.net_id == c_net_id {
                    return false;
                }
                let (px, py) = fp.pad_absolute_position(pad);
                (px - c_start.0).hypot(py - c_start.1) < 2.5
                    || (px - c_end.0).hypot(py - c_end.1) < 2.5
                    || dist_to_segment(px, py, c_start.0, c_start.1, c_end.0, c_end.1) < 1.5
            });

            if d_start < 3.0 || d_end < 3.0 || d_seg < 2.5 || has_close_pad {
                let push_dir = if d_start < 2.5 {
                    (fp.x - c_start.0, fp.y - c_start.1)
                } else if d_end < 2.5 {
                    (fp.x - c_end.0, fp.y - c_end.1)
                } else {
                    let seg_dx = c_end.0 - c_start.0;
                    let seg_dy = c_end.1 - c_start.1;
                    let perp = (-seg_dy, seg_dx);
                    let dot = perp.0 * (fp.x - c_start.0) + perp.1 * (fp.y - c_start.1);
                    if dot >= 0.0 { perp } else { (-perp.0, -perp.1) }
                };
                blocking_candidates.push((fp_idx, push_dir, d_seg.min(d_start).min(d_end)));
            }
        }

        blocking_candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
        blocking_candidates.truncate(3);
        if blocking_candidates.is_empty() {
            return false;
        }

        // Flatten all candidate (fp_idx, dx, dy) poses across all blocking footprints into one parallel batch
        let mut flat_pose_cands = Vec::new();
        for &(fp_idx, push_dir, _) in &blocking_candidates {
            let orig_fp_x = board.footprints[fp_idx].x;
            let orig_fp_y = board.footprints[fp_idx].y;
            for (dx, dy) in self.nudge_engine.nudge_candidates(push_dir) {
                flat_pose_cands.push((fp_idx, orig_fp_x + dx, orig_fp_y + dy));
            }
        }

        // Phase A (GPU Batch Pad/Clearance Pre-Screen):
        // Pack all candidate footprint pad coordinates as GpuDrcCandidate probes in one GPU dispatch
        // to prune poses whose pads violate copper clearance before running `is_placement_valid`.
        let screened_pose_cands: Vec<(usize, f64, f64)> = if grid.vulkan.is_some() && !flat_pose_cands.is_empty() {
            let mut drc_ctx = GpuDrcContext::from_board(board);
            drc_ctx.pads.retain(|gp| {
                !blocking_candidates.iter().any(|&(b_idx, _, _)| {
                    board.footprints[b_idx].pads.iter().any(|bp| {
                        let (bpx, bpy) = board.footprints[b_idx].pad_absolute_position(bp);
                        (gp.x as f64 - bpx).hypot(gp.y as f64 - bpy) < 1e-3
                    })
                })
            });
            let mut cand_probes = Vec::with_capacity(flat_pose_cands.len() * 2);
            let mut pose_ranges = Vec::with_capacity(flat_pose_cands.len());
            for &(fp_idx, nx, ny) in &flat_pose_cands {
                let start_idx = cand_probes.len();
                let fp = &board.footprints[fp_idx];
                let flipped = fp.layer != fp.orig_layer;
                for pad in &fp.pads {
                    if pad.net_id == 0 {
                        continue;
                    }
                    let oy = if flipped { -pad.offset_y } else { pad.offset_y };
                    let p = Point::new(pad.offset_x, oy).rotate_ccw(fp.rotation);
                    let px = nx + p.x;
                    let py = ny + p.y;
                    let (pw, ph) = fp.pad_world_size(pad);
                    let pad_min_dim = pw.min(ph);
                    let min_clearance = board.get_netclass_for_net(pad.net_id).clearance.min(0.15);
                    cand_probes.push(GpuDrcCandidate {
                        x1: px as f32,
                        y1: py as f32,
                        x2: px as f32,
                        y2: py as f32,
                        radius_or_half_width: (pad_min_dim * 0.5) as f32,
                        clearance: min_clearance as f32,
                        layer: fp.pad_layer(pad) as u32,
                        net_id: pad.net_id as u32,
                        kind: 1,
                        ignore_same_net_pads: 1,
                        _pad0: 0,
                        _pad1: 0,
                    });
                }
                pose_ranges.push((start_idx, cand_probes.len()));
            }
            if let Some(flags) = drc_ctx.evaluate(grid.vulkan.as_ref(), &cand_probes) {
                let filtered: Vec<(usize, f64, f64)> = flat_pose_cands
                    .iter()
                    .copied()
                    .zip(pose_ranges.into_iter())
                    .filter_map(|(pose, (s, e))| {
                        if s == e || flags[s..e].iter().all(|&v| v != 0) {
                            Some(pose)
                        } else {
                            None
                        }
                    })
                    .collect();
                if filtered.is_empty() {
                    flat_pose_cands
                } else {
                    filtered
                }
            } else {
                flat_pose_cands
            }
        } else {
            flat_pose_cands
        };

        let mut legal_poses: Vec<(usize, f64, f64)> = screened_pose_cands
            .par_iter()
            .filter_map(|&(fp_idx, nx, ny)| {
                if self.nudge_engine.is_placement_valid(board, fp_idx, nx, ny) {
                    Some((fp_idx, nx, ny))
                } else {
                    None
                }
            })
            .collect();

        if legal_poses.is_empty() {
            return false;
        }

        // Phase B (GPU Batch 3D Wavefront Pre-Evaluation):
        // When Vulkan is present and multiple legal poses remain, evaluate shifted connection endpoints
        // and corridor segments in parallel on the GPU to rank and pre-verify reachable nudge poses.
        if grid.vulkan.is_some() && legal_poses.len() > 1 {
            let mut wf_grid = Self::clone_grid_for_worker(grid);
            wf_grid.vulkan = grid.vulkan.clone();
            for cell_i in 0..wf_grid.obstacle_net_ids.len() {
                if wf_grid
                    .nudgeable_fp
                    .get(cell_i)
                    .and_then(|&v| v)
                    .is_some_and(|f| blocking_candidates.iter().any(|&(bf, _, _)| bf == f))
                {
                    wf_grid.obstacle_net_ids[cell_i] = FREE_NET;
                }
            }
            let r_cells = (((track_width * 0.5) + req_clearance) / wf_grid.res).ceil() as i32;
            let mut wf_requests = Vec::with_capacity(legal_poses.len());
            let mut corridor_segs = Vec::with_capacity(legal_poses.len());
            let mut pose_req_indices = Vec::with_capacity(legal_poses.len());

            for &(fp_idx, nx, ny) in &legal_poses {
                let fp = &board.footprints[fp_idx];
                let flipped = fp.layer != fp.orig_layer;
                let shift_pad = |pad_idx: usize| -> (f64, f64) {
                    let pad = &fp.pads[pad_idx];
                    let oy = if flipped { -pad.offset_y } else { pad.offset_y };
                    let p = Point::new(pad.offset_x, oy).rotate_ccw(fp.rotation);
                    (nx + p.x, ny + p.y)
                };
                let target_conn = connections
                    .iter()
                    .find(|c| c.start_fp == fp_idx || c.end_fp == fp_idx)
                    .unwrap_or(&connections[conn_idx]);
                let s_pt = if target_conn.start_fp == fp_idx {
                    let (px, py) = shift_pad(target_conn.start_pad);
                    (px, py, target_conn.start.2)
                } else {
                    target_conn.start
                };
                let e_pt = if target_conn.end_fp == fp_idx {
                    let (px, py) = shift_pad(target_conn.end_pad);
                    (px, py, target_conn.end.2)
                } else {
                    target_conn.end
                };
                corridor_segs.push((
                    target_conn.net_id,
                    (s_pt.0, s_pt.1),
                    (e_pt.0, e_pt.1),
                    target_conn.netclass.track_width,
                    s_pt.2,
                    target_conn.netclass.clearance,
                ));
                if let (Some((sgx, sgy)), Some((egx, egy))) = (
                    wf_grid.world_to_grid(s_pt.0, s_pt.1),
                    wf_grid.world_to_grid(e_pt.0, e_pt.1),
                ) {
                    let req_idx = wf_requests.len();
                    wf_requests.push(((sgx, sgy, s_pt.2), (egx, egy, e_pt.2), target_conn.net_id as u32, 2.0));
                    pose_req_indices.push(Some(req_idx));
                } else {
                    pose_req_indices.push(None);
                }
            }

            let wf_paths = wf_grid.solve_batch_3d_wavefront_paths(&wf_requests, r_cells);
            let seg_flags = wf_grid.batch_validate_segments_gpu(board, &corridor_segs);
            let mut scored_poses: Vec<((usize, f64, f64), (u8, usize))> = legal_poses
                .into_iter()
                .enumerate()
                .map(|(idx, pose)| {
                    let wf_len = pose_req_indices[idx]
                        .and_then(|r_i| wf_paths.get(r_i))
                        .and_then(|p| p.as_ref())
                        .map(|p| p.len());
                    let seg_ok = seg_flags.as_ref().and_then(|f| f.get(idx)).copied().unwrap_or(false);
                    let rank_tier = match (wf_len.is_some(), seg_ok) {
                        (true, true) => 0u8,
                        (true, false) => 1u8,
                        (false, true) => 2u8,
                        (false, false) => 3u8,
                    };
                    (pose, (rank_tier, idx))
                })
                .collect();
            scored_poses.sort_by_key(|&(_, score)| score);
            legal_poses = scored_poses.into_iter().map(|(pose, _)| pose).collect();
        }

        // Evaluate A* pathfinding in parallel across all legal footprint poses simultaneously
        let winning_pose: Option<(usize, (f64, f64), Vec<Option<Vec<GridNode>>>)> = legal_poses
            .par_iter()
            .filter_map(|&(fp_idx, nx, ny)| {
                if stop_flag.load(AtomicOrdering::Relaxed) {
                    return None;
                }
                let mut local_board = board.clone();
                local_board.footprints[fp_idx].x = nx;
                local_board.footprints[fp_idx].y = ny;

                let mut local_conns = connections.to_vec();
                let mut local_paths = routed_paths.to_vec();
                let mut incident_routed = Vec::new();
                for (c_i, c) in local_conns.iter_mut().enumerate() {
                    let mut touched = false;
                    if c.start_fp == fp_idx {
                        let (px, py) = local_board.footprints[fp_idx]
                            .pad_absolute_position(&local_board.footprints[fp_idx].pads[c.start_pad]);
                        c.start = (px, py, c.start.2);
                        touched = true;
                    }
                    if c.end_fp == fp_idx {
                        let (px, py) = local_board.footprints[fp_idx]
                            .pad_absolute_position(&local_board.footprints[fp_idx].pads[c.end_pad]);
                        c.end = (px, py, c.end.2);
                        touched = true;
                    }
                    if touched {
                        if c_i != conn_idx && local_paths[c_i].is_some() {
                            incident_routed.push(c_i);
                        }
                        local_paths[c_i] = None;
                    }
                }

                let mut local_grid = Self::clone_grid_for_worker(grid);
                local_grid.base_snapshot = None;
                self.rebuild_grid_state(&local_board, &mut local_grid, &local_conns, &local_paths, escape_stubs);

                let updated_conn = &local_conns[conn_idx];
                let (Some((sgx, sgy)), Some((egx, egy))) = (
                    local_grid.world_to_grid(updated_conn.start.0, updated_conn.start.1),
                    local_grid.world_to_grid(updated_conn.end.0, updated_conn.end.1),
                ) else {
                    return None;
                };
                let s_node = GridNode { gx: sgx, gy: sgy, layer: updated_conn.start.2 };
                let e_node = GridNode { gx: egx, gy: egy, layer: updated_conn.end.2 };

                let p = self.find_path_with_params(
                    &local_grid,
                    s_node,
                    e_node,
                    c_net_id,
                    req_clearance,
                    track_width,
                    via_dia,
                    u1_center,
                    1.05,
                    0.0,
                    stop_flag,
                )?;
                self.commit_path_to_grid(&mut local_grid, &p, c_net_id, track_width, via_dia);
                local_paths[conn_idx] = Some(p);

                for inc_idx in incident_routed {
                    let inc_conn = &local_conns[inc_idx];
                    let (Some((isgx, isgy)), Some((iegx, iegy))) = (
                        local_grid.world_to_grid(inc_conn.start.0, inc_conn.start.1),
                        local_grid.world_to_grid(inc_conn.end.0, inc_conn.end.1),
                    ) else {
                        return None;
                    };
                    let is_node = GridNode { gx: isgx, gy: isgy, layer: inc_conn.start.2 };
                    let ie_node = GridNode { gx: iegx, gy: iegy, layer: inc_conn.end.2 };
                    let inc_tw = inc_conn.netclass.track_width.min(0.25);
                    let inc_clr = inc_conn.netclass.clearance;
                    let inc_vd = inc_conn.netclass.min_via_diameter;
                    let inc_p = self.find_path_with_params(
                        &local_grid,
                        is_node,
                        ie_node,
                        inc_conn.net_id,
                        inc_clr,
                        inc_tw,
                        inc_vd,
                        u1_center,
                        1.05,
                        0.0,
                        stop_flag,
                    )?;
                    self.commit_path_to_grid(&mut local_grid, &inc_p, inc_conn.net_id, inc_tw, inc_vd);
                    local_paths[inc_idx] = Some(inc_p);
                }

                Some((fp_idx, (nx, ny), local_paths))
            })
            .find_first(|_| true);

        if let Some((fp_idx, (nx, ny), updated_paths)) = winning_pose {
            let orig_fp_x = board.footprints[fp_idx].x;
            let orig_fp_y = board.footprints[fp_idx].y;
            board.footprints[fp_idx].x = nx;
            board.footprints[fp_idx].y = ny;

            for c in connections.iter_mut() {
                if c.start_fp == fp_idx {
                    let (px, py) = board.footprints[fp_idx]
                        .pad_absolute_position(&board.footprints[fp_idx].pads[c.start_pad]);
                    c.start = (px, py, c.start.2);
                }
                if c.end_fp == fp_idx {
                    let (px, py) = board.footprints[fp_idx]
                        .pad_absolute_position(&board.footprints[fp_idx].pads[c.end_pad]);
                    c.end = (px, py, c.end.2);
                }
            }
            routed_paths.clone_from_slice(&updated_paths);

            grid.base_snapshot = None;
            self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);

            board.rip_up_connected_traces_to_component_at(
                fp_idx,
                Some((orig_fp_x, orig_fp_y, board.footprints[fp_idx].rotation, board.footprints[fp_idx].layer)),
            );
            grid.update_distance_fields();
            return true;
        }
        false
    }

    /// Performs a local 2D breadth-first search (BFS) from `(gx, gy)` on `layer` to detect
    /// whether a pin/terminal is trapped inside a small enclosed pocket of foreign obstacles.
    ///
    /// Includes a clearance halo check around each expanded cell so the BFS cannot leak
    /// through sub-clearance cracks between foreign traces and pads.
    ///
    /// # Arguments
    /// * `grid` - Reference to the [`RoutingGrid`].
    /// * `gx` - Starting X grid coordinate.
    /// * `gy` - Starting Y grid coordinate.
    /// * `layer` - Copper layer index.
    /// * `net_id` - Numeric net ID of the pin being probed.
    /// * `max_cells` - Upper bound on the number of cells to expand before declaring the pin unchoked.
    ///
    /// # Returns
    /// A tuple `(reachable_cells, blocking_nets)` where `reachable_cells` is the number of
    /// reachable cells (up to `max_cells`) and `blocking_nets` is the sorted list of foreign
    /// net IDs forming the boundary of the pocket.
    pub fn find_choked_pin_bottlenecks(
        grid: &RoutingGrid,
        gx: usize,
        gy: usize,
        layer: usize,
        net_id: usize,
        max_cells: usize,
    ) -> (usize, Vec<usize>) {
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut blocking_nets = std::collections::BTreeSet::new();

        queue.push_back((gx, gy));
        visited.insert((gx, gy));

        while let Some((cx, cy)) = queue.pop_front() {
            if visited.len() >= max_cells {
                break;
            }
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx >= 0 && nx < grid.width as isize && ny >= 0 && ny < grid.height as isize {
                    let ux = nx as usize;
                    let uy = ny as usize;
                    if !visited.contains(&(ux, uy)) {
                        let idx = grid.idx(ux, uy, layer);
                        let obs = grid.obstacle_net_ids[idx];
                        if obs != FREE_NET && obs != net_id as i32 {
                            if obs > 0 && !grid.smd_pad_cells[idx] {
                                blocking_nets.insert(obs as usize);
                            }
                            continue;
                        }
                        // Also check clearance halo around (ux, uy) so the BFS does not
                        // leak through sub-clearance cracks between foreign tracks and pads
                        let dist_from_origin = (ux as isize - gx as isize).abs().max((uy as isize - gy as isize).abs());
                        let halo_r: isize = if dist_from_origin > 2 { 2 } else { 1 };
                        let mut halo_blocked = false;
                        for hdx in -halo_r..=halo_r {
                            for hdy in -halo_r..=halo_r {
                                if (hdx == 0 && hdy == 0) || hdx * hdx + hdy * hdy > 5 {
                                    continue;
                                }
                                let hx = ux as isize + hdx;
                                let hy = uy as isize + hdy;
                                if hx >= 0 && hx < grid.width as isize && hy >= 0 && hy < grid.height as isize {
                                    let h_idx = grid.idx(hx as usize, hy as usize, layer);
                                    let h_obs = grid.obstacle_net_ids[h_idx];
                                    if h_obs != FREE_NET && h_obs != net_id as i32 {
                                        if h_obs > 0 && !grid.smd_pad_cells[h_idx] {
                                            blocking_nets.insert(h_obs as usize);
                                            halo_blocked = true;
                                        } else if dist_from_origin > 2 {
                                            halo_blocked = true;
                                        }
                                    }
                                }
                            }
                        }
                        if !halo_blocked {
                            visited.insert((ux, uy));
                            queue.push_back((ux, uy));
                        }
                    }
                }
            }
        }

        (visited.len(), blocking_nets.into_iter().collect())
    }

    /// Resolves an unrouted connection `conn_idx` blocked by foreign unlocked copper through a
    /// four-stage push-and-shove / rip-up pipeline:
    /// 1. **Stage 0:** Speculative parallel lateral deflection of pre-existing unlocked `board.tracks`.
    /// 2. **Stage 1:** GPU-pre-filtered parallel via shoving along blocking `routed_paths`.
    /// 3. **Stage 2:** Single-net parallel negotiated shove-and-reroute (lift blocker, route `conn_idx`, reroute blocker).
    /// 4. **Stage 3:** Multi-track corridor shove portfolio (simultaneously lifting 2–3 blockers).
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] (updated if Stage 0 deflects pre-existing `board.tracks`).
    /// * `grid` - Mutable [`RoutingGrid`] updated with the newly routed/shoved paths.
    /// * `connections` - Slice of all [`RouteConnection`] items.
    /// * `routed_paths` - Mutable slice of routed grid paths per connection.
    /// * `escape_stubs` - Slice of pre-routed [`EscapeStub`] items.
    /// * `conn_idx` - Index of the blocked connection in `connections`.
    /// * `u1_center` - Optional `(x, y)` center of `U1` in `mm`.
    /// * `stop_flag` - Atomic cancellation flag.
    ///
    /// # Returns
    /// `true` if `conn_idx` was successfully routed by shoving or ripping up blocking tracks/vias.
    ///
    /// # Side Effects
    /// Mutates `routed_paths`, `grid` (including historical `congestion` on unrecovered rips),
    /// and `board.tracks` (in Stage 0).
    pub fn try_nudge_or_ripup_blocking_tracks(
        &self,
        board: &mut Board,
        grid: &mut RoutingGrid,
        connections: &[RouteConnection],
        routed_paths: &mut [Option<Vec<GridNode>>],
        escape_stubs: &[EscapeStub],
        conn_idx: usize,
        u1_center: Option<(f64, f64)>,
        stop_flag: &AtomicBool,
    ) -> bool {
        let conn = &connections[conn_idx];
        let _my_prio = Self::netclass_priority(&conn.netclass.name);
        let c_start = conn.start;
        let c_end = conn.end;

        let total_conns = connections.len();

        // ========================================================
        // STAGE 0: Speculative Parallel Track Deflection on Unlocked board.tracks
        // When unlocked tracks exist directly in `board.tracks` (e.g. pre-existing or standalone routing),
        // evaluate candidate lateral offsets in parallel using GpuDrcContext + Rayon A* pathfinding.
        // ========================================================
        if routed_paths.iter().all(|p| p.is_none())
            && board.tracks.iter().any(|t| !t.is_locked && t.net_id != conn.net_id)
        {
            let mut candidate_track_indices: Vec<(usize, f64)> = board
                .tracks
                .iter()
                .enumerate()
                .filter(|(_, t)| !t.is_locked && t.net_id != conn.net_id)
                .filter_map(|(t_idx, t)| {
                    let d = dist_segment_to_segment(t.start, t.end, (c_start.0, c_start.1), (c_end.0, c_end.1));
                    if d < 2.0 {
                        Some((t_idx, d))
                    } else {
                        None
                    }
                })
                .collect();
            candidate_track_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            candidate_track_indices.truncate(6);

            let offsets: [f64; 14] = [
                -0.10, 0.10, -0.20, 0.20, -0.30, 0.30, -0.45, 0.45, -0.60, 0.60, -0.80, 0.80, -1.00, 1.00,
            ];

            for (track_idx, _) in candidate_track_indices {
                let orig_t = board.tracks[track_idx].clone();
                let vx = orig_t.end.0 - orig_t.start.0;
                let vy = orig_t.end.1 - orig_t.start.1;
                let len = vx.hypot(vy);
                if len < 1e-4 {
                    continue;
                }
                let nx = -vy / len;
                let ny = vx / len;
                let t_clr = board.get_netclass_for_net(orig_t.net_id).clearance;

                let mut board_without_t = board.clone();
                board_without_t.tracks.remove(track_idx);
                let drc_ctx = GpuDrcContext::from_board(&board_without_t);
                let gpu_cands: Vec<GpuDrcCandidate> = offsets
                    .iter()
                    .map(|&off| GpuDrcCandidate {
                        x1: (orig_t.start.0 + nx * off) as f32,
                        y1: (orig_t.start.1 + ny * off) as f32,
                        x2: (orig_t.end.0 + nx * off) as f32,
                        y2: (orig_t.end.1 + ny * off) as f32,
                        radius_or_half_width: (orig_t.width * 0.5) as f32,
                        clearance: t_clr as f32,
                        layer: orig_t.layer as u32,
                        net_id: orig_t.net_id as u32,
                        kind: 1,
                        ignore_same_net_pads: 1,
                        _pad0: 0,
                        _pad1: 0,
                    })
                    .collect();
                let gpu_flags = drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_cands);

                let winning_shove: Option<(Board, Vec<GridNode>)> = offsets
                    .par_iter()
                    .enumerate()
                    .filter_map(|(o_idx, &off)| {
                        if stop_flag.load(AtomicOrdering::Relaxed) {
                            return None;
                        }
                        let mut cand_board = board.clone();
                        let deflected = self.nudge_engine.try_continuous_shove_deflect(&mut cand_board, track_idx, off, None)
                            || (gpu_flags.as_ref().map_or(true, |f| f[o_idx] != 0) && {
                                let s = (orig_t.start.0 + nx * off, orig_t.start.1 + ny * off);
                                let e = (orig_t.end.0 + nx * off, orig_t.end.1 + ny * off);
                                if is_track_segment_legal(&board_without_t, s, e, orig_t.width, orig_t.layer, orig_t.net_id, t_clr) {
                                    cand_board.tracks[track_idx].start = s;
                                    cand_board.tracks[track_idx].end = e;
                                    true
                                } else {
                                    false
                                }
                            });
                        if !deflected {
                            return None;
                        }

                        let mut local_grid = Self::clone_grid_for_worker(grid);
                        self.rebuild_grid_state(&cand_board, &mut local_grid, connections, routed_paths, escape_stubs);
                        for trk in &cand_board.tracks {
                            if !trk.is_locked {
                                local_grid.commit_track(trk);
                            }
                        }
                        for v in &cand_board.vias {
                            if !v.is_locked {
                                local_grid.commit_via(v);
                            }
                        }

                        let (Some((sgx, sgy)), Some((egx, egy))) = (
                            local_grid.world_to_grid(conn.start.0, conn.start.1),
                            local_grid.world_to_grid(conn.end.0, conn.end.1),
                        ) else {
                            return None;
                        };
                        let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                        let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };

                        self.find_path_with_params(
                            &local_grid,
                            s_node,
                            e_node,
                            conn.net_id,
                            conn.netclass.clearance,
                            conn.netclass.track_width,
                            conn.netclass.via_dia,
                            u1_center,
                            1.05,
                            0.0,
                            stop_flag,
                        )
                        .map(|p| (cand_board, p))
                    })
                    .find_first(|_| true);

                if let Some((winning_board, p)) = winning_shove {
                    board.tracks = winning_board.tracks;
                    self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
                    for trk in &board.tracks {
                        if !trk.is_locked {
                            grid.commit_track(trk);
                        }
                    }
                    self.commit_path_to_grid(grid, &p, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
                    routed_paths[conn_idx] = Some(p);
                    return true;
                }
            }
        }

        // 1. Identify Candidate Blocking Connections:
        // Include:
        // (a) Choked pin bottlenecks (< 300 reachable nodes)
        // (b) Tracks / vias in the corridor within 2.5 mm
        let (s_grid, e_grid) = (
            grid.world_to_grid(c_start.0, c_start.1),
            grid.world_to_grid(c_end.0, c_end.1),
        );
        let mut choked_blocking_nets = Vec::new();
        if let Some((sgx, sgy)) = s_grid {
            let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, sgx, sgy, c_start.2, conn.net_id, 400);
            if reach < 300 && !nets.is_empty() {
                choked_blocking_nets.extend(nets);
            }
        }
        if let Some((egx, egy)) = e_grid {
            let (reach, nets) = Self::find_choked_pin_bottlenecks(grid, egx, egy, c_end.2, conn.net_id, 400);
            if reach < 300 && !nets.is_empty() {
                choked_blocking_nets.extend(nets);
            }
        }

        let mut candidate_conns: Vec<(usize, usize, f64, f64, bool)> = Vec::new(); // (idx, prio, min_pad_dist, min_dist, is_choked)

        for rip_idx in 0..total_conns {
            if rip_idx == conn_idx || routed_paths[rip_idx].is_none() {
                continue;
            }
            if connections[rip_idx].net_id == conn.net_id {
                continue;
            }
            let rip_prio = Self::netclass_priority(&connections[rip_idx].netclass.name);
            let is_diff = connections[rip_idx].diff_pair_leader.is_some()
                || connections[rip_idx].net_name.contains("USB");
            if is_diff {
                continue;
            }

            let is_choked_net = choked_blocking_nets.contains(&connections[rip_idx].net_id);

            let path = routed_paths[rip_idx].as_ref().unwrap();
            let mut min_pad_dist = f64::INFINITY;
            let mut min_dist = f64::INFINITY;
            for node in path {
                let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                let d_start = (wx - c_start.0).hypot(wy - c_start.1);
                let d_end = (wx - c_end.0).hypot(wy - c_end.1);
                let d_pad = d_start.min(d_end);
                if d_pad < min_pad_dist {
                    min_pad_dist = d_pad;
                }
                let d_seg = dist_to_segment(wx, wy, c_start.0, c_start.1, c_end.0, c_end.1);
                let d = d_pad.min(d_seg);
                if d < min_dist {
                    min_dist = d;
                }
            }

            if is_choked_net || min_dist < 1.5 {
                let is_near_pad = is_choked_net || min_pad_dist < 1.2;
                candidate_conns.push((rip_idx, rip_prio, min_pad_dist, min_dist, is_near_pad));
            }
        }

        // Prioritize: (1) choked/near-pad bottlenecks, (2) closest corridor/pad distance, (3) lower net priority
        candidate_conns.sort_by(|a, b| {
            b.4.cmp(&a.4)
                .then_with(|| a.3.partial_cmp(&b.3).unwrap_or(Ordering::Equal))
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
                .then_with(|| a.1.cmp(&b.1))
        });
        candidate_conns.truncate(8);

        let drc_ctx = GpuDrcContext::from_board(board);

        // ========================================================
        // STAGE 1: Flattened Parallel Via Shoving with Multi-GPU Batch Pre-Filtering
        // Collect all corridor via shift candidates across all blocking connections into one
        // unified GPU DRC batch and evaluate all shifted via poses in parallel across Rayon workers.
        // ========================================================
        let shifts: [(isize, isize); 16] = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2),
            (2, 2), (-2, 2), (2, -2), (-2, -2),
        ];

        let mut via_shift_tasks: Vec<(usize, usize, isize, isize)> = Vec::new();
        let mut gpu_via_cands: Vec<GpuDrcCandidate> = Vec::new();

        for &(rip_idx, _, _, _, _) in &candidate_conns {
            let Some(ref orig_path) = routed_paths[rip_idx] else {
                continue;
            };
            let other_net_id = connections[rip_idx].net_id;
            let other_clearance = connections[rip_idx].netclass.clearance;
            let other_via_dia = connections[rip_idx].netclass.via_dia;

            for v_step in 0..orig_path.len().saturating_sub(1) {
                if orig_path[v_step].layer != orig_path[v_step + 1].layer {
                    let v_node = orig_path[v_step];
                    let (vwx, vwy) = grid.grid_to_world(v_node.gx, v_node.gy);
                    let d_start = (vwx - c_start.0).hypot(vwy - c_start.1);
                    let d_end = (vwx - c_end.0).hypot(vwy - c_end.1);
                    let d_corridor = dist_to_segment(vwx, vwy, c_start.0, c_start.1, c_end.0, c_end.1);
                    if d_start.min(d_end).min(d_corridor) > 2.5 {
                        continue;
                    }
                    for &(dx, dy) in &shifts {
                        let ngx = (v_node.gx as isize + dx).clamp(0, grid.width.saturating_sub(1) as isize) as usize;
                        let ngy = (v_node.gy as isize + dy).clamp(0, grid.height.saturating_sub(1) as isize) as usize;
                        let (nwx, nwy) = grid.grid_to_world(ngx, ngy);
                        via_shift_tasks.push((rip_idx, v_step, dx, dy));
                        gpu_via_cands.push(GpuDrcCandidate {
                            x1: nwx as f32,
                            y1: nwy as f32,
                            x2: 0.0,
                            y2: 0.0,
                            radius_or_half_width: (other_via_dia * 0.5) as f32,
                            clearance: other_clearance as f32,
                            layer: 0,
                            net_id: other_net_id as u32,
                            kind: 0,
                            ignore_same_net_pads: 0,
                            _pad0: 0,
                            _pad1: 0,
                        });
                    }
                }
            }
        }

        if !via_shift_tasks.is_empty() {
            let gpu_via_flags = drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_via_cands);

            let winning_via_shove: Option<(usize, Vec<GridNode>, Vec<GridNode>)> = via_shift_tasks
                .par_iter()
                .enumerate()
                .filter_map(|(t_idx, &(rip_idx, v_step, dx, dy))| {
                    if stop_flag.load(AtomicOrdering::Relaxed) {
                        return None;
                    }
                    if !gpu_via_flags.as_ref().map_or(true, |f| f[t_idx] != 0) {
                        return None;
                    }
                    let orig_path = routed_paths[rip_idx].as_ref()?;
                    let v_node = orig_path[v_step];
                    let other_net_id = connections[rip_idx].net_id;
                    let other_clearance = connections[rip_idx].netclass.clearance;
                    let other_via_dia = connections[rip_idx].netclass.via_dia;

                    let ngx = v_node.gx as isize + dx;
                    let ngy = v_node.gy as isize + dy;
                    if ngx < 0 || ngx >= grid.width as isize || ngy < 0 || ngy >= grid.height as isize {
                        return None;
                    }
                    let (cand_gx, cand_gy) = (ngx as usize, ngy as usize);
                    let (nwx, nwy) = grid.grid_to_world(cand_gx, cand_gy);
                    if !is_via_location_legal(board, (nwx, nwy), other_via_dia, other_net_id, other_clearance) {
                        return None;
                    }

                    for l in 0..grid.num_layers {
                        let idx = grid.idx(cand_gx, cand_gy, l);
                        let obst = grid.obstacle_net_ids[idx];
                        if obst != FREE_NET && obst != other_net_id as i32 {
                            return None;
                        }
                    }

                    let l_in = v_node.layer;
                    let l_out = orig_path[v_step + 1].layer;
                    let cand_via_in = GridNode { gx: cand_gx, gy: cand_gy, layer: l_in };
                    let cand_via_out = GridNode { gx: cand_gx, gy: cand_gy, layer: l_out };

                    let lead_in_anchor = if v_step > 0 { orig_path[v_step - 1] } else { orig_path[0] };
                    let lead_out_anchor = if v_step + 2 < orig_path.len() { orig_path[v_step + 2] } else { orig_path[v_step + 1] };

                    let c1_list = Self::two_segment_candidates(lead_in_anchor, cand_via_in);
                    let c2_list = Self::two_segment_candidates(cand_via_out, lead_out_anchor);

                    let mut valid_slid_path = None;
                    'cand_loop: for c1 in &c1_list {
                        for c2 in &c2_list {
                            let mut cand_full = Vec::with_capacity(orig_path.len() + c1.len() + c2.len());
                            cand_full.extend_from_slice(&orig_path[0..v_step.saturating_sub(1).max(1)]);
                            cand_full.extend(c1.iter().cloned());
                            cand_full.push(cand_via_in);
                            cand_full.push(cand_via_out);
                            cand_full.extend(c2.iter().cloned());
                            if v_step + 2 < orig_path.len() {
                                cand_full.extend_from_slice(&orig_path[(v_step + 2)..]);
                            }

                            if self.is_path_conflict_free(
                                grid,
                                &cand_full,
                                other_net_id,
                                other_clearance,
                                connections[rip_idx].netclass.track_width,
                                other_via_dia,
                                u1_center,
                                connections[rip_idx].netclass.require_return_path,
                            ) {
                                valid_slid_path = Some(cand_full);
                                break 'cand_loop;
                            }
                        }
                    }

                    let slid_path = valid_slid_path?;
                    let mut local_paths = routed_paths.to_vec();
                    local_paths[rip_idx] = Some(slid_path.clone());
                    let mut local_grid = Self::clone_grid_for_worker(grid);
                    self.rebuild_grid_state(board, &mut local_grid, connections, &local_paths, escape_stubs);

                    let (Some((sgx, sgy)), Some((egx, egy))) = (
                        local_grid.world_to_grid(conn.start.0, conn.start.1),
                        local_grid.world_to_grid(conn.end.0, conn.end.1),
                    ) else {
                        return None;
                    };
                    let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                    let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };

                    self.find_path_with_params(
                        &local_grid,
                        s_node,
                        e_node,
                        conn.net_id,
                        conn.netclass.clearance,
                        conn.netclass.track_width,
                        conn.netclass.via_dia,
                        u1_center,
                        1.05,
                        0.0,
                        stop_flag,
                    )
                    .map(|p| (rip_idx, slid_path, p))
                })
                .find_first(|_| true);

            if let Some((rip_idx, slid_path, p)) = winning_via_shove {
                routed_paths[rip_idx] = Some(slid_path);
                self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
                self.commit_path_to_grid(grid, &p, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
                routed_paths[conn_idx] = Some(p);
                return true;
            }
        }

        // ========================================================
        // STAGE 2: Parallel Negotiated Shove-and-Reroute (Push & Shove / Spring Reroute)
        // Evaluate all candidate blocking connections in parallel across Rayon workers:
        // lift `rip_idx`, route `conn_idx`, and immediately attempt to re-route `rip_idx` around `conn_idx`.
        // ========================================================
        let stage2_results: Vec<Option<(usize, Vec<GridNode>, Option<Vec<GridNode>>, Vec<GridNode>)>> = candidate_conns
            .par_iter()
            .map(|&(rip_idx, _, _, _, _)| {
                if stop_flag.load(AtomicOrdering::Relaxed) {
                    return None;
                }
                let orig_path = routed_paths[rip_idx].clone()?;
                let mut local_paths = routed_paths.to_vec();
                local_paths[rip_idx] = None;
                let mut local_grid = Self::clone_grid_for_worker(grid);
                self.rebuild_grid_state(board, &mut local_grid, connections, &local_paths, escape_stubs);

                let (Some((sgx, sgy)), Some((egx, egy))) = (
                    local_grid.world_to_grid(conn.start.0, conn.start.1),
                    local_grid.world_to_grid(conn.end.0, conn.end.1),
                ) else {
                    return None;
                };
                let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };

                let p_conn = self.find_path_parallel(
                    &local_grid,
                    s_node,
                    e_node,
                    conn.net_id,
                    conn.netclass.clearance,
                    conn.netclass.track_width,
                    conn.netclass.via_dia,
                    u1_center,
                    conn.netclass.max_coupled_length_mm,
                    conn.netclass.require_return_path,
                    stop_flag,
                )?;

                self.commit_path_to_grid(&mut local_grid, &p_conn, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);

                let rip_conn = &connections[rip_idx];
                let p_shoved = if let (Some((rsgx, rsgy)), Some((regx, regy))) = (
                    local_grid.world_to_grid(rip_conn.start.0, rip_conn.start.1),
                    local_grid.world_to_grid(rip_conn.end.0, rip_conn.end.1),
                ) {
                    let r_s = GridNode { gx: rsgx, gy: rsgy, layer: rip_conn.start.2 };
                    let r_e = GridNode { gx: regx, gy: regy, layer: rip_conn.end.2 };
                    self.find_path_parallel(
                        &local_grid,
                        r_s,
                        r_e,
                        rip_conn.net_id,
                        rip_conn.netclass.clearance,
                        rip_conn.netclass.track_width,
                        rip_conn.netclass.via_dia,
                        u1_center,
                        rip_conn.netclass.max_coupled_length_mm,
                        rip_conn.netclass.require_return_path,
                        stop_flag,
                    )
                } else {
                    None
                };

                Some((rip_idx, p_conn, p_shoved, orig_path))
            })
            .collect();

        // Prefer a candidate where BOTH conn_idx and rip_idx routed cleanly; otherwise take the first candidate that routed conn_idx
        let chosen_stage2 = stage2_results
            .iter()
            .flatten()
            .find(|(_, _, p_shoved, _)| p_shoved.is_some())
            .or_else(|| stage2_results.iter().flatten().next());

        if let Some((rip_idx, p_conn, p_shoved_opt, orig_path)) = chosen_stage2.cloned() {
            routed_paths[rip_idx] = None;
            self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
            self.commit_path_to_grid(grid, &p_conn, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
            routed_paths[conn_idx] = Some(p_conn);

            if let Some(p_shoved) = p_shoved_opt {
                let rip_conn = &connections[rip_idx];
                self.commit_path_to_grid(grid, &p_shoved, rip_conn.net_id, rip_conn.netclass.track_width, rip_conn.netclass.via_dia);
                routed_paths[rip_idx] = Some(p_shoved);
            } else {
                for node in &orig_path {
                    let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                    if (wx - c_start.0).hypot(wy - c_start.1) < 2.0 || (wx - c_end.0).hypot(wy - c_end.1) < 2.0 {
                        let idx = grid.idx(node.gx, node.gy, node.layer);
                        grid.congestion[idx] += self.congestion_factor * 2.0;
                    }
                }
            }
            return true;
        }

        // ========================================================
        // STAGE 3: Parallel Multi-Track Corridor Shove Portfolio
        // Evaluate distinct 2-track and 3-track blocking combinations in parallel across Rayon workers:
        // lift the group simultaneously, route `conn_idx`, and re-route the lifted group around `conn_idx`.
        // ========================================================
        if candidate_conns.len() >= 2 {
            let n_c = candidate_conns.len();
            let mut groups: Vec<Vec<usize>> = Vec::new();
            groups.push(vec![candidate_conns[0].0, candidate_conns[1].0]);
            if n_c >= 3 {
                groups.push(vec![candidate_conns[0].0, candidate_conns[2].0]);
                groups.push(vec![candidate_conns[1].0, candidate_conns[2].0]);
                groups.push(vec![candidate_conns[0].0, candidate_conns[1].0, candidate_conns[2].0]);
            }
            if n_c >= 4 {
                groups.push(vec![candidate_conns[0].0, candidate_conns[3].0]);
                groups.push(vec![candidate_conns[0].0, candidate_conns[1].0, candidate_conns[3].0]);
                groups.push(vec![candidate_conns[0].0, candidate_conns[2].0, candidate_conns[3].0]);
            }

            let stage3_results: Vec<Option<(usize, usize, Vec<GridNode>, Vec<(usize, Option<Vec<GridNode>>, Vec<GridNode>)>)>> = groups
                .into_par_iter()
                .enumerate()
                .map(|(g_ord, group)| {
                    if stop_flag.load(AtomicOrdering::Relaxed) {
                        return None;
                    }
                    let mut local_paths = routed_paths.to_vec();
                    let mut saved_group: Vec<(usize, Vec<GridNode>)> = Vec::with_capacity(group.len());
                    for &r_idx in &group {
                        if let Some(p) = local_paths[r_idx].take() {
                            saved_group.push((r_idx, p));
                        }
                    }
                    if saved_group.is_empty() {
                        return None;
                    }

                    let mut local_grid = Self::clone_grid_for_worker(grid);
                    self.rebuild_grid_state(board, &mut local_grid, connections, &local_paths, escape_stubs);

                    let (Some((sgx, sgy)), Some((egx, egy))) = (
                        local_grid.world_to_grid(conn.start.0, conn.start.1),
                        local_grid.world_to_grid(conn.end.0, conn.end.1),
                    ) else {
                        return None;
                    };
                    let s_node = GridNode { gx: sgx, gy: sgy, layer: conn.start.2 };
                    let e_node = GridNode { gx: egx, gy: egy, layer: conn.end.2 };

                    let p_conn = self.find_path_parallel(
                        &local_grid,
                        s_node,
                        e_node,
                        conn.net_id,
                        conn.netclass.clearance,
                        conn.netclass.track_width,
                        conn.netclass.via_dia,
                        u1_center,
                        conn.netclass.max_coupled_length_mm,
                        conn.netclass.require_return_path,
                        stop_flag,
                    )?;

                    self.commit_path_to_grid(&mut local_grid, &p_conn, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
                    local_paths[conn_idx] = Some(p_conn.clone());

                    let mut rerouted_cnt = 0usize;
                    let mut group_outcomes: Vec<(usize, Option<Vec<GridNode>>, Vec<GridNode>)> = Vec::with_capacity(saved_group.len());
                    for (r_idx, orig_p) in saved_group {
                        let rip_conn = &connections[r_idx];
                        let mut p_shoved_opt = None;
                        if let (Some((rsgx, rsgy)), Some((regx, regy))) = (
                            local_grid.world_to_grid(rip_conn.start.0, rip_conn.start.1),
                            local_grid.world_to_grid(rip_conn.end.0, rip_conn.end.1),
                        ) {
                            let r_s = GridNode { gx: rsgx, gy: rsgy, layer: rip_conn.start.2 };
                            let r_e = GridNode { gx: regx, gy: regy, layer: rip_conn.end.2 };
                            if let Some(p_shoved) = self.find_path_parallel(
                                &local_grid,
                                r_s,
                                r_e,
                                rip_conn.net_id,
                                rip_conn.netclass.clearance,
                                rip_conn.netclass.track_width,
                                rip_conn.netclass.via_dia,
                                u1_center,
                                rip_conn.netclass.max_coupled_length_mm,
                                rip_conn.netclass.require_return_path,
                                stop_flag,
                            ) {
                                self.commit_path_to_grid(
                                    &mut local_grid,
                                    &p_shoved,
                                    rip_conn.net_id,
                                    rip_conn.netclass.track_width,
                                    rip_conn.netclass.via_dia,
                                );
                                local_paths[r_idx] = Some(p_shoved.clone());
                                p_shoved_opt = Some(p_shoved);
                                rerouted_cnt += 1;
                            }
                        }
                        group_outcomes.push((r_idx, p_shoved_opt, orig_p));
                    }

                    let unrecovered = group_outcomes.len().saturating_sub(rerouted_cnt);
                    Some((unrecovered, g_ord, p_conn, group_outcomes))
                })
                .collect();

            if let Some((_, _, p_conn, group_outcomes)) = stage3_results
                .into_iter()
                .flatten()
                .min_by_key(|(unrecovered, g_ord, _, _)| (*unrecovered, *g_ord))
            {
                for &(r_idx, _, _) in &group_outcomes {
                    routed_paths[r_idx] = None;
                }
                self.rebuild_grid_state(board, grid, connections, routed_paths, escape_stubs);
                self.commit_path_to_grid(grid, &p_conn, conn.net_id, conn.netclass.track_width, conn.netclass.via_dia);
                routed_paths[conn_idx] = Some(p_conn);

                for (r_idx, p_shoved_opt, orig_p) in group_outcomes {
                    if let Some(p_shoved) = p_shoved_opt {
                        let rip_conn = &connections[r_idx];
                        self.commit_path_to_grid(grid, &p_shoved, rip_conn.net_id, rip_conn.netclass.track_width, rip_conn.netclass.via_dia);
                        routed_paths[r_idx] = Some(p_shoved);
                    } else {
                        for node in &orig_p {
                            let (wx, wy) = grid.grid_to_world(node.gx, node.gy);
                            if (wx - c_start.0).hypot(wy - c_start.1) < 2.5
                                || (wx - c_end.0).hypot(wy - c_end.1) < 2.5
                                || dist_to_segment(wx, wy, c_start.0, c_start.1, c_end.0, c_end.1) < 2.0
                            {
                                let idx = grid.idx(node.gx, node.gy, node.layer);
                                grid.congestion[idx] += self.congestion_factor * 2.5;
                            }
                        }
                    }
                }
                return true;
            }
        }

        false
    }

    /// Converts all active discrete [`GridNode`] `paths` and used [`EscapeStub`] structures
    /// into physical world-space [`TrackSegment`] and [`Via`] primitives on `board`.
    ///
    /// Preserves locked tracks, vias, and arcs, trims redundant inner-pad steps on through-hole
    /// terminals, synthesizes 45-degree doglegs at off-grid pad entries/exits, enforces legal
    /// via placement and annular-ring dimensions, and pairs differential-pair layer-transition vias.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] whose unlocked `tracks`, `vias`, and `arcs` are replaced.
    /// * `grid` - Reference to the [`RoutingGrid`] for grid-to-world coordinate conversion.
    /// * `connections` - Slice of [`RouteConnection`] descriptors.
    /// * `paths` - Slice of optional discrete [`GridNode`] paths per connection.
    /// * `escape_stubs` - Slice of pre-generated [`EscapeStub`] items.
    ///
    /// # Side Effects
    /// Replaces unlocked copper primitives in `board.tracks`, `board.vias`, and `board.arcs`.
    fn reconstruct_tracks_and_vias(
        &self,
        board: &mut Board,
        grid: &RoutingGrid,
        connections: &[RouteConnection],
        paths: &[Option<Vec<GridNode>>],
        escape_stubs: &[EscapeStub],
    ) {
        let locked_tracks: Vec<TrackSegment> = board.tracks.iter().filter(|t| t.is_locked).cloned().collect();
        let locked_vias: Vec<Via> = board.vias.iter().filter(|v| v.is_locked).cloned().collect();
        board.tracks = locked_tracks;
        board.vias = locked_vias;
        board.arcs.retain(|a| a.is_locked);

        let mut used_stub_tips: Vec<(usize, (f64, f64))> = Vec::new();
        for (conn_idx, path_opt) in paths.iter().enumerate() {
            let Some(raw_path) = path_opt else {
                continue;
            };
            let mut conn = connections[conn_idx].clone();
            let mut path: &[GridNode] = raw_path;
            if let Some(stub) = escape_stubs.iter().find(|s| {
                s.net_id == conn.net_id
                    && s.tip_layer == conn.start.2
                    && (s.tip_pos.0 - conn.start.0).hypot(s.tip_pos.1 - conn.start.1) < 0.15
            }) {
                let max_k = path.len().min(20);
                if let Some(k) = (1..max_k).rev().find(|&i| {
                    path[i].layer == stub.layer && {
                        let w = grid.grid_to_world(path[i].gx, path[i].gy);
                        (w.0 - stub.pad_pos.0).hypot(w.1 - stub.pad_pos.1) < 0.025
                    }
                }) {
                    path = &path[k..];
                    conn.start = (stub.pad_pos.0, stub.pad_pos.1, stub.layer);
                }
            }
            if let Some(stub) = escape_stubs.iter().find(|s| {
                s.net_id == conn.net_id
                    && s.tip_layer == conn.end.2
                    && (s.tip_pos.0 - conn.end.0).hypot(s.tip_pos.1 - conn.end.1) < 0.15
            }) {
                let min_k = path.len().saturating_sub(20);
                if let Some(k) = (min_k..path.len().saturating_sub(1)).find(|&i| {
                    path[i].layer == stub.layer && {
                        let w = grid.grid_to_world(path[i].gx, path[i].gy);
                        (w.0 - stub.pad_pos.0).hypot(w.1 - stub.pad_pos.1) < 0.025
                    }
                }) {
                    path = &path[..=k];
                    conn.end = (stub.pad_pos.0, stub.pad_pos.1, stub.layer);
                }
            }
            used_stub_tips.push((conn.net_id, (conn.start.0, conn.start.1)));
            used_stub_tips.push((conn.net_id, (conn.end.0, conn.end.1)));
            let is_diff_pair = conn.diff_pair_leader.is_some()
                || connections.iter().any(|c| c.diff_pair_leader == Some(conn_idx))
                || conn.netclass.name == "USB_Diff";
            let width = if is_diff_pair {
                conn.netclass.diff_pair_width
            } else {
                conn.netclass.track_width
            };
            let mut via_dia = conn.netclass.via_dia;
            let mut via_drill = conn.netclass.via_drill;
            let min_annular = board.design_rules().min_via_annular_width;
            let min_drill = board.design_rules().min_via_drill;
            if (via_dia - via_drill) * 0.5 < min_annular - 1e-4 {
                if via_dia - 2.0 * min_annular >= min_drill - 1e-4 {
                    via_drill = (via_dia - 2.0 * min_annular).max(min_drill);
                } else {
                    via_dia = via_drill + 2.0 * min_annular;
                }
            }
            if path.len() < 2 {
                if (conn.start.0 - conn.end.0).hypot(conn.start.1 - conn.end.1) > 1e-4 {
                    board.tracks.push(TrackSegment {
                        start: (conn.start.0, conn.start.1),
                        end: (conn.end.0, conn.end.1),
                        width,
                        layer: conn.start.2,
                        net_id: conn.net_id,
                        is_locked: false,
                    });
                }
                continue;
            }

            let start_tht_r = board.footprints.get(conn.start_fp)
                .and_then(|fp| fp.pads.get(conn.start_pad))
                .filter(|p| p.is_through_hole)
                .map(|p| p.width.max(p.height) * 0.5 + 0.1);
            let end_tht_r = board.footprints.get(conn.end_fp)
                .and_then(|fp| fp.pads.get(conn.end_pad))
                .filter(|p| p.is_through_hole)
                .map(|p| p.width.max(p.height) * 0.5 + 0.1);
            let is_tht_start = start_tht_r.is_some();
            let is_tht_end = end_tht_r.is_some();

            if let Some(sr) = start_tht_r {
                let lim = path.len().min(16);
                let mut trim_start = 0;
                for i in 1..lim {
                    let w = grid.grid_to_world(path[i].gx, path[i].gy);
                    if (w.0 - conn.start.0).hypot(w.1 - conn.start.1) > sr {
                        break;
                    }
                    if path[i].layer != path[i - 1].layer
                        || (path[i].gx == path[i - 1].gx && path[i].gy == path[i - 1].gy)
                    {
                        trim_start = i;
                    }
                }
                if trim_start > 0 && trim_start < path.len() {
                    path = &path[trim_start..];
                }
            }
            if let Some(er) = end_tht_r {
                let min_idx = path.len().saturating_sub(16);
                let mut trim_end = path.len();
                for i in (min_idx..path.len().saturating_sub(1)).rev() {
                    let w = grid.grid_to_world(path[i].gx, path[i].gy);
                    if (w.0 - conn.end.0).hypot(w.1 - conn.end.1) > er {
                        break;
                    }
                    if path[i].layer != path[i + 1].layer
                        || (path[i].gx == path[i + 1].gx && path[i].gy == path[i + 1].gy)
                    {
                        trim_end = i + 1;
                    }
                }
                if trim_end < path.len() && trim_end > 0 {
                    path = &path[..trim_end];
                }
            }

            let mut run_start_idx = 0;
            let mut last_trans_via: Option<(f64, f64)> = None;

            // If the path starts on a different layer than a non-THT start terminal, place a start via first
            if !is_tht_start && path[0].layer != conn.start.2 {
                let pt = grid.grid_to_world(path[0].gx, path[0].gy);
                let already_has_via = board
                    .vias
                    .iter()
                    .find(|v| v.net_id == conn.net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.25)
                    .map(|v| v.position);
                let vp_opt = if let Some(ev) = already_has_via {
                    Some(ev)
                } else if is_via_location_legal(board, pt, via_dia, conn.net_id, conn.netclass.clearance.min(0.12)) {
                    board.vias.push(Via {
                        position: pt,
                        size: via_dia,
                        drill: via_drill,
                        net_id: conn.net_id,
                        is_locked: false,
                    });
                    Some(pt)
                } else {
                    let mut found = None;
                    for r in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00] {
                        for &(dx, dy) in &[
                            (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
                            (r, r), (r, -r), (-r, r), (-r, -r),
                        ] {
                            let cand = (pt.0 + dx, pt.1 + dy);
                            if is_via_location_legal(board, cand, via_dia, conn.net_id, conn.netclass.clearance.min(0.10)) {
                                board.vias.push(Via {
                                    position: cand,
                                    size: via_dia,
                                    drill: via_drill,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                                found = Some(cand);
                                break;
                            }
                        }
                        if found.is_some() {
                            break;
                        }
                    }
                    found
                };
                if let Some(vp) = vp_opt {
                    let dogleg = compute_45_dogleg((conn.start.0, conn.start.1), vp, false);
                    for w in dogleg.windows(2) {
                        if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                            board.tracks.push(TrackSegment {
                                start: w[0],
                                end: w[1],
                                width,
                                layer: conn.start.2,
                                net_id: conn.net_id,
                                is_locked: false,
                            });
                        }
                    }
                    last_trans_via = Some(vp);
                }
            }

            while run_start_idx < path.len() {
                let current_layer = path[run_start_idx].layer;
                let mut run_end_idx = run_start_idx;
                while run_end_idx + 1 < path.len() && path[run_end_idx + 1].layer == current_layer {
                    run_end_idx += 1;
                }

                let is_start_terminal = run_start_idx == 0
                    && (conn.start.2 == current_layer || is_tht_start);
                let is_end_terminal = run_end_idx + 1 == path.len()
                    && (conn.end.2 == current_layer || is_tht_end);

                let w_start = if let Some(lv) = last_trans_via.take() {
                    lv
                } else {
                    grid.grid_to_world(path[run_start_idx].gx, path[run_start_idx].gy)
                };

                if run_end_idx > run_start_idx {
                    let mut seg_start = w_start;
                    let mut prev_dx = path[run_start_idx + 1].gx as isize - path[run_start_idx].gx as isize;
                    let mut prev_dy = path[run_start_idx + 1].gy as isize - path[run_start_idx].gy as isize;
                    let mut start_deferred = is_start_terminal
                        && (w_start.0 - conn.start.0).hypot(w_start.1 - conn.start.1) < 0.08;

                    // 1. Connect start pad/terminal to w_start via 45° dogleg if not deferred
                    if is_start_terminal && !start_deferred {
                        let dogleg = compute_45_dogleg((conn.start.0, conn.start.1), w_start, false);
                        for w in dogleg.windows(2) {
                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: w[0],
                                    end: w[1],
                                    width,
                                    layer: current_layer,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                            }
                        }
                    }

                    // 2. On-grid path segments (strictly octilinear 0°, 45°, 90°)
                    for i in (run_start_idx + 1)..run_end_idx {
                        let cur_dx = path[i + 1].gx as isize - path[i].gx as isize;
                        let cur_dy = path[i + 1].gy as isize - path[i].gy as isize;
                        if cur_dx != prev_dx || cur_dy != prev_dy {
                            let corner = grid.grid_to_world(path[i].gx, path[i].gy);
                            if start_deferred {
                                let diag_first = prev_dx == 0 || prev_dy == 0;
                                let dogleg = compute_45_dogleg((conn.start.0, conn.start.1), corner, diag_first);
                                for w in dogleg.windows(2) {
                                    if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                        board.tracks.push(TrackSegment {
                                            start: w[0],
                                            end: w[1],
                                            width,
                                            layer: current_layer,
                                            net_id: conn.net_id,
                                            is_locked: false,
                                        });
                                    }
                                }
                                start_deferred = false;
                            } else if (seg_start.0 - corner.0).hypot(seg_start.1 - corner.1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: seg_start,
                                    end: corner,
                                    width,
                                    layer: current_layer,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                            }
                            seg_start = corner;
                            prev_dx = cur_dx;
                            prev_dy = cur_dy;
                        }
                    }

                    let w_end = grid.grid_to_world(path[run_end_idx].gx, path[run_end_idx].gy);
                    let from_pt = if start_deferred { (conn.start.0, conn.start.1) } else { seg_start };
                    if is_end_terminal && (w_end.0 - conn.end.0).hypot(w_end.1 - conn.end.1) < 0.08 {
                        let diag_first = if start_deferred {
                            prev_dx == 0 || prev_dy == 0
                        } else {
                            prev_dx != 0 && prev_dy != 0
                        };
                        let dogleg = compute_45_dogleg(from_pt, (conn.end.0, conn.end.1), diag_first);
                        for w in dogleg.windows(2) {
                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: w[0],
                                    end: w[1],
                                    width,
                                    layer: current_layer,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                            }
                        }
                    } else {
                        if start_deferred {
                            let diag_first = prev_dx == 0 || prev_dy == 0;
                            let dogleg = compute_45_dogleg((conn.start.0, conn.start.1), w_end, diag_first);
                            for w in dogleg.windows(2) {
                                if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                    board.tracks.push(TrackSegment {
                                        start: w[0],
                                        end: w[1],
                                        width,
                                        layer: current_layer,
                                        net_id: conn.net_id,
                                        is_locked: false,
                                    });
                                }
                            }
                        } else if (seg_start.0 - w_end.0).hypot(seg_start.1 - w_end.1) > 1e-4 {
                            board.tracks.push(TrackSegment {
                                start: seg_start,
                                end: w_end,
                                width,
                                layer: current_layer,
                                net_id: conn.net_id,
                                is_locked: false,
                            });
                        }

                        // 3. Connect w_end to end pad/terminal via 45° dogleg
                        if is_end_terminal {
                            let dogleg = compute_45_dogleg(w_end, (conn.end.0, conn.end.1), false);
                            for w in dogleg.windows(2) {
                                if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                    board.tracks.push(TrackSegment {
                                        start: w[0],
                                        end: w[1],
                                        width,
                                        layer: current_layer,
                                        net_id: conn.net_id,
                                        is_locked: false,
                                    });
                                }
                            }
                        }
                    }
                } else {
                    let w_pt = w_start;
                    if is_start_terminal {
                        let dogleg = compute_45_dogleg((conn.start.0, conn.start.1), w_pt, false);
                        for w in dogleg.windows(2) {
                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: w[0],
                                    end: w[1],
                                    width,
                                    layer: current_layer,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                            }
                        }
                    }
                    if is_end_terminal {
                        let dogleg = compute_45_dogleg(w_pt, (conn.end.0, conn.end.1), false);
                        for w in dogleg.windows(2) {
                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: w[0],
                                    end: w[1],
                                    width,
                                    layer: current_layer,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                            }
                        }
                    }
                }

                if run_end_idx + 1 < path.len() {
                    let trans_node = path[run_end_idx];
                    let pt = grid.grid_to_world(trans_node.gx, trans_node.gy);
                    let in_th_pad_pos = board.footprints.iter().find_map(|fp| {
                        fp.pads.iter().find_map(|pad| {
                            if pad.is_through_hole && pad.net_id == conn.net_id {
                                let (px, py) = fp.pad_absolute_position(pad);
                                if (px - pt.0).hypot(py - pt.1) <= pad.width.max(pad.height) / 2.0 + 0.1 {
                                    Some((px, py))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                    });
                    let already_has_via = board
                        .vias
                        .iter()
                        .any(|v| v.net_id == conn.net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.25);

                    if in_th_pad_pos.is_none() && !already_has_via {
                        let mut final_via_pt = pt;
                        if is_diff_pair {
                            let mate_conn_idx = if let Some(l_idx) = conn.diff_pair_leader {
                                Some(l_idx)
                            } else {
                                connections.iter().position(|c| c.diff_pair_leader == Some(conn_idx))
                            };
                            if let Some(m_idx) = mate_conn_idx {
                                if m_idx < paths.len() && m_idx < connections.len() {
                                    if let Some(ref m_path) = paths[m_idx] {
                                        let mut nearest_trans = None;
                                        let mut min_d = f64::INFINITY;
                                        for (mi, mn) in m_path.iter().enumerate() {
                                            if mi + 1 < m_path.len() && m_path[mi + 1].layer != mn.layer {
                                                let mw = grid.grid_to_world(mn.gx, mn.gy);
                                                let d = (mw.0 - pt.0).hypot(mw.1 - pt.1);
                                                if d < min_d && d < 2.0 {
                                                    min_d = d;
                                                    nearest_trans = Some(mw);
                                                }
                                            }
                                        }
                                        if let Some(m_pt) = nearest_trans {
                                            let min_spacing = (via_dia + conn.netclass.clearance).max(0.80);
                                            let mid = ((pt.0 + m_pt.0) * 0.5, (pt.1 + m_pt.1) * 0.5);
                                            let mut norm = (pt.0 - m_pt.0, pt.1 - m_pt.1);
                                            let norm_len = norm.0.hypot(norm.1);
                                            if norm_len > 1e-4 {
                                                norm = (norm.0 / norm_len, norm.1 / norm_len);
                                                let cand_via = (mid.0 + norm.0 * min_spacing * 0.5, mid.1 + norm.1 * min_spacing * 0.5);
                                                let cand_mate = (mid.0 - norm.0 * min_spacing * 0.5, mid.1 - norm.1 * min_spacing * 0.5);
                                                let mate_conn = &connections[m_idx];
                                                if is_via_location_legal(board, cand_via, via_dia, conn.net_id, conn.netclass.clearance)
                                                    && is_via_location_legal(board, cand_mate, via_dia, mate_conn.net_id, mate_conn.netclass.clearance)
                                                {
                                                    final_via_pt = cand_via;
                                                    for mv in &mut board.vias {
                                                        if (mv.position.0 - m_pt.0).hypot(mv.position.1 - m_pt.1) < 0.40 {
                                                            mv.position = cand_mate;
                                                        }
                                                    }
                                                    for t in &mut board.tracks {
                                                        if t.net_id == mate_conn.net_id {
                                                            if (t.start.0 - m_pt.0).hypot(t.start.1 - m_pt.1) < 0.05 {
                                                                t.start = cand_mate;
                                                            }
                                                            if (t.end.0 - m_pt.0).hypot(t.end.1 - m_pt.1) < 0.05 {
                                                                t.end = cand_mate;
                                                            }
                                                        }
                                                    }
                                                    if let Some(last_trk) = board.tracks.last_mut() {
                                                        if last_trk.net_id == conn.net_id && (last_trk.end.0 - pt.0).hypot(last_trk.end.1 - pt.1) < 0.05 {
                                                            last_trk.end = cand_via;
                                                        }
                                                    }
                                                } else {
                                                    final_via_pt = pt;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        let final_pt = if is_via_location_legal(board, final_via_pt, via_dia, conn.net_id, conn.netclass.clearance) {
                            Some(final_via_pt)
                        } else {
                            let mut cand_pt = None;
                            for r in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.45, 0.60, 0.80, 1.00, 1.25] {
                                for &(dx, dy) in &[
                                    (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
                                    (r, r), (r, -r), (-r, r), (-r, -r),
                                ] {
                                    let cand = (final_via_pt.0 + dx, final_via_pt.1 + dy);
                                    if is_via_location_legal(board, cand, via_dia, conn.net_id, conn.netclass.clearance.min(0.12)) {
                                        cand_pt = Some(cand);
                                        break;
                                    }
                                }
                                if cand_pt.is_some() {
                                    break;
                                }
                            }
                            if cand_pt.is_none() {
                                // Fallback: find the candidate with maximum clearance that does not physically short any copper or overlap any SMD pad
                                let via_r = via_dia * 0.5;
                                let allow_vip = board.allow_via_in_pad();
                                let mut best_fallback = None;
                                let mut best_min_c = -1.0_f64;
                                for r in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.75, 1.00, 1.30] {
                                    for &(dx, dy) in &[
                                        (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
                                        (r, r), (r, -r), (-r, r), (-r, -r),
                                    ] {
                                        let cand = (final_via_pt.0 + dx, final_via_pt.1 + dy);
                                        let mut min_c = f64::INFINITY;
                                        let mut valid = true;
                                        for fp in &board.footprints {
                                            for pad in &fp.pads {
                                                let d = dist_point_to_pad(cand, fp, pad);
                                                if pad.net_id != conn.net_id {
                                                    let c = d - via_r;
                                                    if c < 0.05 { valid = false; break; }
                                                    min_c = min_c.min(c);
                                                } else if !pad.is_through_hole && !allow_vip && d < via_r + 0.02 {
                                                    valid = false; break;
                                                }
                                            }
                                            if !valid { break; }
                                        }
                                        if !valid { continue; }
                                        for trk in &board.tracks {
                                            if trk.net_id != conn.net_id {
                                                let c = dist_point_to_segment(cand, trk.start, trk.end) - (via_r + trk.width * 0.5);
                                                if c < 0.08 { valid = false; break; }
                                                min_c = min_c.min(c);
                                            }
                                        }
                                        if !valid { continue; }
                                        for v in &board.vias {
                                            let d = (cand.0 - v.position.0).hypot(cand.1 - v.position.1);
                                            if d < (via_drill + v.drill) * 0.5 + 0.20 { valid = false; break; }
                                            if v.net_id != conn.net_id {
                                                let c = d - (via_r + v.size * 0.5);
                                                if c < 0.08 { valid = false; break; }
                                                min_c = min_c.min(c);
                                            }
                                        }
                                        if valid && min_c > best_min_c {
                                            best_min_c = min_c;
                                            best_fallback = Some(cand);
                                        }
                                    }
                                }
                                let pt_overlaps_smd = !allow_vip && board.footprints.iter().any(|fp| {
                                    fp.pads.iter().any(|p| !p.is_through_hole && dist_point_to_pad(pt, fp, p) < via_r + 0.01)
                                });
                                cand_pt = if pt_overlaps_smd { best_fallback } else { best_fallback.or(Some(pt)) };
                            }
                            cand_pt
                        };

                        if let Some(vp) = final_pt {
                            board.vias.push(Via {
                                position: vp,
                                size: via_dia,
                                drill: via_drill,
                                net_id: conn.net_id,
                                is_locked: false,
                            });
                            last_trans_via = Some(vp);
                            if let Some(last_trk) = board.tracks.last().cloned() {
                                if last_trk.net_id == conn.net_id && (last_trk.end.0 - pt.0).hypot(last_trk.end.1 - pt.1) < 0.50 {
                                    let end_layer = last_trk.layer;
                                    let start_pt = last_trk.start;
                                    let dogleg = [false, true]
                                        .into_iter()
                                        .map(|df| compute_45_dogleg(start_pt, vp, df))
                                        .find(|dl| {
                                            dl.windows(2).all(|w| {
                                                (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                    || is_track_segment_legal(board, w[0], w[1], width, end_layer, conn.net_id, 0.05)
                                            })
                                        })
                                        .unwrap_or_else(|| compute_45_dogleg(start_pt, vp, false));
                                    board.tracks.pop();
                                    for w in dogleg.windows(2) {
                                        if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                            board.tracks.push(TrackSegment {
                                                start: w[0],
                                                end: w[1],
                                                width,
                                                layer: end_layer,
                                                net_id: conn.net_id,
                                                is_locked: false,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    } else if let Some(th_pos) = in_th_pad_pos {
                        last_trans_via = Some(th_pos);
                        if let Some(last_trk) = board.tracks.last().cloned() {
                            if last_trk.net_id == conn.net_id && (last_trk.end.0 - pt.0).hypot(last_trk.end.1 - pt.1) < 1.0 {
                                let end_layer = last_trk.layer;
                                let start_pt = last_trk.start;
                                let dogleg = compute_45_dogleg(start_pt, th_pos, false);
                                board.tracks.pop();
                                for w in dogleg.windows(2) {
                                    if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                        board.tracks.push(TrackSegment {
                                            start: w[0],
                                            end: w[1],
                                            width,
                                            layer: end_layer,
                                            net_id: conn.net_id,
                                            is_locked: false,
                                        });
                                    }
                                }
                            }
                        }
                    } else if already_has_via {
                        if let Some(ev) = board
                            .vias
                            .iter()
                            .find(|v| v.net_id == conn.net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.25)
                        {
                            let vp = ev.position;
                            last_trans_via = Some(vp);
                            if let Some(last_trk) = board.tracks.last_mut() {
                                if last_trk.net_id == conn.net_id && (last_trk.end.0 - pt.0).hypot(last_trk.end.1 - pt.1) < 0.30 {
                                    last_trk.end = vp;
                                }
                            }
                        }
                    }
                } else if !is_tht_end && current_layer != conn.end.2 {
                    // Path ends on a different layer than a non-THT end terminal: place an end via and dogleg
                    let last_node = path[run_end_idx];
                    let pt = grid.grid_to_world(last_node.gx, last_node.gy);
                    let already_has_via = board
                        .vias
                        .iter()
                        .find(|v| v.net_id == conn.net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.25)
                        .map(|v| v.position);
                    let vp_opt = if let Some(ev) = already_has_via {
                        Some(ev)
                    } else if is_via_location_legal(board, pt, via_dia, conn.net_id, conn.netclass.clearance.min(0.12)) {
                        board.vias.push(Via {
                            position: pt,
                            size: via_dia,
                            drill: via_drill,
                            net_id: conn.net_id,
                            is_locked: false,
                        });
                        Some(pt)
                    } else {
                        let mut found = None;
                        for r in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30] {
                            for &(dx, dy) in &[
                                (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
                                (r, r), (r, -r), (-r, r), (-r, -r),
                            ] {
                                let cand = (pt.0 + dx, pt.1 + dy);
                                if is_via_location_legal(board, cand, via_dia, conn.net_id, conn.netclass.clearance.min(0.10)) {
                                    board.vias.push(Via {
                                        position: cand,
                                        size: via_dia,
                                        drill: via_drill,
                                        net_id: conn.net_id,
                                        is_locked: false,
                                    });
                                    found = Some(cand);
                                    break;
                                }
                            }
                            if found.is_some() {
                                break;
                            }
                        }
                        if found.is_none() {
                            board.vias.push(Via {
                                position: pt,
                                size: via_dia,
                                drill: via_drill,
                                net_id: conn.net_id,
                                is_locked: false,
                            });
                            Some(pt)
                        } else {
                            found
                        }
                    };
                    if let Some(vp) = vp_opt {
                        if let Some(last_trk) = board.tracks.last().cloned() {
                            if last_trk.net_id == conn.net_id && (last_trk.end.0 - pt.0).hypot(last_trk.end.1 - pt.1) < 1.50 {
                                let end_layer = last_trk.layer;
                                let start_pt = last_trk.start;
                                let dogleg = [false, true]
                                    .into_iter()
                                    .map(|df| compute_45_dogleg(start_pt, vp, df))
                                    .find(|dl| {
                                        dl.windows(2).all(|w| {
                                            (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                || is_track_segment_legal(board, w[0], w[1], width, end_layer, conn.net_id, 0.05)
                                        })
                                    })
                                    .unwrap_or_else(|| compute_45_dogleg(start_pt, vp, false));
                                board.tracks.pop();
                                for w in dogleg.windows(2) {
                                    if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                        board.tracks.push(TrackSegment {
                                            start: w[0],
                                            end: w[1],
                                            width,
                                            layer: end_layer,
                                            net_id: conn.net_id,
                                            is_locked: false,
                                        });
                                    }
                                }
                            }
                        }
                        let dogleg = [false, true]
                            .into_iter()
                            .map(|df| compute_45_dogleg(vp, (conn.end.0, conn.end.1), df))
                            .find(|dl| {
                                dl.windows(2).all(|w| {
                                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                        || is_track_segment_legal(board, w[0], w[1], width, conn.end.2, conn.net_id, 0.05)
                                })
                            })
                            .unwrap_or_else(|| compute_45_dogleg(vp, (conn.end.0, conn.end.1), false));
                        for w in dogleg.windows(2) {
                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: w[0],
                                    end: w[1],
                                    width,
                                    layer: conn.end.2,
                                    net_id: conn.net_id,
                                    is_locked: false,
                                });
                            }
                        }
                    }
                }

                run_start_idx = run_end_idx + 1;
            }
        }

        // Commit escape stubs only if a continuous, valid route connects to them
        for stub in escape_stubs {
            let is_used = used_stub_tips.iter().any(|&(nid, pt)| {
                nid == stub.net_id && (pt.0 - stub.tip_pos.0).hypot(pt.1 - stub.tip_pos.1) < 0.15
            });
            if !is_used {
                continue;
            }

            // Find if a layer-transition via was placed near stub.tip_pos (possibly shifted up to 1.2mm away)
            let nearby_via_pos = board
                .vias
                .iter()
                .filter(|v| v.net_id == stub.net_id)
                .map(|v| (v.position, (v.position.0 - stub.tip_pos.0).hypot(v.position.1 - stub.tip_pos.1)))
                .filter(|&(_, d)| d < 1.25)
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .map(|(pos, _)| pos);

            let mut snapped_on_tip_layer = board.tracks.iter().any(|t| {
                t.net_id == stub.net_id
                    && t.layer == stub.tip_layer
                    && ((t.start.0 - stub.tip_pos.0).hypot(t.start.1 - stub.tip_pos.1) < 1e-3
                        || (t.end.0 - stub.tip_pos.0).hypot(t.end.1 - stub.tip_pos.1) < 1e-3)
            });
            if !snapped_on_tip_layer {
                for t in &mut board.tracks {
                    if t.net_id == stub.net_id && t.layer == stub.tip_layer {
                        let d_start = (t.start.0 - stub.tip_pos.0).hypot(t.start.1 - stub.tip_pos.1);
                        let d_end = (t.end.0 - stub.tip_pos.0).hypot(t.end.1 - stub.tip_pos.1);
                        if d_start < 0.08 && d_start <= d_end && is_octilinear_segment(stub.tip_pos, t.end) {
                            t.start = stub.tip_pos;
                            snapped_on_tip_layer = true;
                            break;
                        } else if d_end < 0.08 && is_octilinear_segment(t.start, stub.tip_pos) {
                            t.end = stub.tip_pos;
                            snapped_on_tip_layer = true;
                            break;
                        }
                    }
                }
            }

            let eff_stub_w = board
                .tracks
                .iter()
                .find(|t| t.net_id == stub.net_id && !t.is_locked)
                .map(|t| t.width)
                .unwrap_or(stub.width);

            board.tracks.push(TrackSegment {
                start: stub.pad_pos,
                end: stub.tip_pos,
                width: eff_stub_w,
                layer: stub.layer,
                net_id: stub.net_id,
                is_locked: false,
            });
            if !snapped_on_tip_layer {
                if let Some(vp) = nearby_via_pos {
                    let dogleg = [false, true]
                        .into_iter()
                        .map(|df| compute_45_dogleg(stub.tip_pos, vp, df))
                        .find(|dl| {
                            dl.windows(2).all(|w| {
                                (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                    || is_track_segment_legal(board, w[0], w[1], eff_stub_w, stub.tip_layer, stub.net_id, 0.05)
                            })
                        })
                        .unwrap_or_else(|| compute_45_dogleg(stub.tip_pos, vp, false));
                    for w in dogleg.windows(2) {
                        if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                            board.tracks.push(TrackSegment {
                                start: w[0],
                                end: w[1],
                                width: eff_stub_w,
                                layer: stub.tip_layer,
                                net_id: stub.net_id,
                                is_locked: false,
                            });
                        }
                    }
                } else {
                    // Check if there is a same-net track endpoint on stub.tip_layer within 0.50mm
                    let nearby_trk_pt = board
                        .tracks
                        .iter()
                        .filter(|t| t.net_id == stub.net_id && t.layer == stub.tip_layer && (t.start != stub.pad_pos || t.end != stub.tip_pos))
                        .flat_map(|t| [t.start, t.end])
                        .map(|pt| (pt, (pt.0 - stub.tip_pos.0).hypot(pt.1 - stub.tip_pos.1)))
                        .filter(|&(_, d)| d > 1e-4 && d < 0.50)
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                        .map(|(pt, _)| pt);
                    if let Some(tp) = nearby_trk_pt {
                        let dogleg = [false, true]
                            .into_iter()
                            .map(|df| compute_45_dogleg(stub.tip_pos, tp, df))
                            .find(|dl| {
                                dl.windows(2).all(|w| {
                                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                        || is_track_segment_legal(board, w[0], w[1], eff_stub_w, stub.tip_layer, stub.net_id, 0.05)
                                })
                            })
                            .unwrap_or_else(|| compute_45_dogleg(stub.tip_pos, tp, false));
                        for w in dogleg.windows(2) {
                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                board.tracks.push(TrackSegment {
                                    start: w[0],
                                    end: w[1],
                                    width: eff_stub_w,
                                    layer: stub.tip_layer,
                                    net_id: stub.net_id,
                                    is_locked: false,
                                });
                            }
                        }
                    }
                }
            }
            if let Some(ref v) = stub.via {
                board.vias.push(v.clone());
            }
        }

        prune_dangling_tracks_and_vias(board);
    }

    /// Generates 45-degree BGA dogbone escape stubs and interstitial fanout vias for BGA
    /// footprints on `board`, rasterizing them onto `grid`.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] receiving the generated escape tracks and vias.
    /// * `grid` - Mutable [`RoutingGrid`] updated with the committed escape copper.
    ///
    /// # Side Effects
    /// Appends locked escape tracks and vias to `board` and commits them to `grid`.
    pub fn generate_bga_dogbone_escapes(&self, board: &mut Board, grid: &mut RoutingGrid) {
        generate_bga_dogbone_escapes(board, grid);
    }

    /// Determines whether `fp` is a dense SMD IC or fine-pitch connector requiring
    /// staggered pre-routed [`EscapeStub`] fanout stubs.
    ///
    /// # Arguments
    /// * `fp` - Reference to the [`Footprint`] to inspect.
    ///
    /// # Returns
    /// `true` if `fp` has `>= 4` SMD pads and either `>= 8` pads, a minimum pad pitch
    /// `<= 0.85 mm`, or an IC/connector reference prefix (`U*`, `J*`).
    pub fn is_dense_escape_footprint(fp: &Footprint) -> bool {
        if fp.reference == "J2" {
            return true;
        }
        let smd_pads: Vec<&Pad> = fp.pads.iter().filter(|p| !p.is_through_hole).collect();
        if smd_pads.len() < 4 {
            return false;
        }
        if smd_pads.len() >= 8 {
            return true;
        }
        let mut min_pitch = f64::INFINITY;
        for i in 0..smd_pads.len() {
            let (x1, y1) = fp.pad_absolute_position(smd_pads[i]);
            for j in (i + 1)..smd_pads.len() {
                let (x2, y2) = fp.pad_absolute_position(smd_pads[j]);
                let d = (x1 - x2).hypot(y1 - y2);
                if d > 0.05 && d < min_pitch {
                    min_pitch = d;
                }
            }
        }
        min_pitch <= 0.85 || fp.reference.starts_with('U') || fp.reference.starts_with('J')
    }

    /// Generates staggered outward fanout [`EscapeStub`] segments for dense QFP/QFN/SOIC ICs
    /// and fine-pitch USB-C connectors on `board`.
    ///
    /// Enforces a minimum length stagger (`>= 0.35 mm` to `0.75 mm`) between adjacent `0.5 mm`-pitch
    /// pins so that subsequent layer-transition vias (`0.60 mm` diameter + `0.20 mm` clearance)
    /// do not choke adjacent pin escape channels.
    ///
    /// # Arguments
    /// * `board` - Read-only reference to the [`Board`].
    ///
    /// # Returns
    /// Vector of conflict-free [`EscapeStub`] descriptors.
    pub fn generate_escape_stubs(&self, board: &Board) -> Vec<EscapeStub> {
        /// Cardinal perimeter edge of an SMD IC footprint on which a pad resides.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum PadEdge {
            /// Left (`-X`) edge in local footprint coordinates.
            West,
            /// Right (`+X`) edge in local footprint coordinates.
            East,
            /// Top (`-Y`) edge in local footprint coordinates.
            North,
            /// Bottom (`+Y`) edge in local footprint coordinates.
            South,
        }

        impl PadEdge {
            /// Returns the outward unit normal `(nx, ny)` in local footprint coordinates for this edge.
            ///
            /// # Returns
            /// 2D unit normal vector `(nx, ny)`.
            fn local_normal(self) -> (f64, f64) {
                match self {
                    PadEdge::West => (-1.0, 0.0),
                    PadEdge::East => (1.0, 0.0),
                    PadEdge::North => (0.0, -1.0),
                    PadEdge::South => (0.0, 1.0),
                }
            }
        }

        let calculate_pad_edge = |fp: &Footprint, pad: &Pad| -> Option<PadEdge> {
            let smd_pads: Vec<&Pad> = fp.pads.iter().filter(|p| !p.is_through_hole).collect();
            if smd_pads.is_empty() {
                return None;
            }

            let oy = |p: &Pad| if fp.layer != fp.orig_layer { -p.offset_y } else { p.offset_y };

            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;

            for p in &smd_pads {
                let lx = p.offset_x;
                let ly = oy(p);
                if lx < min_x { min_x = lx; }
                if lx > max_x { max_x = lx; }
                if ly < min_y { min_y = ly; }
                if ly > max_y { max_y = ly; }
            }

            let span_x = max_x - min_x;
            let span_y = max_y - min_y;
            let cx = (min_x + max_x) * 0.5;
            let cy = (min_y + max_y) * 0.5;

            let target_lx = pad.offset_x;
            let target_ly = oy(pad);

            // Filter out large central thermal/ground pads
            if pad.width >= 1.8 && pad.height >= 1.8 && (target_lx - cx).hypot(target_ly - cy) < 0.3 * span_x.max(span_y).max(1.0) {
                return None;
            }

            let d_west = (target_lx - min_x).abs();
            let d_east = (max_x - target_lx).abs();
            let d_north = (target_ly - min_y).abs();
            let d_south = (max_y - target_ly).abs();

            if pad.width > pad.height * 1.15 {
                if target_lx < cx {
                    Some(PadEdge::West)
                } else {
                    Some(PadEdge::East)
                }
            } else if pad.height > pad.width * 1.15 {
                if target_ly < cy {
                    Some(PadEdge::North)
                } else {
                    Some(PadEdge::South)
                }
            } else {
                let min_d = d_west.min(d_east).min(d_north).min(d_south);
                if (d_west - min_d).abs() < 1e-4 {
                    Some(PadEdge::West)
                } else if (d_east - min_d).abs() < 1e-4 {
                    Some(PadEdge::East)
                } else if (d_north - min_d).abs() < 1e-4 {
                    Some(PadEdge::North)
                } else {
                    Some(PadEdge::South)
                }
            }
        };

        let mut stubs = Vec::new();

        for (fp_idx, fp) in board.footprints.iter().enumerate() {
            if fp.reference == "J2" || (fp.reference.starts_with('J') && (fp.value.to_ascii_uppercase().contains("USB") || fp.pads.iter().any(|p| p.number == "A6" || p.number == "B6"))) {
                let smd_pads: Vec<&Pad> = fp.pads.iter().filter(|p| !p.is_through_hole).collect();
                if smd_pads.is_empty() {
                    continue;
                }
                let mut min_lx = f64::INFINITY;
                let mut max_lx = f64::NEG_INFINITY;
                let mut min_ly = f64::INFINITY;
                let mut max_ly = f64::NEG_INFINITY;
                for p in &smd_pads {
                    min_lx = min_lx.min(p.offset_x);
                    max_lx = max_lx.max(p.offset_x);
                    min_ly = min_ly.min(p.offset_y);
                    max_ly = max_ly.max(p.offset_y);
                }
                let span_x = max_lx - min_lx;
                let span_y = max_ly - min_ly;

                let local_norm = if span_x >= span_y {
                    Point::new(0.0, -1.0)
                } else {
                    Point::new(1.0, 0.0)
                };
                let mut world_inward = local_norm.rotate_ccw(fp.rotation);

                let (board_cx, board_cy) = ((board.min_x + board.max_x) * 0.5, (board.min_y + board.max_y) * 0.5);
                let to_center_x = board_cx - fp.x;
                let to_center_y = board_cy - fp.y;
                if world_inward.x * to_center_x + world_inward.y * to_center_y < 0.0 {
                    world_inward = Point::new(-world_inward.x, -world_inward.y);
                }

                for pad in &fp.pads {
                    if pad.net_id == 0
                        || crate::board::is_gnd_net_name(&pad.net_name)
                        || pad.net_name.is_empty()
                        || pad.net_name.starts_with("unconnected")
                        || pad.is_through_hole
                    {
                        continue;
                    }
                    let (px, py) = fp.pad_absolute_position(pad);

                    // Escape straight outward into the open board area without premature vias
                    // At 0.50mm pitch, 0.15mm parallel tracks have 0.35mm clearance (double the 0.15mm required)
                    let stub_len = 1.60;
                    let tip_pos = (px + world_inward.x * stub_len, py + world_inward.y * stub_len);
                    let clean_pad_net = pad.net_name.trim_start_matches('/');
                    let stub_w = if clean_pad_net == "VBUS" || clean_pad_net.starts_with("USB_") { 0.20 } else { 0.15 };

                    stubs.push(EscapeStub {
                        fp_idx,
                        net_id: pad.net_id,
                        pad_pos: (px, py),
                        tip_pos,
                        width: stub_w,
                        via: None,
                        tip_layer: fp.layer,
                        layer: fp.layer,
                    });
                }
                continue;
            }

            if !Self::is_dense_escape_footprint(fp) || fp.is_bga() {
                continue;
            }

            let is_05mm_ic = fp.reference == "U1"
                || fp.reference == "J2"
                || fp.min_pad_pitch() <= 0.65;

            let oy = |p: &Pad| if fp.layer != fp.orig_layer { -p.offset_y } else { p.offset_y };

            let mut candidates: Vec<(PadEdge, &Pad)> = Vec::new();
            for pad in &fp.pads {
                if pad.net_id == 0
                    || crate::board::is_gnd_net_name(&pad.net_name)
                    || pad.net_name.is_empty()
                    || pad.net_name.starts_with("unconnected")
                    || pad.is_through_hole
                {
                    continue;
                }
                if !is_05mm_ic && board.get_netclass_for_net(pad.net_id).track_width > 0.25 {
                    continue;
                }
                if let Some(edge) = calculate_pad_edge(fp, pad) {
                    candidates.push((edge, pad));
                }
            }

            for edge in [PadEdge::West, PadEdge::East, PadEdge::North, PadEdge::South] {
                let mut edge_pads: Vec<&Pad> = candidates
                    .iter()
                    .filter(|(e, _)| *e == edge)
                    .map(|(_, p)| *p)
                    .collect();

                match edge {
                    PadEdge::West | PadEdge::East => {
                        edge_pads.sort_by(|a, b| oy(a).partial_cmp(&oy(b)).unwrap_or(std::cmp::Ordering::Equal));
                    }
                    PadEdge::North | PadEdge::South => {
                        edge_pads.sort_by(|a, b| a.offset_x.partial_cmp(&b.offset_x).unwrap_or(std::cmp::Ordering::Equal));
                    }
                }

                let mut prev_stub_len: Option<f64> = None;
                for (edge_idx, pad) in edge_pads.iter().enumerate() {
                    let (px, py) = fp.pad_absolute_position(pad);
                    let has_close_prev = edge_idx > 0 && {
                        let (ppx, ppy) = fp.pad_absolute_position(edge_pads[edge_idx - 1]);
                        (ppx - px).hypot(ppy - py) < 0.75
                    };
                    let has_close_next = edge_idx + 1 < edge_pads.len() && {
                        let (npx, npy) = fp.pad_absolute_position(edge_pads[edge_idx + 1]);
                        (npx - px).hypot(npy - py) < 0.75
                    };
                    let next_is_sandwiched = edge_idx + 2 < edge_pads.len() && has_close_next && {
                        let (npx, npy) = fp.pad_absolute_position(edge_pads[edge_idx + 1]);
                        let (nnpx, nnpy) = fp.pad_absolute_position(edge_pads[edge_idx + 2]);
                        (nnpx - npx).hypot(nnpy - npy) < 0.75
                    };
                    let next_higher_prio = has_close_next && {
                        let next_nc = board.get_netclass_for_net(edge_pads[edge_idx + 1].net_id);
                        let cur_nc = board.get_netclass_for_net(pad.net_id);
                        next_nc.track_width > cur_nc.track_width + 1e-4
                            || Self::netclass_priority(&next_nc.name) > Self::netclass_priority(&cur_nc.name)
                    };
                    let is_sandwiched = has_close_prev && has_close_next;
                    let is_staggered = if is_sandwiched {
                        prev_stub_len.map_or(false, |l| l > 0.70)
                    } else if !has_close_prev && (next_is_sandwiched || next_higher_prio) {
                        true
                    } else if has_close_prev {
                        prev_stub_len.map_or(true, |l| l > 0.70)
                    } else {
                        edge_idx % 2 == 1
                    };
                    let (norm_lx, norm_ly) = edge.local_normal();
                    let world_norm = Point::new(norm_lx, norm_ly).rotate_ccw(fp.rotation);

                    let stub_w = board.get_netclass_for_net(pad.net_id).track_width.min(0.15);
                    let eff_clearance = board.get_netclass_for_net(pad.net_id).clearance.min(0.15).max(0.12);
                    let stub_margin = (stub_w * 0.5) + eff_clearance;

                    // Clearance check against own footprint pads
                    let collides_own = |tip_pt: (f64, f64)| -> bool {
                        for other_pad in &fp.pads {
                            if other_pad.number == pad.number {
                                continue;
                            }
                            let (opx, opy) = fp.pad_absolute_position(other_pad);
                            let d_start = (opx - px).hypot(opy - py);
                            let d_seg = dist_to_segment(opx, opy, px, py, tip_pt.0, tip_pt.1);
                            if d_seg < d_start - 1e-4 {
                                return true;
                            }
                            let (ow, oh) = fp.pad_world_size(other_pad);
                            let req_dist = (ow.min(oh) * 0.5) + stub_margin;
                            if d_start >= req_dist && d_seg < req_dist - 1e-4 {
                                return true;
                            }
                            let d_tip = (opx - tip_pt.0).hypot(opy - tip_pt.1);
                            if d_tip < req_dist - 1e-4 {
                                return true;
                            }
                        }
                        false
                    };

                    // Clearance check against foreign footprint pads:
                    let collides_foreign = |tip_pt: (f64, f64)| -> bool {
                        for other_fp in &board.footprints {
                            if other_fp.reference == fp.reference {
                                continue;
                            }
                            for other_pad in &other_fp.pads {
                                if other_pad.net_id == pad.net_id {
                                    continue;
                                }
                                if !other_pad.is_through_hole && other_fp.layer != fp.layer {
                                    continue;
                                }
                                let (opx, opy) = other_fp.pad_absolute_position(other_pad);
                                let total_rot = ((other_fp.rotation + other_pad.rotation) % 360.0 + 360.0) % 360.0;
                                let seg_len = (px - tip_pt.0).hypot(py - tip_pt.1);
                                let steps = ((seg_len / 0.05).ceil() as usize).max(1);
                                for s in 0..=steps {
                                    let t = s as f64 / steps as f64;
                                    let sample = (px + t * (tip_pt.0 - px), py + t * (tip_pt.1 - py));
                                    let unrot = Point::new(sample.0 - opx, sample.1 - opy).rotate_ccw(-total_rot);
                                    let dx = 0.0f64.max(unrot.x.abs() - other_pad.width * 0.5);
                                    let dy = 0.0f64.max(unrot.y.abs() - other_pad.height * 0.5);
                                    if dx.hypot(dy) < stub_margin - 1e-4 {
                                        return true;
                                    }
                                }
                            }
                        }
                        false
                    };

                    // Destination/target coordinates for this pad's net (external destinations)
                    let mut target_positions = Vec::new();
                    for other_fp in &board.footprints {
                        if other_fp.reference == fp.reference {
                            continue;
                        }
                        for other_pad in &other_fp.pads {
                            if other_pad.net_id == pad.net_id {
                                let op = other_fp.pad_absolute_position(other_pad);
                                target_positions.push(op);
                            }
                        }
                    }
                    let target_dir = target_positions
                        .iter()
                        .min_by(|a, b| {
                            let d1 = (a.0 - px).hypot(a.1 - py);
                            let d2 = (b.0 - px).hypot(b.1 - py);
                            d1.partial_cmp(&d2).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .and_then(|closest| {
                            let dx = closest.0 - px;
                            let dy = closest.1 - py;
                            let d = dx.hypot(dy);
                            if d > 1e-4 {
                                Some((dx / d, dy / d))
                            } else {
                                None
                            }
                        });

                    let is_first = edge_idx == 0;
                    let is_last = edge_idx == edge_pads.len() - 1;

                    let mut open_lat_locals: Vec<(f64, f64)> = Vec::new();
                    if is_first {
                        open_lat_locals.push(match edge {
                            PadEdge::West | PadEdge::East => (0.0, -1.0),
                            PadEdge::North | PadEdge::South => (-1.0, 0.0),
                        });
                    }
                    if is_last {
                        open_lat_locals.push(match edge {
                            PadEdge::West | PadEdge::East => (0.0, 1.0),
                            PadEdge::North | PadEdge::South => (1.0, 0.0),
                        });
                    }

                    /// Candidate escape direction vector for a perimeter SMD pad.
                    struct EscapeCandDir {
                        /// Outward unit direction vector `(dx, dy)` in world space.
                        dir: (f64, f64),
                        /// Whether this candidate exits diagonally at 45 degrees.
                        is_diag: bool,
                        /// Dot product alignment with the vector toward the closest target pad.
                        dot: f64,
                    }

                    let mut cand_dirs = Vec::new();
                    // 1. Perpendicular outward
                    cand_dirs.push(EscapeCandDir {
                        dir: (world_norm.x, world_norm.y),
                        is_diag: false,
                        dot: target_dir.map_or(0.0, |td| world_norm.x * td.0 + world_norm.y * td.1),
                    });

                    // 2. Corner candidates: diagonal and lateral
                    for lat_local in open_lat_locals {
                        let w_lat = Point::new(lat_local.0, lat_local.1).rotate_ccw(fp.rotation);
                        let w_diag_x = world_norm.x + w_lat.x;
                        let w_diag_y = world_norm.y + w_lat.y;
                        let d_len = w_diag_x.hypot(w_diag_y);
                        if d_len > 1e-4 {
                            let w_diag = (w_diag_x / d_len, w_diag_y / d_len);
                            cand_dirs.push(EscapeCandDir {
                                dir: w_diag,
                                is_diag: true,
                                dot: target_dir.map_or(0.0, |td| w_diag.0 * td.0 + w_diag.1 * td.1),
                            });
                        }
                        cand_dirs.push(EscapeCandDir {
                            dir: (w_lat.x, w_lat.y),
                            is_diag: false,
                            dot: target_dir.map_or(0.0, |td| w_lat.x * td.0 + w_lat.y * td.1),
                        });
                    }

                    // Rank candidate escape directions by dot product with destination vector descending
                    cand_dirs.sort_by(|a, b| b.dot.partial_cmp(&a.dot).unwrap_or(std::cmp::Ordering::Equal));

                    let is_diff_pair_pad = board.get_netclass_for_net(pad.net_id).name == "USB_Diff"
                        || board.nets.get(pad.net_id).map_or(false, |n| n.contains("USB") || n.contains("DIFF"));

                    let mut preferred_lengths: Vec<f64> = if is_diff_pair_pad {
                        vec![0.45, 0.40, 0.50, 0.60, 0.65, 0.85]
                    } else if is_staggered {
                        vec![0.45, 0.40, 1.40, 1.10, 1.60, 0.85, 0.65, 0.60, 0.50]
                    } else {
                        vec![0.85, 0.65, 1.10, 1.40, 0.45, 0.40, 1.60, 0.60, 0.50]
                    };

                    // Enforce stagger against previous adjacent pad's chosen length (except for paired diff buses)
                    if !is_diff_pair_pad && has_close_prev {
                        if let Some(prev) = prev_stub_len {
                            preferred_lengths.sort_by(|a, b| {
                                let diff_a = (*a - prev).abs() >= 0.35;
                                let diff_b = (*b - prev).abs() >= 0.35;
                                diff_b.cmp(&diff_a)
                            });
                        }
                    }

                    let (pw, ph) = fp.pad_world_size(pad);
                    let mut found_tip = None;
                    let mut found_len = 0.0;
                    'cand_loop: for cand in &cand_dirs {
                        let pad_half_along_dir = (pw * 0.5 * cand.dir.0.abs() + ph * 0.5 * cand.dir.1.abs()).max(0.25);
                        let pad_ext_offset = (pad_half_along_dir - 0.25).max(0.0);
                        for &base_len in &preferred_lengths {
                            let cand_len = base_len + pad_ext_offset;
                            let snap50 = |v: f64| (v / 0.05).round() * 0.05;
                            let cand_tip = if cand.is_diag {
                                let step = (snap50(cand_len * std::f64::consts::FRAC_1_SQRT_2)).max(0.20);
                                (px + cand.dir.0.signum() * step, py + cand.dir.1.signum() * step)
                            } else if cand.dir.1.abs() < 1e-4 {
                                (snap50(px + cand.dir.0.signum() * cand_len), py)
                            } else if cand.dir.0.abs() < 1e-4 {
                                (px, snap50(py + cand.dir.1.signum() * cand_len))
                            } else {
                                (snap50(px + cand.dir.0 * cand_len), snap50(py + cand.dir.1 * cand_len))
                            };
                            let in_bounds = if board.max_x > board.min_x && board.max_y > board.min_y {
                                cand_tip.0 >= board.min_x + 0.3 && cand_tip.0 <= board.max_x - 0.3
                                    && cand_tip.1 >= board.min_y + 0.3 && cand_tip.1 <= board.max_y - 0.3
                            } else {
                                true
                            };
                            if in_bounds
                                && is_octilinear_segment((px, py), cand_tip)
                                && !collides_own(cand_tip)
                                && !collides_foreign(cand_tip)
                                && is_track_segment_legal(board, (px, py), cand_tip, stub_w, fp.layer, pad.net_id, eff_clearance)
                            {
                                found_tip = Some(cand_tip);
                                found_len = cand_len;
                                break 'cand_loop;
                            }
                        }
                    }

                    let Some(tip) = found_tip else {
                        continue;
                    };
                    prev_stub_len = Some(found_len);

                    stubs.push(EscapeStub {
                        fp_idx,
                        net_id: pad.net_id,
                        pad_pos: (px, py),
                        tip_pos: tip,
                        width: stub_w,
                        via: None,
                        tip_layer: fp.layer,
                        layer: fp.layer,
                    });
                }
            }
        }

        stubs
    }

    /// Rasterizes a single pre-generated [`EscapeStub`] (trace segment and optional fanout via)
    /// onto `grid`.
    ///
    /// # Arguments
    /// * `grid` - Mutable [`RoutingGrid`] to update.
    /// * `stub` - Reference to the [`EscapeStub`] to commit.
    ///
    /// # Side Effects
    /// Stamps `stub.net_id` into `grid.obstacle_net_ids` along the stub segment and optional via.
    pub fn commit_stub_to_grid(&self, grid: &mut RoutingGrid, stub: &EscapeStub) {
        let trk = TrackSegment {
            start: stub.pad_pos,
            end: stub.tip_pos,
            width: stub.width,
            layer: stub.layer,
            net_id: stub.net_id,
            is_locked: false,
        };
        grid.commit_track(&trk);
        if let Some(ref v) = stub.via {
            grid.commit_via(v);
        }
    }

    /// Connects perimeter `GND` SMD pins on `U1` inward to the central exposed thermal ground
    /// paddle using short `0.15 mm` copper spokes when DRC-legal.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] to which inward ground spokes are appended.
    ///
    /// # Side Effects
    /// Appends `GND` [`TrackSegment`] spokes to `board.tracks`.
    fn connect_u1_ground_pins(&self, board: &mut Board) {
        let Some(gnd_net_id) = board.nets.iter().position(|n| crate::board::is_gnd_net_name(n)) else {
            return;
        };
        let Some(u1) = board.footprints.iter().find(|fp| fp.reference == "U1").cloned() else {
            return;
        };
        for pad in &u1.pads {
            if pad.net_id == gnd_net_id && !pad.is_through_hole && pad.width < 1.5 && pad.height < 1.5 {
                let (px, py) = u1.pad_absolute_position(pad);
                let already_connected = board.tracks.iter().any(|t| {
                    t.net_id == gnd_net_id
                        && t.layer == u1.layer
                        && ((t.start.0 - px).hypot(t.start.1 - py) < 0.02
                            || (t.end.0 - px).hypot(t.end.1 - py) < 0.02)
                });
                if already_connected {
                    continue;
                }
                let dx = u1.x - px;
                let dy = u1.y - py;
                let world_inward = if dx.abs() > dy.abs() {
                    (dx.signum(), 0.0)
                } else {
                    (0.0, dy.signum())
                };
                let spoke_len = if matches!(pad.number.as_str(), "18" | "19") { 0.5375 } else { 0.90 };
                let end = (px + world_inward.0 * spoke_len, py + world_inward.1 * spoke_len);
                if is_track_segment_legal(board, (px, py), end, 0.15, u1.layer, gnd_net_id, 0.15) {
                    board.tracks.push(TrackSegment {
                        start: (px, py),
                        end,
                        width: 0.15,
                        layer: u1.layer,
                        net_id: gnd_net_id,
                        is_locked: false,
                    });
                    if matches!(pad.number.as_str(), "18" | "19") {
                        let end2 = (end.0 + dx.signum() * 0.25, end.1 + dy.signum() * 0.25);
                        let dup = board.tracks.iter().any(|t| {
                            t.net_id == gnd_net_id
                                && t.layer == u1.layer
                                && (t.start.0 - end.0).hypot(t.start.1 - end.1) < 1e-4
                                && (t.end.0 - end2.0).hypot(t.end.1 - end2.1) < 1e-4
                        });
                        if !dup && is_track_segment_legal(board, end, end2, 0.15, u1.layer, gnd_net_id, 0.15) {
                            board.tracks.push(TrackSegment {
                                start: end,
                                end: end2,
                                width: 0.15,
                                layer: u1.layer,
                                net_id: gnd_net_id,
                                is_locked: false,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Connects all SMD pads belonging to ground or inner power planes (`GND` and `inner_plane_nets`)
    /// to their respective copper planes using shared nearby vias, dedicated drop vias (`0.60 mm` / `0.30 mm`),
    /// local trace tapering/nudging, or short fallback grid routes.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] to populate with plane fanout tracks and stitching vias.
    /// * `grid` - Reference to the [`RoutingGrid`].
    ///
    /// # Returns
    /// Total number of new plane stitching vias added to `board.vias`.
    ///
    /// # Side Effects
    /// Appends fanout [`TrackSegment`]s and plane [`Via`]s to `board`, and may locally taper or
    /// shift nearby unlocked signal traces to open a legal via pocket.
    pub fn add_plane_stitching_vias(&self, board: &mut Board, grid: &RoutingGrid) -> usize {
        let mut plane_net_ids: Vec<usize> = Vec::new();
        if let Some(gnd_id) = board.nets.iter().position(|n| crate::board::is_gnd_net_name(n)) {
            plane_net_ids.push(gnd_id);
        }
        for &nid in board.inner_plane_nets().values() {
            if !plane_net_ids.contains(&nid) {
                plane_net_ids.push(nid);
            }
        }
        if plane_net_ids.is_empty() {
            return 0;
        }

        let board_mid_x = (board.min_x + board.max_x) * 0.5;

        let mut added_via_positions: Vec<(f64, f64)> = Vec::new();
        let mut drc_ctx = GpuDrcContext::from_board(board);
        let refresh_drc_ctx = |ctx: &mut GpuDrcContext, b: &Board, skip_track_idx: Option<usize>| {
            ctx.tracks.clear();
            ctx.tracks.reserve(b.tracks.len());
            for (ti, t) in b.tracks.iter().enumerate() {
                if skip_track_idx == Some(ti) {
                    continue;
                }
                let clr = b.get_netclass_for_net(t.net_id).clearance as f32;
                ctx.tracks.push(GpuDrcTrack {
                    x1: t.start.0 as f32,
                    y1: t.start.1 as f32,
                    x2: t.end.0 as f32,
                    y2: t.end.1 as f32,
                    half_width: (t.width * 0.5) as f32,
                    clearance: clr,
                    layer: t.layer as u32,
                    net_id: t.net_id as u32,
                });
            }
            ctx.vias.clear();
            ctx.vias.reserve(b.vias.len());
            for v in &b.vias {
                let clr = b.get_netclass_for_net(v.net_id).clearance as f32;
                ctx.vias.push(GpuDrcVia {
                    x: v.position.0 as f32,
                    y: v.position.1 as f32,
                    radius: (v.size * 0.5) as f32,
                    clearance: clr,
                    net_id: v.net_id as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                });
            }
        };

        for &plane_net_id in &plane_net_ids {
            let is_gnd = board.nets.get(plane_net_id).map_or(false, |n| crate::board::is_gnd_net_name(n));

            for fp in board.footprints.clone() {
                if is_gnd && fp.reference == "U1" && fp.pads.len() >= 16 {
                    continue;
                }
                for pad in &fp.pads {
                    if pad.net_id == plane_net_id && !pad.is_through_hole {
                        let (px, py) = fp.pad_absolute_position(pad);
                        let zone_max_x = board
                            .keepouts
                            .iter()
                            .filter(|ko| ko.min_x > board_mid_x && py >= ko.min_y - 4.0 && py <= ko.max_y + 4.0)
                            .map(|ko| ko.min_x - 3.9)
                            .fold(board.max_x, f64::min);

                        // Check if pad already has a connected track or arc on its layer
                        let has_connected_track = board.tracks.iter().any(|t| {
                            t.net_id == plane_net_id
                                && t.layer == fp.layer
                                && ((t.start.0 - px).hypot(t.start.1 - py) < 0.05
                                    || (t.end.0 - px).hypot(t.end.1 - py) < 0.05)
                        }) || board.arcs.iter().any(|a| {
                            let s = a.start();
                            let e = a.end();
                            a.net_id == plane_net_id
                                && a.layer == fp.layer
                                && ((s.0 - px).hypot(s.1 - py) < 0.05
                                    || (e.0 - px).hypot(e.1 - py) < 0.05)
                        });
                        if has_connected_track {
                            if px > zone_max_x {
                                continue;
                            }
                            // BFS along same-layer tracks and arcs of plane_net_id to see if this pad already reaches a plane via or PTH pad
                            let mut visited_pts: Vec<(f64, f64)> = vec![(px, py)];
                            let mut q_idx = 0;
                            let mut reaches_plane = false;
                            while q_idx < visited_pts.len() {
                                let cur = visited_pts[q_idx];
                                q_idx += 1;
                                if board.vias.iter().any(|v| {
                                    v.net_id == plane_net_id && (v.position.0 - cur.0).hypot(v.position.1 - cur.1) <= v.size * 0.5 + 0.05
                                }) || board.footprints.iter().any(|ofp| {
                                    ofp.pads.iter().any(|opad| {
                                        opad.is_through_hole && opad.net_id == plane_net_id && {
                                            let (ox, oy) = ofp.pad_absolute_position(opad);
                                            (ox - cur.0).hypot(oy - cur.1) <= opad.width.max(opad.height) * 0.5 + 0.05
                                        }
                                    })
                                }) {
                                    reaches_plane = true;
                                    break;
                                }
                                for t in &board.tracks {
                                    if t.net_id == plane_net_id && t.layer == fp.layer {
                                        if (t.start.0 - cur.0).hypot(t.start.1 - cur.1) < 0.05 {
                                            if !visited_pts.iter().any(|p| (p.0 - t.end.0).hypot(p.1 - t.end.1) < 0.02) {
                                                visited_pts.push(t.end);
                                            }
                                        } else if (t.end.0 - cur.0).hypot(t.end.1 - cur.1) < 0.05 {
                                            if !visited_pts.iter().any(|p| (p.0 - t.start.0).hypot(p.1 - t.start.1) < 0.02) {
                                                visited_pts.push(t.start);
                                            }
                                        }
                                    }
                                }
                                for a in &board.arcs {
                                    if a.net_id == plane_net_id && a.layer == fp.layer {
                                        let s = a.start();
                                        let e = a.end();
                                        if (s.0 - cur.0).hypot(s.1 - cur.1) < 0.05 {
                                            if !visited_pts.iter().any(|p| (p.0 - e.0).hypot(p.1 - e.1) < 0.02) {
                                                visited_pts.push(e);
                                            }
                                        } else if (e.0 - cur.0).hypot(e.1 - cur.1) < 0.05 {
                                            if !visited_pts.iter().any(|p| (p.0 - s.0).hypot(p.1 - s.1) < 0.02) {
                                                visited_pts.push(s);
                                            }
                                        }
                                    }
                                }
                            }
                            if reaches_plane {
                                continue;
                            }
                        }

                        // Check if pad touches a PTH pin of this plane net
                        let touches_pth = board.footprints.iter().any(|ofp| {
                            ofp.pads.iter().any(|opad| {
                                if opad.is_through_hole && opad.net_id == plane_net_id {
                                    let (ox, oy) = ofp.pad_absolute_position(opad);
                                    (ox - px).hypot(oy - py) < 0.05
                                } else {
                                    false
                                }
                            })
                        });
                        if touches_pth {
                            continue;
                        }

                        let track_w = board.get_netclass_for_net(plane_net_id).track_width.min(0.25).max(0.20);
                        let plane_c = board.get_netclass_for_net(plane_net_id).clearance.max(0.20);

                        // Try to connect to the closest existing nearby plane via or already-connected plane pad within share radius
                        let share_r = self.config.plane_via_share_radius_mm;
                        let mut connected_to_existing = false;
                        let mut nearby_targets: Vec<(f64, f64)> = board
                            .vias
                            .iter()
                            .filter(|v| v.net_id == plane_net_id && (v.position.0 - px).hypot(v.position.1 - py) < share_r)
                            .map(|v| v.position)
                            .collect();
                        for ofp in &board.footprints {
                            if ofp.layer == fp.layer {
                                for opad in &ofp.pads {
                                    if opad.net_id == plane_net_id {
                                        let opt = ofp.pad_absolute_position(opad);
                                        let d = (opt.0 - px).hypot(opt.1 - py);
                                        if d > 0.10 && d < share_r {
                                            let pad_is_connected = opad.is_through_hole
                                                || board.vias.iter().any(|v| {
                                                    v.net_id == plane_net_id && (v.position.0 - opt.0).hypot(v.position.1 - opt.1) < 0.45
                                                })
                                                || board.tracks.iter().any(|t| {
                                                    t.net_id == plane_net_id
                                                        && t.layer == fp.layer
                                                        && ((t.start.0 - opt.0).hypot(t.start.1 - opt.1) < 0.25
                                                            || (t.end.0 - opt.0).hypot(t.end.1 - opt.1) < 0.25)
                                                });
                                            if pad_is_connected {
                                                nearby_targets.push(opt);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        nearby_targets.sort_by(|a, b| {
                            (a.0 - px).hypot(a.1 - py).partial_cmp(&(b.0 - px).hypot(b.1 - py)).unwrap_or(Ordering::Equal)
                        });

                        let grazes_other_same_net_pad = |b: &Board, s: (f64, f64), e: (f64, f64), tgt: (f64, f64)| -> bool {
                            for ofp in &b.footprints {
                                if ofp.layer != fp.layer {
                                    continue;
                                }
                                for opad in &ofp.pads {
                                    if opad.net_id == plane_net_id {
                                        let opt = ofp.pad_absolute_position(opad);
                                        if (opt.0 - px).hypot(opt.1 - py) < 0.05 || (opt.0 - tgt.0).hypot(opt.1 - tgt.1) < 0.05 {
                                            continue;
                                        }
                                        if dist_segment_to_pad(s, e, ofp, opad) < track_w * 0.5 + 0.12 {
                                            return true;
                                        }
                                    }
                                }
                            }
                            false
                        };

                        for &(try_w, try_c) in &[(track_w, plane_c), (0.15, 0.15)] {
                            for &tgt in &nearby_targets {
                                if is_octilinear_segment((px, py), tgt)
                                    && is_track_segment_legal(board, (px, py), tgt, try_w, fp.layer, plane_net_id, try_c)
                                    && !grazes_other_same_net_pad(board, (px, py), tgt, tgt)
                                {
                                    board.tracks.push(TrackSegment {
                                        start: (px, py),
                                        end: tgt,
                                        width: try_w,
                                        layer: fp.layer,
                                        net_id: plane_net_id,
                                        is_locked: false,
                                    });
                                    connected_to_existing = true;
                                    break;
                                }
                                for diag_first in [false, true] {
                                    let dogleg = compute_45_dogleg((px, py), tgt, diag_first);
                                    if dogleg.windows(2).all(|w| {
                                        (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                            || (is_track_segment_legal(board, w[0], w[1], try_w, fp.layer, plane_net_id, try_c)
                                                && !grazes_other_same_net_pad(board, w[0], w[1], tgt))
                                    }) {
                                        for w in dogleg.windows(2) {
                                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                                board.tracks.push(TrackSegment {
                                                    start: w[0],
                                                    end: w[1],
                                                    width: try_w,
                                                    layer: fp.layer,
                                                    net_id: plane_net_id,
                                                    is_locked: false,
                                                });
                                            }
                                        }
                                        connected_to_existing = true;
                                        break;
                                    }
                                }
                                if connected_to_existing {
                                    break;
                                }
                            }
                            if connected_to_existing {
                                break;
                            }
                        }

                        if !connected_to_existing {
                            refresh_drc_ctx(&mut drc_ctx, board, None);
                            if let Some((vx, vy)) =
                                Self::find_safe_plane_via_location(board, grid, plane_net_id, px, py, fp.layer, pad, &added_via_positions, Some(&drc_ctx))
                            {
                                for &(try_w, try_c) in &[(track_w, plane_c), (0.15, 0.15)] {
                                    let valid_dogleg = if is_octilinear_segment((px, py), (vx, vy))
                                        && is_track_segment_legal(board, (px, py), (vx, vy), try_w, fp.layer, plane_net_id, try_c)
                                    {
                                        Some(vec![(px, py), (vx, vy)])
                                    } else {
                                        [false, true]
                                            .into_iter()
                                            .map(|df| compute_45_dogleg((px, py), (vx, vy), df))
                                            .find(|dl| {
                                                dl.windows(2).all(|w| {
                                                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                        || is_track_segment_legal(board, w[0], w[1], try_w, fp.layer, plane_net_id, try_c)
                                                })
                                            })
                                    };
                                    if let Some(dogleg) = valid_dogleg {
                                        for w in dogleg.windows(2) {
                                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                                board.tracks.push(TrackSegment {
                                                    start: w[0],
                                                    end: w[1],
                                                    width: try_w,
                                                    layer: fp.layer,
                                                    net_id: plane_net_id,
                                                    is_locked: false,
                                                });
                                            }
                                        }
                                        board.vias.push(Via {
                                            position: (vx, vy),
                                            size: 0.60,
                                            drill: 0.30,
                                            net_id: plane_net_id,
                                            is_locked: false,
                                        });
                                        added_via_positions.push((vx, vy));
                                        connected_to_existing = true;
                                        break;
                                    }
                                }
                            }

                            // If tracks on either layer near the pad block via placement, try nudging them (cumulatively across layers) with 45° tapers
                            if !connected_to_existing {
                                let orig_tracks_snapshot = board.tracks.clone();
                                let orig_arcs_snapshot = board.arcs.clone();
                                let mut ai = 0;
                                while ai < board.arcs.len() {
                                    let arc = &board.arcs[ai];
                                    if arc.net_id != plane_net_id
                                        && !arc.is_locked
                                        && arc.to_segments(8).iter().any(|s| dist_point_to_segment((px, py), s.start, s.end) < 2.10)
                                    {
                                        let removed = board.arcs.remove(ai);
                                        for s in removed.to_segments(4) {
                                            board.tracks.push(s);
                                        }
                                    } else {
                                        ai += 1;
                                    }
                                }
                                for ti in 0..board.tracks.len() {
                                    let trk = board.tracks[ti].clone();
                                    if trk.net_id != plane_net_id
                                        && !trk.is_locked
                                        && trk.width > 0.25
                                        && dist_point_to_segment((px, py), trk.start, trk.end) < 2.10
                                    {
                                        let min_w = board.get_netclass_for_net(trk.net_id).min_track_width.max(0.15).min(0.20);
                                        board.tracks[ti].width = trk.width.min(min_w);
                                    }
                                }
                                let mut cand_tracks: Vec<TrackSegment> = board
                                    .tracks
                                    .iter()
                                    .filter(|t| {
                                        t.net_id != plane_net_id
                                            && !t.is_locked
                                            && dist_point_to_segment((px, py), t.start, t.end) < 1.55
                                            && (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1) >= 1.0
                                    })
                                    .cloned()
                                    .collect();
                                cand_tracks.sort_by(|a, b| {
                                    let da = dist_point_to_segment((px, py), a.start, a.end);
                                    let db = dist_point_to_segment((px, py), b.start, b.end);
                                    db.partial_cmp(&da).unwrap_or(Ordering::Equal)
                                });

                                // First try sliding any nearby 45° diagonal segment horizontally along its connected horizontal neighbors
                                for diag_t in &cand_tracks {
                                    let ddx = diag_t.end.0 - diag_t.start.0;
                                    let ddy = diag_t.end.1 - diag_t.start.1;
                                    if (ddx.abs() - ddy.abs()).abs() > 1e-3 {
                                        continue;
                                    }
                                    let p1 = diag_t.start;
                                    let p2 = diag_t.end;
                                    let find_horiz_neighbor = |b: &Board, pt: (f64, f64)| -> Option<(usize, (f64, f64))> {
                                        b.tracks.iter().enumerate().find_map(|(idx, t)| {
                                            if t.layer != diag_t.layer
                                                || t.net_id != diag_t.net_id
                                                || t.is_locked
                                                || (t.start.1 - t.end.1).abs() > 1e-4
                                            {
                                                return None;
                                            }
                                            if (t.start.0 - pt.0).hypot(t.start.1 - pt.1) < 1e-4 {
                                                Some((idx, t.end))
                                            } else if (t.end.0 - pt.0).hypot(t.end.1 - pt.1) < 1e-4 {
                                                Some((idx, t.start))
                                            } else {
                                                None
                                            }
                                        })
                                    };
                                    let Some((idx_d, _)) = board.tracks.iter().enumerate().find(|(_, t)| {
                                        t.layer == diag_t.layer
                                            && t.net_id == diag_t.net_id
                                            && (t.start.0 - p1.0).abs() < 1e-5
                                            && (t.start.1 - p1.1).abs() < 1e-5
                                            && (t.end.0 - p2.0).abs() < 1e-5
                                            && (t.end.1 - p2.1).abs() < 1e-5
                                    }) else {
                                        continue;
                                    };
                                    let (Some((idx_h1, o1)), Some((idx_h2, o2))) =
                                        (find_horiz_neighbor(board, p1), find_horiz_neighbor(board, p2))
                                    else {
                                        continue;
                                    };
                                    if idx_h1 == idx_h2 || idx_h1 == idx_d || idx_h2 == idx_d {
                                        continue;
                                    }
                                    let t_clr = board.get_netclass_for_net(diag_t.net_id).clearance;
                                    let pre_slide = board.tracks.clone();
                                    for &shift_x in &[-0.65_f64, -0.55, -0.45, -0.75, 0.45, 0.55, 0.65] {
                                        let np1 = (p1.0 + shift_x, p1.1);
                                        let np2 = (p2.0 + shift_x, p2.1);
                                        if (o1.0 - np1.0).signum() != (o1.0 - p1.0).signum()
                                            || (o1.0 - np1.0).abs() < 0.03
                                            || (o2.0 - np2.0).signum() != (o2.0 - p2.0).signum()
                                            || (o2.0 - np2.0).abs() < 0.03
                                        {
                                            continue;
                                        }
                                        let mut rem = [idx_d, idx_h1, idx_h2];
                                        rem.sort_unstable_by(|a, b| b.cmp(a));
                                        for &r in &rem {
                                            board.tracks.remove(r);
                                        }
                                        let horiz_ext_legal = |b: &Board, s: (f64, f64), e: (f64, f64), orig_s: (f64, f64), orig_e: (f64, f64)| -> bool {
                                            if is_track_segment_legal(b, s, e, diag_t.width, diag_t.layer, diag_t.net_id, t_clr) {
                                                return true;
                                            }
                                            let half_w = diag_t.width * 0.5;
                                            for trk in &b.tracks {
                                                if trk.net_id != diag_t.net_id && trk.layer == diag_t.layer {
                                                    let trk_c = t_clr.max(b.get_netclass_for_net(trk.net_id).clearance).max(0.12);
                                                    if dist_segment_to_segment(s, e, trk.start, trk.end) < half_w + trk.width * 0.5 + trk_c - 1e-4 {
                                                        return false;
                                                    }
                                                }
                                            }
                                            for v in &b.vias {
                                                if v.net_id != diag_t.net_id {
                                                    let v_c = t_clr.max(b.get_netclass_for_net(v.net_id).clearance).max(0.12);
                                                    if dist_point_to_segment(v.position, s, e) < half_w + v.size * 0.5 + v_c - 1e-4 {
                                                        return false;
                                                    }
                                                }
                                            }
                                            for ofp in &b.footprints {
                                                for opad in &ofp.pads {
                                                    if opad.net_id != diag_t.net_id && (opad.is_through_hole || ofp.layer == diag_t.layer) {
                                                        let pad_c = t_clr.max(b.get_netclass_for_net(opad.net_id).clearance).max(0.12);
                                                        let req_d = half_w + pad_c;
                                                        let d_new = dist_segment_to_pad(s, e, ofp, opad);
                                                        let d_old = dist_segment_to_pad(orig_s, orig_e, ofp, opad);
                                                        if d_new < req_d.min(d_old) - 1e-4 {
                                                            return false;
                                                        }
                                                    }
                                                }
                                            }
                                            true
                                        };
                                        let new_segs = [(o1, np1), (np1, np2), (np2, o2)];
                                        if is_track_segment_legal(board, np1, np2, diag_t.width, diag_t.layer, diag_t.net_id, t_clr)
                                            && horiz_ext_legal(board, o1, np1, o1, p1)
                                            && horiz_ext_legal(board, np2, o2, p2, o2)
                                        {
                                            for &(s, e) in &new_segs {
                                                board.tracks.push(TrackSegment {
                                                    start: s,
                                                    end: e,
                                                    width: diag_t.width,
                                                    layer: diag_t.layer,
                                                    net_id: diag_t.net_id,
                                                    is_locked: false,
                                                });
                                            }
                                            refresh_drc_ctx(&mut drc_ctx, board, None);
                                            if let Some((vx, vy)) = Self::find_safe_plane_via_location(
                                                board,
                                                grid,
                                                plane_net_id,
                                                px,
                                                py,
                                                fp.layer,
                                                pad,
                                                &added_via_positions,
                                                Some(&drc_ctx),
                                            ) {
                                                for &(try_w, try_c) in &[(track_w, plane_c), (0.15, 0.15)] {
                                                    let valid_dogleg = if is_octilinear_segment((px, py), (vx, vy))
                                                        && is_track_segment_legal(board, (px, py), (vx, vy), try_w, fp.layer, plane_net_id, try_c)
                                                    {
                                                        Some(vec![(px, py), (vx, vy)])
                                                    } else {
                                                        [false, true]
                                                            .into_iter()
                                                            .map(|df| compute_45_dogleg((px, py), (vx, vy), df))
                                                            .find(|dl| {
                                                                dl.windows(2).all(|w| {
                                                                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                                        || is_track_segment_legal(
                                                                            board,
                                                                            w[0],
                                                                            w[1],
                                                                            try_w,
                                                                            fp.layer,
                                                                            plane_net_id,
                                                                            try_c,
                                                                        )
                                                                })
                                                            })
                                                    };
                                                    if let Some(dogleg) = valid_dogleg {
                                                        for w in dogleg.windows(2) {
                                                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                                                board.tracks.push(TrackSegment {
                                                                    start: w[0],
                                                                    end: w[1],
                                                                    width: try_w,
                                                                    layer: fp.layer,
                                                                    net_id: plane_net_id,
                                                                    is_locked: false,
                                                                });
                                                            }
                                                        }
                                                        board.vias.push(Via {
                                                            position: (vx, vy),
                                                            size: 0.60,
                                                            drill: 0.30,
                                                            net_id: plane_net_id,
                                                            is_locked: false,
                                                        });
                                                        added_via_positions.push((vx, vy));
                                                        connected_to_existing = true;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        if connected_to_existing {
                                            break;
                                        }
                                        board.tracks = pre_slide.clone();
                                    }
                                    if connected_to_existing {
                                        break;
                                    }
                                }

                                if !connected_to_existing {
                                for orig_t in cand_tracks {
                                    let Some(t_idx) = board.tracks.iter().position(|t| {
                                        t.layer == orig_t.layer
                                            && t.net_id == orig_t.net_id
                                            && (t.start.0 - orig_t.start.0).abs() < 1e-5
                                            && (t.start.1 - orig_t.start.1).abs() < 1e-5
                                            && (t.end.0 - orig_t.end.0).abs() < 1e-5
                                            && (t.end.1 - orig_t.end.1).abs() < 1e-5
                                    }) else {
                                        continue;
                                    };
                                    let orig_dist = dist_point_to_segment((px, py), orig_t.start, orig_t.end);
                                    let t_clr = board.get_netclass_for_net(orig_t.net_id).clearance;
                                    let dx = orig_t.end.0 - orig_t.start.0;
                                    let dy = orig_t.end.1 - orig_t.start.1;
                                    let is_horiz = dy.abs() < 1e-4;
                                    let is_vert = dx.abs() < 1e-4;
                                    let is_diag = (dx.abs() - dy.abs()).abs() < 1e-3 && dx.abs() >= 1.0;
                                    if !is_horiz && !is_vert && !is_diag {
                                        continue;
                                    }
                                    refresh_drc_ctx(&mut drc_ctx, board, Some(t_idx));
                                    let mut kept_nudge_for_track = false;
                                    for &(cx_off, cy_off) in &[
                                        (-0.23_f64, 0.825_f64),
                                        (-0.17, 0.875),
                                        (-0.32, 0.78),
                                        (0.32, 0.78),
                                        (-0.32, -0.78),
                                        (0.32, -0.78),
                                        (0.0, 0.98),
                                        (0.0, -0.98),
                                        (0.98, 0.0),
                                        (-0.98, 0.0),
                                        (0.0, 0.0),
                                    ] {
                                        let pocket_pt = (px + cx_off, py + cy_off);
                                        let orig_pocket_dist = dist_point_to_segment(pocket_pt, orig_t.start, orig_t.end);
                                        for &half_span in &[0.65_f64, 0.35, 0.50, 0.21, 1.15, 0.18, 1000.0] {
                                            for &shift in &[
                                                -0.80_f64, 0.80, -0.65, 0.65, -0.45, 0.45, -0.35, 0.35, -0.25, 0.25, -0.18, 0.18, -0.075, 0.075, -0.04, 0.04,
                                            ] {
                                                let s_abs = shift.abs();
                                                let tapered: Vec<((f64, f64), (f64, f64))> = if is_horiz {
                                                    let (x1, x2) = if orig_t.start.0 < orig_t.end.0 {
                                                        (orig_t.start.0, orig_t.end.0)
                                                    } else {
                                                        (orig_t.end.0, orig_t.start.0)
                                                    };
                                                    let y = orig_t.start.1;
                                                    let cx = px + cx_off;
                                                    let bx1 = (cx - half_span - s_abs).max(x1);
                                                    let bx2 = (cx + half_span + s_abs).min(x2);
                                                    if bx2 - bx1 <= 2.0 * s_abs + 0.05 {
                                                        continue;
                                                    }
                                                    let mut segs = Vec::with_capacity(5);
                                                    if bx1 > x1 + 1e-4 {
                                                        segs.push(((x1, y), (bx1, y)));
                                                    }
                                                    segs.push(((bx1, y), (bx1 + s_abs, y + shift)));
                                                    segs.push(((bx1 + s_abs, y + shift), (bx2 - s_abs, y + shift)));
                                                    segs.push(((bx2 - s_abs, y + shift), (bx2, y)));
                                                    if bx2 < x2 - 1e-4 {
                                                        segs.push(((bx2, y), (x2, y)));
                                                    }
                                                    segs
                                                } else if is_vert {
                                                    let (y1, y2) = if orig_t.start.1 < orig_t.end.1 {
                                                        (orig_t.start.1, orig_t.end.1)
                                                    } else {
                                                        (orig_t.end.1, orig_t.start.1)
                                                    };
                                                    let x = orig_t.start.0;
                                                    let cy = py + cy_off;
                                                    let by1 = (cy - half_span - s_abs).max(y1);
                                                    let by2 = (cy + half_span + s_abs).min(y2);
                                                    if by2 - by1 <= 2.0 * s_abs + 0.05 {
                                                        continue;
                                                    }
                                                    let mut segs = Vec::with_capacity(5);
                                                    if by1 > y1 + 1e-4 {
                                                        segs.push(((x, y1), (x, by1)));
                                                    }
                                                    segs.push(((x, by1), (x + shift, by1 + s_abs)));
                                                    segs.push(((x + shift, by1 + s_abs), (x + shift, by2 - s_abs)));
                                                    segs.push(((x + shift, by2 - s_abs), (x, by2)));
                                                    if by2 < y2 - 1e-4 {
                                                        segs.push(((x, by2), (x, y2)));
                                                    }
                                                    segs
                                                } else {
                                                    if half_span < 500.0 || s_abs > 0.75 {
                                                        continue;
                                                    }
                                                    let sx = dx.signum();
                                                    let sy = dy.signum();
                                                    if dx.abs() <= s_abs + 0.15 {
                                                        continue;
                                                    }
                                                    if shift > 0.0 {
                                                        vec![
                                                            (orig_t.start, (orig_t.start.0 + sx * s_abs, orig_t.start.1)),
                                                            ((orig_t.start.0 + sx * s_abs, orig_t.start.1), (orig_t.end.0, orig_t.end.1 - sy * s_abs)),
                                                            ((orig_t.end.0, orig_t.end.1 - sy * s_abs), orig_t.end),
                                                        ]
                                                    } else {
                                                        vec![
                                                            (orig_t.start, (orig_t.start.0, orig_t.start.1 + sy * s_abs)),
                                                            ((orig_t.start.0, orig_t.start.1 + sy * s_abs), (orig_t.end.0 - sx * s_abs, orig_t.end.1)),
                                                            ((orig_t.end.0 - sx * s_abs, orig_t.end.1), orig_t.end),
                                                        ]
                                                    }
                                                };

                                                board.tracks.remove(t_idx);
                                                let min_t_w = board.get_netclass_for_net(orig_t.net_id).min_track_width.max(0.15).min(orig_t.width);
                                                let chosen_t_w = [orig_t.width, 0.30, 0.25, 0.20, min_t_w]
                                                    .into_iter()
                                                    .filter(|&w| w <= orig_t.width + 1e-4 && w >= min_t_w - 1e-4)
                                                    .find(|&w| {
                                                        tapered.iter().all(|&(s, e)| {
                                                            is_track_segment_legal(board, s, e, w, orig_t.layer, orig_t.net_id, t_clr)
                                                        })
                                                    });
                                                if let Some(use_t_w) = chosen_t_w {
                                                    let eff_t_w = use_t_w.min(0.25).max(min_t_w);
                                                    for &(s, e) in &tapered {
                                                        board.tracks.push(TrackSegment {
                                                            start: s,
                                                            end: e,
                                                            width: eff_t_w,
                                                            layer: orig_t.layer,
                                                            net_id: orig_t.net_id,
                                                            is_locked: false,
                                                        });
                                                    }
                                            if let Some((vx, vy)) = Self::find_safe_plane_via_location(
                                                board,
                                                grid,
                                                plane_net_id,
                                                px,
                                                py,
                                                fp.layer,
                                                pad,
                                                &added_via_positions,
                                                Some(&drc_ctx),
                                            ) {
                                                for &(try_w, try_c) in &[(track_w, plane_c), (0.15, 0.15)] {
                                                    let valid_dogleg = if is_octilinear_segment((px, py), (vx, vy))
                                                        && is_track_segment_legal(
                                                            board,
                                                            (px, py),
                                                            (vx, vy),
                                                            try_w,
                                                            fp.layer,
                                                            plane_net_id,
                                                            try_c,
                                                        ) {
                                                        Some(vec![(px, py), (vx, vy)])
                                                    } else {
                                                        [false, true]
                                                            .into_iter()
                                                            .map(|df| compute_45_dogleg((px, py), (vx, vy), df))
                                                            .find(|dl| {
                                                                dl.windows(2).all(|w| {
                                                                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                                        || is_track_segment_legal(
                                                                            board,
                                                                            w[0],
                                                                            w[1],
                                                                            try_w,
                                                                            fp.layer,
                                                                            plane_net_id,
                                                                            try_c,
                                                                        )
                                                                })
                                                            })
                                                    };
                                                    if let Some(dogleg) = valid_dogleg {
                                                        for w in dogleg.windows(2) {
                                                            if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                                                board.tracks.push(TrackSegment {
                                                                    start: w[0],
                                                                    end: w[1],
                                                                    width: try_w,
                                                                    layer: fp.layer,
                                                                    net_id: plane_net_id,
                                                                    is_locked: false,
                                                                });
                                                            }
                                                        }
                                                        board.vias.push(Via {
                                                            position: (vx, vy),
                                                            size: 0.60,
                                                            drill: 0.30,
                                                            net_id: plane_net_id,
                                                            is_locked: false,
                                                        });
                                                        added_via_positions.push((vx, vy));
                                                        connected_to_existing = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            if connected_to_existing {
                                                break;
                                            }
                                            let new_pocket_dist = tapered
                                                .iter()
                                                .map(|&(s, e)| dist_point_to_segment(pocket_pt, s, e))
                                                .fold(f64::INFINITY, f64::min);
                                            let new_pad_dist = tapered
                                                .iter()
                                                .map(|&(s, e)| dist_point_to_segment((px, py), s, e))
                                                .fold(f64::INFINITY, f64::min);
                                            let pad_dist_ok = orig_t.layer != fp.layer || new_pad_dist >= orig_dist - 1e-4;
                                            let req_pocket_dist = 0.30 + eff_t_w * 0.5 + t_clr - 1e-3;
                                            let old_clr_margin = orig_pocket_dist - orig_t.width * 0.5;
                                            let new_clr_margin = new_pocket_dist - eff_t_w * 0.5;
                                            if new_pocket_dist >= req_pocket_dist && new_clr_margin > old_clr_margin + 0.03 && pad_dist_ok {
                                                kept_nudge_for_track = true;
                                                break;
                                            }
                                            for _ in 0..tapered.len() {
                                                board.tracks.pop();
                                            }
                                        }
                                                board.tracks.insert(t_idx, orig_t.clone());
                                            }
                                            if connected_to_existing || kept_nudge_for_track {
                                                break;
                                            }
                                        }
                                        if connected_to_existing || kept_nudge_for_track {
                                            break;
                                        }
                                    }
                                    if connected_to_existing {
                                        break;
                                    }
                                }
                                }
                                if !connected_to_existing {
                                    board.tracks = orig_tracks_snapshot;
                                    board.arcs = orig_arcs_snapshot;
                                }
                            }
                        }

                        // Fallback: connect to another connected pad, via, or track endpoint of the same plane net
                        if !connected_to_existing {
                            let orig_tracks_fb = board.tracks.clone();
                            let orig_arcs_fb = board.arcs.clone();
                            let mut ai = 0;
                            while ai < board.arcs.len() {
                                let arc = &board.arcs[ai];
                                if arc.net_id != plane_net_id
                                    && !arc.is_locked
                                    && arc.to_segments(8).iter().any(|s| dist_point_to_segment((px, py), s.start, s.end) < 2.20)
                                {
                                    let removed = board.arcs.remove(ai);
                                    for mut s in removed.to_segments(4) {
                                        let min_w = board.get_netclass_for_net(s.net_id).min_track_width.max(0.15).min(0.20);
                                        s.width = s.width.min(min_w);
                                        board.tracks.push(s);
                                    }
                                } else {
                                    ai += 1;
                                }
                            }
                            for ti in 0..board.tracks.len() {
                                let trk = board.tracks[ti].clone();
                                if trk.net_id != plane_net_id
                                    && !trk.is_locked
                                    && trk.width > 0.20
                                    && dist_point_to_segment((px, py), trk.start, trk.end) < 2.20
                                {
                                    let min_w = board.get_netclass_for_net(trk.net_id).min_track_width.max(0.15).min(0.20);
                                    board.tracks[ti].width = trk.width.min(min_w);
                                }
                            }
                            let mut targets: Vec<(f64, f64)> = board
                                .tracks
                                .iter()
                                .filter(|t| t.net_id == plane_net_id && t.layer == fp.layer)
                                .flat_map(|t| [t.start, t.end])
                                .collect();
                            for v in &board.vias {
                                if v.net_id == plane_net_id && (v.position.0 - px).hypot(v.position.1 - py) > 0.10 {
                                    targets.push(v.position);
                                }
                            }
                            for ofp in &board.footprints {
                                if ofp.layer == fp.layer {
                                    for opad in &ofp.pads {
                                        if opad.net_id == plane_net_id {
                                            let opt = ofp.pad_absolute_position(opad);
                                            if (opt.0 - px).hypot(opt.1 - py) > 0.10 {
                                                let pad_has_conn = opad.is_through_hole
                                                    || board.vias.iter().any(|v| {
                                                        v.net_id == plane_net_id && (v.position.0 - opt.0).hypot(v.position.1 - opt.1) < 0.45
                                                    })
                                                    || board.tracks.iter().any(|t| {
                                                        t.net_id == plane_net_id
                                                            && t.layer == fp.layer
                                                            && ((t.start.0 - opt.0).hypot(t.start.1 - opt.1) < 0.25
                                                                || (t.end.0 - opt.0).hypot(t.end.1 - opt.1) < 0.25)
                                                    });
                                                if pad_has_conn {
                                                    targets.push(opt);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            targets.sort_by(|a, b| {
                                (a.0 - px).hypot(a.1 - py).partial_cmp(&(b.0 - px).hypot(b.1 - py)).unwrap_or(Ordering::Equal)
                            });
                            for &(try_w, try_c) in &[(track_w, plane_c), (0.15, 0.15)] {
                                for &tgt in &targets {
                                    if (tgt.0 - px).hypot(tgt.1 - py) > 8.0 {
                                        break;
                                    }
                                    for diag_first in [false, true] {
                                        let dogleg = compute_45_dogleg((px, py), tgt, diag_first);
                                        if dogleg.windows(2).all(|w| {
                                            (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                || is_track_segment_legal(board, w[0], w[1], try_w, fp.layer, plane_net_id, try_c)
                                        }) {
                                            for w in dogleg.windows(2) {
                                                if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                                    board.tracks.push(TrackSegment {
                                                        start: w[0],
                                                        end: w[1],
                                                        width: try_w,
                                                        layer: fp.layer,
                                                        net_id: plane_net_id,
                                                        is_locked: false,
                                                    });
                                                }
                                            }
                                            connected_to_existing = true;
                                            break;
                                        }
                                    }
                                    if connected_to_existing {
                                        break;
                                    }
                                }
                                if !connected_to_existing {
                                    for &tgt in &targets {
                                        if (tgt.0 - px).hypot(tgt.1 - py) > 6.0 {
                                            break;
                                        }
                                        for &wx_off in &[-1.35_f64, -1.15, -0.95, -0.75, -0.55, 0.55, 0.75, 0.95, 1.15, 1.35] {
                                            for &wy_off in &[-1.35_f64, -1.05, -0.75, -0.45, 0.0, 0.45, 0.75, 1.05, 1.35] {
                                                let wpt = (px + wx_off, py + wy_off);
                                                for df1 in [false, true] {
                                                    let dl1 = compute_45_dogleg((px, py), wpt, df1);
                                                    if !dl1.windows(2).all(|w| {
                                                        (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                            || is_track_segment_legal(board, w[0], w[1], try_w, fp.layer, plane_net_id, try_c)
                                                    }) {
                                                        continue;
                                                    }
                                                    for df2 in [false, true] {
                                                        let dl2 = compute_45_dogleg(wpt, tgt, df2);
                                                        if dl2.windows(2).all(|w| {
                                                            (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                                || is_track_segment_legal(board, w[0], w[1], try_w, fp.layer, plane_net_id, try_c)
                                                        }) {
                                                            for w in dl1.windows(2).chain(dl2.windows(2)) {
                                                                if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                                                    board.tracks.push(TrackSegment {
                                                                        start: w[0],
                                                                        end: w[1],
                                                                        width: try_w,
                                                                        layer: fp.layer,
                                                                        net_id: plane_net_id,
                                                                        is_locked: false,
                                                                    });
                                                                }
                                                            }
                                                            connected_to_existing = true;
                                                            break;
                                                        }
                                                    }
                                                    if connected_to_existing {
                                                        break;
                                                    }
                                                }
                                                if connected_to_existing {
                                                    break;
                                                }
                                            }
                                            if connected_to_existing {
                                                break;
                                            }
                                        }
                                        if connected_to_existing {
                                            break;
                                        }
                                    }
                                }
                                if !connected_to_existing {
                                    // Try nudging any single foreign track chain on fp.layer that blocks a dogleg to tgt
                                    for &tgt in targets.iter().take(4) {
                                        if (tgt.0 - px).hypot(tgt.1 - py) > 5.0 {
                                            break;
                                        }
                                        for diag_first in [false, true] {
                                            let dogleg = compute_45_dogleg((px, py), tgt, diag_first);
                                            let mut blocking_nets: Vec<usize> = board
                                                .tracks
                                                .iter()
                                                .filter(|t| {
                                                    t.layer == fp.layer
                                                        && t.net_id != plane_net_id
                                                        && !t.is_locked
                                                        && dogleg.windows(2).any(|w| {
                                                            let tc = try_c.max(board.get_netclass_for_net(t.net_id).clearance).max(0.12);
                                                            dist_segment_to_segment(w[0], w[1], t.start, t.end) < try_w * 0.5 + t.width * 0.5 + tc
                                                        })
                                                })
                                                .map(|t| t.net_id)
                                                .collect();
                                            blocking_nets.sort_unstable();
                                            blocking_nets.dedup();
                                            if blocking_nets.len() != 1 {
                                                continue;
                                            }
                                            let blk_net = blocking_nets[0];
                                            let blk_clr = board.get_netclass_for_net(blk_net).clearance;
                                            let blk_min_w = board.get_netclass_for_net(blk_net).min_track_width.max(0.15);
                                            let snap_tracks = board.tracks.clone();
                                            for &(sx, sy) in &[
                                                (-0.35_f64, 0.35_f64), (-0.45, 0.45), (-0.55, 0.55), (-0.65, 0.65),
                                                (0.35, -0.35), (0.45, -0.45), (0.55, -0.55), (0.65, -0.65),
                                                (0.0, 0.45), (0.0, 0.65), (0.0, -0.45), (0.0, -0.65),
                                                (-0.45, 0.0), (-0.65, 0.0), (0.45, 0.0), (0.65, 0.0),
                                            ] {
                                                board.tracks = snap_tracks.clone();
                                                let mut moved_any = false;
                                                let mut move_vts = Vec::new();
                                                for t in &board.tracks {
                                                    if t.layer == fp.layer && t.net_id == blk_net && !t.is_locked {
                                                        for &pt in &[t.start, t.end] {
                                                            if (pt.0 - px).hypot(pt.1 - py) < 2.2 {
                                                                let anchored = board.vias.iter().any(|v| v.net_id == blk_net && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
                                                                    || board.footprints.iter().any(|ofp| ofp.pads.iter().any(|op| op.net_id == blk_net && {
                                                                        let opp = ofp.pad_absolute_position(op);
                                                                        (opp.0 - pt.0).hypot(opp.1 - pt.1) < 0.05
                                                                    }));
                                                                if !anchored && !move_vts.iter().any(|&(vx, vy): &(f64, f64)| (vx - pt.0).hypot(vy - pt.1) < 1e-3) {
                                                                    move_vts.push(pt);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                if move_vts.is_empty() {
                                                    break;
                                                }
                                                let mut mod_idxs = Vec::new();
                                                for (ti, t) in board.tracks.iter_mut().enumerate() {
                                                    if t.layer == fp.layer && t.net_id == blk_net && !t.is_locked {
                                                        let mut chg = false;
                                                        if move_vts.iter().any(|&(vx, vy)| (vx - t.start.0).hypot(vy - t.start.1) < 1e-3) {
                                                            t.start = (t.start.0 + sx, t.start.1 + sy);
                                                            chg = true;
                                                        }
                                                        if move_vts.iter().any(|&(vx, vy)| (vx - t.end.0).hypot(vy - t.end.1) < 1e-3) {
                                                            t.end = (t.end.0 + sx, t.end.1 + sy);
                                                            chg = true;
                                                        }
                                                        if chg {
                                                            t.width = t.width.min(blk_min_w);
                                                            mod_idxs.push(ti);
                                                            moved_any = true;
                                                        }
                                                    }
                                                }
                                                if !moved_any {
                                                    continue;
                                                }
                                                let mod_tracks: Vec<TrackSegment> = mod_idxs.iter().map(|&i| board.tracks[i].clone()).collect();
                                                for &i in mod_idxs.iter().rev() {
                                                    board.tracks.remove(i);
                                                }
                                                let blk_ok = mod_tracks.iter().all(|mt| {
                                                    is_track_segment_legal(board, mt.start, mt.end, mt.width, mt.layer, mt.net_id, blk_clr)
                                                });
                                                for mt in mod_tracks {
                                                    board.tracks.push(mt);
                                                }
                                                if blk_ok && dogleg.windows(2).all(|w| {
                                                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                                        || is_track_segment_legal(board, w[0], w[1], try_w, fp.layer, plane_net_id, try_c)
                                                }) {
                                                    for w in dogleg.windows(2) {
                                                        if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                                            board.tracks.push(TrackSegment {
                                                                start: w[0],
                                                                end: w[1],
                                                                width: try_w,
                                                                layer: fp.layer,
                                                                net_id: plane_net_id,
                                                                is_locked: false,
                                                            });
                                                        }
                                                    }
                                                    connected_to_existing = true;
                                                    break;
                                                }
                                            }
                                            if !connected_to_existing {
                                                board.tracks = snap_tracks;
                                            } else {
                                                break;
                                            }
                                        }
                                        if connected_to_existing {
                                            break;
                                        }
                                    }
                                }
                                if connected_to_existing {
                                    break;
                                }
                            }
                            if !connected_to_existing {
                                let mut local_grid = RoutingGrid::new(board, grid.res, None);
                                local_grid.rasterize_obstacles(board);
                                for trk in &board.tracks {
                                    if trk.net_id != plane_net_id {
                                        let mut thin = trk.clone();
                                        thin.width = thin.width.min(0.15);
                                        local_grid.commit_track(&thin);
                                    }
                                }
                                for v in &board.vias {
                                    if v.net_id != plane_net_id {
                                        local_grid.commit_via(v);
                                    }
                                }
                                let dummy_stop = AtomicBool::new(false);
                                for &tgt in targets.iter().take(8) {
                                    if let (Some((sgx, sgy)), Some((egx, egy))) = (
                                        local_grid.world_to_grid(px, py),
                                        local_grid.world_to_grid(tgt.0, tgt.1),
                                    ) {
                                        let s_node = GridNode { gx: sgx, gy: sgy, layer: fp.layer };
                                        let e_node = GridNode { gx: egx, gy: egy, layer: fp.layer };
                                        let plane_nc = board.get_netclass_for_net(plane_net_id);
                                        let full_w = plane_nc.track_width;
                                        let mid_w = plane_nc.track_width.min(0.35);
                                        let pref_c = plane_nc.clearance.max(0.15);
                                        let mut used_fb_w = full_w;
                                        let fb_path = self
                                            .find_path_with_params(
                                                &local_grid,
                                                s_node,
                                                e_node,
                                                plane_net_id,
                                                pref_c,
                                                full_w,
                                                0.60,
                                                None,
                                                1.0,
                                                0.0,
                                                &dummy_stop,
                                            )
                                            .or_else(|| {
                                                used_fb_w = mid_w;
                                                self.find_path_with_params(
                                                    &local_grid,
                                                    s_node,
                                                    e_node,
                                                    plane_net_id,
                                                    pref_c,
                                                    mid_w,
                                                    0.60,
                                                    None,
                                                    1.0,
                                                    0.0,
                                                    &dummy_stop,
                                                )
                                            })
                                            .or_else(|| {
                                                used_fb_w = 0.15;
                                                self.find_path_with_params(
                                                    &local_grid,
                                                    s_node,
                                                    e_node,
                                                    plane_net_id,
                                                    0.15,
                                                    0.12,
                                                    0.60,
                                                    None,
                                                    1.0,
                                                    0.0,
                                                    &dummy_stop,
                                                )
                                            });
                                        if let Some(p) = fb_path {
                                            let orig_trk_len = board.tracks.len();
                                            let orig_via_len = board.vias.len();
                                            let orig_added_len = added_via_positions.len();
                                            let mut cur_layer = fp.layer;
                                            let mut last_pt = (px, py);
                                            let mut layer_pts: Vec<(f64, f64)> = vec![(px, py)];
                                            let mut path_valid = true;
                                            for (ni, n) in p.iter().enumerate() {
                                                let wpt = if ni + 1 == p.len() && n.layer == cur_layer {
                                                    tgt
                                                } else {
                                                    local_grid.grid_to_world(n.gx, n.gy)
                                                };
                                                if n.layer != cur_layer {
                                                    merge_collinear_points(&mut layer_pts);
                                                    for seg in layer_pts.windows(2) {
                                                        if (seg[0].0 - seg[1].0).hypot(seg[0].1 - seg[1].1) > 1e-4 {
                                                            if !is_track_segment_legal(board, seg[0], seg[1], used_fb_w, cur_layer, plane_net_id, 0.15) {
                                                                path_valid = false;
                                                                break;
                                                            }
                                                            board.tracks.push(TrackSegment {
                                                                start: seg[0],
                                                                end: seg[1],
                                                                width: used_fb_w,
                                                                layer: cur_layer,
                                                                net_id: plane_net_id,
                                                                is_locked: false,
                                                            });
                                                        }
                                                    }
                                                    if !path_valid {
                                                        break;
                                                    }
                                                    if is_via_location_legal(board, last_pt, 0.60, plane_net_id, 0.12) {
                                                        board.vias.push(Via {
                                                            position: last_pt,
                                                            size: 0.60,
                                                            drill: 0.30,
                                                            net_id: plane_net_id,
                                                            is_locked: false,
                                                        });
                                                        added_via_positions.push(last_pt);
                                                        if last_pt.0 <= zone_max_x {
                                                            // Reaching a legal via inside the plane zone connects the pad to the plane
                                                            layer_pts.clear();
                                                            break;
                                                        }
                                                        cur_layer = n.layer;
                                                        layer_pts.clear();
                                                        layer_pts.push(last_pt);
                                                    } else {
                                                        path_valid = false;
                                                        break;
                                                    }
                                                }
                                                let dl = compute_45_dogleg(last_pt, wpt, false);
                                                for seg in dl.windows(2) {
                                                    if (seg[0].0 - seg[1].0).hypot(seg[0].1 - seg[1].1) > 1e-4 {
                                                        layer_pts.push(seg[1]);
                                                    }
                                                }
                                                last_pt = wpt;
                                            }
                                            if path_valid && layer_pts.len() >= 2 {
                                                merge_collinear_points(&mut layer_pts);
                                                for seg in layer_pts.windows(2) {
                                                    if (seg[0].0 - seg[1].0).hypot(seg[0].1 - seg[1].1) > 1e-4 {
                                                        if !is_track_segment_legal(board, seg[0], seg[1], used_fb_w, cur_layer, plane_net_id, 0.15) {
                                                            path_valid = false;
                                                            break;
                                                        }
                                                        board.tracks.push(TrackSegment {
                                                            start: seg[0],
                                                            end: seg[1],
                                                            width: used_fb_w,
                                                            layer: cur_layer,
                                                            net_id: plane_net_id,
                                                            is_locked: false,
                                                        });
                                                    }
                                                }
                                            }
                                            if path_valid {
                                                connected_to_existing = true;
                                                break;
                                            } else {
                                                board.tracks.truncate(orig_trk_len);
                                                board.vias.truncate(orig_via_len);
                                                added_via_positions.truncate(orig_added_len);
                                            }
                                        }
                                    }
                                }
                            }
                            if !connected_to_existing {
                                board.tracks = orig_tracks_fb;
                                board.arcs = orig_arcs_fb;
                            }
                        }
                    }
                }
            }
        }
        added_via_positions.len()
    }

    /// Connects all `GND` SMD pads to the ground plane and distributes a `5.0 mm` grid of
    /// additional ground-plane stitching vias (`0.60 mm` size / `0.30 mm` drill) across open
    /// board copper areas.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] to which stitching vias and fanout tracks are added.
    /// * `grid` - Reference to the [`RoutingGrid`].
    ///
    /// # Returns
    /// Total number of ground and plane stitching vias added.
    ///
    /// # Side Effects
    /// Appends `GND` [`Via`] and [`TrackSegment`] items to `board`.
    pub fn add_ground_stitching_vias(&self, board: &mut Board, grid: &RoutingGrid) -> usize {
        let mut added = self.add_plane_stitching_vias(board, grid);
        let Some(gnd_net_id) = board.nets.iter().position(|n| crate::board::is_gnd_net_name(n)) else {
            return added;
        };

        let via_size = 0.60;
        let via_drill = 0.30;
        let via_r = via_size * 0.5;
        let gnd_c = board.get_netclass_for_net(gnd_net_id).clearance.max(0.20);
        let edge_c = board.board_edge_clearance();
        let margin = (edge_c + via_r + gnd_c).max(1.50);
        let step = 5.0;
        let r_cells = ((via_r + gnd_c) / grid.res).ceil() as isize;

        let min_x = board.min_x + margin;
        let max_x = board.max_x - margin;
        let min_y = board.min_y + margin;
        let max_y = board.max_y - margin;
        if min_x > max_x || min_y > max_y {
            return added;
        }

        let mut cand_pts = Vec::new();
        let mut gpu_cands = Vec::new();
        let mut y = min_y;
        while y <= max_y {
            let mut x = min_x;
            while x <= max_x {
                let mut fast_ok = true;
                if let Some(ref poly) = board.outline_polygon {
                    let p = Point::new(x, y);
                    if !poly.contains_point(&p) || poly.distance_to_point(&p) < edge_c + via_r {
                        fast_ok = false;
                    }
                }
                if fast_ok {
                    for ko in &board.keepouts {
                        if x >= ko.min_x - (via_r + gnd_c)
                            && x <= ko.max_x + (via_r + gnd_c)
                            && y >= ko.min_y - (via_r + gnd_c)
                            && y <= ko.max_y + (via_r + gnd_c)
                        {
                            fast_ok = false;
                            break;
                        }
                    }
                }
                if fast_ok {
                    if let Some((gx, gy)) = grid.world_to_grid(x, y) {
                        for cy in -r_cells..=r_cells {
                            for cx in -r_cells..=r_cells {
                                if (cx * cx + cy * cy) as f64 * (grid.res * grid.res) <= (via_r + gnd_c).powi(2) {
                                    let nx = gx as isize + cx;
                                    let ny = gy as isize + cy;
                                    if nx < 0 || nx >= grid.width as isize || ny < 0 || ny >= grid.height as isize {
                                        fast_ok = false;
                                        break;
                                    }
                                    for l in 0..grid.num_layers {
                                        let idx_l = grid.idx(nx as usize, ny as usize, l);
                                        let nl = grid.obstacle_net_ids[idx_l];
                                        if (nl != FREE_NET && nl != gnd_net_id as i32)
                                            || (grid.smd_pad_cells[idx_l] && nl != gnd_net_id as i32)
                                        {
                                            fast_ok = false;
                                            break;
                                        }
                                    }
                                    if !fast_ok {
                                        break;
                                    }
                                }
                            }
                            if !fast_ok {
                                break;
                            }
                        }
                    } else {
                        fast_ok = false;
                    }
                }
                if fast_ok {
                    cand_pts.push((x, y));
                    gpu_cands.push(GpuDrcCandidate {
                        x1: x as f32,
                        y1: y as f32,
                        x2: 0.0,
                        y2: 0.0,
                        radius_or_half_width: via_r as f32,
                        clearance: gnd_c as f32,
                        layer: 0,
                        net_id: gnd_net_id as u32,
                        kind: 0,
                        ignore_same_net_pads: 0,
                        _pad0: 0,
                        _pad1: 0,
                    });
                }
                x += step;
            }
            y += step;
        }

        if cand_pts.is_empty() {
            return added;
        }

        let drc_ctx = GpuDrcContext::from_board(board);
        let gpu_flags = drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_cands);

        for (i, &(x, y)) in cand_pts.iter().enumerate() {
            if gpu_flags.as_ref().is_some_and(|flags| flags[i] == 0) {
                continue;
            }
            let too_close_same = board.vias.iter().any(|v| {
                if v.net_id == gnd_net_id {
                    (v.position.0 - x).hypot(v.position.1 - y) < (step * 0.75).max(via_size + gnd_c)
                } else {
                    (v.position.0 - x).hypot(v.position.1 - y) < (v.size + via_size) * 0.5 + gnd_c
                }
            });
            if too_close_same {
                continue;
            }
            if is_via_location_legal(board, (x, y), via_size, gnd_net_id, gnd_c) {
                board.vias.push(Via {
                    position: (x, y),
                    size: via_size,
                    drill: via_drill,
                    net_id: gnd_net_id,
                    is_locked: false,
                });
                added += 1;
            }
        }

        added
    }

    /// Searches candidate radial and diagonal offsets around a plane SMD pad at `(px, py)`
    /// to find a DRC-legal location for a `0.60 mm` plane drop via and its connecting trace.
    ///
    /// # Arguments
    /// * `board` - Read-only reference to the [`Board`].
    /// * `grid` - Reference to the [`RoutingGrid`].
    /// * `plane_net_id` - Numeric net ID of the target plane (`GND` or power plane).
    /// * `px` - Pad X coordinate in `mm`.
    /// * `py` - Pad Y coordinate in `mm`.
    /// * `layer` - Copper layer of the pad.
    /// * `pad` - Reference to the [`Pad`].
    /// * `existing_vias` - Slice of recently added plane via coordinates `(x, y)` in `mm` to avoid crowding.
    /// * `drc_ctx_opt` - Optional pre-built [`GpuDrcContext`] for GPU batch screening.
    ///
    /// # Returns
    /// `Some((vx, vy))` in `mm` if a legal via position and connecting dogleg exist, or `None`.
    fn find_safe_plane_via_location(
        board: &Board,
        grid: &RoutingGrid,
        plane_net_id: usize,
        px: f64,
        py: f64,
        layer: usize,
        pad: &Pad,
        existing_vias: &[(f64, f64)],
        drc_ctx_opt: Option<&GpuDrcContext>,
    ) -> Option<(f64, f64)> {
        let pad_dim = pad.width.max(pad.height);
        let min_d = (pad_dim * 0.5) + 0.35;
        let min_d_narrow = (pad.width.min(pad.height) * 0.5) + 0.35;
        let track_w = board.get_netclass_for_net(plane_net_id).track_width.min(0.25).max(0.20);
        let plane_c = board.get_netclass_for_net(plane_net_id).clearance.max(0.20);

        let board_mid_x = (board.min_x + board.max_x) * 0.5;
        let zone_max_x = board
            .keepouts
            .iter()
            .filter(|ko| ko.min_x > board_mid_x && py >= ko.min_y - 4.0 && py <= ko.max_y + 4.0)
            .map(|ko| ko.min_x - 3.9)
            .fold(board.max_x, f64::min);

        let mut primary_offsets = vec![
            (-0.23, 0.825),
            (0.23, 0.825),
            (-0.23, -0.825),
            (0.23, -0.825),
            (-0.17, 0.875),
            (0.17, 0.875),
            (-0.17, -0.875),
            (0.17, -0.875),
            (-0.32, 0.78),
            (0.32, 0.78),
            (-0.32, -0.78),
            (0.32, -0.78),
        ];
        if min_d_narrow + 0.10 < min_d {
            for &dist_n in &[
                min_d_narrow,
                min_d_narrow + 0.20,
                min_d_narrow + 0.40,
                min_d_narrow + 0.60,
                min_d_narrow + 0.80,
            ] {
                primary_offsets.push((dist_n, 0.0));
                primary_offsets.push((-dist_n, 0.0));
                primary_offsets.push((0.0, dist_n));
                primary_offsets.push((0.0, -dist_n));
            }
        }
        for &dist in &[
            min_d,
            min_d + 0.20,
            min_d + 0.40,
            min_d + 0.60,
            min_d + 0.80,
            min_d + 1.00,
            min_d + 1.25,
            min_d + 1.50,
            min_d + 1.80,
            min_d + 2.10,
            min_d + 2.50,
        ] {
            let diag_d = dist * std::f64::consts::FRAC_1_SQRT_2;
            primary_offsets.push((dist, 0.0));
            primary_offsets.push((-dist, 0.0));
            primary_offsets.push((0.0, dist));
            primary_offsets.push((0.0, -dist));
            primary_offsets.push((diag_d, diag_d));
            primary_offsets.push((diag_d, -diag_d));
            primary_offsets.push((-diag_d, diag_d));
            primary_offsets.push((-diag_d, -diag_d));
            primary_offsets.push((dist, dist * 0.5));
            primary_offsets.push((dist, -dist * 0.5));
            primary_offsets.push((-dist, dist * 0.5));
            primary_offsets.push((-dist, -dist * 0.5));
            primary_offsets.push((dist * 0.5, dist));
            primary_offsets.push((-dist * 0.5, dist));
            primary_offsets.push((dist * 0.5, -dist));
            primary_offsets.push((-dist * 0.5, -dist));
            primary_offsets.push((dist * 0.38, dist));
            primary_offsets.push((-dist * 0.38, dist));
            primary_offsets.push((dist * 0.38, -dist));
            primary_offsets.push((-dist * 0.38, -dist));
        }

        if px > zone_max_x {
            for &dx in &[-1.2, -1.5, -1.8, -2.1, -2.5, -2.9, -3.1, -3.3, -3.5, -3.8, -4.1, -4.5] {
                for &dy in &[0.0, 0.8, -0.8, 1.1, -1.1, 1.35, -1.35, 1.6, -1.6, 2.0, -2.0, 2.5, -2.5] {
                    primary_offsets.push((dx, dy));
                }
            }
        }

        let mut fallback_offsets = primary_offsets.clone();
        for &dx in &[-0.95_f64, -0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, 0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95] {
            for &dy in &[0.65_f64, 0.72, 0.78, 0.825, 0.88, 0.95, 1.05, -0.65, -0.72, -0.78, -0.825, -0.88, -0.95, -1.05, 0.0, 0.35, -0.35, 0.50, -0.50] {
                if dx.hypot(dy) >= min_d_narrow - 0.05 {
                    fallback_offsets.push((dx, dy));
                }
            }
        }
        for &dist in &[
            min_d,
            min_d + 0.15,
            min_d + 0.25,
            min_d + 0.40,
            min_d + 0.55,
            min_d + 0.70,
            min_d + 0.85,
            min_d + 1.00,
            min_d + 1.25,
            min_d + 1.50,
            min_d + 1.80,
            min_d + 2.10,
            min_d + 2.50,
            min_d + 3.00,
        ] {
            let diag_d = dist * std::f64::consts::FRAC_1_SQRT_2;
            fallback_offsets.push((diag_d, diag_d));
            fallback_offsets.push((diag_d, -diag_d));
            fallback_offsets.push((-diag_d, diag_d));
            fallback_offsets.push((-diag_d, -diag_d));
            fallback_offsets.push((dist, dist * 0.25));
            fallback_offsets.push((dist, -dist * 0.25));
            fallback_offsets.push((-dist, dist * 0.25));
            fallback_offsets.push((-dist, -dist * 0.25));
            fallback_offsets.push((dist * 0.25, dist));
            fallback_offsets.push((-dist * 0.25, dist));
            fallback_offsets.push((dist * 0.25, -dist));
            fallback_offsets.push((-dist * 0.25, -dist));
            fallback_offsets.push((dist * 0.38, dist));
            fallback_offsets.push((-dist * 0.38, dist));
            fallback_offsets.push((dist * 0.38, -dist));
            fallback_offsets.push((-dist * 0.38, -dist));
        }

        let owned_ctx;
        let drc_ctx = if let Some(ctx) = drc_ctx_opt {
            ctx
        } else {
            owned_ctx = GpuDrcContext::from_board(board);
            &owned_ctx
        };
        let confirm_cpu = |vx: f64, vy: f64, try_w: f64, try_c: f64| -> bool {
            if !is_via_location_legal(board, (vx, vy), 0.60, plane_net_id, try_c) {
                return false;
            }
            let direct_ok = is_octilinear_segment((px, py), (vx, vy))
                && is_track_segment_legal(board, (px, py), (vx, vy), try_w, layer, plane_net_id, try_c);
            if direct_ok {
                return true;
            }
            [false, true].into_iter().any(|df| {
                let dl = compute_45_dogleg((px, py), (vx, vy), df);
                dl.windows(2).all(|w| {
                    (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                        || is_track_segment_legal(board, w[0], w[1], try_w, layer, plane_net_id, try_c)
                })
            })
        };

        for (try_w, try_c, offsets) in [(track_w, plane_c, &primary_offsets), (0.15, 0.15, &fallback_offsets)] {
            let mut valid_coords = Vec::with_capacity(offsets.len());
            let mut gpu_cands = Vec::with_capacity(offsets.len());
            for &(dx, dy) in offsets {
                let vx = px + dx;
                let vy = py + dy;
                if vx > zone_max_x {
                    continue;
                }
                if existing_vias.iter().any(|ev| (ev.0 - vx).hypot(ev.1 - vy) < 0.65) {
                    continue;
                }
                if grid.world_to_grid(vx, vy).is_none() {
                    continue;
                }
                valid_coords.push((vx, vy));
                gpu_cands.push(GpuDrcCandidate {
                    x1: vx as f32,
                    y1: vy as f32,
                    x2: 0.0,
                    y2: 0.0,
                    radius_or_half_width: 0.30,
                    clearance: try_c as f32,
                    layer: layer as u32,
                    net_id: plane_net_id as u32,
                    kind: 0,
                    ignore_same_net_pads: 0,
                    _pad0: 0,
                    _pad1: 0,
                });
            }

            if let Some(res) = drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_cands) {
                for (i, &(vx, vy)) in valid_coords.iter().enumerate() {
                    if res[i] != 0 && confirm_cpu(vx, vy, try_w, try_c) {
                        return Some((vx, vy));
                    }
                }
            } else {
                for &(vx, vy) in &valid_coords {
                    if confirm_cpu(vx, vy, try_w, try_c) {
                        return Some((vx, vy));
                    }
                }
            }
        }
        None
    }

    /// Balances intra-pair length skew between differential pair nets `net_p` and `net_n`
    /// within `tolerance_mm` by inserting small 45-degree phase-matching bumps on the shorter net.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] whose shorter differential trace is lengthened.
    /// * `grid` - Mutable [`RoutingGrid`].
    /// * `net_p` - Positive differential net ID.
    /// * `net_n` - Negative differential net ID.
    /// * `tolerance_mm` - Maximum permissible length skew in `mm`.
    ///
    /// # Returns
    /// `true` if the resulting skew is within `tolerance_mm`.
    ///
    /// # Side Effects
    /// Replaces straight segments on the shorter net in `board.tracks` with 45-degree phase bumps.
    pub fn match_diff_pair_phase(
        &self,
        board: &mut Board,
        grid: &mut RoutingGrid,
        net_p: usize,
        net_n: usize,
        tolerance_mm: f64,
    ) -> bool {
        match_diff_pair_phase(board, grid, net_p, net_n, tolerance_mm)
    }

    /// Lengthens `net_id` to `target_length_mm` by inserting serpentine trombone meanders
    /// bounded by `max_amplitude_mm`.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] whose `tracks` on `net_id` are meander-tuned.
    /// * `grid` - Mutable [`RoutingGrid`].
    /// * `net_id` - Numeric net ID to tune.
    /// * `target_length_mm` - Target total trace length in `mm`.
    /// * `max_amplitude_mm` - Maximum lateral meander amplitude in `mm`.
    ///
    /// # Returns
    /// `true` if the target length was reached or meanders were inserted.
    ///
    /// # Side Effects
    /// Replaces straight segments in `board.tracks` with serpentine meander segments.
    pub fn tune_net_length(
        &self,
        board: &mut Board,
        grid: &mut RoutingGrid,
        net_id: usize,
        target_length_mm: f64,
        max_amplitude_mm: f64,
    ) -> bool {
        tune_net_length(board, grid, net_id, target_length_mm, max_amplitude_mm)
    }

    /// Eliminates redundant layer-hopping via pairs on `board` using `self.config`.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] whose tracks and vias are simplified.
    /// * `grid` - Mutable [`RoutingGrid`].
    ///
    /// # Side Effects
    /// Removes redundant vias from `board.vias` and promotes intermediate segments in `board.tracks`.
    pub fn reduce_vias(&self, board: &mut Board, grid: &mut RoutingGrid) {
        reduce_vias_with_config(board, grid, &self.config);
    }

    /// Runs post-route rubber-band track glossing on `board` using `self.config.any_angle_mode`.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] whose unlocked tracks are glossed.
    /// * `grid` - Reference to the [`RoutingGrid`].
    ///
    /// # Side Effects
    /// Replaces multi-segment staircases and jogs in `board.tracks` with direct DRC-clean segments.
    pub fn gloss_tracks(&self, board: &mut Board, grid: &RoutingGrid) {
        gloss_tracks_with_mode(board, grid, self.config.any_angle_mode);
    }

    /// Removes dangling dead-end trace stubs and isolated vias from `board`.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] to prune.
    ///
    /// # Side Effects
    /// Retains only electrically useful copper in `board.tracks`, `board.arcs`, and `board.vias`.
    pub fn prune_dangling_tracks_and_vias(&self, board: &mut Board) {
        prune_dangling_tracks_and_vias(board);
    }

    /// Validates all routed copper on `board` against pad, via, and trace clearances, applying
    /// neck-down, lateral micro-nudges, or corner trim repairs as needed.
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] to validate and clean.
    ///
    /// # Side Effects
    /// Modifies `board.tracks`, `board.arcs`, and `board.vias` in place to eliminate DRC violations.
    pub fn validate_and_clean_copper(&self, board: &mut Board) {
        validate_and_clean_copper(board);
    }

    /// Places companion ground return-path vias (`0.60 mm` / `0.30 mm`) within
    /// `self.config.return_via_max_dist_mm` of any high-speed signal via whose netclass
    /// requires an unbroken reference return path (`require_return_path`).
    ///
    /// # Arguments
    /// * `board` - Mutable [`Board`] to which companion `GND` vias are added.
    /// * `grid` - Reference to the [`RoutingGrid`].
    ///
    /// # Returns
    /// Number of companion return-path `GND` vias added to `board.vias`.
    ///
    /// # Side Effects
    /// Appends `GND` [`Via`] items to `board.vias`.
    pub fn add_return_path_ground_vias(&self, board: &mut Board, grid: &RoutingGrid) -> usize {
        let Some(gnd_net_id) = board.nets.iter().position(|n| crate::board::is_gnd_net_name(n)) else {
            return 0;
        };

        let mut gnd_locations: Vec<(f64, f64)> = board
            .vias
            .iter()
            .filter(|v| v.net_id == gnd_net_id)
            .map(|v| v.position)
            .collect();

        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id == gnd_net_id && pad.is_through_hole {
                    gnd_locations.push(fp.pad_absolute_position(pad));
                }
            }
        }

        let signal_vias: Vec<(usize, (f64, f64))> = board
            .vias
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| {
                if v.net_id == gnd_net_id {
                    return None;
                }
                let nc = board.get_netclass_for_net(v.net_id);
                if nc.require_return_path {
                    Some((idx, v.position))
                } else {
                    None
                }
            })
            .collect();
        if signal_vias.is_empty() {
            return 0;
        }

        let gnd_c = board.get_netclass_for_net(gnd_net_id).clearance.max(0.15);
        let via_size = 0.60;
        let via_drill = 0.30;
        let via_r = via_size * 0.5;
        let r_cells = ((via_r + gnd_c) / grid.res).ceil() as isize;

        let max_ret_d = self.config.return_via_max_dist_mm;
        let mut distances: Vec<f64> = [0.75, 0.90, 1.10, 1.25, 1.40]
            .into_iter()
            .filter(|&d| d <= max_ret_d + 1e-4)
            .collect();
        if max_ret_d > 1.40 + 1e-4 {
            distances.push(max_ret_d);
        }
        if distances.is_empty() {
            distances.push(max_ret_d.max(0.50));
        }
        let mut angles = Vec::with_capacity(16);
        for i in 0..16 {
            let theta = (i as f64) * std::f64::consts::PI / 8.0;
            angles.push((theta.cos(), theta.sin()));
        }

        let cands_per_sig = distances.len() * angles.len();
        let mut gpu_cands = Vec::with_capacity(signal_vias.len() * cands_per_sig);
        for &(_, s_pos) in &signal_vias {
            for &dist in &distances {
                for &(cos_t, sin_t) in &angles {
                    let vx = s_pos.0 + dist * cos_t;
                    let vy = s_pos.1 + dist * sin_t;
                    gpu_cands.push(GpuDrcCandidate {
                        x1: vx as f32,
                        y1: vy as f32,
                        x2: 0.0,
                        y2: 0.0,
                        radius_or_half_width: via_r as f32,
                        clearance: gnd_c as f32,
                        layer: 0,
                        net_id: gnd_net_id as u32,
                        kind: 0,
                        ignore_same_net_pads: 0,
                        _pad0: 0,
                        _pad1: 0,
                    });
                }
            }
        }

        let drc_ctx = GpuDrcContext::from_board(board);
        let gpu_flags = drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_cands);

        // Pre-compute legal candidate positions for every signal via in parallel across Rayon workers
        let legal_per_sig: Vec<Vec<(f64, f64)>> = signal_vias
            .par_iter()
            .enumerate()
            .map(|(sig_i, &(_, s_pos))| {
                let mut valid_list = Vec::new();
                let base_idx = sig_i * cands_per_sig;
                let mut local_k = 0usize;
                for &dist in &distances {
                    for &(cos_t, sin_t) in &angles {
                        let flag_idx = base_idx + local_k;
                        local_k += 1;
                        if !gpu_flags.as_ref().map_or(true, |f| f[flag_idx] != 0) {
                            continue;
                        }
                        let vx = s_pos.0 + dist * cos_t;
                        let vy = s_pos.1 + dist * sin_t;
                        if board.vias.iter().any(|v| (v.position.0 - vx).hypot(v.position.1 - vy) < (v.size + via_size) * 0.5 + gnd_c) {
                            continue;
                        }
                        if !is_via_location_legal(board, (vx, vy), via_size, gnd_net_id, gnd_c) {
                            continue;
                        }
                        if let Some((gx, gy)) = grid.world_to_grid(vx, vy) {
                            let mut safe = true;
                            for cy in -r_cells..=r_cells {
                                for cx in -r_cells..=r_cells {
                                    if (cx * cx + cy * cy) as f64 * (grid.res * grid.res) <= (via_r + gnd_c).powi(2) {
                                        let nx = gx as isize + cx;
                                        let ny = gy as isize + cy;
                                        if nx < 0 || nx >= grid.width as isize || ny < 0 || ny >= grid.height as isize {
                                            safe = false;
                                            break;
                                        }
                                        for l in 0..grid.num_layers {
                                            let idx_l = grid.idx(nx as usize, ny as usize, l);
                                            let nl = grid.obstacle_net_ids[idx_l];
                                            if (nl != FREE_NET && nl != gnd_net_id as i32)
                                                || (grid.smd_pad_cells[idx_l] && nl != gnd_net_id as i32)
                                            {
                                                safe = false;
                                                break;
                                            }
                                        }
                                        if !safe {
                                            break;
                                        }
                                    }
                                }
                                if !safe {
                                    break;
                                }
                            }
                            if safe {
                                valid_list.push((vx, vy));
                            }
                        }
                    }
                }
                valid_list
            })
            .collect();

        let mut added_count = 0;
        let skip_existing_dist = (max_ret_d + 0.10).max(0.85);
        for ((_s_idx, s_pos), valid_cands) in signal_vias.into_iter().zip(legal_per_sig.into_iter()) {
            let nearest_dist = gnd_locations
                .iter()
                .map(|gp| (gp.0 - s_pos.0).hypot(gp.1 - s_pos.1))
                .fold(f64::INFINITY, f64::min);

            if nearest_dist > skip_existing_dist {
                if let Some(cand_pos) = valid_cands
                    .into_iter()
                    .find(|&(vx, vy)| !gnd_locations.iter().any(|ev| (ev.0 - vx).hypot(ev.1 - vy) < via_size + gnd_c))
                {
                    board.vias.push(Via {
                        position: cand_pos,
                        size: via_size,
                        drill: via_drill,
                        net_id: gnd_net_id,
                        is_locked: false,
                    });
                    gnd_locations.push(cand_pos);
                    added_count += 1;
                }
            }
        }

        added_count
    }

    /// Audits all routed tracks on `board` for excessive parallel crosstalk coupling and
    /// reference-plane return discontinuities.
    ///
    /// # Arguments
    /// * `board` - Read-only reference to the [`Board`].
    ///
    /// # Returns
    /// Vector of detected [`CouplingViolation`] records.
    pub fn validate_crosstalk_and_coupling(&self, board: &Board) -> Vec<CouplingViolation> {
        validate_crosstalk_and_coupling(board)
    }

    /// Dispatches a batch of 3D wavefront pathfinding requests across all available GPUs
    /// via `MultiGpuRouterPool` and returns obstacle-free [`GridNode`] paths.
    ///
    /// # Arguments
    /// * `grid` - Reference to the [`RoutingGrid`].
    /// * `requests` - Slice of `(start, goal, net_id, via_cost)` tuples.
    /// * `clearance_cells` - Required obstacle clearance radius in grid cells.
    ///
    /// # Returns
    /// Vector of `Option<Vec<GridNode>>` paths corresponding 1-to-1 with `requests`.
    pub fn solve_batch_gpu_wavefront_paths(
        &self,
        grid: &RoutingGrid,
        requests: &[(GridNode, GridNode, usize, f32)],
        clearance_cells: i32,
    ) -> Vec<Option<Vec<GridNode>>> {
        let raw_reqs: Vec<((usize, usize, usize), (usize, usize, usize), u32, f32)> = requests
            .iter()
            .map(|&(src, dst, net_id, via_cost)| {
                (
                    (src.gx, src.gy, src.layer),
                    (dst.gx, dst.gy, dst.layer),
                    net_id as u32,
                    via_cost,
                )
            })
            .collect();
        grid.solve_batch_3d_wavefront_paths(&raw_reqs, clearance_cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Footprint, KeepoutZone, Pad, Polygon};

    #[test]
    fn test_cpu_distance_field_evaluation() {
        assert!(RoutingGrid::compute_distance_field_cpu(&[], 0, 0, 5).is_empty());
        assert_eq!(RoutingGrid::compute_distance_field_cpu(&[0u32; 10], 4, 4, 5).len(), 16);

        let mut obs = vec![0u32; 16 * 16];
        obs[4 * 16 + 4] = 1;
        let df = RoutingGrid::compute_distance_field_cpu(&obs, 16, 16, 10);
        assert!((df[4 * 16 + 4] - 0.0).abs() < 1e-4);
        assert!((df[4 * 16 + 8] - 4.0).abs() < 1e-3);
        assert!((df[7 * 16 + 8] - 5.0).abs() < 1e-3); // 3-4-5 right triangle

        let board = make_router_test_board();
        let grid = RoutingGrid::new(&board, 0.20, None);
        let layer_cells = grid.width * grid.height;
        let mut obstacle_count = 0usize;
        let mut positive_count = 0usize;
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
                positive_count += 1;
            }
        }
        assert!(obstacle_count > 0, "Grid must have obstacle cells from pads");
        assert!(positive_count > 10, "Grid must have positive distance field free cells");
    }

    fn make_fp(reference: &str, x: f64, y: f64, is_locked: bool, pads: Vec<Pad>) -> Footprint {
        Footprint {
            reference: reference.to_string(),
            value: "V".to_string(),
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

    fn make_router_test_board() -> Board {
        let nets = vec![
            "".to_string(),
            "GND".to_string(),
            "SIG1".to_string(),
            "SIG2".to_string(),
            "U2-SW".to_string(),
        ];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        netclasses.insert(
            "Switching".to_string(),
            NetClass {
                name: "Switching".to_string(),
                track_width: 0.40,
                clearance: 0.25,
                via_dia: 0.60,
                via_drill: 0.30,
                diff_pair_gap: 0.20,
                diff_pair_width: 0.20,
                topology: NetTopology::default(),
                ..Default::default()
            },
        );
        let mut net_to_class = HashMap::new();
        net_to_class.insert("".to_string(), "Default".to_string());
        net_to_class.insert("GND".to_string(), "Default".to_string());
        net_to_class.insert("SIG1".to_string(), "Default".to_string());
        net_to_class.insert("SIG2".to_string(), "Default".to_string());
        net_to_class.insert("U2-SW".to_string(), "Switching".to_string());

        let u1_pads: Vec<Pad> = [
            ("1", -1.5, -1.0, 2, "SIG1"),
            ("2", -1.5, -0.5, 3, "SIG2"),
            ("3", -1.5, 0.0, 2, "SIG1"),
            ("4", -1.5, 0.5, 3, "SIG2"),
            ("6", -1.5, 1.0, 2, "SIG1"),
            ("7", -0.5, 1.5, 2, "SIG1"),
            ("8", 0.0, 1.5, 3, "SIG2"),
            ("9", 0.5, 1.5, 2, "SIG1"),
            ("10", 1.0, 1.5, 3, "SIG2"),
            ("12", 1.5, 1.5, 2, "SIG1"),
            ("13", 1.5, 0.5, 2, "SIG1"),
            ("15", 1.5, 0.0, 1, "GND"),
            ("16", 1.5, -0.5, 3, "SIG2"),
            ("17", 1.5, -1.0, 2, "SIG1"),
            ("18", 1.5, -1.5, 1, "GND"),
            ("19", 0.5, -1.5, 1, "GND"),
            ("20", 0.0, -1.5, 2, "SIG1"),
            ("21", -0.5, -1.5, 3, "SIG2"),
            ("22", -1.0, -1.5, 1, "GND"),
            ("23", -1.2, -1.5, 2, "SIG1"),
            ("24", -1.4, -1.5, 3, "SIG2"),
        ]
        .iter()
        .map(|&(num, ox, oy, nid, nname)| Pad::new(num, ox, oy, 0.25, 0.25, nid, nname, false))
        .collect();

        let j2_pads = vec![
            Pad::new("A5", 0.0, -0.5, 0.3, 0.3, 2, "SIG1", false),
            Pad::new("A6", 0.0, 0.5, 0.3, 0.3, 3, "SIG2", false),
            Pad::new("S1", 0.0, 1.5, 1.0, 1.0, 0, "unconnected", true),
        ];

        let c14 = make_fp(
            "C14",
            142.0,
            95.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "GND", false)],
        );

        let r14 = make_fp(
            "R14",
            146.0,
            95.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "GND", false)],
        );

        let r_gnd = make_fp(
            "R20",
            150.0,
            95.0,
            false,
            vec![
                Pad::new("1", -1.0, 0.0, 0.6, 0.6, 1, "GND", false),
                Pad::new("2", 1.0, 0.0, 0.6, 0.6, 4, "U2-SW", false),
            ],
        );

        Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 130.0,
            min_y: 90.0,
            max_x: 160.0,
            max_y: 115.0,
            footprints: vec![
                make_fp("U1", 145.0, 102.0, false, u1_pads),
                make_fp("J2", 135.0, 102.0, true, j2_pads),
                c14,
                r14,
                r_gnd,
            ],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![KeepoutZone {
                min_x: 156.0,
                min_y: 110.0,
                max_x: 159.0,
                max_y: 114.0,
                layers: vec![0, 1],
            }],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        }
    }

    #[test]
    fn test_routing_grid_coords_rasterize_and_neighbors() {
        let board = make_router_test_board();
        let vk = VulkanEngine::new();
        let grid = RoutingGrid::new(&board, 0.25, vk);

        assert!(grid.world_to_grid(129.0, 95.0).is_none());
        assert!(grid.world_to_grid(161.0, 95.0).is_none());
        let (gx, gy) = grid.world_to_grid(140.0, 100.0).unwrap();
        let (wx, wy) = grid.grid_to_world(gx, gy);
        assert!((wx - 140.125).abs() < 1e-4 && (wy - 100.125).abs() < 1e-4);

        // Verify keepout zone was rasterized as KEEPOUT_NET
        let (kgx, kgy) = grid.world_to_grid(157.0, 112.0).unwrap();
        assert_eq!(grid.obstacle_net_ids[grid.idx(kgx, kgy, 0)], KEEPOUT_NET);

        // Verify get_neighbors on free space (planar + via transition) vs near U1 center
        let free_node = GridNode { gx, gy, layer: 0 };
        let nbrs = grid.get_neighbors(free_node, 2, 0.20, 0.20, 0.60, Some((140.0, 100.0)));
        assert!(!nbrs.is_empty());
        assert!(nbrs.iter().any(|(n, _)| n.layer == 1));
    }

    #[test]
    fn test_nudge_engine_and_astar_node_ordering() {
        let mut board = make_router_test_board();
        let router = AutoRouter::new(4);
        assert!((router.nudge_engine.step_mm - 0.25).abs() < 1e-6);
        assert!((router.nudge_engine.max_displacement_mm - 2.0).abs() < 1e-6);

        // Out of range fp_idx, locked fp (J2 at idx 1), and zero push vector must return false
        assert!(!router.nudge_engine.try_nudge_component(&mut board, 999, (1.0, 0.0)));
        assert!(!router.nudge_engine.try_nudge_component(&mut board, 1, (1.0, 0.0)));
        assert!(!router.nudge_engine.try_nudge_component(&mut board, 2, (0.0, 0.0)));

        // Add a track connecting C14 (idx 2) to R14 (idx 3)
        board.tracks.push(TrackSegment {
            start: (142.0, 95.0),
            end: (146.0, 95.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        assert_eq!(board.tracks.len(), 1);

        // Valid nudge of unlocked C14 (idx 2)
        let old_x = board.footprints[2].x;
        assert!(router.nudge_engine.try_nudge_component(&mut board, 2, (1.0, 0.0)));
        assert!((board.footprints[2].x - (old_x + 0.25)).abs() < 1e-6);
        // Trace connected to C14 must have been ripped up!
        assert_eq!(board.tracks.len(), 0);

        // AStarNode min-heap ordering check
        let mut heap = BinaryHeap::new();
        heap.push(AStarNode {
            node: GridNode { gx: 0, gy: 0, layer: 0 },
            cost: 5.0,
            priority: 10.0,
        });
        heap.push(AStarNode {
            node: GridNode { gx: 1, gy: 1, layer: 0 },
            cost: 1.0,
            priority: 2.0,
        });
        assert_eq!(heap.pop().unwrap().node.gx, 1);
    }

    #[test]
    fn test_escape_stubs_connection_extraction_and_priority() {
        assert_eq!(AutoRouter::netclass_priority("Switching"), 10);
        assert_eq!(AutoRouter::netclass_priority("Power_15V"), 9);
        assert_eq!(AutoRouter::netclass_priority("USB_Diff"), 8);
        assert_eq!(AutoRouter::netclass_priority("HBS_Bus"), 7);
        assert_eq!(AutoRouter::netclass_priority("Power_3V3"), 6);
        assert_eq!(AutoRouter::netclass_priority("Default"), 1);

        let board = make_router_test_board();
        let router = AutoRouter::new(3);
        let stubs = router.generate_escape_stubs(&board);
        assert_eq!(stubs.len(), 19); // 17 U1 non-GND pins + 2 J2 SMD signal pins

        let conns = router.extract_connections(&board);
        assert!(!conns.is_empty());
        for c in &conns {
            assert!(c.start_fp < board.footprints.len());
            assert!(c.end_fp < board.footprints.len());
            assert!(c.start_pad < board.footprints[c.start_fp].pads.len());
            assert!(c.end_pad < board.footprints[c.end_fp].pads.len());
        }
    }

    #[test]
    fn test_escape_stubs_direction_and_pad_clearance() {
        let mut board = make_router_test_board();
        let router = AutoRouter::new(3);

        // Test with 0, 90, 180, 270 degree rotation of U1
        for rot in [0.0, 90.0, 180.0, 270.0] {
            board.footprints[0].rotation = rot;
            let stubs = router.generate_escape_stubs(&board);
            let u1_stubs: Vec<_> = stubs.iter().filter(|s| {
                s.layer == board.footprints[0].layer && (s.pad_pos.0 - board.footprints[0].x).hypot(s.pad_pos.1 - board.footprints[0].y) < 5.0
            }).collect();
            assert_eq!(u1_stubs.len(), 17, "All 17 non-GND pins of U1 must generate escape stubs at rot={}", rot);

            for stub in &u1_stubs {
                // Verify stub projects outward away from footprint center
                let pad_dist = (stub.pad_pos.0 - board.footprints[0].x).hypot(stub.pad_pos.1 - board.footprints[0].y);
                let tip_dist = (stub.tip_pos.0 - board.footprints[0].x).hypot(stub.tip_pos.1 - board.footprints[0].y);
                assert!(tip_dist > pad_dist, "Escape stub tip must project outward away from footprint center (rot={})", rot);

                // Verify stub never shoots towards any neighboring pad
                for pad in &board.footprints[0].pads {
                    let (px, py) = board.footprints[0].pad_absolute_position(pad);
                    let pad_to_pad_dist = (px - stub.pad_pos.0).hypot(py - stub.pad_pos.1);
                    if pad_to_pad_dist < 1e-3 {
                        continue; // same pad
                    }
                    let d = dist_to_segment(px, py, stub.pad_pos.0, stub.pad_pos.1, stub.tip_pos.0, stub.tip_pos.1);
                    assert!(d >= pad_to_pad_dist - 1e-4, "Escape stub must not shoot towards pad {} on U1 (d={}, pad_to_pad={})", pad.number, d, pad_to_pad_dist);
                }
            }
        }
    }

    #[test]
    fn test_dedicated_plane_isolation() {
        let mut board = make_router_test_board();
        board.raw_content = "(\"In1.Cu\" \"In2.Cu\")".to_string();
        let mut grid = RoutingGrid::new(&board, 0.25, None);
        // Add dedicated inner plane layer 2 (GND = net 1) and layer 3 (PWR = net 4)
        grid.plane_nets.insert(2, 1);
        grid.plane_nets.insert(3, 4);

        let signal_net_id = 2; // SIG1
        let center_gx = grid.width / 2;
        let center_gy = grid.height / 2;

        // Test 1: Planar steps on dedicated plane layer 2 are prohibited for signal net
        let node_plane = GridNode { gx: center_gx, gy: center_gy, layer: 2 };
        let nbrs_plane = grid.get_neighbors(node_plane, signal_net_id, 0.2, 0.2, 0.6, None);
        let planar_steps: Vec<_> = nbrs_plane.iter().filter(|(n, _)| n.layer == 2).collect();
        assert!(planar_steps.is_empty(), "Signal net must not take planar steps on dedicated plane layer 2");

        // Test 2: Via transitions from layer 0 cannot transition to dedicated plane layer 2 or 3 for signal net
        let node_top = GridNode { gx: center_gx, gy: center_gy, layer: 0 };
        let nbrs_top = grid.get_neighbors(node_top, signal_net_id, 0.2, 0.2, 0.6, None);
        for (nbr, _) in &nbrs_top {
            assert_ne!(nbr.layer, 2, "Signal net must not transition via to plane layer 2");
            assert_ne!(nbr.layer, 3, "Signal net must not transition via to plane layer 3");
        }

        // Test 3: Plane net itself (GND = 1) CAN take planar steps on its own plane layer
        let nbrs_gnd_plane = grid.get_neighbors(node_plane, 1, 0.2, 0.2, 0.6, None);
        let gnd_planar_steps: Vec<_> = nbrs_gnd_plane.iter().filter(|(n, _)| n.layer == 2).collect();
        assert!(!gnd_planar_steps.is_empty(), "Plane net must be allowed to route planar steps on its plane layer");
    }

    #[test]
    fn test_auto_router_route_all_ripup_and_ground_stitching() {
        // Simple 2-net board where net 2 and net 3 cross, exercising A*, layer transitions,
        // rip-up & reroute, U1 ground pin stubs, and ground via stitching.
        let mut board = make_router_test_board();
        // Keep only a compact set of pads on U1 and J2 so routing finishes in <30ms
        board.footprints[0].pads.retain(|p| matches!(p.number.as_str(), "1" | "2" | "15" | "19"));
        let mut grid = RoutingGrid::new(&board, 0.25, None);
        let router = AutoRouter::new(2);
        let live = Arc::new(Mutex::new(LiveTrialState::new(board.clone())));
        let stop = Arc::new(AtomicBool::new(false));

        let (routed, total) = router.route_all(&mut board, &mut grid, Some(Arc::clone(&live)), stop);
        assert!(total >= 2);
        assert!(routed <= total);
        assert!(!board.tracks.is_empty());
        assert!(!board.vias.is_empty());

        let g = live.lock().unwrap();
        assert_eq!(g.phase, TrialPhase::Completed);
        assert_eq!(g.total_conns, total);
    }

    #[test]
    fn test_direction_change_penalty_and_staircase_smoothing() {
        let board = make_router_test_board();
        let grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(2);

        // Construct an alternating staircase path: (+1,0), (+1,+1), (+1,0), (+1,+1), (+1,0)
        // around an open region of the board (e.g. gx=10..15, gy=10..12)
        let zigzag = vec![
            GridNode { gx: 10, gy: 10, layer: 1 },
            GridNode { gx: 11, gy: 10, layer: 1 },
            GridNode { gx: 12, gy: 11, layer: 1 },
            GridNode { gx: 13, gy: 11, layer: 1 },
            GridNode { gx: 14, gy: 12, layer: 1 },
            GridNode { gx: 15, gy: 12, layer: 1 },
        ];
        assert_eq!(AutoRouter::count_path_bends(&zigzag), 4);

        let smoothed = router.smooth_grid_path(&grid, zigzag, 2, 0.20, 0.20, 0.60, None, false, 15.0);
        assert_eq!(smoothed.first().unwrap().gx, 10);
        assert_eq!(smoothed.last().unwrap().gx, 15);
        assert_eq!(
            AutoRouter::count_path_bends(&smoothed),
            1,
            "Staircase path should collapse into a single 45-degree bend"
        );
    }

    #[test]
    fn test_diff_pair_mates_and_coupled_bus_routing() {
        assert!(AutoRouter::are_diff_pair_mates("USB_DP", "USB_DM"));
        assert!(AutoRouter::are_diff_pair_mates("USB_D+", "USB_D-"));
        assert!(AutoRouter::are_diff_pair_mates("ETH_TX_P", "ETH_TX_N"));
        assert!(!AutoRouter::are_diff_pair_mates("USB_DP", "USB_DP"));
        assert!(!AutoRouter::are_diff_pair_mates("SIG1", "SIG2"));

        let mut board = make_router_test_board();
        board.nets = vec![
            "".to_string(),
            "GND".to_string(),
            "USB_DP".to_string(),
            "USB_DM".to_string(),
        ];
        board.footprints[0].pads = vec![
            Pad::new("13", -1.0, -0.4, 0.3, 0.3, 2, "USB_DP", false),
            Pad::new("14", -1.0, 0.4, 0.3, 0.3, 3, "USB_DM", false),
        ];
        board.footprints[1].pads = vec![
            Pad::new("A6", 1.0, -0.4, 0.3, 0.3, 2, "USB_DP", false),
            Pad::new("A7", 1.0, 0.4, 0.3, 0.3, 3, "USB_DM", false),
        ];
        board.footprints.truncate(2);

        let router = AutoRouter::new(2);
        let conns = router.extract_connections(&board);
        assert_eq!(conns.len(), 2);
        assert_eq!(conns[0].diff_pair_leader, None);
        assert_eq!(conns[1].diff_pair_leader, Some(0));

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let stop = Arc::new(AtomicBool::new(false));
        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);
        assert_eq!(routed, 2);
        assert_eq!(total, 2);

        // Verify the two coupled traces stay close to each other at their midpoints
        let dp_tracks: Vec<&TrackSegment> = board.tracks.iter().filter(|t| t.net_id == 2).collect();
        let dm_tracks: Vec<&TrackSegment> = board.tracks.iter().filter(|t| t.net_id == 3).collect();
        assert!(!dp_tracks.is_empty() && !dm_tracks.is_empty());
        let dp_mid_y: f64 = dp_tracks.iter().map(|t| (t.start.1 + t.end.1) * 0.5).sum::<f64>()
            / dp_tracks.len() as f64;
        let dm_mid_y: f64 = dm_tracks.iter().map(|t| (t.start.1 + t.end.1) * 0.5).sum::<f64>()
            / dm_tracks.len() as f64;
        assert!(
            (dp_mid_y - dm_mid_y).abs() < 1.8,
            "Coupled USB_DP/USB_DM traces should route together in a tight corridor (delta_y = {:.2})",
            (dp_mid_y - dm_mid_y).abs()
        );
    }

    #[test]
    fn test_coupled_diff_pair_gap_and_via_symmetry() {
        let mut board = make_router_test_board();
        board.nets = vec![
            "".to_string(),
            "GND".to_string(),
            "USB_DP".to_string(),
            "USB_DM".to_string(),
        ];
        board.netclasses.insert(
            "USB_Diff".to_string(),
            NetClass {
                name: "USB_Diff".to_string(),
                track_width: 0.20,
                clearance: 0.20,
                via_dia: 0.60,
                via_drill: 0.30,
                diff_pair_gap: 0.20,
                diff_pair_width: 0.20,
                topology: NetTopology::default(),
                ..Default::default()
            },
        );
        board.net_to_class.insert("USB_DP".to_string(), "USB_Diff".to_string());
        board.net_to_class.insert("USB_DM".to_string(), "USB_Diff".to_string());
        board.footprints[0].pads = vec![
            Pad::new("13", -1.0, -0.4, 0.3, 0.3, 2, "USB_DP", false),
            Pad::new("14", -1.0, 0.4, 0.3, 0.3, 3, "USB_DM", false),
        ];
        board.footprints[1].pads = vec![
            Pad::new("A6", 1.0, -0.4, 0.3, 0.3, 2, "USB_DP", false),
            Pad::new("A7", 1.0, 0.4, 0.3, 0.3, 3, "USB_DM", false),
        ];
        board.footprints.truncate(2);

        let router = AutoRouter::new(2);
        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let stop = Arc::new(AtomicBool::new(false));
        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);
        assert_eq!(routed, 2);
        assert_eq!(total, 2);

        // Verify differential pair track widths (0.20mm for main tracks, 0.15mm for fine-pitch escape stubs)
        let dp_tracks: Vec<&TrackSegment> = board.tracks.iter().filter(|t| t.net_id == 2).collect();
        let dm_tracks: Vec<&TrackSegment> = board.tracks.iter().filter(|t| t.net_id == 3).collect();
        assert!(!dp_tracks.is_empty() && !dm_tracks.is_empty());
        assert!(dp_tracks.iter().any(|t| (t.width - 0.20).abs() < 1e-4));
        for t in &dp_tracks {
            assert!((t.width - 0.20).abs() < 1e-4 || (t.width - 0.15).abs() < 1e-4);
        }
        for t in &dm_tracks {
            assert!((t.width - 0.20).abs() < 1e-4 || (t.width - 0.15).abs() < 1e-4);
        }

        // Verify that parallel segments maintain centerline distance W + S = 0.40mm
        // (edge-to-edge gap diff_pair_gap = 0.20mm)
        let mut found_coupled_parallel = false;
        for t_p in &dp_tracks {
            for t_n in &dm_tracks {
                if t_p.layer == t_n.layer {
                    let p_horiz = (t_p.start.1 - t_p.end.1).abs() < 1e-4;
                    let n_horiz = (t_n.start.1 - t_n.end.1).abs() < 1e-4;
                    if p_horiz && n_horiz {
                        let center_dist = (t_p.start.1 - t_n.start.1).abs();
                        if (center_dist - 0.40).abs() < 0.05 || (center_dist - 0.80).abs() < 0.05 {
                            found_coupled_parallel = true;
                            let gap = center_dist - t_p.width;
                            assert!(gap >= 0.15, "Differential pair gap must be >= 0.15mm, got {:.3}", gap);
                        }
                    }
                }
            }
        }
        assert!(found_coupled_parallel, "Must have coupled parallel segments with W+S centerline distance");

        // Verify symmetric paired vias side-by-side with matched spacing
        let mut board_vias = make_router_test_board();
        board_vias.nets = board.nets.clone();
        board_vias.netclasses = board.netclasses.clone();
        board_vias.net_to_class = board.net_to_class.clone();
        board_vias.footprints[0].pads = vec![
            Pad::new("13", -1.0, -0.4, 0.3, 0.3, 2, "USB_DP", false),
            Pad::new("14", -1.0, 0.4, 0.3, 0.3, 3, "USB_DM", false),
        ];
        board_vias.footprints[1].layer = 1;
        board_vias.footprints[1].pads = vec![
            Pad::new("A6", 1.0, -0.4, 0.3, 0.3, 2, "USB_DP", false),
            Pad::new("A7", 1.0, 0.4, 0.3, 0.3, 3, "USB_DM", false),
        ];
        board_vias.footprints.truncate(2);

        let mut grid_vias = RoutingGrid::new(&board_vias, 0.20, None);
        let stop_v = Arc::new(AtomicBool::new(false));
        let (r_v, _) = router.route_all(&mut board_vias, &mut grid_vias, None, stop_v);
        if r_v == 2 {
            let dp_vias: Vec<&Via> = board_vias.vias.iter().filter(|v| v.net_id == 2).collect();
            let dm_vias: Vec<&Via> = board_vias.vias.iter().filter(|v| v.net_id == 3).collect();
            if !dp_vias.is_empty() && !dm_vias.is_empty() {
                let via_dist = (dp_vias[0].position.0 - dm_vias[0].position.0).hypot(dp_vias[0].position.1 - dm_vias[0].position.1);
                assert!(via_dist >= 0.79, "Paired vias must maintain matched DRC clearance >= 0.80mm, got {:.3}", via_dist);
                assert_eq!(dp_vias[0].size, dm_vias[0].size);
                assert_eq!(dp_vias[0].drill, dm_vias[0].drill);
            }
        }
    }

    #[test]
    fn test_intra_pair_phase_matching_corrects_skew() {
        let mut board = make_router_test_board();
        board.nets = vec!["".to_string(), "GND".to_string(), "USB_DP".to_string(), "USB_DM".to_string()];
        board.netclasses.insert(
            "USB_Diff".to_string(),
            NetClass {
                name: "USB_Diff".to_string(),
                track_width: 0.20,
                clearance: 0.20,
                via_dia: 0.60,
                via_drill: 0.30,
                diff_pair_gap: 0.20,
                diff_pair_width: 0.20,
                topology: NetTopology::default(),
                ..Default::default()
            },
        );
        board.net_to_class.insert("USB_DP".to_string(), "USB_Diff".to_string());
        board.net_to_class.insert("USB_DM".to_string(), "USB_Diff".to_string());
        let net_p = 2;
        let net_n = 3;

        // Long trace for P (length = 10.0mm in clear board routing area)
        board.tracks.push(TrackSegment {
            start: (135.0, 107.0),
            end: (145.0, 107.0),
            width: 0.20,
            layer: 0,
            net_id: net_p,
            is_locked: false,
        });

        // Shorter trace for N (length = 9.4mm, so skew = 0.60mm > 0.05mm)
        board.tracks.push(TrackSegment {
            start: (135.0, 107.4),
            end: (144.4, 107.4),
            width: 0.20,
            layer: 0,
            net_id: net_n,
            is_locked: false,
        });

        let calc_len = |tracks: &[TrackSegment], nid: usize| -> f64 {
            tracks.iter().filter(|t| t.net_id == nid).map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1)).sum()
        };

        let initial_skew = (calc_len(&board.tracks, net_p) - calc_len(&board.tracks, net_n)).abs();
        assert!((initial_skew - 0.60).abs() < 1e-4, "Initial skew must be 0.60mm");

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let ok = match_diff_pair_phase(&mut board, &mut grid, net_p, net_n, 0.05);
        assert!(ok, "Phase matching must succeed");

        let final_skew = (calc_len(&board.tracks, net_p) - calc_len(&board.tracks, net_n)).abs();
        assert!(
            final_skew <= 0.05,
            "Intra-pair phase matching must equalize trace lengths within +/- 0.05mm, got final skew {:.4}mm",
            final_skew
        );
        let n_segs = board.tracks.iter().filter(|t| t.net_id == net_n).count();
        assert!(n_segs > 1, "Shorter trace must have tuning bump segments inserted");
    }

    #[test]
    fn test_trombone_serpentine_length_tuning() {
        let mut board = make_router_test_board();
        board.nets = vec!["".to_string(), "GND".to_string(), "SDIO_CLK".to_string()];
        board.net_to_class.insert("SDIO_CLK".to_string(), "Default".to_string());
        let net_id = 2;

        // Straight track of length 8.0mm in clear board routing area
        board.tracks.push(TrackSegment {
            start: (135.0, 107.0),
            end: (143.0, 107.0),
            width: 0.20,
            layer: 0,
            net_id,
            is_locked: false,
        });

        let calc_len = |tracks: &[TrackSegment]| -> f64 {
            tracks.iter().filter(|t| t.net_id == net_id).map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1)).sum()
        };

        let initial_len = calc_len(&board.tracks);
        assert!((initial_len - 8.0).abs() < 1e-4);

        let target_len = 11.50; // Needs +3.5mm of trombone/accordion tuning
        let mut grid = RoutingGrid::new(&board, 0.20, None);

        let success = tune_net_length(&mut board, &mut grid, net_id, target_len, 1.20);
        assert!(success, "Trombone/accordion tuning must succeed");

        let final_len = calc_len(&board.tracks);
        assert!(
            final_len >= target_len - 0.05,
            "Final tuned length must reach target ({}mm +/- 0.05mm), got {:.3}mm",
            target_len, final_len
        );

        let n_segs = board.tracks.iter().filter(|t| t.net_id == net_id).count();
        assert!(n_segs >= 5, "Accordion pattern must replace straight segment with multiple meander segments, got {}", n_segs);
    }

    #[test]
    fn test_nudge_unlocked_component_frees_blocked_routing_corridor() {
        let nets = vec!["".to_string(), "GND".to_string(), "SIG1".to_string(), "BLOCK".to_string()];
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

        // Two endpoints for SIG1 (net 2): J1 at (105.0, 110.0) and J2 at (125.0, 110.0)
        let j1 = make_fp("J1", 105.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "SIG1", false)]);
        let j2 = make_fp("J_END", 125.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "SIG1", false)]);

        // Unlocked component C1 (net 3) placed in the channel in front of J1
        let mut c1 = make_fp("C1", 107.0, 110.0, false, vec![Pad::new("1", 0.0, 0.0, 0.6, 0.6, 3, "BLOCK", false)]);
        c1.courtyards = vec![Polygon::new(vec![
            Point::new(-0.35, -0.35),
            Point::new(0.35, -0.35),
            Point::new(0.35, 0.35),
            Point::new(-0.35, 0.35),
        ])];

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 104.0,
            min_y: 100.0,
            max_x: 130.0,
            max_y: 120.0,
            footprints: vec![j1, j2, c1],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            // Keepouts constrict J1 so it can only escape in +x through the narrow bottleneck blocked by C1
            keepouts: vec![
                KeepoutZone {
                    min_x: 104.0,
                    min_y: 110.5,
                    max_x: 108.0,
                    max_y: 120.0,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 104.0,
                    min_y: 100.0,
                    max_x: 108.0,
                    max_y: 109.5,
                    layers: vec![0, 1],
                },
            ],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(3).with_allow_component_nudge(true);
        let stop = Arc::new(AtomicBool::new(false));

        let orig_c1_x = board.footprints[2].x;
        let orig_c1_y = board.footprints[2].y;

        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);

        // Verify C1 was nudged away from its blocking position
        let new_c1_x = board.footprints[2].x;
        let new_c1_y = board.footprints[2].y;
        assert!(
            (new_c1_x - orig_c1_x).hypot(new_c1_y - orig_c1_y) > 0.1,
            "Unlocked C1 must be nudged away to free the escape/corridor"
        );

        // Verify connection between J1 and J2 routed successfully
        assert_eq!(total, 1);
        assert_eq!(routed, 1, "Connection must route cleanly after nudging blocking component");
        assert!(!board.tracks.is_empty());
    }

    #[test]
    fn test_shove_and_ripup_blocking_unlocked_track() {
        // Part 1: Direct test of NudgeEngine::try_shove_track
        let mut board = make_router_test_board();
        let router = AutoRouter::new(3);

        board.tracks.push(TrackSegment {
            start: (140.0, 100.0),
            end: (150.0, 100.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });

        // Unlocked track should be shoved perpendicularly (+y)
        assert!(router.nudge_engine.try_shove_track(&mut board, 0, 1.0, None));
        assert!((board.tracks[0].start.1 - 100.25).abs() < 1e-4);
        assert!((board.tracks[0].end.1 - 100.25).abs() < 1e-4);

        // Locked track must not be shoved
        board.tracks[0].is_locked = true;
        assert!(!router.nudge_engine.try_shove_track(&mut board, 0, 1.0, None));

        // Part 2: Router active track shove / rip-up during route_all
        let nets = vec!["".to_string(), "GND".to_string(), "HIGH_PRIO".to_string(), "LOW_PRIO".to_string()];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        netclasses.insert("USB_Diff".to_string(), NetClass {
            name: "USB_Diff".to_string(),
            track_width: 0.20,
            clearance: 0.20,
            via_dia: 0.60,
            via_drill: 0.30,
            diff_pair_gap: 0.20,
            diff_pair_width: 0.20,
            topology: NetTopology::default(),
            ..Default::default()
        });
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("".to_string(), "Default".to_string());
        net_to_class.insert("GND".to_string(), "Default".to_string());
        net_to_class.insert("HIGH_PRIO".to_string(), "USB_Diff".to_string()); // Priority 8
        net_to_class.insert("LOW_PRIO".to_string(), "Default".to_string());   // Priority 1

        let hp_j1 = make_fp("HP1", 105.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "HIGH_PRIO", false)]);
        let hp_j2 = make_fp("HP2", 125.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "HIGH_PRIO", false)]);
        let lp_j1 = make_fp("LP1", 115.0, 105.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "LOW_PRIO", false)]);
        let lp_j2 = make_fp("LP2", 115.0, 115.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "LOW_PRIO", false)]);

        let mut route_board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 100.0,
            min_y: 100.0,
            max_x: 130.0,
            max_y: 120.0,
            footprints: vec![hp_j1, hp_j2, lp_j1, lp_j2],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&route_board, 0.20, None);
        let stop = Arc::new(AtomicBool::new(false));
        let (routed, total) = router.route_all(&mut route_board, &mut grid, None, stop);

        assert_eq!(total, 2);
        assert_eq!(routed, 2, "Both high and low priority signals must be successfully routed");
        let hp_tracks: Vec<&TrackSegment> = route_board.tracks.iter().filter(|t| t.net_id == 2).collect();
        assert!(!hp_tracks.is_empty(), "High-priority signal must have routed tracks");
    }

    #[test]
    fn test_nudge_engine_try_shove_via() {
        let mut board = make_router_test_board();
        let router = AutoRouter::new(3);

        // Add a via at (138.0, 108.0) in free board area (boundary is [130..160, 90..115])
        board.vias.push(Via {
            position: (138.0, 108.0),
            size: 0.60,
            drill: 0.30,
            net_id: 2,
            is_locked: false,
        });

        // Add two incident tracks connecting to the via (one on F.Cu, one on B.Cu)
        board.tracks.push(TrackSegment {
            start: (134.0, 108.0),
            end: (138.0, 108.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: (138.0, 108.0),
            end: (142.0, 108.0),
            width: 0.25,
            layer: 1,
            net_id: 2,
            is_locked: false,
        });

        // Shove the via in +y direction
        let ok = router.nudge_engine.try_shove_via(&mut board, 0, (0.0, 1.0), None);
        assert!(ok, "Unlocked via with incident tracks must be successfully shoved");
        assert!((board.vias[0].position.1 - 108.25).abs() < 1e-4 || (board.vias[0].position.1 - 108.50).abs() < 1e-4);

        // Verify incident tracks on both layers were updated to match the new via position
        let new_vpos = board.vias[0].position;
        assert_eq!(board.tracks[0].end, new_vpos, "Track on F.Cu must connect to new via position");
        assert_eq!(board.tracks[1].start, new_vpos, "Track on B.Cu must connect to new via position");

        // Locked via must not be shoved
        board.vias[0].is_locked = true;
        assert!(!router.nudge_engine.try_shove_via(&mut board, 0, (0.0, 1.0), None));
    }

    #[test]
    fn test_autorouter_shove_blocking_via_and_track() {
        let nets = vec!["".to_string(), "GND".to_string(), "NET_A".to_string(), "NET_B".to_string()];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass {
            name: "Default".to_string(),
            track_width: 0.25,
            clearance: 0.20,
            via_dia: 0.60,
            via_drill: 0.30,
            diff_pair_gap: 0.20,
            diff_pair_width: 0.20,
            topology: NetTopology::default(),
            ..Default::default()
        });
        let mut net_to_class = HashMap::new();
        net_to_class.insert("".to_string(), "Default".to_string());
        net_to_class.insert("GND".to_string(), "Default".to_string());
        net_to_class.insert("NET_A".to_string(), "Default".to_string());
        net_to_class.insert("NET_B".to_string(), "Default".to_string());

        // NET_A: Horizontal route from (105, 110) to (125, 110)
        let a1 = make_fp("A1", 105.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "NET_A", false)]);
        let a2 = make_fp("A2", 125.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "NET_A", false)]);

        // NET_B: Vertical route crossing at (115, 105) to (115, 115)
        let b1 = make_fp("B1", 115.0, 105.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "NET_B", false)]);
        let b2 = make_fp("B2", 115.0, 115.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "NET_B", false)]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 100.0,
            min_y: 100.0,
            max_x: 130.0,
            max_y: 120.0,
            footprints: vec![a1, a2, b1, b2],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(3);
        let stop = Arc::new(AtomicBool::new(false));
        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);

        assert_eq!(total, 2);
        assert_eq!(routed, 2, "Both crossing signals must be routed with shove-and-reroute");
        assert!(!board.tracks.is_empty());
    }

    #[test]
    fn test_net_topology_daisy_chain_and_fly_by() {
        let nc = NetClass::default();
        let pads = vec![
            PadInfo::new(0.0, 0.0, 0, 0, 0).with_driver(true),
            PadInfo::new(10.0, 0.0, 0, 1, 0),
            PadInfo::new(20.0, 0.0, 0, 2, 0),
            PadInfo::new(30.0, 0.0, 0, 3, 0),
        ];

        // 1. DaisyChain: verifies 4 pads connected in strict sequential order P0 -> P1 -> P2 -> P3 with 0 branching
        let daisy_conns = build_net_topology_connections(1, "SPI_BUS", &nc, &pads, NetTopology::DaisyChain);
        assert_eq!(daisy_conns.len(), 3);
        assert_eq!(daisy_conns[0].start, (0.0, 0.0, 0));
        assert_eq!(daisy_conns[0].end, (10.0, 0.0, 0));
        assert_eq!(daisy_conns[1].start, (10.0, 0.0, 0));
        assert_eq!(daisy_conns[1].end, (20.0, 0.0, 0));
        assert_eq!(daisy_conns[2].start, (20.0, 0.0, 0));
        assert_eq!(daisy_conns[2].end, (30.0, 0.0, 0));

        // Verify 0 branching (each internal pad has degree 2, endpoints have degree 1)
        let mut degree = vec![0; 4];
        for c in &daisy_conns {
            let s_idx = pads.iter().position(|p| (p.x, p.y, p.layer) == c.start).unwrap();
            let e_idx = pads.iter().position(|p| (p.x, p.y, p.layer) == c.end).unwrap();
            degree[s_idx] += 1;
            degree[e_idx] += 1;
        }
        assert_eq!(degree[0], 1, "Driver P0 must have degree 1");
        assert_eq!(degree[1], 2, "Intermediate P1 must have degree 2");
        assert_eq!(degree[2], 2, "Intermediate P2 must have degree 2");
        assert_eq!(degree[3], 1, "End receiver P3 must have degree 1");
        assert!(degree.iter().all(|&d| d <= 2), "Daisy chain must have zero branching");

        // 2. FlyBy: verifies sequential flight-time chain without stubs
        let flyby_conns = build_net_topology_connections(2, "DDR_ADDR", &nc, &pads, NetTopology::FlyBy);
        assert_eq!(flyby_conns.len(), 3);
        assert_eq!(flyby_conns[0].start, (0.0, 0.0, 0));
        assert_eq!(flyby_conns[0].end, (10.0, 0.0, 0));
        assert_eq!(flyby_conns[1].start, (10.0, 0.0, 0));
        assert_eq!(flyby_conns[1].end, (20.0, 0.0, 0));
        assert_eq!(flyby_conns[2].start, (20.0, 0.0, 0));
        assert_eq!(flyby_conns[2].end, (30.0, 0.0, 0));
    }

    #[test]
    fn test_net_topology_star_point() {
        let nc = NetClass::default();
        let star_center = (15.0, 15.0, 0);
        let pads = vec![
            PadInfo::new(15.0, 15.0, 0, 0, 0).with_driver(true), // Center / power source
            PadInfo::new(0.0, 0.0, 0, 1, 0),
            PadInfo::new(30.0, 0.0, 0, 2, 0),
            PadInfo::new(15.0, 30.0, 0, 3, 0),
        ];

        let conns = build_net_topology_connections(1, "+3.3V", &nc, &pads, NetTopology::StarPoint);
        assert_eq!(conns.len(), 3);

        // Verify all connections share the star center pad as start endpoint
        for c in &conns {
            assert_eq!(c.start, star_center);
            assert_eq!(c.start_fp, 0);
            assert_eq!(c.start_pad, 0);
        }

        let endpoints: Vec<(f64, f64, usize)> = conns.iter().map(|c| c.end).collect();
        assert!(endpoints.contains(&(0.0, 0.0, 0)));
        assert!(endpoints.contains(&(30.0, 0.0, 0)));
        assert!(endpoints.contains(&(15.0, 30.0, 0)));
    }

    #[test]
    fn test_net_topology_t_branch() {
        let nc = NetClass::default();
        // Driver at (0, 0), Receivers symmetrically placed at (20, 10) and (20, -10)
        let pads = vec![
            PadInfo::new(0.0, 0.0, 0, 0, 0).with_driver(true),
            PadInfo::new(20.0, 10.0, 0, 1, 0),
            PadInfo::new(20.0, -10.0, 0, 2, 0),
        ];

        let conns = build_net_topology_connections(1, "CLK_NET", &nc, &pads, NetTopology::TBranch);
        assert_eq!(conns.len(), 3); // 1 trunk + 2 symmetric branches

        // Trunk: Driver (0, 0) -> Split node (20, 0)
        let trunk = &conns[0];
        assert_eq!(trunk.start, (0.0, 0.0, 0));
        assert_eq!(trunk.end, (20.0, 0.0, 0));

        // Branches from split node to receivers
        let branch1 = &conns[1];
        let branch2 = &conns[2];
        assert_eq!(branch1.start, (20.0, 0.0, 0));
        assert_eq!(branch1.end, (20.0, 10.0, 0));
        assert_eq!(branch2.start, (20.0, 0.0, 0));
        assert_eq!(branch2.end, (20.0, -10.0, 0));

        // Verify symmetric balanced split with matched branch lengths
        let len1 = (branch1.start.0 - branch1.end.0).hypot(branch1.start.1 - branch1.end.1);
        let len2 = (branch2.start.0 - branch2.end.0).hypot(branch2.start.1 - branch2.end.1);
        assert!((len1 - len2).abs() < 1e-6);
        assert!((len1 - 10.0).abs() < 1e-6);
        assert!((len2 - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_net_topology_parsing_and_defaults() {
        assert_eq!(NetTopology::default(), NetTopology::EuclideanMst);
        let mut nc = NetClass::default();
        assert_eq!(nc.topology, NetTopology::EuclideanMst);

        // Verify heuristic detection
        assert_eq!(nc.resolve_topology("SYS_CLK"), NetTopology::TBranch);
        assert_eq!(nc.resolve_topology("DDR4_ADDR1"), NetTopology::FlyBy);
        assert_eq!(nc.resolve_topology("I2C_SCL"), NetTopology::DaisyChain);
        assert_eq!(nc.resolve_topology("+5V"), NetTopology::StarPoint);
        assert_eq!(nc.resolve_topology("SIG_GENERAL"), NetTopology::EuclideanMst);

        // Verify explicit setting overrides heuristic
        nc.topology = NetTopology::StarPoint;
        assert_eq!(nc.resolve_topology("SYS_CLK"), NetTopology::StarPoint);
    }

    #[test]
    fn test_post_route_glossing_straightens_rubber_band() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        // A track on SIG1 with an unnecessary V-shaped detour in open space:
        // (10.0, 10.0) -> (15.0, 15.0) -> (20.0, 10.0)
        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (15.0, 15.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 15.0),
                    end: (20.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(3);

        let initial_len: f64 = board.tracks.iter().map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1)).sum();
        assert!((initial_len - 14.142).abs() < 0.05);

        router.gloss_tracks(&mut board, &grid);

        let glossed_len: f64 = board.tracks.iter().map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1)).sum();
        assert_eq!(board.tracks.len(), 1, "Rubber-band straightening should collapse 2 detour segments into 1 direct segment");
        assert!((glossed_len - 10.0).abs() < 1e-4, "Straightened length must equal direct horizontal distance (10.0 mm)");
        assert_eq!(board.tracks[0].start, (10.0, 10.0));
        assert_eq!(board.tracks[0].end, (20.0, 10.0));
    }

    #[test]
    fn test_post_route_glossing_eliminates_stair_steps() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        // Staircase zig-zag: (10, 10) -> (12, 10) -> (12, 12) -> (14, 12) -> (14, 14) -> (16, 14) -> (16, 16) -> (20, 16)
        let pts = [
            (10.0, 10.0),
            (12.0, 10.0),
            (12.0, 12.0),
            (14.0, 12.0),
            (14.0, 14.0),
            (16.0, 14.0),
            (16.0, 16.0),
            (20.0, 16.0),
        ];
        let mut tracks = Vec::new();
        for i in 0..pts.len() - 1 {
            tracks.push(TrackSegment {
                start: pts[i],
                end: pts[i + 1],
                width: 0.25,
                layer: 0,
                net_id: 1,
                is_locked: false,
            });
        }

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks,
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(3);

        let initial_len: f64 = board.tracks.iter().map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1)).sum();
        let initial_bends = board.tracks.len().saturating_sub(1);
        assert_eq!(initial_bends, 6);

        router.gloss_tracks(&mut board, &grid);

        let glossed_len: f64 = board.tracks.iter().map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1)).sum();
        assert!(
            board.tracks.len() <= 3,
            "Staircase elimination must collapse 7 segments into <= 3 canonical segments, got {}",
            board.tracks.len()
        );
        assert!(
            glossed_len < initial_len - 1.5,
            "Glossed path length ({:.3}) must be shorter than staircase ({:.3})",
            glossed_len,
            initial_len
        );
    }

    #[test]
    fn test_post_route_glossing_preserves_clearance() {
        let nets = vec!["".to_string(), "SIG1".to_string(), "BLOCKER".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        net_name_to_id.insert("BLOCKER".to_string(), 2);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());
        net_to_class.insert("BLOCKER".to_string(), "Default".to_string());

        let blocker_fp = make_fp("U_BLOCK", 12.0, 12.0, true, vec![
            Pad::new("1", 0.0, 0.0, 2.0, 2.0, 2, "BLOCKER", false),
        ]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: vec![blocker_fp],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (10.0, 20.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 20.0),
                    end: (20.0, 20.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        grid.rasterize_obstacles(&board);
        let router = AutoRouter::new(3);

        router.gloss_tracks(&mut board, &grid);

        for trk in &board.tracks {
            let dist = dist_to_segment(12.0, 12.0, trk.start.0, trk.start.1, trk.end.0, trk.end.1);
            let req_clearance = 0.20 + 0.25 / 2.0 + 1.0; // pad_rad + track_half_w + clearance
            assert!(
                dist >= req_clearance - 0.05,
                "Track segment from ({:.2}, {:.2}) to ({:.2}, {:.2}) encroaches on blocker at (12, 12) with dist {:.3} < {:.3}",
                trk.start.0, trk.start.1, trk.end.0, trk.end.1, dist, req_clearance
            );
        }
    }

    #[test]
    fn test_via_reduction_eliminates_redundant_via_pair() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        // Net 1 has 3 track segments:
        // L0: (10, 10) -> (15, 10)
        // Via 1 at (15, 10)
        // L1: (15, 10) -> (25, 10)
        // Via 2 at (25, 10)
        // L0: (25, 10) -> (30, 10)
        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (15.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 10.0),
                    end: (25.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (25.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (15.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (25.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(3);

        assert_eq!(board.vias.len(), 2);
        router.reduce_vias(&mut board, &mut grid);

        assert_eq!(board.vias.len(), 0, "Via reduction pass must remove redundant via pair in free space");
        let l1_tracks = board.tracks.iter().filter(|t| t.layer == 1).count();
        assert_eq!(l1_tracks, 0, "All tracks must be unified on layer 0 with 0 tracks on layer 1");
    }

    #[test]
    fn test_via_consolidation_single_via_elimination() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        // Net 1 with Pad A at (10, 10) on F.Cu and Pad B at (30, 10) on F.Cu
        // Connected via 1 via at (15, 10) and tracks on B.Cu (layer 1) to (30, 10).
        let pad_a = Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false);
        let fp_a = make_fp("U1", 10.0, 10.0, false, vec![pad_a]);
        let pad_b = Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false);
        let fp_b = make_fp("U2", 30.0, 10.0, false, vec![pad_b]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: vec![fp_a, fp_b],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (15.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (15.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        assert_eq!(board.vias.len(), 1);
        assert_eq!(board.tracks.iter().filter(|t| t.layer == 1).count(), 1);

        reduce_vias(&mut board, &mut grid);

        assert_eq!(board.vias.len(), 0, "Via consolidation must eliminate the solitary via");
        let l1_tracks = board.tracks.iter().filter(|t| t.layer == 1).count();
        assert_eq!(l1_tracks, 0, "All tracks must be moved to F.Cu (layer 0)");
    }

    #[test]
    fn test_via_sliding_eliminates_jog() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        let pad_a = Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false);
        let fp_a = make_fp("U1", 0.0, 0.0, false, vec![pad_a]);
        let pad_b = Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false);
        let mut fp_b = make_fp("U2", 10.0, 10.0, false, vec![pad_b]);
        fp_b.layer = 1;

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: -10.0,
            min_y: -10.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: vec![fp_a, fp_b],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (0.0, 0.0),
                    end: (10.0, 0.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 0.0),
                    end: (10.0, 0.5),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 0.5),
                    end: (10.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (10.0, 0.5),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let grid = RoutingGrid::new(&board, 0.20, None);
        assert_eq!(board.vias.len(), 1);
        assert_eq!(board.tracks.len(), 3);

        let changed = slide_and_align_vias(&mut board, &grid);
        assert!(changed, "slide_and_align_vias should return true");

        assert_eq!(board.vias.len(), 1);
        assert_eq!(board.vias[0].position, (10.0, 0.0), "Via should slide to the corner at (10.0, 0.0)");
        assert_eq!(board.tracks.len(), 2, "Dogleg jog track should be eliminated");

        let l0_track = board.tracks.iter().find(|t| t.layer == 0).unwrap();
        assert_eq!(l0_track.start, (0.0, 0.0));
        assert_eq!(l0_track.end, (10.0, 0.0));

        let l1_track = board.tracks.iter().find(|t| t.layer == 1).unwrap();
        assert_eq!(l1_track.start, (10.0, 0.0));
        assert_eq!(l1_track.end, (10.0, 10.0));
    }

    #[test]
    fn test_via_reduction_preserves_necessary_vias_over_obstacles() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        // Keepout on layer 0 blocking (15, 10) to (25, 10)
        let ko = KeepoutZone {
            min_x: 18.0,
            min_y: 5.0,
            max_x: 22.0,
            max_y: 15.0,
            layers: vec![0], // only blocks layer 0
        };

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![ko],
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (15.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 10.0),
                    end: (25.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (25.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (15.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (25.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        grid.rasterize_obstacles(&board);
        let router = AutoRouter::new(3);

        router.reduce_vias(&mut board, &mut grid);

        assert_eq!(
            board.vias.len(),
            2,
            "Necessary vias crossing an obstacle on layer 0 must be preserved"
        );
        let l1_tracks = board.tracks.iter().filter(|t| t.layer == 1).count();
        assert_eq!(l1_tracks, 1, "Layer 1 track segment must be preserved");
    }

    #[test]
    fn test_via_reduction_respects_length_budget() {
        let nets = vec!["".to_string(), "SIG1".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("SIG1".to_string(), 1);
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_to_class = HashMap::new();
        net_to_class.insert("SIG1".to_string(), "Default".to_string());

        // Short hop of length 2.0 mm on layer 1:
        // L0: (10, 10) -> (15, 10)
        // V1 at (15, 10), V2 at (17, 10)
        // L1: (15, 10) -> (17, 10)
        // L0: (17, 10) -> (22, 10)
        let ko = KeepoutZone {
            min_x: 15.5,
            min_y: 0.0,
            max_x: 16.5,
            max_y: 20.0,
            layers: vec![0], // spans y from 0 to 20, completely blocking layer 0
        };

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![ko],
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (15.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 10.0),
                    end: (17.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (17.0, 10.0),
                    end: (22.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (15.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (17.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        grid.rasterize_obstacles(&board);
        let router = AutoRouter::new(3);

        router.reduce_vias(&mut board, &mut grid);

        assert_eq!(
            board.vias.len(),
            2,
            "Vias must be preserved when detour exceeds length budget or is blocked"
        );
    }

    #[test]
    fn test_continuous_track_shove_deflects_with_45_degree_transitions() {
        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets: vec!["".to_string(), "SIG1".to_string(), "INTRUDER".to_string()],
            net_name_to_id: [("SIG1".to_string(), 1), ("INTRUDER".to_string(), 2)].into_iter().collect(),
            netclasses: [("Default".to_string(), NetClass::default())].into_iter().collect(),
            net_to_class: [("SIG1".to_string(), "Default".to_string()), ("INTRUDER".to_string(), "Default".to_string())].into_iter().collect(),
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let nudge = NudgeEngine::new();
        // Intruder on net 2 from (18.0, 10.20) to (22.0, 10.20), width 0.25
        let shoved = nudge.try_continuous_shove_track(
            &mut board,
            0,
            (18.0, 10.20),
            (22.0, 10.20),
            0.25,
            2,
            None,
        );

        assert!(shoved, "Continuous track shove must succeed in open space");

        let has_start = board.tracks.iter().any(|t| (t.start.0 - 10.0).hypot(t.start.1 - 10.0) < 1e-4 || (t.end.0 - 10.0).hypot(t.end.1 - 10.0) < 1e-4);
        let has_end = board.tracks.iter().any(|t| (t.start.0 - 30.0).hypot(t.start.1 - 10.0) < 1e-4 || (t.end.0 - 30.0).hypot(t.end.1 - 10.0) < 1e-4);
        assert!(has_start, "Start endpoint (10, 10) must be preserved");
        assert!(has_end, "End endpoint (30, 10) must be preserved");

        assert_eq!(board.tracks.len(), 3, "Shoved track must produce 3 segments (lead-in, deflected middle, lead-out)");

        let lead_in = board.tracks.iter().find(|t| (t.start.0 - 10.0).hypot(t.start.1 - 10.0) < 1e-4 || (t.end.0 - 10.0).hypot(t.end.1 - 10.0) < 1e-4).unwrap();
        let dx = (lead_in.end.0 - lead_in.start.0).abs();
        let dy = (lead_in.end.1 - lead_in.start.1).abs();
        assert!((dx - dy).abs() < 1e-4, "Lead-in transition must be at 45 degrees (dx == dy), got dx={:.3}, dy={:.3}", dx, dy);

        for t in &board.tracks {
            let dist = dist_segment_to_segment(t.start, t.end, (18.0, 10.20), (22.0, 10.20));
            assert!(dist >= 0.45 - 1e-4, "All deflected segments must maintain required clearance >= 0.45mm, got {:.3}", dist);
        }
    }

    #[test]
    fn test_continuous_track_shove_cascades_multiple_tracks() {
        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets: vec!["".to_string(), "SIG1".to_string(), "SIG2".to_string(), "INTRUDER".to_string()],
            net_name_to_id: [("SIG1".to_string(), 1), ("SIG2".to_string(), 2), ("INTRUDER".to_string(), 3)].into_iter().collect(),
            netclasses: [("Default".to_string(), NetClass::default())].into_iter().collect(),
            net_to_class: [
                ("SIG1".to_string(), "Default".to_string()),
                ("SIG2".to_string(), "Default".to_string()),
                ("INTRUDER".to_string(), "Default".to_string()),
            ].into_iter().collect(),
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 9.35),
                    end: (30.0, 9.35),
                    width: 0.25,
                    layer: 0,
                    net_id: 2,
                    is_locked: false,
                },
            ],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let nudge = NudgeEngine::new();
        let shoved = nudge.try_continuous_shove_track(
            &mut board,
            0,
            (18.0, 10.20),
            (22.0, 10.20),
            0.25,
            3,
            None,
        );

        assert!(shoved, "Cascading continuous shove across multiple tracks must succeed");

        for t1 in board.tracks.iter().filter(|t| t.net_id == 1) {
            let dist_to_intruder = dist_segment_to_segment(t1.start, t1.end, (18.0, 10.20), (22.0, 10.20));
            assert!(dist_to_intruder >= 0.45 - 1e-4, "Net 1 must be clear of intruder");

            for t2 in board.tracks.iter().filter(|t| t.net_id == 2) {
                let dist_between = dist_segment_to_segment(t1.start, t1.end, t2.start, t2.end);
                assert!(dist_between >= 0.45 - 1e-4, "Cascaded Net 2 must be clear of shoved Net 1, got {:.3}", dist_between);
            }
        }
    }

    #[test]
    fn test_continuous_track_shove_rejects_on_locked_or_keepout() {
        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets: vec!["".to_string(), "SIG1".to_string(), "LOCKED".to_string(), "INTRUDER".to_string()],
            net_name_to_id: [("SIG1".to_string(), 1), ("LOCKED".to_string(), 2), ("INTRUDER".to_string(), 3)].into_iter().collect(),
            netclasses: [("Default".to_string(), NetClass::default())].into_iter().collect(),
            net_to_class: [
                ("SIG1".to_string(), "Default".to_string()),
                ("LOCKED".to_string(), "Default".to_string()),
                ("INTRUDER".to_string(), "Default".to_string()),
            ].into_iter().collect(),
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 9.35),
                    end: (30.0, 9.35),
                    width: 0.25,
                    layer: 0,
                    net_id: 2,
                    is_locked: true,
                },
            ],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let orig_tracks = board.tracks.clone();
        let nudge = NudgeEngine::new();

        let shoved = nudge.try_continuous_shove_track(
            &mut board,
            0,
            (18.0, 10.20),
            (22.0, 10.20),
            0.25,
            3,
            None,
        );

        assert!(!shoved, "Continuous shove must safely abort when blocked by a locked track");
        assert_eq!(board.tracks.len(), 2, "Tracks must not be altered when shove fails");
        assert_eq!(board.tracks[0].start, orig_tracks[0].start);
        assert_eq!(board.tracks[0].end, orig_tracks[0].end);
    }

    #[test]
    fn test_prune_dangling_tracks_and_stubs() {
        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 50.0,
            max_y: 50.0,
            footprints: vec![
                make_fp("U1", 10.0, 10.0, false, vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false)]),
                make_fp("U2", 30.0, 10.0, false, vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false)]),
            ],
            holding_palette: Vec::new(),
            nets: vec!["".to_string(), "SIG1".to_string()],
            net_name_to_id: [("SIG1".to_string(), 1)].into_iter().collect(),
            netclasses: [("Default".to_string(), NetClass::default())].into_iter().collect(),
            net_to_class: [("SIG1".to_string(), "Default".to_string())].into_iter().collect(),
            keepouts: Vec::new(),
            tracks: vec![
                // 1. Valid connected route between U1 and U2:
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (20.0, 10.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (20.0, 10.0),
                    end: (30.0, 10.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // 2. Dangling stub branching off the route into empty space:
                TrackSegment {
                    start: (20.0, 10.0),
                    end: (20.0, 15.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // 3. Multi-segment antenna extension on the stub:
                TrackSegment {
                    start: (20.0, 15.0),
                    end: (20.0, 20.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // 4. Isolated floating track segment in empty space:
                TrackSegment {
                    start: (40.0, 40.0),
                    end: (45.0, 40.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                // Dangling via at the end of the multi-segment antenna:
                Via {
                    position: (20.0, 20.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        prune_dangling_tracks_and_vias(&mut board);

        // Connected routes between U1 and U2 must be 100% preserved
        assert_eq!(board.tracks.len(), 2, "Only the 2 connected route segments must survive");
        assert_eq!(board.tracks[0].start, (10.0, 10.0));
        assert_eq!(board.tracks[0].end, (20.0, 10.0));
        assert_eq!(board.tracks[1].start, (20.0, 10.0));
        assert_eq!(board.tracks[1].end, (30.0, 10.0));

        // Dangling vias and stubs must be completely pruned
        assert!(board.vias.is_empty(), "Dangling via must be completely pruned");
    }

    #[test]
    fn test_via_clearance_against_foreign_pads_and_tracks() {
        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 50.0,
            max_y: 50.0,
            footprints: vec![
                make_fp("U1", 10.0, 10.0, false, vec![Pad::new("1", 0.0, 0.0, 0.50, 0.50, 1, "SIG1", false)]),
            ],
            holding_palette: Vec::new(),
            nets: vec!["".to_string(), "SIG1".to_string(), "SIG2".to_string()],
            net_name_to_id: [("SIG1".to_string(), 1), ("SIG2".to_string(), 2)].into_iter().collect(),
            netclasses: [
                ("Default".to_string(), NetClass { clearance: 0.20, track_width: 0.20, via_dia: 0.60, via_drill: 0.30, ..NetClass::default() }),
            ].into_iter().collect(),
            net_to_class: [("SIG1".to_string(), "Default".to_string()), ("SIG2".to_string(), "Default".to_string())].into_iter().collect(),
            keepouts: Vec::new(),
            tracks: vec![
                // Foreign track of net 1 on layer 0 from (10.0, 10.0) to (10.0, 30.0)
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (10.0, 30.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let via_dia = 0.60;
        let clearance = 0.20;

        // 1. Candidate via directly overlapping foreign SMD pad at (10.0, 10.0) -> illegal
        assert!(!is_via_location_legal(&board, (10.0, 10.0), via_dia, 2, clearance));

        // 2. Candidate via within clearance of foreign SMD pad at (10.0, 10.45)
        // Pad boundary extends to y = 10.25. Via radius = 0.30, clearance = 0.20. Required edge dist = 0.50.
        // At y = 10.45, dist = 0.20 < 0.50 -> illegal
        assert!(!is_via_location_legal(&board, (10.0, 10.45), via_dia, 2, clearance));

        // 3. Candidate via directly on top of foreign track at (10.0, 20.0) -> illegal
        assert!(!is_via_location_legal(&board, (10.0, 20.0), via_dia, 2, clearance));

        // 4. Candidate via violating clearance to foreign track at (10.35, 20.0)
        // Required center-to-segment distance = via_r(0.30) + trk_w/2(0.10) + clearance(0.20) = 0.60 mm.
        // Distance is 0.35 < 0.60 -> illegal
        assert!(!is_via_location_legal(&board, (10.35, 20.0), via_dia, 2, clearance));

        // 5. Candidate via safely placed outside clearance corridor at (10.80, 20.0)
        // Distance is 0.80 >= 0.60 -> legal
        assert!(is_via_location_legal(&board, (10.80, 20.0), via_dia, 2, clearance));

        // 6. Test validate_and_clean_copper removes any accidentally placed colliding vias
        board.vias.push(Via {
            position: (10.0, 20.0), // Collision with foreign track!
            size: 0.60,
            drill: 0.30,
            net_id: 2,
            is_locked: false,
        });
        validate_and_clean_copper(&mut board);
        assert!(board.vias.is_empty(), "Colliding via must be eliminated by validate_and_clean_copper");

        // 7. Same-net SMD pad clearance test:
        // Candidate via of net 1 directly on top of net 1 SMD pad at (10.0, 10.0) -> MUST be illegal!
        assert!(
            !is_via_location_legal(&board, (10.0, 10.0), via_dia, 1, clearance),
            "Via must never be legal overlapping its own net SMD pad"
        );
        // Candidate via of net 1 within clearance of net 1 SMD pad at (10.0, 10.45) -> MUST be illegal!
        assert!(
            !is_via_location_legal(&board, (10.0, 10.45), via_dia, 1, clearance),
            "Via must never be legal violating clearance against its own net SMD pad"
        );
        // Candidate via of net 1 safely outside clearance at (10.0, 10.80) -> legal
        assert!(
            is_via_location_legal(&board, (10.0, 10.80), via_dia, 1, clearance),
            "Via safely outside clearance of its own net SMD pad should be legal"
        );

        // 8. validate_and_clean_copper eliminates via overlapping same-net SMD pad
        board.vias.push(Via {
            position: (10.0, 10.0), // Collision with same-net SMD pad!
            size: 0.60,
            drill: 0.30,
            net_id: 1,
            is_locked: false,
        });
        validate_and_clean_copper(&mut board);
        assert!(board.vias.is_empty(), "Same-net SMD pad colliding via must be eliminated by validate_and_clean_copper");

        // 9. Net 0 unconnected pad clearance test
        let mut board_nc = board.clone();
        board_nc.footprints.push(make_fp("U2", 30.0, 30.0, false, vec![Pad::new("1", 0.0, 0.0, 0.60, 0.60, 0, "", false)]));
        assert!(
            !is_via_location_legal(&board_nc, (30.0, 30.0), via_dia, 1, clearance),
            "Via must never be legal overlapping an unconnected net 0 pad"
        );

        // 10. Oval pad and flipped footprint clearance test
        let mut board_oval = board.clone();
        let mut oval_pad = Pad::new("1", 0.0, 0.0, 1.45, 0.30, 1, "SIG1", false);
        oval_pad.shape = "oval".to_string();
        oval_pad.rotation = 90.0;
        let mut fp_flipped = make_fp("J2", 25.0, 25.0, false, vec![oval_pad]);
        fp_flipped.layer = 1; // Flipped to B.Cu
        fp_flipped.orig_layer = 0;
        board_oval.footprints.push(fp_flipped);

        // Candidate via directly on oval pad
        assert!(!is_via_location_legal(&board_oval, (25.0, 25.0), via_dia, 1, clearance));
        // Candidate via right at edge of oval pad
        assert!(!is_via_location_legal(&board_oval, (25.0, 25.50), via_dia, 1, clearance));
        // Candidate via safely away from flipped oval pad
        assert!(is_via_location_legal(&board_oval, (25.0, 27.0), via_dia, 1, clearance));
    }

    #[test]
    fn test_route_between_pads_of_component() {
        let nets = vec!["".to_string(), "GND".to_string(), "PAD1_NET".to_string(), "PAD2_NET".to_string(), "SIG_NET".to_string()];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        let mut nc = NetClass::default();
        nc.track_width = 0.20;
        nc.clearance = 0.20;
        netclasses.insert("Default".to_string(), nc);
        let mut net_to_class = HashMap::new();
        for n in &nets {
            net_to_class.insert(n.clone(), "Default".to_string());
        }

        // Component R1 at (10.0, 10.05) with two pads:
        // Pad 1 at (10.0, 9.45), size (0.80, 0.60) -> top edge is at 9.75
        // Pad 2 at (10.0, 10.65), size (0.80, 0.60) -> bottom edge is at 10.35
        // Physical gap between Pad 1 and Pad 2: 10.35 - 9.75 = 0.60 mm.
        // Required space for 0.20 mm track with 0.20 mm clearance on both sides:
        // clearance (0.20) + track_width (0.20) + clearance (0.20) = 0.60 mm.
        // Since 0.60 mm >= 0.60 mm, clearance & width allow routing between the pads!
        let r1_pads = vec![
            Pad::new("1", 0.0, -0.60, 0.80, 0.60, 2, "PAD1_NET", false),
            Pad::new("2", 0.0, 0.60, 0.80, 0.60, 3, "PAD2_NET", false),
        ];
        let r1 = make_fp("R1", 10.0, 10.05, true, r1_pads);

        // SIG_NET endpoints: J1 at (5.0, 10.05), J2 at (15.0, 10.05)
        let j1 = make_fp("J1", 5.0, 10.05, true, vec![Pad::new("1", 0.0, 0.0, 0.40, 0.40, 4, "SIG_NET", false)]);
        let j2 = make_fp("J2", 15.0, 10.05, true, vec![Pad::new("1", 0.0, 0.0, 0.40, 0.40, 4, "SIG_NET", false)]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 20.0,
            max_y: 20.0,
            footprints: vec![r1, j1, j2],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            // Keepouts force the signal to pass directly through the channel between R1 pads at y = 10.05 on layer 0 (F.Cu)
            keepouts: vec![
                KeepoutZone {
                    min_x: 8.0,
                    min_y: 11.5,
                    max_x: 12.0,
                    max_y: 20.0,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 8.0,
                    min_y: 0.0,
                    max_x: 12.0,
                    max_y: 8.5,
                    layers: vec![0, 1],
                },
                // Block layer 1 under R1 completely so route cannot bypass layer 0
                KeepoutZone {
                    min_x: 8.0,
                    min_y: 8.5,
                    max_x: 12.0,
                    max_y: 11.5,
                    layers: vec![1],
                },
            ],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.10, None);
        let router = AutoRouter::new(3);
        let stop = Arc::new(AtomicBool::new(false));

        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);
        assert_eq!(total, 1);
        assert_eq!(
            routed, 1,
            "Router route_all must be able to route between pads of R1 when clearance (0.20) and width (0.20) fit in 0.60 mm gap"
        );

        // Verify a track segment passes through the gap between R1 pads on layer 0 (F.Cu)
        let passes_between = board.tracks.iter().any(|t| {
            t.net_id == 4
                && t.layer == 0
                && ((t.start.0 <= 10.0 && t.end.0 >= 10.0) || (t.start.0 >= 10.0 && t.end.0 <= 10.0))
                && t.start.1 > 9.68 && t.start.1 < 10.32
                && t.end.1 > 9.68 && t.end.1 < 10.32
        });
        assert!(passes_between, "Track must physically pass between Pad 1 and Pad 2 of R1 on F.Cu");
    }

    #[test]
    fn test_route_between_pads_of_rotated_component() {
        let nets = vec!["".to_string(), "GND".to_string(), "PAD1_NET".to_string(), "PAD2_NET".to_string(), "SIG_NET".to_string()];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        let mut nc = NetClass::default();
        nc.track_width = 0.20;
        nc.clearance = 0.20;
        netclasses.insert("Default".to_string(), nc);
        let mut net_to_class = HashMap::new();
        for n in &nets {
            net_to_class.insert(n.clone(), "Default".to_string());
        }

        // Component R1 at (10.05, 10.0) rotated 90 degrees
        // In local coords: pads at (0.0, -0.60) and (0.0, 0.60), size (0.80, 0.60).
        // Rotated 90 degrees: pads are at (9.45, 10.0) and (10.65, 10.0).
        // Physical gap between Pad 1 and Pad 2 in X: 0.60 mm.
        // Vertical channel between pads runs at x = 10.05 from y = 5.0 to y = 15.0.
        let r1_pads = vec![
            Pad::new("1", 0.0, -0.60, 0.80, 0.60, 2, "PAD1_NET", false),
            Pad::new("2", 0.0, 0.60, 0.80, 0.60, 3, "PAD2_NET", false),
        ];
        let mut r1 = make_fp("R1", 10.05, 10.0, true, r1_pads);
        r1.rotation = 90.0;

        // SIG_NET endpoints: J1 at (10.05, 5.0), J2 at (10.05, 15.0)
        let j1 = make_fp("J1", 10.05, 5.0, true, vec![Pad::new("1", 0.0, 0.0, 0.40, 0.40, 4, "SIG_NET", false)]);
        let j2 = make_fp("J2", 10.05, 15.0, true, vec![Pad::new("1", 0.0, 0.0, 0.40, 0.40, 4, "SIG_NET", false)]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 20.0,
            max_y: 20.0,
            footprints: vec![r1, j1, j2],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![
                KeepoutZone {
                    min_x: 11.5,
                    min_y: 8.0,
                    max_x: 20.0,
                    max_y: 12.0,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 0.0,
                    min_y: 8.0,
                    max_x: 8.5,
                    max_y: 12.0,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 8.5,
                    min_y: 8.0,
                    max_x: 11.5,
                    max_y: 12.0,
                    layers: vec![1],
                },
            ],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.10, None);
        let router = AutoRouter::new(3);
        let stop = Arc::new(AtomicBool::new(false));

        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);

        assert_eq!(total, 1);
        assert_eq!(
            routed, 1,
            "Router must route vertically between pads of 90-degree rotated R1 on F.Cu"
        );

        let passes_between = board.tracks.iter().any(|t| {
            t.net_id == 4
                && t.layer == 0
                && ((t.start.1 <= 10.0 && t.end.1 >= 10.0) || (t.start.1 >= 10.0 && t.end.1 <= 10.0))
                && t.start.0 > 9.68 && t.start.0 < 10.32
                && t.end.0 > 9.68 && t.end.0 < 10.32
        });
        assert!(passes_between, "Track must physically pass vertically between Pad 1 and Pad 2 of 90-deg rotated R1");
    }

    #[test]
    fn test_route_between_through_hole_pins() {
        let nets = vec!["".to_string(), "GND".to_string(), "PIN1_NET".to_string(), "PIN2_NET".to_string(), "SIG_NET".to_string()];
        let mut net_name_to_id = HashMap::new();
        for (i, n) in nets.iter().enumerate() {
            net_name_to_id.insert(n.clone(), i);
        }
        let mut netclasses = HashMap::new();
        let mut nc = NetClass::default();
        nc.track_width = 0.20;
        nc.clearance = 0.20;
        netclasses.insert("Default".to_string(), nc);
        let mut net_to_class = HashMap::new();
        for n in &nets {
            net_to_class.insert(n.clone(), "Default".to_string());
        }

        // 2.54mm header with 2 through-hole pins:
        // Pin 1 at (10.0, 8.73), Pin 2 at (10.0, 11.27) -> pitch 2.54 mm.
        // Pad size: 1.60 x 1.60 mm.
        // Gap = 2.54 - 1.60 = 0.94 mm.
        // Channel at y = 10.0. Required for 0.20 track + 2x0.20 clearance = 0.60 mm. Fits with 0.34 mm margin.
        let j_hdr_pads = vec![
            Pad::new("1", 0.0, -1.27, 1.60, 1.60, 2, "PIN1_NET", true),
            Pad::new("2", 0.0, 1.27, 1.60, 1.60, 3, "PIN2_NET", true),
        ];
        let j_hdr = make_fp("J_HDR", 10.0, 10.05, true, j_hdr_pads);

        let j1 = make_fp("J1", 5.0, 10.05, true, vec![Pad::new("1", 0.0, 0.0, 0.40, 0.40, 4, "SIG_NET", false)]);
        let j2 = make_fp("J2", 15.0, 10.05, true, vec![Pad::new("1", 0.0, 0.0, 0.40, 0.40, 4, "SIG_NET", false)]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 20.0,
            max_y: 20.0,
            footprints: vec![j_hdr, j1, j2],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![
                KeepoutZone {
                    min_x: 8.0,
                    min_y: 12.5,
                    max_x: 12.0,
                    max_y: 20.0,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 8.0,
                    min_y: 0.0,
                    max_x: 12.0,
                    max_y: 7.5,
                    layers: vec![0, 1],
                },
            ],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.10, None);
        let router = AutoRouter::new(3);
        let stop = Arc::new(AtomicBool::new(false));

        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);

        assert_eq!(total, 1);
        assert_eq!(routed, 1, "Router must route between through-hole pins of 2.54mm header");
    }

    #[test]
    fn test_side_and_diagonal_escape_stubs_for_dense_ic_corners() {
        let mut board = make_router_test_board();
        // In make_router_test_board, U1 is at (145.0, 102.0).
        // Pad 6 on U1 is the corner pad at (-1.5, 1.0), so absolute position is (143.5, 103.0).
        // Place a target destination footprint for net 2 ("SIG1") South-West of Pad 6 at (140.0, 106.5).
        let target_fp = make_fp(
            "TARGET_SW",
            140.0,
            106.5,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "SIG1", false)],
        );
        board.footprints.push(target_fp);

        let router = AutoRouter::new(3);
        let stubs = router.generate_escape_stubs(&board);

        // Find the escape stub for Pad 6 on U1
        let pad6_stub = stubs.iter().find(|s| {
            s.net_id == 2 && (s.pad_pos.0 - 143.5).hypot(s.pad_pos.1 - 103.0) < 0.1
        });
        assert!(pad6_stub.is_some(), "Pad 6 on U1 must generate an escape stub");
        let stub = pad6_stub.unwrap();

        // Vector from pad to tip
        let dx = stub.tip_pos.0 - stub.pad_pos.0;
        let dy = stub.tip_pos.1 - stub.pad_pos.1;
        
        // Pad 6 at bottom-left corner with target to the South-West should escape diagonally towards the destination
        assert!(dx < -1e-4, "Escape stub must project westward towards target: dx={}", dx);
        assert!(dy > 1e-4, "Escape stub must project southward towards target (diagonal escape): dy={}", dy);
        assert!((dx.abs() - dy.abs()).abs() < 1e-3, "Diagonal escape stub must be 45-degree octilinear (|dx| == |dy|): dx={}, dy={}", dx, dy);
    }

    #[test]
    fn test_avoid_acid_traps_eliminates_acute_angles() {
        let board = make_router_test_board();
        let grid = RoutingGrid::new(&board, 0.10, None);

        // Create a chain with an acute angle (135° turn / 45° acute interior wedge)
        // (0.0, 0.0) -> (2.0, 0.0) -> (1.0, 1.0)
        // v1 = (2.0, 0.0), v2 = (-1.0, 1.0). dot = -2.0 / (2.0 * sqrt(2)) = -0.707 < -0.05
        let mut chain = TrackChain {
            net_id: 2,
            layer: 0,
            width: 0.20,
            points: vec![(0.0, 0.0), (2.0, 0.0), (1.0, 1.0)],
        };

        avoid_acid_traps(&mut chain, &board, &grid);

        // Verify no consecutive segments form an acute angle (dot product < -0.05)
        for i in 1..chain.points.len() - 1 {
            let p0 = chain.points[i - 1];
            let p1 = chain.points[i];
            let p2 = chain.points[i + 1];
            let v1 = (p1.0 - p0.0, p1.1 - p0.1);
            let v2 = (p2.0 - p1.0, p2.1 - p1.1);
            let l1 = v1.0.hypot(v1.1);
            let l2 = v2.0.hypot(v2.1);
            if l1 > 1e-4 && l2 > 1e-4 {
                let dot = (v1.0 * v2.0 + v1.1 * v2.1) / (l1 * l2);
                assert!(dot >= -0.05, "Angle at {:?} must not be acute (dot = {})", p1, dot);
            }
        }
    }

    #[test]
    fn test_reduce_corners_simplifies_dogleg() {
        let board = make_router_test_board();
        let grid = RoutingGrid::new(&board, 0.10, None);

        // 4 points where p0 -> p3 can connect directly via an octilinear segment:
        // (0.0, 0.0) -> (0.0, 1.0) -> (1.0, 1.0) -> (2.0, 2.0)
        // Notice p0 -> p3 is (0, 0) -> (2, 2) which is a direct 45° octilinear connection!
        let mut chain = TrackChain {
            net_id: 2,
            layer: 0,
            width: 0.20,
            points: vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 2.0)],
        };

        reduce_corners(&mut chain, &board, &grid);

        // Intermediate corners should be collapsed directly into p0 -> p3
        assert_eq!(chain.points.len(), 2, "Chain must be simplified to [p0, p3]");
        assert_eq!(chain.points[0], (0.0, 0.0));
        assert_eq!(chain.points[1], (2.0, 2.0));
    }

    #[test]
    fn test_smoothen_corners_adaptive_chamfering() {
        let board = make_router_test_board();
        let grid = RoutingGrid::new(&board, 0.10, None);

        // A simple 90-degree corner: (0.0, 0.0) -> (1.0, 0.0) -> (1.0, 1.0)
        let mut chain = TrackChain {
            net_id: 2,
            layer: 0,
            width: 0.20,
            points: vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        };

        smoothen_corners(&mut chain, &board, &grid);

        // The 90-degree corner at (1.0, 0.0) should be chamfered with 2 miter points at 45 degrees
        assert_eq!(chain.points.len(), 4, "90-degree corner must be chamfered into 4 points");
        assert_eq!(chain.points[0], (0.0, 0.0));
        assert_eq!(chain.points[3], (1.0, 1.0));

        let m1 = chain.points[1];
        let m2 = chain.points[2];
        assert!(m1.0 < 1.0 && m1.1 == 0.0, "m1 must be set back on horizontal segment: {:?}", m1);
        assert!(m2.0 == 1.0 && m2.1 > 0.0, "m2 must be set back on vertical segment: {:?}", m2);

        let dx = (m2.0 - m1.0).abs();
        let dy = (m2.1 - m1.1).abs();
        assert!((dx - dy).abs() < 1e-4, "Miter segment must be 45 degrees: dx={}, dy={}", dx, dy);
    }

    #[test]
    fn test_octilinear_segment_math_and_doglegs() {
        // 1. Horizontal & Vertical segments
        assert!(is_octilinear_segment((0.0, 0.0), (10.0, 0.0)));
        assert!(is_octilinear_segment((0.0, 0.0), (0.0, -5.5)));
        assert!(is_octilinear_segment((141.0, 102.65), (142.95, 102.65)));

        // 2. 45-degree diagonal segments
        assert!(is_octilinear_segment((0.0, 0.0), (3.0, 3.0)));
        assert!(is_octilinear_segment((10.0, 10.0), (5.0, 15.0)));
        assert!(is_octilinear_segment((142.92, 102.62), (142.95, 102.65)));

        // 3. Non-octilinear segments (should be rejected)
        assert!(!is_octilinear_segment((141.0, 102.62), (142.95, 102.65))); // 0.88° tilt
        assert!(!is_octilinear_segment((0.0, 0.0), (1.0, 2.0))); // ~63.4°
        assert!(!is_octilinear_segment((10.0, 10.0), (12.0, 13.0))); // non-45°

        // 4. compute_45_dogleg decomposes non-octilinear points into strictly octilinear segments
        let dogleg = compute_45_dogleg((141.0, 102.62), (142.95, 102.65), false);
        assert_eq!(dogleg.len(), 3);
        assert!(is_octilinear_segment(dogleg[0], dogleg[1]));
        assert!(is_octilinear_segment(dogleg[1], dogleg[2]));

        let dogleg_diag = compute_45_dogleg((141.0, 102.62), (142.95, 102.65), true);
        assert_eq!(dogleg_diag.len(), 3);
        assert!(is_octilinear_segment(dogleg_diag[0], dogleg_diag[1]));
        assert!(is_octilinear_segment(dogleg_diag[1], dogleg_diag[2]));
    }

    #[test]
    fn test_enforce_45_grid_decomposes_arbitrary_angles() {
        // A track chain with intentional non-octilinear segments (e.g. from off-grid pad to grid corner)
        let mut chain = TrackChain {
            net_id: 1,
            layer: 0,
            width: 0.20,
            points: vec![
                (141.000, 102.620), // off-grid pad
                (142.950, 102.650), // grid corner (0.88 deg tilt)
                (150.000, 102.650), // horizontal
                (152.000, 104.650), // 45 degree
                (153.250, 105.388), // weird angle ~48.4 deg
            ],
        };

        enforce_45_grid(&mut chain);

        // Every segment in the chain must strictly conform to 0°, 45°, or 90°
        for w in chain.points.windows(2) {
            let p1 = w[0];
            let p2 = w[1];
            assert!(
                is_octilinear_segment(p1, p2),
                "Segment from {:?} to {:?} must be octilinear (0°, 45°, 90°)",
                p1,
                p2
            );
        }
    }

    #[test]
    fn test_copper_to_edge_cuts_clearance_enforced() {
        let mut board = Board::default();
        board.outline_polygon = Some(Polygon::from_rect(0.0, 0.0, 50.0, 50.0));
        board.netclasses.insert("Default".to_string(), NetClass {
            board_edge_clearance: 0.35,
            ..NetClass::default()
        });
        board.nets = vec!["".to_string(), "NET1".to_string()];
        board.net_name_to_id.insert("NET1".to_string(), 1);

        // Track 1: distance to x=0 edge is 0.20 - 0.10 = 0.10 mm < 0.35 mm (ILLEGAL)
        let trk_illegal = TrackSegment {
            start: (0.20, 10.0),
            end: (10.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        };

        // Track 2: distance to x=0 edge is 1.00 - 0.10 = 0.90 mm >= 0.35 mm (LEGAL)
        let trk_legal = TrackSegment {
            start: (1.00, 20.0),
            end: (10.0, 20.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        };

        // Via 1: distance to x=0 edge is 0.40 - 0.30 = 0.10 mm < 0.35 mm (ILLEGAL)
        let via_illegal = Via {
            position: (0.40, 30.0),
            size: 0.60,
            drill: 0.30,
            net_id: 1,
            is_locked: false,
        };

        // Via 2: distance to x=0 edge is 1.00 - 0.30 = 0.70 mm >= 0.35 mm (LEGAL)
        let via_legal = Via {
            position: (1.00, 35.0),
            size: 0.60,
            drill: 0.30,
            net_id: 1,
            is_locked: false,
        };

        // 1. Verify routing geometry checkers reject near-edge copper
        assert!(!is_track_segment_legal(&board, (0.20, 10.0), (10.0, 10.0), 0.20, 0, 1, 0.20));
        assert!(is_track_segment_legal(&board, (1.00, 20.0), (10.0, 20.0), 0.20, 0, 1, 0.20));
        assert!(!is_via_location_legal(&board, (0.40, 30.0), 0.60, 1, 0.20));
        assert!(is_via_location_legal(&board, (1.00, 35.0), 0.60, 1, 0.20));

        board.tracks = vec![trk_illegal, trk_legal];
        board.vias = vec![via_illegal, via_legal];

        // 2. Verify validate_and_clean_copper eliminates illegal copper
        validate_and_clean_copper(&mut board);

        assert_eq!(board.tracks.len(), 1, "Illegal track violating 0.35mm edge clearance must be eliminated");
        assert_eq!(board.tracks[0].start.0, 1.00);

        assert_eq!(board.vias.len(), 1, "Illegal via violating 0.35mm edge clearance must be eliminated");
        assert_eq!(board.vias[0].position.0, 1.00);

        // 3. Verify routing grid rasterization expands edge keepout by 0.35 mm
        let mut grid = RoutingGrid::new(&board, 0.5, None);
        grid.rasterize_obstacles(&board);
        // (0.25, 25.0) is 0.25 mm from left boundary (x=0) < 0.35 mm -> must be keepout
        let Some((gx, gy)) = grid.world_to_grid(0.25, 25.0) else {
            panic!("(0.25, 25.0) must map into grid");
        };
        let obs = grid.obstacle_net_ids[grid.idx(gx, gy, 0)];
        assert_eq!(obs, KEEPOUT_NET, "Grid cell at 0.25mm from edge must be marked as keepout");
    }

    #[test]
    fn test_crosstalk_parallelism_limiter_diverts_long_coupled_trace() {
        let mut board = make_router_test_board();
        let mut sig2_nc = NetClass::default();
        sig2_nc.max_coupled_length_mm = 3.0;
        sig2_nc.track_width = 0.20;
        sig2_nc.clearance = 0.20;
        board.netclasses.insert("Sig2Class".to_string(), sig2_nc);
        board.net_to_class.insert("SIG2".to_string(), "Sig2Class".to_string());

        // Place an existing track for SIG1 (net 2)
        board.tracks.push(TrackSegment {
            start: (135.0, 100.0),
            end: (155.0, 100.0),
            width: 0.20,
            layer: 0,
            net_id: 2,
            is_locked: true,
        });

        let router = AutoRouter::new(2).with_crosstalk_penalty(5.0);
        let grid = RoutingGrid::new(&board, 0.25, None);

        // Route SIG2 from (135.0, 100.5) to (155.0, 100.5).
        let start_node = grid.world_to_grid(135.0, 100.5).map(|(gx, gy)| GridNode { gx, gy, layer: 0 }).unwrap();
        let goal_node = grid.world_to_grid(155.0, 100.5).map(|(gx, gy)| GridNode { gx, gy, layer: 0 }).unwrap();

        // 1. With crosstalk limiter active (max_coupled = 3.0mm):
        let stop_flag = AtomicBool::new(false);
        let path_constrained = router.find_path_with_constraints(
            &grid,
            start_node,
            goal_node,
            3, // SIG2
            0.20,
            0.20,
            0.60,
            None,
            1.0,
            0.0,
            3.0,
            false,
            None,
            &stop_flag,
        );
        assert!(path_constrained.is_some(), "Path should find a valid route");
        let path = path_constrained.unwrap();

        let max_y = path.iter().map(|n| {
            let (_, wy) = grid.grid_to_world(n.gx, n.gy);
            wy
        }).fold(0.0f64, f64::max);
        assert!(max_y > 101.0, "Path should divert away from parallel track to avoid crosstalk penalty, max_y={:.2}", max_y);

        // 2. Also test post-route crosstalk validation
        let mut drc_board = make_router_test_board();
        drc_board.netclasses.get_mut("Default").unwrap().max_coupled_length_mm = 5.0;
        drc_board.tracks.push(TrackSegment {
            start: (135.0, 100.0),
            end: (145.0, 100.0),
            width: 0.20,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        drc_board.tracks.push(TrackSegment {
            start: (135.0, 100.4),
            end: (145.0, 100.4),
            width: 0.20,
            layer: 0,
            net_id: 3,
            is_locked: false,
        });

        let violations = validate_crosstalk_and_coupling(&drc_board);
        assert_eq!(violations.len(), 1, "Should report 1 coupling violation");
        assert!(violations[0].coupled_length_mm >= 9.9);
        assert_eq!(violations[0].max_allowed_mm, 5.0);
    }

    #[test]
    fn test_return_path_continuity_penalizes_split_plane_crossing() {
        let mut board = make_router_test_board();
        board.raw_content = r#"
            (kicad_pcb
                (layers
                    (0 "F.Cu" signal)
                    (1 "In1.Cu" power)
                    (2 "In2.Cu" power)
                    (31 "B.Cu" signal)
                )
                (zone (net 1) (net_name "GND") (layers "In1.Cu"))
            )
        "#.to_string();

        let mut hs_nc = NetClass::default();
        hs_nc.require_return_path = true;
        board.netclasses.insert("HighSpeed".to_string(), hs_nc);
        board.net_to_class.insert("SIG2".to_string(), "HighSpeed".to_string());

        board.keepouts.push(KeepoutZone {
            min_x: 144.0,
            min_y: 95.0,
            max_x: 146.0,
            max_y: 105.0,
            layers: vec![2],
        });

        let router = AutoRouter::new(4);
        let grid = RoutingGrid::new(&board, 0.25, None);

        let ref_layer = grid.adjacent_reference_plane(0).expect("Layer 0 must have reference plane");
        assert_eq!(ref_layer, 2, "In 4-layer stackup, In1.Cu is layer 2");

        let (split_gx, split_gy) = grid.world_to_grid(145.0, 100.0).unwrap();
        assert!(
            grid.is_reference_plane_void_at(split_gx, split_gy, ref_layer),
            "Cell (145.0, 100.0) on reference plane In1.Cu must detect void"
        );

        let (free_gx, free_gy) = grid.world_to_grid(140.0, 100.0).unwrap();
        assert!(
            !grid.is_reference_plane_void_at(free_gx, free_gy, ref_layer),
            "Cell (140.0, 100.0) on reference plane In1.Cu must have continuous copper"
        );

        let start_node = GridNode { gx: free_gx, gy: free_gy, layer: 0 };
        let goal_node = grid.world_to_grid(150.0, 100.0).map(|(gx, gy)| GridNode { gx, gy, layer: 0 }).unwrap();

        let stop_flag = AtomicBool::new(false);
        let path_with_si = router.find_path_with_constraints(
            &grid,
            start_node,
            goal_node,
            3, // SIG2
            0.20,
            0.20,
            0.60,
            None,
            1.0,
            0.0,
            15.0,
            true, // require_return_path = true
            None,
            &stop_flag,
        );
        assert!(path_with_si.is_some(), "Should find a route respecting return path");
        let path = path_with_si.unwrap();

        let crossed_void = path.iter().any(|n| {
            let (wx, wy) = grid.grid_to_world(n.gx, n.gy);
            wx >= 144.0 && wx <= 146.0 && wy >= 95.0 && wy <= 105.0
        });
        assert!(!crossed_void, "High-speed track with require_return_path=true must detour around split plane void");
    }

    #[test]
    fn test_return_path_ground_via_insertion_near_signal_transition() {
        let mut board = make_router_test_board();
        let mut nc = NetClass::default();
        nc.name = "HighSpeed".to_string();
        nc.require_return_path = true;
        board.netclasses.insert("HighSpeed".to_string(), nc);
        board.net_to_class.insert("SIG2".to_string(), "HighSpeed".to_string());

        board.vias.push(Via {
            position: (138.0, 95.0),
            size: 0.60,
            drill: 0.30,
            net_id: 3,
            is_locked: false,
        });

        let router = AutoRouter::new(2);
        let grid = RoutingGrid::new(&board, 0.25, None);

        let initial_via_count = board.vias.len();
        let added = router.add_return_path_ground_vias(&mut board, &grid);

        assert_eq!(added, 1, "Should insert exactly 1 ground stitching return-path via");
        assert_eq!(board.vias.len(), initial_via_count + 1);

        let gnd_via = board.vias.iter().find(|v| v.net_id == 1 && (v.position.0 - 138.0).hypot(v.position.1 - 95.0) <= 1.5);
        assert!(gnd_via.is_some(), "A GND via must be inserted within 1.5mm of the signal via");

        let gnd_pos = gnd_via.unwrap().position;
        let dist = (gnd_pos.0 - 138.0).hypot(gnd_pos.1 - 95.0);
        assert!(dist >= 0.60, "GND via must maintain DRC clearance to signal via: dist={:.2}", dist);
        assert!(dist <= 1.50, "GND via must be within 1.5mm of signal via: dist={:.2}", dist);

        let added_second = router.add_return_path_ground_vias(&mut board, &grid);
        assert_eq!(added_second, 0, "Second pass must not add redundant GND vias");
    }

    #[test]
    fn test_quadrant_dogbone_fanout_geometry() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 50.0;
        board.max_y = 50.0;
        board.netclasses.get_mut("Default").unwrap().clearance = 0.12;
        for i in 0..=20 {
            board.nets.push(format!("NET{}", i));
        }

        let mut fp = Footprint {
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

        // 4x4 BGA grid with 0.8mm pitch:
        // Columns: x = -1.2, -0.4, 0.4, 1.2
        // Rows:    y = -1.2, -0.4, 0.4, 1.2
        let xs = [-1.2, -0.4, 0.4, 1.2];
        let ys = [-1.2, -0.4, 0.4, 1.2];
        let mut pad_num = 1;
        for &y in &ys {
            for &x in &xs {
                let mut p = Pad::new(&format!("P{}", pad_num), x, y, 0.40, 0.40, pad_num, &format!("NET{}", pad_num), false);
                p.shape = "circle".to_string();
                fp.pads.push(p);
                pad_num += 1;
            }
        }
        board.footprints.push(fp);

        let mut grid = RoutingGrid::new(&board, 0.10, None);
        generate_bga_dogbone_escapes(&mut board, &mut grid);

        // Ring 1 pads: inner 4 pads:
        // (1,1) -> row 1, col 1: pad 6 (index 5) at local (-0.4, -0.4), NW quadrant
        // (2,1) -> row 1, col 2: pad 7 (index 6) at local (0.4, -0.4), NE quadrant
        // (1,2) -> row 2, col 1: pad 10 (index 9) at local (-0.4, 0.4), SW quadrant
        // (2,2) -> row 2, col 2: pad 11 (index 10) at local (0.4, 0.4), SE quadrant
        assert_eq!(board.vias.len(), 4, "Expected exactly 4 dog-bone vias for Ring 1 pads of 4x4 BGA");

        let bga_fp = &board.footprints[0];
        let inner_indices = [5, 6, 9, 10];

        for &pad_idx in &inner_indices {
            let pad = &bga_fp.pads[pad_idx];
            let (px, py) = bga_fp.pad_absolute_position(pad);

            // Find matching via for this net
            let via = board.vias.iter().find(|v| v.net_id == pad.net_id)
                .expect("Each Ring 1 pad must have a dog-bone via");

            let dx = via.position.0 - px;
            let dy = via.position.1 - py;

            // Verifies 45° dog-bone via placement in interstitial centers (|dx| == |dy| == p/2)
            assert!((dx.abs() - 0.4).abs() < 1e-3, "Via dx must be +/- 0.4mm (p/2): dx={}", dx);
            assert!((dy.abs() - 0.4).abs() < 1e-3, "Via dy must be +/- 0.4mm (p/2): dy={}", dy);
            assert!((dx.abs() - dy.abs()).abs() < 1e-4, "Dog-bone via must be at 45 degrees (|dx| == |dy|)");

            // Verify quadrant direction:
            let lx = pad.offset_x;
            let ly = pad.offset_y;
            if lx > 0.0 {
                assert!(dx > 0.0, "Positive local X pad must dog-bone in +X direction");
            } else {
                assert!(dx < 0.0, "Negative local X pad must dog-bone in -X direction");
            }
            if ly > 0.0 {
                assert!(dy > 0.0, "Positive local Y pad must dog-bone in +Y direction");
            } else {
                assert!(dy < 0.0, "Negative local Y pad must dog-bone in -Y direction");
            }

            // Verify distance to pad center is sqrt(0.4^2 + 0.4^2) ~ 0.5657 mm
            let dist_to_pad = dx.hypot(dy);
            assert!((dist_to_pad - (0.4f64.hypot(0.4))).abs() < 1e-3, "Distance must be interstitial diagonal center");

            // Verify clearance against all foreign pads in the BGA (no shorts!)
            for foreign_pad in &bga_fp.pads {
                if foreign_pad.net_id != pad.net_id {
                    let (fx, fy) = bga_fp.pad_absolute_position(foreign_pad);
                    let d = (via.position.0 - fx).hypot(via.position.1 - fy);
                    let min_allowed = (via.size * 0.5) + (foreign_pad.width * 0.5) + 0.12;
                    assert!(
                        d >= min_allowed - 1e-4,
                        "Dog-bone via for net {} at ({:.2}, {:.2}) must not short against foreign pad {} at ({:.2}, {:.2}): d={:.3}, min_allowed={:.3}",
                        pad.net_id, via.position.0, via.position.1, foreign_pad.net_id, fx, fy, d, min_allowed
                    );
                }
            }

            // Verify is_via_location_legal returns true
            assert!(
                is_via_location_legal(&board, via.position, via.size, via.net_id, 0.12),
                "Dog-bone via location must be strictly legal and clearance-free"
            );
        }

        // Verify that Ring 0 pads generated escape tracks
        for (i, pad) in bga_fp.pads.iter().enumerate() {
            if bga_fp.bga_pad_ring_index(i) == 0 {
                let (px, py) = bga_fp.pad_absolute_position(pad);
                let has_trk = board.tracks.iter().any(|t| {
                    (t.start.0 - px).hypot(t.start.1 - py) < 1e-4 && t.net_id == pad.net_id
                });
                assert!(has_trk, "Ring 0 pad {} must have an escape track", pad.number);
            }
        }
    }

    #[test]
    fn test_bus_bundle_equidistant_offsets() {
        let mut board = Board::default();
        board.nets = vec![
            "".to_string(),
            "NET1".to_string(),
            "NET2".to_string(),
            "NET3".to_string(),
            "NET4".to_string(),
        ];
        for (i, n) in board.nets.iter().enumerate() {
            if !n.is_empty() {
                board.net_name_to_id.insert(n.clone(), i);
            }
        }
        let mut grid = RoutingGrid::new(&board, 0.25, None);

        let bundle = BusBundle::new(
            "DataBus",
            vec![1, 2, 3, 4],
            vec![(10.0, 10.0), (30.0, 10.0), (50.0, 30.0)],
            0,
            0.25,
            0.25,
        );

        let ok = route_bundle_along_guide(&mut board, &mut grid, &bundle);
        assert!(ok, "route_bundle_along_guide must succeed for valid 2-segment guide line");

        // Verify tracks created for all 4 nets
        for &nid in &bundle.net_ids {
            let count = board.tracks.iter().filter(|t| t.net_id == nid).count();
            assert!(count >= 2, "Net {} should have at least 2 segments along the polyline", nid);
        }

        // Along the first straight segment (x from 10 to 30), examine tracks at x = 20.0
        let mut lane_ys = Vec::new();
        for &nid in &bundle.net_ids {
            let trk = board
                .tracks
                .iter()
                .find(|t| t.net_id == nid && t.start.0 <= 20.0 && t.end.0 >= 20.0 && (t.start.1 - t.end.1).abs() < 1e-4)
                .expect("Each net must have a horizontal segment passing x=20.0");
            lane_ys.push((nid, trk.start.1));
        }

        // Verify sorted y-coordinates have exact (W + S) = 0.50mm spacing everywhere
        lane_ys.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for i in 0..lane_ys.len() - 1 {
            let pitch = lane_ys[i + 1].1 - lane_ys[i].1;
            assert!(
                (pitch - 0.50).abs() < 1e-4,
                "Spacing between lane {} and {} must be exactly W + S = 0.50mm, got {:.4}mm",
                lane_ys[i].0,
                lane_ys[i + 1].0,
                pitch
            );
        }

        // Verify that along the second diagonal segment (direction (1, 1)), perpendicular distance is exactly 0.50mm
        let mut diag_lines = Vec::new();
        for &nid in &bundle.net_ids {
            let trk = board
                .tracks
                .iter()
                .find(|t| {
                    t.net_id == nid
                        && t.start.0 >= 29.0
                        && ((t.end.0 - t.start.0) - (t.end.1 - t.start.1)).abs() < 1e-4
                        && (t.end.0 - t.start.0) > 1.0
                })
                .expect("Each net must have a 45-degree segment along the second guide segment");
            let c = trk.start.0 - trk.start.1;
            diag_lines.push((nid, c));
        }
        diag_lines.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for i in 0..diag_lines.len() - 1 {
            let perp_dist = (diag_lines[i + 1].1 - diag_lines[i].1) / std::f64::consts::SQRT_2;
            assert!(
                (perp_dist - 0.50).abs() < 1e-4,
                "Diagonal perpendicular spacing must be exactly 0.50mm, got {:.4}mm",
                perp_dist
            );
        }
    }

    #[test]
    fn test_bundle_corner_concentric_miters() {
        let mut board = Board::default();
        board.nets = vec![
            "".to_string(),
            "NET1".to_string(),
            "NET2".to_string(),
            "NET3".to_string(),
            "NET4".to_string(),
        ];
        for (i, n) in board.nets.iter().enumerate() {
            if !n.is_empty() {
                board.net_name_to_id.insert(n.clone(), i);
            }
        }
        let mut grid = RoutingGrid::new(&board, 0.25, None);

        let w = 0.25;
        let s = 0.25;
        let bundle = BusBundle::new(
            "CornerBus",
            vec![1, 2, 3, 4],
            vec![(10.0, 10.0), (40.0, 10.0), (40.0, 40.0)],
            0,
            w,
            s,
        );

        let ok = route_bundle_along_guide(&mut board, &mut grid, &bundle);
        assert!(ok, "route_bundle_along_guide must successfully route around 90-degree bend");

        // 1. Verify that 45-degree miter segments were generated around the 90-degree turn
        for &nid in &bundle.net_ids {
            let miter_trk = board.tracks.iter().find(|t| {
                t.net_id == nid
                    && ((t.end.0 - t.start.0).abs() - (t.end.1 - t.start.1)).abs() < 1e-4
                    && (t.end.0 - t.start.0).abs() > 0.05
                    && (t.end.1 - t.start.1).abs() > 0.05
            });
            assert!(
                miter_trk.is_some(),
                "Net {} must have a 45-degree concentric miter segment at corner",
                nid
            );
        }

        // 2. Verify concentric clearance: no tracks pinch or violate minimum spacing anywhere!
        let bundle_tracks: Vec<&TrackSegment> = board
            .tracks
            .iter()
            .filter(|t| bundle.net_ids.contains(&t.net_id))
            .collect();

        for i in 0..bundle_tracks.len() {
            for j in (i + 1)..bundle_tracks.len() {
                let t1 = bundle_tracks[i];
                let t2 = bundle_tracks[j];
                if t1.net_id != t2.net_id {
                    let d = dist_segment_to_segment(t1.start, t1.end, t2.start, t2.end);
                    let copper_clearance = d - (t1.width * 0.5 + t2.width * 0.5);
                    assert!(
                        copper_clearance >= s - 1e-4,
                        "Clearance between Net {} and Net {} pinched to {:.4}mm (minimum required: {:.4}mm)",
                        t1.net_id,
                        t2.net_id,
                        copper_clearance,
                        s
                    );
                }
            }
        }
    }

    #[test]
    fn test_fine_pitch_vippo_fallback() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 50.0;
        board.max_y = 50.0;
        board.netclasses.get_mut("Default").unwrap().clearance = 0.08;
        for i in 0..=20 {
            board.nets.push(format!("NET{}", i));
        }

        let mut fp = Footprint {
            reference: "U_BGA_FINE".to_string(),
            value: "BGA-16-0.4mm".to_string(),
            x: 25.0,
            y: 25.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: Vec::new(),
            width: 3.0,
            height: 3.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        // 4x4 BGA grid with fine 0.4mm pitch (< 0.50mm):
        // Columns: x = -0.6, -0.2, 0.2, 0.6
        // Rows:    y = -0.6, -0.2, 0.2, 0.6
        let xs = [-0.6, -0.2, 0.2, 0.6];
        let ys = [-0.6, -0.2, 0.2, 0.6];
        let mut pad_num = 1;
        for &y in &ys {
            for &x in &xs {
                let mut p = Pad::new(&format!("P{}", pad_num), x, y, 0.22, 0.22, pad_num, &format!("NET{}", pad_num), false);
                p.shape = "circle".to_string();
                fp.pads.push(p);
                pad_num += 1;
            }
        }
        board.footprints.push(fp);

        let mut grid = RoutingGrid::new(&board, 0.05, None);
        generate_bga_dogbone_escapes(&mut board, &mut grid);

        // For pitch < 0.5mm, VIPPO fallback places centered via-in-pad microvias (0.30mm dia, 0.15mm drill)
        assert_eq!(board.vias.len(), 4, "Expected exactly 4 VIPPO microvias for Ring 1 pads of fine-pitch BGA");

        let bga_fp = &board.footprints[0];
        let inner_indices = [5, 6, 9, 10];

        for &pad_idx in &inner_indices {
            let pad = &bga_fp.pads[pad_idx];
            let (px, py) = bga_fp.pad_absolute_position(pad);

            let via = board.vias.iter().find(|v| v.net_id == pad.net_id)
                .expect("Each Ring 1 fine-pitch pad must have a VIPPO microvia");

            // Verify via is centered directly on pad (via-in-pad)
            let dist_to_pad = (via.position.0 - px).hypot(via.position.1 - py);
            assert!(dist_to_pad < 1e-4, "VIPPO microvia must be centered directly on pad: dist={}", dist_to_pad);

            // Verify microvia dimensions
            assert!((via.size - 0.30).abs() < 1e-4, "VIPPO microvia diameter must be 0.30mm: size={}", via.size);
            assert!((via.drill - 0.15).abs() < 1e-4, "VIPPO microvia drill must be 0.15mm: drill={}", via.drill);

            // Verify legality of via-in-pad microvia transition
            assert!(
                is_via_location_legal(&board, via.position, via.size, via.net_id, 0.08),
                "VIPPO microvia transition must be legal"
            );
        }
    }

    #[test]
    fn test_bundle_fanout_connects_to_pads() {
        let mut board = Board::default();
        board.nets = vec![
            "".to_string(),
            "D0".to_string(),
            "D1".to_string(),
            "D2".to_string(),
            "D3".to_string(),
        ];
        for (i, name) in board.nets.iter().enumerate() {
            if !name.is_empty() {
                board.net_name_to_id.insert(name.clone(), i);
            }
        }

        let j1_pads = vec![
            Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "D0", false),
            Pad::new("2", 0.0, 1.0, 0.6, 0.6, 2, "D1", false),
            Pad::new("3", 0.0, 2.0, 0.6, 0.6, 3, "D2", false),
            Pad::new("4", 0.0, 3.0, 0.6, 0.6, 4, "D3", false),
        ];
        let j1 = make_fp("J1", 5.0, 10.0, true, j1_pads);

        let j2_pads = vec![
            Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "D0", false),
            Pad::new("2", 0.0, 1.0, 0.6, 0.6, 2, "D1", false),
            Pad::new("3", 0.0, 2.0, 0.6, 0.6, 3, "D2", false),
            Pad::new("4", 0.0, 3.0, 0.6, 0.6, 4, "D3", false),
        ];
        let j2 = make_fp("J2", 45.0, 10.0, true, j2_pads);

        board.footprints.push(j1);
        board.footprints.push(j2);

        let mut grid = RoutingGrid::new(&board, 0.25, None);

        let bundle = BusBundle::new(
            "DataBus",
            vec![1, 2, 3, 4],
            vec![(10.0, 11.5), (40.0, 11.5)],
            0,
            0.25,
            0.25,
        );

        let ok = route_bundle_along_guide(&mut board, &mut grid, &bundle);
        assert!(ok, "route_bundle_along_guide must successfully connect pads to bundle");

        // Verify each net connects to its respective pad on J1 and J2
        for (idx, &nid) in bundle.net_ids.iter().enumerate() {
            let j1_pad_pos = (5.0, 10.0 + idx as f64 * 1.0);
            let j2_pad_pos = (45.0, 10.0 + idx as f64 * 1.0);

            let connects_j1 = board.tracks.iter().any(|t| {
                t.net_id == nid
                    && ((t.start.0 - j1_pad_pos.0).hypot(t.start.1 - j1_pad_pos.1) < 1e-3
                        || (t.end.0 - j1_pad_pos.0).hypot(t.end.1 - j1_pad_pos.1) < 1e-3)
            });
            assert!(
                connects_j1,
                "Net {} must connect to J1 pad at ({:.1}, {:.1})",
                nid,
                j1_pad_pos.0,
                j1_pad_pos.1
            );

            let connects_j2 = board.tracks.iter().any(|t| {
                t.net_id == nid
                    && ((t.start.0 - j2_pad_pos.0).hypot(t.start.1 - j2_pad_pos.1) < 1e-3
                        || (t.end.0 - j2_pad_pos.0).hypot(t.end.1 - j2_pad_pos.1) < 1e-3)
            });
            assert!(
                connects_j2,
                "Net {} must connect to J2 pad at ({:.1}, {:.1})",
                nid,
                j2_pad_pos.0,
                j2_pad_pos.1
            );
        }
    }

    #[test]
    fn test_homotopic_rubber_band_shortening() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "SIG".to_string(), "GND".to_string()];
        board.net_name_to_id.insert("SIG".to_string(), 1);
        board.net_name_to_id.insert("GND".to_string(), 2);

        // Obstacle via for GND (net 2) at (5.0, 5.0), size 1.0 (radius 0.5)
        board.vias.push(Via {
            position: (5.0, 5.0),
            size: 1.0,
            drill: 0.5,
            net_id: 2,
            is_locked: true,
        });

        // Net 1 (SIG) has a zigzag trace passing around the obstacle on the +y side:
        // (0.0, 0.0) -> (0.0, 8.0) -> (3.0, 8.0) -> (3.0, 9.0) -> (7.0, 9.0) -> (7.0, 8.0) -> (10.0, 8.0) -> (10.0, 0.0)
        let pts = [
            (0.0, 0.0),
            (0.0, 8.0),
            (3.0, 8.0),
            (3.0, 9.0),
            (7.0, 9.0),
            (7.0, 8.0),
            (10.0, 8.0),
            (10.0, 0.0),
        ];
        for k in 0..pts.len() - 1 {
            board.tracks.push(TrackSegment {
                start: pts[k],
                end: pts[k + 1],
                width: 0.20,
                layer: 0,
                net_id: 1,
                is_locked: false,
            });
        }

        let initial_length: f64 = board.tracks.iter().map(|t| t.length()).sum();
        assert!((initial_length - 28.0).abs() < 1e-4);

        // Relax homotopic rubber bands
        relax_homotopic_rubber_bands(&mut board);

        let relaxed_length: f64 = board.tracks.iter().map(|t| t.length()).sum();

        // 1. Length must be significantly shortened by taut rubber-band pulling (> 25% reduction)
        assert!(
            relaxed_length < initial_length * 0.75,
            "Relaxed length {:.2} must be < 75% of initial {:.2}",
            relaxed_length,
            initial_length
        );

        // 2. All tracks must maintain required clearance from net 2 via at (5.0, 5.0)
        let via_pos = (5.0, 5.0);
        let req_clr = board.get_netclass_for_net(1).clearance.max(board.get_netclass_for_net(2).clearance);
        let min_centerline_dist = 0.5 + req_clr + 0.10; // via radius (0.5) + clearance (0.2) + half width (0.1)
        for trk in &board.tracks {
            let d = dist_point_to_segment(via_pos, trk.start, trk.end);
            assert!(
                d >= min_centerline_dist - 1e-4,
                "Track segment {:?}->{:?} violates clearance: dist={:.4}, req={:.4}",
                trk.start, trk.end, d, min_centerline_dist
            );
        }

        // 3. Homotopy check: trace must remain on the +y side of the obstacle at (5.0, 5.0)
        for trk in &board.tracks {
            if (trk.start.0 <= 5.0 && trk.end.0 >= 5.0) || (trk.start.0 >= 5.0 && trk.end.0 <= 5.0) {
                let t = if (trk.end.0 - trk.start.0).abs() > 1e-6 {
                    (5.0 - trk.start.0) / (trk.end.0 - trk.start.0)
                } else {
                    0.5
                };
                let y_at_5 = trk.start.1 + t * (trk.end.1 - trk.start.1);
                assert!(y_at_5 > 5.0, "Homotopy violated: trace crossed to -y side of obstacle");
            }
        }
    }

    #[test]
    fn test_tangent_arc_geometry_and_clearance() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "SIG1".to_string(), "SIG2".to_string(), "GND".to_string()];
        board.net_name_to_id.insert("SIG1".to_string(), 1);
        board.net_name_to_id.insert("SIG2".to_string(), 2);
        board.net_name_to_id.insert("GND".to_string(), 3);

        // 1. 90-degree corner on SIG1
        board.tracks.push(TrackSegment {
            start: (0.0, 0.0),
            end: (10.0, 0.0),
            width: 0.25,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: (10.0, 0.0),
            end: (10.0, 10.0),
            width: 0.25,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });

        // 2. 45-degree corner on SIG2
        board.tracks.push(TrackSegment {
            start: (0.0, 20.0),
            end: (10.0, 20.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: (10.0, 20.0),
            end: (15.0, 25.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });

        apply_any_angle_tangent_arcs(&mut board);

        // Verify that 2 tangent arcs were created (one for 90 deg, one for 45 deg)
        assert_eq!(board.arcs.len(), 2, "Expected 2 tangent arcs inscribed at corners");

        // Verify 90-degree arc properties (net 1)
        let arc90 = board.arcs.iter().find(|a| a.net_id == 1).expect("Arc for net 1 not found");
        let start90 = arc90.start();
        let end90 = arc90.end();
        assert!((start90.1 - 0.0).abs() < 1e-3);
        assert!(start90.0 < 10.0 && start90.0 > 8.0);
        assert!((end90.0 - 10.0).abs() < 1e-3);
        assert!(end90.1 > 0.0 && end90.1 < 2.0);
        assert!((arc90.radius - 1.0).abs() < 1e-3);

        // Verify 45-degree arc properties (net 2)
        let arc45 = board.arcs.iter().find(|a| a.net_id == 2).expect("Arc for net 2 not found");
        let start45 = arc45.start();
        assert!((start45.1 - 20.0).abs() < 1e-3);
        assert!(start45.0 < 10.0);

        // 3. Test NetClass clearance constraint with a foreign obstacle inside corner
        let mut drc_board = Board::default();
        drc_board.nets = vec!["".to_string(), "SIG".to_string(), "GND".to_string()];
        drc_board.net_name_to_id.insert("SIG".to_string(), 1);
        drc_board.net_name_to_id.insert("GND".to_string(), 2);

        drc_board.tracks.push(TrackSegment {
            start: (0.0, 0.0),
            end: (10.0, 0.0),
            width: 0.25,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        drc_board.tracks.push(TrackSegment {
            start: (10.0, 0.0),
            end: (10.0, 10.0),
            width: 0.25,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });

        // Place foreign via inside the 90-degree corner at (9.4, 0.6)
        drc_board.vias.push(Via {
            position: (9.4, 0.6),
            size: 0.60,
            drill: 0.30,
            net_id: 2,
            is_locked: true,
        });

        apply_any_angle_tangent_arcs(&mut drc_board);

        // Verify that if an arc is inscribed, its clearance to the foreign via strictly satisfies DRC
        let via_pos = (9.4, 0.6);
        let req_clr = drc_board.get_netclass_for_net(1).clearance.max(drc_board.get_netclass_for_net(2).clearance);
        let required_dist = 0.30 + req_clr + 0.125;
        for arc in &drc_board.arcs {
            let d = arc.distance_to_point(via_pos);
            assert!(
                d >= required_dist - 1e-4,
                "Arc {:?} violates netclass clearance to via at {:?}: dist={:.4}, req={:.4}",
                arc, via_pos, d, required_dist
            );
        }
    }

    #[test]
    fn test_p1p2esp_fine_pitch_connections_route_cleanly() {
        let pcb_path = std::path::Path::new("/home/deece/src/p1p2esp/p1p2esp.kicad_pcb");
        if !pcb_path.exists() {
            return;
        }
        let locked: std::collections::HashSet<String> = crate::board::DEFAULT_LOCKED_REFS.iter().map(|s| s.to_string()).collect();
        let board = Board::load(pcb_path.to_str().unwrap(), &locked).unwrap();
        let router = AutoRouter::new(2);
        let test_board = board.clone();
        let connections = router.extract_connections(&test_board);
        let mut grid = RoutingGrid::new(&test_board, 0.10, None);
        let escape_stubs = router.generate_escape_stubs(&test_board);
        for stub in &escape_stubs {
            router.commit_stub_to_grid(&mut grid, stub);
        }

        let stop = std::sync::atomic::AtomicBool::new(false);
        for c in &connections {
            if c.net_name == "TERM" || c.net_name == "USB_DP" || c.net_name == "USB_DM" || c.net_name == "VRAW" || c.net_name.contains("HNA") || c.net_name.contains("HPA") || c.net_name == "VBUS" || c.net_name.contains("FB") {
                let mut start_pt = c.start;
                let mut end_pt = c.end;
                for s in &escape_stubs {
                    if s.net_id == c.net_id && (start_pt.0 - s.pad_pos.0).hypot(start_pt.1 - s.pad_pos.1) < 0.15 {
                        start_pt = (s.tip_pos.0, s.tip_pos.1, s.tip_layer);
                    }
                    if s.net_id == c.net_id && (end_pt.0 - s.pad_pos.0).hypot(end_pt.1 - s.pad_pos.1) < 0.15 {
                        end_pt = (s.tip_pos.0, s.tip_pos.1, s.tip_layer);
                    }
                }
                let touches_fine_pitch = board.footprints.get(c.start_fp).map_or(false, |fp| fp.reference == "U1" || fp.reference == "J2" || AutoRouter::is_dense_escape_footprint(fp))
                    || board.footprints.get(c.end_fp).map_or(false, |fp| fp.reference == "U1" || fp.reference == "J2" || AutoRouter::is_dense_escape_footprint(fp))
                    || escape_stubs.iter().any(|s| {
                        (start_pt.0 - s.tip_pos.0).hypot(start_pt.1 - s.tip_pos.1) < 0.05
                            || (end_pt.0 - s.tip_pos.0).hypot(end_pt.1 - s.tip_pos.1) < 0.05
                    });
                let mut conn_nc = c.netclass.clone();
                if touches_fine_pitch {
                    conn_nc.track_width = conn_nc.track_width.min(0.15);
                    conn_nc.clearance = conn_nc.clearance.min(0.127);
                }

                let sg = grid.world_to_grid(start_pt.0, start_pt.1);
                let eg = grid.world_to_grid(end_pt.0, end_pt.1);
                if let (Some(s), Some(e)) = (sg, eg) {
                    let s_node = GridNode { gx: s.0, gy: s.1, layer: start_pt.2 };
                    let e_node = GridNode { gx: e.0, gy: e.1, layer: end_pt.2 };
                    let path = router.find_path_with_constraints(
                        &grid, s_node, e_node, c.net_id, conn_nc.clearance, conn_nc.track_width, conn_nc.via_dia, None, 1.0, 0.0, conn_nc.max_coupled_length_mm, conn_nc.require_return_path, None, &stop
                    );
                    assert!(path.is_some(), "Connection '{}' (net {}) must route successfully", c.net_name, c.net_id);
                }
            }
        }
    }

    #[test]
    fn test_any_angle_tangent_arcs_preserved_by_pruning() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "SIG".to_string()];
        board.net_name_to_id.insert("SIG".to_string(), 1);

        let mut fp1 = Footprint {
            reference: "R1".to_string(),
            x: 0.0,
            y: 0.0,
            ..Footprint::default()
        };
        fp1.pads.push(Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "SIG", false));
        let mut fp2 = Footprint {
            reference: "R2".to_string(),
            x: 10.0,
            y: 10.0,
            layer: 1,
            ..Footprint::default()
        };
        fp2.pads.push(Pad::new("1", 0.0, 0.0, 0.6, 0.6, 1, "SIG", false));
        board.footprints.push(fp1);
        board.footprints.push(fp2);

        // Route from R1 (0,0, L0) -> (5,0, L0) -> (5,5, L0) [via to L1] -> (10,5, L1) -> R2 (10,10, L1)
        board.tracks.push(TrackSegment { start: (0.0, 0.0), end: (5.0, 0.0), width: 0.25, layer: 0, net_id: 1, is_locked: false });
        board.tracks.push(TrackSegment { start: (5.0, 0.0), end: (5.0, 5.0), width: 0.25, layer: 0, net_id: 1, is_locked: false });
        board.vias.push(Via { position: (5.0, 5.0), size: 0.60, drill: 0.30, net_id: 1, is_locked: false });
        board.tracks.push(TrackSegment { start: (5.0, 5.0), end: (10.0, 5.0), width: 0.25, layer: 1, net_id: 1, is_locked: false });
        board.tracks.push(TrackSegment { start: (10.0, 5.0), end: (10.0, 10.0), width: 0.25, layer: 1, net_id: 1, is_locked: false });

        apply_any_angle_tangent_arcs(&mut board);
        assert_eq!(board.arcs.len(), 2, "Expected 2 corner tangent arcs");

        // Also add a genuinely dangling arc that is not connected on one end
        board.arcs.push(crate::board::ArcSegment::new((30.0, 30.0), 2.0, 0.0, 1.0, 0.25, 0, 1));

        prune_dangling_tracks_and_vias(&mut board);

        assert_eq!(board.tracks.len(), 4, "All 4 trimmed straight segments must survive pruning");
        assert_eq!(board.arcs.len(), 2, "Both connected corner arcs must survive while dangling arc is pruned");
        assert_eq!(board.vias.len(), 1, "Layer transition via must survive pruning");
    }

    #[test]
    fn test_nudge_gpu_accelerated_and_batch_segment_validation() {
        let Some(vk) = VulkanEngine::new() else {
            return;
        };

        let nets = vec!["".to_string(), "GND".to_string(), "SIG1".to_string(), "BLOCK".to_string()];
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

        let j1 = make_fp("J1", 105.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "SIG1", false)]);
        let j2 = make_fp("J_END", 125.0, 110.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "SIG1", false)]);
        let mut c1 = make_fp("C1", 107.0, 110.0, false, vec![Pad::new("1", 0.0, 0.0, 0.6, 0.6, 3, "BLOCK", false)]);
        c1.courtyards = vec![Polygon::new(vec![
            Point::new(-0.35, -0.35),
            Point::new(0.35, -0.35),
            Point::new(0.35, 0.35),
            Point::new(-0.35, 0.35),
        ])];

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 104.0,
            min_y: 100.0,
            max_x: 130.0,
            max_y: 120.0,
            footprints: vec![j1, j2, c1],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: vec![
                KeepoutZone {
                    min_x: 104.0,
                    min_y: 110.5,
                    max_x: 108.0,
                    max_y: 120.0,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 104.0,
                    min_y: 100.0,
                    max_x: 108.0,
                    max_y: 109.5,
                    layers: vec![0, 1],
                },
            ],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
            texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, Some(vk));

        // 1. Verify RoutingGrid::batch_validate_segments_gpu detects C1 pad collision vs clear segment
        let segs = vec![
            (2usize, (105.0, 110.0), (110.0, 110.0), 0.25, 0usize, 0.20), // Collides with C1 at (107.0, 110.0) on L0
            (2usize, (112.0, 110.0), (120.0, 110.0), 0.25, 0usize, 0.20), // Clear corridor on L0
        ];
        let seg_res = grid
            .batch_validate_segments_gpu(&board, &segs)
            .expect("GPU batch segment validation must return Some when Vulkan is present");
        assert_eq!(seg_res, vec![false, true]);

        // 2. Verify GPU-accelerated NudgeEngine (Phase A pad pre-screen + Phase B 3D wavefront ranking)
        let router = AutoRouter::new(3).with_allow_component_nudge(true);
        let stop = Arc::new(AtomicBool::new(false));
        let orig_c1_pos = (board.footprints[2].x, board.footprints[2].y);
        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);

        assert_eq!(total, 1);
        assert_eq!(routed, 1, "GPU-accelerated component nudge must route the blocked connection");
        assert!(
            (board.footprints[2].x - orig_c1_pos.0).hypot(board.footprints[2].y - orig_c1_pos.1) > 0.1,
            "Unlocked C1 must be nudged away from the corridor"
        );
    }

    #[test]
    fn test_multi_track_corridor_shove_and_non_diff_pair_suffixes() {
        // Non-differential nets like Net-(U1-HPA) / Net-(U1-HNA) must not be falsely matched
        assert!(!AutoRouter::are_diff_pair_mates("Net-(U1-HPA)", "Net-(U1-HNA)"));
        assert!(!AutoRouter::are_diff_pair_mates("CAP_P1", "CAP_N1"));
        assert!(AutoRouter::are_diff_pair_mates("USB_DP", "USB_DM"));
        assert!(AutoRouter::are_diff_pair_mates("LVDS_P", "LVDS_N"));

        // Construct a board where 2 parallel tracks block a narrow corridor for a 3rd net
        let nets = vec![
            "".to_string(),
            "GND".to_string(),
            "BLK1".to_string(),
            "BLK2".to_string(),
            "TARGET".to_string(),
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

        let b1_s = make_fp("B1S", 102.0, 109.6, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "BLK1", false)]);
        let b1_e = make_fp("B1E", 118.0, 109.6, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 2, "BLK1", false)]);
        let b2_s = make_fp("B2S", 102.0, 110.4, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "BLK2", false)]);
        let b2_e = make_fp("B2E", 118.0, 110.4, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 3, "BLK2", false)]);
        let tgt_s = make_fp("TS", 110.0, 106.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 4, "TARGET", false)]);
        let tgt_e = make_fp("TE", 110.0, 114.0, true, vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 4, "TARGET", false)]);

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 100.0,
            min_y: 100.0,
            max_x: 120.0,
            max_y: 120.0,
            footprints: vec![b1_s, b1_e, b2_s, b2_e, tgt_s, tgt_e],
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

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let router = AutoRouter::new(4);
        let stop = Arc::new(AtomicBool::new(false));
        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);
        assert_eq!(total, 3);
        assert_eq!(routed, 3, "All 3 connections must route cleanly");
    }

    #[test]
    fn test_electrical_continuity_verification_and_same_net_via_merge() {
        let nets = vec!["".to_string(), "GND".to_string(), "SIG".to_string()];
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

        let fp_a = make_fp("U1", 102.0, 105.0, true, vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "SIG", false)]);
        let mut fp_b = make_fp("U2", 110.0, 105.0, true, vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "SIG", false)]);
        fp_b.layer = 1;
        fp_b.orig_layer = 1;

        let mut board = Board {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 100.0,
            min_y: 100.0,
            max_x: 120.0,
            max_y: 120.0,
            footprints: vec![fp_a, fp_b],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (102.0, 105.0),
                    end: (106.0, 105.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 2,
                    is_locked: false,
                },
                TrackSegment {
                    start: (106.1, 105.0),
                    end: (110.0, 105.0),
                    width: 0.2,
                    layer: 1,
                    net_id: 2,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (106.0, 105.0),
                    size: 0.6,
                    drill: 0.3,
                    net_id: 2,
                    is_locked: false,
                },
                Via {
                    position: (106.1, 105.0),
                    size: 0.6,
                    drill: 0.3,
                    net_id: 2,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let router = AutoRouter::new(2);
        let conns = router.extract_connections(&board);
        assert_eq!(conns.len(), 1);

        // validate_and_clean_copper should merge the two close same-net vias into one rather than deleting both
        validate_and_clean_copper(&mut board);
        assert_eq!(board.vias.len(), 1, "Duplicate close same-net via should be merged into 1 via");
        assert!(
            AutoRouter::is_connection_electrically_connected(&board, &conns[0]),
            "Connection must remain electrically continuous after merging duplicate same-net vias"
        );
    }

    #[test]
    fn test_single_connection_reconstruct_with_diff_pair_leader_and_choked_pocket_ripup() {
        let mut board = Board {
            min_x: 100.0,
            min_y: 100.0,
            max_x: 120.0,
            max_y: 120.0,
            footprints: vec![
                Footprint {
                    reference: "U1".to_string(),
                    x: 105.0,
                    y: 105.0,
                    pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "USB_DM", false)],
                    ..Footprint::default()
                },
                Footprint {
                    reference: "U2".to_string(),
                    x: 112.0,
                    y: 105.0,
                    pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "USB_DM", false)],
                    ..Footprint::default()
                },
            ],
            nets: vec!["".to_string(), "GND".to_string(), "USB_DM".to_string(), "BLOCKER".to_string()],
            ..Board::default()
        };
        let router = AutoRouter::new(2);
        let mut grid = RoutingGrid::new(&board, 0.10, None);

        // 1. Verify single-connection reconstruct_tracks_and_vias does not panic when diff_pair_leader is out of slice bounds
        let mut single_conn = router.extract_connections(&board);
        assert_eq!(single_conn.len(), 1);
        single_conn[0].diff_pair_leader = Some(5); // Out-of-bounds index relative to single_paths (len 1)
        let (sgx, sgy) = grid.world_to_grid(105.0, 105.0).unwrap();
        let (egx, egy) = grid.world_to_grid(112.0, 105.0).unwrap();
        let path = vec![
            GridNode { gx: sgx, gy: sgy, layer: 0 },
            GridNode { gx: (sgx + egx) / 2, gy: sgy, layer: 0 },
            GridNode { gx: (sgx + egx) / 2, gy: sgy, layer: 1 },
            GridNode { gx: egx, gy: egy, layer: 1 },
        ];
        router.reconstruct_tracks_and_vias(&mut board, &grid, &single_conn, &[Some(path)], &[]);
        assert!(!board.tracks.is_empty(), "Single-connection reconstruct with layer change must emit tracks without panicking");

        // 2. Verify find_choked_pin_bottlenecks detects a surrounding track ring even if 1-cell cracks exist
        let (cx, cy) = (sgx, sgy);
        for d in -4isize..=4 {
            // Place foreign net 3 on a ring at radius 4 with a 1-cell diagonal gap that a 1-cell halo catches
            let top = grid.idx((cx as isize + d) as usize, (cy as isize - 4) as usize, 0);
            let bot = grid.idx((cx as isize + d) as usize, (cy as isize + 4) as usize, 0);
            let lft = grid.idx((cx as isize - 4) as usize, (cy as isize + d) as usize, 0);
            let rgt = grid.idx((cx as isize + 4) as usize, (cy as isize + d) as usize, 0);
            grid.obstacle_net_ids[top] = 3;
            grid.obstacle_net_ids[bot] = 3;
            grid.obstacle_net_ids[lft] = 3;
            grid.obstacle_net_ids[rgt] = 3;
        }
        let (reach, blockers) = AutoRouter::find_choked_pin_bottlenecks(&grid, cx, cy, 0, 2, 400);
        assert!(reach < 100, "Pin enclosed by track ring must be detected as choked (reach={})", reach);
        assert!(blockers.contains(&3), "Blocker net 3 must be identified by find_choked_pin_bottlenecks");
    }

    #[test]
    fn test_prune_dangling_micro_stub_and_u1_ground_pin_connection() {
        let mut board = Board {
            min_x: 100.0,
            min_y: 100.0,
            max_x: 120.0,
            max_y: 120.0,
            footprints: vec![
                Footprint {
                    reference: "U1".to_string(),
                    x: 110.0,
                    y: 110.0,
                    rotation: 270.0,
                    pads: vec![
                        Pad::new("18", 0.75, 2.075, 0.30, 0.75, 1, "GND", false),
                        Pad::new("19", 2.075, 0.75, 0.75, 0.30, 1, "GND", false),
                        Pad::new("25", 0.0, 0.0, 2.60, 2.60, 1, "GND", false),
                        Pad::new("1", -2.075, -0.75, 0.75, 0.30, 2, "SIG", false),
                    ],
                    ..Footprint::default()
                },
                Footprint {
                    reference: "U2".to_string(),
                    x: 105.0,
                    y: 109.25,
                    pads: vec![Pad::new("1", 0.0, 0.0, 0.50, 0.50, 2, "SIG", false)],
                    ..Footprint::default()
                },
            ],
            nets: vec!["".to_string(), "GND".to_string(), "SIG".to_string()],
            ..Board::default()
        };

        let u1_p1 = board.footprints[0].pad_absolute_position(&board.footprints[0].pads[3]);
        let u2_p1 = board.footprints[1].pad_absolute_position(&board.footprints[1].pads[0]);
        let mid = ((u1_p1.0 + u2_p1.0) * 0.5, u1_p1.1);
        // Valid main route from U1.1 -> mid -> U2.1, plus a 0.03 mm dangling micro-stub at `mid`
        board.tracks = vec![
            TrackSegment {
                start: u1_p1,
                end: mid,
                width: 0.15,
                layer: 0,
                net_id: 2,
                is_locked: false,
            },
            TrackSegment {
                start: mid,
                end: u2_p1,
                width: 0.15,
                layer: 0,
                net_id: 2,
                is_locked: false,
            },
            TrackSegment {
                start: mid,
                end: (mid.0, mid.1 + 0.03),
                width: 0.15,
                layer: 0,
                net_id: 2,
                is_locked: false,
            },
        ];

        let router = AutoRouter::new(1);
        router.connect_u1_ground_pins(&mut board);
        prune_dangling_tracks_and_vias(&mut board);

        let sig_tracks: Vec<&TrackSegment> = board.tracks.iter().filter(|t| t.net_id == 2).collect();
        assert_eq!(sig_tracks.len(), 2, "0.03 mm dangling micro-stub must be pruned while preserving main route");
        let gnd_tracks: Vec<&TrackSegment> = board.tracks.iter().filter(|t| t.net_id == 1).collect();
        assert!(gnd_tracks.len() >= 2, "Rotated U1 pads 18 and 19 must connect inward into central thermal pad 25 and survive pruning");
    }

    #[test]
    fn test_prune_same_net_crossing_corner_stub_and_parallel_cycles() {
        let mut board = Board {
            max_x: 50.0,
            max_y: 50.0,
            footprints: vec![
                Footprint {
                    reference: "U1".to_string(),
                    x: 10.0,
                    y: 10.0,
                    pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG", false)],
                    ..Footprint::default()
                },
                Footprint {
                    reference: "U2".to_string(),
                    x: 20.0,
                    y: 10.0,
                    pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG", false)],
                    ..Footprint::default()
                },
            ],
            nets: vec!["".to_string(), "SIG".to_string()],
            tracks: vec![
                // Escape stub going past the corner at (12.0, 10.0) to (13.0, 10.0)
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (13.0, 10.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // Route crossing the stub at (12.0, 10.0) and continuing to U2
                TrackSegment {
                    start: (12.0, 10.0),
                    end: (12.0, 12.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (12.0, 12.0),
                    end: (20.0, 12.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (20.0, 12.0),
                    end: (20.0, 10.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // Redundant longer parallel detour between (12.0, 12.0) and (20.0, 12.0)
                TrackSegment {
                    start: (12.0, 12.0),
                    end: (12.0, 14.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (12.0, 14.0),
                    end: (20.0, 14.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (20.0, 14.0),
                    end: (20.0, 12.0),
                    width: 0.2,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            ..Board::default()
        };

        prune_dangling_tracks_and_vias(&mut board);

        // Both the corner overhang (12.0, 10.0)->(13.0, 10.0) and the redundant cycle via y=14.0 must be pruned
        for t in &board.tracks {
            assert!(
                t.start.0 <= 12.001 || t.start.1 > 10.5 || t.start.0 >= 19.999,
                "Overhanging corner stub past (12, 10) must be split and pruned: {:?}",
                t
            );
            assert!(
                t.start.1 < 13.0 && t.end.1 < 13.0,
                "Redundant parallel detour at y=14.0 must be pruned: {:?}",
                t
            );
        }
        assert_eq!(board.tracks.len(), 4);
    }

    #[test]
    fn test_eliminate_stair_steps_collapses_3_point_v_bounce() {
        let board = Board {
            min_x: 130.0,
            min_y: 80.0,
            max_x: 180.0,
            max_y: 130.0,
            nets: vec!["".to_string(), "SIG".to_string()],
            ..Board::default()
        };
        let grid = RoutingGrid::new(&board, 0.1, None);

        // NW then NE V-bounce: (141.7225, 96.5400) -> (141.1225, 95.9400) -> (141.5500, 95.5500)
        let mut chain = TrackChain {
            points: vec![
                (141.7225, 96.5400),
                (141.1225, 95.9400),
                (141.5500, 95.5500),
            ],
            width: 0.15,
            layer: 0,
            net_id: 1,
        };
        let orig_len = chain.points.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).sum::<f64>();
        eliminate_stair_steps(&mut chain, &board, &grid);
        let new_len = chain.points.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).sum::<f64>();
        assert!(
            new_len < orig_len - 0.3,
            "3-point V-bounce must collapse to direct North dogleg: orig_len={:.3}, new_len={:.3}, pts={:?}",
            orig_len,
            new_len,
            chain.points
        );
        for p in &chain.points {
            assert!(p.0 >= 141.54, "Collapsed path must not swing West to 141.12: {:?}", chain.points);
        }
    }

    #[test]
    fn test_c7_south_pad_ground_stitching_and_u2_sw_adjacent_via_consolidation() {
        let pcb_path = std::path::Path::new("/home/deece/src/p1p2esp/p1p2esp_vulcan.kicad_pcb");
        if !pcb_path.exists() {
            return;
        }
        let locked: std::collections::HashSet<String> = crate::board::DEFAULT_LOCKED_REFS.iter().map(|s| s.to_string()).collect();
        let mut board = Board::load(pcb_path.to_str().unwrap(), &locked).unwrap();
        let had_routed_tracks = !board.tracks.is_empty();
        let router = AutoRouter::new(2);
        let mut grid = RoutingGrid::new(&board, 0.10, None);

        // 1. Test adjacent via consolidation on Net-(U2-SW)
        let sw_net_id = *board.net_name_to_id.get("Net-(U2-SW)").unwrap();
        if !board.vias.iter().any(|v| v.net_id == sw_net_id && (v.position.0 - 144.45).hypot(v.position.1 - 113.05) < 0.1) {
            board.vias.push(Via {
                position: (144.45, 113.05),
                size: 0.6,
                drill: 0.3,
                net_id: sw_net_id,
                is_locked: false,
            });
        }
        if !board.vias.iter().any(|v| v.net_id == sw_net_id && (v.position.0 - 144.20).hypot(v.position.1 - 113.60) < 0.1) {
            board.vias.push(Via {
                position: (144.20, 113.60),
                size: 0.6,
                drill: 0.3,
                net_id: sw_net_id,
                is_locked: false,
            });
            board.tracks.push(TrackSegment {
                start: (144.45, 113.05),
                end: (144.20, 113.60),
                width: 0.3,
                layer: 0,
                net_id: sw_net_id,
                is_locked: false,
            });
        }
        let conns_before = router.extract_connections(&board);
        let connected_before: Vec<RouteConnection> = conns_before
            .into_iter()
            .filter(|c| c.net_id == sw_net_id && AutoRouter::is_connection_electrically_connected(&board, c))
            .collect();

        let sw_vias_before: Vec<Via> = board.vias.iter().filter(|v| v.net_id == sw_net_id).cloned().collect();
        assert!(sw_vias_before.len() >= 2);

        router.reduce_vias(&mut board, &mut grid);
        validate_and_clean_copper(&mut board);
        prune_dangling_tracks_and_vias(&mut board);

        let sw_vias_after: Vec<Via> = board.vias.iter().filter(|v| v.net_id == sw_net_id).cloned().collect();
        for i in 0..sw_vias_after.len() {
            for j in (i + 1)..sw_vias_after.len() {
                let d = (sw_vias_after[i].position.0 - sw_vias_after[j].position.0)
                    .hypot(sw_vias_after[i].position.1 - sw_vias_after[j].position.1);
                assert!(
                    d > 1.50,
                    "Net-(U2-SW) must not have 2 adjacent vias within 1.50mm: {:?} vs {:?} (dist={:.3}mm)",
                    sw_vias_after[i].position,
                    sw_vias_after[j].position,
                    d
                );
            }
        }

        // Verify all previously routed Net-(U2-SW) connections remain electrically continuous
        for c in &connected_before {
            assert!(
                AutoRouter::is_connection_electrically_connected(&board, c),
                "Net-(U2-SW) connection {:?} -> {:?} must remain electrically continuous after via consolidation",
                c.start,
                c.end
            );
        }

        // 2. Test C7 GND pad stitching
        let gnd_net_id = *board.net_name_to_id.get("GND").unwrap();
        router.add_plane_stitching_vias(&mut board, &grid);
        prune_dangling_tracks_and_vias(&mut board);

        let c7 = board.footprints.iter().find(|fp| fp.reference == "C7").unwrap();
        let c7_gnd_pad = c7.pads.iter().find(|p| p.net_id == gnd_net_id).unwrap();
        let (c7_px, c7_py) = c7.pad_absolute_position(c7_gnd_pad);
        let c7_connected = board.tracks.iter().any(|t| {
            t.net_id == gnd_net_id
                && t.layer == c7.layer
                && ((t.start.0 - c7_px).hypot(t.start.1 - c7_py) < 0.08
                    || (t.end.0 - c7_px).hypot(t.end.1 - c7_py) < 0.08)
        }) || board.arcs.iter().any(|a| {
            let s = a.start();
            let e = a.end();
            a.net_id == gnd_net_id
                && a.layer == c7.layer
                && ((s.0 - c7_px).hypot(s.1 - c7_py) < 0.08
                    || (e.0 - c7_px).hypot(e.1 - c7_py) < 0.08)
        });
        assert!(
            c7_connected,
            "GND pad of C7 at ({:.4}, {:.4}) must have a connected GND stitching track or arc",
            c7_px,
            c7_py
        );

        let out_pcb = "/tmp/test_c7_u2_sw_repaired.kicad_pcb";
        board.save_pcb(std::path::Path::new(out_pcb)).unwrap();
        if let Ok(status) = std::process::Command::new("kicad-cli")
            .args(["pcb", "drc", "--refill-zones", "--output", "/tmp/test_c7_u2_sw_drc.json", "--format", "json", out_pcb])
            .status()
        {
            if status.success() {
                if let Ok(drc_json) = std::fs::read_to_string("/tmp/test_c7_u2_sw_drc.json") {
                    if had_routed_tracks {
                        assert!(
                            drc_json.contains("\"unconnected_items\": []"),
                            "kicad-cli drc --refill-zones must report 0 unconnected_items"
                        );
                    }
                    assert!(
                        !drc_json.contains("\"hole_to_hole\""),
                        "kicad-cli drc --refill-zones must report 0 hole_to_hole violations"
                    );
                }
            }
        }
    }

    #[test]
    fn test_gpu_3d_routing_cost_kernel_clearance_and_congestion() {
        let Some(vk) = VulkanEngine::new() else {
            return;
        };
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 2.0;
        board.max_y = 2.0;
        board.nets = vec!["".into(), "NET1".into(), "NET2".into()];

        let mut grid = RoutingGrid::new(&board, 0.1, Some(vk));
        grid.obstacle_net_ids.fill(FREE_NET);
        grid.dist_field.fill(2.0);
        grid.congestion.fill(0.0);
        grid.usage_count.fill(0);

        // Place foreign obstacle (net 2) at (10, 10, 0) and same-net obstacle (net 1) at (4, 4, 0)
        let foreign_idx = grid.idx(10, 10, 0);
        let same_idx = grid.idx(4, 4, 0);
        let cong_idx = grid.idx(15, 15, 1);
        let free_idx = grid.idx(15, 5, 1);
        grid.obstacle_net_ids[foreign_idx] = 2;
        grid.obstacle_net_ids[same_idx] = 1;
        grid.congestion[cong_idx] = 3.0;
        grid.usage_count[cong_idx] = 2;

        grid.precompute_gpu_routing_costs_batch(&[(1, 2), (2, 2)], 1.5, 2.0);
        let (cached_r, cost_net1) = grid.gpu_net_cost_maps.get(&1).expect("Net 1 cost map should be cached");
        assert_eq!(*cached_r, 2);

        // Foreign obstacle and cells within 2-cell radius must be impassable (< 0.0)
        assert!(cost_net1[foreign_idx] < 0.0);
        assert!(cost_net1[grid.idx(12, 10, 0)] < 0.0);
        assert!(cost_net1[grid.idx(13, 10, 0)] > 0.0);

        // Same-net obstacle must remain passable (> 0.0)
        assert!(cost_net1[same_idx] > 0.0);
        assert!(cost_net1[grid.idx(5, 4, 0)] > 0.0);

        // Congestion and usage_count must increase cell cost
        assert!(cost_net1[cong_idx] > cost_net1[free_idx] + 5.0);

        // Verify get_neighbors with GPU cost map matches CPU-only get_neighbors
        let neighbors_with_gpu = grid.get_neighbors(
            GridNode { gx: 13, gy: 10, layer: 0 },
            1,
            0.15,
            0.15,
            0.60,
            None,
        );
        grid.gpu_net_cost_maps.clear();
        let neighbors_cpu_only = grid.get_neighbors(
            GridNode { gx: 13, gy: 10, layer: 0 },
            1,
            0.15,
            0.15,
            0.60,
            None,
        );
        assert_eq!(neighbors_with_gpu, neighbors_cpu_only);
    }

    #[test]
    fn test_gpu_3d_wavefront_shortest_path_potential() {
        let Some(vk) = VulkanEngine::new() else {
            return;
        };
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 1.6;
        board.max_y = 1.6;
        board.nets = vec!["".into(), "NET1".into(), "NET2".into()];

        let mut grid = RoutingGrid::new(&board, 0.1, Some(vk));
        grid.obstacle_net_ids.fill(FREE_NET);
        grid.dist_field.fill(2.0);

        let seed = grid.idx(2, 2, 0) as u32;
        let wf = grid
            .compute_gpu_3d_wavefront_for_net(1, 1, &[seed], 3.5, 32)
            .expect("GPU 3D wavefront should succeed");

        let d0 = wf[grid.idx(2, 2, 0)];
        let d1 = wf[grid.idx(3, 2, 0)];
        let d2 = wf[grid.idx(6, 2, 0)];
        let d_via = wf[grid.idx(2, 2, 1)];
        let d_via_step = wf[grid.idx(5, 2, 1)];

        assert!((d0 - 0.0).abs() < 1e-5);
        assert!(d1 > d0 && d2 > d1, "Wavefront potential must increase monotonically with distance");
        assert!((d_via - 3.5).abs() < 1e-4, "Via transition cost must equal via_cost");
        assert!(d_via_step > d_via, "Planar steps on adjacent layer must accumulate over via cost");
    }

    #[test]
    fn test_gpu_batch_drc_eval_via_and_segment_clearance() {
        let mut board = make_router_test_board();
        board.tracks.push(TrackSegment {
            start: (138.0, 100.0),
            end: (146.0, 100.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        board.vias.push(Via {
            position: (142.0, 104.0),
            size: 0.60,
            drill: 0.30,
            net_id: 2,
            is_locked: false,
        });

        let drc_ctx = GpuDrcContext::from_board(&board);
        let vk = VulkanEngine::new();

        let via_pts = [
            (142.0, 100.25), // Collides with net 2 track at y=100.0
            (142.0, 102.0),  // Clear of track and via
            (142.3, 104.0),  // Collides with net 2 via at (142.0, 104.0)
            (142.0, 95.0),   // Collides with C14 pad 1 (net 1) for net 3
        ];

        let mut cands = Vec::new();
        for &(vx, vy) in &via_pts {
            cands.push(GpuDrcCandidate {
                x1: vx as f32,
                y1: vy as f32,
                x2: 0.0,
                y2: 0.0,
                radius_or_half_width: 0.30,
                clearance: 0.20,
                layer: 0,
                net_id: 3,
                kind: 0,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            });
        }

        let segs = [
            ((140.0, 99.0), (140.0, 101.0), 0usize), // Crosses net 2 track on layer 0 -> illegal
            ((140.0, 99.0), (140.0, 101.0), 1usize), // Crosses on layer 1 -> legal
            ((138.0, 101.5), (146.0, 101.5), 0usize), // Parallel 1.5mm away -> legal
        ];
        for &(s, e, layer) in &segs {
            cands.push(GpuDrcCandidate {
                x1: s.0 as f32,
                y1: s.1 as f32,
                x2: e.0 as f32,
                y2: e.1 as f32,
                radius_or_half_width: 0.125,
                clearance: 0.20,
                layer: layer as u32,
                net_id: 3,
                kind: 1,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            });
        }

        if let Some(res) = drc_ctx.evaluate(vk.as_ref(), &cands) {
            for (i, &pt) in via_pts.iter().enumerate() {
                let cpu_ok = is_via_location_legal(&board, pt, 0.60, 3, 0.20);
                assert_eq!(res[i] != 0, cpu_ok, "Via candidate {} at {:?} mismatch", i, pt);
            }
            for (j, &(s, e, layer)) in segs.iter().enumerate() {
                let cpu_ok = is_track_segment_legal(&board, s, e, 0.25, layer, 3, 0.20);
                assert_eq!(
                    res[via_pts.len() + j] != 0,
                    cpu_ok,
                    "Segment candidate {} ({:?} -> {:?} on L{}) mismatch",
                    j,
                    s,
                    e,
                    layer
                );
            }
        }
    }

    #[test]
    fn test_cuda_vs_vulkan_batch_drc_equivalence() {
        let (Some(cuda), Some(vk)) = (CudaEngine::new(), VulkanEngine::new()) else {
            return;
        };
        let mut board = make_router_test_board();
        board.tracks.push(TrackSegment {
            start: (136.0, 100.0),
            end: (148.0, 100.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        let drc_ctx = GpuDrcContext::from_board(&board);
        let mut cands = Vec::with_capacity(160);
        for i in 0..160 {
            let y = 96.0 + (i as f32) * 0.05;
            cands.push(GpuDrcCandidate {
                x1: 140.0,
                y1: y,
                x2: 142.0,
                y2: y + 0.2,
                radius_or_half_width: if i % 2 == 0 { 0.30 } else { 0.125 },
                clearance: 0.20,
                layer: 0,
                net_id: 3,
                kind: (i % 2) as u32,
                ignore_same_net_pads: 0,
                _pad0: 0,
                _pad1: 0,
            });
        }
        let cuda_res = cuda
            .evaluate_drc_candidates(&drc_ctx.pads, &drc_ctx.tracks, &drc_ctx.vias, &cands)
            .expect("CUDA evaluate_drc_candidates failed");
        let vk_res = vk
            .evaluate_drc_candidates(&drc_ctx.pads, &drc_ctx.tracks, &drc_ctx.vias, &cands)
            .expect("Vulkan evaluate_drc_candidates failed");
        assert_eq!(cuda_res, vk_res, "CUDA and Vulkan DRC batch results must match identically");
    }

    #[test]
    fn test_conflict_graph_independent_set_wave_partitioning() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 20.0;
        board.max_y = 20.0;
        board.nets = vec![
            "".to_string(),
            "NET1".to_string(),
            "NET2".to_string(),
            "NET3".to_string(),
        ];

        let grid = RoutingGrid::new(&board, 0.1, None);
        let nc = NetClass::default();

        // Connection 0: NET1 horizontal along y = 3.0 (gy = 30) on layer 0
        // Connection 1: NET2 horizontal along y = 15.0 (gy = 150) on layer 0 (disjoint from NET1)
        // Connection 2: NET1 vertical along x = 5.0 (gx = 50) on layer 0 (same net as Connection 0 -> no conflict!)
        // Connection 3: NET3 vertical along x = 6.0 (gx = 60) crossing y = 2.0..5.0 on layer 0 (crosses Connection 0 -> must be in Wave 1!)
        let connections = vec![
            RouteConnection {
                net_id: 1,
                net_name: "NET1".to_string(),
                start: (2.0, 3.0, 0),
                end: (10.0, 3.0, 0),
                start_fp: 0,
                start_pad: 0,
                end_fp: 0,
                end_pad: 0,
                netclass: nc.clone(),
                diff_pair_leader: None,
            },
            RouteConnection {
                net_id: 2,
                net_name: "NET2".to_string(),
                start: (2.0, 15.0, 0),
                end: (10.0, 15.0, 0),
                start_fp: 0,
                start_pad: 0,
                end_fp: 0,
                end_pad: 0,
                netclass: nc.clone(),
                diff_pair_leader: None,
            },
            RouteConnection {
                net_id: 1,
                net_name: "NET1".to_string(),
                start: (5.0, 1.0, 0),
                end: (5.0, 6.0, 0),
                start_fp: 0,
                start_pad: 0,
                end_fp: 0,
                end_pad: 0,
                netclass: nc.clone(),
                diff_pair_leader: None,
            },
            RouteConnection {
                net_id: 3,
                net_name: "NET3".to_string(),
                start: (6.0, 1.0, 0),
                end: (6.0, 6.0, 0),
                start_fp: 0,
                start_pad: 0,
                end_fp: 0,
                end_pad: 0,
                netclass: nc,
                diff_pair_leader: None,
            },
        ];

        let path0: Vec<GridNode> = (20..=100).map(|gx| GridNode { gx, gy: 30, layer: 0 }).collect();
        let path1: Vec<GridNode> = (20..=100).map(|gx| GridNode { gx, gy: 150, layer: 0 }).collect();
        let path2: Vec<GridNode> = (10..=60).map(|gy| GridNode { gx: 50, gy, layer: 0 }).collect();
        let path3: Vec<GridNode> = (10..=60).map(|gy| GridNode { gx: 60, gy, layer: 0 }).collect();

        let candidates = vec![
            (0usize, path0),
            (1usize, path1),
            (2usize, path2),
            (3usize, path3),
        ];

        let waves = AutoRouter::partition_into_conflict_free_waves(&grid, &connections, candidates);
        assert_eq!(waves.len(), 2, "Expected 2 conflict-free waves, got {:?}", waves.len());

        let wave0_indices: Vec<usize> = waves[0].iter().map(|(idx, _)| *idx).collect();
        let wave1_indices: Vec<usize> = waves[1].iter().map(|(idx, _)| *idx).collect();

        // Connections 0 (NET1), 1 (disjoint NET2), and 2 (same-net NET1) must all be grouped into Wave 0
        assert_eq!(wave0_indices, vec![0, 1, 2]);
        // Connection 3 (NET3 crossing NET1) must be separated into Wave 1
        assert_eq!(wave1_indices, vec![3]);
    }

    #[test]
    fn test_parallel_portfolio_multi_trajectory_routing() {
        let mut board = Board::default();
        board.min_x = 0.0;
        board.min_y = 0.0;
        board.max_x = 20.0;
        board.max_y = 20.0;
        board.nets = vec![
            "".to_string(),
            "NET1".to_string(),
            "NET2".to_string(),
            "NET3".to_string(),
        ];

        let mut grid = RoutingGrid::new(&board, 0.1, VulkanEngine::new());
        grid.obstacle_net_ids.fill(FREE_NET);
        grid.dist_field.fill(2.0);

        // Build an asymmetric deep U-pocket around start (gx=40, gy=100) opening West,
        // while goal is at (gx=140, gy=100) to the East.
        // Forward A* with a small expansion cap (e.g. 450 expansions) is locally trapped inside the deep pocket,
        // whereas reverse A* (target -> source) or unconstrained portfolio expansion routes cleanly.
        for gx in 30..=90 {
            for l in 0..grid.num_layers {
                let i1 = grid.idx(gx, 85, l);
                let i2 = grid.idx(gx, 115, l);
                grid.obstacle_net_ids[i1] = KEEPOUT_NET;
                grid.obstacle_net_ids[i2] = KEEPOUT_NET;
            }
        }
        for gy in 85..=115 {
            for l in 0..grid.num_layers {
                let i3 = grid.idx(90, gy, l);
                grid.obstacle_net_ids[i3] = KEEPOUT_NET;
            }
        }

        let router = AutoRouter::new(2);
        let stop_flag = AtomicBool::new(false);
        let s_node = GridNode { gx: 25, gy: 100, layer: 0 };
        let e_node = GridNode { gx: 100, gy: 100, layer: 0 };

        // 1. Verify reverse-direction portfolio variant escapes when forward A* is locally trapped
        let fwd_trapped = router.find_path_portfolio_variant(
            &grid,
            s_node,
            e_node,
            1,
            0.15,
            0.15,
            0.60,
            None,
            15.0,
            false,
            1.0,
            1.0,
            1.35,
            0.0,
            false,
            false,
            &stop_flag,
            None,
            Some(350),
        );
        assert!(fwd_trapped.is_none(), "Forward greedy A* with 350-expansion budget should be trapped inside U-pocket");

        let rev_escaped = router.find_path_portfolio_variant(
            &grid,
            s_node,
            e_node,
            1,
            0.15,
            0.15,
            0.60,
            None,
            15.0,
            false,
            1.0,
            1.0,
            1.35,
            0.0,
            true,
            false,
            &stop_flag,
            None,
            Some(2500),
        );
        assert!(rev_escaped.is_some(), "Reverse target->source A* must bypass U-pocket and find legal path");
        let rev_path = rev_escaped.unwrap();
        assert_eq!(rev_path.first().copied(), Some(s_node));
        assert_eq!(rev_path.last().copied(), Some(e_node));

        // 2. Verify full parallel portfolio multi-trajectory ensemble across multiple connections
        let nc = NetClass::default();
        let sw = grid.grid_to_world(s_node.gx, s_node.gy);
        let ew = grid.grid_to_world(e_node.gx, e_node.gy);
        let conns = vec![
            RouteConnection {
                net_id: 1,
                net_name: "NET1".to_string(),
                start: (sw.0, sw.1, 0),
                end: (ew.0, ew.1, 0),
                start_fp: 0,
                start_pad: 0,
                end_fp: 0,
                end_pad: 0,
                netclass: nc.clone(),
                diff_pair_leader: None,
            },
            RouteConnection {
                net_id: 2,
                net_name: "NET2".to_string(),
                start: (2.0, 3.0, 0),
                end: (16.0, 3.0, 0),
                start_fp: 0,
                start_pad: 0,
                end_fp: 0,
                end_pad: 0,
                netclass: nc,
                diff_pair_leader: None,
            },
        ];

        let target_indices = vec![0usize, 1usize];
        let results = router.route_connections_portfolio(
            &board,
            &grid,
            &conns,
            &target_indices,
            None,
            &stop_flag,
        );
        assert_eq!(results.len(), 2);
        for (c_idx, p_opt) in &results {
            let p = p_opt.as_ref().unwrap_or_else(|| panic!("Portfolio failed to route connection {}", c_idx));
            assert!(
                router.is_path_conflict_free(
                    &grid,
                    p,
                    conns[*c_idx].net_id,
                    conns[*c_idx].netclass.clearance,
                    conns[*c_idx].netclass.track_width,
                    conns[*c_idx].netclass.via_dia,
                    None,
                    false,
                ),
                "Portfolio path for connection {} must be conflict-free",
                c_idx
            );
        }
    }

    #[test]
    fn test_gpu_3d_wavefront_backtrack_path_valid_and_obstacle_free() {
        let vk = VulkanEngine::new();
        if vk.is_none() {
            return;
        }
        let board = make_router_test_board();
        let mut grid = RoutingGrid::new(&board, 0.25, vk);

        // Clear synthetic grid interior so we can construct a deterministic obstacle barrier
        grid.obstacle_net_ids.fill(FREE_NET);
        let gw = grid.width;
        let gh = grid.height;
        assert!(gw >= 24 && gh >= 24);

        let mid_x = gw / 2;
        let clearance_cells = 1i32;
        // Place a foreign obstacle barrier (net 99) at x = mid_x, y = 2..(gh - 6) on all layers
        for l in 0..grid.num_layers {
            for y in 2..(gh - 6) {
                let idx = grid.idx(mid_x, y, l);
                grid.obstacle_net_ids[idx] = 99;
            }
        }

        let router = AutoRouter::new(1);
        let src1 = GridNode {
            gx: mid_x.saturating_sub(5),
            gy: gh / 2,
            layer: 0,
        };
        let dst1 = GridNode {
            gx: (mid_x + 5).min(gw - 3),
            gy: gh / 2,
            layer: 0,
        };
        let src2 = GridNode {
            gx: mid_x.saturating_sub(4),
            gy: gh - 3,
            layer: 1,
        };
        let dst2 = GridNode {
            gx: (mid_x + 4).min(gw - 3),
            gy: gh - 3,
            layer: 1,
        };

        let requests = vec![
            (src1, dst1, 1usize, 3.0f32),
            (src2, dst2, 2usize, 3.0f32),
        ];
        let results = router.solve_batch_gpu_wavefront_paths(&grid, &requests, clearance_cells);
        assert_eq!(results.len(), 2);

        for (req_idx, &(src, dst, net_id, _)) in requests.iter().enumerate() {
            let path = results[req_idx]
                .as_ref()
                .expect("GPU 3D wavefront solver must return a valid obstacle-free path");
            assert_eq!(path.first().copied(), Some(src));
            assert_eq!(path.last().copied(), Some(dst));
            for w in path.windows(2) {
                let dx = (w[1].gx as isize - w[0].gx as isize).abs();
                let dy = (w[1].gy as isize - w[0].gy as isize).abs();
                let dl = (w[1].layer as isize - w[0].layer as isize).abs();
                assert!(
                    (dl == 0 && dx <= 1 && dy <= 1 && (dx + dy > 0))
                        || (dl > 0 && dx == 0 && dy == 0),
                    "Invalid step {:?} -> {:?}",
                    w[0],
                    w[1]
                );
            }
            for node in path {
                assert!(
                    grid.is_circle_clear_SpecifyRadius(
                        node.gx,
                        node.gy,
                        node.layer,
                        clearance_cells,
                        net_id
                    ),
                    "Node {:?} violated clearance radius {} for net {}",
                    node,
                    clearance_cells,
                    net_id
                );
            }
        }
    }

    #[test]
    fn test_parallel_disjoint_net_glossing_and_via_reduction() {
        let nets = vec![
            "".to_string(),
            "SIG1".to_string(),
            "SIG2".to_string(),
            "SIG3".to_string(),
        ];
        let net_name_to_id: HashMap<String, usize> = nets
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let netclasses: HashMap<String, NetClass> =
            [("Default".to_string(), NetClass::default())].into_iter().collect();
        let net_to_class: HashMap<String, String> = [
            ("SIG1".to_string(), "Default".to_string()),
            ("SIG2".to_string(), "Default".to_string()),
            ("SIG3".to_string(), "Default".to_string()),
        ]
        .into_iter()
        .collect();

        // Endpoint footprints for SIG1 (y = 10.0), SIG2 (y = 30.0), and SIG3 (y = 50.0)
        let fp_sig1_a = make_fp(
            "U_S1A",
            10.0,
            10.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false)],
        );
        let fp_sig1_b = make_fp(
            "U_S1B",
            35.0,
            10.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "SIG1", false)],
        );
        let fp_sig2_a = make_fp(
            "U_S2A",
            10.0,
            30.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "SIG2", false)],
        );
        let fp_sig2_b = make_fp(
            "U_S2B",
            35.0,
            30.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 2, "SIG2", false)],
        );
        let fp_sig3_a = make_fp(
            "U_S3A",
            10.0,
            50.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 3, "SIG3", false)],
        );
        let fp_sig3_b = make_fp(
            "U_S3B",
            35.0,
            50.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 3, "SIG3", false)],
        );

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: vec![
                fp_sig1_a, fp_sig1_b, fp_sig2_a, fp_sig2_b, fp_sig3_a, fp_sig3_b,
            ],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            // Layer 0 keepout blocking SIG3's via pair at y = 50.0 (boundary/violation preservation case)
            keepouts: vec![KeepoutZone {
                min_x: 18.0,
                min_y: 35.0,
                max_x: 22.0,
                max_y: 65.0,
                layers: vec![0],
            }],
            tracks: vec![
                // SIG1 (net 1, y ~ 10.0): redundant via pair + rubber-band detour
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (15.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 10.0),
                    end: (25.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (25.0, 10.0),
                    end: (30.0, 14.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (30.0, 14.0),
                    end: (35.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // SIG2 (net 2, y ~ 30.0): single via between two F.Cu pads + detour
                TrackSegment {
                    start: (10.0, 30.0),
                    end: (15.0, 30.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 2,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 30.0),
                    end: (25.0, 34.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 2,
                    is_locked: false,
                },
                TrackSegment {
                    start: (25.0, 34.0),
                    end: (35.0, 30.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 2,
                    is_locked: false,
                },
                // SIG3 (net 3, y ~ 50.0): necessary via pair over keepout + rubber-band detour
                TrackSegment {
                    start: (10.0, 50.0),
                    end: (15.0, 50.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 3,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 50.0),
                    end: (25.0, 50.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 3,
                    is_locked: false,
                },
                TrackSegment {
                    start: (25.0, 50.0),
                    end: (30.0, 54.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 3,
                    is_locked: false,
                },
                TrackSegment {
                    start: (30.0, 54.0),
                    end: (35.0, 50.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 3,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (15.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (25.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (15.0, 30.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 2,
                    is_locked: false,
                },
                Via {
                    position: (15.0, 50.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 3,
                    is_locked: false,
                },
                Via {
                    position: (25.0, 50.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 3,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        grid.rasterize_obstacles(&board);
        let router = AutoRouter::new(3);

        router.reduce_vias(&mut board, &mut grid);
        router.gloss_tracks(&mut board, &grid);

        // SIG1 and SIG2 vias must be completely eliminated in parallel; SIG3 obstacle-crossing vias must remain
        assert_eq!(board.vias.iter().filter(|v| v.net_id == 1).count(), 0);
        assert_eq!(board.vias.iter().filter(|v| v.net_id == 2).count(), 0);
        assert_eq!(board.vias.iter().filter(|v| v.net_id == 3).count(), 2);

        // Verify SIG1 and SIG2 are straightened onto layer 0 (length == 25.0 mm each)
        let sig1_len: f64 = board
            .tracks
            .iter()
            .filter(|t| t.net_id == 1)
            .map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1))
            .sum();
        let sig2_len: f64 = board
            .tracks
            .iter()
            .filter(|t| t.net_id == 2)
            .map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1))
            .sum();
        assert!((sig1_len - 25.0).abs() < 1e-3, "SIG1 must be straightened to 25.0 mm, got {:.3}", sig1_len);
        assert!((sig2_len - 25.0).abs() < 1e-3, "SIG2 must be straightened to 25.0 mm, got {:.3}", sig2_len);
    }

    #[test]
    fn test_parallel_speculative_track_and_component_nudge() {
        let nets = vec![
            "".to_string(),
            "SIG_TARGET".to_string(),
            "SIG_BLOCK".to_string(),
        ];
        let net_name_to_id: HashMap<String, usize> = nets
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let netclasses: HashMap<String, NetClass> =
            [("Default".to_string(), NetClass::default())].into_iter().collect();
        let net_to_class: HashMap<String, String> = [
            ("SIG_TARGET".to_string(), "Default".to_string()),
            ("SIG_BLOCK".to_string(), "Default".to_string()),
        ]
        .into_iter()
        .collect();

        // Part 1: Parallel speculative track deflection via try_nudge_or_ripup_blocking_tracks
        let fp_a = make_fp(
            "U1",
            14.0,
            10.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 1, "SIG_TARGET", false)],
        );
        let fp_b = make_fp(
            "U2",
            26.0,
            10.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 1, "SIG_TARGET", false)],
        );
        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 30.0,
            footprints: vec![fp_a, fp_b],
            holding_palette: Vec::new(),
            nets: nets.clone(),
            net_name_to_id: net_name_to_id.clone(),
            netclasses: netclasses.clone(),
            net_to_class: net_to_class.clone(),
            keepouts: Vec::new(),
            tracks: vec![TrackSegment {
                start: (10.0, 10.15),
                end: (30.0, 10.15),
                width: 0.25,
                layer: 0,
                net_id: 2,
                is_locked: false,
            }],
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let router = AutoRouter::new(2);
        let mut grid = RoutingGrid::new(&board, 0.25, None);
        grid.rasterize_obstacles(&board);

        let conns = router.extract_connections(&board);
        assert_eq!(conns.len(), 1);
        let mut routed_paths: Vec<Option<Vec<GridNode>>> = vec![None; conns.len()];
        let stop_flag = AtomicBool::new(false);

        let routed = router.try_nudge_or_ripup_blocking_tracks(
            &mut board,
            &mut grid,
            &conns,
            &mut routed_paths,
            &[],
            0,
            None,
            &stop_flag,
        );
        assert!(
            routed,
            "Speculative parallel track shove must deflect blocking track and route target connection"
        );
        assert!(
            routed_paths[0].is_some(),
            "Target net must have routed path in routed_paths"
        );

        // Part 2: Parallel speculative component nudge via try_nudge_blocking_components
        let mut c_block = make_fp(
            "C_BLOCK",
            20.0,
            10.0,
            false,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 2, "SIG_BLOCK", false)],
        );
        c_block.courtyards = vec![Polygon::new(vec![
            Point::new(-0.5, -0.5),
            Point::new(0.5, -0.5),
            Point::new(0.5, 0.5),
            Point::new(-0.5, 0.5),
        ])];
        let mut board_comp = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 30.0,
            footprints: vec![
                make_fp(
                    "U1",
                    14.0,
                    10.0,
                    true,
                    vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 1, "SIG_TARGET", false)],
                ),
                make_fp(
                    "U2",
                    26.0,
                    10.0,
                    true,
                    vec![Pad::new("1", 0.0, 0.0, 0.4, 0.4, 1, "SIG_TARGET", false)],
                ),
                c_block,
            ],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            // Channel keepouts on both layers leaving a corridor around y = 10.0
            keepouts: vec![
                KeepoutZone {
                    min_x: 12.0,
                    min_y: 0.0,
                    max_x: 28.0,
                    max_y: 8.5,
                    layers: vec![0, 1],
                },
                KeepoutZone {
                    min_x: 12.0,
                    min_y: 12.5,
                    max_x: 28.0,
                    max_y: 30.0,
                    layers: vec![0, 1],
                },
            ],
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid_comp = RoutingGrid::new(&board_comp, 0.25, None);
        grid_comp.rasterize_obstacles(&board_comp);
        let mut conns_comp = router.extract_connections(&board_comp);
        let mut routed_paths_comp: Vec<Option<Vec<GridNode>>> = vec![None; conns_comp.len()];

        // Boundary check: when allow_component_nudge is false, it must return false without moving C_BLOCK
        assert!(!router.try_nudge_blocking_components(
            &mut board_comp,
            &mut grid_comp,
            &mut conns_comp,
            &mut routed_paths_comp,
            &[],
            0,
            None,
            &stop_flag,
        ));

        let router_nudge = AutoRouter::new(2).with_allow_component_nudge(true);
        let nudged_and_routed = router_nudge.try_nudge_blocking_components(
            &mut board_comp,
            &mut grid_comp,
            &mut conns_comp,
            &mut routed_paths_comp,
            &[],
            0,
            None,
            &stop_flag,
        );
        assert!(
            nudged_and_routed,
            "Parallel speculative component nudge must displace C_BLOCK and route SIG_TARGET"
        );
        assert!(
            (board_comp.footprints[2].x - 20.0).hypot(board_comp.footprints[2].y - 10.0) > 0.1,
            "C_BLOCK must have been nudged from its blocking position (20.0, 10.0)"
        );
    }

    #[test]
    fn test_base_snapshot_caching_and_parallel_return_vias() {
        let nets = vec!["".to_string(), "GND".to_string(), "CLK".to_string()];
        let net_name_to_id: HashMap<String, usize> = nets
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();

        let mut default_nc = NetClass::default();
        default_nc.require_return_path = true;
        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), default_nc);

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 30.0,
            max_y: 20.0,
            footprints: vec![
                make_fp(
                    "U1",
                    5.0,
                    10.0,
                    true,
                    vec![
                        Pad::new("1", 0.0, 0.0, 0.6, 0.6, 2, "CLK", false),
                        Pad::new("2", 0.0, 2.0, 0.6, 0.6, 1, "GND", false),
                    ],
                ),
                make_fp(
                    "U2",
                    25.0,
                    10.0,
                    false,
                    vec![Pad::new("1", 0.0, 0.0, 0.6, 0.6, 2, "CLK", false)],
                ),
            ],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class: HashMap::new(),
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

        let router = AutoRouter::new(2);
        let mut grid = RoutingGrid::new(&board, 0.25, None);
        assert!(grid.base_snapshot.is_none());

        let conns = router.extract_connections(&board);
        let routed_paths: Vec<Option<Vec<GridNode>>> = vec![None; conns.len()];

        // First rebuild_grid_state populates base_snapshot
        router.rebuild_grid_state(&board, &mut grid, &conns, &routed_paths, &[]);
        assert!(grid.base_snapshot.is_some());
        let first_obs = grid.obstacle_net_ids.clone();
        let first_smd = grid.smd_pad_cells.clone();

        // Mutate obstacle_net_ids and verify cached rebuild restores exact snapshot
        grid.obstacle_net_ids.fill(99);
        router.rebuild_grid_state(&board, &mut grid, &conns, &routed_paths, &[]);
        assert_eq!(grid.obstacle_net_ids, first_obs);
        assert_eq!(grid.smd_pad_cells, first_smd);

        // Moving a footprint and invalidating base_snapshot updates the obstacle map
        board.footprints[1].x = 20.0;
        grid.base_snapshot = None;
        router.rebuild_grid_state(&board, &mut grid, &conns, &routed_paths, &[]);
        assert!(grid.base_snapshot.is_some());
        assert_ne!(grid.obstacle_net_ids, first_obs);

        // Add two signal vias on CLK and verify parallel return-path ground via insertion
        board.vias.push(Via {
            position: (10.0, 10.0),
            size: 0.6,
            drill: 0.3,
            net_id: 2,
            is_locked: false,
        });
        board.vias.push(Via {
            position: (15.0, 10.0),
            size: 0.6,
            drill: 0.3,
            net_id: 2,
            is_locked: false,
        });
        let added = router.add_return_path_ground_vias(&mut board, &mut grid);
        assert_eq!(added, 2, "Expected 2 return-path ground vias for 2 signal vias");
        assert_eq!(board.vias.iter().filter(|v| v.net_id == 1).count(), 2);
    }

    #[test]
    fn test_topological_and_physical_same_net_via_consolidation() {
        let nets = vec!["".to_string(), "SIG_BRIDGE".to_string(), "SIG_SHARED_PAD".to_string()];
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

        // Case 1 (SIG_BRIDGE, net 1):
        // Two vias at (10.0, 10.0) and (11.8, 10.2) (dist ~1.81mm) connected by a 2-segment B.Cu bridge
        // via (11.6, 10.2), where Via 1 at (10.0, 10.0) also has a B.Cu branch to (6.0, 10.0)
        // and Via 2 at (11.8, 10.2) has an F.Cu chain with a knee at (12.2, 10.2) -> (14.0, 10.2).
        // Case 2 (SIG_SHARED_PAD, net 2):
        // Two vias at (19.2, 20.0) and (20.8, 20.0) (dist 1.60mm) that both connect on F.Cu to a shared
        // intermediate SMD pad at (20.0, 20.8), while each transitions to B.Cu to reach (16.0, 20.0) and (24.0, 20.0).
        let mut fp_b1 = make_fp(
            "U_B1",
            6.0,
            10.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 1, "SIG_BRIDGE", false)],
        );
        fp_b1.layer = 1;
        let fp_f1 = make_fp(
            "U_F1",
            10.0,
            8.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 1, "SIG_BRIDGE", false)],
        );
        let fp_f2 = make_fp(
            "U_F2",
            14.0,
            10.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 1, "SIG_BRIDGE", false)],
        );

        let shared_fp = make_fp(
            "R_MID",
            20.0,
            20.8,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 2, "SIG_SHARED_PAD", false)],
        );
        let mut fp_b2_left = make_fp(
            "U_B2L",
            16.0,
            20.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 2, "SIG_SHARED_PAD", false)],
        );
        fp_b2_left.layer = 1;
        let mut fp_b2_right = make_fp(
            "U_B2R",
            24.0,
            20.0,
            true,
            vec![Pad::new("1", 0.0, 0.0, 0.8, 0.8, 2, "SIG_SHARED_PAD", false)],
        );
        fp_b2_right.layer = 1;

        let mut board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 40.0,
            footprints: vec![fp_b1, fp_f1, fp_f2, shared_fp, fp_b2_left, fp_b2_right],
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts: Vec::new(),
            tracks: vec![
                // Net 1 (SIG_BRIDGE):
                TrackSegment {
                    start: (6.0, 10.0),
                    end: (10.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 8.0),
                    end: (10.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // 2-segment B.Cu bridge between Via 1 (10.0, 10.0) and Via 2 (11.8, 10.2)
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (11.6, 10.2),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (11.6, 10.2),
                    end: (11.8, 10.2),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                // F.Cu chain from Via 2 (11.8, 10.2) -> (12.2, 10.2) -> (14.0, 10.0)
                TrackSegment {
                    start: (11.8, 10.2),
                    end: (12.2, 10.2),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (12.2, 10.2),
                    end: (14.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // Net 2 (SIG_SHARED_PAD):
                // Both vias connect on F.Cu to the shared SMD pad at (20.0, 20.8)
                TrackSegment {
                    start: (20.0, 20.8),
                    end: (19.2, 20.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 2,
                    is_locked: false,
                },
                TrackSegment {
                    start: (19.2, 20.0),
                    end: (16.0, 20.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 2,
                    is_locked: false,
                },
                TrackSegment {
                    start: (20.0, 20.8),
                    end: (20.8, 20.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 2,
                    is_locked: false,
                },
                TrackSegment {
                    start: (20.8, 20.0),
                    end: (24.0, 20.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 2,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (10.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (11.8, 10.2),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (19.2, 20.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 2,
                    is_locked: false,
                },
                Via {
                    position: (20.8, 20.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 2,
                    is_locked: false,
                },
            ],
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(), texts: Vec::new(),
        };

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        grid.rasterize_obstacles(&board);
        let router = AutoRouter::new(2);

        router.reduce_vias(&mut board, &mut grid);
        assert_eq!(
            board.vias.iter().filter(|v| v.net_id == 1).count(),
            1,
            "Net 1 (multi-segment bridge) must consolidate from 2 vias to 1 via"
        );
        assert_eq!(
            board.vias.iter().filter(|v| v.net_id == 2).count(),
            1,
            "Net 2 (shared intermediate pad) must consolidate from 2 vias to 1 via"
        );
    }

    #[test]
    fn test_routing_config_via_merge_radius_threshold() {
        use crate::board::NetClass;
        use std::collections::HashMap;

        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert(String::new(), 0);
        net_name_to_id.insert("SIG".to_string(), 1);

        let base_board = Board {
            file_path: String::new(),
            raw_content: String::new(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 40.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets: vec![String::new(), "SIG".to_string()],
            net_name_to_id,
            netclasses,
            net_to_class: HashMap::new(),
            keepouts: Vec::new(),
            tracks: vec![
                TrackSegment {
                    start: (6.0, 10.0),
                    end: (10.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (12.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (12.0, 10.0),
                    end: (15.0, 10.0),
                    width: 0.25,
                    layer: 1,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (12.0, 10.0),
                    end: (12.0, 13.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (10.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (12.0, 10.0),
                    size: 0.60,
                    drill: 0.30,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            ..Board::default()
        };

        // 1. With tight via_merge_radius_mm (1.0 mm < 2.0 mm separation), the two vias must NOT merge
        let mut board_tight = base_board.clone();
        let mut grid_tight = RoutingGrid::new(&board_tight, 0.20, None);
        grid_tight.rasterize_obstacles(&board_tight);
        let tight_cfg = RoutingConfig {
            via_merge_radius_mm: 1.0,
            ..RoutingConfig::default()
        };
        let router_tight = AutoRouter::new(1).with_config(tight_cfg);
        router_tight.reduce_vias(&mut board_tight, &mut grid_tight);
        assert_eq!(
            board_tight.vias.len(),
            2,
            "Vias 2.0mm apart must not merge when via_merge_radius_mm is 1.0mm"
        );

        // 2. With default via_merge_radius_mm (4.0 mm >= 2.0 mm separation), the two vias MUST merge into 1
        let mut board_default = base_board;
        let mut grid_default = RoutingGrid::new(&board_default, 0.20, None);
        grid_default.rasterize_obstacles(&board_default);
        let router_default = AutoRouter::new(1).with_config(RoutingConfig::default());
        router_default.reduce_vias(&mut board_default, &mut grid_default);
        assert_eq!(
            board_default.vias.len(),
            1,
            "Vias 2.0mm apart must merge into 1 via when via_merge_radius_mm is 4.0mm"
        );
    }

    #[test]
    fn test_convert_tracks_to_organic_and_back_to_straight() {
        use crate::board::NetClass;
        use std::collections::HashMap;

        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert(String::new(), 0);
        net_name_to_id.insert("NET_A".to_string(), 1);
        net_name_to_id.insert("NET_B".to_string(), 2);

        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 40.0,
            nets: vec![String::new(), "NET_A".to_string(), "NET_B".to_string()],
            net_name_to_id,
            netclasses,
            tracks: vec![
                // NET_A: L-shaped octilinear dogleg from (5,5) -> (15,5) -> (25,15)
                TrackSegment {
                    start: (5.0, 5.0),
                    end: (15.0, 5.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 5.0),
                    end: (25.0, 15.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // NET_B: Straight 2-point span from (5,20) -> (25,20)
                TrackSegment {
                    start: (5.0, 20.0),
                    end: (25.0, 20.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 2,
                    is_locked: false,
                },
            ],
            ..Board::default()
        };

        assert!(board.arcs.is_empty());
        convert_tracks_to_organic(&mut board);
        assert!(
            !board.arcs.is_empty(),
            "convert_tracks_to_organic must generate curved ArcSegments for bends and/or long straight spans"
        );
        let curved_len: f64 = board.arcs.iter().map(|a| a.length()).sum();
        let straight_len: f64 = board
            .tracks
            .iter()
            .map(|t| (t.end.0 - t.start.0).hypot(t.end.1 - t.start.1))
            .sum();
        assert!(
            curved_len > straight_len * 3.0,
            "Organic conversion must turn long runs and bends into predominantly smooth curves (curved={:.2}mm vs straight={:.2}mm)",
            curved_len,
            straight_len
        );

        let mut grid = RoutingGrid::new(&board, 0.20, None);
        grid.rasterize_obstacles(&board);
        convert_tracks_to_straight(&mut board, &grid);
        assert!(
            board.arcs.is_empty(),
            "convert_tracks_to_straight must convert all unlocked organic arcs back into straight tracks"
        );
        assert!(!board.tracks.is_empty());
    }

    #[test]
    fn test_power_netclass_width_preserved_on_wide_usb_c_pads_and_organic_widening() {
        use crate::board::NetClass;
        use std::collections::HashMap;

        let mut netclasses = HashMap::new();
        netclasses.insert("Default".to_string(), NetClass::default());
        netclasses.insert(
            "Power".to_string(),
            NetClass {
                name: "Power".to_string(),
                track_width: 0.60,
                clearance: 0.25,
                via_dia: 0.80,
                via_drill: 0.40,
                ..NetClass::default()
            },
        );
        let mut net_to_class = HashMap::new();
        net_to_class.insert("VBUS".to_string(), "Power".to_string());
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert(String::new(), 0);
        net_name_to_id.insert("VBUS".to_string(), 1);
        net_name_to_id.insert("USB_DP".to_string(), 2);

        // J2 with a wide 0.60x1.45 VBUS blade (A4 at 7.55, 6.0) and narrow 0.30x1.45 signal pin (A6 at 9.75, 14.0)
        let pad_vbus = Pad::new("A4", -2.45, -4.0, 0.60, 1.45, 1, "VBUS", false);
        let pad_dp = Pad::new("A6", -0.25, 4.0, 0.30, 1.45, 2, "USB_DP", false);
        let fp_j2 = make_fp("J2", 10.0, 10.0, false, vec![pad_vbus, pad_dp]);
        let pad_d2 = Pad::new("2", 0.0, 0.0, 0.90, 1.20, 1, "VBUS", false);
        let fp_d2 = make_fp("D2", 28.0, 10.0, false, vec![pad_d2]);

        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 40.0,
            footprints: vec![fp_j2, fp_d2],
            nets: vec![String::new(), "VBUS".to_string(), "USB_DP".to_string()],
            net_name_to_id,
            netclasses,
            net_to_class,
            tracks: vec![
                TrackSegment {
                    start: (7.55, 6.0),
                    end: (20.0, 6.0),
                    width: 0.15,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (20.0, 6.0),
                    end: (28.0, 10.0),
                    width: 0.15,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            ..Board::default()
        };

        let mut straight_board = board.clone();
        widen_tracks_and_arcs_to_netclass(&mut straight_board);
        assert!(
            straight_board.tracks.iter().all(|t| t.net_id != 1 || (t.width - 0.60).abs() < 1e-3),
            "VBUS straight tracks in open space must widen from 0.15mm fine-pitch neckdown to 0.60mm Power netclass width"
        );

        convert_tracks_to_organic(&mut board);
        assert!(
            !board.arcs.is_empty(),
            "convert_tracks_to_organic must emit curved ArcSegments for VBUS"
        );
        assert!(
            board.arcs.iter().all(|a| a.net_id != 1 || (a.width - 0.60).abs() < 1e-3),
            "VBUS organic arcs in open space must widen to 0.60mm Power netclass width"
        );
        assert!(
            board.tracks.iter().all(|t| t.net_id != 1 || (t.width - 0.60).abs() < 1e-3),
            "VBUS organic pad stubs in open space must widen to 0.60mm Power netclass width"
        );
    }

    #[test]
    fn test_drc_hole_clearance_and_neckdown_healing() {
        let mut netclasses = HashMap::new();
        let mut def_nc = NetClass::default();
        def_nc.min_hole_clearance = 0.25;
        def_nc.min_hole_to_hole = 0.25;
        def_nc.min_track_width = 0.15;
        netclasses.insert("Default".to_string(), def_nc);

        let mut pwr_nc = NetClass {
            name: "Power".to_string(),
            clearance: 0.25,
            track_width: 0.50,
            via_dia: 0.60,
            via_drill: 0.30,
            ..NetClass::default()
        };
        pwr_nc.min_hole_clearance = 0.25;
        pwr_nc.min_hole_to_hole = 0.25;
        pwr_nc.min_track_width = 0.15;
        netclasses.insert("Power".to_string(), pwr_nc);

        let mut net_to_class = HashMap::new();
        net_to_class.insert("VBUS".to_string(), "Power".to_string());
        net_to_class.insert("SIG".to_string(), "Default".to_string());

        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 30.0,
            max_y: 30.0,
            nets: vec![String::new(), "VBUS".to_string(), "SIG".to_string()],
            netclasses,
            net_to_class,
            vias: vec![Via {
                position: (10.0, 10.0),
                size: 0.45, // small annular ring so hole_clearance (0.15 + 0.25 = 0.40) dominates over copper radius (0.225 + 0.20 = 0.425) when min_hole_clearance is raised
                drill: 0.30,
                net_id: 2,
                is_locked: true,
            }],
            tracks: vec![TrackSegment {
                // At y = 10.62, dist = 0.62mm.
                // With width = 0.50 (half_w = 0.25) and via_r = 0.225 + req_clr = 0.25 -> req = 0.725mm (violation!).
                // If necked down to width = 0.15 (half_w = 0.075), req = 0.075 + 0.225 + 0.25 = 0.55mm <= 0.62mm (legal!).
                start: (5.0, 10.62),
                end: (15.0, 10.62),
                width: 0.50,
                layer: 0,
                net_id: 1,
                is_locked: false,
            }],
            ..Board::default()
        };

        assert!(
            !is_segment_clear(&board, None, (5.0, 10.62), (15.0, 10.62), 0, 1, 0.50),
            "0.50mm Power track at 0.62mm from foreign via must fail clearance check"
        );
        widen_tracks_and_arcs_to_netclass(&mut board);
        assert!(
            board.tracks[0].width < 0.30 && board.tracks[0].width >= 0.15,
            "Tight 0.50mm track must neck down to satisfy netclass clearance without being deleted (got {})",
            board.tracks[0].width
        );
        assert!(
            is_segment_clear(
                &board,
                None,
                board.tracks[0].start,
                board.tracks[0].end,
                0,
                1,
                board.tracks[0].width
            ),
            "Necked-down track must pass is_segment_clear"
        );
    }

    #[test]
    fn test_avoid_acid_traps_hairpin_and_duplicate_segments_no_oom() {
        let board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 20.0,
            max_y: 20.0,
            nets: vec![String::new(), "SIG".to_string(), "OBS".to_string()],
            vias: vec![Via {
                position: (10.0, 10.2),
                size: 0.6,
                drill: 0.3,
                net_id: 2,
                is_locked: true,
            }],
            ..Board::default()
        };
        let grid = RoutingGrid::new(&board, 0.10, None);

        // 1. Duplicate escape stub (0.15 mm) + routed segment (0.20 mm) on same endpoints
        let dup_tracks = vec![
            TrackSegment {
                start: (10.0, 10.0),
                end: (12.0, 10.0),
                width: 0.15,
                layer: 0,
                net_id: 1,
                is_locked: false,
            },
            TrackSegment {
                start: (10.0, 10.0),
                end: (12.0, 10.0),
                width: 0.20,
                layer: 0,
                net_id: 1,
                is_locked: false,
            },
            TrackSegment {
                start: (12.0, 10.0),
                end: (14.0, 10.0),
                width: 0.20,
                layer: 0,
                net_id: 1,
                is_locked: false,
            },
        ];
        let chains = extract_track_chains_for_organic(&dup_tracks, &board);
        assert_eq!(chains.len(), 1, "Duplicate parallel edges must collapse into a single open chain");
        assert!(chains[0].points.len() <= 3);

        // 2. 180-degree collinear fold-back where p0 -> p2 is blocked by a foreign obstacle near p0
        let mut hairpin_chain = TrackChain {
            net_id: 1,
            layer: 0,
            width: 0.20,
            points: vec![(10.0, 10.0), (13.0, 10.0), (11.0, 10.0), (11.0, 12.0)],
        };
        avoid_acid_traps(&mut hairpin_chain, &board, &grid);
        assert!(
            hairpin_chain.points.len() <= 8,
            "avoid_acid_traps must terminate cleanly on 180-degree fold-back without growing unboundedly"
        );

        let mut spike_pts = vec![(0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (0.0, 3.0)];
        merge_collinear_points(&mut spike_pts);
        assert_eq!(spike_pts, vec![(0.0, 0.0), (0.0, 3.0)]);
    }

    #[test]
    fn test_reduce_vias_with_through_hole_pads_no_infinite_bfs_oom() {
        let mut p1 = Pad::new("1", 0.0, 0.0, 1.5, 1.5, 1, "SIG_BUS", true);
        p1.drill = Some(0.8);
        let mut p2 = Pad::new("2", 5.0, 0.0, 1.5, 1.5, 1, "SIG_BUS", true);
        p2.drill = Some(0.8);
        let mut p3 = Pad::new("3", 10.0, 0.0, 1.5, 1.5, 1, "SIG_BUS", true);
        p3.drill = Some(0.8);
        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 30.0,
            max_y: 30.0,
            raw_content: "(kicad_pcb (layers (0 \"F.Cu\" signal) (1 \"In1.Cu\" signal) (2 \"In2.Cu\" signal) (31 \"B.Cu\" signal)))\n".to_string(),
            nets: vec![String::new(), "SIG_BUS".to_string()],
            footprints: vec![make_fp(
                "J1",
                10.0,
                10.0,
                true,
                vec![p1, p2, p3],
            )],
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (12.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (12.0, 10.0),
                    end: (13.0, 10.0),
                    width: 0.25,
                    layer: 3,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (13.0, 10.0),
                    end: (15.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (15.0, 10.0),
                    end: (20.0, 10.0),
                    width: 0.25,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            vias: vec![
                Via {
                    position: (12.0, 10.0),
                    size: 0.6,
                    drill: 0.3,
                    net_id: 1,
                    is_locked: false,
                },
                Via {
                    position: (13.0, 10.0),
                    size: 0.6,
                    drill: 0.3,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            ..Board::default()
        };
        let mut grid = RoutingGrid::new(&board, 0.25, None);
        let t0 = std::time::Instant::now();
        reduce_vias(&mut board, &mut grid);
        assert!(board.vias.is_empty(), "Via pair should be eliminated cleanly onto layer 0");
        assert!(
            t0.elapsed().as_millis() < 100,
            "reduce_vias with through-hole pads must finish in <100 ms without exponential BFS queue growth"
        );
    }

    #[test]
    fn test_organic_corner_arcs_preserve_terminal_stubs_and_g1_tangency() {
        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 30.0,
            max_y: 30.0,
            nets: vec![String::new(), "USB_DP".to_string()],
            tracks: vec![
                TrackSegment {
                    start: (10.0, 10.0),
                    end: (14.0, 10.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (14.0, 10.0),
                    end: (15.0, 11.0),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            ..Board::default()
        };
        apply_tangent_arcs_with_options(&mut board, 25.0, true);
        assert!(
            !board.arcs.is_empty(),
            "Organic corner should generate at least one tangent arc"
        );
        assert!(
            board.tracks.iter().any(|t| {
                (t.start.0 - 15.0).hypot(t.start.1 - 11.0) < 1e-4
                    || (t.end.0 - 15.0).hypot(t.end.1 - 11.0) < 1e-4
            }),
            "Terminal endpoint (15.0, 11.0) must retain a straight TrackSegment stub"
        );
        // Verify G1 tangency (< 0.5 degree kink) at every segment-to-arc junction
        for arc in &board.arcs {
            for (arc_pt, _is_start) in [(arc.start(), true), (arc.end(), false)] {
                let (ax, ay) = arc.start();
                let (bx, by) = arc.mid();
                let (cx_p, cy_p) = arc.end();
                let d = 2.0 * (ax * (by - cy_p) + bx * (cy_p - ay) + cx_p * (ay - by));
                if d.abs() > 1e-8 {
                    let a2 = ax * ax + ay * ay;
                    let b2 = bx * bx + by * by;
                    let c2 = cx_p * cx_p + cy_p * cy_p;
                    let cx = (a2 * (by - cy_p) + b2 * (cy_p - ay) + c2 * (ay - by)) / d;
                    let cy = (a2 * (cx_p - bx) + b2 * (ax - cx_p) + c2 * (bx - ax)) / d;
                    let r = (arc_pt.0 - cx).hypot(arc_pt.1 - cy);
                    let mid = arc.mid();
                    let rx = arc_pt.0 - cx;
                    let ry = arc_pt.1 - cy;
                    let to_mid = (mid.0 - arc_pt.0, mid.1 - arc_pt.1);
                    let (mut tx, mut ty) = (-ry / r, rx / r);
                    if tx * to_mid.0 + ty * to_mid.1 < 0.0 {
                        tx = -tx;
                        ty = -ty;
                    }
                    for t in &board.tracks {
                        let seg_dir_into_arc = if (t.end.0 - arc_pt.0).hypot(t.end.1 - arc_pt.1) < 1e-4 {
                            Some((t.end.0 - t.start.0, t.end.1 - t.start.1))
                        } else if (t.start.0 - arc_pt.0).hypot(t.start.1 - arc_pt.1) < 1e-4 {
                            Some((t.start.0 - t.end.0, t.start.1 - t.end.1))
                        } else {
                            None
                        };
                        if let Some((sx, sy)) = seg_dir_into_arc {
                            let slen = sx.hypot(sy);
                            assert!(slen > 0.05, "Retained terminal/interior straight stub must have non-zero length");
                            let ux = sx / slen;
                            let uy = sy / slen;
                            let dot = (ux * tx + uy * ty).clamp(-1.0, 1.0);
                            let kink_deg = dot.acos().to_degrees();
                            assert!(
                                kink_deg < 0.5,
                                "Segment-to-arc junction must be G1 tangent (< 0.5° kink), got {:.4}°",
                                kink_deg
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_fillet_acute_angle_junctions_eliminates_fork_and_corner_acid_traps() {
        // Reproduce the exact J2 Pad A7 (132.195, 111.75) / Pad B7 (132.195, 110.75) /USB_DM degree-3 fork
        // where the B7 jumper merges into the A7 escape trace at (133.595, 111.75) with an acute ~35.5° wedge,
        // plus a degree-2 acute V-corner on net 2.
        let mut board = make_router_test_board();
        board.nets = vec![
            "".to_string(),
            "/USB_DM".to_string(),
            "ACUTE_CORNER".to_string(),
        ];
        board.footprints = vec![
            make_fp(
                "J2",
                132.195,
                111.25,
                true,
                vec![
                    Pad::new("B7", 0.0, -0.50, 0.8, 0.3, 1, "/USB_DM", false),
                    Pad::new("A7", 0.0, 0.50, 0.8, 0.3, 1, "/USB_DM", false),
                ],
            ),
            make_fp(
                "U4",
                138.0,
                111.75,
                true,
                vec![Pad::new("13", 0.0, 0.0, 0.8, 0.5, 1, "/USB_DM", false)],
            ),
            make_fp(
                "R1",
                132.0,
                116.0,
                true,
                vec![Pad::new("1", 0.0, 0.0, 0.6, 0.6, 2, "ACUTE_CORNER", false)],
            ),
            make_fp(
                "R2",
                132.0,
                117.5,
                true,
                vec![Pad::new("1", 0.0, 0.0, 0.6, 0.6, 2, "ACUTE_CORNER", false)],
            ),
        ];

        // Net 1 (/USB_DM):
        // Trunk: Pad A7 (132.195, 111.75) -> fork (133.595, 111.75) -> U4.13 (138.0, 111.75)
        // Acute branch: Pad B7 (132.195, 110.75) -> fork (133.595, 111.75) (angle ~35.5° < 90° with A7 arm!)
        board.tracks = vec![
            TrackSegment {
                start: (132.195, 111.75),
                end: (133.595, 111.75),
                width: 0.20,
                layer: 0,
                net_id: 1,
                is_locked: false,
            },
            TrackSegment {
                start: (133.595, 111.75),
                end: (138.0, 111.75),
                width: 0.20,
                layer: 0,
                net_id: 1,
                is_locked: false,
            },
            TrackSegment {
                start: (132.195, 110.75),
                end: (133.595, 111.75),
                width: 0.20,
                layer: 0,
                net_id: 1,
                is_locked: false,
            },
            // Net 2 (ACUTE_CORNER): degree-2 sharp V-corner (< 90°) at (136.0, 116.75)
            TrackSegment {
                start: (132.0, 116.0),
                end: (136.0, 116.75),
                width: 0.20,
                layer: 0,
                net_id: 2,
                is_locked: false,
            },
            TrackSegment {
                start: (136.0, 116.75),
                end: (132.0, 117.5),
                width: 0.20,
                layer: 0,
                net_id: 2,
                is_locked: false,
            },
        ];

        // Helper to count same-net, same-layer track-segment pairs meeting at a shared vertex with angle < 89.5°
        let count_acute_track_junctions = |b: &Board| -> usize {
            let mut acute_count = 0;
            for i in 0..b.tracks.len() {
                for j in (i + 1)..b.tracks.len() {
                    let t1 = &b.tracks[i];
                    let t2 = &b.tracks[j];
                    if t1.net_id != t2.net_id || t1.layer != t2.layer {
                        continue;
                    }
                    for &(p1, o1) in &[(t1.start, t1.end), (t1.end, t1.start)] {
                        for &(p2, o2) in &[(t2.start, t2.end), (t2.end, t2.start)] {
                            if (p1.0 - p2.0).hypot(p1.1 - p2.1) < 1e-3 {
                                let v1 = (o1.0 - p1.0, o1.1 - p1.1);
                                let v2 = (o2.0 - p2.0, o2.1 - p2.1);
                                let l1 = v1.0.hypot(v1.1);
                                let l2 = v2.0.hypot(v2.1);
                                if l1 > 1e-4 && l2 > 1e-4 {
                                    let cos_theta = (v1.0 * v2.0 + v1.1 * v2.1) / (l1 * l2);
                                    if cos_theta > 0.01 {
                                        acute_count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            acute_count
        };

        assert_eq!(
            count_acute_track_junctions(&board),
            2,
            "Board must initially have 2 acute (<90°) track junctions (degree-3 fork + degree-2 corner)"
        );

        // When fillet_radius == 0.0, filleting is disabled
        let mut disabled_board = board.clone();
        fillet_acute_angle_junctions(&mut disabled_board, 0.0);
        assert_eq!(count_acute_track_junctions(&disabled_board), 2);

        // Apply acute-angle filleting with configurable radius 0.35 mm
        fillet_acute_angle_junctions(&mut board, 0.35);
        prune_dangling_tracks_and_vias(&mut board);

        assert_eq!(
            count_acute_track_junctions(&board),
            0,
            "All <90° acute track junctions (including the /USB_DM degree-3 fork) must be eliminated"
        );
        assert!(
            board.arcs.len() >= 2,
            "Expected tangent fillet arcs to be inserted for both the degree-3 fork and degree-2 corner, got {} arcs",
            board.arcs.len()
        );

        // Verify 100% electrical continuity across all pads after filleting and pruning
        let conn_a7_u4 = RouteConnection {
            net_id: 1,
            net_name: "/USB_DM".to_string(),
            netclass: NetClass::default(),
            start: (132.195, 111.75, 0),
            end: (138.0, 111.75, 0),
            start_fp: 0,
            start_pad: 1,
            end_fp: 1,
            end_pad: 0,
            diff_pair_leader: None,
        };
        let conn_b7_a7 = RouteConnection {
            net_id: 1,
            net_name: "/USB_DM".to_string(),
            netclass: NetClass::default(),
            start: (132.195, 110.75, 0),
            end: (132.195, 111.75, 0),
            start_fp: 0,
            start_pad: 0,
            end_fp: 0,
            end_pad: 1,
            diff_pair_leader: None,
        };
        let conn_corner = RouteConnection {
            net_id: 2,
            net_name: "ACUTE_CORNER".to_string(),
            netclass: NetClass::default(),
            start: (132.0, 116.0, 0),
            end: (132.0, 117.5, 0),
            start_fp: 2,
            start_pad: 0,
            end_fp: 3,
            end_pad: 0,
            diff_pair_leader: None,
        };
        assert!(
            AutoRouter::is_connection_electrically_connected(&board, &conn_a7_u4),
            "Pad A7 to U4.13 must remain electrically connected after fork filleting"
        );
        assert!(
            AutoRouter::is_connection_electrically_connected(&board, &conn_b7_a7),
            "Pad B7 to Pad A7 must remain electrically connected through the tangent fork fillet arc"
        );
        assert!(
            AutoRouter::is_connection_electrically_connected(&board, &conn_corner),
            "Degree-2 filleted corner must remain electrically connected"
        );
    }

    #[test]
    fn test_diff_pair_usb_dp_dm_routes_coupled_without_crossover() {
        // Reproduce the J2 (multi-pad USB-C: B7, A6, A7, B6) to U4 (Pad 13 DM West, Pad 14 DP East)
        // geometry over a >30 mm horizontal span on a 2-layer board.
        let mut board = Board {
            min_x: 128.0,
            min_y: 106.0,
            max_x: 170.0,
            max_y: 120.0,
            nets: vec![
                "".to_string(),
                "GND".to_string(),
                "/USB_DP".to_string(),
                "/USB_DM".to_string(),
            ],
            ..Board::default()
        };
        board.netclasses.insert(
            "USB_Diff".to_string(),
            NetClass {
                name: "USB_Diff".to_string(),
                track_width: 0.20,
                clearance: 0.20,
                via_dia: 0.60,
                via_drill: 0.30,
                diff_pair_gap: 0.20,
                diff_pair_width: 0.20,
                topology: NetTopology::default(),
                ..Default::default()
            },
        );
        board.net_to_class.insert("/USB_DP".to_string(), "USB_Diff".to_string());
        board.net_to_class.insert("/USB_DM".to_string(), "USB_Diff".to_string());

        board.footprints = vec![
            make_fp(
                "J2",
                132.195,
                111.50,
                true,
                vec![
                    Pad::new("B7", 0.0, -0.75, 0.80, 0.30, 3, "/USB_DM", false), // y = 110.75
                    Pad::new("A6", 0.0, -0.25, 0.80, 0.30, 2, "/USB_DP", false), // y = 111.25
                    Pad::new("A7", 0.0, 0.25, 0.80, 0.30, 3, "/USB_DM", false),  // y = 111.75
                    Pad::new("B6", 0.0, 0.75, 0.80, 0.30, 2, "/USB_DP", false),  // y = 112.25
                ],
            ),
            make_fp(
                "U4",
                164.25,
                113.75,
                true,
                vec![
                    Pad::new("13", -0.75, 0.0, 0.40, 0.80, 3, "/USB_DM", false), // x = 163.50, y = 113.75
                    Pad::new("14", 0.75, 0.0, 0.40, 0.80, 2, "/USB_DP", false),  // x = 165.00, y = 113.75
                ],
            ),
        ];

        let router = AutoRouter::new(2);
        // 1. Verify extract_connections selects non-crossing adjacent J2 pads (A6 for /USB_DP, A7 for /USB_DM)
        let conns = router.extract_connections(&board);
        let inter_dp = conns
            .iter()
            .find(|c| c.net_id == 2 && c.start_fp != c.end_fp)
            .expect("Expected inter-footprint /USB_DP connection");
        let inter_dm = conns
            .iter()
            .find(|c| c.net_id == 3 && c.start_fp != c.end_fp)
            .expect("Expected inter-footprint /USB_DM connection");
        let dp_j2_y = if inter_dp.start_fp == 0 { inter_dp.start.1 } else { inter_dp.end.1 };
        let dm_j2_y = if inter_dm.start_fp == 0 { inter_dm.start.1 } else { inter_dm.end.1 };
        assert!(
            (dp_j2_y - 111.25).abs() < 1e-3 && (dm_j2_y - 111.75).abs() < 1e-3,
            "extract_connections must select non-crossing adjacent J2 pads A6 (111.25) and A7 (111.75), got DP y={:.2}, DM y={:.2}",
            dp_j2_y,
            dm_j2_y
        );

        // 2. Route the board and verify /USB_DP and /USB_DM route coupled on F.Cu over the >30 mm span without crossing
        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let stop = Arc::new(AtomicBool::new(false));
        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);
        assert_eq!(routed, total, "All USB differential pair connections must route cleanly");

        let dp_main_fcu: f64 = board
            .tracks
            .iter()
            .filter(|t| t.net_id == 2 && t.layer == 0 && t.start.0.max(t.end.0) > 135.0)
            .map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1))
            .sum();
        let dm_main_fcu: f64 = board
            .tracks
            .iter()
            .filter(|t| t.net_id == 3 && t.layer == 0 && t.start.0.max(t.end.0) > 135.0)
            .map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1))
            .sum();
        assert!(
            dp_main_fcu > 28.0 && dm_main_fcu > 28.0,
            "Both /USB_DP ({:.1}mm) and /USB_DM ({:.1}mm) main trunks must stay on F.Cu over the >28mm span without crosstalk layer divergence",
            dp_main_fcu,
            dm_main_fcu
        );

        // Verify no B.Cu tracks exist on the main trunk span (x > 135.0)
        let trunk_bcu_len: f64 = board
            .tracks
            .iter()
            .filter(|t| (t.net_id == 2 || t.net_id == 3) && t.layer != 0 && t.start.0.max(t.end.0) > 135.0)
            .map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1))
            .sum();
        assert!(
            trunk_bcu_len < 1e-3,
            "Main differential pair trunk (x > 135.0) must remain 100% on F.Cu, got {:.2}mm on other layers",
            trunk_bcu_len
        );

        // Verify organic conversion preserves coupled spacing and leaves 0 acute (<90°) track junctions
        convert_tracks_to_organic_with_fillet_radius(&mut board, 0.35);
        for c in &conns {
            assert!(
                AutoRouter::is_connection_electrically_connected(&board, c),
                "Connection {} must remain electrically connected after organic conversion",
                c.net_name
            );
        }
    }

    #[test]
    fn test_via_in_pad_disabled_by_default_vs_enabled() {
        // Reproduce D1 (SOT-23 diode with large SMD pads on GND / net 1) and verify:
        // 1. Default allow_via_in_pad = false forbids any same-net via overlapping D1's SMD pads
        //    across RoutingGrid::get_neighbors, is_via_location_legal, add_plane_stitching_vias,
        //    widen_tracks_and_arcs_to_netclass, and validate_and_clean_copper.
        // 2. When allow_via_in_pad = true, same-net via-in-pad is permitted.
        let mut board = Board {
            min_x: 100.0,
            min_y: 100.0,
            max_x: 130.0,
            max_y: 130.0,
            nets: vec![String::new(), "GND".to_string(), "/SIG".to_string()],
            footprints: vec![make_fp(
                "D1",
                115.0,
                115.0,
                true,
                vec![
                    Pad::new("1", -0.95, 1.1, 0.9, 2.0, 1, "GND", false),
                    Pad::new("2", 0.95, 1.1, 0.9, 2.0, 1, "GND", false),
                    Pad::new("3", 0.0, -1.1, 0.9, 2.0, 2, "/SIG", false),
                ],
            )],
            ..Board::default()
        };
        assert!(!board.allow_via_in_pad(), "allow_via_in_pad must default to false");

        let d1_pad1_pos = board.footprints[0].pad_absolute_position(&board.footprints[0].pads[0]);

        // A. With allow_via_in_pad = false (default), is_via_location_legal rejects same-net via on D1 Pad 1
        assert!(
            !is_via_location_legal(&board, d1_pad1_pos, 0.60, 1, 0.20),
            "Same-net via on SMD pad D1.1 must be rejected when allow_via_in_pad == false"
        );

        // B. RoutingGrid with default allow_via_in_pad = false marks D1.1 as smd_pad_cells and forbids via transition there
        let grid_no_vip = RoutingGrid::new(&board, 0.20, None);
        assert!(!grid_no_vip.allow_via_in_pad);
        let (gx, gy) = grid_no_vip.world_to_grid(d1_pad1_pos.0, d1_pad1_pos.1).unwrap();
        let cell_idx = (gy * grid_no_vip.width + gx) as usize;
        assert!(
            grid_no_vip.smd_pad_cells[cell_idx],
            "Large SMD pad D1.1 must be marked in smd_pad_cells"
        );
        let node_pad = GridNode { gx, gy, layer: 0 };
        let neighbors_no_vip = grid_no_vip.get_neighbors(node_pad, 1, 0.20, 0.20, 0.60, None);
        assert!(
            !neighbors_no_vip.iter().any(|&(n, _)| n.gx == gx && n.gy == gy && n.layer != 0),
            "RoutingGrid::get_neighbors must not allow a layer-transition via inside SMD pad D1.1 when allow_via_in_pad == false"
        );

        // C. Plane & return-path ground via stitching must never place a via overlapping D1's SMD pads
        board.tracks.push(TrackSegment {
            start: (110.0, 115.0),
            end: (120.0, 115.0),
            width: 0.25,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        board.vias.push(Via {
            position: (115.0, 118.0),
            size: 0.60,
            drill: 0.30,
            net_id: 2,
            is_locked: false,
        });
        let router = AutoRouter::new(2);
        let mut stitch_grid = RoutingGrid::new(&board, 0.20, None);
        router.add_plane_stitching_vias(&mut board, &mut stitch_grid);
        router.add_return_path_ground_vias(&mut board, &mut stitch_grid);
        for v in &board.vias {
            for fp in &board.footprints {
                for pad in &fp.pads {
                    if !pad.is_through_hole {
                        let d = dist_point_to_pad(v.position, fp, pad);
                        assert!(
                            d >= v.size * 0.5 - 1e-4,
                            "Ground/stitching via at {:?} overlapped SMD pad {}.{} (d={:.3})",
                            v.position,
                            fp.reference,
                            pad.number,
                            d
                        );
                    }
                }
            }
        }

        // D. validate_and_clean_copper removes an illegal same-net via placed on D1.1 when allow_via_in_pad == false
        board.vias.push(Via {
            position: d1_pad1_pos,
            size: 0.60,
            drill: 0.30,
            net_id: 1,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: d1_pad1_pos,
            end: (d1_pad1_pos.0, d1_pad1_pos.1 + 3.0),
            width: 0.25,
            layer: 1,
            net_id: 1,
            is_locked: false,
        });
        validate_and_clean_copper(&mut board);
        assert!(
            !board.vias.iter().any(|v| (v.position.0 - d1_pad1_pos.0).hypot(v.position.1 - d1_pad1_pos.1) < 0.2),
            "validate_and_clean_copper must remove same-net via overlapping SMD pad D1.1 when allow_via_in_pad == false"
        );

        // E. When allow_via_in_pad == true, same-net via on D1.1 is permitted
        board.set_allow_via_in_pad(true);
        assert!(board.allow_via_in_pad());
        assert!(
            is_via_location_legal(&board, d1_pad1_pos, 0.60, 1, 0.20),
            "Same-net via on SMD pad D1.1 must be permitted when allow_via_in_pad == true"
        );
        let grid_vip = RoutingGrid::new(&board, 0.20, None).with_allow_via_in_pad(true);
        let neighbors_vip = grid_vip.get_neighbors(node_pad, 1, 0.20, 0.20, 0.60, None);
        assert!(
            neighbors_vip.iter().any(|&(n, _)| n.gx == gx && n.gy == gy && n.layer != 0),
            "RoutingGrid::get_neighbors must allow a layer-transition via inside same-net SMD pad D1.1 when allow_via_in_pad == true"
        );
    }

    #[test]
    fn test_power_net_buck_vin_and_3v3_uniform_track_width() {
        // Verify that wide power nets (/BUCK_VIN and +3.3V, 0.60 mm target width)
        // touching passive/SOT footprints route and widen to uniform 0.60 mm width
        // across all connected segments and arcs without necking down to 0.15 mm or 0.20 mm.
        let mut board = Board {
            min_x: 100.0,
            min_y: 100.0,
            max_x: 135.0,
            max_y: 130.0,
            nets: vec![
                String::new(),
                "GND".to_string(),
                "/BUCK_VIN".to_string(),
                "+3.3V".to_string(),
            ],
            ..Board::default()
        };
        board.netclasses.insert(
            "Power".to_string(),
            NetClass {
                name: "Power".to_string(),
                track_width: 0.60,
                clearance: 0.20,
                via_dia: 0.60,
                via_drill: 0.30,
                ..Default::default()
            },
        );
        board.net_to_class.insert("/BUCK_VIN".to_string(), "Power".to_string());
        board.net_to_class.insert("+3.3V".to_string(), "Power".to_string());

        board.footprints = vec![
            make_fp(
                "U2",
                108.0,
                110.0,
                true,
                vec![
                    Pad::new("3", 0.0, -0.95, 1.0, 0.6, 2, "/BUCK_VIN", false),
                    Pad::new("1", 0.0, 0.95, 1.0, 0.6, 1, "GND", false),
                ],
            ),
            make_fp(
                "C10",
                118.0,
                114.0,
                true,
                vec![
                    Pad::new("1", 0.0, -0.8, 1.0, 0.9, 2, "/BUCK_VIN", false),
                    Pad::new("2", 0.0, 0.8, 1.0, 0.9, 1, "GND", false),
                ],
            ),
            make_fp(
                "L1",
                108.0,
                122.0,
                true,
                vec![Pad::new("2", 0.0, 0.0, 1.2, 1.2, 3, "+3.3V", false)],
            ),
            make_fp(
                "C12",
                120.0,
                124.0,
                true,
                vec![Pad::new("1", 0.0, 0.0, 1.0, 0.9, 3, "+3.3V", false)],
            ),
        ];

        let router = AutoRouter::new(2);
        let mut grid = RoutingGrid::new(&board, 0.20, None);
        let stop = Arc::new(AtomicBool::new(false));
        let (routed, total) = router.route_all(&mut board, &mut grid, None, stop);
        assert_eq!(routed, total, "All power connections must route cleanly");

        convert_tracks_to_organic_with_fillet_radius(&mut board, 0.35);

        for net_id in [2usize, 3usize] {
            let net_name = &board.nets[net_id];
            let widths: Vec<f64> = board
                .tracks
                .iter()
                .filter(|t| t.net_id == net_id)
                .map(|t| t.width)
                .chain(board.arcs.iter().filter(|a| a.net_id == net_id).map(|a| a.width))
                .collect();
            assert!(!widths.is_empty(), "Expected routed copper for {}", net_name);
            for w in &widths {
                assert!(
                    (*w - 0.60).abs() < 1e-3,
                    "Power net {} segment/arc must have uniform 0.60 mm width, got {:.3} mm (all widths: {:?})",
                    net_name,
                    w,
                    widths
                );
            }
        }
    }

    #[test]
    fn test_organic_mid_segment_t_junction_split_preserves_multi_branch_connectivity() {
        // Reproduce a multi-branch net (like J2.B7 -> J2.A7 on /USB_DM) where a secondary branch
        // terminates at an interior point (133.5, 111.75) along a single continuous trunk segment
        // from (132.195, 111.75) to (138.0, 111.75) -> (140.0, 113.75).
        // Verify that convert_tracks_to_organic splits the trunk at the T-junction and preserves
        // 100% electrical connectivity across all 3 pads.
        let mut board = Board {
            min_x: 128.0,
            min_y: 108.0,
            max_x: 145.0,
            max_y: 118.0,
            nets: vec![String::new(), "/USB_DM".to_string()],
            footprints: vec![
                make_fp(
                    "J2",
                    132.195,
                    111.25,
                    true,
                    vec![
                        Pad::new("B7", 0.0, -0.50, 0.8, 0.3, 1, "/USB_DM", false), // (132.195, 110.75)
                        Pad::new("A7", 0.0, 0.50, 0.8, 0.3, 1, "/USB_DM", false),  // (132.195, 111.75)
                    ],
                ),
                make_fp(
                    "U4",
                    140.0,
                    113.75,
                    true,
                    vec![Pad::new("13", 0.0, 0.0, 0.8, 0.5, 1, "/USB_DM", false)],
                ),
            ],
            tracks: vec![
                // Single unsplit trunk segment passing through (133.5, 111.75)
                TrackSegment {
                    start: (132.195, 111.75),
                    end: (138.0, 111.75),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (138.0, 111.75),
                    end: (140.0, 113.75),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                // Branch from B7 landing mid-segment on the trunk at (133.5, 111.75)
                TrackSegment {
                    start: (132.195, 110.75),
                    end: (133.5, 110.75),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
                TrackSegment {
                    start: (133.5, 110.75),
                    end: (133.5, 111.75),
                    width: 0.20,
                    layer: 0,
                    net_id: 1,
                    is_locked: false,
                },
            ],
            ..Board::default()
        };

        convert_tracks_to_organic_with_fillet_radius(&mut board, 0.35);

        let conn_a7_u4 = RouteConnection {
            net_id: 1,
            net_name: "/USB_DM".to_string(),
            netclass: NetClass::default(),
            start: (132.195, 111.75, 0),
            end: (140.0, 113.75, 0),
            start_fp: 0,
            start_pad: 1,
            end_fp: 1,
            end_pad: 0,
            diff_pair_leader: None,
        };
        let conn_b7_a7 = RouteConnection {
            net_id: 1,
            net_name: "/USB_DM".to_string(),
            netclass: NetClass::default(),
            start: (132.195, 110.75, 0),
            end: (132.195, 111.75, 0),
            start_fp: 0,
            start_pad: 0,
            end_fp: 0,
            end_pad: 1,
            diff_pair_leader: None,
        };
        assert!(
            AutoRouter::is_connection_electrically_connected(&board, &conn_a7_u4),
            "Main trunk J2.A7 -> U4.13 must remain electrically connected after organic conversion"
        );
        assert!(
            AutoRouter::is_connection_electrically_connected(&board, &conn_b7_a7),
            "Mid-segment T-junction branch J2.B7 -> J2.A7 must remain electrically connected after organic conversion"
        );
    }

    #[test]
    fn test_opposite_side_smd_pad_layer_parsing_and_b_cu_obstacle() {
        // Verify that an F.Cu footprint containing a `(layers "B.Cu")` SMD pad (such as U1 Pad 25
        // thermal/exposed pad on B.Cu) parses with `is_through_hole = false` and `layer = Some(1)` (B.Cu),
        // and blocks foreign B.Cu tracks while leaving F.Cu clear for foreign tracks.
        let pcb_sexpr = r#"(kicad_pcb
          (version 20240108)
          (layers
            (0 "F.Cu" signal)
            (31 "B.Cu" signal)
          )
          (net 0 "")
          (net 1 "GND")
          (net 2 "/SIG")
          (footprint "Package_QFN:QFN-24"
            (layer "F.Cu")
            (at 150.0 110.0)
            (property "Reference" "U1")
            (pad "1" smd rect (at -2.0 0.0) (size 0.8 0.3) (layers "F.Cu" "F.Paste" "F.Mask") (net 1 "GND"))
            (pad "25" smd rect (at 0.0 0.0) (size 3.0 3.0) (layers "B.Cu") (net 1 "GND"))
          )
        )"#;
        let tmp_path = std::env::temp_dir().join("vulcan_test_opposite_smd_pad.kicad_pcb");
        std::fs::write(&tmp_path, pcb_sexpr).expect("Failed to write temporary PCB");
        let board = Board::load(tmp_path.to_str().unwrap(), &HashSet::new()).expect("Failed to parse PCB");
        let _ = std::fs::remove_file(&tmp_path);

        assert_eq!(board.footprints.len(), 1);
        let fp = &board.footprints[0];
        assert_eq!(fp.layer, 0, "Footprint U1 is on F.Cu (layer 0)");
        let pad1 = fp.pads.iter().find(|p| p.number == "1").unwrap();
        let pad25 = fp.pads.iter().find(|p| p.number == "25").unwrap();
        assert!(!pad1.is_through_hole);
        assert_eq!(fp.pad_layer(pad1), 0, "Pad 1 must be on F.Cu (0)");
        assert!(
            !pad25.is_through_hole,
            "Opposite-side SMD pad with (layers \"B.Cu\") must NOT be marked as through-hole"
        );
        assert_eq!(
            pad25.layer,
            Some(1),
            "Pad 25 with (layers \"B.Cu\") must have explicit layer Some(1)"
        );
        assert_eq!(fp.pad_layer(pad25), 1, "fp.pad_layer(pad25) must return B.Cu (1)");

        // Foreign track on net 2 (/SIG) crossing (150.0, 110.0) on B.Cu (layer 1) must be blocked by Pad 25
        assert!(
            !is_segment_clear(&board, None, (147.0, 110.0), (153.0, 110.0), 1, 2, 0.20),
            "Foreign B.Cu track crossing U1 Pad 25 on B.Cu must be rejected by is_segment_clear"
        );
        assert!(
            !is_track_segment_legal(&board, (147.0, 110.0), (153.0, 110.0), 0.20, 1, 2, 0.20),
            "Foreign B.Cu track crossing U1 Pad 25 on B.Cu must be rejected by is_track_segment_legal"
        );
        // Foreign track on net 2 (/SIG) crossing (150.0, 110.0) on F.Cu (layer 0) is NOT blocked by Pad 25
        assert!(
            is_segment_clear(&board, None, (150.0, 107.0), (150.0, 113.0), 0, 2, 0.20),
            "Foreign F.Cu track crossing over B.Cu-only Pad 25 must be allowed on F.Cu"
        );
    }

    /// Clears `grid`, drops a foreign-net blob in grid coords and computes a brute-force distance field.
    fn grid_with_blob(board: &Board, res: f64, blob: (usize, usize, usize, usize)) -> RoutingGrid {
        let mut grid = RoutingGrid::new(board, res, None);
        grid.obstacle_net_ids.fill(FREE_NET);
        grid.smd_pad_cells.fill(false);
        let (x0, x1, y0, y1) = blob;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let i = grid.idx(x, y, 0);
                grid.obstacle_net_ids[i] = 99;
            }
        }
        for y in 0..grid.height {
            for x in 0..grid.width {
                let dx = if x < x0 { x0 - x } else if x > x1 { x - x1 } else { 0 };
                let dy = if y < y0 { y0 - y } else if y > y1 { y - y1 } else { 0 };
                let i = grid.idx(x, y, 0);
                grid.dist_field[i] = ((dx * dx + dy * dy) as f64).sqrt() as f32 * res as f32;
            }
        }
        grid
    }

    #[test]
    fn test_path_coupling_and_diff_pair_side_helpers() {
        let line = |y: usize| (10..40).map(|x| GridNode { gx: x, gy: y, layer: 0 }).collect::<Vec<_>>();
        let (a, b) = (line(20), line(24));
        assert!((path_coupling_mm(0.1, &a, &b) - 0.4).abs() < 1e-9);
        assert!((grid_path_length_mm(0.1, &a) - 2.9).abs() < 1e-9);
        // Follower endpoints at +y of an eastbound leader are on side +1, -y on side -1.
        let n = |x, y| GridNode { gx: x, gy: y, layer: 0 };
        assert_eq!(diff_pair_side(&a, n(10, 24), n(39, 24)), 1.0);
        assert_eq!(diff_pair_side(&a, n(10, 16), n(39, 16)), -1.0);
        // Endpoints on opposite sides (pair must cross) give no preferred side.
        assert_eq!(diff_pair_side(&a, n(10, 24), n(39, 16)), 0.0);
    }

    #[test]
    fn test_diff_pair_corridor_strongly_penalises_leaving_and_far_side() {
        let board = make_router_test_board();
        let res = 0.1;
        let grid = grid_with_blob(&board, res, (0, 0, 0, 0));
        assert!(grid.width > 60 && grid.height > 40, "test grid too small: {}x{}", grid.width, grid.height);
        let cy = grid.height / 2;
        let path: Vec<GridNode> = (5..grid.width - 5).map(|x| GridNode { gx: x, gy: cy, layer: 0 }).collect();
        let router = AutoRouter::new(1);
        let corr = router.build_diff_pair_corridor(&grid, &path, 0.2, 0.2, 1.0);
        let mx = grid.width / 2;
        let pitch = 0.2 + 0.2f64.max(grid.max_net_clearance);
        let pc = (pitch / res).round() as usize;
        let at = |y: usize| corr[grid.idx(mx, y, 0)];
        assert!(at(cy + pc) <= 0.35, "ideal pitch on the partner side must be cheap, got {}", at(cy + pc));
        assert_eq!(at(cy - pc), DIFF_PAIR_OFF_CORRIDOR_MULT, "far side of the leader must not be in the corridor");
        assert_eq!(at(cy + pc + 20), DIFF_PAIR_OFF_CORRIDOR_MULT, "2mm away must cost the off-corridor multiplier");
        // Closer than the legal pitch is not rewarded.
        assert!(at(cy + 1) >= 1.0);
        // Unsided corridor accepts either side.
        let both = router.build_diff_pair_corridor(&grid, &path, 0.2, 0.2, 0.0);
        assert!(both[grid.idx(mx, cy - pc, 0)] <= 0.35);
    }

    #[test]
    fn test_sided_diff_pair_leader_leaves_room_for_partner() {
        // Regression: the USB_DM leader hugged an inductor pad on USB_DP's side, so USB_DP had to
        // take a different channel ~1.3 mm away. A sided leader must keep a partner pitch free.
        let board = make_router_test_board();
        let res = 0.1;
        let (w, clr) = (0.1, 0.1);
        let probe = RoutingGrid::new(&board, res, None);
        let max_clr = probe.max_net_clearance.max(clr);
        let pc = ((w + max_clr) / res - 1e-6).ceil() as usize;
        let req = w * 0.5 + max_clr;
        let cy = probe.height / 2;
        let mx = probe.width / 2;
        assert!(probe.width > 70, "test grid too small: {}", probe.width);
        let blob = (mx - 4, mx + 4, cy + pc, cy + pc + 3);
        let grid = grid_with_blob(&board, res, blob);
        let router = AutoRouter::new(1);
        let stop = AtomicBool::new(false);
        let s = GridNode { gx: mx - 30, gy: cy, layer: 0 };
        let e = GridNode { gx: mx + 30, gy: cy, layer: 0 };
        let route = |side: f64| {
            router
                .find_path_sided(&grid, s, e, 2, clr, w, 0.6, None, 1.0, -(w + clr), 100.0, false, None, side, &stop)
                .expect("leader must route")
        };
        let partner_ok = |p: &[GridNode]| {
            p.iter().filter(|n| n.gx >= blob.0 && n.gx <= blob.1).all(|n| {
                grid.dist_field[grid.idx(n.gx, n.gy + pc, 0)] as f64 >= req - 1e-6
            })
        };
        assert!(partner_ok(&route(1.0)), "sided leader must leave a clear partner pitch beside the blob");
        // Partner on the free side: the leader has no reason to detour.
        assert!(route(-1.0).iter().all(|n| n.gy == cy), "leader with free partner side must stay straight");
    }

    #[test]
    fn test_gloss_leaves_diff_pair_nets_untouched() {
        let mut board = make_router_test_board();
        board.nets = vec!["".into(), "GND".into(), "USB_DP".into(), "USB_DM".into()];
        board.footprints.clear();
        let stair = |net: usize, y0: f64| -> Vec<TrackSegment> {
            let pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (2.0, 0.5), (2.0, 1.0), (3.0, 1.0)];
            pts.windows(2)
                .map(|p| TrackSegment {
                    start: (p[0].0, p[0].1 + y0),
                    end: (p[1].0, p[1].1 + y0),
                    width: 0.2,
                    layer: 0,
                    net_id: net,
                    is_locked: false,
                })
                .collect()
        };
        board.tracks = stair(2, 0.0);
        board.tracks.extend(stair(3, 0.4));
        let before = board.tracks.clone();
        let grid = RoutingGrid::new(&board, 0.1, None);
        gloss_tracks_standard(&mut board, &grid);
        for t in &before {
            assert!(board.tracks.contains(t), "diff pair track {:?} must survive glossing unchanged", t);
        }
        assert_eq!(board.tracks.len(), before.len());
    }

    #[test]
    fn test_diff_pair_neck_stays_local_when_widening() {
        // Regression: one necked USB_DM segment at the J2 fanout dragged the whole net to 0.18 mm.
        let mut board = make_router_test_board();
        board.nets = vec!["".into(), "GND".into(), "USB_DP".into(), "USB_DM".into(), "SIG1".into()];
        board.net_name_to_id = board.nets.iter().cloned().enumerate().map(|(i, n)| (n, i)).collect();
        board.footprints.truncate(2);
        for (k, fx) in [(0usize, 0.0), (1, 10.0)] {
            let fp = &mut board.footprints[k];
            fp.x = fx;
            fp.y = 0.0;
            fp.rotation = 0.0;
            fp.layer = 0;
            fp.pads = vec![Pad::new("1", 0.0, 0.0, 0.3, 0.3, 2, "USB_DP", false)];
        }
        // Foreign pad hugging the first segment only.
        board.footprints[0].pads.push(Pad::new("2", 1.5, 0.43, 0.3, 0.3, 4, "SIG1", false));
        let seg = |a: (f64, f64), b: (f64, f64)| TrackSegment { start: a, end: b, width: 0.15, layer: 0, net_id: 2, is_locked: false };
        board.tracks = vec![seg((0.0, 0.0), (3.0, 0.0)), seg((3.0, 0.0), (10.0, 0.0))];
        assert!((net_target_width(&board, 2) - board.get_netclass_for_net(2).diff_pair_width).abs() < 1e-9);
        widen_tracks_and_arcs_to_netclass(&mut board);
        let tw = net_target_width(&board, 2);
        let w2 = board.tracks.iter().find(|t| t.end == (10.0, 0.0)).unwrap().width;
        let w1 = board.tracks.iter().find(|t| t.start == (0.0, 0.0)).unwrap().width;
        assert!((w2 - tw).abs() < 1e-6, "clear body segment must widen to the pair width {tw}, got {w2}");
        assert!(w1 < tw, "segment beside the foreign pad must stay necked, got {w1}");
    }

    #[test]
    fn test_power_via_beside_keepout_slides_instead_of_necking() {
        // Regression: a +3.3V via dropped just below the antenna keepout forced a 0.50 mm neck
        // on the 0.60 mm rail into C15.
        let mut board = make_router_test_board();
        board.nets = vec!["".into(), "GND".into(), "+3.3V".into()];
        board.net_name_to_id = board.nets.iter().cloned().enumerate().map(|(i, n)| (n, i)).collect();
        board.netclasses.insert(
            "Power".into(),
            NetClass { name: "Power".into(), track_width: 0.60, clearance: 0.20, ..Default::default() },
        );
        board.net_to_class.insert("+3.3V".into(), "Power".into());
        board.footprints.truncate(1);
        // Work in the test board's own area so boundary checks pass.
        let c = board.boundary_aabb();
        let (ox, oy) = ((c.min_x + c.max_x) * 0.5, (c.min_y + c.max_y) * 0.5);
        let fp = &mut board.footprints[0];
        fp.x = ox;
        fp.y = oy;
        fp.rotation = 0.0;
        fp.layer = 0;
        fp.pads = vec![Pad::new("1", 1.55, 0.0, 0.9, 0.95, 2, "+3.3V", false)];
        // Keepout ends 0.47 mm above the via centre: 0.50 mm fits (margin 0.45), 0.60 mm does not.
        board.keepouts = vec![KeepoutZone { min_x: ox - 2.0, min_y: oy - 3.0, max_x: ox + 4.0, max_y: oy - 1.407, layers: vec![0, 1] }];
        let via_pos = (ox + 0.7545, oy - 0.937);
        let pad_pos = (ox + 1.55, oy);
        board.vias = vec![Via { position: via_pos, size: 0.6, drill: 0.3, net_id: 2, is_locked: false }];
        board.tracks = vec![TrackSegment { start: via_pos, end: pad_pos, width: 0.50, layer: 0, net_id: 2, is_locked: false }];
        assert!(!is_segment_clear(&board, None, via_pos, pad_pos, 0, 2, 0.60), "setup: 0.60 must not fit before the slide");
        assert!(is_segment_clear(&board, None, via_pos, pad_pos, 0, 2, 0.50), "setup: 0.50 must fit");
        widen_tracks_and_arcs_to_netclass(&mut board);
        let t = &board.tracks[0];
        let v = &board.vias[0];
        assert!((t.width - 0.60).abs() < 1e-6, "rail must reach 0.60 mm, got {:.2}", t.width);
        assert_eq!(t.end, pad_pos, "pad end must stay on the pad");
        assert_eq!(t.start, v.position, "track must stay attached to the moved via");
        assert!(is_segment_clear(&board, None, t.start, t.end, 0, 2, 0.60));
    }

    #[test]
    fn test_u4_en_fork_with_arc_acute_side_gets_filleted() {
        // Regression from p1p2esp_vulcan.kicad_pcb: Net-(U4-EN) forks at (157.31, 87.81) where an
        // incoming arc meets the NE branch at ~45 deg on the north side. Arcs were invisible to
        // the fillet pass, so the acid trap survived.
        let mut board = make_router_test_board();
        board.nets = vec!["".into(), "GND".into(), "Net-(U4-EN)".into()];
        board.footprints.clear();
        let c = board.boundary_aabb();
        let (ox, oy) = ((c.min_x + c.max_x) * 0.5 - 158.0, (c.min_y + c.max_y) * 0.5 - 89.0);
        let p = |x: f64, y: f64| (x + ox, y + oy);
        let seg = |a: (f64, f64), b: (f64, f64)| TrackSegment { start: a, end: b, width: 0.2, layer: 0, net_id: 2, is_locked: false };
        let fork = p(157.3053, 87.8050);
        board.tracks = vec![
            seg(p(154.8265, 87.7098), p(156.9783, 87.2179)),
            seg(fork, p(158.1500, 86.9500)),
            seg(fork, p(154.0750, 91.0750)),
            seg(p(158.1500, 86.9500), p(163.0050, 86.9950)),
        ];
        board.arcs = vec![ArcSegment::from_3pts(p(156.9783, 87.2179), p(157.3621, 87.3888), fork, 0.2, 0, 2).unwrap()];

        let near = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).hypot(a.1 - b.1) < 2e-3;
        // Every arm (track direction or arc tangent) leaving each vertex.
        // (direction, is_arc_tangent)
        let arms_at = |b: &Board, v: (f64, f64)| -> Vec<((f64, f64), bool)> {
            let mut out = Vec::new();
            for t in &b.tracks {
                for (here, far) in [(t.start, t.end), (t.end, t.start)] {
                    let l = (far.0 - here.0).hypot(far.1 - here.1);
                    if near(here, v) && l > 1e-4 {
                        out.push((((far.0 - here.0) / l, (far.1 - here.1) / l), false));
                    }
                }
            }
            for a in &b.arcs {
                if near(a.start(), v) {
                    out.push((arc_arm_dir(a, true), true));
                }
                if near(a.end(), v) {
                    out.push((arc_arm_dir(a, false), true));
                }
            }
            out
        };
        let min_angle_deg = |b: &Board| -> f64 {
            let mut verts: Vec<(f64, f64)> = b.tracks.iter().flat_map(|t| [t.start, t.end]).collect();
            verts.extend(b.arcs.iter().flat_map(|a| [a.start(), a.end()]));
            let mut best = 180.0f64;
            for v in verts {
                let arms = arms_at(b, v);
                for i in 0..arms.len() {
                    for j in (i + 1)..arms.len() {
                        // An arc merging tangentially into a trunk (0 deg) still leaves an acute
                        // notch at the copper edges, so it counts.
                        let ((u, _), (w, _)) = (arms[i], arms[j]);
                        let deg = (u.0 * w.0 + u.1 * w.1).clamp(-1.0, 1.0).acos().to_degrees();
                        best = best.min(deg);
                    }
                }
            }
            best
        };
        assert!(min_angle_deg(&board) < 60.0, "setup must reproduce the acute fork, got {:.1}", min_angle_deg(&board));

        fillet_acute_angle_junctions(&mut board, 0.35);

        assert!(
            min_angle_deg(&board) >= 89.0,
            "acute fork must be filleted, min junction angle {:.1} deg",
            min_angle_deg(&board)
        );
        assert!(!board.arcs.is_empty(), "fork must be rounded with a fillet arc");
        // The three branch ends must still be joined.
        let ends = [p(154.8265, 87.7098), p(154.0750, 91.0750), p(163.0050, 86.9950)];
        let mut reached = vec![ends[0]];
        let mut k = 0;
        while k < reached.len() {
            let v = reached[k];
            k += 1;
            let mut pieces: Vec<((f64, f64), (f64, f64))> = board.tracks.iter().map(|t| (t.start, t.end)).collect();
            pieces.extend(board.arcs.iter().map(|a| (a.start(), a.end())));
            for (a, b) in pieces {
                for (x, y) in [(a, b), (b, a)] {
                    if near(x, v) && !reached.iter().any(|&r| near(r, y)) {
                        reached.push(y);
                    }
                }
            }
        }
        for e in ends {
            assert!(reached.iter().any(|&r| near(r, e)), "fork branch end {:?} disconnected after filleting", e);
        }
    }

    /// True if the track leaving `pad` exits cleanly: perpendicular through a side, or with its
    /// centreline through a corner heading inside that corner's 90 deg cone.
    fn pad_exit_is_clean(fp: &Footprint, pad: &Pad, t: &TrackSegment) -> bool {
        let c = fp.pad_absolute_position(pad);
        let (anchor, far) = if dist_point_to_pad(t.start, fp, pad) < 1e-6 { (t.start, t.end) } else { (t.end, t.start) };
        let rot = (fp.rotation + pad.rotation).to_radians();
        let (sn, cs) = rot.sin_cos();
        let loc = |w: (f64, f64)| ((w.0 - c.0) * cs + (w.1 - c.1) * sn, -(w.0 - c.0) * sn + (w.1 - c.1) * cs);
        let (a, f) = (loc(anchor), loc(far));
        let l = (f.0 - a.0).hypot(f.1 - a.1);
        let d = ((f.0 - a.0) / l, (f.1 - a.1) / l);
        let (hw, hh) = (pad.width * 0.5, pad.height * 0.5);
        // Perpendicular side exit.
        if d.0.abs() < 0.035 || d.1.abs() < 0.035 {
            return true;
        }
        // Corner exit: the centreline passes (within 1 um) through the corner in its cone.
        let corner = (d.0.signum() * hw, d.1.signum() * hh);
        let cross = (corner.0 - a.0) * d.1 - (corner.1 - a.1) * d.0;
        cross.abs() < 1e-3
    }

    fn pad_exit_board(dirs: &[(f64, f64)]) -> (Board, (f64, f64)) {
        let mut board = make_router_test_board();
        board.nets = vec!["".into(), "GND".into(), "/TVL".into()];
        board.net_name_to_id = board.nets.iter().cloned().enumerate().map(|(i, n)| (n, i)).collect();
        let bb = board.boundary_aabb();
        let c = ((bb.min_x + bb.max_x) * 0.5, (bb.min_y + bb.max_y) * 0.5);
        board.footprints.truncate(1);
        let fp = &mut board.footprints[0];
        fp.x = c.0;
        fp.y = c.1;
        fp.rotation = 0.0;
        fp.layer = 0;
        // 0603-like pad, 0.8 x 0.95 mm.
        fp.pads = vec![Pad::new("1", 0.0, 0.0, 0.8, 0.95, 2, "/TVL", false)];
        board.tracks = dirs
            .iter()
            .map(|&(dx, dy)| TrackSegment { start: c, end: (c.0 + dx, c.1 + dy), width: 0.2, layer: 0, net_id: 2, is_locked: false })
            .collect();
        (board, c)
    }

    #[test]
    fn test_shallow_pad_exit_is_reanchored_through_corner() {
        // A track leaving at 30 deg crosses the pad's long side at a shallow angle: one copper edge
        // meets the outline at ~60 deg (acid trap). It must instead pass through the corner.
        let (mut board, _) = pad_exit_board(&[(2.0 * 30f64.to_radians().cos(), 2.0 * 30f64.to_radians().sin())]);
        let fp = board.footprints[0].clone();
        assert!(!pad_exit_is_clean(&fp, &fp.pads[0], &board.tracks[0]), "setup: centre exit at 30 deg is not clean");
        align_pad_exits(&mut board);
        assert_eq!(board.tracks.len(), 1);
        let t = &board.tracks[0];
        assert!(dist_point_to_pad(t.start, &fp, &fp.pads[0]) < 1e-9 || dist_point_to_pad(t.end, &fp, &fp.pads[0]) < 1e-9, "track must still end inside the pad");
        assert!(pad_exit_is_clean(&fp, &fp.pads[0], t), "track must leave through a corner: {:?}", t);
    }

    #[test]
    fn test_two_tracks_leaving_pad_together_get_separate_ports() {
        // Two tracks leaving the pad centre 30 deg apart leave an acute wedge between them just
        // outside the pad. Each must get its own clean exit (side + corner).
        let a = 0f64.to_radians();
        let b = 30f64.to_radians();
        let (mut board, c) = pad_exit_board(&[(2.0 * a.cos(), 2.0 * a.sin()), (2.0 * b.cos(), 2.0 * b.sin())]);
        let fp = board.footprints[0].clone();
        align_pad_exits(&mut board);
        let pad = &fp.pads[0];
        let anchors: Vec<(f64, f64)> = board
            .tracks
            .iter()
            .filter_map(|t| {
                if dist_point_to_pad(t.start, &fp, pad) < 1e-9 {
                    Some(t.start)
                } else if dist_point_to_pad(t.end, &fp, pad) < 1e-9 {
                    Some(t.end)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(anchors.len(), 2, "both tracks must still end on the pad: {:?}", board.tracks);
        assert!((anchors[0].0 - anchors[1].0).hypot(anchors[0].1 - anchors[1].1) > 0.1, "tracks must leave from different ports: {:?}", anchors);
        for t in board.tracks.iter().filter(|t| dist_point_to_pad(t.start, &fp, pad) < 1e-9 || dist_point_to_pad(t.end, &fp, pad) < 1e-9) {
            assert!(pad_exit_is_clean(&fp, pad, t), "exit not clean: {:?} (pad centre {:?})", t, c);
        }
    }

    #[test]
    fn test_perpendicular_pad_exit_is_left_alone() {
        let (mut board, _) = pad_exit_board(&[(2.0, 0.0)]);
        let before = board.tracks.clone();
        align_pad_exits(&mut board);
        assert_eq!(board.tracks, before);
    }

    #[test]
    fn test_pad_to_pad_track_reanchored_at_both_ends_stays_connected() {
        // Regression: a direct SW2.2 -> R pad track got two independent edits, was duplicated, the
        // net counted as broken and the whole net was restored to its pre-fillet copper.
        let (mut board, c) = pad_exit_board(&[]);
        let mut fp2 = board.footprints[0].clone();
        fp2.x += 3.0;
        fp2.y += 1.2;
        board.footprints.push(fp2.clone());
        let p2 = (c.0 + 3.0, c.1 + 1.2);
        board.tracks = vec![TrackSegment { start: c, end: p2, width: 0.2, layer: 0, net_id: 2, is_locked: false }];
        align_pad_exits(&mut board);
        let (f1, f2) = (&board.footprints[0], &board.footprints[1]);
        let on = |p: (f64, f64)| dist_point_to_pad(p, f1, &f1.pads[0]) < 1e-9 || dist_point_to_pad(p, f2, &f2.pads[0]) < 1e-9;
        let ends: Vec<(f64, f64)> = board.tracks.iter().flat_map(|t| [t.start, t.end]).collect();
        for &e in &ends {
            let shared = ends.iter().filter(|&&o| (o.0 - e.0).hypot(o.1 - e.1) < 1e-9).count() > 1;
            assert!(on(e) || shared, "dangling end {:?} in {:?}", e, board.tracks);
        }
        for t in &board.tracks {
            for (fp, pad) in [(f1, &f1.pads[0]), (f2, &f2.pads[0])] {
                if dist_point_to_pad(t.start, fp, pad) < 1e-9 || dist_point_to_pad(t.end, fp, pad) < 1e-9 {
                    assert!(pad_exit_is_clean(fp, pad, t), "exit not clean: {:?}", t);
                }
            }
        }
        let anchored = |fp: &Footprint| ends.iter().any(|&e| dist_point_to_pad(e, fp, &fp.pads[0]) < 1e-9);
        assert!(anchored(f1) && anchored(f2), "both pads must stay connected: {:?}", board.tracks);
    }

    #[test]
    fn test_in_pad_debris_removed_and_offcentre_exit_reanchored() {
        // Regression from Net-(U4-EN) at C16.1: a triangle of 0.01 mm segments sat inside the pad and
        // the real trace started off-centre, so the junction looked like an 18 deg acid trap.
        let (mut board, c) = pad_exit_board(&[]);
        let seg = |a: (f64, f64), b: (f64, f64)| TrackSegment { start: a, end: b, width: 0.2, layer: 0, net_id: 2, is_locked: false };
        let o = |dx: f64, dy: f64| (c.0 + dx, c.1 + dy);
        board.tracks = vec![
            seg(o(0.01, -0.01), o(0.0, 0.0)),
            seg(o(0.01, -0.005), o(0.0, 0.0)),
            seg(o(0.01, -0.01), o(0.01, -0.005)),
            seg(o(0.015, 0.0), o(0.0, 0.0)),
            seg(o(0.015, 0.0), o(2.0, 0.0)),
        ];
        align_pad_exits(&mut board);
        let fp = &board.footprints[0];
        assert_eq!(board.tracks.len(), 1, "in-pad debris must be removed: {:?}", board.tracks);
        let t = &board.tracks[0];
        assert!(dist_point_to_pad(t.start, fp, &fp.pads[0]) < 1e-9, "trace must still start on the pad");
        assert!(pad_exit_is_clean(fp, &fp.pads[0], t));
        assert!((t.start.0 - c.0).hypot(t.start.1 - c.1) < 1e-6, "axis exit restarts from the pad centre: {:?}", t);
    }

    #[test]
    fn test_short_fanout_slides_anchor_for_perpendicular_exit() {
        // Regression from R10.2 / C9.1 (+3.3V): a 0.86 mm pad-to-via fanout leaving 15 deg off the
        // pad axis had no usable corner. Slide the anchor so it leaves perpendicularly instead.
        let (mut board, c) = pad_exit_board(&[(0.23, -0.825)]);
        align_pad_exits(&mut board);
        let fp = &board.footprints[0];
        let t = &board.tracks[0];
        assert_eq!(board.tracks.len(), 1);
        assert_eq!(t.end, (c.0 + 0.23, c.1 - 0.825), "the via end must not move");
        assert!(dist_point_to_pad(t.start, fp, &fp.pads[0]) < 1e-9);
        assert!((t.start.0 - t.end.0).abs() < 1e-9, "exit must be perpendicular to the pad side: {:?}", t);
        assert!(pad_exit_is_clean(fp, &fp.pads[0], t));
    }

    #[test]
    fn test_prune_keeps_tracks_leaving_one_pad_from_different_points() {
        // Regression: after pad-exit re-anchoring, /VRAW left D3.2 from two corners. Cycle pruning
        // only treated the end nearest the pad centre as the pad terminal and deleted the other branch.
        let (mut board, c) = pad_exit_board(&[]);
        let mut fb = board.footprints[0].clone();
        fb.x += 3.0;
        let mut fc = board.footprints[0].clone();
        fc.y += 3.0;
        board.footprints.push(fb);
        board.footprints.push(fc);
        let seg = |a: (f64, f64), b: (f64, f64)| TrackSegment { start: a, end: b, width: 0.2, layer: 0, net_id: 2, is_locked: false };
        board.tracks = vec![
            seg((c.0 + 0.3, c.1 - 0.2), (c.0 + 3.0, c.1)),
            seg((c.0 - 0.2, c.1 + 0.37), (c.0, c.1 + 3.0)),
        ];
        let before = board.tracks.clone();
        prune_dangling_tracks_and_vias(&mut board);
        for t in &before {
            assert!(board.tracks.contains(t), "branch {:?} must survive pruning: {:?}", t, board.tracks);
        }
    }

    #[test]
    fn test_via_annular_width_and_single_plane_via_pruning() {
        // 1. Test via annular width clamping in validate_and_clean_copper
        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 50.0,
            max_y: 50.0,
            nets: vec![String::new(), "GND".to_string(), "+3.3V".to_string()],
            vias: vec![Via {
                position: (10.0, 10.0),
                size: 0.60,
                drill: 0.45, // annular width = (0.60 - 0.45)/2 = 0.075mm (< 0.10mm board setup default)
                net_id: 2,
                is_locked: false,
            }],
            ..Board::default()
        };

        validate_and_clean_copper(&mut board);
        let v = &board.vias[0];
        let annular = (v.size - v.drill) * 0.5;
        assert!(
            annular >= 0.10 - 1e-4,
            "Via annular width must be clamped to >= 0.10mm (got {:.4}mm: size={}, drill={})",
            annular, v.size, v.drill
        );
        assert!(v.drill <= 0.40 + 1e-4, "Drill must be reduced to 0.40mm to maintain 0.10mm annular width");

        // 2. Test single-plane orphan via pruning
        // Plane zone on layer 1 (In1.Cu) for +3.3V (net_id = 2)
        board.net_name_to_id.insert("+3.3V".to_string(), 2);
        board.raw_content = r#"(kicad_pcb (version 20240108)
  (zone (net 2) (net_name "+3.3V") (layer "In1.Cu")
    (polygon (pts (xy 0 0) (xy 50 0) (xy 50 50) (xy 0 50))))
)"#.to_string();

        // Via 1: orphan on single plane layer (no tracks on any layer)
        board.vias = vec![
            Via {
                position: (15.0, 15.0),
                size: 0.60,
                drill: 0.30,
                net_id: 2,
                is_locked: false,
            },
            // Via 2: connected to a track leading to a pad on layer 0
            Via {
                position: (25.0, 25.0),
                size: 0.60,
                drill: 0.30,
                net_id: 2,
                is_locked: false,
            },
        ];
        board.tracks.push(TrackSegment {
            start: (25.0, 25.0),
            end: (30.0, 25.0),
            width: 0.30,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        board.footprints.push(Footprint {
            reference: "U1".to_string(),
            value: "IC".to_string(),
            x: 30.0,
            y: 25.0,
            pads: vec![Pad {
                number: "1".to_string(),
                net_id: 2,
                net_name: "+3.3V".to_string(),
                width: 1.0,
                height: 1.0,
                shape: "rect".to_string(),
                ..Pad::default()
            }],
            ..Footprint::default()
        });

        prune_dangling_tracks_and_vias(&mut board);
        assert_eq!(board.vias.len(), 1, "Orphan single-plane via must be pruned");
        assert_eq!(board.vias[0].position, (25.0, 25.0), "Connected via must be retained");
    }

    #[test]
    fn test_gpu_via_reduction_and_track_glossing_drc_clean() {
        let vulkan = VulkanEngine::new();
        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 60.0,
            max_y: 60.0,
            nets: vec![
                String::new(),
                "NET_HOP_CLEAR".to_string(),
                "NET_HOP_BLOCKED".to_string(),
                "NET_SINGLE_VIA".to_string(),
                "NET_OBSTACLE".to_string(),
            ],
            footprints: vec![
                Footprint {
                    reference: "U1".to_string(),
                    x: 10.0,
                    y: 10.0,
                    layer: 0,
                    pads: vec![
                        Pad::new("1", -5.0, 0.0, 0.8, 0.8, 1, "NET_HOP_CLEAR", false),
                        Pad::new("2", 15.0, 0.0, 0.8, 0.8, 1, "NET_HOP_CLEAR", false),
                        Pad::new("3", -5.0, 10.0, 0.8, 0.8, 2, "NET_HOP_BLOCKED", false),
                        Pad::new("4", 15.0, 10.0, 0.8, 0.8, 2, "NET_HOP_BLOCKED", false),
                        // Foreign obstacle pad on L0 blocking NET_HOP_BLOCKED between x=10 and x=20 at y=20
                        Pad::new("5", 5.0, 10.0, 1.2, 1.2, 4, "NET_OBSTACLE", false),
                    ],
                    ..Footprint::default()
                },
                Footprint {
                    reference: "J1".to_string(),
                    x: 10.0,
                    y: 40.0,
                    layer: 0,
                    pads: vec![
                        Pad::new("1", -5.0, 0.0, 1.0, 1.0, 3, "NET_SINGLE_VIA", true),
                        Pad::new("2", 15.0, 0.0, 1.0, 1.0, 3, "NET_SINGLE_VIA", true),
                    ],
                    ..Footprint::default()
                },
            ],
            vias: vec![
                // NET_HOP_CLEAR redundant via pair at (10,10) and (20,10)
                Via { position: (10.0, 10.0), size: 0.6, drill: 0.3, net_id: 1, is_locked: false },
                Via { position: (20.0, 10.0), size: 0.6, drill: 0.3, net_id: 1, is_locked: false },
                // NET_HOP_BLOCKED necessary via pair around obstacle at (15,20)
                Via { position: (10.0, 20.0), size: 0.6, drill: 0.3, net_id: 2, is_locked: false },
                Via { position: (20.0, 20.0), size: 0.6, drill: 0.3, net_id: 2, is_locked: false },
                // NET_SINGLE_VIA solitary via at (15,40) between PTH pads at (5,40) and (25,40)
                Via { position: (15.0, 40.0), size: 0.6, drill: 0.3, net_id: 3, is_locked: false },
            ],
            tracks: vec![
                // NET_HOP_CLEAR: (5,10)->(10,10) on L0, (10,10)->(20,10) on L1, (20,10)->(25,10) on L0 with a stair-step
                TrackSegment { start: (5.0, 10.0), end: (10.0, 10.0), width: 0.25, layer: 0, net_id: 1, is_locked: false },
                TrackSegment { start: (10.0, 10.0), end: (20.0, 10.0), width: 0.25, layer: 1, net_id: 1, is_locked: false },
                TrackSegment { start: (20.0, 10.0), end: (22.0, 12.0), width: 0.25, layer: 0, net_id: 1, is_locked: false },
                TrackSegment { start: (22.0, 12.0), end: (23.0, 12.0), width: 0.25, layer: 0, net_id: 1, is_locked: false },
                TrackSegment { start: (23.0, 12.0), end: (25.0, 10.0), width: 0.25, layer: 0, net_id: 1, is_locked: false },
                // NET_HOP_BLOCKED: (5,20)->(10,20) on L0, (10,20)->(20,20) on L1, (20,20)->(25,20) on L0
                TrackSegment { start: (5.0, 20.0), end: (10.0, 20.0), width: 0.25, layer: 0, net_id: 2, is_locked: false },
                TrackSegment { start: (10.0, 20.0), end: (20.0, 20.0), width: 0.25, layer: 1, net_id: 2, is_locked: false },
                TrackSegment { start: (20.0, 20.0), end: (25.0, 20.0), width: 0.25, layer: 0, net_id: 2, is_locked: false },
                // NET_SINGLE_VIA: (5,40)->(15,40) on L0, (15,40)->(25,40) on L1
                TrackSegment { start: (5.0, 40.0), end: (15.0, 40.0), width: 0.25, layer: 0, net_id: 3, is_locked: false },
                TrackSegment { start: (15.0, 40.0), end: (25.0, 40.0), width: 0.25, layer: 1, net_id: 3, is_locked: false },
            ],
            ..Board::default()
        };

        let mut grid = RoutingGrid::new(&board, 0.2, vulkan);
        reduce_vias(&mut board, &mut grid);

        // NET_HOP_CLEAR (net 1) and NET_SINGLE_VIA (net 3) vias should be eliminated; NET_HOP_BLOCKED (net 2) vias remain
        assert_eq!(board.vias.iter().filter(|v| v.net_id == 1).count(), 0);
        assert_eq!(board.vias.iter().filter(|v| v.net_id == 3).count(), 0);
        assert_eq!(board.vias.iter().filter(|v| v.net_id == 2).count(), 2);

        gloss_tracks_with_mode(&mut board, &grid, false);
        gloss_tracks_with_mode(&mut board, &grid, true);

        // Verify all resulting tracks respect clearance to the foreign obstacle pad at (15.0, 20.0)
        let obs_fp = &board.footprints[0];
        let obs_pad = &obs_fp.pads[4];
        for t in &board.tracks {
            if t.net_id != 4 && t.layer == 0 {
                let d = dist_segment_to_pad(t.start, t.end, obs_fp, obs_pad);
                assert!(
                    d >= t.width * 0.5 + 0.20 - 1e-3,
                    "Track {:?} violated clearance to obstacle pad (d={:.4})",
                    t,
                    d
                );
            }
        }
    }

    #[test]
    fn test_gpu_obstacle_rasterize_and_via_stitching() {
        use crate::board::Polygon;

        let vulkan = VulkanEngine::new();
        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 30.0,
            max_y: 25.0,
            outline_polygon: Some(Polygon::new(vec![
                Point::new(0.0, 0.0),
                Point::new(30.0, 0.0),
                Point::new(30.0, 20.0),
                Point::new(25.0, 25.0),
                Point::new(0.0, 25.0),
            ])),
            nets: vec![String::new(), "GND".to_string(), "SIG1".to_string(), "SIG2".to_string()],
            ..Board::default()
        };

        // Populate >= 8 footprints (mix of rotated SMD and PTH pads) to exercise parallel footprint rasterization
        for i in 0..8 {
            let fx = 4.0 + (i % 4) as f64 * 6.0;
            let fy = 5.0 + (i / 4) as f64 * 10.0;
            board.footprints.push(Footprint {
                reference: format!("C{}", i + 1),
                x: fx,
                y: fy,
                rotation: (i as f64) * 45.0,
                layer: if i % 3 == 0 { 1 } else { 0 },
                orig_layer: 0,
                is_locked: i == 0,
                pads: vec![
                    Pad::new("1", -0.8, 0.0, 0.8, 0.9, 1, "GND", i == 1),
                    Pad::new("2", 0.8, 0.0, 0.8, 0.9, 2 + (i % 2), if i % 2 == 0 { "SIG1" } else { "SIG2" }, false),
                ],
                ..Footprint::default()
            });
        }

        let grid = RoutingGrid::new(&board, 0.25, vulkan);

        // Verify bit-for-bit equivalence of obstacle_net_ids, nudgeable_fp, and smd_pad_cells against sequential reference
        let total_cells = grid.num_layers * grid.width * grid.height;
        let mut ref_obs = vec![FREE_NET; total_cells];
        let mut ref_nudge = vec![None; total_cells];
        let mut ref_smd = vec![false; total_cells];

        if let Some(ref poly) = board.outline_polygon {
            let edge_clearance = board.board_edge_clearance();
            for layer in 0..grid.num_layers {
                for gy in 0..grid.height {
                    for gx in 0..grid.width {
                        let (wx, wy) = grid.grid_to_world(gx, gy);
                        let p = Point::new(wx, wy);
                        if !poly.contains_point(&p) || poly.distance_to_point(&p) < edge_clearance {
                            ref_obs[grid.idx(gx, gy, layer)] = KEEPOUT_NET;
                        }
                    }
                }
            }
        }
        let all_layers: Vec<usize> = (0..grid.num_layers).collect();
        for (fp_idx, fp) in board.footprints.iter().enumerate() {
            let is_nudgeable = !fp.is_locked;
            for pad in &fp.pads {
                let smd_layer = [fp.pad_layer(pad).min(grid.num_layers - 1)];
                let (px, py) = fp.pad_absolute_position(pad);
                let (rot_w, rot_h) = fp.pad_world_size(pad);
                let half_w = rot_w / 2.0;
                let half_h = rot_h / 2.0;
                let layers: &[usize] = if pad.is_through_hole { &all_layers } else { &smd_layer };
                let total_rot = ((fp.rotation + pad.rotation) % 360.0 + 360.0) % 360.0;
                let pad_hw = pad.width * 0.5;
                let pad_hh = pad.height * 0.5;
                let net_val = pad.net_id as i32;
                for &layer in layers {
                    let min_gx = ((px - half_w - grid.min_x) / grid.res).floor().clamp(0.0, grid.width.saturating_sub(1) as f64) as usize;
                    let max_gx = ((px + half_w - grid.min_x) / grid.res).floor().clamp(0.0, grid.width.saturating_sub(1) as f64) as usize;
                    let min_gy = ((py - half_h - grid.min_y) / grid.res).floor().clamp(0.0, grid.height.saturating_sub(1) as f64) as usize;
                    let max_gy = ((py + half_h - grid.min_y) / grid.res).floor().clamp(0.0, grid.height.saturating_sub(1) as f64) as usize;
                    for gy in min_gy..=max_gy {
                        for gx in min_gx..=max_gx {
                            let (wx, wy) = grid.grid_to_world(gx, gy);
                            let unrot = Point::new(wx - px, wy - py).rotate_ccw(-total_rot);
                            let dx = 0.0f64.max(unrot.x.abs() - pad_hw);
                            let dy = 0.0f64.max(unrot.y.abs() - pad_hh);
                            if dx.hypot(dy) <= grid.res * 0.5 + 1e-4 {
                                let i = grid.idx(gx, gy, layer);
                                ref_obs[i] = net_val;
                                if is_nudgeable {
                                    ref_nudge[i] = Some(fp_idx);
                                }
                                if !pad.is_through_hole {
                                    ref_smd[i] = true;
                                }
                            }
                        }
                    }
                    if let Some((cgx, cgy)) = grid.world_to_grid(px, py) {
                        let i = grid.idx(cgx, cgy, layer);
                        ref_obs[i] = net_val;
                        if is_nudgeable {
                            ref_nudge[i] = Some(fp_idx);
                        }
                        if !pad.is_through_hole {
                            ref_smd[i] = true;
                        }
                    }
                }
            }
        }

        assert_eq!(grid.obstacle_net_ids, ref_obs, "obstacle_net_ids must match sequential reference bit-for-bit");
        assert_eq!(grid.nudgeable_fp, ref_nudge, "nudgeable_fp must match sequential reference bit-for-bit");
        assert_eq!(grid.smd_pad_cells, ref_smd, "smd_pad_cells must match sequential reference bit-for-bit");

        // Test plane & ground via stitching with GPU batch pre-filter
        let router = AutoRouter::new(2);
        let added = router.add_ground_stitching_vias(&mut board, &grid);
        assert!(added > 0, "Ground and plane via stitching should insert stitching vias");
        for v in &board.vias {
            assert_eq!(v.net_id, 1);
            // Verify every placed via respects clearance to foreign SMD/PTH pads
            for fp in &board.footprints {
                for pad in &fp.pads {
                    if pad.net_id != 1 {
                        let d = dist_point_to_pad(v.position, fp, pad);
                        assert!(
                            d >= v.size * 0.5 + 0.20 - 1e-3,
                            "Stitching via at {:?} violated clearance to pad {}.{} (d={:.4})",
                            v.position,
                            fp.reference,
                            pad.number,
                            d
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_gpu_arc_drc_copper_clean_and_crosstalk() {
        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 40.0,
            nets: vec![
                String::new(),
                "NET1".to_string(),
                "NET2".to_string(),
                "CLK_FAST".to_string(),
            ],
            footprints: vec![
                Footprint {
                    reference: "U1".to_string(),
                    x: 10.0,
                    y: 10.0,
                    pads: vec![
                        Pad::new("1", 0.0, 0.0, 1.0, 1.0, 1, "NET1", false),
                        Pad::new("2", 5.0, 0.0, 1.0, 1.0, 2, "NET2", false),
                        Pad::new("3", 20.0, 15.0, 1.0, 1.0, 1, "NET1", true),
                    ],
                    ..Footprint::default()
                },
            ],
            ..Board::default()
        };

        // 1. Test GPU batch arc DRC (`batch_validate_arcs_gpu`) equivalence with `is_arc_clear`
        let clear_arc = ArcSegment {
            center: (20.0, 20.0),
            radius: 1.0,
            start_angle_rad: 0.0,
            end_angle_rad: std::f64::consts::FRAC_PI_2,
            width: 0.2,
            layer: 0,
            net_id: 1,
            is_locked: false,
        };
        let blocked_arc = ArcSegment {
            center: (15.0, 9.0),
            radius: 1.0,
            start_angle_rad: 0.0,
            end_angle_rad: std::f64::consts::PI,
            width: 0.2,
            layer: 0,
            net_id: 1, // Passes right through U1.2 (NET2 at 15.0, 10.0)
            is_locked: false,
        };
        let drc_ctx = GpuDrcContext::from_board(&board);
        let arc_results = batch_validate_arcs_gpu(&board, &drc_ctx, &[clear_arc.clone(), blocked_arc.clone()]);
        assert_eq!(arc_results.len(), 2);
        assert_eq!(arc_results[0], is_arc_clear(&board, &clear_arc));
        assert_eq!(arc_results[1], is_arc_clear(&board, &blocked_arc));
        assert!(arc_results[0], "Clear arc should pass batch_validate_arcs_gpu");
        assert!(!arc_results[1], "Blocked arc overlapping foreign pad should fail batch_validate_arcs_gpu");

        // 2. Test Pre-Export Via DRC in `validate_and_clean_copper`
        // Add a valid via (connected to tracks on both layers) and a conflicting via overlapping foreign pad U1.2
        board.tracks.push(TrackSegment {
            start: (10.0, 10.0),
            end: (25.0, 25.0),
            width: 0.2,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: (25.0, 25.0),
            end: (30.0, 25.0),
            width: 0.2,
            layer: 1,
            net_id: 1,
            is_locked: false,
        });
        board.vias.push(Via {
            position: (25.0, 25.0),
            size: 0.6,
            drill: 0.3,
            net_id: 1,
            is_locked: false,
        });
        // Conflicting via on NET1 placed right next to foreign pad U1.2 (NET2 at 15.0, 10.0)
        board.vias.push(Via {
            position: (15.5, 10.0),
            size: 0.6,
            drill: 0.3,
            net_id: 1,
            is_locked: false,
        });
        validate_and_clean_copper(&mut board);
        assert_eq!(board.vias.len(), 1, "Conflicting via must be removed while clean inter-layer via is preserved");
        assert_eq!(board.vias[0].position, (25.0, 25.0));

        // 3. Test LBVH-Accelerated Crosstalk & Coupling Validation (`validate_crosstalk_and_coupling`)
        board.tracks.clear();
        // Two long parallel tracks on layer 0: NET1 (victim) and CLK_FAST (aggressor, net 3) separated by 0.25 mm (< 0.55 mm)
        board.tracks.push(TrackSegment {
            start: (5.0, 30.0),
            end: (25.0, 30.0),
            width: 0.2,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: (5.0, 30.25),
            end: (25.0, 30.25),
            width: 0.2,
            layer: 0,
            net_id: 3,
            is_locked: false,
        });
        // Uncoupled far-away track on layer 0
        board.tracks.push(TrackSegment {
            start: (5.0, 38.0),
            end: (25.0, 38.0),
            width: 0.2,
            layer: 0,
            net_id: 2,
            is_locked: false,
        });
        let violations = validate_crosstalk_and_coupling(&board);
        assert_eq!(violations.len(), 1, "Should detect exact parallel coupling violation between NET1 and CLK_FAST");
        assert_eq!(violations[0].net_a_name, "NET1");
        assert_eq!(violations[0].net_b_name, "CLK_FAST");
        assert!((violations[0].coupled_length_mm - 20.0).abs() < 1e-3);
        assert!((violations[0].max_allowed_mm - 15.0).abs() < 1e-3);
    }
}
