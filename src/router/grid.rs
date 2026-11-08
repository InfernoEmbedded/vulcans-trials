// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! 3D voxelized copper obstacle grid, Euclidean Distance Field (EDF) transforms,
//! and fundamental 2D octilinear segment geometry primitives.

use super::*;

/// Discrete 3D coordinate on the [`RoutingGrid`] representing a cell `(gx, gy)` on a specific copper `layer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridNode {
    /// Horizontal grid column index (`0..width`).
    pub gx: usize,
    /// Vertical grid row index (`0..height`).
    pub gy: usize,
    /// Copper layer index (`0` = `F.Cu`, `1` = `B.Cu`, `2+` = inner layers `In1.Cu`, etc.).
    pub layer: usize,
}

/// 3D uniform spatial grid storing rasterized copper obstacles, Euclidean distance fields,
/// PathFinder congestion history, and precomputed GPU routing cost maps across all copper layers.
pub struct RoutingGrid {
    /// Minimum world X coordinate (in `mm`) of the grid bounding box.
    pub min_x: f64,
    /// Minimum world Y coordinate (in `mm`) of the grid bounding box.
    pub min_y: f64,
    /// Maximum world X coordinate (in `mm`) of the grid bounding box.
    pub max_x: f64,
    /// Maximum world Y coordinate (in `mm`) of the grid bounding box.
    pub max_y: f64,
    /// Spatial size of a single grid cell (in `mm`).
    pub res: f64,
    /// Number of grid columns along the X axis.
    pub width: usize,
    /// Number of grid rows along the Y axis.
    pub height: usize,
    /// Total number of copper layers represented in the 3D grid (`>= 2`).
    pub num_layers: usize,
    /// Mapping from inner copper layer index to dedicated power/ground plane net ID.
    pub plane_nets: HashMap<usize, usize>,
    /// Flat 3D array (`num_layers * width * height`) of net IDs occupying each cell
    /// ([`FREE_NET`] = `0`, [`KEEPOUT_NET`] = `-1`, or positive net ID).
    pub obstacle_net_ids: Vec<i32>,
    /// Flat 3D array (`num_layers * width * height`) of Euclidean distances (in `mm`) to the nearest obstacle on the same layer.
    pub dist_field: Vec<f32>,
    /// Flat 3D array (`num_layers * width * height`) of accumulated PathFinder historical congestion penalties.
    pub congestion: Vec<f32>,
    /// Flat 3D array (`num_layers * width * height`) of present-iteration net occupancy counts per cell.
    pub usage_count: Vec<u32>,
    /// Per-net cached 3D GPU routing cost maps keyed by `net_id`, storing `(clearance_cells, cost_buffer)`.
    pub gpu_net_cost_maps: HashMap<usize, (i32, Vec<f32>)>,
    /// Flat 3D array mapping each grid cell to the unlocked footprint index (`Some(fp_idx)`) that owns the pad occupying it.
    pub nudgeable_fp: Vec<Option<usize>>,
    /// Flat 3D boolean mask indicating whether each cell is covered by an SMD pad (used to prevent illegal via-in-pad collisions).
    pub smd_pad_cells: Vec<bool>,
    /// Per-net copper clearance requirements (in `mm`), indexed by `net_id`.
    pub net_clearances: Vec<f64>,
    /// Maximum netclass clearance (in `mm`) across all nets on the board.
    pub max_net_clearance: f64,
    /// Additional per-millimeter cost penalty applied when routing signal traces on inner layers (`layer >= 2`).
    pub inner_layer_penalty: f64,
    /// Whether vias are permitted inside same-net SMD pads.
    pub allow_via_in_pad: bool,
    /// Optional shared handle to the [`VulkanEngine`] used for GPU distance-field and wavefront compute dispatches.
    pub vulkan: Option<Arc<VulkanEngine>>,
    /// Cached snapshot of the base static obstacle state `(obstacle_net_ids, usage_count, nudgeable_fp, smd_pad_cells)`
    /// prior to dynamic track commitments, enabling fast rip-up and re-route resets.
    pub base_snapshot: Option<Arc<(Vec<i32>, Vec<u32>, Vec<Option<usize>>, Vec<bool>)>>,
}

/// Checks whether a 2D line segment `p1 -> p2` conforms to the standard octilinear 45-degree grid
/// (`0°`, `45°`, `90°`, `135°`, etc.) within `1e-4 mm` tolerance.
///
/// # Arguments
/// * `p1` - Start point `(x, y)` in millimeters (`mm`).
/// * `p2` - End point `(x, y)` in millimeters (`mm`).
///
/// # Returns
/// `true` if the segment is horizontal, vertical, or 45° diagonal within `1e-4 mm`.
#[inline]
pub fn is_octilinear_segment(p1: (f64, f64), p2: (f64, f64)) -> bool {
    let dx = (p2.0 - p1.0).abs();
    let dy = (p2.1 - p1.1).abs();
    dx < 1e-4 || dy < 1e-4 || (dx - dy).abs() < 1e-4
}

/// Computes a KiCad-standard 45° octilinear dogleg polyline connecting `p0` to `p1`.
///
/// When `diagonal_first` is `false`, emits the axial (horizontal or vertical) segment first followed
/// by a 45° diagonal segment into `p1`. When `diagonal_first` is `true`, emits the 45° diagonal
/// segment first followed by the axial segment into `p1`.
///
/// # Arguments
/// * `p0` - Start world coordinates `(x, y)` in millimeters (`mm`).
/// * `p1` - End world coordinates `(x, y)` in millimeters (`mm`).
/// * `diagonal_first` - Whether to place the 45° diagonal leg at `p0` (`true`) or at `p1` (`false`).
///
/// # Returns
/// A vector of 1, 2, or 3 vertices `[(x, y)]` forming the octilinear connection from `p0` to `p1`.
pub fn compute_45_dogleg(p0: (f64, f64), p1: (f64, f64), diagonal_first: bool) -> Vec<(f64, f64)> {
    let dx = p1.0 - p0.0;
    let dy = p1.1 - p0.1;
    let abs_dx = dx.abs();
    let abs_dy = dy.abs();

    if abs_dx < 1e-6 && abs_dy < 1e-6 {
        return vec![p0];
    }

    let sign_x = if dx >= 0.0 { 1.0 } else { -1.0 };
    let sign_y = if dy >= 0.0 { 1.0 } else { -1.0 };

    let corner = if abs_dx > abs_dy {
        // Dominant direction: horizontal
        if diagonal_first {
            (p0.0 + sign_x * abs_dy, p1.1)
        } else {
            (p1.0 - sign_x * abs_dy, p0.1)
        }
    } else {
        // Dominant direction: vertical (or equal)
        if diagonal_first {
            (p1.0, p0.1 + sign_y * abs_dx)
        } else {
            (p0.0, p1.1 - sign_y * abs_dx)
        }
    };

    let dist_to_p0 = (corner.0 - p0.0).hypot(corner.1 - p0.1);
    let dist_to_p1 = (corner.0 - p1.0).hypot(corner.1 - p1.1);

    if dist_to_p0 < 1e-6 || dist_to_p1 < 1e-6 {
        vec![p0, p1]
    } else {
        vec![p0, corner, p1]
    }
}

impl RoutingGrid {
    /// Constructs a new 3D [`RoutingGrid`] from the given [`Board`] at spatial resolution `res` (`mm`),
    /// rasterizes all board outlines, keepouts, pads, and locked copper, and computes the initial
    /// 2D Euclidean distance fields across all layers.
    ///
    /// # Arguments
    /// * `board` - The [`Board`] whose geometry, netclasses, and obstacles are rasterized.
    /// * `res` - Grid cell size in millimeters (`mm`).
    /// * `vulkan` - Optional shared [`VulkanEngine`] for GPU-accelerated distance field and cost map generation.
    ///
    /// # Returns
    /// Fully initialized [`RoutingGrid`] ready for pathfinding.
    pub fn new(board: &Board, res: f64, vulkan: Option<Arc<VulkanEngine>>) -> Self {
        let min_x = board.min_x;
        let min_y = board.min_y;
        let max_x = board.max_x;
        let max_y = board.max_y;
        let width = ((max_x - min_x) / res).ceil() as usize;
        let height = ((max_y - min_y) / res).ceil() as usize;
        let num_layers = board.copper_layers().len().max(2);
        let plane_nets = board.inner_plane_nets();
        let total_cells = num_layers * width * height;

        let mut net_clearances = vec![0.20; board.nets.len()];
        let mut max_net_clearance = 0.25_f64;
        for i in 0..board.nets.len() {
            let c = board.get_netclass_for_net(i).clearance;
            net_clearances[i] = c;
            max_net_clearance = max_net_clearance.max(c);
        }

        let mut grid = Self {
            min_x,
            min_y,
            max_x,
            max_y,
            res,
            width,
            height,
            num_layers,
            plane_nets,
            obstacle_net_ids: vec![FREE_NET; total_cells],
            dist_field: vec![999.0; total_cells],
            congestion: vec![0.0; total_cells],
            usage_count: vec![0; total_cells],
            gpu_net_cost_maps: HashMap::new(),
            nudgeable_fp: vec![None; total_cells],
            smd_pad_cells: vec![false; total_cells],
            net_clearances,
            max_net_clearance,
            inner_layer_penalty: 8.0,
            allow_via_in_pad: board.allow_via_in_pad(),
            vulkan,
            base_snapshot: None,
        };

        grid.rasterize_obstacles(board);
        grid.update_distance_fields();
        grid
    }

    /// Builder-style setter that overrides the inner-layer routing penalty (`inner_layer_penalty`).
    ///
    /// # Arguments
    /// * `pen` - Cost penalty per millimeter for routing on inner copper layers (`layer >= 2`).
    ///
    /// # Returns
    /// Updated [`RoutingGrid`] with the specified inner-layer penalty.
    pub fn with_inner_layer_penalty(mut self, pen: f64) -> Self {
        self.inner_layer_penalty = pen;
        self
    }

    /// Builder-style setter that configures whether vias are allowed inside same-net SMD pads.
    ///
    /// # Arguments
    /// * `allow` - `true` to permit via-in-pad on matching nets, `false` to forbid it.
    ///
    /// # Returns
    /// Updated [`RoutingGrid`] with `allow_via_in_pad` set.
    pub fn with_allow_via_in_pad(mut self, allow: bool) -> Self {
        self.allow_via_in_pad = allow;
        self
    }

    /// Computes the flat 1D slice index into [`RoutingGrid::obstacle_net_ids`] and related 3D buffers
    /// for grid coordinates `(gx, gy)` on `layer`.
    ///
    /// # Arguments
    /// * `gx` - Grid column index (`0..width`).
    /// * `gy` - Grid row index (`0..height`).
    /// * `layer` - Copper layer index (`0..num_layers`).
    ///
    /// # Returns
    /// Flat linear index `layer * (width * height) + gy * width + gx`.
    #[inline]
    pub fn idx(&self, gx: usize, gy: usize, layer: usize) -> usize {
        layer * (self.width * self.height) + gy * self.width + gx
    }

    /// Converts continuous world coordinates `(x, y)` in millimeters (`mm`) to discrete grid indices `(gx, gy)`.
    ///
    /// # Arguments
    /// * `x` - World X coordinate in millimeters (`mm`).
    /// * `y` - World Y coordinate in millimeters (`mm`).
    ///
    /// # Returns
    /// `Some((gx, gy))` if `(x, y)` lies within the grid bounds `[min_x..max_x, min_y..max_y)`,
    /// or `None` if out of bounds.
    pub fn world_to_grid(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        if x < self.min_x || x >= self.max_x || y < self.min_y || y >= self.max_y {
            return None;
        }
        let gx = ((x - self.min_x) / self.res) as usize;
        let gy = ((y - self.min_y) / self.res) as usize;
        if gx < self.width && gy < self.height {
            Some((gx, gy))
        } else {
            None
        }
    }

    /// Converts discrete grid cell coordinates `(gx, gy)` to the cell-center world coordinates `(x, y)` in millimeters (`mm`).
    ///
    /// # Arguments
    /// * `gx` - Grid column index.
    /// * `gy` - Grid row index.
    ///
    /// # Returns
    /// Cell center `(x, y)` in millimeters (`mm`).
    pub fn grid_to_world(&self, gx: usize, gy: usize) -> (f64, f64) {
        (
            self.min_x + (gx as f64 + 0.5) * self.res,
            self.min_y + (gy as f64 + 0.5) * self.res,
        )
    }

    /// Rasterizes the board outline edge keepout, explicit keepout zones, component pads
    /// (both SMD and through-hole), and locked tracks/vias into [`RoutingGrid::obstacle_net_ids`],
    /// [`RoutingGrid::nudgeable_fp`], and [`RoutingGrid::smd_pad_cells`].
    ///
    /// # Arguments
    /// * `board` - Reference to the [`Board`] whose static and locked copper features are stamped.
    ///
    /// # Side Effects
    /// Clears [`RoutingGrid::gpu_net_cost_maps`] and populates obstacle, pad, and nudgeability buffers in `self`.
    pub fn rasterize_obstacles(&mut self, board: &Board) {
        self.gpu_net_cost_maps.clear();
        let max_world_x = self.min_x + (self.width as f64) * self.res;
        let max_world_y = self.min_y + (self.height as f64) * self.res;
        if let Some(ref poly) = board.outline_polygon {
            let boundary = poly.aabb();
            let edge_clearance = board.board_edge_clearance();
            let width = self.width;
            let min_x = self.min_x;
            let min_y = self.min_y;
            let res = self.res;
            let is_rect = poly.vertices.len() == 4;
            let in_min_x = boundary.min_x + edge_clearance;
            let in_max_x = boundary.max_x - edge_clearance;
            let in_min_y = boundary.min_y + edge_clearance;
            let in_max_y = boundary.max_y - edge_clearance;

            let mask_2d: Vec<bool> = (0..self.height)
                .into_par_iter()
                .flat_map_iter(|gy| {
                    let wy = min_y + (gy as f64 + 0.5) * res;
                    let y_out_rect = wy < in_min_y || wy > in_max_y;
                    let y_out_aabb = wy < in_min_y - 1e-9 || wy > in_max_y + 1e-9;
                    (0..width).map(move |gx| {
                        let wx = min_x + (gx as f64 + 0.5) * res;
                        if is_rect {
                            y_out_rect || wx < in_min_x || wx > in_max_x
                        } else if y_out_aabb || wx < in_min_x - 1e-9 || wx > in_max_x + 1e-9 {
                            true
                        } else {
                            let p = Point::new(wx, wy);
                            !poly.contains_point(&p) || poly.distance_to_point(&p) < edge_clearance
                        }
                    })
                })
                .collect();

            let layer_stride = self.width * self.height;
            for (cell_2d, &is_keepout) in mask_2d.iter().enumerate() {
                if is_keepout {
                    for layer in 0..self.num_layers {
                        self.obstacle_net_ids[layer * layer_stride + cell_2d] = KEEPOUT_NET;
                    }
                }
            }
        }

        for ko in &board.keepouts {
            if ko.max_x < self.min_x || ko.min_x > max_world_x || ko.max_y < self.min_y || ko.min_y > max_world_y {
                continue;
            }
            let min_gx = ((ko.min_x - self.min_x) / self.res)
                .ceil()
                .clamp(0.0, self.width.saturating_sub(1) as f64) as usize;
            let min_gy = ((ko.min_y - self.min_y) / self.res)
                .ceil()
                .clamp(0.0, self.height.saturating_sub(1) as f64) as usize;
            let max_gx = ((ko.max_x - self.min_x) / self.res)
                .floor()
                .clamp(0.0, self.width.saturating_sub(1) as f64) as usize;
            let max_gy = ((ko.max_y - self.min_y) / self.res)
                .floor()
                .clamp(0.0, self.height.saturating_sub(1) as f64) as usize;
            for &layer in &ko.layers {
                if layer >= self.num_layers {
                    continue;
                }
                for gy in min_gy..=max_gy {
                    for gx in min_gx..=max_gx {
                        let i = self.idx(gx, gy, layer);
                        self.obstacle_net_ids[i] = KEEPOUT_NET;
                    }
                }
            }
        }

        /// Pre-rasterized grid cells for a single footprint pad during parallel obstacle stamping.
        struct PadStamp {
            /// Parent footprint index in `board.footprints`.
            fp_idx: usize,
            /// Whether the parent footprint is unlocked and eligible for nudging.
            is_nudgeable: bool,
            /// Whether the pad is a plated through-hole spanning all copper layers.
            is_through_hole: bool,
            /// Copper layer index for SMD pads.
            smd_layer: usize,
            /// Obstacle net value (`net_id as i32` or [`KEEPOUT_NET`] for unconnected pads).
            net_val: i32,
            /// Rasterized `(gx, gy)` grid cells covered by the pad copper.
            cells: Vec<(usize, usize)>,
        }

        let min_x = self.min_x;
        let min_y = self.min_y;
        let max_x = self.max_x;
        let max_y = self.max_y;
        let res = self.res;
        let width = self.width;
        let height = self.height;
        let num_layers = self.num_layers;

        let build_fp_stamps = |(fp_idx, fp): (usize, &Footprint)| -> Vec<PadStamp> {
            let is_nudgeable = !fp.is_locked;
            let mut stamps = Vec::with_capacity(fp.pads.len());
            for pad in &fp.pads {
                let smd_layer = fp.pad_layer(pad).min(num_layers - 1);
                let (px, py) = fp.pad_absolute_position(pad);
                let (rot_w, rot_h) = fp.pad_world_size(pad);
                let half_w = rot_w / 2.0;
                let half_h = rot_h / 2.0;
                let total_rot = ((fp.rotation + pad.rotation) % 360.0 + 360.0) % 360.0;
                let pad_hw = pad.width * 0.5;
                let pad_hh = pad.height * 0.5;

                let min_gx = ((px - half_w - min_x) / res)
                    .floor()
                    .clamp(0.0, width.saturating_sub(1) as f64) as usize;
                let max_gx = ((px + half_w - min_x) / res)
                    .floor()
                    .clamp(0.0, width.saturating_sub(1) as f64) as usize;
                let min_gy = ((py - half_h - min_y) / res)
                    .floor()
                    .clamp(0.0, height.saturating_sub(1) as f64) as usize;
                let max_gy = ((py + half_h - min_y) / res)
                    .floor()
                    .clamp(0.0, height.saturating_sub(1) as f64) as usize;

                let net_val = if pad.net_id == 0
                    || pad.net_name.is_empty()
                    || pad.net_name.starts_with("unconnected")
                {
                    KEEPOUT_NET
                } else {
                    pad.net_id as i32
                };

                let mut cells = Vec::with_capacity((max_gx - min_gx + 1) * (max_gy - min_gy + 1) + 1);
                for gy in min_gy..=max_gy {
                    let wy = min_y + (gy as f64 + 0.5) * res;
                    for gx in min_gx..=max_gx {
                        let wx = min_x + (gx as f64 + 0.5) * res;
                        let unrot = Point::new(wx - px, wy - py).rotate_ccw(-total_rot);
                        let dx = 0.0f64.max(unrot.x.abs() - pad_hw);
                        let dy = 0.0f64.max(unrot.y.abs() - pad_hh);
                        if dx.hypot(dy) <= res * 0.5 + 1e-4 {
                            cells.push((gx, gy));
                        }
                    }
                }

                // Always ensure the cell containing the pad origin is marked
                if px >= min_x && px < max_x && py >= min_y && py < max_y {
                    let cgx = ((px - min_x) / res) as usize;
                    let cgy = ((py - min_y) / res) as usize;
                    if cgx < width && cgy < height {
                        cells.push((cgx, cgy));
                    }
                }

                stamps.push(PadStamp {
                    fp_idx,
                    is_nudgeable,
                    is_through_hole: pad.is_through_hole,
                    smd_layer,
                    net_val,
                    cells,
                });
            }
            stamps
        };

        let fp_stamps: Vec<Vec<PadStamp>> = if board.footprints.len() >= 8 {
            board.footprints.par_iter().enumerate().map(build_fp_stamps).collect()
        } else {
            board.footprints.iter().enumerate().map(build_fp_stamps).collect()
        };

        let all_layers: Vec<usize> = (0..self.num_layers).collect();
        for stamps in fp_stamps {
            for stamp in stamps {
                let smd_layer = [stamp.smd_layer];
                let layers: &[usize] = if stamp.is_through_hole {
                    &all_layers
                } else {
                    &smd_layer
                };
                for &layer in layers {
                    for &(gx, gy) in &stamp.cells {
                        let i = self.idx(gx, gy, layer);
                        self.obstacle_net_ids[i] = stamp.net_val;
                        if stamp.is_nudgeable {
                            self.nudgeable_fp[i] = Some(stamp.fp_idx);
                        }
                        if !stamp.is_through_hole {
                            self.smd_pad_cells[i] = true;
                        }
                    }
                }
            }
        }

        for trk in &board.tracks {
            if trk.is_locked {
                self.commit_track(trk);
            }
        }
        for v in &board.vias {
            if v.is_locked {
                self.commit_via(v);
            }
        }
    }

    /// Rasterizes a single [`TrackSegment`] into [`RoutingGrid::obstacle_net_ids`] on its layer,
    /// marking free cells within `trk.width / 2.0` of the segment centerline with `trk.net_id`.
    ///
    /// # Arguments
    /// * `trk` - The [`TrackSegment`] to stamp onto the obstacle grid.
    ///
    /// # Side Effects
    /// Invalidates [`RoutingGrid::gpu_net_cost_maps`] and updates [`RoutingGrid::obstacle_net_ids`].
    pub fn commit_track(&mut self, trk: &TrackSegment) {
        if trk.layer >= self.num_layers {
            return;
        }
        self.gpu_net_cost_maps.clear();
        let dist = (trk.end.0 - trk.start.0).hypot(trk.end.1 - trk.start.1);
        let steps = (dist / (self.res * 0.5)).ceil() as usize;
        let r_cells = (trk.width / (2.0 * self.res)).ceil() as isize;
        for s in 0..=steps {
            let t = s as f64 / steps.max(1) as f64;
            let wx = trk.start.0 + t * (trk.end.0 - trk.start.0);
            let wy = trk.start.1 + t * (trk.end.1 - trk.start.1);
            if let Some((gx, gy)) = self.world_to_grid(wx, wy) {
                for dy in -r_cells..=r_cells {
                    for dx in -r_cells..=r_cells {
                        if (dx * dx + dy * dy) as f64 * (self.res * self.res) <= (trk.width / 2.0).powi(2) + 1e-6 {
                            let nx = gx as isize + dx;
                            let ny = gy as isize + dy;
                            if nx >= 0 && nx < self.width as isize && ny >= 0 && ny < self.height as isize {
                                let idx = self.idx(nx as usize, ny as usize, trk.layer);
                                if self.obstacle_net_ids[idx] == FREE_NET {
                                    self.obstacle_net_ids[idx] = trk.net_id as i32;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Rasterizes a through-hole [`Via`] across all copper layers in [`RoutingGrid::obstacle_net_ids`].
    ///
    /// # Arguments
    /// * `v` - The [`Via`] to stamp across all layers `0..num_layers`.
    ///
    /// # Side Effects
    /// Invalidates [`RoutingGrid::gpu_net_cost_maps`] and marks free cells within the via's radius as `v.net_id`.
    pub fn commit_via(&mut self, v: &Via) {
        self.gpu_net_cost_maps.clear();
        let vr_radius = (v.size / 2.0) + (self.res * 0.40);
        let vr_cells = (vr_radius / self.res).ceil() as isize;
        if let Some((gx, gy)) = self.world_to_grid(v.position.0, v.position.1) {
            for cy in -vr_cells..=vr_cells {
                for cx in -vr_cells..=vr_cells {
                    if (cx * cx + cy * cy) as f64 * (self.res * self.res) <= vr_radius.powi(2) + 1e-6 {
                        let nx = gx as isize + cx;
                        let ny = gy as isize + cy;
                        if nx >= 0 && nx < self.width as isize && ny >= 0 && ny < self.height as isize {
                            for l in 0..self.num_layers {
                                let idx = self.idx(nx as usize, ny as usize, l);
                                if self.obstacle_net_ids[idx] == FREE_NET {
                                    self.obstacle_net_ids[idx] = v.net_id as i32;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Returns a 32-bit layer bitmask indicating which inner copper layers are dedicated to a
    /// foreign power/ground plane (`plane_net != net_id`) and therefore off-limits for planar routing of `net_id`.
    ///
    /// # Arguments
    /// * `net_id` - Numeric net ID being routed.
    ///
    /// # Returns
    /// Bitmask where bit `l` is `1` if layer `l` is a dedicated plane belonging to a net other than `net_id`.
    pub fn plane_layer_mask_for_net(&self, net_id: usize) -> u32 {
        let mut mask = 0u32;
        for (&layer, &plane_net) in &self.plane_nets {
            if layer < 32 && plane_net != net_id {
                mask |= 1u32 << layer;
            }
        }
        mask
    }

    /// Computes a 3D per-cell routing cost and clearance map on the GPU for `net_id`.
    ///
    /// Cells within `clearance_cells` of any foreign obstacle are marked `-1.0f` (impassable).
    ///
    /// # Arguments
    /// * `net_id` - Target net ID to evaluate.
    /// * `clearance_cells` - Required obstacle clearance radius in grid cells.
    /// * `present_congestion_factor` - Multiplier for present-iteration cell occupancy counts.
    /// * `obstacle_clearance_weight` - Penalty weight for proximity to obstacles in the distance field.
    ///
    /// # Returns
    /// `Some(Vec<f32>)` containing `num_layers * width * height` per-cell traversal costs, or `None` if no Vulkan GPU is available.
    pub fn compute_gpu_3d_routing_cost_for_net(
        &self,
        net_id: usize,
        clearance_cells: i32,
        present_congestion_factor: f32,
        obstacle_clearance_weight: f32,
    ) -> Option<Vec<f32>> {
        let vk = self.vulkan.as_ref()?;
        let obs_u32: Vec<u32> = self.obstacle_net_ids.iter().map(|&n| n as u32).collect();
        let params = GpuRouteCostParams {
            width: self.width as u32,
            height: self.height as u32,
            num_layers: self.num_layers as u32,
            net_id: net_id as u32,
            clearance_cells,
            present_congestion_factor,
            obstacle_clearance_weight,
            plane_layer_mask: self.plane_layer_mask_for_net(net_id),
        };
        vk.compute_3d_routing_costs(
            &obs_u32,
            &self.dist_field,
            &self.congestion,
            &self.usage_count,
            &params,
        )
    }

    /// Precomputes 3D routing cost and clearance maps across all available GPUs for a batch of `(net_id, clearance_cells)`.
    ///
    /// # Arguments
    /// * `net_clearances_cells` - Slice of `(net_id, clearance_cells)` pairs to evaluate in parallel across GPUs.
    /// * `present_congestion_factor` - Present congestion penalty multiplier.
    /// * `obstacle_clearance_weight` - Distance-field obstacle proximity weight.
    ///
    /// # Side Effects
    /// Populates [`RoutingGrid::gpu_net_cost_maps`] with the computed cost vectors for each net in the batch.
    pub fn precompute_gpu_routing_costs_batch(
        &mut self,
        net_clearances_cells: &[(usize, i32)],
        present_congestion_factor: f32,
        obstacle_clearance_weight: f32,
    ) {
        self.gpu_net_cost_maps.clear();
        if net_clearances_cells.is_empty() {
            return;
        }
        let Some(ref vk) = self.vulkan else {
            return;
        };
        let obs_u32: Vec<u32> = self.obstacle_net_ids.iter().map(|&n| n as u32).collect();
        let params_batch: Vec<GpuRouteCostParams> = net_clearances_cells
            .iter()
            .map(|&(net_id, clearance_cells)| GpuRouteCostParams {
                width: self.width as u32,
                height: self.height as u32,
                num_layers: self.num_layers as u32,
                net_id: net_id as u32,
                clearance_cells,
                present_congestion_factor,
                obstacle_clearance_weight,
                plane_layer_mask: self.plane_layer_mask_for_net(net_id),
            })
            .collect();
        let results = vk.compute_3d_routing_costs_batch(
            &obs_u32,
            &self.dist_field,
            &self.congestion,
            &self.usage_count,
            &params_batch,
        );
        for (&(net_id, clearance_cells), res_opt) in net_clearances_cells.iter().zip(results.into_iter()) {
            if let Some(cost_vec) = res_opt {
                self.gpu_net_cost_maps.insert(net_id, (clearance_cells, cost_vec));
            }
        }
    }

    /// Computes a 3D shortest-path wavefront potential field on the GPU from target seed indices.
    ///
    /// # Arguments
    /// * `net_id` - Net identifier being routed.
    /// * `clearance_cells` - Required obstacle clearance radius in grid cells.
    /// * `seeds` - Slice of flat 3D grid cell indices (`idx`) representing wavefront target seeds (cost `0.0`).
    /// * `via_cost` - Cost penalty for transitioning between layers.
    /// * `iterations` - Number of Bellman-Ford wavefront relaxation iterations to dispatch on the GPU.
    ///
    /// # Returns
    /// `Some(Vec<f32>)` containing the relaxed 3D potential field, or `None` if GPU compute is unavailable.
    pub fn compute_gpu_3d_wavefront_for_net(
        &self,
        net_id: usize,
        clearance_cells: i32,
        seeds: &[u32],
        via_cost: f32,
        iterations: u32,
    ) -> Option<Vec<f32>> {
        let vk = self.vulkan.as_ref()?;
        let cell_costs = if let Some((cached_r, ref map)) = self.gpu_net_cost_maps.get(&net_id) {
            if *cached_r == clearance_cells {
                map.clone()
            } else {
                self.compute_gpu_3d_routing_cost_for_net(net_id, clearance_cells, 1.5, 2.0)?
            }
        } else {
            self.compute_gpu_3d_routing_cost_for_net(net_id, clearance_cells, 1.5, 2.0)?
        };
        let params = GpuRouteCostParams {
            width: self.width as u32,
            height: self.height as u32,
            num_layers: self.num_layers as u32,
            net_id: net_id as u32,
            clearance_cells,
            present_congestion_factor: 1.5,
            obstacle_clearance_weight: 2.0,
            plane_layer_mask: self.plane_layer_mask_for_net(net_id),
        };
        vk.compute_3d_wavefront(&cell_costs, seeds, &params, via_cost, iterations)
    }

    /// Batch-validates a slice of `(net_id, start, end, width, layer, clearance)` track segments
    /// on the GPU using [`GpuDrcContext`] alongside 3D grid obstacle and plane-layer clearance checks.
    ///
    /// # Arguments
    /// * `board` - Reference to the [`Board`] for DRC context construction and keepout checks.
    /// * `segments` - Slice of candidate track tuples `(net_id, start_xy, end_xy, width_mm, layer, clearance_mm)`.
    ///
    /// # Returns
    /// `Some(Vec<bool>)` indicating whether each candidate segment is DRC-clean, or `None` if no Vulkan GPU is available.
    pub fn batch_validate_segments_gpu(
        &self,
        board: &Board,
        segments: &[(usize, (f64, f64), (f64, f64), f64, usize, f64)],
    ) -> Option<Vec<bool>> {
        if segments.is_empty() {
            return Some(Vec::new());
        }
        let vk = self.vulkan.as_ref()?;
        let drc_ctx = GpuDrcContext::from_board(board);
        let cands: Vec<GpuDrcCandidate> = segments
            .iter()
            .map(|&(net_id, start, end, width, layer, clearance)| GpuDrcCandidate {
                x1: start.0 as f32,
                y1: start.1 as f32,
                x2: end.0 as f32,
                y2: end.1 as f32,
                radius_or_half_width: (width * 0.5) as f32,
                clearance: clearance as f32,
                layer: layer as u32,
                net_id: net_id as u32,
                kind: 1,
                ignore_same_net_pads: 1,
                _pad0: 0,
                _pad1: 0,
            })
            .collect();
        let gpu_flags = drc_ctx.evaluate(Some(vk), &cands)?;
        let results = segments
            .iter()
            .zip(gpu_flags.into_iter())
            .map(|(&(net_id, start, end, width, layer, clearance), flag)| {
                if flag == 0 || layer >= self.num_layers {
                    return false;
                }
                if self.plane_nets.get(&layer).is_some_and(|&pnet| pnet != net_id) {
                    return false;
                }
                let margin = width * 0.5 + clearance;
                for ko in &board.keepouts {
                    if ko.layers.contains(&layer) {
                        let k_min_x = ko.min_x - margin;
                        let k_max_x = ko.max_x + margin;
                        let k_min_y = ko.min_y - margin;
                        let k_max_y = ko.max_y + margin;
                        if (start.0 >= k_min_x && start.0 <= k_max_x && start.1 >= k_min_y && start.1 <= k_max_y)
                            || (end.0 >= k_min_x && end.0 <= k_max_x && end.1 >= k_min_y && end.1 <= k_max_y)
                        {
                            return false;
                        }
                    }
                }
                let seg_len = (end.0 - start.0).hypot(end.1 - start.1);
                let step = (self.res * 0.5).max(0.02);
                let n_steps = ((seg_len / step).ceil() as usize).max(1);
                for s in 0..=n_steps {
                    let t = s as f64 / n_steps as f64;
                    let wx = start.0 + t * (end.0 - start.0);
                    let wy = start.1 + t * (end.1 - start.1);
                    let Some((gx, gy)) = self.world_to_grid(wx, wy) else {
                        return false;
                    };
                    let cell_net = self.obstacle_net_ids[self.idx(gx, gy, layer)];
                    if cell_net != FREE_NET && cell_net != net_id as i32 {
                        return false;
                    }
                }
                true
            })
            .collect();
        Some(results)
    }

    /// Computes a 2D Euclidean Distance Field (in grid cell units) on the CPU using a separable
    /// 2-pass column-then-row squared distance transform bounded by `radius`.
    ///
    /// # Arguments
    /// * `obstacles` - Flat 2D slice (`width * height`) where non-zero entries indicate occupied obstacle cells.
    /// * `width` - Grid width in cells.
    /// * `height` - Grid height in cells.
    /// * `radius` - Maximum search radius in cells (`radius >= 1`).
    ///
    /// # Returns
    /// Vector of `width * height` Euclidean distances (in grid cells) to the nearest obstacle cell.
    pub fn compute_distance_field_cpu(
        obstacles: &[u32],
        width: usize,
        height: usize,
        radius: i32,
    ) -> Vec<f32> {
        let total = width * height;
        if obstacles.len() != total || total == 0 {
            return vec![0.0; total];
        }
        let r2 = (radius * radius + 1) as f32;
        let inf = 1e9f32;

        // 1D squared distance transform along columns
        let mut d1 = vec![inf; total];
        for x in 0..width {
            let mut d = inf;
            for y in 0..height {
                let idx = y * width + x;
                if obstacles[idx] != 0 {
                    d = 0.0;
                } else if d < inf {
                    d += 1.0;
                }
                d1[idx] = if d < inf { d * d } else { inf };
            }
            let mut d = inf;
            for y in (0..height).rev() {
                let idx = y * width + x;
                if obstacles[idx] != 0 {
                    d = 0.0;
                } else if d < inf {
                    d += 1.0;
                }
                let d_sq = if d < inf { d * d } else { inf };
                if d_sq < d1[idx] {
                    d1[idx] = d_sq;
                }
            }
        }

        // Along rows, evaluate min_{|dx| <= radius} (dx*dx + d1[x+dx, y])
        let mut out = vec![0.0f32; total];
        let rad = radius.max(1);
        for y in 0..height {
            let row_off = y * width;
            for x in 0..width {
                let idx = row_off + x;
                if obstacles[idx] != 0 {
                    out[idx] = 0.0;
                    continue;
                }
                let mut min_val = r2;
                let x_min = (x as i32 - rad).max(0) as usize;
                let x_max = (x as i32 + rad).min(width as i32 - 1) as usize;
                for nx in x_min..=x_max {
                    let dx = x as i32 - nx as i32;
                    let val = (dx * dx) as f32 + d1[row_off + nx];
                    if val < min_val {
                        min_val = val;
                    }
                }
                out[idx] = min_val.sqrt();
            }
        }
        out
    }

    /// Recomputes the 2D Euclidean Distance Field (`dist_field`, in `mm`) across all copper layers
    /// from the current [`RoutingGrid::obstacle_net_ids`] buffer.
    ///
    /// Dispatches the distance-field compute shader across all available Vulkan GPUs in parallel
    /// when [`RoutingGrid::vulkan`] is present, or falls back to multi-core Rayon CPU evaluation
    /// via [`RoutingGrid::compute_distance_field_cpu`].
    ///
    /// # Side Effects
    /// Overwrites [`RoutingGrid::dist_field`] with updated obstacle distances in millimeters (`mm`).
    pub fn update_distance_fields(&mut self) {
        let layer_cells = self.width * self.height;
        let w = self.width;
        let h = self.height;
        let res = self.res as f32;
        let num_layers = self.num_layers;

        if let Some(ref vk) = self.vulkan {
            let engines = vk.all_engines();
            let w_u32 = w as u32;
            let h_u32 = h as u32;

            if engines.len() > 1 && num_layers > 1 {
                let num_engines = engines.len();
                let mut layer_obs: Vec<Vec<u32>> = Vec::with_capacity(num_layers);
                for layer in 0..num_layers {
                    let offset = layer * layer_cells;
                    let obs: Vec<u32> = self.obstacle_net_ids[offset..offset + layer_cells]
                        .iter()
                        .map(|&n| if n != FREE_NET { 1u32 } else { 0u32 })
                        .collect();
                    layer_obs.push(obs);
                }

                let mut layer_dfs: Vec<Option<Vec<f32>>> = (0..num_layers).map(|_| None).collect();
                std::thread::scope(|scope| {
                    let mut handles = Vec::new();
                    for (eng_idx, engine) in engines.iter().enumerate() {
                        let assigned_layers: Vec<usize> = (0..num_layers)
                            .filter(|&l| l % num_engines == eng_idx)
                            .collect();
                        if assigned_layers.is_empty() {
                            continue;
                        }
                        let eng_ref = Arc::clone(engine);
                        let mut flat_obs = Vec::with_capacity(assigned_layers.len() * layer_cells);
                        for &l in &assigned_layers {
                            flat_obs.extend_from_slice(&layer_obs[l]);
                        }

                        handles.push(scope.spawn(move || {
                            let mut results = Vec::new();
                            let sub_layers = assigned_layers.len() as u32;
                            if let Some(batch_df) = eng_ref.compute_distance_field_layers(&flat_obs, w_u32, h_u32, sub_layers, 12) {
                                for (i, &l) in assigned_layers.iter().enumerate() {
                                    let off = i * layer_cells;
                                    results.push((l, batch_df[off..off + layer_cells].to_vec()));
                                }
                            }
                            results
                        }));
                    }
                    for handle in handles {
                        if let Ok(results) = handle.join() {
                            for (l, df) in results {
                                layer_dfs[l] = Some(df);
                            }
                        }
                    }
                });

                let mut all_ok = true;
                for (layer, df_opt) in layer_dfs.into_iter().enumerate() {
                    if let Some(df) = df_opt {
                        let offset = layer * layer_cells;
                        for (dst, &val) in self.dist_field[offset..offset + layer_cells].iter_mut().zip(df.iter()) {
                            *dst = val * res;
                        }
                    } else {
                        all_ok = false;
                    }
                }
                if all_ok {
                    return;
                }
            } else if num_layers > 0 {
                let total_cells = num_layers * layer_cells;
                let obs_flat: Vec<u32> = self.obstacle_net_ids[..total_cells]
                    .iter()
                    .map(|&n| if n != FREE_NET { 1u32 } else { 0u32 })
                    .collect();
                if let Some(batch_df) = vk.compute_distance_field_layers(&obs_flat, w_u32, h_u32, num_layers as u32, 12) {
                    for (dst, &val) in self.dist_field[..total_cells].iter_mut().zip(batch_df.iter()) {
                        *dst = val * res;
                    }
                    return;
                }
            }
        }

        // Multi-core CPU fallback: evaluate exact 2D Euclidean Distance Field transforms
        // when Vulkan GPU compute is absent (e.g. headless CI environments or systems without hardware GPUs).
        let layer_results: Vec<Vec<f32>> = (0..num_layers)
            .into_par_iter()
            .map(|layer| {
                let offset = layer * layer_cells;
                let obs_slice = &self.obstacle_net_ids[offset..offset + layer_cells];
                let obs_u32: Vec<u32> = obs_slice
                    .iter()
                    .map(|&n| if n != FREE_NET { 1u32 } else { 0u32 })
                    .collect();
                Self::compute_distance_field_cpu(&obs_u32, w, h, 12)
            })
            .collect();

        for (layer, df) in layer_results.into_iter().enumerate() {
            let offset = layer * layer_cells;
            for (dst, &val) in self.dist_field[offset..offset + layer_cells].iter_mut().zip(df.iter()) {
                *dst = val * res;
            }
        }
    }


    /// Enumerates all legal 3D A* neighbor transitions (8 octilinear planar steps on `node.layer`
    /// plus through-hole via transitions to other layers) from `node` for `target_net_id`.
    ///
    /// Enforces bidirectional netclass clearances (`(track_width / 2.0) + max(req_clearance, obst_clearance)`),
    /// diagonal corner-pinching guards, dedicated inner plane restrictions, and SMD via-in-pad exclusions.
    ///
    /// # Arguments
    /// * `node` - Current [`GridNode`] being expanded.
    /// * `target_net_id` - Numeric net ID of the net being routed.
    /// * `req_clearance` - Required copper clearance (in `mm`) for `target_net_id`.
    /// * `track_width` - Copper width (in `mm`) of the trace being routed.
    /// * `via_dia` - Annular ring outer diameter (in `mm`) of a layer-transition via.
    /// * `u1_center` - Optional world coordinates `(x, y)` of the primary fine-pitch IC (`U1`) for escape clearance relaxation.
    ///
    /// # Returns
    /// Vector of `(neighbor_node, step_cost)` pairs for all legal transitions out of `node`.
    pub fn get_neighbors(
        &self,
        node: GridNode,
        target_net_id: usize,
        req_clearance: f64,
        track_width: f64,
        via_dia: f64,
        u1_center: Option<(f64, f64)>,
    ) -> Vec<(GridNode, f64)> {
        let mut neighbors = Vec::with_capacity(12);
        let gx = node.gx;
        let gy = node.gy;
        let layer = node.layer;

        let planar_dirs = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.4142),
            (1, -1, 1.4142),
            (-1, 1, 1.4142),
            (1, 1, 1.4142),
        ];

        let (h_mult, v_mult): (f64, f64) = if layer % 2 == 0 { (1.0, 1.35) } else { (1.35, 1.0) };

        let is_dedicated_plane = self.plane_nets.contains_key(&layer);
        let allow_planar = !is_dedicated_plane || self.plane_nets.get(&layer) == Some(&target_net_id);

        if allow_planar {
            let gpu_cost_entry = self.gpu_net_cost_maps.get(&target_net_id);
            for (dx, dy, base_len) in planar_dirs {
            let nx = gx as isize + dx;
            let ny = gy as isize + dy;
            if nx >= 0 && nx < self.width as isize && ny >= 0 && ny < self.height as isize {
                let ngx = nx as usize;
                let ngy = ny as usize;
                let n_idx = self.idx(ngx, ngy, layer);
                let net_id = self.obstacle_net_ids[n_idx];
                if net_id != FREE_NET && net_id != target_net_id as i32 {
                    continue;
                }

                if dx != 0 && dy != 0 {
                    let idx_h = self.idx(ngx, gy, layer);
                    let idx_v = self.idx(gx, ngy, layer);
                    let net_h = self.obstacle_net_ids[idx_h];
                    let net_v = self.obstacle_net_ids[idx_v];
                    if (net_h != FREE_NET && net_h != target_net_id as i32)
                        || (net_v != FREE_NET && net_v != target_net_id as i32)
                    {
                        continue;
                    }
                }

                let is_diag = dx != 0 && dy != 0;
                let diag_margin = if is_diag { (self.res * 0.7071).min(0.02) } else { 0.0 };
                let max_clearance = self.max_net_clearance.max(req_clearance);
                let search_clearance = (track_width / 2.0) + max_clearance + diag_margin;
                let r_cells = (search_clearance / self.res).ceil() as isize;
                let mut clearance_violation = false;
                let gpu_certified_clear = if let Some((cached_r, cost_map)) = gpu_cost_entry {
                    *cached_r >= r_cells as i32 && cost_map.get(n_idx).copied().unwrap_or(-1.0) > 0.0
                } else {
                    false
                };
                if !gpu_certified_clear {
                    for cy_off in -r_cells..=r_cells {
                        for cx_off in -r_cells..=r_cells {
                            let cx = ngx as isize + cx_off;
                            let cy = ngy as isize + cy_off;
                            if cx >= 0 && cx < self.width as isize && cy >= 0 && cy < self.height as isize {
                                let c_idx = self.idx(cx as usize, cy as usize, layer);
                                let c_net = self.obstacle_net_ids[c_idx];
                                if c_net != FREE_NET && c_net != target_net_id as i32 {
                                    let raw_obst_c = if c_net > 0 && (c_net as usize) < self.net_clearances.len() {
                                        self.net_clearances[c_net as usize]
                                    } else {
                                        req_clearance
                                    };
                                    let near_u1 = u1_center.is_some_and(|(ux, uy)| {
                                        let (wx, wy) = self.grid_to_world(cx as usize, cy as usize);
                                        (wx - ux).abs() < 2.5 && (wy - uy).abs() < 2.5
                                    });
                                    let eff_c = if self.smd_pad_cells[c_idx] && (near_u1 || (req_clearance <= 0.20 && raw_obst_c <= 0.20)) {
                                        (req_clearance.min(raw_obst_c) - 0.015).max(0.135)
                                    } else if near_u1 && req_clearance <= 0.15 {
                                        req_clearance.min(raw_obst_c).max(0.15)
                                    } else {
                                        req_clearance.max(raw_obst_c)
                                    };
                                    let eff_clearance = (track_width / 2.0) + eff_c + diag_margin;
                                    let c_dist_sq = (cx_off * cx_off + cy_off * cy_off) as f64 * (self.res * self.res);
                                    if c_dist_sq < (eff_clearance - 1e-4).powi(2) {
                                        clearance_violation = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if clearance_violation {
                            break;
                        }
                    }
                }

                if clearance_violation {
                    continue;
                }

                let dir_factor = if dx != 0 && dy == 0 {
                    h_mult
                } else if dx == 0 && dy != 0 {
                    v_mult
                } else {
                    h_mult.max(v_mult)
                };

                let cong = self.congestion[n_idx] as f64;
                let mut step_cost = (base_len * self.res) * dir_factor * (1.0 + cong);
                if layer >= 2 {
                    if self.plane_nets.get(&layer) == Some(&target_net_id) {
                        step_cost *= 0.40;
                    } else {
                        step_cost += self.inner_layer_penalty * (base_len * self.res);
                    }
                }
                neighbors.push((GridNode { gx: ngx, gy: ngy, layer }, step_cost));
            }
        }
    }

        // Through-hole via transitions across all available copper layers (F.Cu, B.Cu, In1.Cu, In2.Cu)
        let mut inside_disallowed_smd_pad = false;
        for l in 0..self.num_layers {
            let cell_idx = self.idx(gx, gy, l);
            if self.smd_pad_cells[cell_idx] {
                if !self.allow_via_in_pad || self.obstacle_net_ids[cell_idx] != target_net_id as i32 {
                    inside_disallowed_smd_pad = true;
                    break;
                }
            }
        }

        if !inside_disallowed_smd_pad {
            let max_clearance = self.max_net_clearance.max(req_clearance);
            let obst_half_w = (track_width / 2.0).max(0.10);
            let search_clearance = (via_dia / 2.0) + obst_half_w + max_clearance + 0.025;
            let vr_cells = (search_clearance / self.res).ceil() as isize;
            let mut via_clearance_violation = false;
            'via_check: for cy_off in -vr_cells..=vr_cells {
                for cx_off in -vr_cells..=vr_cells {
                    let cx = gx as isize + cx_off;
                    let cy = gy as isize + cy_off;
                    if cx >= 0 && cx < self.width as isize && cy >= 0 && cy < self.height as isize {
                        let d_sq = (cx_off * cx_off + cy_off * cy_off) as f64 * (self.res * self.res);
                        for l in 0..self.num_layers {
                            let c_idx = self.idx(cx as usize, cy as usize, l);
                            let c_net = self.obstacle_net_ids[c_idx];
                            if c_net != FREE_NET && c_net != target_net_id as i32 {
                                let obst_c = if c_net > 0 && (c_net as usize) < self.net_clearances.len() {
                                    self.net_clearances[c_net as usize]
                                } else {
                                    req_clearance
                                };
                                let eff_c_val = if req_clearance <= 0.15 || self.smd_pad_cells[c_idx] {
                                    req_clearance.min(obst_c).max(0.12)
                                } else {
                                    req_clearance.max(obst_c)
                                };
                                let obst_r = if self.smd_pad_cells[c_idx] { 0.025 } else { (track_width / 2.0).max(0.075) };
                                let eff_c = (via_dia / 2.0) + obst_r + eff_c_val;
                                if d_sq < eff_c.powi(2) + 1e-6 {
                                    via_clearance_violation = true;
                                    break 'via_check;
                                }
                            }
                            if self.smd_pad_cells[c_idx] {
                                if c_net == target_net_id as i32 && self.allow_via_in_pad {
                                    continue;
                                }
                                let obst_c = if c_net > 0 && (c_net as usize) < self.net_clearances.len() {
                                    self.net_clearances[c_net as usize]
                                } else {
                                    req_clearance
                                };
                                let eff_c = if c_net == target_net_id as i32 {
                                    (via_dia / 2.0) + self.res * 0.5
                                } else {
                                    let eff_c_val = if req_clearance <= 0.15 || self.smd_pad_cells[c_idx] {
                                        req_clearance.min(obst_c).max(0.12)
                                    } else {
                                        req_clearance.max(obst_c)
                                    };
                                    (via_dia / 2.0) + 0.025 + eff_c_val
                                };
                                if d_sq < eff_c.powi(2) - 1e-6 {
                                    via_clearance_violation = true;
                                    break 'via_check;
                                }
                            }
                        }
                    }
                }
            }

            for other_layer in 0..self.num_layers {
                if other_layer == layer {
                    continue;
                }
                if self.plane_nets.contains_key(&other_layer) && self.plane_nets.get(&other_layer) != Some(&target_net_id) {
                    continue;
                }
                let other_idx = self.idx(gx, gy, other_layer);
                let other_net = self.obstacle_net_ids[other_idx];
                if !via_clearance_violation && (other_net == FREE_NET || other_net == target_net_id as i32) {
                    let base_via_cost = if other_net == target_net_id as i32 {
                        0.1
                    } else if self.plane_nets.get(&other_layer) == Some(&target_net_id) {
                        1.2
                    } else {
                        2.0
                    };
                    let ic_penalty = if let Some(center) = u1_center {
                        let (wx, wy) = self.grid_to_world(gx, gy);
                        let d = (wx - center.0).hypot(wy - center.1);
                        if d < 2.8 && other_net != target_net_id as i32 {
                            2.5
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    let via_cost = base_via_cost + ic_penalty;
                    neighbors.push((GridNode { gx, gy, layer: other_layer }, via_cost));
                }
            }
        }

        neighbors
    }

    /// Evaluates proximity to existing routed tracks of foreign nets on the same layer.
    ///
    /// Returns `true` if the candidate step `from -> to` runs parallel to an adjacent foreign track
    /// within capacitive/inductive crosstalk distance (`d <= track_width + 2.0 * req_clearance`).
    ///
    /// # Arguments
    /// * `from` - Starting [`GridNode`] of the step.
    /// * `to` - Ending [`GridNode`] of the step.
    /// * `net_id` - Net ID of the trace being routed.
    /// * `track_width` - Trace width in millimeters (`mm`).
    /// * `req_clearance` - Required copper clearance in millimeters (`mm`).
    ///
    /// # Returns
    /// `true` if a foreign net occupies cells parallel to `from -> to` within the coupling threshold.
    pub fn is_step_coupled(
        &self,
        from: GridNode,
        to: GridNode,
        net_id: usize,
        track_width: f64,
        req_clearance: f64,
    ) -> bool {
        if from.layer != to.layer {
            return false;
        }
        let layer = from.layer;
        let dx = to.gx as isize - from.gx as isize;
        let dy = to.gy as isize - from.gy as isize;
        if dx == 0 && dy == 0 {
            return false;
        }

        let perp_dirs = if dx != 0 && dy == 0 {
            [(0, 1), (0, -1)]
        } else if dx == 0 && dy != 0 {
            [(1, 0), (-1, 0)]
        } else {
            [(-dy, dx), (dy, -dx)]
        };

        let max_dist = track_width + 2.0 * req_clearance;
        let is_diag = dx != 0 && dy != 0;
        let step_scale = if is_diag { 1.41421356 } else { 1.0 };
        let max_k = ((max_dist / (self.res * step_scale)).ceil() as isize).max(1);

        for &(pdx, pdy) in &perp_dirs {
            for k in 1..=max_k {
                let dist = k as f64 * self.res * step_scale;
                if dist > max_dist + 1e-4 {
                    break;
                }
                let off_x = pdx * k;
                let off_y = pdy * k;

                let fx1 = from.gx as isize + off_x;
                let fy1 = from.gy as isize + off_y;
                let fx2 = to.gx as isize + off_x;
                let fy2 = to.gy as isize + off_y;

                if fx1 >= 0 && fx1 < self.width as isize && fy1 >= 0 && fy1 < self.height as isize
                    && fx2 >= 0 && fx2 < self.width as isize && fy2 >= 0 && fy2 < self.height as isize
                {
                    let idx1 = self.idx(fx1 as usize, fy1 as usize, layer);
                    let idx2 = self.idx(fx2 as usize, fy2 as usize, layer);
                    let net1 = self.obstacle_net_ids[idx1];
                    let net2 = self.obstacle_net_ids[idx2];

                    if net1 > 0 && net1 != net_id as i32 && net1 != KEEPOUT_NET {
                        if net1 == net2 {
                            return true;
                        }
                        for cdy in -1..=1 {
                            for cdx in -1..=1 {
                                let cfx = fx2 + cdx;
                                let cfy = fy2 + cdy;
                                if cfx >= 0 && cfx < self.width as isize && cfy >= 0 && cfy < self.height as isize {
                                    let cidx = self.idx(cfx as usize, cfy as usize, layer);
                                    if self.obstacle_net_ids[cidx] == net1 {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Determines the physically adjacent reference plane layer for a given `signal_layer`.
    ///
    /// Maps top copper (`0`) to `In1.Cu` (`2`), bottom copper (`1`) to `In2.Cu` (`3`) or `In1.Cu` (`2`),
    /// and inner signal layers to the closest defined plane layer.
    ///
    /// # Arguments
    /// * `signal_layer` - Copper layer index of the signal trace.
    ///
    /// # Returns
    /// `Some(plane_layer)` if at least one inner reference plane exists, or `None` on 2-layer boards without planes.
    pub fn adjacent_reference_plane(&self, signal_layer: usize) -> Option<usize> {
        if self.plane_nets.is_empty() {
            return None;
        }
        match signal_layer {
            0 => {
                if self.plane_nets.contains_key(&2) {
                    Some(2)
                } else {
                    self.plane_nets.keys().copied().min()
                }
            }
            1 => {
                if self.plane_nets.contains_key(&3) {
                    Some(3)
                } else if self.plane_nets.contains_key(&2) {
                    Some(2)
                } else {
                    self.plane_nets.keys().copied().next()
                }
            }
            l => {
                self.plane_nets.keys().copied().min_by_key(|&p| (p as isize - l as isize).abs())
            }
        }
    }

    /// Checks if the reference plane at `(gx, gy)` on `ref_layer` has a split, cutout, or void
    /// that would interrupt high-speed return current continuity.
    ///
    /// # Arguments
    /// * `gx` - Grid column index.
    /// * `gy` - Grid row index.
    /// * `ref_layer` - Copper layer index of the reference plane.
    ///
    /// # Returns
    /// `true` if `(gx, gy)` is out of bounds, marked as [`KEEPOUT_NET`], or occupied by a foreign net on `ref_layer`.
    pub fn is_reference_plane_void_at(&self, gx: usize, gy: usize, ref_layer: usize) -> bool {
        if gx >= self.width || gy >= self.height || ref_layer >= self.num_layers {
            return true;
        }
        let idx = self.idx(gx, gy, ref_layer);
        let obst = self.obstacle_net_ids[idx];

        // 1. Explicit keepout/cutout zone on plane layer
        if obst == KEEPOUT_NET {
            return true;
        }

        // 2. Foreign net crossing on dedicated plane layer (e.g. split plane boundary into another net)
        if let Some(&plane_net) = self.plane_nets.get(&ref_layer) {
            if obst > 0 && obst != plane_net as i32 {
                return true;
            }
        }
        false
    }

    /// Checks whether all cells within a circle of `radius_cells` around `(gx, gy)` on `layer`
    /// are free of foreign-net obstacles for `net_id`.
    ///
    /// # Arguments
    /// * `gx` - Center grid column index.
    /// * `gy` - Center grid row index.
    /// * `layer` - Copper layer index.
    /// * `radius_cells` - Clearance radius in grid cells.
    /// * `net_id` - Numeric net ID allowed within the circle.
    ///
    /// # Returns
    /// `true` if every cell inside the circle is either [`FREE_NET`] or belongs to `net_id`.
    #[allow(non_snake_case)]
    pub fn is_circle_clear_SpecifyRadius(
        &self,
        gx: usize,
        gy: usize,
        layer: usize,
        radius_cells: i32,
        net_id: usize,
    ) -> bool {
        if gx >= self.width || gy >= self.height || layer >= self.num_layers {
            return false;
        }
        let r = radius_cells.max(0) as isize;
        let r2 = r * r;
        let target = net_id as i32;
        for dy in -r..=r {
            let ny = gy as isize + dy;
            if ny < 0 || ny >= self.height as isize {
                continue;
            }
            for dx in -r..=r {
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let nx = gx as isize + dx;
                if nx < 0 || nx >= self.width as isize {
                    continue;
                }
                let c_idx = self.idx(nx as usize, ny as usize, layer);
                let obs = self.obstacle_net_ids[c_idx];
                if obs != FREE_NET && obs != target {
                    return false;
                }
            }
        }
        true
    }

    /// Snake-case wrapper around [`RoutingGrid::is_circle_clear_SpecifyRadius`].
    ///
    /// # Arguments
    /// * `gx` - Center grid column index.
    /// * `gy` - Center grid row index.
    /// * `layer` - Copper layer index.
    /// * `radius_cells` - Clearance radius in grid cells.
    /// * `net_id` - Numeric net ID allowed within the circle.
    ///
    /// # Returns
    /// `true` if every cell inside the circle is free of foreign obstacles.
    pub fn is_circle_clear_specify_radius(
        &self,
        gx: usize,
        gy: usize,
        layer: usize,
        radius_cells: i32,
        net_id: usize,
    ) -> bool {
        self.is_circle_clear_SpecifyRadius(gx, gy, layer, radius_cells, net_id)
    }

    /// Solves a batch of 3D shortest paths across all available GPUs using [`MultiGpuRouterPool`]
    /// and validates every returned node against [`RoutingGrid::is_circle_clear_SpecifyRadius`].
    ///
    /// # Arguments
    /// * `requests` - Slice of `(src_xyz, dst_xyz, net_id, via_cost)` pathfinding requests.
    /// * `clearance_cells` - Required obstacle clearance radius in grid cells.
    ///
    /// # Returns
    /// Vector of `Option<Vec<GridNode>>` corresponding 1-to-1 with `requests`.
    pub fn solve_batch_3d_wavefront_paths(
        &self,
        requests: &[((usize, usize, usize), (usize, usize, usize), u32, f32)],
        clearance_cells: i32,
    ) -> Vec<Option<Vec<GridNode>>> {
        if requests.is_empty() {
            return Vec::new();
        }
        let obs_u32: Vec<u32> = self.obstacle_net_ids.iter().map(|&n| n as u32).collect();
        let reqs_with_clearance: Vec<((usize, usize, usize), (usize, usize, usize), u32, f32, i32)> =
            requests
                .iter()
                .map(|&(src, dst, net_id, via_cost)| {
                    (src, dst, net_id, via_cost, clearance_cells.max(0))
                })
                .collect();
        let pool = MultiGpuRouterPool::with_vulkan(self.vulkan.clone());
        if pool.device_count() == 0 {
            return vec![None; requests.len()];
        }
        let raw_paths = pool.solve_batch_3d_wavefront_paths_with_clearance(
            &obs_u32,
            self.width,
            self.height,
            self.num_layers,
            &reqs_with_clearance,
        );
        raw_paths
            .into_iter()
            .zip(requests.iter())
            .map(|(path_opt, &(_src, _dst, net_id, _via_cost))| {
                let coords = path_opt?;
                let mut nodes = Vec::with_capacity(coords.len());
                for (gx, gy, layer) in coords {
                    if !self.is_circle_clear_SpecifyRadius(
                        gx,
                        gy,
                        layer,
                        clearance_cells.max(0),
                        net_id as usize,
                    ) {
                        return None;
                    }
                    nodes.push(GridNode { gx, gy, layer });
                }
                Some(nodes)
            })
            .collect()
    }
}

/// Computes the shortest Euclidean distance (in `mm`) from point `(px, py)` to the finite line segment `(x1, y1) -> (x2, y2)`.
///
/// # Arguments
/// * `px` - Query point X coordinate (`mm`).
/// * `py` - Query point Y coordinate (`mm`).
/// * `x1` - Segment start X coordinate (`mm`).
/// * `y1` - Segment start Y coordinate (`mm`).
/// * `x2` - Segment end X coordinate (`mm`).
/// * `y2` - Segment end Y coordinate (`mm`).
///
/// # Returns
/// Minimum Euclidean distance in millimeters (`mm`).
pub fn dist_to_segment(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-6 {
        (px - x1).hypot(py - y1)
    } else {
        let t = (((px - x1) * dx + (py - y1) * dy) / len_sq).clamp(0.0, 1.0);
        (px - (x1 + t * dx)).hypot(py - (y1 + t * dy))
    }
}

/// Tests whether two 2D line segments `p1 -> p2` and `p3 -> p4` strictly cross each other's interior.
///
/// Shared endpoints or collinear touches within `1e-9` tolerance do not count as a strict crossing.
///
/// # Arguments
/// * `p1` - First endpoint of segment A (`mm`).
/// * `p2` - Second endpoint of segment A (`mm`).
/// * `p3` - First endpoint of segment B (`mm`).
/// * `p4` - Second endpoint of segment B (`mm`).
///
/// # Returns
/// `true` if `p1 -> p2` and `p3 -> p4` strictly intersect.
pub fn segments_intersect(p1: (f64, f64), p2: (f64, f64), p3: (f64, f64), p4: (f64, f64)) -> bool {
    /// Computes the signed 2D cross-product orientation of `(a, b, c)`.
    fn ccw(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
        (c.1 - a.1) * (b.0 - a.0) - (b.1 - a.1) * (c.0 - a.0)
    }
    let d1 = ccw(p3, p4, p1);
    let d2 = ccw(p3, p4, p2);
    let d3 = ccw(p1, p2, p3);
    let d4 = ccw(p1, p2, p4);

    ((d1 > 1e-9 && d2 < -1e-9) || (d1 < -1e-9 && d2 > 1e-9))
        && ((d3 > 1e-9 && d4 < -1e-9) || (d3 < -1e-9 && d4 > 1e-9))
}

/// Computes the minimum Euclidean distance (in `mm`) between two finite 2D line segments `p1 -> p2` and `p3 -> p4`.
///
/// Returns `0.0` if the segments intersect, or the minimum point-to-segment distance across all four endpoints.
///
/// # Arguments
/// * `p1` - Start point of segment A (`mm`).
/// * `p2` - End point of segment A (`mm`).
/// * `p3` - Start point of segment B (`mm`).
/// * `p4` - End point of segment B (`mm`).
///
/// # Returns
/// Minimum segment-to-segment distance in millimeters (`mm`).
pub fn dist_segment_to_segment(p1: (f64, f64), p2: (f64, f64), p3: (f64, f64), p4: (f64, f64)) -> f64 {
    if segments_intersect(p1, p2, p3, p4) {
        return 0.0;
    }
    dist_to_segment(p1.0, p1.1, p3.0, p3.1, p4.0, p4.1)
        .min(dist_to_segment(p2.0, p2.1, p3.0, p3.1, p4.0, p4.1))
        .min(dist_to_segment(p3.0, p3.1, p1.0, p1.1, p2.0, p2.1))
        .min(dist_to_segment(p4.0, p4.1, p1.0, p1.1, p2.0, p2.1))
}
