// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! GPU-accelerated and analytic copper DRC clearance validation, dangling stub/cycle pruning,
//! crosstalk coupling analysis, and equidistant bus-bundle river routing.

use super::*;

// ============================================================================
// Robust Copper Geometry, DRC Validation & Automated Pruning
// ============================================================================

/// Flat GPU-ready copper obstacle snapshot for batch DRC clearance evaluation.
pub struct GpuDrcContext {
    /// Flat array of pad obstacles in `#[repr(C)]` GPU layout.
    pub pads: Vec<GpuDrcPad>,
    /// Flat array of track segment obstacles in `#[repr(C)]` GPU layout.
    pub tracks: Vec<GpuDrcTrack>,
    /// Flat array of via obstacles in `#[repr(C)]` GPU layout.
    pub vias: Vec<GpuDrcVia>,
}

impl GpuDrcContext {
    /// Extracts all pads, tracks, and vias from `board` into flat `#[repr(C)]` POD arrays
    /// suitable for CUDA (`batch_drc_eval_kernel`) or Vulkan (`BATCH_DRC_EVAL_WGSL`).
    ///
    /// # Arguments
    /// * `board` - Board state to snapshot.
    ///
    /// # Returns
    /// A populated [`GpuDrcContext`] with netclass copper clearances attached to each primitive.
    pub fn from_board(board: &Board) -> Self {
        Self::build(board, false)
    }

    /// Extracts all pads, tracks, and vias with `clearance = 0.0` for physical copper
    /// short-circuit detection ([`validate_and_clean_copper`]).
    ///
    /// # Arguments
    /// * `board` - Board state to snapshot.
    ///
    /// # Returns
    /// A populated [`GpuDrcContext`] with zero clearance margins (detecting only direct copper overlap).
    pub fn from_board_with_zero_clearance(board: &Board) -> Self {
        Self::build(board, true)
    }

    /// Internal builder converting `board` pads, tracks, and vias into `#[repr(C)]` GPU obstacle buffers.
    ///
    /// # Arguments
    /// * `board` - Board state to snapshot.
    /// * `zero_clearance` - When `true`, sets all obstacle clearances to `0.0 mm`.
    ///
    /// # Returns
    /// A populated [`GpuDrcContext`].
    fn build(board: &Board, zero_clearance: bool) -> Self {
        let mut net_clearances = HashMap::with_capacity(board.netclasses.len() * 4 + 8);
        let default_clr = if zero_clearance {
            0.0f32
        } else {
            board.get_netclass_for_net(0).clearance as f32
        };

        let mut get_clr = |net_id: usize| -> f32 {
            if zero_clearance {
                return 0.0;
            }
            if net_id == 0 {
                return default_clr;
            }
            *net_clearances
                .entry(net_id)
                .or_insert_with(|| board.get_netclass_for_net(net_id).clearance as f32)
        };

        let mut pads = Vec::new();
        for fp in &board.footprints {
            let flipped = fp.layer != fp.orig_layer;
            for pad in &fp.pads {
                let pad_layer = fp.pad_layer(pad);
                let layer_mask = if pad_layer < 32 { 1u32 << pad_layer } else { 1u32 };
                let (px, py) = fp.pad_absolute_position(pad);
                let pad_rot = if flipped { -pad.rotation } else { pad.rotation };
                let total_rot = ((fp.rotation + pad_rot) % 360.0 + 360.0) % 360.0;
                let rot_rad = total_rot.to_radians() as f32;
                let shape_bits = match pad.shape.as_str() {
                    "circle" => 1u32 << 16,
                    "oval" => 2u32 << 16,
                    _ => 0u32,
                };
                let p_mask = shape_bits | if pad.is_through_hole { 0xFFFFu32 } else { layer_mask & 0xFFFFu32 };
                let clr = get_clr(pad.net_id);
                pads.push(GpuDrcPad {
                    x: px as f32,
                    y: py as f32,
                    hw: (pad.width * 0.5) as f32,
                    hh: (pad.height * 0.5) as f32,
                    rot_rad,
                    clearance: clr,
                    layer_mask: p_mask,
                    net_id: pad.net_id as u32,
                });
            }
        }

        let mut tracks = Vec::with_capacity(board.tracks.len());
        for t in &board.tracks {
            let clr = get_clr(t.net_id);
            tracks.push(GpuDrcTrack {
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

        let mut vias = Vec::with_capacity(board.vias.len());
        for v in &board.vias {
            let clr = get_clr(v.net_id);
            vias.push(GpuDrcVia {
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

        Self { pads, tracks, vias }
    }

    /// Evaluates a batch of candidate vias/segments on the GPU (preferring CUDA, falling back to Vulkan).
    ///
    /// # Arguments
    /// * `vulkan` - Optional Vulkan compute engine fallback when CUDA is unavailable.
    /// * `candidates` - Slice of candidate vias/segments to evaluate against `self.pads`, `self.tracks`, and `self.vias`.
    ///
    /// # Returns
    /// `Some(flags)` where `flags[i] != 0` indicates candidate `i` passed GPU DRC screening, or `None` if no GPU backend is active.
    pub fn evaluate(
        &self,
        vulkan: Option<&Arc<VulkanEngine>>,
        candidates: &[GpuDrcCandidate],
    ) -> Option<Vec<u32>> {
        if candidates.is_empty() {
            return Some(Vec::new());
        }
        if let Some(cuda) = CudaEngine::new() {
            if let Some(res) = cuda.evaluate_drc_candidates(&self.pads, &self.tracks, &self.vias, candidates) {
                return Some(res);
            }
        }
        if let Some(vk) = vulkan {
            return vk.evaluate_drc_candidates(&self.pads, &self.tracks, &self.vias, candidates);
        }
        None
    }
}

/// Computes the Euclidean distance in millimeters (`mm`) from a point `p` to a line segment `(a, b)`.
///
/// # Arguments
/// * `p` - Query point `(x, y)` in `mm`.
/// * `a` - Segment start `(x, y)` in `mm`.
/// * `b` - Segment end `(x, y)` in `mm`.
///
/// # Returns
/// Minimum Euclidean distance in `mm` from `p` to segment `ab`.
pub fn dist_point_to_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    dist_to_segment(p.0, p.1, a.0, a.1, b.0, b.1)
}

/// Computes the minimum Euclidean distance in millimeters (`mm`) from point `pt` to the copper area of `pad`.
///
/// Accounts for circular, oval (capsule), and rotated rectangular pad shapes as well as footprint
/// bottom-layer mirroring. Returns `0.0` if `pt` lies inside the pad's copper outline.
///
/// # Arguments
/// * `pt` - Query point `(x, y)` in `mm`.
/// * `fp` - Parent footprint providing world position, rotation, and layer flip state.
/// * `pad` - Pad whose copper shape is queried.
///
/// # Returns
/// Minimum distance in `mm` from `pt` to the pad copper boundary (`0.0` if inside).
pub fn dist_point_to_pad(pt: (f64, f64), fp: &Footprint, pad: &Pad) -> f64 {
    let (px, py) = fp.pad_absolute_position(pad);
    if pad.shape == "circle" {
        let r = pad.width.max(pad.height) * 0.5;
        let d = (pt.0 - px).hypot(pt.1 - py);
        return (d - r).max(0.0);
    }
    let flipped = fp.layer != fp.orig_layer;
    let pad_rot = if flipped { -pad.rotation } else { pad.rotation };
    let total_rot = ((fp.rotation + pad_rot) % 360.0 + 360.0) % 360.0;
    let rad = (-total_rot).to_radians();
    let rx = pt.0 - px;
    let ry = pt.1 - py;
    let lx = rx * rad.cos() - ry * rad.sin();
    let ly = rx * rad.sin() + ry * rad.cos();
    let hw = pad.width * 0.5;
    let hh = pad.height * 0.5;

    if pad.shape == "oval" {
        let r = hw.min(hh);
        if hw >= hh {
            let cap_dx = hw - r;
            let clamped_x = lx.clamp(-cap_dx, cap_dx);
            let d = (lx - clamped_x).hypot(ly);
            (d - r).max(0.0)
        } else {
            let cap_dy = hh - r;
            let clamped_y = ly.clamp(-cap_dy, cap_dy);
            let d = lx.hypot(ly - clamped_y);
            (d - r).max(0.0)
        }
    } else {
        let dx = (lx.abs() - hw).max(0.0);
        let dy = (ly.abs() - hh).max(0.0);
        dx.hypot(dy)
    }
}

/// Computes the minimum Euclidean distance in millimeters (`mm`) from line segment `(s, e)` to the copper area of `pad`.
///
/// Accounts for circular and rotated rectangular/oval pad shapes as well as footprint bottom-layer mirroring.
/// Returns `0.0` if segment `(s, e)` intersects or lies inside the pad's copper outline.
///
/// # Arguments
/// * `s` - Segment start `(x, y)` in `mm`.
/// * `e` - Segment end `(x, y)` in `mm`.
/// * `fp` - Parent footprint providing world position, rotation, and layer flip state.
/// * `pad` - Pad whose copper shape is queried.
///
/// # Returns
/// Minimum distance in `mm` from segment `(s, e)` to the pad copper boundary (`0.0` if intersecting).
pub fn dist_segment_to_pad(s: (f64, f64), e: (f64, f64), fp: &Footprint, pad: &Pad) -> f64 {
    let (px, py) = fp.pad_absolute_position(pad);
    if pad.shape == "circle" {
        let r = pad.width.max(pad.height) * 0.5;
        let d = dist_point_to_segment((px, py), s, e);
        return (d - r).max(0.0);
    }
    let flipped = fp.layer != fp.orig_layer;
    let pad_rot = if flipped { -pad.rotation } else { pad.rotation };
    let total_rot = ((fp.rotation + pad_rot) % 360.0 + 360.0) % 360.0;
    let rad = (-total_rot).to_radians();
    let cos_r = rad.cos();
    let sin_r = rad.sin();
    let to_local = |p: (f64, f64)| -> (f64, f64) {
        let rx = p.0 - px;
        let ry = p.1 - py;
        (rx * cos_r - ry * sin_r, rx * sin_r + ry * cos_r)
    };
    let ls = to_local(s);
    let le = to_local(e);
    let hw = pad.width * 0.5;
    let hh = pad.height * 0.5;

    if ls.0.abs() <= hw && ls.1.abs() <= hh {
        return 0.0;
    }
    if le.0.abs() <= hw && le.1.abs() <= hh {
        return 0.0;
    }

    let c1 = (-hw, -hh);
    let c2 = (hw, -hh);
    let c3 = (hw, hh);
    let c4 = (-hw, hh);

    if segments_intersect(ls, le, c1, c2)
        || segments_intersect(ls, le, c2, c3)
        || segments_intersect(ls, le, c3, c4)
        || segments_intersect(ls, le, c4, c1)
    {
        return 0.0;
    }

    let d1 = (ls.0.abs() - hw).max(0.0).hypot((ls.1.abs() - hh).max(0.0));
    let d2 = (le.0.abs() - hw).max(0.0).hypot((le.1.abs() - hh).max(0.0));
    let d3 = dist_point_to_segment(c1, ls, le);
    let d4 = dist_point_to_segment(c2, ls, le);
    let d5 = dist_point_to_segment(c3, ls, le);
    let d6 = dist_point_to_segment(c4, ls, le);
    d1.min(d2).min(d3).min(d4).min(d5).min(d6)
}

/// Verifies whether candidate via location `pos` is completely legal and clearance-free
/// against all foreign pads, tracks, vias, arcs, board boundary, and keepouts, as well as
/// preventing overlapping or violating clearance against same-net SMD pads and drill holes.
///
/// An optional `ignore_pos` can be supplied to skip drill and proximity collision with
/// an existing via undergoing repositioning or sliding.
///
/// # Arguments
/// * `board` - Board state with obstacles and netclasses.
/// * `pos` - Candidate via center `(x, y)` in millimeters (`mm`).
/// * `via_dia` - Via outer annular ring diameter in millimeters (`mm`).
/// * `net_id` - Net ID of the via.
/// * `clearance` - Minimum copper clearance in millimeters (`mm`).
/// * `ignore_pos` - Optional existing via coordinate `(x, y)` in `mm` to exclude from self-collision checks.
///
/// # Returns
/// `true` if placing a via of diameter `via_dia` at `pos` satisfies all copper, hole, and edge DRC rules.
pub fn is_via_location_legal_ignoring(
    board: &Board,
    pos: (f64, f64),
    via_dia: f64,
    net_id: usize,
    clearance: f64,
    ignore_pos: Option<(f64, f64)>,
) -> bool {
    let via_r = via_dia * 0.5;

    // 1. Board boundary & edge cuts clearance
    let edge_clr = board.get_netclass_for_net(net_id).board_edge_clearance;
    let req_edge_dist = via_r + edge_clr;
    if let Some(ref poly) = board.outline_polygon {
        let b = poly.aabb();
        if pos.0 < b.min_x + req_edge_dist
            || pos.0 > b.max_x - req_edge_dist
            || pos.1 < b.min_y + req_edge_dist
            || pos.1 > b.max_y - req_edge_dist
        {
            return false;
        }
        if poly.vertices.len() > 4 {
            let pt = Point::new(pos.0, pos.1);
            if !poly.contains_point(&pt) || poly.distance_to_point(&pt) < req_edge_dist - 1e-4 {
                return false;
            }
        }
    } else if !board.contains_point_in_boundary(&Point::new(pos.0, pos.1)) {
        return false;
    }

    // 2. Keepouts
    for ko in &board.keepouts {
        let r = via_r + clearance;
        if pos.0 >= ko.min_x - r && pos.0 <= ko.max_x + r && pos.1 >= ko.min_y - r && pos.1 <= ko.max_y + r {
            return false;
        }
    }

    let nc = board.get_netclass_for_net(net_id);
    let min_hole_clr = if clearance < 0.15 { 0.0 } else { nc.min_hole_clearance };
    let min_hole_to_hole = nc.min_hole_to_hole.max(0.25);
    let via_drill = if via_dia <= 0.35 {
        0.15
    } else if via_dia <= 0.48 {
        0.20
    } else {
        nc.via_drill.max(0.30)
    };
    let max_board_clr = board
        .netclasses
        .values()
        .map(|c| c.clearance)
        .fold(clearance, f64::max);

    // 3. Footprint pads (foreign pads on all layers, including net 0; and same-net SMD pads)
    for fp in &board.footprints {
        for pad in &fp.pads {
            let (px, py) = fp.pad_absolute_position(pad);
            let coarse_p = via_r + max_board_clr.max(min_hole_clr) + pad.width.max(pad.height);
            if (pos.0 - px).abs() > coarse_p || (pos.1 - py).abs() > coarse_p {
                continue;
            }
            let d = dist_point_to_pad(pos, fp, pad);
            if pad.net_id != net_id {
                let min_c = if fp.is_bga() && fp.bga_pitch().map_or(false, |p| p.0 < 0.5 || p.1 < 0.5) {
                    0.05
                } else {
                    0.12
                };
                let pad_c = clearance.max(board.get_netclass_for_net(pad.net_id).clearance).max(min_c);
                let req_r = if fp.is_bga() {
                    via_r + pad_c
                } else {
                    (via_r + pad_c).max(via_drill * 0.5 + min_hole_clr)
                };
                if d < req_r - 1e-4 {
                    return false;
                }
            } else if !pad.is_through_hole {
                if !board.allow_via_in_pad() {
                    // Same-net SMD pad: when allow_via_in_pad is false (default), via must NEVER overlap pad copper
                    let is_vippo = fp.is_bga() && via_dia <= 0.35 && (pos.0 - px).hypot(pos.1 - py) < 1e-3;
                    if !is_vippo && d < via_r - 1e-4 {
                        return false;
                    }
                }
            }
        }
    }

    // 4. Foreign tracks on all copper layers
    let min_c = if clearance < 0.12 { clearance } else { 0.12 };
    for trk in &board.tracks {
        if trk.net_id != net_id {
            let coarse_t = via_r + max_board_clr.max(min_hole_clr) + trk.width * 0.5;
            if pos.0 < trk.start.0.min(trk.end.0) - coarse_t
                || pos.0 > trk.start.0.max(trk.end.0) + coarse_t
                || pos.1 < trk.start.1.min(trk.end.1) - coarse_t
                || pos.1 > trk.start.1.max(trk.end.1) + coarse_t
            {
                continue;
            }
            let trk_c = clearance.max(board.get_netclass_for_net(trk.net_id).clearance).max(min_c);
            let d = dist_point_to_segment(pos, trk.start, trk.end);
            let req_r = (via_r + trk_c).max(via_drill * 0.5 + min_hole_clr);
            if d < req_r + trk.width * 0.5 - 1e-4 {
                return false;
            }
        }
    }

    // 5. Foreign vias
    for v in &board.vias {
        if let Some(ig) = ignore_pos {
            if (v.position.0 - ig.0).hypot(v.position.1 - ig.1) < 1e-4 {
                continue;
            }
        }
        if (pos.0 - v.position.0).hypot(pos.1 - v.position.1) < 1e-4 {
            continue; // self
        }
        if v.net_id != net_id {
            let coarse_v = via_r + v.size * 0.5 + max_board_clr.max(min_hole_clr);
            if (pos.0 - v.position.0).abs() > coarse_v || (pos.1 - v.position.1).abs() > coarse_v {
                continue;
            }
            let v_c = clearance.max(board.get_netclass_for_net(v.net_id).clearance).max(min_c);
            let d = (pos.0 - v.position.0).hypot(pos.1 - v.position.1);
            let req_d = (via_r + v.size * 0.5 + v_c)
                .max(via_drill * 0.5 + v.size * 0.5 + min_hole_clr)
                .max(via_r + v.drill * 0.5 + min_hole_clr);
            if d < req_d - 1e-4 {
                return false;
            }
        }
    }

    // 6. Hole-to-hole drill clearance against all through-hole pads (including same net)
    for fp in &board.footprints {
        for pad in &fp.pads {
            if pad.is_through_hole {
                if let Some(drill) = pad.drill {
                    if drill > 0.0 {
                        let (px, py) = fp.pad_absolute_position(pad);
                        let d = (pos.0 - px).hypot(pos.1 - py);
                        let min_hole_dist = (via_drill * 0.5) + (drill * 0.5) + min_hole_to_hole;
                        if d < min_hole_dist - 1e-4 {
                            return false;
                        }
                        if pad.net_id != net_id && d < via_r + (drill * 0.5) + min_hole_clr - 1e-4 {
                            return false;
                        }
                    }
                }
            }
        }
    }

    // 7. Hole-to-hole drill clearance against all other vias (including same net)
    for v in &board.vias {
        if let Some(ig) = ignore_pos {
            if (v.position.0 - ig.0).hypot(v.position.1 - ig.1) < 1e-4 {
                continue;
            }
        }
        if (pos.0 - v.position.0).hypot(pos.1 - v.position.1) < 1e-4 {
            continue; // self
        }
        let d = (pos.0 - v.position.0).hypot(pos.1 - v.position.1);
        let min_hole_dist = if via_dia <= 0.35 && v.drill <= 0.18 {
            (via_drill * 0.5) + (v.drill * 0.5) + 0.10
        } else {
            (via_drill * 0.5) + (v.drill * 0.5) + min_hole_to_hole
        };
        if d < min_hole_dist - 1e-4 {
            return false;
        }
    }

    // 8. Foreign arcs on all copper layers
    for arc in &board.arcs {
        if arc.net_id != net_id {
            let arc_c = clearance.max(board.get_netclass_for_net(arc.net_id).clearance).max(min_c);
            let req_r = (via_r + arc_c).max(via_drill * 0.5 + min_hole_clr) + arc.width * 0.5;
            for seg in arc.to_segments(8) {
                if dist_point_to_segment(pos, seg.start, seg.end) < req_r - 1e-4 {
                    return false;
                }
            }
        }
    }

    true
}

/// Verifies whether candidate via location `pos` is completely legal and clearance-free
/// against all foreign pads, tracks, vias, arcs, board boundary, and keepouts, as well as
/// preventing overlapping or violating clearance against same-net SMD pads and drill holes.
///
/// # Arguments
/// * `board` - Board state with obstacles and netclasses.
/// * `pos` - Candidate via center `(x, y)` in millimeters (`mm`).
/// * `via_dia` - Via outer annular ring diameter in millimeters (`mm`).
/// * `net_id` - Net ID of the via.
/// * `clearance` - Minimum copper clearance in millimeters (`mm`).
///
/// # Returns
/// `true` if placing a via of diameter `via_dia` at `pos` satisfies all copper, hole, and edge DRC rules.
pub fn is_via_location_legal(
    board: &Board,
    pos: (f64, f64),
    via_dia: f64,
    net_id: usize,
    clearance: f64,
) -> bool {
    is_via_location_legal_ignoring(board, pos, via_dia, net_id, clearance, None)
}

/// Verifies whether a candidate track segment is legal and clearance-free against foreign copper, drill holes, board edges, and keepouts.
///
/// # Arguments
/// * `board` - Board state with obstacles and netclasses.
/// * `start` - Segment start `(x, y)` in millimeters (`mm`).
/// * `end` - Segment end `(x, y)` in millimeters (`mm`).
/// * `width` - Trace width in millimeters (`mm`).
/// * `layer` - Copper layer index.
/// * `net_id` - Net ID of the segment.
/// * `clearance` - Minimum copper clearance in millimeters (`mm`).
///
/// # Returns
/// `true` if the segment satisfies all copper, drill-hole, keepout, and board-edge DRC rules.
pub fn is_track_segment_legal(
    board: &Board,
    start: (f64, f64),
    end: (f64, f64),
    width: f64,
    layer: usize,
    net_id: usize,
    clearance: f64,
) -> bool {
    let half_w = width * 0.5;
    let min_hole_clr = board.get_netclass_for_net(net_id).min_hole_clearance;
    let s_min_x = start.0.min(end.0);
    let s_max_x = start.0.max(end.0);
    let s_min_y = start.1.min(end.1);
    let s_max_y = start.1.max(end.1);

    // 1. Board boundary & edge cuts clearance
    let edge_clr = board.get_netclass_for_net(net_id).board_edge_clearance;
    let req_edge_dist = half_w + edge_clr;
    if let Some(ref poly) = board.outline_polygon {
        let b = poly.aabb();
        if s_min_x < b.min_x + req_edge_dist
            || s_max_x > b.max_x - req_edge_dist
            || s_min_y < b.min_y + req_edge_dist
            || s_max_y > b.max_y - req_edge_dist
        {
            return false;
        }
        if poly.vertices.len() > 4 {
            let p1 = Point::new(start.0, start.1);
            let p2 = Point::new(end.0, end.1);
            if !poly.contains_point(&p1) || !poly.contains_point(&p2)
                || poly.distance_to_segment(start, end) < req_edge_dist - 1e-4
            {
                return false;
            }
        }
    } else if !board.contains_point_in_boundary(&Point::new(start.0, start.1))
        || !board.contains_point_in_boundary(&Point::new(end.0, end.1))
    {
        return false;
    }

    // 2. Keepouts
    for ko in &board.keepouts {
        if !ko.layers.contains(&layer) {
            continue;
        }
        let r = half_w + clearance;
        if s_max_x >= ko.min_x - r && s_min_x <= ko.max_x + r && s_max_y >= ko.min_y - r && s_min_y <= ko.max_y + r {
            return false;
        }
    }

    let max_board_clr = board
        .netclasses
        .values()
        .map(|c| c.clearance)
        .fold(clearance.max(0.12), f64::max);

    // 3. Foreign pads (SMD on same layer, through-hole on all layers, including net 0)
    for fp in &board.footprints {
        let mut fp_origin_checked = false;
        let mut start_on_fp = false;
        let mut end_on_fp = false;
        for pad in &fp.pads {
            if pad.net_id != net_id && (pad.is_through_hole || fp.pad_layer(pad) == layer) {
                let (px, py) = fp.pad_absolute_position(pad);
                let max_dim = pad.width.max(pad.height);
                let coarse_p = half_w + max_board_clr.max(min_hole_clr) + max_dim;
                if px < s_min_x - coarse_p
                    || px > s_max_x + coarse_p
                    || py < s_min_y - coarse_p
                    || py > s_max_y + coarse_p
                {
                    continue;
                }
                if pad.is_through_hole {
                    if let Some(drill) = pad.drill {
                        if drill > 0.0
                            && dist_point_to_segment((px, py), start, end)
                                < half_w + drill * 0.5 + min_hole_clr - 1e-4
                        {
                            return false;
                        }
                    }
                }
                let pad_c = clearance.max(board.get_netclass_for_net(pad.net_id).clearance).max(0.12);
                let d = dist_segment_to_pad(start, end, fp, pad);
                let req_d = half_w + pad_c;
                if d < req_d - 1e-4 {
                    if !fp_origin_checked {
                        fp_origin_checked = true;
                        start_on_fp = fp.pads.iter().any(|p| {
                            let (opx, opy) = fp.pad_absolute_position(p);
                            (opx - start.0).hypot(opy - start.1) < 1e-3
                        });
                        end_on_fp = fp.pads.iter().any(|p| {
                            let (opx, opy) = fp.pad_absolute_position(p);
                            (opx - end.0).hypot(opy - end.1) < 1e-3
                        });
                    }
                    if start_on_fp || end_on_fp {
                        let d_orig = match (start_on_fp, end_on_fp) {
                            (true, true) => dist_point_to_pad(start, fp, pad).min(dist_point_to_pad(end, fp, pad)),
                            (true, false) => dist_point_to_pad(start, fp, pad),
                            (false, true) => dist_point_to_pad(end, fp, pad),
                            (false, false) => f64::INFINITY,
                        };
                        let limit = d_orig.min(req_d);
                        if d < limit - 1e-3 {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
            }
        }
    }

    // 4. Foreign tracks on same layer
    for trk in &board.tracks {
        if trk.net_id != net_id && trk.layer == layer {
            let coarse_t = half_w + trk.width * 0.5 + max_board_clr;
            if trk.start.0.max(trk.end.0) < s_min_x - coarse_t
                || trk.start.0.min(trk.end.0) > s_max_x + coarse_t
                || trk.start.1.max(trk.end.1) < s_min_y - coarse_t
                || trk.start.1.min(trk.end.1) > s_max_y + coarse_t
            {
                continue;
            }
            let trk_c = clearance.max(board.get_netclass_for_net(trk.net_id).clearance).max(0.12);
            let d = dist_segment_to_segment(start, end, trk.start, trk.end);
            if d < half_w + trk.width * 0.5 + trk_c - 1e-4 {
                return false;
            }
        }
    }

    // 5. Foreign vias on all layers
    for v in &board.vias {
        if v.net_id != net_id {
            let coarse_v = half_w + v.size * 0.5 + max_board_clr.max(min_hole_clr);
            if v.position.0 < s_min_x - coarse_v
                || v.position.0 > s_max_x + coarse_v
                || v.position.1 < s_min_y - coarse_v
                || v.position.1 > s_max_y + coarse_v
            {
                continue;
            }
            let v_c = clearance.max(board.get_netclass_for_net(v.net_id).clearance).max(0.12);
            let d = dist_point_to_segment(v.position, start, end);
            let req_v_r = (v.size * 0.5 + v_c).max(v.drill * 0.5 + min_hole_clr);
            if d < half_w + req_v_r - 1e-4 {
                return false;
            }
        }
    }

    // 6. Foreign arcs on same layer
    for arc in &board.arcs {
        if arc.net_id != net_id && arc.layer == layer {
            let arc_c = clearance.max(board.get_netclass_for_net(arc.net_id).clearance).max(0.12);
            let req = half_w + arc.width * 0.5 + arc_c - 1e-4;
            for seg in arc.to_segments(8) {
                if dist_segment_to_segment(start, end, seg.start, seg.end) < req {
                    return false;
                }
            }
        }
    }

    true
}

/// Splits unlocked same-net tracks at interior intersections, T-junctions, and pad/via centers,
/// bridges off-center pad touches, and deduplicates identical collinear segments.
///
/// # Arguments
/// * `board` - Mutable board state whose unlocked tracks are split and deduplicated.
///
/// # Side Effects
/// Replaces `board.tracks` with split and deduplicated segments.
pub(crate) fn split_and_dedup_same_net_tracks(board: &mut Board) {
    board.tracks.retain(|t| t.is_locked || (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1) > 1e-4);
    if board.tracks.is_empty() {
        return;
    }

    // Bridge any unlocked track endpoint that sits within 0.005..0.15mm of a same-net pad center
    // directly to the pad center when the pad is otherwise reached via a long (>0.25mm) stub,
    // so folded overshoot loops form explicit cycles for Steiner pruning.
    let mut pad_bridges = Vec::new();
    for fp in &board.footprints {
        for pad in &fp.pads {
            if pad.net_id == 0 {
                continue;
            }
            let (px, py) = fp.pad_absolute_position(pad);
            let has_long_center_stub = board.tracks.iter().any(|ot| {
                !ot.is_locked
                    && ot.net_id == pad.net_id
                    && (pad.is_through_hole || ot.layer == fp.layer)
                    && ((ot.start.0 - px).hypot(ot.start.1 - py) < 0.005
                        || (ot.end.0 - px).hypot(ot.end.1 - py) < 0.005)
                    && (ot.start.0 - ot.end.0).hypot(ot.start.1 - ot.end.1) > 0.25
            });
            if !has_long_center_stub {
                continue;
            }
            for t in &board.tracks {
                if !t.is_locked && t.net_id == pad.net_id && (pad.is_through_hole || t.layer == fp.layer) {
                    for pt in [t.start, t.end] {
                        let d = (pt.0 - px).hypot(pt.1 - py);
                        if d > 0.005 && d <= 0.15 {
                            pad_bridges.push(TrackSegment {
                                start: pt,
                                end: (px, py),
                                width: t.width,
                                layer: t.layer,
                                net_id: t.net_id,
                                is_locked: false,
                            });
                        }
                    }
                }
            }
        }
    }
    board.tracks.extend(pad_bridges);

    let orig_tracks = std::mem::take(&mut board.tracks);
    let n = orig_tracks.len();
    let mut split_tracks = Vec::with_capacity(n + 16);

    for i in 0..n {
        let t = &orig_tracks[i];
        if t.is_locked {
            split_tracks.push(t.clone());
            continue;
        }
        let s1 = t.start;
        let e1 = t.end;
        let dx1 = e1.0 - s1.0;
        let dy1 = e1.1 - s1.1;
        let len_sq = dx1 * dx1 + dy1 * dy1;
        let len = len_sq.sqrt();
        if len <= 1e-4 {
            continue;
        }

        let mut splits: Vec<(f64, (f64, f64))> = Vec::new();
        let try_project = |splits: &mut Vec<(f64, (f64, f64))>, pt: (f64, f64), max_perp: f64, snap_exact: bool| {
            let d_s = (pt.0 - s1.0).hypot(pt.1 - s1.1);
            let d_e = (pt.0 - e1.0).hypot(pt.1 - e1.1);
            if d_s <= 0.008 || d_e <= 0.008 {
                return;
            }
            let u = ((pt.0 - s1.0) * dx1 + (pt.1 - s1.1) * dy1) / len_sq;
            if u > 1e-4 && u < 1.0 - 1e-4 {
                let proj = (s1.0 + u * dx1, s1.1 + u * dy1);
                let perp = (pt.0 - proj.0).hypot(pt.1 - proj.1);
                if perp <= max_perp {
                    splits.push((u, if snap_exact && perp <= 0.02 { pt } else { proj }));
                }
            }
        };

        // 1. Same-net pads
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id == t.net_id && (pad.is_through_hole || fp.layer == t.layer) {
                    try_project(&mut splits, fp.pad_absolute_position(pad), 0.05, true);
                }
            }
        }
        // 2. Same-net vias
        for v in &board.vias {
            if v.net_id == t.net_id {
                try_project(&mut splits, v.position, 0.03, true);
            }
        }
        // 3. Same-net track endpoints & proper segment intersections on same layer
        for j in 0..n {
            if i == j {
                continue;
            }
            let o = &orig_tracks[j];
            if o.net_id != t.net_id || o.layer != t.layer {
                continue;
            }
            try_project(&mut splits, o.start, 0.012, true);
            try_project(&mut splits, o.end, 0.012, true);

            let s2 = o.start;
            let e2 = o.end;
            let dx2 = e2.0 - s2.0;
            let dy2 = e2.1 - s2.1;
            let cross = dx1 * dy2 - dy1 * dx2;
            if cross.abs() > 1e-7 {
                let u1 = ((s2.0 - s1.0) * dy2 - (s2.1 - s1.1) * dx2) / cross;
                let u2 = ((s2.0 - s1.0) * dy1 - (s2.1 - s1.1) * dx1) / cross;
                if u1 > 1e-4 && u1 < 1.0 - 1e-4 && u2 >= -1e-4 && u2 <= 1.0 + 1e-4 {
                    let ipt = (s1.0 + u1 * dx1, s1.1 + u1 * dy1);
                    if (ipt.0 - s1.0).hypot(ipt.1 - s1.1) > 0.005
                        && (ipt.0 - e1.0).hypot(ipt.1 - e1.1) > 0.005
                    {
                        splits.push((u1, ipt));
                    }
                }
            }
        }

        if splits.is_empty() {
            split_tracks.push(t.clone());
        } else {
            splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            let mut pts = Vec::with_capacity(splits.len() + 2);
            pts.push(s1);
            for (_, sp) in splits {
                let last = *pts.last().unwrap();
                if (sp.0 - last.0).hypot(sp.1 - last.1) > 0.004 && (sp.0 - e1.0).hypot(sp.1 - e1.1) > 0.004 {
                    pts.push(sp);
                }
            }
            pts.push(e1);
            for w in pts.windows(2) {
                if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                    split_tracks.push(TrackSegment {
                        start: w[0],
                        end: w[1],
                        width: t.width,
                        layer: t.layer,
                        net_id: t.net_id,
                        is_locked: false,
                    });
                }
            }
        }
    }

    // Deduplicate identical segments on the same (net_id, layer)
    let q_pt = |p: (f64, f64)| -> (i64, i64) {
        ((p.0 * 1000.0).round() as i64, (p.1 * 1000.0).round() as i64)
    };
    let mut deduped: Vec<TrackSegment> = Vec::with_capacity(split_tracks.len());
    let mut seen: HashMap<(usize, usize, (i64, i64), (i64, i64)), usize> = HashMap::new();
    for seg in split_tracks {
        let q1 = q_pt(seg.start);
        let q2 = q_pt(seg.end);
        if !seg.is_locked && q1 == q2 {
            continue;
        }
        let (qa, qb) = if q1 <= q2 { (q1, q2) } else { (q2, q1) };
        let key = (seg.net_id, seg.layer, qa, qb);
        if let Some(&existing_idx) = seen.get(&key) {
            if seg.is_locked || (!deduped[existing_idx].is_locked && seg.width > deduped[existing_idx].width) {
                deduped[existing_idx] = seg;
            }
        } else {
            seen.insert(key, deduped.len());
            deduped.push(seg);
        }
    }
    board.tracks = deduped;
}

/// Prunes redundant parallel tracks, multi-via duplicate bridges, and single-pad loops
/// by extracting a minimal-cost Steiner tree over each net's pads and plane vias.
///
/// # Arguments
/// * `board` - Mutable board state whose redundant net cycles are removed.
///
/// # Returns
/// `true` if at least one redundant track segment was pruned.
///
/// # Side Effects
/// Removes non-tree unlocked [`TrackSegment`] items from `board.tracks`.
pub(crate) fn prune_redundant_net_cycles(board: &mut Board) -> bool {
    if board.footprints.is_empty() || board.tracks.len() < 2 {
        return false;
    }
    let plane_nets = board.inner_plane_nets();
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        crate::board::is_gnd_net_name(net_name)
            || plane_nets.values().any(|&pn| pn == nid)
    };

    let mut net_ids: Vec<usize> = board.tracks.iter().map(|t| t.net_id).collect();
    net_ids.sort_unstable();
    net_ids.dedup();

    let mut drop_track = vec![false; board.tracks.len()];
    let mut any_dropped = false;

    for net_id in net_ids {
        if net_id == 0 || board.arcs.iter().any(|a| a.net_id == net_id) {
            continue;
        }
        let trk_indices: Vec<usize> = board
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.net_id == net_id)
            .map(|(i, _)| i)
            .collect();
        if trk_indices.len() < 2 {
            continue;
        }

        let mut nodes: Vec<(f64, f64, usize)> = Vec::new();
        let mut find_or_add = |pt: (f64, f64), layer: usize| -> usize {
            for (idx, n) in nodes.iter().enumerate() {
                if n.2 == layer && (n.0 - pt.0).hypot(n.1 - pt.1) < 0.018 {
                    return idx;
                }
            }
            nodes.push((pt.0, pt.1, layer));
            nodes.len() - 1
        };

        let mut trk_endpoints: Vec<(usize, usize)> = Vec::with_capacity(trk_indices.len());
        for &ti in &trk_indices {
            let t = &board.tracks[ti];
            let u = find_or_add(t.start, t.layer);
            let v = find_or_add(t.end, t.layer);
            trk_endpoints.push((u, v));
        }

        let num_geom_nodes = nodes.len();
        let mut adj: Vec<Vec<(usize, Option<usize>, f64)>> = vec![Vec::new(); num_geom_nodes];
        let mut micro_self_tracks: Vec<(usize, usize)> = Vec::new();
        for (k, &ti) in trk_indices.iter().enumerate() {
            let (u, v) = trk_endpoints[k];
            let t = &board.tracks[ti];
            let len = (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1);
            if u == v {
                if len <= 1e-4 && !t.is_locked {
                    drop_track[ti] = true;
                    any_dropped = true;
                } else {
                    micro_self_tracks.push((ti, u));
                }
                continue;
            }
            let w = if t.is_locked {
                0.0
            } else {
                len * (if t.layer == 0 { 1.0 } else { 1.25 })
            };
            adj[u].push((v, Some(ti), w));
            adj[v].push((u, Some(ti), w));
        }

        // Vias connect nodes across layers (skipping nodes that step directly to a closer same-layer node at the via)
        for v in board.vias.iter().filter(|v| v.net_id == net_id) {
            let mut v_nodes = Vec::new();
            for u in 0..num_geom_nodes {
                let du = (nodes[u].0 - v.position.0).hypot(nodes[u].1 - v.position.1);
                if du <= v.size * 0.5 + 0.05 {
                    let has_closer_same_layer_neighbor = adj[u].iter().any(|&(nb, edge_opt, _)| {
                        edge_opt.is_some()
                            && nb < num_geom_nodes
                            && nodes[nb].2 == nodes[u].2
                            && (nodes[nb].0 - v.position.0).hypot(nodes[nb].1 - v.position.1) + 1e-4 < du
                    });
                    if !has_closer_same_layer_neighbor {
                        v_nodes.push(u);
                    }
                }
            }
            for i in 0..v_nodes.len() {
                for j in (i + 1)..v_nodes.len() {
                    let a = v_nodes[i];
                    let b = v_nodes[j];
                    adj[a].push((b, None, 0.80));
                    adj[b].push((a, None, 0.80));
                }
            }
        }

        let mut terminals: Vec<usize> = Vec::new();
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id != net_id {
                    continue;
                }
                let (px, py) = fp.pad_absolute_position(pad);
                let mut touching: Vec<(usize, f64, f64)> = Vec::new();
                for u in 0..num_geom_nodes {
                    if pad.is_through_hole || nodes[u].2 == fp.layer {
                        let d_c = (nodes[u].0 - px).hypot(nodes[u].1 - py);
                        let d_p = dist_point_to_pad((nodes[u].0, nodes[u].1), fp, pad);
                        if d_c < 0.12 || d_p < 0.05 {
                            touching.push((u, d_c, d_p));
                        }
                    }
                }
                if !touching.is_empty() {
                    // Any SMD pad joins every track end on its copper (exits may be slid or
                    // corner-anchored by `align_pad_exits`, not just at the centre).
                    let is_smd_with_inpad_ends = !pad.is_through_hole && touching.iter().filter(|t| t.2 < 0.01).count() > 1;
                    let is_large_thermal = !pad.is_through_hole && pad.width >= 2.0 && pad.height >= 2.0;
                    if is_large_thermal || is_smd_with_inpad_ends {
                        let pad_node = adj.len();
                        adj.push(Vec::new());
                        for &(u, dc, dp) in &touching {
                            let cost = if dp < 0.01 { 0.0 } else { dc * 4.0 };
                            adj[pad_node].push((u, None, cost));
                            adj[u].push((pad_node, None, cost));
                        }
                        terminals.push(pad_node);
                    } else if pad.is_through_hole {
                        let pad_node = adj.len();
                        adj.push(Vec::new());
                        let min_dc = touching.iter().map(|t| t.1).fold(f64::INFINITY, f64::min);
                        for &(u, dc, _) in &touching {
                            if dc < 1e-3 || (dc - min_dc).abs() < 1e-4 {
                                adj[pad_node].push((u, None, dc * 4.0));
                                adj[u].push((pad_node, None, dc * 4.0));
                            }
                        }
                        terminals.push(pad_node);
                    } else if let Some(&(best_u, _, _)) = touching
                        .iter()
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                    {
                        terminals.push(best_u);
                    }
                }
            }
        }

        if is_plane_net(net_id) {
            for v in board.vias.iter().filter(|v| v.net_id == net_id) {
                if let Some((u, _)) = (0..num_geom_nodes)
                    .map(|u| (u, (nodes[u].0 - v.position.0).hypot(nodes[u].1 - v.position.1)))
                    .filter(|&(_, d)| d <= v.size * 0.5 + 0.05)
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                {
                    terminals.push(u);
                }
            }
        }

        for (k, &ti) in trk_indices.iter().enumerate() {
            if board.tracks[ti].is_locked {
                terminals.push(trk_endpoints[k].0);
                terminals.push(trk_endpoints[k].1);
            }
        }
        terminals.sort_unstable();
        terminals.dedup();
        if terminals.is_empty() {
            continue;
        }

        let total_n = adj.len();
        let mut comp_id = vec![usize::MAX; total_n];
        let mut comp_count = 0;
        for start in 0..total_n {
            if comp_id[start] != usize::MAX {
                continue;
            }
            let mut stack = vec![start];
            comp_id[start] = comp_count;
            while let Some(u) = stack.pop() {
                for &(nxt, _, _) in &adj[u] {
                    if comp_id[nxt] == usize::MAX {
                        comp_id[nxt] = comp_count;
                        stack.push(nxt);
                    }
                }
            }
            comp_count += 1;
        }

        let mut kept_tracks: HashSet<usize> = HashSet::new();
        for &ti in &trk_indices {
            if board.tracks[ti].is_locked {
                kept_tracks.insert(ti);
            }
        }

        for cid in 0..comp_count {
            let comp_terms: Vec<usize> = terminals.iter().copied().filter(|&t| comp_id[t] == cid).collect();
            if comp_terms.len() < 2 {
                for (k, &ti) in trk_indices.iter().enumerate() {
                    if comp_id[trk_endpoints[k].0] == cid {
                        kept_tracks.insert(ti);
                    }
                }
                continue;
            }
            let mut in_tree = vec![false; total_n];
            in_tree[comp_terms[0]] = true;

            loop {
                if comp_terms.iter().all(|&t| in_tree[t]) {
                    break;
                }
                let mut dist = vec![f64::INFINITY; total_n];
                let mut prev: Vec<Option<(usize, Option<usize>)>> = vec![None; total_n];
                let mut visited = vec![false; total_n];

                for u in 0..total_n {
                    if in_tree[u] {
                        dist[u] = 0.0;
                    }
                }

                loop {
                    let mut best_u = None;
                    let mut best_d = f64::INFINITY;
                    for u in 0..total_n {
                        if comp_id[u] == cid && !visited[u] && dist[u] < best_d {
                            best_d = dist[u];
                            best_u = Some(u);
                        }
                    }
                    let Some(u) = best_u else { break };
                    visited[u] = true;
                    for &(v, ti_opt, w) in &adj[u] {
                        let eff_w = if ti_opt.map_or(false, |ti| kept_tracks.contains(&ti)) { 0.0 } else { w };
                        if dist[u] + eff_w + 1e-9 < dist[v] {
                            dist[v] = dist[u] + eff_w;
                            prev[v] = Some((u, ti_opt));
                        }
                    }
                }

                let next_term = comp_terms
                    .iter()
                    .copied()
                    .filter(|&t| !in_tree[t] && dist[t].is_finite())
                    .min_by(|&a, &b| dist[a].partial_cmp(&dist[b]).unwrap_or(Ordering::Equal));
                let Some(mut cur) = next_term else { break };
                while !in_tree[cur] {
                    in_tree[cur] = true;
                    if let Some((p, ti_opt)) = prev[cur] {
                        if let Some(ti) = ti_opt {
                            kept_tracks.insert(ti);
                        }
                        cur = p;
                    } else {
                        break;
                    }
                }
            }
        }

        for &(ti, u_node) in &micro_self_tracks {
            let node_used = terminals.contains(&u_node)
                || trk_indices.iter().enumerate().any(|(k, &oti)| {
                    oti != ti && kept_tracks.contains(&oti) && (trk_endpoints[k].0 == u_node || trk_endpoints[k].1 == u_node)
                });
            if node_used {
                kept_tracks.insert(ti);
            }
        }

        for (k, &ti) in trk_indices.iter().enumerate() {
            let (u, _) = trk_endpoints[k];
            let has_terms = terminals.iter().any(|&t| comp_id[t] == comp_id[u]);
            if has_terms && !board.tracks[ti].is_locked && !kept_tracks.contains(&ti) {
                drop_track[ti] = true;
                any_dropped = true;
            }
        }
    }

    if any_dropped {
        let mut new_tracks = Vec::with_capacity(board.tracks.len());
        for (i, trk) in board.tracks.drain(..).enumerate() {
            if !drop_track[i] {
                new_tracks.push(trk);
            }
        }
        board.tracks = new_tracks;
    }
    any_dropped
}

/// Automated dangling track, arc, and via pruning.
///
/// Iteratively removes dead-end degree-1 track and arc segments (antennas), redundant cycles
/// (via `prune_redundant_net_cycles`), unconnected or single-layer vias, and floating copper
/// islands (via `prune_orphan_components`).
///
/// # Arguments
/// * `board` - Mutable board state whose dangling tracks, arcs, and vias are pruned.
///
/// # Side Effects
/// Mutates `board.tracks`, `board.arcs`, and `board.vias`.
pub fn prune_dangling_tracks_and_vias(board: &mut Board) {
    if board.footprints.is_empty() {
        return;
    }
    split_and_dedup_same_net_tracks(board);
    let plane_nets = board.inner_plane_nets();
    let is_plane_net = |b: &Board, nid: usize| -> bool {
        let net_name = b.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        crate::board::is_gnd_net_name(net_name)
            || plane_nets.values().any(|&pn| pn == nid)
    };

    const EPS: f64 = 0.05;

    for _outer in 0..2 {
        let mut changed = true;
        let mut pass = 0;
        while changed && pass < 50 {
            changed = false;
            pass += 1;

            let num_tracks = board.tracks.len();
            let num_arcs = board.arcs.len();
            let num_vias = board.vias.len();
            let mut pruned_tracks = vec![false; num_tracks];
            let mut pruned_arcs = vec![false; num_arcs];
            let mut pruned_vias = vec![false; num_vias];

            // 1. Identify track dead-ends
            for i in 0..num_tracks {
                let t = &board.tracks[i];
                if t.is_locked {
                    continue;
                }

                let check_endpoint = |pt: (f64, f64), other_pt: (f64, f64)| -> bool {
                    // A. Check footprint pads of matching net
                    for fp in &board.footprints {
                        for pad in &fp.pads {
                            if pad.net_id == t.net_id {
                                if pad.is_through_hole || fp.pad_layer(pad) == t.layer {
                                    let (px, py) = fp.pad_absolute_position(pad);
                                    let d_center_pt = (pt.0 - px).hypot(pt.1 - py);
                                    let d_center_other = (other_pt.0 - px).hypot(other_pt.1 - py);
                                    let d_pad_pt = dist_point_to_pad(pt, fp, pad);
                                    let d_pad_other = dist_point_to_pad(other_pt, fp, pad);
                                    if d_center_pt < 0.12 || d_pad_pt < EPS {
                                        let other_on_same_pad = d_center_other < 0.12
                                            || (d_pad_other < EPS && d_center_other >= 0.01);
                                        if !other_on_same_pad || d_center_pt + 1e-4 < d_center_other {
                                            return true;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // B. Check vias of matching net
                    for (v_idx, v) in board.vias.iter().enumerate() {
                        if v.net_id == t.net_id && !pruned_vias[v_idx] {
                            let d_pt = (pt.0 - v.position.0).hypot(pt.1 - v.position.1);
                            let d_other = (other_pt.0 - v.position.0).hypot(other_pt.1 - v.position.1);
                            if d_pt <= v.size * 0.5 + EPS && d_pt + 1e-4 < d_other {
                                return true;
                            }
                        }
                    }

                    // C. Check other tracks of matching net on same layer
                    for j in 0..num_tracks {
                        if i != j && !pruned_tracks[j] {
                            let o = &board.tracks[j];
                            if o.net_id == t.net_id && o.layer == t.layer {
                                let d_end_pt = (pt.0 - o.start.0).hypot(pt.1 - o.start.1).min((pt.0 - o.end.0).hypot(pt.1 - o.end.1));
                                let d_end_other = (other_pt.0 - o.start.0).hypot(other_pt.1 - o.start.1).min((other_pt.0 - o.end.0).hypot(other_pt.1 - o.end.1));
                                if d_end_pt < 0.018 && d_end_pt + 1e-4 < d_end_other {
                                    return true;
                                }
                                if o.is_locked {
                                    let d_pt = dist_point_to_segment(pt, o.start, o.end);
                                    let d_other = dist_point_to_segment(other_pt, o.start, o.end);
                                    if d_pt < EPS && d_pt + 1e-4 < d_other {
                                        return true;
                                    }
                                }
                            }
                        }
                    }

                    // D. Check arcs of matching net on same layer
                    for (a_idx, a) in board.arcs.iter().enumerate() {
                        if !pruned_arcs[a_idx] && a.net_id == t.net_id && a.layer == t.layer {
                            let as_pt = a.start();
                            let ae_pt = a.end();
                            let d_pt = (pt.0 - as_pt.0).hypot(pt.1 - as_pt.1).min((pt.0 - ae_pt.0).hypot(pt.1 - ae_pt.1));
                            let d_other = (other_pt.0 - as_pt.0).hypot(other_pt.1 - as_pt.1).min((other_pt.0 - ae_pt.0).hypot(other_pt.1 - ae_pt.1));
                            if d_pt < EPS && d_pt + 1e-4 < d_other {
                                return true;
                            }
                        }
                    }

                    false
                };

                let start_connected = check_endpoint(t.start, t.end);
                let end_connected = check_endpoint(t.end, t.start);

                if !start_connected || !end_connected {
                    pruned_tracks[i] = true;
                    changed = true;
                }
            }

            // 1b. Identify arc dead-ends
            for i in 0..num_arcs {
                let a = &board.arcs[i];
                if a.is_locked {
                    continue;
                }

                let check_arc_endpoint = |pt: (f64, f64)| -> bool {
                    for fp in &board.footprints {
                        for pad in &fp.pads {
                            if pad.net_id == a.net_id {
                                if pad.is_through_hole || fp.pad_layer(pad) == a.layer {
                                    let (px, py) = fp.pad_absolute_position(pad);
                                    if (pt.0 - px).hypot(pt.1 - py) < 0.12
                                        || dist_point_to_pad(pt, fp, pad) < EPS
                                    {
                                        return true;
                                    }
                                }
                            }
                        }
                    }

                    for (v_idx, v) in board.vias.iter().enumerate() {
                        if v.net_id == a.net_id && !pruned_vias[v_idx] {
                            if (pt.0 - v.position.0).hypot(pt.1 - v.position.1) <= v.size * 0.5 + EPS {
                                return true;
                            }
                        }
                    }

                    for j in 0..num_tracks {
                        if !pruned_tracks[j] {
                            let o = &board.tracks[j];
                            if o.net_id == a.net_id && o.layer == a.layer {
                                if (pt.0 - o.start.0).hypot(pt.1 - o.start.1) < EPS
                                    || (pt.0 - o.end.0).hypot(pt.1 - o.end.1) < EPS
                                    || dist_point_to_segment(pt, o.start, o.end) < EPS
                                {
                                    return true;
                                }
                            }
                        }
                    }

                    for j in 0..num_arcs {
                        if i != j && !pruned_arcs[j] {
                            let oa = &board.arcs[j];
                            if oa.net_id == a.net_id && oa.layer == a.layer {
                                let oas = oa.start();
                                let oae = oa.end();
                                if (pt.0 - oas.0).hypot(pt.1 - oas.1) < EPS
                                    || (pt.0 - oae.0).hypot(pt.1 - oae.1) < EPS
                                {
                                    return true;
                                }
                            }
                        }
                    }

                    false
                };

                if !check_arc_endpoint(a.start()) || !check_arc_endpoint(a.end()) {
                    pruned_arcs[i] = true;
                    changed = true;
                }
            }

            // 2. Identify dangling / single-layer vias
            for (v_idx, v) in board.vias.iter().enumerate() {
                if v.is_locked || pruned_vias[v_idx] {
                    continue;
                }

                let touches_pad = board.footprints.iter().any(|fp| {
                    fp.pads.iter().any(|p| {
                        if p.net_id == v.net_id {
                            let (px, py) = fp.pad_absolute_position(p);
                            (v.position.0 - px).hypot(v.position.1 - py) <= p.width.max(p.height) * 0.5 + EPS
                                || dist_point_to_pad(v.position, fp, p) < EPS
                        } else {
                            false
                        }
                    })
                });

                let has_plane = is_plane_net(board, v.net_id);

                let mut connected_tracks = 0;
                let mut connected_layers = HashSet::new();

                for (t_idx, t) in board.tracks.iter().enumerate() {
                    if t.net_id == v.net_id && !pruned_tracks[t_idx] {
                        let d_start = (t.start.0 - v.position.0).hypot(t.start.1 - v.position.1);
                        let d_end = (t.end.0 - v.position.0).hypot(t.end.1 - v.position.1);
                        if d_start <= v.size * 0.5 + EPS || d_end <= v.size * 0.5 + EPS {
                            connected_tracks += 1;
                            connected_layers.insert(t.layer);
                        }
                    }
                }

                for (a_idx, a) in board.arcs.iter().enumerate() {
                    if a.net_id == v.net_id && !pruned_arcs[a_idx] {
                        let as_pt = a.start();
                        let ae_pt = a.end();
                        let d_start = (as_pt.0 - v.position.0).hypot(as_pt.1 - v.position.1);
                        let d_end = (ae_pt.0 - v.position.0).hypot(ae_pt.1 - v.position.1);
                        if d_start <= v.size * 0.5 + EPS || d_end <= v.size * 0.5 + EPS {
                            connected_tracks += 1;
                            connected_layers.insert(a.layer);
                        }
                    }
                }

                let has_outward_reach = |layer: usize| -> bool {
                    let incident_to_v: Vec<usize> = board.tracks.iter().enumerate()
                        .filter(|&(idx, t)| {
                            t.net_id == v.net_id && !pruned_tracks[idx] && t.layer == layer && {
                                let d_s = (t.start.0 - v.position.0).hypot(t.start.1 - v.position.1);
                                let d_e = (t.end.0 - v.position.0).hypot(t.end.1 - v.position.1);
                                d_s <= v.size * 0.5 + EPS || d_e <= v.size * 0.5 + EPS
                            }
                        })
                        .map(|(idx, _)| idx)
                        .collect();

                    let incident_arcs_to_v: Vec<usize> = board.arcs.iter().enumerate()
                        .filter(|&(idx, a)| {
                            a.net_id == v.net_id && !pruned_arcs[idx] && a.layer == layer && {
                                let as_pt = a.start();
                                let ae_pt = a.end();
                                let d_s = (as_pt.0 - v.position.0).hypot(as_pt.1 - v.position.1);
                                let d_e = (ae_pt.0 - v.position.0).hypot(ae_pt.1 - v.position.1);
                                d_s <= v.size * 0.5 + EPS || d_e <= v.size * 0.5 + EPS
                            }
                        })
                        .map(|(idx, _)| idx)
                        .collect();

                    let check_other_pt_reach = |other_pt: (f64, f64)| -> bool {
                        let reaches_pad = board.footprints.iter().any(|fp| {
                            fp.pads.iter().any(|p| {
                                p.net_id == v.net_id && (p.is_through_hole || fp.pad_layer(p) == layer) && {
                                    let (px, py) = fp.pad_absolute_position(p);
                                    (other_pt.0 - px).hypot(other_pt.1 - py) < 0.12 || dist_point_to_pad(other_pt, fp, p) < EPS
                                }
                            })
                        });
                        if reaches_pad {
                            return true;
                        }

                        let reaches_other_via = board.vias.iter().enumerate().any(|(vi, ov)| {
                            vi != v_idx && ov.net_id == v.net_id && !pruned_vias[vi]
                                && (other_pt.0 - ov.position.0).hypot(other_pt.1 - ov.position.1) <= ov.size * 0.5 + EPS
                        });
                        if reaches_other_via {
                            return true;
                        }

                        let reaches_external_track = board.tracks.iter().enumerate().any(|(j, ot)| {
                            ot.net_id == v.net_id && ot.layer == layer && !pruned_tracks[j] && !incident_to_v.contains(&j) && {
                                (other_pt.0 - ot.start.0).hypot(other_pt.1 - ot.start.1) < 0.015
                                    || (other_pt.0 - ot.end.0).hypot(other_pt.1 - ot.end.1) < 0.015
                                    || (ot.is_locked && dist_point_to_segment(other_pt, ot.start, ot.end) < EPS)
                            }
                        });
                        if reaches_external_track {
                            return true;
                        }

                        let reaches_external_arc = board.arcs.iter().enumerate().any(|(j, oa)| {
                            oa.net_id == v.net_id && oa.layer == layer && !pruned_arcs[j] && !incident_arcs_to_v.contains(&j) && {
                                let oas = oa.start();
                                let oae = oa.end();
                                (other_pt.0 - oas.0).hypot(other_pt.1 - oas.1) < EPS
                                    || (other_pt.0 - oae.0).hypot(other_pt.1 - oae.1) < EPS
                            }
                        });
                        reaches_external_arc
                    };

                    for &t_idx in &incident_to_v {
                        let t = &board.tracks[t_idx];
                        let d_s = (t.start.0 - v.position.0).hypot(t.start.1 - v.position.1);
                        let other_pt = if d_s <= v.size * 0.5 + EPS { t.end } else { t.start };
                        if check_other_pt_reach(other_pt) {
                            return true;
                        }
                    }

                    for &a_idx in &incident_arcs_to_v {
                        let a = &board.arcs[a_idx];
                        let as_pt = a.start();
                        let ae_pt = a.end();
                        let d_s = (as_pt.0 - v.position.0).hypot(as_pt.1 - v.position.1);
                        let other_pt = if d_s <= v.size * 0.5 + EPS { ae_pt } else { as_pt };
                        if check_other_pt_reach(other_pt) {
                            return true;
                        }
                    }

                    false
                };

                let valid_connected_layers = connected_layers.iter().filter(|&&l| has_outward_reach(l)).count();

                if !touches_pad {
                    if has_plane {
                        let all_plane_layers = board.layer_plane_nets();
                        let plane_layer_count = all_plane_layers.values().filter(|&&pn| pn == v.net_id).count();
                        if plane_layer_count < 2 {
                            let valid_other_layers = connected_layers
                                .iter()
                                .filter(|&&l| {
                                    let is_plane_layer = all_plane_layers.get(&l).copied() == Some(v.net_id);
                                    !is_plane_layer && has_outward_reach(l)
                                })
                                .count();
                            if valid_other_layers == 0 {
                                pruned_vias[v_idx] = true;
                                changed = true;
                            }
                        }
                    } else if connected_tracks <= 1 || valid_connected_layers < 2 {
                        pruned_vias[v_idx] = true;
                        changed = true;
                    }
                }
            }

            if changed {
                let mut new_tracks = Vec::with_capacity(num_tracks);
                for (idx, trk) in board.tracks.drain(..).enumerate() {
                    if !pruned_tracks[idx] {
                        new_tracks.push(trk);
                    }
                }
                board.tracks = new_tracks;

                let mut new_arcs = Vec::with_capacity(num_arcs);
                for (idx, arc) in board.arcs.drain(..).enumerate() {
                    if !pruned_arcs[idx] {
                        new_arcs.push(arc);
                    }
                }
                board.arcs = new_arcs;

                let mut new_vias = Vec::with_capacity(num_vias);
                for (idx, v) in board.vias.drain(..).enumerate() {
                    if !pruned_vias[idx] {
                        new_vias.push(v);
                    }
                }
                board.vias = new_vias;
            }
        }

        if !prune_redundant_net_cycles(board) {
            break;
        }
    }

    // 3. Remove orphan components disconnected from any pad or plane
    prune_orphan_components(board);
}

/// Prunes disconnected islands of copper (tracks, arcs, and vias) that have no electrical path to any pad, locked primitive, or reference plane.
///
/// # Arguments
/// * `board` - Mutable board state whose floating copper islands are removed.
///
/// # Side Effects
/// Retains only pad-anchored or plane-anchored connected components in `board.tracks`, `board.arcs`, and `board.vias`.
pub(crate) fn prune_orphan_components(board: &mut Board) {
    if board.tracks.is_empty() && board.arcs.is_empty() {
        return;
    }
    let plane_nets = board.inner_plane_nets();
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        crate::board::is_gnd_net_name(net_name)
            || plane_nets.values().any(|&pn| pn == nid)
    };

    const EPS: f64 = 0.05;

    let mut net_ids: Vec<usize> = board
        .tracks
        .iter()
        .map(|t| t.net_id)
        .chain(board.arcs.iter().map(|a| a.net_id))
        .collect();
    net_ids.sort_unstable();
    net_ids.dedup();

    let mut keep_track = vec![true; board.tracks.len()];
    let mut keep_arc = vec![true; board.arcs.len()];
    let mut keep_via = vec![true; board.vias.len()];

    for net_id in net_ids {
        let track_indices: Vec<usize> = board
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| t.net_id == net_id)
            .map(|(i, _)| i)
            .collect();
        let arc_indices: Vec<usize> = board
            .arcs
            .iter()
            .enumerate()
            .filter(|(_, a)| a.net_id == net_id)
            .map(|(i, _)| i)
            .collect();
        let via_indices: Vec<usize> = board
            .vias
            .iter()
            .enumerate()
            .filter(|(_, v)| v.net_id == net_id)
            .map(|(i, _)| i)
            .collect();

        if track_indices.is_empty() && arc_indices.is_empty() {
            continue;
        }

        let pad_positions: Vec<(f64, f64, usize, bool)> = board
            .footprints
            .iter()
            .flat_map(|fp| {
                fp.pads
                    .iter()
                    .filter(|p| p.net_id == net_id)
                    .map(move |p| {
                        let (px, py) = fp.pad_absolute_position(p);
                        (px, py, fp.pad_layer(p), p.is_through_hole)
                    })
            })
            .collect();

        // Track ends may sit anywhere inside a pad (e.g. corner-anchored exits), not just at its centre.
        let on_pad = |pt: (f64, f64), layer: usize| {
            board.footprints.iter().any(|fp| {
                fp.pads.iter().any(|p| {
                    p.net_id == net_id && (p.is_through_hole || fp.pad_layer(p) == layer) && dist_point_to_pad(pt, fp, p) < 1e-3
                })
            })
        };
        let has_plane = is_plane_net(net_id);

        let n_t = track_indices.len();
        let n_a = arc_indices.len();
        let n_v = via_indices.len();
        let total_nodes = n_t + n_a + n_v;
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); total_nodes];

        for i in 0..n_t {
            let ti = &board.tracks[track_indices[i]];
            for j in (i + 1)..n_t {
                let tj = &board.tracks[track_indices[j]];
                if ti.layer == tj.layer {
                    let connected = (ti.start.0 - tj.start.0).hypot(ti.start.1 - tj.start.1) < EPS
                        || (ti.start.0 - tj.end.0).hypot(ti.start.1 - tj.end.1) < EPS
                        || (ti.end.0 - tj.start.0).hypot(ti.end.1 - tj.start.1) < EPS
                        || (ti.end.0 - tj.end.0).hypot(ti.end.1 - tj.end.1) < EPS
                        || dist_point_to_segment(ti.start, tj.start, tj.end) < EPS
                        || dist_point_to_segment(ti.end, tj.start, tj.end) < EPS
                        || dist_point_to_segment(tj.start, ti.start, ti.end) < EPS
                        || dist_point_to_segment(tj.end, ti.start, ti.end) < EPS;
                    if connected {
                        adj[i].push(j);
                        adj[j].push(i);
                    }
                }
            }
            for ai in 0..n_a {
                let a = &board.arcs[arc_indices[ai]];
                if ti.layer == a.layer {
                    let as_pt = a.start();
                    let ae_pt = a.end();
                    let node_a = n_t + ai;
                    let connected = (ti.start.0 - as_pt.0).hypot(ti.start.1 - as_pt.1) < EPS
                        || (ti.start.0 - ae_pt.0).hypot(ti.start.1 - ae_pt.1) < EPS
                        || (ti.end.0 - as_pt.0).hypot(ti.end.1 - as_pt.1) < EPS
                        || (ti.end.0 - ae_pt.0).hypot(ti.end.1 - ae_pt.1) < EPS
                        || dist_point_to_segment(as_pt, ti.start, ti.end) < EPS
                        || dist_point_to_segment(ae_pt, ti.start, ti.end) < EPS;
                    if connected {
                        adj[i].push(node_a);
                        adj[node_a].push(i);
                    }
                }
            }
        }

        for ai in 0..n_a {
            let a = &board.arcs[arc_indices[ai]];
            let as_pt = a.start();
            let ae_pt = a.end();
            let node_a = n_t + ai;
            for aj in (ai + 1)..n_a {
                let oa = &board.arcs[arc_indices[aj]];
                if a.layer == oa.layer {
                    let oas = oa.start();
                    let oae = oa.end();
                    let node_oa = n_t + aj;
                    let connected = (as_pt.0 - oas.0).hypot(as_pt.1 - oas.1) < EPS
                        || (as_pt.0 - oae.0).hypot(as_pt.1 - oae.1) < EPS
                        || (ae_pt.0 - oas.0).hypot(ae_pt.1 - oas.1) < EPS
                        || (ae_pt.0 - oae.0).hypot(ae_pt.1 - oae.1) < EPS;
                    if connected {
                        adj[node_a].push(node_oa);
                        adj[node_oa].push(node_a);
                    }
                }
            }
            for vi in 0..n_v {
                let v = &board.vias[via_indices[vi]];
                let node_v = n_t + n_a + vi;
                let d_s = (as_pt.0 - v.position.0).hypot(as_pt.1 - v.position.1);
                let d_e = (ae_pt.0 - v.position.0).hypot(ae_pt.1 - v.position.1);
                if d_s <= v.size * 0.5 + EPS || d_e <= v.size * 0.5 + EPS {
                    adj[node_a].push(node_v);
                    adj[node_v].push(node_a);
                }
            }
        }

        for i in 0..n_t {
            let ti = &board.tracks[track_indices[i]];
            for vi in 0..n_v {
                let v = &board.vias[via_indices[vi]];
                let node_v = n_t + n_a + vi;
                let d_s = (ti.start.0 - v.position.0).hypot(ti.start.1 - v.position.1);
                let d_e = (ti.end.0 - v.position.0).hypot(ti.end.1 - v.position.1);
                if d_s <= v.size * 0.5 + EPS || d_e <= v.size * 0.5 + EPS {
                    adj[i].push(node_v);
                    adj[node_v].push(i);
                }
            }
        }

        let mut is_anchor = vec![false; total_nodes];
        for i in 0..n_t {
            let ti = &board.tracks[track_indices[i]];
            if ti.is_locked {
                is_anchor[i] = true;
                continue;
            }
            for &(px, py, pad_layer, is_th) in &pad_positions {
                if is_th || pad_layer == ti.layer {
                    if (ti.start.0 - px).hypot(ti.start.1 - py) < 0.12
                        || (ti.end.0 - px).hypot(ti.end.1 - py) < 0.12
                        || dist_point_to_segment((px, py), ti.start, ti.end) < 0.05
                        || on_pad(ti.start, ti.layer)
                        || on_pad(ti.end, ti.layer)
                    {
                        is_anchor[i] = true;
                        break;
                    }
                }
            }
        }

        for ai in 0..n_a {
            let a = &board.arcs[arc_indices[ai]];
            let node_a = n_t + ai;
            if a.is_locked {
                is_anchor[node_a] = true;
                continue;
            }
            let as_pt = a.start();
            let ae_pt = a.end();
            for &(px, py, pad_layer, is_th) in &pad_positions {
                if is_th || pad_layer == a.layer {
                    if (as_pt.0 - px).hypot(as_pt.1 - py) < 0.12
                        || (ae_pt.0 - px).hypot(ae_pt.1 - py) < 0.12
                        || on_pad(as_pt, a.layer)
                        || on_pad(ae_pt, a.layer)
                    {
                        is_anchor[node_a] = true;
                        break;
                    }
                }
            }
        }

        for vi in 0..n_v {
            let v = &board.vias[via_indices[vi]];
            let node_v = n_t + n_a + vi;
            if v.is_locked || has_plane {
                is_anchor[node_v] = true;
                continue;
            }
            for &(px, py, _, _) in &pad_positions {
                if (v.position.0 - px).hypot(v.position.1 - py) <= v.size * 0.5 + 0.10 {
                    is_anchor[node_v] = true;
                    break;
                }
            }
        }

        let mut visited = vec![false; total_nodes];
        let mut queue = VecDeque::new();
        for node in 0..total_nodes {
            if is_anchor[node] {
                visited[node] = true;
                queue.push_back(node);
            }
        }

        while let Some(curr) = queue.pop_front() {
            for &nxt in &adj[curr] {
                if !visited[nxt] {
                    visited[nxt] = true;
                    queue.push_back(nxt);
                }
            }
        }

        for i in 0..n_t {
            if !visited[i] && !board.tracks[track_indices[i]].is_locked {
                keep_track[track_indices[i]] = false;
            }
        }
        for ai in 0..n_a {
            if !visited[n_t + ai] && !board.arcs[arc_indices[ai]].is_locked {
                keep_arc[arc_indices[ai]] = false;
            }
        }
        for vi in 0..n_v {
            if !visited[n_t + n_a + vi] && !board.vias[via_indices[vi]].is_locked {
                keep_via[via_indices[vi]] = false;
            }
        }
    }

    let mut new_tracks = Vec::with_capacity(board.tracks.len());
    for (i, t) in board.tracks.drain(..).enumerate() {
        if keep_track[i] {
            new_tracks.push(t);
        }
    }
    board.tracks = new_tracks;

    let mut new_arcs = Vec::with_capacity(board.arcs.len());
    for (i, a) in board.arcs.drain(..).enumerate() {
        if keep_arc[i] {
            new_arcs.push(a);
        }
    }
    board.arcs = new_arcs;

    let mut new_vias = Vec::with_capacity(board.vias.len());
    for (i, v) in board.vias.drain(..).enumerate() {
        if keep_via[i] {
            new_vias.push(v);
        }
    }
    board.vias = new_vias;
}

/// Comprehensive pre-export DRC validation and copper cleanup.
///
/// Enforces minimum via annular ring widths, adds short pad-entry stubs for shallow arcs touching
/// rectangular pads (working around KiCad 10's `SHAPE_ARC::Collide` quirk), detects and removes
/// any residual short-circuits or board-edge violations between tracks, vias, and pads of different
/// nets (using GPU batch screening when available), merges overlapping same-net vias, and prunes
/// resulting dangling stubs.
///
/// # Arguments
/// * `board` - Mutable board state to validate and clean before export.
///
/// # Side Effects
/// Mutates `board.raw_content`, `board.vias`, `board.tracks`, and `board.arcs`.
pub fn validate_and_clean_copper(board: &mut Board) {
    if let Some(nc_idx) = board.raw_content.find("\n  (net_class \"") {
        let max_fp_end = board
            .footprints
            .iter()
            .filter_map(|fp| fp.fp_span.map(|(_, e)| e))
            .max()
            .unwrap_or(0);
        if nc_idx >= max_fp_end {
            board.raw_content.truncate(nc_idx);
            board.raw_content.push_str("\n)\n");
        }
    }

    // Enforce board setup constraints for minimum via annular width on all emitted vias
    let min_annular = board.design_rules().min_via_annular_width;
    let min_drill = board.design_rules().min_via_drill;
    for v in &mut board.vias {
        if !v.is_locked && (v.size - v.drill) * 0.5 < min_annular - 1e-4 {
            if v.size - 2.0 * min_annular >= min_drill - 1e-4 {
                v.drill = (v.size - 2.0 * min_annular).max(min_drill);
            } else {
                v.size = v.drill + 2.0 * min_annular;
            }
        }
    }

    // KiCad 10 SHAPE_ARC::Collide(SHAPE_RECT, 0) quirk: shallow arcs terminating directly on a
    // rectangular pad center without a straight segment stub can fail pad-to-arc connectivity.
    if !board.arcs.is_empty() && !board.footprints.is_empty() {
        let mut pad_centers: Vec<((f64, f64), usize, usize, bool)> = Vec::new();
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id > 0 {
                    pad_centers.push((
                        fp.pad_absolute_position(pad),
                        pad.net_id,
                        fp.layer,
                        pad.is_through_hole,
                    ));
                }
            }
        }
        let mut stub_tracks = Vec::new();
        for arc in &mut board.arcs {
            if arc.radius <= 0.1 || (arc.end_angle_rad - arc.start_angle_rad).abs() * arc.radius <= 0.60 {
                continue;
            }
            let sign = if arc.end_angle_rad >= arc.start_angle_rad { 1.0 } else { -1.0 };
            let delta = 0.20 / arc.radius;
            let as_pt = arc.start();
            let start_on_pad = pad_centers.iter().any(|&(p, nid, lyr, th)| {
                nid == arc.net_id && (th || lyr == arc.layer) && (as_pt.0 - p.0).hypot(as_pt.1 - p.1) < 0.05
            });
            if start_on_pad {
                let has_track = board.tracks.iter().any(|t| {
                    t.net_id == arc.net_id
                        && t.layer == arc.layer
                        && ((t.start.0 - as_pt.0).hypot(t.start.1 - as_pt.1) < 0.02
                            || (t.end.0 - as_pt.0).hypot(t.end.1 - as_pt.1) < 0.02)
                });
                if !has_track {
                    arc.start_angle_rad += sign * delta;
                    let new_start = arc.start();
                    stub_tracks.push(TrackSegment {
                        start: as_pt,
                        end: new_start,
                        width: arc.width,
                        layer: arc.layer,
                        net_id: arc.net_id,
                        is_locked: arc.is_locked,
                    });
                }
            }
            let ae_pt = arc.end();
            let end_on_pad = pad_centers.iter().any(|&(p, nid, lyr, th)| {
                nid == arc.net_id && (th || lyr == arc.layer) && (ae_pt.0 - p.0).hypot(ae_pt.1 - p.1) < 0.05
            });
            if end_on_pad && (arc.end_angle_rad - arc.start_angle_rad).abs() * arc.radius > 0.40 {
                let has_track = board.tracks.iter().any(|t| {
                    t.net_id == arc.net_id
                        && t.layer == arc.layer
                        && ((t.start.0 - ae_pt.0).hypot(t.start.1 - ae_pt.1) < 0.02
                            || (t.end.0 - ae_pt.0).hypot(t.end.1 - ae_pt.1) < 0.02)
                });
                if !has_track {
                    arc.end_angle_rad -= sign * delta;
                    let new_end = arc.end();
                    stub_tracks.push(TrackSegment {
                        start: new_end,
                        end: ae_pt,
                        width: arc.width,
                        layer: arc.layer,
                        net_id: arc.net_id,
                        is_locked: arc.is_locked,
                    });
                }
            }
        }
        board.tracks.extend(stub_tracks);
    }

    let mut changed = true;
    let mut iteration = 0;
    while changed && iteration < 10 {
        changed = false;
        iteration += 1;

        let mut bad_track_indices = HashSet::new();
        let track_count = board.tracks.len();
        let via_count = board.vias.len();
        let drc_ctx = if track_count >= 16 || via_count > 0 {
            Some(GpuDrcContext::from_board_with_zero_clearance(board))
        } else {
            None
        };
        let gpu_track_ok: Option<Vec<u32>> = if track_count >= 16 {
            let cands: Vec<GpuDrcCandidate> = board
                .tracks
                .iter()
                .map(|t| GpuDrcCandidate {
                    x1: t.start.0 as f32,
                    y1: t.start.1 as f32,
                    x2: t.end.0 as f32,
                    y2: t.end.1 as f32,
                    radius_or_half_width: (t.width * 0.5) as f32,
                    clearance: 0.0,
                    layer: t.layer as u32,
                    net_id: t.net_id as u32,
                    kind: 1,
                    ignore_same_net_pads: 0,
                    _pad0: 0,
                    _pad1: 0,
                })
                .collect();
            drc_ctx.as_ref().and_then(|ctx| ctx.evaluate(None, &cands))
        } else {
            None
        };
        let gpu_via_ok: Option<Vec<u32>> = if via_count > 0 {
            let via_cands: Vec<GpuDrcCandidate> = board
                .vias
                .iter()
                .map(|v| GpuDrcCandidate {
                    x1: v.position.0 as f32,
                    y1: v.position.1 as f32,
                    x2: 0.0,
                    y2: 0.0,
                    radius_or_half_width: (v.size * 0.5) as f32,
                    clearance: 0.25,
                    layer: 0,
                    net_id: v.net_id as u32,
                    kind: 0,
                    ignore_same_net_pads: 0,
                    _pad0: 0,
                    _pad1: 0,
                })
                .collect();
            drc_ctx.as_ref().and_then(|ctx| ctx.evaluate(None, &via_cands))
        } else {
            None
        };

        for i in 0..track_count {
            if board.tracks[i].is_locked {
                continue;
            }
            let t1 = &board.tracks[i];

            // Copper-to-board-edge clearance check
            if let Some(ref poly) = board.outline_polygon {
                let touches_own_pad = board.footprints.iter().any(|fp| {
                    fp.pads.iter().any(|p| {
                        p.net_id == t1.net_id
                            && (p.is_through_hole || fp.layer == t1.layer)
                            && (dist_point_to_pad(t1.start, fp, p) < 0.05
                                || dist_point_to_pad(t1.end, fp, p) < 0.05)
                    })
                });
                let edge_clr = board.get_netclass_for_net(t1.net_id).board_edge_clearance;
                let req_dist = if touches_own_pad { 0.0 } else { t1.width * 0.5 + edge_clr };
                let b = poly.aabb();
                let s_min_x = t1.start.0.min(t1.end.0);
                let s_max_x = t1.start.0.max(t1.end.0);
                let s_min_y = t1.start.1.min(t1.end.1);
                let s_max_y = t1.start.1.max(t1.end.1);
                if s_min_x < b.min_x + req_dist
                    || s_max_x > b.max_x - req_dist
                    || s_min_y < b.min_y + req_dist
                    || s_max_y > b.max_y - req_dist
                {
                    bad_track_indices.insert(i);
                    continue;
                }
                if poly.vertices.len() > 4 {
                    let p1 = Point::new(t1.start.0, t1.start.1);
                    let p2 = Point::new(t1.end.0, t1.end.1);
                    if !poly.contains_point(&p1) || !poly.contains_point(&p2)
                        || (!touches_own_pad && poly.distance_to_segment(t1.start, t1.end) < req_dist - 1e-4)
                    {
                        bad_track_indices.insert(i);
                        continue;
                    }
                }
            }

            if gpu_track_ok.as_ref().map(|v| v[i] == 1).unwrap_or(false) {
                continue;
            }

            for j in 0..track_count {
                if i == j {
                    continue;
                }
                let t2 = &board.tracks[j];
                if t1.net_id != t2.net_id && t1.layer == t2.layer {
                    let short_dist = (t1.width + t2.width) * 0.5;
                    let d = dist_segment_to_segment(t1.start, t1.end, t2.start, t2.end);
                    if d < short_dist - 1e-4 {
                        bad_track_indices.insert(i);
                        break;
                    }
                }
            }
            if bad_track_indices.contains(&i) {
                continue;
            }

            for fp in &board.footprints {
                for pad in &fp.pads {
                    if pad.net_id != t1.net_id {
                        if pad.is_through_hole || fp.pad_layer(pad) == t1.layer {
                            let short_dist = t1.width * 0.5;
                            let d = dist_segment_to_pad(t1.start, t1.end, fp, pad);
                            if d < short_dist - 1e-4 {
                                bad_track_indices.insert(i);
                                break;
                            }
                        }
                    }
                }
                if bad_track_indices.contains(&i) {
                    break;
                }
            }
            if bad_track_indices.contains(&i) {
                continue;
            }

            for v in &board.vias {
                if v.net_id != t1.net_id {
                    let short_dist = (t1.width + v.size) * 0.5;
                    let d = dist_point_to_segment(v.position, t1.start, t1.end);
                    if d < short_dist - 1e-4 {
                        bad_track_indices.insert(i);
                        break;
                    }
                }
            }
        }

        let mut bad_via_indices = HashSet::new();
        for i in 0..via_count {
            if board.vias[i].is_locked {
                continue;
            }
            let v1 = &board.vias[i];

            // Copper-to-board-edge clearance check
            if let Some(ref poly) = board.outline_polygon {
                let edge_clr = board.get_netclass_for_net(v1.net_id).board_edge_clearance;
                let req_dist = v1.size * 0.5 + edge_clr;
                let b = poly.aabb();
                if v1.position.0 < b.min_x + req_dist
                    || v1.position.0 > b.max_x - req_dist
                    || v1.position.1 < b.min_y + req_dist
                    || v1.position.1 > b.max_y - req_dist
                {
                    bad_via_indices.insert(i);
                    continue;
                }
                if poly.vertices.len() > 4 {
                    let pt = Point::new(v1.position.0, v1.position.1);
                    if !poly.contains_point(&pt) || poly.distance_to_point(&pt) < req_dist - 1e-4 {
                        bad_via_indices.insert(i);
                        continue;
                    }
                }
            }

            let min_h2h = board.get_netclass_for_net(v1.net_id).min_hole_to_hole.max(0.25);
            if gpu_via_ok.as_ref().map(|v| v[i] == 1).unwrap_or(false) {
                let same_net_pth_conflict = board.footprints.iter().any(|fp| {
                    fp.pads.iter().any(|pad| {
                        pad.net_id == v1.net_id
                            && pad.is_through_hole
                            && pad.drill.is_some_and(|drill| {
                                if drill <= 0.0 {
                                    return false;
                                }
                                let (px, py) = fp.pad_absolute_position(pad);
                                let d = (v1.position.0 - px).hypot(v1.position.1 - py);
                                d < (v1.drill * 0.5) + (drill * 0.5) + 0.20 - 1e-4
                            })
                    })
                });
                let same_net_via_conflict = (0..via_count).any(|j| {
                    i != j
                        && !bad_via_indices.contains(&j)
                        && board.vias[j].net_id == v1.net_id
                        && {
                            let v2 = &board.vias[j];
                            let d = (v1.position.0 - v2.position.0).hypot(v1.position.1 - v2.position.1);
                            d < (v1.drill * 0.5) + (v2.drill * 0.5) + min_h2h - 1e-4
                        }
                });
                if !same_net_pth_conflict && !same_net_via_conflict {
                    continue;
                }
            }

            for fp in &board.footprints {
                for pad in &fp.pads {
                    let d = dist_point_to_pad(v1.position, fp, pad);
                    if pad.net_id != v1.net_id {
                        let short_dist = v1.size * 0.5;
                        if d < short_dist - 1e-4 {
                            bad_via_indices.insert(i);
                            break;
                        }
                    } else if !pad.is_through_hole {
                        let min_dist = v1.size * 0.5;
                        if d < min_dist - 1e-4 {
                            if !board.allow_via_in_pad() {
                                let is_vippo = fp.is_bga() && v1.size <= 0.35;
                                if !is_vippo {
                                    bad_via_indices.insert(i);
                                    break;
                                }
                            } else {
                                let pad_layer = fp.pad_layer(pad);
                                let has_other_layer_track = board.tracks.iter().any(|t| {
                                    t.net_id == v1.net_id
                                        && t.layer != pad_layer
                                        && ((t.start.0 - v1.position.0).hypot(t.start.1 - v1.position.1) <= v1.size * 0.5 + 0.15
                                            || (t.end.0 - v1.position.0).hypot(t.end.1 - v1.position.1) <= v1.size * 0.5 + 0.15)
                                });
                                if !has_other_layer_track {
                                    bad_via_indices.insert(i);
                                    break;
                                }
                            }
                        }
                    }
                }
                if bad_via_indices.contains(&i) {
                    break;
                }
            }
            if bad_via_indices.contains(&i) {
                continue;
            }

            for fp in &board.footprints {
                for pad in &fp.pads {
                    if pad.is_through_hole {
                        if let Some(drill) = pad.drill {
                            if drill > 0.0 {
                                let (px, py) = fp.pad_absolute_position(pad);
                                let d = (v1.position.0 - px).hypot(v1.position.1 - py);
                                let min_hole_dist = (v1.drill * 0.5) + (drill * 0.5) + 0.20;
                                if d < min_hole_dist - 1e-4 {
                                    bad_via_indices.insert(i);
                                    break;
                                }
                            }
                        }
                    }
                }
                if bad_via_indices.contains(&i) {
                    break;
                }
            }
            if bad_via_indices.contains(&i) {
                continue;
            }

            for j in 0..via_count {
                if i == j || bad_via_indices.contains(&j) {
                    continue;
                }
                let v2 = &board.vias[j];
                let h2h_thresh = if v1.net_id == v2.net_id { min_h2h } else { 0.20 };
                let min_hole_dist = (v1.drill * 0.5) + (v2.drill * 0.5) + h2h_thresh;
                let d = (v1.position.0 - v2.position.0).hypot(v1.position.1 - v2.position.1);
                if d < min_hole_dist - 1e-4 {
                    if v1.net_id == v2.net_id && i < j {
                        // Snap tracks connected to v2 onto v1 using octilinear doglegs and mark only v2 for removal
                        let pos_keep = v1.position;
                        let pos_drop = v2.position;
                        let drop_r = v2.size * 0.5 + 0.05;
                        let nid = v1.net_id;
                        let trk_clr = board.get_netclass_for_net(nid).clearance.max(0.15);
                        let mut to_replace = Vec::new();
                        for (t_idx, trk) in board.tracks.iter().enumerate() {
                            if trk.net_id != nid || trk.is_locked {
                                continue;
                            }
                            let s_drop = (trk.start.0 - pos_drop.0).hypot(trk.start.1 - pos_drop.1) <= drop_r;
                            let e_drop = (trk.end.0 - pos_drop.0).hypot(trk.end.1 - pos_drop.1) <= drop_r;
                            if s_drop || e_drop {
                                let other_pt = if s_drop { trk.end } else { trk.start };
                                to_replace.push((t_idx, other_pt, trk.width, trk.layer));
                            }
                        }
                        let mut new_segs = Vec::new();
                        for &(_, other_pt, width, layer) in &to_replace {
                            if (other_pt.0 - pos_keep.0).hypot(other_pt.1 - pos_keep.1) <= 1e-4 {
                                continue;
                            }
                            let other_on_pad = board.footprints.iter().any(|fp| {
                                fp.pads.iter().any(|p| {
                                    let (px, py) = fp.pad_absolute_position(p);
                                    (px - other_pt.0).hypot(py - other_pt.1) < 0.08
                                })
                            });
                            let postures = if other_on_pad { [true, false] } else { [false, true] };
                            let dl = postures
                                .into_iter()
                                .map(|df| compute_45_dogleg(pos_keep, other_pt, df))
                                .find(|cand| {
                                    cand.windows(2).all(|w| {
                                        (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                            || is_track_segment_legal(board, w[0], w[1], width, layer, nid, trk_clr)
                                    })
                                })
                                .unwrap_or_else(|| compute_45_dogleg(pos_keep, other_pt, other_on_pad));
                            for w in dl.windows(2) {
                                if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                    new_segs.push(TrackSegment {
                                        start: w[0],
                                        end: w[1],
                                        width,
                                        layer,
                                        net_id: nid,
                                        is_locked: false,
                                    });
                                }
                            }
                        }
                        for &(t_idx, _, _, _) in to_replace.iter().rev() {
                            board.tracks.remove(t_idx);
                        }
                        board.tracks.extend(new_segs);
                        bad_via_indices.insert(j);
                        continue;
                    } else if i > j {
                        bad_via_indices.insert(i);
                        break;
                    }
                }
                if v1.net_id != v2.net_id {
                    let short_dist = (v1.size + v2.size) * 0.5;
                    if d < short_dist - 1e-4 {
                        bad_via_indices.insert(i);
                        break;
                    }
                }
            }
        }

        if !bad_track_indices.is_empty() {
            let mut new_tracks = Vec::with_capacity(board.tracks.len() - bad_track_indices.len());
            for (idx, trk) in board.tracks.drain(..).enumerate() {
                if !bad_track_indices.contains(&idx) {
                    new_tracks.push(trk);
                }
            }
            board.tracks = new_tracks;
            changed = true;
        }

        if !bad_via_indices.is_empty() {
            let mut new_vias = Vec::with_capacity(board.vias.len() - bad_via_indices.len());
            for (idx, v) in board.vias.drain(..).enumerate() {
                if !bad_via_indices.contains(&idx) {
                    new_vias.push(v);
                }
            }
            board.vias = new_vias;
            changed = true;
        }

        prune_dangling_tracks_and_vias(board);
    }
}

/// Signal-integrity (SI) violation report indicating that parallel trace coupling between two nets exceeds the permitted threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CouplingViolation {
    /// Net ID of the first coupled net (smaller net ID).
    pub net_a: usize,
    /// Net ID of the second coupled net (larger net ID).
    pub net_b: usize,
    /// Schematic net name of `net_a`.
    pub net_a_name: String,
    /// Schematic net name of `net_b`.
    pub net_b_name: String,
    /// Total accumulated parallel coupled length in millimeters (`mm`).
    pub coupled_length_mm: f64,
    /// Maximum allowed parallel coupled length in millimeters (`mm`) from the stricter netclass.
    pub max_allowed_mm: f64,
    /// Bounding-box minimum corner `(min_x, min_y)` in millimeters (`mm`) of the coupled span.
    pub span_start: (f64, f64),
    /// Bounding-box maximum corner `(max_x, max_y)` in millimeters (`mm`) of the coupled span.
    pub span_end: (f64, f64),
}

/// Evaluates post-routing parallel trace coupling between every non-differential net pair on each copper layer.
///
/// Uses an LBVH spatial index (`build_gpu_lbvh`) to accelerate broad-phase pair queries, identifies
/// parallel runs where centerline separation satisfies
/// $d \le \frac{W_a + W_b}{2} + 2 \max(C_a, C_b) + 0.05\text{ mm}$,
/// accumulates total coupled length per net pair, and emits warnings and [`CouplingViolation`] records
/// when exceeding `max_coupled_length_mm`.
///
/// # Arguments
/// * `board` - Board state whose routed tracks and netclasses are analyzed.
///
/// # Returns
/// Vector of [`CouplingViolation`] records for all net pairs exceeding their coupled length budget.
pub fn validate_crosstalk_and_coupling(board: &Board) -> Vec<CouplingViolation> {
    let mut violations = Vec::new();
    let track_count = board.tracks.len();
    if track_count == 0 {
        return violations;
    }

    let max_w = board
        .tracks
        .iter()
        .map(|t| t.width)
        .fold(0.25_f64, f64::max);
    let max_clr = board
        .netclasses
        .values()
        .map(|nc| nc.clearance)
        .fold(board.get_netclass_for_net(0).clearance, f64::max);
    let max_coupling_thresh = max_w + 2.0 * max_clr + 0.05;
    let aabb_margin = (2.0 * max_coupling_thresh + 0.05) as f32;

    let prims: Vec<([f32; 4], u32, u32)> = board
        .tracks
        .iter()
        .map(|t| {
            let min_x = t.start.0.min(t.end.0) as f32;
            let min_y = t.start.1.min(t.end.1) as f32;
            let max_x = t.start.0.max(t.end.0) as f32;
            let max_y = t.start.1.max(t.end.1) as f32;
            let layer_mask = if t.layer < 32 { 1u32 << t.layer } else { 1u32 };
            ([min_x, min_y, max_x, max_y], layer_mask, t.net_id as u32)
        })
        .collect();
    let bvh_nodes = crate::vulkan::build_gpu_lbvh(&prims);

    let per_track_spans: Vec<Vec<(usize, (usize, usize), f64, f64, f64, f64, f64)>> = (0..track_count)
        .into_par_iter()
        .map(|i| {
            let t1 = &board.tracks[i];
            if t1.net_id == 0 {
                return Vec::new();
            }
            let v1_x = t1.end.0 - t1.start.0;
            let v1_y = t1.end.1 - t1.start.1;
            let len1 = v1_x.hypot(v1_y);
            if len1 < 1e-4 {
                return Vec::new();
            }
            let u1 = (v1_x / len1, v1_y / len1);
            let n1 = (-u1.1, u1.0);
            let nc1 = board.get_netclass_for_net(t1.net_id);
            let n1_name = board.nets.get(t1.net_id).map(|s| s.as_str()).unwrap_or("");

            let q_min_x = (t1.start.0.min(t1.end.0) as f32) - aabb_margin;
            let q_min_y = (t1.start.1.min(t1.end.1) as f32) - aabb_margin;
            let q_max_x = (t1.start.0.max(t1.end.0) as f32) + aabb_margin;
            let q_max_y = (t1.start.1.max(t1.end.1) as f32) + aabb_margin;
            let t1_layer_mask = if t1.layer < 32 { 1u32 << t1.layer } else { 1u32 };
            let t1_net_u32 = t1.net_id as u32;

            let mut hits = Vec::new();
            let mut stack = [0usize; 64];
            let mut sp = 1usize;
            while sp > 0 {
                sp -= 1;
                let node = &bvh_nodes[stack[sp]];
                if node.right_or_count == 0 {
                    continue;
                }
                if (node.layer_mask & t1_layer_mask) == 0 || node.net_id == t1_net_u32 {
                    continue;
                }
                if node.max_x < q_min_x
                    || node.min_x > q_max_x
                    || node.max_y < q_min_y
                    || node.min_y > q_max_y
                {
                    continue;
                }
                if node.right_or_count > 0 {
                    let j = node.left_or_prim as usize;
                    if j <= i {
                        continue;
                    }
                    let t2 = &board.tracks[j];
                    if t1.net_id == t2.net_id || t1.layer != t2.layer || t2.net_id == 0 {
                        continue;
                    }
                    let n2_name = board.nets.get(t2.net_id).map(|s| s.as_str()).unwrap_or("");
                    if AutoRouter::are_diff_pair_mates(n1_name, n2_name) {
                        continue;
                    }

                    let v2_x = t2.end.0 - t2.start.0;
                    let v2_y = t2.end.1 - t2.start.1;
                    let len2 = v2_x.hypot(v2_y);
                    if len2 < 1e-4 {
                        continue;
                    }
                    let u2 = (v2_x / len2, v2_y / len2);
                    let dot = (u1.0 * u2.0 + u1.1 * u2.1).abs();
                    if dot < 0.99 {
                        continue;
                    }

                    let nc2 = board.get_netclass_for_net(t2.net_id);
                    let coupling_thresh =
                        (t1.width + t2.width) * 0.5 + 2.0 * nc1.clearance.max(nc2.clearance) + 0.05;

                    let d_start =
                        ((t2.start.0 - t1.start.0) * n1.0 + (t2.start.1 - t1.start.1) * n1.1).abs();
                    let d_end =
                        ((t2.end.0 - t1.start.0) * n1.0 + (t2.end.1 - t1.start.1) * n1.1).abs();
                    let d_perp = (d_start + d_end) * 0.5;
                    if d_perp > coupling_thresh {
                        continue;
                    }

                    let p_start =
                        (t2.start.0 - t1.start.0) * u1.0 + (t2.start.1 - t1.start.1) * u1.1;
                    let p_end = (t2.end.0 - t1.start.0) * u1.0 + (t2.end.1 - t1.start.1) * u1.1;
                    let t2_min = p_start.min(p_end);
                    let t2_max = p_start.max(p_end);

                    let overlap_start = 0.0f64.max(t2_min);
                    let overlap_end = len1.min(t2_max);

                    if overlap_end > overlap_start + 1e-4 {
                        let coupled_len = overlap_end - overlap_start;
                        let pair_key = if t1.net_id < t2.net_id {
                            (t1.net_id, t2.net_id)
                        } else {
                            (t2.net_id, t1.net_id)
                        };
                        let pt_a = (
                            t1.start.0 + overlap_start * u1.0,
                            t1.start.1 + overlap_start * u1.1,
                        );
                        let pt_b = (
                            t1.start.0 + overlap_end * u1.0,
                            t1.start.1 + overlap_end * u1.1,
                        );
                        hits.push((
                            j,
                            pair_key,
                            coupled_len,
                            pt_a.0.min(pt_b.0),
                            pt_a.1.min(pt_b.1),
                            pt_a.0.max(pt_b.0),
                            pt_a.1.max(pt_b.1),
                        ));
                    }
                } else if sp + 2 <= stack.len() {
                    stack[sp] = (-node.right_or_count) as usize;
                    stack[sp + 1] = node.left_or_prim as usize;
                    sp += 2;
                }
            }
            hits.sort_unstable_by_key(|h| h.0);
            hits
        })
        .collect();

    let mut coupling_map: HashMap<(usize, usize), (f64, f64, f64, f64, f64)> = HashMap::new();
    for spans in per_track_spans {
        for (_, pair_key, coupled_len, min_x, min_y, max_x, max_y) in spans {
            let entry = coupling_map
                .entry(pair_key)
                .or_insert((0.0, min_x, min_y, max_x, max_y));
            entry.0 += coupled_len;
            entry.1 = entry.1.min(min_x);
            entry.2 = entry.2.min(min_y);
            entry.3 = entry.3.max(max_x);
            entry.4 = entry.4.max(max_y);
        }
    }

    for (&(net_a, net_b), &(tot_len, min_x, min_y, max_x, max_y)) in &coupling_map {
        let nc_a = board.get_netclass_for_net(net_a);
        let nc_b = board.get_netclass_for_net(net_b);
        let max_allowed = nc_a.max_coupled_length_mm.min(nc_b.max_coupled_length_mm);

        if tot_len > max_allowed + 1e-3 {
            let net_a_name = board.nets.get(net_a).cloned().unwrap_or_else(|| format!("Net{}", net_a));
            let net_b_name = board.nets.get(net_b).cloned().unwrap_or_else(|| format!("Net{}", net_b));
            let span_start = (min_x, min_y);
            let span_end = (max_x, max_y);

            eprintln!(
                "[SI WARNING] Crosstalk coupled length between '{}' (net {}) and '{}' (net {}) is {:.2} mm (max allowed: {:.2} mm) spanning ({:.2}, {:.2}) to ({:.2}, {:.2})",
                net_a_name, net_a, net_b_name, net_b, tot_len, max_allowed, span_start.0, span_start.1, span_end.0, span_end.1
            );

            violations.push(CouplingViolation {
                net_a,
                net_b,
                net_a_name,
                net_b_name,
                coupled_length_mm: tot_len,
                max_allowed_mm: max_allowed,
                span_start,
                span_end,
            });
        }
    }

    violations
}

/// Generates the parallel equidistant [`TrackSegment`] lanes for all nets in a [`BusBundle`] along its guide polyline.
///
/// Handles corner mitering (concentric 45° chamfers) and terminal pad fan-in/fan-out doglegs.
///
/// # Arguments
/// * `board` - Board state used to locate terminal footprint pads for each net in the bundle.
/// * `bundle` - Bus bundle definition containing net IDs, guide polyline, layer, and track width.
/// * `spacing` - Edge-to-edge copper spacing in millimeters (`mm`) between adjacent lanes in the bundle.
///
/// # Returns
/// Vector of generated [`TrackSegment`] primitives for all lanes in the bundle.
pub fn generate_bundle_tracks(
    board: &Board,
    bundle: &BusBundle,
    spacing: f64,
) -> Vec<TrackSegment> {
    if bundle.guide_points.len() < 2 || bundle.net_ids.is_empty() {
        return Vec::new();
    }

    // 1. Sanitize raw guide points by removing adjacent duplicate/micro points
    let mut raw_points: Vec<(f64, f64)> = Vec::new();
    for &pt in &bundle.guide_points {
        if let Some(&last) = raw_points.last() {
            if (pt.0 - last.0).hypot(pt.1 - last.1) < 1e-4 {
                continue;
            }
        }
        raw_points.push(pt);
    }
    if raw_points.len() < 2 {
        return Vec::new();
    }

    let m = bundle.net_ids.len();
    let half_span = (m as f64 - 1.0) * 0.5;
    let pitch = bundle.track_width + spacing;
    let o_max = half_span * pitch;

    // 2. Corner beveling on guide polyline:
    // Any turn sharper than ~45° (cos_theta < 0.707) is beveled with 45° miter chamfers
    let mut beveled_guide: Vec<(f64, f64)> = Vec::new();
    beveled_guide.push(raw_points[0]);

    for j in 1..raw_points.len() - 1 {
        let prev = raw_points[j - 1];
        let curr = raw_points[j];
        let next = raw_points[j + 1];

        let v1 = (curr.0 - prev.0, curr.1 - prev.1);
        let v2 = (next.0 - curr.0, next.1 - curr.1);
        let l1 = v1.0.hypot(v1.1);
        let l2 = v2.0.hypot(v2.1);

        if l1 < 1e-4 || l2 < 1e-4 {
            continue;
        }

        let u1 = (v1.0 / l1, v1.1 / l1);
        let u2 = (v2.0 / l2, v2.1 / l2);

        let cos_theta = u1.0 * u2.0 + u1.1 * u2.1;

        if cos_theta > 0.9999 {
            // Collinear, omit redundant joint
            continue;
        }

        // If turn angle is sharp (> ~45°, including 90° corners):
        if cos_theta < 0.707 {
            // Setback needed so even the innermost track has a positive-length 45° chamfer
            let min_sb = o_max * (2.0 - std::f64::consts::SQRT_2) + 1.2 * pitch;
            let sb = min_sb.min(0.45 * l1).min(0.45 * l2);

            if sb > 0.05 {
                let p_a = (curr.0 - sb * u1.0, curr.1 - sb * u1.1);
                let p_b = (curr.0 + sb * u2.0, curr.1 + sb * u2.1);
                beveled_guide.push(p_a);
                beveled_guide.push(p_b);
                continue;
            }
        }

        beveled_guide.push(curr);
    }
    beveled_guide.push(*raw_points.last().unwrap());

    if beveled_guide.len() < 2 {
        return Vec::new();
    }

    // 3. Compute parallel lane tracks for each net in the bundle
    let num_segs = beveled_guide.len() - 1;
    let mut seg_unit = Vec::with_capacity(num_segs);
    let mut seg_norm = Vec::with_capacity(num_segs);

    for k in 0..num_segs {
        let p0 = beveled_guide[k];
        let p1 = beveled_guide[k + 1];
        let dx = p1.0 - p0.0;
        let dy = p1.1 - p0.1;
        let len = dx.hypot(dy).max(1e-6);
        let u = (dx / len, dy / len);
        let n = (-u.1, u.0);
        seg_unit.push(u);
        seg_norm.push(n);
    }

    let mut all_tracks = Vec::new();

    for (lane_idx, &net_id) in bundle.net_ids.iter().enumerate() {
        let offset = (lane_idx as f64 - half_span) * pitch;

        // Generate lane vertices along the beveled guide
        let mut lane_pts: Vec<(f64, f64)> = Vec::with_capacity(beveled_guide.len());

        // First point:
        let start_v = (
            beveled_guide[0].0 + offset * seg_norm[0].0,
            beveled_guide[0].1 + offset * seg_norm[0].1,
        );
        lane_pts.push(start_v);

        // Intermediate corners:
        for k in 1..num_segs {
            let u_prev = seg_unit[k - 1];
            let n_prev = seg_norm[k - 1];
            let u_curr = seg_unit[k];
            let n_curr = seg_norm[k];

            let a0 = (
                beveled_guide[k - 1].0 + offset * n_prev.0,
                beveled_guide[k - 1].1 + offset * n_prev.1,
            );
            let b0 = (
                beveled_guide[k].0 + offset * n_curr.0,
                beveled_guide[k].1 + offset * n_curr.1,
            );

            let d = u_prev.0 * u_curr.1 - u_prev.1 * u_curr.0;
            if d.abs() < 1e-6 {
                lane_pts.push((
                    beveled_guide[k].0 + offset * n_curr.0,
                    beveled_guide[k].1 + offset * n_curr.1,
                ));
            } else {
                let t = ((b0.0 - a0.0) * u_curr.1 - (b0.1 - a0.1) * u_curr.0) / d;
                let inter = (a0.0 + t * u_prev.0, a0.1 + t * u_prev.1);
                lane_pts.push(inter);
            }
        }

        // Last point:
        let end_v = (
            beveled_guide[num_segs].0 + offset * seg_norm[num_segs - 1].0,
            beveled_guide[num_segs].1 + offset * seg_norm[num_segs - 1].1,
        );
        lane_pts.push(end_v);

        // Sanitize lane points: deduplicate
        let mut clean_lane_pts: Vec<(f64, f64)> = Vec::new();
        for &pt in &lane_pts {
            if let Some(&last) = clean_lane_pts.last() {
                if (pt.0 - last.0).hypot(pt.1 - last.1) < 1e-4 {
                    continue;
                }
            }
            clean_lane_pts.push(pt);
        }

        // 4. Terminal fan-in and fan-out connection to pads
        let mut matching_pads: Vec<(f64, f64)> = Vec::new();
        for fp in &board.footprints {
            for pad in &fp.pads {
                if pad.net_id == net_id && (pad.is_through_hole || fp.layer == bundle.layer) {
                    let p_pos = fp.pad_absolute_position(pad);
                    if !matching_pads.iter().any(|m: &(f64, f64)| (m.0 - p_pos.0).hypot(m.1 - p_pos.1) < 1e-4) {
                        matching_pads.push(p_pos);
                    }
                }
            }
        }

        let p0_guide = beveled_guide[0];
        let pk_guide = *beveled_guide.last().unwrap();

        let mut start_pad: Option<(f64, f64)> = None;
        let mut end_pad: Option<(f64, f64)> = None;

        if matching_pads.len() >= 2 {
            let mut sorted = matching_pads.clone();
            sorted.sort_by(|a, b| {
                let d_a = (a.0 - p0_guide.0).hypot(a.1 - p0_guide.1);
                let d_b = (b.0 - p0_guide.0).hypot(b.1 - p0_guide.1);
                d_a.partial_cmp(&d_b).unwrap_or(Ordering::Equal)
            });
            start_pad = Some(sorted[0]);

            let mut sorted_end = matching_pads.clone();
            sorted_end.retain(|p| (p.0 - sorted[0].0).hypot(p.1 - sorted[0].1) > 1e-4);
            sorted_end.sort_by(|a, b| {
                let d_a = (a.0 - pk_guide.0).hypot(a.1 - pk_guide.1);
                let d_b = (b.0 - pk_guide.0).hypot(b.1 - pk_guide.1);
                d_a.partial_cmp(&d_b).unwrap_or(Ordering::Equal)
            });
            if !sorted_end.is_empty() {
                end_pad = Some(sorted_end[0]);
            }
        } else if matching_pads.len() == 1 {
            let p = matching_pads[0];
            let d0 = (p.0 - p0_guide.0).hypot(p.1 - p0_guide.1);
            let dk = (p.0 - pk_guide.0).hypot(p.1 - pk_guide.1);
            if d0 < dk {
                start_pad = Some(p);
            } else {
                end_pad = Some(p);
            }
        }

        let mut full_chain: Vec<(f64, f64)> = Vec::new();

        // Prepend fan-in
        if let Some(sp) = start_pad {
            if let Some(&lane_start) = clean_lane_pts.first() {
                let fanin = compute_45_dogleg(sp, lane_start, false);
                for pt in fanin {
                    if full_chain.last().map_or(true, |&last| (last.0 - pt.0).hypot(last.1 - pt.1) > 1e-4) {
                        full_chain.push(pt);
                    }
                }
            }
        }

        // Add spine lane points
        for pt in clean_lane_pts {
            if full_chain.last().map_or(true, |&last| (last.0 - pt.0).hypot(last.1 - pt.1) > 1e-4) {
                full_chain.push(pt);
            }
        }

        // Append fan-out
        if let Some(ep) = end_pad {
            if let Some(&lane_end) = full_chain.last() {
                let fanout = compute_45_dogleg(lane_end, ep, true);
                for pt in fanout {
                    if full_chain.last().map_or(true, |&last| (last.0 - pt.0).hypot(last.1 - pt.1) > 1e-4) {
                        full_chain.push(pt);
                    }
                }
            }
        }

        // Convert full chain to TrackSegments
        for w in 0..full_chain.len().saturating_sub(1) {
            let p_start = full_chain[w];
            let p_end = full_chain[w + 1];
            if (p_start.0 - p_end.0).hypot(p_start.1 - p_end.1) > 1e-4 {
                all_tracks.push(TrackSegment {
                    start: p_start,
                    end: p_end,
                    width: bundle.track_width,
                    layer: bundle.layer,
                    net_id,
                    is_locked: false,
                });
            }
        }
    }

    all_tracks
}

/// Routes an entire [`BusBundle`] along a user-sketched guide polyline using equidistant river routing.
///
/// Offsets each net symmetrically from the guide spine, applies 45° concentric corner miters,
/// connects terminal endpoints to component pads, and runs DRC checks with dynamic bottleneck
/// spacing compression down to the netclass minimum clearance.
///
/// # Arguments
/// * `board` - Mutable board state where legal bundle tracks and bundle metadata are committed.
/// * `grid` - Mutable routing grid re-rasterized after committing the bundle tracks.
/// * `bundle` - Bus bundle specification to route.
///
/// # Returns
/// `true` if a DRC-legal spacing was found and the bundle tracks were committed to `board` and `grid`.
///
/// # Side Effects
/// Appends generated tracks to `board.tracks`, records `bundle` in `board.bundles`, and re-rasterizes `grid`.
pub fn route_bundle_along_guide(
    board: &mut Board,
    grid: &mut RoutingGrid,
    bundle: &BusBundle,
) -> bool {
    if bundle.guide_points.len() < 2 {
        return false;
    }
    if bundle.net_ids.is_empty() {
        return true;
    }

    // Determine min design clearance for nets in this bundle
    let min_class_clearance = bundle
        .net_ids
        .iter()
        .map(|&nid| board.get_netclass_for_net(nid).clearance)
        .fold(0.12_f64, f64::max);

    // Try nominal spacing first, then dynamically compress if localized bottlenecks exist
    let mut candidate_spacings = vec![bundle.track_spacing];
    if bundle.track_spacing > min_class_clearance + 0.005 {
        let steps = 6;
        for s in 1..=steps {
            let frac = s as f64 / steps as f64;
            let comp = bundle.track_spacing - frac * (bundle.track_spacing - min_class_clearance);
            candidate_spacings.push(comp);
        }
    }

    for spacing in candidate_spacings {
        let tracks = generate_bundle_tracks(board, bundle, spacing);
        if tracks.is_empty() {
            continue;
        }

        // Check DRC legality of all generated segments against foreign obstacles
        let mut legal = true;
        for trk in &tracks {
            let clearance = board.get_netclass_for_net(trk.net_id).clearance.max(0.12);
            if !is_track_segment_legal(
                board,
                trk.start,
                trk.end,
                trk.width,
                trk.layer,
                trk.net_id,
                clearance,
            ) {
                legal = false;
                break;
            }
        }

        if legal {
            // Success! Commit tracks to board
            board.tracks.extend(tracks);
            if !board.bundles.iter().any(|b| b.name == bundle.name && b.guide_points == bundle.guide_points) {
                board.bundles.push(bundle.clone());
            }
            grid.rasterize_obstacles(board);
            return true;
        }
    }

    false
}
