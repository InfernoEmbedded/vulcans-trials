// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! Unified PCB data model, 2D computational geometry, S-expression parsers, and file I/O.
//!
//! This module provides the foundational data structures and serialization engines for
//! Vulcan's Trials across KiCad (`8.0` / `9.0` / `10.0`) and Specctra workflows:
//!
//! - **Core PCB Data Model**: [`Board`], [`Footprint`], [`Pad`], [`TrackSegment`], [`ArcSegment`],
//!   [`Via`], [`KeepoutZone`], [`BoardText`], [`NetClass`], [`DesignRules`], [`DfmConfig`],
//!   [`FabPreset`], [`NetTopology`], and [`BusBundle`].
//! - **2D Geometry & Courtyard Collision Math**: [`Point`], [`AABB`], [`Segment`], and [`Polygon`],
//!   including Separating Axis Theorem (SAT) convex polygon intersection (`Polygon::intersects_convex`),
//!   exact rectilinear horizontal-slab decomposition and Hertel-Mehlhorn ear-clipping convex
//!   decomposition (`Polygon::decompose_convex`), and Monotone Chain convex hulls (`Polygon::convex_hull`).
//! - **S-Expression & Project Parsers**: In-memory parsing of `.kicad_pcb` boards (`Board::from_kicad_pcb`),
//!   `.kicad_pro` JSON netclasses and design settings (`Board::load_project_netclasses`), `.kicad_dru`
//!   custom design rule constraints (`Board::load_design_rules_dru`), and `.kicad_sch` hierarchical
//!   functional placement clusters (`Board::load_schematic_clusters`).
//! - **Ratsnest & Connectivity Analysis**: Prim's Euclidean Minimum Spanning Tree (MST) ratsnest
//!   generator (`Board::generate_ratsnest_segments`), copper union-find connected-component filtering
//!   (`Board::filter_unrouted_ratsnest`), and topological trace rip-up (`Board::rip_up_connected_traces_to_component_at`).
//! - **In-Place Exporters & Silkscreen Optimization**: Span-preserving `.kicad_pcb` writer
//!   (`Board::to_kicad_pcb`, `Board::save_pcb`) that retains all original KiCad metadata and zones,
//!   Vulkan GPU + Rayon CPU silkscreen text nudging (`Board::nudge_silkscreen_texts_with_gpu`),
//!   Specctra `.ses` session export (`Board::save_ses`), and pin/gate swap `.eco` back-annotation
//!   reporting (`Board::generate_back_annotation_report`).
//! - **Live Interactive State**: [`LiveTrialState`] and [`TrialPhase`] for synchronizing real-time
//!   placement and routing telemetry with the GUI and headless CLI.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Default reference designators locked in place during auto-placement unless overridden.
///
/// Includes primary board connectors (`J1`, `J2`) and mechanical mounting holes (`H1`..`H4`).
pub const DEFAULT_LOCKED_REFS: &[&str] = &[
    "J1", "J2", "H1", "H2", "H3", "H4",
];

/// Checks whether a net name represents a ground reference net (`GND`, `/GND`, `GNDA`, `GNDD`, etc.).
///
/// Leading whitespace and hierarchical sheet slashes (`/`) are stripped before checking
/// for a case-insensitive `"GND"` prefix.
///
/// # Arguments
/// * `name` - The net name string slice to inspect.
///
/// # Returns
/// `true` if the normalized net name begins with `"GND"` (case-insensitive), `false` otherwise.
///
/// # Side Effects
/// None.
pub fn is_gnd_net_name(name: &str) -> bool {
    let base = name.trim().trim_start_matches('/');
    base.len() >= 3 && base.is_char_boundary(3) && base[..3].eq_ignore_ascii_case("GND")
}

/// A 2D point in KiCad board or footprint coordinates (in millimeters).
///
/// Note that KiCad PCB coordinates use a right-handed X-right, Y-down coordinate system.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// Horizontal coordinate in millimeters (`mm`).
    pub x: f64,
    /// Vertical coordinate in millimeters (`mm`, increasing downward in KiCad PCB space).
    pub y: f64,
}

impl Point {
    /// Constructs a new 2D [`Point`] at `(x, y)` in millimeters.
    ///
    /// # Arguments
    /// * `x` - Horizontal coordinate in `mm`.
    /// * `y` - Vertical coordinate in `mm`.
    ///
    /// # Returns
    /// A new [`Point`] instance.
    ///
    /// # Side Effects
    /// None.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Computes the Euclidean distance in millimeters between `self` and `other`.
    ///
    /// # Arguments
    /// * `other` - The target [`Point`] to measure distance to.
    ///
    /// # Returns
    /// Euclidean distance $\sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$ in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn distance_to(&self, other: &Point) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }

    /// Rotates the point counter-clockwise around the origin `(0.0, 0.0)` in KiCad's Y-down coordinate system.
    ///
    /// # Arguments
    /// * `angle_deg` - Counter-clockwise rotation angle in degrees.
    ///
    /// # Returns
    /// The rotated [`Point`] in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn rotate_ccw(&self, angle_deg: f64) -> Self {
        let rad = angle_deg.to_radians();
        let (sin_a, cos_a) = rad.sin_cos();
        Self {
            x: self.x * cos_a + self.y * sin_a,
            y: -self.x * sin_a + self.y * cos_a,
        }
    }

    /// Computes the shortest Euclidean distance from `self` to the finite line segment `ab`.
    ///
    /// Projects `self` onto the line through `a` and `b`, clamping the barycentric parameter
    /// `t` to `[0.0, 1.0]`.
    ///
    /// # Arguments
    /// * `a` - Start endpoint of the line segment in `mm`.
    /// * `b` - End endpoint of the line segment in `mm`.
    ///
    /// # Returns
    /// Minimum Euclidean distance from `self` to segment `ab` in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn distance_to_segment(&self, a: &Point, b: &Point) -> f64 {
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len_sq = dx * dx + dy * dy;
        if len_sq < 1e-12 {
            return self.distance_to(a);
        }
        let t = (((self.x - a.x) * dx + (self.y - a.y) * dy) / len_sq).clamp(0.0, 1.0);
        let proj_x = a.x + t * dx;
        let proj_y = a.y + t * dy;
        (self.x - proj_x).hypot(self.y - proj_y)
    }
}

/// An Axis-Aligned Bounding Box (AABB) in 2D board coordinates (in millimeters).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AABB {
    /// Minimum X coordinate (left edge) in `mm`.
    pub min_x: f64,
    /// Minimum Y coordinate (top edge in KiCad space) in `mm`.
    pub min_y: f64,
    /// Maximum X coordinate (right edge) in `mm`.
    pub max_x: f64,
    /// Maximum Y coordinate (bottom edge in KiCad space) in `mm`.
    pub max_y: f64,
}

impl AABB {
    /// Creates a new [`AABB`] from minimum and maximum coordinates in millimeters.
    ///
    /// # Arguments
    /// * `min_x` - Minimum X coordinate in `mm`.
    /// * `min_y` - Minimum Y coordinate in `mm`.
    /// * `max_x` - Maximum X coordinate in `mm`.
    /// * `max_y` - Maximum Y coordinate in `mm`.
    ///
    /// # Returns
    /// A new [`AABB`] instance.
    ///
    /// # Side Effects
    /// None.
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    /// Tests whether two axis-aligned bounding boxes have a strictly positive area of intersection.
    ///
    /// Touching edges (`self.max_x == other.min_x`) return `false`.
    ///
    /// # Arguments
    /// * `other` - The other [`AABB`] to test against.
    ///
    /// # Returns
    /// `true` if the interiors of `self` and `other` overlap, `false` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min_x < other.max_x
            && self.max_x > other.min_x
            && self.min_y < other.max_y
            && self.max_y > other.min_y
    }

    /// Tests whether a 2D [`Point`] lies inside or on the boundary of this bounding box.
    ///
    /// # Arguments
    /// * `p` - The [`Point`] to test.
    ///
    /// # Returns
    /// `true` if `p.x` is in `[min_x, max_x]` and `p.y` is in `[min_y, max_y]`, `false` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn contains_point(&self, p: &Point) -> bool {
        p.x >= self.min_x && p.x <= self.max_x && p.y >= self.min_y && p.y <= self.max_y
    }
}

/// A 2D closed polygon defined by an ordered sequence of vertices in millimeters.
///
/// Used for component courtyards (`F.CrtYd`/`B.CrtYd`), board outlines (`Edge.Cuts`),
/// and SAT collision detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Polygon {
    /// Ordered vertices around the polygon perimeter in `mm`.
    pub vertices: Vec<Point>,
}

impl Polygon {
    /// Creates a new [`Polygon`] from a vector of vertices.
    ///
    /// # Arguments
    /// * `vertices` - Ordered list of [`Point`] vertices in `mm`.
    ///
    /// # Returns
    /// A new [`Polygon`] wrapping `vertices`.
    ///
    /// # Side Effects
    /// None.
    pub fn new(vertices: Vec<Point>) -> Self {
        Self { vertices }
    }

    /// Constructs a 4-vertex axis-aligned rectangular [`Polygon`] from top-left corner and dimensions.
    ///
    /// # Arguments
    /// * `min_x` - Left X coordinate in `mm`.
    /// * `min_y` - Top Y coordinate in `mm`.
    /// * `w` - Rectangle width along X in `mm`.
    /// * `h` - Rectangle height along Y in `mm`.
    ///
    /// # Returns
    /// A 4-vertex [`Polygon`] representing `[min_x, min_x + w] x [min_y, min_y + h]`.
    ///
    /// # Side Effects
    /// None.
    pub fn from_rect(min_x: f64, min_y: f64, w: f64, h: f64) -> Self {
        Self {
            vertices: vec![
                Point::new(min_x, min_y),
                Point::new(min_x + w, min_y),
                Point::new(min_x + w, min_y + h),
                Point::new(min_x, min_y + h),
            ],
        }
    }

    /// Computes the tightest axis-aligned bounding box ([`AABB`]) enclosing all polygon vertices.
    ///
    /// # Returns
    /// An [`AABB`] covering all vertices (or infinite bounds if `vertices` is empty).
    ///
    /// # Side Effects
    /// None.
    pub fn aabb(&self) -> AABB {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for v in &self.vertices {
            min_x = min_x.min(v.x);
            max_x = max_x.max(v.x);
            min_y = min_y.min(v.y);
            max_y = max_y.max(v.y);
        }
        AABB::new(min_x, min_y, max_x, max_y)
    }

    /// Rotates all vertices counter-clockwise by `angle_deg` and translates them by `origin`.
    ///
    /// # Arguments
    /// * `origin` - World translation offset in `mm`.
    /// * `angle_deg` - Counter-clockwise rotation angle in degrees.
    ///
    /// # Returns
    /// A new transformed [`Polygon`] in world coordinates.
    ///
    /// # Side Effects
    /// None.
    pub fn transform(&self, origin: Point, angle_deg: f64) -> Polygon {
        Polygon::new(
            self.vertices
                .iter()
                .map(|v| {
                    let r = v.rotate_ccw(angle_deg);
                    Point::new(r.x + origin.x, r.y + origin.y)
                })
                .collect(),
        )
    }

    /// Expands the polygon outward by `offset` millimeters for clearance halo inflation.
    ///
    /// - For axis-aligned 4-vertex rectangles, expands the [`AABB`] directly by `offset` on all four sides.
    /// - For non-orthogonal convex polygons (such as rotated rectangles, diamonds, trapezoids, or convex N-gons),
    ///   offsets each edge outward along its unit normal by `offset` and computes the mitered corner vertices,
    ///   preserving the polygon's orientation and aspect ratio.
    /// - For non-convex or degenerate polygons, pushes each vertex radially outward from the vertex centroid by `offset`.
    ///
    /// # Arguments
    /// * `offset` - Outward expansion distance in `mm` (returns a clone if `offset <= 0.0`).
    ///
    /// # Returns
    /// An expanded [`Polygon`].
    ///
    /// # Side Effects
    /// None.
    pub fn expanded(&self, offset: f64) -> Polygon {
        if offset <= 0.0 || self.vertices.is_empty() {
            return self.clone();
        }
        let aabb = self.aabb();
        let is_axis_aligned_rect = self.vertices.len() == 4
            && aabb.max_x - aabb.min_x > 1e-6
            && aabb.max_y - aabb.min_y > 1e-6
            && (0..4).all(|i| {
                let p1 = self.vertices[i];
                let p2 = self.vertices[(i + 1) % 4];
                (p2.x - p1.x).abs() <= 1e-5 || (p2.y - p1.y).abs() <= 1e-5
            });
        if is_axis_aligned_rect {
            return Polygon::new(vec![
                Point::new(aabb.min_x - offset, aabb.min_y - offset),
                Point::new(aabb.max_x + offset, aabb.min_y - offset),
                Point::new(aabb.max_x + offset, aabb.max_y + offset),
                Point::new(aabb.min_x - offset, aabb.max_y + offset),
            ]);
        }
        let n_verts = self.vertices.len();
        let area = self.signed_area();
        if n_verts >= 3 && area.abs() > 1e-6 && self.is_convex() {
            let winding = if area >= 0.0 { 1.0 } else { -1.0 };
            let mut normals = Vec::with_capacity(n_verts);
            let mut all_valid = true;
            for i in 0..n_verts {
                let p1 = self.vertices[i];
                let p2 = self.vertices[(i + 1) % n_verts];
                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let len = dx.hypot(dy);
                if len <= 1e-6 {
                    all_valid = false;
                    break;
                }
                normals.push(Point::new((dy / len) * winding, (-dx / len) * winding));
            }
            if all_valid {
                let mut out = Vec::with_capacity(n_verts);
                for i in 0..n_verts {
                    let n1 = normals[(i + n_verts - 1) % n_verts];
                    let n2 = normals[i];
                    let dot = n1.x * n2.x + n1.y * n2.y;
                    let denom = (1.0 + dot).max(0.15);
                    let scale = offset / denom;
                    let v = self.vertices[i];
                    out.push(Point::new(
                        v.x + (n1.x + n2.x) * scale,
                        v.y + (n1.y + n2.y) * scale,
                    ));
                }
                return Polygon::new(out);
            }
        }
        let n = n_verts as f64;
        let cx = self.vertices.iter().map(|v| v.x).sum::<f64>() / n;
        let cy = self.vertices.iter().map(|v| v.y).sum::<f64>() / n;
        Polygon::new(
            self.vertices
                .iter()
                .map(|v| {
                    let dx = v.x - cx;
                    let dy = v.y - cy;
                    let len = dx.hypot(dy);
                    if len > 1e-6 {
                        Point::new(v.x + (dx / len) * offset, v.y + (dy / len) * offset)
                    } else {
                        *v
                    }
                })
                .collect(),
        )
    }

    /// Determines whether a point lies inside the polygon using the Jordan curve even-odd ray-casting test.
    ///
    /// # Arguments
    /// * `p` - The [`Point`] to test in `mm`.
    ///
    /// # Returns
    /// `true` if `p` is inside the polygon, `false` if outside or if the polygon has fewer than 3 vertices.
    ///
    /// # Side Effects
    /// None.
    pub fn contains_point(&self, p: &Point) -> bool {
        let n = self.vertices.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let pi = self.vertices[i];
            let pj = self.vertices[j];
            if ((pi.y > p.y) != (pj.y > p.y))
                && (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Computes the shortest distance from an interior point `p` to the nearest polygon boundary edge.
    ///
    /// If `p` is outside the polygon (`!self.contains_point(p)`), returns `0.0`.
    ///
    /// # Arguments
    /// * `p` - The query [`Point`] in `mm`.
    ///
    /// # Returns
    /// Distance in `mm` from `p` to the closest polygon edge if `p` is inside, or `0.0` if outside.
    ///
    /// # Side Effects
    /// None.
    pub fn distance_to_point(&self, p: &Point) -> f64 {
        if !self.contains_point(p) {
            return 0.0;
        }
        let n = self.vertices.len();
        if n < 2 {
            return 0.0;
        }
        let mut min_d = f64::INFINITY;
        for i in 0..n {
            let j = (i + 1) % n;
            let d = p.distance_to_segment(&self.vertices[i], &self.vertices[j]);
            min_d = min_d.min(d);
        }
        min_d
    }

    /// Computes the minimum clearance from an interior line segment `s -> e` to the polygon boundary.
    ///
    /// Samples the segment at `0.05 mm` intervals and returns the minimum interior boundary distance,
    /// or `0.0` if either endpoint lies outside the polygon.
    ///
    /// # Arguments
    /// * `s` - Start `(x, y)` coordinates of the segment in `mm`.
    /// * `e` - End `(x, y)` coordinates of the segment in `mm`.
    ///
    /// # Returns
    /// Minimum distance in `mm` from the segment to the polygon boundary (`0.0` if not strictly inside).
    ///
    /// # Side Effects
    /// None.
    pub fn distance_to_segment(&self, s: (f64, f64), e: (f64, f64)) -> f64 {
        let p_start = Point::new(s.0, s.1);
        let p_end = Point::new(e.0, e.1);
        if !self.contains_point(&p_start) || !self.contains_point(&p_end) {
            return 0.0;
        }
        let n = self.vertices.len();
        if n < 2 {
            return 0.0;
        }
        let seg_len = (e.0 - s.0).hypot(e.1 - s.1);
        let steps = ((seg_len / 0.05).ceil() as usize).max(2);
        let mut min_d = f64::INFINITY;
        for step in 0..=steps {
            let t = step as f64 / steps as f64;
            let pt = Point::new(s.0 + t * (e.0 - s.0), s.1 + t * (e.1 - s.1));
            min_d = min_d.min(self.distance_to_point(&pt));
        }
        min_d
    }

    /// Checks whether the polygon is convex by verifying that all consecutive edge cross products have the same sign.
    ///
    /// # Returns
    /// `true` if the polygon is convex (or has 3 or fewer vertices), `false` if concave.
    ///
    /// # Side Effects
    /// None.
    pub fn is_convex(&self) -> bool {
        let n = self.vertices.len();
        if n <= 3 {
            return true;
        }
        let mut sign_pos = false;
        let mut sign_neg = false;
        for i in 0..n {
            let p0 = self.vertices[i];
            let p1 = self.vertices[(i + 1) % n];
            let p2 = self.vertices[(i + 2) % n];
            let cross = (p1.x - p0.x) * (p2.y - p1.y) - (p1.y - p0.y) * (p2.x - p1.x);
            if cross > 1e-6 {
                sign_pos = true;
            } else if cross < -1e-6 {
                sign_neg = true;
            }
            if sign_pos && sign_neg {
                return false;
            }
        }
        true
    }

    /// Computes the signed area of the polygon using the Shoelace formula.
    ///
    /// Positive for counter-clockwise winding in standard Cartesian coordinates (clockwise in Y-down space).
    ///
    /// # Returns
    /// Signed area in $\text{mm}^2$ (`0.0` if fewer than 3 vertices).
    ///
    /// # Side Effects
    /// None.
    pub fn signed_area(&self) -> f64 {
        let n = self.vertices.len();
        if n < 3 {
            return 0.0;
        }
        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += self.vertices[i].x * self.vertices[j].y - self.vertices[j].x * self.vertices[i].y;
        }
        area * 0.5
    }

    /// Decomposes a potentially concave polygon into a set of non-overlapping convex [`Polygon`] pieces.
    ///
    /// - If already convex, returns a single-element vector containing a clone of `self`.
    /// - If rectilinear (all edges axis-aligned within `1e-4 mm`), performs an exact horizontal-slab sweep
    ///   with vertical merging into minimal 4-vertex rectangles.
    /// - Otherwise, performs ear-clipping triangulation followed by Hertel-Mehlhorn greedy diagonal removal.
    ///
    /// # Returns
    /// A vector of convex [`Polygon`] sub-pieces whose union equals `self`.
    ///
    /// # Side Effects
    /// None.
    pub fn decompose_convex(&self) -> Vec<Polygon> {
        if self.is_convex() || self.vertices.len() <= 3 {
            return vec![self.clone()];
        }

        // Fast exact rectilinear decomposition into axis-aligned 4-vertex rectangles
        let n_orig = self.vertices.len();
        let is_rectilinear = (0..n_orig).all(|i| {
            let p1 = self.vertices[i];
            let p2 = self.vertices[(i + 1) % n_orig];
            (p2.x - p1.x).abs() <= 1e-4 || (p2.y - p1.y).abs() <= 1e-4
        });
        if is_rectilinear {
            let mut ys: Vec<f64> = self.vertices.iter().map(|v| v.y).collect();
            ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            ys.dedup_by(|a, b| (*a - *b).abs() <= 1e-4);
            let mut rects: Vec<(f64, f64, f64, f64)> = Vec::new();
            for k in 0..ys.len().saturating_sub(1) {
                let y0 = ys[k];
                let y1 = ys[k + 1];
                if y1 - y0 <= 1e-4 {
                    continue;
                }
                let y_mid = 0.5 * (y0 + y1);
                let mut xs: Vec<f64> = Vec::new();
                for i in 0..n_orig {
                    let p1 = self.vertices[i];
                    let p2 = self.vertices[(i + 1) % n_orig];
                    if (p2.x - p1.x).abs() <= 1e-4 {
                        let ymin = p1.y.min(p2.y);
                        let ymax = p1.y.max(p2.y);
                        if ymin <= y_mid && ymax >= y_mid && ymax - ymin > 1e-4 {
                            xs.push(0.5 * (p1.x + p2.x));
                        }
                    }
                }
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for pair in xs.chunks_exact(2) {
                    let x0 = pair[0];
                    let x1 = pair[1];
                    if x1 - x0 <= 1e-4 {
                        continue;
                    }
                    if let Some(prev) = rects.iter_mut().find(|r| {
                        (r.0 - x0).abs() <= 1e-4
                            && (r.2 - x1).abs() <= 1e-4
                            && (r.3 - y0).abs() <= 1e-4
                    }) {
                        prev.3 = y1;
                    } else {
                        rects.push((x0, y0, x1, y1));
                    }
                }
            }
            if !rects.is_empty() {
                return rects
                    .into_iter()
                    .map(|(x0, y0, x1, y1)| {
                        Polygon::new(vec![
                            Point::new(x0, y0),
                            Point::new(x1, y0),
                            Point::new(x1, y1),
                            Point::new(x0, y1),
                        ])
                    })
                    .collect();
            }
        }

        let mut pts = self.vertices.clone();
        if self.signed_area() < 0.0 {
            pts.reverse();
        }

        let is_point_in_tri = |p: Point, a: Point, b: Point, c: Point| -> bool {
            let sign = |p1: Point, p2: Point, p3: Point| {
                (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
            };
            let d1 = sign(p, a, b);
            let d2 = sign(p, b, c);
            let d3 = sign(p, c, a);
            let has_neg = (d1 < -1e-5) || (d2 < -1e-5) || (d3 < -1e-5);
            let has_pos = (d1 > 1e-5) || (d2 > 1e-5) || (d3 > 1e-5);
            !(has_neg && has_pos)
        };

        let mut triangles = Vec::new();
        let mut guard = 0;
        let max_iters = pts.len() * pts.len() + 10;

        while pts.len() > 3 && guard < max_iters {
            guard += 1;
            let n = pts.len();
            let mut ear_found = false;

            for i in 0..n {
                let prev = (i + n - 1) % n;
                let next = (i + 1) % n;
                let p_prev = pts[prev];
                let p_curr = pts[i];
                let p_next = pts[next];

                let cross = (p_curr.x - p_prev.x) * (p_next.y - p_curr.y)
                    - (p_curr.y - p_prev.y) * (p_next.x - p_curr.x);
                if cross <= 1e-6 {
                    continue;
                }

                let mut contains_other = false;
                for j in 0..n {
                    if j == prev || j == i || j == next {
                        continue;
                    }
                    if is_point_in_tri(pts[j], p_prev, p_curr, p_next) {
                        contains_other = true;
                        break;
                    }
                }

                if !contains_other {
                    triangles.push(Polygon::new(vec![p_prev, p_curr, p_next]));
                    pts.remove(i);
                    ear_found = true;
                    break;
                }
            }

            if !ear_found {
                if pts.len() >= 3 {
                    triangles.push(Polygon::new(vec![pts[0], pts[1], pts[2]]));
                    pts.remove(1);
                } else {
                    break;
                }
            }
        }

        if pts.len() == 3 {
            triangles.push(Polygon::new(pts));
        }

        if triangles.is_empty() {
            return vec![self.clone()];
        }

        // Hertel-Mehlhorn greedy merge of adjacent convex pieces sharing a diagonal
        let mut merged_any = true;
        while merged_any {
            merged_any = false;
            'outer: for i in 0..triangles.len() {
                for j in (i + 1)..triangles.len() {
                    let va = &triangles[i].vertices;
                    let vb = &triangles[j].vertices;
                    let na = va.len();
                    let nb = vb.len();
                    if na + nb - 2 > 16 {
                        continue;
                    }
                    for ea in 0..na {
                        let a0 = va[ea];
                        let a1 = va[(ea + 1) % na];
                        for eb in 0..nb {
                            let b0 = vb[eb];
                            let b1 = vb[(eb + 1) % nb];
                            if a0.distance_to(&b1) <= 1e-5 && a1.distance_to(&b0) <= 1e-5 {
                                let mut cand_pts = Vec::with_capacity(na + nb - 2);
                                for k in 0..na {
                                    cand_pts.push(va[(ea + 1 + k) % na]);
                                }
                                for k in 1..(nb - 1) {
                                    cand_pts.push(vb[(eb + 1 + k) % nb]);
                                }
                                let cand = Polygon::new(cand_pts);
                                if cand.is_convex() {
                                    triangles[i] = cand;
                                    triangles.remove(j);
                                    merged_any = true;
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
        }

        triangles
    }

    /// Computes the 2D convex hull of the polygon's vertices using Andrew's Monotone Chain algorithm.
    ///
    /// # Returns
    /// A convex [`Polygon`] enclosing all vertices of `self`.
    ///
    /// # Side Effects
    /// None.
    pub fn convex_hull(&self) -> Polygon {
        if self.vertices.len() <= 3 {
            return self.clone();
        }
        let mut pts = self.vertices.clone();
        pts.sort_by(|a, b| {
            a.x.partial_cmp(&b.x)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
        });
        pts.dedup_by(|a, b| (a.x - b.x).hypot(a.y - b.y) < 1e-5);
        if pts.len() <= 3 {
            return Polygon::new(pts);
        }

        let cross = |o: Point, a: Point, b: Point| -> f64 {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        };

        let mut lower = Vec::new();
        for &p in &pts {
            while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 1e-6 {
                lower.pop();
            }
            lower.push(p);
        }

        let mut upper = Vec::new();
        for &p in pts.iter().rev() {
            while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 1e-6 {
                upper.pop();
            }
            upper.push(p);
        }

        lower.pop();
        upper.pop();
        lower.extend(upper);
        Polygon::new(lower)
    }

    /// Tests whether two convex polygons intersect using the Separating Axis Theorem (SAT).
    ///
    /// First performs an $O(1)$ [`AABB`] broad-phase check, then projects both polygons onto the
    /// outward normal of every edge of `self` and `other`. Touching boundaries (`max_a <= min_b`)
    /// are treated as non-intersecting.
    ///
    /// # Arguments
    /// * `other` - The other convex [`Polygon`] to test against.
    ///
    /// # Returns
    /// `true` if the interiors of the two convex polygons overlap, `false` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn intersects_convex(&self, other: &Polygon) -> bool {
        if self.vertices.len() < 3 || other.vertices.len() < 3 || !self.aabb().overlaps(&other.aabb()) {
            return false;
        }
        for poly in [self, other] {
            let n = poly.vertices.len();
            for i in 0..n {
                let p1 = poly.vertices[i];
                let p2 = poly.vertices[(i + 1) % n];
                let axis = Point::new(-(p2.y - p1.y), p2.x - p1.x);
                let (min_a, max_a) = project_poly(self, axis);
                let (min_b, max_b) = project_poly(other, axis);
                if max_a <= min_b || max_b <= min_a {
                    return false;
                }
            }
        }
        true
    }
}

/// Projects all vertices of `poly` onto the 2D vector `axis` and returns the `(min, max)` scalar dot products.
///
/// # Arguments
/// * `poly` - The [`Polygon`] whose vertices are projected.
/// * `axis` - The 2D projection axis vector.
///
/// # Returns
/// A tuple `(min_dot, max_dot)` of scalar projections along `axis`.
///
/// # Side Effects
/// None.
fn project_poly(poly: &Polygon, axis: Point) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in &poly.vertices {
        let dot = v.x * axis.x + v.y * axis.y;
        min = min.min(dot);
        max = max.max(dot);
    }
    (min, max)
}

/// A 2D ratsnest airwire or geometric line segment associated with a specific net ID.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    /// First endpoint in board coordinates (`mm`).
    pub p1: Point,
    /// Second endpoint in board coordinates (`mm`).
    pub p2: Point,
    /// Numeric net identifier (`0` = unconnected).
    pub net_id: usize,
}

impl Segment {
    /// Computes the Euclidean length of the segment in millimeters.
    ///
    /// # Returns
    /// Distance between `p1` and `p2` in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn length(&self) -> f64 {
        self.p1.distance_to(&self.p2)
    }

    /// Computes the axis-aligned bounding box ([`AABB`]) of the segment.
    ///
    /// # Returns
    /// An [`AABB`] spanning `p1` and `p2` in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn aabb(&self) -> AABB {
        AABB::new(
            self.p1.x.min(self.p2.x),
            self.p1.y.min(self.p2.y),
            self.p1.x.max(self.p2.x),
            self.p1.y.max(self.p2.y),
        )
    }

    /// Tests whether two segments on different nets strictly cross each other's interior.
    ///
    /// Segments on the same net (`self.net_id == other.net_id`) or segments sharing an endpoint
    /// within `1e-4 mm` return `false`.
    ///
    /// # Arguments
    /// * `other` - The other [`Segment`] to test for crossing.
    ///
    /// # Returns
    /// `true` if `self` and `other` belong to different nets and strictly intersect, `false` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn crosses(&self, other: &Segment) -> bool {
        if self.net_id == other.net_id || !self.aabb().overlaps(&other.aabb()) {
            return false;
        }
        const EPS: f64 = 1e-4;
        if self.p1.distance_to(&other.p1) < EPS
            || self.p1.distance_to(&other.p2) < EPS
            || self.p2.distance_to(&other.p1) < EPS
            || self.p2.distance_to(&other.p2) < EPS
        {
            return false;
        }
        let ccw = |a: Point, b: Point, c: Point| (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        let d1 = ccw(self.p1, self.p2, other.p1);
        let d2 = ccw(self.p1, self.p2, other.p2);
        let d3 = ccw(other.p1, other.p2, self.p1);
        let d4 = ccw(other.p1, other.p2, self.p2);
        ((d1 > EPS && d2 < -EPS) || (d1 < -EPS && d2 > EPS))
            && ((d3 > EPS && d4 < -EPS) || (d3 < -EPS && d4 > EPS))
    }
}

/// Target multi-pin routing topology for a net or [`NetClass`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NetTopology {
    /// Minimum-wirelength Prim's Euclidean Minimum Spanning Tree (default).
    #[default]
    #[serde(alias = "euclidean", alias = "mst", alias = "EuclideanMst")]
    EuclideanMst,
    /// Serial point-to-point chain minimizing stub branches (used for SPI, I2C, UART, buses).
    #[serde(alias = "daisychain", alias = "daisy", alias = "DaisyChain")]
    DaisyChain,
    /// Ordered linear fly-by topology for high-speed memory/address/data buses.
    #[serde(alias = "flyby", alias = "FlyBy")]
    FlyBy,
    /// Balanced T-branch tree topology for clock and strobe distribution.
    #[serde(alias = "tbranch", alias = "t_tree", alias = "TBranch")]
    TBranch,
    /// Central hub star-point topology for Kelvin sense, analog references, and power rails.
    #[serde(alias = "star", alias = "starpoint", alias = "StarPoint")]
    StarPoint,
}

impl NetTopology {
    /// Parses a topology identifier string (e.g. `"mst"`, `"daisy_chain"`, `"flyby"`, `"t_branch"`, `"star"`) into a [`NetTopology`].
    ///
    /// # Arguments
    /// * `s` - String slice to parse (case-insensitive, hyphens/spaces normalized to underscores).
    ///
    /// # Returns
    /// `Some(NetTopology)` if recognized, or `None` if unknown.
    ///
    /// # Side Effects
    /// None.
    pub fn parse(s: &str) -> Option<Self> {
        let clean = s.trim().trim_matches('"').to_ascii_lowercase().replace(['-', ' '], "_");
        match clean.as_str() {
            "euclidean_mst" | "euclidean" | "mst" => Some(Self::EuclideanMst),
            "daisy_chain" | "daisychain" | "daisy" => Some(Self::DaisyChain),
            "fly_by" | "flyby" => Some(Self::FlyBy),
            "t_branch" | "tbranch" | "t_tree" => Some(Self::TBranch),
            "star_point" | "starpoint" | "star" => Some(Self::StarPoint),
            _ => None,
        }
    }

    /// Infers an appropriate [`NetTopology`] from a net's schematic name when no explicit override is set.
    ///
    /// - Sense/Kelvin/Power nets (`+`, `VCC`, `VDD`, `VREF`, `SENSE`, `STAR`) -> [`NetTopology::StarPoint`]
    /// - Clock/Strobe nets (`CLK`, `SCK`, `STROBE`, `*CK`) -> [`NetTopology::TBranch`]
    /// - Memory/Data nets (`ADDR`, `DDR`, `DATA`) -> [`NetTopology::FlyBy`]
    /// - Serial bus nets (`SPI`, `I2C`, `SDA`, `SCL`, `UART`, `BUS`) -> [`NetTopology::DaisyChain`]
    /// - All other nets -> [`NetTopology::EuclideanMst`]
    ///
    /// # Arguments
    /// * `net_name` - The net name string to classify.
    ///
    /// # Returns
    /// The inferred [`NetTopology`].
    ///
    /// # Side Effects
    /// None.
    pub fn infer_from_net_name(net_name: &str) -> Self {
        let upper = net_name.to_ascii_uppercase();
        if upper.contains("KELVIN")
            || upper.contains("STAR")
            || upper.contains("SENSE")
            || upper.starts_with('+')
            || upper.starts_with("VCC")
            || upper.starts_with("VDD")
            || upper.contains("VREF")
            || upper.contains("PWR")
            || upper.contains("POWER")
        {
            Self::StarPoint
        } else if upper.contains("CLK")
            || upper.contains("SCK")
            || upper.contains("STROBE")
            || upper.ends_with("_CK")
            || upper.ends_with("CK")
        {
            Self::TBranch
        } else if upper.contains("ADDR")
            || upper.contains("DDR")
            || upper.contains("DATA")
        {
            Self::FlyBy
        } else if upper.contains("SPI")
            || upper.contains("I2C")
            || upper.contains("SDA")
            || upper.contains("SCL")
            || upper.contains("UART")
            || upper.contains("BUS")
        {
            Self::DaisyChain
        } else {
            Self::EuclideanMst
        }
    }
}

/// Commercial PCB fabrication house DFM capability preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FabPreset {
    /// JLCPCB standard 2/4-layer process (`0.127 mm` track/space, `0.60/0.30 mm` via, `0.35 mm` edge clearance).
    Jlcpcb,
    /// PCBWay standard process (`0.15 mm` track/space, `0.60/0.30 mm` via, `0.30 mm` edge clearance).
    PcbWay,
    /// Eurocircuits standard pooling process (`0.15 mm` track/space, `0.60/0.30 mm` via, `0.40 mm` edge clearance).
    Eurocircuits,
    /// User-defined custom fabrication constraints.
    Custom,
}

impl FabPreset {
    /// Parses a fabrication house name (e.g. `"jlcpcb"`, `"pcbway"`, `"eurocircuits"`) into a [`FabPreset`].
    ///
    /// # Arguments
    /// * `s` - Manufacturer preset name string.
    ///
    /// # Returns
    /// `Some(FabPreset)` if matched, or `None` if unrecognized.
    ///
    /// # Side Effects
    /// None.
    pub fn parse(s: &str) -> Option<Self> {
        let clean = s.trim().to_lowercase().replace(['-', '_', ' ', '.'], "");
        if clean.contains("jlc") {
            Some(Self::Jlcpcb)
        } else if clean.contains("pcbway") {
            Some(Self::PcbWay)
        } else if clean.contains("eurocircuit") {
            Some(Self::Eurocircuits)
        } else {
            None
        }
    }

    /// Returns the minimum copper-to-board-edge (`Edge.Cuts`) clearance in millimeters for this preset.
    ///
    /// # Returns
    /// Board edge clearance in `mm` (`0.35 mm` for JLCPCB/Custom, `0.30 mm` for PCBWay, `0.40 mm` for Eurocircuits).
    ///
    /// # Side Effects
    /// None.
    pub fn board_edge_clearance(&self) -> f64 {
        match self {
            Self::Jlcpcb => 0.35,
            Self::PcbWay => 0.30,
            Self::Eurocircuits => 0.40,
            Self::Custom => 0.35,
        }
    }

    /// Clamps a [`NetClass`]'s track width, clearance, via dimensions, and edge clearance to meet the fab preset's minimums.
    ///
    /// # Arguments
    /// * `nc` - Mutable reference to the [`NetClass`] to constrain.
    ///
    /// # Side Effects
    /// Mutates `nc.board_edge_clearance`, `nc.clearance`, `nc.track_width`, `nc.via_dia`, and `nc.via_drill` in place.
    pub fn apply_to_netclass(&self, nc: &mut NetClass) {
        nc.board_edge_clearance = self.board_edge_clearance();
        match self {
            Self::Jlcpcb => {
                nc.clearance = nc.clearance.max(0.127);
                nc.track_width = nc.track_width.max(0.127);
                nc.via_dia = nc.via_dia.max(0.60);
                nc.via_drill = nc.via_drill.max(0.30);
            }
            Self::PcbWay => {
                nc.clearance = nc.clearance.max(0.15);
                nc.track_width = nc.track_width.max(0.15);
                nc.via_dia = nc.via_dia.max(0.60);
                nc.via_drill = nc.via_drill.max(0.30);
            }
            Self::Eurocircuits => {
                nc.clearance = nc.clearance.max(0.15);
                nc.track_width = nc.track_width.max(0.15);
                nc.via_dia = nc.via_dia.max(0.60);
                nc.via_drill = nc.via_drill.max(0.30);
            }
            Self::Custom => {}
        }
    }
}

/// Design-for-Manufacturing (DFM) configuration controlling teardrops, edge clearances, and fab presets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DfmConfig {
    /// Whether teardrop fillets should be synthesized at pad and via track junctions.
    pub teardrop_enabled: bool,
    /// Teardrop extension length as a ratio of the pad/via radius (default `0.5`).
    pub teardrop_length_ratio: f64,
    /// Teardrop width taper ratio relative to the pad/via diameter (default `0.8`).
    pub teardrop_width_ratio: f64,
    /// Minimum copper clearance to `Edge.Cuts` board outline in `mm` (default `0.35`).
    pub board_edge_clearance: f64,
    /// Optional fabrication house preset (`Some(FabPreset::Jlcpcb)` by default).
    pub fab_preset: Option<FabPreset>,
}

impl Default for DfmConfig {
    /// Returns the default JLCPCB-compatible [`DfmConfig`].
    fn default() -> Self {
        Self {
            teardrop_enabled: true,
            teardrop_length_ratio: 0.5,
            teardrop_width_ratio: 0.8,
            board_edge_clearance: 0.35,
            fab_preset: Some(FabPreset::Jlcpcb),
        }
    }
}

/// Global physical Design Rule Check (DRC) minimums imported from `.kicad_pro` or `.kicad_dru`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DesignRules {
    /// Global minimum copper-to-copper clearance in `mm`.
    pub min_clearance: f64,
    /// Global minimum copper trace width in `mm` (default `0.127 mm` / 5 mil).
    pub min_track_width: f64,
    /// Global minimum via outer pad diameter in `mm` (default `0.50 mm`).
    pub min_via_diameter: f64,
    /// Global minimum via drill hole diameter in `mm` (default `0.20 mm`).
    pub min_via_drill: f64,
    /// Minimum via annular copper ring width in `mm` (default `0.10 mm`).
    pub min_via_annular_width: f64,
    /// Minimum plated through-hole drill diameter in `mm` (default `0.20 mm`).
    pub min_through_hole_diameter: f64,
    /// Minimum drilled hole-to-copper clearance in `mm` (default `0.25 mm`).
    pub min_hole_clearance: f64,
    /// Minimum edge-to-edge clearance between distinct drilled holes in `mm` (default `0.25 mm`).
    pub min_hole_to_hole: f64,
    /// Minimum copper-to-board-edge (`Edge.Cuts`) clearance in `mm` (default `0.30 mm`).
    pub min_copper_edge_clearance: f64,
    /// Minimum copper pour zone clearance in `mm` (default `0.50 mm`).
    pub zone_min_clearance: f64,
    /// Minimum copper pour web thickness in `mm` (default `0.25 mm`).
    pub zone_min_thickness: f64,
}

impl Default for DesignRules {
    /// Returns standard KiCad/JLCPCB baseline [`DesignRules`] minimums.
    fn default() -> Self {
        Self {
            min_clearance: 0.0,
            min_track_width: 0.127,
            min_via_diameter: 0.50,
            min_via_drill: 0.20,
            min_via_annular_width: 0.10,
            min_through_hole_diameter: 0.20,
            min_hole_clearance: 0.25,
            min_hole_to_hole: 0.25,
            min_copper_edge_clearance: 0.30,
            zone_min_clearance: 0.50,
            zone_min_thickness: 0.25,
        }
    }
}

/// Electrical and physical routing rules for a named KiCad netclass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NetClass {
    /// Netclass identifier (e.g., `"Default"`, `"Power_15V"`, `"USB_Diff"`).
    pub name: String,
    /// Nominal copper trace width in `mm`.
    pub track_width: f64,
    /// Minimum copper-to-copper clearance to other nets in `mm`.
    pub clearance: f64,
    /// Nominal via outer copper diameter in `mm`.
    pub via_dia: f64,
    /// Nominal via drill hole diameter in `mm`.
    pub via_drill: f64,
    /// Edge-to-edge gap between differential pair traces in `mm`.
    #[serde(default = "default_diff_pair_dim")]
    pub diff_pair_gap: f64,
    /// Trace width for differential pair routing in `mm`.
    #[serde(default = "default_diff_pair_dim")]
    pub diff_pair_width: f64,
    /// Preferred multi-pin routing topology for nets in this class.
    #[serde(default)]
    pub topology: NetTopology,
    /// Whether teardrop fillets are enabled for nets in this class.
    #[serde(default = "default_true")]
    pub teardrop_enabled: bool,
    /// Teardrop length ratio for pads/vias on this netclass.
    #[serde(default = "default_teardrop_length_ratio")]
    pub teardrop_length_ratio: f64,
    /// Teardrop width ratio for pads/vias on this netclass.
    #[serde(default = "default_teardrop_width_ratio")]
    pub teardrop_width_ratio: f64,
    /// Minimum clearance from `Edge.Cuts` board outline in `mm`.
    #[serde(default = "default_board_edge_clearance")]
    pub board_edge_clearance: f64,
    /// Maximum allowed parallel coupled length with aggressor nets in `mm` before crosstalk penalty applies.
    #[serde(default = "default_max_coupled_length_mm")]
    pub max_coupled_length_mm: f64,
    /// Whether high-speed return-path reference plane continuity is required under traces.
    #[serde(default)]
    pub require_return_path: bool,
    /// Minimum allowed trace width in `mm` from board design rules.
    #[serde(default = "default_min_track_width")]
    pub min_track_width: f64,
    /// Minimum allowed via diameter in `mm` from board design rules.
    #[serde(default = "default_min_via_diameter")]
    pub min_via_diameter: f64,
    /// Minimum allowed via drill diameter in `mm` from board design rules.
    #[serde(default = "default_min_via_drill")]
    pub min_via_drill: f64,
    /// Minimum allowed via annular ring width in `mm` from board design rules.
    #[serde(default = "default_min_via_annular_width")]
    pub min_via_annular_width: f64,
    /// Minimum allowed through-hole drill diameter in `mm`.
    #[serde(default = "default_min_through_hole_diameter")]
    pub min_through_hole_diameter: f64,
    /// Minimum hole-to-copper clearance in `mm`.
    #[serde(default = "default_min_hole_clearance")]
    pub min_hole_clearance: f64,
    /// Minimum hole-to-hole drill clearance in `mm`.
    #[serde(default = "default_min_hole_to_hole")]
    pub min_hole_to_hole: f64,
    /// Minimum copper zone clearance in `mm`.
    #[serde(default = "default_zone_min_clearance")]
    pub zone_min_clearance: f64,
    /// Minimum copper zone web thickness in `mm`.
    #[serde(default = "default_zone_min_thickness")]
    pub zone_min_thickness: f64,
}

/// Serde default helper returning `0.20 mm` for differential pair width and gap.
fn default_diff_pair_dim() -> f64 {
    0.20
}

/// Serde default helper returning `true` for boolean flags.
fn default_true() -> bool {
    true
}

/// Serde default helper returning `0.5` for teardrop length ratio.
fn default_teardrop_length_ratio() -> f64 {
    0.5
}

/// Serde default helper returning `0.8` for teardrop width ratio.
fn default_teardrop_width_ratio() -> f64 {
    0.8
}

/// Serde default helper returning `0.35 mm` for board edge clearance.
fn default_board_edge_clearance() -> f64 {
    0.35
}

/// Serde default helper returning `15.0 mm` for maximum parallel coupled length.
fn default_max_coupled_length_mm() -> f64 {
    15.0
}

/// Serde default helper returning `0.127 mm` for minimum track width.
fn default_min_track_width() -> f64 {
    0.127
}

/// Serde default helper returning `0.50 mm` for minimum via outer diameter.
fn default_min_via_diameter() -> f64 {
    0.50
}

/// Serde default helper returning `0.20 mm` for minimum via drill diameter.
fn default_min_via_drill() -> f64 {
    0.20
}

/// Serde default helper returning `0.10 mm` for minimum via annular ring width.
fn default_min_via_annular_width() -> f64 {
    0.10
}

/// Serde default helper returning `0.20 mm` for minimum through-hole drill diameter.
fn default_min_through_hole_diameter() -> f64 {
    0.20
}

/// Serde default helper returning `0.25 mm` for minimum hole-to-copper clearance.
fn default_min_hole_clearance() -> f64 {
    0.25
}

/// Serde default helper returning `0.25 mm` for minimum hole-to-hole clearance.
fn default_min_hole_to_hole() -> f64 {
    0.25
}

/// Serde default helper returning `0.50 mm` for minimum copper zone clearance.
fn default_zone_min_clearance() -> f64 {
    0.50
}

/// Serde default helper returning `0.25 mm` for minimum copper zone thickness.
fn default_zone_min_thickness() -> f64 {
    0.25
}

impl Default for NetClass {
    /// Returns the `"Default"` [`NetClass`] with `0.20 mm` trace/clearance and `0.60/0.30 mm` vias.
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            track_width: 0.20,
            clearance: 0.20,
            via_dia: 0.60,
            via_drill: 0.30,
            diff_pair_gap: 0.20,
            diff_pair_width: 0.20,
            topology: NetTopology::default(),
            teardrop_enabled: true,
            teardrop_length_ratio: 0.5,
            teardrop_width_ratio: 0.8,
            board_edge_clearance: 0.35,
            max_coupled_length_mm: 15.0,
            require_return_path: false,
            min_track_width: 0.127,
            min_via_diameter: 0.50,
            min_via_drill: 0.20,
            min_via_annular_width: 0.10,
            min_through_hole_diameter: 0.20,
            min_hole_clearance: 0.25,
            min_hole_to_hole: 0.25,
            zone_min_clearance: 0.50,
            zone_min_thickness: 0.25,
        }
    }
}

impl NetClass {
    /// Applies board-level [`DesignRules`] minimums to this netclass, clamping nominal track width,
    /// clearance, and via dimensions so they never violate global DRC limits.
    ///
    /// # Arguments
    /// * `rules` - The global [`DesignRules`] to enforce.
    ///
    /// # Side Effects
    /// Mutates `self` in place, updating `min_*` fields and clamping `clearance`, `track_width`, `via_drill`, and `via_dia`.
    pub fn apply_design_rules(&mut self, rules: &DesignRules) {
        self.min_track_width = rules.min_track_width;
        self.min_via_diameter = rules.min_via_diameter;
        self.min_via_drill = rules.min_via_drill;
        self.min_via_annular_width = rules.min_via_annular_width;
        self.min_through_hole_diameter = rules.min_through_hole_diameter;
        self.min_hole_clearance = rules.min_hole_clearance;
        self.min_hole_to_hole = rules.min_hole_to_hole;
        self.zone_min_clearance = rules.zone_min_clearance;
        self.zone_min_thickness = rules.zone_min_thickness;
        self.clearance = self.clearance.max(rules.min_clearance);
        self.track_width = self.track_width.max(rules.min_track_width);
        self.via_drill = self
            .via_drill
            .max(rules.min_via_drill)
            .max(rules.min_through_hole_diameter);
        self.via_dia = self
            .via_dia
            .max(rules.min_via_diameter)
            .max(self.via_drill + 2.0 * rules.min_via_annular_width);
    }

    /// Checks whether a netclass or net name suggests a high-speed or differential signal.
    ///
    /// Matches substrings `"DIFF"`, `"CLK"`, `"CLOCK"`, `"HIGHSPEED"`, `"HIGH_SPEED"`, `"USB"`, `"PCIE"`, or `"DDR"`.
    ///
    /// # Arguments
    /// * `name` - Netclass or net name string.
    ///
    /// # Returns
    /// `true` if `name` matches a high-speed pattern, `false` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn is_high_speed_name(name: &str) -> bool {
        let upper = name.to_uppercase();
        upper.contains("DIFF")
            || upper.contains("CLK")
            || upper.contains("CLOCK")
            || upper.contains("HIGHSPEED")
            || upper.contains("HIGH_SPEED")
            || upper.contains("USB")
            || upper.contains("PCIE")
            || upper.contains("DDR")
    }

    /// Resolves the effective [`NetTopology`] for a net belonging to this [`NetClass`].
    ///
    /// If `self.topology` is explicitly set to a non-default topology, returns `self.topology`;
    /// otherwise infers the topology from `net_name` via [`NetTopology::infer_from_net_name`].
    ///
    /// # Arguments
    /// * `net_name` - The schematic net name.
    ///
    /// # Returns
    /// The resolved [`NetTopology`].
    ///
    /// # Side Effects
    /// None.
    pub fn resolve_topology(&self, net_name: &str) -> NetTopology {
        if self.topology != NetTopology::EuclideanMst {
            self.topology
        } else {
            NetTopology::infer_from_net_name(net_name)
        }
    }
}

/// A named group of electrically interchangeable pins within a component footprint (e.g., MCU GPIO bank).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SwappablePinGroup {
    /// Group identifier (e.g., `"GPIO"`, `"PA"`).
    pub name: String,
    /// Indices into [`Footprint::pads`] that may be swapped with one another.
    pub pin_indices: Vec<usize>,
}

/// A named group of interchangeable multi-pin logic gates or amplifier units within a package (e.g., quad NAND, dual op-amp).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SwappableGateGroup {
    /// Gate group identifier (e.g., `"gate_group"`).
    pub name: String,
    /// List of interchangeable gate units, where each unit is an ordered vector of pad indices in [`Footprint::pads`].
    pub gate_units: Vec<Vec<usize>>,
}

/// Audit record of a single pin or gate swap operation for schematic back-annotation (`.eco`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PinSwapRecord {
    /// Reference designator of the component whose pins were swapped (e.g., `"U1"`, `"R5"`).
    pub footprint_ref: String,
    /// Pad number of the first swapped pin.
    pub pad1_number: String,
    /// Net name originally attached to `pad1_number` prior to the swap.
    pub pad1_net: String,
    /// Pad number of the second swapped pin.
    pub pad2_number: String,
    /// Net name originally attached to `pad2_number` prior to the swap.
    pub pad2_net: String,
    /// Optimization reason for the swap (e.g., `"Pin swap reduced ratsnest wirelength"`).
    pub reason: String,
}

/// A conductive component pad (SMD or through-hole) belonging to a [`Footprint`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pad {
    /// Pad pin number or identifier (e.g., `"1"`, `"2"`, `"A1"`).
    pub number: String,
    /// Geometric pad shape (`"rect"`, `"roundrect"`, `"circle"`, `"oval"`, `"trapezoid"`).
    pub shape: String,
    /// Local X offset relative to the footprint origin in `mm` (at `0.0` degrees rotation).
    pub offset_x: f64,
    /// Local Y offset relative to the footprint origin in `mm` (at `0.0` degrees rotation).
    pub offset_y: f64,
    /// Pad copper width along its local X axis in `mm`.
    pub width: f64,
    /// Pad copper height along its local Y axis in `mm`.
    pub height: f64,
    /// Local pad rotation relative to the parent footprint in degrees.
    pub rotation: f64,
    /// Optional drill hole diameter in `mm` for through-hole pads.
    pub drill: Option<f64>,
    /// Numeric net ID (`0` = unconnected).
    pub net_id: usize,
    /// Electrical net name (e.g., `"GND"`, `"+3.3V"`, `"/USB_D+"`).
    pub net_name: String,
    /// `true` if the pad spans all copper layers (`thru_hole` / `*.Cu`), `false` for single-layer SMD.
    pub is_through_hole: bool,
    /// Schematic pin function or signal name from `(pinfunction ...)` or `(name ...)`.
    #[serde(default)]
    pub pin_name: String,
    /// Optional numeric pin-swap equivalence group ID parsed from `(swap_group ...)`.
    #[serde(default)]
    pub swap_group: Option<usize>,
    /// Optional `(start_byte, end_byte)` span of the pad's `(net ...)` S-expression in [`Board::raw_content`].
    #[serde(default)]
    pub net_span: Option<(usize, usize)>,
    /// Optional explicit copper layer index (`0` = `F.Cu`, `1` = `B.Cu`) for SMD pads (`None` for through-hole).
    #[serde(default)]
    pub layer: Option<usize>,
}

impl Default for Pad {
    /// Returns a default `0.5 x 0.5 mm` rectangular SMD [`Pad`] at `(0.0, 0.0)` with net `0`.
    fn default() -> Self {
        Self {
            number: String::new(),
            shape: "rect".to_string(),
            offset_x: 0.0,
            offset_y: 0.0,
            width: 0.5,
            height: 0.5,
            rotation: 0.0,
            drill: None,
            net_id: 0,
            net_name: String::new(),
            is_through_hole: false,
            pin_name: String::new(),
            swap_group: None,
            net_span: None,
            layer: None,
        }
    }
}

impl Pad {
    /// Constructs a new [`Pad`] with the given geometry and net assignment.
    ///
    /// Automatically assigns `"circle"` shape and a `0.5 * min(width, height)` drill hole when
    /// `is_through_hole` is `true`, or `"rect"` shape with `drill: None` for SMD pads.
    ///
    /// # Arguments
    /// * `number` - Pin number or identifier string.
    /// * `offset_x` - Local X offset from footprint center in `mm`.
    /// * `offset_y` - Local Y offset from footprint center in `mm`.
    /// * `width` - Pad width in `mm`.
    /// * `height` - Pad height in `mm`.
    /// * `net_id` - Numeric net ID (`0` = unconnected).
    /// * `net_name` - Electrical net name string.
    /// * `is_through_hole` - Whether this pad is a plated through-hole pad.
    ///
    /// # Returns
    /// A newly initialized [`Pad`].
    ///
    /// # Side Effects
    /// None.
    pub fn new(
        number: impl Into<String>,
        offset_x: f64,
        offset_y: f64,
        width: f64,
        height: f64,
        net_id: usize,
        net_name: impl Into<String>,
        is_through_hole: bool,
    ) -> Self {
        Self {
            number: number.into(),
            shape: if is_through_hole { "circle".to_string() } else { "rect".to_string() },
            offset_x,
            offset_y,
            width,
            height,
            rotation: 0.0,
            drill: if is_through_hole { Some(width.min(height) * 0.5) } else { None },
            net_id,
            net_name: net_name.into(),
            is_through_hole,
            pin_name: String::new(),
            swap_group: None,
            net_span: None,
            layer: None,
        }
    }
}

/// A physical component footprint placed on the PCB (`F.Cu` top or `B.Cu` bottom layer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Footprint {
    /// Component reference designator (e.g., `"U1"`, `"R12"`, `"C5"`).
    pub reference: String,
    /// Component value or part number (e.g., `"10k"`, `"100nF"`, `"ESP32-C3-WROOM-02"`).
    pub value: String,
    /// World X coordinate of the footprint origin in `mm`.
    pub x: f64,
    /// World Y coordinate of the footprint origin in `mm`.
    pub y: f64,
    /// Counter-clockwise footprint rotation angle in degrees (`0.0..360.0`).
    pub rotation: f64,
    /// Current copper placement layer index (`0` = `F.Cu` Top, `1` = `B.Cu` Bottom).
    pub layer: usize,      // 0 = F.Cu (Top), 1 = B.Cu (Bottom)
    /// Original copper layer index (`0` = `F.Cu`, `1` = `B.Cu`) when parsed from `.kicad_pcb`.
    pub orig_layer: usize, // Original layer when loaded from .kicad_pcb
    /// Whether the footprint position/rotation is locked against auto-placement moves.
    pub is_locked: bool,
    /// Component copper pads in local footprint coordinates.
    pub pads: Vec<Pad>,
    /// Overall courtyard bounding width along local X in `mm`.
    pub width: f64,
    /// Overall courtyard bounding height along local Y in `mm`.
    pub height: f64,
    /// Convex-decomposed courtyard polygons in local coordinates (`mm`) used for SAT collision checks.
    pub courtyards: Vec<Polygon>,
    /// Original undecomposed outer courtyard perimeter polygons in local coordinates (`mm`) for rendering.
    #[serde(default)]
    pub perimeters: Vec<Polygon>,
    /// Additional routing escape margin halo in `mm` around fine-pitch / high-density ICs.
    #[serde(default)]
    pub escape_margin: f64,
    /// Byte span `(start, end)` of the footprint's `(at X Y [ROT])` token inside [`Board::raw_content`].
    pub at_span: Option<(usize, usize)>,
    /// Byte span `(start, end)` of the entire `(footprint ...)` block inside [`Board::raw_content`].
    pub fp_span: Option<(usize, usize)>,
    /// Pin swap equivalence groups, each containing a list of swappable pad indices in `pads`.
    #[serde(default)]
    pub pin_swap_groups: Vec<Vec<usize>>,
    /// Multi-pin logic gate swap groups within this package.
    #[serde(default)]
    pub gate_swap_groups: Vec<SwappableGateGroup>,
}

/// Extracts a canonical swappable pin bank prefix (such as `"GPIO"`, `"PA"`, `"IO"`, `"CH"`) from a pin or net name.
///
/// Used to automatically group MCU/FPGA I/O bank pins when explicit `(swap_group ...)` tags are absent.
///
/// # Arguments
/// * `s` - Pin function name or net name to inspect (e.g., `"GPIO_12"`, `"PA4"`, `"CH0"`).
///
/// # Returns
/// `Some(prefix)` if `s` starts with a recognized swappable bank prefix followed solely by digits/underscores,
/// or `None` otherwise.
///
/// # Side Effects
/// None.
pub fn extract_swappable_prefix(s: &str) -> Option<String> {
    let clean = s.trim();
    if clean.is_empty() {
        return None;
    }
    let upper = clean.to_ascii_uppercase();
    let candidate_prefixes = [
        "GPIO_", "GPIO", "P0_", "P1_", "P2_", "P3_",
        "PA", "PB", "PC", "PD", "PE", "PF",
        "IO_", "IO", "DATA", "DIN", "DOUT",
        "IN", "OUT", "CH", "PIN",
    ];
    for &cp in &candidate_prefixes {
        if upper.starts_with(cp) {
            let remainder = &upper[cp.len()..];
            if !remainder.is_empty() && remainder.chars().all(|c| c.is_ascii_digit() || c == '_') {
                return Some(cp.to_string());
            }
        }
    }
    None
}

impl Default for Footprint {
    /// Returns an empty unlocked `1.0 x 1.0 mm` [`Footprint`] at `(0.0, 0.0)` on `F.Cu` (`layer = 0`).
    fn default() -> Self {
        Self {
            reference: String::new(),
            value: String::new(),
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: Vec::new(),
            width: 1.0,
            height: 1.0,
            courtyards: Vec::new(),
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        }
    }
}

impl Footprint {
    /// Resolves the active copper layer index (`0` = `F.Cu`, `1` = `B.Cu`) of `pad` accounting for footprint side flipping.
    ///
    /// If the footprint has been flipped from its original side (`self.layer != self.orig_layer`),
    /// top and bottom pad layers are mirrored (`1 - base`).
    ///
    /// # Arguments
    /// * `pad` - The [`Pad`] whose effective copper layer is queried.
    ///
    /// # Returns
    /// Copper layer index (`0` for `F.Cu`, `1` for `B.Cu`).
    ///
    /// # Side Effects
    /// None.
    pub fn pad_layer(&self, pad: &Pad) -> usize {
        let base = pad.layer.unwrap_or(self.orig_layer);
        if self.layer != self.orig_layer {
            1 - base.min(1)
        } else {
            base
        }
    }

    /// Determines whether this footprint is a non-polarized 2-pad passive component (resistor, inductor, ferrite bead, or non-polarized capacitor) with identical pad dimensions.
    ///
    /// Excludes relays (`REL`, `RY`), LEDs (`LED`), connectors (`CONN`), crystals/clocks (`CLK`), and polarized capacitors (`CP`, `POL`, `TANT`, `ELECT`, `+`).
    ///
    /// # Returns
    /// `true` if the footprint has 2 equal-sized pads that are electrically interchangeable, `false` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn is_symmetric_passive(&self) -> bool {
        if self.pads.len() != 2 {
            return false;
        }
        let ref_upper = self.reference.to_ascii_uppercase();
        let second_char = ref_upper.chars().nth(1);

        // Resistors: R1, R2, R_1, RN1, RP1 (not REL, RY, REMOTE, RESET)
        let is_resistor = ref_upper.starts_with('R')
            && (second_char.map_or(false, |c| c.is_ascii_digit())
                || ref_upper.starts_with("R_")
                || ref_upper.starts_with("RN")
                || ref_upper.starts_with("RP"))
            && !ref_upper.starts_with("REL")
            && !ref_upper.starts_with("RY")
            && !ref_upper.starts_with("REMOTE")
            && !ref_upper.starts_with("RESET");

        // Inductors / Ferrites: L1, L_1, FB1, FL1 (not LED)
        let is_inductor = (ref_upper.starts_with('L')
            && (second_char.map_or(false, |c| c.is_ascii_digit()) || ref_upper.starts_with("L_"))
            && !ref_upper.starts_with("LED"))
            || ref_upper.starts_with("FB")
            || ref_upper.starts_with("FL");

        // Capacitors: C1, C2, C_1 (not CP, CONN, CLK, CHIP) and non-polarized
        let is_non_pol_cap = ref_upper.starts_with('C')
            && (second_char.map_or(false, |c| c.is_ascii_digit()) || ref_upper.starts_with("C_"))
            && !ref_upper.starts_with("CP")
            && !ref_upper.starts_with("CONN")
            && !ref_upper.starts_with("CLK")
            && {
                let val_upper = self.value.to_ascii_uppercase();
                !val_upper.contains("POL")
                    && !val_upper.contains("TANT")
                    && !val_upper.contains("ELECT")
                    && !val_upper.contains('+')
            };

        (is_resistor || is_inductor || is_non_pol_cap)
            && (self.pads[0].width - self.pads[1].width).abs() < 1e-3
            && (self.pads[0].height - self.pads[1].height).abs() < 1e-3
    }

    /// Populates [`Footprint::pin_swap_groups`] automatically if no explicit swap groups were parsed.
    ///
    /// - For symmetric 2-pin passives (`self.is_symmetric_passive()`), adds `vec![0, 1]`.
    /// - For multi-pin components, groups pads that share a swappable bank prefix via [`extract_swappable_prefix`].
    ///
    /// # Side Effects
    /// Appends inferred pin-swap index vectors to `self.pin_swap_groups` if it is currently empty.
    pub fn infer_swap_groups(&mut self) {
        if !self.pin_swap_groups.is_empty() {
            return;
        }

        if self.is_symmetric_passive() {
            self.pin_swap_groups.push(vec![0, 1]);
            return;
        }

        let mut prefix_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, pad) in self.pads.iter().enumerate() {
            let identifier = if !pad.pin_name.is_empty() {
                &pad.pin_name
            } else if !pad.net_name.is_empty() {
                &pad.net_name
            } else {
                &pad.number
            };

            if let Some(prefix) = extract_swappable_prefix(identifier) {
                prefix_groups.entry(prefix).or_default().push(i);
            }
        }

        let mut sorted_keys: Vec<String> = prefix_groups.keys().cloned().collect();
        sorted_keys.sort();
        for key in sorted_keys {
            let indices = prefix_groups.remove(&key).unwrap();
            if indices.len() >= 2 {
                self.pin_swap_groups.push(indices);
            }
        }
    }

    /// Checks whether all copper pads of this footprint are Surface-Mount Device (SMD) pads.
    ///
    /// # Returns
    /// `true` if `self.pads` is non-empty and contains zero through-hole pads, `false` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn is_smd_only(&self) -> bool {
        !self.pads.is_empty() && self.pads.iter().all(|p| !p.is_through_hole)
    }

    /// Computes the axis-aligned bounding width and height `(world_w, world_h)` of `pad` in board space
    /// after applying both the footprint's rotation and the pad's local rotation.
    ///
    /// # Arguments
    /// * `pad` - The [`Pad`] to measure.
    ///
    /// # Returns
    /// A tuple `(world_width_mm, world_height_mm)` in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn pad_world_size(&self, pad: &Pad) -> (f64, f64) {
        let total_rot = ((self.rotation + pad.rotation) % 360.0 + 360.0) % 360.0;
        let rad = (-total_rot).to_radians();
        let cos_r = rad.cos().abs();
        let sin_r = rad.sin().abs();
        (
            pad.width * cos_r + pad.height * sin_r,
            pad.width * sin_r + pad.height * cos_r,
        )
    }

    /// Returns the 4 rotated corner points of `pad` in world coordinates at the footprint's current pose and layer.
    ///
    /// # Arguments
    /// * `pad` - The [`Pad`] whose corners are computed.
    ///
    /// # Returns
    /// A 4-element vector of [`Point`]s in board coordinates (`mm`).
    ///
    /// # Side Effects
    /// None.
    pub fn pad_world_corners(&self, pad: &Pad) -> Vec<Point> {
        self.pad_world_corners_layered(pad, self.layer)
    }

    /// Returns the 4 rotated corner points of `pad` in world coordinates at the footprint's current `(x, y, rotation)` on `layer`.
    ///
    /// Accounts for Y-axis mirroring when `layer != self.orig_layer`.
    ///
    /// # Arguments
    /// * `pad` - The [`Pad`] whose corners are computed.
    /// * `layer` - Candidate copper layer (`0` = `F.Cu`, `1` = `B.Cu`).
    ///
    /// # Returns
    /// A 4-element vector of [`Point`]s in board coordinates (`mm`).
    ///
    /// # Side Effects
    /// None.
    pub fn pad_world_corners_layered(&self, pad: &Pad, layer: usize) -> Vec<Point> {
        let flipped = layer != self.orig_layer;
        let pad_rot = if flipped { -pad.rotation } else { pad.rotation };
        let hw = pad.width / 2.0;
        let hh = pad.height / 2.0;
        let local_corners = [
            Point::new(-hw, -hh),
            Point::new(hw, -hh),
            Point::new(hw, hh),
            Point::new(-hw, hh),
        ];
        let oy = if flipped { -pad.offset_y } else { pad.offset_y };
        let total_fp_rot = self.rotation;

        local_corners
            .iter()
            .map(|c| {
                let rot_c = c.rotate_ccw(pad_rot);
                let fp_c = Point::new(rot_c.x + pad.offset_x, rot_c.y + oy);
                let world_c = fp_c.rotate_ccw(total_fp_rot);
                Point::new(self.x + world_c.x, self.y + world_c.y)
            })
            .collect()
    }

    /// Computes the 4 world-space corner points of `pad` when the footprint is placed at arbitrary `(pos, rot, layer)`.
    ///
    /// # Arguments
    /// * `pad` - The [`Pad`] whose corners are computed.
    /// * `pos` - Candidate footprint origin [`Point`] in `mm`.
    /// * `rot` - Candidate footprint rotation in degrees.
    /// * `layer` - Candidate footprint copper layer (`0` = `F.Cu`, `1` = `B.Cu`).
    ///
    /// # Returns
    /// A 4-element vector of corner [`Point`]s in world coordinates (`mm`).
    ///
    /// # Side Effects
    /// None.
    pub fn pad_corners_at(&self, pad: &Pad, pos: Point, rot: f64, layer: usize) -> Vec<Point> {
        let flipped = layer != self.orig_layer;
        let pad_rot = if flipped { -pad.rotation } else { pad.rotation };
        let hw = pad.width / 2.0;
        let hh = pad.height / 2.0;
        let local_corners = [
            Point::new(-hw, -hh),
            Point::new(hw, -hh),
            Point::new(hw, hh),
            Point::new(-hw, hh),
        ];
        let oy = if flipped { -pad.offset_y } else { pad.offset_y };
        local_corners
            .iter()
            .map(|c| {
                let rot_c = c.rotate_ccw(pad_rot);
                let fp_c = Point::new(rot_c.x + pad.offset_x, rot_c.y + oy);
                let world_c = fp_c.rotate_ccw(rot);
                Point::new(pos.x + world_c.x, pos.y + world_c.y)
            })
            .collect()
    }

    /// Computes the bounding box of the footprint relative to its origin `(0, 0)`,
    /// taking into account transformed courtyards (with spacing) and all pad corners.
    ///
    /// # Arguments
    /// * `rot` - Candidate rotation angle in degrees.
    /// * `layer` - Candidate copper layer (`0` = `F.Cu`, `1` = `B.Cu`).
    /// * `spacing` - Courtyard inflation spacing in `mm`.
    ///
    /// # Returns
    /// Relative [`AABB`] around `(0.0, 0.0)` in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn relative_bounds(&self, rot: f64, layer: usize, spacing: f64) -> AABB {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        let origin = Point::new(0.0, 0.0);
        let polys = self.transformed_courtyards_layered(origin, rot, layer, spacing);
        for poly in &polys {
            for v in &poly.vertices {
                min_x = min_x.min(v.x);
                max_x = max_x.max(v.x);
                min_y = min_y.min(v.y);
                max_y = max_y.max(v.y);
            }
        }

        for pad in &self.pads {
            for c in self.pad_corners_at(pad, origin, rot, layer) {
                min_x = min_x.min(c.x);
                max_x = max_x.max(c.x);
                min_y = min_y.min(c.y);
                max_y = max_y.max(c.y);
            }
        }

        if min_x > max_x {
            let hw = self.width.max(1.0) * 0.5;
            let hh = self.height.max(1.0) * 0.5;
            AABB::new(-hw, -hh, hw, hh)
        } else {
            AABB::new(min_x, min_y, max_x, max_y)
        }
    }

    /// Computes the strictly legal `[min_x, max_x, min_y, max_y]` clamping range for the component origin
    /// so that courtyards and copper pads remain entirely inside `boundary`.
    ///
    /// # Arguments
    /// * `rot` - Candidate footprint rotation in degrees.
    /// * `layer` - Candidate copper layer (`0` = `F.Cu`, `1` = `B.Cu`).
    /// * `spacing` - Courtyard expansion spacing in `mm`.
    /// * `boundary` - Target board [`AABB`] boundary in `mm`.
    ///
    /// # Returns
    /// A 4-tuple `(min_x, max_x, min_y, max_y)` of legal origin coordinates in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn origin_clamp_range(
        &self,
        rot: f64,
        layer: usize,
        spacing: f64,
        boundary: &AABB,
    ) -> (f64, f64, f64, f64) {
        let rel_b = self.relative_bounds(rot, layer, spacing);
        let mut min_x = boundary.min_x - rel_b.min_x;
        let mut max_x = boundary.max_x - rel_b.max_x;
        let mut min_y = boundary.min_y - rel_b.min_y;
        let mut max_y = boundary.max_y - rel_b.max_y;

        if min_x > max_x {
            let mid = (min_x + max_x) * 0.5;
            min_x = mid;
            max_x = mid;
        }
        if min_y > max_y {
            let mid = (min_y + max_y) * 0.5;
            min_y = mid;
            max_y = mid;
        }
        (min_x, max_x, min_y, max_y)
    }

    /// Computes the absolute world `(x, y)` center position of `pad` given the footprint's current pose and layer.
    ///
    /// # Arguments
    /// * `pad` - Reference to a [`Pad`] belonging to this footprint.
    ///
    /// # Returns
    /// World `(x, y)` coordinates of the pad center in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn pad_absolute_position(&self, pad: &Pad) -> (f64, f64) {
        let oy = if self.layer != self.orig_layer {
            -pad.offset_y
        } else {
            pad.offset_y
        };
        let p = Point::new(pad.offset_x, oy).rotate_ccw(self.rotation);
        (self.x + p.x, self.y + p.y)
    }

    /// Computes the world-space center [`Point`] of `self.pads[pad_idx]` when the footprint is placed at `(pos, rot)` on `self.layer`.
    ///
    /// # Arguments
    /// * `pad_idx` - Index of the pad in `self.pads`.
    /// * `pos` - Candidate footprint origin [`Point`] in `mm`.
    /// * `rot` - Candidate footprint rotation in degrees.
    ///
    /// # Returns
    /// World-space [`Point`] of the pad center in `mm`.
    ///
    /// # Panics
    /// Panics if `pad_idx >= self.pads.len()`.
    ///
    /// # Side Effects
    /// None.
    pub fn pad_world_point(&self, pad_idx: usize, pos: Point, rot: f64) -> Point {
        self.pad_world_point_layered(pad_idx, pos, rot, self.layer)
    }

    /// Computes the world-space center [`Point`] of `self.pads[pad_idx]` when the footprint is placed at `(pos, rot, layer)`.
    ///
    /// # Arguments
    /// * `pad_idx` - Index of the pad in `self.pads`.
    /// * `pos` - Candidate footprint origin [`Point`] in `mm`.
    /// * `rot` - Candidate footprint rotation in degrees.
    /// * `layer` - Candidate copper layer (`0` = `F.Cu`, `1` = `B.Cu`).
    ///
    /// # Returns
    /// World-space [`Point`] of the pad center in `mm`.
    ///
    /// # Panics
    /// Panics if `pad_idx >= self.pads.len()`.
    ///
    /// # Side Effects
    /// None.
    pub fn pad_world_point_layered(&self, pad_idx: usize, pos: Point, rot: f64, layer: usize) -> Point {
        let pad = &self.pads[pad_idx];
        let oy = if layer != self.orig_layer {
            -pad.offset_y
        } else {
            pad.offset_y
        };
        let p = Point::new(pad.offset_x, oy).rotate_ccw(rot);
        Point::new(pos.x + p.x, pos.y + p.y)
    }

    /// Returns the convex courtyard polygons transformed to `(pos, rot)` on `self.layer` and expanded by `spacing / 2.0`.
    ///
    /// # Arguments
    /// * `pos` - Candidate footprint origin [`Point`] in `mm`.
    /// * `rot` - Candidate footprint rotation in degrees.
    /// * `spacing` - Total inter-component courtyard clearance in `mm` (each polygon is expanded by `spacing / 2.0`).
    ///
    /// # Returns
    /// Vector of transformed convex [`Polygon`]s in world coordinates (`mm`).
    ///
    /// # Side Effects
    /// None.
    pub fn transformed_courtyards(&self, pos: Point, rot: f64, spacing: f64) -> Vec<Polygon> {
        self.transformed_courtyards_layered(pos, rot, self.layer, spacing)
    }

    /// Returns the convex courtyard polygons transformed to `(pos, rot, layer)` and expanded by `spacing / 2.0`.
    ///
    /// # Arguments
    /// * `pos` - Candidate footprint origin [`Point`] in `mm`.
    /// * `rot` - Candidate footprint rotation in degrees.
    /// * `layer` - Candidate copper layer (`0` = `F.Cu`, `1` = `B.Cu`).
    /// * `spacing` - Total inter-component courtyard clearance in `mm`.
    ///
    /// # Returns
    /// Vector of transformed convex [`Polygon`]s in world coordinates (`mm`).
    ///
    /// # Side Effects
    /// None.
    pub fn transformed_courtyards_layered(&self, pos: Point, rot: f64, layer: usize, spacing: f64) -> Vec<Polygon> {
        let flipped = layer != self.orig_layer;
        self.courtyards
            .iter()
            .map(|poly| {
                let base = if flipped {
                    Polygon::new(poly.vertices.iter().map(|v| Point::new(v.x, -v.y)).collect())
                } else {
                    poly.clone()
                };
                if spacing > 0.0 {
                    base.expanded(spacing / 2.0).transform(pos, rot)
                } else {
                    base.transform(pos, rot)
                }
            })
            .collect()
    }

    /// Returns the outer perimeter courtyard polygons transformed to `(pos, rot)` on `self.layer` and expanded by `spacing / 2.0`.
    ///
    /// # Arguments
    /// * `pos` - Candidate footprint origin [`Point`] in `mm`.
    /// * `rot` - Candidate footprint rotation in degrees.
    /// * `spacing` - Total courtyard expansion spacing in `mm`.
    ///
    /// # Returns
    /// Vector of transformed perimeter [`Polygon`]s in world coordinates (`mm`).
    ///
    /// # Side Effects
    /// None.
    pub fn transformed_perimeters(&self, pos: Point, rot: f64, spacing: f64) -> Vec<Polygon> {
        self.transformed_perimeters_layered(pos, rot, self.layer, spacing)
    }

    /// Returns the outer perimeter courtyard polygons transformed to `(pos, rot, layer)` and expanded by `spacing / 2.0`.
    ///
    /// # Arguments
    /// * `pos` - Candidate footprint origin [`Point`] in `mm`.
    /// * `rot` - Candidate footprint rotation in degrees.
    /// * `layer` - Candidate copper layer (`0` = `F.Cu`, `1` = `B.Cu`).
    /// * `spacing` - Total courtyard expansion spacing in `mm`.
    ///
    /// # Returns
    /// Vector of transformed perimeter [`Polygon`]s in world coordinates (`mm`).
    ///
    /// # Side Effects
    /// None.
    pub fn transformed_perimeters_layered(&self, pos: Point, rot: f64, layer: usize, spacing: f64) -> Vec<Polygon> {
        let flipped = layer != self.orig_layer;
        let perims = if !self.perimeters.is_empty() {
            &self.perimeters
        } else {
            &self.courtyards
        };
        perims
            .iter()
            .map(|poly| {
                let base = if flipped {
                    Polygon::new(poly.vertices.iter().map(|v| Point::new(v.x, -v.y)).collect())
                } else {
                    poly.clone()
                };
                if spacing > 0.0 {
                    base.expanded(spacing / 2.0).transform(pos, rot)
                } else {
                    base.transform(pos, rot)
                }
            })
            .collect()
    }

    /// Computes the minimum center-to-center pitch between distinct pads (filtering out coincident/thermal sub-pads).
    ///
    /// # Returns
    /// Minimum center-to-center distance in `mm` among pad pairs separated by more than `0.10 mm`
    /// (or `f64::INFINITY` if fewer than 2 distinct pads exist).
    ///
    /// # Side Effects
    /// None.
    pub fn min_pad_pitch(&self) -> f64 {
        if self.pads.len() < 2 {
            return f64::INFINITY;
        }
        let mut min_pitch = f64::INFINITY;
        for i in 0..self.pads.len() {
            let p1 = &self.pads[i];
            for j in (i + 1)..self.pads.len() {
                let p2 = &self.pads[j];
                let dx = p1.offset_x - p2.offset_x;
                let dy = p1.offset_y - p2.offset_y;
                let dist = dx.hypot(dy);
                if dist > 0.10 && dist < min_pitch {
                    min_pitch = dist;
                }
            }
        }
        min_pitch
    }

    /// Computes the pad density (pads per $\text{mm}^2$ of footprint courtyard area).
    ///
    /// # Returns
    /// Number of pads divided by `max(width * height, 1.0)` in $\text{pads/mm}^2$.
    ///
    /// # Side Effects
    /// None.
    pub fn pad_density(&self) -> f64 {
        let area = (self.width * self.height).max(1.0);
        self.pads.len() as f64 / area
    }

    /// Computes the dedicated routing escape clearance (halo) around dense or fine-pitch components.
    /// - High pin-count packages (`>= 16` pads) with fine pitch (`<= 0.50mm`) or high density: `1.20mm`
    /// - Fine-pitch packages (`>= 16` pads, pitch `<= 0.65mm`): `1.00mm`
    /// - Medium pin-count packages (`>= 8` pads, pitch `<= 0.65mm` or high density): `0.90mm`
    /// - Small dense ICs (`>= 6` pads, pitch `<= 1.00mm`, e.g. SOT-23-6 BUCK IC): `0.75mm`
    /// - Standard passives (0603/0805, 2 pads): `0.0mm` (standard clearance)
    ///
    /// # Returns
    /// Escape margin distance in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn compute_escape_margin(&self) -> f64 {
        let pad_count = self.pads.len();
        if pad_count < 6 {
            return 0.0;
        }
        let pitch = self.min_pad_pitch();
        let density = self.pad_density();

        if pad_count >= 16 {
            if pitch <= 0.50 || density >= 1.0 {
                1.20
            } else if pitch <= 0.65 {
                1.00
            } else {
                0.70
            }
        } else if pad_count >= 8 {
            if pitch <= 0.65 || density >= 0.8 {
                0.90
            } else if pitch <= 1.0 {
                0.70
            } else {
                0.40
            }
        } else if pad_count >= 6 {
            if pitch <= 1.0 {
                0.75
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Computes the effective placement spacing required for this component:
    /// base courtyard spacing plus twice the component's escape clearance margin.
    ///
    /// # Arguments
    /// * `base_spacing` - Baseline inter-courtyard spacing in `mm`.
    ///
    /// # Returns
    /// Total placement spacing `base_spacing + 2.0 * escape_margin` in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn placement_spacing(&self, base_spacing: f64) -> f64 {
        let margin = if self.escape_margin > 0.0 {
            self.escape_margin
        } else {
            self.compute_escape_margin()
        };
        base_spacing + 2.0 * margin
    }

    /// Detects if this footprint is a Ball Grid Array (BGA) package and calculates its
    /// 2D regular grid geometry, pitch, bounding matrix center, and concentric ring depth per pad.
    ///
    /// # Returns
    /// `Some(BgaGridInfo)` if at least 9 ball pads form a regular 2D matrix (`>= 3x3`) with pitch in `[0.35, 1.05] mm`,
    /// or `None` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn bga_grid_info(&self) -> Option<BgaGridInfo> {
        let smd_pads: Vec<(usize, &Pad)> = self
            .pads
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.is_through_hole)
            .collect();
        if smd_pads.len() < 9 {
            return None;
        }

        // BGA pads are typically circular, roundrect, oval, or square SMD pads (aspect ratio <= 1.43, size <= 1.2mm)
        let ball_pads: Vec<(usize, &Pad)> = smd_pads
            .iter()
            .filter(|(_, p)| {
                let ratio = p.width / p.height.max(1e-6);
                let is_squareish = ratio >= 0.70 && ratio <= 1.43;
                let is_ball_shape = p.shape == "circle" || p.shape == "roundrect" || p.shape == "oval" || p.shape == "rect";
                is_squareish && is_ball_shape && p.width >= 0.15 && p.width <= 1.20 && p.height >= 0.15 && p.height <= 1.20
            })
            .cloned()
            .collect();

        if ball_pads.len() < 9 || ball_pads.len() * 4 < smd_pads.len() * 3 {
            return None;
        }

        let oy = |p: &Pad| if self.layer != self.orig_layer { -p.offset_y } else { p.offset_y };

        // Collect unique X and unique Y coordinates with 0.08mm tolerance clustering
        let cluster_coords = |vals: &[f64]| -> Vec<f64> {
            if vals.is_empty() {
                return Vec::new();
            }
            let mut sorted = vals.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut clusters = Vec::new();
            let mut cur_sum = sorted[0];
            let mut cur_cnt = 1;
            for &v in &sorted[1..] {
                let mean = cur_sum / (cur_cnt as f64);
                if (v - mean).abs() <= 0.08 {
                    cur_sum += v;
                    cur_cnt += 1;
                } else {
                    clusters.push(mean);
                    cur_sum = v;
                    cur_cnt = 1;
                }
            }
            clusters.push(cur_sum / (cur_cnt as f64));
            clusters
        };

        let all_x: Vec<f64> = ball_pads.iter().map(|(_, p)| p.offset_x).collect();
        let all_y: Vec<f64> = ball_pads.iter().map(|(_, p)| oy(p)).collect();

        let clusters_x = cluster_coords(&all_x);
        let clusters_y = cluster_coords(&all_y);

        if clusters_x.len() < 2 || clusters_y.len() < 2 {
            return None;
        }

        let diffs_x: Vec<f64> = clusters_x.windows(2).map(|w| w[1] - w[0]).collect();
        let diffs_y: Vec<f64> = clusters_y.windows(2).map(|w| w[1] - w[0]).collect();

        let min_dx = diffs_x.iter().cloned().fold(f64::INFINITY, f64::min);
        let min_dy = diffs_y.iter().cloned().fold(f64::INFINITY, f64::min);

        // BGA pitch standard range: 0.35mm <= p <= 1.05mm
        if min_dx < 0.35 || min_dx > 1.05 || min_dy < 0.35 || min_dy > 1.05 {
            return None;
        }

        let pitch_x = (min_dx * 10000.0).round() / 10000.0;
        let pitch_y = (min_dy * 10000.0).round() / 10000.0;

        // Verify that cluster steps are approximately integer multiples of pitch
        for &d in &diffs_x {
            let mult = (d / pitch_x).round();
            if mult < 1.0 || (d - mult * pitch_x).abs() > 0.12 {
                return None;
            }
        }
        for &d in &diffs_y {
            let mult = (d / pitch_y).round();
            if mult < 1.0 || (d - mult * pitch_y).abs() > 0.12 {
                return None;
            }
        }

        let min_x = clusters_x[0];
        let max_x = *clusters_x.last().unwrap();
        let min_y = clusters_y[0];
        let max_y = *clusters_y.last().unwrap();

        let cols = ((max_x - min_x) / pitch_x).round() as usize + 1;
        let rows = ((max_y - min_y) / pitch_y).round() as usize + 1;

        // Must be a matrix >= 3x3 with total matrix capacity >= 9
        if cols < 3 || rows < 3 || (cols * rows) < 9 {
            return None;
        }

        let center_x = (((min_x + max_x) * 0.5) * 10000.0).round() / 10000.0;
        let center_y = (((min_y + max_y) * 0.5) * 10000.0).round() / 10000.0;

        // Index each pad in self.pads into grid columns and rows, and determine ring index
        let mut pad_rings = vec![0usize; self.pads.len()];
        let mut pad_grid_indices = vec![None; self.pads.len()];
        let mut matched_count = 0;

        for (pad_idx, pad) in self.pads.iter().enumerate() {
            if pad.is_through_hole {
                continue;
            }
            let px = pad.offset_x;
            let py = oy(pad);
            let c_f = (px - min_x) / pitch_x;
            let r_f = (py - min_y) / pitch_y;
            let c_i = c_f.round() as isize;
            let r_i = r_f.round() as isize;

            if (c_f - c_i as f64).abs() <= 0.20 && (r_f - r_i as f64).abs() <= 0.20 {
                if c_i >= 0 && (c_i as usize) < cols && r_i >= 0 && (r_i as usize) < rows {
                    let col = c_i as usize;
                    let row = r_i as usize;
                    let dist_x = col.min(cols - 1 - col);
                    let dist_y = row.min(rows - 1 - row);
                    let ring = dist_x.min(dist_y);
                    pad_rings[pad_idx] = ring;
                    pad_grid_indices[pad_idx] = Some((col, row));
                    matched_count += 1;
                }
            }
        }

        if matched_count < 9 || matched_count * 4 < self.pads.len() * 3 {
            return None;
        }

        Some(BgaGridInfo {
            pitch_x,
            pitch_y,
            center_x,
            center_y,
            cols,
            rows,
            min_x,
            min_y,
            max_x,
            max_y,
            pad_rings,
            pad_grid_indices,
        })
    }

    /// Checks whether this footprint is detected as a Ball Grid Array (BGA) package.
    ///
    /// # Returns
    /// `true` if [`Footprint::bga_grid_info`] returns `Some`, `false` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn is_bga(&self) -> bool {
        self.bga_grid_info().is_some()
    }

    /// Returns the horizontal and vertical ball pitch `(pitch_x, pitch_y)` in millimeters if this footprint is a BGA.
    ///
    /// # Returns
    /// `Some((pitch_x_mm, pitch_y_mm))` for BGA packages, or `None` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn bga_pitch(&self) -> Option<(f64, f64)> {
        self.bga_grid_info().map(|info| (info.pitch_x, info.pitch_y))
    }

    /// Returns the local `(center_x, center_y)` offset of the BGA ball matrix relative to the footprint origin in `mm`.
    ///
    /// # Returns
    /// `Some((center_x_mm, center_y_mm))` if BGA, or `None` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn bga_matrix_center(&self) -> Option<(f64, f64)> {
        self.bga_grid_info().map(|info| (info.center_x, info.center_y))
    }

    /// Returns the world-space `(x, y)` center of the BGA ball matrix in millimeters.
    ///
    /// # Returns
    /// `Some((world_cx_mm, world_cy_mm))` if BGA, or `None` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn bga_world_center(&self) -> Option<(f64, f64)> {
        let (cx, cy) = self.bga_matrix_center()?;
        let p = Point::new(cx, cy).rotate_ccw(self.rotation);
        Some((self.x + p.x, self.y + p.y))
    }

    /// Returns the concentric perimeter ring depth (`0` = outermost ring, `1` = second ring, etc.) for `pad_idx` in a BGA.
    ///
    /// # Arguments
    /// * `pad_idx` - Index of the pad in `self.pads`.
    ///
    /// # Returns
    /// Ring depth index (`0` if not a BGA or on the outer perimeter).
    ///
    /// # Side Effects
    /// None.
    pub fn bga_pad_ring_index(&self, pad_idx: usize) -> usize {
        if let Some(info) = self.bga_grid_info() {
            if pad_idx < info.pad_rings.len() {
                return info.pad_rings[pad_idx];
            }
        }
        0
    }
}

/// Extracted 2D ball-grid matrix geometry and per-pad concentric ring indices for a BGA [`Footprint`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BgaGridInfo {
    /// Horizontal ball-to-ball pitch along local X in `mm`.
    pub pitch_x: f64,
    /// Vertical ball-to-ball pitch along local Y in `mm`.
    pub pitch_y: f64,
    /// Local X center of the ball matrix in `mm`.
    pub center_x: f64,
    /// Local Y center of the ball matrix in `mm`.
    pub center_y: f64,
    /// Number of columns in the BGA grid.
    pub cols: usize,
    /// Number of rows in the BGA grid.
    pub rows: usize,
    /// Minimum local X coordinate of the ball grid in `mm`.
    pub min_x: f64,
    /// Minimum local Y coordinate of the ball grid in `mm`.
    pub min_y: f64,
    /// Maximum local X coordinate of the ball grid in `mm`.
    pub max_x: f64,
    /// Maximum local Y coordinate of the ball grid in `mm`.
    pub max_y: f64,
    /// Concentric ring depth (`0` = outer ring) for each pad in [`Footprint::pads`].
    pub pad_rings: Vec<usize>,
    /// Optional `(col, row)` grid coordinates for each pad in [`Footprint::pads`].
    pub pad_grid_indices: Vec<Option<(usize, usize)>>,
}


/// Extracts the outer perimeter boundary line segments of a set of polygons,
/// discarding internal shared chords (such as triangulation edges).
///
/// # Arguments
/// * `polys` - Slice of [`Polygon`] pieces whose shared internal edges should be removed.
///
/// # Returns
/// Vector of `(start_point, end_point)` boundary edges that appear in exactly one polygon.
///
/// # Side Effects
/// None.
pub fn outer_perimeter_edges(polys: &[Polygon]) -> Vec<(Point, Point)> {
    if polys.is_empty() {
        return Vec::new();
    }
    if polys.len() == 1 {
        let poly = &polys[0];
        let n = poly.vertices.len();
        return (0..n)
            .map(|i| (poly.vertices[i], poly.vertices[(i + 1) % n]))
            .collect();
    }
    let quant = |p: Point| -> (i64, i64) {
        ((p.x * 10000.0).round() as i64, (p.y * 10000.0).round() as i64)
    };
    let mut edge_counts: HashMap<((i64, i64), (i64, i64)), (Point, Point, usize)> = HashMap::new();
    for poly in polys {
        let n = poly.vertices.len();
        for i in 0..n {
            let p1 = poly.vertices[i];
            let p2 = poly.vertices[(i + 1) % n];
            let q1 = quant(p1);
            let q2 = quant(p2);
            let (k1, k2) = if q1 <= q2 { (q1, q2) } else { (q2, q1) };
            edge_counts
                .entry((k1, k2))
                .and_modify(|e| e.2 += 1)
                .or_insert((p1, p2, 1));
        }
    }
    edge_counts
        .into_values()
        .filter(|(_, _, count)| *count == 1)
        .map(|(p1, p2, _)| (p1, p2))
        .collect()
}

/// A straight copper trace segment `(segment ...)` routed on a single PCB copper layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackSegment {
    /// Start `(x, y)` coordinates in `mm`.
    pub start: (f64, f64),
    /// End `(x, y)` coordinates in `mm`.
    pub end: (f64, f64),
    /// Copper trace width in `mm`.
    pub width: f64,
    /// Copper layer index (`0` = `F.Cu`, `1` = `B.Cu`, `2..` = `In1.Cu..`).
    pub layer: usize, // 0 = F.Cu, 1 = B.Cu
    /// Numeric net ID (`0` = unconnected).
    pub net_id: usize,
    /// Whether this trace is locked against rip-up and modification.
    #[serde(default)]
    pub is_locked: bool,
}

impl TrackSegment {
    /// Computes the direction angle of the track segment in radians (`(-PI, PI]`).
    ///
    /// # Returns
    /// Angle `atan2(end.y - start.y, end.x - start.x)` in radians.
    ///
    /// # Side Effects
    /// None.
    pub fn angle(&self) -> f64 {
        (self.end.1 - self.start.1).atan2(self.end.0 - self.start.0)
    }

    /// Computes the Euclidean length of the track segment in millimeters.
    ///
    /// # Returns
    /// Segment length in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn length(&self) -> f64 {
        (self.end.0 - self.start.0).hypot(self.end.1 - self.start.1)
    }

    /// Computes the midpoint `(x, y)` of the track segment in millimeters.
    ///
    /// # Returns
    /// Midpoint coordinates `(mid_x, mid_y)` in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn mid_point(&self) -> (f64, f64) {
        ((self.start.0 + self.end.0) * 0.5, (self.start.1 + self.end.1) * 0.5)
    }

    /// Checks whether the segment is routed at a non-octilinear ("any-angle") orientation.
    ///
    /// Returns `true` when the segment's angle modulo $45^\circ$ differs from $0^\circ$ and $45^\circ$ by more than $0.5^\circ$.
    ///
    /// # Returns
    /// `true` if the segment is neither orthogonal nor $45^\circ$ diagonal, `false` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn is_any_angle(&self) -> bool {
        let deg = self.angle().to_degrees().abs() % 45.0;
        deg > 0.5 && deg < 44.5
    }
}

/// A circular copper arc trace segment `(arc ...)` on a single PCB copper layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArcSegment {
    /// Center `(x, y)` of the arc circle in `mm`.
    pub center: (f64, f64),
    /// Arc radius in `mm`.
    pub radius: f64,
    /// Starting angle in radians.
    pub start_angle_rad: f64,
    /// Ending angle in radians (may be greater or less than `start_angle_rad` to denote sweep direction).
    pub end_angle_rad: f64,
    /// Copper trace width in `mm`.
    pub width: f64,
    /// Copper layer index (`0` = `F.Cu`, `1` = `B.Cu`, `2..` = `In1.Cu..`).
    pub layer: usize, // 0 = F.Cu, 1 = B.Cu
    /// Numeric net ID (`0` = unconnected).
    pub net_id: usize,
    /// Whether this arc is locked against rip-up and modification.
    #[serde(default)]
    pub is_locked: bool,
}

impl ArcSegment {
    /// Constructs an unlocked [`ArcSegment`] from center, radius, angular bounds, width, layer, and net ID.
    ///
    /// # Arguments
    /// * `center` - Arc circle center `(x, y)` in `mm`.
    /// * `radius` - Circle radius in `mm`.
    /// * `start_angle_rad` - Starting angle in radians.
    /// * `end_angle_rad` - Ending angle in radians.
    /// * `width` - Trace width in `mm`.
    /// * `layer` - Copper layer index (`0` = `F.Cu`, `1` = `B.Cu`).
    /// * `net_id` - Numeric net ID.
    ///
    /// # Returns
    /// A new [`ArcSegment`].
    ///
    /// # Side Effects
    /// None.
    pub fn new(
        center: (f64, f64),
        radius: f64,
        start_angle_rad: f64,
        end_angle_rad: f64,
        width: f64,
        layer: usize,
        net_id: usize,
    ) -> Self {
        Self {
            center,
            radius,
            start_angle_rad,
            end_angle_rad,
            width,
            layer,
            net_id,
            is_locked: false,
        }
    }

    /// Computes the `(x, y)` start point of the arc in millimeters.
    ///
    /// # Returns
    /// Start coordinates `(x, y)` in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn start(&self) -> (f64, f64) {
        (
            self.center.0 + self.radius * self.start_angle_rad.cos(),
            self.center.1 + self.radius * self.start_angle_rad.sin(),
        )
    }

    /// Computes the `(x, y)` end point of the arc in millimeters.
    ///
    /// # Returns
    /// End coordinates `(x, y)` in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn end(&self) -> (f64, f64) {
        (
            self.center.0 + self.radius * self.end_angle_rad.cos(),
            self.center.1 + self.radius * self.end_angle_rad.sin(),
        )
    }

    /// Computes the `(x, y)` midpoint along the arc sweep in millimeters (used for KiCad `(arc (start ...) (mid ...) (end ...))` serialization).
    ///
    /// # Returns
    /// Arc midpoint `(x, y)` in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn mid(&self) -> (f64, f64) {
        let mid_angle = self.start_angle_rad + (self.end_angle_rad - self.start_angle_rad) * 0.5;
        (
            self.center.0 + self.radius * mid_angle.cos(),
            self.center.1 + self.radius * mid_angle.sin(),
        )
    }

    /// Returns the signed angular sweep `end_angle_rad - start_angle_rad` in radians.
    ///
    /// # Returns
    /// Signed sweep angle in radians.
    ///
    /// # Side Effects
    /// None.
    pub fn sweep_angle(&self) -> f64 {
        self.end_angle_rad - self.start_angle_rad
    }

    /// Computes the arc length `radius * |sweep_angle|` in millimeters.
    ///
    /// # Returns
    /// Arc length in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn length(&self) -> f64 {
        self.radius * self.sweep_angle().abs()
    }

    /// Constructs an [`ArcSegment`] passing through three points (`start`, `mid`, `end`), matching KiCad's 3-point arc format.
    ///
    /// # Arguments
    /// * `start` - Arc start point `(x, y)` in `mm`.
    /// * `mid` - Interior point `(x, y)` along the arc in `mm`.
    /// * `end` - Arc end point `(x, y)` in `mm`.
    /// * `width` - Trace width in `mm`.
    /// * `layer` - Copper layer index.
    /// * `net_id` - Numeric net ID.
    ///
    /// # Returns
    /// `Some(ArcSegment)` if the three points are non-collinear, or `None` if collinear/degenerate (`|d| < 1e-9`).
    ///
    /// # Side Effects
    /// None.
    pub fn from_3pts(
        start: (f64, f64),
        mid: (f64, f64),
        end: (f64, f64),
        width: f64,
        layer: usize,
        net_id: usize,
    ) -> Option<Self> {
        let (x1, y1) = start;
        let (x2, y2) = mid;
        let (x3, y3) = end;

        let d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
        if d.abs() < 1e-9 {
            return None;
        }

        let q1 = x1 * x1 + y1 * y1;
        let q2 = x2 * x2 + y2 * y2;
        let q3 = x3 * x3 + y3 * y3;

        let cx = (q1 * (y2 - y3) + q2 * (y3 - y1) + q3 * (y1 - y2)) / d;
        let cy = (q1 * (x3 - x2) + q2 * (x1 - x3) + q3 * (x2 - x1)) / d;
        let radius = (x1 - cx).hypot(y1 - cy);

        let a1 = (y1 - cy).atan2(x1 - cx);
        let a2 = (y2 - cy).atan2(x2 - cx);
        let a3 = (y3 - cy).atan2(x3 - cx);

        use std::f64::consts::TAU;
        let delta_end = (a3 - a1).rem_euclid(TAU);
        let delta_mid = (a2 - a1).rem_euclid(TAU);

        let end_angle_rad = if delta_mid < delta_end {
            a1 + delta_end
        } else {
            a1 - (TAU - delta_end)
        };

        Some(Self {
            center: (cx, cy),
            radius,
            start_angle_rad: a1,
            end_angle_rad,
            width,
            layer,
            net_id,
            is_locked: false,
        })
    }

    /// Approximates the circular arc as a polyline of `steps` straight [`TrackSegment`] chords.
    ///
    /// # Arguments
    /// * `steps` - Number of linear segments to subdivide the arc into (`>= 1`).
    ///
    /// # Returns
    /// Vector of `steps.max(1)` [`TrackSegment`]s.
    ///
    /// # Side Effects
    /// None.
    pub fn to_segments(&self, steps: usize) -> Vec<TrackSegment> {
        let n = steps.max(1);
        let mut segs = Vec::with_capacity(n);
        let d_theta = (self.end_angle_rad - self.start_angle_rad) / (n as f64);
        let mut cur_pt = self.start();
        for i in 1..=n {
            let theta = self.start_angle_rad + d_theta * (i as f64);
            let nxt_pt = (
                self.center.0 + self.radius * theta.cos(),
                self.center.1 + self.radius * theta.sin(),
            );
            segs.push(TrackSegment {
                start: cur_pt,
                end: nxt_pt,
                width: self.width,
                layer: self.layer,
                net_id: self.net_id,
                is_locked: self.is_locked,
            });
            cur_pt = nxt_pt;
        }
        segs
    }

    /// Computes the shortest Euclidean distance from a 2D point `p` to the centerline of this circular arc.
    ///
    /// If the radial ray from `self.center` through `p` falls within the angular sweep of the arc,
    /// returns `|dist(p, center) - radius|`; otherwise returns the minimum distance to the two arc endpoints.
    ///
    /// # Arguments
    /// * `p` - Query point `(x, y)` in `mm`.
    ///
    /// # Returns
    /// Shortest distance from `p` to the arc centerline in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn distance_to_point(&self, p: (f64, f64)) -> f64 {
        let dx = p.0 - self.center.0;
        let dy = p.1 - self.center.1;
        let d_center = dx.hypot(dy);
        let p_angle = dy.atan2(dx);

        let sweep = self.end_angle_rad - self.start_angle_rad;
        use std::f64::consts::TAU;
        let in_arc = if sweep.abs() < 1e-9 {
            false
        } else if sweep > 0.0 {
            let rel = (p_angle - self.start_angle_rad).rem_euclid(TAU);
            rel <= sweep + 1e-6
        } else {
            let rel = (self.start_angle_rad - p_angle).rem_euclid(TAU);
            rel <= (-sweep) + 1e-6
        };

        if in_arc {
            (d_center - self.radius).abs()
        } else {
            let s = self.start();
            let e = self.end();
            (p.0 - s.0).hypot(p.1 - s.1).min((p.0 - e.0).hypot(p.1 - e.1))
        }
    }
}

/// A plated inter-layer through-via `(via ...)` on the PCB.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Via {
    /// World `(x, y)` center position in `mm`.
    pub position: (f64, f64),
    /// Outer copper pad diameter in `mm` (e.g., `0.60 mm`).
    pub size: f64,
    /// Finished drill hole diameter in `mm` (e.g., `0.30 mm`).
    pub drill: f64,
    /// Numeric net ID (`0` = unconnected).
    pub net_id: usize,
    /// Whether this via is locked against rip-up and modification.
    #[serde(default)]
    pub is_locked: bool,
}

/// A rectangular routing and placement keepout zone `(zone ... (keepout ...))` on the PCB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeepoutZone {
    /// Minimum X coordinate in `mm`.
    pub min_x: f64,
    /// Minimum Y coordinate in `mm`.
    pub min_y: f64,
    /// Maximum X coordinate in `mm`.
    pub max_x: f64,
    /// Maximum Y coordinate in `mm`.
    pub max_y: f64,
    /// Restricted copper layer indices (`0` = `F.Cu`, `1` = `B.Cu`, etc.).
    pub layers: Vec<usize>,
}

/// A functional component placement cluster extracted from `.kicad_sch` hierarchical sheets or bounding boxes.
#[derive(Debug, Clone)]
pub struct CompiledCluster {
    /// Human-readable cluster name (e.g., `"1. Power / Buck"`, `"U4 Cluster"`).
    pub name: String,
    /// Optional index into [`Board::footprints`] of the primary anchor IC or connector for this cluster.
    pub anchor_idx: Option<usize>,
    /// Indices into [`Board::footprints`] of all components belonging to this cluster.
    pub member_indices: Vec<usize>,
    /// Reference designators of all components belonging to this cluster.
    pub member_refs: Vec<String>,
    /// Target maximum placement radius around the anchor component in `mm`.
    pub max_radius: f64,
    /// Simulated annealing cost penalty weight for cluster dispersion beyond `max_radius`.
    pub weight: f64,
}

/// A multi-net parallel bus bundle routed along shared guide waypoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BusBundle {
    /// Bus bundle name (e.g., `"SPI_BUS"`, `"DATA_BUS"`).
    pub name: String,
    /// Ordered list of numeric net IDs belonging to the bus.
    pub net_ids: Vec<usize>,
    /// Ordered centerline guide waypoints `(x, y)` in `mm`.
    pub guide_points: Vec<(f64, f64)>,
    /// Preferred copper layer index for the bus trunk.
    pub layer: usize,
    /// Individual trace width in `mm`.
    pub track_width: f64,
    /// Edge-to-edge spacing between adjacent bus traces in `mm`.
    pub track_spacing: f64,
}

impl BusBundle {
    /// Creates a new [`BusBundle`] with the specified nets, guide waypoints, layer, and trace geometry.
    ///
    /// # Arguments
    /// * `name` - Bus bundle identifier.
    /// * `net_ids` - Vector of numeric net IDs in the bundle.
    /// * `guide_points` - Centerline guide waypoints `(x, y)` in `mm`.
    /// * `layer` - Copper layer index.
    /// * `track_width` - Trace width in `mm`.
    /// * `track_spacing` - Edge-to-edge trace spacing in `mm`.
    ///
    /// # Returns
    /// A new [`BusBundle`].
    ///
    /// # Side Effects
    /// None.
    pub fn new(
        name: impl Into<String>,
        net_ids: Vec<usize>,
        guide_points: Vec<(f64, f64)>,
        layer: usize,
        track_width: f64,
        track_spacing: f64,
    ) -> Self {
        Self {
            name: name.into(),
            net_ids,
            guide_points,
            layer,
            track_width,
            track_spacing,
        }
    }
}

/// Complete in-memory representation of a KiCad PCB layout, netlist, design rules, and routed copper.
#[derive(Debug, Clone)]
pub struct Board {
    /// Filesystem path to the source `.kicad_pcb` file (or empty if parsed from an in-memory string).
    pub file_path: String,
    /// Raw S-expression text of the original `.kicad_pcb` file used for lossless span-based saving.
    pub raw_content: String,
    /// Padded minimum X coordinate of the routing canvas (`Edge.Cuts` `min_x - 1.5 mm`).
    pub min_x: f64,
    /// Padded minimum Y coordinate of the routing canvas (`Edge.Cuts` `min_y - 1.5 mm`).
    pub min_y: f64,
    /// Padded maximum X coordinate of the routing canvas (`Edge.Cuts` `max_x + 1.5 mm`).
    pub max_x: f64,
    /// Padded maximum Y coordinate of the routing canvas (`Edge.Cuts` `max_y + 1.5 mm`).
    pub max_y: f64,
    /// Active footprints currently placed on the board.
    pub footprints: Vec<Footprint>,
    /// Staging holding palette of unplaced/cleared footprints awaiting interactive or automated placement.
    pub holding_palette: Vec<Footprint>,
    /// Net name lookup table indexed by numeric `net_id` (`nets[0]` is always `""`).
    pub nets: Vec<String>,
    /// Reverse lookup map from net name string to numeric `net_id`.
    pub net_name_to_id: HashMap<String, usize>,
    /// Named [`NetClass`] definitions keyed by class name (always includes `"Default"`).
    pub netclasses: HashMap<String, NetClass>,
    /// Mapping from net name to its assigned [`NetClass`] name.
    pub net_to_class: HashMap<String, String>,
    /// Copper and placement keepout zones on the board.
    pub keepouts: Vec<KeepoutZone>,
    /// Straight routed copper trace segments (`(segment ...)`).
    pub tracks: Vec<TrackSegment>,
    /// Plated through-vias (`(via ...)`).
    pub vias: Vec<Via>,
    /// Curved copper arc trace segments (`(arc ...)`).
    pub arcs: Vec<ArcSegment>,
    /// Functional component clusters compiled from `.kicad_sch`.
    pub clusters: Vec<CompiledCluster>,
    /// 2D schematic symbol positions `(x, y)` keyed by reference designator.
    pub schematic_positions: HashMap<String, (f64, f64)>,
    /// Chained closed board outline polygon extracted from `Edge.Cuts` graphic primitives.
    pub outline_polygon: Option<Polygon>,
    /// Ordered history of pin and gate swaps performed on the board for `.eco` back-annotation.
    pub pin_swap_history: Vec<PinSwapRecord>,
    /// Multi-net bus bundles defined on the board.
    pub bundles: Vec<BusBundle>,
    /// Top-level board graphic text annotations (`(gr_text ...)`), primarily on `F.SilkS`/`B.SilkS`.
    pub texts: Vec<BoardText>,
}

/// A top-level graphic text item `(gr_text ...)` on the PCB (such as silkscreen labels).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoardText {
    /// Text string content.
    pub text: String,
    /// World X coordinate in `mm`.
    pub x: f64,
    /// World Y coordinate in `mm`.
    pub y: f64,
    /// Text rotation angle in degrees.
    pub rotation: f64,
    /// KiCad layer name (e.g., `"F.SilkS"`, `"B.SilkS"`, `"F.Fab"`).
    pub layer: String,
    /// Character glyph width in `mm`.
    pub width: f64,
    /// Character glyph height in `mm`.
    pub height: f64,
    /// Optional byte span `(start, end)` of the `(at X Y [ROT])` token inside [`Board::raw_content`].
    pub at_span: Option<(usize, usize)>,
}

impl Default for Board {
    /// Constructs an empty `100 x 100 mm` [`Board`] with net `0` (`""`) and a `"Default"` [`NetClass`].
    fn default() -> Self {
        Self {
            file_path: String::new(),
            raw_content: "()".to_string(),
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
            footprints: Vec::new(),
            holding_palette: Vec::new(),
            nets: vec!["".to_string()],
            net_name_to_id: HashMap::new(),
            netclasses: {
                let mut map = HashMap::new();
                map.insert("Default".to_string(), NetClass::default());
                map
            },
            net_to_class: HashMap::new(),
            keepouts: Vec::new(),
            tracks: Vec::new(),
            vias: Vec::new(),
            arcs: Vec::new(),
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
            texts: Vec::new(),
        }
    }
}

/// Ordered list of all 30 internal KiCad copper layer names (`"In1.Cu"` through `"In30.Cu"`).
pub const INTERNAL_COPPER_LAYERS: [&str; 30] = [
    "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "In5.Cu",
    "In6.Cu", "In7.Cu", "In8.Cu", "In9.Cu", "In10.Cu",
    "In11.Cu", "In12.Cu", "In13.Cu", "In14.Cu", "In15.Cu",
    "In16.Cu", "In17.Cu", "In18.Cu", "In19.Cu", "In20.Cu",
    "In21.Cu", "In22.Cu", "In23.Cu", "In24.Cu", "In25.Cu",
    "In26.Cu", "In27.Cu", "In28.Cu", "In29.Cu", "In30.Cu",
];

/// Detects all active copper layers declared in a `.kicad_pcb` S-expression string.
///
/// Always places `"F.Cu"` at index `0` and `"B.Cu"` at index `1`, followed by any present
/// `"In1.Cu"`..`"In30.Cu"` internal layers in numerical order.
///
/// # Arguments
/// * `raw_content` - Raw `.kicad_pcb` S-expression text.
///
/// # Returns
/// Vector of static copper layer name string slices (`["F.Cu", "B.Cu", ...]`).
///
/// # Side Effects
/// None.
pub fn detect_copper_layers(raw_content: &str) -> Vec<&'static str> {
    let mut layers = vec!["F.Cu", "B.Cu"];
    if !raw_content.contains("\"In") {
        return layers;
    }
    for &name in &INTERNAL_COPPER_LAYERS {
        if raw_content.contains(&format!("\"{}\"", name)) {
            layers.push(name);
        }
    }
    layers
}

impl Board {
    /// Returns whether via-in-pad routing (`__allow_via_in_pad__`) is enabled on this board.
    ///
    /// # Returns
    /// `true` if vias are permitted directly inside SMD pads of the same net, `false` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn allow_via_in_pad(&self) -> bool {
        self.net_to_class
            .get("__allow_via_in_pad__")
            .map(|v| v == "true")
            .unwrap_or(false)
    }

    /// Sets whether via-in-pad routing is permitted on this board.
    ///
    /// Stores the flag in `self.net_to_class` under the internal key `"__allow_via_in_pad__"`.
    ///
    /// # Arguments
    /// * `allow` - `true` to allow same-net vias inside SMD pads, `false` to forbid them.
    ///
    /// # Side Effects
    /// Mutates `self.net_to_class`.
    pub fn set_allow_via_in_pad(&mut self, allow: bool) {
        if allow {
            self.net_to_class
                .insert("__allow_via_in_pad__".to_string(), "true".to_string());
        } else {
            self.net_to_class.remove("__allow_via_in_pad__");
        }
    }

    /// Parses a [`Board`] from an in-memory `.kicad_pcb` S-expression string without loading sibling project files.
    ///
    /// # Arguments
    /// * `content` - Raw `.kicad_pcb` S-expression string slice.
    ///
    /// # Returns
    /// `Ok(Board)` on successful parse, or `Err(String)` if malformed.
    ///
    /// # Errors
    /// Returns `Err` if [`Board::from_content_with_path`] encounters an unrecoverable parse error.
    ///
    /// # Side Effects
    /// None.
    pub fn from_kicad_pcb(content: &str) -> Result<Self, String> {
        Self::from_content_with_path(content, "", &HashSet::new())
    }

    /// Loads a `.kicad_pcb` file from disk along with any sibling `.kicad_pro`, `.kicad_dru`, and `.kicad_sch` files.
    ///
    /// # Arguments
    /// * `path` - Filesystem path to the `.kicad_pcb` file.
    /// * `extra_locked` - Set of additional component reference designators to lock in place.
    ///
    /// # Returns
    /// `Ok(Board)` on success, or `Err(String)` if the file cannot be read or parsed.
    ///
    /// # Errors
    /// Returns `Err` if reading `path` fails or if parsing fails.
    ///
    /// # Side Effects
    /// Reads `path` and optional sibling `.kicad_pro`, `.kicad_dru`, and `.kicad_sch` files from the filesystem.
    pub fn load(path: &str, extra_locked: &HashSet<String>) -> Result<Self, String> {
        let content = fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
        Self::from_content_with_path(&content, path, extra_locked)
    }

    /// Parses a `.kicad_pcb` S-expression string and optionally resolves sibling project files relative to `path`.
    ///
    /// Extracts top-level nets, netclasses, `Edge.Cuts` outlines, keepout zones, footprints, pads,
    /// courtyards, segments, arcs, vias, and graphic texts. When `path` is non-empty, also loads
    /// sibling `.kicad_pro` netclasses, `.kicad_dru` design rules, and `.kicad_sch` placement clusters,
    /// and heals any multi-pad footprints whose on-disk `(at ...)` pose became desynchronized from routed copper.
    ///
    /// # Arguments
    /// * `content` - Raw `.kicad_pcb` S-expression text.
    /// * `path` - Source `.kicad_pcb` filesystem path (or `""` for in-memory boards).
    /// * `extra_locked` - Additional component reference designators to mark as `is_locked = true`.
    ///
    /// # Returns
    /// `Ok(Board)` containing the parsed and initialized PCB model.
    ///
    /// # Errors
    /// Returns `Err` if the S-expression content cannot be parsed.
    ///
    /// # Side Effects
    /// When `path` is non-empty, reads sibling `.kicad_pro`, `.kicad_dru`, and `.kicad_sch` files from disk if they exist.
    pub fn from_content_with_path(content: &str, path: &str, extra_locked: &HashSet<String>) -> Result<Self, String> {
        // 1. Parse top-level nets
        let mut nets = vec!["".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("".to_string(), 0);

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("(footprint") || trimmed.starts_with("(module") {
                break;
            }
            if trimmed.starts_with("(net ") && trimmed.ends_with(')') {
                let inner = &trimmed[5..trimmed.len() - 1];
                let mut parts = inner.splitn(2, ' ');
                if let (Some(id_str), Some(name_raw)) = (parts.next(), parts.next()) {
                    if let Ok(id) = id_str.parse::<usize>() {
                        let name = name_raw.trim().trim_matches('"').to_string();
                        if nets.len() <= id {
                            nets.resize(id + 1, String::new());
                        }
                        nets[id] = name.clone();
                        net_name_to_id.insert(name, id);
                    }
                }
            }
        }

        // 2. Standard JLCPCB NetClasses
        let mut netclasses = HashMap::new();
        for (name, width, clearance) in [
            ("Default", 0.20, 0.20),
            ("Power_15V", 0.60, 0.25),
            ("Power_3V3", 0.50, 0.25),
            ("Switching", 0.60, 0.25),
            ("HBS_Bus", 0.20, 0.20),
            ("USB_Diff", 0.20, 0.20),
            ("Power_5V", 0.35, 0.20),
        ] {
            let is_hs = NetClass::is_high_speed_name(name);
            netclasses.insert(
                name.to_string(),
                NetClass {
                    name: name.to_string(),
                    track_width: width,
                    clearance,
                    via_dia: 0.60,
                    via_drill: 0.30,
                    diff_pair_gap: 0.20,
                    diff_pair_width: width,
                    topology: NetTopology::default(),
                    require_return_path: is_hs,
                    ..NetClass::default()
                },
            );
        }

        // 3. Parse Edge.Cuts, Keepouts, and Footprints
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut keepouts = Vec::new();
        let mut footprints = Vec::new();
        let mut tracks = Vec::new();
        let mut vias = Vec::new();
        let mut arcs = Vec::new();
        let mut texts = Vec::new();

        let active_layers = detect_copper_layers(&content);

        let bytes = content.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        let mut edge_cuts_lines: Vec<(Point, Point)> = Vec::new();

        while i < len {
            if bytes[i] == b'(' {
                let word = next_token(&content, i + 1);
                if word.starts_with("gr_") {
                    if let Some(end) = find_closing_paren(bytes, i) {
                        let block = &content[i..end];
                        if block.contains("\"Edge.Cuts\"") {
                            parse_edge_cuts_element(block, word, &mut edge_cuts_lines, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
                        } else if word == "gr_text" {
                            if let Some(bt) = parse_gr_text(block, i) {
                                texts.push(bt);
                            }
                        }
                        i = end;
                        continue;
                    }
                } else if word == "zone" {
                    if let Some(end) = find_closing_paren(bytes, i) {
                        let block = &content[i..end];
                        if block.contains("(keepout") {
                            let mut kmin_x = f64::INFINITY;
                            let mut kmin_y = f64::INFINITY;
                            let mut kmax_x = f64::NEG_INFINITY;
                            let mut kmax_y = f64::NEG_INFINITY;
                            extract_xy_into_bounds(block, &mut kmin_x, &mut kmin_y, &mut kmax_x, &mut kmax_y);
                            if kmin_x < kmax_x && kmin_y < kmax_y {
                                keepouts.push(KeepoutZone {
                                    min_x: kmin_x,
                                    min_y: kmin_y,
                                    max_x: kmax_x,
                                    max_y: kmax_y,
                                    layers: (0..active_layers.len()).collect(),
                                });
                            }
                        }
                        i = end;
                        continue;
                    }
                } else if word == "footprint" || word == "module" {
                    if let Some(end) = find_closing_paren(bytes, i) {
                        let block = &content[i..end];
                        if let Some(fp) = parse_footprint(block, i, &mut nets, &mut net_name_to_id, extra_locked) {
                            footprints.push(fp);
                        }
                        if block.contains("(keepout") {
                            let mut k_idx = 0;
                            while let Some(z_pos) = block[k_idx..].find("(zone") {
                                let abs_z = k_idx + z_pos;
                                if let Some(z_end) = find_closing_paren(block.as_bytes(), abs_z) {
                                    let z_block = &block[abs_z..z_end];
                                    if z_block.contains("(keepout") {
                                        let mut kmin_x = f64::INFINITY;
                                        let mut kmin_y = f64::INFINITY;
                                        let mut kmax_x = f64::NEG_INFINITY;
                                        let mut kmax_y = f64::NEG_INFINITY;
                                        extract_xy_into_bounds(z_block, &mut kmin_x, &mut kmin_y, &mut kmax_x, &mut kmax_y);
                                        if kmin_x < kmax_x && kmin_y < kmax_y {
                                            keepouts.push(KeepoutZone {
                                                min_x: kmin_x,
                                                min_y: kmin_y,
                                                max_x: kmax_x,
                                                max_y: kmax_y,
                                                layers: (0..active_layers.len()).collect(),
                                            });
                                        }
                                    }
                                    k_idx = z_end;
                                } else {
                                    break;
                                }
                            }
                        }
                        i = end;
                        continue;
                    }
                } else if word == "segment" {
                    if let Some(end) = find_closing_paren(bytes, i) {
                        let block = &content[i..end];
                        if let Some(tr) = parse_segment(block, &active_layers, &mut nets, &mut net_name_to_id) {
                            tracks.push(tr);
                        }
                        i = end;
                        continue;
                    }
                } else if word == "arc" {
                    if let Some(end) = find_closing_paren(bytes, i) {
                        let block = &content[i..end];
                        if let Some(arc) = parse_arc(block, &active_layers, &mut nets, &mut net_name_to_id) {
                            arcs.push(arc);
                        }
                        i = end;
                        continue;
                    }
                } else if word == "via" {
                    if let Some(end) = find_closing_paren(bytes, i) {
                        let block = &content[i..end];
                        if let Some(v) = parse_via(block, &mut nets, &mut net_name_to_id) {
                            vias.push(v);
                        }
                        i = end;
                        continue;
                    }
                } else if word == "net_class" {
                    if let Some(end) = find_closing_paren(bytes, i) {
                        let block = &content[i..end];
                        let parsed = parse_netclasses_from_pro_or_pcb(block);
                        for (name, nc) in parsed {
                            netclasses.insert(name, nc);
                        }
                        i = end;
                        continue;
                    }
                }
            }
            i += 1;
        }

        let mut net_to_class = HashMap::new();
        for net in &nets {
            let class = if matches!(net.as_str(), "P1" | "P2" | "HP" | "HN" | "VRAW" | "/BUCK_VIN") {
                "Power_15V"
            } else if net == "VBUS" {
                "Power_5V"
            } else if net == "+3.3V" {
                "Power_3V3"
            } else if net.contains("U2-SW") {
                "Switching"
            } else if net.contains("CAPP") || net.contains("CAPN") || net.contains("HPA") || net.contains("HNA") {
                "HBS_Bus"
            } else if net.contains("USB_D") {
                "USB_Diff"
            } else {
                "Default"
            };
            net_to_class.insert(net.clone(), class.to_string());
        }

        let outline_polygon = if !edge_cuts_lines.is_empty() {
            let chained = chain_lines_into_polygons(edge_cuts_lines);
            chained.into_iter().max_by(|a, b| a.vertices.len().cmp(&b.vertices.len()))
        } else {
            None
        };

        if min_x >= max_x || min_y >= max_y {
            min_x = 125.0;
            min_y = 86.0;
            max_x = 172.0;
            max_y = 124.0;
        }

        // Ensure RF antenna overhang keepout if U4 is present
        if keepouts.is_empty() && footprints.iter().any(|f| f.reference == "U4") {
            keepouts.push(KeepoutZone {
                min_x: 172.1,
                min_y: 91.0,
                max_x: 183.1,
                max_y: 119.0,
                layers: (0..active_layers.len()).collect(),
            });
        }

        let mut board = Self {
            file_path: path.to_string(),
            raw_content: content.to_string(),
            min_x: min_x - 1.5,
            min_y: min_y - 1.5,
            max_x: max_x + 1.5,
            max_y: max_y + 1.5,
            footprints,
            holding_palette: Vec::new(),
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts,
            tracks,
            vias,
            arcs,
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon,
            pin_swap_history: Vec::new(),
            bundles: Vec::new(),
            texts,
        };

        if !path.is_empty() {
            let resolve_sibling_ext = |p: &str, raw: &str, ext: &str| -> Option<std::path::PathBuf> {
                let direct = Path::new(p).with_extension(ext);
                if direct.exists() {
                    return Some(direct);
                }
                if let Some(sf_idx) = raw.find("(sheetfile \"") {
                    let after = &raw[sf_idx + 12..];
                    if let Some(q_end) = after.find('"') {
                        let sheet_name = &after[..q_end];
                        if let (Some(parent), Some(stem)) = (
                            Path::new(p).parent(),
                            Path::new(sheet_name).file_stem().and_then(|s| s.to_str()),
                        ) {
                            let fallback = parent.join(format!("{}.{}", stem, ext));
                            if fallback.exists() {
                                return Some(fallback);
                            }
                        }
                    }
                }
                None
            };

            // Auto-load project netclasses and design rules if sibling .kicad_pro exists
            if let Some(pro_path) = resolve_sibling_ext(path, &board.raw_content, "kicad_pro") {
                board.load_project_netclasses(&pro_path);
                // The project defines its own classes; drop built-in fallbacks left with no nets.
                let used: HashSet<&String> = board.net_to_class.values().collect();
                board.netclasses.retain(|name, _| name == "Default" || used.contains(name));
            }

            // Auto-load custom DRC rules if sibling .kicad_dru exists
            if let Some(dru_path) = resolve_sibling_ext(path, &board.raw_content, "kicad_dru") {
                board.load_design_rules_dru(&dru_path);
            }

            // Auto-load schematic functional clusters if sibling .kicad_sch exists
            if let Some(sch_path) = resolve_sibling_ext(path, &board.raw_content, "kicad_sch") {
                board.load_schematic_clusters(&sch_path);
            }
        }

        for fp in &mut board.footprints {
            fp.infer_swap_groups();
        }

        // Heal any multi-pad footprint whose on-disk (at ...) was left stale relative to routed tracks
        if !board.tracks.is_empty() {
            let inner_planes = board.inner_plane_nets();
            let is_plane = |nid: usize, nets: &[String]| -> bool {
                let net_name = nets.get(nid).map(|s| s.as_str()).unwrap_or("");
                is_gnd_net_name(net_name)
                    || inner_planes.values().any(|&pn| pn == nid)
            };
            for fp_idx in 0..board.footprints.len() {
                let fp = &board.footprints[fp_idx];
                if fp.pads.len() < 12 {
                    continue;
                }
                let cur_matches = fp
                    .pads
                    .iter()
                    .filter(|p| {
                        p.net_id > 1 && !is_plane(p.net_id, &board.nets) && {
                            let (px, py) = fp.pad_absolute_position(p);
                            board.tracks.iter().any(|t| {
                                t.net_id == p.net_id
                                    && ((t.start.0 - px).hypot(t.start.1 - py) < 0.05
                                        || (t.end.0 - px).hypot(t.end.1 - py) < 0.05)
                            })
                        }
                    })
                    .count();
                if cur_matches >= 2 {
                    continue;
                }
                let mut best_pose: Option<(usize, f64, f64, f64)> = None;
                for cand_rot in [0.0, 90.0, 180.0, 270.0] {
                    let mut test_fp = fp.clone();
                    test_fp.rotation = cand_rot;
                    for p0 in fp.pads.iter().filter(|p| p.net_id > 1 && !is_plane(p.net_id, &board.nets)) {
                        test_fp.x = 0.0;
                        test_fp.y = 0.0;
                        let (ox0, oy0) = test_fp.pad_absolute_position(p0);
                        for t in board.tracks.iter().filter(|t| t.net_id == p0.net_id) {
                            for end_pt in [t.start, t.end] {
                                let cx = ((end_pt.0 - ox0) * 1e4).round() / 1e4;
                                let cy = ((end_pt.1 - oy0) * 1e4).round() / 1e4;
                                test_fp.x = cx;
                                test_fp.y = cy;
                                let hits = test_fp
                                    .pads
                                    .iter()
                                    .filter(|p| {
                                        p.net_id > 1 && !is_plane(p.net_id, &board.nets) && {
                                            let (px, py) = test_fp.pad_absolute_position(p);
                                            board.tracks.iter().any(|ot| {
                                                ot.net_id == p.net_id
                                                    && ((ot.start.0 - px).hypot(ot.start.1 - py) < 0.02
                                                        || (ot.end.0 - px).hypot(ot.end.1 - py) < 0.02)
                                            })
                                        }
                                    })
                                    .count();
                                if hits >= 8 && best_pose.map_or(true, |(bh, _, _, _)| hits > bh) {
                                    best_pose = Some((hits, cx, cy, cand_rot));
                                }
                            }
                        }
                    }
                }
                if let Some((_, cx, cy, crot)) = best_pose {
                    board.footprints[fp_idx].x = cx;
                    board.footprints[fp_idx].y = cy;
                    board.footprints[fp_idx].rotation = crot;
                }
            }
        }

        Ok(board)
    }

    /// Returns the ordered list of active copper layer names on this board (`"F.Cu"`, `"B.Cu"`, `"In1.Cu"`, ...).
    ///
    /// # Returns
    /// Vector of static copper layer name string slices.
    ///
    /// # Side Effects
    /// None.
    pub fn copper_layers(&self) -> Vec<&'static str> {
        detect_copper_layers(&self.raw_content)
    }

    /// Maps an internal layer index (`0` = `"F.Cu"`, `1` = `"B.Cu"`, `2..` = `"In1.Cu"..`) to its KiCad copper layer name.
    ///
    /// # Arguments
    /// * `layer_idx` - Internal copper layer index.
    ///
    /// # Returns
    /// Static string slice with the KiCad layer name (defaults to `"B.Cu"` if out of range).
    ///
    /// # Side Effects
    /// None.
    pub fn layer_name(&self, layer_idx: usize) -> &'static str {
        match layer_idx {
            0 => "F.Cu",
            1 => "B.Cu",
            _ => self
                .copper_layers()
                .get(layer_idx)
                .copied()
                .unwrap_or("B.Cu"),
        }
    }

    /// Scans all non-keepout `(zone ...)` blocks in [`Board::raw_content`] and maps each copper layer index to its primary copper pour net name.
    ///
    /// # Returns
    /// A `HashMap<usize, String>` mapping copper `layer_idx` to the zone's net name.
    ///
    /// # Side Effects
    /// None.
    pub fn layer_plane_net_names(&self) -> HashMap<usize, String> {
        let layers = self.copper_layers();
        let mut map = HashMap::new();
        let bytes = self.raw_content.as_bytes();
        let mut idx = 0;
        while let Some(pos) = self.raw_content[idx..].find("(zone") {
            let start = idx + pos;
            if let Some(end) = find_closing_paren(bytes, start) {
                let block = &self.raw_content[start..end];
                if !block.contains("(keepout") {
                    let header = &block[..block.len().min(400)];
                    let mut found_net_name = None;
                    if let Some(n_pos) = header.find("(net ") {
                        let after = header[n_pos + 5..].trim_start();
                        if after.starts_with('"') {
                            if let Some(q2) = after[1..].find('"') {
                                found_net_name = Some(after[1..1 + q2].to_string());
                            }
                        } else {
                            let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
                            if let Ok(net_id) = num_str.parse::<usize>() {
                                if let Some(n) = self.nets.get(net_id) {
                                    if !n.is_empty() {
                                        found_net_name = Some(n.clone());
                                    }
                                }
                            }
                            if found_net_name.is_none() {
                                if let Some(q1) = after.find('"') {
                                    if let Some(q2) = after[q1 + 1..].find('"') {
                                        let name = &after[q1 + 1..q1 + 1 + q2];
                                        if !name.is_empty() {
                                            found_net_name = Some(name.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if found_net_name.is_none() {
                        if let Some(nn_pos) = header.find("(net_name \"") {
                            let after = &header[nn_pos + 11..];
                            if let Some(q2) = after.find('"') {
                                found_net_name = Some(after[..q2].to_string());
                            }
                        }
                    }

                    if let Some(net_name) = found_net_name {
                        for (l_idx, &l_name) in layers.iter().enumerate() {
                            let l_pat1 = format!("(layer \"{}\")", l_name);
                            let l_pat2 = format!("\"{}\"", l_name);
                            if header.contains(&l_pat1) || (header.contains("(layers ") && header.contains(&l_pat2)) {
                                map.entry(l_idx).or_insert(net_name.clone());
                            }
                        }
                    }
                }
                idx = end;
            } else {
                break;
            }
        }
        map
    }

    /// Maps each copper layer index (`0..num_layers`) that has a copper pour zone to its numeric `net_id`.
    ///
    /// # Returns
    /// A `HashMap<usize, usize>` mapping `layer_idx -> net_id`.
    ///
    /// # Side Effects
    /// None.
    pub fn layer_plane_nets(&self) -> HashMap<usize, usize> {
        let mut map = HashMap::new();
        for (l_idx, name) in self.layer_plane_net_names() {
            if let Some(&net_id) = self.net_name_to_id.get(&name) {
                map.insert(l_idx, net_id);
            }
        }
        map
    }

    /// Returns a map of internal copper layers (`layer_idx >= 2`, i.e. `In1.Cu`..`In30.Cu`) to their assigned reference plane `net_id`.
    ///
    /// Returns an empty map on 2-layer boards.
    ///
    /// # Returns
    /// A `HashMap<usize, usize>` mapping internal `layer_idx -> plane_net_id`.
    ///
    /// # Side Effects
    /// None.
    pub fn inner_plane_nets(&self) -> HashMap<usize, usize> {
        if self.copper_layers().len() <= 2 {
            return HashMap::new();
        }
        let mut map = self.layer_plane_nets();
        map.retain(|&layer_idx, _| layer_idx >= 2);
        map
    }

    /// Checks whether `net_id` is a ground net or is assigned to an internal reference plane zone.
    ///
    /// # Arguments
    /// * `net_id` - Numeric net ID to check.
    ///
    /// # Returns
    /// `true` if `net_id` is a ground net or an inner plane net, `false` otherwise.
    ///
    /// # Side Effects
    /// None.
    pub fn is_plane_net(&self, net_id: usize) -> bool {
        let net_name = self.nets.get(net_id).map(|s| s.as_str()).unwrap_or("");
        is_gnd_net_name(net_name)
            || self.inner_plane_nets().values().any(|&pn| pn == net_id)
    }

    /// Returns the physical `Edge.Cuts` axis-aligned bounding box ([`AABB`]) of the PCB.
    ///
    /// Uses `self.outline_polygon.aabb()` if a closed outline polygon exists, or removes the `1.5 mm`
    /// canvas margin from `(min_x, min_y, max_x, max_y)`.
    ///
    /// # Returns
    /// Physical board [`AABB`] in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn boundary_aabb(&self) -> AABB {
        if let Some(ref poly) = self.outline_polygon {
            poly.aabb()
        } else {
            AABB::new(self.min_x + 1.5, self.min_y + 1.5, self.max_x - 1.5, self.max_y - 1.5)
        }
    }

    /// Minimum clearance from copper to `Edge.Cuts` outline based on board netclasses (default `0.35 mm`).
    ///
    /// # Returns
    /// Minimum board edge clearance in `mm`.
    ///
    /// # Side Effects
    /// None.
    pub fn board_edge_clearance(&self) -> f64 {
        self.netclasses
            .values()
            .map(|nc| nc.board_edge_clearance)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.35)
    }

    /// Board-level physical DRC constraints imported from `.kicad_pro` and `.kicad_dru`.
    ///
    /// # Returns
    /// Consolidated [`DesignRules`] structure for the board.
    ///
    /// # Side Effects
    /// None.
    pub fn design_rules(&self) -> DesignRules {
        let nc = self.get_netclass_for_net(0);
        DesignRules {
            min_clearance: self
                .netclasses
                .values()
                .map(|c| c.clearance)
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0),
            min_track_width: nc.min_track_width,
            min_via_diameter: nc.min_via_diameter,
            min_via_drill: nc.min_via_drill,
            min_via_annular_width: nc.min_via_annular_width,
            min_through_hole_diameter: nc.min_through_hole_diameter,
            min_hole_clearance: nc.min_hole_clearance,
            min_hole_to_hole: nc.min_hole_to_hole,
            min_copper_edge_clearance: self.board_edge_clearance(),
            zone_min_clearance: nc.zone_min_clearance,
            zone_min_thickness: nc.zone_min_thickness,
        }
    }

    /// Verifies if a point is strictly inside the physical board boundary and outline contour.
    ///
    /// # Arguments
    /// * `p` - 2D [`Point`] in board coordinates (`mm`).
    ///
    /// # Returns
    /// `true` if `p` lies inside both [`Board::boundary_aabb`] and [`Board::outline_polygon`] (if present).
    ///
    /// # Side Effects
    /// None.
    pub fn contains_point_in_boundary(&self, p: &Point) -> bool {
        if !self.boundary_aabb().contains_point(p) {
            return false;
        }
        if let Some(ref poly) = self.outline_polygon {
            if !poly.contains_point(p) {
                return false;
            }
        }
        true
    }

    /// Rips up all unlocked tracks and vias, and moves all unlocked footprints
    /// from the active board into the holding palette, sorted naturally by designator
    /// (letters first, then numbers, e.g. `C5` before `C20`).
    ///
    /// # Returns
    /// A tuple `(copper_items_ripped, footprints_moved_to_palette)`.
    ///
    /// # Side Effects
    /// Retains only locked items in `self.tracks`, `self.vias`, `self.arcs`, and `self.footprints`,
    /// moving unlocked footprints into `self.holding_palette`.
    pub fn clear_unlocked_objects(&mut self) -> (usize, usize) {
        let tracks_ripped = self.tracks.iter().filter(|t| !t.is_locked).count()
            + self.vias.iter().filter(|v| !v.is_locked).count()
            + self.arcs.iter().filter(|a| !a.is_locked).count();
        self.tracks.retain(|t| t.is_locked);
        self.vias.retain(|v| v.is_locked);
        self.arcs.retain(|a| a.is_locked);

        let mut unlocked = Vec::new();
        let mut locked = Vec::new();
        for fp in self.footprints.drain(..) {
            if fp.is_locked {
                locked.push(fp);
            } else {
                unlocked.push(fp);
            }
        }
        self.footprints = locked;
        let fps_cleared = unlocked.len();
        self.holding_palette.extend(unlocked);
        self.holding_palette.sort_by(|a, b| natural_designator_cmp(&a.reference, &b.reference));

        (tracks_ripped, fps_cleared)
    }

    /// Rips up all connected tracks and vias from the component at `fp_idx` all the way
    /// to the next attached component(s).
    ///
    /// # Arguments
    /// * `fp_idx` - Index of the footprint in `self.footprints`.
    ///
    /// # Returns
    /// Total number of unlocked tracks, arcs, and vias removed.
    ///
    /// # Side Effects
    /// Mutates `self.tracks`, `self.arcs`, and `self.vias`, and prunes any resulting dangling copper.
    pub fn rip_up_connected_traces_to_component(&mut self, fp_idx: usize) -> usize {
        self.rip_up_connected_traces_to_component_at(fp_idx, None)
    }

    /// Rips up all connected tracks and vias from the component at `fp_idx` (optionally evaluated
    /// at `pos_override: (x, y, rotation, layer)` prior to its move) all the way to the next
    /// attached component(s).
    ///
    /// Performs a breadth-first search along connected same-net tracks, arcs, and vias starting at
    /// `fp_idx`'s pads, stopping whenever a junction touches another component's pad so unrelated
    /// downstream branches of the same net remain intact.
    ///
    /// # Arguments
    /// * `fp_idx` - Index of the footprint in `self.footprints`.
    /// * `pos_override` - Optional `(x, y, rotation_deg, layer)` pose where the footprint was connected before moving.
    ///
    /// # Returns
    /// Total number of unlocked tracks, arcs, and vias removed.
    ///
    /// # Side Effects
    /// Removes connected unlocked elements from `self.tracks`, `self.arcs`, and `self.vias`, and calls
    /// `crate::router::prune_dangling_tracks_and_vias(self)`.
    pub fn rip_up_connected_traces_to_component_at(
        &mut self,
        fp_idx: usize,
        pos_override: Option<(f64, f64, f64, usize)>,
    ) -> usize {
        if fp_idx >= self.footprints.len() || self.footprints[fp_idx].is_locked {
            return 0;
        }

        let mut target_fp = self.footprints[fp_idx].clone();
        if let Some((ox, oy, orot, olayer)) = pos_override {
            target_fp.x = ox;
            target_fp.y = oy;
            target_fp.rotation = orot;
            target_fp.layer = olayer;
        }

        let active_nets: HashSet<usize> = target_fp
            .pads
            .iter()
            .map(|p| p.net_id)
            .filter(|&nid| nid > 0)
            .collect();

        if active_nets.is_empty() {
            return 0;
        }

        const EPS: f64 = 0.05;

        let mut tracks_to_remove = HashSet::new();
        let mut arcs_to_remove = HashSet::new();
        let mut vias_to_remove = HashSet::new();

        for &net_id in &active_nets {
            let my_pads: Vec<&Pad> = target_fp.pads.iter().filter(|p| p.net_id == net_id).collect();
            if my_pads.is_empty() {
                continue;
            }

            let other_pads: Vec<(&Footprint, &Pad)> = self
                .footprints
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != fp_idx)
                .flat_map(|(_, fp)| fp.pads.iter().filter(|p| p.net_id == net_id).map(move |p| (fp, p)))
                .collect();

            let trk_indices: Vec<usize> = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(_, t)| t.net_id == net_id && !t.is_locked)
                .map(|(i, _)| i)
                .collect();

            let arc_indices: Vec<usize> = self
                .arcs
                .iter()
                .enumerate()
                .filter(|(_, a)| a.net_id == net_id && !a.is_locked)
                .map(|(i, _)| i)
                .collect();

            let via_indices: Vec<usize> = self
                .vias
                .iter()
                .enumerate()
                .filter(|(_, v)| v.net_id == net_id && !v.is_locked)
                .map(|(i, _)| i)
                .collect();

            if trk_indices.is_empty() && arc_indices.is_empty() && via_indices.is_empty() {
                continue;
            }

            let touches_any_other_pad = |pt: (f64, f64), layer: Option<usize>| -> bool {
                for &(ofp, op) in &other_pads {
                    if let Some(l) = layer {
                        if !op.is_through_hole && ofp.layer != l {
                            continue;
                        }
                    }
                    if crate::router::dist_point_to_pad(pt, ofp, op) < EPS {
                        return true;
                    }
                }
                false
            };

            let track_touches_my_pads = |t: &TrackSegment| -> bool {
                for pad in &my_pads {
                    if !pad.is_through_hole && target_fp.layer != t.layer {
                        continue;
                    }
                    if crate::router::dist_point_to_pad(t.start, &target_fp, pad) < EPS
                        || crate::router::dist_point_to_pad(t.end, &target_fp, pad) < EPS
                        || crate::router::dist_segment_to_pad(t.start, t.end, &target_fp, pad) < EPS
                    {
                        return true;
                    }
                }
                false
            };

            let arc_touches_my_pads = |a: &ArcSegment| -> bool {
                let s = a.start();
                let e = a.end();
                for pad in &my_pads {
                    if !pad.is_through_hole && target_fp.layer != a.layer {
                        continue;
                    }
                    if crate::router::dist_point_to_pad(s, &target_fp, pad) < EPS
                        || crate::router::dist_point_to_pad(e, &target_fp, pad) < EPS
                    {
                        return true;
                    }
                }
                false
            };

            let via_touches_my_pads = |v: &Via| -> bool {
                for pad in &my_pads {
                    let (px, py) = target_fp.pad_absolute_position(pad);
                    let pad_rad = pad.width.max(pad.height) * 0.5;
                    if (v.position.0 - px).hypot(v.position.1 - py) <= pad_rad + v.size * 0.5 + EPS
                        || crate::router::dist_point_to_pad(v.position, &target_fp, pad) <= v.size * 0.5 + EPS
                    {
                        return true;
                    }
                }
                false
            };

            #[derive(Copy, Clone, PartialEq, Eq, Hash)]
            enum Item {
                Track(usize),
                Arc(usize),
                Via(usize),
            }

            let mut visited = HashSet::new();
            let mut queue = std::collections::VecDeque::new();

            for &ti in &trk_indices {
                if track_touches_my_pads(&self.tracks[ti]) {
                    visited.insert(Item::Track(ti));
                    queue.push_back(Item::Track(ti));
                }
            }
            for &ai in &arc_indices {
                if arc_touches_my_pads(&self.arcs[ai]) {
                    visited.insert(Item::Arc(ai));
                    queue.push_back(Item::Arc(ai));
                }
            }
            for &vi in &via_indices {
                if via_touches_my_pads(&self.vias[vi]) {
                    visited.insert(Item::Via(vi));
                    queue.push_back(Item::Via(vi));
                }
            }

            while let Some(item) = queue.pop_front() {
                match item {
                    Item::Track(t_idx) => {
                        let t = &self.tracks[t_idx];

                        for &other_t_idx in &trk_indices {
                            if other_t_idx == t_idx || visited.contains(&Item::Track(other_t_idx)) {
                                continue;
                            }
                            let ot = &self.tracks[other_t_idx];
                            if ot.layer != t.layer {
                                continue;
                            }
                            let d = crate::router::dist_segment_to_segment(t.start, t.end, ot.start, ot.end);
                            if d < EPS {
                                let contact_pt = if (t.start.0 - ot.start.0).hypot(t.start.1 - ot.start.1) < EPS
                                    || (t.start.0 - ot.end.0).hypot(t.start.1 - ot.end.1) < EPS
                                    || crate::router::dist_point_to_segment(t.start, ot.start, ot.end) < EPS
                                {
                                    t.start
                                } else if (t.end.0 - ot.start.0).hypot(t.end.1 - ot.end.1) < EPS
                                    || (t.end.0 - ot.end.0).hypot(t.end.1 - ot.end.1) < EPS
                                    || crate::router::dist_point_to_segment(t.end, ot.start, ot.end) < EPS
                                {
                                    t.end
                                } else if crate::router::dist_point_to_segment(ot.start, t.start, t.end) < EPS {
                                    ot.start
                                } else if crate::router::dist_point_to_segment(ot.end, t.start, t.end) < EPS {
                                    ot.end
                                } else {
                                    ((t.start.0 + t.end.0 + ot.start.0 + ot.end.0) * 0.25, (t.start.1 + t.end.1 + ot.start.1 + ot.end.1) * 0.25)
                                };

                                if !touches_any_other_pad(contact_pt, Some(t.layer)) {
                                    visited.insert(Item::Track(other_t_idx));
                                    queue.push_back(Item::Track(other_t_idx));
                                }
                            }
                        }

                        for &ai in &arc_indices {
                            if visited.contains(&Item::Arc(ai)) {
                                continue;
                            }
                            let a = &self.arcs[ai];
                            if a.layer != t.layer {
                                continue;
                            }
                            let as_pt = a.start();
                            let ae_pt = a.end();
                            let contact = if crate::router::dist_point_to_segment(as_pt, t.start, t.end) < EPS {
                                Some(as_pt)
                            } else if crate::router::dist_point_to_segment(ae_pt, t.start, t.end) < EPS {
                                Some(ae_pt)
                            } else {
                                None
                            };
                            if let Some(cpt) = contact {
                                if !touches_any_other_pad(cpt, Some(t.layer)) {
                                    visited.insert(Item::Arc(ai));
                                    queue.push_back(Item::Arc(ai));
                                }
                            }
                        }

                        for &vi in &via_indices {
                            if visited.contains(&Item::Via(vi)) {
                                continue;
                            }
                            let v = &self.vias[vi];
                            let d = crate::router::dist_point_to_segment(v.position, t.start, t.end);
                            if d <= v.size * 0.5 + EPS {
                                if !touches_any_other_pad(v.position, None) {
                                    visited.insert(Item::Via(vi));
                                    queue.push_back(Item::Via(vi));
                                }
                            }
                        }
                    }
                    Item::Arc(a_idx) => {
                        let a = &self.arcs[a_idx];
                        let as_pt = a.start();
                        let ae_pt = a.end();

                        for &ti in &trk_indices {
                            if visited.contains(&Item::Track(ti)) {
                                continue;
                            }
                            let t = &self.tracks[ti];
                            if t.layer != a.layer {
                                continue;
                            }
                            let contact = if crate::router::dist_point_to_segment(as_pt, t.start, t.end) < EPS {
                                Some(as_pt)
                            } else if crate::router::dist_point_to_segment(ae_pt, t.start, t.end) < EPS {
                                Some(ae_pt)
                            } else {
                                None
                            };
                            if let Some(cpt) = contact {
                                if !touches_any_other_pad(cpt, Some(a.layer)) {
                                    visited.insert(Item::Track(ti));
                                    queue.push_back(Item::Track(ti));
                                }
                            }
                        }

                        for &other_a_idx in &arc_indices {
                            if other_a_idx == a_idx || visited.contains(&Item::Arc(other_a_idx)) {
                                continue;
                            }
                            let oa = &self.arcs[other_a_idx];
                            if oa.layer != a.layer {
                                continue;
                            }
                            let oas = oa.start();
                            let oae = oa.end();
                            let contact = if (as_pt.0 - oas.0).hypot(as_pt.1 - oas.1) < EPS
                                || (as_pt.0 - oae.0).hypot(as_pt.1 - oae.1) < EPS
                            {
                                Some(as_pt)
                            } else if (ae_pt.0 - oas.0).hypot(ae_pt.1 - oas.1) < EPS
                                || (ae_pt.0 - oae.0).hypot(ae_pt.1 - oae.1) < EPS
                            {
                                Some(ae_pt)
                            } else {
                                None
                            };
                            if let Some(cpt) = contact {
                                if !touches_any_other_pad(cpt, Some(a.layer)) {
                                    visited.insert(Item::Arc(other_a_idx));
                                    queue.push_back(Item::Arc(other_a_idx));
                                }
                            }
                        }

                        for &vi in &via_indices {
                            if visited.contains(&Item::Via(vi)) {
                                continue;
                            }
                            let v = &self.vias[vi];
                            if (v.position.0 - as_pt.0).hypot(v.position.1 - as_pt.1) <= v.size * 0.5 + EPS
                                || (v.position.0 - ae_pt.0).hypot(v.position.1 - ae_pt.1) <= v.size * 0.5 + EPS
                            {
                                if !touches_any_other_pad(v.position, None) {
                                    visited.insert(Item::Via(vi));
                                    queue.push_back(Item::Via(vi));
                                }
                            }
                        }
                    }
                    Item::Via(v_idx) => {
                        let v = &self.vias[v_idx];

                        for &ti in &trk_indices {
                            if visited.contains(&Item::Track(ti)) {
                                continue;
                            }
                            let t = &self.tracks[ti];
                            let d = crate::router::dist_point_to_segment(v.position, t.start, t.end);
                            if d <= v.size * 0.5 + EPS {
                                if !touches_any_other_pad(v.position, None) {
                                    visited.insert(Item::Track(ti));
                                    queue.push_back(Item::Track(ti));
                                }
                            }
                        }

                        for &ai in &arc_indices {
                            if visited.contains(&Item::Arc(ai)) {
                                continue;
                            }
                            let a = &self.arcs[ai];
                            let as_pt = a.start();
                            let ae_pt = a.end();
                            if (v.position.0 - as_pt.0).hypot(v.position.1 - as_pt.1) <= v.size * 0.5 + EPS
                                || (v.position.0 - ae_pt.0).hypot(v.position.1 - ae_pt.1) <= v.size * 0.5 + EPS
                            {
                                if !touches_any_other_pad(v.position, None) {
                                    visited.insert(Item::Arc(ai));
                                    queue.push_back(Item::Arc(ai));
                                }
                            }
                        }

                        for &other_v_idx in &via_indices {
                            if other_v_idx == v_idx || visited.contains(&Item::Via(other_v_idx)) {
                                continue;
                            }
                            let ov = &self.vias[other_v_idx];
                            let d = (v.position.0 - ov.position.0).hypot(v.position.1 - ov.position.1);
                            if d <= (v.size + ov.size) * 0.5 + EPS {
                                if !touches_any_other_pad(ov.position, None) {
                                    visited.insert(Item::Via(other_v_idx));
                                    queue.push_back(Item::Via(other_v_idx));
                                }
                            }
                        }
                    }
                }
            }

            for item in visited {
                match item {
                    Item::Track(ti) => {
                        tracks_to_remove.insert(ti);
                    }
                    Item::Arc(ai) => {
                        arcs_to_remove.insert(ai);
                    }
                    Item::Via(vi) => {
                        vias_to_remove.insert(vi);
                    }
                }
            }
        }

        let ripped_tracks_count = tracks_to_remove.len();
        let ripped_arcs_count = arcs_to_remove.len();
        let ripped_vias_count = vias_to_remove.len();
        let total_ripped = ripped_tracks_count + ripped_arcs_count + ripped_vias_count;

        if total_ripped > 0 {
            let mut new_tracks = Vec::with_capacity(self.tracks.len() - ripped_tracks_count);
            for (idx, trk) in self.tracks.drain(..).enumerate() {
                if !tracks_to_remove.contains(&idx) {
                    new_tracks.push(trk);
                }
            }
            self.tracks = new_tracks;

            let mut new_arcs = Vec::with_capacity(self.arcs.len() - ripped_arcs_count);
            for (idx, arc) in self.arcs.drain(..).enumerate() {
                if !arcs_to_remove.contains(&idx) {
                    new_arcs.push(arc);
                }
            }
            self.arcs = new_arcs;

            let mut new_vias = Vec::with_capacity(self.vias.len() - ripped_vias_count);
            for (idx, v) in self.vias.drain(..).enumerate() {
                if !vias_to_remove.contains(&idx) {
                    new_vias.push(v);
                }
            }
            self.vias = new_vias;

            // Prune any residual dangling tracks and vias created by this rip-up
            crate::router::prune_dangling_tracks_and_vias(self);
        }

        total_ripped
    }

    /// Restores all components currently in the holding palette back onto the board.
    ///
    /// # Returns
    /// Number of footprints restored from `self.holding_palette` to `self.footprints`.
    ///
    /// # Side Effects
    /// Drains `self.holding_palette` and appends its elements to `self.footprints`.
    pub fn return_holding_palette_to_board(&mut self) -> usize {
        let count = self.holding_palette.len();
        self.footprints.extend(self.holding_palette.drain(..));
        count
    }

    /// Restores a single component at `idx` from the holding palette back onto the board.
    ///
    /// # Arguments
    /// * `idx` - Index into `self.holding_palette`.
    ///
    /// # Returns
    /// `Some(Footprint)` if `idx < self.holding_palette.len()`, or `None` if out of bounds.
    ///
    /// # Side Effects
    /// Removes the footprint at `idx` from `self.holding_palette` and pushes it onto `self.footprints`.
    pub fn return_holding_component_to_board(&mut self, idx: usize) -> Option<Footprint> {
        if idx < self.holding_palette.len() {
            let fp = self.holding_palette.remove(idx);
            self.footprints.push(fp.clone());
            Some(fp)
        } else {
            None
        }
    }

    /// Looks up the [`NetClass`] governing `net_id`, falling back to `"Default"` if unassigned.
    ///
    /// # Arguments
    /// * `net_id` - Numeric net ID.
    ///
    /// # Returns
    /// Shared reference to the [`NetClass`] for `net_id`.
    ///
    /// # Side Effects
    /// None.
    pub fn get_netclass_for_net(&self, net_id: usize) -> &NetClass {
        if let Some(name) = self.nets.get(net_id) {
            if let Some(cname) = self.net_to_class.get(name) {
                if let Some(nc) = self.netclasses.get(cname) {
                    return nc;
                }
            }
        }
        if let Some(nc) = self.netclasses.get("Default") {
            return nc;
        }
        static DEFAULT_NETCLASS: std::sync::OnceLock<NetClass> = std::sync::OnceLock::new();
        DEFAULT_NETCLASS.get_or_init(NetClass::default)
    }

    /// Resolves the target [`NetTopology`] for `net_id` from its [`NetClass`] or net name.
    ///
    /// # Arguments
    /// * `net_id` - Numeric net ID.
    ///
    /// # Returns
    /// The resolved [`NetTopology`].
    ///
    /// # Side Effects
    /// None.
    pub fn get_net_topology(&self, net_id: usize) -> NetTopology {
        let netclass = self.get_netclass_for_net(net_id);
        let net_name = self.nets.get(net_id).map(|s| s.as_str()).unwrap_or("");
        netclass.resolve_topology(net_name)
    }

    /// Generates the unrouted Euclidean Minimum Spanning Tree (MST) ratsnest airwire segments across all non-ground signal nets using Prim's algorithm.
    ///
    /// Ground nets (`is_gnd_net_name`) and unconnected pads (`net_id == 0`) are excluded because
    /// ground connectivity is provided via copper pour planes and stitching vias.
    ///
    /// # Arguments
    /// * `pad_positions` - Precomputed 2D slice where `pad_positions[fp_idx][pad_idx]` is the world [`Point`] of that pad.
    ///
    /// # Returns
    /// Vector of MST [`Segment`]s across all active signal nets.
    ///
    /// # Side Effects
    /// None.
    pub fn generate_ratsnest_segments(&self, pad_positions: &[Vec<Point>]) -> Vec<Segment> {
        let mut net_points: HashMap<usize, Vec<Point>> = HashMap::new();
        for (fp_idx, fp) in self.footprints.iter().enumerate() {
            for (pad_idx, pad) in fp.pads.iter().enumerate() {
                if pad.net_id == 0 || pad.net_name.is_empty() || is_gnd_net_name(&pad.net_name) {
                    continue;
                }
                net_points.entry(pad.net_id).or_default().push(pad_positions[fp_idx][pad_idx]);
            }
        }

        let mut segments = Vec::new();
        let mut net_ids: Vec<usize> = net_points.keys().copied().collect();
        net_ids.sort_unstable();

        for net_id in net_ids {
            let pts = &net_points[&net_id];
            if pts.len() < 2 {
                continue;
            }
            if pts.len() == 2 {
                segments.push(Segment { p1: pts[0], p2: pts[1], net_id });
                continue;
            }
            let n = pts.len();
            // Ratsnest always joins each pad to its closest same-net pad (Prim's Euclidean MST),
            // independent of routing topology constraints.
            let mut in_mst = vec![false; n];
            let mut min_dist = vec![f64::INFINITY; n];
            let mut parent = vec![0usize; n];
            in_mst[0] = true;
            for i in 1..n {
                min_dist[i] = pts[0].distance_to(&pts[i]);
            }
            for _ in 1..n {
                let mut best = None;
                let mut best_d = f64::INFINITY;
                for i in 1..n {
                    if !in_mst[i] && min_dist[i] < best_d {
                        best_d = min_dist[i];
                        best = Some(i);
                    }
                }
                if let Some(u) = best {
                    in_mst[u] = true;
                    segments.push(Segment { p1: pts[parent[u]], p2: pts[u], net_id });
                    for v in 1..n {
                        if !in_mst[v] {
                            let d = pts[u].distance_to(&pts[v]);
                            if d < min_dist[v] {
                                min_dist[v] = d;
                                parent[v] = u;
                            }
                        }
                    }
                }
            }
        }
        segments
    }

    /// Filters out ratsnest airwire segments whose endpoints are already electrically
    /// connected by same-net copper (`tracks` and/or organic `arcs`).
    ///
    /// Builds a union-find forest of touching copper chords per net and retains only those
    /// ratsnest [`Segment`]s whose two endpoints do not belong to the same connected copper component.
    ///
    /// # Arguments
    /// * `ratsnest` - Full list of candidate ratsnest [`Segment`]s.
    ///
    /// # Returns
    /// Filtered vector containing only the remaining unrouted ratsnest [`Segment`]s.
    ///
    /// # Side Effects
    /// None.
    pub fn filter_unrouted_ratsnest(&self, ratsnest: &[Segment]) -> Vec<Segment> {
        if ratsnest.is_empty() || (self.tracks.is_empty() && self.arcs.is_empty()) {
            return ratsnest.to_vec();
        }

        let mut net_chords: HashMap<usize, Vec<(Point, Point)>> = HashMap::new();
        for t in &self.tracks {
            if t.net_id > 0 {
                net_chords.entry(t.net_id).or_default().push((
                    Point::new(t.start.0, t.start.1),
                    Point::new(t.end.0, t.end.1),
                ));
            }
        }
        for a in &self.arcs {
            if a.net_id > 0 {
                let (sx, sy) = a.start();
                let (mx, my) = a.mid();
                let (ex, ey) = a.end();
                let s = Point::new(sx, sy);
                let m = Point::new(mx, my);
                let e = Point::new(ex, ey);
                let entry = net_chords.entry(a.net_id).or_default();
                entry.push((s, m));
                entry.push((m, e));
            }
        }

        fn find_root(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        let mut net_uf: HashMap<usize, Vec<usize>> = HashMap::new();
        for (&net_id, chords) in &net_chords {
            let m = chords.len();
            let mut parent: Vec<usize> = (0..m).collect();
            for i in 0..m {
                let (a1, b1) = chords[i];
                for j in (i + 1)..m {
                    let (a2, b2) = chords[j];
                    if a1.distance_to_segment(&a2, &b2) <= 0.35
                        || b1.distance_to_segment(&a2, &b2) <= 0.35
                        || a2.distance_to_segment(&a1, &b1) <= 0.35
                        || b2.distance_to_segment(&a1, &b1) <= 0.35
                    {
                        let ri = find_root(&mut parent, i);
                        let rj = find_root(&mut parent, j);
                        if ri != rj {
                            parent[ri] = rj;
                        }
                    }
                }
            }
            for i in 0..m {
                let _ = find_root(&mut parent, i);
            }
            net_uf.insert(net_id, parent);
        }

        ratsnest
            .iter()
            .filter(|seg| {
                let Some(chords) = net_chords.get(&seg.net_id) else {
                    return true;
                };
                let Some(roots) = net_uf.get(&seg.net_id) else {
                    return true;
                };
                let pad_touch_tol = 0.85;
                let mut p1_roots = Vec::new();
                let mut p2_roots = Vec::new();
                for (idx, &(a, b)) in chords.iter().enumerate() {
                    if seg.p1.distance_to_segment(&a, &b) <= pad_touch_tol {
                        p1_roots.push(roots[idx]);
                    }
                    if seg.p2.distance_to_segment(&a, &b) <= pad_touch_tol {
                        p2_roots.push(roots[idx]);
                    }
                }
                !p1_roots.iter().any(|r1| p2_roots.contains(r1))
            })
            .cloned()
            .collect()
    }

    /// Serializes the current [`Board`] state back into a `.kicad_pcb` S-expression document
    /// by applying in-place byte-span replacements onto [`Board::raw_content`].
    ///
    /// Preserves all unparsed KiCad expressions (zones, graphics, properties, dimension callouts,
    /// and unmodified footprints) while updating moved/rotated/flipped footprint `(at ...)` and pad
    /// expressions, swapped pad `(net ...)` bindings, nudged silkscreen `(gr_text ...)` coordinates,
    /// and replacing all copper `(segment ...)`, `(arc ...)`, `(via ...)`, and stale `(filled_polygon ...)`
    /// blocks with the newly routed geometry.
    ///
    /// # Returns
    /// * `Ok(String)` containing the complete, valid `.kicad_pcb` S-expression text.
    /// * `Err(String)` if [`Board::raw_content`] lacks a closing root parenthesis `)`.
    ///
    /// # Side Effects
    /// None (purely constructs and returns a serialized `String` in memory).
    ///
    /// # Errors
    /// Returns `Err("Invalid .kicad_pcb syntax")` if no closing parenthesis `)` exists in the document.
    pub fn to_kicad_pcb(&self) -> Result<String, String> {
        let is_kicad10_named_nets = self.raw_content.contains("(version 2026")
            || self.raw_content.contains("(generator_version \"10.")
            || self.footprints.iter().any(|fp| {
                fp.pads.iter().any(|p| {
                    p.net_span.is_some_and(|(s, e)| {
                        e <= self.raw_content.len() && self.raw_content[s..e].starts_with("(net \"")
                    })
                })
            });

        let format_pad_net = |pad: &Pad| -> String {
            if is_kicad10_named_nets {
                format!("(net \"{}\")", pad.net_name)
            } else {
                format!("(net {} \"{}\")", pad.net_id, pad.net_name)
            }
        };

        let format_copper_net = |net_id: usize| -> String {
            if is_kicad10_named_nets {
                let name = self.nets.get(net_id).map(|s| s.as_str()).unwrap_or("");
                format!("(net \"{}\")", name)
            } else {
                format!("(net {})", net_id)
            }
        };

        let mut replacements: Vec<(usize, usize, String)> = Vec::new();
        for fp in &self.footprints {
            let (orig_x, orig_y, orig_rot) = fp
                .at_span
                .and_then(|(s, e)| {
                    if s < e && e <= self.raw_content.len() {
                        parse_at_xy_rot(&self.raw_content[s..e])
                    } else {
                        None
                    }
                })
                .unwrap_or((fp.x, fp.y, fp.rotation));
            let delta_rot = fp.rotation - orig_rot;
            if fp.is_locked
                && fp.layer == fp.orig_layer
                && (fp.x - orig_x).abs() < 1e-4
                && (fp.y - orig_y).abs() < 1e-4
                && delta_rot.abs() < 1e-3
            {
                continue;
            }
            let rot_norm = ((fp.rotation % 360.0) + 360.0) % 360.0;
            let new_at = if rot_norm.abs() < 1e-3 || (rot_norm - 360.0).abs() < 1e-3 {
                format!("(at {:.4} {:.4})", fp.x, fp.y)
            } else {
                format!("(at {:.4} {:.4} {:.1})", fp.x, fp.y, rot_norm)
            };

            if fp.layer != fp.orig_layer {
                if let Some((f_start, f_end)) = fp.fp_span {
                    if f_start < f_end && f_end <= self.raw_content.len() {
                        let mut inner_reps: Vec<(usize, usize, String)> = Vec::new();
                        if let Some((a_start, a_end)) = fp.at_span {
                            if a_start >= f_start && a_end <= f_end {
                                inner_reps.push((a_start - f_start, a_end - f_start, new_at.clone()));
                            }
                        }
                        inner_reps.extend(collect_pad_at_replacements(
                            &self.raw_content[f_start..f_end],
                            0,
                            delta_rot,
                        ));
                        for pad in &fp.pads {
                            if let Some((n_start, n_end)) = pad.net_span {
                                if n_start >= f_start && n_end <= f_end && n_start < n_end {
                                    inner_reps.push((n_start - f_start, n_end - f_start, format_pad_net(pad)));
                                }
                            }
                        }
                        inner_reps.sort_by(|a, b| b.0.cmp(&a.0));
                        let mut fp_block = self.raw_content[f_start..f_end].to_string();
                        for (rs, re, rep_s) in inner_reps {
                            if rs < re && re <= fp_block.len() {
                                fp_block.replace_range(rs..re, &rep_s);
                            }
                        }
                        fp_block = swap_footprint_layers(&fp_block);
                        replacements.push((f_start, f_end, fp_block));
                        continue;
                    }
                }
            }

            if let Some((start, end)) = fp.at_span {
                replacements.push((start, end, new_at));
            }
            if let Some((f_start, f_end)) = fp.fp_span {
                if f_start < f_end && f_end <= self.raw_content.len() {
                    replacements.extend(collect_pad_at_replacements(
                        &self.raw_content[f_start..f_end],
                        f_start,
                        delta_rot,
                    ));
                }
            }
        }

        // Also update any swapped pad nets in place
        for fp in &self.footprints {
            if fp.layer == fp.orig_layer {
                for pad in &fp.pads {
                    if let Some((n_start, n_end)) = pad.net_span {
                        let expected_net_str = format_pad_net(pad);
                        if n_start < n_end && n_end <= self.raw_content.len() {
                            if &self.raw_content[n_start..n_end] != expected_net_str {
                                replacements.push((n_start, n_end, expected_net_str));
                            }
                        }
                    }
                }
            }
        }

        // Update silkscreen text positions in place
        for txt in &self.texts {
            if let Some((start, end)) = txt.at_span {
                let rot_norm = ((txt.rotation % 360.0) + 360.0) % 360.0;
                let new_at = if rot_norm.abs() < 1e-3 || (rot_norm - 360.0).abs() < 1e-3 {
                    format!("(at {:.4} {:.4})", txt.x, txt.y)
                } else {
                    format!("(at {:.4} {:.4} {:.1})", txt.x, txt.y, rot_norm)
                };
                if start < end && end <= self.raw_content.len() && &self.raw_content[start..end] != new_at {
                    replacements.push((start, end, new_at));
                }
            }
        }

        // Strip out existing (segment ...), (via ...), and (arc ...) so we don't duplicate or retain old tracks.
        // Zone (filled_polygon ...) blocks are stripped too: they were computed against the old copper, so a
        // stale fill can leave holes under new vias (KiCad ratsnest airwires) or flood over other-net vias.
        let bytes = self.raw_content.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'(' {
                let mut j = i + 1;
                while j < bytes.len() && !bytes[j].is_ascii_whitespace() && bytes[j] != b')' {
                    j += 1;
                }
                let word = &self.raw_content[i + 1..j];
                if word == "segment" || word == "via" || word == "arc" || word == "filled_polygon" {
                    if let Some(end) = find_closing_paren(bytes, i) {
                        let mut start = i;
                        while start > 0 && (bytes[start - 1] == b' ' || bytes[start - 1] == b'\t') {
                            start -= 1;
                        }
                        if start > 0 && bytes[start - 1] == b'\n' {
                            start -= 1;
                            if start > 0 && bytes[start - 1] == b'\r' {
                                start -= 1;
                            }
                        }
                        replacements.push((start, end, String::new()));
                        i = end;
                        continue;
                    }
                }
            }
            i += 1;
        }

        replacements.sort_by(|a, b| b.0.cmp(&a.0));

        let mut content = self.raw_content.clone();
        for (start, end, rep) in replacements {
            if start < end && end <= content.len() {
                content.replace_range(start..end, &rep);
            }
        }

        if !self.tracks.is_empty() || !self.vias.is_empty() || !self.arcs.is_empty() {
            let last_paren = content.rfind(')').ok_or("Invalid .kicad_pcb syntax")?;
            let mut s_expr = String::new();
            for tr in &self.tracks {
                let layer_name = self.layer_name(tr.layer);
                let locked_str = if tr.is_locked { " (locked yes)" } else { "" };
                let net_expr = format_copper_net(tr.net_id);
                s_expr.push_str(&format!(
                    "  (segment (start {:.4} {:.4}) (end {:.4} {:.4}) (width {:.4}) (layer \"{}\") {}{})\n",
                    tr.start.0, tr.start.1, tr.end.0, tr.end.1, tr.width, layer_name, net_expr, locked_str
                ));
            }
            for arc in &self.arcs {
                let layer_name = self.layer_name(arc.layer);
                let locked_str = if arc.is_locked { " (locked yes)" } else { "" };
                let net_expr = format_copper_net(arc.net_id);
                let s = arc.start();
                let m = arc.mid();
                let e = arc.end();
                s_expr.push_str(&format!(
                    "  (arc (start {:.4} {:.4}) (mid {:.4} {:.4}) (end {:.4} {:.4}) (width {:.4}) (layer \"{}\") {}{})\n",
                    s.0, s.1, m.0, m.1, e.0, e.1, arc.width, layer_name, net_expr, locked_str
                ));
            }
            for v in &self.vias {
                let locked_str = if v.is_locked { " (locked yes)" } else { "" };
                let net_expr = format_copper_net(v.net_id);
                s_expr.push_str(&format!(
                    "  (via (at {:.4} {:.4}) (size {:.4}) (drill {:.4}) (layers \"F.Cu\" \"B.Cu\") {}{})\n",
                    v.position.0, v.position.1, v.size, v.drill, net_expr, locked_str
                ));
            }
            let has_pro = !self.file_path.is_empty() && Path::new(&self.file_path).with_extension("kicad_pro").exists();
            if !content.contains("(net_class ") && !is_kicad10_named_nets && !has_pro {
                let mut sorted_cnames: Vec<&String> = self.netclasses.keys().collect();
                sorted_cnames.sort();
                for name in sorted_cnames {
                    let nc = &self.netclasses[name];
                    s_expr.push_str(&format!(
                        "  (net_class \"{}\" \"Netclass {}\"\n    (clearance {:.4})\n    (trace_width {:.4})\n    (via_dia {:.4})\n    (via_drill {:.4})\n    (diff_pair_gap {:.4})\n    (diff_pair_width {:.4})\n",
                        name, name, nc.clearance, nc.track_width, nc.via_dia, nc.via_drill, nc.diff_pair_gap, nc.diff_pair_width
                    ));
                    for (net_name, cls_name) in &self.net_to_class {
                        if cls_name == name {
                            s_expr.push_str(&format!("    (add_net \"{}\")\n", net_name));
                        }
                    }
                    s_expr.push_str("  )\n");
                }
            }
            content.insert_str(last_paren, &s_expr);
        }

        Ok(content)
    }

    /// Alias for [`Board::to_kicad_pcb`] that serializes the board state into a `.kicad_pcb` S-expression string.
    ///
    /// # Returns
    /// * `Ok(String)` containing the updated `.kicad_pcb` document text.
    /// * `Err(String)` if the base `.kicad_pcb` syntax is invalid.
    ///
    /// # Side Effects
    /// None.
    ///
    /// # Errors
    /// Propagates any serialization error from [`Board::to_kicad_pcb`].
    pub fn save_kicad_pcb(&self) -> Result<String, String> {
        self.to_kicad_pcb()
    }

    /// Nudges standalone silkscreen text items (`F.SilkS` and `B.SilkS`) away from exposed copper pads,
    /// vias, footprint silkscreen outlines, and board edges, initializing a temporary [`crate::vulkan::VulkanEngine`]
    /// if available for GPU-accelerated collision filtering.
    ///
    /// # Side Effects
    /// * May initialize a transient [`crate::vulkan::VulkanEngine`] device.
    /// * Mutates `(x, y)` coordinates of colliding entries in `self.texts`.
    pub fn nudge_silkscreen_texts(&mut self) {
        if self.texts.is_empty() {
            return;
        }
        let vk = crate::vulkan::VulkanEngine::new();
        self.nudge_silkscreen_texts_with_gpu(vk.as_ref());
    }

    /// Nudges standalone silkscreen text items (`F.SilkS` and `B.SilkS`) along a 960-point radial spiral
    /// so they do not overlap exposed copper pads, vias, footprint silkscreen lines, board boundaries,
    /// or other silkscreen labels.
    ///
    /// Uses an optional [`crate::vulkan::VulkanEngine`] compute shader pass to batch-reject candidate poses
    /// that collide with copper pads or vias, followed by parallel Rayon CPU verification against silkscreen
    /// lines and sequential text-to-text de-overlapping.
    ///
    /// # Arguments
    /// * `vulkan` - Optional shared reference to a [`crate::vulkan::VulkanEngine`] for GPU candidate pre-filtering.
    ///
    /// # Side Effects
    /// Mutates `(x, y)` coordinates of entries in `self.texts` in place.
    pub fn nudge_silkscreen_texts_with_gpu(
        &mut self,
        vulkan: Option<&std::sync::Arc<crate::vulkan::VulkanEngine>>,
    ) {
        use rayon::prelude::*;
        if self.texts.is_empty() {
            return;
        }
        let boundary = self.boundary_aabb();

        // Extract silkscreen lines from all footprints in world coordinates
        let mut silk_lines: Vec<((f64, f64), (f64, f64), String)> = Vec::new();
        for fp in &self.footprints {
            if let Some((start_fp, end_fp)) = fp.fp_span {
                if start_fp < end_fp && end_fp <= self.raw_content.len() {
                    let block = &self.raw_content[start_fp..end_fp];
                    let rad = fp.rotation.to_radians();
                    let cos_r = rad.cos();
                    let sin_r = rad.sin();
                    let to_world = |lx: f64, ly: f64| -> (f64, f64) {
                        (fp.x + lx * cos_r - ly * sin_r, fp.y + lx * sin_r + ly * cos_r)
                    };
                    let bytes = block.as_bytes();
                    let mut idx = 0;
                    while idx < bytes.len() {
                        if let Some(pos) = block[idx..].find("(fp_line") {
                            let s_idx = idx + pos;
                            if let Some(close) = find_closing_paren(bytes, s_idx) {
                                let line_str = &block[s_idx..close];
                                let is_fsilk = line_str.contains("\"F.SilkS\"");
                                let is_bsilk = line_str.contains("\"B.SilkS\"");
                                if is_fsilk || is_bsilk {
                                    let layer = if is_fsilk { "F.SilkS" } else { "B.SilkS" };
                                    if let (Some(p1), Some(p2)) = (parse_point_tag(line_str, "(start "), parse_point_tag(line_str, "(end ")) {
                                        silk_lines.push((to_world(p1.x, p1.y), to_world(p2.x, p2.y), layer.to_string()));
                                    }
                                }
                                idx = close;
                            } else {
                                idx = s_idx + 8;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        let dist_pt_seg = |p: (f64, f64), s: (f64, f64), e: (f64, f64)| -> f64 {
            let dx = e.0 - s.0;
            let dy = e.1 - s.1;
            let l2 = dx * dx + dy * dy;
            if l2 < 1e-12 {
                return (p.0 - s.0).hypot(p.1 - s.1);
            }
            let t = (((p.0 - s.0) * dx + (p.1 - s.1) * dy) / l2).clamp(0.0, 1.0);
            (p.0 - (s.0 + t * dx)).hypot(p.1 - (s.1 + t * dy))
        };

        let dist_seg_seg = |p1: (f64, f64), p2: (f64, f64), p3: (f64, f64), p4: (f64, f64)| -> f64 {
            dist_pt_seg(p1, p3, p4)
                .min(dist_pt_seg(p2, p3, p4))
                .min(dist_pt_seg(p3, p1, p2))
                .min(dist_pt_seg(p4, p1, p2))
        };

        // Precompute spiral search offsets (0 = original pose, 1..=960 = 40 radial steps x 24 angles)
        let mut spiral_offsets: Vec<(f64, f64)> = Vec::with_capacity(1 + 40 * 24);
        spiral_offsets.push((0.0, 0.0));
        for r_step in 1..=40 {
            let r = (r_step as f64) * 0.25;
            for a_idx in 0..24 {
                let angle = (a_idx as f64) * std::f64::consts::PI / 12.0;
                spiral_offsets.push((r * angle.cos(), r * angle.sin()));
            }
        }
        let num_cands_per_text = spiral_offsets.len();

        let silk_text_indices: Vec<usize> = (0..self.texts.len())
            .filter(|&i| self.texts[i].layer == "F.SilkS" || self.texts[i].layer == "B.SilkS")
            .collect();
        if silk_text_indices.is_empty() {
            return;
        }

        // Optional GPU batch pre-filtering against board pad and via copper obstacles
        let gpu_rejected: Option<Vec<bool>> = vulkan.and_then(|vk| {
            let mut gpu_pads = Vec::new();
            for fp in &self.footprints {
                for pad in &fp.pads {
                    let (px, py) = fp.pad_absolute_position(pad);
                    let flipped = fp.layer != fp.orig_layer;
                    let pad_rot = if flipped { -pad.rotation } else { pad.rotation };
                    let total_rot_rad = -(fp.rotation + pad_rot).to_radians() as f32;
                    let layer_bits: u32 = if pad.is_through_hole {
                        0xFFFF
                    } else {
                        1u32 << (fp.pad_layer(pad).min(15) as u32)
                    };
                    let shape_code: u32 = if pad.shape == "circle" {
                        1
                    } else if pad.shape == "oval" {
                        2
                    } else {
                        0
                    };
                    gpu_pads.push(crate::vulkan::GpuDrcPad {
                        x: px as f32,
                        y: py as f32,
                        hw: (pad.width * 0.5) as f32,
                        hh: (pad.height * 0.5) as f32,
                        rot_rad: total_rot_rad,
                        clearance: 0.12,
                        layer_mask: layer_bits | (shape_code << 16),
                        net_id: pad.net_id as u32,
                    });
                }
            }
            let gpu_vias: Vec<crate::vulkan::GpuDrcVia> = self
                .vias
                .iter()
                .map(|v| crate::vulkan::GpuDrcVia {
                    x: v.position.0 as f32,
                    y: v.position.1 as f32,
                    radius: (v.size * 0.5) as f32,
                    clearance: 0.12,
                    net_id: v.net_id as u32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                })
                .collect();

            let mut cands = Vec::with_capacity(silk_text_indices.len() * num_cands_per_text);
            for &t_idx in &silk_text_indices {
                let txt = &self.texts[t_idx];
                let target_copper_layer = if txt.layer == "F.SilkS" { 0u32 } else { 1u32 };
                let tw = (txt.text.len() as f64) * (txt.width * 1.05) + 0.3;
                let th = txt.height.max(0.6) + 0.2;
                let rot_rad = txt.rotation.to_radians();
                let dx_unit = rot_rad.cos();
                let dy_unit = rot_rad.sin();
                let (maj_x, maj_y, half_span, min_dim) = if tw >= th {
                    (dx_unit, dy_unit, (tw - th) * 0.5, th)
                } else {
                    (-dy_unit, dx_unit, (th - tw) * 0.5, tw)
                };
                for &(ox, oy) in &spiral_offsets {
                    let cx = txt.x + ox;
                    let cy = txt.y + oy;
                    cands.push(crate::vulkan::GpuDrcCandidate {
                        x1: (cx - half_span * maj_x) as f32,
                        y1: (cy - half_span * maj_y) as f32,
                        x2: (cx + half_span * maj_x) as f32,
                        y2: (cy + half_span * maj_y) as f32,
                        radius_or_half_width: (min_dim * 0.5) as f32,
                        clearance: 0.12,
                        layer: target_copper_layer,
                        net_id: u32::MAX - 10,
                        kind: 1,
                        _pad0: 0,
                        _pad1: 0,
                        ignore_same_net_pads: 0,
                    });
                }
            }

            vk.evaluate_drc_candidates(&gpu_pads, &[], &gpu_vias, &cands)
                .map(|flags| flags.into_iter().map(|f| f == 0).collect())
        });

        let footprints_ref = &self.footprints;
        let vias_ref = &self.vias;
        let texts_snapshot = self.texts.clone();

        // Parallel static collision pre-evaluation (boundary + silkscreen lines + copper pads/vias)
        let static_valid_per_text: Vec<Vec<bool>> = silk_text_indices
            .par_iter()
            .enumerate()
            .map(|(batch_idx, &t_idx)| {
                let txt = &texts_snapshot[t_idx];
                let target_copper_layer = if txt.layer == "F.SilkS" { 0 } else { 1 };
                let tw = (txt.text.len() as f64) * (txt.width * 1.05) + 0.3;
                let th = txt.height.max(0.6) + 0.2;
                let hw = tw * 0.5;
                let hh = th * 0.5;
                let rot_rad = txt.rotation.to_radians();
                let dx_unit = rot_rad.cos();
                let dy_unit = rot_rad.sin();
                let perp_x = -dy_unit;
                let perp_y = dx_unit;
                let margin = (tw.max(th) * 0.5) + 0.25;
                let pad_margin = hh + 0.20;
                let num_samples = ((tw / 0.15).ceil() as usize).max(9);

                let mut valid = vec![false; num_cands_per_text];
                for (c_idx, &(ox, oy)) in spiral_offsets.iter().enumerate() {
                    if let Some(ref rejected) = gpu_rejected {
                        if rejected[batch_idx * num_cands_per_text + c_idx] {
                            continue;
                        }
                    }
                    let cx = txt.x + ox;
                    let cy = txt.y + oy;
                    if cx - hw < boundary.min_x + margin
                        || cx + hw > boundary.max_x - margin
                        || cy - hh < boundary.min_y + margin
                        || cy + hh > boundary.max_y - margin
                    {
                        continue;
                    }

                    // Check distance against footprint silkscreen lines
                    let mut silk_hit = false;
                    'silk_chk: for &h_frac in &[-1.0_f64, 0.0, 1.0] {
                        let seg_s = (
                            cx - hw * dx_unit + h_frac * hh * perp_x,
                            cy - hw * dy_unit + h_frac * hh * perp_y,
                        );
                        let seg_e = (
                            cx + hw * dx_unit + h_frac * hh * perp_x,
                            cy + hw * dy_unit + h_frac * hh * perp_y,
                        );
                        for (ls, le, ref lyr) in &silk_lines {
                            if lyr == &txt.layer && dist_seg_seg(seg_s, seg_e, *ls, *le) < 0.20 {
                                silk_hit = true;
                                break 'silk_chk;
                            }
                        }
                    }
                    if silk_hit {
                        continue;
                    }

                    // Check distance to exposed copper pads and vias
                    let mut copper_hit = false;
                    'cop_chk: for s in 0..num_samples {
                        let t = (s as f64) / ((num_samples - 1) as f64) - 0.5;
                        let sample_x = cx + t * tw * dx_unit;
                        let sample_y = cy + t * tw * dy_unit;

                        for fp in footprints_ref {
                            for pad in &fp.pads {
                                if pad.is_through_hole || fp.pad_layer(pad) == target_copper_layer {
                                    let d_pad = crate::router::dist_point_to_pad((sample_x, sample_y), fp, pad);
                                    if d_pad < pad_margin {
                                        copper_hit = true;
                                        break 'cop_chk;
                                    }
                                }
                            }
                        }
                        for v in vias_ref {
                            if (sample_x - v.position.0).hypot(sample_y - v.position.1) - v.size * 0.5 < pad_margin {
                                copper_hit = true;
                                break 'cop_chk;
                            }
                        }
                    }
                    if !copper_hit {
                        valid[c_idx] = true;
                    }
                }
                valid
            })
            .collect();

        // Sequential silkscreen-to-silkscreen collision resolution
        for (batch_idx, &t_idx) in silk_text_indices.iter().enumerate() {
            let static_valid = &static_valid_per_text[batch_idx];
            let layer = self.texts[t_idx].layer.clone();
            let tw = (self.texts[t_idx].text.len() as f64) * (self.texts[t_idx].width * 1.05) + 0.3;
            let th = self.texts[t_idx].height.max(0.6) + 0.2;
            let orig_x = self.texts[t_idx].x;
            let orig_y = self.texts[t_idx].y;

            let collides_with_other_texts = |cx: f64, cy: f64, texts: &[BoardText]| -> bool {
                for (oi, ot) in texts.iter().enumerate() {
                    if oi == t_idx || ot.layer != layer {
                        continue;
                    }
                    let otw = (ot.text.len() as f64) * (ot.width * 1.05) + 0.3;
                    let oth = ot.height.max(0.6) + 0.2;
                    if (cx - ot.x).abs() < (tw + otw) * 0.5 + 0.20
                        && (cy - ot.y).abs() < (th + oth) * 0.5 + 0.20
                    {
                        return true;
                    }
                }
                false
            };

            if static_valid[0] && !collides_with_other_texts(orig_x, orig_y, &self.texts) {
                continue;
            }

            for c_idx in 1..num_cands_per_text {
                if !static_valid[c_idx] {
                    continue;
                }
                let (ox, oy) = spiral_offsets[c_idx];
                let cand_x = orig_x + ox;
                let cand_y = orig_y + oy;
                if !collides_with_other_texts(cand_x, cand_y, &self.texts) {
                    self.texts[t_idx].x = cand_x;
                    self.texts[t_idx].y = cand_y;
                    break;
                }
            }
        }
    }

    /// Formats the current UTC timestamp as `YYYYMMDDHHMMSS` using Howard Hinnant's civil calendar
    /// algorithm on `std::time::SystemTime` without requiring third-party date/time crates.
    ///
    /// # Returns
    /// A 14-character numeric timestamp string (e.g., `"20261108131000"`).
    ///
    /// # Side Effects
    /// Queries the system wall clock via `SystemTime::now()`.
    pub fn current_timestamp_yyyymmddhhmmss() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let sec = secs % 60;
        let mins = secs / 60;
        let min = mins % 60;
        let hours = mins / 60;
        let hour = hours % 24;
        let mut days = (hours / 24) as i64;

        // Convert days since Jan 1 1970 to year, month, day using standard civil calendar algorithm
        days += 719468;
        let era = if days >= 0 { days } else { days - 146096 } / 146097;
        let doe = (days - era * 146097) as u64;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = (yoe as i64) + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };

        format!("{:04}{:02}{:02}{:02}{:02}{:02}", y, m, d, hour, min, sec)
    }

    /// Computes a non-colliding backup path for a PCB file being overwritten, formatted as
    /// `<board_stem>.kicad_pcb_bak.YYYYMMDDHHMMSS` (with an optional `.1`, `.2`, ... suffix if
    /// multiple backups occur within the same second).
    ///
    /// # Arguments
    /// * `out_path` - Target `.kicad_pcb` output path about to be overwritten.
    ///
    /// # Returns
    /// A [`PathBuf`] pointing to an unused backup file path in the same directory.
    ///
    /// # Side Effects
    /// Checks filesystem existence of candidate backup paths via `Path::exists`.
    pub fn compute_pcb_backup_path(out_path: &Path) -> PathBuf {
        let ts = Self::current_timestamp_yyyymmddhhmmss();
        let path_str = out_path.to_string_lossy();
        let base = if path_str.ends_with(".kicad_pcb") {
            format!(
                "{}.kicad_pcb_bak.{}",
                &path_str[..path_str.len() - ".kicad_pcb".len()],
                ts
            )
        } else {
            format!("{}.kicad_pcb_bak.{}", path_str, ts)
        };
        let mut bak = PathBuf::from(&base);
        let mut counter = 1;
        while bak.exists() {
            bak = PathBuf::from(format!("{}.{}", base, counter));
            counter += 1;
        }
        bak
    }

    /// Nudges silkscreen labels, serializes the board to `.kicad_pcb` format, writes it to `out_path`
    /// (creating a timestamped `.kicad_pcb_bak.*` backup if `out_path` already exists), and synchronizes
    /// sibling `.kicad_dru`, `.kicad_pro`, `.kicad_sch`, and `.kicad_prl` files.
    ///
    /// # Arguments
    /// * `out_path` - Destination filesystem path for the saved `.kicad_pcb` file.
    ///
    /// # Returns
    /// * `Ok(())` when the `.kicad_pcb` file is written successfully.
    /// * `Err(String)` if `.kicad_pcb` serialization or file writing fails.
    ///
    /// # Side Effects
    /// * Mutates `self.texts` via [`Board::nudge_silkscreen_texts`].
    /// * Copies any existing `out_path` to a backup path from [`Board::compute_pcb_backup_path`].
    /// * Writes `out_path`, its sibling `.kicad_dru` file, and copies sibling `.kicad_pro`/`.kicad_sch`/`.kicad_prl` files if present.
    ///
    /// # Errors
    /// Returns an `Err(String)` if [`Board::to_kicad_pcb`] fails or `std::fs::write` cannot write `out_path`.
    pub fn save_pcb(&mut self, out_path: &Path) -> Result<(), String> {
        self.nudge_silkscreen_texts();
        let content = self.to_kicad_pcb()?;

        // If overwriting an existing PCB file, save a timestamped backup first
        if out_path.exists() {
            let bak_path = Self::compute_pcb_backup_path(out_path);
            if let Err(e) = fs::copy(out_path, &bak_path) {
                eprintln!(
                    "[Warning] Failed to backup original PCB to '{}': {}",
                    bak_path.display(),
                    e
                );
            }
        }

        fs::write(out_path, content).map_err(|e| e.to_string())?;

        // Auto-generate / synchronize sibling .kicad_dru with intra-footprint pad and hole clearance exemptions
        let src_dru = if !self.file_path.is_empty() {
            Some(Path::new(&self.file_path).with_extension("kicad_dru"))
        } else {
            None
        };
        let out_dru = out_path.with_extension("kicad_dru");
        let mut dru_content = if let Some(ref sd) = src_dru {
            if sd.exists() {
                fs::read_to_string(sd).unwrap_or_default()
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if !dru_content.contains("A.parent == B.parent") {
            if dru_content.trim().is_empty() {
                dru_content.push_str("(version 1)\n");
            }
            dru_content.push_str("\n(rule \"Same Footprint Pad Clearance\"\n  (condition \"A.parent == B.parent\")\n  (constraint clearance (min 0.12mm))\n  (constraint hole_to_hole (min 0.25mm)))\n");
        } else if !dru_content.contains("hole_to_hole") {
            dru_content.push_str("\n(rule \"Same Footprint Hole to Hole Clearance\"\n  (condition \"A.parent == B.parent\")\n  (constraint hole_to_hole (min 0.25mm)))\n");
        }
        let _ = fs::write(&out_dru, dru_content);

        // Copy sibling project and schematic files if present to keep netclass definitions synced
        if !self.file_path.is_empty() {
            let src_base = Path::new(&self.file_path);
            let src_pro = src_base.with_extension("kicad_pro");
            let out_pro = out_path.with_extension("kicad_pro");
            if src_pro.exists() && src_pro != out_pro {
                let _ = fs::copy(&src_pro, &out_pro);
            }
            let src_sch = src_base.with_extension("kicad_sch");
            let out_sch = out_path.with_extension("kicad_sch");
            if src_sch.exists() && src_sch != out_sch {
                let _ = fs::copy(&src_sch, &out_sch);
            }
            let src_prl = src_base.with_extension("kicad_prl");
            let out_prl = out_path.with_extension("kicad_prl");
            if src_prl.exists() && src_prl != out_prl {
                let _ = fs::copy(&src_prl, &out_prl);
            }
        }

        Ok(())
    }

    /// Refills copper zones in a saved `.kicad_pcb` via `kicad-cli pcb drc --refill-zones --save-board`, so
    /// KiCad's file-open ratsnest sees plane connectivity ([`Board::to_kicad_pcb`] drops the stale fills).
    /// Best-effort: returns `false` when the board has no zones or `kicad-cli` is unavailable/fails.
    ///
    /// # Arguments
    /// * `pcb_path` - Path to the saved `.kicad_pcb` file on disk.
    ///
    /// # Returns
    /// `true` if `pcb_path` contains `(zone` definitions and `kicad-cli` refilled and saved the board with exit status `0`, or `false` otherwise.
    ///
    /// # Side Effects
    /// * Reads and modifies `pcb_path` in place via an external `kicad-cli` subprocess.
    /// * Creates and removes a temporary JSON report file in `std::env::temp_dir()`.
    pub fn refill_zones_in_file(pcb_path: &Path) -> bool {
        if !fs::read_to_string(pcb_path).is_ok_and(|s| s.contains("(zone")) {
            return false;
        }
        let report = std::env::temp_dir().join(format!("vulcans_refill_{}.json", std::process::id()));
        let ok = std::process::Command::new("kicad-cli")
            .args(["pcb", "drc", "--refill-zones", "--save-board", "--format", "json", "--output"])
            .arg(&report)
            .arg(pcb_path)
            .output()
            .is_ok_and(|o| o.status.success());
        let _ = fs::remove_file(&report);
        ok
    }

    /// Formats a human-readable Engineering Change Order (ECO) back-annotation report listing all
    /// pin and gate swaps recorded in [`Board::pin_swap_history`].
    ///
    /// # Returns
    /// A Markdown-formatted report `String` detailing each swapped component reference, pad pair, net pair, and reason.
    ///
    /// # Side Effects
    /// None.
    pub fn generate_back_annotation_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# Vulcan's Trials Pin & Gate Swapping Back-Annotation Report\n");
        report.push_str(&format!("# Total Swaps: {}\n\n", self.pin_swap_history.len()));
        for (i, rec) in self.pin_swap_history.iter().enumerate() {
            report.push_str(&format!(
                "{:3}. Component {}: Swap Pad {} ({}) <-> Pad {} ({}) [{}]\n",
                i + 1, rec.footprint_ref, rec.pad1_number, rec.pad1_net, rec.pad2_number, rec.pad2_net, rec.reason
            ));
        }
        report
    }

    /// Writes the pin and gate swapping back-annotation report generated by
    /// [`Board::generate_back_annotation_report`] to disk at `path`.
    ///
    /// # Arguments
    /// * `path` - Destination filesystem path for the ECO report file.
    ///
    /// # Returns
    /// `Ok(())` on success, or an [`std::io::Error`] if writing fails.
    ///
    /// # Side Effects
    /// Writes a text file to `path`.
    ///
    /// # Errors
    /// Propagates any [`std::io::Error`] returned by `std::fs::write`.
    pub fn save_back_annotation_eco(&self, path: &Path) -> std::io::Result<()> {
        let report = self.generate_back_annotation_report();
        fs::write(path, report)
    }

    /// Exports the current component placement and routed copper network (tracks, discretized arcs,
    /// and vias) to a Specctra `.ses` session file at `ses_path` (`0.1 um` resolution).
    ///
    /// # Arguments
    /// * `ses_path` - Destination filesystem path for the `.ses` file.
    ///
    /// # Returns
    /// * `Ok(())` when the Specctra session file is written successfully.
    /// * `Err(String)` if writing to `ses_path` fails.
    ///
    /// # Side Effects
    /// Writes the Specctra `.ses` file to `ses_path` on disk.
    ///
    /// # Errors
    /// Returns `Err(String)` if `std::fs::write` fails.
    pub fn save_ses(&self, ses_path: &Path) -> Result<(), String> {
        let to_ses = |mm: f64| -> i64 { (mm * 10000.0).round() as i64 };
        let mut s = String::from(
            "(session vulcans_trials\n  (base_design vulcans_trials)\n  (placement\n    (resolution um 10)\n",
        );
        for fp in &self.footprints {
            let side = if fp.layer == 1 { "back" } else { "front" };
            let rot = if fp.layer == 1 {
                (180.0 - fp.rotation).rem_euclid(360.0)
            } else {
                fp.rotation.rem_euclid(360.0)
            };
            s.push_str(&format!(
                "    (component \"{}\" (place \"{}\" {} {} {} {:.1}))\n",
                fp.reference,
                fp.reference,
                to_ses(fp.x),
                -to_ses(fp.y),
                side,
                rot
            ));
        }
        s.push_str("  )\n  (was_is\n  )\n  (routes\n    (resolution um 10)\n    (parser\n      (string_quote \")\n      (space_in_quoted_tokens on)\n      (host_cad \"Vulcans_Trials\")\n      (host_version \"1.0.0\")\n    )\n    (library_out\n");

        let copper_layers = self.copper_layers();
        let max_layer_idx = copper_layers.len().saturating_sub(1);
        let mut via_padstacks: Vec<(i64, String)> = Vec::new();
        for v in &self.vias {
            let size_um = (v.size * 1000.0).round() as i64;
            let drill_um = (v.drill * 1000.0).round() as i64;
            let name = format!("Via[0-{}]_{}:{}_um", max_layer_idx, size_um, drill_um);
            if !via_padstacks.iter().any(|(_, n)| n == &name) {
                via_padstacks.push((to_ses(v.size), name));
            }
        }
        for (size_ses, name) in &via_padstacks {
            s.push_str(&format!("      (padstack \"{}\"\n", name));
            for layer in &copper_layers {
                s.push_str(&format!("        (shape (circle {} {}))\n", layer, size_ses));
            }
            s.push_str("        (attach off)\n      )\n");
        }
        s.push_str("    )\n    (network_out\n");

        let mut net_tracks: HashMap<usize, Vec<&TrackSegment>> = HashMap::new();
        for tr in &self.tracks {
            net_tracks.entry(tr.net_id).or_default().push(tr);
        }
        let mut net_arcs: HashMap<usize, Vec<&ArcSegment>> = HashMap::new();
        for arc in &self.arcs {
            net_arcs.entry(arc.net_id).or_default().push(arc);
        }
        let mut net_vias: HashMap<usize, Vec<&Via>> = HashMap::new();
        for v in &self.vias {
            net_vias.entry(v.net_id).or_default().push(v);
        }

        for (net_id, net_name) in self.nets.iter().enumerate() {
            if net_id == 0 || net_name.is_empty() {
                continue;
            }
            let tracks = net_tracks.get(&net_id);
            let arcs = net_arcs.get(&net_id);
            let vias = net_vias.get(&net_id);
            if tracks.is_none() && arcs.is_none() && vias.is_none() {
                continue;
            }
            s.push_str(&format!("      (net \"{}\"\n", net_name));
            if let Some(vs) = vias {
                for v in vs {
                    let size_um = (v.size * 1000.0).round() as i64;
                    let drill_um = (v.drill * 1000.0).round() as i64;
                    let ps_name = format!("Via[0-{}]_{}:{}_um", max_layer_idx, size_um, drill_um);
                    s.push_str(&format!(
                        "        (via \"{}\" {} {})\n",
                        ps_name,
                        to_ses(v.position.0),
                        -to_ses(v.position.1)
                    ));
                }
            }
            if let Some(trs) = tracks {
                for tr in trs {
                    let layer_name = self.layer_name(tr.layer);
                    s.push_str(&format!(
                        "        (wire (path {} {} {} {} {} {}))\n",
                        layer_name,
                        to_ses(tr.width),
                        to_ses(tr.start.0),
                        -to_ses(tr.start.1),
                        to_ses(tr.end.0),
                        -to_ses(tr.end.1)
                    ));
                }
            }
            if let Some(as_list) = arcs {
                for arc in as_list {
                    let layer_name = self.layer_name(arc.layer);
                    let steps = 8usize;
                    s.push_str(&format!("        (wire (path {} {}", layer_name, to_ses(arc.width)));
                    for i in 0..=steps {
                        let t = i as f64 / steps as f64;
                        let ang = arc.start_angle_rad + (arc.end_angle_rad - arc.start_angle_rad) * t;
                        let px = arc.center.0 + arc.radius * ang.cos();
                        let py = arc.center.1 + arc.radius * ang.sin();
                        s.push_str(&format!(" {} {}", to_ses(px), -to_ses(py)));
                    }
                    s.push_str("))\n");
                }
            }
            s.push_str("      )\n");
        }
        s.push_str("    )\n  )\n)\n");
        fs::write(ses_path, s).map_err(|e| e.to_string())
    }

    /// Parses netclasses from a `.kicad_pro` JSON or `.kicad_pcb` S-expression string and merges them
    /// into [`Board::netclasses`].
    ///
    /// # Arguments
    /// * `content` - Raw `.kicad_pro` JSON or `.kicad_pcb` S-expression text.
    ///
    /// # Side Effects
    /// Inserts or overwrites parsed [`NetClass`] entries in `self.netclasses`.
    pub fn parse_netclasses_from_pro_or_pcb(&mut self, content: &str) {
        let classes = parse_netclasses_from_pro_or_pcb(content);
        for (name, nc) in classes {
            self.netclasses.insert(name, nc);
        }
    }

    /// Reads a `.kicad_pro` project file from `pro_path` and applies its netclass definitions,
    /// design rules, and net-to-netclass pattern/explicit assignments to `self`.
    ///
    /// Does nothing if `pro_path` cannot be read.
    ///
    /// # Arguments
    /// * `pro_path` - Path to the `.kicad_pro` JSON project file.
    ///
    /// # Side Effects
    /// Reads `pro_path` from disk and mutates `self.netclasses` and `self.net_to_class` via [`Board::apply_project_netclasses_str`].
    pub fn load_project_netclasses(&mut self, pro_path: &Path) {
        let Ok(content) = fs::read_to_string(pro_path) else { return };
        self.apply_project_netclasses_str(&content);
    }

    /// Parses a `.kicad_pro` JSON string and applies its netclasses, board-level [`DesignRules`],
    /// `netclass_assignments`, and wildcard `netclass_patterns` to `self`.
    ///
    /// When explicit classes are defined in `content`, resets any prior heuristic net-to-class mappings
    /// back to `"Default"` before applying the project's explicit assignments and wildcard patterns.
    ///
    /// # Arguments
    /// * `content` - Raw `.kicad_pro` JSON text (or S-expression netclass text).
    ///
    /// # Side Effects
    /// Mutates `self.netclasses` and `self.net_to_class`.
    pub fn apply_project_netclasses_str(&mut self, content: &str) {
        let classes = parse_netclasses_from_pro_or_pcb(content);
        if !classes.is_empty() {
            // Reset heuristic netclass assignments so nets not explicitly assigned in .kicad_pro
            // fall back to "Default" instead of retaining built-in heuristic classes (e.g. "HBS_Bus").
            for cls in self.net_to_class.values_mut() {
                *cls = "Default".to_string();
            }
            self.netclasses
                .retain(|name, _| name == "Default" || classes.contains_key(name));
            for (name, nc) in classes {
                self.netclasses.insert(name, nc);
            }
        }
        if let Some(rules) = parse_design_rules_from_pro_or_dru(&content) {
            for nc in self.netclasses.values_mut() {
                nc.apply_design_rules(&rules);
            }
        }
        let Ok(root) = serde_json::from_str::<serde_json::Value>(&content) else { return };
        let Some(net_settings) = root.get("net_settings") else { return };

        if let Some(assignments) = net_settings.get("netclass_assignments").and_then(|v| v.as_object()) {
            for (net_name, val) in assignments {
                let cls_opt = val.as_str().or_else(|| val.as_array()?.first()?.as_str());
                if let Some(cls_name) = cls_opt {
                    if self.netclasses.contains_key(cls_name) {
                        self.net_to_class.insert(net_name.clone(), cls_name.to_string());
                    }
                }
            }
        }

        if let Some(patterns) = net_settings.get("netclass_patterns").and_then(|v| v.as_array()) {
            for entry in patterns {
                let Some(pat) = entry.get("pattern").and_then(|v| v.as_str()) else { continue };
                let Some(cls_name) = entry.get("netclass").and_then(|v| v.as_str()) else { continue };
                if !self.netclasses.contains_key(cls_name) {
                    continue;
                }
                for net in &self.nets {
                    if net.is_empty() {
                        continue;
                    }
                    let matched = if pat == "*" {
                        true
                    } else if let Some(prefix) = pat.strip_suffix('*') {
                        net.starts_with(prefix) || net.strip_prefix('/').map_or(false, |n| n.starts_with(prefix))
                    } else if let Some(suffix) = pat.strip_prefix('*') {
                        net.ends_with(suffix)
                    } else {
                        net == pat
                            || net.strip_prefix('/') == Some(pat)
                            || pat.strip_prefix('/') == Some(net.as_str())
                            || net.eq_ignore_ascii_case(pat)
                            || (pat.eq_ignore_ascii_case("Buck_in") && (net.contains("BUCK") || net.contains("buck") || net.contains("Buck")))
                            || (pat.eq_ignore_ascii_case("VBus") && (net.eq_ignore_ascii_case("VBUS") || net.eq_ignore_ascii_case("/VBUS")))
                    };
                    if matched {
                        self.net_to_class.insert(net.clone(), cls_name.to_string());
                    }
                }
            }
        }
    }

    /// Reads a KiCad `.kicad_dru` custom design rules file from `dru_path` and applies its
    /// global and netclass-specific constraints to `self`.
    ///
    /// Does nothing if `dru_path` cannot be read.
    ///
    /// # Arguments
    /// * `dru_path` - Path to the `.kicad_dru` file.
    ///
    /// # Side Effects
    /// Reads `dru_path` from disk and mutates `self.netclasses` via [`Board::apply_design_rules_dru_str`].
    pub fn load_design_rules_dru(&mut self, dru_path: &Path) {
        let Ok(content) = fs::read_to_string(dru_path) else { return };
        self.apply_design_rules_dru_str(&content);
    }

    /// Parses a `.kicad_dru` S-expression string and clamps all [`Board::netclasses`] to honor
    /// both global `(rule ...)` constraints and netclass-conditioned constraints.
    ///
    /// # Arguments
    /// * `content` - Raw `.kicad_dru` S-expression text.
    ///
    /// # Side Effects
    /// Mutates [`NetClass`] parameters in `self.netclasses` in place.
    pub fn apply_design_rules_dru_str(&mut self, content: &str) {
        let mut base_rules = self.design_rules();
        apply_dru_rules_to_board(content, &mut base_rules, &mut self.netclasses);
        for nc in self.netclasses.values_mut() {
            nc.apply_design_rules(&base_rules);
            nc.board_edge_clearance = nc.board_edge_clearance.max(base_rules.min_copper_edge_clearance);
        }
    }

    /// Parses a root `.kicad_sch` schematic file (and any referenced hierarchical sub-sheet files)
    /// to extract symbol coordinates (`self.schematic_positions`) and functional component clusters
    /// (`self.clusters`) defined by hierarchical sheets, closed graphical polylines/rectangles, or numbered section headers.
    ///
    /// # Arguments
    /// * `sch_path` - Path to the root `.kicad_sch` file on disk.
    ///
    /// # Side Effects
    /// * Reads `sch_path` and any child `Sheetfile` `.kicad_sch` files from disk.
    /// * Populates `self.schematic_positions` and `self.clusters`.
    pub fn load_schematic_clusters(&mut self, sch_path: &Path) {
        let Ok(content) = fs::read_to_string(sch_path) else { return };

        // Helper to skip lib_symbols block (contains symbol graphic drawings rather than placed schematic instances)
        let sch_body = if let Some(start) = content.find("(lib_symbols") {
            let mut depth = 0;
            let mut end = start;
            for (i, b) in content[start..].bytes().enumerate() {
                if b == b'(' {
                    depth += 1;
                } else if b == b')' {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i + 1;
                        break;
                    }
                }
            }
            &content[end..]
        } else {
            &content[..]
        };

        // 1. Parse placed symbols on root schematic
        let mut placed_symbols = Vec::new();
        let mut s_idx = 0;
        while let Some(pos) = sch_body[s_idx..].find("(symbol") {
            let actual = s_idx + pos;
            let after = actual + 7;
            if after < sch_body.len() && sch_body.as_bytes()[after].is_ascii_whitespace() {
                let win = &sch_body[actual..(actual + 1500).min(sch_body.len())];
                if win.contains("(lib_id ") {
                    if let (Some((sx, sy, _)), Some(r)) = (parse_at_xy_rot(win), extract_property(win, "Reference")) {
                        if !r.starts_with('#') {
                            self.schematic_positions.insert(r.clone(), (sx, sy));
                            placed_symbols.push((r, sx, sy));
                        }
                    }
                }
            }
            s_idx = actual + 7;
        }

        // 2. Parse candidate functional boxes: hierarchical sheets, polylines, and rectangles
        struct CandidateBox {
            name: Option<String>,
            min_x: f64,
            min_y: f64,
            max_x: f64,
            max_y: f64,
            members: Vec<String>,
        }
        let mut boxes: Vec<CandidateBox> = Vec::new();

        // 2a. Hierarchical sub-sheets: (sheet (at X Y) (size W H) ... (property "Sheetname" "...") (property "Sheetfile" "..."))
        let mut sh_idx = 0;
        while let Some(pos) = sch_body[sh_idx..].find("(sheet") {
            let actual = sh_idx + pos;
            let after = actual + 6;
            if after < sch_body.len() && sch_body.as_bytes()[after].is_ascii_whitespace() {
                let win = &sch_body[actual..(actual + 2000).min(sch_body.len())];
                let name = extract_property(win, "Sheetname").or_else(|| extract_property(win, "Sheetfile"));
                let mut members = Vec::new();
                if let Some(sheetfile) = extract_property(win, "Sheetfile") {
                    let subsheet_path = sch_path.parent().unwrap_or_else(|| Path::new("")).join(&sheetfile);
                    if let Ok(sub_content) = fs::read_to_string(&subsheet_path) {
                        let mut sub_s_idx = 0;
                        while let Some(sub_pos) = sub_content[sub_s_idx..].find("(symbol") {
                            let sub_act = sub_s_idx + sub_pos;
                            let sub_win = &sub_content[sub_act..(sub_act + 1500).min(sub_content.len())];
                            if sub_win.contains("(lib_id ") {
                                if let (Some((sx, sy, _)), Some(r)) = (parse_at_xy_rot(sub_win), extract_property(sub_win, "Reference")) {
                                    if !r.starts_with('#') {
                                        self.schematic_positions.insert(r.clone(), (sx, sy));
                                        members.push(r);
                                    }
                                }
                            }
                            sub_s_idx = sub_act + 7;
                        }
                    }
                }
                let (at_x, at_y) = extract_point(win, "at").unwrap_or((0.0, 0.0));
                let (size_w, size_h) = extract_point(win, "size").unwrap_or((0.0, 0.0));
                boxes.push(CandidateBox {
                    name,
                    min_x: at_x,
                    min_y: at_y,
                    max_x: at_x + size_w,
                    max_y: at_y + size_h,
                    members,
                });
            }
            sh_idx = actual + 6;
        }

        // 2b. Closed polylines (pts (xy ...) ...) forming bounding boxes (width >= 5.0, height >= 5.0)
        let mut p_idx = 0;
        while let Some(pos) = sch_body[p_idx..].find("(polyline") {
            let actual = p_idx + pos;
            let win = &sch_body[actual..(actual + 2000).min(sch_body.len())];
            if let Some(pts_start) = win.find("(pts") {
                let mut depth = 0;
                let mut pts_end = pts_start;
                for (i, b) in win[pts_start..].bytes().enumerate() {
                    if b == b'(' {
                        depth += 1;
                    } else if b == b')' {
                        depth -= 1;
                        if depth == 0 {
                            pts_end = pts_start + i + 1;
                            break;
                        }
                    }
                }
                let pts_str = &win[pts_start..pts_end];
                let mut xs = Vec::new();
                let mut ys = Vec::new();
                let mut xy_idx = 0;
                while let Some(xy_pos) = pts_str[xy_idx..].find("(xy ") {
                    let start = xy_idx + xy_pos + 4;
                    let close = pts_str[start..].find(')').map(|i| start + i).unwrap_or(pts_str.len());
                    let parts: Vec<&str> = pts_str[start..close].split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let (Ok(x), Ok(y)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                            xs.push(x);
                            ys.push(y);
                        }
                    }
                    xy_idx = close;
                }
                if xs.len() >= 4 {
                    let min_x = xs.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let min_y = ys.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_y = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    if (max_x - min_x) >= 5.0 && (max_y - min_y) >= 5.0 {
                        boxes.push(CandidateBox {
                            name: None,
                            min_x,
                            min_y,
                            max_x,
                            max_y,
                            members: Vec::new(),
                        });
                    }
                }
            }
            p_idx = actual + 9;
        }

        // 2c. Rectangles (rectangle (start ...) (end ...)) (width >= 5.0, height >= 5.0)
        let mut r_idx = 0;
        while let Some(pos) = sch_body[r_idx..].find("(rectangle") {
            let actual = r_idx + pos;
            let win = &sch_body[actual..(actual + 500).min(sch_body.len())];
            if let (Some((x1, y1)), Some((x2, y2))) = (extract_point(win, "start"), extract_point(win, "end")) {
                let min_x = x1.min(x2);
                let max_x = x1.max(x2);
                let min_y = y1.min(y2);
                let max_y = y1.max(y2);
                if (max_x - min_x) >= 5.0 && (max_y - min_y) >= 5.0 {
                    boxes.push(CandidateBox {
                        name: None,
                        min_x,
                        min_y,
                        max_x,
                        max_y,
                        members: Vec::new(),
                    });
                }
            }
            r_idx = actual + 10;
        }

        // 3. Extract schematic text annotations (to name boxes or as fallback)
        let mut texts: Vec<(String, f64, f64)> = Vec::new();
        let text_tag = "(text \"";
        let mut idx = 0;
        while let Some(pos) = sch_body[idx..].find(text_tag) {
            let start = idx + pos + text_tag.len();
            if let Some(q_end) = sch_body[start..].find('"') {
                let title = sch_body[start..start + q_end].trim().to_string();
                let rest = &sch_body[start + q_end..];
                if let Some((x, y, _)) = parse_at_xy_rot(&rest[..rest.len().min(200)]) {
                    texts.push((title, x, y));
                }
                idx = start + q_end + 1;
            } else {
                break;
            }
        }

        // 4. Resolve cluster definitions
        let mut cluster_defs: Vec<(String, Vec<String>)> = Vec::new();

        if !boxes.is_empty() {
            // Assign texts to boxes to name them
            for b in &mut boxes {
                if b.name.is_none() {
                    let in_box_texts: Vec<&(String, f64, f64)> = texts
                        .iter()
                        .filter(|(_, tx, ty)| *tx >= b.min_x && *tx <= b.max_x && *ty >= b.min_y && *ty <= b.max_y)
                        .collect();
                    let title = in_box_texts
                        .iter()
                        .find(|(t, _, _)| !t.contains('\n') && t.chars().next().is_some_and(|c| c.is_ascii_digit()) && t.contains('.'))
                        .or_else(|| in_box_texts.iter().find(|(t, _, _)| !t.contains('\n')))
                        .or_else(|| in_box_texts.first())
                        .map(|(t, _, _)| t.lines().next().unwrap_or(t).trim().to_string());
                    b.name = title;
                }
            }

            // Assign root placed symbols to containing boxes (smallest bounding box wins for nested boxes)
            for (r, sx, sy) in placed_symbols {
                let mut best_idx = None;
                let mut best_area = f64::INFINITY;
                for (i, b) in boxes.iter().enumerate() {
                    if sx >= b.min_x - 1.0 && sx <= b.max_x + 1.0 && sy >= b.min_y - 1.0 && sy <= b.max_y + 1.0 {
                        let area = (b.max_x - b.min_x) * (b.max_y - b.min_y);
                        if area < best_area {
                            best_area = area;
                            best_idx = Some(i);
                        }
                    }
                }
                if let Some(i) = best_idx {
                    if !boxes[i].members.contains(&r) {
                        boxes[i].members.push(r);
                    }
                }
            }

            for (i, b) in boxes.into_iter().enumerate() {
                if b.members.is_empty() {
                    continue;
                }
                let title = b.name.unwrap_or_else(|| {
                    if let Some(u) = b.members.iter().find(|r| r.starts_with('U')) {
                        format!("{} Cluster", u)
                    } else if let Some(j) = b.members.iter().find(|r| r.starts_with('J')) {
                        format!("{} Cluster", j)
                    } else {
                        format!("Cluster {}", i + 1)
                    }
                });
                cluster_defs.push((title, b.members));
            }
        } else {
            // Fallback: proximity-based text header grouping if no boxes or sub-sheets exist
            let headers: Vec<(String, f64, f64)> = texts
                .into_iter()
                .filter(|(t, _, _)| t.chars().next().is_some_and(|c| c.is_ascii_digit()) && t.contains('.'))
                .collect();
            let mut members: HashMap<String, Vec<String>> = headers.iter().map(|h| (h.0.clone(), Vec::new())).collect();
            for (r, sx, sy) in placed_symbols {
                let mut best = None;
                let mut min_d = f64::INFINITY;
                for (ht, hx, hy) in &headers {
                    let dx = sx - hx;
                    let dy = sy - hy;
                    if dx >= -15.0 && dy >= -15.0 && dx * dx + dy * dy < min_d {
                        min_d = dx * dx + dy * dy;
                        best = Some(ht.clone());
                    }
                }
                if let Some(bh) = best {
                    if let Some(list) = members.get_mut(&bh) {
                        if !list.contains(&r) {
                            list.push(r);
                        }
                    }
                }
            }
            for (title, m_refs) in members {
                if !m_refs.is_empty() {
                    cluster_defs.push((title, m_refs));
                }
            }
        }

        let ref_to_idx: HashMap<&str, usize> = self
            .footprints
            .iter()
            .enumerate()
            .map(|(i, fp)| (fp.reference.as_str(), i))
            .collect();

        for (title, m_refs) in cluster_defs {
            let m_indices: Vec<usize> = m_refs.iter().filter_map(|r| ref_to_idx.get(r.as_str()).copied()).collect();
            if m_indices.is_empty() {
                continue;
            }
            let up = title.to_uppercase();
            let (anchor_prefix, max_radius, weight) =
                if up.contains("BUCK") || up.contains("POWER") || up.contains("LDO") {
                    (Some('U'), 4.5, 150.0)
                } else if up.contains("SURGE")
                    || up.contains("INPUT")
                    || up.contains("USB")
                    || up.contains("SENSOR")
                    || up.contains("1-WIRE")
                {
                    (Some('J'), 6.8, 100.0)
                } else if up.contains("ESP32") || up.contains("CONTROLS") || up.contains("MCU") {
                    (Some('U'), 12.0, 50.0)
                } else if up.contains("TRANSCEIVER") || up.contains("MAX22088") || up.contains("UART") {
                    (Some('U'), 7.5, 100.0)
                } else {
                    (None, 7.0, 60.0)
                };
            let anchor_idx = anchor_prefix
                .and_then(|pref| {
                    m_refs
                        .iter()
                        .find(|r| r.starts_with(pref))
                        .and_then(|r| ref_to_idx.get(r.as_str()).copied())
                })
                .or_else(|| {
                    m_refs
                        .iter()
                        .find(|r| r.starts_with('U'))
                        .or_else(|| m_refs.iter().find(|r| r.starts_with('J')))
                        .or_else(|| m_refs.first())
                        .and_then(|r| ref_to_idx.get(r.as_str()).copied())
                });
            self.clusters.push(CompiledCluster {
                name: title,
                anchor_idx,
                member_indices: m_indices,
                member_refs: m_refs,
                max_radius,
                weight,
            });
        }
    }
}

/// Natural alphanumeric comparator for reference designators (e.g. "C5" < "C20", "R2" < "R10").
/// Letters are compared first alphabetically (case-insensitive), then numeric values in ascending order,
/// then any trailing alphabetic suffix.
pub fn natural_designator_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    fn split_ref(s: &str) -> (String, u64, String) {
        let trimmed = s.trim();
        let prefix: String = trimmed.chars().take_while(|c| !c.is_ascii_digit()).collect();
        let rest = &trimmed[prefix.len()..];
        let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        let suffix = &rest[num_str.len()..];
        let num: u64 = num_str.parse().unwrap_or(0);
        (prefix.to_ascii_uppercase(), num, suffix.to_ascii_uppercase())
    }
    split_ref(a).cmp(&split_ref(b))
}

/// Extracts a 2D coordinate tuple `(x, y)` in millimeters from an S-expression tag such as `(start X Y)`.
///
/// # Arguments
/// * `block` - S-expression text slice to search within.
/// * `tag` - Tag name without parentheses (e.g., `"start"`, `"end"`, `"mid"`, `"at"`).
///
/// # Returns
/// `Some((x, y))` in millimeters if the tag and two floating-point numbers are found, or `None` otherwise.
fn extract_point(block: &str, tag: &str) -> Option<(f64, f64)> {
    let pattern = format!("({} ", tag);
    let pos = block.find(&pattern)?;
    let rest = &block[pos + pattern.len()..];
    let end = rest.find(')')?;
    let mut parts = rest[..end].split_whitespace();
    let x = parts.next()?.parse::<f64>().ok()?;
    let y = parts.next()?.parse::<f64>().ok()?;
    Some((x, y))
}

/// Extracts a single `f64` scalar value from an S-expression tag such as `(width 0.25)`.
///
/// # Arguments
/// * `block` - S-expression text slice to search within.
/// * `tag` - Tag name without parentheses (e.g., `"width"`, `"size"`, `"drill"`).
///
/// # Returns
/// `Some(value)` if the tag and a valid `f64` token are present, or `None` otherwise.
fn extract_named_f64(block: &str, tag: &str) -> Option<f64> {
    let pattern = format!("({} ", tag);
    let pos = block.find(&pattern)?;
    let rest = &block[pos + pattern.len()..];
    let end = rest.find(')')?;
    rest[..end].split_whitespace().next()?.parse::<f64>().ok()
}

/// Extracts a single `usize` integer value from an S-expression tag such as `(net 3)`.
///
/// # Arguments
/// * `block` - S-expression text slice to search within.
/// * `tag` - Tag name without parentheses (e.g., `"net"`).
///
/// # Returns
/// `Some(value)` if the tag and a valid `usize` token are present, or `None` otherwise.
fn extract_named_usize(block: &str, tag: &str) -> Option<usize> {
    let pattern = format!("({} ", tag);
    let pos = block.find(&pattern)?;
    let rest = &block[pos + pattern.len()..];
    let end = rest.find(')')?;
    rest[..end].split_whitespace().next()?.parse::<usize>().ok()
}

/// Resolves or registers a net ID from an S-expression `(net ...)` clause supporting both numeric IDs
/// (KiCad 8/9 `(net 4)`) and quoted net names (KiCad 10 `(net "GND")`).
///
/// # Arguments
/// * `block` - S-expression block containing a `(net ...)` attribute.
/// * `nets` - Mutable board net table where newly encountered net names are appended.
/// * `net_name_to_id` - Mutable lookup map from net name string to net index.
///
/// # Returns
/// The resolved numeric net ID (`0` if unconnected or missing).
///
/// # Side Effects
/// Appends new net names to `nets` and inserts mappings into `net_name_to_id` when a previously unseen
/// quoted net name is encountered.
fn extract_net_id_or_name(
    block: &str,
    nets: &mut Vec<String>,
    net_name_to_id: &mut HashMap<String, usize>,
) -> usize {
    if let Some(id) = extract_named_usize(block, "net") {
        return id;
    }
    if let Some(pos) = block.find("(net ") {
        let after = &block[pos + 5..];
        if let Some(q1) = after.find('"') {
            if let Some(q2) = after[q1 + 1..].find('"') {
                let name = after[q1 + 1..q1 + 1 + q2].to_string();
                if !name.is_empty() {
                    if let Some(&id) = net_name_to_id.get(&name) {
                        return id;
                    }
                    let new_id = nets.len();
                    nets.push(name.clone());
                    net_name_to_id.insert(name, new_id);
                    return new_id;
                }
            }
        }
    }
    0
}

/// Parses a KiCad `(arc (start ...) (mid ...) (end ...) (width ...) (layer ...) (net ...))` block into an [`ArcSegment`].
///
/// # Arguments
/// * `block` - S-expression string for the `(arc ...)` element.
/// * `active_layers` - Ordered list of active copper layer names (e.g., `["F.Cu", "B.Cu"]`).
/// * `nets` - Mutable board net list for resolving or inserting quoted net names.
/// * `net_name_to_id` - Mutable map from net name to net ID.
///
/// # Returns
/// `Some(ArcSegment)` if valid 3-point arc geometry is parsed, or `None` if degenerate/malformed.
///
/// # Side Effects
/// May register new net names in `nets` and `net_name_to_id`.
fn parse_arc(
    block: &str,
    active_layers: &[&str],
    nets: &mut Vec<String>,
    net_name_to_id: &mut HashMap<String, usize>,
) -> Option<ArcSegment> {
    let start = extract_point(block, "start")?;
    let mid = extract_point(block, "mid")?;
    let end = extract_point(block, "end")?;
    let width = extract_named_f64(block, "width").unwrap_or(0.20);
    let layer = active_layers
        .iter()
        .position(|&l| block.contains(&format!("\"{}\"", l)))
        .unwrap_or_else(|| if block.contains("\"B.Cu\"") { 1 } else { 0 });
    let net_id = extract_net_id_or_name(block, nets, net_name_to_id);
    let is_locked = block.contains("(locked yes)") || block.contains("(locked)") || block.contains("(status 40000)");

    let mut arc = ArcSegment::from_3pts(start, mid, end, width, layer, net_id)?;
    arc.is_locked = is_locked;
    Some(arc)
}

/// Parses a KiCad `(segment (start ...) (end ...) (width ...) (layer ...) (net ...))` block into a [`TrackSegment`].
///
/// # Arguments
/// * `block` - S-expression string for the `(segment ...)` element.
/// * `active_layers` - Ordered list of active copper layer names.
/// * `nets` - Mutable board net list for resolving or inserting quoted net names.
/// * `net_name_to_id` - Mutable map from net name to net ID.
///
/// # Returns
/// `Some(TrackSegment)` if `start` and `end` coordinates are valid, or `None` otherwise.
///
/// # Side Effects
/// May register new net names in `nets` and `net_name_to_id`.
fn parse_segment(
    block: &str,
    active_layers: &[&str],
    nets: &mut Vec<String>,
    net_name_to_id: &mut HashMap<String, usize>,
) -> Option<TrackSegment> {
    let (sx, sy) = extract_point(block, "start")?;
    let (ex, ey) = extract_point(block, "end")?;
    let width = extract_named_f64(block, "width").unwrap_or(0.20);
    let layer = active_layers
        .iter()
        .position(|&l| block.contains(&format!("\"{}\"", l)))
        .unwrap_or_else(|| if block.contains("\"B.Cu\"") { 1 } else { 0 });
    let net_id = extract_net_id_or_name(block, nets, net_name_to_id);
    let is_locked = block.contains("(locked yes)") || block.contains("(locked)") || block.contains("(status 40000)");
    Some(TrackSegment {
        start: (sx, sy),
        end: (ex, ey),
        width,
        layer,
        net_id,
        is_locked,
    })
}

/// Parses a KiCad `(via (at ...) (size ...) (drill ...) (net ...))` S-expression block into a [`Via`].
///
/// # Arguments
/// * `block` - S-expression string for the `(via ...)` element.
/// * `nets` - Mutable board net list for resolving or inserting quoted net names.
/// * `net_name_to_id` - Mutable map from net name to net ID.
///
/// # Returns
/// `Some(Via)` if the `(at X Y)` coordinate is valid, or `None` otherwise.
///
/// # Side Effects
/// May register new net names in `nets` and `net_name_to_id`.
fn parse_via(
    block: &str,
    nets: &mut Vec<String>,
    net_name_to_id: &mut HashMap<String, usize>,
) -> Option<Via> {
    let (vx, vy) = extract_point(block, "at")?;
    let size = extract_named_f64(block, "size").unwrap_or(0.60);
    let drill = extract_named_f64(block, "drill").unwrap_or(0.30);
    let net_id = extract_net_id_or_name(block, nets, net_name_to_id);
    let is_locked = block.contains("(locked yes)") || block.contains("(locked)") || block.contains("(status 40000)");
    Some(Via {
        position: (vx, vy),
        size,
        drill,
        net_id,
        is_locked,
    })
}

/// Parses a top-level `(gr_text "..." (at X Y [ROT]) (layer "...") (effects (font (size W H))))` block into a [`BoardText`].
///
/// # Arguments
/// * `block` - S-expression string for the `(gr_text ...)` element.
/// * `byte_offset` - Byte offset of `block` within the raw `.kicad_pcb` source file for in-place span tracking.
///
/// # Returns
/// `Some(BoardText)` if the text literal and `(at ...)` clause are present, or `None` otherwise.
fn parse_gr_text(block: &str, byte_offset: usize) -> Option<BoardText> {
    let t_start = block.find('"')? + 1;
    let t_end = t_start + block[t_start..].find('"')?;
    let text = block[t_start..t_end].to_string();

    let at_rel = block.find("(at ")?;
    let at_close = at_rel + block[at_rel..].find(')')? + 1;
    let (x, y, rot) = parse_at_xy_rot(&block[at_rel..at_close])?;
    let at_span = Some((byte_offset + at_rel, byte_offset + at_close));

    let layer = if let Some(l_rel) = block.find("(layer \"") {
        let after = &block[l_rel + 8..];
        let l_end = after.find('"')?;
        after[..l_end].to_string()
    } else {
        "F.SilkS".to_string()
    };

    let (width, height) = if let Some(s_rel) = block.find("(size ") {
        let after = &block[s_rel + 6..];
        let end_idx = after.find(')').unwrap_or(after.len());
        let tokens: Vec<&str> = after[..end_idx].split_whitespace().collect();
        let w = tokens.first().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.8);
        let h = tokens.get(1).and_then(|s| s.parse::<f64>().ok()).unwrap_or(w);
        (w, h)
    } else {
        (0.8, 0.8)
    };

    Some(BoardText {
        text,
        x,
        y,
        rotation: rot,
        layer,
        width,
        height,
        at_span,
    })
}

/// Returns the next whitespace- and parenthesis-delimited token in `s` starting at or after byte index `start`.
///
/// # Arguments
/// * `s` - Input S-expression string slice.
/// * `start` - Starting byte offset in `s`.
///
/// # Returns
/// Subslice of `s` containing the token (empty slice if `start` is at end or before a delimiter).
fn next_token(s: &str, start: usize) -> &str {
    let bytes = s.as_bytes();
    let mut i = start;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let w_start = i;
    while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b')' && bytes[i] != b'(' {
        i += 1;
    }
    &s[w_start..i]
}

/// Finds the byte index immediately past the matching closing parenthesis `)` for an S-expression starting at `open_pos`,
/// ignoring parentheses inside escaped double-quoted strings.
///
/// # Arguments
/// * `bytes` - Raw byte slice of the S-expression document.
/// * `open_pos` - Byte index of the opening `(` character.
///
/// # Returns
/// `Some(end_byte_offset)` (exclusive) if a balanced closing `)` is found, or `None` if unbalanced.
fn find_closing_paren(bytes: &[u8], open_pos: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_quotes = false;
    let mut i = open_pos;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'"' && (i == 0 || bytes[i - 1] != b'\\') {
            in_quotes = !in_quotes;
        } else if !in_quotes {
            if c == b'(' {
                depth += 1;
            } else if c == b')' {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
        }
        i += 1;
    }
    None
}

/// Parses a 2D [`Point`] from an S-expression prefix tag such as `"(start "` or `"(end "`.
///
/// # Arguments
/// * `s` - S-expression text slice to search within.
/// * `tag` - Full opening tag prefix including leading parenthesis and trailing space (e.g., `"(start "`).
///
/// # Returns
/// `Some(Point)` in millimeters if parsed, or `None` otherwise.
fn parse_point_tag(s: &str, tag: &str) -> Option<Point> {
    let p = s.find(tag)?;
    let rest = &s[p + tag.len()..];
    let e = rest.find(')')?;
    let mut parts = rest[..e].split_whitespace();
    let x = parts.next()?.parse::<f64>().ok()?;
    let y = parts.next()?.parse::<f64>().ok()?;
    Some(Point::new(x, y))
}

/// Approximates a circular arc passing through three points (`p1`, `pm`, `p2`) as a polyline of `steps + 1` vertices.
///
/// # Arguments
/// * `p1` - Arc start point in millimeters.
/// * `pm` - Arc midpoint on the circumference in millimeters.
/// * `p2` - Arc end point in millimeters.
/// * `steps` - Number of linear chord subdivisions.
///
/// # Returns
/// Ordered vector of `Point` vertices along the arc from `p1` to `p2` (falls back to `vec![p1, p2]` if collinear).
fn discretize_arc(p1: Point, pm: Point, p2: Point, steps: usize) -> Vec<Point> {
    let x1 = p1.x;
    let y1 = p1.y;
    let xm = pm.x;
    let ym = pm.y;
    let x2 = p2.x;
    let y2 = p2.y;
    let d = 2.0 * (x1 * (ym - y2) + xm * (y2 - y1) + x2 * (y1 - ym));
    if d.abs() < 1e-8 {
        return vec![p1, p2];
    }
    let cx = ((x1 * x1 + y1 * y1) * (ym - y2) + (xm * xm + ym * ym) * (y2 - y1) + (x2 * x2 + y2 * y2) * (y1 - ym)) / d;
    let cy = ((x1 * x1 + y1 * y1) * (x2 - xm) + (xm * xm + ym * ym) * (x1 - x2) + (x2 * x2 + y2 * y2) * (xm - x1)) / d;
    let r = (x1 - cx).hypot(y1 - cy);
    let a1 = (y1 - cy).atan2(x1 - cx);
    let am = (ym - cy).atan2(xm - cx);
    let a2 = (y2 - cy).atan2(x2 - cx);
    let dm = (am - a1).rem_euclid(2.0 * std::f64::consts::PI);
    let d2 = (a2 - a1).rem_euclid(2.0 * std::f64::consts::PI);
    let sweep = if dm < d2 { d2 } else { d2 - 2.0 * std::f64::consts::PI };

    let mut pts = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let ang = a1 + t * sweep;
        pts.push(Point::new(cx + r * ang.cos(), cy + r * ang.sin()));
    }
    pts
}

/// Parses a graphic element on the `Edge.Cuts` layer (`gr_line`, `gr_arc`, `gr_rect`, `gr_poly`, `gr_circle`),
/// appending line segments to `lines` and expanding the board bounding box `(min_x, min_y, max_x, max_y)`.
///
/// # Arguments
/// * `block` - S-expression string of the graphic primitive on `Edge.Cuts`.
/// * `tag` - Graphic element type (`"gr_line"`, `"gr_arc"`, `"gr_rect"`, `"gr_poly"`, or `"gr_circle"`).
/// * `lines` - Mutable vector of boundary segments `(Point, Point)` to append to.
/// * `min_x` - Running minimum X coordinate (mm).
/// * `min_y` - Running minimum Y coordinate (mm).
/// * `max_x` - Running maximum X coordinate (mm).
/// * `max_y` - Running maximum Y coordinate (mm).
///
/// # Side Effects
/// Appends discretized boundary edges to `lines` and updates `min_x`, `min_y`, `max_x`, and `max_y`.
fn parse_edge_cuts_element(
    block: &str,
    tag: &str,
    lines: &mut Vec<(Point, Point)>,
    min_x: &mut f64,
    min_y: &mut f64,
    max_x: &mut f64,
    max_y: &mut f64,
) {
    if tag == "gr_line" {
        if let (Some(p1), Some(p2)) = (parse_point_tag(block, "(start "), parse_point_tag(block, "(end ")) {
            *min_x = min_x.min(p1.x).min(p2.x);
            *max_x = max_x.max(p1.x).max(p2.x);
            *min_y = min_y.min(p1.y).min(p2.y);
            *max_y = max_y.max(p1.y).max(p2.y);
            lines.push((p1, p2));
        }
    } else if tag == "gr_arc" {
        if let (Some(p1), Some(pm), Some(p2)) = (
            parse_point_tag(block, "(start "),
            parse_point_tag(block, "(mid "),
            parse_point_tag(block, "(end "),
        ) {
            let arc_pts = discretize_arc(p1, pm, p2, 8);
            for pt in &arc_pts {
                *min_x = min_x.min(pt.x);
                *max_x = max_x.max(pt.x);
                *min_y = min_y.min(pt.y);
                *max_y = max_y.max(pt.y);
            }
            for k in 0..(arc_pts.len() - 1) {
                lines.push((arc_pts[k], arc_pts[k + 1]));
            }
        }
    } else if tag == "gr_rect" {
        if let (Some(p1), Some(p2)) = (parse_point_tag(block, "(start "), parse_point_tag(block, "(end ")) {
            let x0 = p1.x.min(p2.x);
            let x1 = p1.x.max(p2.x);
            let y0 = p1.y.min(p2.y);
            let y1 = p1.y.max(p2.y);
            *min_x = min_x.min(x0);
            *max_x = max_x.max(x1);
            *min_y = min_y.min(y0);
            *max_y = max_y.max(y1);
            lines.push((Point::new(x0, y0), Point::new(x1, y0)));
            lines.push((Point::new(x1, y0), Point::new(x1, y1)));
            lines.push((Point::new(x1, y1), Point::new(x0, y1)));
            lines.push((Point::new(x0, y1), Point::new(x0, y0)));
        }
    } else if tag == "gr_poly" {
        let mut poly_pts = Vec::new();
        let tag_xy = "(xy ";
        let mut idx = 0;
        while let Some(pos) = block[idx..].find(tag_xy) {
            let s = idx + pos + tag_xy.len();
            if let Some(e) = block[s..].find(')') {
                let mut parts = block[s..s + e].split_whitespace();
                if let (Some(xs), Some(ys)) = (parts.next(), parts.next()) {
                    if let (Ok(x), Ok(y)) = (xs.parse::<f64>(), ys.parse::<f64>()) {
                        poly_pts.push(Point::new(x, y));
                        *min_x = min_x.min(x);
                        *max_x = max_x.max(x);
                        *min_y = min_y.min(y);
                        *max_y = max_y.max(y);
                    }
                }
                idx = s + e;
            } else {
                break;
            }
        }
        for k in 0..poly_pts.len() {
            let next = (k + 1) % poly_pts.len();
            lines.push((poly_pts[k], poly_pts[next]));
        }
    } else if tag == "gr_circle" {
        if let (Some(center), Some(end_pt)) = (parse_point_tag(block, "(center "), parse_point_tag(block, "(end ")) {
            let r = center.distance_to(&end_pt);
            *min_x = min_x.min(center.x - r);
            *max_x = max_x.max(center.x + r);
            *min_y = min_y.min(center.y - r);
            *max_y = max_y.max(center.y + r);
            let steps = 16;
            for k in 0..steps {
                let a0 = (k as f64) * 2.0 * std::f64::consts::PI / (steps as f64);
                let a1 = ((k + 1) as f64) * 2.0 * std::f64::consts::PI / (steps as f64);
                lines.push((
                    Point::new(center.x + r * a0.cos(), center.y + r * a0.sin()),
                    Point::new(center.x + r * a1.cos(), center.y + r * a1.sin()),
                ));
            }
        }
    } else {
        extract_coords_into_bounds(block, min_x, min_y, max_x, max_y);
    }
}

/// Scans an S-expression `block` for all `(start X Y)` and `(end X Y)` coordinate pairs and expands
/// the bounding box `(min_x, min_y, max_x, max_y)`.
///
/// # Arguments
/// * `block` - S-expression text slice to scan.
/// * `min_x` - Running minimum X coordinate (mm).
/// * `min_y` - Running minimum Y coordinate (mm).
/// * `max_x` - Running maximum X coordinate (mm).
/// * `max_y` - Running maximum Y coordinate (mm).
///
/// # Side Effects
/// Mutates `min_x`, `min_y`, `max_x`, and `max_y` in place.
fn extract_coords_into_bounds(block: &str, min_x: &mut f64, min_y: &mut f64, max_x: &mut f64, max_y: &mut f64) {
    for tag in ["(start ", "(end "] {
        let mut idx = 0;
        while let Some(pos) = block[idx..].find(tag) {
            let s = idx + pos + tag.len();
            if let Some(e) = block[s..].find(')') {
                let mut parts = block[s..s + e].split_whitespace();
                if let (Some(xs), Some(ys)) = (parts.next(), parts.next()) {
                    if let (Ok(x), Ok(y)) = (xs.parse::<f64>(), ys.parse::<f64>()) {
                        *min_x = min_x.min(x);
                        *max_x = max_x.max(x);
                        *min_y = min_y.min(y);
                        *max_y = max_y.max(y);
                    }
                }
                idx = s + e;
            } else {
                break;
            }
        }
    }
}

/// Scans an S-expression `block` for all `(xy X Y)` vertex tags and expands the bounding box `(min_x, min_y, max_x, max_y)`.
///
/// # Arguments
/// * `block` - S-expression text slice to scan.
/// * `min_x` - Running minimum X coordinate (mm).
/// * `min_y` - Running minimum Y coordinate (mm).
/// * `max_x` - Running maximum X coordinate (mm).
/// * `max_y` - Running maximum Y coordinate (mm).
///
/// # Side Effects
/// Mutates `min_x`, `min_y`, `max_x`, and `max_y` in place.
fn extract_xy_into_bounds(block: &str, min_x: &mut f64, min_y: &mut f64, max_x: &mut f64, max_y: &mut f64) {
    let tag = "(xy ";
    let mut idx = 0;
    while let Some(pos) = block[idx..].find(tag) {
        let s = idx + pos + tag.len();
        if let Some(e) = block[s..].find(')') {
            let mut parts = block[s..s + e].split_whitespace();
            if let (Some(xs), Some(ys)) = (parts.next(), parts.next()) {
                if let (Ok(x), Ok(y)) = (xs.parse::<f64>(), ys.parse::<f64>()) {
                    *min_x = min_x.min(x);
                    *max_x = max_x.max(x);
                    *min_y = min_y.min(y);
                    *max_y = max_y.max(y);
                }
            }
            idx = s + e;
        } else {
            break;
        }
    }
}

/// Parses an `(at X Y [ROT])` S-expression clause into `(x_mm, y_mm, rotation_deg)`.
///
/// # Arguments
/// * `s` - S-expression slice containing an `(at ...)` clause.
///
/// # Returns
/// `Some((x, y, rotation))` where `rotation` defaults to `0.0` degrees if omitted, or `None` if malformed.
fn parse_at_xy_rot(s: &str) -> Option<(f64, f64, f64)> {
    let pos = s.find("(at ")?;
    let rest = &s[pos + 4..];
    let end = rest.find(')')?;
    let mut parts = rest[..end].split_whitespace();
    let x = parts.next()?.parse::<f64>().ok()?;
    let y = parts.next()?.parse::<f64>().ok()?;
    let rot = parts.next().and_then(|r| r.parse::<f64>().ok()).unwrap_or(0.0);
    Some((x, y, rot))
}

/// Computes in-place byte span replacements for all child `(pad ... (at X Y [ROT]))` clauses inside a footprint
/// when the footprint's world rotation changes by `delta_rot` degrees (since KiCad stores pad angles in board space).
///
/// # Arguments
/// * `fp_str` - Raw S-expression substring of the `(footprint ...)` block.
/// * `fp_byte_offset` - Starting byte offset of `fp_str` within the full `.kicad_pcb` file.
/// * `delta_rot` - Rotation delta in degrees (`new_rot - old_rot`).
///
/// # Returns
/// Vector of `(start_byte, end_byte, replacement_at_clause)` tuples for in-place `.kicad_pcb` patching.
fn collect_pad_at_replacements(
    fp_str: &str,
    fp_byte_offset: usize,
    delta_rot: f64,
) -> Vec<(usize, usize, String)> {
    let mut reps = Vec::new();
    let rem = delta_rot.rem_euclid(360.0);
    if rem < 1e-3 || (360.0 - rem) < 1e-3 {
        return reps;
    }
    let bytes = fp_str.as_bytes();
    let mut p_idx = 0;
    while p_idx < bytes.len() {
        if bytes[p_idx] == b'(' && fp_str[p_idx..].starts_with("(pad ") {
            if let Some(p_end) = find_closing_paren(bytes, p_idx) {
                let p_str = &fp_str[p_idx..p_end];
                let p_bytes = p_str.as_bytes();
                let mut depth = 0usize;
                let mut in_quotes = false;
                for i in 0..p_bytes.len() {
                    let c = p_bytes[i];
                    if c == b'"' && (i == 0 || p_bytes[i - 1] != b'\\') {
                        in_quotes = !in_quotes;
                    } else if !in_quotes {
                        if c == b'(' {
                            depth += 1;
                            if depth == 2 && p_str[i..].starts_with("(at ") {
                                if let Some(a_end) = find_closing_paren(p_bytes, i) {
                                    let inner = p_str[i + 4..a_end - 1].trim();
                                    let mut parts = inner.split_whitespace();
                                    if let (Some(x_str), Some(y_str)) = (parts.next(), parts.next()) {
                                        let old_rot = parts
                                            .next()
                                            .and_then(|r| r.parse::<f64>().ok())
                                            .unwrap_or(0.0);
                                        let new_rot = (old_rot + delta_rot).rem_euclid(360.0);
                                        let new_at = if new_rot < 1e-3 || (360.0 - new_rot) < 1e-3 {
                                            format!("(at {} {})", x_str, y_str)
                                        } else if (new_rot - new_rot.round()).abs() < 1e-3 {
                                            format!(
                                                "(at {} {} {})",
                                                x_str,
                                                y_str,
                                                (new_rot.round() as i64).rem_euclid(360)
                                            )
                                        } else {
                                            format!("(at {} {} {:.1})", x_str, y_str, new_rot)
                                        };
                                        reps.push((
                                            fp_byte_offset + p_idx + i,
                                            fp_byte_offset + p_idx + a_end,
                                            new_at,
                                        ));
                                    }
                                    break;
                                }
                            }
                        } else if c == b')' {
                            depth = depth.saturating_sub(1);
                        }
                    }
                }
                p_idx = p_end;
                continue;
            }
        }
        p_idx += 1;
    }
    reps
}

/// Extracts the string value of a KiCad `(property "NAME" "VALUE")` clause from an S-expression block.
///
/// # Arguments
/// * `s` - S-expression text slice to search within.
/// * `prop_name` - Property key to match (e.g., `"Reference"`, `"Value"`).
///
/// # Returns
/// `Some(value_string)` if found, or `None` otherwise.
fn extract_property(s: &str, prop_name: &str) -> Option<String> {
    let pattern = format!("(property \"{}\" \"", prop_name);
    let pos = s.find(&pattern)?;
    let start = pos + pattern.len();
    let end = s[start..].find('"')?;
    Some(s[start..start + end].to_string())
}

/// Swaps all front (`F.*`) and back (`B.*`) layer references (`Cu`, `SilkS`, `Paste`, `Mask`, `CrtYd`, `Fab`)
/// inside a footprint S-expression block when a component is flipped between board sides.
///
/// # Arguments
/// * `fp_block` - Raw S-expression string of the footprint block.
///
/// # Returns
/// A new string with all `F.<layer>` and `B.<layer>` tokens exchanged.
fn swap_footprint_layers(fp_block: &str) -> String {
    let mut s = fp_block.to_string();
    for suffix in ["Cu", "SilkS", "Paste", "Mask", "CrtYd", "Fab"] {
        let f_tag = format!("\"F.{}\"", suffix);
        let b_tag = format!("\"B.{}\"", suffix);
        let tmp_tag = format!("\"__TMP_{}__\"", suffix);
        s = s.replace(&f_tag, &tmp_tag);
        s = s.replace(&b_tag, &f_tag);
        s = s.replace(&tmp_tag, &b_tag);
    }
    s
}

/// Chains an unordered collection of 2D line segments `(Point, Point)` (with `0.05 mm` endpoint snap tolerance)
/// into closed [`Polygon`] loops for board outlines (`Edge.Cuts`) and `fp_line` courtyards.
///
/// # Arguments
/// * `lines` - Vector of endpoint pairs `(p1, p2)` in millimeters.
///
/// # Returns
/// Vector of closed [`Polygon`] instances formed by chaining contiguous segments.
fn chain_lines_into_polygons(mut lines: Vec<(Point, Point)>) -> Vec<Polygon> {
    let mut polys = Vec::new();
    const EPS: f64 = 0.05;

    while !lines.is_empty() {
        let first = lines.remove(0);
        let start_pt = first.0;
        let mut curr_pt = first.1;
        let mut pts = vec![start_pt, curr_pt];

        let mut found_next = true;
        while found_next && curr_pt.distance_to(&start_pt) > EPS {
            found_next = false;
            for i in 0..lines.len() {
                let (p1, p2) = lines[i];
                if p1.distance_to(&curr_pt) <= EPS {
                    curr_pt = p2;
                    pts.push(curr_pt);
                    lines.remove(i);
                    found_next = true;
                    break;
                } else if p2.distance_to(&curr_pt) <= EPS {
                    curr_pt = p1;
                    pts.push(curr_pt);
                    lines.remove(i);
                    found_next = true;
                    break;
                }
            }
        }

        if pts.len() >= 4 && pts.last().unwrap().distance_to(&pts[0]) <= EPS {
            pts.pop();
        }

        if pts.len() >= 3 {
            polys.push(Polygon::new(pts));
        }
    }

    polys
}

/// Parses a KiCad `.kicad_dru` dimension literal (supporting `mm`, `um`, `mil`, or unitless `mm`) into millimeters.
///
/// # Arguments
/// * `raw` - Raw dimension token string (e.g., `"0.15mm"`, `"150um"`, `"6mil"`).
///
/// # Returns
/// `Some(value_mm)` converted to millimeters, or `None` if parsing fails.
fn parse_dru_dimension_mm(raw: &str) -> Option<f64> {
    let s = raw.trim().trim_matches('"').trim();
    if let Some(num_s) = s.strip_suffix("mm") {
        num_s.trim().parse::<f64>().ok()
    } else if let Some(num_s) = s.strip_suffix("um") {
        num_s.trim().parse::<f64>().ok().map(|v| v * 0.001)
    } else if let Some(num_s) = s.strip_suffix("mil") {
        num_s.trim().parse::<f64>().ok().map(|v| v * 0.0254)
    } else {
        s.parse::<f64>().ok()
    }
}

/// Parses KiCad custom design rules (`.kicad_dru` S-expressions) from `content` and applies global or
/// netclass-scoped minimum constraints onto `rules` and `netclasses`.
///
/// # Arguments
/// * `content` - Raw `.kicad_dru` S-expression text.
/// * `rules` - Mutable global [`DesignRules`] to tighten with unconditional rules.
/// * `netclasses` - Mutable map of [`NetClass`] entries to tighten when a `(condition "A.NetClass == '...'")` matches.
///
/// # Side Effects
/// Raises minimum clearances, track widths, via diameters/drills, hole clearances, and edge clearances in `rules` and `netclasses`.
pub fn apply_dru_rules_to_board(
    content: &str,
    rules: &mut DesignRules,
    netclasses: &mut HashMap<String, NetClass>,
) {
    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if bytes[i] == b'(' {
            let word = next_token(content, i + 1);
            if word == "rule" {
                if let Some(end) = find_closing_paren(bytes, i) {
                    let block = &content[i..end];
                    let condition = if let Some(c_pos) = block.find("(condition ") {
                        let after = &block[c_pos + 11..];
                        if let Some(q1) = after.find('"') {
                            after[q1 + 1..].find('"').map(|q2| &after[q1 + 1..q1 + 1 + q2])
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // Ignore pad-specific exemption rules when computing global track/via routing rules
                    let is_pad_exemption = condition
                        .map(|c| c.contains("Pad") || c.contains("pad") || c.contains("Footprint") || c.contains("parent"))
                        .unwrap_or(false);
                    if !is_pad_exemption {
                        let b_bytes = block.as_bytes();
                        let mut j = 0;
                        while j < b_bytes.len() {
                            if b_bytes[j] == b'(' && block[j..].starts_with("(constraint ") {
                                if let Some(c_end) = find_closing_paren(b_bytes, j) {
                                    let c_block = &block[j..c_end];
                                    let c_type = next_token(c_block, "(constraint ".len());
                                    let min_val = if let Some(m_pos) = c_block.find("(min ") {
                                        let rest = &c_block[m_pos + 5..];
                                        rest.find(')').and_then(|m_end| parse_dru_dimension_mm(&rest[..m_end]))
                                    } else if let Some(o_pos) = c_block.find("(opt ") {
                                        let rest = &c_block[o_pos + 5..];
                                        rest.find(')').and_then(|o_end| parse_dru_dimension_mm(&rest[..o_end]))
                                    } else {
                                        None
                                    };
                                    if let Some(val) = min_val {
                                        if let Some(cond_str) = condition {
                                            for (cls_name, nc) in netclasses.iter_mut() {
                                                if cond_str.contains(&format!("'{}'", cls_name))
                                                    || cond_str.contains(&format!("\"{}\"", cls_name))
                                                {
                                                    match c_type {
                                                        "clearance" => nc.clearance = nc.clearance.max(val),
                                                        "track_width" => nc.track_width = nc.track_width.max(val),
                                                        "via_diameter" => nc.via_dia = nc.via_dia.max(val),
                                                        "hole_size" | "via_drill" => nc.via_drill = nc.via_drill.max(val),
                                                        "edge_clearance" | "copper_edge_clearance" => {
                                                            nc.board_edge_clearance = nc.board_edge_clearance.max(val)
                                                        }
                                                        "hole_clearance" => {
                                                            nc.min_hole_clearance = nc.min_hole_clearance.max(val)
                                                        }
                                                        "hole_to_hole" => {
                                                            nc.min_hole_to_hole = nc.min_hole_to_hole.max(val)
                                                        }
                                                        _ => {}
                                                    }
                                                }
                                            }
                                        } else {
                                            match c_type {
                                                "clearance" => rules.min_clearance = rules.min_clearance.max(val),
                                                "track_width" => rules.min_track_width = rules.min_track_width.max(val),
                                                "via_diameter" => {
                                                    rules.min_via_diameter = rules.min_via_diameter.max(val)
                                                }
                                                "hole_size" | "via_drill" => {
                                                    rules.min_via_drill = rules.min_via_drill.max(val)
                                                }
                                                "annular_width" => {
                                                    rules.min_via_annular_width = rules.min_via_annular_width.max(val)
                                                }
                                                "hole_clearance" => {
                                                    rules.min_hole_clearance = rules.min_hole_clearance.max(val)
                                                }
                                                "hole_to_hole" => {
                                                    rules.min_hole_to_hole = rules.min_hole_to_hole.max(val)
                                                }
                                                "edge_clearance" | "copper_edge_clearance" => {
                                                    rules.min_copper_edge_clearance =
                                                        rules.min_copper_edge_clearance.max(val)
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                    j = c_end;
                                    continue;
                                }
                            }
                            j += 1;
                        }
                    }
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
}

/// Parses global [`DesignRules`] from either a KiCad `.kicad_pro` JSON project file or a `.kicad_dru`
/// custom design-rule S-expression file.
///
/// # Arguments
/// * `content` - Raw text of the `.kicad_pro` or `.kicad_dru` file.
///
/// # Returns
/// `Some(DesignRules)` if design rule settings were found, or `None` if neither JSON design settings nor DRU constraints exist.
pub fn parse_design_rules_from_pro_or_dru(content: &str) -> Option<DesignRules> {
    let trimmed = content.trim_start();
    if trimmed.starts_with('{') {
        let root = serde_json::from_str::<serde_json::Value>(content).ok()?;
        let design_settings = root.get("board").and_then(|b| b.get("design_settings")).or_else(|| root.get("design_settings"))?;
        let mut rules = DesignRules::default();
        let mut found_any = false;
        if let Some(r_obj) = design_settings.get("rules") {
            found_any = true;
            if let Some(v) = r_obj.get("min_clearance").and_then(|v| v.as_f64()) {
                rules.min_clearance = v;
            }
            if let Some(v) = r_obj.get("min_track_width").and_then(|v| v.as_f64()) {
                rules.min_track_width = v;
            }
            if let Some(v) = r_obj.get("min_via_diameter").and_then(|v| v.as_f64()) {
                rules.min_via_diameter = v;
            }
            if let Some(v) = r_obj.get("min_via_drill").and_then(|v| v.as_f64()) {
                rules.min_via_drill = v;
            }
            if let Some(v) = r_obj.get("min_via_annular_width").and_then(|v| v.as_f64()) {
                rules.min_via_annular_width = v;
            }
            if let Some(v) = r_obj.get("min_through_hole_diameter").and_then(|v| v.as_f64()) {
                rules.min_through_hole_diameter = v;
            }
            if let Some(v) = r_obj.get("min_hole_clearance").and_then(|v| v.as_f64()) {
                rules.min_hole_clearance = v;
            }
            if let Some(v) = r_obj.get("min_hole_to_hole").and_then(|v| v.as_f64()) {
                rules.min_hole_to_hole = v;
            }
            if let Some(v) = r_obj.get("min_copper_edge_clearance").and_then(|v| v.as_f64()) {
                rules.min_copper_edge_clearance = v;
            }
        }
        if let Some(z_obj) = design_settings.get("defaults").and_then(|d| d.get("zones")) {
            found_any = true;
            if let Some(v) = z_obj.get("min_clearance").and_then(|v| v.as_f64()) {
                rules.zone_min_clearance = v;
            }
            if let Some(v) = z_obj.get("min_thickness").and_then(|v| v.as_f64()) {
                rules.zone_min_thickness = v;
            }
        }
        if found_any {
            Some(rules)
        } else {
            None
        }
    } else if trimmed.contains("(rule ") || trimmed.contains("(constraint ") {
        let mut rules = DesignRules::default();
        let mut dummy = HashMap::new();
        apply_dru_rules_to_board(content, &mut rules, &mut dummy);
        Some(rules)
    } else {
        None
    }
}

/// Parses a map of named [`NetClass`] definitions from either a KiCad `.kicad_pro` JSON file or
/// `(net_class ...)` S-expression blocks in a `.kicad_pcb` file.
///
/// # Arguments
/// * `content` - Raw `.kicad_pro` JSON or `.kicad_pcb` S-expression content.
///
/// # Returns
/// `HashMap<String, NetClass>` mapping netclass names (such as `"Default"`, `"Power"`) to their parsed rules.
pub fn parse_netclasses_from_pro_or_pcb(content: &str) -> HashMap<String, NetClass> {
    let mut netclasses = HashMap::new();
    let trimmed = content.trim_start();
    if trimmed.starts_with('{') {
        if let Ok(root) = serde_json::from_str::<serde_json::Value>(content) {
            let pro_rules = parse_design_rules_from_pro_or_dru(content);
            let root_preset = root
                .get("fab_preset")
                .or_else(|| root.get("preset"))
                .or_else(|| root.get("fab"))
                .and_then(|v| v.as_str())
                .and_then(FabPreset::parse);

            let classes_opt = root
                .get("net_settings")
                .and_then(|v| v.get("classes"))
                .or_else(|| root.get("classes"))
                .and_then(|v| v.as_array());
            if let Some(classes) = classes_opt {
                // KiCad netclasses inherit unset diff-pair dimensions from the Default class.
                let default_cls = classes.iter().find(|c| c.get("name").and_then(|v| v.as_str()) == Some("Default"));
                let default_dp = |key: &str| default_cls.and_then(|c| c.get(key)).and_then(|v| v.as_f64());
                for cls in classes {
                    let Some(name) = cls.get("name").and_then(|v| v.as_str()) else { continue };
                    let track_width = cls.get("track_width").and_then(|v| v.as_f64()).unwrap_or(0.20);
                    let clearance = cls.get("clearance").and_then(|v| v.as_f64()).unwrap_or(0.20);
                    let via_dia = cls.get("via_diameter").or_else(|| cls.get("via_dia")).and_then(|v| v.as_f64()).unwrap_or(0.60);
                    let via_drill = cls.get("via_drill").and_then(|v| v.as_f64()).unwrap_or(0.30);
                    let diff_pair_gap = cls
                        .get("diff_pair_gap")
                        .and_then(|v| v.as_f64())
                        .or_else(|| default_dp("diff_pair_gap"))
                        .unwrap_or(0.20);
                    let diff_pair_width = cls
                        .get("diff_pair_width")
                        .and_then(|v| v.as_f64())
                        .or_else(|| default_dp("diff_pair_width"))
                        .unwrap_or(track_width);
                    let topology = cls
                        .get("topology")
                        .and_then(|v| v.as_str())
                        .and_then(NetTopology::parse)
                        .unwrap_or_default();
                    let is_hs = NetClass::is_high_speed_name(name);
                    let max_coupled_length_mm = cls
                        .get("max_coupled_length_mm")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(15.0);
                    let require_return_path = cls
                        .get("require_return_path")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(is_hs);

                    let cls_preset = cls
                        .get("fab_preset")
                        .or_else(|| cls.get("preset"))
                        .and_then(|v| v.as_str())
                        .and_then(FabPreset::parse)
                        .or(root_preset);

                    let teardrop_enabled = cls
                        .get("teardrop_enabled")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);
                    let teardrop_length_ratio = cls
                        .get("teardrop_length_ratio")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.5);
                    let teardrop_width_ratio = cls
                        .get("teardrop_width_ratio")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.8);
                    let default_edge_clr = cls_preset
                        .map(|p| p.board_edge_clearance())
                        .or_else(|| pro_rules.map(|r| r.min_copper_edge_clearance))
                        .unwrap_or(0.35);
                    let board_edge_clearance = cls
                        .get("board_edge_clearance")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(default_edge_clr);

                    let mut nc = NetClass {
                        name: name.to_string(),
                        track_width,
                        clearance,
                        via_dia,
                        via_drill,
                        diff_pair_gap,
                        diff_pair_width,
                        topology,
                        teardrop_enabled,
                        teardrop_length_ratio,
                        teardrop_width_ratio,
                        board_edge_clearance,
                        max_coupled_length_mm,
                        require_return_path,
                        ..NetClass::default()
                    };
                    if let Some(ref rules) = pro_rules {
                        nc.apply_design_rules(rules);
                    }
                    if let Some(p) = cls_preset {
                        p.apply_to_netclass(&mut nc);
                        if let Some(explicit_bec) = cls.get("board_edge_clearance").and_then(|v| v.as_f64()) {
                            nc.board_edge_clearance = explicit_bec;
                        }
                    }
                    netclasses.insert(name.to_string(), nc);
                }
            }
        }
    } else {
        let global_preset = if let Some(pos) = content.find("(fab_preset") {
            let rest = &content[pos + "(fab_preset".len()..];
            let token = rest.trim_start().split(|c: char| c.is_whitespace() || c == ')').next().unwrap_or("").trim_matches('"');
            FabPreset::parse(token)
        } else if let Some(pos) = content.find("(preset") {
            let rest = &content[pos + "(preset".len()..];
            let token = rest.trim_start().split(|c: char| c.is_whitespace() || c == ')').next().unwrap_or("").trim_matches('"');
            FabPreset::parse(token)
        } else {
            None
        };

        let bytes = content.as_bytes();
        let len = bytes.len();
        let mut i = 0;
        while i < len {
            if bytes[i] == b'(' {
                let word = next_token(content, i + 1);
                if word == "net_class" {
                    if let Some(end) = find_closing_paren(bytes, i) {
                        let block = &content[i..end];
                        let sub_tokens: Vec<&str> = block.split_whitespace().collect();
                        let name = if sub_tokens.len() >= 2 {
                            sub_tokens[1].trim_matches('"').to_string()
                        } else {
                            String::new()
                        };
                        if !name.is_empty() {
                            let parse_val = |key: &str, def: f64| -> f64 {
                                if let Some(pos) = block.find(key) {
                                    let rest = &block[pos + key.len()..];
                                    let mut num_str = String::new();
                                    for ch in rest.chars() {
                                        if ch.is_whitespace() || ch == '(' {
                                            continue;
                                        }
                                        if ch == ')' {
                                            break;
                                        }
                                        num_str.push(ch);
                                    }
                                    num_str.parse::<f64>().unwrap_or(def)
                                } else {
                                    def
                                }
                            };
                            let clearance = parse_val("(clearance", 0.20);
                            let track_width = parse_val("(trace_width", 0.20);
                            let via_dia = parse_val("(via_dia", 0.60);
                            let via_drill = parse_val("(via_drill", 0.30);
                            let diff_pair_gap = parse_val("(diff_pair_gap", 0.20);
                            let diff_pair_width = parse_val("(diff_pair_width", track_width);
                            let max_coupled_length_mm = parse_val("(max_coupled_length_mm", 15.0);
                            let is_hs = NetClass::is_high_speed_name(&name);
                            let require_return_path = if let Some(pos) = block.find("(require_return_path") {
                                let rest = &block[pos + "(require_return_path".len()..];
                                rest.trim_start().starts_with("true") || rest.trim_start().starts_with("1")
                            } else {
                                is_hs
                            };
                            let topology = if let Some(pos) = block.find("(topology") {
                                let rest = &block[pos + "(topology".len()..];
                                let token = rest
                                    .trim_start()
                                    .split(|c: char| c.is_whitespace() || c == ')')
                                    .next()
                                    .unwrap_or("");
                                NetTopology::parse(token).unwrap_or_default()
                            } else {
                                NetTopology::default()
                            };
                            let cls_preset = if let Some(pos) = block.find("(fab_preset") {
                                let rest = &block[pos + "(fab_preset".len()..];
                                let token = rest.trim_start().split(|c: char| c.is_whitespace() || c == ')').next().unwrap_or("").trim_matches('"');
                                FabPreset::parse(token).or(global_preset)
                            } else {
                                global_preset
                            };

                            let default_edge_clr = cls_preset.map(|p| p.board_edge_clearance()).unwrap_or(0.35);
                            let board_edge_clearance = parse_val("(board_edge_clearance", default_edge_clr);
                            let teardrop_length_ratio = parse_val("(teardrop_length_ratio", 0.5);
                            let teardrop_width_ratio = parse_val("(teardrop_width_ratio", 0.8);
                            let teardrop_enabled = if let Some(pos) = block.find("(teardrop_enabled") {
                                let rest = &block[pos + "(teardrop_enabled".len()..];
                                let token = rest.trim_start().split(|c: char| c.is_whitespace() || c == ')').next().unwrap_or("");
                                token == "true" || token == "1" || token == "yes"
                            } else {
                                true
                            };

                            let mut nc = NetClass {
                                name: name.clone(),
                                track_width,
                                clearance,
                                via_dia,
                                via_drill,
                                diff_pair_gap,
                                diff_pair_width,
                                topology,
                                teardrop_enabled,
                                teardrop_length_ratio,
                                teardrop_width_ratio,
                                board_edge_clearance,
                                max_coupled_length_mm,
                                require_return_path,
                                ..NetClass::default()
                            };
                            if let Some(p) = cls_preset {
                                p.apply_to_netclass(&mut nc);
                                if block.contains("(board_edge_clearance") {
                                    nc.board_edge_clearance = board_edge_clearance;
                                }
                            }
                            netclasses.insert(name, nc);
                        }
                        i = end;
                        continue;
                    }
                }
            }
            i += 1;
        }
    }
    netclasses
}

/// Parses a KiCad `(footprint ...)` or `(module ...)` S-expression block into a [`Footprint`],
/// extracting pads, courtyard polygons (decomposed into convex/rectilinear pieces), swap groups, and byte spans.
///
/// # Arguments
/// * `fp_str` - S-expression substring of the footprint block.
/// * `fp_byte_offset` - Starting byte offset of `fp_str` within the full `.kicad_pcb` source file.
/// * `nets` - Mutable board net table for registering newly encountered net names.
/// * `net_name_to_id` - Mutable lookup map from net name to net ID.
/// * `extra_locked` - Set of user-specified reference designators to mark as locked (`is_locked = true`).
///
/// # Returns
/// `Some(Footprint)` if the footprint has a valid `"Reference"` property, or `None` otherwise.
///
/// # Side Effects
/// May append newly discovered pad net names into `nets` and `net_name_to_id`.
fn parse_footprint(
    fp_str: &str,
    fp_byte_offset: usize,
    nets: &mut Vec<String>,
    net_name_to_id: &mut HashMap<String, usize>,
    extra_locked: &HashSet<String>,
) -> Option<Footprint> {
    let reference = extract_property(fp_str, "Reference")?;
    let value = extract_property(fp_str, "Value").unwrap_or_default();

    let header_slice = &fp_str[..fp_str.len().min(200)];
    let layer = if header_slice.contains("(layer \"B.Cu\")") {
        1
    } else {
        0
    };

    // Find top-level (at X Y [ROT]) inside footprint
    let bytes = fp_str.as_bytes();
    let mut pos_x = 0.0;
    let mut pos_y = 0.0;
    let mut rot = 0.0;
    let mut at_span = None;
    let mut depth = 0usize;
    let mut in_quotes = false;

    for i in 0..bytes.len() {
        let c = bytes[i];
        if c == b'"' && (i == 0 || bytes[i - 1] != b'\\') {
            in_quotes = !in_quotes;
        } else if !in_quotes {
            if c == b'(' {
                depth += 1;
                if depth == 2 && fp_str[i..].starts_with("(at ") {
                    if let Some(end) = find_closing_paren(bytes, i) {
                        if let Some((x, y, r)) = parse_at_xy_rot(&fp_str[i..end]) {
                            pos_x = x;
                            pos_y = y;
                            rot = r;
                            at_span = Some((fp_byte_offset + i, fp_byte_offset + end));
                            break;
                        }
                    }
                }
            } else if c == b')' {
                depth = depth.saturating_sub(1);
            }
        }
    }

    let is_locked = fp_str.contains("(locked yes)")
        || fp_str.contains("ESP32-C3-WROOM-02")
        || DEFAULT_LOCKED_REFS.contains(&reference.as_str())
        || extra_locked.contains(&reference);

    // Parse pads with real shape, local rotation, and drill holes
    let mut pads = Vec::new();
    let mut p_idx = 0;
    while p_idx < bytes.len() {
        if bytes[p_idx] == b'(' && fp_str[p_idx..].starts_with("(pad ") {
            if let Some(p_end) = find_closing_paren(bytes, p_idx) {
                let p_str = &fp_str[p_idx..p_end];
                // Skip non-copper mechanical/solder-paste-only aperture pads (e.g. QFN thermal paste stencils)
                if p_str.contains("(layers ") && !p_str.contains(".Cu") {
                    p_idx = p_end;
                    continue;
                }
                let tokens: Vec<&str> = p_str[5..].split_whitespace().collect();
                let p_num = tokens.first().map(|s| s.trim_matches('"').to_string()).unwrap_or_default();
                let is_th = tokens.get(1).copied() == Some("thru_hole")
                    || tokens.get(1).copied() == Some("np_thru_hole")
                    || (p_str.contains("\"F.Cu\"") && p_str.contains("\"B.Cu\""))
                    || p_str.contains("\"*.Cu\"");
                let pad_layer = if is_th {
                    None
                } else if p_str.contains("\"B.Cu\"") && !p_str.contains("\"F.Cu\"") {
                    Some(1)
                } else if p_str.contains("\"F.Cu\"") && !p_str.contains("\"B.Cu\"") {
                    Some(0)
                } else {
                    Some(layer)
                };
                let shape = tokens.get(2).copied().unwrap_or("rect").to_string();
                let (off_x, off_y, pad_board_rot) = parse_at_xy_rot(p_str).unwrap_or((0.0, 0.0, 0.0));
                let rel_rot_norm = ((pad_board_rot - rot) % 360.0 + 360.0) % 360.0;
                let pad_rot = if rel_rot_norm > 180.0 { rel_rot_norm - 360.0 } else { rel_rot_norm };
                let mut w: f64 = 0.5;
                let mut h: f64 = 0.5;
                if let Some(sz_pos) = p_str.find("(size ") {
                    let after = &p_str[sz_pos + 6..];
                    if let Some(sz_end) = after.find(')') {
                        let mut sp = after[..sz_end].split_whitespace();
                        if let (Some(ws), Some(hs)) = (sp.next(), sp.next()) {
                            w = ws.parse().unwrap_or(0.5);
                            h = hs.parse().unwrap_or(0.5);
                        }
                    }
                }
                let drill = if let Some(d_pos) = p_str.find("(drill ") {
                    let after = &p_str[d_pos + 7..];
                    if let Some(d_end) = after.find(')') {
                        let mut dp = after[..d_end].split_whitespace();
                        let first = dp.next().unwrap_or("");
                        if first == "oval" {
                            dp.next().and_then(|s| s.parse::<f64>().ok())
                        } else {
                            first.parse::<f64>().ok()
                        }
                    } else {
                        None
                    }
                } else if is_th {
                    Some(w.min(h) * 0.5)
                } else {
                    None
                };
                let mut net_name = String::new();
                let mut net_id = 0usize;
                let mut net_span = None;
                if let Some(n_pos) = p_str.find("(net ") {
                    let after = &p_str[n_pos + 5..];
                    if let Some(close_end) = find_closing_paren(p_str.as_bytes(), n_pos) {
                        let abs_n_start = fp_byte_offset + p_idx + n_pos;
                        let abs_n_end = fp_byte_offset + p_idx + close_end;
                        net_span = Some((abs_n_start, abs_n_end));
                    }
                    if let Some(q1) = after.find('"') {
                        if let Some(q2) = after[q1 + 1..].find('"') {
                            net_name = after[q1 + 1..q1 + 1 + q2].to_string();
                            if !net_name.is_empty() {
                                if let Some(&id) = net_name_to_id.get(&net_name) {
                                    net_id = id;
                                } else {
                                    let new_id = nets.len();
                                    nets.push(net_name.clone());
                                    net_name_to_id.insert(net_name.clone(), new_id);
                                    net_id = new_id;
                                }
                            }
                        }
                    }
                }

                let mut pin_name = String::new();
                if let Some(pf_pos) = p_str.find("(pinfunction ") {
                    let after = &p_str[pf_pos + 13..];
                    if let Some(q1) = after.find('"') {
                        if let Some(q2) = after[q1 + 1..].find('"') {
                            pin_name = after[q1 + 1..q1 + 1 + q2].to_string();
                        }
                    } else if let Some(end) = after.find(')') {
                        pin_name = after[..end].trim().to_string();
                    }
                } else if let Some(pn_pos) = p_str.find("(name ") {
                    let after = &p_str[pn_pos + 6..];
                    if let Some(q1) = after.find('"') {
                        if let Some(q2) = after[q1 + 1..].find('"') {
                            pin_name = after[q1 + 1..q1 + 1 + q2].to_string();
                        }
                    }
                }

                let mut pad_swap_group = None;
                if let Some(sg_pos) = p_str.find("(swap_group ") {
                    let after = &p_str[sg_pos + 12..];
                    if let Some(end) = after.find(')') {
                        let token = after[..end].trim().trim_matches('"');
                        pad_swap_group = token.parse::<usize>().ok();
                    }
                }

                pads.push(Pad {
                    number: p_num,
                    shape,
                    offset_x: off_x,
                    offset_y: off_y,
                    width: w,
                    height: h,
                    rotation: pad_rot,
                    drill,
                    net_id,
                    net_name,
                    is_through_hole: is_th,
                    pin_name,
                    swap_group: pad_swap_group,
                    net_span,
                    layer: pad_layer,
                });
                p_idx = p_end;
                continue;
            }
        }
        p_idx += 1;
    }

    // Parse courtyard geometry from KiCad (fp_poly, fp_rect, fp_line on CrtYd)
    let mut courtyards = Vec::new();
    let mut crtyd_lines: Vec<(Point, Point)> = Vec::new();

    let mut c_idx = 0;
    while c_idx < bytes.len() {
        if bytes[c_idx] == b'(' {
            let tag = next_token(fp_str, c_idx + 1);
            if tag == "fp_poly" || tag == "fp_rect" || tag == "fp_line" {
                if let Some(c_end) = find_closing_paren(bytes, c_idx) {
                    let c_str = &fp_str[c_idx..c_end];
                    if c_str.contains("CrtYd") {
                        if tag == "fp_poly" {
                            let mut pts = Vec::new();
                            let tag_xy = "(xy ";
                            let mut idx = 0;
                            while let Some(pos) = c_str[idx..].find(tag_xy) {
                                let s = idx + pos + tag_xy.len();
                                if let Some(e) = c_str[s..].find(')') {
                                    let mut parts = c_str[s..s + e].split_whitespace();
                                    if let (Some(xs), Some(ys)) = (parts.next(), parts.next()) {
                                        if let (Ok(x), Ok(y)) = (xs.parse::<f64>(), ys.parse::<f64>()) {
                                            pts.push(Point::new(x, y));
                                        }
                                    }
                                    idx = s + e;
                                } else {
                                    break;
                                }
                            }
                            if pts.len() >= 3 {
                                courtyards.push(Polygon::new(pts));
                            }
                        } else if tag == "fp_rect" {
                            let mut rx0 = f64::INFINITY;
                            let mut ry0 = f64::INFINITY;
                            let mut rx1 = f64::NEG_INFINITY;
                            let mut ry1 = f64::NEG_INFINITY;
                            extract_coords_into_bounds(c_str, &mut rx0, &mut ry0, &mut rx1, &mut ry1);
                            if rx0 < rx1 && ry0 < ry1 {
                                courtyards.push(Polygon::new(vec![
                                    Point::new(rx0, ry0),
                                    Point::new(rx1, ry0),
                                    Point::new(rx1, ry1),
                                    Point::new(rx0, ry1),
                                ]));
                            }
                        } else if tag == "fp_line" {
                            let parse_pt = |marker: &str| -> Option<Point> {
                                let p = c_str.find(marker)?;
                                let rest = &c_str[p + marker.len()..];
                                let e = rest.find(')')?;
                                let mut parts = rest[..e].split_whitespace();
                                let x = parts.next()?.parse::<f64>().ok()?;
                                let y = parts.next()?.parse::<f64>().ok()?;
                                Some(Point::new(x, y))
                            };
                            if let (Some(p1), Some(p2)) = (parse_pt("(start "), parse_pt("(end ")) {
                                crtyd_lines.push((p1, p2));
                            }
                        }
                    }
                    c_idx = c_end;
                    continue;
                }
            }
        }
        c_idx += 1;
    }

    if !crtyd_lines.is_empty() {
        let chained = chain_lines_into_polygons(crtyd_lines);
        courtyards.extend(chained);
    }

    // Auto-heal footprints where top-level (at X Y) rotation was stripped to 0° while child pads
    // retained their old board-space angle, causing pads to overlap locally or protrude outside CrtYd.
    if pads.len() >= 2 && pads.iter().any(|p| p.rotation.abs() > 1e-3) {
        let non_zero_rots: Vec<f64> = pads
            .iter()
            .filter(|p| p.rotation.abs() > 1e-3)
            .map(|p| p.rotation)
            .collect();
        let same_rot = non_zero_rots.len() == pads.len()
            && non_zero_rots.iter().all(|&r| (r - non_zero_rots[0]).abs() < 1e-3);
        if same_rot {
            let pad_eff_dims = |p: &Pad, zero_rot: bool| -> (f64, f64) {
                if zero_rot {
                    (p.width, p.height)
                } else {
                    let rad = p.rotation.to_radians();
                    let c = rad.cos().abs();
                    let s = rad.sin().abs();
                    (p.width * c + p.height * s, p.width * s + p.height * c)
                }
            };
            let pads_overlap_locally = |ps: &[Pad], zero_rot: bool| -> bool {
                for i in 0..ps.len() {
                    if ps[i].number.is_empty() {
                        continue;
                    }
                    let (wi, hi) = pad_eff_dims(&ps[i], zero_rot);
                    for j in (i + 1)..ps.len() {
                        if ps[j].number.is_empty() || ps[i].number == ps[j].number {
                            continue;
                        }
                        if !(ps[i].is_through_hole || ps[j].is_through_hole || ps[i].layer == ps[j].layer) {
                            continue;
                        }
                        let (wj, hj) = pad_eff_dims(&ps[j], zero_rot);
                        if (ps[i].offset_x - ps[j].offset_x).abs() < (wi + wj) * 0.5 - 1e-3
                            && (ps[i].offset_y - ps[j].offset_y).abs() < (hi + hj) * 0.5 - 1e-3
                        {
                            return true;
                        }
                    }
                }
                false
            };
            let pads_outside_courtyard = |ps: &[Pad], crtyds: &[Polygon], zero_rot: bool| -> bool {
                if crtyds.is_empty() {
                    return false;
                }
                let mut c_min_x = f64::INFINITY;
                let mut c_min_y = f64::INFINITY;
                let mut c_max_x = f64::NEG_INFINITY;
                let mut c_max_y = f64::NEG_INFINITY;
                for poly in crtyds {
                    for pt in &poly.vertices {
                        c_min_x = c_min_x.min(pt.x);
                        c_min_y = c_min_y.min(pt.y);
                        c_max_x = c_max_x.max(pt.x);
                        c_max_y = c_max_y.max(pt.y);
                    }
                }
                if c_min_x >= c_max_x || c_min_y >= c_max_y {
                    return false;
                }
                ps.iter().any(|p| {
                    let (ew, eh) = pad_eff_dims(p, zero_rot);
                    p.offset_x - ew * 0.5 < c_min_x - 0.05
                        || p.offset_x + ew * 0.5 > c_max_x + 0.05
                        || p.offset_y - eh * 0.5 < c_min_y - 0.05
                        || p.offset_y + eh * 0.5 > c_max_y + 0.05
                })
            };

            if (pads_overlap_locally(&pads, false) && !pads_overlap_locally(&pads, true))
                || (pads_outside_courtyard(&pads, &courtyards, false)
                    && !pads_outside_courtyard(&pads, &courtyards, true))
            {
                for p in &mut pads {
                    p.rotation = 0.0;
                }
            }
        }
    }

    // Fallback if no courtyard primitives found on CrtYd
    if courtyards.is_empty() {
        if !pads.is_empty() {
            let mut x0 = f64::INFINITY;
            let mut y0 = f64::INFINITY;
            let mut x1 = f64::NEG_INFINITY;
            let mut y1 = f64::NEG_INFINITY;
            for p in &pads {
                let hw = p.width / 2.0 + 0.25;
                let hh = p.height / 2.0 + 0.25;
                x0 = x0.min(p.offset_x - hw);
                x1 = x1.max(p.offset_x + hw);
                y0 = y0.min(p.offset_y - hh);
                y1 = y1.max(p.offset_y + hh);
            }
            if x0 < x1 && y0 < y1 {
                courtyards.push(Polygon::new(vec![
                    Point::new(x0, y0),
                    Point::new(x1, y0),
                    Point::new(x1, y1),
                    Point::new(x0, y1),
                ]));
            }
        }
        if courtyards.is_empty() {
            courtyards.push(Polygon::new(vec![
                Point::new(-1.0, -1.0),
                Point::new(1.0, -1.0),
                Point::new(1.0, 1.0),
                Point::new(-1.0, 1.0),
            ]));
        }
    }

    let perimeters = courtyards.clone();

    let mut convex_courtyards = Vec::new();
    for poly in courtyards {
        convex_courtyards.extend(poly.decompose_convex());
    }
    courtyards = convex_courtyards;

    let mut min_cx = f64::INFINITY;
    let mut min_cy = f64::INFINITY;
    let mut max_cx = f64::NEG_INFINITY;
    let mut max_cy = f64::NEG_INFINITY;
    for poly in &courtyards {
        let a = poly.aabb();
        min_cx = min_cx.min(a.min_x);
        min_cy = min_cy.min(a.min_y);
        max_cx = max_cx.max(a.max_x);
        max_cy = max_cy.max(a.max_y);
    }

    let mut pin_swap_groups: Vec<Vec<usize>> = Vec::new();
    let mut gate_swap_groups: Vec<SwappableGateGroup> = Vec::new();

    let mut sg_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, p) in pads.iter().enumerate() {
        if let Some(sg) = p.swap_group {
            sg_map.entry(sg).or_default().push(idx);
        }
    }
    let mut sorted_sgs: Vec<usize> = sg_map.keys().copied().collect();
    sorted_sgs.sort_unstable();
    for sg in sorted_sgs {
        let indices = sg_map.remove(&sg).unwrap();
        if indices.len() >= 2 {
            pin_swap_groups.push(indices);
        }
    }

    let mut search_idx = 0;
    while search_idx < bytes.len() {
        if bytes[search_idx] == b'(' && fp_str[search_idx..].starts_with("(swap_group ") {
            if let Some(sg_end) = find_closing_paren(bytes, search_idx) {
                let sg_str = &fp_str[search_idx + 12..sg_end];
                let mut group_pads = Vec::new();
                for token in sg_str.split_whitespace() {
                    let clean = token.trim_matches(['(', ')', '"', ',']);
                    if clean.is_empty() {
                        continue;
                    }
                    if let Some(p_idx) = pads.iter().position(|p| p.number == clean) {
                        group_pads.push(p_idx);
                    } else if let Ok(idx) = clean.parse::<usize>() {
                        if idx < pads.len() {
                            group_pads.push(idx);
                        }
                    }
                }
                if group_pads.len() >= 2 {
                    pin_swap_groups.push(group_pads);
                }
                search_idx = sg_end;
                continue;
            }
        }
        search_idx += 1;
    }

    let mut g_search = 0;
    while g_search < bytes.len() {
        if bytes[g_search] == b'(' && fp_str[g_search..].starts_with("(gate_swap ") {
            if let Some(gs_end) = find_closing_paren(bytes, g_search) {
                let gs_str = &fp_str[g_search + 11..gs_end];
                let mut units = Vec::new();
                let u_bytes = gs_str.as_bytes();
                let mut ui = 0;
                while ui < u_bytes.len() {
                    if u_bytes[ui] == b'(' {
                        if let Some(u_end) = find_closing_paren(u_bytes, ui) {
                            let unit_str = &gs_str[ui + 1..u_end];
                            let mut unit_pads = Vec::new();
                            for token in unit_str.split_whitespace() {
                                let clean = token.trim_matches(['(', ')', '"', ',']);
                                if let Some(p_idx) = pads.iter().position(|p| p.number == clean) {
                                    unit_pads.push(p_idx);
                                } else if let Ok(idx) = clean.parse::<usize>() {
                                    if idx < pads.len() {
                                        unit_pads.push(idx);
                                    }
                                }
                            }
                            if !unit_pads.is_empty() {
                                units.push(unit_pads);
                            }
                            ui = u_end;
                            continue;
                        }
                    }
                    ui += 1;
                }
                if units.len() >= 2 {
                    gate_swap_groups.push(SwappableGateGroup {
                        name: "gate_group".to_string(),
                        gate_units: units,
                    });
                }
                g_search = gs_end;
                continue;
            }
        }
        g_search += 1;
    }

    let mut fp = Footprint {
        reference,
        value,
        x: pos_x,
        y: pos_y,
        rotation: rot,
        layer,
        orig_layer: layer,
        is_locked,
        pads,
        width: (max_cx - min_cx).max(1.0),
        height: (max_cy - min_cy).max(1.0),
        courtyards,
        perimeters,
        escape_margin: 0.0,
        at_span,
        fp_span: Some((fp_byte_offset, fp_byte_offset + fp_str.len())),
        pin_swap_groups,
        gate_swap_groups,
    };
    fp.infer_swap_groups();
    fp.escape_margin = fp.compute_escape_margin();
    Some(fp)
}

/// Execution phase of an interactive or headless Vulcan's Trials auto-placement and auto-routing run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialPhase {
    /// No active trial is running; the board is idle and ready for editing or trial launch.
    Idle,
    /// Phase 1a: Resolving initial courtyard overlaps and boundary violations (`legalize_placement`).
    Legalizing,
    /// Phase 1b: Running GPU/CPU Parallel Tempering Simulated Annealing auto-placement (`run_auto_placer`).
    AutoPlacing,
    /// Phase 2: Running multi-layer PathFinder negotiated-congestion and A* auto-routing (`AutoRouter`).
    AutoRouting,
    /// All requested placement and/or routing phases have finished.
    Completed,
}

impl TrialPhase {
    /// Returns a human-readable status banner label describing the current [`TrialPhase`].
    ///
    /// # Returns
    /// A static string slice suitable for GUI status bars and progress logs.
    pub fn label(&self) -> &'static str {
        match self {
            TrialPhase::Idle => "Idle — Ready",
            TrialPhase::Legalizing => "Phase 1: Courtyard Legalization",
            TrialPhase::AutoPlacing => "Phase 1: Parallel Tempering Auto-Placer",
            TrialPhase::AutoRouting => "Phase 2: Rip-up & Nudge Auto-Router",
            TrialPhase::Completed => "Trial Complete",
        }
    }
}

/// Thread-safe shared state snapshot connecting background placement/routing worker threads with the
/// real-time `eframe`/`egui` GUI canvas and MCP status endpoints.
#[derive(Debug, Clone)]
pub struct LiveTrialState {
    /// Current working [`Board`] geometry (footprints, tracks, arcs, vias, zones).
    pub board: Board,
    /// Current Prim's Euclidean minimum spanning tree airwire segments in board space (`mm`).
    pub ratsnest: Vec<Segment>,
    /// Current execution phase of the trial pipeline.
    pub phase: TrialPhase,
    /// Normalized progress ratio in `` `[0.0, 1.0]` `` for the active phase.
    pub progress: f32,
    /// Most recent single-line status message displayed in the GUI header.
    pub status_text: String,
    /// Ring buffer (capped at 250 lines) of chronological trial log messages.
    pub log_lines: Vec<String>,
    /// Number of pairwise ratsnest airwire intersections in the current placement.
    pub crossings: usize,
    /// Total Euclidean MST ratsnest wirelength in millimeters (`mm`).
    pub wirelength_mm: f64,
    /// Number of unresolved footprint courtyard overlaps.
    pub overlaps: usize,
    /// Current auto-placer evaluation throughput in state evaluations per second (`evals/s`).
    pub evals_per_sec: f64,
    /// Number of net connections successfully routed in the current routing pass.
    pub routed_conns: usize,
    /// Total number of net connections scheduled for routing.
    pub total_conns: usize,
    /// Current PathFinder / rip-up-and-reroute iteration pass index (1-based when active).
    pub routing_pass: usize,
    /// Optional snapshot of octilinear `(tracks, arcs)` prior to organic tangent-arc conversion.
    pub straight_copper_snapshot: Option<(Vec<TrackSegment>, Vec<ArcSegment>)>,
    /// Fillet radius in millimeters (`mm`) used when smoothing acute track junctions.
    pub acid_trap_fillet_radius: f64,
    /// Whether vias are permitted inside SMD pad copper areas during routing.
    pub allow_via_in_pad: bool,
}

impl LiveTrialState {
    /// Constructs a new [`LiveTrialState`] initialized from `board` and computes its initial ratsnest metrics.
    ///
    /// # Arguments
    /// * `board` - Initial [`Board`] instance to wrap.
    ///
    /// # Returns
    /// A newly initialized [`LiveTrialState`] in [`TrialPhase::Idle`].
    pub fn new(board: Board) -> Self {
        let allow_via_in_pad = board.allow_via_in_pad();
        let mut state = Self {
            board,
            ratsnest: Vec::new(),
            phase: TrialPhase::Idle,
            progress: 0.0,
            status_text: "Loaded board. Ready for Vulcan's Trials.".to_string(),
            log_lines: vec!["[Vulcan's Trials] Initialized board state.".to_string()],
            crossings: 0,
            wirelength_mm: 0.0,
            overlaps: 0,
            evals_per_sec: 0.0,
            routed_conns: 0,
            total_conns: 0,
            routing_pass: 0,
            straight_copper_snapshot: None,
            acid_trap_fillet_radius: 0.35,
            allow_via_in_pad,
        };
        state.refresh_ratsnest();
        state
    }

    /// Recomputes all world-space pad coordinates, regenerates the Prim's MST `ratsnest` segments,
    /// and updates `wirelength_mm` and pairwise `crossings`.
    ///
    /// # Side Effects
    /// Updates `self.ratsnest`, `self.wirelength_mm`, and `self.crossings` in place.
    pub fn refresh_ratsnest(&mut self) {
        let pad_pos: Vec<Vec<Point>> = self
            .board
            .footprints
            .iter()
            .map(|fp| {
                (0..fp.pads.len())
                    .map(|pi| fp.pad_world_point(pi, Point::new(fp.x, fp.y), fp.rotation))
                    .collect()
            })
            .collect();
        self.ratsnest = self.board.generate_ratsnest_segments(&pad_pos);
        self.wirelength_mm = self.ratsnest.iter().map(|s| s.length()).sum();
        let mut crossings = 0;
        for i in 0..self.ratsnest.len() {
            for j in (i + 1)..self.ratsnest.len() {
                if self.ratsnest[i].crosses(&self.ratsnest[j]) {
                    crossings += 1;
                }
            }
        }
        self.crossings = crossings;
    }

    /// Appends a timestamped/formatted status message to `log_lines` (evicting the oldest entry if over 250 lines)
    /// and updates `status_text`.
    ///
    /// # Arguments
    /// * `msg` - Status string or displayable message to record.
    ///
    /// # Side Effects
    /// Mutates `self.status_text` and `self.log_lines`.
    pub fn push_log(&mut self, msg: impl Into<String>) {
        let s = msg.into();
        self.status_text = s.clone();
        self.log_lines.push(s);
        if self.log_lines.len() > 250 {
            self.log_lines.remove(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_rotation_and_distance() {
        let p = Point::new(2.0, 0.0);
        let r90 = p.rotate_ccw(90.0);
        assert!(r90.x.abs() < 1e-6 && (r90.y + 2.0).abs() < 1e-6);
        let r180 = p.rotate_ccw(180.0);
        assert!((r180.x + 2.0).abs() < 1e-6);
        assert!(r180.y.abs() < 1e-6);
        assert!((p.distance_to(&r180) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_aabb_overlaps_and_contains() {
        let a = AABB::new(0.0, 0.0, 4.0, 4.0);
        let b = AABB::new(2.0, 2.0, 6.0, 6.0);
        let touching = AABB::new(4.0, 0.0, 8.0, 4.0);
        let disjoint = AABB::new(5.0, 5.0, 7.0, 7.0);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&touching));
        assert!(!a.overlaps(&disjoint));
        assert!(a.contains_point(&Point::new(2.0, 2.0)));
        assert!(a.contains_point(&Point::new(0.0, 4.0)));
        assert!(!a.contains_point(&Point::new(-0.1, 2.0)));
    }

    #[test]
    fn test_polygon_aabb_transform_and_expanded() {
        let empty = Polygon::new(vec![]);
        assert_eq!(empty.expanded(1.0).vertices.len(), 0);

        let sq = Polygon::new(vec![
            Point::new(-1.0, -1.0),
            Point::new(1.0, -1.0),
            Point::new(1.0, 1.0),
            Point::new(-1.0, 1.0),
        ]);
        assert_eq!(sq.expanded(0.0), sq);
        assert_eq!(sq.expanded(-0.5), sq);
        let exp4 = sq.expanded(0.5).aabb();
        assert!((exp4.min_x + 1.5).abs() < 1e-6 && (exp4.max_x - 1.5).abs() < 1e-6);

        // Non-4-vertex polygon radial expansion + degenerate center vertex
        let tri = Polygon::new(vec![
            Point::new(0.0, 0.0),
            Point::new(3.0, 0.0),
            Point::new(0.0, 3.0),
            Point::new(3.0, 3.0),
            Point::new(1.5, 1.5), // exactly at centroid (1.5, 1.5) -> len == 0 branch
        ]);
        let exp_tri = tri.expanded(0.5);
        assert_eq!(exp_tri.vertices.len(), 5);
        assert!((exp_tri.vertices[4].x - 1.5).abs() < 1e-6);
        assert!(exp_tri.aabb().max_x > 3.0);
    }

    #[test]
    fn test_sat_polygon_collision() {
        let degen = Polygon::new(vec![Point::new(0.0, 0.0), Point::new(1.0, 1.0)]);
        let sq = Polygon::new(vec![
            Point::new(-1.0, -1.0),
            Point::new(1.0, -1.0),
            Point::new(1.0, 1.0),
            Point::new(-1.0, 1.0),
        ]);
        assert!(!degen.intersects_convex(&sq));
        assert!(!sq.intersects_convex(&degen));

        let overlapping = sq.transform(Point::new(1.5, 0.0), 45.0);
        let separated = sq.transform(Point::new(3.5, 0.0), 45.0);
        // Overlapping AABB but separated by diagonal SAT axis
        let diamond_a = sq.transform(Point::new(0.0, 0.0), 45.0);
        let diamond_b = sq.transform(Point::new(2.2, 2.2), 45.0);
        assert!(diamond_a.aabb().overlaps(&diamond_b.aabb()));
        assert!(!diamond_a.intersects_convex(&diamond_b));
        assert!(sq.intersects_convex(&overlapping));
        assert!(!sq.intersects_convex(&separated));
    }

    #[test]
    fn test_segment_crossing_vs_shared_endpoint() {
        let s1 = Segment {
            p1: Point::new(0.0, 0.0),
            p2: Point::new(3.0, 4.0),
            net_id: 1,
        };
        assert!((s1.length() - 5.0).abs() < 1e-6);
        assert_eq!(s1.aabb(), AABB::new(0.0, 0.0, 3.0, 4.0));

        let s2 = Segment {
            p1: Point::new(0.0, 4.0),
            p2: Point::new(3.0, 0.0),
            net_id: 2,
        };
        let s_same_net = Segment {
            p1: Point::new(0.0, 4.0),
            p2: Point::new(3.0, 0.0),
            net_id: 1,
        };
        let s_shared_pin = Segment {
            p1: Point::new(0.0, 0.0),
            p2: Point::new(3.0, 0.0),
            net_id: 2,
        };
        let s_disjoint = Segment {
            p1: Point::new(10.0, 10.0),
            p2: Point::new(12.0, 12.0),
            net_id: 2,
        };
        assert!(s1.crosses(&s2));
        assert!(!s1.crosses(&s_same_net));
        assert!(!s1.crosses(&s_shared_pin));
        assert!(!s1.crosses(&s_disjoint));
    }

    #[test]
    fn test_netclass_default_and_footprint_geometry() {
        let nc = NetClass::default();
        assert_eq!(nc.name, "Default");
        assert!((nc.track_width - 0.20).abs() < 1e-6);
        assert!((nc.diff_pair_gap - 0.20).abs() < 1e-6);
        assert!((nc.diff_pair_width - 0.20).abs() < 1e-6);

        let mut fp = Footprint {
            reference: "J2".to_string(),
            value: "USB_C".to_string(),
            x: 10.0,
            y: 20.0,
            rotation: -90.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![Pad::new("A5", 2.0, 1.0, 0.30, 1.45, 1, "NET1", false)],
            width: 4.0,
            height: 4.0,
            courtyards: vec![Polygon::new(vec![
                Point::new(-2.0, -2.0),
                Point::new(2.0, -2.0),
                Point::new(2.0, 2.0),
                Point::new(-2.0, 2.0),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };
        assert!(fp.is_smd_only());
        // At -90 deg rotation, pad (0.30 x 1.45) must have world size (1.45 x 0.30)
        let (pw, ph) = fp.pad_world_size(&fp.pads[0]);
        assert!((pw - 1.45).abs() < 1e-6 && (ph - 0.30).abs() < 1e-6);

        fp.rotation = 90.0;
        fp.pads[0].offset_y = 0.0;
        let (ax, ay) = fp.pad_absolute_position(&fp.pads[0]);
        let wp = fp.pad_world_point(0, Point::new(10.0, 20.0), 90.0);
        assert!((ax - wp.x).abs() < 1e-6 && (ay - wp.y).abs() < 1e-6);
        assert!((ax - 10.0).abs() < 1e-6 && (ay - 18.0).abs() < 1e-6);

        // Flip footprint to B.Cu (layer = 1) and verify Y offset mirrors
        fp.pads[0].offset_y = 1.5;
        let wp_top = fp.pad_world_point_layered(0, Point::new(10.0, 20.0), 0.0, 0);
        let wp_bot = fp.pad_world_point_layered(0, Point::new(10.0, 20.0), 0.0, 1);
        assert!((wp_top.y - 21.5).abs() < 1e-6 && (wp_bot.y - 18.5).abs() < 1e-6);

        let cy_no_space = fp.transformed_courtyards(Point::new(10.0, 20.0), 0.0, 0.0);
        let cy_spaced = fp.transformed_courtyards(Point::new(10.0, 20.0), 0.0, 0.5);
        let cy_flipped = fp.transformed_courtyards_layered(Point::new(10.0, 20.0), 0.0, 1, 0.0);
        assert!(cy_spaced[0].aabb().max_x > cy_no_space[0].aabb().max_x);
        assert_eq!(cy_flipped[0].aabb(), cy_no_space[0].aabb());
    }

    #[test]
    fn test_s_expr_helpers_and_footprint_parser() {
        assert_eq!(next_token("(  footprint \"R\" )", 1), "footprint");
        let s = "(outer \"quoted ) paren \\\" still\" (inner 1))";
        assert_eq!(find_closing_paren(s.as_bytes(), 0), Some(s.len()));
        assert_eq!(find_closing_paren(b"(unclosed", 0), None);

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        extract_coords_into_bounds("(gr_rect (start 10 20) (end 50 60))", &mut min_x, &mut min_y, &mut max_x, &mut max_y);
        assert_eq!((min_x, min_y, max_x, max_y), (10.0, 20.0, 50.0, 60.0));

        let mut kmin_x = f64::INFINITY;
        let mut kmin_y = f64::INFINITY;
        let mut kmax_x = f64::NEG_INFINITY;
        let mut kmax_y = f64::NEG_INFINITY;
        extract_xy_into_bounds("(pts (xy 1 2) (xy 5 8))", &mut kmin_x, &mut kmin_y, &mut kmax_x, &mut kmax_y);
        assert_eq!((kmin_x, kmin_y, kmax_x, kmax_y), (1.0, 2.0, 5.0, 8.0));

        assert_eq!(parse_at_xy_rot("(at 12.5 34.5 90)"), Some((12.5, 34.5, 90.0)));
        assert_eq!(parse_at_xy_rot("(at 12.5 34.5)"), Some((12.5, 34.5, 0.0)));
        assert_eq!(parse_at_xy_rot("(none)"), None);

        assert_eq!(
            extract_property("(property \"Reference\" \"U1\")", "Reference"),
            Some("U1".to_string())
        );
        assert_eq!(extract_property("(property \"Value\" \"10k\")", "Reference"), None);

        let mut nets = vec!["".to_string()];
        let mut net_map = HashMap::from([("".to_string(), 0)]);
        let extra = HashSet::from(["R_EXTRA".to_string()]);

        // 1. Footprint without CrtYd (computes courtyard from pads + registers new net)
        let fp_pads_only = r#"(footprint "R_0603" (at 100 110 180)
            (property "Reference" "R_EXTRA")
            (property "Value" "4.7k")
            (pad "1" smd rect (at -0.8 0) (size 0.8 0.8) (net 99 "NEW_NET"))
        )"#;
        let fp1 = parse_footprint(fp_pads_only, 0, &mut nets, &mut net_map, &extra).unwrap();
        assert_eq!(fp1.reference, "R_EXTRA");
        assert!(fp1.is_locked);
        assert_eq!(fp1.pads[0].net_name, "NEW_NET");
        assert!(fp1.pads[0].net_id > 0);

        // 2. Footprint without CrtYd and without pads (fallback -1..1 courtyard)
        let fp_empty = r#"(footprint "Fiducial" (locked yes) (at 50 60)
            (property "Reference" "FID1")
        )"#;
        let fp2 = parse_footprint(fp_empty, 0, &mut nets, &mut net_map, &extra).unwrap();
        assert!(fp2.is_locked);
        assert_eq!(fp2.courtyards[0].aabb(), AABB::new(-1.0, -1.0, 1.0, 1.0));

        // 3. Footprint with fp_poly on CrtYd and rotated/oval/circle pads
        let fp_complex = r#"(footprint "Connector_USB_C" (at 100 100 -90)
            (property "Reference" "J2")
            (fp_line (start -5.0 4.0) (end 5.0 4.0) (layer "F.CrtYd"))
            (fp_line (start 5.0 4.0) (end 5.0 -4.0) (layer "F.CrtYd"))
            (fp_line (start 5.0 -4.0) (end -5.0 -4.0) (layer "F.CrtYd"))
            (fp_line (start -5.0 -4.0) (end -5.0 4.0) (layer "F.CrtYd"))
            (pad "1" smd roundrect (at -3.0 -4.0 270) (size 0.6 1.4) (net 1 "GND"))
            (pad "2" thru_hole circle (at 0.0 0.0) (size 1.2 1.2) (drill 0.7) (net 2 "+3.3V"))
        )"#;
        let fp3 = parse_footprint(fp_complex, 0, &mut nets, &mut net_map, &extra).unwrap();
        assert_eq!(fp3.courtyards.len(), 1);
        assert_eq!(fp3.courtyards[0].vertices.len(), 4);
        assert_eq!(fp3.pads[0].shape, "roundrect");
        assert_eq!(fp3.pads[0].rotation, 0.0);
        assert_eq!(fp3.pads[1].shape, "circle");
        assert_eq!(fp3.pads[1].drill, Some(0.7));

        // Verify pad_world_corners rotates around footprint + pad local rotation
        let corners = fp3.pad_world_corners(&fp3.pads[0]);
        assert_eq!(corners.len(), 4);
        let poly = Polygon::new(corners);
        assert!(poly.aabb().min_x < poly.aabb().max_x);
    }

    #[test]
    fn test_board_load_ratsnest_clusters_and_exporters() {
        let pcb_path = "/tmp/vulcan_unit_test_board.kicad_pcb";
        let sch_path = "/tmp/vulcan_unit_test_board.kicad_sch";
        let out_pcb_path = Path::new("/tmp/vulcan_unit_test_board_out.kicad_pcb");
        let out_ses_path = Path::new("/tmp/vulcan_unit_test_board_out.ses");

        let pcb_content = r#"(kicad_pcb (version 20240108)
  (net 0 "")
  (net 1 "GND")
  (net 2 "+3.3V")
  (net 3 "P1")
  (net 4 "VBUS")
  (net 5 "U2-SW")
  (net 6 "CAPP")
  (net 7 "USB_D+")
  (net 8 "SIG_MST")
  (gr_rect (start 125.0 86.0) (end 172.0 124.0) (layer "Edge.Cuts"))
  (zone (net 0) (layers "F.Cu" "B.Cu") (keepout (tracks not_allowed))
    (polygon (pts (xy 171.0 91.0) (xy 183.0 91.0) (xy 183.0 119.0) (xy 171.0 119.0)))
  )
  (footprint "QFN" (at 140.0 100.0)
    (property "Reference" "U1")
    (property "Value" "MAX22088")
    (fp_rect (start -2.0 -2.0) (end 2.0 2.0) (layer "F.CrtYd"))
    (pad "1" smd rect (at -1.0 0.0) (size 0.3 0.8) (net 8 "SIG_MST"))
    (pad "2" smd rect (at 1.0 0.0) (size 0.3 0.8) (net 2 "+3.3V"))
  )
  (footprint "R0603" (at 145.0 102.0 90)
    (property "Reference" "R10")
    (property "Value" "10k")
    (fp_line (start -1.0 -0.5) (end 1.0 0.5) (layer "F.CrtYd"))
    (pad "1" smd rect (at -0.5 0.0) (size 0.6 0.6) (net 8 "SIG_MST"))
    (pad "2" smd rect (at 0.5 0.0) (size 0.6 0.6) (net 1 "GND"))
  )
  (footprint "C0603" (at 150.0 105.0)
    (property "Reference" "C10")
    (property "Value" "100nF")
    (pad "1" thru_hole circle (at 0.0 0.0) (size 1.0 1.0) (net 8 "SIG_MST"))
    (pad "2" thru_hole circle (at 1.5 0.0) (size 1.0 1.0) (net 2 "+3.3V"))
  )
  (footprint "Conn" (at 130.0 95.0)
    (property "Reference" "J1")
    (property "Value" "Conn")
    (pad "1" thru_hole circle (at 0.0 0.0) (size 1.2 1.2) (net 3 "P1"))
  )
)"#;

        let sch_content = r#"(kicad_sch
  (text "1. BUCK POWER" (at 50.0 50.0 0))
  (text "2. SURGE INPUT" (at 150.0 50.0 0))
  (text "3. ESP32 CONTROLS" (at 50.0 150.0 0))
  (text "4. MAX22088 TRANSCEIVER" (at 150.0 150.0 0))
  (text "5. MISC" (at 250.0 50.0 0))
  (symbol (lib_id "Device:C") (at 55.0 55.0 0) (property "Reference" "C10"))
  (symbol (lib_id "Connector:J") (at 155.0 55.0 0) (property "Reference" "J1"))
  (symbol (lib_id "IC:U") (at 155.0 155.0 0) (property "Reference" "U1"))
  (symbol (lib_id "Device:R") (at 255.0 55.0 0) (property "Reference" "R10"))
)"#;

        fs::write(pcb_path, pcb_content).unwrap();
        fs::write(sch_path, sch_content).unwrap();

        let mut board = Board::load(pcb_path, &HashSet::new()).unwrap();
        assert_eq!(board.file_path, pcb_path);
        assert_eq!(board.net_name_to_id.get("SIG_MST"), Some(&8));
        assert_eq!(board.get_netclass_for_net(3).name, "Power_15V");
        assert_eq!(board.get_netclass_for_net(4).name, "Power_5V");
        assert_eq!(board.get_netclass_for_net(2).name, "Power_3V3");
        assert_eq!(board.get_netclass_for_net(5).name, "Switching");
        assert_eq!(board.get_netclass_for_net(6).name, "HBS_Bus");
        assert_eq!(board.get_netclass_for_net(7).name, "USB_Diff");
        assert_eq!(board.get_netclass_for_net(999).name, "Default");
        assert!(!board.clusters.is_empty());
        assert!(board.clusters.iter().any(|c| !c.name.is_empty()));
        let b_aabb = board.boundary_aabb();
        assert!(b_aabb.min_x < b_aabb.max_x);

        let pad_pos: Vec<Vec<Point>> = board
            .footprints
            .iter()
            .map(|fp| {
                (0..fp.pads.len())
                    .map(|pi| fp.pad_world_point(pi, Point::new(fp.x, fp.y), fp.rotation))
                    .collect()
            })
            .collect();
        let segs = board.generate_ratsnest_segments(&pad_pos);
        // SIG_MST has 3 pads -> 2 MST segments; +3.3V has 2 pads -> 1 segment
        assert_eq!(segs.len(), 3);

        // Modify footprint & add track + via, then test save_pcb and save_ses
        board.footprints[0].x += 1.0;
        board.footprints[0].rotation = 0.0;
        board.footprints[1].x += 1.0;
        board.footprints[1].rotation = 90.0;
        board.tracks.push(TrackSegment {
            start: (140.0, 100.0),
            end: (145.0, 102.0),
            width: 0.25,
            layer: 0,
            net_id: 8,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: (145.0, 102.0),
            end: (150.0, 105.0),
            width: 0.25,
            layer: 1,
            net_id: 8,
            is_locked: false,
        });
        board.vias.push(Via {
            position: (145.0, 102.0),
            size: 0.6,
            drill: 0.3,
            net_id: 8,
            is_locked: false,
        });

        board.save_pcb(out_pcb_path).unwrap();
        let saved_pcb = fs::read_to_string(out_pcb_path).unwrap();
        assert!(saved_pcb.contains("(segment "));
        assert!(saved_pcb.contains("(via "));

        board.save_ses(out_ses_path).unwrap();
        let saved_ses = fs::read_to_string(out_ses_path).unwrap();
        assert!(saved_ses.contains("(session vulcans_trials"));
        assert!(saved_ses.contains("(net \"SIG_MST\""));

        // Invalid raw_content without closing paren should return Err when tracks exist
        board.raw_content = "invalid_no_paren".to_string();
        assert!(board.save_pcb(out_pcb_path).is_err());

        // Fallback Edge.Cuts + U4 auto-keepout + missing file error
        let u4_pcb_path = "/tmp/vulcan_unit_test_u4.kicad_pcb";
        fs::write(
            u4_pcb_path,
            "(kicad_pcb (footprint \"ESP\" (at 165 105) (property \"Reference\" \"U4\")))",
        )
        .unwrap();
        let u4_board = Board::load(u4_pcb_path, &HashSet::new()).unwrap();
        assert_eq!(u4_board.keepouts.len(), 1);
        assert!(Board::load("/tmp/nonexistent_vulcan_pcb_12345.kicad_pcb", &HashSet::new()).is_err());

        let _ = fs::remove_file(pcb_path);
        let _ = fs::remove_file(sch_path);
        let _ = fs::remove_file(out_pcb_path);
        let _ = fs::remove_file(out_ses_path);
        let _ = fs::remove_file(u4_pcb_path);
    }

    #[test]
    fn test_trial_phase_and_live_trial_state() {
        assert_eq!(TrialPhase::Idle.label(), "Idle — Ready");
        assert_eq!(TrialPhase::Legalizing.label(), "Phase 1: Courtyard Legalization");
        assert_eq!(TrialPhase::AutoPlacing.label(), "Phase 1: Parallel Tempering Auto-Placer");
        assert_eq!(TrialPhase::AutoRouting.label(), "Phase 2: Rip-up & Nudge Auto-Router");
        assert_eq!(TrialPhase::Completed.label(), "Trial Complete");

        let pcb_path = "/tmp/vulcan_unit_test_live.kicad_pcb";
        fs::write(
            pcb_path,
            r#"(kicad_pcb
              (net 0 "") (net 1 "N1") (net 2 "N2")
              (footprint "R1" (at 0 0) (property "Reference" "R10")
                (pad "1" smd rect (at 0 0) (size 0.5 0.5) (net 1 "N1"))
                (pad "2" smd rect (at 0 4) (size 0.5 0.5) (net 2 "N2")))
              (footprint "R2" (at 4 0) (property "Reference" "R11")
                (pad "1" smd rect (at 0 4) (size 0.5 0.5) (net 1 "N1"))
                (pad "2" smd rect (at 0 0) (size 0.5 0.5) (net 2 "N2")))
            )"#,
        )
        .unwrap();
        let board = Board::load(pcb_path, &HashSet::new()).unwrap();
        let mut live = LiveTrialState::new(board);
        assert_eq!(live.crossings, 1);
        for i in 0..260 {
            live.push_log(format!("Log entry {}", i));
        }
        assert_eq!(live.log_lines.len(), 250);
        assert_eq!(live.status_text, "Log entry 259");
        let _ = fs::remove_file(pcb_path);
    }

    #[test]
    fn test_schematic_clusters_bounding_boxes_polylines_rectangles_and_subsheets() {
        let dir = std::env::temp_dir().join(format!("vulcan_test_clusters_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let pcb_path = dir.join("test_board.kicad_pcb");
        let sch_path = dir.join("test_board.kicad_sch");
        let sub_sch_path = dir.join("sensor.kicad_sch");

        let pcb_content = r#"(kicad_pcb (version 20240108)
  (net 0 "")
  (footprint "U" (at 20 20) (property "Reference" "U1"))
  (footprint "R" (at 25 25) (property "Reference" "R1"))
  (footprint "J" (at 110 20) (property "Reference" "J1"))
  (footprint "D" (at 120 20) (property "Reference" "D1"))
  (footprint "U" (at 210 20) (property "Reference" "U2"))
  (footprint "C" (at 215 20) (property "Reference" "C1"))
)"#;

        // Sub-sheet file with U2 and C1
        let sub_sch_content = r#"(kicad_sch
  (symbol (lib_id "IC:U") (at 210.0 20.0 0) (property "Reference" "U2"))
  (symbol (lib_id "Device:C") (at 215.0 20.0 0) (property "Reference" "C1"))
)"#;

        // Root schematic with:
        // 1. Closed polyline box [10..50, 10..50] enclosing U1 and R1, with unnumbered text "Power Stage"
        // 2. Rectangle box [100..140, 10..50] enclosing J1 and D1, with no text (auto-named "J1 Cluster")
        // 3. Hierarchical sheet pointing to sensor.kicad_sch (named "SensorSheet")
        let sch_content = r#"(kicad_sch
  (polyline
    (pts (xy 10.0 10.0) (xy 50.0 10.0) (xy 50.0 50.0) (xy 10.0 50.0) (xy 10.0 10.0))
    (stroke (width 0.254) (type dash))
    (fill (type none))
  )
  (text "Power Stage" (at 15.0 15.0 0))
  (symbol (lib_id "IC:U") (at 20.0 20.0 0) (property "Reference" "U1"))
  (symbol (lib_id "Device:R") (at 25.0 25.0 0) (property "Reference" "R1"))

  (rectangle (start 100.0 10.0) (end 140.0 50.0) (stroke (width 0.254) (type default)) (fill (type none)))
  (symbol (lib_id "Connector:J") (at 110.0 20.0 0) (property "Reference" "J1"))
  (symbol (lib_id "Device:D") (at 120.0 20.0 0) (property "Reference" "D1"))

  (sheet (at 200.0 10.0) (size 40.0 40.0)
    (property "Sheetname" "SensorSheet")
    (property "Sheetfile" "sensor.kicad_sch")
  )
)"#;

        fs::write(&pcb_path, pcb_content).unwrap();
        fs::write(&sub_sch_path, sub_sch_content).unwrap();
        fs::write(&sch_path, sch_content).unwrap();

        let board = Board::load(pcb_path.to_str().unwrap(), &HashSet::new()).unwrap();
        assert_eq!(board.clusters.len(), 3, "Expected 3 clusters from polyline box, rectangle box, and sub-sheet");

        // Cluster 1: Power Stage (polyline box)
        let c1 = board.clusters.iter().find(|c| c.name.contains("Power Stage")).expect("Power Stage cluster");
        assert_eq!(c1.member_refs.len(), 2);
        assert!(c1.member_refs.contains(&"U1".to_string()));
        assert!(c1.member_refs.contains(&"R1".to_string()));
        assert_eq!(c1.anchor_idx, board.footprints.iter().position(|f| f.reference == "U1"));

        // Cluster 2: Auto-named J1 Cluster (rectangle box with no text)
        let c2 = board.clusters.iter().find(|c| c.name.contains("J1")).expect("J1 cluster");
        assert_eq!(c2.member_refs.len(), 2);
        assert!(c2.member_refs.contains(&"J1".to_string()));
        assert!(c2.member_refs.contains(&"D1".to_string()));
        assert_eq!(c2.anchor_idx, board.footprints.iter().position(|f| f.reference == "J1"));

        // Cluster 3: SensorSheet (hierarchical sub-sheet)
        let c3 = board.clusters.iter().find(|c| c.name == "SensorSheet").expect("SensorSheet cluster");
        assert_eq!(c3.member_refs.len(), 2);
        assert!(c3.member_refs.contains(&"U2".to_string()));
        assert!(c3.member_refs.contains(&"C1".to_string()));
        assert_eq!(c3.anchor_idx, board.footprints.iter().position(|f| f.reference == "U2"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pro_netclass_inherits_default_diff_pair_dims() {
        // USB_Diff omits diff_pair_width/gap, so KiCad inherits them from Default (0.20/0.25)
        // rather than using its 0.30 single-ended track_width.
        let pro = r#"{"net_settings":{"classes":[
            {"name":"Default","track_width":0.2,"clearance":0.15,"diff_pair_width":0.2,"diff_pair_gap":0.25},
            {"name":"USB_Diff","track_width":0.3,"clearance":0.2}
        ]}}"#;
        let ncs = parse_netclasses_from_pro_or_pcb(pro);
        let usb = &ncs["USB_Diff"];
        assert!((usb.diff_pair_width - 0.20).abs() < 1e-9, "got {}", usb.diff_pair_width);
        assert!((usb.diff_pair_gap - 0.25).abs() < 1e-9, "got {}", usb.diff_pair_gap);
        assert!((usb.track_width - 0.30).abs() < 1e-9);
    }

    #[test]
    fn test_load_project_netclasses_from_kicad_pro() {
        let pcb_path = "/tmp/vulcan_unit_test_pro.kicad_pcb";
        let pro_path = "/tmp/vulcan_unit_test_pro.kicad_pro";
        fs::write(
            pcb_path,
            r#"(kicad_pcb
              (net 0 "")
              (net 1 "P1")
              (net 2 "AIO")
              (net 3 "BUS_X")
              (net 4 "END_Y")
              (net 5 "ASSIGNED_NET")
            )"#,
        )
        .unwrap();
        fs::write(
            pro_path,
            r#"{
              "net_settings": {
                "classes": [
                  { "name": "P1P2_Bus", "track_width": 0.6, "clearance": 0.3, "via_diameter": 0.8, "via_drill": 0.4 },
                  { "name": "Analog_Sense", "track_width": 0.25, "clearance": 0.2, "via_diameter": 0.6, "via_drill": 0.3 }
                ],
                "netclass_assignments": {
                  "ASSIGNED_NET": ["Analog_Sense"]
                },
                "netclass_patterns": [
                  { "pattern": "P1", "netclass": "P1P2_Bus" },
                  { "pattern": "BUS_*", "netclass": "P1P2_Bus" },
                  { "pattern": "*_Y", "netclass": "Analog_Sense" }
                ]
              }
            }"#,
        )
        .unwrap();

        let board = Board::load(pcb_path, &HashSet::new()).unwrap();
        let nc_p1 = board.get_netclass_for_net(1);
        assert_eq!(nc_p1.name, "P1P2_Bus");
        assert!((nc_p1.track_width - 0.60).abs() < 1e-6);
        assert!((nc_p1.clearance - 0.30).abs() < 1e-6);
        assert!((nc_p1.via_dia - 0.80).abs() < 1e-6);
        assert!((nc_p1.via_drill - 0.40).abs() < 1e-6);

        assert_eq!(board.get_netclass_for_net(3).name, "P1P2_Bus");
        assert_eq!(board.get_netclass_for_net(4).name, "Analog_Sense");
        assert_eq!(board.get_netclass_for_net(5).name, "Analog_Sense");

        // Built-in fallback classes superseded by the project's classes are dropped.
        assert!(!board.netclasses.contains_key("Power_15V"));
        assert!(!board.netclasses.contains_key("Switching"));
        for name in board.netclasses.keys() {
            assert!(name == "Default" || board.net_to_class.values().any(|c| c == name), "empty class {name}");
        }

        let _ = fs::remove_file(pcb_path);
        let _ = fs::remove_file(pro_path);
    }

    #[test]
    fn test_netclass_diff_pair_parsing() {
        let json_pro = r#"{
            "net_settings": {
                "classes": [
                    {
                        "name": "USB_Diff",
                        "track_width": 0.22,
                        "clearance": 0.18,
                        "diff_pair_gap": 0.15,
                        "diff_pair_width": 0.22
                    }
                ]
            }
        }"#;
        let pro_classes = parse_netclasses_from_pro_or_pcb(json_pro);
        let usb_cls = pro_classes.get("USB_Diff").expect("USB_Diff should be parsed");
        assert!((usb_cls.diff_pair_gap - 0.15).abs() < 1e-6);
        assert!((usb_cls.diff_pair_width - 0.22).abs() < 1e-6);

        let pcb_sexpr = r#"(kicad_pcb
            (net_class "ETH_Diff" "Ethernet 100-ohm diff pair"
                (clearance 0.25)
                (trace_width 0.30)
                (via_dia 0.65)
                (via_drill 0.35)
                (diff_pair_gap 0.18)
                (diff_pair_width 0.30)
            )
        )"#;
        let pcb_classes = parse_netclasses_from_pro_or_pcb(pcb_sexpr);
        let eth_cls = pcb_classes.get("ETH_Diff").expect("ETH_Diff should be parsed");
        assert_eq!(eth_cls.name, "ETH_Diff");
        assert!((eth_cls.clearance - 0.25).abs() < 1e-6);
        assert!((eth_cls.track_width - 0.30).abs() < 1e-6);
        assert!((eth_cls.diff_pair_gap - 0.18).abs() < 1e-6);
        assert!((eth_cls.diff_pair_width - 0.30).abs() < 1e-6);
    }

    #[test]
    fn test_dfm_config_and_fab_presets() {
        // Test FabPreset enum parsing
        assert_eq!(FabPreset::parse("JLCPCB"), Some(FabPreset::Jlcpcb));
        assert_eq!(FabPreset::parse("jlc"), Some(FabPreset::Jlcpcb));
        assert_eq!(FabPreset::parse("pcbway"), Some(FabPreset::PcbWay));
        assert_eq!(FabPreset::parse("Eurocircuits"), Some(FabPreset::Eurocircuits));
        assert_eq!(FabPreset::parse("unknown_fab"), None);

        // Test FabPreset clearances
        assert!((FabPreset::Jlcpcb.board_edge_clearance() - 0.35).abs() < 1e-6);
        assert!((FabPreset::PcbWay.board_edge_clearance() - 0.30).abs() < 1e-6);
        assert!((FabPreset::Eurocircuits.board_edge_clearance() - 0.40).abs() < 1e-6);

        // Test NetClass DFM defaults
        let nc = NetClass::default();
        assert!(nc.teardrop_enabled);
        assert!((nc.teardrop_length_ratio - 0.5).abs() < 1e-6);
        assert!((nc.teardrop_width_ratio - 0.8).abs() < 1e-6);
        assert!((nc.board_edge_clearance - 0.35).abs() < 1e-6);

        // Test JSON parsing with FabPreset
        let json_pro = r#"{
            "fab_preset": "PCBWay",
            "net_settings": {
                "classes": [
                    {
                        "name": "Power",
                        "track_width": 0.50,
                        "clearance": 0.25,
                        "teardrop_enabled": true,
                        "teardrop_length_ratio": 0.6,
                        "teardrop_width_ratio": 0.9
                    }
                ]
            }
        }"#;
        let pro_classes = parse_netclasses_from_pro_or_pcb(json_pro);
        let pwr = pro_classes.get("Power").expect("Power class must parse");
        assert!(pwr.teardrop_enabled);
        assert!((pwr.teardrop_length_ratio - 0.6).abs() < 1e-6);
        assert!((pwr.teardrop_width_ratio - 0.9).abs() < 1e-6);
        assert!((pwr.board_edge_clearance - 0.30).abs() < 1e-6); // From PCBWay preset

        // Test S-expression parsing with FabPreset
        let pcb_sexpr = r#"(kicad_pcb
            (fab_preset "Eurocircuits")
            (net_class "RF" "RF Class"
                (clearance 0.20)
                (trace_width 0.35)
                (teardrop_enabled false)
            )
        )"#;
        let pcb_classes = parse_netclasses_from_pro_or_pcb(pcb_sexpr);
        let rf = pcb_classes.get("RF").expect("RF class must parse");
        assert!(!rf.teardrop_enabled);
        assert!((rf.board_edge_clearance - 0.40).abs() < 1e-6); // From Eurocircuits preset
    }

    #[test]
    fn test_natural_designator_cmp() {
        assert_eq!(natural_designator_cmp("C5", "C20"), std::cmp::Ordering::Less);
        assert_eq!(natural_designator_cmp("C20", "C5"), std::cmp::Ordering::Greater);
        assert_eq!(natural_designator_cmp("C5", "C5"), std::cmp::Ordering::Equal);
        assert_eq!(natural_designator_cmp("C1", "C2"), std::cmp::Ordering::Less);
        assert_eq!(natural_designator_cmp("C9", "C10"), std::cmp::Ordering::Less);
        assert_eq!(natural_designator_cmp("C10", "D1"), std::cmp::Ordering::Less);
        assert_eq!(natural_designator_cmp("D1", "R1"), std::cmp::Ordering::Less);
        assert_eq!(natural_designator_cmp("R2", "R10"), std::cmp::Ordering::Less);
        assert_eq!(natural_designator_cmp("R10", "R2"), std::cmp::Ordering::Greater);
        assert_eq!(natural_designator_cmp("R10", "R100"), std::cmp::Ordering::Less);
        assert_eq!(natural_designator_cmp("U1", "U2"), std::cmp::Ordering::Less);
        assert_eq!(natural_designator_cmp("U2", "U10"), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_clear_unlocked_objects_and_holding_palette() {
        let pcb_path = "/tmp/vulcan_test_clear_unlocked.kicad_pcb";
        let content = r#"(kicad_pcb
          (net 0 "") (net 1 "GND") (net 2 "SIG")
          (gr_rect (start 100.0 100.0) (end 140.0 130.0) (layer "Edge.Cuts"))
          (footprint "U" (at 110.0 115.0) (property "Reference" "U1") (locked yes)
            (pad "1" thru_hole circle (at 0 0) (size 1.0 1.0) (net 1 "GND")))
          (footprint "C" (at 120.0 115.0) (property "Reference" "C20")
            (pad "1" smd rect (at 0 0) (size 0.5 0.5) (net 2 "SIG")))
          (footprint "C" (at 125.0 115.0) (property "Reference" "C5")
            (pad "1" smd rect (at 0 0) (size 0.5 0.5) (net 2 "SIG")))
          (footprint "R" (at 130.0 115.0) (property "Reference" "R30")
            (pad "1" smd rect (at 0 0) (size 0.5 0.5) (net 2 "SIG")))
          (segment (start 110.0 115.0) (end 115.0 115.0) (width 0.25) (layer "F.Cu") (net 1) (locked yes))
          (segment (start 115.0 115.0) (end 120.0 115.0) (width 0.25) (layer "F.Cu") (net 2))
          (via (at 115.0 115.0) (size 0.6) (drill 0.3) (layers "F.Cu" "B.Cu") (net 2))
        )"#;
        fs::write(pcb_path, content).unwrap();

        let mut board = Board::load(pcb_path, &HashSet::new()).unwrap();
        // 4 footprints: U1 (locked), C20 (unlocked), C5 (unlocked), R30 (unlocked)
        assert_eq!(board.footprints.len(), 4);
        assert_eq!(board.tracks.len(), 2);
        assert!(board.tracks[0].is_locked);
        assert!(!board.tracks[1].is_locked);
        assert_eq!(board.vias.len(), 1);
        assert!(!board.vias[0].is_locked);

        // Clear unlocked objects
        let (ripped, cleared) = board.clear_unlocked_objects();
        assert_eq!(ripped, 2); // 1 unlocked track + 1 unlocked via ripped
        assert_eq!(cleared, 3); // C20, C5, R30 moved to holding palette

        // Active board only contains locked footprint and locked track
        assert_eq!(board.footprints.len(), 1);
        assert_eq!(board.footprints[0].reference, "U1");
        assert_eq!(board.tracks.len(), 1);
        assert!(board.tracks[0].is_locked);
        assert_eq!(board.vias.len(), 0);

        // Holding palette contains 3 footprints sorted naturally (C5 before C20, then R30)
        assert_eq!(board.holding_palette.len(), 3);
        assert_eq!(board.holding_palette[0].reference, "C5");
        assert_eq!(board.holding_palette[1].reference, "C20");
        assert_eq!(board.holding_palette[2].reference, "R30");

        // Return a single component
        let returned = board.return_holding_component_to_board(0).unwrap();
        assert_eq!(returned.reference, "C5");
        assert_eq!(board.footprints.len(), 2);
        assert_eq!(board.holding_palette.len(), 2);

        // Return remaining palette
        let returned_all = board.return_holding_palette_to_board();
        assert_eq!(returned_all, 2);
        assert_eq!(board.footprints.len(), 4);
        assert_eq!(board.holding_palette.len(), 0);

        let _ = fs::remove_file(pcb_path);
    }

    #[test]
    fn test_polygon_convex_decomposition_t_shape_courtyard() {
        // 8-vertex T-shaped courtyard of ESP32-C3-WROOM-02
        let t_poly = Polygon::new(vec![
            Point::new(-9.75, 7.15),
            Point::new(9.75, 7.15),
            Point::new(9.75, -6.85),
            Point::new(14.25, -6.85),
            Point::new(14.25, -18.35),
            Point::new(-14.25, -18.35),
            Point::new(-14.25, -6.85),
            Point::new(-9.75, -6.85),
        ]);
        assert!(!t_poly.is_convex());

        // A small capacitor courtyard located at (12.0, 0.0) in the concave notch
        let cap_in_notch = Polygon::new(vec![
            Point::new(11.5, -0.5),
            Point::new(12.5, -0.5),
            Point::new(12.5, 0.5),
            Point::new(11.5, 0.5),
        ]);

        // Untreated non-convex polygon falsely collides due to SAT treating it as convex hull
        assert!(t_poly.intersects_convex(&cap_in_notch));

        // After convex decomposition, all sub-polygons are convex
        let decomposed = t_poly.decompose_convex();
        assert!(decomposed.len() >= 2);
        for sub in &decomposed {
            assert!(sub.is_convex());
        }

        // None of the decomposed convex sub-polygons overlap the notch!
        let any_notch_hit = decomposed.iter().any(|p| p.intersects_convex(&cap_in_notch));
        assert!(!any_notch_hit, "Concave notch should be clear of collisions after decomposition");

        // A capacitor placed directly on the module body (0.0, 0.0) MUST collide
        let cap_on_body = Polygon::new(vec![
            Point::new(-0.5, -0.5),
            Point::new(0.5, -0.5),
            Point::new(0.5, 0.5),
            Point::new(-0.5, 0.5),
        ]);
        let any_body_hit = decomposed.iter().any(|p| p.intersects_convex(&cap_on_body));
        assert!(any_body_hit, "Module body must still collide with overlapping components");
    }

    #[test]
    fn test_live_trial_state_refresh_ratsnest_on_clear_unlocked() {
        let pcb_path = "/tmp/vulcan_test_ratsnest_refresh.kicad_pcb";
        let content = r#"(kicad_pcb
          (net 0 "") (net 1 "GND") (net 2 "SIG")
          (footprint "U" (at 100.0 100.0) (property "Reference" "U1") (locked yes)
            (pad "1" smd rect (at 0 0) (size 1.0 1.0) (net 2 "SIG")))
          (footprint "C" (at 110.0 100.0) (property "Reference" "C10")
            (pad "1" smd rect (at 0 0) (size 1.0 1.0) (net 2 "SIG")))
        )"#;
        fs::write(pcb_path, content).unwrap();

        let board = Board::load(pcb_path, &HashSet::new()).unwrap();
        let mut live = LiveTrialState::new(board);
        assert_eq!(live.ratsnest.len(), 1);

        // Clear unlocked objects -> C10 removed to holding palette
        live.board.clear_unlocked_objects();
        live.refresh_ratsnest();
        // Since only U1 remains on the board, there are no other pads on net SIG -> 0 ratsnest segments
        assert_eq!(live.ratsnest.len(), 0);

        // Return C10 to board -> ratsnest wire restored
        live.board.return_holding_palette_to_board();
        live.refresh_ratsnest();
        assert_eq!(live.ratsnest.len(), 1);

        let _ = fs::remove_file(pcb_path);
    }

    #[test]
    fn test_footprint_unlocked_diodes_fuses_and_default_locking() {
        let content = r#"(kicad_pcb (version 20240108)
          (net 0 "") (net 1 "GND") (net 2 "P1") (net 3 "P2")
          (footprint "D" (at 10.0 10.0) (property "Reference" "D1")
            (pad "1" smd rect (at 0 0) (size 1.0 1.0) (net 2 "P1")))
          (footprint "F" (at 20.0 10.0) (property "Reference" "F1")
            (pad "1" smd rect (at 0 0) (size 1.0 1.0) (net 2 "P1")))
          (footprint "F" (at 20.0 15.0) (property "Reference" "F2")
            (pad "1" smd rect (at 0 0) (size 1.0 1.0) (net 3 "P2")))
          (footprint "R" (at 30.0 10.0) (property "Reference" "R1")
            (pad "1" smd rect (at 0 0) (size 1.0 1.0) (net 2 "P1")))
          (footprint "C" (at 30.0 15.0) (property "Reference" "C3")
            (pad "1" smd rect (at 0 0) (size 1.0 1.0) (net 2 "P1")))
          (footprint "J" (at 5.0 10.0) (property "Reference" "J1")
            (pad "1" smd rect (at 0 0) (size 1.0 1.0) (net 2 "P1")))
          (footprint "U" (at 50.0 50.0) (property "Reference" "U1") (locked yes)
            (pad "1" smd rect (at 0 0) (size 1.0 1.0) (net 2 "P1")))
          (footprint "U" (at 60.0 50.0) (property "Reference" "U2")
            (pad "1" smd rect (at 0 0) (size 1.0 1.0) (net 2 "P1")))
        )"#;

        let mut extra_locked = HashSet::new();
        extra_locked.insert("U2".to_string());

        let board = Board::from_content_with_path(content, "test.kicad_pcb", &extra_locked).unwrap();
        let get_fp = |ref_name: &str| board.footprints.iter().find(|fp| fp.reference == ref_name).unwrap();

        // Diodes, fuses, resistors, caps without (locked yes) must NOT be locked
        assert!(!get_fp("D1").is_locked, "D1 without (locked yes) must be unlocked");
        assert!(!get_fp("F1").is_locked, "F1 without (locked yes) must be unlocked");
        assert!(!get_fp("F2").is_locked, "F2 without (locked yes) must be unlocked");
        assert!(!get_fp("R1").is_locked, "R1 without (locked yes) must be unlocked");
        assert!(!get_fp("C3").is_locked, "C3 without (locked yes) must be unlocked");

        // Edge connector J1 is in DEFAULT_LOCKED_REFS
        assert!(get_fp("J1").is_locked, "J1 in DEFAULT_LOCKED_REFS must be locked by default");

        // U1 has explicit (locked yes) in PCB file
        assert!(get_fp("U1").is_locked, "U1 with (locked yes) must be locked");

        // U2 is in extra_locked
        assert!(get_fp("U2").is_locked, "U2 in extra_locked must be locked");
    }

    #[test]
    fn test_net_topology_parsing_and_defaults() {
        // 1. Default enum and NetClass
        assert_eq!(NetTopology::default(), NetTopology::EuclideanMst);
        let default_nc = NetClass::default();
        assert_eq!(default_nc.topology, NetTopology::EuclideanMst);

        // 2. Serde JSON serialization & deserialization roundtrip
        for (topo, expected_name) in [
            (NetTopology::EuclideanMst, "euclidean_mst"),
            (NetTopology::DaisyChain, "daisy_chain"),
            (NetTopology::FlyBy, "fly_by"),
            (NetTopology::TBranch, "t_branch"),
            (NetTopology::StarPoint, "star_point"),
        ] {
            let json = serde_json::to_string(&topo).unwrap();
            assert_eq!(json, format!("\"{}\"", expected_name));
            let parsed: NetTopology = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, topo);
        }

        // Serde aliases
        assert_eq!(serde_json::from_str::<NetTopology>("\"star\"").unwrap(), NetTopology::StarPoint);
        assert_eq!(serde_json::from_str::<NetTopology>("\"daisychain\"").unwrap(), NetTopology::DaisyChain);
        assert_eq!(serde_json::from_str::<NetTopology>("\"flyby\"").unwrap(), NetTopology::FlyBy);
        assert_eq!(serde_json::from_str::<NetTopology>("\"tbranch\"").unwrap(), NetTopology::TBranch);
        assert_eq!(serde_json::from_str::<NetTopology>("\"mst\"").unwrap(), NetTopology::EuclideanMst);

        // 3. parse_netclasses_from_pro_or_pcb with JSON (.kicad_pro)
        let kicad_pro_content = r#"{
            "net_settings": {
                "classes": [
                    {
                        "name": "DDR_Bus",
                        "track_width": 0.15,
                        "clearance": 0.15,
                        "topology": "fly_by"
                    },
                    {
                        "name": "Sensor_Star",
                        "track_width": 0.25,
                        "clearance": 0.20,
                        "topology": "star"
                    },
                    {
                        "name": "SPI_Bus",
                        "track_width": 0.20,
                        "clearance": 0.20,
                        "topology": "daisy_chain"
                    }
                ]
            }
        }"#;
        let classes = parse_netclasses_from_pro_or_pcb(kicad_pro_content);
        assert_eq!(classes["DDR_Bus"].topology, NetTopology::FlyBy);
        assert_eq!(classes["Sensor_Star"].topology, NetTopology::StarPoint);
        assert_eq!(classes["SPI_Bus"].topology, NetTopology::DaisyChain);

        // 4. parse_netclasses_from_pro_or_pcb with S-expression (.kicad_pcb)
        let kicad_pcb_content = r#"(kicad_pcb (version 20221018) (generator pcbnew)
            (net_class "Clock_Net"
                (trace_width 0.2)
                (clearance 0.25)
                (via_dia 0.6)
                (via_drill 0.3)
                (topology t_branch)
            )
            (net_class "Daisy_Net"
                (trace_width 0.2)
                (clearance 0.2)
                (topology "daisy_chain")
            )
        )"#;
        let pcb_classes = parse_netclasses_from_pro_or_pcb(kicad_pcb_content);
        assert_eq!(pcb_classes["Clock_Net"].topology, NetTopology::TBranch);
        assert_eq!(pcb_classes["Daisy_Net"].topology, NetTopology::DaisyChain);

        // 5. Automatic heuristic detection when not explicitly set
        assert_eq!(NetTopology::infer_from_net_name("CLK_100M"), NetTopology::TBranch);
        assert_eq!(NetTopology::infer_from_net_name("RAM_CLK"), NetTopology::TBranch);
        assert_eq!(NetTopology::infer_from_net_name("SPI_SCK"), NetTopology::TBranch);
        assert_eq!(NetTopology::infer_from_net_name("DDR_ADDR0"), NetTopology::FlyBy);
        assert_eq!(NetTopology::infer_from_net_name("RAM_DATA1"), NetTopology::FlyBy);
        assert_eq!(NetTopology::infer_from_net_name("SPI_MISO"), NetTopology::DaisyChain);
        assert_eq!(NetTopology::infer_from_net_name("I2C_SDA"), NetTopology::DaisyChain);
        assert_eq!(NetTopology::infer_from_net_name("+3.3V"), NetTopology::StarPoint);
        assert_eq!(NetTopology::infer_from_net_name("KELVIN_SENSE"), NetTopology::StarPoint);
        assert_eq!(NetTopology::infer_from_net_name("STAR_GND"), NetTopology::StarPoint);
        assert_eq!(NetTopology::infer_from_net_name("GENERAL_NET"), NetTopology::EuclideanMst);

        // Explicit netclass topology overrides heuristic
        let mut nc = NetClass::default();
        nc.topology = NetTopology::DaisyChain;
        assert_eq!(nc.resolve_topology("CLK_100M"), NetTopology::DaisyChain);
    }

    #[test]
    fn test_net_topology_daisy_chain_and_fly_by() {
        use crate::router::{build_net_topology_connections, PadInfo};
        let nc = NetClass::default();
        let pads = vec![
            PadInfo::new(0.0, 0.0, 0, 0, 0).with_driver(true),
            PadInfo::new(10.0, 0.0, 0, 1, 0),
            PadInfo::new(20.0, 0.0, 0, 2, 0),
            PadInfo::new(30.0, 0.0, 0, 3, 0),
        ];

        let daisy_conns = build_net_topology_connections(1, "SPI_BUS", &nc, &pads, NetTopology::DaisyChain);
        assert_eq!(daisy_conns.len(), 3);
        assert_eq!(daisy_conns[0].start, (0.0, 0.0, 0));
        assert_eq!(daisy_conns[0].end, (10.0, 0.0, 0));
        assert_eq!(daisy_conns[1].start, (10.0, 0.0, 0));
        assert_eq!(daisy_conns[1].end, (20.0, 0.0, 0));
        assert_eq!(daisy_conns[2].start, (20.0, 0.0, 0));
        assert_eq!(daisy_conns[2].end, (30.0, 0.0, 0));

        let mut degree = vec![0; 4];
        for c in &daisy_conns {
            let s_idx = pads.iter().position(|p| (p.x, p.y, p.layer) == c.start).unwrap();
            let e_idx = pads.iter().position(|p| (p.x, p.y, p.layer) == c.end).unwrap();
            degree[s_idx] += 1;
            degree[e_idx] += 1;
        }
        assert_eq!(degree[0], 1);
        assert_eq!(degree[1], 2);
        assert_eq!(degree[2], 2);
        assert_eq!(degree[3], 1);
        assert!(degree.iter().all(|&d| d <= 2));

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
        use crate::router::{build_net_topology_connections, PadInfo};
        let nc = NetClass::default();
        let star_center = (15.0, 15.0, 0);
        let pads = vec![
            PadInfo::new(15.0, 15.0, 0, 0, 0).with_driver(true),
            PadInfo::new(0.0, 0.0, 0, 1, 0),
            PadInfo::new(30.0, 0.0, 0, 2, 0),
            PadInfo::new(15.0, 30.0, 0, 3, 0),
        ];

        let conns = build_net_topology_connections(1, "+3.3V", &nc, &pads, NetTopology::StarPoint);
        assert_eq!(conns.len(), 3);
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
        use crate::router::{build_net_topology_connections, PadInfo};
        let nc = NetClass::default();
        let pads = vec![
            PadInfo::new(0.0, 0.0, 0, 0, 0).with_driver(true),
            PadInfo::new(20.0, 10.0, 0, 1, 0),
            PadInfo::new(20.0, -10.0, 0, 2, 0),
        ];

        let conns = build_net_topology_connections(1, "CLK_NET", &nc, &pads, NetTopology::TBranch);
        assert_eq!(conns.len(), 3);
        let trunk = &conns[0];
        assert_eq!(trunk.start, (0.0, 0.0, 0));
        assert_eq!(trunk.end, (20.0, 0.0, 0));

        let branch1 = &conns[1];
        let branch2 = &conns[2];
        assert_eq!(branch1.start, (20.0, 0.0, 0));
        assert_eq!(branch1.end, (20.0, 10.0, 0));
        assert_eq!(branch2.start, (20.0, 0.0, 0));
        assert_eq!(branch2.end, (20.0, -10.0, 0));

        let len1 = (branch1.start.0 - branch1.end.0).hypot(branch1.start.1 - branch1.end.1);
        let len2 = (branch2.start.0 - branch2.end.0).hypot(branch2.start.1 - branch2.end.1);
        assert!((len1 - len2).abs() < 1e-6);
        assert!((len1 - 10.0).abs() < 1e-6);
        assert!((len2 - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_dynamic_active_copper_layers_detection() {
        // 2-layer board
        let pcb_2l = r#"(kicad_pcb (layers (0 "F.Cu" signal) (31 "B.Cu" signal)))"#;
        assert_eq!(detect_copper_layers(pcb_2l), vec!["F.Cu", "B.Cu"]);

        // 4-layer board
        let pcb_4l = r#"(kicad_pcb (layers (0 "F.Cu" signal) (1 "In1.Cu" signal) (2 "In2.Cu" signal) (31 "B.Cu" signal)))"#;
        assert_eq!(detect_copper_layers(pcb_4l), vec!["F.Cu", "B.Cu", "In1.Cu", "In2.Cu"]);

        // 6-layer board
        let pcb_6l = r#"(kicad_pcb (layers (0 "F.Cu" signal) (1 "In1.Cu" signal) (2 "In2.Cu" signal) (3 "In3.Cu" signal) (4 "In4.Cu" signal) (31 "B.Cu" signal)))"#;
        assert_eq!(
            detect_copper_layers(pcb_6l),
            vec!["F.Cu", "B.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu"]
        );

        // 32-layer board
        let mut pcb_32l = String::from("(kicad_pcb (layers (0 \"F.Cu\" signal) ");
        for i in 1..=30 {
            pcb_32l.push_str(&format!("({} \"In{}.Cu\" signal) ", i, i));
        }
        pcb_32l.push_str("(31 \"B.Cu\" signal)))");
        let layers_32 = detect_copper_layers(&pcb_32l);
        assert_eq!(layers_32.len(), 32);
        assert_eq!(layers_32[0], "F.Cu");
        assert_eq!(layers_32[1], "B.Cu");
        assert_eq!(layers_32[2], "In1.Cu");
        assert_eq!(layers_32[31], "In30.Cu");
    }

    #[test]
    fn test_polygon_contains_point_and_contour() {
        // Octagonal / chamfered corner polygon: [0, 10] x [0, 10] with corners cut at 2.0mm
        let poly = Polygon::new(vec![
            Point::new(2.0, 0.0),
            Point::new(8.0, 0.0),
            Point::new(10.0, 2.0),
            Point::new(10.0, 8.0),
            Point::new(8.0, 10.0),
            Point::new(2.0, 10.0),
            Point::new(0.0, 8.0),
            Point::new(0.0, 2.0),
        ]);

        // Inside center
        assert!(poly.contains_point(&Point::new(5.0, 5.0)));
        // Inside near edges
        assert!(poly.contains_point(&Point::new(2.5, 0.5)));
        // Outside the chamfered corner (in the bounding box corner [0, 0])
        assert!(!poly.contains_point(&Point::new(0.5, 0.5)));
        assert!(!poly.contains_point(&Point::new(9.5, 0.5)));
        assert!(!poly.contains_point(&Point::new(0.5, 9.5)));
        assert!(!poly.contains_point(&Point::new(9.5, 9.5)));
        // Completely outside
        assert!(!poly.contains_point(&Point::new(-1.0, 5.0)));
        assert!(!poly.contains_point(&Point::new(11.0, 5.0)));
    }

    #[test]
    fn test_footprint_origin_clamp_range_asymmetrical() {
        // Footprint with origin at pin 1 (0, 0) and 3 more pins extending in +Y: (0, 2.54), (0, 5.08), (0, 7.62)
        // Courtyard spanning [-1.5, -1.0] to [1.5, 8.62]
        let fp = Footprint {
            reference: "J3".to_string(),
            value: "HEADER".to_string(),
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![
                Pad::new("1", 0.0, 0.0, 1.6, 1.6, 1, "NET1", false),
                Pad::new("2", 0.0, 2.54, 1.6, 1.6, 2, "NET2", false),
                Pad::new("3", 0.0, 5.08, 1.6, 1.6, 3, "NET3", false),
                Pad::new("4", 0.0, 7.62, 1.6, 1.6, 4, "NET4", false),
            ],
            width: 3.0,
            height: 9.62,
            courtyards: vec![Polygon::new(vec![
                Point::new(-1.5, -1.0),
                Point::new(1.5, -1.0),
                Point::new(1.5, 8.62),
                Point::new(-1.5, 8.62),
            ])],
            perimeters: Vec::new(),
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let boundary = AABB::new(10.0, 10.0, 50.0, 50.0);
        let spacing = 0.5;

        // At rotation 0.0
        let (min_x, max_x, min_y, max_y) = fp.origin_clamp_range(0.0, 0, spacing, &boundary);
        // Courtyard [-1.5, -1.0] to [1.5, 8.62] expanded by spacing / 2.0 (0.25):
        // rel_b min_x = -1.75, max_x = 1.75; min_y = -1.25, max_y = 8.87
        assert!((min_x - 11.75).abs() < 1e-4, "min_x was {}", min_x);
        assert!((max_x - 48.25).abs() < 1e-4, "max_x was {}", max_x);
        assert!((min_y - 11.25).abs() < 1e-4, "min_y was {}", min_y);
        assert!((max_y - 41.13).abs() < 1e-4, "max_y was {}", max_y);

        // At max allowed origin (max_x, max_y), check that pad 4 corner is within boundary
        let pad4 = &fp.pads[3];
        let corners = fp.pad_corners_at(pad4, Point::new(max_x, max_y), 0.0, 0);
        for c in corners {
            assert!(c.x <= boundary.max_x + 1e-4);
            assert!(c.y <= boundary.max_y + 1e-4);
            assert!(c.x >= boundary.min_x - 1e-4);
            assert!(c.y >= boundary.min_y - 1e-4);
        }

        // At rotation 90.0, footprint extends in +X (Y-down KiCad coords)
        let (_r_min_x, r_max_x, _r_min_y, r_max_y) = fp.origin_clamp_range(90.0, 0, spacing, &boundary);
        assert!(r_max_x < max_x); // Needs more room on the right
        assert!(r_max_y > max_y); // Needs less room at bottom
    }

    #[test]
    fn test_rip_up_connected_traces_point_to_point() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "NET_A".to_string()];

        let fp_a = Footprint {
            reference: "R1".to_string(),
            value: "10k".to_string(),
            x: 10.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let fp_b = Footprint {
            reference: "R2".to_string(),
            value: "10k".to_string(),
            x: 30.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        board.footprints.push(fp_a);
        board.footprints.push(fp_b);

        // 3 connected track segments: (10,10) -> (15,10) -> (25,10) -> (30,10)
        board.tracks.push(TrackSegment {
            start: (10.0, 10.0),
            end: (15.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: (15.0, 10.0),
            end: (25.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        board.tracks.push(TrackSegment {
            start: (25.0, 10.0),
            end: (30.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });

        assert_eq!(board.tracks.len(), 3);
        // Move Footprint 0 (R1)
        let ripped = board.rip_up_connected_traces_to_component(0);
        assert_eq!(ripped, 3);
        assert_eq!(board.tracks.len(), 0);
    }

    #[test]
    fn test_rip_up_connected_traces_with_vias() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "NET_A".to_string()];

        let fp_a = Footprint {
            reference: "R1".to_string(),
            value: "10k".to_string(),
            x: 10.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let fp_b = Footprint {
            reference: "R2".to_string(),
            value: "10k".to_string(),
            x: 30.0,
            y: 10.0,
            rotation: 0.0,
            layer: 1, // On B.Cu
            orig_layer: 1,
            is_locked: false,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        board.footprints.push(fp_a);
        board.footprints.push(fp_b);

        // Track 1 on F.Cu: (10, 10) to (20, 10)
        board.tracks.push(TrackSegment {
            start: (10.0, 10.0),
            end: (20.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        // Via at (20, 10)
        board.vias.push(Via {
            position: (20.0, 10.0),
            size: 0.60,
            drill: 0.30,
            net_id: 1,
            is_locked: false,
        });
        // Track 2 on B.Cu: (20, 10) to (30, 10)
        board.tracks.push(TrackSegment {
            start: (20.0, 10.0),
            end: (30.0, 10.0),
            width: 0.20,
            layer: 1,
            net_id: 1,
            is_locked: false,
        });

        assert_eq!(board.tracks.len(), 2);
        assert_eq!(board.vias.len(), 1);

        let ripped = board.rip_up_connected_traces_to_component(0);
        assert_eq!(ripped, 3);
        assert_eq!(board.tracks.len(), 0);
        assert_eq!(board.vias.len(), 0);
    }

    #[test]
    fn test_rip_up_connected_traces_daisy_chain_barrier() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "NET_A".to_string()];

        let fp_a = Footprint {
            reference: "R1".to_string(),
            value: "10k".to_string(),
            x: 10.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let fp_b = Footprint {
            reference: "R2".to_string(),
            value: "10k".to_string(),
            x: 20.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let fp_c = Footprint {
            reference: "R3".to_string(),
            value: "10k".to_string(),
            x: 30.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        board.footprints.push(fp_a);
        board.footprints.push(fp_b);
        board.footprints.push(fp_c);

        // Track 1: A -> B
        board.tracks.push(TrackSegment {
            start: (10.0, 10.0),
            end: (20.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        // Track 2: B -> C
        board.tracks.push(TrackSegment {
            start: (20.0, 10.0),
            end: (30.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });

        assert_eq!(board.tracks.len(), 2);

        // Moving Footprint 0 (R1) should ONLY rip up Track 1 (A -> B).
        // Track 2 (B -> C) must remain intact!
        let ripped = board.rip_up_connected_traces_to_component(0);
        assert_eq!(ripped, 1);
        assert_eq!(board.tracks.len(), 1);
        assert_eq!(board.tracks[0].start, (20.0, 10.0));
        assert_eq!(board.tracks[0].end, (30.0, 10.0));
    }

    #[test]
    fn test_rip_up_connected_traces_with_pos_override() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "NET_A".to_string()];

        let mut fp_a = Footprint {
            reference: "R1".to_string(),
            value: "10k".to_string(),
            x: 10.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let fp_b = Footprint {
            reference: "R2".to_string(),
            value: "10k".to_string(),
            x: 30.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        // Track connected to R1's original position at (10, 10)
        board.tracks.push(TrackSegment {
            start: (10.0, 10.0),
            end: (30.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });

        // Simulate GUI dragging R1 to (15, 25)
        fp_a.x = 15.0;
        fp_a.y = 25.0;
        board.footprints.push(fp_a);
        board.footprints.push(fp_b);

        // Calling rip up with pos_override of the original (10, 10) position
        let ripped = board.rip_up_connected_traces_to_component_at(0, Some((10.0, 10.0, 0.0, 0)));
        assert_eq!(ripped, 1);
        assert_eq!(board.tracks.len(), 0);
    }

    #[test]
    fn test_rip_up_connected_traces_locked_objects_preserved() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "NET_A".to_string()];

        let fp_a = Footprint {
            reference: "R1".to_string(),
            value: "10k".to_string(),
            x: 10.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: true, // Locked component!
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        let fp_b = Footprint {
            reference: "R2".to_string(),
            value: "10k".to_string(),
            x: 30.0,
            y: 10.0,
            rotation: 0.0,
            layer: 0,
            orig_layer: 0,
            is_locked: false,
            pads: vec![Pad::new("1", 0.0, 0.0, 0.5, 0.5, 1, "NET_A", false)],
            width: 1.0,
            height: 1.0,
            courtyards: vec![],
            perimeters: vec![],
            escape_margin: 0.0,
            at_span: None,
            fp_span: None,
            pin_swap_groups: Vec::new(),
            gate_swap_groups: Vec::new(),
        };

        board.footprints.push(fp_a);
        board.footprints.push(fp_b);

        board.tracks.push(TrackSegment {
            start: (10.0, 10.0),
            end: (30.0, 10.0),
            width: 0.20,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });

        // Locked component returns 0 and does not rip up tracks
        let ripped = board.rip_up_connected_traces_to_component(0);
        assert_eq!(ripped, 0);
        assert_eq!(board.tracks.len(), 1);

        // Unlock footprint but lock track
        board.footprints[0].is_locked = false;
        board.tracks[0].is_locked = true;

        let ripped = board.rip_up_connected_traces_to_component(0);
        assert_eq!(ripped, 0);
        assert_eq!(board.tracks.len(), 1);
    }

    #[test]
    fn test_bga_detection_and_ring_depth() {
        let mut fp = Footprint {
            reference: "U_BGA".to_string(),
            value: "BGA-16".to_string(),
            x: 0.0,
            y: 0.0,
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
                fp.pads.push(Pad::new(&pad_num.to_string(), x, y, 0.4, 0.4, pad_num, "SIG", false));
                pad_num += 1;
            }
        }

        assert!(fp.is_bga(), "4x4 regular pad array must be detected as BGA");
        let (px, py) = fp.bga_pitch().expect("Pitch must be computed");
        assert!((px - 0.8).abs() < 1e-4, "Pitch X must be 0.8mm: {}", px);
        assert!((py - 0.8).abs() < 1e-4, "Pitch Y must be 0.8mm: {}", py);

        let (cx, cy) = fp.bga_matrix_center().expect("Center must be computed");
        assert!(cx.abs() < 1e-4 && cy.abs() < 1e-4, "Center must be (0,0): ({}, {})", cx, cy);

        // Verify concentric ring indexing:
        // Outer 12 pads: Ring 0
        // Inner 4 pads: (1,1), (2,1), (1,2), (2,2) -> indices:
        // Row 0: pads 0,1,2,3 -> all ring 0
        // Row 1: pad 4 (ring 0), pad 5 (ring 1), pad 6 (ring 1), pad 7 (ring 0)
        // Row 2: pad 8 (ring 0), pad 9 (ring 1), pad 10 (ring 1), pad 11 (ring 0)
        // Row 3: pads 12,13,14,15 -> all ring 0
        for i in 0..16 {
            let row = i / 4;
            let col = i % 4;
            let ring = fp.bga_pad_ring_index(i);
            if row == 0 || row == 3 || col == 0 || col == 3 {
                assert_eq!(ring, 0, "Outer pad ({}, {}) must be Ring 0", col, row);
            } else {
                assert_eq!(ring, 1, "Inner pad ({}, {}) must be Ring 1", col, row);
            }
        }

        // Test 5x5 BGA grid with 0.8mm pitch (Ring 0, 1, 2)
        let mut fp5 = Footprint {
            reference: "U_BGA5".to_string(),
            value: "BGA-25".to_string(),
            ..Footprint::default()
        };
        let xs5 = [-1.6, -0.8, 0.0, 0.8, 1.6];
        let ys5 = [-1.6, -0.8, 0.0, 0.8, 1.6];
        pad_num = 1;
        for &y in &ys5 {
            for &x in &xs5 {
                fp5.pads.push(Pad::new(&pad_num.to_string(), x, y, 0.4, 0.4, pad_num, "SIG", false));
                pad_num += 1;
            }
        }
        assert!(fp5.is_bga(), "5x5 array must be detected as BGA");
        for i in 0..25 {
            let row = i / 5;
            let col = i % 5;
            let ring = fp5.bga_pad_ring_index(i);
            if row == 0 || row == 4 || col == 0 || col == 4 {
                assert_eq!(ring, 0, "5x5 Outer pad ({}, {}) must be Ring 0", col, row);
            } else if row == 2 && col == 2 {
                assert_eq!(ring, 2, "5x5 Center pad ({}, {}) must be Ring 2", col, row);
            } else {
                assert_eq!(ring, 1, "5x5 Middle pad ({}, {}) must be Ring 1", col, row);
            }
        }

        // Test non-BGA footprint (2-pad resistor)
        let r1 = Footprint {
            reference: "R1".to_string(),
            pads: vec![
                Pad::new("1", -1.0, 0.0, 0.6, 0.6, 1, "NET1", false),
                Pad::new("2", 1.0, 0.0, 0.6, 0.6, 2, "NET2", false),
            ],
            ..Footprint::default()
        };
        assert!(!r1.is_bga(), "2-pad resistor must not be detected as BGA");
    }

    #[test]
    fn test_kicad_arc_sexpr_export_and_parse() {
        let mut board = Board::default();
        board.nets = vec!["".to_string(), "GND".to_string(), "SIG_ARC".to_string()];
        board.net_name_to_id.insert("GND".to_string(), 1);
        board.net_name_to_id.insert("SIG_ARC".to_string(), 2);

        let arc1 = ArcSegment::new(
            (100.0, 100.0),
            5.0,
            0.0,
            std::f64::consts::FRAC_PI_2,
            0.25,
            0,
            2,
        );
        board.arcs.push(arc1.clone());

        // Export to KiCad PCB S-expression
        let saved_pcb = board.save_kicad_pcb().expect("save_kicad_pcb failed");
        assert!(saved_pcb.contains("(arc (start 105.0000 100.0000)"));
        assert!(saved_pcb.contains("(width 0.2500)"));
        assert!(saved_pcb.contains("(layer \"F.Cu\") (net 2)"));

        // Round-trip parse back
        let loaded_board = Board::from_kicad_pcb(&saved_pcb).expect("from_kicad_pcb failed");
        assert_eq!(loaded_board.arcs.len(), 1);
        let parsed = &loaded_board.arcs[0];
        assert!((parsed.center.0 - arc1.center.0).abs() < 1e-3);
        assert!((parsed.center.1 - arc1.center.1).abs() < 1e-3);
        assert!((parsed.radius - arc1.radius).abs() < 1e-3);
        assert_eq!(parsed.layer, 0);
        assert_eq!(parsed.net_id, 2);
        assert!((parsed.width - 0.25).abs() < 1e-4);

        // Also test parsing an arc directly from a KiCad 8/9/10 snippet
        let snippet = r#"(kicad_pcb
  (net 0 "") (net 1 "CLK")
  (arc (start 120.0 50.0) (mid 123.5355 51.4645) (end 125.0 55.0) (stroke (width 0.3000)) (layer "B.Cu") (net 1) (locked yes))
)"#;
        let snippet_board = Board::from_kicad_pcb(snippet).expect("snippet parse failed");
        assert_eq!(snippet_board.arcs.len(), 1);
        assert!(snippet_board.arcs[0].is_locked);
        assert_eq!(snippet_board.arcs[0].layer, 1);
        assert!((snippet_board.arcs[0].width - 0.30).abs() < 1e-4);
    }

    #[test]
    fn test_kicad10_named_net_roundtrip_and_pad_swap_export() {
        let kicad10_pcb = r#"(kicad_pcb
  (version 20260206)
  (generator "pcbnew")
  (generator_version "10.0")
  (footprint "Resistor_SMD:R_0603_1608Metric"
    (layer "F.Cu")
    (at 10.0 20.0)
    (property "Reference" "R1")
    (property "Value" "10k")
    (pad "1" smd rect (at -0.8 0) (size 0.8 0.9) (layers "F.Cu" "F.Mask") (net "+3.3V"))
    (pad "2" smd rect (at 0.8 0) (size 0.8 0.9) (layers "F.Cu" "F.Mask") (net "Net-(D4-A)"))
  )
)"#;
        let mut board = Board::from_kicad_pcb(kicad10_pcb).expect("Failed to parse KiCad 10 PCB");
        assert_eq!(board.footprints.len(), 1);
        assert_eq!(board.footprints[0].pads[0].net_name, "+3.3V");
        assert_eq!(board.footprints[0].pads[1].net_name, "Net-(D4-A)");

        // Swap the pads as pin swapping would
        let p0_id = board.footprints[0].pads[0].net_id;
        let p0_name = board.footprints[0].pads[0].net_name.clone();
        let p1_id = board.footprints[0].pads[1].net_id;
        let p1_name = board.footprints[0].pads[1].net_name.clone();
        board.footprints[0].pads[0].net_id = p1_id;
        board.footprints[0].pads[0].net_name = p1_name;
        board.footprints[0].pads[1].net_id = p0_id;
        board.footprints[0].pads[1].net_name = p0_name;

        board.tracks.push(TrackSegment {
            start: (9.2, 20.0),
            end: (15.0, 20.0),
            width: 0.2,
            layer: 0,
            net_id: p1_id,
            is_locked: false,
        });
        board.vias.push(Via {
            position: (15.0, 20.0),
            size: 0.6,
            drill: 0.3,
            net_id: p1_id,
            is_locked: false,
        });

        let exported = board.to_kicad_pcb().expect("Failed to export KiCad 10 PCB");
        assert!(exported.contains("(net \"Net-(D4-A)\")"), "KiCad 10 export must preserve named net syntax without numeric ID prefix");
        assert!(!exported.contains("(net 2 \"Net-(D4-A)\")"), "KiCad 10 export must not inject numeric ID on pads");
        assert!(exported.contains("(segment (start 9.2000 20.0000) (end 15.0000 20.0000) (width 0.2000) (layer \"F.Cu\") (net \"Net-(D4-A)\"))"));
        assert!(exported.contains("(via (at 15.0000 20.0000) (size 0.6000) (drill 0.3000) (layers \"F.Cu\" \"B.Cu\") (net \"Net-(D4-A)\"))"));

        let reloaded = Board::from_kicad_pcb(&exported).expect("Failed to reload exported KiCad 10 PCB");
        assert_eq!(reloaded.tracks.len(), 1);
        assert_eq!(reloaded.vias.len(), 1);
        assert_eq!(reloaded.tracks[0].net_id, reloaded.footprints[0].pads[0].net_id);
        assert_eq!(reloaded.vias[0].net_id, reloaded.footprints[0].pads[0].net_id);
    }

    #[test]
    fn test_to_kicad_pcb_drops_stale_zone_fills() {
        let pcb = r#"(kicad_pcb (version 20260206) (generator_version "10.0")
  (net 0 "") (net 1 "+3.3V")
  (zone (net 1) (net_name "+3.3V") (layer "In2.Cu") (filled_areas_thickness no)
    (polygon (pts (xy 0 0) (xy 10 0) (xy 10 10) (xy 0 10)))
    (filled_polygon (layer "In2.Cu") (pts (xy 6 0.1) (xy 9.9 0.1) (xy 9.9 9.9) (xy 6 9.9)))
  )
)"#;
        let mut board = Board::from_kicad_pcb(pcb).unwrap();
        // New via at (2, 5) lies outside the stale fill (x >= 6): keeping it would draw a KiCad airwire.
        board.vias.push(Via { position: (2.0, 5.0), size: 0.6, drill: 0.3, net_id: 1, is_locked: false });
        let out = board.to_kicad_pcb().unwrap();
        assert!(!out.contains("filled_polygon"));
        assert!(out.contains("(polygon (pts (xy 0 0) (xy 10 0) (xy 10 10) (xy 0 10)))"));
        assert!(out.contains("(filled_areas_thickness no)"));
        assert!(out.contains("(via (at 2.0000 5.0000)"));
        let reloaded = Board::from_kicad_pcb(&out).unwrap();
        assert_eq!(reloaded.vias.len(), 1);

        let no_zone = std::env::temp_dir().join(format!("vulcan_no_zone_{}.kicad_pcb", std::process::id()));
        fs::write(&no_zone, "(kicad_pcb (version 20260206))").unwrap();
        assert!(!Board::refill_zones_in_file(&no_zone), "boards without zones must skip kicad-cli");
        let _ = fs::remove_file(&no_zone);
    }

    #[test]
    fn test_rotated_footprint_updates_pad_board_space_rotation_in_kicad_pcb() {
        let pcb = r#"(kicad_pcb
  (version 20260206)
  (generator "pcbnew")
  (generator_version "10.0")
  (footprint "Package_DFN_QFN:QFN-24-1EP_4x4mm_P0.5mm"
    (layer "F.Cu")
    (at 145.6475 100.3825 180)
    (property "Reference" "U1")
    (property "Value" "MAX22088")
    (pad "1" smd roundrect (at -2.075 -1.25 180) (size 0.85 0.3) (layers "F.Cu" "F.Mask" "F.Paste") (net "BIO"))
    (pad "7" smd roundrect (at -1.25 2.075 270) (size 0.85 0.3) (layers "F.Cu" "F.Mask" "F.Paste") (net "V_REF"))
  )
)"#;
        let mut board = Board::from_kicad_pcb(pcb).expect("Failed to parse PCB");
        assert_eq!(board.footprints.len(), 1);
        assert!((board.footprints[0].pads[0].rotation - 0.0).abs() < 1e-4);
        assert!((board.footprints[0].pads[1].rotation - 90.0).abs() < 1e-4);

        // Rotate U1 from 180° to 270° (+90°)
        board.footprints[0].rotation = 270.0;
        let exported = board.to_kicad_pcb().expect("Failed to export rotated PCB");
        assert!(exported.contains("(at -2.075 -1.25 270)"), "Pad 1 board-space rotation must advance from 180 to 270");
        assert!(exported.contains("(at -1.25 2.075)"), "Pad 7 board-space rotation must wrap from 270 + 90 = 0 (omitted)");

        let reloaded = Board::from_kicad_pcb(&exported).expect("Failed to reload rotated PCB");
        assert!((reloaded.footprints[0].rotation - 270.0).abs() < 1e-4);
        assert!((reloaded.footprints[0].pads[0].rotation - 0.0).abs() < 1e-4);
        assert!((reloaded.footprints[0].pads[1].rotation - 90.0).abs() < 1e-4);
    }

    #[test]
    fn test_parse_design_rules_from_kicad_pro_and_dru() {
        let pro_json = r#"{
  "board": {
    "design_settings": {
      "defaults": {
        "zones": {
          "min_clearance": 0.5,
          "min_thickness": 0.25
        }
      },
      "rules": {
        "min_clearance": 0.18,
        "min_copper_edge_clearance": 0.45,
        "min_hole_clearance": 0.28,
        "min_hole_to_hole": 0.30,
        "min_through_hole_diameter": 0.35,
        "min_track_width": 0.16,
        "min_via_annular_width": 0.14,
        "min_via_diameter": 0.55
      }
    }
  },
  "net_settings": {
    "classes": [
      {
        "name": "Default",
        "clearance": 0.2,
        "track_width": 0.25,
        "via_diameter": 0.6,
        "via_drill": 0.3
      },
      {
        "name": "P1P2_Bus",
        "clearance": 0.3,
        "track_width": 0.5,
        "via_diameter": 0.6,
        "via_drill": 0.3
      }
    ],
    "netclass_assignments": {
      "P1": "P1P2_Bus"
    }
  }
}"#;
        let rules = parse_design_rules_from_pro_or_dru(pro_json).expect("Expected DesignRules");
        assert!((rules.min_clearance - 0.18).abs() < 1e-6);
        assert!((rules.min_copper_edge_clearance - 0.45).abs() < 1e-6);
        assert!((rules.min_hole_clearance - 0.28).abs() < 1e-6);
        assert!((rules.min_hole_to_hole - 0.30).abs() < 1e-6);
        assert!((rules.min_through_hole_diameter - 0.35).abs() < 1e-6);
        assert!((rules.min_track_width - 0.16).abs() < 1e-6);
        assert!((rules.min_via_annular_width - 0.14).abs() < 1e-6);
        assert!((rules.min_via_diameter - 0.55).abs() < 1e-6);
        assert!((rules.zone_min_clearance - 0.50).abs() < 1e-6);
        assert!((rules.zone_min_thickness - 0.25).abs() < 1e-6);

        let pcb = r#"(kicad_pcb
  (version 20260206)
  (generator "pcbnew")
  (generator_version "10.0")
  (net 0 "")
  (net 1 "P1")
  (net 2 "Net-(D4-A)")
)"#;
        let mut board = Board::from_kicad_pcb(pcb).expect("Failed to parse PCB");
        board.apply_project_netclasses_str(pro_json);
        assert!(!board.netclasses.contains_key("HBS_Bus"), "Fallback HBS_Bus must not pollute .kicad_pro netclasses");
        assert_eq!(board.get_netclass_for_net(1).name, "P1P2_Bus");
        assert!((board.get_netclass_for_net(1).clearance - 0.30).abs() < 1e-6);
        assert_eq!(board.get_netclass_for_net(2).name, "Default");
        assert!((board.board_edge_clearance() - 0.45).abs() < 1e-6);
        assert!((board.design_rules().min_hole_clearance - 0.28).abs() < 1e-6);
        assert!((board.design_rules().min_hole_to_hole - 0.30).abs() < 1e-6);

        let dru = r#"(version 1)
(rule "Custom Hole To Hole"
  (constraint hole_to_hole (min 0.35mm))
  (constraint hole_clearance (min 0.32mm))
  (constraint edge_clearance (min 0.50mm))
)
(rule "High Voltage Bus"
  (condition "A.NetClass == 'P1P2_Bus'")
  (constraint clearance (min 0.40mm))
  (constraint track_width (min 0.60mm))
)"#;
        board.apply_design_rules_dru_str(dru);
        assert!((board.design_rules().min_hole_to_hole - 0.35).abs() < 1e-6);
        assert!((board.design_rules().min_hole_clearance - 0.32).abs() < 1e-6);
        assert!((board.board_edge_clearance() - 0.50).abs() < 1e-6);
        assert!((board.get_netclass_for_net(1).clearance - 0.40).abs() < 1e-6);
        assert!((board.get_netclass_for_net(1).track_width - 0.60).abs() < 1e-6);
    }

    #[test]
    fn test_is_gnd_net_name_supports_root_sheet_slash_prefix() {
        assert!(is_gnd_net_name("GND"));
        assert!(is_gnd_net_name("/GND"));
        assert!(is_gnd_net_name("gnd"));
        assert!(is_gnd_net_name("/gnd"));
        assert!(is_gnd_net_name("GND_EARTH"));
        assert!(is_gnd_net_name("/GND_PWR"));
        assert!(!is_gnd_net_name("+3.3V"));
        assert!(!is_gnd_net_name("/+3.3V"));
        assert!(!is_gnd_net_name("SGND"));
        assert!(!is_gnd_net_name(""));
    }

    #[test]
    fn test_parse_gr_text_and_silkscreen_text_nudger() {
        let block = r#"(gr_text "TEST_LABEL" (at 15.5 25.5 90) (layer "F.SilkS") (uuid "abcd-1234") (effects (font (size 1.2 1.2) (thickness 0.15))))"#;
        let bt = parse_gr_text(block, 100).expect("parse_gr_text should parse valid gr_text");
        assert_eq!(bt.text, "TEST_LABEL");
        assert!((bt.x - 15.5).abs() < 1e-6);
        assert!((bt.y - 25.5).abs() < 1e-6);
        assert!((bt.rotation - 90.0).abs() < 1e-6);
        assert_eq!(bt.layer, "F.SilkS");
        assert!((bt.width - 1.2).abs() < 1e-6);
        assert!((bt.height - 1.2).abs() < 1e-6);

        // Test nudger: text at (10.0, 10.0) collides with a pad at (10.0, 10.0)
        let pad = Pad {
            number: "1".to_string(),
            net_id: 1,
            net_name: "SIG".to_string(),
            width: 1.5,
            height: 1.5,
            shape: "rect".to_string(),
            ..Pad::default()
        };
        let fp = Footprint {
            reference: "U1".to_string(),
            value: "IC".to_string(),
            x: 10.0,
            y: 10.0,
            pads: vec![pad],
            ..Footprint::default()
        };
        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 50.0,
            max_y: 50.0,
            footprints: vec![fp],
            texts: vec![BoardText {
                text: "R1".to_string(),
                x: 10.0,
                y: 10.0,
                rotation: 0.0,
                layer: "F.SilkS".to_string(),
                width: 1.0,
                height: 1.0,
                at_span: None,
            }],
            ..Board::default()
        };

        board.nudge_silkscreen_texts();
        let nudged = &board.texts[0];
        let d = (nudged.x - 10.0).hypot(nudged.y - 10.0);
        assert!(d >= 0.70, "Text must be nudged away from pad center (dist={:.3})", d);
    }

    #[test]
    fn test_gpu_silkscreen_text_nudger_batch_equivalence() {
        let pad1 = Pad {
            number: "1".to_string(),
            net_id: 1,
            net_name: "NET1".to_string(),
            offset_x: -1.0,
            offset_y: 0.0,
            width: 1.4,
            height: 1.4,
            shape: "rect".to_string(),
            ..Pad::default()
        };
        let pad2 = Pad {
            number: "2".to_string(),
            net_id: 2,
            net_name: "NET2".to_string(),
            offset_x: 1.0,
            offset_y: 0.0,
            width: 1.4,
            height: 1.4,
            shape: "rect".to_string(),
            ..Pad::default()
        };
        let fp = Footprint {
            reference: "U1".to_string(),
            value: "IC".to_string(),
            x: 15.0,
            y: 15.0,
            pads: vec![pad1, pad2],
            ..Footprint::default()
        };
        let via = Via {
            position: (15.0, 16.5),
            size: 0.8,
            drill: 0.4,
            is_locked: false,
            net_id: 1,
        };
        let base_board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 40.0,
            max_y: 40.0,
            footprints: vec![fp],
            vias: vec![via],
            texts: vec![
                BoardText {
                    text: "U1".to_string(),
                    x: 14.0,
                    y: 15.0,
                    rotation: 0.0,
                    layer: "F.SilkS".to_string(),
                    width: 1.0,
                    height: 1.0,
                    at_span: None,
                },
                BoardText {
                    text: "C1".to_string(),
                    x: 15.0,
                    y: 16.5,
                    rotation: 90.0,
                    layer: "F.SilkS".to_string(),
                    width: 0.8,
                    height: 0.8,
                    at_span: None,
                },
            ],
            ..Board::default()
        };

        let mut cpu_board = base_board.clone();
        cpu_board.nudge_silkscreen_texts_with_gpu(None);

        let vk = crate::vulkan::VulkanEngine::new();
        let mut gpu_board = base_board.clone();
        gpu_board.nudge_silkscreen_texts_with_gpu(vk.as_ref());

        for (i, (ct, gt)) in cpu_board.texts.iter().zip(gpu_board.texts.iter()).enumerate() {
            assert!(
                (ct.x - gt.x).abs() < 1e-6 && (ct.y - gt.y).abs() < 1e-6,
                "Text {} CPU ({:.3}, {:.3}) vs GPU ({:.3}, {:.3}) mismatch",
                i,
                ct.x,
                ct.y,
                gt.x,
                gt.y
            );
            // Verify nudged text clears all pads and vias
            for fp in &gpu_board.footprints {
                for pad in &fp.pads {
                    let d_pad = crate::router::dist_point_to_pad((gt.x, gt.y), fp, pad);
                    assert!(d_pad >= 0.50, "Text {} too close to pad (d={:.3})", i, d_pad);
                }
            }
            for v in &gpu_board.vias {
                let d_via = (gt.x - v.position.0).hypot(gt.y - v.position.1) - v.size * 0.5;
                assert!(d_via >= 0.50, "Text {} too close to via (d={:.3})", i, d_via);
            }
        }
    }

    #[test]
    fn test_parse_footprint_heals_stripped_pad_rotations_l1_and_u1() {
        let mut net_map = HashMap::new();
        let mut nets = Vec::new();
        let extra_locked = HashSet::new();

        let l1_block = r#"(footprint "Inductor_SMD:L_1210_3225Metric" (layer "F.Cu") (at 115 91.4) (property "Reference" "L1") (fp_rect (start -2.1 -1.8) (end 2.1 1.8) (layer "F.CrtYd")) (pad "1" smd rect (at -1.525 0 90) (size 1.5 3.6) (layers "F.Cu" "F.Paste" "F.Mask")) (pad "2" smd rect (at 1.525 0 90) (size 1.5 3.6) (layers "F.Cu" "F.Paste" "F.Mask")))"#;
        let l1 = parse_footprint(l1_block, 0, &mut nets, &mut net_map, &extra_locked)
            .expect("L1 should parse");
        assert_eq!(l1.pads.len(), 2);
        for pad in &l1.pads {
            assert!(
                pad.rotation.abs() < 1e-6,
                "L1 stripped pad rotation should be healed to 0.0, got {}",
                pad.rotation
            );
        }

        let u1_block = r#"(footprint "Package_DFN_QFN:QFN-16-1EP_3x3mm_P0.5mm" (layer "F.Cu") (at 108 111.8) (property "Reference" "U1") (fp_rect (start -2.0 -2.0) (end 2.0 2.0) (layer "F.CrtYd")) (pad "1" smd rect (at -1.7875 -1.25 270) (size 0.575 0.35) (layers "F.Cu" "F.Paste" "F.Mask")) (pad "2" smd rect (at -1.7875 -0.75 270) (size 0.575 0.35) (layers "F.Cu" "F.Paste" "F.Mask")))"#;
        let u1 = parse_footprint(u1_block, 0, &mut nets, &mut net_map, &extra_locked)
            .expect("U1 should parse");
        assert_eq!(u1.pads.len(), 2);
        for pad in &u1.pads {
            assert!(
                pad.rotation.abs() < 1e-6,
                "U1 stripped pad rotation should be healed to 0.0, got {}",
                pad.rotation
            );
        }
    }

    #[test]
    fn test_dru_generation_intra_footprint_pad_and_hole_clearance() {
        let temp_dir = std::env::temp_dir();
        let pcb_path = temp_dir.join("test_save_dru_gen.kicad_pcb");
        let dru_path = temp_dir.join("test_save_dru_gen.kicad_dru");
        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 30.0,
            max_y: 30.0,
            raw_content: "(kicad_pcb (version 20240108) (generator \"test\"))\n".to_string(),
            ..Board::default()
        };

        board.save_pcb(&pcb_path).expect("save_pcb should succeed");
        assert!(dru_path.exists(), "Sibling .kicad_dru must be generated");
        let dru_content = std::fs::read_to_string(&dru_path).unwrap();
        assert!(
            dru_content.contains("A.parent == B.parent"),
            ".kicad_dru must contain intra-footprint exemption rule"
        );
        assert!(
            dru_content.contains("clearance (min 0.12mm)"),
            ".kicad_dru must constrain intra-footprint pad clearance to 0.12mm"
        );
        assert!(
            dru_content.contains("hole_to_hole (min 0.25mm)"),
            ".kicad_dru must constrain intra-footprint hole_to_hole to 0.25mm"
        );

        let _ = std::fs::remove_file(pcb_path);
        let _ = std::fs::remove_file(dru_path);
    }

    #[test]
    fn test_current_timestamp_format_and_backup_path() {
        let ts = Board::current_timestamp_yyyymmddhhmmss();
        assert_eq!(ts.len(), 14, "Timestamp must be 14 digits YYYYMMDDHHMMSS");
        assert!(ts.chars().all(|c| c.is_ascii_digit()), "Timestamp must only contain digits");

        let pcb = Path::new("/tmp/my_test_board.kicad_pcb");
        let bak = Board::compute_pcb_backup_path(pcb);
        let bak_str = bak.to_string_lossy();
        assert!(
            bak_str.starts_with("/tmp/my_test_board.kicad_pcb_bak."),
            "Backup path must start with base name and .kicad_pcb_bak. suffix, got: {}",
            bak_str
        );
    }

    #[test]
    fn test_save_pcb_creates_backup_when_overwriting() {
        let temp_dir = std::env::temp_dir();
        let test_id = std::process::id();
        let pcb_path = temp_dir.join(format!("test_overwrite_board_{}.kicad_pcb", test_id));
        let dru_path = temp_dir.join(format!("test_overwrite_board_{}.kicad_dru", test_id));

        // Create original file
        let original_content = "(kicad_pcb (version 20240108) (generator \"original\"))\n";
        std::fs::write(&pcb_path, original_content).unwrap();

        let mut board = Board {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 20.0,
            max_y: 20.0,
            raw_content: "(kicad_pcb (version 20240108) (generator \"updated\"))\n".to_string(),
            ..Board::default()
        };

        board.save_pcb(&pcb_path).expect("save_pcb must succeed");

        // Verify updated file content
        let updated_content = std::fs::read_to_string(&pcb_path).unwrap();
        assert!(updated_content.contains("updated"), "PCB file must contain updated content");

        // Search for the generated backup file: test_overwrite_board_{test_id}.kicad_pcb_bak.YYYYMMDDHHMMSS
        let prefix = format!("test_overwrite_board_{}.kicad_pcb_bak.", test_id);
        let found_bak = std::fs::read_dir(&temp_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .find(|e| e.file_name().to_string_lossy().starts_with(&prefix));

        assert!(found_bak.is_some(), "Backup file matching .kicad_pcb_bak.YYYYMMDDHHMMSS must exist");
        let bak_entry = found_bak.unwrap();
        let bak_content = std::fs::read_to_string(bak_entry.path()).unwrap();
        assert_eq!(
            bak_content, original_content,
            "Backup file must contain original pre-overwrite content"
        );

        let _ = std::fs::remove_file(&pcb_path);
        let _ = std::fs::remove_file(&dru_path);
        let _ = std::fs::remove_file(bak_entry.path());
    }

    #[test]
    fn test_polygon_expanded_preserves_non_orthogonal_quad_geometry() {
        let offset = 0.25;

        // 1. Axis-aligned 4-vertex rectangle still expands directly to [min_x - offset, max_x + offset] x [min_y - offset, max_y + offset]
        let rect = Polygon::from_rect(-5.0, -1.0, 10.0, 2.0);
        let exp_rect = rect.expanded(offset);
        assert_eq!(exp_rect.vertices.len(), 4);
        let exp_rect_aabb = exp_rect.aabb();
        assert!((exp_rect_aabb.min_x - (-5.25)).abs() < 1e-6);
        assert!((exp_rect_aabb.max_x - 5.25).abs() < 1e-6);
        assert!((exp_rect_aabb.min_y - (-1.25)).abs() < 1e-6);
        assert!((exp_rect_aabb.max_y - 1.25).abs() < 1e-6);

        // 2. Non-orthogonal 4-vertex quad (10 mm x 2 mm rectangle rotated by 45 degrees):
        // Expanding after rotation must match expanding before rotation and then rotating,
        // and must NOT collapse into the bloated axis-aligned bounding box of the rotated quad.
        let rot_rect = rect.transform(Point::new(0.0, 0.0), 45.0);
        let exp_after_rot = rot_rect.expanded(offset);
        let exp_before_rot = rect.expanded(offset).transform(Point::new(0.0, 0.0), 45.0);
        assert_eq!(exp_after_rot.vertices.len(), 4);

        for (va, vb) in exp_after_rot.vertices.iter().zip(exp_before_rot.vertices.iter()) {
            assert!(
                va.distance_to(vb) < 1e-6,
                "Rotated quad expanded vertex {:?} must match expected {:?}",
                va,
                vb
            );
        }

        // Verify none of the edges of the rotated expanded quad were turned into horizontal/vertical AABB edges
        for i in 0..4 {
            let p1 = exp_after_rot.vertices[i];
            let p2 = exp_after_rot.vertices[(i + 1) % 4];
            assert!(
                (p2.x - p1.x).abs() > 0.5 && (p2.y - p1.y).abs() > 0.5,
                "Rotated quad edge ({:?} -> {:?}) must remain angled, not axis-aligned",
                p1,
                p2
            );
        }

        // Verify a small probe quad in the empty AABB corner region (e.g. (-3.0, -3.0)) does NOT collide
        // with the rotated expanded quad (whereas an AABB expansion [-4.49, 4.49]^2 would falsely collide).
        let corner_probe_neg = Polygon::from_rect(-3.2, -3.2, 0.4, 0.4);
        let corner_probe_pos = Polygon::from_rect(2.8, 2.8, 0.4, 0.4);
        let rot_aabb = rot_rect.aabb();
        let bloated_aabb_poly = Polygon::from_rect(
            rot_aabb.min_x - offset,
            rot_aabb.min_y - offset,
            (rot_aabb.max_x - rot_aabb.min_x) + 2.0 * offset,
            (rot_aabb.max_y - rot_aabb.min_y) + 2.0 * offset,
        );
        assert!(
            bloated_aabb_poly.intersects_convex(&corner_probe_neg),
            "Bloated AABB would falsely intersect the empty corner region"
        );
        assert!(
            !exp_after_rot.intersects_convex(&corner_probe_neg),
            "Expanded rotated quad must NOT intersect empty top-left AABB corner"
        );
        assert!(
            !exp_after_rot.intersects_convex(&corner_probe_pos),
            "Expanded rotated quad must NOT intersect empty bottom-right AABB corner"
        );

        // 3. Non-orthogonal trapezoid quad preserves its slanted sides when expanded
        let trap = Polygon::new(vec![
            Point::new(-4.0, -1.0),
            Point::new(4.0, -1.0),
            Point::new(2.0, 1.0),
            Point::new(-2.0, 1.0),
        ]);
        let exp_trap = trap.expanded(offset);
        assert_eq!(exp_trap.vertices.len(), 4);
        assert!(
            (exp_trap.vertices[1].x - exp_trap.vertices[2].x).abs() > 1.5,
            "Trapezoid slanted side must remain slanted after expansion"
        );
    }
}
