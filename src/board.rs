// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};

pub const DEFAULT_LOCKED_REFS: &[&str] = &[
    "J1", "J2", "D1", "F1", "F2", "R1", "R2", "C3", "C4",
    "D4", "D5", "R18", "R19", "SW1", "SW2", "U4",
    "H1", "H2", "H3", "H4",
];

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_to(&self, other: &Point) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }

    pub fn rotate_ccw(&self, angle_deg: f64) -> Self {
        let rad = angle_deg.to_radians();
        let (sin_a, cos_a) = rad.sin_cos();
        Self {
            x: self.x * cos_a + self.y * sin_a,
            y: -self.x * sin_a + self.y * cos_a,
        }
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AABB {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl AABB {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    pub fn overlaps(&self, other: &AABB) -> bool {
        self.min_x < other.max_x
            && self.max_x > other.min_x
            && self.min_y < other.max_y
            && self.max_y > other.min_y
    }

    pub fn contains_point(&self, p: &Point) -> bool {
        p.x >= self.min_x && p.x <= self.max_x && p.y >= self.min_y && p.y <= self.max_y
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Polygon {
    pub vertices: Vec<Point>,
}

impl Polygon {
    pub fn new(vertices: Vec<Point>) -> Self {
        Self { vertices }
    }

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

    pub fn expanded(&self, offset: f64) -> Polygon {
        if offset <= 0.0 || self.vertices.is_empty() {
            return self.clone();
        }
        let aabb = self.aabb();
        if self.vertices.len() == 4 {
            return Polygon::new(vec![
                Point::new(aabb.min_x - offset, aabb.min_y - offset),
                Point::new(aabb.max_x + offset, aabb.min_y - offset),
                Point::new(aabb.max_x + offset, aabb.max_y + offset),
                Point::new(aabb.min_x - offset, aabb.max_y + offset),
            ]);
        }
        let n = self.vertices.len() as f64;
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

    pub fn decompose_convex(&self) -> Vec<Polygon> {
        if self.is_convex() || self.vertices.len() <= 3 {
            return vec![self.clone()];
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
            vec![self.clone()]
        } else {
            triangles
        }
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub p1: Point,
    pub p2: Point,
    pub net_id: usize,
}

impl Segment {
    pub fn length(&self) -> f64 {
        self.p1.distance_to(&self.p2)
    }

    pub fn aabb(&self) -> AABB {
        AABB::new(
            self.p1.x.min(self.p2.x),
            self.p1.y.min(self.p2.y),
            self.p1.x.max(self.p2.x),
            self.p1.y.max(self.p2.y),
        )
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NetTopology {
    #[default]
    #[serde(alias = "euclidean", alias = "mst", alias = "EuclideanMst")]
    EuclideanMst,
    #[serde(alias = "daisychain", alias = "daisy", alias = "DaisyChain")]
    DaisyChain,
    #[serde(alias = "flyby", alias = "FlyBy")]
    FlyBy,
    #[serde(alias = "tbranch", alias = "t_tree", alias = "TBranch")]
    TBranch,
    #[serde(alias = "star", alias = "starpoint", alias = "StarPoint")]
    StarPoint,
}

impl NetTopology {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FabPreset {
    Jlcpcb,
    PcbWay,
    Eurocircuits,
    Custom,
}

impl FabPreset {
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

    pub fn board_edge_clearance(&self) -> f64 {
        match self {
            Self::Jlcpcb => 0.35,
            Self::PcbWay => 0.30,
            Self::Eurocircuits => 0.40,
            Self::Custom => 0.35,
        }
    }

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DfmConfig {
    pub teardrop_enabled: bool,
    pub teardrop_length_ratio: f64,
    pub teardrop_width_ratio: f64,
    pub board_edge_clearance: f64,
    pub fab_preset: Option<FabPreset>,
}

impl Default for DfmConfig {
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NetClass {
    pub name: String,
    pub track_width: f64,
    pub clearance: f64,
    pub via_dia: f64,
    pub via_drill: f64,
    #[serde(default = "default_diff_pair_dim")]
    pub diff_pair_gap: f64,
    #[serde(default = "default_diff_pair_dim")]
    pub diff_pair_width: f64,
    #[serde(default)]
    pub topology: NetTopology,
    #[serde(default = "default_true")]
    pub teardrop_enabled: bool,
    #[serde(default = "default_teardrop_length_ratio")]
    pub teardrop_length_ratio: f64,
    #[serde(default = "default_teardrop_width_ratio")]
    pub teardrop_width_ratio: f64,
    #[serde(default = "default_board_edge_clearance")]
    pub board_edge_clearance: f64,
    #[serde(default = "default_max_coupled_length_mm")]
    pub max_coupled_length_mm: f64,
    #[serde(default)]
    pub require_return_path: bool,
}

fn default_diff_pair_dim() -> f64 {
    0.20
}

fn default_true() -> bool {
    true
}

fn default_teardrop_length_ratio() -> f64 {
    0.5
}

fn default_teardrop_width_ratio() -> f64 {
    0.8
}

fn default_board_edge_clearance() -> f64 {
    0.35
}

fn default_max_coupled_length_mm() -> f64 {
    15.0
}

impl Default for NetClass {
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
        }
    }
}

impl NetClass {
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

    pub fn resolve_topology(&self, net_name: &str) -> NetTopology {
        if self.topology != NetTopology::EuclideanMst {
            self.topology
        } else {
            NetTopology::infer_from_net_name(net_name)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SwappablePinGroup {
    pub name: String,
    pub pin_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SwappableGateGroup {
    pub name: String,
    pub gate_units: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PinSwapRecord {
    pub footprint_ref: String,
    pub pad1_number: String,
    pub pad1_net: String,
    pub pad2_number: String,
    pub pad2_net: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pad {
    pub number: String,
    pub shape: String,
    pub offset_x: f64,
    pub offset_y: f64,
    pub width: f64,
    pub height: f64,
    pub rotation: f64,
    pub drill: Option<f64>,
    pub net_id: usize,
    pub net_name: String,
    pub is_through_hole: bool,
    #[serde(default)]
    pub pin_name: String,
    #[serde(default)]
    pub swap_group: Option<usize>,
    #[serde(default)]
    pub net_span: Option<(usize, usize)>,
}

impl Default for Pad {
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
        }
    }
}

impl Pad {
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
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Footprint {
    pub reference: String,
    pub value: String,
    pub x: f64,
    pub y: f64,
    pub rotation: f64,
    pub layer: usize,      // 0 = F.Cu (Top), 1 = B.Cu (Bottom)
    pub orig_layer: usize, // Original layer when loaded from .kicad_pcb
    pub is_locked: bool,
    pub pads: Vec<Pad>,
    pub width: f64,
    pub height: f64,
    pub courtyards: Vec<Polygon>,
    #[serde(default)]
    pub perimeters: Vec<Polygon>,
    #[serde(default)]
    pub escape_margin: f64,
    pub at_span: Option<(usize, usize)>,
    pub fp_span: Option<(usize, usize)>,
    #[serde(default)]
    pub pin_swap_groups: Vec<Vec<usize>>,
    #[serde(default)]
    pub gate_swap_groups: Vec<SwappableGateGroup>,
}

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

    pub fn is_smd_only(&self) -> bool {
        !self.pads.is_empty() && self.pads.iter().all(|p| !p.is_through_hole)
    }

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

    pub fn pad_world_corners(&self, pad: &Pad) -> Vec<Point> {
        self.pad_world_corners_layered(pad, self.layer)
    }

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

    /// Computes the bounding box of the footprint relative to its origin (0, 0),
    /// taking into account transformed courtyards (with spacing) and all pad corners.
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

    /// Computes the strictly legal [min_x, max_x, min_y, max_y] clamping range for the component origin
    /// so that courtyards and copper pads remain entirely inside `boundary`.
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

    pub fn pad_absolute_position(&self, pad: &Pad) -> (f64, f64) {
        let oy = if self.layer != self.orig_layer {
            -pad.offset_y
        } else {
            pad.offset_y
        };
        let p = Point::new(pad.offset_x, oy).rotate_ccw(self.rotation);
        (self.x + p.x, self.y + p.y)
    }

    pub fn pad_world_point(&self, pad_idx: usize, pos: Point, rot: f64) -> Point {
        self.pad_world_point_layered(pad_idx, pos, rot, self.layer)
    }

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

    pub fn transformed_courtyards(&self, pos: Point, rot: f64, spacing: f64) -> Vec<Polygon> {
        self.transformed_courtyards_layered(pos, rot, self.layer, spacing)
    }

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

    pub fn transformed_perimeters(&self, pos: Point, rot: f64, spacing: f64) -> Vec<Polygon> {
        self.transformed_perimeters_layered(pos, rot, self.layer, spacing)
    }

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

    /// Computes the pad density (pads per mm² of footprint courtyard area).
    pub fn pad_density(&self) -> f64 {
        let area = (self.width * self.height).max(1.0);
        self.pads.len() as f64 / area
    }

    /// Computes the dedicated routing escape clearance (halo) around dense or fine-pitch components.
    /// - High pin-count packages (>= 16 pads) with fine pitch (<= 0.50mm) or high density: 1.20mm
    /// - Fine-pitch packages (>= 16 pads, pitch <= 0.65mm): 1.00mm
    /// - Medium pin-count packages (>= 8 pads, pitch <= 0.65mm or high density): 0.90mm
    /// - Small dense ICs (>= 6 pads, pitch <= 1.00mm, e.g. SOT-23-6 BUCK IC): 0.75mm
    /// - Standard passives (0603/0805, 2 pads): 0.0mm (standard clearance)
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
    pub fn placement_spacing(&self, base_spacing: f64) -> f64 {
        let margin = if self.escape_margin > 0.0 {
            self.escape_margin
        } else {
            self.compute_escape_margin()
        };
        base_spacing + 2.0 * margin
    }
}

/// Extracts the outer perimeter boundary line segments of a set of polygons,
/// discarding internal shared chords (such as triangulation edges).
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackSegment {
    pub start: (f64, f64),
    pub end: (f64, f64),
    pub width: f64,
    pub layer: usize, // 0 = F.Cu, 1 = B.Cu
    pub net_id: usize,
    #[serde(default)]
    pub is_locked: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Via {
    pub position: (f64, f64),
    pub size: f64,
    pub drill: f64,
    pub net_id: usize,
    #[serde(default)]
    pub is_locked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeepoutZone {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub layers: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct CompiledCluster {
    pub name: String,
    pub anchor_idx: Option<usize>,
    pub member_indices: Vec<usize>,
    pub member_refs: Vec<String>,
    pub max_radius: f64,
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct Board {
    pub file_path: String,
    pub raw_content: String,
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub footprints: Vec<Footprint>,
    pub holding_palette: Vec<Footprint>,
    pub nets: Vec<String>,
    pub net_name_to_id: HashMap<String, usize>,
    pub netclasses: HashMap<String, NetClass>,
    pub net_to_class: HashMap<String, String>,
    pub keepouts: Vec<KeepoutZone>,
    pub tracks: Vec<TrackSegment>,
    pub vias: Vec<Via>,
    pub clusters: Vec<CompiledCluster>,
    pub schematic_positions: HashMap<String, (f64, f64)>,
    pub outline_polygon: Option<Polygon>,
    pub pin_swap_history: Vec<PinSwapRecord>,
}

impl Default for Board {
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
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon: None,
            pin_swap_history: Vec::new(),
        }
    }
}

pub const INTERNAL_COPPER_LAYERS: [&str; 30] = [
    "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "In5.Cu",
    "In6.Cu", "In7.Cu", "In8.Cu", "In9.Cu", "In10.Cu",
    "In11.Cu", "In12.Cu", "In13.Cu", "In14.Cu", "In15.Cu",
    "In16.Cu", "In17.Cu", "In18.Cu", "In19.Cu", "In20.Cu",
    "In21.Cu", "In22.Cu", "In23.Cu", "In24.Cu", "In25.Cu",
    "In26.Cu", "In27.Cu", "In28.Cu", "In29.Cu", "In30.Cu",
];

pub fn detect_copper_layers(raw_content: &str) -> Vec<&'static str> {
    let mut layers = vec!["F.Cu", "B.Cu"];
    for &name in &INTERNAL_COPPER_LAYERS {
        if raw_content.contains(&format!("\"{}\"", name)) {
            layers.push(name);
        }
    }
    layers
}

impl Board {
    pub fn load(path: &str, extra_locked: &HashSet<String>) -> Result<Self, String> {
        let content = fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;

        // 1. Parse top-level nets
        let mut nets = vec!["".to_string()];
        let mut net_name_to_id = HashMap::new();
        net_name_to_id.insert("".to_string(), 0);

        for line in content.lines() {
            let trimmed = line.trim();
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
                        if let Some(tr) = parse_segment(block, &active_layers) {
                            tracks.push(tr);
                        }
                        i = end;
                        continue;
                    }
                } else if word == "via" {
                    if let Some(end) = find_closing_paren(bytes, i) {
                        let block = &content[i..end];
                        if let Some(v) = parse_via(block) {
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
            raw_content: content,
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
            clusters: Vec::new(),
            schematic_positions: HashMap::new(),
            outline_polygon,
            pin_swap_history: Vec::new(),
        };

        // Auto-load project netclasses if sibling .kicad_pro exists
        let pro_path = Path::new(path).with_extension("kicad_pro");
        if pro_path.exists() {
            board.load_project_netclasses(&pro_path);
            // The project defines its own classes; drop built-in fallbacks left with no nets.
            let used: HashSet<&String> = board.net_to_class.values().collect();
            board.netclasses.retain(|name, _| name == "Default" || used.contains(name));
        }

        // Auto-load schematic functional clusters if sibling .kicad_sch exists
        let sch_path = Path::new(path).with_extension("kicad_sch");
        if sch_path.exists() {
            board.load_schematic_clusters(&sch_path);
        }

        for fp in &mut board.footprints {
            fp.infer_swap_groups();
        }

        Ok(board)
    }

    pub fn copper_layers(&self) -> Vec<&'static str> {
        detect_copper_layers(&self.raw_content)
    }

    pub fn layer_name(&self, layer_idx: usize) -> &'static str {
        self.copper_layers()
            .get(layer_idx)
            .copied()
            .unwrap_or(if layer_idx == 0 { "F.Cu" } else { "B.Cu" })
    }

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

    pub fn layer_plane_nets(&self) -> HashMap<usize, usize> {
        let mut map = HashMap::new();
        for (l_idx, name) in self.layer_plane_net_names() {
            if let Some(&net_id) = self.net_name_to_id.get(&name) {
                map.insert(l_idx, net_id);
            }
        }
        map
    }

    pub fn inner_plane_nets(&self) -> HashMap<usize, usize> {
        let mut map = self.layer_plane_nets();
        map.retain(|&layer_idx, _| layer_idx >= 2);
        map
    }

    pub fn boundary_aabb(&self) -> AABB {
        if let Some(ref poly) = self.outline_polygon {
            poly.aabb()
        } else {
            AABB::new(self.min_x + 1.5, self.min_y + 1.5, self.max_x - 1.5, self.max_y - 1.5)
        }
    }

    /// Minimum clearance from copper to Edge.Cuts outline based on board netclasses (default 0.35 mm).
    pub fn board_edge_clearance(&self) -> f64 {
        self.netclasses
            .values()
            .map(|nc| nc.board_edge_clearance)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.35)
    }

    /// Verifies if a point is strictly inside the physical board boundary and outline contour.
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
    /// (letters first, then numbers, e.g. C5 before C20).
    pub fn clear_unlocked_objects(&mut self) -> (usize, usize) {
        let tracks_ripped = self.tracks.iter().filter(|t| !t.is_locked).count()
            + self.vias.iter().filter(|v| !v.is_locked).count();
        self.tracks.retain(|t| t.is_locked);
        self.vias.retain(|v| v.is_locked);

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
    pub fn rip_up_connected_traces_to_component(&mut self, fp_idx: usize) -> usize {
        self.rip_up_connected_traces_to_component_at(fp_idx, None)
    }

    /// Rips up all connected tracks and vias from the component at `fp_idx` (optionally evaluated
    /// at `pos_override: (x, y, rotation, layer)` prior to its move) all the way to the next
    /// attached component(s).
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

            let via_indices: Vec<usize> = self
                .vias
                .iter()
                .enumerate()
                .filter(|(_, v)| v.net_id == net_id && !v.is_locked)
                .map(|(i, _)| i)
                .collect();

            if trk_indices.is_empty() && via_indices.is_empty() {
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
                    Item::Via(vi) => {
                        vias_to_remove.insert(vi);
                    }
                }
            }
        }

        let ripped_tracks_count = tracks_to_remove.len();
        let ripped_vias_count = vias_to_remove.len();
        let total_ripped = ripped_tracks_count + ripped_vias_count;

        if total_ripped > 0 {
            let mut new_tracks = Vec::with_capacity(self.tracks.len() - ripped_tracks_count);
            for (idx, trk) in self.tracks.drain(..).enumerate() {
                if !tracks_to_remove.contains(&idx) {
                    new_tracks.push(trk);
                }
            }
            self.tracks = new_tracks;

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
    pub fn return_holding_palette_to_board(&mut self) -> usize {
        let count = self.holding_palette.len();
        self.footprints.extend(self.holding_palette.drain(..));
        count
    }

    /// Restores a single component at `idx` from the holding palette back onto the board.
    pub fn return_holding_component_to_board(&mut self, idx: usize) -> Option<Footprint> {
        if idx < self.holding_palette.len() {
            let fp = self.holding_palette.remove(idx);
            self.footprints.push(fp.clone());
            Some(fp)
        } else {
            None
        }
    }

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

    pub fn get_net_topology(&self, net_id: usize) -> NetTopology {
        let netclass = self.get_netclass_for_net(net_id);
        let net_name = self.nets.get(net_id).map(|s| s.as_str()).unwrap_or("");
        netclass.resolve_topology(net_name)
    }

    pub fn generate_ratsnest_segments(&self, pad_positions: &[Vec<Point>]) -> Vec<Segment> {
        let mut net_points: HashMap<usize, Vec<Point>> = HashMap::new();
        for (fp_idx, fp) in self.footprints.iter().enumerate() {
            for (pad_idx, pad) in fp.pads.iter().enumerate() {
                if pad.net_id == 0 || pad.net_name.is_empty() || pad.net_name == "GND" {
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

    pub fn save_pcb(&self, out_path: &Path) -> Result<(), String> {
        let mut replacements: Vec<(usize, usize, String)> = Vec::new();
        for fp in &self.footprints {
            if fp.is_locked && fp.layer == fp.orig_layer {
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
                        let mut fp_block = self.raw_content[f_start..f_end].to_string();
                        if let Some((a_start, a_end)) = fp.at_span {
                            if a_start >= f_start && a_end <= f_end {
                                fp_block.replace_range((a_start - f_start)..(a_end - f_start), &new_at);
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
        }

        // Also update any swapped pad nets in place
        for fp in &self.footprints {
            if fp.layer == fp.orig_layer {
                for pad in &fp.pads {
                    if let Some((n_start, n_end)) = pad.net_span {
                        let expected_net_str = format!("(net {} \"{}\")", pad.net_id, pad.net_name);
                        if n_start < n_end && n_end <= self.raw_content.len() {
                            if &self.raw_content[n_start..n_end] != expected_net_str {
                                replacements.push((n_start, n_end, expected_net_str));
                            }
                        }
                    }
                }
            }
        }

        // Strip out existing (segment ...) and (via ...) so we don't duplicate or retain old tracks
        let bytes = self.raw_content.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'(' {
                let mut j = i + 1;
                while j < bytes.len() && !bytes[j].is_ascii_whitespace() && bytes[j] != b')' {
                    j += 1;
                }
                let word = &self.raw_content[i + 1..j];
                if word == "segment" || word == "via" {
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

        if !self.tracks.is_empty() || !self.vias.is_empty() {
            let last_paren = content.rfind(')').ok_or("Invalid .kicad_pcb syntax")?;
            let mut s_expr = String::new();
            for tr in &self.tracks {
                let layer_name = self.layer_name(tr.layer);
                let locked_str = if tr.is_locked { " (locked yes)" } else { "" };
                s_expr.push_str(&format!(
                    "  (segment (start {:.4} {:.4}) (end {:.4} {:.4}) (width {:.4}) (layer \"{}\") (net {}){})\n",
                    tr.start.0, tr.start.1, tr.end.0, tr.end.1, tr.width, layer_name, tr.net_id, locked_str
                ));
            }
            for v in &self.vias {
                let locked_str = if v.is_locked { " (locked yes)" } else { "" };
                s_expr.push_str(&format!(
                    "  (via (at {:.4} {:.4}) (size {:.4}) (drill {:.4}) (layers \"F.Cu\" \"B.Cu\") (net {}){})\n",
                    v.position.0, v.position.1, v.size, v.drill, v.net_id, locked_str
                ));
            }
            if !content.contains("(net_class ") {
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

        fs::write(out_path, content).map_err(|e| e.to_string())?;

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
            let src_dru = src_base.with_extension("kicad_dru");
            let out_dru = out_path.with_extension("kicad_dru");
            if src_dru.exists() && src_dru != out_dru {
                let _ = fs::copy(&src_dru, &out_dru);
            }
            let src_prl = src_base.with_extension("kicad_prl");
            let out_prl = out_path.with_extension("kicad_prl");
            if src_prl.exists() && src_prl != out_prl {
                let _ = fs::copy(&src_prl, &out_prl);
            }
        }

        Ok(())
    }

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

    pub fn save_back_annotation_eco(&self, path: &Path) -> std::io::Result<()> {
        let report = self.generate_back_annotation_report();
        fs::write(path, report)
    }

    pub fn save_ses(&self, ses_path: &Path) -> Result<(), String> {
        let mut s = String::from("(session vulcans_trials\n  (base_design vulcans_trials)\n  (placement\n    (resolution mm 1000)\n");
        for fp in &self.footprints {
            let side = if fp.layer == 1 { "back" } else { "front" };
            s.push_str(&format!(
                "    (component \"{}\" (place \"{}\" {:.4} {:.4} {} {:.1}))\n",
                fp.reference, fp.reference, fp.x, fp.y, side, fp.rotation
            ));
        }
        s.push_str("  )\n  (routes\n    (resolution mm 1000)\n    (network\n");
        let mut net_tracks: HashMap<usize, Vec<&TrackSegment>> = HashMap::new();
        for tr in &self.tracks {
            net_tracks.entry(tr.net_id).or_default().push(tr);
        }
        for (net_id, tracks) in net_tracks {
            if let Some(net_name) = self.nets.get(net_id) {
                s.push_str(&format!("      (net \"{}\"\n        (wire\n", net_name));
                for tr in tracks {
                    let layer_name = self.layer_name(tr.layer);
                    s.push_str(&format!(
                        "          (path \"{}\" {:.4} {:.4} {:.4} {:.4} {:.4})\n",
                        layer_name, tr.width, tr.start.0, tr.start.1, tr.end.0, tr.end.1
                    ));
                }
                s.push_str("        )\n      )\n");
            }
        }
        s.push_str("    )\n  )\n)\n");
        fs::write(ses_path, s).map_err(|e| e.to_string())
    }

    pub fn parse_netclasses_from_pro_or_pcb(&mut self, content: &str) {
        let classes = parse_netclasses_from_pro_or_pcb(content);
        for (name, nc) in classes {
            self.netclasses.insert(name, nc);
        }
    }

    pub fn load_project_netclasses(&mut self, pro_path: &Path) {
        let Ok(content) = fs::read_to_string(pro_path) else { return };
        self.parse_netclasses_from_pro_or_pcb(&content);
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
                        net.starts_with(prefix)
                    } else if let Some(suffix) = pat.strip_prefix('*') {
                        net.ends_with(suffix)
                    } else {
                        net == pat
                    };
                    if matched {
                        self.net_to_class.insert(net.clone(), cls_name.to_string());
                    }
                }
            }
        }
    }

    fn load_schematic_clusters(&mut self, sch_path: &Path) {
        let Ok(content) = fs::read_to_string(sch_path) else { return };
        let mut headers: Vec<(String, f64, f64)> = Vec::new();
        let text_tag = "(text \"";
        let mut idx = 0;
        while let Some(pos) = content[idx..].find(text_tag) {
            let start = idx + pos + text_tag.len();
            if let Some(q_end) = content[start..].find('"') {
                let title = content[start..start + q_end].trim().to_string();
                if title.chars().next().is_some_and(|c| c.is_ascii_digit()) && title.contains('.') {
                    let rest = &content[start + q_end..];
                    if let Some((x, y, _)) = parse_at_xy_rot(&rest[..rest.len().min(200)]) {
                        headers.push((title, x, y));
                    }
                }
                idx = start + q_end + 1;
            } else {
                break;
            }
        }
        let mut members: HashMap<String, Vec<String>> = headers.iter().map(|h| (h.0.clone(), Vec::new())).collect();
        let mut s_idx = 0;
        while let Some(pos) = content[s_idx..].find("(symbol") {
            let actual = s_idx + pos;
            let after = actual + 7;
            if after < content.len() && content.as_bytes()[after].is_ascii_whitespace() {
                let win = &content[actual..(actual + 1500).min(content.len())];
                if win.contains("(lib_id ") {
                    if let (Some((sx, sy, _)), Some(r)) = (parse_at_xy_rot(win), extract_property(win, "Reference")) {
                        if !r.starts_with('#') {
                            self.schematic_positions.insert(r.clone(), (sx, sy));
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
                    }
                }
            }
            s_idx = actual + 7;
        }

        let ref_to_idx: HashMap<&str, usize> = self
            .footprints
            .iter()
            .enumerate()
            .map(|(i, fp)| (fp.reference.as_str(), i))
            .collect();

        for (title, m_refs) in members {
            let m_indices: Vec<usize> = m_refs.iter().filter_map(|r| ref_to_idx.get(r.as_str()).copied()).collect();
            if m_indices.is_empty() {
                continue;
            }
            let up = title.to_uppercase();
            let (anchor_prefix, max_radius, weight) = if up.contains("BUCK") || up.contains("POWER") {
                (Some('U'), 4.5, 150.0)
            } else if up.contains("SURGE") || up.contains("INPUT") || up.contains("USB") {
                (Some('J'), 6.8, 100.0)
            } else if up.contains("ESP32") || up.contains("CONTROLS") {
                (Some('U'), 12.0, 50.0)
            } else if up.contains("TRANSCEIVER") || up.contains("MAX22088") {
                (Some('U'), 7.5, 100.0)
            } else {
                (None, 7.0, 60.0)
            };
            let anchor_idx = anchor_prefix.and_then(|pref| {
                m_refs
                    .iter()
                    .find(|r| r.starts_with(pref))
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

fn extract_named_f64(block: &str, tag: &str) -> Option<f64> {
    let pattern = format!("({} ", tag);
    let pos = block.find(&pattern)?;
    let rest = &block[pos + pattern.len()..];
    let end = rest.find(')')?;
    rest[..end].split_whitespace().next()?.parse::<f64>().ok()
}

fn extract_named_usize(block: &str, tag: &str) -> Option<usize> {
    let pattern = format!("({} ", tag);
    let pos = block.find(&pattern)?;
    let rest = &block[pos + pattern.len()..];
    let end = rest.find(')')?;
    rest[..end].split_whitespace().next()?.parse::<usize>().ok()
}

fn parse_segment(block: &str, active_layers: &[&str]) -> Option<TrackSegment> {
    let (sx, sy) = extract_point(block, "start")?;
    let (ex, ey) = extract_point(block, "end")?;
    let width = extract_named_f64(block, "width").unwrap_or(0.20);
    let layer = active_layers
        .iter()
        .position(|&l| block.contains(&format!("\"{}\"", l)))
        .unwrap_or_else(|| if block.contains("\"B.Cu\"") { 1 } else { 0 });
    let net_id = extract_named_usize(block, "net").unwrap_or(0);
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

fn parse_via(block: &str) -> Option<Via> {
    let (vx, vy) = extract_point(block, "at")?;
    let size = extract_named_f64(block, "size").unwrap_or(0.60);
    let drill = extract_named_f64(block, "drill").unwrap_or(0.30);
    let net_id = extract_named_usize(block, "net").unwrap_or(0);
    let is_locked = block.contains("(locked yes)") || block.contains("(locked)") || block.contains("(status 40000)");
    Some(Via {
        position: (vx, vy),
        size,
        drill,
        net_id,
        is_locked,
    })
}

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

fn parse_point_tag(s: &str, tag: &str) -> Option<Point> {
    let p = s.find(tag)?;
    let rest = &s[p + tag.len()..];
    let e = rest.find(')')?;
    let mut parts = rest[..e].split_whitespace();
    let x = parts.next()?.parse::<f64>().ok()?;
    let y = parts.next()?.parse::<f64>().ok()?;
    Some(Point::new(x, y))
}

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

fn extract_property(s: &str, prop_name: &str) -> Option<String> {
    let pattern = format!("(property \"{}\" \"", prop_name);
    let pos = s.find(&pattern)?;
    let start = pos + pattern.len();
    let end = s[start..].find('"')?;
    Some(s[start..start + end].to_string())
}

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

pub fn parse_netclasses_from_pro_or_pcb(content: &str) -> HashMap<String, NetClass> {
    let mut netclasses = HashMap::new();
    let trimmed = content.trim_start();
    if trimmed.starts_with('{') {
        if let Ok(root) = serde_json::from_str::<serde_json::Value>(content) {
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
                for cls in classes {
                    let Some(name) = cls.get("name").and_then(|v| v.as_str()) else { continue };
                    let track_width = cls.get("track_width").and_then(|v| v.as_f64()).unwrap_or(0.20);
                    let clearance = cls.get("clearance").and_then(|v| v.as_f64()).unwrap_or(0.20);
                    let via_dia = cls.get("via_diameter").or_else(|| cls.get("via_dia")).and_then(|v| v.as_f64()).unwrap_or(0.60);
                    let via_drill = cls.get("via_drill").and_then(|v| v.as_f64()).unwrap_or(0.30);
                    let diff_pair_gap = cls.get("diff_pair_gap").and_then(|v| v.as_f64()).unwrap_or(0.20);
                    let diff_pair_width = cls.get("diff_pair_width").and_then(|v| v.as_f64()).unwrap_or(track_width);
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
                    let default_edge_clr = cls_preset.map(|p| p.board_edge_clearance()).unwrap_or(0.35);
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
                    };
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
                let is_th = tokens.get(1).copied() == Some("thru_hole") || tokens.get(1).copied() == Some("np_thru_hole");
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialPhase {
    Idle,
    Legalizing,
    AutoPlacing,
    AutoRouting,
    Completed,
}

impl TrialPhase {
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

#[derive(Debug, Clone)]
pub struct LiveTrialState {
    pub board: Board,
    pub ratsnest: Vec<Segment>,
    pub phase: TrialPhase,
    pub progress: f32,
    pub status_text: String,
    pub log_lines: Vec<String>,
    pub crossings: usize,
    pub wirelength_mm: f64,
    pub overlaps: usize,
    pub evals_per_sec: f64,
    pub routed_conns: usize,
    pub total_conns: usize,
    pub routing_pass: usize,
}

impl LiveTrialState {
    pub fn new(board: Board) -> Self {
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
        };
        state.refresh_ratsnest();
        state
    }

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
}

