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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetClass {
    pub name: String,
    pub track_width: f64,
    pub clearance: f64,
    pub via_dia: f64,
    pub via_drill: f64,
}

impl Default for NetClass {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            track_width: 0.20,
            clearance: 0.20,
            via_dia: 0.60,
            via_drill: 0.30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pad {
    pub number: String,
    pub offset_x: f64,
    pub offset_y: f64,
    pub width: f64,
    pub height: f64,
    pub net_id: usize,
    pub net_name: String,
    pub is_through_hole: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Footprint {
    pub reference: String,
    pub value: String,
    pub x: f64,
    pub y: f64,
    pub rotation: f64,
    pub is_locked: bool,
    pub pads: Vec<Pad>,
    pub width: f64,
    pub height: f64,
    pub courtyards: Vec<Polygon>,
    pub at_span: Option<(usize, usize)>,
}

impl Footprint {
    pub fn pad_absolute_position(&self, pad: &Pad) -> (f64, f64) {
        let p = Point::new(pad.offset_x, pad.offset_y).rotate_ccw(self.rotation);
        (self.x + p.x, self.y + p.y)
    }

    pub fn pad_world_point(&self, pad_idx: usize, pos: Point, rot: f64) -> Point {
        let pad = &self.pads[pad_idx];
        let p = Point::new(pad.offset_x, pad.offset_y).rotate_ccw(rot);
        Point::new(pos.x + p.x, pos.y + p.y)
    }

    pub fn transformed_courtyards(&self, pos: Point, rot: f64, spacing: f64) -> Vec<Polygon> {
        self.courtyards
            .iter()
            .map(|poly| {
                if spacing > 0.0 {
                    poly.expanded(spacing / 2.0).transform(pos, rot)
                } else {
                    poly.transform(pos, rot)
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackSegment {
    pub start: (f64, f64),
    pub end: (f64, f64),
    pub width: f64,
    pub layer: usize, // 0 = F.Cu, 1 = B.Cu
    pub net_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Via {
    pub position: (f64, f64),
    pub size: f64,
    pub drill: f64,
    pub net_id: usize,
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
    pub nets: Vec<String>,
    pub net_name_to_id: HashMap<String, usize>,
    pub netclasses: HashMap<String, NetClass>,
    pub net_to_class: HashMap<String, String>,
    pub keepouts: Vec<KeepoutZone>,
    pub tracks: Vec<TrackSegment>,
    pub vias: Vec<Via>,
    pub clusters: Vec<CompiledCluster>,
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

        // 2. Standard JLCPCB 2-layer NetClasses
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
            netclasses.insert(
                name.to_string(),
                NetClass {
                    name: name.to_string(),
                    track_width: width,
                    clearance,
                    via_dia: 0.60,
                    via_drill: 0.30,
                },
            );
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

        // 3. Parse Edge.Cuts, Keepouts, and Footprints
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut keepouts = Vec::new();
        let mut footprints = Vec::new();

        let bytes = content.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        while i < len {
            if bytes[i] == b'(' {
                let word = next_token(&content, i + 1);
                if word == "gr_rect" || word == "gr_line" {
                    if let Some(end) = find_closing_paren(bytes, i) {
                        let block = &content[i..end];
                        if block.contains("\"Edge.Cuts\"") {
                            extract_coords_into_bounds(block, &mut min_x, &mut min_y, &mut max_x, &mut max_y);
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
                                    min_x: if kmin_x > 170.0 { kmin_x.max(172.1) } else { kmin_x },
                                    min_y: kmin_y,
                                    max_x: kmax_x,
                                    max_y: kmax_y,
                                    layers: vec![0, 1],
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
                        i = end;
                        continue;
                    }
                }
            }
            i += 1;
        }

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
                layers: vec![0, 1],
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
            nets,
            net_name_to_id,
            netclasses,
            net_to_class,
            keepouts,
            tracks: Vec::new(),
            vias: Vec::new(),
            clusters: Vec::new(),
        };

        // Auto-load schematic functional clusters if sibling .kicad_sch exists
        let sch_path = Path::new(path).with_extension("kicad_sch");
        if sch_path.exists() {
            board.load_schematic_clusters(&sch_path);
        }

        Ok(board)
    }

    pub fn boundary_aabb(&self) -> AABB {
        AABB::new(self.min_x + 1.0, self.min_y + 1.0, self.max_x - 1.0, self.max_y - 1.0)
    }

    pub fn get_netclass_for_net(&self, net_id: usize) -> &NetClass {
        if let Some(name) = self.nets.get(net_id) {
            if let Some(cname) = self.net_to_class.get(name) {
                if let Some(nc) = self.netclasses.get(cname) {
                    return nc;
                }
            }
        }
        self.netclasses.get("Default").unwrap()
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
            // Prim's Euclidean MST
            let n = pts.len();
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
            if fp.is_locked {
                continue;
            }
            if let Some((start, end)) = fp.at_span {
                let rot_norm = ((fp.rotation % 360.0) + 360.0) % 360.0;
                let new_at = if rot_norm.abs() < 1e-3 || (rot_norm - 360.0).abs() < 1e-3 {
                    format!("(at {:.4} {:.4})", fp.x, fp.y)
                } else {
                    format!("(at {:.4} {:.4} {:.1})", fp.x, fp.y, rot_norm)
                };
                replacements.push((start, end, new_at));
            }
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
                let layer_name = if tr.layer == 0 { "F.Cu" } else { "B.Cu" };
                let net_name = self.nets.get(tr.net_id).map(|s| s.as_str()).unwrap_or("");
                s_expr.push_str(&format!(
                    "  (segment (start {:.4} {:.4}) (end {:.4} {:.4}) (width {:.4}) (layer \"{}\") (net \"{}\"))\n",
                    tr.start.0, tr.start.1, tr.end.0, tr.end.1, tr.width, layer_name, net_name
                ));
            }
            for v in &self.vias {
                let net_name = self.nets.get(v.net_id).map(|s| s.as_str()).unwrap_or("");
                s_expr.push_str(&format!(
                    "  (via (at {:.4} {:.4}) (size {:.4}) (drill {:.4}) (layers \"F.Cu\" \"B.Cu\") (net \"{}\"))\n",
                    v.position.0, v.position.1, v.size, v.drill, net_name
                ));
            }
            content.insert_str(last_paren, &s_expr);
        }

        fs::write(out_path, content).map_err(|e| e.to_string())
    }

    pub fn save_ses(&self, ses_path: &Path) -> Result<(), String> {
        let mut s = String::from("(session vulcans_trials\n  (base_design vulcans_trials)\n  (placement\n    (resolution mm 1000)\n");
        for fp in &self.footprints {
            s.push_str(&format!(
                "    (component \"{}\" (place \"{}\" {:.4} {:.4} front {:.1}))\n",
                fp.reference, fp.reference, fp.x, fp.y, fp.rotation
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
                    let layer_name = if tr.layer == 0 { "F.Cu" } else { "B.Cu" };
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
        if headers.is_empty() {
            return;
        }
        let mut members: HashMap<String, Vec<String>> = headers.iter().map(|h| (h.0.clone(), Vec::new())).collect();
        let sym_tag = "(symbol ";
        let mut s_idx = 0;
        while let Some(pos) = content[s_idx..].find(sym_tag) {
            let actual = s_idx + pos;
            let win = &content[actual..(actual + 1200).min(content.len())];
            if let (Some((sx, sy, _)), Some(r)) = (parse_at_xy_rot(win), extract_property(win, "Reference")) {
                if !r.starts_with('#') {
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
            s_idx = actual + sym_tag.len();
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
                max_radius,
                weight,
            });
        }
    }
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

fn parse_footprint(
    fp_str: &str,
    fp_byte_offset: usize,
    nets: &mut Vec<String>,
    net_name_to_id: &mut HashMap<String, usize>,
    extra_locked: &HashSet<String>,
) -> Option<Footprint> {
    let reference = extract_property(fp_str, "Reference")?;
    let value = extract_property(fp_str, "Value").unwrap_or_default();

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

    // Parse pads
    let mut pads = Vec::new();
    let mut p_idx = 0;
    while p_idx < bytes.len() {
        if bytes[p_idx] == b'(' && fp_str[p_idx..].starts_with("(pad ") {
            if let Some(p_end) = find_closing_paren(bytes, p_idx) {
                let p_str = &fp_str[p_idx..p_end];
                let tokens: Vec<&str> = p_str[5..].split_whitespace().collect();
                let p_num = tokens.first().map(|s| s.trim_matches('"').to_string()).unwrap_or_default();
                let is_th = tokens.get(1).copied() == Some("thru_hole");
                let (off_x, off_y, _) = parse_at_xy_rot(p_str).unwrap_or((0.0, 0.0, 0.0));
                let mut w = 0.5;
                let mut h = 0.5;
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
                let mut net_name = String::new();
                let mut net_id = 0usize;
                if let Some(n_pos) = p_str.find("(net ") {
                    let after = &p_str[n_pos + 5..];
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
                pads.push(Pad {
                    number: p_num,
                    offset_x: off_x,
                    offset_y: off_y,
                    width: w,
                    height: h,
                    net_id,
                    net_name,
                    is_through_hole: is_th,
                });
                p_idx = p_end;
                continue;
            }
        }
        p_idx += 1;
    }

    // Parse courtyard lines/rects on CrtYd
    let mut crtyd_pts = Vec::new();
    let mut c_idx = 0;
    while c_idx < bytes.len() {
        if bytes[c_idx] == b'(' && (fp_str[c_idx..].starts_with("(fp_line") || fp_str[c_idx..].starts_with("(fp_rect")) {
            if let Some(c_end) = find_closing_paren(bytes, c_idx) {
                let c_str = &fp_str[c_idx..c_end];
                if c_str.contains("CrtYd") {
                    let mut min_cx = f64::INFINITY;
                    let mut min_cy = f64::INFINITY;
                    let mut max_cx = f64::NEG_INFINITY;
                    let mut max_cy = f64::NEG_INFINITY;
                    extract_coords_into_bounds(c_str, &mut min_cx, &mut min_cy, &mut max_cx, &mut max_cy);
                    if min_cx < max_cx || min_cy < max_cy {
                        crtyd_pts.push(Point::new(min_cx, min_cy));
                        crtyd_pts.push(Point::new(max_cx, max_cy));
                    }
                }
                c_idx = c_end;
                continue;
            }
        }
        c_idx += 1;
    }

    let (min_cx, min_cy, max_cx, max_cy) = if !crtyd_pts.is_empty() {
        let poly = Polygon::new(crtyd_pts);
        let a = poly.aabb();
        (a.min_x, a.min_y, a.max_x, a.max_y)
    } else if !pads.is_empty() {
        let mut x0 = f64::INFINITY;
        let mut y0 = f64::INFINITY;
        let mut x1 = f64::NEG_INFINITY;
        let mut y1 = f64::NEG_INFINITY;
        for p in &pads {
            x0 = x0.min(p.offset_x - p.width / 2.0 - 0.25);
            x1 = x1.max(p.offset_x + p.width / 2.0 + 0.25);
            y0 = y0.min(p.offset_y - p.height / 2.0 - 0.25);
            y1 = y1.max(p.offset_y + p.height / 2.0 + 0.25);
        }
        (x0, y0, x1, y1)
    } else {
        (-1.0, -1.0, 1.0, 1.0)
    };

    let courtyard_poly = Polygon::new(vec![
        Point::new(min_cx, min_cy),
        Point::new(max_cx, min_cy),
        Point::new(max_cx, max_cy),
        Point::new(min_cx, max_cy),
    ]);

    Some(Footprint {
        reference,
        value,
        x: pos_x,
        y: pos_y,
        rotation: rot,
        is_locked,
        pads,
        width: (max_cx - min_cx).max(1.0),
        height: (max_cy - min_cy).max(1.0),
        courtyards: vec![courtyard_poly],
        at_span,
    })
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
        let pad_pos: Vec<Vec<Point>> = board
            .footprints
            .iter()
            .map(|fp| {
                (0..fp.pads.len())
                    .map(|pi| fp.pad_world_point(pi, Point::new(fp.x, fp.y), fp.rotation))
                    .collect()
            })
            .collect();
        let ratsnest = board.generate_ratsnest_segments(&pad_pos);
        let wirelength_mm: f64 = ratsnest.iter().map(|s| s.length()).sum();
        let mut crossings = 0;
        for i in 0..ratsnest.len() {
            for j in (i + 1)..ratsnest.len() {
                if ratsnest[i].crosses(&ratsnest[j]) {
                    crossings += 1;
                }
            }
        }
        Self {
            board,
            ratsnest,
            phase: TrialPhase::Idle,
            progress: 0.0,
            status_text: "Loaded board. Ready for Vulcan's Trials.".to_string(),
            log_lines: vec!["[Vulcan's Trials] Initialized board state.".to_string()],
            crossings,
            wirelength_mm,
            overlaps: 0,
            evals_per_sec: 0.0,
            routed_conns: 0,
            total_conns: 0,
            routing_pass: 0,
        }
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
        let r180 = p.rotate_ccw(180.0);
        assert!((r180.x + 2.0).abs() < 1e-6);
        assert!(r180.y.abs() < 1e-6);
        assert!((p.distance_to(&r180) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_sat_polygon_collision() {
        let sq = Polygon::new(vec![
            Point::new(-1.0, -1.0),
            Point::new(1.0, -1.0),
            Point::new(1.0, 1.0),
            Point::new(-1.0, 1.0),
        ]);
        let overlapping = sq.transform(Point::new(1.5, 0.0), 45.0);
        let separated = sq.transform(Point::new(3.5, 0.0), 45.0);
        assert!(sq.intersects_convex(&overlapping));
        assert!(!sq.intersects_convex(&separated));
    }

    #[test]
    fn test_segment_crossing_vs_shared_endpoint() {
        let s1 = Segment {
            p1: Point::new(0.0, 0.0),
            p2: Point::new(4.0, 4.0),
            net_id: 1,
        };
        let s2 = Segment {
            p1: Point::new(0.0, 4.0),
            p2: Point::new(4.0, 0.0),
            net_id: 2,
        };
        let s_same_net = Segment {
            p1: Point::new(0.0, 4.0),
            p2: Point::new(4.0, 0.0),
            net_id: 1,
        };
        let s_shared_pin = Segment {
            p1: Point::new(0.0, 0.0),
            p2: Point::new(4.0, 0.0),
            net_id: 2,
        };
        assert!(s1.crosses(&s2));
        assert!(!s1.crosses(&s_same_net));
        assert!(!s1.crosses(&s_shared_pin));
    }
}
