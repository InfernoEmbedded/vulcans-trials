// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! Differential-pair coupling/phase/length tuning, post-route topological track glossing,
//! via reduction/consolidation/sliding, and BGA dogbone fanout escape generation.

use super::*;

/// Aligns parallel horizontal and vertical segments of a routed differential pair
/// (`leader_net` and `follower_net`) to the exact target center-to-center pitch
/// `diff_width + diff_gap` when DRC-legal.
///
/// # Arguments
/// * `board` - Mutable [`Board`] whose follower trace segments are adjusted.
/// * `leader_net` - Numeric net ID of the leader trace.
/// * `follower_net` - Numeric net ID of the follower trace.
/// * `diff_width` - Differential trace width in `mm`.
/// * `diff_gap` - Target edge-to-edge differential gap in `mm`.
/// * `leader_start` - Leader start terminal `(x, y, layer)`.
/// * `leader_end` - Leader end terminal `(x, y, layer)`.
/// * `follower_start` - Follower start terminal `(x, y, layer)`.
/// * `follower_end` - Follower end terminal `(x, y, layer)`.
///
/// # Side Effects
/// Modifies `follower_net` segments in `board.tracks` in place to tighten coupled pitch.
pub fn enforce_coupled_diff_pair_spacing(
    board: &mut Board,
    leader_net: usize,
    follower_net: usize,
    diff_width: f64,
    diff_gap: f64,
    leader_start: (f64, f64, usize),
    leader_end: (f64, f64, usize),
    follower_start: (f64, f64, usize),
    follower_end: (f64, f64, usize),
) {
    let target_d = diff_width + diff_gap;
    let pad_margin = 1.5;

    let is_near_pad = |pt: (f64, f64)| -> bool {
        (pt.0 - leader_start.0).hypot(pt.1 - leader_start.1) <= pad_margin
            || (pt.0 - leader_end.0).hypot(pt.1 - leader_end.1) <= pad_margin
            || (pt.0 - follower_start.0).hypot(pt.1 - follower_start.1) <= pad_margin
            || (pt.0 - follower_end.0).hypot(pt.1 - follower_end.1) <= pad_margin
    };

    let n_tracks = board.tracks.len();
    for li in 0..n_tracks {
        if board.tracks[li].net_id != leader_net {
            continue;
        }
        let l_start = board.tracks[li].start;
        let l_end = board.tracks[li].end;
        let l_is_horiz = (l_start.1 - l_end.1).abs() < 1e-4;
        let l_is_vert = (l_start.0 - l_end.0).abs() < 1e-4;
        if !l_is_horiz && !l_is_vert {
            continue;
        }

        for fi in 0..n_tracks {
            if board.tracks[fi].net_id != follower_net || board.tracks[fi].layer != board.tracks[li].layer {
                continue;
            }
            let f_start = board.tracks[fi].start;
            let f_end = board.tracks[fi].end;
            let f_is_horiz = (f_start.1 - f_end.1).abs() < 1e-4;
            let f_is_vert = (f_start.0 - f_end.0).abs() < 1e-4;

            if l_is_horiz && f_is_horiz {
                let l_min_x = l_start.0.min(l_end.0);
                let l_max_x = l_start.0.max(l_end.0);
                let f_min_x = f_start.0.min(f_end.0);
                let f_max_x = f_start.0.max(f_end.0);
                let overlap = (l_max_x.min(f_max_x) - l_min_x.max(f_min_x)).max(0.0);
                if overlap > 0.40 {
                    let dy = f_start.1 - l_start.1;
                    if dy.abs() < 2.5 && dy.abs() > 0.05 {
                        let sign = dy.signum();
                        let target_y = l_start.1 + sign * target_d;
                        let old_y = f_start.1;
                        if !is_near_pad(f_start) && !is_near_pad(f_end) {
                            let cand_s = (board.tracks[fi].start.0, target_y);
                            let cand_e = (board.tracks[fi].end.0, target_y);
                            if is_track_segment_legal(board, cand_s, cand_e, board.tracks[fi].width, board.tracks[fi].layer, follower_net, 0.12) {
                                let mut neighbors_ok = true;
                                for oi in 0..n_tracks {
                                    if oi == fi || board.tracks[oi].net_id != follower_net {
                                        continue;
                                    }
                                    let mut ns = board.tracks[oi].start;
                                    let mut ne = board.tracks[oi].end;
                                    let mut touched = false;
                                    if (ns.0 - f_start.0).hypot(ns.1 - old_y) < 1e-3 || (ns.0 - f_end.0).hypot(ns.1 - old_y) < 1e-3 {
                                        ns.1 = target_y;
                                        touched = true;
                                    }
                                    if (ne.0 - f_start.0).hypot(ne.1 - old_y) < 1e-3 || (ne.0 - f_end.0).hypot(ne.1 - old_y) < 1e-3 {
                                        ne.1 = target_y;
                                        touched = true;
                                    }
                                    if touched && !board.footprints.is_empty()
                                        && !is_track_segment_legal(board, ns, ne, board.tracks[oi].width, board.tracks[oi].layer, follower_net, 0.12)
                                    {
                                        neighbors_ok = false;
                                        break;
                                    }
                                }
                                if neighbors_ok {
                                    board.tracks[fi].start.1 = target_y;
                                    board.tracks[fi].end.1 = target_y;
                                    for oi in 0..n_tracks {
                                        if oi == fi || board.tracks[oi].net_id != follower_net {
                                            continue;
                                        }
                                        if (board.tracks[oi].start.0 - f_start.0).hypot(board.tracks[oi].start.1 - old_y) < 1e-3 {
                                            board.tracks[oi].start.1 = target_y;
                                        }
                                        if (board.tracks[oi].end.0 - f_start.0).hypot(board.tracks[oi].end.1 - old_y) < 1e-3 {
                                            board.tracks[oi].end.1 = target_y;
                                        }
                                        if (board.tracks[oi].start.0 - f_end.0).hypot(board.tracks[oi].start.1 - old_y) < 1e-3 {
                                            board.tracks[oi].start.1 = target_y;
                                        }
                                        if (board.tracks[oi].end.0 - f_end.0).hypot(board.tracks[oi].end.1 - old_y) < 1e-3 {
                                            board.tracks[oi].end.1 = target_y;
                                        }
                                    }
                                }
                            }
                        } else if overlap > 2.0 * pad_margin + 0.40 {
                            let (x_min, x_max) = (f_min_x.max(l_min_x), f_max_x.min(l_max_x));
                            let coupled_x1 = x_min + pad_margin;
                            let coupled_x2 = x_max - pad_margin;
                            let fan_dx = (old_y - target_y).abs();
                            let orig_track = board.tracks[fi].clone();

                            let mut replacement = Vec::new();
                            let dir_sign = if f_start.0 < f_end.0 { 1.0 } else { -1.0 };
                            let (c_start_x, c_end_x) = if dir_sign > 0.0 {
                                (coupled_x1, coupled_x2)
                            } else {
                                (coupled_x2, coupled_x1)
                            };

                            let p_fan1_start = (c_start_x - dir_sign * fan_dx, old_y);
                            let p_fan1_end = (c_start_x, target_y);
                            let p_fan2_start = (c_end_x, target_y);
                            let p_fan2_end = (c_end_x + dir_sign * fan_dx, old_y);

                            if (f_start.0 - p_fan1_start.0).abs() > 1e-4 {
                                replacement.push(TrackSegment {
                                    start: f_start,
                                    end: p_fan1_start,
                                    width: orig_track.width,
                                    layer: orig_track.layer,
                                    net_id: follower_net,
                                    is_locked: false,
                                });
                            }
                            replacement.push(TrackSegment {
                                start: p_fan1_start,
                                end: p_fan1_end,
                                width: orig_track.width,
                                layer: orig_track.layer,
                                net_id: follower_net,
                                is_locked: false,
                            });
                            replacement.push(TrackSegment {
                                start: p_fan1_end,
                                end: p_fan2_start,
                                width: orig_track.width,
                                layer: orig_track.layer,
                                net_id: follower_net,
                                is_locked: false,
                            });
                            replacement.push(TrackSegment {
                                start: p_fan2_start,
                                end: p_fan2_end,
                                width: orig_track.width,
                                layer: orig_track.layer,
                                net_id: follower_net,
                                is_locked: false,
                            });
                            if (f_end.0 - p_fan2_end.0).abs() > 1e-4 {
                                replacement.push(TrackSegment {
                                    start: p_fan2_end,
                                    end: f_end,
                                    width: orig_track.width,
                                    layer: orig_track.layer,
                                    net_id: follower_net,
                                    is_locked: false,
                                });
                            }

                            if replacement.iter().all(|r| is_track_segment_legal(board, r.start, r.end, r.width, r.layer, follower_net, 0.12)) {
                                board.tracks.remove(fi);
                                for r in replacement {
                                    board.tracks.push(r);
                                }
                                break;
                            }
                        }
                    }
                }
            } else if l_is_vert && f_is_vert {
                let l_min_y = l_start.1.min(l_end.1);
                let l_max_y = l_start.1.max(l_end.1);
                let f_min_y = f_start.1.min(f_end.1);
                let f_max_y = f_start.1.max(f_end.1);
                let overlap = (l_max_y.min(f_max_y) - l_min_y.max(f_min_y)).max(0.0);
                if overlap > 0.40 {
                    let dx = f_start.0 - l_start.0;
                    if dx.abs() < 2.5 && dx.abs() > 0.05 {
                        let sign = dx.signum();
                        let target_x = l_start.0 + sign * target_d;
                        let old_x = f_start.0;
                        if !is_near_pad(f_start) && !is_near_pad(f_end) {
                            let cand_s = (target_x, board.tracks[fi].start.1);
                            let cand_e = (target_x, board.tracks[fi].end.1);
                            if is_track_segment_legal(board, cand_s, cand_e, board.tracks[fi].width, board.tracks[fi].layer, follower_net, 0.12) {
                                let mut neighbors_ok = true;
                                for oi in 0..n_tracks {
                                    if oi == fi || board.tracks[oi].net_id != follower_net {
                                        continue;
                                    }
                                    let mut ns = board.tracks[oi].start;
                                    let mut ne = board.tracks[oi].end;
                                    let mut touched = false;
                                    if (ns.0 - old_x).hypot(ns.1 - f_start.1) < 1e-3 || (ns.0 - old_x).hypot(ns.1 - f_end.1) < 1e-3 {
                                        ns.0 = target_x;
                                        touched = true;
                                    }
                                    if (ne.0 - old_x).hypot(ne.1 - f_start.1) < 1e-3 || (ne.0 - old_x).hypot(ne.1 - f_end.1) < 1e-3 {
                                        ne.0 = target_x;
                                        touched = true;
                                    }
                                    if touched && !board.footprints.is_empty()
                                        && !is_track_segment_legal(board, ns, ne, board.tracks[oi].width, board.tracks[oi].layer, follower_net, 0.12)
                                    {
                                        neighbors_ok = false;
                                        break;
                                    }
                                }
                                if neighbors_ok {
                                    board.tracks[fi].start.0 = target_x;
                                    board.tracks[fi].end.0 = target_x;
                                    for oi in 0..n_tracks {
                                        if oi == fi || board.tracks[oi].net_id != follower_net {
                                            continue;
                                        }
                                        if (board.tracks[oi].start.0 - old_x).hypot(board.tracks[oi].start.1 - f_start.1) < 1e-3 {
                                            board.tracks[oi].start.0 = target_x;
                                        }
                                        if (board.tracks[oi].end.0 - old_x).hypot(board.tracks[oi].end.1 - f_start.1) < 1e-3 {
                                            board.tracks[oi].end.0 = target_x;
                                        }
                                        if (board.tracks[oi].start.0 - old_x).hypot(board.tracks[oi].start.1 - f_end.1) < 1e-3 {
                                            board.tracks[oi].start.0 = target_x;
                                        }
                                        if (board.tracks[oi].end.0 - old_x).hypot(board.tracks[oi].end.1 - f_end.1) < 1e-3 {
                                            board.tracks[oi].end.0 = target_x;
                                        }
                                    }
                                }
                            }
                        } else if overlap > 2.0 * pad_margin + 0.40 {
                            let (y_min, y_max) = (f_min_y.max(l_min_y), f_max_y.min(l_max_y));
                            let coupled_y1 = y_min + pad_margin;
                            let coupled_y2 = y_max - pad_margin;
                            let fan_dy = (old_x - target_x).abs();
                            let orig_track = board.tracks[fi].clone();

                            let mut replacement = Vec::new();
                            let dir_sign = if f_start.1 < f_end.1 { 1.0 } else { -1.0 };
                            let (c_start_y, c_end_y) = if dir_sign > 0.0 {
                                (coupled_y1, coupled_y2)
                            } else {
                                (coupled_y2, coupled_y1)
                            };

                            let p_fan1_start = (old_x, c_start_y - dir_sign * fan_dy);
                            let p_fan1_end = (target_x, c_start_y);
                            let p_fan2_start = (target_x, c_end_y);
                            let p_fan2_end = (old_x, c_end_y + dir_sign * fan_dy);

                            if (f_start.1 - p_fan1_start.1).abs() > 1e-4 {
                                replacement.push(TrackSegment {
                                    start: f_start,
                                    end: p_fan1_start,
                                    width: orig_track.width,
                                    layer: orig_track.layer,
                                    net_id: follower_net,
                                    is_locked: false,
                                });
                            }
                            replacement.push(TrackSegment {
                                start: p_fan1_start,
                                end: p_fan1_end,
                                width: orig_track.width,
                                layer: orig_track.layer,
                                net_id: follower_net,
                                is_locked: false,
                            });
                            replacement.push(TrackSegment {
                                start: p_fan1_end,
                                end: p_fan2_start,
                                width: orig_track.width,
                                layer: orig_track.layer,
                                net_id: follower_net,
                                is_locked: false,
                            });
                            replacement.push(TrackSegment {
                                start: p_fan2_start,
                                end: p_fan2_end,
                                width: orig_track.width,
                                layer: orig_track.layer,
                                net_id: follower_net,
                                is_locked: false,
                            });
                            if (f_end.1 - p_fan2_end.1).abs() > 1e-4 {
                                replacement.push(TrackSegment {
                                    start: p_fan2_end,
                                    end: f_end,
                                    width: orig_track.width,
                                    layer: orig_track.layer,
                                    net_id: follower_net,
                                    is_locked: false,
                                });
                            }

                            if replacement.iter().all(|r| is_track_segment_legal(board, r.start, r.end, r.width, r.layer, follower_net, 0.12)) {
                                board.tracks.remove(fi);
                                for r in replacement {
                                    board.tracks.push(r);
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Matches the total routed copper length of differential pair nets `net_p` and `net_n`
/// within `tolerance_mm` by inserting small 45-degree phase-matching bumps on the shorter net.
///
/// # Arguments
/// * `board` - Mutable [`Board`] whose tracks are modified.
/// * `grid` - Mutable [`RoutingGrid`] used to verify obstacle clearance for bump vertices.
/// * `net_p` - Positive differential net ID.
/// * `net_n` - Negative differential net ID.
/// * `tolerance_mm` - Maximum permissible length skew in `mm`.
///
/// # Returns
/// `true` if the final skew between `net_p` and `net_n` is `<= tolerance_mm`.
///
/// # Side Effects
/// Replaces straight [`TrackSegment`]s on the shorter net in `board.tracks` with 45-degree bumps.
pub fn match_diff_pair_phase(
    board: &mut Board,
    grid: &mut RoutingGrid,
    net_p: usize,
    net_n: usize,
    tolerance_mm: f64,
) -> bool {
    let calc_net_len = |tracks: &[TrackSegment], nid: usize| -> f64 {
        tracks
            .iter()
            .filter(|t| t.net_id == nid)
            .map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1))
            .sum()
    };

    let len_p = calc_net_len(&board.tracks, net_p);
    let len_n = calc_net_len(&board.tracks, net_n);
    match_diff_pair_phase_lengths(board, grid, net_p, net_n, len_p, len_n, tolerance_mm)
}

/// Phase-matches a differential pair (`net_p`, `net_n`) given explicit signal-path lengths
/// `len_p` and `len_n` in `mm` (excluding secondary branches such as USB-C A/B pin jumpers
/// that would otherwise read as skew and insert spurious meanders).
///
/// # Arguments
/// * `board` - Mutable [`Board`] whose shorter net receives phase-matching bumps.
/// * `grid` - Mutable [`RoutingGrid`] for obstacle checking.
/// * `net_p` - Positive differential net ID.
/// * `net_n` - Negative differential net ID.
/// * `len_p` - Measured main-path length of `net_p` in `mm`.
/// * `len_n` - Measured main-path length of `net_n` in `mm`.
/// * `tolerance_mm` - Maximum permissible skew in `mm`.
///
/// # Returns
/// `true` if the remaining skew is `<= tolerance_mm`.
///
/// # Side Effects
/// Replaces straight segments on the shorter net in `board.tracks` with outward 45-degree trapezoid bumps.
pub fn match_diff_pair_phase_lengths(
    board: &mut Board,
    grid: &mut RoutingGrid,
    net_p: usize,
    net_n: usize,
    len_p: f64,
    len_n: f64,
    tolerance_mm: f64,
) -> bool {
    let skew = (len_p - len_n).abs();
    if skew <= tolerance_mm {
        return true;
    }

    let (short_net, long_net, delta_l) = if len_p < len_n {
        (net_p, net_n, len_n - len_p)
    } else {
        (net_n, net_p, len_p - len_n)
    };

    let mut candidates: Vec<TrackSegment> = board
        .tracks
        .iter()
        .filter(|t| t.net_id == short_net && !t.is_locked)
        .cloned()
        .collect();

    if candidates.is_empty() {
        return false;
    }

    candidates.sort_by(|a, b| {
        let la = (a.start.0 - a.end.0).hypot(a.start.1 - a.end.1);
        let lb = (b.start.0 - b.end.0).hypot(b.start.1 - b.end.1);
        lb.partial_cmp(&la).unwrap_or(Ordering::Equal)
    });

    let mut remaining_skew = delta_l;
    let k45 = 2.0 * (std::f64::consts::SQRT_2 - 1.0);

    // Small bumps keep the pair coupled, so large skews need several; straight remnants of a
    // bumped segment go back on the worklist.
    let mut queue = candidates;
    let mut bumps = 0;
    while !queue.is_empty() && bumps < 32 {
        let orig = queue.remove(0);
        if remaining_skew <= tolerance_mm {
            break;
        }
        let seg_len = (orig.start.0 - orig.end.0).hypot(orig.start.1 - orig.end.1);
        if seg_len < 1.50 {
            continue;
        }

        let dir = (
            (orig.end.0 - orig.start.0) / seg_len,
            (orig.end.1 - orig.start.1) / seg_len,
        );
        let norm = (-dir.1, dir.0);

        let mid = (
            (orig.start.0 + orig.end.0) * 0.5,
            (orig.start.1 + orig.end.1) * 0.5,
        );
        let dist_to_mate = |pt: (f64, f64)| -> f64 {
            let mut min_d = f64::INFINITY;
            for t in &board.tracks {
                if t.net_id == long_net {
                    let d = (t.start.0 - pt.0).hypot(t.start.1 - pt.1).min((t.end.0 - pt.0).hypot(t.end.1 - pt.1));
                    if d < min_d { min_d = d; }
                }
            }
            min_d
        };

        let d_pos = dist_to_mate((mid.0 + norm.0 * 0.1, mid.1 + norm.1 * 0.1));
        let d_neg = dist_to_mate((mid.0 - norm.0 * 0.1, mid.1 - norm.1 * 0.1));
        let out_sign = if d_pos >= d_neg { 1.0 } else { -1.0 };
        let out_norm = (norm.0 * out_sign, norm.1 * out_sign);

        // Keep skew bumps small (~2x a typical pair gap) so they don't decouple the pair.
        let max_h = 0.50f64.min((seg_len - 0.50) * 0.45);
        let h = (remaining_skew / k45).min(max_h);
        if h < 0.02 {
            continue;
        }

        let w_top = (seg_len - 2.0 * h - 0.20).min(0.40).max(0.20);
        let jog_span = 2.0 * h + w_top;
        if jog_span >= seg_len - 0.10 {
            continue;
        }
        let p1_offset = (seg_len - jog_span) * 0.5;
        let p1 = (orig.start.0 + dir.0 * p1_offset, orig.start.1 + dir.1 * p1_offset);
        let p2 = (
            p1.0 + dir.0 * h + out_norm.0 * h,
            p1.1 + dir.1 * h + out_norm.1 * h,
        );
        let p3 = (p2.0 + dir.0 * w_top, p2.1 + dir.1 * w_top);
        let p4 = (p1.0 + dir.0 * jog_span, p1.1 + dir.1 * jog_span);

        let check_free = |pt: (f64, f64)| -> bool {
            if let Some((gx, gy)) = grid.world_to_grid(pt.0, pt.1) {
                let idx = grid.idx(gx, gy, orig.layer);
                let occ = grid.obstacle_net_ids[idx];
                occ == FREE_NET || occ == short_net as i32
            } else {
                false
            }
        };

        if !check_free(p2) || !check_free(p3) {
            continue;
        }

        let mut replacement = Vec::new();
        if (orig.start.0 - p1.0).hypot(orig.start.1 - p1.1) > 1e-4 {
            replacement.push(TrackSegment {
                start: orig.start,
                end: p1,
                width: orig.width,
                layer: orig.layer,
                net_id: short_net,
                is_locked: false,
            });
        }
        replacement.push(TrackSegment {
            start: p1,
            end: p2,
            width: orig.width,
            layer: orig.layer,
            net_id: short_net,
            is_locked: false,
        });
        replacement.push(TrackSegment {
            start: p2,
            end: p3,
            width: orig.width,
            layer: orig.layer,
            net_id: short_net,
            is_locked: false,
        });
        replacement.push(TrackSegment {
            start: p3,
            end: p4,
            width: orig.width,
            layer: orig.layer,
            net_id: short_net,
            is_locked: false,
        });
        if (p4.0 - orig.end.0).hypot(p4.1 - orig.end.1) > 1e-4 {
            replacement.push(TrackSegment {
                start: p4,
                end: orig.end,
                width: orig.width,
                layer: orig.layer,
                net_id: short_net,
                is_locked: false,
            });
        }

        let clearance = board.get_netclass_for_net(short_net).clearance;
        if !board.footprints.is_empty()
            && !replacement.iter().all(|r| {
                is_track_segment_legal(board, r.start, r.end, r.width, r.layer, short_net, clearance)
            })
        {
            continue;
        }

        if let Some(pos) = board.tracks.iter().position(|t| t == &orig) {
            board.tracks.remove(pos);
            let seg_l = |t: &TrackSegment| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1);
            for r in &replacement {
                let straight = r.start == orig.start || r.end == orig.end;
                if straight && seg_l(r) >= 1.5 {
                    queue.push(r.clone());
                }
            }
            queue.sort_by(|a, b| seg_l(b).partial_cmp(&seg_l(a)).unwrap_or(Ordering::Equal));
            board.tracks.extend(replacement);
            bumps += 1;
            let added_len = k45 * h;
            remaining_skew = (remaining_skew - added_len).max(0.0);
        }
    }

    remaining_skew <= tolerance_mm
}

/// Tunes the total copper length of `net_id` up to `target_length_mm` by inserting
/// serpentine accordion/trombone meanders bounded by `max_amplitude_mm`.
///
/// # Arguments
/// * `board` - Mutable [`Board`] whose unlocked segments on `net_id` are lengthened.
/// * `grid` - Mutable [`RoutingGrid`] used to check lateral obstacle clearance.
/// * `net_id` - Numeric net ID to length-tune.
/// * `target_length_mm` - Desired total trace length in `mm`.
/// * `max_amplitude_mm` - Maximum lateral meander excursion in `mm`.
///
/// # Returns
/// `true` if the net length reached `target_length_mm - 0.05` or at least one meander was inserted.
///
/// # Side Effects
/// Replaces straight [`TrackSegment`]s of `net_id` in `board.tracks` with serpentine meander teeth.
pub fn tune_net_length(
    board: &mut Board,
    grid: &mut RoutingGrid,
    net_id: usize,
    target_length_mm: f64,
    max_amplitude_mm: f64,
) -> bool {
    let calc_net_len = |tracks: &[TrackSegment]| -> f64 {
        tracks
            .iter()
            .filter(|t| t.net_id == net_id)
            .map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1))
            .sum()
    };

    let mut current_len = calc_net_len(&board.tracks);
    if current_len >= target_length_mm - 0.05 {
        return true;
    }

    let mut needed = target_length_mm - current_len;
    let netclass = board.get_netclass_for_net(net_id);
    let clearance = netclass.clearance;
    let track_width = netclass.track_width;

    let pitch = (track_width + clearance).max(0.40);
    let max_amp = max_amplitude_mm.max(0.20);

    let candidates: Vec<TrackSegment> = board
        .tracks
        .iter()
        .filter(|t| t.net_id == net_id && !t.is_locked)
        .cloned()
        .collect();

    let mut sorted_candidates = candidates;
    sorted_candidates.sort_by(|a, b| {
        let la = (a.start.0 - a.end.0).hypot(a.start.1 - a.end.1);
        let lb = (b.start.0 - b.end.0).hypot(b.start.1 - b.end.1);
        lb.partial_cmp(&la).unwrap_or(Ordering::Equal)
    });

    let mut modified = false;

    for orig in sorted_candidates {
        if needed <= 0.05 {
            break;
        }

        let seg_len = (orig.start.0 - orig.end.0).hypot(orig.start.1 - orig.end.1);
        if seg_len < pitch + 0.30 {
            continue;
        }

        let dir = (
            (orig.end.0 - orig.start.0) / seg_len,
            (orig.end.1 - orig.start.1) / seg_len,
        );
        let norm = (-dir.1, dir.0);

        let check_side = |sign: f64| -> bool {
            let test_pts = [0.25, 0.5, 0.75];
            for &f in &test_pts {
                let px = orig.start.0 + f * (orig.end.0 - orig.start.0) + sign * norm.0 * max_amp;
                let py = orig.start.1 + f * (orig.end.1 - orig.start.1) + sign * norm.1 * max_amp;
                if let Some((gx, gy)) = grid.world_to_grid(px, py) {
                    let idx = grid.idx(gx, gy, orig.layer);
                    let occ = grid.obstacle_net_ids[idx];
                    if occ != FREE_NET && occ != net_id as i32 {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            true
        };

        let free_pos = check_side(1.0);
        let free_neg = check_side(-1.0);
        let (use_sign, alternate) = if free_pos && free_neg {
            (1.0, true)
        } else if free_pos {
            (1.0, false)
        } else if free_neg {
            (-1.0, false)
        } else {
            continue;
        };

        let margin = 0.20;
        let usable_len = seg_len - 2.0 * margin;
        let max_teeth = (usable_len / (pitch * 2.0)).floor() as usize;
        if max_teeth == 0 {
            continue;
        }

        let teeth_count = max_teeth.min(((needed / (2.0 * max_amp)).ceil() as usize).max(1));
        let amp_needed = (needed / (teeth_count as f64 * 2.0)).min(max_amp).max(0.10);

        let mut replacement = Vec::new();
        let total_teeth_span = teeth_count as f64 * pitch * 2.0;
        let start_offset = margin + (usable_len - total_teeth_span) / 2.0;

        let mut curr_pt = orig.start;
        let tooth_start_base = (
            orig.start.0 + dir.0 * start_offset,
            orig.start.1 + dir.1 * start_offset,
        );

        if (curr_pt.0 - tooth_start_base.0).hypot(curr_pt.1 - tooth_start_base.1) > 1e-4 {
            replacement.push(TrackSegment {
                start: curr_pt,
                end: tooth_start_base,
                width: orig.width,
                layer: orig.layer,
                net_id: orig.net_id,
                is_locked: false,
            });
            curr_pt = tooth_start_base;
        }

        for t in 0..teeth_count {
            let tooth_sign = if alternate {
                if t % 2 == 0 { 1.0 } else { -1.0 }
            } else {
                use_sign
            };
            let fold_amp = amp_needed * tooth_sign;
            let p_peak1 = (
                curr_pt.0 + norm.0 * fold_amp,
                curr_pt.1 + norm.1 * fold_amp,
            );
            replacement.push(TrackSegment {
                start: curr_pt,
                end: p_peak1,
                width: orig.width,
                layer: orig.layer,
                net_id: orig.net_id,
                is_locked: false,
            });
            let p_peak2 = (
                p_peak1.0 + dir.0 * pitch,
                p_peak1.1 + dir.1 * pitch,
            );
            replacement.push(TrackSegment {
                start: p_peak1,
                end: p_peak2,
                width: orig.width,
                layer: orig.layer,
                net_id: orig.net_id,
                is_locked: false,
            });
            let p_base2 = (
                p_peak2.0 - norm.0 * fold_amp,
                p_peak2.1 - norm.1 * fold_amp,
            );
            replacement.push(TrackSegment {
                start: p_peak2,
                end: p_base2,
                width: orig.width,
                layer: orig.layer,
                net_id: orig.net_id,
                is_locked: false,
            });
            curr_pt = p_base2;

            if t + 1 < teeth_count {
                let next_base = (
                    curr_pt.0 + dir.0 * pitch,
                    curr_pt.1 + dir.1 * pitch,
                );
                replacement.push(TrackSegment {
                    start: curr_pt,
                    end: next_base,
                    width: orig.width,
                    layer: orig.layer,
                    net_id: orig.net_id,
                    is_locked: false,
                });
                curr_pt = next_base;
            }
        }

        if (curr_pt.0 - orig.end.0).hypot(curr_pt.1 - orig.end.1) > 1e-4 {
            replacement.push(TrackSegment {
                start: curr_pt,
                end: orig.end,
                width: orig.width,
                layer: orig.layer,
                net_id: orig.net_id,
                is_locked: false,
            });
        }

        if !board.footprints.is_empty()
            && !replacement.iter().all(|r| {
                is_track_segment_legal(board, r.start, r.end, r.width, r.layer, net_id, clearance)
            })
        {
            continue;
        }

        let new_seg_len: f64 = replacement
            .iter()
            .map(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1))
            .sum();
        let added_len = new_seg_len - seg_len;

        if added_len > 0.0 {
            if let Some(pos) = board.tracks.iter().position(|t| t == &orig) {
                board.tracks.remove(pos);
                for r in replacement {
                    board.tracks.push(r);
                }
                needed = (needed - added_len).max(0.0);
                modified = true;
                if needed <= 0.05 {
                    break;
                }
            }
        }
    }

    current_len = calc_net_len(&board.tracks);
    current_len >= target_length_mm - 0.05 || modified
}

/// Contiguous single-layer polyline chain of connected [`TrackSegment`]s sharing a net and width.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackChain {
    /// Numeric net ID (`1..`).
    pub net_id: usize,
    /// Copper layer index (`0..num_layers`).
    pub layer: usize,
    /// Uniform trace width in `mm` along this chain.
    pub width: f64,
    /// Ordered sequence of 2D vertex coordinates `(x, y)` in `mm`.
    pub points: Vec<(f64, f64)>,
}

/// Quantizes a 2D world point `(x, y)` in `mm` to integer `0.1 um` (`1e-4 mm`) units
/// for exact topological endpoint matching.
///
/// # Arguments
/// * `p` - World coordinates `(x, y)` in `mm`.
///
/// # Returns
/// Quantized integer pair `(qx, qy)`.
pub(crate) fn quant_point(p: (f64, f64)) -> (i64, i64) {
    ((p.0 * 10000.0).round() as i64, (p.1 * 10000.0).round() as i64)
}

/// Extracts maximal degree-2 polyline [`TrackChain`]s from a slice of [`TrackSegment`]s,
/// splitting chains at pads, vias, T-junctions, layer changes, or trace width changes.
///
/// # Arguments
/// * `tracks` - Slice of [`TrackSegment`]s (typically belonging to one net).
/// * `board` - Reference to the [`Board`] used to identify pad, via, and locked-copper terminals.
///
/// # Returns
/// Vector of contiguous [`TrackChain`] polylines.
pub fn extract_track_chains(tracks: &[TrackSegment], board: &Board) -> Vec<TrackChain> {
    extract_track_chains_impl(tracks, board, false)
}

/// Extracts maximal degree-2 polyline [`TrackChain`]s for organic curve fitting, grouping
/// segments on the same layer together even if local neck-down widths vary.
///
/// # Arguments
/// * `tracks` - Slice of [`TrackSegment`]s.
/// * `board` - Reference to the [`Board`].
///
/// # Returns
/// Vector of contiguous [`TrackChain`] polylines using the minimum segment width in each chain.
pub fn extract_track_chains_for_organic(tracks: &[TrackSegment], board: &Board) -> Vec<TrackChain> {
    extract_track_chains_impl(tracks, board, true)
}

/// Internal implementation of [`extract_track_chains`] and [`extract_track_chains_for_organic`].
///
/// # Arguments
/// * `tracks` - Slice of [`TrackSegment`]s to chain.
/// * `board` - Reference to the [`Board`].
/// * `unify_signal_widths` - If `true`, segments of different widths on the same layer are chained together.
///
/// # Returns
/// Vector of extracted [`TrackChain`] polylines.
pub(crate) fn extract_track_chains_impl(
    tracks: &[TrackSegment],
    board: &Board,
    unify_signal_widths: bool,
) -> Vec<TrackChain> {
    if tracks.is_empty() {
        return Vec::new();
    }

    let mut groups: HashMap<(usize, i64), Vec<&TrackSegment>> = HashMap::new();
    for trk in tracks {
        let w_key = if unify_signal_widths {
            0
        } else {
            (trk.width * 1000.0).round() as i64
        };
        groups.entry((trk.layer, w_key)).or_default().push(trk);
    }

    let mut all_chains = Vec::new();

    for ((layer, w_key), segs) in groups {
        let net_id = segs[0].net_id;
        let width = segs.iter().map(|s| s.width).fold(segs[0].width, f64::min);
        let is_unified_signal = w_key == 0;

        let mut adj: HashMap<(i64, i64), Vec<(usize, (i64, i64))>> = HashMap::new();
        let mut pt_coords: HashMap<(i64, i64), (f64, f64)> = HashMap::new();
        let mut seen_edges: HashSet<((i64, i64), (i64, i64))> = HashSet::new();
        let mut visited_segs = vec![false; segs.len()];

        for (idx, seg) in segs.iter().enumerate() {
            let q1 = quant_point(seg.start);
            let q2 = quant_point(seg.end);
            if q1 == q2 {
                visited_segs[idx] = true;
                continue;
            }
            let edge_key = if q1 <= q2 { (q1, q2) } else { (q2, q1) };
            if !seen_edges.insert(edge_key) {
                visited_segs[idx] = true;
                continue;
            }
            pt_coords.entry(q1).or_insert(seg.start);
            pt_coords.entry(q2).or_insert(seg.end);
            adj.entry(q1).or_default().push((idx, q2));
            adj.entry(q2).or_default().push((idx, q1));
        }

        let is_terminal = |q: (i64, i64)| -> bool {
            if let Some(&pt) = pt_coords.get(&q) {
                let near_pad = board.footprints.iter().any(|fp| {
                    fp.pads.iter().any(|p| {
                        p.net_id == net_id
                            && (p.is_through_hole || fp.layer == layer)
                            && (fp.pad_absolute_position(p).0 - pt.0).hypot(fp.pad_absolute_position(p).1 - pt.1) < 0.015
                    })
                });
                let near_via = board.vias.iter().any(|v| {
                    v.net_id == net_id && (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.015
                });
                let near_other_track = tracks.iter().any(|t| {
                    t.net_id == net_id
                        && (t.layer != layer || (!is_unified_signal && (t.width - width).abs() > 1e-4))
                        && ((t.start.0 - pt.0).hypot(t.start.1 - pt.1) < 1e-3
                            || (t.end.0 - pt.0).hypot(t.end.1 - pt.1) < 1e-3)
                }) || board.tracks.iter().any(|t| {
                    t.is_locked
                        && t.net_id == net_id
                        && t.layer == layer
                        && ((t.start.0 - pt.0).hypot(t.start.1 - pt.1) < 0.05
                            || (t.end.0 - pt.0).hypot(t.end.1 - pt.1) < 0.05
                            || dist_point_to_segment(pt, t.start, t.end) < 0.05)
                }) || board.arcs.iter().any(|a| {
                    a.net_id == net_id
                        && a.layer == layer
                        && ((a.start().0 - pt.0).hypot(a.start().1 - pt.1) < 0.05
                            || (a.end().0 - pt.0).hypot(a.end().1 - pt.1) < 0.05)
                });
                near_pad || near_via || near_other_track
            } else {
                false
            }
        };

        let mut start_nodes: Vec<(i64, i64)> = adj
            .iter()
            .filter(|(&pt, edges)| edges.len() != 2 || is_terminal(pt))
            .map(|(&pt, _)| pt)
            .collect();
        start_nodes.sort_unstable();

        for start_pt in start_nodes {
            let edges = adj.get(&start_pt).cloned().unwrap_or_default();
            for (seg_idx, nxt_pt) in edges {
                if visited_segs[seg_idx] {
                    continue;
                }
                visited_segs[seg_idx] = true;
                let mut chain_w = segs[seg_idx].width;

                let mut chain_pts = vec![pt_coords[&start_pt], pt_coords[&nxt_pt]];
                let mut curr_pt = nxt_pt;

                while let Some(curr_edges) = adj.get(&curr_pt) {
                    if curr_edges.len() != 2 || is_terminal(curr_pt) {
                        break;
                    }
                    let next_edge = curr_edges.iter().find(|(e_idx, _)| !visited_segs[*e_idx]);
                    if let Some(&(nxt_seg, next_next_pt)) = next_edge {
                        visited_segs[nxt_seg] = true;
                        chain_w = chain_w.min(segs[nxt_seg].width);
                        chain_pts.push(pt_coords[&next_next_pt]);
                        curr_pt = next_next_pt;
                    } else {
                        break;
                    }
                }

                if chain_pts.len() >= 2 {
                    all_chains.push(TrackChain {
                        net_id,
                        layer,
                        width: chain_w,
                        points: chain_pts,
                    });
                }
            }
        }

        for idx in 0..segs.len() {
            if visited_segs[idx] {
                continue;
            }
            let seg = segs[idx];
            let q1 = quant_point(seg.start);
            let q2 = quant_point(seg.end);
            if q1 == q2 {
                continue;
            }
            visited_segs[idx] = true;
            let mut chain_w = seg.width;
            let mut chain_pts = vec![pt_coords[&q1], pt_coords[&q2]];
            let mut curr_pt = q2;

            while let Some(curr_edges) = adj.get(&curr_pt) {
                if curr_edges.len() != 2 || is_terminal(curr_pt) {
                    break;
                }
                let next_edge = curr_edges.iter().find(|(e_idx, _)| !visited_segs[*e_idx]);
                if let Some(&(nxt_seg, next_next_pt)) = next_edge {
                    visited_segs[nxt_seg] = true;
                    chain_w = chain_w.min(segs[nxt_seg].width);
                    chain_pts.push(pt_coords[&next_next_pt]);
                    curr_pt = next_next_pt;
                    if curr_pt == q1 {
                        break;
                    }
                } else {
                    break;
                }
            }

            if chain_pts.len() >= 2 {
                all_chains.push(TrackChain {
                    net_id,
                    layer,
                    width: chain_w,
                    points: chain_pts,
                });
            }
        }
    }

    all_chains
}

/// Simplifies a 2D polyline in place by removing duplicate vertices, redundant intermediate
/// collinear vertices, and zero-area 180-degree foldback spikes.
///
/// # Arguments
/// * `pts` - Mutable vector of `(x, y)` vertices in `mm`.
///
/// # Side Effects
/// Compacts `pts` in place.
pub fn merge_collinear_points(pts: &mut Vec<(f64, f64)>) {
    if pts.len() < 2 {
        return;
    }
    let mut cleaned: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
    for &pt in pts.iter() {
        if let Some(&last) = cleaned.last() {
            if (pt.0 - last.0).hypot(pt.1 - last.1) < 1e-4 {
                continue;
            }
        }
        while cleaned.len() >= 2 {
            let prev = cleaned[cleaned.len() - 2];
            let curr = cleaned[cleaned.len() - 1];
            let next = pt;

            let d_prev_curr = (curr.0 - prev.0).hypot(curr.1 - prev.1);
            let d_curr_next = (next.0 - curr.0).hypot(next.1 - curr.1);
            let d_prev_next = (next.0 - prev.0).hypot(next.1 - prev.1);

            let cross = (curr.0 - prev.0) * (next.1 - prev.1) - (curr.1 - prev.1) * (next.0 - prev.0);
            let dot = (curr.0 - prev.0) * (next.0 - curr.0) + (curr.1 - prev.1) * (next.1 - curr.1);
            let is_forward_collinear =
                cross.abs() < 1e-5 && dot > 0.0 && (d_prev_curr + d_curr_next - d_prev_next).abs() < 1e-4;
            let is_reverse_spike =
                dot < 0.0 && ((d_prev_curr - d_curr_next).abs() - d_prev_next).abs() < 1e-3;

            if is_forward_collinear || is_reverse_spike {
                cleaned.pop();
                if let Some(&last) = cleaned.last() {
                    if (pt.0 - last.0).hypot(pt.1 - last.1) < 1e-4 {
                        break;
                    }
                }
            } else {
                break;
            }
        }
        if cleaned.last().map_or(true, |&last| (pt.0 - last.0).hypot(pt.1 - last.1) >= 1e-4) {
            cleaned.push(pt);
        }
    }
    *pts = cleaned;
}

/// Counts the number of non-collinear interior bends along a 2D polyline `pts`.
///
/// # Arguments
/// * `pts` - Slice of `(x, y)` vertices in `mm`.
///
/// # Returns
/// Number of interior vertices where the unit tangent direction changes (`dot < 0.9999`).
pub fn count_chain_bends(pts: &[(f64, f64)]) -> usize {
    if pts.len() < 3 {
        return 0;
    }
    let mut bends = 0;
    for k in 1..pts.len() - 1 {
        let dx1 = pts[k].0 - pts[k - 1].0;
        let dy1 = pts[k].1 - pts[k - 1].1;
        let dx2 = pts[k + 1].0 - pts[k].0;
        let dy2 = pts[k + 1].1 - pts[k].1;
        let len1 = dx1.hypot(dy1);
        let len2 = dx2.hypot(dy2);
        if len1 < 1e-6 || len2 < 1e-6 {
            continue;
        }
        let dot = (dx1 * dx2 + dy1 * dy2) / (len1 * len2);
        if dot < 0.9999 {
            bends += 1;
        }
    }
    bends
}

/// Checks whether a candidate trace segment from `p1` to `p2` of `width` on `layer` for `net_id`
/// satisfies all board outline, keepout, dedicated-plane, foreign pad, foreign via, foreign track,
/// and same-net SMD pad grazing rules.
///
/// # Arguments
/// * `board` - Reference to the [`Board`].
/// * `grid` - Optional reference to the [`RoutingGrid`] for plane-net lookup.
/// * `p1` - Segment start `(x, y)` in `mm`.
/// * `p2` - Segment end `(x, y)` in `mm`.
/// * `layer` - Copper layer index.
/// * `net_id` - Numeric net ID of the segment.
/// * `width` - Trace width in `mm`.
///
/// # Returns
/// `true` if the segment is completely DRC-clean against all board obstacles.
pub fn is_segment_clear(
    board: &Board,
    grid: Option<&RoutingGrid>,
    p1: (f64, f64),
    p2: (f64, f64),
    layer: usize,
    net_id: usize,
    width: f64,
) -> bool {
    let clearance = board.get_netclass_for_net(net_id).clearance;
    let half_w = width * 0.5;
    let margin = half_w + clearance;

    // 0. Dedicated plane isolation
    if let Some(g) = grid {
        if g.plane_nets.contains_key(&layer) && g.plane_nets.get(&layer) != Some(&net_id) {
            return false;
        }
    } else if layer >= 2 {
        if let Some(&plane_net) = board.inner_plane_nets().get(&layer) {
            if plane_net != net_id {
                return false;
            }
        }
    }

    // 1. Board bounds check
    let min_x = p1.0.min(p2.0) - margin;
    let max_x = p1.0.max(p2.0) + margin;
    let min_y = p1.1.min(p2.1) - margin;
    let max_y = p1.1.max(p2.1) + margin;

    if let Some(ref poly) = board.outline_polygon {
        let boundary = poly.aabb();
        let edge_clr = board.get_netclass_for_net(net_id).board_edge_clearance;
        if min_x < boundary.min_x + edge_clr
            || max_x > boundary.max_x - edge_clr
            || min_y < boundary.min_y + edge_clr
            || max_y > boundary.max_y - edge_clr
        {
            return false;
        }
        if poly.vertices.len() > 4 {
            let p1_pt = Point::new(p1.0, p1.1);
            let p2_pt = Point::new(p2.0, p2.1);
            if !poly.contains_point(&p1_pt) || !poly.contains_point(&p2_pt)
                || poly.distance_to_segment(p1, p2) < edge_clr + half_w - 1e-4
            {
                return false;
            }
        }
    }

    // 2. Keepouts
    for ko in &board.keepouts {
        if ko.layers.contains(&layer) {
            let k_min_x = ko.min_x - margin;
            let k_max_x = ko.max_x + margin;
            let k_min_y = ko.min_y - margin;
            let k_max_y = ko.max_y + margin;

            if (p1.0 >= k_min_x && p1.0 <= k_max_x && p1.1 >= k_min_y && p1.1 <= k_max_y)
                || (p2.0 >= k_min_x && p2.0 <= k_max_x && p2.1 >= k_min_y && p2.1 <= k_max_y)
            {
                return false;
            }

            let corners = [
                (ko.min_x, ko.min_y),
                (ko.max_x, ko.min_y),
                (ko.max_x, ko.max_y),
                (ko.min_x, ko.max_y),
            ];
            for k in 0..4 {
                let e1 = corners[k];
                let e2 = corners[(k + 1) % 4];
                if dist_segment_to_segment(p1, p2, e1, e2) < margin - 1e-4 {
                    return false;
                }
            }
        }
    }

    let seg_min_x = p1.0.min(p2.0);
    let seg_max_x = p1.0.max(p2.0);
    let seg_min_y = p1.1.min(p2.1);
    let seg_max_y = p1.1.max(p2.1);
    let max_board_clr = board
        .netclasses
        .values()
        .map(|c| c.clearance)
        .fold(clearance, f64::max);

    // 3. Foreign Pads
    let min_hole_clr = board.get_netclass_for_net(net_id).min_hole_clearance;
    for fp in &board.footprints {
        for pad in &fp.pads {
            let pad_layer = fp.pad_layer(pad);
            if pad.net_id != net_id && (pad.is_through_hole || pad_layer == layer) {
                let max_dim = pad.width.max(pad.height);
                let coarse_box = half_w + max_board_clr.max(min_hole_clr) + max_dim;
                let (px, py) = fp.pad_absolute_position(pad);
                if px < seg_min_x - coarse_box
                    || px > seg_max_x + coarse_box
                    || py < seg_min_y - coarse_box
                    || py > seg_max_y + coarse_box
                {
                    continue;
                }
                let d_center = dist_to_segment(px, py, p1.0, p1.1, p2.0, p2.1);
                if pad.is_through_hole {
                    if let Some(drill) = pad.drill {
                        if drill > 0.0 && d_center < half_w + drill * 0.5 + min_hole_clr - 1e-4 {
                            return false;
                        }
                    }
                }
                let pad_clr = board.get_netclass_for_net(pad.net_id).clearance;
                let req_clr = clearance.max(pad_clr);
                let required_dist = half_w + req_clr;
                let pad_rad = 0.5 * pad.width.hypot(pad.height);
                if d_center >= required_dist + pad_rad - 1e-4 {
                    continue;
                }
                if dist_segment_to_pad(p1, p2, fp, pad) < required_dist - 1e-4 {
                    return false;
                }
            }
        }
    }

    // 4. Foreign Vias
    for via in &board.vias {
        if via.net_id != net_id {
            let coarse_v = half_w + via.size * 0.5 + max_board_clr.max(min_hole_clr);
            if via.position.0 < seg_min_x - coarse_v
                || via.position.0 > seg_max_x + coarse_v
                || via.position.1 < seg_min_y - coarse_v
                || via.position.1 > seg_max_y + coarse_v
            {
                continue;
            }
            let via_rad = via.size * 0.5;
            let via_clr = board.get_netclass_for_net(via.net_id).clearance;
            let req_clr = clearance.max(via_clr);
            let required_dist = half_w + (via_rad + req_clr).max(via.drill * 0.5 + min_hole_clr);
            if dist_to_segment(via.position.0, via.position.1, p1.0, p1.1, p2.0, p2.1) < required_dist - 1e-4 {
                return false;
            }
        }
    }

    // 5. Foreign Tracks
    for trk in &board.tracks {
        if trk.net_id != net_id && trk.layer == layer {
            let coarse_t = half_w + trk.width * 0.5 + max_board_clr;
            let t_min_x = trk.start.0.min(trk.end.0);
            let t_max_x = trk.start.0.max(trk.end.0);
            let t_min_y = trk.start.1.min(trk.end.1);
            let t_max_y = trk.start.1.max(trk.end.1);
            if t_max_x < seg_min_x - coarse_t
                || t_min_x > seg_max_x + coarse_t
                || t_max_y < seg_min_y - coarse_t
                || t_min_y > seg_max_y + coarse_t
            {
                continue;
            }
            let trk_clr = board.get_netclass_for_net(trk.net_id).clearance;
            let req_clr = clearance.max(trk_clr);
            let required_dist = half_w + (trk.width * 0.5) + req_clr;
            if dist_segment_to_segment(p1, p2, trk.start, trk.end) < required_dist - 1e-4 {
                return false;
            }
        }
    }

    // 6. RoutingGrid rasterized keepouts (or synthetic grid obstacles when board has no footprints)
    if let Some(g) = grid {
        let check_all_grid = board.footprints.is_empty();
        let seg_len = (p1.0 - p2.0).hypot(p1.1 - p2.1);
        let step = (g.res * 0.5).max(0.02);
        let n_steps = (seg_len / step).ceil() as usize;
        let n_steps = n_steps.max(1);
        for s in 0..=n_steps {
            let t = s as f64 / n_steps as f64;
            let x = p1.0 + t * (p2.0 - p1.0);
            let y = p1.1 + t * (p2.1 - p1.1);
            if let Some((gx, gy)) = g.world_to_grid(x, y) {
                let idx = g.idx(gx, gy, layer);
                let cell_net = g.obstacle_net_ids[idx];
                if cell_net == KEEPOUT_NET || (check_all_grid && cell_net != FREE_NET && cell_net != net_id as i32) {
                    return false;
                }
            }
        }
    }

    true
}

/// Performs rubber-band octilinear line-of-sight shortcutting on a [`TrackChain`],
/// collapsing intermediate vertices whenever a direct octilinear segment between non-adjacent
/// vertices is DRC-clear and does not bypass any pad or via anchor.
///
/// # Arguments
/// * `chain` - Mutable [`TrackChain`] to straighten.
/// * `board` - Reference to the [`Board`].
/// * `grid` - Reference to the [`RoutingGrid`].
///
/// # Side Effects
/// Replaces `chain.points` with a simplified vertex sequence.
pub fn rubber_band_straighten(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    rubber_band_straighten_gpu(chain, board, grid, None);
}

/// Replaces multi-bend staircase zig-zags along a [`TrackChain`] with canonical 1-bend
/// 45-degree dogleg corners (`[p_i, corner, p_j]`) when DRC-clear.
///
/// # Arguments
/// * `chain` - Mutable [`TrackChain`] to simplify.
/// * `board` - Reference to the [`Board`].
/// * `grid` - Reference to the [`RoutingGrid`].
///
/// # Side Effects
/// Mutates `chain.points` in place.
pub fn eliminate_stair_steps(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    eliminate_stair_steps_gpu(chain, board, grid, None);
}

/// Collapses short lateral S-bends and micro-jogs (`<= max(2 * width, 0.30 mm)`) along a
/// [`TrackChain`] when a direct octilinear bypass is DRC-clear.
///
/// # Arguments
/// * `chain` - Mutable [`TrackChain`] to simplify.
/// * `board` - Reference to the [`Board`].
/// * `grid` - Reference to the [`RoutingGrid`].
///
/// # Side Effects
/// Removes redundant micro-jog vertices from `chain.points`.
pub fn reduce_jogs(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    reduce_jogs_gpu(chain, board, grid, None);
}

/// Eliminates acute interior trace corners (`< 90` degrees) and 180-degree foldback spikes
/// along a [`TrackChain`] by replacing them with direct bypasses or obtuse 45-degree miters
/// so etching acid cannot be trapped in sharp copper wedges.
///
/// # Arguments
/// * `chain` - Mutable [`TrackChain`] to clean.
/// * `board` - Reference to the [`Board`].
/// * `grid` - Reference to the [`RoutingGrid`].
///
/// # Side Effects
/// Modifies `chain.points` in place to remove acute vertices.
pub fn avoid_acid_traps(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    if chain.points.len() < 3 {
        return;
    }
    let is_pad_center = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        })
    };
    let is_special_point = |pt: (f64, f64)| -> bool {
        is_pad_center(pt) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };
    let is_oct = is_octilinear_segment;

    let mut changed = true;
    let mut pass = 0;
    while changed && pass < 4 {
        changed = false;
        pass += 1;
        let mut i = 1;
        while i + 1 < chain.points.len() && chain.points.len() < 1024 {
            let p0 = chain.points[i - 1];
            let p1 = chain.points[i];
            let p2 = chain.points[i + 1];

            if is_special_point(p1) {
                i += 1;
                continue;
            }

            let v1 = (p1.0 - p0.0, p1.1 - p0.1);
            let v2 = (p2.0 - p1.0, p2.1 - p1.1);
            let l1 = v1.0.hypot(v1.1);
            let l2 = v2.0.hypot(v2.1);

            if l1 < 1e-4 || l2 < 1e-4 {
                i += 1;
                continue;
            }

            let dot = (v1.0 * v2.0 + v1.1 * v2.1) / (l1 * l2);

            // 180-degree fold-back spike: p0 -> p2 is already covered by p0 -> p1 or p1 -> p2
            if dot < -0.98 {
                chain.points.remove(i);
                merge_collinear_points(&mut chain.points);
                changed = true;
                continue;
            }

            // Acute angle: turn angle > 90 degrees (dot product < -0.05)
            // creating an interior acute wedge (< 90 degrees) that traps etching acid
            if dot < -0.05 {
                // 1. Direct bypass across acute corner: p0 -> p2
                if is_oct(p0, p2) && is_segment_clear(board, Some(grid), p0, p2, chain.layer, chain.net_id, chain.width) {
                    chain.points.remove(i);
                    merge_collinear_points(&mut chain.points);
                    changed = true;
                    continue;
                }

                // 2. Mitered obtuse bypass: cut back along v1 and v2 with candidate setbacks
                let u1 = (v1.0 / l1, v1.1 / l1);
                let u2 = (v2.0 / l2, v2.1 / l2);

                let setbacks = [1.50, 1.25, 1.00, 0.75, 0.50, 0.35];
                let mut resolved = false;

                for &sb_factor in &setbacks {
                    let sb_base = sb_factor * chain.width;
                    let candidate_pairs = [
                        (sb_base, sb_base),
                        (sb_base, sb_base * std::f64::consts::SQRT_2),
                        (sb_base * std::f64::consts::SQRT_2, sb_base),
                    ];

                    for &(s1, s2) in &candidate_pairs {
                        if s1 >= 0.45 * l1 || s2 >= 0.45 * l2 || s1 < 0.02 || s2 < 0.02 {
                            continue;
                        }

                        let m1 = (p1.0 - s1 * u1.0, p1.1 - s1 * u1.1);
                        let m2 = (p1.0 + s2 * u2.0, p1.1 + s2 * u2.1);
                        let vm = (m2.0 - m1.0, m2.1 - m1.1);
                        let lm = vm.0.hypot(vm.1);
                        if lm < 1e-3 {
                            continue;
                        }
                        let um = (vm.0 / lm, vm.1 / lm);
                        let dot1 = u1.0 * um.0 + u1.1 * um.1;
                        let dot2 = um.0 * u2.0 + um.1 * u2.1;
                        if dot1 < -0.02 || dot2 < -0.02 {
                            continue;
                        }

                        if is_oct(m1, m2) && is_segment_clear(board, Some(grid), m1, m2, chain.layer, chain.net_id, chain.width) {
                            chain.points[i] = m1;
                            chain.points.insert(i + 1, m2);
                            merge_collinear_points(&mut chain.points);
                            resolved = true;
                            changed = true;
                            break;
                        }
                    }
                    if resolved {
                        break;
                    }
                }

                if resolved {
                    i += 1;
                    continue;
                }
            }

            i += 1;
        }
    }
}

/// Collapses 3-segment U-turns and 2-segment corners across sliding 4-point windows on a
/// [`TrackChain`] whenever a direct octilinear shortcut is DRC-clear.
///
/// # Arguments
/// * `chain` - Mutable [`TrackChain`] to simplify.
/// * `board` - Reference to the [`Board`].
/// * `grid` - Reference to the [`RoutingGrid`].
///
/// # Side Effects
/// Removes redundant corner vertices from `chain.points`.
pub fn reduce_corners(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    reduce_corners_gpu(chain, board, grid, None);
}

/// Shifts axis-aligned and 45-degree diagonal segments of a [`TrackChain`] laterally by
/// `50..150 um` away from nearby foreign obstacles when doing so strictly improves clearance
/// according to the Euclidean distance field `grid.dist_field`.
///
/// # Arguments
/// * `chain` - Mutable [`TrackChain`] whose segments are repositioned.
/// * `board` - Reference to the [`Board`].
/// * `grid` - Reference to the [`RoutingGrid`].
///
/// # Side Effects
/// Updates interior vertex coordinates in `chain.points`.
pub fn reposition_lines(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    if chain.points.len() < 4 {
        return;
    }
    let is_pad_center = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        })
    };
    let is_special_point = |pt: (f64, f64)| -> bool {
        is_pad_center(pt) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };

    let sample_min_obstacle_dist = |p1: (f64, f64), p2: (f64, f64)| -> f64 {
        let mut min_d = f64::INFINITY;
        let samples = [0.15, 0.35, 0.50, 0.65, 0.85];
        for &t in &samples {
            let sx = p1.0 + t * (p2.0 - p1.0);
            let sy = p1.1 + t * (p2.1 - p1.1);
            if let Some((gx, gy)) = grid.world_to_grid(sx, sy) {
                if gx < grid.width && gy < grid.height {
                    let d = grid.dist_field[grid.idx(gx, gy, chain.layer)] as f64;
                    if d < min_d {
                        min_d = d;
                    }
                }
            }
        }
        min_d
    };

    let mut i = 0;
    while i + 3 < chain.points.len() {
        let p0 = chain.points[i];
        let p1 = chain.points[i + 1];
        let p2 = chain.points[i + 2];
        let p3 = chain.points[i + 3];

        if is_special_point(p1) || is_special_point(p2) {
            i += 1;
            continue;
        }

        let dx = p2.0 - p1.0;
        let dy = p2.1 - p1.1;
        let seg_len = dx.hypot(dy);

        // Only evaluate non-trivial axis-aligned segments
        if seg_len < 0.35 {
            i += 1;
            continue;
        }

        let is_horiz = dy.abs() < 1e-4 && dx.abs() > 1e-4;
        let is_vert = dx.abs() < 1e-4 && dy.abs() > 1e-4;
        let is_diag45 = (dx.abs() - dy.abs()).abs() < 1e-4 && dx.abs() > 1e-4;

        if is_diag45 {
            let p01_horiz = (p1.1 - p0.1).abs() < 1e-4 && (p1.0 - p0.0).abs() > 1e-4;
            let p01_vert = (p1.0 - p0.0).abs() < 1e-4 && (p1.1 - p0.1).abs() > 1e-4;
            let p23_horiz = (p3.1 - p2.1).abs() < 1e-4 && (p3.0 - p2.0).abs() > 1e-4;
            let p23_vert = (p3.0 - p2.0).abs() < 1e-4 && (p3.1 - p2.1).abs() > 1e-4;
            if (p01_horiz && p23_vert) || (p01_vert && p23_horiz) {
                let min_trk_dist = |a: (f64, f64), b: (f64, f64)| -> f64 {
                    board
                        .tracks
                        .iter()
                        .filter(|t| t.net_id != chain.net_id && t.layer == chain.layer)
                        .map(|t| dist_segment_to_segment(a, b, t.start, t.end) - (chain.width + t.width) * 0.5)
                        .fold(f64::INFINITY, f64::min)
                };
                let cur_gap = min_trk_dist(p1, p2);
                if cur_gap < 0.15 {
                    for &s in &[0.05, -0.05, 0.10, -0.10, 0.15, -0.15] {
                        let (p1_new, p2_new) = if p01_horiz {
                            ((p1.0 + s, p1.1), (p2.0, p2.1 - s * dx.signum() * dy.signum()))
                        } else {
                            ((p1.0, p1.1 + s), (p2.0 - s * dx.signum() * dy.signum(), p2.1))
                        };
                        if (p1_new.0 - p0.0).hypot(p1_new.1 - p0.1) < 0.05
                            || (p3.0 - p2_new.0).hypot(p3.1 - p2_new.1) < 0.05
                            || (p2_new.0 - p1_new.0).hypot(p2_new.1 - p1_new.1) < 0.10
                        {
                            continue;
                        }
                        if min_trk_dist(p1_new, p2_new) >= 0.155
                            && is_segment_clear(board, Some(grid), p0, p1_new, chain.layer, chain.net_id, chain.width)
                            && is_segment_clear(board, Some(grid), p1_new, p2_new, chain.layer, chain.net_id, chain.width)
                            && is_segment_clear(board, Some(grid), p2_new, p3, chain.layer, chain.net_id, chain.width)
                        {
                            chain.points[i + 1] = p1_new;
                            chain.points[i + 2] = p2_new;
                            break;
                        }
                    }
                }
            }
            i += 1;
            continue;
        }

        if !is_horiz && !is_vert {
            i += 1;
            continue;
        }

        // Check if connecting legs p0->p1 and p2->p3 are perpendicular to p1->p2
        let p0_p1_perp = if is_horiz {
            (p1.0 - p0.0).abs() < 1e-4 && (p1.1 - p0.1).abs() > 1e-4
        } else {
            (p1.1 - p0.1).abs() < 1e-4 && (p1.0 - p0.0).abs() > 1e-4
        };
        let p2_p3_perp = if is_horiz {
            (p3.0 - p2.0).abs() < 1e-4 && (p3.1 - p2.1).abs() > 1e-4
        } else {
            (p3.1 - p2.1).abs() < 1e-4 && (p3.0 - p2.0).abs() > 1e-4
        };

        if !p0_p1_perp || !p2_p3_perp {
            i += 1;
            continue;
        }

        let normal = if is_horiz { (0.0, 1.0) } else { (1.0, 0.0) };
        let d_curr = sample_min_obstacle_dist(p1, p2);

        let candidate_offsets = [0.10, 0.05, -0.05, -0.10];
        let mut best_shift = None;
        let mut best_dist = d_curr;

        for &shift in &candidate_offsets {
            let p1_new = (p1.0 + normal.0 * shift, p1.1 + normal.1 * shift);
            let p2_new = (p2.0 + normal.0 * shift, p2.1 + normal.1 * shift);

            let l_in = (p1_new.0 - p0.0).hypot(p1_new.1 - p0.1);
            let l_out = (p3.0 - p2_new.0).hypot(p3.1 - p2_new.1);
            if l_in < 0.05 || l_out < 0.05 {
                continue;
            }

            if !is_segment_clear(board, Some(grid), p0, p1_new, chain.layer, chain.net_id, chain.width) {
                continue;
            }
            if !is_segment_clear(board, Some(grid), p1_new, p2_new, chain.layer, chain.net_id, chain.width) {
                continue;
            }
            if !is_segment_clear(board, Some(grid), p2_new, p3, chain.layer, chain.net_id, chain.width) {
                continue;
            }

            let d_cand = sample_min_obstacle_dist(p1_new, p2_new);
            // Accept shift if it strictly increases clearance to the closest obstacle by >= 0.03mm
            if d_cand > best_dist + 0.03 {
                best_dist = d_cand;
                best_shift = Some((p1_new, p2_new));
            }
        }

        if let Some((p1_new, p2_new)) = best_shift {
            chain.points[i + 1] = p1_new;
            chain.points[i + 2] = p2_new;
        }

        i += 1;
    }
}

/// Chamfers sharp 90-degree corners along a [`TrackChain`] into pairs of 135-degree
/// (45-degree miter) bends using adaptive setbacks (`0.35x` to `1.50x` trace width).
///
/// # Arguments
/// * `chain` - Mutable [`TrackChain`] to chamfer.
/// * `board` - Reference to the [`Board`].
/// * `grid` - Reference to the [`RoutingGrid`].
///
/// # Side Effects
/// Replaces 90-degree corner vertices in `chain.points` with mitered vertex pairs.
pub fn smoothen_corners(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    if chain.points.len() < 3 {
        return;
    }
    let is_pad_center = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        })
    };
    let is_special_point = |pt: (f64, f64)| -> bool {
        is_pad_center(pt) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };

    let mut new_points = Vec::with_capacity(chain.points.len() * 2);
    new_points.push(chain.points[0]);

    for i in 1..chain.points.len() - 1 {
        let prev = *new_points.last().unwrap();
        let curr = chain.points[i];
        let next = chain.points[i + 1];

        if is_special_point(curr) {
            new_points.push(curr);
            continue;
        }

        let v1 = (curr.0 - prev.0, curr.1 - prev.1);
        let v2 = (next.0 - curr.0, next.1 - curr.1);
        let l1 = v1.0.hypot(v1.1);
        let l2 = v2.0.hypot(v2.1);

        if l1 < 1e-4 || l2 < 1e-4 {
            new_points.push(curr);
            continue;
        }

        let dot = (v1.0 * v2.0 + v1.1 * v2.1) / (l1 * l2);
        let cross = (v1.0 * v2.1 - v1.1 * v2.0) / (l1 * l2);

        let is_90_deg = dot.abs() < 0.05 && cross.abs() > 0.95;

        if is_90_deg {
            // Adaptive 45° chamfering: try largest setback first, falling back to tighter clearances
            let candidate_multipliers = [1.50, 1.25, 1.00, 0.75, 0.50, 0.35];
            let mut chamfered = false;

            for &mult in &candidate_multipliers {
                let setback = (mult * chain.width).min(0.45 * l1).min(0.45 * l2);
                if setback < 0.02 {
                    continue;
                }
                let m1 = (curr.0 - setback * (v1.0 / l1), curr.1 - setback * (v1.1 / l1));
                let m2 = (curr.0 + setback * (v2.0 / l2), curr.1 + setback * (v2.1 / l2));

                if is_octilinear_segment(m1, m2)
                    && is_segment_clear(board, Some(grid), m1, m2, chain.layer, chain.net_id, chain.width)
                {
                    new_points.push(m1);
                    new_points.push(m2);
                    chamfered = true;
                    break;
                }
            }

            if chamfered {
                continue;
            }
        }

        new_points.push(curr);
    }

    new_points.push(*chain.points.last().unwrap());
    chain.points = new_points;
}

/// Alias for [`smoothen_corners`] that applies adaptive 45-degree miters to 90-degree corners.
///
/// # Arguments
/// * `chain` - Mutable [`TrackChain`] to miter.
/// * `board` - Reference to the [`Board`].
/// * `grid` - Reference to the [`RoutingGrid`].
///
/// # Side Effects
/// Mutates `chain.points` in place.
pub fn smooth_45_miters(chain: &mut TrackChain, board: &Board, grid: &RoutingGrid) {
    smoothen_corners(chain, board, grid);
}

/// Runs up to 8 iterative passes of via-pair hop elimination, single-via consolidation,
/// and via sliding/alignment on `board` using [`RoutingConfig::default`], followed by
/// dangling-copper pruning.
///
/// # Arguments
/// * `board` - Mutable [`Board`] whose vias and tracks are simplified.
/// * `grid` - Mutable [`RoutingGrid`].
///
/// # Side Effects
/// Removes redundant vias from `board.vias` and updates `board.tracks`.
pub fn reduce_vias(board: &mut Board, grid: &mut RoutingGrid) {
    reduce_vias_with_config(board, grid, &RoutingConfig::default());
}

/// Runs up to 8 iterative passes of via-pair hop elimination, single-via consolidation,
/// and via sliding/alignment on `board` using the provided [`RoutingConfig`].
///
/// # Arguments
/// * `board` - Mutable [`Board`] whose vias and tracks are simplified.
/// * `grid` - Mutable [`RoutingGrid`].
/// * `config` - [`RoutingConfig`] controlling via merge radii and detour budgets.
///
/// # Side Effects
/// Mutates `board.vias` and `board.tracks`, concluding with [`prune_dangling_tracks_and_vias`].
pub fn reduce_vias_with_config(board: &mut Board, grid: &mut RoutingGrid, config: &RoutingConfig) {
    let max_passes = 8;
    for _pass in 0..max_passes {
        let changed_pairs = eliminate_via_pairs_with_config(board, grid, config);
        let changed_single = consolidate_single_vias_with_config(board, grid, config);
        let changed_slide = slide_and_align_vias_with_config(board, grid, config);
        if !changed_pairs && !changed_single && !changed_slide {
            break;
        }
    }
    prune_dangling_tracks_and_vias(board);
}

/// Executes a single pass of via-pair elimination, single-via consolidation, and via
/// sliding/alignment on `board` using [`RoutingConfig::default`].
///
/// # Arguments
/// * `board` - Mutable [`Board`].
/// * `grid` - Reference to the [`RoutingGrid`].
///
/// # Returns
/// `true` if any via was removed, merged, or slid during the pass.
pub fn reduce_vias_single_pass(board: &mut Board, grid: &RoutingGrid) -> bool {
    let config = RoutingConfig::default();
    let p = eliminate_via_pairs_with_config(board, grid, &config);
    let s = consolidate_single_vias_with_config(board, grid, &config);
    let l = slide_and_align_vias_with_config(board, grid, &config);
    p || s || l
}

/// Eliminates redundant layer-hopping via pairs (`L_orig -> L_alt -> L_orig`) and merges
/// adjacent same-net vias using [`RoutingConfig::default`].
///
/// # Arguments
/// * `board` - Mutable [`Board`].
/// * `grid` - Reference to the [`RoutingGrid`].
///
/// # Returns
/// `true` if at least one via pair was eliminated or consolidated.
pub fn eliminate_via_pairs(board: &mut Board, grid: &RoutingGrid) -> bool {
    eliminate_via_pairs_with_config(board, grid, &RoutingConfig::default())
}

/// Eliminates redundant layer-hopping via pairs (`L_orig -> L_alt -> L_orig`) and merges
/// adjacent same-net vias using the thresholds in `config`.
///
/// # Arguments
/// * `board` - Mutable [`Board`].
/// * `grid` - Reference to the [`RoutingGrid`].
/// * `config` - [`RoutingConfig`] specifying `via_hop_max_detour_ratio`, `via_merge_radius_mm`, and `via_detour_budget_mm`.
///
/// # Returns
/// `true` if at least one via mutation was applied across up to 20 sub-iterations.
pub fn eliminate_via_pairs_with_config(board: &mut Board, grid: &RoutingGrid, config: &RoutingConfig) -> bool {
    let mut any_changed = false;
    for _ in 0..20 {
        if !eliminate_via_pair_once(board, grid, config) {
            break;
        }
        any_changed = true;
    }
    any_changed
}

/// Tests whether two 2D axis-aligned bounding boxes `(min_x, min_y, max_x, max_y)` overlap.
///
/// # Arguments
/// * `a` - First AABB `(min_x, min_y, max_x, max_y)` in `mm`.
/// * `b` - Second AABB `(min_x, min_y, max_x, max_y)` in `mm`.
///
/// # Returns
/// `true` if `a` and `b` intersect or touch.
#[inline]
pub(crate) fn aabb_overlaps(a: (f64, f64, f64, f64), b: (f64, f64, f64, f64)) -> bool {
    a.0 <= b.2 && a.2 >= b.0 && a.1 <= b.3 && a.3 >= b.1
}

/// Computes the 2D axis-aligned bounding box `(min_x, min_y, max_x, max_y)` enclosing `pts`,
/// expanded outward on all sides by `margin` in `mm`.
///
/// # Arguments
/// * `pts` - Non-empty slice of `(x, y)` coordinates in `mm`.
/// * `margin` - Expansion margin in `mm`.
///
/// # Returns
/// Expanded AABB tuple `(min_x, min_y, max_x, max_y)`.
#[inline]
pub(crate) fn points_expanded_aabb(pts: &[(f64, f64)], margin: f64) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &(x, y) in pts {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
}

/// Atomic via-reduction mutation discovered for a single net during [`eliminate_via_pair_once`].
pub(crate) enum ViaPairMutation {
    /// Promotes a layer-hopping trace chain from `l_alt` back onto `l_orig` and drops both bounding vias.
    EliminateHop {
        /// Numeric net ID.
        net_id: usize,
        /// Target copper layer onto which the hop is promoted.
        l_orig: usize,
        /// Trace width in `mm`.
        track_width: f64,
        /// Indices of existing [`TrackSegment`]s in `board.tracks` to remove.
        chain_segs: Vec<usize>,
        /// Replacement polyline vertices `(x, y)` in `mm` on `l_orig`.
        new_pts: Vec<(f64, f64)>,
        /// Quantized position of the first eliminated via.
        q1: (i64, i64),
        /// Quantized position of the second eliminated via.
        q2: (i64, i64),
        /// Expanded spatial bounding box `(min_x, min_y, max_x, max_y)` in `mm` for conflict-free batching.
        aabb: (f64, f64, f64, f64),
    },
    /// Merges a redundant nearby same-net via (`drop_q`) into a kept via by re-anchoring incident branches.
    ConsolidateAdjacent {
        /// Numeric net ID.
        net_id: usize,
        /// Indices of [`TrackSegment`]s in `board.tracks` to remove.
        rem_track_indices: Vec<usize>,
        /// Replacement [`TrackSegment`]s connecting branches to the kept via.
        new_segs: Vec<TrackSegment>,
        /// Quantized position of the via to remove.
        drop_q: (i64, i64),
        /// Expanded spatial bounding box `(min_x, min_y, max_x, max_y)` in `mm`.
        aabb: (f64, f64, f64, f64),
    },
}

/// Evaluates all nets in parallel to find and apply spatially disjoint [`ViaPairMutation`]s
/// (either eliminating a `L_orig -> L_alt -> L_orig` hop or consolidating adjacent vias).
///
/// # Arguments
/// * `board` - Mutable [`Board`].
/// * `grid` - Reference to the [`RoutingGrid`].
/// * `config` - [`RoutingConfig`] parameters.
///
/// # Returns
/// `true` if at least one via mutation was committed to `board`.
pub(crate) fn eliminate_via_pair_once(board: &mut Board, grid: &RoutingGrid, config: &RoutingConfig) -> bool {
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        crate::board::is_gnd_net_name(net_name)
            || grid.plane_nets.values().any(|&pn| pn == nid)
    };

    // Collect all candidate nets with unlocked vias
    let mut net_ids: Vec<usize> = board
        .vias
        .iter()
        .filter(|v| !v.is_locked)
        .map(|v| v.net_id)
        .collect();
    net_ids.sort_unstable();
    net_ids.dedup();

    if net_ids.is_empty() {
        return false;
    }

    let drc_ctx = GpuDrcContext::from_board(board);
    let num_layers = grid.num_layers.max(2);
    let touches_same_net_pad = |net_id: usize, pt: (f64, f64), layer_opt: Option<usize>, tol: f64| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                if p.net_id != net_id {
                    return false;
                }
                if let Some(l) = layer_opt {
                    if !p.is_through_hole && fp.layer != l {
                        return false;
                    }
                }
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) <= tol
            })
        })
    };

    let candidate_mutations: Vec<ViaPairMutation> = net_ids
        .par_iter()
        .filter_map(|&net_id| {
            let vias_info: Vec<(usize, (f64, f64))> = board
                .vias
                .iter()
                .enumerate()
                .filter(|(_, v)| v.net_id == net_id && !v.is_locked)
                .map(|(i, v)| (i, v.position))
                .collect();

            if vias_info.len() < 2 {
                return None;
            }

            // 1. Via pair layer-hopping elimination (L_orig -> L_alt -> L_orig) for non-plane nets
            if !is_plane_net(net_id) {
                for &(v1_orig_idx, v1_pos) in &vias_info {
                    if board.arcs.iter().any(|a| {
                        a.net_id == net_id
                            && ((a.start().0 - v1_pos.0).hypot(a.start().1 - v1_pos.1) <= 0.08
                                || (a.end().0 - v1_pos.0).hypot(a.end().1 - v1_pos.1) <= 0.08)
                    }) {
                        continue;
                    }
                    let q1 = quant_point(v1_pos);

                    let incident_v1: Vec<(usize, usize, (f64, f64))> = board
                        .tracks
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| t.net_id == net_id && !t.is_locked)
                        .filter_map(|(idx, t)| {
                            if quant_point(t.start) == q1 {
                                Some((idx, t.layer, t.end))
                            } else if quant_point(t.end) == q1 {
                                Some((idx, t.layer, t.start))
                            } else {
                                None
                            }
                        })
                        .collect();

                    for &(t1_idx, l_alt, next_pt) in &incident_v1 {
                        let l_orig_candidates: Vec<usize> = incident_v1
                            .iter()
                            .filter(|&&(idx, l, _)| idx != t1_idx && l != l_alt)
                            .map(|&(_, l, _)| l)
                            .collect();

                        if l_orig_candidates.is_empty() {
                            continue;
                        }
                        let l_orig = l_orig_candidates[0];

                        let mut chain_segs = vec![t1_idx];
                        let mut chain_pts = vec![v1_pos, next_pt];
                        let mut curr_pt = next_pt;
                        let mut found_v2 = None;

                        while chain_segs.len() < 32 {
                            let curr_q = quant_point(curr_pt);

                            if let Some(&(v2_idx, v2_pos)) = vias_info
                                .iter()
                                .find(|&&(idx, pos)| idx != v1_orig_idx && quant_point(pos) == curr_q)
                            {
                                found_v2 = Some((v2_idx, v2_pos));
                                break;
                            }

                            let next_tracks: Vec<(usize, (f64, f64))> = board
                                .tracks
                                .iter()
                                .enumerate()
                                .filter(|&(idx, t)| {
                                    t.net_id == net_id
                                        && !t.is_locked
                                        && t.layer == l_alt
                                        && !chain_segs.contains(&idx)
                                })
                                .filter_map(|(idx, t)| {
                                    if quant_point(t.start) == curr_q {
                                        Some((idx, t.end))
                                    } else if quant_point(t.end) == curr_q {
                                        Some((idx, t.start))
                                    } else {
                                        None
                                    }
                                })
                                .collect();

                            let has_other_layer_branch = board.tracks.iter().any(|t| {
                                t.net_id == net_id
                                    && t.layer != l_alt
                                    && (quant_point(t.start) == curr_q || quant_point(t.end) == curr_q)
                            }) || board.arcs.iter().any(|a| {
                                a.net_id == net_id
                                    && ((a.start().0 - curr_pt.0).hypot(a.start().1 - curr_pt.1) <= 0.08
                                        || (a.end().0 - curr_pt.0).hypot(a.end().1 - curr_pt.1) <= 0.08)
                            });

                            if next_tracks.len() == 1 && !has_other_layer_branch {
                                let (next_seg_idx, nxt_p) = next_tracks[0];
                                chain_segs.push(next_seg_idx);
                                chain_pts.push(nxt_p);
                                curr_pt = nxt_p;
                            } else {
                                break;
                            }
                        }

                        let Some((_v2_orig_idx, v2_pos)) = found_v2 else {
                            continue;
                        };
                        if board.arcs.iter().any(|a| {
                            a.net_id == net_id
                                && ((a.start().0 - v2_pos.0).hypot(a.start().1 - v2_pos.1) <= 0.08
                                    || (a.end().0 - v2_pos.0).hypot(a.end().1 - v2_pos.1) <= 0.08)
                        }) {
                            continue;
                        }

                        let q2 = quant_point(v2_pos);

                        let v2_has_l_orig = board.tracks.iter().enumerate().any(|(idx, t)| {
                            t.net_id == net_id
                                && !t.is_locked
                                && t.layer == l_orig
                                && !chain_segs.contains(&idx)
                                && (quant_point(t.start) == q2 || quant_point(t.end) == q2)
                        });

                        if !v2_has_l_orig {
                            continue;
                        }

                        // Neither v1 nor v2 may have extra branches on layers != l_orig outside chain_segs
                        let has_extra_non_orig_branch = board.tracks.iter().enumerate().any(|(idx, t)| {
                            t.net_id == net_id
                                && t.layer != l_orig
                                && !chain_segs.contains(&idx)
                                && (quant_point(t.start) == q1
                                    || quant_point(t.end) == q1
                                    || quant_point(t.start) == q2
                                    || quant_point(t.end) == q2)
                        });
                        if has_extra_non_orig_branch {
                            continue;
                        }

                        let smd_hop_conflict = board.footprints.iter().any(|fp| {
                            fp.pads.iter().any(|p| {
                                if p.net_id != net_id || p.is_through_hole {
                                    return false;
                                }
                                let (px, py) = fp.pad_absolute_position(p);
                                chain_pts.iter().any(|pt| (px - pt.0).hypot(py - pt.1) < 0.15) && fp.layer != l_orig
                            })
                        });
                        if smd_hop_conflict {
                            continue;
                        }

                        let track_width = board.tracks[t1_idx].width;
                        let orig_hop_len: f64 = chain_pts
                            .windows(2)
                            .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
                            .sum();

                        let clearance = board.get_netclass_for_net(net_id).clearance;

                        // Build degree-2 retreat/advance chains on l_orig from v1_pos and v2_pos (up to 5 steps)
                        let build_l_orig_chain = |start_pos: (f64, f64), other_via_q: (i64, i64)| -> Vec<((f64, f64), Vec<usize>, f64)> {
                            let mut res = vec![(start_pos, Vec::new(), 0.0)];
                            if touches_same_net_pad(net_id, start_pos, Some(l_orig), 0.12) {
                                return res;
                            }
                            let mut cur_p = start_pos;
                            let mut used_segs: Vec<usize> = Vec::new();
                            let mut acc_len = 0.0;
                            for _ in 0..5 {
                                if board.arcs.iter().any(|a| {
                                    a.net_id == net_id
                                        && ((a.start().0 - cur_p.0).hypot(a.start().1 - cur_p.1) <= 0.08
                                            || (a.end().0 - cur_p.0).hypot(a.end().1 - cur_p.1) <= 0.08)
                                }) {
                                    break;
                                }
                                let cur_q = quant_point(cur_p);
                                let inc: Vec<(usize, (f64, f64))> = board
                                    .tracks
                                    .iter()
                                    .enumerate()
                                    .filter(|&(idx, t)| {
                                        t.net_id == net_id
                                            && !t.is_locked
                                            && t.layer == l_orig
                                            && !chain_segs.contains(&idx)
                                            && !used_segs.contains(&idx)
                                    })
                                    .filter_map(|(idx, t)| {
                                        if quant_point(t.start) == cur_q {
                                            Some((idx, t.end))
                                        } else if quant_point(t.end) == cur_q {
                                            Some((idx, t.start))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();
                                if inc.len() != 1 {
                                    break;
                                }
                                let (seg_idx, nxt_p) = inc[0];
                                let nxt_q = quant_point(nxt_p);
                                if nxt_q == other_via_q {
                                    break;
                                }
                                acc_len += (nxt_p.0 - cur_p.0).hypot(nxt_p.1 - cur_p.1);
                                used_segs.push(seg_idx);
                                res.push((nxt_p, used_segs.clone(), acc_len));
                                if touches_same_net_pad(net_id, nxt_p, Some(l_orig), 0.12)
                                    || vias_info.iter().any(|&(_, vp)| quant_point(vp) == nxt_q)
                                {
                                    break;
                                }
                                cur_p = nxt_p;
                            }
                            res
                        };

                        let u_chain = build_l_orig_chain(v1_pos, q2);
                        let w_chain = build_l_orig_chain(v2_pos, q1);

                        let mut hop_cands: Vec<(Vec<(f64, f64)>, Vec<usize>)> = Vec::new();
                        for (u_pt, rem_u, len_u) in &u_chain {
                            for (w_pt, rem_w, len_w) in &w_chain {
                                if rem_u.iter().any(|idx| rem_w.contains(idx)) {
                                    continue;
                                }
                                let orig_len = orig_hop_len + len_u + len_w;
                                let u_pos = *u_pt;
                                let w_pos = *w_pt;
                                let mut extra = rem_u.clone();
                                extra.extend_from_slice(rem_w);

                                // Priority 1: Direct octilinear segment on l_orig
                                let direct_len = (w_pos.0 - u_pos.0).hypot(w_pos.1 - u_pos.1);
                                if is_octilinear_segment(u_pos, w_pos)
                                    && direct_len <= orig_len * config.via_hop_max_detour_ratio + 1e-4
                                    && (direct_len - orig_len) <= 5.0 + 1e-4
                                {
                                    hop_cands.push((vec![u_pos, w_pos], extra.clone()));
                                }

                                // Priority 2: Exact path projection on l_orig (when at v1_pos and v2_pos)
                                if rem_u.is_empty() && rem_w.is_empty() {
                                    hop_cands.push((chain_pts.clone(), Vec::new()));
                                }

                                // Priority 3: Canonical 45-degree detour on l_orig within length budget
                                let dx = w_pos.0 - u_pos.0;
                                let dy = w_pos.1 - u_pos.1;
                                let s = dx.abs().min(dy.abs());
                                if s > 1e-4 {
                                    let sx = s * dx.signum();
                                    let sy = s * dy.signum();
                                    let c_a = (u_pos.0 + sx, u_pos.1 + sy);
                                    let c_b = (w_pos.0 - sx, w_pos.1 - sy);

                                    for c in [c_a, c_b] {
                                        let d1 = (c.0 - u_pos.0).hypot(c.1 - u_pos.1);
                                        let d2 = (w_pos.0 - c.0).hypot(w_pos.1 - c.1);
                                        let cand_len = d1 + d2;
                                        if cand_len <= orig_len * config.via_hop_max_detour_ratio + 1e-4
                                            && (cand_len - orig_len) <= 5.0 + 1e-4
                                        {
                                            hop_cands.push((vec![u_pos, c, w_pos], extra.clone()));
                                        }
                                    }
                                }
                            }
                        }

                        let gpu_hop_flags: Option<Vec<bool>> = if grid.vulkan.is_some() && !hop_cands.is_empty() {
                            let mut gpu_cands = Vec::with_capacity(hop_cands.len() * 2);
                            let mut spans = Vec::with_capacity(hop_cands.len());
                            for (pts, _) in &hop_cands {
                                let start_idx = gpu_cands.len();
                                for w in pts.windows(2) {
                                    if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                        gpu_cands.push(GpuDrcCandidate {
                                            x1: w[0].0 as f32,
                                            y1: w[0].1 as f32,
                                            x2: w[1].0 as f32,
                                            y2: w[1].1 as f32,
                                            radius_or_half_width: (track_width * 0.5) as f32,
                                            clearance: clearance as f32,
                                            layer: l_orig as u32,
                                            net_id: net_id as u32,
                                            kind: 1,
                                            ignore_same_net_pads: 1,
                                            _pad0: 0,
                                            _pad1: 0,
                                        });
                                    }
                                }
                                spans.push((start_idx, gpu_cands.len()));
                            }
                            drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_cands).map(|res| {
                                spans
                                    .into_iter()
                                    .map(|(s, e)| res[s..e].iter().all(|&v| v != 0))
                                    .collect()
                            })
                        } else {
                            None
                        };

                        let mut winner: Option<(Vec<(f64, f64)>, Vec<usize>)> = None;
                        for (idx, (pts, extra)) in hop_cands.into_iter().enumerate() {
                            if !gpu_hop_flags.as_ref().map_or(true, |f| f[idx]) {
                                continue;
                            }
                            let all_clear = pts.windows(2).all(|w| {
                                (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                    || (is_segment_clear(board, Some(grid), w[0], w[1], l_orig, net_id, track_width)
                                        && is_track_segment_legal(board, w[0], w[1], track_width, l_orig, net_id, clearance))
                            });
                            if all_clear {
                                winner = Some((pts, extra));
                                break;
                            }
                        }

                        if let Some((new_pts, extra_rem)) = winner {
                            let mut all_pts = chain_pts;
                            all_pts.extend_from_slice(&new_pts);
                            let mut full_chain_segs = chain_segs;
                            full_chain_segs.extend(extra_rem);
                            full_chain_segs.sort_unstable();
                            full_chain_segs.dedup();
                            let aabb = points_expanded_aabb(&all_pts, 0.60);
                            return Some(ViaPairMutation::EliminateHop {
                                net_id,
                                l_orig,
                                track_width,
                                chain_segs: full_chain_segs,
                                new_pts,
                                q1,
                                q2,
                                aabb,
                            });
                        }
                    }
                }
            }

            // 2. Physical & Topological Same-Net Via Consolidation (merges close vias within via_merge_radius_mm)
            let mut close_pairs: Vec<(f64, usize, usize)> = Vec::new();
            for i in 0..vias_info.len() {
                for j in (i + 1)..vias_info.len() {
                    let d = (vias_info[i].1 .0 - vias_info[j].1 .0).hypot(vias_info[i].1 .1 - vias_info[j].1 .1);
                    if d >= 1e-4 && d <= config.via_merge_radius_mm {
                        close_pairs.push((d, i, j));
                    }
                }
            }
            close_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

            // Helper: 3D BFS reachability from keep_pos across net_id's tracks & vias (excluding drop_idx and excluded_tracks)
            let is_reachable_from_keep = |keep_pos: (f64, f64),
                                          drop_idx: usize,
                                          target_pt: (f64, f64),
                                          target_layer: usize,
                                          excluded_tracks: &[usize],
                                          extra_tracks: &[TrackSegment]|
             -> bool {
                if (target_pt.0 - keep_pos.0).hypot(target_pt.1 - keep_pos.1) <= 0.08 {
                    return true;
                }
                let mut active_segs: Vec<((f64, f64), (f64, f64), usize)> = board
                    .tracks
                    .iter()
                    .enumerate()
                    .filter(|&(idx, t)| t.net_id == net_id && !excluded_tracks.contains(&idx))
                    .map(|(_, t)| (t.start, t.end, t.layer))
                    .collect();
                for et in extra_tracks {
                    active_segs.push((et.start, et.end, et.layer));
                }
                for a in board.arcs.iter().filter(|a| a.net_id == net_id) {
                    active_segs.push((a.start(), a.end(), a.layer));
                }
                let pth_pads: Vec<(f64, f64)> = board
                    .footprints
                    .iter()
                    .flat_map(|fp| {
                        fp.pads.iter().filter_map(move |pad| {
                            if pad.net_id == net_id && pad.is_through_hole {
                                Some(fp.pad_absolute_position(pad))
                            } else {
                                None
                            }
                        })
                    })
                    .collect();

                let mut visited_segs = vec![false; active_segs.len()];
                let mut visited_vias = vec![false; vias_info.len()];
                let mut visited_pth = vec![false; pth_pads.len()];
                let mut queue: std::collections::VecDeque<((f64, f64), usize)> = std::collections::VecDeque::new();
                for l in 0..num_layers {
                    queue.push_back((keep_pos, l));
                }

                while let Some((cur_pt, cur_l)) = queue.pop_front() {
                    if cur_l == target_layer && (cur_pt.0 - target_pt.0).hypot(cur_pt.1 - target_pt.1) <= 0.08 {
                        return true;
                    }
                    // Traverse tracks on cur_l
                    for (s_idx, &(s, e, l)) in active_segs.iter().enumerate() {
                        if visited_segs[s_idx] || l != cur_l {
                            continue;
                        }
                        if (s.0 - cur_pt.0).hypot(s.1 - cur_pt.1) <= 0.08 {
                            visited_segs[s_idx] = true;
                            queue.push_back((e, l));
                        } else if (e.0 - cur_pt.0).hypot(e.1 - cur_pt.1) <= 0.08 {
                            visited_segs[s_idx] = true;
                            queue.push_back((s, l));
                        }
                    }
                    // Traverse vias (excluding drop_idx)
                    for (vi, &(v_board_idx, vp)) in vias_info.iter().enumerate() {
                        if v_board_idx == drop_idx || visited_vias[vi] {
                            continue;
                        }
                        if (vp.0 - cur_pt.0).hypot(vp.1 - cur_pt.1) <= 0.08 {
                            visited_vias[vi] = true;
                            for l in 0..num_layers {
                                queue.push_back((vp, l));
                            }
                        }
                    }
                    // Traverse through-hole pads of net_id (expand each PTH pad at most once!)
                    for (pi, &pp) in pth_pads.iter().enumerate() {
                        if visited_pth[pi] {
                            continue;
                        }
                        if (pp.0 - cur_pt.0).hypot(pp.1 - cur_pt.1) <= 0.08 {
                            visited_pth[pi] = true;
                            for l in 0..num_layers {
                                if l != cur_l {
                                    queue.push_back((pp, l));
                                }
                            }
                        }
                    }
                }
                false
            };

            for &(_d_vias, i, j) in &close_pairs {
                let (v1_idx, v1_pos) = vias_info[i];
                let (v2_idx, v2_pos) = vias_info[j];
                for &(keep_idx, keep_pos, drop_idx, drop_pos) in &[
                    (v1_idx, v1_pos, v2_idx, v2_pos),
                    (v2_idx, v2_pos, v1_idx, v1_pos),
                ] {
                    let _ = keep_idx;
                    // Never drop a thermal via inside a large exposed thermal pad
                    let in_thermal_pad = board.footprints.iter().any(|fp| {
                        fp.pads.iter().any(|p| {
                            (p.width >= 2.0 || p.height >= 2.0) && {
                                let (px, py) = fp.pad_absolute_position(p);
                                (px - drop_pos.0).hypot(py - drop_pos.1) < 1.2
                            }
                        })
                    });
                    if in_thermal_pad {
                        continue;
                    }

                    let drop_r = board.vias.get(drop_idx).map_or(0.30, |v| v.size * 0.5) + 0.05;
                    let keep_r = board.vias.get(keep_idx).map_or(0.30, |v| v.size * 0.5) + 0.05;
                    if board.arcs.iter().any(|a| {
                        a.net_id == net_id
                            && ((a.start().0 - drop_pos.0).hypot(a.start().1 - drop_pos.1) <= drop_r
                                || (a.end().0 - drop_pos.0).hypot(a.end().1 - drop_pos.1) <= drop_r)
                    }) {
                        continue;
                    }
                    let q_drop = quant_point(drop_pos);
                    let q_keep = quant_point(keep_pos);

                    let drop_incident: Vec<(usize, usize, (f64, f64), f64)> = board
                        .tracks
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| t.net_id == net_id && !t.is_locked)
                        .filter_map(|(idx, t)| {
                            if quant_point(t.start) == q_drop || (t.start.0 - drop_pos.0).hypot(t.start.1 - drop_pos.1) <= drop_r {
                                Some((idx, t.layer, t.end, t.width))
                            } else if quant_point(t.end) == q_drop || (t.end.0 - drop_pos.0).hypot(t.end.1 - drop_pos.1) <= drop_r {
                                Some((idx, t.layer, t.start, t.width))
                            } else {
                                None
                            }
                        })
                        .collect();
                    if drop_incident.is_empty() {
                        continue;
                    }

                    // Trace maximal degree-2 chains starting from each incident segment of drop_pos
                    /// Maximal degree-2 trace chain incident to a candidate via being consolidated.
                    struct IncidentChain {
                        /// Indices of constituent [`TrackSegment`]s in `board.tracks`.
                        segs: Vec<usize>,
                        /// Ordered chain vertices `(x, y)` in `mm` starting at `drop_pos`.
                        pts: Vec<(f64, f64)>,
                        /// Copper layer of the chain.
                        layer: usize,
                        /// Trace width in `mm`.
                        width: f64,
                    }

                    let mut chains: Vec<IncidentChain> = Vec::with_capacity(drop_incident.len());
                    for &(t0_idx, layer, p1, width) in &drop_incident {
                        let mut segs = vec![t0_idx];
                        let mut pts = vec![drop_pos, p1];
                        let mut cur_p = p1;
                        while segs.len() < 6 {
                            let cur_q = quant_point(cur_p);
                            if cur_q == q_keep
                                || (cur_p.0 - keep_pos.0).hypot(cur_p.1 - keep_pos.1) <= keep_r
                                || touches_same_net_pad(net_id, cur_p, Some(layer), 0.08)
                                || vias_info.iter().any(|&(vi, vp)| vi != drop_idx && (quant_point(vp) == cur_q || (vp.0 - cur_p.0).hypot(vp.1 - cur_p.1) <= 0.08))
                                || board.arcs.iter().any(|a| {
                                    a.net_id == net_id
                                        && ((a.start().0 - cur_p.0).hypot(a.start().1 - cur_p.1) <= 0.08
                                            || (a.end().0 - cur_p.0).hypot(a.end().1 - cur_p.1) <= 0.08)
                                })
                            {
                                break;
                            }
                            let next_trks: Vec<(usize, (f64, f64))> = board
                                .tracks
                                .iter()
                                .enumerate()
                                .filter(|&(idx, t)| {
                                    t.net_id == net_id
                                        && !t.is_locked
                                        && t.layer == layer
                                        && !segs.contains(&idx)
                                })
                                .filter_map(|(idx, t)| {
                                    if quant_point(t.start) == cur_q {
                                        Some((idx, t.end))
                                    } else if quant_point(t.end) == cur_q {
                                        Some((idx, t.start))
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            if next_trks.len() != 1 {
                                break;
                            }
                            let (nxt_idx, nxt_p) = next_trks[0];
                            segs.push(nxt_idx);
                            pts.push(nxt_p);
                            cur_p = nxt_p;
                        }
                        chains.push(IncidentChain { segs, pts, layer, width });
                    }

                    let mut rem_track_indices: Vec<usize> = Vec::new();
                    let mut new_segs: Vec<TrackSegment> = Vec::new();
                    let mut all_valid = true;
                    let clearance = board.get_netclass_for_net(net_id).clearance.max(0.15);

                    for ch in &chains {
                        let term_pt = *ch.pts.last().unwrap();
                        let has_intermediate_pad = ch.pts[1..ch.pts.len() - 1]
                            .iter()
                            .any(|&pt| touches_same_net_pad(net_id, pt, Some(ch.layer), 0.08));

                        // Case A: Chain is a direct bridge to keep_pos or terminates at an anchor already connected to keep_pos
                        if !has_intermediate_pad {
                            let mut test_excl = rem_track_indices.clone();
                            test_excl.extend_from_slice(&ch.segs);
                            if is_reachable_from_keep(keep_pos, drop_idx, term_pt, ch.layer, &test_excl, &new_segs) {
                                rem_track_indices.extend_from_slice(&ch.segs);
                                continue;
                            }
                        }

                        // Case B: External branch — try connecting keep_pos to pts[r] along the chain
                        let mut branch_cands: Vec<(usize, Vec<(f64, f64)>)> = Vec::new();
                        for r in 1..ch.pts.len() {
                            let target_pt = ch.pts[r];
                            let orig_sub_len: f64 = ch.pts[0..=r]
                                .windows(2)
                                .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
                                .sum();

                            let touches_dense_ic_pad = board.footprints.iter().any(|fp| {
                                fp.pads.len() >= 8
                                    && fp.pads.iter().any(|p| {
                                        let (px, py) = fp.pad_absolute_position(p);
                                        (px - target_pt.0).hypot(py - target_pt.1) < 0.05
                                    })
                            });

                            let mut cand_paths: Vec<Vec<(f64, f64)>> = Vec::with_capacity(3);
                            if is_octilinear_segment(keep_pos, target_pt) {
                                cand_paths.push(vec![keep_pos, target_pt]);
                            }
                            cand_paths.push(compute_45_dogleg(keep_pos, target_pt, false));
                            cand_paths.push(compute_45_dogleg(keep_pos, target_pt, true));

                            for dl in cand_paths {
                                let cand_len: f64 = dl.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).sum();
                                if cand_len > orig_sub_len + config.via_detour_budget_mm {
                                    continue;
                                }
                                if touches_dense_ic_pad && dl.len() >= 2 {
                                    let prev = dl[dl.len() - 2];
                                    if (prev.0 - target_pt.0).abs() > 1e-3 && (prev.1 - target_pt.1).abs() > 1e-3 {
                                        continue;
                                    }
                                }
                                branch_cands.push((r, dl));
                            }

                            if touches_same_net_pad(net_id, target_pt, Some(ch.layer), 0.08) {
                                break;
                            }
                        }

                        let gpu_branch_flags: Option<Vec<bool>> = if grid.vulkan.is_some() && !branch_cands.is_empty() {
                            let mut gpu_cands = Vec::with_capacity(branch_cands.len() * 2);
                            let mut spans = Vec::with_capacity(branch_cands.len());
                            for (_, dl) in &branch_cands {
                                let start_idx = gpu_cands.len();
                                for w in dl.windows(2) {
                                    if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                        gpu_cands.push(GpuDrcCandidate {
                                            x1: w[0].0 as f32,
                                            y1: w[0].1 as f32,
                                            x2: w[1].0 as f32,
                                            y2: w[1].1 as f32,
                                            radius_or_half_width: (ch.width * 0.5) as f32,
                                            clearance: clearance as f32,
                                            layer: ch.layer as u32,
                                            net_id: net_id as u32,
                                            kind: 1,
                                            ignore_same_net_pads: 1,
                                            _pad0: 0,
                                            _pad1: 0,
                                        });
                                    }
                                }
                                spans.push((start_idx, gpu_cands.len()));
                            }
                            drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_cands).map(|res| {
                                spans
                                    .into_iter()
                                    .map(|(s, e)| res[s..e].iter().all(|&v| v != 0))
                                    .collect()
                            })
                        } else {
                            None
                        };

                        let mut found_branch_repl = false;
                        for (idx, (r, dl)) in branch_cands.into_iter().enumerate() {
                            if !gpu_branch_flags.as_ref().map_or(true, |f| f[idx]) {
                                continue;
                            }
                            let cpu_ok = dl.windows(2).all(|w| {
                                (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                                    || (is_segment_clear(board, Some(grid), w[0], w[1], ch.layer, net_id, ch.width)
                                        && is_track_segment_legal(board, w[0], w[1], ch.width, ch.layer, net_id, clearance))
                            });
                            if cpu_ok {
                                rem_track_indices.extend_from_slice(&ch.segs[0..r]);
                                for w in dl.windows(2) {
                                    if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                                        new_segs.push(TrackSegment {
                                            start: w[0],
                                            end: w[1],
                                            width: ch.width,
                                            layer: ch.layer,
                                            net_id,
                                            is_locked: false,
                                        });
                                    }
                                }
                                found_branch_repl = true;
                                break;
                            }
                        }

                        if !found_branch_repl {
                            all_valid = false;
                            break;
                        }
                    }

                    if all_valid {
                        rem_track_indices.sort_unstable();
                        rem_track_indices.dedup();

                        // Final 3D Continuity Guard: verify all chain endpoints and any pad at drop_pos remain reachable from keep_pos
                        let chains_continuous = chains.iter().all(|ch| {
                            let term_pt = *ch.pts.last().unwrap();
                            is_reachable_from_keep(keep_pos, drop_idx, term_pt, ch.layer, &rem_track_indices, &new_segs)
                        });
                        let drop_pads_continuous = board.footprints.iter().all(|fp| {
                            fp.pads.iter().all(|p| {
                                if p.net_id != net_id {
                                    return true;
                                }
                                let pp = fp.pad_absolute_position(p);
                                if (pp.0 - drop_pos.0).hypot(pp.1 - drop_pos.1) <= drop_r + 0.15 {
                                    is_reachable_from_keep(keep_pos, drop_idx, pp, fp.layer, &rem_track_indices, &new_segs)
                                } else {
                                    true
                                }
                            })
                        });
                        let all_incident_reaches = drop_incident.iter().all(|&(_, l, other_pt, _)| {
                            is_reachable_from_keep(keep_pos, drop_idx, other_pt, l, &rem_track_indices, &new_segs)
                        });

                        if chains_continuous && drop_pads_continuous && all_incident_reaches {
                            let mut bbox_pts = vec![keep_pos, drop_pos];
                            for ch in &chains {
                                bbox_pts.extend_from_slice(&ch.pts);
                            }
                            for s in &new_segs {
                                bbox_pts.push(s.start);
                                bbox_pts.push(s.end);
                            }
                            let aabb = points_expanded_aabb(&bbox_pts, 0.60);
                            return Some(ViaPairMutation::ConsolidateAdjacent {
                                net_id,
                                rem_track_indices,
                                new_segs,
                                drop_q: q_drop,
                                aabb,
                            });
                        }
                    }
                }
            }

            None
        })
        .collect();

    if candidate_mutations.is_empty() {
        return false;
    }

    // Select spatially disjoint mutations across nets so independent regions update in a single pass
    let mut accepted_aabbs: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut rem_tracks_all: Vec<usize> = Vec::new();
    let mut add_tracks_all: Vec<TrackSegment> = Vec::new();
    let mut hop_via_checks: Vec<(usize, usize, (i64, i64), (i64, i64))> = Vec::new();
    let mut drop_vias_exact: Vec<(usize, (i64, i64))> = Vec::new();

    for m in candidate_mutations {
        let aabb = match &m {
            ViaPairMutation::EliminateHop { aabb, .. } => *aabb,
            ViaPairMutation::ConsolidateAdjacent { aabb, .. } => *aabb,
        };
        if accepted_aabbs.iter().any(|&other| aabb_overlaps(aabb, other)) {
            continue;
        }
        accepted_aabbs.push(aabb);
        match m {
            ViaPairMutation::EliminateHop {
                net_id,
                l_orig,
                track_width,
                chain_segs,
                new_pts,
                q1,
                q2,
                ..
            } => {
                rem_tracks_all.extend(chain_segs);
                for w in new_pts.windows(2) {
                    if (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1) > 1e-5 {
                        add_tracks_all.push(TrackSegment {
                            start: w[0],
                            end: w[1],
                            width: track_width,
                            layer: l_orig,
                            net_id,
                            is_locked: false,
                        });
                    }
                }
                hop_via_checks.push((net_id, l_orig, q1, q2));
            }
            ViaPairMutation::ConsolidateAdjacent {
                net_id,
                rem_track_indices,
                new_segs,
                drop_q,
                ..
            } => {
                rem_tracks_all.extend(rem_track_indices);
                add_tracks_all.extend(new_segs);
                drop_vias_exact.push((net_id, drop_q));
            }
        }
    }

    rem_tracks_all.sort_unstable();
    rem_tracks_all.dedup();
    for &idx in rem_tracks_all.iter().rev() {
        board.tracks.remove(idx);
    }
    board.tracks.extend(add_tracks_all);

    let mut vias_to_remove = Vec::new();
    for (i, v) in board.vias.iter().enumerate() {
        if v.is_locked {
            continue;
        }
        let q = quant_point(v.position);
        if drop_vias_exact.iter().any(|&(nid, dq)| nid == v.net_id && dq == q) {
            vias_to_remove.push(i);
            continue;
        }
        for &(nid, l_orig, q1, q2) in &hop_via_checks {
            if v.net_id == nid && (q == q1 || q == q2) {
                let has_other_layer_tracks = board.tracks.iter().any(|t| {
                    t.net_id == nid
                        && t.layer != l_orig
                        && (quant_point(t.start) == q || quant_point(t.end) == q)
                });
                if !has_other_layer_tracks {
                    vias_to_remove.push(i);
                    break;
                }
            }
        }
    }
    vias_to_remove.sort_unstable();
    vias_to_remove.dedup();
    for &idx in vias_to_remove.iter().rev() {
        board.vias.remove(idx);
    }

    true
}

/// Automated BGA Escape Solver.
///
/// Analyzes regular 2D grid array BGA footprints, assigns pads to 4 quadrants (NE, NW, SE, SW),
/// and generates deterministic fanouts:
/// - **Ring 0 (outer perimeter):** direct surface escape tracks shooting outward without vias.
/// - **Ring 1 & inner rings:** diagonal 45° dog-bone escape stubs placed at interstitial cell centers
///   connecting pad to dog-bone via and routing out through interstitial avenue corridors.
/// - **Via-in-Pad (VIPPO) fallback:** centered microvia (`0.30 mm` dia, `0.15 mm` drill) for pitch `< 0.50 mm`
///   or where standard annular ring cannot clear adjacent pads.
///
/// # Arguments
/// * `board` - Mutable board state where generated escape tracks and vias are appended.
/// * `grid` - Mutable routing grid where generated escape tracks and vias are committed and distance fields refreshed.
///
/// # Side Effects
/// Mutates `board.tracks`, `board.vias`, and `grid` (commits tracks/vias and recomputes distance fields).
pub fn generate_bga_dogbone_escapes(board: &mut Board, grid: &mut RoutingGrid) {
    let num_layers = board.copper_layers().len().max(2);

    for fp_idx in 0..board.footprints.len() {
        let fp = board.footprints[fp_idx].clone();
        let Some(bga_info) = fp.bga_grid_info() else {
            continue;
        };

        let pitch = bga_info.pitch_x.min(bga_info.pitch_y);
        let (cx, cy) = (bga_info.center_x, bga_info.center_y);

        // Distance from pad center to adjacent 4-pad interstitial cell center:
        let inter_dist = (bga_info.pitch_x * 0.5).hypot(bga_info.pitch_y * 0.5);

        // Standard dog-bone via parameters
        let std_via_dia = 0.45;
        let std_via_drill = 0.20;
        let std_via_r = std_via_dia * 0.5;

        // Microvia / VIPPO parameters
        let vippo_via_dia = 0.30;
        let vippo_via_drill = 0.15;

        // Check if VIPPO fallback is required:
        // Fallback condition: pitch < 0.50mm, OR standard dog-bone annular ring (0.45mm)
        // cannot clear adjacent pads with netclass clearance.
        let default_c = 0.12;
        let max_pad_r = fp
            .pads
            .iter()
            .filter(|p| !p.is_through_hole)
            .map(|p| p.width.max(p.height) * 0.5)
            .fold(0.20f64, f64::max);
        let vippo_required = pitch < 0.50 || (inter_dist < std_via_r + max_pad_r + default_c);

        let oy = |p: &Pad| if fp.layer != fp.orig_layer { -p.offset_y } else { p.offset_y };

        for (pad_idx, pad) in fp.pads.iter().enumerate() {
            if pad.is_through_hole {
                continue;
            }
            if pad.net_name.starts_with("unconnected") {
                continue;
            }

            let pad_net_id = if pad.net_id > 0 {
                pad.net_id
            } else if !pad.net_name.is_empty() {
                board.net_name_to_id.get(&pad.net_name).copied().unwrap_or(0)
            } else {
                0
            };

            let (px_world, py_world) = fp.pad_absolute_position(pad);

            // Avoid duplicate escapes if already present
            let already_routed = board.tracks.iter().any(|t| {
                (t.start.0 - px_world).hypot(t.start.1 - py_world) < 1e-4
                    || (t.end.0 - px_world).hypot(t.end.1 - py_world) < 1e-4
            }) || board.vias.iter().any(|v| {
                (v.position.0 - px_world).hypot(v.position.1 - py_world) < 1e-4
            });
            if already_routed {
                continue;
            }

            let lx = pad.offset_x;
            let ly = oy(pad);
            let dx = lx - cx;
            let dy = ly - cy;

            // Quadrant signs (+/- p/2 away from center)
            let sign_x = if dx >= 0.0 { 1.0 } else { -1.0 };
            let sign_y = if dy >= 0.0 { 1.0 } else { -1.0 };

            let ring = bga_info.pad_rings[pad_idx];
            let netclass = board.get_netclass_for_net(pad_net_id);
            let trk_w = (netclass.track_width).min(0.20);
            let stub_w = (netclass.track_width).min(0.15);

            if ring == 0 {
                // Ring 0 (outer perimeter): direct surface escape tracks shooting outward away from chip without vias
                let (col, row) = bga_info.pad_grid_indices[pad_idx].unwrap_or((0, 0));
                let outward_local = if col == 0 && row == 0 {
                    (-1.0 / 2.0f64.sqrt(), -1.0 / 2.0f64.sqrt())
                } else if col == bga_info.cols - 1 && row == 0 {
                    (1.0 / 2.0f64.sqrt(), -1.0 / 2.0f64.sqrt())
                } else if col == 0 && row == bga_info.rows - 1 {
                    (-1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt())
                } else if col == bga_info.cols - 1 && row == bga_info.rows - 1 {
                    (1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt())
                } else if col == 0 {
                    (-1.0, 0.0)
                } else if col == bga_info.cols - 1 {
                    (1.0, 0.0)
                } else if row == 0 {
                    (0.0, -1.0)
                } else if row == bga_info.rows - 1 {
                    (0.0, 1.0)
                } else {
                    let len = dx.hypot(dy).max(1e-6);
                    (dx / len, dy / len)
                };

                let world_dir = Point::new(outward_local.0, outward_local.1).rotate_ccw(fp.rotation);
                let escape_len = (pitch * 1.5).max(1.0);
                let tip = (px_world + world_dir.x * escape_len, py_world + world_dir.y * escape_len);

                let trk = TrackSegment {
                    start: (px_world, py_world),
                    end: tip,
                    width: trk_w,
                    layer: fp.layer,
                    net_id: pad_net_id,
                    is_locked: false,
                };
                board.tracks.push(trk.clone());
                grid.commit_track(&trk);
            } else if vippo_required {
                // Ring 1 & inner rings with VIPPO fallback:
                // Centered via-in-pad microvia (0.30mm diameter, 0.15mm drill)
                let via_pos = (px_world, py_world);
                let target_layer = if num_layers <= 2 {
                    1
                } else if ring == 1 {
                    1
                } else {
                    2 + (ring - 2) % (num_layers - 2)
                };

                let vippo_via = Via {
                    position: via_pos,
                    size: vippo_via_dia,
                    drill: vippo_via_drill,
                    net_id: pad_net_id,
                    is_locked: false,
                };
                board.vias.push(vippo_via.clone());
                grid.commit_via(&vippo_via);

                // Escape track radiating out from microvia on target layer
                let outward_local = (sign_x / 2.0f64.sqrt(), sign_y / 2.0f64.sqrt());
                let world_dir = Point::new(outward_local.0, outward_local.1).rotate_ccw(fp.rotation);
                let escape_len = (pitch * 2.0).max(1.0);
                let tip = (px_world + world_dir.x * escape_len, py_world + world_dir.y * escape_len);

                let chan_trk = TrackSegment {
                    start: (px_world, py_world),
                    end: tip,
                    width: stub_w,
                    layer: target_layer,
                    net_id: pad_net_id,
                    is_locked: false,
                };
                board.tracks.push(chan_trk.clone());
                grid.commit_track(&chan_trk);
            } else {
                // Ring 1 & inner rings: standard 45° dog-bone escape stubs
                // Dog-bone via placed at diagonal center of adjacent 4-pad interstitial cell (dx = +/- p/2, dy = +/- p/2)
                let delta_x = sign_x * (bga_info.pitch_x * 0.5);
                let delta_y = sign_y * (bga_info.pitch_y * 0.5);
                let via_loc_x = lx + delta_x;
                let via_loc_y = ly + delta_y;

                let world_via_pt = Point::new(delta_x, delta_y).rotate_ccw(fp.rotation);
                let world_via = (px_world + world_via_pt.x, py_world + world_via_pt.y);

                // Stub track connecting pad center to dog-bone via
                let stub_trk = TrackSegment {
                    start: (px_world, py_world),
                    end: world_via,
                    width: stub_w,
                    layer: fp.layer,
                    net_id: pad_net_id,
                    is_locked: false,
                };

                let target_layer = if num_layers <= 2 {
                    1
                } else if ring == 1 {
                    1
                } else {
                    2 + (ring - 2) % (num_layers - 2)
                };

                let dogbone_via = Via {
                    position: world_via,
                    size: std_via_dia,
                    drill: std_via_drill,
                    net_id: pad_net_id,
                    is_locked: false,
                };

                // Channel escape corridor: radiating out through interstitial avenue
                let (target_local_x, target_local_y) = if dx.abs() >= dy.abs() {
                    let end_x = if sign_x > 0.0 {
                        bga_info.max_x + pitch * 1.5
                    } else {
                        bga_info.min_x - pitch * 1.5
                    };
                    (end_x, via_loc_y)
                } else {
                    let end_y = if sign_y > 0.0 {
                        bga_info.max_y + pitch * 1.5
                    } else {
                        bga_info.min_y - pitch * 1.5
                    };
                    (via_loc_x, end_y)
                };

                let corridor_disp = Point::new(target_local_x - lx, target_local_y - ly).rotate_ccw(fp.rotation);
                let world_corridor_end = (px_world + corridor_disp.x, py_world + corridor_disp.y);

                let chan_trk = TrackSegment {
                    start: world_via,
                    end: world_corridor_end,
                    width: stub_w,
                    layer: target_layer,
                    net_id: pad_net_id,
                    is_locked: false,
                };

                board.tracks.push(stub_trk.clone());
                grid.commit_track(&stub_trk);

                board.vias.push(dogbone_via.clone());
                grid.commit_via(&dogbone_via);

                board.tracks.push(chan_trk.clone());
                grid.commit_track(&chan_trk);
            }
        }
    }

    grid.update_distance_fields();
}

/// Identifies connections with a single via transitioning between layers and consolidates them onto a single layer.
///
/// Tests whether the alternate-layer segment chain can be moved or projected onto
/// the primary layer without DRC violations, eliminating the solitary via.
///
/// # Arguments
/// * `board` - Mutable board state whose tracks and vias are updated.
/// * `grid` - Routing grid used for clearance and optional GPU batch screening.
///
/// # Returns
/// `true` if at least one solitary via was eliminated.
///
/// # Side Effects
/// Mutates `board.tracks` and `board.vias`.
pub fn consolidate_single_vias(board: &mut Board, grid: &RoutingGrid) -> bool {
    consolidate_single_vias_with_config(board, grid, &RoutingConfig::default())
}

/// Consolidates single-via layer transitions using an explicit [`RoutingConfig`].
///
/// Iteratively invokes `consolidate_single_via_once` up to 20 passes until convergence.
///
/// # Arguments
/// * `board` - Mutable board state whose tracks and vias are updated.
/// * `grid` - Routing grid used for clearance and optional GPU batch screening.
/// * `config` - Routing configuration controlling `via_hop_max_detour_ratio`.
///
/// # Returns
/// `true` if at least one solitary via was eliminated across all passes.
///
/// # Side Effects
/// Mutates `board.tracks` and `board.vias`.
pub fn consolidate_single_vias_with_config(
    board: &mut Board,
    grid: &RoutingGrid,
    config: &RoutingConfig,
) -> bool {
    let mut any_changed = false;
    for _ in 0..20 {
        if !consolidate_single_via_once(board, grid, config) {
            break;
        }
        any_changed = true;
    }
    any_changed
}

/// Validated candidate mutation for eliminating a single layer-transition via by re-projecting its incident stub chain.
pub(crate) struct SingleViaConsolidation {
    /// Net ID of the consolidated trace.
    net_id: usize,
    /// Target primary copper layer index onto which the chain is projected.
    l_prim: usize,
    /// Trace width in millimeters (`mm`).
    track_width: f64,
    /// Indices of existing [`TrackSegment`] items in `board.tracks` to remove.
    chain_segs: Vec<usize>,
    /// Replacement polyline vertices `(x, y)` in `mm` on `l_prim`.
    new_pts: Vec<(f64, f64)>,
    /// Quantized integer coordinates `(qx, qy)` of the solitary via to remove.
    q_v: (i64, i64),
    /// Expanded 2D AABB `(min_x, min_y, max_x, max_y)` in `mm` used for conflict-free batching.
    aabb: (f64, f64, f64, f64),
}

/// Executes a single parallelized pass of single-via consolidation across all non-overlapping nets.
///
/// # Arguments
/// * `board` - Mutable board state whose tracks and vias are updated.
/// * `grid` - Routing grid used for clearance and optional GPU batch screening.
/// * `config` - Routing configuration controlling detour length bounds.
///
/// # Returns
/// `true` if at least one single via was consolidated in this pass.
///
/// # Side Effects
/// Mutates `board.tracks` and `board.vias`, and may dispatch a GPU DRC batch query.
pub(crate) fn consolidate_single_via_once(
    board: &mut Board,
    grid: &RoutingGrid,
    config: &RoutingConfig,
) -> bool {
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        crate::board::is_gnd_net_name(net_name)
            || grid.plane_nets.values().any(|&pn| pn == nid)
    };

    let vias_info: Vec<(usize, (f64, f64), usize)> = board
        .vias
        .iter()
        .enumerate()
        .filter(|(_, v)| !v.is_locked && !is_plane_net(v.net_id))
        .map(|(i, v)| (i, v.position, v.net_id))
        .collect();

    if vias_info.is_empty() {
        return false;
    }

    let drc_ctx = GpuDrcContext::from_board(board);

    /// Candidate stub-projection option for eliminating a solitary via onto a primary layer.
    struct ViaStubOption {
        /// Target primary copper layer index.
        l_prim: usize,
        /// Trace width in millimeters (`mm`).
        track_width: f64,
        /// Required netclass copper clearance in millimeters (`mm`).
        clearance: f64,
        /// Indices of alternate-layer track segments in `board.tracks` to replace.
        chain_segs: Vec<usize>,
        /// Ordered vertex coordinates `(x, y)` in `mm` of the alternate-layer chain.
        chain_pts: Vec<(f64, f64)>,
        /// Candidate replacement polylines on `l_prim` (direct, projected, or 45° dogleg).
        cand_paths: Vec<Vec<(f64, f64)>>,
    }

    let via_options: Vec<(usize, (i64, i64), Vec<ViaStubOption>)> = vias_info
        .par_iter()
        .filter_map(|&(v_idx, v_pos, net_id)| {
            if board.arcs.iter().any(|a| {
                a.net_id == net_id
                    && ((a.start().0 - v_pos.0).hypot(a.start().1 - v_pos.1) <= 0.08
                        || (a.end().0 - v_pos.0).hypot(a.end().1 - v_pos.1) <= 0.08)
            }) {
                return None;
            }
            let q_v = quant_point(v_pos);
            let clearance = board.get_netclass_for_net(net_id).clearance;

            let incident: Vec<(usize, usize, (f64, f64), f64)> = board
                .tracks
                .iter()
                .enumerate()
                .filter(|(_, t)| t.net_id == net_id && !t.is_locked)
                .filter_map(|(idx, t)| {
                    if quant_point(t.start) == q_v {
                        Some((idx, t.layer, t.end, t.width))
                    } else if quant_point(t.end) == q_v {
                        Some((idx, t.layer, t.start, t.width))
                    } else {
                        None
                    }
                })
                .collect();

            let mut layers: Vec<usize> = incident.iter().map(|&(_, l, _, _)| l).collect();
            layers.sort_unstable();
            layers.dedup();
            if layers.len() < 2 {
                return None;
            }

            let mut opts = Vec::new();
            for &l_prim in &layers {
                let alt_layers: Vec<usize> = layers.iter().copied().filter(|&l| l != l_prim).collect();
                if alt_layers.len() != 1 {
                    continue;
                }
                for &l_alt in &alt_layers {
                    let alt_incidents: Vec<(usize, (f64, f64), f64)> = incident
                        .iter()
                        .filter(|&&(_, l, _, _)| l == l_alt)
                        .map(|&(idx, _, pt, w)| (idx, pt, w))
                        .collect();

                    if alt_incidents.len() != 1 {
                        continue;
                    }

                    let (t1_idx, next_pt, track_width) = alt_incidents[0];
                    let mut chain_segs = vec![t1_idx];
                    let mut chain_pts = vec![v_pos, next_pt];
                    let mut curr_pt = next_pt;
                    let mut hit_another_via = false;

                    while chain_segs.len() < 32 {
                        let curr_q = quant_point(curr_pt);
                        if board.vias.iter().enumerate().any(|(i, other_v)| {
                            i != v_idx && other_v.net_id == net_id && quant_point(other_v.position) == curr_q
                        }) || board.arcs.iter().any(|a| {
                            a.net_id == net_id
                                && ((a.start().0 - curr_pt.0).hypot(a.start().1 - curr_pt.1) <= 0.08
                                    || (a.end().0 - curr_pt.0).hypot(a.end().1 - curr_pt.1) <= 0.08)
                        }) {
                            hit_another_via = true;
                            break;
                        }

                        let next_tracks: Vec<(usize, (f64, f64))> = board
                            .tracks
                            .iter()
                            .enumerate()
                            .filter(|&(idx, t)| {
                                t.net_id == net_id
                                    && !t.is_locked
                                    && t.layer == l_alt
                                    && !chain_segs.contains(&idx)
                            })
                            .filter_map(|(idx, t)| {
                                if quant_point(t.start) == curr_q {
                                    Some((idx, t.end))
                                } else if quant_point(t.end) == curr_q {
                                    Some((idx, t.start))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if next_tracks.len() > 1 {
                            hit_another_via = true;
                            break;
                        } else if let Some(&(next_seg_idx, nxt_p)) = next_tracks.first() {
                            chain_segs.push(next_seg_idx);
                            chain_pts.push(nxt_p);
                            curr_pt = nxt_p;
                        } else {
                            break;
                        }
                    }

                    if hit_another_via {
                        continue;
                    }

                    let smd_layer_conflict = board.footprints.iter().any(|fp| {
                        fp.pads.iter().any(|p| {
                            if p.net_id != net_id || p.is_through_hole {
                                return false;
                            }
                            let (px, py) = fp.pad_absolute_position(p);
                            chain_pts.iter().any(|pt| (px - pt.0).hypot(py - pt.1) < 0.15) && fp.layer != l_prim
                        })
                    });
                    if smd_layer_conflict {
                        continue;
                    }

                    let pad_on_l_prim = board.footprints.iter().any(|fp| {
                        fp.pads.iter().any(|p| {
                            if p.net_id != net_id {
                                return false;
                            }
                            let (px, py) = fp.pad_absolute_position(p);
                            if (px - curr_pt.0).hypot(py - curr_pt.1) < 0.15 {
                                p.is_through_hole || fp.layer == l_prim
                            } else {
                                false
                            }
                        })
                    });

                    let curr_q = quant_point(curr_pt);
                    let connects_to_track_on_l_prim = board.tracks.iter().enumerate().any(|(idx, t)| {
                        t.net_id == net_id
                            && t.layer == l_prim
                            && !t.is_locked
                            && !chain_segs.contains(&idx)
                            && (quant_point(t.start) == curr_q || quant_point(t.end) == curr_q)
                    });

                    if !pad_on_l_prim && !connects_to_track_on_l_prim {
                        continue;
                    }

                    let orig_len: f64 = chain_pts
                        .windows(2)
                        .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
                        .sum();

                    let mut cand_paths: Vec<Vec<(f64, f64)>> = Vec::with_capacity(4);

                    // Priority 1: Direct segment on l_prim (strictly octilinear 0°, 45°, 90°)
                    let direct_len = (curr_pt.0 - v_pos.0).hypot(curr_pt.1 - v_pos.1);
                    if is_octilinear_segment(v_pos, curr_pt)
                        && direct_len <= orig_len * config.via_hop_max_detour_ratio + 1e-4
                        && (direct_len - orig_len) <= 5.0 + 1e-4
                    {
                        cand_paths.push(vec![v_pos, curr_pt]);
                    }

                    // Priority 2: Same path coordinates on l_prim
                    cand_paths.push(chain_pts.clone());

                    // Priority 3: Canonical 45-degree detour on l_prim
                    let dx = curr_pt.0 - v_pos.0;
                    let dy = curr_pt.1 - v_pos.1;
                    let s = dx.abs().min(dy.abs());
                    if s > 1e-4 {
                        let sx = s * dx.signum();
                        let sy = s * dy.signum();
                        let c_a = (v_pos.0 + sx, v_pos.1 + sy);
                        let c_b = (curr_pt.0 - sx, curr_pt.1 - sy);
                        for c in [c_a, c_b] {
                            let d1 = (c.0 - v_pos.0).hypot(c.1 - v_pos.1);
                            let d2 = (curr_pt.0 - c.0).hypot(curr_pt.1 - c.1);
                            let cand_len = d1 + d2;
                            if cand_len <= orig_len * config.via_hop_max_detour_ratio + 1e-4
                                && (cand_len - orig_len) <= 5.0 + 1e-4
                            {
                                cand_paths.push(vec![v_pos, c, curr_pt]);
                            }
                        }
                    }

                    if !cand_paths.is_empty() {
                        opts.push(ViaStubOption {
                            l_prim,
                            track_width,
                            clearance,
                            chain_segs,
                            chain_pts,
                            cand_paths,
                        });
                    }
                }
            }

            if opts.is_empty() {
                None
            } else {
                Some((net_id, q_v, opts))
            }
        })
        .collect();

    if via_options.is_empty() {
        return false;
    }

    let mut opt_spans: Vec<Vec<Vec<(usize, usize)>>> = Vec::with_capacity(via_options.len());
    let mut gpu_cands: Vec<GpuDrcCandidate> = Vec::new();
    if grid.vulkan.is_some() {
        for (net_id, _, opts) in &via_options {
            let mut via_spans = Vec::with_capacity(opts.len());
            for opt in opts {
                let mut path_spans = Vec::with_capacity(opt.cand_paths.len());
                for path in &opt.cand_paths {
                    let s_idx = gpu_cands.len();
                    for w in path.windows(2) {
                        if (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) > 1e-4 {
                            gpu_cands.push(GpuDrcCandidate {
                                x1: w[0].0 as f32,
                                y1: w[0].1 as f32,
                                x2: w[1].0 as f32,
                                y2: w[1].1 as f32,
                                radius_or_half_width: (opt.track_width * 0.5) as f32,
                                clearance: opt.clearance as f32,
                                layer: opt.l_prim as u32,
                                net_id: *net_id as u32,
                                kind: 1,
                                ignore_same_net_pads: 1,
                                _pad0: 0,
                                _pad1: 0,
                            });
                        }
                    }
                    path_spans.push((s_idx, gpu_cands.len()));
                }
                via_spans.push(path_spans);
            }
            opt_spans.push(via_spans);
        }
    }
    let gpu_res = if !gpu_cands.is_empty() {
        drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_cands)
    } else {
        None
    };

    let candidates: Vec<SingleViaConsolidation> = via_options
        .into_par_iter()
        .enumerate()
        .filter_map(|(v_i, (net_id, q_v, opts))| {
            for (o_i, opt) in opts.into_iter().enumerate() {
                for (p_i, path) in opt.cand_paths.into_iter().enumerate() {
                    if let Some(ref res) = gpu_res {
                        let (s, e) = opt_spans[v_i][o_i][p_i];
                        if !res[s..e].iter().all(|&v| v != 0) {
                            continue;
                        }
                    }
                    let all_clear = path.windows(2).all(|w| {
                        (w[0].0 - w[1].0).hypot(w[0].1 - w[1].1) <= 1e-4
                            || (is_segment_clear(board, Some(grid), w[0], w[1], opt.l_prim, net_id, opt.track_width)
                                && is_track_segment_legal(board, w[0], w[1], opt.track_width, opt.l_prim, net_id, opt.clearance))
                    });
                    if all_clear {
                        let mut all_pts = opt.chain_pts;
                        all_pts.extend_from_slice(&path);
                        let aabb = points_expanded_aabb(&all_pts, 0.60);
                        return Some(SingleViaConsolidation {
                            net_id,
                            l_prim: opt.l_prim,
                            track_width: opt.track_width,
                            chain_segs: opt.chain_segs,
                            new_pts: path,
                            q_v,
                            aabb,
                        });
                    }
                }
            }
            None
        })
        .collect();

    if candidates.is_empty() {
        return false;
    }

    let mut accepted_nets: HashSet<usize> = HashSet::new();
    let mut accepted_aabbs: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut rem_tracks_all: Vec<usize> = Vec::new();
    let mut add_tracks_all: Vec<TrackSegment> = Vec::new();
    let mut rem_vias_q: Vec<(usize, (i64, i64))> = Vec::new();

    for c in candidates {
        if accepted_nets.contains(&c.net_id)
            || accepted_aabbs.iter().any(|&other| aabb_overlaps(c.aabb, other))
        {
            continue;
        }
        accepted_nets.insert(c.net_id);
        accepted_aabbs.push(c.aabb);
        rem_tracks_all.extend(c.chain_segs);
        for w in c.new_pts.windows(2) {
            if (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1) > 1e-5 {
                add_tracks_all.push(TrackSegment {
                    start: w[0],
                    end: w[1],
                    width: c.track_width,
                    layer: c.l_prim,
                    net_id: c.net_id,
                    is_locked: false,
                });
            }
        }
        rem_vias_q.push((c.net_id, c.q_v));
    }

    rem_tracks_all.sort_unstable();
    rem_tracks_all.dedup();
    for &idx in rem_tracks_all.iter().rev() {
        board.tracks.remove(idx);
    }
    board.tracks.extend(add_tracks_all);

    for (nid, q_v) in rem_vias_q {
        if let Some(pos) = board.vias.iter().position(|v| {
            v.net_id == nid && !v.is_locked && quant_point(v.position) == q_v
        }) {
            board.vias.remove(pos);
        }
    }

    true
}

/// Slides and aligns signal vias along incident track vectors to eliminate micro-jogs,
/// doglegs, reduce wirelength, and snap close vias to through-hole pads.
///
/// # Arguments
/// * `board` - Mutable board state whose tracks and vias are updated.
/// * `grid` - Routing grid used for clearance and optional GPU batch screening.
///
/// # Returns
/// `true` if at least one via was moved or snapped into a through-hole pad.
///
/// # Side Effects
/// Mutates `board.vias` and `board.tracks`.
pub fn slide_and_align_vias(board: &mut Board, grid: &RoutingGrid) -> bool {
    slide_and_align_vias_with_config(board, grid, &RoutingConfig::default())
}

/// Slides and aligns signal vias along incident track vectors using an explicit [`RoutingConfig`].
///
/// Iteratively runs `slide_and_align_via_once` up to 20 passes until convergence.
///
/// # Arguments
/// * `board` - Mutable board state whose tracks and vias are updated.
/// * `grid` - Routing grid used for clearance and optional GPU batch screening.
/// * `config` - Routing configuration controlling `via_pad_snap_dist_mm`.
///
/// # Returns
/// `true` if at least one via was moved or snapped across all passes.
///
/// # Side Effects
/// Mutates `board.vias` and `board.tracks`.
pub fn slide_and_align_vias_with_config(
    board: &mut Board,
    grid: &RoutingGrid,
    config: &RoutingConfig,
) -> bool {
    let mut any_changed = false;
    for _ in 0..20 {
        if !slide_and_align_via_once(board, grid, config) {
            break;
        }
        any_changed = true;
    }
    any_changed
}

/// Validated candidate mutation for sliding a via or snapping it into a nearby through-hole pad.
pub(crate) struct ViaSlideAction {
    /// Index of the via in `board.vias`.
    v_idx: usize,
    /// Original via position `(x, y)` in millimeters (`mm`).
    v_pos: (f64, f64),
    /// Net ID of the via and incident tracks.
    net_id: usize,
    /// New target position `(x, y)` in millimeters (`mm`) for the via or incident track endpoints.
    target_pos: (f64, f64),
    /// Whether the via is absorbed into a through-hole pad and should be removed from `board.vias`.
    remove_via: bool,
    /// Indices of incident [`TrackSegment`] items in `board.tracks` whose endpoints attach to `v_pos`.
    incident_indices: Vec<usize>,
    /// Expanded 2D AABB `(min_x, min_y, max_x, max_y)` in `mm` used for conflict-free parallel batching.
    aabb: (f64, f64, f64, f64),
}

/// Executes a single parallelized pass of via pad-snapping, micro-jog collapsing, and vector sliding.
///
/// # Arguments
/// * `board` - Mutable board state whose tracks and vias are updated.
/// * `grid` - Routing grid used for clearance and optional GPU batch screening.
/// * `config` - Routing configuration controlling pad snap distance.
///
/// # Returns
/// `true` if at least one via was moved or absorbed into a pad in this pass.
///
/// # Side Effects
/// Mutates `board.vias` and `board.tracks`.
pub(crate) fn slide_and_align_via_once(
    board: &mut Board,
    grid: &RoutingGrid,
    config: &RoutingConfig,
) -> bool {
    let is_plane_net = |nid: usize| -> bool {
        let net_name = board.nets.get(nid).map(|s| s.as_str()).unwrap_or("");
        crate::board::is_gnd_net_name(net_name)
            || grid.plane_nets.values().any(|&pn| pn == nid)
    };

    let num_vias = board.vias.len();
    if num_vias == 0 {
        return false;
    }

    let drc_ctx = GpuDrcContext::from_board(board);

    let slide_candidates: Vec<ViaSlideAction> = (0..num_vias)
        .into_par_iter()
        .filter_map(|v_idx| {
            let (v_pos, net_id, via_size, is_locked) = {
                let v = &board.vias[v_idx];
                (v.position, v.net_id, v.size, v.is_locked)
            };
            if is_locked || is_plane_net(net_id) {
                return None;
            }

            let clearance = board.get_netclass_for_net(net_id).clearance;

            let incident: Vec<(usize, usize, (f64, f64), f64)> = board
                .tracks
                .iter()
                .enumerate()
                .filter(|(_, t)| t.net_id == net_id && !t.is_locked)
                .filter_map(|(idx, t)| {
                    if (t.start.0 - v_pos.0).hypot(t.start.1 - v_pos.1) < 1e-4 {
                        Some((idx, t.layer, t.end, t.width))
                    } else if (t.end.0 - v_pos.0).hypot(t.end.1 - v_pos.1) < 1e-4 {
                        Some((idx, t.layer, t.start, t.width))
                    } else {
                        None
                    }
                })
                .collect();

            let mut inc_layers: Vec<usize> = incident.iter().map(|&(_, l, _, _)| l).collect();
            inc_layers.sort_unstable();
            inc_layers.dedup();
            let has_unanchored_touch = board.tracks.iter().enumerate().any(|(idx, t)| {
                t.net_id == net_id
                    && !incident.iter().any(|&(i, _, _, _)| i == idx)
                    && dist_point_to_segment(v_pos, t.start, t.end) <= via_size * 0.5 + 0.05
            }) || board.arcs.iter().any(|a| {
                a.net_id == net_id
                    && ((a.start().0 - v_pos.0).hypot(a.start().1 - v_pos.1) <= via_size * 0.5 + 0.05
                        || (a.end().0 - v_pos.0).hypot(a.end().1 - v_pos.1) <= via_size * 0.5 + 0.05)
            });
            if incident.is_empty() || inc_layers.len() < 2 || has_unanchored_touch {
                return None;
            }

            let incident_indices: Vec<usize> = incident.iter().map(|&(i, _, _, _)| i).collect();

            // 1. Pad Snapping: check if close to a same-net through-hole pad (< via_pad_snap_dist_mm)
            for fp in &board.footprints {
                for pad in &fp.pads {
                    if pad.net_id == net_id && pad.is_through_hole {
                        let (px, py) = fp.pad_absolute_position(pad);
                        let d = (v_pos.0 - px).hypot(v_pos.1 - py);
                        if d > 1e-4 && d < config.via_pad_snap_dist_mm {
                            let all_clear = incident.iter().all(|&(_, l, other, w)| {
                                is_octilinear_segment((px, py), other)
                                    && is_track_segment_legal(board, (px, py), other, w, l, net_id, clearance)
                                    && is_segment_clear(board, Some(grid), (px, py), other, l, net_id, w)
                            });
                            if all_clear {
                                let mut pts = vec![v_pos, (px, py)];
                                pts.extend(incident.iter().map(|&(_, _, other, _)| other));
                                return Some(ViaSlideAction {
                                    v_idx,
                                    v_pos,
                                    net_id,
                                    target_pos: (px, py),
                                    remove_via: true,
                                    incident_indices,
                                    aabb: points_expanded_aabb(&pts, 0.60),
                                });
                            }
                        }
                    }
                }
            }

            // 2. Collinear & Jog Elimination:
            // If via is connected to a track that has a dogleg or micro-jog (< 1.0mm) before turning
            for &(t_idx, layer, other_pt, _) in &incident {
                let seg_len = (other_pt.0 - v_pos.0).hypot(other_pt.1 - v_pos.1);
                if seg_len < 1.0 && seg_len > 1e-4 {
                    let is_pad = board.footprints.iter().any(|fp| {
                        fp.pads.iter().any(|p| {
                            let (px, py) = fp.pad_absolute_position(p);
                            (px - other_pt.0).hypot(py - other_pt.1) < 1e-3
                        })
                    });
                    if !is_pad {
                        let has_connecting_track = board.tracks.iter().enumerate().any(|(i, t)| {
                            i != t_idx
                                && t.net_id == net_id
                                && t.layer == layer
                                && !t.is_locked
                                && ((t.start.0 - other_pt.0).hypot(t.start.1 - other_pt.1) < 1e-4
                                    || (t.end.0 - other_pt.0).hypot(t.end.1 - other_pt.1) < 1e-4)
                        });

                        if has_connecting_track
                            && is_via_location_legal_ignoring(board, other_pt, via_size, net_id, clearance, Some(v_pos))
                        {
                            let other_tracks_ok = incident.iter().all(|&(ot_idx, ot_layer, ot_other, ot_w)| {
                                if ot_idx == t_idx || (ot_other.0 - other_pt.0).hypot(ot_other.1 - other_pt.1) < 1e-4 {
                                    true
                                } else {
                                    is_octilinear_segment(other_pt, ot_other)
                                        && is_track_segment_legal(board, other_pt, ot_other, ot_w, ot_layer, net_id, clearance)
                                        && is_segment_clear(board, Some(grid), other_pt, ot_other, ot_layer, net_id, ot_w)
                                }
                            });

                            if other_tracks_ok {
                                let mut pts = vec![v_pos, other_pt];
                                pts.extend(incident.iter().map(|&(_, _, other, _)| other));
                                return Some(ViaSlideAction {
                                    v_idx,
                                    v_pos,
                                    net_id,
                                    target_pos: other_pt,
                                    remove_via: false,
                                    incident_indices,
                                    aabb: points_expanded_aabb(&pts, 0.60),
                                });
                            }
                        }
                    }
                }
            }

            // 3. Vector Sliding:
            // Test candidate positions along vectors of connected tracks (batch-evaluated on GPU when available)
            let orig_tot_len: f64 = incident.iter().map(|&(_, _, other, _)| (other.0 - v_pos.0).hypot(other.1 - v_pos.1)).sum();
            let mut raw_cands: Vec<((f64, f64), f64)> = Vec::new();

            for &(_, _, other_pt, _) in &incident {
                let dx = other_pt.0 - v_pos.0;
                let dy = other_pt.1 - v_pos.1;
                let dist = dx.hypot(dy);
                if dist < 0.15 {
                    continue;
                }
                let ux = dx / dist;
                let uy = dy / dist;

                let steps = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.50, 2.00];
                for &s in &steps {
                    if s >= dist - 0.05 {
                        break;
                    }
                    let cand = (v_pos.0 + s * ux, v_pos.1 + s * uy);
                    let cand_len: f64 = incident.iter().map(|&(_, _, other, _)| (other.0 - cand.0).hypot(other.1 - cand.1)).sum();
                    if cand_len >= orig_tot_len - 0.05 {
                        continue;
                    }
                    let oct_ok = incident.iter().all(|&(_, _, other, _)| {
                        (other.0 - cand.0).hypot(other.1 - cand.1) < 1e-4 || is_octilinear_segment(cand, other)
                    });
                    if oct_ok {
                        raw_cands.push((cand, cand_len));
                    }
                }
            }

            let mut best_cand: Option<((f64, f64), f64)> = None;
            if !raw_cands.is_empty() {
                let gpu_flags: Option<Vec<bool>> = if raw_cands.len() >= 4 {
                    let mut gpu_batch = Vec::with_capacity(raw_cands.len() * (1 + incident.len()));
                    let mut spans = Vec::with_capacity(raw_cands.len());
                    for &(cand, _) in &raw_cands {
                        let start_idx = gpu_batch.len();
                        gpu_batch.push(GpuDrcCandidate {
                            x1: cand.0 as f32,
                            y1: cand.1 as f32,
                            x2: v_pos.0 as f32,
                            y2: v_pos.1 as f32,
                            radius_or_half_width: (via_size * 0.5) as f32,
                            clearance: clearance as f32,
                            layer: 0,
                            net_id: net_id as u32,
                            kind: 0,
                            ignore_same_net_pads: 0,
                            _pad0: 1,
                            _pad1: 0,
                        });
                        for &(_, l, other, w) in &incident {
                            if (other.0 - cand.0).hypot(other.1 - cand.1) >= 1e-4 {
                                gpu_batch.push(GpuDrcCandidate {
                                    x1: cand.0 as f32,
                                    y1: cand.1 as f32,
                                    x2: other.0 as f32,
                                    y2: other.1 as f32,
                                    radius_or_half_width: (w * 0.5) as f32,
                                    clearance: clearance as f32,
                                    layer: l as u32,
                                    net_id: net_id as u32,
                                    kind: 1,
                                    ignore_same_net_pads: 1,
                                    _pad0: 0,
                                    _pad1: 0,
                                });
                            }
                        }
                        spans.push((start_idx, gpu_batch.len()));
                    }
                    drc_ctx.evaluate(grid.vulkan.as_ref(), &gpu_batch).map(|res| {
                        spans
                            .into_iter()
                            .map(|(s, e)| res[s..e].iter().all(|&v| v != 0))
                            .collect()
                    })
                } else {
                    None
                };

                let verify_cand = |cand: (f64, f64)| -> bool {
                    if !is_via_location_legal_ignoring(board, cand, via_size, net_id, clearance, Some(v_pos)) {
                        return false;
                    }
                    incident.iter().all(|&(_, l, other, w)| {
                        if (other.0 - cand.0).hypot(other.1 - cand.1) < 1e-4 {
                            true
                        } else {
                            is_track_segment_legal(board, cand, other, w, l, net_id, clearance)
                                && is_segment_clear(board, Some(grid), cand, other, l, net_id, w)
                        }
                    })
                };

                for (idx, &(cand, cand_len)) in raw_cands.iter().enumerate() {
                    let min_len = best_cand.map(|(_, l)| l).unwrap_or(orig_tot_len - 0.05);
                    if cand_len < min_len - 1e-4 {
                        if gpu_flags.as_ref().map(|f| f[idx]).unwrap_or(true) && verify_cand(cand) {
                            best_cand = Some((cand, cand_len));
                        }
                    }
                }
            }

            if let Some((best_pos, _)) = best_cand {
                let mut pts = vec![v_pos, best_pos];
                pts.extend(incident.iter().map(|&(_, _, other, _)| other));
                return Some(ViaSlideAction {
                    v_idx,
                    v_pos,
                    net_id,
                    target_pos: best_pos,
                    remove_via: false,
                    incident_indices,
                    aabb: points_expanded_aabb(&pts, 0.60),
                });
            }

            None
        })
        .collect();

    if slide_candidates.is_empty() {
        return false;
    }

    let mut accepted_nets: HashSet<usize> = HashSet::new();
    let mut accepted_aabbs: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut vias_to_remove: Vec<usize> = Vec::new();

    for action in slide_candidates {
        if accepted_nets.contains(&action.net_id)
            || accepted_aabbs.iter().any(|&other| aabb_overlaps(action.aabb, other))
        {
            continue;
        }
        accepted_nets.insert(action.net_id);
        accepted_aabbs.push(action.aabb);

        if action.remove_via {
            vias_to_remove.push(action.v_idx);
        } else {
            board.vias[action.v_idx].position = action.target_pos;
        }
        for &t_idx in &action.incident_indices {
            let t = &mut board.tracks[t_idx];
            if (t.start.0 - action.v_pos.0).hypot(t.start.1 - action.v_pos.1) < 1e-4 {
                t.start = action.target_pos;
            } else {
                t.end = action.target_pos;
            }
        }
    }

    vias_to_remove.sort_unstable();
    vias_to_remove.dedup();
    for &idx in vias_to_remove.iter().rev() {
        board.vias.remove(idx);
    }
    board.tracks.retain(|t| (t.start.0 - t.end.0).hypot(t.start.1 - t.end.1) > 1e-4);

    true
}

/// Decomposes any non-octilinear segment in a [`TrackChain`] into standard 45° doglegs.
///
/// # Arguments
/// * `chain` - Mutable track chain whose polyline vertices are regularized to 0°/45°/90° orientations.
///
/// # Side Effects
/// Mutates `chain.points` in place and merges collinear vertices.
pub fn enforce_45_grid(chain: &mut TrackChain) {
    if chain.points.len() < 2 {
        return;
    }
    let mut new_points = Vec::with_capacity(chain.points.len() * 2);
    new_points.push(chain.points[0]);

    for k in 0..chain.points.len() - 1 {
        let p0 = *new_points.last().unwrap();
        let p1 = chain.points[k + 1];
        if (p0.0 - p1.0).hypot(p0.1 - p1.1) < 1e-5 {
            continue;
        }
        if is_octilinear_segment(p0, p1) {
            new_points.push(p1);
        } else {
            let dogleg = compute_45_dogleg(p0, p1, false);
            for &dp in &dogleg[1..] {
                if (dp.0 - new_points.last().unwrap().0).hypot(dp.1 - new_points.last().unwrap().1) > 1e-5 {
                    new_points.push(dp);
                }
            }
        }
    }

    chain.points = new_points;
    merge_collinear_points(&mut chain.points);
}

/// Runs standard octilinear post-route track glossing across all unlocked signal nets on the board.
///
/// # Arguments
/// * `board` - Mutable board state whose unlocked tracks are glossed.
/// * `grid` - Routing grid providing clearance data and optional GPU acceleration.
///
/// # Side Effects
/// Replaces unlocked signal tracks in `board.tracks` with straightened, de-jogged, and mitered segments.
pub fn gloss_tracks(board: &mut Board, grid: &RoutingGrid) {
    gloss_tracks_with_mode(board, grid, false);
}

/// Runs post-route track glossing in either standard 45° octilinear mode or organic any-angle tangent-arc mode.
///
/// When `any_angle_mode` is `true`, performs standard glossing followed by homotopic rubber-band
/// relaxation, $G^1$-continuous tangent-arc corner replacement, and acute-junction filleting.
///
/// # Arguments
/// * `board` - Mutable board state whose tracks and arcs are updated.
/// * `grid` - Routing grid providing clearance data and optional GPU acceleration.
/// * `any_angle_mode` - When `true`, enables homotopic rubber-banding and tangent-arc generation.
///
/// # Side Effects
/// Mutates `board.tracks` and `board.arcs`.
pub fn gloss_tracks_with_mode(board: &mut Board, grid: &RoutingGrid, any_angle_mode: bool) {
    if any_angle_mode {
        gloss_tracks_standard(board, grid);
        relax_homotopic_rubber_bands_with_grid(board, Some(grid));
        apply_any_angle_tangent_arcs(board);
        fillet_acute_angle_junctions(board, default_acid_trap_fillet_radius());
    } else {
        gloss_tracks_standard(board, grid);
    }
}

/// Canonical lookup key `(quant_p1, quant_p2, layer, net_id)` for pre-evaluated GPU shortcut clearance queries.
pub(crate) type GpuShortcutKey = ((i64, i64), (i64, i64), usize, usize);
/// Hash map mapping a [`GpuShortcutKey`] to whether the candidate segment passed GPU DRC screening.
pub(crate) type GpuShortcutMap = HashMap<GpuShortcutKey, bool>;

/// Constructs an order-independent [`GpuShortcutKey`] for a candidate segment between `p1` and `p2`.
///
/// # Arguments
/// * `p1` - First endpoint `(x, y)` in millimeters (`mm`).
/// * `p2` - Second endpoint `(x, y)` in millimeters (`mm`).
/// * `layer` - Copper layer index.
/// * `net_id` - Net ID of the segment.
///
/// # Returns
/// Canonicalized `(min_q, max_q, layer, net_id)` tuple.
#[inline]
pub(crate) fn canonical_shortcut_key(p1: (f64, f64), p2: (f64, f64), layer: usize, net_id: usize) -> GpuShortcutKey {
    let q1 = quant_point(p1);
    let q2 = quant_point(p2);
    if q1 <= q2 {
        (q1, q2, layer, net_id)
    } else {
        (q2, q1, layer, net_id)
    }
}

/// Checks whether a candidate shortcut segment is DRC-clean, consulting a pre-evaluated [`GpuShortcutMap`] first.
///
/// If the GPU map marks the segment as colliding (`false`), returns `false` in $O(1)$ without CPU geometry checks;
/// otherwise performs exact verification via [`is_segment_clear`].
///
/// # Arguments
/// * `board` - Board state with obstacles.
/// * `grid` - Routing grid.
/// * `p1` - Segment start `(x, y)` in `mm`.
/// * `p2` - Segment end `(x, y)` in `mm`.
/// * `layer` - Copper layer index.
/// * `net_id` - Net ID of the segment.
/// * `width` - Trace width in `mm`.
/// * `gpu_map` - Optional pre-evaluated GPU shortcut lookup table.
///
/// # Returns
/// `true` if the segment is clear of all obstacles and DRC violations.
#[inline]
pub(crate) fn is_segment_clear_with_gpu(
    board: &Board,
    grid: &RoutingGrid,
    p1: (f64, f64),
    p2: (f64, f64),
    layer: usize,
    net_id: usize,
    width: f64,
    gpu_map: Option<&GpuShortcutMap>,
) -> bool {
    if let Some(map) = gpu_map {
        let key = canonical_shortcut_key(p1, p2, layer, net_id);
        if let Some(&false) = map.get(&key) {
            return false;
        }
    }
    is_segment_clear(board, Some(grid), p1, p2, layer, net_id, width)
}

/// Enumerates all potential octilinear and 45°-dogleg shortcut segments across `chains` for batch GPU DRC evaluation.
///
/// # Arguments
/// * `chains` - Slice of track chains to inspect.
/// * `board` - Board state used to look up netclass clearances.
/// * `out_keys` - Output vector receiving canonical shortcut keys.
/// * `out_cands` - Output vector receiving corresponding [`GpuDrcCandidate`] descriptors.
/// * `seen` - Deduplication set of already-queued keys.
///
/// # Side Effects
/// Appends unique keys and GPU candidate descriptors to `out_keys`, `out_cands`, and `seen`.
pub(crate) fn collect_chain_shortcut_candidates(
    chains: &[TrackChain],
    board: &Board,
    out_keys: &mut Vec<GpuShortcutKey>,
    out_cands: &mut Vec<GpuDrcCandidate>,
    seen: &mut HashSet<GpuShortcutKey>,
) {
    for chain in chains {
        let n = chain.points.len();
        if n < 3 {
            continue;
        }
        let clearance = board.get_netclass_for_net(chain.net_id).clearance as f32;
        let half_w = (chain.width * 0.5) as f32;
        let mut push_seg = |p1: (f64, f64), p2: (f64, f64)| {
            if (p1.0 - p2.0).hypot(p1.1 - p2.1) <= 1e-4 {
                return;
            }
            let key = canonical_shortcut_key(p1, p2, chain.layer, chain.net_id);
            if seen.insert(key) {
                out_keys.push(key);
                out_cands.push(GpuDrcCandidate {
                    x1: p1.0 as f32,
                    y1: p1.1 as f32,
                    x2: p2.0 as f32,
                    y2: p2.1 as f32,
                    radius_or_half_width: half_w,
                    clearance,
                    layer: chain.layer as u32,
                    net_id: chain.net_id as u32,
                    kind: 1,
                    ignore_same_net_pads: 1,
                    _pad0: 0,
                    _pad1: 0,
                });
            }
        };

        for i in 0..n {
            for j in (i + 2)..n {
                let p_i = chain.points[i];
                let p_j = chain.points[j];
                if is_octilinear_segment(p_i, p_j) {
                    push_seg(p_i, p_j);
                }
                if j - i <= 16 {
                    let dx = p_j.0 - p_i.0;
                    let dy = p_j.1 - p_i.1;
                    let s = dx.abs().min(dy.abs());
                    if s > 1e-4 {
                        let sx = s * dx.signum();
                        let sy = s * dy.signum();
                        let c_a = (p_i.0 + sx, p_i.1 + sy);
                        let c_b = (p_j.0 - sx, p_j.1 - sy);
                        push_seg(p_i, c_a);
                        push_seg(c_a, p_j);
                        push_seg(p_i, c_b);
                        push_seg(c_b, p_j);
                    }
                }
            }
        }
    }
}

/// GPU-accelerated variant of [`rubber_band_straighten`] using pre-screened shortcut results from `gpu_map`.
///
/// # Arguments
/// * `chain` - Mutable track chain to straighten.
/// * `board` - Board state with obstacles.
/// * `grid` - Routing grid.
/// * `gpu_map` - Optional pre-evaluated GPU shortcut clearance map.
///
/// # Side Effects
/// Mutates `chain.points` in place.
pub(crate) fn rubber_band_straighten_gpu(
    chain: &mut TrackChain,
    board: &Board,
    grid: &RoutingGrid,
    gpu_map: Option<&GpuShortcutMap>,
) {
    if chain.points.len() < 3 {
        return;
    }
    let is_pad_center = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        })
    };
    let is_special_point = |pt: (f64, f64)| -> bool {
        is_pad_center(pt) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };
    let starts_at_pad = is_pad_center(chain.points[0]);
    let ends_at_pad = is_pad_center(*chain.points.last().unwrap());

    let mut changed = true;
    let mut pass = 0;
    while changed && pass < 4 {
        changed = false;
        pass += 1;
        let mut new_points = Vec::with_capacity(chain.points.len());
        let mut i = 0;
        new_points.push(chain.points[0]);

        while i + 1 < chain.points.len() {
            if i == 0 && starts_at_pad {
                i += 1;
                new_points.push(chain.points[i]);
                continue;
            }
            let max_j = if ends_at_pad {
                chain.points.len().saturating_sub(2)
            } else {
                chain.points.len().saturating_sub(1)
            };
            let mut best_j = None;
            for j in (i + 2..=max_j).rev() {
                let p1 = chain.points[i];
                let p2 = chain.points[j];
                if (i + 1..j).any(|k| {
                    let pk = chain.points[k];
                    is_special_point(pk)
                        && (pk.0 - p1.0).hypot(pk.1 - p1.1) > 0.15
                        && (pk.0 - p2.0).hypot(pk.1 - p2.1) > 0.15
                }) {
                    continue;
                }
                if is_octilinear_segment(p1, p2)
                    && is_segment_clear_with_gpu(board, grid, p1, p2, chain.layer, chain.net_id, chain.width, gpu_map)
                {
                    best_j = Some(j);
                    break;
                }
            }

            if let Some(j) = best_j {
                new_points.push(chain.points[j]);
                i = j;
                changed = true;
            } else {
                i += 1;
                new_points.push(chain.points[i]);
            }
        }
        chain.points = new_points;
        merge_collinear_points(&mut chain.points);
    }
}

/// GPU-accelerated variant of [`eliminate_stair_steps`] using pre-screened shortcut results from `gpu_map`.
///
/// # Arguments
/// * `chain` - Mutable track chain whose staircase jogs are collapsed into canonical 45° doglegs.
/// * `board` - Board state with obstacles.
/// * `grid` - Routing grid.
/// * `gpu_map` - Optional pre-evaluated GPU shortcut clearance map.
///
/// # Side Effects
/// Mutates `chain.points` in place.
pub(crate) fn eliminate_stair_steps_gpu(
    chain: &mut TrackChain,
    board: &Board,
    grid: &RoutingGrid,
    gpu_map: Option<&GpuShortcutMap>,
) {
    if chain.points.len() < 3 {
        return;
    }
    let is_special_point = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        }) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };
    let mut changed = true;
    let mut pass = 0;
    while changed && pass < 4 {
        changed = false;
        pass += 1;
        let mut i = 0;
        while i + 2 < chain.points.len() {
            let max_j = (i + 16).min(chain.points.len() - 1);
            let mut best_replacement = None;

            for j in (i + 2..=max_j).rev() {
                let p_i = chain.points[i];
                let p_j = chain.points[j];
                if (i + 1..j).any(|k| {
                    let pk = chain.points[k];
                    is_special_point(pk)
                        && (pk.0 - p_i.0).hypot(pk.1 - p_i.1) > 0.15
                        && (pk.0 - p_j.0).hypot(pk.1 - p_j.1) > 0.15
                }) {
                    continue;
                }
                let dx = p_j.0 - p_i.0;
                let dy = p_j.1 - p_i.1;
                let s = dx.abs().min(dy.abs());
                let canon_len = s * std::f64::consts::SQRT_2 + (dx.abs() - dy.abs()).abs();
                let orig_len: f64 = chain.points[i..=j]
                    .windows(2)
                    .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
                    .sum();

                let existing_bends = count_chain_bends(&chain.points[i..=j]);
                if existing_bends < 2 && j - i < 3 && orig_len <= canon_len + 0.02 {
                    continue;
                }

                if dx.abs() < 1e-4 || dy.abs() < 1e-4 || (dx.abs() - dy.abs()).abs() < 1e-4 {
                    if is_segment_clear_with_gpu(board, grid, p_i, p_j, chain.layer, chain.net_id, chain.width, gpu_map) {
                        best_replacement = Some((j, vec![p_i, p_j]));
                        break;
                    }
                }

                if s > 1e-4 {
                    let sx = s * dx.signum();
                    let sy = s * dy.signum();

                    let c_a = (p_i.0 + sx, p_i.1 + sy);
                    let clear_a = is_segment_clear_with_gpu(board, grid, p_i, c_a, chain.layer, chain.net_id, chain.width, gpu_map)
                        && is_segment_clear_with_gpu(board, grid, c_a, p_j, chain.layer, chain.net_id, chain.width, gpu_map);

                    let c_b = (p_j.0 - sx, p_j.1 - sy);
                    let clear_b = is_segment_clear_with_gpu(board, grid, p_i, c_b, chain.layer, chain.net_id, chain.width, gpu_map)
                        && is_segment_clear_with_gpu(board, grid, c_b, p_j, chain.layer, chain.net_id, chain.width, gpu_map);

                    if clear_a && clear_b {
                        let orig_cx: f64 = chain.points[i..=j].iter().map(|p| p.0).sum::<f64>() / (j - i + 1) as f64;
                        let orig_cy: f64 = chain.points[i..=j].iter().map(|p| p.1).sum::<f64>() / (j - i + 1) as f64;
                        let dist_a = (c_a.0 - orig_cx).hypot(c_a.1 - orig_cy);
                        let dist_b = (c_b.0 - orig_cx).hypot(c_b.1 - orig_cy);
                        let chosen_c = if dist_a <= dist_b { c_a } else { c_b };
                        best_replacement = Some((j, vec![p_i, chosen_c, p_j]));
                        break;
                    } else if clear_a {
                        best_replacement = Some((j, vec![p_i, c_a, p_j]));
                        break;
                    } else if clear_b {
                        best_replacement = Some((j, vec![p_i, c_b, p_j]));
                        break;
                    }
                }
            }

            if let Some((j, repl)) = best_replacement {
                let mut new_points = Vec::with_capacity(chain.points.len());
                new_points.extend_from_slice(&chain.points[..i]);
                new_points.extend(repl);
                new_points.extend_from_slice(&chain.points[j + 1..]);
                chain.points = new_points;
                merge_collinear_points(&mut chain.points);
                changed = true;
                i += 1;
            } else {
                i += 1;
            }
        }
    }
}

/// GPU-accelerated variant of [`reduce_jogs`] using pre-screened shortcut results from `gpu_map`.
///
/// # Arguments
/// * `chain` - Mutable track chain whose short micro-jogs are collapsed.
/// * `board` - Board state with obstacles.
/// * `grid` - Routing grid.
/// * `gpu_map` - Optional pre-evaluated GPU shortcut clearance map.
///
/// # Side Effects
/// Mutates `chain.points` in place.
pub(crate) fn reduce_jogs_gpu(
    chain: &mut TrackChain,
    board: &Board,
    grid: &RoutingGrid,
    gpu_map: Option<&GpuShortcutMap>,
) {
    if chain.points.len() < 4 {
        return;
    }
    let mut changed = true;
    let mut pass = 0;
    let max_jog_offset = (2.0 * chain.width).max(0.30);
    let is_oct = is_octilinear_segment;
    let is_special_point = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        }) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };

    while changed && pass < 4 {
        changed = false;
        pass += 1;
        let mut i = 0;
        while i + 3 < chain.points.len() {
            let p0 = chain.points[i];
            let p1 = chain.points[i + 1];
            let p2 = chain.points[i + 2];
            let p3 = chain.points[i + 3];

            let jog_len = (p2.0 - p1.0).hypot(p2.1 - p1.1);
            let lat_offset = dist_to_segment(p2.0, p2.1, p0.0, p0.1, p1.0, p1.1);
            let is_micro_jog = jog_len <= max_jog_offset || lat_offset <= max_jog_offset;

            if is_micro_jog {
                if !is_special_point(p1)
                    && !is_special_point(p2)
                    && is_oct(p0, p3)
                    && is_segment_clear_with_gpu(board, grid, p0, p3, chain.layer, chain.net_id, chain.width, gpu_map)
                {
                    chain.points.remove(i + 2);
                    chain.points.remove(i + 1);
                    merge_collinear_points(&mut chain.points);
                    changed = true;
                    continue;
                }

                if !is_special_point(p1)
                    && is_oct(p0, p2)
                    && is_segment_clear_with_gpu(board, grid, p0, p2, chain.layer, chain.net_id, chain.width, gpu_map)
                {
                    chain.points.remove(i + 1);
                    merge_collinear_points(&mut chain.points);
                    changed = true;
                    continue;
                }

                if !is_special_point(p2)
                    && is_oct(p1, p3)
                    && is_segment_clear_with_gpu(board, grid, p1, p3, chain.layer, chain.net_id, chain.width, gpu_map)
                {
                    chain.points.remove(i + 2);
                    merge_collinear_points(&mut chain.points);
                    changed = true;
                    continue;
                }
            }
            i += 1;
        }
    }
}

/// GPU-accelerated variant of [`reduce_corners`] using pre-screened shortcut results from `gpu_map`.
///
/// # Arguments
/// * `chain` - Mutable track chain whose 4-point corner sequences are simplified.
/// * `board` - Board state with obstacles.
/// * `grid` - Routing grid.
/// * `gpu_map` - Optional pre-evaluated GPU shortcut clearance map.
///
/// # Side Effects
/// Mutates `chain.points` in place.
pub(crate) fn reduce_corners_gpu(
    chain: &mut TrackChain,
    board: &Board,
    grid: &RoutingGrid,
    gpu_map: Option<&GpuShortcutMap>,
) {
    if chain.points.len() < 4 {
        return;
    }
    let is_pad_center = |pt: (f64, f64)| -> bool {
        board.footprints.iter().any(|fp| {
            fp.pads.iter().any(|p| {
                let (px, py) = fp.pad_absolute_position(p);
                (px - pt.0).hypot(py - pt.1) < 0.05
            })
        })
    };
    let is_special_point = |pt: (f64, f64)| -> bool {
        is_pad_center(pt) || board.vias.iter().any(|v| (v.position.0 - pt.0).hypot(v.position.1 - pt.1) < 0.05)
    };
    let is_oct = is_octilinear_segment;

    let mut changed = true;
    let mut pass = 0;
    while changed && pass < 4 {
        changed = false;
        pass += 1;
        let mut i = 0;
        while i + 3 < chain.points.len() {
            let p0 = chain.points[i];
            let p1 = chain.points[i + 1];
            let p2 = chain.points[i + 2];
            let p3 = chain.points[i + 3];

            if !is_special_point(p1)
                && !is_special_point(p2)
                && is_oct(p0, p3)
                && is_segment_clear_with_gpu(board, grid, p0, p3, chain.layer, chain.net_id, chain.width, gpu_map)
            {
                chain.points.remove(i + 2);
                chain.points.remove(i + 1);
                merge_collinear_points(&mut chain.points);
                changed = true;
                continue;
            }

            if !is_special_point(p1)
                && is_oct(p0, p2)
                && is_segment_clear_with_gpu(board, grid, p0, p2, chain.layer, chain.net_id, chain.width, gpu_map)
            {
                chain.points.remove(i + 1);
                merge_collinear_points(&mut chain.points);
                changed = true;
                continue;
            }

            if !is_special_point(p2)
                && is_oct(p1, p3)
                && is_segment_clear_with_gpu(board, grid, p1, p3, chain.layer, chain.net_id, chain.width, gpu_map)
            {
                chain.points.remove(i + 2);
                merge_collinear_points(&mut chain.points);
                changed = true;
                continue;
            }

            i += 1;
        }
    }
}

/// Extracts [`TrackChain`] polylines for a single net, enforces 45° octilinear alignment, and runs initial acid-trap avoidance.
///
/// # Arguments
/// * `net_id` - Net ID being prepared for glossing.
/// * `net_tracks` - Unlocked track segments belonging to `net_id`.
/// * `board_without_net` - Board snapshot with the current wave's unlocked tracks removed.
/// * `grid` - Routing grid.
///
/// # Returns
/// Vector of prepared [`TrackChain`] polylines ready for shortcut evaluation and glossing.
pub(crate) fn prepare_net_chains_for_gloss(
    net_id: usize,
    net_tracks: &[TrackSegment],
    board_without_net: &Board,
    grid: &RoutingGrid,
) -> Vec<TrackChain> {
    crate::memlog::set_phase_fast(&format!("gloss_net_{}::extract_chains(in={})", net_id, net_tracks.len()));
    let mut chains = extract_track_chains(net_tracks, board_without_net);
    for (ci, chain) in chains.iter_mut().enumerate() {
        crate::memlog::set_phase_fast(&format!("gloss_net_{}_c{}::enforce_45_grid(pts={})", net_id, ci, chain.points.len()));
        enforce_45_grid(chain);
        merge_collinear_points(&mut chain.points);
        crate::memlog::set_phase_fast(&format!("gloss_net_{}_c{}::avoid_acid_traps_1(pts={})", net_id, ci, chain.points.len()));
        avoid_acid_traps(chain, board_without_net, grid);
        merge_collinear_points(&mut chain.points);
    }
    chains
}

/// Runs the full sequence of chain-level glossing passes on prepared chains and decomposes them back into [`TrackSegment`] items.
///
/// Executes rubber-band straightening, stair-step elimination, jog reduction, corner reduction,
/// line repositioning, 90° corner mitering, and final acid-trap avoidance.
///
/// # Arguments
/// * `net_id` - Net ID of the chains.
/// * `chains` - Prepared [`TrackChain`] polylines for `net_id`.
/// * `board_without_net` - Board snapshot without the current wave's unlocked tracks.
/// * `grid` - Routing grid.
/// * `gpu_map` - Optional pre-evaluated GPU shortcut clearance map.
///
/// # Returns
/// Vector of glossed [`TrackSegment`] primitives for `net_id`.
pub(crate) fn finish_gloss_net_chains(
    net_id: usize,
    mut chains: Vec<TrackChain>,
    board_without_net: &Board,
    grid: &RoutingGrid,
    gpu_map: Option<&GpuShortcutMap>,
) -> Vec<TrackSegment> {
    for (ci, chain) in chains.iter_mut().enumerate() {
        crate::memlog::set_phase_fast(&format!("gloss_net_{}_c{}::rubber_band_straighten(pts={})", net_id, ci, chain.points.len()));
        rubber_band_straighten_gpu(chain, board_without_net, grid, gpu_map);
        crate::memlog::set_phase_fast(&format!("gloss_net_{}_c{}::eliminate_stair_steps(pts={})", net_id, ci, chain.points.len()));
        eliminate_stair_steps_gpu(chain, board_without_net, grid, gpu_map);
        crate::memlog::set_phase_fast(&format!("gloss_net_{}_c{}::reduce_jogs(pts={})", net_id, ci, chain.points.len()));
        reduce_jogs_gpu(chain, board_without_net, grid, gpu_map);
        crate::memlog::set_phase_fast(&format!("gloss_net_{}_c{}::reduce_corners(pts={})", net_id, ci, chain.points.len()));
        reduce_corners_gpu(chain, board_without_net, grid, gpu_map);
        merge_collinear_points(&mut chain.points);
        crate::memlog::set_phase_fast(&format!("gloss_net_{}_c{}::reposition_lines(pts={})", net_id, ci, chain.points.len()));
        reposition_lines(chain, board_without_net, grid);
        merge_collinear_points(&mut chain.points);
        crate::memlog::set_phase_fast(&format!("gloss_net_{}_c{}::smoothen_corners(pts={})", net_id, ci, chain.points.len()));
        smoothen_corners(chain, board_without_net, grid);
        crate::memlog::set_phase_fast(&format!("gloss_net_{}_c{}::avoid_acid_traps_2(pts={})", net_id, ci, chain.points.len()));
        avoid_acid_traps(chain, board_without_net, grid);
        enforce_45_grid(chain);
        merge_collinear_points(&mut chain.points);
    }

    let mut out = Vec::new();
    for chain in chains {
        for k in 0..chain.points.len().saturating_sub(1) {
            let p1 = chain.points[k];
            let p2 = chain.points[k + 1];
            if (p1.0 - p2.0).hypot(p1.1 - p2.1) > 1e-5 {
                out.push(TrackSegment {
                    start: p1,
                    end: p2,
                    width: chain.width,
                    layer: chain.layer,
                    net_id,
                    is_locked: false,
                });
            }
        }
    }
    out
}

/// Glosses all unlocked tracks of a single isolated net, optionally evaluating shortcut candidates on the GPU.
///
/// # Arguments
/// * `net_id` - Net ID to gloss.
/// * `net_tracks` - Unlocked track segments of `net_id`.
/// * `board_without_net` - Board state with `net_tracks` removed.
/// * `grid` - Routing grid.
///
/// # Returns
/// Vector of glossed [`TrackSegment`] items.
pub(crate) fn gloss_single_net_chains(net_id: usize, net_tracks: &[TrackSegment], board_without_net: &Board, grid: &RoutingGrid) -> Vec<TrackSegment> {
    let chains = prepare_net_chains_for_gloss(net_id, net_tracks, board_without_net, grid);
    let gpu_map: Option<GpuShortcutMap> = if grid.vulkan.is_some() {
        let mut keys = Vec::new();
        let mut cands = Vec::new();
        let mut seen = HashSet::new();
        collect_chain_shortcut_candidates(&chains, board_without_net, &mut keys, &mut cands, &mut seen);
        if !cands.is_empty() {
            let drc_ctx = GpuDrcContext::from_board(board_without_net);
            drc_ctx.evaluate(grid.vulkan.as_ref(), &cands).map(|res| {
                keys.into_iter().zip(res.into_iter()).map(|(k, v)| (k, v != 0)).collect()
            })
        } else {
            None
        }
    } else {
        None
    };
    finish_gloss_net_chains(net_id, chains, board_without_net, grid, gpu_map.as_ref())
}

/// Runs wave-parallel octilinear track glossing across all non-differential unlocked nets on `board`.
///
/// Partitions nets into spatially disjoint waves using `0.60 mm`-expanded 2D AABBs so that all nets
/// within a wave can be extracted, batch-screened on the GPU, and glossed in parallel via Rayon
/// without mutual clearance interference. Coupled differential pairs are left untouched to preserve
/// their phase-matched spacing.
///
/// # Arguments
/// * `board` - Mutable board state whose unlocked tracks are glossed.
/// * `grid` - Routing grid providing clearance data and optional GPU acceleration.
///
/// # Side Effects
/// Replaces unlocked non-differential tracks in `board.tracks`.
pub fn gloss_tracks_standard(board: &mut Board, grid: &RoutingGrid) {
    let mut net_ids: Vec<usize> = board
        .tracks
        .iter()
        .filter(|t| !t.is_locked)
        .map(|t| t.net_id)
        .collect();
    net_ids.sort_unstable();
    net_ids.dedup();
    // Glossing each half of a diff pair independently pulls the pair apart; leave coupled pairs as routed.
    let pair_nets: Vec<usize> = net_ids
        .iter()
        .copied()
        .filter(|&n| net_ids.iter().any(|&o| AutoRouter::are_diff_pair_mates(&board.nets[n], &board.nets[o])))
        .collect();
    net_ids.retain(|n| !pair_nets.contains(n));

    if net_ids.is_empty() {
        return;
    }

    // Compute 0.60mm-expanded 2D AABB for each net's unlocked tracks and partition into spatially disjoint independent sets
    let mut net_aabbs: HashMap<usize, (f64, f64, f64, f64)> = HashMap::with_capacity(net_ids.len());
    for &net_id in &net_ids {
        let mut pts = Vec::new();
        for t in board.tracks.iter().filter(|t| t.net_id == net_id && !t.is_locked) {
            pts.push(t.start);
            pts.push(t.end);
        }
        if !pts.is_empty() {
            net_aabbs.insert(net_id, points_expanded_aabb(&pts, 0.60));
        }
    }

    let mut waves: Vec<Vec<usize>> = Vec::new();
    for &net_id in &net_ids {
        let Some(&aabb) = net_aabbs.get(&net_id) else {
            continue;
        };
        let mut placed = false;
        for wave in &mut waves {
            let conflicts = wave.iter().any(|&other_net| {
                net_aabbs
                    .get(&other_net)
                    .map_or(false, |&other_aabb| aabb_overlaps(aabb, other_aabb))
            });
            if !conflicts {
                wave.push(net_id);
                placed = true;
                break;
            }
        }
        if !placed {
            waves.push(vec![net_id]);
        }
    }

    for wave in waves {
        if wave.len() == 1 {
            let net_id = wave[0];
            let net_tracks: Vec<TrackSegment> = board
                .tracks
                .iter()
                .filter(|t| t.net_id == net_id && !t.is_locked)
                .cloned()
                .collect();
            if net_tracks.is_empty() {
                continue;
            }
            board.tracks.retain(|t| t.net_id != net_id || t.is_locked);
            let glossed = gloss_single_net_chains(net_id, &net_tracks, board, grid);
            board.tracks.extend(glossed);
        } else {
            let wave_set: HashSet<usize> = wave.iter().copied().collect();
            let mut tracks_by_net: HashMap<usize, Vec<TrackSegment>> = HashMap::with_capacity(wave.len());
            for t in board.tracks.iter().filter(|t| !t.is_locked && wave_set.contains(&t.net_id)) {
                tracks_by_net.entry(t.net_id).or_default().push(t.clone());
            }
            // All nets in a wave have disjoint 0.60mm-expanded AABBs, so removing all unlocked tracks
            // in the wave yields the exact obstacle board for every net in the wave.
            board.tracks.retain(|t| !wave_set.contains(&t.net_id) || t.is_locked);

            let prepared_wave: Vec<(usize, Vec<TrackChain>)> = wave
                .par_iter()
                .filter_map(|&net_id| {
                    let net_tracks = tracks_by_net.get(&net_id)?;
                    if net_tracks.is_empty() {
                        return None;
                    }
                    let chains = prepare_net_chains_for_gloss(net_id, net_tracks, board, grid);
                    Some((net_id, chains))
                })
                .collect();

            // Pre-generate all candidate shortcut segments across all chains in the wave and evaluate in a single GPU batch dispatch
            let gpu_map: Option<GpuShortcutMap> = if grid.vulkan.is_some() {
                let mut keys = Vec::new();
                let mut shortcut_cands = Vec::new();
                let mut seen = HashSet::new();
                for (_, chains) in &prepared_wave {
                    collect_chain_shortcut_candidates(chains, board, &mut keys, &mut shortcut_cands, &mut seen);
                }
                if !shortcut_cands.is_empty() {
                    let drc_ctx = GpuDrcContext::from_board(board);
                    drc_ctx.evaluate(grid.vulkan.as_ref(), &shortcut_cands).map(|res| {
                        keys.into_iter().zip(res.into_iter()).map(|(k, v)| (k, v != 0)).collect()
                    })
                } else {
                    None
                }
            } else {
                None
            };

            let wave_results: Vec<Vec<TrackSegment>> = prepared_wave
                .into_par_iter()
                .map(|(net_id, chains)| {
                    finish_gloss_net_chains(net_id, chains, board, grid, gpu_map.as_ref())
                })
                .collect();

            for glossed in wave_results {
                board.tracks.extend(glossed);
            }
        }
    }
}
