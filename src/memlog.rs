// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! # Process Memory Telemetry & OOM Safety Watchdog (`memlog`)
//!
//! Provides low-overhead Resident Set Size (RSS), Virtual Memory High-Water Mark (`VmHWM`),
//! and heap/stack data (`VmData`) instrumentation along with a 20 ms background watchdog
//! thread that protects the host workstation from runaway memory allocations.
//!
//! ## Architecture & Primary Workflows
//!
//! - **Fast Linux `/proc` Sampling:** [`read_proc_statm_mb`] reads `/proc/self/statm` in a single
//!   file read (converting 4 KiB pages to MiB) for sub-microsecond inner-loop checks, while
//!   [`read_proc_status_mb`] parses `/proc/self/status` for exact kernel `VmHWM` high-water marks.
//! - **Phase Tracking & Checkpoints:** Placement and routing stages call [`checkpoint`],
//!   [`checkpoint_board`], or [`set_phase_fast`] to attribute memory deltas ($\Delta\text{ MB}$)
//!   to specific algorithmic phases in `/tmp/vulcans_trials_mem.log` (or the path specified by
//!   the `VULCANS_MEM_LOG` environment variable).
//! - **Hard OOM Guard:** Both explicit checkpoints and the background `vulcan-mem-watchdog` thread
//!   compare current RSS against [`DEFAULT_MAX_RSS_MB`] (or `VULCANS_MAX_RSS_MB`, minimum 256 MiB).
//!   If exceeded, the logger flushes a stack backtrace and terminates the process with exit code `99`
//!   before the system swap thrashing or kernel OOM-killer can freeze the desktop session.

use crate::board::Board;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Default filesystem path where process memory and phase transition logs are written
/// when `VULCANS_MEM_LOG` is not set.
pub const DEFAULT_MEM_LOG_PATH: &str = "/tmp/vulcans_trials_mem.log";

/// Default maximum Resident Set Size (RSS) safety ceiling in megabytes (`4096.0` MiB / 4 GiB)
/// before the memory watchdog aborts the process. Can be overridden via `VULCANS_MAX_RSS_MB`.
pub const DEFAULT_MAX_RSS_MB: f64 = 4096.0;

/// Internal singleton state protected by [`MEM_STATE`] tracking the active log file,
/// phase label, and historical RSS watermarks.
struct MemLoggerState {
    /// Monotonic timestamp captured when the memory logger singleton was first initialized.
    start_instant: Instant,
    /// Active filesystem path of the memory log file.
    log_path: String,
    /// Open writable file handle for the memory log, if opening succeeded.
    file: Option<File>,
    /// Resident Set Size in megabytes (`MB`) recorded at the previous logged event.
    last_rss_mb: f64,
    /// Peak Resident Set Size in megabytes (`MB`) observed across the lifetime of the process.
    peak_rss_mb: f64,
    /// Name of the currently executing placement or routing phase.
    current_phase: String,
    /// Configured hard Resident Set Size ceiling in megabytes (`MB`) before emergency abort.
    max_rss_mb: f64,
}

/// Process-global lazily initialized mutex holding the [`MemLoggerState`] singleton.
static MEM_STATE: OnceLock<Mutex<MemLoggerState>> = OnceLock::new();

/// Atomic flag ensuring the `vulcan-mem-watchdog` background thread is spawned at most once.
static WATCHDOG_STARTED: AtomicBool = AtomicBool::new(false);

/// Reads `(rss_mb, vmsize_mb, data_mb)` rapidly from `/proc/self/statm` (Linux 4 KiB pages).
///
/// Parses the first (total program size), second (resident set size), and sixth (data + stack)
/// page counters from `/proc/self/statm` and converts them from 4 KiB pages into megabytes (MiB).
///
/// # Returns
///
/// A 3-tuple `(rss_mb, vmsize_mb, data_mb)` in megabytes (`MB`). Returns `(0.0, 0.0, 0.0)`
/// on non-Linux platforms or if `/proc/self/statm` cannot be read.
pub fn read_proc_statm_mb() -> (f64, f64, f64) {
    if let Ok(s) = std::fs::read_to_string("/proc/self/statm") {
        let mut parts = s.split_whitespace();
        let size_pages: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let rss_pages: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let _shared: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let _text: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let _lib: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let data_pages: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let page_kb = 4.0 / 1024.0;
        return (
            rss_pages as f64 * page_kb,
            size_pages as f64 * page_kb,
            data_pages as f64 * page_kb,
        );
    }
    (0.0, 0.0, 0.0)
}

/// Reads `(vm_rss_mb, vm_hwm_mb, vm_data_mb)` from `/proc/self/status`.
///
/// Extracts `VmRSS:`, `VmHWM:` (kernel-tracked peak RSS high-water mark), and `VmData:` lines
/// reported in kilobytes (KiB) by the Linux kernel and converts them to megabytes (MiB). Falls
/// back to [`read_proc_statm_mb`] if `/proc/self/status` is unavailable.
///
/// # Returns
///
/// A 3-tuple `(vm_rss_mb, vm_hwm_mb, vm_data_mb)` in megabytes (`MB`).
pub fn read_proc_status_mb() -> (f64, f64, f64) {
    let (statm_rss, _, statm_data) = read_proc_statm_mb();
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return (statm_rss, statm_rss, statm_data);
    };
    let mut rss_kb = 0u64;
    let mut hwm_kb = 0u64;
    let mut data_kb = 0u64;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss_kb = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            hwm_kb = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        } else if let Some(rest) = line.strip_prefix("VmData:") {
            data_kb = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        }
    }
    (
        rss_kb as f64 / 1024.0,
        hwm_kb as f64 / 1024.0,
        data_kb as f64 / 1024.0,
    )
}

/// Retrieves or initializes the global [`MEM_STATE`] mutex singleton.
///
/// On first invocation, reads `VULCANS_MEM_LOG` (defaulting to [`DEFAULT_MEM_LOG_PATH`]) and
/// `VULCANS_MAX_RSS_MB` (defaulting to [`DEFAULT_MAX_RSS_MB`], clamped to $\ge 256.0\text{ MB}$),
/// creates/truncates the log file, and writes the initial header and `phase="init"` entry.
///
/// # Returns
///
/// A `'static` reference to the [`Mutex<MemLoggerState>`] singleton.
///
/// # Side Effects
///
/// Creates or truncates the log file on disk and writes the header banner on first call.
fn get_state() -> &'static Mutex<MemLoggerState> {
    MEM_STATE.get_or_init(|| {
        let log_path = std::env::var("VULCANS_MEM_LOG")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_MEM_LOG_PATH.to_string());
        let max_rss_mb = std::env::var("VULCANS_MAX_RSS_MB")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|&v| v >= 256.0)
            .unwrap_or(DEFAULT_MAX_RSS_MB);
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&log_path)
            .ok();
        let (rss, hwm, data) = read_proc_status_mb();
        if let Some(ref mut f) = file {
            let _ = writeln!(
                f,
                "=== Vulcan's Trials Memory & Phase Log (pid={}, max_rss_limit={:.0} MB) ===",
                std::process::id(),
                max_rss_mb
            );
            let _ = writeln!(
                f,
                "[+0.000s] RSS={:7.1} MB (Δ=  +0.0 MB, HWM={:7.1} MB, Data={:7.1} MB) | phase=\"init\"",
                rss, hwm, data
            );
            let _ = f.flush();
        }
        Mutex::new(MemLoggerState {
            start_instant: Instant::now(),
            log_path,
            file,
            last_rss_mb: rss,
            peak_rss_mb: rss.max(hwm),
            current_phase: "init".to_string(),
            max_rss_mb,
        })
    })
}

/// Ensures the background 20ms memory watchdog thread is running and resets the log for a new run.
///
/// # Arguments
///
/// * `custom_path` - Optional override file path for the memory log. If `Some(p)` is non-empty and
///   differs from the current log path, a new truncated log file is opened at `p`.
///
/// # Side Effects
///
/// * Opens/truncates the target log file if `custom_path` changes the destination.
/// * Writes an initialization entry to the log file and flushes it.
/// * Spawns the detached `vulcan-mem-watchdog` thread (if not already running), which polls
///   [`read_proc_statm_mb`] every 20 ms, logs `[WATCHDOG-SPIKE]` events whenever RSS jumps by
///   $\ge 48.0\text{ MB}$, and aborts the process with `std::process::exit(99)` if RSS reaches
///   `max_rss_mb`.
pub fn init_memlog(custom_path: Option<&str>) {
    let state_mtx = get_state();
    if let Ok(mut st) = state_mtx.lock() {
        if let Some(p) = custom_path {
            if !p.is_empty() && p != st.log_path {
                st.log_path = p.to_string();
                st.file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&st.log_path)
                    .ok();
            }
        }
        let (rss, hwm, data) = read_proc_status_mb();
        st.last_rss_mb = rss;
        st.peak_rss_mb = st.peak_rss_mb.max(rss).max(hwm);
        let elapsed = st.start_instant.elapsed().as_secs_f64();
        let max_rss = st.max_rss_mb;
        let path_disp = st.log_path.clone();
        if let Some(ref mut f) = st.file {
            let _ = writeln!(
                f,
                "[+{:7.3}s] RSS={:7.1} MB (HWM={:7.1} MB, Data={:7.1} MB) | memlog initialized at '{}' (limit={:.0} MB)",
                elapsed, rss, hwm, data, path_disp, max_rss
            );
            let _ = f.flush();
        }
    }

    if !WATCHDOG_STARTED.swap(true, Ordering::SeqCst) {
        let _ = std::thread::Builder::new()
            .name("vulcan-mem-watchdog".to_string())
            .spawn(|| {
                let mut prev_watchdog_rss = 0.0f64;
                loop {
                    std::thread::sleep(Duration::from_millis(20));
                    let (rss, _vmsize, data) = read_proc_statm_mb();
                    if rss <= 0.0 {
                        continue;
                    }
                    let spike = (rss - prev_watchdog_rss) >= 48.0;
                    if spike || rss >= 1024.0 {
                        if let Ok(mut st) = get_state().lock() {
                            let elapsed = st.start_instant.elapsed().as_secs_f64();
                            if rss > st.peak_rss_mb {
                                st.peak_rss_mb = rss;
                            }
                            let phase = st.current_phase.clone();
                            let max_rss = st.max_rss_mb;
                            if spike {
                                let delta = rss - prev_watchdog_rss;
                                prev_watchdog_rss = rss;
                                if let Some(ref mut f) = st.file {
                                    let _ = writeln!(
                                        f,
                                        "[+{:7.3}s] [WATCHDOG-SPIKE] RSS={:7.1} MB (Δ={:+7.1} MB, Data={:7.1} MB) | active_phase=\"{}\"",
                                        elapsed, rss, delta, data, phase
                                    );
                                    let _ = f.flush();
                                    let _ = f.sync_data();
                                }
                            }
                            if rss >= max_rss {
                                if let Some(ref mut f) = st.file {
                                    let _ = writeln!(
                                        f,
                                        "[+{:7.3}s] [WATCHDOG-OOM-ABORT] RSS={:7.1} MB (Data={:7.1} MB) exceeded safety limit {:.0} MB during active_phase=\"{}\" — aborting process to protect host!",
                                        elapsed, rss, data, max_rss, phase
                                    );
                                    let _ = f.flush();
                                    let _ = f.sync_all();
                                }
                                eprintln!(
                                    "\n[WATCHDOG-OOM-ABORT] RSS={:.1} MB exceeded safety limit {:.0} MB during phase '{}' (see {})",
                                    rss, max_rss, phase, st.log_path
                                );
                                std::process::exit(99);
                            }
                        }
                    } else if prev_watchdog_rss == 0.0 || rss < prev_watchdog_rss - 32.0 {
                        prev_watchdog_rss = rss;
                    }
                }
            });
    }
}

/// Updates the active phase label without writing a file line unless RSS jumped by >= 16 MB
/// or exceeded the safety limit. Suitable for high-frequency inner loops.
///
/// # Arguments
///
/// * `phase` - Short identifier describing the current inner-loop phase.
///
/// # Side Effects
///
/// * Updates `current_phase` and `peak_rss_mb` in `MEM_STATE`.
/// * Writes and flushes a log line if $|\Delta\text{RSS}| \ge 16.0\text{ MB}$ or `rss >= max_rss_mb`.
/// * Terminates the process via `std::process::exit(99)` after writing a forced stack backtrace
///   if `rss >= max_rss_mb`.
pub fn set_phase_fast(phase: &str) {
    let (rss, _, data) = read_proc_statm_mb();
    if let Ok(mut st) = get_state().lock() {
        st.current_phase.clear();
        st.current_phase.push_str(phase);
        if rss > st.peak_rss_mb {
            st.peak_rss_mb = rss;
        }
        let delta = rss - st.last_rss_mb;
        let max_rss = st.max_rss_mb;
        if delta.abs() >= 16.0 || rss >= max_rss {
            let elapsed = st.start_instant.elapsed().as_secs_f64();
            st.last_rss_mb = rss;
            let peak = st.peak_rss_mb;
            if let Some(ref mut f) = st.file {
                let _ = writeln!(
                    f,
                    "[+{:7.3}s] RSS={:7.1} MB (Δ={:+7.1} MB, Peak={:7.1} MB, Data={:7.1} MB) | phase=\"{}\"",
                    elapsed, rss, delta, peak, data, phase
                );
                let _ = f.flush();
                if delta >= 64.0 || rss >= max_rss {
                    let _ = f.sync_data();
                }
            }
            if rss >= max_rss {
                let bt = std::backtrace::Backtrace::force_capture();
                if let Some(ref mut f) = st.file {
                    let _ = writeln!(
                        f,
                        "[MEM-GUARD-ABORT] RSS={:.1} MB exceeded {:.0} MB at phase=\"{}\"\nBacktrace:\n{}",
                        rss, max_rss, phase, bt
                    );
                    let _ = f.flush();
                    let _ = f.sync_all();
                }
                eprintln!(
                    "\n[MEM-GUARD-ABORT] RSS={:.1} MB exceeded {:.0} MB at phase '{}'",
                    rss, max_rss, phase
                );
                std::process::exit(99);
            }
        }
    }
}

/// Logs an explicit memory checkpoint with arbitrary detail text and flushes immediately.
///
/// # Arguments
///
/// * `phase` - Name of the execution phase reaching this checkpoint.
/// * `details` - Optional supplementary key-value metrics or context string to append to the log line.
///
/// # Side Effects
///
/// * Lazily initializes the memory logger and watchdog via [`init_memlog`] if not already started.
/// * Reads `/proc/self/status`, updates `current_phase`, `last_rss_mb`, and `peak_rss_mb`, and
///   writes a formatted timestamped entry to the memory log file.
/// * Terminates the process with exit code `99` and a full backtrace if `rss >= max_rss_mb`.
pub fn checkpoint(phase: &str, details: &str) {
    if !WATCHDOG_STARTED.load(Ordering::Relaxed) {
        init_memlog(None);
    }
    let (rss, hwm, data) = read_proc_status_mb();
    if let Ok(mut st) = get_state().lock() {
        st.current_phase = phase.to_string();
        if rss > st.peak_rss_mb {
            st.peak_rss_mb = rss;
        }
        if hwm > st.peak_rss_mb {
            st.peak_rss_mb = hwm;
        }
        let delta = rss - st.last_rss_mb;
        st.last_rss_mb = rss;
        let elapsed = st.start_instant.elapsed().as_secs_f64();
        let peak = st.peak_rss_mb;
        let max_rss = st.max_rss_mb;
        if let Some(ref mut f) = st.file {
            if details.is_empty() {
                let _ = writeln!(
                    f,
                    "[+{:7.3}s] RSS={:7.1} MB (Δ={:+7.1} MB, HWM={:7.1} MB, Data={:7.1} MB) | phase=\"{}\"",
                    elapsed, rss, delta, peak, data, phase
                );
            } else {
                let _ = writeln!(
                    f,
                    "[+{:7.3}s] RSS={:7.1} MB (Δ={:+7.1} MB, HWM={:7.1} MB, Data={:7.1} MB) | phase=\"{}\" | {}",
                    elapsed, rss, delta, peak, data, phase, details
                );
            }
            let _ = f.flush();
            if delta.abs() >= 16.0 || rss >= 512.0 {
                let _ = f.sync_data();
            }
        }
        if rss >= max_rss {
            let bt = std::backtrace::Backtrace::force_capture();
            if let Some(ref mut f) = st.file {
                let _ = writeln!(
                    f,
                    "[MEM-GUARD-ABORT] RSS={:.1} MB exceeded {:.0} MB at phase=\"{}\" ({})\nBacktrace:\n{}",
                    rss, max_rss, phase, details, bt
                );
                let _ = f.flush();
                let _ = f.sync_all();
            }
            eprintln!(
                "\n[MEM-GUARD-ABORT] RSS={:.1} MB exceeded {:.0} MB at phase '{}' ({})",
                rss, max_rss, phase, details
            );
            std::process::exit(99);
        }
    }
}

/// Logs an explicit memory checkpoint along with current board copper counts (`tracks`, `arcs`, `vias`).
///
/// # Arguments
///
/// * `phase` - Name of the execution phase reaching this checkpoint.
/// * `board` - Reference to the [`Board`] whose `tracks`, `arcs`, and `vias` counts are formatted into the log entry.
///
/// # Side Effects
///
/// Delegates to [`checkpoint`], writing to the memory log file and enforcing the OOM safety limit.
pub fn checkpoint_board(phase: &str, board: &Board) {
    let details = format!(
        "tracks={} arcs={} vias={}",
        board.tracks.len(),
        board.arcs.len(),
        board.vias.len()
    );
    checkpoint(phase, &details);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_proc_statm_and_checkpoint_logging() {
        let (rss, vmsize, _data) = read_proc_statm_mb();
        #[cfg(target_os = "linux")]
        {
            assert!(rss > 0.0, "Process RSS in MB must be positive on Linux");
            assert!(vmsize >= rss, "Virtual memory size must be >= RSS");
        }

        let tmp_log = "/tmp/vulcans_trials_mem_unit_test.log";
        init_memlog(Some(tmp_log));
        let board = Board::default();
        checkpoint_board("unit_test_phase_1", &board);
        set_phase_fast("unit_test_fast_phase_2");
        checkpoint("unit_test_phase_3", "custom_detail=42");

        let content = std::fs::read_to_string(tmp_log).expect("Memory log file should be written");
        assert!(content.contains("unit_test_phase_1"));
        assert!(content.contains("tracks=0 arcs=0 vias=0"));
        assert!(content.contains("unit_test_phase_3"));
        assert!(content.contains("custom_detail=42"));
        let _ = std::fs::remove_file(tmp_log);
    }
}
