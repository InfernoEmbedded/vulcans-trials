// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

//! KiCad 8/9/10 Inter-Process Communication (IPC) client and Protocol Buffers wire engine.
//!
//! This module provides real-time and batch synchronization between Vulcan's Trials and a
//! running KiCad PCBNew session over Unix domain sockets. It supports two transport modes:
//!
//! 1. **Native KiCad 9/10 NNG + Protobuf (`kiapi.*`) IPC:** Dynamically loads `libnng.so.1`
//!    (`Req0` socket dialect) to communicate with KiCad's native API server (typically at
//!    `/tmp/kicad/api.sock`). Encodes and decodes `kiapi.common.*` and `kiapi.board.*`
//!    Protocol Buffers wire messages directly without external codegen dependencies.
//! 2. **Line-Delimited JSON over `UnixStream`:** Used for lightweight streaming telemetry,
//!    60 FPS (~16 ms) rate-limited viewport updates during Simulated Annealing placement
//!    and PathFinder autorouting, and deterministic mock socket integration tests.
//!
//! # Synchronization Capabilities
//! - **Socket Discovery:** Automatically probes `$KICAD_IPC_PATH`, `$KICAD_IPC_SOCKET`,
//!   `$KICAD_API_SOCKET`, `/tmp/kicad/api.sock`, `$XDG_RUNTIME_DIR/kicad/`, and `/tmp/`.
//! - **Board Push/Pull Sync:** Synchronizes saved `.kicad_pcb` and Specctra `.ses` state into
//!   KiCad via atomic `RevertDocument` reload or transactional `BeginCommit` / `EndCommit`
//!   batches (`FlipItems`, `UpdateItems`, `DeleteItems`, `CreateItems`, `RefillZones`, and
//!   `RefreshEditor`).
//! - **Footprint & Group Coordinate Transforms:** Applies rigid-body rotation and translation
//!   in nanometers (`nm`) and degrees (`deg`) across `FootprintInstance` protobuf trees,
//!   including child fields, pads, pad net reassignments from pin/gate swapping, graphic shapes,
//!   polygons, and keepout/copper zones.

use crate::board::{Board, Footprint, Point};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::net::UnixStream;

/// KiCad 10 native IPC protocol command & message types.
/// Serialized as line-delimited JSON objects over Unix domain socket or named pipe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IpcMessage {
    /// Handshake initiating session between Vulcan's Trials and KiCad 10.
    Handshake {
        /// Human-readable client application name (`"Vulcan's Trials"`).
        client: String,
        /// Semantic version string of the client crate (`CARGO_PKG_VERSION`).
        version: String,
    },
    /// Live displacement update for an existing footprint.
    MoveFootprint {
        /// Component reference designator (e.g., `"U1"`, `"R12"`).
        reference: String,
        /// Target X coordinate in millimeters (`mm`).
        x: f64,
        /// Target Y coordinate in millimeters (`mm`).
        y: f64,
        /// Target orientation angle in degrees (`deg`).
        rotation: f64,
    },
    /// Direct addition of a routed copper track segment.
    AddTrack {
        /// Track start `(x, y)` coordinates in millimeters (`mm`).
        start: (f64, f64),
        /// Track end `(x, y)` coordinates in millimeters (`mm`).
        end: (f64, f64),
        /// Copper trace width in millimeters (`mm`).
        width: f64,
        /// KiCad copper layer name (e.g., `"F.Cu"`, `"In1.Cu"`, `"B.Cu"`).
        layer: String,
        /// Internal board net identifier (`0` for unconnected).
        net_id: usize,
    },
    /// Direct addition of a routed via.
    AddVia {
        /// Via center `(x, y)` coordinates in millimeters (`mm`).
        position: (f64, f64),
        /// Overall copper annular ring outer diameter in millimeters (`mm`).
        size: f64,
        /// Finished drill hole diameter in millimeters (`mm`).
        drill: f64,
        /// Internal board net identifier (`0` for unconnected).
        net_id: usize,
    },
    /// Clear all dynamic autorouted tracks and vias before rerouting pass.
    ClearTracks,
    /// Atomically flush pending canvas mutations and trigger 60 FPS viewport repaint.
    CommitBatch,
    /// Pull and synchronize the saved board (`.kicad_pcb` and `.ses`) into KiCad.
    PullBoard {
        /// Filesystem path to the saved `.kicad_pcb` board file.
        pcb_path: String,
        /// Filesystem path to the exported Specctra `.ses` session file.
        ses_path: String,
    },
}

/// Type alias for [`IpcMessage`] representing an outgoing IPC command payload.
pub type IpcCommand = IpcMessage;

/// Client connection managing the KiCad 10 IPC socket connection, stream buffering,
/// delta calculation, and ~60 FPS rate-limited throttling.
pub struct KiCadIpcClient {
    /// Underlying Unix domain stream socket for line-delimited JSON IPC connections.
    #[cfg(unix)]
    stream: Option<UnixStream>,
    /// Native KiCad 9/10 NNG (`libnng.so.1`) + Protocol Buffers client handle, if connected to an NNG endpoint.
    nng: Option<NngKiCadClient>,
    /// Normalized filesystem path to the connected Unix domain socket.
    socket_path: Option<PathBuf>,
    /// Whether the client currently holds an active, healthy socket connection.
    is_connected: bool,

    // Throttling configuration (default 16ms for ~60 FPS viewport refresh)
    /// Minimum time interval between consecutive batch flushes (defaults to 16 ms for ~60 FPS).
    throttle_interval: Duration,
    /// Timestamp of the most recent batch commit or board push.
    last_flush_time: Instant,

    // Footprint displacement coalescing buffer: reference -> (x, y, rotation)
    /// Coalescing buffer mapping footprint reference designator to latest pending `(x_mm, y_mm, rotation_deg)`.
    pending_moves: HashMap<String, (f64, f64, f64)>,
    // Last sent positions to suppress redundant sub-millimeter noise
    /// Cache of last transmitted `(x_mm, y_mm, rotation_deg)` per footprint reference to filter sub-5-micron noise.
    last_sent_moves: HashMap<String, (f64, f64, f64)>,

    // Queue of pending copper mutations
    /// Queued [`IpcMessage::AddTrack`] commands awaiting the next batch commit.
    pending_tracks: Vec<IpcMessage>,
    /// Queued [`IpcMessage::AddVia`] commands awaiting the next batch commit.
    pending_vias: Vec<IpcMessage>,

    // Telemetry & metrics
    /// Cumulative count of individual IPC messages or board items transmitted over the session.
    total_messages_sent: usize,
    /// Cumulative count of atomic [`IpcMessage::CommitBatch`] flushes completed.
    total_batches_committed: usize,
}

impl std::fmt::Debug for KiCadIpcClient {
    /// Formats the [`KiCadIpcClient`] connection status, queue lengths, and telemetry counters for debugging.
    ///
    /// # Arguments
    /// * `f` - The mutable formatter state to write diagnostic fields into.
    ///
    /// # Returns
    /// `std::fmt::Result` indicating whether formatting succeeded.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KiCadIpcClient")
            .field("socket_path", &self.socket_path)
            .field("is_connected", &self.is_connected)
            .field("is_nng", &self.nng.is_some())
            .field("throttle_interval", &self.throttle_interval)
            .field("pending_moves", &self.pending_moves.len())
            .field("pending_tracks", &self.pending_tracks.len())
            .field("pending_vias", &self.pending_vias.len())
            .field("total_messages_sent", &self.total_messages_sent)
            .field("total_batches_committed", &self.total_batches_committed)
            .finish()
    }
}

/// Returns true if `path` refers to a native KiCad 9/10 NNG IPC socket (such as `/tmp/kicad/api.sock`)
/// rather than a mock line-delimited JSON `UnixListener` test socket (`/tmp/kicad_ipc_test_*.sock`).
///
/// # Arguments
/// * `path` - Filesystem path to the candidate Unix domain socket.
///
/// # Returns
/// `true` if the filename or parent directory matches native KiCad NNG socket naming conventions; `false` otherwise.
pub fn is_native_kicad_nng_socket_path(path: &Path) -> bool {
    let fname = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let parent_name = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("");
    fname == "api.sock"
        || parent_name == "kicad"
        || fname.contains("_nng_")
        || fname == "kicad.sock"
}

impl KiCadIpcClient {
    /// Creates a new unconnected IPC client instance with default 60 FPS (16ms) throttling.
    ///
    /// # Returns
    /// An unconnected [`KiCadIpcClient`] with empty coalescing buffers and a 16 ms throttle interval.
    pub fn new() -> Self {
        Self {
            #[cfg(unix)]
            stream: None,
            nng: None,
            socket_path: None,
            is_connected: false,
            throttle_interval: Duration::from_millis(16),
            last_flush_time: Instant::now(),
            pending_moves: HashMap::new(),
            last_sent_moves: HashMap::new(),
            pending_tracks: Vec::new(),
            pending_vias: Vec::new(),
            total_messages_sent: 0,
            total_batches_committed: 0,
        }
    }

    /// Normalizes a socket path string, stripping `ipc://` or `ipc:` URI schemes.
    ///
    /// # Arguments
    /// * `path` - Raw socket path or URI string (e.g., `"ipc:///tmp/kicad/api.sock"` or `"/tmp/kicad/api.sock"`).
    ///
    /// # Returns
    /// A [`PathBuf`] containing the stripped filesystem path.
    pub fn normalize_socket_path(path: &str) -> PathBuf {
        let trimmed = path.trim();
        let stripped = if let Some(s) = trimmed.strip_prefix("ipc://") {
            s
        } else if let Some(s) = trimmed.strip_prefix("ipc:") {
            s
        } else {
            trimmed
        };
        PathBuf::from(stripped)
    }

    /// Connects to a specific Unix domain socket path and performs initial handshake.
    /// Uses native NNG Req0 + Protobuf (`kiapi.*`) for KiCad 9/10 sockets (`/tmp/kicad/api.sock`),
    /// and line-delimited JSON over `UnixStream` for mock test sockets.
    ///
    /// # Arguments
    /// * `path` - Target Unix domain socket path or `ipc://` URI.
    ///
    /// # Returns
    /// Connected [`KiCadIpcClient`] ready for streaming or board synchronization.
    ///
    /// # Side Effects
    /// Opens a Unix domain socket or NNG `Req0` dialer and transmits an initial handshake or `Ping` message.
    ///
    /// # Errors
    /// Returns `std::io::Error` if the socket does not exist, connection is refused, NNG library loading fails,
    /// or the initial handshake/ping times out.
    pub fn connect<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let p = Self::normalize_socket_path(&path.as_ref().to_string_lossy());
        if is_native_kicad_nng_socket_path(&p) {
            let nng = NngKiCadClient::connect(&p)?;
            return Ok(Self {
                #[cfg(unix)]
                stream: None,
                nng: Some(nng),
                socket_path: Some(p),
                is_connected: true,
                throttle_interval: Duration::from_millis(16),
                last_flush_time: Instant::now(),
                pending_moves: HashMap::new(),
                last_sent_moves: HashMap::new(),
                pending_tracks: Vec::new(),
                pending_vias: Vec::new(),
                total_messages_sent: 1,
                total_batches_committed: 0,
            });
        }

        #[cfg(unix)]
        {
            let stream = UnixStream::connect(&p)?;
            stream.set_write_timeout(Some(Duration::from_millis(250)))?;
            stream.set_read_timeout(Some(Duration::from_millis(250)))?;

            let mut client = Self {
                stream: Some(stream),
                nng: None,
                socket_path: Some(p),
                is_connected: true,
                throttle_interval: Duration::from_millis(16),
                last_flush_time: Instant::now(),
                pending_moves: HashMap::new(),
                last_sent_moves: HashMap::new(),
                pending_tracks: Vec::new(),
                pending_vias: Vec::new(),
                total_messages_sent: 0,
                total_batches_committed: 0,
            };

            // Send handshake
            client.send_handshake()?;
            Ok(client)
        }
        #[cfg(not(unix))]
        {
            let _ = p;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "KiCad IPC domain sockets are supported on Unix platforms",
            ))
        }
    }

    /// Attempts to connect to a custom path if given, or automatically discovers an active KiCad 10 IPC socket.
    ///
    /// # Arguments
    /// * `custom_path` - Optional user-specified socket path or URI override.
    ///
    /// # Returns
    /// `Some(KiCadIpcClient)` if connection succeeded, or `None` if no reachable socket was found.
    ///
    /// # Side Effects
    /// Performs socket discovery I/O and prints connection status or diagnostic errors to stdout/stderr.
    ///
    /// # Errors
    /// Returns `None` if both the custom path (when provided) and auto-discovered paths fail to connect.
    pub fn connect_or_discover(custom_path: Option<&str>) -> Option<Self> {
        Self::connect_or_discover_inner(custom_path, false)
    }

    /// Quietly attempts to connect to a custom path or discover an active KiCad 10 IPC socket without logging to stderr on retry failure.
    ///
    /// # Arguments
    /// * `custom_path` - Optional user-specified socket path or URI override.
    ///
    /// # Returns
    /// `Some(KiCadIpcClient)` if connection succeeded, or `None` if no reachable socket was found.
    ///
    /// # Side Effects
    /// Performs socket discovery and connection I/O silently without printing to stdout or stderr.
    ///
    /// # Errors
    /// Returns `None` if no active KiCad IPC socket can be found or connected.
    pub fn connect_or_discover_quiet(custom_path: Option<&str>) -> Option<Self> {
        Self::connect_or_discover_inner(custom_path, true)
    }

    /// Internal implementation of socket connection and fallback discovery with configurable logging verbosity.
    ///
    /// # Arguments
    /// * `custom_path` - Optional explicit socket path to try before falling back to discovery (if it was the default `/tmp/kicad/api.sock`).
    /// * `quiet` - When `true`, suppresses stdout and stderr connection diagnostics.
    ///
    /// # Returns
    /// `Some(KiCadIpcClient)` when a connection is established, or `None` on failure.
    ///
    /// # Side Effects
    /// Connects to a Unix socket and optionally writes status logs to stdout/stderr.
    ///
    /// # Errors
    /// Returns `None` if the explicit non-default socket fails or if auto-discovery finds no responsive socket.
    fn connect_or_discover_inner(custom_path: Option<&str>, quiet: bool) -> Option<Self> {
        if let Some(cp) = custom_path {
            let norm = Self::normalize_socket_path(cp);
            if !norm.as_os_str().is_empty() {
                match Self::connect(&norm) {
                    Ok(client) => return Some(client),
                    Err(e) => {
                        if !quiet {
                            eprintln!(
                                "[KiCad-IPC] Failed to connect to specified socket '{}': {}",
                                norm.display(),
                                e
                            );
                        }
                        if norm != Path::new("/tmp/kicad/api.sock") {
                            return None;
                        }
                    }
                }
            }
        }

        if let Some(discovered) = Self::discover_socket_path() {
            match Self::connect(&discovered) {
                Ok(client) => {
                    if !quiet {
                        println!(
                            "[KiCad-IPC] Connected to active KiCad 10 socket at {}",
                            discovered.display()
                        );
                    }
                    Some(client)
                }
                Err(e) => {
                    if !quiet {
                        eprintln!(
                            "[KiCad-IPC] Failed to connect to discovered socket '{}': {}",
                            discovered.display(),
                            e
                        );
                    }
                    None
                }
            }
        } else {
            None
        }
    }

    /// Discovers active KiCad 10 IPC Unix socket paths on Linux/macOS.
    /// Checks:
    /// 1. `$KICAD_IPC_PATH`, `$KICAD_IPC_SOCKET`, or `$KICAD_API_SOCKET` env vars (supports `ipc://` prefix)
    /// 2. Well-known KiCad 9/10 socket paths (`/tmp/kicad/api.sock`, `/tmp/kicad.sock`)
    /// 3. `$XDG_RUNTIME_DIR/kicad/`
    /// 4. `/tmp/kicad/`
    /// 5. `/tmp/`
    ///
    /// # Returns
    /// `Some(PathBuf)` pointing to the first existing KiCad IPC socket discovered, or `None` if none exist.
    ///
    /// # Side Effects
    /// Reads process environment variables and queries filesystem metadata in `/tmp` and `$XDG_RUNTIME_DIR`.
    ///
    /// # Errors
    /// Returns `None` if no matching socket file exists on the filesystem.
    pub fn discover_socket_path() -> Option<PathBuf> {
        // 1. Environment variable override
        if let Ok(env_path) = env::var("KICAD_IPC_PATH")
            .or_else(|_| env::var("KICAD_IPC_SOCKET"))
            .or_else(|_| env::var("KICAD_API_SOCKET"))
        {
            let p = Self::normalize_socket_path(&env_path);
            if p.exists() {
                return Some(p);
            }
        }

        // 2. Direct well-known KiCad 9/10 socket paths
        let direct_candidates = [
            "/tmp/kicad/api.sock",
            "/tmp/kicad.sock",
            "/tmp/kicad/kicad.sock",
        ];
        for candidate in &direct_candidates {
            let p = Path::new(candidate);
            if p.exists() {
                return Some(p.to_path_buf());
            }
        }

        // 3. $XDG_RUNTIME_DIR/kicad/
        if let Ok(runtime_dir) = env::var("XDG_RUNTIME_DIR") {
            let kicad_dir = Path::new(&runtime_dir).join("kicad");
            if let Some(found) = scan_dir_for_socket(&kicad_dir) {
                return Some(found);
            }
        }

        // 4. /tmp/kicad/
        if let Some(found) = scan_dir_for_socket(Path::new("/tmp/kicad")) {
            return Some(found);
        }

        // 5. /tmp/
        if let Some(found) = scan_dir_for_socket(Path::new("/tmp")) {
            return Some(found);
        }

        None
    }

    /// Returns whether the client is currently connected.
    ///
    /// # Returns
    /// `true` if the client has an active NNG or `UnixStream` socket connection; `false` otherwise.
    pub fn is_connected(&self) -> bool {
        self.is_connected
    }

    /// Returns the socket path connected to, if any.
    ///
    /// # Returns
    /// `Some(&Path)` referencing the connected Unix domain socket path, or `None` if unconnected.
    pub fn socket_path(&self) -> Option<&Path> {
        self.socket_path.as_deref()
    }

    /// Sets the minimum throttling interval between flushes (e.g. 16.6ms for 60 FPS).
    ///
    /// # Arguments
    /// * `interval` - Minimum [`Duration`] required between automatic streaming batch commits.
    ///
    /// # Side Effects
    /// Updates `self.throttle_interval`.
    pub fn set_throttle_interval(&mut self, interval: Duration) {
        self.throttle_interval = interval;
    }

    /// Returns current throttling interval.
    ///
    /// # Returns
    /// The configured minimum [`Duration`] between streaming flushes.
    pub fn throttle_interval(&self) -> Duration {
        self.throttle_interval
    }

    /// Total messages sent and total batches committed.
    ///
    /// # Returns
    /// A tuple `(total_messages_sent, total_batches_committed)` for telemetry reporting.
    pub fn stats(&self) -> (usize, usize) {
        (self.total_messages_sent, self.total_batches_committed)
    }

    /// Sends the initial Handshake message.
    ///
    /// # Returns
    /// `Ok(())` when the handshake (or NNG `Ping`) succeeds.
    ///
    /// # Side Effects
    /// Writes a `Ping` protobuf request (NNG mode) or `IpcMessage::Handshake` JSON line (`UnixStream` mode)
    /// to the socket and increments `self.total_messages_sent`.
    ///
    /// # Errors
    /// Returns `std::io::Error` if the socket write or NNG request fails.
    pub fn send_handshake(&mut self) -> std::io::Result<()> {
        if let Some(ref mut nng) = self.nng {
            nng.ping()?;
            self.total_messages_sent += 1;
            return Ok(());
        }
        let msg = IpcMessage::Handshake {
            client: "Vulcan's Trials".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        self.send_raw_message(&msg)
    }

    /// Directly serializes and writes an IPC message as a single line JSON over the socket.
    ///
    /// # Arguments
    /// * `msg` - Reference to the [`IpcMessage`] command to serialize and transmit.
    ///
    /// # Returns
    /// `Ok(())` on successful transmission.
    ///
    /// # Side Effects
    /// Writes newline-delimited JSON to `self.stream` and increments `self.total_messages_sent`.
    /// If a socket write error occurs, marks `self.is_connected = false` and drops `self.stream`.
    ///
    /// # Errors
    /// Returns `std::io::Error` if JSON serialization fails, the stream is disconnected, or socket I/O fails.
    fn send_raw_message(&mut self, msg: &IpcMessage) -> std::io::Result<()> {
        if self.nng.is_some() {
            self.total_messages_sent += 1;
            return Ok(());
        }
        #[cfg(unix)]
        {
            if let Some(ref mut stream) = self.stream {
                let json = serde_json::to_string(msg).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                })?;
                if let Err(e) = stream.write_all(json.as_bytes()).and_then(|_| stream.write_all(b"\n")) {
                    self.is_connected = false;
                    self.stream = None;
                    return Err(e);
                }
                self.total_messages_sent += 1;
                Ok(())
            } else {
                self.is_connected = false;
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    "KiCad IPC stream not connected",
                ))
            }
        }
        #[cfg(not(unix))]
        {
            let _ = msg;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "Unix domain sockets required",
            ))
        }
    }

    /// Queues a footprint displacement in the coalescing buffer.
    ///
    /// # Arguments
    /// * `reference` - Component reference designator (e.g., `"U1"`).
    /// * `x` - Target X coordinate in millimeters (`mm`).
    /// * `y` - Target Y coordinate in millimeters (`mm`).
    /// * `rotation` - Target rotation angle in degrees (`deg`).
    ///
    /// # Side Effects
    /// Inserts or overwrites the latest pending placement for `reference` in `self.pending_moves`.
    pub fn queue_move_footprint(&mut self, reference: impl Into<String>, x: f64, y: f64, rotation: f64) {
        self.pending_moves.insert(reference.into(), (x, y, rotation));
    }

    /// Queues a track segment creation.
    ///
    /// # Arguments
    /// * `start` - Start `(x, y)` coordinates in millimeters (`mm`).
    /// * `end` - End `(x, y)` coordinates in millimeters (`mm`).
    /// * `width` - Trace width in millimeters (`mm`).
    /// * `layer` - KiCad copper layer name (e.g., `"F.Cu"`).
    /// * `net_id` - Internal board net ID.
    ///
    /// # Side Effects
    /// Appends an [`IpcMessage::AddTrack`] entry to `self.pending_tracks`.
    pub fn queue_add_track(
        &mut self,
        start: (f64, f64),
        end: (f64, f64),
        width: f64,
        layer: impl Into<String>,
        net_id: usize,
    ) {
        self.pending_tracks.push(IpcMessage::AddTrack {
            start,
            end,
            width,
            layer: layer.into(),
            net_id,
        });
    }

    /// Queues a via creation.
    ///
    /// # Arguments
    /// * `position` - Via center `(x, y)` coordinates in millimeters (`mm`).
    /// * `size` - Via outer copper diameter in millimeters (`mm`).
    /// * `drill` - Via drill hole diameter in millimeters (`mm`).
    /// * `net_id` - Internal board net ID.
    ///
    /// # Side Effects
    /// Appends an [`IpcMessage::AddVia`] entry to `self.pending_vias`.
    pub fn queue_add_via(&mut self, position: (f64, f64), size: f64, drill: f64, net_id: usize) {
        self.pending_vias.push(IpcMessage::AddVia {
            position,
            size,
            drill,
            net_id,
        });
    }

    /// Sends ClearTracks command to clear autorouted copper before rerouting.
    ///
    /// # Returns
    /// `Ok(())` when the command is sent or skipped (when unconnected or using transactional NNG sync).
    ///
    /// # Side Effects
    /// Sends [`IpcMessage::ClearTracks`] over the `UnixStream` socket if connected in JSON stream mode.
    ///
    /// # Errors
    /// Returns `std::io::Error` if writing to the underlying `UnixStream` fails.
    pub fn clear_tracks(&mut self) -> std::io::Result<()> {
        if !self.is_connected {
            return Ok(());
        }
        if self.nng.is_some() {
            return Ok(());
        }
        self.send_raw_message(&IpcMessage::ClearTracks)
    }

    /// Stream footprint move with delta threshold check and 60 FPS interval throttling.
    /// Returns true if a batch flush was triggered.
    ///
    /// # Arguments
    /// * `reference` - Component reference designator.
    /// * `x` - New X coordinate in millimeters (`mm`).
    /// * `y` - New Y coordinate in millimeters (`mm`).
    /// * `rotation` - New orientation angle in degrees (`deg`).
    /// * `force_flush` - If `true`, flushes immediately regardless of `self.throttle_interval`.
    ///
    /// # Returns
    /// `Ok(true)` if a batch commit was flushed to the socket, or `Ok(false)` if coalesced without flushing.
    ///
    /// # Side Effects
    /// Queues the footprint move if displacement exceeds `0.005 mm` or `0.05 deg`, and flushes pending moves
    /// via [`Self::commit_batch`] when the throttle interval has elapsed or `force_flush` is set.
    ///
    /// # Errors
    /// Returns `std::io::Error` if flushing the batch to the socket fails.
    pub fn stream_footprint_move(
        &mut self,
        reference: &str,
        x: f64,
        y: f64,
        rotation: f64,
        force_flush: bool,
    ) -> std::io::Result<bool> {
        let is_delta_significant = if let Some(&(lx, ly, lr)) = self.last_sent_moves.get(reference) {
            (x - lx).hypot(y - ly) > 0.005 || (rotation - lr).abs() > 0.05
        } else {
            true
        };

        if is_delta_significant {
            self.queue_move_footprint(reference, x, y, rotation);
        }

        if force_flush || (self.last_flush_time.elapsed() >= self.throttle_interval && !self.pending_moves.is_empty()) {
            self.commit_batch()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Batch stream displacement deltas for footprints during Simulated Annealing.
    /// Only footprints that have significantly moved since their last sent coordinates are queued.
    /// Flushes to IPC if throttle interval (~16ms) has elapsed.
    ///
    /// # Arguments
    /// * `footprints` - Slice of board footprints being placed.
    /// * `positions` - Parallel slice of candidate `(x, y)` coordinates in millimeters (`mm`).
    /// * `rotations` - Parallel slice of candidate orientation angles in degrees (`deg`).
    ///
    /// # Returns
    /// Number of footprints whose movement exceeded the delta threshold and were queued.
    ///
    /// # Side Effects
    /// Mutates `self.pending_moves` and, if `self.throttle_interval` has elapsed, flushes the batch over IPC.
    pub fn stream_footprint_deltas(
        &mut self,
        footprints: &[Footprint],
        positions: &[Point],
        rotations: &[f64],
    ) -> usize {
        if !self.is_connected || footprints.is_empty() {
            return 0;
        }

        let mut queued = 0;
        for (i, fp) in footprints.iter().enumerate() {
            if i >= positions.len() || i >= rotations.len() {
                break;
            }
            let cur_x = positions[i].x;
            let cur_y = positions[i].y;
            let cur_rot = rotations[i];

            let significant = if let Some(&(lx, ly, lr)) = self.last_sent_moves.get(&fp.reference) {
                (cur_x - lx).hypot(cur_y - ly) > 0.005 || (cur_rot - lr).abs() > 0.05
            } else {
                true
            };

            if significant {
                self.queue_move_footprint(&fp.reference, cur_x, cur_y, cur_rot);
                queued += 1;
            }
        }

        if self.last_flush_time.elapsed() >= self.throttle_interval && !self.pending_moves.is_empty() {
            let _ = self.commit_batch();
        }

        queued
    }

    /// Stream all current board tracks and vias to KiCad IPC canvas.
    /// Queues AddTrack and AddVia commands and emits CommitBatch.
    ///
    /// # Arguments
    /// * `board` - Reference to the [`Board`] whose routed tracks, discretized arcs, and vias should be streamed.
    ///
    /// # Returns
    /// `Ok(count)` with the total number of copper segments and vias queued and committed (`0` if throttled or unconnected).
    ///
    /// # Side Effects
    /// Populates `self.pending_tracks` and `self.pending_vias` and flushes them via [`Self::commit_batch`].
    ///
    /// # Errors
    /// Returns `std::io::Error` if committing the batch over the IPC socket fails.
    pub fn stream_tracks_and_vias(&mut self, board: &Board) -> std::io::Result<usize> {
        if !self.is_connected {
            return Ok(0);
        }

        // Throttle updates unless 16ms has elapsed
        if self.last_flush_time.elapsed() < self.throttle_interval {
            return Ok(0);
        }

        let mut count = 0;
        for trk in &board.tracks {
            let layer_name = board.layer_name(trk.layer);
            self.queue_add_track(trk.start, trk.end, trk.width, layer_name, trk.net_id);
            count += 1;
        }

        for arc in &board.arcs {
            let layer_name = board.layer_name(arc.layer);
            for seg in arc.to_segments(12) {
                self.queue_add_track(seg.start, seg.end, seg.width, layer_name, seg.net_id);
                count += 1;
            }
        }

        for via in &board.vias {
            self.queue_add_via(via.position, via.size, via.drill, via.net_id);
            count += 1;
        }

        self.commit_batch()?;
        Ok(count)
    }

    /// Pulls the full saved board (footprints, tracks, arcs, vias, and zone refills)
    /// into KiCad over the active IPC socket, bypassing throttle intervals.
    ///
    /// # Arguments
    /// * `board` - Reference to the synchronized [`Board`] state.
    /// * `pcb_path` - Path to the saved `.kicad_pcb` file on disk.
    /// * `ses_path` - Path to the exported Specctra `.ses` file on disk.
    ///
    /// # Returns
    /// `Ok(count)` with the number of board items synchronized into KiCad.
    ///
    /// # Side Effects
    /// - In NNG mode: Invokes `NngKiCadClient::push_board_state` to revert or transactionally update
    ///   footprints, tracks, arcs, vias, and copper zones in the live KiCad 10 editor.
    /// - In JSON stream mode: Sends `ClearTracks`, queues all footprints/tracks/arcs/vias, commits the batch,
    ///   and transmits [`IpcMessage::PullBoard`].
    /// - Marks `self.is_connected = false` if an unrecoverable socket error occurs.
    ///
    /// # Errors
    /// Returns `std::io::Error` if the client is disconnected or if socket transmission / KiCad RPC fails.
    pub fn pull_board_into_kicad(
        &mut self,
        board: &Board,
        pcb_path: &Path,
        ses_path: &Path,
    ) -> std::io::Result<usize> {
        if !self.is_connected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "KiCad IPC stream not connected",
            ));
        }

        if let Some(ref mut nng) = self.nng {
            self.pending_moves.clear();
            self.pending_tracks.clear();
            self.pending_vias.clear();
            match nng.push_board_state(board, pcb_path) {
                Ok(synced) => {
                    self.total_messages_sent += synced;
                    self.total_batches_committed += 1;
                    self.last_flush_time = Instant::now();
                    return Ok(synced);
                }
                Err(e) => {
                    self.is_connected = false;
                    self.nng = None;
                    return Err(e);
                }
            }
        }

        self.clear_tracks()?;

        for fp in &board.footprints {
            self.queue_move_footprint(&fp.reference, fp.x, fp.y, fp.rotation);
        }

        for trk in &board.tracks {
            let layer_name = board.layer_name(trk.layer);
            self.queue_add_track(trk.start, trk.end, trk.width, layer_name, trk.net_id);
        }

        for arc in &board.arcs {
            let layer_name = board.layer_name(arc.layer);
            for seg in arc.to_segments(12) {
                self.queue_add_track(seg.start, seg.end, seg.width, layer_name, seg.net_id);
            }
        }

        for via in &board.vias {
            self.queue_add_via(via.position, via.size, via.drill, via.net_id);
        }

        let committed = self.commit_batch()?;

        let pull_msg = IpcMessage::PullBoard {
            pcb_path: pcb_path.to_string_lossy().to_string(),
            ses_path: ses_path.to_string_lossy().to_string(),
        };
        self.send_raw_message(&pull_msg)?;

        #[cfg(unix)]
        {
            if let Some(ref mut stream) = self.stream {
                if let Err(e) = stream.flush() {
                    self.is_connected = false;
                    self.stream = None;
                    return Err(e);
                }
            }
        }

        Ok(committed)
    }

    /// Flushes all queued moves, tracks, and vias, then issues CommitBatch and flushes socket writer.
    ///
    /// # Returns
    /// `Ok(total_queued)` indicating the total number of queued items drained and committed.
    ///
    /// # Side Effects
    /// Drains `self.pending_moves`, `self.pending_tracks`, and `self.pending_vias`, updates
    /// `self.last_sent_moves`, writes messages and [`IpcMessage::CommitBatch`] to the socket stream,
    /// and updates `self.last_flush_time` and `self.total_batches_committed`.
    ///
    /// # Errors
    /// Returns `std::io::Error` if writing to or flushing the underlying Unix socket fails.
    pub fn commit_batch(&mut self) -> std::io::Result<usize> {
        let total_queued = self.pending_moves.len() + self.pending_tracks.len() + self.pending_vias.len();
        if total_queued == 0 {
            return Ok(0);
        }

        if !self.is_connected {
            self.pending_moves.clear();
            self.pending_tracks.clear();
            self.pending_vias.clear();
            return Ok(total_queued);
        }

        if self.nng.is_some() {
            for (ref_des, coords) in self.pending_moves.drain() {
                self.last_sent_moves.insert(ref_des, coords);
            }
            self.pending_tracks.clear();
            self.pending_vias.clear();
            self.total_messages_sent += total_queued;
            self.total_batches_committed += 1;
            self.last_flush_time = Instant::now();
            return Ok(total_queued);
        }

        // 1. Flush footprint moves
        let moves: Vec<(String, (f64, f64, f64))> = self.pending_moves.drain().collect();
        for (ref_des, (x, y, rot)) in moves {
            let msg = IpcMessage::MoveFootprint {
                reference: ref_des.clone(),
                x,
                y,
                rotation: rot,
            };
            self.send_raw_message(&msg)?;
            self.last_sent_moves.insert(ref_des, (x, y, rot));
        }

        // 2. Flush tracks
        let tracks: Vec<IpcMessage> = self.pending_tracks.drain(..).collect();
        for msg in tracks {
            self.send_raw_message(&msg)?;
        }

        // 3. Flush vias
        let vias: Vec<IpcMessage> = self.pending_vias.drain(..).collect();
        for msg in vias {
            self.send_raw_message(&msg)?;
        }

        // 4. Send CommitBatch & flush stream
        self.send_raw_message(&IpcMessage::CommitBatch)?;

        #[cfg(unix)]
        {
            if let Some(ref mut stream) = self.stream {
                if let Err(e) = stream.flush() {
                    self.is_connected = false;
                    self.stream = None;
                    return Err(e);
                }
            }
        }

        self.last_flush_time = Instant::now();
        self.total_batches_committed += 1;
        Ok(total_queued)
    }
}

/// Convenience helper that connects (or discovers `/tmp/kicad/api.sock`) and pushes `board` into KiCad.
///
/// # Arguments
/// * `client_slot` - Mutable slot holding an optional reusable [`KiCadIpcClient`] connection.
/// * `custom_socket_path` - Optional user-configured socket path override.
/// * `board` - Reference to the saved [`Board`] to synchronize into KiCad.
/// * `pcb_path` - Filesystem path to the written `.kicad_pcb` file.
/// * `ses_path` - Filesystem path to the written Specctra `.ses` file.
///
/// # Returns
/// `Ok(synced_count)` with the number of synchronized board items on success.
///
/// # Side Effects
/// Connects or reconnects `*client_slot` if disconnected or stale, and pushes the board state over IPC.
///
/// # Errors
/// Returns `Err(String)` if no active KiCad IPC socket can be discovered or if the board push fails after retry.
pub fn push_saved_board_to_kicad(
    client_slot: &mut Option<KiCadIpcClient>,
    custom_socket_path: Option<&str>,
    board: &Board,
    pcb_path: &Path,
    ses_path: &Path,
) -> Result<usize, String> {
    let needs_reconnect = client_slot.as_ref().map_or(true, |c| !c.is_connected());
    if needs_reconnect {
        *client_slot = KiCadIpcClient::connect_or_discover_quiet(custom_socket_path);
    }
    if let Some(ref mut client) = client_slot {
        match client.pull_board_into_kicad(board, pcb_path, ses_path) {
            Ok(n) => return Ok(n),
            Err(_) => {
                *client_slot = KiCadIpcClient::connect_or_discover_quiet(custom_socket_path);
                if let Some(ref mut retry_client) = client_slot {
                    return retry_client
                        .pull_board_into_kicad(board, pcb_path, ses_path)
                        .map_err(|e| e.to_string());
                }
            }
        }
    }
    Err("No active KiCad IPC socket found".to_string())
}

// ============================================================================
// Native KiCad 10 NNG (`libnng.so.1`) + Protobuf (`kiapi.*`) IPC Implementation
// ============================================================================

/// Null-terminated NNG socket option name for receive timeout (`"recv-timeout\0"`).
const NNG_OPT_RECVTIMEO: &[u8] = b"recv-timeout\0";
/// Null-terminated NNG socket option name for send timeout (`"send-timeout\0"`).
const NNG_OPT_SENDTIMEO: &[u8] = b"send-timeout\0";
/// NNG flag (`NNG_FLAG_ALLOC = 1`) instructing `nng_recv` to allocate the receive buffer dynamically.
const NNG_FLAG_ALLOC: i32 = 1;

/// KiCad `BoardLayer` protobuf enum value for front copper (`BL_F_CU = 3`).
const BL_F_CU: i32 = 3;
/// KiCad `BoardLayer` protobuf enum value for inner copper layer 1 (`BL_IN1_CU = 4`).
const BL_IN1_CU: i32 = 4;
/// KiCad `BoardLayer` protobuf enum value for back copper (`BL_B_CU = 34`).
const BL_B_CU: i32 = 34;

/// KiCad `KiCadObjectType` protobuf enum value for PCB footprints (`KOT_PCB_FOOTPRINT = 1`).
const KOT_PCB_FOOTPRINT: i32 = 1;
/// KiCad `KiCadObjectType` protobuf enum value for PCB copper traces (`KOT_PCB_TRACE = 11`).
const KOT_PCB_TRACE: i32 = 11;
/// KiCad `KiCadObjectType` protobuf enum value for PCB vias (`KOT_PCB_VIA = 12`).
const KOT_PCB_VIA: i32 = 12;
/// KiCad `KiCadObjectType` protobuf enum value for PCB copper arcs (`KOT_PCB_ARC = 13`).
const KOT_PCB_ARC: i32 = 13;

/// C-compatible NNG socket handle wrapper matching `nng_socket` in `libnng.so.1`.
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct NngSocket {
    /// Non-zero opaque socket identifier assigned by `nng_req0_open` / `nng_rep0_open`.
    id: u32,
}

/// FFI function pointer signature for `nng_req0_open`.
type FnNngReq0Open = unsafe extern "C" fn(*mut NngSocket) -> i32;
/// FFI function pointer signature for `nng_rep0_open`.
type FnNngRep0Open = unsafe extern "C" fn(*mut NngSocket) -> i32;
/// FFI function pointer signature for `nng_close`.
type FnNngClose = unsafe extern "C" fn(NngSocket) -> i32;
/// FFI function pointer signature for `nng_dial`.
type FnNngDial = unsafe extern "C" fn(NngSocket, *const std::os::raw::c_char, *mut usize, i32) -> i32;
/// FFI function pointer signature for `nng_listen`.
type FnNngListen = unsafe extern "C" fn(NngSocket, *const std::os::raw::c_char, *mut usize, i32) -> i32;
/// FFI function pointer signature for `nng_send`.
type FnNngSend = unsafe extern "C" fn(NngSocket, *const u8, usize, i32) -> i32;
/// FFI function pointer signature for `nng_recv`.
type FnNngRecv = unsafe extern "C" fn(NngSocket, *mut *mut u8, *mut usize, i32) -> i32;
/// FFI function pointer signature for `nng_free`.
type FnNngFree = unsafe extern "C" fn(*mut u8, usize);
/// FFI function pointer signature for `nng_socket_set_ms` / `nng_setopt_ms`.
type FnNngSocketSetMs = unsafe extern "C" fn(NngSocket, *const std::os::raw::c_char, i32) -> i32;

/// Dynamically loaded `libnng.so.1` symbol table for NNG `Req0`/`Rep0` socket operations.
#[allow(dead_code)]
struct NngLib {
    /// Loaded shared library handle kept alive for the lifetime of the process.
    _lib: libloading::Library,
    /// Symbol pointer to `nng_req0_open`.
    req0_open: FnNngReq0Open,
    /// Optional symbol pointer to `nng_rep0_open` (used by unit test mock servers).
    rep0_open: Option<FnNngRep0Open>,
    /// Symbol pointer to `nng_close`.
    close: FnNngClose,
    /// Symbol pointer to `nng_dial`.
    dial: FnNngDial,
    /// Optional symbol pointer to `nng_listen` (used by unit test mock servers).
    listen: Option<FnNngListen>,
    /// Symbol pointer to `nng_send`.
    send: FnNngSend,
    /// Symbol pointer to `nng_recv`.
    recv: FnNngRecv,
    /// Symbol pointer to `nng_free`.
    free: FnNngFree,
    /// Optional symbol pointer to `nng_socket_set_ms` or fallback `nng_setopt_ms`.
    set_ms: Option<FnNngSocketSetMs>,
}

impl NngLib {
    /// Returns a static reference to the lazily initialized `libnng.so.1` symbol table.
    ///
    /// # Returns
    /// `Ok(&'static NngLib)` when `libnng.so.1` is available and all required symbols are resolved.
    ///
    /// # Side Effects
    /// On first invocation, dynamically loads `libnng.so.1` into the process address space via `OnceLock`.
    ///
    /// # Errors
    /// Returns `std::io::Error` (`ErrorKind::Unsupported`) if `libnng.so.1` cannot be loaded or is missing symbols.
    fn load() -> std::io::Result<&'static Self> {
        static INSTANCE: std::sync::OnceLock<Result<NngLib, String>> = std::sync::OnceLock::new();
        let res = INSTANCE.get_or_init(Self::load_once);
        match res {
            Ok(lib) => Ok(lib),
            Err(msg) => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                msg.clone(),
            )),
        }
    }

    /// Probes standard Linux library paths to load `libnng.so.1` and resolve NNG C API function pointers.
    ///
    /// # Returns
    /// `Ok(NngLib)` containing resolved function pointers.
    ///
    /// # Side Effects
    /// Calls `dlopen` / `dlsym` via `libloading::Library::new`.
    ///
    /// # Errors
    /// Returns `Err(String)` describing the last dynamic linker error if all candidate paths fail.
    fn load_once() -> Result<Self, String> {
        let candidates = [
            "libnng.so.1",
            "/usr/lib/x86_64-linux-gnu/libnng.so.1",
            "/usr/lib64/libnng.so.1",
            "/usr/lib/libnng.so.1",
            "libnng.so",
        ];
        let mut last_err = None;
        for name in &candidates {
            unsafe {
                match libloading::Library::new(name) {
                    Ok(lib) => {
                        let req0_open = *lib
                            .get::<FnNngReq0Open>(b"nng_req0_open\0")
                            .map_err(|e| e.to_string())?;
                        let rep0_open = lib.get::<FnNngRep0Open>(b"nng_rep0_open\0").ok().map(|s| *s);
                        let close = *lib
                            .get::<FnNngClose>(b"nng_close\0")
                            .map_err(|e| e.to_string())?;
                        let dial = *lib
                            .get::<FnNngDial>(b"nng_dial\0")
                            .map_err(|e| e.to_string())?;
                        let listen = lib.get::<FnNngListen>(b"nng_listen\0").ok().map(|s| *s);
                        let send = *lib
                            .get::<FnNngSend>(b"nng_send\0")
                            .map_err(|e| e.to_string())?;
                        let recv = *lib
                            .get::<FnNngRecv>(b"nng_recv\0")
                            .map_err(|e| e.to_string())?;
                        let free = *lib
                            .get::<FnNngFree>(b"nng_free\0")
                            .map_err(|e| e.to_string())?;
                        let set_ms = lib
                            .get::<FnNngSocketSetMs>(b"nng_socket_set_ms\0")
                            .or_else(|_| lib.get::<FnNngSocketSetMs>(b"nng_setopt_ms\0"))
                            .ok()
                            .map(|s| *s);
                        return Ok(Self {
                            _lib: lib,
                            req0_open,
                            rep0_open,
                            close,
                            dial,
                            listen,
                            send,
                            recv,
                            free,
                            set_ms,
                        });
                    }
                    Err(e) => last_err = Some(e),
                }
            }
        }
        Err(format!("Failed to load libnng.so.1: {:?}", last_err))
    }
}

/// Active KiCad 9/10 NNG `Req0` RPC client speaking `kiapi.*` Protocol Buffers messages.
struct NngKiCadClient {
    /// Reference to the statically loaded `libnng.so.1` symbol table.
    lib: &'static NngLib,
    /// Open NNG `Req0` socket handle.
    sock: NngSocket,
    /// Session token returned by KiCad in `ApiResponseHeader` (tag 1) and echoed in subsequent requests.
    token: String,
    /// Unique client identifier string (`"vulcans-trials-<pid>"`) sent in `ApiRequestHeader` (tag 2).
    client_name: String,
}

impl Drop for NngKiCadClient {
    /// Closes the underlying NNG `Req0` socket handle when the client is dropped.
    ///
    /// # Side Effects
    /// Invokes `nng_close` via FFI if `self.sock.id != 0` and resets `self.sock.id` to `0`.
    fn drop(&mut self) {
        if self.sock.id != 0 {
            unsafe {
                let _ = (self.lib.close)(self.sock);
            }
            self.sock.id = 0;
        }
    }
}

impl NngKiCadClient {
    /// Opens an NNG `Req0` socket, dials `ipc://<socket_path>`, and verifies connectivity with a `Ping` command.
    ///
    /// # Arguments
    /// * `socket_path` - Filesystem path to the KiCad NNG Unix domain socket (e.g., `/tmp/kicad/api.sock`).
    ///
    /// # Returns
    /// Connected [`NngKiCadClient`] with an initialized session token.
    ///
    /// # Side Effects
    /// Loads `libnng.so.1`, opens an NNG socket with 10-second send/receive timeouts, dials the IPC endpoint,
    /// and sends a `kiapi.common.commands.Ping` protobuf request.
    ///
    /// # Errors
    /// Returns `std::io::Error` if `libnng.so.1` is unavailable, `nng_req0_open` or `nng_dial` fails,
    /// or the initial `Ping` command fails.
    fn connect(socket_path: &Path) -> std::io::Result<Self> {
        let lib = NngLib::load()?;
        let mut sock = NngSocket::default();
        unsafe {
            let rv = (lib.req0_open)(&mut sock);
            if rv != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("nng_req0_open failed with code {}", rv),
                ));
            }
            if let Some(set_ms) = lib.set_ms {
                let _ = set_ms(sock, NNG_OPT_RECVTIMEO.as_ptr() as *const _, 10_000);
                let _ = set_ms(sock, NNG_OPT_SENDTIMEO.as_ptr() as *const _, 10_000);
            }
            let uri = format!("ipc://{}\0", socket_path.to_string_lossy());
            let rv = (lib.dial)(sock, uri.as_ptr() as *const _, std::ptr::null_mut(), 0);
            if rv != 0 {
                let _ = (lib.close)(sock);
                return Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionRefused,
                    format!("nng_dial({}) failed with code {}", socket_path.display(), rv),
                ));
            }
        }

        let mut client = Self {
            lib,
            sock,
            token: String::new(),
            client_name: format!("vulcans-trials-{}", std::process::id()),
        };
        client.ping()?;
        Ok(client)
    }

    /// Wraps a protobuf command payload in an `ApiRequest` envelope, transmits it over NNG, and decodes the `ApiResponse`.
    ///
    /// # Arguments
    /// * `type_url` - Full protobuf `google.protobuf.Any` type URL (e.g., `"type.googleapis.com/kiapi.common.commands.Ping"`).
    /// * `payload` - Raw protobuf-encoded bytes of the inner command message.
    ///
    /// # Returns
    /// `Ok((response_type_url, response_payload_bytes))` unpacked from field 3 (`message`) of `ApiResponse`.
    ///
    /// # Side Effects
    /// Performs synchronous NNG `nng_send` and `nng_recv` calls, frees the NNG-allocated buffer via `nng_free`,
    /// and updates `self.token` if the response header contains a non-empty KiCad session token.
    ///
    /// # Errors
    /// Returns `std::io::Error` if `nng_send` or `nng_recv` fails, or if KiCad returns an `ApiStatusCode` other than `AS_OK` (`1`).
    fn send_command(&mut self, type_url: &str, payload: &[u8]) -> std::io::Result<(String, Vec<u8>)> {
        let mut header = Vec::new();
        if !self.token.is_empty() {
            proto_encode_str(1, &self.token, &mut header);
        }
        proto_encode_str(2, &self.client_name, &mut header);

        let any_msg = proto_encode_any(type_url, payload);
        let mut req = Vec::new();
        proto_encode_bytes(1, &header, &mut req);
        proto_encode_bytes(2, &any_msg, &mut req);

        let resp_bytes = unsafe {
            let rv = (self.lib.send)(self.sock, req.as_ptr(), req.len(), 0);
            if rv != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    format!("nng_send({}) failed with code {}", type_url, rv),
                ));
            }
            let mut buf_ptr: *mut u8 = std::ptr::null_mut();
            let mut buf_len: usize = 0;
            let rv = (self.lib.recv)(self.sock, &mut buf_ptr, &mut buf_len, NNG_FLAG_ALLOC);
            if rv != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("nng_recv({}) failed with code {}", type_url, rv),
                ));
            }
            let slice = if buf_ptr.is_null() || buf_len == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(buf_ptr, buf_len).to_vec()
            };
            if !buf_ptr.is_null() {
                (self.lib.free)(buf_ptr, buf_len);
            }
            slice
        };

        // Parse ApiResponse: header (1), status (2), message (3)
        let mut resp_token = String::new();
        let mut status_code: u64 = 0;
        let mut error_msg = String::new();
        let mut msg_type_url = String::new();
        let mut msg_val = Vec::new();

        proto_walk(&resp_bytes, |tag, wt, val| {
            match (tag, wt) {
                (1, 2) => {
                    proto_walk(val, |htag, hwt, hval| {
                        if htag == 1 && hwt == 2 {
                            if let Ok(s) = std::str::from_utf8(hval) {
                                resp_token = s.to_string();
                            }
                        }
                    });
                }
                (2, 2) => {
                    proto_walk(val, |stag, swt, sval| match (stag, swt) {
                        (1, 0) => status_code = proto_bytes_to_varint(sval),
                        (2, 2) => {
                            if let Ok(s) = std::str::from_utf8(sval) {
                                error_msg = s.to_string();
                            }
                        }
                        _ => {}
                    });
                }
                (3, 2) => {
                    let (turl, inner) = proto_decode_any(val);
                    msg_type_url = turl;
                    msg_val = inner;
                }
                _ => {}
            }
        });

        if !resp_token.is_empty() {
            self.token = resp_token;
        }
        // AS_OK = 1
        if status_code != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "KiCad IPC error on {} (status={}): {}",
                    type_url, status_code, error_msg
                ),
            ));
        }

        Ok((msg_type_url, msg_val))
    }

    /// Sends a `kiapi.common.commands.Ping` health-check request to KiCad.
    ///
    /// # Returns
    /// `Ok(())` when KiCad responds with `AS_OK`.
    ///
    /// # Side Effects
    /// Performs an NNG request/response round-trip and refreshes `self.token`.
    ///
    /// # Errors
    /// Returns `std::io::Error` if the NNG socket is disconnected or KiCad rejects the ping.
    fn ping(&mut self) -> std::io::Result<()> {
        let _ = self.send_command("type.googleapis.com/kiapi.common.commands.Ping", &[])?;
        Ok(())
    }

    /// Queries KiCad for open PCB documents (`DOCTYPE_PCB = 3`) and selects the `DocumentSpecifier` matching `board` or `pcb_path`.
    ///
    /// # Arguments
    /// * `board` - Reference to the [`Board`] being synchronized (used to match `board.file_path`).
    /// * `pcb_path` - Output `.kicad_pcb` path (matched after stripping `_vulcan`, `_routed`, or `_out` suffixes).
    ///
    /// # Returns
    /// Raw protobuf-encoded `DocumentSpecifier` bytes (`Vec<u8>`) for the target open PCB document.
    ///
    /// # Side Effects
    /// Sends `kiapi.common.commands.GetOpenDocuments` over the NNG socket.
    ///
    /// # Errors
    /// Returns `std::io::Error` (`ErrorKind::NotFound`) if no PCB is open in KiCad or if multiple PCBs are open
    /// and none match the stem of `pcb_path` or `board.file_path`.
    fn get_open_pcb_document(&mut self, board: &Board, pcb_path: &Path) -> std::io::Result<Vec<u8>> {
        let mut req = Vec::new();
        // DOCTYPE_PCB = 3
        proto_encode_varint_field(1, 3, &mut req);
        let (_, resp) = self.send_command(
            "type.googleapis.com/kiapi.common.commands.GetOpenDocuments",
            &req,
        )?;
        let mut docs: Vec<Vec<u8>> = Vec::new();
        proto_walk(&resp, |tag, wt, val| {
            if (tag == 1 || tag == 2) && wt == 2 {
                // Check if val contains an inner DocumentSpecifier at tag 1 (KiCad 10 repeated documents wrapper)
                let mut inner_spec = None;
                proto_walk(val, |itag, iwt, ival| {
                    if itag == 1 && iwt == 2 {
                        inner_spec = Some(ival.to_vec());
                    }
                });
                if let Some(spec) = inner_spec {
                    docs.push(spec);
                } else {
                    docs.push(val.to_vec());
                }
            }
        });
        if docs.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No PCB document is currently open in KiCad",
            ));
        }

        let clean_stem = |p: &Path| -> String {
            let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            fname
                .strip_suffix("_vulcan.kicad_pcb")
                .or_else(|| fname.strip_suffix("_routed.kicad_pcb"))
                .or_else(|| fname.strip_suffix("_out.kicad_pcb"))
                .or_else(|| fname.strip_suffix(".kicad_pcb"))
                .unwrap_or(fname)
                .to_string()
        };

        let out_stem = clean_stem(pcb_path);
        let src_stem = clean_stem(Path::new(&board.file_path));

        for doc in &docs {
            let mut board_filename = String::new();
            proto_walk(doc, |tag, wt, val| {
                if tag == 4 && wt == 2 {
                    if let Ok(s) = std::str::from_utf8(val) {
                        board_filename = s.to_string();
                    }
                }
            });
            let doc_stem = clean_stem(Path::new(&board_filename));
            if !doc_stem.is_empty()
                && ((out_stem == doc_stem || doc_stem.starts_with(&out_stem) || out_stem.starts_with(&doc_stem))
                    || (!src_stem.is_empty()
                        && (src_stem == doc_stem
                            || doc_stem.starts_with(&src_stem)
                            || src_stem.starts_with(&doc_stem))))
            {
                return Ok(doc.clone());
            }
        }

        if docs.len() == 1 {
            return Ok(docs[0].clone());
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Open PCB in KiCad does not match saved board '{}' (source '{}')",
                pcb_path.display(),
                board.file_path
            ),
        ))
    }

    /// Queries KiCad's net table for the given open PCB document via `kiapi.board.commands.GetNets`.
    ///
    /// # Arguments
    /// * `doc_spec` - Raw protobuf bytes of the target `DocumentSpecifier`.
    ///
    /// # Returns
    /// A [`HashMap`] mapping net name (`String`) to KiCad's internal integer `NetCode` (`i32`).
    ///
    /// # Side Effects
    /// Sends `kiapi.board.commands.GetNets` over the NNG socket.
    ///
    /// # Errors
    /// Returns `std::io::Error` if the RPC call fails.
    fn get_nets_map(&mut self, doc_spec: &[u8]) -> std::io::Result<HashMap<String, i32>> {
        let mut req = Vec::new();
        proto_encode_bytes(1, doc_spec, &mut req);
        let (_, resp) = self.send_command(
            "type.googleapis.com/kiapi.board.commands.GetNets",
            &req,
        )?;
        let mut map = HashMap::new();
        proto_walk(&resp, |tag, wt, val| {
            if tag == 1 && wt == 2 {
                let mut code: i32 = 0;
                let mut name = String::new();
                proto_walk(val, |ntag, nwt, nval| match (ntag, nwt) {
                    (1, 2) => {
                        proto_walk(nval, |ctag, cwt, cval| {
                            if ctag == 1 && cwt == 0 {
                                code = proto_bytes_to_varint(cval) as i32;
                            }
                        });
                    }
                    (2, 2) => {
                        if let Ok(s) = std::str::from_utf8(nval) {
                            name = s.to_string();
                        }
                    }
                    _ => {}
                });
                map.insert(name, code);
            }
        });
        Ok(map)
    }

    /// Synchronizes the complete `board` state into the open KiCad PCB editor over NNG + Protobuf.
    ///
    /// First attempts an atomic disk reload via `kiapi.common.commands.RevertDocument` followed by
    /// `RefillZones`. If `RevertDocument` is unsupported by the running KiCad build, falls back to
    /// an explicit transactional `BeginCommit` / `EndCommit` session that flips/transforms footprints,
    /// deletes stale copper items, creates all tracks/arcs/vias in chunks of 500, refills copper zones,
    /// and refreshes the PCBNew canvas.
    ///
    /// # Arguments
    /// * `board` - Reference to the [`Board`] containing updated placements, pin swaps, tracks, arcs, and vias.
    /// * `pcb_path` - Path to the saved `.kicad_pcb` file on disk.
    ///
    /// # Returns
    /// `Ok(total_synced)` indicating the total number of footprints, tracks, arcs, and vias synchronized.
    ///
    /// # Side Effects
    /// Mutates the live open PCB document inside KiCad 10 and triggers zone refills and canvas repainting.
    ///
    /// # Errors
    /// Returns `std::io::Error` if no matching PCB document is open or if any NNG transactional command fails.
    fn push_board_state(&mut self, board: &Board, pcb_path: &Path) -> std::io::Result<usize> {
        let doc_spec = self.get_open_pcb_document(board, pcb_path)?;

        // 1. Try atomic document revert from disk.
        // When Vulcan's Trials saves, the .kicad_pcb file on disk is already updated with all
        // placements, routed tracks, vias, silkscreen offsets, netclasses, and filled zones.
        // RevertDocument commands KiCad 10 / PCBNew to reload the document from disk (the exact IPC
        // equivalent of File -> Revert), perfectly synchronizing all board objects in a single atomic step.
        let mut rev_req = Vec::new();
        proto_encode_bytes(1, &doc_spec, &mut rev_req);
        if self
            .send_command(
                "type.googleapis.com/kiapi.common.commands.RevertDocument",
                &rev_req,
            )
            .is_ok()
        {
            // Successfully reverted board in KiCad! Refill zones and return total items.
            let mut refill_req = Vec::new();
            proto_encode_bytes(1, &doc_spec, &mut refill_req);
            let _ = self.send_command(
                "type.googleapis.com/kiapi.board.commands.RefillZones",
                &refill_req,
            );
            let total_items = board.footprints.len() + board.tracks.len() + board.arcs.len() + board.vias.len();
            return Ok(total_items);
        }

        let kicad_nets = self.get_nets_map(&doc_spec).unwrap_or_default();

        let mut item_header = Vec::new();
        proto_encode_bytes(1, &doc_spec, &mut item_header);

        // Fetch existing footprints, tracks, vias, and arcs on the board
        let mut get_items_req = Vec::new();
        proto_encode_bytes(1, &item_header, &mut get_items_req);
        for kot in [KOT_PCB_FOOTPRINT, KOT_PCB_TRACE, KOT_PCB_VIA, KOT_PCB_ARC] {
            proto_encode_varint_field(2, kot as u64, &mut get_items_req);
        }
        let get_items_resp = self
            .send_command(
                "type.googleapis.com/kiapi.common.commands.GetItems",
                &get_items_req,
            )
            .map(|(_, resp)| resp)
            .unwrap_or_default();

        let mut live_footprints: HashMap<String, (Vec<u8>, Vec<u8>, i64, i64, f64, i32)> =
            HashMap::new();
        let mut old_copper_ids: Vec<Vec<u8>> = Vec::new();

        proto_walk(&get_items_resp, |tag, wt, val| {
            if tag == 3 && wt == 2 {
                let (type_url, item_bytes) = proto_decode_any(val);
                if type_url.ends_with(".FootprintInstance") {
                    if let Some((ref_des, kiid_bytes, x_nm, y_nm, deg, layer)) =
                        parse_footprint_instance_summary(&item_bytes)
                    {
                        live_footprints.insert(
                            ref_des,
                            (item_bytes, kiid_bytes, x_nm, y_nm, deg, layer),
                        );
                    }
                } else if type_url.ends_with(".Track")
                    || type_url.ends_with(".Arc")
                    || type_url.ends_with(".Via")
                {
                    if let Some(kiid_bytes) = extract_item_kiid(&item_bytes) {
                        old_copper_ids.push(kiid_bytes);
                    }
                }
            }
        });

        // Begin atomic KiCad undo/redo commit session
        let mut begin_req = Vec::new();
        proto_encode_bytes(1, &item_header, &mut begin_req);
        let (_, begin_resp) = self.send_command(
            "type.googleapis.com/kiapi.common.commands.BeginCommit",
            &begin_req,
        )?;
        let mut commit_id = Vec::new();
        proto_walk(&begin_resp, |tag, wt, val| {
            if tag == 1 && wt == 2 {
                commit_id = val.to_vec();
            }
        });

        let sync_result = (|| -> std::io::Result<usize> {
            let mut total_synced = 0usize;

            // 1. Update footprints (flip layer if needed, then transform coordinates & pad nets)
            let mut updated_fp_anys: Vec<Vec<u8>> = Vec::new();
            for fp in &board.footprints {
                let Some((mut raw_fp, kiid_bytes, mut old_x, mut old_y, mut old_deg, old_layer)) =
                    live_footprints.remove(&fp.reference)
                else {
                    continue;
                };

                let target_layer = if fp.layer == 1 { BL_B_CU } else { BL_F_CU };
                if old_layer != 0 && old_layer != target_layer {
                    let mut flip_req = Vec::new();
                    proto_encode_bytes(1, &item_header, &mut flip_req);
                    proto_encode_bytes(2, &kiid_bytes, &mut flip_req);
                    // BFD_TOP_BOTTOM = 2
                    proto_encode_varint_field(3, 2, &mut flip_req);
                    if self
                        .send_command(
                            "type.googleapis.com/kiapi.board.commands.FlipItems",
                            &flip_req,
                        )
                        .is_ok()
                    {
                        let mut get_by_id = Vec::new();
                        proto_encode_bytes(1, &item_header, &mut get_by_id);
                        proto_encode_bytes(2, &kiid_bytes, &mut get_by_id);
                        if let Ok((_, by_id_resp)) = self.send_command(
                            "type.googleapis.com/kiapi.common.commands.GetItemsById",
                            &get_by_id,
                        ) {
                            proto_walk(&by_id_resp, |t, w, v| {
                                if t == 3 && w == 2 {
                                    let (turl, ibytes) = proto_decode_any(v);
                                    if turl.ends_with(".FootprintInstance") {
                                        if let Some((_, _, nx, ny, ndeg, _)) =
                                            parse_footprint_instance_summary(&ibytes)
                                        {
                                            raw_fp = ibytes;
                                            old_x = nx;
                                            old_y = ny;
                                            old_deg = ndeg;
                                        }
                                    }
                                }
                            });
                        }
                    }
                }

                let new_x = mm_to_nm(fp.x);
                let new_y = mm_to_nm(fp.y);
                let new_deg = normalize_angle_deg(fp.rotation);
                let moved = (new_x - old_x).abs() > 50
                    || (new_y - old_y).abs() > 50
                    || normalize_angle_deg(new_deg - old_deg).abs() > 1e-3
                    || old_layer != target_layer;

                // Build pad net map for this footprint ( in case pin/gate swap modified pad nets )
                let mut pad_nets: HashMap<String, (i32, String)> = HashMap::new();
                for pad in &fp.pads {
                    if !pad.number.is_empty() && !pad.net_name.is_empty() {
                        let code = kicad_nets
                            .get(&pad.net_name)
                            .copied()
                            .unwrap_or(pad.net_id as i32);
                        pad_nets.insert(pad.number.clone(), (code, pad.net_name.clone()));
                    }
                }

                if moved || !board.pin_swap_history.is_empty() {
                    let transformed = transform_footprint_instance_proto(
                        &raw_fp,
                        old_x,
                        old_y,
                        old_deg,
                        new_x,
                        new_y,
                        new_deg,
                        fp.is_locked,
                        &pad_nets,
                    );
                    updated_fp_anys.push(proto_encode_any(
                        "type.googleapis.com/kiapi.board.types.FootprintInstance",
                        &transformed,
                    ));
                    total_synced += 1;
                }
            }

            if !updated_fp_anys.is_empty() {
                let mut update_req = Vec::new();
                proto_encode_bytes(1, &item_header, &mut update_req);
                for any_fp in &updated_fp_anys {
                    proto_encode_bytes(2, any_fp, &mut update_req);
                }
                let _ = self.send_command(
                    "type.googleapis.com/kiapi.common.commands.UpdateItems",
                    &update_req,
                )?;
            }

            // 2. Delete old tracks, arcs, and vias before creating the saved set
            if !old_copper_ids.is_empty() {
                let mut del_req = Vec::new();
                proto_encode_bytes(1, &item_header, &mut del_req);
                for id_bytes in &old_copper_ids {
                    proto_encode_bytes(2, id_bytes, &mut del_req);
                }
                let _ = self.send_command(
                    "type.googleapis.com/kiapi.common.commands.DeleteItems",
                    &del_req,
                )?;
            }

            // Helper to resolve board net_id -> (KiCad NetCode, NetName)
            let resolve_net = |net_id: usize| -> (i32, String) {
                let net_name = board
                    .nets
                    .get(net_id)
                    .cloned()
                    .unwrap_or_default();
                let code = kicad_nets
                    .get(&net_name)
                    .copied()
                    .unwrap_or(net_id as i32);
                (code, net_name)
            };

            // 3. Build and create all tracks, arcs, and vias in KiCad
            let mut new_items: Vec<Vec<u8>> =
                Vec::with_capacity(board.tracks.len() + board.arcs.len() + board.vias.len());

            for trk in &board.tracks {
                let (net_code, net_name) = resolve_net(trk.net_id);
                let layer_enum = board_layer_to_kiapi(board.layer_name(trk.layer));
                let trk_bytes = build_track_proto(
                    trk.start,
                    trk.end,
                    trk.width,
                    layer_enum,
                    net_code,
                    &net_name,
                    trk.is_locked,
                );
                new_items.push(proto_encode_any(
                    "type.googleapis.com/kiapi.board.types.Track",
                    &trk_bytes,
                ));
                total_synced += 1;
            }

            for arc in &board.arcs {
                let (net_code, net_name) = resolve_net(arc.net_id);
                let layer_enum = board_layer_to_kiapi(board.layer_name(arc.layer));
                let arc_bytes = build_arc_proto(
                    arc.start(),
                    arc.mid(),
                    arc.end(),
                    arc.width,
                    layer_enum,
                    net_code,
                    &net_name,
                    arc.is_locked,
                );
                new_items.push(proto_encode_any(
                    "type.googleapis.com/kiapi.board.types.Arc",
                    &arc_bytes,
                ));
                total_synced += 1;
            }

            for via in &board.vias {
                let (net_code, net_name) = resolve_net(via.net_id);
                let via_bytes = build_via_proto(
                    via.position,
                    via.size,
                    via.drill,
                    net_code,
                    &net_name,
                    via.is_locked,
                );
                new_items.push(proto_encode_any(
                    "type.googleapis.com/kiapi.board.types.Via",
                    &via_bytes,
                ));
                total_synced += 1;
            }

            if !new_items.is_empty() {
                for chunk in new_items.chunks(500) {
                    let mut create_req = Vec::new();
                    proto_encode_bytes(1, &item_header, &mut create_req);
                    for item in chunk {
                        proto_encode_bytes(2, item, &mut create_req);
                    }
                    let _ = self.send_command(
                        "type.googleapis.com/kiapi.common.commands.CreateItems",
                        &create_req,
                    )?;
                }
            }

            Ok(total_synced)
        })();

        // Always close the commit session (commit on Ok, drop on Err)
        let mut end_req = Vec::new();
        if !commit_id.is_empty() {
            proto_encode_bytes(1, &commit_id, &mut end_req);
        }
        let action = if sync_result.is_ok() { 1 } else { 2 }; // CMA_COMMIT = 1, CMA_DROP = 2
        proto_encode_varint_field(2, action, &mut end_req);
        proto_encode_str(3, "Vulcan's Trials PCB save sync", &mut end_req);
        proto_encode_bytes(4, &item_header, &mut end_req);
        let _ = self.send_command(
            "type.googleapis.com/kiapi.common.commands.EndCommit",
            &end_req,
        );

        let synced = sync_result?;

        // Refill copper zones in KiCad after updating footprints/tracks/vias and refresh display
        let mut refill_req = Vec::new();
        proto_encode_bytes(1, &doc_spec, &mut refill_req);
        let _ = self.send_command(
            "type.googleapis.com/kiapi.board.commands.RefillZones",
            &refill_req,
        );
        let _ = self.send_command(
            "type.googleapis.com/kiapi.common.commands.RefreshEditor",
            &[],
        );

        Ok(synced)
    }
}

/// Converts a distance in millimeters (`mm`) to KiCad internal integer nanometers (`nm`).
///
/// # Arguments
/// * `mm` - Coordinate or dimension in millimeters (`1.0 mm = 1,000,000 nm`).
///
/// # Returns
/// Nearest rounded integer value in nanometers (`i64`).
fn mm_to_nm(mm: f64) -> i64 {
    (mm * 1_000_000.0).round() as i64
}

/// Normalizes an orientation angle in degrees into the half-open interval `(-180.0, 180.0]`.
///
/// # Arguments
/// * `deg` - Input angle in degrees.
///
/// # Returns
/// Equivalent angle wrapped to `(-180.0, 180.0]` degrees.
fn normalize_angle_deg(deg: f64) -> f64 {
    let mut a = deg % 360.0;
    if a > 180.0 {
        a -= 360.0;
    } else if a <= -180.0 {
        a += 360.0;
    }
    a
}

/// Maps a KiCad copper layer name (`"F.Cu"`, `"In1.Cu"`..`"In30.Cu"`, `"B.Cu"`) to its `kiapi.board.types.BoardLayer` enum value.
///
/// # Arguments
/// * `layer_name` - KiCad copper layer string (e.g., `"F.Cu"`, `"In2.Cu"`, `"B.Cu"`).
///
/// # Returns
/// Protobuf `BoardLayer` enum integer (`BL_F_CU = 3`, `BL_IN1_CU = 4..33`, `BL_B_CU = 34`), defaulting to `BL_F_CU` (`3`).
fn board_layer_to_kiapi(layer_name: &str) -> i32 {
    match layer_name {
        "F.Cu" => BL_F_CU,
        "B.Cu" => BL_B_CU,
        s if s.starts_with("In") && s.ends_with(".Cu") => {
            let num_str = &s[2..s.len() - 3];
            if let Ok(n) = num_str.parse::<i32>() {
                if (1..=30).contains(&n) {
                    return BL_IN1_CU + (n - 1);
                }
            }
            BL_F_CU
        }
        _ => BL_F_CU,
    }
}

/// Appends a base-128 Protocol Buffers unsigned varint encoding of `v` into `out`.
///
/// # Arguments
/// * `v` - 64-bit unsigned integer value (or two's-complement cast `i64 as u64`) to encode.
/// * `out` - Destination byte buffer.
///
/// # Side Effects
/// Appends 1 to 10 varint bytes to `out`.
fn proto_encode_varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Appends a Protocol Buffers field tag (`(field_num << 3) | wire_type`) as a varint into `out`.
///
/// # Arguments
/// * `field_num` - Protobuf field number (`1..`).
/// * `wire_type` - Protobuf wire type (`0` = varint, `1` = 64-bit, `2` = length-delimited, `5` = 32-bit).
/// * `out` - Destination byte buffer.
///
/// # Side Effects
/// Appends the encoded tag varint to `out`.
fn proto_encode_tag(field_num: u32, wire_type: u8, out: &mut Vec<u8>) {
    proto_encode_varint(((field_num as u64) << 3) | (wire_type as u64), out);
}

/// Appends a non-zero varint field (wire type `0`) with proto3 default-zero suppression into `out`.
///
/// # Arguments
/// * `field_num` - Protobuf field number.
/// * `val` - Varint value to encode (omitted if `val == 0`).
/// * `out` - Destination byte buffer.
///
/// # Side Effects
/// Appends the field tag and varint value to `out` when `val != 0`.
fn proto_encode_varint_field(field_num: u32, val: u64, out: &mut Vec<u8>) {
    if val != 0 {
        proto_encode_tag(field_num, 0, out);
        proto_encode_varint(val, out);
    }
}

/// Appends a 64-bit IEEE-754 floating-point field (wire type `1`, little-endian `double`) into `out`.
///
/// # Arguments
/// * `field_num` - Protobuf field number.
/// * `val` - `f64` value to encode in little-endian byte order.
/// * `out` - Destination byte buffer.
///
/// # Side Effects
/// Appends the field tag and 8 little-endian bytes to `out`.
fn proto_encode_f64_field(field_num: u32, val: f64, out: &mut Vec<u8>) {
    proto_encode_tag(field_num, 1, out);
    out.extend_from_slice(&val.to_le_bytes());
}

/// Appends a length-delimited byte slice field (wire type `2`) into `out`.
///
/// # Arguments
/// * `field_num` - Protobuf field number.
/// * `bytes` - Raw payload bytes (`[u8]`) for the length-delimited field or sub-message.
/// * `out` - Destination byte buffer.
///
/// # Side Effects
/// Appends the field tag, payload byte length varint, and `bytes` slice to `out`.
fn proto_encode_bytes(field_num: u32, bytes: &[u8], out: &mut Vec<u8>) {
    proto_encode_tag(field_num, 2, out);
    proto_encode_varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

/// Appends a non-empty UTF-8 string field (wire type `2`) into `out`.
///
/// # Arguments
/// * `field_num` - Protobuf field number.
/// * `s` - UTF-8 string value (omitted if empty per proto3 semantics).
/// * `out` - Destination byte buffer.
///
/// # Side Effects
/// Appends the length-delimited string bytes to `out` when `!s.is_empty()`.
fn proto_encode_str(field_num: u32, s: &str, out: &mut Vec<u8>) {
    if !s.is_empty() {
        proto_encode_bytes(field_num, s.as_bytes(), out);
    }
}

/// Encodes a `google.protobuf.Any` message wrapping `type_url` (field 1) and serialized `value` bytes (field 2).
///
/// # Arguments
/// * `type_url` - Full protobuf type URL string.
/// * `value` - Serialized inner protobuf message bytes (`[u8]`).
///
/// # Returns
/// Serialized `google.protobuf.Any` message bytes (`Vec<u8>`).
fn proto_encode_any(type_url: &str, value: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(type_url.len() + value.len() + 8);
    proto_encode_str(1, type_url, &mut out);
    if !value.is_empty() {
        proto_encode_bytes(2, value, &mut out);
    }
    out
}

/// Encodes a `kiapi.common.types.Vector2` protobuf message with nanometer coordinates `(x_nm, y_nm)`.
///
/// # Arguments
/// * `x_nm` - X coordinate in nanometers (`nm`, field 1).
/// * `y_nm` - Y coordinate in nanometers (`nm`, field 2).
///
/// # Returns
/// Serialized `Vector2` protobuf bytes (`Vec<u8>`).
fn proto_encode_vector2(x_nm: i64, y_nm: i64) -> Vec<u8> {
    let mut out = Vec::with_capacity(20);
    proto_encode_varint_field(1, x_nm as u64, &mut out);
    proto_encode_varint_field(2, y_nm as u64, &mut out);
    out
}

/// Encodes a `kiapi.common.types.Distance` protobuf message with a length in nanometers (`nm`).
///
/// # Arguments
/// * `value_nm` - Distance in nanometers (`nm`, field 1).
///
/// # Returns
/// Serialized `Distance` protobuf bytes (`Vec<u8>`).
fn proto_encode_distance(value_nm: i64) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    proto_encode_varint_field(1, value_nm as u64, &mut out);
    out
}

/// Encodes a `kiapi.common.types.Angle` protobuf message in degrees (`f64`, field 1).
///
/// # Arguments
/// * `deg` - Angle in degrees (`f64`).
///
/// # Returns
/// Serialized `Angle` protobuf bytes (`Vec<u8>`).
fn proto_encode_angle(deg: f64) -> Vec<u8> {
    let mut out = Vec::with_capacity(9);
    proto_encode_f64_field(1, deg, &mut out);
    out
}

/// Encodes a `kiapi.board.types.Net` protobuf message containing a `NetCode` (field 1) and net name string (field 2).
///
/// # Arguments
/// * `code` - KiCad integer net code (`i32`).
/// * `name` - Net name string (e.g., `"GND"`, `"+3V3"`).
///
/// # Returns
/// Serialized `Net` protobuf bytes (`Vec<u8>`).
fn proto_encode_net(code: i32, name: &str) -> Vec<u8> {
    let mut net_code = Vec::with_capacity(8);
    proto_encode_varint_field(1, code as u32 as u64, &mut net_code);
    let mut out = Vec::with_capacity(name.len() + 12);
    proto_encode_bytes(1, &net_code, &mut out);
    proto_encode_str(2, name, &mut out);
    out
}

/// Builds a serialized `kiapi.board.types.Track` protobuf message from millimeter geometry and net metadata.
///
/// # Arguments
/// * `start` - Track start `(x, y)` coordinates in millimeters (`mm`, field 2).
/// * `end` - Track end `(x, y)` coordinates in millimeters (`mm`, field 3).
/// * `width_mm` - Copper track width in millimeters (`mm`, field 4).
/// * `layer_enum` - KiCad `BoardLayer` enum value (field 6).
/// * `net_code` - KiCad integer net code (field 7).
/// * `net_name` - KiCad net name string (field 7).
/// * `is_locked` - Whether the track is locked (`LS_LOCKED = 2` vs `LS_UNLOCKED = 1`, field 5).
///
/// # Returns
/// Serialized `Track` protobuf payload (`Vec<u8>`).
fn build_track_proto(
    start: (f64, f64),
    end: (f64, f64),
    width_mm: f64,
    layer_enum: i32,
    net_code: i32,
    net_name: &str,
    is_locked: bool,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    let s_vec = proto_encode_vector2(mm_to_nm(start.0), mm_to_nm(start.1));
    let e_vec = proto_encode_vector2(mm_to_nm(end.0), mm_to_nm(end.1));
    let w_dist = proto_encode_distance(mm_to_nm(width_mm));
    let net_msg = proto_encode_net(net_code, net_name);
    proto_encode_bytes(2, &s_vec, &mut out);
    proto_encode_bytes(3, &e_vec, &mut out);
    proto_encode_bytes(4, &w_dist, &mut out);
    proto_encode_varint_field(5, if is_locked { 2 } else { 1 }, &mut out);
    proto_encode_varint_field(6, layer_enum as u64, &mut out);
    proto_encode_bytes(7, &net_msg, &mut out);
    out
}

/// Builds a serialized `kiapi.board.types.Arc` protobuf message from 3-point millimeter geometry and net metadata.
///
/// # Arguments
/// * `start` - Arc start `(x, y)` coordinates in millimeters (`mm`, field 2).
/// * `mid` - Arc midpoint `(x, y)` coordinates on the arc in millimeters (`mm`, field 3).
/// * `end` - Arc end `(x, y)` coordinates in millimeters (`mm`, field 4).
/// * `width_mm` - Copper arc width in millimeters (`mm`, field 5).
/// * `layer_enum` - KiCad `BoardLayer` enum value (field 7).
/// * `net_code` - KiCad integer net code (field 8).
/// * `net_name` - KiCad net name string (field 8).
/// * `is_locked` - Whether the arc is locked (`LS_LOCKED = 2` vs `LS_UNLOCKED = 1`, field 6).
///
/// # Returns
/// Serialized `Arc` protobuf payload (`Vec<u8>`).
fn build_arc_proto(
    start: (f64, f64),
    mid: (f64, f64),
    end: (f64, f64),
    width_mm: f64,
    layer_enum: i32,
    net_code: i32,
    net_name: &str,
    is_locked: bool,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(80);
    let s_vec = proto_encode_vector2(mm_to_nm(start.0), mm_to_nm(start.1));
    let m_vec = proto_encode_vector2(mm_to_nm(mid.0), mm_to_nm(mid.1));
    let e_vec = proto_encode_vector2(mm_to_nm(end.0), mm_to_nm(end.1));
    let w_dist = proto_encode_distance(mm_to_nm(width_mm));
    let net_msg = proto_encode_net(net_code, net_name);
    proto_encode_bytes(2, &s_vec, &mut out);
    proto_encode_bytes(3, &m_vec, &mut out);
    proto_encode_bytes(4, &e_vec, &mut out);
    proto_encode_bytes(5, &w_dist, &mut out);
    proto_encode_varint_field(6, if is_locked { 2 } else { 1 }, &mut out);
    proto_encode_varint_field(7, layer_enum as u64, &mut out);
    proto_encode_bytes(8, &net_msg, &mut out);
    out
}

/// Builds a serialized `kiapi.board.types.Via` protobuf message (through-hole `F.Cu` to `B.Cu` padstack) from millimeter dimensions.
///
/// # Arguments
/// * `position` - Via center `(x, y)` coordinates in millimeters (`mm`, field 2).
/// * `size_mm` - Outer copper pad diameter in millimeters (`mm`).
/// * `drill_mm` - Drill hole diameter in millimeters (`mm`).
/// * `net_code` - KiCad integer net code (field 5).
/// * `net_name` - KiCad net name string (field 5).
/// * `is_locked` - Whether the via is locked (`LS_LOCKED = 2` vs `LS_UNLOCKED = 1`, field 4).
///
/// # Returns
/// Serialized `Via` protobuf payload (`Vec<u8>`).
fn build_via_proto(
    position: (f64, f64),
    size_mm: f64,
    drill_mm: f64,
    net_code: i32,
    net_name: &str,
    is_locked: bool,
) -> Vec<u8> {
    let pos_vec = proto_encode_vector2(mm_to_nm(position.0), mm_to_nm(position.1));
    let drill_nm = mm_to_nm(drill_mm);
    let size_nm = mm_to_nm(size_mm);

    // DrillProperties: start_layer=3(F.Cu), end_layer=34(B.Cu), diameter=Vector2, shape=1(DS_CIRCLE)
    let mut drill_props = Vec::with_capacity(28);
    proto_encode_varint_field(1, BL_F_CU as u64, &mut drill_props);
    proto_encode_varint_field(2, BL_B_CU as u64, &mut drill_props);
    let drill_vec = proto_encode_vector2(drill_nm, drill_nm);
    proto_encode_bytes(3, &drill_vec, &mut drill_props);
    proto_encode_varint_field(4, 1, &mut drill_props);

    // PadStackLayer: layer=3(F.Cu), shape=1(PSS_CIRCLE), size=Vector2
    let mut cu_layer = Vec::with_capacity(24);
    proto_encode_varint_field(1, BL_F_CU as u64, &mut cu_layer);
    proto_encode_varint_field(2, 1, &mut cu_layer);
    let size_vec = proto_encode_vector2(size_nm, size_nm);
    proto_encode_bytes(3, &size_vec, &mut cu_layer);

    // PadStack: type=1(PST_NORMAL), layers=[3, 34], drill=3, copper_layers=5
    let mut pad_stack = Vec::with_capacity(64);
    proto_encode_varint_field(1, 1, &mut pad_stack);
    proto_encode_varint_field(2, BL_F_CU as u64, &mut pad_stack);
    proto_encode_varint_field(2, BL_B_CU as u64, &mut pad_stack);
    proto_encode_bytes(3, &drill_props, &mut pad_stack);
    proto_encode_bytes(5, &cu_layer, &mut pad_stack);

    let net_msg = proto_encode_net(net_code, net_name);
    let mut out = Vec::with_capacity(112);
    proto_encode_bytes(2, &pos_vec, &mut out);
    proto_encode_bytes(3, &pad_stack, &mut out);
    proto_encode_varint_field(4, if is_locked { 2 } else { 1 }, &mut out);
    proto_encode_bytes(5, &net_msg, &mut out);
    proto_encode_varint_field(6, 1, &mut out); // VT_THROUGH = 1
    out
}

/// Decodes a base-128 Protocol Buffers varint from `buf` starting at `*pos`, advancing `*pos` past the decoded bytes.
///
/// # Arguments
/// * `buf` - Input byte slice (`[u8]`) containing wire-encoded protobuf data.
/// * `pos` - Mutable cursor index into `buf`.
///
/// # Returns
/// `Some(u64)` with the decoded varint value, or `None` if `buf` is truncated or the varint exceeds 64 bits.
///
/// # Side Effects
/// Advances `*pos` by the number of bytes consumed (up to 10 bytes).
///
/// # Errors
/// Returns `None` on unexpected end-of-buffer or varint overflow (`shift >= 64`).
fn proto_decode_varint(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift = 0;
    while *pos < buf.len() {
        let b = buf[*pos];
        *pos += 1;
        result |= ((b & 0x7F) as u64) << shift;
        if (b & 0x80) == 0 {
            return Some(result);
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

/// Decodes a standalone varint byte slice `buf` from offset `0`, returning `0` if empty or malformed.
///
/// # Arguments
/// * `buf` - Slice of raw varint bytes (`[u8]`) yielded by [`proto_walk`].
///
/// # Returns
/// Decoded `u64` value, or `0` if decoding fails.
fn proto_bytes_to_varint(buf: &[u8]) -> u64 {
    let mut pos = 0;
    proto_decode_varint(buf, &mut pos).unwrap_or(0)
}

/// Iterates sequentially over all top-level Protocol Buffers fields in `buf`, invoking `f(field_num, wire_type, field_bytes)` for each field.
///
/// # Arguments
/// * `buf` - Serialized protobuf message bytes (`[u8]`).
/// * `f` - Callback closure receiving `(tag: u32, wire_type: u8, value_slice: &[u8])`.
///
/// # Side Effects
/// Invokes `f` for each valid wire field until `buf` is exhausted or a malformed tag/length is encountered.
fn proto_walk<F: FnMut(u32, u8, &[u8])>(buf: &[u8], mut f: F) {
    let mut pos = 0;
    while pos < buf.len() {
        let Some(tag_wire) = proto_decode_varint(buf, &mut pos) else {
            break;
        };
        let tag = (tag_wire >> 3) as u32;
        let wt = (tag_wire & 0x7) as u8;
        match wt {
            0 => {
                let start = pos;
                if proto_decode_varint(buf, &mut pos).is_none() {
                    break;
                }
                f(tag, wt, &buf[start..pos]);
            }
            1 => {
                if pos + 8 > buf.len() {
                    break;
                }
                f(tag, wt, &buf[pos..pos + 8]);
                pos += 8;
            }
            2 => {
                let Some(len) = proto_decode_varint(buf, &mut pos) else {
                    break;
                };
                let len = len as usize;
                if pos + len > buf.len() {
                    break;
                }
                f(tag, wt, &buf[pos..pos + len]);
                pos += len;
            }
            5 => {
                if pos + 4 > buf.len() {
                    break;
                }
                f(tag, wt, &buf[pos..pos + 4]);
                pos += 4;
            }
            _ => break,
        }
    }
}

/// Walks a serialized protobuf message `buf` and produces a rewritten message buffer, allowing selective in-place field transformation.
///
/// For each field `(tag, wire_type, val)`, invokes `f(tag, wire_type, val, &mut out)`. If `f` returns `false`,
/// the original field is copied verbatim into `out`; if `f` returns `true`, `f` is responsible for writing the replacement field into `out`.
///
/// # Arguments
/// * `buf` - Source protobuf message bytes (`[u8]`).
/// * `f` - Rewriting closure returning `true` when it has emitted a custom replacement for the current field.
///
/// # Returns
/// Newly allocated `Vec<u8>` containing the rewritten protobuf message.
fn proto_rewrite<F: FnMut(u32, u8, &[u8], &mut Vec<u8>) -> bool>(buf: &[u8], mut f: F) -> Vec<u8> {
    let mut out = Vec::with_capacity(buf.len() + 16);
    proto_walk(buf, |tag, wt, val| {
        if !f(tag, wt, val, &mut out) {
            proto_encode_tag(tag, wt, &mut out);
            if wt == 2 {
                proto_encode_varint(val.len() as u64, &mut out);
            }
            out.extend_from_slice(val);
        }
    });
    out
}

/// Unpacks a `google.protobuf.Any` message buffer into its `(type_url, value_bytes)` pair.
///
/// # Arguments
/// * `buf` - Serialized `google.protobuf.Any` wire bytes (`[u8]`).
///
/// # Returns
/// A tuple `(type_url, value)` where `type_url` is field 1 and `value` is the raw inner message at field 2.
fn proto_decode_any(buf: &[u8]) -> (String, Vec<u8>) {
    let mut type_url = String::new();
    let mut value = Vec::new();
    proto_walk(buf, |tag, wt, val| match (tag, wt) {
        (1, 2) => {
            if let Ok(s) = std::str::from_utf8(val) {
                type_url = s.to_string();
            }
        }
        (2, 2) => value = val.to_vec(),
        _ => {}
    });
    (type_url, value)
}

/// Decodes a `kiapi.common.types.Vector2` protobuf message into `(x_nm, y_nm)` nanometer coordinates.
///
/// # Arguments
/// * `buf` - Serialized `Vector2` protobuf bytes (`[u8]`).
///
/// # Returns
/// Tuple `(x_nm, y_nm)` as signed 64-bit integers in nanometers (`nm`).
fn proto_decode_vector2(buf: &[u8]) -> (i64, i64) {
    let mut x: i64 = 0;
    let mut y: i64 = 0;
    proto_walk(buf, |tag, wt, val| match (tag, wt) {
        (1, 0) => x = proto_bytes_to_varint(val) as i64,
        (2, 0) => y = proto_bytes_to_varint(val) as i64,
        _ => {}
    });
    (x, y)
}

/// Decodes a `kiapi.common.types.Angle` protobuf message into degrees (`f64`).
///
/// # Arguments
/// * `buf` - Serialized `Angle` protobuf bytes (`[u8]`).
///
/// # Returns
/// Orientation angle in degrees (`f64`, defaulting to `0.0` if field 1 is absent).
fn proto_decode_angle(buf: &[u8]) -> f64 {
    let mut deg = 0.0;
    proto_walk(buf, |tag, wt, val| {
        if tag == 1 && wt == 1 && val.len() == 8 {
            deg = f64::from_le_bytes(val.try_into().unwrap());
        }
    });
    deg
}

/// Extracts the `kiapi.common.types.KIID` protobuf sub-message (field 1, wire type 2) from a board item message.
///
/// # Arguments
/// * `item_bytes` - Serialized protobuf bytes of a `FootprintInstance`, `Track`, `Arc`, or `Via`.
///
/// # Returns
/// `Some(Vec<u8>)` containing the raw `KIID` sub-message bytes, or `None` if field 1 is missing.
///
/// # Errors
/// Returns `None` if no `KIID` field is present at tag 1.
fn extract_item_kiid(item_bytes: &[u8]) -> Option<Vec<u8>> {
    let mut kiid = None;
    proto_walk(item_bytes, |tag, wt, val| {
        if tag == 1 && wt == 2 && kiid.is_none() {
            kiid = Some(val.to_vec());
        }
    });
    kiid
}

/// Extracts the UTF-8 string value from a `kiapi.board.types.Field` protobuf message (`Field.text.text.text`).
///
/// # Arguments
/// * `field_bytes` - Serialized `Field` protobuf message bytes (`[u8]`).
///
/// # Returns
/// Extracted string (such as a footprint reference designator `"U1"`), or an empty `String` if not found.
fn extract_field_text_str(field_bytes: &[u8]) -> String {
    let mut res = String::new();
    // Field.text = tag 3 (BoardText) -> BoardText.text = tag 2 (Text) -> Text.text = tag 5 (string)
    proto_walk(field_bytes, |tag, wt, val| {
        if tag == 3 && wt == 2 {
            proto_walk(val, |btag, bwt, bval| {
                if btag == 2 && bwt == 2 {
                    proto_walk(bval, |ttag, twt, tval| {
                        if ttag == 5 && twt == 2 {
                            if let Ok(s) = std::str::from_utf8(tval) {
                                res = s.to_string();
                            }
                        }
                    });
                }
            });
        }
    });
    res
}

/// Parses the identity and placement summary of a `kiapi.board.types.FootprintInstance` protobuf message.
///
/// # Arguments
/// * `fp_bytes` - Serialized `FootprintInstance` protobuf message bytes (`[u8]`).
///
/// # Returns
/// `Some((reference_designator, kiid_bytes, x_nm, y_nm, orientation_deg, board_layer_enum))` if both a `KIID`
/// and a non-empty reference designator are present.
///
/// # Errors
/// Returns `None` if the message lacks a `KIID` (field 1) or has an empty reference field (field 7 or definition field 6.7).
fn parse_footprint_instance_summary(
    fp_bytes: &[u8],
) -> Option<(String, Vec<u8>, i64, i64, f64, i32)> {
    let mut kiid = None;
    let mut pos = (0i64, 0i64);
    let mut deg = 0.0f64;
    let mut layer = BL_F_CU;
    let mut ref_str = String::new();

    proto_walk(fp_bytes, |tag, wt, val| match (tag, wt) {
        (1, 2) => kiid = Some(val.to_vec()),
        (2, 2) => pos = proto_decode_vector2(val),
        (3, 2) => deg = proto_decode_angle(val),
        (4, 0) => layer = proto_bytes_to_varint(val) as i32,
        (7, 2) => {
            let s = extract_field_text_str(val);
            if !s.is_empty() {
                ref_str = s;
            }
        }
        (6, 2) if ref_str.is_empty() => {
            proto_walk(val, |dtag, dwt, dval| {
                if dtag == 7 && dwt == 2 {
                    let s = extract_field_text_str(dval);
                    if !s.is_empty() {
                        ref_str = s;
                    }
                }
            });
        }
        _ => {}
    });

    let kiid = kiid?;
    if ref_str.is_empty() {
        return None;
    }
    Some((ref_str, kiid, pos.0, pos.1, deg, layer))
}

/// Applies rigid-body rotation + translation to a `FootprintInstance` protobuf message and all
/// its child items (`Field`, `BoardText`, `BoardTextBox`, `Pad`, `BoardGraphicShape`, `Zone`, `Dimension`),
/// matching KiCad 10's `FOOTPRINT::Deserialize` coordinate expectations.
///
/// Also updates pad net assignments (`Pad.net`, field 4) when `pad_nets` contains pin/gate-swapped
/// net mappings, and converts axis-aligned graphic rectangles (`BoardGraphicShape` tag 5) into
/// 4-corner closed polygons (`PolySet` tag 8) when rotated by a non-zero angle.
///
/// # Arguments
/// * `fp_bytes` - Original serialized `FootprintInstance` protobuf bytes (`[u8]`).
/// * `old_x` - Previous footprint anchor X coordinate in nanometers (`nm`).
/// * `old_y` - Previous footprint anchor Y coordinate in nanometers (`nm`).
/// * `old_deg` - Previous footprint orientation in degrees (`deg`).
/// * `new_x` - Target footprint anchor X coordinate in nanometers (`nm`).
/// * `new_y` - Target footprint anchor Y coordinate in nanometers (`nm`).
/// * `new_deg` - Target footprint orientation in degrees (`deg`).
/// * `is_locked` - Whether the footprint is locked (`LS_LOCKED = 2` vs `LS_UNLOCKED = 1`).
/// * `pad_nets` - Map from pad number string to `(kicad_net_code, net_name)` for pad net synchronization.
///
/// # Returns
/// Transformed `FootprintInstance` protobuf message bytes (`Vec<u8>`) ready for `UpdateItems`.
#[allow(clippy::too_many_arguments)]
fn transform_footprint_instance_proto(
    fp_bytes: &[u8],
    old_x: i64,
    old_y: i64,
    old_deg: f64,
    new_x: i64,
    new_y: i64,
    new_deg: f64,
    is_locked: bool,
    pad_nets: &HashMap<String, (i32, String)>,
) -> Vec<u8> {
    let delta_deg = normalize_angle_deg(new_deg - old_deg);
    let rad = (-delta_deg).to_radians();
    let (sin_r, cos_r) = rad.sin_cos();

    let xform_vec2 = |v_bytes: &[u8]| -> Vec<u8> {
        let (vx, vy) = proto_decode_vector2(v_bytes);
        let dx = (vx - old_x) as f64;
        let dy = (vy - old_y) as f64;
        let rx = dx * cos_r - dy * sin_r;
        let ry = dx * sin_r + dy * cos_r;
        proto_encode_vector2(new_x + rx.round() as i64, new_y + ry.round() as i64)
    };

    let xform_angle = |a_bytes: &[u8]| -> Vec<u8> {
        let a = proto_decode_angle(a_bytes);
        proto_encode_angle(normalize_angle_deg(a + delta_deg))
    };

    let xform_text = |text_bytes: &[u8]| -> Vec<u8> {
        proto_rewrite(text_bytes, |tag, wt, val, out| match (tag, wt) {
            (2, 2) => {
                proto_encode_bytes(2, &xform_vec2(val), out);
                true
            }
            (3, 2) => {
                let attrs = proto_rewrite(val, |atag, awt, aval, aout| {
                    if atag == 4 && awt == 2 {
                        proto_encode_bytes(4, &xform_angle(aval), aout);
                        true
                    } else {
                        false
                    }
                });
                proto_encode_bytes(3, &attrs, out);
                true
            }
            _ => false,
        })
    };

    let xform_board_text = |bt_bytes: &[u8]| -> Vec<u8> {
        proto_rewrite(bt_bytes, |tag, wt, val, out| {
            if tag == 2 && wt == 2 {
                proto_encode_bytes(2, &xform_text(val), out);
                true
            } else {
                false
            }
        })
    };

    let xform_field = |field_bytes: &[u8]| -> Vec<u8> {
        proto_rewrite(field_bytes, |tag, wt, val, out| {
            if tag == 3 && wt == 2 {
                proto_encode_bytes(3, &xform_board_text(val), out);
                true
            } else {
                false
            }
        })
    };

    let xform_polyset = |ps_bytes: &[u8]| -> Vec<u8> {
        let xform_polyline = |pl_bytes: &[u8]| -> Vec<u8> {
            proto_rewrite(pl_bytes, |tag, wt, val, out| {
                if tag == 1 && wt == 2 {
                    let node = proto_rewrite(val, |ntag, nwt, nval, nout| match (ntag, nwt) {
                        (1, 2) => {
                            proto_encode_bytes(1, &xform_vec2(nval), nout);
                            true
                        }
                        (2, 2) => {
                            let arc = proto_rewrite(nval, |atag, awt, aval, aout| {
                                if (1..=3).contains(&atag) && awt == 2 {
                                    proto_encode_bytes(atag, &xform_vec2(aval), aout);
                                    true
                                } else {
                                    false
                                }
                            });
                            proto_encode_bytes(2, &arc, nout);
                            true
                        }
                        _ => false,
                    });
                    proto_encode_bytes(1, &node, out);
                    true
                } else {
                    false
                }
            })
        };
        proto_rewrite(ps_bytes, |tag, wt, val, out| {
            if tag == 1 && wt == 2 {
                let poly = proto_rewrite(val, |ptag, pwt, pval, pout| {
                    if (ptag == 1 || ptag == 2) && pwt == 2 {
                        proto_encode_bytes(ptag, &xform_polyline(pval), pout);
                        true
                    } else {
                        false
                    }
                });
                proto_encode_bytes(1, &poly, out);
                true
            } else {
                false
            }
        })
    };

    let mut has_pos = false;
    let mut has_orient = false;
    let mut has_locked = false;

    let mut out = proto_rewrite(fp_bytes, |tag, wt, val, out| match (tag, wt) {
        (2, 2) => {
            has_pos = true;
            proto_encode_bytes(2, &proto_encode_vector2(new_x, new_y), out);
            true
        }
        (3, 2) => {
            has_orient = true;
            proto_encode_bytes(3, &proto_encode_angle(new_deg), out);
            true
        }
        (5, 0) => {
            has_locked = true;
            proto_encode_varint_field(5, if is_locked { 2 } else { 1 }, out);
            true
        }
        (7..=10, 2) => {
            proto_encode_bytes(tag, &xform_field(val), out);
            true
        }
        (6, 2) => {
            // Footprint definition
            let def = proto_rewrite(val, |dtag, dwt, dval, dout| match (dtag, dwt) {
                (7..=10, 2) => {
                    proto_encode_bytes(dtag, &xform_field(dval), dout);
                    true
                }
                (11, 2) => {
                    let (turl, ibytes) = proto_decode_any(dval);
                    let new_ibytes = if turl.ends_with(".Pad") {
                        let mut pad_num = String::new();
                        proto_walk(&ibytes, |ptag, pwt, pval| {
                            if ptag == 3 && pwt == 2 {
                                if let Ok(s) = std::str::from_utf8(pval) {
                                    pad_num = s.to_string();
                                }
                            }
                        });
                        let override_net = pad_nets.get(&pad_num);
                        let mut wrote_net = false;
                        let mut p_out = proto_rewrite(&ibytes, |ptag, pwt, pval, pout| match (ptag, pwt) {
                            (4, 2) if override_net.is_some() => {
                                let (c, n) = override_net.unwrap();
                                proto_encode_bytes(4, &proto_encode_net(*c, n), pout);
                                wrote_net = true;
                                true
                            }
                            (6, 2) => {
                                let mut has_angle = false;
                                let mut ps = proto_rewrite(pval, |stag, swt, sval, sout| {
                                    if stag == 6 && swt == 2 {
                                        has_angle = true;
                                        proto_encode_bytes(6, &xform_angle(sval), sout);
                                        true
                                    } else {
                                        false
                                    }
                                });
                                if !has_angle && delta_deg.abs() > 1e-6 {
                                    proto_encode_bytes(
                                        6,
                                        &proto_encode_angle(normalize_angle_deg(delta_deg)),
                                        &mut ps,
                                    );
                                }
                                proto_encode_bytes(6, &ps, pout);
                                true
                            }
                            _ => false,
                        });
                        if !wrote_net {
                            if let Some((c, n)) = override_net {
                                proto_encode_bytes(4, &proto_encode_net(*c, n), &mut p_out);
                            }
                        }
                        p_out
                    } else if turl.ends_with(".BoardText") {
                        xform_board_text(&ibytes)
                    } else if turl.ends_with(".Field") {
                        xform_field(&ibytes)
                    } else if turl.ends_with(".BoardTextBox") {
                        proto_rewrite(&ibytes, |btag, bwt, bval, bout| {
                            if btag == 2 && bwt == 2 {
                                let tb = proto_rewrite(bval, |ttag, twt, tval, tout| match (ttag, twt) {
                                    (2 | 3, 2) => {
                                        proto_encode_bytes(ttag, &xform_vec2(tval), tout);
                                        true
                                    }
                                    (4, 2) => {
                                        let attrs = proto_rewrite(tval, |atag, awt, aval, aout| {
                                            if atag == 4 && awt == 2 {
                                                proto_encode_bytes(4, &xform_angle(aval), aout);
                                                true
                                            } else {
                                                false
                                            }
                                        });
                                        proto_encode_bytes(4, &attrs, tout);
                                        true
                                    }
                                    _ => false,
                                });
                                proto_encode_bytes(2, &tb, bout);
                                true
                            } else {
                                false
                            }
                        })
                    } else if turl.ends_with(".BoardGraphicShape") {
                        proto_rewrite(&ibytes, |gtag, gwt, gval, gout| {
                            if gtag == 1 && gwt == 2 {
                                let gs = proto_rewrite(gval, |stag, swt, sval, sout| match (stag, swt) {
                                    (4 | 6 | 7 | 9, 2) => {
                                        let pts = proto_rewrite(sval, |ptag, pwt, pval, pout| {
                                            if (1..=4).contains(&ptag) && pwt == 2 {
                                                proto_encode_bytes(ptag, &xform_vec2(pval), pout);
                                                true
                                            } else {
                                                false
                                            }
                                        });
                                        proto_encode_bytes(stag, &pts, sout);
                                        true
                                    }
                                    (5, 2) => {
                                        if delta_deg.abs() < 1e-6 {
                                            let rect = proto_rewrite(sval, |rtag, rwt, rval, rout| {
                                                if (rtag == 1 || rtag == 2) && rwt == 2 {
                                                    proto_encode_bytes(rtag, &xform_vec2(rval), rout);
                                                    true
                                                } else {
                                                    false
                                                }
                                            });
                                            proto_encode_bytes(5, &rect, sout);
                                        } else {
                                            // Convert rotated rectangle to 4-corner closed PolySet (tag 8)
                                            let mut tl = (0i64, 0i64);
                                            let mut br = (0i64, 0i64);
                                            proto_walk(sval, |rtag, rwt, rval| match (rtag, rwt) {
                                                (1, 2) => tl = proto_decode_vector2(rval),
                                                (2, 2) => br = proto_decode_vector2(rval),
                                                _ => {}
                                            });
                                            let corners = [
                                                (tl.0, tl.1),
                                                (br.0, tl.1),
                                                (br.0, br.1),
                                                (tl.0, br.1),
                                            ];
                                            let mut pl = Vec::new();
                                            for (cx, cy) in corners {
                                                let v = proto_encode_vector2(cx, cy);
                                                let xv = xform_vec2(&v);
                                                let mut node = Vec::new();
                                                proto_encode_bytes(1, &xv, &mut node);
                                                proto_encode_bytes(1, &node, &mut pl);
                                            }
                                            proto_encode_varint_field(2, 1, &mut pl); // closed = true
                                            let mut pwh = Vec::new();
                                            proto_encode_bytes(1, &pl, &mut pwh);
                                            let mut ps = Vec::new();
                                            proto_encode_bytes(1, &pwh, &mut ps);
                                            proto_encode_bytes(8, &ps, sout);
                                        }
                                        true
                                    }
                                    (8, 2) => {
                                        proto_encode_bytes(8, &xform_polyset(sval), sout);
                                        true
                                    }
                                    _ => false,
                                });
                                proto_encode_bytes(1, &gs, gout);
                                true
                            } else {
                                false
                            }
                        })
                    } else if turl.ends_with(".Zone") {
                        proto_rewrite(&ibytes, |ztag, zwt, zval, zout| match (ztag, zwt) {
                            (4, 2) => {
                                proto_encode_bytes(4, &xform_polyset(zval), zout);
                                true
                            }
                            (10, 2) => {
                                let fp = proto_rewrite(zval, |ftag, fwt, fval, fout| {
                                    if ftag == 2 && fwt == 2 {
                                        proto_encode_bytes(2, &xform_polyset(fval), fout);
                                        true
                                    } else {
                                        false
                                    }
                                });
                                proto_encode_bytes(10, &fp, zout);
                                true
                            }
                            _ => false,
                        })
                    } else {
                        ibytes
                    };
                    proto_encode_bytes(11, &proto_encode_any(&turl, &new_ibytes), dout);
                    true
                }
                _ => false,
            });
            proto_encode_bytes(6, &def, out);
            true
        }
        _ => false,
    });

    if !has_pos {
        proto_encode_bytes(2, &proto_encode_vector2(new_x, new_y), &mut out);
    }
    if !has_orient {
        proto_encode_bytes(3, &proto_encode_angle(new_deg), &mut out);
    }
    if !has_locked {
        proto_encode_varint_field(5, if is_locked { 2 } else { 1 }, &mut out);
    }
    out
}

/// Helper function to scan a directory for KiCad IPC sockets.
///
/// # Arguments
/// * `dir` - Directory path to inspect for `.sock` Unix domain socket files.
///
/// # Returns
/// `Some(PathBuf)` of the first matching KiCad socket file found in `dir`, or `None` if `dir` is not a directory
/// or contains no matching `.sock` entry.
///
/// # Side Effects
/// Reads directory entries from the filesystem via `std::fs::read_dir`.
///
/// # Errors
/// Returns `None` if `dir` does not exist, cannot be read, or contains no KiCad socket files.
fn scan_dir_for_socket(dir: &Path) -> Option<PathBuf> {
    if !dir.is_dir() {
        return None;
    }
    let is_kicad_dir = dir.to_string_lossy().contains("kicad");
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        let fname = p.file_name().and_then(|f| f.to_str()).unwrap_or("");
        if fname.ends_with(".sock") {
            if is_kicad_dir || fname.starts_with("kicad_ipc") || fname.contains("kicad") || fname == "api.sock" {
                return Some(p);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    #[cfg(unix)]
    use std::os::unix::net::UnixListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;

    #[test]
    #[cfg(unix)]
    fn test_mock_ipc_server_handshake_and_batch_formatting() {
        let sock_path = format!("/tmp/kicad_ipc_test_handshake_{}.sock", std::process::id());
        let _ = fs::remove_file(&sock_path);

        let listener = UnixListener::bind(&sock_path).expect("Failed to bind test unix socket");
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_srv = Arc::clone(&stop_flag);

        let server_thread = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("Accept failed");
            let mut reader = BufReader::new(stream);
            let mut received_messages = Vec::new();

            let mut line = String::new();
            while !stop_srv.load(Ordering::Relaxed) {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        let trimmed = line.trim();
                        if !trimmed.is_empty() {
                            let msg: IpcMessage = serde_json::from_str(trimmed)
                                .unwrap_or_else(|e| panic!("Failed to parse JSON '{}': {}", trimmed, e));
                            let is_commit = matches!(msg, IpcMessage::CommitBatch);
                            received_messages.push(msg);
                            if is_commit {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Read line error: {}", e);
                        break;
                    }
                }
            }
            received_messages
        });

        // Client connects
        let mut client = KiCadIpcClient::connect(&sock_path).expect("Client connect failed");
        assert!(client.is_connected());

        // Queue a move and commit
        client.queue_move_footprint("U1", 25.4, 30.5, 90.0);
        let count = client.commit_batch().expect("Commit batch failed");
        assert_eq!(count, 1);

        let received = server_thread.join().expect("Server thread join failed");
        let _ = fs::remove_file(&sock_path);

        assert_eq!(received.len(), 3);
        assert_eq!(
            received[0],
            IpcMessage::Handshake {
                client: "Vulcan's Trials".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            }
        );
        assert_eq!(
            received[1],
            IpcMessage::MoveFootprint {
                reference: "U1".to_string(),
                x: 25.4,
                y: 30.5,
                rotation: 90.0,
            }
        );
        assert_eq!(received[2], IpcMessage::CommitBatch);
    }

    #[test]
    #[cfg(unix)]
    fn test_kicad_ipc_footprint_move_stream() {
        let sock_path = format!("/tmp/kicad_ipc_test_stream_{}.sock", std::process::id());
        let _ = fs::remove_file(&sock_path);

        let listener = UnixListener::bind(&sock_path).expect("Failed to bind test unix socket");
        let server_thread = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("Accept failed");
            let reader = BufReader::new(stream);
            let mut messages = Vec::new();
            for line in reader.lines().flatten() {
                if !line.trim().is_empty() {
                    let msg: IpcMessage = serde_json::from_str(&line).expect("Valid JSON");
                    let is_commit = matches!(msg, IpcMessage::CommitBatch);
                    messages.push(msg);
                    if is_commit {
                        break;
                    }
                }
            }
            messages
        });

        let mut client = KiCadIpcClient::connect(&sock_path).expect("Connect failed");
        client.set_throttle_interval(Duration::from_millis(50));

        // Rapid stream updates for C1: intermediate jitter should coalesce into final state
        client.stream_footprint_move("C1", 10.0, 10.0, 0.0, false).unwrap();
        client.stream_footprint_move("C1", 10.1, 10.1, 0.0, false).unwrap();
        client.stream_footprint_move("C1", 10.2, 10.2, 0.0, false).unwrap();
        client.stream_footprint_move("C1", 15.0, 20.0, 90.0, false).unwrap();

        // Flush and commit
        client.commit_batch().unwrap();

        let received = server_thread.join().unwrap();
        let _ = fs::remove_file(&sock_path);

        // Handshake + 1 coalesced MoveFootprint (final coords) + CommitBatch
        assert_eq!(received.len(), 3);
        assert!(matches!(received[0], IpcMessage::Handshake { .. }));
        assert_eq!(
            received[1],
            IpcMessage::MoveFootprint {
                reference: "C1".to_string(),
                x: 15.0,
                y: 20.0,
                rotation: 90.0,
            }
        );
        assert_eq!(received[2], IpcMessage::CommitBatch);
    }

    #[test]
    #[cfg(unix)]
    fn test_kicad_ipc_track_and_via_batch_commit() {
        let sock_path = format!("/tmp/kicad_ipc_test_tracks_{}.sock", std::process::id());
        let _ = fs::remove_file(&sock_path);

        let listener = UnixListener::bind(&sock_path).expect("Failed to bind test unix socket");
        let server_thread = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("Accept failed");
            let reader = BufReader::new(stream);
            let mut messages = Vec::new();
            for line in reader.lines().flatten() {
                if !line.trim().is_empty() {
                    let msg: IpcMessage = serde_json::from_str(&line).expect("Valid JSON");
                    let is_commit = matches!(msg, IpcMessage::CommitBatch);
                    messages.push(msg);
                    if is_commit {
                        break;
                    }
                }
            }
            messages
        });

        let mut client = KiCadIpcClient::connect(&sock_path).expect("Connect failed");

        client.queue_add_track((0.0, 0.0), (10.0, 0.0), 0.25, "F.Cu", 1);
        client.queue_add_track((10.0, 0.0), (10.0, 5.0), 0.25, "B.Cu", 1);
        client.queue_add_via((10.0, 5.0), 0.60, 0.30, 1);

        let committed = client.commit_batch().expect("Commit batch failed");
        assert_eq!(committed, 3);

        let received = server_thread.join().unwrap();
        let _ = fs::remove_file(&sock_path);

        // Handshake + 2 AddTrack + 1 AddVia + 1 CommitBatch = 5 messages
        assert_eq!(received.len(), 5);
        assert!(matches!(received[0], IpcMessage::Handshake { .. }));
        assert_eq!(
            received[1],
            IpcMessage::AddTrack {
                start: (0.0, 0.0),
                end: (10.0, 0.0),
                width: 0.25,
                layer: "F.Cu".to_string(),
                net_id: 1,
            }
        );
        assert_eq!(
            received[2],
            IpcMessage::AddTrack {
                start: (10.0, 0.0),
                end: (10.0, 5.0),
                width: 0.25,
                layer: "B.Cu".to_string(),
                net_id: 1,
            }
        );
        assert_eq!(
            received[3],
            IpcMessage::AddVia {
                position: (10.0, 5.0),
                size: 0.60,
                drill: 0.30,
                net_id: 1,
            }
        );
        assert_eq!(received[4], IpcMessage::CommitBatch);
    }

    #[test]
    fn test_kicad_ipc_socket_discovery() {
        let dummy_sock = format!("/tmp/kicad_ipc_mock_disc_{}.sock", std::process::id());
        fs::write(&dummy_sock, b"").unwrap();

        env::set_var("KICAD_IPC_PATH", &dummy_sock);
        let disc = KiCadIpcClient::discover_socket_path();
        env::remove_var("KICAD_IPC_PATH");
        let _ = fs::remove_file(&dummy_sock);

        assert_eq!(disc, Some(PathBuf::from(dummy_sock)));
    }

    #[test]
    fn test_normalize_socket_path() {
        assert_eq!(
            KiCadIpcClient::normalize_socket_path("ipc:///tmp/kicad/api.sock"),
            PathBuf::from("/tmp/kicad/api.sock")
        );
        assert_eq!(
            KiCadIpcClient::normalize_socket_path("ipc:/tmp/kicad/api.sock"),
            PathBuf::from("/tmp/kicad/api.sock")
        );
        assert_eq!(
            KiCadIpcClient::normalize_socket_path("  ipc:///tmp/kicad.sock  "),
            PathBuf::from("/tmp/kicad.sock")
        );
        assert_eq!(
            KiCadIpcClient::normalize_socket_path("/tmp/kicad/api.sock"),
            PathBuf::from("/tmp/kicad/api.sock")
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_connect_with_ipc_uri_scheme() {
        let sock_path = format!("/tmp/kicad_ipc_uri_test_{}.sock", std::process::id());
        let _ = fs::remove_file(&sock_path);

        let listener = UnixListener::bind(&sock_path).expect("Failed to bind test unix socket");
        let server_thread = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("Accept failed");
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            let _ = reader.read_line(&mut line);
            line
        });

        // Pass path with ipc:// URI prefix
        let uri = format!("ipc://{}", sock_path);
        let client = KiCadIpcClient::connect(&uri).expect("Connect with ipc:// failed");
        assert!(client.is_connected());

        let line = server_thread.join().expect("Server thread join failed");
        let _ = fs::remove_file(&sock_path);
        assert!(line.contains("Handshake"));
    }

    #[test]
    #[cfg(unix)]
    fn test_pull_board_into_kicad_sends_footprint_tracks_and_pull_board() {
        let sock_path = format!("/tmp/kicad_ipc_pull_board_test_{}.sock", std::process::id());
        let _ = fs::remove_file(&sock_path);

        let listener = UnixListener::bind(&sock_path).expect("Failed to bind test unix socket");
        let server_thread = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("Accept failed");
            let reader = BufReader::new(stream);
            let mut messages = Vec::new();
            for line in reader.lines().flatten() {
                if !line.trim().is_empty() {
                    let msg: IpcMessage = serde_json::from_str(&line).expect("Valid JSON");
                    let is_pull = matches!(msg, IpcMessage::PullBoard { .. });
                    messages.push(msg);
                    if is_pull {
                        break;
                    }
                }
            }
            messages
        });

        let mut client =
            KiCadIpcClient::connect_or_discover_quiet(Some(&sock_path)).expect("Connect failed");
        let mut board = crate::board::Board::default();
        board.tracks.push(crate::board::TrackSegment {
            start: (1.0, 2.0),
            end: (5.0, 6.0),
            width: 0.25,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        board.vias.push(crate::board::Via {
            position: (5.0, 6.0),
            size: 0.60,
            drill: 0.30,
            net_id: 1,
            is_locked: false,
        });

        let synced = client
            .pull_board_into_kicad(
                &board,
                Path::new("/tmp/test_pull_out.kicad_pcb"),
                Path::new("/tmp/test_pull_out.ses"),
            )
            .expect("pull_board_into_kicad must succeed");
        assert_eq!(synced, 2);

        let received = server_thread.join().expect("Server thread join failed");
        let _ = fs::remove_file(&sock_path);

        assert!(received.iter().any(|m| matches!(m, IpcMessage::ClearTracks)));
        assert!(received.iter().any(|m| matches!(m, IpcMessage::AddTrack { .. })));
        assert!(received.iter().any(|m| matches!(m, IpcMessage::AddVia { .. })));
        assert!(received.iter().any(|m| matches!(m, IpcMessage::CommitBatch)));
        assert_eq!(
            received.last(),
            Some(&IpcMessage::PullBoard {
                pcb_path: "/tmp/test_pull_out.kicad_pcb".to_string(),
                ses_path: "/tmp/test_pull_out.ses".to_string(),
            })
        );
    }

    #[test]
    fn test_nng_protobuf_push_board_state_roundtrip() {
        let Ok(lib) = NngLib::load() else {
            return;
        };
        let (Some(rep0_open), Some(listen_fn)) = (lib.rep0_open, lib.listen) else {
            return;
        };

        let sock_path = format!("/tmp/kicad_nng_test_{}.sock", std::process::id());
        let _ = fs::remove_file(&sock_path);

        let mut srv_sock = NngSocket::default();
        unsafe {
            assert_eq!(rep0_open(&mut srv_sock), 0);
            if let Some(set_ms) = lib.set_ms {
                let _ = set_ms(srv_sock, NNG_OPT_RECVTIMEO.as_ptr() as *const _, 5_000);
                let _ = set_ms(srv_sock, NNG_OPT_SENDTIMEO.as_ptr() as *const _, 5_000);
            }
            let uri = format!("ipc://{}\0", sock_path);
            assert_eq!(listen_fn(srv_sock, uri.as_ptr() as *const _, std::ptr::null_mut(), 0), 0);
        }

        let srv_sock_id = srv_sock.id;
        let server_thread = thread::spawn(move || {
            let srv_lib = NngLib::load().unwrap();
            let s = NngSocket { id: srv_sock_id };
            let mut seen_commands = Vec::new();

            loop {
                let req_bytes = unsafe {
                    let mut buf_ptr: *mut u8 = std::ptr::null_mut();
                    let mut buf_len: usize = 0;
                    let rv = (srv_lib.recv)(s, &mut buf_ptr, &mut buf_len, NNG_FLAG_ALLOC);
                    if rv != 0 {
                        break;
                    }
                    let slice = std::slice::from_raw_parts(buf_ptr, buf_len).to_vec();
                    (srv_lib.free)(buf_ptr, buf_len);
                    slice
                };

                let mut cmd_type_url = String::new();
                proto_walk(&req_bytes, |tag, wt, val| {
                    if tag == 2 && wt == 2 {
                        let (turl, _) = proto_decode_any(val);
                        cmd_type_url = turl;
                    }
                });
                seen_commands.push(cmd_type_url.clone());

                let mut resp_payload = Vec::new();
                let mut resp_type_url = "type.googleapis.com/kiapi.common.types.CommandStatusResponse";

                if cmd_type_url.ends_with(".GetOpenDocuments") {
                    resp_type_url = "type.googleapis.com/kiapi.common.commands.GetOpenDocumentsResponse";
                    let mut doc = Vec::new();
                    proto_encode_varint_field(1, 3, &mut doc);
                    proto_encode_str(4, "test_board.kicad_pcb", &mut doc);
                    proto_encode_bytes(1, &doc, &mut resp_payload);
                } else if cmd_type_url.ends_with(".GetNets") {
                    resp_type_url = "type.googleapis.com/kiapi.board.commands.NetsResponse";
                    let net = proto_encode_net(1, "VCC");
                    proto_encode_bytes(1, &net, &mut resp_payload);
                } else if cmd_type_url.ends_with(".GetItems") {
                    resp_type_url = "type.googleapis.com/kiapi.common.commands.GetItemsResponse";
                    // Return 1 existing footprint R1 and 1 old track
                    let mut kiid_fp = Vec::new();
                    proto_encode_str(1, "00000000-0000-4000-8000-000000000001", &mut kiid_fp);
                    let mut text_msg = Vec::new();
                    proto_encode_bytes(2, &proto_encode_vector2(10_000_000, 20_000_000), &mut text_msg);
                    proto_encode_str(5, "R1", &mut text_msg);
                    let mut bt_msg = Vec::new();
                    proto_encode_bytes(2, &text_msg, &mut bt_msg);
                    let mut field_msg = Vec::new();
                    proto_encode_bytes(3, &bt_msg, &mut field_msg);

                    let mut fp_inst = Vec::new();
                    proto_encode_bytes(1, &kiid_fp, &mut fp_inst);
                    proto_encode_bytes(2, &proto_encode_vector2(10_000_000, 20_000_000), &mut fp_inst);
                    proto_encode_bytes(3, &proto_encode_angle(0.0), &mut fp_inst);
                    proto_encode_varint_field(4, BL_F_CU as u64, &mut fp_inst);
                    proto_encode_bytes(7, &field_msg, &mut fp_inst);
                    let any_fp = proto_encode_any(
                        "type.googleapis.com/kiapi.board.types.FootprintInstance",
                        &fp_inst,
                    );
                    proto_encode_bytes(3, &any_fp, &mut resp_payload);

                    let mut kiid_trk = Vec::new();
                    proto_encode_str(1, "00000000-0000-4000-8000-000000000002", &mut kiid_trk);
                    let mut old_trk = Vec::new();
                    proto_encode_bytes(1, &kiid_trk, &mut old_trk);
                    let any_trk = proto_encode_any(
                        "type.googleapis.com/kiapi.board.types.Track",
                        &old_trk,
                    );
                    proto_encode_bytes(3, &any_trk, &mut resp_payload);
                } else if cmd_type_url.ends_with(".BeginCommit") {
                    resp_type_url = "type.googleapis.com/kiapi.common.commands.BeginCommitResponse";
                    let mut cid = Vec::new();
                    proto_encode_str(1, "00000000-0000-4000-8000-000000000099", &mut cid);
                    proto_encode_bytes(1, &cid, &mut resp_payload);
                } else if cmd_type_url.ends_with(".RevertDocument") {
                    // Simulate KiCad instance without RevertDocument so this test exercises
                    // the complete fallback item-by-item sync pipeline
                    resp_type_url = "type.googleapis.com/kiapi.common.types.CommandStatusResponse";
                    proto_encode_varint_field(1, 5, &mut resp_payload);
                } else {
                    proto_encode_varint_field(1, 1, &mut resp_payload);
                }

                let mut header = Vec::new();
                proto_encode_str(1, "test-kicad-token", &mut header);
                let mut status = Vec::new();
                if cmd_type_url.ends_with(".RevertDocument") {
                    proto_encode_varint_field(1, 5, &mut status); // status != 1 (error)
                } else {
                    proto_encode_varint_field(1, 1, &mut status); // AS_OK = 1
                }
                let any_resp = proto_encode_any(resp_type_url, &resp_payload);
                let mut api_resp = Vec::new();
                proto_encode_bytes(1, &header, &mut api_resp);
                proto_encode_bytes(2, &status, &mut api_resp);
                proto_encode_bytes(3, &any_resp, &mut api_resp);

                unsafe {
                    let _ = (srv_lib.send)(s, api_resp.as_ptr(), api_resp.len(), 0);
                }

                if cmd_type_url.ends_with(".RefillZones") {
                    break;
                }
            }

            unsafe {
                let _ = (srv_lib.close)(s);
            }
            seen_commands
        });

        let mut client = KiCadIpcClient::connect(&sock_path).expect("NNG connect must succeed");
        assert!(client.is_connected());

        let mut board = crate::board::Board::default();
        board.nets = vec!["".to_string(), "VCC".to_string()];
        board.footprints.push(crate::board::Footprint {
            reference: "R1".to_string(),
            x: 15.0,
            y: 25.0,
            rotation: 90.0,
            layer: 0,
            ..Default::default()
        });
        board.tracks.push(crate::board::TrackSegment {
            start: (15.0, 25.0),
            end: (20.0, 25.0),
            width: 0.25,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });
        board.arcs.push(crate::board::ArcSegment::new(
            (20.0, 26.0),
            1.0,
            -std::f64::consts::FRAC_PI_2,
            0.0,
            0.25,
            0,
            1,
        ));
        board.vias.push(crate::board::Via {
            position: (21.0, 26.0),
            size: 0.60,
            drill: 0.30,
            net_id: 1,
            is_locked: false,
        });

        let synced = client
            .pull_board_into_kicad(
                &board,
                Path::new("/tmp/test_board_vulcan.kicad_pcb"),
                Path::new("/tmp/test_board_vulcan.ses"),
            )
            .expect("NNG pull_board_into_kicad must succeed");
        // 1 updated footprint + 1 track + 1 arc + 1 via = 4 items synced
        assert_eq!(synced, 4);

        let commands = server_thread.join().expect("NNG server thread join");
        let _ = fs::remove_file(&sock_path);

        assert!(commands.iter().any(|c| c.ends_with(".Ping")));
        assert!(commands.iter().any(|c| c.ends_with(".GetOpenDocuments")));
        assert!(commands.iter().any(|c| c.ends_with(".GetNets")));
        assert!(commands.iter().any(|c| c.ends_with(".GetItems")));
        assert!(commands.iter().any(|c| c.ends_with(".BeginCommit")));
        assert!(commands.iter().any(|c| c.ends_with(".UpdateItems")));
        assert!(commands.iter().any(|c| c.ends_with(".DeleteItems")));
        assert!(commands.iter().any(|c| c.ends_with(".CreateItems")));
        assert!(commands.iter().any(|c| c.ends_with(".EndCommit")));
        assert!(commands.iter().any(|c| c.ends_with(".RefillZones")));
    }

    #[test]
    fn test_nng_protobuf_revert_document_on_save() {
        let Ok(lib) = NngLib::load() else {
            return;
        };
        let (Some(rep0_open), Some(listen_fn)) = (lib.rep0_open, lib.listen) else {
            return;
        };

        let sock_path = format!("/tmp/kicad_nng_revert_test_{}.sock", std::process::id());
        let _ = fs::remove_file(&sock_path);

        let mut srv_sock = NngSocket::default();
        unsafe {
            assert_eq!(rep0_open(&mut srv_sock), 0);
            if let Some(set_ms) = lib.set_ms {
                let _ = set_ms(srv_sock, NNG_OPT_RECVTIMEO.as_ptr() as *const _, 5_000);
                let _ = set_ms(srv_sock, NNG_OPT_SENDTIMEO.as_ptr() as *const _, 5_000);
            }
            let uri = format!("ipc://{}\0", sock_path);
            assert_eq!(listen_fn(srv_sock, uri.as_ptr() as *const _, std::ptr::null_mut(), 0), 0);
        }

        let srv_sock_id = srv_sock.id;
        let server_thread = thread::spawn(move || {
            let srv_lib = NngLib::load().unwrap();
            let s = NngSocket { id: srv_sock_id };
            let mut seen_commands = Vec::new();

            loop {
                let req_bytes = unsafe {
                    let mut buf_ptr: *mut u8 = std::ptr::null_mut();
                    let mut buf_len: usize = 0;
                    let rv = (srv_lib.recv)(s, &mut buf_ptr, &mut buf_len, NNG_FLAG_ALLOC);
                    if rv != 0 {
                        break;
                    }
                    let slice = std::slice::from_raw_parts(buf_ptr, buf_len).to_vec();
                    (srv_lib.free)(buf_ptr, buf_len);
                    slice
                };

                let mut cmd_type_url = String::new();
                proto_walk(&req_bytes, |tag, wt, val| {
                    if tag == 2 && wt == 2 {
                        let (turl, _) = proto_decode_any(val);
                        cmd_type_url = turl;
                    }
                });
                seen_commands.push(cmd_type_url.clone());

                let mut resp_payload = Vec::new();
                let mut resp_type_url = "type.googleapis.com/kiapi.common.types.CommandStatusResponse";

                if cmd_type_url.ends_with(".GetOpenDocuments") {
                    resp_type_url = "type.googleapis.com/kiapi.common.commands.GetOpenDocumentsResponse";
                    let mut doc = Vec::new();
                    proto_encode_varint_field(1, 3, &mut doc);
                    proto_encode_str(4, "revert_board.kicad_pcb", &mut doc);
                    proto_encode_bytes(1, &doc, &mut resp_payload);
                } else if cmd_type_url.ends_with(".RevertDocument") {
                    resp_type_url = "type.googleapis.com/kiapi.common.commands.RevertDocument";
                    proto_encode_varint_field(1, 1, &mut resp_payload);
                } else if cmd_type_url.ends_with(".RefillZones") {
                    resp_type_url = "type.googleapis.com/kiapi.common.types.CommandStatusResponse";
                    proto_encode_varint_field(1, 1, &mut resp_payload);
                } else {
                    proto_encode_varint_field(1, 1, &mut resp_payload);
                }

                let mut header = Vec::new();
                proto_encode_str(1, "test-kicad-token", &mut header);
                let mut status = Vec::new();
                proto_encode_varint_field(1, 1, &mut status); // AS_OK = 1
                let any_resp = proto_encode_any(resp_type_url, &resp_payload);
                let mut api_resp = Vec::new();
                proto_encode_bytes(1, &header, &mut api_resp);
                proto_encode_bytes(2, &status, &mut api_resp);
                proto_encode_bytes(3, &any_resp, &mut api_resp);

                unsafe {
                    let _ = (srv_lib.send)(s, api_resp.as_ptr(), api_resp.len(), 0);
                }

                if cmd_type_url.ends_with(".RefillZones") {
                    break;
                }
            }

            unsafe {
                let _ = (srv_lib.close)(s);
            }
            seen_commands
        });

        let mut client = KiCadIpcClient::connect(&sock_path).expect("NNG connect must succeed");
        assert!(client.is_connected());

        let mut board = crate::board::Board::default();
        board.file_path = "/tmp/revert_board.kicad_pcb".to_string();
        board.footprints.push(crate::board::Footprint {
            reference: "U1".to_string(),
            x: 10.0,
            y: 20.0,
            ..Default::default()
        });
        board.tracks.push(crate::board::TrackSegment {
            start: (10.0, 20.0),
            end: (15.0, 20.0),
            width: 0.25,
            layer: 0,
            net_id: 1,
            is_locked: false,
        });

        let synced = client
            .pull_board_into_kicad(
                &board,
                Path::new("/tmp/revert_board.kicad_pcb"),
                Path::new("/tmp/revert_board.ses"),
            )
            .expect("NNG pull_board_into_kicad with RevertDocument must succeed");
        // 1 footprint + 1 track = 2 items
        assert_eq!(synced, 2);

        let commands = server_thread.join().expect("NNG server thread join");
        let _ = fs::remove_file(&sock_path);

        assert!(commands.iter().any(|c| c.ends_with(".Ping")));
        assert!(commands.iter().any(|c| c.ends_with(".GetOpenDocuments")));
        assert!(commands.iter().any(|c| c.ends_with(".RevertDocument")));
        assert!(commands.iter().any(|c| c.ends_with(".RefillZones")));
        // Must NOT need BeginCommit / UpdateItems when RevertDocument succeeds
        assert!(!commands.iter().any(|c| c.ends_with(".BeginCommit")));
    }

    #[test]
    fn test_nng_kicad10_get_open_documents_nested_response() {
        let Ok(lib) = NngLib::load() else {
            return;
        };
        let (Some(rep0_open), Some(listen_fn)) = (lib.rep0_open, lib.listen) else {
            return;
        };

        let sock_path = format!("/tmp/kicad_nng_nested_doc_test_{}.sock", std::process::id());
        let _ = fs::remove_file(&sock_path);

        let mut srv_sock = NngSocket::default();
        unsafe {
            assert_eq!(rep0_open(&mut srv_sock), 0);
            if let Some(set_ms) = lib.set_ms {
                let _ = set_ms(srv_sock, NNG_OPT_RECVTIMEO.as_ptr() as *const _, 5_000);
                let _ = set_ms(srv_sock, NNG_OPT_SENDTIMEO.as_ptr() as *const _, 5_000);
            }
            let uri = format!("ipc://{}\0", sock_path);
            assert_eq!(listen_fn(srv_sock, uri.as_ptr() as *const _, std::ptr::null_mut(), 0), 0);
        }

        let srv_sock_id = srv_sock.id;
        let server_thread = thread::spawn(move || {
            let srv_lib = NngLib::load().unwrap();
            let s = NngSocket { id: srv_sock_id };

            let _req_bytes = unsafe {
                let mut buf_ptr: *mut u8 = std::ptr::null_mut();
                let mut buf_len: usize = 0;
                let rv = (srv_lib.recv)(s, &mut buf_ptr, &mut buf_len, NNG_FLAG_ALLOC);
                assert_eq!(rv, 0);
                let slice = std::slice::from_raw_parts(buf_ptr, buf_len).to_vec();
                (srv_lib.free)(buf_ptr, buf_len);
                slice
            };

            // KiCad 10 nested GetOpenDocumentsResponse:
            // Tag 2 contains Document wrapper, inside is Tag 1 with DocumentSpecifier
            let mut doc_spec = Vec::new();
            proto_encode_varint_field(1, 3, &mut doc_spec); // DOCTYPE_PCB = 3
            proto_encode_str(4, "nested_board.kicad_pcb", &mut doc_spec);

            let mut doc_wrapper = Vec::new();
            proto_encode_bytes(1, &doc_spec, &mut doc_wrapper);

            let mut get_docs_resp = Vec::new();
            proto_encode_bytes(2, &doc_wrapper, &mut get_docs_resp);

            let any_resp = proto_encode_any(
                "type.googleapis.com/kiapi.common.commands.GetOpenDocumentsResponse",
                &get_docs_resp,
            );

            let mut header = Vec::new();
            proto_encode_str(1, "mock-token", &mut header);
            let mut status = Vec::new();
            proto_encode_varint_field(1, 1, &mut status); // AS_OK = 1
            let mut resp_msg = Vec::new();
            proto_encode_bytes(1, &header, &mut resp_msg);
            proto_encode_bytes(2, &status, &mut resp_msg);
            proto_encode_bytes(3, &any_resp, &mut resp_msg);

            unsafe {
                assert_eq!((srv_lib.send)(s, resp_msg.as_ptr(), resp_msg.len(), 0), 0);
                let _ = (srv_lib.close)(s);
            }
        });

        let mut client = NngKiCadClient {
            lib: Box::leak(Box::new(NngLib::load().unwrap())),
            sock: NngSocket::default(),
            token: "mock-token".to_string(),
            client_name: "test-client".to_string(),
        };

        unsafe {
            let lib = client.lib;
            assert_eq!((lib.req0_open)(&mut client.sock), 0);
            let uri = format!("ipc://{}\0", sock_path);
            assert_eq!((lib.dial)(client.sock, uri.as_ptr() as *const _, std::ptr::null_mut(), 0), 0);
        }

        let board = crate::board::Board {
            file_path: "/tmp/nested_board.kicad_pcb".to_string(),
            ..crate::board::Board::default()
        };

        let doc = client
            .get_open_pcb_document(&board, Path::new("/tmp/nested_board_vulcan.kicad_pcb"))
            .expect("Must resolve nested KiCad 10 document specifier");

        let mut extracted_name = String::new();
        proto_walk(&doc, |tag, wt, val| {
            if tag == 4 && wt == 2 {
                extracted_name = std::str::from_utf8(val).unwrap().to_string();
            }
        });
        assert_eq!(extracted_name, "nested_board.kicad_pcb");

        server_thread.join().expect("Server thread join");
        let _ = fs::remove_file(&sock_path);
    }

    #[test]
    fn test_nng_worker_thread_drop_does_not_unload_libnng() {
        for i in 0..4 {
            let h = thread::spawn(move || {
                let mut slot = None;
                let board = crate::board::Board::default();
                let _ = push_saved_board_to_kicad(
                    &mut slot,
                    Some(&format!("/tmp/nonexistent_nng_stop_test_{}.sock", i)),
                    &board,
                    Path::new("/tmp/nonexistent.kicad_pcb"),
                    Path::new("/tmp/nonexistent.ses"),
                );
            });
            h.join().expect("Worker thread must exit cleanly without unloading libnng");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

