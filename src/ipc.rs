// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

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
        client: String,
        version: String,
    },
    /// Live displacement update for an existing footprint.
    MoveFootprint {
        reference: String,
        x: f64,
        y: f64,
        rotation: f64,
    },
    /// Direct addition of a routed copper track segment.
    AddTrack {
        start: (f64, f64),
        end: (f64, f64),
        width: f64,
        layer: String,
        net_id: usize,
    },
    /// Direct addition of a routed via.
    AddVia {
        position: (f64, f64),
        size: f64,
        drill: f64,
        net_id: usize,
    },
    /// Clear all dynamic autorouted tracks and vias before rerouting pass.
    ClearTracks,
    /// Atomically flush pending canvas mutations and trigger 60 FPS viewport repaint.
    CommitBatch,
}

pub type IpcCommand = IpcMessage;

/// Client connection managing the KiCad 10 IPC socket connection, stream buffering,
/// delta calculation, and ~60 FPS rate-limited throttling.
pub struct KiCadIpcClient {
    #[cfg(unix)]
    stream: Option<UnixStream>,
    socket_path: Option<PathBuf>,
    is_connected: bool,

    // Throttling configuration (default 16ms for ~60 FPS viewport refresh)
    throttle_interval: Duration,
    last_flush_time: Instant,

    // Footprint displacement coalescing buffer: reference -> (x, y, rotation)
    pending_moves: HashMap<String, (f64, f64, f64)>,
    // Last sent positions to suppress redundant sub-millimeter noise
    last_sent_moves: HashMap<String, (f64, f64, f64)>,

    // Queue of pending copper mutations
    pending_tracks: Vec<IpcMessage>,
    pending_vias: Vec<IpcMessage>,

    // Telemetry & metrics
    total_messages_sent: usize,
    total_batches_committed: usize,
}

impl std::fmt::Debug for KiCadIpcClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KiCadIpcClient")
            .field("socket_path", &self.socket_path)
            .field("is_connected", &self.is_connected)
            .field("throttle_interval", &self.throttle_interval)
            .field("pending_moves", &self.pending_moves.len())
            .field("pending_tracks", &self.pending_tracks.len())
            .field("pending_vias", &self.pending_vias.len())
            .field("total_messages_sent", &self.total_messages_sent)
            .field("total_batches_committed", &self.total_batches_committed)
            .finish()
    }
}

impl KiCadIpcClient {
    /// Creates a new unconnected IPC client instance with default 60 FPS (16ms) throttling.
    pub fn new() -> Self {
        Self {
            #[cfg(unix)]
            stream: None,
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

    /// Connects to a specific Unix domain socket path and performs initial handshake.
    pub fn connect<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let p = path.as_ref().to_path_buf();
        #[cfg(unix)]
        {
            let stream = UnixStream::connect(&p)?;
            stream.set_write_timeout(Some(Duration::from_millis(250)))?;
            stream.set_read_timeout(Some(Duration::from_millis(250)))?;

            let mut client = Self {
                stream: Some(stream),
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
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "KiCad IPC domain sockets are supported on Unix platforms",
            ))
        }
    }

    /// Attempts to connect to a custom path if given, or automatically discovers an active KiCad 10 IPC socket.
    pub fn connect_or_discover(custom_path: Option<&str>) -> Option<Self> {
        if let Some(cp) = custom_path {
            if !cp.trim().is_empty() {
                match Self::connect(cp.trim()) {
                    Ok(client) => return Some(client),
                    Err(e) => {
                        eprintln!("[KiCad-IPC] Failed to connect to specified socket '{}': {}", cp, e);
                        return None;
                    }
                }
            }
        }

        if let Some(discovered) = Self::discover_socket_path() {
            match Self::connect(&discovered) {
                Ok(client) => {
                    println!("[KiCad-IPC] Connected to active KiCad 10 socket at {}", discovered.display());
                    Some(client)
                }
                Err(e) => {
                    eprintln!("[KiCad-IPC] Failed to connect to discovered socket '{}': {}", discovered.display(), e);
                    None
                }
            }
        } else {
            None
        }
    }

    /// Discovers active KiCad 10 IPC Unix socket paths on Linux/macOS.
    /// Checks:
    /// 1. $KICAD_IPC_PATH or $KICAD_IPC_SOCKET env vars
    /// 2. $XDG_RUNTIME_DIR/kicad/kicad_ipc_*.sock
    /// 3. /tmp/kicad_ipc_*.sock
    pub fn discover_socket_path() -> Option<PathBuf> {
        // 1. Environment variable override
        if let Ok(env_path) = env::var("KICAD_IPC_PATH").or_else(|_| env::var("KICAD_IPC_SOCKET")) {
            let p = PathBuf::from(env_path);
            if p.exists() {
                return Some(p);
            }
        }

        // 2. $XDG_RUNTIME_DIR/kicad/
        if let Ok(runtime_dir) = env::var("XDG_RUNTIME_DIR") {
            let kicad_dir = Path::new(&runtime_dir).join("kicad");
            if let Some(found) = scan_dir_for_socket(&kicad_dir) {
                return Some(found);
            }
        }

        // 3. /tmp/
        if let Some(found) = scan_dir_for_socket(Path::new("/tmp")) {
            return Some(found);
        }

        None
    }

    /// Returns whether the client is currently connected.
    pub fn is_connected(&self) -> bool {
        self.is_connected
    }

    /// Returns the socket path connected to, if any.
    pub fn socket_path(&self) -> Option<&Path> {
        self.socket_path.as_deref()
    }

    /// Sets the minimum throttling interval between flushes (e.g. 16.6ms for 60 FPS).
    pub fn set_throttle_interval(&mut self, interval: Duration) {
        self.throttle_interval = interval;
    }

    /// Returns current throttling interval.
    pub fn throttle_interval(&self) -> Duration {
        self.throttle_interval
    }

    /// Total messages sent and total batches committed.
    pub fn stats(&self) -> (usize, usize) {
        (self.total_messages_sent, self.total_batches_committed)
    }

    /// Sends the initial Handshake message.
    pub fn send_handshake(&mut self) -> std::io::Result<()> {
        let msg = IpcMessage::Handshake {
            client: "Vulcan's Trials".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        self.send_raw_message(&msg)
    }

    /// Directly serializes and writes an IPC message as a single line JSON over the socket.
    fn send_raw_message(&mut self, msg: &IpcMessage) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            if let Some(ref mut stream) = self.stream {
                let json = serde_json::to_string(msg).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                })?;
                stream.write_all(json.as_bytes())?;
                stream.write_all(b"\n")?;
                self.total_messages_sent += 1;
                Ok(())
            } else {
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
    pub fn queue_move_footprint(&mut self, reference: impl Into<String>, x: f64, y: f64, rotation: f64) {
        self.pending_moves.insert(reference.into(), (x, y, rotation));
    }

    /// Queues a track segment creation.
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
    pub fn queue_add_via(&mut self, position: (f64, f64), size: f64, drill: f64, net_id: usize) {
        self.pending_vias.push(IpcMessage::AddVia {
            position,
            size,
            drill,
            net_id,
        });
    }

    /// Sends ClearTracks command to clear autorouted copper before rerouting.
    pub fn clear_tracks(&mut self) -> std::io::Result<()> {
        if !self.is_connected {
            return Ok(());
        }
        self.send_raw_message(&IpcMessage::ClearTracks)
    }

    /// Stream footprint move with delta threshold check and 60 FPS interval throttling.
    /// Returns true if a batch flush was triggered.
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

        for via in &board.vias {
            self.queue_add_via(via.position, via.size, via.drill, via.net_id);
            count += 1;
        }

        self.commit_batch()?;
        Ok(count)
    }

    /// Flushes all queued moves, tracks, and vias, then issues CommitBatch and flushes socket writer.
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
                stream.flush()?;
            }
        }

        self.last_flush_time = Instant::now();
        self.total_batches_committed += 1;
        Ok(total_queued)
    }
}

/// Helper function to scan a directory for KiCad IPC sockets.
fn scan_dir_for_socket(dir: &Path) -> Option<PathBuf> {
    if !dir.is_dir() {
        return None;
    }
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        let fname = p.file_name().and_then(|f| f.to_str()).unwrap_or("");
        if (fname.starts_with("kicad_ipc") || fname.contains("kicad")) && fname.ends_with(".sock") {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::os::unix::net::UnixListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;

    #[test]
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
}
