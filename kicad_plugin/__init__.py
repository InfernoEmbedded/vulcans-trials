# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

"""Vulcan's Trials — KiCad ActionPlugin & IPC Sync Integration.

Registers a toolbar button in KiCad Pcbnew, resolves the vulcans-trials
binary (bundled in PCM package or installed on system), connects to
KiCad IPC socket streaming when available, and launches Vulcan's Trials.
"""

import os
import shutil
import socket
import subprocess
import sys

try:
    import pcbnew
    import wx
    HAS_GUI = True
except ImportError:
    HAS_GUI = False


def find_binary() -> str | None:
    """Resolve the vulcans-trials executable across bundled, PATH, and build locations."""
    is_win = sys.platform.startswith("win")
    exe_name = "vulcans-trials.exe" if is_win else "vulcans-trials"
    plugin_dir = os.path.dirname(os.path.realpath(__file__))

    # 1. Bundled inside PCM package
    bundled_candidates = [
        os.path.join(plugin_dir, "bin", exe_name),
        os.path.join(plugin_dir, "..", "bin", exe_name),
        os.path.join(plugin_dir, exe_name),
    ]
    for c in bundled_candidates:
        if c and os.path.isfile(c) and (is_win or os.access(c, os.X_OK)):
            return os.path.abspath(c)

    # 2. Environment variable override
    env_bin = os.environ.get("VULCANS_TRIALS_BIN")
    if env_bin and os.path.isfile(env_bin) and (is_win or os.access(env_bin, os.X_OK)):
        return os.path.abspath(env_bin)

    # 3. System PATH
    path_bin = shutil.which(exe_name) or shutil.which("vulcans-trials")
    if path_bin:
        return os.path.abspath(path_bin)

    # 4. Local workspace / target builds & cargo home
    workspace_candidates = [
        os.path.abspath(os.path.join(plugin_dir, "..", "target", "release", exe_name)),
        os.path.expanduser(f"~/watt-home/src/vulcans-trials/target/release/{exe_name}"),
        os.path.expanduser(f"~/.cargo/bin/{exe_name}"),
        "/usr/local/bin/vulcans-trials",
        "/usr/bin/vulcans-trials",
    ]
    if is_win:
        workspace_candidates.extend([
            os.path.expandvars(r"%LOCALAPPDATA%\vulcans-trials\vulcans-trials.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\vulcans-trials\vulcans-trials.exe"),
        ])

    for c in workspace_candidates:
        if c and os.path.isfile(c) and (is_win or os.access(c, os.X_OK)):
            return os.path.abspath(c)

    return None


def get_ipc_endpoint() -> str | None:
    """Detect available KiCad 9/10 IPC streaming socket or endpoint."""
    # Check environment variable
    env_ipc = os.environ.get("KICAD_IPC_PATH") or os.environ.get("KICAD_IPC_SOCKET")
    if env_ipc:
        return env_ipc

    # Check default Unix domain sockets
    if not sys.platform.startswith("win"):
        pid = os.getpid()
        candidates = [
            f"/tmp/kicad_ipc_{pid}.sock",
            "/tmp/kicad_ipc.sock",
            os.path.expanduser("~/.kicad_ipc.sock"),
        ]
        for sock in candidates:
            if os.path.exists(sock):
                return sock

    # Check Windows named pipe
    if sys.platform.startswith("win"):
        pipe_name = r"\\.\pipe\kicad_ipc"
        if os.path.exists(pipe_name):
            return pipe_name

    # Check kicad.ipc module if present in KiCad 10
    try:
        import kicad.ipc  # type: ignore
        if hasattr(kicad.ipc, "get_socket_path"):
            return kicad.ipc.get_socket_path()
    except (ImportError, AttributeError):
        pass

    return None


def connect_ipc(endpoint: str | None = None) -> bool:
    """Test connection to KiCad IPC streaming socket."""
    if not endpoint:
        endpoint = get_ipc_endpoint()
    if not endpoint:
        return False

    try:
        if sys.platform.startswith("win") or "://" in endpoint or ":" in endpoint:
            # TCP address host:port
            host, port_str = endpoint.rsplit(":", 1)
            port = int(port_str)
            with socket.create_connection((host, port), timeout=0.5):
                return True
        else:
            # Unix domain socket
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(endpoint)
                return True
    except Exception:
        # IPC endpoint exists but connection failed or not ready
        return False


if HAS_GUI:
    class VulcansTrialsPlugin(pcbnew.ActionPlugin):
        def defaults(self):
            self.name = "Vulcan's Trials"
            self.category = "Auto-Place & Auto-Route"
            self.description = "Launch Vulcan's Trials GPU-accelerated auto-placer and auto-router"
            self.show_toolbar_button = True
            self.icon_file_name = os.path.join(os.path.dirname(__file__), "icon.png")

        def Run(self):
            board = pcbnew.GetBoard()
            board_path = board.GetFileName() if board else ""
            if not board_path or not os.path.exists(board_path):
                wx.MessageBox(
                    "Please save your KiCad board (.kicad_pcb) before launching Vulcan's Trials.",
                    "Vulcan's Trials — InfernoEmbedded",
                    wx.OK | wx.ICON_INFORMATION,
                )
                return

            binary = find_binary()
            if not binary:
                wx.MessageBox(
                    "Could not find 'vulcans-trials' binary.\n\n"
                    "Install the full PCM package containing bundled binaries, or build from source with:\n"
                    "  cargo build --release\n"
                    "or set the VULCANS_TRIALS_BIN environment variable.",
                    "Vulcan's Trials — Binary Not Found",
                    wx.OK | wx.ICON_ERROR,
                )
                return

            out_path = os.path.splitext(board_path)[0] + "_vulcan.kicad_pcb"
            cmd = [binary, "--input", board_path, "--output", out_path]

            ipc_ep = get_ipc_endpoint()
            if ipc_ep:
                connect_ipc(ipc_ep)
                cmd.extend(["--ipc", ipc_ep])

            subprocess.Popen(cmd)


try:
    if HAS_GUI:
        VulcansTrialsPlugin().register()
except Exception:
    # Allows safe importing in headless/unit-test environments where KiCad's GUI program is not initialized
    pass
