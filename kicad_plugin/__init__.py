# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

import os
import shutil
import subprocess
import pcbnew
import wx


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

        plugin_dir = os.path.dirname(os.path.realpath(__file__))
        candidates = [
            shutil.which("vulcans-trials"),
            os.path.abspath(os.path.join(plugin_dir, "..", "target", "release", "vulcans-trials")),
            os.path.expanduser("~/watt-home/src/vulcans-trials/target/release/vulcans-trials"),
            os.path.expanduser("~/.cargo/bin/vulcans-trials"),
            "/usr/local/bin/vulcans-trials",
            "/usr/bin/vulcans-trials",
        ]
        binary = next((c for c in candidates if c and os.path.isfile(c) and os.access(c, os.X_OK)), None)

        if not binary:
            wx.MessageBox(
                "Could not find 'vulcans-trials' binary. Build with 'cargo build --release' in ~/watt-home/src/vulcans-trials.",
                "Vulcan's Trials — Error",
                wx.OK | wx.ICON_ERROR,
            )
            return

        out_path = os.path.splitext(board_path)[0] + "_vulcan.kicad_pcb"
        subprocess.Popen([binary, "--input", board_path, "--output", out_path])


try:
    VulcansTrialsPlugin().register()
except Exception:
    # Allows safe importing in headless/unit-test environments where KiCad's GUI program is not initialized
    pass
