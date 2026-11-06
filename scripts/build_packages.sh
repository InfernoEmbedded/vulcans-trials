#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 InfernoEmbedded — Vulcan's Trials KiCad Multi-Platform Package Builder

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Verify Python 3 is available
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 is required to build KiCad PCM packages." >&2
    exit 1
fi

# Forward all arguments directly to build_kicad_packages.py
exec python3 "${SCRIPT_DIR}/build_kicad_packages.py" "$@"
