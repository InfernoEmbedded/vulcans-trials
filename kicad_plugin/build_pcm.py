#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials PCM Repository Builder

import hashlib
import json
import os
import struct
import time
import zipfile
import zlib

PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
PCM_DIR = os.path.join(PLUGIN_DIR, "pcm")


def make_icon_png() -> bytes:
    """Generate a valid 64x64 RGBA PNG icon (orange/gold Vulkan flame motif)."""
    w, h = 64, 64
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # Filter type 0 (None)
        for x in range(w):
            dx, dy = x - 32, y - 32
            r2 = dx * dx + dy * dy
            if r2 < 180:
                raw.extend((255, 165, 30, 255))
            elif r2 < 780:
                raw.extend((210, 65, 20, 255))
            else:
                raw.extend((24, 26, 32, 255))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def build_pcm() -> None:
    os.makedirs(PCM_DIR, exist_ok=True)
    meta_path = os.path.join(PLUGIN_DIR, "metadata.json")
    init_path = os.path.join(PLUGIN_DIR, "__init__.py")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    identifier = meta["identifier"]
    version = meta["versions"][0]["version"]
    icon_bytes = make_icon_png()

    zip_name = f"{identifier}_v{version}_pcm.zip"
    zip_path = os.path.join(PCM_DIR, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(meta_path, "metadata.json")
        zf.write(init_path, "plugins/__init__.py")
        zf.writestr("resources/icon.png", icon_bytes)

    install_size = os.path.getsize(meta_path) + os.path.getsize(init_path) + len(icon_bytes)
    pkg_entry = dict(meta)
    pkg_entry.pop("$schema", None)
    pkg_entry["versions"] = [dict(meta["versions"][0])]
    pkg_entry["versions"][0].update(
        {
            "download_sha256": sha256_file(zip_path),
            "download_size": os.path.getsize(zip_path),
            "download_url": f"file://{zip_path}",
            "install_size": install_size,
        }
    )

    packages_path = os.path.join(PCM_DIR, "packages.json")
    with open(packages_path, "w", encoding="utf-8") as f:
        json.dump({"packages": [pkg_entry]}, f, indent=4)

    resources_path = os.path.join(PCM_DIR, "resources.zip")
    with zipfile.ZipFile(resources_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{identifier}/icon.png", icon_bytes)

    now = int(time.time())
    repo_data = {
        "$schema": "https://go.kicad.org/pcm/schemas/v1",
        "name": "InfernoEmbedded KiCad PCM Repository",
        "maintainer": meta["maintainer"],
        "packages": {
            "url": f"file://{packages_path}",
            "sha256": sha256_file(packages_path),
            "update_timestamp": now,
        },
        "resources": {
            "url": f"file://{resources_path}",
            "sha256": sha256_file(resources_path),
            "update_timestamp": now,
        },
    }
    repo_path = os.path.join(PCM_DIR, "repository.json")
    with open(repo_path, "w", encoding="utf-8") as f:
        json.dump(repo_data, f, indent=4)

    print(f"Built PCM package:    {zip_path}")
    print(f"Built PCM repository: {repo_path}")


if __name__ == "__main__":
    build_pcm()
