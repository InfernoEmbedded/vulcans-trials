#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials PCM Repository Builder

import hashlib
import json
import os
import time
import zipfile

PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
PCM_DIR = os.path.join(PLUGIN_DIR, "pcm")


def get_icon_bytes() -> bytes:
    """Return RGBA PNG icon bytes from icon.png, generating if missing."""
    icon_path = os.path.join(PLUGIN_DIR, "icon.png")
    if not os.path.exists(icon_path):
        from generate_icon import generate_icons
        generate_icons(PLUGIN_DIR)
    with open(icon_path, "rb") as f:
        return f.read()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def build_pcm() -> None:
    os.makedirs(PCM_DIR, exist_ok=True)
    meta_path = os.path.join(PLUGIN_DIR, "metadata.json")
    init_path = os.path.join(PLUGIN_DIR, "__init__.py")
    icon_path = os.path.join(PLUGIN_DIR, "icon.png")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    identifier = meta["identifier"]
    version = meta["versions"][0]["version"]
    icon_bytes = get_icon_bytes()

    zip_name = f"{identifier}_v{version}_pcm.zip"
    zip_path = os.path.join(PCM_DIR, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(meta_path, "metadata.json")
        zf.write(init_path, "plugins/__init__.py")
        zf.write(icon_path, "plugins/icon.png")
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
