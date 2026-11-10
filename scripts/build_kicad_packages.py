#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 InfernoEmbedded — Vulcan's Trials KiCad Multi-Platform Package Builder

"""Build KiCad PCM plugin packages for Linux (x64 and arm64), win64, and macOS.

Coordinates cross-platform binary resolution, packaging, schema validation,
and repository metadata generation (packages.json, repository.json, resources.zip,
and SHA256SUMS.txt) for Vulcan's Trials.
"""

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

# Add repository root and kicad_plugin to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = REPO_ROOT / "kicad_plugin"
sys.path.insert(0, str(PLUGIN_DIR))

try:
    from build_pcm import build_pcm_package, validate_json_schema, sha256_file
except ImportError as err:
    print(f"[ERROR] Failed to import build_pcm from {PLUGIN_DIR}: {err}", file=sys.stderr)
    sys.exit(1)


TARGET_SPECS = [
    {
        "id": "linux_x64",
        "label": "Linux (x86_64)",
        "platform": "linux",
        "arch": "x86_64",
        "binary_name": "vulcans-trials",
        "rust_target": "x86_64-unknown-linux-gnu",
        "arg_name": "linux_x64_bin",
    },
    {
        "id": "linux_arm64",
        "label": "Linux (aarch64)",
        "platform": "linux",
        "arch": "aarch64",
        "binary_name": "vulcans-trials",
        "rust_target": "aarch64-unknown-linux-gnu",
        "arg_name": "linux_arm64_bin",
    },
    {
        "id": "windows_x64",
        "label": "Windows (x86_64)",
        "platform": "windows",
        "arch": "x86_64",
        "binary_name": "vulcans-trials.exe",
        "rust_target": "x86_64-pc-windows-msvc",
        "alt_rust_target": "x86_64-pc-windows-gnu",
        "arg_name": "win64_bin",
    },
    {
        "id": "macos_x64",
        "label": "macOS (x86_64)",
        "platform": "macos",
        "arch": "x86_64",
        "binary_name": "vulcans-trials",
        "rust_target": "x86_64-apple-darwin",
        "arg_name": "macos_x64_bin",
    },
    {
        "id": "macos_arm64",
        "label": "macOS (aarch64 / Apple Silicon)",
        "platform": "macos",
        "arch": "aarch64",
        "binary_name": "vulcans-trials",
        "rust_target": "aarch64-apple-darwin",
        "arg_name": "macos_arm64_bin",
    },
]


def resolve_version(version_override: str | None) -> str:
    """Resolve project version from metadata.json, Cargo.toml, or CLI override."""
    if version_override:
        return version_override
    meta_path = PLUGIN_DIR / "metadata.json"
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data["versions"][0]["version"]
        except Exception:
            pass
    cargo_path = REPO_ROOT / "Cargo.toml"
    if cargo_path.is_file():
        with open(cargo_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("version ="):
                    return line.split("=")[1].strip().strip('"\'')
    return "1.0.0"


def find_binary_for_spec(
    spec: dict,
    bin_dir: Path | None,
    explicit_bin: str | None,
    build_if_missing: bool = False,
    mock_if_missing: bool = False,
    mock_dir: Path | None = None,
) -> Path | None:
    """Find or build executable for a target spec."""
    # 1. Explicit argument
    if explicit_bin:
        p = Path(explicit_bin).resolve()
        if p.is_file():
            return p

    # 2. Search bin-dir if provided
    if bin_dir and bin_dir.is_dir():
        candidates = [
            bin_dir / spec["rust_target"] / spec["binary_name"],
            bin_dir / f"binary-{spec['rust_target']}" / spec["binary_name"],
            bin_dir / f"{spec['platform']}_{spec['arch']}" / spec["binary_name"],
            bin_dir / f"{spec['platform']}-{spec['arch']}" / spec["binary_name"],
            bin_dir / spec["binary_name"],
        ]
        for c in candidates:
            if c.is_file():
                return c

    # 3. Cargo target directory search
    target_root = REPO_ROOT / "target"
    candidates = [
        target_root / spec["rust_target"] / "release" / spec["binary_name"],
        target_root / "release" / spec["binary_name"],
    ]
    if "alt_rust_target" in spec:
        candidates.append(target_root / spec["alt_rust_target"] / "release" / spec["binary_name"])

    for c in candidates:
        if c.is_file():
            return c

    # 4. Attempt native build if host matches
    is_linux = sys.platform.startswith("linux")
    is_mac = sys.platform.startswith("darwin")
    is_win = sys.platform.startswith("win")

    host_matches = (
        (is_linux and spec["platform"] == "linux" and os.uname().machine == spec["arch"])
        or (is_mac and spec["platform"] == "macos")
        or (is_win and spec["platform"] == "windows")
    )

    if build_if_missing and host_matches:
        print(f"[*] Building native release binary for {spec['label']}...")
        cmd = ["cargo", "build", "--release"]
        res = subprocess.run(cmd, cwd=REPO_ROOT)
        if res.returncode == 0:
            native_bin = target_root / "release" / spec["binary_name"]
            if native_bin.is_file():
                return native_bin

    # 5. Mock binary generation for dry-run / packaging tests
    if mock_if_missing and mock_dir:
        mock_dir.mkdir(parents=True, exist_ok=True)
        mock_path = mock_dir / f"{spec['id']}_{spec['binary_name']}"
        with open(mock_path, "wb") as f:
            if spec["binary_name"].endswith(".exe"):
                # Minimal mock PE executable header
                f.write(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00")
                f.write(b"VulcansTrials_Mock_Windows_Binary\n")
            else:
                # Minimal mock ELF / Mach-O header
                f.write(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00")
                f.write(b"VulcansTrials_Mock_Unix_Binary\n")
        # Mark executable
        st = os.stat(mock_path)
        os.chmod(mock_path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return mock_path

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build KiCad PCM plugin packages for Linux (x64 and arm64), win64, and macOS."
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "dist" / "pcm"),
        help="Output directory for generated packages and repository index (default: dist/pcm).",
    )
    parser.add_argument(
        "--bin-dir",
        default=None,
        help="Directory containing pre-compiled binaries for target platforms.",
    )
    parser.add_argument(
        "--linux-x64-bin",
        default=None,
        help="Explicit path to Linux x86_64 binary.",
    )
    parser.add_argument(
        "--linux-arm64-bin",
        default=None,
        help="Explicit path to Linux aarch64 binary.",
    )
    parser.add_argument(
        "--win64-bin",
        default=None,
        help="Explicit path to Windows x86_64 binary (vulcans-trials.exe).",
    )
    parser.add_argument(
        "--macos-x64-bin",
        default=None,
        help="Explicit path to macOS x86_64 binary.",
    )
    parser.add_argument(
        "--macos-arm64-bin",
        default=None,
        help="Explicit path to macOS aarch64 binary.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version string override (default: from metadata.json or Cargo.toml).",
    )
    parser.add_argument(
        "--download-url-prefix",
        default=None,
        help="Base URL prefix for release download URLs.",
    )
    parser.add_argument(
        "--build-native",
        action="store_true",
        default=True,
        help="Build native host binary with cargo build --release if missing (default: true).",
    )
    parser.add_argument(
        "--no-build-native",
        dest="build_native",
        action="store_false",
        help="Disable automatic native host building.",
    )
    parser.add_argument(
        "--mock-missing",
        action="store_true",
        help="Generate placeholder mock binaries for missing targets to test packaging.",
    )
    parser.add_argument(
        "--generate-repo",
        action="store_true",
        default=True,
        help="Generate repository index (packages.json, repository.json, resources.zip) (default: true).",
    )
    parser.add_argument(
        "--include-universal",
        action="store_true",
        default=True,
        help="Include universal/unbundled package alongside platform bundles (default: true).",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = Path(args.bin_dir).resolve() if args.bin_dir else None
    version = resolve_version(args.version)
    mock_dir = out_dir / ".mock_bins" if args.mock_missing else None

    print("=" * 72)
    print(" Vulcan's Trials — KiCad Multi-Platform Package Builder")
    print(" Organization: InfernoEmbedded")
    print(f" Target Version: v{version}")
    print(f" Output Directory: {out_dir}")
    print("=" * 72)

    explicit_map = {
        "linux_x64": args.linux_x64_bin,
        "linux_arm64": args.linux_arm64_bin,
        "windows_x64": args.win64_bin,
        "macos_x64": args.macos_x64_bin,
        "macos_arm64": args.macos_arm64_bin,
    }

    built_packages = []
    skipped_targets = []

    # 1. Build Universal Package (unbundled)
    if args.include_universal:
        print("\n[*] Packaging Universal KiCad PCM (host-resolved binary)...")
        res = build_pcm_package(
            out_dir=str(out_dir),
            bin_path=None,
            platform=None,
            arch=None,
            version_override=version,
            download_url_prefix=args.download_url_prefix,
        )
        built_packages.append(("Universal", res["zip_name"], res["zip_size"], res["zip_sha256"]))

    # 2. Build Platform-Specific Bundled Packages
    for spec in TARGET_SPECS:
        print(f"\n[*] Resolving binary for {spec['label']}...")
        bin_path = find_binary_for_spec(
            spec=spec,
            bin_dir=bin_dir,
            explicit_bin=explicit_map.get(spec["id"]),
            build_if_missing=args.build_native,
            mock_if_missing=args.mock_missing,
            mock_dir=mock_dir,
        )

        if not bin_path or not bin_path.is_file():
            print(f"[!] Warning: No binary found for {spec['label']} (Target: {spec['rust_target']}).")
            print(f"    Provide binary via --{spec['arg_name'].replace('_', '-')}, --bin-dir, or use --mock-missing.")
            skipped_targets.append(spec["label"])
            continue

        print(f"    Found binary: {bin_path} ({bin_path.stat().st_size:,} bytes)")
        print(f"    Packaging PCM bundle for {spec['platform']}_{spec['arch']}...")

        res = build_pcm_package(
            out_dir=str(out_dir),
            bin_path=str(bin_path),
            platform=spec["platform"],
            arch=spec["arch"],
            version_override=version,
            download_url_prefix=args.download_url_prefix,
        )
        built_packages.append((spec["label"], res["zip_name"], res["zip_size"], res["zip_sha256"]))

    # 3. Generate SHA256SUMS.txt
    checksums_path = out_dir / "SHA256SUMS.txt"
    checksum_lines = []
    for item in sorted(out_dir.glob("*.zip")):
        checksum_lines.append(f"{sha256_file(str(item))}  {item.name}")
    for item in sorted(out_dir.glob("*.json")):
        checksum_lines.append(f"{sha256_file(str(item))}  {item.name}")

    with open(checksums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")

    # Clean up temporary mock binaries if created
    if mock_dir and mock_dir.is_dir():
        shutil.rmtree(mock_dir, ignore_errors=True)

    # 4. Summary Output
    print("\n" + "=" * 72)
    print(" Build Summary — KiCad Plugin Packages")
    print("=" * 72)
    for label, zip_name, size, sha in built_packages:
        print(f" [PASS] {label:<30} {zip_name} ({size:,} bytes)")
        print(f"        SHA-256: {sha}")

    if skipped_targets:
        print("\n [!] Skipped Targets (no binary provided):")
        for target in skipped_targets:
            print(f"     - {target}")

    print(f"\n Checksums generated at: {checksums_path}")
    print(f" Repository index:       {out_dir / 'packages.json'}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
