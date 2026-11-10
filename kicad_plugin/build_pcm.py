#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials KiCad PCM Package & Repository Builder

"""Vulcan's Trials — KiCad Plugin & Content Manager (PCM) Package Builder.

Packages the Vulcan's Trials KiCad action plugin and optional native binaries
into a standardized KiCad PCM release zip archive, validates metadata against
the official schema (https://go.kicad.org/pcm/schemas/v1), and generates
repository index metadata (packages.json, repository.json, resources.zip)
ready for submission to the official KiCad Addons repository.
"""

import argparse
import hashlib
import json
import os
import stat
import sys
import time
import zipfile

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
SCHEMA_PATH = os.path.join(PLUGIN_DIR, "schema_v1.json")


def load_schema() -> dict:
    """Load KiCad PCM v1 JSON schema, preferring local cached file with online fallback."""
    if os.path.isfile(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback to fetching online if local file is missing
    import urllib.request
    req = urllib.request.Request(
        "https://go.kicad.org/pcm/schemas/v1",
        headers={"User-Agent": "VulcansTrialsPCMBuilder/1.0"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def validate_json_schema(data: dict, schema_type: str = "Package") -> None:
    """Validate JSON data against KiCad PCM v1 schema."""
    if not HAS_JSONSCHEMA:
        print(f"[WARN] jsonschema not installed; skipping schema validation for {schema_type}.")
        return

    full_schema = load_schema()
    if schema_type == "Package":
        validator_schema = dict(full_schema)
    elif schema_type == "PackageArray":
        validator_schema = {
            "$schema": full_schema.get("$schema", "http://json-schema.org/draft-07/schema#"),
            "$ref": "#/definitions/PackageArray",
            "definitions": full_schema["definitions"],
        }
    elif schema_type == "Repository":
        validator_schema = {
            "$schema": full_schema.get("$schema", "http://json-schema.org/draft-07/schema#"),
            "$ref": "#/definitions/Repository",
            "definitions": full_schema["definitions"],
        }
    else:
        raise ValueError(f"Unknown schema type: {schema_type}")

    jsonschema.validate(instance=data, schema=validator_schema)


def sha256_file(path: str) -> str:
    """Compute lowercase hex SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def ensure_icons(plugin_dir: str) -> dict[str, bytes]:
    """Ensure all required icons (26x26, 32x32, 64x64, SVG) are generated and return their bytes."""
    icon_png = os.path.join(plugin_dir, "icon.png")
    icon_svg = os.path.join(plugin_dir, "icon.svg")
    res_dir = os.path.join(plugin_dir, "resources")
    res_64_png = os.path.join(res_dir, "icon.png")
    res_32_png = os.path.join(res_dir, "icon_32x32.png")
    res_svg = os.path.join(res_dir, "icon.svg")

    if not all(os.path.exists(p) for p in [icon_png, icon_svg, res_64_png, res_32_png, res_svg]):
        from generate_icon import generate_icons
        generate_icons(plugin_dir)

    icons = {}
    with open(icon_png, "rb") as f:
        icons["toolbar_png"] = f.read()
    with open(icon_svg, "rb") as f:
        icons["toolbar_svg"] = f.read()
    with open(res_64_png, "rb") as f:
        icons["pcm_64_png"] = f.read()
    with open(res_32_png, "rb") as f:
        icons["pcm_32_png"] = f.read()
    with open(res_svg, "rb") as f:
        icons["pcm_svg"] = f.read()
    return icons


def build_pcm_package(
    out_dir: str | None = None,
    bin_path: str | None = None,
    platform: str | None = None,
    arch: str | None = None,
    version_override: str | None = None,
    download_url_prefix: str | None = None,
) -> dict:
    """Build the KiCad PCM package zip archive and repository index."""
    if out_dir is None:
        out_dir = os.path.join(PLUGIN_DIR, "pcm")
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(PLUGIN_DIR, "metadata.json")
    init_path = os.path.join(PLUGIN_DIR, "__init__.py")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Validate source metadata against schema
    validate_json_schema(meta, "Package")

    identifier = meta["identifier"]
    version = version_override or meta["versions"][0]["version"]

    icons = ensure_icons(PLUGIN_DIR)

    # Determine zip archive filename
    target_tag = ""
    if platform:
        target_tag += f"_{platform}"
    if arch:
        target_tag += f"_{arch}"
    zip_name = f"{identifier}_v{version}{target_tag}_pcm.zip"
    zip_path = os.path.join(out_dir, zip_name)

    # Base URL for package download
    if download_url_prefix is None:
        download_url_prefix = f"https://github.com/InfernoEmbedded/vulcans-trials/releases/download/v{version}"
    download_url = f"{download_url_prefix}/{zip_name}"

    # Build the standardized ZIP package
    uncompressed_size = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. metadata.json (at root of ZIP)
        # Note: Package internal metadata must have exactly 1 version object and
        # must NOT contain download_* or install_size fields.
        internal_meta = json.loads(json.dumps(meta))
        internal_meta.pop("$schema", None)
        v_internal = dict(internal_meta["versions"][0])
        v_internal["version"] = version
        v_internal.pop("download_sha256", None)
        v_internal.pop("download_size", None)
        v_internal.pop("download_url", None)
        v_internal.pop("install_size", None)
        if platform:
            v_internal["platforms"] = [platform]
        internal_meta["versions"] = [v_internal]
        meta_bytes = (json.dumps(internal_meta, indent=4) + "\n").encode("utf-8")
        zf.writestr("metadata.json", meta_bytes)
        uncompressed_size += len(meta_bytes)

        # 2. plugins/ directory
        with open(init_path, "rb") as f:
            init_bytes = f.read()
        zf.writestr("plugins/__init__.py", init_bytes)
        uncompressed_size += len(init_bytes)

        zf.writestr("plugins/icon.png", icons["toolbar_png"])
        uncompressed_size += len(icons["toolbar_png"])

        zf.writestr("plugins/icon.svg", icons["toolbar_svg"])
        uncompressed_size += len(icons["toolbar_svg"])

        # 3. Optional bundled native binary
        if bin_path and os.path.isfile(bin_path):
            bin_name = os.path.basename(bin_path)
            with open(bin_path, "rb") as f:
                bin_bytes = f.read()

            for arcname in [f"plugins/bin/{bin_name}", f"bin/{bin_name}"]:
                zinfo = zipfile.ZipInfo(arcname)
                zinfo.date_time = time.gmtime(time.time())[:6]
                zinfo.compress_type = zipfile.ZIP_DEFLATED
                # Set executable permission (0o755 | regular file)
                zinfo.external_attr = (0o755 | stat.S_IFREG) << 16
                zf.writestr(zinfo, bin_bytes)
                uncompressed_size += len(bin_bytes)

        # 4. resources/ directory (official KiCad PCM validator only permits /resources/icon.png)
        zf.writestr("resources/icon.png", icons["pcm_64_png"])
        uncompressed_size += len(icons["pcm_64_png"])

    zip_sha256 = sha256_file(zip_path)
    zip_size = os.path.getsize(zip_path)

    # Construct repository package entry
    pkg_entry = dict(meta)
    pkg_entry.pop("$schema", None)
    version_info = dict(meta["versions"][0])
    version_info["version"] = version
    version_info["download_sha256"] = zip_sha256
    version_info["download_size"] = zip_size
    version_info["download_url"] = download_url
    version_info["install_size"] = uncompressed_size

    if platform:
        version_info["platforms"] = [platform]

    pkg_entry["versions"] = [version_info]

    # Validate package entry against Package schema
    validate_json_schema(pkg_entry, "Package")

    # Write packages.json (merging versions if already present in out_dir)
    packages_path = os.path.join(out_dir, "packages.json")
    if os.path.isfile(packages_path):
        try:
            with open(packages_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            packages_list = existing_data.get("packages", [])
            matched = False
            for p in packages_list:
                if p.get("identifier") == identifier:
                    matched = True
                    # Replace matching version if same version and platforms
                    filtered_versions = [
                        v for v in p.get("versions", [])
                        if not (v.get("version") == version and v.get("platforms") == version_info.get("platforms"))
                    ]
                    filtered_versions.append(version_info)
                    p["versions"] = filtered_versions
                    pkg_entry["versions"] = filtered_versions
                    break
            if not matched:
                packages_list.append(pkg_entry)
            packages_data = {"packages": packages_list}
        except Exception:
            packages_data = {"packages": [pkg_entry]}
    else:
        packages_data = {"packages": [pkg_entry]}

    validate_json_schema(packages_data, "PackageArray")
    with open(packages_path, "w", encoding="utf-8") as f:
        json.dump(packages_data, f, indent=4)

    # Write package_submission.json (ready to submit to official KiCad repository)
    submission_path = os.path.join(out_dir, "package_submission.json")
    with open(submission_path, "w", encoding="utf-8") as f:
        json.dump(pkg_entry, f, indent=4)

    # Write resources.zip (for KiCad PCM repository remote icons)
    resources_path = os.path.join(out_dir, "resources.zip")
    with zipfile.ZipFile(resources_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{identifier}/icon.png", icons["pcm_64_png"])
        zf.writestr(f"{identifier}/icon_32x32.png", icons["pcm_32_png"])
        zf.writestr(f"{identifier}/icon.svg", icons["pcm_svg"])

    # Write repository.json
    now = int(time.time())
    now_utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))
    raw_gh_prefix = f"https://raw.githubusercontent.com/InfernoEmbedded/vulcans-trials/gh-pages"
    repo_data = {
        "$schema": "https://go.kicad.org/pcm/schemas/v1",
        "name": "InfernoEmbedded KiCad PCM Repository",
        "maintainer": meta["maintainer"],
        "packages": {
            "url": f"{raw_gh_prefix}/packages.json",
            "sha256": sha256_file(packages_path),
            "update_timestamp": now,
            "update_time_utc": now_utc,
        },
        "resources": {
            "url": f"{raw_gh_prefix}/resources.zip",
            "sha256": sha256_file(resources_path),
            "update_timestamp": now,
            "update_time_utc": now_utc,
        },
    }
    validate_json_schema(repo_data, "Repository")
    repo_path = os.path.join(out_dir, "repository.json")
    with open(repo_path, "w", encoding="utf-8") as f:
        json.dump(repo_data, f, indent=4)

    results = {
        "zip_path": zip_path,
        "zip_name": zip_name,
        "zip_sha256": zip_sha256,
        "zip_size": zip_size,
        "install_size": uncompressed_size,
        "download_url": download_url,
        "packages_path": packages_path,
        "repository_path": repo_path,
        "resources_path": resources_path,
        "submission_path": submission_path,
    }

    print("=" * 64)
    print("Vulcan's Trials — KiCad PCM Package Builder (InfernoEmbedded)")
    print("=" * 64)
    print(f"Archive:       {zip_path}")
    print(f"SHA-256:       {zip_sha256}")
    print(f"Download Size: {zip_size:,} bytes")
    print(f"Install Size:  {uncompressed_size:,} bytes")
    print(f"Download URL:  {download_url}")
    print(f"Packages:      {packages_path}")
    print(f"Repository:    {repo_path}")
    print(f"Resources:     {resources_path}")
    print(f"Submission:    {submission_path}")
    print("=" * 64)
    print("[SUCCESS] All artifacts created and schema-validated.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build KiCad PCM package (.zip) and repository metadata for Vulcan's Trials."
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(PLUGIN_DIR, "pcm"),
        help="Output directory for generated packages and repository index.",
    )
    parser.add_argument(
        "--bin",
        default=None,
        help="Optional path to vulcans-trials executable to bundle into PCM zip.",
    )
    parser.add_argument(
        "--platform",
        choices=["linux", "windows", "macos"],
        default=None,
        help="Target platform when bundling binary.",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="Target architecture (e.g. x86_64, aarch64).",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version override string (e.g. 0.1.0).",
    )
    parser.add_argument(
        "--download-url-prefix",
        default=None,
        help="Base URL prefix for release download URLs.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate metadata.json against KiCad PCM v1 schema without building.",
    )

    args = parser.parse_args()

    if args.validate_only:
        meta_path = os.path.join(PLUGIN_DIR, "metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        validate_json_schema(meta, "Package")
        print(f"[PASS] {meta_path} conforms to KiCad PCM v1 schema.")
        sys.exit(0)

    build_pcm_package(
        out_dir=args.out_dir,
        bin_path=args.bin,
        platform=args.platform,
        arch=args.arch,
        version_override=args.version,
        download_url_prefix=args.download_url_prefix,
    )


if __name__ == "__main__":
    main()
