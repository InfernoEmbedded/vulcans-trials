#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 InfernoEmbedded — Vulcan's Trials App Icon Generator

"""Generates application icons for Linux, Windows, macOS, and KiCad PCM.

Produces:
- Vector SVG: assets/icon.svg and assets/icons/hicolor/scalable/apps/vulcans-trials.svg
- Raster PNGs (16x16 through 1024x1024): assets/icon_{size}x{size}.png
- FreeDesktop XDG icon hierarchy: assets/icons/hicolor/{size}x{size}/apps/vulcans-trials.png
- Windows Icon: assets/icon.ico (multi-resolution 16..256) and assets/vulcans-trials.rc
- macOS Icon: assets/icon.icns (multi-resolution 16..1024) and assets/Info.plist
- Linux Desktop Entry: assets/vulcans-trials.desktop
"""

import io
import os
import sys
from pathlib import Path
from PIL import Image
import cairosvg

# Import base vector SVG from kicad_plugin
REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = REPO_ROOT / "kicad_plugin"
sys.path.insert(0, str(PLUGIN_DIR))

from generate_icon import ICON_SVG

SIZES = [16, 24, 32, 48, 64, 128, 256, 512, 1024]
ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

DESKTOP_ENTRY = """[Desktop Entry]
Version=1.0
Type=Application
Name=Vulcan's Trials
GenericName=PCB Auto-Placer and Auto-Router
Comment=Vulkan GPU & Multi-Core CPU Auto-Placer and Auto-Router for KiCad
Exec=vulcans-trials %f
Icon=vulcans-trials
Terminal=false
Categories=Development;Engineering;Electronics;
MimeType=application/x-kicad-pcb;
StartupWMClass=com.infernoembedded.vulcans-trials
Keywords=kicad;pcb;autorouter;autoplacer;vulkan;eda;routing;placement;
"""

INFO_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>vulcans-trials</string>
    <key>CFBundleIdentifier</key>
    <string>com.infernoembedded.vulcans-trials</string>
    <key>CFBundleName</key>
    <string>Vulcan's Trials</string>
    <key>CFBundleDisplayName</key>
    <string>Vulcan's Trials</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
"""

WINDOWS_RC = """// SPDX-License-Identifier: GPL-3.0-only
// Vulcan's Trials — Windows Executable Icon Resource
1 ICON "icon.ico"
"""


def generate_all_icons(assets_dir: Path | None = None) -> None:
    if assets_dir is None:
        assets_dir = REPO_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    svg_path = assets_dir / "icon.svg"
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(ICON_SVG.strip() + "\n")
    print(f"[+] Wrote vector SVG: {svg_path}")

    # Generate multi-resolution raster PNG images
    png_images = {}
    for size in SIZES:
        png_bytes = cairosvg.svg2png(
            bytestring=ICON_SVG.encode("utf-8"),
            output_width=size,
            output_height=size,
        )
        png_path = assets_dir / f"icon_{size}x{size}.png"
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        png_images[size] = Image.open(io.BytesIO(png_bytes))
        print(f"[+] Generated {size}x{size} PNG: {png_path}")

    # Default icon.png (256x256)
    default_png = assets_dir / "icon.png"
    shutil_copy = png_images[256]
    shutil_copy.save(default_png)
    print(f"[+] Default app icon: {default_png}")

    # 1. Windows ICO
    ico_path = assets_dir / "icon.ico"
    png_images[256].save(
        ico_path,
        format="ICO",
        sizes=ICO_SIZES,
    )
    print(f"[+] Generated Windows multi-res icon: {ico_path}")

    rc_path = assets_dir / "vulcans-trials.rc"
    with open(rc_path, "w", encoding="utf-8") as f:
        f.write(WINDOWS_RC)
    print(f"[+] Generated Windows resource file: {rc_path}")

    # 2. macOS ICNS
    icns_path = assets_dir / "icon.icns"
    png_images[1024].save(icns_path, format="ICNS")
    print(f"[+] Generated macOS Apple Icon Image: {icns_path}")

    plist_path = assets_dir / "Info.plist"
    with open(plist_path, "w", encoding="utf-8") as f:
        f.write(INFO_PLIST)
    print(f"[+] Generated macOS Info.plist: {plist_path}")

    # 3. Linux XDG Desktop Entry & Hicolor icon hierarchy
    desktop_path = assets_dir / "vulcans-trials.desktop"
    with open(desktop_path, "w", encoding="utf-8") as f:
        f.write(DESKTOP_ENTRY)
    print(f"[+] Generated XDG desktop entry: {desktop_path}")

    hicolor_dir = assets_dir / "icons" / "hicolor"
    for size in [16, 24, 32, 48, 64, 128, 256, 512]:
        target_dir = hicolor_dir / f"{size}x{size}" / "apps"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "vulcans-trials.png"
        png_images[size].save(target_file)

    scalable_dir = hicolor_dir / "scalable" / "apps"
    scalable_dir.mkdir(parents=True, exist_ok=True)
    with open(scalable_dir / "vulcans-trials.svg", "w", encoding="utf-8") as f:
        f.write(ICON_SVG.strip() + "\n")
    print(f"[+] Generated full XDG hicolor icon theme hierarchy under {hicolor_dir}")


def main() -> None:
    generate_all_icons()
    print("\n[SUCCESS] All application icons and OS integration files generated successfully.")


if __name__ == "__main__":
    main()
