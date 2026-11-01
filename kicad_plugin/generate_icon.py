# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials Icon Generator

"""Generates the high-contrast 26x26 RGBA volcano toolbar icon and SVG."""

import os
import cairosvg

ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 26 26" width="26" height="26">
  <defs>
    <!-- Fire blast gradient -->
    <linearGradient id="fireGrad" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" stop-color="#D50000"/>
      <stop offset="25%" stop-color="#FF3D00"/>
      <stop offset="60%" stop-color="#FF9100"/>
      <stop offset="85%" stop-color="#FFD600"/>
      <stop offset="100%" stop-color="#FFFDE7"/>
    </linearGradient>

    <!-- Mountain basalt gradient -->
    <linearGradient id="rockGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#2D3238"/>
      <stop offset="35%" stop-color="#3E464F"/>
      <stop offset="70%" stop-color="#262B30"/>
      <stop offset="100%" stop-color="#181B1E"/>
    </linearGradient>

    <!-- Crater lava -->
    <radialGradient id="craterGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#FFFDE7"/>
      <stop offset="35%" stop-color="#FFD600"/>
      <stop offset="70%" stop-color="#FF3D00"/>
      <stop offset="100%" stop-color="#B71C1C"/>
    </radialGradient>
  </defs>

  <!-- Eruption Fire Cloud / Blast Plume -->
  <path d="M9.5,14 C7,11 6.5,7.5 9,4 C10.8,1.8 15.2,1.8 17,4 C19.5,7.5 19,11 16.5,14 Z" fill="url(#fireGrad)"/>
  <!-- Central white-hot jet core -->
  <path d="M11.5,14 Q10.5,7.5 12.5,3.2 Q13,2 13.5,3.2 Q15.5,7.5 14.5,14 Z" fill="#FFFFF0"/>

  <!-- Volcanic Sparks / Flying Molten Ejecta -->
  <circle cx="4.5" cy="6.5" r="1.0" fill="#FFD600"/>
  <circle cx="8" cy="2" r="1.0" fill="#FF9100"/>
  <circle cx="13" cy="0.8" r="1.1" fill="#FFFF8D"/>
  <circle cx="18" cy="2" r="1.0" fill="#FF9100"/>
  <circle cx="21.5" cy="6.5" r="1.0" fill="#FFD600"/>
  <circle cx="20" cy="10.5" r="0.9" fill="#FF3D00"/>
  <circle cx="6" cy="10.5" r="0.9" fill="#FF3D00"/>
  <circle cx="10" cy="0.9" r="0.7" fill="#FFEA00"/>
  <circle cx="16" cy="0.9" r="0.7" fill="#FFEA00"/>

  <!-- Dark Volcanic Cone (Basalt / Obsidian) -->
  <path d="M2,24 C3.8,24 6.2,23.5 8.8,18.8 C9.8,17 10.2,14.5 10.2,14.5 L15.8,14.5 C15.8,14.5 16.2,17 17.2,18.8 C19.8,23.5 22.2,24 24,24 Z"
        fill="url(#rockGrad)" stroke="#12151B" stroke-width="0.8" stroke-linejoin="round"/>

  <!-- Left flank illumination ridge (sharp definition on dark backgrounds) -->
  <path d="M10.2,14.5 C10.2,14.5 9.8,17 8.8,18.8 C6.5,22.8 4.2,23.6 2.5,23.9"
        stroke="#7B889B" stroke-width="0.75" fill="none" stroke-linecap="round"/>

  <!-- Crater Rim & Molten Caldera Lake -->
  <ellipse cx="13" cy="14.5" rx="3.2" ry="1.3" fill="#8B0000"/>
  <ellipse cx="13" cy="14.5" rx="2.7" ry="0.9" fill="url(#craterGrad)"/>

  <!-- Lava Streams Cascading Down Flanks -->
  <!-- Main cascading river -->
  <path d="M12.5,15 Q11.8,17.5 13.5,20 Q14.8,21.8 13.6,24"
        fill="none" stroke="#FF3D00" stroke-width="1.4" stroke-linecap="round"/>
  <path d="M12.5,15 Q11.8,17.5 13.5,20 Q14.8,21.8 13.6,24"
        fill="none" stroke="#FFEA00" stroke-width="0.6" stroke-linecap="round"/>
  <!-- Secondary branching stream -->
  <path d="M12,18.5 Q10.5,20.5 9.8,23.5"
        fill="none" stroke="#FF5722" stroke-width="0.9" stroke-linecap="round"/>
  <path d="M12,18.5 Q10.5,20.5 9.8,23.5"
        fill="none" stroke="#FFD600" stroke-width="0.4" stroke-linecap="round"/>
</svg>"""


def generate_icons(plugin_dir: str = None) -> None:
    if plugin_dir is None:
        plugin_dir = os.path.dirname(os.path.abspath(__file__))

    svg_path = os.path.join(plugin_dir, "icon.svg")
    png_path = os.path.join(plugin_dir, "icon.png")
    resources_dir = os.path.join(plugin_dir, "resources")
    os.makedirs(resources_dir, exist_ok=True)

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(ICON_SVG.strip() + "\n")

    res_svg_path = os.path.join(resources_dir, "icon.svg")
    with open(res_svg_path, "w", encoding="utf-8") as f:
        f.write(ICON_SVG.strip() + "\n")

    # 26x26 RGBA toolbar button icon
    cairosvg.svg2png(
        bytestring=ICON_SVG.encode("utf-8"),
        write_to=png_path,
        output_width=26,
        output_height=26,
    )

    # 64x64 PCM package manager primary icon
    res_64_path = os.path.join(resources_dir, "icon.png")
    cairosvg.svg2png(
        bytestring=ICON_SVG.encode("utf-8"),
        write_to=res_64_path,
        output_width=64,
        output_height=64,
    )

    # 64x64 explicit naming
    cairosvg.svg2png(
        bytestring=ICON_SVG.encode("utf-8"),
        write_to=os.path.join(resources_dir, "icon_64x64.png"),
        output_width=64,
        output_height=64,
    )

    # 32x32 compact PCM icon
    res_32_path = os.path.join(resources_dir, "icon_32x32.png"),
    cairosvg.svg2png(
        bytestring=ICON_SVG.encode("utf-8"),
        write_to=os.path.join(resources_dir, "icon_32x32.png"),
        output_width=32,
        output_height=32,
    )

    print(f"Generated {svg_path}")
    print(f"Generated {png_path} (26x26 RGBA)")
    print(f"Generated {res_svg_path}")
    print(f"Generated {res_64_path} (64x64 RGBA)")
    print(f"Generated {os.path.join(resources_dir, 'icon_32x32.png')} (32x32 RGBA)")


if __name__ == "__main__":
    generate_icons()

