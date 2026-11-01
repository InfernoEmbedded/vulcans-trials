#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

"""Unit tests for Vulcan's Trials KiCad PCM packaging and Python integration."""

import json
import os
import shutil
import tempfile
import unittest
import zipfile

PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
import sys
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

import build_pcm
import generate_icon


class TestKiCadPCMPackage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_pcm_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_schema_file_exists_and_valid(self):
        schema_path = os.path.join(PLUGIN_DIR, "schema_v1.json")
        self.assertTrue(os.path.isfile(schema_path))
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.assertIn("definitions", schema)
        self.assertIn("Package", schema["definitions"])
        self.assertIn("PackageVersion", schema["definitions"])
        self.assertIn("Repository", schema["definitions"])

    def test_metadata_json_validation(self):
        meta_path = os.path.join(PLUGIN_DIR, "metadata.json")
        self.assertTrue(os.path.isfile(meta_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["identifier"], "com.infernoembedded.vulcans-trials")
        self.assertEqual(meta["type"], "plugin")
        self.assertEqual(meta["license"], "GPL-3.0")
        self.assertEqual(meta["category"], "Auto-Place & Auto-Route")

        v0 = meta["versions"][0]
        self.assertEqual(v0["kicad_version"], "8.0")
        self.assertEqual(v0["kicad_version_max"], "10.0")
        self.assertIn("linux", v0["platforms"])
        self.assertIn("windows", v0["platforms"])
        self.assertIn("macos", v0["platforms"])

        # Validate against schema
        build_pcm.validate_json_schema(meta, "Package")

    def test_icon_generation_and_sizes(self):
        generate_icon.generate_icons(self.temp_dir)
        res_dir = os.path.join(self.temp_dir, "resources")

        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "icon.png")))
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "icon.svg")))
        self.assertTrue(os.path.isfile(os.path.join(res_dir, "icon.png")))
        self.assertTrue(os.path.isfile(os.path.join(res_dir, "icon_32x32.png")))
        self.assertTrue(os.path.isfile(os.path.join(res_dir, "icon_64x64.png")))
        self.assertTrue(os.path.isfile(os.path.join(res_dir, "icon.svg")))

    def test_build_pcm_package_output_and_hashes(self):
        res = build_pcm.build_pcm_package(out_dir=self.temp_dir)
        zip_path = res["zip_path"]
        self.assertTrue(os.path.isfile(zip_path))

        with zipfile.ZipFile(zip_path, "r") as z:
            names = set(z.namelist())
            self.assertIn("metadata.json", names)
            self.assertIn("plugins/__init__.py", names)
            self.assertIn("plugins/icon.png", names)
            self.assertIn("plugins/icon.svg", names)
            self.assertIn("resources/icon.png", names)
            self.assertIn("resources/icon_32x32.png", names)
            self.assertIn("resources/icon_64x64.png", names)
            self.assertIn("resources/icon.svg", names)

        self.assertTrue(os.path.isfile(res["packages_path"]))
        self.assertTrue(os.path.isfile(res["repository_path"]))
        self.assertTrue(os.path.isfile(res["resources_path"]))
        self.assertTrue(os.path.isfile(res["submission_path"]))

        with open(res["packages_path"], "r", encoding="utf-8") as f:
            pkgs = json.load(f)
        build_pcm.validate_json_schema(pkgs, "PackageArray")

        with open(res["repository_path"], "r", encoding="utf-8") as f:
            repo = json.load(f)
        build_pcm.validate_json_schema(repo, "Repository")

        self.assertEqual(pkgs["packages"][0]["versions"][0]["download_sha256"], res["zip_sha256"])


if __name__ == "__main__":
    unittest.main()
