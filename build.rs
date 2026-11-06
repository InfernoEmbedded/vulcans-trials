// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 InfernoEmbedded — Vulcan's Trials Build Script

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=assets/vulcans-trials.rc");
    println!("cargo:rerun-if-changed=assets/icon.ico");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }

    let out_dir = match env::var("OUT_DIR") {
        Ok(d) => d,
        Err(_) => return,
    };

    let rc_path = Path::new("assets/vulcans-trials.rc");
    if !rc_path.exists() {
        return;
    }

    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_env == "msvc" {
        let res_path = format!("{}/vulcans-trials.res", out_dir);
        let status = Command::new("rc.exe")
            .args(["/fo", &res_path, "assets/vulcans-trials.rc"])
            .status();
        if let Ok(s) = status {
            if s.success() {
                println!("cargo:rustc-link-arg={}", res_path);
            }
        }
    } else {
        // GNU / MinGW target
        let obj_path = format!("{}/vulcans-trials.o", out_dir);
        let windres = env::var("WINDRES").unwrap_or_else(|_| "windres".to_string());
        let status = Command::new(&windres)
            .args(["assets/vulcans-trials.rc", "-o", &obj_path])
            .status();
        if let Ok(s) = status {
            if s.success() {
                println!("cargo:rustc-link-arg={}", obj_path);
            }
        }
    }
}
