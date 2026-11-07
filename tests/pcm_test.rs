// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Alastair D'Silva — Vulcan's Trials

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn get_workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn create_temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vulcans_pcm_test_{}_{}", std::process::id(), tag));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("Failed to create temp directory");
    dir
}

fn python_cmd() -> &'static str {
    if cfg!(windows) {
        "python"
    } else {
        "python3"
    }
}

#[test]
fn test_metadata_json_syntax_and_schema_compliance() {
    let root = get_workspace_root();
    let meta_path = root.join("kicad_plugin").join("metadata.json");
    assert!(meta_path.exists(), "metadata.json must exist at {}", meta_path.display());

    let content = fs::read_to_string(&meta_path).expect("Failed to read metadata.json");
    let v: Value = serde_json::from_str(&content).expect("metadata.json must be valid JSON");

    // 1. Mandatory top-level fields
    assert_eq!(v["$schema"].as_str(), Some("https://go.kicad.org/pcm/schemas/v1"));
    assert_eq!(v["name"].as_str(), Some("Vulcan's Trials"));
    assert_eq!(v["identifier"].as_str(), Some("com.infernoembedded.vulcans-trials"));
    assert_eq!(v["type"].as_str(), Some("plugin"));
    assert_eq!(v["category"].as_str(), Some("Auto-Place & Auto-Route"));
    assert_eq!(v["license"].as_str(), Some("GPL-3.0"));
    assert_eq!(v["author"]["name"].as_str(), Some("InfernoEmbedded"));
    assert_eq!(v["maintainer"]["name"].as_str(), Some("InfernoEmbedded"));

    // 2. KiCad 8.0, 9.0, and 10.0 version range compatibility
    let versions = v["versions"].as_array().expect("versions must be an array");
    assert!(!versions.is_empty(), "versions must contain at least one version entry");

    let v0 = &versions[0];
    assert_eq!(v0["version"].as_str(), Some("1.0.0"));
    assert_eq!(v0["status"].as_str(), Some("stable"));
    assert_eq!(v0["kicad_version"].as_str(), Some("8.0"));
    assert_eq!(v0["kicad_version_max"].as_str(), Some("10.0"));

    let platforms = v0["platforms"].as_array().expect("platforms must be an array");
    let plat_strings: Vec<&str> = platforms.iter().filter_map(|p| p.as_str()).collect();
    assert!(plat_strings.contains(&"linux"));
    assert!(plat_strings.contains(&"macos"));
    assert!(plat_strings.contains(&"windows"));

    // 3. Schema validation via build_pcm.py --validate-only
    let script_path = root.join("kicad_plugin").join("build_pcm.py");
    let output = Command::new(python_cmd())
        .arg(&script_path)
        .arg("--validate-only")
        .current_dir(&root)
        .output()
        .expect("Failed to execute build_pcm.py --validate-only");

    assert!(
        output.status.success(),
        "build_pcm.py --validate-only failed:\nSTDOUT:\n{}\nSTDERR:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_build_pcm_produces_valid_zip_archive_and_repo_index() {
    let root = get_workspace_root();
    let temp_dir = create_temp_dir("build");
    let script_path = root.join("kicad_plugin").join("build_pcm.py");

    let output = Command::new(python_cmd())
        .arg(&script_path)
        .arg("--out-dir")
        .arg(&temp_dir)
        .current_dir(&root)
        .output()
        .expect("Failed to execute build_pcm.py");

    assert!(
        output.status.success(),
        "build_pcm.py failed:\nSTDOUT:\n{}\nSTDERR:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // 1. Verify ZIP archive exists and has non-zero size
    let zip_path = temp_dir.join("com.infernoembedded.vulcans-trials_v1.0.0_pcm.zip");
    assert!(zip_path.exists(), "PCM zip archive must exist at {}", zip_path.display());
    let zip_size = fs::metadata(&zip_path).unwrap().len();
    assert!(zip_size > 1000, "Zip size should be > 1KB (got {} bytes)", zip_size);

    // 2. Verify ZIP archive internal directory layout
    let unzip_check = Command::new(python_cmd())
        .arg("-c")
        .arg(format!(
            "import zipfile; z = zipfile.ZipFile(r'{}'); print(sorted(z.namelist()))",
            zip_path.display()
        ))
        .output()
        .expect("Failed to inspect zip archive with python zipfile");

    assert!(unzip_check.status.success());
    let file_list_str = String::from_utf8_lossy(&unzip_check.stdout);
    assert!(file_list_str.contains("'metadata.json'"), "Missing metadata.json in zip");
    assert!(file_list_str.contains("'plugins/__init__.py'"), "Missing plugins/__init__.py in zip");
    assert!(file_list_str.contains("'plugins/icon.png'"), "Missing plugins/icon.png in zip");
    assert!(file_list_str.contains("'plugins/icon.svg'"), "Missing plugins/icon.svg in zip");
    assert!(file_list_str.contains("'resources/icon.png'"), "Missing resources/icon.png in zip");
    assert!(file_list_str.contains("'resources/icon_32x32.png'"), "Missing resources/icon_32x32.png in zip");
    assert!(file_list_str.contains("'resources/icon_64x64.png'"), "Missing resources/icon_64x64.png in zip");
    assert!(file_list_str.contains("'resources/icon.svg'"), "Missing resources/icon.svg in zip");

    // 3. Verify packages.json repository index
    let pkg_json_path = temp_dir.join("packages.json");
    assert!(pkg_json_path.exists(), "packages.json must be created");
    let pkg_content = fs::read_to_string(&pkg_json_path).unwrap();
    let pkg_val: Value = serde_json::from_str(&pkg_content).unwrap();

    let packages = pkg_val["packages"].as_array().expect("packages must be an array");
    assert_eq!(packages.len(), 1);
    let pkg = &packages[0];
    assert_eq!(pkg["identifier"].as_str(), Some("com.infernoembedded.vulcans-trials"));

    let versions = pkg["versions"].as_array().unwrap();
    assert_eq!(versions.len(), 1);
    let v_entry = &versions[0];
    assert_eq!(v_entry["version"].as_str(), Some("1.0.0"));
    assert_eq!(v_entry["kicad_version"].as_str(), Some("8.0"));
    assert_eq!(v_entry["kicad_version_max"].as_str(), Some("10.0"));
    assert_eq!(v_entry["download_size"].as_u64(), Some(zip_size));

    let sha256 = v_entry["download_sha256"].as_str().expect("download_sha256 must be string");
    assert_eq!(sha256.len(), 64, "SHA-256 must be 64 characters");
    assert!(sha256.chars().all(|c| c.is_ascii_hexdigit()), "SHA-256 must be hex");

    // 4. Verify repository.json
    let repo_json_path = temp_dir.join("repository.json");
    assert!(repo_json_path.exists(), "repository.json must be created");
    let repo_content = fs::read_to_string(&repo_json_path).unwrap();
    let repo_val: Value = serde_json::from_str(&repo_content).unwrap();
    assert_eq!(repo_val["name"].as_str(), Some("InfernoEmbedded KiCad PCM Repository"));
    assert!(repo_val["packages"]["sha256"].as_str().is_some());
    assert!(repo_val["resources"]["sha256"].as_str().is_some());

    // 5. Verify resources.zip
    let res_zip_path = temp_dir.join("resources.zip");
    assert!(res_zip_path.exists(), "resources.zip must be created");
    assert!(fs::metadata(&res_zip_path).unwrap().len() > 100);

    // 6. Verify package_submission.json
    let sub_json_path = temp_dir.join("package_submission.json");
    assert!(sub_json_path.exists(), "package_submission.json must be created");

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_build_pcm_bundled_binary_packaging() {
    let root = get_workspace_root();
    let temp_dir = create_temp_dir("bundled");
    let script_path = root.join("kicad_plugin").join("build_pcm.py");
    let bin_path = root.join("target").join("release").join("vulcans-trials");

    if !bin_path.exists() {
        // Skip if release binary not compiled yet
        return;
    }

    let output = Command::new(python_cmd())
        .arg(&script_path)
        .arg("--out-dir")
        .arg(&temp_dir)
        .arg("--bin")
        .arg(&bin_path)
        .arg("--platform")
        .arg("linux")
        .arg("--arch")
        .arg("x86_64")
        .current_dir(&root)
        .output()
        .expect("Failed to execute build_pcm.py with bundled binary");

    assert!(output.status.success());

    let zip_path = temp_dir.join("com.infernoembedded.vulcans-trials_v1.0.0_linux_x86_64_pcm.zip");
    assert!(zip_path.exists());

    let unzip_check = Command::new(python_cmd())
        .arg("-c")
        .arg(format!(
            "import zipfile; z = zipfile.ZipFile(r'{}'); print(sorted(z.namelist()))",
            zip_path.display()
        ))
        .output()
        .unwrap();

    let file_list_str = String::from_utf8_lossy(&unzip_check.stdout);
    assert!(file_list_str.contains("'plugins/bin/vulcans-trials'"));
    assert!(file_list_str.contains("'bin/vulcans-trials'"));

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_build_multi_platform_kicad_packages_script() {
    let root = get_workspace_root();
    let temp_dir = create_temp_dir("multi_platform");
    let script_path = root.join("scripts").join("build_kicad_packages.py");
    assert!(script_path.exists(), "build_kicad_packages.py must exist");

    let output = Command::new(python_cmd())
        .arg(&script_path)
        .arg("--out-dir")
        .arg(&temp_dir)
        .arg("--mock-missing")
        .arg("--no-build-native")
        .current_dir(&root)
        .output()
        .expect("Failed to execute build_kicad_packages.py");

    assert!(
        output.status.success(),
        "build_kicad_packages.py failed:\nSTDOUT:\n{}\nSTDERR:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify all 4 target platforms (Linux x64/arm64, Windows, macOS x64/arm64) plus universal package
    let expected_archives = [
        "com.infernoembedded.vulcans-trials_v1.0.0_pcm.zip",
        "com.infernoembedded.vulcans-trials_v1.0.0_linux_x86_64_pcm.zip",
        "com.infernoembedded.vulcans-trials_v1.0.0_linux_aarch64_pcm.zip",
        "com.infernoembedded.vulcans-trials_v1.0.0_windows_x86_64_pcm.zip",
        "com.infernoembedded.vulcans-trials_v1.0.0_macos_x86_64_pcm.zip",
        "com.infernoembedded.vulcans-trials_v1.0.0_macos_aarch64_pcm.zip",
    ];

    for name in &expected_archives {
        let zip_p = temp_dir.join(name);
        assert!(zip_p.exists(), "Expected package {} was not created", name);
        assert!(fs::metadata(&zip_p).unwrap().len() > 1000);
    }

    // Verify packages.json repository index
    let pkg_json = temp_dir.join("packages.json");
    assert!(pkg_json.exists(), "packages.json must be generated");
    let content = fs::read_to_string(&pkg_json).unwrap();
    let val: Value = serde_json::from_str(&content).unwrap();
    let pkg = &val["packages"][0];
    assert_eq!(pkg["identifier"].as_str(), Some("com.infernoembedded.vulcans-trials"));

    let versions = pkg["versions"].as_array().unwrap();
    assert!(versions.len() >= 4, "Must register entries for all platforms (got {})", versions.len());

    let mut found_linux = false;
    let mut found_windows = false;
    let mut found_macos = false;

    for v in versions {
        if let Some(plats) = v["platforms"].as_array() {
            for p in plats {
                match p.as_str() {
                    Some("linux") => found_linux = true,
                    Some("windows") => found_windows = true,
                    Some("macos") => found_macos = true,
                    _ => {}
                }
            }
        }
    }

    assert!(found_linux, "packages.json missing Linux platform");
    assert!(found_windows, "packages.json missing Windows platform");
    assert!(found_macos, "packages.json missing macOS platform");

    // Verify SHA256SUMS.txt
    let sha_path = temp_dir.join("SHA256SUMS.txt");
    assert!(sha_path.exists(), "SHA256SUMS.txt must be generated");
    let sha_text = fs::read_to_string(&sha_path).unwrap();
    for name in &expected_archives {
        assert!(sha_text.contains(name), "SHA256SUMS.txt must include {}", name);
    }

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_app_icons_exist_for_all_os_and_window_managers() {
    let root = get_workspace_root();
    let assets = root.join("assets");
    assert!(assets.exists(), "assets directory must exist");

    // 1. Windows icon (.ico)
    let ico = assets.join("icon.ico");
    assert!(ico.exists(), "assets/icon.ico must exist");
    assert!(fs::metadata(&ico).unwrap().len() > 10000, "assets/icon.ico must contain multi-res mipmaps");

    // 2. macOS Apple Icon Image (.icns) and Info.plist
    let icns = assets.join("icon.icns");
    assert!(icns.exists(), "assets/icon.icns must exist");
    assert!(fs::metadata(&icns).unwrap().len() > 50000, "assets/icon.icns must contain retina icon data");

    let plist = assets.join("Info.plist");
    assert!(plist.exists(), "assets/Info.plist must exist");
    let plist_text = fs::read_to_string(&plist).unwrap();
    assert!(plist_text.contains("<string>com.infernoembedded.vulcans-trials</string>"));
    assert!(plist_text.contains("<string>icon.icns</string>"));

    // 3. Linux XDG Desktop Entry and Hicolor theme hierarchy
    let desktop = assets.join("vulcans-trials.desktop");
    assert!(desktop.exists(), "assets/vulcans-trials.desktop must exist");
    let desktop_text = fs::read_to_string(&desktop).unwrap();
    assert!(desktop_text.contains("Icon=vulcans-trials"));
    assert!(desktop_text.contains("StartupWMClass=com.infernoembedded.vulcans-trials"));

    let hicolor = assets.join("icons").join("hicolor");
    for size in [16, 24, 32, 48, 64, 128, 256, 512] {
        let png = hicolor.join(format!("{}x{}", size, size)).join("apps").join("vulcans-trials.png");
        assert!(png.exists(), "Missing XDG icon at {}", png.display());
    }
    let svg = hicolor.join("scalable").join("apps").join("vulcans-trials.svg");
    assert!(svg.exists(), "Missing scalable SVG at {}", svg.display());
}

#[test]
fn test_windows_rc_script_and_build_script_exist() {
    let root = get_workspace_root();
    let rc = root.join("assets").join("vulcans-trials.rc");
    assert!(rc.exists(), "assets/vulcans-trials.rc must exist");
    let rc_text = fs::read_to_string(&rc).unwrap();
    assert!(rc_text.contains("1 ICON \"icon.ico\""));

    let build_rs = root.join("build.rs");
    assert!(build_rs.exists(), "build.rs must exist");
    let build_text = fs::read_to_string(&build_rs).unwrap();
    assert!(build_text.contains("cargo:rerun-if-changed=assets/vulcans-trials.rc"));
}


