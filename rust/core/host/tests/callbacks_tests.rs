// Covers callback utility helpers for runtime-directory resolution.
use std::path::{Path, PathBuf};

use super::runtime_root_for_core;

#[test]
fn runtime_root_uses_repo_local_root_for_libretro_layout() {
    let runtime_root = runtime_root_for_core(Path::new(
        "/repo/local/libretro/mupen64plus_next_libretro.so",
    ))
    .expect("runtime root should resolve");

    assert_eq!(runtime_root, PathBuf::from("/repo/local"));
}

#[test]
fn runtime_root_uses_core_directory_for_generic_layout() {
    let runtime_root = runtime_root_for_core(Path::new("/opt/cores/mupen64plus_next_libretro.so"))
        .expect("runtime root should resolve");

    assert_eq!(runtime_root, PathBuf::from("/opt/cores"));
}
