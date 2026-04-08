// Covers libretro API loader helpers that do not require a real core binary.
use super::{empty_system_av_info, empty_system_info, printable_symbol};

#[test]
fn printable_symbol_strips_trailing_nul() {
    assert_eq!(printable_symbol(b"retro_run\0"), "retro_run");
}

#[test]
fn printable_symbol_preserves_names_without_nul() {
    assert_eq!(printable_symbol(b"retro_reset"), "retro_reset");
}

#[test]
fn empty_system_info_starts_with_null_string_fields() {
    let info = empty_system_info();
    assert!(info.library_name.is_null());
    assert!(info.library_version.is_null());
    assert!(info.valid_extensions.is_null());
    assert!(!info.need_fullpath);
    assert!(!info.block_extract);
}

#[test]
fn empty_system_av_info_starts_zeroed() {
    let info = empty_system_av_info();
    assert_eq!(info.geometry.base_width, 0);
    assert_eq!(info.geometry.base_height, 0);
    assert_eq!(info.geometry.aspect_ratio, 0.0);
    assert_eq!(info.timing.fps, 0.0);
    assert_eq!(info.timing.sample_rate, 0.0);
}
