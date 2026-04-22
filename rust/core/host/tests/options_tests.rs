// rust/core/host/tests/options_tests.rs
// Covers small libretro option parsing and renderer override helpers.
use std::ffi::CString;

use super::{default_option_value, override_option};

#[test]
fn default_option_value_returns_first_choice() {
    assert_eq!(
        default_option_value("Renderer; angrylion|gliden64"),
        "angrylion"
    );
}

#[test]
fn override_option_uses_requested_renderer() {
    let renderer = CString::new("gliden64").expect("renderer");

    assert_eq!(
        override_option("mupen64plus-rdp-plugin", "angrylion", renderer.as_c_str()),
        "gliden64"
    );
}

#[test]
fn override_option_keeps_gliden64_at_native_resolution() {
    let renderer = CString::new("gliden64").expect("renderer");

    assert_eq!(
        override_option("mupen64plus-43screensize", "640x480", renderer.as_c_str()),
        "320x240"
    );
}
