// rust/core/host/options.rs
//! Small helpers for libretro core-option parsing and overrides.

use std::ffi::CStr;

/// Extract the default value from a libretro option spec like
/// `"Renderer; angrylion|gliden64"`.
pub fn default_option_value(spec: &str) -> String {
    let Some((_, values)) = spec.split_once("; ") else {
        return spec.to_owned();
    };
    values.split('|').next().unwrap_or_default().to_owned()
}

/// Override core options that the host wants to pin explicitly.
pub fn override_option(key: &str, default_value: &str, renderer: &CStr) -> String {
    let renderer_name = renderer.to_string_lossy();
    match key {
        "mupen64plus-rdp-plugin" => renderer_name.into_owned(),
        // Keep GLideN64 readback close to native N64 resolution. Higher
        // viewport defaults move more pixels over the CPU/GPU boundary and are
        // counterproductive for RL observations.
        "mupen64plus-43screensize" if renderer_name == "gliden64" => "320x240".to_owned(),
        _ => default_value.to_owned(),
    }
}

#[cfg(test)]
#[path = "tests/options_tests.rs"]
mod tests;
