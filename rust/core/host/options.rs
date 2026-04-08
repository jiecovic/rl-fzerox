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
    match key {
        "mupen64plus-rdp-plugin" => renderer.to_string_lossy().into_owned(),
        _ => default_value.to_owned(),
    }
}

#[cfg(test)]
#[path = "tests/options_tests.rs"]
mod tests;
