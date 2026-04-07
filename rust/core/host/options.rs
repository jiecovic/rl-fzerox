// rust/core/host/options.rs
pub fn default_option_value(spec: &str) -> String {
    let Some((_, values)) = spec.split_once("; ") else {
        return spec.to_owned();
    };
    values.split('|').next().unwrap_or_default().to_owned()
}

use std::ffi::CStr;

pub fn override_option(key: &str, default_value: &str, renderer: &CStr) -> String {
    match key {
        "mupen64plus-rdp-plugin" => renderer.to_string_lossy().into_owned(),
        _ => default_value.to_owned(),
    }
}

#[cfg(test)]
#[path = "tests/options_tests.rs"]
mod tests;
