// rust/core/libretro/info.rs
use std::ffi::CStr;

use libretro_sys::SystemInfo;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreInfo {
    pub api_version: u32,
    pub library_name: String,
    pub library_version: String,
    pub valid_extensions: Vec<String>,
    pub requires_full_path: bool,
    pub blocks_extract: bool,
}

impl CoreInfo {
    pub fn from_system_info(api_version: u32, system_info: &SystemInfo) -> Self {
        Self {
            api_version,
            library_name: c_string_field(system_info.library_name),
            library_version: c_string_field(system_info.library_version),
            valid_extensions: split_extensions(system_info.valid_extensions),
            requires_full_path: system_info.need_fullpath,
            blocks_extract: system_info.block_extract,
        }
    }
}

pub fn c_string_field(value: *const i8) -> String {
    if value.is_null() {
        return String::new();
    }

    // SAFETY: libretro system info fields are documented as valid NUL-terminated
    // strings for the duration of the call site that owns the struct, and we copy
    // the contents immediately.
    unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned()
}

pub fn split_extensions(value: *const i8) -> Vec<String> {
    c_string_field(value)
        .split('|')
        .filter(|entry| !entry.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

#[cfg(test)]
#[path = "tests/info_tests.rs"]
mod tests;
