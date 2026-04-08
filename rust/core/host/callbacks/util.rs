// rust/core/host/callbacks/util.rs
use std::ffi::{CStr, CString, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

use libretro_sys::LogLevel;

use crate::core::error::CoreError;

pub(super) unsafe extern "C" fn log_callback(_level: LogLevel, _fmt: *const i8) {}

pub(super) fn path_to_c_string(path: &Path) -> Result<CString, CoreError> {
    CString::new(path.to_string_lossy().as_bytes()).map_err(|_| CoreError::InvalidPath {
        path: path.to_path_buf(),
    })
}

pub(super) fn c_string(value: *const i8) -> String {
    if value.is_null() {
        return String::new();
    }

    unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned()
}

pub(super) fn read_u32(data: *mut c_void) -> Option<u32> {
    if data.is_null() {
        return None;
    }

    Some(unsafe { ptr::read(data.cast::<u32>()) })
}

pub(super) fn write_ptr<T>(data: *mut c_void, value: T) -> bool {
    if data.is_null() {
        return false;
    }

    unsafe {
        ptr::write(data.cast::<T>(), value);
    }
    true
}

pub(super) fn runtime_root_for_core(core_path: &Path) -> Result<PathBuf, CoreError> {
    let core_dir = core_path
        .parent()
        .ok_or_else(|| CoreError::InvalidPath {
            path: core_path.to_path_buf(),
        })?
        .to_path_buf();

    if core_dir
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value == "libretro")
        && let Some(runtime_root) = core_dir.parent()
    {
        return Ok(runtime_root.to_path_buf());
    }

    Ok(core_dir)
}
