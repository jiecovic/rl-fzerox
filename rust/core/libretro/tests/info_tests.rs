// Covers parsing libretro extension strings from raw system-info metadata.
use std::ffi::CString;

use libretro_sys::SystemInfo;

use super::{CoreInfo, split_extensions};

#[test]
fn split_extensions_handles_empty_entries() {
    let extensions = CString::new("n64||z64|v64").expect("valid c string");
    let result = split_extensions(extensions.as_ptr());
    assert_eq!(result, vec!["n64", "z64", "v64"]);
}

#[test]
fn core_info_copies_owned_fields_from_system_info() {
    let library_name = CString::new("Mupen").expect("library name");
    let library_version = CString::new("1.0").expect("library version");
    let extensions = CString::new("n64|z64").expect("extensions");
    let system_info = SystemInfo {
        library_name: library_name.as_ptr(),
        library_version: library_version.as_ptr(),
        valid_extensions: extensions.as_ptr(),
        need_fullpath: true,
        block_extract: false,
    };

    let info = CoreInfo::from_system_info(1, &system_info);

    assert_eq!(info.api_version, 1);
    assert_eq!(info.library_name, "Mupen");
    assert_eq!(info.library_version, "1.0");
    assert_eq!(info.valid_extensions, vec!["n64", "z64"]);
    assert!(info.requires_full_path);
    assert!(!info.blocks_extract);
}
