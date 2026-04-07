use std::ffi::CString;

use super::split_extensions;

#[test]
fn split_extensions_handles_empty_entries() {
    let extensions = CString::new("n64||z64|v64").expect("valid c string");
    let result = split_extensions(extensions.as_ptr());
    assert_eq!(result, vec!["n64", "z64", "v64"]);
}
