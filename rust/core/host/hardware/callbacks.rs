// rust/core/host/hardware/callbacks.rs
//! Libretro hardware-render callback functions.

use std::ffi::{CStr, c_char, c_void};
use std::ptr;

use libretro_sys::ProcAddressFn;

use super::egl::egl_fns;

pub(super) unsafe extern "C" fn get_current_framebuffer() -> libc::uintptr_t {
    0
}

pub(super) unsafe extern "C" fn get_proc_address(sym: *const c_char) -> ProcAddressFn {
    if sym.is_null() {
        return missing_proc_stub;
    }
    let symbol_name = unsafe { CStr::from_ptr(sym) };
    let pointer = egl_fns()
        .map(|egl| egl.proc_address(symbol_name))
        .unwrap_or(ptr::null());
    if pointer.is_null() {
        return missing_proc_stub;
    }
    unsafe { std::mem::transmute::<*const c_void, ProcAddressFn>(pointer) }
}

unsafe extern "C" fn missing_proc_stub() {}
