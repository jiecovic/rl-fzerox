// rust/core/host/hardware/gl.rs
//! OpenGL symbol loading and readback helpers.

use std::ffi::{CString, c_void};

use super::egl::egl_fns;

pub(super) const GL_RGB: GlEnum = 0x1907;
pub(super) const GL_UNSIGNED_BYTE: GlEnum = 0x1401;
pub(super) const GL_PACK_ALIGNMENT: GlEnum = 0x0D05;

type GlEnum = libc::c_uint;
type GlInt = libc::c_int;
pub(super) type GlSizei = libc::c_int;
type GlReadPixels =
    unsafe extern "C" fn(GlInt, GlInt, GlSizei, GlSizei, GlEnum, GlEnum, *mut c_void);
type GlPixelStorei = unsafe extern "C" fn(GlEnum, GlInt);
type GlFinish = unsafe extern "C" fn();

#[derive(Clone, Copy)]
pub(super) struct GlFns {
    pub(super) read_pixels: GlReadPixels,
    pub(super) pixel_storei: GlPixelStorei,
    pub(super) finish: GlFinish,
}

impl GlFns {
    pub(super) fn load() -> Result<Self, String> {
        Ok(Self {
            read_pixels: gl_symbol("glReadPixels")?,
            pixel_storei: gl_symbol("glPixelStorei")?,
            finish: gl_symbol("glFinish")?,
        })
    }
}

pub(super) fn flip_rgb_rows(rgb_bottom_left: &[u8], width: usize, height: usize) -> Vec<u8> {
    let row_len = width * 3;
    let mut rgb = vec![0_u8; rgb_bottom_left.len()];
    for y in 0..height {
        let src = (height - 1 - y) * row_len;
        let dst = y * row_len;
        rgb[dst..dst + row_len].copy_from_slice(&rgb_bottom_left[src..src + row_len]);
    }
    rgb
}

fn gl_symbol<T>(name: &str) -> Result<T, String>
where
    T: Copy,
{
    let symbol_name = CString::new(name).map_err(|error| error.to_string())?;
    let pointer = egl_fns()?.proc_address(symbol_name.as_c_str());
    if pointer.is_null() {
        return Err(format!("missing GL symbol {name}"));
    }
    Ok(unsafe { std::mem::transmute_copy::<*const c_void, T>(&pointer) })
}
