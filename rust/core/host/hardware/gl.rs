// rust/core/host/hardware/gl.rs
//! OpenGL symbol loading and readback helpers.

use std::ffi::{CString, c_void};

use super::egl::egl_fns;

#[derive(Clone, Copy)]
pub(super) struct GlProtocolValues {
    pub(super) rgb: GlEnum,
    pub(super) unsigned_byte: GlEnum,
    pub(super) pack_alignment: GlEnum,
}

pub(super) const GL_VALUES: GlProtocolValues = GlProtocolValues {
    rgb: 0x1907,
    unsigned_byte: 0x1401,
    pack_alignment: 0x0D05,
};

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

pub(super) fn flip_rgb_rows_into(
    rgb_bottom_left: &[u8],
    width: usize,
    height: usize,
    rgb: &mut Vec<u8>,
) {
    let row_len = width * 3;
    rgb.resize(rgb_bottom_left.len(), 0);
    if row_len == 0 || height == 0 {
        return;
    }
    for (src_row, dst_row) in rgb_bottom_left
        .chunks_exact(row_len)
        .take(height)
        .rev()
        .zip(rgb.chunks_exact_mut(row_len))
    {
        dst_row.copy_from_slice(src_row);
    }
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
