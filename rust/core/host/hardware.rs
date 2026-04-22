// rust/core/host/hardware.rs
//! Minimal synchronous hardware-render frontend support.
//!
//! This intentionally starts with a blocking OpenGL readback path. It is slow
//! compared with a real async PBO pipeline, but it proves whether a renderer can
//! run correctly before adding more moving parts.

use std::ffi::c_void;
use std::ptr;

use libretro_sys::{HW_FRAME_BUFFER_VALID, HwContextResetFn, HwContextType, HwRenderCallback};

use crate::core::video::VideoFrame;

mod callbacks;
mod egl;
mod gl;

use egl::{
    EglContext, EglDisplay, EglFns, EglSurface, bind_opengl_api, choose_config, context_type,
    create_context, create_display, create_surface, egl_fns, make_current,
};
use gl::{GL_PACK_ALIGNMENT, GL_RGB, GL_UNSIGNED_BYTE, GlFns, GlSizei, flip_rgb_rows};

pub struct HardwareRenderContext {
    egl: &'static EglFns,
    display: EglDisplay,
    surface: EglSurface,
    context: EglContext,
    context_reset: HwContextResetFn,
    context_destroy: HwContextResetFn,
    gl: GlFns,
    scratch: Vec<u8>,
}

impl HardwareRenderContext {
    pub fn from_callback(callback: &mut HwRenderCallback) -> Result<Self, String> {
        let context_type = context_type(callback.context_type);
        if !matches!(
            context_type,
            Some(HwContextType::OpenGL | HwContextType::OpenGLCore)
        ) {
            return Err(format!(
                "hardware context {:?} is not supported by this host yet",
                context_type
            ));
        }

        let egl = egl_fns()?;
        let display = create_display(egl)?;
        let config = choose_config(egl, display)?;
        let surface = create_surface(egl, display, config)?;
        bind_opengl_api(egl)?;
        let context = create_context(egl, display, config, callback)?;
        make_current(egl, display, surface, context)?;

        callback.get_current_framebuffer = callbacks::get_current_framebuffer;
        callback.get_proc_address = callbacks::get_proc_address;

        let gl = GlFns::load()?;
        Ok(Self {
            egl,
            display,
            surface,
            context,
            context_reset: callback.context_reset,
            context_destroy: callback.context_destroy,
            gl,
            scratch: Vec::new(),
        })
    }

    pub fn can_capture(data: *const c_void) -> bool {
        data == HW_FRAME_BUFFER_VALID
    }

    pub fn reset_core_context(&self) -> Result<(), String> {
        make_current(self.egl, self.display, self.surface, self.context)?;
        unsafe {
            (self.context_reset)();
        }
        Ok(())
    }

    pub fn destroy_core_context(&self) {
        if self.context_destroy as usize == 0 {
            return;
        }
        unsafe {
            (self.context_destroy)();
        }
    }

    pub fn capture_frame(&mut self, width: usize, height: usize) -> Option<VideoFrame> {
        if width == 0 || height == 0 {
            return None;
        }
        make_current(self.egl, self.display, self.surface, self.context).ok()?;
        let byte_len = width.checked_mul(height)?.checked_mul(3)?;
        self.scratch.resize(byte_len, 0);
        unsafe {
            (self.gl.finish)();
            (self.gl.pixel_storei)(GL_PACK_ALIGNMENT, 1);
            (self.gl.read_pixels)(
                0,
                0,
                width as GlSizei,
                height as GlSizei,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                self.scratch.as_mut_ptr().cast::<c_void>(),
            );
        }
        Some(VideoFrame {
            width,
            height,
            rgb: flip_rgb_rows(&self.scratch, width, height),
        })
    }
}

impl Drop for HardwareRenderContext {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.egl.make_current)(
                self.display,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            );
            let _ = (self.egl.destroy_context)(self.display, self.context);
            let _ = (self.egl.destroy_surface)(self.display, self.surface);
            let _ = (self.egl.terminate)(self.display);
        }
    }
}

// SAFETY: This is required by PyO3's `Python::detach`/`Ungil` bound for native
// emulator calls that release the GIL while continuing on the same OS thread.
// `PyEmulator` is declared `unsendable`, and the runtime does not hand
// `HardwareRenderContext` to worker threads. Do not use this as permission to
// move EGL/OpenGL contexts across threads.
unsafe impl Send for HardwareRenderContext {}
