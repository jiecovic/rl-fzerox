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
use gl::{GL_VALUES, GlFns, GlSizei, flip_rgb_rows_into};

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
        // SAFETY: The callback comes from libretro's hardware-render contract
        // and is invoked after this host has made the matching EGL context current.
        unsafe {
            (self.context_reset)();
        }
        Ok(())
    }

    pub fn destroy_core_context(&self) {
        if self.context_destroy as usize == 0 {
            return;
        }
        // SAFETY: The core registered this optional destroy callback together
        // with the hardware-render context and owns any core-side GL resources.
        unsafe {
            (self.context_destroy)();
        }
    }

    pub fn capture_frame(&mut self, width: usize, height: usize) -> Option<VideoFrame> {
        let mut frame = VideoFrame {
            width,
            height,
            rgb: Vec::new(),
        };
        self.capture_frame_into(&mut frame, width, height)
            .then_some(frame)
    }

    pub fn capture_frame_into(
        &mut self,
        frame: &mut VideoFrame,
        width: usize,
        height: usize,
    ) -> bool {
        if width == 0 || height == 0 {
            return false;
        }
        if make_current(self.egl, self.display, self.surface, self.context).is_err() {
            return false;
        }
        let Some(byte_len) = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(3))
        else {
            return false;
        };
        self.scratch.resize(byte_len, 0);
        // SAFETY: `self.gl` contains loaded OpenGL function pointers with the
        // declared signatures. The current EGL context is active above, and
        // `scratch` is sized to hold width * height * RGB bytes.
        unsafe {
            (self.gl.finish)();
            (self.gl.pixel_storei)(GL_VALUES.pack_alignment, 1);
            (self.gl.read_pixels)(
                0,
                0,
                width as GlSizei,
                height as GlSizei,
                GL_VALUES.rgb,
                GL_VALUES.unsigned_byte,
                self.scratch.as_mut_ptr().cast::<c_void>(),
            );
        }
        frame.width = width;
        frame.height = height;
        flip_rgb_rows_into(&self.scratch, width, height, &mut frame.rgb);
        true
    }
}

impl Drop for HardwareRenderContext {
    fn drop(&mut self) {
        // SAFETY: These EGL handles were created by `from_callback` and remain
        // owned by this context until drop. Null draw/read/context values are
        // the EGL API's way to unbind the current context before destruction.
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
