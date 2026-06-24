// rust/core/host/callbacks/state/hardware.rs
//! CallbackState hardware-render context setup and lifecycle.

use std::ffi::c_void;

use crate::core::error::{CoreError, HardwareRenderError};
use crate::core::host::hardware::HardwareRenderContext;

use super::CallbackState;

impl CallbackState {
    pub fn take_hardware_render_error(&mut self) -> Option<HardwareRenderError> {
        self.hardware_render_error.take()
    }

    pub fn reset_hardware_context(&mut self) -> Result<(), CoreError> {
        let Some(context) = self.hardware_render.as_ref() else {
            return Ok(());
        };
        context.reset_core_context().map_err(CoreError::from)
    }

    pub fn destroy_hardware_context(&mut self) {
        if let Some(context) = self.hardware_render.as_ref() {
            context.destroy_core_context();
        }
    }

    pub(super) fn set_hardware_render(&mut self, data: *mut c_void) -> bool {
        if data.is_null() {
            self.hardware_render_error = Some(HardwareRenderError::NullCallback);
            return false;
        }

        let callback = unsafe { &mut *data.cast::<libretro_sys::HwRenderCallback>() };
        match HardwareRenderContext::from_callback(callback) {
            Ok(context) => {
                self.hardware_render = Some(context);
                self.hardware_render_error = None;
                true
            }
            Err(error) => {
                self.hardware_render_error = Some(error);
                false
            }
        }
    }
}
