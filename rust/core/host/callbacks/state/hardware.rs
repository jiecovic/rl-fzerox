// rust/core/host/callbacks/hardware_state.rs
//! CallbackState hardware-render context setup and lifecycle.

use std::ffi::c_void;

use crate::core::error::CoreError;
use crate::core::host::hardware::HardwareRenderContext;

use super::CallbackState;

impl CallbackState {
    pub fn take_hardware_render_error(&mut self) -> Option<String> {
        self.hardware_render_error.take()
    }

    pub fn reset_hardware_context(&mut self) -> Result<(), CoreError> {
        let Some(context) = self.hardware_render.as_ref() else {
            return Ok(());
        };
        context
            .reset_core_context()
            .map_err(|message| CoreError::HardwareRenderFailed { message })
    }

    pub fn destroy_hardware_context(&mut self) {
        if let Some(context) = self.hardware_render.as_ref() {
            context.destroy_core_context();
        }
    }

    pub(super) fn set_hardware_render(&mut self, data: *mut c_void) -> bool {
        if data.is_null() {
            self.hardware_render_error = Some("SET_HW_RENDER received null callback".to_owned());
            return false;
        }

        let callback = unsafe { &mut *data.cast::<libretro_sys::HwRenderCallback>() };
        match HardwareRenderContext::from_callback(callback) {
            Ok(context) => {
                self.hardware_render = Some(context);
                self.hardware_render_error = None;
                true
            }
            Err(message) => {
                self.hardware_render_error = Some(message);
                false
            }
        }
    }
}
