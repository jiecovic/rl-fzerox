// rust/core/host/runtime/bootstrap.rs
//! Host bootstrap and core-lifecycle helpers.

use std::ptr;

use libretro_sys::{DEVICE_JOYPAD, GameInfo};

use super::host::Host;
use crate::core::callbacks::{
    CallbackGuard, audio_sample_batch_callback, audio_sample_callback, environment_callback,
    input_device, input_poll_callback, input_state_callback, video_refresh_callback,
};
use crate::core::error::CoreError;
use crate::core::stdio::with_silenced_stdio;

impl Host {
    pub(super) fn configure_callbacks(&mut self) {
        self.install_core_callbacks(
            environment_callback,
            video_refresh_callback,
            audio_sample_callback,
            audio_sample_batch_callback,
            input_poll_callback,
            input_state_callback,
        );
    }

    pub(super) fn initialize(&mut self) {
        with_silenced_stdio(|| {
            self.init_core();
        });
    }

    pub(super) fn load_game(&mut self) -> Result<(), CoreError> {
        let game = GameInfo {
            path: self.rom_path_c_string.as_ptr(),
            data: self.rom_bytes.as_ptr().cast(),
            size: self.rom_bytes.len(),
            meta: ptr::null(),
        };
        let loaded = with_silenced_stdio(|| {
            self.set_core_controller_port_device(0, input_device());
            self.set_core_controller_port_device(1, DEVICE_JOYPAD);
            self.load_core_game(&game)
        });
        if loaded {
            self.reset_hardware_context()
        } else if let Some(message) = self.callbacks.take_hardware_render_error() {
            Err(CoreError::HardwareRenderFailed { message })
        } else {
            Err(CoreError::LoadGameFailed {
                path: self.rom_path.clone(),
            })
        }
    }

    fn reset_hardware_context(&mut self) -> Result<(), CoreError> {
        // A hardware context reset is a core callback in reverse: the core can
        // synchronously call back into the frontend for API-specific handles.
        let callbacks = self.callbacks.as_mut() as *mut _;
        let _guard = CallbackGuard::activate(callbacks);
        with_silenced_stdio(|| self.callbacks.reset_hardware_context())
    }

    pub(super) fn refresh_av_info(&mut self) {
        let av_info = self.core_system_av_info();
        self.display_aspect_ratio = resolve_display_aspect_ratio(
            av_info.geometry.base_width as usize,
            av_info.geometry.base_height as usize,
            av_info.geometry.aspect_ratio as f64,
        );
        self.native_fps = av_info.timing.fps;
        self.native_sample_rate = av_info.timing.sample_rate;
        self.frame_shape = (
            av_info.geometry.base_height as usize,
            av_info.geometry.base_width as usize,
            3,
        );
        self.refresh_shape_from_frame();
    }

    pub(super) fn refresh_shape_from_frame(&mut self) {
        if let Some((width, height)) = self.callbacks.geometry() {
            self.frame_shape = (height, width, 3);
        }
    }

    pub(super) fn run_until_frame(
        &mut self,
        baseline_serial: u64,
        limit: usize,
    ) -> Result<(), CoreError> {
        // Some reset/load paths do not synchronously hand us a frame. We keep
        // stepping until the callback-side frame serial advances or we hit the
        // explicit wait limit.
        for _ in 0..limit {
            self.run_core_frame();
            if self.callbacks.frame_serial() > baseline_serial {
                return Ok(());
            }
        }
        Err(CoreError::NoFrameAvailable)
    }
}

/// Use the core-reported display aspect ratio when available, otherwise fall
/// back to the raw geometry ratio.
pub(super) fn resolve_display_aspect_ratio(width: usize, height: usize, aspect_ratio: f64) -> f64 {
    if aspect_ratio > 0.0 {
        return aspect_ratio;
    }
    if width == 0 || height == 0 {
        return 0.0;
    }
    width as f64 / height as f64
}
