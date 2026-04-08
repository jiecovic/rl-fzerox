// rust/core/host/runtime/bootstrap.rs
//! Host bootstrap and core-lifecycle helpers.

use std::ptr;

use libretro_sys::{DEVICE_JOYPAD, GameInfo};

use super::Host;
use crate::core::callbacks::{
    audio_sample_batch_callback, audio_sample_callback, environment_callback, input_device,
    input_poll_callback, input_state_callback, video_refresh_callback,
};
use crate::core::error::CoreError;
use crate::core::stdio::with_silenced_stdio;

impl Host {
    pub(super) fn configure_callbacks(&mut self) {
        self.call_core(|core| unsafe {
            core.set_environment(environment_callback);
            core.set_video_refresh(video_refresh_callback);
            core.set_audio_sample(audio_sample_callback);
            core.set_audio_sample_batch(audio_sample_batch_callback);
            core.set_input_poll(input_poll_callback);
            core.set_input_state(input_state_callback);
        });
    }

    pub(super) fn initialize(&mut self) {
        with_silenced_stdio(|| {
            self.call_core(|core| unsafe {
                core.init();
            });
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
            self.call_core(|core| unsafe {
                core.set_controller_port_device(0, input_device());
                core.set_controller_port_device(1, DEVICE_JOYPAD);
                core.load_game(&game)
            })
        });
        if loaded {
            Ok(())
        } else {
            Err(CoreError::LoadGameFailed {
                path: self.rom_path.clone(),
            })
        }
    }

    pub(super) fn refresh_av_info(&mut self) {
        let av_info = self.call_core(|core| unsafe { core.system_av_info() });
        self.display_aspect_ratio = resolve_display_aspect_ratio(
            av_info.geometry.base_width as usize,
            av_info.geometry.base_height as usize,
            av_info.geometry.aspect_ratio as f64,
        );
        self.native_fps = av_info.timing.fps;
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
            self.call_core(|core| unsafe {
                core.run();
            });
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
