// rust/core/host.rs
use std::ffi::CString;
use std::path::{Path, PathBuf};

use libretro_sys::{DEVICE_JOYPAD, GameInfo};

use crate::core::api::LoadedCore;
use crate::core::callbacks::{
    CallbackGuard, CallbackState, VideoFrame, audio_sample_batch_callback, audio_sample_callback,
    environment_callback, input_device, input_poll_callback, input_state_callback,
    video_refresh_callback,
};
use crate::core::error::CoreError;

const RESET_WARMUP_FRAMES: usize = 120;

pub struct Host {
    core: LoadedCore,
    callbacks: Box<CallbackState>,
    rom_path: PathBuf,
    rom_path_c_string: CString,
    rom_bytes: Vec<u8>,
    display_aspect_ratio: f64,
    native_fps: f64,
    frame_shape: (usize, usize, usize),
    frame_index: usize,
    closed: bool,
}

impl Host {
    pub fn open(core_path: &Path, rom_path: &Path) -> Result<Self, CoreError> {
        if !rom_path.is_file() {
            return Err(CoreError::MissingRom(rom_path.to_path_buf()));
        }

        let core = LoadedCore::load(core_path)?;
        let callbacks = Box::new(CallbackState::new(core_path, rom_path)?);
        let rom_bytes = std::fs::read(rom_path).map_err(|error| CoreError::ReadFile {
            path: rom_path.to_path_buf(),
            message: error.to_string(),
        })?;
        let rom_path_c_string =
            CString::new(rom_path.to_string_lossy().as_bytes()).map_err(|_| {
                CoreError::InvalidPath {
                    path: rom_path.to_path_buf(),
                }
            })?;

        let mut host = Self {
            core,
            callbacks,
            rom_path: rom_path.to_path_buf(),
            rom_path_c_string,
            rom_bytes,
            display_aspect_ratio: 0.0,
            native_fps: 0.0,
            frame_shape: (0, 0, 3),
            frame_index: 0,
            closed: false,
        };
        host.configure_callbacks();
        host.initialize();
        host.load_game()?;
        host.refresh_av_info();
        Ok(host)
    }

    pub fn name(&self) -> &str {
        &self.core.info().library_name
    }

    pub fn native_fps(&self) -> f64 {
        self.native_fps
    }

    pub fn display_aspect_ratio(&self) -> f64 {
        self.display_aspect_ratio
    }

    pub fn frame_shape(&self) -> (usize, usize, usize) {
        self.frame_shape
    }

    pub fn frame_index(&self) -> usize {
        self.frame_index
    }

    pub fn reset(&mut self) -> Result<(), CoreError> {
        self.ensure_open()?;
        let baseline = self.callbacks.frame_serial();
        self.call_core(|core| unsafe {
            core.reset();
        });
        self.run_until_frame(baseline, RESET_WARMUP_FRAMES)?;
        self.frame_index = 0;
        self.refresh_shape_from_frame();
        Ok(())
    }

    pub fn step_frames(&mut self, count: usize) -> Result<(), CoreError> {
        self.ensure_open()?;
        for _ in 0..count {
            self.call_core(|core| unsafe {
                core.run();
            });
        }
        self.frame_index += count;
        self.refresh_shape_from_frame();
        if self.callbacks.frame().is_none() {
            return Err(CoreError::NoFrameAvailable);
        }
        Ok(())
    }

    pub fn frame_rgb(&self) -> Result<Vec<u8>, CoreError> {
        self.callbacks
            .frame()
            .map(|frame| frame.rgb.clone())
            .ok_or(CoreError::NoFrameAvailable)
    }

    pub fn close(&mut self) {
        if self.closed {
            return;
        }

        self.call_core(|core| unsafe {
            core.unload_game();
            core.deinit();
        });
        self.closed = true;
    }

    fn configure_callbacks(&mut self) {
        self.call_core(|core| unsafe {
            core.set_environment(environment_callback);
            core.set_video_refresh(video_refresh_callback);
            core.set_audio_sample(audio_sample_callback);
            core.set_audio_sample_batch(audio_sample_batch_callback);
            core.set_input_poll(input_poll_callback);
            core.set_input_state(input_state_callback);
        });
    }

    fn initialize(&mut self) {
        self.call_core(|core| unsafe {
            core.init();
        });
    }

    fn load_game(&mut self) -> Result<(), CoreError> {
        let game = GameInfo {
            path: self.rom_path_c_string.as_ptr(),
            data: self.rom_bytes.as_ptr().cast(),
            size: self.rom_bytes.len(),
            meta: std::ptr::null(),
        };
        let loaded = self.call_core(|core| unsafe {
            core.set_controller_port_device(0, input_device());
            core.set_controller_port_device(1, DEVICE_JOYPAD);
            core.load_game(&game)
        });
        if loaded {
            Ok(())
        } else {
            Err(CoreError::LoadGameFailed {
                path: self.rom_path.clone(),
            })
        }
    }

    fn refresh_av_info(&mut self) {
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

    fn refresh_shape_from_frame(&mut self) {
        if let Some(VideoFrame { width, height, .. }) = self.callbacks.frame() {
            self.frame_shape = (*height, *width, 3);
            return;
        }

        if let Some((width, height)) = self.callbacks.geometry() {
            self.frame_shape = (height, width, 3);
        }
    }

    fn run_until_frame(&mut self, baseline_serial: u64, limit: usize) -> Result<(), CoreError> {
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

    fn call_core<R>(&mut self, action: impl FnOnce(&LoadedCore) -> R) -> R {
        let callbacks = self.callbacks.as_mut() as *mut CallbackState;
        let _guard = CallbackGuard::activate(callbacks);
        action(&self.core)
    }

    fn ensure_open(&self) -> Result<(), CoreError> {
        if self.closed {
            return Err(CoreError::AlreadyClosed);
        }
        Ok(())
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        self.close();
    }
}

fn resolve_display_aspect_ratio(width: usize, height: usize, aspect_ratio: f64) -> f64 {
    if aspect_ratio > 0.0 {
        return aspect_ratio;
    }
    if width == 0 || height == 0 {
        return 0.0;
    }
    width as f64 / height as f64
}

#[cfg(test)]
mod tests {
    use super::resolve_display_aspect_ratio;

    #[test]
    fn resolve_display_aspect_ratio_prefers_reported_ratio() {
        let ratio = resolve_display_aspect_ratio(640, 240, 4.0 / 3.0);
        assert!((ratio - (4.0 / 3.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn resolve_display_aspect_ratio_falls_back_to_geometry() {
        let ratio = resolve_display_aspect_ratio(640, 480, 0.0);
        assert!((ratio - (4.0 / 3.0)).abs() < f64::EPSILON);
    }
}
