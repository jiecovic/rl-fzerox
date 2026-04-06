// rust/core/host.rs
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::slice;

use libretro_sys::{DEVICE_JOYPAD, GameInfo, MEMORY_SYSTEM_RAM};

use crate::core::api::LoadedCore;
use crate::core::callbacks::{
    CallbackGuard, CallbackState, audio_sample_batch_callback, audio_sample_callback,
    environment_callback, input_device, input_poll_callback, input_state_callback,
    video_refresh_callback,
};
use crate::core::error::CoreError;
use crate::core::video::VideoFrame;

const FRAME_WAIT_LIMIT: usize = 120;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BaselineKind {
    Startup,
    Custom,
}

pub struct Host {
    core: LoadedCore,
    callbacks: Box<CallbackState>,
    rom_path: PathBuf,
    rom_path_c_string: CString,
    rom_bytes: Vec<u8>,
    baseline_state: Vec<u8>,
    baseline_frame: Option<VideoFrame>,
    baseline_kind: BaselineKind,
    display_aspect_ratio: f64,
    native_fps: f64,
    frame_shape: (usize, usize, usize),
    frame_index: usize,
    closed: bool,
}

impl Host {
    pub fn open(
        core_path: &Path,
        rom_path: &Path,
        runtime_dir: Option<&Path>,
        baseline_state_path: Option<&Path>,
    ) -> Result<Self, CoreError> {
        if !rom_path.is_file() {
            return Err(CoreError::MissingRom(rom_path.to_path_buf()));
        }

        let core = LoadedCore::load(core_path)?;
        let callbacks = Box::new(CallbackState::new(core_path, runtime_dir)?);
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
            baseline_state: Vec::new(),
            baseline_frame: None,
            baseline_kind: BaselineKind::Startup,
            display_aspect_ratio: 0.0,
            native_fps: 0.0,
            frame_shape: (0, 0, 3),
            frame_index: 0,
            closed: false,
        };
        host.configure_callbacks();
        host.initialize();
        host.load_game()?;
        match baseline_state_path {
            Some(path) if path.is_file() => host.load_baseline_from_file(path)?,
            None => host.capture_startup_baseline()?,
            Some(_) => host.capture_startup_baseline()?,
        }
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

    pub fn system_ram_size(&mut self) -> Result<usize, CoreError> {
        self.memory_size(MEMORY_SYSTEM_RAM)
    }

    pub fn baseline_kind(&self) -> &'static str {
        match self.baseline_kind {
            BaselineKind::Startup => "startup",
            BaselineKind::Custom => "custom",
        }
    }

    pub fn reset(&mut self) -> Result<(), CoreError> {
        self.ensure_open()?;
        self.restore_baseline()?;
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

    pub fn set_joypad_mask(&mut self, joypad_mask: u16) -> Result<(), CoreError> {
        self.ensure_open()?;
        self.callbacks.set_joypad_mask(joypad_mask);
        Ok(())
    }

    pub fn save_state(&mut self, path: &Path) -> Result<(), CoreError> {
        self.ensure_open()?;
        let state = self.serialize_state()?;
        self.write_state_bytes(path, &state)
    }

    pub fn capture_current_as_baseline(
        &mut self,
        save_path: Option<&Path>,
    ) -> Result<(), CoreError> {
        self.ensure_open()?;
        let state = self.serialize_state()?;
        if let Some(path) = save_path {
            self.write_state_bytes(path, &state)?;
        }
        self.baseline_state = state;
        self.baseline_frame = Some(
            self.callbacks
                .frame()
                .cloned()
                .ok_or(CoreError::NoFrameAvailable)?,
        );
        self.baseline_kind = BaselineKind::Custom;
        self.refresh_av_info();
        self.refresh_shape_from_frame();
        Ok(())
    }

    pub fn frame_rgb(&self) -> Result<&[u8], CoreError> {
        self.callbacks
            .frame()
            .map(|frame| frame.rgb.as_slice())
            .ok_or(CoreError::NoFrameAvailable)
    }

    pub fn read_system_ram(&mut self, offset: usize, length: usize) -> Result<Vec<u8>, CoreError> {
        self.read_memory(MEMORY_SYSTEM_RAM, offset, length)
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

    fn capture_startup_baseline(&mut self) -> Result<(), CoreError> {
        // Capture one canonical startup baseline up front. Every reset then
        // restores this snapshot instead of replaying startup frames again.
        self.initialize_running_state()?;
        self.capture_current_baseline(BaselineKind::Startup)
    }

    fn load_baseline_from_file(&mut self, baseline_state_path: &Path) -> Result<(), CoreError> {
        self.initialize_running_state()?;
        let baseline_state =
            std::fs::read(baseline_state_path).map_err(|error| CoreError::ReadFile {
                path: baseline_state_path.to_path_buf(),
                message: error.to_string(),
            })?;
        let baseline_serial = self.callbacks.frame_serial();
        self.restore_state_bytes(&baseline_state)?;
        if self.callbacks.frame_serial() == baseline_serial {
            self.run_until_frame(baseline_serial, FRAME_WAIT_LIMIT)?;
        }
        self.capture_current_baseline(BaselineKind::Custom)
    }

    fn initialize_running_state(&mut self) -> Result<(), CoreError> {
        let baseline_serial = self.callbacks.frame_serial();
        self.call_core(|core| unsafe {
            core.reset();
        });
        self.run_until_frame(baseline_serial, FRAME_WAIT_LIMIT)
    }

    fn capture_current_baseline(&mut self, baseline_kind: BaselineKind) -> Result<(), CoreError> {
        self.refresh_av_info();
        self.refresh_shape_from_frame();
        self.baseline_state = self.serialize_state()?;
        self.baseline_frame = Some(
            self.callbacks
                .frame()
                .cloned()
                .ok_or(CoreError::NoFrameAvailable)?,
        );
        self.baseline_kind = baseline_kind;
        Ok(())
    }

    fn restore_baseline(&mut self) -> Result<(), CoreError> {
        let baseline_frame = self
            .baseline_frame
            .clone()
            .ok_or(CoreError::NoFrameAvailable)?;
        let baseline_state = self.baseline_state.clone();
        self.restore_state_bytes(&baseline_state)?;
        self.callbacks.set_frame(baseline_frame);
        Ok(())
    }

    fn restore_state_bytes(&mut self, state: &[u8]) -> Result<(), CoreError> {
        let restored =
            self.call_core(|core| unsafe { core.unserialize(state.as_ptr().cast(), state.len()) });
        if restored {
            Ok(())
        } else {
            Err(CoreError::UnserializeFailed)
        }
    }

    fn serialize_state(&mut self) -> Result<Vec<u8>, CoreError> {
        let size = self.call_core(|core| unsafe { core.serialize_size() });
        if size == 0 {
            return Err(CoreError::UnsupportedSaveState);
        }

        let mut state = vec![0_u8; size];
        let serialized = self
            .call_core(|core| unsafe { core.serialize(state.as_mut_ptr().cast(), state.len()) });
        if !serialized {
            return Err(CoreError::SerializeFailed);
        }
        Ok(state)
    }

    fn write_state_bytes(&self, path: &Path, state: &[u8]) -> Result<(), CoreError> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|error| CoreError::CreateDirectory {
                path: parent.to_path_buf(),
                message: error.to_string(),
            })?;
        }

        std::fs::write(path, state).map_err(|error| CoreError::WriteFile {
            path: path.to_path_buf(),
            message: error.to_string(),
        })
    }

    fn memory_size(&mut self, memory_id: u32) -> Result<usize, CoreError> {
        self.ensure_open()?;
        let size = self.call_core(|core| unsafe { core.memory_size(memory_id) });
        if size == 0 {
            return Err(CoreError::MemoryUnavailable { memory_id });
        }
        Ok(size)
    }

    fn read_memory(
        &mut self,
        memory_id: u32,
        offset: usize,
        length: usize,
    ) -> Result<Vec<u8>, CoreError> {
        self.ensure_open()?;
        let available = self.memory_size(memory_id)?;
        let end = offset
            .checked_add(length)
            .ok_or(CoreError::MemoryOutOfRange {
                memory_id,
                offset,
                length,
                available,
            })?;
        if end > available {
            return Err(CoreError::MemoryOutOfRange {
                memory_id,
                offset,
                length,
                available,
            });
        }

        let data = self.call_core(|core| unsafe { core.memory_data(memory_id) });
        if data.is_null() {
            return Err(CoreError::MemoryUnavailable { memory_id });
        }

        let bytes = unsafe { slice::from_raw_parts(data.cast::<u8>().add(offset), length) };
        Ok(bytes.to_vec())
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
