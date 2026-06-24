// rust/core/host/runtime/host.rs
//! Main `Host` façade over one loaded libretro core instance.

use std::ffi::{CString, c_void};
use std::path::{Path, PathBuf};

use libretro_sys::{
    AudioSampleBatchFn, AudioSampleFn, EnvironmentFn, GameInfo, InputPollFn, InputStateFn,
    SystemAvInfo, VideoRefreshFn,
};

use crate::core::api::LoadedCore;
use crate::core::callbacks::{CallbackGuard, CallbackState};
use crate::core::error::CoreError;
use crate::core::observation::ObservationCropProfile;
use crate::core::rom::validate_supported_rom;
use crate::core::stdio::with_silenced_stdio;
use crate::core::video::VideoFrame;

use super::spin::SpinMacroState;
use super::step::StepCounters;

// Maximum number of `core.run()` calls we allow while waiting for a reset/load
// path to produce a fresh frame before treating it as a core failure.
pub(super) const FRAME_WAIT_LIMIT: usize = 120;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BaselineKind {
    Startup,
    Custom,
}

pub struct Host {
    pub(super) core: LoadedCore,
    pub(super) callbacks: Box<CallbackState>,
    pub(super) rom_path: PathBuf,
    pub(super) rom_path_c_string: CString,
    pub(super) rom_bytes: Vec<u8>,
    pub(super) baseline_state: Vec<u8>,
    pub(super) baseline_frame: Option<VideoFrame>,
    pub(super) baseline_kind: BaselineKind,
    pub(super) display_aspect_ratio: f64,
    pub(super) observation_crop_profile: ObservationCropProfile,
    pub(super) native_fps: f64,
    pub(super) native_sample_rate: f64,
    pub(super) frame_shape: (usize, usize, usize),
    pub(super) frame_index: usize,
    pub(super) system_ram_size: usize,
    pub(super) step_counters: StepCounters,
    pub(super) spin_macro: SpinMacroState,
    pub(super) closed: bool,
}

impl Host {
    /// Open a core, load the ROM, and establish the reset baseline used by
    /// later episode resets.
    pub fn open(
        core_path: &Path,
        rom_path: &Path,
        runtime_dir: Option<&Path>,
        baseline_state_path: Option<&Path>,
        renderer: &str,
    ) -> Result<Self, CoreError> {
        if !rom_path.is_file() {
            return Err(CoreError::MissingRom(rom_path.to_path_buf()));
        }

        let rom_bytes = std::fs::read(rom_path).map_err(|error| CoreError::ReadFile {
            path: rom_path.to_path_buf(),
            source: error,
        })?;
        validate_supported_rom(rom_path, &rom_bytes)?;

        let core = LoadedCore::load(core_path)?;
        let callbacks = Box::new(CallbackState::new(core_path, runtime_dir, renderer)?);
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
            observation_crop_profile: ObservationCropProfile::from_renderer_name(renderer),
            native_fps: 0.0,
            native_sample_rate: 0.0,
            frame_shape: (0, 0, 3),
            frame_index: 0,
            system_ram_size: 0,
            step_counters: StepCounters::default(),
            spin_macro: SpinMacroState::default(),
            closed: false,
        };
        host.configure_callbacks();
        host.initialize();
        host.load_game()?;
        host.cache_system_ram_size()?;
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

    pub fn native_sample_rate(&self) -> f64 {
        self.native_sample_rate
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

    pub fn baseline_kind(&self) -> &'static str {
        match self.baseline_kind {
            BaselineKind::Startup => "startup",
            BaselineKind::Custom => "custom",
        }
    }

    pub fn close(&mut self) {
        if self.closed {
            return;
        }

        self.destroy_hardware_context();
        self.unload_core_game();
        self.deinit_core();
        self.closed = true;
    }

    pub(super) fn call_core<R>(&mut self, action: impl FnOnce(&LoadedCore) -> R) -> R {
        // Many libretro entry points synchronously call back into the frontend.
        // Installing the active callback state around every core call keeps that
        // routing explicit and easy to audit.
        let callbacks = self.callbacks.as_mut() as *mut CallbackState;
        let _guard = CallbackGuard::activate(callbacks);
        action(&self.core)
    }

    fn destroy_hardware_context(&mut self) {
        let callbacks = self.callbacks.as_mut() as *mut CallbackState;
        let _guard = CallbackGuard::activate(callbacks);
        with_silenced_stdio(|| self.callbacks.destroy_hardware_context());
    }

    pub(super) fn ensure_open(&self) -> Result<(), CoreError> {
        if self.closed {
            return Err(CoreError::AlreadyClosed);
        }
        Ok(())
    }

    pub(super) fn install_core_callbacks(
        &mut self,
        environment: EnvironmentFn,
        video_refresh: VideoRefreshFn,
        audio_sample: AudioSampleFn,
        audio_sample_batch: AudioSampleBatchFn,
        input_poll: InputPollFn,
        input_state: InputStateFn,
    ) {
        self.call_core(|core| {
            // SAFETY: The callback functions are static extern "C" shims with
            // signatures required by libretro. `call_core` installs the active
            // callback state for the duration of this synchronous core call.
            unsafe {
                core.set_environment(environment);
                core.set_video_refresh(video_refresh);
                core.set_audio_sample(audio_sample);
                core.set_audio_sample_batch(audio_sample_batch);
                core.set_input_poll(input_poll);
                core.set_input_state(input_state);
            }
        });
    }

    pub(super) fn init_core(&mut self) {
        self.call_core(|core| {
            // SAFETY: The libretro core is loaded and its callback state is
            // active via `call_core`; initialization is synchronous.
            unsafe {
                core.init();
            }
        });
    }

    pub(super) fn deinit_core(&mut self) {
        self.call_core(|core| {
            // SAFETY: The core remains loaded while `Host` owns it. This is
            // only called during close/drop after game resources are released.
            unsafe {
                core.deinit();
            }
        });
    }

    pub(super) fn load_core_game(&mut self, game: &GameInfo) -> bool {
        self.call_core(|core| {
            // SAFETY: `game` points at `Host`-owned ROM/path buffers that live
            // at least for this synchronous libretro load call.
            unsafe { core.load_game(game) }
        })
    }

    pub(super) fn unload_core_game(&mut self) {
        self.call_core(|core| {
            // SAFETY: The loaded game belongs to this host/core pair and is
            // unloaded before the core is deinitialized.
            unsafe {
                core.unload_game();
            }
        });
    }

    pub(super) fn set_core_controller_port_device(&mut self, port: u32, device: u32) {
        self.call_core(|core| {
            // SAFETY: `port` and `device` are libretro wire values supplied by
            // this runtime before or during normal emulation.
            unsafe {
                core.set_controller_port_device(port, device);
            }
        });
    }

    pub(super) fn core_system_av_info(&mut self) -> SystemAvInfo {
        self.call_core(|core| {
            // SAFETY: The core writes into an internal stack value owned by the
            // wrapper and returns it by value.
            unsafe { core.system_av_info() }
        })
    }

    pub(super) fn reset_core(&mut self) {
        self.call_core(|core| {
            // SAFETY: Reset is a synchronous libretro lifecycle call while the
            // active callback state is installed by `call_core`.
            unsafe {
                core.reset();
            }
        });
    }

    pub(super) fn run_core_frame(&mut self) {
        self.call_core(|core| {
            // SAFETY: One emulated frame is run while callbacks and frontend
            // state are active for the duration of the call.
            unsafe {
                core.run();
            }
        });
    }

    pub(super) fn core_serialize_size(&mut self) -> usize {
        self.call_core(|core| {
            // SAFETY: Query-only libretro call with no raw buffers involved.
            unsafe { core.serialize_size() }
        })
    }

    pub(super) fn serialize_core_state(&mut self, state: &mut [u8]) -> bool {
        self.call_core(|core| {
            // SAFETY: `state` is a valid writable byte slice of the size just
            // reported by the core and lives for this synchronous call.
            unsafe { core.serialize(state.as_mut_ptr().cast(), state.len()) }
        })
    }

    pub(super) fn unserialize_core_state(&mut self, state: &[u8]) -> bool {
        self.call_core(|core| {
            // SAFETY: `state` is an immutable byte slice containing a savestate
            // blob owned by the host for the duration of this synchronous call.
            unsafe { core.unserialize(state.as_ptr().cast(), state.len()) }
        })
    }

    pub(super) fn core_memory_size(&mut self, memory_id: u32) -> usize {
        self.call_core(|core| {
            // SAFETY: Query-only libretro memory call for a wire memory id.
            unsafe { core.memory_size(memory_id) }
        })
    }

    pub(super) fn core_memory_data(&mut self, memory_id: u32) -> *mut c_void {
        self.call_core(|core| {
            // SAFETY: The returned pointer is owned by the core. Callers must
            // check for null and bound any slice by `core_memory_size`.
            unsafe { core.memory_data(memory_id) }
        })
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        self.close();
    }
}
