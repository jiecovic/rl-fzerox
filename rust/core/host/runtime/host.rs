// rust/core/host/runtime/host.rs
//! Main `Host` façade over one loaded libretro core instance.

use std::ffi::CString;
use std::path::{Path, PathBuf};

use crate::core::api::LoadedCore;
use crate::core::callbacks::{CallbackGuard, CallbackState};
use crate::core::error::CoreError;
use crate::core::observation::ObservationCropProfile;
use crate::core::rom::validate_supported_rom;
use crate::core::stdio::with_silenced_stdio;
use crate::core::video::VideoFrame;

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
    pub(super) frame_shape: (usize, usize, usize),
    pub(super) frame_index: usize,
    pub(super) step_counters: StepCounters,
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
            message: error.to_string(),
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
            frame_shape: (0, 0, 3),
            frame_index: 0,
            step_counters: StepCounters::default(),
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
        self.call_core(|core| unsafe {
            core.unload_game();
            core.deinit();
        });
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
}

impl Drop for Host {
    fn drop(&mut self) {
        self.close();
    }
}
