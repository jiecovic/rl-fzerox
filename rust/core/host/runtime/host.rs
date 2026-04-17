// rust/core/host/runtime/host.rs
//! Main `Host` façade over one loaded libretro core instance.

use std::ffi::CString;
use std::path::{Path, PathBuf};

use libretro_sys::MEMORY_SYSTEM_RAM;

use crate::core::api::LoadedCore;
use crate::core::callbacks::{CallbackGuard, CallbackState};
use crate::core::error::CoreError;
use crate::core::input::ControllerState;
use crate::core::observation::{ObservationCropProfile, ObservationPreset, ObservationSpec};
use crate::core::stdio::with_silenced_stdio;
use crate::core::telemetry::{StepTelemetrySample, TelemetrySnapshot};
use crate::core::video::VideoFrame;

use super::step::{NativeStepResult, RepeatedStepConfig, StepCounters, StepStatus};
use super::step_accumulator::StepAccumulator;

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

        let core = LoadedCore::load(core_path)?;
        let callbacks = Box::new(CallbackState::new(core_path, runtime_dir, renderer)?);
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

    pub fn system_ram_size(&mut self) -> Result<usize, CoreError> {
        self.memory_size(MEMORY_SYSTEM_RAM)
    }

    pub fn baseline_kind(&self) -> &'static str {
        match self.baseline_kind {
            BaselineKind::Startup => "startup",
            BaselineKind::Custom => "custom",
        }
    }

    pub fn load_baseline(&mut self, path: &Path) -> Result<(), CoreError> {
        self.ensure_open()?;
        self.load_baseline_from_file(path)
    }

    pub fn load_baseline_bytes(&mut self, state: &[u8]) -> Result<(), CoreError> {
        self.ensure_open()?;
        self.load_baseline_from_state_bytes(state)
    }

    pub fn reset(&mut self) -> Result<(), CoreError> {
        self.ensure_open()?;
        self.restore_baseline()?;
        self.callbacks
            .set_controller_state(ControllerState::default());
        self.frame_index = 0;
        self.step_counters = StepCounters::default();
        self.refresh_shape_from_frame();
        Ok(())
    }

    pub fn step_frames(&mut self, count: usize, capture_video: bool) -> Result<(), CoreError> {
        self.ensure_open()?;
        self.callbacks.set_capture_video(capture_video);
        let run_frames = || {
            for _ in 0..count {
                self.call_core(|core| unsafe {
                    core.run();
                });
            }
        };
        with_silenced_stdio(run_frames);
        self.callbacks.set_capture_video(true);
        self.frame_index += count;
        self.refresh_shape_from_frame();
        if capture_video && !self.callbacks.has_frame() {
            return Err(CoreError::NoFrameAvailable);
        }
        Ok(())
    }

    /// Execute one outer env step fully inside the host runtime.
    ///
    /// The chosen controller state is held for exactly `action_repeat`
    /// internal frames. The host aggregates step-level telemetry features and
    /// stop-state counters across those frames, then returns the final stacked
    /// observation once at the end.
    ///
    /// We intentionally capture video only on the last repeated frame. That
    /// keeps the common path fast at the cost of not returning the exact first
    /// terminal/truncation frame image when a stop condition occurs earlier in
    /// the repeated step.
    #[allow(dead_code)]
    pub fn step_repeat_raw(
        &mut self,
        config: RepeatedStepConfig,
    ) -> Result<NativeStepResult<'_>, CoreError> {
        self.ensure_open()?;
        if config.action_repeat == 0 {
            return Err(CoreError::InvalidStepRepeatCount {
                count: config.action_repeat,
            });
        }

        let initial_sample = self.telemetry_sample()?;
        self.callbacks.set_controller_state(config.controller_state);
        let step_result = with_silenced_stdio(|| {
            let mut accumulator = StepAccumulator::new(&initial_sample, config, self.frame_index);

            for repeat_index in 0..config.action_repeat {
                let capture_video = repeat_index + 1 == config.action_repeat;
                self.callbacks.set_capture_video(capture_video);
                if config.lean_timer_assist {
                    self.patch_lean_timers_for_slide_assist(config.controller_state)?;
                }
                self.call_core(|core| unsafe {
                    core.run();
                });
                self.frame_index += 1;

                let telemetry = self.telemetry_sample()?;
                accumulator.observe(&telemetry, self.frame_index);
            }

            Ok(accumulator.finish())
        });
        self.callbacks.set_capture_video(true);
        self.refresh_shape_from_frame();

        let summary = step_result?;
        if !self.callbacks.has_frame() {
            return Err(CoreError::NoFrameAvailable);
        }

        let final_telemetry = self.telemetry()?;
        let status = StepStatus::from_step(self.step_counters, &summary, &final_telemetry, config);
        self.step_counters = status.counters;
        let observation = self.observation_frame(config.preset, config.frame_stack)?;
        Ok(NativeStepResult {
            observation,
            summary,
            status,
            final_telemetry,
        })
    }

    pub fn set_controller_state(
        &mut self,
        controller_state: ControllerState,
    ) -> Result<(), CoreError> {
        self.ensure_open()?;
        self.callbacks.set_controller_state(controller_state);
        Ok(())
    }

    pub fn save_state(&mut self, path: &Path) -> Result<(), CoreError> {
        self.ensure_open()?;
        self.save_state_to_path(path)
    }

    pub fn capture_current_as_baseline(
        &mut self,
        save_path: Option<&Path>,
    ) -> Result<(), CoreError> {
        self.ensure_open()?;
        self.capture_current_as_baseline_to_path(save_path)
    }

    pub fn frame_rgb(&mut self) -> Result<&[u8], CoreError> {
        self.callbacks
            .frame()
            .map(|frame| frame.rgb.as_slice())
            .ok_or(CoreError::NoFrameAvailable)
    }

    pub fn read_system_ram(&mut self, offset: usize, length: usize) -> Result<Vec<u8>, CoreError> {
        self.read_memory(MEMORY_SYSTEM_RAM, offset, length)
    }

    pub fn telemetry(&mut self) -> Result<TelemetrySnapshot, CoreError> {
        let system_ram = self.system_ram_slice()?;
        crate::core::telemetry::read_snapshot(system_ram)
    }

    fn telemetry_sample(&mut self) -> Result<StepTelemetrySample, CoreError> {
        let system_ram = self.system_ram_slice()?;
        crate::core::telemetry::read_step_sample(system_ram)
    }

    pub fn observation_spec(
        &self,
        preset: ObservationPreset,
    ) -> Result<ObservationSpec, CoreError> {
        let (frame_height, frame_width, _) = self.frame_shape;
        preset.resolve(
            frame_width,
            frame_height,
            self.display_aspect_ratio,
            self.observation_crop_profile,
        )
    }

    pub fn observation_frame(
        &mut self,
        preset: ObservationPreset,
        frame_stack: usize,
    ) -> Result<&[u8], CoreError> {
        let spec = self.observation_spec(preset)?;
        self.callbacks.stacked_observation_frame(
            preset.observation_aspect_ratio(self.display_aspect_ratio),
            spec.frame_width,
            spec.frame_height,
            spec.channels == 3,
            preset.crop(self.observation_crop_profile),
            frame_stack,
        )
    }

    pub fn display_frame(&mut self, preset: ObservationPreset) -> Result<&[u8], CoreError> {
        let spec = self.observation_spec(preset)?;
        self.callbacks.observation_frame(
            self.display_aspect_ratio,
            spec.display_width,
            spec.display_height,
            true,
            preset.crop(self.observation_crop_profile),
        )
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
