// rust/core/host/runtime/repeat.rs
//! Repeated-step host methods used by training and watch playback.

use crate::core::error::CoreError;
use crate::core::input::ControllerState;
use crate::core::stdio::with_silenced_stdio;

use super::host::Host;
use super::spin::SpinStepStats;
use super::step::{
    AudioFrameBatch, DisplayFrameBatch, NativeMultiObservationStepResult, NativeStepResult,
    NativeWatchStepResult, ObservationFrameBatch, ObservationRenderConfig, RepeatedStepConfig,
    StepStatus, StepSummary,
};
use super::step_accumulator::StepAccumulator;

#[derive(Clone, Copy, Debug)]
enum RepeatedStepCapture {
    /// Training path: capture only the final framebuffer needed for the next
    /// observation and avoid per-frame display/audio work.
    FinalFrame,
    /// Watch path: capture every repeated inner frame for smooth playback.
    WatchFrames {
        observation_config: ObservationRenderConfig,
        capture_audio: bool,
    },
}

#[derive(Debug)]
struct RepeatedStepExecution {
    summary: StepSummary,
    display_frames: DisplayFrameBatch,
    display_controller_masks: Vec<u16>,
    audio_frames: AudioFrameBatch,
}

impl RepeatedStepCapture {
    fn capture_video(self, repeat_index: usize, action_repeat: usize) -> bool {
        match self {
            Self::FinalFrame => repeat_index + 1 == action_repeat,
            Self::WatchFrames { .. } => true,
        }
    }

    fn capture_audio(self) -> Option<bool> {
        match self {
            Self::FinalFrame => None,
            Self::WatchFrames { capture_audio, .. } => Some(capture_audio),
        }
    }

    fn watch_observation_config(self) -> Option<ObservationRenderConfig> {
        match self {
            Self::FinalFrame => None,
            Self::WatchFrames {
                observation_config, ..
            } => Some(observation_config),
        }
    }
}

impl RepeatedStepExecution {
    fn new(summary: StepSummary, action_repeat: usize, capture: RepeatedStepCapture) -> Self {
        Self {
            summary,
            display_frames: DisplayFrameBatch::default(),
            display_controller_masks: match capture {
                RepeatedStepCapture::FinalFrame => Vec::new(),
                RepeatedStepCapture::WatchFrames { .. } => Vec::with_capacity(action_repeat),
            },
            audio_frames: AudioFrameBatch::default(),
        }
    }
}

impl Host {
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
    pub fn step_repeat_raw(
        &mut self,
        config: RepeatedStepConfig,
        observation_config: ObservationRenderConfig,
    ) -> Result<NativeStepResult<'_>, CoreError> {
        let execution = self.execute_repeated_step(config, RepeatedStepCapture::FinalFrame)?;
        let (status, final_telemetry) = self.finalize_repeated_step(config, &execution.summary)?;
        let observation = self.render_observation_from_config(observation_config)?;
        Ok(NativeStepResult {
            observation,
            summary: execution.summary,
            status,
            final_telemetry,
        })
    }

    /// Execute one outer env step and render multiple native observations from
    /// the resulting post-step state.
    pub fn step_repeat_multi_observation_raw(
        &mut self,
        config: RepeatedStepConfig,
        observation_configs: &[ObservationRenderConfig],
    ) -> Result<NativeMultiObservationStepResult, CoreError> {
        let execution = self.execute_repeated_step(config, RepeatedStepCapture::FinalFrame)?;
        let (status, final_telemetry) = self.finalize_repeated_step(config, &execution.summary)?;
        let mut observations = ObservationFrameBatch::with_capacity(observation_configs.len());
        for observation_config in observation_configs {
            observations.push_frame(self.render_observation_from_config(*observation_config)?);
        }
        Ok(NativeMultiObservationStepResult {
            observations,
            summary: execution.summary,
            status,
            final_telemetry,
        })
    }

    /// Execute one repeated step and capture a display frame after each inner frame.
    ///
    /// This is intentionally separate from `step_repeat_raw`: watch mode needs
    /// smooth visual playback when actions are held for multiple frames, while
    /// training and benchmarks should avoid those extra frame copies.
    pub fn step_repeat_watch_raw(
        &mut self,
        config: RepeatedStepConfig,
        observation_config: ObservationRenderConfig,
        capture_audio: bool,
    ) -> Result<NativeWatchStepResult<'_>, CoreError> {
        let execution = self.execute_repeated_step(
            config,
            RepeatedStepCapture::WatchFrames {
                observation_config,
                capture_audio,
            },
        )?;
        let (status, final_telemetry) = self.finalize_repeated_step(config, &execution.summary)?;
        let observation = self.render_observation_from_config(observation_config)?;
        Ok(NativeWatchStepResult {
            observation,
            display_frames: execution.display_frames,
            display_controller_masks: execution.display_controller_masks,
            audio_frames: execution.audio_frames,
            summary: execution.summary,
            status,
            final_telemetry,
        })
    }

    fn execute_repeated_step(
        &mut self,
        config: RepeatedStepConfig,
        capture: RepeatedStepCapture,
    ) -> Result<RepeatedStepExecution, CoreError> {
        self.ensure_open()?;
        if config.action_repeat == 0 {
            return Err(CoreError::InvalidStepRepeatCount {
                count: config.action_repeat,
            });
        }

        let initial_sample = self.telemetry_sample()?;
        let mut spin_stats = self.spin_macro.begin_step(config.spin_request);
        let spin_controls_active = self.spin_controls_active(spin_stats.started);
        let base_controller_state = config.race_controls.to_controller_state();
        if !spin_controls_active {
            self.callbacks.set_controller_state(base_controller_state);
        }
        let watch_capture_audio = capture.capture_audio();
        let step_result: Result<RepeatedStepExecution, CoreError> = with_silenced_stdio(|| {
            let mut accumulator = StepAccumulator::new(&initial_sample, config, self.frame_index);
            let mut execution =
                RepeatedStepExecution::new(StepSummary::default(), config.action_repeat, capture);
            if let Some(capture_audio) = watch_capture_audio {
                self.callbacks.set_capture_audio(capture_audio);
            }

            for repeat_index in 0..config.action_repeat {
                self.callbacks
                    .set_capture_video(capture.capture_video(repeat_index, config.action_repeat));
                if watch_capture_audio.is_some_and(|enabled| enabled) {
                    self.callbacks.clear_audio_samples();
                }
                let controller_state = self.controller_for_repeated_frame(
                    base_controller_state,
                    spin_controls_active,
                    &mut spin_stats,
                    config.spin_cooldown_frames,
                );
                if config.lean_timer_assist {
                    self.patch_lean_timers_for_slide_assist(controller_state)?;
                }
                self.run_core_frame();
                if watch_capture_audio.is_some_and(|enabled| enabled) {
                    execution
                        .audio_frames
                        .push_frame_samples(self.callbacks.drain_audio_samples());
                }
                self.frame_index += 1;

                let telemetry = self.telemetry_sample()?;
                accumulator.observe(&telemetry, self.frame_index);
                self.capture_watch_frame_outputs(
                    capture,
                    controller_state,
                    config.action_repeat,
                    &mut execution,
                )?;
            }

            execution.summary = self.finish_repeated_step(accumulator, spin_stats);
            Ok(execution)
        });
        if watch_capture_audio.is_some() {
            self.callbacks.set_capture_audio(false);
        }
        self.callbacks.set_capture_video(true);
        self.refresh_shape_from_frame();

        let execution = step_result?;
        if !self.callbacks.has_frame() {
            return Err(CoreError::NoFrameAvailable);
        }
        Ok(execution)
    }

    fn capture_watch_frame_outputs(
        &mut self,
        capture: RepeatedStepCapture,
        controller_state: ControllerState,
        action_repeat: usize,
        execution: &mut RepeatedStepExecution,
    ) -> Result<(), CoreError> {
        let Some(observation_config) = capture.watch_observation_config() else {
            return Ok(());
        };

        execution
            .display_controller_masks
            .push(controller_state.race_control_mask());
        let display_frame = self.display_frame(observation_config.layout)?;
        execution
            .display_frames
            .reserve_frame_capacity(display_frame.len(), action_repeat);
        execution.display_frames.push_frame(display_frame);
        Ok(())
    }

    fn spin_controls_active(&self, spin_started: bool) -> bool {
        if spin_started {
            return true;
        }
        let spin_status = self.spin_macro.status();
        spin_status.active || spin_status.cooldown_frames > 0
    }

    fn controller_for_repeated_frame(
        &mut self,
        base_controller_state: ControllerState,
        spin_controls_active: bool,
        spin_stats: &mut SpinStepStats,
        spin_cooldown_frames: usize,
    ) -> ControllerState {
        if !spin_controls_active {
            return base_controller_state;
        }
        let frame_controller = self
            .spin_macro
            .next_controller(base_controller_state, spin_cooldown_frames);
        if frame_controller.macro_owns_lean {
            spin_stats.active_frames += 1;
            spin_stats.lean_owned_frames += 1;
        }
        self.callbacks
            .set_controller_state(frame_controller.controller_state);
        frame_controller.controller_state
    }

    fn finish_repeated_step(
        &self,
        accumulator: StepAccumulator,
        spin_stats: SpinStepStats,
    ) -> StepSummary {
        let mut summary = accumulator.finish();
        summary.spin_macro_started = spin_stats.started;
        summary.spin_macro_active_frames = spin_stats.active_frames;
        summary.lean_macro_owned_frames = spin_stats.lean_owned_frames;
        summary
    }

    fn finalize_repeated_step(
        &mut self,
        config: RepeatedStepConfig,
        summary: &StepSummary,
    ) -> Result<(StepStatus, crate::core::telemetry::TelemetrySnapshot), CoreError> {
        let final_telemetry = self.telemetry()?;
        let status = StepStatus::from_step_with_spin(
            self.step_counters,
            summary,
            &final_telemetry,
            config,
            self.spin_macro.status(),
        );
        self.step_counters = status.counters;
        Ok((status, final_telemetry))
    }

    fn render_observation_from_config(
        &mut self,
        observation_config: ObservationRenderConfig,
    ) -> Result<&[u8], CoreError> {
        self.observation_frame(
            observation_config.layout,
            observation_config.frame_stack,
            observation_config.stack_mode,
            observation_config.minimap_layer,
            observation_config.resize_filter,
            observation_config.minimap_resize_filter,
        )
    }
}
