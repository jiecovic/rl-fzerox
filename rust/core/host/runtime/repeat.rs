// rust/core/host/runtime/repeat.rs
//! Repeated-step host methods used by training and watch playback.

use crate::core::error::CoreError;
use crate::core::stdio::with_silenced_stdio;

use super::host::Host;
use super::step::{
    DisplayFrameBatch, NativeMultiObservationStepResult, NativeStepResult, NativeWatchStepResult,
    ObservationFrameBatch, ObservationRenderConfig, RepeatedStepConfig, StepStatus, StepSummary,
};
use super::step_accumulator::StepAccumulator;

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
        let summary = self.execute_repeated_step_summary(config)?;
        let (status, final_telemetry) = self.finalize_repeated_step(config, &summary)?;
        let observation = self.render_observation_from_config(observation_config)?;
        Ok(NativeStepResult {
            observation,
            summary,
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
        let summary = self.execute_repeated_step_summary(config)?;
        let (status, final_telemetry) = self.finalize_repeated_step(config, &summary)?;
        let mut observations = ObservationFrameBatch::with_capacity(observation_configs.len());
        for observation_config in observation_configs {
            observations.push_frame(self.render_observation_from_config(*observation_config)?);
        }
        Ok(NativeMultiObservationStepResult {
            observations,
            summary,
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
    ) -> Result<NativeWatchStepResult<'_>, CoreError> {
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
        let step_result = with_silenced_stdio(|| {
            let mut accumulator = StepAccumulator::new(&initial_sample, config, self.frame_index);
            let mut display_frames = DisplayFrameBatch::default();
            let mut display_controller_masks = Vec::with_capacity(config.action_repeat);

            for _ in 0..config.action_repeat {
                self.callbacks.set_capture_video(true);
                let controller_state = self.controller_for_repeated_frame(
                    base_controller_state,
                    spin_controls_active,
                    &mut spin_stats,
                    config.spin_cooldown_frames,
                );
                display_controller_masks.push(controller_state.race_control_mask());
                if config.lean_timer_assist {
                    self.patch_lean_timers_for_slide_assist(controller_state)?;
                }
                self.call_core(|core| unsafe {
                    core.run();
                });
                self.frame_index += 1;

                let telemetry = self.telemetry_sample()?;
                accumulator.observe(&telemetry, self.frame_index);
                let display_frame = self.display_frame(observation_config.layout)?;
                display_frames.reserve_frame_capacity(display_frame.len(), config.action_repeat);
                display_frames.push_frame(display_frame);
            }

            Ok((
                self.finish_repeated_step(accumulator, spin_stats),
                display_frames,
                display_controller_masks,
            ))
        });
        self.callbacks.set_capture_video(true);
        self.refresh_shape_from_frame();

        let (summary, display_frames, display_controller_masks) = step_result?;
        if !self.callbacks.has_frame() {
            return Err(CoreError::NoFrameAvailable);
        }

        let (status, final_telemetry) = self.finalize_repeated_step(config, &summary)?;
        let observation = self.render_observation_from_config(observation_config)?;
        Ok(NativeWatchStepResult {
            observation,
            display_frames,
            display_controller_masks,
            summary,
            status,
            final_telemetry,
        })
    }

    fn execute_repeated_step_summary(
        &mut self,
        config: RepeatedStepConfig,
    ) -> Result<StepSummary, CoreError> {
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
        let step_result = with_silenced_stdio(|| {
            let mut accumulator = StepAccumulator::new(&initial_sample, config, self.frame_index);

            for repeat_index in 0..config.action_repeat {
                let capture_video = repeat_index + 1 == config.action_repeat;
                self.callbacks.set_capture_video(capture_video);
                let controller_state = self.controller_for_repeated_frame(
                    base_controller_state,
                    spin_controls_active,
                    &mut spin_stats,
                    config.spin_cooldown_frames,
                );
                if config.lean_timer_assist {
                    self.patch_lean_timers_for_slide_assist(controller_state)?;
                }
                self.call_core(|core| unsafe {
                    core.run();
                });
                self.frame_index += 1;

                let telemetry = self.telemetry_sample()?;
                accumulator.observe(&telemetry, self.frame_index);
            }

            Ok(self.finish_repeated_step(accumulator, spin_stats))
        });
        self.callbacks.set_capture_video(true);
        self.refresh_shape_from_frame();

        let summary = step_result?;
        if !self.callbacks.has_frame() {
            return Err(CoreError::NoFrameAvailable);
        }
        Ok(summary)
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
        base_controller_state: crate::core::input::ControllerState,
        spin_controls_active: bool,
        spin_stats: &mut super::spin::SpinStepStats,
        spin_cooldown_frames: usize,
    ) -> crate::core::input::ControllerState {
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
        spin_stats: super::spin::SpinStepStats,
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
