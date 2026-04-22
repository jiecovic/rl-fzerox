// rust/core/host/runtime/repeat.rs
//! Repeated-step host methods used by training and watch playback.

use crate::core::error::CoreError;
use crate::core::stdio::with_silenced_stdio;

use super::host::Host;
use super::step::{NativeStepResult, NativeWatchStepResult, RepeatedStepConfig, StepStatus};
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
        let observation = self.observation_frame(
            config.preset,
            config.frame_stack,
            config.stack_mode,
            config.minimap_layer,
            config.resize_filter,
            config.minimap_resize_filter,
        )?;
        Ok(NativeStepResult {
            observation,
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
    #[allow(dead_code)]
    pub fn step_repeat_watch_raw(
        &mut self,
        config: RepeatedStepConfig,
    ) -> Result<NativeWatchStepResult<'_>, CoreError> {
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
            let mut display_frames = Vec::with_capacity(config.action_repeat);

            for _ in 0..config.action_repeat {
                self.callbacks.set_capture_video(true);
                if config.lean_timer_assist {
                    self.patch_lean_timers_for_slide_assist(config.controller_state)?;
                }
                self.call_core(|core| unsafe {
                    core.run();
                });
                self.frame_index += 1;

                let telemetry = self.telemetry_sample()?;
                accumulator.observe(&telemetry, self.frame_index);
                display_frames.push(self.display_frame(config.preset)?.to_vec());
            }

            Ok((accumulator.finish(), display_frames))
        });
        self.callbacks.set_capture_video(true);
        self.refresh_shape_from_frame();

        let (summary, display_frames) = step_result?;
        if !self.callbacks.has_frame() {
            return Err(CoreError::NoFrameAvailable);
        }

        let final_telemetry = self.telemetry()?;
        let status = StepStatus::from_step(self.step_counters, &summary, &final_telemetry, config);
        self.step_counters = status.counters;
        let observation = self.observation_frame(
            config.preset,
            config.frame_stack,
            config.stack_mode,
            config.minimap_layer,
            config.resize_filter,
            config.minimap_resize_filter,
        )?;
        Ok(NativeWatchStepResult {
            observation,
            display_frames,
            summary,
            status,
            final_telemetry,
        })
    }
}
