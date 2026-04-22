// rust/core/host/runtime/step_accumulator.rs
//! Step-local aggregation for one repeated env step.
//!
//! The accumulator is reset once per outer env step, then updated from each
//! internal telemetry sample the host observes while holding the chosen action.

use crate::core::telemetry::StepTelemetrySample;

use super::step::{RepeatedStepConfig, StepSummary};

/// Collect step-level aggregates across repeated internal emulator frames.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
pub(super) struct StepAccumulator {
    summary: StepSummary,
    previous_energy: f32,
    previous_state_flags: u32,
    stuck_min_speed_kph: f32,
    energy_loss_epsilon: f32,
}

#[cfg_attr(not(test), allow(dead_code))]
impl StepAccumulator {
    /// Start one fresh step accumulator from the telemetry state before the
    /// repeated inner-frame loop begins.
    pub(super) fn new(
        telemetry: &StepTelemetrySample,
        config: RepeatedStepConfig,
        frame_index: usize,
    ) -> Self {
        Self {
            summary: StepSummary {
                max_race_distance: telemetry.race_distance,
                final_frame_index: frame_index,
                ..StepSummary::default()
            },
            previous_energy: telemetry.energy,
            previous_state_flags: telemetry.state_flags,
            stuck_min_speed_kph: config.stuck_min_speed_kph,
            energy_loss_epsilon: config.energy_loss_epsilon,
        }
    }

    /// Incorporate one internal telemetry sample into the current env-step
    /// aggregates.
    pub(super) fn observe(&mut self, telemetry: &StepTelemetrySample, frame_index: usize) {
        self.summary.frames_run += 1;
        self.summary.final_frame_index = frame_index;
        self.summary.max_race_distance =
            self.summary.max_race_distance.max(telemetry.race_distance);
        if telemetry.reverse_timer > 0 {
            self.summary.reverse_active_frames += 1;
        }
        if telemetry.damage_taken() {
            self.summary.damage_taken_frames += 1;
        }
        if telemetry.airborne() {
            self.summary.airborne_frames += 1;
        }

        let energy_delta = telemetry.energy - self.previous_energy;
        if energy_delta < -self.energy_loss_epsilon {
            self.summary.energy_loss_total += -energy_delta;
        } else if energy_delta > self.energy_loss_epsilon {
            self.summary.energy_gain_total += energy_delta;
        }

        if telemetry.speed_kph < self.stuck_min_speed_kph {
            self.summary.low_speed_frames += 1;
            self.summary.consecutive_low_speed_frames += 1;
        } else {
            self.summary.consecutive_low_speed_frames = 0;
        }

        self.summary.entered_state_flags |= telemetry.state_flags & !self.previous_state_flags;

        self.previous_energy = telemetry.energy;
        self.previous_state_flags = telemetry.state_flags;
    }

    /// Finish the current repeated env step and return the collected summary.
    pub(super) fn finish(self) -> StepSummary {
        self.summary
    }
}
