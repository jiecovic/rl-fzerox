// rust/core/host/runtime/step_accumulator.rs
//! Step-local aggregation for one repeated env step.
//!
//! The accumulator is reset once per outer env step, then updated from each
//! internal telemetry sample the host observes while holding the chosen action.

use crate::core::telemetry::{
    CourseEffect, StepTelemetrySample, course_effect_raw_from_state_flags,
};

use super::step::{RepeatedStepConfig, StepSummary};

/// Collect step-level aggregates across repeated internal emulator frames.
#[derive(Debug)]
pub(super) struct StepAccumulator {
    summary: StepSummary,
    previous_energy: f32,
    previous_state_flags: u32,
    stuck_min_speed_kph: f32,
    energy_loss_epsilon: f32,
}

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
                max_race_distance_speed_kph: telemetry.speed_kph,
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
        if telemetry.race_distance > self.summary.max_race_distance {
            self.summary.max_race_distance = telemetry.race_distance;
            self.summary.max_race_distance_speed_kph = telemetry.speed_kph;
        }
        if telemetry.reverse_timer > 0 {
            self.summary.reverse_active_frames += 1;
        }
        let collision_recoil = telemetry.collision_recoil();
        let damage_taken = telemetry.damage_taken();
        if collision_recoil {
            self.summary.collision_recoil_active_frames += 1;
        }
        if damage_taken {
            self.summary.damage_taken_frames += 1;
        }
        if collision_recoil || damage_taken {
            self.summary.impact_frames += 1;
        }
        if telemetry.airborne() {
            self.summary.airborne_frames += 1;
        }
        if telemetry.outside_track_bounds() {
            self.summary.outside_track_min_height_above_ground =
                Some(match self.summary.outside_track_min_height_above_ground {
                    Some(previous) => previous.min(telemetry.height_above_ground),
                    None => telemetry.height_above_ground,
                });
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
        self.record_entered_course_effect(telemetry.state_flags);

        self.previous_energy = telemetry.energy;
        self.previous_state_flags = telemetry.state_flags;
    }

    /// Finish the current repeated env step and return the collected summary.
    pub(super) fn finish(self) -> StepSummary {
        self.summary
    }

    fn record_entered_course_effect(&mut self, state_flags: u32) {
        let previous = course_effect_raw_from_state_flags(self.previous_state_flags);
        let current = course_effect_raw_from_state_flags(state_flags);
        if current == previous || current == CourseEffect::None as u32 {
            return;
        }
        self.summary.entered_course_effects |= 1 << current;
    }
}
