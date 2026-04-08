// rust/core/host/runtime/step.rs
//! Native env-step result types for the repeated-step host API.
//!
//! These types describe one outer RL step, not one internal emulator frame.
//! They intentionally carry only the step-level aggregates the current
//! `race_v2` reward and env limits need.

use crate::core::telemetry::TelemetrySnapshot;
use crate::core::{input::ControllerState, observation::ObservationPreset};

/// Aggregated per-step features collected across repeated internal frames.
///
/// This is the native summary Python reward trackers and limit trackers will
/// consume once the repeated step loop moves behind the Rust boundary.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepSummary {
    /// Number of internal frames actually executed for this env step.
    pub frames_run: usize,
    /// Highest race distance reached during this repeated step.
    pub max_race_distance: f32,
    /// Total reverse progress magnitude accumulated during this repeated step.
    pub reverse_progress_total: f32,
    /// Reverse-progress streak length at the end of the repeated step.
    pub consecutive_reverse_frames: usize,
    /// Total energy lost during the repeated step.
    pub energy_loss_total: f32,
    /// Low-speed streak length at the end of the repeated step.
    pub consecutive_low_speed_frames: usize,
    /// OR of state flags newly entered at least once during this env step.
    pub entered_state_flags: u32,
    /// Frame index after the repeated step completed.
    pub final_frame_index: usize,
}

/// Native env-step payload returned after executing a repeated step.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct NativeStepResult<'a> {
    /// Final stacked observation tensor for this outer env step.
    pub observation: &'a [u8],
    /// Aggregated step features spanning the internal repeated frames.
    pub summary: StepSummary,
    /// Final telemetry snapshot after the repeated step completed.
    pub final_telemetry: TelemetrySnapshot,
}

/// Native repeated-step request parameters.
#[derive(Clone, Copy, Debug)]
pub struct RepeatedStepConfig {
    /// Held controller state for the full outer env step.
    pub controller_state: ControllerState,
    /// Number of internal emulator frames to execute.
    pub action_repeat: usize,
    /// Observation preset used for the final stacked observation.
    pub preset: ObservationPreset,
    /// Number of observation frames stacked in the returned tensor.
    pub frame_stack: usize,
    /// Speed threshold used by the stuck limit tracker.
    pub stuck_min_speed_kph: f32,
    /// Epsilon used for reward-side reverse-progress aggregation.
    pub reverse_progress_epsilon: f32,
    /// Epsilon used for reward-side energy-loss aggregation.
    pub energy_loss_epsilon: f32,
    /// Epsilon used for wrong-way streak counting.
    pub wrong_way_progress_epsilon: f32,
}
