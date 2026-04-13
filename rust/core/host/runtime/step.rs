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
    /// Number of internal frames spent with the game reverse timer active.
    pub reverse_active_frames: usize,
    /// Number of internal frames spent below the configured stuck-speed threshold.
    pub low_speed_frames: usize,
    /// Total energy lost during the repeated step.
    pub energy_loss_total: f32,
    /// Total energy regained during the repeated step.
    pub energy_gain_total: f32,
    /// Number of internal frames where the game damage pulse was active.
    pub damage_taken_frames: usize,
    /// Low-speed streak length at the end of the repeated step.
    pub consecutive_low_speed_frames: usize,
    /// OR of state flags newly entered at least once during this env step.
    pub entered_state_flags: u32,
    /// Frame index after the repeated step completed.
    pub final_frame_index: usize,
}

/// Carried limit counters that persist across outer env steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepCounters {
    /// Total internal frames executed in the current episode.
    pub step_count: usize,
    /// Consecutive low-speed internal frames at the current episode frontier.
    pub stalled_steps: usize,
    /// Consecutive internal frames since the last meaningful new progress frontier.
    pub progress_frontier_stalled_frames: usize,
    /// Highest race distance reached so far in the current in-race episode.
    pub progress_frontier_distance: f32,
    /// Whether the progress frontier has been initialized from in-race telemetry yet.
    pub progress_frontier_initialized: bool,
}

/// Native stop/counter state after one repeated env step completes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepStatus {
    /// Updated carried counters after the repeated env step.
    pub counters: StepCounters,
    /// Current game reverse-warning timer after the repeated env step.
    pub reverse_timer: usize,
    /// Terminal reason detected on the final executed internal frame, if any.
    pub termination_reason: Option<&'static str>,
    /// Truncation reason detected while advancing the repeated env step, if any.
    pub truncation_reason: Option<&'static str>,
}

impl StepStatus {
    pub fn from_step(
        previous: StepCounters,
        summary: &StepSummary,
        final_telemetry: &TelemetrySnapshot,
        config: RepeatedStepConfig,
    ) -> Self {
        let counters = StepCounters {
            step_count: previous.step_count + summary.frames_run,
            stalled_steps: carried_streak(
                previous.stalled_steps,
                summary.consecutive_low_speed_frames,
                summary.frames_run,
                final_telemetry.in_race_mode,
            ),
            ..carried_progress_frontier(previous, summary, final_telemetry, config)
        };
        let reverse_timer = final_telemetry.player.reverse_timer.max(0) as usize;
        let truncation_reason = if counters.step_count >= config.max_episode_steps {
            Some("timeout")
        } else if reverse_timer >= config.wrong_way_timer_limit {
            Some("wrong_way")
        } else if counters.stalled_steps >= config.stuck_step_limit {
            Some("stuck")
        } else if config
            .progress_frontier_stall_limit_frames
            .is_some_and(|limit| counters.progress_frontier_stalled_frames >= limit)
        {
            Some("progress_stalled")
        } else {
            None
        };
        let termination_reason = final_telemetry
            .player
            .terminal_reason()
            .or_else(|| energy_depleted_reason(final_telemetry, config));
        Self {
            counters,
            reverse_timer,
            termination_reason,
            truncation_reason,
        }
    }

    pub fn terminated(&self) -> bool {
        self.termination_reason.is_some()
    }

    pub fn truncated(&self) -> bool {
        self.truncation_reason.is_some()
    }
}

fn energy_depleted_reason(
    telemetry: &TelemetrySnapshot,
    config: RepeatedStepConfig,
) -> Option<&'static str> {
    if !config.terminate_on_energy_depleted
        || !telemetry.in_race_mode
        || !telemetry.player.active()
        || telemetry.player.max_energy <= 0.0
        || telemetry.player.energy > 0.0
    {
        return None;
    }

    Some("energy_depleted")
}

fn carried_streak(
    previous: usize,
    trailing_in_step: usize,
    frames_run: usize,
    in_race_mode: bool,
) -> usize {
    if !in_race_mode || trailing_in_step == 0 {
        return 0;
    }
    if trailing_in_step == frames_run {
        return previous + trailing_in_step;
    }
    trailing_in_step
}

fn carried_progress_frontier(
    previous: StepCounters,
    summary: &StepSummary,
    final_telemetry: &TelemetrySnapshot,
    config: RepeatedStepConfig,
) -> StepCounters {
    if !final_telemetry.in_race_mode || summary.frames_run == 0 {
        return StepCounters {
            progress_frontier_stalled_frames: 0,
            progress_frontier_distance: 0.0,
            progress_frontier_initialized: false,
            ..StepCounters::default()
        };
    }

    if !previous.progress_frontier_initialized {
        return StepCounters {
            progress_frontier_stalled_frames: 0,
            progress_frontier_distance: summary.max_race_distance,
            progress_frontier_initialized: true,
            ..StepCounters::default()
        };
    }

    if summary.max_race_distance
        >= previous.progress_frontier_distance + config.progress_frontier_epsilon
    {
        return StepCounters {
            progress_frontier_stalled_frames: 0,
            progress_frontier_distance: summary.max_race_distance,
            progress_frontier_initialized: true,
            ..StepCounters::default()
        };
    }

    StepCounters {
        progress_frontier_stalled_frames: previous.progress_frontier_stalled_frames
            + summary.frames_run,
        progress_frontier_distance: previous.progress_frontier_distance,
        progress_frontier_initialized: true,
        ..StepCounters::default()
    }
}

/// Native env-step payload returned after executing a repeated step.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct NativeStepResult<'a> {
    /// Final stacked observation tensor for this outer env step.
    pub observation: &'a [u8],
    /// Aggregated step features spanning the internal repeated frames.
    pub summary: StepSummary,
    /// Native counter/stop state after the repeated env step completed.
    pub status: StepStatus,
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
    /// Epsilon used for reward-side energy-loss aggregation.
    pub energy_loss_epsilon: f32,
    /// Maximum number of internal frames allowed in one episode.
    pub max_episode_steps: usize,
    /// Low-speed frame limit that triggers a stuck truncation.
    pub stuck_step_limit: usize,
    /// Reverse timer limit that triggers wrong-way truncation.
    pub wrong_way_timer_limit: usize,
    /// Maximum internal frames allowed without beating the best race-distance frontier.
    pub progress_frontier_stall_limit_frames: Option<usize>,
    /// Minimum frontier improvement required to reset the progress-stall timer.
    pub progress_frontier_epsilon: f32,
    /// Treat depleted player energy as an immediate terminal failure.
    pub terminate_on_energy_depleted: bool,
    /// Patch shoulder timers so short drift taps cannot become side attacks.
    pub shoulder_slide_timer_assist: bool,
}
