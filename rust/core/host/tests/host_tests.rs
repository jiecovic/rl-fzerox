// Covers host bootstrap helpers and runtime step-summary/status types.
use super::{
    resolve_display_aspect_ratio,
    step::{RepeatedStepConfig, StepCounters, StepStatus, StepSummary},
};
use crate::core::input::ControllerState;
use crate::core::observation::ObservationPreset;
use crate::core::telemetry::{PlayerTelemetry, TelemetrySnapshot};

#[test]
fn resolve_display_aspect_ratio_prefers_reported_ratio() {
    let ratio = resolve_display_aspect_ratio(640, 240, 4.0 / 3.0);
    assert!((ratio - (4.0 / 3.0)).abs() < f64::EPSILON);
}

#[test]
fn resolve_display_aspect_ratio_falls_back_to_geometry() {
    let ratio = resolve_display_aspect_ratio(640, 480, 0.0);
    assert!((ratio - (4.0 / 3.0)).abs() < f64::EPSILON);
}

#[test]
fn step_summary_defaults_to_empty_step_accumulators() {
    let summary = StepSummary::default();

    assert_eq!(summary.frames_run, 0);
    assert_eq!(summary.consecutive_reverse_frames, 0);
    assert_eq!(summary.consecutive_low_speed_frames, 0);
    assert_eq!(summary.energy_gain_total, 0.0);
    assert_eq!(summary.entered_state_flags, 0);
    assert_eq!(summary.final_frame_index, 0);
}

#[test]
fn step_status_derives_carried_counters_and_timeout_once_per_outer_step() {
    let status = StepStatus::from_step(
        StepCounters {
            step_count: 9,
            stalled_steps: 2,
            reverse_steps: 2,
        },
        &StepSummary {
            frames_run: 2,
            consecutive_low_speed_frames: 2,
            consecutive_reverse_frames: 2,
            ..StepSummary::default()
        },
        &telemetry(true, 1 << 25),
        repeated_step_config(10, 4, 4),
    );

    assert_eq!(status.counters.step_count, 11);
    assert_eq!(status.counters.stalled_steps, 4);
    assert_eq!(status.counters.reverse_steps, 4);
    assert_eq!(status.termination_reason, Some("finished"));
    assert_eq!(status.truncation_reason, Some("timeout"));
}

#[test]
fn step_status_resets_non_race_trailing_counters() {
    let status = StepStatus::from_step(
        StepCounters {
            step_count: 3,
            stalled_steps: 5,
            reverse_steps: 6,
        },
        &StepSummary {
            frames_run: 1,
            consecutive_low_speed_frames: 1,
            consecutive_reverse_frames: 1,
            ..StepSummary::default()
        },
        &telemetry(false, 0),
        repeated_step_config(100, 5, 5),
    );

    assert_eq!(status.counters.step_count, 4);
    assert_eq!(status.counters.stalled_steps, 0);
    assert_eq!(status.counters.reverse_steps, 0);
    assert!(status.termination_reason.is_none());
    assert!(status.truncation_reason.is_none());
}

fn repeated_step_config(
    max_episode_steps: usize,
    stuck_step_limit: usize,
    wrong_way_step_limit: usize,
) -> RepeatedStepConfig {
    RepeatedStepConfig {
        controller_state: ControllerState::default(),
        action_repeat: 1,
        preset: ObservationPreset::NativeCropV1,
        frame_stack: 4,
        stuck_min_speed_kph: 50.0,
        reverse_progress_epsilon: 0.5,
        energy_loss_epsilon: 0.1,
        wrong_way_progress_epsilon: 0.5,
        max_episode_steps,
        stuck_step_limit,
        wrong_way_step_limit,
    }
}

fn telemetry(in_race_mode: bool, state_flags: u32) -> TelemetrySnapshot {
    TelemetrySnapshot {
        total_lap_count: 3,
        game_mode_raw: 1,
        game_mode_name: if in_race_mode { "gp_race" } else { "title" },
        in_race_mode,
        total_racers: 30,
        course_index: 0,
        player: PlayerTelemetry {
            state_flags,
            speed_kph: 0.0,
            energy: 100.0,
            max_energy: 178.0,
            boost_timer: 0,
            race_distance: 100.0,
            lap_distance: 100.0,
            race_time_ms: 0,
            lap: 1,
            laps_completed: 0,
            position: 1,
        },
    }
}
