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
    assert_eq!(summary.reverse_active_frames, 0);
    assert_eq!(summary.low_speed_frames, 0);
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
        },
        &StepSummary {
            frames_run: 2,
            consecutive_low_speed_frames: 2,
            ..StepSummary::default()
        },
        &telemetry(true, 1 << 25, 100),
        repeated_step_config(10, 4, 180),
    );

    assert_eq!(status.counters.step_count, 11);
    assert_eq!(status.counters.stalled_steps, 4);
    assert_eq!(status.reverse_timer, 100);
    assert_eq!(status.termination_reason, Some("finished"));
    assert_eq!(status.truncation_reason, Some("timeout"));
}

#[test]
fn step_status_resets_non_race_trailing_counters() {
    let status = StepStatus::from_step(
        StepCounters {
            step_count: 3,
            stalled_steps: 5,
        },
        &StepSummary {
            frames_run: 1,
            consecutive_low_speed_frames: 1,
            ..StepSummary::default()
        },
        &telemetry(false, 0, 0),
        repeated_step_config(100, 5, 180),
    );

    assert_eq!(status.counters.step_count, 4);
    assert_eq!(status.counters.stalled_steps, 0);
    assert_eq!(status.reverse_timer, 0);
    assert!(status.termination_reason.is_none());
    assert!(status.truncation_reason.is_none());
}

#[test]
fn step_status_uses_configured_reverse_timer_limit_for_wrong_way() {
    let status = StepStatus::from_step(
        StepCounters::default(),
        &StepSummary {
            frames_run: 1,
            ..StepSummary::default()
        },
        &telemetry(true, 0, 180),
        repeated_step_config(100, 5, 180),
    );

    assert_eq!(status.reverse_timer, 180);
    assert_eq!(status.truncation_reason, Some("wrong_way"));
}

#[test]
fn step_status_does_not_truncate_below_configured_reverse_timer_limit() {
    let status = StepStatus::from_step(
        StepCounters::default(),
        &StepSummary {
            frames_run: 1,
            ..StepSummary::default()
        },
        &telemetry(true, 0, 100),
        repeated_step_config(100, 5, 180),
    );

    assert_eq!(status.reverse_timer, 100);
    assert_eq!(status.truncation_reason, None);
}

#[test]
fn step_status_terminates_on_active_race_energy_depletion() {
    let status = StepStatus::from_step(
        StepCounters::default(),
        &StepSummary {
            frames_run: 1,
            ..StepSummary::default()
        },
        &telemetry_with_energy(true, 1 << 30, 0, 0.0, 178.0),
        repeated_step_config(100, 5, 180),
    );

    assert_eq!(status.termination_reason, Some("energy_depleted"));
}

#[test]
fn step_status_allows_disabling_energy_depletion_termination() {
    let mut config = repeated_step_config(100, 5, 180);
    config.terminate_on_energy_depleted = false;

    let status = StepStatus::from_step(
        StepCounters::default(),
        &StepSummary {
            frames_run: 1,
            ..StepSummary::default()
        },
        &telemetry_with_energy(true, 1 << 30, 0, 0.0, 178.0),
        config,
    );

    assert_eq!(status.termination_reason, None);
}

#[test]
fn step_status_ignores_energy_depletion_outside_active_race_state() {
    let config = repeated_step_config(100, 5, 180);

    let non_race = StepStatus::from_step(
        StepCounters::default(),
        &StepSummary {
            frames_run: 1,
            ..StepSummary::default()
        },
        &telemetry_with_energy(false, 1 << 30, 0, 0.0, 178.0),
        config,
    );
    let inactive = StepStatus::from_step(
        StepCounters::default(),
        &StepSummary {
            frames_run: 1,
            ..StepSummary::default()
        },
        &telemetry_with_energy(true, 0, 0, 0.0, 178.0),
        config,
    );

    assert_eq!(non_race.termination_reason, None);
    assert_eq!(inactive.termination_reason, None);
}

fn repeated_step_config(
    max_episode_steps: usize,
    stuck_step_limit: usize,
    wrong_way_timer_limit: usize,
) -> RepeatedStepConfig {
    RepeatedStepConfig {
        controller_state: ControllerState::default(),
        action_repeat: 1,
        preset: ObservationPreset::NativeCropV1,
        frame_stack: 4,
        stuck_min_speed_kph: 50.0,
        energy_loss_epsilon: 0.1,
        max_episode_steps,
        stuck_step_limit,
        wrong_way_timer_limit,
        terminate_on_energy_depleted: true,
    }
}

fn telemetry(in_race_mode: bool, state_flags: u32, reverse_timer: i32) -> TelemetrySnapshot {
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
            reverse_timer,
            race_distance: 100.0,
            lap_distance: 100.0,
            race_time_ms: 0,
            lap: 1,
            laps_completed: 0,
            position: 1,
        },
    }
}

fn telemetry_with_energy(
    in_race_mode: bool,
    state_flags: u32,
    reverse_timer: i32,
    energy: f32,
    max_energy: f32,
) -> TelemetrySnapshot {
    let mut snapshot = telemetry(in_race_mode, state_flags, reverse_timer);
    snapshot.player.energy = energy;
    snapshot.player.max_energy = max_energy;
    snapshot
}
