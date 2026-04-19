// Covers host bootstrap helpers and runtime step-summary/status types.
use super::{
    resolve_display_aspect_ratio,
    step::{RepeatedStepConfig, StepCounters, StepStatus, StepSummary},
};
use crate::core::input::ControllerState;
use crate::core::observation::{ObservationPreset, ObservationStackMode};
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
    assert_eq!(summary.damage_taken_frames, 0);
    assert_eq!(summary.entered_state_flags, 0);
    assert_eq!(summary.final_frame_index, 0);
}

#[test]
fn step_status_derives_carried_counters_and_timeout_once_per_outer_step() {
    let status = StepStatus::from_step(
        StepCounters {
            step_count: 9,
            stalled_steps: 2,
            progress_frontier_stalled_frames: 0,
            progress_frontier_distance: 100.0,
            progress_frontier_initialized: true,
        },
        &StepSummary {
            frames_run: 2,
            max_race_distance: 130.0,
            consecutive_low_speed_frames: 2,
            ..StepSummary::default()
        },
        &telemetry(true, 1 << 25, 100),
        repeated_step_config(10, 4, 180),
    );

    assert_eq!(status.counters.step_count, 11);
    assert_eq!(status.counters.stalled_steps, 4);
    assert_eq!(status.counters.progress_frontier_stalled_frames, 0);
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
            progress_frontier_stalled_frames: 7,
            progress_frontier_distance: 120.0,
            progress_frontier_initialized: true,
        },
        &StepSummary {
            frames_run: 1,
            max_race_distance: 110.0,
            consecutive_low_speed_frames: 1,
            ..StepSummary::default()
        },
        &telemetry(false, 0, 0),
        repeated_step_config(100, 5, 180),
    );

    assert_eq!(status.counters.step_count, 4);
    assert_eq!(status.counters.stalled_steps, 0);
    assert_eq!(status.counters.progress_frontier_stalled_frames, 0);
    assert!(!status.counters.progress_frontier_initialized);
    assert_eq!(status.reverse_timer, 0);
    assert!(status.termination_reason.is_none());
    assert!(status.truncation_reason.is_none());
}

#[test]
fn step_status_accumulates_progress_frontier_stall_frames_until_new_best_distance() {
    let status = StepStatus::from_step(
        StepCounters {
            step_count: 10,
            stalled_steps: 0,
            progress_frontier_stalled_frames: 6,
            progress_frontier_distance: 150.0,
            progress_frontier_initialized: true,
        },
        &StepSummary {
            frames_run: 3,
            max_race_distance: 170.0,
            ..StepSummary::default()
        },
        &telemetry(true, 0, 0),
        repeated_step_config(100, 5, 180),
    );

    assert_eq!(status.counters.progress_frontier_stalled_frames, 9);
    assert_eq!(status.counters.progress_frontier_distance, 150.0);
    assert_eq!(status.truncation_reason, None);
}

#[test]
fn step_status_truncates_when_progress_frontier_stall_limit_is_reached() {
    let mut config = repeated_step_config(100, 5, 180);
    config.progress_frontier_stall_limit_frames = Some(8);
    config.progress_frontier_epsilon = 25.0;

    let status = StepStatus::from_step(
        StepCounters {
            step_count: 10,
            stalled_steps: 0,
            progress_frontier_stalled_frames: 6,
            progress_frontier_distance: 150.0,
            progress_frontier_initialized: true,
        },
        &StepSummary {
            frames_run: 2,
            max_race_distance: 170.0,
            ..StepSummary::default()
        },
        &telemetry(true, 0, 0),
        config,
    );

    assert_eq!(status.counters.progress_frontier_stalled_frames, 8);
    assert_eq!(status.truncation_reason, Some("progress_stalled"));
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
fn step_status_allows_disabling_wrong_way_truncation() {
    let mut config = repeated_step_config(100, 5, 180);
    config.wrong_way_timer_limit = None;

    let status = StepStatus::from_step(
        StepCounters::default(),
        &StepSummary {
            frames_run: 1,
            ..StepSummary::default()
        },
        &telemetry(true, 0, 10_000),
        config,
    );

    assert_eq!(status.reverse_timer, 10_000);
    assert_eq!(status.truncation_reason, None);
}

#[test]
fn step_status_reports_spinning_out_before_energy_depletion_fallback() {
    let status = StepStatus::from_step(
        StepCounters::default(),
        &StepSummary {
            frames_run: 1,
            ..StepSummary::default()
        },
        &telemetry_with_energy(true, (1 << 14) | (1 << 30), 0, 0.0, 178.0),
        repeated_step_config(100, 5, 180),
    );

    assert_eq!(status.termination_reason, Some("spinning_out"));
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
        preset: ObservationPreset::Crop84x116,
        frame_stack: 4,
        stack_mode: ObservationStackMode::Rgb,
        stuck_min_speed_kph: 50.0,
        energy_loss_epsilon: 0.1,
        max_episode_steps,
        stuck_step_limit,
        wrong_way_timer_limit: Some(wrong_way_timer_limit),
        progress_frontier_stall_limit_frames: Some(900),
        progress_frontier_epsilon: 25.0,
        terminate_on_energy_depleted: true,
        lean_timer_assist: false,
    }
}

fn telemetry(in_race_mode: bool, state_flags: u32, reverse_timer: i32) -> TelemetrySnapshot {
    TelemetrySnapshot {
        total_lap_count: 3,
        difficulty_raw: 0,
        difficulty_name: "novice",
        camera_setting_raw: 2,
        camera_setting_name: "regular",
        race_intro_timer: 0,
        game_mode_raw: 1,
        game_mode_name: if in_race_mode { "gp_race" } else { "title" },
        in_race_mode,
        total_racers: 30,
        course_index: 0,
        course_length: 80_000.0,
        player: PlayerTelemetry {
            state_flags,
            speed_kph: 0.0,
            energy: 100.0,
            max_energy: 178.0,
            boost_timer: 0,
            recoil_tilt_magnitude: 0.0,
            damage_rumble_counter: 0,
            reverse_timer,
            race_distance: 100.0,
            lap_distance: 100.0,
            race_time_ms: 0,
            lap: 1,
            laps_completed: 0,
            position: 1,
            geometry: Default::default(),
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
