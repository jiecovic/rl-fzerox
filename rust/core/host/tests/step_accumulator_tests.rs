// Covers step-local aggregation for one repeated env step.
use super::step_accumulator::StepAccumulator;
use crate::core::host::RepeatedStepConfig;
use crate::core::input::ControllerState;
use crate::core::observation::{ObservationPreset, ObservationStackMode};
use crate::core::telemetry::StepTelemetrySample;

#[test]
fn step_accumulator_tracks_progress_energy_loss_and_entered_flags() {
    let initial = telemetry(100.0, 100.0, 120.0, 0b001, 0, 0);
    let mut accumulator = StepAccumulator::new(&initial, repeated_step_config(100, 5), 40);

    accumulator.observe(&telemetry(130.0, 92.0, 120.0, 0b101, 0, 1), 41);
    accumulator.observe(&telemetry(140.0, 88.0, 45.0, 0b111, 12, 0), 42);

    let summary = accumulator.finish();

    assert_eq!(summary.frames_run, 2);
    assert_eq!(summary.max_race_distance, 140.0);
    assert_eq!(summary.reverse_active_frames, 1);
    assert_eq!(summary.low_speed_frames, 1);
    assert_eq!(summary.energy_loss_total, 12.0);
    assert_eq!(summary.energy_gain_total, 0.0);
    assert_eq!(summary.damage_taken_frames, 1);
    assert_eq!(summary.consecutive_low_speed_frames, 1);
    assert_eq!(summary.entered_state_flags, 0b110);
    assert_eq!(summary.final_frame_index, 42);
}

#[test]
fn step_accumulator_counts_reverse_active_frames_when_reverse_timer_runs() {
    let initial = telemetry(100.0, 100.0, 120.0, 0b001, 0, 0);
    let mut accumulator = StepAccumulator::new(&initial, repeated_step_config(100, 5), 5);

    accumulator.observe(&telemetry(90.0, 100.0, 30.0, 0b001, 99, 0), 6);
    accumulator.observe(&telemetry(95.0, 100.0, 80.0, 0b001, 0, 0), 7);
    accumulator.observe(&telemetry(93.0, 100.0, 40.0, 0b001, 100, 0), 8);

    let summary = accumulator.finish();

    assert_eq!(summary.frames_run, 3);
    assert_eq!(summary.max_race_distance, 100.0);
    assert_eq!(summary.reverse_active_frames, 2);
    assert_eq!(summary.low_speed_frames, 2);
    assert_eq!(summary.consecutive_low_speed_frames, 1);
    assert_eq!(summary.final_frame_index, 8);
}

#[test]
fn step_accumulator_tracks_low_speed_streak_and_energy_changes() {
    let initial = telemetry(100.0, 100.0, 30.0, 0b001, 5, 0);
    let mut accumulator = StepAccumulator::new(&initial, repeated_step_config(100, 5), 11);

    accumulator.observe(&telemetry(99.0, 110.0, 30.0, 0b001, 6, 0), 12);
    accumulator.observe(&telemetry(98.0, 108.0, 20.0, 0b001, 7, 0), 13);

    let summary = accumulator.finish();

    assert_eq!(summary.frames_run, 2);
    assert_eq!(summary.max_race_distance, 100.0);
    assert_eq!(summary.energy_loss_total, 2.0);
    assert_eq!(summary.energy_gain_total, 10.0);
    assert_eq!(summary.reverse_active_frames, 2);
    assert_eq!(summary.low_speed_frames, 2);
    assert_eq!(summary.consecutive_low_speed_frames, 2);
    assert_eq!(summary.final_frame_index, 13);
}

#[test]
fn step_accumulator_tracks_energy_gain_separately_from_loss() {
    let initial = telemetry(100.0, 100.0, 120.0, 0b001, 0, 0);
    let mut accumulator = StepAccumulator::new(&initial, repeated_step_config(100, 5), 20);

    accumulator.observe(&telemetry(102.0, 96.0, 120.0, 0b001, 0, 0), 21);
    accumulator.observe(&telemetry(104.0, 101.5, 120.0, 0b001, 0, 0), 22);

    let summary = accumulator.finish();

    assert_eq!(summary.energy_loss_total, 4.0);
    assert_eq!(summary.energy_gain_total, 5.5);
}

#[test]
fn step_accumulator_counts_received_damage_state_as_damage_taken() {
    let initial = telemetry(100.0, 100.0, 120.0, 0b001, 0, 0);
    let mut accumulator = StepAccumulator::new(&initial, repeated_step_config(100, 5), 20);

    accumulator.observe(&telemetry(102.0, 100.0, 120.0, 1 << 17, 0, 0), 21);

    let summary = accumulator.finish();

    assert_eq!(summary.damage_taken_frames, 1);
}

#[test]
fn step_accumulator_tracks_summary_needed_for_stop_state_derivation() {
    let initial = telemetry(100.0, 100.0, 120.0, 0b001, 99, 0);
    let mut accumulator = StepAccumulator::new(&initial, repeated_step_config(10, 2), 99);

    accumulator.observe(
        &telemetry(99.0, 100.0, 30.0, (1 << 25) | 0b001, 100, 0),
        100,
    );

    let summary = accumulator.finish();

    assert_eq!(summary.frames_run, 1);
    assert_eq!(summary.low_speed_frames, 1);
    assert_eq!(summary.consecutive_low_speed_frames, 1);
    assert_eq!(summary.reverse_active_frames, 1);
    assert_eq!(summary.entered_state_flags, 1 << 25);
}

fn repeated_step_config(max_episode_steps: usize, stuck_step_limit: usize) -> RepeatedStepConfig {
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
        wrong_way_timer_limit: Some(180),
        progress_frontier_stall_limit_frames: Some(900),
        progress_frontier_epsilon: 25.0,
        terminate_on_energy_depleted: true,
        lean_timer_assist: false,
    }
}

fn telemetry(
    race_distance: f32,
    energy: f32,
    speed_kph: f32,
    state_flags: u32,
    reverse_timer: i32,
    damage_rumble_counter: i32,
) -> StepTelemetrySample {
    StepTelemetrySample {
        state_flags,
        speed_kph,
        energy,
        race_distance,
        reverse_timer,
        damage_rumble_counter,
    }
}
