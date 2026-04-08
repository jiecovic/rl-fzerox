// Covers step-local aggregation for one repeated env step.
use super::step_accumulator::StepAccumulator;
use crate::core::telemetry::{PlayerTelemetry, TelemetrySnapshot};

#[test]
fn step_accumulator_tracks_progress_energy_loss_and_entered_flags() {
    let initial = telemetry(100.0, 100.0, 120.0, 0b001);
    let mut accumulator = StepAccumulator::new(&initial, 50.0, 0.5, 0.1, 0.5, 40);

    accumulator.observe(&telemetry(130.0, 92.0, 120.0, 0b101), 41);
    accumulator.observe(&telemetry(140.0, 88.0, 45.0, 0b111), 42);

    let summary = accumulator.finish();

    assert_eq!(summary.frames_run, 2);
    assert_eq!(summary.max_race_distance, 140.0);
    assert_eq!(summary.reverse_progress_total, 0.0);
    assert_eq!(summary.energy_loss_total, 12.0);
    assert_eq!(summary.consecutive_low_speed_frames, 1);
    assert_eq!(summary.consecutive_reverse_frames, 0);
    assert_eq!(summary.entered_state_flags, 0b110);
    assert_eq!(summary.final_frame_index, 42);
}

#[test]
fn step_accumulator_resets_streaks_when_step_ends_recovered() {
    let initial = telemetry(100.0, 100.0, 120.0, 0b001);
    let mut accumulator = StepAccumulator::new(&initial, 50.0, 0.5, 0.1, 0.5, 5);

    accumulator.observe(&telemetry(90.0, 100.0, 30.0, 0b001), 6);
    accumulator.observe(&telemetry(95.0, 100.0, 80.0, 0b001), 7);
    accumulator.observe(&telemetry(93.0, 100.0, 40.0, 0b001), 8);

    let summary = accumulator.finish();

    assert_eq!(summary.frames_run, 3);
    assert_eq!(summary.max_race_distance, 100.0);
    assert_eq!(summary.reverse_progress_total, 12.0);
    assert_eq!(summary.consecutive_reverse_frames, 1);
    assert_eq!(summary.consecutive_low_speed_frames, 1);
    assert_eq!(summary.final_frame_index, 8);
}

#[test]
fn step_accumulator_keeps_trailing_streaks_when_every_frame_matches() {
    let initial = telemetry(100.0, 100.0, 30.0, 0b001);
    let mut accumulator = StepAccumulator::new(&initial, 50.0, 0.5, 0.1, 0.5, 11);

    accumulator.observe(&telemetry(99.0, 110.0, 30.0, 0b001), 12);
    accumulator.observe(&telemetry(98.0, 108.0, 20.0, 0b001), 13);

    let summary = accumulator.finish();

    assert_eq!(summary.frames_run, 2);
    assert_eq!(summary.max_race_distance, 100.0);
    assert_eq!(summary.reverse_progress_total, 2.0);
    assert_eq!(summary.energy_loss_total, 2.0);
    assert_eq!(summary.consecutive_reverse_frames, 2);
    assert_eq!(summary.consecutive_low_speed_frames, 2);
    assert_eq!(summary.final_frame_index, 13);
}

fn telemetry(
    race_distance: f32,
    energy: f32,
    speed_kph: f32,
    state_flags: u32,
) -> TelemetrySnapshot {
    TelemetrySnapshot {
        system_ram_size: 0x00800000,
        game_frame_count: 100,
        game_mode_raw: 1,
        game_mode_name: String::from("gp_race"),
        course_index: 0,
        in_race_mode: true,
        player: PlayerTelemetry {
            state_flags,
            state_labels: Vec::new(),
            speed_raw: 0.0,
            speed_kph,
            energy,
            max_energy: 178.0,
            boost_timer: 0,
            race_distance,
            laps_completed_distance: 0.0,
            lap_distance: race_distance,
            race_distance_position: race_distance,
            race_time_ms: 0,
            lap: 1,
            laps_completed: 0,
            position: 30,
            character: 0,
            machine_index: 0,
        },
    }
}
