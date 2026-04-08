// Covers host bootstrap helpers and runtime step-summary types.
use super::{resolve_display_aspect_ratio, step::StepSummary};

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
    assert_eq!(summary.entered_state_flags, 0);
    assert_eq!(summary.final_frame_index, 0);
}
