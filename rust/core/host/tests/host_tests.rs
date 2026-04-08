// Covers host bootstrap helpers that resolve the effective display aspect ratio.
use super::resolve_display_aspect_ratio;

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
