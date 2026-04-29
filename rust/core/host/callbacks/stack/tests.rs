// rust/core/host/callbacks/stack/tests.rs

use super::StackedObservationBuffer;
use crate::core::observation::ObservationStackMode;

#[test]
fn minimap_extra_channel_is_appended_after_existing_stack_channels() {
    let mut stack = StackedObservationBuffer::new(6, 2, 3, ObservationStackMode::Gray, 1);

    stack
        .update(&[10, 20, 30, 40, 50, 60], 1, Some(&[90, 120]))
        .expect("initial stack should accept minimap layer");

    assert_eq!(stack.as_slice(), &[18, 18, 90, 48, 48, 120]);

    stack
        .update(&[70, 80, 90, 100, 110, 120], 2, Some(&[130, 160]))
        .expect("next stack should update minimap layer");

    assert_eq!(stack.as_slice(), &[18, 78, 130, 48, 108, 160]);
}

#[test]
fn gray_stack_encodes_all_frames_as_luma() {
    let mut stack = StackedObservationBuffer::new(6, 2, 3, ObservationStackMode::Gray, 1);

    stack
        .update(&[10, 20, 30, 40, 50, 60], 1, Some(&[90, 120]))
        .expect("initial stack should accept minimap layer");

    assert_eq!(stack.as_slice(), &[18, 18, 90, 48, 48, 120]);

    stack
        .update(&[70, 80, 90, 100, 110, 120], 2, Some(&[130, 160]))
        .expect("next stack should update grayscale stack");

    assert_eq!(stack.as_slice(), &[18, 78, 130, 48, 108, 160]);
}

#[test]
fn luma_chroma_stack_preserves_yellow_purple_cue() {
    let mut stack = StackedObservationBuffer::new(6, 2, 3, ObservationStackMode::LumaChroma, 1);

    stack
        .update(&[255, 255, 0, 255, 0, 255], 1, Some(&[90, 120]))
        .expect("initial stack should accept luma-chroma stack");

    assert_eq!(
        stack.as_slice(),
        &[226, 191, 226, 191, 90, 106, 1, 106, 1, 120]
    );
}
