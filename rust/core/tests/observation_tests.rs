// rust/core/tests/observation_tests.rs
use super::{ObservationCropProfile, ObservationPreset, ObservationStackMode};
use crate::core::observation::ObservationLayout;
use crate::core::video::VideoCrop;

#[test]
fn crop_profile_keeps_existing_angrylion_crop() {
    assert_eq!(
        ObservationPreset::Crop84x84.crop(ObservationCropProfile::Angrylion),
        VideoCrop {
            top: 16,
            bottom: 16,
            left: 24,
            right: 24,
        }
    );
}

#[test]
fn crop_profile_uses_measured_gliden64_borders() {
    assert_eq!(
        ObservationPreset::Crop84x84.crop(ObservationCropProfile::Gliden64),
        VideoCrop {
            top: 15,
            bottom: 17,
            left: 12,
            right: 12,
        }
    );
}

#[test]
fn gliden64_crop_resolves_to_half_size_watch_display() {
    let spec = ObservationPreset::Crop84x84
        .resolve(320, 240, 4.0 / 3.0, ObservationCropProfile::Gliden64)
        .expect("gliden64 crop should resolve");

    assert_eq!((spec.display_width, spec.display_height), (296, 222));
    assert_eq!((spec.frame_width, spec.frame_height), (84, 84));
}

#[test]
fn crop_72x96_resolves_to_impala_geometry() {
    let spec = ObservationPreset::Crop72x96
        .resolve(640, 240, 4.0 / 3.0, ObservationCropProfile::Angrylion)
        .expect("crop_72x96 should resolve");

    assert_eq!(spec.layout_name, "crop_72x96");
    assert_eq!((spec.frame_width, spec.frame_height), (96, 72));
}

#[test]
fn crop_84x84_resolves_to_square_nature_geometry() {
    let spec = ObservationPreset::Crop84x84
        .resolve(640, 240, 4.0 / 3.0, ObservationCropProfile::Angrylion)
        .expect("crop_84x84 should resolve");

    assert_eq!(spec.layout_name, "crop_84x84");
    assert_eq!((spec.frame_width, spec.frame_height), (84, 84));
}

#[test]
fn custom_layout_uses_requested_target_geometry() {
    let spec = ObservationLayout::custom(72, 96)
        .resolve(640, 240, 4.0 / 3.0, ObservationCropProfile::Angrylion)
        .expect("custom layout should resolve");

    assert_eq!(spec.layout_name, "custom_72x96");
    assert_eq!((spec.frame_width, spec.frame_height), (96, 72));
    assert_eq!((spec.display_width, spec.display_height), (592, 444));
}

#[test]
fn legacy_native_crop_name_is_rejected() {
    assert!(ObservationPreset::parse("native_crop_v4").is_err());
}

#[test]
fn supported_stack_modes_report_expected_channel_counts() {
    assert_eq!(ObservationStackMode::Rgb.stacked_channels(3, 4), 12);
    assert_eq!(ObservationStackMode::Gray.stacked_channels(3, 4), 4);
    assert_eq!(ObservationStackMode::LumaChroma.stacked_channels(3, 4), 8);
}
