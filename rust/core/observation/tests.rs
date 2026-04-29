// rust/core/observation/tests.rs

use super::{ObservationCropProfile, ObservationPreset, ObservationStackMode};
use crate::core::video::VideoCrop;

#[test]
fn crop_profile_keeps_existing_angrylion_crop() {
    assert_eq!(
        ObservationPreset::Crop84x116.crop(ObservationCropProfile::Angrylion),
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
        ObservationPreset::Crop84x116.crop(ObservationCropProfile::Gliden64),
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
    let spec = ObservationPreset::Crop84x116
        .resolve(320, 240, 4.0 / 3.0, ObservationCropProfile::Gliden64)
        .expect("gliden64 crop should resolve");

    assert_eq!((spec.display_width, spec.display_height), (296, 222));
    assert_eq!((spec.frame_width, spec.frame_height), (116, 84));
}

#[test]
fn crop_98x130_resolves_to_compact_deep_geometry() {
    let spec = ObservationPreset::Crop98x130
        .resolve(640, 240, 4.0 / 3.0, ObservationCropProfile::Angrylion)
        .expect("crop_98x130 should resolve");

    assert_eq!(spec.preset_name, "crop_98x130");
    assert_eq!((spec.frame_width, spec.frame_height), (130, 98));
}

#[test]
fn crop_66x82_resolves_to_small_racing_geometry() {
    let spec = ObservationPreset::Crop66x82
        .resolve(640, 240, 4.0 / 3.0, ObservationCropProfile::Angrylion)
        .expect("crop_66x82 should resolve");

    assert_eq!(spec.preset_name, "crop_66x82");
    assert_eq!((spec.frame_width, spec.frame_height), (82, 66));
}

#[test]
fn crop_60x76_resolves_to_compact_aspect_geometry() {
    let spec = ObservationPreset::Crop60x76
        .resolve(640, 240, 4.0 / 3.0, ObservationCropProfile::Angrylion)
        .expect("crop_60x76 should resolve");

    assert_eq!(spec.preset_name, "crop_60x76");
    assert_eq!((spec.frame_width, spec.frame_height), (76, 60));
}

#[test]
fn crop_68x68_resolves_to_square_nature_geometry() {
    let spec = ObservationPreset::Crop68x68
        .resolve(640, 240, 4.0 / 3.0, ObservationCropProfile::Angrylion)
        .expect("crop_68x68 should resolve");

    assert_eq!(spec.preset_name, "crop_68x68");
    assert_eq!((spec.frame_width, spec.frame_height), (68, 68));
}

#[test]
fn crop_84x84_resolves_to_square_nature_geometry() {
    let spec = ObservationPreset::Crop84x84
        .resolve(640, 240, 4.0 / 3.0, ObservationCropProfile::Angrylion)
        .expect("crop_84x84 should resolve");

    assert_eq!(spec.preset_name, "crop_84x84");
    assert_eq!((spec.frame_width, spec.frame_height), (84, 84));
}

#[test]
fn crop_76x100_resolves_to_nature_geometry() {
    let spec = ObservationPreset::Crop76x100
        .resolve(640, 240, 4.0 / 3.0, ObservationCropProfile::Angrylion)
        .expect("crop_76x100 should resolve");

    assert_eq!(spec.preset_name, "crop_76x100");
    assert_eq!((spec.frame_width, spec.frame_height), (100, 76));
}

#[test]
fn crop_64x64_resolves_to_square_geometry() {
    let spec = ObservationPreset::Crop64x64
        .resolve(640, 240, 4.0 / 3.0, ObservationCropProfile::Angrylion)
        .expect("crop_64x64 should resolve");

    assert_eq!(spec.preset_name, "crop_64x64");
    assert_eq!((spec.frame_width, spec.frame_height), (64, 64));
}

#[test]
fn legacy_native_crop_name_aliases_to_canonical_name() {
    let preset = ObservationPreset::parse("native_crop_v4").expect("legacy name resolves");

    assert_eq!(preset, ObservationPreset::Crop98x130);
    assert_eq!(preset.name(), "crop_98x130");
}

#[test]
fn supported_stack_modes_report_expected_channel_counts() {
    assert_eq!(ObservationStackMode::Rgb.stacked_channels(3, 4), 12);
    assert_eq!(ObservationStackMode::Gray.stacked_channels(3, 4), 4);
    assert_eq!(ObservationStackMode::LumaChroma.stacked_channels(3, 4), 8);
}
