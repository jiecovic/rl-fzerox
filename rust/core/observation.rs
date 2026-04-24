// rust/core/observation.rs
//! Observation presets owned by the native layer.
//!
//! Python selects a preset name plus frame-stack depth; Rust owns the spatial
//! geometry, crop policy, and display dimensions for that preset.

use crate::core::error::CoreError;
use crate::core::video::{VideoCrop, cropped_dimensions, display_size};

/// Renderer-specific crop profile for raw framebuffers.
///
/// Angrylion exposes F-Zero X as a 640x240 software frame, while GLideN64's
/// smallest useful hardware viewport is 320x240. The visible game borders are
/// stable but not the same number of pixels in each renderer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationCropProfile {
    Angrylion,
    Gliden64,
}

impl ObservationCropProfile {
    pub fn from_renderer_name(renderer: &str) -> Self {
        match renderer {
            "gliden64" => Self::Gliden64,
            _ => Self::Angrylion,
        }
    }
}

/// Named single-frame observation layouts exposed to Python.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationPreset {
    Crop84x116,
    Crop92x124,
    Crop116x164,
    Crop98x130,
    Crop66x82,
    Crop60x76,
    Crop68x68,
    Crop84x84,
    Crop76x100,
    Crop64x64,
}

/// How repeated observation frames are encoded along the channel axis.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ObservationStackMode {
    /// Keep every stacked frame as RGB: `3 * frame_stack` channels.
    Rgb,
    /// Encode every stacked frame as grayscale: `frame_stack` channels.
    Gray,
    /// Encode every frame as luminance plus a yellow-vs-purple chroma cue.
    LumaChroma,
}

/// Resolved spatial spec for one observation frame plus the matching display
/// size used by the watch UI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservationSpec {
    pub preset_name: &'static str,
    pub frame_width: usize,
    pub frame_height: usize,
    pub channels: usize,
    pub display_width: usize,
    pub display_height: usize,
}

impl ObservationPreset {
    /// Parse the stable string name recorded in YAML configs and run metadata.
    pub fn parse(name: &str) -> Result<Self, CoreError> {
        match name {
            "crop_84x116" => Ok(Self::Crop84x116),
            "crop_92x124" => Ok(Self::Crop92x124),
            "crop_116x164" => Ok(Self::Crop116x164),
            "crop_98x130" => Ok(Self::Crop98x130),
            "crop_66x82" => Ok(Self::Crop66x82),
            "crop_60x76" => Ok(Self::Crop60x76),
            "crop_68x68" => Ok(Self::Crop68x68),
            "crop_84x84" => Ok(Self::Crop84x84),
            "crop_76x100" => Ok(Self::Crop76x100),
            "crop_64x64" => Ok(Self::Crop64x64),
            // V4 LEGACY SHIM: accept old saved run manifests and CLI overrides.
            "native_crop_v1" => Ok(Self::Crop84x116),
            "native_crop_v2" => Ok(Self::Crop92x124),
            "native_crop_v3" => Ok(Self::Crop116x164),
            "native_crop_v4" => Ok(Self::Crop98x130),
            "native_crop_v6" => Ok(Self::Crop66x82),
            _ => Err(CoreError::InvalidObservationPreset {
                name: name.to_owned(),
            }),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Crop84x116 => "crop_84x116",
            Self::Crop92x124 => "crop_92x124",
            Self::Crop116x164 => "crop_116x164",
            Self::Crop98x130 => "crop_98x130",
            Self::Crop66x82 => "crop_66x82",
            Self::Crop60x76 => "crop_60x76",
            Self::Crop68x68 => "crop_68x68",
            Self::Crop84x84 => "crop_84x84",
            Self::Crop76x100 => "crop_76x100",
            Self::Crop64x64 => "crop_64x64",
        }
    }

    pub fn crop(self, crop_profile: ObservationCropProfile) -> VideoCrop {
        match (self, crop_profile) {
            (
                Self::Crop84x116
                | Self::Crop92x124
                | Self::Crop116x164
                | Self::Crop98x130
                | Self::Crop66x82
                | Self::Crop60x76
                | Self::Crop68x68
                | Self::Crop84x84
                | Self::Crop76x100
                | Self::Crop64x64,
                ObservationCropProfile::Angrylion,
            ) => VideoCrop {
                top: 16,
                bottom: 16,
                left: 24,
                right: 24,
            },
            (
                Self::Crop84x116
                | Self::Crop92x124
                | Self::Crop116x164
                | Self::Crop98x130
                | Self::Crop66x82
                | Self::Crop60x76
                | Self::Crop68x68
                | Self::Crop84x84
                | Self::Crop76x100
                | Self::Crop64x64,
                ObservationCropProfile::Gliden64,
            ) => VideoCrop {
                top: 15,
                bottom: 17,
                left: 12,
                right: 12,
            },
        }
    }

    pub fn resolve(
        self,
        raw_frame_width: usize,
        raw_frame_height: usize,
        display_aspect_ratio: f64,
        crop_profile: ObservationCropProfile,
    ) -> Result<ObservationSpec, CoreError> {
        let crop = self.crop(crop_profile);
        let (cropped_width, cropped_height) =
            cropped_dimensions(raw_frame_width, raw_frame_height, crop)?;
        let (display_width, display_height) =
            display_size(cropped_width, cropped_height, display_aspect_ratio);
        let (frame_width, frame_height, channels) = match self {
            Self::Crop84x116 => (116, 84, 3),
            Self::Crop92x124 => (124, 92, 3),
            Self::Crop116x164 => (164, 116, 3),
            Self::Crop98x130 => (130, 98, 3),
            Self::Crop66x82 => (82, 66, 3),
            Self::Crop60x76 => (76, 60, 3),
            Self::Crop68x68 => (68, 68, 3),
            Self::Crop84x84 => (84, 84, 3),
            Self::Crop76x100 => (100, 76, 3),
            Self::Crop64x64 => (64, 64, 3),
        };
        Ok(ObservationSpec {
            preset_name: self.name(),
            frame_width,
            frame_height,
            channels,
            display_width,
            display_height,
        })
    }

    pub fn observation_aspect_ratio(self, display_aspect_ratio: f64) -> f64 {
        match self {
            Self::Crop84x116
            | Self::Crop92x124
            | Self::Crop116x164
            | Self::Crop98x130
            | Self::Crop66x82
            | Self::Crop60x76
            | Self::Crop68x68
            | Self::Crop84x84
            | Self::Crop76x100
            | Self::Crop64x64 => display_aspect_ratio,
        }
    }
}

impl ObservationStackMode {
    pub fn parse(name: &str) -> Result<Self, CoreError> {
        match name {
            "rgb" => Ok(Self::Rgb),
            "gray" => Ok(Self::Gray),
            "luma_chroma" => Ok(Self::LumaChroma),
            _ => Err(CoreError::InvalidObservationPreset {
                name: name.to_owned(),
            }),
        }
    }

    pub fn stacked_channels(self, single_frame_channels: usize, frame_stack: usize) -> usize {
        match self {
            Self::Rgb => single_frame_channels * frame_stack,
            Self::Gray => frame_stack,
            Self::LumaChroma => frame_stack * 2,
        }
    }
}

#[cfg(test)]
mod tests {
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
}
