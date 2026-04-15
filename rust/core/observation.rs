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
    NativeCropV1,
    NativeCropV2,
    NativeCropV3,
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
            "native_crop_v1" => Ok(Self::NativeCropV1),
            "native_crop_v2" => Ok(Self::NativeCropV2),
            "native_crop_v3" => Ok(Self::NativeCropV3),
            _ => Err(CoreError::InvalidObservationPreset {
                name: name.to_owned(),
            }),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::NativeCropV1 => "native_crop_v1",
            Self::NativeCropV2 => "native_crop_v2",
            Self::NativeCropV3 => "native_crop_v3",
        }
    }

    pub fn crop(self, crop_profile: ObservationCropProfile) -> VideoCrop {
        match (self, crop_profile) {
            (
                Self::NativeCropV1 | Self::NativeCropV2 | Self::NativeCropV3,
                ObservationCropProfile::Angrylion,
            ) => VideoCrop {
                top: 16,
                bottom: 16,
                left: 24,
                right: 24,
            },
            (
                Self::NativeCropV1 | Self::NativeCropV2 | Self::NativeCropV3,
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
            Self::NativeCropV1 => (116, 84, 3),
            Self::NativeCropV2 => (124, 92, 3),
            Self::NativeCropV3 => (164, 116, 3),
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
            Self::NativeCropV1 | Self::NativeCropV2 | Self::NativeCropV3 => display_aspect_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ObservationCropProfile, ObservationPreset};
    use crate::core::video::VideoCrop;

    #[test]
    fn crop_profile_keeps_existing_angrylion_crop() {
        assert_eq!(
            ObservationPreset::NativeCropV1.crop(ObservationCropProfile::Angrylion),
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
            ObservationPreset::NativeCropV1.crop(ObservationCropProfile::Gliden64),
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
        let spec = ObservationPreset::NativeCropV1
            .resolve(320, 240, 4.0 / 3.0, ObservationCropProfile::Gliden64)
            .expect("gliden64 crop should resolve");

        assert_eq!((spec.display_width, spec.display_height), (296, 222));
        assert_eq!((spec.frame_width, spec.frame_height), (116, 84));
    }
}
