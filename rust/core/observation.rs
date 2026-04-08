// rust/core/observation.rs
//! Observation presets owned by the native layer.
//!
//! Python selects a preset name plus frame-stack depth; Rust owns the spatial
//! geometry, crop policy, and display dimensions for that preset.

use crate::core::error::CoreError;
use crate::core::video::{VideoCrop, cropped_dimensions, display_size};

/// Named single-frame observation layouts exposed to Python.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationPreset {
    NativeCropV1,
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
            _ => Err(CoreError::InvalidObservationPreset {
                name: name.to_owned(),
            }),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::NativeCropV1 => "native_crop_v1",
        }
    }

    pub fn crop(self) -> VideoCrop {
        match self {
            Self::NativeCropV1 => VideoCrop {
                top: 16,
                bottom: 16,
                left: 24,
                right: 24,
            },
        }
    }

    pub fn resolve(
        self,
        raw_frame_width: usize,
        raw_frame_height: usize,
        display_aspect_ratio: f64,
    ) -> Result<ObservationSpec, CoreError> {
        let crop = self.crop();
        let (cropped_width, cropped_height) =
            cropped_dimensions(raw_frame_width, raw_frame_height, crop)?;
        let (display_width, display_height) =
            display_size(cropped_width, cropped_height, display_aspect_ratio);
        let (frame_width, frame_height, channels) = match self {
            Self::NativeCropV1 => (222, 78, 3),
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
            Self::NativeCropV1 => {
                // `0.0` means "preserve the cropped native framebuffer aspect"
                // instead of stretching to the human-facing display ratio.
                let _ = display_aspect_ratio;
                0.0
            }
        }
    }
}
