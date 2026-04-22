// rust/core/host/runtime/render.rs
//! Host frame, observation, display, and minimap rendering methods.

use crate::core::callbacks::StackedObservationRequest;
use crate::core::error::CoreError;
use crate::core::minimap::MinimapLayerRequest;
use crate::core::observation::{ObservationPreset, ObservationSpec};
use crate::core::video::VideoResizeFilter;

use super::host::Host;

impl Host {
    pub fn frame_rgb(&mut self) -> Result<&[u8], CoreError> {
        self.callbacks
            .frame()
            .map(|frame| frame.rgb.as_slice())
            .ok_or(CoreError::NoFrameAvailable)
    }

    pub fn observation_spec(
        &self,
        preset: ObservationPreset,
    ) -> Result<ObservationSpec, CoreError> {
        let (frame_height, frame_width, _) = self.frame_shape;
        preset.resolve(
            frame_width,
            frame_height,
            self.display_aspect_ratio,
            self.observation_crop_profile,
        )
    }

    pub fn observation_frame(
        &mut self,
        preset: ObservationPreset,
        frame_stack: usize,
        stack_mode: crate::core::observation::ObservationStackMode,
        minimap_layer: bool,
        resize_filter: VideoResizeFilter,
        minimap_resize_filter: VideoResizeFilter,
    ) -> Result<&[u8], CoreError> {
        let spec = self.observation_spec(preset)?;
        let aspect_ratio = preset.observation_aspect_ratio(self.display_aspect_ratio);
        let crop = preset.crop(self.observation_crop_profile);
        let minimap_layer_request =
            self.minimap_layer_request(minimap_layer, &spec, minimap_resize_filter);
        self.callbacks
            .stacked_observation_frame(StackedObservationRequest {
                aspect_ratio,
                target_width: spec.frame_width,
                target_height: spec.frame_height,
                rgb: spec.channels == 3,
                crop,
                resize_filter,
                frame_stack,
                stack_mode,
                minimap_layer: minimap_layer_request,
            })
    }

    pub fn display_frame(&mut self, preset: ObservationPreset) -> Result<&[u8], CoreError> {
        let spec = self.observation_spec(preset)?;
        self.callbacks.observation_frame(
            self.display_aspect_ratio,
            spec.display_width,
            spec.display_height,
            true,
            preset.crop(self.observation_crop_profile),
            VideoResizeFilter::Nearest,
        )
    }

    fn minimap_layer_request(
        &mut self,
        enabled: bool,
        spec: &ObservationSpec,
        resize_filter: VideoResizeFilter,
    ) -> Option<MinimapLayerRequest> {
        if !enabled {
            return None;
        }
        let course_index = self
            .telemetry()
            .map(|telemetry| telemetry.course_index as usize)
            .unwrap_or(usize::MAX);
        Some(MinimapLayerRequest {
            crop_profile: self.observation_crop_profile,
            course_index,
            target_width: spec.frame_width,
            target_height: spec.frame_height,
            resize_filter,
        })
    }
}
