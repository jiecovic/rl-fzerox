// rust/core/minimap.rs
//! Optional minimap observation layer rendered from raw emulator frames.

mod catalog;
mod marker;
mod render;

use crate::core::error::CoreError;
use crate::core::observation::ObservationCropProfile;
use crate::core::video::{RawVideoFrame, VideoFrame, VideoResizeFilter, sample_rgb};

use catalog::{MinimapRoi, course_minimap_mask, course_minimap_transform, minimap_roi};
use marker::MinimapMarkerHold;
use render::{MinimapRenderScratch, MinimapRenderTarget, render_layer_into, write_zero_layer};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MinimapLayerRequest {
    pub crop_profile: ObservationCropProfile,
    pub course_index: usize,
    pub target_width: usize,
    pub target_height: usize,
    pub resize_filter: VideoResizeFilter,
}

#[derive(Debug, Default)]
pub(crate) struct MinimapLayerRenderer {
    marker_hold: MinimapMarkerHold,
    marker_scratch: Vec<u8>,
    render_scratch: MinimapRenderScratch,
}

impl MinimapLayerRenderer {
    pub fn render_from_raw_into(
        &mut self,
        frame: &RawVideoFrame,
        request: MinimapLayerRequest,
        output: &mut Vec<u8>,
    ) -> Result<(), CoreError> {
        let Some(roi) = minimap_roi(request.crop_profile) else {
            self.clear();
            return write_zero_layer(request, output);
        };
        let Some(mask) = course_minimap_mask(request.crop_profile, request.course_index) else {
            self.clear();
            return write_zero_layer(request, output);
        };
        self.render_with_marker_hold(request, roi, mask, output, |x, y| sample_rgb(frame, x, y))
    }

    pub fn render_from_frame_into(
        &mut self,
        frame: &VideoFrame,
        request: MinimapLayerRequest,
        output: &mut Vec<u8>,
    ) -> Result<(), CoreError> {
        let Some(roi) = minimap_roi(request.crop_profile) else {
            self.clear();
            return write_zero_layer(request, output);
        };
        let Some(mask) = course_minimap_mask(request.crop_profile, request.course_index) else {
            self.clear();
            return write_zero_layer(request, output);
        };
        self.render_with_marker_hold(request, roi, mask, output, |x, y| {
            let index = y.checked_mul(frame.width)?.checked_add(x)?.checked_mul(3)?;
            let pixel = frame.rgb.get(index..index + 3)?;
            Some([pixel[0], pixel[1], pixel[2]])
        })
    }

    pub fn clear(&mut self) {
        self.marker_hold.clear();
        self.marker_scratch.clear();
        self.render_scratch.clear();
    }

    fn render_with_marker_hold(
        &mut self,
        request: MinimapLayerRequest,
        roi: MinimapRoi,
        mask: &[u8],
        output: &mut Vec<u8>,
        sample: impl FnMut(usize, usize) -> Option<[u8; 3]>,
    ) -> Result<(), CoreError> {
        self.marker_hold.ensure_key(request);
        let marker_count = render_layer_into(
            request,
            roi,
            mask,
            course_minimap_transform(request.course_index),
            MinimapRenderTarget {
                output,
                marker_layer: Some(&mut self.marker_scratch),
                scratch: &mut self.render_scratch,
            },
            sample,
        )?;
        self.marker_hold.update(marker_count, &self.marker_scratch);
        self.marker_hold.overlay(output);
        Ok(())
    }
}

#[cfg(test)]
#[path = "minimap/tests.rs"]
mod tests;
