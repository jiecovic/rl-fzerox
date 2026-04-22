// rust/core/host/video/plan.rs
//! Render-plan construction for cropping and resampling raw frames.

use crate::core::error::CoreError;
use crate::core::video::{VideoCrop, VideoResizeFilter};

/// Cache key for one raw-source to processed-target render plan.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ProcessedFramePlanKey {
    pub source_width: usize,
    pub source_height: usize,
    pub crop: VideoCrop,
    pub aspect_ratio_bits: u64,
    pub target_width: usize,
    pub target_height: usize,
    pub rgb: bool,
    pub resize_filter: VideoResizeFilter,
}

/// Inputs needed to precompute one processed frame plan.
#[derive(Clone, Copy, Debug)]
pub struct ProcessedFramePlanRequest {
    pub source_width: usize,
    pub source_height: usize,
    pub aspect_ratio: f64,
    pub target_width: usize,
    pub target_height: usize,
    pub rgb: bool,
    pub crop: VideoCrop,
    pub resize_filter: VideoResizeFilter,
}

impl ProcessedFramePlanRequest {
    pub fn key(self) -> ProcessedFramePlanKey {
        ProcessedFramePlanKey {
            source_width: self.source_width,
            source_height: self.source_height,
            crop: self.crop,
            aspect_ratio_bits: self.aspect_ratio.to_bits(),
            target_width: self.target_width,
            target_height: self.target_height,
            rgb: self.rgb,
            resize_filter: self.resize_filter,
        }
    }
}

/// Precomputed crop/resize plan reused for observation/display rendering.
#[derive(Clone, Copy, Debug)]
pub struct ProcessedFramePlan {
    pub key: ProcessedFramePlanKey,
    pub crop_x: usize,
    pub crop_y: usize,
    pub crop_width: usize,
    pub crop_height: usize,
    pub display_width: usize,
    pub display_height: usize,
    pub output_len: usize,
}

/// Resolve the output size after applying a display aspect ratio.
pub fn display_size(width: usize, height: usize, aspect_ratio: f64) -> (usize, usize) {
    if aspect_ratio <= 0.0 {
        return (width, height);
    }

    let display_height = ((width as f64) / aspect_ratio).round() as usize;
    (width, display_height.max(1))
}

/// Precompute the crop and sampling plan for one fixed source/target pair.
pub fn build_processed_frame_plan(
    request: ProcessedFramePlanRequest,
) -> Result<ProcessedFramePlan, CoreError> {
    let (crop_x, crop_y, crop_width, crop_height) =
        crop_bounds(request.source_width, request.source_height, request.crop)?;
    let (display_width, display_height) =
        display_size(crop_width, crop_height, request.aspect_ratio);
    let channels = if request.rgb { 3 } else { 1 };

    Ok(ProcessedFramePlan {
        key: request.key(),
        crop_x,
        crop_y,
        crop_width,
        crop_height,
        display_width,
        display_height,
        output_len: request.target_width * request.target_height * channels,
    })
}

/// Return the frame dimensions after applying the configured crop.
pub fn cropped_dimensions(
    frame_width: usize,
    frame_height: usize,
    crop: VideoCrop,
) -> Result<(usize, usize), CoreError> {
    let (_, _, width, height) = crop_bounds(frame_width, frame_height, crop)?;
    Ok((width, height))
}

pub(super) fn crop_bounds(
    frame_width: usize,
    frame_height: usize,
    crop: VideoCrop,
) -> Result<(usize, usize, usize, usize), CoreError> {
    if crop.left + crop.right >= frame_width || crop.top + crop.bottom >= frame_height {
        return Err(CoreError::InvalidVideoCrop {
            frame_width,
            frame_height,
            top: crop.top,
            bottom: crop.bottom,
            left: crop.left,
            right: crop.right,
        });
    }

    Ok((
        crop.left,
        crop.top,
        frame_width - crop.left - crop.right,
        frame_height - crop.top - crop.bottom,
    ))
}
