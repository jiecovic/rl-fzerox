// rust/core/host/video/plan.rs
//! Render-plan construction for cropping and resampling raw frames.

use crate::core::error::CoreError;
use crate::core::video::{VideoCrop, VideoResizeFilter};

/// Cache key for one raw-source to processed-target render plan.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
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

/// Precomputed crop/resize plan reused for observation/display rendering.
#[derive(Clone, Debug)]
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
    source_width: usize,
    source_height: usize,
    aspect_ratio: f64,
    target_width: usize,
    target_height: usize,
    rgb: bool,
    crop: VideoCrop,
    resize_filter: VideoResizeFilter,
) -> Result<ProcessedFramePlan, CoreError> {
    let (crop_x, crop_y, crop_width, crop_height) = crop_bounds(source_width, source_height, crop)?;
    let (display_width, display_height) = display_size(crop_width, crop_height, aspect_ratio);
    let channels = if rgb { 3 } else { 1 };

    Ok(ProcessedFramePlan {
        key: ProcessedFramePlanKey {
            source_width,
            source_height,
            crop,
            aspect_ratio_bits: aspect_ratio.to_bits(),
            target_width,
            target_height,
            rgb,
            resize_filter,
        },
        crop_x,
        crop_y,
        crop_width,
        crop_height,
        display_width,
        display_height,
        output_len: target_width * target_height * channels,
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
