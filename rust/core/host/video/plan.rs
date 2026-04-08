// rust/core/host/video/plan.rs
//! Render-plan construction for cropping and resampling raw frames.

use crate::core::error::CoreError;
use crate::core::video::VideoCrop;

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
}

/// Precomputed sampling plan reused for observation/display rendering.
#[derive(Clone, Debug)]
pub struct ProcessedFramePlan {
    pub key: ProcessedFramePlanKey,
    pub crop_x: usize,
    pub crop_y: usize,
    pub output_len: usize,
    pub direct_row_copy: bool,
    pub x_index: Vec<usize>,
    pub y_index: Vec<usize>,
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
) -> Result<ProcessedFramePlan, CoreError> {
    let (crop_x, crop_y, crop_width, crop_height) = crop_bounds(source_width, source_height, crop)?;
    let (display_width, display_height) = display_size(crop_width, crop_height, aspect_ratio);
    let direct_row_copy = rgb && target_width == display_width && target_height == display_height;
    let x_index = if direct_row_copy {
        Vec::new()
    } else {
        compose_axis_index_map(crop_width, display_width, target_width)
    };
    let y_index = if direct_row_copy {
        axis_index_map(crop_height, display_height)
    } else {
        compose_axis_index_map(crop_height, display_height, target_height)
    };
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
        },
        crop_x,
        crop_y,
        output_len: target_width * target_height * channels,
        direct_row_copy,
        x_index,
        y_index,
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

fn axis_index_map(input_size: usize, output_size: usize) -> Vec<usize> {
    if output_size <= 1 || input_size <= 1 {
        return vec![0; output_size.max(1)];
    }

    let scale = (input_size - 1) as f64 / (output_size - 1) as f64;
    (0..output_size)
        .map(|index| ((index as f64) * scale).round() as usize)
        .collect()
}

fn compose_axis_index_map(
    input_size: usize,
    intermediate_size: usize,
    output_size: usize,
) -> Vec<usize> {
    let intermediate_index = axis_index_map(input_size, intermediate_size);
    let output_index = axis_index_map(intermediate_size, output_size);
    output_index
        .into_iter()
        .map(|index| intermediate_index[index])
        .collect()
}
