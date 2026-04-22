// rust/core/minimap/render.rs
//! Minimap crop, mask, transform, and resize pipeline.

use crate::core::error::CoreError;
use crate::core::video::{ImageResizeScratch, ResizeRequest, VideoResizeFilter, resize_luma_into};

use super::MinimapLayerRequest;
use super::catalog::{MinimapRoi, MinimapTransform};
use super::marker::{TRACK_LUMA, is_player_marker_color};

#[derive(Debug, Default)]
pub(super) struct MinimapRenderScratch {
    roi_layer: Vec<u8>,
    transformed_roi_layer: Vec<u8>,
    transformed_marker_layer: Vec<u8>,
    resized_marker_layer: Vec<u8>,
    resize: ImageResizeScratch,
}

impl MinimapRenderScratch {
    pub(super) fn clear(&mut self) {
        self.roi_layer.clear();
        self.transformed_roi_layer.clear();
        self.transformed_marker_layer.clear();
        self.resized_marker_layer.clear();
    }
}

pub(super) struct MinimapRenderTarget<'a> {
    pub output: &'a mut Vec<u8>,
    pub marker_layer: Option<&'a mut Vec<u8>>,
    pub scratch: &'a mut MinimapRenderScratch,
}

pub(super) fn render_layer_into(
    request: MinimapLayerRequest,
    roi: MinimapRoi,
    mask: &[u8],
    transform: MinimapTransform,
    target: MinimapRenderTarget<'_>,
    mut sample: impl FnMut(usize, usize) -> Option<[u8; 3]>,
) -> Result<usize, CoreError> {
    let MinimapRenderTarget {
        output,
        marker_layer,
        scratch,
    } = target;
    let output_len = request
        .target_width
        .checked_mul(request.target_height)
        .ok_or(CoreError::NoFrameAvailable)?;
    let roi_len = roi
        .width
        .checked_mul(roi.height)
        .ok_or(CoreError::NoFrameAvailable)?;
    if mask.len() != roi.width * roi.height {
        return Err(CoreError::NoFrameAvailable);
    }

    scratch.roi_layer.clear();
    scratch.roi_layer.resize(roi_len, 0);
    let mut marker_count = 0_usize;
    let mut marker_layer = marker_layer;
    if let Some(marker_layer) = marker_layer.as_deref_mut() {
        marker_layer.clear();
        marker_layer.resize(roi_len, 0);
    }

    if let Some(marker_layer) = marker_layer.as_deref_mut() {
        for (roi_y, ((mask_row, output_row), marker_row)) in mask
            .chunks_exact(roi.width)
            .zip(scratch.roi_layer.chunks_exact_mut(roi.width))
            .zip(marker_layer.chunks_exact_mut(roi.width))
            .enumerate()
        {
            marker_count += render_roi_row(
                roi,
                roi_y,
                mask_row,
                output_row,
                Some(marker_row),
                &mut sample,
            )?;
        }
    } else {
        for (roi_y, (mask_row, output_row)) in mask
            .chunks_exact(roi.width)
            .zip(scratch.roi_layer.chunks_exact_mut(roi.width))
            .enumerate()
        {
            render_roi_row(roi, roi_y, mask_row, output_row, None, &mut sample)?;
        }
    }

    let (roi_layer, layer_width, layer_height): (&[u8], usize, usize) = match transform {
        MinimapTransform::Identity => (&scratch.roi_layer, roi.width, roi.height),
        MinimapTransform::Rotate90Clockwise => {
            let (width, height) = rotate_luma_90_clockwise_into(
                &scratch.roi_layer,
                roi.width,
                roi.height,
                &mut scratch.transformed_roi_layer,
            );
            (&scratch.transformed_roi_layer, width, height)
        }
    };

    resize_luma_into(
        roi_layer,
        ResizeRequest {
            input_width: layer_width,
            input_height: layer_height,
            output_width: request.target_width,
            output_height: request.target_height,
            filter: request.resize_filter,
        },
        output,
        &mut scratch.resize,
    )?;
    if let Some(marker_layer) = marker_layer {
        let marker_input = match transform {
            MinimapTransform::Identity => marker_layer.as_slice(),
            MinimapTransform::Rotate90Clockwise => {
                rotate_luma_90_clockwise_into(
                    marker_layer,
                    roi.width,
                    roi.height,
                    &mut scratch.transformed_marker_layer,
                );
                scratch.transformed_marker_layer.as_slice()
            }
        };
        resize_luma_into(
            marker_input,
            ResizeRequest {
                input_width: layer_width,
                input_height: layer_height,
                output_width: request.target_width,
                output_height: request.target_height,
                filter: VideoResizeFilter::Nearest,
            },
            &mut scratch.resized_marker_layer,
            &mut scratch.resize,
        )?;
        marker_layer.clear();
        marker_layer.extend_from_slice(&scratch.resized_marker_layer);
    }
    debug_assert_eq!(output.len(), output_len);
    Ok(marker_count)
}

fn rotate_luma_90_clockwise_into(
    layer: &[u8],
    width: usize,
    height: usize,
    output: &mut Vec<u8>,
) -> (usize, usize) {
    let output_width = height;
    let output_height = width;
    output.clear();
    output.resize(layer.len(), 0);
    for (y, row) in layer.chunks_exact(width).enumerate() {
        for (x, value) in row.iter().enumerate() {
            let target_x = height - 1 - y;
            let target_y = x;
            output[target_y * output_width + target_x] = *value;
        }
    }
    (output_width, output_height)
}

fn render_roi_row(
    roi: MinimapRoi,
    roi_y: usize,
    mask_row: &[u8],
    output_row: &mut [u8],
    marker_row: Option<&mut [u8]>,
    sample: &mut impl FnMut(usize, usize) -> Option<[u8; 3]>,
) -> Result<usize, CoreError> {
    let mut marker_count = 0_usize;
    match marker_row {
        Some(marker_row) => {
            for (roi_x, ((mask_value, output_value), marker_value)) in mask_row
                .iter()
                .zip(output_row.iter_mut())
                .zip(marker_row.iter_mut())
                .enumerate()
            {
                let [red, green, blue] =
                    sample(roi.x + roi_x, roi.y + roi_y).ok_or(CoreError::NoFrameAvailable)?;
                if *mask_value == 0 {
                    *output_value = 0;
                    continue;
                }
                if is_player_marker_color(red, green, blue) {
                    *marker_value = 1;
                    marker_count += 1;
                }
                *output_value = TRACK_LUMA;
            }
        }
        None => {
            for (roi_x, (mask_value, output_value)) in
                mask_row.iter().zip(output_row.iter_mut()).enumerate()
            {
                sample(roi.x + roi_x, roi.y + roi_y).ok_or(CoreError::NoFrameAvailable)?;
                *output_value = if *mask_value == 0 { 0 } else { TRACK_LUMA };
            }
        }
    }
    Ok(marker_count)
}

pub(super) fn write_zero_layer(
    request: MinimapLayerRequest,
    output: &mut Vec<u8>,
) -> Result<(), CoreError> {
    let output_len = request
        .target_width
        .checked_mul(request.target_height)
        .ok_or(CoreError::NoFrameAvailable)?;
    output.clear();
    output.resize(output_len, 0);
    Ok(())
}
