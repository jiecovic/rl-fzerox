// rust/core/minimap/render.rs
//! Minimap crop, mask, transform, and resize pipeline.

use crate::core::error::CoreError;
use crate::core::video::{VideoResizeFilter, resize_luma};

use super::MinimapLayerRequest;
use super::catalog::{MinimapRoi, MinimapTransform};
use super::marker::{TRACK_LUMA, is_player_marker_color};

pub(super) fn render_layer_into(
    request: MinimapLayerRequest,
    roi: MinimapRoi,
    mask: &[u8],
    transform: MinimapTransform,
    output: &mut Vec<u8>,
    marker_layer: Option<&mut Vec<u8>>,
    mut sample: impl FnMut(usize, usize) -> Option<[u8; 3]>,
) -> Result<usize, CoreError> {
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

    let mut roi_output = vec![0_u8; roi_len];
    let mut marker_count = 0_usize;
    let mut marker_layer = marker_layer;
    if let Some(marker_layer) = marker_layer.as_deref_mut() {
        marker_layer.clear();
        marker_layer.resize(roi_len, 0);
    }

    if let Some(marker_layer) = marker_layer.as_deref_mut() {
        for (roi_y, ((mask_row, output_row), marker_row)) in mask
            .chunks_exact(roi.width)
            .zip(roi_output.chunks_exact_mut(roi.width))
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
            .zip(roi_output.chunks_exact_mut(roi.width))
            .enumerate()
        {
            render_roi_row(roi, roi_y, mask_row, output_row, None, &mut sample)?;
        }
    }

    let transformed_roi_storage;
    let (roi_layer, layer_width, layer_height): (&[u8], usize, usize) = match transform {
        MinimapTransform::Identity => (&roi_output, roi.width, roi.height),
        MinimapTransform::Rotate90Clockwise => {
            let (rotated, width, height) =
                rotate_luma_90_clockwise(&roi_output, roi.width, roi.height);
            transformed_roi_storage = rotated;
            (&transformed_roi_storage, width, height)
        }
    };

    let resized_output = resize_luma(
        roi_layer,
        layer_width,
        layer_height,
        request.target_width,
        request.target_height,
        request.resize_filter,
    )?;
    output.clear();
    output.extend_from_slice(&resized_output);
    if let Some(marker_layer) = marker_layer {
        let transformed_marker_storage;
        let marker_input = match transform {
            MinimapTransform::Identity => marker_layer.as_slice(),
            MinimapTransform::Rotate90Clockwise => {
                let (rotated, _, _) = rotate_luma_90_clockwise(marker_layer, roi.width, roi.height);
                transformed_marker_storage = rotated;
                transformed_marker_storage.as_slice()
            }
        };
        let resized_marker = resize_luma(
            marker_input,
            layer_width,
            layer_height,
            request.target_width,
            request.target_height,
            VideoResizeFilter::Nearest,
        )?;
        marker_layer.clear();
        marker_layer.extend_from_slice(&resized_marker);
    }
    debug_assert_eq!(output.len(), output_len);
    Ok(marker_count)
}

fn rotate_luma_90_clockwise(layer: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    let output_width = height;
    let output_height = width;
    let mut output = vec![0_u8; layer.len()];
    for (y, row) in layer.chunks_exact(width).enumerate() {
        for (x, value) in row.iter().enumerate() {
            let target_x = height - 1 - y;
            let target_y = x;
            output[target_y * output_width + target_x] = *value;
        }
    }
    (output, output_width, output_height)
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
