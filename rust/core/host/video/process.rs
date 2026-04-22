// rust/core/host/video/process.rs
use std::borrow::Cow;

use crate::core::error::CoreError;
use crate::core::video::plan::{ProcessedFramePlan, crop_bounds, display_size};
#[cfg(test)]
use crate::core::video::plan::{ProcessedFramePlanRequest, build_processed_frame_plan};
use crate::core::video::{ImageResizeScratch, ResizeRequest, resize_rgb, resize_rgb_into};
use crate::core::video::{
    PixelLayout, RawVideoFrame, VideoCrop, VideoFrame, VideoResizeFilter, rgb_to_luma_in_place,
    rgb_to_luma_into, rgb_to_luma_vec,
};

use super::convert::{expand_5_to_8, expand_6_to_8};

pub fn processed_frame(
    frame: &VideoFrame,
    aspect_ratio: f64,
    target_width: usize,
    target_height: usize,
    rgb: bool,
    crop: VideoCrop,
    resize_filter: VideoResizeFilter,
) -> Result<Vec<u8>, CoreError> {
    // Owned decoded-frame path for fallback/display use. The training hot path
    // should prefer `processed_frame_from_raw_into`, which reuses buffers.
    let display_frame = processed_rgb_frame(frame, aspect_ratio, crop, resize_filter)?;
    let resized = if display_frame.width != target_width || display_frame.height != target_height {
        resize_rgb(
            &display_frame.rgb,
            display_frame.width,
            display_frame.height,
            target_width,
            target_height,
            resize_filter,
        )?
    } else {
        display_frame.rgb
    };

    if rgb {
        Ok(resized)
    } else {
        Ok(rgb_to_luma_vec(&resized))
    }
}

pub fn processed_rgb_frame(
    frame: &VideoFrame,
    aspect_ratio: f64,
    crop: VideoCrop,
    resize_filter: VideoResizeFilter,
) -> Result<VideoFrame, CoreError> {
    let (crop_x, crop_y, crop_width, crop_height) = crop_bounds(frame.width, frame.height, crop)?;
    let cropped: Cow<'_, [u8]> =
        if crop_x == 0 && crop_y == 0 && crop_width == frame.width && crop_height == frame.height {
            Cow::Borrowed(&frame.rgb)
        } else {
            Cow::Owned(crop_rgb(
                &frame.rgb,
                frame.width,
                crop_x,
                crop_y,
                crop_width,
                crop_height,
            ))
        };

    let (display_width, display_height) = display_size(crop_width, crop_height, aspect_ratio);
    let aspect_corrected = if display_width != crop_width || display_height != crop_height {
        resize_rgb(
            &cropped,
            crop_width,
            crop_height,
            display_width,
            display_height,
            resize_filter,
        )?
    } else {
        cropped.into_owned()
    };
    Ok(VideoFrame {
        width: display_width,
        height: display_height,
        rgb: aspect_corrected,
    })
}

#[cfg(test)]
pub fn processed_frame_from_raw(
    frame: &RawVideoFrame,
    aspect_ratio: f64,
    target_width: usize,
    target_height: usize,
    rgb: bool,
    crop: VideoCrop,
    resize_filter: VideoResizeFilter,
) -> Result<Vec<u8>, CoreError> {
    let plan = build_processed_frame_plan(ProcessedFramePlanRequest {
        source_width: frame.width,
        source_height: frame.height,
        aspect_ratio,
        target_width,
        target_height,
        rgb,
        crop,
        resize_filter,
    })?;
    let mut output = Vec::new();
    let mut scratch_output = Vec::new();
    let mut resize_scratch = ImageResizeScratch::default();
    processed_frame_from_raw_into(
        frame,
        &plan,
        &mut output,
        &mut scratch_output,
        &mut resize_scratch,
    )?;
    Ok(output)
}

pub fn processed_frame_from_raw_into(
    frame: &RawVideoFrame,
    plan: &ProcessedFramePlan,
    output: &mut Vec<u8>,
    scratch_output: &mut Vec<u8>,
    resize_scratch: &mut ImageResizeScratch,
) -> Result<(), CoreError> {
    crop_raw_rgb_into(
        frame,
        plan.crop_x,
        plan.crop_y,
        plan.crop_width,
        plan.crop_height,
        output,
    )?;

    let display_resized =
        plan.display_width != plan.crop_width || plan.display_height != plan.crop_height;
    let target_resized = plan.key.target_width != plan.display_width
        || plan.key.target_height != plan.display_height;

    if display_resized {
        resize_rgb_into(
            output,
            ResizeRequest {
                input_width: plan.crop_width,
                input_height: plan.crop_height,
                output_width: plan.display_width,
                output_height: plan.display_height,
                filter: plan.key.resize_filter,
            },
            scratch_output,
            resize_scratch,
        )?;
        if target_resized {
            resize_rgb_into(
                scratch_output,
                ResizeRequest {
                    input_width: plan.display_width,
                    input_height: plan.display_height,
                    output_width: plan.key.target_width,
                    output_height: plan.key.target_height,
                    filter: plan.key.resize_filter,
                },
                output,
                resize_scratch,
            )?;
            if !plan.key.rgb {
                rgb_to_luma_in_place(output);
            }
        } else {
            write_final_rgb_from_scratch(scratch_output, output, plan.key.rgb);
        }
    } else if target_resized {
        resize_rgb_into(
            output,
            ResizeRequest {
                input_width: plan.display_width,
                input_height: plan.display_height,
                output_width: plan.key.target_width,
                output_height: plan.key.target_height,
                filter: plan.key.resize_filter,
            },
            scratch_output,
            resize_scratch,
        )?;
        write_final_rgb_from_scratch(scratch_output, output, plan.key.rgb);
    } else if !plan.key.rgb {
        rgb_to_luma_in_place(output);
    }

    debug_assert_eq!(output.len(), plan.output_len);
    Ok(())
}

fn write_final_rgb_from_scratch(rgb: &mut Vec<u8>, output: &mut Vec<u8>, keep_rgb: bool) {
    if keep_rgb {
        std::mem::swap(output, rgb);
    } else {
        write_final_rgb(rgb, output, false);
    }
}

fn write_final_rgb(rgb: &[u8], output: &mut Vec<u8>, keep_rgb: bool) {
    if keep_rgb {
        output.clear();
        output.extend_from_slice(rgb);
    } else {
        output.clear();
        output.resize(rgb.len() / 3, 0);
        rgb_to_luma_into(rgb, output);
    }
}

fn crop_raw_rgb_into(
    frame: &RawVideoFrame,
    crop_x: usize,
    crop_y: usize,
    crop_width: usize,
    crop_height: usize,
    output: &mut Vec<u8>,
) -> Result<(), CoreError> {
    let row_len = crop_width * 3;
    output.clear();
    output.resize(row_len * crop_height, 0);
    for (row, dst) in output.chunks_exact_mut(row_len).enumerate() {
        write_raw_rgb_row(frame, crop_x, crop_y + row, crop_width, dst)?;
    }
    Ok(())
}

fn write_raw_rgb_row(
    frame: &RawVideoFrame,
    start_x: usize,
    y: usize,
    width: usize,
    dst: &mut [u8],
) -> Result<(), CoreError> {
    match frame.pixel_layout {
        PixelLayout::Argb8888 => {
            let row_offset = y
                .checked_mul(frame.pitch)
                .and_then(|offset| offset.checked_add(start_x.checked_mul(4)?))
                .ok_or(CoreError::NoFrameAvailable)?;
            let row_bytes = frame
                .bytes
                .get(row_offset..row_offset + (width * 4))
                .ok_or(CoreError::NoFrameAvailable)?;
            for (src_pixel, dst_pixel) in row_bytes.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
                dst_pixel[0] = src_pixel[2];
                dst_pixel[1] = src_pixel[1];
                dst_pixel[2] = src_pixel[0];
            }
            Ok(())
        }
        PixelLayout::Rgb565 => {
            let row_offset = y
                .checked_mul(frame.pitch)
                .and_then(|offset| offset.checked_add(start_x.checked_mul(2)?))
                .ok_or(CoreError::NoFrameAvailable)?;
            let row_bytes = frame
                .bytes
                .get(row_offset..row_offset + (width * 2))
                .ok_or(CoreError::NoFrameAvailable)?;
            for (src_pixel, dst_pixel) in row_bytes.chunks_exact(2).zip(dst.chunks_exact_mut(3)) {
                let pixel = u16::from_le_bytes([src_pixel[0], src_pixel[1]]);
                let red = ((pixel >> 11) & 0x1f) as u8;
                let green = ((pixel >> 5) & 0x3f) as u8;
                let blue = (pixel & 0x1f) as u8;
                dst_pixel[0] = expand_5_to_8(red);
                dst_pixel[1] = expand_6_to_8(green);
                dst_pixel[2] = expand_5_to_8(blue);
            }
            Ok(())
        }
        PixelLayout::Argb1555 => {
            let row_offset = y
                .checked_mul(frame.pitch)
                .and_then(|offset| offset.checked_add(start_x.checked_mul(2)?))
                .ok_or(CoreError::NoFrameAvailable)?;
            let row_bytes = frame
                .bytes
                .get(row_offset..row_offset + (width * 2))
                .ok_or(CoreError::NoFrameAvailable)?;
            for (src_pixel, dst_pixel) in row_bytes.chunks_exact(2).zip(dst.chunks_exact_mut(3)) {
                let pixel = u16::from_le_bytes([src_pixel[0], src_pixel[1]]);
                let red = ((pixel >> 10) & 0x1f) as u8;
                let green = ((pixel >> 5) & 0x1f) as u8;
                let blue = (pixel & 0x1f) as u8;
                dst_pixel[0] = expand_5_to_8(red);
                dst_pixel[1] = expand_5_to_8(green);
                dst_pixel[2] = expand_5_to_8(blue);
            }
            Ok(())
        }
    }
}

fn crop_rgb(
    rgb: &[u8],
    frame_width: usize,
    crop_x: usize,
    crop_y: usize,
    crop_width: usize,
    crop_height: usize,
) -> Vec<u8> {
    let mut cropped = vec![0_u8; crop_width * crop_height * 3];
    for row in 0..crop_height {
        let src_start = ((crop_y + row) * frame_width + crop_x) * 3;
        let src_end = src_start + (crop_width * 3);
        let dst_start = row * crop_width * 3;
        let dst_end = dst_start + (crop_width * 3);
        cropped[dst_start..dst_end].copy_from_slice(&rgb[src_start..src_end]);
    }
    cropped
}
