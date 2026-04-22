// rust/core/host/video/process.rs
use crate::core::error::CoreError;
#[cfg(test)]
use crate::core::video::plan::build_processed_frame_plan;
use crate::core::video::plan::{ProcessedFramePlan, crop_bounds, display_size};
use crate::core::video::resize_rgb;
use crate::core::video::{PixelLayout, RawVideoFrame, VideoCrop, VideoFrame, VideoResizeFilter};

pub fn processed_frame(
    frame: &VideoFrame,
    aspect_ratio: f64,
    target_width: usize,
    target_height: usize,
    rgb: bool,
    crop: VideoCrop,
    resize_filter: VideoResizeFilter,
) -> Result<Vec<u8>, CoreError> {
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
        Ok(to_grayscale(&resized, target_width, target_height))
    }
}

pub fn processed_rgb_frame(
    frame: &VideoFrame,
    aspect_ratio: f64,
    crop: VideoCrop,
    resize_filter: VideoResizeFilter,
) -> Result<VideoFrame, CoreError> {
    let (crop_x, crop_y, crop_width, crop_height) = crop_bounds(frame.width, frame.height, crop)?;
    let cropped =
        if crop_x == 0 && crop_y == 0 && crop_width == frame.width && crop_height == frame.height {
            frame.rgb.clone()
        } else {
            crop_rgb(
                &frame.rgb,
                frame.width,
                crop_x,
                crop_y,
                crop_width,
                crop_height,
            )
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
        cropped
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
    let plan = build_processed_frame_plan(
        frame.width,
        frame.height,
        aspect_ratio,
        target_width,
        target_height,
        rgb,
        crop,
        resize_filter,
    )?;
    let mut output = Vec::new();
    processed_frame_from_raw_into(frame, &plan, &mut output)?;
    Ok(output)
}

pub fn processed_frame_from_raw_into(
    frame: &RawVideoFrame,
    plan: &ProcessedFramePlan,
    output: &mut Vec<u8>,
) -> Result<(), CoreError> {
    let cropped = crop_raw_rgb(
        frame,
        plan.crop_x,
        plan.crop_y,
        plan.crop_width,
        plan.crop_height,
    )?;
    let display_rgb =
        if plan.display_width != plan.crop_width || plan.display_height != plan.crop_height {
            resize_rgb(
                &cropped,
                plan.crop_width,
                plan.crop_height,
                plan.display_width,
                plan.display_height,
                plan.key.resize_filter,
            )?
        } else {
            cropped
        };
    let target_rgb = if plan.key.target_width != plan.display_width
        || plan.key.target_height != plan.display_height
    {
        resize_rgb(
            &display_rgb,
            plan.display_width,
            plan.display_height,
            plan.key.target_width,
            plan.key.target_height,
            plan.key.resize_filter,
        )?
    } else {
        display_rgb
    };

    if plan.key.rgb {
        output.clear();
        output.extend_from_slice(&target_rgb);
    } else {
        output.clear();
        output.extend_from_slice(&to_grayscale(
            &target_rgb,
            plan.key.target_width,
            plan.key.target_height,
        ));
    }
    debug_assert_eq!(output.len(), plan.output_len);
    Ok(())
}

fn crop_raw_rgb(
    frame: &RawVideoFrame,
    crop_x: usize,
    crop_y: usize,
    crop_width: usize,
    crop_height: usize,
) -> Result<Vec<u8>, CoreError> {
    let mut cropped = vec![0_u8; crop_width * crop_height * 3];
    for row in 0..crop_height {
        let dst_start = row * crop_width * 3;
        let dst_end = dst_start + (crop_width * 3);
        write_raw_rgb_row(
            frame,
            crop_x,
            crop_y + row,
            crop_width,
            &mut cropped[dst_start..dst_end],
        )?;
    }
    Ok(cropped)
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
            for column in 0..width {
                let src_index = column * 4;
                let dst_index = column * 3;
                dst[dst_index] = row_bytes[src_index + 2];
                dst[dst_index + 1] = row_bytes[src_index + 1];
                dst[dst_index + 2] = row_bytes[src_index];
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
            for column in 0..width {
                let src_index = column * 2;
                let pixel = u16::from_le_bytes([row_bytes[src_index], row_bytes[src_index + 1]]);
                let red = ((pixel >> 11) & 0x1f) as u8;
                let green = ((pixel >> 5) & 0x3f) as u8;
                let blue = (pixel & 0x1f) as u8;
                let dst_index = column * 3;
                dst[dst_index] = (red << 3) | (red >> 2);
                dst[dst_index + 1] = (green << 2) | (green >> 4);
                dst[dst_index + 2] = (blue << 3) | (blue >> 2);
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
            for column in 0..width {
                let src_index = column * 2;
                let pixel = u16::from_le_bytes([row_bytes[src_index], row_bytes[src_index + 1]]);
                let red = ((pixel >> 10) & 0x1f) as u8;
                let green = ((pixel >> 5) & 0x1f) as u8;
                let blue = (pixel & 0x1f) as u8;
                let dst_index = column * 3;
                dst[dst_index] = (red << 3) | (red >> 2);
                dst[dst_index + 1] = (green << 3) | (green >> 2);
                dst[dst_index + 2] = (blue << 3) | (blue >> 2);
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

fn to_grayscale(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut gray = vec![0_u8; width * height];
    for (index, pixel) in gray.iter_mut().enumerate().take(width * height) {
        let src = index * 3;
        let value = (0.299_f32 * rgb[src] as f32)
            + (0.587_f32 * rgb[src + 1] as f32)
            + (0.114_f32 * rgb[src + 2] as f32);
        *pixel = value.round() as u8;
    }
    gray
}
