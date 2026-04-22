// rust/core/host/video/resize.rs
//! Small wrapper around `fast_image_resize` for policy-frame resampling.

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

use crate::core::error::CoreError;
use crate::core::video::VideoResizeFilter;

pub fn resize_rgb(
    rgb: &[u8],
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    filter: VideoResizeFilter,
) -> Result<Vec<u8>, CoreError> {
    resize_u8_image(
        rgb,
        input_width,
        input_height,
        output_width,
        output_height,
        PixelType::U8x3,
        filter,
    )
}

pub fn resize_luma(
    luma: &[u8],
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    filter: VideoResizeFilter,
) -> Result<Vec<u8>, CoreError> {
    resize_u8_image(
        luma,
        input_width,
        input_height,
        output_width,
        output_height,
        PixelType::U8,
        filter,
    )
}

fn resize_u8_image(
    pixels: &[u8],
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    pixel_type: PixelType,
    filter: VideoResizeFilter,
) -> Result<Vec<u8>, CoreError> {
    if input_width == output_width && input_height == output_height {
        return Ok(pixels.to_vec());
    }

    let source = ImageRef::new(
        checked_u32(input_width)?,
        checked_u32(input_height)?,
        pixels,
        pixel_type,
    )
    .map_err(|error| CoreError::ResizeFailed {
        message: error.to_string(),
    })?;
    let mut target = Image::new(
        checked_u32(output_width)?,
        checked_u32(output_height)?,
        pixel_type,
    );
    let options = ResizeOptions::new().resize_alg(resize_algorithm(filter));
    Resizer::new()
        .resize(&source, &mut target, &options)
        .map_err(|error| CoreError::ResizeFailed {
            message: error.to_string(),
        })?;
    Ok(target.into_vec())
}

fn resize_algorithm(filter: VideoResizeFilter) -> ResizeAlg {
    match filter {
        VideoResizeFilter::Nearest => ResizeAlg::Nearest,
        VideoResizeFilter::Bilinear => ResizeAlg::Interpolation(FilterType::Bilinear),
    }
}

fn checked_u32(value: usize) -> Result<u32, CoreError> {
    u32::try_from(value).map_err(|_| CoreError::ResizeFailed {
        message: format!("resize dimension {value} does not fit into u32"),
    })
}
