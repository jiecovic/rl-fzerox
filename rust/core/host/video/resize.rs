// rust/core/host/video/resize.rs
//! Small wrapper around `fast_image_resize` for policy-frame resampling.

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

use crate::core::error::CoreError;
use crate::core::video::VideoResizeFilter;

#[derive(Debug, Default)]
pub struct ImageResizeScratch {
    resizer: Resizer,
}

#[derive(Clone, Copy, Debug)]
pub struct ResizeRequest {
    pub input_width: usize,
    pub input_height: usize,
    pub output_width: usize,
    pub output_height: usize,
    pub filter: VideoResizeFilter,
}

pub fn resize_rgb(
    rgb: &[u8],
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    filter: VideoResizeFilter,
) -> Result<Vec<u8>, CoreError> {
    let mut output = Vec::new();
    let mut scratch = ImageResizeScratch::default();
    resize_rgb_into(
        rgb,
        ResizeRequest {
            input_width,
            input_height,
            output_width,
            output_height,
            filter,
        },
        &mut output,
        &mut scratch,
    )?;
    Ok(output)
}

pub fn resize_rgb_into(
    rgb: &[u8],
    request: ResizeRequest,
    output: &mut Vec<u8>,
    scratch: &mut ImageResizeScratch,
) -> Result<(), CoreError> {
    resize_u8_image_into(rgb, request, PixelType::U8x3, output, scratch)
}

pub fn resize_luma_into(
    luma: &[u8],
    request: ResizeRequest,
    output: &mut Vec<u8>,
    scratch: &mut ImageResizeScratch,
) -> Result<(), CoreError> {
    resize_u8_image_into(luma, request, PixelType::U8, output, scratch)
}

fn resize_u8_image_into(
    pixels: &[u8],
    request: ResizeRequest,
    pixel_type: PixelType,
    output: &mut Vec<u8>,
    scratch: &mut ImageResizeScratch,
) -> Result<(), CoreError> {
    if request.input_width == request.output_width && request.input_height == request.output_height
    {
        output.clear();
        output.extend_from_slice(pixels);
        return Ok(());
    }

    let output_len = request
        .output_width
        .checked_mul(request.output_height)
        .and_then(|pixels| pixels.checked_mul(pixel_type.size()))
        .ok_or_else(|| CoreError::ResizeFailed {
            message: "resize output dimensions overflow".to_owned(),
        })?;
    let source = ImageRef::new(
        checked_u32(request.input_width)?,
        checked_u32(request.input_height)?,
        pixels,
        pixel_type,
    )
    .map_err(|error| CoreError::ResizeFailed {
        message: error.to_string(),
    })?;
    output.clear();
    output.resize(output_len, 0);
    let mut target = Image::from_slice_u8(
        checked_u32(request.output_width)?,
        checked_u32(request.output_height)?,
        output.as_mut_slice(),
        pixel_type,
    )
    .map_err(|error| CoreError::ResizeFailed {
        message: error.to_string(),
    })?;
    let options = ResizeOptions::new().resize_alg(resize_algorithm(request.filter));
    scratch
        .resizer
        .resize(&source, &mut target, &options)
        .map_err(|error| CoreError::ResizeFailed {
            message: error.to_string(),
        })?;
    Ok(())
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
