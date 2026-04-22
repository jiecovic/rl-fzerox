// rust/core/host/video/convert.rs
use std::ffi::c_void;

use crate::core::video::{PixelLayout, RawVideoFrame, VideoFrame};

pub fn decode_frame(frame: &RawVideoFrame) -> Option<VideoFrame> {
    convert_frame(
        frame.bytes.as_ptr().cast::<c_void>(),
        frame.width,
        frame.height,
        frame.pitch,
        frame.pixel_layout,
    )
}

pub fn convert_frame(
    data: *const c_void,
    width: usize,
    height: usize,
    pitch: usize,
    pixel_layout: PixelLayout,
) -> Option<VideoFrame> {
    if data.is_null() || data as usize == usize::MAX {
        return None;
    }

    let rgb = match pixel_layout {
        PixelLayout::Argb1555 => convert_argb1555(data, width, height, pitch),
        PixelLayout::Argb8888 => convert_argb8888(data, width, height, pitch),
        PixelLayout::Rgb565 => convert_rgb565(data, width, height, pitch),
    };
    Some(VideoFrame { width, height, rgb })
}

pub(crate) fn convert_argb8888(
    data: *const c_void,
    width: usize,
    height: usize,
    pitch: usize,
) -> Vec<u8> {
    let mut rgb = vec![0_u8; width * height * 3];
    for row in 0..height {
        let src =
            unsafe { std::slice::from_raw_parts((data as *const u8).add(row * pitch), width * 4) };
        let dst = &mut rgb[row * width * 3..(row + 1) * width * 3];
        for (src_pixel, dst_pixel) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
            dst_pixel[0] = src_pixel[2];
            dst_pixel[1] = src_pixel[1];
            dst_pixel[2] = src_pixel[0];
        }
    }
    rgb
}

pub(crate) fn convert_rgb565(
    data: *const c_void,
    width: usize,
    height: usize,
    pitch: usize,
) -> Vec<u8> {
    let mut rgb = vec![0_u8; width * height * 3];
    for row in 0..height {
        let src =
            unsafe { std::slice::from_raw_parts((data as *const u8).add(row * pitch), width * 2) };
        let dst = &mut rgb[row * width * 3..(row + 1) * width * 3];
        for (src_pixel, dst_pixel) in src.chunks_exact(2).zip(dst.chunks_exact_mut(3)) {
            let pixel = u16::from_le_bytes([src_pixel[0], src_pixel[1]]);
            let red = ((pixel >> 11) & 0x1f) as u8;
            let green = ((pixel >> 5) & 0x3f) as u8;
            let blue = (pixel & 0x1f) as u8;

            dst_pixel[0] = expand_5_to_8(red);
            dst_pixel[1] = expand_6_to_8(green);
            dst_pixel[2] = expand_5_to_8(blue);
        }
    }
    rgb
}

pub(crate) fn convert_argb1555(
    data: *const c_void,
    width: usize,
    height: usize,
    pitch: usize,
) -> Vec<u8> {
    let mut rgb = vec![0_u8; width * height * 3];
    for row in 0..height {
        let src =
            unsafe { std::slice::from_raw_parts((data as *const u8).add(row * pitch), width * 2) };
        let dst = &mut rgb[row * width * 3..(row + 1) * width * 3];
        for (src_pixel, dst_pixel) in src.chunks_exact(2).zip(dst.chunks_exact_mut(3)) {
            let pixel = u16::from_le_bytes([src_pixel[0], src_pixel[1]]);
            let red = ((pixel >> 10) & 0x1f) as u8;
            let green = ((pixel >> 5) & 0x1f) as u8;
            let blue = (pixel & 0x1f) as u8;

            dst_pixel[0] = expand_5_to_8(red);
            dst_pixel[1] = expand_5_to_8(green);
            dst_pixel[2] = expand_5_to_8(blue);
        }
    }
    rgb
}

#[inline]
pub(crate) fn sample_rgb(frame: &RawVideoFrame, x: usize, y: usize) -> Option<[u8; 3]> {
    match frame.pixel_layout {
        PixelLayout::Argb8888 => {
            let offset = y.checked_mul(frame.pitch)?.checked_add(x.checked_mul(4)?)?;
            let bytes = frame.bytes.get(offset..offset + 4)?;
            Some([bytes[2], bytes[1], bytes[0]])
        }
        PixelLayout::Rgb565 => {
            let pixel = read_le_u16(frame, x, y)?;
            let red = ((pixel >> 11) & 0x1f) as u8;
            let green = ((pixel >> 5) & 0x3f) as u8;
            let blue = (pixel & 0x1f) as u8;
            Some([
                expand_5_to_8(red),
                expand_6_to_8(green),
                expand_5_to_8(blue),
            ])
        }
        PixelLayout::Argb1555 => {
            let pixel = read_le_u16(frame, x, y)?;
            let red = ((pixel >> 10) & 0x1f) as u8;
            let green = ((pixel >> 5) & 0x1f) as u8;
            let blue = (pixel & 0x1f) as u8;
            Some([
                expand_5_to_8(red),
                expand_5_to_8(green),
                expand_5_to_8(blue),
            ])
        }
    }
}

#[inline(always)]
fn read_le_u16(frame: &RawVideoFrame, x: usize, y: usize) -> Option<u16> {
    let offset = y.checked_mul(frame.pitch)?.checked_add(x.checked_mul(2)?)?;
    let bytes = frame.bytes.get(offset..offset + 2)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

#[inline(always)]
pub(super) fn expand_5_to_8(value: u8) -> u8 {
    (value << 3) | (value >> 2)
}

#[inline(always)]
pub(super) fn expand_6_to_8(value: u8) -> u8 {
    (value << 2) | (value >> 4)
}
