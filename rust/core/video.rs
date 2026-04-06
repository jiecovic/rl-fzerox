// rust/core/video.rs
use std::ffi::c_void;

const HW_FRAME_BUFFER_VALID: usize = usize::MAX;

#[derive(Clone, Debug)]
pub struct VideoFrame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

#[derive(Clone, Copy)]
pub enum PixelLayout {
    Argb1555,
    Argb8888,
    Rgb565,
}

pub fn convert_frame(
    data: *const c_void,
    width: usize,
    height: usize,
    pitch: usize,
    pixel_layout: PixelLayout,
) -> Option<VideoFrame> {
    if data.is_null() || data as usize == HW_FRAME_BUFFER_VALID {
        return None;
    }

    let rgb = match pixel_layout {
        PixelLayout::Argb1555 => convert_argb1555(data, width, height, pitch),
        PixelLayout::Argb8888 => convert_argb8888(data, width, height, pitch),
        PixelLayout::Rgb565 => convert_rgb565(data, width, height, pitch),
    };
    Some(VideoFrame { width, height, rgb })
}

fn convert_argb8888(data: *const c_void, width: usize, height: usize, pitch: usize) -> Vec<u8> {
    let mut rgb = vec![0_u8; width * height * 3];
    for row in 0..height {
        let src =
            unsafe { std::slice::from_raw_parts((data as *const u8).add(row * pitch), width * 4) };
        let dst = &mut rgb[row * width * 3..(row + 1) * width * 3];
        for column in 0..width {
            let src_index = column * 4;
            let dst_index = column * 3;
            dst[dst_index] = src[src_index + 2];
            dst[dst_index + 1] = src[src_index + 1];
            dst[dst_index + 2] = src[src_index];
        }
    }
    rgb
}

fn convert_rgb565(data: *const c_void, width: usize, height: usize, pitch: usize) -> Vec<u8> {
    let mut rgb = vec![0_u8; width * height * 3];
    for row in 0..height {
        let src =
            unsafe { std::slice::from_raw_parts((data as *const u8).add(row * pitch), width * 2) };
        let dst = &mut rgb[row * width * 3..(row + 1) * width * 3];
        for column in 0..width {
            let src_index = column * 2;
            let pixel = u16::from_le_bytes([src[src_index], src[src_index + 1]]);
            let red = ((pixel >> 11) & 0x1f) as u8;
            let green = ((pixel >> 5) & 0x3f) as u8;
            let blue = (pixel & 0x1f) as u8;

            let dst_index = column * 3;
            dst[dst_index] = expand_5_to_8(red);
            dst[dst_index + 1] = expand_6_to_8(green);
            dst[dst_index + 2] = expand_5_to_8(blue);
        }
    }
    rgb
}

fn convert_argb1555(data: *const c_void, width: usize, height: usize, pitch: usize) -> Vec<u8> {
    let mut rgb = vec![0_u8; width * height * 3];
    for row in 0..height {
        let src =
            unsafe { std::slice::from_raw_parts((data as *const u8).add(row * pitch), width * 2) };
        let dst = &mut rgb[row * width * 3..(row + 1) * width * 3];
        for column in 0..width {
            let src_index = column * 2;
            let pixel = u16::from_le_bytes([src[src_index], src[src_index + 1]]);
            let red = ((pixel >> 10) & 0x1f) as u8;
            let green = ((pixel >> 5) & 0x1f) as u8;
            let blue = (pixel & 0x1f) as u8;

            let dst_index = column * 3;
            dst[dst_index] = expand_5_to_8(red);
            dst[dst_index + 1] = expand_5_to_8(green);
            dst[dst_index + 2] = expand_5_to_8(blue);
        }
    }
    rgb
}

fn expand_5_to_8(value: u8) -> u8 {
    (value << 3) | (value >> 2)
}

fn expand_6_to_8(value: u8) -> u8 {
    (value << 2) | (value >> 4)
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use super::{convert_argb1555, convert_argb8888, convert_rgb565};

    #[test]
    fn convert_argb8888_maps_bytes_to_rgb() {
        let pixels = [3_u8, 2, 1, 0, 6, 5, 4, 0];
        let rgb = convert_argb8888(pixels.as_ptr().cast::<c_void>(), 2, 1, 8);
        assert_eq!(rgb, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn convert_rgb565_maps_values_to_rgb() {
        let pixel = 0xF800_u16.to_le_bytes();
        let rgb = convert_rgb565(pixel.as_ptr().cast::<c_void>(), 1, 1, 2);
        assert_eq!(rgb, vec![255, 0, 0]);
    }

    #[test]
    fn convert_argb1555_maps_values_to_rgb() {
        let pixel = 0x7C00_u16.to_le_bytes();
        let rgb = convert_argb1555(pixel.as_ptr().cast::<c_void>(), 1, 1, 2);
        assert_eq!(rgb, vec![255, 0, 0]);
    }
}
