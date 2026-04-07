// rust/core/host/video.rs
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

pub fn display_size(width: usize, height: usize, aspect_ratio: f64) -> (usize, usize) {
    if aspect_ratio <= 0.0 {
        return (width, height);
    }

    let display_height = ((width as f64) / aspect_ratio).round() as usize;
    (width, display_height.max(1))
}

pub fn observation_frame(
    frame: &VideoFrame,
    aspect_ratio: f64,
    target_width: usize,
    target_height: usize,
    rgb: bool,
) -> Vec<u8> {
    let (display_width, display_height) = display_size(frame.width, frame.height, aspect_ratio);
    let aspect_corrected = if display_width != frame.width || display_height != frame.height {
        resize_rgb(
            &frame.rgb,
            frame.width,
            frame.height,
            display_width,
            display_height,
        )
    } else {
        frame.rgb.clone()
    };
    let resized = if display_width != target_width || display_height != target_height {
        resize_rgb(
            &aspect_corrected,
            display_width,
            display_height,
            target_width,
            target_height,
        )
    } else {
        aspect_corrected
    };

    if rgb {
        resized
    } else {
        to_grayscale(&resized, target_width, target_height)
    }
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

fn resize_rgb(
    rgb: &[u8],
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
) -> Vec<u8> {
    if input_width == output_width && input_height == output_height {
        return rgb.to_vec();
    }

    let x_index = axis_index_map(input_width, output_width);
    let y_index = axis_index_map(input_height, output_height);
    let mut resized = vec![0_u8; output_width * output_height * 3];
    for (output_y, &input_y) in y_index.iter().enumerate() {
        for (output_x, &input_x) in x_index.iter().enumerate() {
            let src_index = (input_y * input_width + input_x) * 3;
            let dst_index = (output_y * output_width + output_x) * 3;
            resized[dst_index..dst_index + 3].copy_from_slice(&rgb[src_index..src_index + 3]);
        }
    }
    resized
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

fn to_grayscale(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut gray = vec![0_u8; width * height];
    for index in 0..(width * height) {
        let src = index * 3;
        let value = (0.299_f32 * rgb[src] as f32)
            + (0.587_f32 * rgb[src + 1] as f32)
            + (0.114_f32 * rgb[src + 2] as f32);
        gray[index] = value.round() as u8;
    }
    gray
}

#[cfg(test)]
#[path = "tests/video_tests.rs"]
mod tests;
