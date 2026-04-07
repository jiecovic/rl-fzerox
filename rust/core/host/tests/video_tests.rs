use std::ffi::c_void;

use super::{VideoFrame, convert_argb1555, convert_argb8888, convert_rgb565, observation_frame};

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

#[test]
fn observation_frame_aspect_corrects_then_downscales() {
    let mut rgb = vec![0_u8; 640 * 240 * 3];
    for row in 0..240 {
        for column in 0..320 {
            let index = (row * 640 + column) * 3;
            rgb[index] = 255;
        }
    }
    let frame = VideoFrame {
        width: 640,
        height: 240,
        rgb,
    };

    let observation = observation_frame(&frame, 4.0 / 3.0, 160, 120, true);

    assert_eq!(observation.len(), 160 * 120 * 3);
    let left_red_total: usize = observation
        .chunks_exact(3)
        .enumerate()
        .filter(|(index, _)| index % 160 < 80)
        .map(|(_, pixel)| pixel[0] as usize)
        .sum();
    assert!(left_red_total > 160 * 120 * 100);
}
