// Covers pixel conversion plus crop/aspect/resize behavior in the video path.
use std::ffi::c_void;

use super::{
    PixelLayout, RawVideoFrame, VideoCrop, VideoFrame, VideoResizeFilter, convert_argb1555,
    convert_argb8888, convert_rgb565, decode_frame, processed_frame, processed_frame_from_raw,
};

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

    let observation = processed_frame(
        &frame,
        4.0 / 3.0,
        160,
        120,
        true,
        VideoCrop::default(),
        VideoResizeFilter::Nearest,
    )
    .expect("observation should render");

    assert_eq!(observation.len(), 160 * 120 * 3);
    let left_red_total: usize = observation
        .chunks_exact(3)
        .enumerate()
        .filter(|(index, _)| index % 160 < 80)
        .map(|(_, pixel)| pixel[0] as usize)
        .sum();
    assert!(left_red_total > 160 * 120 * 100);
}

#[test]
fn observation_frame_from_raw_matches_decoded_path() {
    let mut bytes = vec![0_u8; 4 * 2 * 4];
    for row in 0..2 {
        for column in 0..4 {
            let offset = (row * 4 + column) * 4;
            bytes[offset] = (10 * column) as u8;
            bytes[offset + 1] = (20 * row) as u8;
            bytes[offset + 2] = (30 + column + row) as u8;
            bytes[offset + 3] = 0;
        }
    }
    let raw = RawVideoFrame {
        width: 4,
        height: 2,
        pitch: 16,
        pixel_layout: PixelLayout::Argb8888,
        bytes,
    };
    let decoded = decode_frame(&raw).expect("raw frame should decode");

    let from_raw = processed_frame_from_raw(
        &raw,
        4.0 / 3.0,
        3,
        2,
        true,
        VideoCrop::default(),
        VideoResizeFilter::Nearest,
    )
    .expect("raw observation should render");
    let from_decoded = processed_frame(
        &decoded,
        4.0 / 3.0,
        3,
        2,
        true,
        VideoCrop::default(),
        VideoResizeFilter::Nearest,
    )
    .expect("decoded observation");

    assert_eq!(from_raw, from_decoded);
}

#[test]
fn processed_frame_crops_before_resize() {
    let mut rgb = vec![0_u8; 10 * 6 * 3];
    for row in 1..5 {
        for column in 2..8 {
            let index = (row * 10 + column) * 3;
            rgb[index + 1] = 255;
        }
    }
    let frame = VideoFrame {
        width: 10,
        height: 6,
        rgb,
    };

    let observation = processed_frame(
        &frame,
        0.0,
        6,
        4,
        true,
        VideoCrop {
            top: 1,
            bottom: 1,
            left: 2,
            right: 2,
        },
        VideoResizeFilter::Nearest,
    )
    .expect("cropped observation should render");

    assert_eq!(observation.len(), 6 * 4 * 3);
    assert!(observation.chunks_exact(3).all(|pixel| pixel[1] == 255));
}
