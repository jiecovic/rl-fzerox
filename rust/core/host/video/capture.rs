// rust/core/host/video/capture.rs
use std::ffi::c_void;
use std::slice;

use crate::core::video::{PixelLayout, RawVideoFrame};

const HW_FRAME_BUFFER_VALID: usize = usize::MAX;

pub fn capture_raw_frame(
    data: *const c_void,
    width: usize,
    height: usize,
    pitch: usize,
    pixel_layout: PixelLayout,
) -> Option<RawVideoFrame> {
    if data.is_null() || data as usize == HW_FRAME_BUFFER_VALID {
        return None;
    }

    let bytes = unsafe { slice::from_raw_parts(data.cast::<u8>(), pitch.checked_mul(height)?) };
    Some(RawVideoFrame {
        width,
        height,
        pitch,
        pixel_layout,
        bytes: bytes.to_vec(),
    })
}

pub fn capture_raw_frame_into(
    frame: &mut RawVideoFrame,
    data: *const c_void,
    width: usize,
    height: usize,
    pitch: usize,
    pixel_layout: PixelLayout,
) -> bool {
    if data.is_null() || data as usize == HW_FRAME_BUFFER_VALID {
        return false;
    }
    let Some(byte_len) = pitch.checked_mul(height) else {
        return false;
    };
    let bytes = unsafe { slice::from_raw_parts(data.cast::<u8>(), byte_len) };
    frame.width = width;
    frame.height = height;
    frame.pitch = pitch;
    frame.pixel_layout = pixel_layout;
    frame.bytes.resize(byte_len, 0);
    frame.bytes.copy_from_slice(bytes);
    true
}
