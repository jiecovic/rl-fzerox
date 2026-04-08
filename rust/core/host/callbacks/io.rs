// rust/core/host/callbacks/io.rs
//! Extern callback entry points registered with the libretro core.

use std::ffi::c_void;

use libretro_sys::{DEVICE_ANALOG, DEVICE_JOYPAD};

use super::guard::with_state_mut;

pub fn input_device() -> u32 {
    DEVICE_JOYPAD
}

pub extern "C" fn environment_callback(cmd: u32, data: *mut c_void) -> bool {
    with_state_mut(|state| state.handle_environment(cmd, data)).unwrap_or(false)
}

pub extern "C" fn video_refresh_callback(
    data: *const c_void,
    width: u32,
    height: u32,
    pitch: usize,
) {
    let _ = with_state_mut(|state| {
        state.store_video_frame(data, width as usize, height as usize, pitch);
    });
}

pub extern "C" fn audio_sample_callback(_left: i16, _right: i16) {}

pub extern "C" fn audio_sample_batch_callback(_data: *const i16, frames: usize) -> usize {
    frames
}

pub extern "C" fn input_poll_callback() {}

pub extern "C" fn input_state_callback(_port: u32, _device: u32, _index: u32, _id: u32) -> i16 {
    if _port != 0 {
        return 0;
    }

    match _device {
        DEVICE_JOYPAD if _index == 0 => {
            with_state_mut(|state| state.joypad_state(_id)).unwrap_or(0)
        }
        DEVICE_ANALOG => with_state_mut(|state| state.analog_state(_index, _id)).unwrap_or(0),
        _ => 0,
    }
}
