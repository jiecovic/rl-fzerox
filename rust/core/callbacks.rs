// rust/core/callbacks.rs
use std::cell::Cell;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_void};
use std::path::Path;
use std::ptr;

use libretro_sys::{
    DEVICE_JOYPAD, ENVIRONMENT_GET_CAN_DUPE, ENVIRONMENT_GET_CURRENT_SOFTWARE_FRAMEBUFFER,
    ENVIRONMENT_GET_LIBRETRO_PATH, ENVIRONMENT_GET_LOG_INTERFACE, ENVIRONMENT_GET_OVERSCAN,
    ENVIRONMENT_GET_SAVE_DIRECTORY, ENVIRONMENT_GET_SYSTEM_DIRECTORY, ENVIRONMENT_GET_VARIABLE,
    ENVIRONMENT_GET_VARIABLE_UPDATE, ENVIRONMENT_SET_CONTROLLER_INFO, ENVIRONMENT_SET_GEOMETRY,
    ENVIRONMENT_SET_HW_RENDER, ENVIRONMENT_SET_INPUT_DESCRIPTORS, ENVIRONMENT_SET_MEMORY_MAPS,
    ENVIRONMENT_SET_PIXEL_FORMAT, ENVIRONMENT_SET_PROC_ADDRESS_CALLBACK,
    ENVIRONMENT_SET_SUBSYSTEM_INFO, ENVIRONMENT_SET_VARIABLES, GameGeometry, LogCallback, LogLevel,
    PixelFormat, Variable,
};

use crate::core::error::CoreError;

const ENVIRONMENT_GET_FASTFORWARDING: u32 = 49 | (1 << 31);
const ENVIRONMENT_GET_TARGET_REFRESH_RATE: u32 = 50 | (1 << 31);
const ENVIRONMENT_GET_INPUT_BITMASKS: u32 = 51 | (1 << 31);
const ENVIRONMENT_GET_AUDIO_VIDEO_ENABLE: u32 = 47 | (1 << 31);
const ENVIRONMENT_GET_CORE_OPTIONS_VERSION: u32 = 52 | (1 << 31);
const ENVIRONMENT_SET_CORE_OPTIONS_DISPLAY: u32 = 55 | (1 << 31);
const ENVIRONMENT_SET_CORE_OPTIONS_UPDATE_DISPLAY_CALLBACK: u32 = 69;
const AUDIO_VIDEO_ENABLE_VIDEO: u32 = 1 << 0;
const AUDIO_VIDEO_ENABLE_AUDIO: u32 = 1 << 1;
const HW_FRAME_BUFFER_VALID: usize = usize::MAX;

thread_local! {
    static ACTIVE_STATE: Cell<*mut CallbackState> = const { Cell::new(ptr::null_mut()) };
}

#[derive(Clone, Debug)]
pub struct VideoFrame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

pub struct CallbackState {
    core_path: CString,
    system_dir: CString,
    save_dir: CString,
    variables: BTreeMap<String, CString>,
    frame: Option<VideoFrame>,
    frame_serial: u64,
    pixel_format: PixelLayout,
    log_callback: LogCallback,
    geometry: Option<(usize, usize)>,
}

pub struct CallbackGuard {
    previous: *mut CallbackState,
}

#[derive(Clone, Copy)]
enum PixelLayout {
    Argb1555,
    Argb8888,
    Rgb565,
}

impl CallbackState {
    pub fn new(core_path: &Path, rom_path: &Path) -> Result<Self, CoreError> {
        let system_dir = rom_path
            .parent()
            .ok_or_else(|| CoreError::InvalidPath {
                path: rom_path.to_path_buf(),
            })?
            .to_path_buf();

        Ok(Self {
            core_path: path_to_c_string(core_path)?,
            system_dir: path_to_c_string(&system_dir)?,
            save_dir: path_to_c_string(&system_dir)?,
            variables: BTreeMap::new(),
            frame: None,
            frame_serial: 0,
            pixel_format: PixelLayout::Argb1555,
            log_callback: LogCallback { log: log_callback },
            geometry: None,
        })
    }

    pub fn frame(&self) -> Option<&VideoFrame> {
        self.frame.as_ref()
    }

    pub fn frame_serial(&self) -> u64 {
        self.frame_serial
    }

    pub fn geometry(&self) -> Option<(usize, usize)> {
        self.geometry
    }

    fn handle_environment(&mut self, cmd: u32, data: *mut c_void) -> bool {
        match cmd {
            ENVIRONMENT_SET_VARIABLES => self.set_variables(data),
            ENVIRONMENT_GET_VARIABLE => self.get_variable(data),
            ENVIRONMENT_GET_VARIABLE_UPDATE => write_ptr(data, false),
            ENVIRONMENT_GET_SYSTEM_DIRECTORY => write_ptr(data, self.system_dir.as_ptr()),
            ENVIRONMENT_GET_SAVE_DIRECTORY => write_ptr(data, self.save_dir.as_ptr()),
            ENVIRONMENT_GET_LIBRETRO_PATH => write_ptr(data, self.core_path.as_ptr()),
            ENVIRONMENT_GET_LOG_INTERFACE => write_ptr(data, self.log_callback.clone()),
            ENVIRONMENT_SET_PIXEL_FORMAT => self.set_pixel_format(data),
            ENVIRONMENT_GET_OVERSCAN => write_ptr(data, false),
            ENVIRONMENT_GET_CAN_DUPE => write_ptr(data, true),
            ENVIRONMENT_GET_INPUT_BITMASKS => write_ptr(data, true),
            ENVIRONMENT_GET_FASTFORWARDING => write_ptr(data, false),
            ENVIRONMENT_GET_TARGET_REFRESH_RATE => write_ptr(data, 60.0_f32),
            ENVIRONMENT_GET_AUDIO_VIDEO_ENABLE => {
                write_ptr(data, AUDIO_VIDEO_ENABLE_VIDEO | AUDIO_VIDEO_ENABLE_AUDIO)
            }
            ENVIRONMENT_GET_CORE_OPTIONS_VERSION => write_ptr(data, 0_u32),
            ENVIRONMENT_SET_CORE_OPTIONS_DISPLAY
            | ENVIRONMENT_SET_CORE_OPTIONS_UPDATE_DISPLAY_CALLBACK
            | ENVIRONMENT_SET_INPUT_DESCRIPTORS
            | ENVIRONMENT_SET_CONTROLLER_INFO
            | ENVIRONMENT_SET_MEMORY_MAPS
            | ENVIRONMENT_SET_PROC_ADDRESS_CALLBACK
            | ENVIRONMENT_SET_SUBSYSTEM_INFO => true,
            ENVIRONMENT_SET_GEOMETRY => {
                self.set_geometry(data);
                true
            }
            ENVIRONMENT_GET_CURRENT_SOFTWARE_FRAMEBUFFER => false,
            ENVIRONMENT_SET_HW_RENDER => false,
            _ => false,
        }
    }

    fn set_variables(&mut self, data: *mut c_void) -> bool {
        let entries = data.cast::<Variable>();
        if entries.is_null() {
            return false;
        }

        let mut index = 0_usize;
        loop {
            let entry = unsafe { &*entries.add(index) };
            if entry.key.is_null() {
                break;
            }

            let key = c_string(entry.key);
            let spec = c_string(entry.value);
            let default_value = default_option_value(&spec);
            let value = override_option(&key, &default_value);
            let c_value = match CString::new(value) {
                Ok(value) => value,
                Err(_) => return false,
            };
            self.variables.insert(key, c_value);
            index += 1;
        }
        true
    }

    fn get_variable(&mut self, data: *mut c_void) -> bool {
        let variable = data.cast::<Variable>();
        if variable.is_null() {
            return false;
        }

        let key_ptr = unsafe { (*variable).key };
        if key_ptr.is_null() {
            return false;
        }

        let key = c_string(key_ptr);
        let value = match self.variables.get(&key) {
            Some(value) => value.as_ptr(),
            None => ptr::null(),
        };
        unsafe {
            (*variable).value = value;
        }
        !value.is_null()
    }

    fn set_pixel_format(&mut self, data: *mut c_void) -> bool {
        let Some(format) = read_u32(data) else {
            return false;
        };
        match format {
            value if value == PixelFormat::ARGB1555 as u32 => {
                self.pixel_format = PixelLayout::Argb1555;
                true
            }
            value if value == PixelFormat::ARGB8888 as u32 => {
                self.pixel_format = PixelLayout::Argb8888;
                true
            }
            value if value == PixelFormat::RGB565 as u32 => {
                self.pixel_format = PixelLayout::Rgb565;
                true
            }
            _ => false,
        }
    }

    fn set_geometry(&mut self, data: *mut c_void) {
        if data.is_null() {
            return;
        }
        let geometry = unsafe { &*data.cast::<GameGeometry>() };
        self.geometry = Some((geometry.base_width as usize, geometry.base_height as usize));
    }

    fn store_video_frame(
        &mut self,
        data: *const c_void,
        width: usize,
        height: usize,
        pitch: usize,
    ) {
        if data.is_null() || data as usize == HW_FRAME_BUFFER_VALID {
            return;
        }

        let rgb = match self.pixel_format {
            PixelLayout::Argb1555 => convert_argb1555(data, width, height, pitch),
            PixelLayout::Argb8888 => convert_argb8888(data, width, height, pitch),
            PixelLayout::Rgb565 => convert_rgb565(data, width, height, pitch),
        };
        self.frame = Some(VideoFrame { width, height, rgb });
        self.frame_serial += 1;
        self.geometry = Some((width, height));
    }
}

impl CallbackGuard {
    pub fn activate(state: *mut CallbackState) -> Self {
        let previous = ACTIVE_STATE.with(|slot| {
            let previous = slot.get();
            slot.set(state);
            previous
        });
        Self { previous }
    }
}

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        ACTIVE_STATE.with(|slot| slot.set(self.previous));
    }
}

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
    0
}

unsafe extern "C" fn log_callback(_level: LogLevel, _fmt: *const i8) {}

fn with_state_mut<R>(callback: impl FnOnce(&mut CallbackState) -> R) -> Option<R> {
    ACTIVE_STATE.with(|slot| {
        let state = slot.get();
        if state.is_null() {
            return None;
        }

        // SAFETY: The active callback state pointer is set by CallbackGuard for the
        // duration of a libretro call into this thread. The pointed-to state lives in
        // the owning Host and outlives the callback invocation.
        Some(callback(unsafe { &mut *state }))
    })
}

fn path_to_c_string(path: &Path) -> Result<CString, CoreError> {
    CString::new(path.to_string_lossy().as_bytes()).map_err(|_| CoreError::InvalidPath {
        path: path.to_path_buf(),
    })
}

fn c_string(value: *const i8) -> String {
    if value.is_null() {
        return String::new();
    }

    unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned()
}

fn default_option_value(spec: &str) -> String {
    let Some((_, values)) = spec.split_once("; ") else {
        return spec.to_owned();
    };
    values.split('|').next().unwrap_or_default().to_owned()
}

fn override_option(key: &str, default_value: &str) -> String {
    match key {
        "mupen64plus-rdp-plugin" => "angrylion".to_owned(),
        _ => default_value.to_owned(),
    }
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

fn read_u32(data: *mut c_void) -> Option<u32> {
    if data.is_null() {
        return None;
    }

    Some(unsafe { ptr::read(data.cast::<u32>()) })
}

fn write_ptr<T>(data: *mut c_void, value: T) -> bool {
    if data.is_null() {
        return false;
    }

    unsafe {
        ptr::write(data.cast::<T>(), value);
    }
    true
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use super::{convert_argb1555, convert_argb8888, convert_rgb565, default_option_value};

    #[test]
    fn default_option_value_returns_first_choice() {
        assert_eq!(
            default_option_value("Renderer; angrylion|gliden64"),
            "angrylion"
        );
    }

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
