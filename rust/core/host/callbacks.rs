// rust/core/host/callbacks.rs
use std::cell::Cell;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

use bitflags::bitflags;
use libretro_sys::{
    DEVICE_ANALOG, DEVICE_JOYPAD, ENVIRONMENT_GET_CAN_DUPE,
    ENVIRONMENT_GET_CURRENT_SOFTWARE_FRAMEBUFFER, ENVIRONMENT_GET_LIBRETRO_PATH,
    ENVIRONMENT_GET_LOG_INTERFACE, ENVIRONMENT_GET_OVERSCAN, ENVIRONMENT_GET_SAVE_DIRECTORY,
    ENVIRONMENT_GET_SYSTEM_DIRECTORY, ENVIRONMENT_GET_VARIABLE, ENVIRONMENT_GET_VARIABLE_UPDATE,
    ENVIRONMENT_SET_CONTROLLER_INFO, ENVIRONMENT_SET_GEOMETRY, ENVIRONMENT_SET_HW_RENDER,
    ENVIRONMENT_SET_INPUT_DESCRIPTORS, ENVIRONMENT_SET_MEMORY_MAPS, ENVIRONMENT_SET_PIXEL_FORMAT,
    ENVIRONMENT_SET_PROC_ADDRESS_CALLBACK, ENVIRONMENT_SET_SUBSYSTEM_INFO,
    ENVIRONMENT_SET_VARIABLES, GameGeometry, LogCallback, LogLevel, PixelFormat, Variable,
};

use crate::core::error::CoreError;
use crate::core::input::ControllerState;
use crate::core::options::{default_option_value, override_option};
use crate::core::video::{
    PixelLayout, RawVideoFrame, VideoFrame, capture_raw_frame, decode_frame,
    observation_frame_from_raw,
};

const ENVIRONMENT_EXPERIMENTAL: u32 = 1 << 31;

// These frontend environment command ids come from libretro.h. The
// "experimental" ones are tagged by OR-ing bit 31, matching the libretro
// header convention used by libretro-sys.
#[repr(u32)]
enum EnvironmentCmd {
    GetAudioVideoEnable = 47 | ENVIRONMENT_EXPERIMENTAL,
    GetFastForwarding = 49 | ENVIRONMENT_EXPERIMENTAL,
    GetTargetRefreshRate = 50 | ENVIRONMENT_EXPERIMENTAL,
    GetInputBitmasks = 51 | ENVIRONMENT_EXPERIMENTAL,
    GetCoreOptionsVersion = 52 | ENVIRONMENT_EXPERIMENTAL,
    SetCoreOptionsDisplay = 55 | ENVIRONMENT_EXPERIMENTAL,
    SetCoreOptionsUpdateDisplayCallback = 69,
}

bitflags! {
    // Flags returned for ENVIRONMENT_GET_AUDIO_VIDEO_ENABLE from libretro.h.
    struct AudioVideoEnable: u32 {
        const VIDEO = 1 << 0;
        const AUDIO = 1 << 1;
    }
}

impl EnvironmentCmd {
    const fn code(self) -> u32 {
        self as u32
    }
}

thread_local! {
    static ACTIVE_STATE: Cell<*mut CallbackState> = const { Cell::new(ptr::null_mut()) };
}

pub struct CallbackState {
    core_path: CString,
    system_dir: CString,
    save_dir: CString,
    rdp_plugin: CString,
    variables: BTreeMap<String, CString>,
    controller_state: ControllerState,
    frame: Option<VideoFrame>,
    raw_frame: Option<RawVideoFrame>,
    capture_video: bool,
    frame_serial: u64,
    pixel_format: PixelLayout,
    log_callback: LogCallback,
    geometry: Option<(usize, usize)>,
}

pub struct CallbackGuard {
    previous: *mut CallbackState,
}

impl CallbackState {
    pub fn new(
        core_path: &Path,
        runtime_dir: Option<&Path>,
        renderer: &str,
    ) -> Result<Self, CoreError> {
        let system_dir = match runtime_dir {
            Some(runtime_dir) => runtime_dir.to_path_buf(),
            None => runtime_root_for_core(core_path)?,
        };
        std::fs::create_dir_all(&system_dir).map_err(|error| CoreError::CreateDirectory {
            path: system_dir.clone(),
            message: error.to_string(),
        })?;

        Ok(Self {
            core_path: path_to_c_string(core_path)?,
            system_dir: path_to_c_string(&system_dir)?,
            save_dir: path_to_c_string(&system_dir)?,
            rdp_plugin: CString::new(renderer).map_err(|_| CoreError::InvalidPath {
                path: core_path.to_path_buf(),
            })?,
            variables: BTreeMap::new(),
            controller_state: ControllerState::default(),
            frame: None,
            raw_frame: None,
            capture_video: true,
            frame_serial: 0,
            pixel_format: PixelLayout::Argb1555,
            log_callback: LogCallback { log: log_callback },
            geometry: None,
        })
    }

    pub fn frame(&mut self) -> Option<&VideoFrame> {
        if self.frame.is_none() {
            self.frame = self.raw_frame.as_ref().and_then(decode_frame);
        }
        self.frame.as_ref()
    }

    pub fn frame_serial(&self) -> u64 {
        self.frame_serial
    }

    pub fn geometry(&self) -> Option<(usize, usize)> {
        self.geometry
    }

    pub fn has_frame(&self) -> bool {
        self.frame.is_some() || self.raw_frame.is_some()
    }

    pub fn set_frame(&mut self, frame: VideoFrame) {
        self.geometry = Some((frame.width, frame.height));
        self.raw_frame = None;
        self.frame = Some(frame);
        self.frame_serial += 1;
    }

    pub fn set_capture_video(&mut self, capture_video: bool) {
        self.capture_video = capture_video;
    }

    pub fn observation_frame(
        &mut self,
        aspect_ratio: f64,
        target_width: usize,
        target_height: usize,
        rgb: bool,
    ) -> Option<Vec<u8>> {
        if let Some(raw_frame) = self.raw_frame.as_ref() {
            return observation_frame_from_raw(
                raw_frame,
                aspect_ratio,
                target_width,
                target_height,
                rgb,
            );
        }

        let frame = self.frame()?;
        Some(crate::core::video::observation_frame(
            frame,
            aspect_ratio,
            target_width,
            target_height,
            rgb,
        ))
    }

    pub fn set_controller_state(&mut self, controller_state: ControllerState) {
        self.controller_state = controller_state;
    }

    fn joypad_state(&self, button_id: u32) -> i16 {
        self.controller_state.joypad_state(button_id)
    }

    fn analog_state(&self, index: u32, id: u32) -> i16 {
        self.controller_state.analog_state(index, id)
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
            cmd if cmd == EnvironmentCmd::GetInputBitmasks.code() => write_ptr(data, true),
            cmd if cmd == EnvironmentCmd::GetFastForwarding.code() => write_ptr(data, false),
            cmd if cmd == EnvironmentCmd::GetTargetRefreshRate.code() => write_ptr(data, 60.0_f32),
            cmd if cmd == EnvironmentCmd::GetAudioVideoEnable.code() => {
                write_ptr(data, AudioVideoEnable::all().bits())
            }
            cmd if cmd == EnvironmentCmd::GetCoreOptionsVersion.code() => write_ptr(data, 0_u32),
            cmd if is_passthrough_environment_cmd(cmd) => true,
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
            let value = override_option(&key, &default_value, self.rdp_plugin.as_c_str());
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
        self.geometry = Some((width, height));
        if !self.capture_video {
            return;
        }

        if let Some(raw_frame) = capture_raw_frame(data, width, height, pitch, self.pixel_format) {
            self.raw_frame = Some(raw_frame);
            self.frame = None;
            self.frame_serial += 1;
        }
    }
}

fn is_passthrough_environment_cmd(cmd: u32) -> bool {
    matches!(
        cmd,
        ENVIRONMENT_SET_INPUT_DESCRIPTORS
            | ENVIRONMENT_SET_CONTROLLER_INFO
            | ENVIRONMENT_SET_MEMORY_MAPS
            | ENVIRONMENT_SET_PROC_ADDRESS_CALLBACK
            | ENVIRONMENT_SET_SUBSYSTEM_INFO
    ) || cmd == EnvironmentCmd::SetCoreOptionsDisplay.code()
        || cmd == EnvironmentCmd::SetCoreOptionsUpdateDisplayCallback.code()
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

fn runtime_root_for_core(core_path: &Path) -> Result<PathBuf, CoreError> {
    let core_dir = core_path
        .parent()
        .ok_or_else(|| CoreError::InvalidPath {
            path: core_path.to_path_buf(),
        })?
        .to_path_buf();

    if core_dir
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value == "libretro")
        && let Some(runtime_root) = core_dir.parent()
    {
        return Ok(runtime_root.to_path_buf());
    }

    Ok(core_dir)
}

#[cfg(test)]
#[path = "tests/callbacks_tests.rs"]
mod tests;
