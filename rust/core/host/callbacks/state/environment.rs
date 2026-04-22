// rust/core/host/callbacks/state/environment.rs
//! CallbackState handling for libretro environment commands.

use std::ffi::{CString, c_void};
use std::ptr;

use libretro_sys::{GameGeometry, PixelFormat, Variable};

use super::super::environment::{AudioVideoEnable, EnvironmentCommand};
use super::super::util::{c_string, read_u32, write_ptr};
use super::CallbackState;
use crate::core::options::{default_option_value, override_option};
use crate::core::video::PixelLayout;

impl CallbackState {
    pub(in crate::core::host::callbacks) fn handle_environment(
        &mut self,
        cmd: u32,
        data: *mut c_void,
    ) -> bool {
        match EnvironmentCommand::from_raw(cmd) {
            EnvironmentCommand::SetVariables => self.set_variables(data),
            EnvironmentCommand::GetVariable => self.get_variable(data),
            EnvironmentCommand::GetVariableUpdate => write_ptr(data, false),
            EnvironmentCommand::GetSystemDirectory => write_ptr(data, self.system_dir.as_ptr()),
            EnvironmentCommand::GetSaveDirectory => write_ptr(data, self.save_dir.as_ptr()),
            EnvironmentCommand::GetLibretroPath => write_ptr(data, self.core_path.as_ptr()),
            EnvironmentCommand::GetLogInterface => write_ptr(data, self.log_callback.clone()),
            EnvironmentCommand::SetPixelFormat => self.set_pixel_format(data),
            EnvironmentCommand::GetOverscan => write_ptr(data, false),
            EnvironmentCommand::GetCanDupe => write_ptr(data, true),
            EnvironmentCommand::GetInputBitmasks => write_ptr(data, true),
            EnvironmentCommand::GetFastForwarding => write_ptr(data, false),
            EnvironmentCommand::GetTargetRefreshRate => write_ptr(data, 60.0_f32),
            EnvironmentCommand::GetAudioVideoEnable => {
                write_ptr(data, AudioVideoEnable::all().bits())
            }
            EnvironmentCommand::GetCoreOptionsVersion => write_ptr(data, 0_u32),
            EnvironmentCommand::SetGeometry => {
                self.set_geometry(data);
                true
            }
            EnvironmentCommand::GetCurrentSoftwareFramebuffer => false,
            EnvironmentCommand::SetHardwareRender => self.set_hardware_render(data),
            EnvironmentCommand::Passthrough => true,
            EnvironmentCommand::Unknown => false,
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
}
