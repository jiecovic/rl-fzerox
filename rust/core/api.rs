// rust/core/api.rs
use std::ffi::c_void;
use std::path::Path;
use std::ptr;

use libloading::{Library, Symbol};
use libretro_sys::{
    API_VERSION, AudioSampleBatchFn, AudioSampleFn, EnvironmentFn, GameInfo, InputPollFn,
    InputStateFn, SystemAvInfo, SystemInfo, VideoRefreshFn,
};

use crate::core::error::CoreError;
use crate::core::info::CoreInfo;

type RetroSetEnvironmentFn = unsafe extern "C" fn(callback: EnvironmentFn);
type RetroSetVideoRefreshFn = unsafe extern "C" fn(callback: VideoRefreshFn);
type RetroSetAudioSampleFn = unsafe extern "C" fn(callback: AudioSampleFn);
type RetroSetAudioSampleBatchFn = unsafe extern "C" fn(callback: AudioSampleBatchFn);
type RetroSetInputPollFn = unsafe extern "C" fn(callback: InputPollFn);
type RetroSetInputStateFn = unsafe extern "C" fn(callback: InputStateFn);
type RetroInitFn = unsafe extern "C" fn();
type RetroDeinitFn = unsafe extern "C" fn();
type RetroApiVersionFn = unsafe extern "C" fn() -> u32;
type RetroGetSystemInfoFn = unsafe extern "C" fn(info: *mut SystemInfo);
type RetroGetSystemAvInfoFn = unsafe extern "C" fn(info: *mut SystemAvInfo);
type RetroSetControllerPortDeviceFn = unsafe extern "C" fn(port: u32, device: u32);
type RetroResetFn = unsafe extern "C" fn();
type RetroRunFn = unsafe extern "C" fn();
type RetroSerializeSizeFn = unsafe extern "C" fn() -> usize;
type RetroSerializeFn = unsafe extern "C" fn(data: *mut c_void, size: usize) -> bool;
type RetroUnserializeFn = unsafe extern "C" fn(data: *const c_void, size: usize) -> bool;
type RetroGetMemoryDataFn = unsafe extern "C" fn(id: u32) -> *mut c_void;
type RetroGetMemorySizeFn = unsafe extern "C" fn(id: u32) -> usize;
type RetroLoadGameFn = unsafe extern "C" fn(game: *const GameInfo) -> bool;
type RetroUnloadGameFn = unsafe extern "C" fn();

pub struct LoadedCore {
    _library: Library,
    symbols: Symbols,
    info: CoreInfo,
}

#[derive(Clone, Copy)]
struct Symbols {
    retro_set_environment: RetroSetEnvironmentFn,
    retro_set_video_refresh: RetroSetVideoRefreshFn,
    retro_set_audio_sample: RetroSetAudioSampleFn,
    retro_set_audio_sample_batch: RetroSetAudioSampleBatchFn,
    retro_set_input_poll: RetroSetInputPollFn,
    retro_set_input_state: RetroSetInputStateFn,
    retro_init: RetroInitFn,
    retro_deinit: RetroDeinitFn,
    retro_get_system_av_info: RetroGetSystemAvInfoFn,
    retro_set_controller_port_device: RetroSetControllerPortDeviceFn,
    retro_reset: RetroResetFn,
    retro_run: RetroRunFn,
    retro_serialize_size: RetroSerializeSizeFn,
    retro_serialize: RetroSerializeFn,
    retro_unserialize: RetroUnserializeFn,
    retro_get_memory_data: RetroGetMemoryDataFn,
    retro_get_memory_size: RetroGetMemorySizeFn,
    retro_load_game: RetroLoadGameFn,
    retro_unload_game: RetroUnloadGameFn,
}

impl LoadedCore {
    pub fn load(core_path: &Path) -> Result<Self, CoreError> {
        if !core_path.is_file() {
            return Err(CoreError::MissingCore(core_path.to_path_buf()));
        }

        let library = load_library(core_path)?;
        let retro_api_version: RetroApiVersionFn = load_symbol(&library, b"retro_api_version\0")?;
        let api_version = unsafe { retro_api_version() };
        if api_version != API_VERSION {
            return Err(CoreError::UnsupportedApiVersion {
                actual: api_version,
                expected: API_VERSION,
            });
        }

        let retro_get_system_info: RetroGetSystemInfoFn =
            load_symbol(&library, b"retro_get_system_info\0")?;
        let system_info = unsafe {
            let mut info = SystemInfo {
                library_name: ptr::null(),
                library_version: ptr::null(),
                valid_extensions: ptr::null(),
                need_fullpath: false,
                block_extract: false,
            };
            retro_get_system_info(&mut info as *mut SystemInfo);
            info
        };

        let info = CoreInfo::from_system_info(api_version, &system_info);
        let symbols = Symbols {
            retro_set_environment: load_symbol(&library, b"retro_set_environment\0")?,
            retro_set_video_refresh: load_symbol(&library, b"retro_set_video_refresh\0")?,
            retro_set_audio_sample: load_symbol(&library, b"retro_set_audio_sample\0")?,
            retro_set_audio_sample_batch: load_symbol(&library, b"retro_set_audio_sample_batch\0")?,
            retro_set_input_poll: load_symbol(&library, b"retro_set_input_poll\0")?,
            retro_set_input_state: load_symbol(&library, b"retro_set_input_state\0")?,
            retro_init: load_symbol(&library, b"retro_init\0")?,
            retro_deinit: load_symbol(&library, b"retro_deinit\0")?,
            retro_get_system_av_info: load_symbol(&library, b"retro_get_system_av_info\0")?,
            retro_set_controller_port_device: load_symbol(
                &library,
                b"retro_set_controller_port_device\0",
            )?,
            retro_reset: load_symbol(&library, b"retro_reset\0")?,
            retro_run: load_symbol(&library, b"retro_run\0")?,
            retro_serialize_size: load_symbol(&library, b"retro_serialize_size\0")?,
            retro_serialize: load_symbol(&library, b"retro_serialize\0")?,
            retro_unserialize: load_symbol(&library, b"retro_unserialize\0")?,
            retro_get_memory_data: load_symbol(&library, b"retro_get_memory_data\0")?,
            retro_get_memory_size: load_symbol(&library, b"retro_get_memory_size\0")?,
            retro_load_game: load_symbol(&library, b"retro_load_game\0")?,
            retro_unload_game: load_symbol(&library, b"retro_unload_game\0")?,
        };

        Ok(Self {
            _library: library,
            symbols,
            info,
        })
    }

    pub fn info(&self) -> &CoreInfo {
        &self.info
    }

    pub unsafe fn set_environment(&self, callback: EnvironmentFn) {
        unsafe { (self.symbols.retro_set_environment)(callback) };
    }

    pub unsafe fn set_video_refresh(&self, callback: VideoRefreshFn) {
        unsafe { (self.symbols.retro_set_video_refresh)(callback) };
    }

    pub unsafe fn set_audio_sample(&self, callback: AudioSampleFn) {
        unsafe { (self.symbols.retro_set_audio_sample)(callback) };
    }

    pub unsafe fn set_audio_sample_batch(&self, callback: AudioSampleBatchFn) {
        unsafe { (self.symbols.retro_set_audio_sample_batch)(callback) };
    }

    pub unsafe fn set_input_poll(&self, callback: InputPollFn) {
        unsafe { (self.symbols.retro_set_input_poll)(callback) };
    }

    pub unsafe fn set_input_state(&self, callback: InputStateFn) {
        unsafe { (self.symbols.retro_set_input_state)(callback) };
    }

    pub unsafe fn init(&self) {
        unsafe { (self.symbols.retro_init)() };
    }

    pub unsafe fn deinit(&self) {
        unsafe { (self.symbols.retro_deinit)() };
    }

    pub unsafe fn system_av_info(&self) -> SystemAvInfo {
        let mut info = SystemAvInfo {
            geometry: libretro_sys::GameGeometry {
                base_width: 0,
                base_height: 0,
                max_width: 0,
                max_height: 0,
                aspect_ratio: 0.0,
            },
            timing: libretro_sys::SystemTiming {
                fps: 0.0,
                sample_rate: 0.0,
            },
        };
        unsafe { (self.symbols.retro_get_system_av_info)(&mut info as *mut SystemAvInfo) };
        info
    }

    pub unsafe fn set_controller_port_device(&self, port: u32, device: u32) {
        unsafe { (self.symbols.retro_set_controller_port_device)(port, device) };
    }

    pub unsafe fn reset(&self) {
        unsafe { (self.symbols.retro_reset)() };
    }

    pub unsafe fn run(&self) {
        unsafe { (self.symbols.retro_run)() };
    }

    pub unsafe fn serialize_size(&self) -> usize {
        unsafe { (self.symbols.retro_serialize_size)() }
    }

    pub unsafe fn serialize(&self, data: *mut c_void, size: usize) -> bool {
        unsafe { (self.symbols.retro_serialize)(data, size) }
    }

    pub unsafe fn unserialize(&self, data: *const c_void, size: usize) -> bool {
        unsafe { (self.symbols.retro_unserialize)(data, size) }
    }

    pub unsafe fn memory_data(&self, memory_id: u32) -> *mut c_void {
        unsafe { (self.symbols.retro_get_memory_data)(memory_id) }
    }

    pub unsafe fn memory_size(&self, memory_id: u32) -> usize {
        unsafe { (self.symbols.retro_get_memory_size)(memory_id) }
    }

    pub unsafe fn load_game(&self, game: &GameInfo) -> bool {
        unsafe { (self.symbols.retro_load_game)(game as *const GameInfo) }
    }

    pub unsafe fn unload_game(&self) {
        unsafe { (self.symbols.retro_unload_game)() };
    }
}

fn load_library(core_path: &Path) -> Result<Library, CoreError> {
    unsafe { Library::new(core_path) }.map_err(|error| CoreError::LoadLibrary {
        path: core_path.to_path_buf(),
        message: error.to_string(),
    })
}

fn load_symbol<T: Copy>(library: &Library, symbol_name: &[u8]) -> Result<T, CoreError> {
    let symbol: Symbol<'_, T> =
        unsafe { library.get(symbol_name) }.map_err(|error| CoreError::MissingSymbol {
            symbol: printable_symbol(symbol_name),
            message: error.to_string(),
        })?;
    Ok(*symbol)
}

fn printable_symbol(symbol_name: &[u8]) -> String {
    symbol_name
        .strip_suffix(b"\0")
        .unwrap_or(symbol_name)
        .iter()
        .copied()
        .map(char::from)
        .collect()
}
