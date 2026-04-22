// rust/core/host/callbacks/state.rs
//! Callback-owned mutable state shared by libretro environment/input/video
//! callbacks.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::Path;

use libretro_sys::LogCallback;

use super::stack::{StackedObservationBuffer, StackedObservationKey};
use super::util::{log_callback, path_to_c_string, runtime_root_for_core};
use crate::core::error::CoreError;
use crate::core::host::hardware::HardwareRenderContext;
use crate::core::input::ControllerState;
use crate::core::minimap::MinimapLayerRenderer;
use crate::core::video::{
    PixelLayout, ProcessedFramePlan, ProcessedFramePlanKey, RawVideoFrame, VideoCrop, VideoFrame,
    VideoResizeFilter, decode_frame,
};

mod environment;
mod hardware;
mod render;
mod video;

/// State that lives on the frontend side of the libretro boundary.
///
/// It caches the latest frame, current controller state, option values, and
/// reusable scratch buffers for observation/display rendering.
pub struct CallbackState {
    pub(super) core_path: CString,
    pub(super) system_dir: CString,
    pub(super) save_dir: CString,
    pub(super) rdp_plugin: CString,
    pub(super) variables: BTreeMap<String, CString>,
    pub(super) controller_state: ControllerState,
    pub(super) frame: Option<VideoFrame>,
    pub(super) raw_frame: Option<RawVideoFrame>,
    pub(super) observation_buffer: Vec<u8>,
    pub(super) display_buffer: Vec<u8>,
    pub(super) minimap_buffer: Vec<u8>,
    pub(super) minimap_renderer: MinimapLayerRenderer,
    pub(super) stacked_observation_buffers: Vec<StackedObservationCacheEntry>,
    pub(super) render_plans: Vec<RenderPlanCacheEntry>,
    pub(super) capture_video: bool,
    pub(super) frame_serial: u64,
    pub(super) pixel_format: PixelLayout,
    pub(super) hardware_render: Option<HardwareRenderContext>,
    pub(super) hardware_render_error: Option<String>,
    pub(super) log_callback: LogCallback,
    pub(super) geometry: Option<(usize, usize)>,
}

pub(super) struct StackedObservationCacheEntry {
    pub(super) key: StackedObservationKey,
    pub(super) buffer: StackedObservationBuffer,
}

pub(super) struct RenderPlanCacheEntry {
    pub(super) key: ProcessedFramePlanKey,
    pub(super) plan: ProcessedFramePlan,
}

#[derive(Clone, Copy)]
pub(super) struct RenderRequest {
    pub(super) source_width: usize,
    pub(super) source_height: usize,
    pub(super) aspect_ratio: f64,
    pub(super) target_width: usize,
    pub(super) target_height: usize,
    pub(super) rgb: bool,
    pub(super) crop: VideoCrop,
    pub(super) resize_filter: VideoResizeFilter,
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
            observation_buffer: Vec::new(),
            display_buffer: Vec::new(),
            minimap_buffer: Vec::new(),
            minimap_renderer: MinimapLayerRenderer::default(),
            stacked_observation_buffers: Vec::new(),
            render_plans: Vec::new(),
            capture_video: true,
            frame_serial: 0,
            pixel_format: PixelLayout::Argb1555,
            hardware_render: None,
            hardware_render_error: None,
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
        self.clear_stacked_observation_buffers();
        self.minimap_renderer.clear();
    }

    pub fn set_capture_video(&mut self, capture_video: bool) {
        self.capture_video = capture_video;
    }

    pub fn set_controller_state(&mut self, controller_state: ControllerState) {
        self.controller_state = controller_state;
    }

    pub(super) fn joypad_state(&self, button_id: u32) -> i16 {
        self.controller_state.joypad_state(button_id)
    }

    pub(super) fn analog_state(&self, index: u32, id: u32) -> i16 {
        self.controller_state.analog_state(index, id)
    }
}
