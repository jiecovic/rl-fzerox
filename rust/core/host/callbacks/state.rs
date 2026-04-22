// rust/core/host/callbacks/state.rs
//! Callback-owned mutable state shared by libretro environment/input/video
//! callbacks.

use std::collections::BTreeMap;
use std::ffi::{CString, c_void};
use std::path::Path;
use std::ptr;

use libretro_sys::{
    ENVIRONMENT_GET_CAN_DUPE, ENVIRONMENT_GET_CURRENT_SOFTWARE_FRAMEBUFFER,
    ENVIRONMENT_GET_LIBRETRO_PATH, ENVIRONMENT_GET_LOG_INTERFACE, ENVIRONMENT_GET_OVERSCAN,
    ENVIRONMENT_GET_SAVE_DIRECTORY, ENVIRONMENT_GET_SYSTEM_DIRECTORY, ENVIRONMENT_GET_VARIABLE,
    ENVIRONMENT_GET_VARIABLE_UPDATE, ENVIRONMENT_SET_GEOMETRY, ENVIRONMENT_SET_HW_RENDER,
    ENVIRONMENT_SET_PIXEL_FORMAT, ENVIRONMENT_SET_VARIABLES, GameGeometry, LogCallback,
    PixelFormat, Variable,
};

use super::environment::{AudioVideoEnable, EnvironmentCmd, is_passthrough_environment_cmd};
use super::stack::{StackedObservationBuffer, StackedObservationKey, StackedObservationRequest};
use super::util::{
    c_string, log_callback, path_to_c_string, read_u32, runtime_root_for_core, write_ptr,
};
use crate::core::error::CoreError;
use crate::core::host::hardware::HardwareRenderContext;
use crate::core::input::ControllerState;
use crate::core::minimap::{MinimapLayerRenderer, MinimapLayerRequest};
use crate::core::options::{default_option_value, override_option};
use crate::core::video::{
    PixelLayout, ProcessedFramePlan, ProcessedFramePlanKey, RawVideoFrame, VideoCrop, VideoFrame,
    VideoResizeFilter, build_processed_frame_plan, capture_raw_frame, capture_raw_frame_into,
    decode_frame, processed_frame, processed_frame_from_raw_into,
};

/// State that lives on the frontend side of the libretro boundary.
///
/// It caches the latest frame, current controller state, option values, and
/// reusable scratch buffers for observation/display rendering.
pub struct CallbackState {
    core_path: CString,
    system_dir: CString,
    save_dir: CString,
    rdp_plugin: CString,
    variables: BTreeMap<String, CString>,
    controller_state: ControllerState,
    frame: Option<VideoFrame>,
    raw_frame: Option<RawVideoFrame>,
    observation_buffer: Vec<u8>,
    display_buffer: Vec<u8>,
    minimap_buffer: Vec<u8>,
    minimap_renderer: MinimapLayerRenderer,
    stacked_observation_buffers: BTreeMap<StackedObservationKey, StackedObservationBuffer>,
    render_plans: BTreeMap<ProcessedFramePlanKey, ProcessedFramePlan>,
    capture_video: bool,
    frame_serial: u64,
    pixel_format: PixelLayout,
    hardware_render: Option<HardwareRenderContext>,
    hardware_render_error: Option<String>,
    log_callback: LogCallback,
    geometry: Option<(usize, usize)>,
}

#[derive(Clone, Copy)]
struct RenderRequest {
    source_width: usize,
    source_height: usize,
    aspect_ratio: f64,
    target_width: usize,
    target_height: usize,
    rgb: bool,
    crop: VideoCrop,
    resize_filter: VideoResizeFilter,
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
            stacked_observation_buffers: BTreeMap::new(),
            render_plans: BTreeMap::new(),
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

    pub fn observation_frame(
        &mut self,
        aspect_ratio: f64,
        target_width: usize,
        target_height: usize,
        rgb: bool,
        crop: VideoCrop,
        resize_filter: VideoResizeFilter,
    ) -> Result<&[u8], CoreError> {
        // Prefer the raw-frame fast path so we can render directly into the
        // reusable output buffers without fully decoding to an intermediate
        // RGB frame first.
        if let Some((source_width, source_height)) = self
            .raw_frame
            .as_ref()
            .map(|raw_frame| (raw_frame.width, raw_frame.height))
        {
            let request = RenderRequest {
                source_width,
                source_height,
                aspect_ratio,
                target_width,
                target_height,
                rgb,
                crop,
                resize_filter,
            };
            let plan = self.render_plan(request)?.clone();
            let use_display_buffer = self.use_display_buffer(target_width, target_height);
            let raw_frame = self.raw_frame.as_ref().ok_or(CoreError::NoFrameAvailable)?;
            let target_buffer = if use_display_buffer {
                &mut self.display_buffer
            } else {
                &mut self.observation_buffer
            };
            processed_frame_from_raw_into(raw_frame, &plan, target_buffer)?;
            return Ok(target_buffer.as_slice());
        }

        let frame = self.frame().ok_or(CoreError::NoFrameAvailable)?;
        let rendered = processed_frame(
            frame,
            aspect_ratio,
            target_width,
            target_height,
            rgb,
            crop,
            resize_filter,
        )?;
        let use_display_buffer = self.use_display_buffer(target_width, target_height);
        let target_buffer = if use_display_buffer {
            &mut self.display_buffer
        } else {
            &mut self.observation_buffer
        };
        target_buffer.clear();
        target_buffer.extend_from_slice(&rendered);
        Ok(target_buffer.as_slice())
    }

    pub fn stacked_observation_frame(
        &mut self,
        request: StackedObservationRequest,
    ) -> Result<&[u8], CoreError> {
        let render_request = RenderRequest {
            source_width: self
                .raw_frame
                .as_ref()
                .map(|raw_frame| raw_frame.width)
                .or_else(|| self.frame.as_ref().map(|frame| frame.width))
                .ok_or(CoreError::NoFrameAvailable)?,
            source_height: self
                .raw_frame
                .as_ref()
                .map(|raw_frame| raw_frame.height)
                .or_else(|| self.frame.as_ref().map(|frame| frame.height))
                .ok_or(CoreError::NoFrameAvailable)?,
            aspect_ratio: request.aspect_ratio,
            target_width: request.target_width,
            target_height: request.target_height,
            rgb: request.rgb,
            crop: request.crop,
            resize_filter: request.resize_filter,
        };
        let frame_serial = self.frame_serial;
        let stack_key = StackedObservationKey {
            render_plan: render_request.plan_key(),
            frame_stack: request.frame_stack,
            stack_mode: request.stack_mode,
            extra_channels_per_pixel: usize::from(request.minimap_layer.is_some()),
        };
        self.render_observation_into_buffer(render_request)?;
        let minimap_layer = match request.minimap_layer {
            Some(minimap_request) => {
                self.render_minimap_layer_into_buffer(minimap_request)?;
                Some(self.minimap_buffer.clone())
            }
            None => None,
        };

        let observation_buffer = self.observation_buffer.as_slice();
        let stack_buffer = self
            .stacked_observation_buffers
            .entry(stack_key)
            .or_insert_with(|| {
                StackedObservationBuffer::new(
                    observation_buffer.len(),
                    request.frame_stack,
                    if request.rgb { 3 } else { 1 },
                    request.stack_mode,
                    usize::from(request.minimap_layer.is_some()),
                )
            });
        stack_buffer.update(observation_buffer, frame_serial, minimap_layer.as_deref())?;
        Ok(stack_buffer.as_slice())
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

    pub(super) fn handle_environment(&mut self, cmd: u32, data: *mut c_void) -> bool {
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
            ENVIRONMENT_SET_HW_RENDER => self.set_hardware_render(data),
            _ => false,
        }
    }

    pub fn take_hardware_render_error(&mut self) -> Option<String> {
        self.hardware_render_error.take()
    }

    pub fn reset_hardware_context(&mut self) -> Result<(), CoreError> {
        let Some(context) = self.hardware_render.as_ref() else {
            return Ok(());
        };
        context
            .reset_core_context()
            .map_err(|message| CoreError::HardwareRenderFailed { message })
    }

    pub fn destroy_hardware_context(&mut self) {
        if let Some(context) = self.hardware_render.as_ref() {
            context.destroy_core_context();
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

    fn set_hardware_render(&mut self, data: *mut c_void) -> bool {
        if data.is_null() {
            self.hardware_render_error = Some("SET_HW_RENDER received null callback".to_owned());
            return false;
        }

        let callback = unsafe { &mut *data.cast::<libretro_sys::HwRenderCallback>() };
        match HardwareRenderContext::from_callback(callback) {
            Ok(context) => {
                self.hardware_render = Some(context);
                self.hardware_render_error = None;
                true
            }
            Err(message) => {
                self.hardware_render_error = Some(message);
                false
            }
        }
    }

    pub(super) fn store_video_frame(
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

        if HardwareRenderContext::can_capture(data) {
            if let Some(hardware_render) = self.hardware_render.as_mut()
                && let Some(frame) = hardware_render.capture_frame(width, height)
            {
                self.raw_frame = None;
                self.frame = Some(frame);
                self.frame_serial += 1;
            }
            return;
        }

        let updated = match self.raw_frame.as_mut() {
            Some(raw_frame) => {
                capture_raw_frame_into(raw_frame, data, width, height, pitch, self.pixel_format)
            }
            None => false,
        };

        if updated {
            self.frame = None;
            self.frame_serial += 1;
            return;
        }

        if let Some(raw_frame) = capture_raw_frame(data, width, height, pitch, self.pixel_format) {
            self.raw_frame = Some(raw_frame);
            self.frame = None;
            self.frame_serial += 1;
        }
    }

    fn render_plan(&mut self, request: RenderRequest) -> Result<&ProcessedFramePlan, CoreError> {
        // Observation and display targets are stable for a run, so the sampling
        // plan is cached once per source/target combination.
        let key = ProcessedFramePlanKey {
            source_width: request.source_width,
            source_height: request.source_height,
            crop: request.crop,
            aspect_ratio_bits: request.aspect_ratio.to_bits(),
            target_width: request.target_width,
            target_height: request.target_height,
            rgb: request.rgb,
            resize_filter: request.resize_filter,
        };
        if !self.render_plans.contains_key(&key) {
            let plan = build_processed_frame_plan(
                request.source_width,
                request.source_height,
                request.aspect_ratio,
                request.target_width,
                request.target_height,
                request.rgb,
                request.crop,
                request.resize_filter,
            )?;
            self.render_plans.insert(key.clone(), plan);
        }
        self.render_plans
            .get(&key)
            .ok_or(CoreError::NoFrameAvailable)
    }

    fn render_observation_into_buffer(&mut self, request: RenderRequest) -> Result<(), CoreError> {
        if self
            .raw_frame
            .as_ref()
            .map(|raw_frame| (raw_frame.width, raw_frame.height))
            .is_some()
        {
            let plan = self.render_plan(request)?.clone();
            let raw_frame = self.raw_frame.as_ref().ok_or(CoreError::NoFrameAvailable)?;
            processed_frame_from_raw_into(raw_frame, &plan, &mut self.observation_buffer)?;
            return Ok(());
        }

        let frame = self.frame().ok_or(CoreError::NoFrameAvailable)?;
        let rendered = processed_frame(
            frame,
            request.aspect_ratio,
            request.target_width,
            request.target_height,
            request.rgb,
            request.crop,
            request.resize_filter,
        )?;
        self.observation_buffer.clear();
        self.observation_buffer.extend_from_slice(&rendered);
        Ok(())
    }

    fn render_minimap_layer_into_buffer(
        &mut self,
        request: MinimapLayerRequest,
    ) -> Result<(), CoreError> {
        if let Some(raw_frame) = self.raw_frame.as_ref() {
            self.minimap_renderer.render_from_raw_into(
                raw_frame,
                request,
                &mut self.minimap_buffer,
            )?;
            return Ok(());
        }

        let rendered = {
            let frame = self.frame().cloned().ok_or(CoreError::NoFrameAvailable)?;
            let mut output = Vec::new();
            self.minimap_renderer
                .render_from_frame_into(&frame, request, &mut output)?;
            output
        };
        self.minimap_buffer = rendered;
        Ok(())
    }

    fn clear_stacked_observation_buffers(&mut self) {
        for stack in self.stacked_observation_buffers.values_mut() {
            stack.clear();
        }
    }

    #[cfg(test)]
    pub(super) fn set_frame_for_test_without_reset(&mut self, frame: VideoFrame) {
        self.geometry = Some((frame.width, frame.height));
        self.raw_frame = None;
        self.frame = Some(frame);
        self.frame_serial += 1;
    }

    fn use_display_buffer(&self, target_width: usize, target_height: usize) -> bool {
        if let Some((frame_width, frame_height)) = self.geometry {
            if target_width == frame_width {
                return true;
            }
            if target_width <= frame_width && target_height <= frame_height {
                return false;
            }
        }
        target_width >= target_height
    }
}

impl RenderRequest {
    fn plan_key(self) -> ProcessedFramePlanKey {
        ProcessedFramePlanKey {
            source_width: self.source_width,
            source_height: self.source_height,
            crop: self.crop,
            aspect_ratio_bits: self.aspect_ratio.to_bits(),
            target_width: self.target_width,
            target_height: self.target_height,
            rgb: self.rgb,
            resize_filter: self.resize_filter,
        }
    }
}
