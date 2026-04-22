// rust/bindings/emulator.rs
//! Python binding facade for the native libretro host runtime.

use std::path::Path;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use crate::bindings::error::map_core_error;
use crate::core::error::CoreError;
use crate::core::host::Host;
use crate::core::video::VideoResizeFilter;

mod frame;
mod methods;
mod state;
mod step;
mod telemetry;

pub use state::encode_state_flags;
pub use step::{PyStepStatus, PyStepSummary};
pub use telemetry::{PyPlayerTelemetry, PyTelemetry};

#[derive(Debug)]
struct FrameObservationOptions {
    stack_mode: String,
    minimap_layer: bool,
    resize_filter: String,
    minimap_resize_filter: String,
}

impl Default for FrameObservationOptions {
    fn default() -> Self {
        Self {
            stack_mode: "rgb".to_owned(),
            minimap_layer: false,
            resize_filter: "nearest".to_owned(),
            minimap_resize_filter: "nearest".to_owned(),
        }
    }
}

impl FrameObservationOptions {
    fn from_py_dict(options: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut parsed = Self::default();
        let Some(options) = options else {
            return Ok(parsed);
        };

        if let Some(value) = options.get_item("stack_mode")? {
            parsed.stack_mode = value.extract()?;
        }
        if let Some(value) = options.get_item("minimap_layer")? {
            parsed.minimap_layer = value.extract()?;
        }
        if let Some(value) = options.get_item("resize_filter")? {
            parsed.resize_filter = value.extract()?;
        }
        if let Some(value) = options.get_item("minimap_resize_filter")? {
            parsed.minimap_resize_filter = value.extract()?;
        }
        Ok(parsed)
    }
}

/// Python-facing wrapper around one native `Host` instance.
#[pyclass(name = "Emulator", unsendable)]
pub struct PyEmulator {
    host: Host,
}

#[pymethods]
impl PyEmulator {
    #[new]
    #[pyo3(signature = (core_path, rom_path, runtime_dir=None, baseline_state_path=None, renderer="angrylion"))]
    fn new(
        py: Python<'_>,
        core_path: &str,
        rom_path: &str,
        runtime_dir: Option<&str>,
        baseline_state_path: Option<&str>,
        renderer: &str,
    ) -> PyResult<Self> {
        let host = py
            .detach(|| {
                Host::open(
                    Path::new(core_path),
                    Path::new(rom_path),
                    runtime_dir.map(Path::new),
                    baseline_state_path.map(Path::new),
                    renderer,
                )
            })
            .map_err(map_core_error)?;
        Ok(Self { host })
    }

    #[getter]
    fn name(&self) -> String {
        self.host.name().to_owned()
    }

    #[getter]
    fn native_fps(&self) -> f64 {
        self.host.native_fps()
    }

    #[getter]
    fn display_aspect_ratio(&self) -> f64 {
        self.host.display_aspect_ratio()
    }

    #[getter]
    fn frame_shape(&self) -> (usize, usize, usize) {
        self.host.frame_shape()
    }

    #[getter]
    fn frame_index(&self) -> usize {
        self.host.frame_index()
    }

    #[getter]
    fn system_ram_size(&mut self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.host.system_ram_size())
            .map_err(map_core_error)
    }

    #[getter]
    fn baseline_kind(&self) -> &'static str {
        self.host.baseline_kind()
    }

    fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        methods::control::reset(self, py)
    }

    #[pyo3(signature = (count=1, capture_video=true))]
    fn step_frames(&mut self, py: Python<'_>, count: usize, capture_video: bool) -> PyResult<()> {
        methods::control::step_frames(self, py, count, capture_video)
    }

    #[pyo3(signature = (
        action_repeat,
        preset,
        frame_stack,
        stuck_min_speed_kph,
        energy_loss_epsilon,
        max_episode_steps,
        stuck_step_limit,
        wrong_way_timer_limit,
        progress_frontier_stall_limit_frames=None,
        progress_frontier_epsilon=100.0,
        terminate_on_energy_depleted=true,
        lean_timer_assist=false,
        stack_mode="rgb",
        minimap_layer=false,
        resize_filter="nearest",
        minimap_resize_filter="nearest",
        joypad_mask=0,
        left_stick_x=0.0,
        left_stick_y=0.0,
        right_stick_x=0.0,
        right_stick_y=0.0,
    ))]
    #[expect(
        clippy::too_many_arguments,
        reason = "PyO3 method signature is the stable Python training API"
    )]
    fn step_repeat_raw<'py>(
        &mut self,
        py: Python<'py>,
        action_repeat: usize,
        preset: &str,
        frame_stack: usize,
        stuck_min_speed_kph: f32,
        energy_loss_epsilon: f32,
        max_episode_steps: usize,
        stuck_step_limit: usize,
        wrong_way_timer_limit: Option<usize>,
        progress_frontier_stall_limit_frames: Option<usize>,
        progress_frontier_epsilon: f32,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool,
        stack_mode: &str,
        minimap_layer: bool,
        resize_filter: &str,
        minimap_resize_filter: &str,
        joypad_mask: u16,
        left_stick_x: f32,
        left_stick_y: f32,
        right_stick_x: f32,
        right_stick_y: f32,
    ) -> PyResult<Bound<'py, PyTuple>> {
        methods::repeat::step_repeat_raw(
            self,
            py,
            methods::repeat::RepeatStepArgs {
                action_repeat,
                preset,
                frame_stack,
                stuck_min_speed_kph,
                energy_loss_epsilon,
                max_episode_steps,
                stuck_step_limit,
                wrong_way_timer_limit,
                progress_frontier_stall_limit_frames,
                progress_frontier_epsilon,
                terminate_on_energy_depleted,
                lean_timer_assist,
                stack_mode,
                minimap_layer,
                resize_filter,
                minimap_resize_filter,
                joypad_mask,
                left_stick_x,
                left_stick_y,
                right_stick_x,
                right_stick_y,
            },
        )
    }

    #[pyo3(signature = (
        action_repeat,
        preset,
        frame_stack,
        stuck_min_speed_kph,
        energy_loss_epsilon,
        max_episode_steps,
        stuck_step_limit,
        wrong_way_timer_limit,
        progress_frontier_stall_limit_frames=None,
        progress_frontier_epsilon=100.0,
        terminate_on_energy_depleted=true,
        lean_timer_assist=false,
        stack_mode="rgb",
        minimap_layer=false,
        resize_filter="nearest",
        minimap_resize_filter="nearest",
        joypad_mask=0,
        left_stick_x=0.0,
        left_stick_y=0.0,
        right_stick_x=0.0,
        right_stick_y=0.0,
    ))]
    #[expect(
        clippy::too_many_arguments,
        reason = "PyO3 method signature is the stable Python watch API"
    )]
    fn step_repeat_watch_raw<'py>(
        &mut self,
        py: Python<'py>,
        action_repeat: usize,
        preset: &str,
        frame_stack: usize,
        stuck_min_speed_kph: f32,
        energy_loss_epsilon: f32,
        max_episode_steps: usize,
        stuck_step_limit: usize,
        wrong_way_timer_limit: Option<usize>,
        progress_frontier_stall_limit_frames: Option<usize>,
        progress_frontier_epsilon: f32,
        terminate_on_energy_depleted: bool,
        lean_timer_assist: bool,
        stack_mode: &str,
        minimap_layer: bool,
        resize_filter: &str,
        minimap_resize_filter: &str,
        joypad_mask: u16,
        left_stick_x: f32,
        left_stick_y: f32,
        right_stick_x: f32,
        right_stick_y: f32,
    ) -> PyResult<Bound<'py, PyTuple>> {
        methods::repeat::step_repeat_watch_raw(
            self,
            py,
            methods::repeat::RepeatStepArgs {
                action_repeat,
                preset,
                frame_stack,
                stuck_min_speed_kph,
                energy_loss_epsilon,
                max_episode_steps,
                stuck_step_limit,
                wrong_way_timer_limit,
                progress_frontier_stall_limit_frames,
                progress_frontier_epsilon,
                terminate_on_energy_depleted,
                lean_timer_assist,
                stack_mode,
                minimap_layer,
                resize_filter,
                minimap_resize_filter,
                joypad_mask,
                left_stick_x,
                left_stick_y,
                right_stick_x,
                right_stick_y,
            },
        )
    }

    #[pyo3(signature = (
        joypad_mask=0,
        left_stick_x=0.0,
        left_stick_y=0.0,
        right_stick_x=0.0,
        right_stick_y=0.0,
    ))]
    fn set_controller_state(
        &mut self,
        py: Python<'_>,
        joypad_mask: u16,
        left_stick_x: f32,
        left_stick_y: f32,
        right_stick_x: f32,
        right_stick_y: f32,
    ) -> PyResult<()> {
        methods::control::set_controller_state(
            self,
            py,
            joypad_mask,
            left_stick_x,
            left_stick_y,
            right_stick_x,
            right_stick_y,
        )
    }

    fn save_state(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        methods::control::save_state(self, py, path)
    }

    fn load_baseline(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        methods::control::load_baseline(self, py, path)
    }

    fn load_baseline_bytes(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        methods::control::load_baseline_bytes(self, state)
    }

    #[pyo3(signature = (path=None))]
    fn capture_current_as_baseline(&mut self, py: Python<'_>, path: Option<&str>) -> PyResult<()> {
        methods::control::capture_current_as_baseline(self, py, path)
    }

    fn frame_rgb<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        methods::frame::frame_rgb(self, py)
    }

    fn observation_spec<'py>(
        &mut self,
        py: Python<'py>,
        preset: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        methods::frame::observation_spec(self, py, preset)
    }

    #[pyo3(signature = (preset, frame_stack, options=None))]
    fn frame_observation<'py>(
        &mut self,
        py: Python<'py>,
        preset: &str,
        frame_stack: usize,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        methods::frame::frame_observation(self, py, preset, frame_stack, options)
    }

    #[pyo3(signature = (preset))]
    fn frame_display<'py>(&mut self, py: Python<'py>, preset: &str) -> PyResult<Bound<'py, PyAny>> {
        methods::frame::frame_display(self, py, preset)
    }

    fn telemetry(&mut self, py: Python<'_>) -> PyResult<Py<PyTelemetry>> {
        methods::control::telemetry(self, py)
    }

    fn read_system_ram<'py>(
        &mut self,
        py: Python<'py>,
        offset: usize,
        length: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        methods::control::read_system_ram(self, py, offset, length)
    }

    fn write_system_ram(
        &mut self,
        py: Python<'_>,
        offset: usize,
        data: &Bound<'_, PyBytes>,
    ) -> PyResult<()> {
        methods::control::write_system_ram(self, py, offset, data)
    }

    fn game_rng_state(&mut self, py: Python<'_>) -> PyResult<(u32, u32, u32, u32)> {
        methods::control::game_rng_state(self, py)
    }

    fn randomize_game_rng(&mut self, py: Python<'_>, seed: u64) -> PyResult<(u32, u32, u32, u32)> {
        methods::control::randomize_game_rng(self, py, seed)
    }

    fn close(&mut self) {
        methods::control::close(self);
    }
}

fn parse_resize_filter(name: &str) -> PyResult<VideoResizeFilter> {
    VideoResizeFilter::parse(name).ok_or_else(|| {
        map_core_error(CoreError::InvalidResizeFilter {
            name: name.to_owned(),
        })
    })
}
