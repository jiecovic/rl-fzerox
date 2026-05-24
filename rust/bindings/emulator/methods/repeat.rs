// rust/bindings/emulator/methods/repeat.rs
//! Repeated-step method bodies used by training and watch playback.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::bindings::emulator::frame::{frame_to_pyarray, frames_to_pylist};
use crate::bindings::emulator::step::{step_status_to_py, step_summary_to_py};
use crate::bindings::emulator::telemetry::telemetry_to_py;
use crate::bindings::emulator::{
    ObservationImageRequest, PyEmulator, parse_observation_layout, parse_resize_filter,
};
use crate::bindings::error::map_core_error;
use crate::core::host::{ObservationRenderConfig, RepeatedStepConfig};
use crate::core::input::ControllerState;
use crate::core::observation::ObservationStackMode;

pub(in crate::bindings::emulator) struct RepeatStepArgs {
    pub action_repeat: usize,
    pub stuck_min_speed_kph: f32,
    pub energy_loss_epsilon: f32,
    pub max_episode_steps: usize,
    pub progress_frontier_stall_limit_frames: Option<usize>,
    pub progress_frontier_epsilon: f32,
    pub terminate_on_energy_depleted: bool,
    pub lean_timer_assist: bool,
    pub joypad_mask: u16,
    pub left_stick_x: f32,
    pub left_stick_y: f32,
    pub right_stick_x: f32,
    pub right_stick_y: f32,
}

pub(in crate::bindings::emulator) fn step_repeat_raw<'py>(
    emulator: &mut PyEmulator,
    py: Python<'py>,
    args: RepeatStepArgs,
    observation_request: ObservationImageRequest,
) -> PyResult<Bound<'py, PyTuple>> {
    let step_config = prepare_repeated_step_config(&args);
    let prepared = prepare_observation_render(emulator, &observation_request)?;
    let result = py
        .detach(|| emulator.host.step_repeat_raw(step_config, prepared.config))
        .map_err(map_core_error)?;
    let observation = frame_to_pyarray(
        py,
        result.observation,
        prepared.frame_height,
        prepared.frame_width,
        prepared.stacked_channels,
    )?;
    let summary = step_summary_to_py(py, &result.summary)?;
    let status = step_status_to_py(py, &result.status)?;
    let telemetry = telemetry_to_py(py, &result.final_telemetry)?;
    PyTuple::new(
        py,
        [
            observation,
            summary.into_bound(py).into_any(),
            status.into_bound(py).into_any(),
            telemetry.into_bound(py).into_any(),
        ],
    )
}

pub(in crate::bindings::emulator) fn step_repeat_watch_raw<'py>(
    emulator: &mut PyEmulator,
    py: Python<'py>,
    args: RepeatStepArgs,
    observation_request: ObservationImageRequest,
) -> PyResult<Bound<'py, PyTuple>> {
    let step_config = prepare_repeated_step_config(&args);
    let prepared = prepare_observation_render(emulator, &observation_request)?;
    let result = py
        .detach(|| {
            emulator
                .host
                .step_repeat_watch_raw(step_config, prepared.config)
        })
        .map_err(map_core_error)?;
    let observation = frame_to_pyarray(
        py,
        result.observation,
        prepared.frame_height,
        prepared.frame_width,
        prepared.stacked_channels,
    )?;
    let display_frames = frames_to_pylist(
        py,
        &result.display_frames,
        prepared.display_height,
        prepared.display_width,
        3,
    )?;
    let summary = step_summary_to_py(py, &result.summary)?;
    let status = step_status_to_py(py, &result.status)?;
    let telemetry = telemetry_to_py(py, &result.final_telemetry)?;
    PyTuple::new(
        py,
        [
            observation,
            display_frames.into_any(),
            summary.into_bound(py).into_any(),
            status.into_bound(py).into_any(),
            telemetry.into_bound(py).into_any(),
        ],
    )
}

pub(in crate::bindings::emulator) fn step_repeat_multi_observation_raw<'py>(
    emulator: &mut PyEmulator,
    py: Python<'py>,
    args: RepeatStepArgs,
    observation_requests: &Bound<'_, PyList>,
) -> PyResult<Bound<'py, PyTuple>> {
    if observation_requests.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "observation_requests must contain at least one observation recipe",
        ));
    }

    let step_config = prepare_repeated_step_config(&args);
    let mut prepared_observations = Vec::with_capacity(observation_requests.len());
    for request in observation_requests.iter() {
        let request_dict = request.cast::<PyDict>()?;
        let parsed_request = ObservationImageRequest::from_py_dict(request_dict)?;
        prepared_observations.push(prepare_observation_render(emulator, &parsed_request)?);
    }
    let observation_configs = prepared_observations
        .iter()
        .map(|prepared| prepared.config)
        .collect::<Vec<_>>();
    let result = py
        .detach(|| {
            emulator
                .host
                .step_repeat_multi_observation_raw(step_config, &observation_configs)
        })
        .map_err(map_core_error)?;
    let observation_list = PyList::empty(py);
    for (frame, prepared) in result
        .observations
        .frames
        .iter()
        .zip(prepared_observations.iter())
    {
        observation_list.append(frame_to_pyarray(
            py,
            frame.as_slice(),
            prepared.frame_height,
            prepared.frame_width,
            prepared.stacked_channels,
        )?)?;
    }
    let summary = step_summary_to_py(py, &result.summary)?;
    let status = step_status_to_py(py, &result.status)?;
    let telemetry = telemetry_to_py(py, &result.final_telemetry)?;
    PyTuple::new(
        py,
        [
            observation_list.into_any(),
            summary.into_bound(py).into_any(),
            status.into_bound(py).into_any(),
            telemetry.into_bound(py).into_any(),
        ],
    )
}

struct PreparedObservationRender {
    config: ObservationRenderConfig,
    frame_height: usize,
    frame_width: usize,
    display_height: usize,
    display_width: usize,
    stacked_channels: usize,
}

fn prepare_repeated_step_config(args: &RepeatStepArgs) -> RepeatedStepConfig {
    let controller_state = ControllerState::from_normalized(
        args.joypad_mask,
        args.left_stick_x,
        args.left_stick_y,
        args.right_stick_x,
        args.right_stick_y,
    );
    RepeatedStepConfig {
        controller_state,
        action_repeat: args.action_repeat,
        stuck_min_speed_kph: args.stuck_min_speed_kph,
        energy_loss_epsilon: args.energy_loss_epsilon,
        max_episode_steps: args.max_episode_steps,
        progress_frontier_stall_limit_frames: args.progress_frontier_stall_limit_frames,
        progress_frontier_epsilon: args.progress_frontier_epsilon,
        terminate_on_energy_depleted: args.terminate_on_energy_depleted,
        lean_timer_assist: args.lean_timer_assist,
    }
}

fn prepare_observation_render(
    emulator: &mut PyEmulator,
    request: &ObservationImageRequest,
) -> PyResult<PreparedObservationRender> {
    let layout = parse_observation_layout(&request.preset, request.height, request.width)?;
    let stack_mode = ObservationStackMode::parse(&request.stack_mode).map_err(map_core_error)?;
    let resize_filter = parse_resize_filter(&request.resize_filter)?;
    let minimap_resize_filter = parse_resize_filter(&request.minimap_resize_filter)?;
    let spec = emulator
        .host
        .observation_spec(layout)
        .map_err(map_core_error)?;
    Ok(PreparedObservationRender {
        config: ObservationRenderConfig {
            layout,
            frame_stack: request.frame_stack,
            stack_mode,
            minimap_layer: request.minimap_layer,
            resize_filter,
            minimap_resize_filter,
        },
        frame_height: spec.frame_height,
        frame_width: spec.frame_width,
        display_height: spec.display_height,
        display_width: spec.display_width,
        stacked_channels: stack_mode.stacked_channels(spec.channels, request.frame_stack)
            + usize::from(request.minimap_layer),
    })
}
