// rust/bindings/emulator/methods/repeat.rs
//! Repeated-step method bodies used by training and watch playback.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

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

pub(in crate::bindings::emulator) fn step_repeat_raw<'py>(
    emulator: &mut PyEmulator,
    py: Python<'py>,
    request: &Bound<'_, PyDict>,
) -> PyResult<Bound<'py, PyTuple>> {
    let request = RepeatObservationBindingRequest::from_py_dict(request)?;
    let prepared = prepare_observation_render(emulator, &request.observation)?;
    let result = py
        .detach(|| {
            emulator
                .host
                .step_repeat_raw(request.step_config, prepared.config)
        })
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
    request: &Bound<'_, PyDict>,
) -> PyResult<Bound<'py, PyTuple>> {
    let request = RepeatObservationBindingRequest::from_py_dict(request)?;
    let prepared = prepare_observation_render(emulator, &request.observation)?;
    let result = py
        .detach(|| {
            emulator
                .host
                .step_repeat_watch_raw(request.step_config, prepared.config)
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
    request: &Bound<'_, PyDict>,
) -> PyResult<Bound<'py, PyTuple>> {
    let request = RepeatMultiObservationBindingRequest::from_py_dict(request)?;
    let mut prepared_observations = Vec::with_capacity(request.observations.len());
    for observation in &request.observations {
        prepared_observations.push(prepare_observation_render(emulator, observation)?);
    }
    let observation_configs = prepared_observations
        .iter()
        .map(|prepared| prepared.config)
        .collect::<Vec<_>>();
    let result = py
        .detach(|| {
            emulator
                .host
                .step_repeat_multi_observation_raw(request.step_config, &observation_configs)
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

struct RepeatObservationBindingRequest {
    step_config: RepeatedStepConfig,
    observation: ObservationImageRequest,
}

struct RepeatMultiObservationBindingRequest {
    step_config: RepeatedStepConfig,
    observations: Vec<ObservationImageRequest>,
}

struct PreparedObservationRender {
    config: ObservationRenderConfig,
    frame_height: usize,
    frame_width: usize,
    display_height: usize,
    display_width: usize,
    stacked_channels: usize,
}

impl RepeatObservationBindingRequest {
    fn from_py_dict(request: &Bound<'_, PyDict>) -> PyResult<Self> {
        let step_request = required_dict(request, "step")?;
        let observation_request = required_dict(request, "observation")?;
        let step_config = repeated_step_config(&step_request)?;
        let observation = observation_request_from_dict(&observation_request)?;
        Ok(Self {
            step_config,
            observation,
        })
    }
}

impl RepeatMultiObservationBindingRequest {
    fn from_py_dict(request: &Bound<'_, PyDict>) -> PyResult<Self> {
        let step_request = required_dict(request, "step")?;
        let step_config = repeated_step_config(&step_request)?;
        let observations_raw = required_item(request, "observations")?;
        let observations_list = observations_raw.cast::<PyList>()?;
        if observations_list.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "observations must contain at least one observation recipe",
            ));
        }
        let mut observations = Vec::with_capacity(observations_list.len());
        for observation in observations_list.iter() {
            observations.push(observation_request_from_dict(
                observation.cast::<PyDict>()?,
            )?);
        }
        Ok(Self {
            step_config,
            observations,
        })
    }
}

fn repeated_step_config(request: &Bound<'_, PyDict>) -> PyResult<RepeatedStepConfig> {
    let controller_state = ControllerState::from_normalized(
        optional_item(request, "joypad_mask", 0)?,
        optional_item(request, "left_stick_x", 0.0)?,
        optional_item(request, "left_stick_y", 0.0)?,
        optional_item(request, "right_stick_x", 0.0)?,
        optional_item(request, "right_stick_y", 0.0)?,
    );
    Ok(RepeatedStepConfig {
        controller_state,
        action_repeat: required_item(request, "action_repeat")?.extract()?,
        stuck_min_speed_kph: required_item(request, "stuck_min_speed_kph")?.extract()?,
        energy_loss_epsilon: required_item(request, "energy_loss_epsilon")?.extract()?,
        max_episode_steps: required_item(request, "max_episode_steps")?.extract()?,
        progress_frontier_stall_limit_frames: optional_item(
            request,
            "progress_frontier_stall_limit_frames",
            None,
        )?,
        progress_frontier_epsilon: optional_item(request, "progress_frontier_epsilon", 100.0)?,
        terminate_on_energy_depleted: optional_item(request, "terminate_on_energy_depleted", true)?,
        lean_timer_assist: optional_item(request, "lean_timer_assist", false)?,
    })
}

fn observation_request_from_dict(request: &Bound<'_, PyDict>) -> PyResult<ObservationImageRequest> {
    ObservationImageRequest::from_py_dict(request)
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

fn required_dict<'py>(request: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyDict>> {
    Ok(required_item(request, key)?.cast_into::<PyDict>()?)
}

fn required_item<'py>(request: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    request.get_item(key)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("repeat request missing {key:?}"))
    })
}

fn optional_item<'py, T>(request: &Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T>
where
    T: pyo3::prelude::FromPyObjectOwned<'py, Error = pyo3::PyErr>,
{
    match request.get_item(key)? {
        Some(value) => value.extract(),
        None => Ok(default),
    }
}
