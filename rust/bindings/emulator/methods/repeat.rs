// rust/bindings/emulator/methods/repeat.rs
//! Repeated-step method bodies used by training and watch playback.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::bindings::emulator::frame::{frame_to_pyarray, frames_to_pylist};
use crate::bindings::emulator::step::{step_status_to_py, step_summary_to_py};
use crate::bindings::emulator::telemetry::telemetry_to_py;
use crate::bindings::emulator::{PyEmulator, parse_resize_filter};
use crate::bindings::error::map_core_error;
use crate::core::host::RepeatedStepConfig;
use crate::core::input::ControllerState;
use crate::core::observation::{ObservationPreset, ObservationStackMode};

pub(in crate::bindings::emulator) struct RepeatStepArgs<'a> {
    pub action_repeat: usize,
    pub preset: &'a str,
    pub frame_stack: usize,
    pub stuck_min_speed_kph: f32,
    pub energy_loss_epsilon: f32,
    pub max_episode_steps: usize,
    pub stuck_step_limit: usize,
    pub wrong_way_timer_limit: Option<usize>,
    pub progress_frontier_stall_limit_frames: Option<usize>,
    pub progress_frontier_epsilon: f32,
    pub terminate_on_energy_depleted: bool,
    pub lean_timer_assist: bool,
    pub stack_mode: &'a str,
    pub minimap_layer: bool,
    pub resize_filter: &'a str,
    pub minimap_resize_filter: &'a str,
    pub joypad_mask: u16,
    pub left_stick_x: f32,
    pub left_stick_y: f32,
    pub right_stick_x: f32,
    pub right_stick_y: f32,
}

pub(in crate::bindings::emulator) fn step_repeat_raw<'py>(
    emulator: &mut PyEmulator,
    py: Python<'py>,
    args: RepeatStepArgs<'_>,
) -> PyResult<Bound<'py, PyTuple>> {
    let prepared = prepare_repeated_step(emulator, &args)?;
    let result = py
        .detach(|| emulator.host.step_repeat_raw(prepared.config))
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
    args: RepeatStepArgs<'_>,
) -> PyResult<Bound<'py, PyTuple>> {
    let prepared = prepare_repeated_step(emulator, &args)?;
    let result = py
        .detach(|| emulator.host.step_repeat_watch_raw(prepared.config))
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

struct PreparedRepeatStep {
    config: RepeatedStepConfig,
    frame_height: usize,
    frame_width: usize,
    display_height: usize,
    display_width: usize,
    stacked_channels: usize,
}

fn prepare_repeated_step(
    emulator: &mut PyEmulator,
    args: &RepeatStepArgs<'_>,
) -> PyResult<PreparedRepeatStep> {
    let preset = ObservationPreset::parse(args.preset).map_err(map_core_error)?;
    let stack_mode = ObservationStackMode::parse(args.stack_mode).map_err(map_core_error)?;
    let resize_filter = parse_resize_filter(args.resize_filter)?;
    let minimap_resize_filter = parse_resize_filter(args.minimap_resize_filter)?;
    let spec = emulator
        .host
        .observation_spec(preset)
        .map_err(map_core_error)?;
    let controller_state = ControllerState::from_normalized(
        args.joypad_mask,
        args.left_stick_x,
        args.left_stick_y,
        args.right_stick_x,
        args.right_stick_y,
    );
    Ok(PreparedRepeatStep {
        config: RepeatedStepConfig {
            controller_state,
            action_repeat: args.action_repeat,
            preset,
            frame_stack: args.frame_stack,
            stack_mode,
            minimap_layer: args.minimap_layer,
            resize_filter,
            minimap_resize_filter,
            stuck_min_speed_kph: args.stuck_min_speed_kph,
            energy_loss_epsilon: args.energy_loss_epsilon,
            max_episode_steps: args.max_episode_steps,
            stuck_step_limit: args.stuck_step_limit,
            wrong_way_timer_limit: args.wrong_way_timer_limit,
            progress_frontier_stall_limit_frames: args.progress_frontier_stall_limit_frames,
            progress_frontier_epsilon: args.progress_frontier_epsilon,
            terminate_on_energy_depleted: args.terminate_on_energy_depleted,
            lean_timer_assist: args.lean_timer_assist,
        },
        frame_height: spec.frame_height,
        frame_width: spec.frame_width,
        display_height: spec.display_height,
        display_width: spec.display_width,
        stacked_channels: stack_mode.stacked_channels(spec.channels, args.frame_stack)
            + usize::from(args.minimap_layer),
    })
}
