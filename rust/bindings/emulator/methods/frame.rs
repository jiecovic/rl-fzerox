// rust/bindings/emulator/methods/frame.rs
//! Frame, observation, and display method bodies.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::bindings::emulator::frame::frame_to_pyarray;
use crate::bindings::emulator::{FrameObservationOptions, PyEmulator, parse_resize_filter};
use crate::bindings::error::map_core_error;
use crate::core::observation::{ObservationPreset, ObservationStackMode};

pub(in crate::bindings::emulator) fn frame_rgb<'py>(
    emulator: &mut PyEmulator,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyBytes>> {
    let frame = emulator.host.frame_rgb().map_err(map_core_error)?;
    Ok(PyBytes::new(py, frame))
}

pub(in crate::bindings::emulator) fn observation_spec<'py>(
    emulator: &mut PyEmulator,
    py: Python<'py>,
    preset: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let preset = ObservationPreset::parse(preset).map_err(map_core_error)?;
    let spec = emulator
        .host
        .observation_spec(preset)
        .map_err(map_core_error)?;
    let dict = PyDict::new(py);
    dict.set_item("preset", spec.preset_name)?;
    dict.set_item("width", spec.frame_width)?;
    dict.set_item("height", spec.frame_height)?;
    dict.set_item("channels", spec.channels)?;
    dict.set_item("display_width", spec.display_width)?;
    dict.set_item("display_height", spec.display_height)?;
    Ok(dict)
}

pub(in crate::bindings::emulator) fn frame_observation<'py>(
    emulator: &mut PyEmulator,
    py: Python<'py>,
    preset: &str,
    frame_stack: usize,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let options = FrameObservationOptions::from_py_dict(options)?;
    let preset = ObservationPreset::parse(preset).map_err(map_core_error)?;
    let stack_mode = ObservationStackMode::parse(&options.stack_mode).map_err(map_core_error)?;
    let resize_filter = parse_resize_filter(&options.resize_filter)?;
    let minimap_resize_filter = parse_resize_filter(&options.minimap_resize_filter)?;
    let spec = emulator
        .host
        .observation_spec(preset)
        .map_err(map_core_error)?;
    let frame = py
        .detach(|| {
            emulator.host.observation_frame(
                preset,
                frame_stack,
                stack_mode,
                options.minimap_layer,
                resize_filter,
                minimap_resize_filter,
            )
        })
        .map_err(map_core_error)?;
    frame_to_pyarray(
        py,
        frame,
        spec.frame_height,
        spec.frame_width,
        stack_mode.stacked_channels(spec.channels, frame_stack)
            + usize::from(options.minimap_layer),
    )
}

pub(in crate::bindings::emulator) fn frame_display<'py>(
    emulator: &mut PyEmulator,
    py: Python<'py>,
    preset: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let preset = ObservationPreset::parse(preset).map_err(map_core_error)?;
    let spec = emulator
        .host
        .observation_spec(preset)
        .map_err(map_core_error)?;
    let frame = py
        .detach(|| emulator.host.display_frame(preset))
        .map_err(map_core_error)?;
    frame_to_pyarray(py, frame, spec.display_height, spec.display_width, 3)
}
