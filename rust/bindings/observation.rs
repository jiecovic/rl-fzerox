// rust/bindings/observation.rs
//! Python helpers for native-owned observation layout rules.

use pyo3::prelude::*;

use crate::bindings::error::map_core_error;
use crate::core::observation::ObservationStackMode;

#[pyfunction]
#[pyo3(signature = (single_frame_channels, frame_stack, stack_mode="rgb", minimap_layer=false))]
pub fn stacked_observation_channels(
    single_frame_channels: usize,
    frame_stack: usize,
    stack_mode: &str,
    minimap_layer: bool,
) -> PyResult<usize> {
    let parsed_stack_mode = ObservationStackMode::parse(stack_mode).map_err(map_core_error)?;
    Ok(
        parsed_stack_mode.stacked_channels(single_frame_channels, frame_stack)
            + usize::from(minimap_layer),
    )
}
