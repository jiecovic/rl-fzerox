// rust/bindings/emulator/state.rs
//! Shared racer-state flag helpers for Python-facing telemetry and step objects.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::telemetry::RACER_STATE_FLAG_SPECS;
pub(super) use crate::core::telemetry::RACER_STATE_FLAGS;

pub(super) fn has_state_flag(state_flags: u32, flag: u32) -> bool {
    (state_flags & flag) != 0
}

pub(super) fn state_flag_labels(state_flags: u32) -> Vec<&'static str> {
    RACER_STATE_FLAG_SPECS
        .iter()
        .filter(|spec| has_state_flag(state_flags, spec.mask))
        .map(|spec| spec.label)
        .collect()
}

#[pyfunction]
pub fn encode_state_flags(labels: Vec<String>) -> PyResult<u32> {
    let mut state_flags = 0u32;
    for label in labels {
        let Some(mask) = RACER_STATE_FLAG_SPECS
            .iter()
            .find(|spec| spec.label == label)
            .map(|spec| spec.mask)
        else {
            return Err(PyValueError::new_err(format!(
                "Unknown racer state label: {label}"
            )));
        };
        state_flags |= mask;
    }
    Ok(state_flags)
}
