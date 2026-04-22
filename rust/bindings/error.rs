// rust/bindings/error.rs
//! Mapping from native `CoreError` values into Python exception classes.

use pyo3::PyErr;
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};

use crate::core::error::CoreError;

/// Convert native host/core errors into a small Python exception surface.
pub fn map_core_error(error: CoreError) -> PyErr {
    match error {
        CoreError::MissingCore(_) | CoreError::MissingRom(_) => {
            PyFileNotFoundError::new_err(error.to_string())
        }
        CoreError::InvalidObservationPreset { .. } | CoreError::InvalidResizeFilter { .. } => {
            PyValueError::new_err(error.to_string())
        }
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}
