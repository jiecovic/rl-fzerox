// rust/bindings/error.rs
//! Mapping from native `CoreError` values into Python exception classes.

use pyo3::PyErr;
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyRuntimeError, PyValueError};

use crate::core::error::CoreError;

/// Convert native host/core errors into a small Python exception surface.
pub fn map_core_error(error: CoreError) -> PyErr {
    match error {
        CoreError::MissingCore(_) | CoreError::MissingRom(_) => {
            PyFileNotFoundError::new_err(error.to_string())
        }
        CoreError::CreateDirectory { .. }
        | CoreError::ReadFile { .. }
        | CoreError::WriteFile { .. } => PyOSError::new_err(error.to_string()),
        CoreError::InvalidSaveRamSize { .. }
        | CoreError::InvalidPath { .. }
        | CoreError::MemoryOutOfRange { .. }
        | CoreError::InvalidStepRepeatCount { .. }
        | CoreError::InvalidObservationPreset { .. }
        | CoreError::InvalidResizeFilter { .. }
        | CoreError::InvalidVideoCrop { .. }
        | CoreError::InvalidVideoBuffer { .. }
        | CoreError::InvalidRaceStartSetup { .. }
        | CoreError::InvalidRomHeader { .. }
        | CoreError::UnsupportedRom { .. } => PyValueError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}
