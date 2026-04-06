// rust/bindings/error.rs
use pyo3::PyErr;
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError};

use crate::core::error::CoreError;

pub fn map_core_error(error: CoreError) -> PyErr {
    match error {
        CoreError::MissingCore(_) | CoreError::MissingRom(_) => {
            PyFileNotFoundError::new_err(error.to_string())
        }
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}
