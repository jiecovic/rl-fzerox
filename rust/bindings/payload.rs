// rust/bindings/payload.rs
//! Small helpers for dictionary-shaped Python/native boundary payloads.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

pub(crate) fn required_item<'py>(
    data: &Bound<'py, PyDict>,
    context: &str,
    key: &str,
) -> PyResult<Bound<'py, PyAny>> {
    data.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("{context} missing {key:?}")))
}

pub(crate) fn required_dict<'py>(
    data: &Bound<'py, PyDict>,
    context: &str,
    key: &str,
) -> PyResult<Bound<'py, PyDict>> {
    Ok(required_item(data, context, key)?.cast_into::<PyDict>()?)
}

pub(crate) fn optional_item<'py, T>(data: &Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T>
where
    T: pyo3::prelude::FromPyObjectOwned<'py, Error = pyo3::PyErr>,
{
    match data.get_item(key)? {
        Some(value) => value.extract(),
        None => Ok(default),
    }
}
