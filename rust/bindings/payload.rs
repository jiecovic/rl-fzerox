// rust/bindings/payload.rs
//! Small helpers for dictionary-shaped Python/native boundary payloads.

use pyo3::PyErr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

macro_rules! set_py_dict_items {
    ($dict:expr, { $($key:literal => $value:expr),* $(,)? }) => {{
        $(
            $dict.set_item($key, $value)?;
        )*
        Ok::<(), pyo3::PyErr>(())
    }};
}

pub(crate) use set_py_dict_items;

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
    required_item(data, context, key)?
        .cast_into::<PyDict>()
        .map_err(|error| payload_type_error(context, key, error))
}

pub(crate) fn required_list<'py>(
    data: &Bound<'py, PyDict>,
    context: &str,
    key: &str,
) -> PyResult<Bound<'py, PyList>> {
    required_item(data, context, key)?
        .cast_into::<PyList>()
        .map_err(|error| payload_type_error(context, key, error))
}

pub(crate) fn required_extract<'py, T>(
    data: &Bound<'py, PyDict>,
    context: &str,
    key: &str,
) -> PyResult<T>
where
    T: pyo3::prelude::FromPyObjectOwned<'py, Error = PyErr>,
{
    required_item(data, context, key)?
        .extract()
        .map_err(|error| payload_type_error(context, key, error))
}

pub(crate) fn optional_extract<'py, T>(
    data: &Bound<'py, PyDict>,
    context: &str,
    key: &str,
    default: T,
) -> PyResult<T>
where
    T: pyo3::prelude::FromPyObjectOwned<'py, Error = PyErr>,
{
    match data.get_item(key)? {
        Some(value) => value
            .extract()
            .map_err(|error| payload_type_error(context, key, error)),
        None => Ok(default),
    }
}

pub(crate) fn payload_type_error(context: &str, key: &str, error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{context} field {key:?} has invalid type: {error}"))
}
