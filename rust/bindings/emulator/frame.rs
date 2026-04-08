// rust/bindings/emulator/frame.rs
//! Frame buffer conversion helpers for the PyO3 emulator binding.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

/// Materialize a Python-owned NumPy array view of a frame buffer.
///
/// This still performs one copy into Python-owned memory; it just avoids the
/// extra `PyBytes` hop that the older path used.
pub(super) fn frame_to_pyarray<'py>(
    py: Python<'py>,
    frame: &[u8],
    height: usize,
    width: usize,
    channels: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let array = PyArray1::<u8>::from_vec(py, frame.to_vec());
    Ok(array.reshape([height, width, channels])?.into_any())
}
