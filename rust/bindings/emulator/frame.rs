// rust/bindings/emulator/frame.rs
//! Frame buffer conversion helpers for the PyO3 emulator binding.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::core::host::DisplayFrameBatch;

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

pub(super) fn frames_to_pylist<'py>(
    py: Python<'py>,
    frames: &DisplayFrameBatch,
    height: usize,
    width: usize,
    channels: usize,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for frame in frames.bytes.chunks_exact(frames.frame_len) {
        list.append(frame_to_pyarray(py, frame, height, width, channels)?)?;
    }
    Ok(list)
}
