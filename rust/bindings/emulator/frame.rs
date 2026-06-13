// rust/bindings/emulator/frame.rs
//! Frame buffer conversion helpers for the PyO3 emulator binding.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::host::{AudioFrameBatch, DisplayFrameBatch};

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

pub(super) fn frame_batch_to_pyarray<'py>(
    py: Python<'py>,
    frames: DisplayFrameBatch,
    height: usize,
    width: usize,
    channels: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let expected_frame_len = height * width * channels;
    if frames.frame_len != 0 && frames.frame_len != expected_frame_len {
        return Err(PyValueError::new_err(format!(
            "display frame length {} does not match expected shape {}x{}x{}",
            frames.frame_len, height, width, channels,
        )));
    }
    if frames.frame_len != 0 && !frames.bytes.len().is_multiple_of(frames.frame_len) {
        return Err(PyValueError::new_err(format!(
            "display frame byte count {} is not divisible by frame length {}",
            frames.bytes.len(),
            frames.frame_len,
        )));
    }
    if frames.frame_len == 0 && !frames.bytes.is_empty() {
        return Err(PyValueError::new_err(
            "display frame batch has bytes but no frame length",
        ));
    }
    let frame_count = frames.frame_count();
    let array = PyArray1::<u8>::from_vec(py, frames.bytes);
    Ok(array
        .reshape([frame_count, height, width, channels])?
        .into_any())
}

pub(super) fn audio_batch_to_pyarrays<'py>(
    py: Python<'py>,
    audio: AudioFrameBatch,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let counted_samples = audio
        .frame_counts
        .iter()
        .try_fold(0usize, |total, frame_count| {
            let frame_samples = (*frame_count as usize).checked_mul(2)?;
            total.checked_add(frame_samples)
        })
        .ok_or_else(|| PyValueError::new_err("audio frame counts overflow sample count"))?;
    if counted_samples != audio.samples.len() {
        return Err(PyValueError::new_err(format!(
            "audio sample count {} does not match frame counts {}",
            audio.samples.len(),
            counted_samples,
        )));
    }
    let samples = PyArray1::<i16>::from_vec(py, audio.samples);
    let frame_counts = PyArray1::<u32>::from_vec(py, audio.frame_counts);
    Ok((samples.into_any(), frame_counts.into_any()))
}
