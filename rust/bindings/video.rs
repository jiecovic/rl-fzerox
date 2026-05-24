// rust/bindings/video.rs
//! Python helpers for native-owned video layout rules.

use pyo3::prelude::*;

use crate::core::video::display_size as native_display_size;

#[pyfunction]
#[pyo3(signature = (width, height, aspect_ratio))]
pub fn display_size(width: usize, height: usize, aspect_ratio: f64) -> (usize, usize) {
    native_display_size(width, height, aspect_ratio)
}
