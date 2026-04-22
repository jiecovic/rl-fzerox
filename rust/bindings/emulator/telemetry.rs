// rust/bindings/emulator/telemetry.rs
//! PyO3 telemetry objects exposed directly to Python.

mod player;
mod root;

use pyo3::prelude::*;

use crate::core::telemetry::TelemetrySnapshot;

pub use player::PyPlayerTelemetry;
pub use root::PyTelemetry;

pub(super) fn telemetry_to_py(
    py: Python<'_>,
    telemetry: &TelemetrySnapshot,
) -> PyResult<Py<PyTelemetry>> {
    let player = Py::new(py, PyPlayerTelemetry::from_native(&telemetry.player))?;
    Py::new(py, PyTelemetry::from_native(telemetry, player))
}
