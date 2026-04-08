// rust/bindings/emulator/step.rs
//! Step-summary conversion helpers for the PyO3 emulator binding.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::core::host::StepSummary;

/// Convert one native step summary into the tuple layout expected by the
/// Python-side repeated-step fast path.
pub(super) fn step_summary_to_pytuple<'py>(
    py: Python<'py>,
    summary: &StepSummary,
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        [
            summary.frames_run.into_pyobject(py)?.into_any(),
            summary.max_race_distance.into_pyobject(py)?.into_any(),
            summary.reverse_progress_total.into_pyobject(py)?.into_any(),
            summary
                .consecutive_reverse_frames
                .into_pyobject(py)?
                .into_any(),
            summary.energy_loss_total.into_pyobject(py)?.into_any(),
            summary
                .consecutive_low_speed_frames
                .into_pyobject(py)?
                .into_any(),
            summary.entered_state_flags.into_pyobject(py)?.into_any(),
            summary.final_frame_index.into_pyobject(py)?.into_any(),
        ],
    )
}
