// rust/bindings/emulator/step.rs
//! PyO3 step-summary object exposed directly to Python.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::bindings::emulator::state::{
    FLAG_COLLISION_RECOIL, FLAG_CRASHED, FLAG_DASH_PAD_BOOST, FLAG_FALLING_OFF_TRACK,
    FLAG_FINISHED, FLAG_RETIRED, FLAG_SPINNING_OUT, has_state_flag, state_flag_labels,
};
use crate::core::host::StepSummary;

#[pyclass(
    name = "StepSummary",
    module = "fzerox_emulator._native",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub struct PyStepSummary {
    inner: StepSummary,
}

#[pymethods]
impl PyStepSummary {
    #[new]
    #[pyo3(signature = (
        frames_run,
        max_race_distance,
        reverse_progress_total=0.0,
        consecutive_reverse_frames=0,
        energy_loss_total=0.0,
        consecutive_low_speed_frames=0,
        entered_state_flags=0,
        final_frame_index=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        frames_run: usize,
        max_race_distance: f32,
        reverse_progress_total: f32,
        consecutive_reverse_frames: usize,
        energy_loss_total: f32,
        consecutive_low_speed_frames: usize,
        entered_state_flags: u32,
        final_frame_index: usize,
    ) -> Self {
        Self {
            inner: StepSummary {
                frames_run,
                max_race_distance,
                reverse_progress_total,
                consecutive_reverse_frames,
                energy_loss_total,
                consecutive_low_speed_frames,
                entered_state_flags,
                final_frame_index,
            },
        }
    }

    #[getter]
    fn frames_run(&self) -> usize {
        self.inner.frames_run
    }

    #[getter]
    fn max_race_distance(&self) -> f32 {
        self.inner.max_race_distance
    }

    #[getter]
    fn reverse_progress_total(&self) -> f32 {
        self.inner.reverse_progress_total
    }

    #[getter]
    fn consecutive_reverse_frames(&self) -> usize {
        self.inner.consecutive_reverse_frames
    }

    #[getter]
    fn energy_loss_total(&self) -> f32 {
        self.inner.energy_loss_total
    }

    #[getter]
    fn consecutive_low_speed_frames(&self) -> usize {
        self.inner.consecutive_low_speed_frames
    }

    #[getter]
    fn entered_state_flags(&self) -> u32 {
        self.inner.entered_state_flags
    }

    #[getter]
    fn entered_state_labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, state_flag_labels(self.inner.entered_state_flags))
    }

    #[getter]
    fn entered_collision_recoil(&self) -> bool {
        has_state_flag(self.inner.entered_state_flags, FLAG_COLLISION_RECOIL)
    }

    #[getter]
    fn entered_spinning_out(&self) -> bool {
        has_state_flag(self.inner.entered_state_flags, FLAG_SPINNING_OUT)
    }

    #[getter]
    fn entered_falling_off_track(&self) -> bool {
        has_state_flag(self.inner.entered_state_flags, FLAG_FALLING_OFF_TRACK)
    }

    #[getter]
    fn entered_crashed(&self) -> bool {
        has_state_flag(self.inner.entered_state_flags, FLAG_CRASHED)
    }

    #[getter]
    fn entered_retired(&self) -> bool {
        has_state_flag(self.inner.entered_state_flags, FLAG_RETIRED)
    }

    #[getter]
    fn entered_finished(&self) -> bool {
        has_state_flag(self.inner.entered_state_flags, FLAG_FINISHED)
    }

    #[getter]
    fn entered_dash_pad_boost(&self) -> bool {
        has_state_flag(self.inner.entered_state_flags, FLAG_DASH_PAD_BOOST)
    }

    #[getter]
    fn final_frame_index(&self) -> usize {
        self.inner.final_frame_index
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("frames_run", self.frames_run())?;
        dict.set_item("max_race_distance", self.max_race_distance())?;
        dict.set_item("reverse_progress_total", self.reverse_progress_total())?;
        dict.set_item(
            "consecutive_reverse_frames",
            self.consecutive_reverse_frames(),
        )?;
        dict.set_item("energy_loss_total", self.energy_loss_total())?;
        dict.set_item(
            "consecutive_low_speed_frames",
            self.consecutive_low_speed_frames(),
        )?;
        dict.set_item("entered_state_flags", self.entered_state_flags())?;
        dict.set_item("entered_state_labels", self.entered_state_labels(py)?)?;
        dict.set_item("final_frame_index", self.final_frame_index())?;
        Ok(dict)
    }
}

pub(super) fn step_summary_to_py(
    py: Python<'_>,
    summary: &StepSummary,
) -> PyResult<Py<PyStepSummary>> {
    Py::new(
        py,
        PyStepSummary {
            inner: summary.clone(),
        },
    )
}
