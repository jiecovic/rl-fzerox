// rust/bindings/emulator/step.rs
//! PyO3 step objects exposed directly to Python.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::bindings::emulator::state::{
    FLAG_COLLISION_RECOIL, FLAG_CRASHED, FLAG_DASH_PAD_BOOST, FLAG_FALLING_OFF_TRACK,
    FLAG_FINISHED, FLAG_RETIRED, FLAG_SPINNING_OUT, has_state_flag, state_flag_labels,
};
use crate::core::host::{StepStatus, StepSummary};

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
        reverse_active_frames=0,
        low_speed_frames=0,
        energy_loss_total=0.0,
        energy_gain_total=0.0,
        damage_taken_frames=0,
        consecutive_low_speed_frames=0,
        entered_state_flags=0,
        final_frame_index=0,
        airborne_frames=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        frames_run: usize,
        max_race_distance: f32,
        reverse_active_frames: usize,
        low_speed_frames: usize,
        energy_loss_total: f32,
        energy_gain_total: f32,
        damage_taken_frames: usize,
        consecutive_low_speed_frames: usize,
        entered_state_flags: u32,
        final_frame_index: usize,
        airborne_frames: usize,
    ) -> Self {
        Self {
            inner: StepSummary {
                frames_run,
                max_race_distance,
                reverse_active_frames,
                low_speed_frames,
                energy_loss_total,
                energy_gain_total,
                damage_taken_frames,
                airborne_frames,
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
    fn reverse_active_frames(&self) -> usize {
        self.inner.reverse_active_frames
    }

    #[getter]
    fn low_speed_frames(&self) -> usize {
        self.inner.low_speed_frames
    }

    #[getter]
    fn energy_loss_total(&self) -> f32 {
        self.inner.energy_loss_total
    }

    #[getter]
    fn energy_gain_total(&self) -> f32 {
        self.inner.energy_gain_total
    }

    #[getter]
    fn damage_taken_frames(&self) -> usize {
        self.inner.damage_taken_frames
    }

    #[getter]
    fn airborne_frames(&self) -> usize {
        self.inner.airborne_frames
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
        dict.set_item("reverse_active_frames", self.reverse_active_frames())?;
        dict.set_item("low_speed_frames", self.low_speed_frames())?;
        dict.set_item("energy_loss_total", self.energy_loss_total())?;
        dict.set_item("energy_gain_total", self.energy_gain_total())?;
        dict.set_item("damage_taken_frames", self.damage_taken_frames())?;
        dict.set_item("airborne_frames", self.airborne_frames())?;
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

#[pyclass(
    name = "StepStatus",
    module = "fzerox_emulator._native",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub struct PyStepStatus {
    inner: StepStatus,
}

#[pymethods]
impl PyStepStatus {
    #[new]
    #[pyo3(signature = (
        step_count,
        stalled_steps,
        reverse_timer=0,
        progress_frontier_stalled_frames=0,
        termination_reason=None,
        truncation_reason=None,
    ))]
    fn new(
        step_count: usize,
        stalled_steps: usize,
        reverse_timer: usize,
        progress_frontier_stalled_frames: usize,
        termination_reason: Option<String>,
        truncation_reason: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: StepStatus {
                counters: crate::core::host::StepCounters {
                    step_count,
                    stalled_steps,
                    progress_frontier_stalled_frames,
                    progress_frontier_distance: 0.0,
                    progress_frontier_initialized: false,
                },
                reverse_timer,
                termination_reason: parse_reason(termination_reason)?,
                truncation_reason: parse_reason(truncation_reason)?,
            },
        })
    }

    #[getter]
    fn step_count(&self) -> usize {
        self.inner.counters.step_count
    }

    #[getter]
    fn stalled_steps(&self) -> usize {
        self.inner.counters.stalled_steps
    }

    #[getter]
    fn reverse_timer(&self) -> usize {
        self.inner.reverse_timer
    }

    #[getter]
    fn progress_frontier_stalled_frames(&self) -> usize {
        self.inner.counters.progress_frontier_stalled_frames
    }

    #[getter]
    fn terminated(&self) -> bool {
        self.inner.terminated()
    }

    #[getter]
    fn truncated(&self) -> bool {
        self.inner.truncated()
    }

    #[getter]
    fn termination_reason(&self) -> Option<&'static str> {
        self.inner.termination_reason
    }

    #[getter]
    fn truncation_reason(&self) -> Option<&'static str> {
        self.inner.truncation_reason
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("step_count", self.step_count())?;
        dict.set_item("stalled_steps", self.stalled_steps())?;
        dict.set_item("reverse_timer", self.reverse_timer())?;
        dict.set_item(
            "progress_frontier_stalled_frames",
            self.progress_frontier_stalled_frames(),
        )?;
        dict.set_item("terminated", self.terminated())?;
        dict.set_item("truncated", self.truncated())?;
        dict.set_item("termination_reason", self.termination_reason())?;
        dict.set_item("truncation_reason", self.truncation_reason())?;
        Ok(dict)
    }
}

pub(super) fn step_status_to_py(py: Python<'_>, status: &StepStatus) -> PyResult<Py<PyStepStatus>> {
    Py::new(
        py,
        PyStepStatus {
            inner: status.clone(),
        },
    )
}

fn parse_reason(reason: Option<String>) -> PyResult<Option<&'static str>> {
    match reason.as_deref() {
        None => Ok(None),
        Some("finished") => Ok(Some("finished")),
        Some("spinning_out") => Ok(Some("spinning_out")),
        Some("crashed") => Ok(Some("crashed")),
        Some("retired") => Ok(Some("retired")),
        Some("falling_off_track") => Ok(Some("falling_off_track")),
        Some("energy_depleted") => Ok(Some("energy_depleted")),
        Some("stuck") => Ok(Some("stuck")),
        Some("wrong_way") => Ok(Some("wrong_way")),
        Some("progress_stalled") => Ok(Some("progress_stalled")),
        Some("timeout") => Ok(Some("timeout")),
        Some(other) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown step reason: {other}"
        ))),
    }
}
