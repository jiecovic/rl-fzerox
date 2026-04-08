// rust/bindings/emulator/telemetry.rs
//! Telemetry conversion helpers for the PyO3 emulator binding.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::core::telemetry::{PlayerTelemetry, TelemetrySnapshot};

/// Convert the rich native telemetry snapshot into a Python dictionary for
/// introspection/debugging use.
pub(super) fn telemetry_to_pydict<'py>(
    py: Python<'py>,
    telemetry: &TelemetrySnapshot,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("system_ram_size", telemetry.system_ram_size)?;
    dict.set_item("game_frame_count", telemetry.game_frame_count)?;
    dict.set_item("game_mode_raw", telemetry.game_mode_raw)?;
    dict.set_item("game_mode_name", &telemetry.game_mode_name)?;
    dict.set_item("course_index", telemetry.course_index)?;
    dict.set_item("in_race_mode", telemetry.in_race_mode)?;
    dict.set_item("player", player_to_pydict(py, &telemetry.player)?)?;
    Ok(dict)
}

/// Convert the hot-path telemetry shape into tuples so Python can rebuild typed
/// telemetry objects with less object churn than nested dict/list creation.
pub(super) fn telemetry_to_pytuple<'py>(
    py: Python<'py>,
    telemetry: &TelemetrySnapshot,
) -> PyResult<Bound<'py, PyTuple>> {
    let player = player_to_pytuple(py, &telemetry.player)?;
    PyTuple::new(
        py,
        [
            telemetry.system_ram_size.into_pyobject(py)?.into_any(),
            telemetry.game_frame_count.into_pyobject(py)?.into_any(),
            telemetry.game_mode_raw.into_pyobject(py)?.into_any(),
            telemetry
                .game_mode_name
                .as_str()
                .into_pyobject(py)?
                .into_any(),
            telemetry.course_index.into_pyobject(py)?.into_any(),
            telemetry
                .in_race_mode
                .into_pyobject(py)?
                .to_owned()
                .into_any(),
            player.into_any(),
        ],
    )
}

/// Convert the player telemetry sub-struct into a Python dictionary.
fn player_to_pydict<'py>(
    py: Python<'py>,
    player: &PlayerTelemetry,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("state_flags", player.state_flags)?;
    dict.set_item("state_labels", PyList::new(py, &player.state_labels)?)?;
    dict.set_item("speed_raw", player.speed_raw)?;
    dict.set_item("speed_kph", player.speed_kph)?;
    dict.set_item("energy", player.energy)?;
    dict.set_item("max_energy", player.max_energy)?;
    dict.set_item("boost_timer", player.boost_timer)?;
    dict.set_item("race_distance", player.race_distance)?;
    dict.set_item("laps_completed_distance", player.laps_completed_distance)?;
    dict.set_item("lap_distance", player.lap_distance)?;
    dict.set_item("race_distance_position", player.race_distance_position)?;
    dict.set_item("race_time_ms", player.race_time_ms)?;
    dict.set_item("lap", player.lap)?;
    dict.set_item("laps_completed", player.laps_completed)?;
    dict.set_item("position", player.position)?;
    dict.set_item("character", player.character)?;
    dict.set_item("machine_index", player.machine_index)?;
    Ok(dict)
}

/// Convert the player telemetry sub-struct into the tuple layout expected by
/// the Python-side fast path.
fn player_to_pytuple<'py>(
    py: Python<'py>,
    player: &PlayerTelemetry,
) -> PyResult<Bound<'py, PyTuple>> {
    let state_labels = PyTuple::new(py, player.state_labels.iter().copied())?;
    PyTuple::new(
        py,
        [
            player.state_flags.into_pyobject(py)?.into_any(),
            state_labels.into_any(),
            player.speed_raw.into_pyobject(py)?.into_any(),
            player.speed_kph.into_pyobject(py)?.into_any(),
            player.energy.into_pyobject(py)?.into_any(),
            player.max_energy.into_pyobject(py)?.into_any(),
            player.boost_timer.into_pyobject(py)?.into_any(),
            player.race_distance.into_pyobject(py)?.into_any(),
            player.laps_completed_distance.into_pyobject(py)?.into_any(),
            player.lap_distance.into_pyobject(py)?.into_any(),
            player.race_distance_position.into_pyobject(py)?.into_any(),
            player.race_time_ms.into_pyobject(py)?.into_any(),
            player.lap.into_pyobject(py)?.into_any(),
            player.laps_completed.into_pyobject(py)?.into_any(),
            player.position.into_pyobject(py)?.into_any(),
            player.character.into_pyobject(py)?.into_any(),
            player.machine_index.into_pyobject(py)?.into_any(),
        ],
    )
}
