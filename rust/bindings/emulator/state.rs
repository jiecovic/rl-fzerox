// rust/bindings/emulator/state.rs
//! Shared racer-state flag helpers for Python-facing telemetry and step objects.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct RacerStateFlagSpec {
    pub mask: u32,
    pub label: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RacerStateFlags {
    pub collision_recoil: u32,
    pub spinning_out: u32,
    pub retired: u32,
    pub falling_off_track: u32,
    pub can_boost: u32,
    pub cpu_controlled: u32,
    pub dash_pad_boost: u32,
    pub finished: u32,
    pub airborne: u32,
    pub crashed: u32,
    pub active: u32,
}

pub(super) const RACER_STATE_FLAGS: RacerStateFlags = RacerStateFlags {
    collision_recoil: 1 << 13,
    spinning_out: 1 << 14,
    retired: 1 << 18,
    falling_off_track: 1 << 19,
    can_boost: 1 << 20,
    cpu_controlled: 1 << 23,
    dash_pad_boost: 1 << 24,
    finished: 1 << 25,
    airborne: 1 << 26,
    crashed: 1 << 27,
    active: 1 << 30,
};

pub(super) const RACER_STATE_FLAG_SPECS: [RacerStateFlagSpec; 11] = [
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.collision_recoil,
        label: "collision_recoil",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.spinning_out,
        label: "spinning_out",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.retired,
        label: "retired",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.falling_off_track,
        label: "falling_off_track",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.can_boost,
        label: "can_boost",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.cpu_controlled,
        label: "cpu_controlled",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.dash_pad_boost,
        label: "dash_pad_boost",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.finished,
        label: "finished",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.airborne,
        label: "airborne",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.crashed,
        label: "crashed",
    },
    RacerStateFlagSpec {
        mask: RACER_STATE_FLAGS.active,
        label: "active",
    },
];

pub(super) fn has_state_flag(state_flags: u32, flag: u32) -> bool {
    (state_flags & flag) != 0
}

pub(super) fn state_flag_labels(state_flags: u32) -> Vec<&'static str> {
    RACER_STATE_FLAG_SPECS
        .iter()
        .filter(|spec| has_state_flag(state_flags, spec.mask))
        .map(|spec| spec.label)
        .collect()
}

#[pyfunction]
pub fn encode_state_flags(labels: Vec<String>) -> PyResult<u32> {
    let mut state_flags = 0u32;
    for label in labels {
        let Some(mask) = RACER_STATE_FLAG_SPECS
            .iter()
            .find(|spec| spec.label == label)
            .map(|spec| spec.mask)
        else {
            return Err(PyValueError::new_err(format!(
                "Unknown racer state label: {label}"
            )));
        };
        state_flags |= mask;
    }
    Ok(state_flags)
}
