// rust/bindings/emulator/state.rs
//! Shared racer-state flag helpers for Python-facing telemetry and step objects.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct RacerStateFlagSpec {
    pub mask: u32,
    pub label: &'static str,
}

pub(super) const FLAG_COLLISION_RECOIL: u32 = 1 << 13;
pub(super) const FLAG_SPINNING_OUT: u32 = 1 << 14;
pub(super) const FLAG_RETIRED: u32 = 1 << 18;
pub(super) const FLAG_FALLING_OFF_TRACK: u32 = 1 << 19;
pub(super) const FLAG_CAN_BOOST: u32 = 1 << 20;
pub(super) const FLAG_CPU_CONTROLLED: u32 = 1 << 23;
pub(super) const FLAG_DASH_PAD_BOOST: u32 = 1 << 24;
pub(super) const FLAG_FINISHED: u32 = 1 << 25;
pub(super) const FLAG_AIRBORNE: u32 = 1 << 26;
pub(super) const FLAG_CRASHED: u32 = 1 << 27;
pub(super) const FLAG_ACTIVE: u32 = 1 << 30;

pub(super) const RACER_STATE_FLAG_SPECS: [RacerStateFlagSpec; 11] = [
    RacerStateFlagSpec {
        mask: FLAG_COLLISION_RECOIL,
        label: "collision_recoil",
    },
    RacerStateFlagSpec {
        mask: FLAG_SPINNING_OUT,
        label: "spinning_out",
    },
    RacerStateFlagSpec {
        mask: FLAG_RETIRED,
        label: "retired",
    },
    RacerStateFlagSpec {
        mask: FLAG_FALLING_OFF_TRACK,
        label: "falling_off_track",
    },
    RacerStateFlagSpec {
        mask: FLAG_CAN_BOOST,
        label: "can_boost",
    },
    RacerStateFlagSpec {
        mask: FLAG_CPU_CONTROLLED,
        label: "cpu_controlled",
    },
    RacerStateFlagSpec {
        mask: FLAG_DASH_PAD_BOOST,
        label: "dash_pad_boost",
    },
    RacerStateFlagSpec {
        mask: FLAG_FINISHED,
        label: "finished",
    },
    RacerStateFlagSpec {
        mask: FLAG_AIRBORNE,
        label: "airborne",
    },
    RacerStateFlagSpec {
        mask: FLAG_CRASHED,
        label: "crashed",
    },
    RacerStateFlagSpec {
        mask: FLAG_ACTIVE,
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

pub(super) fn terminal_reason(state_flags: u32) -> Option<&'static str> {
    [
        ("finished", FLAG_FINISHED),
        ("crashed", FLAG_CRASHED),
        ("retired", FLAG_RETIRED),
        ("falling_off_track", FLAG_FALLING_OFF_TRACK),
    ]
    .into_iter()
    .find_map(|(label, mask)| has_state_flag(state_flags, mask).then_some(label))
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
