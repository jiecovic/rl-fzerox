# src/rl_fzerox/core/game/flags.py
from __future__ import annotations

FLAG_COLLISION_RECOIL = 1 << 13
FLAG_SPINNING_OUT = 1 << 14
FLAG_RETIRED = 1 << 18
FLAG_FALLING_OFF_TRACK = 1 << 19
FLAG_CAN_BOOST = 1 << 20
FLAG_CPU_CONTROLLED = 1 << 23
FLAG_DASH_PAD_BOOST = 1 << 24
FLAG_FINISHED = 1 << 25
FLAG_AIRBORNE = 1 << 26
FLAG_CRASHED = 1 << 27
FLAG_ACTIVE = 1 << 30

RACER_FLAG_LABELS: tuple[tuple[int, str], ...] = (
    (FLAG_COLLISION_RECOIL, "collision_recoil"),
    (FLAG_SPINNING_OUT, "spinning_out"),
    (FLAG_RETIRED, "retired"),
    (FLAG_FALLING_OFF_TRACK, "falling_off_track"),
    (FLAG_CAN_BOOST, "can_boost"),
    (FLAG_CPU_CONTROLLED, "cpu_controlled"),
    (FLAG_DASH_PAD_BOOST, "dash_pad_boost"),
    (FLAG_FINISHED, "finished"),
    (FLAG_AIRBORNE, "airborne"),
    (FLAG_CRASHED, "crashed"),
    (FLAG_ACTIVE, "active"),
)


def decode_racer_flags(state_flags: int) -> tuple[str, ...]:
    """Return semantic labels for one raw racer state bitfield."""

    return tuple(label for bit, label in RACER_FLAG_LABELS if state_flags & bit)
