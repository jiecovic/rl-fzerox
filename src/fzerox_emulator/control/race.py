# src/fzerox_emulator/control/race.py
"""F-Zero X gameplay controls used by the repeated-step boundary.

This is the semantic control layer: Python code describes gameplay intent here,
while Rust maps that intent onto Mupen64Plus-Next/libretro controller details.
For example, ``air_brake=True`` means F-Zero X air brake, not "RetroPad A" or
"N64 C-Down" at the Python call site.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RaceControlMaskCatalog:
    """Compact semantic masks used only for UI/history display state.

    These values are intentionally project-owned. They are not libretro button
    ids and must not be passed to the raw ``set_controller_state`` API.
    """

    gas: int = 1 << 0
    air_brake: int = 1 << 1
    boost: int = 1 << 2
    lean_left: int = 1 << 3
    lean_right: int = 1 << 4

    @property
    def accelerate(self) -> int:
        """Alias for the gas button used by existing env/action code."""

        return self.gas

    @property
    def lean(self) -> int:
        return self.lean_left | self.lean_right


RACE_CONTROL_MASKS = RaceControlMaskCatalog()


@dataclass(frozen=True, slots=True)
class RaceControlState:
    """One held F-Zero X gameplay control request.

    ``stick_x`` and ``pitch`` are normalized to ``[-1, 1]``. Button fields are
    gameplay-level controls. Rust owns the lower-level mapping:

    - gas -> N64 A
    - air brake -> N64 C-Down
    - boost -> N64 B
    - lean left -> N64 L
    - lean right -> N64 R
    """

    gas: bool = False
    air_brake: bool = False
    boost: bool = False
    lean_left: bool = False
    lean_right: bool = False
    stick_x: float = 0.0
    pitch: float = 0.0

    @classmethod
    def from_mask(
        cls,
        control_mask: int,
        *,
        stick_x: float = 0.0,
        pitch: float = 0.0,
    ) -> RaceControlState:
        """Build a semantic state from a project-owned control mask."""

        return cls(
            gas=bool(control_mask & RACE_CONTROL_MASKS.gas),
            air_brake=bool(control_mask & RACE_CONTROL_MASKS.air_brake),
            boost=bool(control_mask & RACE_CONTROL_MASKS.boost),
            lean_left=bool(control_mask & RACE_CONTROL_MASKS.lean_left),
            lean_right=bool(control_mask & RACE_CONTROL_MASKS.lean_right),
            stick_x=stick_x,
            pitch=pitch,
        )

    @property
    def control_mask(self) -> int:
        """Return the project-owned semantic mask for UI/history consumers."""

        mask = 0
        if self.gas:
            mask |= RACE_CONTROL_MASKS.gas
        if self.air_brake:
            mask |= RACE_CONTROL_MASKS.air_brake
        if self.boost:
            mask |= RACE_CONTROL_MASKS.boost
        if self.lean_left:
            mask |= RACE_CONTROL_MASKS.lean_left
        if self.lean_right:
            mask |= RACE_CONTROL_MASKS.lean_right
        return mask
