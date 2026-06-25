# src/rl_fzerox/core/envs/engine/controls/masks.py
"""Mutable action-mask controller for one active env runtime.

This file owns the current allowed values for each action branch and projects
them into Gym-compatible mask arrays for policy and watch consumers.
"""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator.arrays import ActionMask
from rl_fzerox.core.envs.actions import ActionAdapter

from .mask_config import (
    ActionMaskBranches,
    ActionMaskOverrides,
    ActionMaskSnapshot,
    validate_configured_overrides,
)
from .mask_queries import split_action_mask_by_branch


@dataclass(slots=True)
class ActionMaskController:
    """Compose static and live gameplay action masks."""

    adapter: ActionAdapter
    base_overrides: ActionMaskOverrides | None
    boost_unmask_max_speed_kph: float | None
    lean_unmask_min_speed_kph: float | None
    mask_air_brake_on_ground: bool
    mask_pitch_on_ground: bool
    pitch_neutral_index: int = 2
    _boost_unlocked: bool | None = None
    _lean_allowed_values: tuple[int, ...] | None = None
    _air_brake_allowed_values: tuple[int, ...] | None = None
    _spin_allowed_values: tuple[int, ...] | None = None
    _lean_episode_masked: bool = False
    _air_brake_episode_masked: bool = False
    _spin_episode_masked: bool = False
    _speed_kph: float | None = None
    _airborne: bool | None = None

    @classmethod
    def from_config(
        cls,
        *,
        adapter: ActionAdapter,
        base_overrides: ActionMaskOverrides | None,
        boost_unmask_max_speed_kph: float | None = None,
        lean_unmask_min_speed_kph: float | None = None,
        mask_air_brake_on_ground: bool = True,
        mask_pitch_on_ground: bool = True,
        pitch_neutral_index: int = 2,
    ) -> ActionMaskController:
        validate_configured_overrides(
            adapter=adapter,
            base_overrides=base_overrides,
        )
        return cls(
            adapter=adapter,
            base_overrides=base_overrides,
            boost_unmask_max_speed_kph=boost_unmask_max_speed_kph,
            lean_unmask_min_speed_kph=lean_unmask_min_speed_kph,
            mask_air_brake_on_ground=bool(mask_air_brake_on_ground),
            mask_pitch_on_ground=bool(mask_pitch_on_ground),
            pitch_neutral_index=int(pitch_neutral_index),
        )

    def action_mask(self) -> ActionMask:
        """Return the flattened boolean mask for the current action adapter."""

        return self.adapter.action_mask(
            base_overrides=self.base_overrides,
            stage_overrides=None,
            dynamic_overrides=_dynamic_action_mask_overrides(
                boost_unlocked=self._boost_unlocked,
                airborne=self._airborne,
                lean_allowed_values=self._lean_allowed_values,
                air_brake_allowed_values=self._air_brake_allowed_values,
                spin_allowed_values=self._spin_allowed_values,
                lean_episode_masked=self._lean_episode_masked,
                air_brake_episode_masked=self._air_brake_episode_masked,
                spin_episode_masked=self._spin_episode_masked,
                speed_kph=self._speed_kph,
                lean_unmask_min_speed_kph=self.lean_unmask_min_speed_kph,
                mask_air_brake_on_ground=self.mask_air_brake_on_ground,
                mask_pitch_on_ground=self.mask_pitch_on_ground,
                pitch_neutral_index=self.pitch_neutral_index,
            ),
        )

    def action_mask_branches(self) -> ActionMaskBranches:
        """Return the current action mask grouped by adapter branch label."""

        return self.action_mask_snapshot().branches

    def action_mask_snapshot(self) -> ActionMaskSnapshot:
        """Return the flat mask and branch view from one mask computation."""

        mask = self.action_mask()
        return ActionMaskSnapshot(
            flat=mask,
            branches=split_action_mask_by_branch(self.adapter.action_dimensions, mask),
        )

    def control_gate_action_mask_branches(self) -> ActionMaskBranches:
        """Return action masks used to suppress unavailable controls.

        Active air-brake pulses mask new policy requests but should not by
        themselves suppress the held effective brake button. Gameplay gates such
        as episode masks, grounded air-only masks, and static config masks still
        apply through this view.
        """

        air_brake_allowed_values = self._air_brake_allowed_values
        self._air_brake_allowed_values = None
        try:
            return self.action_mask_branches()
        finally:
            self._air_brake_allowed_values = air_brake_allowed_values

    def set_boost_unlocked(self, boost_unlocked: bool | None) -> None:
        """Update live boost availability used by the dynamic action mask."""

        self._boost_unlocked = boost_unlocked

    def set_lean_allowed_values(self, values: tuple[int, ...] | None) -> None:
        """Update live lean restrictions used by lean primitive semantics."""

        self._lean_allowed_values = values

    def set_air_brake_allowed_values(self, values: tuple[int, ...] | None) -> None:
        """Update live air-brake restrictions used by pulse semantics."""

        self._air_brake_allowed_values = values

    def set_spin_allowed_values(self, values: tuple[int, ...] | None) -> None:
        """Update live spin restrictions used by the native spin macro."""

        self._spin_allowed_values = values

    def set_lean_episode_masked(self, masked: bool) -> None:
        """Force lean and lean-backed spin neutral for the current episode."""

        self._lean_episode_masked = bool(masked)

    def set_air_brake_episode_masked(self, masked: bool) -> None:
        """Force air-brake neutral for the current episode."""

        self._air_brake_episode_masked = bool(masked)

    def set_spin_episode_masked(self, masked: bool) -> None:
        """Force native spin requests neutral for the current episode."""

        self._spin_episode_masked = bool(masked)

    def set_speed_kph(self, speed_kph: float | None) -> None:
        """Update the live speed used by dynamic speed-gated masks."""

        self._speed_kph = speed_kph

    def set_airborne(self, airborne: bool | None) -> None:
        """Update live airborne state for air-only input masks."""

        self._airborne = airborne

    @property
    def current_boost_unmask_max_speed_kph(self) -> float | None:
        return self.boost_unmask_max_speed_kph

    def current_boost_min_energy_fraction(self, default: float) -> float:
        return float(default)

    @property
    def lean_episode_masked(self) -> bool:
        return self._lean_episode_masked

    @property
    def air_brake_episode_masked(self) -> bool:
        return self._air_brake_episode_masked

    @property
    def spin_episode_masked(self) -> bool:
        return self._spin_episode_masked


def _dynamic_action_mask_overrides(
    *,
    boost_unlocked: bool | None,
    airborne: bool | None = None,
    lean_allowed_values: tuple[int, ...] | None = None,
    air_brake_allowed_values: tuple[int, ...] | None = None,
    spin_allowed_values: tuple[int, ...] | None = None,
    lean_episode_masked: bool = False,
    air_brake_episode_masked: bool = False,
    spin_episode_masked: bool = False,
    speed_kph: float | None = None,
    lean_unmask_min_speed_kph: float | None = None,
    mask_air_brake_on_ground: bool = True,
    mask_pitch_on_ground: bool = True,
    pitch_neutral_index: int = 2,
) -> ActionMaskOverrides | None:
    overrides: ActionMaskOverrides = {}
    # `None` means we do not yet have live telemetry for the current episode.
    # In that case keep the branch open instead of masking boost prematurely.
    if boost_unlocked is False:
        overrides["boost"] = (0,)
    lean_values = (0,) if lean_episode_masked else lean_allowed_values
    if lean_values is None:
        if (
            lean_unmask_min_speed_kph is not None
            and speed_kph is not None
            and speed_kph < lean_unmask_min_speed_kph
        ):
            lean_values = (0,)
    if lean_values is not None:
        overrides["lean"] = lean_values
        if lean_values == (0,):
            overrides["lean_left"] = (0,)
            overrides["lean_right"] = (0,)
    if spin_episode_masked or lean_episode_masked:
        overrides["spin"] = (0,)
    elif spin_allowed_values is not None:
        overrides["spin"] = spin_allowed_values
    if air_brake_episode_masked or (airborne is False and mask_air_brake_on_ground):
        overrides["air_brake"] = (0,)
    elif air_brake_allowed_values is not None:
        overrides["air_brake"] = air_brake_allowed_values
    if airborne is False and mask_pitch_on_ground:
        overrides["pitch"] = (pitch_neutral_index,)
    if not overrides:
        return None
    return overrides
