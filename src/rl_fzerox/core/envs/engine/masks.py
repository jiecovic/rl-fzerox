# src/rl_fzerox/core/envs/engine/masks.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from fzerox_emulator.arrays import ActionMask
from rl_fzerox.core.config.schema import CurriculumConfig
from rl_fzerox.core.domain.hybrid_action import HYBRID_DISCRETE_ACTION_KEY
from rl_fzerox.core.envs.actions import ActionAdapter, ActionValue
from rl_fzerox.core.envs.actions.base import DiscreteActionDimension
from rl_fzerox.core.envs.actions.hybrid.layouts import PITCH_BUCKETS

ActionMaskOverrides: TypeAlias = dict[str, tuple[int, ...]]
ActionMaskBranches: TypeAlias = dict[str, tuple[bool, ...]]


@dataclass(frozen=True, slots=True)
class ActionMaskSnapshot:
    """One decision-time action mask and its branch view."""

    flat: ActionMask
    branches: ActionMaskBranches


@dataclass(slots=True)
class ActionMaskController:
    """Compose static, curriculum, and live gameplay action masks."""

    adapter: ActionAdapter
    base_overrides: ActionMaskOverrides | None
    stage_overrides: tuple[ActionMaskOverrides | None, ...]
    stage_names: tuple[str, ...]
    stage_lean_unmask_min_speed_kph: tuple[float | None, ...]
    stage_boost_unmask_max_speed_kph: tuple[float | None, ...]
    stage_boost_min_energy_fraction: tuple[float | None, ...]
    boost_unmask_max_speed_kph: float | None
    lean_unmask_min_speed_kph: float | None
    pitch_neutral_index: int = PITCH_BUCKETS.neutral_index
    _stage_index: int | None = None
    _boost_unlocked: bool | None = None
    _lean_allowed_values: tuple[int, ...] | None = None
    _speed_kph: float | None = None
    _airborne: bool | None = None

    @classmethod
    def from_config(
        cls,
        *,
        adapter: ActionAdapter,
        base_overrides: ActionMaskOverrides | None,
        curriculum_config: CurriculumConfig | None,
        boost_unmask_max_speed_kph: float | None = None,
        lean_unmask_min_speed_kph: float | None = None,
    ) -> ActionMaskController:
        stage_overrides = _curriculum_stage_overrides(curriculum_config)
        _validate_configured_overrides(
            adapter=adapter,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
        )
        return cls(
            adapter=adapter,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            stage_names=_curriculum_stage_names(curriculum_config),
            stage_lean_unmask_min_speed_kph=_curriculum_stage_lean_gates(curriculum_config),
            stage_boost_unmask_max_speed_kph=_curriculum_stage_boost_speed_gates(curriculum_config),
            stage_boost_min_energy_fraction=_curriculum_stage_boost_energy_gates(curriculum_config),
            boost_unmask_max_speed_kph=boost_unmask_max_speed_kph,
            lean_unmask_min_speed_kph=lean_unmask_min_speed_kph,
            _stage_index=0 if stage_overrides else None,
        )

    def action_mask(self) -> ActionMask:
        """Return the flattened boolean mask for the current action adapter."""

        stage_overrides = None
        if self._stage_index is not None:
            stage_overrides = self.stage_overrides[self._stage_index]
        lean_unmask_min_speed_kph = self.lean_unmask_min_speed_kph
        if self._stage_index is not None:
            stage_lean_gate = self.stage_lean_unmask_min_speed_kph[self._stage_index]
            if stage_lean_gate is not None:
                lean_unmask_min_speed_kph = stage_lean_gate
        return self.adapter.action_mask(
            base_overrides=self.base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=_dynamic_action_mask_overrides(
                boost_unlocked=self._boost_unlocked,
                airborne=self._airborne,
                lean_allowed_values=self._lean_allowed_values,
                speed_kph=self._speed_kph,
                lean_unmask_min_speed_kph=lean_unmask_min_speed_kph,
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

    def set_curriculum_stage(self, stage_index: int) -> None:
        """Switch the active curriculum stage for subsequent action masks."""

        if not self.stage_overrides:
            raise RuntimeError("No curriculum stages are configured for this env")
        if not 0 <= stage_index < len(self.stage_overrides):
            raise ValueError(f"Invalid curriculum stage index {stage_index}")
        self._stage_index = int(stage_index)

    def sync_checkpoint_stage(self, stage_index: int | None) -> None:
        """Align the active watch-stage mask with checkpoint metadata.

        Missing metadata falls back to stage 0 when a curriculum exists so the
        env does not keep a stale later stage from a previously watched policy.
        """

        if not self.stage_overrides:
            self._stage_index = None
            return
        if stage_index is None:
            self._stage_index = 0
            return
        self.set_curriculum_stage(stage_index)

    def set_boost_unlocked(self, boost_unlocked: bool | None) -> None:
        """Update live boost availability used by the dynamic action mask."""

        self._boost_unlocked = boost_unlocked

    def set_lean_allowed_values(self, values: tuple[int, ...] | None) -> None:
        """Update live lean restrictions used by lean primitive semantics."""

        self._lean_allowed_values = values

    def set_speed_kph(self, speed_kph: float | None) -> None:
        """Update the live speed used by dynamic speed-gated masks."""

        self._speed_kph = speed_kph

    def set_airborne(self, airborne: bool | None) -> None:
        """Update live airborne state for air-only input masks."""

        self._airborne = airborne

    @property
    def stage_index(self) -> int | None:
        """Return the active curriculum stage index, if any."""

        return self._stage_index

    @property
    def stage_name(self) -> str | None:
        """Return the active curriculum stage name, if any."""

        if self._stage_index is None:
            return None
        return self.stage_names[self._stage_index]

    @property
    def current_boost_unmask_max_speed_kph(self) -> float | None:
        if self._stage_index is None:
            return self.boost_unmask_max_speed_kph
        stage_gate = self.stage_boost_unmask_max_speed_kph[self._stage_index]
        if stage_gate is not None:
            return stage_gate
        return self.boost_unmask_max_speed_kph

    def current_boost_min_energy_fraction(self, default: float) -> float:
        if self._stage_index is None:
            return float(default)
        stage_gate = self.stage_boost_min_energy_fraction[self._stage_index]
        return float(default if stage_gate is None else stage_gate)


def _curriculum_stage_overrides(
    curriculum_config: CurriculumConfig | None,
) -> tuple[ActionMaskOverrides | None, ...]:
    if curriculum_config is None or not curriculum_config.enabled:
        return ()
    return tuple(
        stage.action_mask.branch_overrides() if stage.action_mask is not None else None
        for stage in curriculum_config.stages
    )


def _curriculum_stage_names(curriculum_config: CurriculumConfig | None) -> tuple[str, ...]:
    if curriculum_config is None or not curriculum_config.enabled:
        return ()
    return tuple(stage.name for stage in curriculum_config.stages)


def _curriculum_stage_lean_gates(
    curriculum_config: CurriculumConfig | None,
) -> tuple[float | None, ...]:
    if curriculum_config is None or not curriculum_config.enabled:
        return ()
    return tuple(stage.lean_unmask_min_speed_kph for stage in curriculum_config.stages)


def _curriculum_stage_boost_speed_gates(
    curriculum_config: CurriculumConfig | None,
) -> tuple[float | None, ...]:
    if curriculum_config is None or not curriculum_config.enabled:
        return ()
    return tuple(stage.boost_unmask_max_speed_kph for stage in curriculum_config.stages)


def _curriculum_stage_boost_energy_gates(
    curriculum_config: CurriculumConfig | None,
) -> tuple[float | None, ...]:
    if curriculum_config is None or not curriculum_config.enabled:
        return ()
    return tuple(stage.boost_min_energy_fraction for stage in curriculum_config.stages)


def _validate_configured_overrides(
    *,
    adapter: ActionAdapter,
    base_overrides: ActionMaskOverrides | None,
    stage_overrides: tuple[ActionMaskOverrides | None, ...],
) -> None:
    """Reject mask branches that the active action adapter cannot consume."""

    valid_labels = frozenset(dimension.label for dimension in adapter.action_dimensions)
    _validate_override_branches(
        overrides=base_overrides,
        valid_labels=valid_labels,
        source_label="env.action.mask",
    )
    for stage_index, overrides in enumerate(stage_overrides):
        _validate_override_branches(
            overrides=overrides,
            valid_labels=valid_labels,
            source_label=f"curriculum.stages[{stage_index}].action_mask",
        )


def _validate_override_branches(
    *,
    overrides: ActionMaskOverrides | None,
    valid_labels: frozenset[str],
    source_label: str,
) -> None:
    if overrides is None:
        return

    unknown_labels = sorted(set(overrides) - valid_labels)
    if not unknown_labels:
        return

    unknown = ", ".join(repr(label) for label in unknown_labels)
    valid = ", ".join(repr(label) for label in sorted(valid_labels)) or "none"
    raise ValueError(
        f"Unsupported action mask branch in {source_label}: {unknown}. "
        f"Valid branches for this action adapter: {valid}."
    )


def split_action_mask_by_branch(
    dimensions: tuple[DiscreteActionDimension, ...],
    mask: ActionMask,
) -> ActionMaskBranches:
    """Split one flattened MultiDiscrete mask into stable branch labels."""

    flat_mask = tuple(bool(value) for value in mask.reshape(-1).tolist())
    branches: ActionMaskBranches = {}
    cursor = 0
    for dimension in dimensions:
        next_cursor = cursor + dimension.size
        branches[dimension.label] = flat_mask[cursor:next_cursor]
        cursor = next_cursor
    if cursor != len(flat_mask):
        raise RuntimeError(
            "Action mask length does not match adapter dimensions: "
            f"mask={len(flat_mask)}, dimensions={cursor}"
        )
    return branches


def action_branch_value_allowed(
    branches: ActionMaskBranches | None,
    label: str,
    index: int,
    *,
    missing_allowed: bool,
) -> bool:
    """Return whether one discrete branch value is selectable.

    `missing_allowed` is explicit because a missing branch means different
    things at different call sites: continuous gas has no discrete branch but is
    still available, while optional cockpit buttons with no branch are not wired
    into the policy at all.
    """

    if branches is None:
        return True
    branch = branches.get(label)
    if branch is None:
        return missing_allowed
    if not 0 <= index < len(branch):
        return False
    return branch[index]


def action_branch_non_neutral_allowed(
    branches: ActionMaskBranches | None,
    label: str,
    *,
    neutral_index: int,
    missing_allowed: bool,
) -> bool:
    """Return whether any non-neutral value is selectable for one branch."""

    if branches is None:
        return True
    branch = branches.get(label)
    if branch is None:
        return missing_allowed
    return any(allowed for index, allowed in enumerate(branch) if index != neutral_index)


def selected_action_branches(
    branches: ActionMaskBranches,
    action: ActionValue | None,
) -> dict[str, int]:
    """Return selected discrete branch values in action-mask branch order."""

    if action is None or not branches:
        return {}
    discrete_values = _discrete_action_values(action)
    if discrete_values is None:
        return {}
    labels = tuple(branches)
    return {
        label: int(discrete_values[index])
        for index, label in enumerate(labels)
        if index < len(discrete_values)
    }


def action_mask_violations(
    branches: ActionMaskBranches,
    action: ActionValue | None,
) -> tuple[str, ...]:
    """Return branch/value pairs where an action selects a masked value."""

    selected = selected_action_branches(branches, action)
    violations: list[str] = []
    for label, value in selected.items():
        branch = branches[label]
        if not 0 <= value < len(branch) or not branch[value]:
            violations.append(f"{label}={value}")
    return tuple(violations)


def _discrete_action_values(action: ActionValue) -> tuple[int, ...] | None:
    source: object
    if isinstance(action, Mapping):
        source = action.get(HYBRID_DISCRETE_ACTION_KEY)
        if source is None:
            return None
    else:
        source = action
    values = np.asarray(source).reshape(-1)
    if values.size == 0:
        return None
    return tuple(int(value) for value in values)


def _dynamic_action_mask_overrides(
    *,
    boost_unlocked: bool | None,
    airborne: bool | None = None,
    lean_allowed_values: tuple[int, ...] | None = None,
    speed_kph: float | None = None,
    lean_unmask_min_speed_kph: float | None = None,
    pitch_neutral_index: int = PITCH_BUCKETS.neutral_index,
) -> ActionMaskOverrides | None:
    overrides: ActionMaskOverrides = {}
    # `None` means we do not yet have live telemetry for the current episode.
    # In that case keep the branch open instead of masking boost prematurely.
    if boost_unlocked is False:
        overrides["boost"] = (0,)
    lean_values = lean_allowed_values
    if lean_values is None:
        if (
            lean_unmask_min_speed_kph is not None
            and speed_kph is not None
            and speed_kph < lean_unmask_min_speed_kph
        ):
            lean_values = (0,)
    if lean_values is not None:
        overrides["lean"] = lean_values
    if airborne is False:
        overrides["air_brake"] = (0,)
        overrides["pitch"] = (pitch_neutral_index,)
    if not overrides:
        return None
    return overrides
