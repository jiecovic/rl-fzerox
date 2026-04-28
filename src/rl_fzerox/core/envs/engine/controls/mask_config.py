from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from fzerox_emulator.arrays import ActionMask
from rl_fzerox.core.config.schema import CurriculumConfig
from rl_fzerox.core.envs.actions import ActionAdapter

ActionMaskOverrides: TypeAlias = dict[str, tuple[int, ...]]
ActionMaskBranches: TypeAlias = dict[str, tuple[bool, ...]]


@dataclass(frozen=True, slots=True)
class ActionMaskSnapshot:
    """One decision-time action mask and its branch view."""

    flat: ActionMask
    branches: ActionMaskBranches


def curriculum_stage_overrides(
    curriculum_config: CurriculumConfig | None,
) -> tuple[ActionMaskOverrides | None, ...]:
    if curriculum_config is None or not curriculum_config.enabled:
        return ()
    return tuple(
        stage.action_mask.branch_overrides() if stage.action_mask is not None else None
        for stage in curriculum_config.stages
    )


def curriculum_stage_names(curriculum_config: CurriculumConfig | None) -> tuple[str, ...]:
    if curriculum_config is None or not curriculum_config.enabled:
        return ()
    return tuple(stage.name for stage in curriculum_config.stages)


def curriculum_stage_lean_gates(
    curriculum_config: CurriculumConfig | None,
) -> tuple[float | None, ...]:
    if curriculum_config is None or not curriculum_config.enabled:
        return ()
    return tuple(stage.lean_unmask_min_speed_kph for stage in curriculum_config.stages)


def curriculum_stage_boost_speed_gates(
    curriculum_config: CurriculumConfig | None,
) -> tuple[float | None, ...]:
    if curriculum_config is None or not curriculum_config.enabled:
        return ()
    return tuple(stage.boost_unmask_max_speed_kph for stage in curriculum_config.stages)


def curriculum_stage_boost_energy_gates(
    curriculum_config: CurriculumConfig | None,
) -> tuple[float | None, ...]:
    if curriculum_config is None or not curriculum_config.enabled:
        return ()
    return tuple(stage.boost_min_energy_fraction for stage in curriculum_config.stages)


def validate_configured_overrides(
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
