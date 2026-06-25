# src/rl_fzerox/core/envs/engine/controls/mask_config.py
"""Typed containers for action-mask branch metadata.

These small structures keep raw mask arrays connected to branch names and values
so env, watch, and policy-drive code can inspect masks without guessing indices.
"""

from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator.arrays import ActionMask
from rl_fzerox.core.envs.actions import ActionAdapter

type ActionMaskOverrides = dict[str, tuple[int, ...]]
type ActionMaskBranches = dict[str, tuple[bool, ...]]


@dataclass(frozen=True, slots=True)
class ActionMaskSnapshot:
    """One decision-time action mask and its branch view."""

    flat: ActionMask
    branches: ActionMaskBranches


def validate_configured_overrides(
    *,
    adapter: ActionAdapter,
    base_overrides: ActionMaskOverrides | None,
) -> None:
    """Reject mask branches that the active action adapter cannot consume."""

    valid_labels = frozenset(dimension.label for dimension in adapter.action_dimensions)
    _validate_override_branches(
        overrides=base_overrides,
        valid_labels=valid_labels,
        source_label="env.action.mask",
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
