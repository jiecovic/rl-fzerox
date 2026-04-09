# src/rl_fzerox/core/envs/engine/masks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from rl_fzerox.core.config.schema import CurriculumConfig
from rl_fzerox.core.envs.actions import ActionAdapter

ActionMaskOverrides: TypeAlias = dict[str, tuple[int, ...]]


@dataclass(slots=True)
class ActionMaskController:
    """Compose static, curriculum, and live gameplay action masks."""

    adapter: ActionAdapter
    base_overrides: ActionMaskOverrides | None
    stage_overrides: tuple[ActionMaskOverrides | None, ...]
    stage_names: tuple[str, ...]
    _stage_index: int | None = None
    _boost_unlocked: bool | None = None

    @classmethod
    def from_config(
        cls,
        *,
        adapter: ActionAdapter,
        base_overrides: ActionMaskOverrides | None,
        curriculum_config: CurriculumConfig | None,
    ) -> ActionMaskController:
        stage_overrides = _curriculum_stage_overrides(curriculum_config)
        return cls(
            adapter=adapter,
            base_overrides=base_overrides,
            stage_overrides=stage_overrides,
            stage_names=_curriculum_stage_names(curriculum_config),
            _stage_index=0 if stage_overrides else None,
        )

    def action_mask(self) -> np.ndarray:
        """Return the flattened boolean mask for the current action adapter."""

        stage_overrides = None
        if self._stage_index is not None:
            stage_overrides = self.stage_overrides[self._stage_index]
        return self.adapter.action_mask(
            base_overrides=self.base_overrides,
            stage_overrides=stage_overrides,
            dynamic_overrides=_dynamic_action_mask_overrides(
                boost_unlocked=self._boost_unlocked,
            ),
        )

    def set_curriculum_stage(self, stage_index: int) -> None:
        """Switch the active curriculum stage for subsequent action masks."""

        if not self.stage_overrides:
            raise RuntimeError("No curriculum stages are configured for this env")
        if not 0 <= stage_index < len(self.stage_overrides):
            raise ValueError(f"Invalid curriculum stage index {stage_index}")
        self._stage_index = int(stage_index)

    def set_boost_unlocked(self, boost_unlocked: bool | None) -> None:
        """Update live boost availability used by the dynamic action mask."""

        self._boost_unlocked = boost_unlocked

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


def _dynamic_action_mask_overrides(
    *,
    boost_unlocked: bool | None,
) -> ActionMaskOverrides | None:
    # `None` means we do not yet have live telemetry for the current episode.
    # In that case keep the branch open instead of masking boost prematurely.
    if boost_unlocked is not False:
        return None
    return {"boost": (0,)}
