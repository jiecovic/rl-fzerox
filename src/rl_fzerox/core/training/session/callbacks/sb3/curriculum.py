# src/rl_fzerox/core/training/session/callbacks/sb3/curriculum.py
from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from rl_fzerox.core.runtime_spec.schema import CurriculumConfig, EnvConfig
from rl_fzerox.core.training.session.callbacks.metrics import episode_dicts, info_sequence
from rl_fzerox.core.training.session.callbacks.tuning import (
    apply_stage_train_overrides,
    record_stage_train_overrides,
)
from rl_fzerox.core.training.session.curriculum import ActionMaskCurriculumController


class CurriculumCallback(BaseCallback):
    """Promote curriculum stages and apply their rollout-time overrides."""

    def __init__(
        self,
        curriculum: CurriculumConfig,
        *,
        env_config: EnvConfig | None = None,
        initial_stage_index: int | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self._controller = ActionMaskCurriculumController(
            curriculum,
            env_config=env_config,
            initial_stage_index=initial_stage_index,
        )

    def _on_training_start(self) -> None:
        self._apply_current_stage()

    def _on_step(self) -> bool:
        infos = info_sequence(self.locals.get("infos"))
        if infos is None:
            return True

        promoted_stage = self._controller.record_episodes(episode_dicts(infos))
        if promoted_stage is not None:
            self._apply_current_stage()
        return True

    def _on_rollout_end(self) -> None:
        stage_index = self._controller.stage_index
        self.logger.record(
            "curriculum/stage",
            -1 if stage_index is None else stage_index,
        )
        record_stage_train_overrides(
            logger=self.logger,
            overrides=self._controller.stage_train_overrides,
        )

    def _apply_current_stage(self) -> None:
        stage_index = self._controller.stage_index
        if stage_index is None:
            return
        self.training_env.env_method("set_curriculum_stage", stage_index)
        apply_stage_train_overrides(
            model=self.model,
            overrides=self._controller.stage_train_overrides,
        )
