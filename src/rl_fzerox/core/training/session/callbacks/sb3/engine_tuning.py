# src/rl_fzerox/core/training/session/callbacks/sb3/engine_tuning.py
"""SB3 callback bridge for online engine-tuning episode updates."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from rl_fzerox.core.engine_tuning import EngineTuningResetSampler
from rl_fzerox.core.engine_tuning.training import EngineTuningTrainingSession
from rl_fzerox.core.training.session.callbacks.metrics import episode_dicts, info_sequence
from rl_fzerox.core.training.session.callbacks.sb3.env_adapter import training_env_adapter


class EngineTuningCallback(BaseCallback):
    """Update adaptive engine-setting stats from completed episodes."""

    def __init__(
        self,
        *,
        session: EngineTuningTrainingSession,
    ) -> None:
        super().__init__(verbose=0)
        self._session = session

    def _on_training_start(self) -> None:
        self._publish_sampler(self._session.initial_sampler_snapshot())

    def _on_step(self) -> bool:
        infos = info_sequence(self.locals.get("infos"))
        if infos is None:
            return True
        episodes = episode_dicts(infos)
        if not episodes:
            return True
        self._session.record_episodes(episodes)
        return True

    def _on_rollout_end(self) -> None:
        update = self._session.finish_rollout()
        if update.sampler_snapshot is not None:
            self._publish_sampler(update.sampler_snapshot)
        for key, value in update.log_values.items():
            self.logger.record(key, value)

    def _publish_sampler(self, sampler: EngineTuningResetSampler) -> None:
        training_env_adapter(self.training_env).set_engine_tuning_sampler(sampler)
