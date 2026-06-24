# src/rl_fzerox/core/training/session/callbacks/sb3/engine_tuning.py
from __future__ import annotations

from collections.abc import Sequence

from stable_baselines3.common.callbacks import BaseCallback

from rl_fzerox.core.engine_tuning import EngineTuningContext
from rl_fzerox.core.engine_tuning.training import EngineTuningTrainingController
from rl_fzerox.core.envs.engine.reset.track_sampling import engine_tuning_context_for_entry
from rl_fzerox.core.runtime_spec.schema import EnvConfig
from rl_fzerox.core.training.session.callbacks.metrics import episode_dicts, info_sequence


class EngineTuningCallback(BaseCallback):
    """Update adaptive engine-setting stats from completed episodes."""

    def __init__(
        self,
        *,
        controller: EngineTuningTrainingController,
        contexts: Sequence[EngineTuningContext],
    ) -> None:
        super().__init__(verbose=0)
        self._controller = controller
        self._contexts = tuple(contexts)
        self._sampler_dirty = False

    def _on_training_start(self) -> None:
        self._publish_sampler()

    def _on_step(self) -> bool:
        infos = info_sequence(self.locals.get("infos"))
        if infos is None:
            return True
        episodes = episode_dicts(infos)
        if not episodes:
            return True
        if self._controller.record_episodes(episodes):
            self._sampler_dirty = True
        return True

    def _on_rollout_end(self) -> None:
        rollout_changed = self._controller.record_rollout_episodes()
        if self._sampler_dirty or rollout_changed:
            self._publish_sampler()
            self._sampler_dirty = False
        for key, value in self._controller.log_values().items():
            self.logger.record(key, value)

    def _publish_sampler(self) -> None:
        self.training_env.env_method(
            "set_engine_tuning_sampler",
            self._controller.reset_sampler_snapshot(self._contexts),
        )


def engine_tuning_contexts(env_config: EnvConfig) -> tuple[EngineTuningContext, ...]:
    contexts: dict[str, EngineTuningContext] = {}
    for entry in env_config.track_sampling.entries:
        context = engine_tuning_context_for_entry(entry)
        contexts.setdefault(context.key, context)
    return tuple(contexts[key] for key in sorted(contexts))
