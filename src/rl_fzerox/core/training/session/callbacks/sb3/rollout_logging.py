# src/rl_fzerox/core/training/session/callbacks/sb3/rollout_logging.py
from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from rl_fzerox.core.training.session.callbacks.metrics import (
    RolloutInfoAccumulator,
    info_sequence,
)


class InfoLoggingCallback(BaseCallback):
    """Log rollout-aggregated state means and episode outcomes."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._rollout_info = RolloutInfoAccumulator()

    def _on_rollout_start(self) -> None:
        self._rollout_info = RolloutInfoAccumulator()

    def _on_step(self) -> bool:
        infos = info_sequence(self.locals.get("infos"))
        if infos is None:
            return True

        self._rollout_info.add_infos(infos)
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_info.record_to(self.logger)
