# src/rl_fzerox/core/training/session/callbacks/sb3/artifacts.py
"""SB3 callbacks that persist latest/best policy artifacts."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from rl_fzerox.core.engine_tuning import EngineTuningRuntimeState
from rl_fzerox.core.engine_tuning.training import EngineTuningTrainingController
from rl_fzerox.core.training.runs import RunPaths
from rl_fzerox.core.training.session.artifacts import (
    current_policy_artifact_metadata,
    save_artifacts_atomically,
    save_recent_checkpoint_artifacts,
    trim_recent_checkpoint_artifacts,
)
from rl_fzerox.core.training.session.callbacks.checkpoints import CheckpointPolicy
from rl_fzerox.core.training.session.callbacks.metrics import info_sequence


class RollingArtifactCallback(BaseCallback):
    """Maintain rolling latest and best training artifacts."""

    def __init__(
        self,
        *,
        engine_tuning_controller: EngineTuningTrainingController | None,
        policy: CheckpointPolicy,
        run_paths: RunPaths,
        lineage_step_offset: int,
    ) -> None:
        super().__init__(verbose=0)
        self._engine_tuning_controller = engine_tuning_controller
        self._policy = policy
        self._run_paths = run_paths
        self._lineage_step_offset = lineage_step_offset
        self._best_episode_return: float | None = None
        self._rollout_count = 0

    def _save_latest(self) -> None:
        save_artifacts_atomically(
            model=self.model,
            model_path=self._run_paths.latest_model_path,
            policy_path=self._run_paths.latest_policy_path,
            engine_tuning_state=self._engine_tuning_state(),
            policy_metadata=current_policy_artifact_metadata(
                self.model,
                lineage_step_offset=self._lineage_step_offset,
            ),
        )

    def _save_recent(self) -> None:
        num_timesteps = getattr(self.model, "num_timesteps", None)
        if not isinstance(num_timesteps, int):
            return
        save_recent_checkpoint_artifacts(
            self.model,
            self._run_paths,
            engine_tuning_state=self._engine_tuning_state(),
            num_timesteps=num_timesteps,
            policy_metadata=current_policy_artifact_metadata(
                self.model,
                lineage_step_offset=self._lineage_step_offset,
            ),
        )
        trim_recent_checkpoint_artifacts(
            self._run_paths,
            keep_last=self._policy.recent_limit,
        )

    def _save_periodic(self) -> None:
        if self._policy.save_latest:
            self._save_latest()
        if self._policy.save_recent:
            self._save_recent()

    def _save_best(self, episode_return: float) -> None:
        if self._best_episode_return is not None and episode_return <= self._best_episode_return:
            return
        self._best_episode_return = episode_return
        save_artifacts_atomically(
            model=self.model,
            model_path=self._run_paths.best_model_path,
            policy_path=self._run_paths.best_policy_path,
            engine_tuning_state=self._engine_tuning_state(),
            policy_metadata=current_policy_artifact_metadata(
                self.model,
                lineage_step_offset=self._lineage_step_offset,
            ),
        )

    def _on_training_start(self) -> None:
        if self._policy.save_latest:
            self._save_latest()

    def _on_step(self) -> bool:
        if (
            self._policy.step_interval is not None
            and self.n_calls % self._policy.step_interval == 0
        ):
            self._save_periodic()

        infos = info_sequence(self.locals.get("infos"))
        if infos is None:
            return True

        if not self._policy.save_best:
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue
            episode = info.get("episode")
            if not isinstance(episode, dict):
                continue
            episode_return = episode.get("r")
            if isinstance(episode_return, int | float):
                self._save_best(float(episode_return))
        return True

    def _on_rollout_end(self) -> None:
        if self._policy.rollout_interval is None:
            return
        self._rollout_count += 1
        if self._rollout_count % self._policy.rollout_interval == 0:
            self._save_periodic()

    def _engine_tuning_state(self) -> EngineTuningRuntimeState | None:
        if self._engine_tuning_controller is None:
            return None
        return self._engine_tuning_controller.runtime_state
