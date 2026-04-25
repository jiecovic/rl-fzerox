# src/rl_fzerox/core/training/session/callbacks/sb3.py
from __future__ import annotations

from rl_fzerox.core.config.schema import CurriculumConfig, EnvConfig, TrainConfig
from rl_fzerox.core.training.runs import RunPaths
from rl_fzerox.core.training.session.artifacts import (
    current_policy_artifact_metadata,
    save_artifacts_atomically,
)
from rl_fzerox.core.training.session.curriculum import ActionMaskCurriculumController

from .metrics import RolloutInfoAccumulator, episode_dicts, info_sequence
from .track_sampling import StepBalancedTrackSamplingController
from .tuning import apply_stage_train_overrides, record_stage_train_overrides


def build_callbacks(
    *,
    env_config: EnvConfig | None = None,
    train_config: TrainConfig,
    curriculum_config: CurriculumConfig,
    run_paths: RunPaths,
    initial_curriculum_stage_index: int | None = None,
):
    """Construct the SB3 callback list used during training."""

    try:
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc

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

    class RollingArtifactCallback(BaseCallback):
        """Maintain rolling latest and best training artifacts."""

        def __init__(self, *, save_freq: int, run_paths: RunPaths) -> None:
            super().__init__(verbose=0)
            self._save_freq = save_freq
            self._run_paths = run_paths
            self._best_episode_return: float | None = None

        def _save_latest(self) -> None:
            save_artifacts_atomically(
                model=self.model,
                model_path=self._run_paths.latest_model_path,
                policy_path=self._run_paths.latest_policy_path,
                policy_metadata=current_policy_artifact_metadata(self.training_env, self.model),
            )

        def _save_best(self, episode_return: float) -> None:
            if (
                self._best_episode_return is not None
                and episode_return <= self._best_episode_return
            ):
                return
            self._best_episode_return = episode_return
            save_artifacts_atomically(
                model=self.model,
                model_path=self._run_paths.best_model_path,
                policy_path=self._run_paths.best_policy_path,
                policy_metadata=current_policy_artifact_metadata(self.training_env, self.model),
            )

        def _on_training_start(self) -> None:
            self._save_latest()

        def _on_step(self) -> bool:
            if self.n_calls % self._save_freq == 0:
                self._save_latest()

            infos = info_sequence(self.locals.get("infos"))
            if infos is None:
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

    class CurriculumCallback(BaseCallback):
        """Promote curriculum stages and apply their rollout-time overrides."""

        def __init__(
            self,
            curriculum: CurriculumConfig,
            *,
            initial_stage_index: int | None = None,
        ) -> None:
            super().__init__(verbose=0)
            self._controller = ActionMaskCurriculumController(
                curriculum,
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

    class StepBalancedTrackSamplingCallback(BaseCallback):
        """Refresh track sampling weights from completed-episode frame counts."""

        def __init__(self, controller: StepBalancedTrackSamplingController) -> None:
            super().__init__(verbose=0)
            self._controller = controller

        def _on_step(self) -> bool:
            infos = info_sequence(self.locals.get("infos"))
            if infos is None:
                return True

            weights = self._controller.record_episodes(episode_dicts(infos))
            if weights is not None:
                self.training_env.env_method("set_track_sampling_weights", weights)
            return True

        def _on_rollout_end(self) -> None:
            for key, value in self._controller.log_values().items():
                self.logger.record(key, value)

    adjusted_save_freq = max(1, train_config.save_freq // train_config.num_envs)
    callbacks: list[BaseCallback] = [
        RollingArtifactCallback(
            save_freq=adjusted_save_freq,
            run_paths=run_paths,
        ),
        InfoLoggingCallback(),
    ]
    if env_config is not None:
        track_balance_controller = StepBalancedTrackSamplingController.from_configs(
            env_config=env_config,
            curriculum_config=curriculum_config,
        )
        if track_balance_controller is not None:
            callbacks.append(StepBalancedTrackSamplingCallback(track_balance_controller))
    if curriculum_config.enabled:
        callbacks.append(
            CurriculumCallback(
                curriculum_config,
                initial_stage_index=initial_curriculum_stage_index,
            )
        )
    return CallbackList(callbacks)
