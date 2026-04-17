# src/rl_fzerox/core/training/session/callbacks.py
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rl_fzerox.core.config.schema import (
    CurriculumConfig,
    CurriculumTrainOverridesConfig,
    TrainConfig,
)
from rl_fzerox.core.training.runs import RunPaths
from rl_fzerox.core.training.session.artifacts import (
    current_policy_artifact_metadata,
    save_artifacts_atomically,
)
from rl_fzerox.core.training.session.curriculum import ActionMaskCurriculumController

_STATE_LOG_KEYS: tuple[tuple[str, str], ...] = (
    ("race_distance", "state/race_distance_mean"),
    ("speed_kph", "state/speed_kph_mean"),
    ("position", "state/position_mean"),
    ("lap", "state/lap_mean"),
    ("race_laps_completed", "state/race_laps_completed_mean"),
)
_STEP_RATE_LOG_KEYS: tuple[tuple[str, str], ...] = (
    ("damage_taken_frames", "state/damage_taken_step_rate"),
    ("collision_recoil_entered", "state/collision_recoil_entry_rate"),
)
_EPISODE_LOG_KEYS: tuple[tuple[str, str], ...] = (
    ("position", "episode/final_position_mean"),
    ("race_laps_completed", "episode/race_laps_completed_mean"),
)
_FINISHED_EPISODE_LOG_KEYS: tuple[tuple[str, str, float], ...] = (
    ("race_time_ms", "episode/finish_time_s_mean", 0.001),
    ("episode_step", "episode/finish_steps_mean", 1.0),
    ("position", "episode/finish_position_mean", 1.0),
)
_TERMINATION_REASON_KEYS: tuple[str, ...] = (
    "finished",
    "spinning_out",
    "crashed",
    "retired",
    "falling_off_track",
    "energy_depleted",
)
_TRUNCATION_REASON_KEYS: tuple[str, ...] = (
    "stuck",
    "wrong_way",
    "progress_stalled",
    "timeout",
)


@runtime_checkable
class _PpoTunableModel(Protocol):
    learning_rate: float | Callable[[float], float]
    lr_schedule: Callable[[float], float]
    clip_range: Callable[[float], float]
    n_epochs: int
    batch_size: int
    ent_coef: float
    policy: object


@runtime_checkable
class _OptimizerLike(Protocol):
    param_groups: list[dict[str, object]]


@runtime_checkable
class _PolicyWithOptimizer(Protocol):
    optimizer: _OptimizerLike


@dataclass
class _MeanAccumulator:
    total: float = 0.0
    count: int = 0

    def add_many(self, values: Sequence[float]) -> None:
        self.total += float(sum(values))
        self.count += len(values)

    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count


@dataclass
class _RateAccumulator:
    positive_count: int = 0
    count: int = 0

    def add_many(self, values: Sequence[bool]) -> None:
        self.positive_count += sum(1 for value in values if value)
        self.count += len(values)

    def rate(self) -> float | None:
        if self.count == 0:
            return None
        return self.positive_count / self.count


@dataclass
class RolloutInfoAccumulator:
    state_metrics: dict[str, _MeanAccumulator] = field(
        default_factory=lambda: {key: _MeanAccumulator() for key, _ in _STATE_LOG_KEYS}
    )
    step_rates: dict[str, _RateAccumulator] = field(
        default_factory=lambda: {key: _RateAccumulator() for key, _ in _STEP_RATE_LOG_KEYS}
    )
    episode_metrics: dict[str, _MeanAccumulator] = field(
        default_factory=lambda: {key: _MeanAccumulator() for key, _ in _EPISODE_LOG_KEYS}
    )
    finished_episode_metrics: dict[str, _MeanAccumulator] = field(
        default_factory=lambda: {
            key: _MeanAccumulator() for key, _, _ in _FINISHED_EPISODE_LOG_KEYS
        }
    )
    termination_counts: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in _TERMINATION_REASON_KEYS}
    )
    truncation_counts: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in _TRUNCATION_REASON_KEYS}
    )
    episode_count: int = 0

    def add_infos(self, infos: Sequence[object]) -> None:
        for info_key, _ in _STATE_LOG_KEYS:
            values = _numeric_values(infos, info_key)
            if values:
                self.state_metrics[info_key].add_many(values)

        for info_key, _ in _STEP_RATE_LOG_KEYS:
            values = _positive_values(infos, info_key)
            if values:
                self.step_rates[info_key].add_many(values)

        episodes = _episode_dicts(infos)
        self.episode_count += len(episodes)
        for episode_key, _ in _EPISODE_LOG_KEYS:
            values = _numeric_episode_values(episodes, episode_key)
            if values:
                self.episode_metrics[episode_key].add_many(values)

        finished_episodes = _finished_episode_dicts(episodes)
        for episode_key, _, scale in _FINISHED_EPISODE_LOG_KEYS:
            values = _numeric_episode_values(finished_episodes, episode_key)
            if values:
                scaled_values = [value * scale for value in values]
                self.finished_episode_metrics[episode_key].add_many(scaled_values)

        for episode in episodes:
            termination_reason = episode.get("termination_reason")
            if (
                isinstance(termination_reason, str)
                and termination_reason in self.termination_counts
            ):
                self.termination_counts[termination_reason] += 1
            truncation_reason = episode.get("truncation_reason")
            if isinstance(truncation_reason, str) and truncation_reason in self.truncation_counts:
                self.truncation_counts[truncation_reason] += 1

    def record_to(self, logger) -> None:
        for info_key, log_key in _STATE_LOG_KEYS:
            mean = self.state_metrics[info_key].mean()
            if mean is not None:
                logger.record(log_key, mean)

        for info_key, log_key in _STEP_RATE_LOG_KEYS:
            rate = self.step_rates[info_key].rate()
            if rate is not None:
                logger.record(log_key, rate)

        for episode_key, log_key in _EPISODE_LOG_KEYS:
            mean = self.episode_metrics[episode_key].mean()
            if mean is not None:
                logger.record(log_key, mean)

        for episode_key, log_key, _ in _FINISHED_EPISODE_LOG_KEYS:
            mean = self.finished_episode_metrics[episode_key].mean()
            if mean is not None:
                logger.record(log_key, mean)

        if self.episode_count == 0:
            return

        for reason in _TERMINATION_REASON_KEYS:
            logger.record(
                f"episode/{reason}_rate",
                self.termination_counts[reason] / self.episode_count,
            )
        for reason in _TRUNCATION_REASON_KEYS:
            logger.record(
                f"episode/{reason}_rate",
                self.truncation_counts[reason] / self.episode_count,
            )


def build_callbacks(
    *,
    train_config: TrainConfig,
    curriculum_config: CurriculumConfig,
    run_paths: RunPaths,
):
    """Construct the SB3 callback list used during training."""

    try:
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc

    # Keep the SB3-specific classes local so this module stays importable for
    # tests and tooling even when the optional training extras are not present.
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
                policy_metadata=current_policy_artifact_metadata(self.training_env),
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
                policy_metadata=current_policy_artifact_metadata(self.training_env),
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

        def __init__(self, curriculum: CurriculumConfig) -> None:
            super().__init__(verbose=0)
            self._controller = ActionMaskCurriculumController(curriculum)

        def _on_training_start(self) -> None:
            self._apply_current_stage()

        def _on_step(self) -> bool:
            infos = info_sequence(self.locals.get("infos"))
            if infos is None:
                return True

            promoted_stage = self._controller.record_episodes(_episode_dicts(infos))
            if promoted_stage is not None:
                self._apply_current_stage()
            return True

        def _on_rollout_end(self) -> None:
            stage_index = self._controller.stage_index
            self.logger.record(
                "curriculum/stage",
                -1 if stage_index is None else stage_index,
            )
            _record_stage_train_overrides(
                logger=self.logger,
                overrides=self._controller.stage_train_overrides,
            )

        def _apply_current_stage(self) -> None:
            stage_index = self._controller.stage_index
            if stage_index is None:
                return
            self.training_env.env_method("set_curriculum_stage", stage_index)
            _apply_stage_train_overrides(
                model=self.model,
                overrides=self._controller.stage_train_overrides,
            )

    adjusted_save_freq = max(1, train_config.save_freq // train_config.num_envs)
    callbacks: list[BaseCallback] = [
        RollingArtifactCallback(
            save_freq=adjusted_save_freq,
            run_paths=run_paths,
        ),
        InfoLoggingCallback(),
    ]
    if curriculum_config.enabled:
        callbacks.append(CurriculumCallback(curriculum_config))
    return CallbackList(callbacks)


def _numeric_values(infos: Sequence[object], key: str) -> list[float]:
    values: list[float] = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        value = info.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _positive_values(infos: Sequence[object], key: str) -> list[bool]:
    values: list[bool] = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        value = info.get(key)
        if isinstance(value, bool):
            values.append(value)
        elif isinstance(value, int | float):
            values.append(float(value) > 0.0)
    return values


def info_sequence(infos: object) -> Sequence[object] | None:
    if isinstance(infos, list | tuple):
        return infos
    return None


def _episode_dicts(infos: Sequence[object]) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        episode = info.get("episode")
        if isinstance(episode, dict):
            episodes.append(episode)
    return episodes


def _finished_episode_dicts(episodes: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [episode for episode in episodes if episode.get("termination_reason") == "finished"]


def _apply_stage_train_overrides(
    *,
    model: object,
    overrides: CurriculumTrainOverridesConfig | None,
) -> None:
    if overrides is None:
        return
    if not isinstance(model, _PpoTunableModel):
        raise RuntimeError("Curriculum train overrides require a PPO-family model")
    if overrides.learning_rate is not None:
        _set_model_learning_rate(model, float(overrides.learning_rate))
    if overrides.n_epochs is not None:
        model.n_epochs = int(overrides.n_epochs)
    if overrides.batch_size is not None:
        model.batch_size = int(overrides.batch_size)
    if overrides.clip_range is not None:
        model.clip_range = _constant_schedule(float(overrides.clip_range))
    if overrides.ent_coef is not None:
        model.ent_coef = float(overrides.ent_coef)


def _set_model_learning_rate(model: _PpoTunableModel, learning_rate: float) -> None:
    model.learning_rate = learning_rate
    model.lr_schedule = _constant_schedule(learning_rate)
    policy = model.policy
    if not isinstance(policy, _PolicyWithOptimizer):
        return
    for param_group in policy.optimizer.param_groups:
        param_group["lr"] = learning_rate


def _constant_schedule(value: float) -> Callable[[float], float]:
    def schedule(_progress_remaining: float) -> float:
        return value

    return schedule


def _record_stage_train_overrides(
    *,
    logger,
    overrides: CurriculumTrainOverridesConfig | None,
) -> None:
    if overrides is None:
        return
    if overrides.learning_rate is not None:
        logger.record("curriculum/learning_rate", float(overrides.learning_rate))
    if overrides.n_epochs is not None:
        logger.record("curriculum/n_epochs", int(overrides.n_epochs))
    if overrides.batch_size is not None:
        logger.record("curriculum/batch_size", int(overrides.batch_size))
    if overrides.clip_range is not None:
        logger.record("curriculum/clip_range", float(overrides.clip_range))
    if overrides.ent_coef is not None:
        logger.record("curriculum/ent_coef", float(overrides.ent_coef))


def _numeric_episode_values(
    episodes: Sequence[dict[str, object]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for episode in episodes:
        value = episode.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values
