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


@dataclass(frozen=True, slots=True)
class _MetricLogSpec:
    info_key: str
    log_key: str
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class _FrameRatioLogSpec:
    numerator_key: str
    denominator_key: str
    log_key: str


@dataclass(frozen=True, slots=True)
class _EpisodeReasonSpecs:
    termination: tuple[str, ...]
    truncation: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RolloutInfoLogSpecs:
    state_metrics: tuple[_MetricLogSpec, ...]
    step_rates: tuple[_MetricLogSpec, ...]
    frame_ratios: tuple[_FrameRatioLogSpec, ...]
    episode_metrics: tuple[_MetricLogSpec, ...]
    finished_episode_metrics: tuple[_MetricLogSpec, ...]
    episode_reasons: _EpisodeReasonSpecs


ROLLOUT_INFO_LOG_SPECS = _RolloutInfoLogSpecs(
    state_metrics=(
        _MetricLogSpec("race_distance", "state/race_distance_mean"),
        _MetricLogSpec("speed_kph", "state/speed_kph_mean"),
        _MetricLogSpec("position", "state/position_mean"),
        _MetricLogSpec("lap", "state/lap_mean"),
        _MetricLogSpec("race_laps_completed", "state/race_laps_completed_mean"),
    ),
    step_rates=(
        _MetricLogSpec("damage_taken_frames", "state/damage_taken_step_rate"),
        _MetricLogSpec("collision_recoil_entered", "state/collision_recoil_entry_rate"),
        _MetricLogSpec("boost_used", "action/boost_used_step_rate"),
        _MetricLogSpec("lean_used", "action/lean_used_step_rate"),
    ),
    frame_ratios=(
        _FrameRatioLogSpec(
            numerator_key="airborne_frames",
            denominator_key="frames_run",
            log_key="state/airborne_frame_ratio",
        ),
    ),
    episode_metrics=(
        _MetricLogSpec("position", "episode/final_position_mean"),
        _MetricLogSpec("race_laps_completed", "episode/race_laps_completed_mean"),
    ),
    finished_episode_metrics=(
        _MetricLogSpec("race_time_ms", "episode/finish_time_s_mean", scale=0.001),
        _MetricLogSpec("episode_step", "episode/finish_steps_mean"),
        _MetricLogSpec("position", "episode/finish_position_mean"),
    ),
    episode_reasons=_EpisodeReasonSpecs(
        termination=(
            "finished",
            "spinning_out",
            "crashed",
            "retired",
            "falling_off_track",
            "energy_depleted",
        ),
        truncation=(
            "stuck",
            "wrong_way",
            "progress_stalled",
            "timeout",
        ),
    ),
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


class _LoggerLike(Protocol):
    def record(self, key: str, value: object) -> None: ...


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
class _RatioAccumulator:
    numerator_total: float = 0.0
    denominator_total: float = 0.0

    def add(self, numerator: float, denominator: float) -> None:
        if denominator <= 0.0:
            return
        self.numerator_total += numerator
        self.denominator_total += denominator

    def ratio(self) -> float | None:
        if self.denominator_total <= 0.0:
            return None
        return self.numerator_total / self.denominator_total


@dataclass
class RolloutInfoAccumulator:
    state_metrics: dict[str, _MeanAccumulator] = field(
        default_factory=lambda: {
            spec.info_key: _MeanAccumulator() for spec in ROLLOUT_INFO_LOG_SPECS.state_metrics
        }
    )
    step_rates: dict[str, _RateAccumulator] = field(
        default_factory=lambda: {
            spec.info_key: _RateAccumulator() for spec in ROLLOUT_INFO_LOG_SPECS.step_rates
        }
    )
    frame_ratios: dict[str, _RatioAccumulator] = field(
        default_factory=lambda: {
            spec.log_key: _RatioAccumulator() for spec in ROLLOUT_INFO_LOG_SPECS.frame_ratios
        }
    )
    episode_metrics: dict[str, _MeanAccumulator] = field(
        default_factory=lambda: {
            spec.info_key: _MeanAccumulator() for spec in ROLLOUT_INFO_LOG_SPECS.episode_metrics
        }
    )
    finished_episode_metrics: dict[str, _MeanAccumulator] = field(
        default_factory=lambda: {
            spec.info_key: _MeanAccumulator()
            for spec in ROLLOUT_INFO_LOG_SPECS.finished_episode_metrics
        }
    )
    course_finish_times_s: dict[str, _MeanAccumulator] = field(default_factory=dict)
    termination_counts: dict[str, int] = field(
        default_factory=lambda: {
            reason: 0 for reason in ROLLOUT_INFO_LOG_SPECS.episode_reasons.termination
        }
    )
    truncation_counts: dict[str, int] = field(
        default_factory=lambda: {
            reason: 0 for reason in ROLLOUT_INFO_LOG_SPECS.episode_reasons.truncation
        }
    )
    episode_count: int = 0

    def add_infos(self, infos: Sequence[object]) -> None:
        for spec in ROLLOUT_INFO_LOG_SPECS.state_metrics:
            values = _numeric_values(infos, spec.info_key)
            if values:
                self.state_metrics[spec.info_key].add_many(values)

        for spec in ROLLOUT_INFO_LOG_SPECS.step_rates:
            values = _positive_values(infos, spec.info_key)
            if values:
                self.step_rates[spec.info_key].add_many(values)

        for spec in ROLLOUT_INFO_LOG_SPECS.frame_ratios:
            for numerator, denominator in _numeric_pair_values(
                infos,
                numerator_key=spec.numerator_key,
                denominator_key=spec.denominator_key,
            ):
                self.frame_ratios[spec.log_key].add(numerator, denominator)

        episodes = _episode_dicts(infos)
        self.episode_count += len(episodes)
        for spec in ROLLOUT_INFO_LOG_SPECS.episode_metrics:
            values = _numeric_episode_values(episodes, spec.info_key)
            if values:
                self.episode_metrics[spec.info_key].add_many(values)

        finished_episodes = _finished_episode_dicts(episodes)
        for spec in ROLLOUT_INFO_LOG_SPECS.finished_episode_metrics:
            values = _numeric_episode_values(finished_episodes, spec.info_key)
            if values:
                scaled_values = [value * spec.scale for value in values]
                self.finished_episode_metrics[spec.info_key].add_many(scaled_values)
        for episode in finished_episodes:
            self._add_course_finish_time(episode)

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

    def record_to(self, logger: _LoggerLike) -> None:
        for spec in ROLLOUT_INFO_LOG_SPECS.state_metrics:
            mean = self.state_metrics[spec.info_key].mean()
            if mean is not None:
                logger.record(spec.log_key, mean)

        for spec in ROLLOUT_INFO_LOG_SPECS.step_rates:
            rate = self.step_rates[spec.info_key].rate()
            if rate is not None:
                logger.record(spec.log_key, rate)

        for spec in ROLLOUT_INFO_LOG_SPECS.frame_ratios:
            ratio = self.frame_ratios[spec.log_key].ratio()
            if ratio is not None:
                logger.record(spec.log_key, ratio)

        for spec in ROLLOUT_INFO_LOG_SPECS.episode_metrics:
            mean = self.episode_metrics[spec.info_key].mean()
            if mean is not None:
                logger.record(spec.log_key, mean)

        for spec in ROLLOUT_INFO_LOG_SPECS.finished_episode_metrics:
            mean = self.finished_episode_metrics[spec.info_key].mean()
            if mean is not None:
                logger.record(spec.log_key, mean)
        for course_key, accumulator in sorted(self.course_finish_times_s.items()):
            mean = accumulator.mean()
            if mean is not None:
                logger.record(f"episode/by_course/{course_key}/finish_time_s_mean", mean)

        if self.episode_count == 0:
            return

        for reason in ROLLOUT_INFO_LOG_SPECS.episode_reasons.termination:
            logger.record(
                f"episode/{reason}_rate",
                self.termination_counts[reason] / self.episode_count,
            )
        for reason in ROLLOUT_INFO_LOG_SPECS.episode_reasons.truncation:
            logger.record(
                f"episode/{reason}_rate",
                self.truncation_counts[reason] / self.episode_count,
            )

    def _add_course_finish_time(self, episode: dict[str, object]) -> None:
        course_key = _course_log_key(episode)
        if course_key is None:
            return
        race_time_ms = episode.get("race_time_ms")
        if not isinstance(race_time_ms, int | float):
            return
        self.course_finish_times_s.setdefault(course_key, _MeanAccumulator()).add_many(
            [float(race_time_ms) * 0.001]
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


def _numeric_pair_values(
    infos: Sequence[object],
    *,
    numerator_key: str,
    denominator_key: str,
) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        numerator = info.get(numerator_key)
        denominator = info.get(denominator_key)
        if isinstance(numerator, int | float) and isinstance(denominator, int | float):
            values.append((float(numerator), float(denominator)))
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


def _course_log_key(episode: dict[str, object]) -> str | None:
    for key in ("track_course_id", "track_id", "track_course_name", "course_index"):
        value = episode.get(key)
        if isinstance(value, int):
            return f"course_{value}"
        if isinstance(value, str) and value.strip():
            sanitized = _sanitize_log_component(value)
            if sanitized:
                return sanitized
    return None


def _sanitize_log_component(value: str) -> str:
    normalized = value.strip().lower()
    characters = [character if character.isalnum() else "_" for character in normalized]
    collapsed = "_".join(part for part in "".join(characters).split("_") if part)
    return collapsed


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
    logger: _LoggerLike,
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
