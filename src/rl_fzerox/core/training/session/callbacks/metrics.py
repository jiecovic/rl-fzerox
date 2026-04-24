# src/rl_fzerox/core/training/session/callbacks/metrics.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


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
        _MetricLogSpec("boost_pad_entered", "state/boost_pad_entry_step_rate"),
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
        _MetricLogSpec("boost_pad_entries", "episode/boost_pad_entries_mean"),
        _MetricLogSpec("boost_pad_entries_per_lap", "episode/boost_pad_entries_per_lap_mean"),
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
            "progress_stalled",
            "timeout",
        ),
    ),
)


class CallbackLogger(Protocol):
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

        episodes = episode_dicts(infos)
        self.episode_count += len(episodes)
        for spec in ROLLOUT_INFO_LOG_SPECS.episode_metrics:
            values = _numeric_episode_values(episodes, spec.info_key)
            if values:
                self.episode_metrics[spec.info_key].add_many(values)

        finished_episodes = finished_episode_dicts(episodes)
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

    def record_to(self, logger: CallbackLogger) -> None:
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
        course_key = course_log_key(episode)
        if course_key is None:
            return
        race_time_ms = episode.get("race_time_ms")
        if not isinstance(race_time_ms, int | float):
            return
        self.course_finish_times_s.setdefault(course_key, _MeanAccumulator()).add_many(
            [float(race_time_ms) * 0.001]
        )


def info_sequence(infos: object) -> Sequence[object] | None:
    if isinstance(infos, list | tuple):
        return infos
    return None


def episode_dicts(infos: Sequence[object]) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        episode = info.get("episode")
        if isinstance(episode, dict):
            episodes.append(episode)
    return episodes


def finished_episode_dicts(episodes: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [episode for episode in episodes if episode.get("termination_reason") == "finished"]


def course_log_key(episode: dict[str, object]) -> str | None:
    for key in ("track_course_id", "track_id", "track_course_name", "course_index"):
        value = episode.get(key)
        if isinstance(value, int):
            return f"course_{value}"
        if isinstance(value, str) and value.strip():
            sanitized = _sanitize_log_component(value)
            if sanitized:
                return sanitized
    return None


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


def _sanitize_log_component(value: str) -> str:
    normalized = value.strip().lower()
    characters = [character if character.isalnum() else "_" for character in normalized]
    collapsed = "_".join(part for part in "".join(characters).split("_") if part)
    return collapsed
