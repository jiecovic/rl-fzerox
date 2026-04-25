# src/rl_fzerox/core/training/session/callbacks/track_sampling.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rl_fzerox.core.config.schema import CurriculumConfig, EnvConfig, TrackSamplingConfig


@dataclass(slots=True)
class _TrackStepStats:
    base_weight: float
    completed_frames: int = 0
    episode_count: int = 0
    ema_episode_frames: float | None = None
    current_weight: float = 1.0

    def record_episode(self, frame_count: int, *, ema_alpha: float) -> None:
        self.completed_frames += frame_count
        self.episode_count += 1
        if self.ema_episode_frames is None:
            self.ema_episode_frames = float(frame_count)
            return
        self.ema_episode_frames = (
            (1.0 - ema_alpha) * self.ema_episode_frames + ema_alpha * frame_count
        )


class StepBalancedTrackSamplingController:
    """Adapt track reset weights so each map trends toward equal frame coverage."""

    def __init__(
        self,
        *,
        track_base_weights: dict[str, float],
        action_repeat: int,
        update_episodes: int,
        ema_alpha: float,
        max_weight_scale: float,
    ) -> None:
        self._stats = {
            track_id: _TrackStepStats(base_weight=weight, current_weight=weight)
            for track_id, weight in track_base_weights.items()
        }
        self._action_repeat = max(1, int(action_repeat))
        self._update_episodes = max(1, int(update_episodes))
        self._ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self._max_weight_scale = max(1.0, float(max_weight_scale))
        self._episodes_since_update = 0
        self.update_count = 0

    @classmethod
    def from_configs(
        cls,
        *,
        env_config: EnvConfig,
        curriculum_config: CurriculumConfig,
    ) -> StepBalancedTrackSamplingController | None:
        configs = _step_balanced_sampling_configs(env_config, curriculum_config)
        if not configs:
            return None

        base_weights: dict[str, float] = {}
        for config in configs:
            for entry in config.entries:
                base_weights.setdefault(entry.id, float(entry.weight))
        if len(base_weights) <= 1:
            return None

        settings = configs[0]
        return cls(
            track_base_weights=base_weights,
            action_repeat=env_config.action_repeat,
            update_episodes=settings.step_balance_update_episodes,
            ema_alpha=settings.step_balance_ema_alpha,
            max_weight_scale=settings.step_balance_max_weight_scale,
        )

    def record_episodes(self, episodes: Sequence[dict[str, object]]) -> dict[str, float] | None:
        recorded_count = 0
        for episode in episodes:
            track_id = _episode_track_id(episode)
            if track_id is None or track_id not in self._stats:
                continue
            frame_count = _episode_frame_count(episode, action_repeat=self._action_repeat)
            if frame_count is None:
                continue
            self._stats[track_id].record_episode(frame_count, ema_alpha=self._ema_alpha)
            recorded_count += 1

        if recorded_count == 0:
            return None

        self._episodes_since_update += recorded_count
        if self._episodes_since_update < self._update_episodes:
            return None

        self._episodes_since_update = 0
        self.update_count += 1
        return self._compute_weights()

    def log_values(self) -> dict[str, float]:
        total_weight = sum(stats.current_weight for stats in self._stats.values())
        total_episodes = sum(stats.episode_count for stats in self._stats.values())
        total_frames = sum(stats.completed_frames for stats in self._stats.values())
        values: dict[str, float] = {
            "track_sampling/step_balance_updates": float(self.update_count),
            "track_sampling/step_balance_episodes": float(total_episodes),
        }
        for track_id, stats in self._stats.items():
            key = _sanitize_log_key(track_id)
            values[f"track_sampling/{key}/weight"] = stats.current_weight
            values[f"track_sampling/{key}/prob"] = (
                stats.current_weight / total_weight if total_weight > 0.0 else 0.0
            )
            values[f"track_sampling/{key}/episodes"] = float(stats.episode_count)
            values[f"track_sampling/{key}/episode_share"] = (
                stats.episode_count / total_episodes if total_episodes > 0 else 0.0
            )
            values[f"track_sampling/{key}/frames"] = float(stats.completed_frames)
            values[f"track_sampling/{key}/frame_share"] = (
                stats.completed_frames / total_frames if total_frames > 0 else 0.0
            )
            if stats.ema_episode_frames is not None:
                values[f"track_sampling/{key}/ema_episode_frames"] = stats.ema_episode_frames
        return values

    def _compute_weights(self) -> dict[str, float]:
        reference_length = self._reference_episode_length()
        raw_weights: dict[str, float] = {}
        for track_id, stats in self._stats.items():
            length = stats.ema_episode_frames
            scale = 1.0 if length is None else reference_length / max(length, 1.0)
            scale = max(1.0 / self._max_weight_scale, min(self._max_weight_scale, scale))
            raw_weights[track_id] = stats.base_weight * scale

        total_base_weight = sum(stats.base_weight for stats in self._stats.values())
        total_raw_weight = sum(raw_weights.values())
        if total_raw_weight <= 0.0:
            return {
                track_id: stats.current_weight for track_id, stats in self._stats.items()
            }

        normalized = {
            track_id: weight * total_base_weight / total_raw_weight
            for track_id, weight in raw_weights.items()
        }
        for track_id, weight in normalized.items():
            self._stats[track_id].current_weight = weight
        return normalized

    def _reference_episode_length(self) -> float:
        weighted_lengths = [
            (stats.base_weight, stats.ema_episode_frames)
            for stats in self._stats.values()
            if stats.ema_episode_frames is not None
        ]
        if not weighted_lengths:
            return 1.0
        total_weight = sum(weight for weight, _ in weighted_lengths)
        if total_weight <= 0.0:
            return 1.0
        return max(
            1.0,
            sum(weight * length for weight, length in weighted_lengths) / total_weight,
        )


def _step_balanced_sampling_configs(
    env_config: EnvConfig,
    curriculum_config: CurriculumConfig,
) -> tuple[TrackSamplingConfig, ...]:
    configs: list[TrackSamplingConfig] = []
    if _is_step_balanced(env_config.track_sampling):
        configs.append(env_config.track_sampling)
    if curriculum_config.enabled:
        for stage in curriculum_config.stages:
            if stage.track_sampling is not None and _is_step_balanced(stage.track_sampling):
                configs.append(stage.track_sampling)
    return tuple(configs)


def _is_step_balanced(config: TrackSamplingConfig) -> bool:
    return config.enabled and config.sampling_mode == "step_balanced" and bool(config.entries)


def _episode_track_id(episode: dict[str, object]) -> str | None:
    value = episode.get("track_id")
    if isinstance(value, str) and value:
        return value
    return None


def _episode_frame_count(
    episode: dict[str, object],
    *,
    action_repeat: int,
) -> int | None:
    episode_step = episode.get("episode_step")
    if isinstance(episode_step, int | float) and not isinstance(episode_step, bool):
        frame_count = int(episode_step)
        return frame_count if frame_count > 0 else None

    monitor_length = episode.get("l")
    if isinstance(monitor_length, int | float) and not isinstance(monitor_length, bool):
        frame_count = int(monitor_length) * action_repeat
        return frame_count if frame_count > 0 else None
    return None


def _sanitize_log_key(value: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    return sanitized.strip("_") or "unknown"
