# src/rl_fzerox/core/training/session/callbacks/track_sampling/episodes.py
from __future__ import annotations

from collections.abc import Mapping

from rl_fzerox.core.runtime_spec.schema import CurriculumConfig, EnvConfig, TrackSamplingConfig


def runtime_track_sampling_configs(
    env_config: EnvConfig,
    curriculum_config: CurriculumConfig,
) -> tuple[TrackSamplingConfig, ...]:
    configs: list[TrackSamplingConfig] = []
    if uses_track_sampling_runtime(env_config.track_sampling):
        configs.append(env_config.track_sampling)
    if curriculum_config.enabled:
        for stage in curriculum_config.stages:
            if stage.track_sampling is not None and uses_track_sampling_runtime(
                stage.track_sampling,
            ):
                configs.append(stage.track_sampling)
    return tuple(configs)


def uses_track_sampling_runtime(config: TrackSamplingConfig) -> bool:
    return (
        config.enabled
        and uses_track_sampling_runtime_mode(config.sampling_mode)
        and bool(config.entries)
    )


def episode_track_id(episode: Mapping[str, object]) -> str | None:
    value = episode.get("track_id")
    if isinstance(value, str) and value:
        return value
    return None


def episode_frame_count(
    episode: Mapping[str, object],
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


def episode_finished(episode: Mapping[str, object]) -> bool:
    return episode.get("termination_reason") == "finished"


def episode_completion_fraction(episode: Mapping[str, object]) -> float | None:
    value = episode.get("episode_completion_fraction")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    if episode_finished(episode):
        return 1.0
    return None


def uses_step_balance_scheduler(sampling_mode: str) -> bool:
    return sampling_mode in {"step_balanced", "adaptive_step_balanced"}


def uses_track_sampling_runtime_mode(sampling_mode: str) -> bool:
    return sampling_mode in {"deficit_budget", "fixed_env"} or uses_step_balance_scheduler(
        sampling_mode
    )


def sanitize_log_key(value: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    return sanitized.strip("_") or "unknown"
