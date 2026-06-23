# src/rl_fzerox/ui/watch/live_series.py
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from rl_fzerox.core.envs.observations.state.utils import clamp
from rl_fzerox.core.runtime_info import float_info, optional_int_info


class _LiveEpisodeSnapshot(Protocol):
    @property
    def episode(self) -> int: ...

    @property
    def policy_decision_frame(self) -> bool: ...

    @property
    def info(self) -> dict[str, object]: ...

    @property
    def episode_reward(self) -> float: ...


@dataclass(frozen=True, slots=True)
class KoStarRewardEvent:
    env_step: int
    previous_count: int
    current_count: int
    gained: int
    reward: float


@dataclass(frozen=True, slots=True)
class LiveSeriesLimits:
    """Bound watch chart history so video responsiveness stays independent."""

    max_samples: int = 4_096
    max_ko_events: int = 64


@dataclass(frozen=True, slots=True)
class LiveSeriesPublishPolicy:
    """Throttle live-chart snapshots independently from video frame publishing."""

    interval_seconds: float = 0.10


LIVE_SERIES_LIMITS = LiveSeriesLimits()
LIVE_SERIES_PUBLISH_POLICY = LiveSeriesPublishPolicy()


def _int_buffer() -> deque[int]:
    return deque(maxlen=LIVE_SERIES_LIMITS.max_samples)


def _float_buffer() -> deque[float]:
    return deque(maxlen=LIVE_SERIES_LIMITS.max_samples)


def _ko_star_event_buffer() -> deque[KoStarRewardEvent]:
    return deque(maxlen=LIVE_SERIES_LIMITS.max_ko_events)


@dataclass(frozen=True, slots=True)
class EpisodeLiveSeriesSnapshot:
    episode: int
    env_steps: tuple[int, ...]
    speed_kph: tuple[float, ...]
    step_rewards: tuple[float, ...]
    progress_speed_multiplier: tuple[float, ...]
    position_progress_multiplier: tuple[float, ...]
    progress_speed_position_multiplier: tuple[float, ...]
    edge_ratio: tuple[float, ...]
    outside_edge_excess_ratio: tuple[float, ...]
    height_above_ground: tuple[float, ...]
    ko_star_events: tuple[KoStarRewardEvent, ...]
    current_ko_star_count: int | None
    current_return: float
    current_progress: float
    max_progress: float


@dataclass(slots=True)
class EpisodeLiveSeriesTracker:
    episode: int | None = None
    env_steps: deque[int] = field(default_factory=_int_buffer)
    speed_kph: deque[float] = field(default_factory=_float_buffer)
    step_rewards: deque[float] = field(default_factory=_float_buffer)
    progress_speed_multiplier: deque[float] = field(default_factory=_float_buffer)
    position_progress_multiplier: deque[float] = field(default_factory=_float_buffer)
    progress_speed_position_multiplier: deque[float] = field(default_factory=_float_buffer)
    edge_ratio: deque[float] = field(default_factory=_float_buffer)
    outside_edge_excess_ratio: deque[float] = field(default_factory=_float_buffer)
    height_above_ground: deque[float] = field(default_factory=_float_buffer)
    ko_star_events: deque[KoStarRewardEvent] = field(default_factory=_ko_star_event_buffer)
    current_ko_star_count: int | None = None
    current_return: float = 0.0
    current_progress: float = 0.0
    max_progress: float = 0.0

    def observe_snapshot(
        self,
        snapshot: _LiveEpisodeSnapshot,
        *,
        action_repeat: int,
    ) -> None:
        if not snapshot.policy_decision_frame:
            return
        self.observe_decision(
            episode=snapshot.episode,
            info=snapshot.info,
            episode_reward=snapshot.episode_reward,
            telemetry_data=_snapshot_telemetry_data(snapshot),
            action_repeat=action_repeat,
        )

    def observe_decision(
        self,
        *,
        episode: int,
        info: dict[str, object],
        episode_reward: float,
        telemetry_data: Mapping[str, object] | None,
        action_repeat: int,
    ) -> None:
        if self.episode != episode:
            self._reset_episode(episode)
        env_step = _env_step(info, action_repeat=action_repeat)
        progress = _progress_fraction(info)
        speed_kph = _speed_kph(info)
        step_reward = _info_float(info, "step_reward")
        speed_multiplier = _info_float(info, "progress_speed_multiplier", default=1.0)
        position_multiplier = _info_float(
            info,
            "position_progress_multiplier",
            default=1.0,
        )
        combined_multiplier = _info_float(
            info,
            "progress_speed_position_multiplier",
            default=speed_multiplier * position_multiplier,
        )
        edge_ratio = _edge_ratio(telemetry_data)
        outside_edge_excess_ratio = _outside_edge_excess_ratio(telemetry_data)
        height_above_ground = _player_telemetry_float(telemetry_data, "height_above_ground")
        self.current_ko_star_count = _ko_star_count(info, telemetry_data)
        ko_star_event = _ko_star_reward_event(info, env_step=env_step)
        if ko_star_event is not None:
            if self.ko_star_events and self.ko_star_events[-1].env_step == env_step:
                self.ko_star_events[-1] = ko_star_event
            else:
                self.ko_star_events.append(ko_star_event)
        self.current_return = float(episode_reward)
        self.current_progress = progress
        self.max_progress = max(self.max_progress, progress)
        if self.env_steps and env_step == self.env_steps[-1]:
            self.speed_kph[-1] = speed_kph
            self.step_rewards[-1] = step_reward
            self.progress_speed_multiplier[-1] = speed_multiplier
            self.position_progress_multiplier[-1] = position_multiplier
            self.progress_speed_position_multiplier[-1] = combined_multiplier
            self.edge_ratio[-1] = edge_ratio
            self.outside_edge_excess_ratio[-1] = outside_edge_excess_ratio
            self.height_above_ground[-1] = height_above_ground
            return
        self.env_steps.append(env_step)
        self.speed_kph.append(speed_kph)
        self.step_rewards.append(step_reward)
        self.progress_speed_multiplier.append(speed_multiplier)
        self.position_progress_multiplier.append(position_multiplier)
        self.progress_speed_position_multiplier.append(combined_multiplier)
        self.edge_ratio.append(edge_ratio)
        self.outside_edge_excess_ratio.append(outside_edge_excess_ratio)
        self.height_above_ground.append(height_above_ground)

    def _reset_episode(self, episode: int) -> None:
        self.episode = episode
        self.env_steps.clear()
        self.speed_kph.clear()
        self.step_rewards.clear()
        self.progress_speed_multiplier.clear()
        self.position_progress_multiplier.clear()
        self.progress_speed_position_multiplier.clear()
        self.edge_ratio.clear()
        self.outside_edge_excess_ratio.clear()
        self.height_above_ground.clear()
        self.ko_star_events.clear()
        self.current_ko_star_count = None
        self.current_return = 0.0
        self.current_progress = 0.0
        self.max_progress = 0.0

    def snapshot(self) -> EpisodeLiveSeriesSnapshot | None:
        if self.episode is None:
            return None
        return EpisodeLiveSeriesSnapshot(
            episode=self.episode,
            env_steps=tuple(self.env_steps),
            speed_kph=tuple(self.speed_kph),
            step_rewards=tuple(self.step_rewards),
            progress_speed_multiplier=tuple(self.progress_speed_multiplier),
            position_progress_multiplier=tuple(self.position_progress_multiplier),
            progress_speed_position_multiplier=tuple(self.progress_speed_position_multiplier),
            edge_ratio=tuple(self.edge_ratio),
            outside_edge_excess_ratio=tuple(self.outside_edge_excess_ratio),
            height_above_ground=tuple(self.height_above_ground),
            ko_star_events=tuple(self.ko_star_events),
            current_ko_star_count=self.current_ko_star_count,
            current_return=self.current_return,
            current_progress=self.current_progress,
            max_progress=self.max_progress,
        )


def _env_step(info: dict[str, object], *, action_repeat: int) -> int:
    episode_frames = optional_int_info(info, "episode_step", minimum=0)
    if episode_frames is None:
        return 0
    repeat = max(1, int(action_repeat))
    return (episode_frames + repeat - 1) // repeat


def _progress_fraction(info: dict[str, object]) -> float:
    return float_info(info, "episode_completion_fraction", minimum=0.0, maximum=1.0)


def _speed_kph(info: dict[str, object]) -> float:
    return max(0.0, _info_float(info, "speed_kph"))


def _info_float(info: dict[str, object], key: str, *, default: float = 0.0) -> float:
    return float_info(info, key, default=default)


def _player_telemetry_float(
    telemetry_data: Mapping[str, object] | None,
    key: str,
) -> float:
    player_data = _player_telemetry_data(telemetry_data)
    if player_data is None:
        return 0.0
    return float_info(player_data, key)


def _ko_star_count(
    info: dict[str, object],
    telemetry_data: Mapping[str, object] | None,
) -> int | None:
    value = optional_int_info(info, "ko_star_count", minimum=0)
    if value is not None:
        return value
    player_data = _player_telemetry_data(telemetry_data)
    if player_data is None:
        return None
    return optional_int_info(player_data, "ko_star_count", minimum=0)


def _ko_star_reward_event(
    info: dict[str, object],
    *,
    env_step: int,
) -> KoStarRewardEvent | None:
    if info.get("ko_star_reward_event") is not True:
        return None
    previous_count = _info_int(info, "ko_star_reward_previous_count")
    current_count = _info_int(info, "ko_star_reward_current_count")
    gained = _info_int(info, "ko_star_reward_gain")
    reward = _info_float(info, "ko_star_reward_value")
    if previous_count is None or current_count is None or gained is None:
        return None
    return KoStarRewardEvent(
        env_step=env_step,
        previous_count=previous_count,
        current_count=current_count,
        gained=gained,
        reward=reward,
    )


def _info_int(info: dict[str, object], key: str) -> int | None:
    return optional_int_info(info, key)


def _edge_ratio(telemetry_data: Mapping[str, object] | None) -> float:
    player_data = _player_telemetry_data(telemetry_data)
    if player_data is None:
        return 0.0
    ratio = _raw_edge_ratio(player_data)
    if ratio is None:
        return 0.0
    return clamp(ratio, -1.0, 1.0)


def _outside_edge_excess_ratio(telemetry_data: Mapping[str, object] | None) -> float:
    player_data = _player_telemetry_data(telemetry_data)
    if player_data is None:
        return 0.0
    ratio = _raw_edge_ratio(player_data)
    if ratio is None:
        return 0.0
    absolute_ratio = abs(ratio)
    if absolute_ratio <= 1.10:
        return 0.0
    return absolute_ratio - 1.0


def _raw_edge_ratio(player_data: Mapping[object, object]) -> float | None:
    offset = _mapping_float(player_data, "signed_lateral_offset")
    side_radius = (
        _mapping_float(player_data, "current_radius_left")
        if offset >= 0.0
        else _mapping_float(player_data, "current_radius_right")
    )
    if side_radius <= 0.0:
        return None
    return offset / side_radius


def _snapshot_telemetry_data(
    snapshot: _LiveEpisodeSnapshot,
) -> Mapping[str, object] | None:
    telemetry_data = getattr(snapshot, "telemetry_data", None)
    if not isinstance(telemetry_data, Mapping):
        return None
    return telemetry_data


def _player_telemetry_data(
    telemetry_data: Mapping[str, object] | None,
) -> Mapping[object, object] | None:
    if telemetry_data is None:
        return None
    player_data = telemetry_data.get("player")
    if not isinstance(player_data, Mapping):
        return None
    return player_data


def _mapping_float(mapping: Mapping[object, object], key: str) -> float:
    return float_info(mapping, key)
