# src/rl_fzerox/ui/watch/view/live_episode.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from rl_fzerox.core.envs.observations.state.utils import clamp


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
    current_return: float
    current_progress: float
    max_progress: float


@dataclass(slots=True)
class EpisodeLiveSeriesTracker:
    episode: int | None = None
    env_steps: list[int] = field(default_factory=list)
    speed_kph: list[float] = field(default_factory=list)
    step_rewards: list[float] = field(default_factory=list)
    progress_speed_multiplier: list[float] = field(default_factory=list)
    position_progress_multiplier: list[float] = field(default_factory=list)
    progress_speed_position_multiplier: list[float] = field(default_factory=list)
    edge_ratio: list[float] = field(default_factory=list)
    outside_edge_excess_ratio: list[float] = field(default_factory=list)
    height_above_ground: list[float] = field(default_factory=list)
    current_return: float = 0.0
    current_progress: float = 0.0
    max_progress: float = 0.0

    def observe_snapshot(
        self,
        snapshot: _LiveEpisodeSnapshot,
        *,
        action_repeat: int,
    ) -> None:
        if self.episode != snapshot.episode:
            self.episode = snapshot.episode
            self.env_steps = []
            self.speed_kph = []
            self.step_rewards = []
            self.progress_speed_multiplier = []
            self.position_progress_multiplier = []
            self.progress_speed_position_multiplier = []
            self.edge_ratio = []
            self.outside_edge_excess_ratio = []
            self.height_above_ground = []
            self.current_return = 0.0
            self.current_progress = 0.0
            self.max_progress = 0.0
        if not snapshot.policy_decision_frame:
            return
        env_step = _env_step(snapshot.info, action_repeat=action_repeat)
        progress = _progress_fraction(snapshot.info)
        speed_kph = _speed_kph(snapshot.info)
        step_reward = _info_float(snapshot.info, "step_reward")
        speed_multiplier = _info_float(snapshot.info, "progress_speed_multiplier", default=1.0)
        position_multiplier = _info_float(
            snapshot.info,
            "position_progress_multiplier",
            default=1.0,
        )
        combined_multiplier = _info_float(
            snapshot.info,
            "progress_speed_position_multiplier",
            default=speed_multiplier * position_multiplier,
        )
        edge_ratio = _edge_ratio(snapshot)
        outside_edge_excess_ratio = _outside_edge_excess_ratio(snapshot)
        height_above_ground = _player_telemetry_float(snapshot, "height_above_ground")
        self.current_return = float(snapshot.episode_reward)
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
            current_return=self.current_return,
            current_progress=self.current_progress,
            max_progress=self.max_progress,
        )


def _env_step(info: dict[str, object], *, action_repeat: int) -> int:
    episode_frames = info.get("episode_step")
    if isinstance(episode_frames, int | float) and not isinstance(episode_frames, bool):
        repeat = max(1, int(action_repeat))
        frames = max(0, int(episode_frames))
        return (frames + repeat - 1) // repeat
    return 0


def _progress_fraction(info: dict[str, object]) -> float:
    value = info.get("episode_completion_fraction")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def _speed_kph(info: dict[str, object]) -> float:
    return max(0.0, _info_float(info, "speed_kph"))


def _info_float(info: dict[str, object], key: str, *, default: float = 0.0) -> float:
    value = info.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return default


def _player_telemetry_float(snapshot: _LiveEpisodeSnapshot, key: str) -> float:
    player_data = _player_telemetry_data(snapshot)
    if player_data is None:
        return 0.0
    value = player_data.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _edge_ratio(snapshot: _LiveEpisodeSnapshot) -> float:
    player_data = _player_telemetry_data(snapshot)
    if player_data is None:
        return 0.0
    ratio = _raw_edge_ratio(player_data)
    if ratio is None:
        return 0.0
    return clamp(ratio, -1.0, 1.0)


def _outside_edge_excess_ratio(snapshot: _LiveEpisodeSnapshot) -> float:
    player_data = _player_telemetry_data(snapshot)
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


def _player_telemetry_data(snapshot: _LiveEpisodeSnapshot) -> Mapping[object, object] | None:
    telemetry_data = getattr(snapshot, "telemetry_data", None)
    if not isinstance(telemetry_data, Mapping):
        return None
    player_data = telemetry_data.get("player")
    if not isinstance(player_data, Mapping):
        return None
    return player_data


def _mapping_float(mapping: Mapping[object, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0
