# src/rl_fzerox/ui/watch/view/live_episode.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


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
    returns: tuple[float, ...]
    lateral_offset: tuple[float, ...]
    outside_edge_excess_ratio: tuple[float, ...]
    future_local_nearest_segment_distance: tuple[float, ...]
    current_future_local_nearest_segment_index: int | None
    current_return: float
    current_progress: float
    max_progress: float


@dataclass(slots=True)
class EpisodeLiveSeriesTracker:
    episode: int | None = None
    env_steps: list[int] = field(default_factory=list)
    speed_kph: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    lateral_offset: list[float] = field(default_factory=list)
    outside_edge_excess_ratio: list[float] = field(default_factory=list)
    future_local_nearest_segment_distance: list[float] = field(default_factory=list)
    current_future_local_nearest_segment_index: int | None = None
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
            self.returns = []
            self.lateral_offset = []
            self.outside_edge_excess_ratio = []
            self.future_local_nearest_segment_distance = []
            self.current_future_local_nearest_segment_index = None
            self.current_return = 0.0
            self.current_progress = 0.0
            self.max_progress = 0.0
        if not snapshot.policy_decision_frame:
            return
        env_step = _env_step(snapshot.info, action_repeat=action_repeat)
        progress = _progress_fraction(snapshot.info)
        speed_kph = _speed_kph(snapshot.info)
        lateral_offset = _player_telemetry_float(snapshot, "signed_lateral_offset")
        outside_edge_excess_ratio = _outside_edge_excess_ratio(snapshot)
        future_distance = _player_telemetry_float(
            snapshot,
            "future_local_nearest_segment_distance",
        )
        future_segment_index = _player_telemetry_optional_int(
            snapshot,
            "future_local_nearest_segment_index",
        )
        self.current_return = float(snapshot.episode_reward)
        self.current_progress = progress
        self.max_progress = max(self.max_progress, progress)
        self.current_future_local_nearest_segment_index = future_segment_index
        if self.env_steps and env_step == self.env_steps[-1]:
            self.speed_kph[-1] = speed_kph
            self.returns[-1] = self.current_return
            self.lateral_offset[-1] = lateral_offset
            self.outside_edge_excess_ratio[-1] = outside_edge_excess_ratio
            self.future_local_nearest_segment_distance[-1] = future_distance
            return
        self.env_steps.append(env_step)
        self.speed_kph.append(speed_kph)
        self.returns.append(self.current_return)
        self.lateral_offset.append(lateral_offset)
        self.outside_edge_excess_ratio.append(outside_edge_excess_ratio)
        self.future_local_nearest_segment_distance.append(future_distance)

    def snapshot(self) -> EpisodeLiveSeriesSnapshot | None:
        if self.episode is None:
            return None
        return EpisodeLiveSeriesSnapshot(
            episode=self.episode,
            env_steps=tuple(self.env_steps),
            speed_kph=tuple(self.speed_kph),
            returns=tuple(self.returns),
            lateral_offset=tuple(self.lateral_offset),
            outside_edge_excess_ratio=tuple(self.outside_edge_excess_ratio),
            future_local_nearest_segment_distance=tuple(self.future_local_nearest_segment_distance),
            current_future_local_nearest_segment_index=(
                self.current_future_local_nearest_segment_index
            ),
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
    value = info.get("speed_kph")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, float(value))
    return 0.0


def _player_telemetry_float(snapshot: _LiveEpisodeSnapshot, key: str) -> float:
    player_data = _player_telemetry_data(snapshot)
    if player_data is None:
        return 0.0
    value = player_data.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _player_telemetry_optional_int(snapshot: _LiveEpisodeSnapshot, key: str) -> int | None:
    player_data = _player_telemetry_data(snapshot)
    if player_data is None:
        return None
    value = player_data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    return None


def _outside_edge_excess_ratio(snapshot: _LiveEpisodeSnapshot) -> float:
    player_data = _player_telemetry_data(snapshot)
    if player_data is None:
        return 0.0
    offset = _mapping_float(player_data, "signed_lateral_offset")
    side_radius = (
        _mapping_float(player_data, "current_radius_left")
        if offset >= 0.0
        else _mapping_float(player_data, "current_radius_right")
    )
    if side_radius <= 0.0:
        return 0.0
    return max(0.0, (abs(offset) / side_radius) - 1.0)


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
