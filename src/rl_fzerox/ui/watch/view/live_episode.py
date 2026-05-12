# src/rl_fzerox/ui/watch/view/live_episode.py
from __future__ import annotations

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
    current_return: float
    current_progress: float
    max_progress: float


@dataclass(slots=True)
class EpisodeLiveSeriesTracker:
    episode: int | None = None
    env_steps: list[int] = field(default_factory=list)
    speed_kph: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
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
            self.current_return = 0.0
            self.current_progress = 0.0
            self.max_progress = 0.0
        if not snapshot.policy_decision_frame:
            return
        env_step = _env_step(snapshot.info, action_repeat=action_repeat)
        progress = _progress_fraction(snapshot.info)
        speed_kph = _speed_kph(snapshot.info)
        self.current_return = float(snapshot.episode_reward)
        self.current_progress = progress
        self.max_progress = max(self.max_progress, progress)
        if self.env_steps and env_step == self.env_steps[-1]:
            self.speed_kph[-1] = speed_kph
            self.returns[-1] = self.current_return
            return
        self.env_steps.append(env_step)
        self.speed_kph.append(speed_kph)
        self.returns.append(self.current_return)

    def snapshot(self) -> EpisodeLiveSeriesSnapshot | None:
        if self.episode is None:
            return None
        return EpisodeLiveSeriesSnapshot(
            episode=self.episode,
            env_steps=tuple(self.env_steps),
            speed_kph=tuple(self.speed_kph),
            returns=tuple(self.returns),
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
