# src/rl_fzerox/core/training/session/callbacks/sb3/env_adapter.py
"""Narrow adapter around SB3 vector-env methods used by callbacks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rl_fzerox.core.engine_tuning import EngineTuningResetSampler
from rl_fzerox.core.envs.engine.reset.track_sampling import TrackSamplingQueuedReset
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig


@runtime_checkable
class EnvMethodVecEnv(Protocol):
    def env_method(
        self,
        method_name: str,
        *method_args: object,
        indices: Sequence[int] | None = None,
        **method_kwargs: object,
    ) -> list[object]: ...


@dataclass(frozen=True, slots=True)
class TrainingEnvAdapter:
    """Typed access to project-owned env methods exposed through SB3 VecEnv."""

    env: EnvMethodVecEnv

    def set_track_sampling_config(self, config: TrackSamplingConfig) -> None:
        self.env.env_method("set_track_sampling_config", config)

    def set_track_sampling_weights(self, weights: Mapping[str, float]) -> None:
        self.env.env_method("set_track_sampling_weights", weights)

    def track_sampling_reset_queue_lengths(self) -> tuple[int, ...]:
        raw_lengths = self.env.env_method("track_sampling_reset_queue_length")
        return tuple(
            int(length) if isinstance(length, int | float) else 0 for length in raw_lengths
        )

    def clear_track_sampling_reset_queue(self) -> None:
        self.env.env_method("clear_track_sampling_reset_queue")

    def extend_track_sampling_reset_queue(
        self,
        *,
        env_index: int,
        queued_resets: Sequence[TrackSamplingQueuedReset],
    ) -> None:
        self.env.env_method(
            "extend_track_sampling_reset_queue",
            tuple(queued_resets),
            indices=[env_index],
        )

    def set_engine_tuning_sampler(self, sampler: EngineTuningResetSampler) -> None:
        self.env.env_method("set_engine_tuning_sampler", sampler)


def training_env_adapter(env: object) -> TrainingEnvAdapter:
    if not isinstance(env, EnvMethodVecEnv):
        raise RuntimeError("SB3 training env must expose env_method()")
    return TrainingEnvAdapter(env=env)
