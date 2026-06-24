# tests/core/training/test_sb3_env_adapter.py
from __future__ import annotations

from collections.abc import Sequence

from rl_fzerox.core.envs.engine.reset.track_sampling import TrackSamplingQueuedReset
from rl_fzerox.core.training.session.callbacks.sb3.env_adapter import training_env_adapter


class _FakeVecEnv:
    def __init__(self, lengths: list[object] | None = None) -> None:
        self.lengths = lengths or []
        self.calls: list[tuple[str, tuple[object, ...], Sequence[int] | None]] = []

    def env_method(
        self,
        method_name: str,
        *method_args: object,
        indices: Sequence[int] | None = None,
        **method_kwargs: object,
    ) -> list[object]:
        del method_kwargs
        self.calls.append((method_name, method_args, indices))
        if method_name == "track_sampling_reset_queue_length":
            return self.lengths
        return []


def test_training_env_adapter_normalizes_track_sampling_queue_lengths() -> None:
    adapter = training_env_adapter(_FakeVecEnv([2, 0.0, "missing"]))

    assert adapter.track_sampling_reset_queue_lengths() == (2, 0, 0)


def test_training_env_adapter_routes_indexed_track_sampling_queue_extensions() -> None:
    fake_env = _FakeVecEnv()
    adapter = training_env_adapter(fake_env)
    queued_reset = TrackSamplingQueuedReset(course_id="mute_city", deficit_lane="uniform")

    adapter.extend_track_sampling_reset_queue(env_index=3, queued_resets=(queued_reset,))

    assert fake_env.calls == [
        ("extend_track_sampling_reset_queue", ((queued_reset,),), [3]),
    ]
