# tests/ui/viewer_runtime_support.py
"""Shared helpers for Watch runtime tests.

The helpers keep policy and observation fixtures out of the behavior-specific
test files, so each split runtime test module can import only the production
surface it is actually verifying.
"""

from __future__ import annotations

import numpy as np

from fzerox_emulator.arrays import Float32Array, UInt8Array


class PolicyStub:
    def predict(
        self,
        observation: object,
        state: object | None = None,
        episode_start: object | None = None,
        deterministic: bool = True,
    ) -> tuple[int, object | None]:
        _ = (observation, episode_start, deterministic)
        return 0, state


def sample_image() -> UInt8Array:
    return np.zeros((2, 2, 3), dtype=np.uint8)


def sample_state(values: list[float]) -> Float32Array:
    return np.asarray(values, dtype=np.float32)
