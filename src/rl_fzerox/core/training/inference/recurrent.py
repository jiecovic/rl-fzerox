# src/rl_fzerox/core/training/inference/recurrent.py
"""Recurrent policy state helpers for step-by-step inference."""

from __future__ import annotations

import numpy as np

from fzerox_emulator.arrays import BoolArray, PolicyState


class RecurrentInferenceState:
    """Mutable recurrent state passed between policy predictions."""

    def __init__(self) -> None:
        self._predict_state: PolicyState = None
        self._episode_start = _episode_start_array(True)

    @property
    def predict_state(self) -> PolicyState:
        return self._predict_state

    @property
    def episode_start(self) -> BoolArray:
        return self._episode_start

    def reset(self) -> None:
        self._predict_state = None
        self._episode_start = _episode_start_array(True)

    def advance(self, next_state: PolicyState) -> None:
        self._predict_state = next_state
        self._episode_start = _episode_start_array(False)


def _episode_start_array(value: bool) -> BoolArray:
    return np.array([value], dtype=bool)
