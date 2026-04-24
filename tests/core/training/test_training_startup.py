# tests/core/training/test_training_startup.py
from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces

from rl_fzerox.core.training.session.model.startup import (
    _format_bytes,
    _format_parameter_count,
    _minibatch_observation_bytes,
    _training_memory_summary,
)


def test_minibatch_observation_bytes_uses_float32_tensors_for_dict_observations() -> None:
    observation_space = spaces.Dict(
        {
            "image": spaces.Box(low=0, high=255, shape=(60, 76, 12), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(51,), dtype=np.float32),
        }
    )

    bytes_used = _minibatch_observation_bytes(
        observation_space=observation_space,
        batch_size=256,
    )

    expected = (256 * 60 * 76 * 12 * 4) + (256 * 51 * 4)
    assert bytes_used == expected


def test_training_memory_summary_skips_cuda_lines_for_cpu_models() -> None:
    model = th.nn.Linear(4, 2)
    observation_space = spaces.Box(low=0, high=255, shape=(60, 76, 12), dtype=np.uint8)

    summary = _training_memory_summary(
        model=model,
        observation_space=observation_space,
        batch_size=256,
    )

    assert summary.parameter_count == "10"
    assert summary.cuda_now is None
    assert summary.cuda_estimate is None


def test_format_bytes_uses_binary_units() -> None:
    assert _format_bytes(999) == "999B"
    assert _format_bytes(1024) == "1.0KiB"
    assert _format_bytes(5 * 1024 * 1024) == "5.0MiB"


def test_format_parameter_count_uses_metric_units() -> None:
    assert _format_parameter_count(999) == "999"
    assert _format_parameter_count(1_250) == "1.2K"
    assert _format_parameter_count(1_250_000) == "1.2M"
    assert _format_parameter_count(1_250_000_000) == "1.2B"
