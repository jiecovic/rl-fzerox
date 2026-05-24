# tests/core/training/test_training_startup.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common import logger as sb3_logger

from rl_fzerox.core.training.runs.paths import explicit_run_paths
from rl_fzerox.core.training.session.model.startup import (
    _format_bytes,
    _format_parameter_count,
    _minibatch_observation_bytes,
    _training_memory_summary,
    build_tensorboard_logger,
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


def test_tensorboard_logger_offsets_global_step_for_forked_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_paths = explicit_run_paths(tmp_path / "run")
    seen_steps: list[int] = []

    class FakeKVWriter(sb3_logger.KVWriter):
        def write(
            self,
            key_values: dict[str, float],
            key_excluded: dict[str, tuple[str, ...]],
            step: int = 0,
        ) -> None:
            del key_values, key_excluded
            seen_steps.append(step)

        def close(self) -> None:
            return None

    base_logger = sb3_logger.Logger(
        folder=str(run_paths.tensorboard_dir),
        output_formats=[FakeKVWriter()],
    )
    monkeypatch.setattr(
        sb3_logger,
        "configure",
        lambda folder, format_strings: base_logger,
    )

    logger = build_tensorboard_logger(run_paths, step_offset=816_040)
    logger.record("rollout/ep_rew_mean", 4.2)
    logger.dump(step=7)

    assert seen_steps == [816_047]
