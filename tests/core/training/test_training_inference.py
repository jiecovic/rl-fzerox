# tests/core/training/test_training_inference.py
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.training.inference import LoadedPolicy, PolicyRunner


class _FakePolicy:
    def __init__(self, action: list[int]) -> None:
        self._action = np.array(action, dtype=np.int64)
        self.deterministic_calls: list[bool] = []

    def predict(self, observation: ObservationValue, deterministic: bool = True):
        _ = observation
        self.deterministic_calls.append(deterministic)
        return self._action.copy(), None


def test_policy_runner_reloads_updated_policy_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")

    loaded_policy = LoadedPolicy(
        run_dir=tmp_path,
        policy_path=policy_path,
        artifact="latest",
    )
    runner = PolicyRunner(loaded_policy, _FakePolicy([2, 0]))

    observation = np.zeros((84, 116, 12), dtype=np.uint8)
    assert runner.predict(observation).tolist() == [2, 0]

    policy_path.write_bytes(b"v2")
    stat = policy_path.stat()
    os.utime(policy_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    monkeypatch.setattr(
        "rl_fzerox.core.training.inference.resolve_policy_artifact_path",
        lambda run_dir, *, artifact: policy_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.core.training.inference._load_saved_policy",
        lambda path: _FakePolicy([4, 1]),
    )

    assert runner.predict(observation).tolist() == [4, 1]
    assert runner.reload_age_seconds < 1.0


def test_policy_runner_reports_reload_age_since_initial_load(tmp_path: Path) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")

    loaded_policy = LoadedPolicy(
        run_dir=tmp_path,
        policy_path=policy_path,
        artifact="latest",
    )
    runner = PolicyRunner(loaded_policy, _FakePolicy([2, 0]))

    time.sleep(0.01)

    assert runner.reload_age_seconds > 0.0


def test_policy_runner_can_sample_non_deterministic_actions(tmp_path: Path) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")
    fake_policy = _FakePolicy([2, 0])

    runner = PolicyRunner(
        LoadedPolicy(
            run_dir=tmp_path,
            policy_path=policy_path,
            artifact="latest",
        ),
        fake_policy,
    )

    observation = np.zeros((84, 116, 12), dtype=np.uint8)
    assert runner.predict(observation, deterministic=False).tolist() == [2, 0]

    assert fake_policy.deterministic_calls == [False]


def test_policy_runner_exposes_reload_error_until_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")

    runner = PolicyRunner(
        LoadedPolicy(
            run_dir=tmp_path,
            policy_path=policy_path,
            artifact="latest",
        ),
        _FakePolicy([2, 0]),
    )

    policy_path.write_bytes(b"v2")
    stat = policy_path.stat()
    os.utime(policy_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    monkeypatch.setattr(
        "rl_fzerox.core.training.inference.resolve_policy_artifact_path",
        lambda run_dir, *, artifact: policy_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.core.training.inference._load_saved_policy",
        lambda path: (_ for _ in ()).throw(RuntimeError("bad checkpoint")),
    )

    observation = np.zeros((84, 116, 12), dtype=np.uint8)
    assert runner.predict(observation).tolist() == [2, 0]
    assert runner.reload_error == "bad checkpoint"
    assert runner.last_reload_error == "bad checkpoint"

    monkeypatch.setattr(
        "rl_fzerox.core.training.inference._load_saved_policy",
        lambda path: _FakePolicy([4, 1]),
    )

    assert runner.predict(observation).tolist() == [4, 1]
    assert runner.reload_error is None
    assert runner.last_reload_error == "bad checkpoint"
