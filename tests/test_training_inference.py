# tests/test_training_inference.py
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from rl_fzerox.core.training.inference import LoadedPolicy, PolicyRunner


class _FakePolicy:
    def __init__(self, action: list[int]) -> None:
        self._action = np.array(action, dtype=np.int64)

    def predict(self, observation: np.ndarray, deterministic: bool = True):
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

    observation = np.zeros((120, 160, 12), dtype=np.uint8)
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
