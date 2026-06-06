# tests/core/career_mode/runner/test_policy_drive.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.envs.engine.stepping import policy_drive_info


def test_policy_drive_info_drops_training_lifecycle_state() -> None:
    info = policy_drive_info(
        {
            "game_mode": "gp_race",
            "termination_reason": "progress_stalled",
            "terminated": False,
            "truncated": True,
            "truncation_reason": "progress_stalled",
        }
    )

    assert info == {"game_mode": "gp_race"}


def test_policy_drive_info_keeps_native_terminal_reason() -> None:
    info = policy_drive_info(
        {
            "game_mode": "gp_race",
            "termination_reason": "finished",
            "truncated": False,
        }
    )

    assert info == {
        "game_mode": "gp_race",
        "termination_reason": "finished",
    }


def test_career_policy_race_uses_policy_drive_boundary() -> None:
    source = Path("src/rl_fzerox/core/career_mode/runner/policy.py").read_text(encoding="utf-8")

    assert "FZeroXEnvEngine" not in source
    assert "PolicyDriveRuntime" in source
    assert "WatchEnvStep" not in source
    assert "step_watch(" not in source
    assert "step_control_watch(" not in source


def test_env_policy_drive_runtime_owns_engine_policy_drive_calls() -> None:
    source = Path("src/rl_fzerox/core/envs/engine/policy_drive/runtime.py").read_text(
        encoding="utf-8"
    )

    assert "FZeroXEnvEngine" not in source
    assert "step_watch(" not in source
    assert "step_control_watch(" not in source
    assert "WatchEnvStep" not in source
    assert "build_engine_runtime_components" in source
