# tests/core/training/test_training_preload.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.config.schema import (
    TrainConfig,
)
from rl_fzerox.core.training.session.model import (
    maybe_preload_training_parameters,
)


def test_maybe_preload_training_parameters_loads_requested_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ppo_cnn_0042"
    run_dir.mkdir(parents=True)
    model_path = run_dir / "latest_model.zip"
    model_path.write_bytes(b"checkpoint")
    train_config_path = run_dir / "train_config.yaml"
    train_config_path.write_text(
        "\n".join(
            [
                "seed: 7",
                "emulator:",
                f"  core_path: {tmp_path / 'core.so'}",
                f"  rom_path: {tmp_path / 'rom.n64'}",
                "env: {}",
                "reward: {}",
                "policy: {}",
                "curriculum: {}",
                "train:",
                "  algorithm: maskable_ppo",
                "  total_timesteps: 1000",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "core.so").touch()
    (tmp_path / "rom.n64").touch()

    class _FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"
            self.calls: list[tuple[str, bool, str]] = []

        def set_parameters(
            self,
            load_path_or_dict: str,
            *,
            exact_match: bool,
            device: str,
        ) -> None:
            self.calls.append((load_path_or_dict, exact_match, device))

    model = _FakeModel()

    maybe_preload_training_parameters(
        model=model,
        train_config=TrainConfig(
            init_run_dir=run_dir,
            init_artifact="latest",
        ),
    )

    assert model.calls == [(str(model_path.resolve()), True, "cpu")]


def test_maybe_preload_training_parameters_ignores_stale_non_train_config(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "ppo_cnn_0042"
    run_dir.mkdir(parents=True)
    model_path = run_dir / "latest_model.zip"
    model_path.write_bytes(b"checkpoint")
    (run_dir / "train_config.yaml").write_text(
        "\n".join(
            [
                "seed: 7",
                "emulator: {}",
                "env: {}",
                "reward:",
                "  energy_gain_reward_scale: 12.0",
                "  energy_gain_collision_cooldown_frames: 240",
                "policy: {}",
                "curriculum: {}",
                "train:",
                "  algorithm: maskable_ppo",
                "  total_timesteps: 1000",
            ]
        ),
        encoding="utf-8",
    )

    class _FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"
            self.calls: list[tuple[str, bool, str]] = []

        def set_parameters(
            self,
            load_path_or_dict: str,
            *,
            exact_match: bool,
            device: str,
        ) -> None:
            self.calls.append((load_path_or_dict, exact_match, device))

    model = _FakeModel()

    maybe_preload_training_parameters(
        model=model,
        train_config=TrainConfig(
            init_run_dir=run_dir,
            init_artifact="latest",
        ),
    )

    assert model.calls == [(str(model_path.resolve()), True, "cpu")]


def test_maybe_preload_training_parameters_rejects_non_string_train_keys(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "ppo_cnn_0042"
    run_dir.mkdir(parents=True)
    (run_dir / "latest_model.zip").write_bytes(b"checkpoint")
    (run_dir / "train_config.yaml").write_text(
        "\n".join(
            [
                "train:",
                "  algorithm: maskable_ppo",
                "  total_timesteps: 1000",
                "  1: malformed",
            ]
        ),
        encoding="utf-8",
    )

    class _FakeModel:
        device = "cpu"

        def set_parameters(
            self,
            load_path_or_dict: str,
            *,
            exact_match: bool,
            device: str,
        ) -> None:
            raise AssertionError("set_parameters should not be reached on malformed config")

    with pytest.raises(ValueError, match="train keys must be strings"):
        maybe_preload_training_parameters(
            model=_FakeModel(),
            train_config=TrainConfig(
                init_run_dir=run_dir,
                init_artifact="latest",
            ),
        )


def test_maybe_preload_training_parameters_rejects_algorithm_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ppo_cnn_0042"
    run_dir.mkdir(parents=True)
    (run_dir / "latest_model.zip").write_bytes(b"checkpoint")
    (tmp_path / "core.so").touch()
    (tmp_path / "rom.n64").touch()
    (run_dir / "train_config.yaml").write_text(
        "\n".join(
            [
                "seed: 7",
                "emulator:",
                f"  core_path: {tmp_path / 'core.so'}",
                f"  rom_path: {tmp_path / 'rom.n64'}",
                "env: {}",
                "reward: {}",
                "policy: {}",
                "curriculum: {}",
                "train:",
                "  algorithm: maskable_ppo",
                "  total_timesteps: 1000",
            ]
        ),
        encoding="utf-8",
    )

    class _FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"

        def set_parameters(
            self,
            load_path_or_dict: str,
            *,
            exact_match: bool,
            device: str,
        ) -> None:
            raise AssertionError("set_parameters should not be reached on mismatch")

    with pytest.raises(
        RuntimeError,
        match="Warm-start checkpoint algorithm mismatch",
    ):
        maybe_preload_training_parameters(
            model=_FakeModel(),
            train_config=TrainConfig(
                algorithm="maskable_recurrent_ppo",
                init_run_dir=run_dir,
                init_artifact="latest",
            ),
        )


