# tests/core/training/test_training_preload.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.config.schema import (
    TrainConfig,
)
from rl_fzerox.core.training.session.model import (
    maybe_resume_training_model,
)


def test_maybe_resume_training_model_loads_weights_only_artifact(tmp_path: Path) -> None:
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

    resumed_model = maybe_resume_training_model(
        model=model,
        train_env=None,
        train_config=TrainConfig(
            resume_run_dir=run_dir,
            resume_artifact="latest",
            resume_mode="weights_only",
        ),
    )

    assert resumed_model is model
    assert model.calls == [(str(model_path.resolve()), True, "cpu")]


def test_maybe_resume_training_model_loads_full_model_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "ppo_cnn_0042"
    run_dir.mkdir(parents=True)
    model_path = run_dir / "best_model.zip"
    model_path.write_bytes(b"checkpoint")
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

    class _LoadedModel:
        pass

    loaded_model = _LoadedModel()
    load_calls: list[tuple[str, object, str]] = []

    class _FakeAlgorithm:
        @classmethod
        def load(cls, path: str, *, env: object, device: str) -> _LoadedModel:
            load_calls.append((path, env, device))
            return loaded_model

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.preload.resolve_training_algorithm_class",
        lambda algorithm: _FakeAlgorithm,
    )
    train_env = object()

    resumed_model = maybe_resume_training_model(
        model=object(),
        train_env=train_env,
        train_config=TrainConfig(
            resume_run_dir=run_dir,
            resume_artifact="best",
            resume_mode="full_model",
            device="cuda",
        ),
    )

    assert resumed_model is loaded_model
    assert load_calls == [(str(model_path.resolve()), train_env, "cuda")]


def test_maybe_resume_training_model_ignores_stale_non_train_config(
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

    resumed_model = maybe_resume_training_model(
        model=model,
        train_env=None,
        train_config=TrainConfig(
            resume_run_dir=run_dir,
            resume_artifact="latest",
            resume_mode="weights_only",
        ),
    )

    assert resumed_model is model
    assert model.calls == [(str(model_path.resolve()), True, "cpu")]


def test_maybe_resume_training_model_rejects_non_string_train_keys(
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
        maybe_resume_training_model(
            model=_FakeModel(),
            train_env=None,
            train_config=TrainConfig(
                resume_run_dir=run_dir,
                resume_artifact="latest",
                resume_mode="weights_only",
            ),
        )


def test_maybe_resume_training_model_rejects_algorithm_mismatch(tmp_path: Path) -> None:
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
        match="Resume checkpoint algorithm mismatch",
    ):
        maybe_resume_training_model(
            model=_FakeModel(),
            train_env=None,
            train_config=TrainConfig(
                algorithm="maskable_recurrent_ppo",
                resume_run_dir=run_dir,
                resume_artifact="latest",
                resume_mode="weights_only",
            ),
        )
