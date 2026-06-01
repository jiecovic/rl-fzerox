# tests/core/training/test_training_preload.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from torch import nn

from rl_fzerox.core.runtime_spec.schema import TrainConfig
from rl_fzerox.core.training.session.model.preload import maybe_resume_training_model


class _DummyModel:
    def __init__(self, *, auxiliary_state_head_arch: tuple[int, ...] | None) -> None:
        if auxiliary_state_head_arch is not None:
            trunk_layers: list[nn.Module] = []
            input_dim = 8
            for width in auxiliary_state_head_arch:
                trunk_layers.append(nn.Linear(input_dim, width))
                trunk_layers.append(nn.ReLU())
                input_dim = width
            trunk: nn.Module = nn.Identity() if not trunk_layers else nn.Sequential(*trunk_layers)
            self.policy = SimpleNamespace(_auxiliary_state_heads=SimpleNamespace(trunk=trunk))
        else:
            self.policy = SimpleNamespace()
        self.device = "cpu"
        self.set_parameters_calls: list[tuple[str, bool, str]] = []

    def set_parameters(self, path: str, *, exact_match: bool, device: str) -> None:
        self.set_parameters_calls.append((path, exact_match, device))


def _source_run_config(*, auxiliary_state_head_arch: tuple[int, ...] | None) -> SimpleNamespace:
    return SimpleNamespace(
        policy=SimpleNamespace(
            auxiliary_state_enabled=auxiliary_state_head_arch is not None,
            auxiliary_state_head_arch=auxiliary_state_head_arch or (),
        )
    )


def _resume_train_config(
    tmp_path: Path,
    *,
    resume_mode: Literal["weights_only", "full_model"] = "weights_only",
) -> TrainConfig:
    return TrainConfig(
        algorithm="maskable_hybrid_action_ppo",
        resume_run_dir=tmp_path,
        resume_artifact="latest",
        resume_mode=resume_mode,
        resume_source_algorithm="maskable_hybrid_action_ppo",
        device="cpu",
    )


def test_weights_only_resume_relaxes_exact_match_when_aux_bank_presence_differs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _DummyModel(auxiliary_state_head_arch=(128,))
    train_config = _resume_train_config(tmp_path)
    model_path = tmp_path / "model.zip"

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.preload.load_train_run_config",
        lambda run_dir: _source_run_config(auxiliary_state_head_arch=None),
    )
    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.preload.resolve_model_artifact_path",
        lambda run_dir, artifact: model_path,
    )

    maybe_resume_training_model(
        model=model,
        train_env=object(),
        train_config=train_config,
    )

    assert model.set_parameters_calls == [(str(model_path), False, "cpu")]


def test_weights_only_resume_keeps_exact_match_when_aux_bank_signature_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _DummyModel(auxiliary_state_head_arch=(128,))
    train_config = _resume_train_config(tmp_path)
    model_path = tmp_path / "model.zip"

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.preload.load_train_run_config",
        lambda run_dir: _source_run_config(auxiliary_state_head_arch=(128,)),
    )
    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.preload.resolve_model_artifact_path",
        lambda run_dir, artifact: model_path,
    )

    maybe_resume_training_model(
        model=model,
        train_env=object(),
        train_config=train_config,
    )

    assert model.set_parameters_calls == [(str(model_path), True, "cpu")]


def test_weights_only_resume_relaxes_exact_match_when_aux_bank_arch_differs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _DummyModel(auxiliary_state_head_arch=(64,))
    train_config = _resume_train_config(tmp_path)
    model_path = tmp_path / "model.zip"

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.preload.load_train_run_config",
        lambda run_dir: _source_run_config(auxiliary_state_head_arch=(128,)),
    )
    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.preload.resolve_model_artifact_path",
        lambda run_dir, artifact: model_path,
    )

    maybe_resume_training_model(
        model=model,
        train_env=object(),
        train_config=train_config,
    )

    assert model.set_parameters_calls == [(str(model_path), False, "cpu")]


def test_full_model_resume_rejects_aux_bank_signature_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _DummyModel(auxiliary_state_head_arch=(128,))
    train_config = _resume_train_config(tmp_path, resume_mode="full_model")

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.preload.load_train_run_config",
        lambda run_dir: _source_run_config(auxiliary_state_head_arch=(64,)),
    )

    with pytest.raises(RuntimeError, match="Full-model resume requires the same auxiliary-state"):
        maybe_resume_training_model(
            model=model,
            train_env=object(),
            train_config=train_config,
        )
