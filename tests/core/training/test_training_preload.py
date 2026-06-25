# tests/core/training/test_training_preload.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import ValidationError
from torch import nn

from rl_fzerox.core.runtime_spec.schema import (
    ActionConfig,
    EmulatorConfig,
    EnvConfig,
    PolicyActionBiasConfig,
    PolicyConfig,
    TrainAppConfig,
    TrainConfig,
)
from rl_fzerox.core.training.session.model.action_bias import MODEL_ACTION_BIAS_OFFSETS_ATTR
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


def _resume_train_config(
    tmp_path: Path,
    *,
    resume_mode: Literal["weights_only", "full_model"] = "weights_only",
    source_auxiliary_state_head_arch: tuple[int, ...] | None = None,
) -> TrainConfig:
    return TrainConfig(
        algorithm="maskable_hybrid_action_ppo",
        resume_run_dir=tmp_path,
        resume_artifact="latest",
        resume_mode=resume_mode,
        resume_source_algorithm="maskable_hybrid_action_ppo",
        resume_source_auxiliary_state_enabled=source_auxiliary_state_head_arch is not None,
        resume_source_auxiliary_state_head_arch=source_auxiliary_state_head_arch or (),
        device="cpu",
    )


def _train_app_config(
    tmp_path: Path,
    *,
    train_config: TrainConfig | None = None,
    policy_config: PolicyConfig | None = None,
    action_config: ActionConfig | None = None,
) -> TrainAppConfig:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    return TrainAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(action=action_config or ActionConfig()),
        policy=policy_config or PolicyConfig(),
        train=train_config or TrainConfig(algorithm="maskable_hybrid_action_ppo"),
    )


def test_weights_only_resume_relaxes_exact_match_when_aux_bank_presence_differs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _DummyModel(auxiliary_state_head_arch=(128,))
    train_config = _resume_train_config(tmp_path)
    model_path = tmp_path / "model.zip"

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


def test_non_managed_resume_allows_nonstructural_config_edits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _DummyModel(auxiliary_state_head_arch=None)
    model_path = tmp_path / "model.zip"
    train_config = _resume_train_config(tmp_path)
    current_config = _train_app_config(
        tmp_path,
        train_config=train_config,
        policy_config=PolicyConfig(activation="gelu"),
    )

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.preload.resolve_model_artifact_path",
        lambda run_dir, artifact: model_path,
    )

    maybe_resume_training_model(
        model=model,
        train_env=object(),
        train_config=train_config,
        current_run_config=current_config,
    )

    assert model.set_parameters_calls == [(str(model_path), True, "cpu")]


def test_resume_requires_explicit_auxiliary_state_metadata(
    tmp_path: Path,
) -> None:
    model = _DummyModel(auxiliary_state_head_arch=None)
    train_config = TrainConfig(
        algorithm="maskable_hybrid_action_ppo",
        resume_run_dir=tmp_path,
        resume_artifact="latest",
        resume_source_algorithm="maskable_hybrid_action_ppo",
    )

    with pytest.raises(RuntimeError, match="Resume requires source checkpoint metadata"):
        maybe_resume_training_model(
            model=model,
            train_env=object(),
            train_config=train_config,
        )


def test_weights_only_resume_keeps_exact_match_when_aux_bank_signature_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _DummyModel(auxiliary_state_head_arch=(128,))
    train_config = _resume_train_config(tmp_path, source_auxiliary_state_head_arch=(128,))
    model_path = tmp_path / "model.zip"

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


def test_weights_only_resume_imports_source_action_bias_marker_before_reconcile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _DummyModel(auxiliary_state_head_arch=None)
    model_path = tmp_path / "model.zip"
    train_config = _resume_train_config(tmp_path)
    policy_config = PolicyConfig(
        action_bias=PolicyActionBiasConfig(spin_idle_logit=0.5),
    )
    train_env = SimpleNamespace(get_attr=lambda _attr_name: [])
    calls: list[tuple[PolicyConfig, dict[str, float]]] = []
    source_offsets = {
        "gas_on_logit": 0.0,
        "air_brake_on_logit": 9.0,
        "spin_idle_logit": 0.0,
    }

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.preload.resolve_model_artifact_path",
        lambda run_dir, artifact: model_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.preload.load_action_bias_offsets_from_archive",
        lambda path: source_offsets,
    )

    def fake_apply_weights_only_action_bias_delta(
        model: _DummyModel,
        *,
        train_env: object,
        policy_config: PolicyConfig,
        source_offsets: dict[str, float],
    ) -> None:
        assert model.set_parameters_calls == [(str(model_path), True, "cpu")]
        assert getattr(model, MODEL_ACTION_BIAS_OFFSETS_ATTR) == source_offsets
        calls.append((policy_config, source_offsets))

    monkeypatch.setattr(
        "rl_fzerox.core.training.session.model.preload.apply_weights_only_action_bias_delta",
        fake_apply_weights_only_action_bias_delta,
    )

    maybe_resume_training_model(
        model=model,
        train_env=train_env,
        train_config=train_config,
        policy_config=policy_config,
    )

    assert calls == [(policy_config, source_offsets)]


def test_weights_only_resume_relaxes_exact_match_when_aux_bank_arch_differs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _DummyModel(auxiliary_state_head_arch=(64,))
    train_config = _resume_train_config(tmp_path, source_auxiliary_state_head_arch=(128,))
    model_path = tmp_path / "model.zip"

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
    train_config = _resume_train_config(
        tmp_path,
        resume_mode="full_model",
        source_auxiliary_state_head_arch=(64,),
    )

    with pytest.raises(RuntimeError, match="Full-model resume requires the same auxiliary-state"):
        maybe_resume_training_model(
            model=model,
            train_env=object(),
            train_config=train_config,
        )


def test_managed_resume_requires_source_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="resume_source_algorithm"):
        TrainConfig(
            algorithm="maskable_hybrid_action_ppo",
            resume_run_dir=tmp_path,
            resume_artifact="latest",
            resume_source_metadata_required=True,
        )


def test_managed_resume_uses_injected_source_metadata_without_yaml_loaders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _DummyModel(auxiliary_state_head_arch=None)
    model_path = tmp_path / "model.zip"
    train_config = TrainConfig(
        algorithm="maskable_hybrid_action_ppo",
        resume_run_dir=tmp_path,
        resume_artifact="latest",
        resume_mode="weights_only",
        resume_source_algorithm="maskable_hybrid_action_ppo",
        resume_source_auxiliary_state_enabled=False,
        resume_source_metadata_required=True,
        device="cpu",
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
