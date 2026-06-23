# tests/core/runtime_spec/test_inference.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.runtime_spec.inference import inference_train_app_config
from rl_fzerox.core.runtime_spec.schema import (
    EmulatorConfig,
    StateFeatureDropoutGroupConfig,
    TrainAppConfig,
)


def test_inference_runtime_config_removes_stochastic_episode_regularization(tmp_path: Path) -> None:
    config = _train_app_config(
        tmp_path,
        lean_probability=0.5,
        air_brake_probability=1.0,
        spin_probability=0.25,
        state_feature_dropout_groups=(
            StateFeatureDropoutGroupConfig(
                feature_names=("vehicle_state.airborne",),
                dropout_prob=0.4,
            ),
            StateFeatureDropoutGroupConfig(
                feature_names=("track_position.edge_ratio",),
                dropout_prob=1.0,
            ),
        ),
    )

    inference_config = inference_train_app_config(config)

    assert inference_config.env.action.lean_episode_mask_probability == 0.0
    assert inference_config.env.action.air_brake_episode_mask_probability == 1.0
    assert inference_config.env.action.spin_episode_mask_probability == 0.0
    assert tuple(
        group.model_dump(mode="python")
        for group in inference_config.train.state_feature_dropout_groups
    ) == (
        {
            "feature_names": ("track_position.edge_ratio",),
            "dropout_prob": 1.0,
        },
    )


def test_inference_runtime_config_does_not_mutate_source_config(tmp_path: Path) -> None:
    config = _train_app_config(
        tmp_path,
        lean_probability=0.5,
        air_brake_probability=0.25,
        spin_probability=0.75,
        state_feature_dropout_groups=(
            StateFeatureDropoutGroupConfig(
                feature_names=("vehicle_state.airborne",),
                dropout_prob=0.4,
            ),
        ),
    )

    inference_train_app_config(config)

    assert config.env.action.lean_episode_mask_probability == 0.5
    assert config.env.action.air_brake_episode_mask_probability == 0.25
    assert config.env.action.spin_episode_mask_probability == 0.75
    assert config.train.state_feature_dropout_groups[0].dropout_prob == 0.4


def _train_app_config(
    tmp_path: Path,
    /,
    *,
    lean_probability: float,
    air_brake_probability: float,
    spin_probability: float,
    state_feature_dropout_groups: tuple[StateFeatureDropoutGroupConfig, ...],
) -> TrainAppConfig:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    config = TrainAppConfig(
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        )
    )
    action_config = config.env.action.model_copy(
        update={
            "lean_episode_mask_probability": lean_probability,
            "air_brake_episode_mask_probability": air_brake_probability,
            "spin_episode_mask_probability": spin_probability,
        }
    )
    return config.model_copy(
        update={
            "env": config.env.model_copy(update={"action": action_config}),
            "train": config.train.model_copy(
                update={
                    "state_feature_dropout_groups": state_feature_dropout_groups,
                }
            ),
        }
    )
