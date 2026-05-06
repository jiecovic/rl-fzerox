# tests/core/config/test_config_loader_actions.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.config import load_train_app_config


def _write_yaml(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_train_app_config_reads_configured_hybrid_fields(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action:",
            "    name: configured_hybrid",
            "    layout_continuous_axes: [steer, drive]",
            "    layout_discrete_axes: [boost, lean]",
            "    continuous_drive_deadzone: 0.0",
            "train:",
            "  algorithm: maskable_hybrid_action_ppo",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.env.action.name == "configured_hybrid"
    assert config.env.action.layout_continuous_axes == ("steer", "drive")
    assert config.env.action.layout_discrete_axes == ("boost", "lean")
    assert config.env.action.continuous_drive_deadzone == 0.0
    assert config.train.algorithm == "maskable_hybrid_action_ppo"


def test_load_train_app_config_reads_recurrent_hybrid_fields(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action:",
            "    name: configured_hybrid",
            "    layout_continuous_axes: [steer, drive, air_brake]",
            "    layout_discrete_axes: [boost, lean, pitch]",
            "    continuous_drive_deadzone: 0.0",
            "    continuous_air_brake_mode: disable_on_ground",
            "    boost_unmask_max_speed_kph: 700.0",
            "    lean_unmask_min_speed_kph: 500.0",
            "train:",
            "  algorithm: maskable_hybrid_recurrent_ppo",
            "  total_timesteps: 1000",
            "policy:",
            "  recurrent:",
            "    enabled: true",
        ],
    )

    config = load_train_app_config(config_path)
    runtime = config.env.action.runtime()

    assert config.env.action.name == "configured_hybrid"
    assert config.env.action.layout_continuous_axes == ("steer", "drive", "air_brake")
    assert config.env.action.layout_discrete_axes == ("boost", "lean", "pitch")
    assert config.env.action.continuous_air_brake_mode == "disable_on_ground"
    assert config.env.action.boost_unmask_max_speed_kph == 700.0
    assert config.env.action.lean_unmask_min_speed_kph == 500.0
    assert runtime.boost_decision_interval_frames == 1
    assert runtime.boost_request_lockout_frames == 5
    assert config.train.algorithm == "maskable_hybrid_recurrent_ppo"
    assert config.policy.recurrent.enabled is True


def test_load_train_app_config_reads_hybrid_action_sac_fields(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action:",
            "    name: configured_hybrid",
            "    layout_continuous_axes: [steer, drive]",
            "    layout_discrete_axes: [boost, lean]",
            "    continuous_drive_deadzone: 0.05",
            "train:",
            "  algorithm: hybrid_action_sac",
            "  ent_coef: auto",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.env.action.name == "configured_hybrid"
    assert config.env.action.continuous_drive_deadzone == 0.05
    assert config.train.algorithm == "hybrid_action_sac"


@pytest.mark.parametrize(
    ("legacy_field", "legacy_value"),
    (
        ("continuous_air_brake_enabled", "true"),
        ("continuous_air_brake_disable_on_ground", "true"),
        ("boost_unmask_min_speed_kph", "700.0"),
    ),
)
def test_load_train_app_config_rejects_removed_legacy_action_fields(
    tmp_path: Path,
    legacy_field: str,
    legacy_value: str,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action:",
            "    name: configured_hybrid",
            "    layout_continuous_axes: [steer, drive]",
            "    layout_discrete_axes: [boost, lean]",
            f"    {legacy_field}: {legacy_value}",
            "train:",
            "  algorithm: hybrid_action_sac",
            "  ent_coef: auto",
            "  total_timesteps: 1000",
        ],
    )

    with pytest.raises(ValueError, match=legacy_field):
        load_train_app_config(config_path)


def test_load_train_app_config_rejects_legacy_action_branches_field(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action:",
            "    branches:",
            "      steer:",
            "        type: continuous",
            "      gas:",
            "        type: discrete",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    with pytest.raises(ValueError, match="env.action.branches"):
        load_train_app_config(config_path)


def test_load_train_app_config_reads_mask_overrides(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action:",
            "    name: configured_hybrid",
            "    layout_continuous_axes: [steer]",
            "    layout_discrete_axes: [gas, air_brake, boost, lean, pitch]",
            "    mask:",
            "      gas: [idle, engaged]",
            "      air_brake: [idle]",
            "      boost: [idle]",
            "      lean: [idle, left, right]",
            "      pitch: [down_full, down, neutral]",
            "train:",
            "  algorithm: maskable_hybrid_action_ppo",
            "  total_timesteps: 1000",
        ],
    )

    runtime = load_train_app_config(config_path).env.action.runtime()

    assert runtime.mask_overrides == {
        "gas": (0, 1),
        "air_brake": (0,),
        "boost": (0,),
        "lean": (0, 1, 2),
        "pitch": (0, 1, 2),
    }


def test_load_train_app_config_rejects_invalid_continuous_drive_zone(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action:",
            "    name: configured_hybrid",
            "    layout_continuous_axes: [steer, drive]",
            "    layout_discrete_axes: [boost, lean]",
            "    continuous_drive_deadzone: 0.9",
            "    continuous_drive_full_threshold: 0.8",
            "train:",
            "  algorithm: maskable_hybrid_action_ppo",
            "  total_timesteps: 1000",
        ],
    )

    with pytest.raises(ValueError, match="continuous_drive_deadzone"):
        load_train_app_config(config_path)


def test_load_train_app_config_accepts_lean_fields(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action:",
            "    continuous_lean_deadzone: 0.25",
            "    lean_unmask_min_speed_kph: 500.0",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.env.action.continuous_lean_deadzone == 0.25
    assert config.env.action.lean_unmask_min_speed_kph == 500.0
