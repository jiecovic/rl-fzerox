# tests/core/config/test_config_loader_actions.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.config import load_train_app_config


def _write_yaml(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_train_app_config_reads_maskable_hybrid_action_ppo_fields(
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
            "    name: hybrid_steer_drive_boost_lean_primitive",
            "    continuous_drive_mode: pwm",
            "    continuous_drive_deadzone: 0.0",
            "train:",
            "  algorithm: maskable_hybrid_action_ppo",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.env.action.name == "hybrid_steer_drive_boost_lean_primitive"
    assert config.env.action.continuous_drive_mode == "pwm"
    assert config.env.action.continuous_drive_deadzone == 0.0
    assert config.train.algorithm == "maskable_hybrid_action_ppo"


def test_load_train_app_config_reads_maskable_hybrid_recurrent_ppo_fields(
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
            "    name: hybrid_steer_drive_boost_lean_primitive",
            "    continuous_drive_mode: pwm",
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

    assert config.env.action.name == "hybrid_steer_drive_boost_lean_primitive"
    assert config.env.action.continuous_drive_mode == "pwm"
    assert config.env.action.continuous_drive_deadzone == 0.0
    assert config.env.action.continuous_air_brake_mode == "disable_on_ground"
    assert config.env.action.boost_unmask_max_speed_kph == 700.0
    assert config.env.action.runtime().boost_decision_interval_frames == 1
    assert config.env.action.runtime().boost_request_lockout_frames == 5
    assert config.env.action.lean_unmask_min_speed_kph == 500.0
    assert config.train.algorithm == "maskable_hybrid_recurrent_ppo"
    assert config.policy.recurrent.enabled is True


def test_load_train_app_config_compiles_action_branches(tmp_path: Path) -> None:
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
            "        response_power: 0.8",
            "      gas:",
            "        type: discrete",
            "        mask: [idle, engaged]",
            "      boost:",
            "        type: discrete",
            "        mask: [idle]",
            "        unmask_max_speed_kph: null",
            "      lean:",
            "        type: discrete",
            "        mask: [idle, left, right]",
            "        mode: release_cooldown",
            "        unmask_min_speed_kph: null",
            "train:",
            "  algorithm: maskable_hybrid_recurrent_ppo",
            "  total_timesteps: 1000",
            "policy:",
            "  recurrent:",
            "    enabled: true",
        ],
    )

    config = load_train_app_config(config_path)
    action_config = config.env.action.runtime()

    assert config.env.action.name == "steer_drive_boost_lean"
    assert action_config.name == "hybrid_steer_gas_boost_lean"
    assert action_config.steer_response_power == 0.8
    assert action_config.boost_unmask_max_speed_kph is None
    assert action_config.boost_decision_interval_frames == 1
    assert action_config.boost_request_lockout_frames == 5
    assert action_config.lean_unmask_min_speed_kph is None
    assert action_config.lean_mode == "release_cooldown"
    assert action_config.mask_overrides == {
        "gas": (0, 1),
        "boost": (0,),
        "lean": (0, 1, 2),
    }
    assert config.env.action.branches is not None
    assert config.env.action.branches.boost is not None
    assert config.env.action.branches.boost.mask == ("idle",)


def test_load_train_app_config_compiles_continuous_gas_branch(tmp_path: Path) -> None:
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
            "        type: continuous",
            "        deadzone: 0.05",
            "        full_threshold: 0.85",
            "        min_level: 0.25",
            "      boost:",
            "        type: discrete",
            "        mask: [idle]",
            "      lean:",
            "        type: discrete",
            "        mask: [idle, left, right]",
            "train:",
            "  algorithm: maskable_hybrid_action_ppo",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)
    action_config = config.env.action.runtime()

    assert action_config.name == "hybrid_steer_drive_boost_lean"
    assert action_config.continuous_drive_mode == "pwm"
    assert action_config.continuous_drive_deadzone == 0.05
    assert action_config.continuous_drive_full_threshold == 0.85
    assert action_config.continuous_drive_min_level == 0.25
    assert action_config.mask_overrides == {
        "boost": (0,),
        "lean": (0, 1, 2),
    }


def test_load_train_app_config_compiles_airborne_pitch_branch(tmp_path: Path) -> None:
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
            "        type: continuous",
            "      air_brake:",
            "        type: discrete",
            "        mask: [idle, engaged]",
            "      boost:",
            "        type: discrete",
            "        mask: [idle]",
            "      lean:",
            "        type: discrete",
            "        mask: [idle, left, right]",
            "      pitch:",
            "        type: discrete",
            "        mask: unrestricted",
            "train:",
            "  algorithm: maskable_hybrid_action_ppo",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)
    action_config = config.env.action.runtime()

    assert action_config.name == "hybrid_steer_drive_air_brake_boost_lean_pitch"
    assert action_config.continuous_drive_mode == "pwm"
    assert action_config.continuous_air_brake_mode == "disable_on_ground"
    assert action_config.mask_overrides == {
        "air_brake": (0, 1),
        "boost": (0,),
        "lean": (0, 1, 2),
        "pitch": (0, 1, 2, 3, 4),
    }


def test_load_train_app_config_compiles_neutral_pitch_mask(tmp_path: Path) -> None:
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
            "        type: continuous",
            "      air_brake:",
            "        type: discrete",
            "      boost:",
            "        type: discrete",
            "      lean:",
            "        type: discrete",
            "      pitch:",
            "        type: discrete",
            "        mask: [neutral]",
            "train:",
            "  algorithm: maskable_hybrid_action_ppo",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)
    action_config = config.env.action.runtime()

    assert action_config.mask_overrides == {"pitch": (2,)}


def test_load_train_app_config_rejects_invalid_continuous_gas_zone(tmp_path: Path) -> None:
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
            "        type: continuous",
            "        deadzone: 0.9",
            "        full_threshold: 0.8",
            "      boost:",
            "        type: discrete",
            "      lean:",
            "        type: discrete",
            "train:",
            "  algorithm: maskable_hybrid_action_ppo",
            "  total_timesteps: 1000",
        ],
    )

    with pytest.raises(ValueError, match="deadzone must be lower than full_threshold"):
        load_train_app_config(config_path)


def test_load_train_app_config_prefers_action_branches_over_adapter_fields(
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
            "    name: steer_drive",
            "    mask:",
            "      boost: [idle]",
            "    branches:",
            "      steer:",
            "        type: continuous",
            "      gas:",
            "        type: discrete",
            "        mask: [idle, engaged]",
            "      boost:",
            "        type: discrete",
            "        mask: [idle, engaged]",
            "      lean:",
            "        type: discrete",
            "        mask: [idle]",
            "train:",
            "  algorithm: maskable_hybrid_recurrent_ppo",
            "  total_timesteps: 1000",
            "policy:",
            "  recurrent:",
            "    enabled: true",
        ],
    )

    config = load_train_app_config(config_path)
    action_config = config.env.action.runtime()

    assert action_config.name == "hybrid_steer_gas_boost_lean"
    assert action_config.mask_overrides == {
        "gas": (0, 1),
        "boost": (0, 1),
        "lean": (0,),
    }


def test_load_train_app_config_rejects_masked_continuous_action_branch(
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
            "    branches:",
            "      steer:",
            "        type: continuous",
            "        mask: [idle]",
            "      gas:",
            "        type: discrete",
            "      boost:",
            "        type: discrete",
            "      lean:",
            "        type: discrete",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    with pytest.raises(ValueError, match="continuous action branch 'steer' cannot define a mask"):
        load_train_app_config(config_path)


def test_load_train_app_config_migrates_legacy_air_brake_fields(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    disabled_config_path = tmp_path / "disabled.yaml"
    _write_yaml(
        disabled_config_path,
        [
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action:",
            "    continuous_air_brake_enabled: false",
        ],
    )
    ground_gate_config_path = tmp_path / "ground_gate.yaml"
    _write_yaml(
        ground_gate_config_path,
        [
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action:",
            "    continuous_air_brake_disable_on_ground: true",
        ],
    )

    disabled_config = load_train_app_config(disabled_config_path)
    ground_gate_config = load_train_app_config(ground_gate_config_path)

    assert disabled_config.env.action.continuous_air_brake_mode == "off"
    assert ground_gate_config.env.action.continuous_air_brake_mode == "disable_on_ground"


def test_load_train_app_config_migrates_legacy_boost_speed_gate(tmp_path: Path) -> None:
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
            "    boost_unmask_min_speed_kph: 800.0",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.env.action.boost_unmask_max_speed_kph == 800.0


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
