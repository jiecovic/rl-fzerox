# tests/core/config/test_config_loader.py
from __future__ import annotations

from pathlib import Path

import pytest

import rl_fzerox.core.config.loader as config_loader_module
import rl_fzerox.core.config.paths as config_paths_module
from rl_fzerox.core.config import load_train_app_config, load_watch_app_config


def _write_yaml(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture
def isolated_repo_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    project_root = tmp_path / "repo"
    config_root = project_root / "conf"

    _write_yaml(
        config_root / "watch.yaml",
        [
            "seed: 123",
            "emulator:",
            "  core_path: /absolute/path/to/mupen64plus_next_libretro.so",
            "  rom_path: /absolute/path/to/fzerox.n64",
            "  runtime_dir: null",
            "  baseline_state_path: null",
            "env:",
            "  action_repeat: 2",
            "watch:",
            "  episodes: 1",
            "  fps: null",
        ],
    )

    monkeypatch.setattr(config_loader_module, "config_root_dir", lambda: config_root)
    monkeypatch.setattr(config_paths_module, "config_root_dir", lambda: config_root)
    monkeypatch.setattr(config_paths_module, "project_root_dir", lambda: project_root)

    return project_root, config_root


def test_load_watch_app_config_requires_existing_file() -> None:
    with pytest.raises(ValueError, match="Could not load watch config"):
        load_watch_app_config(Path("/does/not/exist.yaml"))


def test_load_train_app_config_uses_train_error_label() -> None:
    with pytest.raises(ValueError, match="Could not load train config"):
        load_train_app_config(Path("/does/not/exist.yaml"))


def test_load_watch_app_config_reads_yaml_file(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    runtime_dir = tmp_path / "runtime"
    baseline_state_path = tmp_path / "baseline.state"
    config_path = tmp_path / "watch.yaml"
    core_path.touch()
    rom_path.touch()
    runtime_dir.mkdir()
    baseline_state_path.write_bytes(b"baseline")
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            f"  runtime_dir: {runtime_dir}",
            f"  baseline_state_path: {baseline_state_path}",
            "env:",
            "  action_repeat: 3",
            "  reset_to_race: true",
            "  camera_setting: close_behind",
            "  terminate_on_energy_depleted: false",
            "watch:",
            "  episodes: 2",
            "  fps: 30",
            "  deterministic_policy: false",
            "  device: cuda",
        ],
    )

    config = load_watch_app_config(config_path)

    assert config.emulator.core_path == core_path
    assert config.emulator.rom_path == rom_path
    assert config.emulator.runtime_dir == runtime_dir
    assert config.emulator.baseline_state_path == baseline_state_path
    assert config.seed == 7
    assert config.env.action_repeat == 3
    assert config.env.reset_to_race is True
    assert config.env.camera_setting == "close_behind"
    assert config.env.terminate_on_energy_depleted is False
    assert config.watch.episodes == 2
    assert config.watch.fps == 30
    assert config.watch.control_fps == 30
    assert config.watch.render_fps == 30
    assert config.watch.deterministic_policy is False
    assert config.watch.device == "cuda"


def test_load_watch_app_config_accepts_larger_observation_preset(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "watch.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  observation:",
            "    preset: native_crop_v2",
            "watch:",
            "  fps: auto",
        ],
    )

    config = load_watch_app_config(config_path)

    assert config.env.observation.preset == "native_crop_v2"


def test_load_watch_app_config_accepts_default_v3_observation_preset(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "watch.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  observation:",
            "    preset: native_crop_v3",
            "watch:",
            "  fps: auto",
        ],
    )

    config = load_watch_app_config(config_path)

    assert config.env.observation.preset == "native_crop_v3"


def test_load_watch_app_config_accepts_image_state_observation_mode(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "watch.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  observation:",
            "    mode: image_state",
            "    preset: native_crop_v3",
        ],
    )

    config = load_watch_app_config(config_path)

    assert config.env.observation.mode == "image_state"


def test_load_watch_app_config_accepts_auto_watch_fps(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "watch.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "watch:",
            "  fps: auto",
        ],
    )

    config = load_watch_app_config(config_path)

    assert config.watch.fps == "auto"
    assert config.watch.control_fps == "auto"
    assert config.watch.render_fps == "auto"


def test_load_watch_app_config_allows_missing_baseline_state_path(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "watch.yaml"
    core_path.touch()
    rom_path.touch()

    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "  baseline_state_path: ./future.state",
            "env:",
            "  action_repeat: 3",
            "watch:",
            "  episodes: 1",
            "  fps: 30",
        ],
    )

    config = load_watch_app_config(config_path)

    assert config.emulator.baseline_state_path == (tmp_path / "future.state").resolve()


def test_load_watch_app_config_resolves_relative_paths_against_config_file(
    tmp_path: Path,
) -> None:
    assets_dir = tmp_path / "assets"
    runtime_dir = tmp_path / "runtime"
    baseline_state_path = assets_dir / "baseline.state"
    assets_dir.mkdir()
    runtime_dir.mkdir()
    core_path = assets_dir / "mupen64plus_next_libretro.so"
    rom_path = assets_dir / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    baseline_state_path.write_bytes(b"baseline")

    config_dir = tmp_path / "configs" / "local"
    config_path = config_dir / "watch.yaml"
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            "  core_path: ../../assets/mupen64plus_next_libretro.so",
            "  rom_path: ../../assets/fzerox.n64",
            "  runtime_dir: ../../runtime",
            "  baseline_state_path: ../../assets/baseline.state",
            "env:",
            "  action_repeat: 3",
            "watch:",
            "  episodes: 2",
            "  fps: 30",
        ],
    )

    config = load_watch_app_config(config_path)

    assert config.emulator.core_path == core_path.resolve()
    assert config.emulator.rom_path == rom_path.resolve()
    assert config.emulator.runtime_dir == runtime_dir.resolve()
    assert config.emulator.baseline_state_path == baseline_state_path.resolve()


def test_load_watch_app_config_supports_repo_root_hydra_configs(
    tmp_path: Path,
    isolated_repo_layout: tuple[Path, Path],
) -> None:
    _, config_root = isolated_repo_layout
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    runtime_dir = tmp_path / "runtime"
    core_path.touch()
    rom_path.touch()
    runtime_dir.mkdir()

    config = load_watch_app_config(
        config_root / "watch.yaml",
        overrides=[
            f"emulator.core_path={core_path}",
            f"emulator.rom_path={rom_path}",
            f"emulator.runtime_dir={runtime_dir}",
            "emulator.baseline_state_path=null",
            "seed=11",
            "env.action_repeat=4",
        ],
    )

    assert config.emulator.core_path == core_path
    assert config.emulator.rom_path == rom_path
    assert config.emulator.runtime_dir == runtime_dir
    assert config.emulator.baseline_state_path is None
    assert config.seed == 11
    assert config.env.action_repeat == 4


def test_load_watch_app_config_resolves_repo_relative_paths_from_project_root(
    isolated_repo_layout: tuple[Path, Path],
) -> None:
    project_root, config_root = isolated_repo_layout
    artifacts_dir = project_root / "local" / "test-artifacts"
    runtime_dir = project_root / "local" / "runtime"
    artifacts_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    core_path = artifacts_dir / "core.so"
    rom_path = artifacts_dir / "rom.n64"
    core_path.touch()
    rom_path.touch()

    config_path = config_root / "local" / "watch.test.yaml"
    _write_yaml(
        config_path,
        [
            "defaults:",
            "  - /watch",
            "  - _self_",
            "emulator:",
            "  core_path: local/test-artifacts/core.so",
            "  rom_path: local/test-artifacts/rom.n64",
            "  runtime_dir: local/runtime",
            "  baseline_state_path: null",
        ],
    )

    config = load_watch_app_config(config_path)

    assert config.emulator.core_path == core_path.resolve()
    assert config.emulator.rom_path == rom_path.resolve()
    assert config.emulator.runtime_dir == runtime_dir.resolve()


def test_load_watch_app_config_applies_hydra_overrides(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    runtime_dir = tmp_path / "runtime"
    config_path = tmp_path / "watch.yaml"
    core_path.touch()
    rom_path.touch()
    runtime_dir.mkdir()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            f"  runtime_dir: {runtime_dir}",
            "env:",
            "  action_repeat: 2",
            "watch:",
            "  episodes: 1",
            "  fps: 30",
        ],
    )

    config = load_watch_app_config(
        config_path,
        overrides=["seed=19", "env.action_repeat=5", "watch.episodes=3"],
    )

    assert config.seed == 19
    assert config.env.action_repeat == 5
    assert config.watch.episodes == 3


def test_load_train_app_config_reads_policy_activation(tmp_path: Path) -> None:
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
            "  action_repeat: 2",
            "policy:",
            "  activation: relu",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.policy.activation == "relu"


def test_load_train_app_config_reads_auto_extractor_features_dim(tmp_path: Path) -> None:
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
            "policy:",
            "  extractor:",
            "    features_dim: auto",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.policy.extractor.features_dim == "auto"


def test_load_train_app_config_resolves_init_run_dir(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0042"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "train:",
            "  total_timesteps: 1000",
            f"  init_run_dir: {run_dir}",
            "  init_artifact: latest",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.train.init_run_dir == run_dir.resolve()
    assert config.train.init_artifact == "latest"


def test_load_train_app_config_reads_state_extractor_features_dim(tmp_path: Path) -> None:
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
            "policy:",
            "  extractor:",
            "    state_features_dim: 32",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.policy.extractor.state_features_dim == 32


def test_load_train_app_config_reads_fusion_extractor_features_dim(tmp_path: Path) -> None:
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
            "policy:",
            "  extractor:",
            "    fusion_features_dim: 512",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.policy.extractor.fusion_features_dim == 512


def test_load_train_app_config_reads_maskable_curriculum_fields(tmp_path: Path) -> None:
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
            "    mask:",
            "      shoulder: [0]",
            "curriculum:",
            "  enabled: true",
            "  smoothing_episodes: 4",
            "  min_stage_episodes: 2",
            "  stages:",
            "    - name: basic_drive",
            "      until:",
            "        race_laps_completed_mean_gte: 3.0",
            "      action_mask:",
            "        shoulder: [0]",
            "    - name: drift_enabled",
            "      action_mask:",
            "        shoulder: [0, 1, 2]",
            "train:",
            "  algorithm: maskable_ppo",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.train.algorithm == "maskable_ppo"
    assert config.env.action.mask is not None
    assert config.env.action.mask.shoulder == (0,)
    assert config.curriculum.enabled is True
    assert config.curriculum.smoothing_episodes == 4
    assert config.curriculum.min_stage_episodes == 2
    assert len(config.curriculum.stages) == 2
    assert config.curriculum.stages[0].until is not None
    assert config.curriculum.stages[0].until.race_laps_completed_mean_gte == 3.0


def test_load_train_app_config_reads_recurrent_policy_fields(tmp_path: Path) -> None:
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
            "policy:",
            "  recurrent:",
            "    enabled: true",
            "    hidden_size: 512",
            "    n_lstm_layers: 1",
            "    shared_lstm: false",
            "    enable_critic_lstm: true",
            "train:",
            "  algorithm: maskable_recurrent_ppo",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.train.algorithm == "maskable_recurrent_ppo"
    assert config.policy.recurrent.enabled is True
    assert config.policy.recurrent.hidden_size == 512
    assert config.policy.recurrent.n_lstm_layers == 1
    assert config.policy.recurrent.shared_lstm is False
    assert config.policy.recurrent.enable_critic_lstm is True


def test_load_train_app_config_reads_sac_fields(tmp_path: Path) -> None:
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
            "    name: continuous_steer_drive_shoulder",
            "    steer_response_power: 0.7",
            "    continuous_drive_mode: always_accelerate",
            "    continuous_drive_deadzone: 0.15",
            "    continuous_shoulder_deadzone: 0.25",
            "train:",
            "  algorithm: sac",
            "  total_timesteps: 1000",
            "  buffer_size: 30000",
            "  learning_starts: 5000",
            "  train_freq: 1",
            "  gradient_steps: 1",
            "  ent_coef: auto",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.env.action.name == "continuous_steer_drive_shoulder"
    assert config.env.action.steer_response_power == 0.7
    assert config.env.action.continuous_drive_mode == "always_accelerate"
    assert config.env.action.continuous_drive_deadzone == 0.15
    assert config.env.action.continuous_shoulder_deadzone == 0.25
    assert config.train.algorithm == "sac"
    assert config.train.buffer_size == 30_000
    assert config.train.learning_starts == 5_000
    assert config.train.ent_coef == "auto"


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
            "    name: hybrid_steer_drive_boost_shoulder_primitive",
            "    continuous_drive_mode: pwm",
            "    continuous_drive_deadzone: 0.0",
            "train:",
            "  algorithm: maskable_hybrid_action_ppo",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.env.action.name == "hybrid_steer_drive_boost_shoulder_primitive"
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
            "    name: hybrid_steer_drive_boost_shoulder_primitive",
            "    continuous_drive_mode: pwm",
            "    continuous_drive_deadzone: 0.0",
            "    continuous_air_brake_mode: disable_on_ground",
            "    boost_unmask_max_speed_kph: 700.0",
            "    boost_decision_interval_frames: 30",
            "    boost_request_lockout_frames: 45",
            "    shoulder_unmask_min_speed_kph: 500.0",
            "train:",
            "  algorithm: maskable_hybrid_recurrent_ppo",
            "  total_timesteps: 1000",
            "policy:",
            "  recurrent:",
            "    enabled: true",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.env.action.name == "hybrid_steer_drive_boost_shoulder_primitive"
    assert config.env.action.continuous_drive_mode == "pwm"
    assert config.env.action.continuous_drive_deadzone == 0.0
    assert config.env.action.continuous_air_brake_mode == "disable_on_ground"
    assert config.env.action.boost_unmask_max_speed_kph == 700.0
    assert config.env.action.boost_decision_interval_frames == 30
    assert config.env.action.boost_request_lockout_frames == 45
    assert config.env.action.shoulder_unmask_min_speed_kph == 500.0
    assert config.train.algorithm == "maskable_hybrid_recurrent_ppo"
    assert config.policy.recurrent.enabled is True


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


def test_load_train_app_config_migrates_legacy_shoulder_fields(tmp_path: Path) -> None:
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
            "    continuous_drift_deadzone: 0.25",
            "    drift_unmask_min_speed_kph: 500.0",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.env.action.continuous_shoulder_deadzone == 0.25
    assert config.env.action.shoulder_unmask_min_speed_kph == 500.0


def test_repo_watch_template_exists() -> None:
    config_path = config_paths_module.project_root_dir() / "conf" / "watch.yaml"

    assert config_path.is_file()
