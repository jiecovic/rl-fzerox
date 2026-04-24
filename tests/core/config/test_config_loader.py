# tests/core/config/test_config_loader.py
from __future__ import annotations

from pathlib import Path

import pytest

import rl_fzerox.core.config.paths as config_paths_module
from rl_fzerox.core.config import load_train_app_config, load_watch_app_config


def _write_yaml(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
            "  stuck_truncation_enabled: false",
            "  wrong_way_truncation_enabled: false",
            "watch:",
            "  episodes: 2",
            "  control_fps: 30",
            "  render_fps: 30",
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
    assert config.env.stuck_truncation_enabled is False
    assert config.env.wrong_way_truncation_enabled is False
    assert config.watch.episodes == 2
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
            "    preset: crop_92x124",
            "watch:",
            "  control_fps: auto",
        ],
    )

    config = load_watch_app_config(config_path)

    assert config.env.observation.preset == "crop_92x124"


def test_load_watch_app_config_accepts_default_large_observation_preset(tmp_path: Path) -> None:
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
            "    preset: crop_116x164",
            "watch:",
            "  control_fps: auto",
        ],
    )

    config = load_watch_app_config(config_path)

    assert config.env.observation.preset == "crop_116x164"


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
            "    preset: crop_116x164",
        ],
    )

    config = load_watch_app_config(config_path)

    assert config.env.observation.mode == "image_state"


def test_load_watch_app_config_accepts_split_watch_fps(tmp_path: Path) -> None:
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
            "  control_fps: auto",
            "  render_fps: auto",
        ],
    )

    config = load_watch_app_config(config_path)

    assert config.watch.control_fps == "auto"
    assert config.watch.render_fps == "auto"


def test_load_watch_app_config_rejects_removed_legacy_watch_fps(tmp_path: Path) -> None:
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
            "watch:",
            "  fps: 30",
        ],
    )

    with pytest.raises(ValueError, match="fps"):
        load_watch_app_config(config_path)


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
            "  control_fps: 30",
            "  render_fps: 30",
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
            "  control_fps: 30",
            "  render_fps: 30",
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
            "  control_fps: 30",
            "  render_fps: 30",
        ],
    )

    config = load_watch_app_config(
        config_path,
        overrides=["seed=19", "env.action_repeat=5", "watch.episodes=3"],
    )

    assert config.seed == 19
    assert config.env.action_repeat == 5
    assert config.watch.episodes == 3


def test_repo_watch_template_exists() -> None:
    config_path = config_paths_module.project_root_dir() / "conf" / "watch.yaml"

    assert config_path.is_file()
