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
            "watch:",
            "  episodes: 2",
            "  fps: 30",
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
    assert config.watch.episodes == 2
    assert config.watch.fps == 30


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


def test_repo_watch_template_exists() -> None:
    config_path = config_paths_module.project_root_dir() / "conf" / "watch.yaml"

    assert config_path.is_file()
