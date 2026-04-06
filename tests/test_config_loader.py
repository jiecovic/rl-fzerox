# tests/test_config_loader.py
from pathlib import Path

import pytest

from rl_fzerox.core.config import default_config_dir, load_watch_app_config


def test_load_watch_app_config_requires_emulator_paths() -> None:
    with pytest.raises(ValueError, match="Could not load config 'watch'"):
        load_watch_app_config()


def test_load_watch_app_config_accepts_explicit_override_list(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    config = load_watch_app_config(
        config_dir=default_config_dir(),
        overrides=[
            f"+emulator.core_path={core_path}",
            f"+emulator.rom_path={rom_path}",
            "watch.seed=7",
            "env.action_repeat=3",
        ],
    )

    assert config.emulator.core_path == core_path
    assert config.emulator.rom_path == rom_path
    assert config.watch.seed == 7
    assert config.env.action_repeat == 3


def test_default_config_dir_points_at_repo_conf_directory() -> None:
    config_dir = default_config_dir()

    assert config_dir == Path(__file__).resolve().parents[1] / "conf"
    assert config_dir.is_dir()
