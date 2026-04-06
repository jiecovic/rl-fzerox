# tests/test_config_loader.py
from pathlib import Path

import pytest

from rl_fzerox.core.config import load_watch_app_config


def test_load_watch_app_config_requires_existing_file() -> None:
    with pytest.raises(ValueError, match="Could not load watch config"):
        load_watch_app_config(Path("/does/not/exist.yaml"))


def test_load_watch_app_config_reads_yaml_file(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "watch.yaml"
    core_path.touch()
    rom_path.touch()
    config_path.write_text(
        "\n".join(
            [
                "seed: 7",
                "emulator:",
                f"  core_path: {core_path}",
                f"  rom_path: {rom_path}",
                "env:",
                "  action_repeat: 3",
                "watch:",
                "  episodes: 2",
                "  fps: 30",
            ]
        ),
        encoding="utf-8",
    )

    config = load_watch_app_config(config_path)

    assert config.emulator.core_path == core_path
    assert config.emulator.rom_path == rom_path
    assert config.seed == 7
    assert config.env.action_repeat == 3
    assert config.watch.episodes == 2
    assert config.watch.fps == 30


def test_repo_watch_template_exists() -> None:
    config_path = Path(__file__).resolve().parents[1] / "conf" / "watch.yaml"

    assert config_path == Path(__file__).resolve().parents[1] / "conf" / "watch.yaml"
    assert config_path.is_file()
