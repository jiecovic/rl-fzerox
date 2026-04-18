# tests/core/config/conftest.py
from __future__ import annotations

from pathlib import Path

import pytest

import rl_fzerox.core.config.loader as config_loader_module
import rl_fzerox.core.config.paths as config_paths_module


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
