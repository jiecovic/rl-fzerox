# tests/apps/test_watch.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.apps.watch import main
from rl_fzerox.core.config.schema import EmulatorConfig, WatchAppConfig, WatchConfig


def test_watch_rejects_artifact_without_run_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()

    config = WatchAppConfig(
        seed=7,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        watch=WatchConfig(),
    )

    monkeypatch.setattr(
        "rl_fzerox.apps.watch.load_watch_app_config",
        lambda *args, **kwargs: config,
    )

    with pytest.raises(
        SystemExit,
        match="--artifact requires --run-dir or watch.policy_run_dir",
    ):
        main(["--config", str(tmp_path / "watch.yaml"), "--artifact", "best"])
