# tests/test_probe.py
from pathlib import Path

import pytest

from rl_fzerox.core.emulator import probe_core


def test_probe_core_rejects_missing_library(tmp_path: Path) -> None:
    missing_core = tmp_path / "missing_libretro_core.so"

    with pytest.raises(FileNotFoundError, match="Libretro core not found"):
        probe_core(str(missing_core))
