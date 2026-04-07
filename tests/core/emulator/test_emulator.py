# tests/core/emulator/test_emulator.py
from pathlib import Path

import pytest

from rl_fzerox.core.emulator import Emulator


def test_emulator_rejects_missing_core(tmp_path: Path) -> None:
    missing_core = tmp_path / "missing_core.so"
    rom_path = tmp_path / "fzerox.n64"
    rom_path.touch()

    with pytest.raises(FileNotFoundError, match="Libretro core not found"):
        Emulator(core_path=missing_core, rom_path=rom_path)


def test_emulator_rejects_missing_rom(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    missing_rom = tmp_path / "missing_rom.n64"
    core_path.touch()

    with pytest.raises(FileNotFoundError, match="ROM not found"):
        Emulator(core_path=core_path, rom_path=missing_rom)


def test_emulator_rejects_unsupported_renderer(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    with pytest.raises(RuntimeError, match="gliden64"):
        Emulator(core_path=core_path, rom_path=rom_path, renderer="gliden64")
