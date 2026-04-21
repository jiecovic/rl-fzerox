# tests/core/envs/test_engine_info.py
from __future__ import annotations

import struct

import pytest

from rl_fzerox.core.envs.engine.info import VEHICLE_SETUP_RAM, backend_step_info


class _Backend:
    name = "fake"
    frame_index = 12
    display_aspect_ratio = 4 / 3
    native_fps = 60.0

    def __init__(self) -> None:
        self._ram = bytearray(0x0030_0000)

    def read_system_ram(self, offset: int, length: int) -> bytes:
        return bytes(self._ram[offset : offset + length])

    def write_i16(self, offset: int, value: int) -> None:
        self._ram[offset : offset + 2] = struct.pack("<h", value)

    def write_f32(self, offset: int, value: float) -> None:
        self._ram[offset : offset + 4] = struct.pack("<f", value)


def test_backend_step_info_includes_ram_engine_setting() -> None:
    backend = _Backend()
    backend.write_i16(VEHICLE_SETUP_RAM.player_characters, 0)
    backend.write_f32(VEHICLE_SETUP_RAM.player_engine, 0.5)
    backend.write_f32(VEHICLE_SETUP_RAM.character_last_engine, 0.5)
    backend.write_f32(
        VEHICLE_SETUP_RAM.player_racer_base + VEHICLE_SETUP_RAM.racer_engine_curve,
        0.371747,
    )

    info = backend_step_info(backend)

    assert info["engine_setting_ram"] == 0.5
    assert info["engine_setting_percent_ram"] == 50.0
    assert info["character_engine_setting_ram"] == 0.5
    assert info["racer_engine_curve_ram"] == pytest.approx(0.371747)
