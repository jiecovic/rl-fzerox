# tests/test_telemetry.py
from __future__ import annotations

from dataclasses import dataclass
from struct import pack_into

import pytest

from rl_fzerox.core.game.telemetry import (
    G_COURSE_INDEX,
    G_GAME_FRAME_COUNT,
    G_GAME_MODE,
    G_RACERS,
    RACER_CHARACTER,
    RACER_ENERGY,
    RACER_LAP,
    RACER_MACHINE_INDEX,
    RACER_MAX_ENERGY,
    RACER_POSITION,
    RACER_RACE_TIME,
    RACER_SIZE,
    RACER_SPEED,
    RACER_STATE_FLAGS,
    read_telemetry,
)


@dataclass
class FakeMemoryEmulator:
    system_ram: bytes

    @property
    def system_ram_size(self) -> int:
        return len(self.system_ram)

    def read_system_ram(self, offset: int, length: int) -> bytes:
        return self.system_ram[offset : offset + length]


def test_read_telemetry_decodes_player_one_race_values() -> None:
    memory = bytearray(0x00300000)
    player_base = G_RACERS + RACER_SIZE * 0

    pack_into("<I", memory, G_GAME_FRAME_COUNT, 321)
    pack_into("<I", memory, G_GAME_MODE, 0x00000001)
    pack_into("<I", memory, G_COURSE_INDEX, 0)
    pack_into("<I", memory, player_base + RACER_STATE_FLAGS, (1 << 20) | (1 << 30))
    pack_into("<f", memory, player_base + RACER_SPEED, 123.5)
    pack_into("<f", memory, player_base + RACER_ENERGY, 92.25)
    pack_into("<f", memory, player_base + RACER_MAX_ENERGY, 100.0)
    pack_into("<i", memory, player_base + RACER_RACE_TIME, 12_345)
    pack_into("<h", memory, player_base + RACER_LAP, 2)
    pack_into("<i", memory, player_base + RACER_POSITION, 3)
    memory[player_base + RACER_CHARACTER] = 0
    memory[player_base + RACER_MACHINE_INDEX] = 7

    telemetry = read_telemetry(FakeMemoryEmulator(bytes(memory)))

    assert telemetry.system_ram_size == len(memory)
    assert telemetry.game_frame_count == 321
    assert telemetry.game_mode_raw == 1
    assert telemetry.game_mode_name == "gp_race"
    assert telemetry.in_race_mode is True
    assert telemetry.course_index == 0
    assert telemetry.player.speed_raw == pytest.approx(123.5)
    assert telemetry.player.speed_kph == pytest.approx(123.5 * 21.6)
    assert telemetry.player.energy == pytest.approx(92.25)
    assert telemetry.player.max_energy == pytest.approx(100.0)
    assert telemetry.player.race_time_ms == 12_345
    assert telemetry.player.lap == 2
    assert telemetry.player.position == 3
    assert telemetry.player.character == 0
    assert telemetry.player.machine_index == 7
    assert telemetry.player.state_labels == ("can_boost", "active")
