# tests/test_telemetry.py
from __future__ import annotations

from dataclasses import dataclass
from struct import pack_into

import pytest

from rl_fzerox.core.game.telemetry import (
    GLOBALS,
    RACER,
    RACER_SIZE,
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
    player_base = GLOBALS.racers + RACER_SIZE * 0

    pack_into("<I", memory, GLOBALS.game_frame_count, 321)
    pack_into("<I", memory, GLOBALS.game_mode, 0x00000001)
    pack_into("<I", memory, GLOBALS.course_index, 0)
    pack_into("<I", memory, player_base + RACER.state_flags, (1 << 20) | (1 << 30))
    pack_into("<f", memory, player_base + RACER.speed, 123.5)
    pack_into("<f", memory, player_base + RACER.energy, 92.25)
    pack_into("<f", memory, player_base + RACER.max_energy, 100.0)
    pack_into("<f", memory, player_base + RACER.race_distance, 12_345.5)
    pack_into("<f", memory, player_base + RACER.laps_completed_distance, 10_000.0)
    pack_into("<f", memory, player_base + RACER.lap_distance, 2_345.5)
    pack_into("<f", memory, player_base + RACER.race_distance_position, 12_340.0)
    pack_into("<i", memory, player_base + RACER.race_time, 12_345)
    pack_into("<h", memory, player_base + RACER.lap, 2)
    pack_into("<i", memory, player_base + RACER.position, 3)
    memory[player_base + RACER.character] = 0
    memory[player_base + RACER.machine_index] = 7

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
    assert telemetry.player.race_distance == pytest.approx(12_345.5)
    assert telemetry.player.laps_completed_distance == pytest.approx(10_000.0)
    assert telemetry.player.lap_distance == pytest.approx(2_345.5)
    assert telemetry.player.race_distance_position == pytest.approx(12_340.0)
    assert telemetry.player.race_time_ms == 12_345
    assert telemetry.player.lap == 2
    assert telemetry.player.position == 3
    assert telemetry.player.character == 0
    assert telemetry.player.machine_index == 7
    assert telemetry.player.state_labels == ("can_boost", "active")
