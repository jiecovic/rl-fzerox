# src/rl_fzerox/core/game/telemetry.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from struct import unpack_from
from typing import Protocol

KSEG0_BASE = 0x80000000
SYSTEM_RAM_SIZE_MIN = 0x00300000
PLAYER_RACER_INDEX = 0
RACER_SIZE = 0x3A8
SPEED_TO_KPH = 21.6
# Mupen64Plus-Next exposes the live RAM buffer in little-endian host order.
MEMORY_ENDIANNESS = "little"


def _rdram_offset(vram_address: int) -> int:
    return vram_address - KSEG0_BASE


# Derived from inspectredc/fzerox US rev0 symbol_addrs.txt.
G_GAME_MODE = _rdram_offset(0x800DCE44)
G_GAME_FRAME_COUNT = _rdram_offset(0x800CCFE0)
G_COURSE_INDEX = _rdram_offset(0x800F8514)
G_RACERS = _rdram_offset(0x802C4920)

# Derived from inspectredc/fzerox include/unk_structs.h for struct Racer.
RACER_STATE_FLAGS = 0x004
RACER_SPEED = 0x098
RACER_BOOST_TIMER = 0x21C
RACER_ENERGY = 0x228
RACER_MAX_ENERGY = 0x22C
RACER_RACE_TIME = 0x2A0
RACER_LAP = 0x2A8
RACER_LAPS_COMPLETED = 0x2AA
RACER_POSITION = 0x2AC
RACER_CHARACTER = 0x2C8
RACER_MACHINE_INDEX = 0x2C9


class MemoryReadableEmulator(Protocol):
    """Minimal emulator contract needed to decode live RDRAM telemetry."""

    @property
    def system_ram_size(self) -> int: ...

    def read_system_ram(self, offset: int, length: int) -> bytes: ...


class GameMode(IntEnum):
    """Core F-Zero X game mode ids from inspectredc/fzerox include/fzx_game.h."""

    TITLE = 0x00
    GP_RACE = 0x01
    PRACTICE = 0x02
    VS_2P = 0x03
    VS_3P = 0x04
    VS_4P = 0x05
    RECORDS = 0x06
    MAIN_MENU = 0x07
    MACHINE_SELECT = 0x08
    MACHINE_SETTINGS = 0x09
    COURSE_SELECT = 0x0A
    SKIPPABLE_CREDITS = 0x0B
    UNSKIPPABLE_CREDITS = 0x0C
    COURSE_EDIT = 0x0D
    TIME_ATTACK = 0x0E
    GP_RACE_NEXT_COURSE = 0x0F
    CREATE_MACHINE = 0x10
    GP_END_CUTSCENE = 0x11
    GP_RACE_NEXT_MACHINE_SETTINGS = 0x12
    RECORDS_COURSE_SELECT = 0x13
    OPTIONS_MENU = 0x14
    DEATH_RACE = 0x15
    EAD_DEMO = 0x16


RACE_MODE_IDS = {
    GameMode.GP_RACE,
    GameMode.PRACTICE,
    GameMode.VS_2P,
    GameMode.VS_3P,
    GameMode.VS_4P,
    GameMode.TIME_ATTACK,
    GameMode.DEATH_RACE,
}

RACER_FLAG_LABELS: tuple[tuple[int, str], ...] = (
    (1 << 13, "collision_recoil"),
    (1 << 14, "spinning_out"),
    (1 << 18, "retired"),
    (1 << 19, "falling_off_track"),
    (1 << 20, "can_boost"),
    (1 << 23, "cpu_controlled"),
    (1 << 24, "dash_pad_boost"),
    (1 << 25, "finished"),
    (1 << 26, "airborne"),
    (1 << 27, "crashed"),
    (1 << 30, "active"),
)


@dataclass(frozen=True)
class PlayerTelemetry:
    """Live player-one racer values decoded from the emulated RDRAM buffer."""

    state_flags: int
    state_labels: tuple[str, ...]
    speed_raw: float
    speed_kph: float
    energy: float
    max_energy: float
    boost_timer: int
    race_time_ms: int
    lap: int
    laps_completed: int
    position: int
    character: int
    machine_index: int


@dataclass(frozen=True)
class FZeroXTelemetry:
    """High-level telemetry slice for the current F-Zero X emulator state."""

    system_ram_size: int
    game_frame_count: int
    game_mode_raw: int
    game_mode_name: str
    course_index: int
    in_race_mode: bool
    player: PlayerTelemetry

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the telemetry."""

        return asdict(self)


def read_telemetry(emulator: MemoryReadableEmulator) -> FZeroXTelemetry:
    """Read the current F-Zero X telemetry from emulated system RAM."""

    if emulator.system_ram_size < SYSTEM_RAM_SIZE_MIN:
        raise RuntimeError(
            "System RAM is smaller than expected for F-Zero X telemetry: "
            f"{emulator.system_ram_size} bytes"
        )

    player_base = G_RACERS + (PLAYER_RACER_INDEX * RACER_SIZE)
    header_start = G_GAME_FRAME_COUNT
    header_end = G_COURSE_INDEX + 4
    header = emulator.read_system_ram(header_start, header_end - header_start)
    player_window = emulator.read_system_ram(player_base, RACER_MACHINE_INDEX + 1)

    game_mode_raw = _u32(header, G_GAME_MODE - header_start)
    mode_id = game_mode_raw & 0x1F
    try:
        game_mode = GameMode(mode_id)
        game_mode_name = game_mode.name.lower()
    except ValueError:
        game_mode = None
        game_mode_name = f"unknown_{mode_id:#x}"

    player_state_flags = _u32(player_window, RACER_STATE_FLAGS)
    player_speed_raw = _f32(player_window, RACER_SPEED)
    player = PlayerTelemetry(
        state_flags=player_state_flags,
        state_labels=_decode_racer_flags(player_state_flags),
        speed_raw=player_speed_raw,
        speed_kph=player_speed_raw * SPEED_TO_KPH,
        energy=_f32(player_window, RACER_ENERGY),
        max_energy=_f32(player_window, RACER_MAX_ENERGY),
        boost_timer=_s32(player_window, RACER_BOOST_TIMER),
        race_time_ms=_s32(player_window, RACER_RACE_TIME),
        lap=_s16(player_window, RACER_LAP),
        laps_completed=_s16(player_window, RACER_LAPS_COMPLETED),
        position=_s32(player_window, RACER_POSITION),
        character=_u8(player_window, RACER_CHARACTER),
        machine_index=_u8(player_window, RACER_MACHINE_INDEX),
    )

    return FZeroXTelemetry(
        system_ram_size=emulator.system_ram_size,
        game_frame_count=_u32(header, G_GAME_FRAME_COUNT - header_start),
        game_mode_raw=game_mode_raw,
        game_mode_name=game_mode_name,
        course_index=_u32(header, G_COURSE_INDEX - header_start),
        in_race_mode=game_mode in RACE_MODE_IDS,
        player=player,
    )


def _decode_racer_flags(state_flags: int) -> tuple[str, ...]:
    return tuple(label for bit, label in RACER_FLAG_LABELS if state_flags & bit)


def _u8(memory: bytes, offset: int) -> int:
    return memory[offset]


def _s16(memory: bytes, offset: int) -> int:
    return int.from_bytes(
        memory[offset : offset + 2],
        byteorder=MEMORY_ENDIANNESS,
        signed=True,
    )


def _s32(memory: bytes, offset: int) -> int:
    return int.from_bytes(
        memory[offset : offset + 4],
        byteorder=MEMORY_ENDIANNESS,
        signed=True,
    )


def _u32(memory: bytes, offset: int) -> int:
    return int.from_bytes(
        memory[offset : offset + 4],
        byteorder=MEMORY_ENDIANNESS,
        signed=False,
    )


def _f32(memory: bytes, offset: int) -> float:
    return float(unpack_from("<f", memory, offset)[0])
