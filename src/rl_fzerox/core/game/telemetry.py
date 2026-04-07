# src/rl_fzerox/core/game/telemetry.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import IntEnum
from struct import unpack_from
from typing import Protocol, TypeAlias, TypeGuard

from rl_fzerox.core.game.flags import decode_racer_flags

KSEG0_BASE = 0x80000000
SYSTEM_RAM_SIZE_MIN = 0x00300000
PLAYER_RACER_INDEX = 0
RACER_SIZE = 0x3A8
SPEED_TO_KPH = 21.6
# Mupen64Plus-Next exposes the live RAM buffer in little-endian host order.
MEMORY_ENDIANNESS = "little"
# Keep game-specific decoding here while offsets and semantics are still being
# validated. Once the field set is stable, move this decoding into the Rust
# host so raw memory access and structured telemetry live at the same boundary.


def _rdram_offset(vram_address: int) -> int:
    return vram_address - KSEG0_BASE


@dataclass(frozen=True)
class GlobalOffsets:
    """Global RDRAM addresses used by the current telemetry decoder."""

    game_mode: int
    game_frame_count: int
    course_index: int
    racers: int


@dataclass(frozen=True)
class RacerOffsets:
    """Byte offsets within `struct Racer` for the fields we decode."""

    state_flags: int
    speed: int
    boost_timer: int
    energy: int
    max_energy: int
    race_distance: int
    laps_completed_distance: int
    lap_distance: int
    race_distance_position: int
    race_time: int
    lap: int
    laps_completed: int
    position: int
    character: int
    machine_index: int


GLOBALS = GlobalOffsets(
    # Derived from inspectredc/fzerox US rev0 symbol_addrs.txt.
    game_mode=_rdram_offset(0x800DCE44),
    game_frame_count=_rdram_offset(0x800CCFE0),
    course_index=_rdram_offset(0x800F8514),
    racers=_rdram_offset(0x802C4920),
)

RACER = RacerOffsets(
    # Derived from inspectredc/fzerox include/unk_structs.h for struct Racer.
    state_flags=0x004,
    speed=0x098,
    boost_timer=0x21C,
    energy=0x228,
    max_energy=0x22C,
    race_distance=0x23C,
    laps_completed_distance=0x240,
    lap_distance=0x244,
    race_distance_position=0x248,
    race_time=0x2A0,
    lap=0x2A8,
    laps_completed=0x2AA,
    position=0x2AC,
    character=0x2C8,
    machine_index=0x2C9,
)


class MemoryReadableEmulator(Protocol):
    """Minimal emulator contract needed to decode live RDRAM telemetry."""

    @property
    def system_ram_size(self) -> int: ...

    def read_system_ram(self, offset: int, length: int) -> bytes: ...


class StructuredTelemetryReadableEmulator(Protocol):
    """Optional emulator contract for native structured telemetry reads."""

    def telemetry_data(self) -> dict[str, object]: ...


TelemetryReadableEmulator: TypeAlias = (
    MemoryReadableEmulator | StructuredTelemetryReadableEmulator
)


def _is_structured_telemetry_readable(
    emulator: object,
) -> TypeGuard[StructuredTelemetryReadableEmulator]:
    return callable(getattr(emulator, "telemetry_data", None))


def _is_memory_readable(
    emulator: object,
) -> TypeGuard[MemoryReadableEmulator]:
    return (
        hasattr(emulator, "system_ram_size")
        and hasattr(emulator, "read_system_ram")
    )


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
    race_distance: float
    laps_completed_distance: float
    lap_distance: float
    race_distance_position: float
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


def read_telemetry(emulator: TelemetryReadableEmulator) -> FZeroXTelemetry:
    """Read the current F-Zero X telemetry from emulated system RAM."""

    if _is_structured_telemetry_readable(emulator):
        native_data = emulator.telemetry_data()
        if not isinstance(native_data, Mapping):
            raise RuntimeError("Native telemetry payload must resolve to a mapping")
        return _telemetry_from_mapping(native_data)

    if not _is_memory_readable(emulator):
        raise RuntimeError(
            "Telemetry requires either a native telemetry_data() path or raw system RAM access"
        )

    if emulator.system_ram_size < SYSTEM_RAM_SIZE_MIN:
        raise RuntimeError(
            "System RAM is smaller than expected for F-Zero X telemetry: "
            f"{emulator.system_ram_size} bytes"
        )

    player_base = GLOBALS.racers + (PLAYER_RACER_INDEX * RACER_SIZE)
    header_start = GLOBALS.game_frame_count
    header_end = GLOBALS.course_index + 4
    header = emulator.read_system_ram(header_start, header_end - header_start)
    player_window = emulator.read_system_ram(player_base, RACER.machine_index + 1)

    game_mode_raw = _u32(header, GLOBALS.game_mode - header_start)
    mode_id = game_mode_raw & 0x1F
    try:
        game_mode = GameMode(mode_id)
        game_mode_name = game_mode.name.lower()
    except ValueError:
        game_mode = None
        game_mode_name = f"unknown_{mode_id:#x}"

    player_state_flags = _u32(player_window, RACER.state_flags)
    player_speed_raw = _f32(player_window, RACER.speed)
    player = PlayerTelemetry(
        state_flags=player_state_flags,
        state_labels=decode_racer_flags(player_state_flags),
        speed_raw=player_speed_raw,
        speed_kph=player_speed_raw * SPEED_TO_KPH,
        energy=_f32(player_window, RACER.energy),
        max_energy=_f32(player_window, RACER.max_energy),
        boost_timer=_s32(player_window, RACER.boost_timer),
        race_distance=_f32(player_window, RACER.race_distance),
        laps_completed_distance=_f32(player_window, RACER.laps_completed_distance),
        lap_distance=_f32(player_window, RACER.lap_distance),
        race_distance_position=_f32(player_window, RACER.race_distance_position),
        race_time_ms=_s32(player_window, RACER.race_time),
        lap=_s16(player_window, RACER.lap),
        laps_completed=_s16(player_window, RACER.laps_completed),
        position=_s32(player_window, RACER.position),
        character=_u8(player_window, RACER.character),
        machine_index=_u8(player_window, RACER.machine_index),
    )

    return FZeroXTelemetry(
        system_ram_size=emulator.system_ram_size,
        game_frame_count=_u32(header, GLOBALS.game_frame_count - header_start),
        game_mode_raw=game_mode_raw,
        game_mode_name=game_mode_name,
        course_index=_u32(header, GLOBALS.course_index - header_start),
        in_race_mode=game_mode in RACE_MODE_IDS,
        player=player,
    )


def _telemetry_from_mapping(data: Mapping[str, object]) -> FZeroXTelemetry:
    player_data = data.get("player")
    if not isinstance(player_data, Mapping):
        raise RuntimeError("Native telemetry payload is missing a player mapping")

    return FZeroXTelemetry(
        system_ram_size=_int_value(data, "system_ram_size"),
        game_frame_count=_int_value(data, "game_frame_count"),
        game_mode_raw=_int_value(data, "game_mode_raw"),
        game_mode_name=_str_value(data, "game_mode_name"),
        course_index=_int_value(data, "course_index"),
        in_race_mode=_bool_value(data, "in_race_mode"),
        player=_player_from_mapping(player_data),
    )


def _player_from_mapping(data: Mapping[str, object]) -> PlayerTelemetry:
    return PlayerTelemetry(
        state_flags=_int_value(data, "state_flags"),
        state_labels=_str_tuple(data, "state_labels"),
        speed_raw=_float_value(data, "speed_raw"),
        speed_kph=_float_value(data, "speed_kph"),
        energy=_float_value(data, "energy"),
        max_energy=_float_value(data, "max_energy"),
        boost_timer=_int_value(data, "boost_timer"),
        race_distance=_float_value(data, "race_distance"),
        laps_completed_distance=_float_value(data, "laps_completed_distance"),
        lap_distance=_float_value(data, "lap_distance"),
        race_distance_position=_float_value(data, "race_distance_position"),
        race_time_ms=_int_value(data, "race_time_ms"),
        lap=_int_value(data, "lap"),
        laps_completed=_int_value(data, "laps_completed"),
        position=_int_value(data, "position"),
        character=_int_value(data, "character"),
        machine_index=_int_value(data, "machine_index"),
    )


def _int_value(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"Native telemetry field {key!r} must be an int")
    return value


def _float_value(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"Native telemetry field {key!r} must be numeric")
    return float(value)


def _bool_value(data: Mapping[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"Native telemetry field {key!r} must be a bool")
    return value


def _str_value(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"Native telemetry field {key!r} must be a str")
    return value


def _str_tuple(data: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list | tuple):
        raise RuntimeError(f"Native telemetry field {key!r} must be a list or tuple")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise RuntimeError(f"Native telemetry field {key!r} must contain only strings")
        items.append(item)
    return tuple(items)


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
