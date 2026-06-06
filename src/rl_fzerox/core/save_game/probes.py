# src/rl_fzerox/core/save_game/probes.py
"""Typed memory probe reports for save-game reverse engineering."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from rl_fzerox.core.save_game.sram import SaveRamSummary, summarize_save_ram

MemoryProbeRegion: TypeAlias = Literal["save_ram", "system_ram"]
MemoryProbeFormat: TypeAlias = Literal[
    "bytes",
    "bitset_u8",
    "u8",
    "u16_be",
    "u16_le",
    "u32_be",
    "u32_le",
]
MemoryProbeValue: TypeAlias = int | str | tuple[int, ...]

MEMORY_PROBE_REGIONS: tuple[MemoryProbeRegion, ...] = ("save_ram", "system_ram")
MEMORY_PROBE_FORMATS: tuple[MemoryProbeFormat, ...] = (
    "bytes",
    "bitset_u8",
    "u8",
    "u16_be",
    "u16_le",
    "u32_be",
    "u32_le",
)


@dataclass(frozen=True, slots=True)
class MemoryProbeDefinition:
    """One named memory slice to decode from save RAM or live system RAM."""

    key: str
    region: MemoryProbeRegion
    offset: int
    length: int
    value_format: MemoryProbeFormat
    label: str

    @property
    def end_offset(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True, slots=True)
class MemoryProbeReading:
    """Decoded result for one memory probe definition."""

    key: str
    label: str
    region: MemoryProbeRegion
    offset: int
    length: int
    value_format: MemoryProbeFormat
    raw_hex: str
    value: MemoryProbeValue


@dataclass(frozen=True, slots=True)
class MemoryProbeReport:
    """Complete probe report for one save/live memory snapshot."""

    save_ram: SaveRamSummary | None
    readings: tuple[MemoryProbeReading, ...]


def parse_memory_probe_definition(spec: str) -> MemoryProbeDefinition:
    """Parse `key=region:offset:length:format[:label]` probe specifications."""

    key, separator, payload = spec.partition("=")
    if not separator or not key:
        raise ValueError("probe must use key=region:offset:length:format[:label]")

    parts = payload.split(":", 4)
    if len(parts) < 4:
        raise ValueError("probe payload must contain region, offset, length, and format")

    region = _parse_region(parts[0])
    offset = _parse_non_negative_int(parts[1], field="offset")
    length = _parse_positive_int(parts[2], field="length")
    value_format = _parse_format(parts[3])
    label = parts[4] if len(parts) == 5 and parts[4] else key
    definition = MemoryProbeDefinition(
        key=key,
        region=region,
        offset=offset,
        length=length,
        value_format=value_format,
        label=label,
    )
    _validate_probe_width(definition)
    return definition


def read_memory_probes(
    buffers: Mapping[MemoryProbeRegion, bytes],
    definitions: Iterable[MemoryProbeDefinition],
) -> tuple[MemoryProbeReading, ...]:
    """Read named probe definitions from already-captured memory buffers."""

    return tuple(_read_probe(buffers, definition) for definition in definitions)


def collect_memory_probe_report(
    definitions: Iterable[MemoryProbeDefinition],
    *,
    read_save_ram: Callable[[], bytes] | None,
    read_system_ram: Callable[[int, int], bytes] | None = None,
) -> MemoryProbeReport:
    """Collect probe readings without forcing callers to read full system RAM."""

    definitions_tuple = tuple(definitions)
    save_ram = read_save_ram() if read_save_ram is not None else None
    readings = [
        _read_live_probe(
            definition,
            save_ram=save_ram,
            read_system_ram=read_system_ram,
        )
        for definition in definitions_tuple
    ]
    return MemoryProbeReport(
        save_ram=None if save_ram is None else summarize_save_ram(save_ram),
        readings=tuple(readings),
    )


def _read_live_probe(
    definition: MemoryProbeDefinition,
    *,
    save_ram: bytes | None,
    read_system_ram: Callable[[int, int], bytes] | None,
) -> MemoryProbeReading:
    match definition.region:
        case "save_ram":
            if save_ram is None:
                raise ValueError("save_ram probe requires a save-RAM reader")
            return _read_probe({"save_ram": save_ram}, definition)
        case "system_ram":
            if read_system_ram is None:
                raise ValueError("system_ram probe requires a system-RAM reader")
            data = read_system_ram(definition.offset, definition.length)
            if len(data) != definition.length:
                raise ValueError(
                    f"system_ram probe {definition.key!r} read {len(data)} bytes; "
                    f"expected {definition.length}"
                )
            return _reading_from_bytes(definition, data)


def _read_probe(
    buffers: Mapping[MemoryProbeRegion, bytes],
    definition: MemoryProbeDefinition,
) -> MemoryProbeReading:
    buffer = buffers.get(definition.region)
    if buffer is None:
        raise ValueError(f"missing buffer for {definition.region!r}")
    if definition.end_offset > len(buffer):
        raise ValueError(
            f"probe {definition.key!r} reads {definition.region} bytes "
            f"{definition.offset}:{definition.end_offset}, but buffer has {len(buffer)} bytes"
        )
    return _reading_from_bytes(definition, buffer[definition.offset : definition.end_offset])


def _reading_from_bytes(
    definition: MemoryProbeDefinition,
    data: bytes,
) -> MemoryProbeReading:
    return MemoryProbeReading(
        key=definition.key,
        label=definition.label,
        region=definition.region,
        offset=definition.offset,
        length=definition.length,
        value_format=definition.value_format,
        raw_hex=data.hex(),
        value=_decode_probe_value(data, definition.value_format),
    )


def _decode_probe_value(data: bytes, value_format: MemoryProbeFormat) -> MemoryProbeValue:
    match value_format:
        case "bytes":
            return data.hex()
        case "bitset_u8":
            return tuple(
                byte_index * 8 + bit
                for byte_index, byte in enumerate(data)
                for bit in range(8)
                if byte & (1 << bit)
            )
        case "u8":
            return data[0]
        case "u16_be":
            return int.from_bytes(data, byteorder="big")
        case "u16_le":
            return int.from_bytes(data, byteorder="little")
        case "u32_be":
            return int.from_bytes(data, byteorder="big")
        case "u32_le":
            return int.from_bytes(data, byteorder="little")


def _parse_region(value: str) -> MemoryProbeRegion:
    match value:
        case "save_ram":
            return "save_ram"
        case "system_ram":
            return "system_ram"
        case _:
            raise ValueError(f"unknown probe region {value!r}")


def _parse_format(value: str) -> MemoryProbeFormat:
    match value:
        case "bytes":
            return "bytes"
        case "bitset_u8":
            return "bitset_u8"
        case "u8":
            return "u8"
        case "u16_be":
            return "u16_be"
        case "u16_le":
            return "u16_le"
        case "u32_be":
            return "u32_be"
        case "u32_le":
            return "u32_le"
        case _:
            raise ValueError(f"unknown probe format {value!r}")


def _parse_non_negative_int(value: str, *, field: str) -> int:
    parsed = _parse_int(value, field=field)
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def _parse_positive_int(value: str, *, field: str) -> int:
    parsed = _parse_int(value, field=field)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _parse_int(value: str, *, field: str) -> int:
    try:
        return int(value, 0)
    except ValueError as error:
        raise ValueError(f"{field} must be an integer") from error


def _validate_probe_width(definition: MemoryProbeDefinition) -> None:
    expected_widths = {
        "u8": 1,
        "u16_be": 2,
        "u16_le": 2,
        "u32_be": 4,
        "u32_le": 4,
    }
    expected = expected_widths.get(definition.value_format)
    if expected is not None and definition.length != expected:
        raise ValueError(
            f"{definition.value_format} probe {definition.key!r} must have length {expected}"
        )
