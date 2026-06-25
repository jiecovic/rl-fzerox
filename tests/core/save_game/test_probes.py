# tests/core/save_game/test_probes.py
"""Coverage for ad-hoc save/system memory probe parsing and reads."""

from __future__ import annotations

import pytest

from rl_fzerox.core.save_game import (
    MemoryProbeDefinition,
    collect_memory_probe_report,
    parse_memory_probe_definition,
    read_memory_probes,
)


def test_parse_memory_probe_definition_accepts_hex_offsets_and_labels() -> None:
    definition = parse_memory_probe_definition("cup_flags=save_ram:0x10:4:bytes:Cup clear flags")

    assert definition.key == "cup_flags"
    assert definition.region == "save_ram"
    assert definition.offset == 16
    assert definition.length == 4
    assert definition.value_format == "bytes"
    assert definition.label == "Cup clear flags"


def test_read_memory_probes_decodes_common_formats() -> None:
    save_ram = bytes([0x01, 0x02, 0xA0, 0x0F, 0x80])
    definitions = (
        MemoryProbeDefinition("byte", "save_ram", 0, 1, "u8", "Byte"),
        MemoryProbeDefinition("word", "save_ram", 1, 2, "u16_be", "Word"),
        MemoryProbeDefinition("bits", "save_ram", 3, 2, "bitset_u8", "Bits"),
    )

    readings = read_memory_probes({"save_ram": save_ram}, definitions)

    assert readings[0].value == 1
    assert readings[1].value == 0x02A0
    assert readings[2].raw_hex == "0f80"
    assert readings[2].value == (0, 1, 2, 3, 15)


def test_collect_memory_probe_report_reads_system_ram_by_probe_range() -> None:
    system_reads: list[tuple[int, int]] = []

    def read_system_ram(offset: int, length: int) -> bytes:
        system_reads.append((offset, length))
        return bytes(range(offset, offset + length))

    report = collect_memory_probe_report(
        (
            MemoryProbeDefinition("save", "save_ram", 1, 2, "u16_le", "Save"),
            MemoryProbeDefinition("live", "system_ram", 4, 4, "u32_be", "Live"),
        ),
        read_save_ram=lambda: b"\x10\x20\x30\x40",
        read_system_ram=read_system_ram,
    )

    assert report.save_ram is not None
    assert report.save_ram.size == 4
    assert system_reads == [(4, 4)]
    assert [reading.value for reading in report.readings] == [0x3020, 0x04050607]


def test_memory_probe_validation_rejects_invalid_definitions() -> None:
    with pytest.raises(ValueError, match="key=region"):
        parse_memory_probe_definition("bad")
    with pytest.raises(ValueError, match="unknown probe region"):
        parse_memory_probe_definition("x=bad:0:1:u8")
    with pytest.raises(ValueError, match="must have length 2"):
        parse_memory_probe_definition("x=save_ram:0:1:u16_be")
    with pytest.raises(ValueError, match="length"):
        parse_memory_probe_definition("x=save_ram:0:0:bytes")


def test_read_memory_probes_rejects_out_of_range_reads() -> None:
    definition = MemoryProbeDefinition("x", "save_ram", 2, 2, "bytes", "X")

    with pytest.raises(ValueError, match="buffer has 3 bytes"):
        read_memory_probes({"save_ram": b"\x00\x01\x02"}, (definition,))


def test_collect_memory_probe_report_requires_matching_readers() -> None:
    with pytest.raises(ValueError, match="save-RAM reader"):
        collect_memory_probe_report(
            (MemoryProbeDefinition("save", "save_ram", 0, 1, "u8", "Save"),),
            read_save_ram=None,
        )

    with pytest.raises(ValueError, match="system-RAM reader"):
        collect_memory_probe_report(
            (MemoryProbeDefinition("live", "system_ram", 0, 1, "u8", "Live"),),
            read_save_ram=None,
        )


def test_collect_memory_probe_report_rejects_short_system_ram_reads() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        collect_memory_probe_report(
            (MemoryProbeDefinition("live", "system_ram", 0, 2, "u16_be", "Live"),),
            read_save_ram=None,
            read_system_ram=lambda _offset, _length: b"\x00",
        )
