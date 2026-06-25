# tests/core/save_game/test_sram.py
"""Coverage for offset-agnostic save-RAM summaries and diff reports."""

from dataclasses import asdict

import pytest

from rl_fzerox.core.save_game import diff_save_ram, diff_save_ram_bits, summarize_save_ram


def test_summarize_save_ram_reports_stable_basic_facts() -> None:
    summary = summarize_save_ram(bytes([0, 1, 1, 255]))

    assert summary.size == 4
    assert summary.nonzero_bytes == 3
    assert summary.unique_byte_values == 3
    assert len(summary.sha256) == 64


def test_diff_save_ram_coalesces_changed_ranges() -> None:
    before = bytes([0, 1, 2, 3, 4, 5, 6])
    after = bytes([0, 9, 8, 3, 4, 7, 6])

    diff = diff_save_ram(before, after)

    assert diff.changed_bytes == 3
    assert diff.truncated_ranges == 0
    assert [(item.offset, item.length) for item in diff.changed_ranges] == [(1, 2), (5, 1)]
    assert diff.changed_ranges[0].before_hex == "0102"
    assert diff.changed_ranges[0].after_hex == "0908"


def test_diff_save_ram_can_truncate_reported_ranges() -> None:
    before = bytes([0, 0, 0, 0, 0])
    after = bytes([1, 0, 2, 0, 3])

    diff = diff_save_ram(before, after, max_ranges=2)

    assert diff.changed_bytes == 3
    assert [(item.offset, item.length) for item in diff.changed_ranges] == [(0, 1), (2, 1)]
    assert diff.truncated_ranges == 1


def test_diff_save_ram_can_omit_reported_ranges() -> None:
    before = bytes([0, 0, 0])
    after = bytes([1, 0, 2])

    diff = diff_save_ram(before, after, max_ranges=0)

    assert diff.changed_bytes == 2
    assert diff.changed_ranges == ()
    assert diff.truncated_ranges == 2


def test_diff_save_ram_rejects_mismatched_sizes() -> None:
    with pytest.raises(ValueError, match="same size"):
        diff_save_ram(b"\x00", b"\x00\x00")
    with pytest.raises(ValueError, match="max_ranges"):
        diff_save_ram(b"\x00", b"\x00", max_ranges=-1)


def test_diff_save_ram_bits_reports_changed_bits() -> None:
    before = bytes([0b0000_0000, 0b0000_0011])
    after = bytes([0b0000_0101, 0b0000_0010])

    diff = diff_save_ram_bits(before, after)

    assert diff.changed_bits == 3
    assert diff.truncated_bits == 0
    assert [asdict(item) for item in diff.bit_changes] == [
        {"offset": 0, "bit": 0, "mask_hex": "0x01", "before": 0, "after": 1},
        {"offset": 0, "bit": 2, "mask_hex": "0x04", "before": 0, "after": 1},
        {"offset": 1, "bit": 0, "mask_hex": "0x01", "before": 1, "after": 0},
    ]


def test_diff_save_ram_bits_can_truncate_reported_bits() -> None:
    diff = diff_save_ram_bits(b"\x00", b"\xff", max_bits=3)

    assert diff.changed_bits == 8
    assert [(item.offset, item.bit) for item in diff.bit_changes] == [(0, 0), (0, 1), (0, 2)]
    assert diff.truncated_bits == 5


def test_diff_save_ram_bits_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same size"):
        diff_save_ram_bits(b"\x00", b"\x00\x00")
    with pytest.raises(ValueError, match="max_bits"):
        diff_save_ram_bits(b"\x00", b"\x00", max_bits=-1)
