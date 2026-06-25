# src/rl_fzerox/core/save_game/sram.py
"""Conservative summaries and byte diffs for portable F-Zero X save RAM.

These helpers intentionally do not depend on mapped save offsets. They provide
stable diagnostics for tooling that compares raw SRAM/SRM snapshots while the
game-specific unlock decoder lives in `unlocks.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class SaveRamSummary:
    """Basic facts that are safe to report before save offsets are mapped."""

    size: int
    sha256: str
    nonzero_bytes: int
    unique_byte_values: int


@dataclass(frozen=True, slots=True)
class SaveRamRangeDiff:
    """One contiguous changed byte range between two save-RAM snapshots."""

    offset: int
    length: int
    before_hex: str
    after_hex: str


@dataclass(frozen=True, slots=True)
class SaveRamBitDiff:
    """One changed bit, numbered least-significant-bit first within its byte."""

    offset: int
    bit: int
    mask_hex: str
    before: int
    after: int


@dataclass(frozen=True, slots=True)
class SaveRamDiff:
    """Byte-level diff between two same-sized save-RAM snapshots."""

    before: SaveRamSummary
    after: SaveRamSummary
    changed_bytes: int
    changed_ranges: tuple[SaveRamRangeDiff, ...]
    truncated_ranges: int


@dataclass(frozen=True, slots=True)
class SaveRamBitDiffReport:
    """Bit-level diff between two same-sized save-RAM snapshots."""

    before: SaveRamSummary
    after: SaveRamSummary
    changed_bits: int
    bit_changes: tuple[SaveRamBitDiff, ...]
    truncated_bits: int


@dataclass(frozen=True, slots=True)
class _RangeDiffResult:
    changed_bytes: int
    changed_ranges: tuple[SaveRamRangeDiff, ...]
    total_ranges: int


def summarize_save_ram(data: bytes) -> SaveRamSummary:
    """Return a stable, offset-agnostic summary for a save-RAM buffer."""

    return SaveRamSummary(
        size=len(data),
        sha256=sha256(data).hexdigest(),
        nonzero_bytes=sum(byte != 0 for byte in data),
        unique_byte_values=len(set(data)),
    )


def diff_save_ram_bits(before: bytes, after: bytes, *, max_bits: int = 256) -> SaveRamBitDiffReport:
    """Return changed bits for equal-sized save-RAM buffers."""

    if len(before) != len(after):
        raise ValueError(
            f"save buffers must have the same size: before={len(before)}, after={len(after)}"
        )
    if max_bits < 0:
        raise ValueError("max_bits must be non-negative")

    changed_bits = sum(
        (left ^ right).bit_count() for left, right in zip(before, after, strict=True)
    )
    bit_changes = _bit_changes(before, after, max_bits=max_bits)
    return SaveRamBitDiffReport(
        before=summarize_save_ram(before),
        after=summarize_save_ram(after),
        changed_bits=changed_bits,
        bit_changes=bit_changes,
        truncated_bits=max(0, changed_bits - len(bit_changes)),
    )


def diff_save_ram(before: bytes, after: bytes, *, max_ranges: int = 64) -> SaveRamDiff:
    """Return coalesced changed byte ranges for equal-sized save-RAM buffers."""

    if len(before) != len(after):
        raise ValueError(
            f"save buffers must have the same size: before={len(before)}, after={len(after)}"
        )
    if max_ranges < 0:
        raise ValueError("max_ranges must be non-negative")

    result = _changed_range_diffs(before, after, max_ranges=max_ranges)
    return SaveRamDiff(
        before=summarize_save_ram(before),
        after=summarize_save_ram(after),
        changed_bytes=result.changed_bytes,
        changed_ranges=result.changed_ranges,
        truncated_ranges=max(0, result.total_ranges - len(result.changed_ranges)),
    )


def _bit_changes(
    before: bytes,
    after: bytes,
    *,
    max_bits: int,
) -> tuple[SaveRamBitDiff, ...]:
    if max_bits == 0:
        return ()

    bit_changes: list[SaveRamBitDiff] = []
    for offset, (left, right) in enumerate(zip(before, after, strict=True)):
        changed_bits = left ^ right
        if changed_bits == 0:
            continue
        for bit in range(8):
            mask = 1 << bit
            if not changed_bits & mask:
                continue
            bit_changes.append(
                SaveRamBitDiff(
                    offset=offset,
                    bit=bit,
                    mask_hex=f"0x{mask:02x}",
                    before=1 if left & mask else 0,
                    after=1 if right & mask else 0,
                )
            )
            if len(bit_changes) == max_bits:
                return tuple(bit_changes)
    return tuple(bit_changes)


def _changed_range_diffs(
    before: bytes,
    after: bytes,
    *,
    max_ranges: int,
) -> _RangeDiffResult:
    changed_bytes = 0
    total_ranges = 0
    ranges: list[SaveRamRangeDiff] = []
    range_start: int | None = None
    previous_changed_offset: int | None = None

    for offset, (left, right) in enumerate(zip(before, after, strict=True)):
        if left == right:
            if range_start is not None and previous_changed_offset is not None:
                total_ranges += 1
                _append_range_diff(
                    ranges,
                    before=before,
                    after=after,
                    start=range_start,
                    end=previous_changed_offset + 1,
                    max_ranges=max_ranges,
                )
                range_start = None
                previous_changed_offset = None
            continue

        changed_bytes += 1
        if range_start is None:
            range_start = offset
        previous_changed_offset = offset

    if range_start is not None and previous_changed_offset is not None:
        total_ranges += 1
        _append_range_diff(
            ranges,
            before=before,
            after=after,
            start=range_start,
            end=previous_changed_offset + 1,
            max_ranges=max_ranges,
        )

    return _RangeDiffResult(
        changed_bytes=changed_bytes,
        changed_ranges=tuple(ranges),
        total_ranges=total_ranges,
    )


def _append_range_diff(
    ranges: list[SaveRamRangeDiff],
    *,
    before: bytes,
    after: bytes,
    start: int,
    end: int,
    max_ranges: int,
) -> None:
    if len(ranges) < max_ranges:
        ranges.append(_range_diff(before, after, start, end))


def _range_diff(before: bytes, after: bytes, start: int, end: int) -> SaveRamRangeDiff:
    return SaveRamRangeDiff(
        offset=start,
        length=end - start,
        before_hex=before[start:end].hex(),
        after_hex=after[start:end].hex(),
    )
