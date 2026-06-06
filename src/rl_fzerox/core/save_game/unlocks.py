# src/rl_fzerox/core/save_game/unlocks.py
"""Decode F-Zero X save-file unlock progress."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rl_fzerox.core.domain.race_difficulty import (
    RACE_DIFFICULTIES,
    RaceDifficultyName,
    race_difficulty_raw_value,
)


@dataclass(frozen=True, slots=True)
class GpCupProgressOffset:
    """Logical save offset for one GP cup progression byte."""

    cup_id: str
    offset: int


@dataclass(frozen=True, slots=True)
class FZeroXSaveLayout:
    """Minimal save layout needed for unlock-progress inspection."""

    title: bytes
    raw_sra_size: int
    libretro_srm_size: int
    libretro_sra_offset: int
    byteswap_word_size: int
    gp_progress_offsets: tuple[GpCupProgressOffset, ...]


FZEROX_SAVE_LAYOUT = FZeroXSaveLayout(
    title=b"F-ZERO X",
    raw_sra_size=0x8000,
    libretro_srm_size=0x48800,
    libretro_sra_offset=0x20800,
    byteswap_word_size=4,
    gp_progress_offsets=(
        GpCupProgressOffset(cup_id="jack", offset=0x0A),
        GpCupProgressOffset(cup_id="queen", offset=0x0B),
        GpCupProgressOffset(cup_id="king", offset=0x0C),
        GpCupProgressOffset(cup_id="joker", offset=0x0D),
    ),
)


@dataclass(frozen=True, slots=True)
class GpCupProgress:
    """Decoded highest-cleared GP difficulty for one cup."""

    cup_id: str
    raw_value: int
    highest_cleared_difficulty: RaceDifficultyName | None


@dataclass(frozen=True, slots=True)
class FZeroXUnlockState:
    """Known unlock state decoded from a normalized F-Zero X save file."""

    gp_cup_progress: tuple[GpCupProgress, ...]

    def gp_cup_cleared(self, *, difficulty: RaceDifficultyName, cup_id: str) -> bool:
        clear_level = race_difficulty_raw_value(difficulty) + 1
        return self._cup_progress_value(cup_id) >= clear_level

    def _cup_progress_value(self, cup_id: str) -> int:
        for progress in self.gp_cup_progress:
            if progress.cup_id == cup_id:
                return progress.raw_value
        return 0


class FZeroXSaveDecodeError(ValueError):
    """Raised when a save file is not an inspectable F-Zero X save."""


def read_fzerox_unlock_state(save_path: Path) -> FZeroXUnlockState | None:
    """Read unlock progress from a local save file when it exists and is valid."""

    try:
        return decode_fzerox_unlock_state(save_path.read_bytes())
    except (OSError, FZeroXSaveDecodeError):
        return None


def decode_fzerox_unlock_state(save_data: bytes) -> FZeroXUnlockState:
    """Decode known F-Zero X unlock state from raw SRA or libretro SRM bytes."""

    logical_save = _logical_sra_payload(save_data, layout=FZEROX_SAVE_LAYOUT)
    progress = tuple(
        _decode_gp_progress(logical_save, offset)
        for offset in FZEROX_SAVE_LAYOUT.gp_progress_offsets
    )
    return FZeroXUnlockState(gp_cup_progress=progress)


def _decode_gp_progress(
    logical_save: bytes,
    offset: GpCupProgressOffset,
) -> GpCupProgress:
    raw_value = logical_save[offset.offset]
    return GpCupProgress(
        cup_id=offset.cup_id,
        raw_value=raw_value,
        highest_cleared_difficulty=_difficulty_for_progress(raw_value),
    )


def _difficulty_for_progress(raw_value: int) -> RaceDifficultyName | None:
    if raw_value <= 0:
        return None

    clear_index = min(raw_value, len(RACE_DIFFICULTIES)) - 1
    return RACE_DIFFICULTIES[clear_index].name


def _logical_sra_payload(save_data: bytes, *, layout: FZeroXSaveLayout) -> bytes:
    if len(save_data) == layout.libretro_srm_size:
        payload = save_data[
            layout.libretro_sra_offset : layout.libretro_sra_offset + layout.raw_sra_size
        ]
    elif len(save_data) >= layout.raw_sra_size:
        payload = save_data[: layout.raw_sra_size]
    else:
        raise FZeroXSaveDecodeError(
            f"save file too small for F-Zero X SRA data: {len(save_data)} bytes"
        )

    if payload.startswith(layout.title):
        return payload

    swapped_payload = _byteswap_chunks(payload, chunk_size=layout.byteswap_word_size)
    if swapped_payload.startswith(layout.title):
        return swapped_payload

    raise FZeroXSaveDecodeError("save file does not contain an F-Zero X SRA header")


def _byteswap_chunks(payload: bytes, *, chunk_size: int) -> bytes:
    chunks = (
        payload[index : index + chunk_size][::-1]
        for index in range(0, len(payload), chunk_size)
    )
    return b"".join(chunks)
