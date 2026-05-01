# src/rl_fzerox/ui/watch/view/panels/content/records/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

RecordInfo: TypeAlias = dict[str, object]


@dataclass(frozen=True, slots=True)
class RecordGroup:
    """Track records grouped by cup while preserving first-seen ordering."""

    cup: str | None
    records: tuple[RecordInfo, ...]
    first_index: int


BUILT_IN_CUP_ORDER = ("jack", "queen", "king", "joker")
BUILT_IN_COURSES_PER_CUP = 6
