# src/rl_fzerox/core/manager/registry/lineages/order.py
"""Helpers for stable lineage deletion ordering."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LineageRunLink:
    """Run graph edge needed to delete children before parents."""

    run_id: str
    parent_run_id: str | None
    source_run_id: str | None


def delete_order_for_lineage(rows: Iterable[LineageRunLink]) -> tuple[str, ...]:
    parent_by_id = {row.run_id: row.parent_run_id or row.source_run_id for row in rows}
    depth_by_id: dict[str, int] = {}

    def depth(run_id: str) -> int:
        existing = depth_by_id.get(run_id)
        if existing is not None:
            return existing
        parent_id = parent_by_id[run_id]
        if parent_id is None or parent_id not in parent_by_id:
            depth_by_id[run_id] = 0
            return 0
        resolved_depth = depth(parent_id) + 1
        depth_by_id[run_id] = resolved_depth
        return resolved_depth

    ordered = sorted(
        parent_by_id,
        key=lambda run_id: (depth(run_id), run_id),
        reverse=True,
    )
    return tuple(ordered)
