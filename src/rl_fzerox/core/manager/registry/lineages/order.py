# src/rl_fzerox/core/manager/registry/lineages/order.py
from __future__ import annotations

import sqlite3
from pathlib import Path

from rl_fzerox.core.manager.registry.common import optional_str


def delete_order_for_lineage(rows: list[sqlite3.Row]) -> tuple[str, ...]:
    parent_by_id = {
        str(row["id"]): optional_str(row["parent_run_id"]) or optional_str(row["source_run_id"])
        for row in rows
    }
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
        (str(row["id"]) for row in rows),
        key=lambda run_id: (depth(run_id), run_id),
        reverse=True,
    )
    return tuple(ordered)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
