# src/rl_fzerox/core/manager/registry/lineages/metadata.py
from __future__ import annotations

from typing import TYPE_CHECKING

from rl_fzerox.core.manager.registry.common import utc_now

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def update_lineage_groups(
    store: ManagerStore,
    *,
    lineage_id: str,
    group_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Assign one lineage to display/TensorBoard groups without moving run files."""

    store.initialize()
    normalized_group_names = normalize_lineage_group_names(group_names)
    updated_at = utc_now()
    with store._connect() as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM runs
            WHERE lineage_id = ?
            LIMIT 1
            """,
            (lineage_id,),
        ).fetchone()
        if row is None:
            raise ValueError("lineage not found")

        connection.execute(
            "DELETE FROM lineage_groups WHERE lineage_id = ?",
            (lineage_id,),
        )
        for group_name in normalized_group_names:
            connection.execute(
                """
                INSERT INTO lineage_groups(lineage_id, group_name, updated_at)
                VALUES (?, ?, ?)
                """,
                (lineage_id, group_name, updated_at),
            )
    return normalized_group_names


def normalize_lineage_group_names(group_names: tuple[str, ...]) -> tuple[str, ...]:
    normalized_names = {
        normalized
        for group_name in group_names
        if (normalized := " ".join(group_name.strip().split()))
    }
    return tuple(sorted(normalized_names, key=str.casefold))
