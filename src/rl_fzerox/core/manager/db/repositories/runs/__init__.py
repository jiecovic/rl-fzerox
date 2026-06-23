# src/rl_fzerox/core/manager/db/repositories/runs/__init__.py
"""Facade for run, draft, template, and event repository operations."""

from __future__ import annotations

from rl_fzerox.core.manager.db.repositories.runs.drafts import (
    assert_draft_name_available,
    insert_draft,
    update_draft,
    upsert_template,
)
from rl_fzerox.core.manager.db.repositories.runs.events import (
    append_run_event,
    list_recent_managed_run_events,
)
from rl_fzerox.core.manager.db.repositories.runs.mapping import (
    managed_run_from_model,
    managed_run_summary_from_model,
)
from rl_fzerox.core.manager.db.repositories.runs.records import (
    get_managed_run,
    insert_run,
    list_managed_runs,
    list_visible_managed_run_summaries,
    rename_run,
    resolve_lineage_id,
    set_run_status,
)

__all__ = [
    "append_run_event",
    "assert_draft_name_available",
    "get_managed_run",
    "insert_draft",
    "insert_run",
    "list_managed_runs",
    "list_recent_managed_run_events",
    "list_visible_managed_run_summaries",
    "managed_run_from_model",
    "managed_run_summary_from_model",
    "rename_run",
    "resolve_lineage_id",
    "set_run_status",
    "update_draft",
    "upsert_template",
]
