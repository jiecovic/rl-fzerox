# src/rl_fzerox/core/manager/registry/runs/__init__.py
"""Run-row persistence and maintenance helpers.

This package owns launched-run CRUD plus the runtime sidecars that hang off a
run row: metrics events, worker leases, commands, and filesystem cleanup.
"""

from rl_fzerox.core.manager.registry.runs.commands import (
    clear_run_command,
    pending_run_command,
    request_run_command,
)
from rl_fzerox.core.manager.registry.runs.lifecycle import (
    create_run,
    get_run,
    list_recent_run_events,
    list_runs,
    list_visible_run_summaries,
    list_visible_runs,
    update_run_name,
    update_run_status,
)
from rl_fzerox.core.manager.registry.runs.maintenance import (
    drain_pending_filesystem_operations,
    reconcile_orphaned_runs,
)
from rl_fzerox.core.manager.registry.runs.runtime import (
    append_run_event,
    clear_run_runtime,
    update_run_fork_source,
    upsert_run_runtime,
)
from rl_fzerox.core.manager.registry.runs.track_sampling import (
    clear_run_track_sampling_state,
    get_run_track_sampling_state,
    migrate_run_track_sampling_state_json,
    upsert_run_track_sampling_state,
)
from rl_fzerox.core.manager.registry.runs.workers import (
    clear_run_worker,
    heartbeat_run_worker,
    register_run_worker,
)

__all__ = [
    "append_run_event",
    "clear_run_command",
    "clear_run_runtime",
    "clear_run_track_sampling_state",
    "clear_run_worker",
    "create_run",
    "drain_pending_filesystem_operations",
    "get_run",
    "get_run_track_sampling_state",
    "heartbeat_run_worker",
    "list_recent_run_events",
    "list_runs",
    "list_visible_run_summaries",
    "list_visible_runs",
    "migrate_run_track_sampling_state_json",
    "pending_run_command",
    "reconcile_orphaned_runs",
    "register_run_worker",
    "request_run_command",
    "update_run_fork_source",
    "update_run_name",
    "update_run_status",
    "upsert_run_runtime",
    "upsert_run_track_sampling_state",
]
