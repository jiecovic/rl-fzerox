# src/rl_fzerox/core/manager/registry/runs/commands.py
from __future__ import annotations

from typing import TYPE_CHECKING

from rl_fzerox.core.manager.db.models import RunCommandModel, RunEventModel, RunModel
from rl_fzerox.core.manager.db.repositories.runs import managed_run_from_model
from rl_fzerox.core.manager.models import ManagedRun, RunCommand
from rl_fzerox.core.manager.registry.common import run_command, utc_now

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def request_run_command(
    store: ManagerStore,
    *,
    run_id: str,
    command: RunCommand,
) -> ManagedRun | None:
    store.initialize()
    requested_at = utc_now()
    with store._orm_session() as session:
        run = session.get(RunModel, run_id)
        if run is None:
            return None
        pending = session.get(RunCommandModel, run_id)
        if pending is None:
            session.add(
                RunCommandModel(
                    run_id=run_id,
                    command=command,
                    requested_at=requested_at,
                )
            )
        else:
            pending.command = command
            pending.requested_at = requested_at
        session.add(
            RunEventModel(
                run_id=run_id,
                created_at=requested_at,
                kind=f"{command}_requested",
                message=f"{command} requested from manager",
            )
        )
        session.flush()
        return managed_run_from_model(session, run)


def pending_run_command(store: ManagerStore, run_id: str) -> RunCommand | None:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        pending = session.get(RunCommandModel, run_id)
        return run_command(None if pending is None else pending.command)


def clear_run_command(
    store: ManagerStore,
    run_id: str,
    *,
    command: RunCommand | None = None,
) -> None:
    store._ensure_schema_initialized()
    with store._orm_session() as session:
        pending = session.get(RunCommandModel, run_id)
        if pending is None:
            return
        if command is None or pending.command == command:
            session.delete(pending)
