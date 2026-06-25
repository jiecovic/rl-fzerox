# src/rl_fzerox/core/manager/registry/drafts/store.py
"""Draft and template registry operations for managed run specs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError as SqlAlchemyIntegrityError

from rl_fzerox.core.manager.db.models import RunDraftModel, RunTemplateModel
from rl_fzerox.core.manager.db.repositories.configs import create_config_snapshot
from rl_fzerox.core.manager.db.repositories.runs import (
    assert_draft_name_available,
    insert_draft,
)
from rl_fzerox.core.manager.db.repositories.runs import (
    update_draft as update_draft_record,
)
from rl_fzerox.core.manager.errors import ManagerNameConflictError
from rl_fzerox.core.manager.models import ManagedRunDraft, ManagedRunTemplate
from rl_fzerox.core.manager.registry.common import (
    new_record_id,
    optional_source_artifact,
    utc_now,
)
from rl_fzerox.core.manager.registry.drafts.fork_sources import (
    reset_draft_source,
    snapshot_draft_source,
)
from rl_fzerox.core.manager.run_spec import ManagedRunConfig, reset_fork_action_bias_deltas
from rl_fzerox.core.manager.storage.serialization import load_config_json

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def create_draft(
    store: ManagerStore,
    *,
    name: str,
    config: ManagedRunConfig,
    source_run_id: str | None = None,
    source_artifact: Literal["latest", "best"] | None = None,
) -> ManagedRunDraft:
    """Persist one mutable draft and pin a fork source when requested."""

    store.initialize()
    created_at = utc_now()
    draft_id = new_record_id(name)
    normalized_name = name.strip() or draft_id
    source_snapshot_dir = None
    source_num_timesteps = None
    source_run = None if source_run_id is None else store.get_run(source_run_id)
    if source_run_id is not None and source_run is None:
        raise ValueError(f"run not found: {source_run_id}")
    if source_run_id is not None and source_artifact is not None:
        config = reset_fork_action_bias_deltas(config)
    draft: ManagedRunDraft | None = None

    try:
        with store._orm_session() as session:
            assert_draft_name_available(session, normalized_name)
            config_snapshot = create_config_snapshot(
                session,
                kind="draft",
                config=config,
                created_at=created_at,
            )
            draft = ManagedRunDraft(
                id=draft_id,
                name=normalized_name,
                config=config,
                config_hash=config_snapshot.config_hash,
                source_run_id=source_run_id,
                source_artifact=source_artifact,
                source_snapshot_dir=source_snapshot_dir,
                source_num_timesteps=source_num_timesteps,
                created_at=created_at,
                updated_at=created_at,
            )
            if source_run_id is not None and source_artifact is not None:
                assert source_run is not None
                source_snapshot_dir, source_num_timesteps = snapshot_draft_source(
                    manager_db_path=store.db_path,
                    draft_id=draft_id,
                    source_run=source_run,
                    source_artifact=source_artifact,
                )
                draft = replace(
                    draft,
                    source_run_id=source_run_id,
                    source_artifact=source_artifact,
                    source_snapshot_dir=source_snapshot_dir,
                    source_num_timesteps=source_num_timesteps,
                )
            insert_draft(
                session,
                draft=draft,
                config_snapshot_id=config_snapshot.id,
            )
    except SqlAlchemyIntegrityError as error:
        raise ManagerNameConflictError(kind="draft", name=normalized_name) from error
    except Exception:
        reset_draft_source(source_snapshot_dir)
        raise
    if draft is None:
        raise RuntimeError("managed draft creation failed before insert")
    return draft


def list_drafts(store: ManagerStore) -> tuple[ManagedRunDraft, ...]:
    store.initialize()
    with store._orm_session() as session:
        drafts = tuple(
            session.scalars(
                select(RunDraftModel).order_by(
                    RunDraftModel.updated_at.desc(),
                    RunDraftModel.id.desc(),
                )
            )
        )
        return tuple(_draft_from_model(draft) for draft in drafts)


def delete_draft(store: ManagerStore, draft_id: str) -> bool:
    store.initialize()
    source_snapshot_dir: Path | None = None
    with store._orm_session() as session:
        draft = session.get(RunDraftModel, draft_id)
        if draft is None:
            return False
        if draft.source_snapshot_dir is not None:
            source_snapshot_dir = Path(draft.source_snapshot_dir).expanduser().resolve()
        session.delete(draft)
    if source_snapshot_dir is not None:
        reset_draft_source(source_snapshot_dir)
    return True


def update_draft(
    store: ManagerStore,
    *,
    draft_id: str,
    name: str,
    config: ManagedRunConfig,
    source_run_id: str | None = None,
    source_artifact: Literal["latest", "best"] | None = None,
) -> ManagedRunDraft | None:
    store.initialize()
    updated_at = utc_now()
    normalized_name = name.strip() or draft_id
    current_draft = store.get_draft(draft_id)
    if current_draft is None:
        return None
    source_run = None if source_run_id is None else store.get_run(source_run_id)
    if source_run_id is not None and source_run is None:
        raise ValueError(f"run not found: {source_run_id}")
    next_snapshot_dir = current_draft.source_snapshot_dir
    next_source_num_timesteps = current_draft.source_num_timesteps
    source_changed = (
        current_draft.source_run_id != source_run_id
        or current_draft.source_artifact != source_artifact
    )
    if source_changed and current_draft.source_run_id is not None:
        raise ValueError("changing a fork draft source is not supported; create a new fork draft")
    updated = False
    try:
        with store._orm_session() as session:
            assert_draft_name_available(
                session,
                normalized_name,
                exclude_draft_id=draft_id,
            )
            if source_changed:
                next_snapshot_dir = None
                next_source_num_timesteps = None
                if source_run_id is not None and source_artifact is not None:
                    assert source_run is not None
                    next_snapshot_dir, next_source_num_timesteps = snapshot_draft_source(
                        manager_db_path=store.db_path,
                        draft_id=draft_id,
                        source_run=source_run,
                        source_artifact=source_artifact,
                    )
            config_snapshot = create_config_snapshot(
                session,
                kind="draft",
                config=config,
                created_at=updated_at,
            )
            updated = update_draft_record(
                session,
                draft_id=draft_id,
                name=normalized_name,
                config_snapshot_id=config_snapshot.id,
                source_run_id=source_run_id,
                source_artifact=source_artifact,
                source_snapshot_dir=None if next_snapshot_dir is None else str(next_snapshot_dir),
                source_num_timesteps=next_source_num_timesteps,
                updated_at=updated_at,
            )
    except SqlAlchemyIntegrityError as error:
        raise ManagerNameConflictError(kind="draft", name=normalized_name) from error
    return store.get_draft(draft_id) if updated else None


def get_draft(store: ManagerStore, draft_id: str) -> ManagedRunDraft | None:
    store.initialize()
    with store._orm_session() as session:
        draft = session.get(RunDraftModel, draft_id)
        return None if draft is None else _draft_from_model(draft)


def list_templates(store: ManagerStore) -> tuple[ManagedRunTemplate, ...]:
    store.initialize()
    with store._orm_session() as session:
        templates = tuple(session.scalars(select(RunTemplateModel).order_by(RunTemplateModel.name)))
        return tuple(_template_from_model(template) for template in templates)


def default_template(store: ManagerStore) -> ManagedRunTemplate:
    store.refresh_system_templates()
    templates = list_templates(store)
    if not templates:
        raise RuntimeError("Manager DB did not initialize a default run template")
    return templates[0]


def _draft_from_model(draft: RunDraftModel) -> ManagedRunDraft:
    return ManagedRunDraft(
        id=draft.id,
        name=draft.name,
        config=load_config_json(draft.config_snapshot.config_json),
        config_hash=draft.config_snapshot.config_hash,
        created_at=draft.created_at,
        updated_at=draft.updated_at,
        source_run_id=draft.source_run_id,
        source_artifact=optional_source_artifact(draft.source_artifact),
        source_snapshot_dir=(
            None if draft.source_snapshot_dir is None else Path(draft.source_snapshot_dir)
        ),
        source_num_timesteps=draft.source_num_timesteps,
    )


def _template_from_model(template: RunTemplateModel) -> ManagedRunTemplate:
    return ManagedRunTemplate(
        id=template.id,
        name=template.name,
        config=load_config_json(template.config_snapshot.config_json),
        config_hash=template.config_snapshot.config_hash,
        created_at=template.created_at,
        updated_at=template.updated_at,
    )
