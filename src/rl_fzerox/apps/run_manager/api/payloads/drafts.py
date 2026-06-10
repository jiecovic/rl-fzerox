# src/rl_fzerox/apps/run_manager/api/payloads/drafts.py
from __future__ import annotations

from rl_fzerox.core.manager import ManagedRunDraft, ManagedRunTemplate


def template_payload(template: ManagedRunTemplate) -> dict[str, object]:
    return {
        "id": template.id,
        "name": template.name,
        "created_at": template.created_at,
        "updated_at": template.updated_at,
        "config": template.config.model_dump(mode="json"),
    }


def draft_payload(draft: ManagedRunDraft) -> dict[str, object]:
    return {
        "id": draft.id,
        "name": draft.name,
        "source_run_id": draft.source_run_id,
        "source_artifact": draft.source_artifact,
        "source_num_timesteps": draft.source_num_timesteps,
        "created_at": draft.created_at,
        "updated_at": draft.updated_at,
        "config": draft.config.model_dump(mode="json"),
    }
