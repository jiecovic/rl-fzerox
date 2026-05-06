# src/rl_fzerox/core/manager/registry/drafts/__init__.py
"""Draft and template persistence helpers for the manager UI."""
from rl_fzerox.core.manager.registry.drafts.store import (
    create_draft,
    default_template,
    delete_draft,
    get_draft,
    list_drafts,
    list_templates,
    update_draft,
)

__all__ = [
    "create_draft",
    "default_template",
    "delete_draft",
    "get_draft",
    "list_drafts",
    "list_templates",
    "update_draft",
]
