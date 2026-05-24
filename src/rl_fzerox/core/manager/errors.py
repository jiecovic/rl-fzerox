# src/rl_fzerox/core/manager/errors.py
from __future__ import annotations


class ManagerNameConflictError(ValueError):
    """Raised when a draft or run name collides with an existing record."""

    def __init__(self, *, kind: str, name: str) -> None:
        self.kind = kind
        self.name = name
        super().__init__(f"name already exists: {name}")
