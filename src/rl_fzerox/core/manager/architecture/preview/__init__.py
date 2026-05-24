# src/rl_fzerox/core/manager/architecture/preview/__init__.py
"""Owned preview submodule for policy-architecture rendering.

The heavy lifting is split across state, params, actions, lanes, and summary so
callers only import the single façade here.
"""

from rl_fzerox.core.manager.architecture.preview.summary import (
    policy_architecture_preview,
)

__all__ = ["policy_architecture_preview"]
