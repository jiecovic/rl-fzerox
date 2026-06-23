# src/rl_fzerox/core/manager/transfer/errors.py
"""Errors raised by manager run transfer operations."""

from __future__ import annotations


class RunBundleError(RuntimeError):
    """Raised when a run bundle cannot be exported or imported safely."""
