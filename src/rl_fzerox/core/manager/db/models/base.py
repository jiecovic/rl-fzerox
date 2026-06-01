# src/rl_fzerox/core/manager/db/models/base.py
"""Declarative base shared by manager ORM models."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class ManagerBase(DeclarativeBase):
    """Base class for manager DB ORM models."""
