# src/rl_fzerox/apps/run_manager/api/validation.py
from __future__ import annotations

from fastapi import HTTPException


def required_name(value: str, *, subject: str) -> str:
    name = value.strip()
    if not name:
        raise HTTPException(status_code=400, detail=f"{subject} name is required")
    return name
