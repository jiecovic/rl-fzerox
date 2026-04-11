# src/rl_fzerox/core/training/inference/metadata.py
from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict

from rl_fzerox.core.training.session.artifacts import (
    load_policy_artifact_metadata,
    policy_artifact_metadata_path,
)


class _LoadedPolicyMetadataFields(TypedDict):
    curriculum_stage_index: NotRequired[int | None]
    curriculum_stage_name: NotRequired[str | None]


def _loaded_policy_metadata_fields(policy_path: Path) -> _LoadedPolicyMetadataFields:
    metadata = load_policy_artifact_metadata(policy_path)
    if metadata is None:
        return {}
    return {
        "curriculum_stage_index": metadata.curriculum_stage_index,
        "curriculum_stage_name": metadata.curriculum_stage_name,
    }


def _policy_metadata_mtime_ns(policy_path: Path) -> int | None:
    """Return the sidecar metadata mtime, if the checkpoint currently has one."""

    metadata_path = policy_artifact_metadata_path(policy_path)
    if not metadata_path.is_file():
        return None
    return metadata_path.stat().st_mtime_ns
