# src/rl_fzerox/core/training/inference/metadata.py
from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict

from rl_fzerox.core.training.session.artifacts import load_policy_artifact_metadata


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
