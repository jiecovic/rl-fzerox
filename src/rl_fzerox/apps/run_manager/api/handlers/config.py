# src/rl_fzerox/apps/run_manager/api/handlers/config.py
from __future__ import annotations

from rl_fzerox.core.manager import ManagedRunConfig
from rl_fzerox.core.manager.architecture import (
    policy_architecture_preview,
    run_manager_config_metadata,
)


def config_metadata_payload() -> dict[str, object]:
    metadata = run_manager_config_metadata()
    return metadata.model_dump(mode="json")


def policy_preview_payload(config: ManagedRunConfig) -> dict[str, object]:
    preview = policy_architecture_preview(config)
    return preview.model_dump(mode="json")
