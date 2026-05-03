# src/rl_fzerox/core/manager/export_web_test_fixtures.py
from __future__ import annotations

import json
from pathlib import Path

from rl_fzerox.core.manager.architecture import (
    policy_architecture_preview,
    run_manager_config_metadata,
)
from rl_fzerox.core.manager.config import default_managed_run_config


def export_web_test_fixtures() -> Path:
    """Write frontend test fixtures from backend-owned manager defaults."""

    config = default_managed_run_config()
    payload = {
        "managed_run_config": config.model_dump(mode="json"),
        "config_metadata": run_manager_config_metadata().model_dump(mode="json"),
        "policy_preview": policy_architecture_preview(config).model_dump(mode="json"),
    }
    output_path = (
        Path(__file__).resolve().parents[4]
        / "src/rl_fzerox/apps/run_manager/web/src/test/generated/manager-fixtures.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    export_web_test_fixtures()


if __name__ == "__main__":
    main()
