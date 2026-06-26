# src/rl_fzerox/core/manager/fixtures/web.py
"""Export frontend test fixtures from the canonical manager backend surface.

Run-manager web tests and default UI state read these generated fixtures. When
backend default values change, regenerate them with ``npm run sync-fixtures`` in
``web/run-manager`` so the frontend keeps the backend as the
single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from rl_fzerox.core.manager.architecture import (
    policy_architecture_preview,
    run_manager_config_metadata,
)
from rl_fzerox.core.manager.architecture.models import RuntimeAssetInfo
from rl_fzerox.core.manager.run_spec import default_managed_run_config

WEB_FIXTURE_RUNTIME_ASSETS = (
    RuntimeAssetInfo(
        id="libretro_core",
        label="Mupen64Plus-Next libretro core",
        path="local/libretro/mupen64plus_next_libretro.so",
        exists=False,
    ),
    RuntimeAssetInfo(
        id="fzerox_rom",
        label="F-Zero X US ROM",
        path="local/roms/fzerox_usa.n64",
        exists=False,
    ),
)


def web_test_fixture_payload() -> dict[str, object]:
    """Return frontend test fixtures from backend-owned manager defaults."""
    config = default_managed_run_config()
    return {
        "managed_run_config": config.model_dump(mode="json"),
        "config_metadata": run_manager_config_metadata(
            runtime_assets=WEB_FIXTURE_RUNTIME_ASSETS,
        ).model_dump(mode="json"),
        "policy_preview": policy_architecture_preview(config).model_dump(mode="json"),
    }


def web_test_fixture_text() -> str:
    return json.dumps(web_test_fixture_payload(), indent=2, sort_keys=True) + "\n"


def web_test_fixture_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "web/run-manager/src/test/generated/manager-fixtures.json"
    )


def export_web_test_fixtures() -> Path:
    """Write frontend test fixtures from backend-owned manager defaults."""

    output_path = web_test_fixture_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(web_test_fixture_text(), encoding="utf-8")
    return output_path


def check_web_test_fixtures() -> bool:
    """Return whether checked-in frontend fixtures match backend defaults."""

    output_path = web_test_fixture_output_path()
    if not output_path.exists():
        return False
    try:
        checked_in_payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return checked_in_payload == web_test_fixture_payload()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export run-manager frontend fixtures")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in generated fixture is stale",
    )
    args = parser.parse_args()

    if not args.check:
        export_web_test_fixtures()
        return

    if check_web_test_fixtures():
        return
    raise SystemExit(
        "run-manager frontend fixtures are stale; backend defaults changed, so "
        "run npm run sync-fixtures --prefix web/run-manager"
    )


if __name__ == "__main__":
    main()
