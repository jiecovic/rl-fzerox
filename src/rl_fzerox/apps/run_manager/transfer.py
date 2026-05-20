# src/rl_fzerox/apps/run_manager/transfer.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.core.manager import ManagerStore, default_manager_db_path
from rl_fzerox.core.manager.transfer import RunBundleError, export_run_bundle, import_run_bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export or import portable run-manager run bundles.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=default_manager_db_path(),
        help="Manager database path. Defaults to local/manager/runs.db.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export one run to a zip bundle.")
    export_parser.add_argument("run_id", help="Run id to export.")
    export_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .zip path. Defaults to local/exports/<run_id>.zip.",
    )
    export_parser.add_argument(
        "--allow-running",
        action="store_true",
        help="Allow exporting a running run. Use only when you accept an inconsistent snapshot.",
    )

    import_parser = subparsers.add_parser("import", help="Import one run zip bundle.")
    import_parser.add_argument("bundle", type=Path, help="Bundle .zip path to import.")
    import_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional replacement run id. By default the exported run id is preserved.",
    )
    import_parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Target managed-runs root. Defaults to local/runs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    store = ManagerStore(args.db)
    try:
        match args.command:
            case "export":
                bundle_path = export_run_bundle(
                    store=store,
                    run_id=args.run_id,
                    output_path=args.output,
                    allow_running=args.allow_running,
                )
                print(bundle_path)
            case "import":
                result = import_run_bundle(
                    store=store,
                    bundle_path=args.bundle,
                    run_id=args.run_id,
                    managed_runs_root=args.runs_root,
                )
                print(f"imported {result.run_id} -> {result.run_dir}")
    except RunBundleError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
