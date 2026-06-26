# scripts/release_checkpoint_bundle.py
"""Maintainer tool for preparing checkpoint bundles for GitHub Releases."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_fzerox.core.manager.checkpoints import (  # noqa: E402
    CheckpointBundlePackageError,
    CheckpointBundleSourceArtifact,
    CheckpointReleaseError,
    checkpoint_catalog_entry_for_bundle,
    default_checkpoint_catalog_path,
    github_release_upload_command,
    make_github_release_asset_url,
    package_checkpoint_bundle,
    package_evaluation_checkpoint_bundle,
    write_checkpoint_catalog_entry,
)
from rl_fzerox.core.manager.store import ManagerStore, default_manager_db_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.dry_run and args.bundle is None:
            raise CheckpointReleaseError(
                "--dry-run requires --bundle because packaging writes a ZIP"
            )
        bundle_path = _bundle_path(args)
        url = args.url or make_github_release_asset_url(
            repo=args.repo,
            release_tag=args.release_tag,
            filename=bundle_path.name,
        )
        entry = checkpoint_catalog_entry_for_bundle(bundle_path=bundle_path, url=url)
        upload_command = github_release_upload_command(
            repo=args.repo,
            release_tag=args.release_tag,
            bundle_path=bundle_path,
            clobber=args.clobber,
        )
        if args.upload and not args.dry_run:
            subprocess.run(upload_command, check=True)
        if not args.dry_run:
            result = write_checkpoint_catalog_entry(
                catalog_path=args.catalog,
                entry=entry,
                updated_at=args.updated_at,
            )
            catalog_action = "replaced" if result.replaced else "added"
        else:
            catalog_action = "would update"
    except (CheckpointBundlePackageError, CheckpointReleaseError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"GitHub release upload failed with code {exc.returncode}") from exc

    print(f"bundle: {bundle_path}")
    print(f"sha256: {entry.bundle.sha256}")
    print(f"size_bytes: {entry.bundle.size_bytes}")
    print(f"catalog: {catalog_action} {entry.id} {entry.version} in {args.catalog}")
    print(f"upload: {shlex.join(upload_command)}")
    return 0


def _bundle_path(args: argparse.Namespace) -> Path:
    if args.bundle is not None:
        return args.bundle.expanduser().resolve()
    if args.version is None:
        raise CheckpointReleaseError("--version is required when packaging from a run/evaluation")

    store = ManagerStore(args.db)
    if args.evaluation_id is not None:
        return package_evaluation_checkpoint_bundle(
            store=store,
            evaluation_id=args.evaluation_id,
            version=args.version,
            checkpoint_id=args.checkpoint_id,
            checkpoint_name=args.name,
            output_path=args.output,
            overwrite=args.overwrite,
        ).bundle_path
    if args.run_id is None:
        raise CheckpointReleaseError("one of --bundle, --run-id, or --evaluation-id is required")
    return package_checkpoint_bundle(
        store=store,
        run_id=args.run_id,
        artifact=_artifact(args.artifact),
        version=args.version,
        checkpoint_id=args.checkpoint_id,
        checkpoint_name=args.name,
        output_path=args.output,
        allow_running=args.allow_running,
        overwrite=args.overwrite,
    ).bundle_path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package or validate one checkpoint bundle and update the release catalog."
    )
    parser.add_argument("--db", type=Path, default=default_manager_db_path())
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--run-id")
    source.add_argument("--evaluation-id")
    parser.add_argument("--artifact", choices=("latest", "best", "final"), default="best")
    parser.add_argument("--version")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-running", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--catalog", type=Path, default=default_checkpoint_catalog_path())
    parser.add_argument("--repo", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--url")
    parser.add_argument("--updated-at")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--clobber", action="store_true")
    return parser.parse_args(argv)


def _artifact(value: str) -> CheckpointBundleSourceArtifact:
    match value:
        case "latest":
            return "latest"
        case "best":
            return "best"
        case "final":
            return "final"
        case _:
            raise CheckpointReleaseError(f"unsupported artifact: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
