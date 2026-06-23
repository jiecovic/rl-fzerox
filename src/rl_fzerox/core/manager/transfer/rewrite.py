# src/rl_fzerox/core/manager/transfer/rewrite.py
"""Rewrite source-local paths after importing a run bundle."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from rl_fzerox.core.manager.transfer.files import run_files
from rl_fzerox.core.manager.transfer.models import RunBundleManifest
from rl_fzerox.core.runtime_spec.paths import project_root_dir


def path_replacements(
    *,
    manifest: RunBundleManifest,
    target_run_dir: Path,
    target_runs_root: Path,
) -> tuple[tuple[str, str], ...]:
    """Return longest-first path replacements from source bundle to local import."""

    source_run_dir = Path(manifest.run.run_dir)
    source_lineage_dir = source_run_dir.parent
    source_runs_root = source_lineage_dir.parent
    replacements = {
        str(source_run_dir): str(target_run_dir),
        str(source_lineage_dir): str(target_run_dir.parent),
        str(source_runs_root): str(target_runs_root),
        manifest.project_root: str(project_root_dir()),
    }
    return tuple(
        sorted(
            (
                (source, target)
                for source, target in replacements.items()
                if source and source != target
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )


def rewrite_imported_manifest_paths(
    run_dir: Path,
    *,
    replacements: tuple[tuple[str, str], ...],
) -> None:
    """Rewrite paths in imported text manifests and JSON artifacts."""

    for file_path in _rewrite_candidate_files(run_dir):
        if file_path.suffix.lower() == ".json" and _rewrite_json_paths(
            file_path,
            replacements=replacements,
        ):
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rewritten = rewrite_path_text(text, replacements=replacements)
        if rewritten != text:
            file_path.write_text(rewritten, encoding="utf-8")


def rewritten_optional_path(
    value: str | None,
    *,
    replacements: tuple[tuple[str, str], ...],
) -> str | None:
    """Rewrite one optional path string and keep it only when the target exists."""

    if value is None:
        return None
    rewritten = rewrite_path_text(value, replacements=replacements)
    return rewritten if Path(rewritten).exists() else None


def rewrite_path_text(text: str, *, replacements: tuple[tuple[str, str], ...]) -> str:
    """Apply ordered literal path replacements to one text payload."""

    rewritten = text
    for source, target in replacements:
        rewritten = rewritten.replace(source, target)
    return rewritten


def _rewrite_candidate_files(run_dir: Path) -> Iterable[Path]:
    suffixes = {".json", ".yaml", ".yml", ".txt", ".toml"}
    for file_path in run_files(run_dir):
        if file_path.suffix.lower() in suffixes:
            yield file_path


def _rewrite_json_paths(
    file_path: Path,
    *,
    replacements: tuple[tuple[str, str], ...],
) -> bool:
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False

    rewritten, changed = _rewrite_json_value(value, replacements=replacements)
    if changed:
        file_path.write_text(
            json.dumps(rewritten, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return True


def _rewrite_json_value(
    value: object,
    *,
    replacements: tuple[tuple[str, str], ...],
) -> tuple[object, bool]:
    if isinstance(value, str):
        rewritten = rewrite_path_text(value, replacements=replacements)
        return rewritten, rewritten != value
    if isinstance(value, list):
        changed = False
        rewritten_items: list[object] = []
        for item in value:
            rewritten_item, item_changed = _rewrite_json_value(
                item,
                replacements=replacements,
            )
            rewritten_items.append(rewritten_item)
            changed = changed or item_changed
        return rewritten_items, changed
    if isinstance(value, dict):
        changed = False
        rewritten_dict: dict[object, object] = {}
        for key, item in value.items():
            rewritten_item, item_changed = _rewrite_json_value(
                item,
                replacements=replacements,
            )
            rewritten_dict[key] = rewritten_item
            changed = changed or item_changed
        return rewritten_dict, changed
    return value, False
