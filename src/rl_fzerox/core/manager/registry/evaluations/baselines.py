# src/rl_fzerox/core/manager/registry/evaluations/baselines.py
"""Registry operations for evaluation baseline suites."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rl_fzerox.core.manager.db.repositories import evaluations as evaluation_repository
from rl_fzerox.core.manager.models import ManagedEvaluation, ManagedEvaluationBaselineSuite
from rl_fzerox.core.manager.registry.common import utc_now
from rl_fzerox.core.training.runs import RUN_LAYOUT

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def list_evaluation_baseline_suites(
    store: ManagerStore,
) -> tuple[ManagedEvaluationBaselineSuite, ...]:
    """Return materialization status for each preset without creating DB rows."""

    store.initialize()
    now = utc_now()
    with store._orm_session() as session:
        presets = evaluation_repository.list_evaluation_presets(session)
        persisted_suites = {
            (suite.preset_id, suite.preset_version): suite
            for suite in evaluation_repository.list_evaluation_baseline_suites(session)
        }
    return tuple(
        _with_filesystem_suite_status(
            persisted_suites.get(
                (preset.id, preset.version),
                _baseline_suite_for_preset(
                    store,
                    preset_id=preset.id,
                    preset_version=preset.version,
                    created_at=None,
                ),
            ),
            updated_at=now,
        )
        for preset in presets
    )


def baseline_suite_for_evaluation(evaluation: ManagedEvaluation) -> ManagedEvaluationBaselineSuite:
    """Return the suite row implied by one preset-backed evaluation."""

    return _with_filesystem_suite_status(
        _baseline_suite_for_preset(
            None,
            preset_id=evaluation.preset_id,
            preset_version=evaluation.preset_version,
            created_at=None,
            evaluations_root=evaluation.evaluation_dir.parent,
        )
    )


def baseline_suite_for_preset(
    store: ManagerStore | None,
    *,
    preset_id: str,
    preset_version: int,
    created_at: str | None,
    evaluations_root: Path | None = None,
) -> ManagedEvaluationBaselineSuite:
    """Return the baseline-suite record implied by one preset version."""

    return _baseline_suite_for_preset(
        store,
        preset_id=preset_id,
        preset_version=preset_version,
        created_at=created_at,
        evaluations_root=evaluations_root,
    )


def _baseline_suite_for_preset(
    store: ManagerStore | None,
    *,
    preset_id: str,
    preset_version: int,
    created_at: str | None,
    evaluations_root: Path | None = None,
) -> ManagedEvaluationBaselineSuite:
    if evaluations_root is None:
        if store is None:
            raise ValueError("store or evaluations_root is required")
        root = store.evaluations_root()
    else:
        root = evaluations_root
    suite_id = f"{preset_id}-v{preset_version}"
    suite_dir = root / "_baseline_suites" / suite_id
    return ManagedEvaluationBaselineSuite(
        id=suite_id,
        preset_id=preset_id,
        preset_version=preset_version,
        status="not_created",
        suite_dir=suite_dir,
        manifest_path=suite_dir / RUN_LAYOUT.config_filename,
        created_at=created_at,
        updated_at=created_at,
    )


def _with_filesystem_suite_status(
    suite: ManagedEvaluationBaselineSuite,
    *,
    updated_at: str | None = None,
) -> ManagedEvaluationBaselineSuite:
    if suite.error_message is not None:
        status = "failed"
    elif suite.manifest_path is not None and suite.manifest_path.is_file():
        status = "ready"
    else:
        status = "not_created"
    materialized_at = suite.materialized_at
    if status == "ready" and materialized_at is None:
        materialized_at = updated_at
    return ManagedEvaluationBaselineSuite(
        id=suite.id,
        preset_id=suite.preset_id,
        preset_version=suite.preset_version,
        status=status,
        suite_dir=suite.suite_dir,
        manifest_path=suite.manifest_path,
        error_message=suite.error_message,
        created_at=suite.created_at,
        updated_at=updated_at or suite.updated_at,
        materialized_at=materialized_at,
    )
