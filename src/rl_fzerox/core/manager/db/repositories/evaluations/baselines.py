# src/rl_fzerox/core/manager/db/repositories/evaluations/baselines.py
"""Repository operations for evaluation baseline-suite rows."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from rl_fzerox.core.manager.db.models.evaluations import EvaluationBaselineSuiteModel
from rl_fzerox.core.manager.db.repositories.evaluations.mapping import (
    evaluation_baseline_suite_from_model,
)
from rl_fzerox.core.manager.models import ManagedEvaluationBaselineSuite


def ensure_evaluation_baseline_suite(
    session: Session,
    *,
    suite: ManagedEvaluationBaselineSuite,
) -> ManagedEvaluationBaselineSuite:
    """Ensure one preset-version baseline-suite row exists."""

    existing = session.get(EvaluationBaselineSuiteModel, suite.id)
    if existing is None:
        session.add(
            EvaluationBaselineSuiteModel(
                id=suite.id,
                preset_id=suite.preset_id,
                preset_version=suite.preset_version,
                status=suite.status,
                suite_dir=str(suite.suite_dir),
                manifest_path=None if suite.manifest_path is None else str(suite.manifest_path),
                error_message=suite.error_message,
                created_at=suite.created_at or "",
                updated_at=suite.updated_at or "",
                materialized_at=suite.materialized_at,
            )
        )
        session.flush()
        return suite
    return evaluation_baseline_suite_from_model(existing)


def upsert_evaluation_baseline_suite_status(
    session: Session,
    *,
    suite: ManagedEvaluationBaselineSuite,
) -> ManagedEvaluationBaselineSuite:
    """Persist the latest known filesystem status for one baseline suite."""

    existing = session.get(EvaluationBaselineSuiteModel, suite.id)
    if existing is None:
        return ensure_evaluation_baseline_suite(session, suite=suite)
    existing.status = suite.status
    existing.suite_dir = str(suite.suite_dir)
    existing.manifest_path = None if suite.manifest_path is None else str(suite.manifest_path)
    existing.error_message = suite.error_message
    existing.updated_at = suite.updated_at or existing.updated_at
    existing.materialized_at = suite.materialized_at
    session.flush()
    return evaluation_baseline_suite_from_model(existing)


def list_evaluation_baseline_suites(
    session: Session,
) -> tuple[ManagedEvaluationBaselineSuite, ...]:
    """Return known evaluation baseline suites."""

    suites = tuple(
        session.scalars(
            select(EvaluationBaselineSuiteModel).order_by(
                EvaluationBaselineSuiteModel.preset_id.asc(),
                EvaluationBaselineSuiteModel.preset_version.asc(),
            )
        )
    )
    return tuple(evaluation_baseline_suite_from_model(suite) for suite in suites)
