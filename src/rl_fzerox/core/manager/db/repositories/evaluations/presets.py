# src/rl_fzerox/core/manager/db/repositories/evaluations/presets.py
"""Repository operations for evaluation preset rows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
from rl_fzerox.core.evaluation.models import EvaluationTargetSpec
from rl_fzerox.core.manager.db.models.evaluations import (
    EvaluationBaselineSuiteModel,
    EvaluationModel,
    EvaluationPresetModel,
)
from rl_fzerox.core.manager.db.repositories.evaluations.mapping import (
    evaluation_baseline_suite_from_model,
    evaluation_preset_from_model,
)
from rl_fzerox.core.manager.models import (
    ManagedEvaluationBaselineSuite,
    ManagedEvaluationPreset,
)


@dataclass(frozen=True)
class _DefaultPresetSettings:
    seed: int
    repeats_per_target: int
    gp_baseline_variant_count: int


_DEFAULT_PRESET_SETTINGS = _DefaultPresetSettings(
    seed=2_262_218_583,
    repeats_per_target=10,
    gp_baseline_variant_count=10,
)


def upsert_default_evaluation_presets(session: Session, *, now: str) -> None:
    """Create or refresh repo-owned built-in evaluation presets."""

    for preset in _default_evaluation_presets(now=now):
        existing = session.get(EvaluationPresetModel, preset.id)
        if existing is None:
            session.add(_preset_model_from_dataclass(preset))
        elif not existing.builtin:
            continue
        else:
            _sync_builtin_preset_model(existing, preset=preset, updated_at=now)


def list_evaluation_presets(session: Session) -> tuple[ManagedEvaluationPreset, ...]:
    """Return evaluation presets in manager display order."""

    presets = tuple(
        session.scalars(
            select(EvaluationPresetModel).order_by(
                EvaluationPresetModel.builtin.desc(),
                EvaluationPresetModel.name.asc(),
                EvaluationPresetModel.id.asc(),
            )
        )
    )
    return tuple(evaluation_preset_from_model(preset) for preset in presets)


def get_evaluation_preset(
    session: Session,
    preset_id: str,
) -> ManagedEvaluationPreset | None:
    """Return one evaluation preset by stable id."""

    preset = session.get(EvaluationPresetModel, preset_id)
    return None if preset is None else evaluation_preset_from_model(preset)


def insert_evaluation_preset(
    session: Session,
    *,
    preset: ManagedEvaluationPreset,
) -> ManagedEvaluationPreset:
    """Insert one custom evaluation preset."""

    session.add(_preset_model_from_dataclass(preset))
    session.flush()
    return preset


def delete_evaluation_preset(
    session: Session,
    preset_id: str,
) -> tuple[ManagedEvaluationPreset, tuple[ManagedEvaluationBaselineSuite, ...]] | None:
    """Delete one unused custom preset and return its baseline-suite rows."""

    preset = session.get(EvaluationPresetModel, preset_id)
    if preset is None:
        return None
    managed = evaluation_preset_from_model(preset)
    if managed.builtin:
        raise ValueError("built-in evaluation presets cannot be deleted")
    referenced_evaluation_id = session.scalar(
        select(EvaluationModel.id).where(EvaluationModel.preset_id == preset_id).limit(1)
    )
    if referenced_evaluation_id is not None:
        raise ValueError("evaluation preset is referenced by evaluation records")
    suites = tuple(
        session.scalars(
            select(EvaluationBaselineSuiteModel).where(
                EvaluationBaselineSuiteModel.preset_id == preset_id
            )
        )
    )
    managed_suites = tuple(evaluation_baseline_suite_from_model(suite) for suite in suites)
    for suite in suites:
        session.delete(suite)
    session.delete(preset)
    return managed, managed_suites


def _preset_model_from_dataclass(preset: ManagedEvaluationPreset) -> EvaluationPresetModel:
    return EvaluationPresetModel(
        id=preset.id,
        name=preset.name,
        version=preset.version,
        seed=preset.seed,
        renderer=preset.renderer,
        target_json=json.dumps(asdict(preset.target), sort_keys=True),
        builtin=preset.builtin,
        created_at=preset.created_at,
        updated_at=preset.updated_at,
    )


def _sync_builtin_preset_model(
    model: EvaluationPresetModel,
    *,
    preset: ManagedEvaluationPreset,
    updated_at: str,
) -> None:
    target_json = json.dumps(asdict(preset.target), sort_keys=True)
    changed = (
        model.name != preset.name
        or model.version != preset.version
        or model.seed != preset.seed
        or model.renderer != preset.renderer
        or model.target_json != target_json
    )
    if not changed:
        return
    model.name = preset.name
    model.version = preset.version
    model.seed = preset.seed
    model.renderer = preset.renderer
    model.target_json = target_json
    model.updated_at = updated_at


def _default_evaluation_presets(*, now: str) -> tuple[ManagedEvaluationPreset, ...]:
    all_course_ids = tuple(course.id for course in BUILT_IN_COURSES)
    return (
        ManagedEvaluationPreset(
            id="time_attack_all_courses",
            name="Time Attack course · all courses",
            version=1,
            seed=_DEFAULT_PRESET_SETTINGS.seed,
            renderer="gliden64",
            target=EvaluationTargetSpec(
                mode="time_attack_course",
                course_ids=all_course_ids,
                repeats_per_target=_DEFAULT_PRESET_SETTINGS.repeats_per_target,
            ),
            builtin=True,
            created_at=now,
            updated_at=now,
        ),
        ManagedEvaluationPreset(
            id="gp_course_master_all_courses",
            name="GP course · Master · all courses",
            version=2,
            seed=_DEFAULT_PRESET_SETTINGS.seed,
            renderer="gliden64",
            target=EvaluationTargetSpec(
                mode="gp_course",
                course_ids=all_course_ids,
                difficulties=("master",),
                repeats_per_target=_DEFAULT_PRESET_SETTINGS.repeats_per_target,
                baseline_variant_count=_DEFAULT_PRESET_SETTINGS.gp_baseline_variant_count,
            ),
            builtin=True,
            created_at=now,
            updated_at=now,
        ),
    )
