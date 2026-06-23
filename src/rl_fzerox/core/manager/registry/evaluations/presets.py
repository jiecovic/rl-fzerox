# src/rl_fzerox/core/manager/registry/evaluations/presets.py
"""Registry operations for evaluation presets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from rl_fzerox.core.evaluation.models import EvaluationTargetSpec
from rl_fzerox.core.manager.db.repositories import evaluations as evaluation_repository
from rl_fzerox.core.manager.db.repositories.filesystem import queue_delete_tree
from rl_fzerox.core.manager.models import ManagedEvaluationPreset
from rl_fzerox.core.manager.registry.common import new_record_id, utc_now
from rl_fzerox.core.manager.registry.evaluations.baselines import baseline_suite_for_preset

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


def get_evaluation_preset(
    store: ManagerStore,
    preset_id: str,
) -> ManagedEvaluationPreset | None:
    """Return one persisted evaluation preset."""

    store.initialize()
    with store._orm_session() as session:
        return evaluation_repository.get_evaluation_preset(session, preset_id)


def list_evaluation_presets(store: ManagerStore) -> tuple[ManagedEvaluationPreset, ...]:
    """Return persisted evaluation presets."""

    store.initialize()
    with store._orm_session() as session:
        return evaluation_repository.list_evaluation_presets(session)


def create_evaluation_preset(
    store: ManagerStore,
    *,
    name: str,
    seed: int,
    renderer: Literal["angrylion", "gliden64"],
    target: EvaluationTargetSpec,
) -> ManagedEvaluationPreset:
    """Create one immutable custom benchmark preset."""

    store.initialize()
    _validate_preset_target(target)
    created_at = utc_now()
    preset = ManagedEvaluationPreset(
        id=new_record_id(name),
        name=name,
        version=1,
        seed=seed,
        renderer=renderer,
        target=target,
        builtin=False,
        created_at=created_at,
        updated_at=created_at,
    )
    with store._orm_session() as session:
        return evaluation_repository.insert_evaluation_preset(session, preset=preset)


def delete_evaluation_preset(store: ManagerStore, preset_id: str) -> bool:
    """Delete one unused custom preset and its materialized baseline suite."""

    store.initialize()
    deleted_at = utc_now()
    with store._orm_session() as session:
        deleted = evaluation_repository.delete_evaluation_preset(session, preset_id)
        if deleted is None:
            return False
        preset, suites = deleted
        if suites:
            for suite in suites:
                queue_delete_tree(session, path=suite.suite_dir, created_at=deleted_at)
        else:
            suite = baseline_suite_for_preset(
                store,
                preset_id=preset.id,
                preset_version=preset.version,
                created_at=None,
            )
            queue_delete_tree(session, path=suite.suite_dir, created_at=deleted_at)
    store._drain_pending_filesystem_operations()
    return True


def _validate_preset_target(target: EvaluationTargetSpec) -> None:
    if target.mode == "gp_course" and len(target.difficulties) != 1:
        raise ValueError("gp_course evaluation presets require exactly one difficulty")
    if target.mode == "time_attack_course" and target.difficulties:
        raise ValueError("time_attack_course evaluation presets must not set difficulties")
