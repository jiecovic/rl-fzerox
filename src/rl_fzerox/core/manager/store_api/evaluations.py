# src/rl_fzerox/core/manager/store_api/evaluations.py
"""Evaluation methods mixed into the public ManagerStore facade."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from rl_fzerox.core.evaluation.models import (
    EvaluationCheckpointArtifact,
    EvaluationPolicyMode,
    EvaluationTargetSpec,
)
from rl_fzerox.core.manager.models import (
    ManagedEvaluation,
    ManagedEvaluationBaselineSuite,
    ManagedEvaluationPreset,
    PolicySourceKind,
)
from rl_fzerox.core.manager.registry import evaluations as evaluation_registry
from rl_fzerox.core.manager.store_api.common import manager_store as _manager_store


class EvaluationStoreMixin:
    """ManagerStore facade methods for managed evaluation records."""

    def create_evaluation(
        self,
        *,
        name: str,
        source_run_id: str,
        source_artifact: EvaluationCheckpointArtifact,
        policy_mode: EvaluationPolicyMode,
        preset_id: str,
        evaluations_root: Path | None = None,
    ) -> ManagedEvaluation:
        return evaluation_registry.create_evaluation(
            _manager_store(self),
            name=name,
            source_run_id=source_run_id,
            source_artifact=source_artifact,
            policy_mode=policy_mode,
            preset_id=preset_id,
            evaluations_root=evaluations_root,
        )

    def create_evaluation_from_policy_source(
        self,
        *,
        name: str,
        source_policy_kind: PolicySourceKind,
        source_policy_id: str,
        source_artifact: EvaluationCheckpointArtifact,
        policy_mode: EvaluationPolicyMode,
        preset_id: str,
        evaluations_root: Path | None = None,
    ) -> ManagedEvaluation:
        return evaluation_registry.create_evaluation_from_policy_source(
            _manager_store(self),
            name=name,
            source_policy_kind=source_policy_kind,
            source_policy_id=source_policy_id,
            source_artifact=source_artifact,
            policy_mode=policy_mode,
            preset_id=preset_id,
            evaluations_root=evaluations_root,
        )

    def get_evaluation(self, evaluation_id: str) -> ManagedEvaluation | None:
        return evaluation_registry.get_evaluation(_manager_store(self), evaluation_id)

    def list_evaluations(self) -> tuple[ManagedEvaluation, ...]:
        return evaluation_registry.list_evaluations(_manager_store(self))

    def get_evaluation_preset(
        self,
        preset_id: str,
    ) -> ManagedEvaluationPreset | None:
        return evaluation_registry.get_evaluation_preset(_manager_store(self), preset_id)

    def list_evaluation_presets(
        self,
    ) -> tuple[ManagedEvaluationPreset, ...]:
        return evaluation_registry.list_evaluation_presets(_manager_store(self))

    def create_evaluation_preset(
        self,
        *,
        name: str,
        seed: int,
        renderer: Literal["angrylion", "gliden64"],
        target: EvaluationTargetSpec,
    ) -> ManagedEvaluationPreset:
        return evaluation_registry.create_evaluation_preset(
            _manager_store(self),
            name=name,
            seed=seed,
            renderer=renderer,
            target=target,
        )

    def delete_evaluation_preset(self, preset_id: str) -> bool:
        return evaluation_registry.delete_evaluation_preset(_manager_store(self), preset_id)

    def list_evaluation_baseline_suites(
        self,
    ) -> tuple[ManagedEvaluationBaselineSuite, ...]:
        return evaluation_registry.list_evaluation_baseline_suites(_manager_store(self))

    def delete_evaluation(self, evaluation_id: str) -> bool:
        return evaluation_registry.delete_evaluation(_manager_store(self), evaluation_id)

    def update_evaluation_name(
        self,
        *,
        evaluation_id: str,
        name: str,
    ) -> ManagedEvaluation | None:
        return evaluation_registry.update_evaluation_name(
            _manager_store(self),
            evaluation_id=evaluation_id,
            name=name,
        )

    def mark_evaluation_running(
        self,
        evaluation_id: str,
    ) -> ManagedEvaluation:
        return evaluation_registry.mark_evaluation_running(_manager_store(self), evaluation_id)

    def mark_evaluation_completed(
        self,
        evaluation_id: str,
    ) -> ManagedEvaluation:
        return evaluation_registry.mark_evaluation_completed(_manager_store(self), evaluation_id)

    def mark_evaluation_failed(
        self,
        evaluation_id: str,
        *,
        error_message: str,
    ) -> ManagedEvaluation:
        return evaluation_registry.mark_evaluation_failed(
            _manager_store(self),
            evaluation_id,
            error_message=error_message,
        )

    def request_evaluation_cancel(
        self,
        evaluation_id: str,
    ) -> ManagedEvaluation | None:
        return evaluation_registry.request_evaluation_cancel(_manager_store(self), evaluation_id)

    def mark_evaluation_cancelled(
        self,
        evaluation_id: str,
    ) -> ManagedEvaluation:
        return evaluation_registry.mark_evaluation_cancelled(_manager_store(self), evaluation_id)

    def evaluation_cancel_request_path(self, evaluation_id: str) -> Path:
        return evaluation_registry.evaluation_cancel_request_path(
            _manager_store(self), evaluation_id
        )
