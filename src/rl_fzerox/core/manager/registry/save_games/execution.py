# src/rl_fzerox/core/manager/registry/save_games/execution.py
"""Resolve save-attempt targets and runner execution contexts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rl_fzerox.core.career_mode.course_setup import (
    CourseSetupTarget,
    missing_course_setup_targets,
    required_course_setup_targets,
    resolve_course_setup,
    resolve_cup_setup,
)
from rl_fzerox.core.career_mode.execution.context import SaveAttemptExecutionContext
from rl_fzerox.core.career_mode.progress.unlocks import (
    UnlockRuleTarget,
    build_unlock_progress,
    default_unlock_targets,
)
from rl_fzerox.core.manager.db.repositories import runs as run_repository
from rl_fzerox.core.manager.db.repositories import save_games as save_game_repository
from rl_fzerox.core.manager.models import (
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
    ManagedSaveCupSetup,
    ManagedSaveUnlockProgress,
    ManagedSaveUnlockTarget,
)
from rl_fzerox.core.training.runs import resolve_policy_artifact_path

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from rl_fzerox.core.manager.store import ManagerStore


def get_save_attempt_execution_context(
    store: ManagerStore,
    attempt_id: str,
) -> SaveAttemptExecutionContext | None:
    """Resolve one running attempt into the inputs required by the runner."""

    store.initialize()
    with store._orm_session() as session:
        attempt = save_game_repository.get_save_attempt(session, attempt_id)
        if attempt is None:
            return None
        if attempt.status != "running":
            raise ValueError("save attempt is not running")
        if attempt.target_kind is None:
            raise ValueError("save attempt is missing an unlock target")
        save_game = save_game_repository.get_save_game(session, attempt.save_game_id)
        if save_game is None:
            raise KeyError("save game not found")
        target = _target_for_attempt(attempt)

        course_setups = save_game_repository.list_course_setups(session, attempt.save_game_id)
        course_setup_target = CourseSetupTarget(
            difficulty=attempt.difficulty,
            cup_id=attempt.cup_id,
            course_id=attempt.course_id,
        )
        attempts = save_game_repository.list_save_attempts(session, attempt.save_game_id)
        progress = build_unlock_progress(save_game.save_path, attempts=attempts)
        progress_target = find_unlock_progress_target(
            progress,
            target_kind=target.kind,
            difficulty=target.difficulty,
            cup_id=target.cup_id,
            course_id=target.course_id,
        )
        if progress_target is None:
            raise ValueError("save attempt target is not part of the unlock path")
        cup_setups = save_game_repository.list_cup_setups(session, attempt.save_game_id)
        cup_setup = _resolve_cup_setup_or_default(
            cup_setups,
            course_setup_target,
            progress=progress,
            save_game_id=save_game.id,
            timestamp=save_game.updated_at,
        )
        if cup_setup is None:
            raise ValueError("save attempt is missing a resolved cup vehicle setup")
        missing_targets = missing_course_setup_targets(
            course_setups,
            course_setup_target,
        )
        if missing_targets:
            missing_labels = ", ".join(_course_setup_target_label(item) for item in missing_targets)
            raise ValueError(
                f"save attempt is missing a resolved course setup for {missing_labels}"
            )
        resolved_targets = required_course_setup_targets(course_setup_target)
        course_setup_target = resolved_targets[0]
        course_setup = resolve_course_setup(
            course_setups,
            course_setup_target,
        )
        if course_setup is None:
            raise ValueError("save attempt is missing a resolved course setup")
        if not save_game_repository.run_exists(session, course_setup.policy_run_id):
            raise KeyError("policy run not found")
        policy_run = run_repository.get_managed_run(session, course_setup.policy_run_id)
        if policy_run is None:
            raise KeyError("policy run not found")
        return SaveAttemptExecutionContext(
            save_game=save_game,
            attempt=attempt,
            target=progress_target,
            course_setup_target=course_setup_target,
            course_setup=course_setup,
            cup_setup=cup_setup,
            policy_run=policy_run,
            policy_artifact=course_setup.policy_artifact,
            policy_path=resolve_policy_artifact_path(
                policy_run.run_dir,
                artifact=course_setup.policy_artifact,
            ),
        )


def validate_policy_attempt_setup(
    session: Session,
    *,
    save_game_id: str,
    progress: ManagedSaveUnlockProgress,
    target: ManagedSaveUnlockTarget,
    timestamp: str,
    error_subject: str,
) -> None:
    """Validate that one policy-backed attempt can be launched."""

    course_setups = save_game_repository.list_course_setups(session, save_game_id)
    cup_setups = save_game_repository.list_cup_setups(session, save_game_id)
    course_setup_target = CourseSetupTarget(
        difficulty=target.difficulty,
        cup_id=target.cup_id,
        course_id=target.course_id,
    )
    cup_setup = _resolve_cup_setup_or_default(
        cup_setups,
        course_setup_target,
        progress=progress,
        save_game_id=save_game_id,
        timestamp=timestamp,
    )
    if cup_setup is None:
        raise ValueError(f"{error_subject} has no matching cup vehicle setup")
    missing_targets = missing_course_setup_targets(course_setups, course_setup_target)
    if missing_targets:
        missing_labels = ", ".join(_course_setup_target_label(item) for item in missing_targets)
        raise ValueError(f"{error_subject} has no matching course setup for {missing_labels}")
    resolved_targets = required_course_setup_targets(course_setup_target)
    course_setup_target = resolved_targets[0]
    course_setup = resolve_course_setup(course_setups, course_setup_target)
    if course_setup is None:
        raise ValueError(f"{error_subject} has no matching course setup")
    _validate_required_course_setup_runs(
        session,
        course_setups=course_setups,
        targets=resolved_targets,
    )


def find_unlock_progress_target(
    progress: ManagedSaveUnlockProgress,
    *,
    target_kind: str,
    difficulty: str,
    cup_id: str,
    course_id: str | None,
) -> ManagedSaveUnlockTarget | None:
    return next(
        (
            target
            for target in progress.targets
            if target.kind == target_kind
            and target.difficulty == difficulty
            and target.cup_id == cup_id
            and target.course_id == course_id
        ),
        None,
    )


def _resolve_cup_setup_or_default(
    cup_setups: tuple[ManagedSaveCupSetup, ...],
    target: CourseSetupTarget,
    *,
    progress: ManagedSaveUnlockProgress,
    save_game_id: str,
    timestamp: str,
) -> ManagedSaveCupSetup | None:
    cup_setup = resolve_cup_setup(cup_setups, target)
    if cup_setup is not None or target.cup_id is None:
        return cup_setup
    default_vehicle_id = progress.unlocked_vehicle_ids[0] if progress.unlocked_vehicle_ids else None
    if default_vehicle_id is None:
        return None
    return ManagedSaveCupSetup(
        id=f"{save_game_id}:default-cup-setup:{target.difficulty or 'all'}:{target.cup_id}",
        save_game_id=save_game_id,
        cup_id=target.cup_id,
        vehicle_id=default_vehicle_id,
        created_at=timestamp,
        updated_at=timestamp,
        difficulty=target.difficulty,
    )


def _target_for_attempt(attempt: ManagedSaveAttempt) -> UnlockRuleTarget:
    for target in default_unlock_targets():
        if (
            target.kind == attempt.target_kind
            and target.difficulty == attempt.difficulty
            and target.cup_id == attempt.cup_id
            and target.course_id == attempt.course_id
        ):
            return target
    raise ValueError("save attempt target is not part of the unlock path")


def _validate_required_course_setup_runs(
    session: Session,
    *,
    course_setups: tuple[ManagedSaveCourseSetup, ...],
    targets: tuple[CourseSetupTarget, ...],
) -> None:
    for target in targets:
        course_setup = resolve_course_setup(course_setups, target)
        if course_setup is None:
            raise ValueError(f"save attempt has no matching course setup for {target.course_id}")
        if not save_game_repository.run_exists(session, course_setup.policy_run_id):
            raise KeyError("policy run not found")


def _course_setup_target_label(target: CourseSetupTarget) -> str:
    parts = [
        target.difficulty or "*",
        target.cup_id or "*",
        target.course_id or "*",
    ]
    return "/".join(parts)
