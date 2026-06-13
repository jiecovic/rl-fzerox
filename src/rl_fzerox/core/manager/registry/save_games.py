# src/rl_fzerox/core/manager/registry/save_games.py
"""Manager-store operations for portable save games."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from rl_fzerox.core.career_mode.course_setup import (
    CourseSetupTarget,
    missing_course_setup_targets,
    required_course_setup_targets,
    resolve_course_setup,
    resolve_cup_setup,
)
from rl_fzerox.core.career_mode.progress import (
    UnlockRuleTarget,
    build_unlock_progress,
    default_unlock_targets,
)
from rl_fzerox.core.career_mode.runner.context import SaveAttemptExecutionContext
from rl_fzerox.core.manager.artifacts.paths import predicted_managed_save_game_path
from rl_fzerox.core.manager.db.repositories import runs as run_repository
from rl_fzerox.core.manager.db.repositories import save_games as save_game_repository
from rl_fzerox.core.manager.db.repositories.filesystem import queue_delete_tree
from rl_fzerox.core.manager.models import (
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
    ManagedSaveCupSetup,
    ManagedSaveGame,
    ManagedSaveUnlockProgress,
    ManagedSaveUnlockTarget,
    SaveAttemptStatus,
    SaveGameStatus,
)
from rl_fzerox.core.manager.registry.common import new_record_id, utc_now
from rl_fzerox.core.runtime_spec.vehicle_catalog import vehicle_by_id
from rl_fzerox.core.training.runs import resolve_policy_artifact_path

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from rl_fzerox.core.manager.store import ManagerStore


def create_save_game(
    store: ManagerStore,
    *,
    name: str,
    save_games_root: Path | None = None,
) -> ManagedSaveGame:
    """Create one manager-owned save-game record."""

    if not name.strip():
        raise ValueError("save-game name is required")
    store.initialize()
    created_at = utc_now()
    save_game_id = new_record_id(name)
    normalized_name = name.strip()
    save_path = predicted_managed_save_game_path(save_game_id, output_root=save_games_root)
    save_game = ManagedSaveGame(
        id=save_game_id,
        name=normalized_name,
        status="created",
        save_path=save_path,
        created_at=created_at,
        updated_at=created_at,
    )
    with store._orm_session() as session:
        save_game_repository.assert_save_game_name_available(session, normalized_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_game_repository.insert_save_game(session, save_game)
    return save_game


def get_save_game(store: ManagerStore, save_game_id: str) -> ManagedSaveGame | None:
    """Return one save game by id."""

    store.initialize()
    with store._orm_session() as session:
        return save_game_repository.get_save_game(session, save_game_id)


def list_save_games(store: ManagerStore) -> tuple[ManagedSaveGame, ...]:
    """Return all save games in manager display order."""

    store.initialize()
    with store._orm_session() as session:
        return save_game_repository.list_save_games(session)


def rename_save_game(
    store: ManagerStore,
    *,
    save_game_id: str,
    name: str,
) -> ManagedSaveGame | None:
    """Rename one manager-owned save game."""

    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("save-game name is required")
    store.initialize()
    with store._orm_session() as session:
        return save_game_repository.rename_save_game(
            session,
            save_game_id=save_game_id,
            name=normalized_name,
            updated_at=utc_now(),
        )


def delete_save_game(store: ManagerStore, save_game_id: str) -> bool:
    """Delete one manager-owned save game and queue its filesystem cleanup."""

    store.initialize()
    deleted_at = utc_now()
    with store._orm_session() as session:
        save_game = save_game_repository.get_save_game(session, save_game_id)
        if save_game is None:
            return False
        if save_game.status == "running":
            raise ValueError("stop or pause the career runner before deleting this save")
        save_root = save_game.save_path.parent
        if save_root.exists():
            queue_delete_tree(session, path=save_root, created_at=deleted_at)
        save_game_repository.delete_save_game(session, save_game_id)
    store._drain_pending_filesystem_operations()
    return True


def unlock_progress(
    store: ManagerStore,
    save_game_id: str,
) -> ManagedSaveUnlockProgress:
    """Return save-file unlock progress for one save game."""

    store.initialize()
    with store._orm_session() as session:
        save_game = save_game_repository.get_save_game(session, save_game_id)
        if save_game is None:
            raise KeyError("save game not found")
        attempts = save_game_repository.list_save_attempts(session, save_game_id)
    return build_unlock_progress(save_game.save_path, attempts=attempts)


def list_course_setups(
    store: ManagerStore,
    save_game_id: str,
) -> tuple[ManagedSaveCourseSetup, ...]:
    """Return course setups for one save game."""

    store.initialize()
    with store._orm_session() as session:
        return save_game_repository.list_course_setups(session, save_game_id)


def list_cup_setups(
    store: ManagerStore,
    save_game_id: str,
) -> tuple[ManagedSaveCupSetup, ...]:
    """Return cup vehicle setups for one save game."""

    store.initialize()
    with store._orm_session() as session:
        return save_game_repository.list_cup_setups(session, save_game_id)


def start_save_attempt(
    store: ManagerStore,
    *,
    save_game_id: str,
    target_kind: str | None = None,
    difficulty: str | None = None,
    cup_id: str | None = None,
    course_id: str | None = None,
) -> ManagedSaveAttempt:
    """Record the start of one unlock attempt."""

    store.initialize()
    now = utc_now()
    attempt = ManagedSaveAttempt(
        id=new_record_id(f"{save_game_id} attempt"),
        save_game_id=save_game_id,
        target_kind=target_kind,
        status="running",
        difficulty=difficulty,
        cup_id=cup_id,
        course_id=course_id,
        started_at=now,
    )
    with store._orm_session() as session:
        if save_game_repository.get_save_game(session, save_game_id) is None:
            raise KeyError("save game not found")
        save_game_repository.insert_save_attempt(session, attempt)
        save_game_repository.touch_save_game(
            session,
            save_game_id=save_game_id,
            updated_at=now,
        )
    return attempt


def start_next_save_attempt(
    store: ManagerStore,
    save_game_id: str,
) -> ManagedSaveAttempt:
    """Resolve and record the next policy-backed unlock attempt."""

    store.initialize()
    started_at = utc_now()
    with store._orm_session() as session:
        save_game = save_game_repository.get_save_game(session, save_game_id)
        if save_game is None:
            raise KeyError("save game not found")
        attempts = save_game_repository.list_save_attempts(session, save_game_id)
        if any(attempt.status == "running" for attempt in attempts):
            raise ValueError("save game already has a running attempt")
        progress = build_unlock_progress(save_game.save_path, attempts=attempts)
        if progress.next_target is None:
            raise ValueError("save game has no pending unlock target")
        return _insert_policy_backed_save_attempt(
            session,
            save_game_id=save_game_id,
            progress=progress,
            target=progress.next_target,
            started_at=started_at,
            error_subject="next unlock target",
        )


def start_target_save_attempt(
    store: ManagerStore,
    save_game_id: str,
    *,
    target_kind: str,
    difficulty: str,
    cup_id: str,
    course_id: str | None = None,
) -> ManagedSaveAttempt:
    """Resolve and record one selected policy-backed unlock attempt."""

    store.initialize()
    started_at = utc_now()
    with store._orm_session() as session:
        save_game = save_game_repository.get_save_game(session, save_game_id)
        if save_game is None:
            raise KeyError("save game not found")
        attempts = save_game_repository.list_save_attempts(session, save_game_id)
        if any(attempt.status == "running" for attempt in attempts):
            raise ValueError("save game already has a running attempt")
        progress = build_unlock_progress(save_game.save_path, attempts=attempts)
        target = _progress_target(
            progress,
            target_kind=target_kind,
            difficulty=difficulty,
            cup_id=cup_id,
            course_id=course_id,
        )
        if target is None:
            raise ValueError("selected unlock target is not part of the unlock path")
        if target.status != "pending":
            raise ValueError(f"selected unlock target is {target.status}")
        return _insert_policy_backed_save_attempt(
            session,
            save_game_id=save_game_id,
            progress=progress,
            target=target,
            started_at=started_at,
            error_subject="selected unlock target",
        )


def start_or_reuse_next_save_attempt(
    store: ManagerStore,
    save_game_id: str,
) -> ManagedSaveAttempt:
    """Return the active attempt, or start the next pending unlock target."""

    store.initialize()
    with store._orm_session() as session:
        if save_game_repository.get_save_game(session, save_game_id) is None:
            raise KeyError("save game not found")
        running_attempt = save_game_repository.running_save_attempt(session, save_game_id)
        if running_attempt is not None:
            return running_attempt
    return start_next_save_attempt(store, save_game_id)


def list_save_attempts(
    store: ManagerStore,
    save_game_id: str,
) -> tuple[ManagedSaveAttempt, ...]:
    """Return attempts for one save game."""

    store.initialize()
    with store._orm_session() as session:
        return save_game_repository.list_save_attempts(session, save_game_id)


def fail_running_save_attempts(
    store: ManagerStore,
    *,
    save_game_id: str,
    failure_reason: str,
) -> int:
    """Fail running attempts that are not backed by a live runner process."""

    store.initialize()
    finished_at = utc_now()
    with store._orm_session() as session:
        count = save_game_repository.fail_running_save_attempts(
            session,
            save_game_id=save_game_id,
            finished_at=finished_at,
            failure_reason=failure_reason,
        )
        if count:
            save_game_repository.touch_save_game(
                session,
                save_game_id=save_game_id,
                updated_at=finished_at,
            )
        return count


def discard_running_save_attempts(
    store: ManagerStore,
    *,
    save_game_id: str,
) -> int:
    """Delete attempts that never reached a real race result."""

    store.initialize()
    updated_at = utc_now()
    with store._orm_session() as session:
        count = save_game_repository.delete_running_save_attempts(
            session,
            save_game_id=save_game_id,
        )
        if count:
            save_game_repository.touch_save_game(
                session,
                save_game_id=save_game_id,
                updated_at=updated_at,
            )
        return count


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
            target=target.to_progress_target(status="pending"),
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


def finish_save_attempt(
    store: ManagerStore,
    *,
    attempt_id: str,
    status: SaveAttemptStatus,
    finish_position: int | None = None,
    finish_time_s: float | None = None,
    failure_reason: str | None = None,
) -> ManagedSaveAttempt | None:
    """Mark one unlock attempt as succeeded or failed."""

    if status == "running":
        raise ValueError("finished attempts cannot keep running status")
    store.initialize()
    finished_at = utc_now()
    with store._orm_session() as session:
        attempt = save_game_repository.finish_save_attempt(
            session,
            attempt_id=attempt_id,
            status=status,
            finished_at=finished_at,
            finish_position=finish_position,
            finish_time_s=finish_time_s,
            failure_reason=failure_reason,
        )
        if attempt is not None:
            save_game_repository.touch_save_game(
                session,
                save_game_id=attempt.save_game_id,
                updated_at=finished_at,
            )
        return attempt


def upsert_course_setup(
    store: ManagerStore,
    *,
    save_game_id: str,
    policy_run_id: str,
    policy_artifact: Literal["latest", "best"],
    engine_setting_raw_value: int = 50,
    difficulty: str | None = None,
    cup_id: str | None = None,
    course_id: str | None = None,
) -> ManagedSaveCourseSetup:
    """Create or update one save-game course setup."""

    _validate_course_setup_target(course_id=course_id)
    _validate_engine_setting(engine_setting_raw_value=engine_setting_raw_value)
    store.initialize()
    now = utc_now()
    with store._orm_session() as session:
        save_game = save_game_repository.get_save_game(session, save_game_id)
        if save_game is None:
            raise KeyError("save game not found")
        if not save_game_repository.run_exists(session, policy_run_id):
            raise KeyError("policy run not found")
        course_setup = save_game_repository.upsert_course_setup(
            session,
            setup_id=new_record_id(f"{save_game_id} setup"),
            save_game_id=save_game_id,
            policy_run_id=policy_run_id,
            policy_artifact=policy_artifact,
            engine_setting_raw_value=engine_setting_raw_value,
            created_at=now,
            updated_at=now,
            difficulty=difficulty,
            cup_id=cup_id,
            course_id=course_id,
        )
        save_game_repository.touch_save_game(
            session,
            save_game_id=save_game_id,
            updated_at=now,
        )
        return course_setup


def upsert_cup_setup(
    store: ManagerStore,
    *,
    save_game_id: str,
    cup_id: str,
    vehicle_id: str,
    difficulty: str | None = None,
) -> ManagedSaveCupSetup:
    """Create or update one save-game cup vehicle setup."""

    vehicle_by_id(vehicle_id)
    store.initialize()
    now = utc_now()
    with store._orm_session() as session:
        save_game = save_game_repository.get_save_game(session, save_game_id)
        if save_game is None:
            raise KeyError("save game not found")
        progress = build_unlock_progress(save_game.save_path)
        if vehicle_id not in progress.unlocked_vehicle_ids:
            raise ValueError(f"vehicle {vehicle_id!r} is not unlocked in this save game")
        cup_setup = save_game_repository.upsert_cup_setup(
            session,
            setup_id=new_record_id(f"{save_game_id} cup setup"),
            save_game_id=save_game_id,
            cup_id=cup_id,
            vehicle_id=vehicle_id,
            created_at=now,
            updated_at=now,
            difficulty=difficulty,
        )
        save_game_repository.touch_save_game(
            session,
            save_game_id=save_game_id,
            updated_at=now,
        )
        return cup_setup


def update_save_game_status(
    store: ManagerStore,
    *,
    save_game_id: str,
    status: SaveGameStatus,
) -> ManagedSaveGame | None:
    """Update one save-game lifecycle status."""

    store.initialize()
    now = utc_now()
    last_finished_at = now if status in {"finished", "failed"} else None
    with store._orm_session() as session:
        return save_game_repository.update_save_game_status(
            session,
            save_game_id=save_game_id,
            status=status,
            updated_at=now,
            last_finished_at=last_finished_at,
        )


def _insert_policy_backed_save_attempt(
    session: Session,
    *,
    save_game_id: str,
    progress: ManagedSaveUnlockProgress,
    target: ManagedSaveUnlockTarget,
    started_at: str,
    error_subject: str,
) -> ManagedSaveAttempt:
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
        timestamp=started_at,
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
    attempt = ManagedSaveAttempt(
        id=new_record_id(f"{save_game_id} attempt"),
        save_game_id=save_game_id,
        target_kind=target.kind,
        status="running",
        difficulty=target.difficulty,
        cup_id=target.cup_id,
        course_id=target.course_id,
        started_at=started_at,
    )
    save_game_repository.insert_save_attempt(session, attempt)
    save_game_repository.touch_save_game(
        session,
        save_game_id=save_game_id,
        updated_at=started_at,
    )
    return attempt


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


def _progress_target(
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


def _validate_course_setup_target(*, course_id: str | None) -> None:
    if course_id is None:
        raise ValueError("course setups require course")


def _course_setup_target_label(target: CourseSetupTarget) -> str:
    parts = [
        target.difficulty or "*",
        target.cup_id or "*",
        target.course_id or "*",
    ]
    return "/".join(parts)


def _validate_engine_setting(*, engine_setting_raw_value: int) -> None:
    if not 0 <= engine_setting_raw_value <= 100:
        raise ValueError(
            f"engine_setting_raw_value must be in [0, 100], got {engine_setting_raw_value}"
        )
