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
from rl_fzerox.core.manager.models import (
    CourseSetupScope,
    ManagedSaveAttempt,
    ManagedSaveCourseSetup,
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


def start_save_attempt(
    store: ManagerStore,
    *,
    save_game_id: str,
    target_kind: str | None = None,
    policy_run_id: str | None = None,
    policy_artifact: Literal["latest", "best"] | None = None,
    difficulty: str | None = None,
    cup_id: str | None = None,
    course_id: str | None = None,
) -> ManagedSaveAttempt:
    """Record the start of one unlock attempt."""

    if policy_run_id is None and policy_artifact is not None:
        raise ValueError("policy_artifact requires policy_run_id")
    if policy_run_id is not None and policy_artifact is None:
        raise ValueError("policy_run_id requires policy_artifact")
    store.initialize()
    now = utc_now()
    attempt = ManagedSaveAttempt(
        id=new_record_id(f"{save_game_id} attempt"),
        save_game_id=save_game_id,
        target_kind=target_kind,
        policy_run_id=policy_run_id,
        policy_artifact=policy_artifact,
        status="running",
        difficulty=difficulty,
        cup_id=cup_id,
        course_id=course_id,
        started_at=now,
    )
    with store._orm_session() as session:
        if save_game_repository.get_save_game(session, save_game_id) is None:
            raise KeyError("save game not found")
        if policy_run_id is not None and not save_game_repository.run_exists(
            session, policy_run_id
        ):
            raise KeyError("policy run not found")
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
        if attempt.policy_run_id is None or attempt.policy_artifact is None:
            raise ValueError("save attempt is missing a resolved policy")

        save_game = save_game_repository.get_save_game(session, attempt.save_game_id)
        if save_game is None:
            raise KeyError("save game not found")
        target = _target_for_attempt(attempt)
        policy_run = run_repository.get_managed_run(session, attempt.policy_run_id)
        if policy_run is None:
            raise KeyError("policy run not found")

        course_setups = save_game_repository.list_course_setups(session, attempt.save_game_id)
        course_setup_target = CourseSetupTarget(
            difficulty=attempt.difficulty,
            cup_id=attempt.cup_id,
            course_id=attempt.course_id,
        )
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
        return SaveAttemptExecutionContext(
            save_game=save_game,
            attempt=attempt,
            target=target.to_progress_target(status="pending"),
            course_setup_target=course_setup_target,
            course_setup=course_setup,
            policy_run=policy_run,
            policy_artifact=attempt.policy_artifact,
            policy_path=resolve_policy_artifact_path(
                policy_run.run_dir,
                artifact=attempt.policy_artifact,
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
    scope: CourseSetupScope,
    policy_run_id: str,
    policy_artifact: Literal["latest", "best"],
    vehicle_id: str = "blue_falcon",
    engine_setting_raw_value: int = 50,
    difficulty: str | None = None,
    cup_id: str | None = None,
    course_id: str | None = None,
) -> ManagedSaveCourseSetup:
    """Create or update one save-game course setup."""

    _validate_course_setup_scope(
        scope=scope,
        difficulty=difficulty,
        cup_id=cup_id,
        course_id=course_id,
    )
    _validate_vehicle_setup(
        vehicle_id=vehicle_id,
        engine_setting_raw_value=engine_setting_raw_value,
    )
    store.initialize()
    now = utc_now()
    with store._orm_session() as session:
        save_game = save_game_repository.get_save_game(session, save_game_id)
        if save_game is None:
            raise KeyError("save game not found")
        progress = build_unlock_progress(save_game.save_path)
        if vehicle_id not in progress.unlocked_vehicle_ids:
            raise ValueError(f"vehicle {vehicle_id!r} is not unlocked in this save game")
        if not save_game_repository.run_exists(session, policy_run_id):
            raise KeyError("policy run not found")
        course_setup = save_game_repository.upsert_course_setup(
            session,
            setup_id=new_record_id(f"{save_game_id} setup"),
            save_game_id=save_game_id,
            scope=scope,
            policy_run_id=policy_run_id,
            policy_artifact=policy_artifact,
            vehicle_id=vehicle_id,
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
    target: ManagedSaveUnlockTarget,
    started_at: str,
    error_subject: str,
) -> ManagedSaveAttempt:
    course_setups = save_game_repository.list_course_setups(session, save_game_id)
    course_setup_target = CourseSetupTarget(
        difficulty=target.difficulty,
        cup_id=target.cup_id,
        course_id=target.course_id,
    )
    missing_targets = missing_course_setup_targets(course_setups, course_setup_target)
    if missing_targets:
        missing_labels = ", ".join(_course_setup_target_label(item) for item in missing_targets)
        raise ValueError(f"{error_subject} has no matching course setup for {missing_labels}")
    resolved_targets = required_course_setup_targets(course_setup_target)
    course_setup_target = resolved_targets[0]
    course_setup = resolve_course_setup(course_setups, course_setup_target)
    if course_setup is None:
        raise ValueError(f"{error_subject} has no matching course setup")
    if not save_game_repository.run_exists(session, course_setup.policy_run_id):
        raise KeyError("policy run not found")
    attempt = ManagedSaveAttempt(
        id=new_record_id(f"{save_game_id} attempt"),
        save_game_id=save_game_id,
        target_kind=target.kind,
        policy_run_id=course_setup.policy_run_id,
        policy_artifact=course_setup.policy_artifact,
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


def _validate_course_setup_scope(
    *,
    scope: CourseSetupScope,
    difficulty: str | None,
    cup_id: str | None,
    course_id: str | None,
) -> None:
    if scope == "global" and any(value is not None for value in (difficulty, cup_id, course_id)):
        raise ValueError("global course setups cannot include difficulty, cup, or course")
    if scope == "difficulty" and (
        difficulty is None or cup_id is not None or course_id is not None
    ):
        raise ValueError("difficulty course setups require only difficulty")
    if scope == "cup" and (cup_id is None or course_id is not None):
        raise ValueError("cup course setups require cup and may include difficulty")
    if scope == "course" and course_id is None:
        raise ValueError("course course setups require course")


def _course_setup_target_label(target: CourseSetupTarget) -> str:
    parts = [
        target.difficulty or "*",
        target.cup_id or "*",
        target.course_id or "*",
    ]
    return "/".join(parts)


def _validate_vehicle_setup(*, vehicle_id: str, engine_setting_raw_value: int) -> None:
    vehicle_by_id(vehicle_id)
    if not 0 <= engine_setting_raw_value <= 100:
        raise ValueError(
            f"engine_setting_raw_value must be in [0, 100], got {engine_setting_raw_value}"
        )
