# src/rl_fzerox/core/manager/registry/save_games/setups.py
"""Course and cup setup operations for manager-owned save games."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from rl_fzerox.core.career_mode.progress.unlocks import build_unlock_progress
from rl_fzerox.core.domain.engine import (
    engine_percent_to_slider_step,
    validate_engine_slider_step,
)
from rl_fzerox.core.manager.db.repositories import save_games as save_game_repository
from rl_fzerox.core.manager.models import ManagedSaveCourseSetup, ManagedSaveCupSetup
from rl_fzerox.core.manager.registry.common import new_record_id, utc_now
from rl_fzerox.core.runtime_spec.vehicle_catalog import vehicle_by_id

if TYPE_CHECKING:
    from rl_fzerox.core.manager.store import ManagerStore


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


def upsert_course_setup(
    store: ManagerStore,
    *,
    save_game_id: str,
    policy_run_id: str,
    policy_artifact: Literal["latest", "best"],
    engine_setting_raw_value: int = engine_percent_to_slider_step(50),
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


def _validate_course_setup_target(*, course_id: str | None) -> None:
    if course_id is None:
        raise ValueError("course setups require course")


def _validate_engine_setting(*, engine_setting_raw_value: int) -> None:
    validate_engine_slider_step(engine_setting_raw_value)
