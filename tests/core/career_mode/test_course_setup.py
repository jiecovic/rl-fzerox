# tests/core/career_mode/test_course_setup.py

from __future__ import annotations

from typing import Literal

from rl_fzerox.core.career_mode.course_setup import (
    CourseSetupTarget,
    resolve_course_setup,
)
from rl_fzerox.core.manager.models import CourseSetupScope, ManagedSaveCourseSetup


def test_resolve_course_setup_prefers_course_over_broader_scopes() -> None:
    setups = (
        _setup("global", "run-global"),
        _setup("difficulty", "run-expert", difficulty="expert"),
        _setup("cup", "run-joker", difficulty="expert", cup_id="joker"),
        _setup(
            "course",
            "run-big-hand",
            difficulty="expert",
            cup_id="joker",
            course_id="big_hand",
        ),
    )

    resolved = resolve_course_setup(
        setups,
        CourseSetupTarget(difficulty="expert", cup_id="joker", course_id="big_hand"),
    )

    assert resolved is not None
    assert resolved.policy_run_id == "run-big-hand"


def test_resolve_course_setup_uses_cup_before_difficulty() -> None:
    setups = (
        _setup("difficulty", "run-expert", difficulty="expert"),
        _setup("cup", "run-jack", difficulty="expert", cup_id="jack"),
    )

    resolved = resolve_course_setup(
        setups,
        CourseSetupTarget(difficulty="expert", cup_id="jack", course_id="mute_city"),
    )

    assert resolved is not None
    assert resolved.policy_run_id == "run-jack"


def test_resolve_course_setup_uses_first_course_for_cup_target() -> None:
    setups = (
        _setup("course", "run-port-town", cup_id="jack", course_id="port_town"),
        _setup("course", "run-mute-city", cup_id="jack", course_id="mute_city"),
        _setup("course", "run-silence", cup_id="jack", course_id="silence"),
    )

    resolved = resolve_course_setup(
        setups,
        CourseSetupTarget(difficulty="novice", cup_id="jack"),
    )

    assert resolved is not None
    assert resolved.policy_run_id == "run-mute-city"


def test_resolve_course_setup_treats_missing_scope_fields_as_wildcards() -> None:
    setups = (
        _setup("cup", "run-any-joker", cup_id="joker"),
        _setup("global", "run-global"),
    )

    resolved = resolve_course_setup(
        setups,
        CourseSetupTarget(difficulty="master", cup_id="joker", course_id="rainbow_road"),
    )

    assert resolved is not None
    assert resolved.policy_run_id == "run-any-joker"


def test_resolve_course_setup_rejects_non_matching_filters() -> None:
    setups = (
        _setup("course", "run-wrong-course", course_id="mute_city"),
        _setup("difficulty", "run-expert", difficulty="expert"),
        _setup("global", "run-global"),
    )

    resolved = resolve_course_setup(
        setups,
        CourseSetupTarget(difficulty="standard", cup_id="joker", course_id="big_hand"),
    )

    assert resolved is not None
    assert resolved.policy_run_id == "run-global"


def test_resolve_course_setup_uses_newer_duplicate_for_same_scope() -> None:
    setups = (
        _setup("global", "run-old", setup_id="a", updated_at="2026-01-01T00:00:00Z"),
        _setup("global", "run-new", setup_id="b", updated_at="2026-01-02T00:00:00Z"),
    )

    resolved = resolve_course_setup(setups, CourseSetupTarget())

    assert resolved is not None
    assert resolved.policy_run_id == "run-new"


def test_resolve_course_setup_returns_none_without_match() -> None:
    resolved = resolve_course_setup(
        (_setup("difficulty", "run-expert", difficulty="expert"),),
        CourseSetupTarget(difficulty="standard"),
    )

    assert resolved is None


def _setup(
    scope: CourseSetupScope,
    policy_run_id: str,
    *,
    setup_id: str | None = None,
    policy_artifact: Literal["latest", "best"] = "latest",
    difficulty: str | None = None,
    cup_id: str | None = None,
    course_id: str | None = None,
    updated_at: str = "2026-01-01T00:00:00Z",
) -> ManagedSaveCourseSetup:
    return ManagedSaveCourseSetup(
        id=setup_id or f"{scope}-{policy_run_id}",
        save_game_id="save",
        scope=scope,
        policy_run_id=policy_run_id,
        policy_artifact=policy_artifact,
        vehicle_id="blue_falcon",
        engine_setting_raw_value=50,
        difficulty=difficulty,
        cup_id=cup_id,
        course_id=course_id,
        created_at="2026-01-01T00:00:00Z",
        updated_at=updated_at,
    )
