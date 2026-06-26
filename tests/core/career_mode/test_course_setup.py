# tests/core/career_mode/test_course_setup.py
"""Tests for resolving policy/vehicle setup rows for Career Mode targets."""

from __future__ import annotations

from typing import Literal

from rl_fzerox.core.career_mode.course_setup import (
    CourseSetupTarget,
    missing_course_setup_targets,
    required_course_setup_targets,
    resolve_course_setup,
    resolve_cup_setup,
)
from rl_fzerox.core.manager.models import ManagedSaveCourseSetup, ManagedSaveCupSetup


def test_resolve_course_setup_prefers_specific_difficulty_over_generic_row() -> None:
    setups = (
        _course_setup(
            "run-generic",
            cup_id="joker",
            course_id="big_hand",
            updated_at="2026-01-02T00:00:00Z",
        ),
        _course_setup(
            "run-expert",
            difficulty="expert",
            cup_id="joker",
            course_id="big_hand",
            updated_at="2026-01-01T00:00:00Z",
        ),
    )

    resolved = resolve_course_setup(
        setups,
        CourseSetupTarget(difficulty="expert", cup_id="joker", course_id="big_hand"),
    )

    assert resolved is not None
    assert resolved.policy_source_id == "run-expert"


def test_resolve_course_setup_prefers_cup_specific_over_course_only_row() -> None:
    setups = (
        _course_setup("run-course-only", difficulty="expert", course_id="mute_city"),
        _course_setup("run-jack", difficulty="expert", cup_id="jack", course_id="mute_city"),
    )

    resolved = resolve_course_setup(
        setups,
        CourseSetupTarget(difficulty="expert", cup_id="jack", course_id="mute_city"),
    )

    assert resolved is not None
    assert resolved.policy_source_id == "run-jack"


def test_resolve_course_setup_does_not_use_course_rows_for_cup_target() -> None:
    setups = (
        _course_setup("run-port-town", cup_id="jack", course_id="port_town"),
        _course_setup("run-mute-city", cup_id="jack", course_id="mute_city"),
        _course_setup("run-silence", cup_id="jack", course_id="silence"),
    )

    resolved = resolve_course_setup(
        setups,
        CourseSetupTarget(difficulty="novice", cup_id="jack"),
    )

    assert resolved is None


def test_required_course_setup_targets_expands_cup_target_in_game_order() -> None:
    targets = required_course_setup_targets(
        CourseSetupTarget(difficulty="novice", cup_id="jack"),
    )

    assert [target.course_id for target in targets] == [
        "mute_city",
        "silence",
        "sand_ocean",
        "devils_forest",
        "big_blue",
        "port_town",
    ]


def test_missing_course_setup_targets_accepts_course_rows_for_cup_target() -> None:
    setups = (
        _course_setup("run-1", cup_id="jack", course_id="mute_city"),
        _course_setup("run-2", cup_id="jack", course_id="silence"),
        _course_setup("run-3", cup_id="jack", course_id="sand_ocean"),
        _course_setup("run-4", cup_id="jack", course_id="devils_forest"),
        _course_setup("run-5", cup_id="jack", course_id="big_blue"),
        _course_setup("run-6", cup_id="jack", course_id="port_town"),
    )

    missing = missing_course_setup_targets(
        setups,
        CourseSetupTarget(difficulty="novice", cup_id="jack"),
    )

    assert missing == ()


def test_resolve_course_setup_preserves_per_course_engine_values() -> None:
    setups = (
        _course_setup("run-1", cup_id="jack", course_id="mute_city", engine=60),
        _course_setup("run-2", cup_id="jack", course_id="silence", engine=40),
    )

    mute_city = resolve_course_setup(
        setups,
        CourseSetupTarget(difficulty="novice", cup_id="jack", course_id="mute_city"),
    )
    silence = resolve_course_setup(
        setups,
        CourseSetupTarget(difficulty="novice", cup_id="jack", course_id="silence"),
    )

    assert mute_city is not None
    assert silence is not None
    assert mute_city.engine_setting_raw_value == 60
    assert silence.engine_setting_raw_value == 40


def test_missing_course_setup_targets_reports_missing_cup_courses() -> None:
    setups = (
        _course_setup("run-1", cup_id="jack", course_id="mute_city"),
        _course_setup("run-2", cup_id="jack", course_id="silence"),
    )

    missing = missing_course_setup_targets(
        setups,
        CourseSetupTarget(difficulty="novice", cup_id="jack"),
    )

    assert [target.course_id for target in missing] == [
        "sand_ocean",
        "devils_forest",
        "big_blue",
        "port_town",
    ]


def test_resolve_course_setup_treats_missing_difficulty_and_cup_as_wildcards() -> None:
    setups = (
        _course_setup("run-any-rainbow", course_id="rainbow_road"),
        _course_setup("run-joker-rainbow", cup_id="joker", course_id="rainbow_road"),
    )

    resolved = resolve_course_setup(
        setups,
        CourseSetupTarget(difficulty="master", cup_id="joker", course_id="rainbow_road"),
    )

    assert resolved is not None
    assert resolved.policy_source_id == "run-joker-rainbow"


def test_resolve_course_setup_rejects_non_matching_filters() -> None:
    setups = (
        _course_setup("run-wrong-course", course_id="mute_city"),
        _course_setup("run-expert", difficulty="expert", course_id="big_hand"),
    )

    resolved = resolve_course_setup(
        setups,
        CourseSetupTarget(difficulty="standard", cup_id="joker", course_id="big_hand"),
    )

    assert resolved is None


def test_resolve_course_setup_uses_newer_duplicate_for_same_target() -> None:
    setups = (
        _course_setup(
            "run-old",
            setup_id="a",
            cup_id="jack",
            course_id="mute_city",
            updated_at="2026-01-01T00:00:00Z",
        ),
        _course_setup(
            "run-new",
            setup_id="b",
            cup_id="jack",
            course_id="mute_city",
            updated_at="2026-01-02T00:00:00Z",
        ),
    )

    resolved = resolve_course_setup(
        setups,
        CourseSetupTarget(cup_id="jack", course_id="mute_city"),
    )

    assert resolved is not None
    assert resolved.policy_source_id == "run-new"


def test_resolve_course_setup_returns_none_without_course_target() -> None:
    resolved = resolve_course_setup(
        (_course_setup("run-expert", difficulty="expert", course_id="mute_city"),),
        CourseSetupTarget(difficulty="standard"),
    )

    assert resolved is None


def test_resolve_cup_setup_prefers_specific_difficulty_over_generic_row() -> None:
    setups = (
        _cup_setup("blue_falcon", cup_id="jack", updated_at="2026-01-02T00:00:00Z"),
        _cup_setup(
            "deep_claw",
            difficulty="expert",
            cup_id="jack",
            updated_at="2026-01-01T00:00:00Z",
        ),
    )

    resolved = resolve_cup_setup(
        setups,
        CourseSetupTarget(difficulty="expert", cup_id="jack"),
    )

    assert resolved is not None
    assert resolved.vehicle_id == "deep_claw"


def _course_setup(
    policy_source_id: str,
    *,
    setup_id: str | None = None,
    policy_artifact: Literal["latest", "best"] = "latest",
    difficulty: str | None = None,
    cup_id: str | None = None,
    course_id: str | None = None,
    engine: int = 50,
    updated_at: str = "2026-01-01T00:00:00Z",
) -> ManagedSaveCourseSetup:
    return ManagedSaveCourseSetup(
        id=setup_id or f"course-{policy_source_id}",
        save_game_id="save",
        policy_source_kind="run",
        policy_source_id=policy_source_id,
        policy_artifact=policy_artifact,
        engine_setting_raw_value=engine,
        difficulty=difficulty,
        cup_id=cup_id,
        course_id=course_id,
        created_at="2026-01-01T00:00:00Z",
        updated_at=updated_at,
    )


def _cup_setup(
    vehicle_id: str,
    *,
    setup_id: str | None = None,
    difficulty: str | None = None,
    cup_id: str = "jack",
    updated_at: str = "2026-01-01T00:00:00Z",
) -> ManagedSaveCupSetup:
    return ManagedSaveCupSetup(
        id=setup_id or f"cup-{vehicle_id}",
        save_game_id="save",
        cup_id=cup_id,
        vehicle_id=vehicle_id,
        difficulty=difficulty,
        created_at="2026-01-01T00:00:00Z",
        updated_at=updated_at,
    )
