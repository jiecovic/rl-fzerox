# tests/core/domain/test_courses.py
from __future__ import annotations

from rl_fzerox.core.domain.courses import (
    BUILT_IN_COURSES,
    built_in_course_by_ref,
    built_in_course_configs,
    built_in_course_ref_by_id,
    built_in_course_refs_by_cup,
)


def test_built_in_course_catalog_has_expected_identity_invariants() -> None:
    refs = tuple(course.ref for course in BUILT_IN_COURSES)
    indices = tuple(course.course_index for course in BUILT_IN_COURSES)

    assert len(BUILT_IN_COURSES) == 24
    assert len(set(refs)) == len(refs)
    assert indices == tuple(range(24))
    assert all(course.records is not None for course in BUILT_IN_COURSES)


def test_built_in_course_refs_follow_cup_order() -> None:
    assert built_in_course_refs_by_cup("jack") == (
        "jack/mute_city",
        "jack/silence",
        "jack/sand_ocean",
        "jack/devils_forest",
        "jack/big_blue",
        "jack/port_town",
    )
    assert built_in_course_refs_by_cup("joker")[-1] == "joker/big_hand"


def test_built_in_course_lookup_helpers_return_config_payloads() -> None:
    config = built_in_course_by_ref("queen/mute_city_2")

    assert config is not None
    assert config["id"] == "mute_city_2"
    assert config["display_name"] == "Mute City 2"
    assert built_in_course_ref_by_id("mute_city") == ("jack/mute_city",)
    assert built_in_course_ref_by_id("mute_city", cup="queen") == ()
    assert built_in_course_configs()[0][0] == "jack/mute_city"
