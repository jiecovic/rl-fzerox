# tests/core/domain/test_x_cup.py
from __future__ import annotations

from rl_fzerox.core.domain.courses import (
    X_CUP_COURSE,
    generated_x_cup_course_identity,
    generated_x_cup_slot_key,
)


def test_x_cup_course_spec_matches_generated_course_contract() -> None:
    assert X_CUP_COURSE.course_index == 48
    assert X_CUP_COURSE.generated_kind == "x_cup"
    assert X_CUP_COURSE.race_mode == "gp_race"
    assert X_CUP_COURSE.default_generated_count == 6
    assert X_CUP_COURSE.max_generated_count == 128
    assert X_CUP_COURSE.rotation_defaults.completion_threshold == 0.9


def test_generated_x_cup_slot_keys_are_one_based_runtime_keys() -> None:
    assert generated_x_cup_slot_key(0) == "x_cup_slot_1"
    assert generated_x_cup_slot_key(5) == "x_cup_slot_6"


def test_generated_x_cup_identity_is_deterministic_and_changes_by_generation() -> None:
    first = generated_x_cup_course_identity(master_seed=123, slot=2, generation=0)
    repeat = generated_x_cup_course_identity(master_seed=123, slot=2, generation=0)
    next_generation = generated_x_cup_course_identity(master_seed=123, slot=2, generation=1)

    assert first == repeat
    assert first != next_generation
    assert first.course_id.startswith("x_cup_")
    assert first.display_name.startswith("X Cup ")
    assert len(first.course_hash) == 64
    assert first.slot == 2
    assert first.generation == 0
