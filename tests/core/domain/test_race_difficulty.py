# tests/core/domain/test_race_difficulty.py
from __future__ import annotations

from rl_fzerox.core.domain.race import (
    RACE_DIFFICULTIES,
    default_gp_difficulty,
    is_race_difficulty_name,
    race_difficulty_names,
    race_difficulty_raw_value,
)


def test_race_difficulty_names_and_default_are_stable() -> None:
    assert race_difficulty_names() == ("novice", "standard", "expert", "master")
    assert default_gp_difficulty() == "novice"


def test_race_difficulty_raw_values_match_game_order() -> None:
    assert race_difficulty_raw_value("novice") == 0
    assert race_difficulty_raw_value("standard") == 1
    assert race_difficulty_raw_value("expert") == 2
    assert race_difficulty_raw_value("master") == 3


def test_race_difficulty_specs_are_unique_and_contiguous() -> None:
    names = tuple(spec.name for spec in RACE_DIFFICULTIES)
    raw_values = tuple(spec.raw_value for spec in RACE_DIFFICULTIES)

    assert names == race_difficulty_names()
    assert len(set(names)) == len(names)
    assert raw_values == tuple(range(len(RACE_DIFFICULTIES)))


def test_is_race_difficulty_name_narrows_known_names() -> None:
    assert is_race_difficulty_name("master") is True
    assert is_race_difficulty_name("beginner") is False
