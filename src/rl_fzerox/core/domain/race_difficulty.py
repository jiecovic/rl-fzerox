# src/rl_fzerox/core/domain/race_difficulty.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeGuard

type RaceDifficultyName = Literal["novice", "standard", "expert", "master"]


@dataclass(frozen=True, slots=True)
class RaceDifficultySpec:
    name: RaceDifficultyName
    raw_value: int


RACE_DIFFICULTIES: tuple[RaceDifficultySpec, ...] = (
    RaceDifficultySpec(name="novice", raw_value=0),
    RaceDifficultySpec(name="standard", raw_value=1),
    RaceDifficultySpec(name="expert", raw_value=2),
    RaceDifficultySpec(name="master", raw_value=3),
)

_RACE_DIFFICULTY_BY_NAME = {spec.name: spec for spec in RACE_DIFFICULTIES}


def race_difficulty_names() -> tuple[RaceDifficultyName, ...]:
    return tuple(spec.name for spec in RACE_DIFFICULTIES)


def default_gp_difficulty() -> RaceDifficultyName:
    return "novice"


def race_difficulty_raw_value(name: RaceDifficultyName) -> int:
    return _RACE_DIFFICULTY_BY_NAME[name].raw_value


def is_race_difficulty_name(value: str) -> TypeGuard[RaceDifficultyName]:
    return value in _RACE_DIFFICULTY_BY_NAME
