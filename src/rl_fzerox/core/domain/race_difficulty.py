# src/rl_fzerox/core/domain/race_difficulty.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeGuard

type RaceDifficultyName = Literal["novice", "standard", "expert", "master"]


@dataclass(frozen=True, slots=True)
class RaceDifficultySpec:
    """GP difficulty name and its 0-based game-facing raw value."""

    name: RaceDifficultyName
    raw_value: int


RACE_DIFFICULTIES: Final[tuple[RaceDifficultySpec, ...]] = (
    RaceDifficultySpec(name="novice", raw_value=0),
    RaceDifficultySpec(name="standard", raw_value=1),
    RaceDifficultySpec(name="expert", raw_value=2),
    RaceDifficultySpec(name="master", raw_value=3),
)
_DEFAULT_GP_DIFFICULTY: Final[RaceDifficultyName] = "novice"

_RACE_DIFFICULTY_BY_NAME: Final[Mapping[RaceDifficultyName, RaceDifficultySpec]] = MappingProxyType(
    {spec.name: spec for spec in RACE_DIFFICULTIES}
)


def race_difficulty_names() -> tuple[RaceDifficultyName, ...]:
    """Return GP difficulties in menu/progression order."""

    return tuple(spec.name for spec in RACE_DIFFICULTIES)


def default_gp_difficulty() -> RaceDifficultyName:
    """Return the GP difficulty used when a config omits one."""

    return _DEFAULT_GP_DIFFICULTY


def race_difficulty_raw_value(name: RaceDifficultyName) -> int:
    """Return the F-Zero X raw GP difficulty value for save/menu code."""

    return _RACE_DIFFICULTY_BY_NAME[name].raw_value


def is_race_difficulty_name(value: str) -> TypeGuard[RaceDifficultyName]:
    """Return whether a string is one of the supported GP difficulty names."""

    return value in _RACE_DIFFICULTY_BY_NAME
