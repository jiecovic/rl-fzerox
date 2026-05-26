# src/rl_fzerox/core/domain/x_cup.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

XCupGeneratedCourseKind: TypeAlias = Literal["x_cup"]
XCupRaceMode: TypeAlias = Literal["gp_race"]


@dataclass(frozen=True, slots=True)
class XCupCourseSpec:
    """Stable game/menu identity for generated X Cup courses."""

    course_index: int
    generated_kind: XCupGeneratedCourseKind
    race_mode: XCupRaceMode
    default_generated_count: int
    max_generated_count: int
    generator_version: int
    menu_right_presses_from_jack: int
    rng_patch_timing: str
    display_hash_chars: int
    id_prefix: str
    display_prefix: str
    materializer_mode: str
    baseline_cache_kind: str
    cache_dir: str


X_CUP_COURSE = XCupCourseSpec(
    course_index=48,
    generated_kind="x_cup",
    race_mode="gp_race",
    default_generated_count=6,
    max_generated_count=128,
    generator_version=1,
    menu_right_presses_from_jack=4,
    rng_patch_timing="after_x_cup_select",
    display_hash_chars=8,
    id_prefix="x_cup",
    display_prefix="X Cup",
    materializer_mode="x_cup_generated_course",
    baseline_cache_kind="x_cup_baseline",
    cache_dir="x_cup",
)
