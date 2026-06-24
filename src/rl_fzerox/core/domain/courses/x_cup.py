# src/rl_fzerox/core/domain/courses/x_cup.py
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from rl_fzerox.core.seed import derive_seed

type XCupGeneratedCourseKind = Literal["x_cup"]
type XCupRaceMode = Literal["gp_race"]


@dataclass(frozen=True, slots=True)
class XCupRotationDefaults:
    """Default policy for replacing solved generated X Cup slots."""

    completion_threshold: float = 0.9
    min_episodes: int = 24
    max_episodes: int | None = None
    ema_alpha: float = 0.3


@dataclass(frozen=True, slots=True)
class XCupRetentionPolicy:
    """Internal disk-retention policy for generated run-local states."""

    inactive_buffer_courses: int = 2


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
    rotation_defaults: XCupRotationDefaults = field(default_factory=XCupRotationDefaults)
    retention_policy: XCupRetentionPolicy = field(default_factory=XCupRetentionPolicy)


@dataclass(frozen=True, slots=True)
class GeneratedXCupCourseIdentity:
    """Stable generated-course identity used before materialization writes state."""

    course_id: str
    display_name: str
    course_hash: str
    seed: int
    slot: int
    generation: int


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


def generated_x_cup_course_identity(
    *,
    master_seed: int | None,
    slot: int,
    generation: int,
) -> GeneratedXCupCourseIdentity:
    """Derive the stable identity for one generated X Cup slot generation."""

    seed = _x_cup_seed(0 if master_seed is None else master_seed, slot=slot, generation=generation)
    course_hash = _x_cup_course_hash(seed=seed)
    short_hash = course_hash[: X_CUP_COURSE.display_hash_chars]
    return GeneratedXCupCourseIdentity(
        course_id=f"{X_CUP_COURSE.id_prefix}_{short_hash}",
        display_name=f"{X_CUP_COURSE.display_prefix} {short_hash}",
        course_hash=course_hash,
        seed=seed,
        slot=slot,
        generation=generation,
    )


def generated_x_cup_slot_key(slot: int) -> str:
    """Return the stable runtime sampling key for one generated X Cup slot."""

    return f"{X_CUP_COURSE.id_prefix}_slot_{int(slot) + 1}"


def _x_cup_seed(master_seed: int, *, slot: int, generation: int) -> int:
    seed = derive_seed(
        master_seed,
        X_CUP_COURSE.generator_version,
        X_CUP_COURSE.course_index,
        int(slot),
        int(generation),
    )
    if seed is None:
        raise ValueError("X Cup seed derivation requires a concrete master seed")
    return seed


def _x_cup_course_hash(*, seed: int) -> str:
    return _sha256_json(
        {
            "generator_version": X_CUP_COURSE.generator_version,
            "race_mode": X_CUP_COURSE.race_mode,
            "rng_patch_timing": X_CUP_COURSE.rng_patch_timing,
            "seed": seed,
            "source_course_index": X_CUP_COURSE.course_index,
        }
    ).hex()


def _sha256_json(payload: Mapping[str, object]) -> bytes:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).digest()
