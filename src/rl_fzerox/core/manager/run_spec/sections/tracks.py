# src/rl_fzerox/core/manager/run_spec/sections/tracks.py
"""Track-pool and race-mode section of the manager-owned run-spec model."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from rl_fzerox.core.domain.courses import BUILT_IN_COURSES
from rl_fzerox.core.domain.race_difficulty import default_gp_difficulty
from rl_fzerox.core.manager.run_spec.common import (
    GpDifficulty,
    RaceMode,
    TrackPoolMode,
    TrackSamplingMode,
)


def default_selected_course_ids() -> tuple[str, ...]:
    """Return the default manager course pool in game order."""

    return tuple(course.id for course in BUILT_IN_COURSES)


def built_in_course_id_set() -> frozenset[str]:
    return frozenset(course.id for course in BUILT_IN_COURSES)


class ManagedTracksConfig(BaseModel):
    """Track pool and race-mode knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    pool_mode: TrackPoolMode = "built_in"
    race_mode: RaceMode = "time_attack"
    gp_difficulty: GpDifficulty | None = None
    sampling_mode: TrackSamplingMode = "step_balanced"
    step_balance_update_episodes: int = Field(default=5, ge=1)
    step_balance_ema_alpha: float = Field(default=0.1, gt=0.0, le=1.0)
    step_balance_max_weight_scale: float = Field(default=5.0, ge=1.0)
    adaptive_step_balance_completion_weight: float = Field(default=0.35, ge=0.0)
    adaptive_step_balance_target_completion: float = Field(default=0.9, ge=0.0, le=1.0)
    selected_course_ids: tuple[str, ...] = Field(default_factory=default_selected_course_ids)

    @model_validator(mode="after")
    def _validate_selected_course_ids(self) -> ManagedTracksConfig:
        if self.race_mode != "gp_race":
            self.gp_difficulty = None
        elif self.gp_difficulty is None:
            self.gp_difficulty = default_gp_difficulty()
        if self.pool_mode == "x_cup" and self.race_mode != "gp_race":
            raise ValueError("tracks.pool_mode=x_cup requires tracks.race_mode=gp_race")
        if self.pool_mode == "built_in" and not self.selected_course_ids:
            raise ValueError("tracks.selected_course_ids must not be empty")
        if len(set(self.selected_course_ids)) != len(self.selected_course_ids):
            raise ValueError("tracks.selected_course_ids must not contain duplicates")
        unknown_ids = sorted(set(self.selected_course_ids) - built_in_course_id_set())
        if unknown_ids:
            joined = ", ".join(unknown_ids)
            raise ValueError(f"tracks.selected_course_ids contains unknown courses: {joined}")
        return self

    @model_serializer(mode="wrap")
    def _serialize(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        data = handler(self)
        if self.race_mode != "gp_race":
            data.pop("gp_difficulty", None)
        return data
