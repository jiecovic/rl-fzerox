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

from rl_fzerox.core.domain.courses import BUILT_IN_COURSES, X_CUP_COURSE
from rl_fzerox.core.domain.race import default_gp_difficulty
from rl_fzerox.core.manager.run_spec.common import (
    DeficitBudgetDifficultyMetric,
    GpDifficulty,
    RaceMode,
    TrackSamplingMode,
)


def default_selected_course_ids() -> tuple[str, ...]:
    """Return the default manager course pool in game order."""

    return tuple(course.id for course in BUILT_IN_COURSES)


def built_in_course_id_set() -> frozenset[str]:
    return frozenset(course.id for course in BUILT_IN_COURSES)


def default_gp_difficulties() -> tuple[GpDifficulty, ...]:
    """Return the default GP difficulty pool."""

    return (default_gp_difficulty(),)


class ManagedXCupAutoRegenerationConfig(BaseModel):
    """Manager-owned policy for rotating solved generated X Cup slots."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    completion_threshold: float = Field(
        default=X_CUP_COURSE.rotation_defaults.completion_threshold,
        ge=0.0,
        le=1.0,
    )
    min_episodes: int = Field(
        default=X_CUP_COURSE.rotation_defaults.min_episodes,
        ge=1,
    )
    max_episodes: int | None = Field(
        default=X_CUP_COURSE.rotation_defaults.max_episodes,
        ge=1,
    )
    ema_alpha: float = Field(
        default=X_CUP_COURSE.rotation_defaults.ema_alpha,
        gt=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def _validate_episode_window(self) -> ManagedXCupAutoRegenerationConfig:
        if self.max_episodes is not None and self.max_episodes < self.min_episodes:
            raise ValueError("tracks.x_cup_auto_regeneration.max_episodes must be >= min_episodes")
        return self


class ManagedTracksConfig(BaseModel):
    """Track pool and race-mode knobs exposed by the run manager."""

    model_config = ConfigDict(extra="forbid")

    race_mode: RaceMode = "time_attack"
    gp_difficulties: tuple[GpDifficulty, ...] = Field(default_factory=default_gp_difficulties)
    include_x_cup: bool = False
    baseline_variant_count: int = Field(default=1, ge=1, le=8)
    x_cup_course_count: int = Field(
        default=X_CUP_COURSE.default_generated_count,
        ge=1,
        le=X_CUP_COURSE.max_generated_count,
    )
    x_cup_auto_regeneration: ManagedXCupAutoRegenerationConfig = Field(
        default_factory=ManagedXCupAutoRegenerationConfig
    )
    sampling_mode: TrackSamplingMode = "step_balanced"
    step_balance_update_episodes: int = Field(default=5, ge=1)
    step_balance_ema_alpha: float = Field(default=0.1, gt=0.0, le=1.0)
    step_balance_max_weight_scale: float = Field(default=5.0, ge=1.0)
    deficit_budget_uniform_fraction: float = Field(default=0.7, ge=0.0, le=1.0)
    deficit_budget_focus_sharpness: float = Field(default=1.0, ge=0.0)
    deficit_budget_ema_alpha: float = Field(default=0.02, gt=0.0, le=1.0)
    deficit_budget_weight_update_rollouts: int = Field(default=20, ge=1)
    deficit_budget_difficulty_metric: DeficitBudgetDifficultyMetric = "completion_ema"
    deficit_budget_warmup_min_episodes_per_course: int = Field(default=10, ge=0)
    deficit_budget_uniform_staleness_rotations: float = Field(default=2.0, ge=0.0)
    selected_course_ids: tuple[str, ...] = Field(default_factory=default_selected_course_ids)

    def active_course_count(self) -> int:
        """Return distinct reset targets exposed to course sampling."""

        return len(self.selected_course_ids) + self._active_x_cup_course_count()

    @model_validator(mode="after")
    def _validate_selected_course_ids(self) -> ManagedTracksConfig:
        if self.race_mode != "gp_race":
            self.gp_difficulties = ()
            self.include_x_cup = False
            self.baseline_variant_count = 1
            self.x_cup_auto_regeneration.enabled = False
        else:
            self.gp_difficulties = _unique_gp_difficulties(
                self.gp_difficulties or default_gp_difficulties()
            )
        if self.include_x_cup and self.race_mode != "gp_race":
            raise ValueError("tracks.include_x_cup=true requires tracks.race_mode=gp_race")
        if self.x_cup_auto_regeneration.enabled and not self.include_x_cup:
            raise ValueError("tracks.x_cup_auto_regeneration.enabled=true requires X Cup")
        if not self.selected_course_ids and not self.include_x_cup:
            raise ValueError("tracks.selected_course_ids must not be empty")
        if len(set(self.selected_course_ids)) != len(self.selected_course_ids):
            raise ValueError("tracks.selected_course_ids must not contain duplicates")
        unknown_ids = sorted(set(self.selected_course_ids) - built_in_course_id_set())
        if unknown_ids:
            joined = ", ".join(unknown_ids)
            raise ValueError(f"tracks.selected_course_ids contains unknown courses: {joined}")
        if self.active_course_count() <= 0:
            raise ValueError("tracks must expose at least one active course")
        return self

    @model_serializer(mode="wrap")
    def _serialize(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        data = handler(self)
        if self.race_mode != "gp_race":
            data.pop("gp_difficulties", None)
            data.pop("baseline_variant_count", None)
        return data

    def _active_x_cup_course_count(self) -> int:
        if not self.include_x_cup:
            return 0
        return self.x_cup_course_count


def _unique_gp_difficulties(
    difficulties: tuple[GpDifficulty, ...],
) -> tuple[GpDifficulty, ...]:
    seen: set[GpDifficulty] = set()
    ordered: list[GpDifficulty] = []
    for difficulty in difficulties:
        if difficulty in seen:
            continue
        seen.add(difficulty)
        ordered.append(difficulty)
    return tuple(ordered)
