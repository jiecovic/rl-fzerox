# src/rl_fzerox/core/evaluation/executor.py
"""Concrete single-course evaluation executor for the training environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fzerox_emulator.arrays import ActionMask
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.evaluation.env_control import (
    action_masks as env_action_masks,
)
from rl_fzerox.core.evaluation.env_control import (
    set_locked_reset_course,
)
from rl_fzerox.core.evaluation.models import (
    CourseResultStatus,
    EvaluationCourseResult,
    EvaluationCourseTarget,
    EvaluationPolicyMode,
)
from rl_fzerox.core.runtime_info import (
    bool_info,
    optional_float_info,
    optional_int_info,
    optional_str_info,
)

DEFAULT_GP_RACER_COUNT = 30


class _SingleCourseEnv(Protocol):
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[ObservationValue, dict[str, object]]: ...

    def step(
        self,
        action: ActionValue,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]: ...


class _PolicyRunner(Protocol):
    @property
    def supports_action_masks(self) -> bool: ...

    def reset(self) -> None: ...

    def predict(
        self,
        observation: ObservationValue,
        *,
        deterministic: bool = True,
        action_masks: ActionMask | None = None,
        refresh: bool = True,
    ) -> ActionValue: ...


@dataclass(slots=True)
class FZeroXSingleCourseEpisodeExecutor:
    """Run one course episode through ``FZeroXEnv`` and ``PolicyRunner``.

    The executor owns no checkpoint loading or manager state. Callers provide a
    frozen policy runner and an env configured with the target track pool; the
    executor only locks the next reset to the requested course and records the
    terminal episode summary.
    """

    env: _SingleCourseEnv
    policy_runner: _PolicyRunner
    max_env_steps: int

    def run_course(
        self,
        target: EvaluationCourseTarget,
        *,
        policy_path: Path,
        policy_mode: EvaluationPolicyMode,
        seed: int,
    ) -> EvaluationCourseResult:
        _ = policy_path
        if self.max_env_steps < 1:
            raise ValueError(f"max_env_steps must be at least 1, got {self.max_env_steps}")

        set_locked_reset_course(self.env, target.course_id)
        self.policy_runner.reset()
        observation, info = self.env.reset(seed=seed)
        stats = _EpisodeStats()

        terminated = False
        truncated = False
        for _ in range(self.max_env_steps):
            action = self.policy_runner.predict(
                observation,
                deterministic=policy_mode == "deterministic",
                action_masks=env_action_masks(self.env)
                if self.policy_runner.supports_action_masks
                else None,
                refresh=False,
            )
            observation, reward, terminated, truncated, info = self.env.step(action)
            stats.record(info=info, reward=reward)
            if terminated or truncated:
                break

        status = _course_status(info, terminated=terminated, truncated=truncated)
        if not terminated and not truncated:
            status = "truncated"
        return stats.course_result(target=target, status=status, seed=seed, info=info)


@dataclass(slots=True)
class _EpisodeStats:
    env_steps: int = 0
    episode_return: float = 0.0
    speed_sum: float = 0.0
    speed_count: int = 0

    def record(self, *, info: dict[str, object], reward: float | None = None) -> None:
        if reward is not None:
            self.env_steps += 1
            self.episode_return += float(reward)
        speed = optional_float_info(info, "speed_kph")
        if speed is not None:
            self.speed_sum += speed
            self.speed_count += 1

    def course_result(
        self,
        *,
        target: EvaluationCourseTarget,
        status: CourseResultStatus,
        seed: int,
        info: dict[str, object],
    ) -> EvaluationCourseResult:
        engine_setting_raw_value = optional_int_info(info, "track_engine_setting_raw_value")
        if engine_setting_raw_value is None:
            engine_setting_raw_value = target.engine_setting_raw_value
        laps_completed = optional_int_info(info, "race_laps_completed")
        if laps_completed is None:
            laps_completed = optional_int_info(info, "laps_completed")
        return EvaluationCourseResult(
            course_id=optional_str_info(info, "track_course_id", non_empty=True)
            or target.course_id,
            course_name=optional_str_info(info, "track_course_name", non_empty=True)
            or target.course_name,
            cup_id=target.cup_id,
            difficulty=optional_str_info(info, "track_gp_difficulty", non_empty=True)
            or target.difficulty,
            vehicle_id=optional_str_info(info, "track_vehicle", non_empty=True)
            or target.vehicle_id,
            seed=seed,
            status=status,
            race_time_ms=optional_int_info(info, "race_time_ms"),
            position=_result_position(info, target=target, status=status),
            completion_ratio=optional_float_info(info, "episode_completion_fraction"),
            laps_completed=laps_completed,
            total_laps=optional_int_info(info, "total_lap_count"),
            env_steps=self.env_steps,
            episode_length_steps=self.env_steps,
            episode_return=self.episode_return,
            engine_setting_raw_value=engine_setting_raw_value,
            ko_stars=optional_int_info(info, "ko_star_count"),
            failure_reason=_failure_reason(info, status=status),
            boost_pad_entries=optional_int_info(info, "boost_pad_entries"),
            average_speed=(
                None if self.speed_count <= 0 else self.speed_sum / float(self.speed_count)
            ),
        )


def _course_status(
    info: dict[str, object],
    *,
    terminated: bool,
    truncated: bool,
) -> CourseResultStatus:
    if truncated:
        return "truncated"
    if bool_info(info, "finished") or info.get("termination_reason") == "finished":
        return "finished"
    if bool_info(info, "retired") or info.get("termination_reason") in {
        "retired",
        "energy_depleted",
    }:
        return "retired"
    if bool_info(info, "crashed") or info.get("termination_reason") in {
        "crashed",
        "falling_off_track",
        "spinning_out",
    }:
        return "crashed"
    return "failed" if terminated else "truncated"


def _failure_reason(info: dict[str, object], *, status: CourseResultStatus) -> str | None:
    if status == "finished":
        return None
    reason = info.get("termination_reason") or info.get("truncation_reason")
    if isinstance(reason, str) and reason:
        return reason
    return status


def _result_position(
    info: dict[str, object],
    *,
    target: EvaluationCourseTarget,
    status: CourseResultStatus,
) -> int | None:
    position = optional_int_info(info, "position")
    if status == "finished":
        return position
    if _is_gp_result(info, target=target):
        return DEFAULT_GP_RACER_COUNT
    return position


def _is_gp_result(info: dict[str, object], *, target: EvaluationCourseTarget) -> bool:
    return (
        target.difficulty is not None
        or optional_str_info(info, "track_gp_difficulty", non_empty=True) is not None
    )
