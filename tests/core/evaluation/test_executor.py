# tests/core/evaluation/test_executor.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fzerox_emulator.arrays import ActionMask
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.evaluation import (
    EvaluationCourseTarget,
    FZeroXSingleCourseEpisodeExecutor,
)


@dataclass(slots=True)
class _FakeEnv:
    steps: list[tuple[float, bool, bool, dict[str, object]]]
    locked_courses: list[str | None]
    masks_requested: int = 0

    def set_locked_reset_course(self, course_id: str | None) -> None:
        self.locked_courses.append(course_id)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[ObservationValue, dict[str, object]]:
        _ = seed, options
        return _observation(), {"speed_kph": 0.0}

    def step(
        self,
        action: ActionValue,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        _ = action
        reward, terminated, truncated, info = self.steps.pop(0)
        return _observation(), reward, terminated, truncated, info

    def action_masks(self) -> ActionMask:
        self.masks_requested += 1
        return np.array([True, False], dtype=bool)


@dataclass(slots=True)
class _FakePolicyRunner:
    supports_action_masks: bool
    calls: list[dict[str, object]]
    reset_count: int = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict(
        self,
        observation: ObservationValue,
        *,
        deterministic: bool = True,
        action_masks: ActionMask | None = None,
        refresh: bool = True,
    ) -> ActionValue:
        self.calls.append(
            {
                "observation": observation,
                "deterministic": deterministic,
                "action_masks": action_masks,
                "refresh": refresh,
            }
        )
        return np.array([0], dtype=np.int64)


class _TransparentWrapper:
    """Small Gymnasium-like wrapper that does not expose env controls directly."""

    def __init__(self, env: _FakeEnv) -> None:
        self.env = env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[ObservationValue, dict[str, object]]:
        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        action: ActionValue,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self.env.step(action)

    def get_wrapper_attr(self, name: str) -> object:
        value = getattr(self.env, name)
        if value is None:
            raise AttributeError(name)
        return value


def test_fzerox_single_course_executor_maps_finished_episode_info() -> None:
    env = _FakeEnv(
        steps=[
            (1.5, False, False, {"speed_kph": 300.0, "boost_pad_entries": 1}),
            (
                2.5,
                True,
                False,
                {
                    "finished": True,
                    "termination_reason": "finished",
                    "track_course_id": "mute_city",
                    "track_course_name": "Mute City",
                    "track_gp_difficulty": "master",
                    "track_vehicle": "blue_falcon",
                    "track_engine_setting_raw_value": 90,
                    "race_time_ms": 86_123,
                    "position": 1,
                    "episode_completion_fraction": 1.0,
                    "race_laps_completed": 3,
                    "total_lap_count": 3,
                    "ko_star_count": 2,
                    "boost_pad_entries": 3,
                    "speed_kph": 900.0,
                },
            ),
        ],
        locked_courses=[],
    )
    policy = _FakePolicyRunner(supports_action_masks=True, calls=[])
    executor = FZeroXSingleCourseEpisodeExecutor(
        env=env,
        policy_runner=policy,
        max_env_steps=100,
    )

    result = executor.run_course(
        EvaluationCourseTarget(target_id="target", course_id="mute_city"),
        policy_mode="deterministic",
        seed=123,
    )

    assert env.locked_courses == ["mute_city"]
    assert env.masks_requested == 2
    assert policy.reset_count == 1
    assert policy.calls[0]["deterministic"] is True
    assert policy.calls[0]["refresh"] is False
    assert policy.calls[0]["action_masks"] is not None
    assert result.status == "finished"
    assert result.race_time_ms == 86_123
    assert result.position == 1
    assert result.laps_completed == 3
    assert result.env_steps == 2
    assert result.episode_return == 4.0
    assert result.engine_setting_raw_value == 90
    assert result.boost_pad_entries == 3
    assert result.average_speed == 600.0


def test_fzerox_single_course_executor_uses_controls_through_wrapped_env() -> None:
    base_env = _FakeEnv(
        steps=[
            (
                1.0,
                True,
                False,
                {
                    "finished": True,
                    "termination_reason": "finished",
                    "track_course_id": "mute_city",
                },
            )
        ],
        locked_courses=[],
    )
    policy = _FakePolicyRunner(supports_action_masks=True, calls=[])
    executor = FZeroXSingleCourseEpisodeExecutor(
        env=_TransparentWrapper(base_env),
        policy_runner=policy,
        max_env_steps=100,
    )

    result = executor.run_course(
        EvaluationCourseTarget(target_id="target", course_id="mute_city"),
        policy_mode="deterministic",
        seed=123,
    )

    assert base_env.locked_courses == ["mute_city"]
    assert base_env.masks_requested == 1
    assert result.status == "finished"


def test_fzerox_single_course_executor_truncates_at_step_limit() -> None:
    env = _FakeEnv(
        steps=[
            (
                0.1,
                False,
                False,
                {
                    "episode_completion_fraction": 0.25,
                    "termination_reason": None,
                    "speed_kph": 250.0,
                },
            )
        ],
        locked_courses=[],
    )
    policy = _FakePolicyRunner(supports_action_masks=False, calls=[])
    executor = FZeroXSingleCourseEpisodeExecutor(
        env=env,
        policy_runner=policy,
        max_env_steps=1,
    )

    result = executor.run_course(
        EvaluationCourseTarget(
            target_id="target",
            course_id="sand_ocean",
            course_name="Sand Ocean",
            engine_setting_raw_value=64,
        ),
        policy_mode="stochastic",
        seed=456,
    )

    assert env.masks_requested == 0
    assert policy.calls[0]["deterministic"] is False
    assert policy.calls[0]["action_masks"] is None
    assert result.status == "truncated"
    assert result.failure_reason == "truncated"
    assert result.completion_ratio == 0.25
    assert result.engine_setting_raw_value == 64


def test_fzerox_single_course_executor_reports_failed_gp_position_as_last() -> None:
    env = _FakeEnv(
        steps=[
            (
                1.0,
                True,
                False,
                {
                    "termination_reason": "falling_off_track",
                    "track_gp_difficulty": "master",
                    "position": 1,
                    "total_racers": 30,
                },
            )
        ],
        locked_courses=[],
    )
    policy = _FakePolicyRunner(supports_action_masks=False, calls=[])
    executor = FZeroXSingleCourseEpisodeExecutor(
        env=env,
        policy_runner=policy,
        max_env_steps=100,
    )

    result = executor.run_course(
        EvaluationCourseTarget(
            target_id="target",
            course_id="white_land",
            difficulty="master",
        ),
        policy_mode="deterministic",
        seed=789,
    )

    assert result.status == "crashed"
    assert result.position == 30


def _observation() -> ObservationValue:
    return np.zeros((1, 1, 3), dtype=np.uint8)
