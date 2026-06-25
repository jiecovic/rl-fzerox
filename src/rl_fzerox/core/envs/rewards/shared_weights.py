# src/rl_fzerox/core/envs/rewards/shared_weights.py
"""Protocol for reward weights consumed by shared helper modules.

Some reward helpers operate on the common progress and event fields through this
small, explicit profile dependency.
"""
from __future__ import annotations

from typing import Protocol


class SharedRewardWeights(Protocol):
    """Common weight fields used by helper modules shared across reward profiles."""

    @property
    def progress_bucket_distance(self) -> float: ...

    @property
    def progress_bucket_reward(self) -> float: ...

    @property
    def progress_reward_interval_frames(self) -> int: ...

    @property
    def progress_track_distance_tolerance(self) -> float: ...

    @property
    def progress_speed_min_multiplier(self) -> float: ...

    @property
    def progress_speed_min_kph(self) -> float: ...

    @property
    def progress_speed_reference_kph(self) -> float: ...

    @property
    def progress_speed_max_kph(self) -> float: ...

    @property
    def progress_speed_max_multiplier(self) -> float: ...

    @property
    def progress_speed_curve_power(self) -> float: ...

    @property
    def position_progress_min_multiplier(self) -> float: ...

    @property
    def position_progress_max_multiplier(self) -> float: ...

    @property
    def boost_pad_reward_cannot_boost(self) -> float: ...

    @property
    def boost_pad_reward_can_boost(self) -> float: ...

    @property
    def boost_pad_reward_progress_window(self) -> float: ...

    @property
    def dirt_entry_penalty(self) -> float: ...

    @property
    def ice_entry_penalty(self) -> float: ...

    @property
    def lap_completion_bonus(self) -> float: ...

    @property
    def lap_position_scale(self) -> float: ...

    @property
    def airborne_landing_reward(self) -> float: ...

    @property
    def airborne_landing_grace_frames(self) -> int: ...

    @property
    def airborne_landing_min_peak_height(self) -> float: ...

    @property
    def failure_penalty(self) -> float: ...

    @property
    def truncation_penalty(self) -> float: ...
