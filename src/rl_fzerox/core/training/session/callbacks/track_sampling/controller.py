# src/rl_fzerox/core/training/session/callbacks/track_sampling/controller.py
from __future__ import annotations

from collections.abc import Mapping, Sequence

from rl_fzerox.core.runtime_spec.schema import CurriculumConfig, EnvConfig
from rl_fzerox.core.training.session.callbacks.track_sampling.courses import (
    ResolvedTrackSamplingCourses,
    resolve_track_sampling_courses_from_configs,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.episodes import (
    episode_completion_fraction,
    episode_finished,
    episode_frame_count,
    episode_track_id,
    runtime_track_sampling_configs,
    uses_alt_baseline_sample,
    uses_step_balance_scheduler,
    uses_track_sampling_runtime_mode,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeState,
    TrackStepStats,
    aggregate_runtime_entries,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.weights import (
    STEP_BALANCE_SCHEDULER_SETTINGS,
    CourseSamplingWeights,
    adaptive_target_bonus,
    blend_course_sampling_weights,
    distribute_course_weight,
    expected_episode_frames,
    target_frame_debt,
)


class StepBalancedTrackSamplingController:
    """Adapt reset weights to hit desired recent frame shares per course."""

    def __init__(
        self,
        *,
        resolved_courses: ResolvedTrackSamplingCourses,
        sampling_mode: str = "step_balanced",
        action_repeat: int,
        update_episodes: int,
        ema_alpha: float,
        max_weight_scale: float,
        adaptive_completion_weight: float = 0.35,
        adaptive_target_completion: float = 0.9,
        adaptive_min_confidence_episodes: int = 24,
        adaptive_confidence_scale: float = 4.0,
        x_cup_generation_ema_alpha: float = 0.3,
        log_details: bool = False,
        restored_state: TrackSamplingRuntimeState | None = None,
    ) -> None:
        resolved = resolved_courses
        self._entry_base_weights = dict(resolved.entry_base_weights)
        self._entry_course_keys = dict(resolved.entry_course_keys)
        self._courses = dict(resolved.courses)
        self._course_entry_ids = resolved.course_entry_ids
        self._course_entry_base_totals = resolved.course_entry_base_totals
        self._stats = {
            course_key: TrackStepStats(
                base_weight=course.base_weight_mean,
                current_weight=course.base_weight_mean,
            )
            for course_key, course in self._courses.items()
        }
        if not uses_track_sampling_runtime_mode(sampling_mode):
            raise ValueError(f"Unsupported track sampling runtime mode: {sampling_mode!r}")
        self._sampling_mode = sampling_mode
        self._action_repeat = max(1, int(action_repeat))
        self._update_episodes = max(1, int(update_episodes))
        self._ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self._max_weight_scale = max(1.0, float(max_weight_scale))
        self._adaptive_completion_weight = max(0.0, float(adaptive_completion_weight))
        self._adaptive_target_completion = max(0.0, min(1.0, float(adaptive_target_completion)))
        self._adaptive_min_confidence_episodes = max(1, int(adaptive_min_confidence_episodes))
        self._adaptive_confidence_scale = max(1.0, float(adaptive_confidence_scale))
        self._x_cup_generation_ema_alpha = max(0.0, min(1.0, float(x_cup_generation_ema_alpha)))
        self._log_details = log_details
        self._episodes_since_update = 0
        self.update_count = 0
        if self._restore_state(restored_state) and uses_step_balance_scheduler(
            self._sampling_mode,
        ):
            self._compute_weights()

    @classmethod
    def from_configs(
        cls,
        *,
        env_config: EnvConfig,
        curriculum_config: CurriculumConfig,
        restored_state: TrackSamplingRuntimeState | None = None,
    ) -> StepBalancedTrackSamplingController | None:
        configs = runtime_track_sampling_configs(env_config, curriculum_config)
        if not configs:
            return None

        resolved_courses = resolve_track_sampling_courses_from_configs(configs)
        if len(resolved_courses.courses) <= 1:
            return None

        settings = configs[0]
        return cls(
            resolved_courses=resolved_courses,
            sampling_mode=settings.sampling_mode,
            action_repeat=env_config.action_repeat,
            update_episodes=settings.step_balance_update_episodes,
            ema_alpha=settings.step_balance_ema_alpha,
            max_weight_scale=settings.step_balance_max_weight_scale,
            adaptive_completion_weight=settings.adaptive_step_balance_completion_weight,
            adaptive_target_completion=settings.adaptive_step_balance_target_completion,
            adaptive_min_confidence_episodes=(
                settings.adaptive_step_balance_min_confidence_episodes
            ),
            adaptive_confidence_scale=settings.adaptive_step_balance_confidence_scale,
            x_cup_generation_ema_alpha=settings.x_cup_rotation.ema_alpha,
            log_details=settings.step_balance_log_details,
            restored_state=restored_state,
        )

    def record_episodes(self, episodes: Sequence[Mapping[str, object]]) -> dict[str, float] | None:
        recorded_count = 0
        for episode in episodes:
            if uses_alt_baseline_sample(episode):
                continue
            track_id = episode_track_id(episode)
            if track_id is None:
                continue
            course_key = self._entry_course_keys.get(track_id, track_id)
            if course_key not in self._stats:
                continue
            frame_count = episode_frame_count(episode, action_repeat=self._action_repeat)
            if frame_count is None:
                continue
            self._stats[course_key].record_episode(
                frame_count,
                ema_alpha=self._ema_alpha,
                generation_ema_alpha=(
                    self._x_cup_generation_ema_alpha
                    if self._courses[course_key].generated.slot is not None
                    else None
                ),
                completion_fraction=episode_completion_fraction(episode),
                finished=episode_finished(episode),
            )
            recorded_count += 1

        if recorded_count == 0:
            return None

        self._episodes_since_update += recorded_count
        if self._episodes_since_update < self._update_episodes:
            return None

        self._episodes_since_update = 0
        self.update_count += 1
        if not uses_step_balance_scheduler(self._sampling_mode):
            return self.current_weights()
        return self._compute_weights()

    def log_values(self) -> dict[str, float]:
        if not uses_step_balance_scheduler(self._sampling_mode):
            return {}
        if not self._log_details:
            return {}

        total_weight = sum(stats.current_weight for stats in self._stats.values())
        sampling_weights = self._course_sampling_weights()
        total_target_frame_weight = sum(
            weights.target_frame_weight for weights in sampling_weights.values()
        )
        expected_frame_weights = {
            course_key: weights.reset_weight * weights.expected_episode_frames
            for course_key, weights in sampling_weights.items()
        }
        total_expected_frame_weight = sum(expected_frame_weights.values())
        values: dict[str, float] = {}
        for course_key, stats in self._stats.items():
            course = self._courses[course_key]
            if not course.log_enabled:
                continue
            key = course.log_key
            values[f"track_sampling/{key}/prob"] = (
                stats.current_weight / total_weight if total_weight > 0.0 else 0.0
            )
            weights = sampling_weights[course_key]
            values[f"track_sampling/{key}/target_frame_share"] = (
                0.0
                if total_target_frame_weight <= 0.0
                else weights.target_frame_weight / total_target_frame_weight
            )
            values[f"track_sampling/{key}/expected_frame_share"] = (
                0.0
                if total_expected_frame_weight <= 0.0
                else expected_frame_weights[course_key] / total_expected_frame_weight
            )
        return values

    def current_weights(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for course_key, stats in self._stats.items():
            weights.update(
                distribute_course_weight(
                    course_weight=stats.current_weight,
                    entry_ids=self._course_entry_ids[course_key],
                    entry_base_weights=self._entry_base_weights,
                    total_entry_base_weight=self._course_entry_base_totals[course_key],
                )
            )
        return weights

    def runtime_state(self) -> TrackSamplingRuntimeState:
        return TrackSamplingRuntimeState(
            sampling_mode=self._sampling_mode,
            action_repeat=self._action_repeat,
            update_episodes=self._update_episodes,
            ema_alpha=self._ema_alpha,
            max_weight_scale=self._max_weight_scale,
            adaptive_completion_weight=self._adaptive_completion_weight,
            adaptive_target_completion=self._adaptive_target_completion,
            adaptive_min_confidence_episodes=self._adaptive_min_confidence_episodes,
            adaptive_confidence_scale=self._adaptive_confidence_scale,
            deficit_budget_difficulty_metric="completion_ema",
            update_count=self.update_count,
            episodes_since_update=self._episodes_since_update,
            entries=tuple(
                self._courses[course_key].runtime_entry(stats=stats)
                for course_key, stats in sorted(self._stats.items())
            ),
        )

    def _compute_weights(self) -> dict[str, float]:
        sampling_weights = self._course_sampling_weights()
        raw_weights = {
            course_key: weights.reset_weight for course_key, weights in sampling_weights.items()
        }
        total_base_weight = self._total_base_weight()
        total_raw_weight = sum(raw_weights.values())
        if total_raw_weight <= 0.0:
            return self.current_weights()

        normalized = {
            track_id: weight * total_base_weight / total_raw_weight
            for track_id, weight in raw_weights.items()
        }
        for track_id, weight in normalized.items():
            self._stats[track_id].current_weight = weight
        return self.current_weights()

    def _course_sampling_weights(self) -> dict[str, CourseSamplingWeights]:
        reference_length = self._reference_episode_length()
        target_frame_weights = {
            course_key: stats.base_weight * self._target_frame_bonus(stats)
            for course_key, stats in self._stats.items()
        }
        total_target_frame_weight = sum(target_frame_weights.values())
        total_completed_frames = sum(stats.completed_frames for stats in self._stats.values())
        debt_weights = {
            course_key: self._course_sampling_weight(
                stats,
                target_frame_weight=target_frame_weights[course_key],
                total_target_frame_weight=total_target_frame_weight,
                total_completed_frames=total_completed_frames,
                reference_length=reference_length,
            )
            for course_key, stats in self._stats.items()
        }
        steady_state_weights = {
            course_key: self._steady_state_course_sampling_weight(
                stats,
                target_frame_weight=target_frame_weights[course_key],
                reference_length=reference_length,
            )
            for course_key, stats in self._stats.items()
        }
        if any(weights.frame_debt > 0.0 for weights in debt_weights.values()):
            return blend_course_sampling_weights(
                steady_state_weights=steady_state_weights,
                debt_weights=debt_weights,
                steady_state_share=STEP_BALANCE_SCHEDULER_SETTINGS.steady_state_probability_share,
            )
        return steady_state_weights

    def _course_sampling_weight(
        self,
        stats: TrackStepStats,
        *,
        target_frame_weight: float,
        total_target_frame_weight: float,
        total_completed_frames: int,
        reference_length: float,
    ) -> CourseSamplingWeights:
        episode_frames = expected_episode_frames(stats, fallback_frames=reference_length)
        frame_debt = target_frame_debt(
            stats,
            target_frame_weight=target_frame_weight,
            total_target_frame_weight=total_target_frame_weight,
            total_completed_frames=total_completed_frames,
        )
        reset_weight = 0.0 if frame_debt <= 0.0 else frame_debt / episode_frames
        return CourseSamplingWeights(
            target_frame_weight=target_frame_weight,
            expected_episode_frames=episode_frames,
            reset_weight=reset_weight,
            frame_debt=frame_debt,
        )

    def _steady_state_course_sampling_weight(
        self,
        stats: TrackStepStats,
        *,
        target_frame_weight: float,
        reference_length: float,
    ) -> CourseSamplingWeights:
        episode_frames = expected_episode_frames(stats, fallback_frames=reference_length)
        return CourseSamplingWeights(
            target_frame_weight=target_frame_weight,
            expected_episode_frames=episode_frames,
            reset_weight=target_frame_weight / episode_frames,
            frame_debt=0.0,
        )

    def _target_frame_bonus(self, stats: TrackStepStats) -> float:
        if self._sampling_mode != "adaptive_step_balanced":
            return 1.0
        return self._adaptive_completion_bonus(stats)

    def _total_base_weight(self) -> float:
        return sum(stats.base_weight for stats in self._stats.values())

    def _reference_episode_length(self) -> float:
        weighted_lengths = [
            (stats.base_weight, stats.ema_episode_frames)
            for stats in self._stats.values()
            if stats.ema_episode_frames is not None
        ]
        if not weighted_lengths:
            return 1.0
        total_weight = sum(weight for weight, _ in weighted_lengths)
        if total_weight <= 0.0:
            return 1.0
        return max(
            1.0,
            sum(weight * length for weight, length in weighted_lengths) / total_weight,
        )

    def _restore_state(self, restored_state: TrackSamplingRuntimeState | None) -> bool:
        if restored_state is None or not uses_track_sampling_runtime_mode(
            restored_state.sampling_mode,
        ):
            return False
        state_by_course_key = {
            entry.course_key: entry for entry in aggregate_runtime_entries(restored_state.entries)
        }
        restored_any = False
        for course_key, stats in self._stats.items():
            entry = state_by_course_key.get(course_key)
            if entry is None:
                continue
            stats.completed_frames = max(0, int(entry.completed_frames))
            stats.episode_count = max(0, int(entry.episode_count))
            stats.finished_episode_count = max(0, int(entry.finished_episode_count))
            stats.success_sample_count = max(0, int(entry.success_sample_count))
            stats.completion_sample_count = max(0, int(entry.completion_sample_count))
            stats.completion_fraction_total = max(0.0, float(entry.completion_fraction_total))
            stats.ema_episode_frames = (
                None
                if entry.ema_episode_frames is None
                else max(0.0, float(entry.ema_episode_frames))
            )
            stats.ema_completion_fraction = (
                None
                if entry.ema_completion_fraction is None
                else max(0.0, min(1.0, float(entry.ema_completion_fraction)))
            )
            stats.generation_episode_count = max(0, int(entry.generation_episode_count))
            stats.generation_finished_episode_count = max(
                0,
                int(entry.generation_finished_episode_count),
            )
            stats.generation_success_sample_count = max(
                0,
                int(entry.generation_success_sample_count),
            )
            stats.generation_ema_completion_fraction = (
                None
                if entry.generation_ema_completion_fraction is None
                else max(0.0, min(1.0, float(entry.generation_ema_completion_fraction)))
            )
            stats.current_weight = max(0.0, float(entry.current_weight))
            self._courses[course_key] = self._courses[course_key].with_runtime_generation(entry)
            restored_any = True
        self._episodes_since_update = max(0, int(restored_state.episodes_since_update))
        self.update_count = max(0, int(restored_state.update_count))
        return restored_any

    def _adaptive_completion_bonus(self, stats: TrackStepStats) -> float:
        return adaptive_target_bonus(
            sampling_mode=self._sampling_mode,
            max_weight_scale=self._max_weight_scale,
            completion_weight=self._adaptive_completion_weight,
            target_completion=self._adaptive_target_completion,
            update_episodes=self._update_episodes,
            min_confidence_episodes=self._adaptive_min_confidence_episodes,
            confidence_scale=self._adaptive_confidence_scale,
            completion_fraction=stats.ema_completion_fraction,
            finished_episode_count=stats.finished_episode_count,
            success_sample_count=stats.success_sample_count,
        )
