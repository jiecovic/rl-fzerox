# src/rl_fzerox/core/training/session/callbacks/track_sampling/controller.py
from __future__ import annotations

from collections.abc import Mapping, Sequence

from rl_fzerox.core.runtime_spec.schema import CurriculumConfig, EnvConfig
from rl_fzerox.core.training.session.callbacks.track_sampling.episodes import (
    dynamic_step_balanced_sampling_configs,
    episode_completion_fraction,
    episode_finished,
    episode_frame_count,
    episode_track_id,
    sanitize_log_key,
    uses_dynamic_runtime_mode,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeEntry,
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
        track_base_weights: dict[str, float],
        sampling_mode: str = "step_balanced",
        action_repeat: int,
        update_episodes: int,
        ema_alpha: float,
        max_weight_scale: float,
        adaptive_completion_weight: float = 0.35,
        adaptive_target_completion: float = 0.9,
        adaptive_min_confidence_episodes: int = 24,
        adaptive_confidence_scale: float = 4.0,
        log_details: bool = False,
        track_course_keys: dict[str, str] | None = None,
        track_log_keys: dict[str, str] | None = None,
        track_labels: dict[str, str] | None = None,
        track_log_enabled: dict[str, bool] | None = None,
        restored_state: TrackSamplingRuntimeState | None = None,
    ) -> None:
        self._entry_base_weights = {
            track_id: float(weight) for track_id, weight in track_base_weights.items()
        }
        course_keys = {
            track_id: (
                (track_course_keys or {}).get(track_id)
                or (track_log_keys or {}).get(track_id)
                or track_id
            )
            for track_id in self._entry_base_weights
        }
        self._entry_course_keys = course_keys
        course_entry_ids: dict[str, list[str]] = {}
        course_labels: dict[str, str] = {}
        course_log_keys: dict[str, str] = {}
        course_log_enabled: dict[str, bool] = {}
        course_base_weight_sums: dict[str, float] = {}
        course_base_weight_counts: dict[str, int] = {}

        for track_id, base_weight in self._entry_base_weights.items():
            course_key = course_keys[track_id]
            course_entry_ids.setdefault(course_key, []).append(track_id)
            course_labels.setdefault(
                course_key,
                (track_labels or {}).get(track_id, course_key),
            )
            course_log_keys.setdefault(
                course_key,
                sanitize_log_key((track_log_keys or {}).get(track_id, course_key)),
            )
            course_log_enabled[course_key] = course_log_enabled.get(course_key, False) or (
                (track_log_enabled or {}).get(track_id, True)
            )
            course_base_weight_sums[course_key] = (
                course_base_weight_sums.get(course_key, 0.0) + base_weight
            )
            course_base_weight_counts[course_key] = course_base_weight_counts.get(course_key, 0) + 1

        self._course_entry_ids = {
            course_key: tuple(entry_ids) for course_key, entry_ids in course_entry_ids.items()
        }
        self._course_entry_base_totals = {
            course_key: sum(self._entry_base_weights[entry_id] for entry_id in entry_ids)
            for course_key, entry_ids in self._course_entry_ids.items()
        }
        self._course_log_keys = course_log_keys
        self._course_log_enabled = course_log_enabled
        self._course_labels = course_labels
        self._stats = {
            course_key: TrackStepStats(
                base_weight=course_base_weight_sums[course_key]
                / max(course_base_weight_counts[course_key], 1),
                current_weight=course_base_weight_sums[course_key]
                / max(course_base_weight_counts[course_key], 1),
            )
            for course_key in self._course_entry_ids
        }
        if not uses_dynamic_runtime_mode(sampling_mode):
            raise ValueError(f"Unsupported dynamic track sampling mode: {sampling_mode!r}")
        self._sampling_mode = sampling_mode
        self._action_repeat = max(1, int(action_repeat))
        self._update_episodes = max(1, int(update_episodes))
        self._ema_alpha = max(0.0, min(1.0, float(ema_alpha)))
        self._max_weight_scale = max(1.0, float(max_weight_scale))
        self._adaptive_completion_weight = max(0.0, float(adaptive_completion_weight))
        self._adaptive_target_completion = max(0.0, min(1.0, float(adaptive_target_completion)))
        self._adaptive_min_confidence_episodes = max(1, int(adaptive_min_confidence_episodes))
        self._adaptive_confidence_scale = max(1.0, float(adaptive_confidence_scale))
        self._log_details = log_details
        self._episodes_since_update = 0
        self.update_count = 0
        if self._restore_state(restored_state):
            self._compute_weights()

    @classmethod
    def from_configs(
        cls,
        *,
        env_config: EnvConfig,
        curriculum_config: CurriculumConfig,
        restored_state: TrackSamplingRuntimeState | None = None,
    ) -> StepBalancedTrackSamplingController | None:
        configs = dynamic_step_balanced_sampling_configs(env_config, curriculum_config)
        if not configs:
            return None

        base_weights: dict[str, float] = {}
        requested_course_keys: dict[str, str] = {}
        requested_log_keys: dict[str, str] = {}
        requested_log_enabled: dict[str, bool] = {}
        requested_labels: dict[str, str] = {}
        for config in configs:
            for entry in config.entries:
                base_weights.setdefault(entry.id, float(entry.weight))
                course_key = entry.course_id or entry.id
                requested_course_keys.setdefault(entry.id, course_key)
                requested_log_keys.setdefault(entry.id, course_key)
                requested_log_enabled.setdefault(entry.id, entry.log_per_course)
                requested_labels.setdefault(
                    entry.id,
                    entry.course_name or entry.course_id or entry.display_name or entry.id,
                )
        if len(set(requested_course_keys.values())) <= 1:
            return None

        settings = configs[0]
        return cls(
            track_base_weights=base_weights,
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
            log_details=settings.step_balance_log_details,
            track_course_keys=requested_course_keys,
            track_log_keys=requested_log_keys,
            track_labels=requested_labels,
            track_log_enabled=requested_log_enabled,
            restored_state=restored_state,
        )

    def record_episodes(self, episodes: Sequence[Mapping[str, object]]) -> dict[str, float] | None:
        recorded_count = 0
        for episode in episodes:
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
        return self._compute_weights()

    def log_values(self) -> dict[str, float]:
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
            if not self._course_log_enabled[course_key]:
                continue
            key = self._course_log_keys[course_key]
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
            update_count=self.update_count,
            episodes_since_update=self._episodes_since_update,
            entries=tuple(
                TrackSamplingRuntimeEntry(
                    track_id=course_key,
                    course_key=course_key,
                    label=self._course_labels[course_key],
                    base_weight=stats.base_weight,
                    current_weight=stats.current_weight,
                    completed_frames=stats.completed_frames,
                    episode_count=stats.episode_count,
                    finished_episode_count=stats.finished_episode_count,
                    success_sample_count=stats.success_sample_count,
                    ema_episode_frames=stats.ema_episode_frames,
                    ema_completion_fraction=stats.ema_completion_fraction,
                )
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
        if restored_state is None or not uses_dynamic_runtime_mode(restored_state.sampling_mode):
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
            stats.current_weight = max(0.0, float(entry.current_weight))
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
