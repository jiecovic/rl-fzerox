# src/rl_fzerox/core/training/session/callbacks/track_sampling/deficit.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random

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
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
    TrackStepStats,
    aggregate_runtime_entries,
)


@dataclass(frozen=True, slots=True)
class DeficitBudgetSettings:
    uniform_fraction: float
    min_weight: float
    max_weight: float
    ema_alpha: float
    weight_update_rollouts: int
    x_cup_generation_ema_alpha: float = 0.3


@dataclass(frozen=True, slots=True)
class DeficitQueueSettings:
    initial_queue_length: int = 8
    refill_low_watermark: int = 4
    minimum_refill_size: int = 16


DEFICIT_QUEUE_SETTINGS = DeficitQueueSettings()

_CRASH_TERMINATION_REASONS = frozenset(
    {
        "crashed",
        "damage",
        "depleted",
        "energy_depleted",
        "falling_off_track",
        "off_track",
        "retired",
        "spinning_out",
    }
)


class DeficitBudgetTrackSamplingController:
    """Assign reset courses from deterministic step-budget deficit accounts."""

    def __init__(
        self,
        *,
        resolved_courses: ResolvedTrackSamplingCourses,
        action_repeat: int,
        settings: DeficitBudgetSettings,
        restored_state: TrackSamplingRuntimeState | None = None,
        seed: int = 0,
    ) -> None:
        resolved = resolved_courses
        self._entry_course_keys = dict(resolved.entry_course_keys)
        self._courses = dict(resolved.courses)
        self._course_entry_ids = resolved.course_entry_ids
        self._course_keys = tuple(sorted(self._course_entry_ids))
        self._settings = settings
        self._action_repeat = max(1, int(action_repeat))
        self._rng = Random(seed)
        self._stats = {
            course_key: TrackStepStats(
                base_weight=self._courses[course_key].base_weight_mean,
                current_weight=1.0,
            )
            for course_key in self._course_keys
        }
        self._deficit_steps = {course_key: 0.0 for course_key in self._course_keys}
        self._reserved_reset_steps = {course_key: 0.0 for course_key in self._course_keys}
        self._rollout_steps = {course_key: 0 for course_key in self._course_keys}
        self._accounted_env_steps = {course_key: 0 for course_key in self._course_keys}
        self._ema_problem = {course_key: 0.0 for course_key in self._course_keys}
        self._best_completion = {course_key: 0.0 for course_key in self._course_keys}
        self._crash_episode_count = {course_key: 0 for course_key in self._course_keys}
        self._rollouts_since_weight_update = 0
        self.update_count = 0
        self._restore_state(restored_state)
        self._seed_problem_scores_from_restored_stats()
        self._recompute_target_weights()
        self._rebuild_deficits_from_completed_steps()

    @classmethod
    def from_configs(
        cls,
        *,
        env_config: EnvConfig,
        curriculum_config: CurriculumConfig,
        restored_state: TrackSamplingRuntimeState | None = None,
    ) -> DeficitBudgetTrackSamplingController | None:
        configs = tuple(
            config
            for config in runtime_track_sampling_configs(env_config, curriculum_config)
            if config.sampling_mode == "deficit_budget"
        )
        if not configs:
            return None

        resolved_courses = resolve_track_sampling_courses_from_configs(configs)
        if len(resolved_courses.courses) <= 1:
            return None

        settings_source = configs[0]
        return cls(
            resolved_courses=resolved_courses,
            action_repeat=env_config.action_repeat,
            settings=DeficitBudgetSettings(
                uniform_fraction=settings_source.deficit_budget_uniform_fraction,
                min_weight=settings_source.deficit_budget_min_weight,
                max_weight=settings_source.deficit_budget_max_weight,
                ema_alpha=settings_source.deficit_budget_ema_alpha,
                weight_update_rollouts=settings_source.deficit_budget_weight_update_rollouts,
                x_cup_generation_ema_alpha=settings_source.x_cup_rotation.ema_alpha,
            ),
            restored_state=restored_state,
        )

    def add_rollout_budget(self, *, total_steps: int) -> None:
        """Add one rollout of target budget into each course deficit account."""

        target_fractions = self._target_fractions()
        for course_key, fraction in target_fractions.items():
            self._deficit_steps[course_key] += max(0, int(total_steps)) * fraction
            self._reserved_reset_steps[course_key] = 0.0
            self._rollout_steps[course_key] = 0

    def record_step_infos(self, infos: Sequence[Mapping[str, object]]) -> None:
        for info in infos:
            track_id = info.get("track_id")
            if not isinstance(track_id, str):
                continue
            course_key = self._entry_course_keys.get(track_id, track_id)
            if course_key not in self._deficit_steps:
                continue
            self._deficit_steps[course_key] -= 1.0
            self._rollout_steps[course_key] += 1
            self._accounted_env_steps[course_key] += 1

    def record_episodes(self, episodes: Sequence[Mapping[str, object]]) -> None:
        for episode in episodes:
            track_id = episode_track_id(episode)
            if track_id is None:
                continue
            course_key = self._entry_course_keys.get(track_id, track_id)
            stats = self._stats.get(course_key)
            if stats is None:
                continue
            frame_count = episode_frame_count(episode, action_repeat=self._action_repeat)
            if frame_count is None:
                continue
            finished = episode_finished(episode)
            stats.record_episode(
                frame_count,
                ema_alpha=self._settings.ema_alpha,
                generation_ema_alpha=(
                    self._settings.x_cup_generation_ema_alpha
                    if self._courses[course_key].generated.slot is not None
                    else None
                ),
                completion_fraction=episode_completion_fraction(episode),
                finished=finished,
            )
            if _episode_crashed(episode):
                self._crash_episode_count[course_key] += 1

    def maybe_update_weights(self) -> None:
        self._rollouts_since_weight_update += 1
        if self._rollouts_since_weight_update < self._settings.weight_update_rollouts:
            return
        self._rollouts_since_weight_update = 0
        self.update_count += 1
        self._update_problem_scores()
        self._recompute_target_weights()
        self._rebuild_deficits_from_completed_steps()

    def initial_queues(
        self,
        *,
        num_envs: int,
        queue_length: int,
        fallback_assignment_steps: float = 1.0,
    ) -> dict[int, tuple[str, ...]]:
        queues: dict[int, tuple[str, ...]] = {}
        for env_index in range(max(0, int(num_envs))):
            queues[env_index] = tuple(
                self.next_course_key(fallback_assignment_steps=fallback_assignment_steps)
                for _ in range(max(0, queue_length))
            )
        return queues

    def refill_queues(
        self,
        queue_lengths: Sequence[int],
        *,
        fallback_assignment_steps: float,
    ) -> dict[int, tuple[str, ...]]:
        refill_size = DEFICIT_QUEUE_SETTINGS.minimum_refill_size
        refills: dict[int, tuple[str, ...]] = {}
        for env_index, queue_length in enumerate(queue_lengths):
            if int(queue_length) > DEFICIT_QUEUE_SETTINGS.refill_low_watermark:
                continue
            refills[env_index] = tuple(
                self.next_course_key(fallback_assignment_steps=fallback_assignment_steps)
                for _ in range(refill_size)
            )
        return refills

    def next_course_key(self, *, fallback_assignment_steps: float = 1.0) -> str:
        course_key = max(
            self._course_keys,
            key=lambda course_key: (
                self._deficit_steps[course_key] - self._reserved_reset_steps[course_key],
                self._rng.random() * 1e-9,
            ),
        )
        self._reserve_course_assignment(
            course_key,
            assignment_cost=self._assignment_cost(
                course_key,
                fallback_assignment_steps=fallback_assignment_steps,
            ),
        )
        return course_key

    def log_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        target_fractions = self._target_fractions()
        rollout_total = sum(self._rollout_steps.values())
        for course_key, stats in self._stats.items():
            course = self._courses[course_key]
            if not course.log_enabled:
                continue
            key = course.log_key
            success_count = max(stats.success_sample_count, 1)
            values[f"track_sampling/{key}/target_step_share"] = target_fractions[course_key]
            values[f"track_sampling/{key}/actual_step_share"] = (
                0.0 if rollout_total <= 0 else self._rollout_steps[course_key] / rollout_total
            )
            values[f"track_sampling/{key}/deficit_steps"] = self._deficit_steps[course_key]
            values[f"track_sampling/{key}/reserved_reset_steps"] = self._reserved_reset_steps[
                course_key
            ]
            values[f"track_sampling/{key}/problem_ema"] = self._ema_problem[course_key]
            values[f"track_sampling/{key}/adaptive_weight"] = stats.current_weight
            values[f"track_sampling/{key}/finish_rate"] = (
                stats.finished_episode_count / success_count
            )
            values[f"track_sampling/{key}/crash_rate"] = (
                self._crash_episode_count[course_key] / success_count
            )
            values[f"track_sampling/{key}/avg_completion"] = stats.ema_completion_fraction or 0.0
        return values

    def runtime_state(self) -> TrackSamplingRuntimeState:
        return TrackSamplingRuntimeState(
            sampling_mode="deficit_budget",
            action_repeat=self._action_repeat,
            update_episodes=self._settings.weight_update_rollouts,
            ema_alpha=self._settings.ema_alpha,
            max_weight_scale=self._settings.max_weight,
            adaptive_completion_weight=1.0 - self._settings.uniform_fraction,
            adaptive_target_completion=1.0,
            adaptive_min_confidence_episodes=1,
            adaptive_confidence_scale=self._settings.max_weight,
            update_count=self.update_count,
            episodes_since_update=self._rollouts_since_weight_update,
            entries=tuple(
                self._courses[course_key].runtime_entry(
                    stats=stats,
                    completed_frames=self._accounted_env_steps[course_key] * self._action_repeat,
                )
                for course_key, stats in sorted(self._stats.items())
            ),
        )

    def _restore_state(self, restored_state: TrackSamplingRuntimeState | None) -> None:
        if restored_state is None or restored_state.sampling_mode != "deficit_budget":
            return
        restored_entries = {
            entry.course_key: entry for entry in aggregate_runtime_entries(restored_state.entries)
        }
        for course_key, stats in self._stats.items():
            entry = restored_entries.get(course_key)
            if entry is None:
                continue
            stats.completed_frames = max(0, int(entry.completed_frames))
            self._accounted_env_steps[course_key] = (
                max(0, int(entry.completed_frames)) // self._action_repeat
            )
            stats.episode_count = max(0, int(entry.episode_count))
            stats.finished_episode_count = max(0, int(entry.finished_episode_count))
            stats.success_sample_count = max(0, int(entry.success_sample_count))
            stats.ema_episode_frames = entry.ema_episode_frames
            stats.ema_completion_fraction = entry.ema_completion_fraction
            stats.generation_episode_count = max(0, int(entry.generation_episode_count))
            stats.generation_finished_episode_count = max(
                0,
                int(entry.generation_finished_episode_count),
            )
            stats.generation_success_sample_count = max(
                0,
                int(entry.generation_success_sample_count),
            )
            stats.generation_ema_completion_fraction = entry.generation_ema_completion_fraction
            stats.current_weight = max(0.0, float(entry.current_weight))
            if _needs_first_generation_backfill(entry):
                stats.generation_episode_count = stats.episode_count
                stats.generation_finished_episode_count = stats.finished_episode_count
                stats.generation_success_sample_count = stats.success_sample_count
                stats.generation_ema_completion_fraction = stats.ema_completion_fraction
            if entry.ema_completion_fraction is not None:
                self._best_completion[course_key] = max(0.0, float(entry.ema_completion_fraction))
            self._courses[course_key] = self._courses[course_key].with_runtime_generation(entry)
        self.update_count = max(0, int(restored_state.update_count))
        self._rollouts_since_weight_update = max(0, int(restored_state.episodes_since_update))

    def _update_problem_scores(self) -> None:
        alpha = self._settings.ema_alpha
        for course_key, stats in self._stats.items():
            raw_problem = self._raw_problem_score(course_key, stats)
            self._ema_problem[course_key] = (1.0 - alpha) * self._ema_problem[
                course_key
            ] + alpha * raw_problem

    def _seed_problem_scores_from_restored_stats(self) -> None:
        if not any(stats.success_sample_count > 0 for stats in self._stats.values()):
            return
        for course_key, stats in self._stats.items():
            self._ema_problem[course_key] = self._raw_problem_score(course_key, stats)

    def _raw_problem_score(self, course_key: str, stats: TrackStepStats) -> float:
        avg_completion = stats.ema_completion_fraction or 0.0
        self._best_completion[course_key] = max(
            self._best_completion[course_key],
            avg_completion,
        )
        success_count = max(stats.success_sample_count, 1)
        crash_rate = self._crash_episode_count[course_key] / success_count
        finish_rate = stats.finished_episode_count / success_count
        regression = max(0.0, self._best_completion[course_key] - avg_completion)
        return (
            1.5 * crash_rate
            + 1.0 * (1.0 - finish_rate)
            + 1.0 * (1.0 - avg_completion)
            + 0.5 * regression
        )

    def _recompute_target_weights(self) -> None:
        values = tuple(self._ema_problem[course_key] for course_key in self._course_keys)
        minimum = min(values)
        maximum = max(values)
        if maximum <= minimum:
            normalized = {course_key: 0.0 for course_key in self._course_keys}
        else:
            spread = maximum - minimum
            normalized = {
                course_key: (self._ema_problem[course_key] - minimum) / spread
                for course_key in self._course_keys
            }
        for course_key, stats in self._stats.items():
            stats.current_weight = (
                self._settings.min_weight
                + (self._settings.max_weight - self._settings.min_weight) * normalized[course_key]
            )

    def _target_fractions(self) -> dict[str, float]:
        uniform_share = 1.0 / len(self._course_keys)
        total_weight = sum(
            self._stats[course_key].current_weight for course_key in self._course_keys
        )
        adaptive_fractions = {
            course_key: (
                uniform_share
                if total_weight <= 0.0
                else self._stats[course_key].current_weight / total_weight
            )
            for course_key in self._course_keys
        }
        uniform_fraction = self._settings.uniform_fraction
        adaptive_fraction = 1.0 - uniform_fraction
        return {
            course_key: uniform_fraction * uniform_share
            + adaptive_fraction * adaptive_fractions[course_key]
            for course_key in self._course_keys
        }

    def _rebuild_deficits_from_completed_steps(self) -> None:
        """Derive live scheduler debt from durable per-course step totals."""

        total_steps = sum(self._accounted_env_steps.values())
        if total_steps <= 0:
            return
        target_fractions = self._target_fractions()
        for course_key in self._course_keys:
            target_steps = total_steps * target_fractions[course_key]
            self._deficit_steps[course_key] = target_steps - self._accounted_env_steps[course_key]
            self._reserved_reset_steps[course_key] = 0.0

    def _assignment_cost(
        self,
        course_key: str,
        *,
        fallback_assignment_steps: float,
    ) -> float:
        stats = self._stats[course_key]
        if stats.ema_episode_frames is not None and stats.ema_episode_frames > 0.0:
            return max(1.0, float(stats.ema_episode_frames) / self._action_repeat)

        known_costs = tuple(
            float(candidate.ema_episode_frames) / self._action_repeat
            for candidate in self._stats.values()
            if candidate.ema_episode_frames is not None and candidate.ema_episode_frames > 0.0
        )
        if known_costs:
            return max(1.0, sum(known_costs) / len(known_costs))
        return max(1.0, float(fallback_assignment_steps))

    def _reserve_course_assignment(self, course_key: str, *, assignment_cost: float) -> None:
        self._reserved_reset_steps[course_key] += max(1.0, float(assignment_cost))


def _episode_crashed(episode: Mapping[str, object]) -> bool:
    reason = episode.get("termination_reason")
    if not isinstance(reason, str):
        return False
    return reason in _CRASH_TERMINATION_REASONS


def _needs_first_generation_backfill(entry: TrackSamplingRuntimeEntry) -> bool:
    return (
        entry.generated_course_slot is not None
        and entry.generated_course_generation == 1
        and entry.episode_count > 0
        and entry.generation_episode_count == 0
        and entry.generation_success_sample_count == 0
        and entry.generation_ema_completion_fraction is None
    )
