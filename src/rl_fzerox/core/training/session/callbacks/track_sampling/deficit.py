# src/rl_fzerox/core/training/session/callbacks/track_sampling/deficit.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from random import Random

from rl_fzerox.core.envs.engine.reset.track_sampling import (
    TrackSamplingDeficitLane,
    TrackSamplingQueuedReset,
)
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
)
from rl_fzerox.core.training.session.callbacks.track_sampling.state import (
    DeficitBudgetCourseSchedulerState,
    DeficitBudgetSchedulerState,
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
    TrackStepStats,
    aggregate_runtime_entries,
)


@dataclass(frozen=True, slots=True)
class DeficitBudgetSettings:
    uniform_fraction: float
    focus_sharpness: float
    ema_alpha: float
    weight_update_rollouts: int
    difficulty_metric: str = "completion_ema"
    warmup_min_episodes_per_course: int = 10
    uniform_staleness_rotations: float = 2.0
    x_cup_generation_ema_alpha: float = 0.3


@dataclass(frozen=True, slots=True)
class DeficitQueueSettings:
    initial_queue_length: int = 2
    refill_low_watermark: int = 0
    minimum_refill_size: int = 1


DEFICIT_QUEUE_SETTINGS = DeficitQueueSettings()

_DEFICIT_LANES: tuple[TrackSamplingDeficitLane, ...] = ("uniform", "adaptive")


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
        self._deficit_steps = {
            lane: {course_key: 0.0 for course_key in self._course_keys} for lane in _DEFICIT_LANES
        }
        self._reserved_reset_steps = {
            lane: {course_key: 0.0 for course_key in self._course_keys} for lane in _DEFICIT_LANES
        }
        self._lane_deficit_steps = {lane: 0.0 for lane in _DEFICIT_LANES}
        self._lane_reserved_reset_steps = {lane: 0.0 for lane in _DEFICIT_LANES}
        self._uniform_assignment_count = 0
        self._last_uniform_assignment_index = {course_key: 0 for course_key in self._course_keys}
        self._rollout_steps = {course_key: 0 for course_key in self._course_keys}
        self._accounted_env_steps = {course_key: 0 for course_key in self._course_keys}
        self._scheduler_env_steps = {course_key: 0 for course_key in self._course_keys}
        self._ema_problem = {course_key: 0.0 for course_key in self._course_keys}
        self._rollouts_since_weight_update = 0
        self.update_count = 0
        restored = self._restore_state(restored_state)
        self._seed_problem_scores_from_restored_stats()
        self._recompute_target_weights()
        if restored and not self._restore_scheduler_state(restored_state):
            self._seed_legacy_deficit_steps_from_accounted_steps()

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
                focus_sharpness=settings_source.deficit_budget_focus_sharpness,
                ema_alpha=settings_source.deficit_budget_ema_alpha,
                weight_update_rollouts=settings_source.deficit_budget_weight_update_rollouts,
                difficulty_metric=settings_source.deficit_budget_difficulty_metric,
                warmup_min_episodes_per_course=(
                    settings_source.deficit_budget_warmup_min_episodes_per_course
                ),
                uniform_staleness_rotations=(
                    settings_source.deficit_budget_uniform_staleness_rotations
                ),
                x_cup_generation_ema_alpha=settings_source.x_cup_rotation.ema_alpha,
            ),
            restored_state=restored_state,
        )

    def add_rollout_budget(self, *, total_steps: int) -> None:
        """Add one rollout of target budget into each course deficit account."""

        steps = max(0, int(total_steps))
        uniform_fraction = _clamped_fraction(self._settings.uniform_fraction)
        adaptive_fraction = 1.0 - uniform_fraction
        uniform_share = 1.0 / len(self._course_keys)
        adaptive_fractions = self._adaptive_target_fractions()
        self._lane_deficit_steps["uniform"] += steps * uniform_fraction
        self._lane_deficit_steps["adaptive"] += steps * adaptive_fraction
        for lane in _DEFICIT_LANES:
            self._lane_reserved_reset_steps[lane] = 0.0
        for course_key in self._course_keys:
            self._deficit_steps["uniform"][course_key] += steps * uniform_fraction * uniform_share
            self._deficit_steps["adaptive"][course_key] += (
                steps * adaptive_fraction * adaptive_fractions[course_key]
            )
            for lane in _DEFICIT_LANES:
                self._reserved_reset_steps[lane][course_key] = 0.0
            self._rollout_steps[course_key] = 0

    def record_step_infos(self, infos: Sequence[Mapping[str, object]]) -> None:
        for info in infos:
            uses_alt_baseline = uses_alt_baseline_sample(info)
            track_id = info.get("track_id")
            if not isinstance(track_id, str):
                continue
            course_key = self._entry_course_keys.get(track_id, track_id)
            if course_key not in self._course_keys:
                continue
            self._scheduler_env_steps[course_key] += 1
            self._record_step_for_deficit_lane(course_key, info)
            self._rollout_steps[course_key] += 1
            if not uses_alt_baseline:
                self._accounted_env_steps[course_key] += 1

    def record_episodes(self, episodes: Sequence[Mapping[str, object]]) -> None:
        for episode in episodes:
            if uses_alt_baseline_sample(episode):
                continue
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

    def maybe_update_weights(self) -> bool:
        self._rollouts_since_weight_update += 1
        if self._rollouts_since_weight_update < self._settings.weight_update_rollouts:
            return False
        self._rollouts_since_weight_update = 0
        self.update_count += 1
        self._update_problem_scores()
        self._recompute_target_weights()
        return True

    def clear_reserved_assignments(self) -> None:
        """Forget queued-but-unconsumed reset reservations after queue replacement."""

        for lane in _DEFICIT_LANES:
            self._lane_reserved_reset_steps[lane] = 0.0
            for course_key in self._course_keys:
                self._reserved_reset_steps[lane][course_key] = 0.0

    def initial_queues(
        self,
        *,
        num_envs: int,
        queue_length: int,
        fallback_assignment_steps: float = 1.0,
    ) -> dict[int, tuple[TrackSamplingQueuedReset, ...]]:
        queues: dict[int, tuple[TrackSamplingQueuedReset, ...]] = {}
        for env_index in range(max(0, int(num_envs))):
            queues[env_index] = tuple(
                self.next_queued_reset(fallback_assignment_steps=fallback_assignment_steps)
                for _ in range(max(0, queue_length))
            )
        return queues

    def refill_queues(
        self,
        queue_lengths: Sequence[int],
        *,
        fallback_assignment_steps: float,
    ) -> dict[int, tuple[TrackSamplingQueuedReset, ...]]:
        refill_size = DEFICIT_QUEUE_SETTINGS.minimum_refill_size
        refills: dict[int, tuple[TrackSamplingQueuedReset, ...]] = {}
        for env_index, queue_length in enumerate(queue_lengths):
            if int(queue_length) > DEFICIT_QUEUE_SETTINGS.refill_low_watermark:
                continue
            refills[env_index] = tuple(
                self.next_queued_reset(fallback_assignment_steps=fallback_assignment_steps)
                for _ in range(refill_size)
            )
        return refills

    def next_course_key(self, *, fallback_assignment_steps: float = 1.0) -> str:
        """Return only the scheduled course id for legacy tests and callers."""

        return self.next_queued_reset(fallback_assignment_steps=fallback_assignment_steps).course_id

    def next_queued_reset(
        self,
        *,
        fallback_assignment_steps: float = 1.0,
    ) -> TrackSamplingQueuedReset:
        lane = self._next_deficit_lane()
        course_key = self._next_course_key_for_lane(lane)
        self._reserve_course_assignment(
            lane,
            course_key,
            assignment_cost=self._assignment_cost(
                course_key,
                fallback_assignment_steps=fallback_assignment_steps,
            ),
        )
        return TrackSamplingQueuedReset(course_id=course_key, deficit_lane=lane)

    def log_values(self) -> dict[str, float]:
        return {
            "track_sampling/lane/uniform_deficit_steps": self._lane_deficit_steps["uniform"],
            "track_sampling/lane/adaptive_deficit_steps": self._lane_deficit_steps["adaptive"],
            "track_sampling/lane/uniform_stale_course_count": float(
                self._uniform_stale_course_count()
            ),
            "track_sampling/lane/uniform_staleness_max_assignment_gap": float(
                self._uniform_staleness_max_assignment_gap()
            ),
        }

    def runtime_state(self) -> TrackSamplingRuntimeState:
        return TrackSamplingRuntimeState(
            sampling_mode="deficit_budget",
            action_repeat=self._action_repeat,
            update_episodes=self._settings.weight_update_rollouts,
            ema_alpha=self._settings.ema_alpha,
            max_weight_scale=max(1.0, self._settings.focus_sharpness),
            adaptive_completion_weight=1.0 - self._settings.uniform_fraction,
            adaptive_target_completion=1.0,
            adaptive_min_confidence_episodes=1,
            adaptive_confidence_scale=max(1.0, self._settings.focus_sharpness),
            deficit_budget_difficulty_metric=self._settings.difficulty_metric,
            deficit_budget_warmup_min_episodes_per_course=(
                self._settings.warmup_min_episodes_per_course
            ),
            update_count=self.update_count,
            episodes_since_update=self._rollouts_since_weight_update,
            entries=tuple(
                self._courses[course_key].runtime_entry(
                    stats=stats,
                    completed_frames=self._accounted_env_steps[course_key] * self._action_repeat,
                )
                for course_key, stats in sorted(self._stats.items())
            ),
            deficit_budget_scheduler=self._scheduler_state(),
        )

    def _restore_state(self, restored_state: TrackSamplingRuntimeState | None) -> bool:
        if restored_state is None or restored_state.sampling_mode != "deficit_budget":
            return False
        restored_entries = {
            entry.course_key: entry for entry in aggregate_runtime_entries(restored_state.entries)
        }
        restored_any = False
        for course_key, stats in self._stats.items():
            entry = restored_entries.get(course_key)
            if entry is None:
                continue
            restored_any = True
            stats.completed_frames = max(0, int(entry.completed_frames))
            self._accounted_env_steps[course_key] = (
                max(0, int(entry.completed_frames)) // self._action_repeat
            )
            self._scheduler_env_steps[course_key] = self._accounted_env_steps[course_key]
            stats.episode_count = max(0, int(entry.episode_count))
            stats.finished_episode_count = max(0, int(entry.finished_episode_count))
            stats.success_sample_count = max(0, int(entry.success_sample_count))
            stats.completion_sample_count = max(0, int(entry.completion_sample_count))
            stats.completion_fraction_total = max(0.0, float(entry.completion_fraction_total))
            stats.ema_episode_frames = entry.ema_episode_frames
            stats.ema_completion_fraction = entry.ema_completion_fraction
            stats.ema_finish_rate = entry.ema_finish_rate
            stats.current_problem_score = entry.current_problem_score
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
            self._courses[course_key] = self._courses[course_key].with_runtime_generation(entry)
        self.update_count = max(0, int(restored_state.update_count))
        self._rollouts_since_weight_update = max(0, int(restored_state.episodes_since_update))
        return restored_any

    def _restore_scheduler_state(
        self,
        restored_state: TrackSamplingRuntimeState | None,
    ) -> bool:
        if restored_state is None or restored_state.deficit_budget_scheduler is None:
            return False
        scheduler = restored_state.deficit_budget_scheduler
        restored_entries = {entry.course_key: entry for entry in scheduler.entries}
        if not set(self._course_keys).issubset(restored_entries):
            return False
        restored_any = False
        for course_key in self._course_keys:
            entry = restored_entries.get(course_key)
            if entry is None:
                continue
            restored_any = True
            self._deficit_steps["uniform"][course_key] = float(entry.uniform_deficit_steps)
            self._deficit_steps["adaptive"][course_key] = float(entry.adaptive_deficit_steps)
            self._scheduler_env_steps[course_key] = max(0, int(entry.scheduler_env_steps))
            self._last_uniform_assignment_index[course_key] = max(
                0,
                int(entry.last_uniform_assignment_index),
            )
        if not restored_any:
            return False
        self._lane_deficit_steps["uniform"] = float(scheduler.uniform_lane_deficit_steps)
        self._lane_deficit_steps["adaptive"] = float(scheduler.adaptive_lane_deficit_steps)
        self._uniform_assignment_count = max(0, int(scheduler.uniform_assignment_count))
        return True

    def _seed_legacy_deficit_steps_from_accounted_steps(self) -> None:
        """Rebuild deficit debt from old runtime states without scheduler accounts.

        LEGACY: remove once persisted runtime states without
        deficit_budget_scheduler are no longer supported. The fallback is
        intentionally isolated because aggregate measurement stats cannot
        reconstruct exact scheduler debt, especially when alt baselines
        contributed reset steps.
        """

        total_steps = sum(self._accounted_env_steps.values())
        if total_steps <= 0:
            return
        uniform_fraction = _clamped_fraction(self._settings.uniform_fraction)
        adaptive_fraction = 1.0 - uniform_fraction
        uniform_share = 1.0 / len(self._course_keys)
        adaptive_fractions = self._adaptive_target_fractions()
        self._lane_deficit_steps["uniform"] = 0.0
        self._lane_deficit_steps["adaptive"] = 0.0
        for course_key in self._course_keys:
            actual_steps = float(self._accounted_env_steps[course_key])
            self._deficit_steps["uniform"][course_key] = uniform_fraction * (
                total_steps * uniform_share - actual_steps
            )
            self._deficit_steps["adaptive"][course_key] = adaptive_fraction * (
                total_steps * adaptive_fractions[course_key] - actual_steps
            )
            self._lane_deficit_steps["uniform"] += self._deficit_steps["uniform"][course_key]
            self._lane_deficit_steps["adaptive"] += self._deficit_steps["adaptive"][course_key]
            self._scheduler_env_steps[course_key] = max(
                self._scheduler_env_steps[course_key],
                self._accounted_env_steps[course_key],
            )

    def _update_problem_scores(self) -> None:
        alpha = self._settings.ema_alpha
        for course_key, stats in self._stats.items():
            raw_problem = self._raw_problem_score(stats)
            self._ema_problem[course_key] = (1.0 - alpha) * self._ema_problem[
                course_key
            ] + alpha * raw_problem

    def _seed_problem_scores_from_restored_stats(self) -> None:
        if not any(stats.success_sample_count > 0 for stats in self._stats.values()):
            return
        has_persisted_problem = any(
            stats.current_problem_score > 0.0 for stats in self._stats.values()
        )
        for course_key, stats in self._stats.items():
            self._ema_problem[course_key] = max(
                0.0,
                float(stats.current_problem_score)
                if has_persisted_problem
                else self._raw_problem_score(stats),
            )

    def _scheduler_state(self) -> DeficitBudgetSchedulerState:
        return DeficitBudgetSchedulerState(
            uniform_lane_deficit_steps=self._lane_deficit_steps["uniform"],
            adaptive_lane_deficit_steps=self._lane_deficit_steps["adaptive"],
            uniform_assignment_count=self._uniform_assignment_count,
            entries=tuple(
                DeficitBudgetCourseSchedulerState(
                    course_key=course_key,
                    uniform_deficit_steps=self._deficit_steps["uniform"][course_key],
                    adaptive_deficit_steps=self._deficit_steps["adaptive"][course_key],
                    scheduler_env_steps=self._scheduler_env_steps[course_key],
                    last_uniform_assignment_index=self._last_uniform_assignment_index[course_key],
                )
                for course_key in self._course_keys
            ),
        )

    def _raw_problem_score(self, stats: TrackStepStats) -> float:
        metric = self._settings.difficulty_metric
        completion_problem = _completion_problem_score(stats)
        finish_problem = _finish_problem_score(stats)
        if metric == "finish_ema":
            return finish_problem
        if metric == "mixed":
            return max(completion_problem, finish_problem)
        return completion_problem

    def _recompute_target_weights(self) -> None:
        sharpness = max(0.0, float(self._settings.focus_sharpness))
        if self._in_warmup():
            for stats in self._stats.values():
                stats.current_problem_score = 1.0
                stats.current_weight = 1.0
            return
        for course_key, stats in self._stats.items():
            problem = max(0.0, self._ema_problem[course_key])
            stats.current_problem_score = problem
            stats.current_weight = 1.0 if sharpness <= 0.0 else problem**sharpness

    def _target_fractions(self) -> dict[str, float]:
        uniform_fraction = _clamped_fraction(self._settings.uniform_fraction)
        adaptive_fraction = 1.0 - uniform_fraction
        uniform_share = 1.0 / len(self._course_keys)
        adaptive_fractions = self._adaptive_target_fractions()
        return {
            course_key: uniform_fraction * uniform_share
            + adaptive_fraction * adaptive_fractions[course_key]
            for course_key in self._course_keys
        }

    def _adaptive_target_fractions(self) -> dict[str, float]:
        uniform_share = 1.0 / len(self._course_keys)
        total_weight = sum(
            self._stats[course_key].current_weight for course_key in self._course_keys
        )
        return {
            course_key: (
                uniform_share
                if total_weight <= 0.0
                else self._stats[course_key].current_weight / total_weight
            )
            for course_key in self._course_keys
        }

    def _record_step_for_deficit_lane(
        self,
        course_key: str,
        info: Mapping[str, object],
    ) -> None:
        lane = _deficit_lane_value(info.get("track_sampling_deficit_lane"))
        if lane is not None:
            self._lane_deficit_steps[lane] -= 1.0
            self._deficit_steps[lane][course_key] -= 1.0
            self._consume_reserved_step(lane, course_key)
            return
        uniform_fraction = _clamped_fraction(self._settings.uniform_fraction)
        self._lane_deficit_steps["uniform"] -= uniform_fraction
        self._lane_deficit_steps["adaptive"] -= 1.0 - uniform_fraction
        self._deficit_steps["uniform"][course_key] -= uniform_fraction
        self._deficit_steps["adaptive"][course_key] -= 1.0 - uniform_fraction

    def _consume_reserved_step(
        self,
        lane: TrackSamplingDeficitLane,
        course_key: str,
    ) -> None:
        course_reserved = self._reserved_reset_steps[lane][course_key]
        if course_reserved > 0.0:
            self._reserved_reset_steps[lane][course_key] = max(0.0, course_reserved - 1.0)
        lane_reserved = self._lane_reserved_reset_steps[lane]
        if lane_reserved > 0.0:
            self._lane_reserved_reset_steps[lane] = max(0.0, lane_reserved - 1.0)

    def _next_deficit_lane(self) -> TrackSamplingDeficitLane:
        uniform_fraction = _clamped_fraction(self._settings.uniform_fraction)
        if uniform_fraction >= 1.0:
            return "uniform"
        if uniform_fraction <= 0.0:
            return "adaptive"
        return max(
            _DEFICIT_LANES,
            key=lambda lane: (
                self._lane_deficit_steps[lane] - self._lane_reserved_reset_steps[lane],
                1.0 if lane == "uniform" else 0.0,
            ),
        )

    def _next_course_key_for_lane(self, lane: TrackSamplingDeficitLane) -> str:
        stale_course_key = self._stale_uniform_course_key() if lane == "uniform" else None
        if stale_course_key is not None:
            return stale_course_key
        return max(
            self._course_keys,
            key=lambda course_key: (
                self._deficit_steps[lane][course_key]
                - self._reserved_reset_steps[lane][course_key],
                self._rng.random() * 1e-9,
            ),
        )

    def _stale_uniform_course_key(self) -> str | None:
        max_gap = self._uniform_staleness_max_assignment_gap()
        if max_gap <= 0:
            return None
        candidates = tuple(
            course_key
            for course_key in self._course_keys
            if self._uniform_assignment_count - self._last_uniform_assignment_index[course_key]
            >= max_gap
        )
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda course_key: (
                self._uniform_assignment_count - self._last_uniform_assignment_index[course_key],
                self._deficit_steps["uniform"][course_key]
                - self._reserved_reset_steps["uniform"][course_key],
                self._rng.random() * 1e-9,
            ),
        )

    def _uniform_stale_course_count(self) -> int:
        max_gap = self._uniform_staleness_max_assignment_gap()
        if max_gap <= 0:
            return 0
        return sum(
            1
            for course_key in self._course_keys
            if self._uniform_assignment_count - self._last_uniform_assignment_index[course_key]
            >= max_gap
        )

    def _uniform_staleness_max_assignment_gap(self) -> int:
        rotations = max(0.0, float(self._settings.uniform_staleness_rotations))
        if rotations <= 0.0:
            return 0
        return max(len(self._course_keys), ceil(len(self._course_keys) * rotations))

    def _in_warmup(self) -> bool:
        minimum = max(0, int(self._settings.warmup_min_episodes_per_course))
        if minimum <= 0:
            return False
        return any(stats.success_sample_count < minimum for stats in self._stats.values())

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

    def _reserve_course_assignment(
        self,
        lane: TrackSamplingDeficitLane,
        course_key: str,
        *,
        assignment_cost: float,
    ) -> None:
        cost = max(1.0, float(assignment_cost))
        self._reserved_reset_steps[lane][course_key] += cost
        self._lane_reserved_reset_steps[lane] += cost
        if lane == "uniform":
            self._uniform_assignment_count += 1
            self._last_uniform_assignment_index[course_key] = self._uniform_assignment_count


def _clamped_fraction(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _deficit_lane_value(value: object) -> TrackSamplingDeficitLane | None:
    if value == "uniform":
        return "uniform"
    if value == "adaptive":
        return "adaptive"
    return None


def _needs_first_generation_backfill(entry: TrackSamplingRuntimeEntry) -> bool:
    return (
        entry.generated_course_slot is not None
        and entry.generated_course_generation == 1
        and entry.episode_count > 0
        and entry.generation_episode_count == 0
        and entry.generation_success_sample_count == 0
        and entry.generation_ema_completion_fraction is None
    )


def _completion_problem_score(stats: TrackStepStats) -> float:
    avg_completion = stats.ema_completion_fraction or 0.0
    return max(0.0, 1.0 - avg_completion)


def _finish_problem_score(stats: TrackStepStats) -> float:
    finish_rate = stats.ema_finish_rate
    if finish_rate is None:
        finish_rate = (
            0.0
            if stats.success_sample_count <= 0
            else stats.finished_episode_count / stats.success_sample_count
        )
    return max(0.0, 1.0 - max(0.0, min(1.0, finish_rate)))
