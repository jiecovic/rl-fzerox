# src/rl_fzerox/core/training/session/callbacks/track_sampling/deficit/controller.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rl_fzerox.core.envs.engine.reset.track_sampling import (
    TrackSamplingDeficitLane,
    TrackSamplingQueuedReset,
)
from rl_fzerox.core.runtime_spec.schema import EnvConfig
from rl_fzerox.core.training.session.callbacks.track_sampling.courses import (
    ResolvedTrackSamplingCourses,
    resolve_track_sampling_courses_from_configs,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.deficit.ledger import (
    DeficitBudgetLedger,
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
        self._ledger = DeficitBudgetLedger(course_keys=self._course_keys, seed=seed)
        self._stats = {
            course_key: TrackStepStats(
                base_weight=self._courses[course_key].base_weight_mean,
                current_weight=1.0,
            )
            for course_key in self._course_keys
        }
        self._accounted_env_steps = {course_key: 0 for course_key in self._course_keys}
        self._ema_problem = {course_key: 0.0 for course_key in self._course_keys}
        self._rollouts_since_weight_update = 0
        self.update_count = 0
        self._restore_state(restored_state)
        self._seed_problem_scores_from_restored_stats()
        self._recompute_target_weights()
        scheduler_state = (
            None if restored_state is None else restored_state.deficit_budget_scheduler
        )
        self._ledger.restore_scheduler_state(scheduler_state)

    @classmethod
    def from_configs(
        cls,
        *,
        env_config: EnvConfig,
        restored_state: TrackSamplingRuntimeState | None = None,
    ) -> DeficitBudgetTrackSamplingController | None:
        configs = tuple(
            config
            for config in runtime_track_sampling_configs(env_config)
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
        self._ledger.add_rollout_budget(
            steps=steps,
            uniform_fraction=self._settings.uniform_fraction,
            adaptive_fractions=self._adaptive_target_fractions(),
        )

    def record_step_infos(self, infos: Sequence[Mapping[str, object]]) -> None:
        for info in infos:
            uses_alt_baseline = uses_alt_baseline_sample(info)
            track_id = info.get("track_id")
            if not isinstance(track_id, str):
                continue
            course_key = self._entry_course_keys.get(track_id, track_id)
            if course_key not in self._course_keys:
                continue
            self._ledger.record_scheduler_step(course_key)
            self._ledger.record_deficit_step(
                course_key=course_key,
                lane=_deficit_lane_value(info.get("track_sampling_deficit_lane")),
                uniform_fraction=self._settings.uniform_fraction,
            )
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

        self._ledger.clear_reserved_assignments()

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

    def next_queued_reset(
        self,
        *,
        fallback_assignment_steps: float = 1.0,
    ) -> TrackSamplingQueuedReset:
        lane = self._ledger.next_lane(uniform_fraction=self._settings.uniform_fraction)
        course_key = self._ledger.next_course_key(
            lane=lane,
            uniform_staleness_rotations=self._settings.uniform_staleness_rotations,
        )
        self._ledger.reserve_course_assignment(
            lane=lane,
            course_key=course_key,
            assignment_cost=self._assignment_cost(
                course_key,
                fallback_assignment_steps=fallback_assignment_steps,
            ),
        )
        return TrackSamplingQueuedReset(course_id=course_key, deficit_lane=lane)

    def log_values(self) -> dict[str, float]:
        return {
            "track_sampling/lane/uniform_deficit_steps": self._ledger.lane_deficit_steps("uniform"),
            "track_sampling/lane/adaptive_deficit_steps": self._ledger.lane_deficit_steps(
                "adaptive",
            ),
            "track_sampling/lane/uniform_stale_course_count": float(
                self._ledger.uniform_stale_course_count(
                    uniform_staleness_rotations=self._settings.uniform_staleness_rotations,
                )
            ),
            "track_sampling/lane/uniform_staleness_max_assignment_gap": float(
                self._ledger.uniform_staleness_max_assignment_gap(
                    uniform_staleness_rotations=self._settings.uniform_staleness_rotations,
                )
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
            deficit_budget_scheduler=self._ledger.state(),
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
            self._ledger.set_scheduler_env_steps(course_key, self._accounted_env_steps[course_key])
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
