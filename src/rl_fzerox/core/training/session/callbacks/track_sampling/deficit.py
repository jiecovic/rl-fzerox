# src/rl_fzerox/core/training/session/callbacks/track_sampling/deficit.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random

from rl_fzerox.core.runtime_spec.schema import CurriculumConfig, EnvConfig
from rl_fzerox.core.training.session.callbacks.track_sampling.episodes import (
    episode_completion_fraction,
    episode_finished,
    episode_frame_count,
    episode_track_id,
    runtime_track_sampling_configs,
    sanitize_log_key,
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
        track_base_weights: dict[str, float],
        action_repeat: int,
        settings: DeficitBudgetSettings,
        track_course_keys: dict[str, str],
        track_log_keys: dict[str, str],
        track_labels: dict[str, str],
        track_log_enabled: dict[str, bool],
        track_generated_course_slots: dict[str, int] | None = None,
        track_generated_course_generations: dict[str, int] | None = None,
        track_generated_course_ids: dict[str, str] | None = None,
        track_generated_course_names: dict[str, str] | None = None,
        track_generated_course_hashes: dict[str, str] | None = None,
        track_generated_course_seeds: dict[str, int] | None = None,
        track_generated_course_segment_counts: dict[str, int] | None = None,
        track_generated_course_lengths: dict[str, float] | None = None,
        restored_state: TrackSamplingRuntimeState | None = None,
        seed: int = 0,
    ) -> None:
        self._entry_course_keys = dict(track_course_keys)
        self._course_entry_ids = _course_entry_ids(track_course_keys)
        self._course_keys = tuple(sorted(self._course_entry_ids))
        self._course_log_keys = dict(track_log_keys)
        self._course_log_enabled = dict(track_log_enabled)
        self._course_labels = dict(track_labels)
        self._course_generated_slots = dict(track_generated_course_slots or {})
        self._course_generated_generations = dict(track_generated_course_generations or {})
        self._course_generated_course_ids = dict(track_generated_course_ids or {})
        self._course_generated_course_names = dict(track_generated_course_names or {})
        self._course_generated_course_hashes = dict(track_generated_course_hashes or {})
        self._course_generated_course_seeds = dict(track_generated_course_seeds or {})
        self._course_generated_course_segment_counts = dict(
            track_generated_course_segment_counts or {}
        )
        self._course_generated_course_lengths = dict(track_generated_course_lengths or {})
        self._settings = settings
        self._action_repeat = max(1, int(action_repeat))
        self._rng = Random(seed)
        self._stats = {
            course_key: TrackStepStats(
                base_weight=_mean_entry_weight(
                    self._course_entry_ids[course_key],
                    track_base_weights,
                ),
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

        base_weights: dict[str, float] = {}
        course_keys: dict[str, str] = {}
        log_keys: dict[str, str] = {}
        labels: dict[str, str] = {}
        log_enabled: dict[str, bool] = {}
        generated_slots: dict[str, int] = {}
        generated_generations: dict[str, int] = {}
        generated_course_ids: dict[str, str] = {}
        generated_course_names: dict[str, str] = {}
        generated_course_hashes: dict[str, str] = {}
        generated_course_seeds: dict[str, int] = {}
        generated_course_segment_counts: dict[str, int] = {}
        generated_course_lengths: dict[str, float] = {}
        for config in configs:
            for entry in config.entries:
                course_key = entry.runtime_course_key or entry.course_id or entry.id
                base_weights.setdefault(entry.id, float(entry.weight))
                course_keys.setdefault(entry.id, course_key)
                log_keys.setdefault(entry.id, sanitize_log_key(course_key))
                labels.setdefault(
                    course_key,
                    entry.course_name or entry.course_id or entry.display_name or entry.id,
                )
                log_enabled.setdefault(course_key, entry.log_per_course)
                if entry.generated_course_slot is not None:
                    generated_slots.setdefault(course_key, int(entry.generated_course_slot))
                if entry.generated_course_generation is not None:
                    generated_generations.setdefault(
                        course_key,
                        int(entry.generated_course_generation),
                    )
                if entry.course_id is not None and entry.generated_course_slot is not None:
                    generated_course_ids.setdefault(course_key, entry.course_id)
                if entry.generated_course_slot is not None:
                    generated_course_names.setdefault(
                        course_key,
                        entry.course_name or entry.display_name or entry.course_id or entry.id,
                    )
                if entry.generated_course_hash is not None:
                    generated_course_hashes.setdefault(course_key, entry.generated_course_hash)
                if entry.generated_course_seed is not None:
                    generated_course_seeds.setdefault(course_key, int(entry.generated_course_seed))
                if entry.generated_course_segment_count is not None:
                    generated_course_segment_counts.setdefault(
                        course_key,
                        int(entry.generated_course_segment_count),
                    )
                if entry.generated_course_length is not None:
                    generated_course_lengths.setdefault(
                        course_key,
                        float(entry.generated_course_length),
                    )
        if len(set(course_keys.values())) <= 1:
            return None

        settings_source = configs[0]
        return cls(
            track_base_weights=base_weights,
            action_repeat=env_config.action_repeat,
            settings=DeficitBudgetSettings(
                uniform_fraction=settings_source.deficit_budget_uniform_fraction,
                min_weight=settings_source.deficit_budget_min_weight,
                max_weight=settings_source.deficit_budget_max_weight,
                ema_alpha=settings_source.deficit_budget_ema_alpha,
                weight_update_rollouts=settings_source.deficit_budget_weight_update_rollouts,
            ),
            track_course_keys=course_keys,
            track_log_keys=log_keys,
            track_labels=labels,
            track_log_enabled=log_enabled,
            track_generated_course_slots=generated_slots,
            track_generated_course_generations=generated_generations,
            track_generated_course_ids=generated_course_ids,
            track_generated_course_names=generated_course_names,
            track_generated_course_hashes=generated_course_hashes,
            track_generated_course_seeds=generated_course_seeds,
            track_generated_course_segment_counts=generated_course_segment_counts,
            track_generated_course_lengths=generated_course_lengths,
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

    def initial_queues(self, *, num_envs: int, queue_length: int) -> dict[int, tuple[str, ...]]:
        queues: dict[int, tuple[str, ...]] = {}
        assignment_cost = max(1.0, float(queue_length))
        for env_index in range(max(0, int(num_envs))):
            first_course = self._course_keys[env_index % len(self._course_keys)]
            self._reserve_course_assignment(first_course, assignment_cost=assignment_cost)
            extra = tuple(
                self.next_course_key(assignment_cost=assignment_cost)
                for _ in range(max(0, queue_length - 1))
            )
            queues[env_index] = (first_course, *extra)
        return queues

    def refill_queues(
        self,
        queue_lengths: Sequence[int],
        *,
        rollout_steps: int,
    ) -> dict[int, tuple[str, ...]]:
        refill_size = DEFICIT_QUEUE_SETTINGS.minimum_refill_size
        refills: dict[int, tuple[str, ...]] = {}
        assignment_cost = max(1.0, float(rollout_steps))
        for env_index, queue_length in enumerate(queue_lengths):
            if int(queue_length) > DEFICIT_QUEUE_SETTINGS.refill_low_watermark:
                continue
            refills[env_index] = tuple(
                self.next_course_key(assignment_cost=assignment_cost) for _ in range(refill_size)
            )
        return refills

    def next_course_key(self, *, assignment_cost: float = 1.0) -> str:
        course_key = max(
            self._course_keys,
            key=lambda course_key: (
                self._deficit_steps[course_key] - self._reserved_reset_steps[course_key],
                self._rng.random() * 1e-9,
            ),
        )
        self._reserve_course_assignment(course_key, assignment_cost=assignment_cost)
        return course_key

    def log_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        target_fractions = self._target_fractions()
        rollout_total = sum(self._rollout_steps.values())
        for course_key, stats in self._stats.items():
            if not self._course_log_enabled[course_key]:
                continue
            key = self._course_log_keys.get(course_key, sanitize_log_key(course_key))
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
                TrackSamplingRuntimeEntry(
                    track_id=course_key,
                    course_key=course_key,
                    label=self._course_labels.get(course_key, course_key),
                    base_weight=stats.base_weight,
                    current_weight=stats.current_weight,
                    completed_frames=self._accounted_env_steps[course_key] * self._action_repeat,
                    episode_count=stats.episode_count,
                    finished_episode_count=stats.finished_episode_count,
                    success_sample_count=stats.success_sample_count,
                    ema_episode_frames=stats.ema_episode_frames,
                    ema_completion_fraction=stats.ema_completion_fraction,
                    generated_course_slot=self._course_generated_slots.get(course_key),
                    generated_course_generation=self._course_generated_generations.get(course_key),
                    generated_course_id=self._course_generated_course_ids.get(course_key),
                    generated_course_name=self._course_generated_course_names.get(course_key),
                    generated_course_hash=self._course_generated_course_hashes.get(course_key),
                    generated_course_seed=self._course_generated_course_seeds.get(course_key),
                    generated_course_segment_count=(
                        self._course_generated_course_segment_counts.get(course_key)
                    ),
                    generated_course_length=self._course_generated_course_lengths.get(course_key),
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
            stats.current_weight = max(0.0, float(entry.current_weight))
            if entry.ema_completion_fraction is not None:
                self._best_completion[course_key] = max(0.0, float(entry.ema_completion_fraction))
            if entry.generated_course_slot is not None:
                self._course_generated_slots[course_key] = entry.generated_course_slot
            if entry.generated_course_generation is not None:
                self._course_generated_generations[course_key] = entry.generated_course_generation
            if entry.generated_course_id is not None:
                self._course_generated_course_ids[course_key] = entry.generated_course_id
            if entry.generated_course_name is not None:
                self._course_generated_course_names[course_key] = entry.generated_course_name
            if entry.generated_course_hash is not None:
                self._course_generated_course_hashes[course_key] = entry.generated_course_hash
            if entry.generated_course_seed is not None:
                self._course_generated_course_seeds[course_key] = entry.generated_course_seed
            if entry.generated_course_segment_count is not None:
                self._course_generated_course_segment_counts[course_key] = (
                    entry.generated_course_segment_count
                )
            if entry.generated_course_length is not None:
                self._course_generated_course_lengths[course_key] = entry.generated_course_length
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

    def _reserve_course_assignment(self, course_key: str, *, assignment_cost: float) -> None:
        self._reserved_reset_steps[course_key] += max(1.0, float(assignment_cost))


def _course_entry_ids(track_course_keys: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for entry_id, course_key in track_course_keys.items():
        grouped.setdefault(course_key, []).append(entry_id)
    return {course_key: tuple(entry_ids) for course_key, entry_ids in grouped.items()}


def _mean_entry_weight(entry_ids: Sequence[str], base_weights: Mapping[str, float]) -> float:
    if not entry_ids:
        return 1.0
    return sum(base_weights[entry_id] for entry_id in entry_ids) / len(entry_ids)


def _episode_crashed(episode: Mapping[str, object]) -> bool:
    reason = episode.get("termination_reason")
    if not isinstance(reason, str):
        return False
    return reason in _CRASH_TERMINATION_REASONS
