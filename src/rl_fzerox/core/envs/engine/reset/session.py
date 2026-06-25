# src/rl_fzerox/core/envs/engine/reset/session.py
"""Reset coordinator for track selection, baseline loading, and presentation sync.

`EngineResetCoordinator` is the reset-side owner of sampled targets, queued
deficit resets, engine tuning choices, cached baselines, and reset seeds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from fzerox_emulator import EmulatorBackend, FZeroXTelemetry
from rl_fzerox.core.boot import sync_race_intro_target
from rl_fzerox.core.engine_tuning import (
    EngineTuningResetSampler,
    EngineTuningSelectionMode,
)
from rl_fzerox.core.runtime_spec.schema import (
    EnvConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)

from ..info import (
    backend_step_info,
    has_custom_baseline,
    read_live_telemetry,
)
from .camera import CAMERA_SYNC_CONTROLS, sync_camera_setting
from .lives import randomize_gp_lives_on_reset
from .race import load_track_baseline, reset_race_state
from .seeding import EngineResetSeeds
from .tracks import (
    SelectedTrack,
    TrackBaselineCache,
    TrackResetSelector,
    TrackSamplingQueuedReset,
    select_reset_track_by_course_id,
)


@dataclass(frozen=True, slots=True)
class EngineResetResult:
    selected_track: SelectedTrack | None
    selected_track_info: dict[str, object] | None
    info: dict[str, object]
    telemetry: FZeroXTelemetry | None
    uses_custom_baseline: bool


class EngineResetCoordinator:
    """Own reset-time track selection, baseline loading, and boot synchronization."""

    def __init__(
        self,
        *,
        backend: EmulatorBackend,
        config: EnvConfig,
        env_index: int,
    ) -> None:
        self._backend = backend
        self._config = config
        self._track_sampling_weight_overrides: dict[str, float] = {}
        self._active_track_sampling = self._track_sampling_with_weight_overrides(
            self._config.track_sampling
        )
        self._locked_reset_course_id: str | None = None
        self._sequential_track_sampling = False
        self._queued_resets: list[TrackSamplingQueuedReset] = []
        self._track_selector = TrackResetSelector(env_index=env_index)
        self._track_baseline_cache = TrackBaselineCache()
        self._reset_seeds = EngineResetSeeds()
        self._engine_tuning_sampler: EngineTuningResetSampler | None = None
        self._engine_tuning_selection: EngineTuningSelectionMode = "sample"

    def set_track_sampling_weights(self, weights_by_track_id: dict[str, float]) -> None:
        """Update adaptive reset weights used by step-balanced track sampling."""

        self._track_sampling_weight_overrides = {
            str(track_id): float(weight)
            for track_id, weight in weights_by_track_id.items()
            if isinstance(track_id, str)
            and isinstance(weight, int | float)
            and math.isfinite(float(weight))
            and float(weight) >= 0.0
        }
        self._active_track_sampling = self._track_sampling_with_weight_overrides(
            self._config.track_sampling
        )

    def set_track_sampling_config(self, config: TrackSamplingConfig) -> None:
        """Replace the base track-sampling config used at future episode resets."""

        self._config = self._config.model_copy(update={"track_sampling": config})
        self._queued_resets.clear()
        self._active_track_sampling = self._track_sampling_with_weight_overrides(
            self._config.track_sampling
        )
        self._prune_baseline_cache_to_active_tracks()

    def set_engine_tuning_sampler(self, sampler: EngineTuningResetSampler | None) -> None:
        """Replace the adaptive engine choices used at future resets."""

        self._engine_tuning_sampler = sampler

    def set_engine_tuning_selection(self, selection: EngineTuningSelectionMode) -> None:
        """Choose whether adaptive engine tuning samples or picks greedy values."""

        self._engine_tuning_selection = selection

    def extend_track_sampling_reset_queue(
        self,
        queued_resets: Sequence[TrackSamplingQueuedReset | str],
    ) -> None:
        """Append externally scheduled reset slots for deficit-budget sampling."""

        self._queued_resets.extend(_queued_reset(value) for value in queued_resets)

    def clear_track_sampling_reset_queue(self) -> None:
        """Drop externally scheduled reset slots that have not been consumed yet."""

        self._queued_resets.clear()

    def track_sampling_reset_queue_length(self) -> int:
        """Return how many externally scheduled reset courses remain queued."""

        return len(self._queued_resets)

    def set_locked_reset_course(self, course_id: str | None) -> None:
        """Lock subsequent sampled resets to one course for watch/manual inspection."""

        self._locked_reset_course_id = course_id if course_id else None

    def set_sequential_track_sampling(self, enabled: bool) -> None:
        """Use configured track order for watch resets instead of training sampling."""

        self._sequential_track_sampling = bool(enabled)

    def set_next_sequential_course(self, course_id: str | None) -> None:
        """Align the next sequential watch reset to one configured course."""

        if not course_id:
            return
        self._track_selector.set_next_sequential_course(
            self._active_track_sampling,
            course_id=course_id,
        )

    def select_episode_track(self, seed: int | None) -> SelectedTrack | None:
        self._reset_seeds.remember_reset_seed(seed)
        if self._locked_reset_course_id is not None:
            selected_track = select_reset_track_by_course_id(
                self._active_track_sampling,
                course_id=self._locked_reset_course_id,
                seed=seed,
                engine_tuning_sampler=self._engine_tuning_sampler,
                engine_tuning_selection=self._engine_tuning_selection,
            )
            if selected_track is not None:
                return selected_track
        if self._sequential_track_sampling:
            return self._track_selector.select_sequential(
                self._active_track_sampling,
                seed=self._reset_seeds.track_sampling_seed(seed),
                engine_tuning_sampler=self._engine_tuning_sampler,
                engine_tuning_selection=self._engine_tuning_selection,
            )
        if self._active_track_sampling.sampling_mode == "deficit_budget":
            return self._select_queued_track(seed=seed)
        return self._track_selector.select(
            self._active_track_sampling,
            seed=self._reset_seeds.track_sampling_seed(seed),
            engine_tuning_sampler=self._engine_tuning_sampler,
            engine_tuning_selection=self._engine_tuning_selection,
        )

    def reset_race(
        self,
        *,
        seed: int | None,
        selected_track: SelectedTrack | None,
    ) -> EngineResetResult:
        selected_track_info = None if selected_track is None else selected_track.info()
        if selected_track is not None:
            load_track_baseline(
                backend=self._backend,
                cache=self._track_baseline_cache,
                selected_track=selected_track,
                cache_enabled=self._config.cache_track_baselines,
            )
        _, info, telemetry = reset_race_state(
            backend=self._backend,
            config=self._config,
            sampled_track_baseline=selected_track is not None,
            selected_track=selected_track,
        )
        if selected_track_info is not None:
            info.update(selected_track_info)
        if self._locked_reset_course_id is not None:
            info["track_sampling_locked_course_id"] = self._locked_reset_course_id

        uses_custom_baseline = selected_track is not None or has_custom_baseline(info)
        telemetry = self._maybe_randomize_game_rng(seed, telemetry, info)
        self._maybe_randomize_gp_lives(seed, telemetry, selected_track, info)
        telemetry = sync_reset_presentation(
            self._backend,
            camera_setting=self._config.camera_setting,
            race_intro_target_timer=self._config.race_intro_target_timer,
            telemetry=telemetry,
            info=info,
        )
        info.update(backend_step_info(self._backend))
        return EngineResetResult(
            selected_track=selected_track,
            selected_track_info=selected_track_info,
            info=info,
            telemetry=telemetry,
            uses_custom_baseline=uses_custom_baseline,
        )

    def reward_episode_seed(self, seed: int | None) -> int | None:
        return self._reset_seeds.reward_episode_seed(seed)

    def action_episode_mask_seed(self, seed: int | None) -> int | None:
        return self._reset_seeds.action_episode_mask_seed(seed)

    def advance_reset_count(self) -> None:
        self._reset_seeds.advance_reset_count()

    def _track_sampling_with_weight_overrides(
        self,
        config: TrackSamplingConfig,
    ) -> TrackSamplingConfig:
        if config.sampling_mode != "step_balanced" or not self._track_sampling_weight_overrides:
            return config
        entries = _track_sampling_entries_with_weight_overrides(
            config.entries,
            weights_by_track_id=self._track_sampling_weight_overrides,
        )
        return config.model_copy(update={"entries": entries})

    def _prune_baseline_cache_to_active_tracks(self) -> None:
        self._track_baseline_cache.retain_paths(
            entry.baseline_state_path
            for entry in self._active_track_sampling.entries
            if entry.baseline_state_path is not None
        )

    def _select_queued_track(self, *, seed: int | None) -> SelectedTrack | None:
        if not self._active_track_sampling.enabled:
            return None
        if not self._queued_resets:
            return self._track_selector.select(
                self._active_track_sampling,
                seed=self._reset_seeds.track_sampling_seed(seed),
                engine_tuning_sampler=self._engine_tuning_sampler,
                engine_tuning_selection=self._engine_tuning_selection,
            )
        queued_reset = self._queued_resets.pop(0)
        selected_track = select_reset_track_by_course_id(
            self._active_track_sampling,
            course_id=queued_reset.course_id,
            sampling_mode="deficit_budget",
            seed=self._reset_seeds.track_sampling_seed(seed),
            engine_tuning_sampler=self._engine_tuning_sampler,
            engine_tuning_selection=self._engine_tuning_selection,
        )
        if selected_track is None:
            raise RuntimeError(
                f"deficit-budget reset queue referenced unknown course {queued_reset.course_id!r}"
            )
        return replace(selected_track, deficit_budget_lane=queued_reset.deficit_lane)

    def _maybe_randomize_game_rng(
        self,
        seed: int | None,
        telemetry: FZeroXTelemetry | None,
        info: dict[str, object],
    ) -> FZeroXTelemetry | None:
        if not self._config.randomize_game_rng_on_reset:
            return telemetry
        if self._config.randomize_game_rng_requires_race_mode and (
            telemetry is None or not telemetry.in_race_mode
        ):
            info["rng_randomized"] = False
            info["rng_randomization_skip_reason"] = "not_in_race"
            return telemetry

        rng_seed = self._reset_seeds.reset_rng_seed(seed)
        if rng_seed is None:
            return telemetry
        rng_state = self._backend.randomize_game_rng(rng_seed)
        info["rng_randomized"] = True
        info["rng_seed"] = rng_seed
        info["rng_state"] = rng_state
        return read_live_telemetry(self._backend) or telemetry

    def _maybe_randomize_gp_lives(
        self,
        seed: int | None,
        telemetry: FZeroXTelemetry | None,
        selected_track: SelectedTrack | None,
        info: dict[str, object],
    ) -> None:
        if not self._config.randomize_gp_lives_on_reset:
            return
        randomize_gp_lives_on_reset(
            backend=self._backend,
            telemetry=telemetry,
            target_gp_difficulty=None if selected_track is None else selected_track.gp_difficulty,
            jitter_min=self._config.gp_lives_jitter_min,
            jitter_max=self._config.gp_lives_jitter_max,
            seed=self._reset_seeds.gp_lives_jitter_seed(seed),
            info=info,
        )


def sync_reset_presentation(
    backend: EmulatorBackend,
    *,
    camera_setting: str | None,
    race_intro_target_timer: int | None,
    telemetry: FZeroXTelemetry | None,
    info: dict[str, object],
) -> FZeroXTelemetry | None:
    camera_ready_timer = _camera_ready_intro_timer(
        camera_setting=camera_setting,
        race_intro_target_timer=race_intro_target_timer,
    )
    if camera_ready_timer is not None:
        _, telemetry = sync_race_intro_target(
            backend,
            target_timer=camera_ready_timer,
        )
    telemetry = sync_camera_setting(
        backend,
        target_name=_validated_camera_setting(camera_setting),
        telemetry=telemetry,
        info=info,
    )
    race_intro_info, telemetry = sync_race_intro_target(
        backend,
        target_timer=race_intro_target_timer,
    )
    info.update(race_intro_info)
    return telemetry


def _track_sampling_entries_with_weight_overrides(
    entries: tuple[TrackSamplingEntryConfig, ...],
    *,
    weights_by_track_id: dict[str, float],
) -> tuple[TrackSamplingEntryConfig, ...]:
    grouped_entries: dict[str, list[TrackSamplingEntryConfig]] = {}
    for entry in entries:
        if entry.baseline_group_id is None:
            continue
        grouped_entries.setdefault(entry.baseline_group_id, []).append(entry)

    adjusted: list[TrackSamplingEntryConfig] = []
    for entry in entries:
        group_id = entry.baseline_group_id
        if group_id is None:
            adjusted.append(
                entry.model_copy(update={"weight": weights_by_track_id.get(entry.id, entry.weight)})
            )
            continue
        if group_id != entry.id:
            # The group is emitted when the base entry is visited.
            continue
        group = tuple(grouped_entries.get(group_id, (entry,)))
        override_weight = weights_by_track_id.get(
            group_id,
            sum(float(item.weight) for item in group),
        )
        ratio_total = sum(_baseline_group_ratio(item) for item in group)
        if ratio_total <= 0.0:
            adjusted.extend(group)
            continue
        adjusted.extend(
            item.model_copy(
                update={
                    "weight": float(override_weight) * _baseline_group_ratio(item) / ratio_total
                }
            )
            for item in group
        )
    return tuple(adjusted)


def _baseline_group_ratio(entry: TrackSamplingEntryConfig) -> float:
    if entry.baseline_group_weight is None:
        return 1.0
    return max(0.0, float(entry.baseline_group_weight))


def _queued_reset(value: TrackSamplingQueuedReset | str) -> TrackSamplingQueuedReset:
    if isinstance(value, TrackSamplingQueuedReset):
        return value
    return TrackSamplingQueuedReset(course_id=str(value))


def _camera_ready_intro_timer(
    *,
    camera_setting: str | None,
    race_intro_target_timer: int | None,
) -> int | None:
    if camera_setting is None:
        return race_intro_target_timer
    if race_intro_target_timer is None:
        return CAMERA_SYNC_CONTROLS.ready_intro_timer
    return max(race_intro_target_timer, CAMERA_SYNC_CONTROLS.ready_intro_timer)


def _validated_camera_setting(camera_setting: str | None):
    if camera_setting is None:
        return None
    if camera_setting == "overhead":
        return "overhead"
    if camera_setting == "close_behind":
        return "close_behind"
    if camera_setting == "regular":
        return "regular"
    if camera_setting == "wide":
        return "wide"
    raise ValueError(f"Unsupported camera setting {camera_setting!r}")
