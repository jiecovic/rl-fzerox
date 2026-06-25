# src/rl_fzerox/core/envs/engine/reset/track_sampling/selection.py
from __future__ import annotations

from rl_fzerox.core.engine_tuning import (
    EngineTuningResetSampler,
    EngineTuningSelectionMode,
)
from rl_fzerox.core.envs.engine.reset.track_sampling.grouping import (
    entry_matches_course_request,
    group_entries_by_course,
    sequential_course_buckets,
    sequential_track_sampling_fingerprint,
    track_sampling_fingerprint,
)
from rl_fzerox.core.envs.engine.reset.track_sampling.models import (
    SelectedTrack,
)
from rl_fzerox.core.envs.engine.reset.track_sampling.modes import (
    balanced_cycle,
    pick_track_entry,
    selectable_entries,
    weighted_course_entry,
)
from rl_fzerox.core.envs.engine.reset.track_sampling.projection import (
    selected_track_from_entry,
)
from rl_fzerox.core.runtime_spec.schema import (
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)


class TrackResetSelector:
    """Select reset baselines using the configured course sampler."""

    def __init__(self, *, env_index: int = 0) -> None:
        self._env_index = max(0, int(env_index))
        self._fingerprint: tuple[object, ...] | None = None
        self._cycle: tuple[TrackSamplingEntryConfig, ...] = ()
        self._sequential_course_buckets: tuple[tuple[TrackSamplingEntryConfig, ...], ...] = ()
        self._cursor = 0

    def select(
        self,
        config: TrackSamplingConfig,
        *,
        seed: int | None,
        engine_tuning_sampler: EngineTuningResetSampler | None = None,
        engine_tuning_selection: EngineTuningSelectionMode = "sample",
    ) -> SelectedTrack | None:
        if config.sampling_mode == "equal":
            return self._select_balanced(
                config,
                sampling_mode=config.sampling_mode,
                seed=seed,
                engine_tuning_sampler=engine_tuning_sampler,
                engine_tuning_selection=engine_tuning_selection,
            )
        if config.sampling_mode == "step_balanced":
            return select_reset_track_by_course_weight(
                config,
                seed=seed,
                sampling_mode=config.sampling_mode,
                engine_tuning_sampler=engine_tuning_sampler,
                engine_tuning_selection=engine_tuning_selection,
            )
        if config.sampling_mode in {"deficit_budget", "fixed_env"}:
            return self._select_fixed_env(
                config,
                sampling_mode=config.sampling_mode,
                seed=seed,
                engine_tuning_sampler=engine_tuning_sampler,
                engine_tuning_selection=engine_tuning_selection,
            )
        raise ValueError(f"Unsupported track sampling mode: {config.sampling_mode!r}")

    def select_sequential(
        self,
        config: TrackSamplingConfig,
        *,
        seed: int | None,
        engine_tuning_sampler: EngineTuningResetSampler | None = None,
        engine_tuning_selection: EngineTuningSelectionMode = "sample",
    ) -> SelectedTrack | None:
        """Select configured tracks in list order, ignoring training weights."""

        if not config.enabled:
            return None
        selectable_entries_ = selectable_entries(config.entries)
        if not selectable_entries_:
            raise ValueError("track sampling is enabled but has no entries")
        self._sync_sequential_cycle(config.model_copy(update={"entries": selectable_entries_}))
        position = self._cursor % len(self._sequential_course_buckets)
        entry = pick_track_entry(
            self._sequential_course_buckets[position],
            seed=seed,
        )
        self._cursor += 1
        return selected_track_from_entry(
            entry,
            sampling_mode="sequential",
            cycle_position=position,
            seed=seed,
            engine_tuning_config=config.engine_tuning,
            engine_tuning_sampler=engine_tuning_sampler,
            engine_tuning_selection=engine_tuning_selection,
        )

    def set_next_sequential_course(
        self,
        config: TrackSamplingConfig,
        *,
        course_id: str,
    ) -> bool:
        """Align the next sequential watch reset to the requested course."""

        if not config.enabled or not config.entries:
            return False
        self._sync_sequential_cycle(config)
        for index, bucket in enumerate(self._sequential_course_buckets):
            if any(entry_matches_course_request(entry, course_id) for entry in bucket):
                self._cursor = index
                return True
        return False

    def _select_balanced(
        self,
        config: TrackSamplingConfig,
        *,
        sampling_mode: str,
        seed: int | None,
        engine_tuning_sampler: EngineTuningResetSampler | None,
        engine_tuning_selection: EngineTuningSelectionMode,
    ) -> SelectedTrack | None:
        if not config.enabled:
            return None
        selectable_entries_ = selectable_entries(config.entries)
        if not selectable_entries_:
            raise ValueError("track sampling is enabled but has no entries")
        self._sync_balanced_cycle(config.model_copy(update={"entries": selectable_entries_}))
        position = self._cursor % len(self._cycle)
        entry = self._cycle[position]
        self._cursor += 1
        return selected_track_from_entry(
            entry,
            sampling_mode=sampling_mode,
            cycle_position=position,
            seed=seed,
            engine_tuning_config=config.engine_tuning,
            engine_tuning_sampler=engine_tuning_sampler,
            engine_tuning_selection=engine_tuning_selection,
        )

    def _sync_balanced_cycle(self, config: TrackSamplingConfig) -> None:
        fingerprint = track_sampling_fingerprint(config)
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint
        self._cycle = balanced_cycle(config.entries)
        self._cursor = self._env_index % len(self._cycle)

    def _sync_sequential_cycle(self, config: TrackSamplingConfig) -> None:
        fingerprint = ("sequential", *sequential_track_sampling_fingerprint(config))
        if fingerprint == self._fingerprint:
            return
        self._fingerprint = fingerprint
        self._sequential_course_buckets = sequential_course_buckets(config.entries)
        self._cursor = 0

    def _select_fixed_env(
        self,
        config: TrackSamplingConfig,
        *,
        sampling_mode: str,
        seed: int | None,
        engine_tuning_sampler: EngineTuningResetSampler | None,
        engine_tuning_selection: EngineTuningSelectionMode,
    ) -> SelectedTrack | None:
        if not config.enabled:
            return None
        selectable_entries_ = selectable_entries(config.entries)
        if not selectable_entries_:
            raise ValueError("track sampling is enabled but has no entries")
        course_buckets = tuple(
            entries for _, entries in group_entries_by_course(selectable_entries_)
        )
        position = self._env_index % len(course_buckets)
        entry = pick_track_entry(course_buckets[position], seed=seed)
        return selected_track_from_entry(
            entry,
            sampling_mode=sampling_mode,
            cycle_position=position,
            seed=seed,
            engine_tuning_config=config.engine_tuning,
            engine_tuning_sampler=engine_tuning_sampler,
            engine_tuning_selection=engine_tuning_selection,
        )


def select_reset_track_by_course_id(
    config: TrackSamplingConfig,
    *,
    course_id: str,
    sampling_mode: str = "locked",
    seed: int | None = None,
    engine_tuning_sampler: EngineTuningResetSampler | None = None,
    engine_tuning_selection: EngineTuningSelectionMode = "sample",
) -> SelectedTrack | None:
    """Select one configured reset baseline for a specific course id."""

    if not config.enabled:
        return None
    matching_entries = tuple(
        entry
        for entry in selectable_entries(config.entries)
        if entry_matches_course_request(entry, course_id)
    )
    if matching_entries:
        return selected_track_from_entry(
            pick_track_entry(matching_entries, seed=seed),
            sampling_mode=sampling_mode,
            seed=seed,
            engine_tuning_config=config.engine_tuning,
            engine_tuning_sampler=engine_tuning_sampler,
            engine_tuning_selection=engine_tuning_selection,
        )
    return None


def select_reset_track_by_course_weight(
    config: TrackSamplingConfig,
    *,
    seed: int | None,
    sampling_mode: str,
    engine_tuning_sampler: EngineTuningResetSampler | None = None,
    engine_tuning_selection: EngineTuningSelectionMode = "sample",
) -> SelectedTrack | None:
    """Select one course by summed weight, then one baseline within that course."""

    entry = weighted_course_entry(config, seed=seed)
    if entry is None:
        return None
    return selected_track_from_entry(
        entry,
        sampling_mode=sampling_mode,
        seed=seed,
        engine_tuning_config=config.engine_tuning,
        engine_tuning_sampler=engine_tuning_sampler,
        engine_tuning_selection=engine_tuning_selection,
    )
