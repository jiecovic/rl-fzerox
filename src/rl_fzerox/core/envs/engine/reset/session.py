# src/rl_fzerox/core/envs/engine/reset/session.py
from __future__ import annotations

import math
from dataclasses import dataclass

from fzerox_emulator import EmulatorBackend, FZeroXTelemetry
from rl_fzerox.core.boot import sync_race_intro_target
from rl_fzerox.core.config.schema import CurriculumConfig, EnvConfig, TrackSamplingConfig

from ..info import (
    backend_step_info,
    has_custom_baseline,
    read_live_telemetry,
)
from .camera import sync_camera_setting
from .race import load_track_baseline, reset_race_state
from .seeding import EngineResetSeeds
from .tracks import (
    SelectedTrack,
    TrackBaselineCache,
    TrackResetSelector,
    select_reset_track_by_course_id,
)


@dataclass(frozen=True, slots=True)
class EngineResetResult:
    selected_track: SelectedTrack | None
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
        curriculum_config: CurriculumConfig | None,
        env_index: int,
    ) -> None:
        self._backend = backend
        self._config = config
        self._curriculum_config = curriculum_config
        self._stage_index: int | None = None
        self._track_sampling_weight_overrides: dict[str, float] = {}
        self._active_track_sampling = self._stage_track_sampling_config(self._stage_index)
        self._locked_reset_course_id: str | None = None
        self._track_selector = TrackResetSelector(env_index=env_index)
        self._track_baseline_cache = TrackBaselineCache()
        self._reset_seeds = EngineResetSeeds()

    def set_curriculum_stage(self, stage_index: int | None) -> None:
        self._stage_index = stage_index
        self._active_track_sampling = self._stage_track_sampling_config(stage_index)

    def set_track_sampling_weights(self, weights_by_track_id: dict[str, float]) -> None:
        """Update adaptive reset weights used by step-balanced track sampling."""

        self._track_sampling_weight_overrides = {
            str(track_id): float(weight)
            for track_id, weight in weights_by_track_id.items()
            if isinstance(track_id, str)
            and isinstance(weight, int | float)
            and math.isfinite(float(weight))
            and float(weight) > 0.0
        }
        self._active_track_sampling = self._stage_track_sampling_config(self._stage_index)

    def set_locked_reset_course(self, course_id: str | None) -> None:
        """Lock subsequent sampled resets to one course for watch/manual inspection."""

        self._locked_reset_course_id = course_id if course_id else None

    def select_episode_track(self, seed: int | None) -> SelectedTrack | None:
        self._reset_seeds.remember_reset_seed(seed)
        if self._locked_reset_course_id is not None:
            selected_track = select_reset_track_by_course_id(
                self._active_track_sampling,
                course_id=self._locked_reset_course_id,
            )
            if selected_track is not None:
                return selected_track
        return self._track_selector.select(
            self._active_track_sampling,
            seed=self._reset_seeds.track_sampling_seed(seed),
        )

    def reset_race(
        self,
        *,
        seed: int | None,
        selected_track: SelectedTrack | None,
    ) -> EngineResetResult:
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
        )
        if selected_track is not None:
            info.update(selected_track.info())
        if self._locked_reset_course_id is not None:
            info["track_sampling_locked_course_id"] = self._locked_reset_course_id

        uses_custom_baseline = selected_track is not None or has_custom_baseline(info)
        telemetry = self._maybe_randomize_game_rng(seed, telemetry, info)
        telemetry = sync_camera_setting(
            self._backend,
            target_name=self._config.camera_setting,
            telemetry=telemetry,
            info=info,
        )
        race_intro_info, telemetry = sync_race_intro_target(
            self._backend,
            target_timer=self._config.race_intro_target_timer,
        )
        info.update(race_intro_info)
        info.update(backend_step_info(self._backend))
        return EngineResetResult(
            selected_track=selected_track,
            info=info,
            telemetry=telemetry,
            uses_custom_baseline=uses_custom_baseline,
        )

    def reward_episode_seed(self, seed: int | None) -> int | None:
        return self._reset_seeds.reward_episode_seed(seed)

    def advance_reset_count(self) -> None:
        self._reset_seeds.advance_reset_count()

    def _stage_track_sampling_config(self, stage_index: int | None) -> TrackSamplingConfig:
        if (
            self._curriculum_config is None
            or not self._curriculum_config.enabled
            or stage_index is None
        ):
            return self._track_sampling_with_weight_overrides(self._config.track_sampling)
        stage = self._curriculum_config.stages[stage_index]
        return self._track_sampling_with_weight_overrides(
            stage.track_sampling or self._config.track_sampling
        )

    def _track_sampling_with_weight_overrides(
        self,
        config: TrackSamplingConfig,
    ) -> TrackSamplingConfig:
        if config.sampling_mode != "step_balanced" or not self._track_sampling_weight_overrides:
            return config
        entries = tuple(
            entry.model_copy(
                update={"weight": self._track_sampling_weight_overrides.get(entry.id, entry.weight)}
            )
            for entry in config.entries
        )
        return config.model_copy(update={"entries": entries})

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
