# src/rl_fzerox/core/envs/gym_runtime/reset.py
from __future__ import annotations

from fzerox_emulator import EmulatorBackend
from rl_fzerox.core.envs.engine.components import EngineRuntimeComponents
from rl_fzerox.core.envs.engine.reset import EngineResetCoordinator
from rl_fzerox.core.envs.engine.reset.episode_start import (
    EpisodeStartSeeds,
    begin_engine_episode_observation,
)
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.runtime_spec.schema import EnvConfig


def reset_gym_episode(
    *,
    backend: EmulatorBackend,
    config: EnvConfig,
    components: EngineRuntimeComponents,
    reset_coordinator: EngineResetCoordinator,
    seed: int | None,
) -> tuple[ObservationValue, dict[str, object]]:
    """Reset one materialized Gym/watch episode and build its initial observation."""

    selected_track = reset_coordinator.select_episode_track(seed)
    reset_result = reset_coordinator.reset_race(
        seed=seed,
        selected_track=selected_track,
    )
    components.episode.begin_reset(
        active_track=reset_result.selected_track,
        active_track_info=reset_result.selected_track_info,
    )
    info = reset_result.info
    telemetry = reset_result.telemetry
    components.episode.uses_custom_baseline = reset_result.uses_custom_baseline
    observation, info = begin_engine_episode_observation(
        backend=backend,
        config=config,
        components=components,
        seed=seed,
        seeds=EpisodeStartSeeds(
            action_episode_mask=reset_coordinator.action_episode_mask_seed(seed),
            reward_episode=reset_coordinator.reward_episode_seed(seed),
        ),
        telemetry=telemetry,
        info=info,
        course_id=None if selected_track is None else selected_track.course_id,
    )
    reset_coordinator.advance_reset_count()
    return observation, info
