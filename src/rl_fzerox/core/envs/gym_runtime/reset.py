# src/rl_fzerox/core/envs/gym_runtime/reset.py
from __future__ import annotations

from fzerox_emulator import ControllerState, EmulatorBackend
from rl_fzerox.core.envs.actions import ResettableActionAdapter
from rl_fzerox.core.envs.engine.components import EngineRuntimeComponents
from rl_fzerox.core.envs.engine.controls import sync_dynamic_action_masks
from rl_fzerox.core.envs.engine.info import set_curriculum_info, telemetry_info
from rl_fzerox.core.envs.engine.reset import EngineResetCoordinator
from rl_fzerox.core.envs.engine.stepping import set_episode_boost_pad_info
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
    components.episode.begin_reset(active_track=selected_track)
    reset_result = reset_coordinator.reset_race(
        seed=seed,
        selected_track=selected_track,
    )
    info = reset_result.info
    telemetry = reset_result.telemetry
    components.episode.uses_custom_baseline = reset_result.uses_custom_baseline
    backend.set_controller_state(ControllerState())
    components.control_state.reset()
    components.mask_controller.set_lean_allowed_values(
        components.control_state.lean_action_mask_override()
    )
    components.mask_controller.set_spin_allowed_values(None)
    sync_dynamic_action_masks(
        mask_controller=components.mask_controller,
        control_state=components.control_state,
        telemetry=telemetry,
        boost_min_energy_fraction=config.boost_min_energy_fraction,
        mask_boost_when_active=components.action_config.mask_boost_when_active,
        mask_boost_when_airborne=components.action_config.mask_boost_when_airborne,
    )
    components.episode.last_telemetry = telemetry
    components.reward_tracker.reset(
        telemetry,
        episode_seed=reset_coordinator.reward_episode_seed(seed),
        course_id=None if selected_track is None else selected_track.course_id,
    )
    components.step_assembler.reward_summary_config = components.reward_tracker.summary_config()
    if isinstance(components.action_adapter, ResettableActionAdapter):
        components.action_adapter.reset()
    info["seed"] = seed
    set_curriculum_info(
        info,
        stage_index=components.mask_controller.stage_index,
        stage_name=components.mask_controller.stage_name,
    )
    if telemetry is not None:
        info.update(telemetry_info(telemetry))
    info.update(components.reward_tracker.info(telemetry))
    set_episode_boost_pad_info(
        info,
        episode_boost_pad_entries=components.episode.boost_pad_entries,
    )
    info["episode_step"] = components.episode.frame_count
    info["episode_return"] = components.episode.return_value
    info["episode_airborne_frames"] = components.episode.airborne_frames
    image_observation = components.observation_builder.render_image()
    observation = components.observation_builder.build_observation(
        image=image_observation,
        telemetry=telemetry,
        control_state=components.control_state,
    )
    components.observation_builder.set_info(
        info,
        image_shape=tuple(int(value) for value in image_observation.shape),
    )
    components.episode.last_info = dict(info)
    reset_coordinator.advance_reset_count()
    return observation, info
