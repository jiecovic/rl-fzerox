# src/rl_fzerox/core/envs/engine/reset/episode_start.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import ControllerState, EmulatorBackend, FZeroXTelemetry
from rl_fzerox.core.envs.actions import ResettableActionAdapter
from rl_fzerox.core.envs.engine.components import EngineRuntimeComponents
from rl_fzerox.core.envs.engine.controls import sync_dynamic_action_masks
from rl_fzerox.core.envs.engine.controls.episode_dropout import sample_episode_action_masks
from rl_fzerox.core.envs.engine.info import telemetry_info
from rl_fzerox.core.envs.engine.stepping import set_episode_boost_pad_info
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.runtime_spec.schema import EnvConfig


@dataclass(frozen=True, slots=True)
class EpisodeStartSeeds:
    """Derived seeds used by one episode-start initialization."""

    action_episode_mask: int | None
    reward_episode: int | None


def begin_engine_episode_observation(
    *,
    backend: EmulatorBackend,
    config: EnvConfig,
    components: EngineRuntimeComponents,
    seed: int | None,
    seeds: EpisodeStartSeeds,
    telemetry: FZeroXTelemetry | None,
    info: dict[str, object],
    course_id: str | None,
) -> tuple[ObservationValue, dict[str, object]]:
    """Initialize shared episode state and build the initial observation."""

    backend.set_controller_state(ControllerState())
    components.control_state.reset()
    episode_action_masks = sample_episode_action_masks(
        lean_probability=components.action_config.lean_episode_mask_probability,
        air_brake_probability=components.action_config.air_brake_episode_mask_probability,
        spin_probability=components.action_config.spin_episode_mask_probability,
        seed=seeds.action_episode_mask,
        available_branches={
            dimension.label for dimension in components.action_adapter.action_dimensions
        },
    )
    components.episode.lean_episode_masked = episode_action_masks.lean
    components.episode.air_brake_episode_masked = episode_action_masks.air_brake
    components.episode.spin_episode_masked = episode_action_masks.spin
    components.mask_controller.set_lean_episode_masked(components.episode.lean_episode_masked)
    components.mask_controller.set_air_brake_episode_masked(
        components.episode.air_brake_episode_masked
    )
    components.mask_controller.set_spin_episode_masked(components.episode.spin_episode_masked)
    components.mask_controller.set_lean_allowed_values(
        components.control_state.lean_action_mask_override()
    )
    components.mask_controller.set_air_brake_allowed_values(
        components.control_state.air_brake_action_mask_override()
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
        episode_seed=seeds.reward_episode,
        course_id=course_id,
    )
    components.step_assembler.reward_summary_config = components.reward_tracker.summary_config()
    if isinstance(components.action_adapter, ResettableActionAdapter):
        components.action_adapter.reset()

    info["seed"] = seed
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
    info["lean_episode_masked"] = components.episode.lean_episode_masked
    info["air_brake_episode_masked"] = components.episode.air_brake_episode_masked
    info["spin_episode_masked"] = components.episode.spin_episode_masked
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
    return observation, info
