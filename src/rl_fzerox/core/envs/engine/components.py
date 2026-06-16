# src/rl_fzerox/core/envs/engine/components.py
from __future__ import annotations

from dataclasses import dataclass

from fzerox_emulator import EmulatorBackend
from rl_fzerox.core.envs.actions import ActionAdapter, build_action_adapter
from rl_fzerox.core.envs.engine.controls import ActionMaskController, ControlStateTracker
from rl_fzerox.core.envs.engine.episode import EngineEpisodeState
from rl_fzerox.core.envs.engine.observation import EngineObservationBuilder
from rl_fzerox.core.envs.engine.rendering import backend_renderer
from rl_fzerox.core.envs.engine.stepping import EngineStepAssembler
from rl_fzerox.core.envs.rewards import RewardSummaryConfig, RewardTracker, build_reward_tracker
from rl_fzerox.core.runtime_spec.renderers import RendererName
from rl_fzerox.core.runtime_spec.schema import CurriculumConfig, EnvConfig, RewardConfig
from rl_fzerox.core.runtime_spec.schema.actions import ActionRuntimeConfig


@dataclass(slots=True)
class EngineRuntimeComponents:
    """Shared low-level runtime pieces without reset or episode-lifecycle ownership."""

    renderer: RendererName
    action_config: ActionRuntimeConfig
    action_adapter: ActionAdapter
    observation_builder: EngineObservationBuilder
    reward_tracker: RewardTracker
    reward_summary_config: RewardSummaryConfig
    mask_controller: ActionMaskController
    control_state: ControlStateTracker
    step_assembler: EngineStepAssembler
    episode: EngineEpisodeState


def build_engine_runtime_components(
    *,
    backend: EmulatorBackend,
    config: EnvConfig,
    reward_config: RewardConfig | None,
    curriculum_config: CurriculumConfig | None,
) -> EngineRuntimeComponents:
    """Build shared action/observation/reward/stepping pieces for one emulator."""

    renderer = backend_renderer(backend)
    action_config = config.action.runtime()
    action_adapter = build_action_adapter(action_config)
    observation_builder = EngineObservationBuilder.from_engine_config(
        backend=backend,
        config=config,
        renderer=renderer,
    )
    reward_tracker = build_reward_tracker(
        config=reward_config,
        max_episode_steps=config.max_episode_steps,
    )
    reward_summary_config = reward_tracker.summary_config()
    mask_controller = ActionMaskController.from_config(
        adapter=action_adapter,
        base_overrides=action_config.mask_overrides,
        curriculum_config=curriculum_config,
        boost_unmask_max_speed_kph=action_config.boost_unmask_max_speed_kph,
        lean_unmask_min_speed_kph=action_config.lean_unmask_min_speed_kph,
        mask_air_brake_on_ground=action_config.mask_air_brake_on_ground,
        mask_pitch_on_ground=action_config.mask_pitch_on_ground,
        pitch_neutral_index=action_config.pitch_buckets // 2,
    )
    control_state = ControlStateTracker(
        lean_mode=action_config.lean_mode,
        lean_initial_lockout_frames=action_config.lean_initial_lockout_frames,
        air_brake_pulse_frames=action_config.air_brake_pulse_frames,
        boost_decision_interval_frames=action_config.boost_decision_interval_frames,
        boost_request_lockout_frames=action_config.boost_request_lockout_frames,
        action_history_len=observation_builder.action_history_len,
        action_history_controls=observation_builder.action_history_controls,
        split_lean_history=action_config.split_lean_history,
    )
    step_assembler = EngineStepAssembler(
        backend=backend,
        config=config,
        action_config=action_config,
        reward_summary_config=reward_summary_config,
        reward_tracker=reward_tracker,
        observation_builder=observation_builder,
        mask_controller=mask_controller,
        control_state=control_state,
        renderer=renderer,
    )
    return EngineRuntimeComponents(
        renderer=renderer,
        action_config=action_config,
        action_adapter=action_adapter,
        observation_builder=observation_builder,
        reward_tracker=reward_tracker,
        reward_summary_config=reward_summary_config,
        mask_controller=mask_controller,
        control_state=control_state,
        step_assembler=step_assembler,
        episode=EngineEpisodeState(),
    )
