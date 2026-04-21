# src/rl_fzerox/core/envs/engine/observation.py
from __future__ import annotations

from dataclasses import dataclass

from gymnasium import spaces

from fzerox_emulator import EmulatorBackend, FZeroXTelemetry, ObservationSpec
from fzerox_emulator.arrays import ObservationFrame
from rl_fzerox.core.config.schema import EnvConfig
from rl_fzerox.core.domain.observation_components import (
    ActionHistoryControlName,
    StateComponentsSettings,
)
from rl_fzerox.core.envs.engine.control_state import ControlStateTracker
from rl_fzerox.core.envs.engine.info import set_observation_info
from rl_fzerox.core.envs.observations import (
    ObservationValue,
    action_history_settings_for_observation,
    build_observation,
    build_observation_space,
)


@dataclass(slots=True)
class EngineObservationBuilder:
    """Own image/state observation setup for one env engine."""

    backend: EmulatorBackend
    config: EnvConfig
    spec: ObservationSpec
    state_components: StateComponentsSettings | None
    zeroed_state_components: tuple[str, ...]
    action_history_len: int | None
    action_history_controls: tuple[ActionHistoryControlName, ...]
    space: spaces.Space

    @classmethod
    def from_engine_config(
        cls,
        *,
        backend: EmulatorBackend,
        config: EnvConfig,
    ) -> EngineObservationBuilder:
        spec = backend.observation_spec(config.observation.preset)
        state_components = config.observation.state_components_data()
        action_history_len, action_history_controls = action_history_settings_for_observation(
            state_components=state_components,
            fallback_len=config.observation.action_history_len,
            fallback_controls=config.observation.action_history_controls,
        )
        space = build_observation_space(
            spec,
            frame_stack=config.observation.frame_stack,
            stack_mode=config.observation.stack_mode,
            minimap_layer=config.observation.minimap_layer,
            mode=config.observation.mode,
            state_profile=config.observation.state_profile,
            course_context=config.observation.course_context,
            ground_effect_context=config.observation.ground_effect_context,
            action_history_len=action_history_len,
            action_history_controls=action_history_controls,
            state_components=state_components,
        )
        return cls(
            backend=backend,
            config=config,
            spec=spec,
            state_components=state_components,
            zeroed_state_components=tuple(config.observation.zeroed_state_components),
            action_history_len=action_history_len,
            action_history_controls=action_history_controls,
            space=space,
        )

    def render_image(self) -> ObservationFrame:
        return self.backend.render_observation(
            preset=self.config.observation.preset,
            frame_stack=self.config.observation.frame_stack,
            stack_mode=self.config.observation.stack_mode,
            minimap_layer=self.config.observation.minimap_layer,
        )

    def build_observation(
        self,
        *,
        image: ObservationFrame,
        telemetry: FZeroXTelemetry | None,
        control_state: ControlStateTracker,
    ) -> ObservationValue:
        """Build the policy observation from the rendered image plus control context."""

        return build_observation(
            image=image,
            telemetry=telemetry,
            mode=self.config.observation.mode,
            state_profile=self.config.observation.state_profile,
            course_context=self.config.observation.course_context,
            ground_effect_context=self.config.observation.ground_effect_context,
            action_history_len=self.action_history_len,
            action_history_controls=self.action_history_controls,
            action_history=control_state.action_history_fields(),
            state_components=self.state_components,
            zeroed_state_components=self.zeroed_state_components,
            **control_state.observation_fields(),
        )

    def set_info(
        self,
        info: dict[str, object],
        *,
        image_shape: tuple[int, ...],
    ) -> None:
        set_observation_info(
            info,
            observation_shape=image_shape,
            observation_spec=self.spec,
            frame_stack=self.config.observation.frame_stack,
            observation_stack_mode=self.config.observation.stack_mode,
            observation_minimap_layer=self.config.observation.minimap_layer,
            observation_mode=self.config.observation.mode,
            observation_state_profile=self.config.observation.state_profile,
            observation_course_context=self.config.observation.course_context,
            observation_ground_effect_context=self.config.observation.ground_effect_context,
            action_history_len=self.action_history_len,
            action_history_controls=self.action_history_controls,
            observation_state_components=self.state_components,
            observation_zeroed_state_components=self.zeroed_state_components,
        )
