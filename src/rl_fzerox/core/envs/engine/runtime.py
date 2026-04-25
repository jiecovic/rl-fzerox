# src/rl_fzerox/core/envs/engine/runtime.py
from __future__ import annotations

from gymnasium import spaces

from fzerox_emulator import ControllerState, EmulatorBackend, FZeroXTelemetry
from fzerox_emulator.arrays import ActionMask, RgbFrame
from rl_fzerox.core.boot import sync_race_intro_target
from rl_fzerox.core.config.schema import (
    CurriculumConfig,
    EnvConfig,
    RewardConfig,
    TrackSamplingConfig,
)
from rl_fzerox.core.envs.actions import (
    ActionValue,
    ResettableActionAdapter,
    build_action_adapter,
)
from rl_fzerox.core.envs.actions.continuous_controls import action_drive_axis
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.envs.rewards import build_reward_tracker

from .controls import (
    ActionMaskBranches,
    ActionMaskController,
    ActionMaskSnapshot,
    ControlStateTracker,
    apply_dynamic_control_gates,
    sync_dynamic_action_masks,
)
from .episode import EngineEpisodeState
from .info import (
    backend_step_info,
    has_custom_baseline,
    read_live_telemetry,
    set_curriculum_info,
    telemetry_info,
)
from .observation import EngineObservationBuilder
from .reset import (
    EngineResetSeeds,
    SelectedTrack,
    TrackBaselineCache,
    TrackResetSelector,
    load_track_baseline,
    reset_race_state,
    sync_camera_setting,
)
from .stepping import (
    EngineStepAssembler,
    EnvStepRequest,
    WatchEnvStep,
    set_episode_boost_pad_info,
)


class FZeroXEnvEngine:
    """Environment step engine around one emulator backend.

    Rust owns the repeated inner-frame loop for one outer env step. Python
    consumes the returned step summary and stop state to apply reward shaping,
    build policy observations, and assemble Gym-facing info.
    """

    def __init__(
        self,
        *,
        backend: EmulatorBackend,
        config: EnvConfig,
        reward_config: RewardConfig | None = None,
        curriculum_config: CurriculumConfig | None = None,
        env_index: int = 0,
    ) -> None:
        self.backend = backend
        self.config = config
        self._curriculum_config = curriculum_config
        self._action_config = config.action.runtime()
        self._action_adapter = build_action_adapter(self._action_config)
        self._observation_builder = EngineObservationBuilder.from_engine_config(
            backend=backend,
            config=config,
        )
        self._reward_tracker = build_reward_tracker(
            config=reward_config,
            max_episode_steps=config.max_episode_steps,
        )
        self._reward_summary_config = self._reward_tracker.summary_config()
        self._action_space = self._action_adapter.action_space
        self._observation_space = self._observation_builder.space
        self._mask_controller = ActionMaskController.from_config(
            adapter=self._action_adapter,
            base_overrides=self._action_config.mask_overrides,
            curriculum_config=curriculum_config,
            boost_unmask_max_speed_kph=self._action_config.boost_unmask_max_speed_kph,
            lean_unmask_min_speed_kph=self._action_config.lean_unmask_min_speed_kph,
        )
        self._active_track_sampling = self._stage_track_sampling_config(
            self._mask_controller.stage_index
        )
        self._track_selector = TrackResetSelector(env_index=env_index)
        self._track_baseline_cache = TrackBaselineCache()
        self._control_state = ControlStateTracker(
            lean_mode=self._action_config.lean_mode,
            lean_initial_lockout_frames=self._action_config.lean_initial_lockout_frames,
            boost_decision_interval_frames=self._action_config.boost_decision_interval_frames,
            boost_request_lockout_frames=self._action_config.boost_request_lockout_frames,
            action_history_len=self._observation_builder.action_history_len,
            action_history_controls=self._observation_builder.action_history_controls,
        )
        self._step_assembler = EngineStepAssembler(
            backend=self.backend,
            config=self.config,
            action_config=self._action_config,
            reward_summary_config=self._reward_summary_config,
            reward_tracker=self._reward_tracker,
            observation_builder=self._observation_builder,
            mask_controller=self._mask_controller,
            control_state=self._control_state,
        )
        self._episode = EngineEpisodeState()
        self._reset_seeds = EngineResetSeeds()

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    @property
    def last_requested_control_state(self) -> ControllerState:
        return self._episode.last_requested_control_state

    @property
    def last_gas_level(self) -> float:
        return self._episode.last_gas_level

    def action_masks(self) -> ActionMask:
        """Return the flattened boolean action mask for the current stage."""

        return self._mask_controller.action_mask()

    def action_mask_branches(self) -> ActionMaskBranches:
        """Return the current action mask grouped by branch label."""

        return self._mask_controller.action_mask_branches()

    def action_mask_snapshot(self) -> ActionMaskSnapshot:
        """Return the flat mask and branch view from one mask computation."""

        return self._mask_controller.action_mask_snapshot()

    def set_curriculum_stage(self, stage_index: int) -> None:
        """Switch the active curriculum stage for subsequent action masks."""

        self._mask_controller.set_curriculum_stage(stage_index)
        self._active_track_sampling = self._stage_track_sampling_config(
            self._mask_controller.stage_index
        )

    def sync_checkpoint_curriculum_stage(self, stage_index: int | None) -> None:
        """Align watch-time stage masks with the loaded checkpoint metadata."""

        self._mask_controller.sync_checkpoint_stage(stage_index)
        self._active_track_sampling = self._stage_track_sampling_config(
            self._mask_controller.stage_index
        )

    @property
    def curriculum_stage_index(self) -> int | None:
        """Return the active curriculum stage index, if any."""

        return self._mask_controller.stage_index

    @property
    def curriculum_stage_name(self) -> str | None:
        """Return the active curriculum stage name, if any."""

        return self._mask_controller.stage_name

    def reset(self, seed: int | None = None) -> tuple[ObservationValue, dict[str, object]]:
        """Reset one episode and return the first policy observation."""

        self._reset_seeds.remember_reset_seed(seed)
        selected_track = self._select_reset_track(seed)
        self._episode.begin_reset(active_track=selected_track)
        if selected_track is not None:
            load_track_baseline(
                backend=self.backend,
                cache=self._track_baseline_cache,
                selected_track=selected_track,
                cache_enabled=self.config.cache_track_baselines,
            )
        _, info, telemetry = reset_race_state(
            backend=self.backend,
            config=self.config,
            sampled_track_baseline=selected_track is not None,
        )
        if selected_track is not None:
            info.update(selected_track.info())
        self._episode.uses_custom_baseline = selected_track is not None or has_custom_baseline(info)
        telemetry = self._maybe_randomize_game_rng(seed, telemetry, info)
        telemetry = sync_camera_setting(
            self.backend,
            target_name=self.config.camera_setting,
            telemetry=telemetry,
            info=info,
        )
        race_intro_info, telemetry = sync_race_intro_target(
            self.backend,
            target_timer=self.config.race_intro_target_timer,
        )
        info.update(race_intro_info)
        info.update(backend_step_info(self.backend))
        self.backend.set_controller_state(self._episode.held_controller_state)
        self._control_state.reset()
        self._mask_controller.set_lean_allowed_values(
            self._control_state.lean_action_mask_override()
        )
        sync_dynamic_action_masks(
            mask_controller=self._mask_controller,
            control_state=self._control_state,
            telemetry=telemetry,
            boost_min_energy_fraction=self.config.boost_min_energy_fraction,
        )
        self._episode.last_telemetry = telemetry
        self._reward_tracker.reset(
            telemetry,
            episode_seed=self._reward_episode_seed(seed),
        )
        if isinstance(self._action_adapter, ResettableActionAdapter):
            self._action_adapter.reset()
        info["seed"] = seed
        set_curriculum_info(
            info,
            stage_index=self.curriculum_stage_index,
            stage_name=self.curriculum_stage_name,
        )
        if telemetry is not None:
            info.update(telemetry_info(telemetry))
        info.update(self._reward_tracker.info(telemetry))
        set_episode_boost_pad_info(
            info,
            episode_boost_pad_entries=self._episode.boost_pad_entries,
        )
        image_observation = self._observation_builder.render_image()
        observation = self._observation_builder.build_observation(
            image=image_observation,
            telemetry=telemetry,
            control_state=self._control_state,
        )
        self._observation_builder.set_info(
            info,
            image_shape=tuple(int(value) for value in image_observation.shape),
        )
        self._episode.last_info = dict(info)
        self._reset_seeds.advance_reset_count()
        return observation, info

    def step(
        self,
        action: ActionValue,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self._step_control_state(
            self._action_adapter.decode(action),
            action_drive_axis=action_drive_axis(action, self._action_space),
        )

    def step_watch(self, action: ActionValue) -> WatchEnvStep:
        """Step a policy action while collecting watch-only intermediate frames."""

        return self._step_control_state_watch(
            self._action_adapter.decode(action),
            action_drive_axis=action_drive_axis(action, self._action_space),
        )

    def action_to_control_state(self, action: ActionValue) -> ControllerState:
        """Decode one policy action into the held controller state it represents."""

        return self._action_adapter.decode(action)

    def step_control(
        self,
        control_state: ControllerState,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self._step_control_state(control_state, action_drive_axis=None)

    def step_control_watch(self, control_state: ControllerState) -> WatchEnvStep:
        """Step manual controls while collecting watch-only intermediate frames."""

        return self._step_control_state_watch(control_state, action_drive_axis=None)

    def _step_control_state(
        self,
        control_state: ControllerState,
        *,
        action_drive_axis: float | None,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        requested_control_state = control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        self._episode.held_controller_state = applied_control_state
        return self._run_env_step(
            applied_control_state,
            action_repeat=self.config.action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
        )

    def _step_control_state_watch(
        self,
        control_state: ControllerState,
        *,
        action_drive_axis: float | None,
    ) -> WatchEnvStep:
        requested_control_state = control_state
        applied_control_state = self._apply_control_semantics(requested_control_state)
        self._episode.held_controller_state = applied_control_state
        return self._run_env_step_result(
            applied_control_state,
            action_repeat=self.config.action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            capture_display_frames=True,
        )

    def step_frame(
        self,
        control_state: ControllerState | None = None,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        """Advance one frame through the same reward path used by step()."""

        requested_control_state = (
            self._episode.held_controller_state if control_state is None else control_state
        )
        self._episode.held_controller_state = self._apply_control_semantics(requested_control_state)
        return self._run_env_step(
            self._episode.held_controller_state,
            action_repeat=1,
            requested_control_state=requested_control_state,
            action_drive_axis=None,
        )

    def render(self) -> RgbFrame:
        return self.backend.render_display(preset=self.config.observation.preset)

    def close(self) -> None:
        self.backend.close()

    def _maybe_randomize_game_rng(
        self,
        seed: int | None,
        telemetry: FZeroXTelemetry | None,
        info: dict[str, object],
    ) -> FZeroXTelemetry | None:
        if not self.config.randomize_game_rng_on_reset:
            return telemetry
        if self.config.randomize_game_rng_requires_race_mode and (
            telemetry is None or not telemetry.in_race_mode
        ):
            info["rng_randomized"] = False
            info["rng_randomization_skip_reason"] = "not_in_race"
            return telemetry

        rng_seed = self._reset_seeds.reset_rng_seed(seed)
        if rng_seed is None:
            return telemetry
        rng_state = self.backend.randomize_game_rng(rng_seed)
        info["rng_randomized"] = True
        info["rng_seed"] = rng_seed
        info["rng_state"] = rng_state
        return read_live_telemetry(self.backend) or telemetry

    def _reward_episode_seed(self, seed: int | None) -> int | None:
        return self._reset_seeds.reward_episode_seed(seed)

    def _select_reset_track(self, seed: int | None) -> SelectedTrack | None:
        return self._track_selector.select(
            self._active_track_sampling,
            seed=self._reset_seeds.track_sampling_seed(seed),
        )

    def _stage_track_sampling_config(self, stage_index: int | None) -> TrackSamplingConfig:
        if (
            self._curriculum_config is None
            or not self._curriculum_config.enabled
            or stage_index is None
        ):
            return self.config.track_sampling
        stage = self._curriculum_config.stages[stage_index]
        return stage.track_sampling or self.config.track_sampling

    def _run_env_step(
        self,
        control_state: ControllerState,
        *,
        action_repeat: int,
        requested_control_state: ControllerState | None = None,
        action_drive_axis: float | None = None,
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        return self._run_env_step_result(
            control_state,
            action_repeat=action_repeat,
            requested_control_state=requested_control_state,
            action_drive_axis=action_drive_axis,
            capture_display_frames=False,
        ).gym_result()

    def _run_env_step_result(
        self,
        control_state: ControllerState,
        *,
        action_repeat: int,
        requested_control_state: ControllerState | None,
        action_drive_axis: float | None,
        capture_display_frames: bool,
    ) -> WatchEnvStep:
        assembly = self._step_assembler.run(
            EnvStepRequest(
                control_state=control_state,
                action_repeat=action_repeat,
                requested_control_state=requested_control_state,
                action_drive_axis=action_drive_axis,
                capture_display_frames=capture_display_frames,
                active_track=self._episode.active_track,
                episode_return=self._episode.return_value,
                episode_boost_pad_entries=self._episode.boost_pad_entries,
                curriculum_stage_index=self.curriculum_stage_index,
                curriculum_stage_name=self.curriculum_stage_name,
            )
        )
        self._episode.record_step(
            telemetry=assembly.telemetry,
            requested_control_state=assembly.requested_control_state,
            gas_level=assembly.gas_level,
            return_value=assembly.episode_return,
            boost_pad_entries=assembly.episode_boost_pad_entries,
            done=assembly.step.terminated or assembly.step.truncated,
            info=assembly.step.info,
        )
        return assembly.step

    def _apply_control_semantics(self, control_state: ControllerState) -> ControllerState:
        """Apply telemetry gates and configured lean semantics."""

        gated_control_state = apply_dynamic_control_gates(
            control_state,
            mask_controller=self._mask_controller,
            continuous_air_brake_mode=self._action_config.continuous_air_brake_mode,
            last_telemetry=self._episode.last_telemetry,
        )
        return self._control_state.apply_lean_semantics(gated_control_state)
