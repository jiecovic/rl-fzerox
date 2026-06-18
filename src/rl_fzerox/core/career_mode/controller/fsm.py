# src/rl_fzerox/core/career_mode/controller/fsm.py
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rl_fzerox.core.career_mode.controller.lifecycle import (
    CareerControllerLifecycleEvents,
    CareerPostTerminalProgressSync,
    CareerRecordingSegmentStatus,
    CareerRecordingSegmentTracker,
    PostRaceContinuation,
    race_terminal_reason,
    recording_status_from_attempt_status,
    terminal_info,
)
from rl_fzerox.core.career_mode.controller.presentation import (
    CareerControllerViewState,
    career_debug_context,
    career_viewer_info,
)
from rl_fzerox.core.career_mode.controller.setup import (
    CareerCameraSync,
    CareerEngineSetupFlow,
    CareerMenuStepQueue,
    is_neutral_settle_step,
)
from rl_fzerox.core.career_mode.controller.setup.routing import CareerMenuRoutingMixin
from rl_fzerox.core.career_mode.execution.race import SaveRaceExecutionPlan
from rl_fzerox.core.career_mode.execution.save_file import SaveRamSession
from rl_fzerox.core.career_mode.execution.setup import (
    career_mode_race_setup_config,
    save_race_setup_from_config,
)
from rl_fzerox.core.career_mode.navigation import (
    CareerPhase,
    DifficultyPopupState,
    MenuFacts,
    MenuInput,
    RawMenuStep,
    in_gp_race,
    phase_from_step,
    raw_step,
)
from rl_fzerox.core.career_mode.policy import CareerModePolicyControl, CareerPolicyResolver
from rl_fzerox.core.career_mode.progress.attempt import (
    CareerAttemptProgress,
    CareerProgressTransition,
)
from rl_fzerox.core.envs.engine.reset.camera import (
    CameraSyncBackend,
)
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.models import SaveAttemptStatus
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig, WatchAppConfig


class CareerRuntimeEmulator(CameraSyncBackend, SaveRamSession, Protocol):
    """Emulator operations needed by the Career Mode controller."""


class CareerRuntimeSession(Protocol):
    """Narrow runtime surface Career Mode needs from its session loop."""

    @property
    def emulator(self) -> CareerRuntimeEmulator: ...


class CareerModeController(CareerMenuRoutingMixin):
    """Own the live Career Mode FSM and side-effect boundaries.

    The controller is the source of truth for menu/race state transitions while
    the watch runner is active. Persistent save progress is an output of those
    transitions, not the authority for live control or recording lifecycle.

    FSM map:
    1. BOOT_TO_DIFFICULTY navigates title/main-menu screens until GP mode and
       the configured difficulty are selected.
    2. ENTER_COURSE_SELECT, SELECT_CUP, SELECT_MACHINE, ENTER_MACHINE_SETTINGS,
       and APPLY_ENGINE prepare the requested cup/course/machine setup.
    3. ENTER_RACE waits for the game to expose a fresh GP race, resolves the
       policy checkpoint for that course, synchronizes camera state, and then
       enters POLICY_RACE.
    4. POLICY_RACE delegates control to the policy until a terminal race result
       is observed. Terminal results move the FSM into CONTINUE_AFTER_RACE.
    5. CONTINUE_AFTER_RACE advances result screens, next-course handoff,
       post-GP success ceremony/credits, game-over, and retry/menu exits. A
       next-course screen keeps the same cup attempt alive; a post-GP exit or
       game-over/menu exit ends the cup attempt segment.

    Side-effect ownership:
    - Save progress is written from terminal/progress transitions for manager UI
      sync and save history, but it is not queried back as live control truth.
    - Checkpoint refresh is armed only after a finished cup attempt and applied
      at the next policy handoff, never between courses of the same attempt.
    - Recording segments are full cup attempts. The FSM emits
      CareerRecordingSegmentClose when visible game flow leaves the attempt; the
      recorder only consumes that signal and writes files.
    """

    def __init__(
        self,
        setup: CareerModeRaceSetupConfig,
        *,
        db_path: Path,
        save_game_id: str,
        attempt_id: str,
        device: str,
        single_target: bool = False,
        perfect_run: bool = False,
        target_clear_goal: int = 0,
    ) -> None:
        self._setup = setup
        store = ManagerStore(db_path)
        self._progress = CareerAttemptProgress(
            store=store,
            save_game_id=save_game_id,
            attempt_id=attempt_id,
            single_target=single_target,
            perfect_run=perfect_run,
            target_clear_goal=target_clear_goal,
        )
        self._camera = CareerCameraSync()
        self._menu_steps = CareerMenuStepQueue()
        self._phase = CareerPhase.BOOT_TO_DIFFICULTY
        self._engine_setup = CareerEngineSetupFlow()
        self._difficulty_popup_state = DifficultyPopupState.CLOSED
        self._machine_selection_applied = False
        self._post_race = PostRaceContinuation()
        self._last_finished_attempt_id: str | None = None
        self._last_finished_attempt_status: SaveAttemptStatus | None = None
        self._last_finished_attempt_failure_reason: str | None = None
        self._refresh_policy_artifact_on_next_handoff = False
        self._emulator_reset_requested = False
        self._post_terminal_progress = CareerPostTerminalProgressSync()
        self._recording = CareerRecordingSegmentTracker()
        self._policy_resolver = CareerPolicyResolver(
            store=store,
            setup=self._setup,
            course_setups=self._progress.course_setups,
            device=device,
        )

    @classmethod
    def from_config(cls, config: WatchAppConfig) -> CareerModeController:
        setup = save_race_setup_from_config(config)
        if config.watch.manager_db_path is None:
            raise RuntimeError("career mode requires watch.manager_db_path")
        if config.watch.managed_save_game_id is None:
            raise RuntimeError("career mode requires watch.managed_save_game_id")
        if config.watch.save_attempt_id is None:
            raise RuntimeError("career mode requires watch.save_attempt_id")
        return cls(
            setup,
            db_path=config.watch.manager_db_path,
            save_game_id=config.watch.managed_save_game_id,
            attempt_id=config.watch.save_attempt_id,
            device=config.watch.device,
            single_target=config.watch.single_save_target,
            perfect_run=config.watch.single_save_target_perfect,
            target_clear_goal=config.watch.single_save_target_clear_goal,
        )

    def active_policy_control(
        self,
        *,
        session: CareerRuntimeSession,
        current_policy_control: CareerModePolicyControl | None,
        info: dict[str, object],
    ) -> CareerModePolicyControl | None:
        del session
        if self._phase != CareerPhase.POLICY_RACE or not in_gp_race(info):
            return None
        policy_control = self._resolve_policy_control(info)
        if policy_control is not None:
            return policy_control
        return current_policy_control

    def before_step(
        self,
        *,
        session: CareerRuntimeSession,
        info: dict[str, object],
    ) -> bool:
        if self._sync_post_terminal_save_progress(session=session, info=info):
            return True
        return self._sync_camera_before_policy_handoff(session=session, info=info)

    def next_raw_step(self, *, info: dict[str, object]) -> RawMenuStep | None:
        facts = MenuFacts.from_info(info)
        if self._progress.attempt_id is None:
            return None
        if self._phase == CareerPhase.POLICY_RACE:
            if facts.in_gp_race:
                if facts.terminal_race_result:
                    return self._continue_terminal_race_result_step()
                return None
            raise RuntimeError("Career Mode left a race before observing a game result")

        if self._phase == CareerPhase.CONTINUE_AFTER_RACE:
            pending_step = self._menu_steps.peek()
            if pending_step is not None and is_neutral_settle_step(pending_step):
                step = self._menu_steps.pop()
                self._phase = phase_from_step(step)
                return step
            if facts.is_gp_result_screen:
                if facts.terminal_race_result:
                    self._post_race.observe_terminal_result()
                self._progress.observe_post_race_screen(info=info, setup=self._setup)
                if self._post_race.continue_observed_result():
                    return self._continue_after_race_pulse()
                self._menu_steps.clear()
                return raw_step(
                    MenuInput.NEUTRAL,
                    1,
                    phase="continue_after_race:wait_for_terminal_result",
                )
            if facts.in_gp_race:
                if facts.terminal_race_result:
                    self._post_race.observe_terminal_result()
                if (
                    self._post_race.continuing_race_result
                    and self._post_race.observed_terminal_race_result
                ):
                    if self._post_race.new_race_ready_for_policy(info):
                        self._post_race.stop_result_continuation()
                    else:
                        return self._continue_after_race_pulse()
                self._post_race.stop_result_continuation()
                self._menu_steps.clear()
                if not self._post_race.new_race_ready_for_policy(info):
                    return raw_step(
                        MenuInput.NEUTRAL,
                        1,
                        phase="continue_after_race:wait_for_fresh_race",
                    )
                if self._resolve_policy_control(info, refresh_artifact=True) is None:
                    return self._wait_for_policy_resolution()
                if camera_step := self._next_camera_sync_step(info):
                    return camera_step
                return self._enter_policy_race()
            if facts.is_post_gp_screen:
                self._menu_steps.clear()
                return self._continue_post_gp_screen_step()
            if facts.is_gp_next_course_screen:
                self._menu_steps.clear()
                return self._continue_next_course_step()
            if self._menu_steps:
                pending_step = self._menu_steps.peek()
                if pending_step is None or not is_neutral_settle_step(pending_step):
                    self._menu_steps.clear()
                    return raw_step(
                        MenuInput.NEUTRAL,
                        1,
                        phase="continue_after_race:wait_for_known_screen",
                    )
                step = self._menu_steps.pop()
                self._phase = phase_from_step(step)
                return step
            return self._next_menu_step(info)
        if facts.in_gp_race:
            self._menu_steps.clear()
            self._menu_steps.reset_advance_presses()
            if facts.terminal_race_result:
                return self._continue_terminal_race_result_step()
            if (
                self._post_race.awaiting_new_race_after_terminal
                and not self._post_race.new_race_ready_for_policy(info)
            ):
                return raw_step(
                    MenuInput.NEUTRAL,
                    1,
                    phase="enter_race:wait_for_fresh_race",
                )
            if self._resolve_policy_control(info, refresh_artifact=True) is None:
                return self._wait_for_policy_resolution()
            if camera_step := self._next_camera_sync_step(info):
                return camera_step
            return self._enter_policy_race()
        if self._menu_steps:
            if step := self._next_pending_menu_step(facts):
                return step
        return self._next_menu_step(info)

    def _enter_phase(self, phase: CareerPhase) -> None:
        if self._phase is not phase:
            self._menu_steps.reset_advance_presses()
        self._phase = phase

    def policy_owns_control(self) -> bool:
        return self._phase == CareerPhase.POLICY_RACE

    def has_active_attempt(self) -> bool:
        return self._progress.attempt_id is not None

    @property
    def phase(self) -> CareerPhase:
        return self._phase

    def viewer_info(
        self,
        *,
        info: dict[str, object],
        active_policy_control: CareerModePolicyControl | None,
    ) -> dict[str, object]:
        facts = MenuFacts.from_info(info)
        return career_viewer_info(
            info=info,
            state=self._view_state(),
            observed_screen=self._observed_menu_screen(facts),
            active_policy_control=active_policy_control,
        )

    def debug_context(self, info: dict[str, object]) -> str:
        return career_debug_context(info=info, state=self._view_state())

    def drain_lifecycle_events(self) -> CareerControllerLifecycleEvents:
        return CareerControllerLifecycleEvents(
            recording_close=self._recording.pop_close(),
            emulator_reset_requested=self._consume_emulator_reset_request(),
            has_active_attempt=self.has_active_attempt(),
        )

    def _consume_emulator_reset_request(self) -> bool:
        requested = self._emulator_reset_requested
        self._emulator_reset_requested = False
        return requested

    def _view_state(self) -> CareerControllerViewState:
        return CareerControllerViewState(
            attempt_id=self._progress.attempt_id,
            setup=self._setup,
            progress=self._progress.unlock_progress,
            phase=self._phase,
            pending_step_count=self._menu_steps.pending_count,
            difficulty_popup_state=self._difficulty_popup_state,
            camera_synced=self._camera.synced,
            camera_setting=self._camera.target,
            awaiting_new_race_after_terminal=self._post_race.awaiting_new_race_after_terminal,
            continuing_race_result=self._post_race.continuing_race_result,
            fresh_race_ready_frames=self._post_race.fresh_race_ready_frames,
            last_finished_attempt_id=self._last_finished_attempt_id,
            last_finished_attempt_status=self._last_finished_attempt_status,
            last_finished_attempt_failure_reason=self._last_finished_attempt_failure_reason,
        )

    def _resolve_policy_control(
        self,
        info: dict[str, object],
        *,
        refresh_artifact: bool = False,
    ) -> CareerModePolicyControl | None:
        should_refresh_artifact = refresh_artifact and self._refresh_policy_artifact_on_next_handoff
        resolution = self._policy_resolver.resolve(
            info,
            refresh_artifact=should_refresh_artifact,
        )
        if resolution is None:
            return None
        if should_refresh_artifact:
            self._refresh_policy_artifact_on_next_handoff = False
        if resolution.activated_new_policy:
            self._camera.set_target(resolution.camera_setting)
        return resolution.control

    def _enter_policy_race(self) -> RawMenuStep:
        self._phase = CareerPhase.POLICY_RACE
        return raw_step(MenuInput.NEUTRAL, 1, phase="policy_race:handoff")

    def _continue_terminal_race_result_step(self) -> RawMenuStep:
        self._enter_continue_after_race()
        return self._continue_after_race_pulse()

    @staticmethod
    def _wait_for_policy_resolution() -> RawMenuStep:
        return raw_step(MenuInput.NEUTRAL, 1, phase="policy_resolution:wait")

    def observe_step(
        self,
        *,
        session: CareerRuntimeSession,
        info: dict[str, object],
    ) -> bool:
        if self._phase == CareerPhase.CONTINUE_AFTER_RACE:
            return False
        terminal_reason = race_terminal_reason(
            session=session,
            info=info,
        )
        if terminal_reason is None:
            return False
        self._reset_camera_sync()
        resolved_terminal_info = terminal_info(
            session=session,
            info=info,
            terminal_reason=terminal_reason,
        )
        self._recording.observe_terminal_result(resolved_terminal_info)
        info.update(resolved_terminal_info)
        transition = self._progress.handle_terminal_race(
            session=session,
            setup=self._setup,
            info=resolved_terminal_info,
        )
        self._apply_progress_transition(transition)
        if transition.reset_emulator:
            self._request_emulator_reset_for_next_attempt(
                recording_status=recording_status_from_attempt_status(transition.finished_status),
            )
            return True
        if self._progress.attempt_id is None:
            self._enter_continue_after_race()
            return True
        self._enter_continue_after_race()
        return True

    def _apply_progress_transition(self, transition: CareerProgressTransition) -> None:
        self._remember_finished_attempt(transition)
        self._close_recording_from_transition(transition)
        if transition.next_plan is not None:
            self._apply_execution_plan(transition.next_plan)

    def _remember_finished_attempt(self, transition: object) -> None:
        finished_attempt_id = getattr(transition, "finished_attempt_id", None)
        finished_status = getattr(transition, "finished_status", None)
        if not isinstance(finished_attempt_id, str) or finished_status not in {
            "succeeded",
            "failed",
        }:
            return
        self._last_finished_attempt_id = finished_attempt_id
        self._last_finished_attempt_status = finished_status
        failure_reason = getattr(transition, "finished_failure_reason", None)
        self._last_finished_attempt_failure_reason = (
            failure_reason if isinstance(failure_reason, str) else None
        )
        self._refresh_policy_artifact_on_next_handoff = True

    def _close_recording_from_transition(self, transition: CareerProgressTransition) -> None:
        if not transition.attempt_finished:
            return
        self._recording.close(
            status=recording_status_from_attempt_status(transition.finished_status),
        )

    def _apply_execution_plan(self, plan: SaveRaceExecutionPlan) -> None:
        self._progress.apply_execution_plan(plan)
        self._setup = career_mode_race_setup_config(plan.race_setup)
        self._menu_steps.clear()
        self._phase = CareerPhase.CONTINUE_AFTER_RACE
        self._engine_setup.reset_adjustment()
        self._difficulty_popup_state = DifficultyPopupState.CLOSED
        self._machine_selection_applied = False
        self._menu_steps.reset_advance_presses()
        self._policy_resolver.update_context(
            setup=self._setup,
            course_setups=self._progress.course_setups,
        )
        self._post_race.enter_continue_after_race()
        self._post_terminal_progress.reset()
        self._reset_camera_sync()

    def _enter_continue_after_race(self) -> None:
        self._menu_steps.clear()
        self._phase = CareerPhase.CONTINUE_AFTER_RACE
        self._menu_steps.reset_advance_presses()
        self._engine_setup.reset_tap_budget()
        self._difficulty_popup_state = DifficultyPopupState.CLOSED
        self._machine_selection_applied = False
        self._post_race.enter_continue_after_race()
        self._post_terminal_progress.reset()
        self._reset_camera_sync()

    def _camera_ready(self, info: dict[str, object]) -> bool:
        return self._camera.ready(info)

    def _next_camera_sync_step(
        self,
        info: dict[str, object],
    ) -> RawMenuStep | None:
        return self._camera.next_menu_step(info)

    def _sync_camera_before_policy_handoff(
        self,
        *,
        session: CareerRuntimeSession,
        info: dict[str, object],
    ) -> bool:
        return self._camera.sync_before_policy_handoff(
            session=session,
            info=info,
            phase=self._phase,
        )

    def _sync_post_terminal_save_progress(
        self,
        *,
        session: CareerRuntimeSession,
        info: dict[str, object],
    ) -> bool:
        if (
            self._phase != CareerPhase.CONTINUE_AFTER_RACE
            or self._progress.attempt_id is None
            or not self._post_race.observed_terminal_race_result
        ):
            return False
        facts = MenuFacts.from_info(info)
        transition = self._post_terminal_progress.sync(
            session=session,
            setup=self._setup,
            progress=self._progress,
            recording=self._recording,
            post_race=self._post_race,
            facts=facts,
            info=info,
        )
        if transition is None:
            return False

        self._apply_progress_transition(transition)
        if transition.reset_emulator:
            self._request_emulator_reset_for_next_attempt(
                recording_status=recording_status_from_attempt_status(transition.finished_status),
            )
        return transition.attempt_finished

    def _request_emulator_reset_for_next_attempt(
        self,
        *,
        recording_status: CareerRecordingSegmentStatus,
    ) -> None:
        """Restart from title when the current game screen cannot reach the next attempt."""

        self._emulator_reset_requested = True
        self._recording.force_close(status=recording_status)
        self._menu_steps.clear()
        self._phase = CareerPhase.BOOT_TO_DIFFICULTY
        self._engine_setup.reset_adjustment()
        self._difficulty_popup_state = DifficultyPopupState.CLOSED
        self._machine_selection_applied = False
        self._menu_steps.reset_advance_presses()
        self._post_race.reset()
        self._post_terminal_progress.reset()
        self._reset_camera_sync()

    def _reset_camera_sync(self) -> None:
        self._camera.reset()

    def _continue_after_race_pulse(self) -> RawMenuStep:
        self._menu_steps.clear()
        step, settle_step = self._post_race.continue_after_race_pulse()
        self._menu_steps.append(settle_step)
        return step
