# src/rl_fzerox/core/career_mode/runner/controller.py
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Protocol

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.career_mode.runner.camera import CareerCameraSync
from rl_fzerox.core.career_mode.runner.menu import (
    GP_MENU_ORDER,
    MENU_TIMING,
    CareerPhase,
    DifficultyPopupState,
    MenuFacts,
    MenuInput,
    ObservedMenuScreen,
    RawMenuStep,
    continue_after_race_step,
    continue_next_course_step,
    course_id_from_info,
    engine_adjust_steps,
    in_gp_race,
    machine_select_steps,
    observed_menu_screen,
    phase_from_step,
    raw_step,
)
from rl_fzerox.core.career_mode.runner.policy import CareerModePolicyControl
from rl_fzerox.core.career_mode.runner.policy_resolver import CareerPolicyResolver
from rl_fzerox.core.career_mode.runner.progress import (
    CareerAttemptProgress,
)
from rl_fzerox.core.career_mode.runner.race import SaveRaceExecutionPlan
from rl_fzerox.core.career_mode.runner.save_file import SaveRamSession
from rl_fzerox.core.career_mode.runner.setup import (
    career_mode_race_setup_config,
    save_race_setup_from_config,
)
from rl_fzerox.core.career_mode.runner.view import (
    CareerControllerViewState,
    career_debug_context,
    career_viewer_info,
)
from rl_fzerox.core.domain.race_difficulty import (
    is_race_difficulty_name,
    race_difficulty_raw_value,
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


class CareerModeController:
    """Drive the menu path until the configured policy can control a race."""

    def __init__(
        self,
        setup: CareerModeRaceSetupConfig,
        *,
        db_path: Path,
        save_game_id: str,
        attempt_id: str,
        device: str,
        single_target: bool = False,
    ) -> None:
        self._setup = setup
        store = ManagerStore(db_path)
        self._progress = CareerAttemptProgress(
            store=store,
            save_game_id=save_game_id,
            attempt_id=attempt_id,
            single_target=single_target,
        )
        self._camera = CareerCameraSync()
        self._pending_steps: deque[RawMenuStep] = deque()
        self._phase = CareerPhase.BOOT_TO_DIFFICULTY
        self._engine_applied_course_id: str | None = None
        self._engine_adjust_taps = 0
        self._engine_ready_course_id: str | None = None
        self._engine_ready_target: int | None = None
        self._engine_ready_frames = 0
        self._difficulty_popup_state = DifficultyPopupState.CLOSED
        self._machine_selection_applied = False
        self._advance_presses_in_phase = 0
        self._awaiting_new_race_after_terminal = False
        self._continuing_race_result = False
        self._observed_terminal_race_result = False
        self._continue_after_race_pulses = 0
        self._last_progress_sync_continue_pulse = -1
        self._fresh_race_ready_frames = 0
        self._last_finished_attempt_id: str | None = None
        self._last_finished_attempt_status: SaveAttemptStatus | None = None
        self._last_finished_attempt_failure_reason: str | None = None
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
            raise RuntimeError("Career Mode has no active save attempt")
        if self._phase == CareerPhase.POLICY_RACE:
            if facts.in_gp_race:
                if facts.terminal_race_result:
                    return self._continue_terminal_race_result_step()
                return None
            raise RuntimeError("Career Mode left a race before observing a game result")

        if self._phase == CareerPhase.CONTINUE_AFTER_RACE:
            if self._pending_steps and _is_neutral_settle_step(self._pending_steps[0]):
                step = self._pending_steps.popleft()
                self._phase = phase_from_step(step)
                return step
            if facts.is_gp_result_screen:
                if facts.terminal_race_result:
                    self._observed_terminal_race_result = True
                    self._continuing_race_result = True
                if self._observed_terminal_race_result:
                    self._continuing_race_result = True
                    return self._continue_after_race_pulse()
                self._pending_steps.clear()
                return raw_step(
                    MenuInput.NEUTRAL,
                    1,
                    phase="continue_after_race:wait_for_terminal_result",
                )
            if facts.in_gp_race:
                if facts.terminal_race_result:
                    self._observed_terminal_race_result = True
                    self._continuing_race_result = True
                if self._continuing_race_result and self._observed_terminal_race_result:
                    if self._new_race_ready_for_policy(info):
                        self._continuing_race_result = False
                    else:
                        return self._continue_after_race_pulse()
                self._continuing_race_result = False
                self._pending_steps.clear()
                if not self._new_race_ready_for_policy(info):
                    return raw_step(
                        MenuInput.NEUTRAL,
                        1,
                        phase="continue_after_race:wait_for_fresh_race",
                    )
                if self._resolve_policy_control(info) is None:
                    return self._wait_for_policy_resolution()
                if camera_step := self._next_camera_sync_step(info):
                    return camera_step
                return self._enter_policy_race()
            if facts.is_post_gp_screen:
                self._pending_steps.clear()
                return self._continue_post_gp_screen_step()
            if facts.is_gp_next_course_screen:
                self._pending_steps.clear()
                return self._continue_next_course_step()
            if self._pending_steps:
                if not _is_neutral_settle_step(self._pending_steps[0]):
                    self._pending_steps.clear()
                    return raw_step(
                        MenuInput.NEUTRAL,
                        1,
                        phase="continue_after_race:wait_for_known_screen",
                    )
                step = self._pending_steps.popleft()
                self._phase = phase_from_step(step)
                return step
            return self._next_menu_step(info)
        if facts.in_gp_race:
            self._pending_steps.clear()
            self._advance_presses_in_phase = 0
            if facts.terminal_race_result:
                return self._continue_terminal_race_result_step()
            if self._awaiting_new_race_after_terminal and not self._new_race_ready_for_policy(info):
                return raw_step(
                    MenuInput.NEUTRAL,
                    1,
                    phase="enter_race:wait_for_fresh_race",
                )
            if self._resolve_policy_control(info) is None:
                return self._wait_for_policy_resolution()
            if camera_step := self._next_camera_sync_step(info):
                return camera_step
            return self._enter_policy_race()
        if self._pending_steps:
            if step := self._next_pending_menu_step(facts):
                return step
        return self._next_menu_step(info)

    def _next_menu_step(self, info: dict[str, object]) -> RawMenuStep | None:
        facts = MenuFacts.from_info(info)
        for _ in range(8):
            screen = self._observed_menu_screen(facts)
            match screen:
                case ObservedMenuScreen.TITLE:
                    self._difficulty_popup_state = DifficultyPopupState.CLOSED
                    return self._start_until_phase("title_to_main_menu")
                case ObservedMenuScreen.MAIN_MENU_GP:
                    self._phase = CareerPhase.SELECT_DIFFICULTY
                    return self._open_difficulty_popup()
                case ObservedMenuScreen.MAIN_MENU_OTHER:
                    self._phase = CareerPhase.BOOT_TO_DIFFICULTY
                    self._difficulty_popup_state = DifficultyPopupState.CLOSED
                    self._pending_steps.clear()
                    if gp_step := self._main_menu_step_toward_gp(facts):
                        return gp_step
                    return raw_step(
                        MenuInput.NEUTRAL,
                        1,
                        phase="main_menu:wait_for_gp_cursor",
                    )
                case ObservedMenuScreen.DIFFICULTY_POPUP:
                    self._phase = CareerPhase.SELECT_DIFFICULTY
                    self._advance_presses_in_phase = 0
                    return self._select_difficulty_popup_step(facts)
                case ObservedMenuScreen.DIFFICULTY_CONFIRM:
                    self._phase = CareerPhase.ENTER_COURSE_SELECT
                    self._advance_presses_in_phase = 0
                    return self._confirm_difficulty_popup_step(facts)
                case ObservedMenuScreen.TRANSITION:
                    return raw_step(
                        MenuInput.NEUTRAL,
                        1,
                        phase=f"{self._phase.value}:wait_transition",
                    )
                case ObservedMenuScreen.COURSE_SELECT:
                    self._difficulty_popup_state = DifficultyPopupState.CLOSED
                    self._machine_selection_applied = False
                    if not self._course_select_matches_target(facts):
                        return self._leave_wrong_difficulty_course_select()
                    selected_cup_index = facts.course_select_cup_index
                    target_cup_index = GP_MENU_ORDER.cup_right_count(self._setup.cup_id)
                    if selected_cup_index == target_cup_index:
                        self._enter_phase(CareerPhase.ENTER_MACHINE_SELECT)
                        return self._accept_until_phase("enter_machine_select")
                    cup_input = _cup_selection_input(
                        selected_cup_index=selected_cup_index,
                        target_cup_index=target_cup_index,
                    )
                    self._enter_phase(CareerPhase.SELECT_CUP)
                    return self._queue_tap(
                        cup_input,
                        hold_frames=MENU_TIMING.menu_hold_frames,
                        settle_frames=MENU_TIMING.menu_settle_frames,
                        phase=f"select_cup:{cup_input.value}",
                    )
                case ObservedMenuScreen.MACHINE_SELECT:
                    self._difficulty_popup_state = DifficultyPopupState.CLOSED
                    if not self._machine_selection_applied:
                        self._enter_phase(CareerPhase.SELECT_MACHINE)
                        if not self._pending_steps:
                            route_steps = machine_select_steps(self._setup)
                            self._pending_steps.extend(route_steps)
                            self._advance_presses_in_phase = 0
                            if not route_steps:
                                self._machine_selection_applied = True
                        if self._pending_steps:
                            return self._next_pending_menu_step(facts)
                    self._enter_phase(CareerPhase.ENTER_MACHINE_SETTINGS)
                    return self._accept_until_phase("enter_machine_settings")
                case ObservedMenuScreen.MACHINE_SETTINGS:
                    self._difficulty_popup_state = DifficultyPopupState.CLOSED
                    course_id = course_id_from_info(info)
                    if course_id is None:
                        return raw_step(
                            MenuInput.NEUTRAL,
                            1,
                            phase="apply_engine:wait_for_course",
                        )
                    if self._engine_applied_course_id != course_id:
                        self._enter_phase(CareerPhase.APPLY_ENGINE)
                        return self._apply_engine_step(info)
                    self._enter_phase(CareerPhase.ENTER_RACE)
                    return self._accept_until_phase("enter_race")
                case ObservedMenuScreen.GP_RACE:
                    self._pending_steps.clear()
                    self._advance_presses_in_phase = 0
                    if facts.terminal_race_result:
                        return self._continue_terminal_race_result_step()
                    if self._resolve_policy_control(info) is None:
                        return self._wait_for_policy_resolution()
                    if camera_step := self._next_camera_sync_step(info):
                        return camera_step
                    return self._enter_policy_race()
                case ObservedMenuScreen.RESULTS:
                    self._phase = CareerPhase.CONTINUE_AFTER_RACE
                    return self._continue_after_race_pulse()
                case ObservedMenuScreen.GP_NEXT_COURSE:
                    self._phase = CareerPhase.CONTINUE_AFTER_RACE
                    return self._continue_next_course_step()
                case ObservedMenuScreen.POST_GP:
                    self._phase = CareerPhase.CONTINUE_AFTER_RACE
                    return self._continue_post_gp_screen_step()
                case ObservedMenuScreen.UNKNOWN:
                    return raw_step(
                        MenuInput.NEUTRAL,
                        1,
                        phase="menu:wait_for_observed_screen",
                    )
        return raw_step(MenuInput.NEUTRAL, 1, phase=self._phase.value)

    def _next_pending_menu_step(self, facts: MenuFacts) -> RawMenuStep | None:
        """Pop a queued menu step only while its owning screen is still visible."""

        if not self._pending_steps:
            return None
        step = self._pending_steps[0]
        screen = self._observed_menu_screen(facts)
        if not _pending_step_matches_observed_screen(step, screen):
            self._pending_steps.clear()
            return None
        step = self._pending_steps.popleft()
        if step.phase == "main_menu:open_difficulty:settle":
            self._difficulty_popup_state = DifficultyPopupState.OPEN
        if step.phase.startswith("select_difficulty:accept"):
            self._difficulty_popup_state = DifficultyPopupState.SUBMITTED
        self._phase = phase_from_step(step)
        if step.phase.startswith("select_machine") and not self._pending_steps:
            self._machine_selection_applied = True
        return step

    def _enter_phase(self, phase: CareerPhase) -> None:
        if self._phase is not phase:
            self._advance_presses_in_phase = 0
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

    def _view_state(self) -> CareerControllerViewState:
        return CareerControllerViewState(
            attempt_id=self._progress.attempt_id,
            setup=self._setup,
            progress=self._progress.unlock_progress,
            phase=self._phase,
            pending_step_count=len(self._pending_steps),
            difficulty_popup_state=self._difficulty_popup_state,
            camera_synced=self._camera.synced,
            camera_setting=self._camera.target,
            awaiting_new_race_after_terminal=self._awaiting_new_race_after_terminal,
            continuing_race_result=self._continuing_race_result,
            fresh_race_ready_frames=self._fresh_race_ready_frames,
            last_finished_attempt_id=self._last_finished_attempt_id,
            last_finished_attempt_status=self._last_finished_attempt_status,
            last_finished_attempt_failure_reason=self._last_finished_attempt_failure_reason,
        )

    def _observed_menu_screen(self, facts: MenuFacts) -> ObservedMenuScreen:
        return observed_menu_screen(
            facts,
            difficulty_popup_state=self._difficulty_popup_state,
        )

    def _resolve_policy_control(
        self,
        info: dict[str, object],
    ) -> CareerModePolicyControl | None:
        resolution = self._policy_resolver.resolve(info)
        if resolution is None:
            return None
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

    def _start_until_phase(self, phase: str) -> RawMenuStep:
        return self._tap_until_phase(
            MenuInput.START,
            hold_frames=MENU_TIMING.start_hold_frames,
            settle_frames=MENU_TIMING.start_settle_frames,
            phase=phase,
            label="start",
        )

    def _accept_until_phase(self, phase: str) -> RawMenuStep:
        return self._tap_until_phase(
            MenuInput.ACCEPT,
            hold_frames=MENU_TIMING.start_hold_frames,
            settle_frames=MENU_TIMING.start_settle_frames,
            phase=phase,
            label="accept",
        )

    def _tap_until_phase(
        self,
        menu_input: MenuInput,
        *,
        hold_frames: int,
        settle_frames: int,
        phase: str,
        label: str,
    ) -> RawMenuStep:
        self._advance_presses_in_phase += 1
        if self._advance_presses_in_phase > MENU_TIMING.max_advance_presses_per_phase:
            raise RuntimeError(
                f"Career Mode menu phase {phase!r} did not reach the expected screen"
            )
        return self._queue_tap(
            menu_input,
            hold_frames=hold_frames,
            settle_frames=settle_frames,
            phase=f"{phase}:{label}:{self._advance_presses_in_phase}",
        )

    def _queue_tap(
        self,
        menu_input: MenuInput,
        *,
        hold_frames: int,
        settle_frames: int,
        phase: str,
    ) -> RawMenuStep:
        self._pending_steps.append(
            raw_step(MenuInput.NEUTRAL, settle_frames, phase=f"{phase}:settle")
        )
        return raw_step(menu_input, hold_frames, phase=phase)

    def _queue_menu_steps(self, steps: tuple[RawMenuStep, ...]) -> RawMenuStep:
        if not steps:
            return raw_step(MenuInput.NEUTRAL, 1, phase="menu_steps:empty")
        self._pending_steps.extend(steps[1:])
        return steps[0]

    def _reset_engine_adjustment(self) -> None:
        self._engine_applied_course_id = None
        self._engine_adjust_taps = 0
        self._engine_ready_course_id = None
        self._engine_ready_target = None
        self._engine_ready_frames = 0

    def _reset_engine_confirmation(self) -> None:
        self._engine_ready_course_id = None
        self._engine_ready_target = None
        self._engine_ready_frames = 0

    def _open_difficulty_popup(self) -> RawMenuStep:
        self._pending_steps.clear()
        self._difficulty_popup_state = DifficultyPopupState.OPENING
        self._advance_presses_in_phase = 0
        return self._queue_tap(
            MenuInput.ACCEPT,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.difficulty_popup_open_settle_frames,
            phase="main_menu:open_difficulty",
        )

    def _main_menu_step_toward_gp(self, facts: MenuFacts) -> RawMenuStep | None:
        if facts.selected_gp_mode:
            return None
        match facts.selected_mode_raw:
            case 5:
                menu_input = MenuInput.UP
                phase = "main_menu:practice_to_gp"
            case 1 | 2 | 3:
                menu_input = MenuInput.LEFT
                phase = "main_menu:left_to_gp"
            case 4:
                menu_input = MenuInput.LEFT
                phase = "main_menu:options_to_practice"
            case _:
                return raw_step(
                    MenuInput.NEUTRAL,
                    1,
                    phase="main_menu:wait_for_known_cursor",
                )
        return self._queue_tap(
            menu_input,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.menu_settle_frames,
            phase=phase,
        )

    def _select_difficulty_popup_step(self, facts: MenuFacts) -> RawMenuStep:
        self._difficulty_popup_state = DifficultyPopupState.OPEN
        if not facts.selected_gp_mode:
            self._pending_steps.clear()
            self._difficulty_popup_state = DifficultyPopupState.CLOSED
            if gp_step := self._main_menu_step_toward_gp(facts):
                return gp_step
            return raw_step(
                MenuInput.NEUTRAL,
                1,
                phase="difficulty_popup:lost_gp_cursor",
            )
        if not facts.has_difficulty_popup:
            self._pending_steps.clear()
            return raw_step(
                MenuInput.NEUTRAL,
                1,
                phase="difficulty_popup:wait_visible",
            )
        target = self._setup_difficulty_raw_value()
        current = facts.difficulty_cursor_raw
        if current is None:
            self._pending_steps.clear()
            return raw_step(
                MenuInput.NEUTRAL,
                1,
                phase="difficulty_popup:wait_difficulty_read",
            )
        if current == target:
            self._difficulty_popup_state = DifficultyPopupState.SUBMITTED
            return self._queue_tap(
                MenuInput.ACCEPT,
                hold_frames=MENU_TIMING.menu_hold_frames,
                settle_frames=MENU_TIMING.menu_settle_frames,
                phase="select_difficulty:accept",
            )
        menu_input = MenuInput.DOWN if current < target else MenuInput.UP
        direction = "down" if menu_input is MenuInput.DOWN else "up"
        return self._queue_tap(
            menu_input,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.menu_settle_frames,
            phase=f"select_difficulty:{direction}",
        )

    def _confirm_difficulty_popup_step(self, facts: MenuFacts) -> RawMenuStep:
        if facts.difficulty_raw != self._setup_difficulty_raw_value():
            self._pending_steps.clear()
            self._difficulty_popup_state = DifficultyPopupState.OPEN
            return raw_step(
                MenuInput.NEUTRAL,
                1,
                phase="select_difficulty:wait_commit",
            )
        self._difficulty_popup_state = DifficultyPopupState.SUBMITTED
        return self._queue_tap(
            MenuInput.ACCEPT,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.menu_settle_frames,
            phase="enter_course_select:confirm_difficulty",
        )

    def _apply_engine_step(self, info: dict[str, object]) -> RawMenuStep:
        facts = MenuFacts.from_info(info)
        if not facts.is_machine_settings:
            return raw_step(MenuInput.NEUTRAL, 1, phase="apply_engine:wait_for_settings")
        course_id = course_id_from_info(info)
        if course_id is None:
            return raw_step(MenuInput.NEUTRAL, 1, phase="apply_engine:wait_for_course")
        course_setup = self._policy_resolver.resolve_course_setup(info)
        if course_setup is None:
            return raw_step(MenuInput.NEUTRAL, 1, phase="apply_engine:wait_for_setup")
        current = facts.engine_setting_raw_value
        target = course_setup.engine_setting_raw_value
        if current == target:
            self._engine_adjust_taps = 0
            if self._engine_ready_course_id == course_id and self._engine_ready_target == target:
                self._engine_ready_frames += 1
            else:
                self._engine_ready_course_id = course_id
                self._engine_ready_target = target
                self._engine_ready_frames = 1
            if self._engine_ready_frames >= MENU_TIMING.engine_ready_confirm_frames:
                self._engine_applied_course_id = course_id
                return raw_step(MenuInput.NEUTRAL, 1, phase="apply_engine:ready")
            return raw_step(MenuInput.NEUTRAL, 1, phase="apply_engine:confirm")
        self._reset_engine_confirmation()
        if current is None:
            return raw_step(MenuInput.NEUTRAL, 1, phase="apply_engine:wait_for_read")
        remaining_taps = MENU_TIMING.max_engine_adjust_taps - self._engine_adjust_taps
        if remaining_taps <= 0:
            raise RuntimeError(
                f"Career Mode could not reach the requested engine setting {target} from {current}"
            )
        steps = engine_adjust_steps(
            current=current,
            target=target,
            max_taps=remaining_taps,
        )
        self._engine_adjust_taps += _engine_adjust_tap_count(steps)
        return self._queue_menu_steps(steps)

    def _course_select_matches_target(self, facts: MenuFacts) -> bool:
        return facts.difficulty_raw == self._setup_difficulty_raw_value()

    def _setup_difficulty_raw_value(self) -> int:
        difficulty = self._setup.difficulty
        if not is_race_difficulty_name(difficulty):
            known = ", ".join(GP_MENU_ORDER.difficulties)
            raise RuntimeError(f"unknown Career Mode difficulty {difficulty!r}; known: {known}")
        return race_difficulty_raw_value(difficulty)

    def _leave_wrong_difficulty_course_select(self) -> RawMenuStep:
        self._pending_steps.clear()
        self._phase = CareerPhase.BOOT_TO_DIFFICULTY
        self._advance_presses_in_phase = 0
        self._difficulty_popup_state = DifficultyPopupState.CLOSED
        self._machine_selection_applied = False
        return self._queue_tap(
            MenuInput.CANCEL,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.menu_settle_frames,
            phase="course_select:wrong_difficulty:cancel",
        )

    def _continue_post_gp_screen_step(self) -> RawMenuStep:
        self._advance_presses_in_phase = 0
        return self._queue_tap(
            MenuInput.A_BUTTON,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.menu_settle_frames,
            phase="continue_after_race:post_gp_screen",
        )

    def _continue_next_course_step(self) -> RawMenuStep:
        self._advance_presses_in_phase = 0
        step = continue_next_course_step()
        self._pending_steps.append(
            raw_step(
                MenuInput.NEUTRAL,
                MENU_TIMING.result_continue_settle_frames,
                phase=f"{step.phase}:settle",
            )
        )
        return step

    def observe_step(
        self,
        *,
        session: CareerRuntimeSession,
        info: dict[str, object],
    ) -> bool:
        if self._phase == CareerPhase.CONTINUE_AFTER_RACE:
            return False
        terminal_reason = _race_terminal_reason(
            session=session,
            info=info,
        )
        if terminal_reason is None:
            return False
        self._reset_camera_sync()
        terminal_info = _terminal_info(
            session=session,
            info=info,
            terminal_reason=terminal_reason,
        )
        info.update(terminal_info)
        transition = self._progress.handle_terminal_race(
            session=session,
            setup=self._setup,
            info=terminal_info,
        )
        self._remember_finished_attempt(transition)
        if transition.next_plan is not None:
            self._apply_execution_plan(transition.next_plan)
        if self._progress.attempt_id is None:
            self._enter_continue_after_race()
            return True
        self._enter_continue_after_race()
        return True

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

    def _apply_execution_plan(self, plan: SaveRaceExecutionPlan) -> None:
        self._progress.apply_execution_plan(plan)
        self._setup = career_mode_race_setup_config(plan.race_setup)
        self._pending_steps.clear()
        self._phase = CareerPhase.CONTINUE_AFTER_RACE
        self._reset_engine_adjustment()
        self._difficulty_popup_state = DifficultyPopupState.CLOSED
        self._machine_selection_applied = False
        self._advance_presses_in_phase = 0
        self._policy_resolver.update_context(
            setup=self._setup,
            course_setups=self._progress.course_setups,
        )
        self._awaiting_new_race_after_terminal = True
        self._continuing_race_result = True
        self._observed_terminal_race_result = True
        self._continue_after_race_pulses = 0
        self._last_progress_sync_continue_pulse = -1
        self._fresh_race_ready_frames = 0
        self._reset_camera_sync()

    def _enter_continue_after_race(self) -> None:
        self._pending_steps.clear()
        self._phase = CareerPhase.CONTINUE_AFTER_RACE
        self._advance_presses_in_phase = 0
        self._engine_adjust_taps = 0
        self._reset_engine_confirmation()
        self._difficulty_popup_state = DifficultyPopupState.CLOSED
        self._machine_selection_applied = False
        self._awaiting_new_race_after_terminal = True
        self._continuing_race_result = True
        self._observed_terminal_race_result = True
        self._continue_after_race_pulses = 0
        self._last_progress_sync_continue_pulse = -1
        self._fresh_race_ready_frames = 0
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
            or not self._observed_terminal_race_result
        ):
            return False
        facts = MenuFacts.from_info(info)
        if not _post_terminal_progress_screen(facts):
            return False
        if self._last_progress_sync_continue_pulse == self._continue_after_race_pulses:
            return False

        self._last_progress_sync_continue_pulse = self._continue_after_race_pulses
        transition = self._progress.sync_post_terminal_success(
            session=session,
            setup=self._setup,
            info=info,
        )
        self._remember_finished_attempt(transition)
        if transition.next_plan is not None:
            self._apply_execution_plan(transition.next_plan)
        return transition.attempt_finished

    def _reset_camera_sync(self) -> None:
        self._camera.reset()

    def _new_race_ready_for_policy(self, info: dict[str, object]) -> bool:
        facts = MenuFacts.from_info(info)
        if not self._awaiting_new_race_after_terminal:
            return not facts.terminal_race_result

        fresh_race_ready = facts.fresh_race_ready_for_policy or (
            self._observed_terminal_race_result and facts.has_fresh_race_shape
        )
        if not fresh_race_ready:
            self._fresh_race_ready_frames = 0
            return False

        self._fresh_race_ready_frames += 1
        if self._fresh_race_ready_frames < MENU_TIMING.fresh_race_ready_frames:
            return False

        self._awaiting_new_race_after_terminal = False
        self._continuing_race_result = False
        self._observed_terminal_race_result = False
        self._fresh_race_ready_frames = 0
        return True

    def _continue_after_race_pulse(self) -> RawMenuStep:
        self._pending_steps.clear()
        step = continue_after_race_step(self._continue_after_race_pulses)
        self._continue_after_race_pulses += 1
        self._pending_steps.append(
            raw_step(
                MenuInput.NEUTRAL,
                MENU_TIMING.result_continue_settle_frames,
                phase=f"{step.phase}:settle",
            )
        )
        return step


def _is_neutral_settle_step(step: RawMenuStep) -> bool:
    return step.menu_input is MenuInput.NEUTRAL and step.phase.endswith(":settle")


def _pending_step_matches_observed_screen(
    step: RawMenuStep,
    screen: ObservedMenuScreen,
) -> bool:
    if step.menu_input is MenuInput.NEUTRAL:
        return True
    if step.phase.startswith("title_to_main_menu"):
        return screen is ObservedMenuScreen.TITLE
    if step.phase.startswith("main_menu"):
        return screen in {
            ObservedMenuScreen.MAIN_MENU_GP,
            ObservedMenuScreen.MAIN_MENU_OTHER,
        }
    if step.phase.startswith("select_difficulty"):
        return screen is ObservedMenuScreen.DIFFICULTY_POPUP
    if step.phase.startswith("enter_course_select:confirm_difficulty"):
        return screen is ObservedMenuScreen.DIFFICULTY_CONFIRM
    if step.phase.startswith(("select_cup", "enter_machine_select")):
        return screen is ObservedMenuScreen.COURSE_SELECT
    if step.phase.startswith("select_machine"):
        return screen is ObservedMenuScreen.MACHINE_SELECT
    if step.phase.startswith("enter_machine_settings"):
        return screen is ObservedMenuScreen.MACHINE_SELECT
    if step.phase.startswith(("apply_engine", "enter_race")):
        return screen is ObservedMenuScreen.MACHINE_SETTINGS
    if step.phase.startswith("course_select:wrong_difficulty"):
        return screen is ObservedMenuScreen.COURSE_SELECT
    if step.phase.startswith("continue_after_race"):
        return screen in {
            ObservedMenuScreen.GP_RACE,
            ObservedMenuScreen.RESULTS,
            ObservedMenuScreen.GP_NEXT_COURSE,
            ObservedMenuScreen.POST_GP,
        }
    return False


def _cup_selection_input(*, selected_cup_index: int | None, target_cup_index: int) -> MenuInput:
    if selected_cup_index is None or selected_cup_index < target_cup_index:
        return MenuInput.RIGHT
    return MenuInput.LEFT


def _post_terminal_progress_screen(facts: MenuFacts) -> bool:
    return (
        facts.is_gp_result_screen
        or facts.is_gp_next_course_screen
        or facts.is_post_gp_screen
        or (facts.in_gp_race and facts.terminal_race_result)
    )


def _race_terminal_reason(
    *,
    session: CareerRuntimeSession,
    info: dict[str, object],
) -> str | None:
    info_reason = _info_terminal_reason(info=info)
    if info_reason is not None:
        return info_reason

    telemetry = session.emulator.try_read_telemetry()
    if telemetry is None:
        return None
    return _telemetry_terminal_reason(telemetry)


def _info_terminal_reason(
    *,
    info: dict[str, object],
) -> str | None:
    facts = MenuFacts.from_info(info)
    if facts.is_post_gp_screen:
        return "finished"
    reason = _game_terminal_reason(_non_empty_str(info.get("termination_reason")))
    if reason is not None:
        return reason
    for flag_key, flag_reason in (
        ("entered_finished", "finished"),
        ("entered_retired", "retired"),
        ("entered_crashed", "crashed"),
    ):
        if info.get(flag_key) is True:
            return flag_reason
    return None


def _terminal_info(
    *,
    session: CareerRuntimeSession,
    info: dict[str, object],
    terminal_reason: str,
) -> dict[str, object]:
    resolved_info = dict(info)
    resolved_info["termination_reason"] = terminal_reason
    resolved_info["career_mode_race_terminal"] = True

    telemetry = session.emulator.try_read_telemetry()
    if telemetry is None:
        return resolved_info
    resolved_info.setdefault("race_time_ms", telemetry.player.race_time_ms)
    resolved_info.setdefault("position", telemetry.player.position)
    resolved_info.setdefault("track_id", course_id_from_info(resolved_info))
    return resolved_info


def _telemetry_terminal_reason(telemetry: FZeroXTelemetry) -> str | None:
    reason = _game_terminal_reason(telemetry.player.terminal_reason)
    if reason is not None:
        return reason
    return None


def _game_terminal_reason(reason: str | None) -> str | None:
    if reason in {"finished", "retired", "crashed"}:
        return reason
    return None


def _number_info(info: dict[str, object], key: str) -> float | None:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _engine_adjust_tap_count(steps: tuple[RawMenuStep, ...]) -> int:
    return sum(step.menu_input is not MenuInput.NEUTRAL for step in steps)
