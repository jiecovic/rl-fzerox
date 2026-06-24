# src/rl_fzerox/core/career_mode/controller/setup/routing.py
from __future__ import annotations

from typing import TYPE_CHECKING

from rl_fzerox.core.career_mode.controller.setup.menu_flow import (
    cup_selection_input,
    pending_step_matches_observed_screen,
)
from rl_fzerox.core.career_mode.navigation import (
    GP_MENU_ORDER,
    MENU_TIMING,
    CareerPhase,
    DifficultyPopupState,
    MenuFacts,
    MenuInput,
    ObservedMenuScreen,
    RawMenuStep,
    continue_next_course_step,
    course_id_from_info,
    machine_select_steps,
    observed_menu_screen,
    phase_from_step,
    raw_step,
)
from rl_fzerox.core.domain.race import (
    is_race_difficulty_name,
    race_difficulty_raw_value,
)

if TYPE_CHECKING:
    from rl_fzerox.core.career_mode.controller.setup import (
        CareerEngineSetupFlow,
        CareerMenuStepQueue,
    )
    from rl_fzerox.core.career_mode.policy import (
        CareerModePolicyControl,
        CareerPolicyResolver,
    )
    from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig


class CareerMenuRoutingMixin:
    """Menu navigation branch of the Career Mode controller FSM.

    The main FSM owns lifecycle state and side effects. This mixin owns the
    deterministic menu route from title/main menu through difficulty, cup,
    machine, engine setup, and race handoff. It intentionally uses the
    controller's private state so screen transitions remain traceable in one
    state machine without growing the orchestration file back into a monolith.
    """

    if TYPE_CHECKING:
        _setup: CareerModeRaceSetupConfig
        _phase: CareerPhase
        _engine_setup: CareerEngineSetupFlow
        _menu_steps: CareerMenuStepQueue
        _difficulty_popup_state: DifficultyPopupState
        _machine_selection_applied: bool
        _policy_resolver: CareerPolicyResolver

        def _enter_phase(self, phase: CareerPhase) -> None: ...

        def _continue_terminal_race_result_step(self) -> RawMenuStep: ...

        def _resolve_policy_control(
            self,
            info: dict[str, object],
            *,
            refresh_artifact: bool = False,
        ) -> CareerModePolicyControl | None: ...

        @staticmethod
        def _wait_for_policy_resolution() -> RawMenuStep: ...

        def _next_camera_sync_step(self, info: dict[str, object]) -> RawMenuStep | None: ...

        def _enter_policy_race(self) -> RawMenuStep: ...

        def _continue_after_race_pulse(self) -> RawMenuStep: ...

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
                    self._menu_steps.clear()
                    if gp_step := self._main_menu_step_toward_gp(facts):
                        return gp_step
                    return raw_step(
                        MenuInput.NEUTRAL,
                        1,
                        phase="main_menu:wait_for_gp_cursor",
                    )
                case ObservedMenuScreen.DIFFICULTY_POPUP:
                    self._phase = CareerPhase.SELECT_DIFFICULTY
                    self._menu_steps.reset_advance_presses()
                    return self._select_difficulty_popup_step(facts)
                case ObservedMenuScreen.DIFFICULTY_CONFIRM:
                    self._phase = CareerPhase.ENTER_COURSE_SELECT
                    self._menu_steps.reset_advance_presses()
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
                    cup_input = cup_selection_input(
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
                        if not self._menu_steps:
                            route_steps = machine_select_steps(self._setup)
                            self._menu_steps.extend(route_steps)
                            self._menu_steps.reset_advance_presses()
                            if not route_steps:
                                self._machine_selection_applied = True
                        if self._menu_steps:
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
                    if not self._engine_setup.applied_for(course_id):
                        self._enter_phase(CareerPhase.APPLY_ENGINE)
                        return self._apply_engine_step(info)
                    self._enter_phase(CareerPhase.ENTER_RACE)
                    return self._accept_until_phase("enter_race")
                case ObservedMenuScreen.GP_RACE:
                    self._menu_steps.clear()
                    self._menu_steps.reset_advance_presses()
                    if facts.terminal_race_result:
                        return self._continue_terminal_race_result_step()
                    if self._resolve_policy_control(info, refresh_artifact=True) is None:
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

        step = self._menu_steps.peek()
        if step is None:
            return None
        screen = self._observed_menu_screen(facts)
        if not pending_step_matches_observed_screen(step, screen):
            self._menu_steps.clear()
            return None
        step = self._menu_steps.pop()
        if step.phase == "main_menu:open_difficulty:settle":
            self._difficulty_popup_state = DifficultyPopupState.OPEN
        if step.phase.startswith("select_difficulty:accept"):
            self._difficulty_popup_state = DifficultyPopupState.SUBMITTED
        self._phase = phase_from_step(step)
        if step.phase.startswith("select_machine") and not self._menu_steps:
            self._machine_selection_applied = True
        return step

    def _observed_menu_screen(self, facts: MenuFacts) -> ObservedMenuScreen:
        return observed_menu_screen(
            facts,
            difficulty_popup_state=self._difficulty_popup_state,
        )

    def _start_until_phase(self, phase: str) -> RawMenuStep:
        return self._menu_steps.start_until_phase(phase)

    def _accept_until_phase(self, phase: str) -> RawMenuStep:
        return self._menu_steps.accept_until_phase(phase)

    def _queue_tap(
        self,
        menu_input: MenuInput,
        *,
        hold_frames: int,
        settle_frames: int,
        phase: str,
    ) -> RawMenuStep:
        return self._menu_steps.queue_tap(
            menu_input,
            hold_frames=hold_frames,
            settle_frames=settle_frames,
            phase=phase,
        )

    def _queue_menu_steps(self, steps: tuple[RawMenuStep, ...]) -> RawMenuStep:
        return self._menu_steps.queue_menu_steps(steps)

    def _open_difficulty_popup(self) -> RawMenuStep:
        self._menu_steps.clear()
        self._difficulty_popup_state = DifficultyPopupState.OPENING
        self._menu_steps.reset_advance_presses()
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
            self._menu_steps.clear()
            self._difficulty_popup_state = DifficultyPopupState.CLOSED
            if gp_step := self._main_menu_step_toward_gp(facts):
                return gp_step
            return raw_step(
                MenuInput.NEUTRAL,
                1,
                phase="difficulty_popup:lost_gp_cursor",
            )
        if not facts.has_difficulty_popup:
            self._menu_steps.clear()
            return raw_step(
                MenuInput.NEUTRAL,
                1,
                phase="difficulty_popup:wait_visible",
            )
        target = self._setup_difficulty_raw_value()
        current = facts.difficulty_cursor_raw
        if current is None:
            self._menu_steps.clear()
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
            self._menu_steps.clear()
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
        return self._engine_setup.next_step(
            info=info,
            course_setup=self._policy_resolver.resolve_course_setup(info),
            queue_menu_steps=self._queue_menu_steps,
        )

    def _course_select_matches_target(self, facts: MenuFacts) -> bool:
        return facts.difficulty_raw == self._setup_difficulty_raw_value()

    def _setup_difficulty_raw_value(self) -> int:
        difficulty = self._setup.difficulty
        if not is_race_difficulty_name(difficulty):
            known = ", ".join(GP_MENU_ORDER.difficulties)
            raise RuntimeError(f"unknown Career Mode difficulty {difficulty!r}; known: {known}")
        return race_difficulty_raw_value(difficulty)

    def _leave_wrong_difficulty_course_select(self) -> RawMenuStep:
        self._menu_steps.clear()
        self._phase = CareerPhase.BOOT_TO_DIFFICULTY
        self._menu_steps.reset_advance_presses()
        self._difficulty_popup_state = DifficultyPopupState.CLOSED
        self._machine_selection_applied = False
        return self._queue_tap(
            MenuInput.CANCEL,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.menu_settle_frames,
            phase="course_select:wrong_difficulty:cancel",
        )

    def _continue_post_gp_screen_step(self) -> RawMenuStep:
        self._menu_steps.reset_advance_presses()
        return self._queue_tap(
            MenuInput.A_BUTTON,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.menu_settle_frames,
            phase="continue_after_race:post_gp_screen",
        )

    def _continue_next_course_step(self) -> RawMenuStep:
        self._menu_steps.reset_advance_presses()
        step = continue_next_course_step()
        self._menu_steps.append(
            raw_step(
                MenuInput.NEUTRAL,
                MENU_TIMING.result_continue_settle_frames,
                phase=f"{step.phase}:settle",
            )
        )
        return step
