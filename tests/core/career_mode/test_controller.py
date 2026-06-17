# tests/core/career_mode/test_controller.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fzerox_emulator import ControllerState, FZeroXTelemetry
from rl_fzerox.core.career_mode.attempts import CareerProgressTransition
from rl_fzerox.core.career_mode.controller import (
    CareerModeController,
    _cup_selection_input,
)
from rl_fzerox.core.career_mode.controller.recording import (
    CareerRecordingSegmentTracker,
)
from rl_fzerox.core.career_mode.controller.terminal import post_terminal_progress_screen
from rl_fzerox.core.career_mode.execution.race import SaveRaceExecutionPlan, SaveRaceSetup
from rl_fzerox.core.career_mode.navigation import (
    MENU_TIMING,
    CareerPhase,
    DifficultyPopupState,
    MenuFacts,
    MenuInput,
    ObservedMenuScreen,
    engine_adjust_steps,
    observed_menu_screen,
)
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig


def test_cup_selection_moves_left_when_target_is_before_selected_cup() -> None:
    assert _cup_selection_input(selected_cup_index=3, target_cup_index=0) is MenuInput.LEFT


def test_cup_selection_moves_right_when_target_is_after_selected_cup() -> None:
    assert _cup_selection_input(selected_cup_index=0, target_cup_index=3) is MenuInput.RIGHT


def test_cup_selection_moves_right_until_selected_cup_is_known() -> None:
    assert _cup_selection_input(selected_cup_index=None, target_cup_index=0) is MenuInput.RIGHT


def test_menu_facts_accept_game_mode_name_fallback() -> None:
    facts = MenuFacts.from_info({"game_mode_name": "unskippable_credits"})

    assert facts.is_post_gp_screen
    assert post_terminal_progress_screen(facts)


def test_engine_adjust_steps_use_fast_bounded_right_burst() -> None:
    steps = engine_adjust_steps(current=0, target=100)

    active_steps = [step for step in steps if step.menu_input is not MenuInput.NEUTRAL]
    settle_steps = [step for step in steps if step.menu_input is MenuInput.NEUTRAL]

    assert len(active_steps) == MENU_TIMING.engine_adjust_max_taps_per_burst
    assert len(settle_steps) == MENU_TIMING.engine_adjust_max_taps_per_burst
    assert {step.menu_input for step in active_steps} == {MenuInput.RIGHT}
    assert {step.frames for step in active_steps} == {MENU_TIMING.engine_adjust_hold_frames}
    assert {step.frames for step in settle_steps} == {MENU_TIMING.engine_adjust_settle_frames}


def test_engine_adjust_steps_cap_burst_by_remaining_safety_budget() -> None:
    steps = engine_adjust_steps(current=100, target=0, max_taps=3)

    active_steps = [step for step in steps if step.menu_input is not MenuInput.NEUTRAL]

    assert len(active_steps) == 3
    assert {step.menu_input for step in active_steps} == {MenuInput.LEFT}


def test_engine_adjust_steps_empty_when_target_already_selected() -> None:
    assert engine_adjust_steps(current=50, target=50) == ()


def test_policy_race_continues_terminal_gp_result_screen(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller._phase = CareerPhase.POLICY_RACE

    step = controller.next_raw_step(
        info={"game_mode": "gp_race", "termination_reason": "finished"},
    )

    assert step is not None
    assert step.menu_input is MenuInput.ACCEPT
    assert step.phase == "continue_after_race:accept:1"
    assert controller.phase is CareerPhase.CONTINUE_AFTER_RACE

    settle_step = controller.next_raw_step(
        info={"game_mode": "gp_race", "termination_reason": "finished"},
    )

    assert settle_step is not None
    assert settle_step.menu_input is MenuInput.NEUTRAL
    assert settle_step.phase == "continue_after_race:accept:1:settle"


def test_unskippable_credits_are_post_gp_screen() -> None:
    screen = observed_menu_screen(
        MenuFacts.from_info({"game_mode": "unskippable_credits"}),
        difficulty_popup_state=DifficultyPopupState.CLOSED,
    )

    assert screen is ObservedMenuScreen.POST_GP


def test_returned_main_menu_is_post_terminal_progress_screen() -> None:
    assert post_terminal_progress_screen(MenuFacts.from_info({"game_mode": "main_menu"}))


def test_stale_gp_race_terminal_is_not_post_terminal_progress_screen() -> None:
    assert not post_terminal_progress_screen(
        MenuFacts.from_info({"game_mode": "gp_race", "termination_reason": "finished"})
    )


def test_no_active_attempt_emits_no_menu_navigation(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller._progress._attempt_id = None
    controller._phase = CareerPhase.WAIT_FOR_GP_RACE

    step = controller.next_raw_step(
        info={"game_mode": "gp_race", "termination_reason": "finished"},
    )

    assert step is None


def test_checkpoint_refresh_is_armed_only_after_finished_attempt(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    resolver = _PolicyResolverStub()
    controller.__dict__["_policy_resolver"] = resolver

    assert (
        controller._resolve_policy_control({"course_index": 0}, refresh_artifact=True) == "control"
    )
    assert resolver.refresh_requests == [False]

    controller._remember_finished_attempt(
        SimpleNamespace(
            finished_attempt_id="attempt-a",
            finished_status="failed",
            finished_failure_reason=None,
        )
    )

    assert (
        controller._resolve_policy_control({"course_index": 0}, refresh_artifact=True) == "control"
    )
    assert resolver.refresh_requests == [False, True]

    assert (
        controller._resolve_policy_control({"course_index": 0}, refresh_artifact=True) == "control"
    )
    assert resolver.refresh_requests == [False, True, False]


def test_recording_segment_tracker_waits_for_explicit_close() -> None:
    tracker = CareerRecordingSegmentTracker()

    tracker.observe_terminal_result({"termination_reason": "finished"})
    tracker.observe_progress_screen(
        MenuFacts.from_info({"game_mode": "results"}),
        {"game_mode": "results"},
    )
    assert tracker.pop_close() is None

    tracker.observe_progress_screen(
        MenuFacts.from_info({"game_mode": "gp_race_next_course"}),
        {"game_mode": "gp_race_next_course"},
    )
    assert tracker.pop_close() is None

    tracker.observe_progress_screen(
        MenuFacts.from_info({"game_mode": "gp_end_cutscene"}),
        {"game_mode": "gp_end_cutscene"},
    )
    assert tracker.pop_close() is None

    tracker.observe_progress_screen(
        MenuFacts.from_info({"game_mode": "unskippable_credits"}),
        {"game_mode": "unskippable_credits"},
    )
    assert tracker.pop_close() is None

    tracker.observe_progress_screen(
        MenuFacts.from_info({"game_mode": "main_menu"}),
        {"game_mode": "main_menu"},
    )
    assert tracker.pop_close() is None

    tracker.close(status="succeeded")
    close = tracker.pop_close()

    assert close is not None
    assert close.status == "succeeded"
    assert tracker.pop_close() is None


def test_recording_segment_close_is_not_downgraded_by_credit_reset() -> None:
    tracker = CareerRecordingSegmentTracker()

    tracker.observe_terminal_result({"termination_reason": "finished"})
    tracker.close(status="succeeded")
    tracker.force_close(status="failed")
    close = tracker.pop_close()

    assert close is not None
    assert close.status == "succeeded"


def test_recording_segment_close_survives_next_plan_application(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.__dict__["_progress"] = _PostGpProgressStub(next_plan=True)
    controller._phase = CareerPhase.CONTINUE_AFTER_RACE
    controller._post_race.observed_terminal_race_result = True
    controller._recording.observe_terminal_result({"termination_reason": "finished"})

    handled = controller.before_step(
        session=_ControllerSession(),
        info={"game_mode": "unskippable_credits", "termination_reason": "finished"},
    )
    events = controller.drain_lifecycle_events()

    assert handled is True
    assert events.recording_close is not None
    assert events.recording_close.status == "succeeded"


def test_controller_does_not_sync_stale_gp_race_terminal_result(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    progress = _PostGpProgressStub()
    controller.__dict__["_progress"] = progress
    controller._phase = CareerPhase.CONTINUE_AFTER_RACE
    controller._post_race.observed_terminal_race_result = True

    handled = controller.before_step(
        session=_ControllerSession(),
        info={"game_mode": "gp_race", "termination_reason": "finished"},
    )

    assert handled is False
    assert progress.sync_calls == []
    assert controller.drain_lifecycle_events().recording_close is None


def test_controller_does_not_reset_or_close_recording_at_winning_ceremony_start(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.__dict__["_progress"] = _PostGpProgressStub()
    controller._phase = CareerPhase.CONTINUE_AFTER_RACE
    controller._post_race.observed_terminal_race_result = True
    controller._post_race.continue_pulses = 1

    handled = controller.before_step(
        session=_ControllerSession(),
        info={
            "game_mode": "gp_end_cutscene",
            "termination_reason": "finished",
            "frame_index": 100,
        },
    )

    assert handled is False
    events = controller.drain_lifecycle_events()
    assert events.recording_close is None
    assert events.emulator_reset_requested is False


def test_controller_closes_recording_after_winning_ceremony_grace_window(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.__dict__["_progress"] = _PostGpProgressStub()
    controller._phase = CareerPhase.CONTINUE_AFTER_RACE
    controller._post_race.observed_terminal_race_result = True
    controller._post_race.continue_pulses = 1

    controller.before_step(
        session=_ControllerSession(),
        info={
            "game_mode": "gp_end_cutscene",
            "termination_reason": "finished",
            "frame_index": 100,
            "gp_final_rank": 1,
        },
    )
    handled = controller.before_step(
        session=_ControllerSession(),
        info={
            "game_mode": "gp_end_cutscene",
            "termination_reason": "finished",
            "frame_index": 100 + MENU_TIMING.post_gp_cutscene_record_frames,
            "gp_final_rank": 1,
        },
    )
    events = controller.drain_lifecycle_events()

    assert handled is True
    assert events.recording_close is not None
    assert events.recording_close.status == "succeeded"


def test_controller_closes_recording_on_credits(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.__dict__["_progress"] = _PostGpProgressStub()
    controller._phase = CareerPhase.CONTINUE_AFTER_RACE
    controller._post_race.observed_terminal_race_result = True
    controller._post_race.continue_pulses = 2

    handled = controller.before_step(
        session=_ControllerSession(),
        info={"game_mode": "unskippable_credits", "termination_reason": "finished"},
    )
    events = controller.drain_lifecycle_events()

    assert handled is True
    assert events.recording_close is not None
    assert events.recording_close.status == "succeeded"
    assert events.emulator_reset_requested is True


def test_recording_segment_close_marks_game_over_failed() -> None:
    tracker = CareerRecordingSegmentTracker()

    tracker.observe_terminal_result({"termination_reason": "retired"})
    assert tracker.pop_close() is None

    tracker.force_close(status="failed")
    close = tracker.pop_close()

    assert close is not None
    assert close.status == "failed"
    assert tracker.pop_close() is None


class _PolicyResolverStub:
    def __init__(self) -> None:
        self.refresh_requests: list[bool] = []

    def resolve(
        self,
        info: dict[str, object],
        *,
        refresh_artifact: bool = False,
    ) -> object:
        del info
        self.refresh_requests.append(refresh_artifact)
        return SimpleNamespace(
            activated_new_policy=False,
            camera_setting=None,
            control="control",
        )


class _PostGpProgressStub:
    attempt_id = "attempt-1"

    def __init__(self, *, next_plan: bool = False) -> None:
        self._next_plan = next_plan
        self.course_setups = ()
        self.sync_calls: list[dict[str, object]] = []

    def apply_execution_plan(self, plan: object) -> None:
        self.attempt_id = getattr(plan, "attempt_id", self.attempt_id)

    def sync_post_terminal_progress(
        self,
        *,
        session: object,
        setup: object,
        info: dict[str, object],
    ) -> CareerProgressTransition:
        del session, setup
        self.sync_calls.append(dict(info))
        if info.get("game_mode") == "unskippable_credits" or (
            info.get("career_mode_post_gp_cutscene_complete") is True
            and info.get("gp_final_rank") == 1
        ):
            return CareerProgressTransition(
                attempt_finished=True,
                finished_attempt_id="attempt-1",
                finished_status="succeeded",
                next_plan=_execution_plan() if self._next_plan else None,
                reset_emulator=True,
            )
        return CareerProgressTransition(attempt_finished=False)


class _ControllerEmulator:
    def read_save_ram(self) -> bytes:
        return b"save-ram"

    def write_save_ram(self, data: bytes) -> None:
        del data

    def set_controller_state(self, controller_state: ControllerState) -> None:
        del controller_state

    def step_frames(self, count: int, *, capture_video: bool = True) -> None:
        del count, capture_video

    def try_read_telemetry(self) -> FZeroXTelemetry | None:
        return None


class _ControllerSession:
    emulator = _ControllerEmulator()


def _controller(tmp_path: Path) -> CareerModeController:
    db_path = tmp_path / "manager" / "runs.db"
    store = ManagerStore(db_path)
    save_game = store.create_save_game(
        name="Career Save",
        save_games_root=tmp_path / "save-games",
    )
    attempt = store.start_save_attempt(
        save_game_id=save_game.id,
        target_kind="clear_gp_cup",
        difficulty="novice",
        cup_id="jack",
    )
    return CareerModeController(
        _race_setup(),
        db_path=db_path,
        save_game_id=save_game.id,
        attempt_id=attempt.id,
        device="cpu",
    )


def _race_setup() -> CareerModeRaceSetupConfig:
    return CareerModeRaceSetupConfig(
        difficulty="novice",
        cup_id="jack",
        course_id=None,
        vehicle_id="blue_falcon",
        vehicle_display_name="Blue Falcon",
        character_index=0,
        machine_select_slot=0,
        machine_select_row=0,
        machine_select_column=0,
        engine_setting_raw_value=50,
    )


def _execution_plan() -> SaveRaceExecutionPlan:
    race_setup = _race_setup()
    return SaveRaceExecutionPlan(
        attempt_id="attempt-2",
        policy_run_id="run-1",
        policy_run_dir=Path("runs/run-1"),
        policy_artifact="best",
        policy_algorithm="ppo",
        policy_path=Path("runs/run-1/checkpoints/best/policy.zip"),
        race_setup=SaveRaceSetup(
            difficulty=race_setup.difficulty,
            cup_id=race_setup.cup_id,
            course_id=race_setup.course_id,
            vehicle_id=race_setup.vehicle_id,
            vehicle_display_name=race_setup.vehicle_display_name,
            character_index=race_setup.character_index,
            machine_select_slot=race_setup.machine_select_slot,
            machine_select_row=race_setup.machine_select_row,
            machine_select_column=race_setup.machine_select_column,
            engine_setting_raw_value=race_setup.engine_setting_raw_value,
        ),
    )
