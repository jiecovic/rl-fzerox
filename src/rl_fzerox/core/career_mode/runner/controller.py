# src/rl_fzerox/core/career_mode/runner/controller.py
from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.career_mode.course_setup import (
    CourseSetupTarget,
    resolve_course_setup,
)
from rl_fzerox.core.career_mode.runner.menu import (
    CAREER_MENU_DEFAULTS,
    GP_MENU_ORDER,
    MENU_TIMING,
    CareerPhase,
    MenuFacts,
    MenuInput,
    RawMenuStep,
    camera_setting,
    continue_after_race_step,
    continue_next_course_step,
    course_id_from_info,
    in_gp_race,
    machine_select_steps,
    phase_from_step,
    raw_step,
    select_difficulty_steps,
)
from rl_fzerox.core.career_mode.runner.policy import CareerModePolicyControl
from rl_fzerox.core.career_mode.runner.race import (
    SaveRaceExecutionPlan,
    build_save_race_execution_plan,
)
from rl_fzerox.core.career_mode.runner.save_file import persist_save_ram_for_store
from rl_fzerox.core.career_mode.runner.setup import (
    career_mode_race_setup_config,
    save_race_setup_from_config,
)
from rl_fzerox.core.domain.camera import CameraSettingName
from rl_fzerox.core.envs.engine.info import telemetry_info
from rl_fzerox.core.envs.engine.reset.camera import (
    CAMERA_SYNC_CONTROLS,
    sync_camera_setting,
)
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.models import (
    ManagedRun,
    ManagedSaveUnlockProgress,
    SaveAttemptStatus,
)
from rl_fzerox.core.manager.projection.assembly import effective_train_algorithm
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig, WatchAppConfig
from rl_fzerox.core.training.inference import PolicyRunner, load_policy_runner

if TYPE_CHECKING:
    from fzerox_emulator import Emulator


class CareerRuntimeSession(Protocol):
    """Narrow runtime surface Career Mode needs from its session loop."""

    emulator: Emulator


@dataclass(frozen=True, slots=True)
class _LoadedPolicy:
    run: ManagedRun
    runner: PolicyRunner


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
    ) -> None:
        self._setup = setup
        self._store = ManagerStore(db_path)
        self._save_game_id = save_game_id
        self._attempt_id: str | None = attempt_id
        self._device = device
        self._camera_setting: CameraSettingName | None = None
        self._pending_steps: deque[RawMenuStep] = deque()
        self._phase = CareerPhase.BOOT_TO_DIFFICULTY
        self._engine_applied = False
        self._start_presses_in_phase = 0
        self._camera_synced = self._camera_setting is None
        self._camera_sync_taps = 0
        self._awaiting_new_race_after_terminal = False
        self._continuing_race_result = False
        self._observed_terminal_race_result = False
        self._continue_after_race_pulses = 0
        self._fresh_race_ready_frames = 0
        self._course_setups = self._store.list_save_course_setups(save_game_id)
        self._unlock_progress = self._store.save_game_unlock_progress(save_game_id)
        self._policy_cache: dict[tuple[str, str], _LoadedPolicy] = {}
        self._active_policy_key: tuple[str, str] | None = None

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
        facts = MenuFacts.from_info(info)
        if (
            self._phase == CareerPhase.APPLY_ENGINE
            and not self._engine_applied
            and facts.is_machine_settings
        ):
            if (
                self._setup.engine_setting_raw_value
                != CAREER_MENU_DEFAULTS.engine_setting_raw_value
            ):
                raise RuntimeError(
                    "Career Mode cannot yet select non-default engine settings "
                    "through the game menu"
                )
            self._engine_applied = True
        return self._sync_camera_before_policy_handoff(session=session, info=info)

    def next_raw_step(self, *, info: dict[str, object]) -> RawMenuStep | None:
        facts = MenuFacts.from_info(info)
        if self._phase == CareerPhase.POLICY_RACE:
            if facts.in_gp_race:
                return None
            raise RuntimeError("Career Mode left a race before observing a game result")

        if self._phase == CareerPhase.CONTINUE_AFTER_RACE:
            if facts.in_gp_race:
                if self._pending_steps and _is_neutral_settle_step(self._pending_steps[0]):
                    step = self._pending_steps.popleft()
                    self._phase = phase_from_step(step)
                    return step
                if self._continuing_race_result and facts.terminal_race_result:
                    self._observed_terminal_race_result = True
                    if self._new_race_ready_for_policy(info):
                        self._continuing_race_result = False
                    else:
                        return self._continue_after_race_pulse()
                if facts.terminal_race_result:
                    self._observed_terminal_race_result = True
                    self._continuing_race_result = True
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
            if facts.is_skippable_post_gp_screen:
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
            self._start_presses_in_phase = 0
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
            step = self._pending_steps.popleft()
            self._phase = phase_from_step(step)
            return step
        return self._next_menu_step(info)

    def _next_menu_step(self, info: dict[str, object]) -> RawMenuStep | None:
        facts = MenuFacts.from_info(info)
        for _ in range(8):
            match self._phase:
                case CareerPhase.BOOT_TO_DIFFICULTY:
                    if facts.is_course_select:
                        self._phase = CareerPhase.SELECT_CUP
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_machine_select:
                        self._phase = CareerPhase.SELECT_MACHINE
                        self._pending_steps.extend(machine_select_steps(self._setup))
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_machine_settings:
                        self._phase = CareerPhase.APPLY_ENGINE
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_mode_select:
                        if not facts.selected_gp_mode:
                            return self._select_gp_mode_step()
                        self._phase = CareerPhase.SELECT_DIFFICULTY
                        self._pending_steps.extend(select_difficulty_steps(self._setup))
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_title:
                        return self._start_until_phase("boot_to_difficulty")
                    return self._menu_start_until_phase("boot_to_difficulty")
                case CareerPhase.SELECT_DIFFICULTY:
                    if facts.is_course_select:
                        self._phase = CareerPhase.SELECT_CUP
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_mode_select and not facts.selected_gp_mode:
                        self._pending_steps.clear()
                        self._phase = CareerPhase.BOOT_TO_DIFFICULTY
                        return self._select_gp_mode_step()
                    if facts.is_machine_select:
                        self._phase = CareerPhase.SELECT_MACHINE
                        self._pending_steps.clear()
                        self._pending_steps.extend(machine_select_steps(self._setup))
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_machine_settings:
                        self._pending_steps.clear()
                        self._phase = CareerPhase.APPLY_ENGINE
                        self._start_presses_in_phase = 0
                        continue
                    if self._pending_steps:
                        return self._pending_steps.popleft()
                    self._phase = CareerPhase.ENTER_COURSE_SELECT
                    self._start_presses_in_phase = 0
                    continue
                case CareerPhase.ENTER_COURSE_SELECT:
                    if facts.is_course_select:
                        self._phase = CareerPhase.SELECT_CUP
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_mode_select and facts.is_game_mode_transition:
                        return raw_step(
                            MenuInput.NEUTRAL,
                            1,
                            phase="enter_course_select:wait_for_transition",
                        )
                    if facts.is_mode_select and facts.selected_gp_mode:
                        return self._menu_start_until_phase("enter_course_select")
                    if facts.is_machine_select:
                        self._phase = CareerPhase.SELECT_MACHINE
                        self._pending_steps.extend(machine_select_steps(self._setup))
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_machine_settings:
                        self._phase = CareerPhase.APPLY_ENGINE
                        self._start_presses_in_phase = 0
                        continue
                    return self._menu_start_until_phase("enter_course_select")
                case CareerPhase.SELECT_CUP:
                    if facts.is_machine_select:
                        self._phase = CareerPhase.SELECT_MACHINE
                        self._pending_steps.extend(machine_select_steps(self._setup))
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_machine_settings:
                        self._phase = CareerPhase.APPLY_ENGINE
                        self._start_presses_in_phase = 0
                        continue
                    selected_cup_index = facts.course_select_cup_index
                    target_cup_index = GP_MENU_ORDER.cup_right_count(self._setup.cup_id)
                    if selected_cup_index == target_cup_index:
                        self._phase = CareerPhase.ENTER_MACHINE_SELECT
                        self._start_presses_in_phase = 0
                        continue
                    return self._queue_tap(
                        MenuInput.RIGHT,
                        hold_frames=MENU_TIMING.menu_hold_frames,
                        settle_frames=MENU_TIMING.menu_settle_frames,
                        phase="select_cup:right",
                    )
                case CareerPhase.ENTER_MACHINE_SELECT:
                    if facts.is_machine_select:
                        self._phase = CareerPhase.SELECT_MACHINE
                        self._pending_steps.extend(machine_select_steps(self._setup))
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_machine_settings:
                        self._phase = CareerPhase.APPLY_ENGINE
                        self._start_presses_in_phase = 0
                        continue
                    return self._menu_start_until_phase("enter_machine_select")
                case CareerPhase.SELECT_MACHINE:
                    if facts.is_machine_settings:
                        self._pending_steps.clear()
                        self._phase = CareerPhase.APPLY_ENGINE
                        self._start_presses_in_phase = 0
                        continue
                    if self._pending_steps:
                        return self._pending_steps.popleft()
                    self._phase = CareerPhase.ENTER_MACHINE_SETTINGS
                    self._start_presses_in_phase = 0
                    continue
                case CareerPhase.ENTER_MACHINE_SETTINGS:
                    if facts.is_machine_settings:
                        self._phase = CareerPhase.APPLY_ENGINE
                        self._start_presses_in_phase = 0
                        continue
                    return self._menu_start_until_phase("enter_machine_settings")
                case CareerPhase.APPLY_ENGINE:
                    if not self._engine_applied:
                        return raw_step(
                            MenuInput.NEUTRAL,
                            1,
                            phase="apply_engine:wait",
                        )
                    self._phase = CareerPhase.ENTER_RACE
                    self._start_presses_in_phase = 0
                    continue
                case CareerPhase.CONTINUE_AFTER_RACE:
                    if facts.is_title:
                        self._phase = CareerPhase.BOOT_TO_DIFFICULTY
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_mode_select:
                        if not facts.selected_gp_mode:
                            self._phase = CareerPhase.BOOT_TO_DIFFICULTY
                            self._pending_steps.clear()
                            return self._select_gp_mode_step()
                        self._phase = CareerPhase.SELECT_DIFFICULTY
                        self._pending_steps.clear()
                        self._pending_steps.extend(select_difficulty_steps(self._setup))
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_course_select:
                        self._phase = CareerPhase.SELECT_CUP
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_machine_select:
                        self._phase = CareerPhase.SELECT_MACHINE
                        self._pending_steps.extend(machine_select_steps(self._setup))
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_machine_settings:
                        self._phase = CareerPhase.APPLY_ENGINE
                        self._start_presses_in_phase = 0
                        continue
                    if facts.is_skippable_post_gp_screen:
                        return self._continue_post_gp_screen_step()
                    if facts.is_gp_next_course_screen:
                        return self._continue_next_course_step()
                    return raw_step(
                        MenuInput.NEUTRAL,
                        1,
                        phase="continue_after_race:wait_for_known_screen",
                    )
                case CareerPhase.ENTER_RACE:
                    return self._menu_start_until_phase(self._phase.value)
                case CareerPhase.WAIT_FOR_GP_RACE:
                    return raw_step(MenuInput.NEUTRAL, 1, phase=self._phase.value)
                case CareerPhase.POLICY_RACE:
                    return None
        return raw_step(MenuInput.NEUTRAL, 1, phase=self._phase.value)

    def policy_owns_control(self) -> bool:
        return self._phase == CareerPhase.POLICY_RACE

    @property
    def phase(self) -> CareerPhase:
        return self._phase

    def viewer_info(
        self,
        *,
        info: dict[str, object],
        active_policy_control: CareerModePolicyControl | None,
    ) -> dict[str, object]:
        """Add Career Mode control/progress context to viewer telemetry."""

        progress = self._unlock_progress
        current_target_label = _current_target_label(progress, self._setup)
        next_target = progress.next_target
        viewer_info = dict(info)
        viewer_info.update(
            {
                "career_mode_attempt_id": self._attempt_id,
                "career_mode_completed_targets": progress.completed_count,
                "career_mode_controller_context": self.debug_context(info),
                "career_mode_total_targets": progress.total_count,
                "career_mode_inspection_status": progress.inspection_status,
                "career_mode_next_target_label": (
                    next_target.label if next_target is not None else None
                ),
                "career_mode_phase": self._phase.value,
                "career_mode_policy_active": active_policy_control is not None,
                "career_mode_target_label": current_target_label,
            }
        )
        viewer_info.update(self._viewer_fsm_facts(info))
        if active_policy_control is not None:
            viewer_info.update(
                {
                    "career_mode_policy_artifact": (
                        active_policy_control.course_setup.policy_artifact
                    ),
                    "career_mode_policy_run_id": active_policy_control.policy_run.id,
                    "career_mode_policy_run_name": active_policy_control.policy_run.name,
                    "career_mode_policy_scope": active_policy_control.course_setup.scope,
                    "career_mode_policy_course_id": (
                        active_policy_control.course_setup.course_id or course_id_from_info(info)
                    ),
                }
            )
        return viewer_info

    def _viewer_fsm_facts(self, info: dict[str, object]) -> dict[str, object]:
        facts = MenuFacts.from_info(info)
        return {
            "career_mode_fsm_awaiting_fresh_race": (self._awaiting_new_race_after_terminal),
            "career_mode_fsm_camera_synced": self._camera_synced,
            "career_mode_fsm_camera_target": self._camera_setting,
            "career_mode_fsm_completed_laps": facts.completed_laps,
            "career_mode_fsm_completion_fraction": facts.completion_fraction,
            "career_mode_fsm_continuing_result": self._continuing_race_result,
            "career_mode_fsm_course_index": facts.course_index,
            "career_mode_fsm_fresh_race_ready": facts.fresh_race_ready_for_policy,
            "career_mode_fsm_fresh_race_ready_frames": (self._fresh_race_ready_frames),
            "career_mode_fsm_game_mode": facts.game_mode,
            "career_mode_fsm_intro_timer": facts.race_intro_timer,
            "career_mode_fsm_pending_steps": len(self._pending_steps),
            "career_mode_fsm_race_time_ms": facts.race_time_ms,
            "career_mode_fsm_selected_mode_raw": facts.selected_mode_raw,
            "career_mode_fsm_terminal_reason": facts.terminal_reason,
            "career_mode_fsm_terminal_result": facts.terminal_race_result,
            "career_mode_fsm_total_laps": facts.total_laps,
            "career_mode_fsm_transition_raw": facts.transition_state_raw,
        }

    def debug_context(self, info: dict[str, object]) -> str:
        fields = (
            ("phase", self._phase.value),
            ("pending_steps", len(self._pending_steps)),
            ("game_mode", info.get("game_mode")),
            ("game_mode_raw", info.get("game_mode_raw")),
            ("queued_game_mode", info.get("queued_game_mode")),
            ("queued_game_mode_raw", info.get("queued_game_mode_raw")),
            ("menu_selected_mode_raw", info.get("menu_selected_mode_raw")),
            ("menu_transition_state_raw", info.get("menu_transition_state_raw")),
            ("difficulty", info.get("difficulty")),
            ("difficulty_raw", info.get("difficulty_raw")),
            ("course_index", info.get("course_index")),
            ("camera_setting", camera_setting(info)),
            ("race_intro_timer", info.get("race_intro_timer")),
            ("race_time_ms", info.get("race_time_ms")),
            ("laps", _lap_summary(info)),
            ("position", info.get("position")),
            ("termination_reason", info.get("termination_reason")),
        )
        parts = [f"{name}={value!r}" for name, value in fields if value is not None]
        return ", ".join(parts)

    def _resolve_policy_control(
        self,
        info: dict[str, object],
    ) -> CareerModePolicyControl | None:
        course_id = course_id_from_info(info)
        if course_id is None:
            return None
        course_setup = resolve_course_setup(
            self._course_setups,
            CourseSetupTarget(
                difficulty=self._setup.difficulty,
                cup_id=self._setup.cup_id,
                course_id=course_id,
            ),
        )
        if course_setup is None:
            raise RuntimeError(f"no Career Mode course setup matches {course_id!r}")

        key = (course_setup.policy_run_id, course_setup.policy_artifact)
        loaded_policy = self._policy_cache.get(key)
        if loaded_policy is None or key != self._active_policy_key:
            policy_run = self._store.get_run(course_setup.policy_run_id)
            if policy_run is None:
                raise RuntimeError(
                    f"Career Mode policy run not found: {course_setup.policy_run_id}"
                )
        elif loaded_policy is not None:
            policy_run = loaded_policy.run

        if loaded_policy is None:
            runner = load_policy_runner(
                policy_run.run_dir,
                artifact=course_setup.policy_artifact,
                device=self._device,
                algorithm=effective_train_algorithm(policy_run.config),
            )
            loaded_policy = _LoadedPolicy(run=policy_run, runner=runner)
            self._policy_cache[key] = loaded_policy
        if key != self._active_policy_key:
            self._active_policy_key = key
            self._camera_setting = _validated_camera_setting(
                policy_run.config.environment.camera_setting
            )
            self._reset_camera_sync()
        return CareerModePolicyControl(
            course_setup=course_setup,
            policy_run=loaded_policy.run,
            runner=loaded_policy.runner,
        )

    def _enter_policy_race(self) -> RawMenuStep:
        self._phase = CareerPhase.POLICY_RACE
        return raw_step(MenuInput.NEUTRAL, 1, phase="policy_race:handoff")

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

    def _menu_start_until_phase(self, phase: str) -> RawMenuStep:
        return self._start_until_phase(phase)

    def _tap_until_phase(
        self,
        menu_input: MenuInput,
        *,
        hold_frames: int,
        settle_frames: int,
        phase: str,
        label: str,
    ) -> RawMenuStep:
        self._start_presses_in_phase += 1
        if self._start_presses_in_phase > MENU_TIMING.max_start_presses_per_phase:
            raise RuntimeError(
                f"Career Mode menu phase {phase!r} did not reach the expected screen"
            )
        return self._queue_tap(
            menu_input,
            hold_frames=hold_frames,
            settle_frames=settle_frames,
            phase=f"{phase}:{label}:{self._start_presses_in_phase}",
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

    def _select_gp_mode_step(self) -> RawMenuStep:
        return self._queue_tap(
            MenuInput.LEFT,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.menu_settle_frames,
            phase="select_mode:left_to_gp",
        )

    def _continue_post_gp_screen_step(self) -> RawMenuStep:
        self._start_presses_in_phase = 0
        return self._queue_tap(
            MenuInput.A_BUTTON,
            hold_frames=MENU_TIMING.menu_hold_frames,
            settle_frames=MENU_TIMING.menu_settle_frames,
            phase="continue_after_race:post_gp_screen",
        )

    def _continue_next_course_step(self) -> RawMenuStep:
        self._start_presses_in_phase = 0
        step = continue_next_course_step()
        self._pending_steps.append(
            raw_step(
                MenuInput.NEUTRAL,
                MENU_TIMING.menu_settle_frames,
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
        self._handle_terminal_race_step(
            session=session,
            info=terminal_info,
        )
        if self._attempt_id is None:
            self._phase = CareerPhase.WAIT_FOR_GP_RACE
            self._pending_steps.clear()
            return True
        self._enter_continue_after_race()
        return True

    def _handle_terminal_race_step(
        self,
        *,
        session: CareerRuntimeSession,
        info: dict[str, object],
    ) -> None:
        if self._attempt_id is None:
            return

        persist_save_ram_for_store(self._store, self._save_game_id, session)
        progress = self._store.save_game_unlock_progress(self._save_game_id)
        self._unlock_progress = progress
        if _target_succeeded(progress.targets, self._setup):
            self._finish_attempt(info=info, status="succeeded", failure_reason=None)
            self._advance_after_finished_attempt()
            return

        if info.get("termination_reason") == "finished":
            return

        reason = _failure_reason(info)
        self._finish_attempt(info=info, status="failed", failure_reason=reason)
        self._advance_after_finished_attempt()

    def _finish_attempt(
        self,
        *,
        info: dict[str, object],
        status: SaveAttemptStatus,
        failure_reason: str | None,
    ) -> None:
        if self._attempt_id is None:
            return
        finish_time_ms = _positive_int_info(info, "race_time_ms")
        self._store.finish_save_attempt(
            attempt_id=self._attempt_id,
            status=status,
            finish_position=_positive_int_info(info, "position"),
            finish_time_s=(finish_time_ms / 1000.0 if finish_time_ms is not None else None),
            failure_reason=failure_reason,
        )

    def _advance_after_finished_attempt(self) -> None:
        progress = self._store.save_game_unlock_progress(self._save_game_id)
        self._unlock_progress = progress
        if progress.next_target is None:
            self._store.update_save_game_status(
                save_game_id=self._save_game_id,
                status="finished",
            )
            self._attempt_id = None
            return

        next_attempt = self._store.start_next_save_attempt(self._save_game_id)
        context = self._store.get_save_attempt_execution_context(next_attempt.id)
        if context is None:
            raise RuntimeError(
                f"save attempt disappeared before Career Mode could continue: {next_attempt.id}"
            )
        self._apply_execution_plan(build_save_race_execution_plan(context))

    def _apply_execution_plan(self, plan: SaveRaceExecutionPlan) -> None:
        self._attempt_id = plan.attempt_id
        self._setup = career_mode_race_setup_config(plan.race_setup)
        self._course_setups = self._store.list_save_course_setups(self._save_game_id)
        self._unlock_progress = self._store.save_game_unlock_progress(self._save_game_id)
        self._pending_steps.clear()
        self._phase = CareerPhase.CONTINUE_AFTER_RACE
        self._engine_applied = False
        self._start_presses_in_phase = 0
        self._active_policy_key = None
        self._awaiting_new_race_after_terminal = True
        self._continuing_race_result = True
        self._observed_terminal_race_result = True
        self._continue_after_race_pulses = 0
        self._fresh_race_ready_frames = 0
        self._reset_camera_sync()

    def _enter_continue_after_race(self) -> None:
        self._pending_steps.clear()
        self._phase = CareerPhase.CONTINUE_AFTER_RACE
        self._start_presses_in_phase = 0
        self._awaiting_new_race_after_terminal = True
        self._continuing_race_result = True
        self._observed_terminal_race_result = True
        self._continue_after_race_pulses = 0
        self._fresh_race_ready_frames = 0
        self._reset_camera_sync()

    def _camera_ready(self, info: dict[str, object]) -> bool:
        facts = MenuFacts.from_info(info)
        if self._camera_setting is None:
            return True
        if facts.camera_setting == self._camera_setting:
            self._camera_synced = True
            return True
        return self._camera_synced

    def _next_camera_sync_step(
        self,
        info: dict[str, object],
    ) -> RawMenuStep | None:
        facts = MenuFacts.from_info(info)
        if self._camera_setting is None or self._camera_synced:
            return None
        if facts.camera_setting == self._camera_setting:
            self._camera_synced = True
            return None
        if not _race_intro_ready_for_camera(info):
            return raw_step(
                MenuInput.NEUTRAL,
                1,
                phase="camera_sync:wait_intro",
            )
        return raw_step(MenuInput.NEUTRAL, 1, phase="camera_sync:apply")

    def _sync_camera_before_policy_handoff(
        self,
        *,
        session: CareerRuntimeSession,
        info: dict[str, object],
    ) -> bool:
        if (
            self._camera_setting is None
            or self._camera_synced
            or self._phase not in {CareerPhase.ENTER_RACE, CareerPhase.CONTINUE_AFTER_RACE}
            or not in_gp_race(info)
            or not _race_intro_ready_for_camera(info)
        ):
            return False

        telemetry = session.emulator.try_read_telemetry()
        try:
            telemetry = sync_camera_setting(
                session.emulator,
                target_name=self._camera_setting,
                telemetry=telemetry,
                info=info,
            )
        except RuntimeError as exc:
            info["camera_setting_sync"] = "retry"
            info["camera_setting_sync_reason"] = str(exc)
            return False
        if telemetry is None:
            return False
        info.update(telemetry_info(telemetry))
        if telemetry.camera_setting_name == self._camera_setting:
            self._camera_synced = True
            self._camera_sync_taps = int(info.get("camera_setting_taps") or 0)
        return True

    def _reset_camera_sync(self) -> None:
        self._camera_synced = self._camera_setting is None
        self._camera_sync_taps = 0

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
                MENU_TIMING.menu_settle_frames,
                phase=f"{step.phase}:settle",
            )
        )
        return step


def _is_neutral_settle_step(step: RawMenuStep) -> bool:
    return step.menu_input is MenuInput.NEUTRAL and step.phase.endswith(":settle")


def _target_succeeded(
    targets: Iterable[object],
    setup: CareerModeRaceSetupConfig,
) -> bool:
    return any(
        getattr(target, "status", None) == "succeeded"
        and getattr(target, "kind", None) == "clear_gp_cup"
        and getattr(target, "difficulty", None) == setup.difficulty
        and getattr(target, "cup_id", None) == setup.cup_id
        for target in targets
    )


def _current_target_label(
    progress: ManagedSaveUnlockProgress,
    setup: CareerModeRaceSetupConfig,
) -> str:
    target = next(
        (
            target
            for target in progress.targets
            if target.kind == "clear_gp_cup"
            and target.difficulty == setup.difficulty
            and target.cup_id == setup.cup_id
        ),
        None,
    )
    if target is not None:
        return target.label
    return f"Clear {setup.difficulty.title()} {setup.cup_id.title()} Cup"


def _failure_reason(info: dict[str, object]) -> str:
    reason = info.get("termination_reason")
    return reason if isinstance(reason, str) and reason else "race ended before cup clear"


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
    if facts.is_skippable_post_gp_screen:
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


def _positive_int_info(info: dict[str, object], key: str) -> int | None:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _lap_summary(info: dict[str, object]) -> str | None:
    completed = info.get("race_laps_completed")
    total = info.get("total_lap_count")
    if (
        isinstance(completed, bool)
        or isinstance(total, bool)
        or not isinstance(completed, int | float)
        or not isinstance(total, int | float)
    ):
        return None
    return f"{int(completed)}/{int(total)}"


def _race_intro_ready_for_camera(info: dict[str, object]) -> bool:
    transition_state = info.get("menu_transition_state_raw")
    if isinstance(transition_state, int) and not isinstance(transition_state, bool):
        if transition_state != 0:
            return False
    timer = info.get("race_intro_timer")
    if isinstance(timer, bool) or not isinstance(timer, int):
        return False
    return 0 < timer < CAMERA_SYNC_CONTROLS.ready_intro_timer


def _validated_camera_setting(value: str | None) -> CameraSettingName | None:
    match value:
        case None:
            return None
        case "overhead" | "close_behind" | "regular" | "wide":
            return value
        case _:
            raise ValueError(f"Unsupported camera setting {value!r}")
