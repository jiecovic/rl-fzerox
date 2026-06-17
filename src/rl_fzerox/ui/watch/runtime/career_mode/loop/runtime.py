# src/rl_fzerox/ui/watch/runtime/career_mode/loop/runtime.py
from __future__ import annotations

from collections.abc import Mapping

from fzerox_emulator import FZeroXTelemetry
from rl_fzerox.core.career_mode.controller import CareerModeController
from rl_fzerox.core.career_mode.execution.save_file import load_save_ram
from rl_fzerox.core.career_mode.navigation import RawMenuStep, in_gp_race
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.runtime.career_mode.menu import menu_viewer_info
from rl_fzerox.ui.watch.runtime.career_mode.session import CareerModeRuntimeSession
from rl_fzerox.ui.watch.runtime.telemetry import _read_live_telemetry


def career_runtime_error_context(
    exc: Exception,
    *,
    controller: CareerModeController,
    info: dict[str, object],
    last_menu_step: RawMenuStep | None,
) -> str:
    context = controller.debug_context(info)
    message = f"{exc}; {context}"
    if last_menu_step is None:
        return message
    return (
        f"{message}; last_step="
        f"{last_menu_step.phase}:{last_menu_step.menu_input}:{last_menu_step.frames}f"
    )


def fresh_menu_runtime_state(
    session: CareerModeRuntimeSession,
) -> tuple[dict[str, object], dict[str, object], FZeroXTelemetry | None]:
    raw_info = menu_viewer_info(session)
    info = dict(raw_info)
    telemetry = _read_live_telemetry(session.emulator)
    return raw_info, info, telemetry


def reset_emulator_for_next_attempt(
    *,
    config: WatchAppConfig,
    session: CareerModeRuntimeSession,
    controller: CareerModeController,
) -> tuple[dict[str, object], dict[str, object], FZeroXTelemetry | None]:
    load_save_ram(config, session)
    session.emulator.reset()
    raw_info, info, telemetry = fresh_menu_runtime_state(session)
    return raw_info, controller.viewer_info(info=info, active_policy_control=None), telemetry


def should_observe_policy_transition(
    *,
    policy_owns_control: bool,
    active_policy_started: bool,
    info: dict[str, object],
) -> bool:
    if not policy_owns_control:
        return False
    if active_policy_started:
        return True
    return not in_gp_race(info)


def career_mode_attempt_id(info: Mapping[str, object]) -> str | None:
    value = info.get("career_mode_attempt_id")
    return value if isinstance(value, str) and value else None
