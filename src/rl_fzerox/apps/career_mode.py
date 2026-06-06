# src/rl_fzerox/apps/career_mode.py
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rl_fzerox.apps._cli import normalize_cli_overrides
from rl_fzerox.apps.viewer_runtime import manager_viewer_lease_session
from rl_fzerox.apps.watch_cli.delta import (
    apply_watch_config_delta,
    watch_config_delta_from_dotlist,
)
from rl_fzerox.core.career_mode.runner.race import (
    SaveRaceSetup,
    build_save_race_execution_plan,
)
from rl_fzerox.core.manager import ManagerStore, default_manager_db_path
from rl_fzerox.core.manager.projection.assembly import emulator_data
from rl_fzerox.core.runtime_spec.schema import (
    CareerModeRaceSetupConfig,
    EmulatorConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.ui.watch import run_viewer
from rl_fzerox.ui.watch.runtime import start_career_mode_worker


def main(argv: Sequence[str] | None = None) -> None:
    """Launch the visible Career Mode runner for one managed save game."""

    args = _parse_args(argv)
    db_path = args.manager_db_path.expanduser().resolve()
    with manager_viewer_lease_session(
        db_path=db_path,
        lease_id=args.viewer_lease_id,
    ):
        try:
            config = _resolve_career_mode_config(
                db_path=db_path,
                attempt_seed=args.attempt_seed,
                deterministic_policy=args.policy_mode == "deterministic",
                save_game_id=args.save_game_id,
                overrides=args.overrides,
            )
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        run_viewer(config, worker_factory=start_career_mode_worker)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the F-Zero X Career Mode viewer for one managed save game.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Viewer overrides. Use `-- key=value` to separate them from CLI flags.",
    )
    parser.add_argument(
        "--manager-db-path",
        dest="manager_db_path",
        type=Path,
        default=default_manager_db_path(),
        help="Manager SQLite database path.",
    )
    parser.add_argument(
        "--save-game-id",
        dest="save_game_id",
        required=True,
        help="Managed save-game id to run.",
    )
    parser.add_argument(
        "--viewer-lease-id",
        dest="viewer_lease_id",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--attempt-seed",
        dest="attempt_seed",
        type=int,
        default=None,
        help="Seed used only for stochastic policy playback.",
    )
    parser.add_argument(
        "--policy-mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
        help="Initial policy playback mode; the viewer hotkey can still toggle it.",
    )
    args = parser.parse_args(argv)
    if args.attempt_seed is not None and not (0 <= args.attempt_seed <= 0xFFFFFFFF):
        parser.error("--attempt-seed must be between 0 and 4294967295")
    return args


def _resolve_career_mode_config(
    *,
    db_path: Path,
    attempt_seed: int | None,
    deterministic_policy: bool,
    save_game_id: str,
    overrides: Sequence[str],
) -> WatchAppConfig:
    store = ManagerStore(db_path)
    attempt = store.start_or_reuse_next_save_attempt(save_game_id)
    context = store.get_save_attempt_execution_context(attempt.id)
    if context is None:
        raise RuntimeError(f"save attempt disappeared before launch: {attempt.id}")
    plan = build_save_race_execution_plan(context)
    config = _career_mode_base_config(
        db_path=db_path,
        save_game_id=context.save_game.id,
        save_path=context.save_game.save_path,
        attempt_id=context.attempt.id,
        emulator=EmulatorConfig.model_validate(emulator_data(context.policy_run.config)),
        attempt_seed=attempt_seed,
        deterministic_policy=deterministic_policy,
        race_setup=_career_mode_race_setup_config(plan.race_setup),
        label=context.target.label,
    )
    if delta := watch_config_delta_from_dotlist(normalize_cli_overrides(overrides)):
        config = apply_watch_config_delta(config, delta)

    store.update_save_game_status(save_game_id=save_game_id, status="running")
    return config


def _career_mode_base_config(
    *,
    db_path: Path,
    save_game_id: str,
    save_path: Path,
    attempt_id: str,
    emulator: EmulatorConfig,
    attempt_seed: int | None,
    deterministic_policy: bool,
    race_setup: CareerModeRaceSetupConfig,
    label: str,
) -> WatchAppConfig:
    emulator = _career_mode_emulator_config(
        emulator=emulator,
        save_path=save_path,
    )
    return WatchAppConfig(
        seed=None,
        emulator=emulator,
        policy=None,
        train=None,
        watch=WatchConfig(
            control_fps="auto",
            manager_db_path=db_path,
            managed_save_game_id=save_game_id,
            save_attempt_id=attempt_id,
            unlock_target_label=label,
            attempt_seed=attempt_seed,
            deterministic_policy=deterministic_policy,
            start_manual_control=False,
            career_mode_race_setup=race_setup,
        ),
    )


def _career_mode_emulator_config(
    *,
    emulator: EmulatorConfig,
    save_path: Path,
) -> EmulatorConfig:
    runtime_dir = save_path.parent / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return emulator.model_copy(
        update={
            "runtime_dir": runtime_dir,
            "baseline_state_path": None,
        },
    )


def _career_mode_race_setup_config(race_setup: SaveRaceSetup) -> CareerModeRaceSetupConfig:
    return CareerModeRaceSetupConfig(
        difficulty=race_setup.difficulty,
        cup_id=race_setup.cup_id,
        course_id=race_setup.course_id,
        vehicle_id=race_setup.vehicle_id,
        vehicle_display_name=race_setup.vehicle_display_name,
        character_index=race_setup.character_index,
        machine_select_slot=race_setup.machine_select_slot,
        machine_select_row=race_setup.machine_select_row,
        machine_select_column=race_setup.machine_select_column,
        engine_setting_id=race_setup.engine_setting_id,
        engine_setting_raw_value=race_setup.engine_setting_raw_value,
    )


if __name__ == "__main__":
    main()
