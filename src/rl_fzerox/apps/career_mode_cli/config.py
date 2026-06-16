# src/rl_fzerox/apps/career_mode_cli/config.py
from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from pathlib import Path

from rl_fzerox.apps._cli import normalize_cli_overrides
from rl_fzerox.core.career_mode.runner.race import (
    SaveRaceSetup,
    build_save_race_execution_plan,
)
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.models import ManagedRun
from rl_fzerox.core.manager.projection.assembly import emulator_data
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.schema import (
    CareerModeRaceSetupConfig,
    EmulatorConfig,
    WatchAppConfig,
    WatchConfig,
    WatchRecordingConfig,
)
from rl_fzerox.core.runtime_spec.watch_overrides import (
    apply_watch_config_delta,
    watch_config_delta_from_dotlist,
)


def resolve_career_mode_config(
    *,
    db_path: Path,
    attempt_seed: int | None,
    deterministic_policy: bool,
    save_attempt_id: str | None,
    save_game_id: str,
    single_target: bool,
    perfect_run: bool,
    keep_failed_recordings: bool,
    target_clear_goal: int,
    overrides: Sequence[str],
) -> WatchAppConfig:
    store = ManagerStore(db_path)
    if save_attempt_id is None:
        attempt = store.start_or_reuse_next_save_attempt(save_game_id)
        save_attempt_id = attempt.id
    context = store.get_save_attempt_execution_context(save_attempt_id)
    if context is None:
        raise RuntimeError(f"save attempt disappeared before launch: {save_attempt_id}")
    if context.save_game.id != save_game_id:
        raise ValueError("save attempt does not belong to the requested save game")
    plan = build_save_race_execution_plan(context)
    config = career_mode_base_config(
        db_path=db_path,
        save_game_id=context.save_game.id,
        save_path=context.save_game.save_path,
        attempt_id=context.attempt.id,
        emulator=EmulatorConfig.model_validate(emulator_data(context.policy_run.config)),
        attempt_seed=attempt_seed,
        deterministic_policy=deterministic_policy,
        single_target=single_target,
        perfect_run=perfect_run,
        keep_failed_recordings=keep_failed_recordings,
        target_clear_goal=target_clear_goal,
        race_setup=career_mode_race_setup_config(plan.race_setup),
        label=context.target.label,
        policy_observation_layout_shape_hint=career_policy_observation_layout_shape_hint(
            _assigned_policy_run_configs(
                store,
                save_game_id=context.save_game.id,
                fallback_run=context.policy_run,
            )
        ),
    )
    if delta := watch_config_delta_from_dotlist(normalize_cli_overrides(overrides)):
        config = apply_watch_config_delta(config, delta)

    store.update_save_game_status(save_game_id=save_game_id, status="running")
    return config


def career_mode_base_config(
    *,
    db_path: Path,
    save_game_id: str,
    save_path: Path,
    attempt_id: str,
    emulator: EmulatorConfig,
    attempt_seed: int | None,
    deterministic_policy: bool,
    single_target: bool = False,
    perfect_run: bool = False,
    keep_failed_recordings: bool = True,
    target_clear_goal: int = 0,
    race_setup: CareerModeRaceSetupConfig,
    label: str,
    policy_observation_layout_shape_hint: tuple[int, int, int] | None = None,
) -> WatchAppConfig:
    emulator = career_mode_emulator_config(
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
            single_save_target=single_target,
            single_save_target_perfect=perfect_run,
            single_save_target_clear_goal=target_clear_goal,
            unlock_target_label=label,
            attempt_seed=attempt_seed,
            deterministic_policy=deterministic_policy,
            start_manual_control=False,
            career_mode_race_setup=race_setup,
            policy_observation_layout_shape_hint=policy_observation_layout_shape_hint,
            recording=WatchRecordingConfig(keep_failed_segments=keep_failed_recordings),
        ),
    )


def career_policy_observation_layout_shape_hint(
    policy_configs: Sequence[ManagedRunConfig],
) -> tuple[int, int, int] | None:
    """Return a synthetic shape that reserves every assigned policy preview."""

    max_preview_height = 0
    max_preview_width = 0
    max_frame_count = 1
    for config in policy_configs:
        preview_height, preview_width, frame_count = _policy_observation_preview_footprint(config)
        max_preview_height = max(max_preview_height, preview_height)
        max_preview_width = max(max_preview_width, preview_width)
        max_frame_count = max(max_frame_count, frame_count)
    if max_preview_height <= 0 or max_preview_width <= 0:
        return None
    tile_width = ceil(max_preview_width / max_frame_count)
    return max_preview_height, tile_width, 3 * max_frame_count


def _assigned_policy_run_configs(
    store: ManagerStore,
    *,
    save_game_id: str,
    fallback_run: ManagedRun,
) -> tuple[ManagedRunConfig, ...]:
    policy_configs: list[ManagedRunConfig] = [fallback_run.config]
    seen_run_ids = {fallback_run.id}
    for setup in store.list_save_course_setups(save_game_id):
        if setup.policy_run_id in seen_run_ids:
            continue
        policy_run = store.get_run(setup.policy_run_id)
        if policy_run is None:
            continue
        seen_run_ids.add(policy_run.id)
        policy_configs.append(policy_run.config)
    return tuple(policy_configs)


def _policy_observation_preview_footprint(config: ManagedRunConfig) -> tuple[int, int, int]:
    height, width = config.observation.image_geometry(renderer=config.environment.renderer)
    frame_count = int(config.observation.frame_stack) + int(config.observation.minimap_layer)
    frame_count = max(1, frame_count)
    return int(height), int(width) * frame_count, frame_count


def career_mode_emulator_config(
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


def career_mode_race_setup_config(race_setup: SaveRaceSetup) -> CareerModeRaceSetupConfig:
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
        engine_setting_raw_value=race_setup.engine_setting_raw_value,
    )
