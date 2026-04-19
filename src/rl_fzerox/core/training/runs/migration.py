# src/rl_fzerox/core/training/runs/migration.py
"""Explicit maintenance tools for stale saved run manifests."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from rl_fzerox.core.config.paths import resolve_config_data_paths
from rl_fzerox.core.config.schema import TrainAppConfig
from rl_fzerox.core.training.runs.paths import resolve_train_run_config_path

_OBSOLETE_REWARD_FIELDS = frozenset(
    (
        "milestone_distance",
        "randomize_milestone_phase_on_reset",
        "milestone_bonus",
        "milestone_speed_scale",
        "milestone_speed_bonus_cap",
        "bootstrap_progress_scale",
        "bootstrap_regress_penalty_scale",
        "bootstrap_position_multiplier_scale",
        "bootstrap_lap_count",
        "lap_1_completion_bonus",
        "lap_2_completion_bonus",
        "final_lap_completion_bonus",
        "remaining_step_penalty_per_frame",
        "remaining_lap_penalty",
        "energy_loss_penalty_scale",
        "energy_loss_safe_fraction",
        "energy_loss_danger_power",
        "energy_full_refill_bonus",
        "energy_full_refill_cooldown_frames",
        "grounded_air_brake_penalty",
        "drive_axis_negative_penalty_scale",
        "boost_pad_reward_cooldown_frames",
        "manual_boost_request_reward",
        "spinning_out_penalty",
        "terminal_failure_base_penalty",
        "stuck_truncation_base_penalty",
        "wrong_way_truncation_base_penalty",
        "progress_stalled_truncation_base_penalty",
        "timeout_truncation_base_penalty",
        "finish_position_scale",
        "lap_time_bonus_scale",
        "lap_time_bonus_power",
        "finish_time_target_ms",
        "finish_time_bonus_scale",
        "finish_time_bonus_power",
        "energy_gain_reward_scale",
        "energy_gain_collision_cooldown_frames",
    )
)
_OBSOLETE_TRACK_FIELDS = frozenset(
    (
        "finish_time_target_ms",
        "human_reference_finish_time_ms",
        "human_reference_lap_time_ms",
        "reference_source",
        "reference_url",
        "ghost",
    )
)
_OBSOLETE_TRACK_SAMPLING_ENTRY_FIELDS = frozenset(("finish_time_target_ms", "ghost"))
_OBSOLETE_ENV_FIELDS = frozenset(("benchmark_noop_reset",))


@dataclass(frozen=True, slots=True)
class TrainConfigScrubResult:
    """Result from an explicit stale-manifest scrub operation."""

    source_path: Path
    output_path: Path | None
    backup_path: Path | None
    removed_fields: tuple[str, ...]


def scrub_obsolete_train_config_data(config_data: dict[str, object]) -> tuple[str, ...]:
    """Drop known stale saved-run fields before validating old local manifests.

    V4 LEGACY SHIM: this is intentionally narrow and should disappear once
    `exp_v4_*` runs are no longer worth loading.
    """

    return tuple(_scrub_obsolete_fields(config_data))


def scrub_obsolete_train_run_config(
    run_dir: Path,
    *,
    output_path: Path | None = None,
    in_place: bool = False,
    backup: bool = True,
) -> TrainConfigScrubResult:
    """Remove known obsolete saved-manifest fields and validate the result."""

    if in_place and output_path is not None:
        raise ValueError("Use either in_place=True or output_path, not both")

    source_path = resolve_train_run_config_path(run_dir)
    config_data = _load_mapping(source_path)
    removed_fields = _scrub_obsolete_fields(config_data)
    _resolve_paths(config_data, config_dir=source_path.parent)
    TrainAppConfig.model_validate(config_data)

    target_path = source_path if in_place else output_path
    backup_path = None
    if target_path is not None:
        target_path = target_path.expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if in_place and backup:
            backup_path = source_path.with_suffix(source_path.suffix + ".bak")
            shutil.copy2(source_path, backup_path)
        OmegaConf.save(config=OmegaConf.create(config_data), f=str(target_path))

    return TrainConfigScrubResult(
        source_path=source_path,
        output_path=target_path,
        backup_path=backup_path,
        removed_fields=tuple(removed_fields),
    )


def _load_mapping(config_path: Path) -> dict[str, object]:
    loaded = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Train run config must resolve to a mapping: {config_path}")
    normalized: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise ValueError(f"Train run config keys must be strings: {config_path}")
        normalized[key] = value
    return normalized


def _scrub_obsolete_fields(config_data: dict[str, object]) -> list[str]:
    removed: list[str] = []
    _remove_mapping_keys(config_data.get("env"), "env", _OBSOLETE_ENV_FIELDS, removed)
    _remove_mapping_keys(config_data.get("reward"), "reward", _OBSOLETE_REWARD_FIELDS, removed)
    _remove_mapping_keys(config_data.get("track"), "track", _OBSOLETE_TRACK_FIELDS, removed)
    _scrub_track_sampling_sections(config_data, removed)
    return removed


def _resolve_paths(config_data: dict[str, object], *, config_dir: Path) -> None:
    resolve_config_data_paths(
        config_data,
        config_dir=config_dir,
        path_fields={
            "emulator": (
                "core_path",
                "rom_path",
                "runtime_dir",
                "baseline_state_path",
            ),
            "train": (
                "output_root",
                "init_run_dir",
            ),
        },
    )


def _scrub_track_sampling_sections(config_data: dict[str, object], removed: list[str]) -> None:
    env_data = config_data.get("env")
    if isinstance(env_data, dict):
        _scrub_track_sampling_section(env_data.get("track_sampling"), "env.track_sampling", removed)

    curriculum_data = config_data.get("curriculum")
    if not isinstance(curriculum_data, dict):
        return
    stages = curriculum_data.get("stages")
    if not isinstance(stages, list):
        return
    for stage_index, stage in enumerate(stages):
        if isinstance(stage, dict):
            _scrub_track_sampling_section(
                stage.get("track_sampling"),
                f"curriculum.stages.{stage_index}.track_sampling",
                removed,
            )


def _scrub_track_sampling_section(
    track_sampling_data: object,
    prefix: str,
    removed: list[str],
) -> None:
    if not isinstance(track_sampling_data, dict):
        return
    _rename_mapping_key(
        track_sampling_data,
        prefix,
        old_key="mode",
        new_key="sampling_mode",
        removed=removed,
    )
    entries = track_sampling_data.get("entries")
    if not isinstance(entries, list):
        return
    for index, entry in enumerate(entries):
        _remove_mapping_keys(
            entry,
            f"{prefix}.entries.{index}",
            _OBSOLETE_TRACK_SAMPLING_ENTRY_FIELDS,
            removed,
        )


def _rename_mapping_key(
    mapping: dict[object, object],
    prefix: str,
    *,
    old_key: str,
    new_key: str,
    removed: list[str],
) -> None:
    if old_key not in mapping:
        return
    if new_key not in mapping:
        mapping[new_key] = mapping[old_key]
    del mapping[old_key]
    removed.append(f"{prefix}.{old_key}")


def _remove_mapping_keys(
    mapping: object,
    prefix: str,
    obsolete_keys: frozenset[str],
    removed: list[str],
) -> None:
    if not isinstance(mapping, dict):
        return
    for key in sorted(obsolete_keys):
        if key not in mapping:
            continue
        del mapping[key]
        removed.append(f"{prefix}.{key}")
