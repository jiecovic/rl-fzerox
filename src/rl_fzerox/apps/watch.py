# src/rl_fzerox/apps/watch.py
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from rl_fzerox.apps._cli import normalize_hydra_overrides
from rl_fzerox.core.config import load_watch_app_config
from rl_fzerox.core.config.schema import TrainAppConfig, WatchAppConfig, WatchConfig
from rl_fzerox.core.training.runs import (
    apply_train_run_to_watch_config,
    load_train_run_config,
    materialize_watch_session_config,
)
from rl_fzerox.ui.watch import run_viewer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the watch app."""

    parser = argparse.ArgumentParser(
        description="Watch the F-Zero X environment from a Hydra-composed YAML config.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        "--config-file",
        dest="config_path",
        type=Path,
        default=None,
        help="Path to a watch config YAML file.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides. Use `-- key=value` to separate them from CLI flags.",
    )
    parser.add_argument(
        "--run-dir",
        dest="policy_run_dir",
        type=Path,
        default=None,
        help=(
            "Optional training run directory. The watch app loads its latest saved policy artifact."
        ),
    )
    parser.add_argument(
        "--artifact",
        dest="policy_artifact",
        choices=("latest", "best", "final"),
        default=None,
        help="Which saved policy artifact to load from the run directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load the watch config and launch the viewer."""

    args = parse_args(argv)
    normalized_overrides = normalize_hydra_overrides(args.overrides)
    cli_run_dir = (
        args.policy_run_dir.expanduser().resolve() if args.policy_run_dir is not None else None
    )
    cli_override_delta: dict[str, object] = {}
    if args.config_path is None:
        if cli_run_dir is None:
            raise SystemExit("--config is required unless --run-dir is provided")
        if normalized_overrides:
            raise SystemExit("Hydra overrides require --config")
        try:
            train_config = load_train_run_config(cli_run_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        config = _default_watch_config_from_train_run(
            train_config,
            run_dir=cli_run_dir,
            artifact=args.policy_artifact or "latest",
        )
    else:
        try:
            config = load_watch_app_config(args.config_path)
            if normalized_overrides:
                overridden_config = load_watch_app_config(
                    args.config_path,
                    overrides=normalized_overrides,
                )
                cli_override_delta = _watch_config_delta(
                    config,
                    overridden_config,
                    normalized_overrides,
                )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    policy_run_dir = cli_run_dir if cli_run_dir is not None else config.watch.policy_run_dir
    if cli_run_dir is None and cli_override_delta:
        policy_run_dir = _apply_watch_config_delta(config, cli_override_delta).watch.policy_run_dir
    if args.policy_artifact is not None and policy_run_dir is None:
        raise SystemExit("--artifact requires --run-dir or watch.policy_run_dir in the config")
    if policy_run_dir is not None:
        try:
            train_config = load_train_run_config(policy_run_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        config = apply_train_run_to_watch_config(
            config,
            run_dir=policy_run_dir,
            train_config=train_config,
        )
        if cli_override_delta:
            config = _apply_watch_config_delta(config, cli_override_delta)
        if args.policy_artifact is not None:
            config = config.model_copy(
                update={
                    "watch": config.watch.model_copy(
                        update={"policy_artifact": args.policy_artifact}
                    )
                }
            )
    elif cli_override_delta:
        config = _apply_watch_config_delta(config, cli_override_delta)

    config = materialize_watch_session_config(
        config,
        run_dir=config.watch.policy_run_dir,
    )
    run_viewer(config)


def _default_watch_config_from_train_run(
    train_config: TrainAppConfig,
    *,
    run_dir: Path,
    artifact: Literal["latest", "best", "final"],
) -> WatchAppConfig:
    """Build one minimal watch config directly from a saved train run."""

    return WatchAppConfig(
        seed=train_config.seed,
        emulator=train_config.emulator,
        env=train_config.env,
        reward=train_config.reward,
        curriculum=train_config.curriculum,
        watch=WatchConfig(
            policy_run_dir=run_dir,
            policy_artifact=artifact,
        ),
    )


def _watch_config_delta(
    base_config: WatchAppConfig,
    overridden_config: WatchAppConfig,
    overrides: Sequence[str],
) -> dict[str, object]:
    """Return only the effective CLI changes after Hydra composition."""

    delta = _mapping_delta(
        _watch_config_data(base_config),
        _watch_config_data(overridden_config),
    )
    overridden_data = _watch_config_data(overridden_config)
    for override in overrides:
        key_path = _override_key_path(override)
        if key_path is None:
            continue
        override_value = _mapping_path_value(overridden_data, key_path)
        if override_value is None:
            continue
        value, path = override_value
        _set_mapping_path_value(delta, path, value)
    return delta


def _apply_watch_config_delta(
    config: WatchAppConfig,
    delta: Mapping[str, object],
) -> WatchAppConfig:
    """Apply CLI override delta after run-manifest inheritance."""

    if not delta:
        return config
    expanded_delta = _expand_legacy_watch_fps_delta(delta)
    return WatchAppConfig.model_validate(
        _merge_mapping(
            _watch_config_data(config),
            expanded_delta,
        )
    )


def _expand_legacy_watch_fps_delta(delta: Mapping[str, object]) -> dict[str, object]:
    """Keep CLI `watch.fps` overrides equivalent to both split FPS fields."""

    # COMPAT SHIM: old watch CLI/config overrides used one `watch.fps` knob.
    # Expand it here so CLI precedence still behaves like the new split fields.
    expanded: dict[str, object] = dict(delta)
    watch_delta = _string_key_mapping(expanded.get("watch"))
    if watch_delta is None or "fps" not in watch_delta:
        return expanded

    expanded_watch_delta = dict(watch_delta)
    legacy_fps = expanded_watch_delta["fps"]
    expanded_watch_delta.setdefault("control_fps", legacy_fps)
    expanded_watch_delta.setdefault("render_fps", legacy_fps)
    expanded["watch"] = expanded_watch_delta
    return expanded


def _mapping_delta(
    base: Mapping[str, object],
    overridden: Mapping[str, object],
) -> dict[str, object]:
    delta: dict[str, object] = {}
    for key, overridden_value in overridden.items():
        if key not in base:
            delta[key] = overridden_value
            continue

        base_value = base[key]
        base_mapping = _string_key_mapping(base_value)
        overridden_mapping = _string_key_mapping(overridden_value)
        if base_mapping is not None and overridden_mapping is not None:
            nested_delta = _mapping_delta(base_mapping, overridden_mapping)
            if nested_delta:
                delta[key] = nested_delta
            continue

        if overridden_value != base_value:
            delta[key] = overridden_value
    return delta


def _merge_mapping(
    base: Mapping[str, object],
    update: Mapping[str, object],
) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, update_value in update.items():
        existing_mapping = _string_key_mapping(merged.get(key))
        update_mapping = _string_key_mapping(update_value)
        if existing_mapping is not None and update_mapping is not None:
            merged[key] = _merge_mapping(existing_mapping, update_mapping)
            continue
        merged[key] = update_value
    return merged


def _watch_config_data(config: WatchAppConfig) -> dict[str, object]:
    data = _string_key_mapping(config.model_dump(mode="json", exclude_none=False))
    if data is None:
        raise TypeError("Watch config dump must be a string-keyed mapping")
    return data


def _override_key_path(override: str) -> tuple[str, ...] | None:
    key = override.split("=", maxsplit=1)[0].strip()
    while key.startswith("+"):
        key = key[1:]
    if key.startswith("~"):
        key = key[1:]
    if not key or key.startswith("hydra."):
        return None
    if key.startswith("/"):
        key = key.rsplit("/", maxsplit=1)[-1]
    if "@" in key:
        key = key.split("@", maxsplit=1)[0]

    path = tuple(part for part in key.split(".") if part)
    return path or None


def _mapping_path_value(
    mapping: Mapping[str, object],
    path: tuple[str, ...],
) -> tuple[object, tuple[str, ...]] | None:
    current: object = mapping
    for part in path:
        current_mapping = _string_key_mapping(current)
        if current_mapping is None or part not in current_mapping:
            return None
        current = current_mapping[part]
    return current, path


def _set_mapping_path_value(
    mapping: dict[str, object],
    path: tuple[str, ...],
    value: object,
) -> None:
    current = mapping
    for part in path[:-1]:
        next_mapping = _string_key_mapping(current.get(part))
        if next_mapping is None:
            next_mapping = {}
            current[part] = next_mapping
        current = next_mapping
    current[path[-1]] = value


def _string_key_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None

    mapping: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        mapping[key] = item
    return mapping


if __name__ == "__main__":
    main()
