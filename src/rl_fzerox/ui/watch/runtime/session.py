# src/rl_fzerox/ui/watch/runtime/session.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fzerox_emulator import Emulator
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.core.seed import seed_process
from rl_fzerox.core.training.runs import resolve_policy_artifact_path
from rl_fzerox.core.training.session.artifacts import load_engine_tuning_checkpoint_state
from rl_fzerox.ui.watch.runtime.observation import configured_watch_zeroed_features
from rl_fzerox.ui.watch.runtime.policy import (
    _load_policy_runner,
    _sync_policy_curriculum_stage,
)
from rl_fzerox.ui.watch.runtime.timing import _resolve_control_fps, _target_seconds

if TYPE_CHECKING:
    from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName
    from rl_fzerox.core.training.inference import PolicyRunner


@dataclass(slots=True)
class WatchRuntimeSession:
    """Owned emulator/env/policy resources for one watch worker process."""

    emulator: Emulator
    env: FZeroXEnv
    policy_runner: PolicyRunner | None
    native_control_fps: float
    target_control_fps: float | None
    target_control_seconds: float | None
    watch_zeroed_state_features: frozenset[str]
    auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...]

    def close(self) -> None:
        self.env.close()


def open_watch_runtime_session(config: WatchAppConfig) -> WatchRuntimeSession:
    """Create the runtime resources that are stable across watch episodes."""

    seed_process(config.seed)
    emulator = Emulator(
        core_path=config.emulator.core_path,
        rom_path=config.emulator.rom_path,
        runtime_dir=config.emulator.runtime_dir,
        baseline_state_path=config.emulator.baseline_state_path,
        renderer=config.emulator.renderer,
    )
    env = FZeroXEnv(
        backend=emulator,
        config=config.env,
        reward_config=config.reward,
        curriculum_config=config.curriculum,
    )
    env.set_sequential_track_sampling(True)
    env.set_engine_tuning_selection(
        "greedy" if config.watch.deterministic_policy else "sample"
    )
    load_watch_engine_tuning_state(config, env)
    policy_runner = _load_policy_runner(
        config.watch.policy_run_dir,
        artifact=config.watch.policy_artifact,
        device=config.watch.device,
        algorithm=config.watch.policy_algorithm,
    )
    _sync_policy_curriculum_stage(policy_runner, env)

    native_control_fps = env.backend.native_fps / config.env.action_repeat
    target_control_fps = _resolve_control_fps(
        config.watch.control_fps,
        native_control_fps=native_control_fps,
    )
    auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...] = (
        ()
        if config.policy is None
        else tuple(loss.name for loss in config.policy.auxiliary_state.losses)
    )
    return WatchRuntimeSession(
        emulator=emulator,
        env=env,
        policy_runner=policy_runner,
        native_control_fps=native_control_fps,
        target_control_fps=target_control_fps,
        target_control_seconds=_target_seconds(target_control_fps),
        watch_zeroed_state_features=configured_watch_zeroed_features(config),
        auxiliary_target_names=auxiliary_target_names,
    )


def load_watch_engine_tuning_state(config: WatchAppConfig, env: FZeroXEnv) -> None:
    if not config.env.track_sampling.engine_tuning.enabled:
        return
    if config.watch.policy_run_dir is None:
        env.set_engine_tuning_state(None)
        return
    try:
        policy_path = resolve_policy_artifact_path(
            config.watch.policy_run_dir,
            artifact=config.watch.policy_artifact,
        )
    except FileNotFoundError:
        env.set_engine_tuning_state(None)
        return
    env.set_engine_tuning_state(load_engine_tuning_checkpoint_state(policy_path))
