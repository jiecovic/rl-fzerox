# src/rl_fzerox/ui/watch/runtime/live/session.py
"""Runtime session construction for the live watch worker.

This file opens the emulator env, optional policy runner, engine-tuning cache,
and watch-specific state-feature masking inputs. The worker loop consumes the
session; it should not duplicate setup or artifact-resolution rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fzerox_emulator import Emulator
from rl_fzerox.core.engine_tuning.contexts import engine_tuning_contexts_for_track_sampling
from rl_fzerox.core.engine_tuning.training import EngineTuningTrainingController
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, WatchAppConfig
from rl_fzerox.core.seed import seed_process
from rl_fzerox.core.training.runs import resolve_policy_artifact_path
from rl_fzerox.core.training.session.artifacts import (
    engine_tuning_checkpoint_path,
    engine_tuning_model_path,
    load_engine_tuning_checkpoint_state,
)
from rl_fzerox.ui.watch.runtime.observation import configured_watch_zeroed_features
from rl_fzerox.ui.watch.runtime.policy.runner import _load_policy_runner
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
    engine_tuning_cache: WatchEngineTuningStateCache
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
    )
    env.set_sequential_track_sampling(True)
    env.set_engine_tuning_selection("greedy" if config.watch.deterministic_policy else "sample")
    engine_tuning_cache = WatchEngineTuningStateCache(config)
    engine_tuning_cache.refresh(env, track_sampling=config.env.track_sampling)
    policy_runner = _load_policy_runner(
        config.watch.policy_run_dir,
        artifact=config.watch.policy_artifact,
        device=config.watch.device,
        algorithm=config.watch.policy_algorithm,
    )

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
        engine_tuning_cache=engine_tuning_cache,
        watch_zeroed_state_features=configured_watch_zeroed_features(config),
        auxiliary_target_names=auxiliary_target_names,
    )


@dataclass(frozen=True, slots=True)
class _EngineTuningFileSignature:
    policy_path: Path | None
    state_mtime_ns: int | None
    state_size: int | None
    model_mtime_ns: int | None
    model_size: int | None
    context_keys: tuple[str, ...]


@dataclass(slots=True)
class WatchEngineTuningStateCache:
    """Refresh watch engine-tuning sampler only when its checkpoint changes."""

    config: WatchAppConfig
    _signature: _EngineTuningFileSignature | None = None

    def refresh(self, env: FZeroXEnv, *, track_sampling: TrackSamplingConfig) -> None:
        """Install a fresh reset sampler when latest tuner artifacts changed."""

        if not track_sampling.engine_tuning.enabled:
            return
        contexts = engine_tuning_contexts_for_track_sampling(track_sampling)
        policy_path = _watch_policy_artifact_path(self.config)
        signature = _engine_tuning_signature(
            policy_path=policy_path,
            context_keys=tuple(context.key for context in contexts),
        )
        if signature == self._signature:
            return
        state = None if policy_path is None else load_engine_tuning_checkpoint_state(policy_path)
        controller = EngineTuningTrainingController(
            track_sampling.engine_tuning,
            state=state,
        )
        env.set_engine_tuning_sampler(controller.reset_sampler_snapshot(contexts))
        self._signature = signature


def _watch_policy_artifact_path(config: WatchAppConfig) -> Path | None:
    if config.watch.policy_run_dir is None:
        return None
    try:
        return resolve_policy_artifact_path(
            config.watch.policy_run_dir,
            artifact=config.watch.policy_artifact,
        )
    except FileNotFoundError:
        return None


def _engine_tuning_signature(
    *,
    policy_path: Path | None,
    context_keys: tuple[str, ...],
) -> _EngineTuningFileSignature:
    if policy_path is None:
        return _EngineTuningFileSignature(
            policy_path=None,
            state_mtime_ns=None,
            state_size=None,
            model_mtime_ns=None,
            model_size=None,
            context_keys=context_keys,
        )
    state_mtime_ns, state_size = _file_identity(engine_tuning_checkpoint_path(policy_path))
    model_mtime_ns, model_size = _file_identity(engine_tuning_model_path(policy_path))
    return _EngineTuningFileSignature(
        policy_path=policy_path,
        state_mtime_ns=state_mtime_ns,
        state_size=state_size,
        model_mtime_ns=model_mtime_ns,
        model_size=model_size,
        context_keys=context_keys,
    )


def _file_identity(path: Path) -> tuple[int | None, int | None]:
    try:
        stat_result = path.stat()
    except OSError:
        return None, None
    return stat_result.st_mtime_ns, stat_result.st_size
