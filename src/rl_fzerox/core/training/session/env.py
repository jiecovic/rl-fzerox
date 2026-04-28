# src/rl_fzerox/core/training/session/env.py
from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import gymnasium as gym

from fzerox_emulator import Emulator
from rl_fzerox.core.config.schema import TrainAppConfig
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.envs.info import MONITOR_INFO_KEYS
from rl_fzerox.core.seed import derive_seed
from rl_fzerox.core.training.runs import RunPaths
from rl_fzerox.core.training.session.observation_augmentation import (
    maybe_wrap_training_observation_augmentation,
)

_DOMAIN_TRAIN_ENV = 0xA4C4F4B7A62D1131


class MonitorWrapper(Protocol):
    """Callable surface needed from SB3's Monitor class."""

    def __call__(
        self,
        env: gym.Env,
        *,
        info_keywords: tuple[str, ...],
    ) -> gym.Env: ...


def build_training_env(config: TrainAppConfig, run_paths: RunPaths):
    """Construct the configured SB3 vector env around emulator workers."""

    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc

    if config.train.vec_env == "dummy" and config.train.num_envs > 1:
        raise RuntimeError(
            "DummyVecEnv with num_envs > 1 is not supported for the emulator training path. "
            "Use `train.vec_env=subproc` or set `train.num_envs=1`."
        )

    env_fns = [
        _make_env_factory(
            config=config,
            env_index=env_index,
            runtime_dir=run_paths.env_runtime_dir(env_index),
            monitor_cls=Monitor,
        )
        for env_index in range(config.train.num_envs)
    ]
    if config.train.vec_env == "dummy":
        return DummyVecEnv(env_fns)
    if config.train.vec_env == "subproc":
        return SubprocVecEnv(
            env_fns,
            start_method=_subproc_start_method(),
        )
    raise ValueError(f"Unsupported vec env kind: {config.train.vec_env!r}")


def _make_env_factory(
    *,
    config: TrainAppConfig,
    env_index: int,
    runtime_dir: Path,
    monitor_cls: MonitorWrapper,
) -> Callable[[], gym.Env]:
    """Build one worker-local env factory for SB3 vectorized training."""

    def _make_env():
        emulator = Emulator(
            core_path=config.emulator.core_path,
            rom_path=config.emulator.rom_path,
            runtime_dir=runtime_dir,
            baseline_state_path=config.emulator.baseline_state_path,
            renderer=config.emulator.renderer,
        )
        env = FZeroXEnv(
            backend=emulator,
            config=config.env,
            reward_config=config.reward,
            curriculum_config=config.curriculum,
            env_index=env_index,
        )
        env = maybe_wrap_training_observation_augmentation(
            env,
            env_config=config.env,
            train_config=config.train,
        )
        wrapped = monitor_cls(env, info_keywords=MONITOR_INFO_KEYS)
        initial_seed = derive_seed(
            config.seed,
            _DOMAIN_TRAIN_ENV,
            env_index,
        )
        # The env keeps the Gym-compatible seed surface even though emulator
        # resets are deterministic today; this still seeds any future Python-side
        # randomness attached to reset behavior.
        if initial_seed is not None:
            wrapped.reset(seed=int(initial_seed))
        return wrapped

    return _make_env


def _subproc_start_method() -> str:
    """Prefer CUDA-safe multiprocessing start methods for emulator workers."""

    available_methods = mp.get_all_start_methods()
    if "forkserver" in available_methods:
        return "forkserver"
    return "spawn"
