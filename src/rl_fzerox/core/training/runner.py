# src/rl_fzerox/core/training/runner.py
from __future__ import annotations

import shutil

import numpy as np

from rl_fzerox.core.config.schema import TrainAppConfig, TrainConfig
from rl_fzerox.core.emulator import Emulator
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.seed import derive_seed, seed_process
from rl_fzerox.core.training.runs import (
    RunPaths,
    build_run_paths,
    ensure_run_dirs,
    save_train_run_config,
)

_DOMAIN_TRAIN_ENV = 0xA4C4F4B7A62D1131
_MONITOR_INFO_KEYS = (
    "episode_return",
    "episode_step",
    "termination_reason",
    "truncation_reason",
    "race_distance",
    "speed_kph",
    "position",
    "lap",
    "laps_completed",
)
_INFO_LOG_KEYS: tuple[tuple[str, str], ...] = (
    ("step_reward", "env/step_reward_mean"),
    ("race_distance", "env/race_distance_mean"),
    ("speed_kph", "env/speed_kph_mean"),
    ("position", "env/position_mean"),
    ("lap", "env/lap_mean"),
    ("laps_completed", "env/laps_completed_mean"),
)


def run_training(config: TrainAppConfig) -> None:
    """Run one PPO training session from the composed train config."""

    seed_process(config.seed)
    _validate_training_baseline_state(config)

    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    train_env = _build_training_env(config)

    try:
        model = _build_ppo_model(
            train_env=train_env,
            train_config=config.train,
            policy_config=config.policy,
            tensorboard_log=None,
        )
        ensure_run_dirs(run_paths)
        save_train_run_config(config=config, run_dir=run_paths.run_dir)
        model.set_logger(_build_tensorboard_logger(run_paths))
        _print_training_startup(
            model=model,
            train_env=train_env,
            config=config,
            run_paths=run_paths,
        )
        model.save(str(run_paths.latest_model_path))
        model.policy.save(str(run_paths.latest_policy_path))
        callbacks = _build_callbacks(
            train_config=config.train,
            run_paths=run_paths,
        )
        try:
            model.learn(
                total_timesteps=config.train.total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )
        except Exception:
            _cleanup_failed_run(run_paths, model)
            raise
        model.save(str(run_paths.final_model_path))
        model.policy.save(str(run_paths.final_policy_path))
        model.save(str(run_paths.latest_model_path))
        model.policy.save(str(run_paths.latest_policy_path))
    finally:
        train_env.close()


def _validate_training_baseline_state(config: TrainAppConfig) -> None:
    """Fail clearly when a configured local training baseline is missing."""

    baseline_state_path = config.emulator.baseline_state_path
    if baseline_state_path is None:
        return
    if baseline_state_path.exists():
        return
    raise RuntimeError(
        "Configured training baseline state does not exist: "
        f"{baseline_state_path}. Create it from watch with "
        "`emulator.baseline_state_path` set and press `K` at race start."
    )


def _build_training_env(config: TrainAppConfig):
    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc

    return DummyVecEnv(
        [
            _make_env_factory(
                config=config,
                env_index=env_index,
                monitor_cls=Monitor,
            )
            for env_index in range(config.train.num_envs)
        ]
    )


def _make_env_factory(*, config: TrainAppConfig, env_index: int, monitor_cls):
    def _make_env():
        emulator = Emulator(
            core_path=config.emulator.core_path,
            rom_path=config.emulator.rom_path,
            runtime_dir=config.emulator.runtime_dir,
            baseline_state_path=config.emulator.baseline_state_path,
        )
        env = FZeroXEnv(backend=emulator, config=config.env)
        wrapped = monitor_cls(env, info_keywords=_MONITOR_INFO_KEYS)
        initial_seed = derive_seed(
            config.seed,
            _DOMAIN_TRAIN_ENV,
            env_index,
        )
        if initial_seed is not None:
            wrapped.reset(seed=int(initial_seed))
        return wrapped

    return _make_env


def _build_ppo_model(
    *,
    train_env,
    train_config: TrainConfig,
    policy_config,
    tensorboard_log: str | None,
):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 and torch are required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc

    from rl_fzerox.core.policy import FZeroXCnnExtractor

    policy_kwargs = {
        "features_extractor_class": FZeroXCnnExtractor,
        "features_extractor_kwargs": {
            "features_dim": policy_config.extractor.features_dim,
        },
        "net_arch": {
            "pi": [int(value) for value in policy_config.net_arch.pi],
            "vf": [int(value) for value in policy_config.net_arch.vf],
        },
    }

    return PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=train_config.learning_rate,
        n_steps=train_config.n_steps,
        batch_size=train_config.batch_size,
        gamma=train_config.gamma,
        gae_lambda=train_config.gae_lambda,
        clip_range=train_config.clip_range,
        ent_coef=train_config.ent_coef,
        vf_coef=train_config.vf_coef,
        max_grad_norm=train_config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=train_config.verbose,
        device=train_config.device,
    )


def _build_tensorboard_logger(run_paths: RunPaths):
    try:
        from stable_baselines3.common import logger as sb3_logger
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc

    return sb3_logger.configure(str(run_paths.tensorboard_dir), ["tensorboard"])


def _print_training_startup(
    *,
    model,
    train_env,
    config: TrainAppConfig,
    run_paths: RunPaths,
) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        print(f"run_dir: {run_paths.run_dir}")
        print(f"device: {model.device}")
        print(f"seed: {config.seed}")
        print(f"observation_space: {train_env.observation_space}")
        print(f"action_space: {train_env.action_space}")
        print(
            "ppo: "
            f"total_timesteps={config.train.total_timesteps} "
            f"n_steps={config.train.n_steps} "
            f"batch_size={config.train.batch_size} "
            f"lr={config.train.learning_rate}"
        )
        print(model.policy)
        return

    console = Console()
    summary = Table(show_header=False, box=None, pad_edge=False)
    summary.add_column(style="bold cyan")
    summary.add_column()
    summary.add_row("run_dir", str(run_paths.run_dir))
    summary.add_row("device", str(model.device))
    summary.add_row("seed", str(config.seed))
    summary.add_row("observation", str(train_env.observation_space))
    summary.add_row("action", str(train_env.action_space))
    summary.add_row(
        "ppo",
        " ".join(
            [
                f"total_timesteps={config.train.total_timesteps}",
                f"n_steps={config.train.n_steps}",
                f"batch_size={config.train.batch_size}",
                f"lr={config.train.learning_rate}",
            ]
        ),
    )
    console.print(Panel(summary, title="Training", expand=False))
    console.print(Panel(str(model.policy), title="Policy", expand=False))


def _build_callbacks(*, train_config: TrainConfig, run_paths: RunPaths):
    try:
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc

    class InfoLoggingCallback(BaseCallback):
        """Log selected numeric env info values to TensorBoard-compatible sinks."""

        def __init__(self) -> None:
            super().__init__(verbose=0)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos")
            if not isinstance(infos, list):
                return True

            for info_key, log_key in _INFO_LOG_KEYS:
                values = _numeric_values(infos, info_key)
                if values:
                    self.logger.record(log_key, float(np.mean(values)))
            return True

    class RollingArtifactCallback(BaseCallback):
        """Maintain rolling latest and best training artifacts."""

        def __init__(self, *, save_freq: int, run_paths: RunPaths) -> None:
            super().__init__(verbose=0)
            self._save_freq = save_freq
            self._run_paths = run_paths
            self._best_episode_return: float | None = None

        def _save_latest(self) -> None:
            self.model.save(str(self._run_paths.latest_model_path))
            self.model.policy.save(str(self._run_paths.latest_policy_path))

        def _save_best(self, episode_return: float) -> None:
            if (
                self._best_episode_return is not None
                and episode_return <= self._best_episode_return
            ):
                return
            self._best_episode_return = episode_return
            self.model.save(str(self._run_paths.best_model_path))
            self.model.policy.save(str(self._run_paths.best_policy_path))

        def _on_step(self) -> bool:
            if self.n_calls % self._save_freq == 0:
                self._save_latest()

            infos = self.locals.get("infos")
            if not isinstance(infos, list):
                return True

            for info in infos:
                if not isinstance(info, dict):
                    continue
                episode = info.get("episode")
                if not isinstance(episode, dict):
                    continue
                episode_return = episode.get("r")
                if isinstance(episode_return, int | float):
                    self._save_best(float(episode_return))
            return True

    adjusted_save_freq = max(1, train_config.save_freq // train_config.num_envs)
    return CallbackList(
        [
            RollingArtifactCallback(
                save_freq=adjusted_save_freq,
                run_paths=run_paths,
            ),
            InfoLoggingCallback(),
        ]
    )


def _numeric_values(infos: list[object], key: str) -> list[float]:
    values: list[float] = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        value = info.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _cleanup_failed_run(run_paths: RunPaths, model: object) -> None:
    if not run_paths.run_dir.exists():
        return

    num_timesteps = getattr(model, "num_timesteps", None)
    if num_timesteps not in (None, 0):
        return

    shutil.rmtree(run_paths.run_dir, ignore_errors=True)
