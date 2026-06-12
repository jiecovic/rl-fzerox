# src/rl_fzerox/apps/run_manager/training_monitor.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic

from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.manager.models import RunCommand
from rl_fzerox.core.training.runs import RunPaths
from rl_fzerox.core.training.session import (
    current_engine_tuning_checkpoint_state,
    current_policy_artifact_metadata,
    save_artifacts_atomically,
)


@dataclass(frozen=True, slots=True)
class RunMetricSnapshot:
    total_timesteps: int
    num_timesteps: int
    progress_fraction: float
    fps: float | None
    episode_reward_mean: float | None
    episode_length_mean: float | None
    approx_kl: float | None
    entropy_loss: float | None
    value_loss: float | None
    policy_gradient_loss: float | None


class RunControlSignal(RuntimeError):
    """Raised when the manager requests a controlled training stop."""

    command: RunCommand

    def __init__(self, command: RunCommand) -> None:
        self.command = command
        super().__init__(f"manager requested {command}")


def build_manager_training_callback(
    *,
    store: ManagerStore,
    run_id: str,
    launch_token: str,
    run_paths: RunPaths,
    total_timesteps: int,
    lineage_step_offset: int = 0,
):
    """Construct one SB3 callback that publishes manager runtime state."""

    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for training. "
            "Install with `python -m pip install -e .[train]`."
        ) from exc

    class ManagerTrainingCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._last_runtime_flush = 0.0
            self._last_command_poll = 0.0
            self._training_started_at_monotonic: float | None = None
            self._training_started_num_timesteps = 0

        def _on_training_start(self) -> None:
            self._training_started_at_monotonic = monotonic()
            initial_num_timesteps = getattr(self.model, "num_timesteps", 0)
            self._training_started_num_timesteps = (
                initial_num_timesteps if isinstance(initial_num_timesteps, int) else 0
            )
            _heartbeat_or_raise(store=store, run_id=run_id, launch_token=launch_token)
            self._flush_runtime()

        def _on_step(self) -> bool:
            now = monotonic()
            if now - self._last_command_poll >= 1.0:
                self._last_command_poll = now
                _heartbeat_or_raise(store=store, run_id=run_id, launch_token=launch_token)
                pending_command = store.pending_run_command(run_id)
                if pending_command is not None:
                    self._write_resume_checkpoint()
                    self._flush_runtime()
                    raise RunControlSignal(pending_command)
            if _target_reached(self.model, total_timesteps=total_timesteps):
                self._flush_runtime()
                self._last_runtime_flush = now
                return False
            if now - self._last_runtime_flush >= 2.0:
                self._flush_runtime()
                self._last_runtime_flush = now
            return True

        def _on_rollout_end(self) -> None:
            self._flush_runtime()
            self._last_runtime_flush = monotonic()

        def _flush_runtime(self) -> None:
            snapshot = self._snapshot()
            updated_at = _utc_now()
            _heartbeat_or_raise(
                store=store,
                run_id=run_id,
                launch_token=launch_token,
                heartbeat_at=updated_at,
            )
            store.upsert_run_runtime(
                run_id=run_id,
                total_timesteps=snapshot.total_timesteps,
                num_timesteps=snapshot.num_timesteps,
                progress_fraction=snapshot.progress_fraction,
                updated_at=updated_at,
                fps=snapshot.fps,
                episode_reward_mean=snapshot.episode_reward_mean,
                episode_length_mean=snapshot.episode_length_mean,
                approx_kl=snapshot.approx_kl,
                entropy_loss=snapshot.entropy_loss,
                value_loss=snapshot.value_loss,
                policy_gradient_loss=snapshot.policy_gradient_loss,
            )

        def _snapshot(self) -> RunMetricSnapshot:
            num_timesteps = getattr(self.model, "num_timesteps", 0)
            if not isinstance(num_timesteps, int):
                num_timesteps = 0
            logger_values = getattr(self.logger, "name_to_value", None)
            values = logger_values if isinstance(logger_values, dict) else {}
            progress_fraction = min(1.0, max(0.0, num_timesteps / max(1, total_timesteps)))
            fps = _metric_value(values, "time/fps")
            if fps is None:
                fps = _estimated_env_step_rate(
                    started_at_monotonic=self._training_started_at_monotonic,
                    started_num_timesteps=self._training_started_num_timesteps,
                    current_num_timesteps=num_timesteps,
                    current_monotonic=monotonic(),
                )
            return RunMetricSnapshot(
                total_timesteps=total_timesteps,
                num_timesteps=num_timesteps,
                progress_fraction=progress_fraction,
                fps=fps,
                episode_reward_mean=_metric_value(values, "rollout/ep_rew_mean"),
                episode_length_mean=_metric_value(values, "rollout/ep_len_mean"),
                approx_kl=_metric_value(values, "train/approx_kl"),
                entropy_loss=_metric_value(values, "train/entropy_loss"),
                value_loss=_metric_value(values, "train/value_loss"),
                policy_gradient_loss=_metric_value(values, "train/policy_gradient_loss"),
            )

        def _write_resume_checkpoint(self) -> None:
            save_artifacts_atomically(
                model=self.model,
                model_path=run_paths.latest_model_path,
                policy_path=run_paths.latest_policy_path,
                engine_tuning_state=current_engine_tuning_checkpoint_state(
                    self.training_env
                ),
                policy_metadata=current_policy_artifact_metadata(
                    self.training_env,
                    self.model,
                    lineage_step_offset=lineage_step_offset,
                ),
            )

    return ManagerTrainingCallback()


def _metric_value(values: dict[object, object], key: str) -> float | None:
    value = values.get(key)
    return float(value) if isinstance(value, int | float) else None


def _estimated_env_step_rate(
    *,
    started_at_monotonic: float | None,
    started_num_timesteps: int,
    current_num_timesteps: int,
    current_monotonic: float,
) -> float | None:
    if started_at_monotonic is None:
        return None
    elapsed = current_monotonic - started_at_monotonic
    if elapsed <= 0.0:
        return None
    progressed_steps = max(0, current_num_timesteps - started_num_timesteps)
    return progressed_steps / elapsed


def _target_reached(model: object, *, total_timesteps: int) -> bool:
    num_timesteps = getattr(model, "num_timesteps", 0)
    return (
        isinstance(num_timesteps, int)
        and not isinstance(num_timesteps, bool)
        and num_timesteps >= total_timesteps
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _heartbeat_or_raise(
    *,
    store: ManagerStore,
    run_id: str,
    launch_token: str,
    heartbeat_at: str | None = None,
) -> None:
    lease_ok = store.heartbeat_run_worker(
        run_id=run_id,
        launch_token=launch_token,
        heartbeat_at=heartbeat_at or _utc_now(),
    )
    if not lease_ok:
        raise RuntimeError("manager worker lease lost")
