# tests/apps/test_tensorboard_metrics.py
from __future__ import annotations

import time
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

from rl_fzerox.apps.run_manager.tensorboard_metrics import (
    load_run_metric_samples_from_tensorboard,
)
from rl_fzerox.core.manager import default_managed_run_config
from rl_fzerox.core.manager.models import ManagedRun


def test_tensorboard_metric_loader_reads_scalar_history(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True)

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    _write_scalar_event(
        writer,
        step=128,
        values={
            "rollout/ep_rew_mean": 4.5,
            "state/speed_kph_mean": 812.0,
            "time/fps": 305.0,
        },
    )
    _write_scalar_event(
        writer,
        step=256,
        values={
            "rollout/ep_rew_mean": 4.8,
            "rollout/ep_len_mean": 512.0,
            "train/approx_kl": 0.02,
        },
    )
    writer.flush()
    writer.close()

    run = ManagedRun(
        id="run-001",
        name="ppo_test_1",
        status="running",
        config=default_managed_run_config(),
        config_hash="hash",
        run_dir=run_dir,
        created_at="2026-05-04T00:00:00+00:00",
        lineage_id="run-001",
        lineage_step_offset=512,
    )

    samples = load_run_metric_samples_from_tensorboard(run, limit=10)

    assert len(samples) == 2
    assert samples[0].num_timesteps == 128
    assert samples[0].lineage_num_timesteps == 640
    assert samples[0].episode_reward_mean == pytest.approx(4.5)
    assert samples[0].fps == pytest.approx(305.0)
    assert samples[0].metrics["state/speed_kph_mean"] == pytest.approx(812.0)
    assert samples[1].num_timesteps == 256
    assert samples[1].lineage_num_timesteps == 768
    assert samples[1].episode_reward_mean == pytest.approx(4.8)
    assert samples[1].episode_length_mean == pytest.approx(512.0)
    assert samples[1].approx_kl == pytest.approx(0.02)


def test_tensorboard_metric_loader_supports_full_history_mode(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True)

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    _write_scalar_event(writer, step=128, values={"rollout/ep_rew_mean": 1.0})
    _write_scalar_event(writer, step=256, values={"rollout/ep_rew_mean": 2.0})
    _write_scalar_event(writer, step=512, values={"rollout/ep_rew_mean": 3.0})
    writer.flush()
    writer.close()

    run = ManagedRun(
        id="run-001",
        name="ppo_test_1",
        status="running",
        config=default_managed_run_config(),
        config_hash="hash",
        run_dir=run_dir,
        created_at="2026-05-04T00:00:00+00:00",
        lineage_id="run-001",
        lineage_step_offset=0,
    )

    recent = load_run_metric_samples_from_tensorboard(run, limit=2)
    full = load_run_metric_samples_from_tensorboard(run, limit=None)

    assert [sample.num_timesteps for sample in recent] == [256, 512]
    assert [sample.num_timesteps for sample in full] == [128, 256, 512]


def test_tensorboard_metric_loader_does_not_double_offset_lineage_steps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True)

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    _write_scalar_event(writer, step=836_520, values={"rollout/ep_rew_mean": 1.0})
    _write_scalar_event(writer, step=857_000, values={"rollout/ep_rew_mean": 2.0})
    writer.flush()
    writer.close()

    run = ManagedRun(
        id="run-001",
        name="forked-run",
        status="stopped",
        config=default_managed_run_config(),
        config_hash="hash",
        run_dir=run_dir,
        created_at="2026-05-04T00:00:00+00:00",
        lineage_id="lineage-001",
        lineage_step_offset=816_040,
    )

    samples = load_run_metric_samples_from_tensorboard(run, limit=None)

    assert [sample.num_timesteps for sample in samples] == [20_480, 40_960]
    assert [sample.lineage_num_timesteps for sample in samples] == [836_520, 857_000]


def _write_scalar_event(writer: SummaryWriter, *, step: int, values: dict[str, float]) -> None:
    wall_time = time.time()
    for tag, value in values.items():
        writer.add_scalar(tag, value, global_step=step, walltime=wall_time)
