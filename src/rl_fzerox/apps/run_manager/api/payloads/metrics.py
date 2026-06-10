# src/rl_fzerox/apps/run_manager/api/payloads/metrics.py
from __future__ import annotations

from rl_fzerox.core.manager import ManagedRunMetricSample
from rl_fzerox.core.manager.artifacts.tensorboard_views import TensorboardViewGroup


def tensorboard_view_group_payload(group: TensorboardViewGroup) -> dict[str, object]:
    return {
        "name": group.name,
        "slug": group.slug,
        "path": str(group.path),
        "lineage_count": group.lineage_count,
        "run_count": group.run_count,
    }


def run_metric_payload(sample: ManagedRunMetricSample) -> dict[str, object]:
    return {
        "run_id": sample.run_id,
        "created_at": sample.created_at,
        "total_timesteps": sample.total_timesteps,
        "num_timesteps": sample.num_timesteps,
        "lineage_num_timesteps": sample.lineage_num_timesteps,
        "progress_fraction": sample.progress_fraction,
        "metrics": sample.metrics,
        "fps": sample.fps,
        "episode_reward_mean": sample.episode_reward_mean,
        "episode_length_mean": sample.episode_length_mean,
        "approx_kl": sample.approx_kl,
        "entropy_loss": sample.entropy_loss,
        "value_loss": sample.value_loss,
        "policy_gradient_loss": sample.policy_gradient_loss,
    }
