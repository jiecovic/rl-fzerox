# src/rl_fzerox/apps/run_manager/tensorboard_metrics.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rl_fzerox.core.manager.models import ManagedRun, ManagedRunMetricSample
from rl_fzerox.core.training.runs import RUN_LAYOUT

if TYPE_CHECKING:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass(slots=True)
class _ScalarSampleBuilder:
    step: int
    wall_time: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class _TensorboardMetricCacheEntry:
    accumulator: EventAccumulator
    signature: tuple[tuple[str, int, int], ...]
    samples: tuple[ManagedRunMetricSample, ...]


_RUN_METRIC_CACHE: dict[Path, _TensorboardMetricCacheEntry] = {}


def load_run_metric_samples_from_tensorboard(
    run: ManagedRun,
    *,
    limit: int | None,
) -> tuple[ManagedRunMetricSample, ...]:
    """Load sampled scalar history for one run from its TensorBoard event files."""

    event_dir = run.run_dir / RUN_LAYOUT.tensorboard_dirname
    if not event_dir.is_dir():
        return ()

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return ()

    signature = _event_file_signature(event_dir)
    if not signature:
        return ()

    cache_key = event_dir.resolve()
    cached = _RUN_METRIC_CACHE.get(cache_key)
    if cached is not None and cached.signature == signature:
        return _slice_samples(cached.samples, limit)

    accumulator = (
        cached.accumulator
        if cached is not None
        else EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    )
    try:
        accumulator.Reload()
    except Exception:
        return ()

    scalar_tags = accumulator.Tags().get("scalars", ())
    if not scalar_tags:
        return ()

    by_step: dict[int, _ScalarSampleBuilder] = {}
    for tag in scalar_tags:
        for scalar in accumulator.Scalars(tag):
            builder = by_step.get(int(scalar.step))
            if builder is None:
                builder = _ScalarSampleBuilder(
                    step=int(scalar.step),
                    wall_time=float(scalar.wall_time),
                )
                by_step[builder.step] = builder
            elif scalar.wall_time > builder.wall_time:
                builder.wall_time = float(scalar.wall_time)
            builder.metrics[str(tag)] = float(scalar.value)

    if not by_step:
        return ()

    total_timesteps = max(1, run.config.train.total_timesteps)
    ordered = sorted(by_step.values(), key=lambda item: item.step)
    uses_lineage_steps = (
        run.lineage_step_offset > 0
        and len(ordered) > 0
        and ordered[0].step >= run.lineage_step_offset
    )
    samples = tuple(
        ManagedRunMetricSample(
            run_id=run.id,
            created_at=_isoformat_utc(sample.wall_time),
            total_timesteps=total_timesteps,
            num_timesteps=_local_num_timesteps(
                sample.step,
                lineage_step_offset=run.lineage_step_offset,
                uses_lineage_steps=uses_lineage_steps,
            ),
            lineage_num_timesteps=_lineage_num_timesteps(
                sample.step,
                lineage_step_offset=run.lineage_step_offset,
                uses_lineage_steps=uses_lineage_steps,
            ),
            progress_fraction=min(
                1.0,
                max(
                    0.0,
                    _local_num_timesteps(
                        sample.step,
                        lineage_step_offset=run.lineage_step_offset,
                        uses_lineage_steps=uses_lineage_steps,
                    )
                    / total_timesteps,
                ),
            ),
            metrics=sample.metrics,
            fps=sample.metrics.get("time/fps"),
            episode_reward_mean=sample.metrics.get("rollout/ep_rew_mean"),
            episode_length_mean=sample.metrics.get("rollout/ep_len_mean"),
            approx_kl=sample.metrics.get("train/approx_kl"),
            entropy_loss=sample.metrics.get("train/entropy_loss"),
            value_loss=sample.metrics.get("train/value_loss"),
            policy_gradient_loss=sample.metrics.get("train/policy_gradient_loss"),
        )
        for sample in ordered
    )
    _RUN_METRIC_CACHE[cache_key] = _TensorboardMetricCacheEntry(
        accumulator=accumulator,
        signature=signature,
        samples=samples,
    )
    return _slice_samples(samples, limit)


def _slice_samples(
    samples: tuple[ManagedRunMetricSample, ...],
    limit: int | None,
) -> tuple[ManagedRunMetricSample, ...]:
    if limit is None:
        return samples
    return samples[-max(1, limit) :]


def _event_file_signature(event_dir: Path) -> tuple[tuple[str, int, int], ...]:
    files: list[tuple[str, int, int]] = []
    for path in sorted(event_dir.glob("*")):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(files)


def _isoformat_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat(timespec="seconds")


def _local_num_timesteps(
    sample_step: int,
    *,
    lineage_step_offset: int,
    uses_lineage_steps: bool,
) -> int:
    if not uses_lineage_steps:
        return sample_step
    return max(0, sample_step - lineage_step_offset)


def _lineage_num_timesteps(
    sample_step: int,
    *,
    lineage_step_offset: int,
    uses_lineage_steps: bool,
) -> int:
    if uses_lineage_steps:
        return sample_step
    return lineage_step_offset + sample_step
