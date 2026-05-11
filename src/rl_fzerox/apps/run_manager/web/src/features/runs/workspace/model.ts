// src/rl_fzerox/apps/run_manager/web/src/features/runs/workspace/model.ts
import type { ManagedRun, ManagedRunDetail } from "@/shared/api/contract";

export function progressHeadline(run: ManagedRun): string {
  const runtime = run.runtime;
  if (runtime === null) {
    return run.status === "failed" ? "No runtime samples" : "Waiting for first sample";
  }
  return run.lineage_step_offset > 0
    ? `${(runtime.progress_fraction * 100).toFixed(1)}% of this fork`
    : `${(runtime.progress_fraction * 100).toFixed(1)}% complete`;
}

export function progressNote(run: ManagedRunDetail): string {
  const runtime = run.runtime;
  const target = run.config.train.total_timesteps.toLocaleString();
  if (runtime === null) {
    const failureMessage = latestFailureMessage(run);
    const startupMessage = latestStartupMessage(run);
    if (run.status === "failed") {
      return (
        failureMessage ??
        startupMessage ??
        `Run failed before the first callback flush. Target was ${target} steps.`
      );
    }
    return (
      startupMessage ??
      `Target ${target} steps. Runtime metrics appear after the first callback flush.`
    );
  }
  if (run.lineage_step_offset <= 0) {
    return `${runtime.num_timesteps.toLocaleString()} / ${runtime.total_timesteps.toLocaleString()} steps`;
  }
  const lineageSteps = run.lineage_step_offset + runtime.num_timesteps;
  return `${lineageSteps.toLocaleString()} lineage steps · ${runtime.num_timesteps.toLocaleString()} / ${runtime.total_timesteps.toLocaleString()} local fork steps`;
}

export function showsLineageTotals(run: ManagedRun): boolean {
  return run.lineage_step_offset > 0 || run.parent_run_id !== null;
}

export function envStepRateLabel(run: ManagedRun): string {
  const fps = envStepRateValue(run);
  return fps === null ? "n/a" : formatRate(fps);
}

export function trainFpsLabel(run: ManagedRun): string {
  const envStepRate = envStepRateValue(run);
  if (envStepRate === null) {
    return "n/a";
  }
  return formatRate(envStepRate * Math.max(run.action_repeat, 1));
}

export function localWallTimeLabel(run: ManagedRun, nowMs: number): string {
  const seconds = localWallTimeSeconds(run, nowMs);
  return seconds === null ? "n/a" : formatDurationSeconds(seconds);
}

export function lineageWallTimeLabel(
  run: ManagedRun,
  allRuns: ManagedRun[],
  nowMs: number,
): string {
  const seconds = lineageWallTimeSeconds(run, allRuns, nowMs);
  return seconds === null ? "n/a" : formatDurationSeconds(seconds);
}

export function localSimGameTimeLabel(run: ManagedRun): string {
  const seconds = localSimGameTimeSeconds(run);
  return seconds === null ? "n/a" : formatDurationSeconds(seconds);
}

export function lineageSimGameTimeLabel(run: ManagedRun, allRuns: ManagedRun[]): string {
  const seconds = lineageSimGameTimeSeconds(run, allRuns);
  return seconds === null ? "n/a" : formatDurationSeconds(seconds);
}

export function localSimToWallRatioLabel(run: ManagedRun, nowMs: number): string {
  const localWallSeconds = localWallTimeSeconds(run, nowMs);
  const localSimSeconds = localSimGameTimeSeconds(run);
  if (localWallSeconds === null || localSimSeconds === null || localWallSeconds <= 0) {
    return "n/a";
  }
  return `${(localSimSeconds / localWallSeconds).toFixed(2)}x`;
}

export function lineageSimToWallRatioLabel(
  run: ManagedRun,
  allRuns: ManagedRun[],
  nowMs: number,
): string {
  const lineageWallSeconds = lineageWallTimeSeconds(run, allRuns, nowMs);
  const lineageSimSeconds = lineageSimGameTimeSeconds(run, allRuns);
  if (lineageWallSeconds === null || lineageSimSeconds === null || lineageWallSeconds <= 0) {
    return "n/a";
  }
  return `${(lineageSimSeconds / lineageWallSeconds).toFixed(2)}x`;
}

export function lineageStepsLabel(run: ManagedRun): string {
  const runtime = run.runtime;
  if (runtime !== null) {
    return (run.lineage_step_offset + runtime.num_timesteps).toLocaleString();
  }
  if (run.lineage_step_offset > 0) {
    return run.lineage_step_offset.toLocaleString();
  }
  if (run.source_num_timesteps !== null) {
    return run.source_num_timesteps.toLocaleString();
  }
  return "n/a";
}

export function latestStartupMessage(run: ManagedRun): string | null {
  const startupEvent = run.recent_events.find((event) => event.kind.startsWith("startup_"));
  if (startupEvent === undefined) {
    return null;
  }
  return startupEvent.message;
}

export function latestFailureMessage(run: ManagedRun): string | null {
  const failedEvent = run.recent_events.find((event) => event.kind === "failed");
  if (failedEvent === undefined) {
    return null;
  }
  return failedEvent.message;
}

export function formatDurationSeconds(value: number): string {
  const totalSeconds = Math.max(0, Math.floor(value));
  const durationUnits = [
    { label: "y", seconds: 365 * 24 * 3600 },
    { label: "mo", seconds: 30 * 24 * 3600 },
    { label: "d", seconds: 24 * 3600 },
    { label: "h", seconds: 3600 },
    { label: "m", seconds: 60 },
    { label: "s", seconds: 1 },
  ] as const;

  let remainingSeconds = totalSeconds;
  const parts: string[] = [];
  for (const unit of durationUnits) {
    if (remainingSeconds < unit.seconds && parts.length === 0 && unit.label !== "s") {
      continue;
    }
    const amount = Math.floor(remainingSeconds / unit.seconds);
    remainingSeconds -= amount * unit.seconds;
    if (amount > 0 || unit.label === "s") {
      parts.push(`${amount}${unit.label}`);
    }
    if (parts.length >= 3) {
      break;
    }
  }
  return parts.join(" ");
}

function formatRate(value: number): string {
  return value >= 100 ? `${value.toFixed(0)}` : `${value.toFixed(1)}`;
}

function localSimGameTimeSeconds(run: ManagedRun): number | null {
  const runtime = run.runtime;
  if (runtime === null) {
    return null;
  }
  const actionRepeat = Math.max(run.action_repeat, 1);
  return (runtime.num_timesteps * actionRepeat) / 60;
}

function lineageSimGameTimeSeconds(run: ManagedRun, allRuns: ManagedRun[]): number | null {
  return lineageAggregateSeconds(run, allRuns, localSimGameTimeSeconds);
}

function envStepRateValue(run: ManagedRun): number | null {
  const runtime = run.runtime;
  if (runtime === null) {
    return null;
  }
  if (runtime.fps !== null && runtime.fps !== undefined) {
    return runtime.fps;
  }
  const startedAt = run.started_at ?? run.created_at;
  const startedMs = Date.parse(startedAt);
  const updatedMs = Date.parse(runtime.updated_at);
  if (Number.isNaN(startedMs) || Number.isNaN(updatedMs) || updatedMs <= startedMs) {
    return null;
  }
  return runtime.num_timesteps / ((updatedMs - startedMs) / 1000);
}

function localWallTimeSeconds(run: ManagedRun, nowMs: number): number | null {
  const activeRuntimeWallSeconds = activeRuntimeWallTimeSeconds(run);
  if (activeRuntimeWallSeconds !== null) {
    return activeRuntimeWallSeconds;
  }
  const startedAt = run.started_at ?? run.created_at;
  const startedMs = Date.parse(startedAt);
  if (Number.isNaN(startedMs)) {
    return null;
  }
  const stoppedMs = run.stopped_at === null ? nowMs : Date.parse(run.stopped_at);
  if (Number.isNaN(stoppedMs)) {
    return null;
  }
  return Math.max(0, (stoppedMs - startedMs) / 1000);
}

function activeRuntimeWallTimeSeconds(run: ManagedRun): number | null {
  const runtime = run.runtime;
  if (runtime === null) {
    return null;
  }
  const envStepRate = envStepRateValue(run);
  if (envStepRate === null || envStepRate <= 0) {
    return null;
  }
  return runtime.num_timesteps / envStepRate;
}

function lineageWallTimeSeconds(
  run: ManagedRun,
  allRuns: ManagedRun[],
  nowMs: number,
): number | null {
  return lineageAggregateSeconds(run, allRuns, (currentRun) =>
    localWallTimeSeconds(currentRun, nowMs),
  );
}

function lineageAggregateSeconds(
  run: ManagedRun,
  allRuns: ManagedRun[],
  selector: (run: ManagedRun) => number | null,
): number | null {
  const runsById = new Map(allRuns.map((candidate) => [candidate.id, candidate]));
  const visited = new Set<string>();
  let totalSeconds = 0;
  let hasAnyValue = false;
  let currentRun: ManagedRun | null = run;

  while (currentRun !== null && !visited.has(currentRun.id)) {
    visited.add(currentRun.id);
    const seconds = selector(currentRun);
    if (seconds !== null) {
      totalSeconds += seconds;
      hasAnyValue = true;
    }
    currentRun =
      currentRun.parent_run_id === null ? null : (runsById.get(currentRun.parent_run_id) ?? null);
  }

  return hasAnyValue ? totalSeconds : null;
}
