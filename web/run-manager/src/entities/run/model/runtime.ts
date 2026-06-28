// web/run-manager/src/entities/run/model/runtime.ts
import type { ManagedRun, ManagedRunDetail } from "@/shared/api/contract";
import { formatLongDurationSeconds } from "@/shared/ui/format";

export function isPinnedRun(run: ManagedRun): boolean {
  return run.status === "running";
}

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
    if (run.status === "failed") {
      return (
        failureMessage ?? `Run failed before the first callback flush. Target was ${target} steps.`
      );
    }
    return `Target ${target} steps. Runtime metrics appear after the first callback flush.`;
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

export function timeLeftLabel(run: ManagedRun): string {
  const seconds = timeLeftSeconds(run);
  return seconds === null ? "n/a" : formatLongDurationSeconds(seconds);
}

export function localWallTimeLabel(run: ManagedRun, nowMs: number): string {
  const seconds = localWallTimeSeconds(run, nowMs);
  return seconds === null ? "n/a" : formatLongDurationSeconds(seconds);
}

export function lineageWallTimeLabel(
  run: ManagedRun,
  allRuns: ManagedRun[],
  nowMs: number,
): string {
  const seconds = lineageWallTimeSeconds(run, allRuns, nowMs);
  return seconds === null ? "n/a" : formatLongDurationSeconds(seconds);
}

export function localSimGameTimeLabel(run: ManagedRun): string {
  const seconds = localSimGameTimeSeconds(run);
  return seconds === null ? "n/a" : formatLongDurationSeconds(seconds);
}

export function lineageSimGameTimeLabel(run: ManagedRun, allRuns: ManagedRun[]): string {
  const seconds = lineageSimGameTimeSeconds(run, allRuns);
  return seconds === null ? "n/a" : formatLongDurationSeconds(seconds);
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
  if (!lineageWallHistoryCoversStepOffset(run, allRuns)) {
    return "n/a";
  }
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

export function latestActiveStartupMessage(run: ManagedRun): string | null {
  const startupEvent = latestStartupEvent(run);
  if (startupEvent === null) {
    return null;
  }
  const failureEvent = latestFailureEvent(run);
  if (
    failureEvent !== null &&
    eventTimestampIsAtOrAfter(failureEvent.created_at, startupEvent.created_at)
  ) {
    return null;
  }
  if (run.runtime === null) {
    return startupEvent.message;
  }
  const startupTime = Date.parse(startupEvent.created_at);
  const runtimeTime = Date.parse(run.runtime.updated_at);
  if (Number.isNaN(startupTime) || Number.isNaN(runtimeTime)) {
    return null;
  }
  return startupTime > runtimeTime ? startupEvent.message : null;
}

function latestStartupEvent(run: ManagedRun): { created_at: string; message: string } | null {
  const startupEvent = run.recent_events.find((event) => event.kind.startsWith("startup_"));
  if (startupEvent === undefined) {
    return null;
  }
  return {
    created_at: startupEvent.created_at,
    message: startupEvent.message,
  };
}

export function latestFailureMessage(run: ManagedRun): string | null {
  return latestFailureEvent(run)?.message ?? null;
}

function latestFailureEvent(run: ManagedRun): { created_at: string; message: string } | null {
  const failedEvent = run.recent_events.find((event) => event.kind === "failed");
  if (failedEvent === undefined) {
    return null;
  }
  return {
    created_at: failedEvent.created_at,
    message: failedEvent.message,
  };
}

function eventTimestampIsAtOrAfter(candidate: string, baseline: string): boolean {
  const candidateTime = Date.parse(candidate);
  const baselineTime = Date.parse(baseline);
  if (Number.isNaN(candidateTime) || Number.isNaN(baselineTime)) {
    return false;
  }
  return candidateTime >= baselineTime;
}

function formatRate(value: number): string {
  return value >= 100 ? `${value.toFixed(0)}` : `${value.toFixed(1)}`;
}

function localSimGameTimeSeconds(run: ManagedRun): number | null {
  const frames = localExperienceFrames(run);
  return frames === null ? null : frames / 60;
}

function lineageSimGameTimeSeconds(run: ManagedRun, allRuns: ManagedRun[]): number | null {
  const frames = lineageExperienceFrames(run, allRuns);
  return frames === null ? null : frames / 60;
}

function localExperienceFrames(run: ManagedRun): number | null {
  const runtime = run.runtime;
  if (runtime === null) {
    return null;
  }
  return Math.max(0, runtime.num_timesteps) * runActionRepeat(run);
}

function lineageExperienceFrames(run: ManagedRun, allRuns: ManagedRun[]): number | null {
  const localFrames = localExperienceFrames(run);
  const lineageOffsetFrames = lineageFrameOffset(run, allRuns);
  if (localFrames === null && lineageOffsetFrames === null) {
    return null;
  }
  return (lineageOffsetFrames ?? 0) + (localFrames ?? 0);
}

function lineageFrameOffset(run: ManagedRun, allRuns: ManagedRun[]): number | null {
  if (run.lineage_step_offset <= 0) {
    return 0;
  }

  // Match watch experience semantics for normal forks. Published checkpoint
  // roots intentionally cut timing history: they carry a trained snapshot, not
  // trustworthy wall-time data from the original training machine.
  const runsById = new Map(allRuns.map((candidate) => [candidate.id, candidate]));
  const visited = new Set<string>();
  let totalFrames = 0;
  let currentRun: ManagedRun | null = run;

  while (currentRun !== null && !visited.has(currentRun.id)) {
    visited.add(currentRun.id);
    const parentRun: ManagedRun | null =
      currentRun.parent_run_id === null ? null : (runsById.get(currentRun.parent_run_id) ?? null);
    if (parentRun === null) {
      totalFrames += Math.max(0, currentRun.lineage_step_offset) * runActionRepeat(currentRun);
      return totalFrames;
    }
    if (isPublishedCheckpointRoot(parentRun)) {
      return totalFrames;
    }

    const sourceSteps =
      currentRun.source_num_timesteps ??
      currentRun.lineage_step_offset - parentRun.lineage_step_offset;
    if (sourceSteps < 0) {
      return null;
    }
    totalFrames += sourceSteps * runActionRepeat(parentRun);
    currentRun = parentRun;
  }

  return totalFrames;
}

function lineageWallHistoryCoversStepOffset(run: ManagedRun, allRuns: ManagedRun[]): boolean {
  if (run.lineage_step_offset <= 0) {
    return true;
  }
  const runsById = new Map(allRuns.map((candidate) => [candidate.id, candidate]));
  const visited = new Set<string>();
  let currentRun: ManagedRun | null = run;
  while (currentRun !== null && !visited.has(currentRun.id)) {
    visited.add(currentRun.id);
    if (currentRun.lineage_step_offset <= 0) {
      return true;
    }
    if (currentRun.parent_run_id === null) {
      return false;
    }
    currentRun = runsById.get(currentRun.parent_run_id) ?? null;
  }
  return false;
}

function runActionRepeat(run: ManagedRun): number {
  return Math.max(run.action_repeat, 1);
}

function isPublishedCheckpointRoot(run: ManagedRun): boolean {
  return (
    run.status === "archived" && run.parent_run_id === null && run.source_num_timesteps !== null
  );
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

function timeLeftSeconds(run: ManagedRun): number | null {
  const runtime = run.runtime;
  if (runtime === null) {
    return null;
  }
  const envStepRate = envStepRateValue(run);
  if (envStepRate === null || envStepRate <= 0) {
    return null;
  }
  const remainingSteps = Math.max(0, runtime.total_timesteps - runtime.num_timesteps);
  return remainingSteps / envStepRate;
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
    if (currentRun.id !== run.id && isPublishedCheckpointRoot(currentRun)) {
      break;
    }
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
