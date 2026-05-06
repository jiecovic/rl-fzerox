// src/rl_fzerox/apps/run_manager/web/src/features/runs/RunActivityIndicator.tsx
import { latestFailureMessage } from "@/features/runs/workspace/model";
import type { ManagedRun } from "@/shared/api/contract";
import { formatRelativeTime } from "@/shared/ui/format";

export function RunActivityIndicator({ run }: { run: ManagedRun }) {
  if (run.pending_command !== null) {
    return <span>{run.pending_command} requested</span>;
  }
  if (isRunHeartbeatFresh(run)) {
    return (
      <span className="run-live-indicator">
        <span aria-hidden="true" className="run-live-indicator-dot" />
        <span>live</span>
      </span>
    );
  }
  if (run.runtime !== null) {
    return <span>updated {formatRelativeTime(run.runtime.updated_at)}</span>;
  }
  if (run.status === "running") {
    return <span>{startupActivityLabel(run) ?? "starting"}</span>;
  }
  if (run.status === "failed") {
    return <span>{latestFailureMessage(run) ?? startupActivityLabel(run) ?? "failed"}</span>;
  }
  return <span>idle</span>;
}

export function isRunHeartbeatFresh(run: ManagedRun, now: Date = new Date()) {
  if (run.status !== "running" || run.runtime === null) {
    return false;
  }
  const updatedAt = new Date(run.runtime.updated_at);
  if (Number.isNaN(updatedAt.getTime())) {
    return false;
  }
  return now.getTime() - updatedAt.getTime() <= 8_000;
}

function startupActivityLabel(run: ManagedRun) {
  const startupEvent = run.recent_events.find((event) => event.kind.startsWith("startup_"));
  if (startupEvent === undefined) {
    return null;
  }
  return startupEvent.message;
}
