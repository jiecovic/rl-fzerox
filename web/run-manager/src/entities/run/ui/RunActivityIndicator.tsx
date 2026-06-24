// web/run-manager/src/entities/run/ui/RunActivityIndicator.tsx

import { latestActiveStartupMessage, latestFailureMessage } from "@/entities/run/model/runtime";
import type { ManagedRun } from "@/shared/api/contract";
import { formatRelativeTime } from "@/shared/ui/format";

const RUN_HEARTBEAT_FRESH_MS = 8_000;

export function RunActivityIndicator({ run }: { run: ManagedRun }) {
  if (run.pending_command !== null) {
    return <span>{run.pending_command} requested</span>;
  }
  const activeStartupMessage = latestActiveStartupMessage(run);
  if (isRunHeartbeatFresh(run)) {
    return (
      <span className="run-live-indicator">
        <span aria-hidden="true" className="run-live-indicator-dot" />
        <span>live</span>
        {activeStartupMessage !== null ? <span> · {activeStartupMessage}</span> : null}
        {run.runtime !== null ? (
          <span> · last metrics {formatRelativeTime(run.runtime.updated_at)}</span>
        ) : null}
      </span>
    );
  }
  if (run.status === "running") {
    const staleHeartbeatLabel = heartbeatAgeLabel(run);
    const staleMetricsLabel = metricsAgeLabel(run);
    if (
      activeStartupMessage !== null ||
      staleHeartbeatLabel !== null ||
      staleMetricsLabel !== null
    ) {
      return (
        <span>
          {activeStartupMessage ?? staleHeartbeatLabel ?? staleMetricsLabel}
          {activeStartupMessage !== null && staleHeartbeatLabel !== null ? (
            <span> · {staleHeartbeatLabel}</span>
          ) : null}
          {activeStartupMessage !== null &&
          staleHeartbeatLabel === null &&
          staleMetricsLabel !== null ? (
            <span> · {staleMetricsLabel}</span>
          ) : null}
          {staleHeartbeatLabel !== null && staleMetricsLabel !== null ? (
            <span> · {staleMetricsLabel}</span>
          ) : null}
        </span>
      );
    }
  }
  if (run.runtime !== null) {
    return <span>last metrics {formatRelativeTime(run.runtime.updated_at)}</span>;
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
  if (run.status !== "running" || run.worker_heartbeat_at === null) {
    return false;
  }
  const heartbeatAt = new Date(run.worker_heartbeat_at);
  if (Number.isNaN(heartbeatAt.getTime())) {
    return false;
  }
  return now.getTime() - heartbeatAt.getTime() <= RUN_HEARTBEAT_FRESH_MS;
}

function startupActivityLabel(run: ManagedRun) {
  const startupEvent = run.recent_events.find((event) => event.kind.startsWith("startup_"));
  if (startupEvent === undefined) {
    return null;
  }
  return startupEvent.message;
}

function heartbeatAgeLabel(run: ManagedRun) {
  if (run.worker_heartbeat_at === null) {
    return null;
  }
  return `last heartbeat ${formatRelativeTime(run.worker_heartbeat_at)}`;
}

function metricsAgeLabel(run: ManagedRun) {
  if (run.runtime === null) {
    return null;
  }
  return `last metrics ${formatRelativeTime(run.runtime.updated_at)}`;
}
