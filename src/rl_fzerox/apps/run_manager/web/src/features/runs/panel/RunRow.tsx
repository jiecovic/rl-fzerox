import {
  deleteDisabledReason,
  progressLabel,
  runtimePrimaryLabel,
  runtimeSecondaryLabel,
  statusLabel,
  stepLabel,
} from "@/features/runs/panel/model";
import type { RunLineageRun } from "@/features/runs/panel/types";
import { RunActivityIndicator } from "@/features/runs/RunActivityIndicator";
import { formatDate } from "@/shared/ui/format";
import { BranchSourceIcon, ResumeIcon, StopIcon, TrashIcon } from "@/shared/ui/icons";

interface RunRowProps {
  busyActionRunId: string | null;
  entry: RunLineageRun;
  isDeleting: boolean;
  onOpenRun: () => void;
  onRequestDelete: () => void;
  onResumeRun: () => Promise<void>;
  onRunAction: (runId: string, callback: () => Promise<void>) => Promise<void>;
  onStopRun: () => Promise<void>;
}

export function RunRow({
  busyActionRunId,
  entry,
  isDeleting,
  onOpenRun,
  onRequestDelete,
  onResumeRun,
  onRunAction,
  onStopRun,
}: RunRowProps) {
  const { run } = entry;
  const pendingCommand = run.pending_command;
  const busy = busyActionRunId === run.id;
  const canStop = run.status === "running" && pendingCommand === null && !busy && !isDeleting;
  const canResume =
    (run.status === "paused" || run.status === "stopped" || run.status === "failed") &&
    !busy &&
    !isDeleting;
  const canDelete =
    entry.childCount === 0 &&
    entry.dependentDraftCount === 0 &&
    run.status !== "running" &&
    pendingCommand === null &&
    !busy &&
    !isDeleting;
  const deleteReason = deleteDisabledReason(entry, busy, isDeleting);

  return (
    <div className="run-list-row run-lineage-row">
      <button
        aria-label={`Open run ${run.name}`}
        className="run-list-main"
        type="button"
        onClick={onOpenRun}
      >
        <span className="run-list-cell run-list-cell-name">
          <span className="run-branch-name">
            <span
              aria-hidden="true"
              className={
                entry.isRoot
                  ? "run-branch-graph is-root"
                  : entry.childCount === 0
                    ? "run-branch-graph is-leaf"
                    : "run-branch-graph"
              }
            >
              <span className="run-branch-line run-branch-line-top" />
              <span className="run-branch-node">{entry.isRoot ? "R" : entry.depth}</span>
              <span className="run-branch-line run-branch-line-bottom" />
            </span>
            <span className="run-branch-copy">
              <span className="run-branch-header">
                <span className="record-name">{run.name}</span>
                <span
                  className={
                    entry.isRoot ? "run-branch-stage run-branch-stage-root" : "run-branch-stage"
                  }
                >
                  {entry.stageLabel}
                </span>
              </span>
              {entry.source.kind === "root" ? (
                <span className="run-branch-source-row run-branch-source-root">
                  <span className="run-branch-source-kicker">origin</span>
                  <span className="run-branch-source-text">lineage root</span>
                </span>
              ) : (
                <span className="run-branch-source-row">
                  <span aria-hidden="true" className="run-branch-source-icon">
                    <BranchSourceIcon />
                  </span>
                  <span className="run-branch-source-kicker">forked from</span>
                  <span className="run-branch-source-text">{entry.source.parentName}</span>
                  {entry.source.artifactLabel !== null ? (
                    <span className="run-branch-source-chip">{entry.source.artifactLabel}</span>
                  ) : null}
                  {entry.source.stepLabel !== null ? (
                    <span className="run-branch-source-chip">{entry.source.stepLabel}</span>
                  ) : null}
                </span>
              )}
            </span>
          </span>
        </span>
        <span className="run-list-cell run-list-progress">
          <div className="run-list-progress-top">
            <span className="run-record-progress-label">{progressLabel(run)}</span>
          </div>
          <span aria-hidden="true" className="run-record-progress-track">
            <span
              className="run-record-progress-fill"
              style={{ width: `${(run.runtime?.progress_fraction ?? 0) * 100}%` }}
            />
          </span>
          <span className="run-record-subtle">{stepLabel(run)}</span>
        </span>
        <span className="run-list-cell run-list-live">
          <span>{runtimePrimaryLabel(run)}</span>
          <span className="run-record-subtle">{runtimeSecondaryLabel(run)}</span>
        </span>
        <span className="run-list-cell run-list-status">
          <span className="run-status-chip">{statusLabel(run)}</span>
          <span className="run-record-subtle">
            <RunActivityIndicator run={run} />
          </span>
        </span>
        <span className="run-list-cell run-list-created">
          <span className="run-record-subtle run-created-at">{formatDate(run.created_at)}</span>
        </span>
      </button>
      <div className="run-list-actions">
        <button
          aria-label={`Stop run ${run.name}`}
          className="icon-button compact-icon-button run-list-action-button"
          title={pendingCommand !== null ? `${pendingCommand} requested` : "Stop run"}
          type="button"
          disabled={!canStop}
          onClick={() => void onRunAction(run.id, onStopRun)}
        >
          <StopIcon />
        </button>
        <button
          aria-label={`Resume run ${run.name}`}
          className="icon-button compact-icon-button run-list-action-button"
          title="Resume run"
          type="button"
          disabled={!canResume}
          onClick={() => void onRunAction(run.id, onResumeRun)}
        >
          <ResumeIcon />
        </button>
        <button
          aria-label={`Delete run ${run.name}`}
          className="icon-button compact-icon-button danger run-list-action-button"
          title={deleteReason}
          type="button"
          disabled={!canDelete}
          onClick={onRequestDelete}
        >
          <TrashIcon />
        </button>
      </div>
    </div>
  );
}
