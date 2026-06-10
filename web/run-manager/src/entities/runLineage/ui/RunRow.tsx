// web/run-manager/src/entities/runLineage/ui/RunRow.tsx

import { RunActivityIndicator } from "@/entities/run/ui/RunActivityIndicator";
import {
  deleteDisabledReason,
  progressLabel,
  runtimePrimaryLabel,
  runtimeSecondaryLabel,
  statusLabel,
  stepLabel,
} from "@/entities/runLineage/model/lineages";
import type { RunLineageRun } from "@/entities/runLineage/model/types";
import { runLineageMainGridClass, runLineageOuterGridClass } from "@/entities/runLineage/ui/layout";
import { formatDate } from "@/shared/ui/format";
import { BranchSourceIcon, ExportIcon, ResumeIcon, StopIcon, TrashIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface RunRowProps {
  busyActionRunId: string | null;
  entry: RunLineageRun;
  isDeleting: boolean;
  onExportRun: () => Promise<void>;
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
  onExportRun,
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
  const canExport = run.status !== "running" && pendingCommand === null && !busy && !isDeleting;
  const canDelete =
    entry.childCount === 0 &&
    entry.dependentDraftCount === 0 &&
    run.status !== "running" &&
    pendingCommand === null &&
    !busy &&
    !isDeleting;
  const deleteReason = deleteDisabledReason(entry, busy, isDeleting);

  return (
    <div className={`${runLineageOuterGridClass} border-t border-app-border px-3 first:border-t-0`}>
      <button
        aria-label={`Open run ${run.name}`}
        className={`${runLineageMainGridClass} cursor-pointer border-0 bg-transparent p-0 text-left text-app-text`}
        type="button"
        onClick={onOpenRun}
      >
        <span className="grid min-w-0 gap-0.5">
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
                <span className="font-semibold text-app-text">{run.name}</span>
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
        <span className="grid min-w-0 gap-1">
          <div className="flex items-baseline gap-2 max-[760px]:flex-col max-[760px]:items-start">
            <span className="run-record-progress-label">{progressLabel(run)}</span>
          </div>
          <span aria-hidden="true" className="run-record-progress-track">
            <span
              className="run-record-progress-fill"
              style={{ width: `${(run.runtime?.progress_fraction ?? 0) * 100}%` }}
            />
          </span>
          <span className={runSubtleClass}>{stepLabel(run)}</span>
        </span>
        <span className="grid min-w-0 gap-0.5">
          <span className="overflow-hidden text-ellipsis whitespace-nowrap">
            {runtimePrimaryLabel(run)}
          </span>
          <span className={runSubtleClass}>{runtimeSecondaryLabel(run)}</span>
        </span>
        <span className="grid min-w-0 gap-0.5">
          <span className="run-status-chip">{statusLabel(run)}</span>
          <span className={runSubtleClass}>
            <RunActivityIndicator run={run} />
          </span>
        </span>
        <span className="grid min-w-0 gap-0.5">
          <span className={`${runSubtleClass} whitespace-nowrap`}>
            {formatDate(run.created_at)}
          </span>
        </span>
      </button>
      <div className="flex w-[136px] flex-nowrap items-center justify-end gap-1 max-[760px]:justify-start">
        <TooltipIconButton
          aria-label={`Stop run ${run.name}`}
          disabled={!canStop}
          size="compact"
          tooltip={pendingCommand !== null ? `${pendingCommand} requested` : "Stop run"}
          onClick={() => void onRunAction(run.id, onStopRun)}
        >
          <StopIcon />
        </TooltipIconButton>
        <TooltipIconButton
          aria-label={`Resume run ${run.name}`}
          disabled={!canResume}
          size="compact"
          tooltip="Resume run"
          onClick={() => void onRunAction(run.id, onResumeRun)}
        >
          <ResumeIcon />
        </TooltipIconButton>
        <TooltipIconButton
          aria-label={`Export run ${run.name}`}
          disabled={!canExport}
          size="compact"
          tooltip={canExport ? "Export run" : "Stop the run before exporting"}
          onClick={() => void onRunAction(run.id, onExportRun)}
        >
          <ExportIcon />
        </TooltipIconButton>
        <TooltipIconButton
          aria-label={`Delete run ${run.name}`}
          disabled={!canDelete}
          size="compact"
          tone="danger"
          tooltip={deleteReason}
          onClick={onRequestDelete}
        >
          <TrashIcon />
        </TooltipIconButton>
      </div>
    </div>
  );
}

const runSubtleClass = "text-[11px] tabular-nums text-app-muted";
