// web/run-manager/src/widgets/evaluationWorkspace/EvaluationWorkspace.tsx
import { useState } from "react";

import type { ManagedEvaluation, StartEvaluationRequest, WatchDevice } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { FieldSelect } from "@/shared/ui/Field";
import { formatDate } from "@/shared/ui/format";
import { PlayIcon, RenameIcon, StopIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { RenameDialog } from "@/shared/ui/RenameDialog";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

import {
  EVALUATION_MODE_LABELS,
  evaluationProgressStatusLabel,
  evaluationRuntimeLabel,
  evaluationRuntimeStats,
  evaluationSubtitle,
  executionResultLabel,
  formatEta,
  formatFrameRate,
  formatSpeedDetail,
  formatStepCount,
  progressLabel,
  sourceSnapshotLabel,
  statusLabel,
  targetSelectionLabel,
} from "@/widgets/evaluationWorkspace/model";
import { Detail, Metric } from "@/widgets/evaluationWorkspace/parts";
import { ResultsSection } from "@/widgets/evaluationWorkspace/ResultsSection";

interface EvaluationWorkspaceProps {
  evaluation: ManagedEvaluation;
  onCancelEvaluation: (evaluation: ManagedEvaluation) => Promise<ManagedEvaluation>;
  onGlobalError: (message: string | null) => void;
  onRenameEvaluation: (evaluationId: string, name: string) => Promise<void>;
  onStartEvaluation: (
    evaluation: ManagedEvaluation,
    request: StartEvaluationRequest,
  ) => Promise<ManagedEvaluation>;
}

const WORKER_COUNT_OPTIONS = [1, 2, 4, 8, 16] as const;

export function EvaluationWorkspace({
  evaluation,
  onCancelEvaluation,
  onGlobalError,
  onRenameEvaluation,
  onStartEvaluation,
}: EvaluationWorkspaceProps) {
  const [device, setDevice] = useState<WatchDevice>("cuda");
  const [workerCount, setWorkerCount] = useState(1);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const canStart =
    evaluation.status === "created" ||
    evaluation.status === "failed" ||
    evaluation.status === "cancelled";
  const canCancel = evaluation.status === "running";
  const cancelRequested = evaluation.status === "cancelling";
  const runtimeStats = evaluationRuntimeStats(evaluation);

  async function startEvaluation() {
    if (!canStart || starting) {
      return;
    }
    setStarting(true);
    onGlobalError(null);
    try {
      await onStartEvaluation(evaluation, { device, workerCount });
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to start evaluation");
    } finally {
      setStarting(false);
    }
  }

  async function cancelEvaluation() {
    if (!canCancel || cancelling) {
      return;
    }
    setCancelling(true);
    onGlobalError(null);
    try {
      await onCancelEvaluation(evaluation);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to cancel evaluation");
    } finally {
      setCancelling(false);
    }
  }

  async function submitRename(name: string) {
    setRenaming(true);
    onGlobalError(null);
    try {
      await onRenameEvaluation(evaluation.id, name);
      setRenameDialogOpen(false);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to rename evaluation");
    } finally {
      setRenaming(false);
    }
  }

  return (
    <Panel>
      <PanelHeader
        title={
          <span className="inline-flex min-w-0 items-center gap-2">
            <span className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">
              {evaluation.name}
            </span>
            <TooltipIconButton
              aria-label="Rename evaluation"
              disabled={renaming}
              size="small"
              tooltip="Rename"
              onClick={() => setRenameDialogOpen(true)}
            >
              <RenameIcon />
            </TooltipIconButton>
          </span>
        }
        subtitle={evaluationSubtitle(evaluation)}
      />
      <RenameDialog
        busy={renaming}
        initialName={evaluation.name}
        label="Evaluation name"
        open={renameDialogOpen}
        title="Rename evaluation"
        onClose={() => setRenameDialogOpen(false)}
        onSubmit={(name) => void submitRename(name)}
      />

      <div className="grid gap-4">
        <section className="border border-app-border bg-app-surface-muted p-4">
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_auto]">
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-6">
              <Metric label="Status" value={statusLabel(evaluation.status)} />
              <Metric
                label="Checkpoint"
                value={evaluation.checkpoint.source_run_name ?? evaluation.source_run_id ?? "-"}
                detail={`${evaluation.checkpoint.artifact} · ${formatStepCount(
                  evaluation.checkpoint.lineage_num_timesteps,
                )}`}
              />
              <Metric
                label="Target"
                value={`${EVALUATION_MODE_LABELS[evaluation.target.mode]} · ${
                  evaluation.target.repeats_per_target
                }x`}
                detail={targetSelectionLabel(evaluation.target)}
              />
              <Metric label="Seed" value={String(evaluation.seed)} />
              <Metric
                detail={formatSpeedDetail(runtimeStats)}
                label="Speed"
                value={formatFrameRate(runtimeStats?.gameFramesPerSecond ?? null)}
              />
              <Metric label="ETA" value={formatEta(runtimeStats?.etaSeconds ?? null)} />
            </div>

            <div className="flex flex-wrap items-start justify-end gap-2">
              {canCancel || cancelRequested ? (
                <Button
                  className="gap-2"
                  disabled={cancelling || cancelRequested}
                  tone="danger"
                  onClick={() => void cancelEvaluation()}
                >
                  <StopIcon />
                  <span>{cancelling || cancelRequested ? "Cancelling" : "Cancel"}</span>
                </Button>
              ) : (
                <>
                  <FieldSelect
                    aria-label="Evaluation runtime device"
                    className="min-w-[110px]"
                    disabled={!canStart || starting}
                    value={device}
                    onChange={(event) => setDevice(event.currentTarget.value as WatchDevice)}
                  >
                    <option value="cuda">cuda</option>
                    <option value="cpu">cpu</option>
                  </FieldSelect>
                  <FieldSelect
                    aria-label="Evaluation worker count"
                    className="min-w-[120px]"
                    disabled={!canStart || starting}
                    value={String(workerCount)}
                    onChange={(event) => setWorkerCount(Number(event.currentTarget.value))}
                  >
                    {WORKER_COUNT_OPTIONS.map((count) => (
                      <option key={count} value={count}>
                        {count} {count === 1 ? "worker" : "workers"}
                      </option>
                    ))}
                  </FieldSelect>
                  <Button
                    className="gap-2"
                    disabled={!canStart || starting}
                    variant={canStart ? "primary" : undefined}
                    onClick={() => void startEvaluation()}
                  >
                    <PlayIcon />
                    <span>
                      {starting ? "Starting" : evaluation.status === "created" ? "Start" : "Retry"}
                    </span>
                  </Button>
                </>
              )}
            </div>
          </div>

          <div className="mt-4">
            <ProgressBar evaluation={evaluation} />
          </div>
        </section>

        {evaluation.error_message !== null ? (
          <Notice tone="error">{evaluation.error_message}</Notice>
        ) : null}

        <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="border border-app-border bg-app-surface p-4">
            <h3 className="m-0 text-base font-semibold text-app-text">Checkpoint</h3>
            <dl className="mt-4 grid gap-3 text-sm">
              <Detail label="Evaluation id" value={evaluation.id} />
              <Detail
                label="Source run"
                value={evaluation.checkpoint.source_run_name ?? evaluation.source_run_id ?? "-"}
              />
              <Detail label="Artifact" value={evaluation.checkpoint.artifact} />
              <Detail
                label="Source steps"
                value={formatStepCount(evaluation.checkpoint.lineage_num_timesteps)}
              />
              <Detail label="Policy mode" value={evaluation.policy_mode} />
              <Detail label="Renderer" value={evaluation.config.environment.renderer} />
              <Detail label="Snapshot time" value={sourceSnapshotLabel(evaluation)} />
            </dl>
          </div>

          <div className="border border-app-border bg-app-surface p-4">
            <h3 className="m-0 text-base font-semibold text-app-text">Execution</h3>
            <dl className="mt-4 grid gap-3 text-sm">
              <Detail label="Created" value={formatDate(evaluation.created_at)} />
              <Detail label="Runtime" value={evaluationRuntimeLabel(evaluation)} />
              <Detail
                label="Started"
                value={evaluation.started_at === null ? "-" : formatDate(evaluation.started_at)}
              />
              <Detail
                label="Finished"
                value={evaluation.finished_at === null ? "-" : formatDate(evaluation.finished_at)}
              />
              <Detail label="Result" value={executionResultLabel(evaluation)} />
            </dl>
          </div>
        </section>

        <ResultsSection evaluation={evaluation} />
      </div>
    </Panel>
  );
}

function ProgressBar({ evaluation }: { evaluation: ManagedEvaluation }) {
  const { completed_attempts: completed, total_attempts: total } = evaluation.progress;
  const percent = total === null || total <= 0 ? null : Math.min(100, (completed / total) * 100);
  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-4 text-sm text-app-muted">
        <span>{progressLabel(evaluation)}</span>
        <span>{evaluationProgressStatusLabel(evaluation)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-sm bg-app-surface">
        <div
          className="h-full bg-app-accent"
          style={{ width: `${percent ?? (completed > 0 ? 100 : 0)}%` }}
        />
      </div>
    </div>
  );
}
