// web/run-manager/src/widgets/evaluationWorkspace/EvaluationWorkspace.tsx
import { useState } from "react";

import type { ManagedEvaluation } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { formatDate } from "@/shared/ui/format";
import { EvaluationTabIcon, PlayIcon, RenameIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { RenameDialog } from "@/shared/ui/RenameDialog";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface EvaluationWorkspaceProps {
  evaluation: ManagedEvaluation;
  onGlobalError: (message: string | null) => void;
  onRenameEvaluation: (evaluationId: string, name: string) => Promise<void>;
  onStartEvaluation: (evaluation: ManagedEvaluation) => Promise<ManagedEvaluation>;
}

const EVALUATION_MODE_LABELS = {
  gp_course: "GP course",
  time_attack_course: "Time Attack course",
} satisfies Record<ManagedEvaluation["target"]["mode"], string>;

export function EvaluationWorkspace({
  evaluation,
  onGlobalError,
  onRenameEvaluation,
  onStartEvaluation,
}: EvaluationWorkspaceProps) {
  const [starting, setStarting] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const canStart = evaluation.status === "created";

  async function startEvaluation() {
    if (!canStart || starting) {
      return;
    }
    setStarting(true);
    onGlobalError(null);
    try {
      await onStartEvaluation(evaluation);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to start evaluation");
    } finally {
      setStarting(false);
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
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
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
            </div>

            <div className="flex items-start justify-end">
              <Button
                className="gap-2"
                disabled={!canStart || starting}
                variant={canStart ? "primary" : undefined}
                onClick={() => void startEvaluation()}
              >
                <PlayIcon />
                <span>{starting ? "Starting" : "Start"}</span>
              </Button>
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
              <Detail label="Device" value="cpu" />
              <Detail label="Renderer" value={evaluation.config.environment.renderer} />
              <Detail label="Snapshot time" value={sourceSnapshotLabel(evaluation)} />
            </dl>
          </div>

          <div className="border border-app-border bg-app-surface p-4">
            <h3 className="m-0 text-base font-semibold text-app-text">Execution</h3>
            <dl className="mt-4 grid gap-3 text-sm">
              <Detail label="Created" value={formatDate(evaluation.created_at)} />
              <Detail
                label="Started"
                value={evaluation.started_at === null ? "-" : formatDate(evaluation.started_at)}
              />
              <Detail
                label="Finished"
                value={evaluation.finished_at === null ? "-" : formatDate(evaluation.finished_at)}
              />
              <Detail
                label="Result"
                value={evaluation.result_json_path === null ? "pending" : "written"}
              />
            </dl>
          </div>
        </section>

        <section className="border border-app-border bg-app-surface p-4">
          <div className="flex items-center gap-2 text-sm font-bold tracking-[0.04em] text-app-muted uppercase">
            <EvaluationTabIcon />
            <span>Results</span>
          </div>
          <p className="mt-3 mb-0 text-sm text-app-muted">
            {evaluation.result_json_path === null
              ? "No results yet."
              : "Result summary is available."}
          </p>
        </section>
      </div>
    </Panel>
  );
}

function Metric({ detail, label, value }: { detail?: string; label: string; value: string }) {
  return (
    <div className="border border-app-border bg-app-surface p-3">
      <div className="text-xs font-bold tracking-[0.04em] text-app-muted uppercase">{label}</div>
      <div className="mt-2 text-lg font-semibold text-app-text">{value}</div>
      {detail === undefined ? null : <div className="mt-1 text-xs text-app-muted">{detail}</div>}
    </div>
  );
}

function Detail({ label, mono = false, value }: { label: string; mono?: boolean; value: string }) {
  return (
    <div className="grid gap-1">
      <dt className="text-xs font-bold tracking-[0.04em] text-app-muted uppercase">{label}</dt>
      <dd className={mono ? "m-0 break-all font-mono text-app-muted" : "m-0 text-app-muted"}>
        {value}
      </dd>
    </div>
  );
}

function ProgressBar({ evaluation }: { evaluation: ManagedEvaluation }) {
  const { completed_attempts: completed, total_attempts: total } = evaluation.progress;
  const percent = total === null || total <= 0 ? null : Math.min(100, (completed / total) * 100);
  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-4 text-sm text-app-muted">
        <span>{progressLabel(evaluation)}</span>
        <span>{evaluation.progress.result_status ?? evaluation.status}</span>
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

function evaluationSubtitle(evaluation: ManagedEvaluation) {
  return `${EVALUATION_MODE_LABELS[evaluation.target.mode]} · ${
    evaluation.checkpoint.source_run_name ?? evaluation.source_run_id ?? "unknown run"
  } · ${evaluation.policy_mode}`;
}

function progressLabel(evaluation: ManagedEvaluation) {
  const { completed_attempts: completed, total_attempts: total } = evaluation.progress;
  if (total !== null && total > 0) {
    return `${completed.toLocaleString()} / ${total.toLocaleString()} attempts`;
  }
  return completed > 0 ? `${completed.toLocaleString()} attempts` : "not started";
}

function targetSelectionLabel(target: ManagedEvaluation["target"]) {
  const parts = [
    selectionCountLabel(target.cup_ids, "cup"),
    selectionCountLabel(target.course_ids, "course"),
    selectionCountLabel(target.difficulties, "difficulty"),
    selectionCountLabel(target.vehicle_ids, "vehicle"),
  ].filter((part) => part !== null);
  return parts.length === 0 ? "all targets" : parts.join(" · ");
}

function selectionCountLabel(values: readonly string[], singular: string) {
  if (values.length === 0) {
    return null;
  }
  return `${values.length} ${pluralize(values.length, singular)}`;
}

function pluralize(count: number, singular: string) {
  if (count === 1) {
    return singular;
  }
  return singular === "difficulty" ? "difficulties" : `${singular}s`;
}

function formatStepCount(value: number | null) {
  return value === null ? "step unknown" : `${value.toLocaleString()} steps`;
}

function statusLabel(status: ManagedEvaluation["status"]) {
  return status.slice(0, 1).toUpperCase() + status.slice(1);
}

function sourceSnapshotLabel(evaluation: ManagedEvaluation) {
  const sourceMtimeNs = evaluation.checkpoint.source_mtime_ns;
  if (sourceMtimeNs === null) {
    return "unknown";
  }
  try {
    const epochMs = Number(BigInt(sourceMtimeNs) / 1_000_000n);
    return formatDate(new Date(epochMs).toISOString());
  } catch {
    return "unknown";
  }
}
