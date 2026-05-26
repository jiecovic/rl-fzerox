// src/rl_fzerox/apps/run_manager/web/src/features/runs/workspace/RuntimeSummary.tsx
import type { ReactNode } from "react";

import { RunActivityIndicator } from "@/features/runs/RunActivityIndicator";
import { RunTrackPoolPanel } from "@/features/runs/RunTrackPoolPanel";
import type { RunWorkspaceActionState } from "@/features/runs/workspace/actions";
import {
  envStepRateLabel,
  lineageSimGameTimeLabel,
  lineageSimToWallRatioLabel,
  lineageStepsLabel,
  lineageWallTimeLabel,
  localSimGameTimeLabel,
  localSimToWallRatioLabel,
  localWallTimeLabel,
  progressHeadline,
  progressNote,
  showsLineageTotals,
  timeLeftLabel,
  trainFpsLabel,
} from "@/features/runs/workspace/model";
import { useRunClock } from "@/features/runs/workspace/polling";
import type {
  ConfigMetadata,
  ManagedRun,
  ManagedRunDetail,
  TrackSamplingRuntimeState,
} from "@/shared/api/contract";
import { rendererNames } from "@/shared/api/renderers";
import { IconButton } from "@/shared/ui/Button";
import { cn } from "@/shared/ui/cn";
import { FieldSelect } from "@/shared/ui/Field";
import { formatDate } from "@/shared/ui/format";
import {
  ChartIcon,
  CopyIcon,
  FolderIcon,
  ForkIcon,
  ResumeIcon,
  SaveDraftIcon,
  StopIcon,
  WatchIcon,
} from "@/shared/ui/icons";

interface RunRuntimeSummaryProps {
  actions: RunWorkspaceActionState;
  allRuns: ManagedRun[];
  metadata: ConfigMetadata;
  onShowCharts: (runId: string) => void;
  run: ManagedRunDetail;
  trackSamplingState: TrackSamplingRuntimeState | null;
}

export function RunRuntimeSummary({
  actions,
  allRuns,
  metadata,
  onShowCharts,
  run,
  trackSamplingState,
}: RunRuntimeSummaryProps) {
  const nowMs = useRunClock(run.status);
  const runtime = run.runtime;
  const progressFraction = runtime?.progress_fraction ?? 0;
  const hasLineageTotals = showsLineageTotals(run);
  const watchRendererOptions = rendererNames(metadata, actions.selectedWatchRenderer);

  return (
    <div className="mb-[18px] grid grid-cols-1 items-start gap-3.5">
      <div className="grid gap-2">
        <div className="flex items-center justify-between gap-3 text-[13px]">
          <strong>Training progress</strong>
          <span>{progressHeadline(run)}</span>
        </div>
        <div
          aria-hidden="true"
          className="h-3 overflow-hidden border border-app-border bg-app-surface-muted"
        >
          <div
            className="h-full bg-[linear-gradient(90deg,var(--accent)_0%,color-mix(in_srgb,var(--accent)_72%,white)_100%)]"
            style={{ width: `${progressFraction * 100}%` }}
          />
        </div>
        <div
          className={cn(
            "text-xs tabular-nums text-app-muted",
            run.status === "failed" ? "text-app-danger" : undefined,
          )}
        >
          {progressNote(run)}
        </div>
        <div className="grid grid-cols-[repeat(auto-fit,minmax(168px,1fr))] gap-2.5">
          <RunMetric label="Env step / s">{envStepRateLabel(run)}</RunMetric>
          <RunMetric label="Train fps">{trainFpsLabel(run)}</RunMetric>
          <RunMetric label="Time left">{timeLeftLabel(run)}</RunMetric>
          {hasLineageTotals ? (
            <>
              <RunMetric label={run.runtime === null ? "Setup time · local" : "Wall time · local"}>
                {localWallTimeLabel(run, nowMs)}
              </RunMetric>
              <RunMetric label={run.runtime === null ? "Setup time · total" : "Wall time · total"}>
                {lineageWallTimeLabel(run, allRuns, nowMs)}
              </RunMetric>
              <RunMetric label="Sim game time · local">{localSimGameTimeLabel(run)}</RunMetric>
              <RunMetric label="Sim game time · total">
                {lineageSimGameTimeLabel(run, allRuns)}
              </RunMetric>
              <RunMetric label="Sim / wall · local">
                {localSimToWallRatioLabel(run, nowMs)}
              </RunMetric>
              <RunMetric label="Sim / wall · total">
                {lineageSimToWallRatioLabel(run, allRuns, nowMs)}
              </RunMetric>
            </>
          ) : (
            <>
              <RunMetric label={run.runtime === null ? "Setup time" : "Wall time"}>
                {localWallTimeLabel(run, nowMs)}
              </RunMetric>
              <RunMetric label="Sim game time">{localSimGameTimeLabel(run)}</RunMetric>
              <RunMetric label="Sim / wall">{localSimToWallRatioLabel(run, nowMs)}</RunMetric>
            </>
          )}
          {run.lineage_step_offset > 0 ? (
            <RunMetric label="Lineage steps">{lineageStepsLabel(run)}</RunMetric>
          ) : null}
          <RunMetric label="Run id">
            <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2">
              <code className="font-mono text-[13px] leading-normal break-words whitespace-normal text-app-text">
                {run.id}
              </code>
              <IconButton
                aria-label={actions.copiedRunId ? "Run id copied" : "Copy run id"}
                className="tooltip-anchor justify-self-end"
                data-tooltip={actions.copiedRunId ? "Copied" : "Copy run id"}
                data-tooltip-position="top"
                size="compact"
                onClick={() => void actions.copyRunId()}
              >
                <CopyIcon />
              </IconButton>
            </div>
          </RunMetric>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-start gap-2">
        <IconButton
          aria-label={actions.isOpeningDirectory ? "Opening run folder" : "Open run folder"}
          title={actions.isOpeningDirectory ? "Opening..." : "Open folder"}
          disabled={actions.isOpeningDirectory}
          onClick={() => void actions.openRunDirectoryInBrowser()}
        >
          <FolderIcon />
        </IconButton>
        <div className="run-watch-control relative inline-grid auto-cols-max grid-flow-col items-center gap-1.5">
          {actions.watchToast !== null ? (
            <div
              aria-live="polite"
              className={watchToastClass(actions.watchToast.tone)}
              role={actions.watchToast.tone === "error" ? "alert" : "status"}
            >
              {actions.watchToast.message}
            </div>
          ) : null}
          <div className="relative inline-flex items-center">
            <span className="sr-only">Checkpoint artifact</span>
            <FieldSelect
              aria-label="Checkpoint artifact"
              className="h-10 min-w-[82px] px-2.5 pr-[30px] text-xs lowercase hover:border-app-border-strong hover:bg-app-surface-muted"
              value={actions.selectedArtifact}
              disabled={actions.watchingArtifact !== null || actions.isForking}
              onChange={(event) =>
                actions.setSelectedArtifact(event.target.value === "best" ? "best" : "latest")
              }
            >
              <option value="latest">latest</option>
              <option value="best">best</option>
            </FieldSelect>
          </div>
          <div className="relative inline-flex items-center">
            <span className="sr-only">Policy device</span>
            <FieldSelect
              aria-label="Policy device"
              className="h-10 min-w-[82px] px-2.5 pr-[30px] text-xs lowercase hover:border-app-border-strong hover:bg-app-surface-muted"
              value={actions.selectedWatchDevice}
              disabled={actions.watchingArtifact !== null}
              onChange={(event) =>
                actions.setSelectedWatchDevice(event.target.value === "cpu" ? "cpu" : "cuda")
              }
            >
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
            </FieldSelect>
          </div>
          <div className="relative inline-flex items-center">
            <span className="sr-only">Renderer</span>
            <FieldSelect
              aria-label="Watch renderer"
              className="h-10 min-w-[82px] px-2.5 pr-[30px] text-xs lowercase hover:border-app-border-strong hover:bg-app-surface-muted"
              value={actions.selectedWatchRenderer}
              disabled={actions.watchingArtifact !== null}
              onChange={(event) =>
                actions.setSelectedWatchRenderer(
                  watchRendererOptions.find((renderer) => renderer === event.target.value) ??
                    run.config.environment.renderer,
                )
              }
            >
              {watchRendererOptions.map((renderer) => (
                <option key={renderer} value={renderer}>
                  {renderer === run.config.environment.renderer
                    ? `${renderer} (training)`
                    : renderer}
                </option>
              ))}
            </FieldSelect>
          </div>
          <IconButton
            aria-label={
              actions.watchingArtifact === actions.selectedArtifact
                ? `Opening ${actions.selectedArtifact} checkpoint watch`
                : `Watch ${actions.selectedArtifact} checkpoint`
            }
            title={
              actions.watchingArtifact === actions.selectedArtifact
                ? `Opening ${actions.selectedArtifact}...`
                : `Watch ${actions.selectedArtifact}`
            }
            disabled={actions.watchingArtifact !== null}
            onClick={() => void actions.watchRunArtifact(actions.selectedArtifact)}
          >
            <WatchIcon />
          </IconButton>
          <IconButton
            aria-label={
              actions.isForking
                ? `Forking ${actions.selectedArtifact} checkpoint`
                : `Fork ${actions.selectedArtifact} checkpoint`
            }
            title={
              actions.isForking
                ? `Forking ${actions.selectedArtifact}...`
                : `Fork ${actions.selectedArtifact}`
            }
            disabled={actions.isForking}
            onClick={() => void actions.forkRunArtifact(actions.selectedArtifact)}
          >
            <ForkIcon />
          </IconButton>
        </div>
        <IconButton
          aria-label={
            actions.isCreatingDraftFromRun
              ? "Creating draft from run"
              : "Create editable draft from run"
          }
          title={actions.isCreatingDraftFromRun ? "Creating draft..." : "Create editable draft"}
          disabled={actions.isCreatingDraftFromRun}
          onClick={() => void actions.createDraftFromRun()}
        >
          <SaveDraftIcon />
        </IconButton>
        <IconButton
          aria-label="Show run charts"
          title="Charts"
          onClick={() => onShowCharts(run.id)}
        >
          <ChartIcon />
        </IconButton>
        <IconButton
          aria-label={
            actions.isStopping
              ? "Stopping run"
              : run.pending_command === "stop"
                ? "Stop requested"
                : "Stop run"
          }
          title={
            actions.isStopping
              ? "Stopping..."
              : run.pending_command === "stop"
                ? "Stop requested"
                : "Stop"
          }
          disabled={!actions.canStop}
          onClick={() => void actions.stopRun()}
        >
          <StopIcon />
        </IconButton>
        <IconButton
          aria-label={actions.isResuming ? "Resuming run" : "Resume run"}
          title={actions.isResuming ? "Resuming..." : "Resume"}
          disabled={!actions.canResume}
          onClick={() => void actions.resumeRun()}
        >
          <ResumeIcon />
        </IconButton>
      </div>

      <RunTrackPoolPanel
        canReset={run.status === "stopped"}
        isResetting={actions.isResettingTrackPool}
        metadata={metadata}
        onReset={() => void actions.resetTrackPoolState()}
        run={run}
        state={trackSamplingState}
      />
    </div>
  );
}

function RunMetric({ children, label }: { children: ReactNode; label: ReactNode }) {
  return (
    <div className="run-runtime-metric grid gap-1 border border-app-border bg-app-surface-muted px-3 py-2.5">
      <span className="text-[11px] tracking-[0.05em] text-app-muted uppercase">{label}</span>
      <div className="grid gap-1.5 [overflow-wrap:anywhere] text-[15px] leading-normal whitespace-normal tabular-nums text-app-text">
        {children}
      </div>
    </div>
  );
}

function watchToastClass(tone: "error" | "info") {
  return cn(
    "absolute bottom-[calc(100%+8px)] left-0 z-[2] w-max max-w-[min(44ch,calc(100vw-48px))] [overflow-wrap:anywhere] border border-app-border-strong bg-[color-mix(in_srgb,var(--surface)_94%,var(--accent)_6%)] px-2.5 py-2 text-xs leading-normal whitespace-normal text-app-text shadow-[var(--shadow-md)]",
    tone === "error"
      ? "border-[var(--danger-border)] bg-[color-mix(in_srgb,var(--surface)_90%,var(--danger)_10%)]"
      : undefined,
  );
}

export function runWorkspaceSubtitle(run: ManagedRunDetail) {
  return (
    <>
      <span>{run.status}</span>
      <span aria-hidden="true"> · </span>
      <span>created {formatDate(run.created_at)}</span>
      {run.status !== "failed" ? (
        <>
          <span aria-hidden="true"> · </span>
          <RunActivityIndicator run={run} />
        </>
      ) : null}
    </>
  );
}
