// web/run-manager/src/widgets/runWorkspace/workspace/RuntimeSummary.tsx
import type { ReactNode } from "react";
import { RunEngineTuningPanel } from "@/entities/engineTuning/ui/RunEngineTuningPanel";
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
} from "@/entities/run/model/runtime";
import { RunActivityIndicator } from "@/entities/run/ui/RunActivityIndicator";
import { RunTrackPoolPanel } from "@/entities/trackPool/ui/RunTrackPoolPanel";
import { useRunClock } from "@/features/runLiveData/hooks";
import type { RunWorkspaceActionState } from "@/features/runWorkspaceActions/useRunWorkspaceActions";
import type {
  ConfigMetadata,
  EngineTuningRuntimeState,
  ManagedRun,
  ManagedRunDetail,
  TrackSamplingRuntimeState,
} from "@/shared/api/contract";
import { rendererNames } from "@/shared/api/renderers";
import { cn } from "@/shared/ui/cn";
import { FieldSelect } from "@/shared/ui/Field";
import { formatDate } from "@/shared/ui/format";
import {
  ChartIcon,
  CopyIcon,
  EvaluationTabIcon,
  FolderIcon,
  ForkIcon,
  ResumeIcon,
  SaveDraftIcon,
  StopIcon,
  WatchIcon,
} from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface RunRuntimeSummaryProps {
  actions: RunWorkspaceActionState;
  allRuns: ManagedRun[];
  canResetEngineTuning: boolean;
  metadata: ConfigMetadata;
  onCreateEvaluation: () => void;
  onShowCharts: (runId: string) => void;
  run: ManagedRunDetail;
  engineTuningExpanded: boolean;
  engineTuningState: EngineTuningRuntimeState | null;
  isResettingEngineTuning: boolean;
  onEngineTuningExpandedChange: (expanded: boolean) => void;
  onResetEngineTuning: () => void;
  trackSamplingState: TrackSamplingRuntimeState | null;
}

export function RunRuntimeSummary({
  actions,
  allRuns,
  canResetEngineTuning,
  metadata,
  onCreateEvaluation,
  onShowCharts,
  run,
  engineTuningExpanded,
  engineTuningState,
  isResettingEngineTuning,
  onEngineTuningExpandedChange,
  onResetEngineTuning,
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
          <RunMetric label="Seed">{run.config.seed}</RunMetric>
          <RunMetric label="Run id">
            <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2">
              <code className="font-mono text-[13px] leading-normal break-words whitespace-normal text-app-text">
                {run.id}
              </code>
              <TooltipIconButton
                aria-label={actions.copiedRunId ? "Run id copied" : "Copy run id"}
                className="justify-self-end"
                size="compact"
                tooltip={actions.copiedRunId ? "Copied" : "Copy run id"}
                onClick={() => void actions.copyRunId()}
              >
                <CopyIcon />
              </TooltipIconButton>
            </div>
          </RunMetric>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2.5 xl:grid-cols-[minmax(0,auto)_minmax(0,1fr)]">
        <fieldset className={toolbarPanelClass()}>
          <legend className={toolbarPanelTitleClass()}>Run control</legend>
          <div className="flex min-w-0 flex-wrap items-end gap-2">
            <div className="inline-flex min-w-0 flex-wrap items-center gap-1.5">
              <TooltipIconButton
                aria-label={actions.isOpeningDirectory ? "Opening run folder" : "Open run folder"}
                disabled={actions.isOpeningDirectory}
                tooltip={actions.isOpeningDirectory ? "Opening..." : "Open folder"}
                onClick={() => void actions.openRunDirectoryInBrowser()}
              >
                <FolderIcon />
              </TooltipIconButton>
              <TooltipIconButton
                aria-label={
                  actions.isCreatingDraftFromRun
                    ? "Creating draft from run"
                    : "Create editable draft from run"
                }
                disabled={actions.isCreatingDraftFromRun}
                tooltip={
                  actions.isCreatingDraftFromRun ? "Creating draft..." : "Create editable draft"
                }
                onClick={() => void actions.createDraftFromRun()}
              >
                <SaveDraftIcon />
              </TooltipIconButton>
              <TooltipIconButton
                aria-label="Show run charts"
                tooltip="Charts"
                onClick={() => onShowCharts(run.id)}
              >
                <ChartIcon />
              </TooltipIconButton>
              <TooltipIconButton
                aria-label="Create evaluation from run"
                tooltip="Evaluate"
                onClick={onCreateEvaluation}
              >
                <EvaluationTabIcon />
              </TooltipIconButton>
              <TooltipIconButton
                aria-label={
                  actions.isStopping
                    ? "Stopping run"
                    : run.pending_command === "stop"
                      ? "Stop requested"
                      : "Stop run"
                }
                disabled={!actions.canStop}
                tooltip={
                  actions.isStopping
                    ? "Stopping..."
                    : run.pending_command === "stop"
                      ? "Stop requested"
                      : "Stop"
                }
                onClick={() => void actions.stopRun()}
              >
                <StopIcon />
              </TooltipIconButton>
              <TooltipIconButton
                aria-label={actions.isResuming ? "Resuming run" : "Resume run"}
                disabled={!actions.canResume}
                tooltip={actions.isResuming ? "Resuming..." : "Resume latest checkpoint"}
                onClick={() => void actions.resumeRun()}
              >
                <ResumeIcon />
              </TooltipIconButton>
            </div>

            <div aria-hidden="true" className="hidden h-10 w-px bg-app-border sm:block" />
            <ToolbarSelect label="Fork checkpoint">
              <FieldSelect
                aria-label="Fork checkpoint artifact"
                className={toolbarSelectClass("!w-[104px]")}
                value={actions.selectedForkArtifact}
                disabled={actions.isForking}
                onChange={(event) =>
                  actions.setSelectedForkArtifact(event.target.value === "best" ? "best" : "latest")
                }
              >
                <option value="latest">latest</option>
                <option value="best">best</option>
              </FieldSelect>
            </ToolbarSelect>
            <TooltipIconButton
              aria-label={
                actions.isForking
                  ? `Forking ${actions.selectedForkArtifact} checkpoint`
                  : `Fork ${actions.selectedForkArtifact} checkpoint`
              }
              disabled={actions.isForking}
              tooltip={
                actions.isForking
                  ? `Forking ${actions.selectedForkArtifact}...`
                  : `Fork ${actions.selectedForkArtifact}`
              }
              onClick={() => void actions.forkRunArtifact(actions.selectedForkArtifact)}
            >
              <ForkIcon />
            </TooltipIconButton>
          </div>
        </fieldset>

        <fieldset className={toolbarPanelClass()}>
          <legend className={toolbarPanelTitleClass()}>Watch launch</legend>
          <div className="flex min-w-0 flex-wrap items-end gap-2">
            <ToolbarSelect label="Checkpoint">
              <FieldSelect
                aria-label="Watch checkpoint artifact"
                className={toolbarSelectClass("!w-[104px]")}
                value={actions.selectedWatchArtifact}
                disabled={actions.watchingArtifact !== null}
                onChange={(event) =>
                  actions.setSelectedWatchArtifact(
                    event.target.value === "best" ? "best" : "latest",
                  )
                }
              >
                <option value="latest">latest</option>
                <option value="best">best</option>
              </FieldSelect>
            </ToolbarSelect>
            <ToolbarSelect label="Policy mode">
              <FieldSelect
                aria-label="Watch policy mode"
                className={toolbarSelectClass("!w-[142px]")}
                value={actions.selectedWatchPolicyMode}
                disabled={actions.watchingArtifact !== null}
                onChange={(event) =>
                  actions.setSelectedWatchPolicyMode(
                    event.target.value === "stochastic" ? "stochastic" : "deterministic",
                  )
                }
              >
                <option value="deterministic">deterministic</option>
                <option value="stochastic">stochastic</option>
              </FieldSelect>
            </ToolbarSelect>
            <ToolbarSelect label="Device">
              <FieldSelect
                aria-label="Watch policy device"
                className={toolbarSelectClass("!w-[94px]")}
                value={actions.selectedWatchDevice}
                disabled={actions.watchingArtifact !== null}
                onChange={(event) =>
                  actions.setSelectedWatchDevice(event.target.value === "cpu" ? "cpu" : "cuda")
                }
              >
                <option value="cuda">cuda</option>
                <option value="cpu">cpu</option>
              </FieldSelect>
            </ToolbarSelect>
            <ToolbarSelect label="Renderer">
              <FieldSelect
                aria-label="Watch renderer"
                className={toolbarSelectClass("!w-[196px]")}
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
            </ToolbarSelect>
            <TooltipIconButton
              aria-label="Save watch launch settings"
              disabled={actions.watchingArtifact !== null || actions.watchLaunchSettingsSaved}
              tooltip={
                actions.watchLaunchSettingsSaved
                  ? "Watch launch settings saved"
                  : "Save watch launch settings"
              }
              onClick={actions.saveWatchLaunchSettings}
            >
              <SaveDraftIcon />
            </TooltipIconButton>
            <TooltipIconButton
              aria-label={
                actions.watchingArtifact === actions.selectedWatchArtifact
                  ? `Opening ${actions.selectedWatchArtifact} checkpoint watch`
                  : `Watch ${actions.selectedWatchArtifact} checkpoint`
              }
              disabled={actions.watchingArtifact !== null}
              tooltip={
                actions.watchingArtifact === actions.selectedWatchArtifact
                  ? `Opening ${actions.selectedWatchArtifact}...`
                  : `Watch ${actions.selectedWatchArtifact}`
              }
              onClick={() => void actions.watchRunArtifact(actions.selectedWatchArtifact)}
            >
              <WatchIcon />
            </TooltipIconButton>
          </div>
        </fieldset>
      </div>

      <RunTrackPoolPanel
        canReset={run.status === "stopped"}
        clearingAltBaselineCourseKey={actions.clearingAltBaselineCourseKey}
        isClearingAltBaselines={actions.isClearingAltBaselines}
        isResetting={actions.isResettingTrackPool}
        metadata={metadata}
        onClearAltBaselines={actions.clearAltBaselines}
        onClearCourseAltBaselines={actions.clearCourseAltBaselines}
        onReset={() => void actions.resetTrackPoolState()}
        run={run}
        state={trackSamplingState}
      />
      <RunEngineTuningPanel
        artifact={actions.selectedWatchArtifact}
        canReset={canResetEngineTuning}
        enabled={run.config.vehicle.engine_mode === "adaptive_tuner"}
        expanded={engineTuningExpanded}
        isResetting={isResettingEngineTuning}
        metadata={metadata}
        state={engineTuningState}
        onExpandedChange={onEngineTuningExpandedChange}
        onReset={onResetEngineTuning}
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

function ToolbarSelect({ children, label }: { children: ReactNode; label: string }) {
  return (
    <div className="grid min-w-0 gap-1">
      <span className={toolbarLegendClass()}>{label}</span>
      {children}
    </div>
  );
}

function toolbarPanelClass() {
  return "m-0 grid min-w-0 gap-2 border border-app-border bg-app-surface-muted p-2";
}

function toolbarPanelTitleClass() {
  return "px-1 text-[11px] font-semibold tracking-[0.08em] text-app-muted uppercase";
}

function toolbarLegendClass() {
  return "text-[10px] tracking-[0.08em] text-app-muted uppercase";
}

function toolbarSelectClass(widthClass: string) {
  return cn(
    "h-10 px-2.5 pr-[30px] text-xs lowercase hover:border-app-border-strong hover:bg-app-surface",
    widthClass,
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
