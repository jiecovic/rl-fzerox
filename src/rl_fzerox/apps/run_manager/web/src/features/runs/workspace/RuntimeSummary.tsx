// src/rl_fzerox/apps/run_manager/web/src/features/runs/workspace/RuntimeSummary.tsx
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
  trainFpsLabel,
} from "@/features/runs/workspace/model";
import type { ConfigMetadata, ManagedRun, TrackSamplingRuntimeState } from "@/shared/api/contract";
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
  nowMs: number;
  onShowCharts: (runId: string) => void;
  run: ManagedRun;
  trackSamplingState: TrackSamplingRuntimeState | null;
}

export function RunRuntimeSummary({
  actions,
  allRuns,
  metadata,
  nowMs,
  onShowCharts,
  run,
  trackSamplingState,
}: RunRuntimeSummaryProps) {
  const runtime = run.runtime;
  const progressFraction = runtime?.progress_fraction ?? 0;
  const hasLineageTotals = showsLineageTotals(run);

  return (
    <div className="run-runtime-summary">
      <div className="run-runtime-progress">
        <div className="run-runtime-progress-header">
          <strong>Training progress</strong>
          <span>{progressHeadline(run)}</span>
        </div>
        <div aria-hidden="true" className="run-progress-bar">
          <div className="run-progress-bar-fill" style={{ width: `${progressFraction * 100}%` }} />
        </div>
        <div className="run-runtime-progress-note">{progressNote(run)}</div>
        <div className="run-runtime-metrics-grid">
          <div className="run-runtime-metric">
            <span className="run-runtime-metric-label">Env step / s</span>
            <div className="run-runtime-metric-value">{envStepRateLabel(run)}</div>
          </div>
          <div className="run-runtime-metric">
            <span className="run-runtime-metric-label">Train fps</span>
            <div className="run-runtime-metric-value">{trainFpsLabel(run)}</div>
          </div>
          {hasLineageTotals ? (
            <>
              <div className="run-runtime-metric">
                <span className="run-runtime-metric-label">
                  {run.runtime === null ? "Setup time · local" : "Wall time · local"}
                </span>
                <div className="run-runtime-metric-value">{localWallTimeLabel(run, nowMs)}</div>
              </div>
              <div className="run-runtime-metric">
                <span className="run-runtime-metric-label">
                  {run.runtime === null ? "Setup time · total" : "Wall time · total"}
                </span>
                <div className="run-runtime-metric-value">
                  {lineageWallTimeLabel(run, allRuns, nowMs)}
                </div>
              </div>
              <div className="run-runtime-metric">
                <span className="run-runtime-metric-label">Sim game time · local</span>
                <div className="run-runtime-metric-value">{localSimGameTimeLabel(run)}</div>
              </div>
              <div className="run-runtime-metric">
                <span className="run-runtime-metric-label">Sim game time · total</span>
                <div className="run-runtime-metric-value">
                  {lineageSimGameTimeLabel(run, allRuns)}
                </div>
              </div>
              <div className="run-runtime-metric">
                <span className="run-runtime-metric-label">Sim / wall · local</span>
                <div className="run-runtime-metric-value">
                  {localSimToWallRatioLabel(run, nowMs)}
                </div>
              </div>
              <div className="run-runtime-metric">
                <span className="run-runtime-metric-label">Sim / wall · total</span>
                <div className="run-runtime-metric-value">
                  {lineageSimToWallRatioLabel(run, allRuns, nowMs)}
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="run-runtime-metric">
                <span className="run-runtime-metric-label">
                  {run.runtime === null ? "Setup time" : "Wall time"}
                </span>
                <div className="run-runtime-metric-value">{localWallTimeLabel(run, nowMs)}</div>
              </div>
              <div className="run-runtime-metric">
                <span className="run-runtime-metric-label">Sim game time</span>
                <div className="run-runtime-metric-value">{localSimGameTimeLabel(run)}</div>
              </div>
              <div className="run-runtime-metric">
                <span className="run-runtime-metric-label">Sim / wall</span>
                <div className="run-runtime-metric-value">
                  {localSimToWallRatioLabel(run, nowMs)}
                </div>
              </div>
            </>
          )}
          {run.lineage_step_offset > 0 ? (
            <div className="run-runtime-metric">
              <span className="run-runtime-metric-label">Lineage steps</span>
              <div className="run-runtime-metric-value">{lineageStepsLabel(run)}</div>
            </div>
          ) : null}
          <div className="run-runtime-metric">
            <span className="run-runtime-metric-label">Run id</span>
            <div className="run-runtime-metric-value run-runtime-metric-inline">
              <code className="run-runtime-code">{run.id}</code>
              <button
                aria-label={actions.copiedRunId ? "Run id copied" : "Copy run id"}
                className="icon-button compact-icon-button tooltip-anchor run-runtime-copy-button"
                data-tooltip={actions.copiedRunId ? "Copied" : "Copy run id"}
                data-tooltip-position="top"
                type="button"
                onClick={() => void actions.copyRunId()}
              >
                <CopyIcon />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="run-runtime-actions">
        <button
          aria-label={actions.isOpeningDirectory ? "Opening run folder" : "Open run folder"}
          className="icon-button"
          title={actions.isOpeningDirectory ? "Opening..." : "Open folder"}
          type="button"
          disabled={actions.isOpeningDirectory}
          onClick={() => void actions.openRunDirectoryInBrowser()}
        >
          <FolderIcon />
        </button>
        <div className="run-watch-control">
          {actions.watchToast !== null ? (
            <div
              aria-live="polite"
              className={`run-watch-toast run-watch-toast-${actions.watchToast.tone}`}
              role={actions.watchToast.tone === "error" ? "alert" : "status"}
            >
              {actions.watchToast.message}
            </div>
          ) : null}
          <label className="run-watch-select-shell">
            <span className="run-watch-select-label">Checkpoint artifact</span>
            <select
              aria-label="Checkpoint artifact"
              className="run-watch-select"
              value={actions.selectedArtifact}
              disabled={actions.watchingArtifact !== null || actions.isForking}
              onChange={(event) =>
                actions.setSelectedArtifact(event.target.value === "best" ? "best" : "latest")
              }
            >
              <option value="latest">latest</option>
              <option value="best">best</option>
            </select>
          </label>
          <button
            aria-label={
              actions.watchingArtifact === actions.selectedArtifact
                ? `Opening ${actions.selectedArtifact} checkpoint watch`
                : `Watch ${actions.selectedArtifact} checkpoint`
            }
            className="icon-button"
            title={
              actions.watchingArtifact === actions.selectedArtifact
                ? `Opening ${actions.selectedArtifact}...`
                : `Watch ${actions.selectedArtifact}`
            }
            type="button"
            disabled={actions.watchingArtifact !== null}
            onClick={() => void actions.watchRunArtifact(actions.selectedArtifact)}
          >
            <WatchIcon />
          </button>
          <button
            aria-label={
              actions.isForking
                ? `Forking ${actions.selectedArtifact} checkpoint`
                : `Fork ${actions.selectedArtifact} checkpoint`
            }
            className="icon-button"
            title={
              actions.isForking
                ? `Forking ${actions.selectedArtifact}...`
                : `Fork ${actions.selectedArtifact}`
            }
            type="button"
            disabled={actions.isForking}
            onClick={() => void actions.forkRunArtifact(actions.selectedArtifact)}
          >
            <ForkIcon />
          </button>
        </div>
        <button
          aria-label={
            actions.isCreatingDraftFromRun
              ? "Creating draft from run"
              : "Create editable draft from run"
          }
          className="icon-button"
          title={actions.isCreatingDraftFromRun ? "Creating draft..." : "Create editable draft"}
          type="button"
          disabled={actions.isCreatingDraftFromRun}
          onClick={() => void actions.createDraftFromRun()}
        >
          <SaveDraftIcon />
        </button>
        <button
          aria-label="Show run charts"
          className="icon-button"
          title="Charts"
          type="button"
          onClick={() => onShowCharts(run.id)}
        >
          <ChartIcon />
        </button>
        <button
          aria-label={
            actions.isStopping
              ? "Stopping run"
              : run.pending_command === "stop"
                ? "Stop requested"
                : "Stop run"
          }
          className="icon-button"
          title={
            actions.isStopping
              ? "Stopping..."
              : run.pending_command === "stop"
                ? "Stop requested"
                : "Stop"
          }
          type="button"
          disabled={!actions.canStop}
          onClick={() => void actions.stopRun()}
        >
          <StopIcon />
        </button>
        <button
          aria-label={actions.isResuming ? "Resuming run" : "Resume run"}
          className="icon-button"
          title={actions.isResuming ? "Resuming..." : "Resume"}
          type="button"
          disabled={!actions.canResume}
          onClick={() => void actions.resumeRun()}
        >
          <ResumeIcon />
        </button>
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

export function runWorkspaceSubtitle(run: ManagedRun) {
  return (
    <>
      <span>{run.status}</span>
      <span aria-hidden="true"> · </span>
      <span>created {formatDate(run.created_at)}</span>
      <span aria-hidden="true"> · </span>
      <RunActivityIndicator run={run} />
    </>
  );
}
