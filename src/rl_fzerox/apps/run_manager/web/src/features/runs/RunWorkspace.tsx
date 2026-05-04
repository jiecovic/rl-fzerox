import { useEffect, useState } from "react";

import {
  CONFIG_SECTION_TABS,
  type ConfigSection,
} from "@/features/configurator/configurator/sections";
import { FieldLabel } from "@/features/configurator/fields";
import { ActionSection } from "@/features/configurator/sections/ActionSection";
import { EnvironmentSection } from "@/features/configurator/sections/EnvironmentSection";
import { LoggingSection } from "@/features/configurator/sections/LoggingSection";
import { ObservationSection } from "@/features/configurator/sections/ObservationSection";
import { PolicySection } from "@/features/configurator/sections/PolicySection";
import { RewardSection } from "@/features/configurator/sections/RewardSection";
import { TracksSection } from "@/features/configurator/sections/TracksSection";
import { TrainingSection } from "@/features/configurator/sections/TrainingSection";
import { VehicleSection } from "@/features/configurator/sections/VehicleSection";
import { RunActivityIndicator } from "@/features/runs/RunActivityIndicator";
import { RunTrackPoolPanel } from "@/features/runs/RunTrackPoolPanel";
import { fetchPolicyPreview, fetchRunTrackSamplingState } from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedRun,
  PolicyArchitecturePreview,
  TrackSamplingRuntimeState,
} from "@/shared/api/contract";
import { formatDate } from "@/shared/ui/format";
import {
  ChartIcon,
  CopyIcon,
  FolderIcon,
  ForkIcon,
  ResumeIcon,
  StopIcon,
  WatchIcon,
} from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { Tabs } from "@/shared/ui/Tabs";

interface RunWorkspaceProps {
  allRuns: ManagedRun[];
  metadata: ConfigMetadata;
  onFork: (runId: string, artifact: "latest" | "best") => Promise<void>;
  onOpenDirectory: (runId: string) => Promise<void>;
  onRename: (runId: string, name: string) => Promise<void>;
  onResume: (runId: string) => Promise<void>;
  onResetTrackPool: (runId: string) => Promise<void>;
  onShowCharts: (runId: string) => void;
  onStop: (runId: string) => Promise<void>;
  onWatch: (runId: string, artifact: "latest" | "best") => Promise<void>;
  run: ManagedRun;
}

export function RunWorkspace({
  allRuns,
  metadata,
  onFork,
  onOpenDirectory,
  onRename,
  onResume,
  onResetTrackPool,
  onShowCharts,
  onStop,
  onWatch,
  run,
}: RunWorkspaceProps) {
  const [nowMs, setNowMs] = useState(() => Date.now());
  const [section, setSection] = useState<ConfigSection>("training");
  const [policyPreview, setPolicyPreview] = useState<PolicyArchitecturePreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [controlError, setControlError] = useState<string | null>(null);
  const [runName, setRunName] = useState(run.name);
  const [isOpeningDirectory, setIsOpeningDirectory] = useState(false);
  const [isForking, setIsForking] = useState(false);
  const [isRenaming, setIsRenaming] = useState(false);
  const [isResuming, setIsResuming] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [selectedArtifact, setSelectedArtifact] = useState<"latest" | "best">("latest");
  const [watchingArtifact, setWatchingArtifact] = useState<"latest" | "best" | null>(null);
  const [isResettingTrackPool, setIsResettingTrackPool] = useState(false);
  const [copiedRunId, setCopiedRunId] = useState(false);
  const [trackSamplingState, setTrackSamplingState] = useState<TrackSamplingRuntimeState | null>(
    null,
  );

  useEffect(() => {
    let ignore = false;
    setPreviewError(null);
    void fetchPolicyPreview(run.config)
      .then((preview) => {
        if (!ignore) {
          setPolicyPreview(preview);
        }
      })
      .catch((caught) => {
        if (!ignore) {
          setPolicyPreview(null);
          setPreviewError(
            caught instanceof Error ? caught.message : "failed to compute policy preview",
          );
        }
      });
    return () => {
      ignore = true;
    };
  }, [run.config]);

  useEffect(() => {
    setRunName(run.name);
  }, [run.name]);

  useEffect(() => {
    if (run.status !== "running") {
      return undefined;
    }
    const intervalId = window.setInterval(() => {
      setNowMs(Date.now());
    }, 1_000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [run.status]);

  useEffect(() => {
    if (!copiedRunId) {
      return undefined;
    }
    const timeoutId = window.setTimeout(() => {
      setCopiedRunId(false);
    }, 1_200);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [copiedRunId]);

  useEffect(() => {
    let ignore = false;

    async function loadTrackSamplingState() {
      try {
        const state = await fetchRunTrackSamplingState(run.id);
        if (!ignore) {
          setTrackSamplingState(state);
        }
      } catch {
        if (!ignore) {
          setTrackSamplingState(null);
        }
      }
    }

    void loadTrackSamplingState();
    if (run.status !== "running") {
      return () => {
        ignore = true;
      };
    }
    const intervalId = window.setInterval(() => {
      void loadTrackSamplingState();
    }, 2_000);
    return () => {
      ignore = true;
      window.clearInterval(intervalId);
    };
  }, [run.id, run.status]);

  async function resumeRun() {
    setIsResuming(true);
    setControlError(null);
    try {
      await onResume(run.id);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to resume run");
    } finally {
      setIsResuming(false);
    }
  }

  async function stopRun() {
    setIsStopping(true);
    setControlError(null);
    try {
      await onStop(run.id);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to stop run");
    } finally {
      setIsStopping(false);
    }
  }

  async function renameRunLabel() {
    setIsRenaming(true);
    setControlError(null);
    try {
      await onRename(run.id, runName.trim());
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to rename run");
    } finally {
      setIsRenaming(false);
    }
  }

  async function openRunDirectoryInBrowser() {
    setIsOpeningDirectory(true);
    setControlError(null);
    try {
      await onOpenDirectory(run.id);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : "failed to open run directory");
    } finally {
      setIsOpeningDirectory(false);
    }
  }

  async function forkRunArtifact(artifact: "latest" | "best") {
    setIsForking(true);
    setControlError(null);
    try {
      await onFork(run.id, artifact);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : `failed to fork ${artifact}`);
    } finally {
      setIsForking(false);
    }
  }

  async function watchRunArtifact(artifact: "latest" | "best") {
    setWatchingArtifact(artifact);
    setControlError(null);
    try {
      await onWatch(run.id, artifact);
    } catch (caught) {
      setControlError(caught instanceof Error ? caught.message : `failed to watch ${artifact}`);
    } finally {
      setWatchingArtifact((current) => (current === artifact ? null : current));
    }
  }

  async function resetTrackPoolState() {
    setIsResettingTrackPool(true);
    setControlError(null);
    try {
      await onResetTrackPool(run.id);
      setTrackSamplingState(null);
    } catch (caught) {
      setControlError(
        caught instanceof Error ? caught.message : "failed to reset track-pool stats",
      );
    } finally {
      setIsResettingTrackPool(false);
    }
  }

  async function copyRunId() {
    try {
      await navigator.clipboard.writeText(run.id);
      setCopiedRunId(true);
      setControlError(null);
    } catch {
      setControlError("failed to copy run id");
    }
  }

  const pendingCommand = run.pending_command;
  const runtime = run.runtime;
  const progressFraction = runtime?.progress_fraction ?? 0;
  const showsLineageTotals = run.lineage_step_offset > 0 || run.parent_run_id !== null;
  const normalizedRunName = runName.trim();
  const canRename =
    normalizedRunName.length > 0 &&
    normalizedRunName !== run.name &&
    !isRenaming &&
    !isOpeningDirectory;
  const canStop = run.status === "running" && pendingCommand === null && !isResuming;
  const canResume =
    (run.status === "paused" || run.status === "stopped" || run.status === "failed") && !isStopping;

  return (
    <Panel>
      <PanelHeader title={run.name} subtitle={runSubtitle(run)} />

      <div className="run-runtime-summary">
        <div className="run-runtime-progress">
          <div className="run-runtime-progress-header">
            <strong>Training progress</strong>
            <span>{progressHeadline(run)}</span>
          </div>
          <div aria-hidden="true" className="run-progress-bar">
            <div
              className="run-progress-bar-fill"
              style={{ width: `${progressFraction * 100}%` }}
            />
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
            {showsLineageTotals ? (
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
                    {lineageSimGameTimeLabelText(run, allRuns)}
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
                  aria-label={copiedRunId ? "Run id copied" : "Copy run id"}
                  className="icon-button compact-icon-button tooltip-anchor run-runtime-copy-button"
                  data-tooltip={copiedRunId ? "Copied" : "Copy run id"}
                  data-tooltip-position="top"
                  type="button"
                  onClick={() => void copyRunId()}
                >
                  <CopyIcon />
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="run-runtime-actions">
          <button
            aria-label={isOpeningDirectory ? "Opening run folder" : "Open run folder"}
            className="icon-button"
            title={isOpeningDirectory ? "Opening..." : "Open folder"}
            type="button"
            disabled={isOpeningDirectory}
            onClick={() => void openRunDirectoryInBrowser()}
          >
            <FolderIcon />
          </button>
          <div className="run-watch-control">
            <label className="run-watch-select-shell">
              <span className="run-watch-select-label">Checkpoint artifact</span>
              <select
                aria-label="Checkpoint artifact"
                className="run-watch-select"
                value={selectedArtifact}
                disabled={watchingArtifact !== null || isForking}
                onChange={(event) =>
                  setSelectedArtifact(event.target.value === "best" ? "best" : "latest")
                }
              >
                <option value="latest">latest</option>
                <option value="best">best</option>
              </select>
            </label>
            <button
              aria-label={
                watchingArtifact === selectedArtifact
                  ? `Opening ${selectedArtifact} checkpoint watch`
                  : `Watch ${selectedArtifact} checkpoint`
              }
              className="icon-button"
              title={
                watchingArtifact === selectedArtifact
                  ? `Opening ${selectedArtifact}...`
                  : `Watch ${selectedArtifact}`
              }
              type="button"
              disabled={watchingArtifact !== null}
              onClick={() => void watchRunArtifact(selectedArtifact)}
            >
              <WatchIcon />
            </button>
            <button
              aria-label={
                isForking
                  ? `Forking ${selectedArtifact} checkpoint`
                  : `Fork ${selectedArtifact} checkpoint`
              }
              className="icon-button"
              title={isForking ? `Forking ${selectedArtifact}...` : `Fork ${selectedArtifact}`}
              type="button"
              disabled={isForking}
              onClick={() => void forkRunArtifact(selectedArtifact)}
            >
              <ForkIcon />
            </button>
          </div>
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
              isStopping
                ? "Stopping run"
                : pendingCommand === "stop"
                  ? "Stop requested"
                  : "Stop run"
            }
            className="icon-button"
            title={
              isStopping ? "Stopping..." : pendingCommand === "stop" ? "Stop requested" : "Stop"
            }
            type="button"
            disabled={!canStop}
            onClick={() => void stopRun()}
          >
            <StopIcon />
          </button>
          <button
            aria-label={isResuming ? "Resuming run" : "Resume run"}
            className="icon-button"
            title={isResuming ? "Resuming..." : "Resume"}
            type="button"
            disabled={!canResume}
            onClick={() => void resumeRun()}
          >
            <ResumeIcon />
          </button>
        </div>

        <RunTrackPoolPanel
          canReset={run.status === "stopped"}
          isResetting={isResettingTrackPool}
          metadata={metadata}
          onReset={() => void resetTrackPoolState()}
          run={run}
          state={trackSamplingState}
        />
      </div>

      {controlError !== null || previewError !== null ? (
        <div className="configurator-feedback-stack">
          {controlError !== null ? <Notice tone="error">{controlError}</Notice> : null}
          {previewError !== null ? <Notice tone="error">{previewError}</Notice> : null}
        </div>
      ) : null}

      <div className="form-grid run-identity-grid run-identity-grid-readonly">
        <div className="field-shell">
          <FieldLabel
            help="Manager label for this run. Renaming it does not mutate the frozen training config."
            label="Run name"
          />
          <input
            aria-label="Run name"
            value={runName}
            onChange={(event) => setRunName(event.target.value)}
          />
        </div>
        <div className="field-shell">
          <FieldLabel help="Frozen seed stored with this launched run config." label="Seed" />
          <input aria-label="Seed" readOnly value={run.config.seed} />
        </div>
        <div className="run-identity-actions">
          <button
            className="secondary-button"
            type="button"
            disabled={!canRename}
            onClick={() => void renameRunLabel()}
          >
            {isRenaming ? "Saving..." : "Save name"}
          </button>
        </div>
      </div>

      <div className="section-tabs-row">
        <Tabs
          label="Run configuration sections"
          activeId={section}
          items={CONFIG_SECTION_TABS}
          variant="section"
          onSelect={(id) => setSection(id)}
        />
      </div>

      <div className="readonly-config-shell">
        {section === "tracks" ? (
          <TracksSection
            config={run.config}
            defaultConfig={run.config}
            metadata={metadata}
            setConfig={() => undefined}
          />
        ) : null}
        {section === "training" ? (
          <TrainingSection
            config={run.config}
            defaultConfig={run.config}
            setConfig={() => undefined}
          />
        ) : null}
        {section === "observation" ? (
          <ObservationSection
            config={run.config}
            defaultConfig={run.config}
            metadata={metadata}
            preview={policyPreview}
            setConfig={() => undefined}
          />
        ) : null}
        {section === "policy" ? (
          <PolicySection
            config={run.config}
            defaultConfig={run.config}
            metadata={metadata}
            preview={policyPreview}
            setConfig={() => undefined}
          />
        ) : null}
        {section === "reward" ? (
          <RewardSection
            config={run.config}
            defaultConfig={run.config}
            setConfig={() => undefined}
          />
        ) : null}
        {section === "vehicle" ? (
          <VehicleSection
            config={run.config}
            defaultConfig={run.config}
            metadata={metadata}
            setConfig={() => undefined}
          />
        ) : null}
        {section === "action" ? (
          <ActionSection
            config={run.config}
            defaultConfig={run.config}
            metadata={metadata}
            setConfig={() => undefined}
          />
        ) : null}
        {section === "environment" ? (
          <EnvironmentSection
            config={run.config}
            defaultConfig={run.config}
            setConfig={() => undefined}
          />
        ) : null}
        {section === "logging" ? (
          <LoggingSection
            config={run.config}
            defaultConfig={run.config}
            setConfig={() => undefined}
          />
        ) : null}
      </div>
    </Panel>
  );
}

function progressHeadline(run: ManagedRun) {
  const runtime = run.runtime;
  if (runtime === null) {
    return run.status === "failed" ? "No runtime samples" : "Waiting for first sample";
  }
  return run.lineage_step_offset > 0
    ? `${(runtime.progress_fraction * 100).toFixed(1)}% of this fork`
    : `${(runtime.progress_fraction * 100).toFixed(1)}% complete`;
}

function progressNote(run: ManagedRun) {
  const runtime = run.runtime;
  const target = run.config.train.total_timesteps.toLocaleString();
  if (runtime === null) {
    const startupMessage = latestStartupMessage(run);
    if (run.status === "failed") {
      return (
        startupMessage ?? `Run failed before the first callback flush. Target was ${target} steps.`
      );
    }
    return (
      startupMessage ??
      `Target ${target} steps. Runtime metrics appear after the first callback flush.`
    );
  }
  if (run.lineage_step_offset <= 0) {
    return `${runtime.num_timesteps.toLocaleString()} / ${runtime.total_timesteps.toLocaleString()} steps`;
  }
  const lineageSteps = run.lineage_step_offset + runtime.num_timesteps;
  return `${lineageSteps.toLocaleString()} lineage steps · ${runtime.num_timesteps.toLocaleString()} / ${runtime.total_timesteps.toLocaleString()} local fork steps`;
}

function runSubtitle(run: ManagedRun) {
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

function envStepRateLabel(run: ManagedRun) {
  const fps = envStepRateValue(run);
  return fps === null ? "n/a" : formatRate(fps);
}

function trainFpsLabel(run: ManagedRun) {
  const envStepRate = envStepRateValue(run);
  if (envStepRate === null) {
    return "n/a";
  }
  return formatRate(envStepRate * Math.max(run.config.action.action_repeat, 1));
}

function formatRate(value: number) {
  return value >= 100 ? `${value.toFixed(0)}` : `${value.toFixed(1)}`;
}

function localWallTimeLabel(run: ManagedRun, nowMs: number) {
  const seconds = localWallTimeSeconds(run, nowMs);
  return seconds === null ? "n/a" : formatDurationSeconds(seconds);
}

function lineageWallTimeLabel(run: ManagedRun, allRuns: ManagedRun[], nowMs: number) {
  const seconds = lineageWallTimeSeconds(run, allRuns, nowMs);
  return seconds === null ? "n/a" : formatDurationSeconds(seconds);
}

function localSimGameTimeLabel(run: ManagedRun) {
  const seconds = localSimGameTimeSeconds(run);
  return seconds === null ? "n/a" : formatDurationSeconds(seconds);
}

function lineageSimGameTimeLabelText(run: ManagedRun, allRuns: ManagedRun[]) {
  const seconds = lineageSimGameTimeSeconds(run, allRuns);
  return seconds === null ? "n/a" : formatDurationSeconds(seconds);
}

function localSimToWallRatioLabel(run: ManagedRun, nowMs: number) {
  const localWallSeconds = localWallTimeSeconds(run, nowMs);
  const localSimSeconds = localSimGameTimeSeconds(run);
  if (localWallSeconds === null || localSimSeconds === null || localWallSeconds <= 0) {
    return "n/a";
  }
  return `${(localSimSeconds / localWallSeconds).toFixed(2)}x`;
}

function lineageSimToWallRatioLabel(run: ManagedRun, allRuns: ManagedRun[], nowMs: number) {
  const lineageWallSeconds = lineageWallTimeSeconds(run, allRuns, nowMs);
  const lineageSimSeconds = lineageSimGameTimeSeconds(run, allRuns);
  if (lineageWallSeconds === null || lineageSimSeconds === null || lineageWallSeconds <= 0) {
    return "n/a";
  }
  return `${(lineageSimSeconds / lineageWallSeconds).toFixed(2)}x`;
}

function lineageStepsLabel(run: ManagedRun) {
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

function latestStartupMessage(run: ManagedRun) {
  const startupEvent = run.recent_events.find((event) => event.kind.startsWith("startup_"));
  if (startupEvent === undefined) {
    return null;
  }
  return startupEvent.message;
}

function localSimGameTimeSeconds(run: ManagedRun) {
  const runtime = run.runtime;
  if (runtime === null) {
    return null;
  }
  const actionRepeat = Math.max(run.config.action.action_repeat, 1);
  return (runtime.num_timesteps * actionRepeat) / 60;
}

function lineageSimGameTimeSeconds(run: ManagedRun, allRuns: ManagedRun[]) {
  return lineageAggregateSeconds(run, allRuns, localSimGameTimeSeconds);
}

function envStepRateValue(run: ManagedRun) {
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

function localWallTimeSeconds(run: ManagedRun, nowMs: number) {
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

function activeRuntimeWallTimeSeconds(run: ManagedRun) {
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

function lineageWallTimeSeconds(run: ManagedRun, allRuns: ManagedRun[], nowMs: number) {
  return lineageAggregateSeconds(run, allRuns, (currentRun) =>
    localWallTimeSeconds(currentRun, nowMs),
  );
}

function lineageAggregateSeconds(
  run: ManagedRun,
  allRuns: ManagedRun[],
  selector: (run: ManagedRun) => number | null,
) {
  const runsById = new Map(allRuns.map((candidate) => [candidate.id, candidate]));
  const visited = new Set<string>();
  let totalSeconds = 0;
  let hasAnyValue = false;
  let currentRun: ManagedRun | null = run;

  while (currentRun !== null && !visited.has(currentRun.id)) {
    visited.add(currentRun.id);
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

function formatDurationSeconds(value: number) {
  const totalSeconds = Math.max(0, Math.floor(value));
  const durationUnits = [
    { label: "y", seconds: 365 * 24 * 3600 },
    { label: "mo", seconds: 30 * 24 * 3600 },
    { label: "d", seconds: 24 * 3600 },
    { label: "h", seconds: 3600 },
    { label: "m", seconds: 60 },
    { label: "s", seconds: 1 },
  ] as const;

  let remainingSeconds = totalSeconds;
  const parts: string[] = [];
  for (const unit of durationUnits) {
    if (remainingSeconds < unit.seconds && parts.length === 0 && unit.label !== "s") {
      continue;
    }
    const amount = Math.floor(remainingSeconds / unit.seconds);
    remainingSeconds -= amount * unit.seconds;
    if (amount > 0 || unit.label === "s") {
      parts.push(`${amount}${unit.label}`);
    }
    if (parts.length >= 3) {
      break;
    }
  }
  return parts.join(" ");
}
