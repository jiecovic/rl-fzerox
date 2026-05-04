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
import { fetchPolicyPreview, fetchRunTrackSamplingState } from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedRun,
  PolicyArchitecturePreview,
  TrackSamplingRuntimeState,
} from "@/shared/api/contract";
import { formatDate, formatRelativeTime } from "@/shared/ui/format";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { Tabs } from "@/shared/ui/Tabs";

interface RunWorkspaceProps {
  metadata: ConfigMetadata;
  onFork: (runId: string, artifact: "latest" | "best") => Promise<void>;
  onOpenDirectory: (runId: string) => Promise<void>;
  onRename: (runId: string, name: string) => Promise<void>;
  onResume: (runId: string) => Promise<void>;
  onShowCharts: (runId: string) => void;
  onStop: (runId: string) => Promise<void>;
  onWatch: (runId: string, artifact: "latest" | "best") => Promise<void>;
  run: ManagedRun;
}

export function RunWorkspace({
  metadata,
  onFork,
  onOpenDirectory,
  onRename,
  onResume,
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

  const pendingCommand = run.pending_command;
  const runtime = run.runtime;
  const progressFraction = runtime?.progress_fraction ?? 0;
  const visibleTrackSamplingState = showTrackSamplingState(trackSamplingState)
    ? trackSamplingState
    : null;
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
              <strong className="run-runtime-metric-value">{envStepRateLabel(run)}</strong>
            </div>
            <div className="run-runtime-metric">
              <span className="run-runtime-metric-label">Train fps</span>
              <strong className="run-runtime-metric-value">{trainFpsLabel(run)}</strong>
            </div>
            <div className="run-runtime-metric">
              <span className="run-runtime-metric-label">Wall time</span>
              <strong className="run-runtime-metric-value">{wallTimeLabel(run, nowMs)}</strong>
            </div>
            <div className="run-runtime-metric">
              <span className="run-runtime-metric-label">Sim game time</span>
              <strong className="run-runtime-metric-value">{simGameTimeLabel(run)}</strong>
            </div>
            <div className="run-runtime-metric">
              <span className="run-runtime-metric-label">Sim / wall</span>
              <strong className="run-runtime-metric-value">
                {simToWallRatioLabel(run, nowMs)}
              </strong>
            </div>
            {run.lineage_step_offset > 0 ? (
              <div className="run-runtime-metric">
                <span className="run-runtime-metric-label">Lineage steps</span>
                <strong className="run-runtime-metric-value">{lineageStepsLabel(run)}</strong>
              </div>
            ) : null}
          </div>
          {visibleTrackSamplingState !== null ? (
            <div className="run-track-distribution-panel">
              <div className="run-track-distribution-header">
                <div>
                  <strong>Track pool</strong>
                  <div className="run-track-distribution-note">
                    step-balanced · updated {trackSamplingUpdatedLabel(run)}
                  </div>
                </div>
                <div className="run-track-distribution-meta">
                  every {visibleTrackSamplingState.update_episodes} episodes
                </div>
              </div>
              <table className="run-track-distribution-table">
                <thead>
                  <tr>
                    <th>Course</th>
                    <th>Sample</th>
                    <th>Episodes</th>
                    <th>Env steps</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleTrackSamplingState.entries.map((entry) => (
                    <tr key={entry.track_id}>
                      <th scope="row">{entry.label}</th>
                      <td>{formatPercent(entry.current_probability)}</td>
                      <td>
                        {entry.episode_count.toLocaleString()}
                        <span className="run-track-distribution-share">
                          {formatPercent(entry.episode_share)}
                        </span>
                      </td>
                      <td>
                        {entry.completed_env_steps.toLocaleString()}
                        <span className="run-track-distribution-share">
                          {formatPercent(entry.step_share)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
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

function ForkIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="16" viewBox="0 0 20 20" width="16">
      <path
        d="M6 4.5a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0Zm0 11a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0Zm10-5.5a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0ZM4.5 6v3.25c0 .69.56 1.25 1.25 1.25h7.5M4.5 14v-3.25c0-.69.56-1.25 1.25-1.25h7.5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
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
    if (run.status === "failed") {
      return `Run failed before the first callback flush. Target was ${target} steps.`;
    }
    return `Target ${target} steps. Runtime metrics appear after the first callback flush.`;
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

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function wallTimeLabel(run: ManagedRun, nowMs: number) {
  const seconds = wallTimeSeconds(run, nowMs);
  return seconds === null ? "n/a" : formatDurationSeconds(seconds);
}

function simGameTimeLabel(run: ManagedRun) {
  const value = simGameTimeSeconds(run);
  if (value === null) {
    return "n/a";
  }
  return formatDurationSeconds(value);
}

function simToWallRatioLabel(run: ManagedRun, nowMs: number) {
  const wallSeconds = wallTimeSeconds(run, nowMs);
  const simSeconds = localSimGameTimeSeconds(run);
  if (wallSeconds === null || simSeconds === null || wallSeconds <= 0) {
    return "n/a";
  }
  return `${(simSeconds / wallSeconds).toFixed(2)}x`;
}

function simGameTimeSeconds(run: ManagedRun) {
  const localSimSeconds = localSimGameTimeSeconds(run);
  if (localSimSeconds === null) {
    return null;
  }
  const actionRepeat = Math.max(run.config.action.action_repeat, 1);
  return (run.lineage_step_offset * actionRepeat) / 60 + localSimSeconds;
}

function lineageStepsLabel(run: ManagedRun) {
  const runtime = run.runtime;
  if (runtime === null) {
    return "n/a";
  }
  return (run.lineage_step_offset + runtime.num_timesteps).toLocaleString();
}

function localSimGameTimeSeconds(run: ManagedRun) {
  const runtime = run.runtime;
  if (runtime === null) {
    return null;
  }
  const actionRepeat = Math.max(run.config.action.action_repeat, 1);
  return (runtime.num_timesteps * actionRepeat) / 60;
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

function wallTimeSeconds(run: ManagedRun, nowMs: number) {
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

function showTrackSamplingState(state: TrackSamplingRuntimeState | null) {
  return state !== null && state.entries.length > 1;
}

function trackSamplingUpdatedLabel(run: ManagedRun) {
  const updatedAt = run.runtime?.updated_at;
  if (updatedAt === undefined || updatedAt === null) {
    return "recently";
  }
  return formatRelativeTime(updatedAt);
}

function FolderIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 20 20" width="14">
      <path
        d="M2.5 5.5h4.2l1.3 1.7H17a1 1 0 0 1 1 1v6.8a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V6.5a1 1 0 0 1 .5-1Z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}

function ChartIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 20 20" width="14">
      <path
        d="M4.5 15.5V10.5M10 15.5V7.5M15.5 15.5V4.5M3.5 15.5h13"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}

function WatchIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 20 20" width="14">
      <path
        d="M2.5 10s2.8-4.5 7.5-4.5S17.5 10 17.5 10s-2.8 4.5-7.5 4.5S2.5 10 2.5 10Z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
      <circle cx="10" cy="10" r="2.2" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 20 20" width="14">
      <rect
        height="9"
        rx="1.25"
        stroke="currentColor"
        strokeWidth="1.5"
        width="9"
        x="5.5"
        y="5.5"
      />
    </svg>
  );
}

function ResumeIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 20 20" width="14">
      <path
        d="M7 5.5v9l7-4.5-7-4.5Z"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}
