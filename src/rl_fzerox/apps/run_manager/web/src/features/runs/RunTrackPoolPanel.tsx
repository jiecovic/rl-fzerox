import { useEffect, useMemo, useState } from "react";

import type {
  ConfigMetadata,
  ManagedRun,
  TrackSamplingRuntimeEntry,
  TrackSamplingRuntimeState,
} from "@/shared/api/contract";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { formatRelativeTime } from "@/shared/ui/format";

interface RunTrackPoolPanelProps {
  canReset: boolean;
  isResetting: boolean;
  metadata: ConfigMetadata;
  onReset: () => void;
  run: ManagedRun;
  state: TrackSamplingRuntimeState | null;
}

type TrackPoolCourseView = {
  id: string;
  label: string;
  currentProbability: number | null;
  episodeCount: number | null;
  finishedEpisodeCount: number | null;
  successSampleCount: number | null;
  episodeShare: number | null;
  successRate: number | null;
  completedEnvSteps: number | null;
  stepShare: number | null;
};

type TrackPoolCupView = {
  id: string;
  label: string;
  entries: TrackPoolCourseView[];
  currentProbability: number;
  episodeCount: number;
  finishedEpisodeCount: number;
  successSampleCount: number;
  episodeShare: number;
  successRate: number | null;
  completedEnvSteps: number;
  stepShare: number;
};

type TrackPoolView = {
  cups: TrackPoolCupView[];
  totalCourses: number;
  totalEpisodes: number;
  totalEnvSteps: number;
};

export function RunTrackPoolPanel({
  canReset,
  isResetting,
  metadata,
  onReset,
  run,
  state,
}: RunTrackPoolPanelProps) {
  const visibleState = showTrackSamplingState(state) ? state : null;
  const poolView = useMemo(
    () => buildTrackPoolView(metadata, run, visibleState),
    [metadata, run, visibleState],
  );
  const [activeCupId, setActiveCupId] = useState<string | null>(poolView.cups[0]?.id ?? null);
  const [confirmResetOpen, setConfirmResetOpen] = useState(false);

  useEffect(() => {
    setActiveCupId((current) => {
      if (poolView.cups.length === 0) {
        return null;
      }
      if (current !== null && poolView.cups.some((cup) => cup.id === current)) {
        return current;
      }
      return poolView.cups[0]?.id ?? null;
    });
  }, [poolView.cups]);

  if (!expectsTrackSamplingState(run, poolView.totalCourses)) {
    return null;
  }

  const activeCup = poolView.cups.find((cup) => cup.id === activeCupId) ?? poolView.cups[0] ?? null;

  return (
    <div className="run-track-distribution-panel">
      <div className="run-track-distribution-header">
        <div>
          <strong>Track pool</strong>
          <div className="run-track-distribution-note">
            {visibleState === null
              ? "step-balanced"
              : `step-balanced · updated ${trackSamplingUpdatedLabel(run)}`}
          </div>
        </div>
        <div className="run-track-distribution-meta">
          {visibleState === null
            ? `${poolView.totalCourses} courses`
            : `${poolView.totalEpisodes.toLocaleString()} episodes · ${poolView.totalEnvSteps.toLocaleString()} env steps`}
        </div>
        <button
          className="secondary-button run-track-distribution-reset"
          type="button"
          disabled={!canReset || isResetting}
          onClick={() => setConfirmResetOpen(true)}
        >
          {isResetting ? "Resetting..." : "Reset stats"}
        </button>
      </div>
      {poolView.cups.length > 1 ? (
        <div className="run-track-distribution-tabs" role="tablist" aria-label="Track pool cups">
          {poolView.cups.map((cup) => (
            <button
              aria-selected={cup.id === activeCup?.id}
              className={
                cup.id === activeCup?.id
                  ? "run-track-distribution-tab active"
                  : "run-track-distribution-tab"
              }
              key={cup.id}
              role="tab"
              type="button"
              onClick={() => setActiveCupId(cup.id)}
            >
              <span>{shortCupLabel(cup.label)}</span>
              <span className="run-track-distribution-tab-count">{cup.entries.length}</span>
            </button>
          ))}
        </div>
      ) : null}
      {visibleState === null ? (
        <div className="run-track-distribution-empty">{trackPoolEmptyMessage(run)}</div>
      ) : activeCup === null ? null : (
        <div className="run-track-distribution-body">
          <div className="run-track-distribution-summary">
            <div className="run-track-distribution-summary-title">
              <strong>{activeCup.label}</strong>
              <span>{activeCup.entries.length} courses</span>
            </div>
            <div className="run-track-distribution-summary-metrics">
              <span>Sample {formatPercent(activeCup.currentProbability)}</span>
              <span>Finish {formatOptionalPercent(activeCup.successRate)}</span>
              <span>Episodes {formatPercent(activeCup.episodeShare)}</span>
              <span>Env steps {formatPercent(activeCup.stepShare)}</span>
            </div>
          </div>
          <div className="run-track-distribution-legend">
            <LegendItem kind="sample" label="Sample" />
            <LegendItem kind="success" label="Finish" />
            <LegendItem kind="episodes" label="Episodes" />
            <LegendItem kind="steps" label="Env steps" />
          </div>
          <div className="run-track-distribution-chart">
            <div className="run-track-distribution-axis">
              <span>100%</span>
              <span>50%</span>
              <span>0%</span>
            </div>
            <div
              className="run-track-distribution-columns"
              style={{ ["--track-column-count" as string]: `${activeCup.entries.length}` }}
            >
              {activeCup.entries.map((entry) => (
                <div className="run-track-distribution-column" key={entry.id}>
                  <div className="run-track-distribution-column-bars">
                    <DistributionBar
                      kind="sample"
                      label={`Sample ${formatPercent(entry.currentProbability ?? 0)}`}
                      value={entry.currentProbability ?? 0}
                    />
                    <DistributionBar
                      kind="success"
                      label={successLabel(entry)}
                      value={entry.successRate ?? 0}
                    />
                    <DistributionBar
                      kind="episodes"
                      label={`${(entry.episodeCount ?? 0).toLocaleString()} episodes · ${formatPercent(entry.episodeShare ?? 0)}`}
                      value={entry.episodeShare ?? 0}
                    />
                    <DistributionBar
                      kind="steps"
                      label={`${(entry.completedEnvSteps ?? 0).toLocaleString()} env steps · ${formatPercent(entry.stepShare ?? 0)}`}
                      value={entry.stepShare ?? 0}
                    />
                  </div>
                  <div className="run-track-distribution-column-label">
                    <strong>{entry.label}</strong>
                    <span>
                      {successSummary(entry)} · {(entry.completedEnvSteps ?? 0).toLocaleString()}{" "}
                      steps
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      <ConfirmDialog
        open={confirmResetOpen}
        title="Reset track-pool stats"
        description={`Reset the course distribution history for "${run.name}"? This clears the tracked episode, finish, and env-step counts for this run.`}
        confirmLabel="Reset stats"
        onClose={() => setConfirmResetOpen(false)}
        onConfirm={() => {
          setConfirmResetOpen(false);
          onReset();
        }}
      />
    </div>
  );
}

interface DistributionBarProps {
  kind: "sample" | "success" | "episodes" | "steps";
  label: string;
  value: number;
}

function DistributionBar({ kind, label, value }: DistributionBarProps) {
  return (
    <button
      aria-label={label}
      className="run-track-distribution-bar-cell tooltip-anchor"
      data-tooltip={label}
      type="button"
    >
      <div aria-hidden="true" className="run-track-distribution-bar">
        <div
          className={`run-track-distribution-bar-fill run-track-distribution-bar-fill-${kind}`}
          style={{ height: `${Math.max(0, Math.min(value, 1)) * 100}%` }}
        />
      </div>
    </button>
  );
}

interface LegendItemProps {
  kind: "sample" | "success" | "episodes" | "steps";
  label: string;
}

function LegendItem({ kind, label }: LegendItemProps) {
  return (
    <span className="run-track-distribution-legend-item">
      <span
        aria-hidden="true"
        className={`run-track-distribution-legend-swatch run-track-distribution-legend-swatch-${kind}`}
      />
      {label}
    </span>
  );
}

function buildTrackPoolView(
  metadata: ConfigMetadata,
  run: ManagedRun,
  state: TrackSamplingRuntimeState | null,
): TrackPoolView {
  const cupOrder = metadata.track_cups.map((cup) => cup.id);
  const cupLabels = new Map(metadata.track_cups.map((cup) => [cup.id, cup.label]));
  const builtInCourses = new Map(metadata.built_in_courses.map((course) => [course.id, course]));
  const runtimeEntries = new Map<string, TrackSamplingRuntimeEntry>(
    (state?.entries ?? []).map((entry) => [entry.course_key, entry]),
  );
  const cups = new Map<string, TrackPoolCupView>();

  for (const courseId of run.config.tracks.selected_course_ids) {
    const courseInfo = builtInCourses.get(courseId);
    const runtimeEntry = runtimeEntries.get(courseId) ?? null;
    const cupId = courseInfo?.cup ?? "pool";
    const cupLabel = courseInfo?.cup_label ?? cupLabels.get(cupId) ?? "Pool";
    const cup: TrackPoolCupView = cups.get(cupId) ?? {
      id: cupId,
      label: cupLabel,
      entries: [],
      currentProbability: 0,
      episodeCount: 0,
      finishedEpisodeCount: 0,
      successSampleCount: 0,
      episodeShare: 0,
      successRate: null,
      completedEnvSteps: 0,
      stepShare: 0,
    };
    const courseView: TrackPoolCourseView = {
      id: courseId,
      label: runtimeEntry?.label ?? courseInfo?.display_name ?? courseId,
      currentProbability: runtimeEntry?.current_probability ?? null,
      episodeCount: runtimeEntry?.episode_count ?? null,
      finishedEpisodeCount: runtimeEntry?.finished_episode_count ?? null,
      successSampleCount: runtimeEntry?.success_sample_count ?? null,
      episodeShare: runtimeEntry?.episode_share ?? null,
      successRate: runtimeEntry?.success_rate ?? null,
      completedEnvSteps: runtimeEntry?.completed_env_steps ?? null,
      stepShare: runtimeEntry?.step_share ?? null,
    };
    cup.entries.push(courseView);
    cup.currentProbability += runtimeEntry?.current_probability ?? 0;
    cup.episodeCount += runtimeEntry?.episode_count ?? 0;
    cup.finishedEpisodeCount += runtimeEntry?.finished_episode_count ?? 0;
    cup.successSampleCount += runtimeEntry?.success_sample_count ?? 0;
    cup.episodeShare += runtimeEntry?.episode_share ?? 0;
    cup.completedEnvSteps += runtimeEntry?.completed_env_steps ?? 0;
    cup.stepShare += runtimeEntry?.step_share ?? 0;
    cup.successRate =
      cup.successSampleCount <= 0 ? null : cup.finishedEpisodeCount / cup.successSampleCount;
    cups.set(cupId, cup);
  }

  const sortedCups = [...cups.values()].sort(
    (left, right) => cupIndex(cupOrder, left.id) - cupIndex(cupOrder, right.id),
  );
  return {
    cups: sortedCups,
    totalCourses: run.config.tracks.selected_course_ids.length,
    totalEpisodes: sortedCups.reduce((sum, cup) => sum + cup.episodeCount, 0),
    totalEnvSteps: sortedCups.reduce((sum, cup) => sum + cup.completedEnvSteps, 0),
  };
}

function cupIndex(cupOrder: string[], cupId: string) {
  const index = cupOrder.indexOf(cupId);
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}

function showTrackSamplingState(state: TrackSamplingRuntimeState | null) {
  return state !== null && state.entries.length > 1;
}

function expectsTrackSamplingState(run: ManagedRun, totalCourses: number) {
  return run.config.tracks.sampling_mode === "step_balanced" && totalCourses > 1;
}

function trackSamplingUpdatedLabel(run: ManagedRun) {
  const updatedAt = run.runtime?.updated_at;
  if (updatedAt === undefined || updatedAt === null) {
    return "recently";
  }
  return formatRelativeTime(updatedAt);
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatOptionalPercent(value: number | null) {
  return value === null ? "n/a" : formatPercent(value);
}

function successLabel(entry: TrackPoolCourseView) {
  if ((entry.successSampleCount ?? 0) <= 0 || entry.successRate === null) {
    return "Finish rate not tracked yet";
  }
  return `${(entry.finishedEpisodeCount ?? 0).toLocaleString()} of ${(entry.successSampleCount ?? 0).toLocaleString()} finished · ${formatPercent(entry.successRate)}`;
}

function successSummary(entry: TrackPoolCourseView) {
  if ((entry.successSampleCount ?? 0) <= 0 || entry.successRate === null) {
    return "finish n/a";
  }
  return `${formatPercent(entry.successRate)} finish`;
}

function shortCupLabel(label: string) {
  return label.replace(/\s+Cup$/u, "");
}

function trackPoolEmptyMessage(run: ManagedRun) {
  return run.status === "running"
    ? "Waiting for the worker to publish the current track-pool distribution."
    : "No track-pool stats yet. Resume the run to accumulate a fresh distribution.";
}
