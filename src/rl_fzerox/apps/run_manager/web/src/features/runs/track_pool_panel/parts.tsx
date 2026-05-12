// src/rl_fzerox/apps/run_manager/web/src/features/runs/track_pool_panel/parts.tsx
import {
  formatOptionalPercent,
  formatPercent,
  shortCupLabel,
  successLabel,
  successSummary,
} from "@/features/runs/track_pool_panel/model";
import type { TrackPoolCourseView, TrackPoolCupView } from "@/features/runs/track_pool_panel/types";

interface DistributionBarProps {
  kind: "sample" | "success" | "episodes" | "steps";
  label: string;
  targetValue?: number | null;
  value: number;
}

export function DistributionBar({ kind, label, targetValue, value }: DistributionBarProps) {
  const clampedTarget =
    targetValue === undefined || targetValue === null
      ? null
      : Math.max(0, Math.min(targetValue, 1));
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
        {clampedTarget === null ? null : (
          <div
            className={`run-track-distribution-bar-target run-track-distribution-bar-target-${kind}`}
            style={{ bottom: `${clampedTarget * 100}%` }}
          />
        )}
      </div>
    </button>
  );
}

interface LegendItemProps {
  kind: "sample" | "success" | "episodes" | "steps" | "target";
  label: string;
}

export function LegendItem({ kind, label }: LegendItemProps) {
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

export function CupTabs({
  activeCup,
  cups,
  onSelectCup,
}: {
  activeCup: TrackPoolCupView | null;
  cups: readonly TrackPoolCupView[];
  onSelectCup: (cupId: string) => void;
}) {
  if (cups.length <= 1) {
    return null;
  }
  return (
    <div className="run-track-distribution-tabs" role="tablist" aria-label="Track pool cups">
      {cups.map((cup) => (
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
          onClick={() => onSelectCup(cup.id)}
        >
          <span>{shortCupLabel(cup.label)}</span>
          <span className="run-track-distribution-tab-count">{cup.entries.length}</span>
        </button>
      ))}
    </div>
  );
}

export function TrackPoolBody({ activeCup }: { activeCup: TrackPoolCupView }) {
  return (
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
        <LegendItem kind="target" label="Step target" />
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
            <TrackPoolColumn entry={entry} key={entry.id} />
          ))}
        </div>
      </div>
    </div>
  );
}

function TrackPoolColumn({ entry }: { entry: TrackPoolCourseView }) {
  return (
    <div className="run-track-distribution-column">
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
          label={`${(entry.completedEnvSteps ?? 0).toLocaleString()} env steps · ${formatPercent(entry.stepShare ?? 0)} · target ${formatPercent(entry.targetStepShare ?? 0)}`}
          targetValue={entry.targetStepShare}
          value={entry.stepShare ?? 0}
        />
      </div>
      <div className="run-track-distribution-column-label">
        <strong>{entry.label}</strong>
        <span>
          {successSummary(entry)} · {(entry.completedEnvSteps ?? 0).toLocaleString()} steps
        </span>
      </div>
    </div>
  );
}
