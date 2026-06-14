// web/run-manager/src/entities/trackPool/ui/TrackPoolParts.tsx
import { type KeyboardEvent, memo } from "react";
import type { TrackPoolCourseView, TrackPoolCupView } from "@/entities/trackPool/model/types";
import {
  completionSummary,
  displaySuccessRate,
  formatOptionalPercent,
  formatPercent,
  shortCupLabel,
  successLabel,
  successSummary,
  xCupGenerationSummary,
  xCupRegenerationSummary,
} from "@/entities/trackPool/model/view";
import { cn } from "@/shared/ui/cn";
import { AppTooltip } from "@/shared/ui/Tooltip";

interface DistributionBarProps {
  completionValue?: number | null;
  kind: "sample" | "success" | "episodes" | "steps";
  label: string;
  targetValue?: number | null;
  value: number;
}

export const DistributionBar = memo(function DistributionBar({
  completionValue,
  kind,
  label,
  targetValue,
  value,
}: DistributionBarProps) {
  const clampedTarget =
    targetValue === undefined || targetValue === null
      ? null
      : Math.max(0, Math.min(targetValue, 1));
  const clampedCompletion =
    completionValue === undefined || completionValue === null
      ? null
      : Math.max(0, Math.min(completionValue, 1));
  return (
    <AppTooltip content={label}>
      <button aria-label={label} className="grid h-full items-end" type="button">
        <div
          aria-hidden="true"
          className="relative h-full overflow-hidden border border-app-border bg-app-surface-muted"
        >
          <div
            className={`absolute inset-x-0 bottom-0 run-track-distribution-bar-fill-${kind}`}
            style={{ height: `${Math.max(0, Math.min(value, 1)) * 100}%` }}
          />
          {clampedTarget === null ? null : (
            <div
              className="run-track-distribution-bar-target pointer-events-none absolute -left-px -right-px z-[1] translate-y-1/2"
              style={{ bottom: `${clampedTarget * 100}%` }}
            />
          )}
          {clampedCompletion === null ? null : (
            <div
              className="run-track-distribution-completion-marker pointer-events-none absolute -left-px -right-px z-[2] translate-y-1/2"
              style={{ bottom: `${clampedCompletion * 100}%` }}
            />
          )}
        </div>
      </button>
    </AppTooltip>
  );
});

interface LegendItemProps {
  kind: "sample" | "success" | "episodes" | "steps" | "completion" | "target";
  label: string;
}

export function LegendItem({ kind, label }: LegendItemProps) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-app-muted">
      <span
        aria-hidden="true"
        className={`h-2.5 w-2.5 run-track-distribution-legend-swatch-${kind}`}
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
  const activeIndex = Math.max(
    0,
    cups.findIndex((cup) => cup.id === activeCup?.id),
  );

  function selectCupByIndex(index: number, event: KeyboardEvent<HTMLDivElement>) {
    const cup = cups[index];
    if (cup === undefined) {
      return;
    }
    event.preventDefault();
    onSelectCup(cup.id);
    const tab = [...event.currentTarget.querySelectorAll<HTMLButtonElement>("[role='tab']")].find(
      (candidate) => candidate.dataset.cupId === cup.id,
    );
    tab?.focus();
  }

  return (
    <div
      className="flex flex-wrap gap-1.5 border-b border-app-border px-3.5 py-2.5"
      role="tablist"
      aria-label="Track pool cups"
      onKeyDown={(event) => {
        if (event.key === "ArrowRight" || event.key === "ArrowDown") {
          selectCupByIndex((activeIndex + 1) % cups.length, event);
        } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
          selectCupByIndex((activeIndex - 1 + cups.length) % cups.length, event);
        } else if (event.key === "Home") {
          selectCupByIndex(0, event);
        } else if (event.key === "End") {
          selectCupByIndex(cups.length - 1, event);
        }
      }}
    >
      {cups.map((cup) => (
        <button
          aria-selected={cup.id === activeCup?.id}
          className={trackCupTabClass(cup.id === activeCup?.id)}
          data-cup-id={cup.id}
          key={cup.id}
          role="tab"
          tabIndex={cup.id === activeCup?.id ? 0 : -1}
          type="button"
          onClick={() => onSelectCup(cup.id)}
        >
          <span>{shortCupLabel(cup.label)}</span>
          <span className="text-[11px] text-app-muted">{cup.entries.length}</span>
        </button>
      ))}
    </div>
  );
}

export function TrackPoolBody({
  activeCup,
  sampleBarUsesTargetShare,
  stepMetricLabel,
  showStepTarget,
  xCupRegenerationMinEpisodes,
  xCupRegenerationThreshold,
}: {
  activeCup: TrackPoolCupView;
  sampleBarUsesTargetShare: boolean;
  stepMetricLabel: string;
  showStepTarget: boolean;
  xCupRegenerationMinEpisodes: number | null;
  xCupRegenerationThreshold: number | null;
}) {
  const shareLabel = sampleBarUsesTargetShare ? "Target" : "Sample";
  return (
    <div className="px-3.5 py-3">
      <div className="mb-3 flex items-baseline justify-between gap-3 max-[760px]:grid">
        <div className="grid gap-0.5">
          <strong>{activeCup.label}</strong>
          <span className="text-xs tabular-nums text-app-muted">
            {activeCup.entries.length} courses
          </span>
        </div>
        <div className="flex flex-wrap justify-end gap-x-3 gap-y-1 text-xs tabular-nums text-app-muted max-[760px]:justify-start">
          <span>
            {shareLabel} {formatPercent(effectiveCupShare(activeCup, sampleBarUsesTargetShare))}
          </span>
          <span>Finish {formatOptionalPercent(activeCup.successRate)}</span>
          <span>Episodes {formatPercent(activeCup.episodeShare)}</span>
          <span>Env steps {formatPercent(activeCup.stepShare)}</span>
        </div>
      </div>
      <div className="mb-3 flex flex-wrap gap-3">
        <LegendItem kind="sample" label={shareLabel} />
        <LegendItem kind="success" label="Finish" />
        <LegendItem kind="episodes" label="Episodes" />
        <LegendItem kind="steps" label="Env steps" />
        <LegendItem kind="completion" label="Completion" />
        {showStepTarget ? <LegendItem kind="target" label="Step target" /> : null}
      </div>
      <div className="grid grid-cols-[32px_minmax(0,1fr)] gap-2.5">
        <div className="grid h-44 items-stretch text-[11px] tabular-nums text-app-muted">
          <span className="flex items-start justify-end">100%</span>
          <span className="flex items-center justify-end">50%</span>
          <span className="flex items-end justify-end">0%</span>
        </div>
        <div
          className="run-track-distribution-columns grid gap-2.5"
          style={{ ["--track-column-count" as string]: `${activeCup.entries.length}` }}
        >
          {activeCup.entries.map((entry) => (
            <TrackPoolColumn
              entry={entry}
              key={entry.id}
              sampleBarUsesTargetShare={sampleBarUsesTargetShare}
              stepMetricLabel={stepMetricLabel}
              showStepTarget={showStepTarget}
              xCupRegenerationMinEpisodes={xCupRegenerationMinEpisodes}
              xCupRegenerationThreshold={xCupRegenerationThreshold}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

const TrackPoolColumn = memo(function TrackPoolColumn({
  entry,
  sampleBarUsesTargetShare,
  stepMetricLabel,
  showStepTarget,
  xCupRegenerationMinEpisodes,
  xCupRegenerationThreshold,
}: {
  entry: TrackPoolCourseView;
  sampleBarUsesTargetShare: boolean;
  stepMetricLabel: string;
  showStepTarget: boolean;
  xCupRegenerationMinEpisodes: number | null;
  xCupRegenerationThreshold: number | null;
}) {
  const completion = completionSummary(entry);
  const generation = xCupGenerationSummary(entry);
  const regeneration = xCupRegenerationSummary(
    entry,
    xCupRegenerationThreshold,
    xCupRegenerationMinEpisodes,
  );
  const shareLabel = sampleBarUsesTargetShare ? "Target" : "Sample";
  const shareValue = effectiveCourseShare(entry, sampleBarUsesTargetShare);
  return (
    <div className="grid min-w-0 gap-2">
      <div className="grid h-44 grid-cols-4 items-center gap-1.5">
        <DistributionBar
          kind="sample"
          label={`${shareLabel} ${formatPercent(shareValue)}`}
          value={shareValue}
        />
        <DistributionBar
          completionValue={entry.emaCompletionFraction}
          kind="success"
          label={successLabel(entry)}
          value={displaySuccessRate(entry) ?? 0}
        />
        <DistributionBar
          kind="episodes"
          label={`${(entry.episodeCount ?? 0).toLocaleString()} episodes · ${formatPercent(entry.episodeShare ?? 0)}`}
          value={entry.episodeShare ?? 0}
        />
        <DistributionBar
          kind="steps"
          label={stepShareLabel(entry, stepMetricLabel, showStepTarget)}
          targetValue={showStepTarget ? entry.targetStepShare : null}
          value={entry.stepShare ?? 0}
        />
      </div>
      <div className="grid gap-0.5">
        <strong className="block text-[13px] leading-tight [overflow-wrap:anywhere]">
          {entry.label}
        </strong>
        <span className="text-[11px] tabular-nums text-app-muted [overflow-wrap:anywhere]">
          {generation === null ? "" : `${generation} · sampling: `}
          {successSummary(entry)}
          {completion === null ? "" : ` · ${completion}`}
          {" · "}
          {(entry.completedEnvSteps ?? 0).toLocaleString()} steps
        </span>
        {regeneration === null ? null : (
          <span className="text-[11px] tabular-nums text-app-muted [overflow-wrap:anywhere]">
            regen: {regeneration}
          </span>
        )}
      </div>
    </div>
  );
}, sameTrackPoolColumnProps);

function effectiveCupShare(cup: TrackPoolCupView, sampleBarUsesTargetShare: boolean) {
  if (!sampleBarUsesTargetShare) {
    return cup.currentProbability;
  }
  return cup.entries.reduce(
    (total, entry) => total + effectiveCourseShare(entry, sampleBarUsesTargetShare),
    0,
  );
}

function effectiveCourseShare(entry: TrackPoolCourseView, sampleBarUsesTargetShare: boolean) {
  if (!sampleBarUsesTargetShare) {
    return entry.currentProbability ?? 0;
  }
  return entry.targetStepShare ?? entry.currentProbability ?? 0;
}

const TRACK_POOL_COURSE_RENDER_KEYS = [
  "completedEnvSteps",
  "currentProbability",
  "emaCompletionFraction",
  "episodeCount",
  "episodeShare",
  "finishedEpisodeCount",
  "generationEmaCompletionFraction",
  "generationEpisodeCount",
  "generationFinishedEpisodeCount",
  "generationSuccessRate",
  "generationSuccessSampleCount",
  "generatedCourseGeneration",
  "generatedCourseSlot",
  "id",
  "label",
  "stepShare",
  "targetStepShare",
  "successRate",
  "successSampleCount",
] as const satisfies readonly (keyof TrackPoolCourseView)[];

function sameTrackPoolColumnProps(
  left: {
    entry: TrackPoolCourseView;
    sampleBarUsesTargetShare: boolean;
    stepMetricLabel: string;
    showStepTarget: boolean;
    xCupRegenerationMinEpisodes: number | null;
    xCupRegenerationThreshold: number | null;
  },
  right: {
    entry: TrackPoolCourseView;
    sampleBarUsesTargetShare: boolean;
    stepMetricLabel: string;
    showStepTarget: boolean;
    xCupRegenerationMinEpisodes: number | null;
    xCupRegenerationThreshold: number | null;
  },
) {
  return (
    left.sampleBarUsesTargetShare === right.sampleBarUsesTargetShare &&
    left.stepMetricLabel === right.stepMetricLabel &&
    left.showStepTarget === right.showStepTarget &&
    left.xCupRegenerationMinEpisodes === right.xCupRegenerationMinEpisodes &&
    left.xCupRegenerationThreshold === right.xCupRegenerationThreshold &&
    TRACK_POOL_COURSE_RENDER_KEYS.every((key) => left.entry[key] === right.entry[key])
  );
}

function stepShareLabel(
  entry: TrackPoolCourseView,
  stepMetricLabel: string,
  showStepTarget: boolean,
) {
  const baseLabel = `${(entry.completedEnvSteps ?? 0).toLocaleString()} ${stepMetricLabel} · ${formatPercent(entry.stepShare ?? 0)}`;
  if (!showStepTarget) {
    return baseLabel;
  }
  return `${baseLabel} · target ${formatPercent(entry.targetStepShare ?? 0)}`;
}

function trackCupTabClass(active: boolean) {
  return cn(
    "inline-flex min-h-[30px] items-center gap-2 border border-app-border bg-app-surface px-2.5 text-xs font-semibold tabular-nums text-app-muted hover:border-app-border-strong hover:text-app-text",
    active
      ? "border-[color-mix(in_srgb,var(--accent)_48%,var(--border-strong))] bg-[color-mix(in_srgb,var(--accent)_10%,var(--surface))] text-app-text"
      : undefined,
  );
}
