// web/run-manager/src/entities/trackPool/ui/TrackPoolParts.tsx
import { type KeyboardEvent, memo } from "react";
import type { TrackPoolCourseView, TrackPoolCupView } from "@/entities/trackPool/model/types";
import {
  completionLabel,
  completionSummary,
  displaySuccessRate,
  formatOptionalPercent,
  formatPercent,
  samplerSignalSummary,
  shortCupLabel,
  successLabel,
  successSummary,
  xCupGenerationSummary,
  xCupRegenerationSummary,
} from "@/entities/trackPool/model/view";
import { cn } from "@/shared/ui/cn";
import { AppTooltip } from "@/shared/ui/Tooltip";

interface DistributionBarProps {
  kind:
    | "sample"
    | "success"
    | "completion"
    | "completion-ema"
    | "finish-ema"
    | "episodes"
    | "steps"
    | "signal";
  label: string;
  targetValue?: number | null;
  value: number;
}

export const DistributionBar = memo(function DistributionBar({
  kind,
  label,
  targetValue,
  value,
}: DistributionBarProps) {
  const clampedTarget =
    targetValue === undefined || targetValue === null
      ? null
      : Math.max(0, Math.min(targetValue, 1));
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
        </div>
      </button>
    </AppTooltip>
  );
});

interface LegendItemProps {
  kind:
    | "sample"
    | "success"
    | "episodes"
    | "steps"
    | "signal"
    | "finish-ema"
    | "completion"
    | "completion-ema"
    | "target";
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
  deficitBudgetDifficultyMetric,
  deficitBudgetWarmupMinEpisodesPerCourse,
  sampleBarUsesTargetShare,
  stepMetricLabel,
  showStepTarget,
  xCupRegenerationMinEpisodes,
  xCupRegenerationThreshold,
}: {
  activeCup: TrackPoolCupView;
  deficitBudgetDifficultyMetric: string | null;
  deficitBudgetWarmupMinEpisodesPerCourse: number | null;
  sampleBarUsesTargetShare: boolean;
  stepMetricLabel: string;
  showStepTarget: boolean;
  xCupRegenerationMinEpisodes: number | null;
  xCupRegenerationThreshold: number | null;
}) {
  const shareLabel = sampleBarUsesTargetShare ? "Target step share" : "Sample";
  const signalLabel = deficitBudgetSignalLabel(deficitBudgetDifficultyMetric);
  const usesDeficitBudget = signalLabel !== null;
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
          {usesDeficitBudget ? (
            <span>
              Problem score {signalLabel}
              {deficitBudgetWarmupMinEpisodesPerCourse === null
                ? ""
                : ` · warmup ${deficitBudgetWarmupMinEpisodesPerCourse}/course`}
            </span>
          ) : null}
        </div>
      </div>
      {usesDeficitBudget ? (
        <div className="grid gap-4">
          <TrackPoolChart
            activeCup={activeCup}
            heading="Global statistics"
            sampleBarUsesTargetShare={sampleBarUsesTargetShare}
            samplerSignalLabel={signalLabel}
            showStepTarget={showStepTarget}
            stepMetricLabel="env steps"
            variant="global"
            xCupRegenerationMinEpisodes={xCupRegenerationMinEpisodes}
            xCupRegenerationThreshold={xCupRegenerationThreshold}
          />
          <TrackPoolChart
            activeCup={activeCup}
            heading="EMA adaptive signals"
            sampleBarUsesTargetShare={sampleBarUsesTargetShare}
            samplerSignalLabel={signalLabel}
            showStepTarget={false}
            stepMetricLabel={stepMetricLabel}
            variant="ema"
            xCupRegenerationMinEpisodes={xCupRegenerationMinEpisodes}
            xCupRegenerationThreshold={xCupRegenerationThreshold}
          />
        </div>
      ) : (
        <TrackPoolChart
          activeCup={activeCup}
          heading={null}
          sampleBarUsesTargetShare={sampleBarUsesTargetShare}
          samplerSignalLabel={null}
          showStepTarget={showStepTarget}
          stepMetricLabel={stepMetricLabel}
          variant="combined"
          xCupRegenerationMinEpisodes={xCupRegenerationMinEpisodes}
          xCupRegenerationThreshold={xCupRegenerationThreshold}
        />
      )}
    </div>
  );
}

function TrackPoolChart({
  activeCup,
  heading,
  sampleBarUsesTargetShare,
  samplerSignalLabel,
  showStepTarget,
  stepMetricLabel,
  variant,
  xCupRegenerationMinEpisodes,
  xCupRegenerationThreshold,
}: {
  activeCup: TrackPoolCupView;
  heading: string | null;
  sampleBarUsesTargetShare: boolean;
  samplerSignalLabel: string | null;
  showStepTarget: boolean;
  stepMetricLabel: string;
  variant: TrackPoolChartVariant;
  xCupRegenerationMinEpisodes: number | null;
  xCupRegenerationThreshold: number | null;
}) {
  const shareLabel = sampleBarUsesTargetShare ? "Target step share" : "Sample";
  return (
    <section className="grid gap-3">
      {heading === null ? null : (
        <div className="flex items-baseline justify-between gap-3 border-t border-app-border pt-3">
          <strong className="text-sm">{heading}</strong>
          <span className="text-xs text-app-muted">
            {variant === "global" ? "episode and step accounting" : "scheduler input signals"}
          </span>
        </div>
      )}
      <div className="flex flex-wrap gap-3">
        {variant === "combined" ? <LegendItem kind="sample" label={shareLabel} /> : null}
        {variant !== "ema" ? (
          <>
            <LegendItem kind="success" label="Finish" />
            <LegendItem kind="completion" label="Completion" />
            <LegendItem kind="episodes" label="Episodes" />
          </>
        ) : null}
        {variant !== "ema" ? <LegendItem kind="steps" label="Env steps" /> : null}
        {variant === "ema" ? (
          <>
            <LegendItem kind="completion-ema" label="Completion EMA" />
            <LegendItem kind="finish-ema" label="Finish EMA" />
            <LegendItem kind="signal" label="Problem score" />
          </>
        ) : null}
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
              samplerSignalLabel={samplerSignalLabel}
              stepMetricLabel={stepMetricLabel}
              showStepTarget={showStepTarget}
              variant={variant}
              xCupRegenerationMinEpisodes={xCupRegenerationMinEpisodes}
              xCupRegenerationThreshold={xCupRegenerationThreshold}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

type TrackPoolChartVariant = "combined" | "global" | "ema";

const TrackPoolColumn = memo(function TrackPoolColumn({
  entry,
  sampleBarUsesTargetShare,
  samplerSignalLabel,
  stepMetricLabel,
  showStepTarget,
  variant,
  xCupRegenerationMinEpisodes,
  xCupRegenerationThreshold,
}: {
  entry: TrackPoolCourseView;
  sampleBarUsesTargetShare: boolean;
  samplerSignalLabel: string | null;
  stepMetricLabel: string;
  showStepTarget: boolean;
  variant: TrackPoolChartVariant;
  xCupRegenerationMinEpisodes: number | null;
  xCupRegenerationThreshold: number | null;
}) {
  const completion = completionSummary(entry);
  const samplerSignal = samplerSignalSummary(entry);
  const showSamplerSignalText = variant === "ema";
  const generation = xCupGenerationSummary(entry);
  const regeneration = xCupRegenerationSummary(
    entry,
    xCupRegenerationThreshold,
    xCupRegenerationMinEpisodes,
  );
  const shareLabel = sampleBarUsesTargetShare ? "Target step share" : "Sample";
  const shareValue = effectiveCourseShare(entry, sampleBarUsesTargetShare);
  const bars = trackPoolBars({
    entry,
    samplerSignalLabel,
    shareLabel,
    shareValue,
    showStepTarget,
    stepMetricLabel,
    variant,
  });
  return (
    <div className="grid min-w-0 gap-2">
      <div className={cn("grid h-44 items-center gap-1.5", columnClassForBarCount(bars.length))}>
        {bars.map((bar) => (
          <DistributionBar {...bar} key={bar.kind} />
        ))}
      </div>
      <div className="grid gap-0.5">
        <strong className="block text-[13px] leading-tight [overflow-wrap:anywhere]">
          {entry.label}
        </strong>
        <span className="text-[11px] tabular-nums text-app-muted [overflow-wrap:anywhere]">
          {generation === null ? "" : `${generation} · sampling: `}
          {successSummary(entry)}
          {completion === null ? "" : ` · ${completion}`}
          {showSamplerSignalText && samplerSignal !== null ? ` · ${samplerSignal}` : ""}
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

function trackPoolBars({
  entry,
  samplerSignalLabel,
  shareLabel,
  shareValue,
  showStepTarget,
  stepMetricLabel,
  variant,
}: {
  entry: TrackPoolCourseView;
  samplerSignalLabel: string | null;
  shareLabel: string;
  shareValue: number;
  showStepTarget: boolean;
  stepMetricLabel: string;
  variant: TrackPoolChartVariant;
}): DistributionBarProps[] {
  if (variant === "global") {
    return globalStatisticBars(entry, stepMetricLabel, showStepTarget);
  }
  if (variant === "ema") {
    return [
      {
        kind: "completion-ema",
        label: `completion EMA ${formatOptionalPercent(entry.emaCompletionFraction)}`,
        value: entry.emaCompletionFraction ?? 0,
      },
      {
        kind: "finish-ema",
        label: `finish EMA ${formatOptionalPercent(entry.emaFinishRate)}`,
        value: entry.emaFinishRate ?? 0,
      },
      {
        kind: "signal",
        label: `${samplerSignalLabel ?? "sampler"} problem score ${formatPercent(entry.currentProblemScore ?? 0)}`,
        value: entry.currentProblemScore ?? 0,
      },
    ];
  }
  return [
    {
      kind: "sample",
      label: `${shareLabel} ${formatPercent(shareValue)}`,
      value: shareValue,
    },
    ...globalStatisticBars(entry, stepMetricLabel, showStepTarget),
  ];
}

function globalStatisticBars(
  entry: TrackPoolCourseView,
  stepMetricLabel: string,
  showStepTarget: boolean,
): DistributionBarProps[] {
  return [
    {
      kind: "success",
      label: successLabel(entry),
      value: displaySuccessRate(entry) ?? 0,
    },
    {
      kind: "completion",
      label: completionLabel(entry),
      value: entry.completionRate ?? 0,
    },
    {
      kind: "episodes",
      label: `${(entry.episodeCount ?? 0).toLocaleString()} episodes · ${formatPercent(entry.episodeShare ?? 0)}`,
      value: entry.episodeShare ?? 0,
    },
    {
      kind: "steps",
      label: stepShareLabel(entry, stepMetricLabel, showStepTarget),
      targetValue: showStepTarget ? entry.targetStepShare : null,
      value: entry.stepShare ?? 0,
    },
  ];
}

function columnClassForBarCount(count: number) {
  if (count <= 3) {
    return "grid-cols-3";
  }
  if (count === 4) {
    return "grid-cols-4";
  }
  return "grid-cols-5";
}

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
  "completionFractionTotal",
  "completionRate",
  "completionSampleCount",
  "currentProbability",
  "currentProblemScore",
  "emaCompletionFraction",
  "emaFinishRate",
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
    samplerSignalLabel: string | null;
    stepMetricLabel: string;
    showStepTarget: boolean;
    variant: TrackPoolChartVariant;
    xCupRegenerationMinEpisodes: number | null;
    xCupRegenerationThreshold: number | null;
  },
  right: {
    entry: TrackPoolCourseView;
    sampleBarUsesTargetShare: boolean;
    samplerSignalLabel: string | null;
    stepMetricLabel: string;
    showStepTarget: boolean;
    variant: TrackPoolChartVariant;
    xCupRegenerationMinEpisodes: number | null;
    xCupRegenerationThreshold: number | null;
  },
) {
  return (
    left.sampleBarUsesTargetShare === right.sampleBarUsesTargetShare &&
    left.samplerSignalLabel === right.samplerSignalLabel &&
    left.stepMetricLabel === right.stepMetricLabel &&
    left.showStepTarget === right.showStepTarget &&
    left.variant === right.variant &&
    left.xCupRegenerationMinEpisodes === right.xCupRegenerationMinEpisodes &&
    left.xCupRegenerationThreshold === right.xCupRegenerationThreshold &&
    TRACK_POOL_COURSE_RENDER_KEYS.every((key) => left.entry[key] === right.entry[key])
  );
}

function deficitBudgetSignalLabel(metric: string | null) {
  if (metric === "finish_ema") {
    return "finish EMA";
  }
  if (metric === "mixed") {
    return "mixed";
  }
  if (metric === "completion_ema") {
    return "completion EMA";
  }
  return null;
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
