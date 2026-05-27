// src/rl_fzerox/apps/run_manager/web/src/features/runs/track_pool_panel/parts.tsx
import type { KeyboardEvent } from "react";
import {
  completionSummary,
  formatOptionalPercent,
  formatPercent,
  shortCupLabel,
  successLabel,
  successSummary,
} from "@/features/runs/track_pool_panel/model";
import type { TrackPoolCourseView, TrackPoolCupView } from "@/features/runs/track_pool_panel/types";
import { cn } from "@/shared/ui/cn";
import { AppTooltip } from "@/shared/ui/Tooltip";

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
}

interface LegendItemProps {
  kind: "sample" | "success" | "episodes" | "steps" | "target";
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

export function TrackPoolBody({ activeCup }: { activeCup: TrackPoolCupView }) {
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
          <span>Sample {formatPercent(activeCup.currentProbability)}</span>
          <span>Finish {formatOptionalPercent(activeCup.successRate)}</span>
          <span>Episodes {formatPercent(activeCup.episodeShare)}</span>
          <span>Env steps {formatPercent(activeCup.stepShare)}</span>
        </div>
      </div>
      <div className="mb-3 flex flex-wrap gap-3">
        <LegendItem kind="sample" label="Sample" />
        <LegendItem kind="success" label="Finish" />
        <LegendItem kind="episodes" label="Episodes" />
        <LegendItem kind="steps" label="Env steps" />
        <LegendItem kind="target" label="Step target" />
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
            <TrackPoolColumn entry={entry} key={entry.id} />
          ))}
        </div>
      </div>
    </div>
  );
}

function TrackPoolColumn({ entry }: { entry: TrackPoolCourseView }) {
  const completion = completionSummary(entry);
  return (
    <div className="grid min-w-0 gap-2">
      <div className="grid h-44 grid-cols-4 items-center gap-1.5">
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
      <div className="grid gap-0.5">
        <strong className="block text-[13px] leading-tight [overflow-wrap:anywhere]">
          {entry.label}
        </strong>
        <span className="text-[11px] tabular-nums text-app-muted [overflow-wrap:anywhere]">
          {successSummary(entry)}
          {completion === null ? "" : ` · ${completion}`}
          {" · "}
          {(entry.completedEnvSteps ?? 0).toLocaleString()} steps
        </span>
      </div>
    </div>
  );
}

function trackCupTabClass(active: boolean) {
  return cn(
    "inline-flex min-h-[30px] items-center gap-2 border border-app-border bg-app-surface px-2.5 text-xs font-semibold tabular-nums text-app-muted hover:border-app-border-strong hover:text-app-text",
    active
      ? "border-[color-mix(in_srgb,var(--accent)_48%,var(--border-strong))] bg-[color-mix(in_srgb,var(--accent)_10%,var(--surface))] text-app-text"
      : undefined,
  );
}
