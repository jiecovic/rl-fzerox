// src/rl_fzerox/apps/run_manager/web/src/features/runs/charts_panel/RunChartSelectionPanel.tsx
import { useCallback } from "react";
import {
  chartSeriesColor,
  type LineageRunGroup,
  type LineageSelectionState,
  lineageSelectionState,
} from "@/features/runs/charts_panel/model";
import type { ManagedRun } from "@/shared/api/contract";
import { cn } from "@/shared/ui/cn";
import { ChevronIcon } from "@/shared/ui/icons";
import { Notice } from "@/shared/ui/Panel";
import { AppTooltip } from "@/shared/ui/Tooltip";

interface RunChartSelectionPanelProps {
  colorByRunId: ReadonlyMap<string, string>;
  onToggleCollapsedLineage: (lineageId: string) => void;
  onToggleLineage: (lineageRuns: readonly ManagedRun[]) => void;
  onToggleRun: (runId: string) => void;
  openLineageById: Record<string, boolean>;
  selectedRunIds: readonly string[];
  selectionLineageGroups: readonly LineageRunGroup[];
  totalRuns: number;
}

export function RunChartSelectionPanel({
  colorByRunId,
  onToggleCollapsedLineage,
  onToggleLineage,
  onToggleRun,
  openLineageById,
  selectedRunIds,
  selectionLineageGroups,
  totalRuns,
}: RunChartSelectionPanelProps) {
  return (
    <div className="run-chart-selection-panel mb-[18px] grid gap-2.5 border border-app-border bg-app-surface-muted p-3.5">
      <div className="flex items-baseline justify-between gap-3">
        <strong>Selected runs</strong>
        <span className="text-xs text-app-muted">
          {selectedRunIds.length === 0 ? "none" : `${selectedRunIds.length} active`}
        </span>
      </div>
      {totalRuns === 0 ? (
        <Notice>No launched runs yet.</Notice>
      ) : (
        <div className="grid gap-2.5">
          {selectionLineageGroups.map((group) => {
            const selectionState = lineageSelectionState(group.runs, selectedRunIds);
            const lineageOpen = openLineageById[group.lineageId] ?? true;
            return (
              <section
                aria-label={`${group.label} lineage runs`}
                className={selectionLineageClass(selectionState)}
                key={group.lineageId}
              >
                <div className="flex items-center justify-between gap-3 border-b border-[color-mix(in_srgb,var(--border)_88%,transparent)] pb-1.5">
                  <button
                    aria-expanded={lineageOpen}
                    aria-label={`${lineageOpen ? "Collapse" : "Expand"} lineage ${group.label}`}
                    className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)] items-center gap-2.5 border-0 bg-transparent p-0 text-left text-app-text"
                    type="button"
                    onClick={() => onToggleCollapsedLineage(group.lineageId)}
                  >
                    <span
                      aria-hidden="true"
                      className={
                        lineageOpen ? "run-lineage-chevron is-open" : "run-lineage-chevron"
                      }
                    >
                      <ChevronIcon />
                    </span>
                    <span className="grid min-w-0 gap-0.5">
                      <strong className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap text-[13px]">
                        {group.label}
                      </strong>
                      <span className="text-xs text-app-muted">
                        {group.runs.length === group.totalRunCount
                          ? `${group.totalRunCount} runs`
                          : `${group.runs.length}/${group.totalRunCount} runs`}
                      </span>
                    </span>
                  </button>
                  <div className="inline-flex items-center">
                    <LineageSelectionCheckbox
                      label={group.label}
                      state={selectionState}
                      onChange={() => onToggleLineage(group.runs)}
                    />
                  </div>
                </div>
                {lineageOpen ? (
                  <div className="grid grid-cols-2 gap-2 pt-0.5 max-[1100px]:grid-cols-1">
                    {group.runs.map((run, index) => {
                      const checked = selectedRunIds.includes(run.id);
                      const stroke = colorByRunId.get(run.id) ?? chartSeriesColor(index);
                      return (
                        <label
                          className="run-chart-selection-row grid grid-cols-[auto_auto_minmax(0,1fr)] items-start gap-2.5 border border-app-border bg-app-surface px-3 py-2.5"
                          key={run.id}
                        >
                          <input
                            checked={checked}
                            type="checkbox"
                            onChange={() => onToggleRun(run.id)}
                          />
                          <span
                            aria-hidden="true"
                            className="run-chart-selection-swatch mt-1 h-2.5 w-2.5"
                            style={{ background: stroke }}
                          />
                          <span className="run-chart-selection-copy grid min-w-0 gap-1">
                            <strong>{run.name}</strong>
                            <span className="text-xs tabular-nums text-app-muted">
                              {run.status}
                              {run.runtime === null
                                ? ""
                                : ` · ${(run.runtime.progress_fraction * 100).toFixed(1)}% · ${run.runtime.num_timesteps.toLocaleString()} steps`}
                            </span>
                          </span>
                        </label>
                      );
                    })}
                  </div>
                ) : null}
              </section>
            );
          })}
        </div>
      )}
    </div>
  );
}

function LineageSelectionCheckbox({
  label,
  onChange,
  state,
}: {
  label: string;
  onChange: () => void;
  state: LineageSelectionState;
}) {
  const checked = state === "all";
  const inputRef = useCallback(
    (element: HTMLInputElement | null) => {
      if (element !== null) {
        element.indeterminate = state === "partial";
      }
    },
    [state],
  );

  return (
    <AppTooltip content={`Select lineage ${label}`}>
      <label className="inline-flex h-[18px] w-[18px] cursor-pointer items-center justify-center">
        <input
          aria-label={`Select lineage ${label}`}
          className="m-0 h-3.5 w-3.5 accent-app-accent"
          checked={checked}
          ref={inputRef}
          type="checkbox"
          onChange={onChange}
        />
      </label>
    </AppTooltip>
  );
}

function selectionLineageClass(state: LineageSelectionState) {
  return cn(
    "grid gap-2 border border-app-border bg-app-surface px-3 py-2.5 pb-3",
    state === "partial"
      ? "border-[color-mix(in_srgb,var(--accent)_30%,var(--border))] bg-[color-mix(in_srgb,var(--surface)_92%,var(--accent)_8%)]"
      : undefined,
    state === "all"
      ? "border-[color-mix(in_srgb,var(--accent)_50%,var(--border))] bg-[color-mix(in_srgb,var(--surface)_86%,var(--accent)_14%)]"
      : undefined,
  );
}
