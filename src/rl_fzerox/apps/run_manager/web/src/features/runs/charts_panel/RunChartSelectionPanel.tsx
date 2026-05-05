// src/rl_fzerox/apps/run_manager/web/src/features/runs/charts_panel/RunChartSelectionPanel.tsx
import { useCallback } from "react";
import type { ManagedRun } from "@/shared/api/contract";
import { ChevronIcon } from "@/shared/ui/icons";
import { Notice } from "@/shared/ui/Panel";

import {
  chartSeriesColor,
  type LineageRunGroup,
  type LineageSelectionState,
  lineageSelectionState,
} from "./model";

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
    <div className="run-chart-selection-panel">
      <div className="run-chart-selection-header">
        <strong>Selected runs</strong>
        <span>{selectedRunIds.length === 0 ? "none" : `${selectedRunIds.length} active`}</span>
      </div>
      {totalRuns === 0 ? (
        <Notice>No launched runs yet.</Notice>
      ) : (
        <div className="run-chart-selection-list">
          {selectionLineageGroups.map((group) => {
            const selectionState = lineageSelectionState(group.runs, selectedRunIds);
            const lineageOpen = openLineageById[group.lineageId] ?? true;
            return (
              <section
                aria-label={`${group.label} lineage runs`}
                className="run-chart-lineage-group run-chart-selection-lineage-group"
                data-selection-state={selectionState}
                key={group.lineageId}
              >
                <div className="run-chart-lineage-header">
                  <button
                    aria-expanded={lineageOpen}
                    aria-label={`${lineageOpen ? "Collapse" : "Expand"} lineage ${group.label}`}
                    className="run-chart-lineage-disclosure"
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
                    <span className="run-chart-lineage-heading">
                      <strong>{group.label}</strong>
                      <span className="run-chart-lineage-meta">
                        {group.runs.length === group.totalRunCount
                          ? `${group.totalRunCount} runs`
                          : `${group.runs.length}/${group.totalRunCount} runs`}
                      </span>
                    </span>
                  </button>
                  <div className="run-chart-lineage-header-actions">
                    <LineageSelectionCheckbox
                      label={group.label}
                      state={selectionState}
                      onChange={() => onToggleLineage(group.runs)}
                    />
                  </div>
                </div>
                {lineageOpen ? (
                  <div className="run-chart-lineage-run-grid">
                    {group.runs.map((run, index) => {
                      const checked = selectedRunIds.includes(run.id);
                      const stroke = colorByRunId.get(run.id) ?? chartSeriesColor(index);
                      return (
                        <label className="run-chart-selection-row" key={run.id}>
                          <input
                            checked={checked}
                            type="checkbox"
                            onChange={() => onToggleRun(run.id)}
                          />
                          <span
                            aria-hidden="true"
                            className="run-chart-selection-swatch"
                            style={{ background: stroke }}
                          />
                          <span className="run-chart-selection-copy">
                            <strong>{run.name}</strong>
                            <span>
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
    <label className="run-chart-lineage-toggle" title={label}>
      <input
        aria-label={`Select lineage ${label}`}
        checked={checked}
        ref={inputRef}
        type="checkbox"
        onChange={onChange}
      />
    </label>
  );
}
