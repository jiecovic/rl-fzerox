// web/run-manager/src/entities/engineTuning/ui/RunEngineTuningPanel.tsx
import { useEffect, useId, useMemo, useState } from "react";

import type {
  ConfigMetadata,
  EngineTuningRuntimeBin,
  EngineTuningRuntimeContext,
  EngineTuningRuntimeState,
} from "@/shared/api/contract";

interface RunEngineTuningPanelProps {
  artifact: "latest" | "best" | "final";
  enabled: boolean;
  metadata: ConfigMetadata;
  state: EngineTuningRuntimeState | null;
}

export function RunEngineTuningPanel({
  artifact,
  enabled,
  metadata,
  state,
}: RunEngineTuningPanelProps) {
  const contextSelectId = useId();
  const labels = useMemo(() => engineTuningLabels(metadata), [metadata]);
  const contexts = useMemo(() => sortedContexts(state?.contexts ?? []), [state]);
  const [selectedContextKey, setSelectedContextKey] = useState<string | null>(null);

  useEffect(() => {
    if (contexts.length === 0) {
      setSelectedContextKey(null);
      return;
    }
    if (
      selectedContextKey === null ||
      !contexts.some((context) => context.context_key === selectedContextKey)
    ) {
      setSelectedContextKey(contexts[0]?.context_key ?? null);
    }
  }, [contexts, selectedContextKey]);

  if (!enabled && state === null) {
    return null;
  }

  const selectedContext =
    contexts.find((context) => context.context_key === selectedContextKey) ?? contexts[0] ?? null;
  const observedArmCount = state?.arms.filter((arm) => arm.attempts > 0).length ?? 0;
  const missingContextProjection = state !== null && observedArmCount > 0 && contexts.length === 0;

  return (
    <section className="grid gap-3 border border-app-border bg-app-surface p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="grid gap-1">
          <h3 className="m-0 text-base font-bold text-app-text">Engine tuning</h3>
          <p className="m-0 text-sm text-app-muted">
            Estimated engine-bin selection distribution from the {artifact} checkpoint.
          </p>
        </div>
        <div className="text-right text-xs tabular-nums text-app-muted">
          {state === null
            ? "no samples"
            : `${state.update_count.toLocaleString()} updates · ${contexts.length.toLocaleString()} contexts · ${observedArmCount.toLocaleString()} observed arms`}
        </div>
      </div>
      {selectedContext === null ? (
        <p className="m-0 text-sm text-app-muted">
          {missingContextProjection
            ? "Engine tuning samples exist, but this API response has no distribution projection. Restart the run manager backend to load the current engine-tuning API."
            : enabled
              ? "No engine tuning checkpoint data for this artifact yet. Values appear after a checkpoint save includes adaptive engine samples."
              : "Adaptive engine tuning is disabled for this run."}
        </p>
      ) : (
        <div className="grid gap-3">
          <div className="grid gap-1 md:max-w-[520px]">
            <label
              className="text-xs font-semibold tracking-[0.04em] text-app-muted uppercase"
              htmlFor={contextSelectId}
            >
              Context
            </label>
            <select
              className="min-h-10 border border-app-border bg-app-surface-muted px-3 py-2 text-sm text-app-text"
              id={contextSelectId}
              value={selectedContext.context_key}
              onChange={(event) => setSelectedContextKey(event.currentTarget.value)}
            >
              {contexts.map((context) => (
                <option key={context.context_key} value={context.context_key}>
                  {contextLabel(context, labels)} · {context.attempts} attempts
                </option>
              ))}
            </select>
          </div>
          <div className="grid gap-2">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <strong className="text-sm text-app-text">
                {contextLabel(selectedContext, labels)}
              </strong>
              <span className="text-xs tabular-nums text-app-muted">
                greedy engine {selectedContext.recommended_engine_setting_raw_value} ·{" "}
                {selectedContext.attempts.toLocaleString()} attempts
              </span>
            </div>
            <EngineDistributionChart bins={selectedContext.bins} />
          </div>
          <EngineDistributionTable bins={selectedContext.bins} />
        </div>
      )}
    </section>
  );
}

interface EngineTuningLabels {
  courses: ReadonlyMap<string, string>;
  vehicles: ReadonlyMap<string, string>;
}

function sortedContexts(contexts: readonly EngineTuningRuntimeContext[]) {
  return [...contexts].sort((left, right) => {
    if (right.attempts !== left.attempts) {
      return right.attempts - left.attempts;
    }
    return left.context_key.localeCompare(right.context_key);
  });
}

function engineTuningLabels(metadata: ConfigMetadata): EngineTuningLabels {
  return {
    courses: new Map([
      ...metadata.built_in_courses.map((course) => [course.id, course.display_name] as const),
      ["x_cup", "X Cup"],
    ]),
    vehicles: new Map(
      metadata.vehicles.map((vehicle) => [vehicle.id, vehicle.display_name] as const),
    ),
  };
}

function contextLabel(context: EngineTuningRuntimeContext, labels: EngineTuningLabels) {
  const courseLabel = labels.courses.get(context.course_key) ?? humanizeKey(context.course_key);
  const vehicleLabel = labels.vehicles.get(context.vehicle_id) ?? humanizeKey(context.vehicle_id);
  return `${courseLabel} · ${vehicleLabel}`;
}

function EngineDistributionChart({ bins }: { bins: readonly EngineTuningRuntimeBin[] }) {
  const maxProbability = Math.max(...bins.map((bin) => bin.selection_probability), 0.000001);
  const highestProbability = Math.max(...bins.map((bin) => bin.selection_probability), 0);
  const firstBin = bins[0]?.engine_setting_raw_value ?? 0;
  const lastBin = bins.at(-1)?.engine_setting_raw_value ?? 100;

  return (
    <div className="grid gap-1">
      <div
        className="grid h-28 items-end gap-px border border-app-border bg-app-surface-muted p-2"
        style={{ gridTemplateColumns: `repeat(${bins.length}, minmax(2px, 1fr))` }}
      >
        {bins.map((bin) => {
          const isPeak = bin.selection_probability === highestProbability;
          return (
            <div
              className="flex h-full items-end"
              key={bin.engine_setting_raw_value}
              title={`engine ${bin.engine_setting_raw_value} · ${formatPercent(bin.selection_probability)} probability · score ${formatNullable(bin.posterior_mean)} · ${bin.attempts} attempts`}
            >
              <div
                className={
                  isPeak ? "w-full bg-app-accent" : "w-full bg-app-accent/45 hover:bg-app-accent/70"
                }
                style={{
                  height: `${Math.max(3, (bin.selection_probability / maxProbability) * 100)}%`,
                }}
              />
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-xs tabular-nums text-app-muted">
        <span>{firstBin}</span>
        <span>engine raw value</span>
        <span>{lastBin}</span>
      </div>
    </div>
  );
}

function EngineDistributionTable({ bins }: { bins: readonly EngineTuningRuntimeBin[] }) {
  const topBins = [...bins]
    .sort((left, right) => {
      if (right.selection_probability !== left.selection_probability) {
        return right.selection_probability - left.selection_probability;
      }
      return right.posterior_mean - left.posterior_mean;
    })
    .slice(0, 8);

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full border-collapse text-left text-xs tabular-nums">
        <thead className="text-app-muted">
          <tr className="border-b border-app-border">
            <th className="py-1.5 pr-3 font-semibold">Engine</th>
            <th className="py-1.5 pr-3 font-semibold">Prob</th>
            <th className="py-1.5 pr-3 font-semibold">Score</th>
            <th className="py-1.5 pr-3 font-semibold">Finish</th>
            <th className="py-1.5 pr-3 font-semibold">Comp</th>
            <th className="py-1.5 font-semibold">Attempts</th>
          </tr>
        </thead>
        <tbody>
          {topBins.map((bin) => (
            <tr
              className="border-b border-app-border/70 last:border-b-0"
              key={bin.engine_setting_raw_value}
            >
              <td className="py-1.5 pr-3 text-app-text">{bin.engine_setting_raw_value}</td>
              <td className="py-1.5 pr-3 text-app-text">
                {formatPercent(bin.selection_probability)}
              </td>
              <td className="py-1.5 pr-3 text-app-text">{formatNullable(bin.posterior_mean)}</td>
              <td className="py-1.5 pr-3 text-app-text">{formatPercent(bin.finish_rate)}</td>
              <td className="py-1.5 pr-3 text-app-text">{formatPercent(bin.mean_completion)}</td>
              <td className="py-1.5 text-app-text">{bin.attempts}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function humanizeKey(key: string) {
  return key
    .split("_")
    .filter((part) => part.length > 0)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join(" ");
}

function formatNullable(value: number | null) {
  return value === null ? "-" : value.toFixed(3);
}

function formatPercent(value: number | null) {
  return value === null ? "-" : `${(value * 100).toFixed(1)}%`;
}
