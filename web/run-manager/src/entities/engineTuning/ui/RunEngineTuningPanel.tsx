// web/run-manager/src/entities/engineTuning/ui/RunEngineTuningPanel.tsx
import { useEffect, useId, useMemo, useState } from "react";

import type { ConfigMetadata, EngineTuningRuntimeState } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { ConfigDisclosure } from "@/shared/ui/config/ConfigDisclosure";
import { ResetIcon } from "@/shared/ui/icons";

import {
  EngineMeanPerformanceBars,
  EngineSamplingProbabilityBars,
} from "./runEngineTuningPanel/charts";
import { engineStepLabel } from "./runEngineTuningPanel/format";
import {
  backendLabel,
  contextLabel,
  engineTuningLabels,
  engineTuningViewMode,
  measuredCandidatesForContext,
  objectiveCountLabel,
  sortedContexts,
} from "./runEngineTuningPanel/labels";
import {
  EngineBanditBucketTable,
  EngineMeasuredCandidateTable,
  EngineModelCandidateTable,
} from "./runEngineTuningPanel/tables";

interface RunEngineTuningPanelProps {
  artifact: "latest" | "best" | "final";
  canReset: boolean;
  enabled: boolean;
  expanded: boolean;
  isResetting: boolean;
  metadata: ConfigMetadata;
  state: EngineTuningRuntimeState | null;
  onExpandedChange: (expanded: boolean) => void;
  onReset: () => void;
}

export function RunEngineTuningPanel({
  artifact,
  canReset,
  enabled,
  expanded,
  isResetting,
  metadata,
  state,
  onExpandedChange,
  onReset,
}: RunEngineTuningPanelProps) {
  const contextSelectId = useId();
  const labels = useMemo(() => engineTuningLabels(metadata), [metadata]);
  const contexts = useMemo(() => sortedContexts(state?.contexts ?? [], labels), [state, labels]);
  const [confirmResetOpen, setConfirmResetOpen] = useState(false);
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
  const backend = state?.model_backend ?? null;
  const objective = state?.objective ?? "finish_time";
  const viewMode = engineTuningViewMode(backend);
  const observedCandidateCount = state?.candidates.length ?? 0;
  const statusText =
    state === null
      ? "no checkpoint data"
      : state.model_backend === "mlp_ensemble"
        ? `${backendLabel(state.model_backend)} · ${state.update_count.toLocaleString()} updates · ${contexts.length.toLocaleString()} contexts`
        : `${backendLabel(state.model_backend)} · ${state.update_count.toLocaleString()} updates · ${contexts.length.toLocaleString()} contexts · ${observedCandidateCount.toLocaleString()} aggregate candidates`;

  return (
    <>
      <ConfigDisclosure
        defaultOpen={false}
        open={expanded}
        title="Engine tuning"
        onToggle={onExpandedChange}
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="grid gap-1">
            <p className="m-0 text-sm text-app-muted">
              Reset-time engine sampling probabilities from the {artifact} checkpoint.
            </p>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            <span className="text-right text-xs tabular-nums text-app-muted">{statusText}</span>
            <Button
              className="h-8 gap-1.5 px-3 text-xs"
              disabled={!canReset || !enabled || isResetting}
              tone="danger"
              onClick={() => setConfirmResetOpen(true)}
            >
              <ResetIcon />
              <span>{isResetting ? "Resetting" : "Reset tuner"}</span>
            </Button>
          </div>
        </div>
        {selectedContext === null ? (
          <p className="m-0 text-sm text-app-muted">
            {enabled
              ? "No engine tuning checkpoint data for this artifact yet. Values appear after a checkpoint save includes successful finish samples."
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
                    {contextLabel(context, labels)} · {objectiveCountLabel(context, objective)}
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
                  {selectedContext.model_ready ? "greedy engine" : "warmup engine"}{" "}
                  {engineStepLabel(selectedContext.recommended_engine_setting_raw_value)} ·{" "}
                  {objectiveCountLabel(selectedContext, objective)}
                </span>
              </div>
              <EngineSamplingProbabilityBars
                candidates={selectedContext.candidates}
                mode={viewMode}
                objective={objective}
              />
              <EngineMeanPerformanceBars
                candidates={selectedContext.candidates}
                mode={viewMode}
                objective={objective}
                recommendedEngineSettingRawValue={
                  selectedContext.recommended_engine_setting_raw_value
                }
              />
            </div>
            {backend === "bandit" ? (
              <EngineBanditBucketTable
                candidates={selectedContext.candidates}
                objective={objective}
              />
            ) : backend === "gaussian_process" ? (
              <EngineMeasuredCandidateTable
                estimates={selectedContext.candidates}
                measuredCandidates={measuredCandidatesForContext(
                  state?.candidates ?? [],
                  selectedContext.context_key,
                )}
              />
            ) : selectedContext.model_ready ? (
              <EngineModelCandidateTable candidates={selectedContext.candidates} />
            ) : null}
          </div>
        )}
      </ConfigDisclosure>
      <ConfirmDialog
        busy={isResetting}
        busyLabel="Resetting..."
        confirmLabel="Reset tuner"
        description="Remove engine-tuning checkpoint sidecars for this run? Policy checkpoints remain, and resume will start the tuner from scratch."
        open={confirmResetOpen}
        title="Reset engine tuner"
        onClose={() => setConfirmResetOpen(false)}
        onConfirm={() => {
          setConfirmResetOpen(false);
          onReset();
        }}
      />
    </>
  );
}
