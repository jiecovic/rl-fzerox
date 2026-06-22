// web/run-manager/src/pages/evaluations/EvaluationsPanel.tsx
import { useCallback, useEffect, useMemo, useState } from "react";

import type { ConfigSetter } from "@/entities/runConfig/model/state";
import { TracksSection } from "@/entities/runConfig/ui/sections/tracks/TracksSection";
import { VehicleSection } from "@/entities/runConfig/ui/sections/VehicleSection";
import {
  buildEvaluationPresets,
  clonePresetConfig,
  type EvaluationPresetId,
  evaluationTargetFromConfig,
} from "@/pages/evaluations/presets";
import type {
  ConfigMetadata,
  CreateEvaluationRequest,
  EvaluationMode,
  EvaluationSourceArtifact,
  ManagedEvaluation,
  ManagedRun,
  ManagedRunConfig,
  ManagedRunDetail,
  PolicyPlaybackMode,
} from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { ConfigStack } from "@/shared/ui/config/ConfigLayout";
import { FieldInput, FieldSelect, FieldShell } from "@/shared/ui/Field";
import { formatDate } from "@/shared/ui/format";
import { EvaluationTabIcon, PlusIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

interface EvaluationsPanelProps {
  defaultConfig: ManagedRunConfig | null;
  evaluationError: string | null;
  evaluations: ManagedEvaluation[];
  loadRunDetail: (runId: string) => Promise<ManagedRunDetail>;
  metadata: ConfigMetadata | null;
  runDetailsById: Record<string, ManagedRunDetail>;
  runs: ManagedRun[];
  sourceRunId: string | null;
  onCreateEvaluation: (request: CreateEvaluationRequest) => Promise<ManagedEvaluation>;
  onGlobalError: (message: string | null) => void;
}

type SourceArtifactChoice = Extract<EvaluationSourceArtifact, "latest" | "best">;

const TARGET_MODE_LABELS: Record<EvaluationMode, string> = {
  gp_cup: "GP cup",
  time_attack: "Time attack",
};

export function EvaluationsPanel({
  defaultConfig,
  evaluationError,
  evaluations,
  loadRunDetail,
  metadata,
  onCreateEvaluation,
  onGlobalError,
  runDetailsById,
  runs,
  sourceRunId,
}: EvaluationsPanelProps) {
  const [selectedRunId, setSelectedRunId] = useState(runs[0]?.id ?? "");
  const [sourceArtifact, setSourceArtifact] = useState<SourceArtifactChoice>("latest");
  const [policyMode, setPolicyMode] = useState<PolicyPlaybackMode>("deterministic");
  const [presetId, setPresetId] = useState<EvaluationPresetId>("time_attack_blue_falcon");
  const [appliedPresetKey, setAppliedPresetKey] = useState<string | null>(null);
  const [presetConfig, setPresetConfig] = useState<ManagedRunConfig | null>(null);
  const [repeatText, setRepeatText] = useState("3");
  const [seedText, setSeedText] = useState(() => String(randomSeed()));
  const [nameText, setNameText] = useState("");
  const [nameEdited, setNameEdited] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;
  const selectedRunDetail = runDetailsById[selectedRunId] ?? null;
  const presets = useMemo(
    () =>
      defaultConfig === null || metadata === null
        ? []
        : buildEvaluationPresets({
            defaultConfig,
            metadata,
            sourceRun: selectedRunDetail,
          }),
    [defaultConfig, metadata, selectedRunDetail],
  );
  const selectedPreset = presets.find((preset) => preset.id === presetId) ?? presets[0] ?? null;
  const defaultName = useMemo(
    () =>
      selectedRun === null
        ? "evaluation"
        : `${selectedRun.name} · ${selectedPreset?.label ?? "evaluation"}`,
    [selectedPreset?.label, selectedRun],
  );
  const setEvaluationConfig: ConfigSetter = useCallback((nextConfig) => {
    setPresetConfig((currentConfig) => {
      if (currentConfig === null) {
        return currentConfig;
      }
      return typeof nextConfig === "function" ? nextConfig(currentConfig) : nextConfig;
    });
  }, []);

  useEffect(() => {
    if (runs.length === 0) {
      setSelectedRunId("");
      return;
    }
    if (!runs.some((run) => run.id === selectedRunId)) {
      setSelectedRunId(runs[0]?.id ?? "");
    }
  }, [runs, selectedRunId]);

  useEffect(() => {
    if (sourceRunId !== null && runs.some((run) => run.id === sourceRunId)) {
      setSelectedRunId(sourceRunId);
      setNameEdited(false);
    }
  }, [runs, sourceRunId]);

  useEffect(() => {
    if (selectedRunId === "" || runDetailsById[selectedRunId] !== undefined) {
      return undefined;
    }
    let ignore = false;
    void loadRunDetail(selectedRunId).catch((caught) => {
      if (!ignore) {
        onGlobalError(caught instanceof Error ? caught.message : "failed to load run details");
      }
    });
    return () => {
      ignore = true;
    };
  }, [loadRunDetail, onGlobalError, runDetailsById, selectedRunId]);

  useEffect(() => {
    if (sourceRunId !== null && selectedRunId === sourceRunId && selectedRunDetail !== null) {
      setPresetId("source_run");
    }
  }, [selectedRunDetail, selectedRunId, sourceRunId]);

  useEffect(() => {
    if (selectedPreset === null) {
      setPresetConfig(null);
      setAppliedPresetKey(null);
      return;
    }
    if (appliedPresetKey === selectedPreset.cacheKey) {
      return;
    }
    setPresetConfig(clonePresetConfig(selectedPreset.config));
    setAppliedPresetKey(selectedPreset.cacheKey);
  }, [appliedPresetKey, selectedPreset]);

  useEffect(() => {
    if (!nameEdited) {
      setNameText(defaultName);
    }
  }, [defaultName, nameEdited]);

  function selectPreset(nextPresetId: EvaluationPresetId) {
    const nextPreset = presets.find((preset) => preset.id === nextPresetId);
    setPresetId(nextPresetId);
    if (nextPreset !== undefined) {
      setPresetConfig(clonePresetConfig(nextPreset.config));
      setAppliedPresetKey(nextPreset.cacheKey);
      setNameEdited(false);
    }
  }

  async function submitEvaluation() {
    const repeatsPerTarget = Number.parseInt(repeatText, 10);
    const seed = Number.parseInt(seedText, 10);
    const name = nameText.trim() || defaultName;
    if (selectedRun === null) {
      onGlobalError("Select a source run before creating an evaluation.");
      return;
    }
    if (metadata === null || presetConfig === null) {
      onGlobalError("Evaluation preset metadata is not available.");
      return;
    }
    if (!Number.isInteger(repeatsPerTarget) || repeatsPerTarget < 1) {
      onGlobalError("Evaluation repeats must be a positive integer.");
      return;
    }
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      onGlobalError("Evaluation seed must be an integer from 0 to 4294967295.");
      return;
    }
    const target = evaluationTargetFromConfig(presetConfig, metadata);
    setIsCreating(true);
    onGlobalError(null);
    try {
      await onCreateEvaluation({
        courseIds: target.courseIds,
        cupIds: target.cupIds,
        difficulties: target.difficulties,
        name,
        policyMode,
        repeatsPerTarget,
        seed,
        sourceArtifact,
        sourceRunId: selectedRun.id,
        targetMode: target.mode,
        vehicleIds: target.vehicleIds,
      });
      setNameEdited(false);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to create evaluation");
    } finally {
      setIsCreating(false);
    }
  }

  return (
    <Panel>
      <div className="panel-header-row">
        <PanelHeader
          title="Evaluations"
          subtitle="Freeze policy checkpoints for reproducible headless evaluation runs."
        />
      </div>

      {runs.length === 0 ? (
        <Notice>Create or import a run before creating evaluation snapshots.</Notice>
      ) : defaultConfig === null || metadata === null ? (
        <Notice tone="error">Run-manager metadata is required before creating evaluations.</Notice>
      ) : (
        <div className="grid gap-5">
          {evaluationError !== null ? (
            <Notice tone="error">
              Evaluation records could not be loaded: {evaluationError}. Other run-manager data is
              still available.
            </Notice>
          ) : null}
          <section className="border border-app-border bg-app-surface-muted p-4">
            <div className="mb-4 flex items-center gap-2 text-sm font-bold tracking-[0.04em] text-app-muted uppercase">
              <EvaluationTabIcon />
              <span>Checkpoint snapshot</span>
            </div>
            <div className="grid gap-3 xl:grid-cols-[minmax(260px,1fr)_150px_160px_minmax(260px,1fr)]">
              <FieldShell>
                <span>Policy run</span>
                <FieldSelect
                  value={selectedRunId}
                  onChange={(event) => {
                    setSelectedRunId(event.currentTarget.value);
                    setNameEdited(false);
                  }}
                >
                  {runs.map((run) => (
                    <option key={run.id} value={run.id}>
                      {run.name}
                    </option>
                  ))}
                </FieldSelect>
              </FieldShell>
              <FieldShell>
                <span>Artifact</span>
                <FieldSelect
                  value={sourceArtifact}
                  onChange={(event) =>
                    setSourceArtifact(event.currentTarget.value as SourceArtifactChoice)
                  }
                >
                  <option value="latest">latest</option>
                  <option value="best">best</option>
                </FieldSelect>
              </FieldShell>
              <FieldShell>
                <span>Mode</span>
                <FieldSelect
                  value={policyMode}
                  onChange={(event) =>
                    setPolicyMode(event.currentTarget.value as PolicyPlaybackMode)
                  }
                >
                  <option value="deterministic">deterministic</option>
                  <option value="stochastic">stochastic</option>
                </FieldSelect>
              </FieldShell>
              <FieldShell>
                <span>Snapshot name</span>
                <FieldInput
                  value={nameText}
                  onChange={(event) => {
                    setNameEdited(true);
                    setNameText(event.currentTarget.value);
                  }}
                />
              </FieldShell>
            </div>
          </section>

          <section className="border border-app-border bg-app-surface-muted p-4">
            <div className="mb-4 flex items-center gap-2 text-sm font-bold tracking-[0.04em] text-app-muted uppercase">
              <span>Evaluation preset</span>
            </div>
            <div className="grid gap-3 xl:grid-cols-[minmax(260px,1fr)_120px_180px_auto]">
              <FieldShell>
                <span>Preset</span>
                <FieldSelect
                  value={selectedPreset?.id ?? presetId}
                  onChange={(event) =>
                    selectPreset(event.currentTarget.value as EvaluationPresetId)
                  }
                >
                  {presets.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.label}
                    </option>
                  ))}
                </FieldSelect>
              </FieldShell>
              <FieldShell>
                <span>Repeats</span>
                <FieldInput
                  min={1}
                  type="number"
                  value={repeatText}
                  onChange={(event) => setRepeatText(event.currentTarget.value)}
                />
              </FieldShell>
              <FieldShell>
                <span>Seed</span>
                <FieldInput
                  min={0}
                  type="number"
                  value={seedText}
                  onChange={(event) => setSeedText(event.currentTarget.value)}
                />
              </FieldShell>
              <Button
                className="mt-[22px] gap-2"
                disabled={isCreating || presetConfig === null}
                type="button"
                variant="primary"
                onClick={() => void submitEvaluation()}
              >
                <PlusIcon />
                <span>{isCreating ? "Creating" : "Create"}</span>
              </Button>
            </div>
          </section>

          {presetConfig !== null ? (
            <ConfigStack>
              <TracksSection
                config={presetConfig}
                defaultConfig={defaultConfig}
                metadata={metadata}
                setConfig={setEvaluationConfig}
                showSampling={false}
              />
              <VehicleSection
                config={presetConfig}
                defaultConfig={defaultConfig}
                metadata={metadata}
                setConfig={setEvaluationConfig}
                showEngineControls={false}
              />
            </ConfigStack>
          ) : null}

          {evaluations.length === 0 ? (
            <Notice>No evaluation snapshots yet.</Notice>
          ) : (
            <EvaluationTable evaluations={evaluations} />
          )}
        </div>
      )}
    </Panel>
  );
}

function EvaluationTable({ evaluations }: { evaluations: ManagedEvaluation[] }) {
  return (
    <div className="overflow-x-auto border border-app-border bg-app-surface">
      <table className="w-full min-w-[980px] border-collapse text-left text-sm">
        <thead className="border-b border-app-border text-xs font-bold tracking-[0.04em] text-app-muted uppercase">
          <tr>
            <th className="px-4 py-3">Evaluation</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3">Checkpoint</th>
            <th className="px-4 py-3">Target</th>
            <th className="px-4 py-3">Created</th>
            <th className="px-4 py-3">Directory</th>
          </tr>
        </thead>
        <tbody>
          {evaluations.map((evaluation) => (
            <tr className="border-b border-app-border last:border-b-0" key={evaluation.id}>
              <td className="px-4 py-3 align-top">
                <div className="grid gap-1">
                  <strong className="text-app-text">{evaluation.name}</strong>
                  <span className="font-mono text-xs text-app-muted">{evaluation.id}</span>
                </div>
              </td>
              <td className="px-4 py-3 align-top capitalize text-app-muted">{evaluation.status}</td>
              <td className="px-4 py-3 align-top">
                <div className="grid gap-1 text-app-muted">
                  <span>
                    {evaluation.checkpoint.source_run_name ?? evaluation.source_run_id ?? "-"} ·{" "}
                    {evaluation.checkpoint.artifact}
                  </span>
                  <span className="text-xs">
                    {formatStepCount(evaluation.checkpoint.lineage_num_timesteps)}
                  </span>
                </div>
              </td>
              <td className="px-4 py-3 align-top text-app-muted">
                <div className="grid gap-1">
                  <span>
                    {TARGET_MODE_LABELS[evaluation.target.mode]} ·{" "}
                    {evaluation.target.repeats_per_target}x
                  </span>
                  <span className="text-xs">{targetSelectionLabel(evaluation.target)}</span>
                </div>
              </td>
              <td className="px-4 py-3 align-top text-app-muted">
                {formatDate(evaluation.created_at)}
              </td>
              <td className="px-4 py-3 align-top font-mono text-xs text-app-muted">
                {evaluation.evaluation_dir}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function randomSeed() {
  return Math.floor(Math.random() * 0x100000000);
}

function targetSelectionLabel(evaluationTarget: ManagedEvaluation["target"]) {
  const parts = [
    selectionCountLabel(evaluationTarget.cup_ids, "cup"),
    selectionCountLabel(evaluationTarget.course_ids, "course"),
    selectionCountLabel(evaluationTarget.difficulties, "difficulty"),
    selectionCountLabel(evaluationTarget.vehicle_ids, "vehicle"),
  ].filter((part) => part !== null);
  return parts.length === 0 ? "all targets" : parts.join(" · ");
}

function selectionCountLabel(values: readonly string[], singular: string) {
  if (values.length === 0) {
    return null;
  }
  return `${values.length} ${singular}${values.length === 1 ? "" : "s"}`;
}

function formatStepCount(value: number | null) {
  return value === null ? "step unknown" : `${value.toLocaleString()} steps`;
}
