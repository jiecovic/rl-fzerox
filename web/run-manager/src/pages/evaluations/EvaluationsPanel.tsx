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
import { type TabItem, Tabs } from "@/shared/ui/Tabs";

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
type EvaluationWorkspaceTab = "snapshots" | "preset_config" | "records";
type EvaluationTargetDraft = ReturnType<typeof evaluationTargetFromConfig>;

const TARGET_MODE_LABELS: Record<EvaluationMode, string> = {
  gp_cup: "GP cup",
  time_attack: "Time attack",
};

const EVALUATION_WORKSPACE_TABS: readonly TabItem<EvaluationWorkspaceTab>[] = (
  [
    ["snapshots", "Create snapshot"],
    ["preset_config", "Preset config"],
    ["records", "Records"],
  ] as const
).map(([id, label]) => ({ id, label }));

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
  const [activeTab, setActiveTab] = useState<EvaluationWorkspaceTab>("snapshots");
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
  const selectedTarget = useMemo(
    () =>
      metadata !== null && presetConfig !== null
        ? evaluationTargetFromConfig(presetConfig, metadata)
        : null,
    [metadata, presetConfig],
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
          <Tabs
            activeId={activeTab}
            items={EVALUATION_WORKSPACE_TABS}
            label="Evaluation sections"
            variant="section"
            onSelect={setActiveTab}
          />

          {activeTab === "snapshots" ? (
            <SnapshotCreatePanel
              defaultName={defaultName}
              isCreating={isCreating}
              nameText={nameText}
              policyMode={policyMode}
              presetId={selectedPreset?.id ?? presetId}
              presets={presets}
              repeatText={repeatText}
              runs={runs}
              seedText={seedText}
              selectedRunId={selectedRunId}
              selectedTarget={selectedTarget}
              sourceArtifact={sourceArtifact}
              onConfigurePreset={() => setActiveTab("preset_config")}
              onCreate={() => void submitEvaluation()}
              onNameChange={(value) => {
                setNameEdited(true);
                setNameText(value);
              }}
              onPolicyModeChange={setPolicyMode}
              onPresetChange={selectPreset}
              onRepeatChange={setRepeatText}
              onRunChange={(runId) => {
                setSelectedRunId(runId);
                setNameEdited(false);
              }}
              onSeedChange={setSeedText}
              onSourceArtifactChange={setSourceArtifact}
            />
          ) : null}

          {activeTab === "preset_config" && presetConfig !== null ? (
            <PresetConfigPanel
              defaultConfig={defaultConfig}
              metadata={metadata}
              presetConfig={presetConfig}
              presetId={selectedPreset?.id ?? presetId}
              presets={presets}
              selectedTarget={selectedTarget}
              setEvaluationConfig={setEvaluationConfig}
              onPresetChange={selectPreset}
              onUseForSnapshot={() => setActiveTab("snapshots")}
            />
          ) : null}

          {activeTab === "records" ? (
            <RecordsPanel evaluationError={evaluationError} evaluations={evaluations} />
          ) : null}
        </div>
      )}
    </Panel>
  );
}

function SnapshotCreatePanel({
  defaultName,
  isCreating,
  nameText,
  policyMode,
  presetId,
  presets,
  repeatText,
  runs,
  seedText,
  selectedRunId,
  selectedTarget,
  sourceArtifact,
  onConfigurePreset,
  onCreate,
  onNameChange,
  onPolicyModeChange,
  onPresetChange,
  onRepeatChange,
  onRunChange,
  onSeedChange,
  onSourceArtifactChange,
}: {
  defaultName: string;
  isCreating: boolean;
  nameText: string;
  policyMode: PolicyPlaybackMode;
  presetId: EvaluationPresetId;
  presets: ReturnType<typeof buildEvaluationPresets>;
  repeatText: string;
  runs: ManagedRun[];
  seedText: string;
  selectedRunId: string;
  selectedTarget: EvaluationTargetDraft | null;
  sourceArtifact: SourceArtifactChoice;
  onConfigurePreset: () => void;
  onCreate: () => void;
  onNameChange: (value: string) => void;
  onPolicyModeChange: (value: PolicyPlaybackMode) => void;
  onPresetChange: (value: EvaluationPresetId) => void;
  onRepeatChange: (value: string) => void;
  onRunChange: (value: string) => void;
  onSeedChange: (value: string) => void;
  onSourceArtifactChange: (value: SourceArtifactChoice) => void;
}) {
  return (
    <section className="border border-app-border bg-app-surface-muted p-4">
      <div className="mb-4 flex items-center gap-2 text-sm font-bold tracking-[0.04em] text-app-muted uppercase">
        <EvaluationTabIcon />
        <span>Checkpoint snapshot</span>
      </div>
      <div className="grid gap-3 xl:grid-cols-[minmax(280px,1fr)_130px_150px_minmax(280px,1fr)]">
        <FieldShell>
          <span>Policy run</span>
          <FieldSelect value={selectedRunId} onChange={(event) => onRunChange(event.target.value)}>
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
            onChange={(event) => onSourceArtifactChange(event.target.value as SourceArtifactChoice)}
          >
            <option value="latest">latest</option>
            <option value="best">best</option>
          </FieldSelect>
        </FieldShell>
        <FieldShell>
          <span>Mode</span>
          <FieldSelect
            value={policyMode}
            onChange={(event) => onPolicyModeChange(event.target.value as PolicyPlaybackMode)}
          >
            <option value="deterministic">deterministic</option>
            <option value="stochastic">stochastic</option>
          </FieldSelect>
        </FieldShell>
        <FieldShell>
          <span>Snapshot name</span>
          <FieldInput
            placeholder={defaultName}
            value={nameText}
            onChange={(event) => onNameChange(event.target.value)}
          />
        </FieldShell>
      </div>

      <div className="mt-5 grid gap-3 xl:grid-cols-[minmax(280px,1fr)_110px_170px_auto_auto]">
        <FieldShell>
          <span>Evaluation preset</span>
          <FieldSelect
            value={presetId}
            onChange={(event) => onPresetChange(event.target.value as EvaluationPresetId)}
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
            onChange={(event) => onRepeatChange(event.target.value)}
          />
        </FieldShell>
        <FieldShell>
          <span>Seed</span>
          <FieldInput
            min={0}
            type="number"
            value={seedText}
            onChange={(event) => onSeedChange(event.target.value)}
          />
        </FieldShell>
        <Button className="mt-[22px]" type="button" onClick={onConfigurePreset}>
          Edit preset
        </Button>
        <Button
          className="mt-[22px] gap-2"
          disabled={isCreating}
          type="button"
          variant="primary"
          onClick={onCreate}
        >
          <PlusIcon />
          <span>{isCreating ? "Creating" : "Create"}</span>
        </Button>
      </div>

      <div className="mt-4 border-t border-app-border pt-3 text-sm text-app-muted">
        Target: {selectedTarget === null ? "not configured" : targetDraftLabel(selectedTarget)}
      </div>
    </section>
  );
}

function PresetConfigPanel({
  defaultConfig,
  metadata,
  presetConfig,
  presetId,
  presets,
  selectedTarget,
  setEvaluationConfig,
  onPresetChange,
  onUseForSnapshot,
}: {
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  presetConfig: ManagedRunConfig;
  presetId: EvaluationPresetId;
  presets: ReturnType<typeof buildEvaluationPresets>;
  selectedTarget: EvaluationTargetDraft | null;
  setEvaluationConfig: ConfigSetter;
  onPresetChange: (value: EvaluationPresetId) => void;
  onUseForSnapshot: () => void;
}) {
  return (
    <div className="grid gap-5">
      <section className="border border-app-border bg-app-surface-muted p-4">
        <div className="grid gap-3 xl:grid-cols-[minmax(280px,1fr)_auto]">
          <FieldShell>
            <span>Preset</span>
            <FieldSelect
              value={presetId}
              onChange={(event) => onPresetChange(event.target.value as EvaluationPresetId)}
            >
              {presets.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.label}
                </option>
              ))}
            </FieldSelect>
          </FieldShell>
          <Button className="mt-[22px]" type="button" onClick={onUseForSnapshot}>
            Use for snapshot
          </Button>
        </div>
        <div className="mt-3 text-sm text-app-muted">
          Target: {selectedTarget === null ? "not configured" : targetDraftLabel(selectedTarget)}
        </div>
      </section>
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
    </div>
  );
}

function RecordsPanel({
  evaluationError,
  evaluations,
}: {
  evaluationError: string | null;
  evaluations: ManagedEvaluation[];
}) {
  if (evaluationError !== null) {
    return (
      <Notice tone="error">
        Evaluation records could not be loaded: {evaluationError}. Other run-manager data is still
        available.
      </Notice>
    );
  }
  return evaluations.length === 0 ? (
    <Notice>No evaluation snapshots yet.</Notice>
  ) : (
    <EvaluationTable evaluations={evaluations} />
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
  return targetSelectionPartsLabel({
    courseIds: evaluationTarget.course_ids,
    cupIds: evaluationTarget.cup_ids,
    difficulties: evaluationTarget.difficulties,
    vehicleIds: evaluationTarget.vehicle_ids,
  });
}

function targetDraftLabel(evaluationTarget: EvaluationTargetDraft) {
  return `${TARGET_MODE_LABELS[evaluationTarget.mode]} · ${targetSelectionPartsLabel(evaluationTarget)}`;
}

function targetSelectionPartsLabel({
  courseIds,
  cupIds,
  difficulties,
  vehicleIds,
}: {
  courseIds: readonly string[];
  cupIds: readonly string[];
  difficulties: readonly string[];
  vehicleIds: readonly string[];
}) {
  const parts = [
    selectionCountLabel(cupIds, "cup"),
    selectionCountLabel(courseIds, "course"),
    selectionCountLabel(difficulties, "difficulty"),
    selectionCountLabel(vehicleIds, "vehicle"),
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
