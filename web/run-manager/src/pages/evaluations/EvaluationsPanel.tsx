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
import { Button, IconButton } from "@/shared/ui/Button";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { cn } from "@/shared/ui/cn";
import { ConfigStack } from "@/shared/ui/config/ConfigLayout";
import { FieldInput, FieldSelect, FieldShell } from "@/shared/ui/Field";
import { formatDate } from "@/shared/ui/format";
import { EvaluationTabIcon, PlayIcon, PlusIcon, TrashIcon } from "@/shared/ui/icons";
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
  onDeleteEvaluation: (evaluation: ManagedEvaluation) => Promise<void>;
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
  onDeleteEvaluation,
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
  const [deletingEvaluationId, setDeletingEvaluationId] = useState<string | null>(null);
  const [deleteSelectedRequested, setDeleteSelectedRequested] = useState(false);
  const [isDeletingSelected, setIsDeletingSelected] = useState(false);
  const [selectedEvaluationIds, setSelectedEvaluationIds] = useState<ReadonlySet<string>>(
    new Set(),
  );
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
  const deletableEvaluations = useMemo(
    () => evaluations.filter((evaluation) => evaluation.status === "created"),
    [evaluations],
  );
  const selectedEvaluations = useMemo(
    () => deletableEvaluations.filter((evaluation) => selectedEvaluationIds.has(evaluation.id)),
    [deletableEvaluations, selectedEvaluationIds],
  );
  const allDeletableEvaluationsSelected =
    deletableEvaluations.length > 0 &&
    deletableEvaluations.every((evaluation) => selectedEvaluationIds.has(evaluation.id));
  const isDeletingEvaluation = deletingEvaluationId !== null || isDeletingSelected;

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

  useEffect(() => {
    const deletableIds = new Set(deletableEvaluations.map((evaluation) => evaluation.id));
    setSelectedEvaluationIds((current) => {
      const next = new Set([...current].filter((evaluationId) => deletableIds.has(evaluationId)));
      return next.size === current.size ? current : next;
    });
  }, [deletableEvaluations]);

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
      setActiveTab("records");
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to create evaluation");
    } finally {
      setIsCreating(false);
    }
  }

  async function deleteEvaluationRecord(evaluation: ManagedEvaluation) {
    setDeletingEvaluationId(evaluation.id);
    onGlobalError(null);
    try {
      await onDeleteEvaluation(evaluation);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to delete evaluation");
    } finally {
      setDeletingEvaluationId(null);
    }
  }

  function setAllEvaluationsSelected(selected: boolean) {
    setSelectedEvaluationIds(
      selected ? new Set(deletableEvaluations.map(({ id }) => id)) : new Set(),
    );
  }

  function toggleEvaluationSelection(evaluationId: string, selected: boolean) {
    setSelectedEvaluationIds((current) => {
      const next = new Set(current);
      if (selected) {
        next.add(evaluationId);
      } else {
        next.delete(evaluationId);
      }
      return next;
    });
  }

  async function confirmSelectedEvaluationDelete() {
    const targets = selectedEvaluations;
    if (targets.length === 0) {
      setDeleteSelectedRequested(false);
      return;
    }
    setIsDeletingSelected(true);
    onGlobalError(null);
    try {
      for (const evaluation of targets) {
        await onDeleteEvaluation(evaluation);
      }
      setSelectedEvaluationIds(new Set());
      setDeleteSelectedRequested(false);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to delete evaluations");
    } finally {
      setIsDeletingSelected(false);
    }
  }

  return (
    <>
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
          <Notice tone="error">
            Run-manager metadata is required before creating evaluations.
          </Notice>
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
                repeatLabel={
                  selectedTarget?.mode === "gp_cup" ? "Repeats / cup" : "Repeats / course"
                }
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
              <RecordsPanel
                allDeletableEvaluationsSelected={allDeletableEvaluationsSelected}
                deletingEvaluationId={deletingEvaluationId}
                evaluationError={evaluationError}
                evaluations={evaluations}
                isDeletingEvaluation={isDeletingEvaluation}
                selectedEvaluationCount={selectedEvaluations.length}
                selectedEvaluationIds={selectedEvaluationIds}
                onDeleteEvaluation={(evaluation) => void deleteEvaluationRecord(evaluation)}
                onRequestSelectedDelete={() => setDeleteSelectedRequested(true)}
                onSelectAllEvaluations={setAllEvaluationsSelected}
                onToggleEvaluationSelection={toggleEvaluationSelection}
              />
            ) : null}
          </div>
        )}
      </Panel>
      <ConfirmDialog
        busy={isDeletingSelected}
        confirmLabel={
          selectedEvaluations.length === 0
            ? "Delete selected"
            : `Delete ${selectedEvaluations.length} snapshots`
        }
        description={`Delete ${selectedEvaluations.length} created evaluation snapshot${
          selectedEvaluations.length === 1 ? "" : "s"
        } and their copied checkpoint files.`}
        open={deleteSelectedRequested}
        title="Delete selected evaluations"
        onClose={() => setDeleteSelectedRequested(false)}
        onConfirm={() => void confirmSelectedEvaluationDelete()}
      />
    </>
  );
}

function SnapshotCreatePanel({
  defaultName,
  isCreating,
  nameText,
  policyMode,
  presetId,
  presets,
  repeatLabel,
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
  repeatLabel: string;
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
          <span>{repeatLabel}</span>
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
  allDeletableEvaluationsSelected,
  deletingEvaluationId,
  evaluationError,
  evaluations,
  isDeletingEvaluation,
  selectedEvaluationCount,
  selectedEvaluationIds,
  onDeleteEvaluation,
  onRequestSelectedDelete,
  onSelectAllEvaluations,
  onToggleEvaluationSelection,
}: {
  allDeletableEvaluationsSelected: boolean;
  deletingEvaluationId: string | null;
  evaluationError: string | null;
  evaluations: ManagedEvaluation[];
  isDeletingEvaluation: boolean;
  selectedEvaluationCount: number;
  selectedEvaluationIds: ReadonlySet<string>;
  onDeleteEvaluation: (evaluation: ManagedEvaluation) => void;
  onRequestSelectedDelete: () => void;
  onSelectAllEvaluations: (selected: boolean) => void;
  onToggleEvaluationSelection: (evaluationId: string, selected: boolean) => void;
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
    <div className="grid gap-3">
      <Notice>
        These records are frozen checkpoint snapshots. Headless execution and live progress are the
        next evaluation-runner phase; GP cup snapshots are not startable yet.
      </Notice>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-app-muted">
          {selectedEvaluationCount === 0
            ? "No snapshots selected"
            : `${selectedEvaluationCount} snapshot${
                selectedEvaluationCount === 1 ? "" : "s"
              } selected`}
        </div>
        <Button
          className="gap-2"
          disabled={selectedEvaluationCount === 0 || isDeletingEvaluation}
          tone="danger"
          type="button"
          onClick={onRequestSelectedDelete}
        >
          <TrashIcon />
          <span>
            {selectedEvaluationCount === 0
              ? "Delete selected"
              : `Delete selected (${selectedEvaluationCount})`}
          </span>
        </Button>
      </div>
      <EvaluationTable
        allDeletableEvaluationsSelected={allDeletableEvaluationsSelected}
        deletingEvaluationId={deletingEvaluationId}
        evaluations={evaluations}
        isDeletingEvaluation={isDeletingEvaluation}
        selectedEvaluationIds={selectedEvaluationIds}
        onDeleteEvaluation={onDeleteEvaluation}
        onSelectAllEvaluations={onSelectAllEvaluations}
        onToggleEvaluationSelection={onToggleEvaluationSelection}
      />
    </div>
  );
}

function EvaluationTable({
  allDeletableEvaluationsSelected,
  deletingEvaluationId,
  evaluations,
  isDeletingEvaluation,
  selectedEvaluationIds,
  onDeleteEvaluation,
  onSelectAllEvaluations,
  onToggleEvaluationSelection,
}: {
  allDeletableEvaluationsSelected: boolean;
  deletingEvaluationId: string | null;
  evaluations: ManagedEvaluation[];
  isDeletingEvaluation: boolean;
  selectedEvaluationIds: ReadonlySet<string>;
  onDeleteEvaluation: (evaluation: ManagedEvaluation) => void;
  onSelectAllEvaluations: (selected: boolean) => void;
  onToggleEvaluationSelection: (evaluationId: string, selected: boolean) => void;
}) {
  return (
    <div className="overflow-x-auto border border-app-border bg-app-surface">
      <table className="w-full min-w-[980px] border-collapse text-left text-sm">
        <thead className="border-b border-app-border text-xs font-bold tracking-[0.04em] text-app-muted uppercase">
          <tr>
            <th className="w-10 px-4 py-3">
              <label className="grid place-items-center">
                <input
                  aria-label="Select all created evaluations"
                  checked={allDeletableEvaluationsSelected}
                  className={evaluationCheckboxClass}
                  disabled={isDeletingEvaluation}
                  type="checkbox"
                  onChange={(event) => onSelectAllEvaluations(event.currentTarget.checked)}
                />
              </label>
            </th>
            <th className="px-4 py-3">Evaluation</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3">Checkpoint</th>
            <th className="px-4 py-3">Target</th>
            <th className="px-4 py-3">Execution</th>
            <th className="px-4 py-3">Created</th>
            <th className="px-4 py-3">Directory</th>
            <th className="px-4 py-3 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {evaluations.map((evaluation) => {
            const isCreated = evaluation.status === "created";
            const selected = selectedEvaluationIds.has(evaluation.id);
            return (
              <tr className={evaluationRowClass(selected)} key={evaluation.id}>
                <td className="px-4 py-3 align-top">
                  <label className="grid place-items-center">
                    <input
                      aria-label={`Select evaluation ${evaluation.name}`}
                      checked={selected}
                      className={evaluationCheckboxClass}
                      disabled={!isCreated || isDeletingEvaluation}
                      type="checkbox"
                      onChange={(event) =>
                        onToggleEvaluationSelection(evaluation.id, event.currentTarget.checked)
                      }
                    />
                  </label>
                </td>
                <td className="px-4 py-3 align-top">
                  <div className="grid gap-1">
                    <strong className="text-app-text">{evaluation.name}</strong>
                    <span className="font-mono text-xs text-app-muted">{evaluation.id}</span>
                  </div>
                </td>
                <td className="px-4 py-3 align-top capitalize text-app-muted">
                  {evaluation.status}
                </td>
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
                <td className="px-4 py-3 align-top">
                  <div className="grid gap-2">
                    <span className="text-app-muted">{evaluationExecutionLabel(evaluation)}</span>
                    <Button
                      className="w-fit gap-2"
                      disabled
                      title={evaluationExecutionTitle(evaluation)}
                      type="button"
                    >
                      <PlayIcon />
                      <span>Start</span>
                    </Button>
                  </div>
                </td>
                <td className="px-4 py-3 align-top text-app-muted">
                  {formatDate(evaluation.created_at)}
                </td>
                <td className="px-4 py-3 align-top font-mono text-xs text-app-muted">
                  {evaluation.evaluation_dir}
                </td>
                <td className="px-4 py-3 text-right align-top">
                  <IconButton
                    aria-label={`Delete evaluation ${evaluation.name}`}
                    disabled={
                      !isCreated || deletingEvaluationId === evaluation.id || isDeletingEvaluation
                    }
                    size="small"
                    tone="danger"
                    onClick={() => onDeleteEvaluation(evaluation)}
                  >
                    <TrashIcon />
                  </IconButton>
                </td>
              </tr>
            );
          })}
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

function evaluationExecutionLabel(evaluation: ManagedEvaluation) {
  if (evaluation.status === "created") {
    return evaluation.target.mode === "gp_cup" ? "GP runner pending" : "runner pending";
  }
  return evaluation.status;
}

function evaluationExecutionTitle(evaluation: ManagedEvaluation) {
  if (evaluation.target.mode === "gp_cup") {
    return "GP/cup evaluation execution is not implemented yet.";
  }
  return "Headless evaluation execution is not wired to Run Manager yet.";
}

const evaluationCheckboxClass = "h-4 w-4 accent-app-accent";

function evaluationRowClass(selected: boolean) {
  return cn(
    "border-b border-app-border transition-colors last:border-b-0",
    selected ? "bg-app-surface-muted" : undefined,
  );
}
