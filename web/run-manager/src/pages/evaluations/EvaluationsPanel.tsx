// web/run-manager/src/pages/evaluations/EvaluationsPanel.tsx
import { useCallback, useEffect, useMemo, useState } from "react";

import {
  buildEvaluationPresets,
  createDefaultEvaluationPreset,
  customEvaluationPresetFromConfig,
  type EvaluationPresetId,
  type EvaluationPresetOverride,
  type EvaluationPresetStorage,
  evaluationTargetFromConfig,
  normalizeEvaluationPresetConfig,
  readEvaluationPresetStorage,
  storedPresetFromEvaluationPreset,
  updateEvaluationPresetOverrideMap,
  writeEvaluationPresetStorage,
} from "@/entities/evaluation/model/presets";
import type { ConfigSetter } from "@/entities/runConfig/model/state";
import { PresetConfigPanel } from "@/pages/evaluations/sections/PresetConfigPanel";
import { RecordsPanel } from "@/pages/evaluations/sections/RecordsPanel";
import type {
  ConfigMetadata,
  EvaluationMode,
  ManagedEvaluation,
  ManagedRun,
  ManagedRunConfig,
  ManagedRunDetail,
} from "@/shared/api/contract";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { type TabItem, Tabs } from "@/shared/ui/Tabs";

interface EvaluationsPanelProps {
  defaultConfig: ManagedRunConfig | null;
  evaluationError: string | null;
  evaluations: ManagedEvaluation[];
  loadRunDetail: (runId: string) => Promise<ManagedRunDetail>;
  metadata: ConfigMetadata | null;
  onDeleteEvaluation: (evaluation: ManagedEvaluation) => Promise<void>;
  onGlobalError: (message: string | null) => void;
  onOpenEvaluation: (evaluation: ManagedEvaluation) => void;
  runs: ManagedRun[];
}

type EvaluationWorkspaceTab = "records" | "preset_config";

const EVALUATION_WORKSPACE_TABS: readonly TabItem<EvaluationWorkspaceTab>[] = (
  [
    ["records", "Records"],
    ["preset_config", "Presets"],
  ] as const
).map(([id, label]) => ({ id, label }));

export function EvaluationsPanel({
  defaultConfig,
  evaluationError,
  evaluations,
  loadRunDetail,
  metadata,
  onDeleteEvaluation,
  onGlobalError,
  onOpenEvaluation,
  runs,
}: EvaluationsPanelProps) {
  const [presetId, setPresetId] = useState<EvaluationPresetId>("time_attack_blue_falcon");
  const [presetStorage, setPresetStorage] = useState<EvaluationPresetStorage>(() =>
    readEvaluationPresetStorage(),
  );
  const [deletingEvaluationId, setDeletingEvaluationId] = useState<string | null>(null);
  const [deleteSelectedRequested, setDeleteSelectedRequested] = useState(false);
  const [isDeletingSelected, setIsDeletingSelected] = useState(false);
  const [importRunId, setImportRunId] = useState("");
  const [isImportingPreset, setIsImportingPreset] = useState(false);
  const [selectedEvaluationIds, setSelectedEvaluationIds] = useState<ReadonlySet<string>>(
    new Set(),
  );
  const [activeTab, setActiveTab] = useState<EvaluationWorkspaceTab>("records");

  const presets = useMemo(
    () =>
      defaultConfig === null || metadata === null
        ? []
        : buildEvaluationPresets({
            customPresets: presetStorage.customPresets,
            defaultConfig,
            metadata,
            overrides: presetStorage.overrides,
          }),
    [defaultConfig, metadata, presetStorage],
  );
  const selectedPreset = presets.find((preset) => preset.id === presetId) ?? presets[0] ?? null;
  const presetConfig = selectedPreset?.config ?? null;
  const selectedTarget = useMemo(
    () =>
      metadata !== null && presetConfig !== null
        ? evaluationTargetFromConfig(
            presetConfig,
            metadata,
            selectedPreset?.targetMode ?? "time_attack_course",
          )
        : null,
    [metadata, presetConfig, selectedPreset?.targetMode],
  );

  const updatePreset = useCallback(
    (patch: EvaluationPresetOverride) => {
      if (selectedPreset === null) {
        return;
      }
      setPresetStorage((current) => {
        const next =
          selectedPreset.builtin === true
            ? {
                ...current,
                overrides: updateEvaluationPresetOverrideMap(
                  current.overrides,
                  selectedPreset.id,
                  patch,
                ),
              }
            : {
                ...current,
                customPresets: current.customPresets.map((preset) =>
                  preset.id === selectedPreset.id
                    ? {
                        ...storedPresetFromEvaluationPreset(selectedPreset),
                        ...patch,
                        config:
                          patch.config === undefined
                            ? storedPresetFromEvaluationPreset(selectedPreset).config
                            : patch.config,
                      }
                    : preset,
                ),
              };
        writeEvaluationPresetStorage(next);
        return next;
      });
    },
    [selectedPreset],
  );

  const setEvaluationConfig: ConfigSetter = useCallback(
    (nextConfig) => {
      if (selectedPreset === null) {
        return;
      }
      const config =
        typeof nextConfig === "function" ? nextConfig(selectedPreset.config) : nextConfig;
      if (metadata === null) {
        updatePreset({ config });
        return;
      }
      updatePreset({
        config: normalizeEvaluationPresetConfig(config, metadata, selectedPreset.targetMode),
      });
    },
    [metadata, selectedPreset, updatePreset],
  );

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
    const deletableIds = new Set(deletableEvaluations.map((evaluation) => evaluation.id));
    setSelectedEvaluationIds((current) => {
      const next = new Set([...current].filter((evaluationId) => deletableIds.has(evaluationId)));
      return next.size === current.size ? current : next;
    });
  }, [deletableEvaluations]);

  function selectPreset(nextPresetId: EvaluationPresetId) {
    setPresetId(nextPresetId);
  }

  function persistPresetStorage(
    updater: (current: EvaluationPresetStorage) => EvaluationPresetStorage,
  ) {
    setPresetStorage((current) => {
      const next = updater(current);
      writeEvaluationPresetStorage(next);
      return next;
    });
  }

  function createCustomPreset() {
    if (defaultConfig === null || metadata === null) {
      return;
    }
    const nextPreset = createDefaultEvaluationPreset({
      defaultConfig,
      metadata,
    });
    persistPresetStorage((current) => ({
      ...current,
      customPresets: [...current.customPresets, nextPreset],
    }));
    setPresetId(nextPreset.id);
  }

  function deleteSelectedPreset() {
    if (selectedPreset === null || selectedPreset.builtin) {
      return;
    }
    persistPresetStorage((current) => ({
      ...current,
      customPresets: current.customPresets.filter((preset) => preset.id !== selectedPreset.id),
    }));
    setPresetId("time_attack_blue_falcon");
  }

  async function importRunPreset() {
    if (metadata === null || importRunId.length === 0) {
      return;
    }
    setIsImportingPreset(true);
    onGlobalError(null);
    try {
      const runDetail = await loadRunDetail(importRunId);
      const targetMode: EvaluationMode =
        runDetail.config.tracks.race_mode === "gp_race" ? "gp_course" : "time_attack_course";
      const nextPreset = customEvaluationPresetFromConfig({
        config: runDetail.config,
        label: `${runDetail.name} eval preset`,
        metadata,
        sourceArtifact: runDetail.source_artifact ?? "latest",
        targetMode,
      });
      persistPresetStorage((current) => ({
        ...current,
        customPresets: [...current.customPresets, nextPreset],
      }));
      setPresetId(nextPreset.id);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to import run preset");
    } finally {
      setIsImportingPreset(false);
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
        <PanelHeader
          title="Evaluations"
          subtitle="Measure saved policy checkpoints on selected courses, cups, and vehicles."
        />

        <div className="mt-5">
          <div className="section-tabs-row">
            <Tabs
              activeId={activeTab}
              items={EVALUATION_WORKSPACE_TABS}
              label="Evaluation sections"
              variant="section"
              onSelect={setActiveTab}
            />
          </div>

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
              onOpenEvaluation={onOpenEvaluation}
              onRequestSelectedDelete={() => setDeleteSelectedRequested(true)}
              onSelectAllEvaluations={setAllEvaluationsSelected}
              onToggleEvaluationSelection={toggleEvaluationSelection}
            />
          ) : null}

          {activeTab === "preset_config" ? (
            defaultConfig === null || metadata === null ? (
              <Notice tone="error">Run-manager metadata is required to edit presets.</Notice>
            ) : presetConfig === null ? (
              <Notice>No evaluation presets are available.</Notice>
            ) : (
              <PresetConfigPanel
                defaultConfig={defaultConfig}
                importRunId={importRunId}
                importingPreset={isImportingPreset}
                metadata={metadata}
                presetConfig={presetConfig}
                presetId={selectedPreset?.id ?? presetId}
                presets={presets}
                runs={runs}
                selectedPreset={selectedPreset}
                selectedTarget={selectedTarget}
                setEvaluationConfig={setEvaluationConfig}
                onCreatePreset={createCustomPreset}
                onDeletePreset={deleteSelectedPreset}
                onImportRunChange={setImportRunId}
                onImportRunPreset={() => void importRunPreset()}
                onPresetChange={selectPreset}
                onPresetSettingsChange={updatePreset}
              />
            )
          ) : null}
        </div>
      </Panel>
      <ConfirmDialog
        busy={isDeletingSelected}
        confirmLabel={
          selectedEvaluations.length === 0
            ? "Delete selected"
            : `Delete ${selectedEvaluations.length} evaluations`
        }
        description={`Delete ${selectedEvaluations.length} created evaluation${
          selectedEvaluations.length === 1 ? "" : "s"
        } and their checkpoint copies.`}
        open={deleteSelectedRequested}
        title="Delete selected evaluations"
        onClose={() => setDeleteSelectedRequested(false)}
        onConfirm={() => void confirmSelectedEvaluationDelete()}
      />
    </>
  );
}
