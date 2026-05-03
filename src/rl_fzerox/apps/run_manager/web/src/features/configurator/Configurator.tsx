import { useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import { ActionBar } from "@/features/configurator/configurator/ActionBar";
import { configuratorDraftName } from "@/features/configurator/configurator/draftName";
import { RandomizeIcon } from "@/features/configurator/configurator/icons";
import {
  CONFIG_SECTION_TABS,
  type ConfigSection,
} from "@/features/configurator/configurator/sections";
import { FieldLabel } from "@/features/configurator/fields";
import { ActionSection } from "@/features/configurator/sections/ActionSection";
import { EnvironmentSection } from "@/features/configurator/sections/EnvironmentSection";
import { LoggingSection } from "@/features/configurator/sections/LoggingSection";
import { ObservationSection } from "@/features/configurator/sections/ObservationSection";
import { PolicySection } from "@/features/configurator/sections/PolicySection";
import { RewardSection } from "@/features/configurator/sections/RewardSection";
import { TracksSection } from "@/features/configurator/sections/TracksSection";
import { TrainingSection } from "@/features/configurator/sections/TrainingSection";
import { VehicleSection } from "@/features/configurator/sections/VehicleSection";
import { fetchPolicyPreview } from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRunConfig,
  PolicyArchitecturePreview,
} from "@/shared/api/contract";
import { Notice, Panel } from "@/shared/ui/Panel";
import { Tabs } from "@/shared/ui/Tabs";

interface ConfiguratorProps {
  baseConfig: ManagedRunConfig;
  existingNames: string[];
  initialDraftName?: string;
  loadedDraft: ManagedDraft | null;
  metadata: ConfigMetadata;
  onDraftNameChange?: (name: string) => void;
  onSaveDraft: (name: string, config: ManagedRunConfig) => Promise<ManagedDraft>;
  onUpdateDraft: (id: string, name: string, config: ManagedRunConfig) => Promise<ManagedDraft>;
}

export function Configurator({
  baseConfig,
  existingNames,
  initialDraftName,
  loadedDraft,
  metadata,
  onDraftNameChange,
  onSaveDraft,
  onUpdateDraft,
}: ConfiguratorProps) {
  const baselineConfig = loadedDraft?.config ?? baseConfig;
  const baselineDraftName = configuratorDraftName(baseConfig, initialDraftName, loadedDraft);
  const [draftName, setDraftName] = useState(baselineDraftName);
  const [config, setConfig] = useState(baseConfig);
  const [section, setSection] = useState<ConfigSection>("tracks");
  const [policyPreview, setPolicyPreview] = useState<PolicyArchitecturePreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const resetSourceKeyRef = useRef<string | null>(null);
  const normalizedDraftName = draftName.trim();
  const normalizedBaselineDraftName = baselineDraftName.trim();
  const normalizedLoadedDraftName = loadedDraft?.name.trim().toLowerCase() ?? null;
  const reservedNames = useMemo(
    () =>
      new Set(
        existingNames.map((name) => name.trim().toLowerCase()).filter((name) => name.length > 0),
      ),
    [existingNames],
  );
  const createNameConflict =
    normalizedDraftName.length > 0 && reservedNames.has(normalizedDraftName.toLowerCase());
  const updateNameConflict =
    loadedDraft !== null &&
    normalizedDraftName.length > 0 &&
    normalizedDraftName.toLowerCase() !== normalizedLoadedDraftName &&
    reservedNames.has(normalizedDraftName.toLowerCase());
  const nameError =
    normalizedDraftName.length === 0
      ? "Run name is required."
      : updateNameConflict
        ? "This name is already used by another draft, run, or open editor."
        : loadedDraft === null && createNameConflict
          ? "This name is already used by another draft, run, or open editor."
          : null;
  const isDirty = useMemo(() => {
    if (normalizedDraftName !== normalizedBaselineDraftName) {
      return true;
    }
    return JSON.stringify(config) !== JSON.stringify(baselineConfig);
  }, [baselineConfig, config, normalizedBaselineDraftName, normalizedDraftName]);
  const notifyDraftNameChange = useEffectEvent((name: string) => {
    onDraftNameChange?.(name);
  });
  const resetSourceKey = loadedDraft?.id ?? `new:${initialDraftName ?? baselineDraftName}`;

  useEffect(() => {
    if (resetSourceKeyRef.current === resetSourceKey) {
      return;
    }
    resetSourceKeyRef.current = resetSourceKey;
    setDraftName(baselineDraftName);
    setConfig(baselineConfig);
    setSection("tracks");
    setError(null);
    setPreviewError(null);
  }, [baselineConfig, baselineDraftName, resetSourceKey]);

  useEffect(() => {
    let ignore = false;
    setPreviewError(null);
    void fetchPolicyPreview(config)
      .then((preview) => {
        if (!ignore) {
          setPolicyPreview(preview);
        }
      })
      .catch((caught) => {
        if (!ignore) {
          setPolicyPreview(null);
          setPreviewError(
            caught instanceof Error ? caught.message : "failed to compute policy preview",
          );
        }
      });
    return () => {
      ignore = true;
    };
  }, [config]);

  useEffect(() => {
    notifyDraftNameChange(draftName);
  }, [draftName]);

  async function saveDraft() {
    if (createNameConflict || normalizedDraftName.length === 0) {
      return;
    }
    setIsSaving(true);
    setError(null);
    try {
      await onSaveDraft(normalizedDraftName, config);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to save draft");
    } finally {
      setIsSaving(false);
    }
  }

  async function updateDraft() {
    if (loadedDraft === null) {
      return;
    }
    if (updateNameConflict || normalizedDraftName.length === 0) {
      return;
    }
    setIsUpdating(true);
    setError(null);
    try {
      await onUpdateDraft(loadedDraft.id, normalizedDraftName, config);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to update draft");
    } finally {
      setIsUpdating(false);
    }
  }

  function randomizeSeed() {
    const values = new Uint32Array(1);
    crypto.getRandomValues(values);
    setConfig((currentConfig) => ({ ...currentConfig, seed: values[0] }));
  }

  function resetToDefault() {
    setConfig(baseConfig);
    setError(null);
    setPreviewError(null);
  }

  function resetToDraft() {
    if (loadedDraft === null) {
      return;
    }
    setDraftName(loadedDraft.name);
    setConfig(loadedDraft.config);
    setError(null);
    setPreviewError(null);
  }

  return (
    <Panel>
      <ActionBar
        canSave={!isSaving && !isUpdating && !createNameConflict && normalizedDraftName.length > 0}
        canUpdate={
          loadedDraft !== null &&
          !isSaving &&
          !isUpdating &&
          !updateNameConflict &&
          normalizedDraftName.length > 0
        }
        hasLoadedDraft={loadedDraft !== null}
        isDirty={isDirty}
        isSaving={isSaving}
        isUpdating={isUpdating}
        onResetToDefault={resetToDefault}
        onResetToDraft={resetToDraft}
        onSaveDraft={() => void saveDraft()}
        onUpdateDraft={() => void updateDraft()}
      />

      <div className="form-grid run-identity-grid">
        <div className="field-shell">
          <FieldLabel help="Run name used when this configuration is launched." label="Run name" />
          <input
            aria-invalid={nameError !== null}
            aria-label="Run name"
            spellCheck={false}
            value={draftName}
            onChange={(event) => setDraftName(event.target.value)}
          />
        </div>
        <div className="seed-control">
          <div className="field-shell">
            <FieldLabel
              help="Base random seed used when the run config is generated."
              label="Seed"
            />
            <input
              aria-label="Seed"
              type="number"
              value={config.seed}
              onChange={(event) => setConfig({ ...config, seed: Number(event.target.value) })}
            />
          </div>
          <button
            aria-label="Randomize seed"
            className="icon-button seed-randomize-button tooltip-anchor"
            data-tooltip="Randomize seed"
            type="button"
            onClick={randomizeSeed}
          >
            <RandomizeIcon />
          </button>
        </div>
      </div>
      {nameError !== null ? <Notice tone="error">{nameError}</Notice> : null}

      <div className="section-tabs-row">
        <Tabs
          label="Run configurator sections"
          activeId={section}
          items={CONFIG_SECTION_TABS}
          variant="section"
          onSelect={(id) => setSection(id)}
        />
      </div>

      {section === "tracks" ? (
        <TracksSection
          config={config}
          defaultConfig={baseConfig}
          metadata={metadata}
          setConfig={setConfig}
        />
      ) : null}
      {section === "training" ? (
        <TrainingSection config={config} defaultConfig={baseConfig} setConfig={setConfig} />
      ) : null}
      {section === "observation" ? (
        <ObservationSection
          config={config}
          defaultConfig={baseConfig}
          metadata={metadata}
          preview={policyPreview}
          setConfig={setConfig}
        />
      ) : null}
      {section === "policy" ? (
        <PolicySection
          config={config}
          defaultConfig={baseConfig}
          metadata={metadata}
          preview={policyPreview}
          setConfig={setConfig}
        />
      ) : null}
      {section === "reward" ? (
        <RewardSection config={config} defaultConfig={baseConfig} setConfig={setConfig} />
      ) : null}
      {section === "vehicle" ? (
        <VehicleSection
          config={config}
          defaultConfig={baseConfig}
          metadata={metadata}
          setConfig={setConfig}
        />
      ) : null}
      {section === "action" ? (
        <ActionSection
          config={config}
          defaultConfig={baseConfig}
          metadata={metadata}
          setConfig={setConfig}
        />
      ) : null}
      {section === "environment" ? (
        <EnvironmentSection config={config} defaultConfig={baseConfig} setConfig={setConfig} />
      ) : null}
      {section === "logging" ? (
        <LoggingSection config={config} defaultConfig={baseConfig} setConfig={setConfig} />
      ) : null}

      {error !== null ? <Notice tone="error">{error}</Notice> : null}
      {previewError !== null ? <Notice tone="error">{previewError}</Notice> : null}
    </Panel>
  );
}
