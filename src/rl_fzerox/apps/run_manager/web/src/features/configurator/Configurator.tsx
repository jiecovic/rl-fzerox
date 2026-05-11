// src/rl_fzerox/apps/run_manager/web/src/features/configurator/Configurator.tsx
import { useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { ActionBar } from "@/features/configurator/configurator/ActionBar";
import { configuratorDraftName } from "@/features/configurator/configurator/draftName";
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
  ManagedRun,
  ManagedRunConfig,
  PolicyArchitecturePreview,
} from "@/shared/api/contract";
import { formatDate } from "@/shared/ui/format";
import { RandomizeIcon } from "@/shared/ui/icons";
import { Notice, Panel } from "@/shared/ui/Panel";
import { Tabs } from "@/shared/ui/Tabs";

interface ConfiguratorProps {
  active?: boolean;
  baseConfig: ManagedRunConfig;
  existingNames: string[];
  forkSourceArtifact?: "latest" | "best" | null;
  forkSourceRunLabel?: string | null;
  initialDraftName?: string;
  initialConfig?: ManagedRunConfig | null;
  loadedDraft: ManagedDraft | null;
  metadata: ConfigMetadata;
  onDraftNameChange?: (name: string) => void;
  onLaunchRun: (
    name: string,
    config: ManagedRunConfig,
    draftId: string | null,
  ) => Promise<ManagedRun>;
  onSaveDraft: (name: string, config: ManagedRunConfig) => Promise<ManagedDraft>;
  onUpdateDraft: (id: string, name: string, config: ManagedRunConfig) => Promise<ManagedDraft>;
}

const POLICY_PREVIEW_DEBOUNCE_MS = 250;

export function Configurator({
  active = true,
  baseConfig,
  existingNames,
  forkSourceArtifact = null,
  forkSourceRunLabel = null,
  initialDraftName,
  initialConfig = null,
  loadedDraft,
  metadata,
  onDraftNameChange,
  onLaunchRun,
  onSaveDraft,
  onUpdateDraft,
}: ConfiguratorProps) {
  const baselineConfig = loadedDraft?.config ?? initialConfig ?? baseConfig;
  const baselineDraftName = configuratorDraftName(baseConfig, initialDraftName, loadedDraft);
  const [draftName, setDraftName] = useState(baselineDraftName);
  const [config, setConfig] = useState(baselineConfig);
  const [section, setSection] = useState<ConfigSection>("tracks");
  const [policyPreview, setPolicyPreview] = useState<PolicyArchitecturePreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const configRef = useRef(config);
  const resetSourceKeyRef = useRef<string | null>(null);
  configRef.current = config;
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
        ? "This draft name is already used by another draft or open editor."
        : loadedDraft === null && createNameConflict
          ? "This draft name is already used by another draft or open editor."
          : null;
  const isDirty = useMemo(() => {
    if (normalizedDraftName !== normalizedBaselineDraftName) {
      return true;
    }
    return stableJson(config) !== stableJson(baselineConfig);
  }, [baselineConfig, config, normalizedBaselineDraftName, normalizedDraftName]);
  const notifyDraftNameChange = useEffectEvent((name: string) => {
    onDraftNameChange?.(name);
  });
  const resetSourceKey =
    loadedDraft?.id ??
    [
      "new",
      initialDraftName ?? baselineDraftName,
      forkSourceRunLabel ?? "",
      forkSourceArtifact ?? "",
    ].join(":");
  const checkpointLocked =
    (loadedDraft !== null &&
      loadedDraft.source_run_id !== null &&
      loadedDraft.source_artifact !== null) ||
    forkSourceArtifact !== null;

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

  const previewConfig = useDebouncedValue(config, POLICY_PREVIEW_DEBOUNCE_MS);

  useEffect(() => {
    if (!active) {
      setPreviewError(null);
      return undefined;
    }
    let ignore = false;
    setPreviewError(null);
    void fetchPolicyPreview(previewConfig)
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
  }, [active, previewConfig]);

  useEffect(() => {
    notifyDraftNameChange(draftName);
  }, [draftName]);

  function committedConfigSnapshot() {
    const activeElement = document.activeElement;
    if (
      activeElement instanceof HTMLInputElement ||
      activeElement instanceof HTMLTextAreaElement ||
      activeElement instanceof HTMLSelectElement
    ) {
      flushSync(() => {
        activeElement.blur();
      });
    } else {
      flushSync(() => undefined);
    }
    return configRef.current;
  }

  async function saveDraft() {
    if (createNameConflict || normalizedDraftName.length === 0) {
      return;
    }
    const committedConfig = committedConfigSnapshot();
    setIsSaving(true);
    setError(null);
    try {
      const savedDraft = await onSaveDraft(normalizedDraftName, committedConfig);
      setDraftName(savedDraft.name);
      setConfig(savedDraft.config);
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
    const committedConfig = committedConfigSnapshot();
    setIsUpdating(true);
    setError(null);
    try {
      const savedDraft = await onUpdateDraft(loadedDraft.id, normalizedDraftName, committedConfig);
      setDraftName(savedDraft.name);
      setConfig(savedDraft.config);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to update draft");
    } finally {
      setIsUpdating(false);
    }
  }

  async function launchRun() {
    if (normalizedDraftName.length === 0) {
      return;
    }
    const committedConfig = committedConfigSnapshot();
    setIsTraining(true);
    setError(null);
    try {
      await onLaunchRun(normalizedDraftName, committedConfig, loadedDraft?.id ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to launch training run");
    } finally {
      setIsTraining(false);
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

  const forkedAtLabel = loadedDraft === null ? null : formatDate(loadedDraft.created_at);
  const forkSourceStepCount = loadedDraft?.source_num_timesteps ?? null;

  return (
    <Panel>
      <ActionBar
        canSave={
          !isSaving &&
          !isUpdating &&
          !isTraining &&
          !createNameConflict &&
          normalizedDraftName.length > 0
        }
        canTrain={!isSaving && !isUpdating && !isTraining && normalizedDraftName.length > 0}
        canUpdate={
          loadedDraft !== null &&
          !isSaving &&
          !isUpdating &&
          !isTraining &&
          !updateNameConflict &&
          normalizedDraftName.length > 0
        }
        hasLoadedDraft={loadedDraft !== null}
        isDirty={isDirty}
        isSaving={isSaving}
        isTraining={isTraining}
        isUpdating={isUpdating}
        onResetToDefault={resetToDefault}
        onResetToDraft={resetToDraft}
        onSaveDraft={() => void saveDraft()}
        onTrain={() => void launchRun()}
        onUpdateDraft={() => void updateDraft()}
      />

      {error !== null || previewError !== null ? (
        <div className="configurator-feedback-stack">
          {error !== null ? <Notice tone="error">{error}</Notice> : null}
          {previewError !== null ? <Notice tone="error">{previewError}</Notice> : null}
        </div>
      ) : null}

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
              onChange={(event) => {
                const seed = Number(event.target.value);
                setConfig((currentConfig) => ({ ...currentConfig, seed }));
              }}
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
        {forkSourceRunLabel !== null && forkSourceArtifact !== null ? (
          <section className="fork-context-card" aria-label="Fork source context">
            <div className="fork-context-header">
              <span className="fork-context-kicker">Forked from</span>
              <strong>{forkSourceRunLabel}</strong>
              <span className="fork-context-artifact">{forkSourceArtifact} checkpoint</span>
            </div>
            {forkedAtLabel !== null || forkSourceStepCount !== null ? (
              <div className="fork-context-meta">
                {forkedAtLabel !== null ? <span>forked {forkedAtLabel}</span> : null}
                {forkSourceStepCount !== null ? (
                  <span>@ {forkSourceStepCount.toLocaleString()} steps</span>
                ) : null}
              </div>
            ) : null}
          </section>
        ) : null}
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
          checkpointLocked={checkpointLocked}
          config={config}
          defaultConfig={baseConfig}
          metadata={metadata}
          preview={policyPreview}
          setConfig={setConfig}
        />
      ) : null}
      {section === "policy" ? (
        <PolicySection
          checkpointLocked={checkpointLocked}
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
          checkpointLocked={checkpointLocked}
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
    </Panel>
  );
}

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      setDebouncedValue(value);
    }, delayMs);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [delayMs, value]);

  return debouncedValue;
}

function stableJson(value: unknown): string {
  return JSON.stringify(sortObjectKeys(value));
}

function sortObjectKeys(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortObjectKeys);
  }
  if (value === null || typeof value !== "object") {
    return value;
  }
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nestedValue]) => [key, sortObjectKeys(nestedValue)]),
  );
}
