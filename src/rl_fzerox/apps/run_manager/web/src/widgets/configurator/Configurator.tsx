// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/Configurator.tsx
import { useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { CONFIG_SECTION_TABS, type ConfigSection } from "@/entities/runConfig/model/sections";
import { ActionSection } from "@/entities/runConfig/ui/sections/ActionSection";
import { EnvironmentSection } from "@/entities/runConfig/ui/sections/EnvironmentSection";
import { LoggingSection } from "@/entities/runConfig/ui/sections/LoggingSection";
import { ObservationSection } from "@/entities/runConfig/ui/sections/ObservationSection";
import { PolicySection } from "@/entities/runConfig/ui/sections/PolicySection";
import { RewardSection } from "@/entities/runConfig/ui/sections/RewardSection";
import { TracksSection } from "@/entities/runConfig/ui/sections/TracksSection";
import { TrainingSection } from "@/entities/runConfig/ui/sections/TrainingSection";
import { fixedEnvAssignmentIssue } from "@/entities/runConfig/ui/sections/tracks/fixedEnvAssignment";
import { VehicleSection } from "@/entities/runConfig/ui/sections/VehicleSection";
import { fetchPolicyPreview } from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRun,
  ManagedRunConfig,
  PolicyArchitecturePreview,
} from "@/shared/api/contract";
import { ConfigGrid } from "@/shared/ui/config/ConfigLayout";
import { FieldLabel } from "@/shared/ui/configFields";
import {
  blurOnEnter,
  editableNumberInputProps,
  parseSafeIntegerInput,
  useEditableNumberInput,
} from "@/shared/ui/configFields/numberInput";
import { FieldInput, FieldShell } from "@/shared/ui/Field";
import { formatDate } from "@/shared/ui/format";
import { RandomizeIcon } from "@/shared/ui/icons";
import { Notice, Panel } from "@/shared/ui/Panel";
import { Tabs } from "@/shared/ui/Tabs";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";
import { ActionBar } from "@/widgets/configurator/ActionBar";
import { configuratorDraftName } from "@/widgets/configurator/draftName";

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
  resumeConfig?: ManagedRunConfig | null;
  resumeDraftName?: string | null;
  onConfigChange?: (config: ManagedRunConfig) => void;
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
  resumeConfig = null,
  resumeDraftName = null,
  onConfigChange,
  onDraftNameChange,
  onLaunchRun,
  onSaveDraft,
  onUpdateDraft,
}: ConfiguratorProps) {
  const baselineConfig = loadedDraft?.config ?? initialConfig ?? baseConfig;
  const baselineDraftName = configuratorDraftName(baseConfig, initialDraftName, loadedDraft);
  const [draftName, setDraftName] = useState(resumeDraftName ?? baselineDraftName);
  const [config, setConfig] = useState(resumeConfig ?? baselineConfig);
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
  const seedInput = useEditableNumberInput({
    format: String,
    formattedValue: String(config.seed),
    normalize: Math.round,
    onCommit: (seed) => setConfig((currentConfig) => ({ ...currentConfig, seed })),
    parse: (rawValue) => parseSafeIntegerInput(rawValue, { min: 0 }),
  });
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
  const baselineConfigJson = useMemo(() => stableJson(baselineConfig), [baselineConfig]);
  const configJson = useMemo(() => stableJson(config), [config]);
  const isDirty = useMemo(() => {
    if (normalizedDraftName !== normalizedBaselineDraftName) {
      return true;
    }
    return configJson !== baselineConfigJson;
  }, [baselineConfigJson, configJson, normalizedBaselineDraftName, normalizedDraftName]);
  const notifyConfigChange = useEffectEvent((nextConfig: ManagedRunConfig) => {
    onConfigChange?.(nextConfig);
  });
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
    const controller = new AbortController();
    setPreviewError(null);
    void fetchPolicyPreview(previewConfig, { signal: controller.signal })
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
      controller.abort();
    };
  }, [active, previewConfig]);

  useEffect(() => {
    notifyDraftNameChange(draftName);
  }, [draftName]);

  useEffect(() => {
    notifyConfigChange(config);
  }, [config]);

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
  const configIssue = fixedEnvAssignmentIssue(config);
  const canSubmitConfig = configIssue === null;

  return (
    <Panel>
      <ActionBar
        canSave={
          !isSaving &&
          !isUpdating &&
          !isTraining &&
          canSubmitConfig &&
          !createNameConflict &&
          normalizedDraftName.length > 0
        }
        canTrain={
          !isSaving &&
          !isUpdating &&
          !isTraining &&
          canSubmitConfig &&
          normalizedDraftName.length > 0
        }
        canUpdate={
          loadedDraft !== null &&
          !isSaving &&
          !isUpdating &&
          !isTraining &&
          canSubmitConfig &&
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
      {configIssue !== null ? (
        <div className="configurator-feedback-stack">
          <Notice tone="error">{configIssue}</Notice>
        </div>
      ) : null}

      <ConfigGrid className="mb-7 grid-cols-[minmax(0,1fr)_240px] items-end">
        <FieldShell>
          <FieldLabel help="Run name used when this configuration is launched." label="Run name" />
          <FieldInput
            aria-invalid={nameError !== null}
            aria-label="Run name"
            spellCheck={false}
            value={draftName}
            onChange={(event) => setDraftName(event.target.value)}
          />
        </FieldShell>
        <div className="grid grid-cols-[minmax(0,1fr)_40px] items-end gap-2">
          <FieldShell>
            <FieldLabel
              help="Base random seed used when the run config is generated."
              label="Seed"
            />
            <FieldInput
              aria-label="Seed"
              {...editableNumberInputProps("integer")}
              value={seedInput.rawValue}
              onBlur={seedInput.commitRawValue}
              onChange={(event) => seedInput.changeRawValue(event.target.value)}
              onKeyDown={blurOnEnter}
            />
          </FieldShell>
          <TooltipIconButton
            aria-label="Randomize seed"
            tooltip="Randomize seed"
            onClick={randomizeSeed}
          >
            <RandomizeIcon />
          </TooltipIconButton>
        </div>
        {forkSourceRunLabel !== null && forkSourceArtifact !== null ? (
          <section
            aria-label="Fork source context"
            className="col-span-full grid gap-2 border border-app-border bg-app-surface-muted px-3.5 py-3"
          >
            <div className="flex min-w-0 items-center gap-2.5">
              <span className="text-xs font-bold text-app-muted">Forked from</span>
              <strong className="min-w-0 text-sm font-bold text-app-text">
                {forkSourceRunLabel}
              </strong>
              <span className="inline-flex h-6 items-center border border-app-border bg-app-surface px-2 text-xs font-bold text-app-muted lowercase">
                {forkSourceArtifact} checkpoint
              </span>
            </div>
            {forkedAtLabel !== null || forkSourceStepCount !== null ? (
              <div className="flex flex-wrap gap-2.5 text-xs text-app-muted">
                {forkedAtLabel !== null ? <span>forked {forkedAtLabel}</span> : null}
                {forkSourceStepCount !== null ? (
                  <span>@ {forkSourceStepCount.toLocaleString()} steps</span>
                ) : null}
              </div>
            ) : null}
          </section>
        ) : null}
      </ConfigGrid>
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
        <EnvironmentSection
          config={config}
          defaultConfig={baseConfig}
          metadata={metadata}
          setConfig={setConfig}
        />
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
