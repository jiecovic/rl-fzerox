import { useEffect, useState } from "react";
import { FieldLabel } from "@/features/configurator/fields";
import { LoggingSection } from "@/features/configurator/sections/LoggingSection";
import { ObservationSection } from "@/features/configurator/sections/ObservationSection";
import { PolicySection } from "@/features/configurator/sections/PolicySection";
import { RewardSection } from "@/features/configurator/sections/RewardSection";
import { TrainingSection } from "@/features/configurator/sections/TrainingSection";
import { fetchPolicyPreview } from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRunConfig,
  PolicyArchitecturePreview,
} from "@/shared/api/contract";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

interface ConfiguratorProps {
  baseConfig: ManagedRunConfig;
  loadedDraft: ManagedDraft | null;
  metadata: ConfigMetadata;
  onSaveDraft: (name: string, config: ManagedRunConfig) => Promise<ManagedDraft>;
  onUpdateDraft: (id: string, name: string, config: ManagedRunConfig) => Promise<ManagedDraft>;
}

type ConfigSection = "training" | "observation" | "policy" | "reward" | "logging";

export function Configurator({
  baseConfig,
  loadedDraft,
  metadata,
  onSaveDraft,
  onUpdateDraft,
}: ConfiguratorProps) {
  const [draftName, setDraftName] = useState("ppo_allcups_recurrent");
  const [config, setConfig] = useState(baseConfig);
  const [section, setSection] = useState<ConfigSection>("training");
  const [policyPreview, setPolicyPreview] = useState<PolicyArchitecturePreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (loadedDraft !== null) {
      setDraftName(loadedDraft.name);
      setConfig(loadedDraft.config);
      setSection("training");
      setError(null);
      return;
    }
    setDraftName("ppo_allcups_recurrent");
    setConfig(baseConfig);
    setSection("training");
    setError(null);
  }, [baseConfig, loadedDraft]);

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

  async function saveDraft() {
    setIsSaving(true);
    setError(null);
    try {
      await onSaveDraft(draftName, config);
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
    setIsUpdating(true);
    setError(null);
    try {
      await onUpdateDraft(loadedDraft.id, draftName, config);
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

  return (
    <Panel>
      <PanelHeader
        title="Run configurator"
        subtitle="Configure a run, then save it as a draft or start training from here."
      />
      {loadedDraft !== null ? (
        <div className="configurator-status">
          <Notice>Editing draft `{loadedDraft.name}`. Save as new or update in place.</Notice>
        </div>
      ) : null}
      <div className="form-grid run-identity-grid">
        <div className="field-shell">
          <FieldLabel help="Run name used when this configuration is launched." label="Run name" />
          <input
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

      <div className="section-tabs-row">
        <div className="section-tabs">
          <SectionButton
            active={section === "training"}
            label="Training"
            onClick={() => setSection("training")}
          />
          <SectionButton
            active={section === "observation"}
            label="Observation"
            onClick={() => setSection("observation")}
          />
          <SectionButton
            active={section === "policy"}
            label="Policy"
            onClick={() => setSection("policy")}
          />
          <SectionButton
            active={section === "reward"}
            label="Reward"
            onClick={() => setSection("reward")}
          />
          <SectionButton
            active={section === "logging"}
            label="Logging"
            onClick={() => setSection("logging")}
          />
        </div>
        <div className="section-actions">
          <button
            className="secondary-button draft-action-button"
            type="button"
            disabled={isSaving || isUpdating}
            onClick={() => void saveDraft()}
          >
            <SaveDraftIcon />
            <span>
              {isSaving ? "Saving..." : loadedDraft === null ? "Save draft" : "Save as new draft"}
            </span>
          </button>
          {loadedDraft !== null ? (
            <button
              className="secondary-button draft-action-button draft-update-button"
              type="button"
              disabled={isSaving || isUpdating}
              onClick={() => void updateDraft()}
            >
              <UpdateDraftIcon />
              <span>{isUpdating ? "Updating..." : "Update draft"}</span>
            </button>
          ) : null}
          <button className="primary-button" type="button" disabled>
            Train
          </button>
        </div>
      </div>

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
      {section === "logging" ? (
        <LoggingSection config={config} defaultConfig={baseConfig} setConfig={setConfig} />
      ) : null}

      {error !== null ? <Notice tone="error">{error}</Notice> : null}
      {previewError !== null ? <Notice tone="error">{previewError}</Notice> : null}
    </Panel>
  );
}

function SectionButton({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      className={active ? "section-tab active" : "section-tab"}
      type="button"
      onClick={onClick}
    >
      {label}
    </button>
  );
}

function RandomizeIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="18" viewBox="0 0 20 20" width="18">
      <path
        d="M15.2 7.2a5.6 5.6 0 0 0-9.7-1.4M4.8 12.8a5.6 5.6 0 0 0 9.7 1.4"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.7"
      />
      <path
        d="M15.6 3.8v3.6h-3.6M4.4 16.2v-3.6h3.6"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      />
    </svg>
  );
}

function SaveDraftIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 20 20" width="14">
      <path
        d="M4.5 3.5h8l3 3v10h-11zM7 3.5v5h6v-5M7 13h6"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}

function UpdateDraftIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 20 20" width="14">
      <path
        d="M4 10a6 6 0 0 1 10.2-4.2M16 10a6 6 0 0 1-10.2 4.2"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.5"
      />
      <path
        d="M12.8 3.9h1.9v1.9M7.2 16.1H5.3v-1.9"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}
