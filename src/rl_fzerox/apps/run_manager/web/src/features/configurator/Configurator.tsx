import { useState } from "react";

import { Notice, Panel, PanelHeader } from "@/components/ui/Panel";
import type { ManagedRunConfig } from "@/contract";
import { FieldLabel } from "@/features/configurator/fields";
import { LoggingSection } from "@/features/configurator/LoggingSection";
import { ModelSection } from "@/features/configurator/ModelSection";
import { RewardSection } from "@/features/configurator/RewardSection";
import { TrainingSection } from "@/features/configurator/TrainingSection";

interface ConfiguratorProps {
  baseConfig: ManagedRunConfig;
  onSaveDraft: (name: string, config: ManagedRunConfig) => Promise<void>;
}

type ConfigSection = "training" | "observation" | "reward" | "logging";

export function Configurator({ baseConfig, onSaveDraft }: ConfiguratorProps) {
  const [draftName, setDraftName] = useState("ppo_allcups_recurrent");
  const [config, setConfig] = useState(baseConfig);
  const [section, setSection] = useState<ConfigSection>("training");
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      <div className="form-grid run-identity-grid">
        <label>
          <FieldLabel help="Run name used when this configuration is launched." label="Run name" />
          <input
            spellCheck={false}
            value={draftName}
            onChange={(event) => setDraftName(event.target.value)}
          />
        </label>
        <div className="seed-control">
          <label>
            <FieldLabel
              help="Base random seed used when the run config is generated."
              label="Seed"
            />
            <input
              type="number"
              value={config.seed}
              onChange={(event) => setConfig({ ...config, seed: Number(event.target.value) })}
            />
          </label>
          <button
            aria-label="Randomize seed"
            className="icon-button seed-randomize-button"
            title="Randomize seed"
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
            label="Observation / policy"
            onClick={() => setSection("observation")}
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
            className="secondary-button"
            type="button"
            disabled={isSaving}
            onClick={() => void saveDraft()}
          >
            {isSaving ? "Saving..." : "Save draft"}
          </button>
          <button className="primary-button" type="button" disabled>
            Train
          </button>
        </div>
      </div>

      {section === "training" ? (
        <TrainingSection config={config} defaultConfig={baseConfig} setConfig={setConfig} />
      ) : null}
      {section === "observation" ? (
        <ModelSection config={config} defaultConfig={baseConfig} setConfig={setConfig} />
      ) : null}
      {section === "reward" ? (
        <RewardSection config={config} defaultConfig={baseConfig} setConfig={setConfig} />
      ) : null}
      {section === "logging" ? (
        <LoggingSection config={config} defaultConfig={baseConfig} setConfig={setConfig} />
      ) : null}

      {error !== null ? <Notice tone="error">{error}</Notice> : null}
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
