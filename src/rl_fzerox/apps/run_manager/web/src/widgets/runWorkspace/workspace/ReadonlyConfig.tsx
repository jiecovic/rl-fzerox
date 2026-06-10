// src/rl_fzerox/apps/run_manager/web/src/widgets/runWorkspace/workspace/ReadonlyConfig.tsx
import type { ReactNode } from "react";
import type {
  ConfigMetadata,
  ManagedRunDetail,
  PolicyArchitecturePreview,
} from "@/shared/api/contract";
import { Tabs } from "@/shared/ui/Tabs";
import {
  CONFIG_SECTION_TABS,
  type ConfigSection,
} from "@/widgets/configurator/configurator/sections";
import { ActionSection } from "@/widgets/configurator/sections/ActionSection";
import { EnvironmentSection } from "@/widgets/configurator/sections/EnvironmentSection";
import { LoggingSection } from "@/widgets/configurator/sections/LoggingSection";
import { ObservationSection } from "@/widgets/configurator/sections/ObservationSection";
import { PolicySection } from "@/widgets/configurator/sections/PolicySection";
import { RewardSection } from "@/widgets/configurator/sections/RewardSection";
import { TracksSection } from "@/widgets/configurator/sections/TracksSection";
import { TrainingSection } from "@/widgets/configurator/sections/TrainingSection";
import { VehicleSection } from "@/widgets/configurator/sections/VehicleSection";

interface RunReadonlyConfigProps {
  metadata: ConfigMetadata;
  onSectionChange: (section: ConfigSection) => void;
  policyPreview: PolicyArchitecturePreview | null;
  run: ManagedRunDetail;
  section: ConfigSection;
}

interface ReadonlySectionRendererProps {
  metadata: ConfigMetadata;
  policyPreview: PolicyArchitecturePreview | null;
  run: ManagedRunDetail;
}

const NOOP_SET_CONFIG = () => undefined;

const SECTION_RENDERERS: Record<ConfigSection, (props: ReadonlySectionRendererProps) => ReactNode> =
  {
    action: ({ metadata, run }) => (
      <ActionSection
        config={run.config}
        defaultConfig={run.config}
        metadata={metadata}
        setConfig={NOOP_SET_CONFIG}
      />
    ),
    environment: ({ metadata, run }) => (
      <EnvironmentSection
        config={run.config}
        defaultConfig={run.config}
        metadata={metadata}
        setConfig={NOOP_SET_CONFIG}
      />
    ),
    logging: ({ run }) => (
      <LoggingSection config={run.config} defaultConfig={run.config} setConfig={NOOP_SET_CONFIG} />
    ),
    observation: ({ metadata, policyPreview, run }) => (
      <ObservationSection
        config={run.config}
        defaultConfig={run.config}
        metadata={metadata}
        preview={policyPreview}
        setConfig={NOOP_SET_CONFIG}
      />
    ),
    policy: ({ metadata, policyPreview, run }) => (
      <PolicySection
        config={run.config}
        defaultConfig={run.config}
        metadata={metadata}
        preview={policyPreview}
        setConfig={NOOP_SET_CONFIG}
      />
    ),
    reward: ({ run }) => (
      <RewardSection config={run.config} defaultConfig={run.config} setConfig={NOOP_SET_CONFIG} />
    ),
    tracks: ({ metadata, run }) => (
      <TracksSection
        config={run.config}
        defaultConfig={run.config}
        metadata={metadata}
        setConfig={NOOP_SET_CONFIG}
      />
    ),
    training: ({ run }) => (
      <TrainingSection config={run.config} defaultConfig={run.config} setConfig={NOOP_SET_CONFIG} />
    ),
    vehicle: ({ metadata, run }) => (
      <VehicleSection
        config={run.config}
        defaultConfig={run.config}
        metadata={metadata}
        setConfig={NOOP_SET_CONFIG}
      />
    ),
  };

export function RunReadonlyConfig({
  metadata,
  onSectionChange,
  policyPreview,
  run,
  section,
}: RunReadonlyConfigProps) {
  return (
    <>
      <div className="section-tabs-row">
        <Tabs
          label="Run configuration sections"
          activeId={section}
          items={CONFIG_SECTION_TABS}
          variant="section"
          onSelect={onSectionChange}
        />
      </div>

      <fieldset className="readonly-config-fieldset" disabled>
        <div className="readonly-config-shell">
          {SECTION_RENDERERS[section]({ metadata, policyPreview, run })}
        </div>
      </fieldset>
    </>
  );
}
