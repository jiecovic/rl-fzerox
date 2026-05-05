import { type ReactNode, useLayoutEffect, useRef, useState } from "react";

import {
  CONFIG_SECTION_TABS,
  type ConfigSection,
} from "@/features/configurator/configurator/sections";
import { ActionSection } from "@/features/configurator/sections/ActionSection";
import { EnvironmentSection } from "@/features/configurator/sections/EnvironmentSection";
import { LoggingSection } from "@/features/configurator/sections/LoggingSection";
import { ObservationSection } from "@/features/configurator/sections/ObservationSection";
import { PolicySection } from "@/features/configurator/sections/PolicySection";
import { RewardSection } from "@/features/configurator/sections/RewardSection";
import { TracksSection } from "@/features/configurator/sections/TracksSection";
import { TrainingSection } from "@/features/configurator/sections/TrainingSection";
import { VehicleSection } from "@/features/configurator/sections/VehicleSection";
import type { ConfigMetadata, ManagedRun, PolicyArchitecturePreview } from "@/shared/api/contract";
import { Tabs } from "@/shared/ui/Tabs";

interface RunReadonlyConfigProps {
  metadata: ConfigMetadata;
  policyPreview: PolicyArchitecturePreview | null;
  run: ManagedRun;
}

interface ReadonlySectionRendererProps {
  metadata: ConfigMetadata;
  policyPreview: PolicyArchitecturePreview | null;
  run: ManagedRun;
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
    environment: ({ run }) => (
      <EnvironmentSection
        config={run.config}
        defaultConfig={run.config}
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

export function RunReadonlyConfig({ metadata, policyPreview, run }: RunReadonlyConfigProps) {
  const [section, setSection] = useState<ConfigSection>("training");
  const shellRef = useRef<HTMLDivElement | null>(null);

  useLayoutEffect(() => {
    const shell = shellRef.current;
    if (shell === null) {
      return;
    }
    // Readonly run config still reuses editable form components. Disable native
    // controls at the DOM boundary so spinner arrows and keyboard input cannot
    // mutate displayed values.
    for (const control of shell.querySelectorAll<
      HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement
    >("input, select, textarea")) {
      control.disabled = true;
      if (control instanceof HTMLInputElement || control instanceof HTMLTextAreaElement) {
        control.readOnly = true;
      }
      control.tabIndex = -1;
      control.setAttribute("aria-readonly", "true");
    }
  });

  return (
    <>
      <div className="section-tabs-row">
        <Tabs
          label="Run configuration sections"
          activeId={section}
          items={CONFIG_SECTION_TABS}
          variant="section"
          onSelect={(id) => setSection(id)}
        />
      </div>

      <div ref={shellRef} className="readonly-config-shell">
        {SECTION_RENDERERS[section]({ metadata, policyPreview, run })}
      </div>
    </>
  );
}
