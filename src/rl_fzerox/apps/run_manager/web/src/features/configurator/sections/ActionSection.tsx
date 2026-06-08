// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/ActionSection.tsx

import { ConfigStack } from "@/features/configurator/ConfigLayout";
import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import {
  type ConfigSectionPatch,
  patchConfigSection,
} from "@/features/configurator/configurator/state";
import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/features/configurator/disclosureState";
import { AuxiliaryBranchesDisclosure } from "@/features/configurator/sections/action/AuxiliaryBranchesDisclosure";
import { ControlFamilyDisclosure } from "@/features/configurator/sections/action/ControlFamilyDisclosure";
import { EntropyGroupWeightsPanel } from "@/features/configurator/sections/action/EntropyGroupWeightsPanel";
import {
  actionCompatibilityNote,
  actionSummaryRows,
  normalizedActionConfig,
} from "@/features/configurator/sections/action/model";
import type { ActionSectionProps } from "@/features/configurator/sections/action/types";

type ActionDisclosureState = Record<"auxiliary" | "family", boolean>;

export function ActionSection({
  config,
  defaultConfig,
  checkpointLocked = false,
  metadata,
  setConfig,
}: ActionSectionProps) {
  const action = config.action;
  const compatibilityNote = actionCompatibilityNote(action);
  const [openSections, setOpenSections] = usePersistentDisclosureMap<ActionDisclosureState>(
    "run-manager:action:sections",
    {
      auxiliary: false,
      family: false,
    },
  );

  const updatePolicy = (patch: ConfigSectionPatch<"policy">) => {
    patchConfigSection(setConfig, "policy", patch);
  };

  const updateTrain = (patch: ConfigSectionPatch<"train">) => {
    patchConfigSection(setConfig, "train", patch);
  };

  const updateAction = (patch: ConfigSectionPatch<"action">) => {
    setConfig((currentConfig) => {
      const actionPatch = typeof patch === "function" ? patch(currentConfig) : patch;
      return {
        ...currentConfig,
        action: normalizedActionConfig({
          ...currentConfig.action,
          ...actionPatch,
        }),
      };
    });
  };
  const setSectionOpen = (section: keyof typeof openSections, open: boolean) => {
    setOpenSections((current) => ({ ...current, [section]: open }));
  };

  return (
    <ConfigStack className="action-disclosure-stack">
      <DisclosureToolbar
        collapseLabel="Collapse all action sections"
        expandLabel="Expand all action sections"
        onCollapseAll={() => setOpenSections({ auxiliary: false, family: false })}
        onExpandAll={() => setOpenSections({ auxiliary: true, family: true })}
      />
      <ControlFamilyDisclosure
        checkpointLocked={checkpointLocked}
        config={config}
        defaultConfig={defaultConfig}
        metadata={metadata}
        open={openSections.family}
        setOpen={(open) => setSectionOpen("family", open)}
        setConfig={setConfig}
        updateAction={updateAction}
        updatePolicy={updatePolicy}
        updateTrain={updateTrain}
      />

      <AuxiliaryBranchesDisclosure
        checkpointLocked={checkpointLocked}
        config={config}
        defaultConfig={defaultConfig}
        metadata={metadata}
        open={openSections.auxiliary}
        setOpen={(open) => setSectionOpen("auxiliary", open)}
        updateAction={updateAction}
        updatePolicy={updatePolicy}
        updateTrain={updateTrain}
      />

      <EntropyGroupWeightsPanel
        action={config.action}
        defaultTrain={defaultConfig.train}
        train={config.train}
        updateTrain={updateTrain}
      />

      <ConfigPanel title="Action surface" wide>
        <div className="action-summary-grid">
          {actionSummaryRows(action).map((row) => (
            <div className="action-summary-item" key={row.label}>
              <span>{row.label}</span>
              <strong>{row.value}</strong>
            </div>
          ))}
        </div>
        {compatibilityNote !== null ? <p className="action-note">{compatibilityNote}</p> : null}
      </ConfigPanel>
    </ConfigStack>
  );
}
