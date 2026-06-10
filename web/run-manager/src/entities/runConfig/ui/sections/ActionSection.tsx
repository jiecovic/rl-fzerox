// web/run-manager/src/entities/runConfig/ui/sections/ActionSection.tsx

import { type ConfigSectionPatch, patchConfigSection } from "@/entities/runConfig/model/state";
import {
  ActionNote,
  ActionSummaryGrid,
  ActionSummaryItem,
} from "@/entities/runConfig/ui/sections/action/ActionLayout";
import { AuxiliaryBranchesDisclosure } from "@/entities/runConfig/ui/sections/action/AuxiliaryBranchesDisclosure";
import { ControlFamilyDisclosure } from "@/entities/runConfig/ui/sections/action/ControlFamilyDisclosure";
import { EntropyGroupWeightsPanel } from "@/entities/runConfig/ui/sections/action/EntropyGroupWeightsPanel";
import {
  actionCompatibilityNote,
  actionSummaryRows,
  normalizedActionConfig,
} from "@/entities/runConfig/ui/sections/action/model";
import type { ActionSectionProps } from "@/entities/runConfig/ui/sections/action/types";
import { ConfigStack } from "@/shared/ui/config/ConfigLayout";
import { ConfigPanel } from "@/shared/ui/config/ConfigPanel";
import { DisclosureToolbar } from "@/shared/ui/config/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/shared/ui/config/disclosureState";

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
    <ConfigStack className="gap-2.5 [&_.config-disclosure-body]:gap-[18px]">
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
        <ActionSummaryGrid>
          {actionSummaryRows(action).map((row) => (
            <ActionSummaryItem key={row.label}>
              <span>{row.label}</span>
              <strong>{row.value}</strong>
            </ActionSummaryItem>
          ))}
        </ActionSummaryGrid>
        {compatibilityNote !== null ? <ActionNote>{compatibilityNote}</ActionNote> : null}
      </ConfigPanel>
    </ConfigStack>
  );
}
