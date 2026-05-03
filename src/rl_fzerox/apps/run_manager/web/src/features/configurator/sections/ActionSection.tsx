import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/features/configurator/disclosureState";
import { AuxiliaryBranchesDisclosure } from "@/features/configurator/sections/action/AuxiliaryBranchesDisclosure";
import { ControlFamilyDisclosure } from "@/features/configurator/sections/action/ControlFamilyDisclosure";
import {
  actionCompatibilityNote,
  actionSummaryRows,
  normalizedActionConfig,
} from "@/features/configurator/sections/action/model";
import type { ActionSectionProps } from "@/features/configurator/sections/action/types";
import type { ManagedRunConfig } from "@/shared/api/contract";

type ActionDisclosureState = Record<"auxiliary" | "family", boolean>;

export function ActionSection({ config, defaultConfig, metadata, setConfig }: ActionSectionProps) {
  const action = config.action;
  const compatibilityNote = actionCompatibilityNote(action);
  const [openSections, setOpenSections] = usePersistentDisclosureMap<ActionDisclosureState>(
    "run-manager:action:sections",
    {
      auxiliary: false,
      family: false,
    },
  );

  const updatePolicy = (patch: Partial<ManagedRunConfig["policy"]>) => {
    setConfig({ ...config, policy: { ...config.policy, ...patch } });
  };

  const updateAction = (patch: Partial<ManagedRunConfig["action"]>) => {
    setConfig({
      ...config,
      action: normalizedActionConfig({
        ...config.action,
        ...patch,
      }),
    });
  };
  const setSectionOpen = (section: keyof typeof openSections, open: boolean) => {
    setOpenSections((current) => ({ ...current, [section]: open }));
  };

  return (
    <div className="config-stack action-disclosure-stack">
      <DisclosureToolbar
        collapseLabel="Collapse all action sections"
        expandLabel="Expand all action sections"
        onCollapseAll={() => setOpenSections({ auxiliary: false, family: false })}
        onExpandAll={() => setOpenSections({ auxiliary: true, family: true })}
      />
      <ControlFamilyDisclosure
        config={config}
        defaultConfig={defaultConfig}
        metadata={metadata}
        open={openSections.family}
        setOpen={(open) => setSectionOpen("family", open)}
        setConfig={setConfig}
        updateAction={updateAction}
        updatePolicy={updatePolicy}
      />

      <AuxiliaryBranchesDisclosure
        config={config}
        defaultConfig={defaultConfig}
        metadata={metadata}
        open={openSections.auxiliary}
        setOpen={(open) => setSectionOpen("auxiliary", open)}
        updateAction={updateAction}
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
    </div>
  );
}
