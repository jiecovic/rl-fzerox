// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/RewardSection.tsx

import {
  type ConfigSectionPatch,
  type ConfigSetter,
  patchConfigSection,
} from "@/features/configurator/configurator/state";
import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/features/configurator/disclosureState";
import type { ManagedRunConfig } from "@/shared/api/contract";

import {
  allRewardSectionsOpen,
  type RewardDisclosureId,
  type RewardDisclosureState,
} from "./reward/disclosureState";
import { RewardPanels } from "./reward/RewardPanels";

interface ConfigSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  setConfig: ConfigSetter;
}

export function RewardSection({ config, defaultConfig, setConfig }: ConfigSectionProps) {
  const [openSections, setOpenSections] = usePersistentDisclosureMap<RewardDisclosureState>(
    "run-manager:reward:sections",
    allRewardSectionsOpen(false),
  );
  const updateReward = (patch: Partial<ManagedRunConfig["reward"]>) => {
    patchConfigSection(setConfig, "reward", patch);
  };
  const updateAction = (patch: ConfigSectionPatch<"action">) => {
    patchConfigSection(setConfig, "action", patch);
  };

  const setSectionOpen = (id: RewardDisclosureId, open: boolean) => {
    setOpenSections((current) => ({ ...current, [id]: open }));
  };

  return (
    <div className="reward-accordion-stack">
      <DisclosureToolbar
        collapseLabel="Collapse all reward sections"
        expandLabel="Expand all reward sections"
        onCollapseAll={() => setOpenSections(allRewardSectionsOpen(false))}
        onExpandAll={() => setOpenSections(allRewardSectionsOpen(true))}
      />
      <RewardPanels
        config={config}
        defaultConfig={defaultConfig}
        openSections={openSections}
        setSectionOpen={setSectionOpen}
        updateAction={updateAction}
        updateReward={updateReward}
      />
    </div>
  );
}
