import { useState } from "react";

import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import type { ManagedRunConfig } from "@/shared/api/contract";

import {
  allRewardSectionsOpen,
  initialRewardDisclosureState,
  type RewardDisclosureId,
  type RewardDisclosureState,
} from "./reward/disclosureState";
import { RewardPanels } from "./reward/RewardPanels";

interface ConfigSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  setConfig: (config: ManagedRunConfig) => void;
}

export function RewardSection({ config, defaultConfig, setConfig }: ConfigSectionProps) {
  const [openSections, setOpenSections] = useState<RewardDisclosureState>(
    initialRewardDisclosureState,
  );
  const updateReward = (patch: Partial<ManagedRunConfig["reward"]>) => {
    setConfig({ ...config, reward: { ...config.reward, ...patch } });
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
        updateReward={updateReward}
      />
    </div>
  );
}
