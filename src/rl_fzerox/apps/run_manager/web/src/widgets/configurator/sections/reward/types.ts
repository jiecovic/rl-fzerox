// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/reward/types.ts

import type { ManagedRunConfig } from "@/shared/api/contract";
import type { ConfigSectionPatch } from "@/widgets/configurator/configurator/state";
import type { RewardDisclosureState } from "@/widgets/configurator/sections/reward/disclosureState";

export interface RewardPanelProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  openSections: RewardDisclosureState;
  setSectionOpen: (id: keyof RewardDisclosureState, open: boolean) => void;
  updateAction: (patch: ConfigSectionPatch<"action">) => void;
  updateReward: (patch: Partial<ManagedRunConfig["reward"]>) => void;
}
