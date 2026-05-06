// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/reward/types.ts
import type { ManagedRunConfig } from "@/shared/api/contract";

import type { RewardDisclosureState } from "./disclosureState";

export interface RewardPanelProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  openSections: RewardDisclosureState;
  setSectionOpen: (id: keyof RewardDisclosureState, open: boolean) => void;
  updateReward: (patch: Partial<ManagedRunConfig["reward"]>) => void;
}
