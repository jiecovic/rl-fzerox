import type { ManagedRunConfig } from "@/shared/api/contract";

import type { RewardDisclosureState } from "./disclosureState";

export interface RewardPanelProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  openSections: RewardDisclosureState;
  setSectionOpen: (id: keyof RewardDisclosureState, open: boolean) => void;
  updateReward: (patch: Partial<ManagedRunConfig["reward"]>) => void;
}
