// web/run-manager/src/entities/runConfig/ui/sections/reward/types.ts

import type { ConfigSectionPatch } from "@/entities/runConfig/model/state";
import type { RewardDisclosureState } from "@/entities/runConfig/ui/sections/reward/disclosureState";
import type { ManagedRunConfig } from "@/shared/api/contract";

export interface RewardPanelProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  openSections: RewardDisclosureState;
  setSectionOpen: (id: keyof RewardDisclosureState, open: boolean) => void;
  updateAction: (patch: ConfigSectionPatch<"action">) => void;
  updateReward: (patch: Partial<ManagedRunConfig["reward"]>) => void;
}
