// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/reward/types.ts
import type { ConfigSectionPatch } from "@/features/configurator/configurator/state";
import type { RewardDisclosureState } from "@/features/configurator/sections/reward/disclosureState";
import type { ManagedRunConfig } from "@/shared/api/contract";

export interface RewardPanelProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  openSections: RewardDisclosureState;
  setSectionOpen: (id: keyof RewardDisclosureState, open: boolean) => void;
  updateAction: (patch: ConfigSectionPatch<"action">) => void;
  updateReward: (patch: Partial<ManagedRunConfig["reward"]>) => void;
}
