// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/reward/disclosureState.ts
export type RewardDisclosureId =
  | "time"
  | "progress"
  | "airborne"
  | "track"
  | "energy"
  | "actions"
  | "damage";

export type RewardDisclosureState = Record<RewardDisclosureId, boolean>;

export function initialRewardDisclosureState(): RewardDisclosureState {
  return allRewardSectionsOpen(true);
}

export function allRewardSectionsOpen(open: boolean): RewardDisclosureState {
  return {
    actions: open,
    airborne: open,
    damage: open,
    energy: open,
    progress: open,
    time: open,
    track: open,
  };
}
