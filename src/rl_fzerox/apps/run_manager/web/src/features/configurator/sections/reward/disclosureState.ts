export type RewardDisclosureId =
  | "time"
  | "progress"
  | "airborne"
  | "track"
  | "energy"
  | "actions"
  | "lean"
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
    lean: open,
    progress: open,
    time: open,
    track: open,
  };
}
