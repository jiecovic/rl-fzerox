// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/options.ts
import type { TracksConfig } from "@/features/configurator/sections/tracks/types";

export const X_CUP = {
  id: "x",
  label: "X Cup",
} as const;

export const RACE_MODE_DESCRIPTIONS: Record<TracksConfig["race_mode"], string> = {
  time_attack: "Single-course time-trial episodes.",
  gp_race: "Grand Prix race rules across the selected pool.",
};

export const TRACK_SAMPLING_DESCRIPTIONS: Record<TracksConfig["sampling_mode"], string> = {
  equal: "Sample courses uniformly by episode.",
  step_balanced: "Bias toward courses with fewer recent frames.",
  adaptive_step_balanced: "Keep step balance, then tilt a bit toward lower-completion courses.",
  deficit_budget:
    "Deterministically spend step-budget slices: equal coverage plus difficulty focus.",
  fixed_env: "Pin each vector env slot to one course for deterministic per-rollout coverage.",
};

export function formatTrackOptionLabel(value: string) {
  return value
    .split("_")
    .map((word) => (word === "gp" ? "GP" : word.charAt(0).toUpperCase() + word.slice(1)))
    .join(" ");
}

export function shortCupLabel(label: string) {
  return label.replace(" Cup", "");
}
