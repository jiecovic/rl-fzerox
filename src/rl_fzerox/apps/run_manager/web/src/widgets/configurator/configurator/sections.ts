// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/configurator/sections.ts
export type ConfigSection =
  | "training"
  | "observation"
  | "policy"
  | "reward"
  | "tracks"
  | "vehicle"
  | "action"
  | "environment"
  | "logging";

export const CONFIG_SECTION_TABS = [
  { id: "training", label: "Training" },
  { id: "observation", label: "Observation" },
  { id: "policy", label: "Policy" },
  { id: "reward", label: "Reward" },
  { id: "tracks", label: "Tracks" },
  { id: "vehicle", label: "Vehicle" },
  { id: "action", label: "Action" },
  { id: "environment", label: "Environment" },
  { id: "logging", label: "Logging" },
] satisfies { id: ConfigSection; label: string }[];
