// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/logging/derived.ts
import type { ManagedRunConfig } from "@/shared/api/contract";

const recentRetentionScale = {
  finiteMax: 50,
  unlimitedSentinel: 51,
} as const;

export function checkpointSpanSteps(train: ManagedRunConfig["train"]) {
  return train.num_envs * train.n_steps * train.checkpoint_every_rollouts;
}

export function checkpointCadenceSummary(train: ManagedRunConfig["train"]) {
  const envSteps = checkpointSpanSteps(train).toLocaleString();
  return `${checkpointRolloutLabel(train)} = ${envSteps} env steps`;
}

export function checkpointRolloutLabel(train: ManagedRunConfig["train"]) {
  return `${train.checkpoint_every_rollouts} rollout${train.checkpoint_every_rollouts === 1 ? "" : "s"}`;
}

export function recentRetentionSummary(train: ManagedRunConfig["train"]) {
  if (!train.save_recent_checkpoints) {
    return "off";
  }
  if (train.recent_checkpoint_limit === null) {
    return "unlimited";
  }
  return `keep last ${train.recent_checkpoint_limit}`;
}

export function recentRetentionSliderValue(train: ManagedRunConfig["train"]) {
  return train.recent_checkpoint_limit ?? recentRetentionScale.unlimitedSentinel;
}

export function recentRetentionValueFromSlider(value: number) {
  return value >= recentRetentionScale.unlimitedSentinel ? null : value;
}

export function recentRetentionSliderTicks() {
  return [
    { value: 1, label: "1" },
    { value: 10, label: "10" },
    { value: 25, label: "25" },
    { value: recentRetentionScale.unlimitedSentinel, label: "∞" },
  ] as const;
}

export function recentRetentionFiniteMax() {
  return recentRetentionScale.finiteMax;
}

export function recentRetentionSliderMax() {
  return recentRetentionScale.unlimitedSentinel;
}

export function checkpointSummaryRows(train: ManagedRunConfig["train"]) {
  return [
    {
      label: "Periodic cadence",
      detail: checkpointRolloutLabel(train),
      value: `${checkpointSpanSteps(train).toLocaleString()} env steps`,
    },
    {
      label: "Latest artifact",
      detail: "rolling checkpoints/latest/{model,policy}.zip",
      value: train.save_latest_checkpoint ? "on" : "off",
    },
    {
      label: "Best artifact",
      detail: "overwrite on improved episode return",
      value: train.save_best_checkpoint ? "on" : "off",
    },
    {
      label: "Recent snapshots",
      detail: "numbered snapshots in checkpoints/",
      value: recentRetentionSummary(train),
    },
    {
      label: "Final artifact",
      detail: "saved once at training end",
      value: "always",
    },
  ];
}
