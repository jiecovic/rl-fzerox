// web/run-manager/src/entities/runConfig/ui/sections/environment/derived.ts
import type { ManagedRunConfig } from "@/shared/api/contract";
import { formatDecimal, formatDurationSeconds, formatInteger } from "@/shared/ui/format";

export function episodeFrameSummary(config: ManagedRunConfig) {
  return `${formatInteger(config.environment.max_episode_steps)} internal frames ≈ ${formatDurationSeconds(config.environment.max_episode_steps / 60)} of game time`;
}

export function episodeDecisionSummary(config: ManagedRunConfig) {
  const actionRepeat = Math.max(config.action.action_repeat, 1);
  return `${formatInteger(Math.ceil(config.environment.max_episode_steps / actionRepeat))} policy decisions at repeat x${actionRepeat}.`;
}

export function noProgressSummary(config: ManagedRunConfig) {
  const stallLimit = config.environment.progress_frontier_stall_limit_frames;
  if (stallLimit === null) {
    return "No-progress truncation is disabled.";
  }
  return `${formatInteger(stallLimit)} internal frames ≈ ${formatDurationSeconds(stallLimit / 60)} before truncation.`;
}

export function environmentSummaryRows(config: ManagedRunConfig) {
  const actionRepeat = Math.max(config.action.action_repeat, 1);
  const stallLimit = config.environment.progress_frontier_stall_limit_frames;
  return [
    {
      label: "Renderer",
      detail: "Requested Mupen video backend for training and watch bootstraps.",
      value: config.environment.renderer,
    },
    {
      label: "Camera",
      detail: "Camera mode synchronized when an episode resets.",
      value: formatCameraSetting(config.environment.camera_setting),
    },
    {
      label: "Episode frame cap",
      detail: "Counted per emulated frame, not per policy step.",
      value: formatInteger(config.environment.max_episode_steps),
    },
    {
      label: "Policy decision cap",
      detail: `ceil(${formatInteger(config.environment.max_episode_steps)} / repeat ${actionRepeat})`,
      value: formatInteger(Math.ceil(config.environment.max_episode_steps / actionRepeat)),
    },
    {
      label: "No-progress timeout",
      detail: stallLimit === null ? "Disabled." : `${formatInteger(stallLimit)} / 60 native fps`,
      value: stallLimit === null ? "Off" : formatDurationSeconds(stallLimit / 60),
    },
    {
      label: "Frontier reset rule",
      detail: "Timer resets once max race distance beats the prior frontier by ε.",
      value: `ε = ${formatDecimal(config.environment.progress_frontier_epsilon)}`,
    },
  ];
}

function formatCameraSetting(value: string) {
  return value.replaceAll("_", " ");
}
