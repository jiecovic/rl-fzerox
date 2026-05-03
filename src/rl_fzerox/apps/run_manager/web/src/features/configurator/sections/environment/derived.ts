import type { ManagedRunConfig } from "@/shared/api/contract";

export function episodeFrameSummary(config: ManagedRunConfig) {
  return `${formatInteger(config.environment.max_episode_steps)} internal frames ≈ ${formatDuration(config.environment.max_episode_steps / 60)} of game time`;
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
  return `${formatInteger(stallLimit)} internal frames ≈ ${formatDuration(stallLimit / 60)} before truncation.`;
}

export function environmentSummaryRows(config: ManagedRunConfig) {
  const actionRepeat = Math.max(config.action.action_repeat, 1);
  const stallLimit = config.environment.progress_frontier_stall_limit_frames;
  return [
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
      value: stallLimit === null ? "Off" : formatDuration(stallLimit / 60),
    },
    {
      label: "Frontier reset rule",
      detail: "Timer resets once max race distance beats the prior frontier by ε.",
      value: `ε = ${formatDecimal(config.environment.progress_frontier_epsilon)}`,
    },
  ];
}

function formatInteger(value: number) {
  return value.toLocaleString();
}

function formatDecimal(value: number) {
  if (Number.isInteger(value)) {
    return formatInteger(value);
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatDuration(value: number) {
  if (value < 60) {
    return `${value.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })} s`;
  }
  const wholeMinutes = Math.floor(value / 60);
  const remainingSeconds = value - wholeMinutes * 60;
  if (Math.abs(remainingSeconds - Math.round(remainingSeconds)) < 1e-9) {
    return `${wholeMinutes}m ${Math.round(remainingSeconds)}s`;
  }
  return `${wholeMinutes}m ${remainingSeconds.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}s`;
}
