// src/rl_fzerox/apps/run_manager/web/src/features/runs/charts_panel/model/catalog.ts

import { buildEnvStepRatePoints, metricPoints } from "@/features/runs/charts_panel/model/points";
import type { RunChartDescriptor, RunChartGroup } from "@/features/runs/charts_panel/model/types";
import type { RunMetricRangeMode } from "@/shared/api/client";
import type { ManagedRun, ManagedRunMetricSample } from "@/shared/api/contract";

export const RUN_CHART_GROUPS = [
  { id: "progress", title: "Progress" },
  { id: "rollout", title: "Rollout" },
  { id: "timing", title: "Timing" },
  { id: "optimization", title: "PPO" },
  { id: "reward", title: "Reward" },
  { id: "action", title: "Actions" },
  { id: "state", title: "State" },
  { id: "episode", title: "Episodes" },
  { id: "curriculum", title: "Curriculum" },
  { id: "sampling", title: "Sampling" },
  { id: "other", title: "Other" },
] as const;

export type RunChartGroupId = (typeof RUN_CHART_GROUPS)[number]["id"];

export const INITIAL_GROUP_OPEN: Record<RunChartGroupId, boolean> = {
  action: true,
  curriculum: false,
  episode: true,
  optimization: true,
  other: false,
  progress: true,
  reward: true,
  rollout: true,
  sampling: false,
  state: true,
  timing: true,
};

export const CHART_RANGE_OPTIONS: readonly { label: string; value: RunMetricRangeMode }[] = [
  { label: "Recent", value: "recent" },
  { label: "From start", value: "full" },
];

export const DEFAULT_CHART_RANGE_MODE: RunMetricRangeMode = "full";

const EXPLICIT_CHARTS = [
  {
    id: "episode_reward_mean",
    emptyText: "Waiting for rollout reward samples.",
    group: "rollout",
    metricKeys: ["rollout/ep_rew_mean"],
    title: "Episode reward",
    buildPoints: (_run: ManagedRun, samples: ManagedRunMetricSample[]) =>
      metricPoints(samples, "rollout/ep_rew_mean"),
  },
  {
    id: "episode_length_mean",
    emptyText: "Waiting for rollout length samples.",
    group: "rollout",
    metricKeys: ["rollout/ep_len_mean"],
    title: "Episode length",
    buildPoints: (_run: ManagedRun, samples: ManagedRunMetricSample[]) =>
      metricPoints(samples, "rollout/ep_len_mean"),
  },
  {
    id: "env_step_rate",
    emptyText: "Waiting for timing samples.",
    group: "timing",
    metricKeys: ["time/fps"],
    title: "Env step / s",
    buildPoints: buildEnvStepRatePoints,
  },
] as const satisfies readonly RunChartDescriptor[];

const METRIC_TITLE_OVERRIDES: Record<string, string> = {
  "curriculum/batch_size": "Batch size",
  "curriculum/clip_range": "Clip range",
  "curriculum/ent_coef": "Entropy coefficient",
  "curriculum/learning_rate": "Learning rate",
  "curriculum/n_epochs": "Epochs",
  "episode/airborne_episode_rate": "Airborne episode rate",
  "episode/airborne_failure_rate": "Airborne failure rate",
  "episode/airborne_finish_rate": "Airborne finish rate",
  "episode/boost_pad_entries_mean": "Boost-pad entries",
  "episode/boost_pad_entries_per_lap_mean": "Boost-pad entries / lap",
  "episode/crashed_rate": "Crash rate",
  "episode/energy_depleted_rate": "Energy depleted rate",
  "episode/falling_off_track_rate": "Fell off track rate",
  "episode/final_position_mean": "Final position",
  "episode/finish_position_mean": "Finish position",
  "episode/finish_steps_mean": "Finish steps",
  "episode/finish_time_s_mean": "Finish time (s)",
  "episode/finished_rate": "Finished rate",
  "episode/progress_stalled_rate": "Progress stalled rate",
  "episode/race_laps_completed_mean": "Laps completed",
  "episode/retired_rate": "Retired rate",
  "episode/spinning_out_rate": "Spin-out rate",
  "episode/timeout_rate": "Timeout rate",
  "reward/step_raw_mean": "Raw step reward",
  "reward_clip/abs_excess_mean": "Clip excess",
  "reward_clip/any_step_rate": "Any clip rate",
  "reward_clip/negative_step_rate": "Negative clip rate",
  "reward_clip/positive_step_rate": "Positive clip rate",
  "rollout/ep_len_mean": "Episode length",
  "rollout/ep_rew_mean": "Episode reward",
  "action/air_brake_used_step_rate": "Air-brake use rate",
  "action/gas_level_mean": "Gas level",
  "action/gas_used_step_rate": "Gas use rate",
  "action/spin_macro_frame_ratio": "Spin macro frame ratio",
  "action/spin_requested_step_rate": "Spin request rate",
  "action/spin_started_step_rate": "Spin start rate",
  "state/airborne_frame_ratio": "Airborne frame ratio",
  "state/boost_pad_entry_step_rate": "Boost-pad entry rate",
  "state/damage_taken_step_rate": "Damage-taken rate",
  "state/lap_mean": "Lap",
  "state/position_mean": "Position",
  "state/race_distance_mean": "Race distance",
  "state/race_laps_completed_mean": "Laps completed",
  "state/speed_kph_mean": "Speed (kph)",
  "time/fps": "Env step / s",
  "time/iterations": "Iterations",
  "time/time_elapsed": "Elapsed time",
  "time/total_timesteps": "Total steps",
  "track_sampling/coverage": "Track coverage",
  "train/approx_kl": "Approx KL",
  "train/aux_loss": "Auxiliary loss",
  "train/clip_fraction": "Clip fraction",
  "train/clip_range": "Clip range",
  "train/entropy_loss": "Entropy loss",
  "train/explained_variance": "Explained variance",
  "train/learning_rate": "Learning rate",
  "train/loss": "Loss",
  "train/n_updates": "Updates",
  "train/policy_gradient_loss": "Policy gradient loss",
  "train/std": "Policy std",
  "train/value_loss": "Value loss",
};

export function buildChartGroups(
  runs: ManagedRun[],
  metricsByRun: Record<string, ManagedRunMetricSample[]>,
): RunChartGroup[] {
  const charts = [...EXPLICIT_CHARTS, ...buildMetricCharts(runs, metricsByRun)];
  return RUN_CHART_GROUPS.map((group) => ({
    id: group.id,
    title: group.title,
    charts: charts.filter((chart) => chart.group === group.id),
  })).filter((group) => group.charts.length > 0);
}

function buildMetricCharts(
  runs: ManagedRun[],
  metricsByRun: Record<string, ManagedRunMetricSample[]>,
): RunChartDescriptor[] {
  const explicitMetricKeys = new Set<string>(
    EXPLICIT_CHARTS.flatMap((chart) => chart.metricKeys ?? []),
  );
  const metricKeys = new Set<string>();
  for (const run of runs) {
    const samples = metricsByRun[run.id] ?? [];
    for (const sample of samples) {
      for (const key of Object.keys(sample.metrics)) {
        if (key !== "time/total_timesteps" && !explicitMetricKeys.has(key)) {
          metricKeys.add(key);
        }
      }
    }
  }
  return [...metricKeys].sort().map((key) => ({
    id: `metric:${key}`,
    emptyText: "Waiting for sampled metrics.",
    group: chartGroupForMetricKey(key),
    title: metricTitle(key),
    buildPoints: (_run: ManagedRun, samples: ManagedRunMetricSample[]) =>
      metricPoints(samples, key),
  }));
}

function chartGroupForMetricKey(key: string): RunChartGroupId {
  if (key.startsWith("rollout/")) {
    return "rollout";
  }
  if (key.startsWith("time/")) {
    return "timing";
  }
  if (key.startsWith("train/")) {
    return "optimization";
  }
  if (key.startsWith("train_aux/")) {
    return "optimization";
  }
  if (key.startsWith("reward/") || key.startsWith("reward_clip/")) {
    return "reward";
  }
  if (key.startsWith("action/")) {
    return "action";
  }
  if (key.startsWith("state/")) {
    return "state";
  }
  if (key.startsWith("episode/")) {
    return "episode";
  }
  if (key.startsWith("curriculum/")) {
    return "curriculum";
  }
  if (key.startsWith("track_sampling/")) {
    return "sampling";
  }
  return "other";
}

function metricTitle(key: string) {
  if (key.startsWith("train_aux/")) {
    return `Aux: ${humanizeAuxiliaryMetricKey(key.slice("train_aux/".length))}`;
  }
  const override = METRIC_TITLE_OVERRIDES[key];
  if (override !== undefined) {
    return override;
  }
  const leaf = key.includes("/") ? key.slice(key.lastIndexOf("/") + 1) : key;
  return humanizeMetricToken(leaf);
}

function humanizeAuxiliaryMetricKey(token: string) {
  const leaf = token.includes(".") ? token.slice(token.lastIndexOf(".") + 1) : token;
  return humanizeMetricToken(leaf);
}

function humanizeMetricToken(token: string) {
  return token
    .replaceAll("_", " ")
    .replaceAll(".", " ")
    .replaceAll(" kph", " (kph)")
    .replace(/\bep\b/giu, "episode")
    .replace(/\bkl\b/giu, "KL")
    .replace(/\bfps\b/giu, "FPS")
    .replace(/\bs\b/giu, "s")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}
