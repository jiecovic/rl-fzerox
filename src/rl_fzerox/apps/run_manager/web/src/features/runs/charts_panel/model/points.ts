import type { RunPlotPoint } from "@/features/runs/charts/RunPlotCard";
import type { ManagedRun, ManagedRunMetricSample } from "@/shared/api/contract";

const LEGACY_SAMPLE_FIELD_BY_METRIC_KEY: Partial<Record<string, keyof ManagedRunMetricSample>> = {
  "rollout/ep_len_mean": "episode_length_mean",
  "rollout/ep_rew_mean": "episode_reward_mean",
  "train/approx_kl": "approx_kl",
  "train/entropy_loss": "entropy_loss",
  "train/policy_gradient_loss": "policy_gradient_loss",
  "train/value_loss": "value_loss",
};

export function metricPoints(samples: ManagedRunMetricSample[], key: string) {
  return samples
    .map((sample) => {
      const value = metricValueFromSample(sample, key);
      return value === undefined ? null : { step: chartStep(sample), value };
    })
    .filter((point): point is RunPlotPoint => point !== null);
}

export function buildEnvStepRatePoints(run: ManagedRun, samples: ManagedRunMetricSample[]) {
  return samples
    .map((sample, index) => {
      const value = sampleEnvStepRateValue(run, samples, index);
      return value === null ? null : { step: chartStep(sample), value };
    })
    .filter((point): point is RunPlotPoint => point !== null);
}

function metricValueFromSample(sample: ManagedRunMetricSample, key: string) {
  const metricValue = sample.metrics[key];
  if (metricValue !== undefined) {
    return metricValue;
  }
  const legacyField = LEGACY_SAMPLE_FIELD_BY_METRIC_KEY[key];
  if (legacyField === undefined) {
    return undefined;
  }
  const legacyValue = sample[legacyField];
  return typeof legacyValue === "number" ? legacyValue : undefined;
}

function sampleEnvStepRateValue(run: ManagedRun, samples: ManagedRunMetricSample[], index: number) {
  const sample = samples[index];
  if (sample === undefined) {
    return null;
  }
  const loggedFps = sample.metrics["time/fps"];
  if (loggedFps !== undefined) {
    return loggedFps;
  }
  if (sample.fps !== null && sample.fps !== undefined) {
    return sample.fps;
  }
  const currentMs = Date.parse(sample.created_at);
  if (Number.isNaN(currentMs)) {
    return null;
  }
  const previous = index > 0 ? samples[index - 1] : null;
  if (previous !== null) {
    const previousMs = Date.parse(previous.created_at);
    const elapsedSeconds = (currentMs - previousMs) / 1000;
    if (!Number.isNaN(previousMs) && elapsedSeconds > 0) {
      return Math.max(0, sample.num_timesteps - previous.num_timesteps) / elapsedSeconds;
    }
  }
  const startedAt = run.started_at ?? run.created_at;
  const startedMs = Date.parse(startedAt);
  const elapsedSeconds = (currentMs - startedMs) / 1000;
  if (Number.isNaN(startedMs) || elapsedSeconds <= 0) {
    return null;
  }
  return sample.num_timesteps / elapsedSeconds;
}

function chartStep(sample: ManagedRunMetricSample) {
  return sample.lineage_num_timesteps;
}
