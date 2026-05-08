// src/rl_fzerox/apps/run_manager/web/src/test/fixtures.ts
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRun,
  ManagedRunConfig,
  ManagedRunMetricSample,
  PolicyArchitecturePreview,
} from "@/shared/api/contract";
import generatedFixtures from "@/test/generated/manager-fixtures.json";

type GeneratedFixtures = {
  managed_run_config: ManagedRunConfig;
  config_metadata: ConfigMetadata;
  policy_preview: PolicyArchitecturePreview;
};

const generated = generatedFixtures as GeneratedFixtures;

export const managedRunConfigFixture: ManagedRunConfig = generated.managed_run_config;
export const configMetadataFixture: ConfigMetadata = generated.config_metadata;
export const policyPreviewFixture: PolicyArchitecturePreview = generated.policy_preview;

export function draftFixture(overrides: Partial<ManagedDraft> = {}): ManagedDraft {
  return {
    id: "draft-001",
    name: "ppo_allcups_recurrent",
    source_run_id: null,
    source_artifact: null,
    source_num_timesteps: null,
    created_at: "2026-05-01T16:11:28+00:00",
    updated_at: "2026-05-01T16:11:28+00:00",
    config: managedRunConfigFixture,
    ...overrides,
  };
}

export function runFixture(overrides: Partial<ManagedRun> = {}): ManagedRun {
  return {
    id: "run-001",
    name: "ppo_test_1",
    status: "running",
    created_at: "2026-05-03T18:52:02+00:00",
    lineage_id: "run-001",
    lineage_step_offset: 0,
    started_at: "2026-05-03T18:52:10+00:00",
    stopped_at: null,
    parent_run_id: null,
    source_run_id: null,
    source_artifact: null,
    source_num_timesteps: null,
    pending_command: null,
    worker_heartbeat_at: "2026-05-03T18:55:02+00:00",
    recent_events: [],
    runtime: {
      total_timesteps: 50_000_000,
      num_timesteps: 1_250_000,
      progress_fraction: 0.025,
      updated_at: "2026-05-03T18:55:00+00:00",
      fps: 912,
      episode_reward_mean: 4.2,
      episode_length_mean: 481,
      approx_kl: 0.013,
      entropy_loss: -1.1,
      value_loss: 0.27,
      policy_gradient_loss: -0.03,
    },
    config: managedRunConfigFixture,
    ...overrides,
  };
}

export function runMetricSampleFixture(
  overrides: Partial<ManagedRunMetricSample> = {},
): ManagedRunMetricSample {
  return {
    run_id: "run-001",
    created_at: "2026-05-03T18:55:00+00:00",
    total_timesteps: 50_000_000,
    num_timesteps: 1_250_000,
    lineage_num_timesteps: 1_250_000,
    progress_fraction: 0.025,
    metrics: {
      "rollout/ep_rew_mean": 4.2,
      "rollout/ep_len_mean": 481,
      "time/fps": 912,
      "train/approx_kl": 0.013,
      "train/entropy_loss": -1.1,
      "train/value_loss": 0.27,
      "train/policy_gradient_loss": -0.03,
      "train/clip_fraction": 0.14,
      "train/explained_variance": 0.68,
      "reward/step_raw_mean": 0.11,
      "state/speed_kph_mean": 842,
      "episode/finished_rate": 0.37,
    },
    fps: 912,
    episode_reward_mean: 4.2,
    episode_length_mean: 481,
    approx_kl: 0.013,
    entropy_loss: -1.1,
    value_loss: 0.27,
    policy_gradient_loss: -0.03,
    ...overrides,
  };
}
