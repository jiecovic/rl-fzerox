// web/run-manager/src/test/fixtures.ts
import type {
  CheckpointCatalogResponse,
  ConfigMetadata,
  ManagedDraft,
  ManagedRunConfig,
  ManagedRunDetail,
  ManagedRunMetricSample,
  ManagedSaveGame,
  PolicyArchitecturePreview,
} from "@/shared/api/contract";
import {
  configMetadataSchema,
  managedRunConfigSchema,
  policyArchitecturePreviewSchema,
} from "@/shared/api/contract";
import generatedFixtures from "@/test/generated/manager-fixtures.json";

type GeneratedFixtures = {
  managed_run_config: ManagedRunConfig;
  config_metadata: ConfigMetadata;
  policy_preview: PolicyArchitecturePreview;
};

const generated: GeneratedFixtures = {
  config_metadata: configMetadataSchema.parse(generatedFixtures.config_metadata),
  managed_run_config: managedRunConfigSchema.parse(generatedFixtures.managed_run_config),
  policy_preview: policyArchitecturePreviewSchema.parse(generatedFixtures.policy_preview),
};

export const managedRunConfigFixture: ManagedRunConfig = generated.managed_run_config;
export const configMetadataFixture: ConfigMetadata = generated.config_metadata;
export const policyPreviewFixture: PolicyArchitecturePreview = generated.policy_preview;

export function checkpointCatalogFixture(
  overrides: Partial<CheckpointCatalogResponse> = {},
): CheckpointCatalogResponse {
  return {
    catalog: {
      format_name: "rl-fzerox-checkpoint-catalog",
      schema_version: 1,
      updated_at: "2026-06-26T12:15:44+00:00",
    },
    entries: [
      {
        id: "blue-falcon-fine-tuned",
        version: "v1",
        name: "72 x 96 IMPALA - like Blue Falcon Fine Tuned",
        bundle: {
          filename: "rl-fzerox-checkpoint-blue-falcon-fine-tuned-v1.zip",
          sha256: "a".repeat(64),
          size_bytes: 33_569_307,
          url: "https://github.com/jiecovic/rl-fzerox/releases/download/checkpoints-v1/bundle.zip",
        },
        installed_checkpoint_id: null,
        manifest: {
          checkpoint: {
            id: "blue-falcon-fine-tuned",
            name: "72 x 96 IMPALA - like Blue Falcon Fine Tuned",
            version: "v1",
            source_artifact: "best",
            local_num_timesteps: 68_288_256,
            lineage_num_timesteps: 1_979_774_040,
          },
          compatibility: {
            training_algorithm: "maskable_hybrid_recurrent_ppo",
          },
          exported_at: "2026-06-26T12:15:44+00:00",
          files: [],
          format_name: "rl-fzerox-checkpoint-bundle",
          schema_version: 1,
        },
      },
    ],
    installed_checkpoints: [],
    ...overrides,
  };
}

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

export function runFixture(overrides: Partial<ManagedRunDetail> = {}): ManagedRunDetail {
  const config = overrides.config ?? managedRunConfigFixture;
  return {
    id: "run-001",
    name: "ppo_test_1",
    status: "running",
    config_hash: "test-config-hash",
    action_repeat: config.action.action_repeat,
    vehicle_setup: {
      selection_mode: config.vehicle.selection_mode,
      selected_vehicle_ids: config.vehicle.selected_vehicle_ids,
      engine_mode: config.vehicle.engine_mode,
      engine_setting_raw_value: config.vehicle.engine_setting_raw_value,
      engine_setting_min_raw_value: config.vehicle.engine_setting_min_raw_value,
      engine_setting_max_raw_value: config.vehicle.engine_setting_max_raw_value,
    },
    created_at: "2026-05-03T18:52:02+00:00",
    lineage_id: "run-001",
    lineage_groups: [],
    lineage_step_offset: 0,
    started_at: "2026-05-03T18:52:10+00:00",
    stopped_at: null,
    parent_run_id: null,
    source_run_id: null,
    source_artifact: null,
    source_num_timesteps: null,
    active_alt_baseline_count: 0,
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
    config,
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

export function saveGameFixture(overrides: Partial<ManagedSaveGame> = {}): ManagedSaveGame {
  return {
    id: "save-001",
    name: "unlock save",
    status: "created",
    runner_active: false,
    save_path: "/tmp/save-001/fzerox.srm",
    created_at: "2026-06-02T10:30:00+00:00",
    updated_at: "2026-06-02T10:30:00+00:00",
    last_finished_at: null,
    runner_settings: {
      attempt_seed: 123,
      device: "cuda",
      policy_mode: "deterministic",
      recording_enabled: false,
      recording_input_hud_enabled: false,
      recording_upscale_factor: 2,
      recording_path: null,
      target_restart_on_retire: false,
      target_clear_goal: 1,
      keep_failed_recordings: false,
      reload_policy_between_attempts: true,
      renderer: "gliden64",
    },
    unlock_progress: {
      inspection_status: "not_inspected",
      completed_count: 0,
      total_count: 2,
      unlocked_vehicle_count: 6,
      unlocked_vehicle_ids: [
        "blue_falcon",
        "golden_fox",
        "wild_goose",
        "fire_stingray",
        "white_cat",
        "red_gazelle",
      ],
      next_target: {
        sequence_index: 0,
        kind: "clear_gp_cup",
        status: "pending",
        label: "Clear Novice Jack Cup",
        difficulty: "novice",
        cup_id: "jack",
        course_id: null,
      },
      targets: [
        {
          sequence_index: 0,
          kind: "clear_gp_cup",
          status: "pending",
          label: "Clear Novice Jack Cup",
          difficulty: "novice",
          cup_id: "jack",
          course_id: null,
        },
        {
          sequence_index: 1,
          kind: "clear_gp_cup",
          status: "locked",
          label: "Clear Novice Queen Cup",
          difficulty: "novice",
          cup_id: "queen",
          course_id: null,
        },
      ],
    },
    attempts: [],
    course_setups: [],
    cup_setups: [],
    ...overrides,
  };
}
