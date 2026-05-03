import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRunConfig,
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
    created_at: "2026-05-01T16:11:28+00:00",
    updated_at: "2026-05-01T16:11:28+00:00",
    config: managedRunConfigFixture,
    ...overrides,
  };
}
