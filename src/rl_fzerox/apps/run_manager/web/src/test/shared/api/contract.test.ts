// src/rl_fzerox/apps/run_manager/web/src/test/shared/api/contract.test.ts
import { describe, expect, it } from "vitest";

import {
  configMetadataSchema,
  managedRunConfigSchema,
  policyArchitecturePreviewSchema,
} from "@/shared/api/contract";
import generatedFixtures from "@/test/generated/manager-fixtures.json";

describe("manager API contract fixtures", () => {
  it("parses backend-generated configurator payloads", () => {
    expect(() => configMetadataSchema.parse(generatedFixtures.config_metadata)).not.toThrow();
    expect(() => managedRunConfigSchema.parse(generatedFixtures.managed_run_config)).not.toThrow();
    expect(() =>
      policyArchitecturePreviewSchema.parse(generatedFixtures.policy_preview),
    ).not.toThrow();
  });
});
