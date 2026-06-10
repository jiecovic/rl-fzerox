// src/rl_fzerox/apps/run_manager/web/src/test/widgets/configurator/ProgressDerived.test.ts
import { describe, expect, it } from "vitest";

import {
  progressBucketRewardFromDensity,
  progressRewardDensityFromValues,
} from "@/widgets/configurator/sections/reward/progressDerived";

describe("progress reward derived values", () => {
  it("preserves reward density when switching to continuous progress", () => {
    expect(progressRewardDensityFromValues(5, 0.01)).toBe(2);
    expect(progressBucketRewardFromDensity(0, 2)).toBe(2);
    expect(progressRewardDensityFromValues(0, 2)).toBe(2);
  });
});
