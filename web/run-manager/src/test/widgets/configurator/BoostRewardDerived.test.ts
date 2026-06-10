// web/run-manager/src/test/widgets/configurator/BoostRewardDerived.test.ts
import { describe, expect, it } from "vitest";
import {
  boostRequestRewardPreviewPoints,
  boostRequestRewardValue,
} from "@/entities/runConfig/ui/sections/reward/boostDerived";
import type { ManagedRunConfig } from "@/shared/api/contract";
import { managedRunConfigFixture } from "@/test/fixtures";

function configWithBoostShaping(
  rewardOverrides: Partial<ManagedRunConfig["reward"]>,
): ManagedRunConfig {
  return {
    ...managedRunConfigFixture,
    reward: {
      ...managedRunConfigFixture.reward,
      ...rewardOverrides,
    },
  };
}

describe("boost reward derived values", () => {
  it("keeps boost request reward unscaled when energy shaping is off", () => {
    const config = configWithBoostShaping({
      manual_boost_reward_energy_shaping: false,
      manual_boost_reward_min_energy_value: -3,
    });

    expect(boostRequestRewardValue(config, 0.1)).toBe(config.reward.manual_boost_reward);
  });

  it("interpolates boost request reward by current energy when shaping is on", () => {
    const config = configWithBoostShaping({
      manual_boost_reward: 8,
      manual_boost_reward_energy_curve: "linear",
      manual_boost_reward_energy_shaping: true,
      manual_boost_reward_full_energy_fraction: 1,
      manual_boost_reward_min_energy_fraction: 0,
      manual_boost_reward_min_energy_value: 2,
    });

    expect(boostRequestRewardValue(config, 0.5)).toBeCloseTo(5);
  });

  it("holds boost request reward at the minimum below the low-energy threshold", () => {
    const config = configWithBoostShaping({
      manual_boost_reward: 8,
      manual_boost_reward_energy_curve: "linear",
      manual_boost_reward_energy_shaping: true,
      manual_boost_reward_full_energy_fraction: 1,
      manual_boost_reward_min_energy_fraction: 0.5,
      manual_boost_reward_min_energy_value: 2,
    });

    expect(boostRequestRewardValue(config, 0.25)).toBeCloseTo(2);
    expect(boostRequestRewardValue(config, 0.75)).toBeCloseTo(5);
  });

  it("allows the low-energy reward to be negative", () => {
    const config = configWithBoostShaping({
      manual_boost_reward: 8,
      manual_boost_reward_energy_curve: "linear",
      manual_boost_reward_energy_shaping: true,
      manual_boost_reward_full_energy_fraction: 1,
      manual_boost_reward_min_energy_fraction: 0.5,
      manual_boost_reward_min_energy_value: -4,
    });

    expect(boostRequestRewardValue(config, 0.25)).toBeCloseTo(-4);
    expect(boostRequestRewardValue(config, 0.75)).toBeCloseTo(2);
  });

  it("labels the full-reward energy point in the preview curve", () => {
    const config = configWithBoostShaping({
      manual_boost_reward_energy_shaping: true,
      manual_boost_reward_full_energy_fraction: 0.8,
    });

    expect(boostRequestRewardPreviewPoints(config)).toContainEqual(
      expect.objectContaining({
        label: "80%",
        xValue: 80,
      }),
    );
  });
});
