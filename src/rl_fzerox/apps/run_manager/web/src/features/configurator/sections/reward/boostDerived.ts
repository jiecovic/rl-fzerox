// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/reward/boostDerived.ts
import type { RewardCurvePreviewPoint } from "@/features/configurator/sections/reward/RewardCurvePreview";
import type { ManagedRunConfig } from "@/shared/api/contract";

export function boostRequestRewardPreviewPoints(
  config: ManagedRunConfig,
): readonly RewardCurvePreviewPoint[] {
  const sampleCount = 48;
  const fullRewardFraction = clampFraction(
    config.reward.manual_boost_reward_full_energy_fraction,
    1,
  );
  const fractions = [
    ...Array.from({ length: sampleCount + 1 }, (_, index) => index / sampleCount),
    0.25,
    0.5,
    0.75,
    fullRewardFraction,
  ]
    .map(roundPreviewValue)
    .filter((fraction, index, values) => values.indexOf(fraction) === index)
    .sort((left, right) => left - right);
  return fractions.map((energyFraction) => ({
    label: energyTickLabel(energyFraction, fullRewardFraction),
    xValue: roundPreviewValue(energyFraction * 100),
    yValue: roundPreviewValue(boostRequestRewardEnergyMultiplier(config, energyFraction)),
  }));
}

export function boostRequestRewardEnergyMultiplier(
  config: ManagedRunConfig,
  energyFraction: number,
): number {
  if (!config.reward.manual_boost_reward_energy_shaping) {
    return 1;
  }
  const minMultiplier = clampFraction(config.reward.manual_boost_reward_min_energy_multiplier, 0);
  const fullRewardFraction = clampFraction(
    config.reward.manual_boost_reward_full_energy_fraction,
    1,
  );
  const ratio = Math.min(Math.max(energyFraction, 0) / Math.max(fullRewardFraction, 1e-9), 1);
  const shapedRatio =
    config.reward.manual_boost_reward_energy_curve === "smoothstep"
      ? ratio * ratio * (3 - 2 * ratio)
      : ratio;
  return minMultiplier + (1 - minMultiplier) * shapedRatio;
}

function energyTickLabel(energyFraction: number, fullRewardFraction: number) {
  if (
    energyFraction === 0 ||
    energyFraction === 0.25 ||
    energyFraction === 0.5 ||
    energyFraction === 0.75 ||
    energyFraction === 1 ||
    energyFraction === fullRewardFraction
  ) {
    return `${formatPreviewNumber(energyFraction * 100)}%`;
  }
  return undefined;
}

function clampFraction(value: number, fallback: number) {
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : fallback;
}

function roundPreviewValue(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function formatPreviewNumber(value: number) {
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}
