// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/reward/boostDerived.ts

import type { RewardCurvePreviewPoint } from "@/entities/runConfig/ui/sections/reward/RewardCurvePreview";
import type { ManagedRunConfig } from "@/shared/api/contract";

export function boostRequestRewardPreviewPoints(
  config: ManagedRunConfig,
): readonly RewardCurvePreviewPoint[] {
  const sampleCount = 48;
  const fullRewardFraction = clampFraction(
    config.reward.manual_boost_reward_full_energy_fraction,
    1,
  );
  const minEnergyFraction = clampFraction(config.reward.manual_boost_reward_min_energy_fraction, 0);
  const fractions = [
    ...Array.from({ length: sampleCount + 1 }, (_, index) => index / sampleCount),
    0.25,
    0.5,
    0.75,
    minEnergyFraction,
    fullRewardFraction,
  ]
    .map(roundPreviewValue)
    .filter((fraction, index, values) => values.indexOf(fraction) === index)
    .sort((left, right) => left - right);
  return fractions.map((energyFraction) => ({
    label: energyTickLabel(energyFraction, { fullRewardFraction, minEnergyFraction }),
    xValue: roundPreviewValue(energyFraction * 100),
    yValue: roundPreviewValue(boostRequestRewardValue(config, energyFraction)),
  }));
}

export function boostRequestRewardValue(config: ManagedRunConfig, energyFraction: number): number {
  if (!config.reward.manual_boost_reward_energy_shaping) {
    return config.reward.manual_boost_reward;
  }
  const lowReward = finiteNumber(config.reward.manual_boost_reward_min_energy_value, 0);
  const fullReward = finiteNumber(config.reward.manual_boost_reward, 0);
  const minEnergyFraction = clampFraction(config.reward.manual_boost_reward_min_energy_fraction, 0);
  const fullRewardFraction = clampFraction(
    config.reward.manual_boost_reward_full_energy_fraction,
    1,
  );
  const ratio =
    energyFraction <= minEnergyFraction
      ? 0
      : Math.min(
          Math.max(energyFraction - minEnergyFraction, 0) /
            Math.max(fullRewardFraction - minEnergyFraction, Number.EPSILON),
          1,
        );
  const shapedRatio =
    config.reward.manual_boost_reward_energy_curve === "smoothstep"
      ? ratio * ratio * (3 - 2 * ratio)
      : ratio;
  return lowReward + (fullReward - lowReward) * shapedRatio;
}

function energyTickLabel(
  energyFraction: number,
  markers: { fullRewardFraction: number; minEnergyFraction: number },
) {
  if (
    energyFraction === 0 ||
    energyFraction === 0.25 ||
    energyFraction === 0.5 ||
    energyFraction === 0.75 ||
    energyFraction === 1 ||
    energyFraction === markers.fullRewardFraction ||
    energyFraction === markers.minEnergyFraction
  ) {
    return `${formatPreviewNumber(energyFraction * 100)}%`;
  }
  return undefined;
}

function clampFraction(value: number, fallback: number) {
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : fallback;
}

function finiteNumber(value: number, fallback: number) {
  return Number.isFinite(value) ? value : fallback;
}

function roundPreviewValue(value: number) {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function formatPreviewNumber(value: number) {
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}
