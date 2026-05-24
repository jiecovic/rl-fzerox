// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/action/model.ts
import type { ManagedRunConfig } from "@/shared/api/contract";

type ManagedActionConfig = ManagedRunConfig["action"];

const FZEROX_NATIVE_FPS = 60;
const MIN_STEER_BUCKETS = 3;

export interface ActionSummaryRow {
  label: string;
  value: string;
}

export function normalizedActionConfig(action: ManagedActionConfig): ManagedActionConfig {
  const nextAction = {
    ...action,
    steer_buckets: normalizeOddBucketCount(action.steer_buckets),
    pitch_buckets: normalizeOddBucketCount(action.pitch_buckets),
  };
  if (!nextAction.include_air_brake) {
    nextAction.enable_air_brake = false;
  }
  if (!nextAction.include_boost) {
    nextAction.enable_boost = false;
  }
  if (!nextAction.include_lean) {
    nextAction.enable_lean = false;
  } else if (nextAction.lean_output_mode === "independent_buttons") {
    nextAction.lean_mode = "raw";
  }
  if (!nextAction.include_pitch) {
    nextAction.enable_pitch = false;
  } else if (nextAction.pitch_mode === "continuous") {
    nextAction.enable_pitch = true;
  }
  return nextAction;
}

export function normalizeOddBucketCount(value: number) {
  const rounded = Math.max(MIN_STEER_BUCKETS, Math.round(value));
  return rounded % 2 === 0 ? rounded + 1 : rounded;
}

export function effectiveControlFps(actionRepeat: number) {
  return FZEROX_NATIVE_FPS / actionRepeat;
}

export function displayActionRepeat(actionRepeat: number) {
  return `repeat x${actionRepeat}`;
}

export function formatControlFps(value: number) {
  return Number.isInteger(value) ? `${value.toFixed(1)} Hz` : `${value.toFixed(2)} Hz`;
}

export function actionSummaryRows(action: ManagedActionConfig): readonly ActionSummaryRow[] {
  const head = actionHeadShape(action);
  return [
    {
      label: "Control cadence",
      value: `${displayActionRepeat(action.action_repeat)} · ${formatControlFps(effectiveControlFps(action.action_repeat))}`,
    },
    {
      label: "Steering output",
      value:
        action.steering_mode === "continuous"
          ? "1 continuous lane"
          : `${action.steer_buckets} steer logits`,
    },
    {
      label: "Throttle output",
      value: throttleOutputSummary(action),
    },
    {
      label: "Auxiliary output",
      value: auxiliaryOutputSummary(action),
    },
    {
      label: "Total output",
      value: `${head.continuousDims} continuous · ${head.discreteLogits} logits`,
    },
  ];
}

export function actionCompatibilityNote(action: ManagedActionConfig): string | null {
  if (!action.force_full_throttle) {
    return null;
  }
  return action.drive_mode === "pwm"
    ? "Runtime throttle is clamped to full while the continuous throttle lane stays in the policy output."
    : "Runtime throttle is forced engaged while the discrete throttle branch stays in the policy output.";
}

function actionHeadShape(action: ManagedActionConfig): {
  continuousDims: number;
  discreteLogits: number;
} {
  let continuousDims = action.steering_mode === "continuous" ? 1 : 0;
  let discreteLogits = action.steering_mode === "discrete" ? action.steer_buckets : 0;

  if (action.drive_mode === "pwm") {
    continuousDims += 1;
  } else {
    discreteLogits += 2;
  }
  if (action.include_air_brake) {
    if (action.air_brake_mode === "pwm") {
      continuousDims += 1;
    } else {
      discreteLogits += 2;
    }
  }
  if (action.include_boost) {
    discreteLogits += 2;
  }
  if (action.include_lean) {
    discreteLogits += action.lean_output_mode === "three_way" ? 3 : 4;
  }
  if (action.include_pitch) {
    if (action.pitch_mode === "continuous") {
      continuousDims += 1;
    } else {
      discreteLogits += action.pitch_buckets;
    }
  }
  return { continuousDims, discreteLogits };
}

function auxiliaryOutputSummary(action: ManagedActionConfig): string {
  const labels: string[] = [];
  if (action.include_air_brake) {
    if (!action.enable_air_brake) {
      labels.push(action.air_brake_mode === "pwm" ? "air brake pwm masked" : "air brake masked");
    } else if (action.air_brake_mode === "pwm") {
      labels.push(action.mask_air_brake_on_ground ? "air brake pwm, air-only" : "air brake pwm");
    } else if (action.mask_air_brake_on_ground) {
      labels.push("air brake, air-only");
    } else {
      labels.push("air brake");
    }
  }
  if (action.include_boost) {
    labels.push(boostOutputSummary(action));
  }
  if (action.include_lean) {
    labels.push(leanOutputSummary(action));
  }
  if (action.include_pitch) {
    labels.push(pitchOutputSummary(action));
  }
  return labels.length > 0 ? labels.join(" · ") : "No auxiliary branches";
}

function throttleOutputSummary(action: ManagedActionConfig): string {
  const base = action.drive_mode === "pwm" ? "1 continuous pwm lane" : "2 button logits";
  return action.force_full_throttle ? `${base}, runtime forced full` : base;
}

function boostOutputSummary(action: ManagedActionConfig): string {
  if (!action.enable_boost) {
    return "boost masked";
  }
  const guards: string[] = [];
  if (action.boost_unmask_max_speed_kph !== null) {
    guards.push(`≤ ${action.boost_unmask_max_speed_kph} kph`);
  }
  if (action.boost_min_energy_fraction > 0) {
    guards.push(`≥ ${Math.round(action.boost_min_energy_fraction * 100)}% energy`);
  }
  return guards.length > 0 ? `boost, ${guards.join(", ")}` : "boost";
}

function leanOutputSummary(action: ManagedActionConfig): string {
  const base =
    action.lean_output_mode === "independent_buttons"
      ? "lean buttons"
      : action.lean_output_mode === "four_way_categorical"
        ? "4-way lean"
        : "lean";
  if (!action.enable_lean) {
    return `${base} masked`;
  }
  if (action.lean_output_mode === "independent_buttons") {
    return action.lean_unmask_min_speed_kph === null
      ? `${base}, fully free`
      : `${base}, above ${action.lean_unmask_min_speed_kph} kph`;
  }
  if (action.lean_output_mode === "four_way_categorical") {
    return action.lean_mode === "raw" ? `${base}, raw` : `${base}, ${action.lean_mode}`;
  }
  return action.lean_mode === "raw" ? `${base}, raw` : `${base}, ${action.lean_mode}`;
}

function pitchOutputSummary(action: ManagedActionConfig): string {
  if (action.pitch_mode === "continuous") {
    return "pitch continuous";
  }
  if (!action.enable_pitch) {
    return "pitch masked";
  }
  return `${action.pitch_buckets} pitch logits`;
}
