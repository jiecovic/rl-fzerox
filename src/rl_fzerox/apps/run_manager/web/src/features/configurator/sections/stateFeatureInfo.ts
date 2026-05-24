// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/stateFeatureInfo.ts
interface FeatureInfo {
  label: string;
  help: string;
}

const FEATURE_INFO: Record<string, FeatureInfo> = {
  "machine_context.body_stat": {
    label: "Body stat",
    help: "Normalized machine body rating from the selected vehicle.",
  },
  "machine_context.boost_stat": {
    label: "Boost stat",
    help: "Normalized machine boost rating from the selected vehicle.",
  },
  "machine_context.engine": {
    label: "Engine balance",
    help: "Engine setting from acceleration to max-speed bias, normalized to 0..1.",
  },
  "machine_context.grip_stat": {
    label: "Grip stat",
    help: "Normalized machine grip rating from the selected vehicle.",
  },
  "machine_context.weight": {
    label: "Machine weight",
    help: "Machine weight normalized across the stock vehicle range.",
  },
  "surface_state.on_dirt_surface": {
    label: "On dirt",
    help: "Whether the current ground effect is dirt.",
  },
  "surface_state.on_ice_surface": {
    label: "On ice",
    help: "Whether the current ground effect is ice.",
  },
  "surface_state.on_refill_surface": {
    label: "On refill",
    help: "Whether the machine is on an energy-refill surface.",
  },
  "track_position.edge_ratio": {
    label: "Edge ratio",
    help: "Signed distance to track center relative to local track half width.",
  },
  "track_position.height_above_ground_norm": {
    label: "Ground height",
    help: "Height above ground clipped to non-negative values and normalized by the current ground-height normalizer.",
  },
  "track_position.lap_progress": {
    label: "Progress scalar",
    help: "Configured progress scalar, either lap progress or normalized segment index.",
  },
  "track_position.outside_track_bounds": {
    label: "Outside bounds",
    help: "Whether the track-edge estimate says the machine is outside track bounds.",
  },
  "vehicle_state.airborne": {
    label: "Airborne",
    help: "Whether the machine is currently off the ground.",
  },
  "vehicle_state.boost_active": {
    label: "Boost active",
    help: "Whether manual boost is currently active in game state.",
  },
  "vehicle_state.boost_unlocked": {
    label: "Boost available",
    help: "Whether the game currently allows manual boost.",
  },
  "vehicle_state.energy_frac": {
    label: "Energy",
    help: "Current energy divided by maximum energy.",
  },
  "vehicle_state.lateral_velocity_norm": {
    label: "Lateral velocity",
    help: "Local sideways velocity normalized and clipped to -1..1.",
  },
  "vehicle_state.reverse_active": {
    label: "Reverse timer active",
    help: "Whether the in-game reverse timer indicates backward movement.",
  },
  "vehicle_state.sliding_active": {
    label: "Sliding",
    help: "Whether lateral velocity is high enough to count as sliding while grounded.",
  },
  "vehicle_state.speed_norm": {
    label: "Speed",
    help: "Current speed in kph normalized by the configured speed normalizer.",
  },
};

const CONTROL_LABELS: Record<string, string> = {
  air_brake: "Air brake",
  boost: "Boost",
  lean: "Lean",
  lean_left: "Lean left",
  lean_right: "Lean right",
  pitch: "Pitch",
  steer: "Stick X",
  thrust: "Gas",
};

export function stateFeatureInfo(componentName: string, featureName: string): FeatureInfo {
  const configuredInfo = FEATURE_INFO[featureName];
  if (configuredInfo !== undefined) {
    return configuredInfo;
  }
  if (featureName.startsWith("course_context.course_builtin_")) {
    const index = featureName.slice("course_context.course_builtin_".length);
    return {
      label: `Course ${index}`,
      help: "One-hot course identity entry for the built-in F-Zero X course list.",
    };
  }
  if (featureName.startsWith("control_history.prev_")) {
    return controlHistoryInfo(featureName);
  }
  if (componentName === "control_history") {
    return controlHistoryInfo(`control_history.prev_${featureName}`);
  }
  return {
    label: fallbackLabel(componentName, featureName),
    help: "State-vector entry exposed to the policy.",
  };
}

function controlHistoryInfo(featureName: string): FeatureInfo {
  const match = /^control_history\.prev_(.+)_(\d+)$/.exec(featureName);
  if (match === null) {
    return {
      label: fallbackLabel("control_history", featureName),
      help: "Previous control value exposed to the policy.",
    };
  }
  const [, control = "", age = ""] = match;
  const label = CONTROL_LABELS[control] ?? fallbackWords(control);
  return {
    label: `${label} at time t-${age}`,
    help: `Control request from ${age} previous policy step(s) ago.`,
  };
}

function fallbackLabel(componentName: string, featureName: string) {
  const prefix = `${componentName}.`;
  const rawName = featureName.startsWith(prefix) ? featureName.slice(prefix.length) : featureName;
  return fallbackWords(rawName);
}

function fallbackWords(value: string) {
  return value
    .split("_")
    .filter(Boolean)
    .map((part) => `${part[0]?.toUpperCase() ?? ""}${part.slice(1)}`)
    .join(" ");
}
