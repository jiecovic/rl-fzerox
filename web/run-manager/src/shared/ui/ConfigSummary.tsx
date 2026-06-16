// web/run-manager/src/shared/ui/ConfigSummary.tsx
import type { ManagedRunConfig } from "@/shared/api/contract";

export function ConfigSummary({ config }: { config: ManagedRunConfig }) {
  const selectedVehicles = config.vehicle.selected_vehicle_ids;
  const vehicleSummary =
    config.vehicle.selection_mode === "fixed" && selectedVehicles.length === 1
      ? displayVehicleName(selectedVehicles[0] ?? "vehicle")
      : `${selectedVehicles.length} vehicles`;
  const engineSummary =
    config.vehicle.engine_mode === "fixed"
      ? `engine ${config.vehicle.engine_setting_raw_value}`
      : config.vehicle.engine_mode === "random_range"
        ? `engine ${config.vehicle.engine_setting_min_raw_value}-${config.vehicle.engine_setting_max_raw_value}`
        : `adaptive engine ${config.vehicle.engine_setting_min_raw_value}-${config.vehicle.engine_setting_max_raw_value}`;
  const actionSummary = `${displayActionRepeatSummary(config)} · ${displaySteeringSummary(config)} · ${displayDriveSummary(config)}${
    config.action.include_air_brake ||
    config.action.include_boost ||
    config.action.include_lean ||
    config.action.include_pitch
      ? ` · ${displayAuxiliarySummary(config)}`
      : ""
  }`;
  return (
    <div className="summary-grid">
      <SummaryItem
        label="Training"
        value={`${config.train.num_envs} envs · ${config.train.total_timesteps.toLocaleString()} steps · lr ${config.train.learning_rate.toExponential(2)}`}
      />
      <SummaryItem
        label="Observation"
        value={`${displayObservationResolution(config)} · ${config.observation.stack_mode} x${config.observation.frame_stack}${config.observation.minimap_layer ? " · minimap" : ""}`}
      />
      <SummaryItem
        label="Policy"
        value={`${config.policy.conv_profile} · ${config.policy.recurrent_enabled ? `LSTM ${config.policy.recurrent_hidden_size}` : "no LSTM"} · ${
          config.policy.fusion_features_dim === null
            ? "fusion off"
            : `fusion ${config.policy.fusion_features_dim}`
        }`}
      />
      <SummaryItem label="Vehicle" value={`${vehicleSummary} · ${engineSummary}`} />
      <SummaryItem label="Action" value={actionSummary} />
      <SummaryItem label="Environment" value={displayEnvironmentSummary(config)} />
    </div>
  );
}

function SummaryItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="summary-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function displayVehicleName(vehicleId: string) {
  return vehicleId
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function displaySteeringSummary(config: ManagedRunConfig) {
  return config.action.steering_mode === "continuous"
    ? "continuous steer"
    : `${config.action.steer_buckets}-bucket steer`;
}

function displayActionRepeatSummary(config: ManagedRunConfig) {
  return `repeat x${config.action.action_repeat}`;
}

function displayDriveSummary(config: ManagedRunConfig) {
  const base =
    config.action.drive_mode === "pwm" ? "continuous pwm throttle" : "discrete button throttle";
  return config.action.force_full_throttle ? `${base}, forced full` : base;
}

function displayEnvironmentSummary(config: ManagedRunConfig) {
  const stallLimit = config.environment.progress_frontier_stall_limit_frames;
  const stallSummary =
    stallLimit === null
      ? "stall off"
      : `stall ${stallLimit.toLocaleString()}f @ ε${config.environment.progress_frontier_epsilon.toLocaleString()}`;
  return `${config.environment.max_episode_steps.toLocaleString()} frames · ${stallSummary}`;
}

function displayObservationResolution(config: ManagedRunConfig) {
  const { resolution } = config.observation;
  if (resolution.mode === "custom") {
    return `${resolution.height} x ${resolution.width}`;
  }
  if (resolution.mode === "source_crop") {
    return "original crop";
  }
  return resolution.preset;
}

function displayAuxiliarySummary(config: ManagedRunConfig) {
  const labels: string[] = [];
  if (config.action.include_air_brake) {
    const episodeMask = episodeMaskSummary(config.action.air_brake_episode_mask_probability ?? 0);
    if (!config.action.enable_air_brake) {
      labels.push(
        config.action.air_brake_mode === "pwm" ? "air brake pwm masked" : "air brake masked",
      );
    } else if (config.action.air_brake_mode === "pwm") {
      labels.push(
        config.action.mask_air_brake_on_ground ? "air brake pwm, air-only" : "air brake pwm",
      );
    } else {
      labels.push(
        config.action.mask_air_brake_on_ground
          ? `air brake, air-only${episodeMask}`
          : `air brake${episodeMask}`,
      );
    }
  }
  if (config.action.include_boost) {
    if (!config.action.enable_boost) {
      labels.push("boost masked");
    } else {
      const boostGuards = [
        config.action.boost_unmask_max_speed_kph === null
          ? null
          : `≤ ${config.action.boost_unmask_max_speed_kph} kph`,
        config.action.boost_min_energy_fraction > 0
          ? `≥ ${Math.round(config.action.boost_min_energy_fraction * 100)}% energy`
          : null,
        config.action.mask_boost_when_active ? "idle only" : null,
        config.action.mask_boost_when_airborne ? "grounded only" : null,
        config.action.boost_decision_interval_steps > 1
          ? `every ${config.action.boost_decision_interval_steps} env steps`
          : null,
        config.action.boost_request_lockout_frames > 0
          ? `${config.action.boost_request_lockout_frames}f cooldown`
          : null,
      ].filter((value): value is string => value !== null);
      labels.push(boostGuards.length === 0 ? "boost" : `boost, ${boostGuards.join(", ")}`);
    }
  }
  if (config.action.include_lean) {
    const leanSummary =
      config.action.lean_output_mode === "independent_buttons"
        ? "lean buttons"
        : config.action.lean_output_mode === "four_way_categorical"
          ? "4-way lean"
          : "lean";
    const leanEpisodeMaskProbability = config.action.lean_episode_mask_probability ?? 0;
    const leanEpisodeMask =
      leanEpisodeMaskProbability > 0
        ? `, ${(leanEpisodeMaskProbability * 100).toFixed(0)}% episode mask`
        : "";
    if (!config.action.enable_lean) {
      labels.push(`${leanSummary} masked`);
    } else if (config.action.lean_output_mode === "independent_buttons") {
      labels.push(
        config.action.lean_unmask_min_speed_kph === null
          ? `${leanSummary}, fully free${leanEpisodeMask}`
          : `${leanSummary}, ≥ ${config.action.lean_unmask_min_speed_kph} kph${leanEpisodeMask}`,
      );
    } else {
      labels.push(
        config.action.lean_mode === "raw"
          ? `${leanSummary}, raw${leanEpisodeMask}`
          : `${leanSummary}, ${config.action.lean_mode}${leanEpisodeMask}`,
      );
    }
  }
  if (config.action.include_spin) {
    const spinSummary = config.action.enable_spin ? "spin macro" : "spin macro masked";
    const episodeMask = episodeMaskSummary(config.action.spin_episode_mask_probability ?? 0);
    labels.push(
      config.action.spin_cooldown_frames > 0
        ? `${spinSummary}, ${config.action.spin_cooldown_frames}f cooldown${episodeMask}`
        : `${spinSummary}${episodeMask}`,
    );
  }
  if (config.action.include_pitch) {
    if (config.action.pitch_mode === "continuous") {
      labels.push("pitch continuous");
    } else {
      labels.push(
        config.action.enable_pitch ? `${config.action.pitch_buckets} pitch logits` : "pitch masked",
      );
    }
  }
  return labels.join(" · ");
}

function episodeMaskSummary(probability: number) {
  return probability > 0 ? `, ${(probability * 100).toFixed(0)}% episode mask` : "";
}
