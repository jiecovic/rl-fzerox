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
      : `engine ${config.vehicle.engine_setting_min_raw_value}-${config.vehicle.engine_setting_max_raw_value}`;
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
        value={`${config.observation.preset} · ${config.observation.stack_mode} x${config.observation.frame_stack}${config.observation.minimap_layer ? " · minimap" : ""}`}
      />
      <SummaryItem
        label="Policy"
        value={`${config.policy.conv_profile} · ${config.policy.recurrent_enabled ? `LSTM ${config.policy.recurrent_hidden_size}` : "no LSTM"} · fusion ${config.policy.fusion_features_dim}`}
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

function displayAuxiliarySummary(config: ManagedRunConfig) {
  const labels: string[] = [];
  if (config.action.include_air_brake) {
    if (!config.action.enable_air_brake) {
      labels.push(
        config.action.air_brake_mode === "pwm" ? "air brake pwm masked" : "air brake masked",
      );
    } else if (config.action.air_brake_mode === "pwm") {
      labels.push(
        config.action.mask_air_brake_on_ground ? "air brake pwm, air-only" : "air brake pwm",
      );
    } else {
      labels.push(config.action.mask_air_brake_on_ground ? "air brake, air-only" : "air brake");
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
      ].filter((value): value is string => value !== null);
      labels.push(boostGuards.length === 0 ? "boost" : `boost, ${boostGuards.join(", ")}`);
    }
  }
  if (config.action.include_lean) {
    const leanSummary =
      config.action.lean_output_mode === "independent_buttons" ? "lean buttons" : "lean";
    if (!config.action.enable_lean) {
      labels.push(`${leanSummary} masked`);
    } else if (config.action.lean_output_mode === "independent_buttons") {
      labels.push(
        config.action.lean_unmask_min_speed_kph === null
          ? `${leanSummary}, fully free`
          : `${leanSummary}, ≥ ${config.action.lean_unmask_min_speed_kph} kph`,
      );
    } else {
      labels.push(
        config.action.lean_mode === "raw"
          ? `${leanSummary}, raw`
          : `${leanSummary}, ${config.action.lean_mode}`,
      );
    }
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
