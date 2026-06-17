// web/run-manager/src/entities/runConfig/ui/sections/vehicle/model.ts
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";
import { engineSliderStepPercentLabel } from "@/shared/domain/engineBuckets";

export interface VehicleRow {
  id: string;
  label: string;
  vehicles: ConfigMetadata["vehicles"];
}

export function orderedVehicles(metadata: ConfigMetadata) {
  return [...metadata.vehicles].sort(
    (left, right) => left.machine_select_slot - right.machine_select_slot,
  );
}

export function vehicleRows(metadata: ConfigMetadata): VehicleRow[] {
  const rows = new Map<number, ConfigMetadata["vehicles"]>();
  for (const vehicle of orderedVehicles(metadata)) {
    const existing = rows.get(vehicle.menu_row) ?? [];
    rows.set(vehicle.menu_row, [...existing, vehicle]);
  }
  return [...rows.entries()]
    .sort(([left], [right]) => left - right)
    .map(([rowIndex, vehicles]) => ({
      id: `vehicle-row-${rowIndex + 1}`,
      label: `Row ${rowIndex + 1}`,
      vehicles: [...vehicles].sort((left, right) => left.menu_column - right.menu_column),
    }));
}

export function selectionSummary(selectedVehicles: readonly ConfigMetadata["vehicles"][number][]) {
  if (selectedVehicles.length === 0) {
    return "No machines selected";
  }
  if (selectedVehicles.length <= 3) {
    return selectedVehicles.map((vehicle) => vehicle.display_name).join(", ");
  }
  const preview = selectedVehicles
    .slice(0, 3)
    .map((vehicle) => vehicle.display_name)
    .join(", ");
  return `${preview} +${selectedVehicles.length - 3} more`;
}

export function engineSettingSummary(config: ManagedRunConfig["vehicle"]) {
  if (config.engine_mode === "fixed") {
    return engineSettingStepSummary(config.engine_setting_raw_value);
  }
  const range = `${engineSettingStepSummary(
    config.engine_setting_min_raw_value,
  )}-${engineSettingStepSummary(config.engine_setting_max_raw_value)}`;
  if (config.engine_mode !== "adaptive_tuner") {
    return range;
  }
  const backend =
    config.adaptive_engine_tuner_backend === "mlp_ensemble"
      ? "MLP exp"
      : config.adaptive_engine_tuner_backend === "gaussian_process"
        ? "GP exp"
        : "bandit";
  return `adaptive ${backend} ${range}`;
}

function engineSettingStepSummary(step: number) {
  return engineSliderStepPercentLabel(step);
}

export function vehicleSlotLabel(machineSelectSlot: number) {
  return `slot ${String(machineSelectSlot + 1).padStart(2, "0")}`;
}
