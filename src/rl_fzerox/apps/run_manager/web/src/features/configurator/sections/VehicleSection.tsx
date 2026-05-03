import { useMemo } from "react";

import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentCollapsedIds } from "@/features/configurator/disclosureState";
import { ToggleSwitch } from "@/features/configurator/fields";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";

import { EngineSettingControl } from "./vehicle/EngineSettingControl";
import {
  engineSettingSummary,
  orderedVehicles,
  selectionSummary,
  vehicleRows,
  vehicleSlotLabel,
} from "./vehicle/model";

interface VehicleSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  setConfig: (config: ManagedRunConfig) => void;
}

export function VehicleSection({
  config,
  defaultConfig,
  metadata,
  setConfig,
}: VehicleSectionProps) {
  const randomEngineDefaults = { min: 20, max: 80 } as const;
  const rows = useMemo(() => vehicleRows(metadata), [metadata]);
  const allVehicles = useMemo(() => orderedVehicles(metadata), [metadata]);
  const allVehicleIds = useMemo(() => allVehicles.map((vehicle) => vehicle.id), [allVehicles]);
  const [collapsedRowIds, setCollapsedRowIds] = usePersistentCollapsedIds(
    "run-manager:vehicle:rows",
    rows.map((row) => row.id),
  );
  const selectedVehicleIds = config.vehicle.selected_vehicle_ids;
  const defaultVehicleIds = defaultConfig.vehicle.selected_vehicle_ids;
  const selectedVehicleSet = useMemo(() => new Set(selectedVehicleIds), [selectedVehicleIds]);
  const collapsedRowIdSet = useMemo(() => new Set(collapsedRowIds), [collapsedRowIds]);
  const selectedVehicles = allVehicles.filter((vehicle) => selectedVehicleSet.has(vehicle.id));
  const selectedRowCount = rows.filter((row) =>
    row.vehicles.some((vehicle) => selectedVehicleSet.has(vehicle.id)),
  ).length;
  const engineTicks = [
    { value: 0, label: "0" },
    { value: 50, label: "50" },
    { value: 100, label: "100" },
  ] as const;

  const updateVehicle = (patch: Partial<ManagedRunConfig["vehicle"]>) => {
    setConfig({ ...config, vehicle: { ...config.vehicle, ...patch } });
  };

  const setSelectedVehicles = (nextIds: Iterable<string>) => {
    const nextSet = new Set(nextIds);
    const ordered = allVehicleIds.filter((vehicleId) => nextSet.has(vehicleId));
    if (ordered.length === 0) {
      return;
    }
    updateVehicle({
      selection_mode: "pool",
      selected_vehicle_ids: ordered,
    });
  };

  const setEngineMode = (mode: ManagedRunConfig["vehicle"]["engine_mode"]) => {
    if (mode === "random_range" && config.vehicle.engine_mode !== "random_range") {
      updateVehicle({
        engine_mode: "random_range",
        engine_setting_min_raw_value: randomEngineDefaults.min,
        engine_setting_max_raw_value: randomEngineDefaults.max,
      });
      return;
    }
    updateVehicle({ engine_mode: mode });
  };

  const setRowCollapsed = (rowId: string, collapsed: boolean) => {
    setCollapsedRowIds((currentIds) =>
      collapsed
        ? currentIds.includes(rowId)
          ? currentIds
          : [...currentIds, rowId]
        : currentIds.filter((currentId) => currentId !== rowId),
    );
  };

  const toggleVehicle = (vehicleId: string) => {
    const nextSet = new Set(selectedVehicleSet);
    if (nextSet.has(vehicleId)) {
      if (nextSet.size === 1) {
        return;
      }
      nextSet.delete(vehicleId);
    } else {
      nextSet.add(vehicleId);
    }
    setSelectedVehicles(nextSet);
  };

  const toggleRow = (rowVehicleIds: readonly string[], enabled: boolean) => {
    const nextSet = new Set(selectedVehicleSet);
    if (enabled) {
      for (const vehicleId of rowVehicleIds) {
        nextSet.add(vehicleId);
      }
    } else {
      for (const vehicleId of rowVehicleIds) {
        nextSet.delete(vehicleId);
      }
      if (nextSet.size === 0) {
        nextSet.add(rowVehicleIds[0] ?? allVehicleIds[0] ?? "blue_falcon");
      }
    }
    setSelectedVehicles(nextSet);
  };

  return (
    <div className="config-stack">
      <div className="form-grid two vehicle-panel-grid">
        <ConfigPanel
          onReset={() =>
            updateVehicle({
              engine_mode: defaultConfig.vehicle.engine_mode,
              engine_setting_raw_value: defaultConfig.vehicle.engine_setting_raw_value,
              engine_setting_min_raw_value: defaultConfig.vehicle.engine_setting_min_raw_value,
              engine_setting_max_raw_value: defaultConfig.vehicle.engine_setting_max_raw_value,
            })
          }
          title="Engine setting"
        >
          <div className="vehicle-engine-panel">
            <div className="vehicle-engine-mode-row">
              <div className="vehicle-engine-mode-copy">
                <strong>Random range</strong>
                <small>
                  {config.vehicle.engine_mode === "fixed"
                    ? "Keep one global slider value for every selected machine."
                    : "Resample one global slider value inside the selected range on each reset."}
                </small>
              </div>
              <ToggleSwitch
                checked={config.vehicle.engine_mode === "random_range"}
                hideLabel
                label="Random range"
                tooltip={
                  config.vehicle.engine_mode === "random_range"
                    ? "Random range enabled"
                    : "Random range disabled"
                }
                onChange={(checked) => setEngineMode(checked ? "random_range" : "fixed")}
              />
            </div>
            <EngineSettingControl
              defaultFixedValue={defaultConfig.vehicle.engine_setting_raw_value}
              defaultRangeMax={randomEngineDefaults.max}
              defaultRangeMin={randomEngineDefaults.min}
              fixedValue={config.vehicle.engine_setting_raw_value}
              help={
                config.vehicle.engine_mode === "fixed"
                  ? "F-Zero X engine slider. Lower values bias acceleration, higher values bias top speed."
                  : "Sample one engine slider value inside this range on each episode reset."
              }
              label={config.vehicle.engine_mode === "fixed" ? "Engine slider" : "Engine range"}
              max={100}
              min={0}
              mode={config.vehicle.engine_mode}
              rangeMax={config.vehicle.engine_setting_max_raw_value}
              rangeMin={config.vehicle.engine_setting_min_raw_value}
              ticks={engineTicks}
              onFixedChange={(value) => updateVehicle({ engine_setting_raw_value: value })}
              onRangeChange={(value) =>
                updateVehicle({
                  engine_setting_max_raw_value: value.max,
                  engine_setting_min_raw_value: value.min,
                })
              }
            />
          </div>
        </ConfigPanel>

        <ConfigPanel title="Selection summary">
          <div className="vehicle-summary-grid">
            <VehicleMetric label="Selected machines" value={String(selectedVehicleIds.length)} />
            <VehicleMetric label="Rows covered" value={String(selectedRowCount)} />
            <VehicleMetric label="Engine setting" value={engineSettingSummary(config.vehicle)} />
          </div>
          <p className="vehicle-note">{selectionSummary(selectedVehicles)}</p>
        </ConfigPanel>
      </div>

      <ConfigPanel
        onReset={() =>
          updateVehicle({
            selection_mode: "pool",
            selected_vehicle_ids: defaultVehicleIds,
          })
        }
        title="Machine roster"
        wide
      >
        <div className="vehicle-roster-shell">
          <div className="section-toolbar-row">
            <div className="vehicle-roster-actions">
              <button
                className="secondary-button"
                disabled={selectedVehicleIds.length === allVehicleIds.length}
                type="button"
                onClick={() => setSelectedVehicles(allVehicleIds)}
              >
                Select all
              </button>
              <button
                className="secondary-button"
                disabled={arraysEqual(selectedVehicleIds, defaultVehicleIds)}
                type="button"
                onClick={() => setSelectedVehicles(defaultVehicleIds)}
              >
                Restore defaults
              </button>
            </div>
            <DisclosureToolbar
              collapseLabel="Collapse all machine rows"
              expandLabel="Expand all machine rows"
              onCollapseAll={() => setCollapsedRowIds(rows.map((row) => row.id))}
              onExpandAll={() => setCollapsedRowIds([])}
            />
          </div>

          <div className="vehicle-row-stack">
            {rows.map((row) => {
              const rowSelectedCount = row.vehicles.filter((vehicle) =>
                selectedVehicleSet.has(vehicle.id),
              ).length;
              const rowSelectedIds = row.vehicles.map((vehicle) => vehicle.id);
              const rowFullySelected = rowSelectedCount === row.vehicles.length;
              const rowCollapsed = collapsedRowIdSet.has(row.id);
              return (
                <details
                  className="config-disclosure vehicle-row-section"
                  key={row.id}
                  open={!rowCollapsed}
                  onToggle={(event) => setRowCollapsed(row.id, !event.currentTarget.open)}
                >
                  <summary className="config-disclosure-summary vehicle-row-summary">
                    <span className="config-disclosure-title vehicle-row-summary-label">
                      <span className="config-disclosure-copy">
                        <strong>{row.label}</strong>
                        <small>
                          {rowSelectedCount} of {row.vehicles.length} selected
                        </small>
                      </span>
                    </span>
                    <div className="vehicle-row-controls">
                      <ToggleSwitch
                        checked={rowFullySelected}
                        hideLabel
                        label={`${row.label} enabled`}
                        tooltip={rowFullySelected ? "Deselect row" : "Select row"}
                        onChange={(checked) => toggleRow(rowSelectedIds, checked)}
                      />
                    </div>
                  </summary>
                  <div className="config-disclosure-body">
                    <div className="vehicle-card-grid">
                      {row.vehicles.map((vehicle) => {
                        const selected = selectedVehicleSet.has(vehicle.id);
                        const isOnlySelected = selected && selectedVehicleIds.length === 1;
                        return (
                          <button
                            aria-label={vehicle.display_name}
                            aria-disabled={isOnlySelected || undefined}
                            aria-pressed={selected}
                            className={
                              selected
                                ? isOnlySelected
                                  ? "vehicle-card selected blocked"
                                  : "vehicle-card selected"
                                : "vehicle-card"
                            }
                            key={vehicle.id}
                            type="button"
                            onClick={() => {
                              if (!isOnlySelected) {
                                toggleVehicle(vehicle.id);
                              }
                            }}
                          >
                            <span className="vehicle-card-mark">
                              {vehicleSlotLabel(vehicle.machine_select_slot)}
                            </span>
                            <div className="vehicle-card-copy">
                              <strong>{vehicle.display_name}</strong>
                              <span>Machine {vehicle.character_index + 1}</span>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </details>
              );
            })}
          </div>
        </div>
      </ConfigPanel>
    </div>
  );
}

function VehicleMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="vehicle-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function arraysEqual(left: readonly string[], right: readonly string[]) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
