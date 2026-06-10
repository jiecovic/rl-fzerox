// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/VehicleSection.tsx
import { useMemo } from "react";
import { type ConfigSetter, patchConfigSection } from "@/entities/runConfig/model/state";
import { EngineSettingControl } from "@/entities/runConfig/ui/sections/vehicle/EngineSettingControl";
import {
  engineSettingSummary,
  orderedVehicles,
  selectionSummary,
  vehicleRows,
  vehicleSlotLabel,
} from "@/entities/runConfig/ui/sections/vehicle/model";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { cn } from "@/shared/ui/cn";
import { ConfigGrid, ConfigStack } from "@/shared/ui/config/ConfigLayout";
import { ConfigPanel } from "@/shared/ui/config/ConfigPanel";
import { DisclosureToolbar } from "@/shared/ui/config/DisclosureToolbar";
import { usePersistentCollapsedIds } from "@/shared/ui/config/disclosureState";
import { ToggleSwitch } from "@/shared/ui/configFields";

interface VehicleSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  setConfig: ConfigSetter;
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
    patchConfigSection(setConfig, "vehicle", patch);
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
    patchConfigSection(
      setConfig,
      "vehicle",
      (currentConfig): Partial<ManagedRunConfig["vehicle"]> =>
        mode === "random_range" && currentConfig.vehicle.engine_mode !== "random_range"
          ? {
              engine_mode: "random_range",
              engine_setting_min_raw_value: randomEngineDefaults.min,
              engine_setting_max_raw_value: randomEngineDefaults.max,
            }
          : { engine_mode: mode },
    );
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
    setConfig((currentConfig) => {
      const nextSet = new Set(currentConfig.vehicle.selected_vehicle_ids);
      if (nextSet.has(vehicleId)) {
        if (nextSet.size === 1) {
          return currentConfig;
        }
        nextSet.delete(vehicleId);
      } else {
        nextSet.add(vehicleId);
      }
      const ordered = allVehicleIds.filter((currentId) => nextSet.has(currentId));
      if (ordered.length === 0) {
        return currentConfig;
      }
      return {
        ...currentConfig,
        vehicle: {
          ...currentConfig.vehicle,
          selection_mode: "pool",
          selected_vehicle_ids: ordered,
        },
      };
    });
  };

  const toggleRow = (rowVehicleIds: readonly string[], enabled: boolean) => {
    setConfig((currentConfig) => {
      const nextSet = new Set(currentConfig.vehicle.selected_vehicle_ids);
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
      const ordered = allVehicleIds.filter((currentId) => nextSet.has(currentId));
      if (ordered.length === 0) {
        return currentConfig;
      }
      return {
        ...currentConfig,
        vehicle: {
          ...currentConfig.vehicle,
          selection_mode: "pool",
          selected_vehicle_ids: ordered,
        },
      };
    });
  };

  return (
    <ConfigStack>
      <ConfigGrid columns="two" className="items-stretch">
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
          <div className="grid gap-3">
            <div className="grid min-h-[34px] grid-cols-[minmax(0,1fr)_auto] items-center gap-3">
              <div className="grid gap-1">
                <strong className="text-[13px] text-app-text">Random range</strong>
                <small className="m-0 text-xs leading-snug text-app-muted">
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
          <div className="grid grid-cols-3 gap-2.5">
            <VehicleMetric label="Selected machines" value={String(selectedVehicleIds.length)} />
            <VehicleMetric label="Rows covered" value={String(selectedRowCount)} />
            <VehicleMetric label="Engine setting" value={engineSettingSummary(config.vehicle)} />
          </div>
          <p className="min-h-[34px] text-xs leading-snug text-app-muted">
            {selectionSummary(selectedVehicles)}
          </p>
        </ConfigPanel>
      </ConfigGrid>

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
        <div className="grid gap-3">
          <div className="section-toolbar-row">
            <div className="flex flex-wrap gap-2">
              <Button
                className="h-9 px-3"
                disabled={selectedVehicleIds.length === allVehicleIds.length}
                type="button"
                onClick={() => setSelectedVehicles(allVehicleIds)}
              >
                Select all
              </Button>
              <Button
                className="h-9 px-3"
                disabled={arraysEqual(selectedVehicleIds, defaultVehicleIds)}
                type="button"
                onClick={() => setSelectedVehicles(defaultVehicleIds)}
              >
                Restore defaults
              </Button>
            </div>
            <DisclosureToolbar
              collapseLabel="Collapse all machine rows"
              expandLabel="Expand all machine rows"
              onCollapseAll={() => setCollapsedRowIds(rows.map((row) => row.id))}
              onExpandAll={() => setCollapsedRowIds([])}
            />
          </div>

          <div className="grid gap-3">
            {rows.map((row) => {
              const rowSelectedCount = row.vehicles.filter((vehicle) =>
                selectedVehicleSet.has(vehicle.id),
              ).length;
              const rowSelectedIds = row.vehicles.map((vehicle) => vehicle.id);
              const rowFullySelected = rowSelectedCount === row.vehicles.length;
              const rowCollapsed = collapsedRowIdSet.has(row.id);
              return (
                <details
                  className="config-disclosure"
                  key={row.id}
                  open={!rowCollapsed}
                  onToggle={(event) => setRowCollapsed(row.id, !event.currentTarget.open)}
                >
                  <summary className="config-disclosure-summary hover:border-app-border-strong">
                    <span className="config-disclosure-title min-w-0">
                      <span className="config-disclosure-copy">
                        <strong>{row.label}</strong>
                        <small>
                          {rowSelectedCount} of {row.vehicles.length} selected
                        </small>
                      </span>
                    </span>
                    <div className="inline-flex items-center gap-2">
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
                    <div className="grid grid-cols-6 gap-3 max-[1080px]:grid-cols-3">
                      {row.vehicles.map((vehicle) => {
                        const selected = selectedVehicleSet.has(vehicle.id);
                        const isOnlySelected = selected && selectedVehicleIds.length === 1;
                        return (
                          <button
                            aria-label={vehicle.display_name}
                            aria-disabled={isOnlySelected || undefined}
                            aria-pressed={selected}
                            className={vehicleCardClass(selected, isOnlySelected)}
                            key={vehicle.id}
                            type="button"
                            onClick={() => {
                              if (!isOnlySelected) {
                                toggleVehicle(vehicle.id);
                              }
                            }}
                          >
                            {selected ? (
                              <span className="pointer-events-none absolute top-2 right-2 border border-app-accent bg-app-accent px-2 py-0.5 text-[10px] font-bold tracking-wide text-app-accent-text uppercase">
                                selected
                              </span>
                            ) : null}
                            <span
                              className={cn(
                                "inline-flex w-fit min-w-[54px] items-center justify-center border border-app-border bg-app-surface px-2 py-1 text-[11px] tabular-nums text-app-muted uppercase",
                                selected
                                  ? "border-app-accent bg-[color-mix(in_srgb,var(--accent)_18%,var(--surface))] text-app-text"
                                  : undefined,
                              )}
                            >
                              {vehicleSlotLabel(vehicle.machine_select_slot)}
                            </span>
                            <div className="grid gap-1">
                              <strong className="text-sm font-bold text-app-text">
                                {vehicle.display_name}
                              </strong>
                              <span className="m-0 text-xs leading-snug text-app-muted">
                                Machine {vehicle.character_index + 1}
                              </span>
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
    </ConfigStack>
  );
}

function VehicleMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid min-h-[72px] gap-1 border border-app-border bg-app-surface p-2.5">
      <span className="text-xs leading-snug text-app-muted">{label}</span>
      <strong className="text-base tabular-nums text-app-text">{value}</strong>
    </div>
  );
}

function arraysEqual(left: readonly string[], right: readonly string[]) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function vehicleCardClass(selected: boolean, blocked: boolean) {
  return cn(
    "vehicle-card relative grid min-h-[104px] gap-3 border border-app-border bg-app-surface-muted p-3 text-left hover:border-app-border-strong",
    selected
      ? "border-app-accent bg-[color-mix(in_srgb,var(--accent)_18%,var(--surface-muted))] shadow-[inset_0_0_0_2px_color-mix(in_srgb,var(--accent)_52%,transparent),0_0_0_1px_color-mix(in_srgb,var(--accent)_36%,transparent)]"
      : undefined,
    blocked ? "cursor-not-allowed" : undefined,
  );
}
