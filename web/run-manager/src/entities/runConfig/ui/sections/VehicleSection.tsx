// web/run-manager/src/entities/runConfig/ui/sections/VehicleSection.tsx
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
import {
  bucketSideCountFromRawValues,
  centeredEngineBuckets,
  ENGINE_SLIDER,
  enginePercentToSliderStep,
  engineSliderStepPercentLabel,
} from "@/shared/domain/engineBuckets";
import { Button } from "@/shared/ui/Button";
import { cn } from "@/shared/ui/cn";
import { ConfigGrid, ConfigStack } from "@/shared/ui/config/ConfigLayout";
import { ConfigPanel } from "@/shared/ui/config/ConfigPanel";
import { DisclosureToolbar } from "@/shared/ui/config/DisclosureToolbar";
import { usePersistentCollapsedIds } from "@/shared/ui/config/disclosureState";
import {
  NumberField,
  RangeNumberField,
  SegmentedChoiceStrip,
  ToggleSwitch,
} from "@/shared/ui/configFields";

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
  const randomEngineDefaults = {
    max: enginePercentToSliderStep(80),
    min: enginePercentToSliderStep(20),
  } as const;
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
    { value: 0, label: "0%" },
    { value: enginePercentToSliderStep(50), label: "50%" },
    { value: ENGINE_SLIDER.maxStep, label: "100%" },
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
      (currentConfig): Partial<ManagedRunConfig["vehicle"]> => {
        const bucketSideCount =
          bucketSideCountFromRawValues(
            currentConfig.vehicle.adaptive_engine_bandit_bucket_raw_values,
          ) ||
          bucketSideCountFromRawValues(
            defaultConfig.vehicle.adaptive_engine_bandit_bucket_raw_values,
          );
        if (mode === "fixed") {
          return { engine_mode: mode };
        }
        if (mode === "adaptive_tuner") {
          return {
            adaptive_engine_bandit_bucket_raw_values: centeredEngineBuckets({
              sideCount: bucketSideCount,
              minimum: ENGINE_SLIDER.minStep,
              maximum: ENGINE_SLIDER.maxStep,
            }),
            engine_mode: mode,
            engine_setting_min_raw_value: ENGINE_SLIDER.minStep,
            engine_setting_max_raw_value: ENGINE_SLIDER.maxStep,
          };
        }
        if (currentConfig.vehicle.engine_mode !== "fixed") {
          return { engine_mode: mode };
        }
        return {
          engine_mode: mode,
          engine_setting_min_raw_value: randomEngineDefaults.min,
          engine_setting_max_raw_value: randomEngineDefaults.max,
        };
      },
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
              adaptive_engine_ensemble_members:
                defaultConfig.vehicle.adaptive_engine_ensemble_members,
              adaptive_engine_mlp_hidden_dim: defaultConfig.vehicle.adaptive_engine_mlp_hidden_dim,
              adaptive_engine_mlp_training_steps:
                defaultConfig.vehicle.adaptive_engine_mlp_training_steps,
              adaptive_engine_mlp_learning_rate:
                defaultConfig.vehicle.adaptive_engine_mlp_learning_rate,
              adaptive_engine_mlp_bootstrap_keep_probability:
                defaultConfig.vehicle.adaptive_engine_mlp_bootstrap_keep_probability,
              adaptive_engine_mlp_warmup_successes:
                defaultConfig.vehicle.adaptive_engine_mlp_warmup_successes,
              adaptive_engine_stat_decay: defaultConfig.vehicle.adaptive_engine_stat_decay,
              adaptive_engine_tuner_backend: defaultConfig.vehicle.adaptive_engine_tuner_backend,
              adaptive_engine_tuner_objective:
                defaultConfig.vehicle.adaptive_engine_tuner_objective,
              adaptive_engine_uniform_exploration:
                defaultConfig.vehicle.adaptive_engine_uniform_exploration,
              adaptive_engine_greedy_plateau_seconds:
                defaultConfig.vehicle.adaptive_engine_greedy_plateau_seconds,
              adaptive_engine_bandit_bucket_raw_values:
                defaultConfig.vehicle.adaptive_engine_bandit_bucket_raw_values,
              engine_mode: defaultConfig.vehicle.engine_mode,
              engine_setting_raw_value: defaultConfig.vehicle.engine_setting_raw_value,
              engine_setting_min_raw_value: defaultConfig.vehicle.engine_setting_min_raw_value,
              engine_setting_max_raw_value: defaultConfig.vehicle.engine_setting_max_raw_value,
            })
          }
          title="Engine setting"
        >
          <div className="grid gap-3">
            <div className="grid min-h-[34px] gap-2">
              <div className="grid gap-1">
                <strong className="text-[13px] text-app-text">Engine mode</strong>
                <small className="m-0 text-xs leading-snug text-app-muted">
                  {engineModeDescription(config.vehicle.engine_mode)}
                </small>
              </div>
              <SegmentedChoiceStrip
                ariaLabel="Engine mode"
                options={[
                  {
                    active: config.vehicle.engine_mode === "fixed",
                    key: "fixed",
                    label: "Fixed",
                    onClick: () => setEngineMode("fixed"),
                  },
                  {
                    active: config.vehicle.engine_mode === "random_range",
                    key: "random_range",
                    label: "Random",
                    onClick: () => setEngineMode("random_range"),
                  },
                  {
                    active: config.vehicle.engine_mode === "adaptive_tuner",
                    key: "adaptive_tuner",
                    label: "Adaptive",
                    onClick: () => setEngineMode("adaptive_tuner"),
                  },
                ]}
              />
            </div>
            <EngineSettingControl
              defaultFixedValue={defaultConfig.vehicle.engine_setting_raw_value}
              defaultRangeMax={randomEngineDefaults.max}
              defaultRangeMin={randomEngineDefaults.min}
              fixedValue={config.vehicle.engine_setting_raw_value}
              help={
                config.vehicle.engine_mode === "fixed"
                  ? "F-Zero X engine slider step. Labels show the rounded in-game ENG percent."
                  : config.vehicle.engine_mode === "random_range"
                    ? "Sample one representable engine slider step inside this range on each episode reset."
                    : "Candidate engine slider-step range used by the adaptive tuner at episode reset."
              }
              label={config.vehicle.engine_mode === "fixed" ? "Engine slider" : "Engine range"}
              max={ENGINE_SLIDER.maxStep}
              min={ENGINE_SLIDER.minStep}
              mode={config.vehicle.engine_mode}
              rangeMax={config.vehicle.engine_setting_max_raw_value}
              rangeMin={config.vehicle.engine_setting_min_raw_value}
              ticks={engineTicks}
              onFixedChange={(value) => updateVehicle({ engine_setting_raw_value: value })}
              onRangeChange={(value) => {
                const bucketSideCount =
                  bucketSideCountFromRawValues(
                    config.vehicle.adaptive_engine_bandit_bucket_raw_values,
                  ) ||
                  bucketSideCountFromRawValues(
                    defaultConfig.vehicle.adaptive_engine_bandit_bucket_raw_values,
                  );
                const adaptive_engine_bandit_bucket_raw_values = centeredEngineBuckets({
                  sideCount: bucketSideCount,
                  minimum: value.min,
                  maximum: value.max,
                });
                updateVehicle({
                  ...(adaptive_engine_bandit_bucket_raw_values.length > 0
                    ? { adaptive_engine_bandit_bucket_raw_values }
                    : {}),
                  engine_setting_max_raw_value: value.max,
                  engine_setting_min_raw_value: value.min,
                });
              }}
            />
            {config.vehicle.engine_mode === "adaptive_tuner" ? (
              <AdaptiveEngineControls
                config={config}
                defaultConfig={defaultConfig}
                onChange={updateVehicle}
              />
            ) : null}
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

function AdaptiveEngineControls({
  config,
  defaultConfig,
  onChange,
}: {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  onChange: (patch: Partial<ManagedRunConfig["vehicle"]>) => void;
}) {
  const vehicle = config.vehicle;
  const defaultVehicle = defaultConfig.vehicle;
  const isBanditBackend = vehicle.adaptive_engine_tuner_backend === "bandit";
  const isGaussianProcessBackend = vehicle.adaptive_engine_tuner_backend === "gaussian_process";
  const isMlpEnsembleBackend = vehicle.adaptive_engine_tuner_backend === "mlp_ensemble";
  const banditBucketSideCount =
    bucketSideCountFromRawValues(vehicle.adaptive_engine_bandit_bucket_raw_values) ||
    bucketSideCountFromRawValues(defaultVehicle.adaptive_engine_bandit_bucket_raw_values);
  const banditBuckets = useMemo(
    () => vehicle.adaptive_engine_bandit_bucket_raw_values,
    [vehicle.adaptive_engine_bandit_bucket_raw_values],
  );

  function setBanditBucketSideCount(sideCount: number) {
    const adaptive_engine_bandit_bucket_raw_values = centeredEngineBuckets({
      sideCount,
      minimum: vehicle.engine_setting_min_raw_value,
      maximum: vehicle.engine_setting_max_raw_value,
    });
    if (adaptive_engine_bandit_bucket_raw_values.length > 0) {
      onChange({ adaptive_engine_bandit_bucket_raw_values });
    }
  }

  return (
    <div className="grid gap-3 border-t border-app-border pt-3">
      <div className="grid gap-1">
        <strong className="text-[13px] text-app-text">Adaptive engine tuner</strong>
        <small className="m-0 text-xs leading-snug text-app-muted">
          Learns reset-time engine settings from the selected score while preserving uniform
          exploration.
        </small>
      </div>
      <div className="grid gap-2">
        <strong className="text-[13px] text-app-text">Tuner backend</strong>
        <SegmentedChoiceStrip
          ariaLabel="Adaptive engine tuner backend"
          options={[
            {
              active: vehicle.adaptive_engine_tuner_backend === "bandit",
              key: "bandit",
              label: "Bandit",
              onClick: () => onChange({ adaptive_engine_tuner_backend: "bandit" }),
            },
            {
              active: vehicle.adaptive_engine_tuner_backend === "gaussian_process",
              key: "gaussian_process",
              label: "GP (exp)",
              onClick: () => onChange({ adaptive_engine_tuner_backend: "gaussian_process" }),
            },
            {
              active: vehicle.adaptive_engine_tuner_backend === "mlp_ensemble",
              key: "mlp_ensemble",
              label: "MLP (exp)",
              onClick: () => onChange({ adaptive_engine_tuner_backend: "mlp_ensemble" }),
            },
          ]}
        />
        <small className="m-0 text-xs leading-snug text-app-muted">
          Bandit samples coarse measured buckets. GP and MLP are experimental model-backed
          alternatives. Backend uncertainty scales are derived from the episode horizon.
        </small>
      </div>
      {isBanditBackend ? (
        <div className="grid gap-2">
          <strong className="text-[13px] text-app-text">Bandit objective</strong>
          <SegmentedChoiceStrip
            ariaLabel="Bandit engine tuner objective"
            options={[
              {
                active: vehicle.adaptive_engine_tuner_objective === "finish_time",
                key: "finish_time",
                label: "Finish time",
                onClick: () => onChange({ adaptive_engine_tuner_objective: "finish_time" }),
              },
              {
                active: vehicle.adaptive_engine_tuner_objective === "episode_return",
                key: "episode_return",
                label: "Episode return",
                onClick: () => onChange({ adaptive_engine_tuner_objective: "episode_return" }),
              },
              {
                active: vehicle.adaptive_engine_tuner_objective === "completion",
                key: "completion",
                label: "Completion",
                onClick: () => onChange({ adaptive_engine_tuner_objective: "completion" }),
              },
              {
                active: vehicle.adaptive_engine_tuner_objective === "finish_rate",
                key: "finish_rate",
                label: "Finish rate",
                onClick: () => onChange({ adaptive_engine_tuner_objective: "finish_rate" }),
              },
            ]}
          />
          <small className="m-0 text-xs leading-snug text-app-muted">
            Finish time uses successful races only. Episode return, completion, and finish rate use
            default-baseline attempts, including failed and retired attempts. Return mode is tied to
            the reward settings.
          </small>
        </div>
      ) : null}
      <div className="grid gap-3 lg:grid-cols-3">
        {isBanditBackend ? (
          <NumberField
            help="Generates explicit centered engine buckets. 5 means five values below 50%, 50%, and five values above 50%."
            label="Buckets per side"
            resetValue={bucketSideCountFromRawValues(
              defaultVehicle.adaptive_engine_bandit_bucket_raw_values,
            )}
            step="1"
            value={banditBucketSideCount}
            onChange={setBanditBucketSideCount}
          />
        ) : null}
        {isGaussianProcessBackend ? (
          <RangeNumberField
            help="GP-only discount factor for old successful-finish aggregates. Higher values remember longer."
            label="Stat decay"
            max={0.999}
            min={0.001}
            numberStep="0.001"
            rangeStep={0.001}
            resetValue={defaultVehicle.adaptive_engine_stat_decay}
            value={vehicle.adaptive_engine_stat_decay}
            onChange={(adaptive_engine_stat_decay) => onChange({ adaptive_engine_stat_decay })}
          />
        ) : null}
        {isMlpEnsembleBackend ? (
          <>
            <NumberField
              help="Number of bootstrapped MLP members used for Thompson-style engine selection. More members improve uncertainty estimates but cost more CPU."
              label="Ensemble members"
              resetValue={defaultVehicle.adaptive_engine_ensemble_members}
              step="1"
              value={vehicle.adaptive_engine_ensemble_members}
              onChange={(adaptive_engine_ensemble_members) =>
                onChange({ adaptive_engine_ensemble_members })
              }
            />
            <NumberField
              help="Hidden width of each MLP ensemble member."
              label="Hidden size"
              resetValue={defaultVehicle.adaptive_engine_mlp_hidden_dim}
              step="1"
              value={vehicle.adaptive_engine_mlp_hidden_dim}
              onChange={(adaptive_engine_mlp_hidden_dim) =>
                onChange({ adaptive_engine_mlp_hidden_dim })
              }
            />
            <NumberField
              help="Adam optimization steps applied to each ensemble member after one PPO rollout."
              label="Training steps"
              resetValue={defaultVehicle.adaptive_engine_mlp_training_steps}
              step="1"
              value={vehicle.adaptive_engine_mlp_training_steps}
              onChange={(adaptive_engine_mlp_training_steps) =>
                onChange({ adaptive_engine_mlp_training_steps })
              }
            />
            <NumberField
              help="Adam learning rate for the MLP ensemble update."
              label="Learning rate"
              resetValue={defaultVehicle.adaptive_engine_mlp_learning_rate}
              step="0.0005"
              value={vehicle.adaptive_engine_mlp_learning_rate}
              onChange={(adaptive_engine_mlp_learning_rate) =>
                onChange({ adaptive_engine_mlp_learning_rate })
              }
            />
            <RangeNumberField
              help="Bootstrap probability that each successful rollout sample trains each member."
              label="Bootstrap keep"
              max={1}
              min={0.01}
              numberStep="0.01"
              rangeStep={0.01}
              resetValue={defaultVehicle.adaptive_engine_mlp_bootstrap_keep_probability}
              value={vehicle.adaptive_engine_mlp_bootstrap_keep_probability}
              onChange={(adaptive_engine_mlp_bootstrap_keep_probability) =>
                onChange({ adaptive_engine_mlp_bootstrap_keep_probability })
              }
            />
            <NumberField
              help="Successful finishes required for one course and vehicle before the MLP sampler leaves uniform cold-start exploration."
              label="Warmup finishes"
              resetValue={defaultVehicle.adaptive_engine_mlp_warmup_successes}
              step="1"
              value={vehicle.adaptive_engine_mlp_warmup_successes}
              onChange={(adaptive_engine_mlp_warmup_successes) =>
                onChange({ adaptive_engine_mlp_warmup_successes })
              }
            />
          </>
        ) : null}
        <RangeNumberField
          help="Probability of taking a uniformly random engine value."
          label="Uniform exploration"
          max={1}
          min={0}
          numberStep="0.01"
          rangeStep={0.01}
          resetValue={defaultVehicle.adaptive_engine_uniform_exploration}
          value={vehicle.adaptive_engine_uniform_exploration}
          onChange={(adaptive_engine_uniform_exploration) =>
            onChange({ adaptive_engine_uniform_exploration })
          }
        />
        {!isBanditBackend ? (
          <NumberField
            help="Deterministic watch and career import treat predicted finish times within this many seconds of best as practically equal, then choose the soft plateau center."
            label="Greedy plateau seconds"
            resetValue={defaultVehicle.adaptive_engine_greedy_plateau_seconds}
            step="0.1"
            value={vehicle.adaptive_engine_greedy_plateau_seconds}
            onChange={(adaptive_engine_greedy_plateau_seconds) =>
              onChange({ adaptive_engine_greedy_plateau_seconds })
            }
          />
        ) : null}
      </div>
      {isBanditBackend ? <BanditBucketPreview buckets={banditBuckets} /> : null}
    </div>
  );
}

function BanditBucketPreview({ buckets }: { buckets: readonly number[] }) {
  const hasBuckets = buckets.length > 0;
  return (
    <div className="grid gap-2 border border-app-border bg-app-surface-muted p-3">
      <div className="flex items-center justify-between gap-3">
        <strong className="text-[13px] text-app-text">Bandit buckets</strong>
        <span className="text-xs tabular-nums text-app-muted">{buckets.length} values</span>
      </div>
      {hasBuckets ? (
        <div className="flex flex-wrap gap-1.5">
          {buckets.map((bucket) => (
            <span
              className="min-w-11 border border-app-border bg-app-surface px-2 py-1 text-center text-xs font-semibold tabular-nums text-app-text"
              key={bucket}
            >
              {engineSliderStepPercentLabel(bucket)}
            </span>
          ))}
        </div>
      ) : (
        <span className="text-xs text-app-danger">
          No centered bucket falls inside the selected engine range.
        </span>
      )}
    </div>
  );
}

function engineModeDescription(mode: ManagedRunConfig["vehicle"]["engine_mode"]) {
  if (mode === "fixed") {
    return "Keep one global slider value for every selected machine.";
  }
  if (mode === "random_range") {
    return "Resample one global slider value inside the selected range on each reset.";
  }
  return "Learn ordered engine values from successful finish times during training.";
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
