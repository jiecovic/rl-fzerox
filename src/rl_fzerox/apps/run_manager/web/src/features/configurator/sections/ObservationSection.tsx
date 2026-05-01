import { useState } from "react";

import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import {
  BooleanField,
  DiscreteSliderNumberField,
  IntegerField,
  SelectField,
} from "@/features/configurator/fields";
import type {
  ConfigMetadata,
  ManagedRunConfig,
  PolicyArchitecturePreview,
  StateComponentConfig,
  StateFeatureConfig,
} from "@/shared/api/contract";
import { CollapseAllIcon, ExpandAllIcon } from "@/shared/ui/DisclosureIcons";

import { stateFeatureInfo } from "./stateFeatureInfo";

interface ObservationSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  preview: PolicyArchitecturePreview | null;
  setConfig: (config: ManagedRunConfig) => void;
}

export function ObservationSection({
  config,
  defaultConfig,
  metadata,
  preview,
  setConfig,
}: ObservationSectionProps) {
  const updateObservation = (patch: Partial<ManagedRunConfig["observation"]>) => {
    setConfig({ ...config, observation: { ...config.observation, ...patch } });
  };
  const stackModeOptions = metadata.stack_modes.map(
    (option) => option.value,
  ) as ManagedRunConfig["observation"]["stack_mode"][];
  const resizeFilterOptions = metadata.resize_filters.map(
    (option) => option.value,
  ) as ManagedRunConfig["observation"]["resize_filter"][];
  const selectedPreset = metadata.observation_presets.find(
    (preset) => preset.value === config.observation.preset,
  );

  return (
    <div className="config-stack">
      <div className="form-grid two">
        <ConfigPanel title="Image observation">
          <div className="config-field-grid image-observation-grid">
            <SelectField
              help="Rendered crop preset used for the policy image input."
              label="Input resolution"
              options={metadata.observation_presets.map((preset) => preset.value)}
              resetValue={defaultConfig.observation.preset}
              value={config.observation.preset}
              onChange={(value) => updateObservation({ preset: value })}
            />
            <SelectField
              help="Image channel encoding used by the frame stack."
              label="Color mode"
              options={stackModeOptions}
              resetValue={defaultConfig.observation.stack_mode}
              value={config.observation.stack_mode}
              onChange={(value) => updateObservation({ stack_mode: value })}
            />
            <DiscreteSliderNumberField
              help="Number of recent image observations exposed to the policy."
              label="Frame stack"
              maxManual={8}
              minManual={1}
              resetValue={defaultConfig.observation.frame_stack}
              sliderValues={[1, 2, 3, 4, 5, 6, 7, 8]}
              value={config.observation.frame_stack}
              onChange={(value) => updateObservation({ frame_stack: value })}
            />
            <BooleanField
              help="Append a one-channel minimap image to the frame stack."
              label="Minimap layer"
              resetValue={defaultConfig.observation.minimap_layer}
              value={config.observation.minimap_layer}
              onChange={(value) => updateObservation({ minimap_layer: value })}
            />
            <SelectField
              help="Resize filter used for the main image crop."
              label="Resize filter"
              options={resizeFilterOptions}
              resetValue={defaultConfig.observation.resize_filter}
              value={config.observation.resize_filter}
              onChange={(value) => updateObservation({ resize_filter: value })}
            />
            <SelectField
              help="Resize filter used for the minimap layer."
              label="Minimap filter"
              options={resizeFilterOptions}
              resetValue={defaultConfig.observation.minimap_resize_filter}
              value={config.observation.minimap_resize_filter}
              onChange={(value) => updateObservation({ minimap_resize_filter: value })}
            />
          </div>
        </ConfigPanel>

        <ConfigPanel title="Derived shape">
          <div className="shape-summary-grid">
            <ShapeMetric
              label="Image"
              value={
                preview !== null
                  ? `${preview.image_shape.height} x ${preview.image_shape.width} x ${preview.image_shape.channels}`
                  : selectedPreset !== undefined
                    ? `${selectedPreset.height} x ${selectedPreset.width}`
                    : "pending"
              }
            />
            <ShapeMetric
              label="State width"
              value={preview !== null ? String(preview.state_dim) : "pending"}
            />
            <ShapeMetric label="Stack" value={`${config.observation.frame_stack} frames`} />
            <ShapeMetric
              label="Zeroed entries"
              value={String(
                preview?.state_features.filter((feature) => feature.mode === "zero").length ?? 0,
              )}
            />
          </div>
        </ConfigPanel>
      </div>

      <ConfigPanel title="State vector components" wide>
        <StateComponentPanels
          config={config}
          metadata={metadata}
          updateObservation={updateObservation}
        />
      </ConfigPanel>
    </div>
  );
}

function StateComponentPanels({
  config,
  metadata,
  updateObservation,
}: {
  config: ManagedRunConfig;
  metadata: ConfigMetadata;
  updateObservation: (patch: Partial<ManagedRunConfig["observation"]>) => void;
}) {
  const [openSections, setOpenSections] = useState(() => allStateComponentsOpen(metadata, true));

  const setSectionOpen = (name: string, open: boolean) => {
    setOpenSections((current) => ({ ...current, [name]: open }));
  };

  function updateComponent(name: string, patch: Partial<StateComponentConfig>) {
    updateObservation({
      state_components: config.observation.state_components.map((component) =>
        component.name === name ? { ...component, ...patch } : component,
      ),
    });
  }

  function setComponentEnabled(componentName: string, enabled: boolean) {
    const componentInfo = metadata.state_components.find((item) => item.name === componentName);
    const nextPatch =
      componentName === "course_context" && enabled
        ? { mode: "include" as const, encoding: "one_hot_builtin" as const }
        : { mode: enabled ? ("include" as const) : ("exclude" as const) };
    const componentFeatureNames = new Set(
      componentInfo?.features.map((feature) => feature.name) ?? [],
    );
    updateObservation({
      state_components: config.observation.state_components.map((component) =>
        component.name === componentName ? { ...component, ...nextPatch } : component,
      ),
      state_feature_modes: config.observation.state_feature_modes.filter(
        (feature) => !componentFeatureNames.has(feature.name),
      ),
    });
  }

  function setFeaturesZeroed(
    component: StateComponentConfig,
    featureNames: readonly string[],
    zeroed: boolean,
  ) {
    const featureNameSet = new Set(featureNames);
    const retainedModes = config.observation.state_feature_modes.filter(
      (feature) => !featureNameSet.has(feature.name),
    );
    const nextMode: StateFeatureConfig["mode"] = zeroed ? "zero" : "include";
    const nextEntries =
      nextMode === component.mode
        ? []
        : featureNames.map((featureName) => ({ name: featureName, mode: nextMode }));
    const nextModes =
      nextEntries.length === 0
        ? retainedModes
        : [...retainedModes, ...nextEntries].sort((left, right) =>
            left.name.localeCompare(right.name),
          );
    updateObservation({ state_feature_modes: nextModes });
  }

  return (
    <div className="state-vector-editor">
      <div className="reward-accordion-toolbar">
        <button
          aria-label="Expand all state-vector categories"
          className="icon-button compact-icon-button tooltip-anchor"
          data-tooltip="Expand all"
          type="button"
          onClick={() => setOpenSections(allStateComponentsOpen(metadata, true))}
        >
          <ExpandAllIcon />
        </button>
        <button
          aria-label="Collapse all state-vector categories"
          className="icon-button compact-icon-button tooltip-anchor"
          data-tooltip="Collapse all"
          type="button"
          onClick={() => setOpenSections(allStateComponentsOpen(metadata, false))}
        >
          <CollapseAllIcon />
        </button>
      </div>
      {metadata.state_components.map((componentInfo) => {
        const component = config.observation.state_components.find(
          (item) => item.name === componentInfo.name,
        );
        if (component === undefined) {
          return null;
        }
        const enabled = component.mode !== "exclude";
        const rows = stateFeatureRows(component.name, componentInfo.features);
        const zeroedCount = rows.filter((row) => isRowZeroed(config, component, row)).length;
        const allRowsEnabled = enabled && zeroedCount === 0;
        return (
          <details
            className="state-component-panel"
            key={component.name}
            open={openSections[component.name] ?? true}
            onToggle={(event) => setSectionOpen(component.name, event.currentTarget.open)}
          >
            <summary className="state-component-summary">
              <span>
                <strong>{componentInfo.label}</strong>
                <small>
                  {componentSummary(
                    component.name,
                    rows.length,
                    rows.reduce((count, row) => count + row.featureNames.length, 0),
                    zeroedCount,
                    enabled,
                  )}
                </small>
              </span>
              <SwitchControl
                checked={enabled}
                hideLabel
                label="category enabled"
                tooltip={enabled ? "Disable category" : "Enable category"}
                onChange={(checked) => setComponentEnabled(component.name, checked)}
              />
            </summary>

            <div className="state-component-body">
              <ComponentSettings
                component={component}
                disabled={!enabled}
                updateComponent={updateComponent}
              />
              <table className="state-feature-table">
                <thead>
                  <tr>
                    <th scope="col">Entry</th>
                    <th scope="col">Type</th>
                    <th scope="col">Range / size</th>
                    <th scope="col">
                      <span className="state-feature-enabled-header">
                        <SwitchControl
                          checked={allRowsEnabled}
                          disabled={!enabled}
                          hideLabel
                          label="all entries enabled"
                          tooltip={
                            allRowsEnabled
                              ? "Zero out all entries in this category"
                              : "Enable all entries in this category"
                          }
                          onChange={(checked) =>
                            setFeaturesZeroed(
                              component,
                              rows.flatMap((row) => row.featureNames),
                              !checked,
                            )
                          }
                        />
                      </span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => {
                    const rowEnabled = !isRowZeroed(config, component, row);
                    return (
                      <tr key={row.id}>
                        <td>
                          <div className="state-feature-name">
                            <span>{row.label}</span>
                            <button
                              aria-label={`${row.label}: ${row.help}`}
                              className="field-help"
                              data-tooltip={row.help}
                              type="button"
                            >
                              ?
                            </button>
                            {row.id === "track_position.lap_progress" ? (
                              <ProgressSourceToggle
                                disabled={!enabled}
                                value={component.progress_source ?? "segment_progress"}
                                onChange={(value) =>
                                  updateComponent(component.name, { progress_source: value })
                                }
                              />
                            ) : null}
                          </div>
                        </td>
                        <td className="state-feature-kind">{row.kind}</td>
                        <td className="state-feature-range">{row.range}</td>
                        <td>
                          <SwitchControl
                            checked={rowEnabled}
                            disabled={!enabled}
                            hideLabel
                            label="entry enabled"
                            tooltip={rowEnabled ? "Zero out this entry" : "Enable this entry"}
                            onChange={(checked) =>
                              setFeaturesZeroed(component, row.featureNames, !checked)
                            }
                          />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </details>
        );
      })}
    </div>
  );
}

function ComponentSettings({
  component,
  disabled,
  updateComponent,
}: {
  component: StateComponentConfig;
  disabled: boolean;
  updateComponent: (name: string, patch: Partial<StateComponentConfig>) => void;
}) {
  const showSettings = component.name === "control_history" || disabled;
  if (!showSettings) {
    return null;
  }
  return (
    <div className="state-component-settings">
      {component.name === "control_history" ? (
        <IntegerField
          help="Number of prior action samples exposed in the state vector."
          label="History length"
          min={1}
          value={component.length ?? 1}
          onChange={(value) => updateComponent(component.name, { length: value })}
        />
      ) : null}
      {disabled ? <span className="state-component-disabled">category disabled</span> : null}
    </div>
  );
}

function SwitchControl({
  checked,
  disabled = false,
  hideLabel = false,
  label,
  tooltip,
  onChange,
}: {
  checked: boolean;
  disabled?: boolean;
  hideLabel?: boolean;
  label: string;
  tooltip?: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label
      className={disabled ? "state-switch disabled tooltip-anchor" : "state-switch tooltip-anchor"}
      data-tooltip={tooltip}
    >
      <input
        checked={checked}
        disabled={disabled}
        type="checkbox"
        onChange={(event) => onChange(event.target.checked)}
      />
      <span aria-hidden="true" />
      {hideLabel ? null : <strong>{label}</strong>}
    </label>
  );
}

function ProgressSourceToggle({
  disabled,
  value,
  onChange,
}: {
  disabled: boolean;
  value: NonNullable<StateComponentConfig["progress_source"]>;
  onChange: (value: NonNullable<StateComponentConfig["progress_source"]>) => void;
}) {
  return (
    <fieldset className="state-progress-toggle" disabled={disabled}>
      <legend>Progress scalar source</legend>
      {PROGRESS_SOURCE_OPTIONS.map((option) => (
        <button
          className={option === value ? "active" : undefined}
          key={option}
          type="button"
          onClick={() => onChange(option)}
        >
          {progressOptionLabel(option)}
        </button>
      ))}
    </fieldset>
  );
}

function componentSummary(
  componentName: string,
  rowCount: number,
  featureCount: number,
  zeroedCount: number,
  enabled: boolean,
) {
  if (!enabled) {
    return "category off";
  }
  if (componentName === "course_context") {
    return zeroedCount > 0
      ? `${featureCount}-wide one-hot · zeroed`
      : `${featureCount}-wide one-hot`;
  }
  if (zeroedCount === 0) {
    return `${rowCount} entries`;
  }
  return `${rowCount} entries · ${zeroedCount} zeroed`;
}

function allStateComponentsOpen(metadata: ConfigMetadata, open: boolean) {
  return Object.fromEntries(metadata.state_components.map((component) => [component.name, open]));
}

function ShapeMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="shape-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function effectiveFeatureMode(
  config: ManagedRunConfig,
  component: StateComponentConfig,
  featureName: string,
): StateFeatureConfig["mode"] {
  return (
    config.observation.state_feature_modes.find((feature) => feature.name === featureName)?.mode ??
    component.mode
  );
}

interface StateFeatureRow {
  id: string;
  label: string;
  help: string;
  kind: string;
  range: string;
  featureNames: readonly string[];
}

function stateFeatureRows(
  componentName: string,
  features: ConfigMetadata["state_components"][number]["features"],
): StateFeatureRow[] {
  if (componentName === "course_context" && features.length > 0) {
    return [
      {
        id: "course_context.course_id",
        label: "Course id",
        help: `Current course encoded as a ${features.length}-wide one-hot vector.`,
        kind: "one-hot",
        range: `${features.length} entries`,
        featureNames: features.map((feature) => feature.name),
      },
    ];
  }
  return features.map((feature) => {
    const info = stateFeatureInfo(componentName, feature.name);
    return {
      id: feature.name,
      label: info.label,
      help: info.help,
      kind: featureKind(feature.name, feature.low),
      range: formatFeatureRange(feature.low, feature.high),
      featureNames: [feature.name],
    };
  });
}

function featureKind(featureName: string, low: number) {
  if (BOOLEAN_FEATURES.has(featureName)) {
    return "binary";
  }
  if (low < 0) {
    return "signed scalar";
  }
  return "scalar";
}

function progressOptionLabel(value: NonNullable<StateComponentConfig["progress_source"]>) {
  if (value === "lap_progress") {
    return "Continuous progress";
  }
  if (value === "segment_progress") {
    return "Lap segment";
  }
  return value;
}

const PROGRESS_SOURCE_OPTIONS: readonly NonNullable<StateComponentConfig["progress_source"]>[] = [
  "lap_progress",
  "segment_progress",
];

const BOOLEAN_FEATURES = new Set([
  "surface_state.on_dirt_surface",
  "surface_state.on_ice_surface",
  "surface_state.on_refill_surface",
  "track_position.outside_track_bounds",
  "vehicle_state.airborne",
  "vehicle_state.boost_active",
  "vehicle_state.boost_unlocked",
  "vehicle_state.reverse_active",
  "vehicle_state.sliding_active",
]);

function isRowZeroed(
  config: ManagedRunConfig,
  component: StateComponentConfig,
  row: StateFeatureRow,
) {
  return row.featureNames.every(
    (featureName) => effectiveFeatureMode(config, component, featureName) === "zero",
  );
}

function formatFeatureRange(low: number, high: number) {
  return `[${formatBound(low)}, ${formatBound(high)}]`;
}

function formatBound(value: number) {
  if (Number.isInteger(value)) {
    return String(value);
  }
  return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}
