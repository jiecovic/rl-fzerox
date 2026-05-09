// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/observation/StateComponentPanels.tsx
import { type ReactNode, useEffect, useState } from "react";

import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/features/configurator/disclosureState";
import { IntegerField, ToggleSwitch } from "@/features/configurator/fields";
import { formatDecimalInput } from "@/features/configurator/fields/format";
import type {
  ConfigMetadata,
  ManagedRunConfig,
  StateComponentConfig,
  StateFeatureDropoutConfig,
} from "@/shared/api/contract";
import { HelpTooltipButton } from "@/shared/ui/HelpTooltipButton";

import {
  allStateComponentsOpen,
  componentSummary,
  isRowZeroed,
  rowDropoutProb,
  stateFeatureRows,
} from "./featureRows";
import { ProgressSourceToggle } from "./ProgressSourceToggle";

interface StateComponentPanelsProps {
  checkpointLocked?: boolean;
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  updateObservation: (patch: Partial<ManagedRunConfig["observation"]>) => void;
  updatePolicy: (patch: Partial<ManagedRunConfig["policy"]>) => void;
}

export function StateComponentPanels({
  checkpointLocked = false,
  config,
  defaultConfig,
  metadata,
  updateObservation,
  updatePolicy,
}: StateComponentPanelsProps) {
  const [openSections, setOpenSections] = usePersistentDisclosureMap(
    "run-manager:observation:state-components",
    allStateComponentsOpen(metadata, false),
  );

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

  function setComponentEnabled(componentName: StateComponentConfig["name"], enabled: boolean) {
    const currentComponents = new Map(
      config.observation.state_components.map((component) => [component.name, component]),
    );
    const nextComponentNames = new Set<StateComponentConfig["name"]>(currentComponents.keys());
    if (enabled) {
      nextComponentNames.add(componentName);
    } else {
      nextComponentNames.delete(componentName);
    }
    const componentFeatureNames = new Set(
      metadata.state_components
        .find((item) => item.name === componentName)
        ?.features.map((feature) => feature.name) ?? [],
    );
    updateObservation({
      state_components: defaultConfig.observation.state_components
        .filter((component) => nextComponentNames.has(component.name))
        .map((component) => currentComponents.get(component.name) ?? component),
      state_feature_dropouts: config.observation.state_feature_dropouts.filter(
        (feature) => enabled || !componentFeatureNames.has(feature.name),
      ),
    });
  }

  function setFeatureIncluded(
    componentInfo: ConfigMetadata["state_components"][number],
    componentName: StateComponentConfig["name"],
    featureNames: readonly string[],
    included: boolean,
  ) {
    const component = config.observation.state_components.find(
      (item) => item.name === componentName,
    );
    if (component === undefined) {
      return;
    }
    const nextIncludedFeatures = new Set(includedFeatureNames(componentInfo, component));
    if (included) {
      for (const featureName of featureNames) {
        nextIncludedFeatures.add(featureName);
      }
    } else {
      for (const featureName of featureNames) {
        nextIncludedFeatures.delete(featureName);
      }
    }
    const orderedIncludedFeatures = orderedFeatureNames(componentInfo, nextIncludedFeatures);
    const defaultIncludedFeatures = defaultIncludedFeatureNames(componentInfo);
    const nextIncluded = arraysEqual(orderedIncludedFeatures, defaultIncludedFeatures)
      ? null
      : orderedIncludedFeatures;
    const removedFeatureNames = new Set(included ? [] : featureNames);
    updateObservation({
      state_components: config.observation.state_components.map((item) =>
        item.name === componentName
          ? {
              ...item,
              included_features: nextIncluded,
            }
          : item,
      ),
      state_feature_dropouts: included
        ? config.observation.state_feature_dropouts
        : config.observation.state_feature_dropouts.filter(
            (feature) => !removedFeatureNames.has(feature.name),
          ),
    });
  }

  function setFeatureDropoutProb(featureNames: readonly string[], dropoutProb: number) {
    const featureNameSet = new Set(featureNames);
    const retainedDropouts = config.observation.state_feature_dropouts.filter(
      (feature) => !featureNameSet.has(feature.name),
    );
    const clampedDropoutProb = Math.max(0, Math.min(1, dropoutProb));
    const nextEntries = featureNames
      .map((featureName) => {
        const nextEntry: StateFeatureDropoutConfig = {
          name: featureName,
          dropout_prob: clampedDropoutProb,
        };
        return nextEntry.dropout_prob <= 0.0 ? null : nextEntry;
      })
      .filter((entry): entry is StateFeatureDropoutConfig => entry !== null);
    const nextDropouts =
      nextEntries.length === 0
        ? retainedDropouts
        : [...retainedDropouts, ...nextEntries].sort((left, right) =>
            left.name.localeCompare(right.name),
          );
    updateObservation({ state_feature_dropouts: nextDropouts });
  }

  function setRowsActive(
    componentInfo: ConfigMetadata["state_components"][number],
    componentName: StateComponentConfig["name"],
    rows: ReturnType<typeof stateFeatureRows>,
    active: boolean,
  ) {
    const component = config.observation.state_components.find(
      (item) => item.name === componentName,
    );
    if (component === undefined) {
      return;
    }
    const featureNames = rows.flatMap((row) => row.featureNames);
    const selectedFeatures = active ? new Set(featureNames) : new Set<string>();
    const orderedIncludedFeatures = orderedFeatureNames(componentInfo, selectedFeatures);
    const defaultIncludedFeatures = defaultIncludedFeatureNames(componentInfo);
    const nextIncluded = arraysEqual(orderedIncludedFeatures, defaultIncludedFeatures)
      ? null
      : orderedIncludedFeatures;
    const featureNameSet = new Set(featureNames);

    updateObservation({
      state_components: config.observation.state_components.map((item) =>
        item.name === componentName
          ? {
              ...item,
              included_features: nextIncluded,
            }
          : item,
      ),
      state_feature_dropouts: active
        ? config.observation.state_feature_dropouts
        : config.observation.state_feature_dropouts.filter(
            (feature) => !featureNameSet.has(feature.name),
          ),
    });
  }

  function isRowIncluded(
    componentInfo: ConfigMetadata["state_components"][number],
    component: StateComponentConfig,
    row: ReturnType<typeof stateFeatureRows>[number],
  ) {
    const includedFeatures = new Set(includedFeatureNames(componentInfo, component));
    return row.featureNames.every((featureName) => includedFeatures.has(featureName));
  }

  function setAuxiliaryStateEnabled(enabled: boolean) {
    updatePolicy({
      auxiliary_state_enabled: enabled,
      auxiliary_state_losses: enabled ? config.policy.auxiliary_state_losses : [],
    });
  }

  function auxiliaryLossRow(
    targetName: ManagedRunConfig["policy"]["auxiliary_state_losses"][number]["name"] | undefined,
  ) {
    if (targetName === undefined) {
      return null;
    }
    return config.policy.auxiliary_state_losses.find((loss) => loss.name === targetName) ?? null;
  }

  function setAuxiliaryLossEnabled(
    targetName: ManagedRunConfig["policy"]["auxiliary_state_losses"][number]["name"],
    enabled: boolean,
  ) {
    const retainedLosses = config.policy.auxiliary_state_losses.filter(
      (loss) => loss.name !== targetName,
    );
    const nextLosses = enabled
      ? [
          ...retainedLosses,
          {
            name: targetName,
            weight: 1.0,
            grounded_only: false,
          },
        ].sort((left, right) => left.name.localeCompare(right.name))
      : retainedLosses;
    updatePolicy({
      auxiliary_state_enabled: enabled ? true : config.policy.auxiliary_state_enabled,
      auxiliary_state_losses: nextLosses,
    });
  }

  function setAuxiliaryLossWeight(
    targetName: ManagedRunConfig["policy"]["auxiliary_state_losses"][number]["name"],
    weight: number,
  ) {
    updatePolicy({
      auxiliary_state_enabled: true,
      auxiliary_state_losses: config.policy.auxiliary_state_losses.map((loss) =>
        loss.name === targetName ? { ...loss, weight } : loss,
      ),
    });
  }

  function setAuxiliaryGroundedOnly(
    targetName: ManagedRunConfig["policy"]["auxiliary_state_losses"][number]["name"],
    groundedOnly: boolean,
  ) {
    updatePolicy({
      auxiliary_state_enabled: true,
      auxiliary_state_losses: config.policy.auxiliary_state_losses.map((loss) =>
        loss.name === targetName ? { ...loss, grounded_only: groundedOnly } : loss,
      ),
    });
  }

  return (
    <div className="state-vector-editor">
      <div className="state-auxiliary-toolbar">
        <span className="state-auxiliary-toolbar-copy">
          <strong>Auxiliary RAM supervision</strong>
          <small>Optional supervised targets over the shared policy latent.</small>
        </span>
        <ToggleSwitch
          checked={config.policy.auxiliary_state_enabled}
          disabled={false}
          hideLabel
          label="auxiliary state supervision enabled"
          tooltip={
            config.policy.auxiliary_state_enabled
              ? "Disable auxiliary RAM supervision and clear active aux losses"
              : "Enable auxiliary RAM supervision"
          }
          onChange={setAuxiliaryStateEnabled}
        />
      </div>
      <DisclosureToolbar
        collapseLabel="Collapse all state-vector categories"
        expandLabel="Expand all state-vector categories"
        onCollapseAll={() => setOpenSections(allStateComponentsOpen(metadata, false))}
        onExpandAll={() => setOpenSections(allStateComponentsOpen(metadata, true))}
      />
      {metadata.state_components.map((componentInfo) => {
        const component = config.observation.state_components.find(
          (item) => item.name === componentInfo.name,
        );
        const defaultComponent = defaultConfig.observation.state_components.find(
          (item) => item.name === componentInfo.name,
        );
        const displayComponent = component ?? defaultComponent;
        if (displayComponent === undefined) {
          return null;
        }
        const enabled = component !== undefined;
        const rows = stateFeatureRows(displayComponent.name, componentInfo.features);
        const notIncludedCount = enabled
          ? rows.filter((row) => !isRowIncluded(componentInfo, displayComponent, row)).length
          : rows.length;
        const zeroedCount = rows.filter(
          (row) => isRowIncluded(componentInfo, displayComponent, row) && isRowZeroed(config, row),
        ).length;
        const allRowsIncluded = enabled && notIncludedCount === 0;
        return (
          <details
            className="config-disclosure state-component-panel"
            key={displayComponent.name}
            open={openSections[displayComponent.name] ?? false}
            onToggle={(event) => setSectionOpen(displayComponent.name, event.currentTarget.open)}
          >
            <summary className="config-disclosure-summary state-component-summary">
              <span className="config-disclosure-title">
                <span className="config-disclosure-copy">
                  <strong>{componentInfo.label}</strong>
                  <small>
                    {componentSummary(
                      displayComponent.name,
                      rows.length,
                      rows.reduce((count, row) => count + row.featureNames.length, 0),
                      zeroedCount,
                      notIncludedCount,
                      enabled,
                    )}
                  </small>
                </span>
              </span>
              <ToggleSwitch
                checked={enabled}
                disabled={checkpointLocked}
                hideLabel
                label="category enabled"
                tooltip={
                  checkpointLocked
                    ? "Forked checkpoints keep the original state-vector shape."
                    : enabled
                      ? "Disable category"
                      : "Enable category"
                }
                onChange={(checked) => setComponentEnabled(displayComponent.name, checked)}
              />
            </summary>

            <div className="config-disclosure-body state-component-body">
              <ComponentSettings
                component={displayComponent}
                checkpointLocked={checkpointLocked}
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
                      <span className="state-feature-column-label">
                        <span>Episode dropout</span>
                        <HelpTooltipButton
                          label="Episode dropout"
                          text="Probability that this feature is replaced with 0 for the whole episode. 0 means never. 1 means always."
                        />
                      </span>
                    </th>
                    <th scope="col">
                      <span className="state-feature-column-label">
                        <span>Use value</span>
                        <HelpTooltipButton
                          label="Use value"
                          text="On: the policy receives this feature value. Off: the input slot remains, but the policy receives 0."
                        />
                      </span>
                    </th>
                    <th scope="col">
                      <span className="state-feature-enabled-header">
                        <span>Policy input</span>
                        <ToggleSwitch
                          checked={allRowsIncluded}
                          disabled={!enabled || checkpointLocked}
                          hideLabel
                          label="all entries used as policy input"
                          tooltip={
                            checkpointLocked
                              ? "Forked checkpoints keep the original state-vector shape."
                              : allRowsIncluded
                                ? "Remove every row in this category from the observation state"
                                : "Include every row in this category in the observation state"
                          }
                          onChange={(checked) =>
                            setRowsActive(componentInfo, displayComponent.name, rows, checked)
                          }
                        />
                      </span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => {
                    const rowIncluded =
                      enabled && isRowIncluded(componentInfo, displayComponent, row);
                    const effectiveDropoutProb = rowIncluded ? rowDropoutProb(config, row) : 0;
                    const auxiliaryLoss = auxiliaryLossRow(row.auxiliaryTargetName);
                    const auxiliaryLossEnabled = auxiliaryLoss !== null;
                    const auxiliaryTargetName = row.auxiliaryTargetName;
                    const progressSourceControl =
                      row.id === "track_position.lap_progress" ? (
                        <ProgressSourceToggle
                          disabled={!enabled}
                          value={displayComponent.progress_source ?? "segment_progress"}
                          onChange={(value) =>
                            updateComponent(displayComponent.name, {
                              progress_source: value,
                            })
                          }
                        />
                      ) : null;
                    return (
                      <tr key={row.id}>
                        <td>
                          <div className="state-feature-entry">
                            <div className="state-feature-name">
                              <span>{row.label}</span>
                              <HelpTooltipButton label={row.label} text={row.help} />
                            </div>
                            {auxiliaryTargetName !== undefined ? (
                              <FeatureAuxiliaryLossControls
                                auxiliaryEnabled={config.policy.auxiliary_state_enabled}
                                disabled={false}
                                featureLabel={row.label}
                                groundedOnly={auxiliaryLoss?.grounded_only ?? false}
                                lossEnabled={auxiliaryLossEnabled}
                                supportsGroundedOnly={row.auxiliarySupportsGroundedOnly}
                                weight={auxiliaryLoss?.weight ?? 1.0}
                                extraControls={progressSourceControl}
                                onToggle={(checked) =>
                                  setAuxiliaryLossEnabled(auxiliaryTargetName, checked)
                                }
                                onWeightChange={(value) =>
                                  setAuxiliaryLossWeight(auxiliaryTargetName, value)
                                }
                                onGroundedOnlyChange={(checked) =>
                                  setAuxiliaryGroundedOnly(auxiliaryTargetName, checked)
                                }
                              />
                            ) : (
                              progressSourceControl
                            )}
                          </div>
                        </td>
                        <td className="state-feature-kind">{row.kind}</td>
                        <td className="state-feature-range">{row.range}</td>
                        <td className="state-feature-dropout-cell">
                          <FeatureDropoutInput
                            disabled={!rowIncluded}
                            label={`${row.label} episode dropout`}
                            value={effectiveDropoutProb}
                            onChange={(value) => setFeatureDropoutProb(row.featureNames, value)}
                          />
                        </td>
                        <td className="state-feature-toggle-cell">
                          <ToggleSwitch
                            checked={effectiveDropoutProb < 1.0}
                            disabled={!rowIncluded}
                            hideLabel
                            label={`${row.label} uses real value`}
                            tooltip={
                              effectiveDropoutProb < 1.0
                                ? "Turn off to send 0 for this input."
                                : "Turn on to use this feature value."
                            }
                            onChange={(checked) =>
                              setFeatureDropoutProb(row.featureNames, checked ? 0.0 : 1.0)
                            }
                          />
                        </td>
                        <td>
                          <ToggleSwitch
                            checked={rowIncluded}
                            disabled={!enabled || checkpointLocked}
                            hideLabel
                            label="use entry as policy input"
                            tooltip={
                              checkpointLocked
                                ? "Forked checkpoints keep the original state-vector shape."
                                : rowIncluded
                                  ? "Remove this entry from the observation state"
                                  : "Include this entry in the observation state"
                            }
                            onChange={(checked) =>
                              setFeatureIncluded(
                                componentInfo,
                                displayComponent.name,
                                row.featureNames,
                                checked,
                              )
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

function FeatureDropoutInput({
  disabled,
  label,
  value,
  onChange,
}: {
  disabled: boolean;
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  const [rawValue, setRawValue] = useState(formatDecimalInput(value, "0.01"));

  useEffect(() => {
    setRawValue(formatDecimalInput(value, "0.01"));
  }, [value]);

  function tryCommitRaw(nextRawValue: string) {
    if (nextRawValue.trim() === "") {
      return;
    }
    const parsed = Number(nextRawValue);
    if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
      return;
    }
    onChange(Math.round(parsed * 100) / 100);
  }

  function commitValue() {
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
      setRawValue(formatDecimalInput(value, "0.01"));
      return;
    }
    const normalized = Math.round(parsed * 100) / 100;
    onChange(normalized);
    setRawValue(formatDecimalInput(normalized, "0.01"));
  }

  return (
    <input
      aria-label={label}
      className="state-feature-dropout-input"
      disabled={disabled}
      inputMode="decimal"
      max={1}
      min={0}
      step="0.01"
      type="number"
      value={rawValue}
      onBlur={commitValue}
      onChange={(event) => {
        const nextRawValue = event.target.value;
        setRawValue(nextRawValue);
        tryCommitRaw(nextRawValue);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.currentTarget.blur();
        }
      }}
    />
  );
}

function includedFeatureNames(
  componentInfo: ConfigMetadata["state_components"][number],
  component: StateComponentConfig,
): readonly string[] {
  return component.included_features ?? defaultIncludedFeatureNames(componentInfo);
}

function defaultIncludedFeatureNames(
  componentInfo: ConfigMetadata["state_components"][number],
): string[] {
  return componentInfo.features
    .filter((feature) => feature.default_enabled)
    .map((feature) => feature.name);
}

function orderedFeatureNames(
  componentInfo: ConfigMetadata["state_components"][number],
  selectedFeatures: ReadonlySet<string>,
): string[] {
  return componentInfo.features
    .map((feature) => feature.name)
    .filter((featureName) => selectedFeatures.has(featureName));
}

function arraysEqual(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function FeatureAuxiliaryLossControls({
  auxiliaryEnabled,
  disabled,
  extraControls,
  featureLabel,
  lossEnabled,
  weight,
  groundedOnly,
  supportsGroundedOnly,
  onToggle,
  onWeightChange,
  onGroundedOnlyChange,
}: {
  auxiliaryEnabled: boolean;
  disabled: boolean;
  extraControls?: ReactNode;
  featureLabel: string;
  lossEnabled: boolean;
  weight: number;
  groundedOnly: boolean;
  supportsGroundedOnly: boolean;
  onToggle: (checked: boolean) => void;
  onWeightChange: (value: number) => void;
  onGroundedOnlyChange: (checked: boolean) => void;
}) {
  if (disabled) {
    return <span className="state-feature-aux-unsupported">not available</span>;
  }

  const weightEnabled = auxiliaryEnabled && lossEnabled;
  return (
    <div className="state-feature-aux-controls">
      <ToggleSwitch
        checked={lossEnabled}
        disabled={false}
        hideLabel
        label={`${featureLabel} auxiliary loss enabled`}
        tooltip={lossEnabled ? "Disable aux loss" : "Enable aux loss"}
        onChange={onToggle}
      />
      <div className="state-feature-aux-weight" title="Auxiliary loss weight">
        <FeatureAuxWeightInput
          disabled={!weightEnabled}
          label={`${featureLabel} auxiliary loss weight`}
          value={weight}
          onChange={onWeightChange}
        />
      </div>
      {supportsGroundedOnly ? (
        <label className="state-feature-aux-grounded">
          <input
            checked={groundedOnly}
            disabled={!weightEnabled}
            type="checkbox"
            onChange={(event) => onGroundedOnlyChange(event.target.checked)}
          />
          <span>grounded only</span>
        </label>
      ) : null}
      {extraControls}
    </div>
  );
}

function FeatureAuxWeightInput({
  disabled,
  label,
  value,
  onChange,
}: {
  disabled: boolean;
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  const [rawValue, setRawValue] = useState(formatDecimalInput(value, "0.01"));

  useEffect(() => {
    setRawValue(formatDecimalInput(value, "0.01"));
  }, [value]);

  function tryCommitRaw(nextRawValue: string) {
    if (nextRawValue.trim() === "") {
      return;
    }
    const parsed = Number(nextRawValue);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return;
    }
    onChange(Math.round(parsed * 100) / 100);
  }

  function commitValue() {
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      setRawValue(formatDecimalInput(value, "0.01"));
      return;
    }
    const normalized = Math.round(parsed * 100) / 100;
    onChange(normalized);
    setRawValue(formatDecimalInput(normalized, "0.01"));
  }

  return (
    <input
      aria-label={label}
      className="state-feature-aux-weight-input"
      disabled={disabled}
      inputMode="decimal"
      min={0.01}
      step="0.01"
      type="number"
      value={rawValue}
      onBlur={commitValue}
      onChange={(event) => {
        const nextRawValue = event.target.value;
        setRawValue(nextRawValue);
        tryCommitRaw(nextRawValue);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.currentTarget.blur();
        }
      }}
    />
  );
}

function ComponentSettings({
  component,
  checkpointLocked = false,
  disabled,
  updateComponent,
}: {
  component: StateComponentConfig;
  checkpointLocked?: boolean;
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
        <fieldset className="fork-lock-fieldset" disabled={checkpointLocked}>
          <IntegerField
            help="Number of prior action samples exposed in the state vector."
            label="History length"
            min={1}
            value={component.length ?? 1}
            onChange={(value) => updateComponent(component.name, { length: value })}
          />
        </fieldset>
      ) : null}
      {disabled ? <span className="state-component-disabled">category disabled</span> : null}
    </div>
  );
}
