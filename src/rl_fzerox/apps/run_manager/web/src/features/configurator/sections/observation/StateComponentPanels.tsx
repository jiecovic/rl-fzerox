// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/observation/StateComponentPanels.tsx
import { useEffect, useState } from "react";

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
}

export function StateComponentPanels({
  checkpointLocked = false,
  config,
  defaultConfig,
  metadata,
  updateObservation,
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

  function setFeaturesZeroed(featureNames: readonly string[], zeroed: boolean) {
    const featureNameSet = new Set(featureNames);
    const retainedDropouts = config.observation.state_feature_dropouts.filter(
      (feature) => !featureNameSet.has(feature.name),
    );
    const nextEntries = featureNames
      .map((featureName) => {
        const nextEntry: StateFeatureDropoutConfig = {
          name: featureName,
          dropout_prob: zeroed ? 1.0 : 0.0,
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

  return (
    <div className="state-vector-editor">
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
        const zeroedCount = rows.filter((row) => isRowZeroed(config, row)).length;
        const allRowsEnabled = enabled && zeroedCount === 0;
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
                          text="Per-feature dropout probability from 0 to 1. 0 keeps the feature every episode. 1 drops it every episode. The dropout mask is sampled once on reset and stays fixed for the whole episode, not per step."
                        />
                      </span>
                    </th>
                    <th scope="col">
                      <span className="state-feature-enabled-header">
                        <ToggleSwitch
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
                    const rowEnabled = !isRowZeroed(config, row);
                    const effectiveDropoutProb = rowEnabled ? rowDropoutProb(config, row) : 1;
                    return (
                      <tr key={row.id}>
                        <td>
                          <div className="state-feature-name">
                            <span>{row.label}</span>
                            <HelpTooltipButton label={row.label} text={row.help} />
                            {row.id === "track_position.lap_progress" ? (
                              <ProgressSourceToggle
                                disabled={!enabled}
                                value={displayComponent.progress_source ?? "segment_progress"}
                                onChange={(value) =>
                                  updateComponent(displayComponent.name, { progress_source: value })
                                }
                              />
                            ) : null}
                          </div>
                        </td>
                        <td className="state-feature-kind">{row.kind}</td>
                        <td className="state-feature-range">{row.range}</td>
                        <td className="state-feature-dropout-cell">
                          <FeatureDropoutInput
                            disabled={!enabled || !rowEnabled}
                            label={`${row.label} episode dropout`}
                            value={effectiveDropoutProb}
                            onChange={(value) => setFeatureDropoutProb(row.featureNames, value)}
                          />
                        </td>
                        <td>
                          <ToggleSwitch
                            checked={rowEnabled}
                            disabled={!enabled}
                            hideLabel
                            label="entry enabled"
                            tooltip={rowEnabled ? "Zero out this entry" : "Enable this entry"}
                            onChange={(checked) => setFeaturesZeroed(row.featureNames, !checked)}
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
