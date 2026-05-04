import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentDisclosureMap } from "@/features/configurator/disclosureState";
import { IntegerField, ToggleSwitch } from "@/features/configurator/fields";
import type {
  ConfigMetadata,
  ManagedRunConfig,
  StateComponentConfig,
  StateFeatureConfig,
} from "@/shared/api/contract";

import {
  allStateComponentsOpen,
  componentSummary,
  isRowZeroed,
  stateFeatureRows,
} from "./featureRows";
import { ProgressSourceToggle } from "./ProgressSourceToggle";

interface StateComponentPanelsProps {
  checkpointLocked?: boolean;
  config: ManagedRunConfig;
  metadata: ConfigMetadata;
  updateObservation: (patch: Partial<ManagedRunConfig["observation"]>) => void;
}

export function StateComponentPanels({
  checkpointLocked = false,
  config,
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
        if (component === undefined) {
          return null;
        }
        const enabled = component.mode !== "exclude";
        const rows = stateFeatureRows(component.name, componentInfo.features);
        const zeroedCount = rows.filter((row) => isRowZeroed(config, component, row)).length;
        const allRowsEnabled = enabled && zeroedCount === 0;
        return (
          <details
            className="config-disclosure state-component-panel"
            key={component.name}
            open={openSections[component.name] ?? false}
            onToggle={(event) => setSectionOpen(component.name, event.currentTarget.open)}
          >
            <summary className="config-disclosure-summary state-component-summary">
              <span className="config-disclosure-title">
                <span className="config-disclosure-copy">
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
                onChange={(checked) => setComponentEnabled(component.name, checked)}
              />
            </summary>

            <div className="config-disclosure-body state-component-body">
              <ComponentSettings
                component={component}
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
                          <ToggleSwitch
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
