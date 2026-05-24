// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/observation/featureRows.ts
import { stateFeatureInfo } from "@/features/configurator/sections/stateFeatureInfo";
import type { ConfigMetadata, ManagedRunConfig, StateComponentConfig } from "@/shared/api/contract";

export const TRACK_POSITION_PROGRESS_ROW_ID = "track_position.lap_progress";

export interface StateFeatureRow {
  id: string;
  label: string;
  help: string;
  kind: string;
  range: string;
  featureNames: readonly string[];
  defaultEnabled: boolean;
  auxiliaryTargetName?: ManagedRunConfig["policy"]["auxiliary_state_losses"][number]["name"];
  auxiliarySupportsGroundedOnly: boolean;
}

type StateComponentInfo = ConfigMetadata["state_components"][number];

export function stateComponentInfoForConfig(
  componentInfo: StateComponentInfo,
  config: ManagedRunConfig,
): StateComponentInfo {
  if (componentInfo.name !== "control_history" || config.action.lean_output_mode === "three_way") {
    return componentInfo;
  }
  return {
    ...componentInfo,
    features: componentInfo.features.flatMap((feature) => independentLeanHistoryFeatures(feature)),
  };
}

function independentLeanHistoryFeatures(
  feature: StateComponentInfo["features"][number],
): StateComponentInfo["features"] {
  const match = /^control_history\.prev_lean_(\d+)$/.exec(feature.name);
  if (match === null) {
    return [feature];
  }
  const [, age = ""] = match;
  return [
    { ...feature, name: `control_history.prev_lean_left_${age}`, low: 0, high: 1 },
    { ...feature, name: `control_history.prev_lean_right_${age}`, low: 0, high: 1 },
  ];
}

export function stateFeatureRows(
  componentName: string,
  features: ConfigMetadata["state_components"][number]["features"],
): StateFeatureRow[] {
  if (componentName === "course_context" && features.length > 0) {
    const auxiliaryTargetName = features[0]?.auxiliary_target_name ?? undefined;
    const auxiliarySupportsGroundedOnly = features.some(
      (feature) => feature.auxiliary_supports_grounded_only,
    );
    return [
      {
        id: "course_context.course_id",
        label: "Course id",
        help: `Current course encoded as a ${features.length}-wide one-hot vector.`,
        kind: "one-hot",
        range: `${features.length} entries`,
        featureNames: features.map((feature) => feature.name),
        defaultEnabled: true,
        auxiliaryTargetName,
        auxiliarySupportsGroundedOnly,
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
      defaultEnabled: feature.default_enabled,
      auxiliaryTargetName: feature.auxiliary_target_name ?? undefined,
      auxiliarySupportsGroundedOnly: feature.auxiliary_supports_grounded_only,
    };
  });
}

export function componentSummary(
  componentName: string,
  rowCount: number,
  featureCount: number,
  zeroedCount: number,
  notIncludedCount: number,
  enabled: boolean,
) {
  if (!enabled) {
    return "category off";
  }
  if (componentName === "course_context") {
    if (notIncludedCount > 0) {
      return `${featureCount}-wide one-hot · not included`;
    }
    return zeroedCount > 0
      ? `${featureCount}-wide one-hot · zeroed`
      : `${featureCount}-wide one-hot`;
  }
  if (zeroedCount === 0) {
    if (notIncludedCount === 0) {
      return `${rowCount} entries`;
    }
    return `${rowCount} entries · ${notIncludedCount} not included`;
  }
  if (notIncludedCount === 0) {
    return `${rowCount} entries · ${zeroedCount} zeroed`;
  }
  return `${rowCount} entries · ${zeroedCount} zeroed · ${notIncludedCount} not included`;
}

export function allStateComponentsOpen(metadata: ConfigMetadata, open: boolean) {
  return Object.fromEntries(metadata.state_components.map((component) => [component.name, open]));
}

export function effectiveFeatureDropoutProb(config: ManagedRunConfig, featureName: string): number {
  return (
    config.observation.state_feature_dropouts.find((feature) => feature.name === featureName)
      ?.dropout_prob ?? 0
  );
}

export function isRowZeroed(config: ManagedRunConfig, row: StateFeatureRow) {
  return row.featureNames.every(
    (featureName) => effectiveFeatureDropoutProb(config, featureName) >= 1.0,
  );
}

export function rowDropoutProb(config: ManagedRunConfig, row: StateFeatureRow) {
  return Math.max(
    0,
    ...row.featureNames.map((featureName) => effectiveFeatureDropoutProb(config, featureName)),
  );
}

export function progressOptionLabel(value: NonNullable<StateComponentConfig["progress_source"]>) {
  if (value === "lap_progress") {
    return "Continuous progress";
  }
  if (value === "segment_progress") {
    return "Lap segment";
  }
  return value;
}

export const progressSourceOptions: readonly NonNullable<
  StateComponentConfig["progress_source"]
>[] = ["lap_progress", "segment_progress"];

const booleanFeatures = new Set([
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

function featureKind(featureName: string, low: number) {
  if (booleanFeatures.has(featureName)) {
    return "binary";
  }
  if (low < 0) {
    return "signed scalar";
  }
  return "scalar";
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
