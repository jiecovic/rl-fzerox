// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/ObservationSection.tsx
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
} from "@/shared/api/contract";
import { StateComponentPanels } from "./observation/StateComponentPanels";

interface ObservationSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  checkpointLocked?: boolean;
  preview: PolicyArchitecturePreview | null;
  setConfig: (config: ManagedRunConfig) => void;
}

export function ObservationSection({
  config,
  defaultConfig,
  metadata,
  checkpointLocked = false,
  preview,
  setConfig,
}: ObservationSectionProps) {
  const updateObservation = (patch: Partial<ManagedRunConfig["observation"]>) => {
    setConfig({ ...config, observation: { ...config.observation, ...patch } });
  };
  const updatePolicy = (patch: Partial<ManagedRunConfig["policy"]>) => {
    setConfig({ ...config, policy: { ...config.policy, ...patch } });
  };
  const stackModeOptions = metadata.stack_modes.map(
    (option) => option.value,
  ) as ManagedRunConfig["observation"]["stack_mode"][];
  const resizeFilterOptions = metadata.resize_filters.map(
    (option) => option.value,
  ) as ManagedRunConfig["observation"]["resize_filter"][];
  const resolutionModeOptions: ManagedRunConfig["observation"]["resolution_mode"][] = [
    "preset",
    "custom",
  ];
  const selectedPreset = metadata.observation_presets.find(
    (preset) => preset.value === config.observation.preset,
  );
  const customResolution = config.observation.custom_resolution;
  const sourceGeometry = metadata.observation_source_geometries.find(
    (geometry) => geometry.renderer === config.environment.renderer,
  );
  const fallbackCustomHeight =
    selectedPreset?.height ?? metadata.observation_resolution_bounds.min_dimension;
  const fallbackCustomWidth =
    selectedPreset?.width ?? metadata.observation_resolution_bounds.min_dimension;
  const previewGeometry =
    preview !== null
      ? {
          height: preview.image_shape.height,
          width: preview.image_shape.width,
          channels: preview.image_shape.channels,
        }
      : null;
  const configuredGeometry =
    config.observation.resolution_mode === "custom" && customResolution !== null
      ? { height: customResolution.height, width: customResolution.width, channels: null }
      : selectedPreset !== undefined
        ? { height: selectedPreset.height, width: selectedPreset.width, channels: null }
        : null;
  const activeGeometry = previewGeometry ?? configuredGeometry;
  const imageMetricValue =
    activeGeometry === null
      ? "pending"
      : activeGeometry.channels === null
        ? `${activeGeometry.height} x ${activeGeometry.width}`
        : `${activeGeometry.height} x ${activeGeometry.width} x ${activeGeometry.channels}`;
  const pixelCountValue =
    activeGeometry === null ? "pending" : `${activeGeometry.height * activeGeometry.width} px`;
  const aspectRatioValue =
    activeGeometry === null
      ? "pending"
      : formatAspectRatio(activeGeometry.width, activeGeometry.height);
  const sourceCropValue =
    sourceGeometry === undefined ? "pending" : `${sourceGeometry.height} x ${sourceGeometry.width}`;

  function updateResolutionMode(mode: ManagedRunConfig["observation"]["resolution_mode"]) {
    if (mode === "preset") {
      updateObservation({
        resolution_mode: "preset",
        custom_resolution: null,
      });
      return;
    }
    updateObservation({
      resolution_mode: "custom",
      custom_resolution: customResolution ?? {
        height: fallbackCustomHeight,
        width: fallbackCustomWidth,
      },
    });
  }

  return (
    <div className="config-stack">
      <div className="form-grid two">
        <ConfigPanel title="Image observation">
          <div className="config-field-grid image-observation-grid">
            <fieldset
              className="fork-lock-fieldset image-observation-fieldset"
              disabled={checkpointLocked}
            >
              <SelectField
                help="Choose a stable native preset or provide a bounded custom resize target."
                label="Resolution source"
                options={resolutionModeOptions}
                resetValue={defaultConfig.observation.resolution_mode}
                value={config.observation.resolution_mode}
                onChange={updateResolutionMode}
              />
              {config.observation.resolution_mode === "preset" ? (
                <SelectField
                  help="Rendered crop preset used for the policy image input."
                  label="Input resolution"
                  options={metadata.observation_presets.map((preset) => preset.value)}
                  resetValue={defaultConfig.observation.preset}
                  value={config.observation.preset}
                  onChange={(value) => updateObservation({ preset: value })}
                />
              ) : (
                <div className="custom-resolution-grid">
                  <IntegerField
                    help={`Custom target height after crop. Allowed range: ${metadata.observation_resolution_bounds.min_dimension}-${metadata.observation_resolution_bounds.max_height}px.`}
                    label="Custom height"
                    min={metadata.observation_resolution_bounds.min_dimension}
                    max={metadata.observation_resolution_bounds.max_height}
                    resetValue={fallbackCustomHeight}
                    value={customResolution?.height ?? fallbackCustomHeight}
                    onChange={(value) =>
                      updateObservation({
                        custom_resolution: {
                          height: value,
                          width: customResolution?.width ?? fallbackCustomWidth,
                        },
                      })
                    }
                  />
                  <IntegerField
                    help={`Custom target width after crop. Allowed range: ${metadata.observation_resolution_bounds.min_dimension}-${metadata.observation_resolution_bounds.max_width}px.`}
                    label="Custom width"
                    min={metadata.observation_resolution_bounds.min_dimension}
                    max={metadata.observation_resolution_bounds.max_width}
                    resetValue={fallbackCustomWidth}
                    value={customResolution?.width ?? fallbackCustomWidth}
                    onChange={(value) =>
                      updateObservation({
                        custom_resolution: {
                          height: customResolution?.height ?? fallbackCustomHeight,
                          width: value,
                        },
                      })
                    }
                  />
                </div>
              )}
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
            </fieldset>
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
            <ShapeMetric label="Image" value={imageMetricValue} />
            <ShapeMetric label="Source crop" value={sourceCropValue} />
            <ShapeMetric label="Pixels" value={pixelCountValue} />
            <ShapeMetric label="Aspect" value={aspectRatioValue} />
            <ShapeMetric
              label="State width"
              value={preview !== null ? String(preview.state_dim) : "pending"}
            />
            <ShapeMetric label="Stack" value={`${config.observation.frame_stack} frames`} />
            <ShapeMetric
              label="Zeroed entries"
              value={String(
                preview?.state_features.filter((feature) => feature.dropout_prob >= 1).length ?? 0,
              )}
            />
          </div>
        </ConfigPanel>
      </div>

      <ConfigPanel title="State vector components" wide>
        <StateComponentPanels
          checkpointLocked={checkpointLocked}
          config={config}
          defaultConfig={defaultConfig}
          metadata={metadata}
          updateObservation={updateObservation}
          updatePolicy={updatePolicy}
        />
      </ConfigPanel>
    </div>
  );
}

function formatAspectRatio(width: number, height: number): string {
  const divisor = gcd(width, height);
  return `${width / divisor}:${height / divisor}`;
}

function gcd(left: number, right: number): number {
  let a = Math.abs(left);
  let b = Math.abs(right);
  while (b !== 0) {
    const remainder = a % b;
    a = b;
    b = remainder;
  }
  return a === 0 ? 1 : a;
}

function ShapeMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="shape-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
