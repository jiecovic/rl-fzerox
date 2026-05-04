import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import {
  BooleanField,
  DiscreteSliderNumberField,
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
            <fieldset className="fork-lock-fieldset" disabled={checkpointLocked}>
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
          checkpointLocked={checkpointLocked}
          config={config}
          metadata={metadata}
          updateObservation={updateObservation}
        />
      </ConfigPanel>
    </div>
  );
}

function ShapeMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="shape-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
