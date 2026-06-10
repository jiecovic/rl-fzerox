// web/run-manager/src/entities/runConfig/ui/sections/ObservationSection.tsx

import {
  type ConfigSectionPatch,
  type ConfigSetter,
  patchConfigSection,
} from "@/entities/runConfig/model/state";
import { StateComponentPanels } from "@/entities/runConfig/ui/sections/observation/StateComponentPanels";
import type {
  ConfigMetadata,
  ManagedRunConfig,
  PolicyArchitecturePreview,
} from "@/shared/api/contract";
import {
  ConfigFieldGroup,
  ConfigFieldset,
  ConfigGrid,
  ConfigStack,
} from "@/shared/ui/config/ConfigLayout";
import { ConfigPanel } from "@/shared/ui/config/ConfigPanel";
import {
  BooleanField,
  DiscreteSliderNumberField,
  IntegerField,
  SelectField,
} from "@/shared/ui/configFields";
import { FieldNote, FieldShell } from "@/shared/ui/Field";

type ObservationResolution = ManagedRunConfig["observation"]["resolution"];
type ObservationResolutionMode = ObservationResolution["mode"];
type ObservationPreset = Extract<ObservationResolution, { mode: "preset" }>["preset"];

interface ObservationSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  checkpointLocked?: boolean;
  preview: PolicyArchitecturePreview | null;
  setConfig: ConfigSetter;
}

export function ObservationSection({
  config,
  defaultConfig,
  metadata,
  checkpointLocked = false,
  preview,
  setConfig,
}: ObservationSectionProps) {
  const updateObservation = (patch: ConfigSectionPatch<"observation">) => {
    patchConfigSection(setConfig, "observation", patch);
  };
  const updatePolicy = (patch: ConfigSectionPatch<"policy">) => {
    patchConfigSection(setConfig, "policy", patch);
  };
  const stackModeOptions = metadata.stack_modes.map(
    (option) => option.value,
  ) as ManagedRunConfig["observation"]["stack_mode"][];
  const resizeFilterOptions = metadata.resize_filters.map(
    (option) => option.value,
  ) as ManagedRunConfig["observation"]["resize_filter"][];
  const observationPresetOptions = metadata.observation_presets.map(
    (preset) => preset.value,
  ) as ObservationPreset[];
  const observationPresetLabels = Object.fromEntries(
    metadata.observation_presets.map((preset) => [preset.value, preset.label]),
  ) as Partial<Record<ObservationPreset, string>>;
  const defaultPreset: ObservationPreset = observationPresetOptions[0] ?? "crop_84x84";
  const resolutionModeOptions: ObservationResolutionMode[] = ["preset", "custom", "source_crop"];
  const resolutionModeLabels: Partial<Record<ObservationResolutionMode, string>> = {
    preset: "Preset",
    custom: "Custom",
    source_crop: "Original crop",
  };
  const resolution = config.observation.resolution;
  const defaultResolution = defaultConfig.observation.resolution;
  const currentPreset = resolution.mode === "preset" ? resolution.preset : defaultPreset;
  const selectedPreset = metadata.observation_presets.find(
    (preset) => preset.value === currentPreset,
  );
  const customResolution = resolution.mode === "custom" ? resolution : null;
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
    resolution.mode === "custom" && customResolution !== null
      ? { height: customResolution.height, width: customResolution.width, channels: null }
      : resolution.mode === "source_crop" && sourceGeometry !== undefined
        ? { height: sourceGeometry.height, width: sourceGeometry.width, channels: null }
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
  const imageAspectRatioValue =
    activeGeometry === null
      ? "pending"
      : formatAspectRatio(activeGeometry.width, activeGeometry.height);
  const sourceCropValue =
    sourceGeometry === undefined ? "pending" : `${sourceGeometry.height} x ${sourceGeometry.width}`;
  const sourceAspectRatioValue =
    sourceGeometry === undefined
      ? "pending"
      : formatAspectRatio(sourceGeometry.width, sourceGeometry.height);

  function updateResolutionMode(mode: ObservationResolutionMode) {
    if (mode === "preset") {
      updateObservation({
        resolution: { mode: "preset", preset: currentPreset },
      });
      return;
    }
    if (mode === "source_crop") {
      updateObservation({
        resolution: { mode: "source_crop" },
      });
      return;
    }
    updateObservation({
      resolution: customResolution ?? {
        mode: "custom",
        height: fallbackCustomHeight,
        width: fallbackCustomWidth,
      },
    });
  }

  return (
    <ConfigStack>
      <ConfigGrid columns="two">
        <ConfigPanel title="Image observation">
          <ConfigFieldGroup className="grid-cols-2">
            <ConfigFieldset
              className="col-span-full grid-cols-2 items-start [&>.range-field]:col-span-full"
              disabled={checkpointLocked}
            >
              <SelectField
                help="Choose a named resize target, provide a bounded custom size, or keep the cropped source size."
                label="Resolution source"
                options={resolutionModeOptions}
                optionLabels={resolutionModeLabels}
                resetValue={defaultResolution.mode}
                value={resolution.mode}
                onChange={updateResolutionMode}
              />
              {resolution.mode === "preset" ? (
                <SelectField
                  help="Rendered crop preset used for the policy image input."
                  label="Input resolution"
                  options={observationPresetOptions}
                  optionLabels={observationPresetLabels}
                  resetValue={
                    defaultResolution.mode === "preset" ? defaultResolution.preset : defaultPreset
                  }
                  value={resolution.preset}
                  onChange={(value) =>
                    updateObservation({ resolution: { mode: "preset", preset: value } })
                  }
                />
              ) : resolution.mode === "custom" ? (
                <div className="col-span-full grid grid-cols-2 gap-3">
                  <IntegerField
                    help={`Custom target height after crop. Allowed range: ${metadata.observation_resolution_bounds.min_dimension}-${metadata.observation_resolution_bounds.max_height}px.`}
                    label="Custom height"
                    min={metadata.observation_resolution_bounds.min_dimension}
                    max={metadata.observation_resolution_bounds.max_height}
                    resetValue={fallbackCustomHeight}
                    value={customResolution?.height ?? fallbackCustomHeight}
                    onChange={(value) =>
                      updateObservation({
                        resolution: {
                          mode: "custom",
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
                        resolution: {
                          mode: "custom",
                          height: customResolution?.height ?? fallbackCustomHeight,
                          width: value,
                        },
                      })
                    }
                  />
                </div>
              ) : (
                <FieldShell>
                  <span>Input resolution</span>
                  <FieldNote className="grid min-h-10 items-center border border-app-border bg-app-surface px-2.5 text-app-text">
                    {sourceGeometry === undefined
                      ? "pending"
                      : `${sourceGeometry.height} x ${sourceGeometry.width}`}
                  </FieldNote>
                </FieldShell>
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
            </ConfigFieldset>
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
          </ConfigFieldGroup>
        </ConfigPanel>

        <ConfigPanel title="Derived shape">
          <div className="grid grid-cols-2 gap-2.5 max-[560px]:grid-cols-1">
            <ShapeMetric label="Image" value={imageMetricValue} />
            <ShapeMetric label="Source crop" value={sourceCropValue} />
            <ShapeMetric label="Image aspect" value={imageAspectRatioValue} />
            <ShapeMetric label="Source aspect" value={sourceAspectRatioValue} />
            <ShapeMetric label="Pixels" value={pixelCountValue} />
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
      </ConfigGrid>

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
    </ConfigStack>
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
    <div className="border border-app-border bg-app-surface p-2.5">
      <span className="mb-1 block text-xs text-app-muted">{label}</span>
      <strong className="font-bold tabular-nums text-app-text">{value}</strong>
    </div>
  );
}
