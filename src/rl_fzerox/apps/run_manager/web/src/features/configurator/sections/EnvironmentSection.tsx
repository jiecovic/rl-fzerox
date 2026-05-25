// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/EnvironmentSection.tsx
import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import { type ConfigSetter, patchConfigSection } from "@/features/configurator/configurator/state";
import {
  BooleanField,
  RangeIntegerField,
  RangeNumberField,
  SelectField,
} from "@/features/configurator/fields";
import {
  environmentSummaryRows,
  episodeDecisionSummary,
  episodeFrameSummary,
  noProgressSummary,
} from "@/features/configurator/sections/environment/derived";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";
import { rendererNames } from "@/shared/api/renderers";

interface EnvironmentSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  setConfig: ConfigSetter;
}

export function EnvironmentSection({
  config,
  defaultConfig,
  metadata,
  setConfig,
}: EnvironmentSectionProps) {
  const updateEnvironment = (patch: Partial<ManagedRunConfig["environment"]>) => {
    patchConfigSection(setConfig, "environment", patch);
  };
  const stallTruncationEnabled = config.environment.progress_frontier_stall_limit_frames !== null;
  const rendererOptions = rendererNames(metadata, config.environment.renderer);
  const cameraOptions = metadata.camera_settings.map(
    (option) => option.value,
  ) as ManagedRunConfig["environment"]["camera_setting"][];
  const cameraOptionLabels = Object.fromEntries(
    metadata.camera_settings.map((option) => [option.value, option.label]),
  );

  return (
    <div className="config-stack training-panel-grid">
      <ConfigPanel
        title="Runtime"
        onReset={() =>
          updateEnvironment({
            renderer: defaultConfig.environment.renderer,
            camera_setting: defaultConfig.environment.camera_setting,
          })
        }
      >
        <div className="training-field-grid">
          <SelectField
            help="Video backend requested from the Mupen core. gliden64 matches the established training runs; angrylion is slower and mainly useful for stricter software-rendering comparisons."
            label="Renderer"
            options={rendererOptions}
            resetValue={defaultConfig.environment.renderer}
            value={config.environment.renderer}
            onChange={(value) => updateEnvironment({ renderer: value })}
          />
          <SelectField
            help="Camera mode synchronized during reset. close behind matches the current manager default and most training runs."
            label="Camera"
            optionLabels={cameraOptionLabels}
            options={cameraOptions}
            resetValue={defaultConfig.environment.camera_setting}
            value={config.environment.camera_setting}
            onChange={(value) => updateEnvironment({ camera_setting: value })}
          />
        </div>
      </ConfigPanel>

      <ConfigPanel
        title="Episode bounds"
        onReset={() =>
          updateEnvironment({
            max_episode_steps: defaultConfig.environment.max_episode_steps,
          })
        }
      >
        <div className="training-field-grid">
          <div className="field-with-note environment-note-field">
            <RangeIntegerField
              help="Hard cap on one episode, counted per native emulated frame. Policy decisions are fewer when action repeat is greater than one."
              label="Max episode steps"
              max={60_000}
              min={600}
              rangeStep={300}
              ticks={[
                { value: 600, label: "600" },
                { value: 12_000, label: "12k" },
                { value: 30_000, label: "30k" },
                { value: 60_000, label: "60k" },
              ]}
              resetValue={defaultConfig.environment.max_episode_steps}
              value={config.environment.max_episode_steps}
              onChange={(value) => updateEnvironment({ max_episode_steps: value })}
            />
            <div className="field-note">
              {episodeFrameSummary(config)}. {episodeDecisionSummary(config)}
            </div>
          </div>
        </div>
      </ConfigPanel>

      <ConfigPanel
        title="No-progress truncation"
        onReset={() =>
          updateEnvironment({
            progress_frontier_stall_limit_frames:
              defaultConfig.environment.progress_frontier_stall_limit_frames,
            progress_frontier_epsilon: defaultConfig.environment.progress_frontier_epsilon,
          })
        }
      >
        <div className="training-field-grid">
          <BooleanField
            help="Stop an episode that keeps running without meaningfully advancing the best race-distance frontier."
            label="Enable no-progress truncation"
            resetValue={defaultConfig.environment.progress_frontier_stall_limit_frames !== null}
            value={stallTruncationEnabled}
            onChange={(value) =>
              updateEnvironment({
                progress_frontier_stall_limit_frames: value
                  ? (config.environment.progress_frontier_stall_limit_frames ??
                    defaultConfig.environment.progress_frontier_stall_limit_frames ??
                    900)
                  : null,
              })
            }
          />
          <fieldset className="dependent-fieldset" disabled={!stallTruncationEnabled}>
            <div className="field-with-note environment-note-field">
              <RangeIntegerField
                help="Maximum internal frames allowed without beating the previous progress frontier by at least epsilon."
                label="Stall limit frames"
                max={7_200}
                min={60}
                rangeStep={60}
                ticks={[
                  { value: 60, label: "60" },
                  { value: 900, label: "900" },
                  { value: 3_600, label: "3.6k" },
                  { value: 7_200, label: "7.2k" },
                ]}
                resetValue={defaultConfig.environment.progress_frontier_stall_limit_frames ?? 900}
                value={config.environment.progress_frontier_stall_limit_frames ?? 900}
                onChange={(value) =>
                  updateEnvironment({ progress_frontier_stall_limit_frames: value })
                }
              />
              <div className="field-note">{noProgressSummary(config)}</div>
            </div>
            <RangeNumberField
              help="Frontier progress must satisfy new_frontier >= old_frontier + epsilon to reset the stall timer. Smaller epsilon resets the timer on tiny advances; larger epsilon demands cleaner progress."
              label="Frontier epsilon"
              max={1_000}
              min={0}
              numberStep="1"
              rangeStep={5}
              resetValue={defaultConfig.environment.progress_frontier_epsilon}
              value={config.environment.progress_frontier_epsilon}
              onChange={(value) => updateEnvironment({ progress_frontier_epsilon: value })}
            />
          </fieldset>
        </div>
      </ConfigPanel>

      <ConfigPanel title="Episode summary" wide>
        <table className="derived-table">
          <tbody>
            {environmentSummaryRows(config).map((row) => (
              <tr key={row.label}>
                <th>{row.label}</th>
                <td>{row.detail}</td>
                <td>{row.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ConfigPanel>
    </div>
  );
}
