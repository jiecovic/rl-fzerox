// web/run-manager/src/entities/runConfig/ui/sections/EnvironmentSection.tsx

import { type ConfigSetter, patchConfigSection } from "@/entities/runConfig/model/state";
import {
  environmentSummaryRows,
  episodeDecisionSummary,
  episodeFrameSummary,
  noProgressSummary,
} from "@/entities/runConfig/ui/sections/environment/derived";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";
import { rendererNames } from "@/shared/api/renderers";
import { ConfigFieldGroup, ConfigFieldset, ConfigPanelGrid } from "@/shared/ui/config/ConfigLayout";
import { ConfigPanel } from "@/shared/ui/config/ConfigPanel";
import {
  BooleanField,
  RangeIntegerField,
  RangeNumberField,
  SelectField,
} from "@/shared/ui/configFields";

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
    <ConfigPanelGrid>
      <ConfigPanel
        title="Runtime"
        onReset={() =>
          updateEnvironment({
            renderer: defaultConfig.environment.renderer,
            camera_setting: defaultConfig.environment.camera_setting,
          })
        }
      >
        <ConfigFieldGroup>
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
        </ConfigFieldGroup>
      </ConfigPanel>

      <ConfigPanel
        title="Episode bounds"
        onReset={() =>
          updateEnvironment({
            max_episode_steps: defaultConfig.environment.max_episode_steps,
          })
        }
      >
        <ConfigFieldGroup>
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
        </ConfigFieldGroup>
      </ConfigPanel>

      <ConfigPanel
        title="GP lives"
        onReset={() =>
          updateEnvironment({
            randomize_gp_lives_on_reset: defaultConfig.environment.randomize_gp_lives_on_reset,
            gp_lives_jitter_min: defaultConfig.environment.gp_lives_jitter_min,
            gp_lives_jitter_max: defaultConfig.environment.gp_lives_jitter_max,
          })
        }
      >
        <ConfigFieldGroup>
          <BooleanField
            help="On each GP race reset, patch spare machines to the game's difficulty default plus an inclusive random jitter. This affects train and watch reset baselines, not Career Mode menu play."
            label="Randomize GP lives"
            resetValue={defaultConfig.environment.randomize_gp_lives_on_reset}
            value={config.environment.randomize_gp_lives_on_reset}
            onChange={(value) => updateEnvironment({ randomize_gp_lives_on_reset: value })}
          />
          <ConfigFieldset disabled={!config.environment.randomize_gp_lives_on_reset}>
            <RangeIntegerField
              help="Inclusive lower jitter added to the game default lives for the selected difficulty. Negative values can start an episode with fewer spare machines."
              label="Jitter min"
              max={8}
              min={-5}
              rangeStep={1}
              resetValue={defaultConfig.environment.gp_lives_jitter_min}
              value={config.environment.gp_lives_jitter_min}
              onChange={(value) => updateEnvironment({ gp_lives_jitter_min: value })}
            />
            <RangeIntegerField
              help="Inclusive upper jitter added to the game default lives for the selected difficulty. The final value is floored at zero and otherwise not upper-clamped."
              label="Jitter max"
              max={12}
              min={-5}
              rangeStep={1}
              resetValue={defaultConfig.environment.gp_lives_jitter_max}
              value={config.environment.gp_lives_jitter_max}
              onChange={(value) => updateEnvironment({ gp_lives_jitter_max: value })}
            />
            <div className="field-note">
              Defaults: Novice 5, Standard 4, Expert 3, Master 2 plus signed jitter. Final lives are
              floored at zero.
            </div>
          </ConfigFieldset>
        </ConfigFieldGroup>
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
        <ConfigFieldGroup>
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
          <ConfigFieldset disabled={!stallTruncationEnabled}>
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
          </ConfigFieldset>
        </ConfigFieldGroup>
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
    </ConfigPanelGrid>
  );
}
