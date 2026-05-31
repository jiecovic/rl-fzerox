// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/RaceSamplingPanels.tsx
import { ConfigGrid } from "@/features/configurator/ConfigLayout";
import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import { IntegerField, NumberField } from "@/features/configurator/fields";
import { ChoiceStrip } from "@/features/configurator/sections/tracks/ChoiceStrip";
import {
  formatTrackOptionLabel,
  GP_DIFFICULTY_DESCRIPTIONS,
  RACE_MODE_DESCRIPTIONS,
  TRACK_SAMPLING_DESCRIPTIONS,
} from "@/features/configurator/sections/tracks/options";
import type {
  GpDifficulty,
  TracksConfig,
  TrackUpdate,
} from "@/features/configurator/sections/tracks/types";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";

interface RaceModePanelProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  updateTracks: TrackUpdate;
}

interface GpDifficultyPanelProps {
  config: ManagedRunConfig;
  defaultGpDifficulty: GpDifficulty;
  metadata: ConfigMetadata;
  updateTracks: TrackUpdate;
}

interface CourseSamplingPanelProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  samplingDefaults: Pick<
    TracksConfig,
    | "adaptive_step_balance_completion_weight"
    | "adaptive_step_balance_confidence_scale"
    | "adaptive_step_balance_min_confidence_episodes"
    | "adaptive_step_balance_target_completion"
    | "step_balance_ema_alpha"
    | "step_balance_max_weight_scale"
    | "step_balance_update_episodes"
  >;
  updateTracks: TrackUpdate;
}

export function RaceSetupPanels({
  config,
  defaultConfig,
  defaultGpDifficulty,
  metadata,
  updateTracks,
}: RaceModePanelProps & { defaultGpDifficulty: GpDifficulty }) {
  return (
    <ConfigGrid columns="two" className="items-stretch">
      <RaceModePanel
        config={config}
        defaultConfig={defaultConfig}
        metadata={metadata}
        updateTracks={updateTracks}
      />
      <GpDifficultyPanel
        config={config}
        defaultGpDifficulty={defaultGpDifficulty}
        metadata={metadata}
        updateTracks={updateTracks}
      />
    </ConfigGrid>
  );
}

export function RaceModePanel({
  config,
  defaultConfig,
  metadata,
  updateTracks,
}: RaceModePanelProps) {
  return (
    <ConfigPanel
      onReset={() => updateTracks({ race_mode: defaultConfig.tracks.race_mode })}
      title="Race mode"
    >
      <ChoiceStrip
        description={
          RACE_MODE_DESCRIPTIONS[config.tracks.race_mode] ?? RACE_MODE_DESCRIPTIONS.time_attack
        }
        options={metadata.race_modes.map((option) => ({
          active: config.tracks.race_mode === option.value,
          key: option.value,
          label: formatTrackOptionLabel(option.value),
          onClick: () =>
            updateTracks({
              race_mode: option.value as TracksConfig["race_mode"],
            }),
        }))}
      />
    </ConfigPanel>
  );
}

export function GpDifficultyPanel({
  config,
  defaultGpDifficulty,
  metadata,
  updateTracks,
}: GpDifficultyPanelProps) {
  return (
    <ConfigPanel
      onReset={() => updateTracks({ gp_difficulty: defaultGpDifficulty })}
      title="GP difficulty"
    >
      <ChoiceStrip
        description={
          config.tracks.race_mode === "gp_race"
            ? (GP_DIFFICULTY_DESCRIPTIONS[config.tracks.gp_difficulty ?? defaultGpDifficulty] ??
              GP_DIFFICULTY_DESCRIPTIONS.novice)
            : "Only used when GP race mode is selected."
        }
        options={metadata.gp_difficulties.map((option) => ({
          active: (config.tracks.gp_difficulty ?? defaultGpDifficulty) === option.value,
          disabled: config.tracks.race_mode !== "gp_race",
          key: option.value,
          label: option.label,
          tooltip:
            config.tracks.race_mode !== "gp_race"
              ? "GP difficulty is only applied to GP race baselines."
              : undefined,
          onClick: () =>
            updateTracks({
              gp_difficulty: option.value as GpDifficulty,
            }),
        }))}
      />
    </ConfigPanel>
  );
}

export function CourseSamplingPanel({
  config,
  defaultConfig,
  metadata,
  samplingDefaults,
  updateTracks,
}: CourseSamplingPanelProps) {
  const usesDynamicStepBalancing = config.tracks.sampling_mode !== "equal";
  const usesAdaptiveStepBalancing = config.tracks.sampling_mode === "adaptive_step_balanced";
  return (
    <ConfigGrid className="items-stretch">
      <ConfigPanel
        wide
        onReset={() =>
          updateTracks({
            sampling_mode: defaultConfig.tracks.sampling_mode,
            ...samplingDefaults,
          })
        }
        title="Course sampling"
      >
        <div className="grid gap-3">
          <ChoiceStrip
            description={
              TRACK_SAMPLING_DESCRIPTIONS[config.tracks.sampling_mode] ??
              TRACK_SAMPLING_DESCRIPTIONS.step_balanced
            }
            options={metadata.track_sampling_modes.map((option) => ({
              active: config.tracks.sampling_mode === option.value,
              key: option.value,
              label: formatTrackOptionLabel(option.value),
              onClick: () =>
                updateTracks({
                  sampling_mode: option.value as TracksConfig["sampling_mode"],
                }),
            }))}
          />
          {usesDynamicStepBalancing ? (
            <div className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))] gap-3">
              <IntegerField
                help="Episodes collected before recomputing course weights."
                label="Update episodes"
                min={1}
                resetValue={defaultConfig.tracks.step_balance_update_episodes}
                value={config.tracks.step_balance_update_episodes}
                onChange={(value) => updateTracks({ step_balance_update_episodes: value })}
              />
              <NumberField
                help="EMA smoothing for recent episode length and completion statistics."
                label="EMA alpha"
                resetValue={defaultConfig.tracks.step_balance_ema_alpha}
                step="0.01"
                value={config.tracks.step_balance_ema_alpha}
                onChange={(value) => updateTracks({ step_balance_ema_alpha: value })}
              />
              {usesAdaptiveStepBalancing ? (
                <AdaptiveSamplingFields
                  config={config}
                  defaultConfig={defaultConfig}
                  updateTracks={updateTracks}
                />
              ) : null}
            </div>
          ) : null}
        </div>
      </ConfigPanel>
    </ConfigGrid>
  );
}

function AdaptiveSamplingFields({
  config,
  defaultConfig,
  updateTracks,
}: {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  updateTracks: TrackUpdate;
}) {
  return (
    <>
      <NumberField
        help="Caps the adaptive frame-budget boost for low-completion courses. Reset frequency is still converted from target frame share using recent episode length."
        label="Max target scale"
        resetValue={defaultConfig.tracks.step_balance_max_weight_scale}
        step="0.1"
        value={config.tracks.step_balance_max_weight_scale}
        onChange={(value) => updateTracks({ step_balance_max_weight_scale: value })}
      />
      <IntegerField
        help="Courses below this many sampled episodes receive extra pressure before difficulty takes over."
        label="Min confidence episodes"
        min={1}
        resetValue={defaultConfig.tracks.adaptive_step_balance_min_confidence_episodes}
        value={config.tracks.adaptive_step_balance_min_confidence_episodes}
        onChange={(value) => updateTracks({ adaptive_step_balance_min_confidence_episodes: value })}
      />
      <NumberField
        help="Maximum target-share multiplier for courses that have not reached the confidence sample count."
        label="Confidence scale"
        resetValue={defaultConfig.tracks.adaptive_step_balance_confidence_scale}
        step="0.1"
        value={config.tracks.adaptive_step_balance_confidence_scale}
        onChange={(value) => updateTracks({ adaptive_step_balance_confidence_scale: value })}
      />
      <NumberField
        help="How strongly low completion raises a course's step-budget share."
        label="Completion weight"
        resetValue={defaultConfig.tracks.adaptive_step_balance_completion_weight}
        step="0.05"
        value={config.tracks.adaptive_step_balance_completion_weight}
        onChange={(value) => updateTracks({ adaptive_step_balance_completion_weight: value })}
      />
      <NumberField
        help="Courses below this completion fraction get extra sampling pressure."
        label="Target completion"
        resetValue={defaultConfig.tracks.adaptive_step_balance_target_completion}
        step="0.01"
        value={config.tracks.adaptive_step_balance_target_completion}
        onChange={(value) => updateTracks({ adaptive_step_balance_target_completion: value })}
      />
    </>
  );
}
