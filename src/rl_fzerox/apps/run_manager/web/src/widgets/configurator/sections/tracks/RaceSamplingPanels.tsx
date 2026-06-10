// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/tracks/RaceSamplingPanels.tsx

import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";
import { ConfigGrid } from "@/widgets/configurator/ConfigLayout";
import { ConfigPanel } from "@/widgets/configurator/ConfigPanel";
import { IntegerField, NumberField } from "@/widgets/configurator/fields";
import { ChoiceStrip, ToggleChoiceStrip } from "@/widgets/configurator/sections/tracks/ChoiceStrip";
import { fixedEnvAssignmentSummary } from "@/widgets/configurator/sections/tracks/fixedEnvAssignment";
import {
  formatTrackOptionLabel,
  RACE_MODE_DESCRIPTIONS,
  TRACK_SAMPLING_DESCRIPTIONS,
} from "@/widgets/configurator/sections/tracks/options";
import type {
  GpDifficulty,
  TracksConfig,
  TrackUpdate,
} from "@/widgets/configurator/sections/tracks/types";

interface RaceModePanelProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  updateTracks: TrackUpdate;
}

interface GpDifficultyPanelProps {
  config: ManagedRunConfig;
  defaultGpDifficulties: readonly GpDifficulty[];
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
    | "deficit_budget_ema_alpha"
    | "deficit_budget_max_weight"
    | "deficit_budget_min_weight"
    | "deficit_budget_uniform_fraction"
    | "deficit_budget_weight_update_rollouts"
    | "step_balance_ema_alpha"
    | "step_balance_max_weight_scale"
    | "step_balance_update_episodes"
  >;
  updateTracks: TrackUpdate;
}

export function RaceSetupPanels({
  config,
  defaultConfig,
  defaultGpDifficulties,
  metadata,
  updateTracks,
}: RaceModePanelProps & { defaultGpDifficulties: readonly GpDifficulty[] }) {
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
        defaultGpDifficulties={defaultGpDifficulties}
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
  defaultGpDifficulties,
  metadata,
  updateTracks,
}: GpDifficultyPanelProps) {
  const selectedDifficulties = selectedGpDifficulties(
    config.tracks.gp_difficulties,
    defaultGpDifficulties,
  );
  const selectedLabels = metadata.gp_difficulties
    .filter((option) => selectedDifficulties.includes(option.value as GpDifficulty))
    .map((option) => option.label);
  return (
    <ConfigPanel
      onReset={() => updateTracks({ gp_difficulties: [...defaultGpDifficulties] })}
      title="GP difficulties"
    >
      <ToggleChoiceStrip
        description={
          config.tracks.race_mode === "gp_race"
            ? selectedLabels.join(", ")
            : "Only used when GP race mode is selected."
        }
        options={metadata.gp_difficulties.map((option) => ({
          active: selectedDifficulties.includes(option.value as GpDifficulty),
          disabled: config.tracks.race_mode !== "gp_race",
          key: option.value,
          label: option.label,
          tooltip:
            config.tracks.race_mode !== "gp_race"
              ? "GP difficulty is only applied to GP race baselines."
              : undefined,
          onClick: () => {
            const difficulty = option.value as GpDifficulty;
            const nextDifficulties = toggleGpDifficulty(selectedDifficulties, difficulty);
            if (nextDifficulties === null) {
              return;
            }
            updateTracks({ gp_difficulties: nextDifficulties });
          },
        }))}
      />
    </ConfigPanel>
  );
}

function selectedGpDifficulties(
  configured: readonly GpDifficulty[],
  defaults: readonly GpDifficulty[],
) {
  return configured.length > 0 ? configured : defaults;
}

function toggleGpDifficulty(
  currentDifficulties: readonly GpDifficulty[],
  difficulty: GpDifficulty,
) {
  if (currentDifficulties.includes(difficulty)) {
    if (currentDifficulties.length === 1) {
      return null;
    }
    return currentDifficulties.filter((currentDifficulty) => currentDifficulty !== difficulty);
  }
  return [...currentDifficulties, difficulty];
}

export function CourseSamplingPanel({
  config,
  defaultConfig,
  metadata,
  samplingDefaults,
  updateTracks,
}: CourseSamplingPanelProps) {
  const usesDynamicStepBalancing =
    config.tracks.sampling_mode === "step_balanced" ||
    config.tracks.sampling_mode === "adaptive_step_balanced";
  const usesAdaptiveStepBalancing = config.tracks.sampling_mode === "adaptive_step_balanced";
  const usesDeficitBudget = config.tracks.sampling_mode === "deficit_budget";
  const usesFixedEnvAssignment = config.tracks.sampling_mode === "fixed_env";
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
          {usesFixedEnvAssignment ? <FixedEnvAssignmentPreview config={config} /> : null}
          {usesDeficitBudget ? (
            <DeficitBudgetFields
              config={config}
              defaultConfig={defaultConfig}
              updateTracks={updateTracks}
            />
          ) : null}
        </div>
      </ConfigPanel>
    </ConfigGrid>
  );
}

function FixedEnvAssignmentPreview({ config }: { config: ManagedRunConfig }) {
  const summary = fixedEnvAssignmentSummary(config);
  const allowedCourseCounts = allowedFixedCourseCounts(config.train.num_envs);
  return (
    <div
      className={[
        "border px-3 py-2 text-[13px]",
        summary.issue === null
          ? "border-app-border bg-app-surface"
          : "border-app-danger/70 bg-app-danger/10 text-app-danger",
      ].join(" ")}
    >
      {summary.issue === null && summary.envsPerCourse !== null ? (
        <span>
          {config.train.num_envs} envs · {summary.activeCourseCount} courses ·{" "}
          {summary.envsPerCourse} env{summary.envsPerCourse === 1 ? "" : "s"}/course
        </span>
      ) : (
        <span>{summary.issue}</span>
      )}
      <span className="ml-3 text-app-muted">
        Allowed active course counts for {config.train.num_envs} envs:{" "}
        {allowedCourseCounts.join(", ")}
      </span>
    </div>
  );
}

function allowedFixedCourseCounts(numEnvs: number): number[] {
  const counts: number[] = [];
  for (let count = 1; count <= numEnvs; count += 1) {
    if (numEnvs % count === 0) {
      counts.push(count);
    }
  }
  return counts;
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

function DeficitBudgetFields({
  config,
  defaultConfig,
  updateTracks,
}: {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  updateTracks: TrackUpdate;
}) {
  const equalCoverageShare = Math.max(
    0,
    Math.min(1, config.tracks.deficit_budget_uniform_fraction),
  );
  const difficultyFocusShare = 1 - equalCoverageShare;
  const equalCoverageEnvEquivalent = equalCoverageShare * config.train.num_envs;
  const difficultyFocusEnvEquivalent = difficultyFocusShare * config.train.num_envs;

  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))] gap-3">
      <NumberField
        help="Fraction of each rollout step budget reserved as equal coverage for every active course."
        label="Equal coverage share"
        resetValue={defaultConfig.tracks.deficit_budget_uniform_fraction}
        step="0.05"
        value={config.tracks.deficit_budget_uniform_fraction}
        onChange={(value) => updateTracks({ deficit_budget_uniform_fraction: value })}
      />
      <div className="grid content-start gap-1 border border-app-border bg-app-surface-muted px-3 py-2">
        <span className="text-[11px] tracking-[0.05em] text-app-muted uppercase">
          Difficulty focus share
        </span>
        <strong className="text-base tabular-nums text-app-text">
          {formatPercent(difficultyFocusShare)}
        </strong>
        <span className="text-xs leading-normal text-app-muted">
          Budget-equivalent split at {config.train.num_envs} envs:{" "}
          {equalCoverageEnvEquivalent.toFixed(1)} equal · {difficultyFocusEnvEquivalent.toFixed(1)}{" "}
          difficulty.
        </span>
      </div>
      <NumberField
        help="Lowest adaptive difficulty weight before uniform budget mixing."
        label="Min weight"
        resetValue={defaultConfig.tracks.deficit_budget_min_weight}
        step="0.1"
        value={config.tracks.deficit_budget_min_weight}
        onChange={(value) => updateTracks({ deficit_budget_min_weight: value })}
      />
      <NumberField
        help="Highest adaptive difficulty weight before uniform budget mixing."
        label="Max weight"
        resetValue={defaultConfig.tracks.deficit_budget_max_weight}
        step="0.1"
        value={config.tracks.deficit_budget_max_weight}
        onChange={(value) => updateTracks({ deficit_budget_max_weight: value })}
      />
      <NumberField
        help="EMA smoothing for the per-course difficulty score."
        label="Difficulty EMA"
        resetValue={defaultConfig.tracks.deficit_budget_ema_alpha}
        step="0.005"
        value={config.tracks.deficit_budget_ema_alpha}
        onChange={(value) => updateTracks({ deficit_budget_ema_alpha: value })}
      />
      <IntegerField
        help="Rollouts between difficulty-weight recomputations. Step deficits still update every rollout."
        label="Update rollouts"
        min={1}
        resetValue={defaultConfig.tracks.deficit_budget_weight_update_rollouts}
        value={config.tracks.deficit_budget_weight_update_rollouts}
        onChange={(value) => updateTracks({ deficit_budget_weight_update_rollouts: value })}
      />
    </div>
  );
}

function formatPercent(value: number) {
  return `${(Math.max(0, Math.min(1, value)) * 100).toFixed(0)}%`;
}
