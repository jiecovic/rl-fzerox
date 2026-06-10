// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/tracks/XCupSection.tsx

import type { ManagedRunConfig } from "@/shared/api/contract";
import { AppTooltip } from "@/shared/ui/Tooltip";
import { BooleanField, IntegerField, NumberField } from "@/widgets/configurator/fields";
import {
  courseCardClass,
  courseSelectedBadgeClass,
  cupSwitchGroupClass,
} from "@/widgets/configurator/sections/tracks/coursePoolStyle";
import { X_CUP } from "@/widgets/configurator/sections/tracks/options";
import { TrackCupBanner } from "@/widgets/configurator/sections/tracks/TrackCupBanner";
import type { TrackUpdate } from "@/widgets/configurator/sections/tracks/types";

interface XCupSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  collapsed: boolean;
  selectedCourseIds: readonly string[];
  xCupAvailable: boolean;
  xCupEnabled: boolean;
  onCollapsedChange: (cupId: string, collapsed: boolean) => void;
  updateTracks: TrackUpdate;
}

export function XCupSection({
  config,
  defaultConfig,
  collapsed,
  selectedCourseIds,
  xCupAvailable,
  xCupEnabled,
  onCollapsedChange,
  updateTracks,
}: XCupSectionProps) {
  const rotation = config.tracks.x_cup_auto_regeneration;
  const defaultRotation = defaultConfig.tracks.x_cup_auto_regeneration;
  const defaultMaxEpisodes = Math.max(rotation.min_episodes * 4, 100);
  const updateRotation = (patch: Partial<typeof rotation>) =>
    updateTracks({
      x_cup_auto_regeneration: {
        ...rotation,
        ...patch,
      },
    });

  return (
    <details
      className="config-disclosure track-cup-section"
      data-cup={X_CUP.id}
      open={!collapsed}
      onToggle={(event) => onCollapsedChange(X_CUP.id, !event.currentTarget.open)}
    >
      <summary className="config-disclosure-summary hover:border-app-border-strong">
        <span className="config-disclosure-title flex min-w-0 items-center gap-3">
          <TrackCupBanner cupId={X_CUP.id} label={X_CUP.label} />
          <div className="grid min-w-0 gap-1">
            <strong className="block text-sm">{X_CUP.label}</strong>
            <small className="text-xs font-medium text-app-muted">
              {xCupEnabled
                ? `${config.tracks.x_cup_course_count} generated`
                : xCupAvailable
                  ? "0 selected"
                  : "GP race only"}
            </small>
          </div>
        </span>
        <div className={cupSwitchGroupClass}>
          <AppTooltip content={xCupTooltip(xCupAvailable, xCupEnabled, selectedCourseIds)}>
            <span className="inline-flex">
              <button
                aria-label={`${xCupEnabled ? "Disable" : "Enable"} ${X_CUP.label}`}
                aria-pressed={xCupEnabled}
                className={xCupEnabled ? "switch-button active" : "switch-button"}
                disabled={!xCupAvailable || (xCupEnabled && selectedCourseIds.length === 0)}
                type="button"
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  updateTracks({ include_x_cup: !xCupEnabled });
                }}
              >
                <span aria-hidden="true" />
                <strong>{xCupEnabled ? "On" : "Off"}</strong>
              </button>
            </span>
          </AppTooltip>
        </div>
      </summary>
      <div className="config-disclosure-body">
        <div className="grid grid-cols-[repeat(auto-fit,minmax(164px,1fr))] gap-3">
          <div
            className={courseCardClass(xCupEnabled, "cursor-default min-h-0")}
            data-cup={X_CUP.id}
          >
            {xCupEnabled ? <span className={courseSelectedBadgeClass}>selected</span> : null}
            <div className="course-minimap">
              <TrackCupBanner cupId={X_CUP.id} label={X_CUP.label} large />
            </div>
            <div className="grid gap-1 text-left">
              <strong className="m-0 text-sm font-bold">Generated courses</strong>
              <span className="m-0 text-xs leading-snug text-app-muted">
                Deterministic GP X Cup baselines materialized at training start.
              </span>
            </div>
            <IntegerField
              help="Number of generated X Cup baselines materialized before training starts."
              label="Generated courses"
              min={1}
              resetValue={defaultConfig.tracks.x_cup_course_count}
              value={config.tracks.x_cup_course_count}
              onChange={(value) => updateTracks({ x_cup_course_count: value })}
            />
          </div>
          {xCupEnabled ? (
            <div className="grid gap-4 border border-app-border bg-app-surface/40 p-4">
              <div className="grid gap-1">
                <strong className="text-sm font-bold text-app-text">Auto regeneration</strong>
                <span className="text-xs leading-snug text-app-muted">
                  Replace a generated slot after it has enough samples and reaches the completion
                  threshold. Old inactive baselines are kept only as a small safety buffer.
                </span>
              </div>
              <div className="grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] gap-3">
                <BooleanField
                  help="When enabled, solved generated X Cup slots are replaced with fresh deterministic courses during training."
                  label="Auto regenerate"
                  resetValue={defaultRotation.enabled}
                  value={rotation.enabled}
                  onChange={(value) => updateRotation({ enabled: value })}
                />
                {rotation.enabled ? (
                  <>
                    <NumberField
                      help="EMA completion fraction required before replacing a generated slot."
                      label="Completion threshold"
                      resetValue={defaultRotation.completion_threshold}
                      step="0.01"
                      value={rotation.completion_threshold}
                      onChange={(value) => updateRotation({ completion_threshold: value })}
                    />
                    <IntegerField
                      help="Minimum sampled episodes before a generated slot can be replaced."
                      label="Min episodes"
                      min={1}
                      resetValue={defaultRotation.min_episodes}
                      value={rotation.min_episodes}
                      onChange={(value) =>
                        updateRotation({
                          min_episodes: value,
                          max_episodes:
                            rotation.max_episodes !== null && rotation.max_episodes < value
                              ? value
                              : rotation.max_episodes,
                        })
                      }
                    />
                    <BooleanField
                      help="When enabled, regenerate a slot once this many sampled episodes is reached, even if the current generated course is still below the completion threshold."
                      label="Episode cap"
                      resetValue={defaultRotation.max_episodes !== null}
                      value={rotation.max_episodes !== null}
                      onChange={(value) =>
                        updateRotation({
                          max_episodes: value
                            ? (rotation.max_episodes ?? defaultMaxEpisodes)
                            : null,
                        })
                      }
                    />
                    {rotation.max_episodes !== null ? (
                      <IntegerField
                        help="Maximum sampled episodes before replacing a generated slot regardless of completion. Disable the cap to keep trying until the completion threshold is reached."
                        label="Max episodes"
                        min={rotation.min_episodes}
                        resetValue={defaultRotation.max_episodes ?? defaultMaxEpisodes}
                        value={rotation.max_episodes}
                        onChange={(value) => updateRotation({ max_episodes: value })}
                      />
                    ) : null}
                    <NumberField
                      help="EMA smoothing for replacement eligibility completion stats. Higher values react faster to the latest generated course."
                      label="EMA alpha"
                      resetValue={defaultRotation.ema_alpha}
                      step="0.01"
                      value={rotation.ema_alpha}
                      onChange={(value) => updateRotation({ ema_alpha: value })}
                    />
                  </>
                ) : null}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </details>
  );
}

function xCupTooltip(
  xCupAvailable: boolean,
  xCupEnabled: boolean,
  selectedCourseIds: readonly string[],
) {
  if (!xCupAvailable) {
    return "GP race only";
  }
  if (xCupEnabled && selectedCourseIds.length === 0) {
    return "Select a built-in course before disabling X Cup";
  }
  return "Generated GP courses";
}
