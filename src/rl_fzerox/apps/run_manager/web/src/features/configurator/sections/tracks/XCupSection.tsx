// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/XCupSection.tsx
import { IntegerField } from "@/features/configurator/fields";
import {
  courseCardClass,
  courseSelectedBadgeClass,
  cupSwitchGroupClass,
} from "@/features/configurator/sections/tracks/coursePoolStyle";
import { X_CUP } from "@/features/configurator/sections/tracks/options";
import { TrackCupBanner } from "@/features/configurator/sections/tracks/TrackCupBanner";
import type { TrackUpdate } from "@/features/configurator/sections/tracks/types";
import type { ManagedRunConfig } from "@/shared/api/contract";
import { AppTooltip } from "@/shared/ui/Tooltip";

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
