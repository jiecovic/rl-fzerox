// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/BuiltInCupSection.tsx

import {
  courseCardClass,
  cupSwitchGroupClass,
} from "@/features/configurator/sections/tracks/coursePoolStyle";
import { TrackCupBanner } from "@/features/configurator/sections/tracks/TrackCupBanner";
import { TrackMinimap } from "@/features/configurator/sections/tracks/TrackMinimap";
import type { TrackCupView } from "@/features/configurator/sections/tracks/types";
import { AppTooltip } from "@/shared/ui/Tooltip";

interface BuiltInCupSectionProps {
  cup: TrackCupView;
  collapsed: boolean;
  selectedCourseIds: readonly string[];
  selectedCourseSet: ReadonlySet<string>;
  xCupEnabled: boolean;
  onCollapsedChange: (cupId: string, collapsed: boolean) => void;
  onToggleCourse: (courseId: string) => void;
  onToggleCup: (courseIds: readonly string[]) => void;
}

export function BuiltInCupSection({
  cup,
  collapsed,
  selectedCourseIds,
  selectedCourseSet,
  xCupEnabled,
  onCollapsedChange,
  onToggleCourse,
  onToggleCup,
}: BuiltInCupSectionProps) {
  const selectedCount = cup.courses.filter((course) => selectedCourseSet.has(course.id)).length;
  const allCupCoursesSelected = selectedCount === cup.courses.length;
  const disablingWouldClearPool =
    allCupCoursesSelected && selectedCourseIds.length === cup.courses.length && !xCupEnabled;

  return (
    <details
      className="config-disclosure track-cup-section"
      data-cup={cup.id}
      open={!collapsed}
      onToggle={(event) => onCollapsedChange(cup.id, !event.currentTarget.open)}
    >
      <summary className="config-disclosure-summary hover:border-app-border-strong">
        <span className="config-disclosure-title flex min-w-0 items-center gap-3">
          <TrackCupBanner cupId={cup.id} label={cup.label} />
          <div className="grid min-w-0 gap-1">
            <strong className="block text-sm">{cup.label}</strong>
            <small className="text-xs font-medium text-app-muted">{selectedCount} selected</small>
          </div>
        </span>
        <div className={cupSwitchGroupClass}>
          <AppTooltip content={allCupCoursesSelected ? "Included" : "Excluded"}>
            <span className="inline-flex">
              <button
                aria-label={`${allCupCoursesSelected ? "Disable" : "Enable"} ${cup.label}`}
                aria-pressed={allCupCoursesSelected}
                className={allCupCoursesSelected ? "switch-button active" : "switch-button"}
                disabled={disablingWouldClearPool}
                type="button"
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  onToggleCup(cup.course_ids);
                }}
              >
                <span aria-hidden="true" />
                <strong>{allCupCoursesSelected ? "On" : "Off"}</strong>
              </button>
            </span>
          </AppTooltip>
        </div>
      </summary>
      <div className="config-disclosure-body">
        <div className="grid grid-cols-[repeat(auto-fit,minmax(164px,1fr))] gap-3">
          {cup.courses.map((course) => {
            const isSelected = selectedCourseSet.has(course.id);
            const isOnlySelected = isSelected && selectedCourseIds.length === 1 && !xCupEnabled;
            return (
              <button
                aria-label={course.display_name}
                aria-pressed={isSelected}
                className={courseCardClass(isSelected)}
                data-cup={cup.id}
                disabled={isOnlySelected}
                key={course.id}
                type="button"
                onClick={() => onToggleCourse(course.id)}
              >
                <TrackMinimap courseId={course.id} cup={course.cup} />
                <div className="grid gap-1 text-left">
                  <strong className="m-0 text-sm font-bold">{course.display_name}</strong>
                  <span className="m-0 text-xs leading-snug text-app-muted">
                    {cup.label} · Course {course.cup_order_index}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </details>
  );
}
