// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/CoursePoolPanel.tsx
import type { Dispatch, SetStateAction } from "react";

import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { BuiltInCupSection } from "@/features/configurator/sections/tracks/BuiltInCupSection";
import { arraysEqual } from "@/features/configurator/sections/tracks/coursePoolModel";
import { shortCupLabel, X_CUP } from "@/features/configurator/sections/tracks/options";
import type {
  TrackCupView,
  TracksConfig,
  TrackUpdate,
} from "@/features/configurator/sections/tracks/types";
import { XCupSection } from "@/features/configurator/sections/tracks/XCupSection";
import type { ManagedRunConfig } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";

interface CoursePoolPanelProps {
  allCourseIds: readonly string[];
  collapsibleCupIds: readonly string[];
  collapsedCupIdSet: ReadonlySet<string>;
  config: ManagedRunConfig;
  cups: readonly TrackCupView[];
  defaultConfig: ManagedRunConfig;
  defaultCourseIds: readonly string[];
  selectedCourseIds: readonly string[];
  selectedCourseSet: ReadonlySet<string>;
  updateTracks: TrackUpdate;
  setCollapsedCupIds: Dispatch<SetStateAction<readonly string[]>>;
  setCupCollapsed: (cupId: string, collapsed: boolean) => void;
  toggleCourse: (courseId: string) => void;
  toggleCup: (courseIds: readonly string[]) => void;
}

export function CoursePoolPanel({
  allCourseIds,
  collapsibleCupIds,
  collapsedCupIdSet,
  config,
  cups,
  defaultConfig,
  defaultCourseIds,
  selectedCourseIds,
  selectedCourseSet,
  updateTracks,
  setCollapsedCupIds,
  setCupCollapsed,
  toggleCourse,
  toggleCup,
}: CoursePoolPanelProps) {
  const xCupEnabled = config.tracks.include_x_cup;
  const xCupAvailable = config.tracks.race_mode === "gp_race";
  const selectedCupCount = cups.filter((cup) =>
    cup.courses.some((course) => selectedCourseSet.has(course.id)),
  ).length;
  const xCupCollapsed = collapsedCupIdSet.has(X_CUP.id);
  const selectionSummary = cups
    .map((cup) => {
      const selectedCount = cup.courses.filter((course) => selectedCourseSet.has(course.id)).length;
      return selectedCount > 0
        ? `${shortCupLabel(cup.label)} ${selectedCount}/${cup.courses.length}`
        : null;
    })
    .filter((value): value is string => value !== null);

  return (
    <ConfigPanel
      onReset={() =>
        updateTracks({
          include_x_cup: defaultConfig.tracks.include_x_cup,
          selected_course_ids: [...defaultCourseIds],
          x_cup_course_count: defaultConfig.tracks.x_cup_course_count,
        })
      }
      title="Course pool"
      wide
    >
      <div className="grid gap-3.5">
        <CoursePoolSummary
          selectedCourseIds={selectedCourseIds}
          selectedCupCount={selectedCupCount}
          selectionSummary={selectionSummary}
          tracks={config.tracks}
          xCupEnabled={xCupEnabled}
        />

        <div className="section-toolbar-row">
          <div className="flex flex-wrap gap-2">
            <Button
              className="h-9 px-3"
              disabled={selectedCourseIds.length === allCourseIds.length}
              type="button"
              onClick={() => updateTracks({ selected_course_ids: [...allCourseIds] })}
            >
              Select all
            </Button>
            <Button
              className="h-9 px-3"
              disabled={arraysEqual(selectedCourseIds, defaultCourseIds)}
              type="button"
              onClick={() => updateTracks({ selected_course_ids: [...defaultCourseIds] })}
            >
              Restore defaults
            </Button>
          </div>
          <DisclosureToolbar
            collapseLabel="Collapse all cups"
            expandLabel="Expand all cups"
            onCollapseAll={() => setCollapsedCupIds([...collapsibleCupIds])}
            onExpandAll={() => setCollapsedCupIds([])}
          />
        </div>

        <div className="grid gap-3.5">
          {cups.map((cup) => (
            <BuiltInCupSection
              collapsed={collapsedCupIdSet.has(cup.id)}
              cup={cup}
              key={cup.id}
              selectedCourseIds={selectedCourseIds}
              selectedCourseSet={selectedCourseSet}
              xCupEnabled={xCupEnabled}
              onCollapsedChange={setCupCollapsed}
              onToggleCourse={toggleCourse}
              onToggleCup={toggleCup}
            />
          ))}
          <XCupSection
            collapsed={xCupCollapsed}
            config={config}
            defaultConfig={defaultConfig}
            selectedCourseIds={selectedCourseIds}
            xCupAvailable={xCupAvailable}
            xCupEnabled={xCupEnabled}
            onCollapsedChange={setCupCollapsed}
            updateTracks={updateTracks}
          />
        </div>
      </div>
    </ConfigPanel>
  );
}

function CoursePoolSummary({
  selectedCourseIds,
  selectedCupCount,
  selectionSummary,
  tracks,
  xCupEnabled,
}: {
  selectedCourseIds: readonly string[];
  selectedCupCount: number;
  selectionSummary: readonly string[];
  tracks: TracksConfig;
  xCupEnabled: boolean;
}) {
  return (
    <div className="grid grid-cols-[140px_120px_minmax(0,1fr)] gap-3 border border-app-border bg-app-surface p-3.5">
      <div className="grid gap-1">
        <span className="text-xs leading-snug text-app-muted">Selected courses</span>
        <strong className="text-lg tabular-nums">
          {selectedCourseIds.length + (xCupEnabled ? tracks.x_cup_course_count : 0)}
        </strong>
      </div>
      <div className="grid gap-1">
        <span className="text-xs leading-snug text-app-muted">Cups covered</span>
        <strong className="text-lg tabular-nums">{selectedCupCount + (xCupEnabled ? 1 : 0)}</strong>
      </div>
      <div className="flex flex-wrap content-center gap-2">
        {selectionSummary.map((summary) => (
          <span
            className="border border-app-border bg-app-surface-muted px-2 py-1 text-xs leading-snug text-app-muted"
            key={summary}
          >
            {summary}
          </span>
        ))}
        {xCupEnabled ? (
          <span className="border border-app-border bg-app-surface-muted px-2 py-1 text-xs leading-snug text-app-muted">
            X {tracks.x_cup_course_count}
          </span>
        ) : null}
      </div>
    </div>
  );
}
