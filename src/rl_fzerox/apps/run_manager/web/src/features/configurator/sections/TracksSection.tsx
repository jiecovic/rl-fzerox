// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/TracksSection.tsx
import { useMemo } from "react";
import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentCollapsedIds } from "@/features/configurator/disclosureState";
import { SegmentedChoiceStrip } from "@/features/configurator/fields";
import { TrackCupBanner } from "@/features/configurator/sections/tracks/TrackCupBanner";
import { TrackMinimap } from "@/features/configurator/sections/tracks/TrackMinimap";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";

interface TracksSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  setConfig: (config: ManagedRunConfig) => void;
}

type BuiltInCourse = ConfigMetadata["built_in_courses"][number];

const TRACK_POOL_MODE_DESCRIPTIONS: Record<ManagedRunConfig["tracks"]["pool_mode"], string> = {
  built_in: "Fixed cup courses grouped by the original roster.",
  x_cup: "Use F-Zero X's random track generator instead of fixed courses.",
};

const RACE_MODE_DESCRIPTIONS: Record<ManagedRunConfig["tracks"]["race_mode"], string> = {
  time_attack: "Single-course time-trial episodes.",
  gp_race: "Grand Prix race rules across the selected pool.",
};

const TRACK_SAMPLING_DESCRIPTIONS: Record<ManagedRunConfig["tracks"]["sampling_mode"], string> = {
  equal: "Sample courses uniformly by episode.",
  step_balanced: "Bias toward courses with fewer recent frames.",
};

export function TracksSection({ config, defaultConfig, metadata, setConfig }: TracksSectionProps) {
  const allCourseIds = useMemo(
    () => metadata.built_in_courses.map((course) => course.id),
    [metadata.built_in_courses],
  );
  const defaultCourseIds = useMemo(() => {
    const selected = metadata.built_in_courses
      .filter((course) => course.default_selected)
      .map((course) => course.id);
    return selected.length > 0 ? selected : allCourseIds;
  }, [allCourseIds, metadata.built_in_courses]);
  const cups = useMemo(() => {
    const courseById = new Map(metadata.built_in_courses.map((course) => [course.id, course]));
    return metadata.track_cups
      .slice()
      .sort((left, right) => left.order - right.order)
      .map((cup) => ({
        ...cup,
        courses: cup.course_ids
          .map((courseId) => courseById.get(courseId))
          .filter((course): course is BuiltInCourse => course !== undefined)
          .sort((left, right) => left.course_index - right.course_index)
          .map((course, index) => ({ ...course, cup_order_index: index + 1 })),
      }))
      .filter((cup) => cup.courses.length > 0);
  }, [metadata.built_in_courses, metadata.track_cups]);
  const [collapsedCupIds, setCollapsedCupIds] = usePersistentCollapsedIds(
    "run-manager:tracks:cups",
    cups.map((cup) => cup.id),
  );
  const selectedCourseIds = config.tracks.selected_course_ids;
  const usingXCup = config.tracks.pool_mode === "x_cup";
  const selectedCourseSet = useMemo(() => new Set(selectedCourseIds), [selectedCourseIds]);
  const collapsedCupIdSet = useMemo(() => new Set(collapsedCupIds), [collapsedCupIds]);
  const selectedCupCount = cups.filter((cup) =>
    cup.courses.some((course) => selectedCourseSet.has(course.id)),
  ).length;
  const selectionSummary = cups
    .map((cup) => {
      const selectedCount = cup.courses.filter((course) => selectedCourseSet.has(course.id)).length;
      return selectedCount > 0
        ? `${shortCupLabel(cup.label)} ${selectedCount}/${cup.courses.length}`
        : null;
    })
    .filter((value): value is string => value !== null);
  const updateTracks = (patch: Partial<ManagedRunConfig["tracks"]>) => {
    const nextTracks = { ...config.tracks, ...patch };
    if (nextTracks.pool_mode === "x_cup") {
      nextTracks.race_mode = "gp_race";
    }
    setConfig({ ...config, tracks: nextTracks });
  };

  const setSelectedCourses = (nextIds: Iterable<string>) => {
    const nextSet = new Set(nextIds);
    const ordered = allCourseIds.filter((courseId) => nextSet.has(courseId));
    if (ordered.length === 0) {
      return;
    }
    updateTracks({ selected_course_ids: ordered });
  };

  const toggleCourse = (courseId: string) => {
    const nextSet = new Set(selectedCourseSet);
    if (nextSet.has(courseId)) {
      if (nextSet.size === 1) {
        return;
      }
      nextSet.delete(courseId);
    } else {
      nextSet.add(courseId);
    }
    setSelectedCourses(nextSet);
  };

  const toggleCup = (courseIds: readonly string[]) => {
    const nextSet = new Set(selectedCourseSet);
    const allSelected = courseIds.every((courseId) => nextSet.has(courseId));
    if (allSelected) {
      if (nextSet.size === courseIds.length) {
        return;
      }
      for (const courseId of courseIds) {
        nextSet.delete(courseId);
      }
    } else {
      for (const courseId of courseIds) {
        nextSet.add(courseId);
      }
    }
    setSelectedCourses(nextSet);
  };

  const setCupCollapsed = (cupId: string, collapsed: boolean) => {
    setCollapsedCupIds((currentIds) =>
      collapsed
        ? currentIds.includes(cupId)
          ? currentIds
          : [...currentIds, cupId]
        : currentIds.filter((currentId) => currentId !== cupId),
    );
  };

  return (
    <div className="config-stack">
      <div className="form-grid three track-panel-grid">
        <ConfigPanel
          onReset={() => updateTracks({ pool_mode: defaultConfig.tracks.pool_mode })}
          title="Track source"
        >
          <ChoiceStrip
            description={
              TRACK_POOL_MODE_DESCRIPTIONS[config.tracks.pool_mode] ??
              TRACK_POOL_MODE_DESCRIPTIONS.built_in
            }
            options={metadata.track_pool_modes.map((option) => ({
              active: config.tracks.pool_mode === option.value,
              key: option.value,
              label: option.label,
              onClick: () =>
                updateTracks({
                  pool_mode: option.value as ManagedRunConfig["tracks"]["pool_mode"],
                }),
            }))}
          />
        </ConfigPanel>

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
              disabled: usingXCup && option.value !== "gp_race",
              key: option.value,
              label: formatTrackOptionLabel(option.value),
              tooltip:
                usingXCup && option.value !== "gp_race"
                  ? "X Cup is currently only supported under GP race."
                  : undefined,
              onClick: () =>
                updateTracks({
                  race_mode: option.value as ManagedRunConfig["tracks"]["race_mode"],
                }),
            }))}
          />
        </ConfigPanel>

        <ConfigPanel
          onReset={() => updateTracks({ sampling_mode: defaultConfig.tracks.sampling_mode })}
          title="Course sampling"
        >
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
                  sampling_mode: option.value as ManagedRunConfig["tracks"]["sampling_mode"],
                }),
            }))}
          />
        </ConfigPanel>
      </div>

      <ConfigPanel
        onReset={() => updateTracks({ selected_course_ids: defaultCourseIds })}
        title="Course pool"
        wide
      >
        {usingXCup ? (
          <div className="track-xcup-shell">
            <div className="track-xcup-card">
              <TrackCupBanner cupId="x" label="X Cup" large />
              <div className="track-xcup-copy">
                <strong>X Cup random generator</strong>
                <span>
                  The game generates six courses at runtime, so there is no fixed roster to toggle
                  here.
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="track-pool-shell">
            <div className="track-pool-summary">
              <div className="track-pool-metric">
                <span>Selected courses</span>
                <strong>{selectedCourseIds.length}</strong>
              </div>
              <div className="track-pool-metric">
                <span>Cups covered</span>
                <strong>{selectedCupCount}</strong>
              </div>
              <div className="track-pool-fragments">
                {selectionSummary.map((summary) => (
                  <span key={summary}>{summary}</span>
                ))}
              </div>
            </div>

            <div className="section-toolbar-row">
              <div className="track-pool-actions">
                <button
                  className="secondary-button"
                  disabled={selectedCourseIds.length === allCourseIds.length}
                  type="button"
                  onClick={() => updateTracks({ selected_course_ids: allCourseIds })}
                >
                  Select all
                </button>
                <button
                  className="secondary-button"
                  disabled={arraysEqual(selectedCourseIds, defaultCourseIds)}
                  type="button"
                  onClick={() => updateTracks({ selected_course_ids: defaultCourseIds })}
                >
                  Restore defaults
                </button>
              </div>
              <DisclosureToolbar
                collapseLabel="Collapse all cups"
                expandLabel="Expand all cups"
                onCollapseAll={() => setCollapsedCupIds(cups.map((cup) => cup.id))}
                onExpandAll={() => setCollapsedCupIds([])}
              />
            </div>

            <div className="track-cup-stack">
              {cups.map((cup) => {
                const selectedCount = cup.courses.filter((course) =>
                  selectedCourseSet.has(course.id),
                ).length;
                const allCupCoursesSelected = selectedCount === cup.courses.length;
                const cupCollapsed = collapsedCupIdSet.has(cup.id);
                const disablingWouldClearPool =
                  allCupCoursesSelected && selectedCourseIds.length === cup.courses.length;
                return (
                  <details
                    className="config-disclosure track-cup-section"
                    data-cup={cup.id}
                    key={cup.id}
                    open={!cupCollapsed}
                    onToggle={(event) => setCupCollapsed(cup.id, !event.currentTarget.open)}
                  >
                    <summary className="config-disclosure-summary track-cup-summary">
                      <span className="config-disclosure-title track-cup-summary-label">
                        <TrackCupBanner cupId={cup.id} label={cup.label} />
                        <div className="track-cup-title-copy config-disclosure-copy">
                          <strong>{cup.label}</strong>
                          <small>{selectedCount} selected</small>
                        </div>
                      </span>
                      <div className="track-cup-controls">
                        <button
                          aria-label={`${allCupCoursesSelected ? "Disable" : "Enable"} ${cup.label}`}
                          aria-pressed={allCupCoursesSelected}
                          className={
                            allCupCoursesSelected
                              ? "switch-button active tooltip-anchor"
                              : "switch-button tooltip-anchor"
                          }
                          data-tooltip={allCupCoursesSelected ? "Included" : "Excluded"}
                          disabled={disablingWouldClearPool}
                          type="button"
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            toggleCup(cup.course_ids);
                          }}
                        >
                          <span aria-hidden="true" />
                          <strong>{allCupCoursesSelected ? "On" : "Off"}</strong>
                        </button>
                      </div>
                    </summary>
                    <div className="config-disclosure-body">
                      <div className="track-card-grid">
                        {cup.courses.map((course) => {
                          const isSelected = selectedCourseSet.has(course.id);
                          const isOnlySelected = isSelected && selectedCourseIds.length === 1;
                          return (
                            <button
                              aria-label={course.display_name}
                              aria-pressed={isSelected}
                              className={isSelected ? "course-card selected" : "course-card"}
                              data-cup={cup.id}
                              disabled={isOnlySelected}
                              key={course.id}
                              type="button"
                              onClick={() => toggleCourse(course.id)}
                            >
                              <TrackMinimap courseId={course.id} cup={course.cup} />
                              <div className="course-card-copy">
                                <strong>{course.display_name}</strong>
                                <span>
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
              })}
            </div>
          </div>
        )}
      </ConfigPanel>
    </div>
  );
}

function ChoiceStrip({
  description,
  options,
}: {
  description: string;
  options: readonly {
    active: boolean;
    disabled?: boolean;
    key: string;
    label: string;
    tooltip?: string;
    onClick: () => void;
  }[];
}) {
  return (
    <div className="track-choice-panel">
      <SegmentedChoiceStrip ariaLabel="Selection" options={options} />
      <p className="track-choice-note">{description}</p>
    </div>
  );
}

function arraysEqual(left: readonly string[], right: readonly string[]) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function formatTrackOptionLabel(value: string) {
  return value
    .split("_")
    .map((word) => (word === "gp" ? "GP" : word.charAt(0).toUpperCase() + word.slice(1)))
    .join(" ");
}

function shortCupLabel(label: string) {
  return label.replace(" Cup", "");
}
