// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/TracksSection.tsx
import { useMemo } from "react";
import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import type { ConfigSetter } from "@/features/configurator/configurator/state";
import { DisclosureToolbar } from "@/features/configurator/DisclosureToolbar";
import { usePersistentCollapsedIds } from "@/features/configurator/disclosureState";
import { IntegerField, NumberField, SegmentedChoiceStrip } from "@/features/configurator/fields";
import { TrackCupBanner } from "@/features/configurator/sections/tracks/TrackCupBanner";
import { TrackMinimap } from "@/features/configurator/sections/tracks/TrackMinimap";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";

interface TracksSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  setConfig: ConfigSetter;
}

type BuiltInCourse = ConfigMetadata["built_in_courses"][number];
type GpDifficulty = NonNullable<ManagedRunConfig["tracks"]["gp_difficulty"]>;

const RACE_MODE_DESCRIPTIONS: Record<ManagedRunConfig["tracks"]["race_mode"], string> = {
  time_attack: "Single-course time-trial episodes.",
  gp_race: "Grand Prix race rules across the selected pool.",
};

const GP_DIFFICULTY_DESCRIPTIONS: Record<GpDifficulty, string> = {
  novice: "Lightest GP difficulty and the current default.",
  standard: "Standard GP AI and race pressure.",
  expert: "More aggressive GP field behavior.",
  master: "Highest GP difficulty tier.",
};

const TRACK_SAMPLING_DESCRIPTIONS: Record<ManagedRunConfig["tracks"]["sampling_mode"], string> = {
  equal: "Sample courses uniformly by episode.",
  step_balanced: "Bias toward courses with fewer recent frames.",
  adaptive_step_balanced: "Keep step balance, then tilt a bit toward lower-completion courses.",
};

export function TracksSection({ config, defaultConfig, metadata, setConfig }: TracksSectionProps) {
  const defaultGpDifficulty: GpDifficulty = defaultConfig.tracks.gp_difficulty ?? "novice";
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
  const collapsibleCupIds = useMemo(() => [...cups.map((cup) => cup.id), "x"], [cups]);
  const [collapsedCupIds, setCollapsedCupIds] = usePersistentCollapsedIds(
    "run-manager:tracks:cups",
    collapsibleCupIds,
  );
  const selectedCourseIds = config.tracks.selected_course_ids;
  const xCupEnabled = config.tracks.include_x_cup;
  const xCupAvailable = config.tracks.race_mode === "gp_race";
  const usesDynamicStepBalancing = config.tracks.sampling_mode !== "equal";
  const usesAdaptiveStepBalancing = config.tracks.sampling_mode === "adaptive_step_balanced";
  const selectedCourseSet = useMemo(() => new Set(selectedCourseIds), [selectedCourseIds]);
  const collapsedCupIdSet = useMemo(() => new Set(collapsedCupIds), [collapsedCupIds]);
  const selectedCupCount = cups.filter((cup) =>
    cup.courses.some((course) => selectedCourseSet.has(course.id)),
  ).length;
  const xCupCollapsed = collapsedCupIdSet.has("x");
  const selectionSummary = cups
    .map((cup) => {
      const selectedCount = cup.courses.filter((course) => selectedCourseSet.has(course.id)).length;
      return selectedCount > 0
        ? `${shortCupLabel(cup.label)} ${selectedCount}/${cup.courses.length}`
        : null;
    })
    .filter((value): value is string => value !== null);
  const normalizedTracks = (tracks: ManagedRunConfig["tracks"]) => {
    const nextTracks = { ...tracks };
    if (nextTracks.race_mode === "gp_race" && nextTracks.gp_difficulty == null) {
      nextTracks.gp_difficulty = defaultGpDifficulty;
    }
    if (nextTracks.race_mode !== "gp_race") {
      nextTracks.gp_difficulty = null;
      nextTracks.include_x_cup = false;
      if (nextTracks.selected_course_ids.length === 0) {
        nextTracks.selected_course_ids = defaultCourseIds;
      }
    }
    return nextTracks;
  };

  const updateTracks = (patch: Partial<ManagedRunConfig["tracks"]>) => {
    setConfig((currentConfig) => ({
      ...currentConfig,
      tracks: normalizedTracks({ ...currentConfig.tracks, ...patch }),
    }));
  };

  const samplingDefaults = {
    adaptive_step_balance_completion_weight:
      defaultConfig.tracks.adaptive_step_balance_completion_weight,
    adaptive_step_balance_target_completion:
      defaultConfig.tracks.adaptive_step_balance_target_completion,
    step_balance_ema_alpha: defaultConfig.tracks.step_balance_ema_alpha,
    step_balance_max_weight_scale: defaultConfig.tracks.step_balance_max_weight_scale,
    step_balance_update_episodes: defaultConfig.tracks.step_balance_update_episodes,
  } satisfies Pick<
    ManagedRunConfig["tracks"],
    | "adaptive_step_balance_completion_weight"
    | "adaptive_step_balance_target_completion"
    | "step_balance_ema_alpha"
    | "step_balance_max_weight_scale"
    | "step_balance_update_episodes"
  >;

  const orderedSelectedCourseIds = (nextIds: Iterable<string>) => {
    const nextSet = new Set(nextIds);
    return allCourseIds.filter((courseId) => nextSet.has(courseId));
  };

  const toggleCourse = (courseId: string) => {
    setConfig((currentConfig) => {
      const nextSet = new Set(currentConfig.tracks.selected_course_ids);
      if (nextSet.has(courseId)) {
        if (nextSet.size === 1 && !currentConfig.tracks.include_x_cup) {
          return currentConfig;
        }
        nextSet.delete(courseId);
      } else {
        nextSet.add(courseId);
      }
      const ordered = orderedSelectedCourseIds(nextSet);
      if (ordered.length === 0 && !currentConfig.tracks.include_x_cup) {
        return currentConfig;
      }
      return {
        ...currentConfig,
        tracks: normalizedTracks({
          ...currentConfig.tracks,
          selected_course_ids: ordered,
        }),
      };
    });
  };

  const toggleCup = (courseIds: readonly string[]) => {
    setConfig((currentConfig) => {
      const nextSet = new Set(currentConfig.tracks.selected_course_ids);
      const allSelected = courseIds.every((courseId) => nextSet.has(courseId));
      if (allSelected) {
        if (nextSet.size === courseIds.length && !currentConfig.tracks.include_x_cup) {
          return currentConfig;
        }
        for (const courseId of courseIds) {
          nextSet.delete(courseId);
        }
      } else {
        for (const courseId of courseIds) {
          nextSet.add(courseId);
        }
      }
      const ordered = orderedSelectedCourseIds(nextSet);
      if (ordered.length === 0 && !currentConfig.tracks.include_x_cup) {
        return currentConfig;
      }
      return {
        ...currentConfig,
        tracks: normalizedTracks({
          ...currentConfig.tracks,
          selected_course_ids: ordered,
        }),
      };
    });
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
      <div className="form-grid two track-panel-grid">
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
                  race_mode: option.value as ManagedRunConfig["tracks"]["race_mode"],
                }),
            }))}
          />
        </ConfigPanel>

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
      </div>

      <div className="form-grid track-panel-grid">
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
          <div className="track-choice-panel">
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
            {usesDynamicStepBalancing ? (
              <div className="track-sampling-grid">
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
                  <>
                    <NumberField
                      help="Caps the adaptive frame-budget boost for low-completion courses. Reset frequency is still converted from target frame share using recent episode length."
                      label="Max target scale"
                      resetValue={defaultConfig.tracks.step_balance_max_weight_scale}
                      step="0.1"
                      value={config.tracks.step_balance_max_weight_scale}
                      onChange={(value) => updateTracks({ step_balance_max_weight_scale: value })}
                    />
                    <NumberField
                      help="How strongly low completion raises a course's step-budget share."
                      label="Completion weight"
                      resetValue={defaultConfig.tracks.adaptive_step_balance_completion_weight}
                      step="0.05"
                      value={config.tracks.adaptive_step_balance_completion_weight}
                      onChange={(value) =>
                        updateTracks({ adaptive_step_balance_completion_weight: value })
                      }
                    />
                    <NumberField
                      help="Courses below this completion fraction get extra sampling pressure."
                      label="Target completion"
                      resetValue={defaultConfig.tracks.adaptive_step_balance_target_completion}
                      step="0.01"
                      value={config.tracks.adaptive_step_balance_target_completion}
                      onChange={(value) =>
                        updateTracks({ adaptive_step_balance_target_completion: value })
                      }
                    />
                  </>
                ) : null}
              </div>
            ) : null}
          </div>
        </ConfigPanel>
      </div>

      <ConfigPanel
        onReset={() =>
          updateTracks({
            include_x_cup: defaultConfig.tracks.include_x_cup,
            selected_course_ids: defaultCourseIds,
            x_cup_course_count: defaultConfig.tracks.x_cup_course_count,
          })
        }
        title="Course pool"
        wide
      >
        <div className="track-pool-shell">
          <div className="track-pool-summary">
            <div className="track-pool-metric">
              <span>Selected courses</span>
              <strong>
                {selectedCourseIds.length + (xCupEnabled ? config.tracks.x_cup_course_count : 0)}
              </strong>
            </div>
            <div className="track-pool-metric">
              <span>Cups covered</span>
              <strong>{selectedCupCount + (xCupEnabled ? 1 : 0)}</strong>
            </div>
            <div className="track-pool-fragments">
              {selectionSummary.map((summary) => (
                <span key={summary}>{summary}</span>
              ))}
              {xCupEnabled ? <span>X {config.tracks.x_cup_course_count}</span> : null}
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
              onCollapseAll={() => setCollapsedCupIds(collapsibleCupIds)}
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
                allCupCoursesSelected &&
                selectedCourseIds.length === cup.courses.length &&
                !xCupEnabled;
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
                        const isOnlySelected =
                          isSelected && selectedCourseIds.length === 1 && !xCupEnabled;
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
            <details
              className="config-disclosure track-cup-section"
              data-cup="x"
              open={!xCupCollapsed}
              onToggle={(event) => setCupCollapsed("x", !event.currentTarget.open)}
            >
              <summary className="config-disclosure-summary track-cup-summary">
                <span className="config-disclosure-title track-cup-summary-label">
                  <TrackCupBanner cupId="x" label="X Cup" />
                  <div className="track-cup-title-copy config-disclosure-copy">
                    <strong>X Cup</strong>
                    <small>
                      {xCupEnabled
                        ? `${config.tracks.x_cup_course_count} generated`
                        : xCupAvailable
                          ? "0 selected"
                          : "GP race only"}
                    </small>
                  </div>
                </span>
                <div className="track-cup-controls">
                  <button
                    aria-label={`${xCupEnabled ? "Disable" : "Enable"} X Cup`}
                    aria-pressed={xCupEnabled}
                    className={
                      xCupEnabled
                        ? "switch-button active tooltip-anchor"
                        : "switch-button tooltip-anchor"
                    }
                    data-tooltip={
                      !xCupAvailable
                        ? "GP race only"
                        : xCupEnabled && selectedCourseIds.length === 0
                          ? "Select a built-in course before disabling X Cup"
                          : "Generated GP courses"
                    }
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
                </div>
              </summary>
              <div className="config-disclosure-body">
                <div className="track-card-grid">
                  <div
                    className={
                      xCupEnabled
                        ? "course-card selected track-xcup-course-card"
                        : "course-card track-xcup-course-card"
                    }
                    data-cup="x"
                  >
                    <div className="course-minimap">
                      <TrackCupBanner cupId="x" label="X Cup" large />
                    </div>
                    <div className="course-card-copy">
                      <strong>Generated courses</strong>
                      <span>Deterministic GP X Cup baselines materialized at training start.</span>
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
          </div>
        </div>
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
