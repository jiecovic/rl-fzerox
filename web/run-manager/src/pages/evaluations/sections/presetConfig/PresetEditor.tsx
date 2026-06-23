// web/run-manager/src/pages/evaluations/sections/presetConfig/PresetEditor.tsx
import { type Dispatch, type SetStateAction, useMemo } from "react";

import { BuiltInCupSection } from "@/entities/runConfig/ui/sections/tracks/BuiltInCupSection";
import {
  arraysEqual,
  orderedCourseIds,
} from "@/entities/runConfig/ui/sections/tracks/coursePoolModel";
import { shortCupLabel } from "@/entities/runConfig/ui/sections/tracks/options";
import type { TrackCupView } from "@/entities/runConfig/ui/sections/tracks/types";
import {
  defaultGpDifficulties,
  selectGpDifficulty,
} from "@/pages/evaluations/sections/presetConfig/formModel";
import type {
  PresetEditorState,
  PresetFormState,
} from "@/pages/evaluations/sections/presetConfig/types";
import type {
  ConfigMetadata,
  EvaluationMode,
  ManagedEvaluationPreset,
} from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { cn } from "@/shared/ui/cn";
import { ConfigPanel } from "@/shared/ui/config/ConfigPanel";
import { DisclosureToolbar } from "@/shared/ui/config/DisclosureToolbar";
import { FieldInput, FieldSelect, FieldShell } from "@/shared/ui/Field";

export function PresetEditor({
  allCourseIds,
  collapsedCupIdSet,
  collapsibleCupIds,
  cups,
  defaultCourseIds,
  editor,
  error,
  isSaving,
  metadata,
  readOnly = false,
  subtitle,
  setCollapsedCupIds,
  setForm,
  onCancel,
  onCupCollapsed,
  onSave,
}: {
  allCourseIds: readonly string[];
  collapsedCupIdSet: ReadonlySet<string>;
  collapsibleCupIds: readonly string[];
  cups: readonly TrackCupView[];
  defaultCourseIds: readonly string[];
  editor: PresetEditorState;
  error: string | null;
  isSaving: boolean;
  metadata: ConfigMetadata;
  readOnly?: boolean;
  subtitle?: string;
  setCollapsedCupIds: Dispatch<SetStateAction<readonly string[]>>;
  setForm: (update: SetStateAction<PresetFormState>) => void;
  onCancel: () => void;
  onCupCollapsed: (cupId: string, collapsed: boolean) => void;
  onSave: () => void;
}) {
  const form = editor.form;
  const title =
    editor.mode === "view"
      ? form.name
      : editor.mode === "duplicate"
        ? "Duplicate preset"
        : "New preset";

  function setTargetMode(targetMode: EvaluationMode) {
    setForm((current) => ({
      ...current,
      difficulties:
        targetMode === "gp_course"
          ? current.difficulties.length === 1
            ? current.difficulties.slice(0, 1)
            : defaultGpDifficulties(metadata)
          : [],
      targetMode,
    }));
  }

  return (
    <section className="grid gap-4 border border-app-border-strong bg-app-surface p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="grid gap-1">
          <h3 className="m-0 text-base font-semibold text-app-text">{title}</h3>
          {subtitle === undefined ? null : (
            <span className="text-sm text-app-muted">{subtitle}</span>
          )}
        </div>
        {readOnly ? null : (
          <div className="flex flex-wrap gap-2">
            <Button disabled={isSaving} onClick={onCancel}>
              Cancel
            </Button>
            <Button disabled={isSaving} variant="primary" onClick={onSave}>
              {isSaving ? "Saving" : "Save preset"}
            </Button>
          </div>
        )}
      </div>

      <ConfigPanel title="Preset settings" wide>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <FieldShell className="sm:col-span-2 lg:col-span-2">
            <span>Name</span>
            <FieldInput
              disabled={readOnly}
              value={form.name}
              onChange={(event) => setForm({ ...form, name: event.currentTarget.value })}
            />
          </FieldShell>
          <FieldShell>
            <span>Mode</span>
            <FieldSelect
              disabled={readOnly}
              value={form.targetMode}
              onChange={(event) => setTargetMode(event.currentTarget.value as EvaluationMode)}
            >
              <option value="time_attack_course">Time Attack course</option>
              <option value="gp_course">GP course</option>
            </FieldSelect>
          </FieldShell>
          <FieldShell>
            <span>Renderer</span>
            <FieldSelect
              disabled={readOnly}
              value={form.renderer}
              onChange={(event) =>
                setForm({
                  ...form,
                  renderer: event.currentTarget.value as ManagedEvaluationPreset["renderer"],
                })
              }
            >
              <option value="gliden64">gliden64</option>
              <option value="angrylion">angrylion</option>
            </FieldSelect>
          </FieldShell>
          <FieldShell>
            <span>Seed</span>
            <FieldInput
              disabled={readOnly}
              inputMode="numeric"
              value={form.seed}
              onChange={(event) => setForm({ ...form, seed: event.currentTarget.value })}
            />
          </FieldShell>
          <FieldShell>
            <span>Repeats</span>
            <FieldInput
              disabled={readOnly}
              inputMode="numeric"
              value={form.repeatsPerTarget}
              onChange={(event) =>
                setForm({ ...form, repeatsPerTarget: event.currentTarget.value })
              }
            />
          </FieldShell>
        </div>
      </ConfigPanel>

      {form.targetMode === "gp_course" ? (
        <GpDifficultyEditor form={form} metadata={metadata} readOnly={readOnly} setForm={setForm} />
      ) : null}

      <PresetCoursePoolEditor
        allCourseIds={allCourseIds}
        collapsedCupIdSet={collapsedCupIdSet}
        collapsibleCupIds={collapsibleCupIds}
        cups={cups}
        defaultCourseIds={defaultCourseIds}
        form={form}
        readOnly={readOnly}
        setCollapsedCupIds={setCollapsedCupIds}
        setForm={setForm}
        onCupCollapsed={onCupCollapsed}
      />

      {error === null ? null : (
        <p className="m-0 text-sm text-app-danger" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

function GpDifficultyEditor({
  form,
  metadata,
  readOnly,
  setForm,
}: {
  form: PresetFormState;
  metadata: ConfigMetadata;
  readOnly: boolean;
  setForm: (update: SetStateAction<PresetFormState>) => void;
}) {
  const selectedDifficulties = new Set(form.difficulties);
  const selectedLabels = metadata.gp_difficulties
    .filter((option) => selectedDifficulties.has(option.value))
    .map((option) => option.label);

  return (
    <ConfigPanel
      onReset={
        readOnly
          ? undefined
          : () =>
              setForm((current) => ({
                ...current,
                difficulties: defaultGpDifficulties(metadata),
              }))
      }
      title="GP difficulty"
      wide
    >
      <div className="grid gap-2.5">
        <fieldset className="m-0 inline-flex w-fit max-w-full min-w-0 flex-wrap justify-self-start overflow-hidden rounded-lg border border-app-border bg-app-surface-muted p-0.5">
          <legend className="sr-only">GP difficulties</legend>
          {metadata.gp_difficulties.map((option) => {
            const active = selectedDifficulties.has(option.value);
            return (
              <button
                aria-pressed={active}
                className={cn(
                  "min-h-8 min-w-0 rounded-md border border-transparent px-3.5 text-sm font-medium whitespace-nowrap text-app-muted transition-colors",
                  active ? "border-app-border-strong bg-app-surface text-app-text" : undefined,
                  !active
                    ? "hover:bg-[color-mix(in_srgb,var(--accent)_8%,var(--surface-muted))] hover:text-app-text"
                    : undefined,
                )}
                key={option.value}
                disabled={readOnly}
                type="button"
                onClick={() =>
                  setForm((current) => ({
                    ...current,
                    difficulties: selectGpDifficulty(option.value, metadata),
                  }))
                }
              >
                {option.label}
              </button>
            );
          })}
        </fieldset>
        <p className="m-0 text-xs leading-snug text-app-muted">
          {selectedLabels.length === 0 ? "None selected" : selectedLabels.join(", ")}
        </p>
      </div>
    </ConfigPanel>
  );
}

function PresetCoursePoolEditor({
  allCourseIds,
  collapsedCupIdSet,
  collapsibleCupIds,
  cups,
  defaultCourseIds,
  form,
  readOnly,
  setCollapsedCupIds,
  setForm,
  onCupCollapsed,
}: {
  allCourseIds: readonly string[];
  collapsedCupIdSet: ReadonlySet<string>;
  collapsibleCupIds: readonly string[];
  cups: readonly TrackCupView[];
  defaultCourseIds: readonly string[];
  form: PresetFormState;
  readOnly: boolean;
  setCollapsedCupIds: Dispatch<SetStateAction<readonly string[]>>;
  setForm: (update: SetStateAction<PresetFormState>) => void;
  onCupCollapsed: (cupId: string, collapsed: boolean) => void;
}) {
  const selectedCourseSet = useMemo(() => new Set(form.courseIds), [form.courseIds]);
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

  function setCourseIds(courseIds: Iterable<string>) {
    const ordered = orderedCourseIds(courseIds, allCourseIds);
    if (ordered.length === 0) {
      return;
    }
    setForm((current) => ({ ...current, courseIds: ordered }));
  }

  function toggleCourse(courseId: string) {
    const nextIds = new Set(form.courseIds);
    if (nextIds.has(courseId)) {
      if (nextIds.size === 1) {
        return;
      }
      nextIds.delete(courseId);
    } else {
      nextIds.add(courseId);
    }
    setCourseIds(nextIds);
  }

  function toggleCup(courseIds: readonly string[]) {
    const nextIds = new Set(form.courseIds);
    const allSelected = courseIds.every((courseId) => nextIds.has(courseId));
    if (allSelected) {
      for (const courseId of courseIds) {
        nextIds.delete(courseId);
      }
    } else {
      for (const courseId of courseIds) {
        nextIds.add(courseId);
      }
    }
    setCourseIds(nextIds);
  }

  return (
    <ConfigPanel
      onReset={
        readOnly
          ? undefined
          : () => setForm((current) => ({ ...current, courseIds: [...defaultCourseIds] }))
      }
      title="Course pool"
      wide
    >
      <div className="grid gap-3.5">
        <CoursePoolSummary
          selectedCourseCount={form.courseIds.length}
          selectedCupCount={selectedCupCount}
          selectionSummary={selectionSummary}
        />

        <div className="section-toolbar-row">
          <div className="flex flex-wrap gap-2">
            <Button
              className="h-9 px-3"
              disabled={readOnly || form.courseIds.length === allCourseIds.length}
              type="button"
              onClick={() => setCourseIds(allCourseIds)}
            >
              Select all
            </Button>
            <Button
              className="h-9 px-3"
              disabled={readOnly || arraysEqual(form.courseIds, defaultCourseIds)}
              type="button"
              onClick={() => setCourseIds(defaultCourseIds)}
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
              selectedCourseIds={form.courseIds}
              selectedCourseSet={selectedCourseSet}
              readOnly={readOnly}
              xCupEnabled={false}
              onCollapsedChange={onCupCollapsed}
              onToggleCourse={toggleCourse}
              onToggleCup={toggleCup}
            />
          ))}
        </div>
      </div>
    </ConfigPanel>
  );
}

function CoursePoolSummary({
  selectedCourseCount,
  selectedCupCount,
  selectionSummary,
}: {
  selectedCourseCount: number;
  selectedCupCount: number;
  selectionSummary: readonly string[];
}) {
  return (
    <div className="grid grid-cols-[140px_120px_minmax(0,1fr)] gap-3 border border-app-border bg-app-surface p-3.5 max-sm:grid-cols-1">
      <div className="grid gap-1">
        <span className="text-xs leading-snug text-app-muted">Selected courses</span>
        <strong className="text-lg tabular-nums">{selectedCourseCount}</strong>
      </div>
      <div className="grid gap-1">
        <span className="text-xs leading-snug text-app-muted">Cups covered</span>
        <strong className="text-lg tabular-nums">{selectedCupCount}</strong>
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
      </div>
    </div>
  );
}
