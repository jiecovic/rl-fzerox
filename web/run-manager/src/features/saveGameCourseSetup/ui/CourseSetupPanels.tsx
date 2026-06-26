// web/run-manager/src/features/saveGameCourseSetup/ui/CourseSetupPanels.tsx
import { memo, useEffect, useMemo, useState } from "react";
import { courseCardClass } from "@/entities/runConfig/ui/sections/tracks/coursePoolStyle";
import { TrackCupBanner } from "@/entities/runConfig/ui/sections/tracks/TrackCupBanner";
import { TrackMinimap } from "@/entities/runConfig/ui/sections/tracks/TrackMinimap";
import type {
  CourseSetupDraft,
  CourseSetupDraftMap,
  CourseSetupValues,
  CupSetupDraft,
  CupSetupDraftMap,
  CupSetupValues,
  CupView,
  PolicyArtifactDraft,
  PolicySelectionDraft,
  PolicySourceOption,
} from "@/features/saveGameCourseSetup/model/courseSetup";
import {
  courseSetupKey,
  courseSetupsForCups,
  courseSetupValues,
  cupSetupKey,
  cupSetupValues,
  dirtyCourseSetupDrafts,
  dirtyCupSetupDrafts,
  EMPTY_COURSE_SETUP_DRAFT,
  EMPTY_CUP_SETUP_DRAFT,
  exactCourseSetupDraft,
  selectedPolicySource,
  sharedPolicySelectionDraft,
} from "@/features/saveGameCourseSetup/model/courseSetup";
import {
  EngineDraftInput,
  PolicyArtifactSelect,
  PolicyDraftSelect,
  PolicySelectionSelect,
  VehicleDraftSelect,
} from "@/features/saveGameCourseSetup/ui/CourseSetupFields";
import type { ConfigMetadata } from "@/shared/api/contract";
import { clampRawEngineValue } from "@/shared/domain/engineBuckets";
import { Button } from "@/shared/ui/Button";
import { cn } from "@/shared/ui/cn";
import { DisclosureToolbar } from "@/shared/ui/config/DisclosureToolbar";
import { ChevronIcon, ImportIcon, ResetIcon, SaveDraftIcon } from "@/shared/ui/icons";

const EMPTY_POLICY_SELECTION_DRAFT: PolicySelectionDraft = {
  policyArtifact: "best",
  policySourceId: "",
  policySourceKind: "run",
};

export interface CourseSetupSaveScope {
  courseSetups: readonly CourseSetupValues[];
  cupSetups: readonly CupSetupValues[];
}

export const GlobalPolicyPanel = memo(function GlobalPolicyPanel({
  cups,
  onImportEngineTuning,
  onApplySetups,
  policySources,
  updating,
}: {
  cups: readonly CupView[];
  onImportEngineTuning: (draft: PolicySelectionDraft) => Promise<void>;
  onApplySetups: (setups: readonly CourseSetupValues[], draft: PolicySelectionDraft) => void;
  policySources: readonly PolicySourceOption[];
  updating: boolean;
}) {
  const [draft, setDraft] = useState<PolicySelectionDraft>(EMPTY_POLICY_SELECTION_DRAFT);
  const [importing, setImporting] = useState(false);
  const selectedSource = useMemo(
    () => selectedPolicySource(policySources, draft),
    [draft, policySources],
  );
  const canImportEngines = !updating && selectedSource?.canImportEngineTuning === true;
  const canApply = !updating && draft.policySourceId !== "";
  const allCourseSetups = useMemo(() => courseSetupsForCups(cups), [cups]);

  async function importEngineTuning() {
    if (!canImportEngines) {
      return;
    }
    setImporting(true);
    try {
      await onImportEngineTuning(draft);
    } finally {
      setImporting(false);
    }
  }

  return (
    <div className="grid content-start gap-3 border border-app-border bg-app-surface-muted p-4">
      <div className="grid gap-1">
        <h4 className="m-0 text-sm font-bold tracking-[0.04em] text-app-muted uppercase">
          Bulk policy
        </h4>
        <p className="m-0 text-sm text-app-muted">
          Copy a trained policy artifact into course rows without changing engines.
        </p>
      </div>
      <div className="grid gap-3 lg:grid-cols-[minmax(18rem,1fr)_140px_auto] lg:items-end">
        <PolicySelectionSelect
          disabled={updating}
          draft={draft}
          label="Policy"
          policySources={policySources}
          onDraftChange={setDraft}
        />
        <PolicyArtifactSelect
          disabled={updating}
          draft={draft}
          label="Artifact"
          policySources={policySources}
          onDraftChange={setDraft}
        />
        <div className="flex flex-wrap gap-2 lg:justify-end">
          <Button
            className="min-w-[160px]"
            disabled={!canApply}
            type="button"
            onClick={() => onApplySetups(allCourseSetups, draft)}
          >
            Apply to all courses
          </Button>
          <Button
            className="min-w-[180px]"
            disabled={!canImportEngines || importing}
            type="button"
            onClick={() => void importEngineTuning()}
          >
            {importing ? "Importing engines" : "Import learned engines"}
          </Button>
        </div>
      </div>
    </div>
  );
});

export const CourseSetupPanel = memo(function CourseSetupPanel({
  cups,
  defaultExpandedCupId,
  dirtySetupCount,
  metadata,
  onCourseSetupDraftChange,
  onCupSetupDraftChange,
  onImportEngineTuning,
  onResetEngineSetups,
  onSaveSetups,
  policySources,
  courseSetupDrafts,
  savedCourseSetupDrafts,
  cupSetupDrafts,
  savedCupSetupDrafts,
  savingCourseSetups,
  updating,
  unlockedVehicleIds,
}: {
  cups: readonly CupView[];
  defaultExpandedCupId: string | null;
  dirtySetupCount: number;
  metadata: ConfigMetadata;
  onCourseSetupDraftChange: (values: CourseSetupValues, draft: PolicyArtifactDraft) => void;
  onCupSetupDraftChange: (values: CupSetupValues, vehicleId: string) => void;
  onImportEngineTuning: (
    draft: PolicySelectionDraft,
    scope?: CourseSetupSaveScope,
  ) => Promise<void>;
  onResetEngineSetups: (setups: readonly CourseSetupValues[]) => void;
  onSaveSetups: (scope?: CourseSetupSaveScope) => void;
  policySources: readonly PolicySourceOption[];
  courseSetupDrafts: CourseSetupDraftMap;
  savedCourseSetupDrafts: CourseSetupDraftMap;
  cupSetupDrafts: CupSetupDraftMap;
  savedCupSetupDrafts: CupSetupDraftMap;
  savingCourseSetups: boolean;
  updating: boolean;
  unlockedVehicleIds: readonly string[];
}) {
  const [collapsedCupIds, setCollapsedCupIds] = useState<readonly string[]>(() =>
    defaultCollapsedCupIds(cups, defaultExpandedCupId),
  );
  const collapsedCupIdSet = useMemo(() => new Set(collapsedCupIds), [collapsedCupIds]);
  const saveDisabled = dirtySetupCount === 0 || savingCourseSetups || updating;
  const allCourseSetups = useMemo(() => courseSetupsForCups(cups), [cups]);

  function setCupCollapsed(cupId: string, collapsed: boolean) {
    setCollapsedCupIds((current) =>
      collapsed
        ? current.includes(cupId)
          ? current
          : [...current, cupId]
        : current.filter((existingCupId) => existingCupId !== cupId),
    );
  }

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="grid gap-1">
          <h4 className="m-0 text-sm font-bold tracking-[0.04em] text-app-muted uppercase">
            Course setup
          </h4>
          <p className="m-0 text-sm text-app-muted">
            Use cup controls as shortcuts, then adjust individual course policy or engine values.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <DisclosureToolbar
            collapseLabel="Collapse all course setup cups"
            expandLabel="Expand all course setup cups"
            onCollapseAll={() => setCollapsedCupIds(cups.map((cup) => cup.id))}
            onExpandAll={() => setCollapsedCupIds([])}
          />
          <ResetEngineSetupsButton
            disabled={updating || savingCourseSetups}
            label="Reset engines to default"
            setups={allCourseSetups}
            onResetEngineSetups={onResetEngineSetups}
          />
          <Button
            className="gap-2"
            disabled={saveDisabled}
            type="button"
            variant={dirtySetupCount > 0 ? "primary" : "secondary"}
            onClick={() => onSaveSetups()}
          >
            <SaveDraftIcon />
            <span>
              {savingCourseSetups
                ? "Saving"
                : dirtySetupCount > 0
                  ? `Save ${dirtySetupCount} change${dirtySetupCount === 1 ? "" : "s"}`
                  : "Saved"}
            </span>
          </Button>
        </div>
      </div>
      <div className="grid gap-3">
        {cups.map((cup) => (
          <CupSetupBlock
            key={cup.id}
            collapsed={collapsedCupIdSet.has(cup.id)}
            cup={cup}
            courseSetupDrafts={courseSetupDrafts}
            cupSetupDrafts={cupSetupDrafts}
            metadata={metadata}
            savedCourseSetupDrafts={savedCourseSetupDrafts}
            savedCupSetupDrafts={savedCupSetupDrafts}
            updating={updating || savingCourseSetups}
            unlockedVehicleIds={unlockedVehicleIds}
            onCollapsedChange={setCupCollapsed}
            onCourseSetupDraftChange={onCourseSetupDraftChange}
            onCupSetupDraftChange={onCupSetupDraftChange}
            onImportEngineTuning={onImportEngineTuning}
            onResetEngineSetups={onResetEngineSetups}
            onSaveSetups={onSaveSetups}
            policySources={policySources}
          />
        ))}
      </div>
    </div>
  );
});

function defaultCollapsedCupIds(
  cups: readonly CupView[],
  defaultExpandedCupId: string | null,
): readonly string[] {
  return cups.filter((cup) => cup.id !== defaultExpandedCupId).map((cup) => cup.id);
}

const CupSetupBlock = memo(function CupSetupBlock({
  collapsed,
  cup,
  cupSetupDrafts,
  metadata,
  onCollapsedChange,
  onCourseSetupDraftChange,
  onCupSetupDraftChange,
  onImportEngineTuning,
  onResetEngineSetups,
  onSaveSetups,
  policySources,
  courseSetupDrafts,
  savedCourseSetupDrafts,
  savedCupSetupDrafts,
  updating,
  unlockedVehicleIds,
}: {
  collapsed: boolean;
  cup: CupView;
  cupSetupDrafts: CupSetupDraftMap;
  metadata: ConfigMetadata;
  onCollapsedChange: (cupId: string, collapsed: boolean) => void;
  onCourseSetupDraftChange: (values: CourseSetupValues, draft: PolicyArtifactDraft) => void;
  onCupSetupDraftChange: (values: CupSetupValues, vehicleId: string) => void;
  onImportEngineTuning: (
    draft: PolicySelectionDraft,
    scope?: CourseSetupSaveScope,
  ) => Promise<void>;
  onResetEngineSetups: (setups: readonly CourseSetupValues[]) => void;
  onSaveSetups: (scope?: CourseSetupSaveScope) => void;
  policySources: readonly PolicySourceOption[];
  courseSetupDrafts: CourseSetupDraftMap;
  savedCourseSetupDrafts: CourseSetupDraftMap;
  savedCupSetupDrafts: CupSetupDraftMap;
  updating: boolean;
  unlockedVehicleIds: readonly string[];
}) {
  const cupValues = cupSetupValues(cup);
  const cupDraft = cupSetupDraft(cupSetupDrafts, cupValues);
  const courseValuesList = useMemo(
    () => cup.courses.map((course) => courseSetupValues(cup, course.id)),
    [cup],
  );
  const saveScope = useMemo(
    () => ({ courseSetups: courseValuesList, cupSetups: [cupValues] }),
    [courseValuesList, cupValues],
  );
  const [importingEngines, setImportingEngines] = useState(false);
  const cupDirtyCount = useMemo(() => {
    const dirtyCourseCount = dirtyCourseSetupDrafts(
      courseSetupDrafts,
      savedCourseSetupDrafts,
    ).filter((draft) => draft.cupId === cup.id).length;
    const dirtyCupCount = dirtyCupSetupDrafts(cupSetupDrafts, savedCupSetupDrafts).filter(
      (draft) => draft.cupId === cup.id,
    ).length;
    return dirtyCourseCount + dirtyCupCount;
  }, [courseSetupDrafts, cup.id, cupSetupDrafts, savedCourseSetupDrafts, savedCupSetupDrafts]);
  const courseDrafts = useMemo(
    () =>
      courseValuesList
        .map((values) => exactCourseSetupDraft(courseSetupDrafts, values))
        .filter((draft): draft is CourseSetupDraft => draft !== null),
    [courseSetupDrafts, courseValuesList],
  );
  const fallbackDraft = useMemo<PolicyArtifactDraft>(
    () => ({
      ...EMPTY_COURSE_SETUP_DRAFT,
      vehicleId: cupDraft.vehicleId,
    }),
    [cupDraft.vehicleId],
  );
  const bulkPolicyDraft = useMemo<PolicySelectionDraft>(
    () =>
      sharedPolicySelectionDraft(courseDrafts, courseValuesList.length) ??
      EMPTY_POLICY_SELECTION_DRAFT,
    [courseDrafts, courseValuesList.length],
  );
  const [selectedCourseKey, setSelectedCourseKey] = useState<string | null>(null);
  const selectedSource = useMemo(
    () => selectedPolicySource(policySources, bulkPolicyDraft),
    [bulkPolicyDraft, policySources],
  );
  const canImportEngines = !updating && selectedSource?.canImportEngineTuning === true;

  function applyBulkCoursePolicy(nextDraft: PolicySelectionDraft) {
    for (const values of courseValuesList) {
      const currentDraft = exactCourseSetupDraft(courseSetupDrafts, values) ?? fallbackDraft;
      onCourseSetupDraftChange(values, {
        ...currentDraft,
        policyArtifact: nextDraft.policyArtifact,
        policySourceId: nextDraft.policySourceId,
        policySourceKind: nextDraft.policySourceKind,
        vehicleId: cupDraft.vehicleId,
      });
    }
  }

  useEffect(() => {
    if (selectedCourseKey === null) {
      return undefined;
    }

    function adjustSelectedCourseEngine(delta: number) {
      const selectedValues = courseValuesList.find(
        (values) => courseSetupKey(values) === selectedCourseKey,
      );
      if (selectedValues === undefined) {
        return;
      }
      const currentDraft = {
        ...(exactCourseSetupDraft(courseSetupDrafts, selectedValues) ?? fallbackDraft),
        vehicleId: cupDraft.vehicleId,
      };
      onCourseSetupDraftChange(selectedValues, {
        ...currentDraft,
        engineSettingRawValue: clampRawEngineValue(currentDraft.engineSettingRawValue + delta),
      });
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (isFormControlTarget(event.target) || updating) {
        return;
      }
      if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
        event.preventDefault();
        adjustSelectedCourseEngine(-1);
      }
      if (event.key === "ArrowRight" || event.key === "ArrowUp") {
        event.preventDefault();
        adjustSelectedCourseEngine(1);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    courseSetupDrafts,
    courseValuesList,
    cupDraft.vehicleId,
    fallbackDraft,
    onCourseSetupDraftChange,
    selectedCourseKey,
    updating,
  ]);

  return (
    <details
      className="config-disclosure track-cup-section bg-app-surface-muted"
      data-cup={cup.id}
      open={!collapsed}
      onToggle={(event) => onCollapsedChange(cup.id, !event.currentTarget.open)}
    >
      <summary className="config-disclosure-summary hover:border-app-border-strong">
        <span className="config-disclosure-title flex min-w-0 items-center gap-3">
          <TrackCupBanner cupId={cup.id} label={cup.label} />
          <span className="min-w-0">
            <strong className="block truncate text-base text-app-text">{cup.label}</strong>
          </span>
        </span>
      </summary>
      {collapsed ? null : (
        <div className="config-disclosure-body gap-3">
          <div className="grid gap-3 xl:grid-cols-[minmax(220px,1fr)_120px_minmax(170px,220px)_auto] xl:items-end">
            <PolicySelectionSelect
              disabled={updating}
              draft={bulkPolicyDraft}
              label={`${cup.label} policy`}
              policySources={policySources}
              visibleLabel="Policy"
              onDraftChange={applyBulkCoursePolicy}
            />
            <PolicyArtifactSelect
              disabled={updating}
              draft={bulkPolicyDraft}
              label={`${cup.label} artifact`}
              policySources={policySources}
              visibleLabel="Artifact"
              onDraftChange={applyBulkCoursePolicy}
            />
            <VehicleDraftSelect
              disabled={updating}
              draft={cupDraft}
              label={`${cup.label} vehicle`}
              metadata={metadata}
              unlockedVehicleIds={unlockedVehicleIds}
              visibleLabel="Vehicle"
              onDraftChange={(nextDraft) => onCupSetupDraftChange(cupValues, nextDraft.vehicleId)}
            />
            <div className="flex flex-wrap gap-2 xl:flex-nowrap xl:justify-end">
              <Button
                className="min-w-[96px] gap-2 px-3"
                disabled={updating || bulkPolicyDraft.policySourceId === ""}
                type="button"
                onClick={() => applyBulkCoursePolicy(bulkPolicyDraft)}
              >
                <ChevronIcon />
                <span>Apply</span>
              </Button>
              <Button
                aria-label={`Save ${cup.label} setup`}
                className="min-w-[92px] gap-2 px-3"
                disabled={updating || cupDirtyCount === 0}
                type="button"
                variant={cupDirtyCount > 0 ? "primary" : "secondary"}
                onClick={() => onSaveSetups(saveScope)}
              >
                <SaveDraftIcon />
                <span>Save</span>
              </Button>
              <Button
                className="min-w-[142px] gap-2 px-3"
                disabled={!canImportEngines || importingEngines}
                type="button"
                onClick={async () => {
                  setImportingEngines(true);
                  try {
                    await onImportEngineTuning(bulkPolicyDraft, saveScope);
                  } finally {
                    setImportingEngines(false);
                  }
                }}
              >
                <ImportIcon />
                <span>{importingEngines ? "Importing" : "Import engines"}</span>
              </Button>
              <ResetEngineSetupsButton
                disabled={updating}
                label="Reset engines to default"
                className="px-3"
                setups={courseValuesList}
                onResetEngineSetups={onResetEngineSetups}
              />
            </div>
          </div>
          <div className="grid grid-cols-1 gap-3 border-t border-app-border pt-3 xl:grid-cols-2 2xl:grid-cols-3">
            {cup.courses.map((course, courseIndex) => {
              const values = courseSetupValues(cup, course.id);
              const courseDraft = {
                ...(exactCourseSetupDraft(courseSetupDrafts, values) ?? fallbackDraft),
                vehicleId: cupDraft.vehicleId,
              };
              const updateCourseDraft = (nextDraft: PolicyArtifactDraft) =>
                onCourseSetupDraftChange(values, {
                  ...nextDraft,
                  vehicleId: cupDraft.vehicleId,
                });
              const currentCourseKey = courseSetupKey(values);
              return (
                <div
                  key={course.id}
                  className={courseCardClass(
                    selectedCourseKey === currentCourseKey,
                    cn(
                      "min-h-[172px] content-start gap-3 p-3",
                      selectedCourseKey === currentCourseKey &&
                        "shadow-[inset_0_0_0_1px_var(--accent)]",
                    ),
                  )}
                  data-cup={cup.id}
                  onPointerDown={(event) => {
                    if (isFormControlTarget(event.target)) {
                      return;
                    }
                    setSelectedCourseKey(currentCourseKey);
                  }}
                >
                  <div className="grid min-w-0 grid-cols-[104px_minmax(0,1fr)] items-start gap-3">
                    <TrackMinimap
                      className="h-[72px] w-[104px] self-start"
                      courseId={course.id}
                      cup={course.cup}
                    />
                    <div className="min-w-0 text-left">
                      <div className="flex min-w-0 items-center gap-2">
                        <span className="grid h-5 w-5 shrink-0 place-items-center border border-app-border bg-app-surface font-mono text-[11px] text-app-muted tabular-nums">
                          {courseIndex + 1}
                        </span>
                        <strong className="block min-w-0 truncate text-sm font-bold text-app-text">
                          {course.display_name}
                        </strong>
                      </div>
                      <span className="text-xs text-app-muted">
                        Course {course.course_index + 1}
                      </span>
                    </div>
                  </div>
                  <div className="grid gap-3">
                    <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_112px]">
                      <PolicyDraftSelect
                        disabled={updating}
                        draft={courseDraft}
                        label={`${course.display_name} policy`}
                        lockedVehicleId={cupDraft.vehicleId}
                        metadata={metadata}
                        policySources={policySources}
                        unlockedVehicleIds={unlockedVehicleIds}
                        visibleLabel="Policy"
                        onDraftChange={updateCourseDraft}
                      />
                      <PolicyArtifactSelect
                        disabled={updating}
                        draft={courseDraft}
                        label={`${course.display_name} artifact`}
                        policySources={policySources}
                        visibleLabel="Artifact"
                        onDraftChange={updateCourseDraft}
                      />
                    </div>
                    <EngineDraftInput
                      disabled={updating}
                      draft={courseDraft}
                      label={`${course.display_name} engine`}
                      visibleLabel="Engine"
                      onDraftChange={updateCourseDraft}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </details>
  );
});

function cupSetupDraft(drafts: CupSetupDraftMap, values: CupSetupValues): CupSetupDraft {
  return (
    drafts[cupSetupKey(values)] ?? {
      ...values,
      ...EMPTY_CUP_SETUP_DRAFT,
    }
  );
}

function isFormControlTarget(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    target.closest("a, button, input, select, textarea, [role='slider']") !== null
  );
}

function ResetEngineSetupsButton({
  className,
  disabled,
  label,
  onResetEngineSetups,
  setups,
}: {
  className?: string;
  disabled: boolean;
  label: string;
  onResetEngineSetups: (setups: readonly CourseSetupValues[]) => void;
  setups: readonly CourseSetupValues[];
}) {
  return (
    <Button
      className={`gap-2 ${className ?? ""}`}
      disabled={disabled}
      type="button"
      onClick={() => onResetEngineSetups(setups)}
    >
      <ResetIcon />
      <span>{label}</span>
    </Button>
  );
}
