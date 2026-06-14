// web/run-manager/src/features/saveGameCourseSetup/ui/CourseSetupPanels.tsx
import { memo, useMemo, useState } from "react";
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
} from "@/features/saveGameCourseSetup/model/courseSetup";
import {
  courseSetupsForCups,
  courseSetupValues,
  cupSetupKey,
  cupSetupValues,
  EMPTY_COURSE_SETUP_DRAFT,
  EMPTY_CUP_SETUP_DRAFT,
  exactCourseSetupDraft,
  sharedCourseDraft,
} from "@/features/saveGameCourseSetup/model/courseSetup";
import {
  EngineDraftInput,
  PolicyArtifactSelect,
  PolicyDraftSelect,
  PolicySelectionSelect,
  VehicleDraftSelect,
} from "@/features/saveGameCourseSetup/ui/CourseSetupFields";
import type { ConfigMetadata, ManagedRun } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { DisclosureToolbar } from "@/shared/ui/config/DisclosureToolbar";
import { ResetIcon, SaveDraftIcon } from "@/shared/ui/icons";

const EMPTY_POLICY_SELECTION_DRAFT: PolicySelectionDraft = {
  policyArtifact: "best",
  policyRunId: "",
};

export const GlobalPolicyPanel = memo(function GlobalPolicyPanel({
  assignableRuns,
  cups,
  onImportEngineTuning,
  onApplySetups,
  updating,
}: {
  assignableRuns: readonly ManagedRun[];
  cups: readonly CupView[];
  onImportEngineTuning: (draft: PolicySelectionDraft) => Promise<void>;
  onApplySetups: (setups: readonly CourseSetupValues[], draft: PolicySelectionDraft) => void;
  updating: boolean;
}) {
  const [draft, setDraft] = useState<PolicySelectionDraft>(EMPTY_POLICY_SELECTION_DRAFT);
  const [importing, setImporting] = useState(false);
  const selectedRun = assignableRuns.find((run) => run.id === draft.policyRunId) ?? null;
  const canImportEngines = !updating && selectedRun?.vehicle_setup.engine_mode === "adaptive_tuner";
  const canApply = !updating && draft.policyRunId !== "";
  const allCourseSetups = courseSetupsForCups(cups);

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
          assignableRuns={assignableRuns}
          disabled={updating}
          draft={draft}
          label="Policy"
          onDraftChange={setDraft}
        />
        <PolicyArtifactSelect
          disabled={updating}
          draft={draft}
          label="Artifact"
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
  assignableRuns,
  cups,
  dirtySetupCount,
  metadata,
  onCourseSetupDraftChange,
  onCupSetupDraftChange,
  onResetEngineSetups,
  onSaveSetups,
  courseSetupDrafts,
  cupSetupDrafts,
  savingCourseSetups,
  updating,
  unlockedVehicleIds,
}: {
  assignableRuns: readonly ManagedRun[];
  cups: readonly CupView[];
  dirtySetupCount: number;
  metadata: ConfigMetadata;
  onCourseSetupDraftChange: (values: CourseSetupValues, draft: PolicyArtifactDraft) => void;
  onCupSetupDraftChange: (values: CupSetupValues, vehicleId: string) => void;
  onResetEngineSetups: (setups: readonly CourseSetupValues[]) => void;
  onSaveSetups: () => void;
  courseSetupDrafts: CourseSetupDraftMap;
  cupSetupDrafts: CupSetupDraftMap;
  savingCourseSetups: boolean;
  updating: boolean;
  unlockedVehicleIds: readonly string[];
}) {
  const [collapsedCupIds, setCollapsedCupIds] = useState<readonly string[]>([]);
  const collapsedCupIdSet = useMemo(() => new Set(collapsedCupIds), [collapsedCupIds]);
  const saveDisabled = dirtySetupCount === 0 || savingCourseSetups || updating;
  const allCourseSetups = courseSetupsForCups(cups);

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
          <Button
            className="gap-2"
            disabled={updating || savingCourseSetups}
            type="button"
            onClick={() => onResetEngineSetups(allCourseSetups)}
          >
            <ResetIcon />
            <span>Reset all engines to 50</span>
          </Button>
          <Button
            className="gap-2"
            disabled={saveDisabled}
            type="button"
            variant={dirtySetupCount > 0 ? "primary" : "secondary"}
            onClick={onSaveSetups}
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
            assignableRuns={assignableRuns}
            collapsed={collapsedCupIdSet.has(cup.id)}
            cup={cup}
            courseSetupDrafts={courseSetupDrafts}
            cupSetupDrafts={cupSetupDrafts}
            metadata={metadata}
            updating={updating || savingCourseSetups}
            unlockedVehicleIds={unlockedVehicleIds}
            onCollapsedChange={setCupCollapsed}
            onCourseSetupDraftChange={onCourseSetupDraftChange}
            onCupSetupDraftChange={onCupSetupDraftChange}
            onResetEngineSetups={onResetEngineSetups}
          />
        ))}
      </div>
    </div>
  );
});

const CupSetupBlock = memo(function CupSetupBlock({
  assignableRuns,
  collapsed,
  cup,
  cupSetupDrafts,
  metadata,
  onCollapsedChange,
  onCourseSetupDraftChange,
  onCupSetupDraftChange,
  onResetEngineSetups,
  courseSetupDrafts,
  updating,
  unlockedVehicleIds,
}: {
  assignableRuns: readonly ManagedRun[];
  collapsed: boolean;
  cup: CupView;
  cupSetupDrafts: CupSetupDraftMap;
  metadata: ConfigMetadata;
  onCollapsedChange: (cupId: string, collapsed: boolean) => void;
  onCourseSetupDraftChange: (values: CourseSetupValues, draft: PolicyArtifactDraft) => void;
  onCupSetupDraftChange: (values: CupSetupValues, vehicleId: string) => void;
  onResetEngineSetups: (setups: readonly CourseSetupValues[]) => void;
  courseSetupDrafts: CourseSetupDraftMap;
  updating: boolean;
  unlockedVehicleIds: readonly string[];
}) {
  const cupValues = cupSetupValues(cup);
  const cupDraft = cupSetupDraft(cupSetupDrafts, cupValues);
  const courseValuesList = cup.courses.map((course) => courseSetupValues(cup, course.id));
  const courseDrafts = courseValuesList
    .map((values) => exactCourseSetupDraft(courseSetupDrafts, values))
    .filter((draft): draft is CourseSetupDraft => draft !== null);
  const fallbackDraft: PolicyArtifactDraft = {
    ...EMPTY_COURSE_SETUP_DRAFT,
    vehicleId: cupDraft.vehicleId,
  };
  const sharedDraft = sharedCourseDraft(courseDrafts, courseValuesList.length) ?? fallbackDraft;
  const bulkDraft: PolicyArtifactDraft = {
    ...sharedDraft,
    vehicleId: cupDraft.vehicleId,
  };
  const bulkPolicyDraft: PolicySelectionDraft = {
    policyArtifact: bulkDraft.policyArtifact,
    policyRunId: bulkDraft.policyRunId,
  };

  function applyBulkCoursePolicy(nextDraft: PolicySelectionDraft) {
    for (const values of courseValuesList) {
      const currentDraft = exactCourseSetupDraft(courseSetupDrafts, values) ?? fallbackDraft;
      onCourseSetupDraftChange(values, {
        ...currentDraft,
        policyArtifact: nextDraft.policyArtifact,
        policyRunId: nextDraft.policyRunId,
        vehicleId: cupDraft.vehicleId,
      });
    }
  }

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
          <div className="grid gap-3 md:grid-cols-[minmax(240px,1fr)_130px_minmax(180px,240px)] md:items-end">
            <PolicySelectionSelect
              assignableRuns={assignableRuns}
              disabled={updating}
              draft={bulkPolicyDraft}
              label={`${cup.label} policy`}
              visibleLabel="Policy"
              onDraftChange={applyBulkCoursePolicy}
            />
            <PolicyArtifactSelect
              disabled={updating}
              draft={bulkPolicyDraft}
              label={`${cup.label} artifact`}
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
          </div>
          <div className="flex justify-end">
            <Button
              className="gap-2"
              disabled={updating}
              type="button"
              onClick={() => onResetEngineSetups(courseValuesList)}
            >
              <ResetIcon />
              <span>Reset {cup.label} engines to 50</span>
            </Button>
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
              return (
                <div
                  key={course.id}
                  className={courseCardClass(false, "min-h-[172px] content-start gap-3 p-3")}
                  data-cup={cup.id}
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
                      <span className="text-xs text-app-muted">Course {courseIndex + 1}</span>
                    </div>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_112px_96px]">
                    <PolicyDraftSelect
                      assignableRuns={assignableRuns}
                      disabled={updating}
                      draft={courseDraft}
                      label={`${course.display_name} policy`}
                      lockedVehicleId={cupDraft.vehicleId}
                      metadata={metadata}
                      unlockedVehicleIds={unlockedVehicleIds}
                      visibleLabel="Policy"
                      onDraftChange={updateCourseDraft}
                    />
                    <PolicyArtifactSelect
                      disabled={updating}
                      draft={courseDraft}
                      label={`${course.display_name} artifact`}
                      visibleLabel="Artifact"
                      onDraftChange={updateCourseDraft}
                    />
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
