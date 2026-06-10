// src/rl_fzerox/apps/run_manager/web/src/features/saveGameCourseSetup/ui/CourseSetupPanels.tsx
import { useState } from "react";
import { courseCardClass } from "@/entities/runConfig/ui/sections/tracks/coursePoolStyle";
import { TrackCupBanner } from "@/entities/runConfig/ui/sections/tracks/TrackCupBanner";
import { TrackMinimap } from "@/entities/runConfig/ui/sections/tracks/TrackMinimap";
import type {
  CourseSetupDraft,
  CourseSetupDraftMap,
  CourseSetupScopeValues,
  CupView,
  PolicyArtifactDraft,
} from "@/features/saveGameCourseSetup/model/courseSetup";
import {
  courseSetupScopeValues,
  courseSetupsForCups,
  cupSetupScopeValues,
  EMPTY_COURSE_SETUP_DRAFT,
  exactCourseSetupDraft,
  preferredVehicleSetup,
  sharedCourseDraft,
} from "@/features/saveGameCourseSetup/model/courseSetup";
import type { ConfigMetadata, ManagedRun, SavePolicyArtifact } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { DisclosureToolbar } from "@/shared/ui/config/DisclosureToolbar";
import { IntegerTextInput } from "@/shared/ui/configFields";
import { FieldSelect, FieldShell } from "@/shared/ui/Field";
import { SaveDraftIcon } from "@/shared/ui/icons";

export function GlobalPolicyPanel({
  assignableRuns,
  cups,
  metadata,
  onApplySetups,
  updating,
  unlockedVehicleIds,
}: {
  assignableRuns: readonly ManagedRun[];
  cups: readonly CupView[];
  metadata: ConfigMetadata;
  onApplySetups: (setups: readonly CourseSetupScopeValues[], draft: PolicyArtifactDraft) => void;
  updating: boolean;
  unlockedVehicleIds: readonly string[];
}) {
  const [draft, setDraft] = useState<PolicyArtifactDraft>(EMPTY_COURSE_SETUP_DRAFT);
  const canApply = !updating && draft.policyRunId !== "";
  const allCourseSetups = courseSetupsForCups(cups);

  return (
    <div className="grid content-start gap-3 border border-app-border bg-app-surface-muted p-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="grid gap-1">
          <h4 className="m-0 text-sm font-bold tracking-[0.04em] text-app-muted uppercase">
            Default setup
          </h4>
          <p className="m-0 text-sm text-app-muted">
            Stage a trained policy artifact, then copy it into each course setup.
          </p>
        </div>
        <Button
          className="min-w-[160px]"
          disabled={!canApply}
          type="button"
          onClick={() => onApplySetups(allCourseSetups, draft)}
        >
          Apply to all courses
        </Button>
      </div>
      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_140px]">
        <PolicyDraftSelect
          assignableRuns={assignableRuns}
          disabled={updating}
          draft={draft}
          label="Default policy"
          metadata={metadata}
          unlockedVehicleIds={unlockedVehicleIds}
          onDraftChange={setDraft}
        />
        <ArtifactDraftSelect
          disabled={updating}
          draft={draft}
          label="Artifact"
          onDraftChange={setDraft}
        />
      </div>
    </div>
  );
}

export function CourseSetupPanel({
  assignableRuns,
  cups,
  dirtyCourseSetupCount,
  metadata,
  onCourseSetupDraftChange,
  onSaveSetups,
  courseSetupDrafts,
  savingCourseSetups,
  updating,
  unlockedVehicleIds,
}: {
  assignableRuns: readonly ManagedRun[];
  cups: readonly CupView[];
  dirtyCourseSetupCount: number;
  metadata: ConfigMetadata;
  onCourseSetupDraftChange: (
    scopeValues: CourseSetupScopeValues,
    draft: PolicyArtifactDraft,
  ) => void;
  onSaveSetups: () => void;
  courseSetupDrafts: CourseSetupDraftMap;
  savingCourseSetups: boolean;
  updating: boolean;
  unlockedVehicleIds: readonly string[];
}) {
  const [collapsedCupIds, setCollapsedCupIds] = useState<readonly string[]>([]);
  const collapsedCupIdSet = new Set(collapsedCupIds);
  const saveDisabled = dirtyCourseSetupCount === 0 || savingCourseSetups || updating;

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
            disabled={saveDisabled}
            type="button"
            variant={dirtyCourseSetupCount > 0 ? "primary" : "secondary"}
            onClick={onSaveSetups}
          >
            <SaveDraftIcon />
            <span>
              {savingCourseSetups
                ? "Saving"
                : dirtyCourseSetupCount > 0
                  ? `Save ${dirtyCourseSetupCount} change${dirtyCourseSetupCount === 1 ? "" : "s"}`
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
            metadata={metadata}
            updating={updating || savingCourseSetups}
            unlockedVehicleIds={unlockedVehicleIds}
            onCollapsedChange={setCupCollapsed}
            onCourseSetupDraftChange={onCourseSetupDraftChange}
          />
        ))}
      </div>
    </div>
  );
}

function CupSetupBlock({
  assignableRuns,
  collapsed,
  cup,
  metadata,
  onCollapsedChange,
  onCourseSetupDraftChange,
  courseSetupDrafts,
  updating,
  unlockedVehicleIds,
}: {
  assignableRuns: readonly ManagedRun[];
  collapsed: boolean;
  cup: CupView;
  metadata: ConfigMetadata;
  onCollapsedChange: (cupId: string, collapsed: boolean) => void;
  onCourseSetupDraftChange: (
    scopeValues: CourseSetupScopeValues,
    draft: PolicyArtifactDraft,
  ) => void;
  courseSetupDrafts: CourseSetupDraftMap;
  updating: boolean;
  unlockedVehicleIds: readonly string[];
}) {
  const cupScopeValues = cupSetupScopeValues(cup);
  const cupFallbackDraft = exactCourseSetupDraft(courseSetupDrafts, cupScopeValues);
  const courseScopeValues = cup.courses.map((course) => courseSetupScopeValues(cup, course.id));
  const courseDrafts = courseScopeValues
    .map((scopeValues) => exactCourseSetupDraft(courseSetupDrafts, scopeValues))
    .filter((draft): draft is CourseSetupDraft => draft !== null);
  // New bulk edits fan out to course rows, but existing save games can still
  // have cup-scoped rows. Use them only to seed empty course rows.
  const fallbackDraft = cupFallbackDraft ?? EMPTY_COURSE_SETUP_DRAFT;
  const bulkDraft = sharedCourseDraft(courseDrafts, courseScopeValues.length) ?? fallbackDraft;

  function applyBulkCourseDraft(
    nextDraft: PolicyArtifactDraft,
    options: { replaceEngine: boolean },
  ) {
    for (const scopeValues of courseScopeValues) {
      const currentDraft = exactCourseSetupDraft(courseSetupDrafts, scopeValues) ?? fallbackDraft;
      onCourseSetupDraftChange(scopeValues, {
        ...nextDraft,
        engineSettingRawValue: options.replaceEngine
          ? nextDraft.engineSettingRawValue
          : currentDraft.engineSettingRawValue,
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
      <div className="config-disclosure-body gap-3">
        <div className="grid gap-3 md:grid-cols-[minmax(240px,1fr)_130px_minmax(180px,240px)] md:items-end">
          <PolicyDraftSelect
            assignableRuns={assignableRuns}
            disabled={updating}
            draft={bulkDraft}
            label={`${cup.label} policy`}
            metadata={metadata}
            unlockedVehicleIds={unlockedVehicleIds}
            visibleLabel="Policy"
            onDraftChange={(nextDraft) => applyBulkCourseDraft(nextDraft, { replaceEngine: true })}
          />
          <ArtifactDraftSelect
            disabled={updating}
            draft={bulkDraft}
            label={`${cup.label} artifact`}
            visibleLabel="Artifact"
            onDraftChange={(nextDraft) => applyBulkCourseDraft(nextDraft, { replaceEngine: false })}
          />
          <VehicleDraftSelect
            disabled={updating}
            draft={bulkDraft}
            label={`${cup.label} vehicle`}
            metadata={metadata}
            unlockedVehicleIds={unlockedVehicleIds}
            visibleLabel="Vehicle"
            onDraftChange={(nextDraft) => applyBulkCourseDraft(nextDraft, { replaceEngine: false })}
          />
        </div>
        <div className="grid grid-cols-1 gap-3 border-t border-app-border pt-3 xl:grid-cols-2 2xl:grid-cols-3">
          {cup.courses.map((course, courseIndex) => {
            const courseScopeValues = courseSetupScopeValues(cup, course.id);
            const courseDraft =
              exactCourseSetupDraft(courseSetupDrafts, courseScopeValues) ?? fallbackDraft;
            const updateCourseDraft = (nextDraft: PolicyArtifactDraft) =>
              onCourseSetupDraftChange(courseScopeValues, {
                ...nextDraft,
                vehicleId: bulkDraft.vehicleId,
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
                    lockedVehicleId={bulkDraft.vehicleId}
                    metadata={metadata}
                    unlockedVehicleIds={unlockedVehicleIds}
                    visibleLabel="Policy"
                    onDraftChange={updateCourseDraft}
                  />
                  <ArtifactDraftSelect
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
    </details>
  );
}

function PolicyDraftSelect({
  assignableRuns,
  disabled,
  draft,
  label,
  lockedVehicleId,
  metadata,
  onDraftChange,
  unlockedVehicleIds,
  visibleLabel,
}: {
  assignableRuns: readonly ManagedRun[];
  disabled: boolean;
  draft: PolicyArtifactDraft;
  label: string;
  lockedVehicleId?: string;
  metadata: ConfigMetadata;
  onDraftChange: (draft: PolicyArtifactDraft) => void;
  unlockedVehicleIds: readonly string[];
  visibleLabel?: string;
}) {
  return (
    <FieldShell>
      <span>{visibleLabel ?? label}</span>
      <FieldSelect
        aria-label={label}
        disabled={disabled || assignableRuns.length === 0}
        value={draft.policyRunId}
        onChange={(event) => {
          const policyRunId = event.currentTarget.value;
          const selectedRun = assignableRuns.find((run) => run.id === policyRunId) ?? null;
          const preferredSetup = preferredVehicleSetup({
            currentDraft: draft,
            metadata,
            run: selectedRun,
            unlockedVehicleIds,
          });
          onDraftChange({
            ...draft,
            ...preferredSetup,
            policyRunId,
            vehicleId: lockedVehicleId ?? preferredSetup.vehicleId,
          });
        }}
      >
        <option disabled value="">
          Select policy
        </option>
        {assignableRuns.map((run) => (
          <option key={run.id} value={run.id}>
            {run.name}
          </option>
        ))}
      </FieldSelect>
    </FieldShell>
  );
}

function VehicleDraftSelect({
  disabled,
  draft,
  label,
  metadata,
  onDraftChange,
  unlockedVehicleIds,
  visibleLabel,
}: {
  disabled: boolean;
  draft: PolicyArtifactDraft;
  label: string;
  metadata: ConfigMetadata;
  onDraftChange: (draft: PolicyArtifactDraft) => void;
  unlockedVehicleIds: readonly string[];
  visibleLabel?: string;
}) {
  const unlockedVehicleSet = new Set(unlockedVehicleIds);
  const vehicleOptions = metadata.vehicles.filter((vehicle) => unlockedVehicleSet.has(vehicle.id));
  return (
    <FieldShell>
      <span>{visibleLabel ?? label}</span>
      <FieldSelect
        aria-label={label}
        disabled={disabled || vehicleOptions.length === 0}
        value={draft.vehicleId}
        onChange={(event) => {
          onDraftChange({ ...draft, vehicleId: event.currentTarget.value });
        }}
      >
        {vehicleOptions.map((vehicle) => (
          <option key={vehicle.id} value={vehicle.id}>
            {vehicle.display_name}
          </option>
        ))}
      </FieldSelect>
    </FieldShell>
  );
}

function EngineDraftInput({
  disabled,
  draft,
  label,
  onDraftChange,
  visibleLabel,
}: {
  disabled: boolean;
  draft: PolicyArtifactDraft;
  label: string;
  onDraftChange: (draft: PolicyArtifactDraft) => void;
  visibleLabel?: string;
}) {
  return (
    <FieldShell>
      <span>{visibleLabel ?? label}</span>
      <IntegerTextInput
        aria-label={label}
        className="h-[34px] indent-0 tabular-nums"
        disabled={disabled}
        max={100}
        min={0}
        value={draft.engineSettingRawValue}
        onChange={(value) => {
          onDraftChange({
            ...draft,
            engineSettingRawValue: value,
          });
        }}
      />
    </FieldShell>
  );
}

function ArtifactDraftSelect({
  disabled,
  draft,
  label,
  onDraftChange,
  visibleLabel,
}: {
  disabled: boolean;
  draft: PolicyArtifactDraft;
  label: string;
  onDraftChange: (draft: PolicyArtifactDraft) => void;
  visibleLabel?: string;
}) {
  return (
    <FieldShell>
      <span>{visibleLabel ?? label}</span>
      <FieldSelect
        aria-label={label}
        disabled={disabled || draft.policyRunId === ""}
        value={draft.policyArtifact}
        onChange={(event) => {
          onDraftChange({
            ...draft,
            policyArtifact: event.currentTarget.value as SavePolicyArtifact,
          });
        }}
      >
        <option value="best">best</option>
        <option value="latest">latest</option>
      </FieldSelect>
    </FieldShell>
  );
}
