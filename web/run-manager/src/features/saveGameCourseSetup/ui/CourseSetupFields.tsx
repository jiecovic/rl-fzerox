// web/run-manager/src/features/saveGameCourseSetup/ui/CourseSetupFields.tsx
import { memo, useMemo } from "react";
import type {
  PolicyArtifactDraft,
  PolicySelectionDraft,
} from "@/features/saveGameCourseSetup/model/courseSetup";
import { preferredVehicleSetup } from "@/features/saveGameCourseSetup/model/courseSetup";
import type { ConfigMetadata, ManagedRun, SavePolicyArtifact } from "@/shared/api/contract";
import { IntegerTextInput } from "@/shared/ui/configFields";
import { FieldSelect, FieldShell } from "@/shared/ui/Field";

export function PolicySelectionSelect({
  assignableRuns,
  disabled,
  draft,
  label,
  onDraftChange,
  visibleLabel,
}: {
  assignableRuns: readonly ManagedRun[];
  disabled: boolean;
  draft: PolicySelectionDraft;
  label: string;
  onDraftChange: (draft: PolicySelectionDraft) => void;
  visibleLabel?: string;
}) {
  return (
    <FieldShell>
      <span>{visibleLabel ?? label}</span>
      <FieldSelect
        aria-label={label}
        disabled={disabled || assignableRuns.length === 0}
        value={draft.policyRunId}
        onChange={(event) =>
          onDraftChange({
            ...draft,
            policyRunId: event.currentTarget.value,
          })
        }
      >
        <option disabled value="">
          Select policy
        </option>
        <PolicyRunOptions assignableRuns={assignableRuns} />
      </FieldSelect>
    </FieldShell>
  );
}

export function PolicyArtifactSelect<TDraft extends PolicySelectionDraft>({
  disabled,
  draft,
  label,
  onDraftChange,
  visibleLabel,
}: {
  disabled: boolean;
  draft: TDraft;
  label: string;
  onDraftChange: (draft: TDraft) => void;
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

export function PolicyDraftSelect({
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
        <PolicyRunOptions assignableRuns={assignableRuns} />
      </FieldSelect>
    </FieldShell>
  );
}

const PolicyRunOptions = memo(function PolicyRunOptions({
  assignableRuns,
}: {
  assignableRuns: readonly ManagedRun[];
}) {
  return (
    <>
      {assignableRuns.map((run) => (
        <option key={run.id} value={run.id}>
          {run.name}
        </option>
      ))}
    </>
  );
});

export function VehicleDraftSelect({
  disabled,
  draft,
  label,
  metadata,
  onDraftChange,
  unlockedVehicleIds,
  visibleLabel,
}: {
  disabled: boolean;
  draft: Pick<PolicyArtifactDraft, "vehicleId">;
  label: string;
  metadata: ConfigMetadata;
  onDraftChange: (draft: Pick<PolicyArtifactDraft, "vehicleId">) => void;
  unlockedVehicleIds: readonly string[];
  visibleLabel?: string;
}) {
  const unlockedVehicleSet = useMemo(() => new Set(unlockedVehicleIds), [unlockedVehicleIds]);
  const vehicleOptions = useMemo(
    () => metadata.vehicles.filter((vehicle) => unlockedVehicleSet.has(vehicle.id)),
    [metadata.vehicles, unlockedVehicleSet],
  );
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

export function EngineDraftInput({
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
