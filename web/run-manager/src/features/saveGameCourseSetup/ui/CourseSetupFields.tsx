// web/run-manager/src/features/saveGameCourseSetup/ui/CourseSetupFields.tsx
import { memo, useMemo } from "react";
import { SingleSlider } from "@/entities/runConfig/ui/sections/vehicle/engineSetting/SingleSlider";
import type {
  PolicyArtifactDraft,
  PolicySelectionDraft,
  PolicySourceOption,
} from "@/features/saveGameCourseSetup/model/courseSetup";
import {
  policySelectionDraftForSource,
  policySourceKey,
  policySourceOptionKey,
  preferredVehicleSetup,
  selectedPolicySource,
} from "@/features/saveGameCourseSetup/model/courseSetup";
import type { ConfigMetadata, SavePolicyArtifact } from "@/shared/api/contract";
import {
  ENGINE_SLIDER,
  enginePercentToSliderStep,
  engineSliderStepPercentLabel,
} from "@/shared/domain/engineBuckets";
import { FieldSelect, FieldShell } from "@/shared/ui/Field";

export function PolicySelectionSelect({
  disabled,
  draft,
  label,
  onDraftChange,
  policySources,
  visibleLabel,
}: {
  disabled: boolean;
  draft: PolicySelectionDraft;
  label: string;
  onDraftChange: (draft: PolicySelectionDraft) => void;
  policySources: readonly PolicySourceOption[];
  visibleLabel?: string;
}) {
  return (
    <FieldShell>
      <span>{visibleLabel ?? label}</span>
      <FieldSelect
        aria-label={label}
        disabled={disabled || policySources.length === 0}
        value={policySourceKey(draft)}
        onChange={(event) => {
          const selectedKey = event.currentTarget.value;
          const source =
            policySources.find((candidate) => policySourceOptionKey(candidate) === selectedKey) ??
            null;
          if (source !== null) {
            onDraftChange(policySelectionDraftForSource(draft, source));
          }
        }}
      >
        <option disabled value="">
          Select policy
        </option>
        <PolicySourceOptions policySources={policySources} />
      </FieldSelect>
    </FieldShell>
  );
}

export function PolicyArtifactSelect<TDraft extends PolicySelectionDraft>({
  disabled,
  draft,
  label,
  onDraftChange,
  policySources,
  visibleLabel,
}: {
  disabled: boolean;
  draft: TDraft;
  label: string;
  onDraftChange: (draft: TDraft) => void;
  policySources: readonly PolicySourceOption[];
  visibleLabel?: string;
}) {
  const source = selectedPolicySource(policySources, draft);
  const fixedArtifact = source?.kind === "evaluation";
  return (
    <FieldShell>
      <span>{visibleLabel ?? label}</span>
      <FieldSelect
        aria-label={label}
        disabled={disabled || draft.policySourceId === "" || fixedArtifact}
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
        <option value="final">final</option>
      </FieldSelect>
    </FieldShell>
  );
}

export function PolicyDraftSelect({
  disabled,
  draft,
  label,
  lockedVehicleId,
  metadata,
  onDraftChange,
  policySources,
  unlockedVehicleIds,
  visibleLabel,
}: {
  disabled: boolean;
  draft: PolicyArtifactDraft;
  label: string;
  lockedVehicleId?: string;
  metadata: ConfigMetadata;
  onDraftChange: (draft: PolicyArtifactDraft) => void;
  policySources: readonly PolicySourceOption[];
  unlockedVehicleIds: readonly string[];
  visibleLabel?: string;
}) {
  return (
    <FieldShell>
      <span>{visibleLabel ?? label}</span>
      <FieldSelect
        aria-label={label}
        disabled={disabled || policySources.length === 0}
        value={policySourceKey(draft)}
        onChange={(event) => {
          const selectedKey = event.currentTarget.value;
          const source =
            policySources.find((candidate) => policySourceOptionKey(candidate) === selectedKey) ??
            null;
          if (source === null) {
            return;
          }
          const preferredSetup = preferredVehicleSetup({
            currentDraft: draft,
            metadata,
            source,
            unlockedVehicleIds,
          });
          const policySelection = policySelectionDraftForSource(draft, source);
          onDraftChange({
            ...draft,
            ...preferredSetup,
            ...policySelection,
            vehicleId: lockedVehicleId ?? preferredSetup.vehicleId,
          });
        }}
      >
        <option disabled value="">
          Select policy
        </option>
        <PolicySourceOptions policySources={policySources} />
      </FieldSelect>
    </FieldShell>
  );
}

const PolicySourceOptions = memo(function PolicySourceOptions({
  policySources,
}: {
  policySources: readonly PolicySourceOption[];
}) {
  return (
    <>
      {policySources.map((source) => (
        <option key={policySourceOptionKey(source)} value={policySourceOptionKey(source)}>
          {source.label}
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
  const engineTicks = [
    { value: 0, label: "0%" },
    { value: enginePercentToSliderStep(50), label: "50%" },
    { value: ENGINE_SLIDER.maxStep, label: "100%" },
  ] as const;
  return (
    <FieldShell>
      <span>{visibleLabel ?? label}</span>
      <SingleSlider
        disabled={disabled}
        label={label}
        max={ENGINE_SLIDER.maxStep}
        min={ENGINE_SLIDER.minStep}
        step={1}
        ticks={engineTicks}
        value={draft.engineSettingRawValue}
        onChange={(engineSettingRawValue) => {
          onDraftChange({
            ...draft,
            engineSettingRawValue,
          });
        }}
      />
      <div className="grid justify-end gap-0.5 text-right tabular-nums">
        <span className="text-sm font-semibold text-app-text">
          {engineSliderStepPercentLabel(draft.engineSettingRawValue)}
        </span>
      </div>
    </FieldShell>
  );
}
