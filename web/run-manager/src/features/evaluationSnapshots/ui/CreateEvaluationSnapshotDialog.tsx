// web/run-manager/src/features/evaluationSnapshots/ui/CreateEvaluationSnapshotDialog.tsx
import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useRef, useState } from "react";

import type {
  CreateEvaluationRequest,
  ManagedEvaluation,
  ManagedEvaluationPreset,
  ManagedRunDetail,
  PolicyPlaybackMode,
} from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { FieldSelect, FieldShell } from "@/shared/ui/Field";

interface CreateEvaluationSnapshotDialogProps {
  evaluationPresets: ManagedEvaluationPreset[];
  open: boolean;
  run: ManagedRunDetail;
  onClose: () => void;
  onCreateEvaluation: (request: CreateEvaluationRequest) => Promise<ManagedEvaluation>;
  onGlobalError: (message: string | null) => void;
  onOpenEvaluation: (evaluation: ManagedEvaluation) => void;
}

export function CreateEvaluationSnapshotDialog({
  evaluationPresets,
  open,
  run,
  onClose,
  onCreateEvaluation,
  onGlobalError,
  onOpenEvaluation,
}: CreateEvaluationSnapshotDialogProps) {
  const [policyMode, setPolicyMode] = useState<PolicyPlaybackMode>("deterministic");
  const initialPresetId = preferredPresetId(run, evaluationPresets);
  const [presetId, setPresetId] = useState(initialPresetId);
  const [isCreating, setIsCreating] = useState(false);
  const requestInFlightRef = useRef(false);
  const preset =
    evaluationPresets.find((candidate) => candidate.id === presetId) ??
    evaluationPresets[0] ??
    null;
  const defaultName = `${run.name} · ${preset?.name ?? "evaluation"}`;

  useEffect(() => {
    if (!open) {
      return;
    }
    setPresetId(initialPresetId);
    setPolicyMode("deterministic");
  }, [initialPresetId, open]);

  async function submit() {
    if (requestInFlightRef.current) {
      return;
    }
    if (preset === null) {
      onGlobalError("No evaluation preset is available.");
      return;
    }
    requestInFlightRef.current = true;
    setIsCreating(true);
    onGlobalError(null);
    try {
      const created = await onCreateEvaluation({
        config: preset.config,
        courseIds: preset.target.course_ids,
        cupIds: preset.target.cup_ids,
        difficulties: preset.target.difficulties,
        name: defaultName,
        policyMode,
        presetId: preset.id,
        repeatsPerTarget: preset.target.repeats_per_target,
        seed: preset.seed,
        sourceArtifact: preset.source_artifact,
        sourceRunId: run.id,
        targetMode: preset.target.mode,
        vehicleIds: preset.target.vehicle_ids,
      });
      onOpenEvaluation(created);
      onClose();
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to create evaluation");
    } finally {
      requestInFlightRef.current = false;
      setIsCreating(false);
    }
  }

  return (
    <Dialog.Root
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && !isCreating) {
          onClose();
        }
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-[rgba(11,16,24,0.72)]" />
        <Dialog.Content
          className="fixed left-1/2 top-1/2 z-50 grid w-[min(620px,calc(100vw-48px))] -translate-x-1/2 -translate-y-1/2 gap-5 border border-app-border-strong bg-app-surface p-[22px]"
          onEscapeKeyDown={(event) => {
            if (isCreating) {
              event.preventDefault();
            }
          }}
          onPointerDownOutside={(event) => {
            if (isCreating) {
              event.preventDefault();
            }
          }}
        >
          <Dialog.Title className="m-0 text-lg font-semibold text-app-text">
            Create evaluation
          </Dialog.Title>
          <Dialog.Description className="m-0 text-sm text-app-muted">
            Pick the preset to run against this checkpoint.
          </Dialog.Description>

          <div className="grid gap-3">
            <FieldShell>
              <span>Evaluation preset</span>
              <FieldSelect
                value={presetId}
                onChange={(event) => {
                  setPresetId(event.target.value);
                }}
              >
                {evaluationPresets.map((candidate) => (
                  <option key={candidate.id} value={candidate.id}>
                    {candidate.name}
                  </option>
                ))}
              </FieldSelect>
            </FieldShell>
            <FieldShell>
              <span>Policy mode</span>
              <FieldSelect
                value={policyMode}
                onChange={(event) => setPolicyMode(event.currentTarget.value as PolicyPlaybackMode)}
              >
                <option value="deterministic">deterministic</option>
                <option value="stochastic">stochastic</option>
              </FieldSelect>
            </FieldShell>
          </div>

          <div className="grid gap-2 border border-app-border bg-app-surface-muted p-3 text-sm text-app-muted">
            <PresetLine label="Name" value={defaultName} />
            <PresetLine
              label="Target"
              value={preset === null ? "-" : evaluationTargetLabel(preset)}
            />
            <PresetLine label="Preset" value={presetSummary(preset)} />
            <PresetLine label="Policy" value={policyMode} />
          </div>

          <div className="flex justify-end gap-2.5">
            <Button disabled={isCreating} onClick={onClose}>
              Cancel
            </Button>
            <Button
              disabled={isCreating || preset === null}
              variant="primary"
              onClick={() => void submit()}
            >
              {isCreating ? "Creating" : "Create evaluation"}
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function preferredPresetId(
  run: ManagedRunDetail,
  presets: readonly ManagedEvaluationPreset[],
): string {
  const preferredId =
    run.config.tracks.race_mode === "gp_race"
      ? "gp_course_master_blue_falcon_all_courses"
      : "time_attack_blue_falcon_all_courses";
  return presets.some((preset) => preset.id === preferredId) ? preferredId : (presets[0]?.id ?? "");
}

function evaluationTargetLabel(preset: ManagedEvaluationPreset) {
  const target = preset.target;
  const mode = target.mode === "gp_course" ? "GP course" : "Time Attack course";
  const parts = [
    target.cup_ids.length > 0 ? selectionCountLabel(target.cup_ids, "cup") : null,
    target.course_ids.length > 0 ? selectionCountLabel(target.course_ids, "course") : null,
    target.difficulties.length > 0 ? selectionCountLabel(target.difficulties, "difficulty") : null,
    target.vehicle_ids.length > 0 ? selectionCountLabel(target.vehicle_ids, "vehicle") : null,
  ].filter((part) => part !== null);
  return `${mode} · ${parts.length === 0 ? "all targets" : parts.join(" · ")}`;
}

function presetSummary(preset: ManagedEvaluationPreset | null) {
  if (preset === null) {
    return "-";
  }
  return `${preset.source_artifact} · ${preset.target.repeats_per_target}x · seed ${preset.seed} · ${preset.renderer}`;
}

function selectionCountLabel(values: readonly unknown[], singular: string) {
  return `${values.length} ${pluralize(values.length, singular)}`;
}

function pluralize(count: number, singular: string) {
  if (count === 1) {
    return singular;
  }
  return singular === "difficulty" ? "difficulties" : `${singular}s`;
}

function PresetLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid gap-1 sm:grid-cols-[116px_minmax(0,1fr)]">
      <span className="font-bold tracking-[0.04em] text-app-muted uppercase">{label}</span>
      <span className="min-w-0 break-words text-app-text">{value}</span>
    </div>
  );
}
