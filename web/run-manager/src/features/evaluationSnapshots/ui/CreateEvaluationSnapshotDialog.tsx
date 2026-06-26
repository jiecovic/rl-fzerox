// web/run-manager/src/features/evaluationSnapshots/ui/CreateEvaluationSnapshotDialog.tsx
import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useRef, useState } from "react";

import type {
  CreateEvaluationRequest,
  EvaluationSourceArtifact,
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
  const [sourceArtifact, setSourceArtifact] = useState<EvaluationSourceArtifact>("latest");
  const [policyMode, setPolicyMode] = useState<PolicyPlaybackMode>("deterministic");
  const initialPresetId = preferredPresetId(run, evaluationPresets);
  const [presetId, setPresetId] = useState(initialPresetId);
  const [isCreating, setIsCreating] = useState(false);
  const requestInFlightRef = useRef(false);
  const preset =
    evaluationPresets.find((candidate) => candidate.id === presetId) ??
    evaluationPresets[0] ??
    null;
  const defaultName = `${run.name} · ${sourceArtifact} ${checkpointStepLabel(run)} · ${
    preset?.name ?? "evaluation"
  }`;

  useEffect(() => {
    if (!open) {
      return;
    }
    setPresetId(initialPresetId);
    setSourceArtifact("latest");
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
        name: defaultName,
        policyMode,
        presetId: preset.id,
        sourceArtifact,
        sourceRunId: run.id,
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
              <span>Checkpoint</span>
              <FieldSelect
                value={sourceArtifact}
                onChange={(event) =>
                  setSourceArtifact(event.currentTarget.value as EvaluationSourceArtifact)
                }
              >
                <option value="latest">latest</option>
                <option value="best">best</option>
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
            <PresetLine label="Snapshot" value={`${sourceArtifact} · ${policyMode}`} />
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
      ? "gp_course_master_all_courses"
      : "time_attack_all_courses";
  return presets.some((preset) => preset.id === preferredId) ? preferredId : (presets[0]?.id ?? "");
}

function evaluationTargetLabel(preset: ManagedEvaluationPreset) {
  const target = preset.target;
  const mode = target.mode === "gp_course" ? "GP course" : "Time Attack course";
  const parts = [
    target.cup_ids.length > 0 ? selectionCountLabel(target.cup_ids, "cup") : null,
    target.course_ids.length > 0 ? selectionCountLabel(target.course_ids, "course") : null,
    difficultySelectionLabel(target.difficulties),
  ].filter((part) => part !== null);
  return `${mode} · ${parts.length === 0 ? "all targets" : parts.join(" · ")}`;
}

function difficultySelectionLabel(difficulties: readonly string[]) {
  if (difficulties.length === 0) {
    return null;
  }
  return difficulties.map(titleLabel).join(", ");
}

function presetSummary(preset: ManagedEvaluationPreset | null) {
  if (preset === null) {
    return "-";
  }
  const variants =
    preset.target.mode === "gp_course" && preset.target.baseline_variant_count > 1
      ? ` · ${preset.target.baseline_variant_count} variants`
      : "";
  return `${preset.target.repeats_per_target}x${variants} · seed ${preset.seed} · ${preset.renderer}`;
}

function checkpointStepLabel(run: ManagedRunDetail) {
  const localSteps = run.runtime?.num_timesteps ?? run.source_num_timesteps;
  if (localSteps === null || localSteps === undefined) {
    return "steps unknown";
  }
  return `${(run.lineage_step_offset + localSteps).toLocaleString()} steps`;
}

function selectionCountLabel(values: readonly unknown[], singular: string) {
  return `${values.length} ${pluralize(values.length, singular)}`;
}

function pluralize(count: number, singular: string) {
  return count === 1 ? singular : `${singular}s`;
}

function titleLabel(value: string) {
  return value
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(" ");
}

function PresetLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid gap-1 sm:grid-cols-[116px_minmax(0,1fr)]">
      <span className="font-bold tracking-[0.04em] text-app-muted uppercase">{label}</span>
      <span className="min-w-0 break-words text-app-text">{value}</span>
    </div>
  );
}
