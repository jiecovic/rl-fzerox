// web/run-manager/src/features/evaluationSnapshots/ui/CreateEvaluationSnapshotDialog.tsx
import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  buildEvaluationPresets,
  clonePresetConfig,
  type EvaluationPreset,
  type EvaluationPresetId,
  evaluationTargetFromConfig,
  readEvaluationPresetStorage,
} from "@/entities/evaluation/model/presets";
import type {
  ConfigMetadata,
  CreateEvaluationRequest,
  ManagedEvaluation,
  ManagedRunConfig,
  ManagedRunDetail,
  PolicyPlaybackMode,
} from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { FieldSelect, FieldShell } from "@/shared/ui/Field";

interface CreateEvaluationSnapshotDialogProps {
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  open: boolean;
  run: ManagedRunDetail;
  onClose: () => void;
  onCreateEvaluation: (request: CreateEvaluationRequest) => Promise<ManagedEvaluation>;
  onGlobalError: (message: string | null) => void;
  onOpenEvaluation: (evaluation: ManagedEvaluation) => void;
}

export function CreateEvaluationSnapshotDialog({
  defaultConfig,
  metadata,
  open,
  run,
  onClose,
  onCreateEvaluation,
  onGlobalError,
  onOpenEvaluation,
}: CreateEvaluationSnapshotDialogProps) {
  const [presetStorage, setPresetStorage] = useState(() => readEvaluationPresetStorage());
  const [policyMode, setPolicyMode] = useState<PolicyPlaybackMode>("deterministic");
  const presets = useMemo(
    () =>
      buildEvaluationPresets({
        customPresets: presetStorage.customPresets,
        defaultConfig,
        metadata,
        overrides: presetStorage.overrides,
      }),
    [defaultConfig, metadata, presetStorage],
  );
  const initialPresetId: EvaluationPresetId =
    run.config.tracks.race_mode === "gp_race" ? "gp_course_blue_falcon" : "time_attack_blue_falcon";
  const [presetId, setPresetId] = useState<EvaluationPresetId>(initialPresetId);
  const [isCreating, setIsCreating] = useState(false);
  const requestInFlightRef = useRef(false);
  const preset = presets.find((candidate) => candidate.id === presetId) ?? presets[0] ?? null;
  const presetConfig = useMemo(
    () => (preset === null ? null : clonePresetConfig(preset.config)),
    [preset],
  );
  const target =
    presetConfig === null || preset === null
      ? null
      : evaluationTargetFromConfig(presetConfig, metadata, preset.targetMode);
  const defaultName = `${run.name} · ${preset?.label ?? "evaluation"}`;

  useEffect(() => {
    if (!open) {
      return;
    }
    setPresetStorage(readEvaluationPresetStorage());
    setPresetId(initialPresetId);
    setPolicyMode("deterministic");
  }, [initialPresetId, open]);

  async function submit() {
    if (requestInFlightRef.current) {
      return;
    }
    if (presetConfig === null || target === null) {
      onGlobalError("No evaluation preset is available.");
      return;
    }
    if (preset === null) {
      onGlobalError("Select an evaluation preset.");
      return;
    }
    requestInFlightRef.current = true;
    setIsCreating(true);
    onGlobalError(null);
    try {
      const created = await onCreateEvaluation({
        config: presetConfig,
        courseIds: target.courseIds,
        cupIds: target.cupIds,
        difficulties: target.difficulties,
        name: defaultName,
        policyMode,
        repeatsPerTarget: preset.repeatsPerTarget,
        seed: preset.seed,
        sourceArtifact: preset.sourceArtifact,
        sourceRunId: run.id,
        targetMode: target.mode,
        vehicleIds: target.vehicleIds,
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
                  setPresetId(event.target.value as EvaluationPresetId);
                }}
              >
                {presets.map((candidate) => (
                  <option key={candidate.id} value={candidate.id}>
                    {candidate.label}
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
              value={target === null ? "-" : evaluationTargetLabel(target)}
            />
            <PresetLine label="Preset" value={presetSummary(preset, presetConfig)} />
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

function evaluationTargetLabel(target: NonNullable<ReturnType<typeof evaluationTargetFromConfig>>) {
  const mode = target.mode === "gp_course" ? "GP course" : "Time Attack course";
  const parts = [
    target.cupIds.length > 0 ? selectionCountLabel(target.cupIds, "cup") : null,
    target.courseIds.length > 0 ? selectionCountLabel(target.courseIds, "course") : null,
    target.difficulties.length > 0 ? selectionCountLabel(target.difficulties, "difficulty") : null,
    target.vehicleIds.length > 0 ? selectionCountLabel(target.vehicleIds, "vehicle") : null,
  ].filter((part) => part !== null);
  return `${mode} · ${parts.length === 0 ? "all targets" : parts.join(" · ")}`;
}

function presetSummary(preset: EvaluationPreset | null, config: ManagedRunConfig | null) {
  if (preset === null || config === null) {
    return "-";
  }
  return `${preset.sourceArtifact} · ${preset.repeatsPerTarget}x · seed ${preset.seed} · ${config.environment.renderer}`;
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
