// web/run-manager/src/pages/evaluations/sections/PresetConfigPanel.tsx
import {
  type EvaluationPreset,
  type EvaluationPresetId,
  type EvaluationPresetOverride,
  type EvaluationTargetDraft,
  randomEvaluationSeed,
} from "@/entities/evaluation/model/presets";
import type { ConfigSetter } from "@/entities/runConfig/model/state";
import { TracksSection } from "@/entities/runConfig/ui/sections/tracks/TracksSection";
import { VehicleSection } from "@/entities/runConfig/ui/sections/VehicleSection";
import type {
  ConfigMetadata,
  EvaluationMode,
  EvaluationSourceArtifact,
  ManagedRun,
  ManagedRunConfig,
} from "@/shared/api/contract";
import { rendererNames } from "@/shared/api/renderers";
import { Button } from "@/shared/ui/Button";
import { ConfigStack } from "@/shared/ui/config/ConfigLayout";
import { FieldInput, FieldSelect, FieldShell } from "@/shared/ui/Field";
import { ImportIcon, PlusIcon, RandomizeIcon, TrashIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface PresetConfigPanelProps {
  defaultConfig: ManagedRunConfig;
  importRunId: string;
  importingPreset: boolean;
  metadata: ConfigMetadata;
  presetConfig: ManagedRunConfig;
  presetId: EvaluationPresetId;
  presets: EvaluationPreset[];
  runs: ManagedRun[];
  selectedTarget: EvaluationTargetDraft | null;
  selectedPreset: EvaluationPreset | null;
  setEvaluationConfig: ConfigSetter;
  onCreatePreset: () => void;
  onDeletePreset: () => void;
  onImportRunChange: (runId: string) => void;
  onImportRunPreset: () => void;
  onPresetChange: (value: EvaluationPresetId) => void;
  onPresetSettingsChange: (patch: EvaluationPresetOverride) => void;
}

const TARGET_MODE_LABELS: Record<EvaluationMode, string> = {
  gp_course: "GP course",
  time_attack_course: "Time Attack course",
};

export function PresetConfigPanel({
  defaultConfig,
  importRunId,
  importingPreset,
  metadata,
  presetConfig,
  presetId,
  presets,
  runs,
  selectedTarget,
  selectedPreset,
  setEvaluationConfig,
  onCreatePreset,
  onDeletePreset,
  onImportRunChange,
  onImportRunPreset,
  onPresetChange,
  onPresetSettingsChange,
}: PresetConfigPanelProps) {
  const rendererOptions = rendererNames(metadata, presetConfig.environment.renderer);
  return (
    <div className="grid gap-5">
      <section className="border border-app-border bg-app-surface-muted p-4">
        <div className="grid gap-3 xl:grid-cols-[minmax(240px,1fr)_minmax(260px,1fr)_auto] xl:items-end">
          <FieldShell>
            <span>Preset</span>
            <FieldSelect
              value={presetId}
              onChange={(event) => onPresetChange(event.target.value as EvaluationPresetId)}
            >
              {presets.map((preset) => (
                <option key={preset.id} value={preset.id}>
                  {preset.label}
                </option>
              ))}
            </FieldSelect>
          </FieldShell>
          <FieldShell>
            <span>Name</span>
            <FieldInput
              disabled={selectedPreset === null}
              value={selectedPreset?.label ?? ""}
              onChange={(event) =>
                onPresetSettingsChange({ label: event.currentTarget.value.trimStart() })
              }
            />
          </FieldShell>
          <div className="flex flex-wrap items-end gap-2">
            <Button
              className="gap-2"
              disabled={selectedPreset === null}
              type="button"
              onClick={onCreatePreset}
            >
              <PlusIcon />
              <span>New preset</span>
            </Button>
            <Button
              className="gap-2"
              disabled={selectedPreset === null || selectedPreset.builtin}
              tone="danger"
              type="button"
              onClick={onDeletePreset}
            >
              <TrashIcon />
              <span>Delete</span>
            </Button>
          </div>
        </div>
        <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-[140px_180px_120px_220px]">
          <FieldShell>
            <span>Artifact</span>
            <FieldSelect
              disabled={selectedPreset === null}
              value={selectedPreset?.sourceArtifact ?? "latest"}
              onChange={(event) =>
                onPresetSettingsChange({
                  sourceArtifact: event.currentTarget.value as EvaluationSourceArtifact,
                })
              }
            >
              <option value="latest">latest</option>
              <option value="best">best</option>
              <option value="final">final</option>
            </FieldSelect>
          </FieldShell>
          <FieldShell>
            <span>Renderer</span>
            <FieldSelect
              value={presetConfig.environment.renderer}
              onChange={(event) =>
                setEvaluationConfig((config) => ({
                  ...config,
                  environment: {
                    ...config.environment,
                    renderer: event.currentTarget
                      .value as ManagedRunConfig["environment"]["renderer"],
                  },
                }))
              }
            >
              {rendererOptions.map((renderer) => (
                <option key={renderer} value={renderer}>
                  {renderer}
                </option>
              ))}
            </FieldSelect>
          </FieldShell>
          <FieldShell>
            <span>Repeats</span>
            <FieldInput
              disabled={selectedPreset === null}
              max={1000}
              min={1}
              type="number"
              value={selectedPreset?.repeatsPerTarget ?? 1}
              onChange={(event) =>
                onPresetSettingsChange({
                  repeatsPerTarget: clampInteger(event.currentTarget.valueAsNumber, 1, 1000),
                })
              }
            />
          </FieldShell>
          <div className="grid grid-cols-[minmax(0,1fr)_40px] items-end gap-2">
            <FieldShell>
              <span>Seed</span>
              <FieldInput
                disabled={selectedPreset === null}
                max={4_294_967_295}
                min={0}
                type="number"
                value={selectedPreset?.seed ?? 0}
                onChange={(event) =>
                  onPresetSettingsChange({
                    seed: clampInteger(event.currentTarget.valueAsNumber, 0, 4_294_967_295),
                  })
                }
              />
            </FieldShell>
            <TooltipIconButton
              aria-label="Randomize evaluation seed"
              disabled={selectedPreset === null}
              tooltip="Randomize seed"
              onClick={() => onPresetSettingsChange({ seed: randomEvaluationSeed() })}
            >
              <RandomizeIcon />
            </TooltipIconButton>
          </div>
        </div>
        <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(260px,1fr)_auto]">
          <FieldShell>
            <span>Import preset from run</span>
            <FieldSelect
              value={importRunId}
              onChange={(event) => onImportRunChange(event.currentTarget.value)}
            >
              <option value="">Select run</option>
              {runs.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.name}
                </option>
              ))}
            </FieldSelect>
          </FieldShell>
          <div className="flex items-end">
            <Button
              className="gap-2"
              disabled={importRunId.length === 0 || importingPreset}
              type="button"
              onClick={onImportRunPreset}
            >
              <ImportIcon />
              <span>{importingPreset ? "Importing" : "Import"}</span>
            </Button>
          </div>
        </div>
        <div className="mt-3 text-sm text-app-muted">
          Target: {selectedTarget === null ? "not configured" : targetDraftLabel(selectedTarget)}
        </div>
      </section>
      <ConfigStack>
        <TracksSection
          config={presetConfig}
          defaultConfig={defaultConfig}
          gpDifficultySelection="single"
          metadata={metadata}
          setConfig={setEvaluationConfig}
          showSampling={false}
        />
        <VehicleSection
          config={presetConfig}
          defaultConfig={defaultConfig}
          metadata={metadata}
          setConfig={setEvaluationConfig}
          showEngineControls={false}
        />
      </ConfigStack>
    </div>
  );
}

function targetDraftLabel(evaluationTarget: EvaluationTargetDraft) {
  return `${TARGET_MODE_LABELS[evaluationTarget.mode]} · ${targetSelectionPartsLabel(evaluationTarget)}`;
}

function targetSelectionPartsLabel({
  courseIds,
  cupIds,
  difficulties,
  vehicleIds,
}: {
  courseIds: readonly string[];
  cupIds: readonly string[];
  difficulties: readonly string[];
  vehicleIds: readonly string[];
}) {
  const parts = [
    selectionCountLabel(cupIds, "cup"),
    selectionCountLabel(courseIds, "course"),
    selectionCountLabel(difficulties, "difficulty"),
    selectionCountLabel(vehicleIds, "vehicle"),
  ].filter((part) => part !== null);
  return parts.length === 0 ? "all targets" : parts.join(" · ");
}

function selectionCountLabel(values: readonly string[], singular: string) {
  if (values.length === 0) {
    return null;
  }
  return `${values.length} ${pluralize(values.length, singular)}`;
}

function pluralize(count: number, singular: string) {
  if (count === 1) {
    return singular;
  }
  return singular === "difficulty" ? "difficulties" : `${singular}s`;
}

function clampInteger(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, Math.trunc(value)));
}
