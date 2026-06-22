// web/run-manager/src/pages/evaluations/EvaluationsPanel.tsx
import { useEffect, useMemo, useState } from "react";

import type {
  ConfigMetadata,
  CreateEvaluationRequest,
  EvaluationMode,
  EvaluationSourceArtifact,
  ManagedEvaluation,
  ManagedRun,
  PolicyPlaybackMode,
} from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { FieldInput, FieldSelect, FieldShell } from "@/shared/ui/Field";
import { formatDate } from "@/shared/ui/format";
import { EvaluationTabIcon, PlusIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

interface EvaluationsPanelProps {
  evaluations: ManagedEvaluation[];
  metadata: ConfigMetadata | null;
  runs: ManagedRun[];
  sourceRunId: string | null;
  onCreateEvaluation: (request: CreateEvaluationRequest) => Promise<ManagedEvaluation>;
  onGlobalError: (message: string | null) => void;
}

type SourceArtifactChoice = Extract<EvaluationSourceArtifact, "latest" | "best">;

const TARGET_MODE_LABELS: Record<EvaluationMode, string> = {
  best_of: "Best of",
  career_target: "Career target",
  gp_cup: "GP cup",
  time_attack: "Time attack",
};

export function EvaluationsPanel({
  evaluations,
  metadata,
  onCreateEvaluation,
  onGlobalError,
  runs,
  sourceRunId,
}: EvaluationsPanelProps) {
  const [selectedRunId, setSelectedRunId] = useState(runs[0]?.id ?? "");
  const [sourceArtifact, setSourceArtifact] = useState<SourceArtifactChoice>("latest");
  const [policyMode, setPolicyMode] = useState<PolicyPlaybackMode>("deterministic");
  const [targetMode, setTargetMode] = useState<EvaluationMode>("time_attack");
  const [courseIds, setCourseIds] = useState<string[]>([]);
  const [cupIds, setCupIds] = useState<string[]>([]);
  const [difficulties, setDifficulties] = useState<string[]>([]);
  const [vehicleIds, setVehicleIds] = useState<string[]>([]);
  const [repeatText, setRepeatText] = useState("3");
  const [seedText, setSeedText] = useState(() => String(randomSeed()));
  const [nameText, setNameText] = useState("");
  const [nameEdited, setNameEdited] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;
  const defaultName = useMemo(
    () =>
      selectedRun === null
        ? "evaluation"
        : `${selectedRun.name} ${TARGET_MODE_LABELS[targetMode].toLowerCase()}`,
    [selectedRun, targetMode],
  );
  const courses = metadata?.built_in_courses ?? [];
  const cups = metadata?.track_cups ?? [];
  const difficultyOptions = metadata?.gp_difficulties ?? [];
  const vehicles = metadata?.vehicles ?? [];

  useEffect(() => {
    if (runs.length === 0) {
      setSelectedRunId("");
      return;
    }
    if (!runs.some((run) => run.id === selectedRunId)) {
      setSelectedRunId(runs[0]?.id ?? "");
    }
  }, [runs, selectedRunId]);

  useEffect(() => {
    if (sourceRunId !== null && runs.some((run) => run.id === sourceRunId)) {
      setSelectedRunId(sourceRunId);
      setNameEdited(false);
    }
  }, [runs, sourceRunId]);

  useEffect(() => {
    if (!nameEdited) {
      setNameText(defaultName);
    }
  }, [defaultName, nameEdited]);

  async function submitEvaluation() {
    const repeatsPerTarget = Number.parseInt(repeatText, 10);
    const seed = Number.parseInt(seedText, 10);
    const name = nameText.trim() || defaultName;
    if (selectedRun === null) {
      onGlobalError("Select a source run before creating an evaluation.");
      return;
    }
    if (!Number.isInteger(repeatsPerTarget) || repeatsPerTarget < 1) {
      onGlobalError("Evaluation repeats must be a positive integer.");
      return;
    }
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      onGlobalError("Evaluation seed must be an integer from 0 to 4294967295.");
      return;
    }
    setIsCreating(true);
    onGlobalError(null);
    try {
      await onCreateEvaluation({
        courseIds,
        cupIds,
        difficulties,
        name,
        policyMode,
        repeatsPerTarget,
        seed,
        sourceArtifact,
        sourceRunId: selectedRun.id,
        targetMode,
        vehicleIds,
      });
      setNameEdited(false);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to create evaluation");
    } finally {
      setIsCreating(false);
    }
  }

  return (
    <Panel>
      <div className="panel-header-row">
        <PanelHeader
          title="Evaluations"
          subtitle="Freeze policy checkpoints for reproducible headless evaluation runs."
        />
      </div>

      {runs.length === 0 ? (
        <Notice>Create or import a run before creating evaluation snapshots.</Notice>
      ) : (
        <div className="grid gap-5">
          <section className="border border-app-border bg-app-surface-muted p-4">
            <div className="mb-4 flex items-center gap-2 text-sm font-bold tracking-[0.04em] text-app-muted uppercase">
              <EvaluationTabIcon />
              <span>Snapshot source</span>
            </div>
            <div className="grid gap-3 xl:grid-cols-[minmax(220px,1fr)_150px_160px_minmax(260px,1fr)]">
              <FieldShell>
                <span>Policy run</span>
                <FieldSelect
                  value={selectedRunId}
                  onChange={(event) => setSelectedRunId(event.currentTarget.value)}
                >
                  {runs.map((run) => (
                    <option key={run.id} value={run.id}>
                      {run.name}
                    </option>
                  ))}
                </FieldSelect>
              </FieldShell>
              <FieldShell>
                <span>Artifact</span>
                <FieldSelect
                  value={sourceArtifact}
                  onChange={(event) =>
                    setSourceArtifact(event.currentTarget.value as SourceArtifactChoice)
                  }
                >
                  <option value="latest">latest</option>
                  <option value="best">best</option>
                </FieldSelect>
              </FieldShell>
              <FieldShell>
                <span>Mode</span>
                <FieldSelect
                  value={policyMode}
                  onChange={(event) =>
                    setPolicyMode(event.currentTarget.value as PolicyPlaybackMode)
                  }
                >
                  <option value="deterministic">deterministic</option>
                  <option value="stochastic">stochastic</option>
                </FieldSelect>
              </FieldShell>
              <FieldShell>
                <span>Snapshot name</span>
                <FieldInput
                  value={nameText}
                  onChange={(event) => {
                    setNameEdited(true);
                    setNameText(event.currentTarget.value);
                  }}
                />
              </FieldShell>
            </div>
          </section>

          <section className="border border-app-border bg-app-surface-muted p-4">
            <div className="mb-4 flex items-center gap-2 text-sm font-bold tracking-[0.04em] text-app-muted uppercase">
              <span>Target config</span>
            </div>
            <div className="grid gap-3 xl:grid-cols-[180px_120px_180px_180px_minmax(220px,1fr)_180px_auto]">
              <FieldShell>
                <span>Target</span>
                <FieldSelect
                  value={targetMode}
                  onChange={(event) => setTargetMode(event.currentTarget.value as EvaluationMode)}
                >
                  {Object.entries(TARGET_MODE_LABELS).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </FieldSelect>
              </FieldShell>
              <FieldShell>
                <span>Repeats</span>
                <FieldInput
                  min={1}
                  type="number"
                  value={repeatText}
                  onChange={(event) => setRepeatText(event.currentTarget.value)}
                />
              </FieldShell>
              <MultiSelectField
                label="Cups"
                options={cups.map((cup) => ({ label: cup.label, value: cup.id }))}
                values={cupIds}
                onChange={setCupIds}
              />
              <MultiSelectField
                label="Difficulty"
                options={difficultyOptions}
                values={difficulties}
                onChange={setDifficulties}
              />
              <MultiSelectField
                label="Courses"
                options={courses.map((course) => ({
                  label: `${course.display_name} · ${course.cup_label}`,
                  value: course.id,
                }))}
                values={courseIds}
                onChange={setCourseIds}
              />
              <MultiSelectField
                label="Vehicles"
                options={vehicles.map((vehicle) => ({
                  label: vehicle.display_name,
                  value: vehicle.id,
                }))}
                values={vehicleIds}
                onChange={setVehicleIds}
              />
              <FieldShell>
                <span>Seed</span>
                <FieldInput
                  min={0}
                  type="number"
                  value={seedText}
                  onChange={(event) => setSeedText(event.currentTarget.value)}
                />
              </FieldShell>
              <Button
                className="mt-[22px] gap-2"
                disabled={isCreating}
                type="button"
                variant="primary"
                onClick={() => void submitEvaluation()}
              >
                <PlusIcon />
                <span>{isCreating ? "Creating" : "Create"}</span>
              </Button>
            </div>
          </section>

          {evaluations.length === 0 ? (
            <Notice>No evaluation snapshots yet.</Notice>
          ) : (
            <EvaluationTable evaluations={evaluations} />
          )}
        </div>
      )}
    </Panel>
  );
}

function MultiSelectField({
  label,
  onChange,
  options,
  values,
}: {
  label: string;
  options: readonly { label: string; value: string }[];
  values: readonly string[];
  onChange: (values: string[]) => void;
}) {
  return (
    <FieldShell>
      <span>{label}</span>
      <FieldSelect
        className="h-[102px] py-2 text-left normal-case"
        multiple
        size={Math.min(4, Math.max(2, options.length))}
        value={[...values]}
        onChange={(event) =>
          onChange(Array.from(event.currentTarget.selectedOptions, (option) => option.value))
        }
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </FieldSelect>
      <span className="text-[11px] text-app-muted">none selected = all</span>
    </FieldShell>
  );
}

function EvaluationTable({ evaluations }: { evaluations: ManagedEvaluation[] }) {
  return (
    <div className="overflow-x-auto border border-app-border bg-app-surface">
      <table className="w-full min-w-[980px] border-collapse text-left text-sm">
        <thead className="border-b border-app-border text-xs font-bold tracking-[0.04em] text-app-muted uppercase">
          <tr>
            <th className="px-4 py-3">Evaluation</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3">Checkpoint</th>
            <th className="px-4 py-3">Target</th>
            <th className="px-4 py-3">Created</th>
            <th className="px-4 py-3">Directory</th>
          </tr>
        </thead>
        <tbody>
          {evaluations.map((evaluation) => (
            <tr className="border-b border-app-border last:border-b-0" key={evaluation.id}>
              <td className="px-4 py-3 align-top">
                <div className="grid gap-1">
                  <strong className="text-app-text">{evaluation.name}</strong>
                  <span className="font-mono text-xs text-app-muted">{evaluation.id}</span>
                </div>
              </td>
              <td className="px-4 py-3 align-top capitalize text-app-muted">{evaluation.status}</td>
              <td className="px-4 py-3 align-top">
                <div className="grid gap-1 text-app-muted">
                  <span>
                    {evaluation.checkpoint.source_run_name ?? evaluation.source_run_id ?? "-"} ·{" "}
                    {evaluation.checkpoint.artifact}
                  </span>
                  <span className="text-xs">
                    {formatStepCount(evaluation.checkpoint.lineage_num_timesteps)}
                  </span>
                </div>
              </td>
              <td className="px-4 py-3 align-top text-app-muted">
                <div className="grid gap-1">
                  <span>
                    {TARGET_MODE_LABELS[evaluation.target.mode]} ·{" "}
                    {evaluation.target.repeats_per_target}x
                  </span>
                  <span className="text-xs">{targetSelectionLabel(evaluation.target)}</span>
                </div>
              </td>
              <td className="px-4 py-3 align-top text-app-muted">
                {formatDate(evaluation.created_at)}
              </td>
              <td className="px-4 py-3 align-top font-mono text-xs text-app-muted">
                {evaluation.evaluation_dir}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function randomSeed() {
  return Math.floor(Math.random() * 0x100000000);
}

function targetSelectionLabel(evaluationTarget: ManagedEvaluation["target"]) {
  const parts = [
    selectionCountLabel(evaluationTarget.cup_ids, "cup"),
    selectionCountLabel(evaluationTarget.course_ids, "course"),
    selectionCountLabel(evaluationTarget.difficulties, "difficulty"),
    selectionCountLabel(evaluationTarget.vehicle_ids, "vehicle"),
  ].filter((part) => part !== null);
  return parts.length === 0 ? "all targets" : parts.join(" · ");
}

function selectionCountLabel(values: readonly string[], singular: string) {
  if (values.length === 0) {
    return null;
  }
  return `${values.length} ${singular}${values.length === 1 ? "" : "s"}`;
}

function formatStepCount(value: number | null) {
  return value === null ? "step unknown" : `${value.toLocaleString()} steps`;
}
