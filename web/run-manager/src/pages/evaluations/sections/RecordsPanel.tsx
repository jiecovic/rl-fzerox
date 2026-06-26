// web/run-manager/src/pages/evaluations/sections/RecordsPanel.tsx
import type { ManagedEvaluation } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { cn } from "@/shared/ui/cn";
import { formatDate } from "@/shared/ui/format";
import { TrashIcon } from "@/shared/ui/icons";
import { Notice } from "@/shared/ui/Panel";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface RecordsPanelProps {
  allDeletableEvaluationsSelected: boolean;
  deletingEvaluationId: string | null;
  evaluationError: string | null;
  evaluations: ManagedEvaluation[];
  isDeletingEvaluation: boolean;
  selectedEvaluationCount: number;
  selectedEvaluationIds: ReadonlySet<string>;
  onDeleteEvaluation: (evaluation: ManagedEvaluation) => void;
  onOpenEvaluation: (evaluation: ManagedEvaluation) => void;
  onRequestSelectedDelete: () => void;
  onSelectAllEvaluations: (selected: boolean) => void;
  onToggleEvaluationSelection: (evaluationId: string, selected: boolean) => void;
}

const TARGET_MODE_LABELS: Record<ManagedEvaluation["target"]["mode"], string> = {
  gp_course: "GP course",
  time_attack_course: "Time Attack course",
};

export function RecordsPanel({
  allDeletableEvaluationsSelected,
  deletingEvaluationId,
  evaluationError,
  evaluations,
  isDeletingEvaluation,
  selectedEvaluationCount,
  selectedEvaluationIds,
  onDeleteEvaluation,
  onOpenEvaluation,
  onRequestSelectedDelete,
  onSelectAllEvaluations,
  onToggleEvaluationSelection,
}: RecordsPanelProps) {
  if (evaluationError !== null) {
    return <Notice tone="error">Evaluation records are unavailable. See the global error.</Notice>;
  }
  return evaluations.length === 0 ? (
    <Notice>
      No evaluation records yet. Open a run and use the evaluation action to create one.
    </Notice>
  ) : (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-app-muted">
          {selectedEvaluationCount === 0
            ? "No evaluations selected"
            : `${selectedEvaluationCount} evaluation${
                selectedEvaluationCount === 1 ? "" : "s"
              } selected`}
        </div>
        <Button
          className="gap-2"
          disabled={selectedEvaluationCount === 0 || isDeletingEvaluation}
          tone="danger"
          type="button"
          onClick={onRequestSelectedDelete}
        >
          <TrashIcon />
          <span>
            {selectedEvaluationCount === 0
              ? "Delete selected"
              : `Delete selected (${selectedEvaluationCount})`}
          </span>
        </Button>
      </div>
      <EvaluationTable
        allDeletableEvaluationsSelected={allDeletableEvaluationsSelected}
        deletingEvaluationId={deletingEvaluationId}
        evaluations={evaluations}
        isDeletingEvaluation={isDeletingEvaluation}
        selectedEvaluationIds={selectedEvaluationIds}
        onDeleteEvaluation={onDeleteEvaluation}
        onOpenEvaluation={onOpenEvaluation}
        onSelectAllEvaluations={onSelectAllEvaluations}
        onToggleEvaluationSelection={onToggleEvaluationSelection}
      />
    </div>
  );
}

function EvaluationTable({
  allDeletableEvaluationsSelected,
  deletingEvaluationId,
  evaluations,
  isDeletingEvaluation,
  selectedEvaluationIds,
  onDeleteEvaluation,
  onOpenEvaluation,
  onSelectAllEvaluations,
  onToggleEvaluationSelection,
}: Omit<
  RecordsPanelProps,
  "evaluationError" | "onRequestSelectedDelete" | "selectedEvaluationCount"
>) {
  return (
    <div className="overflow-x-auto border border-app-border bg-app-surface">
      <table className="w-full min-w-[760px] border-collapse text-left text-sm">
        <thead className="border-b border-app-border text-xs font-bold tracking-[0.04em] text-app-muted uppercase">
          <tr>
            <th className="w-10 px-4 py-3">
              <label className="grid place-items-center" data-evaluation-row-interaction>
                <input
                  aria-label="Select all inactive evaluations"
                  checked={allDeletableEvaluationsSelected}
                  className={evaluationCheckboxClass}
                  disabled={isDeletingEvaluation}
                  type="checkbox"
                  onChange={(event) => onSelectAllEvaluations(event.currentTarget.checked)}
                />
              </label>
            </th>
            <th className="px-4 py-3">Evaluation</th>
            <th className="px-4 py-3">Checkpoint</th>
            <th className="px-4 py-3">Target</th>
            <th className="px-4 py-3">Progress</th>
            <th className="px-4 py-3">Created</th>
            <th className="w-12 px-4 py-3">
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {evaluations.map((evaluation) => {
            const isDeletable = evaluation.status !== "running";
            const selected = selectedEvaluationIds.has(evaluation.id);
            return (
              <tr
                className={evaluationRowClass(selected)}
                key={evaluation.id}
                tabIndex={0}
                onClick={(event) => {
                  if (isEvaluationRowInteractionTarget(event.target)) {
                    return;
                  }
                  onOpenEvaluation(evaluation);
                }}
                onKeyDown={(event) => {
                  if (event.target !== event.currentTarget) {
                    return;
                  }
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onOpenEvaluation(evaluation);
                  }
                }}
              >
                <td className="px-4 py-3 align-top" data-evaluation-row-interaction>
                  <label className="grid place-items-center">
                    <input
                      aria-label={`Select evaluation ${evaluation.name}`}
                      checked={selected}
                      className={evaluationCheckboxClass}
                      disabled={!isDeletable || isDeletingEvaluation}
                      type="checkbox"
                      onChange={(event) =>
                        onToggleEvaluationSelection(evaluation.id, event.currentTarget.checked)
                      }
                    />
                  </label>
                </td>
                <td className="px-4 py-3 align-top">
                  <div className="grid gap-1">
                    <strong className="text-app-text">{evaluation.name}</strong>
                    <span className="text-xs capitalize text-app-muted">{evaluation.status}</span>
                  </div>
                </td>
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
                    <span>{targetRuntimeLabel(evaluation.target)}</span>
                    <span className="text-xs">{targetSelectionLabel(evaluation.target)}</span>
                  </div>
                </td>
                <td className="px-4 py-3 align-top">
                  <span className="text-app-muted">{evaluationExecutionLabel(evaluation)}</span>
                </td>
                <td className="px-4 py-3 align-top text-app-muted">
                  {formatDate(evaluation.created_at)}
                </td>
                <td className="px-4 py-3 align-top" data-evaluation-row-interaction>
                  <TooltipIconButton
                    aria-label={`Delete evaluation ${evaluation.name}`}
                    disabled={
                      !isDeletable || deletingEvaluationId === evaluation.id || isDeletingEvaluation
                    }
                    size="compact"
                    tone="danger"
                    tooltip="Delete evaluation"
                    onClick={() => onDeleteEvaluation(evaluation)}
                  >
                    <TrashIcon />
                  </TooltipIconButton>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function targetSelectionLabel(evaluationTarget: ManagedEvaluation["target"]) {
  const parts = [
    selectionCountLabel(evaluationTarget.cup_ids, "cup"),
    selectionCountLabel(evaluationTarget.course_ids, "course"),
    difficultySelectionLabel(evaluationTarget.difficulties),
    selectionCountLabel(evaluationTarget.vehicle_ids, "vehicle"),
  ].filter((part) => part !== null);
  return parts.length === 0 ? "all targets" : parts.join(" · ");
}

function targetRuntimeLabel(evaluationTarget: ManagedEvaluation["target"]) {
  const variants =
    evaluationTarget.mode === "gp_course" && evaluationTarget.baseline_variant_count > 1
      ? ` · ${evaluationTarget.baseline_variant_count} variants`
      : "";
  return `${TARGET_MODE_LABELS[evaluationTarget.mode]} · ${evaluationTarget.repeats_per_target}x${variants}`;
}

function difficultySelectionLabel(difficulties: readonly string[]) {
  if (difficulties.length === 0) {
    return null;
  }
  return difficulties.map(titleLabel).join(", ");
}

function selectionCountLabel(values: readonly string[], singular: string) {
  if (values.length === 0) {
    return null;
  }
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

function formatStepCount(value: number | null) {
  return value === null ? "step unknown" : `${value.toLocaleString()} steps`;
}

function evaluationExecutionLabel(evaluation: ManagedEvaluation) {
  const progress = evaluationProgressLabel(evaluation);
  if (evaluation.status === "created") {
    return "not started";
  }
  if (progress !== null) {
    return `${evaluation.status} · ${progress}`;
  }
  return evaluation.error_message === null
    ? evaluation.status
    : `${evaluation.status} · ${evaluation.error_message}`;
}

function evaluationProgressLabel(evaluation: ManagedEvaluation) {
  const { completed_attempts: completed, total_attempts: total } = evaluation.progress;
  if (total !== null && total > 0) {
    return `${completed}/${total} course runs`;
  }
  return completed > 0 ? `${completed} course runs` : null;
}

const evaluationCheckboxClass = "h-4 w-4 accent-app-accent";

function evaluationRowClass(selected: boolean) {
  return cn(
    "cursor-pointer border-b border-app-border transition-colors last:border-b-0 hover:bg-app-surface-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-app-accent",
    selected ? "bg-app-surface-muted" : undefined,
  );
}

function isEvaluationRowInteractionTarget(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    target.closest("[data-evaluation-row-interaction],a,button,input,label,select,textarea") !==
      null
  );
}
