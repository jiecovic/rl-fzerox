// web/run-manager/src/pages/evaluations/sections/RecordsPanel.tsx
import type { ManagedEvaluation } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { formatDate } from "@/shared/ui/format";
import { TrashIcon } from "@/shared/ui/icons";
import {
  ListActionsCell,
  ListActionsHeaderCell,
  ListRow,
  ListSelectAllHeaderCell,
  ListSelectionCell,
  ListTable,
  ListTableHead,
} from "@/shared/ui/ListTable";
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
    <ListTable>
      <ListTableHead>
        <tr>
          <ListSelectAllHeaderCell
            aria-label="Select all inactive evaluations"
            checked={allDeletableEvaluationsSelected}
            disabled={isDeletingEvaluation}
            onChange={onSelectAllEvaluations}
          />
          <th className="px-4 py-3">Evaluation</th>
          <th className="px-4 py-3">Checkpoint</th>
          <th className="px-4 py-3">Target</th>
          <th className="px-4 py-3">Progress</th>
          <th className="px-4 py-3">Created</th>
          <ListActionsHeaderCell />
        </tr>
      </ListTableHead>
      <tbody>
        {evaluations.map((evaluation) => {
          const isDeletable = evaluation.status !== "running";
          const selected = selectedEvaluationIds.has(evaluation.id);
          return (
            <ListRow
              key={evaluation.id}
              selected={selected}
              onOpen={() => onOpenEvaluation(evaluation)}
            >
              <ListSelectionCell
                aria-label={`Select evaluation ${evaluation.name}`}
                checked={selected}
                disabled={!isDeletable || isDeletingEvaluation}
                onChange={(checked) => onToggleEvaluationSelection(evaluation.id, checked)}
              />
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
              <ListActionsCell>
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
              </ListActionsCell>
            </ListRow>
          );
        })}
      </tbody>
    </ListTable>
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
