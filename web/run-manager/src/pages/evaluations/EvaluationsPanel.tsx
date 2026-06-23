// web/run-manager/src/pages/evaluations/EvaluationsPanel.tsx
import { useEffect, useMemo, useState } from "react";

import { PresetConfigPanel } from "@/pages/evaluations/sections/PresetConfigPanel";
import { RecordsPanel } from "@/pages/evaluations/sections/RecordsPanel";
import type {
  EvaluationBaselineSuite,
  ManagedEvaluation,
  ManagedEvaluationPreset,
} from "@/shared/api/contract";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { Panel, PanelHeader } from "@/shared/ui/Panel";
import { type TabItem, Tabs } from "@/shared/ui/Tabs";

interface EvaluationsPanelProps {
  evaluationBaselineSuites: EvaluationBaselineSuite[];
  evaluationError: string | null;
  evaluations: ManagedEvaluation[];
  evaluationPresets: ManagedEvaluationPreset[];
  onDeleteEvaluation: (evaluation: ManagedEvaluation) => Promise<void>;
  onGlobalError: (message: string | null) => void;
  onOpenEvaluation: (evaluation: ManagedEvaluation) => void;
}

type EvaluationWorkspaceTab = "records" | "preset_config";

const EVALUATION_WORKSPACE_TABS: readonly TabItem<EvaluationWorkspaceTab>[] = (
  [
    ["records", "Records"],
    ["preset_config", "Presets"],
  ] as const
).map(([id, label]) => ({ id, label }));

export function EvaluationsPanel({
  evaluationBaselineSuites,
  evaluationError,
  evaluations,
  evaluationPresets,
  onDeleteEvaluation,
  onGlobalError,
  onOpenEvaluation,
}: EvaluationsPanelProps) {
  const [deletingEvaluationId, setDeletingEvaluationId] = useState<string | null>(null);
  const [deleteSelectedRequested, setDeleteSelectedRequested] = useState(false);
  const [isDeletingSelected, setIsDeletingSelected] = useState(false);
  const [selectedEvaluationIds, setSelectedEvaluationIds] = useState<ReadonlySet<string>>(
    new Set(),
  );
  const [activeTab, setActiveTab] = useState<EvaluationWorkspaceTab>("records");

  const deletableEvaluations = useMemo(
    () => evaluations.filter((evaluation) => evaluation.status === "created"),
    [evaluations],
  );
  const selectedEvaluations = useMemo(
    () => deletableEvaluations.filter((evaluation) => selectedEvaluationIds.has(evaluation.id)),
    [deletableEvaluations, selectedEvaluationIds],
  );
  const allDeletableEvaluationsSelected =
    deletableEvaluations.length > 0 &&
    deletableEvaluations.every((evaluation) => selectedEvaluationIds.has(evaluation.id));
  const isDeletingEvaluation = deletingEvaluationId !== null || isDeletingSelected;

  useEffect(() => {
    const deletableIds = new Set(deletableEvaluations.map((evaluation) => evaluation.id));
    setSelectedEvaluationIds((current) => {
      const next = new Set([...current].filter((evaluationId) => deletableIds.has(evaluationId)));
      return next.size === current.size ? current : next;
    });
  }, [deletableEvaluations]);

  async function deleteEvaluationRecord(evaluation: ManagedEvaluation) {
    setDeletingEvaluationId(evaluation.id);
    onGlobalError(null);
    try {
      await onDeleteEvaluation(evaluation);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to delete evaluation");
    } finally {
      setDeletingEvaluationId(null);
    }
  }

  function setAllEvaluationsSelected(selected: boolean) {
    setSelectedEvaluationIds(
      selected ? new Set(deletableEvaluations.map(({ id }) => id)) : new Set(),
    );
  }

  function toggleEvaluationSelection(evaluationId: string, selected: boolean) {
    setSelectedEvaluationIds((current) => {
      const next = new Set(current);
      if (selected) {
        next.add(evaluationId);
      } else {
        next.delete(evaluationId);
      }
      return next;
    });
  }

  async function confirmSelectedEvaluationDelete() {
    const targets = selectedEvaluations;
    if (targets.length === 0) {
      setDeleteSelectedRequested(false);
      return;
    }
    setIsDeletingSelected(true);
    onGlobalError(null);
    try {
      for (const evaluation of targets) {
        await onDeleteEvaluation(evaluation);
      }
      setSelectedEvaluationIds(new Set());
      setDeleteSelectedRequested(false);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to delete evaluations");
    } finally {
      setIsDeletingSelected(false);
    }
  }

  return (
    <>
      <Panel>
        <PanelHeader
          title="Evaluations"
          subtitle="Evaluate saved policy checkpoints on persisted benchmark presets."
        />

        <div className="mt-5">
          <div className="section-tabs-row">
            <Tabs
              activeId={activeTab}
              items={EVALUATION_WORKSPACE_TABS}
              label="Evaluation sections"
              variant="section"
              onSelect={setActiveTab}
            />
          </div>

          {activeTab === "records" ? (
            <RecordsPanel
              allDeletableEvaluationsSelected={allDeletableEvaluationsSelected}
              deletingEvaluationId={deletingEvaluationId}
              evaluationError={evaluationError}
              evaluations={evaluations}
              isDeletingEvaluation={isDeletingEvaluation}
              selectedEvaluationCount={selectedEvaluations.length}
              selectedEvaluationIds={selectedEvaluationIds}
              onDeleteEvaluation={(evaluation) => void deleteEvaluationRecord(evaluation)}
              onOpenEvaluation={onOpenEvaluation}
              onRequestSelectedDelete={() => setDeleteSelectedRequested(true)}
              onSelectAllEvaluations={setAllEvaluationsSelected}
              onToggleEvaluationSelection={toggleEvaluationSelection}
            />
          ) : null}

          {activeTab === "preset_config" ? (
            <PresetConfigPanel
              baselineSuites={evaluationBaselineSuites}
              presets={evaluationPresets}
            />
          ) : null}
        </div>
      </Panel>
      <ConfirmDialog
        busy={isDeletingSelected}
        confirmLabel={
          selectedEvaluations.length === 0
            ? "Delete selected"
            : `Delete ${selectedEvaluations.length} evaluations`
        }
        description={`Delete ${selectedEvaluations.length} created evaluation${
          selectedEvaluations.length === 1 ? "" : "s"
        } and their checkpoint copies.`}
        open={deleteSelectedRequested}
        title="Delete selected evaluations"
        onClose={() => setDeleteSelectedRequested(false)}
        onConfirm={() => void confirmSelectedEvaluationDelete()}
      />
    </>
  );
}
