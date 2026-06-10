// web/run-manager/src/features/runsPanelActions/useRunsPanelActions.ts
import { useState } from "react";
import type { PendingDelete, RunLineageGroup } from "@/entities/runLineage/model/types";
import type { ManagedRun } from "@/shared/api/contract";

interface RunsPanelActionOptions {
  onDeleteLineage: (lineageId: string) => Promise<void>;
  onDeleteRun: (run: ManagedRun) => Promise<void>;
}

export function useRunsPanelActions({ onDeleteLineage, onDeleteRun }: RunsPanelActionOptions) {
  const [pendingDelete, setPendingDelete] = useState<PendingDelete | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [busyActionRunId, setBusyActionRunId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function runAction(runId: string, callback: () => Promise<void>) {
    setActionError(null);
    setBusyActionRunId(runId);
    try {
      await callback();
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "run action failed");
    } finally {
      setBusyActionRunId((current) => (current === runId ? null : current));
    }
  }

  async function confirmDelete() {
    if (pendingDelete === null) {
      return;
    }
    setActionError(null);
    setIsDeleting(true);
    try {
      if (pendingDelete.kind === "lineage") {
        await onDeleteLineage(pendingDelete.lineage.id);
      } else {
        await onDeleteRun(pendingDelete.run);
      }
      setPendingDelete(null);
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "delete failed");
    } finally {
      setIsDeleting(false);
    }
  }

  function requestLineageDelete(lineage: RunLineageGroup) {
    setActionError(null);
    setPendingDelete({ kind: "lineage", lineage });
  }

  function requestRunDelete(run: ManagedRun) {
    setActionError(null);
    setPendingDelete({ kind: "run", run });
  }

  function closePendingDelete() {
    if (!isDeleting) {
      setPendingDelete(null);
    }
  }

  return {
    actionError,
    busyActionRunId,
    closePendingDelete,
    confirmDelete,
    isDeleting,
    pendingDelete,
    requestLineageDelete,
    requestRunDelete,
    runAction,
  };
}
