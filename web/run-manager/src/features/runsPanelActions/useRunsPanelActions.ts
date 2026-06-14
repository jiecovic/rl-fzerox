// web/run-manager/src/features/runsPanelActions/useRunsPanelActions.ts
import { useState } from "react";
import type { PendingDelete, RunLineageGroup } from "@/entities/runLineage/model/types";
import type { ManagedRun } from "@/shared/api/contract";

interface RunsPanelActionOptions {
  onDeleteLineage: (lineageId: string) => Promise<void>;
  onDeleteRun: (run: ManagedRun) => Promise<void>;
  onGlobalError: (message: string | null) => void;
}

export function useRunsPanelActions({
  onDeleteLineage,
  onDeleteRun,
  onGlobalError,
}: RunsPanelActionOptions) {
  const [pendingDelete, setPendingDelete] = useState<PendingDelete | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [busyActionRunId, setBusyActionRunId] = useState<string | null>(null);

  async function runAction(runId: string, callback: () => Promise<void>) {
    onGlobalError(null);
    setBusyActionRunId(runId);
    try {
      await callback();
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "run action failed");
    } finally {
      setBusyActionRunId((current) => (current === runId ? null : current));
    }
  }

  async function confirmDelete() {
    if (pendingDelete === null) {
      return;
    }
    onGlobalError(null);
    setIsDeleting(true);
    try {
      if (pendingDelete.kind === "lineage") {
        await onDeleteLineage(pendingDelete.lineage.id);
      } else {
        await onDeleteRun(pendingDelete.run);
      }
      setPendingDelete(null);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "delete failed");
    } finally {
      setIsDeleting(false);
    }
  }

  function requestLineageDelete(lineage: RunLineageGroup) {
    onGlobalError(null);
    setPendingDelete({ kind: "lineage", lineage });
  }

  function requestRunDelete(run: ManagedRun) {
    onGlobalError(null);
    setPendingDelete({ kind: "run", run });
  }

  function closePendingDelete() {
    if (!isDeleting) {
      setPendingDelete(null);
    }
  }

  return {
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
