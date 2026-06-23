// web/run-manager/src/app/workspace/actions/evaluations.ts
import type { Dispatch, SetStateAction } from "react";

import { upsertEvaluation } from "@/app/workspace/model";
import type { WorkspaceSessions } from "@/app/workspace/sessions";
import {
  cancelEvaluation,
  createEvaluation,
  createEvaluationPreset,
  deleteEvaluation,
  deleteEvaluationPreset,
  startEvaluation,
  updateEvaluation,
} from "@/shared/api/client";
import type {
  CreateEvaluationPresetRequest,
  CreateEvaluationRequest,
  ManagedEvaluation,
  ManagedEvaluationPreset,
  StartEvaluationRequest,
} from "@/shared/api/contract";

interface WorkspaceEvaluationActionsOptions {
  reloadManagerData: (options?: { showLoading?: boolean }) => Promise<void>;
  sessions: WorkspaceSessions;
  setEvaluations: Dispatch<SetStateAction<ManagedEvaluation[]>>;
  setGlobalError: Dispatch<SetStateAction<string | null>>;
}

export interface WorkspaceEvaluationActions {
  cancelManagedEvaluation: (evaluation: ManagedEvaluation) => Promise<ManagedEvaluation>;
  createManagedEvaluation: (request: CreateEvaluationRequest) => Promise<ManagedEvaluation>;
  createManagedEvaluationPreset: (
    request: CreateEvaluationPresetRequest,
  ) => Promise<ManagedEvaluationPreset>;
  removeManagedEvaluation: (evaluation: ManagedEvaluation) => Promise<void>;
  removeManagedEvaluationPreset: (preset: ManagedEvaluationPreset) => Promise<void>;
  renameManagedEvaluation: (evaluationId: string, name: string) => Promise<void>;
  startManagedEvaluation: (
    evaluation: ManagedEvaluation,
    request: StartEvaluationRequest,
  ) => Promise<ManagedEvaluation>;
}

export function workspaceEvaluationActions({
  reloadManagerData,
  sessions,
  setEvaluations,
  setGlobalError,
}: WorkspaceEvaluationActionsOptions): WorkspaceEvaluationActions {
  async function createManagedEvaluation(request: CreateEvaluationRequest) {
    try {
      const evaluation = await createEvaluation(request);
      setEvaluations((current) => upsertEvaluation(current, evaluation));
      await reloadManagerData({ showLoading: false });
      return evaluation;
    } catch (caught) {
      await reloadManagerData();
      throw caught;
    }
  }

  async function createManagedEvaluationPreset(request: CreateEvaluationPresetRequest) {
    try {
      const preset = await createEvaluationPreset(request);
      await reloadManagerData({ showLoading: false });
      return preset;
    } catch (caught) {
      await reloadManagerData();
      throw caught;
    }
  }

  async function removeManagedEvaluationPreset(preset: ManagedEvaluationPreset) {
    try {
      await deleteEvaluationPreset(preset.id);
      await reloadManagerData({ showLoading: false });
    } catch (caught) {
      await reloadManagerData();
      throw caught;
    }
  }

  async function startManagedEvaluation(
    evaluation: ManagedEvaluation,
    request: StartEvaluationRequest,
  ) {
    try {
      const updated = await startEvaluation(evaluation.id, request);
      setEvaluations((current) => upsertEvaluation(current, updated));
      await reloadManagerData({ showLoading: false });
      return updated;
    } catch (caught) {
      await reloadManagerData({ showLoading: false });
      throw caught;
    }
  }

  async function cancelManagedEvaluation(evaluation: ManagedEvaluation) {
    try {
      const updated = await cancelEvaluation(evaluation.id);
      setEvaluations((current) => upsertEvaluation(current, updated));
      await reloadManagerData({ showLoading: false });
      return updated;
    } catch (caught) {
      await reloadManagerData({ showLoading: false });
      throw caught;
    }
  }

  async function removeManagedEvaluation(evaluation: ManagedEvaluation) {
    setGlobalError(null);
    try {
      const deleted = await deleteEvaluation(evaluation.id);
      if (deleted) {
        setEvaluations((current) => current.filter((entry) => entry.id !== evaluation.id));
        sessions.closeEvaluationTabsForEvaluation(evaluation.id);
      }
    } catch (caught) {
      await reloadManagerData();
      throw caught;
    }
  }

  async function renameManagedEvaluation(evaluationId: string, name: string) {
    const updated = await updateEvaluation(evaluationId, { name });
    setEvaluations((current) => upsertEvaluation(current, updated));
    sessions.renameEvaluationTab(evaluationId, updated.name);
  }

  return {
    cancelManagedEvaluation,
    createManagedEvaluation,
    createManagedEvaluationPreset,
    removeManagedEvaluation,
    removeManagedEvaluationPreset,
    renameManagedEvaluation,
    startManagedEvaluation,
  };
}
