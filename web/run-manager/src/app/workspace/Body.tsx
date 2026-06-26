// web/run-manager/src/app/workspace/Body.tsx
import { useEffect, useMemo, useState } from "react";

import type { WorkspaceActions } from "@/app/workspace/actions";
import { configuratorBaseConfigForDraftEditor } from "@/app/workspace/model";
import type { WorkspaceSessions } from "@/app/workspace/sessions";
import { CheckpointsPanel } from "@/pages/checkpoints/CheckpointsPanel";
import { DraftsPanel } from "@/pages/drafts/DraftsPanel";
import { EvaluationsPanel } from "@/pages/evaluations/EvaluationsPanel";
import { RunsPanel } from "@/pages/runs/RunsPanel";
import { SaveGamesPanel } from "@/pages/saveGames/SaveGamesPanel";
import type {
  CheckpointCatalogResponse,
  ConfigMetadata,
  EvaluationBaselineSuite,
  ManagedDraft,
  ManagedEvaluation,
  ManagedEvaluationPreset,
  ManagedRun,
  ManagedRunConfig,
  ManagedRunDetail,
  ManagedSaveGame,
} from "@/shared/api/contract";
import { FloatingNotice } from "@/shared/ui/FloatingNotice";
import { Notice } from "@/shared/ui/Panel";
import { Configurator } from "@/widgets/configurator/Configurator";
import { EvaluationWorkspace } from "@/widgets/evaluationWorkspace/EvaluationWorkspace";
import { ChartsPanel } from "@/widgets/runCharts/ChartsPanel";
import { RunWorkspace } from "@/widgets/runWorkspace/RunWorkspace";
import { SaveGameWorkspace } from "@/widgets/saveGameWorkspace/SaveGameWorkspace";

interface WorkspaceBodyProps {
  actions: WorkspaceActions;
  defaultConfig: ManagedRunConfig | null;
  drafts: ManagedDraft[];
  checkpointCatalog: CheckpointCatalogResponse | null;
  checkpointCatalogError: string | null;
  error: string | null;
  evaluationBaselineSuites: EvaluationBaselineSuite[];
  evaluationError: string | null;
  evaluations: ManagedEvaluation[];
  evaluationPresets: ManagedEvaluationPreset[];
  isLoading: boolean;
  loadRunDetail: (runId: string) => Promise<ManagedRunDetail>;
  metadata: ConfigMetadata | null;
  runs: ManagedRun[];
  runDetailsById: Record<string, ManagedRunDetail>;
  saveGames: ManagedSaveGame[];
  sessions: WorkspaceSessions;
  onDismissError: () => void;
  onRefreshSaveGameStatus: (saveGameId: string) => Promise<void>;
}

export function WorkspaceBody({
  actions,
  defaultConfig,
  drafts,
  checkpointCatalog,
  checkpointCatalogError,
  error,
  evaluationBaselineSuites,
  evaluationError,
  evaluations,
  evaluationPresets,
  isLoading,
  loadRunDetail,
  metadata,
  runs,
  runDetailsById,
  saveGames,
  sessions,
  onDismissError,
  onRefreshSaveGameStatus,
}: WorkspaceBodyProps) {
  const [runDetailError, setRunDetailError] = useState<string | null>(null);
  const setGlobalError = actions.setGlobalError;
  const activeRunTab = sessions.activeRunTab;
  const activeDraftEditor = sessions.activeDraftEditor;
  const activeEvaluationSession = sessions.activeEvaluationSession;
  const activeSaveGameSession = sessions.activeSaveGameSession;
  const activeEvaluation =
    activeEvaluationSession === null
      ? null
      : (evaluations.find((evaluation) => evaluation.id === activeEvaluationSession.evaluationId) ??
        null);
  const checkpointRuns = useMemo(
    () =>
      checkpointCatalog?.installed_checkpoints.flatMap((checkpoint) => checkpoint.run ?? []) ?? [],
    [checkpointCatalog?.installed_checkpoints],
  );
  const workspaceRuns = useMemo(
    () => mergeWorkspaceRuns(runs, checkpointRuns),
    [checkpointRuns, runs],
  );
  const activeRunSummary =
    activeRunTab === null
      ? null
      : (workspaceRuns.find((candidate) => candidate.id === activeRunTab.runId) ?? null);
  const activeRunDetail =
    activeRunTab === null ? null : (runDetailsById[activeRunTab.runId] ?? null);
  const activeRun = useMemo(
    () =>
      activeRunSummary === null || activeRunDetail === null
        ? null
        : mergeRunDetail(activeRunSummary, activeRunDetail),
    [activeRunDetail, activeRunSummary],
  );

  useEffect(() => {
    if (activeRunTab === null || activeRunDetail !== null) {
      setRunDetailError(null);
      return undefined;
    }
    let ignore = false;
    setRunDetailError(null);
    void loadRunDetail(activeRunTab.runId).catch((caught) => {
      if (!ignore) {
        const message = caught instanceof Error ? caught.message : "failed to load run details";
        setRunDetailError(message);
        setGlobalError(message);
      }
    });
    return () => {
      ignore = true;
    };
  }, [activeRunDetail, activeRunTab, loadRunDetail, setGlobalError]);

  return (
    <div className="workspace">
      {error !== null ? (
        <FloatingNotice tone="error" onDismiss={onDismissError}>
          {error}
        </FloatingNotice>
      ) : null}
      {isLoading ? <Notice>Loading manager data...</Notice> : null}

      {!isLoading && sessions.activeTabId === "drafts" ? (
        <DraftsPanel
          drafts={drafts}
          onCreateDraft={sessions.createNewDraft}
          onDeleteDraft={(draft) => actions.removeDraft(draft.id)}
          onOpenDraft={sessions.openDraft}
        />
      ) : null}
      {!isLoading && sessions.activeTabId === "runs" ? (
        <RunsPanel
          drafts={drafts}
          runs={runs}
          onDeleteLineage={actions.removeLineage}
          onDeleteRun={actions.removeRun}
          onExportRun={actions.exportManagedRun}
          onGlobalError={actions.setGlobalError}
          onImportRunBundle={actions.importManagedRunBundle}
          onOpenRun={sessions.openRun}
          onResumeRun={(run) => actions.resumeManagedRun(run.id)}
          onStopRun={(run) => actions.stopManagedRun(run.id)}
          onUpdateLineageGroups={actions.updateManagedLineageGroups}
        />
      ) : null}
      {!isLoading && sessions.activeTabId === "checkpoints" ? (
        <CheckpointsPanel
          catalog={checkpointCatalog}
          error={checkpointCatalogError}
          onGlobalError={actions.setGlobalError}
          onInstallCheckpoint={actions.installManagedCatalogCheckpoint}
          onDeleteCheckpoint={actions.removeManagedCheckpoint}
          onOpenCheckpoint={(checkpoint) => {
            if (checkpoint.run !== null) {
              sessions.openRun(checkpoint.run);
            }
          }}
        />
      ) : null}
      {!isLoading && sessions.activeTabId === "charts" ? (
        <ChartsPanel
          focusedRunId={sessions.chartsFocusRunId}
          onGlobalError={actions.setGlobalError}
          onOpenRun={sessions.openRun}
          runs={workspaceRuns}
        />
      ) : null}
      {!isLoading && sessions.activeTabId === "evaluations" ? (
        <EvaluationsPanel
          evaluationBaselineSuites={evaluationBaselineSuites}
          evaluationError={evaluationError}
          evaluations={evaluations}
          evaluationPresets={evaluationPresets}
          metadata={metadata}
          onDeleteEvaluation={actions.removeManagedEvaluation}
          onCreateEvaluationPreset={actions.createManagedEvaluationPreset}
          onDeleteEvaluationPreset={actions.removeManagedEvaluationPreset}
          onGlobalError={actions.setGlobalError}
          onOpenEvaluation={sessions.openEvaluation}
        />
      ) : null}
      {!isLoading && activeEvaluationSession !== null ? (
        activeEvaluation === null ? (
          <Notice tone="error">This evaluation is no longer available.</Notice>
        ) : (
          <EvaluationWorkspace
            evaluation={activeEvaluation}
            onCancelEvaluation={actions.cancelManagedEvaluation}
            onGlobalError={actions.setGlobalError}
            onRenameEvaluation={actions.renameManagedEvaluation}
            onStartEvaluation={actions.startManagedEvaluation}
          />
        )
      ) : null}
      {!isLoading && sessions.activeTabId === "save-games" ? (
        <SaveGamesPanel
          saveGames={saveGames}
          onCreateSaveGame={sessions.createNewSaveGame}
          onDeleteSaveGame={actions.removeSaveGame}
          onOpenSaveGame={sessions.openSaveGame}
        />
      ) : null}
      {!isLoading && activeSaveGameSession !== null ? (
        <SaveGameWorkspace
          checkpointCatalog={checkpointCatalog}
          evaluations={evaluations}
          metadata={metadata}
          runs={workspaceRuns}
          saveGame={
            activeSaveGameSession.saveGameId === null
              ? null
              : (saveGames.find((saveGame) => saveGame.id === activeSaveGameSession.saveGameId) ??
                null)
          }
          session={activeSaveGameSession}
          onCreateSaveGame={actions.createManagedSaveGame}
          onGlobalError={actions.setGlobalError}
          onOpenSaveGameDirectory={actions.openManagedSaveGameDirectory}
          onPatchSession={sessions.patchSaveGameSession}
          onImportEngineTuning={actions.importManagedSaveEngineTuning}
          onRenameSaveGame={actions.renameManagedSaveGame}
          onRefreshStatus={onRefreshSaveGameStatus}
          onUpdateRunnerSettings={actions.updateManagedSaveRunnerSettings}
          onUpsertCourseSetup={actions.upsertManagedSaveCourseSetup}
          onUpsertCupSetup={actions.upsertManagedSaveCupSetup}
          onStartCareerMode={actions.startManagedCareerMode}
        />
      ) : null}
      {!isLoading && activeRunTab !== null ? (
        activeRunSummary === null ? (
          <Notice tone="error">This run is no longer available.</Notice>
        ) : metadata === null ? (
          <Notice tone="error">
            Run metadata is unavailable. Reload the run manager; the live run list may still be
            connected while the full manager payload failed to load.
          </Notice>
        ) : activeRun === null ? (
          <Notice>
            {runDetailError === null ? "Loading run details..." : "Run details unavailable."}
          </Notice>
        ) : (
          <RunWorkspace
            allRuns={workspaceRuns}
            metadata={metadata}
            onClearAltBaselines={actions.clearManagedRunAltBaselines}
            onClearCourseAltBaselines={actions.clearManagedRunCourseAltBaselines}
            onCreateDraftFromRun={actions.createDraftFromManagedRun}
            onCreateEvaluation={actions.createManagedEvaluation}
            evaluationPresets={evaluationPresets}
            onFork={actions.forkManagedRun}
            onGlobalError={actions.setGlobalError}
            onOpenDirectory={actions.openManagedRunDirectory}
            onRename={actions.renameManagedRun}
            onResetEngineTuning={actions.resetManagedRunEngineTuning}
            onResetTrackPool={actions.resetManagedRunTrackPool}
            onResume={actions.resumeManagedRun}
            onOpenEvaluation={sessions.openEvaluation}
            onShowCharts={sessions.showRunCharts}
            onStop={actions.stopManagedRun}
            onWatch={actions.watchManagedRun}
            run={activeRun}
          />
        )
      ) : null}
      {!isLoading && activeDraftEditor !== null ? (
        defaultConfig !== null && metadata !== null ? (
          <Configurator
            key={activeDraftEditor.sessionId}
            active
            baseConfig={configuratorBaseConfigForDraftEditor(defaultConfig, activeDraftEditor)}
            existingNames={sessions.reservedNamesForSession(activeDraftEditor.sessionId)}
            forkAltBaselineCount={forkDraftAltBaselineCount(activeDraftEditor.forkSource, runs)}
            forkCopyAltBaselines={activeDraftEditor.forkSource?.copyAltBaselines ?? null}
            forkSourceArtifact={activeDraftEditor.forkSource?.artifact ?? null}
            forkSourceEngineTunerBackend={
              activeDraftEditor.forkSource?.sourceEngineTunerBackend ?? null
            }
            forkSourceEngineTuning={activeDraftEditor.forkSource?.sourceEngineTuning ?? null}
            forkSourceEngineTuningKnown={
              activeDraftEditor.forkSource?.sourceEngineTuningKnown ?? false
            }
            forkSourceRunLabel={sessions.forkSourceRunLabel(activeDraftEditor.forkSource)}
            initialConfig={activeDraftEditor.initialConfig}
            initialDraftName={activeDraftEditor.initialDraftName}
            loadedDraft={activeDraftEditor.loadedDraft}
            metadata={metadata}
            resumeConfig={activeDraftEditor.currentConfig}
            resumeDraftName={activeDraftEditor.currentDraftName}
            onConfigChange={(config) =>
              sessions.patchDraftEditor(activeDraftEditor.sessionId, {
                currentConfig: config,
              })
            }
            onDraftNameChange={(name) => {
              sessions.setDraftEditorTitle(activeDraftEditor.sessionId, name);
              sessions.patchDraftEditor(activeDraftEditor.sessionId, {
                currentDraftName: name,
              });
            }}
            onGlobalError={actions.setGlobalError}
            onLaunchRun={(name, config, draftId, engineTuningSourceAction) =>
              actions.launchTrainingRun(
                activeDraftEditor.sessionId,
                name,
                config,
                draftId,
                engineTuningSourceAction,
              )
            }
            onSaveDraft={(name, config) =>
              actions.saveDraft(activeDraftEditor.sessionId, name, config)
            }
            onUpdateDraft={(id, name, config) =>
              actions.updateExistingDraft(activeDraftEditor.sessionId, id, name, config)
            }
          />
        ) : (
          <Notice tone="error">Run manager metadata is missing.</Notice>
        )
      ) : null}
    </div>
  );
}

function mergeRunDetail(summary: ManagedRun, detail: ManagedRunDetail): ManagedRunDetail {
  return {
    ...detail,
    ...summary,
    config: detail.config,
  };
}

function mergeWorkspaceRuns(visibleRuns: ManagedRun[], checkpointRuns: ManagedRun[]) {
  const byId = new Map<string, ManagedRun>();
  for (const run of visibleRuns) {
    byId.set(run.id, run);
  }
  for (const run of checkpointRuns) {
    if (!byId.has(run.id)) {
      byId.set(run.id, run);
    }
  }
  return [...byId.values()];
}

function forkDraftAltBaselineCount(
  forkSource: { copyAltBaselines: boolean; runId: string } | null,
  runs: readonly ManagedRun[],
) {
  if (forkSource === null) {
    return null;
  }
  if (!forkSource.copyAltBaselines) {
    return 0;
  }
  return runs.find((run) => run.id === forkSource.runId)?.active_alt_baseline_count ?? null;
}
