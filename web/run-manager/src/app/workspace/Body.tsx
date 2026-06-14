// web/run-manager/src/app/workspace/Body.tsx
import { useEffect, useMemo, useState } from "react";

import type { WorkspaceActions } from "@/app/workspace/actions";
import type { WorkspaceSessions } from "@/app/workspace/sessions";
import { DraftsPanel } from "@/pages/drafts/DraftsPanel";
import { RunsPanel } from "@/pages/runs/RunsPanel";
import { SaveGamesPanel } from "@/pages/saveGames/SaveGamesPanel";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRun,
  ManagedRunConfig,
  ManagedRunDetail,
  ManagedSaveGame,
} from "@/shared/api/contract";
import { FloatingNotice } from "@/shared/ui/FloatingNotice";
import { Notice } from "@/shared/ui/Panel";
import { Configurator } from "@/widgets/configurator/Configurator";
import { ChartsPanel } from "@/widgets/runCharts/ChartsPanel";
import { RunWorkspace } from "@/widgets/runWorkspace/RunWorkspace";
import { SaveGameWorkspace } from "@/widgets/saveGameWorkspace/SaveGameWorkspace";

interface WorkspaceBodyProps {
  actions: WorkspaceActions;
  defaultConfig: ManagedRunConfig | null;
  drafts: ManagedDraft[];
  error: string | null;
  isLoading: boolean;
  loadRunDetail: (runId: string) => Promise<ManagedRunDetail>;
  metadata: ConfigMetadata | null;
  runs: ManagedRun[];
  runDetailsById: Record<string, ManagedRunDetail>;
  saveGames: ManagedSaveGame[];
  sessions: WorkspaceSessions;
  onRefreshSaveGameStatus: (saveGameId: string) => Promise<void>;
}

export function WorkspaceBody({
  actions,
  defaultConfig,
  drafts,
  error,
  isLoading,
  loadRunDetail,
  metadata,
  runs,
  runDetailsById,
  saveGames,
  sessions,
  onRefreshSaveGameStatus,
}: WorkspaceBodyProps) {
  const [runDetailError, setRunDetailError] = useState<string | null>(null);
  const activeRunTab = sessions.activeRunTab;
  const activeDraftEditor = sessions.activeDraftEditor;
  const activeSaveGameSession = sessions.activeSaveGameSession;
  const activeRunSummary =
    activeRunTab === null
      ? null
      : (runs.find((candidate) => candidate.id === activeRunTab.runId) ?? null);
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
        setRunDetailError(caught instanceof Error ? caught.message : "failed to load run details");
      }
    });
    return () => {
      ignore = true;
    };
  }, [activeRunDetail, activeRunTab, loadRunDetail]);

  return (
    <div className="workspace">
      {error !== null ? <FloatingNotice tone="error">{error}</FloatingNotice> : null}
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
          onImportRunBundle={actions.importManagedRunBundle}
          onOpenRun={sessions.openRun}
          onResumeRun={(run) => actions.resumeManagedRun(run.id)}
          onStopRun={(run) => actions.stopManagedRun(run.id)}
          onUpdateLineageGroups={actions.updateManagedLineageGroups}
        />
      ) : null}
      {!isLoading && sessions.activeTabId === "charts" ? (
        <ChartsPanel
          focusedRunId={sessions.chartsFocusRunId}
          onOpenRun={sessions.openRun}
          runs={runs}
        />
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
          metadata={metadata}
          runs={runs}
          saveGame={
            activeSaveGameSession.saveGameId === null
              ? null
              : (saveGames.find((saveGame) => saveGame.id === activeSaveGameSession.saveGameId) ??
                null)
          }
          session={activeSaveGameSession}
          onCreateSaveGame={actions.createManagedSaveGame}
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
        metadata === null ? (
          <Notice tone="error">Run manager metadata is missing.</Notice>
        ) : activeRunSummary === null ? (
          <Notice tone="error">This run is no longer available.</Notice>
        ) : activeRun === null ? (
          <Notice tone={runDetailError === null ? undefined : "error"}>
            {runDetailError ?? "Loading run details..."}
          </Notice>
        ) : (
          <RunWorkspace
            allRuns={runs}
            metadata={metadata}
            onClearAltBaselines={actions.clearManagedRunAltBaselines}
            onClearCourseAltBaselines={actions.clearManagedRunCourseAltBaselines}
            onCreateDraftFromRun={actions.createDraftFromManagedRun}
            onFork={actions.forkManagedRun}
            onOpenDirectory={actions.openManagedRunDirectory}
            onRename={actions.renameManagedRun}
            onResetEngineTuning={actions.resetManagedRunEngineTuning}
            onResetTrackPool={actions.resetManagedRunTrackPool}
            onResume={actions.resumeManagedRun}
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
            baseConfig={defaultConfig}
            existingNames={sessions.reservedNamesForSession(activeDraftEditor.sessionId)}
            forkAltBaselineCount={forkDraftAltBaselineCount(activeDraftEditor.forkSource, runs)}
            forkCopyAltBaselines={activeDraftEditor.forkSource?.copyAltBaselines ?? null}
            forkSourceArtifact={activeDraftEditor.forkSource?.artifact ?? null}
            forkSourceEngineTunerBackend={
              activeDraftEditor.forkSource?.sourceEngineTunerBackend ?? null
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
