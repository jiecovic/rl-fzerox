// src/rl_fzerox/apps/run_manager/web/src/app/workspace/Body.tsx
import { useEffect, useState } from "react";

import type { WorkspaceActions } from "@/app/workspace/actions";
import type { WorkspaceSessions } from "@/app/workspace/sessions";
import { Configurator } from "@/features/configurator/Configurator";
import { DraftsPanel } from "@/features/drafts/DraftsPanel";
import { ChartsPanel } from "@/features/runs/ChartsPanel";
import { RunsPanel } from "@/features/runs/RunsPanel";
import { RunWorkspace } from "@/features/runs/RunWorkspace";
import { SaveGamesPanel } from "@/features/save_games/SaveGamesPanel";
import { SaveGameWorkspace } from "@/features/save_games/SaveGameWorkspace";
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
  onRefresh: () => Promise<void>;
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
  onRefresh,
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
  const activeRun =
    activeRunSummary === null || activeRunDetail === null
      ? null
      : mergeRunDetail(activeRunSummary, activeRunDetail);

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
          onRenameSaveGame={actions.renameManagedSaveGame}
          onRefresh={onRefresh}
          onUpsertCourseSetup={actions.upsertManagedSaveCourseSetup}
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
            onCreateDraftFromRun={actions.createDraftFromManagedRun}
            onFork={actions.forkManagedRun}
            onOpenDirectory={actions.openManagedRunDirectory}
            onRename={actions.renameManagedRun}
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
            forkSourceArtifact={activeDraftEditor.forkSource?.artifact ?? null}
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
            onLaunchRun={(name, config, draftId) =>
              actions.launchTrainingRun(activeDraftEditor.sessionId, name, config, draftId)
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
