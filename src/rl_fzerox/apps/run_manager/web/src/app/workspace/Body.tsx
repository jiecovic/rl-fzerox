// src/rl_fzerox/apps/run_manager/web/src/app/workspace/Body.tsx
import { useEffect, useState } from "react";

import type { WorkspaceActions } from "@/app/workspace/actions";
import type { WorkspaceSessions } from "@/app/workspace/sessions";
import { Configurator } from "@/features/configurator/Configurator";
import { DraftsPanel } from "@/features/drafts/DraftsPanel";
import { ChartsPanel } from "@/features/runs/ChartsPanel";
import { RunsPanel } from "@/features/runs/RunsPanel";
import { RunWorkspace } from "@/features/runs/RunWorkspace";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRun,
  ManagedRunConfig,
  ManagedRunDetail,
} from "@/shared/api/contract";
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
  sessions: WorkspaceSessions;
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
  sessions,
}: WorkspaceBodyProps) {
  const [runDetailError, setRunDetailError] = useState<string | null>(null);
  const activeRunTab = sessions.activeRunTab;
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
      {error !== null ? <Notice tone="error">{error}</Notice> : null}
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
      {!isLoading && activeRunTab !== null && metadata !== null ? (
        activeRunSummary === null ? (
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
      {!isLoading && sessions.activeDraftEditor !== null ? (
        defaultConfig !== null && metadata !== null ? (
          sessions.draftEditors.map((session) => (
            <div hidden={sessions.activeTabId !== session.sessionId} key={session.sessionId}>
              <Configurator
                active={sessions.activeTabId === session.sessionId}
                baseConfig={defaultConfig}
                existingNames={sessions.reservedNamesForSession(session.sessionId)}
                forkSourceArtifact={session.forkSource?.artifact ?? null}
                forkSourceRunLabel={sessions.forkSourceRunLabel(session.forkSource)}
                initialConfig={session.initialConfig}
                initialDraftName={session.initialDraftName}
                loadedDraft={session.loadedDraft}
                metadata={metadata}
                onDraftNameChange={(name) => sessions.setDraftEditorTitle(session.sessionId, name)}
                onLaunchRun={(name, config, draftId) =>
                  actions.launchTrainingRun(session.sessionId, name, config, draftId)
                }
                onSaveDraft={(name, config) => actions.saveDraft(session.sessionId, name, config)}
                onUpdateDraft={(id, name, config) =>
                  actions.updateExistingDraft(session.sessionId, id, name, config)
                }
              />
            </div>
          ))
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
