// src/rl_fzerox/apps/run_manager/web/src/app/workspace/Body.tsx
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
} from "@/shared/api/contract";
import { Notice } from "@/shared/ui/Panel";

interface WorkspaceBodyProps {
  actions: WorkspaceActions;
  defaultConfig: ManagedRunConfig | null;
  drafts: ManagedDraft[];
  error: string | null;
  isLoading: boolean;
  metadata: ConfigMetadata | null;
  runs: ManagedRun[];
  sessions: WorkspaceSessions;
}

export function WorkspaceBody({
  actions,
  defaultConfig,
  drafts,
  error,
  isLoading,
  metadata,
  runs,
  sessions,
}: WorkspaceBodyProps) {
  const activeRunTab = sessions.activeRunTab;
  const activeRun =
    activeRunTab === null
      ? null
      : (runs.find((candidate) => candidate.id === activeRunTab.runId) ?? null);

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
        activeRun === null ? (
          <Notice tone="error">This run is no longer available.</Notice>
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
