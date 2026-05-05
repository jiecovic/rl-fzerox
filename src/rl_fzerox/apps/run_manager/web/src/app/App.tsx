import { useCallback, useEffect, useState } from "react";

import { loadManagerData } from "@/app/managerData";
import { Configurator } from "@/features/configurator/Configurator";
import { DraftsPanel } from "@/features/drafts/DraftsPanel";
import { ChartsPanel } from "@/features/runs/ChartsPanel";
import { RunsPanel } from "@/features/runs/RunsPanel";
import { RunWorkspace } from "@/features/runs/RunWorkspace";
import {
  createDraftWithSource,
  deleteDraft,
  deleteLineage,
  deleteRun,
  fetchRuns,
  launchRun,
  openRunDirectory,
  renameRun,
  resetRunTrackSamplingState,
  resumeRun,
  stopRun,
  updateDraftWithSource,
  watchRun,
} from "@/shared/api/client";
import type {
  ConfigMetadata,
  ManagedDraft,
  ManagedRun,
  ManagedRunConfig,
} from "@/shared/api/contract";
import { Notice } from "@/shared/ui/Panel";
import { ScrollButtons } from "@/shared/ui/ScrollButtons";
import { Tabs } from "@/shared/ui/Tabs";
import { type Theme, ThemeToggle } from "@/shared/ui/ThemeToggle";

type WorkspaceTabId = "drafts" | "runs" | "charts" | `editor:${string}` | `run:${string}`;

interface ForkSource {
  artifact: "latest" | "best";
  runId: string;
}

interface DraftEditorSession {
  draftId: string | null;
  forkSource: ForkSource | null;
  initialDraftName: string;
  initialConfig: ManagedRunConfig | null;
  loadedDraft: ManagedDraft | null;
  sessionId: `editor:${string}`;
  title: string;
}

interface RunSession {
  runId: string;
  sessionId: `run:${string}`;
  title: string;
}

export function App() {
  const [theme, setTheme] = useState<Theme>("dark");
  const [activeTabId, setActiveTabId] = useState<WorkspaceTabId>("drafts");
  const [drafts, setDrafts] = useState<ManagedDraft[]>([]);
  const [runs, setRuns] = useState<ManagedRun[]>([]);
  const [metadata, setMetadata] = useState<ConfigMetadata | null>(null);
  const [defaultConfig, setDefaultConfig] = useState<ManagedRunConfig | null>(null);
  const [draftEditors, setDraftEditors] = useState<DraftEditorSession[]>([]);
  const [runTabs, setRunTabs] = useState<RunSession[]>([]);
  const [chartsFocusRunId, setChartsFocusRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const reloadManagerData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const managerData = await loadManagerData();
      setDrafts(managerData.drafts);
      setRuns(managerData.runs);
      setMetadata(managerData.metadata);
      setDefaultConfig((current) => current ?? managerData.templates[0]?.config ?? null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "failed to load run manager data");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    void reloadManagerData();
  }, [reloadManagerData]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void fetchRuns()
        .then((nextRuns) => setRuns([...nextRuns].sort(compareRuns)))
        .catch(() => undefined);
    }, 2_000);
    return () => window.clearInterval(intervalId);
  }, []);

  const activeDraftEditor =
    activeTabId === "drafts" || activeTabId === "runs" || activeTabId === "charts"
      ? null
      : (draftEditors.find((session) => session.sessionId === activeTabId) ?? null);
  const activeRunTab =
    activeTabId === "drafts" || activeTabId === "runs" || activeTabId === "charts"
      ? null
      : (runTabs.find((session) => session.sessionId === activeTabId) ?? null);
  const workspaceTabs: Array<{
    closable?: boolean;
    id: WorkspaceTabId;
    label: string;
    tone?: "draft" | "run";
  }> = [
    { id: "drafts", label: "Drafts" },
    { id: "runs", label: "Runs" },
    { id: "charts", label: "Charts" },
    ...runTabs.map((session) => ({
      id: session.sessionId,
      label: `Run · ${runs.find((run) => run.id === session.runId)?.name ?? session.title}`,
      closable: true,
      tone: "run" as const,
    })),
    ...draftEditors.map((session) => ({
      id: session.sessionId,
      label: `${session.forkSource === null ? "Draft" : "Fork draft"} · ${session.title}`,
      closable: true,
      tone: "draft" as const,
    })),
  ];

  async function saveDraft(
    sessionId: DraftEditorSession["sessionId"],
    name: string,
    config: ManagedRunConfig,
  ) {
    const session = draftEditors.find((current) => current.sessionId === sessionId) ?? null;
    const draft = await createDraftWithSource(
      name,
      config,
      session?.forkSource?.runId ?? null,
      session?.forkSource?.artifact ?? null,
    );
    setDrafts((current) => upsertDraft(current, draft));
    patchDraftEditor(sessionId, {
      draftId: draft.id,
      forkSource: draftForkSource(draft),
      initialDraftName: draft.name,
      initialConfig: null,
      loadedDraft: draft,
      title: draft.name,
    });
    return draft;
  }

  async function updateExistingDraft(
    sessionId: DraftEditorSession["sessionId"],
    id: string,
    name: string,
    config: ManagedRunConfig,
  ) {
    const session = draftEditors.find((current) => current.sessionId === sessionId) ?? null;
    const draft = await updateDraftWithSource(
      id,
      name,
      config,
      session?.forkSource?.runId ?? null,
      session?.forkSource?.artifact ?? null,
    );
    setDrafts((current) => upsertDraft(current, draft));
    patchDraftEditor(sessionId, {
      draftId: draft.id,
      forkSource: draftForkSource(draft),
      initialDraftName: draft.name,
      initialConfig: null,
      loadedDraft: draft,
      title: draft.name,
    });
    return draft;
  }

  async function removeDraft(id: string) {
    await deleteDraft(id);
    setDrafts((current) => current.filter((draft) => draft.id !== id));
    closeEditorsForDraft(id);
  }

  async function removeRun(run: ManagedRun) {
    await deleteRun(run.id);
    setRuns((current) => current.filter((candidate) => candidate.id !== run.id));
    closeRunTabsForRun(run.id);
    setChartsFocusRunId((current) => (current === run.id ? null : current));
  }

  async function removeLineage(lineageId: string) {
    const lineageRunIds = runs
      .filter((candidate) => candidate.lineage_id === lineageId)
      .map((candidate) => candidate.id);
    await deleteLineage(lineageId);
    closeRunTabsForRuns(lineageRunIds);
    closeEditorsForSourceRuns(lineageRunIds);
    setChartsFocusRunId((current) =>
      current !== null && lineageRunIds.includes(current) ? null : current,
    );
    await reloadManagerData();
  }

  async function launchTrainingRun(
    sessionId: DraftEditorSession["sessionId"],
    name: string,
    config: ManagedRunConfig,
    draftId: string | null,
  ) {
    const session = draftEditors.find((current) => current.sessionId === sessionId) ?? null;
    const run = await launchRun(
      name,
      config,
      draftId,
      session?.forkSource?.runId ?? null,
      session?.forkSource?.artifact ?? null,
    );
    setRuns((current) => upsertRun(current, run));
    openRun(run);
    return run;
  }

  async function forkManagedRun(runId: string, artifact: "latest" | "best") {
    const sourceRun = runs.find((candidate) => candidate.id === runId) ?? null;
    const sourceConfig = sourceRun?.config ?? defaultConfig;
    if (sourceConfig === null) {
      throw new Error("fork source config is unavailable");
    }
    const baseName =
      sourceRun === null
        ? artifact === "best"
          ? "fork best"
          : "fork"
        : artifact === "best"
          ? `${sourceRun.name} best fork`
          : `${sourceRun.name} fork`;
    const initialDraftName = nextAvailableDraftName(baseName, allKnownNames());
    const draft = await createDraftWithSource(initialDraftName, sourceConfig, runId, artifact);
    setDrafts((current) => upsertDraft(current, draft));
    openDraft(draft);
  }

  async function createDraftFromManagedRun(runId: string) {
    const sourceRun = runs.find((candidate) => candidate.id === runId) ?? null;
    if (sourceRun === null) {
      throw new Error("run config is unavailable");
    }
    const initialDraftName = nextAvailableDraftName(`${sourceRun.name} draft`, allKnownNames());
    const draft = await createDraftWithSource(initialDraftName, sourceRun.config, null, null);
    setDrafts((current) => upsertDraft(current, draft));
    openDraft(draft);
  }

  async function stopManagedRun(runId: string) {
    const run = await stopRun(runId);
    setRuns((current) => upsertRun(current, run));
  }

  async function resumeManagedRun(runId: string) {
    const run = await resumeRun(runId);
    setRuns((current) => upsertRun(current, run));
  }

  async function renameManagedRun(runId: string, name: string) {
    const run = await renameRun(runId, name);
    setRuns((current) => upsertRun(current, run));
  }

  async function openManagedRunDirectory(runId: string) {
    await openRunDirectory(runId);
  }

  async function watchManagedRun(runId: string, artifact: "latest" | "best") {
    await watchRun(runId, artifact);
  }

  async function resetManagedRunTrackPool(runId: string) {
    await resetRunTrackSamplingState(runId);
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-mark">FX</div>
          <div>
            <h1>F-Zero X runs</h1>
            <p>Local training configuration and run control.</p>
          </div>
        </div>
        <ThemeToggle
          theme={theme}
          onToggle={() => setTheme((current) => (current === "light" ? "dark" : "light"))}
        />
      </header>

      <div className="navigation-strip">
        <Tabs
          label="Run manager sections"
          activeId={activeTabId}
          items={workspaceTabs}
          variant="workspace"
          onClose={(id) => closeWorkspaceTab(id)}
          onSelect={(id) => setActiveTabId(id)}
        />
      </div>

      <div className="workspace">
        {error !== null ? <Notice tone="error">{error}</Notice> : null}
        {isLoading ? <Notice>Loading manager data...</Notice> : null}

        {!isLoading && activeTabId === "drafts" ? (
          <DraftsPanel
            drafts={drafts}
            onCreateDraft={createNewDraft}
            onDeleteDraft={(draft) => removeDraft(draft.id)}
            onOpenDraft={openDraft}
          />
        ) : null}
        {!isLoading && activeTabId === "runs" ? (
          <RunsPanel
            drafts={drafts}
            runs={runs}
            onDeleteLineage={removeLineage}
            onDeleteRun={removeRun}
            onOpenRun={openRun}
            onResumeRun={(run) => resumeManagedRun(run.id)}
            onStopRun={(run) => stopManagedRun(run.id)}
          />
        ) : null}
        {!isLoading && activeTabId === "charts" ? (
          <ChartsPanel focusedRunId={chartsFocusRunId} onOpenRun={openRun} runs={runs} />
        ) : null}
        {!isLoading && activeRunTab !== null && metadata !== null
          ? runTabs.map((session) => {
              const run = runs.find((candidate) => candidate.id === session.runId) ?? null;
              return (
                <div hidden={activeTabId !== session.sessionId} key={session.sessionId}>
                  {run === null ? (
                    <Notice tone="error">This run is no longer available.</Notice>
                  ) : (
                    <RunWorkspace
                      allRuns={runs}
                      onCreateDraftFromRun={createDraftFromManagedRun}
                      metadata={metadata}
                      run={run}
                      onOpenDirectory={openManagedRunDirectory}
                      onFork={forkManagedRun}
                      onRename={renameManagedRun}
                      onResume={resumeManagedRun}
                      onResetTrackPool={resetManagedRunTrackPool}
                      onShowCharts={showRunCharts}
                      onStop={stopManagedRun}
                      onWatch={watchManagedRun}
                    />
                  )}
                </div>
              );
            })
          : null}
        {!isLoading && activeDraftEditor !== null ? (
          defaultConfig !== null && metadata !== null ? (
            draftEditors.map((session) => (
              <div hidden={activeTabId !== session.sessionId} key={session.sessionId}>
                <Configurator
                  baseConfig={defaultConfig}
                  existingNames={reservedNamesForSession(session.sessionId)}
                  forkSourceArtifact={session.forkSource?.artifact ?? null}
                  forkSourceRunLabel={forkSourceRunLabel(session.forkSource)}
                  initialDraftName={session.initialDraftName}
                  initialConfig={session.initialConfig}
                  loadedDraft={session.loadedDraft}
                  metadata={metadata}
                  onDraftNameChange={(name) => setDraftEditorTitle(session.sessionId, name)}
                  onLaunchRun={(name, config, draftId) =>
                    launchTrainingRun(session.sessionId, name, config, draftId)
                  }
                  onSaveDraft={(name, config) => saveDraft(session.sessionId, name, config)}
                  onUpdateDraft={(id, name, config) =>
                    updateExistingDraft(session.sessionId, id, name, config)
                  }
                />
              </div>
            ))
          ) : (
            <Notice tone="error">Run manager metadata is missing.</Notice>
          )
        ) : null}
      </div>
      <ScrollButtons />
    </main>
  );

  function openDraft(draft: ManagedDraft) {
    const sessionId = editorSessionId(draft.id);
    setDraftEditors((current) =>
      current.some((session) => session.sessionId === sessionId)
        ? current
        : [
            ...current,
            {
              draftId: draft.id,
              forkSource: draftForkSource(draft),
              initialDraftName: draft.name,
              initialConfig: null,
              loadedDraft: draft,
              sessionId,
              title: draft.name,
            },
          ],
    );
    setActiveTabId(sessionId);
  }

  function openRun(run: ManagedRun) {
    const sessionId = runSessionId(run.id);
    setRunTabs((current) =>
      current.some((session) => session.sessionId === sessionId)
        ? current
        : [...current, { runId: run.id, sessionId, title: run.name }],
    );
    setActiveTabId(sessionId);
  }

  function showRunCharts(runId: string) {
    setChartsFocusRunId(runId);
    setActiveTabId("charts");
  }

  function createNewDraft() {
    const sessionId = editorSessionId(crypto.randomUUID());
    const initialDraftName = nextAvailableDraftName(defaultDraftName(), allKnownNames());
    setDraftEditors((current) => [
      ...current,
      {
        draftId: null,
        forkSource: null,
        initialDraftName,
        initialConfig: null,
        loadedDraft: null,
        sessionId,
        title: initialDraftName,
      },
    ]);
    setActiveTabId(sessionId);
  }

  function closeDraftEditor(sessionId: DraftEditorSession["sessionId"]) {
    const closingIndex = draftEditors.findIndex((session) => session.sessionId === sessionId);
    if (closingIndex === -1) {
      return;
    }
    const remaining = draftEditors.filter((session) => session.sessionId !== sessionId);
    setDraftEditors(remaining);
    if (activeTabId === sessionId) {
      const fallbackSession =
        remaining[closingIndex - 1] ?? remaining[closingIndex] ?? remaining.at(-1) ?? null;
      setActiveTabId(fallbackSession?.sessionId ?? "drafts");
    }
  }

  function closeRunTab(sessionId: RunSession["sessionId"]) {
    const closingIndex = runTabs.findIndex((session) => session.sessionId === sessionId);
    if (closingIndex === -1) {
      return;
    }
    const remaining = runTabs.filter((session) => session.sessionId !== sessionId);
    setRunTabs(remaining);
    if (activeTabId === sessionId) {
      const fallbackSession =
        remaining[closingIndex - 1] ?? remaining[closingIndex] ?? remaining.at(-1) ?? null;
      setActiveTabId(fallbackSession?.sessionId ?? "runs");
    }
  }

  function closeRunTabsForRun(runId: string) {
    closeRunTabsForRuns([runId]);
  }

  function closeRunTabsForRuns(runIds: readonly string[]) {
    if (runIds.length === 0) {
      return;
    }
    const runIdSet = new Set(runIds);
    const removedSessions = runTabs.filter((session) => runIdSet.has(session.runId));
    if (removedSessions.length === 0) {
      return;
    }
    const removedIds = new Set(removedSessions.map((session) => session.sessionId));
    const remaining = runTabs.filter((session) => !removedIds.has(session.sessionId));
    setRunTabs(remaining);
    if (removedIds.has(activeTabId as RunSession["sessionId"])) {
      setActiveTabId("runs");
    }
  }

  function closeWorkspaceTab(id: WorkspaceTabId) {
    if (id === "charts") {
      setActiveTabId("runs");
      return;
    }
    if (id.startsWith("editor:")) {
      closeDraftEditor(id as DraftEditorSession["sessionId"]);
      return;
    }
    if (id.startsWith("run:")) {
      closeRunTab(id as RunSession["sessionId"]);
    }
  }

  function setDraftEditorTitle(sessionId: DraftEditorSession["sessionId"], title: string) {
    patchDraftEditor(sessionId, { title: normalizeDraftTabTitle(title) });
  }

  function patchDraftEditor(
    sessionId: DraftEditorSession["sessionId"],
    patch: Partial<Omit<DraftEditorSession, "sessionId">>,
  ) {
    setDraftEditors((current) => {
      let changed = false;
      const next = current.map((session) => {
        if (session.sessionId !== sessionId) {
          return session;
        }
        const updated = { ...session, ...patch };
        changed =
          updated.title !== session.title ||
          updated.initialDraftName !== session.initialDraftName ||
          updated.initialConfig !== session.initialConfig ||
          updated.forkSource !== session.forkSource ||
          updated.draftId !== session.draftId ||
          updated.loadedDraft !== session.loadedDraft;
        return changed ? updated : session;
      });
      return changed ? next : current;
    });
  }

  function closeEditorsForDraft(id: string) {
    const removedSessions = draftEditors.filter((session) => session.draftId === id);
    if (removedSessions.length === 0) {
      return;
    }
    const removedSessionIds = new Set(removedSessions.map((session) => session.sessionId));
    const remaining = draftEditors.filter((session) => !removedSessionIds.has(session.sessionId));
    setDraftEditors(remaining);
    if (removedSessionIds.has(activeTabId as DraftEditorSession["sessionId"])) {
      setActiveTabId("drafts");
    }
  }

  function closeEditorsForSourceRuns(runIds: readonly string[]) {
    if (runIds.length === 0) {
      return;
    }
    const sourceRunIds = new Set(runIds);
    const removedSessionIds = new Set(
      draftEditors
        .filter((session) => {
          const sourceRunId =
            session.loadedDraft?.source_run_id ?? session.forkSource?.runId ?? null;
          return sourceRunId !== null && sourceRunIds.has(sourceRunId);
        })
        .map((session) => session.sessionId),
    );
    if (removedSessionIds.size === 0) {
      return;
    }
    setDraftEditors((current) =>
      current.filter((session) => !removedSessionIds.has(session.sessionId)),
    );
    if (removedSessionIds.has(activeTabId as DraftEditorSession["sessionId"])) {
      setActiveTabId("drafts");
    }
  }

  function forkSourceRunLabel(source: ForkSource | null) {
    if (source === null) {
      return null;
    }
    const run = runs.find((candidate) => candidate.id === source.runId) ?? null;
    return run?.name ?? source.runId;
  }

  function reservedNamesForSession(sessionId: DraftEditorSession["sessionId"]) {
    const names = new Set<string>();
    for (const draft of drafts) {
      names.add(draft.name);
    }
    for (const session of draftEditors) {
      if (session.sessionId !== sessionId) {
        names.add(session.title);
      }
    }
    return [...names];
  }

  function allKnownNames() {
    const names = new Set<string>();
    for (const draft of drafts) {
      names.add(draft.name);
    }
    for (const run of runs) {
      names.add(run.name);
    }
    for (const session of draftEditors) {
      names.add(session.title);
    }
    return names;
  }
}

function editorSessionId(seed: string): `editor:${string}` {
  return `editor:${seed}`;
}

function runSessionId(seed: string): `run:${string}` {
  return `run:${seed}`;
}

function normalizeDraftTabTitle(title: string) {
  const trimmed = title.trim();
  return trimmed.length > 0 ? trimmed : "New draft";
}

function defaultDraftName() {
  return "ppo_allcups_recurrent";
}

function nextAvailableDraftName(baseName: string, takenNames: Iterable<string>) {
  const normalizedTaken = new Set(
    [...takenNames].map((name) => name.trim().toLowerCase()).filter((name) => name.length > 0),
  );
  if (!normalizedTaken.has(baseName.toLowerCase())) {
    return baseName;
  }
  let suffix = 2;
  while (normalizedTaken.has(`${baseName} ${suffix}`.toLowerCase())) {
    suffix += 1;
  }
  return `${baseName} ${suffix}`;
}

function upsertDraft(current: ManagedDraft[], nextDraft: ManagedDraft) {
  const withoutPrevious = current.filter((draft) => draft.id !== nextDraft.id);
  return [nextDraft, ...withoutPrevious].sort(compareDrafts);
}

function compareDrafts(left: ManagedDraft, right: ManagedDraft) {
  if (left.updated_at !== right.updated_at) {
    return right.updated_at.localeCompare(left.updated_at);
  }
  return right.id.localeCompare(left.id);
}

function upsertRun(current: ManagedRun[], nextRun: ManagedRun) {
  const withoutPrevious = current.filter((run) => run.id !== nextRun.id);
  return [nextRun, ...withoutPrevious].sort(compareRuns);
}

function compareRuns(left: ManagedRun, right: ManagedRun) {
  if (left.created_at !== right.created_at) {
    return right.created_at.localeCompare(left.created_at);
  }
  return right.id.localeCompare(left.id);
}

function draftForkSource(draft: ManagedDraft): ForkSource | null {
  if (draft.source_run_id === null || draft.source_artifact === null) {
    return null;
  }
  return {
    runId: draft.source_run_id,
    artifact: draft.source_artifact,
  };
}
