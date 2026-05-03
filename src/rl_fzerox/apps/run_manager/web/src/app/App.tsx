import { useCallback, useEffect, useState } from "react";

import { loadManagerData } from "@/app/managerData";
import { Configurator } from "@/features/configurator/Configurator";
import { DraftsPanel } from "@/features/drafts/DraftsPanel";
import { RunInspector } from "@/features/runs/RunInspector";
import { RunsPanel } from "@/features/runs/RunsPanel";
import { createDraft, deleteDraft, updateDraft } from "@/shared/api/client";
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

type WorkspaceTabId = "drafts" | "runs" | `editor:${string}`;

interface DraftEditorSession {
  draftId: string | null;
  initialDraftName: string;
  loadedDraft: ManagedDraft | null;
  sessionId: `editor:${string}`;
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
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
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

  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;
  const activeDraftEditor =
    activeTabId === "drafts" || activeTabId === "runs"
      ? null
      : (draftEditors.find((session) => session.sessionId === activeTabId) ?? null);
  const workspaceTabs: Array<{ closable?: boolean; id: WorkspaceTabId; label: string }> = [
    { id: "drafts", label: "Drafts" },
    { id: "runs", label: "Runs" },
    ...draftEditors.map((session) => ({
      id: session.sessionId,
      label: session.title,
      closable: true,
    })),
  ];

  async function saveDraft(
    sessionId: DraftEditorSession["sessionId"],
    name: string,
    config: ManagedRunConfig,
  ) {
    const draft = await createDraft(name, config);
    setDrafts((current) => upsertDraft(current, draft));
    patchDraftEditor(sessionId, {
      draftId: draft.id,
      initialDraftName: draft.name,
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
    const draft = await updateDraft(id, name, config);
    setDrafts((current) => upsertDraft(current, draft));
    patchDraftEditor(sessionId, {
      draftId: draft.id,
      initialDraftName: draft.name,
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
          onClose={(id) => closeDraftEditor(id as DraftEditorSession["sessionId"])}
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
        {!isLoading && activeTabId === "runs" && selectedRun === null ? (
          <RunsPanel runs={runs} onOpenRun={openRun} />
        ) : null}
        {!isLoading && activeTabId === "runs" && selectedRun !== null ? (
          <RunInspector run={selectedRun} />
        ) : null}
        {!isLoading && activeDraftEditor !== null ? (
          defaultConfig !== null && metadata !== null ? (
            draftEditors.map((session) => (
              <div hidden={activeTabId !== session.sessionId} key={session.sessionId}>
                <Configurator
                  baseConfig={defaultConfig}
                  existingNames={reservedNamesForSession(session.sessionId)}
                  initialDraftName={session.initialDraftName}
                  loadedDraft={session.loadedDraft}
                  metadata={metadata}
                  onDraftNameChange={(name) => setDraftEditorTitle(session.sessionId, name)}
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
              initialDraftName: draft.name,
              loadedDraft: draft,
              sessionId,
              title: draft.name,
            },
          ],
    );
    setActiveTabId(sessionId);
  }

  function openRun(run: ManagedRun) {
    setSelectedRunId(run.id);
    setActiveTabId("runs");
  }

  function createNewDraft() {
    const sessionId = editorSessionId(crypto.randomUUID());
    const initialDraftName = nextAvailableDraftName(defaultDraftName(), allKnownNames());
    setDraftEditors((current) => [
      ...current,
      { draftId: null, initialDraftName, loadedDraft: null, sessionId, title: initialDraftName },
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

  function reservedNamesForSession(sessionId: DraftEditorSession["sessionId"]) {
    const names = new Set<string>();
    for (const draft of drafts) {
      names.add(draft.name);
    }
    for (const run of runs) {
      names.add(run.name);
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
