import { useCallback, useEffect, useState } from "react";

import { loadManagerData } from "@/app/managerData";
import { Configurator } from "@/features/configurator/Configurator";
import { DraftInspector } from "@/features/drafts/DraftInspector";
import { DraftsPanel } from "@/features/drafts/DraftsPanel";
import { InspectBanner } from "@/features/inspect/InspectBanner";
import { RunInspector } from "@/features/runs/RunInspector";
import { RunsPanel } from "@/features/runs/RunsPanel";
import { createDraft, deleteDraft } from "@/shared/api/client";
import type {
  ManagedDraft,
  ManagedRun,
  ManagedRunConfig,
  ManagedTemplate,
} from "@/shared/api/contract";
import { Notice } from "@/shared/ui/Panel";
import { ScrollButtons } from "@/shared/ui/ScrollButtons";
import { Tabs } from "@/shared/ui/Tabs";
import { type Theme, ThemeToggle } from "@/shared/ui/ThemeToggle";

type Page = "configure" | "drafts" | "runs";

export function App() {
  const [theme, setTheme] = useState<Theme>("dark");
  const [page, setPage] = useState<Page>("configure");
  const [templates, setTemplates] = useState<ManagedTemplate[]>([]);
  const [drafts, setDrafts] = useState<ManagedDraft[]>([]);
  const [runs, setRuns] = useState<ManagedRun[]>([]);
  const [selectedDraftId, setSelectedDraftId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const reloadManagerData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const managerData = await loadManagerData();
      setTemplates(managerData.templates);
      setDrafts(managerData.drafts);
      setRuns(managerData.runs);
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

  const selectedDraft = drafts.find((draft) => draft.id === selectedDraftId) ?? null;
  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;
  const defaultConfig = templates[0]?.config ?? null;

  async function saveDraft(name: string, config: ManagedRunConfig) {
    const draft = await createDraft(name, config);
    setDrafts((current) => [draft, ...current]);
    setSelectedDraftId(draft.id);
  }

  async function removeDraft(id: string) {
    await deleteDraft(id);
    setDrafts((current) => current.filter((draft) => draft.id !== id));
    setSelectedDraftId(null);
    setPage("drafts");
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
          activeId={page}
          items={[
            { id: "configure", label: "Configure" },
            { id: "drafts", label: "Drafts" },
            { id: "runs", label: "Runs" },
          ]}
          onSelect={setPage}
        />
        {selectedDraft !== null ? (
          <InspectBanner
            title={selectedDraft.name}
            subtitle="draft"
            onOpen={() => {
              setPage("drafts");
            }}
            onClose={() => {
              setSelectedDraftId(null);
            }}
          />
        ) : null}
        {selectedRun !== null ? (
          <InspectBanner
            title={selectedRun.name}
            subtitle={selectedRun.status}
            onOpen={() => {
              setPage("runs");
            }}
            onClose={() => {
              setSelectedRunId(null);
            }}
          />
        ) : null}
      </div>

      <div className="workspace">
        {error !== null ? <Notice tone="error">{error}</Notice> : null}
        {isLoading ? <Notice>Loading manager data...</Notice> : null}

        {!isLoading && page === "configure" && defaultConfig !== null ? (
          <Configurator baseConfig={defaultConfig} onSaveDraft={saveDraft} />
        ) : null}
        {!isLoading && page === "drafts" && selectedDraft === null ? (
          <DraftsPanel drafts={drafts} onOpenDraft={openDraft} />
        ) : null}
        {!isLoading && page === "drafts" && selectedDraft !== null ? (
          <DraftInspector
            draft={selectedDraft}
            onDelete={() => void removeDraft(selectedDraft.id)}
          />
        ) : null}
        {!isLoading && page === "runs" && selectedRun === null ? (
          <RunsPanel runs={runs} onOpenRun={openRun} />
        ) : null}
        {!isLoading && page === "runs" && selectedRun !== null ? (
          <RunInspector run={selectedRun} />
        ) : null}
      </div>
      <ScrollButtons />
    </main>
  );

  function openDraft(draft: ManagedDraft) {
    setSelectedDraftId(draft.id);
    setPage("drafts");
  }

  function openRun(run: ManagedRun) {
    setSelectedRunId(run.id);
    setPage("runs");
  }
}
