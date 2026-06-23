// web/run-manager/src/app/App.tsx
import { useEffect, useState } from "react";
import { useWorkspaceActions } from "@/app/workspace/actions";
import { WorkspaceBody } from "@/app/workspace/Body";
import { primaryWorkspaceTabs } from "@/app/workspace/model";
import { useWorkspaceSessions } from "@/app/workspace/sessions";
import { useManagerData } from "@/app/workspace/useManagerData";
import { ScrollButtons } from "@/shared/ui/ScrollButtons";
import { Tabs } from "@/shared/ui/Tabs";
import { type Theme, ThemeToggle } from "@/shared/ui/ThemeToggle";
import { AppTooltipProvider } from "@/shared/ui/Tooltip";

export function App() {
  const [theme, setTheme] = useState<Theme>("dark");
  const [globalError, setGlobalError] = useState<string | null>(null);
  const managerData = useManagerData();
  const sessions = useWorkspaceSessions({
    drafts: managerData.drafts,
    evaluations: managerData.evaluations,
    runs: managerData.runs,
    saveGames: managerData.saveGames,
  });
  const actions = useWorkspaceActions({
    drafts: managerData.drafts,
    loadRunDetail: managerData.loadRunDetail,
    reloadManagerData: () => managerData.reloadManagerData(),
    runs: managerData.runs,
    sessions,
    setGlobalError,
    setDrafts: managerData.setDrafts,
    setEvaluations: managerData.setEvaluations,
    setRuns: managerData.setRuns,
    setSaveGames: managerData.setSaveGames,
    upsertRunDetail: managerData.upsertRunDetail,
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  return (
    <AppTooltipProvider>
      <main className="app-shell">
        <header className="app-header ml-[126px]">
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

        <div className="mt-[18px] grid grid-cols-[118px_minmax(0,1fr)] items-start gap-2">
          <aside className="sticky top-4 border border-app-border bg-app-surface p-1">
            <Tabs
              label="Run manager sections"
              activeId={sessions.activePrimaryTabId}
              items={primaryWorkspaceTabs}
              variant="sidebar"
              onSelect={(id) => sessions.setActiveTabId(id)}
            />
          </aside>

          <div className="min-w-0">
            {sessions.sessionTabs.length > 0 ? (
              <div className="border border-b-0 border-app-border bg-app-surface px-4">
                <Tabs
                  label="Open workspace tabs"
                  activeId={sessions.activeTabId}
                  items={sessions.sessionTabs}
                  variant="workspace"
                  onClose={(id) => sessions.closeWorkspaceTab(id)}
                  onSelect={(id) => sessions.setActiveTabId(id)}
                />
              </div>
            ) : null}

            <WorkspaceBody
              actions={actions}
              defaultConfig={managerData.defaultConfig}
              drafts={managerData.drafts}
              error={globalError ?? managerData.error}
              evaluationBaselineSuites={managerData.evaluationBaselineSuites}
              evaluationError={managerData.evaluationError}
              evaluations={managerData.evaluations}
              evaluationPresets={managerData.evaluationPresets}
              isLoading={managerData.isLoading}
              loadRunDetail={managerData.loadRunDetail}
              metadata={managerData.metadata}
              runs={managerData.runs}
              runDetailsById={managerData.runDetailsById}
              saveGames={managerData.saveGames}
              sessions={sessions}
              onRefreshSaveGameStatus={managerData.refreshSaveGameStatus}
            />
          </div>
        </div>
        <ScrollButtons />
      </main>
    </AppTooltipProvider>
  );
}
