// web/run-manager/src/app/App.tsx
import { useEffect, useState } from "react";
import { useWorkspaceActions } from "@/app/workspace/actions";
import { WorkspaceBody } from "@/app/workspace/Body";
import { useWorkspaceSessions } from "@/app/workspace/sessions";
import { useManagerData } from "@/app/workspace/useManagerData";
import { ScrollButtons } from "@/shared/ui/ScrollButtons";
import { Tabs } from "@/shared/ui/Tabs";
import { type Theme, ThemeToggle } from "@/shared/ui/ThemeToggle";
import { AppTooltipProvider } from "@/shared/ui/Tooltip";

export function App() {
  const [theme, setTheme] = useState<Theme>("dark");
  const managerData = useManagerData();
  const sessions = useWorkspaceSessions({
    drafts: managerData.drafts,
    runs: managerData.runs,
    saveGames: managerData.saveGames,
  });
  const actions = useWorkspaceActions({
    drafts: managerData.drafts,
    loadRunDetail: managerData.loadRunDetail,
    reloadManagerData: () => managerData.reloadManagerData(),
    runs: managerData.runs,
    sessions,
    setDrafts: managerData.setDrafts,
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
            activeId={sessions.activeTabId}
            items={sessions.workspaceTabs}
            variant="workspace"
            onClose={(id) => sessions.closeWorkspaceTab(id)}
            onSelect={(id) => sessions.setActiveTabId(id)}
          />
        </div>

        <WorkspaceBody
          actions={actions}
          defaultConfig={managerData.defaultConfig}
          drafts={managerData.drafts}
          error={managerData.error}
          isLoading={managerData.isLoading}
          loadRunDetail={managerData.loadRunDetail}
          metadata={managerData.metadata}
          runs={managerData.runs}
          runDetailsById={managerData.runDetailsById}
          saveGames={managerData.saveGames}
          sessions={sessions}
          onRefresh={() => managerData.reloadManagerData({ showLoading: false })}
        />
        <ScrollButtons />
      </main>
    </AppTooltipProvider>
  );
}
