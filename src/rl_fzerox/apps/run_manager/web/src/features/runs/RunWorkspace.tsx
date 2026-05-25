// src/rl_fzerox/apps/run_manager/web/src/features/runs/RunWorkspace.tsx
import { useEffect, useState } from "react";
import type { ConfigSection } from "@/features/configurator/configurator/sections";
import { useRunWorkspaceActions } from "@/features/runs/workspace/actions";
import { RunIdentityPanel } from "@/features/runs/workspace/IdentityPanel";
import { useRunPolicyPreview, useRunTrackSamplingState } from "@/features/runs/workspace/polling";
import { RunReadonlyConfig } from "@/features/runs/workspace/ReadonlyConfig";
import { RunRuntimeSummary, runWorkspaceSubtitle } from "@/features/runs/workspace/RuntimeSummary";
import type {
  ConfigMetadata,
  ManagedRun,
  ManagedRunDetail,
  WatchDevice,
  WatchRenderer,
} from "@/shared/api/contract";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

interface RunWorkspaceProps {
  allRuns: ManagedRun[];
  metadata: ConfigMetadata;
  onCreateDraftFromRun: (runId: string) => Promise<void>;
  onFork: (runId: string, artifact: "latest" | "best") => Promise<void>;
  onOpenDirectory: (runId: string) => Promise<void>;
  onRename: (runId: string, name: string) => Promise<void>;
  onResume: (runId: string) => Promise<void>;
  onResetTrackPool: (runId: string) => Promise<void>;
  onShowCharts: (runId: string) => void;
  onStop: (runId: string) => Promise<void>;
  onWatch: (
    runId: string,
    artifact: "latest" | "best",
    device: WatchDevice,
    renderer: WatchRenderer,
  ) => Promise<"started" | "already_running">;
  run: ManagedRunDetail;
}

export function RunWorkspace({
  allRuns,
  metadata,
  onCreateDraftFromRun,
  onFork,
  onOpenDirectory,
  onRename,
  onResume,
  onResetTrackPool,
  onShowCharts,
  onStop,
  onWatch,
  run,
}: RunWorkspaceProps) {
  const [runName, setRunName] = useState(run.name);
  const [configSection, setConfigSection] = useState<ConfigSection>("training");
  const previewEnabled = configSection === "observation" || configSection === "policy";
  const { policyPreview, previewError } = useRunPolicyPreview(run.config, previewEnabled);
  const { setTrackSamplingState, trackSamplingError, trackSamplingState } =
    useRunTrackSamplingState(run.id, run.status);
  const actions = useRunWorkspaceActions({
    clearTrackSamplingState: setTrackSamplingState,
    onCreateDraftFromRun,
    onFork,
    onOpenDirectory,
    onRename,
    onResume,
    onResetTrackPool,
    onStop,
    onWatch,
    run,
    runName,
  });
  const hasFeedback =
    actions.controlError !== null ||
    trackSamplingError !== null ||
    (previewEnabled && previewError !== null);

  useEffect(() => {
    setRunName(run.name);
  }, [run.name]);

  return (
    <Panel>
      <PanelHeader title={run.name} subtitle={runWorkspaceSubtitle(run)} />

      <RunRuntimeSummary
        actions={actions}
        allRuns={allRuns}
        metadata={metadata}
        onShowCharts={onShowCharts}
        run={run}
        trackSamplingState={trackSamplingState}
      />

      {hasFeedback ? (
        <div className="configurator-feedback-stack">
          {actions.controlError !== null ? (
            <Notice tone="error">{actions.controlError}</Notice>
          ) : null}
          {trackSamplingError !== null ? <Notice tone="error">{trackSamplingError}</Notice> : null}
          {previewEnabled && previewError !== null ? (
            <Notice tone="error">{previewError}</Notice>
          ) : null}
        </div>
      ) : null}

      <RunIdentityPanel
        canRename={actions.canRename}
        isRenaming={actions.isRenaming}
        onRename={actions.renameRunLabel}
        onRunNameChange={setRunName}
        run={run}
        runName={runName}
      />

      <RunReadonlyConfig
        metadata={metadata}
        onSectionChange={setConfigSection}
        policyPreview={policyPreview}
        run={run}
        section={configSection}
      />
    </Panel>
  );
}
