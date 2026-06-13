// web/run-manager/src/widgets/runWorkspace/RunWorkspace.tsx
import { useEffect, useState } from "react";
import type { ConfigSection } from "@/entities/runConfig/model/sections";
import {
  useRunEngineTuningState,
  useRunPolicyPreview,
  useRunTrackSamplingState,
} from "@/features/runLiveData/hooks";
import { ForkAltBaselinesDialog } from "@/features/runWorkspaceActions/ForkAltBaselinesDialog";
import { useRunWorkspaceActions } from "@/features/runWorkspaceActions/useRunWorkspaceActions";
import type {
  ConfigMetadata,
  ManagedRun,
  ManagedRunDetail,
  PolicyPlaybackMode,
  WatchDevice,
  WatchRenderer,
} from "@/shared/api/contract";
import { RenameIcon } from "@/shared/ui/icons";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";
import { RenameDialog } from "@/shared/ui/RenameDialog";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";
import { RunReadonlyConfig } from "@/widgets/runWorkspace/workspace/ReadonlyConfig";
import {
  RunRuntimeSummary,
  runWorkspaceSubtitle,
} from "@/widgets/runWorkspace/workspace/RuntimeSummary";

interface RunWorkspaceProps {
  allRuns: ManagedRun[];
  metadata: ConfigMetadata;
  onClearAltBaselines: (runId: string) => Promise<void>;
  onCreateDraftFromRun: (runId: string) => Promise<void>;
  onFork: (runId: string, artifact: "latest" | "best", copyAltBaselines: boolean) => Promise<void>;
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
    policyMode: PolicyPlaybackMode,
  ) => Promise<"started" | "already_running">;
  run: ManagedRunDetail;
}

export function RunWorkspace({
  allRuns,
  metadata,
  onClearAltBaselines,
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
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [configSection, setConfigSection] = useState<ConfigSection>("training");
  const [engineTuningExpansion, setEngineTuningExpansion] = useState({
    expanded: false,
    runId: run.id,
  });
  const engineTuningExpanded =
    engineTuningExpansion.runId === run.id ? engineTuningExpansion.expanded : false;
  const setEngineTuningExpanded = (expanded: boolean) => {
    setEngineTuningExpansion({ expanded, runId: run.id });
  };
  const previewEnabled = configSection === "observation" || configSection === "policy";
  const { policyPreview, previewError } = useRunPolicyPreview(run.config, previewEnabled);
  const { setTrackSamplingState, trackSamplingError, trackSamplingState } =
    useRunTrackSamplingState(run.id, run.status);
  const actions = useRunWorkspaceActions({
    clearTrackSamplingState: setTrackSamplingState,
    onClearAltBaselines,
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
  const engineTuningEnabled = run.config.vehicle.engine_mode === "adaptive_tuner";
  const { engineTuningError, engineTuningState } = useRunEngineTuningState(
    run.id,
    run.status,
    engineTuningEnabled && engineTuningExpanded,
    actions.selectedWatchArtifact,
  );
  const hasFeedback =
    actions.controlError !== null ||
    engineTuningError !== null ||
    trackSamplingError !== null ||
    (previewEnabled && previewError !== null);

  useEffect(() => {
    setRunName(run.name);
  }, [run.name]);

  async function submitRunRename(name: string) {
    setRunName(name);
    const renamed = await actions.renameRunLabel(name);
    if (renamed) {
      setRenameDialogOpen(false);
    }
  }

  return (
    <Panel>
      <PanelHeader
        title={
          <span className="inline-flex min-w-0 items-center gap-2">
            <span className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">
              {run.name}
            </span>
            <TooltipIconButton
              aria-label="Rename run"
              disabled={actions.isRenaming}
              size="small"
              tooltip="Rename"
              onClick={() => setRenameDialogOpen(true)}
            >
              <RenameIcon />
            </TooltipIconButton>
          </span>
        }
        subtitle={runWorkspaceSubtitle(run)}
      />
      <RenameDialog
        busy={actions.isRenaming}
        initialName={run.name}
        label="Run name"
        open={renameDialogOpen}
        title="Rename run"
        onClose={() => setRenameDialogOpen(false)}
        onSubmit={(name) => void submitRunRename(name)}
      />
      <ForkAltBaselinesDialog
        altBaselineCount={actions.pendingForkAltBaselineChoice?.count ?? 0}
        open={actions.pendingForkAltBaselineChoice !== null}
        onClose={actions.cancelForkAltBaselineChoice}
        onSelect={(copyAltBaselines) => void actions.confirmForkAltBaselineChoice(copyAltBaselines)}
      />

      <RunRuntimeSummary
        actions={actions}
        allRuns={allRuns}
        metadata={metadata}
        onShowCharts={onShowCharts}
        run={run}
        engineTuningExpanded={engineTuningExpanded}
        engineTuningState={engineTuningState}
        trackSamplingState={trackSamplingState}
        onEngineTuningExpandedChange={setEngineTuningExpanded}
      />

      {hasFeedback ? (
        <div className="configurator-feedback-stack">
          {actions.controlError !== null ? (
            <Notice tone="error">{actions.controlError}</Notice>
          ) : null}
          {engineTuningError !== null ? <Notice tone="error">{engineTuningError}</Notice> : null}
          {trackSamplingError !== null ? <Notice tone="error">{trackSamplingError}</Notice> : null}
          {previewEnabled && previewError !== null ? (
            <Notice tone="error">{previewError}</Notice>
          ) : null}
        </div>
      ) : null}

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
