// web/run-manager/src/widgets/runWorkspace/RunWorkspace.tsx
import { useEffect, useRef, useState } from "react";
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
import { Panel, PanelHeader } from "@/shared/ui/Panel";
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
  onClearCourseAltBaselines: (runId: string, courseKey: string) => Promise<void>;
  onCreateDraftFromRun: (runId: string) => Promise<void>;
  onFork: (runId: string, artifact: "latest" | "best", copyAltBaselines: boolean) => Promise<void>;
  onGlobalError: (message: string | null) => void;
  onOpenDirectory: (runId: string) => Promise<void>;
  onRename: (runId: string, name: string) => Promise<void>;
  onResetEngineTuning: (runId: string) => Promise<void>;
  onResume: (runId: string) => Promise<void>;
  onResetTrackPool: (runId: string) => Promise<void>;
  onSelectEvaluationSourceRun: (runId: string) => void;
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
  onClearCourseAltBaselines,
  onCreateDraftFromRun,
  onFork,
  onGlobalError,
  onOpenDirectory,
  onRename,
  onResetEngineTuning,
  onResume,
  onResetTrackPool,
  onSelectEvaluationSourceRun,
  onShowCharts,
  onStop,
  onWatch,
  run,
}: RunWorkspaceProps) {
  const [runName, setRunName] = useState(run.name);
  const lastObservedWatchFailureRef = useRef<{ key: string | null; runId: string } | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [configSection, setConfigSection] = useState<ConfigSection>("training");
  const [isResettingEngineTuning, setIsResettingEngineTuning] = useState(false);
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
    onGlobalError,
    onClearAltBaselines,
    onClearCourseAltBaselines,
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
  const { engineTuningError, engineTuningState, setEngineTuningState } = useRunEngineTuningState(
    run.id,
    run.status,
    engineTuningEnabled && engineTuningExpanded,
    actions.selectedWatchArtifact,
  );
  const watchFailure = latestWatchFailure(run);
  const watchFailureKeyValue = watchFailureKey(watchFailure);
  const watchFailureMessage = watchFailure?.message ?? null;
  useEffect(() => {
    setRunName(run.name);
  }, [run.name]);

  useEffect(() => {
    const previous = lastObservedWatchFailureRef.current;
    if (previous === null || previous.runId !== run.id) {
      lastObservedWatchFailureRef.current = { key: watchFailureKeyValue, runId: run.id };
      return;
    }
    if (watchFailureKeyValue !== previous.key) {
      lastObservedWatchFailureRef.current = { key: watchFailureKeyValue, runId: run.id };
      if (watchFailureMessage !== null) {
        onGlobalError(watchFailureMessage);
      }
    }
  }, [onGlobalError, run.id, watchFailureKeyValue, watchFailureMessage]);

  useEffect(() => {
    const message = engineTuningError ?? trackSamplingError ?? previewError ?? null;
    if (message !== null) {
      onGlobalError(message);
    }
  }, [engineTuningError, onGlobalError, previewError, trackSamplingError]);

  async function submitRunRename(name: string) {
    setRunName(name);
    const renamed = await actions.renameRunLabel(name);
    if (renamed) {
      setRenameDialogOpen(false);
    }
  }

  async function resetEngineTuning() {
    setIsResettingEngineTuning(true);
    onGlobalError(null);
    try {
      await onResetEngineTuning(run.id);
      setEngineTuningState(null);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "failed to reset engine tuner";
      onGlobalError(message);
    } finally {
      setIsResettingEngineTuning(false);
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
        onSelectEvaluationSourceRun={onSelectEvaluationSourceRun}
        onShowCharts={onShowCharts}
        run={run}
        engineTuningExpanded={engineTuningExpanded}
        canResetEngineTuning={run.status !== "running"}
        engineTuningState={engineTuningState}
        isResettingEngineTuning={isResettingEngineTuning}
        trackSamplingState={trackSamplingState}
        onEngineTuningExpandedChange={setEngineTuningExpanded}
        onResetEngineTuning={() => void resetEngineTuning()}
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

function latestWatchFailure(run: ManagedRunDetail): { created_at: string; message: string } | null {
  const event = run.recent_events.find((candidate) => candidate.kind === "watch_failed");
  if (event === undefined) {
    return null;
  }
  return {
    created_at: event.created_at,
    message: event.message,
  };
}

function watchFailureKey(event: { created_at: string; message: string } | null): string | null {
  if (event === null) {
    return null;
  }
  return `${event.created_at}\n${event.message}`;
}
