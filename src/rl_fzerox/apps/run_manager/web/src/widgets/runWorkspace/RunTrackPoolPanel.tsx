// src/rl_fzerox/apps/run_manager/web/src/widgets/runWorkspace/RunTrackPoolPanel.tsx
import { useMemo, useState } from "react";
import { Button } from "@/shared/ui/Button";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import {
  buildTrackPoolView,
  expectsTrackSamplingState,
  showTrackSamplingState,
  trackPoolEmptyMessage,
  trackSamplingModeLabel,
  trackSamplingUpdatedLabel,
} from "@/widgets/runWorkspace/track_pool_panel/model";
import { CupTabs, TrackPoolBody } from "@/widgets/runWorkspace/track_pool_panel/parts";
import type { RunTrackPoolPanelProps } from "@/widgets/runWorkspace/track_pool_panel/types";

export function RunTrackPoolPanel({
  canReset,
  isResetting,
  metadata,
  onReset,
  run,
  state,
}: RunTrackPoolPanelProps) {
  const visibleState = showTrackSamplingState(state) ? state : null;
  const poolView = useMemo(
    () => buildTrackPoolView(metadata, run, visibleState),
    [metadata, run, visibleState],
  );
  const poolSelectionKey = `${run.id}:${run.config.tracks.selected_course_ids.join("\0")}`;
  const firstCupId = poolView.cups[0]?.id ?? null;
  const [cupSelection, setCupSelection] = useState<{ cupId: string; poolKey: string } | null>(null);
  const [confirmResetOpen, setConfirmResetOpen] = useState(false);
  const selectedCupId =
    cupSelection?.poolKey === poolSelectionKey &&
    poolView.cups.some((cup) => cup.id === cupSelection.cupId)
      ? cupSelection.cupId
      : firstCupId;

  function selectCup(cupId: string) {
    setCupSelection({ cupId, poolKey: poolSelectionKey });
  }

  if (!expectsTrackSamplingState(run, poolView.totalCourses)) {
    return null;
  }

  const activeCup = poolView.cups.find((cup) => cup.id === selectedCupId) ?? null;

  return (
    <div className="col-span-full border border-app-border bg-app-surface">
      <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-3 border-b border-app-border px-3.5 py-3 max-[760px]:grid-cols-1">
        <div>
          <strong>Track pool</strong>
          <div className="text-xs text-app-muted">
            {visibleState === null
              ? trackSamplingModeLabel(run.config.tracks.sampling_mode)
              : `${trackSamplingModeLabel(run.config.tracks.sampling_mode)} · updated ${trackSamplingUpdatedLabel(run)}`}
          </div>
        </div>
        <div className="text-xs text-app-muted">
          {visibleState === null
            ? `${poolView.totalCourses} courses`
            : `${poolView.totalEpisodes.toLocaleString()} episodes · ${poolView.totalEnvSteps.toLocaleString()} env steps`}
        </div>
        <Button
          className="h-8 justify-self-end px-3 text-xs max-[760px]:justify-self-start"
          type="button"
          disabled={!canReset || isResetting}
          onClick={() => setConfirmResetOpen(true)}
        >
          {isResetting ? "Resetting..." : "Reset stats"}
        </Button>
      </div>
      <CupTabs activeCup={activeCup} cups={poolView.cups} onSelectCup={selectCup} />
      {visibleState === null ? (
        <div className="px-3.5 py-3 text-sm text-app-muted">{trackPoolEmptyMessage(run)}</div>
      ) : activeCup === null ? null : (
        <TrackPoolBody
          activeCup={activeCup}
          stepMetricLabel={poolView.stepMetricLabel}
          showStepTarget={poolView.showStepTarget}
          xCupRegenerationMinEpisodes={
            run.config.tracks.x_cup_auto_regeneration.enabled
              ? run.config.tracks.x_cup_auto_regeneration.min_episodes
              : null
          }
          xCupRegenerationThreshold={
            run.config.tracks.x_cup_auto_regeneration.enabled
              ? run.config.tracks.x_cup_auto_regeneration.completion_threshold
              : null
          }
        />
      )}
      <ConfirmDialog
        confirmLabel="Reset stats"
        description={`Reset the course distribution history for "${run.name}"? This clears the tracked episode, finish, and env-step counts for this run.`}
        open={confirmResetOpen}
        title="Reset track-pool stats"
        onClose={() => setConfirmResetOpen(false)}
        onConfirm={() => {
          setConfirmResetOpen(false);
          onReset();
        }}
      />
    </div>
  );
}
