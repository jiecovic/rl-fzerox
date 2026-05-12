// src/rl_fzerox/apps/run_manager/web/src/features/runs/RunTrackPoolPanel.tsx
import { useMemo, useState } from "react";
import {
  buildTrackPoolView,
  expectsTrackSamplingState,
  showTrackSamplingState,
  trackPoolEmptyMessage,
  trackSamplingModeLabel,
  trackSamplingUpdatedLabel,
} from "@/features/runs/track_pool_panel/model";
import { CupTabs, TrackPoolBody } from "@/features/runs/track_pool_panel/parts";
import type { RunTrackPoolPanelProps } from "@/features/runs/track_pool_panel/types";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";

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
    <div className="run-track-distribution-panel">
      <div className="run-track-distribution-header">
        <div>
          <strong>Track pool</strong>
          <div className="run-track-distribution-note">
            {visibleState === null
              ? trackSamplingModeLabel(run.config.tracks.sampling_mode)
              : `${trackSamplingModeLabel(run.config.tracks.sampling_mode)} · updated ${trackSamplingUpdatedLabel(run)}`}
          </div>
        </div>
        <div className="run-track-distribution-meta">
          {visibleState === null
            ? `${poolView.totalCourses} courses`
            : `${poolView.totalEpisodes.toLocaleString()} episodes · ${poolView.totalEnvSteps.toLocaleString()} env steps`}
        </div>
        <button
          className="secondary-button run-track-distribution-reset"
          type="button"
          disabled={!canReset || isResetting}
          onClick={() => setConfirmResetOpen(true)}
        >
          {isResetting ? "Resetting..." : "Reset stats"}
        </button>
      </div>
      <CupTabs activeCup={activeCup} cups={poolView.cups} onSelectCup={selectCup} />
      {visibleState === null ? (
        <div className="run-track-distribution-empty">{trackPoolEmptyMessage(run)}</div>
      ) : activeCup === null ? null : (
        <TrackPoolBody activeCup={activeCup} />
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
