// src/rl_fzerox/apps/run_manager/web/src/app/workspace/runEquality.ts
import type { ManagedRun } from "@/shared/api/contract";

export function sameRunPayload(left: readonly ManagedRun[], right: readonly ManagedRun[]) {
  if (left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index += 1) {
    if (!sameRunSummary(left[index], right[index])) {
      return false;
    }
  }
  return true;
}

function sameRunSummary(left: ManagedRun, right: ManagedRun) {
  if (
    left.id !== right.id ||
    left.name !== right.name ||
    left.status !== right.status ||
    left.config_hash !== right.config_hash ||
    left.action_repeat !== right.action_repeat ||
    left.created_at !== right.created_at ||
    left.lineage_id !== right.lineage_id ||
    left.lineage_step_offset !== right.lineage_step_offset ||
    left.parent_run_id !== right.parent_run_id ||
    left.source_run_id !== right.source_run_id ||
    left.source_artifact !== right.source_artifact ||
    left.source_num_timesteps !== right.source_num_timesteps ||
    left.started_at !== right.started_at ||
    left.stopped_at !== right.stopped_at ||
    left.pending_command !== right.pending_command ||
    left.worker_heartbeat_at !== right.worker_heartbeat_at
  ) {
    return false;
  }
  if (left.lineage_groups.join("\0") !== right.lineage_groups.join("\0")) {
    return false;
  }
  return sameRuntime(left.runtime, right.runtime) && sameRecentEvents(left, right);
}

function sameRuntime(left: ManagedRun["runtime"], right: ManagedRun["runtime"]) {
  if (left === null || right === null) {
    return left === right;
  }
  return (
    left.total_timesteps === right.total_timesteps &&
    left.num_timesteps === right.num_timesteps &&
    left.progress_fraction === right.progress_fraction &&
    left.updated_at === right.updated_at &&
    left.fps === right.fps &&
    left.episode_reward_mean === right.episode_reward_mean &&
    left.episode_length_mean === right.episode_length_mean &&
    left.approx_kl === right.approx_kl &&
    left.entropy_loss === right.entropy_loss &&
    left.value_loss === right.value_loss &&
    left.policy_gradient_loss === right.policy_gradient_loss
  );
}

function sameRecentEvents(left: ManagedRun, right: ManagedRun) {
  if (left.recent_events.length !== right.recent_events.length) {
    return false;
  }
  return left.recent_events.every((event, index) => {
    const other = right.recent_events[index];
    return (
      other !== undefined &&
      event.created_at === other.created_at &&
      event.kind === other.kind &&
      event.message === other.message
    );
  });
}
