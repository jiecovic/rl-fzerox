import type { ManagedRun } from "@/shared/api/contract";
import { Notice, Panel, PanelHeader } from "@/shared/ui/Panel";

interface RunsPanelProps {
  runs: ManagedRun[];
  onOpenRun: (run: ManagedRun) => void;
}

export function RunsPanel({ runs, onOpenRun }: RunsPanelProps) {
  if (runs.length === 0) {
    return <Notice>No launched runs yet. Training launch is not wired in this slice.</Notice>;
  }

  return (
    <Panel>
      <PanelHeader title="Runs" subtitle="Click a row to inspect it." />
      <div className="record-list">
        {runs.map((run) => (
          <button className="record-row" key={run.id} type="button" onClick={() => onOpenRun(run)}>
            <span className="record-name">{run.name}</span>
            <span>{run.status}</span>
            <span>{run.config.train.num_envs} envs</span>
            <span>{run.config.train.total_timesteps.toLocaleString()} steps</span>
            <span>{run.config.policy.conv_profile}</span>
          </button>
        ))}
      </div>
    </Panel>
  );
}
