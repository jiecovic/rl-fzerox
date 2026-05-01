import type { ManagedRun } from "@/shared/api/contract";
import { ConfigSummary } from "@/shared/ui/ConfigSummary";
import { formatDate } from "@/shared/ui/format";
import { Panel, PanelHeader } from "@/shared/ui/Panel";

export function RunInspector({ run }: { run: ManagedRun }) {
  return (
    <Panel>
      <PanelHeader
        title={run.name}
        subtitle={`${run.status} · created ${formatDate(run.created_at)}`}
      />
      <ConfigSummary config={run.config} />
      <div className="action-row">
        <button className="primary-button" type="button" disabled>
          Fork
        </button>
        <button className="secondary-button" type="button" disabled>
          Watch
        </button>
      </div>
    </Panel>
  );
}
