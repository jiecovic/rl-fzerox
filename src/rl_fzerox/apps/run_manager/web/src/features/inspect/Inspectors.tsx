import { ConfigSummary } from "@/components/ui/ConfigSummary";
import { formatDate } from "@/components/ui/format";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import type { ManagedDraft, ManagedRun } from "@/contract";

interface DraftInspectorProps {
  draft: ManagedDraft;
  onDelete: () => void;
}

export function DraftInspector({ draft, onDelete }: DraftInspectorProps) {
  return (
    <Panel>
      <PanelHeader title={draft.name} subtitle={`saved ${formatDate(draft.updated_at)}`} />
      <ConfigSummary config={draft.config} />
      <div className="action-row">
        <button className="primary-button" type="button" disabled>
          Train
        </button>
        <button className="secondary-button danger" type="button" onClick={onDelete}>
          Delete draft
        </button>
      </div>
    </Panel>
  );
}

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
