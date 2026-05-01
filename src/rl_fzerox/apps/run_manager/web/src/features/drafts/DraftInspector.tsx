import type { ManagedDraft } from "@/shared/api/contract";
import { ConfigSummary } from "@/shared/ui/ConfigSummary";
import { formatDate } from "@/shared/ui/format";
import { Panel, PanelHeader } from "@/shared/ui/Panel";

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
