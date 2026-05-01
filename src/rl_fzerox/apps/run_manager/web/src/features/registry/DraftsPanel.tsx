import { Notice, Panel, PanelHeader } from "@/components/ui/Panel";
import type { ManagedDraft } from "@/contract";

interface DraftsPanelProps {
  drafts: ManagedDraft[];
  onOpenDraft: (draft: ManagedDraft) => void;
}

export function DraftsPanel({ drafts, onOpenDraft }: DraftsPanelProps) {
  if (drafts.length === 0) {
    return <Notice>No drafts yet. Configure a run and save it first.</Notice>;
  }

  return (
    <Panel>
      <PanelHeader title="Drafts" subtitle="Click a row to inspect it." />
      <div className="record-list">
        {drafts.map((draft) => (
          <button
            className="record-row"
            key={draft.id}
            type="button"
            onClick={() => onOpenDraft(draft)}
          >
            <span className="record-name">{draft.name}</span>
            <span>{draft.config.train.num_envs} envs</span>
            <span>{draft.config.train.total_timesteps.toLocaleString()} steps</span>
            <span>{draft.config.train.learning_rate.toExponential(2)}</span>
            <span>{draft.config.policy.conv_profile}</span>
          </button>
        ))}
      </div>
    </Panel>
  );
}
