// web/run-manager/src/entities/draft/ui/DraftList.tsx
import type { ManagedDraft } from "@/shared/api/contract";
import { formatDate } from "@/shared/ui/format";
import { TrashIcon } from "@/shared/ui/icons";
import {
  ListActionsCell,
  ListActionsHeaderCell,
  ListRow,
  ListSelectAllHeaderCell,
  ListSelectionCell,
  ListTable,
  ListTableHead,
} from "@/shared/ui/ListTable";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface DraftListProps {
  allDraftsSelected: boolean;
  drafts: readonly ManagedDraft[];
  isDeleting: boolean;
  onOpenDraft: (draft: ManagedDraft) => void;
  onRequestDelete: (draft: ManagedDraft) => void;
  onSelectAll: (selected: boolean) => void;
  onToggleDraftSelection: (draftId: string, selected: boolean) => void;
  selectedDraftIds: ReadonlySet<string>;
}

export function DraftList({
  allDraftsSelected,
  drafts,
  isDeleting,
  onOpenDraft,
  onRequestDelete,
  onSelectAll,
  onToggleDraftSelection,
  selectedDraftIds,
}: DraftListProps) {
  return (
    <ListTable minWidthClass="min-w-[780px]">
      <ListTableHead>
        <tr>
          <ListSelectAllHeaderCell
            aria-label="Select all drafts"
            checked={allDraftsSelected}
            disabled={isDeleting}
            onChange={onSelectAll}
          />
          <th className="px-4 py-3">Draft</th>
          <th className="px-4 py-3">Envs</th>
          <th className="px-4 py-3">Steps</th>
          <th className="px-4 py-3">LR</th>
          <th className="px-4 py-3">CNN</th>
          <th className="px-4 py-3">Created</th>
          <ListActionsHeaderCell />
        </tr>
      </ListTableHead>
      <tbody>
        {drafts.map((draft) => (
          <DraftRow
            draft={draft}
            isDeleting={isDeleting}
            key={draft.id}
            selected={selectedDraftIds.has(draft.id)}
            onOpenDraft={onOpenDraft}
            onRequestDelete={onRequestDelete}
            onToggleDraftSelection={onToggleDraftSelection}
          />
        ))}
      </tbody>
    </ListTable>
  );
}

function DraftRow({
  draft,
  isDeleting,
  onOpenDraft,
  onRequestDelete,
  onToggleDraftSelection,
  selected,
}: {
  draft: ManagedDraft;
  isDeleting: boolean;
  onOpenDraft: (draft: ManagedDraft) => void;
  onRequestDelete: (draft: ManagedDraft) => void;
  onToggleDraftSelection: (draftId: string, selected: boolean) => void;
  selected: boolean;
}) {
  return (
    <ListRow selected={selected} onOpen={() => onOpenDraft(draft)}>
      <ListSelectionCell
        aria-label={`Select draft ${draft.name}`}
        checked={selected}
        disabled={isDeleting}
        onChange={(checked) => onToggleDraftSelection(draft.id, checked)}
      />
      <td className="px-4 py-3 align-top">
        <strong className="text-app-text">{draft.name}</strong>
      </td>
      <td className="px-4 py-3 align-top text-app-muted">{draft.config.train.num_envs}</td>
      <td className="px-4 py-3 align-top text-app-muted">
        {draft.config.train.total_timesteps.toLocaleString()}
      </td>
      <td className="px-4 py-3 align-top text-app-muted">
        {draft.config.train.learning_rate.toExponential(2)}
      </td>
      <td className="px-4 py-3 align-top text-app-muted">{draft.config.policy.conv_profile}</td>
      <td className="px-4 py-3 align-top whitespace-nowrap text-app-muted">
        {formatDate(draft.created_at)}
      </td>
      <ListActionsCell>
        <TooltipIconButton
          aria-label={`Delete draft ${draft.name}`}
          disabled={isDeleting}
          size="compact"
          tone="danger"
          tooltip="Delete draft"
          onClick={() => onRequestDelete(draft)}
        >
          <TrashIcon />
        </TooltipIconButton>
      </ListActionsCell>
    </ListRow>
  );
}
