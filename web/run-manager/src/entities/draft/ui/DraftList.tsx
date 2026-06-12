// web/run-manager/src/entities/draft/ui/DraftList.tsx
import type { ManagedDraft } from "@/shared/api/contract";
import { cn } from "@/shared/ui/cn";
import { formatDate } from "@/shared/ui/format";
import { TrashIcon } from "@/shared/ui/icons";
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
    <div className="grid gap-2">
      <div className="grid grid-cols-[38px_minmax(0,1fr)_40px] items-center px-3.5 pb-1 text-[11px] font-bold tracking-[0.04em] text-app-muted uppercase">
        <label className={draftSelectCellClass}>
          <input
            aria-label="Select all drafts"
            checked={allDraftsSelected}
            className={draftCheckboxClass}
            disabled={isDeleting}
            type="checkbox"
            onChange={(event) => onSelectAll(event.currentTarget.checked)}
          />
        </label>
        <div className={draftRecordGridClass}>
          <span>Draft</span>
          <span>Envs</span>
          <span>Steps</span>
          <span>LR</span>
          <span>CNN</span>
          <span>Created at</span>
        </div>
        <span className="w-10" />
      </div>
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
    </div>
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
    <div className={draftRowClass(selected)}>
      <label className={draftSelectCellClass}>
        <input
          aria-label={`Select draft ${draft.name}`}
          checked={selected}
          className={draftCheckboxClass}
          disabled={isDeleting}
          type="checkbox"
          onChange={(event) => onToggleDraftSelection(draft.id, event.currentTarget.checked)}
        />
      </label>
      <button className={draftRecordButtonClass} type="button" onClick={() => onOpenDraft(draft)}>
        <span className="font-semibold text-app-text">{draft.name}</span>
        <span>{draft.config.train.num_envs} envs</span>
        <span>{draft.config.train.total_timesteps.toLocaleString()} steps</span>
        <span>{draft.config.train.learning_rate.toExponential(2)}</span>
        <span>{draft.config.policy.conv_profile}</span>
        <span className="whitespace-nowrap">{formatDate(draft.created_at)}</span>
      </button>
      <TooltipIconButton
        aria-label={`Delete draft ${draft.name}`}
        className="mr-2.5 justify-self-end"
        disabled={isDeleting}
        size="compact"
        tone="danger"
        tooltip="Delete draft"
        onClick={() => onRequestDelete(draft)}
      >
        <TrashIcon />
      </TooltipIconButton>
    </div>
  );
}

const draftRecordGridClass =
  "grid grid-cols-[2fr_0.8fr_1fr_0.9fr_1.1fr_1.45fr] items-center gap-3 [&>span]:whitespace-nowrap";

const draftRecordButtonClass = cn(
  draftRecordGridClass,
  "min-h-12 min-w-0 border-0 bg-transparent px-3.5 text-left text-app-muted",
);

const draftSelectCellClass = "grid min-h-full place-items-center";
const draftCheckboxClass = "h-4 w-4 accent-app-accent";

function draftRowClass(selected: boolean) {
  return cn(
    "grid min-h-12 grid-cols-[38px_minmax(0,1fr)_40px] items-center overflow-hidden rounded-lg border border-app-border bg-app-surface hover:border-app-border-strong hover:bg-app-surface-muted",
    selected ? "border-app-border-strong bg-app-surface-muted" : undefined,
  );
}
