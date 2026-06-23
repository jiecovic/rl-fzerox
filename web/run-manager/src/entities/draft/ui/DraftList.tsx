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
    <div className="overflow-x-auto border border-app-border bg-app-surface">
      <table className="w-full min-w-[780px] border-collapse text-left text-sm">
        <thead className="border-b border-app-border text-xs font-bold tracking-[0.04em] text-app-muted uppercase">
          <tr>
            <th className="w-10 px-4 py-3">
              <label className="grid place-items-center" data-draft-row-interaction>
                <input
                  aria-label="Select all drafts"
                  checked={allDraftsSelected}
                  className={draftCheckboxClass}
                  disabled={isDeleting}
                  type="checkbox"
                  onChange={(event) => onSelectAll(event.currentTarget.checked)}
                />
              </label>
            </th>
            <th className="px-4 py-3">Draft</th>
            <th className="px-4 py-3">Envs</th>
            <th className="px-4 py-3">Steps</th>
            <th className="px-4 py-3">LR</th>
            <th className="px-4 py-3">CNN</th>
            <th className="px-4 py-3">Created</th>
            <th className="w-12 px-4 py-3">
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
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
      </table>
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
    <tr
      className={draftRowClass(selected)}
      tabIndex={0}
      onClick={(event) => {
        if (isDraftRowInteractionTarget(event.target)) {
          return;
        }
        onOpenDraft(draft);
      }}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget) {
          return;
        }
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpenDraft(draft);
        }
      }}
    >
      <td className="px-4 py-3 align-top" data-draft-row-interaction>
        <label className="grid place-items-center">
          <input
            aria-label={`Select draft ${draft.name}`}
            checked={selected}
            className={draftCheckboxClass}
            disabled={isDeleting}
            type="checkbox"
            onChange={(event) => onToggleDraftSelection(draft.id, event.currentTarget.checked)}
          />
        </label>
      </td>
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
      <td className="px-4 py-3 align-top" data-draft-row-interaction>
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
      </td>
    </tr>
  );
}

const draftCheckboxClass = "h-4 w-4 accent-app-accent";

function draftRowClass(selected: boolean) {
  return cn(
    "cursor-pointer border-b border-app-border transition-colors last:border-b-0 hover:bg-app-surface-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-app-accent",
    selected ? "bg-app-surface-muted" : undefined,
  );
}

function isDraftRowInteractionTarget(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    target.closest("[data-draft-row-interaction],a,button,input,label,select,textarea") !== null
  );
}
