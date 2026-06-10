// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/ActionBar.tsx
import { Button, buttonClassName } from "@/shared/ui/Button";
import { cn } from "@/shared/ui/cn";
import { ResetIcon, SaveDraftIcon } from "@/shared/ui/icons";

interface ActionBarProps {
  canSave: boolean;
  canTrain: boolean;
  canUpdate: boolean;
  hasLoadedDraft: boolean;
  isDirty: boolean;
  isSaving: boolean;
  isTraining: boolean;
  isUpdating: boolean;
  onResetToDefault: () => void;
  onResetToDraft: () => void;
  onSaveDraft: () => void;
  onTrain: () => void;
  onUpdateDraft: () => void;
}

export function ActionBar({
  canSave,
  canTrain,
  canUpdate,
  hasLoadedDraft,
  isDirty,
  isSaving,
  isTraining,
  isUpdating,
  onResetToDefault,
  onResetToDraft,
  onSaveDraft,
  onTrain,
  onUpdateDraft,
}: ActionBarProps) {
  return (
    <div className="mb-4 flex justify-end">
      <div className="flex flex-wrap items-center justify-end gap-2">
        <button
          className={saveButtonClass(hasLoadedDraft, isDirty)}
          type="button"
          disabled={!canSave}
          onClick={onSaveDraft}
        >
          <SaveDraftIcon />
          <UnsavedDot active={!hasLoadedDraft && isDirty} />
          <span>
            {isSaving ? "Saving..." : hasLoadedDraft ? "Save as new draft" : "Save draft"}
          </span>
        </button>
        {hasLoadedDraft ? (
          <button
            className={updateButtonClass(isDirty)}
            type="button"
            disabled={!canUpdate}
            onClick={onUpdateDraft}
          >
            <SaveDraftIcon />
            <UnsavedDot active={isDirty} />
            <span>{isUpdating ? "Saving..." : "Save draft"}</span>
          </button>
        ) : null}
        {hasLoadedDraft ? (
          <Button
            className="gap-2 tabular-nums [&_svg]:shrink-0"
            disabled={isSaving || isUpdating || isTraining}
            onClick={onResetToDraft}
          >
            <ResetIcon />
            <span>Reset to draft</span>
          </Button>
        ) : null}
        <Button
          className="gap-2 tabular-nums [&_svg]:shrink-0"
          disabled={isSaving || isUpdating || isTraining}
          onClick={onResetToDefault}
        >
          <ResetIcon />
          <span>Reset to default</span>
        </Button>
        <Button variant="primary" disabled={!canTrain} onClick={onTrain}>
          {isTraining ? "Launching..." : "Train"}
        </Button>
      </div>
    </div>
  );
}

function saveButtonClass(hasLoadedDraft: boolean, isDirty: boolean) {
  return buttonClassName({
    className: cn(
      "min-w-0 gap-2 tabular-nums [&_svg]:shrink-0",
      hasLoadedDraft || !isDirty ? undefined : dirtyButtonClass,
    ),
  });
}

function updateButtonClass(isDirty: boolean) {
  return buttonClassName({
    className: cn(
      "min-w-0 gap-2 tabular-nums [&_svg]:shrink-0",
      isDirty ? dirtyButtonClass : undefined,
    ),
    variant: "accentSoft",
  });
}

function UnsavedDot({ active }: { active: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "h-2 w-2 flex-none rounded-full bg-app-accent shadow-[0_0_0_3px_color-mix(in_srgb,var(--accent)_16%,transparent)]",
        active ? "visible opacity-100" : "invisible opacity-0",
      )}
    />
  );
}

const dirtyButtonClass =
  "border-app-accent shadow-[inset_0_0_0_1px_color-mix(in_srgb,var(--accent)_46%,transparent)]";
