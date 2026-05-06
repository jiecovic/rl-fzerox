// src/rl_fzerox/apps/run_manager/web/src/features/configurator/configurator/ActionBar.tsx
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
    <div className="configurator-actions-row">
      <div className="section-actions">
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
          <button
            className="secondary-button draft-action-button"
            type="button"
            disabled={isSaving || isUpdating || isTraining}
            onClick={onResetToDraft}
          >
            <ResetIcon />
            <span>Reset to draft</span>
          </button>
        ) : null}
        <button
          className="secondary-button draft-action-button"
          type="button"
          disabled={isSaving || isUpdating || isTraining}
          onClick={onResetToDefault}
        >
          <ResetIcon />
          <span>Reset to default</span>
        </button>
        <button className="primary-button" type="button" disabled={!canTrain} onClick={onTrain}>
          {isTraining ? "Launching..." : "Train"}
        </button>
      </div>
    </div>
  );
}

function saveButtonClass(hasLoadedDraft: boolean, isDirty: boolean) {
  return hasLoadedDraft || !isDirty
    ? "secondary-button draft-action-button draft-commit-button draft-save-button"
    : "secondary-button draft-action-button draft-commit-button draft-save-button dirty-action-button";
}

function updateButtonClass(isDirty: boolean) {
  return isDirty
    ? "secondary-button draft-action-button draft-commit-button draft-update-button dirty-action-button"
    : "secondary-button draft-action-button draft-commit-button draft-update-button";
}

function UnsavedDot({ active }: { active: boolean }) {
  return (
    <span aria-hidden="true" className={active ? "dirty-action-dot active" : "dirty-action-dot"} />
  );
}
