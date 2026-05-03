import { ResetIcon } from "@/features/configurator/fields";

import { SaveDraftIcon, UnsavedDot } from "./icons";

interface ActionBarProps {
  canSave: boolean;
  canUpdate: boolean;
  hasLoadedDraft: boolean;
  isDirty: boolean;
  isSaving: boolean;
  isUpdating: boolean;
  onResetToDefault: () => void;
  onResetToDraft: () => void;
  onSaveDraft: () => void;
  onUpdateDraft: () => void;
}

export function ActionBar({
  canSave,
  canUpdate,
  hasLoadedDraft,
  isDirty,
  isSaving,
  isUpdating,
  onResetToDefault,
  onResetToDraft,
  onSaveDraft,
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
            disabled={isSaving || isUpdating}
            onClick={onResetToDraft}
          >
            <ResetIcon />
            <span>Reset to draft</span>
          </button>
        ) : null}
        <button
          className="secondary-button draft-action-button"
          type="button"
          disabled={isSaving || isUpdating}
          onClick={onResetToDefault}
        >
          <ResetIcon />
          <span>Reset to default</span>
        </button>
        <button className="primary-button" type="button" disabled>
          Train
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
