// src/rl_fzerox/apps/run_manager/web/src/shared/ui/ConfirmDialog.tsx
import { useEffect } from "react";

interface ConfirmDialogProps {
  busy?: boolean;
  cancelLabel?: string;
  confirmLabel: string;
  description: string;
  open: boolean;
  title: string;
  onClose: () => void;
  onConfirm: () => void;
}

export function ConfirmDialog({
  busy = false,
  cancelLabel = "Cancel",
  confirmLabel,
  description,
  open,
  title,
  onClose,
  onConfirm,
}: ConfirmDialogProps) {
  useEffect(() => {
    if (!open) {
      return undefined;
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && !busy) {
        onClose();
      }
    }

    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("keydown", handleEscape);
    };
  }, [busy, onClose, open]);

  if (!open) {
    return null;
  }

  return (
    <div className="dialog-backdrop">
      <button
        aria-label="Close dialog"
        className="dialog-dismiss-surface"
        type="button"
        disabled={busy}
        onClick={onClose}
      />
      <div aria-modal="true" className="dialog-panel" role="dialog" aria-label={title}>
        <h3>{title}</h3>
        <p>{description}</p>
        <div className="dialog-actions">
          <button className="secondary-button" type="button" disabled={busy} onClick={onClose}>
            {cancelLabel}
          </button>
          <button
            className="secondary-button danger"
            type="button"
            disabled={busy}
            onClick={onConfirm}
          >
            {busy ? "Deleting..." : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
