// src/rl_fzerox/apps/run_manager/web/src/shared/ui/ConfirmDialog.tsx
import { useEffect } from "react";

import { Button } from "@/shared/ui/Button";

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
    <div className="fixed inset-0 z-40 grid place-items-center bg-[rgba(11,16,24,0.72)] p-6">
      <button
        aria-label="Close dialog"
        className="absolute inset-0 border-0 bg-transparent p-0"
        type="button"
        disabled={busy}
        onClick={onClose}
      />
      <div
        aria-modal="true"
        className="relative z-[1] w-[min(420px,100%)] border border-app-border-strong bg-app-surface p-[22px]"
        role="dialog"
        aria-label={title}
      >
        <h3 className="m-0 text-lg font-semibold text-app-text">{title}</h3>
        <p className="mt-3 mb-0 leading-normal text-app-muted">{description}</p>
        <div className="mt-[22px] flex justify-end gap-2.5">
          <Button disabled={busy} onClick={onClose}>
            {cancelLabel}
          </Button>
          <Button tone="danger" disabled={busy} onClick={onConfirm}>
            {busy ? "Deleting..." : confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  );
}
