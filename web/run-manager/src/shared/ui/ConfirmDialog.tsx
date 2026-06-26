// web/run-manager/src/shared/ui/ConfirmDialog.tsx
import * as Dialog from "@radix-ui/react-dialog";

import { Button } from "@/shared/ui/Button";

type ConfirmButtonTone = "danger" | "default";
type ConfirmButtonVariant = "accentSoft" | "primary" | "secondary";

interface ConfirmDialogProps {
  busy?: boolean;
  busyLabel?: string;
  cancelLabel?: string;
  confirmLabel: string;
  confirmTone?: ConfirmButtonTone;
  confirmVariant?: ConfirmButtonVariant;
  description: string;
  error?: string | null;
  open: boolean;
  title: string;
  onClose: () => void;
  onConfirm: () => void;
}

export function ConfirmDialog({
  busy = false,
  busyLabel = "Deleting...",
  cancelLabel = "Cancel",
  confirmLabel,
  confirmTone = "danger",
  confirmVariant = "secondary",
  description,
  error = null,
  open,
  title,
  onClose,
  onConfirm,
}: ConfirmDialogProps) {
  return (
    <Dialog.Root
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && !busy) {
          onClose();
        }
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-[rgba(11,16,24,0.72)]" />
        <Dialog.Content
          className="fixed left-1/2 top-1/2 z-50 w-[min(420px,calc(100vw-48px))] -translate-x-1/2 -translate-y-1/2 border border-app-border-strong bg-app-surface p-[22px]"
          onEscapeKeyDown={(event) => {
            if (busy) {
              event.preventDefault();
            }
          }}
          onPointerDownOutside={(event) => {
            if (busy) {
              event.preventDefault();
            }
          }}
        >
          <Dialog.Title className="m-0 text-lg font-semibold text-app-text">{title}</Dialog.Title>
          <Dialog.Description className="mt-3 mb-0 leading-normal text-app-muted">
            {description}
          </Dialog.Description>
          {error !== null ? (
            <p className="mt-3 mb-0 text-sm text-app-danger" role="alert">
              {error}
            </p>
          ) : null}
          <div className="mt-[22px] flex justify-end gap-2.5">
            <Button disabled={busy} onClick={onClose}>
              {cancelLabel}
            </Button>
            <Button disabled={busy} tone={confirmTone} variant={confirmVariant} onClick={onConfirm}>
              {busy ? busyLabel : confirmLabel}
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
