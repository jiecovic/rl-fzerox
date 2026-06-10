// web/run-manager/src/shared/ui/ConfirmDialog.tsx
import * as Dialog from "@radix-ui/react-dialog";

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
          <div className="mt-[22px] flex justify-end gap-2.5">
            <Button disabled={busy} onClick={onClose}>
              {cancelLabel}
            </Button>
            <Button tone="danger" disabled={busy} onClick={onConfirm}>
              {busy ? "Deleting..." : confirmLabel}
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
