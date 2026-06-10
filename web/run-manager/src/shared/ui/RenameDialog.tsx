// web/run-manager/src/shared/ui/RenameDialog.tsx
import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";

import { Button } from "@/shared/ui/Button";
import { FieldInput, FieldShell } from "@/shared/ui/Field";

interface RenameDialogProps {
  busy?: boolean;
  initialName: string;
  label: string;
  open: boolean;
  title: string;
  onClose: () => void;
  onSubmit: (name: string) => void;
}

export function RenameDialog({
  busy = false,
  initialName,
  label,
  open,
  title,
  onClose,
  onSubmit,
}: RenameDialogProps) {
  const [name, setName] = useState(initialName);
  const trimmedName = name.trim();
  const canSubmit = trimmedName.length > 0 && trimmedName !== initialName.trim() && !busy;

  useEffect(() => {
    if (open) {
      setName(initialName);
    }
  }, [initialName, open]);

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
          className="fixed left-1/2 top-1/2 z-50 grid w-[min(460px,calc(100vw-48px))] -translate-x-1/2 -translate-y-1/2 gap-5 border border-app-border-strong bg-app-surface p-[22px]"
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
          <form
            className="grid gap-5"
            onSubmit={(event) => {
              event.preventDefault();
              if (canSubmit) {
                onSubmit(trimmedName);
              }
            }}
          >
            <FieldShell>
              <span>{label}</span>
              <FieldInput
                autoFocus
                aria-label={label}
                value={name}
                onChange={(event) => setName(event.currentTarget.value)}
              />
            </FieldShell>
            <div className="flex justify-end gap-2.5">
              <Button disabled={busy} onClick={onClose}>
                Cancel
              </Button>
              <Button disabled={!canSubmit} variant="primary" type="submit">
                {busy ? "Saving" : "Save"}
              </Button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
