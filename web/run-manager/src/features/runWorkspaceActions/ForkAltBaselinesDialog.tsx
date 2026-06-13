// web/run-manager/src/features/runWorkspaceActions/ForkAltBaselinesDialog.tsx
import * as Dialog from "@radix-ui/react-dialog";

import { Button } from "@/shared/ui/Button";

interface ForkAltBaselinesDialogProps {
  altBaselineCount: number;
  open: boolean;
  onClose: () => void;
  onSelect: (copyAltBaselines: boolean) => void;
}

export function ForkAltBaselinesDialog({
  altBaselineCount,
  open,
  onClose,
  onSelect,
}: ForkAltBaselinesDialogProps) {
  const baselineLabel =
    altBaselineCount === 1 ? "1 active alt baseline" : `${altBaselineCount} active alt baselines`;
  return (
    <Dialog.Root
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) {
          onClose();
        }
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-[rgba(11,16,24,0.72)]" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[min(460px,calc(100vw-48px))] -translate-x-1/2 -translate-y-1/2 border border-app-border-strong bg-app-surface p-[22px]">
          <Dialog.Title className="m-0 text-lg font-semibold text-app-text">
            Fork alt baselines
          </Dialog.Title>
          <Dialog.Description className="mt-3 mb-0 leading-normal text-app-muted">
            This run has {baselineLabel}. Copy them into the fork?
          </Dialog.Description>
          <div className="mt-[22px] flex flex-wrap justify-end gap-2.5">
            <Button onClick={onClose}>Cancel</Button>
            <Button onClick={() => onSelect(false)}>Do not copy</Button>
            <Button variant="primary" onClick={() => onSelect(true)}>
              Copy alt baselines
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
