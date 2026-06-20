// web/run-manager/src/widgets/configurator/EngineTuningSourceDialog.tsx
import * as Dialog from "@radix-ui/react-dialog";

import type { EngineTunerBackend, EngineTuningSourceAction } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";

interface EngineTuningSourceDialogProps {
  lossy: boolean;
  open: boolean;
  sourceBackend: EngineTunerBackend | null;
  onClose: () => void;
  onSelect: (action: EngineTuningSourceAction) => void;
}

export function EngineTuningSourceDialog({
  lossy,
  open,
  sourceBackend,
  onClose,
  onSelect,
}: EngineTuningSourceDialogProps) {
  const sourceLabel =
    sourceBackend === null ? "the source checkpoint" : sourceBackendLabel(sourceBackend);
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
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 grid w-[min(500px,calc(100vw-48px))] -translate-x-1/2 -translate-y-1/2 gap-5 border border-app-border-strong bg-app-surface p-[22px]">
          <Dialog.Title className="m-0 text-lg font-semibold text-app-text">
            Engine tuner history
          </Dialog.Title>
          <Dialog.Description className="m-0 leading-normal text-app-muted">
            This fork can take over engine-tuning history from {sourceLabel}, or start with a clean
            tuner.
            {lossy
              ? " The tuner backend, engine range, or slider buckets changed, so taking over history can discard incompatible data."
              : ""}
          </Dialog.Description>
          <div className="flex flex-wrap justify-end gap-2.5">
            <Button onClick={onClose}>Cancel</Button>
            <Button onClick={() => onSelect("discard")}>Start clean</Button>
            <Button variant="primary" onClick={() => onSelect("convert")}>
              Take over history
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function sourceBackendLabel(backend: EngineTunerBackend) {
  if (backend === "gaussian_process") {
    return "Gaussian-process tuning";
  }
  if (backend === "mlp_ensemble") {
    return "MLP ensemble tuning";
  }
  return "bandit tuning";
}
