import { ConfigDisclosure } from "@/features/configurator/ConfigDisclosure";
import { BranchToggles } from "@/features/configurator/sections/action/auxiliary_branches/BranchToggles";
import { resetAuxiliaryBranchesAction } from "@/features/configurator/sections/action/auxiliary_branches/model";
import { RuntimeCards } from "@/features/configurator/sections/action/auxiliary_branches/RuntimeCards";
import type { AuxiliaryBranchesDisclosureProps } from "@/features/configurator/sections/action/auxiliary_branches/types";

export function AuxiliaryBranchesDisclosure({
  checkpointLocked = false,
  config,
  defaultConfig,
  metadata,
  open,
  setOpen,
  updateAction,
}: AuxiliaryBranchesDisclosureProps) {
  const action = config.action;

  return (
    <ConfigDisclosure
      onReset={() =>
        updateAction(
          resetAuxiliaryBranchesAction({
            action,
            checkpointLocked,
            defaultAction: defaultConfig.action,
          }),
        )
      }
      onToggle={setOpen}
      open={open}
      title="Auxiliary branches"
    >
      <div className="action-aux-stack">
        <BranchToggles
          action={action}
          checkpointLocked={checkpointLocked}
          updateAction={updateAction}
        />
        <RuntimeCards
          action={action}
          checkpointLocked={checkpointLocked}
          defaultAction={defaultConfig.action}
          metadata={metadata}
          updateAction={updateAction}
        />
      </div>
    </ConfigDisclosure>
  );
}
