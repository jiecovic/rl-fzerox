// web/run-manager/src/entities/runConfig/ui/sections/action/AuxiliaryBranchesDisclosure.tsx

import { ActionAuxStack } from "@/entities/runConfig/ui/sections/action/ActionLayout";
import { BranchToggles } from "@/entities/runConfig/ui/sections/action/branches/BranchToggles";
import { resetAuxiliaryBranchesAction } from "@/entities/runConfig/ui/sections/action/branches/model";
import type { AuxiliaryBranchesDisclosureProps } from "@/entities/runConfig/ui/sections/action/branches/types";
import { RuntimeCards } from "@/entities/runConfig/ui/sections/action/runtime/RuntimeCards";
import { ConfigDisclosure } from "@/shared/ui/config/ConfigDisclosure";

export function AuxiliaryBranchesDisclosure({
  checkpointLocked = false,
  config,
  defaultConfig,
  metadata,
  open,
  setOpen,
  updateAction,
  updatePolicy,
  updateTrain,
}: AuxiliaryBranchesDisclosureProps) {
  const action = config.action;

  return (
    <ConfigDisclosure
      onReset={() => {
        updateAction(
          resetAuxiliaryBranchesAction({
            action,
            checkpointLocked,
            defaultAction: defaultConfig.action,
          }),
        );
        updateTrain({
          actor_regularization: defaultConfig.train.actor_regularization,
        });
        updatePolicy({
          air_brake_on_logit: defaultConfig.policy.air_brake_on_logit,
          spin_idle_logit: defaultConfig.policy.spin_idle_logit,
        });
      }}
      onToggle={setOpen}
      open={open}
      title="Auxiliary branches"
    >
      <ActionAuxStack>
        <BranchToggles
          action={action}
          checkpointLocked={checkpointLocked}
          updateAction={updateAction}
        />
        <RuntimeCards
          action={action}
          checkpointLocked={checkpointLocked}
          defaultAction={defaultConfig.action}
          defaultPolicy={defaultConfig.policy}
          defaultTrain={defaultConfig.train}
          metadata={metadata}
          policy={config.policy}
          train={config.train}
          updateAction={updateAction}
          updatePolicy={updatePolicy}
          updateTrain={updateTrain}
        />
      </ActionAuxStack>
    </ConfigDisclosure>
  );
}
