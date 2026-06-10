// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/action/AuxiliaryBranchesDisclosure.tsx
import { ConfigDisclosure } from "@/features/configurator/ConfigDisclosure";
import { ActionAuxStack } from "@/features/configurator/sections/action/ActionLayout";
import { BranchToggles } from "@/features/configurator/sections/action/branches/BranchToggles";
import { resetAuxiliaryBranchesAction } from "@/features/configurator/sections/action/branches/model";
import type { AuxiliaryBranchesDisclosureProps } from "@/features/configurator/sections/action/branches/types";
import { RuntimeCards } from "@/features/configurator/sections/action/runtime/RuntimeCards";

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
