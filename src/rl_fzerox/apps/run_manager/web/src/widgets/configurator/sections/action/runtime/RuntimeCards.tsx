// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/action/runtime/RuntimeCards.tsx

import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";
import { ActionRuntimeStack } from "@/widgets/configurator/sections/action/ActionLayout";
import type {
  AuxiliaryActionConfig,
  UpdateAction,
  UpdatePolicy,
  UpdateTrain,
} from "@/widgets/configurator/sections/action/branches/types";
import { PitchControlCard } from "@/widgets/configurator/sections/action/regularization/PitchControlCard";
import { SignedBiasBalanceCard } from "@/widgets/configurator/sections/action/regularization/SignedBiasBalanceCard";
import { SteeringGuardCard } from "@/widgets/configurator/sections/action/regularization/SteeringGuardCard";
import { AirBrakeCard } from "@/widgets/configurator/sections/action/runtime/AirBrakeCard";
import { BoostGuardsCard } from "@/widgets/configurator/sections/action/runtime/BoostGuardsCard";
import { LeanControlCard } from "@/widgets/configurator/sections/action/runtime/LeanControlCard";
import { SpinControlCard } from "@/widgets/configurator/sections/action/runtime/SpinControlCard";

interface RuntimeCardsProps {
  action: AuxiliaryActionConfig;
  checkpointLocked: boolean;
  defaultAction: AuxiliaryActionConfig;
  defaultPolicy: ManagedRunConfig["policy"];
  defaultTrain: ManagedRunConfig["train"];
  metadata: ConfigMetadata;
  policy: ManagedRunConfig["policy"];
  train: ManagedRunConfig["train"];
  updateAction: UpdateAction;
  updatePolicy: UpdatePolicy;
  updateTrain: UpdateTrain;
}

export function RuntimeCards({
  action,
  checkpointLocked,
  defaultAction,
  defaultPolicy,
  defaultTrain,
  metadata,
  policy,
  train,
  updateAction,
  updatePolicy,
  updateTrain,
}: RuntimeCardsProps) {
  return (
    <ActionRuntimeStack>
      <AirBrakeCard
        action={action}
        checkpointLocked={checkpointLocked}
        defaultAction={defaultAction}
        metadata={metadata}
        updateAction={updateAction}
      />
      <BoostGuardsCard action={action} defaultAction={defaultAction} updateAction={updateAction} />
      <LeanControlCard
        action={action}
        checkpointLocked={checkpointLocked}
        defaultAction={defaultAction}
        metadata={metadata}
        updateAction={updateAction}
      />
      <SpinControlCard
        action={action}
        defaultAction={defaultAction}
        defaultPolicy={defaultPolicy}
        policy={policy}
        updateAction={updateAction}
        updatePolicy={updatePolicy}
      />
      <SteeringGuardCard
        action={action}
        defaultTrain={defaultTrain}
        train={train}
        updateTrain={updateTrain}
      />
      <SignedBiasBalanceCard
        action={action}
        defaultTrain={defaultTrain}
        train={train}
        updateTrain={updateTrain}
      />
      <PitchControlCard
        action={action}
        checkpointLocked={checkpointLocked}
        defaultAction={defaultAction}
        defaultTrain={defaultTrain}
        metadata={metadata}
        train={train}
        updateAction={updateAction}
        updateTrain={updateTrain}
      />
    </ActionRuntimeStack>
  );
}
