// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/action/regularization/SteeringGuardCard.tsx
import { FieldLabel, NumberField } from "@/features/configurator/fields";
import {
  ActionCard,
  ActionControlHeader,
  ActionControlPanel,
  ActionFieldGrid,
  ActionFieldset,
  ActionNote,
} from "@/features/configurator/sections/action/ActionLayout";
import type {
  AuxiliaryActionConfig,
  UpdateTrain,
} from "@/features/configurator/sections/action/branches/types";
import type { ManagedRunConfig } from "@/shared/api/contract";
import { SwitchButton } from "@/shared/ui/Field";

interface SteeringGuardCardProps {
  action: AuxiliaryActionConfig;
  defaultTrain: ManagedRunConfig["train"];
  train: ManagedRunConfig["train"];
  updateTrain: UpdateTrain;
}

export function SteeringGuardCard({
  action,
  defaultTrain,
  train,
  updateTrain,
}: SteeringGuardCardProps) {
  const steerStdCapLossWeight = train.actor_regularization.steer_std_cap_loss_weight;

  return (
    <ActionCard
      description="Optionally keep continuous steering exploration inside a useful range."
      title="Steering guard"
    >
      {action.steering_mode === "continuous" ? null : (
        <ActionNote>
          Steering is discrete right now, so this continuous std guard is inactive.
        </ActionNote>
      )}
      <ActionFieldset disabled={action.steering_mode !== "continuous"}>
        <ActionControlPanel>
          <ActionControlHeader>
            <FieldLabel
              help="Add a policy-side loss that softly caps the continuous steering Gaussian std. It does not change the action head or clamp the sampled action."
              label="Steer std cap loss"
              onReset={() =>
                updateTrain({
                  actor_regularization: {
                    ...train.actor_regularization,
                    steer_std_cap_loss_weight:
                      defaultTrain.actor_regularization.steer_std_cap_loss_weight,
                    steer_std_cap: defaultTrain.actor_regularization.steer_std_cap,
                  },
                })
              }
            />
            <SwitchButton
              active={steerStdCapLossWeight > 0}
              className="justify-self-end"
              label="Steer std cap loss"
              onClick={() =>
                updateTrain({
                  actor_regularization: {
                    ...train.actor_regularization,
                    steer_std_cap_loss_weight: steerStdCapLossWeight > 0 ? 0 : 0.01,
                  },
                })
              }
            />
          </ActionControlHeader>
          <ActionFieldGrid columns="two" disabled={steerStdCapLossWeight <= 0}>
            <NumberField
              help="Coefficient for relu(steer_std - cap)^2. Keep this low; it is a guardrail, not a steering objective."
              label="Loss weight"
              resetValue={defaultTrain.actor_regularization.steer_std_cap_loss_weight}
              step="0.001"
              value={steerStdCapLossWeight > 0 ? steerStdCapLossWeight : 0}
              onChange={(value) =>
                updateTrain({
                  actor_regularization: {
                    ...train.actor_regularization,
                    steer_std_cap_loss_weight: Math.max(0, value),
                  },
                })
              }
            />
            <NumberField
              help="Soft upper bound for the continuous steering distribution std before action clipping."
              label="Std cap"
              resetValue={defaultTrain.actor_regularization.steer_std_cap}
              step="0.05"
              value={train.actor_regularization.steer_std_cap}
              onChange={(value) =>
                updateTrain({
                  actor_regularization: {
                    ...train.actor_regularization,
                    steer_std_cap: Math.max(0.001, value),
                  },
                })
              }
            />
          </ActionFieldGrid>
        </ActionControlPanel>
      </ActionFieldset>
    </ActionCard>
  );
}
