// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/action/regularization/SignedBiasBalanceCard.tsx
import { FieldLabel, NumberField } from "@/features/configurator/fields";
import {
  ActionCard,
  ActionControlHeader,
  ActionControlPanel,
  ActionFieldGrid,
  ActionFieldset,
  ActionNote,
  ActionPanelGrid,
} from "@/features/configurator/sections/action/ActionLayout";
import type {
  AuxiliaryActionConfig,
  UpdateTrain,
} from "@/features/configurator/sections/action/branches/types";
import type { ManagedRunConfig } from "@/shared/api/contract";
import { SwitchButton } from "@/shared/ui/Field";

interface SignedBiasBalanceCardProps {
  action: AuxiliaryActionConfig;
  defaultTrain: ManagedRunConfig["train"];
  train: ManagedRunConfig["train"];
  updateTrain: UpdateTrain;
}

export function SignedBiasBalanceCard({
  action,
  defaultTrain,
  train,
  updateTrain,
}: SignedBiasBalanceCardProps) {
  const steerBalanceLossWeight = train.actor_regularization.steer_signed_balance_loss_weight;
  const leanBalanceLossWeight = train.actor_regularization.lean_signed_balance_loss_weight;

  return (
    <ActionCard
      description="Optional actor losses that discourage persistent left/right distribution bias."
      title="Signed bias balance"
    >
      <ActionPanelGrid>
        <ActionControlPanel>
          {action.steering_mode === "continuous" ? null : (
            <ActionNote>Steer balance is inactive while steering is discrete.</ActionNote>
          )}
          <ActionFieldset disabled={action.steering_mode !== "continuous"}>
            <ActionControlHeader>
              <FieldLabel
                help="Penalize persistent continuous steering mean outside the deadzone. This is a soft actor regularizer, not action masking or reward shaping."
                label="Steer bias loss"
                onReset={() =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      steer_signed_balance_loss_weight:
                        defaultTrain.actor_regularization.steer_signed_balance_loss_weight,
                      steer_signed_balance_deadzone:
                        defaultTrain.actor_regularization.steer_signed_balance_deadzone,
                    },
                  })
                }
              />
              <SwitchButton
                active={steerBalanceLossWeight > 0}
                className="justify-self-end"
                label="Steer bias loss"
                onClick={() =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      steer_signed_balance_loss_weight: steerBalanceLossWeight > 0 ? 0 : 0.0005,
                    },
                  })
                }
              />
            </ActionControlHeader>
            <ActionFieldGrid columns="two" disabled={steerBalanceLossWeight <= 0}>
              <NumberField
                help="Coefficient for relu(abs(batch_mean_steer) - deadzone)^2. Use a small value so steering remains free to turn."
                label="Loss weight"
                resetValue={defaultTrain.actor_regularization.steer_signed_balance_loss_weight}
                step="0.0001"
                value={steerBalanceLossWeight > 0 ? steerBalanceLossWeight : 0}
                onChange={(value) =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      steer_signed_balance_loss_weight: Math.max(0, value),
                    },
                  })
                }
              />
              <NumberField
                help="Allowed absolute minibatch steering bias before the loss activates."
                label="Deadzone"
                resetValue={defaultTrain.actor_regularization.steer_signed_balance_deadzone}
                step="0.05"
                value={train.actor_regularization.steer_signed_balance_deadzone}
                onChange={(value) =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      steer_signed_balance_deadzone: Math.min(1, Math.max(0, value)),
                    },
                  })
                }
              />
            </ActionFieldGrid>
          </ActionFieldset>
        </ActionControlPanel>
        <ActionControlPanel>
          {action.include_lean ? null : (
            <ActionNote>Lean balance is inactive while lean is not in the output.</ActionNote>
          )}
          <ActionFieldset disabled={!action.include_lean}>
            <ActionControlHeader>
              <FieldLabel
                help="Penalize persistent expected lean imbalance outside the deadzone. Three-way and four-way lean use P(right)-P(left); split lean uses P(right engaged)-P(left engaged)."
                label="Lean bias loss"
                onReset={() =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      lean_signed_balance_loss_weight:
                        defaultTrain.actor_regularization.lean_signed_balance_loss_weight,
                      lean_signed_balance_deadzone:
                        defaultTrain.actor_regularization.lean_signed_balance_deadzone,
                    },
                  })
                }
              />
              <SwitchButton
                active={leanBalanceLossWeight > 0}
                className="justify-self-end"
                label="Lean bias loss"
                onClick={() =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      lean_signed_balance_loss_weight: leanBalanceLossWeight > 0 ? 0 : 0.001,
                    },
                  })
                }
              />
            </ActionControlHeader>
            <ActionFieldGrid columns="two" disabled={leanBalanceLossWeight <= 0}>
              <NumberField
                help="Coefficient for relu(abs(batch_expected_lean) - deadzone)^2. Start small; this should remove bias, not prevent side attacks."
                label="Loss weight"
                resetValue={defaultTrain.actor_regularization.lean_signed_balance_loss_weight}
                step="0.001"
                value={leanBalanceLossWeight > 0 ? leanBalanceLossWeight : 0}
                onChange={(value) =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      lean_signed_balance_loss_weight: Math.max(0, value),
                    },
                  })
                }
              />
              <NumberField
                help="Allowed absolute minibatch lean bias before the loss activates."
                label="Deadzone"
                resetValue={defaultTrain.actor_regularization.lean_signed_balance_deadzone}
                step="0.05"
                value={train.actor_regularization.lean_signed_balance_deadzone}
                onChange={(value) =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      lean_signed_balance_deadzone: Math.min(1, Math.max(0, value)),
                    },
                  })
                }
              />
            </ActionFieldGrid>
          </ActionFieldset>
        </ActionControlPanel>
      </ActionPanelGrid>
    </ActionCard>
  );
}
