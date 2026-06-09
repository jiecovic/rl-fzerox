// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/action/auxiliary_branches/ActorRegularizationCards.tsx
import {
  FieldLabel,
  NumberField,
  RangeIntegerField,
  SegmentedChoiceStrip,
} from "@/features/configurator/fields";
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
  UpdateAction,
  UpdateTrain,
} from "@/features/configurator/sections/action/auxiliary_branches/types";
import { normalizeOddBucketCount } from "@/features/configurator/sections/action/model";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";
import { FieldShell, SwitchButton } from "@/shared/ui/Field";

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

interface PitchControlCardProps {
  action: AuxiliaryActionConfig;
  checkpointLocked: boolean;
  defaultAction: AuxiliaryActionConfig;
  defaultTrain: ManagedRunConfig["train"];
  metadata: ConfigMetadata;
  train: ManagedRunConfig["train"];
  updateAction: UpdateAction;
  updateTrain: UpdateTrain;
}

export function PitchControlCard({
  action,
  checkpointLocked,
  defaultAction,
  defaultTrain,
  metadata,
  train,
  updateAction,
  updateTrain,
}: PitchControlCardProps) {
  const groundedPitchLossWeight = train.actor_regularization.grounded_pitch_neutral_loss_weight;
  const pitchStdCapLossWeight = train.actor_regularization.pitch_std_cap_loss_weight;

  return (
    <ActionCard
      description="Choose whether pitch is one analog lane or a discrete bucket head."
      title="Pitch control"
    >
      {action.include_pitch ? null : (
        <ActionNote>
          Pitch is not in the action output right now, so these controls are inactive.
        </ActionNote>
      )}
      <ActionFieldset disabled={!action.include_pitch}>
        <fieldset className="fork-lock-fieldset" disabled={checkpointLocked}>
          <FieldShell>
            <FieldLabel
              help="Choose whether pitch is a continuous analog lane or a discrete bucket head."
              label="Pitch mode"
            />
            <SegmentedChoiceStrip
              ariaLabel="Pitch mode"
              options={metadata.steering_modes.map((option) => ({
                active: action.pitch_mode === option.value,
                key: option.value,
                label: option.label,
                onClick: () =>
                  updateAction({
                    pitch_mode: option.value as ManagedRunConfig["action"]["pitch_mode"],
                  }),
              }))}
            />
          </FieldShell>
          {action.pitch_mode === "discrete" ? (
            <>
              <RangeIntegerField
                help="Odd bucket counts preserve one neutral center action while adding more upward and downward pitch resolution."
                label="Pitch buckets"
                max={31}
                min={3}
                rangeStep={2}
                resetValue={defaultAction.pitch_buckets}
                ticks={[
                  { label: "3", value: 3 },
                  { label: "5", value: 5 },
                  { label: "9", value: 9 },
                  { label: "15", value: 15 },
                  { label: "31", value: 31 },
                ]}
                value={action.pitch_buckets}
                onChange={(value) =>
                  updateAction({ pitch_buckets: normalizeOddBucketCount(value) })
                }
              />
              <FieldShell>
                <FieldLabel
                  help="Mask discrete pitch to the neutral bucket while the vehicle is grounded. Continuous pitch uses the actor loss controls instead."
                  label="Ground mask"
                  onReset={() =>
                    updateAction({ mask_pitch_on_ground: defaultAction.mask_pitch_on_ground })
                  }
                />
                <SegmentedChoiceStrip
                  ariaLabel="Discrete pitch ground mask"
                  options={[
                    {
                      active: !action.mask_pitch_on_ground,
                      key: "off",
                      label: "Off",
                      onClick: () => updateAction({ mask_pitch_on_ground: false }),
                    },
                    {
                      active: action.mask_pitch_on_ground,
                      key: "on",
                      label: "On",
                      onClick: () => updateAction({ mask_pitch_on_ground: true }),
                    },
                  ]}
                />
              </FieldShell>
            </>
          ) : null}
        </fieldset>
        <ActionNote>
          {action.pitch_mode === "continuous"
            ? "Continuous pitch maps the vertical stick axis directly."
            : "Discrete pitch losses use the categorical distribution over pitch buckets."}
        </ActionNote>
        <ActionPanelGrid>
          <ActionControlPanel>
            <ActionControlHeader>
              <FieldLabel
                help="Add a policy-side loss that pulls expected pitch toward neutral while the vehicle is grounded."
                label="Grounded neutral loss"
                onReset={() =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      grounded_pitch_neutral_loss_weight:
                        defaultTrain.actor_regularization.grounded_pitch_neutral_loss_weight,
                    },
                  })
                }
              />
              <SwitchButton
                active={groundedPitchLossWeight > 0}
                className="justify-self-end"
                label="Grounded pitch neutral loss"
                onClick={() =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      grounded_pitch_neutral_loss_weight: groundedPitchLossWeight > 0 ? 0 : 0.01,
                    },
                  })
                }
              />
            </ActionControlHeader>
            <ActionFieldGrid columns="single" disabled={groundedPitchLossWeight <= 0}>
              <NumberField
                help="Coefficient for squared expected pitch while grounded. It affects the actor loss only and does not mask the action."
                label="Mean loss weight"
                resetValue={defaultTrain.actor_regularization.grounded_pitch_neutral_loss_weight}
                step="0.001"
                value={groundedPitchLossWeight > 0 ? groundedPitchLossWeight : 0}
                onChange={(value) =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      grounded_pitch_neutral_loss_weight: Math.max(0, value),
                    },
                  })
                }
              />
            </ActionFieldGrid>
          </ActionControlPanel>
          <ActionControlPanel>
            <ActionControlHeader>
              <FieldLabel
                help="Add one policy-side loss that caps pitch distribution std. Discrete pitch only uses the grounded cap; continuous pitch also uses the airborne cap."
                label="Pitch std cap loss"
                onReset={() =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      pitch_std_cap_loss_weight:
                        defaultTrain.actor_regularization.pitch_std_cap_loss_weight,
                      grounded_pitch_std_cap:
                        defaultTrain.actor_regularization.grounded_pitch_std_cap,
                      airborne_pitch_std_cap:
                        defaultTrain.actor_regularization.airborne_pitch_std_cap,
                    },
                  })
                }
              />
              <SwitchButton
                active={pitchStdCapLossWeight > 0}
                className="justify-self-end"
                label="Pitch std cap loss"
                onClick={() =>
                  updateTrain({
                    actor_regularization: {
                      ...train.actor_regularization,
                      pitch_std_cap_loss_weight: pitchStdCapLossWeight > 0 ? 0 : 0.05,
                    },
                  })
                }
              />
            </ActionControlHeader>
            {action.pitch_mode === "discrete" ? (
              <ActionFieldGrid columns="two" disabled={pitchStdCapLossWeight <= 0}>
                <PitchStdCapFields
                  defaultTrain={defaultTrain}
                  pitchStdCapLossWeight={pitchStdCapLossWeight}
                  train={train}
                  updateTrain={updateTrain}
                />
              </ActionFieldGrid>
            ) : (
              <ActionFieldGrid columns="three" disabled={pitchStdCapLossWeight <= 0}>
                <PitchStdCapFields
                  defaultTrain={defaultTrain}
                  includeAirborneCap
                  pitchStdCapLossWeight={pitchStdCapLossWeight}
                  train={train}
                  updateTrain={updateTrain}
                />
              </ActionFieldGrid>
            )}
          </ActionControlPanel>
        </ActionPanelGrid>
      </ActionFieldset>
    </ActionCard>
  );
}

function PitchStdCapFields({
  defaultTrain,
  includeAirborneCap = false,
  pitchStdCapLossWeight,
  train,
  updateTrain,
}: {
  defaultTrain: ManagedRunConfig["train"];
  includeAirborneCap?: boolean;
  pitchStdCapLossWeight: number;
  train: ManagedRunConfig["train"];
  updateTrain: UpdateTrain;
}) {
  return (
    <>
      <NumberField
        help="Shared coefficient for relu(pitch_std - state_cap)^2. Grounded and airborne samples are disjoint, so this weight is shared."
        label="Loss weight"
        resetValue={defaultTrain.actor_regularization.pitch_std_cap_loss_weight}
        step="0.001"
        value={pitchStdCapLossWeight > 0 ? pitchStdCapLossWeight : 0}
        onChange={(value) =>
          updateTrain({
            actor_regularization: {
              ...train.actor_regularization,
              pitch_std_cap_loss_weight: Math.max(0, value),
            },
          })
        }
      />
      <NumberField
        help="Soft upper bound for the pitch distribution std while grounded."
        label="Grounded cap"
        resetValue={defaultTrain.actor_regularization.grounded_pitch_std_cap}
        step="0.05"
        value={train.actor_regularization.grounded_pitch_std_cap}
        onChange={(value) =>
          updateTrain({
            actor_regularization: {
              ...train.actor_regularization,
              grounded_pitch_std_cap: Math.max(0.001, value),
            },
          })
        }
      />
      {includeAirborneCap ? (
        <NumberField
          help="Soft upper bound for the continuous pitch distribution std while airborne."
          label="Airborne cap"
          resetValue={defaultTrain.actor_regularization.airborne_pitch_std_cap}
          step="0.05"
          value={train.actor_regularization.airborne_pitch_std_cap}
          onChange={(value) =>
            updateTrain({
              actor_regularization: {
                ...train.actor_regularization,
                airborne_pitch_std_cap: Math.max(0.001, value),
              },
            })
          }
        />
      ) : null}
    </>
  );
}
