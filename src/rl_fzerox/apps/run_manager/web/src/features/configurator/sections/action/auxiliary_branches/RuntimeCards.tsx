// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/action/auxiliary_branches/RuntimeCards.tsx
import {
  FieldLabel,
  IntegerField,
  NumberField,
  OptionalNumberField,
  RangeIntegerField,
  RangeNumberField,
  SegmentedChoiceStrip,
} from "@/features/configurator/fields";
import { formatEditableDecimal } from "@/features/configurator/fields/format";
import { useEditableNumberInput } from "@/features/configurator/fields/numberInput";
import { resetHandler } from "@/features/configurator/fields/reset";
import type {
  AuxiliaryActionConfig,
  UpdateAction,
  UpdatePolicy,
  UpdateTrain,
} from "@/features/configurator/sections/action/auxiliary_branches/types";
import {
  airBrakeModeDescription,
  leanModeDescription,
} from "@/features/configurator/sections/action/descriptions";
import { normalizeOddBucketCount } from "@/features/configurator/sections/action/model";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";
import { FieldInput, FieldNote, FieldShell, SwitchButton } from "@/shared/ui/Field";

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
  const groundedPitchLossWeight = train.actor_regularization.grounded_pitch_neutral_loss_weight;
  const pitchStdCapLossWeight = train.actor_regularization.pitch_std_cap_loss_weight;
  const steerStdCapLossWeight = train.actor_regularization.steer_std_cap_loss_weight;

  return (
    <div className="action-behavior-grid">
      <section className="action-runtime-card">
        <div className="action-runtime-header config-disclosure-copy">
          <strong>Air brake</strong>
          <small>Choose the output family, optional PWM shaping, and grounded-use guard.</small>
        </div>
        {action.include_air_brake ? null : (
          <p className="action-note">
            Air brake is not in the action output right now, so these runtime rules are inactive.
          </p>
        )}
        <fieldset
          className="dependent-fieldset action-runtime-fields"
          disabled={!action.include_air_brake}
        >
          <fieldset className="fork-lock-fieldset" disabled={checkpointLocked}>
            <div className="field-shell">
              <FieldLabel
                help="Choose between a continuous PWM air-brake lane and a stock-style digital air-brake button."
                label="Air brake mode"
              />
              <SegmentedChoiceStrip
                ariaLabel="Air brake mode"
                options={metadata.drive_modes.map((option) => ({
                  active: action.air_brake_mode === option.value,
                  key: option.value,
                  label: option.label,
                  tooltip: airBrakeModeDescription(
                    option.value as ManagedRunConfig["action"]["air_brake_mode"],
                  ),
                  onClick: () =>
                    updateAction({
                      air_brake_mode: option.value as ManagedRunConfig["action"]["air_brake_mode"],
                    }),
                }))}
              />
            </div>
            {action.air_brake_mode === "pwm" ? (
              <div className="action-axis-fields action-axis-fields-triple">
                <RangeNumberField
                  help="Ignore small positive air-brake values below this threshold."
                  label="Deadzone"
                  max={0.5}
                  min={0}
                  rangeStep={0.01}
                  resetValue={defaultAction.continuous_air_brake_deadzone}
                  ticks={[
                    { label: "0", value: 0 },
                    { label: "0.1", value: 0.1 },
                    { label: "0.25", value: 0.25 },
                    { label: "0.5", value: 0.5 },
                  ]}
                  value={action.continuous_air_brake_deadzone}
                  onChange={(value) => updateAction({ continuous_air_brake_deadzone: value })}
                />
                <RangeNumberField
                  help="Clamp air-brake duty to full once the continuous brake lane reaches this value."
                  label="Full threshold"
                  max={1}
                  min={0.1}
                  rangeStep={0.01}
                  resetValue={defaultAction.continuous_air_brake_full_threshold}
                  ticks={[
                    { label: "0.1", value: 0.1 },
                    { label: "0.5", value: 0.5 },
                    { label: "0.85", value: 0.85 },
                    { label: "1", value: 1 },
                  ]}
                  value={action.continuous_air_brake_full_threshold}
                  onChange={(value) => updateAction({ continuous_air_brake_full_threshold: value })}
                />
                <RangeNumberField
                  help="Minimum air-brake duty once the lane is above the deadzone. Zero keeps the first engaged pulses sparse; higher values make early braking more assertive."
                  label="Minimum duty"
                  max={1}
                  min={0}
                  rangeStep={0.01}
                  resetValue={defaultAction.continuous_air_brake_min_duty}
                  ticks={[
                    { label: "0", value: 0 },
                    { label: "0.25", value: 0.25 },
                    { label: "0.5", value: 0.5 },
                    { label: "1", value: 1 },
                  ]}
                  value={action.continuous_air_brake_min_duty}
                  onChange={(value) => updateAction({ continuous_air_brake_min_duty: value })}
                />
              </div>
            ) : null}
          </fieldset>
          <div className="field-shell">
            <FieldLabel
              help="When enabled, the air-brake branch is masked back to neutral while grounded. Turn this off to let the policy use air brake freely on ground too."
              label="Mask on ground"
            />
            <SegmentedChoiceStrip
              ariaLabel="Air brake grounded mask"
              options={[
                {
                  active: !action.mask_air_brake_on_ground,
                  key: "allow_on_ground",
                  label: "Off",
                  onClick: () => updateAction({ mask_air_brake_on_ground: false }),
                },
                {
                  active: action.mask_air_brake_on_ground,
                  key: "mask_on_ground",
                  label: "On",
                  onClick: () => updateAction({ mask_air_brake_on_ground: true }),
                },
              ]}
            />
          </div>
        </fieldset>
      </section>

      <section className="action-runtime-card">
        <div className="action-runtime-header config-disclosure-copy">
          <strong>Boost guards</strong>
          <small>Only allow manual boost when these runtime conditions are satisfied.</small>
        </div>
        {action.include_boost ? null : (
          <p className="action-note">
            Boost is not in the action output right now, so these runtime rules are inactive.
          </p>
        )}
        <fieldset
          className="dependent-fieldset action-runtime-fields"
          disabled={!action.include_boost}
        >
          <div className="action-runtime-two-col">
            <OptionalNumberField
              defaultValue={900}
              help="Optionally block manual boost once the vehicle is above this speed. Leave this empty to ignore speed and rely on the normal unlock plus the energy guard."
              label="Only allow below speed"
              max={2000}
              min={0}
              resetValue={defaultAction.boost_unmask_max_speed_kph}
              step="10"
              value={action.boost_unmask_max_speed_kph}
              onChange={(value) => updateAction({ boost_unmask_max_speed_kph: value })}
            />
            <RangeNumberField
              help="Require at least this much energy before the manual boost branch is allowed."
              label="Minimum energy %"
              max={100}
              min={0}
              rangeStep={1}
              resetValue={defaultAction.boost_min_energy_fraction * 100}
              ticks={[
                { label: "0", value: 0 },
                { label: "10", value: 10 },
                { label: "25", value: 25 },
                { label: "50", value: 50 },
                { label: "100", value: 100 },
              ]}
              value={action.boost_min_energy_fraction * 100}
              onChange={(value) => updateAction({ boost_min_energy_fraction: value / 100 })}
            />
          </div>
          <div className="action-runtime-two-col">
            <div className="field-shell">
              <FieldLabel
                help="When enabled, manual boost is masked while a manual boost or dash-pad boost effect is already active."
                label="Mask while boosted"
                onReset={() =>
                  updateAction({ mask_boost_when_active: defaultAction.mask_boost_when_active })
                }
              />
              <SegmentedChoiceStrip
                ariaLabel="Boost active mask"
                options={[
                  {
                    active: !action.mask_boost_when_active,
                    key: "allow_while_boosted",
                    label: "Off",
                    onClick: () => updateAction({ mask_boost_when_active: false }),
                  },
                  {
                    active: action.mask_boost_when_active,
                    key: "mask_while_boosted",
                    label: "On",
                    onClick: () => updateAction({ mask_boost_when_active: true }),
                  },
                ]}
              />
            </div>
            <div className="field-shell">
              <FieldLabel
                help="When enabled, manual boost is masked while the vehicle is airborne."
                label="Mask while airborne"
                onReset={() =>
                  updateAction({
                    mask_boost_when_airborne: defaultAction.mask_boost_when_airborne,
                  })
                }
              />
              <SegmentedChoiceStrip
                ariaLabel="Boost airborne mask"
                options={[
                  {
                    active: !action.mask_boost_when_airborne,
                    key: "allow_airborne_boost",
                    label: "Off",
                    onClick: () => updateAction({ mask_boost_when_airborne: false }),
                  },
                  {
                    active: action.mask_boost_when_airborne,
                    key: "mask_airborne_boost",
                    label: "On",
                    onClick: () => updateAction({ mask_boost_when_airborne: true }),
                  },
                ]}
              />
            </div>
          </div>
          <div className="action-runtime-two-col">
            <IntegerField
              help="Allow a manual boost decision only once per this many env steps. The native-frame spacing is derived from action repeat."
              label="Decision interval env steps"
              min={1}
              note={boostDecisionIntervalSummary(
                action.action_repeat,
                action.boost_decision_interval_steps,
              )}
              resetValue={defaultAction.boost_decision_interval_steps}
              value={action.boost_decision_interval_steps}
              onChange={(value) => updateAction({ boost_decision_interval_steps: value })}
            />
          </div>
          <div className="action-runtime-two-col">
            <IntegerField
              help="After a manual boost request, keep the boost branch masked for this many native frames. Useful as the spam guard when the active-boost mask is off."
              label="Request cooldown frames"
              resetValue={defaultAction.boost_request_lockout_frames}
              value={action.boost_request_lockout_frames}
              onChange={(value) => updateAction({ boost_request_lockout_frames: value })}
            />
          </div>
        </fieldset>
      </section>

      <section className="action-runtime-card">
        <div className="action-runtime-header config-disclosure-copy">
          <strong>Lean control</strong>
          <small>Define the lean output shape, optional guards, and any post-processing.</small>
        </div>
        {action.include_lean ? null : (
          <p className="action-note">
            Lean is not in the action output right now, so these runtime rules are inactive.
          </p>
        )}
        <fieldset
          className="dependent-fieldset action-runtime-fields"
          disabled={!action.include_lean}
        >
          <fieldset className="fork-lock-fieldset" disabled={checkpointLocked}>
            <div className="field-shell">
              <FieldLabel
                help="Choose whether lean is one categorical branch or two independent left and right button branches."
                label="Lean output"
              />
              <SegmentedChoiceStrip
                ariaLabel="Lean output"
                options={metadata.lean_output_modes.map((option) => ({
                  active: action.lean_output_mode === option.value,
                  key: option.value,
                  label: option.label,
                  onClick: () => {
                    const leanOutputMode =
                      option.value as ManagedRunConfig["action"]["lean_output_mode"];
                    updateAction({
                      lean_output_mode: leanOutputMode,
                      ...(leanOutputMode === "three_way"
                        ? {}
                        : { enable_spin: false, include_spin: false }),
                    });
                  },
                }))}
              />
            </div>
          </fieldset>
          {action.lean_output_mode === "three_way" ||
          action.lean_output_mode === "four_way_categorical" ? (
            <>
              <div className="field-shell">
                <FieldLabel
                  help="Choose how the categorical lean branch is post-processed before it reaches the emulator."
                  label="Lean mode"
                />
                <SegmentedChoiceStrip
                  ariaLabel="Lean mode"
                  options={metadata.lean_modes.map((option) => ({
                    active: action.lean_mode === option.value,
                    key: option.value,
                    label: option.label,
                    onClick: () =>
                      updateAction({
                        lean_mode: option.value as ManagedRunConfig["action"]["lean_mode"],
                      }),
                  }))}
                />
              </div>
              <p className="action-note">{leanModeDescription(action.lean_mode)}</p>
            </>
          ) : (
            <p className="action-note">
              Independent buttons expose separate left and right lean outputs. They can co-activate
              and always bypass lean hold or cooldown assistance.
            </p>
          )}
          <div className="action-runtime-two-col">
            <OptionalNumberField
              defaultValue={700}
              help="Optionally block lean below this vehicle speed."
              label="Only allow above speed"
              max={1500}
              min={0}
              resetValue={defaultAction.lean_unmask_min_speed_kph}
              step="10"
              value={action.lean_unmask_min_speed_kph}
              onChange={(value) => updateAction({ lean_unmask_min_speed_kph: value })}
            />
            <IntegerField
              help="Keep lean masked to idle for the first N frames of each episode."
              label="Initial lockout"
              value={action.lean_initial_lockout_frames}
              onChange={(value) => updateAction({ lean_initial_lockout_frames: value })}
            />
          </div>
        </fieldset>
      </section>

      <section className="action-runtime-card">
        <div className="action-runtime-header config-disclosure-copy">
          <strong>Spin control</strong>
          <small>Configure the native spin macro request guard.</small>
        </div>
        {action.include_spin ? null : (
          <p className="action-note">
            Spin is not in the action output right now, so this runtime guard is inactive.
          </p>
        )}
        <fieldset
          className="dependent-fieldset action-runtime-fields"
          disabled={!action.include_spin}
        >
          <div className="action-runtime-two-col">
            <IntegerField
              help="After a completed native spin macro, keep spin requests masked for this many native frames."
              label="Cooldown frames"
              resetValue={defaultAction.spin_cooldown_frames}
              value={action.spin_cooldown_frames}
              onChange={(value) => updateAction({ spin_cooldown_frames: value })}
            />
            <SpinIdleLogitField
              resetValue={defaultPolicy.spin_idle_logit}
              value={policy.spin_idle_logit}
              onChange={(value) => updatePolicy({ spin_idle_logit: value })}
            />
          </div>
        </fieldset>
      </section>

      <section className="action-runtime-card">
        <div className="action-runtime-header config-disclosure-copy">
          <strong>Steering guard</strong>
          <small>Optionally keep continuous steering exploration inside a useful range.</small>
        </div>
        {action.steering_mode === "continuous" ? null : (
          <p className="action-note">
            Steering is discrete right now, so this continuous std guard is inactive.
          </p>
        )}
        <fieldset
          className="dependent-fieldset action-runtime-fields"
          disabled={action.steering_mode !== "continuous"}
        >
          <div className="action-runtime-control-panel">
            <div className="action-runtime-control-header">
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
                className="action-runtime-compact-switch"
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
            </div>
            <fieldset
              className="dependent-fieldset action-runtime-two-col"
              disabled={steerStdCapLossWeight <= 0}
            >
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
            </fieldset>
          </div>
        </fieldset>
      </section>

      <section className="action-runtime-card">
        <div className="action-runtime-header config-disclosure-copy">
          <strong>Pitch control</strong>
          <small>Choose whether pitch is one analog lane or a discrete bucket head.</small>
        </div>
        {action.include_pitch ? null : (
          <p className="action-note">
            Pitch is not in the action output right now, so these controls are inactive.
          </p>
        )}
        <fieldset
          className="dependent-fieldset action-runtime-fields"
          disabled={!action.include_pitch}
        >
          <fieldset className="fork-lock-fieldset" disabled={checkpointLocked}>
            <div className="field-shell">
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
            </div>
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
                <div className="field-shell">
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
                </div>
              </>
            ) : null}
          </fieldset>
          <p className="action-note">
            {action.pitch_mode === "continuous"
              ? "Continuous pitch maps the vertical stick axis directly."
              : "Discrete pitch losses use the categorical distribution over pitch buckets."}
          </p>
          <div className="action-runtime-panel-grid">
            <div className="action-runtime-control-panel">
              <div className="action-runtime-control-header">
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
                  className="action-runtime-compact-switch"
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
              </div>
              <fieldset
                className="dependent-fieldset action-runtime-field-grid action-runtime-field-grid-single"
                disabled={groundedPitchLossWeight <= 0}
              >
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
              </fieldset>
            </div>
            <div className="action-runtime-control-panel action-runtime-control-panel-wide">
              <div className="action-runtime-control-header">
                <FieldLabel
                  help="Add one policy-side loss that caps pitch distribution std. Grounded and airborne samples use separate caps but share this weight."
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
                  className="action-runtime-compact-switch"
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
              </div>
              <fieldset
                className="dependent-fieldset action-runtime-field-grid action-runtime-field-grid-three"
                disabled={pitchStdCapLossWeight <= 0}
              >
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
                <NumberField
                  help="Soft upper bound for the pitch distribution std while airborne. No airborne mean-to-zero loss is added."
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
              </fieldset>
            </div>
          </div>
        </fieldset>
      </section>
    </div>
  );
}

function boostDecisionIntervalSummary(actionRepeat: number, intervalSteps: number): string {
  const repeatFrames = Math.max(1, Math.trunc(actionRepeat));
  const envStepInterval = Math.max(1, Math.trunc(intervalSteps));
  const nativeFrames = envStepInterval * repeatFrames;
  const decisionsPerSecond = 60 / nativeFrames;
  return [
    envStepInterval === 1 ? "Every env step" : `Every ${envStepInterval} env steps`,
    `${nativeFrames} native frames`,
    `${formatCadence(decisionsPerSecond)} decisions/s`,
  ].join(" · ");
}

function formatCadence(value: number): string {
  return value.toLocaleString(undefined, {
    maximumFractionDigits: value >= 10 ? 1 : 2,
    minimumFractionDigits: 0,
  });
}

function SpinIdleLogitField({
  resetValue,
  value,
  onChange,
}: {
  resetValue: number;
  value: number;
  onChange: (value: number) => void;
}) {
  const input = useEditableNumberInput({
    format: formatEditableDecimal,
    formattedValue: formatEditableDecimal(value),
    normalize: (nextValue) => nextValue,
    onCommit: onChange,
    onValidInput: onChange,
    parse: parseFiniteDecimal,
  });
  const noteValue = parseFiniteDecimal(input.rawValue) ?? value;
  const idleProbability = spinIdleProbability(noteValue);
  const sideProbability = (1 - idleProbability) / 2;

  return (
    <FieldShell>
      <FieldLabel
        help="Initial logit bias toward the idle spin action. Positive values reduce random spin requests when the spin branch is first enabled."
        label="No-spin logit"
        onReset={resetHandler(value, resetValue, onChange)}
      />
      <FieldInput
        aria-label="No-spin logit"
        className="min-w-[9ch] max-w-[14ch] justify-self-start"
        inputMode="decimal"
        spellCheck={false}
        value={input.rawValue}
        onBlur={input.commitRawValue}
        onChange={(event) => input.changeRawValue(event.target.value)}
      />
      <FieldNote>
        {`logit ${formatSignedDecimal(noteValue)} -> idle ${formatPercent(idleProbability)}, left/right ${formatPercent(sideProbability)} each`}
      </FieldNote>
    </FieldShell>
  );
}

function parseFiniteDecimal(rawValue: string): number | null {
  const normalized = rawValue.trim();
  if (normalized === "" || normalized === "-" || normalized === "." || normalized === "-.") {
    return null;
  }
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function spinIdleProbability(value: number): number {
  const idleWeight = Math.exp(value);
  return idleWeight / (idleWeight + 2);
}

function formatPercent(value: number): string {
  return `${(value * 100).toLocaleString(undefined, {
    maximumFractionDigits: 1,
    minimumFractionDigits: 1,
  })}%`;
}

function formatSignedDecimal(value: number): string {
  const formatted = value.toLocaleString(undefined, {
    maximumFractionDigits: 2,
    minimumFractionDigits: 0,
  });
  return value > 0 ? `+${formatted}` : formatted;
}
