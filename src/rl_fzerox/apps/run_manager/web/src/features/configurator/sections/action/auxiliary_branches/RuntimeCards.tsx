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
import type {
  AuxiliaryActionConfig,
  UpdateAction,
  UpdateTrain,
} from "@/features/configurator/sections/action/auxiliary_branches/types";
import {
  airBrakeModeDescription,
  leanModeDescription,
} from "@/features/configurator/sections/action/descriptions";
import { normalizeOddBucketCount } from "@/features/configurator/sections/action/model";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";

interface RuntimeCardsProps {
  action: AuxiliaryActionConfig;
  checkpointLocked: boolean;
  defaultAction: AuxiliaryActionConfig;
  defaultTrain: ManagedRunConfig["train"];
  metadata: ConfigMetadata;
  train: ManagedRunConfig["train"];
  updateAction: UpdateAction;
  updateTrain: UpdateTrain;
}

export function RuntimeCards({
  action,
  checkpointLocked,
  defaultAction,
  defaultTrain,
  metadata,
  train,
  updateAction,
  updateTrain,
}: RuntimeCardsProps) {
  const groundedPitchLossWeight = train.actor_regularization.grounded_pitch_neutral_loss_weight;

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
            ) : null}
          </fieldset>
          {action.pitch_mode === "continuous" ? (
            <>
              <p className="action-note">
                Continuous pitch maps the vertical stick axis directly, matching continuous steering
                semantics.
              </p>
              <div className="action-runtime-two-col">
                <div className="field-shell">
                  <FieldLabel
                    help="Add a policy-side loss that pulls continuous pitch toward neutral while the vehicle is grounded."
                    label="Grounded neutral loss"
                    onReset={() =>
                      updateTrain({
                        actor_regularization: {
                          ...defaultTrain.actor_regularization,
                        },
                      })
                    }
                  />
                  <SegmentedChoiceStrip
                    ariaLabel="Grounded pitch neutral loss"
                    options={[
                      {
                        active: groundedPitchLossWeight <= 0,
                        key: "off",
                        label: "Off",
                        onClick: () =>
                          updateTrain({
                            actor_regularization: {
                              grounded_pitch_neutral_loss_weight: 0,
                            },
                          }),
                      },
                      {
                        active: groundedPitchLossWeight > 0,
                        key: "on",
                        label: "On",
                        onClick: () =>
                          updateTrain({
                            actor_regularization: {
                              grounded_pitch_neutral_loss_weight:
                                groundedPitchLossWeight > 0 ? groundedPitchLossWeight : 0.01,
                            },
                          }),
                      },
                    ]}
                  />
                </div>
                <fieldset className="dependent-fieldset" disabled={groundedPitchLossWeight <= 0}>
                  <NumberField
                    help="Coefficient for squared continuous pitch mean while grounded. It affects the actor loss only and does not mask the action."
                    label="Loss weight"
                    resetValue={
                      defaultTrain.actor_regularization.grounded_pitch_neutral_loss_weight
                    }
                    step="0.001"
                    value={groundedPitchLossWeight > 0 ? groundedPitchLossWeight : 0}
                    onChange={(value) =>
                      updateTrain({
                        actor_regularization: {
                          grounded_pitch_neutral_loss_weight: Math.max(0, value),
                        },
                      })
                    }
                  />
                </fieldset>
              </div>
            </>
          ) : null}
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
