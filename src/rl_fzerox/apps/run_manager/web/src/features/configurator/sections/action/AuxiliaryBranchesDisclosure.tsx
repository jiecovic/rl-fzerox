import { ConfigDisclosure } from "@/features/configurator/ConfigDisclosure";
import {
  FieldLabel,
  IntegerField,
  OptionalNumberField,
  RangeIntegerField,
  RangeNumberField,
  SegmentedChoiceStrip,
} from "@/features/configurator/fields";
import { ActionToggleRow } from "@/features/configurator/sections/action/ActionToggleRow";
import {
  airBrakeModeDescription,
  leanModeDescription,
} from "@/features/configurator/sections/action/descriptions";
import { normalizeOddBucketCount } from "@/features/configurator/sections/action/model";
import type { ActionUpdateContext } from "@/features/configurator/sections/action/types";
import type { ManagedRunConfig } from "@/shared/api/contract";
import { HelpTooltipButton } from "@/shared/ui/HelpTooltipButton";

export function AuxiliaryBranchesDisclosure({
  checkpointLocked = false,
  config,
  defaultConfig,
  metadata,
  open,
  setOpen,
  updateAction,
}: Omit<ActionUpdateContext, "updatePolicy" | "setConfig"> & {
  open: boolean;
  setOpen: (open: boolean) => void;
}) {
  const action = config.action;

  return (
    <ConfigDisclosure
      onToggle={setOpen}
      open={open}
      title="Auxiliary branches"
      onReset={() =>
        updateAction({
          include_air_brake: checkpointLocked
            ? config.action.include_air_brake
            : defaultConfig.action.include_air_brake,
          air_brake_mode: checkpointLocked
            ? config.action.air_brake_mode
            : defaultConfig.action.air_brake_mode,
          enable_air_brake: defaultConfig.action.enable_air_brake,
          mask_air_brake_on_ground: defaultConfig.action.mask_air_brake_on_ground,
          continuous_air_brake_deadzone: defaultConfig.action.continuous_air_brake_deadzone,
          continuous_air_brake_full_threshold:
            defaultConfig.action.continuous_air_brake_full_threshold,
          continuous_air_brake_min_duty: defaultConfig.action.continuous_air_brake_min_duty,
          include_boost: checkpointLocked
            ? config.action.include_boost
            : defaultConfig.action.include_boost,
          enable_boost: defaultConfig.action.enable_boost,
          boost_unmask_max_speed_kph: defaultConfig.action.boost_unmask_max_speed_kph,
          boost_min_energy_fraction: defaultConfig.action.boost_min_energy_fraction,
          include_lean: checkpointLocked
            ? config.action.include_lean
            : defaultConfig.action.include_lean,
          enable_lean: defaultConfig.action.enable_lean,
          lean_output_mode: checkpointLocked
            ? config.action.lean_output_mode
            : defaultConfig.action.lean_output_mode,
          lean_mode: defaultConfig.action.lean_mode,
          lean_unmask_min_speed_kph: defaultConfig.action.lean_unmask_min_speed_kph,
          lean_initial_lockout_frames: defaultConfig.action.lean_initial_lockout_frames,
          include_pitch: checkpointLocked
            ? config.action.include_pitch
            : defaultConfig.action.include_pitch,
          enable_pitch: defaultConfig.action.enable_pitch,
          pitch_mode: checkpointLocked ? config.action.pitch_mode : defaultConfig.action.pitch_mode,
          pitch_buckets: checkpointLocked
            ? config.action.pitch_buckets
            : defaultConfig.action.pitch_buckets,
        })
      }
    >
      <div className="action-aux-stack">
        <div className="action-toggle-header">
          <span>Branch</span>
          <span className="action-toggle-heading">
            <span>Output</span>
            <HelpTooltipButton
              label="Output"
              text="Keep this branch in the final action output shape."
            />
          </span>
          <span className="action-toggle-heading">
            <span>Enabled</span>
            <HelpTooltipButton
              label="Enabled"
              position="left"
              text="Mask or unmask this branch at runtime without changing the output shape."
            />
          </span>
        </div>
        <div className="action-toggle-grid">
          <ActionToggleRow
            description={
              action.air_brake_mode === "pwm"
                ? "Expose one continuous air-brake lane that runtime maps back onto the N64 button with accumulator PWM."
                : "Expose left / right air brake as a digital branch."
            }
            enabled={action.enable_air_brake}
            enabledLabel="Air brake enabled"
            outputDisabledReason={
              checkpointLocked ? "Forked checkpoints keep the original action outputs." : undefined
            }
            output={action.include_air_brake}
            outputLabel="Air brake in output"
            label="Air brake"
            onEnabledChange={(checked) => updateAction({ enable_air_brake: checked })}
            onOutputChange={(checked) =>
              updateAction({
                enable_air_brake: checked,
                include_air_brake: checked,
              })
            }
          />
          <ActionToggleRow
            description="Expose manual boost as a digital branch."
            enabled={action.enable_boost}
            enabledLabel="Boost enabled"
            outputDisabledReason={
              checkpointLocked ? "Forked checkpoints keep the original action outputs." : undefined
            }
            output={action.include_boost}
            outputLabel="Boost in output"
            label="Boost"
            onEnabledChange={(checked) => updateAction({ enable_boost: checked })}
            onOutputChange={(checked) =>
              updateAction({
                enable_boost: checked,
                include_boost: checked,
              })
            }
          />
          <ActionToggleRow
            description={
              action.lean_output_mode === "independent_buttons"
                ? "Expose separate left and right lean buttons that can co-activate."
                : "Expose lean left / neutral / right as a 3-logit branch."
            }
            enabled={action.enable_lean}
            enabledLabel="Lean enabled"
            outputDisabledReason={
              checkpointLocked ? "Forked checkpoints keep the original action outputs." : undefined
            }
            output={action.include_lean}
            outputLabel="Lean in output"
            label="Lean"
            onEnabledChange={(checked) => updateAction({ enable_lean: checked })}
            onOutputChange={(checked) =>
              updateAction({
                enable_lean: checked,
                include_lean: checked,
              })
            }
          />
          <ActionToggleRow
            description={
              action.pitch_mode === "continuous"
                ? "Expose one continuous airborne pitch lane."
                : `Expose airborne pitch as a ${action.pitch_buckets}-logit branch.`
            }
            enabled={action.enable_pitch}
            enabledDisabledReason={
              action.pitch_mode === "continuous"
                ? "Continuous pitch cannot be runtime-masked."
                : undefined
            }
            enabledLabel="Pitch enabled"
            outputDisabledReason={
              checkpointLocked ? "Forked checkpoints keep the original action outputs." : undefined
            }
            output={action.include_pitch}
            outputLabel="Pitch in output"
            label="Pitch"
            onEnabledChange={(checked) => updateAction({ enable_pitch: checked })}
            onOutputChange={(checked) =>
              updateAction({
                enable_pitch: checked,
                include_pitch: checked,
              })
            }
          />
        </div>

        <div className="action-behavior-grid">
          <section className="action-runtime-card">
            <div className="action-runtime-header config-disclosure-copy">
              <strong>Air brake</strong>
              <small>Choose the output family, optional PWM shaping, and grounded-use guard.</small>
            </div>
            {action.include_air_brake ? null : (
              <p className="action-note">
                Air brake is not in the action output right now, so these runtime rules are
                inactive.
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
                          air_brake_mode:
                            option.value as ManagedRunConfig["action"]["air_brake_mode"],
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
                      resetValue={defaultConfig.action.continuous_air_brake_deadzone}
                      ticks={[
                        { value: 0, label: "0" },
                        { value: 0.1, label: "0.1" },
                        { value: 0.25, label: "0.25" },
                        { value: 0.5, label: "0.5" },
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
                      resetValue={defaultConfig.action.continuous_air_brake_full_threshold}
                      ticks={[
                        { value: 0.1, label: "0.1" },
                        { value: 0.5, label: "0.5" },
                        { value: 0.85, label: "0.85" },
                        { value: 1, label: "1" },
                      ]}
                      value={action.continuous_air_brake_full_threshold}
                      onChange={(value) =>
                        updateAction({ continuous_air_brake_full_threshold: value })
                      }
                    />
                    <RangeNumberField
                      help="Minimum air-brake duty once the lane is above the deadzone. Zero keeps the first engaged pulses sparse; higher values make early braking more assertive."
                      label="Minimum duty"
                      max={1}
                      min={0}
                      rangeStep={0.01}
                      resetValue={defaultConfig.action.continuous_air_brake_min_duty}
                      ticks={[
                        { value: 0, label: "0" },
                        { value: 0.25, label: "0.25" },
                        { value: 0.5, label: "0.5" },
                        { value: 1, label: "1" },
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
                  resetValue={defaultConfig.action.boost_unmask_max_speed_kph}
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
                  resetValue={defaultConfig.action.boost_min_energy_fraction * 100}
                  ticks={[
                    { value: 0, label: "0" },
                    { value: 10, label: "10" },
                    { value: 25, label: "25" },
                    { value: 50, label: "50" },
                    { value: 100, label: "100" },
                  ]}
                  value={action.boost_min_energy_fraction * 100}
                  onChange={(value) => updateAction({ boost_min_energy_fraction: value / 100 })}
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
                    help="Choose whether lean is one 3-way axis or two independent left and right buttons that can be pressed together."
                    label="Lean output"
                  />
                  <SegmentedChoiceStrip
                    ariaLabel="Lean output"
                    options={metadata.lean_output_modes.map((option) => ({
                      active: action.lean_output_mode === option.value,
                      key: option.value,
                      label: option.label,
                      onClick: () =>
                        updateAction({
                          lean_output_mode:
                            option.value as ManagedRunConfig["action"]["lean_output_mode"],
                        }),
                    }))}
                  />
                </div>
              </fieldset>
              {action.lean_output_mode === "three_way" ? (
                <>
                  <div className="field-shell">
                    <FieldLabel
                      help="Choose how the 3-way lean axis is post-processed before it reaches the emulator."
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
                  Independent buttons expose separate left and right lean outputs. They can
                  co-activate and always bypass lean hold or cooldown assistance.
                </p>
              )}
              <div className="action-runtime-two-col">
                <OptionalNumberField
                  defaultValue={700}
                  help="Optionally block lean below this vehicle speed."
                  label="Only allow above speed"
                  max={1500}
                  min={0}
                  resetValue={defaultConfig.action.lean_unmask_min_speed_kph}
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
              <small>
                Choose whether airborne pitch is one analog lane or a discrete bucket head.
              </small>
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
                    help="Choose whether airborne pitch is a continuous analog lane or a discrete bucket head."
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
                    resetValue={defaultConfig.action.pitch_buckets}
                    ticks={[
                      { value: 3, label: "3" },
                      { value: 5, label: "5" },
                      { value: 9, label: "9" },
                      { value: 15, label: "15" },
                      { value: 31, label: "31" },
                    ]}
                    value={action.pitch_buckets}
                    onChange={(value) =>
                      updateAction({ pitch_buckets: normalizeOddBucketCount(value) })
                    }
                  />
                ) : (
                  <p className="action-note">
                    Continuous pitch maps the airborne pitch axis directly, so runtime masking stays
                    unavailable.
                  </p>
                )}
              </fieldset>
            </fieldset>
          </section>
        </div>
      </div>
    </ConfigDisclosure>
  );
}
