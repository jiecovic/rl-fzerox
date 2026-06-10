// web/run-manager/src/entities/runConfig/ui/sections/action/ControlFamilyDisclosure.tsx

import {
  ActionCard,
  ActionFields,
  ActionInlineToggle,
  ActionStack,
  ActionTripleFields,
} from "@/entities/runConfig/ui/sections/action/ActionLayout";
import { throttleModeDescription } from "@/entities/runConfig/ui/sections/action/descriptions";
import {
  displayActionRepeat,
  effectiveControlFps,
  formatControlFps,
  normalizedActionConfig,
  normalizeOddBucketCount,
} from "@/entities/runConfig/ui/sections/action/model";
import type { ActionUpdateContext } from "@/entities/runConfig/ui/sections/action/types";
import type { ManagedRunConfig } from "@/shared/api/contract";
import { ConfigDisclosure } from "@/shared/ui/config/ConfigDisclosure";
import {
  FieldLabel,
  NumberField,
  RangeIntegerField,
  RangeNumberField,
  SegmentedChoiceStrip,
  ToggleSwitch,
} from "@/shared/ui/configFields";
import { FieldShell } from "@/shared/ui/Field";

export function ControlFamilyDisclosure({
  checkpointLocked = false,
  config,
  defaultConfig,
  metadata,
  open,
  setOpen,
  updateAction,
  updatePolicy,
  setConfig,
}: ActionUpdateContext & {
  open: boolean;
  setOpen: (open: boolean) => void;
}) {
  const action = config.action;

  return (
    <ConfigDisclosure
      onToggle={setOpen}
      open={open}
      title="Control family"
      onReset={() =>
        setConfig((currentConfig) => ({
          ...currentConfig,
          action: normalizedActionConfig({
            ...currentConfig.action,
            action_repeat: defaultConfig.action.action_repeat,
            steer_buckets: checkpointLocked
              ? currentConfig.action.steer_buckets
              : defaultConfig.action.steer_buckets,
            steering_mode: checkpointLocked
              ? currentConfig.action.steering_mode
              : defaultConfig.action.steering_mode,
            drive_mode: checkpointLocked
              ? currentConfig.action.drive_mode
              : defaultConfig.action.drive_mode,
            force_full_throttle: defaultConfig.action.force_full_throttle,
            continuous_drive_deadzone: defaultConfig.action.continuous_drive_deadzone,
            continuous_drive_full_threshold: defaultConfig.action.continuous_drive_full_threshold,
            continuous_drive_min_thrust: defaultConfig.action.continuous_drive_min_thrust,
          }),
          policy: {
            ...currentConfig.policy,
            gas_on_logit: defaultConfig.policy.gas_on_logit,
          },
        }))
      }
    >
      <ActionStack>
        <ActionCard
          description="Choose between one analog lane and a discrete bucket head."
          title="Steering"
        >
          <fieldset className="fork-lock-fieldset" disabled={checkpointLocked}>
            <FieldShell>
              <FieldLabel
                help="Choose whether steering is one analog lane or a discrete bucket head."
                label="Steering mode"
              />
              <SegmentedChoiceStrip
                ariaLabel="Steering mode"
                options={metadata.steering_modes.map((option) => ({
                  active: action.steering_mode === option.value,
                  key: option.value,
                  label: option.label,
                  onClick: () =>
                    updateAction({
                      steering_mode: option.value as ManagedRunConfig["action"]["steering_mode"],
                    }),
                }))}
              />
            </FieldShell>
            {action.steering_mode === "discrete" ? (
              <ActionFields>
                <RangeIntegerField
                  help="Odd bucket counts preserve one neutral center action while adding more left and right resolution."
                  label="Steer buckets"
                  max={31}
                  min={3}
                  rangeStep={2}
                  resetValue={defaultConfig.action.steer_buckets}
                  ticks={[
                    { value: 3, label: "3" },
                    { value: 7, label: "7" },
                    { value: 15, label: "15" },
                    { value: 31, label: "31" },
                  ]}
                  value={action.steer_buckets}
                  onChange={(value) =>
                    updateAction({ steer_buckets: normalizeOddBucketCount(value) })
                  }
                />
              </ActionFields>
            ) : null}
          </fieldset>
        </ActionCard>

        <ActionCard
          description="Choose between a continuous PWM throttle lane and a stock-style digital button press."
          title="Throttle"
        >
          <fieldset className="fork-lock-fieldset" disabled={checkpointLocked}>
            <FieldShell>
              <FieldLabel
                help="Choose between a continuous PWM throttle lane and a stock-style digital gas button. PWM is a workaround that approximates fractional throttle by pulsing the real N64 button across frames."
                label="Throttle mode"
              />
              <SegmentedChoiceStrip
                ariaLabel="Throttle mode"
                options={metadata.drive_modes.map((option) => ({
                  active: action.drive_mode === option.value,
                  key: option.value,
                  label: option.label,
                  tooltip: throttleModeDescription(
                    option.value as ManagedRunConfig["action"]["drive_mode"],
                  ),
                  onClick: () =>
                    updateAction({
                      drive_mode: option.value as ManagedRunConfig["action"]["drive_mode"],
                    }),
                }))}
              />
            </FieldShell>
          </fieldset>
          {action.drive_mode === "pwm" ? (
            <ActionTripleFields>
              <RangeNumberField
                help="Ignore small positive throttle values below this threshold."
                label="Deadzone"
                max={0.5}
                min={0}
                rangeStep={0.01}
                resetValue={defaultConfig.action.continuous_drive_deadzone}
                ticks={[
                  { value: 0, label: "0" },
                  { value: 0.1, label: "0.1" },
                  { value: 0.25, label: "0.25" },
                  { value: 0.5, label: "0.5" },
                ]}
                value={action.continuous_drive_deadzone}
                onChange={(value) => updateAction({ continuous_drive_deadzone: value })}
              />
              <RangeNumberField
                help="Clamp throttle to full once the continuous lane reaches this value."
                label="Full threshold"
                max={1}
                min={0.1}
                rangeStep={0.01}
                resetValue={defaultConfig.action.continuous_drive_full_threshold}
                ticks={[
                  { value: 0.1, label: "0.1" },
                  { value: 0.5, label: "0.5" },
                  { value: 0.85, label: "0.85" },
                  { value: 1, label: "1" },
                ]}
                value={action.continuous_drive_full_threshold}
                onChange={(value) => updateAction({ continuous_drive_full_threshold: value })}
              />
              <RangeNumberField
                help="Throttle duty used at and below the deadzone, and the lower endpoint of the ramp up to full throttle."
                label="Minimum thrust"
                max={1}
                min={0}
                rangeStep={0.01}
                resetValue={defaultConfig.action.continuous_drive_min_thrust}
                ticks={[
                  { value: 0, label: "0" },
                  { value: 0.25, label: "0.25" },
                  { value: 0.5, label: "0.5" },
                  { value: 1, label: "1" },
                ]}
                value={action.continuous_drive_min_thrust}
                onChange={(value) => updateAction({ continuous_drive_min_thrust: value })}
              />
            </ActionTripleFields>
          ) : (
            <ActionFields>
              <div className="field-with-note">
                <NumberField
                  help="Initial logit bias toward the engaged gas button when throttle uses the discrete N64-style button branch."
                  label="Gas-on logit"
                  resetValue={defaultConfig.policy.gas_on_logit}
                  step="0.1"
                  value={config.policy.gas_on_logit}
                  onChange={(value) => updatePolicy({ gas_on_logit: value })}
                />
                <div className="field-note">
                  {`sigmoid(${formatSignedDecimal(config.policy.gas_on_logit)}) ≈ ${formatPercent(gasOnProbability(config.policy.gas_on_logit))} initial engage probability`}
                </div>
              </div>
            </ActionFields>
          )}
          <FieldShell className="min-h-full">
            <FieldLabel
              help="Force discrete throttle fully engaged at runtime. This is useful when you want to remove throttle variation without changing the action head shape."
              label="Force full throttle"
            />
            <ActionInlineToggle className="grid-cols-1 justify-items-start">
              <ToggleSwitch
                checked={action.force_full_throttle}
                hideLabel
                label="Force full throttle"
                onChange={(checked) => updateAction({ force_full_throttle: checked })}
              />
            </ActionInlineToggle>
          </FieldShell>
        </ActionCard>

        <ActionCard
          description="Set how many native frames each sampled action stays active."
          title="Control cadence"
        >
          <div className="field-with-note">
            <RangeIntegerField
              help="Hold each policy decision for N native emulator frames before sampling the next action."
              label="Action repeat"
              max={8}
              min={1}
              rangeStep={1}
              resetValue={defaultConfig.action.action_repeat}
              ticks={[
                { value: 1, label: "1" },
                { value: 2, label: "2" },
                { value: 3, label: "3" },
                { value: 4, label: "4" },
                { value: 6, label: "6" },
                { value: 8, label: "8" },
              ]}
              value={action.action_repeat}
              onChange={(value) => updateAction({ action_repeat: value })}
            />
            <div className="field-note">
              {`${displayActionRepeat(action.action_repeat)} keeps one decision active for ${action.action_repeat} native frames, giving an effective control rate of ${formatControlFps(effectiveControlFps(action.action_repeat))}.`}
            </div>
          </div>
        </ActionCard>
      </ActionStack>
    </ConfigDisclosure>
  );
}

function gasOnProbability(value: number) {
  return 1 / (1 + Math.exp(-value));
}

function formatPercent(value: number) {
  return `${(value * 100).toLocaleString(undefined, {
    maximumFractionDigits: 1,
    minimumFractionDigits: 1,
  })}%`;
}

function formatSignedDecimal(value: number) {
  const formatted = value.toLocaleString(undefined, {
    maximumFractionDigits: 2,
    minimumFractionDigits: 0,
  });
  return value > 0 ? `+${formatted}` : formatted;
}
