// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/action/runtime/AirBrakeCard.tsx
import { FieldLabel, RangeNumberField, SegmentedChoiceStrip } from "@/features/configurator/fields";
import {
  ActionCard,
  ActionFieldset,
  ActionNote,
  ActionTripleFields,
} from "@/features/configurator/sections/action/ActionLayout";
import type {
  AuxiliaryActionConfig,
  UpdateAction,
} from "@/features/configurator/sections/action/branches/types";
import { airBrakeModeDescription } from "@/features/configurator/sections/action/descriptions";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";
import { FieldShell } from "@/shared/ui/Field";

interface AirBrakeCardProps {
  action: AuxiliaryActionConfig;
  checkpointLocked: boolean;
  defaultAction: AuxiliaryActionConfig;
  metadata: ConfigMetadata;
  updateAction: UpdateAction;
}

export function AirBrakeCard({
  action,
  checkpointLocked,
  defaultAction,
  metadata,
  updateAction,
}: AirBrakeCardProps) {
  return (
    <ActionCard
      description="Choose the output family, optional PWM shaping, and grounded-use guard."
      title="Air brake"
    >
      {action.include_air_brake ? null : (
        <ActionNote>
          Air brake is not in the action output right now, so these runtime rules are inactive.
        </ActionNote>
      )}
      <ActionFieldset disabled={!action.include_air_brake}>
        <fieldset className="fork-lock-fieldset" disabled={checkpointLocked}>
          <FieldShell>
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
          </FieldShell>
          {action.air_brake_mode === "pwm" ? (
            <ActionTripleFields>
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
            </ActionTripleFields>
          ) : null}
        </fieldset>
        <FieldShell>
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
        </FieldShell>
      </ActionFieldset>
    </ActionCard>
  );
}
