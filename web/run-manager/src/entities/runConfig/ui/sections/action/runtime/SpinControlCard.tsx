// web/run-manager/src/entities/runConfig/ui/sections/action/runtime/SpinControlCard.tsx

import {
  ActionCard,
  ActionFieldset,
  ActionNote,
  ActionTripleFields,
} from "@/entities/runConfig/ui/sections/action/ActionLayout";
import type {
  AuxiliaryActionConfig,
  UpdateAction,
  UpdatePolicy,
} from "@/entities/runConfig/ui/sections/action/branches/types";
import type { ManagedRunConfig } from "@/shared/api/contract";
import { FieldLabel, IntegerField, RangeNumberField } from "@/shared/ui/configFields";
import { formatEditableDecimal } from "@/shared/ui/configFields/format";
import {
  blurOnEnter,
  editableNumberInputProps,
  parseDecimalInput,
  useEditableNumberInput,
} from "@/shared/ui/configFields/numberInput";
import { resetHandler } from "@/shared/ui/configFields/reset";
import { FieldInput, FieldNote, FieldShell } from "@/shared/ui/Field";

interface SpinControlCardProps {
  action: AuxiliaryActionConfig;
  defaultAction: AuxiliaryActionConfig;
  defaultPolicy: ManagedRunConfig["policy"];
  policy: ManagedRunConfig["policy"];
  updateAction: UpdateAction;
  updatePolicy: UpdatePolicy;
}

export function SpinControlCard({
  action,
  defaultAction,
  defaultPolicy,
  policy,
  updateAction,
  updatePolicy,
}: SpinControlCardProps) {
  const spinEpisodeMaskProbability = action.spin_episode_mask_probability ?? 0;
  const defaultSpinEpisodeMaskProbability = defaultAction.spin_episode_mask_probability ?? 0;

  return (
    <ActionCard description="Configure the native spin macro request guard." title="Spin control">
      {action.include_spin ? null : (
        <ActionNote>
          Spin is not in the action output right now, so this runtime guard is inactive.
        </ActionNote>
      )}
      <ActionFieldset disabled={!action.include_spin}>
        <ActionTripleFields>
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
          <RangeNumberField
            help="At episode reset, sample this probability and force the spin branch to no-spin for the full episode."
            label="Episode mask probability"
            max={1}
            min={0}
            numberStep="0.01"
            rangeStep={0.01}
            resetValue={defaultSpinEpisodeMaskProbability}
            ticks={[
              { label: "0", value: 0 },
              { label: "0.1", value: 0.1 },
              { label: "0.5", value: 0.5 },
              { label: "1", value: 1 },
            ]}
            value={spinEpisodeMaskProbability}
            onChange={(value) => updateAction({ spin_episode_mask_probability: value })}
          />
        </ActionTripleFields>
      </ActionFieldset>
    </ActionCard>
  );
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
    parse: parseDecimalInput,
  });
  const noteValue = parseDecimalInput(input.rawValue) ?? value;
  const idleProbability = spinIdleProbability(noteValue);
  const sideProbability = (1 - idleProbability) / 2;

  return (
    <FieldShell>
      <FieldLabel
        help="Logit offset added to the learned idle spin action. Positive values suppress spin; negative values encourage left/right spin. The displayed probability assumes all learned spin logits are zero."
        label="No-spin logit"
        onReset={resetHandler(value, resetValue, onChange)}
      />
      <FieldInput
        aria-label="No-spin logit"
        className="min-w-[9ch] max-w-[14ch] justify-self-start"
        {...editableNumberInputProps("decimal")}
        value={input.rawValue}
        onBlur={input.commitRawValue}
        onChange={(event) => input.changeRawValue(event.target.value)}
        onKeyDown={blurOnEnter}
      />
      <FieldNote>
        {`logit ${formatSignedDecimal(noteValue)} -> idle ${formatPercent(idleProbability)}, left/right ${formatPercent(sideProbability)} each`}
      </FieldNote>
    </FieldShell>
  );
}

function spinIdleProbability(value: number): number {
  const idleWeight = Math.exp(value);
  return idleWeight / (idleWeight + 2);
}

function formatPercent(value: number): string {
  const percent = value * 100;
  if (percent > 0 && percent < 0.1) {
    return "<0.1%";
  }
  if (percent < 100 && percent > 99.9) {
    return ">99.9%";
  }
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
