// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/action/runtime/SpinControlCard.tsx

import {
  ActionCard,
  ActionFieldset,
  ActionNote,
  ActionTwoColumn,
} from "@/entities/runConfig/ui/sections/action/ActionLayout";
import type {
  AuxiliaryActionConfig,
  UpdateAction,
  UpdatePolicy,
} from "@/entities/runConfig/ui/sections/action/branches/types";
import type { ManagedRunConfig } from "@/shared/api/contract";
import { FieldLabel, IntegerField } from "@/shared/ui/configFields";
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
  return (
    <ActionCard description="Configure the native spin macro request guard." title="Spin control">
      {action.include_spin ? null : (
        <ActionNote>
          Spin is not in the action output right now, so this runtime guard is inactive.
        </ActionNote>
      )}
      <ActionFieldset disabled={!action.include_spin}>
        <ActionTwoColumn>
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
        </ActionTwoColumn>
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
        help="Initial logit bias toward the idle spin action. Positive values reduce random spin requests when the spin branch is first enabled."
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
