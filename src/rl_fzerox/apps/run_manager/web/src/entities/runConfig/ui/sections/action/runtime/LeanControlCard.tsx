// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/action/runtime/LeanControlCard.tsx

import {
  ActionCard,
  ActionFieldset,
  ActionNote,
  ActionTripleFields,
} from "@/entities/runConfig/ui/sections/action/ActionLayout";
import type {
  AuxiliaryActionConfig,
  UpdateAction,
} from "@/entities/runConfig/ui/sections/action/branches/types";
import { leanModeDescription } from "@/entities/runConfig/ui/sections/action/descriptions";
import type { ConfigMetadata, ManagedRunConfig } from "@/shared/api/contract";
import {
  FieldLabel,
  IntegerField,
  OptionalNumberField,
  RangeNumberField,
  SegmentedChoiceStrip,
} from "@/shared/ui/configFields";
import { FieldShell } from "@/shared/ui/Field";

interface LeanControlCardProps {
  action: AuxiliaryActionConfig;
  checkpointLocked: boolean;
  defaultAction: AuxiliaryActionConfig;
  metadata: ConfigMetadata;
  updateAction: UpdateAction;
}

export function LeanControlCard({
  action,
  checkpointLocked,
  defaultAction,
  metadata,
  updateAction,
}: LeanControlCardProps) {
  const leanEpisodeMaskProbability = action.lean_episode_mask_probability ?? 0;
  const defaultLeanEpisodeMaskProbability = defaultAction.lean_episode_mask_probability ?? 0;

  return (
    <ActionCard
      description="Define the lean output shape, optional guards, and any post-processing."
      title="Lean control"
    >
      {action.include_lean ? null : (
        <ActionNote>
          Lean is not in the action output right now, so these runtime rules are inactive.
        </ActionNote>
      )}
      <ActionFieldset disabled={!action.include_lean}>
        <fieldset className="fork-lock-fieldset" disabled={checkpointLocked}>
          <FieldShell>
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
          </FieldShell>
        </fieldset>
        {action.lean_output_mode === "three_way" ||
        action.lean_output_mode === "four_way_categorical" ? (
          <>
            <FieldShell>
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
            </FieldShell>
            <ActionNote>{leanModeDescription(action.lean_mode)}</ActionNote>
          </>
        ) : (
          <ActionNote>
            Independent buttons expose separate left and right lean outputs. They can co-activate
            and always bypass lean hold or cooldown assistance.
          </ActionNote>
        )}
        <ActionTripleFields>
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
          <RangeNumberField
            help="At episode reset, sample this probability and force lean plus lean-backed spin neutral for the full episode."
            label="Episode mask probability"
            max={1}
            min={0}
            numberStep="0.01"
            rangeStep={0.01}
            resetValue={defaultLeanEpisodeMaskProbability}
            ticks={[
              { label: "0", value: 0 },
              { label: "0.1", value: 0.1 },
              { label: "0.5", value: 0.5 },
              { label: "1", value: 1 },
            ]}
            value={leanEpisodeMaskProbability}
            onChange={(value) => updateAction({ lean_episode_mask_probability: value })}
          />
        </ActionTripleFields>
      </ActionFieldset>
    </ActionCard>
  );
}
