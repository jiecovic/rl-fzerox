// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/action/runtime/BoostGuardsCard.tsx

import {
  ActionCard,
  ActionFieldset,
  ActionNote,
  ActionTwoColumn,
} from "@/entities/runConfig/ui/sections/action/ActionLayout";
import type {
  AuxiliaryActionConfig,
  UpdateAction,
} from "@/entities/runConfig/ui/sections/action/branches/types";
import {
  FieldLabel,
  IntegerField,
  OptionalNumberField,
  RangeNumberField,
  SegmentedChoiceStrip,
} from "@/shared/ui/configFields";
import { FieldShell } from "@/shared/ui/Field";

interface BoostGuardsCardProps {
  action: AuxiliaryActionConfig;
  defaultAction: AuxiliaryActionConfig;
  updateAction: UpdateAction;
}

export function BoostGuardsCard({ action, defaultAction, updateAction }: BoostGuardsCardProps) {
  return (
    <ActionCard
      description="Only allow manual boost when these runtime conditions are satisfied."
      title="Boost guards"
    >
      {action.include_boost ? null : (
        <ActionNote>
          Boost is not in the action output right now, so these runtime rules are inactive.
        </ActionNote>
      )}
      <ActionFieldset disabled={!action.include_boost}>
        <ActionTwoColumn>
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
        </ActionTwoColumn>
        <ActionTwoColumn>
          <FieldShell>
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
          </FieldShell>
          <FieldShell>
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
          </FieldShell>
        </ActionTwoColumn>
        <ActionTwoColumn>
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
        </ActionTwoColumn>
        <ActionTwoColumn>
          <IntegerField
            help="After a manual boost request, keep the boost branch masked for this many native frames. Useful as the spam guard when the active-boost mask is off."
            label="Request cooldown frames"
            resetValue={defaultAction.boost_request_lockout_frames}
            value={action.boost_request_lockout_frames}
            onChange={(value) => updateAction({ boost_request_lockout_frames: value })}
          />
        </ActionTwoColumn>
      </ActionFieldset>
    </ActionCard>
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
