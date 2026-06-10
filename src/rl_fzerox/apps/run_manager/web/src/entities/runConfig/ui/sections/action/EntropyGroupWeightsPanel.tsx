// src/rl_fzerox/apps/run_manager/web/src/entities/runConfig/ui/sections/action/EntropyGroupWeightsPanel.tsx

import type { ConfigSectionPatch } from "@/entities/runConfig/model/state";
import type { ManagedRunConfig } from "@/shared/api/contract";
import { ConfigPanel } from "@/shared/ui/config/ConfigPanel";
import { BooleanField, NumberField } from "@/shared/ui/configFields";
import { HelpTooltipButton } from "@/shared/ui/HelpTooltipButton";

interface EntropyGroup {
  key: string;
  label: string;
}

interface EntropyGroupWeightsPanelProps {
  action: ManagedRunConfig["action"];
  defaultTrain: ManagedRunConfig["train"];
  train: ManagedRunConfig["train"];
  updateTrain: (patch: ConfigSectionPatch<"train">) => void;
}

export function EntropyGroupWeightsPanel({
  action,
  defaultTrain,
  train,
  updateTrain,
}: EntropyGroupWeightsPanelProps) {
  const groups = actionEntropyGroups(action);
  if (groups.length === 0) {
    return null;
  }

  const explicitWeights = train.entropy_group_weights;
  const usesDefaultWeights = Object.keys(explicitWeights).length === 0;
  const weightFor = (key: string) => explicitWeights[key] ?? 1;

  function updateGroupWeight(key: string, value: number) {
    updateTrain({
      entropy_group_weights: normalizedEntropyWeights(groups, {
        ...currentEntropyWeights(groups, train),
        [key]: Math.max(0, value),
      }),
    });
  }

  return (
    <ConfigPanel
      title="Entropy groups"
      wide
      onReset={() => updateTrain({ entropy_group_weights: defaultTrain.entropy_group_weights })}
    >
      <div className="grid gap-3">
        <div className="flex flex-wrap items-center gap-2 text-sm text-app-text-muted">
          <span>Per-action multipliers for the global PPO entropy coefficient.</span>
          <HelpTooltipButton
            label="Entropy groups"
            text="sb3x stays action-oblivious here. rl-fzerox names each action branch and passes generic entropy weights into PPO."
          />
          <strong className="text-app-text">
            {usesDefaultWeights ? "standard PPO entropy" : "custom weights"}
          </strong>
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {groups.map((group) => {
            const weight = weightFor(group.key);
            const enabled = weight > 0;
            const effectiveEntCoef = enabled ? train.ent_coef * weight : 0;
            return (
              <section
                aria-label={`${group.label} entropy group`}
                className="grid grid-cols-[minmax(0,1fr)_minmax(9ch,14ch)] items-end gap-3 border border-app-border bg-app-surface p-3"
                key={group.key}
              >
                <BooleanField
                  help={`Include ${group.label} entropy in the PPO entropy bonus.`}
                  label={group.label}
                  value={enabled}
                  onChange={(nextEnabled) => updateGroupWeight(group.key, nextEnabled ? 1 : 0)}
                />
                <fieldset className="dependent-fieldset" disabled={!enabled}>
                  <NumberField
                    help={`Multiplier for ${group.label} entropy before the global entropy coefficient is applied.`}
                    label="Multiplier"
                    step="0.1"
                    value={enabled ? weight : 0}
                    onChange={(value) => updateGroupWeight(group.key, value)}
                  />
                </fieldset>
                <div className="col-span-2 flex items-baseline justify-between gap-3 border-app-border border-t pt-2 text-xs text-app-muted">
                  <span>Effective coeff</span>
                  <strong className="font-mono text-app-text">
                    {formatEntropyCoefficient(effectiveEntCoef)}
                  </strong>
                </div>
              </section>
            );
          })}
        </div>
      </div>
    </ConfigPanel>
  );
}

function currentEntropyWeights(
  groups: EntropyGroup[],
  train: ManagedRunConfig["train"],
): Record<string, number> {
  return Object.fromEntries(
    groups.map((group) => [group.key, train.entropy_group_weights[group.key] ?? 1]),
  );
}

function normalizedEntropyWeights(
  groups: EntropyGroup[],
  weights: Record<string, number>,
): Record<string, number> {
  const normalized = Object.fromEntries(
    groups.map((group) => [group.key, Math.max(0, weights[group.key] ?? 0)]),
  );
  return Object.values(normalized).every((value) => value === 1) ? {} : normalized;
}

function formatEntropyCoefficient(value: number): string {
  if (value === 0) {
    return "0";
  }
  return value.toExponential(2);
}

function actionEntropyGroups(action: ManagedRunConfig["action"]): EntropyGroup[] {
  const groups: EntropyGroup[] = [];
  if (action.steering_mode === "continuous") {
    groups.push({ key: "steer", label: "Steer" });
  }
  if (action.drive_mode === "pwm") {
    groups.push({ key: "drive", label: "Throttle" });
  }
  if (action.include_air_brake && action.air_brake_mode === "pwm") {
    groups.push({ key: "air_brake", label: "Air brake" });
  }
  if (action.include_pitch && action.pitch_mode === "continuous") {
    groups.push({ key: "pitch", label: "Pitch" });
  }
  if (action.steering_mode === "discrete") {
    groups.push({ key: "steer", label: "Steer" });
  }
  if (action.drive_mode === "on_off") {
    groups.push({ key: "gas", label: "Gas" });
  }
  if (action.include_air_brake && action.air_brake_mode === "on_off") {
    groups.push({ key: "air_brake", label: "Air brake" });
  }
  if (action.include_boost) {
    groups.push({ key: "boost", label: "Boost" });
  }
  if (action.include_lean) {
    if (action.lean_output_mode === "independent_buttons") {
      groups.push({ key: "lean_left", label: "Lean left" });
      groups.push({ key: "lean_right", label: "Lean right" });
    } else {
      groups.push({ key: "lean", label: "Lean" });
      if (action.include_spin) {
        groups.push({ key: "spin", label: "Spin" });
      }
    }
  }
  if (action.include_pitch && action.pitch_mode === "discrete") {
    groups.push({ key: "pitch", label: "Pitch" });
  }
  return groups;
}
