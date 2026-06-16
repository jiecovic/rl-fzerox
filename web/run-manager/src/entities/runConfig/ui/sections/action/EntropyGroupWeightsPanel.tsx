// web/run-manager/src/entities/runConfig/ui/sections/action/EntropyGroupWeightsPanel.tsx

import type { ConfigSectionPatch } from "@/entities/runConfig/model/state";
import type { ManagedRunConfig } from "@/shared/api/contract";
import { ConfigPanel } from "@/shared/ui/config/ConfigPanel";
import { BooleanField, NumberField } from "@/shared/ui/configFields";
import { HelpTooltipButton } from "@/shared/ui/HelpTooltipButton";

interface EntropyGroup {
  key: string;
  label: string;
}

const defaultEntropyCoefficient = 0.01;

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

  const coefficients = train.entropy_coefficients;
  const usesDefaultCoefficients = groups.every(
    (group) => coefficientFor(train, group.key) === coefficientFor(defaultTrain, group.key),
  );

  function updateGroupCoefficient(key: string, value: number) {
    updateTrain({
      entropy_coefficients: {
        ...coefficients,
        [key]: Math.max(0, value),
      },
    });
  }

  return (
    <ConfigPanel
      id="action-entropy-coefficients"
      title="Entropy coefficients"
      wide
      onReset={() => updateTrain({ entropy_coefficients: defaultTrain.entropy_coefficients })}
    >
      <div className="grid gap-3">
        <div className="flex flex-wrap items-center gap-2 text-sm text-app-text-muted">
          <span>Per-action PPO entropy coefficients.</span>
          <HelpTooltipButton
            label="Entropy coefficients"
            text="rl-fzerox stores effective per-action coefficients, then adapts them to sb3x group entropy weights at launch."
          />
          <strong className="text-app-text">
            {usesDefaultCoefficients ? "default coefficients" : "custom coefficients"}
          </strong>
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {groups.map((group) => {
            const coefficient = coefficientFor(train, group.key);
            const defaultCoefficient = coefficientFor(defaultTrain, group.key);
            const enabled = coefficient > 0;
            return (
              <section
                aria-label={`${group.label} entropy coefficient`}
                className="grid grid-cols-[minmax(0,1fr)_minmax(9ch,14ch)] items-end gap-3 border border-app-border bg-app-surface p-3"
                key={group.key}
              >
                <BooleanField
                  help={`Include ${group.label} entropy in the PPO entropy bonus.`}
                  label={group.label}
                  value={enabled}
                  onChange={(nextEnabled) =>
                    updateGroupCoefficient(group.key, nextEnabled ? defaultCoefficient : 0)
                  }
                />
                <fieldset className="dependent-fieldset" disabled={!enabled}>
                  <NumberField
                    help={`Effective PPO entropy coefficient for ${group.label}.`}
                    label="Coefficient"
                    resetValue={defaultCoefficient}
                    step="0.001"
                    value={enabled ? coefficient : 0}
                    onChange={(value) => updateGroupCoefficient(group.key, value)}
                  />
                </fieldset>
              </section>
            );
          })}
        </div>
      </div>
    </ConfigPanel>
  );
}

export function coefficientFor(train: ManagedRunConfig["train"], key: string): number {
  return train.entropy_coefficients[key] ?? defaultEntropyCoefficient;
}

export function actionEntropyGroups(action: ManagedRunConfig["action"]): EntropyGroup[] {
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
