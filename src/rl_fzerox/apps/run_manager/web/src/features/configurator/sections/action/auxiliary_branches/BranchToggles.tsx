// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/action/auxiliary_branches/BranchToggles.tsx
import { ActionToggleRow } from "@/features/configurator/sections/action/ActionToggleRow";
import type {
  AuxiliaryActionConfig,
  UpdateAction,
} from "@/features/configurator/sections/action/auxiliary_branches/types";
import { HelpTooltipButton } from "@/shared/ui/HelpTooltipButton";

interface BranchTogglesProps {
  action: AuxiliaryActionConfig;
  checkpointLocked: boolean;
  updateAction: UpdateAction;
}

export function BranchToggles({ action, checkpointLocked, updateAction }: BranchTogglesProps) {
  return (
    <>
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
    </>
  );
}
