// web/run-manager/src/entities/runConfig/ui/sections/action/descriptions.ts
import type { ManagedRunConfig } from "@/shared/api/contract";

export function leanModeDescription(mode: ManagedRunConfig["action"]["lean_mode"]) {
  if (mode === "minimum_hold") {
    return "Fresh lean taps are held for the full tap-guard window so short releases cannot turn into accidental side attacks.";
  }
  if (mode === "release_cooldown") {
    return "After a lean release or side switch, lean is forced back to neutral for the tap-guard window.";
  }
  if (mode === "timer_assist") {
    return "Lean is passed through, but the native runtime patches the game’s Z/R double-tap timers while lean is held.";
  }
  return "Lean is passed through directly with no hold, cooldown, or timer assistance. Side attacks can happen naturally.";
}

export function throttleModeDescription(mode: ManagedRunConfig["action"]["drive_mode"]) {
  if (mode === "pwm") {
    return "Continuous PWM throttle is a workaround. The policy emits one continuous throttle value, and runtime approximates it by pulsing the real N64 gas button across frames.";
  }
  return "Discrete button throttle matches the stock N64 control surface: one digital gas button with idle or engaged states.";
}

export function airBrakeModeDescription(mode: ManagedRunConfig["action"]["air_brake_mode"]) {
  if (mode === "pwm") {
    return "Continuous PWM air brake is a workaround. The policy emits one continuous brake value, and runtime approximates it by pulsing the real N64 air-brake button across frames.";
  }
  return "Discrete button air brake matches the stock N64 control surface: one digital air-brake button with idle or engaged states.";
}
