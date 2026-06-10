// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/vehicle/engineSetting/types.ts
export type EngineMode = "fixed" | "random_range";
export type RangeHandle = "min" | "max";

export interface SliderTick {
  label: string;
  value: number;
}
