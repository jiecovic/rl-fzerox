// web/run-manager/src/entities/runConfig/ui/sections/vehicle/engineSetting/types.ts
export type EngineMode = "fixed" | "random_range" | "adaptive_tuner";
export type RangeHandle = "min" | "max";

export interface SliderTick {
  label: string;
  value: number;
}
