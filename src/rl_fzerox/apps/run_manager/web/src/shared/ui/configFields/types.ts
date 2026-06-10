// src/rl_fzerox/apps/run_manager/web/src/shared/ui/configFields/types.ts
export interface FieldLabelProps {
  help: string;
  label: string;
  onReset?: () => void;
}

export interface SliderTick {
  label: string;
  value: number;
}
