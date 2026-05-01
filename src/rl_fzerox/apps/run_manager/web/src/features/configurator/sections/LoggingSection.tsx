import { RangeNumberField } from "@/features/configurator/fields";
import type { ManagedRunConfig } from "@/shared/api/contract";

interface LoggingSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  setConfig: (config: ManagedRunConfig) => void;
}

export function LoggingSection({ config, defaultConfig, setConfig }: LoggingSectionProps) {
  const updateTrain = (patch: Partial<ManagedRunConfig["train"]>) => {
    setConfig({ ...config, train: { ...config.train, ...patch } });
  };

  return (
    <div className="form-grid three">
      <RangeNumberField
        help="Window size used by SB3-style rolling training statistics."
        label="Stats window"
        max={1000}
        min={10}
        rangeStep={10}
        ticks={[
          { value: 10, label: "10" },
          { value: 500, label: "500" },
          { value: 1000, label: "1k" },
        ]}
        resetValue={defaultConfig.train.stats_window_size}
        value={config.train.stats_window_size}
        onChange={(value) => updateTrain({ stats_window_size: value })}
      />
    </div>
  );
}
