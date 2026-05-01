import { NumberField } from "@/features/configurator/fields";
import type { ManagedRunConfig } from "@/shared/api/contract";

interface ConfigSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  setConfig: (config: ManagedRunConfig) => void;
}

export function RewardSection({ config, defaultConfig, setConfig }: ConfigSectionProps) {
  const updateReward = (patch: Partial<ManagedRunConfig["reward"]>) => {
    setConfig({ ...config, reward: { ...config.reward, ...patch } });
  };

  return (
    <div className="form-grid three">
      <NumberField
        help="Reward added when the policy requests manual boost."
        label="Boost use reward"
        resetValue={defaultConfig.reward.manual_boost_reward}
        value={config.reward.manual_boost_reward}
        onChange={(value) => updateReward({ manual_boost_reward: value })}
        step="0.01"
      />
      <NumberField
        help="Reward for entering a detected boost pad."
        label="Boost pad reward"
        resetValue={defaultConfig.reward.boost_pad_reward}
        value={config.reward.boost_pad_reward}
        onChange={(value) => updateReward({ boost_pad_reward: value })}
        step="0.5"
      />
      <NumberField
        help="Penalty when requesting pitch-up while airborne."
        label="Airborne pitch-up penalty"
        resetValue={defaultConfig.reward.airborne_pitch_up_penalty}
        value={config.reward.airborne_pitch_up_penalty}
        onChange={(value) => updateReward({ airborne_pitch_up_penalty: value })}
        step="0.01"
      />
      <NumberField
        help="Per-step penalty for requesting lean."
        label="Lean request penalty"
        resetValue={defaultConfig.reward.lean_request_penalty}
        value={config.reward.lean_request_penalty}
        onChange={(value) => updateReward({ lean_request_penalty: value })}
        step="0.001"
      />
      <NumberField
        help="Additional lean penalty below the configured speed cutoff."
        label="Low-speed lean penalty"
        resetValue={defaultConfig.reward.lean_low_speed_penalty}
        value={config.reward.lean_low_speed_penalty}
        onChange={(value) => updateReward({ lean_low_speed_penalty: value })}
        step="0.001"
      />
      <NumberField
        help="Speed below which the low-speed lean penalty applies."
        label="Low-speed lean cutoff"
        resetValue={defaultConfig.reward.lean_low_speed_penalty_max_speed_kph}
        value={config.reward.lean_low_speed_penalty_max_speed_kph}
        onChange={(value) => updateReward({ lean_low_speed_penalty_max_speed_kph: value })}
      />
    </div>
  );
}
