import type { ManagedRunConfig } from "@/contract";
import { NumberField, RangeNumberField, SelectField } from "@/features/configurator/fields";

interface ConfigSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  setConfig: (config: ManagedRunConfig) => void;
}

export function ModelSection({ config, defaultConfig, setConfig }: ConfigSectionProps) {
  const updateObservation = (patch: Partial<ManagedRunConfig["observation"]>) => {
    setConfig({ ...config, observation: { ...config.observation, ...patch } });
  };
  const updatePolicy = (patch: Partial<ManagedRunConfig["policy"]>) => {
    setConfig({ ...config, policy: { ...config.policy, ...patch } });
  };
  const updateTrain = (patch: Partial<ManagedRunConfig["train"]>) => {
    setConfig({ ...config, train: { ...config.train, ...patch } });
  };

  return (
    <div className="form-grid three">
      <NumberField
        help="Number of recent image observations exposed to the policy."
        label="Frame stack"
        resetValue={defaultConfig.observation.frame_stack}
        value={config.observation.frame_stack}
        onChange={(value) => updateObservation({ frame_stack: value })}
      />
      <SelectField
        help="Image channel encoding used by the observation builder."
        label="Stack mode"
        value={config.observation.stack_mode}
        options={["rgb", "gray", "luma_chroma"]}
        resetValue={defaultConfig.observation.stack_mode}
        onChange={(value) => updateObservation({ stack_mode: value })}
      />
      <SelectField
        help="Scalar progress signal exposed in the state vector."
        label="Progress scalar"
        value={config.observation.progress_source}
        options={["lap_progress", "segment_progress", "none"]}
        resetValue={defaultConfig.observation.progress_source}
        onChange={(value) => updateObservation({ progress_source: value })}
      />
      <SelectField
        help="CNN channel profile used by the image extractor."
        label="CNN profile"
        value={config.policy.conv_profile}
        options={["nature", "nature_32_64_128", "nature_wide"]}
        resetValue={defaultConfig.policy.conv_profile}
        onChange={(value) => updatePolicy({ conv_profile: value })}
      />
      <NumberField
        help="Hidden size of the recurrent actor and critic LSTMs."
        label="LSTM hidden"
        resetValue={defaultConfig.policy.recurrent_hidden_size}
        value={config.policy.recurrent_hidden_size}
        onChange={(value) => updatePolicy({ recurrent_hidden_size: value })}
      />
      <NumberField
        help="Feature width after image/state fusion before the recurrent layer."
        label="Fusion features"
        resetValue={defaultConfig.policy.fusion_features_dim}
        value={config.policy.fusion_features_dim}
        onChange={(value) => updatePolicy({ fusion_features_dim: value })}
      />
      <RangeNumberField
        help="Per-episode probability of zeroing course context inputs."
        label="Course context dropout"
        max={1}
        min={0}
        numberStep="0.01"
        rangeStep={0.05}
        ticks={[
          { value: 0, label: "0" },
          { value: 0.5, label: ".5" },
          { value: 1, label: "1" },
        ]}
        resetValue={defaultConfig.train.course_context_dropout_prob}
        value={config.train.course_context_dropout_prob}
        onChange={(value) => updateTrain({ course_context_dropout_prob: value })}
      />
    </div>
  );
}
