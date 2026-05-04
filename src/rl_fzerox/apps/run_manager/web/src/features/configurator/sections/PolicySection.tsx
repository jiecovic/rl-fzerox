import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import { BooleanField, IntegerField, SelectField } from "@/features/configurator/fields";
import { PolicyPreviewPanel } from "@/features/configurator/sections/PolicyPreviewPanel";
import { FeatureDimField } from "@/features/configurator/sections/policy/FeatureDimField";
import { LayerListField } from "@/features/configurator/sections/policy/LayerEditors";
import type {
  ConfigMetadata,
  ManagedRunConfig,
  PolicyArchitecturePreview,
} from "@/shared/api/contract";

interface PolicySectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  checkpointLocked?: boolean;
  preview: PolicyArchitecturePreview | null;
  setConfig: (config: ManagedRunConfig) => void;
}

export function PolicySection({
  config,
  defaultConfig,
  metadata,
  checkpointLocked = false,
  preview,
  setConfig,
}: PolicySectionProps) {
  const updatePolicy = (patch: Partial<ManagedRunConfig["policy"]>) => {
    setConfig({ ...config, policy: { ...config.policy, ...patch } });
  };
  const updateConvProfile = (value: ManagedRunConfig["policy"]["conv_profile"]) => {
    updatePolicy({
      conv_profile: value,
      custom_conv_layers:
        value === "custom" && config.policy.custom_conv_layers.length === 0
          ? defaultConfig.policy.custom_conv_layers
          : config.policy.custom_conv_layers,
    });
  };
  const jumpToCnnConfigurator = () => {
    document.getElementById("policy-cnn-configurator")?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };
  const convProfileOptions = metadata.conv_profiles.map(
    (option) => option.value,
  ) as ManagedRunConfig["policy"]["conv_profile"][];
  const activationOptions = metadata.activation_functions.map(
    (option) => option.value,
  ) as ManagedRunConfig["policy"]["activation"][];

  return (
    <div className="config-stack">
      <div className="form-grid three training-panel-grid">
        <ConfigPanel
          title="CNN extractor"
          onReset={
            checkpointLocked
              ? undefined
              : () =>
                  updatePolicy({
                    conv_profile: defaultConfig.policy.conv_profile,
                    custom_conv_layers: defaultConfig.policy.custom_conv_layers,
                    features_dim: defaultConfig.policy.features_dim,
                    state_net_arch: defaultConfig.policy.state_net_arch,
                    fusion_features_dim: defaultConfig.policy.fusion_features_dim,
                    layer_norm: defaultConfig.policy.layer_norm,
                  })
          }
        >
          <fieldset className="fork-lock-fieldset training-field-grid" disabled={checkpointLocked}>
            <SelectField
              help="Convolution stack used for the image branch. Choose custom to edit the conv layers directly."
              label="CNN profile"
              options={convProfileOptions}
              resetValue={defaultConfig.policy.conv_profile}
              value={config.policy.conv_profile}
              onChange={updateConvProfile}
            />
            <button
              className="secondary-button button-with-icon"
              type="button"
              onClick={jumpToCnnConfigurator}
            >
              <CnnConfigIcon />
              {config.policy.conv_profile === "custom" ? "Edit CNN" : "Go to CNN"}
            </button>
            <FeatureDimField
              help="Image feature width after CNN flatten. Auto keeps the raw flatten size."
              label="Image features"
              resetValue={defaultConfig.policy.features_dim}
              value={config.policy.features_dim}
              onChange={(value) => updatePolicy({ features_dim: value })}
            />
            <LayerListField
              help="State branch MLP layers before image/state fusion. Remove all layers to concatenate the raw state vector."
              label="State MLP"
              resetValue={defaultConfig.policy.state_net_arch}
              value={config.policy.state_net_arch}
              onChange={(value) => updatePolicy({ state_net_arch: value })}
            />
            <IntegerField
              help="Feature width after image/state fusion."
              label="Fusion features"
              min={1}
              resetValue={defaultConfig.policy.fusion_features_dim}
              value={config.policy.fusion_features_dim}
              onChange={(value) => updatePolicy({ fusion_features_dim: value })}
            />
            <BooleanField
              help="Apply layer normalization after fusion."
              label="Layer norm"
              resetValue={defaultConfig.policy.layer_norm}
              value={config.policy.layer_norm}
              onChange={(value) => updatePolicy({ layer_norm: value })}
            />
          </fieldset>
        </ConfigPanel>

        <ConfigPanel
          title="Recurrent core"
          onReset={
            checkpointLocked
              ? undefined
              : () =>
                  updatePolicy({
                    recurrent_enable_critic_lstm: defaultConfig.policy.recurrent_enable_critic_lstm,
                    recurrent_enabled: defaultConfig.policy.recurrent_enabled,
                    recurrent_hidden_size: defaultConfig.policy.recurrent_hidden_size,
                    recurrent_n_lstm_layers: defaultConfig.policy.recurrent_n_lstm_layers,
                    recurrent_shared_lstm: defaultConfig.policy.recurrent_shared_lstm,
                  })
          }
        >
          <fieldset className="fork-lock-fieldset training-field-grid" disabled={checkpointLocked}>
            <BooleanField
              help="Insert LSTM actor/critic cores between extractor and heads."
              label="Use LSTM"
              resetValue={defaultConfig.policy.recurrent_enabled}
              value={config.policy.recurrent_enabled}
              onChange={(value) => updatePolicy({ recurrent_enabled: value })}
            />
            <fieldset
              className="dependent-fieldset"
              disabled={checkpointLocked || !config.policy.recurrent_enabled}
            >
              <IntegerField
                help="Hidden width of the recurrent actor and critic."
                label="LSTM hidden"
                min={1}
                resetValue={defaultConfig.policy.recurrent_hidden_size}
                value={config.policy.recurrent_hidden_size}
                onChange={(value) => updatePolicy({ recurrent_hidden_size: value })}
              />
              <IntegerField
                help="Number of recurrent layers."
                label="LSTM layers"
                min={1}
                resetValue={defaultConfig.policy.recurrent_n_lstm_layers}
                value={config.policy.recurrent_n_lstm_layers}
                onChange={(value) => updatePolicy({ recurrent_n_lstm_layers: value })}
              />
              <BooleanField
                help="Share one LSTM between actor and critic."
                label="Shared LSTM"
                resetValue={defaultConfig.policy.recurrent_shared_lstm}
                value={config.policy.recurrent_shared_lstm}
                onChange={(value) => updatePolicy({ recurrent_shared_lstm: value })}
              />
              <BooleanField
                help="Use a critic LSTM. If off, the critic uses non-recurrent features."
                label="Critic LSTM"
                resetValue={defaultConfig.policy.recurrent_enable_critic_lstm}
                value={config.policy.recurrent_enable_critic_lstm}
                onChange={(value) => updatePolicy({ recurrent_enable_critic_lstm: value })}
              />
            </fieldset>
          </fieldset>
        </ConfigPanel>

        <ConfigPanel
          title="Heads"
          onReset={() =>
            updatePolicy({
              pi_net_arch: checkpointLocked
                ? config.policy.pi_net_arch
                : defaultConfig.policy.pi_net_arch,
              vf_net_arch: checkpointLocked
                ? config.policy.vf_net_arch
                : defaultConfig.policy.vf_net_arch,
              activation: defaultConfig.policy.activation,
            })
          }
        >
          <div className="training-field-grid">
            <SelectField
              help="Activation function used by policy/value MLP layers."
              label="Activation"
              options={activationOptions}
              resetValue={defaultConfig.policy.activation}
              value={config.policy.activation}
              onChange={(value) => updatePolicy({ activation: value })}
            />
            <fieldset className="fork-lock-fieldset" disabled={checkpointLocked}>
              <LayerListField
                help="Policy MLP layers after the recurrent core. Add or remove rows to change the head depth."
                label="Policy head"
                resetValue={defaultConfig.policy.pi_net_arch}
                value={config.policy.pi_net_arch}
                onChange={(value) => updatePolicy({ pi_net_arch: value })}
              />
              <LayerListField
                help="Value MLP layers after the recurrent core. Add or remove rows to change the head depth."
                label="Value head"
                resetValue={defaultConfig.policy.vf_net_arch}
                value={config.policy.vf_net_arch}
                onChange={(value) => updatePolicy({ vf_net_arch: value })}
              />
            </fieldset>
          </div>
        </ConfigPanel>
      </div>

      <ConfigPanel title="Architecture preview" wide>
        <PolicyPreviewPanel
          convProfile={config.policy.conv_profile}
          customConvLayers={config.policy.custom_conv_layers}
          preview={preview}
          setCustomConvLayers={(value) => updatePolicy({ custom_conv_layers: value })}
        />
      </ConfigPanel>
    </div>
  );
}

function CnnConfigIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 20 20" width="14">
      <rect height="12" rx="2" stroke="currentColor" strokeWidth="1.4" width="12" x="4" y="4" />
      <path
        d="M8 8h4M8 12h7M12 8v4"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.4"
      />
    </svg>
  );
}
