// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/PolicySection.tsx
import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import {
  type ConfigSectionPatch,
  type ConfigSetter,
  patchConfigSection,
} from "@/features/configurator/configurator/state";
import { BooleanField, IntegerField, SelectField } from "@/features/configurator/fields";
import { PolicyPreviewPanel } from "@/features/configurator/sections/PolicyPreviewPanel";
import { FeatureDimField } from "@/features/configurator/sections/policy/FeatureDimField";
import { LayerListField } from "@/features/configurator/sections/policy/LayerEditors";
import type {
  ConfigMetadata,
  ManagedRunConfig,
  PolicyArchitecturePreview,
} from "@/shared/api/contract";
import { CnnConfigIcon } from "@/shared/ui/icons";

interface PolicySectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  checkpointLocked?: boolean;
  preview: PolicyArchitecturePreview | null;
  setConfig: ConfigSetter;
}

export function PolicySection({
  config,
  defaultConfig,
  metadata,
  checkpointLocked = false,
  preview,
  setConfig,
}: PolicySectionProps) {
  const updatePolicy = (patch: ConfigSectionPatch<"policy">) => {
    patchConfigSection(setConfig, "policy", patch);
  };
  const updateConvProfile = (value: ManagedRunConfig["policy"]["conv_profile"]) => {
    patchConfigSection(setConfig, "policy", (currentConfig) => ({
      conv_profile: value,
      custom_conv_layers:
        value === "custom" && currentConfig.policy.custom_conv_layers.length === 0
          ? defaultConfig.policy.custom_conv_layers
          : currentConfig.policy.custom_conv_layers,
    }));
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
  const fallbackFusionFeaturesDim =
    defaultConfig.policy.fusion_features_dim ?? preview?.fusion_input_dim ?? 768;
  const fusionFeaturesDim = config.policy.fusion_features_dim ?? fallbackFusionFeaturesDim;
  const fusionEnabled = config.policy.fusion_features_dim !== null;
  const updateFusionEnabled = (enabled: boolean) => {
    updatePolicy({ fusion_features_dim: enabled ? fusionFeaturesDim : null });
  };

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
            <BooleanField
              help="Insert a learned MLP after image/state concatenation. Turn off to feed the concatenated features directly to the recurrent core or heads."
              label="Fusion MLP"
              resetValue={defaultConfig.policy.fusion_features_dim !== null}
              value={fusionEnabled}
              onChange={updateFusionEnabled}
            />
            <fieldset className="dependent-fieldset" disabled={checkpointLocked || !fusionEnabled}>
              <IntegerField
                help="Feature width of the learned fusion MLP."
                label="Fusion features"
                min={1}
                resetValue={defaultConfig.policy.fusion_features_dim ?? fallbackFusionFeaturesDim}
                value={fusionFeaturesDim}
                onChange={(value) => updatePolicy({ fusion_features_dim: value })}
              />
            </fieldset>
            <BooleanField
              help="Apply layer normalization after the fusion stage or direct concatenation."
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
              auxiliary_state_head_arch: defaultConfig.policy.auxiliary_state_head_arch,
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
            <fieldset className="fork-lock-fieldset" disabled={false}>
              <LayerListField
                help="Shared MLP trunk used by the grouped auxiliary-state heads before their scalar, binary, and categorical outputs."
                label="Aux head"
                resetValue={defaultConfig.policy.auxiliary_state_head_arch}
                value={config.policy.auxiliary_state_head_arch}
                onChange={(value) => updatePolicy({ auxiliary_state_head_arch: value })}
              />
            </fieldset>
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
          checkpointLocked={checkpointLocked}
          convProfile={config.policy.conv_profile}
          customConvLayers={config.policy.custom_conv_layers}
          preview={preview}
          setCustomConvLayers={(value) => updatePolicy({ custom_conv_layers: value })}
        />
      </ConfigPanel>
    </div>
  );
}
