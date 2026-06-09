// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/PolicySection.tsx

import {
  ConfigFieldGroup,
  ConfigFieldset,
  ConfigGrid,
  ConfigStack,
} from "@/features/configurator/ConfigLayout";
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
import { Button } from "@/shared/ui/Button";
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
    setConfig((currentConfig) => {
      const nextConfig: ManagedRunConfig = {
        ...currentConfig,
        policy: {
          ...currentConfig.policy,
          conv_profile: value,
          custom_conv_layers:
            value === "custom" && currentConfig.policy.custom_conv_layers.length === 0
              ? defaultConfig.policy.custom_conv_layers
              : currentConfig.policy.custom_conv_layers,
        },
      };
      if (value !== "impala_small" && value !== "impala_large") {
        return nextConfig;
      }
      return {
        ...nextConfig,
        observation: {
          ...currentConfig.observation,
          resolution: { mode: "preset", preset: "crop_72x96" },
        },
      };
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
  const optionalActivationOptions = ["none", ...activationOptions] as (
    | "none"
    | ManagedRunConfig["policy"]["activation"]
  )[];
  const layerNormActivationValue = config.policy.layer_norm_activation ?? "none";
  const defaultLayerNormActivationValue = defaultConfig.policy.layer_norm_activation ?? "none";
  const fallbackFusionFeaturesDim =
    defaultConfig.policy.fusion_features_dim ?? preview?.fusion_input_dim ?? 768;
  const fusionFeaturesDim = config.policy.fusion_features_dim ?? fallbackFusionFeaturesDim;
  const fusionEnabled = config.policy.fusion_features_dim !== null;
  const updateFusionEnabled = (enabled: boolean) => {
    updatePolicy({ fusion_features_dim: enabled ? fusionFeaturesDim : null });
  };

  return (
    <ConfigStack>
      <ConfigGrid columns="two" className="items-stretch">
        <ConfigPanel
          className="order-1"
          title="Image extractor"
          onReset={
            checkpointLocked
              ? undefined
              : () =>
                  updatePolicy({
                    conv_profile: defaultConfig.policy.conv_profile,
                    custom_conv_layers: defaultConfig.policy.custom_conv_layers,
                    features_dim: defaultConfig.policy.features_dim,
                    image_projection_activation: defaultConfig.policy.image_projection_activation,
                  })
          }
        >
          <ConfigFieldset disabled={checkpointLocked}>
            <ConfigGrid columns="two" className="items-start">
              <ConfigFieldGroup>
                <SelectField
                  help="Backend CNN profile used for the image branch. Preset layers are read-only until you copy them with Edit as custom in the preview."
                  label="CNN profile"
                  options={convProfileOptions}
                  resetValue={defaultConfig.policy.conv_profile}
                  value={config.policy.conv_profile}
                  onChange={updateConvProfile}
                />
                <Button
                  className="justify-self-start gap-2"
                  type="button"
                  onClick={jumpToCnnConfigurator}
                >
                  <CnnConfigIcon />
                  {config.policy.conv_profile === "custom" ? "Edit CNN" : "View CNN"}
                </Button>
              </ConfigFieldGroup>
              <ConfigFieldGroup>
                <FeatureDimField
                  help="Image feature width after CNN flatten. Auto keeps the raw flatten size."
                  label="Image features"
                  resetValue={defaultConfig.policy.features_dim}
                  value={config.policy.features_dim}
                  onChange={(value) => updatePolicy({ features_dim: value })}
                />
                <ConfigFieldset
                  disabled={checkpointLocked || config.policy.features_dim === "auto"}
                >
                  <SelectField
                    help="Activation after the optional linear projection from flattened CNN output to the configured image feature width. Disabled when Image features is Auto."
                    label="Projection activation"
                    options={activationOptions}
                    resetValue={defaultConfig.policy.image_projection_activation}
                    value={config.policy.image_projection_activation}
                    onChange={(value) => updatePolicy({ image_projection_activation: value })}
                  />
                </ConfigFieldset>
              </ConfigFieldGroup>
            </ConfigGrid>
          </ConfigFieldset>
        </ConfigPanel>

        <ConfigPanel
          className="order-4"
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
          <ConfigFieldset disabled={checkpointLocked}>
            <ConfigGrid columns="three" className="items-start">
              <BooleanField
                help="Insert LSTM actor/critic cores between extractor and heads."
                label="Use LSTM"
                resetValue={defaultConfig.policy.recurrent_enabled}
                value={config.policy.recurrent_enabled}
                onChange={(value) => updatePolicy({ recurrent_enabled: value })}
              />
              <ConfigFieldset disabled={checkpointLocked || !config.policy.recurrent_enabled}>
                <IntegerField
                  help="Hidden width of the recurrent actor and critic."
                  label="LSTM hidden"
                  min={1}
                  resetValue={defaultConfig.policy.recurrent_hidden_size}
                  value={config.policy.recurrent_hidden_size}
                  onChange={(value) => updatePolicy({ recurrent_hidden_size: value })}
                />
              </ConfigFieldset>
              <ConfigFieldset disabled={checkpointLocked || !config.policy.recurrent_enabled}>
                <IntegerField
                  help="Number of recurrent layers."
                  label="LSTM layers"
                  min={1}
                  resetValue={defaultConfig.policy.recurrent_n_lstm_layers}
                  value={config.policy.recurrent_n_lstm_layers}
                  onChange={(value) => updatePolicy({ recurrent_n_lstm_layers: value })}
                />
              </ConfigFieldset>
              <ConfigFieldset disabled={checkpointLocked || !config.policy.recurrent_enabled}>
                <BooleanField
                  help="Share one LSTM between actor and critic."
                  label="Shared LSTM"
                  resetValue={defaultConfig.policy.recurrent_shared_lstm}
                  value={config.policy.recurrent_shared_lstm}
                  onChange={(value) => updatePolicy({ recurrent_shared_lstm: value })}
                />
              </ConfigFieldset>
              <ConfigFieldset disabled={checkpointLocked || !config.policy.recurrent_enabled}>
                <BooleanField
                  help="Use a critic LSTM. If off, the critic uses non-recurrent features."
                  label="Critic LSTM"
                  resetValue={defaultConfig.policy.recurrent_enable_critic_lstm}
                  value={config.policy.recurrent_enable_critic_lstm}
                  onChange={(value) => updatePolicy({ recurrent_enable_critic_lstm: value })}
                />
              </ConfigFieldset>
            </ConfigGrid>
          </ConfigFieldset>
        </ConfigPanel>

        <ConfigPanel
          className="order-2"
          title="State encoder"
          onReset={
            checkpointLocked
              ? undefined
              : () =>
                  updatePolicy({
                    state_net_arch: defaultConfig.policy.state_net_arch,
                    state_activation: defaultConfig.policy.state_activation,
                  })
          }
        >
          <ConfigFieldset disabled={checkpointLocked}>
            <ConfigFieldGroup>
              <LayerListField
                help="State branch MLP layers before image/state fusion. Remove all layers to concatenate the raw state vector."
                label="State MLP"
                resetValue={defaultConfig.policy.state_net_arch}
                value={config.policy.state_net_arch}
                onChange={(value) => updatePolicy({ state_net_arch: value })}
              />
              <ConfigFieldset
                disabled={checkpointLocked || config.policy.state_net_arch.length === 0}
              >
                <SelectField
                  help="Activation after each state-branch MLP layer."
                  label="State activation"
                  options={activationOptions}
                  resetValue={defaultConfig.policy.state_activation}
                  value={config.policy.state_activation}
                  onChange={(value) => updatePolicy({ state_activation: value })}
                />
              </ConfigFieldset>
            </ConfigFieldGroup>
          </ConfigFieldset>
        </ConfigPanel>

        <ConfigPanel
          className="order-3"
          title="Feature fusion"
          onReset={
            checkpointLocked
              ? undefined
              : () =>
                  updatePolicy({
                    fusion_features_dim: defaultConfig.policy.fusion_features_dim,
                    fusion_activation: defaultConfig.policy.fusion_activation,
                    layer_norm: defaultConfig.policy.layer_norm,
                    layer_norm_activation: defaultConfig.policy.layer_norm_activation,
                  })
          }
        >
          <ConfigFieldset disabled={checkpointLocked}>
            <ConfigFieldGroup>
              <ConfigGrid columns="two" className="items-start">
                <BooleanField
                  help="Insert a learned MLP after image/state concatenation. Turn off to feed the concatenated features directly to the recurrent core or heads."
                  label="Fusion MLP"
                  resetValue={defaultConfig.policy.fusion_features_dim !== null}
                  value={fusionEnabled}
                  onChange={updateFusionEnabled}
                />
                <BooleanField
                  help="Apply layer normalization after the fusion stage or direct concatenation."
                  label="Layer norm"
                  resetValue={defaultConfig.policy.layer_norm}
                  value={config.policy.layer_norm}
                  onChange={(value) => updatePolicy({ layer_norm: value })}
                />
              </ConfigGrid>
              <ConfigFieldset disabled={checkpointLocked || !fusionEnabled}>
                <ConfigGrid columns="two" className="items-start">
                  <IntegerField
                    help="Feature width of the learned fusion MLP."
                    label="Fusion features"
                    min={1}
                    resetValue={
                      defaultConfig.policy.fusion_features_dim ?? fallbackFusionFeaturesDim
                    }
                    value={fusionFeaturesDim}
                    onChange={(value) => updatePolicy({ fusion_features_dim: value })}
                  />
                  <SelectField
                    help="Activation after the learned fusion projection."
                    label="Fusion activation"
                    options={activationOptions}
                    resetValue={defaultConfig.policy.fusion_activation}
                    value={config.policy.fusion_activation}
                    onChange={(value) => updatePolicy({ fusion_activation: value })}
                  />
                </ConfigGrid>
              </ConfigFieldset>
              <ConfigFieldset disabled={checkpointLocked || !config.policy.layer_norm}>
                <div className="w-[240px] max-w-full">
                  <SelectField
                    help="Optional activation applied after layer normalization."
                    label="Post-LN activation"
                    optionLabels={{ none: "None" }}
                    options={optionalActivationOptions}
                    resetValue={defaultLayerNormActivationValue}
                    value={layerNormActivationValue}
                    onChange={(value) =>
                      updatePolicy({ layer_norm_activation: value === "none" ? null : value })
                    }
                  />
                </div>
              </ConfigFieldset>
            </ConfigFieldGroup>
          </ConfigFieldset>
        </ConfigPanel>

        <ConfigPanel
          className="order-5"
          title="Heads"
          wide
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
          <ConfigFieldGroup>
            <div className="w-[240px] max-w-full">
              <SelectField
                help="Shared activation after hidden MLP layers in the auxiliary, policy, and value heads."
                label="Aux/policy/value activation"
                options={activationOptions}
                resetValue={defaultConfig.policy.activation}
                value={config.policy.activation}
                onChange={(value) => updatePolicy({ activation: value })}
              />
            </div>
            <ConfigGrid columns="three" className="items-start">
              <ConfigFieldset disabled={false}>
                <LayerListField
                  help="Shared MLP trunk used by the grouped auxiliary-state heads before their scalar, binary, and categorical outputs."
                  label="Aux head"
                  resetValue={defaultConfig.policy.auxiliary_state_head_arch}
                  value={config.policy.auxiliary_state_head_arch}
                  onChange={(value) => updatePolicy({ auxiliary_state_head_arch: value })}
                />
              </ConfigFieldset>
              <ConfigFieldset disabled={checkpointLocked}>
                <LayerListField
                  help="Policy MLP layers after the recurrent core. Add or remove rows to change the head depth."
                  label="Policy head"
                  resetValue={defaultConfig.policy.pi_net_arch}
                  value={config.policy.pi_net_arch}
                  onChange={(value) => updatePolicy({ pi_net_arch: value })}
                />
              </ConfigFieldset>
              <ConfigFieldset disabled={checkpointLocked}>
                <LayerListField
                  help="Value MLP layers after the recurrent core. Add or remove rows to change the head depth."
                  label="Value head"
                  resetValue={defaultConfig.policy.vf_net_arch}
                  value={config.policy.vf_net_arch}
                  onChange={(value) => updatePolicy({ vf_net_arch: value })}
                />
              </ConfigFieldset>
            </ConfigGrid>
          </ConfigFieldGroup>
        </ConfigPanel>
      </ConfigGrid>

      <ConfigPanel title="Architecture preview" wide>
        <PolicyPreviewPanel
          checkpointLocked={checkpointLocked}
          convProfile={config.policy.conv_profile}
          customConvLayers={config.policy.custom_conv_layers}
          preview={preview}
          convertPresetToCustom={(value) => {
            setConfig((currentConfig) => ({
              ...currentConfig,
              policy: {
                ...currentConfig.policy,
                conv_profile: "custom",
                custom_conv_layers: value,
              },
            }));
          }}
          setCustomConvLayers={(value) => {
            setConfig((currentConfig) => ({
              ...currentConfig,
              policy: {
                ...currentConfig.policy,
                conv_profile: "custom",
                custom_conv_layers: value,
              },
            }));
          }}
        />
      </ConfigPanel>
    </ConfigStack>
  );
}
