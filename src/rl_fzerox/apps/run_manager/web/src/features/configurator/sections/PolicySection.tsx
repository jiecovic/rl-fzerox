import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import {
  BooleanField,
  IntegerField,
  NumberField,
  SelectField,
} from "@/features/configurator/fields";
import { FieldLabel } from "@/features/configurator/fields/label";
import { PolicyPreviewPanel } from "@/features/configurator/sections/PolicyPreviewPanel";
import type {
  ConfigMetadata,
  ManagedRunConfig,
  PolicyArchitecturePreview,
} from "@/shared/api/contract";

interface PolicySectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  preview: PolicyArchitecturePreview | null;
  setConfig: (config: ManagedRunConfig) => void;
}

export function PolicySection({
  config,
  defaultConfig,
  metadata,
  preview,
  setConfig,
}: PolicySectionProps) {
  const updatePolicy = (patch: Partial<ManagedRunConfig["policy"]>) => {
    setConfig({ ...config, policy: { ...config.policy, ...patch } });
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
          onReset={() =>
            updatePolicy({
              conv_profile: defaultConfig.policy.conv_profile,
              features_dim: defaultConfig.policy.features_dim,
              state_net_arch: defaultConfig.policy.state_net_arch,
              fusion_features_dim: defaultConfig.policy.fusion_features_dim,
              layer_norm: defaultConfig.policy.layer_norm,
            })
          }
        >
          <div className="training-field-grid">
            <SelectField
              help="Convolution stack used for the image branch."
              label="CNN profile"
              options={convProfileOptions}
              resetValue={defaultConfig.policy.conv_profile}
              value={config.policy.conv_profile}
              onChange={(value) => updatePolicy({ conv_profile: value })}
            />
            <SelectField
              help="Image feature width after CNN flatten. Auto keeps the raw flatten size."
              label="Image features"
              options={["auto", "256", "512", "768", "1024"]}
              resetValue={String(defaultConfig.policy.features_dim)}
              value={String(config.policy.features_dim)}
              onChange={(value) =>
                updatePolicy({ features_dim: value === "auto" ? "auto" : Number(value) })
              }
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
          </div>
        </ConfigPanel>

        <ConfigPanel
          title="Recurrent core"
          onReset={() =>
            updatePolicy({
              recurrent_enable_critic_lstm: defaultConfig.policy.recurrent_enable_critic_lstm,
              recurrent_enabled: defaultConfig.policy.recurrent_enabled,
              recurrent_hidden_size: defaultConfig.policy.recurrent_hidden_size,
              recurrent_n_lstm_layers: defaultConfig.policy.recurrent_n_lstm_layers,
              recurrent_shared_lstm: defaultConfig.policy.recurrent_shared_lstm,
            })
          }
        >
          <div className="training-field-grid">
            <BooleanField
              help="Insert LSTM actor/critic cores between extractor and heads."
              label="Use LSTM"
              resetValue={defaultConfig.policy.recurrent_enabled}
              value={config.policy.recurrent_enabled}
              onChange={(value) => updatePolicy({ recurrent_enabled: value })}
            />
            <fieldset className="dependent-fieldset" disabled={!config.policy.recurrent_enabled}>
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
          </div>
        </ConfigPanel>

        <ConfigPanel
          title="Heads"
          onReset={() =>
            updatePolicy({
              gas_on_logit: defaultConfig.policy.gas_on_logit,
              pi_net_arch: defaultConfig.policy.pi_net_arch,
              vf_net_arch: defaultConfig.policy.vf_net_arch,
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
            <NumberField
              help="Initial logit nudge for the gas-on action."
              label="Gas-on logit"
              resetValue={defaultConfig.policy.gas_on_logit}
              step="0.1"
              value={config.policy.gas_on_logit}
              onChange={(value) => updatePolicy({ gas_on_logit: value })}
            />
          </div>
        </ConfigPanel>
      </div>

      <ConfigPanel title="Architecture preview" wide>
        <PolicyPreviewPanel preview={preview} />
      </ConfigPanel>
    </div>
  );
}

function LayerListField({
  help,
  label,
  resetValue,
  value,
  onChange,
}: {
  help: string;
  label: string;
  resetValue: number[];
  value: number[];
  onChange: (value: number[]) => void;
}) {
  function setLayer(index: number, nextValue: number) {
    if (!Number.isSafeInteger(nextValue) || nextValue <= 0) {
      return;
    }
    onChange(value.map((layer, layerIndex) => (layerIndex === index ? nextValue : layer)));
  }

  function addLayer() {
    onChange([...value, value.at(-1) ?? 256]);
  }

  function removeLayer(index: number) {
    onChange(value.filter((_, layerIndex) => layerIndex !== index));
  }

  return (
    <div className="field-shell layer-list-field">
      <FieldLabel
        help={help}
        label={label}
        onReset={layerListResetHandler(value, resetValue, onChange)}
      />
      <div className="layer-list-editor">
        {value.length === 0 ? <span className="layer-list-empty">No hidden layers</span> : null}
        {value.map((layer, index) => (
          <div className="layer-list-row" key={layerRowKey(label, value, index)}>
            <span className="layer-index">L{index + 1}</span>
            <input
              aria-label={`${label} layer ${index + 1}`}
              min={1}
              step={1}
              type="number"
              value={layer}
              onChange={(event) => setLayer(index, Number(event.target.value))}
            />
            <button
              aria-label={`Remove ${label} layer ${index + 1}`}
              className="field-reset-button tooltip-anchor"
              data-tooltip="Remove layer"
              type="button"
              onClick={() => removeLayer(index)}
            >
              <RemoveLayerIcon />
            </button>
          </div>
        ))}
        <div className="layer-list-row layer-add-row">
          <span className="layer-index">L{value.length + 1}</span>
          <button className="layer-add-placeholder" type="button" onClick={addLayer}>
            Add layer
          </button>
          <button
            aria-label={`Add ${label} layer`}
            className="field-reset-button layer-add-button tooltip-anchor"
            data-tooltip="Add layer"
            type="button"
            onClick={addLayer}
          >
            <AddLayerIcon />
          </button>
        </div>
      </div>
    </div>
  );
}

function layerListResetHandler(
  value: number[],
  resetValue: number[],
  onChange: (value: number[]) => void,
) {
  if (sameLayerList(value, resetValue)) {
    return undefined;
  }
  return () => onChange(resetValue);
}

function sameLayerList(left: number[], right: number[]) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function layerRowKey(label: string, layers: number[], index: number) {
  return `${label}-${layers.slice(0, index + 1).join("-")}`;
}

function AddLayerIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 20 20" width="14">
      <path d="M10 4v12M4 10h12" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  );
}

function RemoveLayerIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="12" viewBox="0 0 20 20" width="12">
      <path d="M5 10h10" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  );
}
