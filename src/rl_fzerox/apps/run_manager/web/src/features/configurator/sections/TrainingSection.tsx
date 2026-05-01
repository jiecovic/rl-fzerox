import type { ReactNode } from "react";
import {
  BooleanField,
  DiscreteSliderNumberField,
  LogRangeNumberField,
  OptionalNumberField,
  RangeIntegerField,
  RangeNumberField,
  ResetIcon,
} from "@/features/configurator/fields";
import type { ManagedRunConfig } from "@/shared/api/contract";

interface ConfigSectionProps {
  config: ManagedRunConfig;
  defaultConfig: ManagedRunConfig;
  setConfig: (config: ManagedRunConfig) => void;
}

export function TrainingSection({ config, defaultConfig, setConfig }: ConfigSectionProps) {
  const updateTrain = (patch: Partial<ManagedRunConfig["train"]>) => {
    setConfig({ ...config, train: { ...config.train, ...patch } });
  };
  const derived = trainingDerivedValues(config.train);
  const rolloutStepValues = compatibleRolloutSteps(config.train);

  return (
    <div className="config-stack training-panel-grid">
      <ConfigPanel
        title="Rollout"
        onReset={() => updateTrain(rolloutDefaults(defaultConfig.train))}
      >
        <div className="training-field-grid">
          <RangeNumberField
            help="Parallel emulator environments used for rollout collection."
            label="Env count"
            max={32}
            min={1}
            rangeStep={1}
            ticks={[
              { value: 1, label: "1" },
              { value: 16, label: "16" },
              { value: 32, label: "32" },
            ]}
            resetValue={defaultConfig.train.num_envs}
            value={config.train.num_envs}
            onChange={(value) => updateTrain({ num_envs: value })}
          />
          <RangeIntegerField
            help="Training step target for this run."
            label="Target steps"
            max={100_000_000}
            min={1_000_000}
            rangeStep={1_000_000}
            ticks={[
              { value: 1_000_000, label: "1M" },
              { value: 50_000_000, label: "50M" },
              { value: 100_000_000, label: "100M" },
            ]}
            resetValue={defaultConfig.train.total_timesteps}
            value={config.train.total_timesteps}
            onChange={(value) => updateTrain({ total_timesteps: value })}
          />
          <DiscreteSliderNumberField
            help="PPO rollout length collected per environment before an update."
            label="Rollout steps"
            maxManual={8192}
            minManual={128}
            resetValue={defaultConfig.train.n_steps}
            snapManualToOptions
            sliderValues={rolloutStepValues}
            value={config.train.n_steps}
            onChange={(value) => updateTrain({ n_steps: value })}
          />
          <DiscreteSliderNumberField
            help="Mini-batch size used during policy updates. Slider snaps to common powers of two, number input accepts any positive integer."
            label="Batch size"
            maxManual={8192}
            minManual={1}
            resetValue={defaultConfig.train.batch_size}
            sliderValues={[64, 128, 256, 512, 1024]}
            value={config.train.batch_size}
            onChange={(value) => updateTrain({ batch_size: value })}
          />
          <RangeNumberField
            help="Optimization epochs over each rollout buffer."
            label="Epochs"
            max={16}
            min={1}
            rangeStep={1}
            ticks={[
              { value: 1, label: "1" },
              { value: 8, label: "8" },
              { value: 16, label: "16" },
            ]}
            resetValue={defaultConfig.train.n_epochs}
            value={config.train.n_epochs}
            onChange={(value) => updateTrain({ n_epochs: value })}
          />
          <RangeNumberField
            help="Episode return discount factor."
            label="Gamma"
            max={0.9999}
            min={0.9}
            numberStep="0.0001"
            rangeStep={0.0001}
            ticks={[
              { value: 0.9, label: "0.90" },
              { value: 0.95, label: "0.95" },
              { value: 0.9999, label: "0.9999" },
            ]}
            resetValue={defaultConfig.train.gamma}
            value={config.train.gamma}
            onChange={(value) => updateTrain({ gamma: value })}
          />
        </div>
      </ConfigPanel>

      <ConfigPanel
        title="Optimization"
        onReset={() => updateTrain(optimizationDefaults(defaultConfig.train))}
      >
        <div className="training-field-grid">
          <LogRangeNumberField
            help="Optimizer step size."
            label="Learning rate"
            max={1e-3}
            min={1e-7}
            ticks={[
              { value: 1e-7, label: "1e-7" },
              { value: 1e-5, label: "1e-5" },
              { value: 1e-4, label: "1e-4" },
              { value: 1e-3, label: "1e-3" },
            ]}
            resetValue={defaultConfig.train.learning_rate}
            value={config.train.learning_rate}
            onChange={(value) => updateTrain({ learning_rate: value })}
          />
          <RangeNumberField
            help="GAE smoothing factor for advantage estimates."
            label="GAE lambda"
            max={1}
            min={0.8}
            numberStep="0.001"
            rangeStep={0.01}
            ticks={[
              { value: 0.8, label: "0.80" },
              { value: 0.9, label: "0.90" },
              { value: 1, label: "1.0" },
            ]}
            resetValue={defaultConfig.train.gae_lambda}
            value={config.train.gae_lambda}
            onChange={(value) => updateTrain({ gae_lambda: value })}
          />
          <RangeNumberField
            help="Policy-ratio clipping width."
            label="Clip range"
            max={0.5}
            min={0.01}
            numberStep="0.001"
            rangeStep={0.01}
            ticks={[
              { value: 0.01, label: ".01" },
              { value: 0.25, label: ".25" },
              { value: 0.5, label: ".50" },
            ]}
            resetValue={defaultConfig.train.clip_range}
            value={config.train.clip_range}
            onChange={(value) => updateTrain({ clip_range: value })}
          />
          <OptionalNumberField
            defaultValue={0.2}
            help="Optional value-function clipping width. Off matches SB3's default no value clipping."
            label="Value clip range"
            max={1}
            min={0}
            resetValue={defaultConfig.train.clip_range_vf}
            step="0.01"
            value={config.train.clip_range_vf}
            onChange={(value) => updateTrain({ clip_range_vf: value })}
          />
          <RangeNumberField
            help="Entropy bonus weight for exploration."
            label="Entropy coefficient"
            max={0.05}
            min={0}
            numberStep="0.0001"
            rangeStep={0.001}
            ticks={[
              { value: 0, label: "0" },
              { value: 0.025, label: ".025" },
              { value: 0.05, label: ".05" },
            ]}
            resetValue={defaultConfig.train.ent_coef}
            value={config.train.ent_coef}
            onChange={(value) => updateTrain({ ent_coef: value })}
          />
          <RangeNumberField
            help="Value-loss coefficient."
            label="Value coefficient"
            max={2}
            min={0.01}
            numberStep="0.01"
            rangeStep={0.05}
            ticks={[
              { value: 0.01, label: ".01" },
              { value: 1, label: "1" },
              { value: 2, label: "2" },
            ]}
            resetValue={defaultConfig.train.vf_coef}
            value={config.train.vf_coef}
            onChange={(value) => updateTrain({ vf_coef: value })}
          />
        </div>
      </ConfigPanel>

      <ConfigPanel
        title="Stability"
        onReset={() => updateTrain(stabilityDefaults(defaultConfig.train))}
      >
        <div className="training-field-grid">
          <RangeNumberField
            help="Gradient norm clipping threshold."
            label="Max grad norm"
            max={5}
            min={0.01}
            numberStep="0.01"
            rangeStep={0.05}
            ticks={[
              { value: 0.01, label: ".01" },
              { value: 2.5, label: "2.5" },
              { value: 5, label: "5" },
            ]}
            resetValue={defaultConfig.train.max_grad_norm}
            value={config.train.max_grad_norm}
            onChange={(value) => updateTrain({ max_grad_norm: value })}
          />
          <BooleanField
            help="Normalize advantages per batch before optimization."
            label="Normalize advantage"
            resetValue={defaultConfig.train.normalize_advantage}
            value={config.train.normalize_advantage}
            onChange={(value) => updateTrain({ normalize_advantage: value })}
          />
          <OptionalNumberField
            defaultValue={0.03}
            help="Optional early-stop KL threshold for PPO updates."
            label="Target KL"
            max={0.2}
            min={0}
            resetValue={defaultConfig.train.target_kl}
            step="0.001"
            value={config.train.target_kl}
            onChange={(value) => updateTrain({ target_kl: value })}
          />
        </div>
      </ConfigPanel>

      <ConfigPanel title="PPO update summary" wide>
        <table className="derived-table">
          <tbody>
            <DerivedRow
              expression={`${config.train.num_envs} envs x ${config.train.n_steps.toLocaleString()} steps`}
              help="Total rollout transitions collected before one PPO update. Larger values make each update more stable but slower to refresh from the current policy."
              label="Rollout samples"
              value={derived.rolloutSamples.toLocaleString()}
            />
            <DerivedRow
              expression={`ceil(${derived.rolloutSamples.toLocaleString()} / ${config.train.batch_size.toLocaleString()})`}
              help="How many mini-batches are needed to consume one rollout once. If this is very small, each optimizer step sees a large chunk of the rollout."
              label="Minibatches per epoch"
              value={derived.minibatchesPerEpoch.toLocaleString()}
            />
            <DerivedRow
              expression={`${derived.minibatchesPerEpoch.toLocaleString()} minibatches x ${config.train.n_epochs} epochs`}
              help="Total optimizer steps from one collected rollout. More steps reuse the same on-policy data harder, which can improve learning or overfit stale rollouts."
              label="Optimizer minibatches per update"
              value={derived.optimizerMinibatches.toLocaleString()}
            />
            <DerivedRow
              expression={`${config.train.batch_size.toLocaleString()} / ${derived.rolloutSamples.toLocaleString()}`}
              help="Share of the rollout seen by one mini-batch. A practical PPO target is roughly 2-10%; going much higher makes each minibatch too close to the full rollout, while very tiny batches can get noisy."
              label="Batch coverage"
              value={`${derived.batchCoverage.toFixed(1)}%`}
            />
          </tbody>
        </table>
      </ConfigPanel>
    </div>
  );
}

function ConfigPanel({
  children,
  onReset,
  title,
  wide = false,
}: {
  children: ReactNode;
  onReset?: () => void;
  title: string;
  wide?: boolean;
}) {
  return (
    <section className={wide ? "config-group wide" : "config-group"}>
      <div className="config-group-header">
        <h3>{title}</h3>
        {onReset !== undefined ? (
          <button
            aria-label={`Reset ${title} defaults`}
            className="reset-button"
            type="button"
            onClick={onReset}
          >
            <ResetIcon />
          </button>
        ) : null}
      </div>
      {children}
    </section>
  );
}

function rolloutDefaults(train: ManagedRunConfig["train"]): Partial<ManagedRunConfig["train"]> {
  return {
    batch_size: train.batch_size,
    gamma: train.gamma,
    n_epochs: train.n_epochs,
    n_steps: train.n_steps,
    num_envs: train.num_envs,
    total_timesteps: train.total_timesteps,
  };
}

function optimizationDefaults(
  train: ManagedRunConfig["train"],
): Partial<ManagedRunConfig["train"]> {
  return {
    clip_range: train.clip_range,
    clip_range_vf: train.clip_range_vf,
    ent_coef: train.ent_coef,
    gae_lambda: train.gae_lambda,
    learning_rate: train.learning_rate,
    vf_coef: train.vf_coef,
  };
}

function stabilityDefaults(train: ManagedRunConfig["train"]): Partial<ManagedRunConfig["train"]> {
  return {
    max_grad_norm: train.max_grad_norm,
    normalize_advantage: train.normalize_advantage,
    stats_window_size: train.stats_window_size,
    target_kl: train.target_kl,
  };
}

function trainingDerivedValues(train: ManagedRunConfig["train"]) {
  const rolloutSamples = train.num_envs * train.n_steps;
  const minibatchesPerEpoch = Math.ceil(rolloutSamples / train.batch_size);
  const optimizerMinibatches = minibatchesPerEpoch * train.n_epochs;
  const batchCoverage = (Math.min(train.batch_size, rolloutSamples) / rolloutSamples) * 100;
  return { rolloutSamples, minibatchesPerEpoch, optimizerMinibatches, batchCoverage };
}

function compatibleRolloutSteps(train: ManagedRunConfig["train"]) {
  const minSteps = 128;
  const maxSteps = 8192;
  const uiStep = 128;
  const requiredMultiple =
    train.batch_size / greatestCommonDivisor(train.batch_size, train.num_envs);
  const stepMultiple = leastCommonMultiple(uiStep, requiredMultiple);
  const values: number[] = [];
  for (let value = stepMultiple; value <= maxSteps; value += stepMultiple) {
    if (value >= minSteps) {
      values.push(value);
    }
  }
  if (values.length > 0) {
    return values;
  }
  return [train.n_steps];
}

function greatestCommonDivisor(left: number, right: number): number {
  let a = Math.abs(Math.trunc(left));
  let b = Math.abs(Math.trunc(right));
  while (b !== 0) {
    const next = a % b;
    a = b;
    b = next;
  }
  return a || 1;
}

function leastCommonMultiple(left: number, right: number): number {
  return Math.abs(left * right) / greatestCommonDivisor(left, right);
}

function DerivedRow({
  expression,
  help,
  label,
  value,
}: {
  expression: string;
  help: string;
  label: string;
  value: string;
}) {
  return (
    <tr>
      <th scope="row">
        <span className="derived-label">
          <span>{label}</span>
          <InfoIcon label={label} text={help} />
        </span>
      </th>
      <td>{expression}</td>
      <td>{value}</td>
    </tr>
  );
}

function InfoIcon({ label, text }: { label: string; text: string }) {
  return (
    <button
      aria-label={`${label}: ${text}`}
      className="field-help"
      data-tooltip={text}
      type="button"
    >
      ?
    </button>
  );
}
