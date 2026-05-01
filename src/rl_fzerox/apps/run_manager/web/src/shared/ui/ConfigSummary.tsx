import type { ManagedRunConfig } from "@/shared/api/contract";

export function ConfigSummary({ config }: { config: ManagedRunConfig }) {
  return (
    <div className="summary-grid">
      <SummaryItem
        label="Training"
        value={`${config.train.num_envs} envs · ${config.train.total_timesteps.toLocaleString()} steps · lr ${config.train.learning_rate.toExponential(2)}`}
      />
      <SummaryItem
        label="Observation"
        value={`${config.observation.stack_mode} x${config.observation.frame_stack} · ${config.observation.progress_source}`}
      />
      <SummaryItem
        label="Policy"
        value={`${config.policy.conv_profile} · LSTM ${config.policy.recurrent_hidden_size} · fusion ${config.policy.fusion_features_dim}`}
      />
      <SummaryItem
        label="Reward"
        value={`boost ${config.reward.manual_boost_reward} · pad ${config.reward.boost_pad_reward} · lean ${config.reward.lean_request_penalty}`}
      />
    </div>
  );
}

function SummaryItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="summary-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
