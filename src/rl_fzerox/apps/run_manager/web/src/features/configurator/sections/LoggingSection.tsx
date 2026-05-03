import { ConfigPanel } from "@/features/configurator/ConfigPanel";
import { BooleanField, RangeIntegerField } from "@/features/configurator/fields";
import {
  checkpointCadenceSummary,
  checkpointSummaryRows,
} from "@/features/configurator/sections/logging/derived";
import { RecentRetentionField } from "@/features/configurator/sections/logging/RecentRetentionField";
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
    <div className="config-stack training-panel-grid">
      <ConfigPanel
        title="Checkpoints"
        onReset={() =>
          updateTrain({
            checkpoint_every_rollouts: defaultConfig.train.checkpoint_every_rollouts,
            save_latest_checkpoint: defaultConfig.train.save_latest_checkpoint,
            save_best_checkpoint: defaultConfig.train.save_best_checkpoint,
            save_recent_checkpoints: defaultConfig.train.save_recent_checkpoints,
            recent_checkpoint_limit: defaultConfig.train.recent_checkpoint_limit,
          })
        }
      >
        <div className="training-field-grid">
          <div className="field-with-note">
            <RangeIntegerField
              help="For PPO runs, periodic checkpoint writes happen after whole rollouts. One rollout equals env count times rollout steps."
              label="Checkpoint cadence"
              max={50}
              min={1}
              rangeStep={1}
              ticks={[
                { value: 1, label: "1" },
                { value: 5, label: "5" },
                { value: 25, label: "25" },
                { value: 50, label: "50" },
              ]}
              resetValue={defaultConfig.train.checkpoint_every_rollouts}
              value={config.train.checkpoint_every_rollouts}
              onChange={(value) => updateTrain({ checkpoint_every_rollouts: value })}
            />
            <div className="field-note">{checkpointCadenceSummary(config.train)}</div>
          </div>
          <BooleanField
            help="Keep one rolling latest checkpoint updated on the periodic cadence."
            label="Save latest"
            resetValue={defaultConfig.train.save_latest_checkpoint}
            value={config.train.save_latest_checkpoint}
            onChange={(value) => updateTrain({ save_latest_checkpoint: value })}
          />
          <BooleanField
            help="Update the best checkpoint whenever the episode return improves."
            label="Save best"
            resetValue={defaultConfig.train.save_best_checkpoint}
            value={config.train.save_best_checkpoint}
            onChange={(value) => updateTrain({ save_best_checkpoint: value })}
          />
          <BooleanField
            help="Keep numbered snapshots under checkpoints/ at the same periodic cadence."
            label="Keep recent"
            resetValue={defaultConfig.train.save_recent_checkpoints}
            value={config.train.save_recent_checkpoints}
            onChange={(value) => updateTrain({ save_recent_checkpoints: value })}
          />
          <fieldset className="dependent-fieldset" disabled={!config.train.save_recent_checkpoints}>
            <RecentRetentionField
              defaultTrain={defaultConfig.train}
              help="Slide right to keep the newest N numbered snapshots. The last stop keeps all numbered checkpoints without trimming."
              label="Recent retention"
              train={config.train}
              onChange={(value) => updateTrain({ recent_checkpoint_limit: value })}
            />
          </fieldset>
        </div>
      </ConfigPanel>

      <ConfigPanel
        title="Stats"
        onReset={() =>
          updateTrain({
            stats_window_size: defaultConfig.train.stats_window_size,
          })
        }
      >
        <div className="training-field-grid">
          <RangeIntegerField
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
      </ConfigPanel>

      <ConfigPanel title="Checkpoint summary" wide>
        <table className="derived-table">
          <tbody>
            {checkpointSummaryRows(config.train).map((row) => (
              <tr key={row.label}>
                <th>{row.label}</th>
                <td>{row.detail}</td>
                <td>{row.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ConfigPanel>
    </div>
  );
}
