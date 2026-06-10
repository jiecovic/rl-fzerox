// src/rl_fzerox/apps/run_manager/web/src/widgets/saveGameWorkspace/RunnerControlPanel.tsx
import type { PolicyPlaybackMode, WatchDevice, WatchRenderer } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { FieldInput, FieldSelect, FieldShell } from "@/shared/ui/Field";
import { PlayIcon, RandomizeIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

export function RunnerControlPanel({
  attemptSeedText,
  canStart,
  onAttemptSeedChange,
  onRandomizeAttemptSeed,
  onPolicyModeChange,
  onRunnerDeviceChange,
  onRunnerRendererChange,
  onStart,
  policyMode,
  rendererOptions,
  runnerDevice,
  runnerRenderer,
  startLabel,
  starting,
  startNote,
}: {
  attemptSeedText: string;
  canStart: boolean;
  onAttemptSeedChange: (attemptSeedText: string) => void;
  onRandomizeAttemptSeed: () => void;
  onPolicyModeChange: (policyMode: PolicyPlaybackMode) => void;
  onRunnerDeviceChange: (device: WatchDevice) => void;
  onRunnerRendererChange: (renderer: WatchRenderer) => void;
  onStart: () => void;
  policyMode: PolicyPlaybackMode;
  rendererOptions: readonly WatchRenderer[];
  runnerDevice: WatchDevice;
  runnerRenderer: WatchRenderer;
  startLabel: string;
  starting: boolean;
  startNote: string;
}) {
  const disabled = startLabel === "Running" || starting;
  return (
    <div className="mb-5 grid gap-3 border border-app-border bg-app-surface px-3 py-3 lg:grid-cols-[minmax(0,1fr)_minmax(220px,48ch)] lg:items-end">
      <div className="grid gap-2 md:grid-cols-[104px_140px_140px_minmax(180px,240px)_max-content] md:items-end">
        <FieldShell>
          <span>Device</span>
          <FieldSelect
            aria-label="Career Mode device"
            disabled={disabled}
            value={runnerDevice}
            onChange={(event) => onRunnerDeviceChange(event.currentTarget.value as WatchDevice)}
          >
            <option value="cuda">cuda</option>
            <option value="cpu">cpu</option>
          </FieldSelect>
        </FieldShell>
        <FieldShell>
          <span>Renderer</span>
          <FieldSelect
            aria-label="Career Mode renderer"
            disabled={disabled}
            value={runnerRenderer}
            onChange={(event) => onRunnerRendererChange(event.currentTarget.value as WatchRenderer)}
          >
            {rendererOptions.map((renderer) => (
              <option key={renderer} value={renderer}>
                {renderer}
              </option>
            ))}
          </FieldSelect>
        </FieldShell>
        <FieldShell>
          <span>Mode</span>
          <FieldSelect
            aria-label="Career Mode initial policy mode"
            disabled={disabled}
            value={policyMode}
            onChange={(event) =>
              onPolicyModeChange(event.currentTarget.value as PolicyPlaybackMode)
            }
          >
            <option value="deterministic">deterministic</option>
            <option value="stochastic">stochastic</option>
          </FieldSelect>
        </FieldShell>
        <FieldShell>
          <span>Runtime seed</span>
          <span className="grid grid-cols-[minmax(0,1fr)_auto] gap-2">
            <FieldInput
              aria-label="Career Mode runtime seed"
              disabled={disabled}
              inputMode="numeric"
              value={attemptSeedText}
              onChange={(event) => onAttemptSeedChange(event.currentTarget.value)}
            />
            <TooltipIconButton
              aria-label="Randomize runtime seed"
              disabled={disabled}
              tooltip="Randomize runtime seed"
              onClick={onRandomizeAttemptSeed}
            >
              <RandomizeIcon />
            </TooltipIconButton>
          </span>
        </FieldShell>
        <Button
          className="w-fit gap-2 px-5"
          disabled={!canStart || starting}
          type="button"
          variant="primary"
          onClick={onStart}
        >
          <PlayIcon />
          <span>{starting ? "Opening" : startLabel}</span>
        </Button>
      </div>
      <span className="text-xs text-app-muted lg:pb-3">{startNote}</span>
    </div>
  );
}
