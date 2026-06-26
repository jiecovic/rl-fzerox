// web/run-manager/src/features/careerRunner/ui/RunnerControlPanel.tsx
import type { PolicyPlaybackMode, WatchDevice, WatchRenderer } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { ToggleSwitch } from "@/shared/ui/configFields";
import { FieldInput, FieldSelect, FieldShell } from "@/shared/ui/Field";
import { PlayIcon, RandomizeIcon, SaveDraftIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

export function RunnerControlPanel({
  attemptSeedText,
  canStart,
  canSaveSettings,
  onAttemptSeedChange,
  onRandomizeAttemptSeed,
  onPolicyModeChange,
  onRecordingEnabledChange,
  onRecordingInputHudEnabledChange,
  onRecordingUpscaleFactorChange,
  onRunnerDeviceChange,
  onRunnerRendererChange,
  onSaveSettings,
  onStart,
  policyMode,
  recordingEnabled,
  recordingInputHudEnabled,
  recordingUpscaleFactor,
  rendererOptions,
  runnerDeviceOptions,
  runnerDevice,
  runnerRenderer,
  startLabel,
  savingSettings,
  starting,
  startNote,
}: {
  attemptSeedText: string;
  canStart: boolean;
  canSaveSettings: boolean;
  onAttemptSeedChange: (attemptSeedText: string) => void;
  onRandomizeAttemptSeed: () => void;
  onPolicyModeChange: (policyMode: PolicyPlaybackMode) => void;
  onRecordingEnabledChange: (recordingEnabled: boolean) => void;
  onRecordingInputHudEnabledChange: (recordingInputHudEnabled: boolean) => void;
  onRecordingUpscaleFactorChange: (recordingUpscaleFactor: number) => void;
  onRunnerDeviceChange: (device: WatchDevice) => void;
  onRunnerRendererChange: (renderer: WatchRenderer) => void;
  onSaveSettings: () => void;
  onStart: () => void;
  policyMode: PolicyPlaybackMode;
  recordingEnabled: boolean;
  recordingInputHudEnabled: boolean;
  recordingUpscaleFactor: number;
  rendererOptions: readonly WatchRenderer[];
  runnerDeviceOptions: readonly WatchDevice[];
  runnerDevice: WatchDevice;
  runnerRenderer: WatchRenderer;
  startLabel: string;
  savingSettings: boolean;
  starting: boolean;
  startNote: string;
}) {
  const disabled = startLabel === "Running" || starting;
  const settingsDisabled = disabled || savingSettings;
  return (
    <div className="mb-5 grid gap-3 border border-app-border bg-app-surface px-3 py-3 xl:grid-cols-[minmax(0,1fr)_minmax(220px,48ch)] xl:items-end">
      <div className="grid gap-3">
        <div className="grid gap-2 md:grid-cols-[104px_140px_140px_minmax(180px,240px)_max-content_max-content] md:items-end">
          <FieldShell>
            <span>Device</span>
            <FieldSelect
              aria-label="Career Mode device"
              disabled={settingsDisabled}
              value={runnerDevice}
              onChange={(event) => onRunnerDeviceChange(event.currentTarget.value as WatchDevice)}
            >
              {runnerDeviceOptions.map((device) => (
                <option key={device} value={device}>
                  {device}
                </option>
              ))}
            </FieldSelect>
          </FieldShell>
          <FieldShell>
            <span>Renderer</span>
            <FieldSelect
              aria-label="Career Mode renderer"
              disabled={settingsDisabled}
              value={runnerRenderer}
              onChange={(event) =>
                onRunnerRendererChange(event.currentTarget.value as WatchRenderer)
              }
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
              disabled={settingsDisabled}
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
                disabled={settingsDisabled}
                inputMode="numeric"
                value={attemptSeedText}
                onChange={(event) => onAttemptSeedChange(event.currentTarget.value)}
              />
              <TooltipIconButton
                aria-label="Randomize runtime seed"
                disabled={settingsDisabled}
                tooltip="Randomize runtime seed"
                onClick={onRandomizeAttemptSeed}
              >
                <RandomizeIcon />
              </TooltipIconButton>
            </span>
          </FieldShell>
          <Button
            className="w-fit gap-2 px-5"
            disabled={!canSaveSettings || savingSettings}
            type="button"
            onClick={onSaveSettings}
          >
            <SaveDraftIcon />
            <span>{savingSettings ? "Saving" : "Save"}</span>
          </Button>
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
        <div className="grid gap-2 border-t border-app-border pt-3 md:grid-cols-[140px_150px_120px] md:items-end">
          <FieldShell>
            <span>Recording</span>
            <ToggleSwitch
              checked={recordingEnabled}
              disabled={settingsDisabled}
              label="Record video"
              onChange={onRecordingEnabledChange}
            />
          </FieldShell>
          <FieldShell>
            <span>Overlay</span>
            <ToggleSwitch
              checked={recordingInputHudEnabled}
              disabled={settingsDisabled || !recordingEnabled}
              label="Input HUD"
              onChange={onRecordingInputHudEnabledChange}
            />
          </FieldShell>
          <FieldShell>
            <span>Upscale</span>
            <FieldSelect
              aria-label="Career Mode recording upscale"
              disabled={settingsDisabled || !recordingEnabled}
              value={String(recordingUpscaleFactor)}
              onChange={(event) =>
                onRecordingUpscaleFactorChange(Number(event.currentTarget.value))
              }
            >
              {RECORDING_UPSCALE_OPTIONS.map((option) => (
                <option key={option.factor} value={option.factor}>
                  {option.label}
                </option>
              ))}
            </FieldSelect>
          </FieldShell>
        </div>
      </div>
      <span className="text-xs text-app-muted lg:pb-3">{startNote}</span>
    </div>
  );
}

const RECORDING_UPSCALE_OPTIONS = [
  { factor: 1, label: "1x native" },
  { factor: 2, label: "2x preview" },
] as const;
