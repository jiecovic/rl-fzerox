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
  onRunnerDeviceChange,
  onRunnerRendererChange,
  onSaveSettings,
  onStart,
  policyMode,
  recordingEnabled,
  recordingInputHudEnabled,
  rendererOptions,
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
  onRunnerDeviceChange: (device: WatchDevice) => void;
  onRunnerRendererChange: (renderer: WatchRenderer) => void;
  onSaveSettings: () => void;
  onStart: () => void;
  policyMode: PolicyPlaybackMode;
  recordingEnabled: boolean;
  recordingInputHudEnabled: boolean;
  rendererOptions: readonly WatchRenderer[];
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
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
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
        <div className="grid gap-2 border-t border-app-border pt-3 md:grid-cols-[140px_160px] md:items-end">
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
        </div>
      </div>
      <span className="text-xs text-app-muted lg:pb-3">{startNote}</span>
    </div>
  );
}
