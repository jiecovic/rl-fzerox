// src/rl_fzerox/apps/run_manager/web/src/widgets/saveGameWorkspace/runnerSeed.ts
export function parseAttemptSeed(value: string): string | "invalid" | null {
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    return null;
  }
  if (!/^\d+$/.test(trimmed)) {
    return "invalid";
  }
  const parsed = Number(trimmed);
  return Number.isSafeInteger(parsed) && parsed >= 0 && parsed <= 0xffffffff
    ? String(parsed)
    : "invalid";
}

export function randomAttemptSeedText(): string {
  const values = new Uint32Array(1);
  crypto.getRandomValues(values);
  return String(values[0]);
}
