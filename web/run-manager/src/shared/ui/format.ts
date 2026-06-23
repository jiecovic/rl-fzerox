// web/run-manager/src/shared/ui/format.ts
export function formatDate(value: string) {
  return value.replace("T", " ").replace("+00:00", " UTC");
}

export function formatRelativeTime(value: string, now: Date = new Date()) {
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) {
    return formatDate(value);
  }
  const diffMs = Math.max(0, now.getTime() - timestamp.getTime());
  const diffSeconds = Math.floor(diffMs / 1000);
  if (diffSeconds < 5) {
    return "just now";
  }
  if (diffSeconds < 60) {
    return `${diffSeconds}s ago`;
  }
  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) {
    return `${diffMinutes}m ago`;
  }
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) {
    return `${diffHours}h ago`;
  }
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) {
    return `${diffDays}d ago`;
  }
  return formatDate(value);
}

export function formatInteger(value: number) {
  return value.toLocaleString();
}

export function formatDecimal(
  value: number,
  options: {
    maximumFractionDigits?: number;
    minimumFractionDigits?: number;
  } = {},
) {
  if (Number.isInteger(value) && options.minimumFractionDigits === undefined) {
    return formatInteger(value);
  }
  return value.toLocaleString(undefined, {
    maximumFractionDigits: options.maximumFractionDigits ?? 2,
    minimumFractionDigits: options.minimumFractionDigits ?? 0,
  });
}

export function formatSignedDecimal(
  value: number,
  options: {
    maximumFractionDigits?: number;
    minimumFractionDigits?: number;
  } = {},
) {
  const formatted = formatDecimal(value, options);
  return value > 0 ? `+${formatted}` : formatted;
}

export function formatRatioPercent(
  value: number,
  options: {
    maximumFractionDigits?: number;
    minimumFractionDigits?: number;
    tinyPositiveLabel?: string;
    tinyPositiveThreshold?: number;
    nearFullLabel?: string;
    nearFullThreshold?: number;
  } = {},
) {
  const percent = value * 100;
  const tinyThreshold = options.tinyPositiveThreshold;
  if (
    tinyThreshold !== undefined &&
    percent > 0 &&
    percent < tinyThreshold &&
    options.tinyPositiveLabel !== undefined
  ) {
    return options.tinyPositiveLabel;
  }
  const nearFullThreshold = options.nearFullThreshold;
  if (
    nearFullThreshold !== undefined &&
    percent < 100 &&
    percent > nearFullThreshold &&
    options.nearFullLabel !== undefined
  ) {
    return options.nearFullLabel;
  }
  return `${formatDecimal(percent, {
    maximumFractionDigits: options.maximumFractionDigits ?? 1,
    minimumFractionDigits: options.minimumFractionDigits ?? 1,
  })}%`;
}

export function formatProbabilityPercent(value: number) {
  return formatRatioPercent(value, {
    maximumFractionDigits: 1,
    minimumFractionDigits: 1,
    nearFullLabel: ">99.9%",
    nearFullThreshold: 99.9,
    tinyPositiveLabel: "<0.1%",
    tinyPositiveThreshold: 0.1,
  });
}

export function formatDurationSeconds(
  value: number,
  options: {
    secondsFractionDigits?: number;
  } = {},
) {
  const secondsFractionDigits = options.secondsFractionDigits ?? 1;
  if (value < 60) {
    return `${value.toLocaleString(undefined, {
      maximumFractionDigits: secondsFractionDigits,
      minimumFractionDigits: secondsFractionDigits,
    })} s`;
  }
  const wholeMinutes = Math.floor(value / 60);
  const remainingSeconds = value - wholeMinutes * 60;
  if (Math.abs(remainingSeconds - Math.round(remainingSeconds)) < 1e-9) {
    return `${wholeMinutes}m ${Math.round(remainingSeconds)}s`;
  }
  return `${wholeMinutes}m ${remainingSeconds.toLocaleString(undefined, {
    maximumFractionDigits: 1,
    minimumFractionDigits: 1,
  })}s`;
}

export function formatLongDurationSeconds(value: number) {
  const totalSeconds = Math.max(0, Math.floor(value));
  const durationUnits = [
    { label: "y", seconds: 365 * 24 * 3600 },
    { label: "mo", seconds: 30 * 24 * 3600 },
    { label: "d", seconds: 24 * 3600 },
    { label: "h", seconds: 3600 },
    { label: "m", seconds: 60 },
    { label: "s", seconds: 1 },
  ] as const;

  let remainingSeconds = totalSeconds;
  const parts: string[] = [];
  for (const unit of durationUnits) {
    if (remainingSeconds < unit.seconds && parts.length === 0 && unit.label !== "s") {
      continue;
    }
    const amount = Math.floor(remainingSeconds / unit.seconds);
    remainingSeconds -= amount * unit.seconds;
    if (amount > 0 || unit.label === "s") {
      parts.push(`${amount}${unit.label}`);
    }
    if (parts.length >= 3) {
      break;
    }
  }
  return parts.join(" ");
}

export function formatRaceTimeMs(value: number) {
  const milliseconds = Math.max(0, Math.round(value));
  const minutes = Math.floor(milliseconds / 60_000);
  const seconds = Math.floor((milliseconds % 60_000) / 1_000);
  const millis = milliseconds % 1_000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

export function formatEtaSeconds(value: number) {
  const totalSeconds = Math.max(0, Math.round(value));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  }
  return `${seconds}s`;
}

export function formatCourseRunRate(value: number) {
  const formatted = value >= 10 ? value.toFixed(0) : value.toFixed(1);
  return `${formatted} runs/min`;
}

export function formatFrameRate(value: number) {
  return `${Math.round(value).toLocaleString()} fps`;
}

export function formatEnvStepRate(value: number) {
  return `${Math.round(value).toLocaleString()} env steps/s`;
}
