export function formatInteger(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

export function formatCompactNumber(value: number) {
  if (value >= 1000 && value % 1000 === 0) {
    return `${value / 1000}k`;
  }
  return String(value);
}

export function formatCompactDecimal(value: number) {
  if (value === 0) {
    return "0";
  }
  return Number.isInteger(value) ? String(value) : String(value).replace(/^0/, "");
}

export function formatEditableDecimal(value: number) {
  const normalized = roundSignificant(value, 12);
  if (!Number.isFinite(normalized) || Object.is(normalized, -0) || normalized === 0) {
    return "0";
  }
  if (Number.isInteger(normalized)) {
    return String(normalized);
  }
  return trimTrailingZeros(normalized.toFixed(12));
}

export function formatDecimalInput(value: number, step: number | string) {
  const fractionDigits = stepFractionDigits(step);
  if (fractionDigits === 0) {
    return String(Math.round(value));
  }
  return trimTrailingZeros(roundDecimal(value, fractionDigits).toFixed(fractionDigits));
}

export function roundToStepPrecision(value: number, step: number | string) {
  const fractionDigits = stepFractionDigits(step);
  return roundDecimal(value, fractionDigits);
}

export function roundSignificant(value: number, digits: number) {
  return Number(value.toPrecision(digits));
}

export function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function stepFractionDigits(step: number | string) {
  const stepText = typeof step === "number" ? String(step) : step;
  if (!stepText.includes(".")) {
    return 0;
  }
  return stepText.split(".")[1]?.length ?? 0;
}

function roundDecimal(value: number, fractionDigits: number) {
  const scale = 10 ** fractionDigits;
  return Math.round(value * scale) / scale;
}

function trimTrailingZeros(value: string) {
  return value.replace(/(\.\d*?[1-9])0+$/u, "$1").replace(/\.0+$/u, "");
}
