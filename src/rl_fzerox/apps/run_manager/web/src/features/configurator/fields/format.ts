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

export function roundSignificant(value: number, digits: number) {
  return Number(value.toPrecision(digits));
}

export function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}
