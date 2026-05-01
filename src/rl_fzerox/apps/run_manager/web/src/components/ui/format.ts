export function formatDate(value: string) {
  return value.replace("T", " ").replace("+00:00", " UTC");
}
