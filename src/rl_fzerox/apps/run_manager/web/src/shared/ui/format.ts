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
