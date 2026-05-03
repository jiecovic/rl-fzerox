export function HelpHint({
  label,
  position,
  tooltip,
}: {
  label: string;
  position?: "left";
  tooltip: string;
}) {
  return (
    <button
      aria-label={`${label}: ${tooltip}`}
      className="field-help"
      data-tooltip={tooltip}
      data-tooltip-position={position}
      type="button"
    >
      ?
    </button>
  );
}
