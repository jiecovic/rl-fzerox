export interface TabItem<T extends string> {
  id: T;
  label: string;
  closable?: boolean;
}

interface TabsProps<T extends string> {
  label: string;
  activeId: T;
  items: TabItem<T>[];
  variant?: "workspace" | "section";
  onSelect: (id: T) => void;
  onClose?: (id: T) => void;
}

export function Tabs<T extends string>({
  label,
  activeId,
  items,
  variant = "workspace",
  onSelect,
  onClose,
}: TabsProps<T>) {
  return (
    <nav
      aria-label={label}
      className={
        variant === "workspace" ? "tab-strip tab-strip-workspace" : "tab-strip tab-strip-section"
      }
    >
      {items.map((item) => (
        <span className={item.id === activeId ? "tab-shell active" : "tab-shell"} key={item.id}>
          <button className="tab-item" type="button" onClick={() => onSelect(item.id)}>
            {item.label}
          </button>
          {item.closable === true && onClose !== undefined ? (
            <button
              aria-label={`Close ${item.label}`}
              className="tab-close"
              type="button"
              onClick={() => onClose(item.id)}
            >
              <CloseIcon />
            </button>
          ) : null}
        </span>
      ))}
    </nav>
  );
}

function CloseIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 16 16" width="14">
      <path
        d="M4.25 4.25 8 8m0 0 3.75 3.75M8 8l3.75-3.75M8 8l-3.75 3.75"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}
