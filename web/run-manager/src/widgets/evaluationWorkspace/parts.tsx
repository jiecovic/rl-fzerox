// web/run-manager/src/widgets/evaluationWorkspace/parts.tsx
import { EvaluationTabIcon } from "@/shared/ui/icons";

export function Metric({
  detail,
  label,
  value,
}: {
  detail?: string;
  label: string;
  value: string;
}) {
  return (
    <div className="border border-app-border bg-app-surface p-3">
      <div className="text-xs font-bold tracking-[0.04em] text-app-muted uppercase">{label}</div>
      <div className="mt-2 text-lg font-semibold text-app-text">{value}</div>
      {detail === undefined ? null : <div className="mt-1 text-xs text-app-muted">{detail}</div>}
    </div>
  );
}

export function Detail({
  label,
  mono = false,
  value,
}: {
  label: string;
  mono?: boolean;
  value: string;
}) {
  return (
    <div className="grid gap-1">
      <dt className="text-xs font-bold tracking-[0.04em] text-app-muted uppercase">{label}</dt>
      <dd className={mono ? "m-0 break-all font-mono text-app-muted" : "m-0 text-app-muted"}>
        {value}
      </dd>
    </div>
  );
}

export function SectionTitle({ title }: { title: string }) {
  return (
    <div className="flex items-center gap-2 text-sm font-bold tracking-[0.04em] text-app-muted uppercase">
      <EvaluationTabIcon />
      <span>{title}</span>
    </div>
  );
}
