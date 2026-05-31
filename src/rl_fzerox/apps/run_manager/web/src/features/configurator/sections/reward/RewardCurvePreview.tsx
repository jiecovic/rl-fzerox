// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/reward/RewardCurvePreview.tsx
export interface RewardCurvePreviewPoint {
  label?: string;
  xValue: number;
  yValue: number;
}

interface RewardCurvePreviewProps {
  points: readonly RewardCurvePreviewPoint[];
  title: string;
  xAxisLabel: string;
  ySuffix?: string;
}

export function RewardCurvePreview({
  points,
  title,
  xAxisLabel,
  ySuffix = "x",
}: RewardCurvePreviewProps) {
  if (points.length === 0) {
    return null;
  }
  const xMin = Math.min(...points.map((point) => point.xValue), 0);
  const xMax = Math.max(...points.map((point) => point.xValue), 1);
  const yValues = points.map((point) => point.yValue);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  const axisMinY = Math.min(yMin, 0);
  const axisMaxY = Math.max(yMax, 1);
  const xSpan = Math.max(xMax - xMin, 1e-9);
  const ySpan = Math.max(axisMaxY - axisMinY, 1e-9);
  const viewBox = {
    height: 190,
    width: 900,
  };
  const plot = {
    bottom: 138,
    left: 70,
    right: 830,
    top: 24,
  };
  const plotWidth = plot.right - plot.left;
  const plotHeight = plot.bottom - plot.top;
  const plottedPoints = points.map((point) => ({
    ...point,
    x: plot.left + ((point.xValue - xMin) / xSpan) * plotWidth,
    y: plot.bottom - ((point.yValue - axisMinY) / ySpan) * plotHeight,
  }));
  const svgPoints = plottedPoints
    .map((point) => {
      return `${point.x.toFixed(2)},${point.y.toFixed(2)}`;
    })
    .join(" ");
  const plottedTickPoints = plottedPoints.filter((point) => point.label !== undefined);
  const yTickPoints = visibleYTickPoints(plottedTickPoints, { yMax, yMin });

  return (
    <div className="reward-curve-preview">
      <div className="reward-curve-preview__header">
        <span>{title}</span>
        <span>
          {formatPreviewNumber(yMin)}
          {ySuffix} - {formatPreviewNumber(yMax)}
          {ySuffix}
        </span>
      </div>
      <svg aria-hidden="true" viewBox={`0 0 ${viewBox.width} ${viewBox.height}`}>
        {plottedTickPoints.map((point) => (
          <g key={`guide-${point.xValue}-${point.yValue}`}>
            <line
              className="reward-curve-preview__guide"
              x1={point.x}
              x2={point.x}
              y1={point.y}
              y2={plot.bottom}
            />
            <line
              className="reward-curve-preview__guide"
              x1={plot.left}
              x2={point.x}
              y1={point.y}
              y2={point.y}
            />
          </g>
        ))}
        <line x1={plot.left} x2={plot.right} y1={plot.bottom} y2={plot.bottom} />
        <line x1={plot.left} x2={plot.left} y1={plot.top} y2={plot.bottom} />
        {yTickPoints.map((point) => (
          <g key={`y-${point.yValue}`}>
            <line
              className="reward-curve-preview__tick"
              x1={plot.left - 7}
              x2={plot.left}
              y1={point.y}
              y2={point.y}
            />
            <text textAnchor="end" x={plot.left - 12} y={point.y + 3}>
              {formatPreviewNumber(point.yValue)}
              {ySuffix}
            </text>
          </g>
        ))}
        {plottedTickPoints.map((point, index) => (
          <g key={`x-${point.xValue}-${point.yValue}`}>
            <line
              className="reward-curve-preview__tick"
              x1={point.x}
              x2={point.x}
              y1={plot.bottom}
              y2={plot.bottom + 6}
            />
            <text
              textAnchor={axisTickAnchor(index, plottedTickPoints.length)}
              x={point.x}
              y={plot.bottom + 28}
            >
              {point.label}
            </text>
          </g>
        ))}
        <text
          className="reward-curve-preview__axis-label"
          textAnchor="middle"
          x={(plot.left + plot.right) / 2}
          y={viewBox.height - 8}
        >
          {xAxisLabel}
        </text>
        <polyline points={svgPoints} />
        {plottedTickPoints.map((point) => (
          <circle key={`point-${point.xValue}-${point.yValue}`} cx={point.x} cy={point.y} r="3" />
        ))}
      </svg>
    </div>
  );
}

interface PlottedRewardCurvePreviewPoint extends RewardCurvePreviewPoint {
  x: number;
  y: number;
}

function visibleYTickPoints(
  points: readonly PlottedRewardCurvePreviewPoint[],
  limits: { yMax: number; yMin: number },
) {
  const minSpacing = 15;
  const uniquePoints = points.filter((point, index, values) => {
    return (
      values.findIndex((otherPoint) => {
        return Math.abs(otherPoint.yValue - point.yValue) < 1e-9;
      }) === index
    );
  });
  const selectedPoints: PlottedRewardCurvePreviewPoint[] = [];
  for (const point of [...uniquePoints].sort((left, right) => {
    return yTickPriority(left, limits) - yTickPriority(right, limits);
  })) {
    if (
      selectedPoints.every((selectedPoint) => Math.abs(selectedPoint.y - point.y) >= minSpacing)
    ) {
      selectedPoints.push(point);
    }
  }
  return selectedPoints.sort((left, right) => left.y - right.y);
}

function yTickPriority(
  point: PlottedRewardCurvePreviewPoint,
  limits: { yMax: number; yMin: number },
) {
  if (Math.abs(point.yValue - limits.yMin) < 1e-9 || Math.abs(point.yValue - limits.yMax) < 1e-9) {
    return 0;
  }
  if (Math.abs(point.yValue - 1) < 1e-9) {
    return 1;
  }
  return 2;
}

function axisTickAnchor(index: number, count: number) {
  if (index === 0) {
    return "start";
  }
  if (index === count - 1) {
    return "end";
  }
  return "middle";
}

function formatPreviewNumber(value: number) {
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}
