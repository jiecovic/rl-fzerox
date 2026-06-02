# src/rl_fzerox/ui/watch/view/panels/visuals/live_chart/geometry.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.ui.watch.view.panels.visuals.live_chart.style import LIVE_CHART_STYLE
from rl_fzerox.ui.watch.view.screen.types import PygameRect


@dataclass(frozen=True, slots=True)
class _FloatTupleStats:
    minimum: float
    maximum: float
    average: float
    max_abs: float


@dataclass(slots=True)
class _FloatTupleStatsCacheEntry:
    values: tuple[float, ...]
    stats: _FloatTupleStats


@dataclass(slots=True)
class _FloatTupleStatsCache:
    max_entries: int
    entries: list[_FloatTupleStatsCacheEntry]

    def stats_for(self, values: tuple[float, ...]) -> _FloatTupleStats:
        for entry in reversed(self.entries):
            if entry.values is values:
                return entry.stats
        stats = _compute_float_tuple_stats(values)
        self.entries.append(_FloatTupleStatsCacheEntry(values=values, stats=stats))
        del self.entries[: max(0, len(self.entries) - self.max_entries)]
        return stats


@dataclass(frozen=True, slots=True)
class _PlotGeometry:
    rect_x: int
    rect_y: int
    rect_width: int
    rect_height: int
    x_start: int
    x_end: int
    y_min: float
    span: float
    bucket_count: int | None


@dataclass(slots=True)
class _PlotPointsCacheEntry:
    x_values: tuple[int, ...]
    y_values: tuple[float, ...]
    geometry: _PlotGeometry
    points: tuple[tuple[int, int], ...]


@dataclass(slots=True)
class _PlotPointsCache:
    max_entries: int
    entries: list[_PlotPointsCacheEntry]

    def points_for(
        self,
        *,
        x_values: tuple[int, ...],
        y_values: tuple[float, ...],
        geometry: _PlotGeometry,
    ) -> tuple[tuple[int, int], ...] | None:
        for entry in reversed(self.entries):
            if (
                entry.x_values is x_values
                and entry.y_values is y_values
                and entry.geometry == geometry
            ):
                return entry.points
        return None

    def store(
        self,
        *,
        x_values: tuple[int, ...],
        y_values: tuple[float, ...],
        geometry: _PlotGeometry,
        points: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        self.entries.append(
            _PlotPointsCacheEntry(
                x_values=x_values,
                y_values=y_values,
                geometry=geometry,
                points=points,
            )
        )
        del self.entries[: max(0, len(self.entries) - self.max_entries)]
        return points


_FLOAT_TUPLE_STATS_CACHE = _FloatTupleStatsCache(max_entries=96, entries=[])
_PLOT_POINTS_CACHE = _PlotPointsCache(max_entries=96, entries=[])


def _compute_float_tuple_stats(values: tuple[float, ...]) -> _FloatTupleStats:
    if not values:
        return _FloatTupleStats(
            minimum=0.0,
            maximum=0.0,
            average=0.0,
            max_abs=0.0,
        )
    minimum = min(values)
    maximum = max(values)
    return _FloatTupleStats(
        minimum=minimum,
        maximum=maximum,
        average=sum(values) / len(values),
        max_abs=max(abs(value) for value in values),
    )


def _float_tuple_stats(values: tuple[float, ...]) -> _FloatTupleStats:
    return _FLOAT_TUPLE_STATS_CACHE.stats_for(values)


def _plot_points(
    *,
    x_values: tuple[int, ...],
    y_values: tuple[float, ...],
    rect: PygameRect,
    x_start: int,
    x_end: int,
    y_min: float,
    span: float,
) -> tuple[tuple[int, int], ...]:
    point_count = min(len(x_values), len(y_values))
    bucket_count = _plot_bucket_count(point_count=point_count, rect_width=rect.width)
    geometry = _PlotGeometry(
        rect_x=rect.x,
        rect_y=rect.y,
        rect_width=rect.width,
        rect_height=rect.height,
        x_start=x_start,
        x_end=x_end,
        y_min=y_min,
        span=span,
        bucket_count=bucket_count,
    )
    cached_points = _PLOT_POINTS_CACHE.points_for(
        x_values=x_values,
        y_values=y_values,
        geometry=geometry,
    )
    if cached_points is not None:
        return cached_points

    if bucket_count is None:
        points = tuple(
            _plot_point(
                step=step,
                value=value,
                rect=rect,
                x_start=x_start,
                x_end=x_end,
                y_min=y_min,
                span=span,
                bucket_count=None,
            )
            for step, value in zip(x_values, y_values, strict=False)
        )
        return _PLOT_POINTS_CACHE.store(
            x_values=x_values,
            y_values=y_values,
            geometry=geometry,
            points=points,
        )

    bucketed: list[tuple[int, int]] = []
    bucket: _PlotPixelBucket | None = None
    for step, value in zip(x_values, y_values, strict=False):
        point_x, point_y = _plot_point(
            step=step,
            value=value,
            rect=rect,
            x_start=x_start,
            x_end=x_end,
            y_min=y_min,
            span=span,
            bucket_count=bucket_count,
        )
        if bucket is None:
            bucket = _PlotPixelBucket.from_point(point_x, point_y)
            continue
        if point_x != bucket.x:
            bucket.append_to(bucketed)
            bucket = _PlotPixelBucket.from_point(point_x, point_y)
            continue
        bucket.observe(point_y)

    if bucket is not None:
        bucket.append_to(bucketed)
    return _PLOT_POINTS_CACHE.store(
        x_values=x_values,
        y_values=y_values,
        geometry=geometry,
        points=tuple(bucketed),
    )


def _plot_bucket_count(*, point_count: int, rect_width: int) -> int | None:
    if point_count <= rect_width * 2:
        return None
    return max(1, min(rect_width, LIVE_CHART_STYLE.plot_max_buckets))


def _plot_point(
    *,
    step: int,
    value: float,
    rect: PygameRect,
    x_start: int,
    x_end: int,
    y_min: float,
    span: float,
    bucket_count: int | None,
) -> tuple[int, int]:
    return (
        _plot_x(
            step=step,
            rect=rect,
            x_start=x_start,
            x_end=x_end,
            bucket_count=bucket_count,
        ),
        rect.y + rect.height - 1 - int(round((value - y_min) / span * max(1, rect.height - 1))),
    )


def _plot_x(
    *,
    step: int,
    rect: PygameRect,
    x_start: int,
    x_end: int,
    bucket_count: int | None,
) -> int:
    ratio = (step - x_start) / (x_end - x_start)
    if bucket_count is None:
        return rect.x + int(round(ratio * max(1, rect.width - 1)))
    if bucket_count <= 1:
        return rect.x
    bucket_index = int(round(ratio * (bucket_count - 1)))
    return rect.x + int(round(bucket_index / (bucket_count - 1) * max(1, rect.width - 1)))


@dataclass(slots=True)
class _PlotPixelBucket:
    x: int
    first_y: int
    min_y: int
    max_y: int
    last_y: int

    @classmethod
    def from_point(cls, x: int, y: int) -> _PlotPixelBucket:
        return cls(
            x=x,
            first_y=y,
            min_y=y,
            max_y=y,
            last_y=y,
        )

    def observe(self, y: int) -> None:
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)
        self.last_y = y

    def append_to(self, points: list[tuple[int, int]]) -> None:
        y_values = (self.first_y, self.min_y, self.max_y, self.last_y)
        for y in y_values:
            point = (self.x, y)
            if points and points[-1] == point:
                continue
            points.append(point)
