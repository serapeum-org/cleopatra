"""Categorical bar-chart visualization.

Provides `BarGlyph` for drawing a bar per category from a `(categories,
values)` pair — class areas, per-label counts, and other categorical
summaries. Each bar is coloured from the shared palette system (see
`cleopatra.styling.palettes`), and the glyph reuses the shared `Glyph`
axis-styling pipeline (title, axis labels, tick sizes) and the disjoint
swatch legend.

`BarGlyph` is the *categorical* counterpart to `LineGlyph.bar`, which draws
bars at numeric `x` positions; here the positions are nominal category labels.

Examples:
    - A three-category bar chart:
        ```python
        >>> from cleopatra.glyphs.stats.bar_glyph import BarGlyph
        >>> glyph = BarGlyph(["water", "forest", "urban"], [12.0, 55.0, 33.0])
        >>> fig, ax, bars = glyph.plot(ylabel="area (km²)")
        >>> len(bars)
        3

        ```
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from cleopatra.glyphs.base.glyph import (
    Glyph,
    _clear_prior_render_artists,
    _mark_render_artists,
    _root_figure,
)
from cleopatra.glyphs.stats._categorical import (
    resolve_categories,
    resolve_category_colors,
    resolve_series,
    resolve_series_colors,
)
from cleopatra.styling.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styling.styles import disjoint_legend

#: The multi-series layouts a 2-D `values` block can be drawn as.
_MULTI_SERIES_MODES = ("stacked", "grouped")

#: Option keys for `BarGlyph`, on top of the shared styling defaults.
#: `cmap` overrides the shared diverging default (`coolwarm_r`) with a
#: sequential one: it colours the *series* of a multi-series chart, sampled
#: evenly across its range, and a diverging map's near-white centre made the
#: middle of an odd series count invisible.
BAR_DEFAULT_OPTIONS = STYLE_DEFAULTS | {
    "palette": "category12",
    "add_legend": False,
    "horizontal": False,
    "width": 0.8,
    "alpha": 1.0,
    "cmap": "viridis",
}


class BarGlyph(Glyph):
    """Visualization class for a categorical bar chart.

    Draws one bar per category from a `(categories, values)` pair, coloured
    from the shared palette system and styled through the shared `Glyph`
    pipeline (title, axis labels, tick sizes, optional legend).

    With a 2-D `values` block of shape `(n_categories, n_series)` and
    `mode="stacked"` or `mode="grouped"`, it instead draws a multi-series
    chart -- a count or total split by a second categorical dimension -- with
    one colour per series (sampled from `cmap`, or set explicitly). The 1-D
    single-series path is unchanged.

    Args:
        categories: Sequence of category labels (coerced to `str`).
        values: Bar heights. 1-D (one per category) for a single-series chart,
            or 2-D `(n_categories, n_series)` for a stacked / grouped
            multi-series chart.
        colors: Optional explicit colours. For a 1-D `values` these are
            per-category (length must equal the number of categories); for a
            2-D `values` they are per-series (length must equal the number of
            series). Default None draws single-series bars from the palette and
            multi-series bars from `cmap`.
        mode: Multi-series layout -- `"stacked"` or `"grouped"`. Required when
            `values` is 2-D; must be left `"single"` (the default) when
            `values` is 1-D.
        series_labels: Optional names for the series of a 2-D chart, used by the
            legend. Length must equal the number of series. Rejected (raises)
            for a 1-D chart, whose legend is labelled per category.
        ax: Pre-existing axes to draw on. Default None.
        fig: Pre-existing figure. Default None.
        **kwargs (Any): Construction-time overrides for `BAR_DEFAULT_OPTIONS`:
            `palette` (qualitative palette name for single-series bars, default
            `"category12"`), `cmap` (sequential colormap sampled per series for
            a multi-series chart, default `"viridis"`), `horizontal` (draw
            `barh` instead of `bar`,
            default False; composes with stacking and grouping), `add_legend`
            (draw a disjoint legend -- per category for 1-D, per series for 2-D,
            default False), `width` (bar thickness in data units along the
            category axis, default `0.8` -- applied as `Axes.bar`'s `width` when
            vertical and as `Axes.barh`'s `height` when `horizontal=True`, and
            for `mode="grouped"` it is the whole group's thickness, split among
            the series; there is no separate `height` keyword), `alpha` (bar
            opacity, default `1.0`), plus the shared styling options (`title`,
            `xlabel`, `ylabel`, their font sizes, `figsize`, ...).

    Raises:
        ValueError: If `values` is neither 1-D nor 2-D, if the lengths differ,
            if the inputs are empty (via `resolve_categories` /
            `resolve_series`), if `mode` and the `values` dimensionality
            disagree, if `series_labels` is given with a 1-D `values`, or if
            `series_labels` does not match the number of series.

    Examples:
        - Horizontal bars with explicit colours:
            ```python
            >>> from cleopatra.glyphs.stats.bar_glyph import BarGlyph
            >>> from matplotlib.colors import to_hex
            >>> glyph = BarGlyph(
            ...     ["a", "b"], [3.0, 7.0], colors=["#1f77b4", "#ff7f0e"],
            ...     horizontal=True,
            ... )
            >>> fig, ax, bars = glyph.plot()
            >>> to_hex(bars[0].get_facecolor())
            '#1f77b4'

            ```
        - A stacked two-series chart returns one bar container per series:
            ```python
            >>> from cleopatra.glyphs.stats.bar_glyph import BarGlyph
            >>> glyph = BarGlyph(
            ...     ["a", "b", "c"], [[1, 2], [3, 4], [5, 6]],
            ...     mode="stacked", series_labels=["x", "y"],
            ... )
            >>> fig, ax, bars = glyph.plot()
            >>> len(bars)
            2

            ```

    See Also:
        cleopatra.glyphs.primitives.line_glyph.LineGlyph.bar: Bars at numeric
            `x` positions (the non-categorical counterpart).
        cleopatra.glyphs.stats.pie_glyph.PieGlyph: The same data as a pie chart.
    """

    #: Option keys this glyph accepts (see `Glyph.option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = BAR_DEFAULT_OPTIONS

    def __init__(
        self,
        categories: Sequence,
        values: Any,
        *,
        colors: Sequence | None = None,
        mode: str = "single",
        series_labels: Sequence[str] | None = None,
        ax: Axes | None = None,
        fig: Figure | None = None,
        **kwargs,
    ):
        super().__init__(default_options=BAR_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs)
        self._multi = np.ndim(values) == 2
        self.mode = self._resolve_mode(mode)
        if self._multi:
            self.categories, self.values = resolve_series(categories, values)
            self.series_labels = self._resolve_series_labels(series_labels)
        else:
            if series_labels is not None:
                raise ValueError(
                    "series_labels applies only to a multi-series (2-D) chart; a "
                    "1-D values is single-series, whose legend is labelled per "
                    "category. Drop series_labels, or pass a 2-D values with "
                    "mode='stacked'/'grouped'."
                )
            self.categories, self.values = resolve_categories(categories, values)
            self.series_labels = None
        self._colors = colors
        #: The bar artists from the most recent `plot` call: a single
        #: `BarContainer` for a 1-D chart, a list of them (one per series) for a
        #: multi-series chart, or `None` before the first `plot`.
        self.bars: BarContainer | list[BarContainer] | None = None

    def _resolve_mode(self, mode: str) -> str:
        """Validate `mode` against the `values` dimensionality.

        Args:
            mode: The requested layout (`"single"`, `"stacked"` or `"grouped"`).

        Returns:
            str: The validated mode.

        Raises:
            ValueError: If a 2-D `values` was given without a multi-series mode,
                if a multi-series mode was given with a 1-D `values`, or if
                `mode` is not one of the three known values.
        """
        if mode not in ("single", *_MULTI_SERIES_MODES):
            raise ValueError(
                f"mode must be 'single', 'stacked' or 'grouped', got {mode!r}."
            )
        if self._multi and mode == "single":
            raise ValueError(
                "a 2-D values block needs mode='stacked' or mode='grouped'; "
                "pass one to say how the series should be laid out."
            )
        if not self._multi and mode in _MULTI_SERIES_MODES:
            raise ValueError(
                f"mode={mode!r} needs a 2-D (n_categories, n_series) values "
                "block; a 1-D values is single-series (mode='single')."
            )
        return mode

    def _resolve_series_labels(self, series_labels: Sequence[str] | None) -> list[str]:
        """Return one label per series, defaulting to `series 1`, `series 2`, ...

        Args:
            series_labels: Caller-supplied series names, or `None`.

        Returns:
            list[str]: Exactly `n_series` labels.

        Raises:
            ValueError: If `series_labels` is given and its length is not the
                number of series.
        """
        n_series = self.values.shape[1]
        if series_labels is None:
            return [f"series {j + 1}" for j in range(n_series)]
        labels = [str(label) for label in series_labels]
        if len(labels) != n_series:
            raise ValueError(
                f"series_labels ({len(labels)}) must match the number of series "
                f"({n_series})."
            )
        return labels

    def plot(
        self,
        ax: Axes | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        colors: Sequence | None = None,
        add_legend: bool | None = None,
        compose: bool = False,
    ) -> tuple[Figure, Axes, BarContainer | list[BarContainer]]:
        """Draw the categorical bar chart.

        Args:
            ax: Axes to draw on. Falls back to the axes supplied at
                construction, otherwise a new figure/axes is created.
            title: Plot title. Overrides `default_options["title"]` when given.
            xlabel: X-axis label for this call. Overrides the construction value.
            ylabel: Y-axis label for this call. Overrides the construction value.
            colors: Colours for this call. For a single-series chart these are
                per-category (length must equal the number of categories); for a
                multi-series chart they are per-series. Defaults to the colours
                given at construction, else the palette (single) / `cmap`
                (multi).
            add_legend: Draw a disjoint legend -- per category for a 1-D chart,
                per series for a multi-series chart. Defaults to None, which
                keeps the `add_legend` option set at construction.
            compose: Draw *over* whatever is already on the axes instead of
                replacing it, by default `False`. `False` clears every glyph's
                prior render artists from the axes first (so a re-plot replaces
                rather than stacks); `True` clears only this glyph's own, so the
                bars layer onto a shared axes -- the same opt-in the other glyphs
                expose.

        Returns:
            tuple[Figure, Axes, BarContainer | list[BarContainer]]: The figure,
                the axes, and the bar artists -- a single `BarContainer` for a
                1-D chart, or a list of them (one per series) for a multi-series
                chart.

        Examples:
            - A vertical bar chart adds no legend by default:
                ```python
                >>> from cleopatra.glyphs.stats.bar_glyph import BarGlyph
                >>> fig, ax, bars = BarGlyph(["x", "y", "z"], [1.0, 2.0, 3.0]).plot()
                >>> ax.get_legend() is None
                True
                >>> [float(b.get_height()) for b in bars]
                [1.0, 2.0, 3.0]

                ```
        """
        if ax is not None:
            self.ax = ax
            self.fig = _root_figure(ax)
        elif self.ax is None:
            self.fig, self.ax = self.create_figure_axes()
        ax = self.ax
        opts = self.default_options

        if title is not None:
            opts["title"] = title
        call_keys = set()
        if xlabel is not None:
            opts["xlabel"] = xlabel
            call_keys.add("xlabel")
        if ylabel is not None:
            opts["ylabel"] = ylabel
            call_keys.add("ylabel")
        # Restore the construction-time axis labels the call did not override and
        # set `_render_explicit_options` (construction keys + this call's), so a
        # one-off `plot(xlabel=...)` does not leak into a later plain `plot()`.
        self._restore_construction_axis_style(call_keys)

        chosen = colors if colors is not None else self._colors
        _clear_prior_render_artists(ax, self, compose=compose)
        if self._multi:
            self.bars, legend_colors, legend_labels = self._draw_multi(ax, opts, chosen)
        else:
            self.bars, legend_colors, legend_labels = self._draw_single(
                ax, opts, chosen
            )

        if opts["title"]:
            ax.set_title(opts["title"], fontsize=opts["title_size"])
        self._apply_axis_style(ax)

        draw_legend = opts["add_legend"] if add_legend is None else add_legend
        if draw_legend:
            disjoint_legend(ax, legend_colors, legend_labels)

        assert self.fig is not None
        return self.fig, ax, self.bars

    def _draw_single(
        self, ax: Axes, opts: dict, chosen: Sequence | None
    ) -> tuple[BarContainer, list, list[str]]:
        """Draw the single-series (1-D) chart and return its legend inputs.

        Args:
            ax: The resolved axes to draw on.
            opts: The glyph's resolved options.
            chosen: Per-category colours for this call, or `None` for the
                palette.

        Returns:
            tuple[BarContainer, list, list[str]]: The drawn bar container, the
                per-category colours and the category labels (the latter two
                feed the optional legend).
        """
        palette_colors = resolve_category_colors(
            len(self.categories), chosen, opts["palette"]
        )
        if opts["horizontal"]:
            bars = ax.barh(
                self.categories,
                self.values,
                height=opts["width"],
                alpha=opts["alpha"],
                color=palette_colors,
            )
        else:
            bars = ax.bar(
                self.categories,
                self.values,
                width=opts["width"],
                alpha=opts["alpha"],
                color=palette_colors,
            )
        # Mark the BarContainer itself (not its unpacked Rectangles): its
        # remove() both detaches the child patches and deregisters the container
        # from ax.containers, so a re-plot leaves no stale/detached containers.
        _mark_render_artists(ax, self, bars)
        return bars, palette_colors, self.categories

    def _draw_multi(
        self, ax: Axes, opts: dict, chosen: Sequence | None
    ) -> tuple[list[BarContainer], list, list[str]]:
        """Draw the multi-series (2-D) chart and return its legend inputs.

        Stacked bars accumulate each series onto the running total of the ones
        before it; grouped bars split the `width` into `n_series` side-by-side
        sub-bars centred on each category. Both compose with `horizontal=True`
        (the `bottom`/`left` and `width`/`height` roles swap). Category ticks are
        placed explicitly, since the bars are drawn at numeric positions rather
        than from the string labels.

        Args:
            ax: The resolved axes to draw on.
            opts: The glyph's resolved options.
            chosen: Per-series colours for this call, or `None` to sample `cmap`.

        Returns:
            tuple[list[BarContainer], list, list[str]]: One bar container per
                series, the per-series colours and the series labels (the latter
                two feed the optional legend).
        """
        n_cat, n_series = self.values.shape
        series_colors = resolve_series_colors(n_series, chosen, opts["cmap"])
        positions = np.arange(n_cat, dtype=float)
        horizontal = opts["horizontal"]
        if self.mode == "stacked":
            thickness = opts["width"]
            offsets = [0.0] * n_series
            baselines = self._stacked_baselines()
        else:
            thickness = opts["width"] / n_series
            offsets = [(j - (n_series - 1) / 2) * thickness for j in range(n_series)]
            baselines = [np.zeros(n_cat) for _ in range(n_series)]

        containers: list[BarContainer] = []
        for j in range(n_series):
            seg = self.values[:, j]
            common = {"alpha": opts["alpha"], "color": series_colors[j]}
            if horizontal:
                container = ax.barh(
                    positions + offsets[j],
                    seg,
                    height=thickness,
                    left=baselines[j],
                    **common,
                )
            else:
                container = ax.bar(
                    positions + offsets[j],
                    seg,
                    width=thickness,
                    bottom=baselines[j],
                    **common,
                )
            containers.append(container)

        if horizontal:
            ax.set_yticks(positions)
            ax.set_yticklabels(self.categories)
        else:
            ax.set_xticks(positions)
            ax.set_xticklabels(self.categories)
        _mark_render_artists(ax, self, *containers)
        return containers, series_colors, self.series_labels

    def _stacked_baselines(self) -> list[np.ndarray]:
        """Return the running baseline (bottom/left) for each stacked series.

        Positive and negative segments accumulate on separate baselines, per
        category: a positive value rests on the running positive total and a
        negative value on the running negative total, so within one category the
        positive segments stack upward from zero and the negative ones stack
        downward from zero without overlapping. For all-positive (or
        all-negative) data this is the plain cumulative stack.

        Returns:
            list[np.ndarray]: One baseline array per series, in series order.
        """
        n_cat, n_series = self.values.shape
        baselines = []
        pos = np.zeros(n_cat)
        neg = np.zeros(n_cat)
        for j in range(n_series):
            seg = self.values[:, j]
            is_pos = seg >= 0
            baselines.append(np.where(is_pos, pos, neg))
            pos = pos + np.where(is_pos, seg, 0.0)
            neg = neg + np.where(is_pos, 0.0, seg)
        return baselines
