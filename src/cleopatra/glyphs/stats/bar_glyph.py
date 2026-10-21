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

from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from cleopatra.glyphs.base.glyph import Glyph, _root_figure
from cleopatra.glyphs.stats._categorical import (
    resolve_categories,
    resolve_category_colors,
)
from cleopatra.styling.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styling.styles import disjoint_legend

#: Option keys for `BarGlyph`, on top of the shared styling defaults.
BAR_DEFAULT_OPTIONS = STYLE_DEFAULTS | {
    "palette": "category12",
    "add_legend": False,
    "horizontal": False,
}


class BarGlyph(Glyph):
    """Visualization class for a categorical bar chart.

    Draws one bar per category from a `(categories, values)` pair, coloured
    from the shared palette system and styled through the shared `Glyph`
    pipeline (title, axis labels, tick sizes, optional legend).

    Args:
        categories: Sequence of category labels (coerced to `str`).
        values: 1-D sequence/array of bar heights, one per category.
        colors: Optional explicit per-category colours (length must equal the
            number of categories). Default None draws from the palette.
        ax: Pre-existing axes to draw on. Default None.
        fig: Pre-existing figure. Default None.
        **kwargs: Construction-time overrides for `BAR_DEFAULT_OPTIONS`:
            `palette` (qualitative palette name, default `"category12"`),
            `horizontal` (draw `barh` instead of `bar`, default False),
            `add_legend` (draw a disjoint category->colour legend, default
            False), plus the shared styling options (`title`, `xlabel`,
            `ylabel`, their font sizes, `figsize`, ...).

    Raises:
        ValueError: If `values` is not 1-D, if the lengths differ, or if the
            inputs are empty (via `resolve_categories`).

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
        ax: Axes | None = None,
        fig: Figure | None = None,
        **kwargs,
    ):
        super().__init__(default_options=BAR_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs)
        self.categories, self.values = resolve_categories(categories, values)
        self._colors = colors
        #: The `BarContainer` from the most recent `plot` call; `None` before.
        self.bars: BarContainer | None = None

    def plot(
        self,
        ax: Axes | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        colors: Sequence | None = None,
        add_legend: bool | None = None,
    ) -> tuple[Figure, Axes, BarContainer]:
        """Draw the categorical bar chart.

        Args:
            ax: Axes to draw on. Falls back to the axes supplied at
                construction, otherwise a new figure/axes is created.
            title: Plot title. Overrides `default_options["title"]` when given.
            xlabel: X-axis label for this call. Overrides the construction value.
            ylabel: Y-axis label for this call. Overrides the construction value.
            colors: Per-category colours for this call (length must equal the
                number of categories). Defaults to the colours given at
                construction, else the palette.
            add_legend: Draw a disjoint category->colour legend. Defaults to
                None, which keeps the `add_legend` option set at construction.

        Returns:
            tuple[Figure, Axes, BarContainer]: The figure, the axes, and the
                bar container (one artist per category).

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

        explicit = set(getattr(self, "_explicit_options", set()))
        if title is not None:
            opts["title"] = title
        if xlabel is not None:
            opts["xlabel"] = xlabel
            explicit.add("xlabel")
        if ylabel is not None:
            opts["ylabel"] = ylabel
            explicit.add("ylabel")
        self._render_explicit_options = explicit

        chosen = colors if colors is not None else self._colors
        palette_colors = resolve_category_colors(
            len(self.categories), chosen, opts["palette"]
        )
        if opts["horizontal"]:
            self.bars = ax.barh(self.categories, self.values, color=palette_colors)
        else:
            self.bars = ax.bar(self.categories, self.values, color=palette_colors)

        if opts["title"]:
            ax.set_title(opts["title"], fontsize=opts["title_size"])
        self._apply_axis_style(ax)

        draw_legend = opts["add_legend"] if add_legend is None else add_legend
        if draw_legend:
            disjoint_legend(ax, palette_colors, self.categories)

        return self.fig, ax, self.bars
