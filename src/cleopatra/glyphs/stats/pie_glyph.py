"""Categorical pie-chart visualization.

Provides `PieGlyph` for drawing a pie chart from a `(categories, values)`
pair — the proportion each category contributes to the whole. Each wedge is
coloured from the shared palette system (see `cleopatra.styling.palettes`),
and the glyph reuses the shared title styling and the disjoint swatch legend.

Examples:
    - A three-category pie chart:
        ```python
        >>> from cleopatra.glyphs.stats.pie_glyph import PieGlyph
        >>> glyph = PieGlyph(["water", "forest", "urban"], [12.0, 55.0, 33.0])
        >>> fig, ax, wedges = glyph.plot()
        >>> len(wedges)
        3

        ```
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

from cleopatra.glyphs.base.glyph import (
    Glyph,
    _clear_prior_render_artists,
    _mark_render_artists,
    _root_figure,
)
from cleopatra.glyphs.stats._categorical import (
    resolve_categories,
    resolve_category_colors,
)
from cleopatra.styling.params import _UNSET, _Unset
from cleopatra.styling.styles import DEFAULT_OPTIONS as STYLE_DEFAULTS
from cleopatra.styling.styles import disjoint_legend

#: Option keys for `PieGlyph`, on top of the shared styling defaults.
PIE_DEFAULT_OPTIONS = STYLE_DEFAULTS | {
    "palette": "category12",
    "add_legend": False,
    "autopct": "%1.1f%%",
}


class PieGlyph(Glyph):
    """Visualization class for a categorical pie chart.

    Draws one wedge per category from a `(categories, values)` pair, sized by
    each value's share of the total and coloured from the shared palette
    system. Negative or all-zero values have no meaningful pie representation
    and are rejected.

    Args:
        categories: Sequence of category labels (coerced to `str`).
        values: 1-D sequence/array of wedge sizes, one per category.
        colors: Optional explicit per-category colours (length must equal the
            number of categories). Default None draws from the palette.
        ax: Pre-existing axes to draw on. Default None.
        fig: Pre-existing figure. Default None.
        **kwargs: Construction-time overrides for `PIE_DEFAULT_OPTIONS`:
            `palette` (qualitative palette name, default `"category12"`),
            `autopct` (per-wedge percentage format, default `"%1.1f%%"`; set
            None to hide), `add_legend` (draw a disjoint category->colour
            legend instead of wedge labels, default False), plus `title` and
            `figsize`. The cartesian axis-styling options inherited from the
            shared defaults (`xlabel`, `ylabel`, the tick-size keys, `grid_alpha`)
            are accepted but **inert** -- a pie has no x/y axes to label.

    Raises:
        ValueError: If `values` is not 1-D, if the lengths differ, if the
            inputs are empty (via `resolve_categories`), if any value is
            negative, or if the values sum to zero.

    Examples:
        - A pie with a category->colour legend instead of wedge labels:
            ```python
            >>> from cleopatra.glyphs.stats.pie_glyph import PieGlyph
            >>> glyph = PieGlyph(["a", "b"], [1.0, 3.0], add_legend=True)
            >>> fig, ax, wedges = glyph.plot()
            >>> ax.get_legend() is not None
            True

            ```

    See Also:
        cleopatra.glyphs.stats.bar_glyph.BarGlyph: The same data as a bar chart.
    """

    #: Option keys this glyph accepts (see `Glyph.option_keys`/`filter_kwargs`).
    DEFAULT_OPTIONS = PIE_DEFAULT_OPTIONS

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
        super().__init__(default_options=PIE_DEFAULT_OPTIONS, fig=fig, ax=ax, **kwargs)
        self.categories, self.values = resolve_categories(categories, values)
        if (self.values < 0).any():
            raise ValueError("pie values must be non-negative.")
        # Values are finite (resolve_categories) and non-negative (above), so a
        # sum of 0 means every value is 0; test with <= to avoid a float ==.
        if float(self.values.sum()) <= 0.0:
            raise ValueError("pie values must not sum to zero.")
        self._colors = colors
        #: The list of wedge `Patch`es from the most recent `plot`; `None` before.
        self.wedges: list[Wedge] | None = None

    def plot(
        self,
        ax: Axes | None = None,
        title: str | None = None,
        colors: Sequence | None = None,
        autopct: str | None | _Unset = _UNSET,
        add_legend: bool | None = None,
        compose: bool = False,
    ) -> tuple[Figure, Axes, list[Wedge]]:
        """Draw the categorical pie chart.

        Args:
            ax: Axes to draw on. Falls back to the axes supplied at
                construction, otherwise a new figure/axes is created.
            title: Plot title. Overrides `default_options["title"]` when given.
            colors: Per-category colours for this call (length must equal the
                number of categories). Defaults to the colours given at
                construction, else the palette.
            autopct: Per-wedge percentage format for this call. Defaults to the
                `autopct` option set at construction; pass `None` to hide the
                percentage labels, or a format string (e.g. `"%1.0f%%"`).
            add_legend: Draw a disjoint category->colour legend instead of
                labelling the wedges. Defaults to None, which keeps the
                `add_legend` option set at construction.
            compose: Draw *over* whatever is already on the axes instead of
                replacing it, by default `False`. `False` clears every glyph's
                prior render artists first (so a re-plot replaces rather than
                stacks the wedges/labels); `True` clears only this glyph's own,
                so the pie layers onto a shared axes.

        Returns:
            tuple[Figure, Axes, list[Wedge]]: The figure, the axes, and the
                list of wedge patches (one per category).

        Examples:
            - Equal values give equal-angle wedges:
                ```python
                >>> from cleopatra.glyphs.stats.pie_glyph import PieGlyph
                >>> fig, ax, wedges = PieGlyph(["a", "b"], [1.0, 1.0]).plot()
                >>> round(float(wedges[0].theta2 - wedges[0].theta1), 1)
                180.0

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

        chosen = colors if colors is not None else self._colors
        palette_colors = resolve_category_colors(
            len(self.categories), chosen, opts["palette"]
        )
        pct = opts["autopct"] if autopct is _UNSET else autopct
        draw_legend = opts["add_legend"] if add_legend is None else add_legend
        # With a legend the wedges carry no inline labels (the legend names
        # them); without one, each wedge is labelled in place.
        labels = None if draw_legend else self.categories

        _clear_prior_render_artists(ax, self, compose=compose)
        rendered = ax.pie(
            self.values, labels=labels, colors=palette_colors, autopct=pct
        )
        self.wedges = rendered[0]
        # `ax.pie` returns (wedges, texts) or (wedges, texts, autotexts); mark
        # every artist it made so a re-plot clears the labels/percentages too,
        # not just the wedges.
        _mark_render_artists(ax, self, *(a for group in rendered for a in group))
        ax.set_aspect("equal")

        if opts["title"]:
            ax.set_title(opts["title"], fontsize=opts["title_size"])
        if draw_legend:
            disjoint_legend(ax, palette_colors, self.categories)

        return self.fig, ax, self.wedges
