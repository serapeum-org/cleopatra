"""Tests for cleopatra.glyphs.stats.bar_glyph.BarGlyph — issue #387.

Covers construction/validation, the drawn bar artists and heights, the
category -> colour mapping (palette default, cycling, explicit colours),
horizontal bars, the optional disjoint legend, title/label styling, and
drawing onto a caller-supplied axes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from cleopatra.glyphs.stats.bar_glyph import BarGlyph
from cleopatra.styling.palettes import get_palette


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


class TestBarGlyphConstruction:
    """Construction and input validation for BarGlyph."""

    def test_stores_labels_and_values(self):
        """Categories are coerced to str and values to a 1-D float array."""
        glyph = BarGlyph([1, 2, 3], [10, 20, 30])
        assert glyph.categories == ["1", "2", "3"], "labels should be strings"
        assert glyph.values.tolist() == [10.0, 20.0, 30.0], "values should be floats"

    def test_length_mismatch_raises(self):
        """A categories/values length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            BarGlyph(["a", "b"], [1.0])

    def test_empty_raises(self):
        """Empty inputs raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            BarGlyph([], [])

    def test_2d_values_without_mode_raise(self):
        """A 2-D values array with the default single mode raises ValueError."""
        with pytest.raises(ValueError, match="mode='stacked' or mode='grouped'"):
            BarGlyph(["a", "b"], [[1.0, 2.0], [3.0, 4.0]])

    def test_3d_values_raise(self):
        """A 3-D values array is rejected even with a multi-series mode."""
        with pytest.raises(ValueError, match="2-D"):
            BarGlyph(["a"], np.zeros((1, 2, 2)), mode="stacked")

    def test_non_finite_values_raise(self):
        """NaN or inf values are rejected at construction, not drawn silently."""
        with pytest.raises(ValueError, match="finite"):
            BarGlyph(["a", "b"], [1.0, np.nan])
        with pytest.raises(ValueError, match="finite"):
            BarGlyph(["a", "b"], [1.0, np.inf])


class TestBarGlyphPlot:
    """Rendering behaviour of BarGlyph.plot."""

    def test_returns_figure_axes_bars(self):
        """plot returns (Figure, Axes, BarContainer) with one bar per category."""
        fig, ax, bars = BarGlyph(["a", "b", "c"], [1.0, 2.0, 3.0]).plot()
        assert isinstance(fig, Figure), "first return should be a Figure"
        assert isinstance(bars, BarContainer), "third return should be a BarContainer"
        assert len(bars) == 3, "one bar per category"

    def test_bar_heights_match_values(self):
        """Each bar's height equals its category value."""
        _, _, bars = BarGlyph(["a", "b", "c"], [1.0, 2.0, 3.0]).plot()
        assert [float(b.get_height()) for b in bars] == [1.0, 2.0, 3.0]

    def test_duplicate_labels_draw_one_bar_per_value(self):
        """Duplicate category labels still draw one bar per value."""
        _, _, bars = BarGlyph(["a", "a", "b"], [1.0, 2.0, 3.0]).plot()
        assert len(bars) == 3, "one bar per value even when labels repeat"

    def test_single_category(self):
        """A single category draws a single bar."""
        _, _, bars = BarGlyph(["only"], [5.0]).plot()
        assert len(bars) == 1, "one bar for one category"
        assert float(bars[0].get_height()) == 5.0, "its height is the value"

    def test_default_palette_colours_distinct_categories(self):
        """With no explicit colours, bars take the palette swatches in order."""
        _, _, bars = BarGlyph(["a", "b", "c"], [1.0, 2.0, 3.0]).plot()
        swatches = list(get_palette("category12").colors)
        assert [to_hex(b.get_facecolor()) for b in bars] == [
            to_hex(swatches[i]) for i in range(3)
        ], "bars should use the first palette swatches in category order"

    def test_palette_cycles_when_more_categories_than_swatches(self):
        """More categories than palette swatches cycle the palette."""
        n = len(get_palette("category12").colors) + 2
        cats = [str(i) for i in range(n)]
        _, _, bars = BarGlyph(cats, list(range(1, n + 1))).plot()
        first = to_hex(bars[0].get_facecolor())
        wrapped = to_hex(bars[len(get_palette("category12").colors)].get_facecolor())
        assert first == wrapped, "the palette should wrap around for extra categories"

    def test_explicit_colours_used_verbatim(self):
        """Caller-supplied colours colour the bars in order."""
        _, _, bars = BarGlyph(
            ["a", "b"], [1.0, 2.0], colors=["#1f77b4", "#ff7f0e"]
        ).plot()
        assert [to_hex(b.get_facecolor()) for b in bars] == ["#1f77b4", "#ff7f0e"]

    def test_explicit_colours_length_mismatch_raises(self):
        """A colours list whose length differs from the categories raises."""
        glyph = BarGlyph(["a", "b"], [1.0, 2.0])
        with pytest.raises(ValueError, match="must match the number of categories"):
            glyph.plot(colors=["#f00"])

    def test_unknown_palette_raises_valueerror(self):
        """An unknown palette name raises ValueError, not a bare KeyError."""
        glyph = BarGlyph(["a", "b"], [1.0, 2.0], palette="does_not_exist")
        with pytest.raises(ValueError, match="unknown palette"):
            glyph.plot()

    def test_horizontal_draws_barh(self):
        """horizontal=True draws bars whose widths (not heights) carry the values."""
        _, _, bars = BarGlyph(["a", "b"], [3.0, 7.0], horizontal=True).plot()
        assert [float(b.get_width()) for b in bars] == [3.0, 7.0]

    def test_default_width_is_unchanged(self):
        """The default bar width stays matplotlib's 0.8, so existing charts are unchanged."""
        _, _, bars = BarGlyph(["a", "b"], [1.0, 2.0]).plot()
        assert [float(b.get_width()) for b in bars] == [0.8, 0.8]

    def test_width_applied_vertical(self):
        """width= is forwarded as the bar thickness on a vertical chart."""
        _, _, bars = BarGlyph(["a", "b"], [1.0, 2.0], width=20).plot()
        assert [float(b.get_width()) for b in bars] == [20.0, 20.0]

    def test_width_applied_horizontal(self):
        """width= is forwarded as the bar thickness (barh's height) when horizontal."""
        _, _, bars = BarGlyph(["a", "b"], [1.0, 2.0], horizontal=True, width=4).plot()
        assert [float(b.get_height()) for b in bars] == [4.0, 4.0]

    def test_alpha_applied(self):
        """alpha= is forwarded to the drawn bars."""
        _, _, bars = BarGlyph(["a", "b"], [1.0, 2.0], alpha=0.7).plot()
        assert all(float(b.get_alpha()) == 0.7 for b in bars)

    def test_width_on_datetime_like_categories(self):
        """width= still applies when categories are date-like (coerced to str labels)."""
        import datetime

        idx = [datetime.date(2000, m, 1) for m in range(1, 7)]
        _, _, bars = BarGlyph(idx, list(range(6)), width=0.5).plot()
        assert all(float(b.get_width()) == 0.5 for b in bars)

    def test_no_legend_by_default(self):
        """No legend is drawn unless requested."""
        _, ax, _ = BarGlyph(["a", "b"], [1.0, 2.0]).plot()
        assert ax.get_legend() is None, "default render draws no legend"

    def test_add_legend_draws_one_entry_per_category(self):
        """add_legend=True draws a disjoint legend naming every category."""
        _, ax, _ = BarGlyph(["a", "b", "c"], [1.0, 2.0, 3.0]).plot(add_legend=True)
        legend = ax.get_legend()
        assert legend is not None, "add_legend should draw a legend"
        assert [t.get_text() for t in legend.get_texts()] == ["a", "b", "c"]

    def test_title_and_labels_applied(self):
        """title/xlabel/ylabel passed to plot are set on the axes."""
        _, ax, _ = BarGlyph(["a", "b"], [1.0, 2.0]).plot(
            title="T", xlabel="X", ylabel="Y"
        )
        assert ax.get_title() == "T"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"

    def test_draws_on_supplied_axes(self):
        """A caller-supplied axes is drawn on and returned."""
        fig, ax = plt.subplots()
        out_fig, out_ax, _ = BarGlyph(["a", "b"], [1.0, 2.0]).plot(ax=ax)
        assert out_ax is ax, "the supplied axes should be used"
        assert out_fig is fig, "its figure should be returned"

    def test_replot_on_same_axes_replaces_not_stacks(self):
        """Re-plotting the same glyph clears its prior bars instead of stacking."""
        fig, ax = plt.subplots()
        glyph = BarGlyph(["a", "b", "c"], [1.0, 2.0, 3.0])
        glyph.plot(ax=ax)
        glyph.plot(ax=ax)
        assert len(ax.patches) == 3, "re-plot should replace, not stack, the bars"

    def test_default_replaces_other_glyph_on_shared_axes(self):
        """A second bar glyph with no compose clears the first glyph's bars."""
        fig, ax = plt.subplots()
        BarGlyph(["a", "b"], [1.0, 2.0]).plot(ax=ax)
        BarGlyph(["c", "d", "e"], [3.0, 4.0, 5.0]).plot(ax=ax)
        assert len(ax.patches) == 3, "default clears the prior glyph's bars"

    def test_compose_overlays_other_glyph_on_shared_axes(self):
        """compose=True keeps the other glyph's bars on a shared axes."""
        fig, ax = plt.subplots()
        BarGlyph(["a", "b"], [1.0, 2.0]).plot(ax=ax)
        BarGlyph(["c", "d"], [3.0, 4.0]).plot(ax=ax, compose=True)
        assert len(ax.patches) == 4, "compose should overlay, keeping both glyphs' bars"

    def test_replot_does_not_accumulate_stale_containers(self):
        """Re-plotting leaves no detached BarContainers in ax.containers."""
        fig, ax = plt.subplots()
        glyph = BarGlyph(["a", "b", "c"], [1.0, 2.0, 3.0])
        for _ in range(3):
            glyph.plot(ax=ax)
        assert len(ax.containers) == 1, "exactly one live bar container after re-plots"

    def test_uses_construction_axes_when_plot_ax_omitted(self):
        """An axes given at construction is used when plot() gets no ax."""
        fig, ax = plt.subplots()
        _, out_ax, _ = BarGlyph(["a", "b"], [1.0, 2.0], ax=ax).plot()
        assert out_ax is ax, "the construction-time axes should be reused"

    def test_plot_label_override_does_not_leak_into_later_plot(self):
        """A one-off plot-time xlabel reverts to the construction label next call."""
        glyph = BarGlyph(["a", "b"], [1.0, 2.0], xlabel="CX")
        _, ax1 = plt.subplots()
        glyph.plot(ax=ax1)
        assert ax1.get_xlabel() == "CX", "construction label on the first plot"
        _, ax2 = plt.subplots()
        glyph.plot(ax=ax2, xlabel="PX")
        assert ax2.get_xlabel() == "PX", "override applies for its own call"
        _, ax3 = plt.subplots()
        glyph.plot(ax=ax3)
        assert ax3.get_xlabel() == "CX", (
            "override must not persist into a later plain plot"
        )


class TestBarGlyphMultiSeries:
    """Stacked and grouped multi-series rendering (2-D values)."""

    VALUES = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_mode_required_for_2d(self):
        """A 2-D values with the default single mode is refused."""
        with pytest.raises(ValueError, match="mode='stacked' or mode='grouped'"):
            BarGlyph(["a", "b", "c"], self.VALUES)

    def test_multi_mode_rejected_for_1d(self):
        """A multi-series mode with a 1-D values is refused."""
        with pytest.raises(ValueError, match="needs a 2-D"):
            BarGlyph(["a", "b"], [1.0, 2.0], mode="grouped")

    def test_unknown_mode_raises(self):
        """An unknown mode is rejected with a clear message."""
        with pytest.raises(ValueError, match="must be 'single', 'stacked' or 'grouped'"):
            BarGlyph(["a", "b", "c"], self.VALUES, mode="piled")

    def test_row_count_must_match_categories(self):
        """A 2-D values whose rows differ from the categories is refused."""
        with pytest.raises(ValueError, match="number of value rows"):
            BarGlyph(["a", "b"], self.VALUES, mode="stacked")

    def test_stacked_returns_one_container_per_series(self):
        """A stacked chart returns one BarContainer per series."""
        _, _, bars = BarGlyph(["a", "b", "c"], self.VALUES, mode="stacked").plot()
        assert isinstance(bars, list), "multi-series returns a list of containers"
        assert len(bars) == 2, "one container per series"
        assert all(isinstance(c, BarContainer) for c in bars)

    def test_stacked_second_series_sits_on_the_first(self):
        """The second stacked series' bottom equals the first series' heights."""
        _, _, bars = BarGlyph(["a", "b", "c"], self.VALUES, mode="stacked").plot()
        assert [b.get_height() for b in bars[0]] == [1.0, 3.0, 5.0]
        assert [b.get_y() for b in bars[1]] == [1.0, 3.0, 5.0], (
            "series 1 rests on top of series 0"
        )

    def test_stacked_mixed_signs_do_not_overlap(self):
        """A negative series stacks downward from 0, not over the positive one."""
        values = np.array([[5.0, -3.0]])
        _, _, bars = BarGlyph(["a"], values, mode="stacked").plot()
        assert bars[0][0].get_y() == 0.0, "positive series 0 rests on zero"
        assert bars[1][0].get_y() == 0.0, (
            "negative series 1 must start at zero and grow down, not sit on series 0"
        )
        assert bars[1][0].get_height() == -3.0, "its height keeps the negative sign"

    def test_stacked_mixed_signs_accumulate_per_sign(self):
        """Successive same-sign segments stack on their own running baseline."""
        values = np.array([[2.0, 3.0, -1.0, -4.0]])
        _, _, bars = BarGlyph(["a"], values, mode="stacked").plot()
        bottoms = [b[0].get_y() for b in bars]
        assert bottoms == [0.0, 2.0, 0.0, -1.0], (
            f"positive segments stack from 0 upward, negatives from 0 down, got {bottoms}"
        )

    def test_grouped_splits_width_among_series(self):
        """A grouped chart splits the width into one sub-bar per series."""
        _, _, bars = BarGlyph(
            ["a", "b", "c"], self.VALUES, mode="grouped", width=0.8
        ).plot()
        assert float(bars[0][0].get_width()) == pytest.approx(0.4), "0.8 / 2 series"

    def test_grouped_sub_bars_are_offset_and_centred(self):
        """Grouped sub-bars straddle each integer category position."""
        _, _, bars = BarGlyph(
            ["a", "b", "c"], self.VALUES, mode="grouped", width=0.8
        ).plot()
        # Category 0: series 0 at -0.2, series 1 at +0.2; left edge = centre - 0.2.
        assert float(bars[0][0].get_x()) == pytest.approx(-0.4)
        assert float(bars[1][0].get_x()) == pytest.approx(0.0)

    def test_category_ticks_labelled(self):
        """Category labels are placed on the category axis."""
        _, ax, _ = BarGlyph(["a", "b", "c"], self.VALUES, mode="stacked").plot()
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]

    def test_horizontal_stacked_geometry(self):
        """Horizontal stacking puts the running baseline on the left, labels on y."""
        _, ax, bars = BarGlyph(
            ["a", "b", "c"], self.VALUES, mode="stacked", horizontal=True
        ).plot()
        assert [b.get_width() for b in bars[0]] == [1.0, 3.0, 5.0]
        assert [b.get_x() for b in bars[1]] == [1.0, 3.0, 5.0], "series 1 starts at left"
        assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]

    def test_horizontal_grouped_offsets(self):
        """Horizontal grouping offsets each series on the y axis, labels on y."""
        _, ax, bars = BarGlyph(
            ["a", "b", "c"], self.VALUES, mode="grouped", horizontal=True, width=0.8
        ).plot()
        assert float(bars[0][0].get_height()) == pytest.approx(0.4), "0.8 / 2 series"
        assert float(bars[0][0].get_y()) == pytest.approx(-0.4), "series 0 below centre"
        assert float(bars[1][0].get_y()) == pytest.approx(0.0), "series 1 above centre"
        assert [b.get_width() for b in bars[0]] == [1.0, 3.0, 5.0], "widths carry values"
        assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]

    def test_compose_layers_multi_series_glyphs(self):
        """compose=True keeps a prior multi-series glyph's bars on a shared axes."""
        fig, ax = plt.subplots()
        BarGlyph(["a", "b", "c"], self.VALUES, mode="grouped").plot(ax=ax)
        BarGlyph(["a", "b", "c"], self.VALUES, mode="stacked").plot(ax=ax, compose=True)
        assert len(ax.containers) == 4, "two grouped + two stacked containers layered"

    def test_explicit_series_colors(self):
        """Explicit colours apply one per series, not per category."""
        _, _, bars = BarGlyph(["a", "b", "c"], self.VALUES, mode="grouped").plot(
            colors=["#ff0000", "#00ff00"]
        )
        assert to_hex(bars[0][0].get_facecolor()) == "#ff0000"
        assert to_hex(bars[1][0].get_facecolor()) == "#00ff00"
        assert to_hex(bars[0][1].get_facecolor()) == "#ff0000", (
            "a series keeps one colour across categories"
        )

    def test_series_colors_length_mismatch_raises(self):
        """A per-series colour list of the wrong length is refused."""
        glyph = BarGlyph(["a", "b", "c"], self.VALUES, mode="stacked")
        with pytest.raises(ValueError, match="number of series"):
            glyph.plot(colors=["#f00"])

    def test_cmap_fallback_colours_the_series(self):
        """With no explicit colours, series are sampled from cmap, one per series."""
        _, _, bars = BarGlyph(
            ["a", "b", "c"], self.VALUES, mode="grouped", cmap="viridis"
        ).plot()
        c0 = to_hex(bars[0][0].get_facecolor())
        c1 = to_hex(bars[1][0].get_facecolor())
        assert c0 != c1, "distinct series take distinct colormap samples"

    def test_default_cmap_keeps_three_series_visible(self):
        """Three series under default options are mutually distinct and none near-white."""
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        _, _, bars = BarGlyph(["a", "b"], values, mode="grouped").plot()
        rgbs = [tuple(series[0].get_facecolor()[:3]) for series in bars]
        hexes = [to_hex(series[0].get_facecolor()) for series in bars]
        assert len(set(hexes)) == 3, f"series colours should be distinct, got {hexes}"
        for rgb, hex_ in zip(rgbs, hexes):
            assert not all(channel > 0.8 for channel in rgb), (
                f"the default colormap must not sample a near-white series, got {hex_}"
            )

    def test_add_legend_labels_the_series(self):
        """add_legend keys off the series_labels, one legend entry per series."""
        _, ax, _ = BarGlyph(
            ["a", "b", "c"], self.VALUES, mode="stacked", series_labels=["x", "y"]
        ).plot(add_legend=True)
        legend = ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["x", "y"]

    def test_series_labels_default_when_unnamed(self):
        """Without series_labels, the legend falls back to series 1, series 2."""
        _, ax, _ = BarGlyph(["a", "b", "c"], self.VALUES, mode="grouped").plot(
            add_legend=True
        )
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "series 1",
            "series 2",
        ]

    def test_series_labels_length_mismatch_raises(self):
        """series_labels whose length differs from the series count is refused."""
        with pytest.raises(ValueError, match="number of series"):
            BarGlyph(["a", "b", "c"], self.VALUES, mode="stacked", series_labels=["x"])

    def test_replot_multi_series_replaces_not_stacks(self):
        """Re-plotting a multi-series glyph clears its prior bars."""
        fig, ax = plt.subplots()
        glyph = BarGlyph(["a", "b", "c"], self.VALUES, mode="grouped")
        glyph.plot(ax=ax)
        glyph.plot(ax=ax)
        assert len(ax.containers) == 2, "two series' containers, not four, after re-plot"
