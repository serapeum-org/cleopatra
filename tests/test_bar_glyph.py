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

    def test_non_1d_values_raise(self):
        """A 2-D values array raises ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            BarGlyph(["a", "b"], [[1.0], [2.0]])

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
        with pytest.raises(ValueError, match="must match the number of categories"):
            BarGlyph(["a", "b"], [1.0, 2.0]).plot(colors=["#f00"])

    def test_unknown_palette_raises_valueerror(self):
        """An unknown palette name raises ValueError, not a bare KeyError."""
        with pytest.raises(ValueError, match="unknown palette"):
            BarGlyph(["a", "b"], [1.0, 2.0], palette="does_not_exist").plot()

    def test_horizontal_draws_barh(self):
        """horizontal=True draws bars whose widths (not heights) carry the values."""
        _, _, bars = BarGlyph(["a", "b"], [3.0, 7.0], horizontal=True).plot()
        assert [float(b.get_width()) for b in bars] == [3.0, 7.0]

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
        assert ax3.get_xlabel() == "CX", "override must not persist into a later plain plot"
