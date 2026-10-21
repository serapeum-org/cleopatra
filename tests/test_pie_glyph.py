"""Tests for cleopatra.glyphs.stats.pie_glyph.PieGlyph — issue #387.

Covers construction/validation (including negative and zero-sum values), the
drawn wedge artists and their angles, the category -> colour mapping, the
autopct percentage labels, the legend-vs-wedge-label contract, title styling,
equal aspect, and drawing onto a caller-supplied axes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

from cleopatra.glyphs.stats.pie_glyph import PieGlyph
from cleopatra.styling.palettes import get_palette


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


class TestPieGlyphConstruction:
    """Construction and input validation for PieGlyph."""

    def test_stores_labels_and_values(self):
        """Categories are coerced to str and values to a 1-D float array."""
        glyph = PieGlyph(["a", "b"], [1, 3])
        assert glyph.categories == ["a", "b"]
        assert glyph.values.tolist() == [1.0, 3.0]

    def test_length_mismatch_raises(self):
        """A categories/values length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            PieGlyph(["a", "b", "c"], [1.0, 2.0])

    def test_negative_value_raises(self):
        """A negative value has no pie representation and raises."""
        with pytest.raises(ValueError, match="non-negative"):
            PieGlyph(["a", "b"], [1.0, -1.0])

    def test_zero_sum_raises(self):
        """All-zero values (zero total) raise."""
        with pytest.raises(ValueError, match="sum to zero"):
            PieGlyph(["a", "b"], [0.0, 0.0])

    def test_non_finite_values_raise(self):
        """NaN or inf values are rejected at construction, not late in matplotlib."""
        with pytest.raises(ValueError, match="finite"):
            PieGlyph(["a", "b"], [1.0, np.nan])
        with pytest.raises(ValueError, match="finite"):
            PieGlyph(["a", "b"], [1.0, np.inf])


class TestPieGlyphPlot:
    """Rendering behaviour of PieGlyph.plot."""

    def test_returns_figure_axes_wedges(self):
        """plot returns (Figure, Axes, list[Wedge]) with one wedge per category."""
        fig, ax, wedges = PieGlyph(["a", "b", "c"], [1.0, 2.0, 3.0]).plot()
        assert isinstance(fig, Figure), "first return should be a Figure"
        assert len(wedges) == 3, "one wedge per category"
        assert all(isinstance(w, Wedge) for w in wedges), "wedges are Wedge patches"

    def test_equal_values_give_equal_angles(self):
        """Equal values split the circle into equal-angle wedges."""
        _, _, wedges = PieGlyph(["a", "b", "c", "d"], [1.0, 1.0, 1.0, 1.0]).plot()
        spans = [float(w.theta2 - w.theta1) for w in wedges]
        assert all(abs(s - 90.0) < 1e-6 for s in spans), "four equal wedges = 90° each"

    def test_wedge_span_proportional_to_value(self):
        """A value's wedge span is its share of the total, in degrees."""
        _, _, wedges = PieGlyph(["a", "b"], [1.0, 3.0]).plot()
        spans = [float(w.theta2 - w.theta1) for w in wedges]
        assert abs(spans[0] - 90.0) < 1e-6, "1/4 of the circle"
        assert abs(spans[1] - 270.0) < 1e-6, "3/4 of the circle"

    def test_single_category_fills_the_circle(self):
        """A single category is one full 360° wedge."""
        _, _, wedges = PieGlyph(["only"], [5.0]).plot()
        assert len(wedges) == 1, "one wedge for one category"
        assert abs(float(wedges[0].theta2 - wedges[0].theta1) - 360.0) < 1e-6

    def test_default_palette_colours_wedges(self):
        """Wedges take the palette swatches in category order."""
        _, _, wedges = PieGlyph(["a", "b", "c"], [1.0, 1.0, 1.0]).plot()
        swatches = list(get_palette("category12").colors)
        assert [to_hex(w.get_facecolor()) for w in wedges] == [
            to_hex(swatches[i]) for i in range(3)
        ]

    def test_explicit_colours_used_verbatim(self):
        """Caller-supplied colours colour the wedges in order."""
        _, _, wedges = PieGlyph(
            ["a", "b"], [1.0, 1.0], colors=["#1f77b4", "#ff7f0e"]
        ).plot()
        assert [to_hex(w.get_facecolor()) for w in wedges] == ["#1f77b4", "#ff7f0e"]

    def test_autopct_draws_percentage_labels_by_default(self):
        """By default each wedge gets an autopct percentage text."""
        _, ax, _ = PieGlyph(["a", "b"], [1.0, 3.0]).plot()
        texts = {t.get_text() for t in ax.texts}
        assert "25.0%" in texts and "75.0%" in texts, "default autopct shows percents"

    def test_autopct_none_hides_percentages(self):
        """autopct=None draws no percentage labels."""
        _, ax, _ = PieGlyph(["a", "b"], [1.0, 3.0]).plot(autopct=None)
        assert not any("%" in t.get_text() for t in ax.texts), "no percent labels"

    def test_labels_on_wedges_without_legend(self):
        """Without a legend the category names label the wedges."""
        _, ax, _ = PieGlyph(["water", "forest"], [1.0, 1.0]).plot()
        assert ax.get_legend() is None, "no legend by default"
        assert {"water", "forest"} <= {t.get_text() for t in ax.texts}

    def test_add_legend_replaces_wedge_labels(self):
        """add_legend=True draws a legend and omits inline wedge labels."""
        _, ax, _ = PieGlyph(["water", "forest"], [1.0, 1.0]).plot(add_legend=True)
        legend = ax.get_legend()
        assert legend is not None, "add_legend should draw a legend"
        assert [t.get_text() for t in legend.get_texts()] == ["water", "forest"]
        non_pct = {t.get_text() for t in ax.texts if "%" not in t.get_text()}
        assert "water" not in non_pct, (
            "wedges should not be labelled when a legend is drawn"
        )

    def test_equal_aspect_and_title(self):
        """The pie axes use equal aspect and honour a title."""
        _, ax, _ = PieGlyph(["a", "b"], [1.0, 1.0]).plot(title="shares")
        assert ax.get_aspect() == 1.0, "a pie must be circular (equal aspect)"
        assert ax.get_title() == "shares"

    def test_draws_on_supplied_axes(self):
        """A caller-supplied axes is drawn on and returned."""
        fig, ax = plt.subplots()
        out_fig, out_ax, _ = PieGlyph(["a", "b"], [1.0, 1.0]).plot(ax=ax)
        assert out_ax is ax and out_fig is fig

    def test_replot_on_same_axes_replaces_not_stacks(self):
        """Re-plotting the same glyph clears its prior wedges instead of stacking."""
        fig, ax = plt.subplots()
        glyph = PieGlyph(["a", "b"], [1.0, 1.0])
        glyph.plot(ax=ax)
        glyph.plot(ax=ax)
        assert len(ax.patches) == 2, "re-plot should replace, not stack, the wedges"
        # the label + percent texts must be cleared too, not only the wedges
        assert len(ax.texts) == 4, "re-plot should clear the prior labels/percents"

    def test_compose_overlays_other_glyph_on_shared_axes(self):
        """compose=True keeps the other glyph's wedges and labels on a shared axes."""
        fig, ax = plt.subplots()
        PieGlyph(["a", "b"], [1.0, 1.0]).plot(ax=ax)
        PieGlyph(["c", "d"], [1.0, 1.0]).plot(ax=ax, compose=True)
        assert len(ax.patches) == 4, "compose should overlay, keeping both glyphs' wedges"
        assert len(ax.texts) == 8, "compose keeps both glyphs' labels/percents"

    def test_uses_construction_axes_when_plot_ax_omitted(self):
        """An axes given at construction is used when plot() gets no ax."""
        fig, ax = plt.subplots()
        _, out_ax, _ = PieGlyph(["a", "b"], [1.0, 1.0], ax=ax).plot()
        assert out_ax is ax, "the construction-time axes should be reused"
