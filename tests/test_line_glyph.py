"""Tests for cleopatra.line_glyph.LineGlyph (T7.3c)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D

from cleopatra.line_glyph import LINE_DEFAULT_OPTIONS, LineGlyph


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


@pytest.fixture()
def xy():
    """A single 1D series.

    Returns:
        tuple[np.ndarray, np.ndarray]: x and y arrays.
    """
    return np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 4.0, 9.0])


class TestLineGlyphInit:
    """Tests for LineGlyph.__init__."""

    def test_stores_arrays(self, xy):
        """x and y are stored as numpy arrays.

        Test scenario:
            Lists are accepted and converted.
        """
        glyph = LineGlyph([0, 1, 2], [3, 4, 5])
        assert isinstance(glyph.x, np.ndarray) and isinstance(glyph.y, np.ndarray)

    def test_line_specific_options(self, xy):
        """Line-specific option keys are present.

        Test scenario:
            marker/linestyle/alpha defaults exist.
        """
        glyph = LineGlyph(*xy)
        assert glyph.default_options["linestyle"] == "-", "Default linestyle '-'"
        assert glyph.default_options["alpha"] == 1.0, "Default alpha 1.0"

    def test_mismatched_lengths_raise(self):
        """x length not matching y rows raises ValueError.

        Test scenario:
            len(x) != y.shape[0] is rejected.
        """
        with pytest.raises(ValueError, match="must match the number of rows"):
            LineGlyph([0, 1, 2], [0, 1])

    def test_invalid_kwarg_raises(self, xy):
        """An unknown kwarg is rejected by the strict merge."""
        with pytest.raises(ValueError, match="not correct"):
            LineGlyph(*xy, nope=1)


class TestLineGlyphLine:
    """Tests for LineGlyph.line."""

    def test_single_series_one_line(self, xy):
        """A 1D series yields a single Line2D with the right data.

        Test scenario:
            One line is drawn whose y-data matches the input.
        """
        glyph = LineGlyph(*xy)
        _, _, lines = glyph.line()
        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"
        assert isinstance(lines[0], Line2D), "Should be a Line2D"
        np.testing.assert_array_almost_equal(lines[0].get_ydata(), xy[1])

    def test_multi_series_multiple_lines(self):
        """2D y draws one line per column.

        Test scenario:
            A (3, 2) y array yields two lines.
        """
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([[0.0, 5.0], [1.0, 6.0], [2.0, 7.0]])
        glyph = LineGlyph(x, y)
        _, _, lines = glyph.line()
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

    def test_labels_applied(self, xy):
        """A label is attached to the drawn line.

        Test scenario:
            label='obs' surfaces on the line.
        """
        glyph = LineGlyph(*xy)
        _, _, lines = glyph.line(label="obs")
        assert lines[0].get_label() == "obs", f"Unexpected label: {lines[0].get_label()}"

    def test_title_override(self, xy):
        """A title passed to line is applied to the axes."""
        glyph = LineGlyph(*xy)
        _, ax, _ = glyph.line(title="Series")
        assert ax.get_title() == "Series", f"Unexpected title: {ax.get_title()}"

    def test_plot_on_supplied_axes(self, xy):
        """Plotting onto a supplied axes reuses that axes/figure."""
        fig, ax = plt.subplots()
        glyph = LineGlyph(*xy)
        out_fig, out_ax, _ = glyph.line(ax=ax)
        assert out_ax is ax and out_fig is fig, "Should reuse supplied axes/figure"


class TestLineGlyphBar:
    """Tests for LineGlyph.bar."""

    def test_one_bar_per_point(self, xy):
        """A bar chart draws one bar per x value.

        Test scenario:
            Four x values -> four bars.
        """
        glyph = LineGlyph(*xy)
        _, _, bars = glyph.bar()
        assert isinstance(bars, BarContainer), f"Expected BarContainer, got {type(bars)}"
        assert len(bars) == 4, f"Expected 4 bars, got {len(bars)}"

    def test_bar_rejects_2d(self):
        """A 2D y is rejected by bar.

        Test scenario:
            bar requires a single series.
        """
        x = np.array([0.0, 1.0])
        y = np.array([[0.0, 1.0], [1.0, 2.0]])
        glyph = LineGlyph(x, y)
        with pytest.raises(ValueError, match="bar requires 1D"):
            glyph.bar()


class TestLineGlyphFillBetween:
    """Tests for LineGlyph.fill_between."""

    def test_returns_polycollection(self, xy):
        """fill_between returns a PolyCollection band.

        Test scenario:
            A band between y and a scalar baseline is drawn.
        """
        glyph = LineGlyph(*xy)
        _, ax, band = glyph.fill_between(y2=0.0)
        assert isinstance(band, PolyCollection), f"Expected PolyCollection, got {type(band)}"
        assert band in ax.collections, "Band should be added to the axes"

    def test_array_lower_bound(self, xy):
        """An array lower bound is accepted for an envelope.

        Test scenario:
            y2 as an array matching x draws without error.
        """
        glyph = LineGlyph(*xy)
        lower = xy[1] - 1.0
        _, _, band = glyph.fill_between(y2=lower)
        assert isinstance(band, PolyCollection), "Should accept an array lower bound"

    def test_fill_between_rejects_2d(self):
        """A 2D y is rejected by fill_between.

        Test scenario:
            fill_between requires a single series.
        """
        x = np.array([0.0, 1.0])
        y = np.array([[0.0, 1.0], [1.0, 2.0]])
        glyph = LineGlyph(x, y)
        with pytest.raises(ValueError, match="fill_between requires 1D"):
            glyph.fill_between()


def test_line_default_options_extend_style_defaults():
    """LINE_DEFAULT_OPTIONS is a superset of the shared style defaults.

    Test scenario:
        The module-level options dict carries both base style keys and
        the line-specific additions.
    """
    assert "figsize" in LINE_DEFAULT_OPTIONS, "Should inherit base style keys"
    assert "linestyle" in LINE_DEFAULT_OPTIONS, "Should add line keys"
