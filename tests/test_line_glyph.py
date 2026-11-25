"""Tests for cleopatra.glyphs.primitives.line_glyph.LineGlyph (T7.3c)."""

from __future__ import annotations

import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D

from cleopatra.glyphs.primitives.line_glyph import LINE_DEFAULT_OPTIONS, LineGlyph
from cleopatra.styling.params import DataStyle


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

    def test_numeric_list_coerced_to_array(self):
        """Ordinary numeric x is coerced to a numpy array.

        Test scenario:
            A plain list of numbers is stored as an ``np.ndarray`` (the
            datetime-preserving path does not apply).
        """
        glyph = LineGlyph([0, 1, 2], [3, 4, 5])
        assert isinstance(glyph.x, np.ndarray), f"Expected ndarray, got {type(glyph.x)}"
        assert glyph.x.dtype.kind in "iuf", f"Expected numeric dtype, got {glyph.x.dtype}"

    def test_datetime64_array_x_preserved(self):
        """A ``datetime64`` ndarray is left untouched by ``_coerce_x``.

        Test scenario:
            ``_coerce_x`` returns the same datetime64-backed object (so a later
            ``bar`` keeps its day-unit width), and its ``shape`` drives the
            length check.
        """
        x = np.array(
            ["1999-01-01", "1999-02-01", "1999-03-01"], dtype="datetime64[D]"
        )
        glyph = LineGlyph(x, np.array([1.0, 2.0, 3.0]))
        assert glyph.x is x, "datetime64 x should be preserved, not re-coerced"
        assert glyph.x.dtype.kind == "M", f"Expected datetime64, got {glyph.x.dtype}"

    def test_datetime_list_without_shape_length_checked(self):
        """A datetime64-backed list (no ``.shape``) still validates its length.

        Test scenario:
            A list whose ``np.asarray`` is ``datetime64`` is preserved as the
            list itself (no ``.shape``); the constructor falls back to ``len``
            and raises on a length mismatch.
        """
        x = [np.datetime64("1999-01-01"), np.datetime64("1999-02-01")]
        glyph = LineGlyph(x, np.array([1.0, 2.0]))
        assert glyph.x is x, "datetime64-backed list should be preserved as-is"
        with pytest.raises(ValueError, match="x length \\(2\\) must match"):
            LineGlyph(x, np.array([1.0, 2.0, 3.0]))


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
        assert lines[0].get_label() == "obs", (
            f"Unexpected label: {lines[0].get_label()}"
        )

    def test_flat_color_applied(self, xy):
        """A plain colour string sets the line colour directly.

        Test scenario:
            Unlike ScatterGlyph/PolygonGlyph, line() has no grouped
            `color=ColorScaling(...)` parameter to conflict with, so a flat
            colour string is accepted as-is (#405).
        """
        glyph = LineGlyph(*xy)
        _, _, lines = glyph.line(color="navy")
        assert lines[0].get_color() == "navy", (
            f"Unexpected colour: {lines[0].get_color()}"
        )

    def test_flat_color_defaults_to_color_1_option(self, xy):
        """With no explicit colour, line() falls back to the color_1 option.

        Test scenario:
            A construction-time color_1 override reaches the drawn line
            without an explicit color= at call time.
        """
        glyph = LineGlyph(*xy, color_1="#123456")
        _, _, lines = glyph.line()
        assert lines[0].get_color() == "#123456", (
            f"Unexpected colour: {lines[0].get_color()}"
        )

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
        assert isinstance(bars, BarContainer), (
            f"Expected BarContainer, got {type(bars)}"
        )
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

    def test_bar_datetime_width_in_data_units_stdlib(self):
        """A numeric width on a datetime axis is read in data units (days).

        Test scenario:
            x is a pandas-free list of ``datetime.datetime`` on a month cadence;
            ``width=20`` must draw bars 20 days wide, matching native
            ``Axes.bar`` -- not collapse to ~0 as a ``datetime64`` array would.
        """
        x = [
            datetime.datetime(1999, 1, 1) + datetime.timedelta(days=30 * i)
            for i in range(12)
        ]
        glyph = LineGlyph(x, np.arange(12.0))
        _, _, bars = glyph.bar(width=20)
        assert round(bars[0].get_width(), 1) == 20.0, (
            f"Expected 20-day bars, got {bars[0].get_width()}"
        )

    def test_bar_datetime_width_in_data_units_pandas(self):
        """A pandas ``DatetimeIndex`` keeps its day-unit width scaling.

        Test scenario:
            The issue's reproduction -- a monthly ``DatetimeIndex`` with
            ``width=20`` -- draws 20-day-wide bars rather than 20 nanoseconds.
        """
        pd = pytest.importorskip("pandas")
        idx = pd.date_range("1999-01-01", periods=12, freq="MS")
        glyph = LineGlyph(idx, np.arange(12.0))
        _, _, bars = glyph.bar(width=20)
        assert round(bars[0].get_width(), 1) == 20.0, (
            f"Expected 20-day bars, got {bars[0].get_width()}"
        )

    def test_bar_accepts_timedelta_width(self):
        """A ``timedelta`` width is accepted on a datetime axis.

        Test scenario:
            ``width=timedelta(days=20)`` draws 20-day bars regardless of how the
            datetime x is represented.
        """
        x = [
            datetime.datetime(1999, 1, 1) + datetime.timedelta(days=30 * i)
            for i in range(12)
        ]
        glyph = LineGlyph(x, np.arange(12.0))
        _, _, bars = glyph.bar(width=datetime.timedelta(days=20))
        assert round(bars[0].get_width(), 1) == 20.0, (
            f"Expected 20-day bars, got {bars[0].get_width()}"
        )

    def test_bar_alpha_keyword(self, xy):
        """An explicit ``alpha`` on ``.bar()`` is honoured without a TypeError.

        Test scenario:
            ``bar(alpha=0.5)`` previously collided with the option-sourced alpha
            and raised ``TypeError``; it now sets the bar opacity.
        """
        glyph = LineGlyph(*xy)
        _, _, bars = glyph.bar(alpha=0.5)
        assert bars[0].get_alpha() == 0.5, (
            f"Expected alpha 0.5, got {bars[0].get_alpha()}"
        )


class TestLineGlyphFillBetween:
    """Tests for LineGlyph.fill_between."""

    def test_returns_polycollection(self, xy):
        """fill_between returns a PolyCollection band.

        Test scenario:
            A band between y and a scalar baseline is drawn.
        """
        glyph = LineGlyph(*xy)
        _, ax, band = glyph.fill_between(y2=0.0)
        assert isinstance(band, PolyCollection), (
            f"Expected PolyCollection, got {type(band)}"
        )
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


class TestAxesReuseAcrossCalls:
    """`LineGlyph` keeps drawing on the axes it created the first time."""

    def test_second_call_without_ax_reuses_the_first_axes(self, xy):
        """A second `line()` overlays the first instead of opening a figure.

        Test scenario:
            With no `ax` given and an axes already stored from an earlier
            call, the glyph must reuse it -- otherwise each call would leak
            a new figure and the series would not compose.
        """
        x, y = xy
        glyph = LineGlyph(x, y)

        glyph.line()
        first_ax, first_fig = glyph.ax, glyph.fig
        glyph.line()

        assert glyph.ax is first_ax, "the stored axes should be reused"
        assert glyph.fig is first_fig, "no new figure should be created"


class TestConstructionTimeAlphaCollision:
    """`alpha` is the one option settable both loosely and through the group.

    Every other `DataStyle` field is rejected as a loose keyword, but `alpha`
    stays legitimate on `LineGlyph` / `HistogramGlyph`, so those are the only
    classes where a real collision can happen. The group is merged second and
    wins; this pins that on the actual class rather than on a synthetic
    option dict.
    """

    def test_group_wins_on_line_glyph(self, xy):
        """A grouped `alpha` overrides a loose one on a real `LineGlyph`.

        Test scenario:
            The group is merged after the loose kwargs, so it wins. Pinned on
            `LineGlyph` because the base `Glyph` option dict has no `alpha`
            key at all -- the collision cannot occur there.
        """
        x, y = xy

        glyph = LineGlyph(x, y, alpha=0.9, data_style=DataStyle(alpha=0.3))

        assert glyph.default_options["alpha"] == 0.3, (
            f"the group should win; got {glyph.default_options['alpha']}"
        )

    def test_loose_alpha_alone_is_still_honoured(self, xy):
        """Without a group, the loose keyword is untouched.

        Test scenario:
            The control for the collision case: `alpha` stays a legitimate
            loose option on this glyph, so adding the group parameter must
            not have changed the ordinary path.
        """
        x, y = xy

        glyph = LineGlyph(x, y, alpha=0.9)

        assert glyph.default_options["alpha"] == 0.9

    def test_unmodelled_group_key_is_not_recorded_as_explicit(self, xy):
        """`alpha_range` is dropped, so it must not appear in `_explicit_options`.

        Test scenario:
            `LineGlyph` has no `alpha_range` option, so the merge refuses to
            write it. Recording it as explicitly passed anyway would make the
            two mechanisms disagree about what the glyph is carrying.
        """
        x, y = xy

        glyph = LineGlyph(x, y, data_style=DataStyle(alpha=0.3))

        assert "alpha" in glyph._explicit_options
        assert "alpha_range" not in glyph._explicit_options, (
            f"a dropped key must not be recorded; got {sorted(glyph._explicit_options)}"
        )
