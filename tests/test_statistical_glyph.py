import matplotlib
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from cleopatra.statistical_glyph import StatisticalGlyph


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


def test_histogram_one_sample():
    # make data
    np.random.seed(1)
    x = 4 + np.random.normal(0, 1.5, 200)
    stat_plot = StatisticalGlyph(x)
    fig, ax, hist = stat_plot.histogram()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(hist, dict)
    assert ["n", "bins", "patches"] == list(hist.keys())


def test_histogram_multiple_sample():
    # make data
    np.random.seed(1)
    x = 4 + np.random.normal(0, 1.5, (200, 3))
    colors = ["red", "green", "blue"]
    stat_plot = StatisticalGlyph(x, color=colors)
    fig, ax, hist = stat_plot.histogram()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(hist, dict)
    assert ["n", "bins", "patches"] == list(hist.keys())


class TestBoxplot:
    """Tests for StatisticalGlyph.boxplot (T7.3c)."""

    def test_single_series_one_box(self):
        """1D values draw a single box.

        Test scenario:
            One box is produced for a 1D sample.
        """
        np.random.seed(1)
        stat = StatisticalGlyph(np.random.normal(0, 1, 100))
        fig, ax, bp = stat.boxplot()
        assert isinstance(fig, Figure) and isinstance(ax, Axes)
        assert len(bp["boxes"]) == 1, f"Expected 1 box, got {len(bp['boxes'])}"

    def test_multi_series_one_box_per_column(self):
        """2D values draw one box per column.

        Test scenario:
            A (50, 3) sample yields three boxes.
        """
        np.random.seed(1)
        stat = StatisticalGlyph(
            np.random.normal(0, 1, (50, 3)), color=["r", "g", "b"]
        )
        _, _, bp = stat.boxplot()
        assert len(bp["boxes"]) == 3, f"Expected 3 boxes, got {len(bp['boxes'])}"

    def test_composes_into_supplied_axes(self):
        """boxplot draws on a caller-supplied axes without plt.show().

        Test scenario:
            Passing ax reuses it and its figure.
        """
        np.random.seed(1)
        fig, ax = plt.subplots()
        stat = StatisticalGlyph(np.random.normal(0, 1, 50))
        out_fig, out_ax, _ = stat.boxplot(ax=ax)
        assert out_ax is ax and out_fig is fig, "Should reuse supplied axes/figure"

    def test_custom_labels(self):
        """Custom tick labels are applied.

        Test scenario:
            Labels passed through are set on the x ticks.
        """
        np.random.seed(1)
        stat = StatisticalGlyph(np.random.normal(0, 1, (30, 2)), color=["r", "g"])
        _, ax, _ = stat.boxplot(labels=["A", "B"])
        assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]

    def test_default_labels_are_one_based_indices(self):
        """With no labels, ticks are labelled 1..n (H1 fix path).

        Test scenario:
            Labels default to 1-based series indices, set via
            set_xticklabels (not boxplot's version-specific kwarg), so
            three columns yield tick labels ["1", "2", "3"] and ticks
            at [1, 2, 3].
        """
        np.random.seed(1)
        stat = StatisticalGlyph(
            np.random.normal(0, 1, (30, 3)), color=["r", "g", "b"]
        )
        _, ax, bp = stat.boxplot()
        assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2", "3"], (
            "Default labels should be 1-based indices"
        )
        assert list(ax.get_xticks()) == [1, 2, 3], (
            f"Ticks should sit at box positions 1..n, got {list(ax.get_xticks())}"
        )
        assert len(bp["boxes"]) == 3, "Three boxes expected for three columns"

    def test_single_series_default_label(self):
        """A 1D sample gets a single '1' tick label (H1 fix path).

        Test scenario:
            One series -> one box labelled "1".
        """
        np.random.seed(1)
        stat = StatisticalGlyph(np.random.normal(0, 1, 40))
        _, ax, _ = stat.boxplot()
        assert [t.get_text() for t in ax.get_xticklabels()] == ["1"], (
            "Single series should be labelled '1'"
        )


class TestMultiboxplot:
    """Tests for StatisticalGlyph.multiboxplot (T7.3c)."""

    def test_boxes_at_custom_positions(self):
        """Boxes are placed at the requested x positions.

        Test scenario:
            Median lines sit at the requested positions.
        """
        np.random.seed(1)
        stat = StatisticalGlyph(
            np.random.normal(0, 1, (40, 3)), color=["r", "g", "b"]
        )
        _, _, bp = stat.multiboxplot(positions=[1, 2, 4])
        medians = [round(float(line.get_xdata().mean())) for line in bp["medians"]]
        assert medians == [1, 2, 4], f"Medians should sit at positions, got {medians}"

    def test_requires_2d(self):
        """1D values are rejected by multiboxplot.

        Test scenario:
            multiboxplot needs one column per box.
        """
        stat = StatisticalGlyph(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="requires 2D"):
            stat.multiboxplot()

    def test_positions_length_mismatch_raises(self):
        """A positions length not matching columns raises ValueError.

        Test scenario:
            3 columns but 2 positions is rejected.
        """
        np.random.seed(1)
        stat = StatisticalGlyph(
            np.random.normal(0, 1, (20, 3)), color=["r", "g", "b"]
        )
        with pytest.raises(ValueError, match="positions length"):
            stat.multiboxplot(positions=[1, 2])

    def test_labels_length_mismatch_raises(self):
        """A labels length not matching columns raises ValueError.

        Test scenario:
            3 columns but 2 labels is rejected.
        """
        np.random.seed(1)
        stat = StatisticalGlyph(
            np.random.normal(0, 1, (20, 3)), color=["r", "g", "b"]
        )
        with pytest.raises(ValueError, match="labels length"):
            stat.multiboxplot(labels=["A", "B"])


class TestStripes:
    """Tests for StatisticalGlyph.stripes (T7.3c)."""

    def test_one_bar_per_value(self):
        """Each value becomes one full-height stripe.

        Test scenario:
            Six values -> six bars; y-axis ticks are removed.
        """
        series = np.array([0.1, 0.3, 0.2, 0.6, 0.9, 0.7])
        stat = StatisticalGlyph(series)
        fig, ax, bars = stat.stripes(cmap="coolwarm")
        assert isinstance(bars, BarContainer), f"Expected BarContainer, got {type(bars)}"
        assert len(bars) == 6, f"Expected 6 stripes, got {len(bars)}"
        assert list(ax.get_yticks()) == [], "Stripes should hide y ticks"

    def test_colours_span_cmap(self):
        """Stripe colours vary with value across the colormap.

        Test scenario:
            The min-value and max-value stripes differ in colour.
        """
        series = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        stat = StatisticalGlyph(series)
        _, _, bars = stat.stripes(cmap="coolwarm")
        assert bars.patches[0].get_facecolor() != bars.patches[-1].get_facecolor(), (
            "First and last stripe should differ in colour"
        )

    def test_explicit_limits_respected(self):
        """Explicit vmin/vmax drive the normalization without error.

        Test scenario:
            Passing vmin/vmax renders the stripes.
        """
        series = np.array([0.0, 5.0, 10.0])
        stat = StatisticalGlyph(series)
        _, _, bars = stat.stripes(cmap="viridis", vmin=-5.0, vmax=15.0)
        assert len(bars) == 3, "Should render all stripes with explicit limits"

    def test_requires_1d(self):
        """2D values are rejected by stripes.

        Test scenario:
            stripes needs a single 1D series.
        """
        stat = StatisticalGlyph(np.ones((4, 2)))
        with pytest.raises(ValueError, match="requires 1D"):
            stat.stripes()
